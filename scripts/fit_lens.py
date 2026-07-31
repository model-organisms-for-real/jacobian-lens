#!/usr/bin/env python
# Inspired by https://github.com/hijohnnylin/neuronpedia/blob/main/utils/neuronpedia-utils/neuronpedia_utils/jlens/fit_lens.py
#
# Reconstruction of the ``fit_lens.py`` that Neuronpedia used to produce the
# lenses published at https://huggingface.co/neuronpedia/jacobian-lens (see the
# ``config.yaml`` next to each lens for the exact command line). Neuronpedia's
# copy calls a patched ``jlens.fit(..., metrics_callback=...)``; upstream
# ``jlens`` has no such hook, so the per-prompt loop is spelled out here on top
# of the public :func:`jlens.jacobian_for_prompt` and everything else — model
# wrapping, the Jacobian estimator, checkpoint format, lens IO — is upstream.
r"""Fit a Jacobian lens for *any* HuggingFace model on *any* text dataset.

The model id and the fitting corpus are both command-line arguments. The fit
reports two per-prompt diagnostics, written to a CSV convergence curve:

``mean_rel_change`` (Δmean)
    Relative Frobenius change of the running-mean Jacobian contributed by the
    latest prompt, averaged over fitted layers (see :func:`relative_change`).
    It decays roughly like ``1/n``; where it flattens is where extra prompts
    stop improving the lens. ``--stop_at_delta`` stops the fit once it
    converges.

``identity_distance``
    ``||J_l - I||_F / sqrt(d_model)`` for the deepest fitted layer (the one
    transport-adjacent to the target layer, where ``J`` should be closest to
    the identity). A stability check on the running mean, not a stopping rule.

Run::

    uv run python scripts/fit_lens.py Qwen/Qwen3.5-0.8B --out_dir out/
    uv run python scripts/fit_lens.py meta-llama/Llama-3.1-8B --n_prompts 1000 \
        --stop_at_delta 1e-3

    # A different corpus (any HF dataset with a text column):
    uv run python scripts/fit_lens.py Qwen/Qwen3.5-0.8B \
        --dataset stas/openwebtext-10k --dataset_config none --text_field text

Fitting is checkpointed after every prompt, so it is safe to interrupt and
resume; the checkpoint uses upstream's format, so :func:`jlens.fit` can resume
one of these runs and vice versa.

Both metrics were reverse-engineered from Neuronpedia's published convergence
curves and reproduce them (see :func:`relative_change` and the numbers in
``scripts/README.md``). One deliberate deviation from their script:
``--hf_cache_dir`` is *not* deleted after the fit unless ``--delete_hf_cache``
is passed, where theirs deleted it by default.
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import re
import shutil
import sys
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import transformers
from tqdm.auto import tqdm

import jlens
from jlens.fitting import _atomic_save  # in-repo helper: temp file + os.replace

logger = logging.getLogger("jlens.fit_lens")


class _TqdmLogStream:
    """stderr shim that emits through :func:`tqdm.write`, so log records (and
    upstream ``jlens`` warnings) scroll above an active bar instead of tearing
    through it."""

    @staticmethod
    def write(message: str) -> None:
        tqdm.write(message.rstrip("\n"), file=sys.stderr)

    @staticmethod
    def flush() -> None:
        sys.stderr.flush()


def route_logging_through_tqdm() -> None:
    """Point the ``jlens`` stderr handler at :class:`_TqdmLogStream`."""
    for handler in logging.getLogger("jlens").handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setStream(_TqdmLogStream())


def load_prompts(
    *,
    dataset: str,
    config: str | None,
    split: str,
    text_field: str,
    n_prompts: int,
    max_chars: int = 2000,
    min_chars: int = 200,
    trust_remote_code: bool = False,
) -> list[str]:
    """Stream ``n_prompts`` text chunks of ~``max_chars`` from a HF dataset.

    Records are concatenated and re-chunked to ``max_chars`` so that short rows
    (e.g. WikiText lines) and long documents (e.g. web text) both yield usable
    prompts. Blank rows and obvious section headers are skipped. The dataset is
    streamed from the HuggingFace Hub at call time; nothing is bundled here.

    Args:
        dataset: HF dataset id (e.g. ``"Salesforce/wikitext"``).
        config: Dataset config/subset name, or ``None`` for datasets without one.
        split: Split to read (e.g. ``"train"``).
        text_field: Name of the text column on each record.
        n_prompts: Number of prompts to return.
        max_chars: Target chunk length (also the hard truncation).
        min_chars: Drop any final/partial chunk shorter than this.
        trust_remote_code: Forwarded to ``datasets.load_dataset``.

    Returns:
        A list of up to ``n_prompts`` text prompts.
    """
    from datasets import load_dataset

    stream = load_dataset(
        dataset,
        config,
        split=split,
        streaming=True,
        trust_remote_code=trust_remote_code,
    )

    prompts: list[str] = []
    buffer = ""
    for record in stream:
        text = str(record.get(text_field, "")).strip()
        if not text or text.startswith("="):
            continue
        buffer += " " + text
        while len(buffer) > max_chars:
            prompts.append(buffer[:max_chars].strip())
            buffer = buffer[max_chars:]
            if len(prompts) >= n_prompts:
                return prompts
    tail = buffer.strip()
    if tail and len(tail) >= min_chars and len(prompts) < n_prompts:
        prompts.append(tail)
    return prompts


@dataclass(frozen=True)
class FitProgress:
    """One row of the convergence curve: the state after fitting one prompt."""

    n_done: int
    prompt_idx: int
    seq_len: int
    n_valid_positions: int
    elapsed_s: float
    identity_distance: float
    mean_rel_change: float


class ConvergenceTracker:
    """Writes the per-prompt convergence metrics to CSV, records milestones, and
    optionally requests early stop once the lens has converged.

    :meth:`record` returns ``True`` to ask the fit loop to stop early. Early
    stop fires only when *all* of these hold:

    * ``stop_at_delta`` is set, and
    * at least ``min_prompts`` prompts have been accumulated, and
    * the mean of the last ``window`` ``Δmean`` values is below
      ``stop_at_delta`` (smoothing avoids tripping on a single noisy step).

    Args:
        csv_path: Convergence curve destination. Appended to (without a
            repeated header) when it already exists, so a resumed fit extends
            the curve instead of truncating it.
        thresholds: Δmean levels to report first-crossing prompt counts for.
        stop_at_delta: Smoothed Δmean below which to stop; ``None`` disables
            early stopping.
        min_prompts: Never stop early before this many prompts.
        window: Number of recent prompts averaged for the stopping test.
    """

    def __init__(
        self,
        csv_path: str,
        thresholds: Sequence[float],
        *,
        stop_at_delta: float | None = None,
        min_prompts: int = 100,
        window: int = 10,
    ) -> None:
        self.csv_path = csv_path
        self.thresholds = tuple(thresholds)
        self.stop_at_delta = stop_at_delta
        self.min_prompts = min_prompts
        self.window = max(1, window)
        self.history: list[tuple[int, float]] = []
        self.stopped_at: int | None = None
        self._crossed: dict[float, int] = {}
        self._recent: deque[float] = deque(maxlen=self.window)
        is_new = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
        self._file = open(csv_path, "a", newline="")  # noqa: SIM115 (see close())
        self._writer = csv.writer(self._file)
        if is_new:
            self._writer.writerow(
                [
                    "n_done",
                    "prompt_idx",
                    "seq_len",
                    "n_valid_positions",
                    "elapsed_s",
                    "identity_distance",
                    "mean_rel_change",
                ]
            )

    def record(self, p: FitProgress) -> bool:
        """Log one prompt; returns ``True`` if the fit should stop early."""
        self._writer.writerow(
            [
                p.n_done,
                p.prompt_idx,
                p.seq_len,
                p.n_valid_positions,
                f"{p.elapsed_s:.3f}",
                f"{p.identity_distance:.6f}",
                f"{p.mean_rel_change:.8f}",
            ]
        )
        self._file.flush()
        if math.isnan(p.mean_rel_change):  # first prompt: no running mean yet
            return False
        self.history.append((p.n_done, p.mean_rel_change))
        self._recent.append(p.mean_rel_change)
        for threshold in self.thresholds:
            if threshold not in self._crossed and p.mean_rel_change < threshold:
                self._crossed[threshold] = p.n_done

        if (
            self.stop_at_delta is not None
            and p.n_done >= self.min_prompts
            and len(self._recent) == self.window
            and (sum(self._recent) / self.window) < self.stop_at_delta
        ):
            smoothed = sum(self._recent) / self.window
            self.stopped_at = p.n_done
            tqdm.write(
                f"Converged: {self.window}-prompt mean Δmean={smoothed:.2e} < "
                f"{self.stop_at_delta:g} at {p.n_done} prompts — stopping early."
            )
            return True
        return False

    def close(self) -> None:
        self._file.close()

    def summary(self) -> str:
        lines = ["Convergence (Δmean = relative change of the running-mean Jacobian):"]
        for threshold in self.thresholds:
            n = self._crossed.get(threshold)
            if n is None:
                lines.append(f"  Δmean < {threshold:g}: not reached within this run")
            else:
                lines.append(f"  Δmean < {threshold:g}: first reached at {n} prompts")
        if self.history:
            last_n, last_value = self.history[-1]
            lines.append(f"  last: Δmean={last_value:.2e} at {last_n} prompts")
            if self.stopped_at is not None:
                lines.append(
                    f"  stopped early at {self.stopped_at} prompts (--stop_at_delta)"
                )
            lines.append(f"  full curve written to {self.csv_path}")
        return "\n".join(lines)


def identity_distance(jacobian: torch.Tensor) -> float:
    """``||J - I||_F / sqrt(d)`` for a square ``J``.

    Subtracts the identity in place on a single copy: no second ``d x d``
    temporary for ``I``, and no cancellation error from expanding the norm.
    """
    difference = jacobian.clone()
    difference.diagonal().sub_(1.0)
    return difference.norm().item() / math.sqrt(jacobian.shape[0])


def relative_change(
    per_prompt: dict[int, torch.Tensor],
    jacobian_sum: dict[int, torch.Tensor],
    n_done: int,
) -> float:
    """Δmean: the relative shift the new prompt makes to the running mean.

    Per layer, ``||M_new - M_old||_F / ||M_new||_F``, where ``M_old`` is the
    mean over the ``n_done`` prompts fitted so far and ``M_new`` folds in the
    prompt just fitted; the layers are then averaged. Verified against
    Neuronpedia's published convergence curves (their first three gemma-3-1b
    rows, 0.49799543 / 0.25377339 / 0.18668778, come back as 0.49802589 /
    0.25371778 / 0.18664706 on a rerun of their command).

    Note this is *not* the ``max_d_mean`` that :func:`jlens.fit` logs: that one
    maxes over layers instead of averaging and divides by ``||M_old||``, so it
    reads roughly 1.4x larger. A ``--stop_at_delta`` threshold is only
    comparable to numbers produced by the same definition.

    Layers are handled one at a time, so this adds a few ``d x d`` temporaries
    rather than a copy of the whole accumulator.
    """
    per_layer = []
    for layer, jacobian in per_prompt.items():
        mean_old = jacobian_sum[layer] / n_done
        mean_new = (jacobian_sum[layer] + jacobian) / (n_done + 1)
        per_layer.append(((mean_new - mean_old).norm() / mean_new.norm()).item())
    return sum(per_layer) / len(per_layer)


def fit_with_metrics(
    model: jlens.LensModel,
    prompts: Sequence[str],
    tracker: ConvergenceTracker,
    *,
    source_layers: Sequence[int],
    target_layer: int,
    dim_batch: int = 8,
    max_seq_len: int = 128,
    skip_first: int = jlens.fitting.SKIP_FIRST_N_POSITIONS,
    checkpoint_path: str | None = None,
    checkpoint_every: int = 1,
    progress: bool = True,
) -> jlens.JacobianLens:
    """:func:`jlens.fit` with per-prompt metrics and early stopping.

    Same accumulation, checkpoint format and skip-on-short-prompt behaviour as
    upstream :func:`jlens.fit`; the per-prompt Jacobians themselves come from
    :func:`jlens.jacobian_for_prompt`. The only additions are the ``tracker``
    callback (which may end the fit early) and the two metrics fed to it.

    Args:
        model: The model to fit on.
        prompts: Text prompts to average over.
        tracker: Receives one :class:`FitProgress` per fitted prompt; stops the
            fit when it returns ``True``.
        source_layers: Layers to fit ``J_l`` at (already resolved, no negatives).
        target_layer: Layer to take gradients with respect to.
        dim_batch: Output dims per backward pass; see :func:`jlens.jacobian_for_prompt`.
        max_seq_len: Truncate each prompt to this many tokens.
        skip_first: Leading positions excluded from the Jacobian average.
        checkpoint_path: If set, resume from and write a checkpoint here.
        checkpoint_every: Write the checkpoint every N prompts.
        progress: Show a tqdm bar (with the live metrics in its postfix) and
            demote the per-prompt log line to DEBUG. With ``False``, every
            prompt logs a line at INFO instead, as upstream :func:`jlens.fit`
            does.

    Returns:
        The fitted :class:`jlens.JacobianLens`.
    """
    d_model = model.d_model
    deepest_layer = max(source_layers)

    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        for key, expected in (
            ("source_layers", list(source_layers)),
            ("target_layer", target_layer),
            ("skip_first", skip_first),
        ):
            if key in state and state[key] != expected:
                raise SystemExit(
                    f"checkpoint at {checkpoint_path} was fitted with {key}="
                    f"{state[key]!r}, not {expected!r}; delete it to start over"
                )
        jacobian_sum = state["jacobian_sum"]
        n_done = state["n_done"]
        next_idx = state["next_idx"]
        logger.info(
            "resuming from checkpoint: %d/%d prompts processed (%d fitted)",
            next_idx,
            len(prompts),
            n_done,
        )
    else:
        jacobian_sum = {
            layer: torch.zeros(d_model, d_model, dtype=torch.float32)
            for layer in source_layers
        }
        n_done = 0
        next_idx = 0

    def write_checkpoint() -> None:
        if checkpoint_path is not None:
            _atomic_save(
                {
                    "jacobian_sum": jacobian_sum,
                    "n_done": n_done,
                    "next_idx": next_idx,
                    "source_layers": list(source_layers),
                    "target_layer": target_layer,
                    "skip_first": skip_first,
                },
                checkpoint_path,
            )

    logger.info(
        "fitting %d source layers (target=L%d) on %d prompts",
        len(source_layers),
        target_layer,
        len(prompts),
    )
    bar = tqdm(
        total=len(prompts),
        initial=next_idx,
        disable=not progress,
        unit="prompt",
        desc="fitting",
        dynamic_ncols=True,
    )
    for prompt_idx, prompt in enumerate(prompts):
        if prompt_idx < next_idx:
            continue
        start_time = time.perf_counter()
        try:
            per_prompt_J, seq_len, n_valid = jlens.jacobian_for_prompt(
                model,
                prompt,
                source_layers,
                target_layer=target_layer,
                dim_batch=dim_batch,
                max_seq_len=max_seq_len,
                skip_first=skip_first,
            )
        except ValueError as exc:
            logger.warning("  skipping prompt %d: %s", prompt_idx, exc)
            next_idx = prompt_idx + 1
            bar.update(next_idx - bar.n)
            continue

        if n_done > 0:
            mean_rel_change = relative_change(per_prompt_J, jacobian_sum, n_done)
        else:
            mean_rel_change = float("nan")

        for layer in source_layers:
            jacobian_sum[layer] += per_prompt_J[layer]
        del per_prompt_J
        n_done += 1
        next_idx = prompt_idx + 1

        row = FitProgress(
            n_done=n_done,
            prompt_idx=prompt_idx,
            seq_len=seq_len,
            n_valid_positions=n_valid,
            elapsed_s=time.perf_counter() - start_time,
            identity_distance=identity_distance(jacobian_sum[deepest_layer] / n_done),
            mean_rel_change=mean_rel_change,
        )
        # With the bar up its postfix carries these numbers live, so the
        # per-prompt line would just be noise; without it, it is the only output.
        logger.log(
            logging.DEBUG if progress else logging.INFO,
            "  prompt %d/%d  n_done=%d seq_len=%d n_valid=%d  %.1fs  "
            "||J_%d-I||/sqrt(d)=%.3f  d_mean=%.2e",
            prompt_idx + 1,
            len(prompts),
            n_done,
            seq_len,
            n_valid,
            row.elapsed_s,
            deepest_layer,
            row.identity_distance,
            mean_rel_change,
        )
        # set_postfix_str, not set_postfix: the latter sorts the keys, which
        # would put Δmean (the number worth watching) last.
        bar.set_postfix_str(
            f"fitted={n_done} dmean={mean_rel_change:.2e} "
            f"ident={row.identity_distance:.3f}",
            refresh=False,
        )
        bar.update(next_idx - bar.n)
        if next_idx % checkpoint_every == 0:
            write_checkpoint()
        if tracker.record(row):
            break

    bar.close()
    write_checkpoint()
    if n_done == 0:
        raise SystemExit("no prompts were long enough to fit on")
    return jlens.JacobianLens(
        jacobians={layer: jacobian_sum[layer] / n_done for layer in source_layers},
        n_prompts=n_done,
        d_model=d_model,
    )


def resolve_layers(n_layers: int, target_layer: int | None) -> tuple[list[int], int]:
    """Resolve ``--target_layer`` and the default source layers below it."""
    target = n_layers - 1 if target_layer is None else target_layer
    if target < 0:
        target += n_layers
    if not 0 < target < n_layers:
        raise SystemExit(
            f"--target_layer {target_layer} out of range for {n_layers} layers"
        )
    return list(range(target)), target


def _slug(model: str) -> str:
    """Filesystem-safe stem derived from a model id or path."""
    base = model.rstrip("/").split("/")[-1]
    return re.sub(r"[^0-9A-Za-z._-]+", "-", base).strip("-") or "model"


def peak_vram_gb() -> float:
    """Peak allocated CUDA memory summed across all visible devices, in GiB."""
    if not torch.cuda.is_available():
        return 0.0
    return (
        sum(torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count()))
        / 1024**3
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("model", help="HF model id or local path (any decoder LM)")
    parser.add_argument("--out_dir", default="out", help="output directory for the lens")
    parser.add_argument(
        "--n_prompts", type=int, default=200, help="prompts to average over"
    )
    parser.add_argument(
        "--dim_batch", type=int, default=8, help="output dims per backward pass"
    )
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument(
        "--target_layer",
        type=int,
        default=None,
        help="layer to take gradients w.r.t. (default: final; negative indexes from end)",
    )
    parser.add_argument(
        "--text_module",
        default=None,
        help="dotted path to the text decoder (auto-detected)",
    )
    parser.add_argument(
        "--no_compile", action="store_true", help="disable per-layer torch.compile"
    )
    parser.add_argument(
        "--device_map",
        default="cuda",
        help=(
            "how to place the model: 'cuda' = single GPU (.cuda()); "
            "'auto' (or any accelerate device_map) = shard across all visible GPUs "
            "for models too large for one card"
        ),
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=("bfloat16", "float16", "float32"),
        help="model dtype",
    )
    parser.add_argument(
        "--hf_cache_dir",
        default=None,
        help=(
            "HuggingFace cache dir for weights. When set, ALL HF downloads are "
            "confined here (see --delete_hf_cache)."
        ),
    )
    parser.add_argument(
        "--delete_hf_cache",
        action="store_true",
        help=(
            "delete --hf_cache_dir after fitting, so disk isn't filled up across "
            "many models. Only ever pass this with a throwaway cache directory."
        ),
    )
    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=1,
        help="write the resumable checkpoint every N prompts",
    )
    parser.add_argument(
        "--no_progress",
        action="store_true",
        help=(
            "disable the progress bar and log one line per prompt instead "
            "(the default when stderr is not a terminal, e.g. piped to a file)"
        ),
    )

    # Dataset selection. Defaults reproduce the WikiText-103 corpus.
    parser.add_argument(
        "--dataset", default="Salesforce/wikitext", help="HF dataset id"
    )
    parser.add_argument(
        "--dataset_config",
        default="wikitext-103-raw-v1",
        help="dataset config/subset; pass 'none' for datasets without one",
    )
    parser.add_argument("--dataset_split", default="train", help="dataset split")
    parser.add_argument(
        "--text_field", default="text", help="text column name on each record"
    )
    parser.add_argument(
        "--max_chars", type=int, default=2000, help="target prompt length (chars)"
    )
    parser.add_argument(
        "--trust_remote_code", action="store_true", help="pass through to HF loaders"
    )

    # Convergence reporting.
    parser.add_argument(
        "--metrics_csv",
        default=None,
        help="per-prompt convergence curve (default: <out_dir>/<model>_convergence.csv)",
    )
    parser.add_argument(
        "--levels",
        default="1e-2,5e-3,1e-3",
        help="comma-separated Δmean thresholds to report 'levelled-off' prompt counts for",
    )
    parser.add_argument(
        "--stop_at_delta",
        type=float,
        default=None,
        help="stop early once the smoothed Δmean drops below this (e.g. 1e-3); off by default",
    )
    parser.add_argument(
        "--min_prompts",
        type=int,
        default=100,
        help="never stop early before this many prompts (only with --stop_at_delta)",
    )
    parser.add_argument(
        "--stop_window",
        type=int,
        default=10,
        help="number of recent prompts averaged for the --stop_at_delta test",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required for fitting.")

    jlens.configure_logging()
    show_progress = not args.no_progress and sys.stderr.isatty()
    if show_progress:
        route_logging_through_tqdm()
    os.makedirs(args.out_dir, exist_ok=True)
    slug = _slug(args.model)
    lens_path = os.path.join(args.out_dir, f"{slug}_jacobian_lens.pt")
    checkpoint_path = os.path.join(args.out_dir, f"{slug}_checkpoint.pt")
    metrics_csv = args.metrics_csv or os.path.join(
        args.out_dir, f"{slug}_convergence.csv"
    )
    config = (
        None
        if args.dataset_config.lower() in ("none", "", "null")
        else args.dataset_config
    )
    thresholds = sorted(
        (float(x) for x in args.levels.split(",") if x.strip()), reverse=True
    )
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]

    # Confine every HF download (weights + hub metadata) to the cache dir so it
    # can be wiped wholesale afterwards. Without this, downloads also leak into
    # the default ~/.cache/huggingface and survive cleanup.
    cache_root: str | None = None
    if args.hf_cache_dir:
        cache_root = os.path.abspath(os.path.expanduser(args.hf_cache_dir))
        os.makedirs(cache_root, exist_ok=True)
        os.environ["HF_HOME"] = cache_root
        os.environ["HF_HUB_CACHE"] = os.path.join(cache_root, "hub")
        os.environ["HF_XET_CACHE"] = os.path.join(cache_root, "xet")
        os.environ["HF_DATASETS_CACHE"] = os.path.join(cache_root, "datasets")
    elif args.delete_hf_cache:
        raise SystemExit("--delete_hf_cache requires --hf_cache_dir")

    hub_cache = os.path.join(cache_root, "hub") if cache_root else None
    load_kwargs: dict = {
        "dtype": dtype,
        "trust_remote_code": args.trust_remote_code,
    }
    if hub_cache:
        load_kwargs["cache_dir"] = hub_cache
    single_gpu = args.device_map.lower() == "cuda"
    if not single_gpu:
        load_kwargs["device_map"] = args.device_map

    try:
        print(f"Loading {args.model} ({args.dtype}, device_map={args.device_map}) ...")
        hf = transformers.AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
        if single_gpu:
            hf = hf.cuda()
        tok = transformers.AutoTokenizer.from_pretrained(
            args.model, cache_dir=hub_cache, trust_remote_code=args.trust_remote_code
        )
        model = jlens.from_hf(
            hf, tok, text_module=args.text_module, compile=not args.no_compile
        )
        print(f"Wrapped: {model!r}")
        source_layers, target_layer = resolve_layers(model.n_layers, args.target_layer)

        print(
            f"Loading {args.n_prompts} prompts from {args.dataset}"
            + (f" ({config})" if config else "")
            + f" [{args.dataset_split}::{args.text_field}] ..."
        )
        prompts = load_prompts(
            dataset=args.dataset,
            config=config,
            split=args.dataset_split,
            text_field=args.text_field,
            n_prompts=args.n_prompts,
            max_chars=args.max_chars,
            trust_remote_code=args.trust_remote_code,
        )
        if not prompts:
            raise SystemExit(
                "no prompts loaded — check --dataset/--dataset_config/--text_field"
            )

        tracker = ConvergenceTracker(
            metrics_csv,
            thresholds,
            stop_at_delta=args.stop_at_delta,
            min_prompts=args.min_prompts,
            window=args.stop_window,
        )
        print(
            f"Fitting lens over {len(prompts)} prompts "
            "(first call compiles, ~1-2 min) ..."
        )
        try:
            lens = fit_with_metrics(
                model,
                prompts,
                tracker,
                source_layers=source_layers,
                target_layer=target_layer,
                dim_batch=args.dim_batch,
                max_seq_len=args.max_seq_len,
                checkpoint_path=checkpoint_path,
                checkpoint_every=args.checkpoint_every,
                progress=show_progress,
            )
        finally:
            tracker.close()
        lens.save(lens_path)

        # The checkpoint only exists to resume an interrupted fit; once the lens
        # is saved it is dead weight, so drop it.
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)

        print(f"Peak CUDA memory during fit (all GPUs): {peak_vram_gb():.2f} GB")
        print(f"Done. Saved lens -> {lens_path}\n{lens!r}")
        print(tracker.summary())
    finally:
        # Free disk: drop the downloaded weights so they don't accumulate when
        # fitting many models in sequence. The lens (in out_dir) is unaffected.
        if cache_root and args.delete_hf_cache:
            print(f"Deleting HuggingFace cache to free disk: {cache_root}")
            shutil.rmtree(cache_root, ignore_errors=True)


if __name__ == "__main__":
    main()
