#!/usr/bin/env python
r"""Stage fitted lenses in the Neuronpedia layout and print an upload command.

Copies what `fit_lens.py` wrote into the directory tree used by
https://huggingface.co/neuronpedia/jacobian-lens ::

    <np_model_id>/jlens/<dataset>/config.yaml
    <np_model_id>/jlens/<dataset>/<model>_convergence.csv
    <np_model_id>/jlens/<dataset>/<model>_jacobian_lens.pt

The `config.yaml` mirrors theirs field for field. Nothing in it is typed twice:
the dataset and fit blocks are parsed out of the recorded command line with
`fit_lens.build_parser()`, and the results block is read from the last row of
the convergence CSV, so a config can never drift from the run it describes.

Add a run by appending to `LENSES` below — `argv` is exactly what was passed to
`fit_lens.py`, and the source filenames follow from it.

Two fields are *not* derived from the fit artifacts and are only right if this
script is run on the machine that did the fitting: the `gpus:` block reads the
local GPU at packaging time (see `gpu_description`), and the recorded command is
whatever `LENSES` says it was. Package on the fitting host.

Run::

    uv run python scripts/package_lenses.py --repo_id <user>/jacobian-lens
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fit_lens import _slug, build_parser  # noqa: E402

REPO_URL = "https://github.com/model-organisms-for-real/jacobian-lens"

ATTRIBUTION = (
    "Jacobian lens ('jlens') by Anthropic PBC — companion code for the "
    "'Verbalizable Workspace' paper (https://github.com/anthropics/jacobian-lens), "
    f"Apache-2.0. Fit with scripts/fit_lens.py from {REPO_URL}, a reconstruction "
    "of Neuronpedia's fit_lens.py."
)

COMMON = [
    "--dataset", "Salesforce/wikitext",
    "--dataset_config", "wikitext-103-raw-v1",
    "--dataset_split", "train",
    "--text_field", "text",
    "--max_chars", "2000",
    "--n_prompts", "1000",
    "--dim_batch", "128",
    "--max_seq_len", "128",
    "--dtype", "bfloat16",
    "--device_map", "cuda",
    "--min_prompts", "100",
    "--stop_window", "10",
    "--stop_at_delta", "0.002",
    "--levels", "1e-2,5e-3,1e-3",
]


@dataclass(frozen=True)
class Lens:
    """One fitted lens to publish.

    Attributes:
        hf_model_name: The model on the Hub that was fitted. Names the files
            (``<basename>_jacobian_lens.pt``) and, lowercased, the directory.
        argv: The arguments `fit_lens.py` was run with, verbatim. The local
            file names follow from ``argv`` the same way `fit_lens.py` derived
            them, so nothing has to be restated here.
        np_model_id: Directory name, if not the lowercased model basename.
    """

    hf_model_name: str
    argv: list[str]
    np_model_id: str | None = None

    @property
    def stem(self) -> str:
        return self.hf_model_name.rstrip("/").split("/")[-1]

    @property
    def directory(self) -> str:
        return self.np_model_id or self.stem.lower()


LENSES = [
    Lens(
        hf_model_name="google/gemma-3-1b-it",
        argv=["/workspace/models/gemma3_1B/gemma3_1b_ancestor",
              "--out_dir", "./output", *COMMON],
    ),
    Lens(
        hf_model_name="allenai/OLMo-2-0425-1B-SFT",
        argv=["/workspace/models/olmo2_1B/olmo2_1b_base_sft",
              "--out_dir", "./output", *COMMON],
    ),
]


def read_results(csv_path: str) -> dict[str, str]:
    """Prompts fitted and the final metrics, from the convergence curve."""
    with open(csv_path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"{csv_path} has no rows")
    last = rows[-1]
    return {
        "prompts_fitted": last["n_done"],
        "final_identity_distance": last["identity_distance"],
        "final_mean_rel_change": last["mean_rel_change"],
    }


def gpu_description() -> tuple[str, str] | None:
    """``(name, total_gb)`` for GPU 0 of *this* host, or None without CUDA.

    This is read at packaging time, not fit time — ``fit_lens.py`` records no
    hardware information — so it describes the GPU the lens was fitted on only
    when packaging runs on the same machine as the fit. That is the intended
    workflow (fit, then package, on one machine); run this script elsewhere and the
    ``gpus:`` block in ``config.yaml`` will describe the wrong card.
    """
    if not torch.cuda.is_available():
        return None
    properties = torch.cuda.get_device_properties(0)
    return properties.name, f"{properties.total_memory / 1024**3:.1f}"


def render_config(lens: Lens, args: argparse.Namespace, command: str,
                  results: dict[str, str], gpu: tuple[str, str] | None) -> str:
    """The `config.yaml` body, in Neuronpedia's field order."""
    dataset_config = (
        "null" if args.dataset_config.lower() in ("none", "", "null")
        else f'"{args.dataset_config}"'
    )
    lines = [
        f"# Jacobian lens fit — generated by {os.path.basename(__file__)}",
        f"# {ATTRIBUTION}",
        "#",
    ]
    if gpu is not None:
        lines += [
            "# GPU used for this fit:",
            f"#   GPU 0 ({gpu[0]}): {gpu[1]} GB total",
            "#",
        ]
    lines += [
        "# Exact command used:",
        f"#   {command}",
        "#",
        f"# Generated: {datetime.now(timezone.utc).isoformat()}",
        f'np_model_id: "{lens.directory}"',
        f'hf_model_name: "{lens.hf_model_name}"',
        "dataset:",
        f'  name: "{args.dataset}"',
        f"  config: {dataset_config}",
        f'  split: "{args.dataset_split}"',
        f'  text_field: "{args.text_field}"',
        f"  max_chars: {args.max_chars}",
        "fit:",
        f"  n_prompts: {args.n_prompts}",
        f"  dim_batch: {args.dim_batch}",
        f"  max_seq_len: {args.max_seq_len}",
        f"  target_layer: {'null' if args.target_layer is None else args.target_layer}",
        f'  dtype: "{args.dtype}"',
        f'  device_map: "{args.device_map}"',
        f"  compile: {str(not args.no_compile).lower()}",
        f"  trust_remote_code: {str(args.trust_remote_code).lower()}",
        f"  stop_at_delta: {args.stop_at_delta}",
        f"  min_prompts: {args.min_prompts}",
        f"  stop_window: {args.stop_window}",
        f'  levels: "{args.levels}"',
    ]
    if gpu is not None:
        lines += [
            "gpus:",
            "  gpu_0:",
            f'    name: "{gpu[0]}"',
            f"    total_gb: {gpu[1]}",
        ]
    lines += [
        "results:",
        f"  prompts_fitted: {results['prompts_fitted']}",
        f"  final_identity_distance: {results['final_identity_distance']}",
        f"  final_mean_rel_change: {results['final_mean_rel_change']}",
        f'command: "{command}"',
        f'attribution: "{ATTRIBUTION}"',
        "",
    ]
    return "\n".join(lines)


def build_parser_for_packaging() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--staging_dir",
        default="dist/hf-lenses",
        help="directory to assemble the upload tree in (default: dist/hf-lenses)",
    )
    parser.add_argument(
        "--repo_id",
        default="<user>/jacobian-lens",
        help="Hub repo the printed upload command targets",
    )
    parser.add_argument(
        "--script",
        default="scripts/fit_lens.py",
        help="how to spell the fitting script in the recorded command line",
    )
    return parser


def main() -> None:
    args = build_parser_for_packaging().parse_args()
    gpu = gpu_description()
    staged: list[str] = []

    for lens in LENSES:
        fit_args = build_parser().parse_args(lens.argv)
        source_stem = os.path.join(fit_args.out_dir, _slug(fit_args.model))
        sources = {
            f"{lens.stem}_jacobian_lens.pt": f"{source_stem}_jacobian_lens.pt",
            f"{lens.stem}_convergence.csv": (
                fit_args.metrics_csv or f"{source_stem}_convergence.csv"
            ),
        }
        missing = [path for path in sources.values() if not os.path.exists(path)]
        if missing:
            raise SystemExit(f"{lens.stem}: missing {', '.join(missing)}")

        dataset_dir = fit_args.dataset.replace("/", "-")
        target_dir = os.path.join(
            args.staging_dir, lens.directory, "jlens", dataset_dir
        )
        os.makedirs(target_dir, exist_ok=True)
        for name, path in sources.items():
            shutil.copy2(path, os.path.join(target_dir, name))

        command = " ".join([args.script, *lens.argv])
        config = render_config(
            lens, fit_args, command,
            read_results(sources[f"{lens.stem}_convergence.csv"]), gpu,
        )
        with open(os.path.join(target_dir, "config.yaml"), "w") as handle:
            handle.write(config)

        staged.append(target_dir)
        size_mb = os.path.getsize(
            os.path.join(target_dir, f"{lens.stem}_jacobian_lens.pt")
        ) / 1024**2
        print(f"{target_dir}/  ({lens.stem}, {size_mb:.0f} MB lens + csv + config)")

    print(f"\nStaged {len(staged)} lens(es) under {args.staging_dir}/. Upload with:\n")
    print("  hf auth login   # once, if not already authenticated")
    print(f"  hf upload {args.repo_id} {args.staging_dir} . --repo-type model \\")
    print('      --commit-message "Add WikiText-103 Jacobian lenses"')


if __name__ == "__main__":
    main()
