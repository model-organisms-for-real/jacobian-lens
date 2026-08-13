# scripts

## `fit_lens.py` — fit a lens the way the published ones were fit

Reconstruction of the `fit_lens.py` Neuronpedia used to produce the lenses at
[`neuronpedia/jacobian-lens`](https://huggingface.co/neuronpedia/jacobian-lens).
Each published lens ships a `config.yaml` recording the exact command; this
script takes the same flags, so those commands run here unchanged.

Everything numerical is upstream `jlens`: `from_hf` for model wrapping,
`jacobian_for_prompt` for the estimator, `JacobianLens` for IO, and upstream's
checkpoint format (an interrupted run of either this script or `jlens.fit()`
can be resumed by the other). What this script adds is the corpus loader, the
per-prompt convergence CSV, and early stopping — the reason it does not just
call `jlens.fit()` is that upstream has no per-prompt callback to hang those on.

### Replicating a published lens

The gemma-3-1b config records:

```
fit_lens.py google/gemma-3-1b-pt --out_dir ... --dataset Salesforce/wikitext \
  --dataset_config wikitext-103-raw-v1 --dataset_split train --text_field text \
  --max_chars 2000 --n_prompts 1000 --dim_batch 128 --max_seq_len 128 \
  --dtype bfloat16 --device_map cuda --min_prompts 100 --stop_window 10 \
  --levels 1e-2,5e-3,1e-3 --hf_cache_dir /tmp/jlens-hf-cache --stop_at_delta 0.002
```

which here is:

```bash
uv run python scripts/fit_lens.py google/gemma-3-1b-pt \
  --out_dir out/gemma-3-1b \
  --dataset Salesforce/wikitext --dataset_config wikitext-103-raw-v1 \
  --dataset_split train --text_field text --max_chars 2000 \
  --n_prompts 1000 --dim_batch 128 --max_seq_len 128 \
  --dtype bfloat16 --device_map cuda \
  --min_prompts 100 --stop_window 10 --stop_at_delta 0.002 \
  --levels 1e-2,5e-3,1e-3
```

Outputs, in `--out_dir`: `<model>_jacobian_lens.pt` (the lens, fp16, as
published) and `<model>_convergence.csv` (the per-prompt curve, same columns as
the published `*_convergence.csv`). `<model>_checkpoint.pt` exists only while
the fit is running.

### Metric fidelity

Neither convergence metric is defined in upstream `jlens`, so both were
reverse-engineered from the published curves and checked against them by
rerunning the command above on the same model, dtype and corpus stream:

| n_done | `identity_distance` published / here | `mean_rel_change` published / here |
| --- | --- | --- |
| 1 | 0.896891 / 0.896472 | — |
| 2 | 0.806387 / 0.806318 | 0.49799543 / 0.49802589 |
| 3 | 0.790692 / 0.790667 | 0.25377339 / 0.25371779 |
| 4 | 0.774229 / 0.774224 | 0.18668778 / 0.18664705 |
| 5 | 0.769498 / 0.769364 | 0.14829850 / 0.14810850 |

* `identity_distance` = `||J_l - I||_F / sqrt(d_model)` at the deepest fitted
  layer (`target_layer - 1`).
* `mean_rel_change` (Δmean) = `||M_new - M_old||_F / ||M_new||_F`, averaged over
  fitted layers. Note this is *not* the `max_d_mean` that `jlens.fit()` logs,
  which maxes over layers and divides by `||M_old||` and so reads ~1.4x larger:
  a `--stop_at_delta` threshold is only meaningful against the definition above.

Residual differences are run-to-run nondeterminism (bf16 + `torch.compile`);
back-to-back reruns here differ by about as much.
