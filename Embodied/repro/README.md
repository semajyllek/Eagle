# `repro/` — batched-inference verification & evidence tooling

Private tooling for the batched-inference change. **Not part of the upstream
PRs** — it lives on the working branches to (1) test/verify changes and
(2) generate the evidence cited in the PR descriptions. Everything is small,
importable, and copy-pasteable, so a reviewer (or you) can re-run any single
piece independently with one import.

Why it exists: a change this large is only acceptable if it's *easy to convince
yourself it doesn't break anything*. These functions reproduce the full
correctness + throughput story from one checkout.

> The tooling tests *this checkout's* code. Because the model loads via
> `trust_remote_code` (which runs the code **bundled in the HF repo**, not this
> repo's), `load_overlaid_worker()` downloads the checkpoint and **overlays** this
> checkout's patched files onto the snapshot before loading. Upstream, where the
> repo *is* the model code, this step is unnecessary.

## How the evidence is produced (per-PR notebooks + combine)

The throughput/eval evidence is **branch-isolated by notebook**, so each PR's
numbers come from running *that PR's own checkout* — nothing is code-swapped at
runtime:

- `repro/notebooks/benchmark_pr1.ipynb`, `benchmark_pr2.ipynb` — each clones the
  tooling (`feature`) **and** its own PR branch, loads the model from that
  checkout (`load_overlaid_worker(src_dir=<PR>/…/locany)`), runs the PR's own CPU
  test, then `run_benchmark(...)` (COCO/LVIS detection speedup + parity +
  output-length buckets + early-eject A/B, and the one-image-many-queries
  grounded eval). Every task calls only `worker.predict`/`worker.predict_batch` —
  the real shipped API — so each record reflects this checkout's own code.
  Writes `results_<tag>.json` and downloads it.
- `repro/notebooks/combine_results.ipynb` — upload the per-PR `results_*.json`;
  it merges them (`repro.combine`) into `EVIDENCE.md` + comparison tables. No model
  load — pure record merge, runs on CPU. (Same as `python -m repro.combine
  results_*.json -o EVIDENCE.md`.)

Regenerate the notebooks after editing the package: `python repro/build_notebook.py`.

### CLI (writes a markdown evidence report)
```bash
cd Embodied
python -m repro cpu      # no GPU/checkpoint: 144x3 integer-exact checks -> repro_evidence.md
python -m repro gpu      # downloads + overlays the checkpoint, runs the full suite
```

**3. Import a single helper** (share just the snippet you care about):
```python
from repro import load_overlaid_worker, sample_images
from repro import logit_equivalence, semantic_parity, speed_sweep
worker, snap = load_overlaid_worker()           # download + overlay this checkout + load
imgs, prompts = sample_images(4)
logit_equivalence(worker, imgs, prompts)        # bf16-floor vs no-pad vs left-pad
semantic_parity(worker, imgs, prompts)          # detections match within tolerance
speed_sweep(worker, imgs[0], prompts[0])        # throughput vs batch size
```

## What each check proves

| Function (module) | Claim it backs |
|---|---|
| `cpu_evidence()` (`cpu_evidence`) | Batched **orchestration is integer-exact** vs an independent re-impl of the original loop, **batch-independent**, and **early-eject-invariant**, across fast/slow/hybrid (144×3). No GPU/checkpoint. |
| `bf16_gemm_demo()` (`equivalence`) | Root cause: a single bf16 matmul is **not batch-invariant** (≈0 in fp32) — so the only batched-vs-single difference is hardware GEMM rounding, not our code. |
| `logit_equivalence()` (`equivalence`) | Batched/left-padded **prefill logits match B=1** to within the model's own bf16 run-to-run floor (padding adds nothing). |
| `semantic_parity()` (`parity`) | **Same detections** (count, `<ref>` labels, coords within a few /1000) batched vs single in `slow`/`hybrid`; `fast` matches structurally. |
| `fast_mode_diagnostic()` (`parity`) | `fast`-mode coordinate drift is **padding-independent** (bf16/MTP sensitivity), not a batching bug. |
| `speed_sweep()` (`perf`) | Sequential (N x B=1) vs this checkout's batched `predict_batch` (1 x B=N) on N distinct images — `endtoend_speedup` is the real wall-clock win, whatever this branch's `predict_batch` does for vision + decode. |
| `eval_parity()` (`eval_parity`) | Task-metric parity on **real data**: COCO/LVIS detection, `worker.predict` (B=1) == `worker.predict_batch` (batched), batched speedup, **output-length buckets** + an **early-eject A/B**. (mAP fills in after offline scoring of the written preds.) |
| `grounded_eval()` (`eval_parity`) | **One image, many queries** (RefCOCOg, grouped by image) — `worker.predict_batch([img]*K, qs)` (`batch_s`) vs sequential `worker.predict` (`single_s`). Branch-isolated: PR1's and PR2's `speedup_x` for the identical call are directly comparable. |
| `run_benchmark()` (`benchmark`) + `combine` | **Branch-isolated evidence**: each PR's *own notebook* loads its checkout and runs the same `run_benchmark`, saving a tagged record; `combine` merges the records — so PR1's numbers are PR1's code (no runtime code-swapping). |

## Module layout

| Module | Contents |
|---|---|
| `model.py` | `load_overlaid_worker()`, `sample_images()` |
| `overlay.py` | overlay patched files onto an HF snapshot (also a CLI) |
| `cpu_evidence.py` | `cpu_evidence()` |
| `equivalence.py` | `bf16_gemm_demo()`, `build_batched_inputs()`, `prefill_logits()`, `logit_equivalence()` |
| `parity.py` | `semantic_parity()`, `fast_mode_diagnostic()`, `grouped_parity()`, box parsers |
| `perf.py` | `speed_sweep()`, `free()` |
| `plots.py` | `plot_speed_sweep()` (matplotlib, lazy import; `as_base64=True` for embedding in `EVIDENCE.md`) |
| `report.py` | `gpu_evidence(worker)` orchestrator; `write_report()` → markdown |
| `eval_parity.py` | `eval_parity()` (COCO/LVIS detection via `worker.predict`/`worker.predict_batch`, output-length buckets + early-eject A/B) and `grounded_eval()` (one-image-many-queries via `worker.predict_batch([img]*K, qs)`) |
| `eval_data.py` | `download_eval_data()` — fetch + extract the COCO/LVIS eval data from the HF dataset |
| `results_table.py` | assemble the PR results table (accuracy + speedup) from the per-dataset summaries |
| `benchmark.py` | `run_benchmark(worker, tag, metadata, …)` — run the suite against a loaded worker and save `results_<tag>.json` (each per-PR notebook calls this) |
| `combine.py` | `python -m repro.combine results_*.json` — merge the per-PR records into `EVIDENCE.md` (detection, length buckets + eject, one-image-many-queries grounded, plus `speed_sweep` plots embedded as base64 if present) |
| `__main__.py` | the `python -m repro cpu|gpu` CLI |
| `build_notebook.py` | regenerates the per-PR + combine notebooks under `repro/notebooks/` |

## Task-metric parity on a real dataset

`run_benchmark` (and the per-PR notebooks) call this for you. In code it is one
call that reuses the already-loaded worker and downloads the data itself (cached):
```python
from repro import eval_parity
summaries = eval_parity(worker, datasets=("COCO", "LVIS"),
                        evaldata="/content/EvalData", limit=50, batch_size=8)
```
It runs the repo's detection inference on a subset twice (B=1 vs batched,
**greedy** so the only difference is grouping), prints a GT-free signal (identical
box structure + max coord diff) and the per-dataset `speedup_x`, and writes
`preds_before/after.jsonl` + `summary.json` under `parity_out/<dataset>/` for
offline AP scoring. A standalone CLI (`python repro/eval_parity.py --model_path
<snap> --test_jsonl ... --out_dir ...`) does one dataset without a loaded worker.
After scoring the preds with the repo's metric
(`evaluation/utils/convert_coco_lvis_to_standard_format.py` +
`evaluation/metrics/coco_lvis_metric.py`) you get the AP per split.

**The PR results table** (accuracy + speedup over datasets):
```bash
# 1. run eval_parity for each dataset (writes summary.json with the speedup)
# 2. score preds_before/after.jsonl with the repo metric to get AP per split
# 3. assemble:
python repro/results_table.py --out_dirs parity_coco parity_lvis \
    --ap_json ap.json > results.md
# ap.json: {"COCO": {"before": 41.2, "after": 41.2}, "LVIS": {"before": 33.8, "after": 33.7}}
```
Speedup + parity come from the summaries automatically; the AP columns are filled
from the metric. Omit `--ap_json` to get the speedup/parity table on its own.

## Notes

- **Output / logging.** The package logs via the standard `logging` module under
  the `repro` logger. The CLIs and the notebook call `logging.basicConfig(...)`;
  in your own scripts do the same to see messages. Functions **return** their
  results (dicts / lists of dicts) — wrap in `pandas.DataFrame` to display.
- **GPU equivalence is semantic within bf16 tolerance, not bitwise** — the model
  is deterministic run-to-run; the batched-vs-single difference is bf16 GEMM
  non-batch-invariance, recovered exactly in fp32 (`bf16_gemm_demo`).
- **All parity helpers use greedy** (`temperature=0`) so before/after are
  comparable; the real eval samples at 0.7.
- Regenerate the notebooks after editing the package: `python repro/build_notebook.py`
  (writes `repro/notebooks/benchmark_pr1.ipynb`, `benchmark_pr2.ipynb`, `combine_results.ipynb`).
