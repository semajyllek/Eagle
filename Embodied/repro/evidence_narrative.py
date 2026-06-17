# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
"""Static narrative for ``EVIDENCE.md`` -- the deep-dive prose, code-change
explanations, and testing methodology that surround the auto-generated tables in
``repro.combine.write_doc``.

The tables come from the per-PR ``results_*.json`` (always fresh numbers); the prose
here is the curated explanation of *what* changed, *why*, and *how it was tested*,
so EVIDENCE.md is a single self-contained document to draw the PR write-ups from.
Each function returns a markdown block.
"""
import os


def code_changes():
    """The per-change deep-dive (motivation + exact diffs + how-tested + speed),
    read from the curated ``repro/evidence/code_changes.md`` so the large diffs live
    in markdown (ported out of the per-PR notebooks) rather than Python strings."""
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "evidence", "code_changes.md"
    )
    try:
        txt = open(path, encoding="utf-8").read()
        if txt.startswith("<!--"):  # drop the editor-only source comment
            txt = txt.split("-->", 1)[1].lstrip("\n")
        return txt
    except Exception as e:
        return f"_(code_changes.md unavailable: {e})_"


def overview():
    return """\
# Batched + compiled LocateAnything-3B — evidence & deep dive

This document is the self-contained record for a stack of changes that take
`nvidia/LocateAnything-3B` from **single-sequence-only** inference to **batched,
optionally `torch.compile`d** inference, with full correctness evidence and measured
speedups. Every number below is each branch's *own* isolated run; the prose explains
the motivation, the exact code change, and how it was validated.

## The problem

The shipped model runs a non-standard decoder: a **Multi-Token-Prediction (MTP)
block-diffusion window** with an **auto-regressive (AR) fallback**, plus a per-step
**truncate-and-recompute of the KV cache**. That scheme is inherently per-sequence
(variable #accepted tokens per step, per-sequence mode switching, per-sequence
termination), which is why the original `generate()` hard-asserts `batch_size == 1`.
So today you can only perceive one (image, prompt) at a time.

## The stack

| PR | what it adds | the lever |
|----|--------------|-----------|
| **PR1** | lift the `batch_size==1` assert; a batched MTP/AR/hybrid driver (left-padded KV cache, per-step compaction, per-row mode/termination); the ViT packing **OOM fix** | **batching** (the big win) |
| **PR2** | unified `predict_batch` where `questions[i]` may be a str *or* a list -> each image's vision tower runs **once** and is reused across its prompts | **grouping** (one-image-many-queries) |
| **PR3** | vectorize the decode-loop bookkeeping + the per-row sampling; remove ViT per-image host syncs | **optimizations** (compile-enabling + eager wins) |
| **compile** | `torch.compile` the decoder (and optionally the ViT) on top of the batched path | **compile** (~1.6x on the forward) |

## How correctness is established

Three independent layers, each described in detail below:

1. **CPU equivalence tests** — no GPU, no checkpoint, no transformers. A
   deterministic `FakeLM` drives the batched driver and proves the *orchestration*
   (masks, cache, positions, mode-switching, termination) is integer-exact vs an
   independent re-implementation of the original single-sequence loop. Hundreds of
   checks across random seeds and all three modes.
2. **GPU semantic parity** — on the real checkpoint, `predict` (B=1, the original
   path) vs `predict_batch` produce the **same detections within tolerance** (same
   #boxes, same `<ref>` labels, coordinates within a few/1000). The only difference
   is the bf16 GEMM floor (below).
3. **The compile mAP gate** — `torch.compile` changes some greedy token streams
   (bf16-class non-determinism), so we score **mAP(eager) vs mAP(compiled)** on
   COCO + LVIS to prove those divergences don't cost real detections.

## The bf16 floor (why "parity" means "within tolerance", not "bit-identical")

The checkpoint runs in **bfloat16**. A bf16 matmul is **not batch-invariant**: cuBLAS
/ inductor pick a different GEMM kernel or accumulation order depending on batch
shape, and bf16's few mantissa bits make those orderings produce slightly different
low bits. Greedy decoding then occasionally flips a near-tied token (e.g. a
coordinate `800` vs `801`), which can cascade into a different-length-but-equivalent
output. This is **pre-existing model behavior**, surfaced three ways — batching
(different batch shape), early-eject (batch shape changes mid-decode), and compile
(inductor fusion reorders accumulation). It is fp32-recoverable but not something
these changes introduce. Hence every correctness claim is "same detections within
tolerance," and the mAP gate is what proves the tolerance is harmless.
"""


def correctness_methodology():
    return """\
# Correctness methodology (in detail)

## 1. CPU equivalence tests — `tests/test_batched_generate.py`

These run with **no GPU, no checkpoint, and no `transformers`**. A small,
fully-deterministic `FakeLM` honours the exact masking / KV-cache / position
contract of the real `Qwen2Model` (it reuses the real block-diffusion window mask
from `mask_sdpa_utils`, so the two cannot drift). Only the attention *numerics*
differ from the real model — the orchestration is identical, which is exactly what
these tests pin down. A typical run prints:

```
OK  runaway-box-guard checks: 5
OK  algorithm-parity checks: 432   batch-independence checks: 432   early-eject checks: 432   mask-window-vectorized checks: 1800
All batched-generation correctness tests passed.
```

Five kinds of check, by example:

1. **Algorithm parity (432)** — `batched_generate` on a *single* row must reproduce,
   token-for-token, an independent re-implementation of the original
   truncate-and-recompute single-sequence loop (`reference_single`). If the batched
   driver's bookkeeping is right, B=1 through it == the original algorithm.
   *Example:* for each seed and mode in `{fast, slow, hybrid}`, decode one FakeLM
   sequence both ways and assert the emitted token lists are equal.

2. **Batch independence (432)** — a row's output must be **identical whether it is
   decoded alone or inside a padded batch** with other (shorter/longer) rows. This
   is the core batching guarantee: left-padding, per-step compaction, ragged appends
   and per-row termination must not let rows interfere.
   *Example:* take rows `r0,r1,r2` of different lengths; assert
   `batched_generate([r0,r1,r2])[i] == batched_generate([ri])` for every `i`.

3. **Early-eject parity (432)** — dropping a finished row from the batch mid-decode
   (and trimming the KV cache in both batch and sequence dims) must **not change any
   surviving row's output**. FakeLM is exact, so this is asserted as bit-identical.

4. **Mask-window vectorized (1800)** — PR3's Stage-A change replaced the per-row
   Python mask dispatch (`apply_per_row_generation_window`, which did
   `input_ids[b,-1].item()` host syncs) with a tensor-only
   `apply_per_row_generation_window_vectorized`. The test asserts the two produce
   **identical masks** across 1800 randomized `[A, ...]` mode/shape combinations —
   so the vectorized form provably matches the reference it replaced.

5. **Runaway-box guard (5)** — `_is_runaway_box_loop` detects the degenerate
   "keep scanning this region" loop (a run of near-identical adjacent `<box>`
   detections drifting by `<= max_delta`). The test asserts it fires on a drifting
   run, and does *not* fire on a big-jump run, an interrupted run, or varied output.

For the PR3 **sampling** vectorization (B) and the **ViT** sync removal, the same
philosophy applies with dedicated gates:

- **B (`sample_tokens_batched`)** — the batched per-mode sampler is checked by the
  same `tests/test_batched_generate.py` suite: `batched_generate` (which now calls
  the batched `sample_group_*`) is compared against the per-row `sample_row_*`
  reference the test still uses. 432/432/432 holds, i.e. the batched sampler is
  token-identical to the per-row one.
- **ViT sync removal (`tests/test_modeling_vit.py`)** — `sdpa_attention`'s
  `.tolist()` slice-bound hoist is asserted `torch.equal` to the original
  `int(cu_seqlens[i])` per-element implementation across 5 ragged packings, and the
  whole function is asserted `allclose` to an independent dense block-diagonal
  `-inf`-mask reference.

## 2. GPU semantic parity — real checkpoint

On the real model, `semantic_parity` compares `predict` (B=1, the original code
path the dispatch still routes single sequences to) against `predict_batch` for a
handful of (image, prompt) pairs, per mode. "Parity" means **same detections**, not
identical token streams: it parses each output's `<box>...</box>` / `<ref>...</ref>`
tags and checks per row — `same_num_boxes`, `same_labels`, and `max_coord_diff` (in
the model's 0..1000 coordinate units, where 5 ~ 0.5% of the image). On COCO/LVIS at
50 samples this lands ~45/50 and ~42/50 within tolerance, with `median_coord_diff`
of 0 — i.e. the matched boxes are coordinate-identical and the few non-matches are
the bf16 floor.

## 3. The compile mAP gate — does compile cost real detections?

Parity says how often compiled output *differs* from eager; it cannot say whether
that difference *costs detections*. The `compile_ap` task answers that directly:

1. **Generate paired predictions.** Over 100 real eval images per dataset, run
   `predict_batch` **eager**, then again with the language model (and optionally the
   ViT) `torch.compile`d — both with the **same shipped settings** (`early_eject`
   on, greedy), so the *only* difference is `torch.compile`. Each pass's predictions
   are written to `compile_ap_out/<DS>/preds_<tag>.jsonl` in the eval's `to_record`
   format (predictions + ground truth per image).

2. **Score each pass with the repo metric.** The pipeline is the model's own
   offline detection metric, run automatically:
   `convert_coco_lvis_to_standard_format.py` turns a `preds_*.jsonl` (+ the COCO/LVIS
   GT json) into the FastEval TSV, then `coco_lvis_metric.py` (which uses the
   `fastevaluate` C++ extension — built from `evaluation/fastevaluate/`, *not* a PyPI
   package) prints Avg Precision / Precision@0.50 / Precision@0.95. The GT json comes
   from the same public `Mountchicken/Rex-Omni-EvalData` dataset the eval images do
   (COCO's `instances_val2017.json` ships inside `coco.tar.gz`; LVIS's
   `lvis_v1_val_with_filename2.json` is a loose file under `missing_annotaitons/`).
   `ap_score.score_compile_ap` drives this, fetching the GT and building
   `fastevaluate` on demand, and parses the printed AP.

3. **Read `ΔAP = AP(compiled) - AP(eager)`.** If it is within run-to-run noise
   (~±1%), the parity divergences are near-ties and compile is safe to ship; a real
   drop would mean compile loses detections (the fp32-logits fix). Measured: COCO and
   LVIS both land within ±1% (several positive), i.e. **compile is AP-neutral**.

The pred jsonl is written with `ensure_ascii=True` so byte-level tokenizer surrogates
in diverse (LVIS) outputs can't corrupt the file, and the metric subprocess output is
decoded with `errors="replace"` so non-ASCII category names in its stdout don't crash
the scorer — both learned the hard way.
"""
