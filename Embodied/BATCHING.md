# Batched inference for LocateAnything

The released `generate()` opened with `assert batch_size == 1, 'only batch size = 1
is supported now'`, so the model could only decode one image at a time. This change
removes that restriction and adds a correct batched decode path for the model's
custom multi-token-prediction (MTP) / auto-regressive (AR) / hybrid decoder.

The `batch_size == 1` path is **unchanged** — batching is strictly additive.

## Usage

The public `generate()` signature is unchanged; just pass a batch. Batched inputs
must be **left-padded** (the standard requirement for batched decoding, which the
model already enforced via its padding-side check):

```python
tokenizer.padding_side = "left"
inputs = processor(text=[t1, t2, t3], images=[i1, i2, i3],
                   return_tensors="pt", padding=True).to(device)

responses = model.generate(
    pixel_values=inputs["pixel_values"],
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    image_grid_hws=inputs.get("image_grid_hws"),
    tokenizer=tokenizer,
    use_cache=True,
    generation_mode="hybrid",
    max_new_tokens=512,
)   # -> list[str], one per input
```

`LocateAnythingWorker.predict_batch(images, questions, ...)` wraps this (sets
left padding, orders pixel values row-major).

> **Tip for large datasets:** `predict_batch` pads every row to the longest
> row *in that call*, with no internal chunking. If you're processing many
> (image, prompt) pairs of varying length in repeated `predict_batch` calls,
> sort by input length before splitting into calls — each call's max length
> (and thus its padding waste) shrinks. `repro/eval_parity.py`'s benchmark
> harness does this and measured ~2-4% lower wall-clock from it alone.

## How it works

A batched generation requires every per-sequence quantity in the original loop to
become per-row, because the decoder is not standard autoregression:

- **Per-row mode & termination.** Each row can be in MTP or AR at the same step,
  switch modes independently (`error_box` → AR, `box_end_ar` → MTP), and finish at
  a different step. The single `use_mtp` flag / `break` become per-row vectors; the
  loop runs until all rows finish.
- **Ragged accepts.** An MTP step emits a variable number of tokens (6 for a box,
  4 for a point, 1 on fallback, variable for refs), so rows grow at different
  rates. Each step's inputs are left-padded to the step's max width.
- **KV cache.** The original truncate-and-recompute scheme is per-sequence. The
  batched path keeps a single **left-padded KV cache that is compacted after every
  step** (dropping the speculative window and padding) so its width tracks the
  longest live sequence rather than growing by the window each step.
- **Masking.** `Qwen2Model`'s inference mask builder is made per-row mode-aware
  (the MTP bidirectional window is applied only to rows whose last token is a mask
  token); `B == 1` behaviour is byte-identical.

The numerically heavy pieces (`sample_tokens`, box/ref decoding, `handle_pattern`)
are reused unchanged — they already operated one row at a time.

### Vision encoder fix

The ViT SDPA fallback (used when flash-attn is unavailable) materialized a dense
`[1, S, S]` block-diagonal mask over the **entire packed batch of images**, i.e.
`O((Σ patches)²)` memory — which OOMs once several high-resolution images are
batched. Since that mask isolates each image, `sdpa_attention` now runs full
self-attention per image slice: identical numerics, memory `Σ nᵢ²` instead of
`(Σ nᵢ)²`.

### Runaway-decode guard

A small number of samples trigger a self-reinforcing greedy-decoding loop: the
model emits a long run of near-identical, slowly-drifting `<box>` detections
("keep scanning this region") instead of `<|im_end|>`, and runs all the way to
`max_new_tokens`. This is a property of greedy decoding itself — it happens with
or without batching — but in a batch it holds all other rows hostage: the batch
loop runs until every row finishes, so one runaway row forces the entire batch to
run for ~2048 steps even if every other row finished in a few dozen.

`batched_generate` detects a run of `_RUNAWAY_BOX_RUN` (default 8) consecutive
`<box>` segments whose per-coordinate deltas are all
`<= _RUNAWAY_BOX_MAX_DELTA` (default 8, out of the model's 0-1000 coordinate
range) and stops the row there, the same as hitting `<|im_end|>`. This prevents
a single degenerate row from dominating the batch's wall-clock.

The thresholds are tuned for the observed pathology (coordinate deltas of 3-7
over 11+ boxes). A dataset with legitimate dense, evenly-spaced detections (e.g.
scene-text/OCR, where each character/word is its own `<box>` with a small
consistent offset) could in principle trip the same heuristic. If that happens,
pass `runaway_box_run`/`runaway_box_max_delta` through
`generate_kwargs`/`predict_batch` to loosen the thresholds, or set
`runaway_box_run=0` to disable the guard entirely.

## Correctness

`tests/test_batched_generate.py` (CPU, no GPU/checkpoint, ~seconds) drives the
batched decoder with a deterministic fake LM that honours the model's exact
masking / KV-cache / position contract, and asserts:

1. **Algorithm parity** — batched decoding of a single row reproduces an
   independent re-implementation of the original truncate-and-recompute loop,
   token-for-token (integer-exact).
2. **Batch independence** — a row's output is identical decoded alone vs. inside a
   padded batch with longer/shorter neighbours.

Both hold across `fast`/`slow`/`hybrid` for many seeds (144 + 144 checks), and a
coverage check confirms the variable-length-accept and mode-switch paths are
exercised.

On the **real model**, two regimes:

- **Short, greedy** (`repro/notebooks/benchmark_pr1.ipynb`, §5 "GPU parity
  check", exact match, ≤256 tokens): batched output equals single-sequence
  output up to floating point. The model is deterministic run-to-run, and the
  only batched-vs-single difference is that **bf16 matmul is not
  batch-invariant** (cuBLAS selects a different GEMM kernel/accumulation at
  batch > 1) — a deterministic effect, recovered exactly in fp32,
  demonstrable with a single `(x[:1] @ W)` vs `(x @ W)[:1]`. In `hybrid`/`slow`
  this shifts a coordinate by at most ~1/1000; `fast` is checked structurally
  only (same boxes + labels) — see `semantic_parity`.

- **Long, saturating `max_new_tokens`** (§6, COCO/LVIS, `hybrid`,
  `max_new_tokens=2048` — every sample in this eval runs the full 2048
  tokens): greedy decoding means a single bf16-induced argmax flip at any step
  changes every token after it, so the §5 bound doesn't carry over directly.
  Structural parity (same #boxes/labels per category, batched vs single) was
  49/50 (COCO) and 48/50 (LVIS); among the matching samples, the largest
  single-coordinate drift was ~370px (COCO) / ~560px (LVIS). This is a property
  of greedy decoding over long horizons combined with non-batch-invariant
  matmul, not specific to batching — the same sensitivity would appear
  comparing two single-sequence runs on different hardware/batch sizes.

## Performance

Measured on A100-80GB, bf16, identical images (so vision is paid per item):

- **Decode throughput scales with batch:** ~2.5× (N=4) → ~3.3× (N=8) → ~4.0×
  (N=16) vs sequential single-sequence decoding.
- **End-to-end speedup ~1.75×** at N=16, because the **vision encoder is
  compute-bound and O(N)** (each image's attention is independent) and dominates
  wall-clock (~75%) for high-resolution images with short outputs. End-to-end is
  therefore Amdahl-bounded by vision, not by the decoder.
- **Heterogeneous output lengths:** the runaway-decode guard stops degenerate
  rows early, preventing a single straggler from holding the batch open for
  thousands of extra steps.

> Follow-up: for the common *one image, many queries* pattern the vision features
> can be encoded once and reused across the batch, which removes the O(N) vision
> cost and lifts end-to-end toward the decode speedup. Left for a separate change.

**Absolute speedups vary run-to-run** (GPU allocation, thermal state, etc.),
not just with batch size/input length. Two otherwise-identical runs of the
COCO/LVIS eval (`hybrid`, `max_new_tokens=2048`, `batch_size=8`) measured
4.19x-5.28x (COCO) and 3.23x-3.80x (LVIS) end-to-end. Read the numbers above
as "several-x", not a guaranteed fixed multiplier.

## Limitations

- **`fast` mode** (pure parallel MTP, no AR verification) is inherently sensitive
  to any logit perturbation; under batched bf16 an uncertain coordinate can flip
  far on borderline detections, even on short generations.
- **`hybrid`/`slow`** are far more stable on short generations (≤256 tokens,
  ~1/1000px drift — see Correctness) but not immune over long ones: on the
  COCO/LVIS eval (`max_new_tokens=2048`, every sample saturates), 1-2/50
  samples per dataset diverge from the single-sequence output after a
  bf16-induced argmax flip, with up to ~560px coordinate drift on the diverged
  sample. The structural detection set (#boxes/labels) still matches for
  ~96-98% of samples.
- Batched inference uses the **sdpa** attention path (required for the
  block-diffusion mask); `magi` remains single-sequence (training/packing).
- Batch size is memory-bound (prefill attention + the large-vocab logits).
