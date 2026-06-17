<!-- Source narrative for EVIDENCE.md's Code-changes section. Ported from the per-PR
benchmark notebooks so the deep-dive is self-contained; edit here, evidence_narrative
injects it. -->

# Code changes — motivation, exact diffs, how tested, speed

_Every change below is gated two ways: **equivalence** (bit-identical or same-detections-within-tolerance) AND **speed** (a measured number, or an honest note when a change is compile-enabling / memory-enabling rather than a direct win). See the per-change speed attribution at the end._


---

## PR1 — batching (`pr/batched-generation`)


## 1 · Motivation

The released `LocateAnythingForConditionalGeneration.generate()` opens with:

```python
assert batch_size == 1, 'only batch size = 1 is supported now'
```

So every call -- one image, one prompt -- runs its own prefill + decode loop end
to end, even though a GPU is most efficient processing several sequences
together. For a serving workload (many images/queries per second) that means N
independent forward passes instead of one batched pass.

Batching is non-trivial here because this model's decoder is **not** standard
autoregression: it's a hybrid Multi-Token-Prediction (MTP) / auto-regressive (AR)
decoder where, per sequence, (a) a step can emit a variable number of accepted
tokens, (b) the row can switch between MTP and AR mode mid-generation, (c) the KV
cache is truncated and recomputed each step, and (d) termination is independent
per sequence. All of that is written assuming a single sequence -- which is
exactly why the assertion above exists.

**PR1** removes that assertion and adds a batched decode path
(`batched_generate`, "Strategy C") that makes every one of those per-sequence
quantities per-row, while leaving the `B==1` path byte-for-byte unchanged (the
same code, just no longer behind the gate). The rest of this notebook:

- §2 shows the actual edits PR1 makes to the *existing* model files, and why each
  one was necessary,
- §3 describes the new code (the batched decode driver and the public
  `predict_batch` API),
- §4 runs PR1's own CPU correctness suite (no GPU) that proves the batched path is
  integer-exact vs. the original loop,
- §5 measures the real speedup, output-length spread, and early-eject
  straggler-removal win on COCO/LVIS.


## 2 · What changed in the model, and why

Batching this decoder took **four small, surgical edits to existing model
files** -- everything else PR1 adds (`batched_generate.py`, `predict_batch`, the
CPU tests) is new code that the `B==1` path never executes. These four diffs are
the part of PR1 most worth reading closely, since they're the only edits to code
the `B==1` path also runs through. (The same diffs are on the PR's "Files
changed" tab on GitHub; §4 below runs the test suite that proves they're
correct.)


### a) `modeling_locateanything.py` -- remove the gate, add the dispatch

This is the change that "opens the gate": the `assert batch_size == 1` is gone,
and when `batch_size > 1` the call is routed to the new `batched_generate` driver
*before* any of the `B==1`-only state (`generated`, `past_key_values`,
`total_gen_length`) is set up. The `B==1` code below is otherwise untouched --
same variables, same generation-mode loop, just now reached only when
`batch_size == 1`.

```diff
diff --git a/Embodied/eaglevl/utils/locany/modeling_locateanything.py b/Embodied/eaglevl/utils/locany/modeling_locateanything.py
index 8e61b1d..fd069dc 100644
--- a/Embodied/eaglevl/utils/locany/modeling_locateanything.py
+++ b/Embodied/eaglevl/utils/locany/modeling_locateanything.py
@@ -315,6 +315,8 @@ class LocateAnythingForConditionalGeneration(LocateAnythingPreTrainedModel, Gene
     ) -> torch.LongTensor:

         verbose = generate_kwargs.pop('verbose', False)
+        # batched-only knob; pop here so it never reaches the per-row samplers.
+        early_eject = generate_kwargs.pop('early_eject', True)
         start_time = time.time()
         prefill_time = None

@@ -324,14 +326,8 @@ class LocateAnythingForConditionalGeneration(LocateAnythingPreTrainedModel, Gene
             image_grid_hws = torch.from_numpy(image_grid_hws).to(pixel_values.device, dtype=torch.int32)

         batch_size, seq_len = input_ids.shape
-        assert batch_size == 1, 'only batch size = 1 is supported now'
         assert generate_kwargs.get('use_cache', False), "Only use_cache=True is supported."

-        generated = input_ids.clone()
-        total_gen_length = min(tokenizer.model_max_length, seq_len + generate_kwargs.get('max_new_tokens', 2048))
-        iter_round = 0
-        past_key_values = None
-
         # Extract visual features once before the loop
         if visual_features is not None:
             vit_embeds = visual_features
@@ -339,11 +335,40 @@ class LocateAnythingForConditionalGeneration(LocateAnythingPreTrainedModel, Gene
             vit_embeds = self.extract_feature(pixel_values, image_grid_hws)
         else:
             vit_embeds = None
-
+
         if image_grid_hws is not None:
             vit_embeds = torch.cat(vit_embeds, dim=0)
             vit_embeds = self.mlp1(vit_embeds)

+        # Batched decoding (Strategy C). The single-sequence loop below is kept as
+        # the proven B==1 fast path; B>1 routes to the batched driver, which keeps
+        # a left-padded, per-step-compacted KV cache and per-row mode/termination.
+        # Batched inputs must be LEFT-padded (`tokenizer.padding_side='left'`) with
+        # `attention_mask`, and `pixel_values`/visual features ordered row-major so
+        # they align with the flattened image-token slots of `input_ids`.
+        if batch_size > 1:
+            from .batched_generate import batched_generate
+            pad_token_id = tokenizer.pad_token_id
+            if pad_token_id is None:
+                pad_token_id = self.token_ids['im_end_token_id']
+            return batched_generate(
+                self,
+                input_ids=input_ids,
+                attention_mask=attention_mask,
+                vit_embeds=vit_embeds,
+                image_token_index=self.config.image_token_index,
+                tokenizer=tokenizer,
+                n_future=n_future_tokens,
+                pad_token_id=pad_token_id,
+                generate_kwargs=generate_kwargs,
+                early_eject=early_eject,
+            )
+
+        generated = input_ids.clone()
+        total_gen_length = min(tokenizer.model_max_length, seq_len + generate_kwargs.get('max_new_tokens', 2048))
+        iter_round = 0
+        past_key_values = None
+
         # ==================== Generation Mode ====================
         # 'fast'   : MTP only, never fall back to AR
         # 'slow'   : AR only, pure auto-regressive decoding
```


### b) `mask_sdpa_utils.py` -- new helper: per-row MTP/AR mask dispatch (additive)

The MTP "generation window" (bidirectional attention over the speculative block)
was applied to the *whole* sequence/batch at once, assuming every row is in the
same mode. In a batch, different rows can be in MTP mode or AR mode **at the same
step** -- so the mask has to be built per row: MTP rows get the bidirectional
window, AR rows stay causal. `apply_per_row_generation_window` does exactly that.
It's a pure addition (nothing existing is removed), and it's imported by both the
model (next diff) and the CPU tests, so the two implementations can't drift apart.

```diff
diff --git a/Embodied/eaglevl/utils/locany/mask_sdpa_utils.py b/Embodied/eaglevl/utils/locany/mask_sdpa_utils.py
index 1bb12f7..8258943 100644
--- a/Embodied/eaglevl/utils/locany/mask_sdpa_utils.py
+++ b/Embodied/eaglevl/utils/locany/mask_sdpa_utils.py
@@ -135,6 +135,40 @@ def update_causal_mask_for_one_gen_window_2d(
     return attn_mask_2d


+def apply_per_row_generation_window(
+    attention_mask_4d: torch.Tensor,
+    input_ids: torch.Tensor,
+    text_mask_token_id: int,
+    update_mask_func,
+) -> torch.Tensor:
+    """Apply an MTP generation-window mask update per row.
+
+    Rows whose last (right-aligned) token is ``text_mask_token_id`` are inside an
+    MTP generation window and get the bidirectional window via ``update_mask_func``;
+    rows ending in a real token stay plain-causal (AR). This is the per-row dispatch
+    used by both ``Qwen2Model`` inference and the CPU tests, factored out here so the
+    two cannot drift.
+
+    Args:
+        attention_mask_4d: additive mask ``[B, 1, S, Skv]`` (0 allow, -inf deny),
+            already causal + padding.
+        input_ids: ``[B, S]`` ids for this step (last column is the newest token).
+        text_mask_token_id: id marking an MTP window position.
+        update_mask_func: callable ``(input_ids_row, mask_2d) -> mask_2d`` that edits
+            the ``[S, Skv]`` mask for a single MTP row (e.g. a ``functools.partial``
+            of ``update_causal_mask_for_one_gen_window_2d``).
+    Returns:
+        ``[B, 1, S, Skv]`` mask with the window applied to MTP rows.
+    """
+    rows = []
+    for b in range(attention_mask_4d.shape[0]):
+        if input_ids[b, -1].item() == text_mask_token_id:
+            rows.append(update_mask_func(input_ids[b], attention_mask_4d[b, 0]).unsqueeze(0))
+        else:
+            rows.append(attention_mask_4d[b])
+    return torch.stack(rows, dim=0)
+
+
 def create_block_diff_mask_by_pe_4d(
     block_size: int,
     x0_len_list: torch.Tensor,
```


### c) `modeling_qwen2.py` -- use the per-row dispatcher in the mask builder

`Qwen2Model._prepare_block_mask_for_inference` previously built **one** mask
update for the whole batch, assuming a single uniform mode -- it even read
`input_ids[0]` to decide it, a `B==1` assumption baked into a function that runs
for every batch size. It now calls `apply_per_row_generation_window` (b) so mixed
AR/MTP batches get the right mask per row. The `seq_length == 1` early-out is
simplified too: a single new query token is always plain AR regardless of batch
size -- the old condition's `input_ids[0][-1]` check was itself a `B==1`
assumption.

```diff
diff --git a/Embodied/eaglevl/utils/locany/modeling_qwen2.py b/Embodied/eaglevl/utils/locany/modeling_qwen2.py
index c3069a9..4a61469 100644
--- a/Embodied/eaglevl/utils/locany/modeling_qwen2.py
+++ b/Embodied/eaglevl/utils/locany/modeling_qwen2.py
@@ -76,6 +76,7 @@ from .mask_sdpa_utils import (
     find_prefix_seq_length_by_pe,
     update_causal_mask_with_pad_non_visible_2d,
     update_causal_mask_for_one_gen_window_2d,
+    apply_per_row_generation_window,
     create_block_diff_mask_by_pe_4d,
     find_pred_pos_from_input_ids
 )
@@ -1279,12 +1280,11 @@ class Qwen2Model(Qwen2PreTrainedModel):
                 past_key_values_length,
                 sliding_window=self.config.sliding_window,
             )
-            # switch to ar mode
-            if seq_length == 1 or (input_ids is not None and input_ids[0][-1].item() != self.text_mask_token_id):
+            # A single new query token is always plain auto-regressive.
+            if seq_length == 1:
                 return attention_mask

-
-            if attention_mask is None or len(attention_mask.shape) != 4:
+            if attention_mask is None or len(attention_mask.shape) != 4 or input_ids is None:
                 return attention_mask

             # For SDLM, the generation window should set to bidirectional attention
@@ -1303,15 +1303,13 @@ class Qwen2Model(Qwen2PreTrainedModel):
                     causal_attn=self.causal_attn,
                 )

-            new_attention_mask = []
-            for b in range(attention_mask.shape[0]):
-                new_attention_mask.append(
-                    update_mask_func(
-                        input_ids[b],
-                        attention_mask[b][0],
-                    ).unsqueeze(0)
-                )
-            return torch.stack(new_attention_mask, dim=0)
+            # Per-row mode dispatch: only rows whose last (right-aligned) token is a
+            # mask token are in MTP mode and get the bidirectional generation window;
+            # rows ending in a real token stay plain-causal (AR). This makes mixed
+            # AR/MTP batches correct while leaving B==1 unchanged. Shared with the
+            # CPU tests via mask_sdpa_utils so the two cannot drift.
+            return apply_per_row_generation_window(
+                attention_mask, input_ids, self.text_mask_token_id, update_mask_func)

         def _prepare_block_mask_for_training():
             block_mask, _ = create_block_diff_mask_by_pe_4d(
```


### d) `modeling_vit.py` -- vision encoder memory fix (independent of the decoder change)

This one isn't about the decoder -- it's a prerequisite for batching to be usable
at all. The SDPA fallback in the vision tower built **one dense `[1, S, S]`
block-diagonal mask over all packed images** (`S` = total patches across the
batch), i.e. `O((sum patches)^2)` memory. Fine for one image; OOMs once several
high-resolution images are packed for a batch. Since that mask only ever isolated
each image's patches from every other image's, PR1 instead runs full
self-attention on each image's slice independently -- identical numerics, memory
`sum n_i^2` instead of `(sum n_i)^2`.

```diff
diff --git a/Embodied/eaglevl/utils/locany/modeling_vit.py b/Embodied/eaglevl/utils/locany/modeling_vit.py
index cc6b383..3c71f33 100644
--- a/Embodied/eaglevl/utils/locany/modeling_vit.py
+++ b/Embodied/eaglevl/utils/locany/modeling_vit.py
@@ -133,23 +133,24 @@ def sdpa_attention(
     Args:
         q, k, v: tensor of shape (batch_size, seqlen, num_heads, head_dim),
             or (tot_seqlens, num_heads, head_dim) if packing.
+
+    The packed sequence is attention-isolated per image (the original mask is
+    block-diagonal over ``cu_seqlens``). Materializing that dense ``[S, S]`` mask
+    is ``O(S^2)`` in the *total* packed length, which OOMs once several images are
+    packed together for batched inference. Instead, run full self-attention on
+    each image's slice independently: identical numerics, but memory is
+    ``sum_i n_i^2`` rather than ``(sum_i n_i)^2``.
     """
     seq_length = q.shape[0]
-    attention_mask = torch.zeros(
-        [1, seq_length, seq_length], device=q.device, dtype=torch.bool
-    )
+    outs = []
     for i in range(1, len(q_cu_seqlens)):
-        attention_mask[
-            ...,
-            q_cu_seqlens[i - 1] : q_cu_seqlens[i],
-            q_cu_seqlens[i - 1] : q_cu_seqlens[i],
-        ] = True
-    q = q.transpose(0, 1)
-    k = k.transpose(0, 1)
-    v = v.transpose(0, 1)
-    attn_output = F.scaled_dot_product_attention(q, k, v, attention_mask, dropout_p=0.0)
-    attn_output = attn_output.transpose(0, 1)
-    attn_output = attn_output.reshape(seq_length, -1)
+        s, e = int(q_cu_seqlens[i - 1]), int(q_cu_seqlens[i])
+        qi = q[s:e].transpose(0, 1)   # [num_heads, n_i, head_dim]
+        ki = k[s:e].transpose(0, 1)
+        vi = v[s:e].transpose(0, 1)
+        oi = F.scaled_dot_product_attention(qi, ki, vi, dropout_p=0.0)
+        outs.append(oi.transpose(0, 1))   # [n_i, num_heads, head_dim]
+    attn_output = torch.cat(outs, dim=0).reshape(seq_length, -1)
     return attn_output
```


## 3 · New code: the batched decode driver and `predict_batch`

The four edits above are what the `B==1` path also runs through. The rest of PR1
is new code, exercised only when `batch_size > 1`.

**`eaglevl/utils/locany/batched_generate.py`** (422 lines, new module) -- the
batched decode loop. Per-row state (`gen_tokens`, `use_mtp`, `finished`,
`cache_len`, ...) replaces the single-sequence scalars; each step left-pads the
ragged per-row accepts to a common width, runs one batched forward, and applies
each row's sampler (`sample_row_mtp`/`sample_row_ar`, faithful copies of the
`B==1` closures -- the *numerics* are unchanged, only the bookkeeping around them
is batched). The excerpt below is the **early-eject** step at the top of the
loop -- the trickiest indexing in this file, gathering the KV cache down to the
active rows and trimming it to the longest survivor:


```python
    past_key_values = None
    Ckv = 0
    first_step = True
    # Rows currently represented in the batch/cache, in batch-dim order. With
    # early-eject this shrinks as rows finish; otherwise it stays the full batch
    # and finished rows ride along as empty (padded) segments.
    cache_rows = list(range(B))

    while not all(finished):
        # ---- Step 0: early-eject -- drop finished rows from the cache. -------
        # Gathers the KV cache down to the still-active rows (batch dim) and trims
        # its width to the longest survivor (sequence dim), so finished stragglers
        # stop costing compute. Cache is left-padded, so the valid region is the
        # rightmost `cache_len[b]` columns; keeping the last `new_w` preserves it.
        if early_eject:
            active = [b for b in cache_rows if not finished[b]]
            if len(active) != len(cache_rows):
                if past_key_values is not None and active:
                    sel = torch.tensor(
                        [cache_rows.index(b) for b in active], device=device
                    )
                    new_w = max(cache_len[b] for b in active)
                    if new_w > 0:
                        past_key_values = tuple(
                            (
                                k.index_select(0, sel)[:, :, Ckv - new_w:, :].contiguous(),
                                v.index_select(0, sel)[:, :, Ckv - new_w:, :].contiguous(),
                            )
                            for k, v in past_key_values
                        )
                        Ckv = new_w
                    else:
                        past_key_values, Ckv = None, 0
                cache_rows = active
        rows = cache_rows
```


**`locateanything_worker.py::predict_batch`** (new method, 73 lines) -- the
public entry point used by the benchmark in §5. It left-pads the batch, calls
`model.generate(...)` (now reaching the new dispatch in (a)), and returns one
decoded string per `(image, question)` pair.

```diff
diff --git a/Embodied/locateanything_worker.py b/Embodied/locateanything_worker.py
index d6cd47a..00889e9 100644
--- a/Embodied/locateanything_worker.py
+++ b/Embodied/locateanything_worker.py
@@ -96,6 +96,79 @@ class LocateAnythingWorker:
             result["stats"] = response[2]
         return result

+    @torch.no_grad()
+    def predict_batch(
+        self,
+        images: list,
+        questions: list,
+        generation_mode: str = "hybrid",
+        max_new_tokens: int = 2048,
+        temperature: float = 0.0,
+        top_p: float = 0.9,
+        repetition_penalty: float = 1.1,
+        early_eject: bool = True,
+    ) -> list:
+        """Run several perception queries in a single batched forward.
+
+        Each ``(image, question)`` pair is decoded independently but the prefill
+        and every decode step are batched on the GPU. Inputs are LEFT-padded
+        (required by the batched decoder). Returns a list of answer strings, one
+        per input pair (no timing stats -- ``verbose`` is meaningless batched).
+
+        Note: with ``temperature > 0`` the per-row sampling is stochastic, so the
+        batched results will not be bitwise-identical to repeated single calls;
+        use ``temperature=0`` (greedy) for reproducible/comparable output.
+
+        ``early_eject`` (default ``True``) drops a row from the batch as soon as it
+        finishes so the rest are not held back by the longest sequence; set
+        ``False`` to keep every row in the batch until all are done (only useful
+        for A/B timing -- outputs match up to the bf16 floor).
+        """
+        assert len(images) == len(questions), "images and questions must align"
+        if len(images) == 1:
+            return [self.predict(
+                images[0], questions[0], generation_mode=generation_mode,
+                max_new_tokens=max_new_tokens, temperature=temperature,
+                verbose=False)["answer"]]
+
+        # Left padding is mandatory for the batched KV-cache layout.
+        self.tokenizer.padding_side = "left"
+        if hasattr(self.processor, "tokenizer"):
+            self.processor.tokenizer.padding_side = "left"
+
+        texts, all_images = [], []
+        for image, question in zip(images, questions):
+            messages = [{"role": "user", "content": [
+                {"type": "image", "image": image},
+                {"type": "text", "text": question},
+            ]}]
+            texts.append(self.processor.py_apply_chat_template(
+                messages, tokenize=False, add_generation_prompt=True))
+            imgs, _ = self.processor.process_vision_info(messages)
+            all_images.extend(imgs)
+
+        inputs = self.processor(
+            text=texts, images=all_images, videos=None,
+            return_tensors="pt", padding=True,
+        ).to(self.device)
+
+        responses = self.model.generate(
+            pixel_values=inputs["pixel_values"].to(self.dtype),
+            input_ids=inputs["input_ids"],
+            attention_mask=inputs["attention_mask"],
+            image_grid_hws=inputs.get("image_grid_hws", None),
+            tokenizer=self.tokenizer,
+            max_new_tokens=max_new_tokens,
+            use_cache=True,
+            generation_mode=generation_mode,
+            temperature=temperature,
+            do_sample=temperature > 0,
+            top_p=top_p,
+            repetition_penalty=repetition_penalty,
+            early_eject=early_eject,
+        )
+        return list(responses)
+
     # ---- Convenience methods for each task ----
```


---

## PR2 — grouping / one-image-many-queries (`pr2/grouped-batch`)


## 1 · Motivation

PR1 batches the *decode*, but for "one image, many queries" (e.g. several
questions about one screenshot, or several referring expressions against one
photo) each query still pays its own pass through the vision encoder -- and the
vision encoder is the dominant cost for high-resolution images with short
outputs (PR1's own numbers: ~75% of wall-clock at N=16 with uniform-length
outputs). Encoding the same image N times is pure waste.

**PR2** adds a *grouped* form to `predict_batch`: pass `questions[i]` as a list
of prompts for `images[i]`, and that image is encoded **once**, with its features
reused across every one of its prompts -- only the (already-batched, from PR1)
decode runs per prompt. This notebook:

- §2 shows the two small edits PR2 makes on top of PR1, and why each is
  necessary,
- §3 re-runs PR1's CPU correctness suite (grouping doesn't touch the decode loop,
  so this should be unaffected),
- §4 measures detection (identical to PR1 -- one query per image, nothing to
  dedup) and the **RefCOCOg** one-image-many-queries showcase where encode-once
  pays off.


## 2 · What changed for PR2, and why

On top of everything in PR1, two more changes enable "encode once, reuse for many
prompts":


### a) `modeling_locateanything.py` -- accept `visual_features` without `pixel_values`

`generate()` already accepted a `visual_features` shortcut (precomputed,
projected image features) as an alternative to `pixel_values` +
`extract_feature(...)`, but two lines *upstream* of that branch unconditionally
touched `pixel_values` -- `.to(self.language_model.dtype)`, and (for
`image_grid_hws`) `.to(pixel_values.device, ...)` -- so calling with
`pixel_values=None` crashed before ever reaching the `visual_features` branch.
PR2 makes both lines conditional / uses `input_ids.device` instead, so a caller
can pass `visual_features` alone. This is the hook PR2's worker change (b) relies
on.

```diff
diff --git a/Embodied/eaglevl/utils/locany/modeling_locateanything.py b/Embodied/eaglevl/utils/locany/modeling_locateanything.py
index fd069dc..b3c983e 100644
--- a/Embodied/eaglevl/utils/locany/modeling_locateanything.py
+++ b/Embodied/eaglevl/utils/locany/modeling_locateanything.py
@@ -320,10 +320,13 @@ class LocateAnythingForConditionalGeneration(LocateAnythingPreTrainedModel, Gene
         start_time = time.time()
         prefill_time = None

-        pixel_values = pixel_values.to(self.language_model.dtype)
+        # `pixel_values` is optional when precomputed `visual_features` are passed
+        # (e.g. encode-once for one image / many prompts).
+        if pixel_values is not None:
+            pixel_values = pixel_values.to(self.language_model.dtype)
         # Convert numpy array to tensor if needed
         if isinstance(image_grid_hws, np.ndarray):
-            image_grid_hws = torch.from_numpy(image_grid_hws).to(pixel_values.device, dtype=torch.int32)
+            image_grid_hws = torch.from_numpy(image_grid_hws).to(input_ids.device, dtype=torch.int32)

         batch_size, seq_len = input_ids.shape
         assert generate_kwargs.get('use_cache', False), "Only use_cache=True is supported."
```


### b) `locateanything_worker.py` -- grouped `predict_batch` + `_encode_image_features`

Two additive changes on top of PR1's `predict_batch`:

- **`_encode_image_features`** (new helper) runs just the vision tower + `mlp1`
  projection for one image, returning its projected feature tensor -- the same
  computation `generate()` would otherwise do internally via `extract_feature`,
  but callable once per distinct image.
- **`predict_batch`**: `questions[i]` may now be a string *or a list of strings*.
  It builds `qlists` (one list per image), encodes each **distinct** image once
  into `feat_cache` (keyed by `id(image)`), lays out one text row per
  `(image, prompt)` reusing that image's cached features, and calls
  `model.generate(visual_features=..., image_grid_hws=None, ...)` -- landing on
  the (a) code path. `_regroup` then reshapes the flat per-row outputs back to
  match the input shape (string in -> string out, list in -> list out), so the
  flat PR1 call pattern (`predict_batch(imgs, ["q1", "q2"])`) is unchanged.

```diff
diff --git a/Embodied/locateanything_worker.py b/Embodied/locateanything_worker.py
index 00889e9..972eed5 100644
--- a/Embodied/locateanything_worker.py
+++ b/Embodied/locateanything_worker.py
@@ -96,6 +96,20 @@ class LocateAnythingWorker:
             result["stats"] = response[2]
         return result

+    @torch.no_grad()
+    def _encode_image_features(self, image):
+        """Encode one image through the vision tower -> projected ``[tokens, C]``."""
+        messages = [{"role": "user", "content": [
+            {"type": "image", "image": image}, {"type": "text", "text": "x"}]}]
+        text = self.processor.py_apply_chat_template(
+            messages, tokenize=False, add_generation_prompt=True)
+        imgs, _ = self.processor.process_vision_info(messages)
+        inp = self.processor(text=[text], images=imgs, videos=None,
+                             return_tensors="pt", padding=True).to(self.device)
+        ghw = torch.as_tensor(inp["image_grid_hws"], dtype=torch.int32, device=self.device)
+        feats = self.model.extract_feature(inp["pixel_values"].to(self.dtype), ghw)
+        return self.model.mlp1(torch.cat(feats, dim=0))
+
     @torch.no_grad()
     def predict_batch(
         self,
@@ -108,16 +122,22 @@ class LocateAnythingWorker:
         repetition_penalty: float = 1.1,
         early_eject: bool = True,
     ) -> list:
-        """Run several perception queries in a single batched forward.
+        """Batched perception over a list of images.
+
+        ``questions[i]`` is either a single prompt string for ``images[i]`` or a
+        list of prompts to run against ``images[i]``. Each image is encoded by the
+        vision tower **once** and its features reused across all of its prompts
+        (the vision encoder is compute-bound and O(#distinct images)); every
+        prompt's decode is batched together on the GPU.

-        Each ``(image, question)`` pair is decoded independently but the prefill
-        and every decode step are batched on the GPU. Inputs are LEFT-padded
-        (required by the batched decoder). Returns a list of answer strings, one
-        per input pair (no timing stats -- ``verbose`` is meaningless batched).
+        The return mirrors the input: ``result[i]`` is a string if ``questions[i]``
+        was a string, else a list of strings aligned to ``questions[i]``. So a flat
+        ``predict_batch(imgs, ["q1", "q2"])`` returns ``["a1", "a2"]`` and a
+        one-image-many-queries ``predict_batch([img], [["q1", "q2"]])`` returns
+        ``[["a1", "a2"]]``.

-        Note: with ``temperature > 0`` the per-row sampling is stochastic, so the
-        batched results will not be bitwise-identical to repeated single calls;
-        use ``temperature=0`` (greedy) for reproducible/comparable output.
+        Inputs are LEFT-padded (required by the batched decoder). Use
+        ``temperature=0`` (greedy) for reproducible/comparable output.

         ``early_eject`` (default ``True``) drops a row from the batch as soon as it
         finishes so the rest are not held back by the longest sequence; set
@@ -125,38 +145,64 @@ class LocateAnythingWorker:
         for A/B timing -- outputs match up to the bf16 floor).
         """
         assert len(images) == len(questions), "images and questions must align"
-        if len(images) == 1:
-            return [self.predict(
-                images[0], questions[0], generation_mode=generation_mode,
+        scalar = [isinstance(q, str) for q in questions]
+        qlists = [[q] if s else list(q) for q, s in zip(questions, scalar)]
+        total_rows = sum(len(ql) for ql in qlists)
+
+        def _regroup(flat):
+            out, off = [], 0
+            for i, ql in enumerate(qlists):
+                chunk = flat[off:off + len(ql)]
+                off += len(ql)
+                out.append(chunk[0] if scalar[i] else chunk)
+            return out
+
+        # A single (image, prompt) -> the proven single-sequence path.
+        if total_rows == 1:
+            i = next(k for k, ql in enumerate(qlists) if ql)
+            ans = self.predict(
+                images[i], qlists[i][0], generation_mode=generation_mode,
                 max_new_tokens=max_new_tokens, temperature=temperature,
-                verbose=False)["answer"]]
+                verbose=False)["answer"]
+            return _regroup([ans])

         # Left padding is mandatory for the batched KV-cache layout.
         self.tokenizer.padding_side = "left"
         if hasattr(self.processor, "tokenizer"):
             self.processor.tokenizer.padding_side = "left"

-        texts, all_images = [], []
-        for image, question in zip(images, questions):
-            messages = [{"role": "user", "content": [
-                {"type": "image", "image": image},
-                {"type": "text", "text": question},
-            ]}]
-            texts.append(self.processor.py_apply_chat_template(
-                messages, tokenize=False, add_generation_prompt=True))
-            imgs, _ = self.processor.process_vision_info(messages)
-            all_images.extend(imgs)
-
+        # Encode each distinct image once; lay out one text row per (image, prompt)
+        # and reuse that image's features for every one of its prompts.
+        feat_cache, texts, row_images, row_feats = {}, [], [], []
+        for img, ql in zip(images, qlists):
+            if not ql:
+                continue
+            if id(img) not in feat_cache:
+                feat_cache[id(img)] = self._encode_image_features(img)
+            for q in ql:
+                msgs = [{"role": "user", "content": [
+                    {"type": "image", "image": img}, {"type": "text", "text": q}]}]
+                texts.append(self.processor.py_apply_chat_template(
+                    msgs, tokenize=False, add_generation_prompt=True))
+                row_images.append(img)
+                row_feats.append(feat_cache[id(img)])
+
+        # The processor needs an image per row to lay out image-token slots; we
+        # only USE the precomputed `row_feats` (one encode per distinct image).
+        proc_images = []
+        for img in row_images:
+            imgs, _ = self.processor.process_vision_info(
+                [{"role": "user", "content": [{"type": "image", "image": img}]}])
+            proc_images.extend(imgs)
         inputs = self.processor(
-            text=texts, images=all_images, videos=None,
-            return_tensors="pt", padding=True,
-        ).to(self.device)
+            text=texts, images=proc_images, videos=None,
+            return_tensors="pt", padding=True).to(self.device)

         responses = self.model.generate(
-            pixel_values=inputs["pixel_values"].to(self.dtype),
+            visual_features=torch.cat(row_feats, dim=0),
+            image_grid_hws=None,                             # features already projected
             input_ids=inputs["input_ids"],
             attention_mask=inputs["attention_mask"],
-            image_grid_hws=inputs.get("image_grid_hws", None),
             tokenizer=self.tokenizer,
             max_new_tokens=max_new_tokens,
             use_cache=True,
@@ -167,7 +213,7 @@ class LocateAnythingWorker:
             repetition_penalty=repetition_penalty,
             early_eject=early_eject,
         )
-        return list(responses)
+        return _regroup(list(responses))

     # ---- Convenience methods for each task ----
```


---

## PR3 — decode optimizations (`pr3/vectorized-decode`)


## 1 · Motivation

PR1 made the decode loop's *algorithm* per-row (a Python `for b in rows:` loop
computing each row's mask/input-block/cache-compaction). It didn't change *how*
those per-row quantities are computed -- each loop iteration does a handful of
small tensor ops, and two spots call `.item()`, forcing a GPU->CPU sync.

Per-row loops cost two things every decode step, independent of how much actual
matmul compute that step does:

- **GPU->CPU syncs** (`.item()`) stall the CPU until the GPU finishes and a
  value is copied back -- `B` of them per forward is `B` stalls, and each one
  also blocks the CPU from queuing the *next* step's kernels.
- **Kernel-launch overhead** -- even sync-free per-row ops (`cat`, `arange`,
  slices) cost ~10-50us of CPU dispatch each; `A` rows x ~10 ops/row is `O(A)`
  launches for work a single batched op would do in `O(1)`.

PR1/PR2's `speed_sweep` numbers were dominated by matmul compute, which grows
with batch size (1.09x at batch_size=1 up to ~1.26x at batch_size=16,
`results_pr2.json`) -- the per-step bookkeeping overhead above is roughly
*constant* per step, so it's a larger fraction of small/cheap steps than large
ones. **PR3 (Stages A & B) targets that constant-per-step overhead** -- four
surgical rewrites, each replacing a per-row Python loop with a closed-form
whole-batch tensor expression, with **zero algorithm change** (every check in
`DECODING_DEEP_DIVE.md` Parts B/C/D still holds). Part E of that doc covers this
in full detail; this notebook:

- §2 shows the four diffs and why each removes sync/launch overhead,
- §3 runs PR3's CPU correctness suite -- the bar here is **bit-exact**
  equivalence with the pre-PR3 per-row code (not just "still valid"), since
  this is a pure refactor,
- §4 measures the real effect on `speed_sweep` and `eval` timings, with a
  calibrated expectation set in §1 of `DECODING_DEEP_DIVE.md` Part E: a modest,
  possibly within-noise improvement at low batch sizes / in `fast` mode, **not**
  a new multiplier on top of PR1/PR2's 1.2x-2.1x.


## 2 · What changed for PR3, and why

Stage A removes two `.item()`-sync sources from the per-forward mask path
(`mask_sdpa_utils.py` + `modeling_qwen2.py`); Stage B replaces
`batched_generate.py`'s two remaining per-row loops (input-block assembly and
cache-compaction index build) with closed-form `[A, W]` / `[A, new_Ckv]` tensor
expressions.


### a) `mask_sdpa_utils.py` -- vectorized mask-window dispatch (additive)

PR1's `apply_per_row_generation_window` (used by `modeling_qwen2.py`, diff (b))
calls `input_ids[b, -1].item()` once per row to decide MTP-window vs.
plain-causal -- `B` syncs per forward, on every step with at least one MTP row.
This adds two new functions, used together as a drop-in replacement when
`use_cache=True`:

- `update_causal_mask_for_one_gen_window_4d` -- the all-rows version of the
  existing per-row 2D mask edit. The edit is pure corner-block slicing on the
  trailing `[S, Skv]` dims with no `input_ids` dependence, so it applies
  unchanged to every row via `...`-indexing. The "mask the previous round's
  last token" edit is written as a length-1 *slice* (not an int index) so that
  for AR rows -- where this edit is computed and discarded, and `Skv` can be
  `< block_size + 1` -- it's an empty no-op instead of an `IndexError`; for
  genuine MTP rows `Skv >= block_size + 1` always, so the slice selects the
  same single column the original int index did.
- `apply_per_row_generation_window_vectorized` -- the per-row MTP/AR test
  becomes a tensor comparison (`input_ids[:, -1] == text_mask_token_id`, no
  sync); the `_4d` edit above is applied once to a clone covering the whole
  batch; `torch.where` selects windowed-vs-causal per row. Zero `.item()`
  calls. The cost: AR rows get the window edit computed and discarded (a few
  wasted FLOPs on a `[block_size, block_size]` mask corner) -- traded for
  removing up to `B` syncs per forward.

This is purely additive -- the original per-row
`apply_per_row_generation_window` / `update_causal_mask_for_one_gen_window_2d`
are untouched, since `use_cache=False` is asserted-unsupported by every
`generate()` path and still uses them.

```diff
diff --git a/Embodied/eaglevl/utils/locany/mask_sdpa_utils.py b/Embodied/eaglevl/utils/locany/mask_sdpa_utils.py
index 8258943..b182d47 100644
--- a/Embodied/eaglevl/utils/locany/mask_sdpa_utils.py
+++ b/Embodied/eaglevl/utils/locany/mask_sdpa_utils.py
@@ -169,6 +169,76 @@ def apply_per_row_generation_window(
     return torch.stack(rows, dim=0)


+def update_causal_mask_for_one_gen_window_4d(
+    attn_mask_4d: torch.Tensor,
+    block_size: int = 4,
+    use_cache: bool = True,
+    causal_attn: bool = False,
+) -> torch.Tensor:
+    """Batched (all-rows) version of ``update_causal_mask_for_one_gen_window_2d``.
+
+    The 2D version's edits are pure corner-block slicing on the trailing
+    ``[S, Skv]`` dims and don't depend on ``input_ids``, so they apply unchanged
+    to every row of a ``[B, 1, S, Skv]`` mask via ``...``-indexing. Used by
+    ``apply_per_row_generation_window_vectorized`` to update all rows at once;
+    the per-row MTP/AR dispatch is applied afterwards via ``torch.where``.
+
+    Args:
+        attn_mask_4d: additive mask ``[..., S, Skv]`` (0 allow, -inf deny),
+            already causal + padding. Edited in place and returned.
+        block_size: size of the diffusion window.
+        use_cache: whether key-value cache is being used.
+        causal_attn: if True, maintains strict causal masking throughout.
+    Returns:
+        ``attn_mask_4d`` with the generation-window edits applied to every row.
+    """
+    if not causal_attn:
+        # Make the diffusion window (last block_size tokens) fully visible to itself.
+        attn_mask_4d[..., -block_size:, -block_size:] = 0.0
+    if use_cache:
+        # Mask the last token from the previous round to prevent recomputation.
+        # Written as a length-1 slice (not `[-block_size - 1]`) so that for rows
+        # where this update is unconditionally computed but discarded (the
+        # vectorized AR-row case in apply_per_row_generation_window_vectorized,
+        # where Skv can be < block_size + 1), the slice is empty instead of
+        # raising IndexError -- for genuine MTP rows Skv >= block_size + 1
+        # always (Skv >= S >= pend_len + block_size >= block_size + 1), so the
+        # slice still selects exactly the one column the 2D version indexed.
+        attn_mask_4d[..., -block_size:, -block_size - 1 : -block_size] = -float("inf")
+    return attn_mask_4d
+
+
+def apply_per_row_generation_window_vectorized(
+    attention_mask_4d: torch.Tensor,
+    input_ids: torch.Tensor,
+    text_mask_token_id: int,
+    update_mask_func_4d,
+) -> torch.Tensor:
+    """Vectorized equivalent of ``apply_per_row_generation_window`` for 4D-capable
+    ``update_mask_func``s (e.g. ``update_causal_mask_for_one_gen_window_4d``).
+
+    The original per-row loop calls ``input_ids[b, -1].item()`` for every row,
+    forcing ``B`` GPU->CPU syncs per forward step. Here the per-row MTP/AR
+    dispatch stays a tensor comparison (``input_ids[:, -1] == text_mask_token_id``,
+    no sync), ``update_mask_func_4d`` is applied once to a clone covering every
+    row, and ``torch.where`` selects the windowed vs. plain-causal mask per row.
+
+    Args:
+        attention_mask_4d: additive mask ``[B, 1, S, Skv]`` (0 allow, -inf deny),
+            already causal + padding.
+        input_ids: ``[B, S]`` ids for this step (last column is the newest token).
+        text_mask_token_id: id marking an MTP window position.
+        update_mask_func_4d: callable ``(mask_4d) -> mask_4d`` that edits the
+            generation window for *every* row in place (e.g. a
+            ``functools.partial`` of ``update_causal_mask_for_one_gen_window_4d``).
+    Returns:
+        ``[B, 1, S, Skv]`` mask with the window applied to MTP rows.
+    """
+    is_mtp_row = (input_ids[:, -1] == text_mask_token_id).view(-1, 1, 1, 1)
+    windowed = update_mask_func_4d(attention_mask_4d.clone())
+    return torch.where(is_mtp_row, windowed, attention_mask_4d)
+
+
 def create_block_diff_mask_by_pe_4d(
     block_size: int,
     x0_len_list: torch.Tensor,
```


### b) `modeling_qwen2.py` -- wire in (a), and drop a dead-code sync

Two changes to `Qwen2Model.forward` / `_prepare_block_mask_for_inference`:

- The `use_cache=True` branch now dispatches through (a)'s vectorized helpers
  instead of the per-row loop.
- `find_prefix_seq_length_by_pe(position_ids)` was called **unconditionally on
  every forward**, doing a `torch.nonzero(...).item()` per row (`B` syncs) --
  but its result (`x0_len`) is consumed *only* by
  `_prepare_block_mask_for_training`, a training-only path inference never
  reaches. The call moves inside that function, so inference stops computing a
  value it never uses. Pure deletion from the inference hot path; the
  training-only function's behavior is unchanged.

```diff
diff --git a/Embodied/eaglevl/utils/locany/modeling_qwen2.py b/Embodied/eaglevl/utils/locany/modeling_qwen2.py
index 4a61469..afe3eed 100644
--- a/Embodied/eaglevl/utils/locany/modeling_qwen2.py
+++ b/Embodied/eaglevl/utils/locany/modeling_qwen2.py
@@ -75,8 +75,9 @@ QWEN2_PRETRAINED_MODEL_ARCHIVE_LIST = [
 from .mask_sdpa_utils import (
     find_prefix_seq_length_by_pe,
     update_causal_mask_with_pad_non_visible_2d,
-    update_causal_mask_for_one_gen_window_2d,
+    update_causal_mask_for_one_gen_window_4d,
     apply_per_row_generation_window,
+    apply_per_row_generation_window_vectorized,
     create_block_diff_mask_by_pe_4d,
     find_pred_pos_from_input_ids
 )
@@ -1270,8 +1271,6 @@ class Qwen2Model(Qwen2PreTrainedModel):

         device = input_ids.device if input_ids is not None else inputs_embeds.device

-        x0_len = find_prefix_seq_length_by_pe(position_ids).to(device=device)
-
         def _prepare_block_mask_for_inference(attention_mask):
             attention_mask = _prepare_4d_causal_attention_mask(
                 attention_mask,
@@ -1287,31 +1286,42 @@ class Qwen2Model(Qwen2PreTrainedModel):
             if attention_mask is None or len(attention_mask.shape) != 4 or input_ids is None:
                 return attention_mask

-            # For SDLM, the generation window should set to bidirectional attention
+            # For SDLM, the generation window should set to bidirectional attention.
+            # Per-row mode dispatch: only rows whose last (right-aligned) token is a
+            # mask token are in MTP mode and get the bidirectional generation window;
+            # rows ending in a real token stay plain-causal (AR). This makes mixed
+            # AR/MTP batches correct while leaving B==1 unchanged. Shared with the
+            # CPU tests via mask_sdpa_utils so the two cannot drift.
             if use_cache:
-                update_mask_func = partial(
-                    update_causal_mask_for_one_gen_window_2d,
+                # Vectorized path: no `.item()` syncs (the 2D update has no
+                # input_ids dependence, so it's applied to every row via `...`
+                # slicing and the per-row dispatch is a tensor `torch.where`).
+                update_mask_func_4d = partial(
+                    update_causal_mask_for_one_gen_window_4d,
                     block_size=self.block_size,
                     use_cache=use_cache,
                     causal_attn=self.causal_attn,
                 )
+                return apply_per_row_generation_window_vectorized(
+                    attention_mask, input_ids, self.text_mask_token_id, update_mask_func_4d)
             else:
+                # use_cache=False is asserted-unsupported by LocateAnything's
+                # generate() paths; kept on the original per-row implementation.
                 update_mask_func = partial(
                     update_causal_mask_with_pad_non_visible_2d,
                     block_size=self.block_size,
                     text_mask_token_id=self.text_mask_token_id,
                     causal_attn=self.causal_attn,
                 )
-
-            # Per-row mode dispatch: only rows whose last (right-aligned) token is a
-            # mask token are in MTP mode and get the bidirectional generation window;
-            # rows ending in a real token stay plain-causal (AR). This makes mixed
-            # AR/MTP batches correct while leaving B==1 unchanged. Shared with the
-            # CPU tests via mask_sdpa_utils so the two cannot drift.
-            return apply_per_row_generation_window(
-                attention_mask, input_ids, self.text_mask_token_id, update_mask_func)
+                return apply_per_row_generation_window(
+                    attention_mask, input_ids, self.text_mask_token_id, update_mask_func)

         def _prepare_block_mask_for_training():
+            # `x0_len` is only needed for the training block-diff mask; computing
+            # it during inference costs B `.item()` syncs per forward for a value
+            # that is never used (`_prepare_block_mask_for_inference` doesn't
+            # consume it).
+            x0_len = find_prefix_seq_length_by_pe(position_ids).to(device=device)
             block_mask, _ = create_block_diff_mask_by_pe_4d(
                 block_size=self.block_size,
                 x0_len_list=x0_len,
```


### c) `batched_generate.py` -- closed-form Step 1 and Step 5

Both of `batched_generate`'s remaining per-row loops (`DECODING_DEEP_DIVE.md`
Parts B.2 and B.6) are replaced with closed-form tensor expressions, built from
per-row Python ints (`pend_lens[b]`, `win_lens[b]`, `cache_len[b]` -- already
plain ints from `.shape[0]`/dict state, not GPU values, so stacking them into
`[A, 1]` tensors costs nothing):

- **Step 1** (input-block assembly): `cur_input_ids`/`cur_pos`/`cur_real` were
  built by constructing each row's `ids`/`pos` via `cat` then left-padding all
  `A` of them to `W`. Now computed directly as `[A, W]` tensors via
  `torch.where`/`torch.gather` against `c = arange(W)` and each row's
  `left_pad[b]`/`win_start[b]` boundaries. The one remaining per-row op is
  copying each row's ragged `pending[b]` tensor into a `[A, Pmax]` buffer --
  unavoidable, since `pending` is a list of variable-length tensors from Step
  4's per-row sampling (untouched by this PR). The now-unused `_left_pad_rows`
  helper is removed.
- **Step 5** (cache compaction): `keep_idx`/`keep_valid` were built by
  concatenating two `arange` ranges (`old_cols`, `pend_cols`) per row. Each
  row's kept columns are two *contiguous* ranges of the new cache, so the
  source column for destination column `c` is affine in `c` -- one offset for
  the old-cache part, another for the pending part -- computed as `[A, 1]`
  tensors and selected with `torch.where` over `c2 = arange(new_Ckv)`.

Net effect: ~10 small per-row ops x `A` rows (each its own kernel launch)
become ~10 whole-batch `[A, W]`-or-`[A, new_Ckv]` ops, plus the one unavoidable
per-row `pending` placement in Step 1.

```diff
diff --git a/Embodied/eaglevl/utils/locany/batched_generate.py b/Embodied/eaglevl/utils/locany/batched_generate.py
index 1abd134..5c041ae 100644
--- a/Embodied/eaglevl/utils/locany/batched_generate.py
+++ b/Embodied/eaglevl/utils/locany/batched_generate.py
@@ -113,17 +113,6 @@ def sample_row_ar(
 # ---------------------------------------------------------------------------
 # Cache helpers (left-padded rectangular legacy cache).
 # ---------------------------------------------------------------------------
-def _left_pad_rows(rows: List[torch.Tensor], pad_value, width: int, dtype, device):
-    """Stack 1-D rows into ``[B, width]``, left-padding each row with ``pad_value``."""
-    B = len(rows)
-    out = torch.full((B, width), pad_value, dtype=dtype, device=device)
-    for b, r in enumerate(rows):
-        n = r.shape[0]
-        if n:
-            out[b, width - n :] = r
-    return out
-
-
 def compact_cache(
     past_key_values,
     keep_src_index: torch.Tensor,
@@ -275,47 +264,65 @@ def batched_generate(
         A = len(rows)

         # ---- Step 1: assemble the (left-padded) current input block. -------
-        seg_ids: List[torch.Tensor] = []
-        seg_pos: List[torch.Tensor] = []
-        win_lens = {b: 0 for b in rows}  # window length contributed this step
-        pend_lens = {b: 0 for b in rows}
-        for b in rows:
-            if finished[b]:  # only reachable when early_eject is False
-                seg_ids.append(torch.empty(0, dtype=torch.long, device=device))
-                seg_pos.append(torch.empty(0, dtype=torch.long, device=device))
-                continue
-            p = pending[b]
-            pend_lens[b] = p.shape[0]
-            base = cache_len[b]
-            pos = torch.arange(base, base + p.shape[0], device=device)
-            if use_mtp[b]:
-                dup = p[-1:].clone()
-                masks = torch.full(
-                    (n_future - 1,), mask_token_id, dtype=torch.long, device=device
-                )
-                ids = torch.cat([p, dup, masks])
-                # window positions continue then shift back by one (block-diffusion pe)
-                wpos = (
-                    torch.arange(
-                        base + p.shape[0], base + p.shape[0] + n_future, device=device
-                    )
-                    - 1
-                )
-                pos = torch.cat([pos, wpos])
-                win_lens[b] = n_future
-            else:
-                ids = p
-            seg_ids.append(ids)
-            seg_pos.append(pos)
-
-        W = max(s.shape[0] for s in seg_ids)
-        cur_input_ids = _left_pad_rows(seg_ids, pad_token_id, W, torch.long, device)
-        cur_pos = _left_pad_rows(seg_pos, 0, W, torch.long, device)
-        cur_real = torch.zeros((A, W), dtype=torch.long, device=device)
+        # Vectorized: per-row scalars (pend_lens, win_lens, seg_lens, cache_len)
+        # are plain Python ints (from `.shape[0]` and dict state -- no GPU sync),
+        # so cur_input_ids/cur_pos/cur_real are built via broadcasted arange +
+        # gather + torch.where over the whole [A, W] block in O(1) ops. The one
+        # remaining per-row GPU op is placing each row's ragged `pending` tensor
+        # into a left-padded `pend_pad`, since `pending` is a Python list of
+        # variable-length tensors produced by Step 4's per-row sampling.
+        win_lens = {
+            b: (n_future if (use_mtp[b] and not finished[b]) else 0) for b in rows
+        }
+        pend_lens = {b: (0 if finished[b] else pending[b].shape[0]) for b in rows}
+
+        pend_lens_l = [pend_lens[b] for b in rows]
+        win_lens_l = [win_lens[b] for b in rows]
+        seg_lens_l = [pend_lens_l[j] + win_lens_l[j] for j in range(A)]
+        cache_len_l = [cache_len[b] for b in rows]
+
+        Pmax = max(pend_lens_l)
+        W = max(seg_lens_l)
+
+        pend_pad = torch.full((A, Pmax), pad_token_id, dtype=torch.long, device=device)
         for j, b in enumerate(rows):
-            n = seg_ids[j].shape[0]
+            n = pend_lens_l[j]
             if n:
-                cur_real[j, W - n :] = 1
+                pend_pad[j, Pmax - n :] = pending[b]
+
+        pend_lens_t = torch.tensor(pend_lens_l, dtype=torch.long, device=device).unsqueeze(1)
+        win_lens_t = torch.tensor(win_lens_l, dtype=torch.long, device=device).unsqueeze(1)
+        seg_lens_t = torch.tensor(seg_lens_l, dtype=torch.long, device=device).unsqueeze(1)
+        cache_len_t = torch.tensor(cache_len_l, dtype=torch.long, device=device).unsqueeze(1)
+
+        c = torch.arange(W, device=device).unsqueeze(0)  # [1, W]
+        left_pad = W - seg_lens_t  # [A, 1]: first real column of this row
+        win_start = W - win_lens_t  # [A, 1]: first MTP-window column (== W if AR)
+
+        cur_real = (c >= left_pad).long()
+        in_window = c >= win_start
+
+        # pend part: gather row j's `pending` content (right-aligned in pend_pad)
+        # into columns [left_pad, win_start).
+        pend_gather_idx = (c + (Pmax - W) + win_lens_t).clamp(0, Pmax - 1)
+        pend_vals = torch.gather(pend_pad, 1, pend_gather_idx)
+
+        # MTP window part: [dup, mask, mask, ..., mask] in columns [win_start, W).
+        dup = pend_pad[:, -1:]
+        mask_fill = torch.full((A, W), mask_token_id, dtype=torch.long, device=device)
+        window_vals = torch.where(c - win_start == 0, dup, mask_fill)
+
+        cur_input_ids = torch.where(in_window, window_vals, pend_vals)
+        cur_input_ids = torch.where(
+            cur_real.bool(), cur_input_ids, torch.full_like(cur_input_ids, pad_token_id)
+        )
+
+        # positions: pend part continues from cache_len; window positions shift
+        # back by one (block-diffusion pe), matching the original per-row arange.
+        pos_pend = cache_len_t + (c - left_pad)
+        pos_win = cache_len_t + pend_lens_t + (c - win_start) - 1
+        cur_pos = torch.where(in_window, pos_win, pos_pend)
+        cur_pos = torch.where(cur_real.bool(), cur_pos, torch.zeros_like(cur_pos))

         # ---- Step 2: cache validity mask + full 2D attention mask. ----------
         if Ckv > 0:
@@ -391,27 +398,24 @@ def batched_generate(
         if new_Ckv == 0:
             past_key_values, Ckv = None, 0
         else:
-            keep_idx = torch.zeros((A, new_Ckv), dtype=torch.long, device=device)
-            keep_valid = torch.zeros((A, new_Ckv), dtype=torch.bool, device=device)
-            for j, b in enumerate(rows):
-                # old valid cache columns (right-aligned in [0, Ckv))
-                old_cols = (
-                    torch.arange(Ckv - cache_len[b], Ckv, device=device)
-                    if cache_len[b]
-                    else torch.empty(0, dtype=torch.long, device=device)
-                )
-                # pending columns inside the current block: real tokens excluding
-                # the (rightmost) window; they sit just before the window.
-                blk_real = pend_lens[b] + win_lens[b]  # real cols in this block
-                pend_start = Ckv + (W - blk_real)  # first pending col (abs)
-                pend_cols = torch.arange(
-                    pend_start, pend_start + pend_lens[b], device=device
-                )
-                src = torch.cat([old_cols, pend_cols])
-                k = src.shape[0]
-                if k:
-                    keep_idx[j, new_Ckv - k :] = src
-                    keep_valid[j, new_Ckv - k :] = True
+            # Vectorized closed form of the per-row index build above. Each row's
+            # kept columns are `[old_cols, pend_cols]`, both contiguous ranges, so
+            # the source index for kept column `c` (c in [new_Ckv-k, new_Ckv), k =
+            # cache_len[b]+pend_lens[b] = new_cache_len_rows[b]) is affine in `c`:
+            # `c + offset_old` while `c < new_Ckv - pend_lens[b]` (the old-cache
+            # part) and `c + offset_pend` after (the pending part), with
+            # `offset_old = Ckv + pend_lens[b] - new_Ckv` and
+            # `offset_pend = Ckv + W - win_lens[b] - new_Ckv`.
+            k_t = cache_len_t + pend_lens_t  # [A,1] == new_cache_len_rows per row
+            offset_old_t = Ckv + pend_lens_t - new_Ckv
+            offset_pend_t = Ckv + W - win_lens_t - new_Ckv
+
+            c2 = torch.arange(new_Ckv, device=device).unsqueeze(0)  # [1, new_Ckv]
+            use_pend = c2 >= (new_Ckv - pend_lens_t)
+            offset = torch.where(use_pend, offset_pend_t, offset_old_t)
+            keep_valid = c2 >= (new_Ckv - k_t)
+            keep_idx = torch.where(keep_valid, c2 + offset, torch.zeros_like(c2))
+
             past_key_values = compact_cache(new_past, keep_idx, keep_valid)
             Ckv = new_Ckv
         for b in rows:
```

### d) `generate_utils.py` + `batched_generate.py` — vectorize Step-4 sampling (PR3 "B")

**Motivation.** Step 4 called `sample_tokens` once **per active row** (`batch_size=1`),
so the V≈152k softmax / top-p / repetition-penalty ran as A separate GPU kernels every
decode step. Group the active rows by mode and run **one** `sample_tokens_batched` call
per mode; the ragged box decode + structural `handle_pattern` stay per row.

```diff
-        # ---- Step 4: per-row sampling + state update. -----------------------
-        for j, b in enumerate(rows):
-            if finished[b]:
-                continue
-            if use_mtp[b]:
-                out_type, out_token = sample_row_mtp(logits[j:j+1, -n_future:, :], ...)
-            else:
-                out_type, out_token = sample_row_ar(logits[j:j+1, -1:, :], ...)
+        # ---- Step 4: batched per-mode sampling + per-row state update. ------
+        active   = [(j, b) for j, b in enumerate(rows) if not finished[b]]
+        mtp_rows = [(j, b) for j, b in active if use_mtp[b]]
+        ar_rows  = [(j, b) for j, b in active if not use_mtp[b]]
+        decoded = {}
+        if mtp_rows:                                   # ONE V-wide kernel for all MTP rows
+            res = sample_group_mtp(logits[[j for j,_ in mtp_rows]][:, -n_future:, :],
+                                   _pad_histories([full_history[b] for _,b in mtp_rows], device), ...)
+            for (_, b), r in zip(mtp_rows, res): decoded[b] = r
+        if ar_rows:                                    # ONE V-wide kernel for all AR rows
+            res = sample_group_ar(logits[[j for j,_ in ar_rows]][:, -1:, :], ...)
+            for (_, b), r in zip(ar_rows, res): decoded[b] = r
```

`sample_tokens_batched` (new in `generate_utils.py`) mirrors `sample_tokens` exactly but
returns `box_avg` as a per-row **list** (the per-row decoded boxes are ragged and can't be
stacked); histories are right-padded with `-1`, which `apply_repetition_penalty` filters as
out-of-vocab so the pad never enters any row's penalty mask.

**How tested.** Numerically identical to the per-row path: the existing CPU suite stays
**432 / 432 / 432** because `batched_generate` (now using `sample_group_*`) is compared
against the per-row `sample_row_*` reference the test still imports.

**Speed.** `step4_sample` drops ~30% (bs16 hybrid 0.285s → 0.193s). But Step 4 is only
~3% of the decode loop, so the **end-to-end gain is ~1% — marginal on its own.** Its real
value is that it removes per-row host work on the sync-free decode path (alongside Stage A/B),
which is what lets the decoder `torch.compile` cleanly.

### e) `modeling_vit.py` — remove ViT per-image host syncs

**Motivation.** `sdpa_attention` sliced each packed image with
`int(q_cu_seqlens[i-1]), int(q_cu_seqlens[i])` — **2 tensor→host syncs per image, on every
ViT layer of every forward** — serializing the GPU pipeline. The ViT analogue of PR3's
decoder sync removal, and it speeds the **eager** vision path (what shipped batched
inference runs, not just the `+vit` compile experiment).

```diff
     seq_length = q.shape[0]
     outs = []
-    for i in range(1, len(q_cu_seqlens)):
-        s, e = int(q_cu_seqlens[i - 1]), int(q_cu_seqlens[i])
+    cu = q_cu_seqlens.tolist()          # one host transfer, not 2x per image per layer
+    for i in range(1, len(cu)):
+        s, e = cu[i - 1], cu[i]
         qi = q[s:e].transpose(0, 1)
```

**How tested.** `tests/test_modeling_vit.py`: (1) the `.tolist()` output is `torch.equal`
to the original per-element implementation across 5 ragged packings; (2) `sdpa_attention`
matches an independent dense block-diagonal `-inf`-mask reference (`allclose`). The
`batched_generate` suite still passes 432/432/432.

**Speed.** **2.79× on the attention** in a 27-layer-forward microbench (MPS, 105.8 → 37.9 ms,
~405 fewer host syncs) — bit-identical *and* significantly faster. Caveats: attention in
isolation, on MPS where syncs are expensive; the end-to-end vision-encode gain is a few %
and CUDA syncs are cheaper, so the CUDA number is more modest. Still a free eager win.

---

## Speed attribution per change (honest)

Each change is held to: **bit-identical / detections-within-tolerance AND a measured speed
justification.** Where a change is ~0 end-to-end, it is labelled as *enabling* (compile- or
memory-enabling) rather than a direct win — surfaced deliberately.

| change | equivalence | measured speed | verdict |
|---|---|---|---|
| **PR1 batching** (driver) | CPU 432×3 + GPU semantic parity ~45/50 | **~2.5× vs B=1** (real workload, vs pristine); 2.6–3.25× on eval | direct win — the big lever |
| **PR1 ViT OOM fix** (per-image self-attn) | identical numerics (max abs diff 3.6e-7) | **memory-enabling** — multi-image batch O(Σnᵢ²) not O((Σnᵢ)²); without it batched inference OOMs | enabler (no speed claim) |
| **PR2 grouping** (encode once) | CPU 432×3; same detections | **~1.9× on RefCOCOg** one-image-many-queries (vision N→1) | direct win (that workload) |
| **PR3 Stage A/B** (vectorized bookkeeping) | CPU 432×3 + 1800 mask-equality | step1 −53%, step5 −14%, but bookkeeping ~3–4% of decode → **~0 end-to-end** | compile-enabling (removes per-step `.item()` syncs) |
| **PR3 B** (vectorized sampling) | CPU 432×3 | step4 −30% → **~1% end-to-end** | marginal; part of the sync-free compile path |
| **PR3 ViT sync removal** | `test_modeling_vit` bit-identical | **2.79× attention** (MPS microbench); few-% eager vision | direct eager win (magnitude hardware-dependent) |
| **compile** (decoder, `torch.compile`) | mAP gate: ΔAP ±1% on COCO+LVIS | **~1.57–1.61× forward → ~1.6× decode-bound wall** | direct win (correctness-gated) |
| **stacked total** | — | **~4× vs the unmodified model** (batching × compile, single measured ratio) | the headline |

**Reading this honestly:** the real levers are **batching**, **PR2 grouping** (its
workload), the **ViT sync removal**, and **compile**. The PR3 decode/sampling vectorization
(Stage A/B + B) is **~0 end-to-end** — clean and equivalent, and it earns its place by
making the decoder sync-free so `torch.compile` sticks, but it is *not* a direct speedup and
is labelled as such.
