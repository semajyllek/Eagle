# --------------------------------------------------------
# NVIDIA
# Copyright (c) 2025 NVIDIA
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
"""
Batched generation for LocateAnything (Strategy C: true batched hybrid/MTP decode).

The single-sequence ``generate()`` in ``modeling_locateanything.py`` runs a
non-standard decoder: a Multi-Token-Prediction (MTP) block-diffusion window with
an auto-regressive (AR) fallback, plus per-step truncate-and-recompute of the KV
cache. That scheme is inherently per-sequence (variable #accepted tokens per
step, per-sequence mode switching, per-sequence termination), which is why the
original code asserts ``batch_size == 1``.

This module lifts that restriction. Every per-sequence scalar becomes a per-row
vector and the cache is kept as a single **left-padded** rectangular tensor that
is *compacted* after every step so its width tracks the longest committed
sequence in the batch (rather than growing by the speculative window each step).

The numerically heavy pieces are reused unchanged from ``generate_utils``
(``sample_tokens``, ``decode_bbox_avg``, ``decode_ref``, ``handle_pattern``);
they already operate one row at a time. The attention masking is delegated to the
(now per-row mode-aware) ``Qwen2Model._prepare_block_mask_for_inference`` so the
validated block-diffusion mask logic is shared with the B==1 path.
"""

import time
from typing import Dict, List, Optional

import torch

from .generate_utils import (handle_pattern, sample_tokens,
                             sample_tokens_batched)


# ---------------------------------------------------------------------------
# Optional per-step profiling. ``profile`` is a caller-owned dict; when ``None``
# (the default), ``_tic``/``_toc`` are no-ops so there is zero overhead.
# ---------------------------------------------------------------------------
def _sync(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def _tic(profile, device):
    if profile is None:
        return None
    _sync(device)
    return time.perf_counter()


def _toc(profile, key, t0, device):
    if t0 is None:
        return
    _sync(device)
    profile[key] = profile.get(key, 0.0) + (time.perf_counter() - t0)


# ---------------------------------------------------------------------------
# Runaway-decode guard.
#
# Greedy decoding can occasionally enter a self-reinforcing "keep scanning
# this region" loop: each tiny <box> makes "emit another similar box here"
# the highest-probability continuation, so the row keeps emitting many
# structurally-identical adjacent <box> detections whose coordinates drift by
# only a few units each time, instead of moving on to <|im_end|>. This is a
# property of the model/greedy-decoding (not specific to batching) that
# otherwise runs the row all the
# way to `max_new_tokens`. Detect a run of `run_len` consecutive <box>
# segments whose per-coordinate deltas are all <= `max_delta` (out of the
# model's 0..1000 coordinate range) and stop the row there, same as hitting
# <|im_end|>.
#
# `run_len`/`max_delta` are tuned for the "scanning loop" pathology (observed
# deltas of 3-7 over 11+ boxes); datasets with legitimate dense, evenly-spaced
# detections (e.g. scene-text/OCR, where each character/word is its own <box>
# with a small consistent offset) may need a larger `run_len` and/or smaller
# `max_delta` to avoid cutting off real output. Pass
# `generate_kwargs["runaway_box_run"]` / `generate_kwargs["runaway_box_max_delta"]`
# (threaded through `predict_batch`) to override; `runaway_box_run<=0` disables
# the guard entirely.
# ---------------------------------------------------------------------------
_RUNAWAY_BOX_RUN = 8
_RUNAWAY_BOX_MAX_DELTA = 8


def _box_coords(tokens: List[int], i: int, token_ids: Dict[str, int]):
    """If ``tokens[i:i+6]`` is a complete ``<box>c1c2c3c4</box>`` segment,
    return its 4 coordinates (0..1000); else ``None``."""
    if i < 0 or i + 6 > len(tokens):
        return None
    if tokens[i] != token_ids["box_start_token_id"]:
        return None
    if tokens[i + 5] != token_ids["box_end_token_id"]:
        return None
    cs, ce = token_ids["coord_start_token_id"], token_ids["coord_end_token_id"]
    coords = tokens[i + 1 : i + 5]
    if not all(cs <= t <= ce for t in coords):
        return None
    return [t - cs for t in coords]


def _is_runaway_box_loop(
    gen_tokens_b: List[int],
    token_ids: Dict[str, int],
    run_len: int = _RUNAWAY_BOX_RUN,
    max_delta: int = _RUNAWAY_BOX_MAX_DELTA,
) -> bool:
    """True if the tail of ``gen_tokens_b`` is ``run_len`` consecutive ``<box>``
    segments each drifting by at most ``max_delta`` from the previous one.
    ``run_len <= 0`` disables the check (always returns ``False``)."""
    if run_len <= 0:
        return False
    n = len(gen_tokens_b)
    seg = 6
    if n < run_len * seg:
        return False
    prev = None
    for k in range(run_len):
        coords = _box_coords(gen_tokens_b, n - (k + 1) * seg, token_ids)
        if coords is None:
            return False
        if (
            prev is not None
            and max(abs(a - b) for a, b in zip(coords, prev)) > max_delta
        ):
            return False
        prev = coords
    return True


# ---------------------------------------------------------------------------
# Per-row sampling (faithful copies of the closures in generate().)
# ---------------------------------------------------------------------------
def sample_row_mtp(
    logits_row: torch.Tensor,
    generated_row: torch.Tensor,
    token_ids: Dict[str, int],
    n_future: int,
    generation_mode: str,
    generate_kwargs: dict,
):
    """MTP sampling for a single row.

    Args:
        logits_row: ``[1, n_future, V]`` logits of the speculative window.
        generated_row: ``[1, L]`` full token history (prompt + generated) for the
            repetition penalty.
    Returns:
        ``(out_type, out_token)`` where ``out_token`` is a 1-D LongTensor.
    """
    # `generation_mode` is carried inside `generate_kwargs` (matching the B==1
    # path), so it is not passed explicitly here.
    probs, confidence, x0, box_avg = sample_tokens(
        logits_row,
        generated_row,
        token_ids,
        keep_k=5,
        **generate_kwargs,
    )
    is_box_empty = (box_avg[0] == 0).all()
    new_tokens = x0[0] if is_box_empty else box_avg[0]

    out_pattern = handle_pattern(new_tokens, token_ids, generation_mode)
    out_token = torch.tensor(out_pattern["tokens"], dtype=x0.dtype, device=x0.device)
    return out_pattern["type"], out_token


def sample_row_ar(
    logits_row: torch.Tensor,
    generated_row: torch.Tensor,
    token_ids: Dict[str, int],
    generation_mode: str,
    generate_kwargs: dict,
):
    """AR (single-token) sampling for a single row. Mirrors ``_sample_token_in_ar``."""
    probs, confidence, x0, _ = sample_tokens(
        logits_row,
        generated_row,
        token_ids,
        **generate_kwargs,
    )
    out_token = x0[0]
    out_type = "continue_ar"
    token_val = out_token[0].item()

    box_end_token_id = token_ids["box_end_token_id"]
    coord_start_token_id = token_ids["coord_start_token_id"]
    coord_end_token_id = token_ids["coord_end_token_id"]
    none_token_id = token_ids["none_token_id"]
    im_end_token_id = token_ids["im_end_token_id"]

    if generation_mode == "hybrid":
        if token_val == box_end_token_id:
            out_type = "box_end_ar"
        elif (
            coord_start_token_id <= token_val <= coord_end_token_id
            or token_val == none_token_id
        ):
            out_type = "coord_ar"
        else:
            out_type = "im_end"
    else:
        if token_val == im_end_token_id:
            out_type = "im_end"

    return out_type, out_token


# ---------------------------------------------------------------------------
# Batched per-mode sampling: one V-wide softmax/top-p/rep-penalty kernel across
# all same-mode rows (vs one per row in sample_row_*), then the cheap ragged box
# decode + structural handling stays per row. Numerically identical to running
# sample_row_* per row (sample_tokens_batched is row-independent; -1-padded
# histories are filtered by apply_repetition_penalty) -- the CPU parity test
# compares batched_generate (these) against the per-row sample_row_* reference.
# ---------------------------------------------------------------------------
def _pad_histories(hist_list: List[torch.Tensor], device, pad_value: int = -1):
    """Stack ragged token histories into ``[B, Lmax]``, right-padded with -1.
    ``apply_repetition_penalty`` filters out-of-vocab ids, so the pad never enters
    any row's penalty mask -- identical to each row's real history."""
    Lmax = max(h.shape[0] for h in hist_list)
    out = torch.full((len(hist_list), Lmax), pad_value, dtype=torch.long, device=device)
    for i, h in enumerate(hist_list):
        out[i, : h.shape[0]] = h
    return out


def sample_group_mtp(logits, generated, token_ids, generation_mode, generate_kwargs):
    """Batched MTP sampling over ``[G, n_future, V]``; returns a list of
    ``(out_type, out_token)`` aligned to the group rows (mirrors sample_row_mtp)."""
    probs, confidence, x0, box_avg = sample_tokens_batched(
        logits, generated, token_ids, keep_k=5, **generate_kwargs
    )
    out = []
    for i in range(x0.shape[0]):
        is_box_empty = (box_avg[i] == 0).all()
        new_tokens = x0[i] if is_box_empty else box_avg[i]
        out_pattern = handle_pattern(new_tokens, token_ids, generation_mode)
        out_token = torch.tensor(
            out_pattern["tokens"], dtype=x0.dtype, device=x0.device
        )
        out.append((out_pattern["type"], out_token))
    return out


def sample_group_ar(logits, generated, token_ids, generation_mode, generate_kwargs):
    """Batched AR sampling over ``[G, 1, V]``; returns a list of
    ``(out_type, out_token)`` aligned to the group rows (mirrors sample_row_ar)."""
    probs, confidence, x0, _ = sample_tokens_batched(
        logits, generated, token_ids, **generate_kwargs
    )
    box_end_token_id = token_ids["box_end_token_id"]
    coord_start_token_id = token_ids["coord_start_token_id"]
    coord_end_token_id = token_ids["coord_end_token_id"]
    none_token_id = token_ids["none_token_id"]
    im_end_token_id = token_ids["im_end_token_id"]

    out = []
    for i in range(x0.shape[0]):
        out_token = x0[i]
        token_val = out_token[0].item()
        out_type = "continue_ar"
        if generation_mode == "hybrid":
            if token_val == box_end_token_id:
                out_type = "box_end_ar"
            elif (
                coord_start_token_id <= token_val <= coord_end_token_id
                or token_val == none_token_id
            ):
                out_type = "coord_ar"
            else:
                out_type = "im_end"
        else:
            if token_val == im_end_token_id:
                out_type = "im_end"
        out.append((out_type, out_token))
    return out


# ---------------------------------------------------------------------------
# Cache helpers (left-padded rectangular legacy cache).
# ---------------------------------------------------------------------------
def compact_cache(
    past_key_values,
    keep_src_index: torch.Tensor,
    keep_valid: torch.Tensor,
):
    """Gather a left-padded cache from an existing (larger) cache.

    Args:
        past_key_values: legacy tuple of ``(k, v)`` per layer, each
            ``[B, H, T_old, D]``.
        keep_src_index: ``[B, T_new]`` long tensor of source column indices into
            ``T_old`` (right-aligned; left padding entries may be any valid index
            and are zeroed via ``keep_valid``).
        keep_valid: ``[B, T_new]`` bool tensor; ``False`` marks left-pad columns.
    Returns:
        New legacy tuple with each tensor shaped ``[B, H, T_new, D]``.
    """
    B, T_new = keep_src_index.shape
    valid = keep_valid[:, None, :, None]  # [B,1,T_new,1]
    new_past = []
    for k, v in past_key_values:
        H, D = k.shape[1], k.shape[3]
        idx = keep_src_index[:, None, :, None].expand(B, H, T_new, D)
        nk = torch.gather(k, 2, idx) * valid
        nv = torch.gather(v, 2, idx) * valid
        new_past.append((nk, nv))
    return tuple(new_past)


# ---------------------------------------------------------------------------
# Main batched driver.
# ---------------------------------------------------------------------------
@torch.no_grad()
def batched_generate(
    model,
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.LongTensor],
    vit_embeds: Optional[torch.Tensor],
    image_token_index: Optional[int],
    tokenizer,
    n_future: int,
    pad_token_id: int,
    generate_kwargs: dict,
):
    """Batched hybrid/MTP/AR decoding.

    Args:
        model: ``LocateAnythingForConditionalGeneration`` (provides
            ``language_model`` and ``token_ids``).
        input_ids: ``[B, P]`` **left-padded** prompt ids (image placeholders
            included). Left padding must use ``pad_token_id``.
        attention_mask: ``[B, P]`` 1/0 mask for the prompts (1 == real token).
        vit_embeds: projected visual features, concatenated across the batch in
            row-major order so they align with the flattened image-token slots of
            ``input_ids``. May be ``None`` for text-only batches.
        image_token_index: id of the image placeholder token.
        tokenizer: used only to decode the final strings.
        n_future: MTP window width (== model block size).
        pad_token_id: id used for left padding (must differ from
            ``image_token_index`` and the mask token).
        generate_kwargs: forwarded to the samplers; supports
            ``generation_mode`` ('fast'|'slow'|'hybrid'), ``max_new_tokens``,
            ``temperature``, ``top_p``, ``top_k``, ``repetition_penalty``.
            Also accepts ``runaway_box_run``/``runaway_box_max_delta`` (see the
            runaway-decode guard above) to tune or disable (``runaway_box_run<=0``)
            the early-stop for degenerate repeated-<box> output.
    Returns:
        ``List[str]`` of length B with the decoded responses (special tokens
        kept, matching the B==1 path).
    """
    lm = model.language_model
    token_ids = model.token_ids
    device = input_ids.device

    # Optional per-step timing breakdown; see ``_tic``/``_toc`` above.
    profile = generate_kwargs.pop("profile", None)

    # Runaway-decode guard tuning; see ``_is_runaway_box_loop`` above.
    runaway_box_run = generate_kwargs.pop("runaway_box_run", None)
    if runaway_box_run is None:
        runaway_box_run = _RUNAWAY_BOX_RUN
    runaway_box_max_delta = generate_kwargs.pop("runaway_box_max_delta", None)
    if runaway_box_max_delta is None:
        runaway_box_max_delta = _RUNAWAY_BOX_MAX_DELTA

    generation_mode = generate_kwargs.get("generation_mode", "hybrid")
    assert generation_mode in (
        "fast",
        "slow",
        "hybrid",
    ), f"Unsupported generation_mode='{generation_mode}'."
    mask_token_id = token_ids["default_mask_token_id"]

    B, P = input_ids.shape
    prompt_lens = (
        attention_mask.sum(dim=1).tolist() if attention_mask is not None else [P] * B
    )
    max_new_tokens = generate_kwargs.get("max_new_tokens", 2048)
    model_max = getattr(tokenizer, "model_max_length", 10**9)

    start_mtp = generation_mode in ("fast", "hybrid")

    # Per-row state.
    gen_tokens: List[List[int]] = [[] for _ in range(B)]  # generated (excl. prompt)
    full_history: List[torch.Tensor] = [
        input_ids[b, P - prompt_lens[b] :].clone() for b in range(B)
    ]
    use_mtp = [start_mtp] * B
    finished = [False] * B
    # `pending` = real tokens to confirm into the cache this step. Step 0 = prompt.
    pending: List[torch.Tensor] = [
        input_ids[b, P - prompt_lens[b] :].clone() for b in range(B)
    ]
    cache_len = [0] * B  # committed tokens already in cache
    max_len = [min(model_max, prompt_lens[b] + max_new_tokens) for b in range(B)]

    past_key_values = None
    Ckv = 0
    first_step = True
    cache_rows = list(range(B))

    while not all(finished):
        rows = cache_rows
        A = len(rows)
        if profile is not None:
            profile["n_steps"] = profile.get("n_steps", 0) + 1
            profile.setdefault("A_history", []).append(A)

        # ---- Step 1: assemble the (left-padded) current input block. -------
        t0 = _tic(profile, device)
        # Vectorized: per-row scalars (pend_lens, win_lens, seg_lens, cache_len)
        # are plain Python ints from `.shape[0]` and dict state -- no GPU sync --
        # so cur_input_ids/cur_pos/cur_real are built via broadcasted arange +
        # gather + torch.where over the whole [A, W] block in O(1) ops. The one
        # remaining per-row GPU op is placing each row's ragged `pending` tensor
        # into a left-padded `pend_pad`, since `pending` is a Python list of
        # variable-length tensors produced by Step 4's per-row sampling.
        win_lens = {
            b: (n_future if (use_mtp[b] and not finished[b]) else 0) for b in rows
        }
        pend_lens = {b: (0 if finished[b] else pending[b].shape[0]) for b in rows}

        pend_lens_l = [pend_lens[b] for b in rows]
        win_lens_l = [win_lens[b] for b in rows]
        seg_lens_l = [pend_lens_l[j] + win_lens_l[j] for j in range(A)]
        cache_len_l = [cache_len[b] for b in rows]

        Pmax = max(pend_lens_l)
        W = max(seg_lens_l)

        pend_pad = torch.full((A, Pmax), pad_token_id, dtype=torch.long, device=device)
        for j, b in enumerate(rows):
            n = pend_lens_l[j]
            if n:
                pend_pad[j, Pmax - n :] = pending[b]

        pend_lens_t = torch.tensor(
            pend_lens_l, dtype=torch.long, device=device
        ).unsqueeze(1)
        win_lens_t = torch.tensor(
            win_lens_l, dtype=torch.long, device=device
        ).unsqueeze(1)
        seg_lens_t = torch.tensor(
            seg_lens_l, dtype=torch.long, device=device
        ).unsqueeze(1)
        cache_len_t = torch.tensor(
            cache_len_l, dtype=torch.long, device=device
        ).unsqueeze(1)

        c = torch.arange(W, device=device).unsqueeze(0)  # [1, W]
        left_pad = W - seg_lens_t  # [A, 1]: first real column of this row
        win_start = W - win_lens_t  # [A, 1]: first MTP-window column (== W if AR)

        cur_real = (c >= left_pad).long()
        in_window = c >= win_start

        # pend part: gather row j's `pending` content (right-aligned in pend_pad)
        # into columns [left_pad, win_start).
        pend_gather_idx = (c + (Pmax - W) + win_lens_t).clamp(0, Pmax - 1)
        pend_vals = torch.gather(pend_pad, 1, pend_gather_idx)

        # MTP window part: [dup, mask, mask, ..., mask] in columns [win_start, W).
        dup = pend_pad[:, -1:]
        mask_fill = torch.full((A, W), mask_token_id, dtype=torch.long, device=device)
        window_vals = torch.where(c - win_start == 0, dup, mask_fill)

        cur_input_ids = torch.where(in_window, window_vals, pend_vals)
        cur_input_ids = torch.where(
            cur_real.bool(), cur_input_ids, torch.full_like(cur_input_ids, pad_token_id)
        )

        # positions: pend part continues from cache_len; window positions shift
        # back by one (block-diffusion pe), matching the original per-row arange.
        pos_pend = cache_len_t + (c - left_pad)
        pos_win = cache_len_t + pend_lens_t + (c - win_start) - 1
        cur_pos = torch.where(in_window, pos_win, pos_pend)
        cur_pos = torch.where(cur_real.bool(), cur_pos, torch.zeros_like(cur_pos))
        _toc(profile, "step1_assemble", t0, device)

        # ---- Step 2: cache validity mask + full 2D attention mask. ----------
        t0 = _tic(profile, device)
        if Ckv > 0:
            cache_mask = torch.zeros((A, Ckv), dtype=torch.long, device=device)
            for j, b in enumerate(rows):
                if cache_len[b]:
                    cache_mask[j, Ckv - cache_len[b] :] = 1
            full_attn = torch.cat([cache_mask, cur_real], dim=1)
        else:
            full_attn = cur_real
        _toc(profile, "step2_mask", t0, device)

        # ---- Step 3: model forward. ----------------------------------------
        t0 = _tic(profile, device)
        model_kwargs = dict(
            input_ids=cur_input_ids,
            attention_mask=full_attn,
            position_ids=cur_pos,
            past_key_values=past_key_values,
            use_cache=True,
        )
        if first_step and vit_embeds is not None:
            model_kwargs["visual_features"] = vit_embeds
            model_kwargs["image_token_index"] = image_token_index
        outputs = lm(**model_kwargs)
        logits = outputs.logits  # [A, W, V]
        new_past = outputs.past_key_values  # width Ckv + W
        first_step = False
        _toc(profile, "step3_forward", t0, device)

        # ---- Step 4: batched per-mode sampling + per-row state update. ------
        # Group active rows by mode and run ONE sample_tokens_batched call per
        # mode (batching the V-wide softmax/top-p/rep-penalty that used to be one
        # kernel per row); the ragged box decode + structural handling stay per
        # row inside sample_group_*. Equivalent to the old per-row sample_row_*
        # (validated by the CPU parity test against that reference).
        t0 = _tic(profile, device)
        active = [(j, b) for j, b in enumerate(rows) if not finished[b]]
        mtp_rows = [(j, b) for j, b in active if use_mtp[b]]
        ar_rows = [(j, b) for j, b in active if not use_mtp[b]]
        decoded = {}  # b -> (out_type, out_token)
        if mtp_rows:
            mj = [j for j, _ in mtp_rows]
            res = sample_group_mtp(
                logits[mj][:, -n_future:, :],
                _pad_histories([full_history[b] for _, b in mtp_rows], device),
                token_ids,
                generation_mode,
                generate_kwargs,
            )
            for (_, b), r in zip(mtp_rows, res):
                decoded[b] = r
        if ar_rows:
            aj = [j for j, _ in ar_rows]
            res = sample_group_ar(
                logits[aj][:, -1:, :],
                _pad_histories([full_history[b] for _, b in ar_rows], device),
                token_ids,
                generation_mode,
                generate_kwargs,
            )
            for (_, b), r in zip(ar_rows, res):
                decoded[b] = r

        for j, b in active:
            out_type, out_token = decoded[b]
            gen_tokens[b].extend(out_token.tolist())
            full_history[b] = torch.cat([full_history[b], out_token])

            if out_type == "im_end":
                finished[b] = True
            elif generation_mode == "hybrid":
                if out_type == "error_box":
                    use_mtp[b] = False
                elif out_type == "box_end_ar":
                    use_mtp[b] = True

            # length-based termination
            if len(gen_tokens[b]) + prompt_lens[b] >= max_len[b]:
                finished[b] = True

            # runaway "keep scanning this region" loop (see _is_runaway_box_loop)
            if not finished[b] and _is_runaway_box_loop(
                gen_tokens[b], token_ids, runaway_box_run, runaway_box_max_delta
            ):
                finished[b] = True

            # next step confirms the tokens we just accepted
            pending[b] = out_token
        _toc(profile, "step4_sample", t0, device)

        # ---- Step 5: compact the cache (keep committed prefix + just-
        #             confirmed pending columns; drop window + pad). ----------
        t0 = _tic(profile, device)
        new_cache_len_rows = {b: cache_len[b] + pend_lens[b] for b in rows}
        new_Ckv = max(new_cache_len_rows.values()) if rows else 0
        if new_Ckv == 0:
            past_key_values, Ckv = None, 0
        else:
            # Vectorized closed form of the per-row index build above. Each row's
            # kept columns are `[old_cols, pend_cols]`, both contiguous ranges, so
            # the source index for kept column `c` (c in [new_Ckv-k, new_Ckv), k =
            # cache_len[b]+pend_lens[b] = new_cache_len_rows[b]) is affine in `c`:
            # `c + offset_old` while `c < new_Ckv - pend_lens[b]` (the old-cache
            # part) and `c + offset_pend` after (the pending part), with
            # `offset_old = Ckv + pend_lens[b] - new_Ckv` and
            # `offset_pend = Ckv + W - win_lens[b] - new_Ckv`.
            k_t = cache_len_t + pend_lens_t  # [A,1] == new_cache_len_rows per row
            offset_old_t = Ckv + pend_lens_t - new_Ckv
            offset_pend_t = Ckv + W - win_lens_t - new_Ckv

            c2 = torch.arange(new_Ckv, device=device).unsqueeze(0)  # [1, new_Ckv]
            use_pend = c2 >= (new_Ckv - pend_lens_t)
            offset = torch.where(use_pend, offset_pend_t, offset_old_t)
            keep_valid = c2 >= (new_Ckv - k_t)
            keep_idx = torch.where(keep_valid, c2 + offset, torch.zeros_like(c2))

            past_key_values = compact_cache(new_past, keep_idx, keep_valid)
            Ckv = new_Ckv
        for b in rows:
            cache_len[b] = new_cache_len_rows[b]
        _toc(profile, "step5_compact", t0, device)

    return [
        tokenizer.decode(torch.tensor(g), skip_special_tokens=False) for g in gen_tokens
    ]
