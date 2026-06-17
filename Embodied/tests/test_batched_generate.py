# --------------------------------------------------------
# NVIDIA
# Copyright (c) 2025 NVIDIA
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
"""
CPU correctness tests for the batched MTP/AR/hybrid decoder.

These tests do NOT need the real checkpoint, GPU, or transformers. They drive the
batched driver (``batched_generate``) with a small, fully-deterministic ``FakeLM``
that honours the exact masking / KV-cache / position contract of the real
``Qwen2Model`` (it reuses the real block-diffusion window mask from
``mask_sdpa_utils``). Correctness is then checked two ways:

  1. Algorithm parity  - ``batched_generate`` on a single row must reproduce an
     independent re-implementation of the original truncate-and-recompute
     single-sequence loop (``reference_single``).
  2. Batch independence - a row's output must be identical whether it is decoded
     alone or inside a padded batch with other (shorter/longer) rows.

If both hold for many random seeds across fast/slow/hybrid modes, the batched
bookkeeping (left-padded cache, per-step compaction, ragged appends, per-row mode
switching and termination, position ids, attention masks) is correct. Only the
attention *numerics* differ from the real model; the orchestration is the same.

Run:  python -m tests.test_batched_generate   (from the Embodied/ directory)
"""

import math
import os
import sys
from functools import partial

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eaglevl.utils.locany.batched_generate import (  # noqa: E402
    _RUNAWAY_BOX_RUN, _is_runaway_box_loop, batched_generate, sample_row_ar,
    sample_row_mtp)
from eaglevl.utils.locany.mask_sdpa_utils import (  # noqa: E402
    apply_per_row_generation_window,
    apply_per_row_generation_window_vectorized,
    update_causal_mask_for_one_gen_window_2d,
    update_causal_mask_for_one_gen_window_4d)

NEG_INF = float("-inf")

# A compact token-id layout that keeps the special tokens reachable under random
# logits so the box / ref / empty / im_end decode paths actually get exercised.
VOCAB = 160
TOKEN_IDS = {
    "box_start_token_id": 10,
    "box_end_token_id": 11,
    "ref_start_token_id": 12,
    "ref_end_token_id": 13,
    "none_token_id": 14,
    "null_token_id": 15,
    "im_end_token_id": 16,
    "switch_token_id": 17,
    "default_mask_token_id": 18,
    "coord_start_token_id": 40,
    "coord_end_token_id": 110,
}
PAD_ID = 0
BLOCK_SIZE = 6  # == n_future


def build_block_mask(attn2d, input_ids, past_len, mask_token_id, block_size):
    """Reproduce the patched Qwen2 inference mask builder.

    Returns an additive ``[B, 1, S, past_len + S]`` mask (0 allow, -inf deny). The
    only part written here is the standard index-causal + 2D-padding base mask
    (the equivalent of transformers' ``_prepare_4d_causal_attention_mask``, kept
    local so the test needs no transformers). The custom part — the per-row MTP
    generation window — is NOT cloned: we call the *same*
    ``apply_per_row_generation_window`` / ``update_causal_mask_for_one_gen_window_2d``
    that ``Qwen2Model`` uses, so the masking under test cannot drift from production.
    """
    B, S = input_ids.shape
    device = input_ids.device
    Skv = past_len + S
    q = torch.arange(S, device=device).view(1, S, 1)
    k = torch.arange(Skv, device=device).view(1, 1, Skv)
    causal = k <= past_len + q  # [1, S, Skv]
    pad = attn2d.bool().view(B, 1, Skv)  # [B, 1, Skv]
    allow = causal & pad  # [B, S, Skv]
    mask = torch.zeros(B, S, Skv, device=device)
    mask.masked_fill_(~allow, NEG_INF)
    mask = mask.unsqueeze(1)  # [B, 1, S, Skv]
    if S == 1:
        return mask
    # Same per-row window dispatch as Qwen2Model._prepare_block_mask_for_inference
    # (use_cache=True path: the vectorized, sync-free dispatch).
    update_mask_func_4d = partial(
        update_causal_mask_for_one_gen_window_4d,
        block_size=block_size,
        use_cache=True,
        causal_attn=False,
    )
    return apply_per_row_generation_window_vectorized(
        mask, input_ids, mask_token_id, update_mask_func_4d
    )


class _Out:
    def __init__(self, logits, past_key_values):
        self.logits = logits
        self.past_key_values = past_key_values


class FakeLM:
    """Deterministic 1-layer attention LM standing in for the real Qwen2 LM.

    The batched decoder's correctness is about *orchestration* (masks, KV cache,
    position ids, ragged appends, per-row mode switching) — which is independent
    of the learned weights. So we replace the transformer with a tiny
    deterministic one that honours the same call contract the driver relies on:
    the same forward signature (``input_ids``/``attention_mask``/``position_ids``/
    ``past_key_values``), the same block-diffusion mask construction
    (``build_block_mask`` reuses the real mask util), and a KV cache that is
    appended and returned as a legacy tuple. Being integer-deterministic, it also
    adds zero numerical noise of its own, so any batched-vs-reference difference
    would be a real bookkeeping bug, not floating point.
    """

    def __init__(
        self,
        vocab=VOCAB,
        dim=32,
        n_heads=2,
        seed=0,
        mask_token_id=TOKEN_IDS["default_mask_token_id"],
        block_size=BLOCK_SIZE,
    ):
        g = torch.Generator().manual_seed(seed)
        self.dim, self.n_heads, self.hd = dim, n_heads, dim // n_heads
        self.mask_token_id, self.block_size = mask_token_id, block_size
        self.embed = torch.randn(vocab, dim, generator=g) * 0.5
        self.pos = torch.randn(4096, dim, generator=g) * 0.1
        self.Wq = torch.randn(dim, dim, generator=g) / math.sqrt(dim)
        self.Wk = torch.randn(dim, dim, generator=g) / math.sqrt(dim)
        self.Wv = torch.randn(dim, dim, generator=g) / math.sqrt(dim)
        self.Wo = torch.randn(dim, vocab, generator=g) / math.sqrt(dim)

    def _heads(self, x):
        B, S, _ = x.shape
        return x.view(B, S, self.n_heads, self.hd).transpose(1, 2)  # [B,H,S,hd]

    def __call__(
        self,
        input_ids,
        attention_mask,
        position_ids,
        past_key_values,
        use_cache=True,
        visual_features=None,
        image_token_index=None,
    ):
        B, S = input_ids.shape
        past_len = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        h = (
            self.embed[input_ids]
            + self.pos[position_ids.clamp(max=self.pos.shape[0] - 1)]
        )

        q = self._heads(h @ self.Wq)
        nk = self._heads(h @ self.Wk)
        nv = self._heads(h @ self.Wv)
        if past_key_values is not None:
            pk, pv = past_key_values[0]
            k = torch.cat([pk, nk], dim=2)
            v = torch.cat([pv, nv], dim=2)
        else:
            k, v = nk, nv

        mask = build_block_mask(
            attention_mask, input_ids, past_len, self.mask_token_id, self.block_size
        )  # [B,1,S,Skv]
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.hd) + mask
        # Safe softmax: fully-masked (pad-query) rows -> uniform-zero, never NaN.
        fully_masked = torch.isneginf(scores).all(dim=-1, keepdim=True)
        scores = scores.masked_fill(fully_masked, 0.0)
        attn = torch.softmax(scores, dim=-1)
        ctx = attn @ v  # [B,H,S,hd]
        ctx = ctx.transpose(1, 2).reshape(B, S, self.dim)
        out = h + ctx
        logits = out @ self.Wo
        return _Out(logits, ((k, v),))


class FakeModel:
    def __init__(self, lm):
        self.language_model = lm
        self.token_ids = TOKEN_IDS


@torch.no_grad()
def reference_single(
    lm,
    prompt_ids,
    n_future,
    generation_mode,
    generate_kwargs,
    max_new_tokens,
    model_max,
):
    """Independent re-implementation of the ORIGINAL single-sequence decode loop.

    This is the "golden" reference: it mirrors ``generate()`` in
    ``modeling_locateanything.py`` (the MTP speculative window, the block-diffusion
    position shift, and the truncate-and-recompute KV cache) but is written from
    scratch here rather than reusing the driver. So ``batched_generate`` on a
    single row matching this output proves the driver reproduces the *original
    algorithm*, not merely that it agrees with itself. Returns the generated ids.
    """
    device = prompt_ids.device
    token_ids = TOKEN_IDS
    generated = prompt_ids.clone().unsqueeze(0)
    seq_len = generated.shape[1]
    total = min(model_max, seq_len + max_new_tokens)
    use_mtp = generation_mode in ("fast", "hybrid")
    mask_id = token_ids["default_mask_token_id"]
    pre_mask = torch.full((1, n_future - 1), mask_id, dtype=torch.long, device=device)
    full_pos = torch.arange(0, total + n_future, device=device).unsqueeze(0)
    past = None
    out = []

    def _slice(seq):
        # Feed only the tokens not yet in the cache (as prepare_inputs_for_generation does).
        pl = past[0][0].size(2) if past is not None else 0
        return seq[:, pl:]

    while generated.shape[1] < total:
        if use_mtp:
            # MTP: append the speculative window = [duplicate of last committed
            # token] + (n_future-1) mask tokens, and predict all n_future at once.
            gwm = torch.cat([generated, generated[:, -1:], pre_mask], dim=1)
            start = past[0][0].size(2) if past is not None else 0
            pos = full_pos[:, start : gwm.size(1)].clone()
            pos[
                0, -n_future:
            ] -= 1  # block-diffusion pe: window shares the last real position
            inp = _slice(gwm)
            posn = pos[:, -inp.size(1) :]
            attn = torch.ones(1, gwm.size(1), dtype=torch.long, device=device)
        else:
            # AR: ordinary single-step causal decoding.
            start = past[0][0].size(2) if past is not None else 0
            pos = full_pos[:, start : generated.size(1)]
            inp = _slice(generated)
            posn = pos[:, -inp.size(1) :]
            attn = torch.ones(1, generated.size(1), dtype=torch.long, device=device)

        res = lm(
            input_ids=inp,
            attention_mask=attn,
            position_ids=posn,
            past_key_values=past,
            use_cache=True,
        )
        # Truncate-and-recompute: keep only the committed-prefix KV and discard the
        # speculative window's KV. ``generated`` is still the pre-append length here,
        # so the tokens accepted this step are re-computed (with real ids) next step.
        past = tuple(
            (k[:, :, : generated.shape[1], :], v[:, :, : generated.shape[1], :])
            for k, v in res.past_key_values
        )

        # Sample via the SAME per-row helpers the batched driver uses, so the only
        # thing under test is the loop's bookkeeping, not the token decoding.
        if use_mtp:
            out_type, out_token = sample_row_mtp(
                res.logits[:, -n_future:, :],
                generated,
                token_ids,
                n_future,
                generation_mode,
                generate_kwargs,
            )
        else:
            out_type, out_token = sample_row_ar(
                res.logits[:, -1:, :],
                generated,
                token_ids,
                generation_mode,
                generate_kwargs,
            )

        generated = torch.cat([generated, out_token.unsqueeze(0)], dim=1)
        out.extend(out_token.tolist())
        if len(out) + seq_len >= total:
            break
        if out_type == "im_end":
            break
        # hybrid only: drop to AR on a malformed box, return to MTP after </box>.
        if generation_mode == "hybrid":
            if out_type == "error_box":
                use_mtp = False
            elif out_type == "box_end_ar":
                use_mtp = True
    return out


def _left_pad_batch(prompts, pad_id):
    P = max(len(p) for p in prompts)
    B = len(prompts)
    ids = torch.full((B, P), pad_id, dtype=torch.long)
    am = torch.zeros((B, P), dtype=torch.long)
    for b, p in enumerate(prompts):
        ids[b, P - len(p) :] = torch.tensor(p, dtype=torch.long)
        am[b, P - len(p) :] = 1
    return ids, am


class IdentityTokenizer:
    """Decodes to a space-joined id string and re-encodes it back, so we can
    compare the batched driver's string output against raw id lists."""

    model_max_length = 4096
    pad_token_id = PAD_ID

    def decode(self, ids, skip_special_tokens=False):
        return " ".join(str(int(i)) for i in ids.tolist())

    @staticmethod
    def parse(text):
        return [int(x) for x in text.split()] if text else []


def _random_prompt(g, length):
    # Avoid emitting mask/pad tokens inside the prompt.
    toks = []
    while len(toks) < length:
        t = int(torch.randint(20, VOCAB, (1,), generator=g).item())
        if t not in (PAD_ID, TOKEN_IDS["default_mask_token_id"]):
            toks.append(t)
    return toks


def _bgen(model, prompts, n_future, gk):
    """Batched decode of a list of prompts -> list[list[int]]."""
    tok = IdentityTokenizer()
    ids, am = _left_pad_batch(prompts, PAD_ID)
    return [
        tok.parse(s)
        for s in batched_generate(
            model,
            input_ids=ids,
            attention_mask=am,
            vit_embeds=None,
            image_token_index=None,
            tokenizer=tok,
            n_future=n_future,
            pad_token_id=PAD_ID,
            generate_kwargs=dict(gk),
        )
    ]


def evaluate(
    seeds=range(12), modes=("slow", "fast", "hybrid"), n_prompts=4, max_new_tokens=40
):
    """Run the parity + batch-independence checks and RETURN structured results
    (no asserts), so notebooks/reports can tabulate them.

    Returns ``{"detail": [per (seed,mode) dict], "coverage": {(type,len): count}}``.
    The coverage counter records which decode patterns (variable-length accepts,
    AR-fallback ``error_box``, etc.) were actually exercised.
    """
    import collections

    import eaglevl.utils.locany.batched_generate as BG

    cov = collections.Counter()
    _orig_hp = BG.handle_pattern

    def _spy(x0, token_ids, generation_mode="hybrid"):
        r = _orig_hp(x0, token_ids, generation_mode)
        cov[(r["type"], len(r["tokens"]))] += 1
        return r

    BG.handle_pattern = _spy
    tok = IdentityTokenizer()
    n_future = BLOCK_SIZE
    detail = []
    try:
        for seed in seeds:
            lm = FakeLM(seed=seed)
            model = FakeModel(lm)
            g = torch.Generator().manual_seed(1000 + seed)
            prompts = [
                _random_prompt(g, int(torch.randint(3, 12, (1,), generator=g).item()))
                for _ in range(n_prompts)
            ]
            for mode in modes:
                gk = {
                    "generation_mode": mode,
                    "max_new_tokens": max_new_tokens,
                    "temperature": 0,
                }

                # (1) algorithm parity: batched(B=1) == independent original loop
                par = par_ok = 0
                for p in prompts:
                    ref = reference_single(
                        lm,
                        torch.tensor(p),
                        n_future,
                        mode,
                        dict(gk),
                        max_new_tokens,
                        tok.model_max_length,
                    )
                    bat = _bgen(model, [p], n_future, gk)[0]
                    par += 1
                    par_ok += int(ref == bat)

                # (2) batch independence: a row is identical alone vs in a group.
                group = _bgen(model, prompts, n_future, gk)
                ind = ind_ok = 0
                for b, p in enumerate(prompts):
                    solo = _bgen(model, [p], n_future, gk)[0]
                    ind += 1
                    ind_ok += int(solo == group[b])

                detail.append(
                    dict(
                        seed=seed,
                        mode=mode,
                        parity_checks=par,
                        parity_pass=par_ok,
                        batch_indep_checks=ind,
                        batch_indep_pass=ind_ok,
                    )
                )
    finally:
        BG.handle_pattern = _orig_hp
    return {"detail": detail, "coverage": dict(cov)}


def _box_seg(c1, c2, c3, c4):
    bs, be = TOKEN_IDS["box_start_token_id"], TOKEN_IDS["box_end_token_id"]
    cs = TOKEN_IDS["coord_start_token_id"]
    return [bs, cs + c1, cs + c2, cs + c3, cs + c4, be]


def test_runaway_box_guard():
    """``_is_runaway_box_loop`` flags a tail of ``_RUNAWAY_BOX_RUN`` consecutive
    <box> segments that drift by only a few units each, but not shorter runs,
    larger jumps, or non-box tokens."""
    # A run of slowly-drifting boxes, like the COCO "keep scanning" loop.
    drifting = []
    for k in range(_RUNAWAY_BOX_RUN):
        drifting += _box_seg(10 + 3 * k, 20, 11 + 3 * k, 21)
    assert _is_runaway_box_loop(drifting, TOKEN_IDS)

    # One fewer segment must not trigger.
    assert not _is_runaway_box_loop(drifting[6:], TOKEN_IDS)

    # A large jump in the most recent segment breaks the run.
    big_jump = drifting[:-6] + _box_seg(60, 20, 61, 21)
    assert not _is_runaway_box_loop(big_jump, TOKEN_IDS)

    # Non-box tokens interrupting the run (e.g. a <ref> in between) break it.
    interrupted = drifting[:-7] + [12, 99, 13] + drifting[-6:]
    assert not _is_runaway_box_loop(interrupted, TOKEN_IDS)

    # A normal, varied detection list (large jumps between distinct objects)
    # must not trigger even with many boxes.
    varied = []
    for k in range(_RUNAWAY_BOX_RUN + 2):
        varied += _box_seg((7 * k) % 70, (13 * k) % 70, (3 * k) % 70, (17 * k) % 70)
    assert not _is_runaway_box_loop(varied, TOKEN_IDS)
    print("OK  runaway-box-guard checks: 5")


def check_mask_window_vectorized(seeds=range(20)):
    """Stage A equivalence check: ``apply_per_row_generation_window_vectorized``
    + ``update_causal_mask_for_one_gen_window_4d`` (the new, sync-free dispatch
    now used by ``Qwen2Model`` for ``use_cache=True``) must produce a
    bit-identical mask to the original per-row
    ``apply_per_row_generation_window`` + ``update_causal_mask_for_one_gen_window_2d``
    it replaced -- for every (B, S, Skv, block_size, causal_attn) combination,
    including mixed MTP/AR batches and the all-AR small-S case (S < block_size+1)
    that the original per-row loop never had to handle.

    Returns ``(checks, passes)``.
    """
    mask_token_id = TOKEN_IDS["default_mask_token_id"]
    checks = passes = 0
    for seed in seeds:
        g = torch.Generator().manual_seed(seed)
        for B in (1, 3, 5):
            for S, Skv in ((5, 5), (10, 10), (10, 14), (1, 1), (12, 20)):
                for block_size in (1, 4, 6):
                    for causal_attn in (False, True):
                        input_ids = torch.randint(20, VOCAB, (B, S), generator=g)
                        # Randomly mark some rows as MTP (last token == mask id).
                        # MTP rows must satisfy S, Skv >= block_size + 1 (the
                        # same invariant the original per-row loop relied on);
                        # skip marking a row MTP if that would violate it, so
                        # both implementations stay well-defined and comparable.
                        can_be_mtp = S >= block_size + 1 and Skv >= block_size + 1
                        if can_be_mtp:
                            is_mtp = torch.rand(B, generator=g) < 0.5
                            input_ids[is_mtp, -1] = mask_token_id
                        attn = (
                            torch.rand(B, 1, S, Skv, generator=g) < 0.5
                        ).float() * NEG_INF
                        attn = torch.nan_to_num(attn, nan=0.0)

                        old_func = partial(
                            update_causal_mask_for_one_gen_window_2d,
                            block_size=block_size,
                            use_cache=True,
                            causal_attn=causal_attn,
                        )
                        new_func = partial(
                            update_causal_mask_for_one_gen_window_4d,
                            block_size=block_size,
                            use_cache=True,
                            causal_attn=causal_attn,
                        )
                        old_mask = apply_per_row_generation_window(
                            attn.clone(), input_ids, mask_token_id, old_func
                        )
                        new_mask = apply_per_row_generation_window_vectorized(
                            attn.clone(), input_ids, mask_token_id, new_func
                        )
                        checks += 1
                        passes += int(torch.equal(old_mask, new_mask))
    return checks, passes


def run_tests():
    test_runaway_box_guard()

    # Wider coverage than the original defaults: more seeds and more prompts
    # per batch broadens the (A, pend_lens, win_lens, cache_len) combinations
    # exercised by the vectorized Step 1/5 closed forms (e.g. A=1 via the
    # parity checks, A=6 via batch-independence, varying ragged
    # pending-token lengths from the MTP/AR/error_box mix).
    rep = evaluate(seeds=range(24), n_prompts=6)
    d = rep["detail"]
    npar = sum(r["parity_checks"] for r in d)
    npar_ok = sum(r["parity_pass"] for r in d)
    nind = sum(r["batch_indep_checks"] for r in d)
    nind_ok = sum(r["batch_indep_pass"] for r in d)
    assert npar == npar_ok, f"PARITY FAILURES: {npar - npar_ok}/{npar}"
    assert nind == nind_ok, f"BATCH-INDEPENDENCE FAILURES: {nind - nind_ok}/{nind}"

    mwv_checks, mwv_passes = check_mask_window_vectorized()
    assert (
        mwv_checks == mwv_passes
    ), f"MASK-WINDOW VECTORIZATION MISMATCHES: {mwv_checks - mwv_passes}/{mwv_checks}"

    print(
        f"OK  algorithm-parity checks: {npar}   batch-independence checks: {nind}"
        f"   mask-window-vectorized checks: {mwv_checks}"
    )
    print("All batched-generation correctness tests passed.")


if __name__ == "__main__":
    run_tests()
