# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
"""Throughput / memory measurements for batched decoding.

``speed_sweep`` calls this branch's real ``worker.predict_batch`` on N distinct
image objects — sequential (N x B=1) vs batched (1 x B=N) — so the result reflects
whatever this checkout's ``predict_batch`` actually does for both vision encoding
and decode. It's branch-isolated: PR1's checkout batches the vision tower for
distinct images (one ``extract_feature`` call at B=N), PR2's checkout does not
(N separate B=1 ``_encode_image_features`` calls via ``feat_cache``) — run on each
PR's own notebook, ``combine`` shows both curves side by side.
"""
import gc
import time

import torch


def free():
    """Reclaim PyTorch's cached (reserved) GPU memory between runs."""
    gc.collect()
    torch.cuda.empty_cache()


def _timed(fn):
    torch.cuda.synchronize()
    t = time.time()
    r = fn()
    torch.cuda.synchronize()
    return r, time.time() - t


def speed_sweep(
    worker,
    image,
    prompt,
    batch_sizes=(1, 2, 4, 8, 16, 32),
    max_new_tokens=256,
    generation_mode="hybrid",
):
    """Sequential (N x ``predict_batch([img],[prompt])``) vs batched (one
    ``predict_batch(imgs, prompts)`` call) over ``batch_sizes``, using **distinct
    image objects** (``image.copy()``) so the batched call sees N independent
    images — the same shape of input a caller with N different images would pass.

    ``endtoend_speedup = sequential_s / batched_s`` is this branch's real
    wall-clock win for that call. What's inside ``batched_s`` depends on this
    checkout's ``predict_batch``: PR1 stacks all N images' ``pixel_values`` and
    encodes vision once at B=N (plus batches decode); PR2's unified
    ``predict_batch`` encodes each distinct image separately at B=1 via
    ``feat_cache`` (plus batches decode). Both are real, just different —
    ``combine`` reports each branch's own number rather than picking one as "the"
    PR1 isolate.
    """
    _ = worker.predict_batch(
        [image.copy(), image.copy()],
        [prompt, prompt],
        max_new_tokens=64,
        temperature=0.0,
    )
    free()
    rows = []
    for n in batch_sizes:
        free()
        torch.cuda.reset_peak_memory_stats()
        imgs, qs = [image.copy() for _ in range(n)], [prompt] * n
        _, t_seq = _timed(
            lambda: [
                worker.predict_batch(
                    [i],
                    [q],
                    generation_mode=generation_mode,
                    max_new_tokens=max_new_tokens,
                    temperature=0.0,
                )[0]
                for i, q in zip(imgs, qs)
            ]
        )
        _, t_bat = _timed(
            lambda: worker.predict_batch(
                imgs,
                qs,
                generation_mode=generation_mode,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
            )
        )
        rows.append(
            dict(
                batch_size=n,
                sequential_s=round(t_seq, 2),
                batched_s=round(t_bat, 2),
                endtoend_speedup=round(t_seq / t_bat, 2),
                peak_GB=round(torch.cuda.max_memory_reserved() / 1e9, 1),
            )
        )
    free()
    return rows


_PROFILE_STEP_KEYS = (
    "step0_eject",
    "step1_assemble",
    "step2_mask",
    "step3_forward",
    "step4_sample",
    "step5_compact",
)


def profile_sweep(
    worker,
    image,
    prompt,
    batch_sizes=(16, 32),
    generation_modes=("hybrid", "fast"),
    max_new_tokens=256,
    repeats=3,
):
    """Per-step decode-loop timing via ``predict_batch(..., profile={})``, at
    larger batch sizes and in ``fast`` mode (every step hits the MTP mask
    dispatch, vs. only MTP steps in ``hybrid``) -- the regimes where PR3's
    closed-form Step1/Step5 and vectorized mask dispatch should matter most.

    Each ``(batch_size, generation_mode)`` cell is run ``repeats`` times (fresh
    ``profile`` dict each time) and averaged, to separate the effect from
    run-to-run GPU noise. ``bookkeeping_s`` is steps 0/1/2/4/5 -- the per-row
    Python-loop bookkeeping PR3 rewrites in closed form; ``step3_forward_s`` is
    the model call itself, which PR3 does not change algorithmically (so it's
    the control -- it should not move).

    Large batch sizes can OOM at prefill (the ``[B, S, V]`` fp32 logits tensor
    is ``O(B)`` over the ~150K vocab, on top of whatever the earlier
    ``eval``/``grounded``/``speed`` tasks left resident). An OOM on a given
    ``(batch_size, mode)`` cell is caught, memory is freed, and that cell is
    recorded with ``oom=True`` (or ``partial_oom=True`` if some but not all
    ``repeats`` succeeded) rather than raising -- so one too-large cell can't
    take the whole ``run_benchmark`` record down with it.
    """
    _ = worker.predict_batch(
        [image.copy(), image.copy()], [prompt, prompt],
        max_new_tokens=64, temperature=0.0,
    )
    free()
    rows = []
    for n in batch_sizes:
        for mode in generation_modes:
            free()
            imgs, qs = [image.copy() for _ in range(n)], [prompt] * n
            accum = {k: 0.0 for k in _PROFILE_STEP_KEYS}
            n_steps_total = 0.0
            total_s = 0.0
            completed = 0
            for _ in range(repeats):
                profile = {}
                try:
                    _, t = _timed(
                        lambda: worker.predict_batch(
                            imgs, qs, generation_mode=mode,
                            max_new_tokens=max_new_tokens, temperature=0.0,
                            profile=profile,
                        )
                    )
                except RuntimeError as e:
                    if "out of memory" not in str(e).lower():
                        raise
                    free()
                    break
                total_s += t
                n_steps_total += profile.get("n_steps", 0)
                for k in _PROFILE_STEP_KEYS:
                    accum[k] += profile.get(k, 0.0)
                completed += 1
            if completed == 0:
                rows.append(dict(
                    batch_size=n, generation_mode=mode, repeats=0, oom=True,
                ))
                continue
            bookkeeping_s = sum(
                accum[k] for k in _PROFILE_STEP_KEYS if k != "step3_forward"
            )
            total_s /= completed
            row = dict(
                batch_size=n,
                generation_mode=mode,
                repeats=completed,
                n_steps=round(n_steps_total / completed, 1),
                total_s=round(total_s, 3),
                step3_forward_s=round(accum["step3_forward"] / completed, 3),
                bookkeeping_s=round(bookkeeping_s / completed, 3),
                bookkeeping_pct=round(100 * bookkeeping_s / completed / total_s, 1)
                if total_s else 0.0,
            )
            for k in _PROFILE_STEP_KEYS:
                row[f"{k}_s"] = round(accum[k] / completed, 4)
            if completed < repeats:
                row["partial_oom"] = True
            rows.append(row)
    free()
    return rows
