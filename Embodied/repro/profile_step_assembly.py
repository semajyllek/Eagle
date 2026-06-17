# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
"""Granular, model-free microbenchmark of the PR1 -> PR3 ``batched_generate``
per-step bookkeeping (Step 1: input-block assembly, Step 5: cache compaction
index build).

PR3 replaces PR1's per-row Python loops (one ``torch.arange``/``torch.cat``/
slice-assign per row) with vectorized ``torch.where``/``torch.gather``/
broadcast-arange ops over the whole ``[A, W]`` / ``[A, new_Ckv]`` block. This
script reimplements both versions of just those two code blocks (verified
numerically identical to the real ``batched_generate`` via ``torch.equal``)
and times them directly with MPS sync -- no model load, no token generation,
so it runs in seconds and isolates exactly the optimized code paths from the
(identical, unaffected) model-forward and KV-cache-gather costs.

Usage:
    .venv-mps/bin/python -m repro.profile_step_assembly
"""
import statistics
import time

import torch

N_FUTURE = 6  # matches modeling_locateanything.py's n_future_tokens default
MASK_TOKEN_ID = 1
PAD_TOKEN_ID = 0


# ---------------------------------------------------------------------------
# PR1 (this checkout): per-row Python loops.
# ---------------------------------------------------------------------------
def _left_pad_rows_pr1(rows, pad_value, width, dtype, device):
    B = len(rows)
    out = torch.full((B, width), pad_value, dtype=dtype, device=device)
    for b, r in enumerate(rows):
        n = r.shape[0]
        if n:
            out[b, width - n :] = r
    return out


def pr1_step1(rows, cache_len, pending, use_mtp, finished, device):
    seg_ids, seg_pos = [], []
    win_lens = {b: 0 for b in rows}
    pend_lens = {b: 0 for b in rows}
    for b in rows:
        if finished[b]:
            seg_ids.append(torch.empty(0, dtype=torch.long, device=device))
            seg_pos.append(torch.empty(0, dtype=torch.long, device=device))
            continue
        p = pending[b]
        pend_lens[b] = p.shape[0]
        base = cache_len[b]
        pos = torch.arange(base, base + p.shape[0], device=device)
        if use_mtp[b]:
            dup = p[-1:].clone()
            masks = torch.full((N_FUTURE - 1,), MASK_TOKEN_ID, dtype=torch.long, device=device)
            ids = torch.cat([p, dup, masks])
            wpos = (
                torch.arange(base + p.shape[0], base + p.shape[0] + N_FUTURE, device=device) - 1
            )
            pos = torch.cat([pos, wpos])
            win_lens[b] = N_FUTURE
        else:
            ids = p
        seg_ids.append(ids)
        seg_pos.append(pos)

    W = max(s.shape[0] for s in seg_ids)
    cur_input_ids = _left_pad_rows_pr1(seg_ids, PAD_TOKEN_ID, W, torch.long, device)
    cur_pos = _left_pad_rows_pr1(seg_pos, 0, W, torch.long, device)
    cur_real = torch.zeros((len(rows), W), dtype=torch.long, device=device)
    for j, b in enumerate(rows):
        n = seg_ids[j].shape[0]
        if n:
            cur_real[j, W - n :] = 1
    return cur_input_ids, cur_pos, cur_real, win_lens, pend_lens, W


def pr1_step5(rows, cache_len, pend_lens, win_lens, Ckv, W, device):
    new_cache_len_rows = {b: cache_len[b] + pend_lens[b] for b in rows}
    new_Ckv = max(new_cache_len_rows.values())
    A = len(rows)
    keep_idx = torch.zeros((A, new_Ckv), dtype=torch.long, device=device)
    keep_valid = torch.zeros((A, new_Ckv), dtype=torch.bool, device=device)
    for j, b in enumerate(rows):
        old_cols = (
            torch.arange(Ckv - cache_len[b], Ckv, device=device)
            if cache_len[b]
            else torch.empty(0, dtype=torch.long, device=device)
        )
        blk_real = pend_lens[b] + win_lens[b]
        pend_start = Ckv + (W - blk_real)
        pend_cols = torch.arange(pend_start, pend_start + pend_lens[b], device=device)
        src = torch.cat([old_cols, pend_cols])
        k = src.shape[0]
        if k:
            keep_idx[j, new_Ckv - k :] = src
            keep_valid[j, new_Ckv - k :] = True
    return keep_idx, keep_valid, new_Ckv


# ---------------------------------------------------------------------------
# PR3: vectorized over [A, W] / [A, new_Ckv].
# ---------------------------------------------------------------------------
def pr3_step1(rows, cache_len, pending, use_mtp, finished, device):
    A = len(rows)
    win_lens = {b: (N_FUTURE if (use_mtp[b] and not finished[b]) else 0) for b in rows}
    pend_lens = {b: (0 if finished[b] else pending[b].shape[0]) for b in rows}

    pend_lens_l = [pend_lens[b] for b in rows]
    win_lens_l = [win_lens[b] for b in rows]
    seg_lens_l = [pend_lens_l[j] + win_lens_l[j] for j in range(A)]
    cache_len_l = [cache_len[b] for b in rows]

    Pmax = max(pend_lens_l)
    W = max(seg_lens_l)

    pend_pad = torch.full((A, Pmax), PAD_TOKEN_ID, dtype=torch.long, device=device)
    for j, b in enumerate(rows):
        n = pend_lens_l[j]
        if n:
            pend_pad[j, Pmax - n :] = pending[b]

    pend_lens_t = torch.tensor(pend_lens_l, dtype=torch.long, device=device).unsqueeze(1)
    win_lens_t = torch.tensor(win_lens_l, dtype=torch.long, device=device).unsqueeze(1)
    cache_len_t = torch.tensor(cache_len_l, dtype=torch.long, device=device).unsqueeze(1)
    seg_lens_t = torch.tensor(seg_lens_l, dtype=torch.long, device=device).unsqueeze(1)

    c = torch.arange(W, device=device).unsqueeze(0)
    left_pad = W - seg_lens_t
    win_start = W - win_lens_t

    cur_real = (c >= left_pad).long()
    in_window = c >= win_start

    pend_gather_idx = (c + (Pmax - W) + win_lens_t).clamp(0, Pmax - 1)
    pend_vals = torch.gather(pend_pad, 1, pend_gather_idx)

    dup = pend_pad[:, -1:]
    mask_fill = torch.full((A, W), MASK_TOKEN_ID, dtype=torch.long, device=device)
    window_vals = torch.where(c - win_start == 0, dup, mask_fill)

    cur_input_ids = torch.where(in_window, window_vals, pend_vals)
    cur_input_ids = torch.where(
        cur_real.bool(), cur_input_ids, torch.full_like(cur_input_ids, PAD_TOKEN_ID)
    )

    pos_pend = cache_len_t + (c - left_pad)
    pos_win = cache_len_t + pend_lens_t + (c - win_start) - 1
    cur_pos = torch.where(in_window, pos_win, pos_pend)
    cur_pos = torch.where(cur_real.bool(), cur_pos, torch.zeros_like(cur_pos))

    return cur_input_ids, cur_pos, cur_real, win_lens, pend_lens, W


def pr3_step5(rows, cache_len, pend_lens, win_lens, Ckv, W, device):
    cache_len_t = torch.tensor([cache_len[b] for b in rows], dtype=torch.long, device=device).unsqueeze(1)
    pend_lens_t = torch.tensor([pend_lens[b] for b in rows], dtype=torch.long, device=device).unsqueeze(1)
    win_lens_t = torch.tensor([win_lens[b] for b in rows], dtype=torch.long, device=device).unsqueeze(1)

    new_cache_len_rows = {b: cache_len[b] + pend_lens[b] for b in rows}
    new_Ckv = max(new_cache_len_rows.values())

    k_t = cache_len_t + pend_lens_t
    offset_old_t = Ckv + pend_lens_t - new_Ckv
    offset_pend_t = Ckv + W - win_lens_t - new_Ckv

    c2 = torch.arange(new_Ckv, device=device).unsqueeze(0)
    use_pend = c2 >= (new_Ckv - pend_lens_t)
    offset = torch.where(use_pend, offset_pend_t, offset_old_t)
    keep_valid = c2 >= (new_Ckv - k_t)
    keep_idx = torch.where(keep_valid, c2 + offset, torch.zeros_like(c2))
    return keep_idx, keep_valid, new_Ckv


# ---------------------------------------------------------------------------
# Synthetic steady-state decode step, near Ckv=2048.
# ---------------------------------------------------------------------------
def make_state(A, cache_len_val, device):
    rows = list(range(A))
    cache_len = {b: cache_len_val for b in rows}
    finished = {b: False for b in rows}
    use_mtp = {b: (b % 2 == 0) for b in rows}  # alternate MTP / AR rows
    pend_choices = [1, 2, 3]
    pending = {
        b: torch.arange(100 + b, 100 + b + pend_choices[b % 3], dtype=torch.long, device=device)
        for b in rows
    }
    return rows, cache_len, pending, use_mtp, finished


def sync(device):
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def bench(fn, args, device, n_iters=200, warmup=20):
    """Per-call latency: sync after every call. Captures the full
    Python-dispatch + queue + execute + sync round trip for one call."""
    for _ in range(warmup):
        fn(*args)
    sync(device)
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        fn(*args)
        sync(device)
        times.append(time.perf_counter() - t0)
    return statistics.median(times) * 1e6  # microseconds


def bench_dispatch(fn, args, device, n_iters=200, warmup=20):
    """Back-to-back dispatch cost: no sync between calls, one sync at the
    end. Approximates the cost added to the critical path when these tiny
    bookkeeping ops are queued behind/alongside other (heavier) GPU work,
    as in the real decode loop -- i.e. CPU-side dispatch overhead, with GPU
    execution able to overlap rather than forcing a round trip each call."""
    for _ in range(warmup):
        fn(*args)
    sync(device)
    t0 = time.perf_counter()
    for _ in range(n_iters):
        fn(*args)
    sync(device)
    return (time.perf_counter() - t0) / n_iters * 1e6  # microseconds


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}\n")

    cache_len_val = 2048
    Ckv = cache_len_val

    for label, bench_fn in (("per-call latency (sync each call)", bench),
                             ("back-to-back dispatch (sync once)", bench_dispatch)):
        print(f"\n=== {label} ===")
        header = f"{'A':>3} | {'step1 PR1 (us)':>14} {'step1 PR3 (us)':>14} {'speedup':>8} | {'step5 PR1 (us)':>14} {'step5 PR3 (us)':>14} {'speedup':>8}"
        print(header)
        print("-" * len(header))

        for A in (1, 2, 4, 8, 16):
            rows, cache_len, pending, use_mtp, finished = make_state(A, cache_len_val, device)

            # Correctness: PR1 and PR3 step1 must agree exactly.
            out1 = pr1_step1(rows, cache_len, pending, use_mtp, finished, device)
            out3 = pr3_step1(rows, cache_len, pending, use_mtp, finished, device)
            for name, a, b in zip(("cur_input_ids", "cur_pos", "cur_real"), out1[:3], out3[:3]):
                assert torch.equal(a, b), f"A={A} step1 {name} mismatch:\nPR1={a}\nPR3={b}"
            win_lens, pend_lens, W = out1[3], out1[4], out1[5]

            # Correctness: PR1 and PR3 step5 must agree exactly.
            k1 = pr1_step5(rows, cache_len, pend_lens, win_lens, Ckv, W, device)
            k3 = pr3_step5(rows, cache_len, pend_lens, win_lens, Ckv, W, device)
            assert torch.equal(k1[0], k3[0]), f"A={A} step5 keep_idx mismatch:\nPR1={k1[0]}\nPR3={k3[0]}"
            assert torch.equal(k1[1], k3[1]), f"A={A} step5 keep_valid mismatch:\nPR1={k1[1]}\nPR3={k3[1]}"

            t1_pr1 = bench_fn(pr1_step1, (rows, cache_len, pending, use_mtp, finished, device), device)
            t1_pr3 = bench_fn(pr3_step1, (rows, cache_len, pending, use_mtp, finished, device), device)
            t5_pr1 = bench_fn(pr1_step5, (rows, cache_len, pend_lens, win_lens, Ckv, W, device), device)
            t5_pr3 = bench_fn(pr3_step5, (rows, cache_len, pend_lens, win_lens, Ckv, W, device), device)

            print(
                f"{A:>3} | {t1_pr1:>14.1f} {t1_pr3:>14.1f} {t1_pr1 / t1_pr3:>7.2f}x | "
                f"{t5_pr1:>14.1f} {t5_pr3:>14.1f} {t5_pr1 / t5_pr3:>7.2f}x"
            )

    print("\nAll PR1/PR3 outputs verified numerically identical (torch.equal) at each A.")
    print("Times are per-call median (latency) / mean (dispatch) over 200 iters (20 warmup).")


if __name__ == "__main__":
    main()
