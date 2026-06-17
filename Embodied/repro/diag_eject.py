# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
"""Diagnose the PR3 early-eject regression: does enabling ``early_eject`` change
the *generated text* for surviving rows, not just drop finished ones?

If ``early_eject=True`` and ``early_eject=False`` produce different ``answer``
strings for the same row with the same checkout's code, that's a correctness bug
in the eject/compaction path (not just a perf characteristic) -- and a likely
explanation for a straggler row whose output length balloons only when eject is on.

Usage:
    .venv-mps/bin/python -m repro.diag_eject --src-dir <PR3>/Embodied/eaglevl/utils/locany
    .venv-mps/bin/python -m repro.diag_eject  # this checkout (PR1)
"""
import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src-dir", default=None, help="locany dir to overlay (default: this checkout)")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--n", type=int, default=3, help="number of distinct images/rows")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    args = ap.parse_args()

    from repro.model import load_overlaid_worker, sample_images

    worker, snap = load_overlaid_worker(device=args.device, src_dir=args.src_dir)
    imgs, prompts = sample_images(args.n)

    results = {}
    for eject in (True, False):
        logger.info("running predict_batch with early_eject=%s ...", eject)
        outs = worker.predict_batch(
            [im.copy() for im in imgs],
            prompts,
            generation_mode="hybrid",
            max_new_tokens=args.max_new_tokens,
            temperature=0.0,
            early_eject=eject,
        )
        results[eject] = outs
        print(f"\n=== early_eject={eject} ===")
        for i, o in enumerate(outs):
            ans = o
            ntok = len(worker.tokenizer(ans, add_special_tokens=False).input_ids)
            print(f"  row {i}: len={ntok} tokens")
            print(f"    answer={ans!r}")

    print("\n=== diff (early_eject True vs False) ===")
    any_diff = False
    for i in range(args.n):
        a = results[True][i]
        b = results[False][i]
        if a != b:
            any_diff = True
            print(f"  row {i}: DIFFERS")
            print(f"    eject=True : {a!r}")
            print(f"    eject=False: {b!r}")
        else:
            print(f"  row {i}: identical")
    if not any_diff:
        print("\nAll rows identical between early_eject=True and early_eject=False.")


if __name__ == "__main__":
    main()
