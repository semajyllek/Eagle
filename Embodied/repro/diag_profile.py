# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
"""Sanity check for the ``profile`` dict in ``batched_generate``.

Runs ``predict_batch(..., profile={})`` and prints the Step1-5 time breakdown
plus ``n_steps``. Use this to confirm the instrumentation works end-to-end
before relying on it in the Colab eval run.

Usage:
    .venv-mps/bin/python -m repro.diag_profile [--src-dir <locany dir>] [--n 3] [--max-new-tokens 64]
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
    ap.add_argument("--max-new-tokens", type=int, default=64)
    args = ap.parse_args()

    from repro.model import load_overlaid_worker, sample_images

    worker, snap = load_overlaid_worker(device=args.device, src_dir=args.src_dir)
    imgs, prompts = sample_images(args.n)

    profile = {}
    logger.info("running predict_batch ...")
    outs = worker.predict_batch(
        [im.copy() for im in imgs],
        prompts,
        generation_mode="hybrid",
        max_new_tokens=args.max_new_tokens,
        temperature=0.0,
        profile=profile,
    )
    n_steps = profile.pop("n_steps", 0)
    a_hist = profile.pop("A_history", [])
    print(f"\n  n_steps={n_steps}")
    total = sum(profile.values())
    for key in ("step1_assemble", "step2_mask", "step3_forward", "step4_sample", "step5_compact"):
        v = profile.get(key, 0.0)
        pct = (v / total * 100) if total else 0.0
        print(f"  {key:>16}: {v:8.4f}s  ({pct:5.1f}%)")
    print(f"  {'sum':>16}: {total:8.4f}s")
    for i, o in enumerate(outs):
        ntok = len(worker.tokenizer(o, add_special_tokens=False).input_ids)
        print(f"  row {i}: {ntok} tokens")


if __name__ == "__main__":
    main()
