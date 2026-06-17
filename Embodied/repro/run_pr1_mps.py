# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
"""Local (MPS/CPU) script version of ``benchmark_pr1.ipynb``'s GPU sections.

Loads *this checkout's* code via ``load_overlaid_worker(device=...)`` and runs
the same semantic-parity smoke check + ``run_benchmark`` call as the notebook,
then writes ``results_<tag>.json`` and an ``EVIDENCE_<tag>_mps.md`` report.

Defaults mirror the notebook (``limit=50``, ``batch_size=8``): COCO+LVIS at
limit=50 is <500 samples, and ``speed_sweep`` (always run by "speed") tops out
at batch_size=16 regardless of ``--batch-size``. Lower ``--limit`` /
``--batch-size`` for a faster first pass -- on MPS, samples that reach
``max_new_tokens`` (2048 for eval, 512 for grounded) can take many minutes each.

Usage:
    .venv-mps/bin/python -m repro.run_pr1_mps --limit 50 --batch-size 8
"""
import argparse
import logging
import os
import subprocess
import sys

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

_HERE = os.path.dirname(os.path.abspath(__file__))
_EMBODIED = os.path.join(_HERE, "..")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--limit", type=int, default=50,
                    help="samples per dataset for the eval task (COCO/LVIS)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--datasets", nargs="+", default=["COCO", "LVIS"])
    ap.add_argument("--tasks", nargs="+", default=["eval", "grounded", "speed"])
    ap.add_argument("--evaldata", default=os.path.join(_EMBODIED, "EvalData"))
    ap.add_argument("--tag", default="PR1")
    ap.add_argument("--out", default="results_pr1_mps.json")
    ap.add_argument("--evidence-out", default="EVIDENCE_PR1_mps.md")
    ap.add_argument("--skip-cpu-tests", action="store_true")
    ap.add_argument("--skip-parity", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, _EMBODIED)

    if not args.skip_cpu_tests:
        logger.info("running CPU correctness suite...")
        subprocess.run(
            [sys.executable, "-m", "tests.test_batched_generate"],
            cwd=_EMBODIED, check=True,
        )

    import pandas as pd

    from repro import diagnose_row, load_overlaid_worker, sample_images, semantic_parity
    from repro import run_benchmark
    from repro.combine import (
        detection_table, eject_table, grounded_table, length_buckets_table,
        speed_plot, write_doc,
    )

    logger.info("loading worker on %s ...", args.device)
    worker, snap = load_overlaid_worker(device=args.device)

    if not args.skip_parity:
        logger.info("semantic-parity smoke check (4 samples, single vs predict_batch)...")
        imgs, prompts = sample_images(4)
        parity = semantic_parity(worker, imgs, prompts)
        df = pd.DataFrame(parity["rows"])
        print(df.to_string(index=False))
        ok = parity["slow_hybrid_pass"] and parity["fast_struct_pass"]
        print("\nPARITY OK" if ok else "\nPARITY FAILED")
        print("slow/hybrid: same detections, coords within tol:", parity["slow_hybrid_pass"])
        print("fast: same detections (structure only):         ", parity["fast_struct_pass"])

        outliers = [r for r in parity["rows"] if (r["max_coord_diff"] or 0) > 5]
        print(f"\nOutlier rows (max_coord_diff > 5): {len(outliers)}")
        for r in outliers:
            raw = next(x for x in parity["raw"] if x["mode"] == r["mode"] and x["row"] == r["row"])
            print(f"\n  {r['mode']} row {r['row']} (max_coord_diff={r['max_coord_diff']}):")
            for b in diagnose_row(raw["single"], raw["batched"]):
                print(f"    box {b['box']}: {b['verdict']}")
                print(f"      single={b['single']}  batched={b['batched']}  diffs={b['diffs']}")
                print(f"      -> {b['note']}")

    logger.info(
        "running benchmark (tag=%s, tasks=%s, datasets=%s, limit=%d, batch_size=%d)...",
        args.tag, args.tasks, args.datasets, args.limit, args.batch_size,
    )
    rec = run_benchmark(
        worker, tag=args.tag,
        metadata={"device": args.device},
        tasks=tuple(args.tasks),
        datasets=tuple(args.datasets),
        limit=args.limit, batch_size=args.batch_size,
        evaldata=args.evaldata, out=args.out,
    )
    res = {rec["tag"]: rec}

    if "eval" in args.tasks:
        print("\nDetection (batched vs B=1):\n" + detection_table(res))
        print("\nOutput-length spread + early-eject A/B:\n" + eject_table(res))
        print("\n" + length_buckets_table(res))
    if "speed" in args.tasks:
        print("\nThroughput sweep (sequential vs batched):\n" + speed_plot(res))
    if "grounded" in args.tasks:
        print("\nOne image, many queries:\n" + grounded_table(res))

    write_doc(res, path=args.evidence_out)


if __name__ == "__main__":
    main()
