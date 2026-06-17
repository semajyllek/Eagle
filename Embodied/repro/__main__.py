# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
"""CLI: ``python -m repro cpu`` (no GPU) or ``gpu`` (real model).

Both write a markdown evidence report. Run from the ``Embodied/`` directory.
"""
import argparse
import logging
import os

logger = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser(
        description="batched-inference verification / evidence"
    )
    ap.add_argument(
        "mode",
        choices=["cpu", "gpu"],
        help="cpu = fake-LM correctness (no GPU); gpu = real-model evidence",
    )
    ap.add_argument("--model", default="nvidia/LocateAnything-3B")
    ap.add_argument(
        "--snapshot", default=None, help="use an already-downloaded snapshot dir"
    )
    ap.add_argument("--report", default="repro_evidence.md")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(
        level=args.log_level.upper(), format="%(levelname)s %(name)s: %(message)s"
    )

    from .cpu_evidence import cpu_evidence
    from .report import write_report

    if args.mode == "cpu":
        cpu = cpu_evidence()
        logger.info(
            "parity %d/%d, batch-independence %d/%d, early-eject %d/%d  passed=%s",
            cpu["totals"]["parity_pass"],
            cpu["totals"]["parity_checks"],
            cpu["totals"]["batch_indep_pass"],
            cpu["totals"]["batch_indep_checks"],
            cpu["totals"]["eject_pass"],
            cpu["totals"]["eject_checks"],
            cpu["passed"],
        )
        write_report(args.report, cpu=cpu)
    else:
        from .model import load_overlaid_worker
        from .report import gpu_evidence

        worker, _ = load_overlaid_worker(
            model_id=args.model, snapshot_dir=args.snapshot
        )
        write_report(args.report, cpu=cpu_evidence(), gpu=gpu_evidence(worker))
    logger.info("wrote %s", os.path.abspath(args.report))


if __name__ == "__main__":
    main()
