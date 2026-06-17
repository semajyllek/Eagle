# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
"""CPU correctness evidence — no GPU or checkpoint required.

Wraps ``tests.test_batched_generate.evaluate`` (a deterministic fake LM honouring
the model's exact mask/KV/position contract) into report-ready tables: batched
decode is integer-exact vs an independent re-implementation of the original loop,
and batch-independent, across fast/slow/hybrid.
"""
import os
import sys

_EMBODIED = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, _EMBODIED)


def cpu_evidence(seeds=range(12)):
    """Return ``{'by_mode', 'coverage', 'totals', 'passed'}`` from the fake-LM suite.

    ``by_mode`` / ``coverage`` are lists of dicts (DataFrame-ready); ``totals`` are
    aggregate parity / batch-independence counts; ``passed`` is the overall bool.
    """
    from tests.test_batched_generate import evaluate

    rep = evaluate(seeds=seeds)
    detail = rep["detail"]

    modes = sorted({r["mode"] for r in detail})
    by_mode = []
    for m in modes:
        sub = [r for r in detail if r["mode"] == m]
        pc = sum(r["parity_checks"] for r in sub)
        pp = sum(r["parity_pass"] for r in sub)
        ic = sum(r["batch_indep_checks"] for r in sub)
        ip = sum(r["batch_indep_pass"] for r in sub)
        ec = sum(r["eject_checks"] for r in sub)
        ep = sum(r["eject_pass"] for r in sub)
        by_mode.append(
            dict(
                mode=m,
                parity_checks=pc,
                parity_pass=pp,
                batch_indep_checks=ic,
                batch_indep_pass=ip,
                eject_checks=ec,
                eject_pass=ep,
                parity_pct=round(100 * pp / pc, 1),
                batch_indep_pct=round(100 * ip / ic, 1),
                eject_pct=round(100 * ep / ec, 1),
            )
        )

    coverage = sorted(
        [
            {"decode_pattern": k[0], "tokens_emitted": k[1], "count": v}
            for k, v in rep["coverage"].items()
        ],
        key=lambda d: -d["count"],
    )

    totals = dict(
        parity_checks=sum(r["parity_checks"] for r in detail),
        parity_pass=sum(r["parity_pass"] for r in detail),
        batch_indep_checks=sum(r["batch_indep_checks"] for r in detail),
        batch_indep_pass=sum(r["batch_indep_pass"] for r in detail),
        eject_checks=sum(r["eject_checks"] for r in detail),
        eject_pass=sum(r["eject_pass"] for r in detail),
    )
    passed = (
        totals["parity_pass"] == totals["parity_checks"]
        and totals["batch_indep_pass"] == totals["batch_indep_checks"]
        and totals["eject_pass"] == totals["eject_checks"]
    )
    return {
        "by_mode": by_mode,
        "coverage": coverage,
        "totals": totals,
        "passed": passed,
    }
