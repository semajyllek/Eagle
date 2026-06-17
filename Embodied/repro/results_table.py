# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
"""Assemble the PR results table (accuracy + speedup) from eval_parity outputs.

`eval_parity.py` writes a ``summary.json`` per dataset (timing, speedup, GT-free
parity). Accuracy (AP) comes from scoring ``preds_before/after.jsonl`` with the
repo's metric; supply those numbers via ``--ap_json`` to fill the AP columns.

    python repro/results_table.py --out_dirs parity_coco parity_lvis \
        --ap_json ap.json > results.md

where ap.json is e.g. ``{"COCO": {"before": 41.2, "after": 41.2},
                          "LVIS": {"before": 33.8, "after": 33.7}}``.
"""
import argparse
import json
import os


def collect_summaries(out_dirs):
    rows = []
    for p in out_dirs:
        sp = p if p.endswith(".json") else os.path.join(p, "summary.json")
        with open(sp) as f:
            rows.append(json.load(f))
    return rows


def to_markdown(summaries, ap=None):
    """Render the PR table. ``ap`` is an optional {dataset: {before, after}} map."""
    ap = ap or {}
    cols = ["dataset", "samples", "AP B=1", "AP batched", "ΔAP",
            "before (s)", "batched (s)", "speedup", "detection parity"]
    out = ["| " + " | ".join(cols) + " |",
           "|" + "|".join("---" for _ in cols) + "|"]
    for s in summaries:
        a = ap.get(s["dataset"], {})
        ab, aa = a.get("before"), a.get("after")
        d_ap = round(aa - ab, 2) if (ab is not None and aa is not None) else ""
        parity = f"{s['parity_same']}/{s['parity_total']} (median {s['median_coord_diff_px']}px)"
        out.append("| " + " | ".join(str(x) for x in [
            s["dataset"], s["samples"],
            ab if ab is not None else "—", aa if aa is not None else "—", d_ap,
            s["before_s"], s["after_s"], f"{s['speedup_x']}×", parity]) + " |")
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dirs", nargs="+", required=True,
                   help="eval_parity out_dir(s) each containing summary.json")
    p.add_argument("--ap_json", default=None,
                   help='optional {"<dataset>": {"before": ap, "after": ap}, ...}')
    args = p.parse_args()
    ap = json.load(open(args.ap_json)) if args.ap_json else None
    print(to_markdown(collect_summaries(args.out_dirs), ap))


if __name__ == "__main__":
    main()
