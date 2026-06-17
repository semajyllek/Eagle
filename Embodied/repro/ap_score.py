# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
"""Offline mAP scoring for the ``compile_ap`` pred files, driven from the notebook.

The repo's detection metric is a two-step offline pipeline
(``convert_coco_lvis_to_standard_format.py`` -> ``coco_lvis_metric.py``, the latter
using the ``fastevaluate`` package, not pycocotools). ``score_compile_ap`` runs that
pipeline on each ``preds_*.jsonl`` written by ``compile_ap_probe`` and parses the
printed metrics, so the benchmark reports ``mAP(eager)`` vs ``mAP(compiled)`` inline
-- everything in Colab, no manual step, all-public data. Best-effort: any failure
(missing ``fastevaluate``/GT, script change) is recorded as ``error`` and the pred
files are still there to score by hand.
"""
import logging
import os
import re
import subprocess
import sys

logger = logging.getLogger(__name__)


def _evaluation_dir():
    return os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "evaluation")
    )


def _fastevaluate_usable():
    """True iff a CLEAN subprocess can import a BUILT ``fastevaluate`` (has
    ``evaluate``). A clean subprocess (cwd=/tmp) avoids the namespace-package false
    positive: ``evaluation/`` is on the parent's ``sys.path`` (the eval task adds
    it), so an in-process ``import fastevaluate`` resolves the un-built build dir as
    an empty namespace package and lies about availability -- but the metric runs in
    its own subprocess, which is what actually has to import it."""
    check = subprocess.run(
        [sys.executable, "-c", "import fastevaluate; assert hasattr(fastevaluate, 'evaluate')"],
        capture_output=True, text=True, errors="replace", cwd="/tmp",
    )
    return check.returncode == 0


def ensure_fastevaluate():
    """The repo metric needs the ``fastevaluate`` **C++ extension** -- it is NOT a
    PyPI package; it must be compiled from ``evaluation/fastevaluate/`` (see that
    dir's README/setup.py). Build + install it from the in-repo source if a clean
    subprocess can't already import the built module. Returns True if usable after."""
    if _fastevaluate_usable():
        return True
    src = os.path.join(_evaluation_dir(), "fastevaluate")
    if not os.path.isdir(src):
        logger.warning("fastevaluate source not found at %s", src)
        return False
    logger.info("building fastevaluate C++ extension from %s", src)
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", src],
            check=True, capture_output=True, text=True, errors="replace",
        )
    except subprocess.CalledProcessError as e:
        logger.warning("fastevaluate build failed: %s", (e.stderr or e.stdout or "")[-400:])
        return False
    return _fastevaluate_usable()


# coco_lvis_metric.py prints a "Metric | Value" table; pull the headline figures.
_METRIC_LABELS = {
    "avg_precision": "Avg Precision",
    "avg_recall": "Avg Recall",
    "precision50": "Precision@0.50",
    "precision95": "Precision@0.95",
}


def _parse_metrics(stdout):
    out = {}
    for key, label in _METRIC_LABELS.items():
        m = re.search(re.escape(label) + r"\s*\|\s*([0-9.]+)", stdout)
        if m:
            out[key] = float(m.group(1))
    return out


def score_pred_file(pred_jsonl, gt_json, eval_type, workdir):
    """Run convert -> coco_lvis_metric on one pred file; return parsed metrics
    (``avg_precision`` etc.) or ``{'error': ...}``."""
    ed = _evaluation_dir()
    convert = os.path.join(ed, "utils", "convert_coco_lvis_to_standard_format.py")
    metric = os.path.join(ed, "metrics", "coco_lvis_metric.py")
    os.makedirs(workdir, exist_ok=True)
    tsv = os.path.join(workdir, os.path.basename(pred_jsonl).replace(".jsonl", ".tsv"))
    try:
        subprocess.run(
            [sys.executable, convert, "--our_pred_jsonl", pred_jsonl,
             "--coco_json", gt_json, "--out_tsv", tsv, "--positive_only"],
            check=True, capture_output=True, text=True, errors="replace", cwd=ed,
        )
        r = subprocess.run(
            [sys.executable, metric, "--gt", gt_json, "--pred_tsv", tsv,
             "--eval_type", eval_type],
            check=True, capture_output=True, text=True, errors="replace", cwd=ed,
        )
        m = _parse_metrics(r.stdout)
        return m or {"error": "no metrics parsed", "stdout_tail": r.stdout[-300:]}
    except subprocess.CalledProcessError as e:
        where = os.path.basename(e.cmd[1]) if len(e.cmd) > 1 else "metric"
        return {"error": f"{where}: {(e.stderr or e.stdout or '')[-300:]}".strip()}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def score_compile_ap(result, gt_json, eval_type, workdir):
    """Fill mAP into a ``compile_ap_probe`` result in place: ``result['map_eager']``
    and ``cfg['map']`` per config. Best-effort -- records ``score_error`` if the
    metric can't run at all; the pred files remain for manual scoring."""
    if not gt_json:
        result["score_error"] = "GT json unavailable"
        return result
    if not ensure_fastevaluate():
        result["score_error"] = "fastevaluate unavailable"
        return result
    result["map_eager"] = score_pred_file(
        result["files"]["eager"], gt_json, eval_type, workdir
    )
    for cfg in result.get("configs", []):
        pf = cfg.get("pred_file")
        if pf:
            cfg["map"] = score_pred_file(pf, gt_json, eval_type, workdir)
    return result
