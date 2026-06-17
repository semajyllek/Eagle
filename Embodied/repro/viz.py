# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
"""Post-hoc parity visualization — strictly separate from the timed benchmark.

Reads the prediction ``.jsonl`` files the benchmark *already* writes
(``compile_ap``'s ``preds_eager.jsonl`` / ``preds_<compiled>.jsonl``, or the eval's
``preds_before.jsonl`` / ``preds_after.jsonl``) and draws the detected boxes + ``<ref>``
labels for the two passes side by side, with the ground-truth boxes (dashed lime)
overlaid on both. So "parity = same detections within tolerance" becomes something you
can *see* -- against each other *and* against the target -- and ``only_diffs=True``
surfaces exactly the images where the two passes disagree (the near-ties behind the
~1% mAP delta).

Nothing here runs inside ``run_benchmark`` — it only reads outputs, so it cannot perturb
any timing. Call it from a separate cell after the run.

    from repro.viz import parity_overlay
    d = "/content/EvalData/compile_ap_out/COCO"
    parity_overlay(f"{d}/preds_eager.jsonl", f"{d}/preds_default_dyn.jsonl",
                   image_root="/content/EvalData", labels=("eager", "compiled"),
                   n=4, only_diffs=True)
"""
import json
import os

from .parity import coord_diff, parse_boxes, parse_refs

_COLORS = ["#e6194B", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
           "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990"]


def _load(jsonl):
    with open(jsonl, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _dets(rec):
    """(boxes_0_1000, labels) for a to_record dict, from its raw_response."""
    raw = rec.get("raw_response", "") or ""
    return parse_boxes(raw), parse_refs(raw)


def _gt_dets(rec):
    """Ground-truth (boxes_abs_px, labels) from a to_record dict's ``gt`` field.

    Detection ``gt`` is ``{category: [[x1,y1,x2,y2], ...]}`` in *absolute pixel*
    coords (the metric IoUs it directly against ``extracted_predictions``) -- unlike
    ``raw_response`` boxes, which are normalized 0-1000. So GT is drawn as-is, with no
    /1000 scaling. Tolerates the grounding shapes too (a bare ``[x1,y1,x2,y2]`` or a
    list of boxes) by treating them as unlabeled."""
    gt = rec.get("gt")
    boxes, labels = [], []
    if isinstance(gt, dict):
        for cat, blist in gt.items():
            for b in (blist or []):
                if isinstance(b, (list, tuple)) and len(b) == 4:
                    boxes.append(list(b))
                    labels.append(str(cat))
    elif isinstance(gt, (list, tuple)):
        if len(gt) == 4 and all(isinstance(v, (int, float)) for v in gt):
            boxes.append(list(gt))
            labels.append("")
        else:
            for b in gt:
                if isinstance(b, (list, tuple)) and len(b) == 4:
                    boxes.append(list(b))
                    labels.append("")
    return boxes, labels


def _label_for(i, labels):
    if not labels:
        return ""
    return labels[i] if i < len(labels) else labels[-1]  # 1 ref -> N boxes: reuse it


_GT_COLOR = "#00ff00"


def _draw(ax, img, boxes, labels, title, gt=None):
    import matplotlib.patches as patches

    w, h = img.size
    ax.imshow(img)
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    # GT first (underneath), absolute px, dashed lime so it reads as "the target".
    if gt:
        gboxes, glabels = gt
        for i, b in enumerate(gboxes):
            if len(b) != 4:
                continue
            x1, y1, x2, y2 = b
            ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                           edgecolor=_GT_COLOR, linewidth=1.5,
                                           linestyle="--"))
            lab = _label_for(i, glabels)
            if lab:
                ax.text(x2, max(0, y1 - 2), f"gt:{lab}", color="black", fontsize=6,
                        ha="right", bbox=dict(facecolor=_GT_COLOR, edgecolor="none", pad=1))
    # Predictions on top, normalized 0-1000 -> px.
    for i, b in enumerate(boxes):
        if len(b) != 4:
            continue
        x1, y1, x2, y2 = (b[0] / 1000.0 * w, b[1] / 1000.0 * h,
                          b[2] / 1000.0 * w, b[3] / 1000.0 * h)
        c = _COLORS[i % len(_COLORS)]
        ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                       fill=False, edgecolor=c, linewidth=2))
        lab = _label_for(i, labels)
        if lab:
            ax.text(x1, max(0, y1 - 2), lab, color="white", fontsize=7,
                    bbox=dict(facecolor=c, edgecolor="none", pad=1))


def _differs(ra, rb, tol):
    """Detection-level disagreement (the eval/compile gate): different #boxes or
    labels, or coords drifting > tol/1000."""
    cd = coord_diff(ra, rb)
    return not (cd is not None and parse_refs(ra) == parse_refs(rb) and cd <= tol)


def parity_overlay(jsonl_a, jsonl_b, image_root="", labels=("A", "B"),
                   n=4, only_diffs=False, tol=5, scale=4, show_gt=True):
    """Side-by-side box/label overlays for two prediction passes. Returns the figure.

    ``jsonl_a``/``jsonl_b``: the two ``preds_*.jsonl`` files (aligned by image_path).
    ``only_diffs``: show only images whose detections disagree at the eval's gate
    (same #boxes + ``<ref>`` labels + coords within ``tol``/1000) -- the interesting
    parity cases. ``n``: how many images. ``image_root``: prefix for ``image_path``.
    ``show_gt``: also overlay the ground-truth boxes (dashed lime) on both panels, so
    each pass is visible against the target -- not just against each other.
    """
    import matplotlib.pyplot as plt
    from PIL import Image

    A = {r["image_path"]: r for r in _load(jsonl_a)}
    B = {r["image_path"]: r for r in _load(jsonl_b)}
    keys = [k for k in A if k in B]
    if only_diffs:
        keys = [k for k in keys
                if _differs(A[k].get("raw_response", "") or "",
                            B[k].get("raw_response", "") or "", tol)]
    keys = keys[:n]
    if not keys:
        print("nothing to show" + (" — no detection-level diffs found "
              "(eager == compiled within tolerance on every image)" if only_diffs else ""))
        return None

    fig, axes = plt.subplots(len(keys), 2, figsize=(2 * scale, len(keys) * scale))
    if len(keys) == 1:
        axes = axes.reshape(1, 2)
    for row, k in enumerate(keys):
        img = Image.open(os.path.join(image_root, k)).convert("RGB")
        ba, la = _dets(A[k])
        bb, lb = _dets(B[k])
        # GT is the same target for both passes; prefer A's record, fall back to B's.
        gt = _gt_dets(A[k]) if show_gt else None
        if show_gt and not gt[0]:
            gt = _gt_dets(B[k])
        ng = f", {len(gt[0])} gt" if show_gt and gt[0] else ""
        _draw(axes[row, 0], img, ba, la,
              f"{labels[0]} — {os.path.basename(k)} ({len(ba)} box{ng})", gt=gt)
        _draw(axes[row, 1], img, bb, lb, f"{labels[1]} ({len(bb)} box)", gt=gt)
    fig.tight_layout()
    return fig
