# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
"""Detection-level parity between single-sequence and batched decoding.

Greedy GPU decoding is not bitwise-stable (bf16 GEMM non-batch-invariance), so
correctness is "same detections within tolerance", not byte-identical strings:
same number of boxes, same `<ref>` labels, coordinates within a few /1000.
"""
import re


def parse_boxes(s):
    return [tuple(map(int, re.findall(r"<(\d+)>", b))) for b in re.findall(r"<box>.*?</box>", s)]


def parse_refs(s):
    return re.findall(r"<ref>(.*?)</ref>", s)


def coord_diff(a, b):
    """Max coordinate difference if box structure matches, else None.

    Boxes can be coordinate-free (e.g. ``<box>none</box>``), so the inner max also
    needs a default to avoid ``max() of empty`` on such boxes.
    """
    pa, pb = parse_boxes(a), parse_boxes(b)
    if len(pa) != len(pb) or any(len(x) != len(y) for x, y in zip(pa, pb)):
        return None
    return max((max((abs(x - y) for x, y in zip(p, q)), default=0)
                for p, q in zip(pa, pb)), default=0)


def semantic_parity(worker, images, prompts, modes=("slow", "fast", "hybrid"),
                    max_new_tokens=256, tol=5):
    """Per-(mode, row) comparison of single-sequence vs batched decoding (greedy).

    Returns ``{"rows": [...], "raw": [...], "slow_hybrid_pass": bool,
    "fast_struct_pass": bool}``. `slow`/`hybrid` (default) must hold to `tol`/1000;
    `fast` (pure MTP, no AR verification) need only match structurally (same #boxes
    + labels). `rows` holds the scalar comparison (for tables / JSON records);
    `raw` holds the actual ``single``/``batched`` answer strings per (mode, row),
    for diagnosing *why* a row diverged (see `diagnose_row`).
    """
    rows = []
    raw = []
    for mode in modes:
        single = [worker.predict(im, q, generation_mode=mode, max_new_tokens=max_new_tokens,
                                 temperature=0.0, verbose=False)["answer"]
                  for im, q in zip(images, prompts)]
        batched = worker.predict_batch(images, prompts, generation_mode=mode,
                                       max_new_tokens=max_new_tokens, temperature=0.0,
                                       repetition_penalty=1.1)
        for i, (s, b) in enumerate(zip(single, batched)):
            rows.append({"mode": mode, "row": i, "same_num_boxes": coord_diff(s, b) is not None,
                         "same_labels": parse_refs(s) == parse_refs(b),
                         "max_coord_diff": coord_diff(s, b), "exact": s == b})
            raw.append({"mode": mode, "row": i, "single": s, "batched": b})

    def ok(ms, coords):
        sub = [r for r in rows if r["mode"] in ms]
        struct = all(r["same_num_boxes"] and r["same_labels"] for r in sub)
        within = all((r["max_coord_diff"] or 0) <= tol for r in sub) if coords else True
        return struct and within

    return {"rows": rows, "raw": raw,
            "slow_hybrid_pass": ok(("slow", "hybrid"), coords=True),
            "fast_struct_pass": ok(("fast",), coords=False)}


def diagnose_row(single, batched, tol=5):
    """Per-box breakdown of how ``single`` and ``batched`` outputs differ.

    For each positionally-zipped (single-box, batched-box) pair (the same pairing
    `coord_diff` uses), reports the two boxes, their per-coordinate absolute
    differences, and a `verdict`/`note` aimed at a reviewer asking "what do we
    think happened here":

    - **`match`** — boxes agree within `tol`.
    - **`reordered`** — *every* coordinate is above `tol`. Not a coordinate
      drift: MTP emitted same-labeled detections in a different order between
      `single` and `batched`, so this box-pair is actually two different objects
      (`same_labels` still passes, since the *set* of labels is unchanged).
    - **`coord flip (degenerate box)`** — one coordinate is above `tol`, *and*
      this box already has `x2<=x1` or `y2<=y1` (4-coord boxes are
      `(x1, x2, y1, y2)`) in `single` and/or `batched`. The underlying detection
      was already geometrically invalid/low-confidence in both runs; batching
      just picked a different one of two bad candidate values.
    - **`coord flip`** — one coordinate is above `tol` on an otherwise-matching,
      geometrically valid box: a near-tied argmax flipped under bf16 batch
      effects, shifting one edge.

    Returns ``None`` if box structure (count/arity) differs (`coord_diff` would
    also return ``None``).
    """
    pa, pb = parse_boxes(single), parse_boxes(batched)
    if len(pa) != len(pb) or any(len(p) != len(q) for p, q in zip(pa, pb)):
        return None

    def degenerate(box):
        # 4-coord boxes are (x1, x2, y1, y2); non-positive width/height
        # means this box is already geometrically invalid.
        return len(box) == 4 and (box[1] <= box[0] or box[3] <= box[2])

    out = []
    for i, (p, q) in enumerate(zip(pa, pb)):
        diffs = [abs(x - y) for x, y in zip(p, q)]
        n_above = sum(d > tol for d in diffs)
        degen = degenerate(p) or degenerate(q)
        if n_above == 0:
            verdict, note = "match", "boxes agree within tol"
        elif n_above == len(diffs) and n_above > 1:
            verdict = "reordered"
            note = ("every coordinate differs -- likely two same-labeled "
                    "detections swapped emission order, not a coordinate "
                    "drift")
        elif degen:
            verdict = "coord flip (degenerate box)"
            note = ("one coordinate differs, but this box already has "
                    "x2<=x1 or y2<=y1 in at least one run -- a "
                    "low-confidence/garbage detection in both runs; "
                    "batching just picked a different bad value")
        else:
            verdict = "coord flip"
            note = ("one coordinate differs on an otherwise-matching, "
                    "geometrically valid box -- a near-tied argmax "
                    "flipped under bf16 batch effects")
        out.append({"box": i, "single": p, "batched": q, "diffs": diffs,
                     "degenerate": degen, "verdict": verdict, "note": note})
    return out


def fast_mode_diagnostic(worker, image, question, max_new_tokens=256):
    """Show that `fast`-mode coordinate drift is padding-independent (the no-pad
    duplicate batch still differs from B=1) — i.e. bf16/MTP sensitivity, not a bug.
    """
    s = worker.predict(image, question, generation_mode="fast", max_new_tokens=max_new_tokens,
                       temperature=0.0, verbose=False)["answer"]
    dup = worker.predict_batch([image, image], [question, question], generation_mode="fast",
                               max_new_tokens=max_new_tokens, temperature=0.0,
                               repetition_penalty=1.1)[0]
    hyb = worker.predict_batch([image, image], [question, question], generation_mode="hybrid",
                               max_new_tokens=max_new_tokens, temperature=0.0,
                               repetition_penalty=1.1)[0]
    return {"fast_single": s, "fast_dup_nopad": dup, "hybrid_dup": hyb,
            "fast_dup_equals_single": s == dup}


def grouped_parity(worker, image, questions, max_new_tokens=256):
    """Grouped ``predict_batch`` (one image, encoded once) vs the same image
    repeated flat (encoded per row). They should match (same image features).
    """
    grouped = worker.predict_batch([image], [questions], generation_mode="hybrid",
                                   max_new_tokens=max_new_tokens, temperature=0.0)[0]
    repb = worker.predict_batch([image] * len(questions), questions, generation_mode="hybrid",
                                max_new_tokens=max_new_tokens, temperature=0.0,
                                repetition_penalty=1.1)
    rows = [{"row": i, "same_num_boxes": len(parse_boxes(a)) == len(parse_boxes(b)),
             "same_labels": parse_refs(a) == parse_refs(b)}
            for i, (a, b) in enumerate(zip(grouped, repb))]
    return {"rows": rows, "match": all(r["same_num_boxes"] and r["same_labels"] for r in rows)}
