# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
"""
Eval parity: same task score *before* (batch_size=1) and *after* (batched).

Runs the repo's detection inference on a subset of a dataset twice — once
one-image-at-a-time (what `main` does today) and once batched — and writes both
prediction files in the exact format `coco_lvis_metric.py` expects. Score each
with the repo's normal metric pipeline and the two numbers should match.

Both runs use **greedy** decoding (`temperature=0`); the real eval samples
(`temperature=0.7`), which would make before/after differ by sampling noise
rather than batching, so greedy is required to isolate the batching effect. The
absolute AP under greedy may differ from the published sampled number — the point
here is *before == after*, not reproducing the headline score.

Usage (run from the `evaluation/` directory, model loaded from a local dir with
the patched code overlaid, or any checkout where the code is the model code):

    python eval_parity.py \
        --model_path /path/to/model \
        --test_jsonl /path/to/EvalData/_annotations/box_eval/COCO.jsonl \
        --image_root /path/to/EvalData \
        --limit 100 --batch_size 8 \
        --out_dir ./parity_out

then score each file with the repo's metric:

    python utils/convert_coco_lvis_to_standard_format.py \
        --our_pred_jsonl parity_out/preds_before.jsonl --coco_json <gt>.json \
        --out_tsv parity_out/before.tsv --positive_only
    python metrics/coco_lvis_metric.py --gt <gt>.json --pred_tsv parity_out/before.tsv
    # ... repeat for preds_after.jsonl ...
"""
import argparse
import json
import logging
import os
import sys
import time

import statistics

import torch
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _eval_helpers():
    """Lazily import the repo's own eval helpers from ``../evaluation`` (kept out
    of module scope so ``import repro`` doesn't require the evaluation deps —
    needed only when actually running the dataset eval)."""
    evaldir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "evaluation"
    )
    if evaldir not in sys.path:
        sys.path.insert(0, evaldir)
    from inference_compat import apply_chat_template, process_vision_info
    from inference_detection_ddp import load_test_data, parse_prediction

    return apply_chat_template, process_vision_info, load_test_data, parse_prediction


def build_question(categories):
    cats = "</c>".join(categories)
    return f"Locate all the instances that matches the following description: {cats}."


def _len_stats(values):
    """min/median/p90/max/mean + the straggler ratio (max/median), or {} if
    empty. Backs the output-length buckets (early-eject argument)."""
    s = sorted(values)
    if not s:
        return {}
    return dict(
        min=s[0],
        median=int(statistics.median(s)),
        p90=s[min(len(s) - 1, int(0.9 * len(s)))],
        max=s[-1],
        mean=round(sum(s) / len(s), 1),
        max_over_median=round(s[-1] / max(1, statistics.median(s)), 1),
    )


def _out_len_buckets(token_lens, edges=(0, 50, 150, 400, 1000, 10**9)):
    """Bucket per-sample output-token lengths -> {label: count} + summary stats,
    so the early-eject argument is grounded in the actual length spread (stragglers
    only matter when lengths vary)."""
    labels = [
        "%d-%d" % (edges[i], edges[i + 1] - 1)
        if edges[i + 1] < 10**9
        else "%d+" % edges[i]
        for i in range(len(edges) - 1)
    ]
    counts = {lab: 0 for lab in labels}
    for n in token_lens:
        for i in range(len(edges) - 1):
            if edges[i] <= n < edges[i + 1]:
                counts[labels[i]] += 1
                break
    return {"buckets": counts, "stats": _len_stats(token_lens)}


def to_record(sample, output):
    _, _, _, parse_prediction = _eval_helpers()
    w, h = sample["image"].size
    try:
        preds = parse_prediction(output, w, h)
    except Exception as e:
        logger.warning("parse failed (%s)", e)
        preds = {}
    return {
        "image_path": sample["image_path"],
        "extracted_predictions": preds,
        "gt": sample["gt"],
        "question": build_question(sample["categories"]),
        "dataset_name": sample["dataset_name"],
        "raw_response": output,
        "task_name": sample["task_name"],
    }


def load_samples(test_jsonl, image_root, limit):
    _, _, load_test_data, _ = _eval_helpers()
    data = load_test_data(test_jsonl)[:limit]
    out, missing = [], 0
    for e in tqdm(data, desc="loading images"):
        path = (
            os.path.join(image_root, e["image_path"]) if image_root else e["image_path"]
        )
        if not os.path.exists(path):
            missing += 1
            continue
        e = dict(e)
        e["image"] = Image.open(path).convert("RGB")
        out.append(e)
    if missing:
        logger.warning("%d/%d images missing under %s", missing, len(data), image_root)
    return out


def agreement(before, after):
    """Quick GT-free parity signal: same #boxes per category and median coord
    diff (median, not max, so a single outlier box doesn't dominate)."""

    def boxes(rec):
        b = {}
        for cat, lst in rec["extracted_predictions"].items():
            b[cat] = [tuple(round(x, 1) for x in box) for box in lst]
        return b

    same_struct, diffs = 0, []
    for ra, rb in zip(before, after):
        ba, bb = boxes(ra), boxes(rb)
        if set(ba) != set(bb) or any(len(ba[c]) != len(bb.get(c, [])) for c in ba):
            continue
        same_struct += 1
        for c in ba:
            for x, y in zip(ba[c], bb[c]):
                diffs.extend(abs(a - b) for a, b in zip(x, y))
    median_diff = statistics.median(diffs) if diffs else 0.0
    return same_struct, len(before), median_diff


def _merge_profile(acc, batch_profile):
    """Accumulate one batch's ``profile`` dict (see ``batched_generate``'s
    ``_tic``/``_toc``) into the running totals in ``acc``."""
    for key in (
        "step0_eject",
        "step1_assemble",
        "step2_mask",
        "step3_forward",
        "step4_sample",
        "step5_compact",
    ):
        acc[key] = acc.get(key, 0.0) + batch_profile.get(key, 0.0)
    acc["n_steps"] = acc.get("n_steps", 0) + batch_profile.get("n_steps", 0)
    acc.setdefault("A_history", []).extend(batch_profile.get("A_history", []))


def _summarize_profile(acc):
    """Reduce an accumulated profile dict to a compact, JSON-friendly summary."""
    a_hist = acc.get("A_history", [])
    out = {k: round(v, 3) for k, v in acc.items() if k not in ("A_history", "n_steps")}
    out["n_steps"] = acc.get("n_steps", 0)
    out["mean_A"] = round(sum(a_hist) / len(a_hist), 2) if a_hist else None
    out["total_s"] = round(
        sum(acc.get(k, 0.0) for k in (
            "step0_eject", "step1_assemble", "step2_mask",
            "step3_forward", "step4_sample", "step5_compact",
        )),
        3,
    )
    return out


def _batched_pass(
    worker,
    samples,
    batch_size,
    generation_mode,
    max_new_tokens,
    desc,
    return_batch_times=False,
    profile=None,
):
    """Run the whole set in batches of ``batch_size`` via the real
    ``worker.predict_batch``; return (records, seconds), or (records, seconds,
    batch_times) if ``return_batch_times``.

    If ``profile`` is an (empty) dict, it is filled in-place with the
    per-step timing breakdown (see ``_merge_profile``/``_summarize_profile``)
    accumulated across every batch.
    """
    out_recs, batch_times, t1 = [], [], time.time()
    for i in tqdm(range(0, len(samples), batch_size), desc=desc):
        chunk = samples[i : i + batch_size]
        images = [s["image"] for s in chunk]
        questions = [build_question(s["categories"]) for s in chunk]
        tb0 = time.time()
        batch_profile = {} if profile is not None else None
        outs = worker.predict_batch(
            images,
            questions,
            generation_mode=generation_mode,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            profile=batch_profile,
        )
        batch_times.append(time.time() - tb0)
        out_recs.extend(to_record(s, o) for s, o in zip(chunk, outs))
        if profile is not None:
            _merge_profile(profile, batch_profile)
    total = time.time() - t1
    if return_batch_times:
        return out_recs, total, batch_times
    return out_recs, total


def run_dataset(
    worker,
    samples,
    dataset,
    batch_size,
    generation_mode,
    max_new_tokens,
    out_dir,
    eject_ab=True,
):
    """Run before (B=1) vs after (batched) over ``samples`` and return a summary
    dict; also writes ``preds_before/after.jsonl`` + ``summary.json`` to ``out_dir``
    for offline AP scoring with the repo metric.

    Before is ``worker.predict`` (B=1, what `main` calls today) and after is
    ``worker.predict_batch`` (what PR1/PR2 add) — the actual shipped API, called
    the way a real caller would, with no separate `model.generate` reimplementation.
    Both default to the same ``repetition_penalty``/sampling config, so "before ==
    after" isolates batching, not a config drift between the two passes.

    Also records the **output-length distribution** (buckets + stats) — stragglers,
    and thus early-eject's value, only exist when lengths vary — and, when
    ``eject_ab``, times the batched pass **with and without early-eject** so the
    straggler-removal win is measured on this dataset's real length spread.
    """
    tokenizer = worker.tokenizer

    def _tok_len(text):
        return len(tokenizer(text, add_special_tokens=False).input_ids)

    # BEFORE: one image at a time (what main can do today) — the sequential
    # baseline the batched path is compared against.
    before, before_times = [], []
    for s in tqdm(samples, desc=f"{dataset} before (B=1)"):
        q = build_question(s["categories"])
        t0 = time.time()
        out = worker.predict(
            s["image"],
            q,
            generation_mode=generation_mode,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            verbose=False,
        )["answer"]
        before_times.append(time.time() - t0)
        before.append(to_record(s, out))
    before_s = sum(before_times)

    # AFTER: batched.
    after_profile = {}
    after, after_s = _batched_pass(
        worker,
        samples,
        batch_size,
        generation_mode,
        max_new_tokens,
        f"{dataset} after (batched)",
        profile=after_profile,
    )

    noeject_s = None
    noeject_profile = {}
    noeject = None

    os.makedirs(out_dir, exist_ok=True)
    pred_files = [("preds_before.jsonl", before), ("preds_after.jsonl", after)]
    if noeject is not None:
        pred_files.append(("preds_after_noeject.jsonl", noeject))
    for name, recs in pred_files:
        # ensure_ascii=True (default): LVIS outputs can carry lone byte-fallback
        # surrogates that ensure_ascii=False can't serialize; ascii-escaping yields a
        # pure-ASCII, codec-proof file the metric's convert step reads as utf-8.
        with open(os.path.join(out_dir, name), "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")

    out_lens = [_tok_len(r["raw_response"]) for r in after]
    out_lens_noeject = (
        [_tok_len(r["raw_response"]) for r in noeject] if noeject is not None else None
    )

    # Same-code, eject-on-vs-off divergence: greedy decoding can fork onto a
    # different token sequence once a row's batch width A changes (eject
    # shrinks A over time; no-eject keeps it constant), since GPU kernels for
    # different A aren't bit-identical and argmax can flip on a near-tie. This
    # flags any sample whose decoded text differs between the two passes and
    # the first token index where the two sequences diverge.
    eject_divergence = None
    if noeject is not None:
        diffs = []
        for i, (a_rec, n_rec) in enumerate(zip(after, noeject)):
            a_text, n_text = a_rec["raw_response"], n_rec["raw_response"]
            if a_text == n_text:
                continue
            ta = tokenizer(a_text, add_special_tokens=False).input_ids
            tb = tokenizer(n_text, add_special_tokens=False).input_ids
            first = next(
                (i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), min(len(ta), len(tb))
            )
            diffs.append(
                {
                    "sample_idx": i,
                    "image_path": a_rec["image_path"],
                    "len_eject": len(ta),
                    "len_noeject": len(tb),
                    "first_diff_token": first,
                }
            )
        eject_divergence = {"n_diff": len(diffs), "samples": diffs}

    n_same, n_tot, median_diff = agreement(before, after)
    summary = {
        "dataset": dataset,
        "samples": len(samples),
        "batch_size": batch_size,
        "generation_mode": generation_mode,
        "max_new_tokens": max_new_tokens,
        "before_s": round(before_s, 1),
        "after_s": round(after_s, 1),
        "speedup_x": round(before_s / after_s, 2) if after_s else None,
        "after_noeject_s": round(noeject_s, 1) if noeject_s else None,
        "eject_speedup_x": round(noeject_s / after_s, 2)
        if noeject_s and after_s
        else None,
        "parity_same": n_same,
        "parity_total": n_tot,
        "median_coord_diff_px": round(median_diff, 1),
        "output_len": _out_len_buckets(out_lens),
        "output_len_noeject": _out_len_buckets(out_lens_noeject) if out_lens_noeject else None,
        "eject_divergence": eject_divergence,
        "after_profile": _summarize_profile(after_profile),
        "after_noeject_profile": _summarize_profile(noeject_profile) if eject_ab else None,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("%s summary: %s", dataset, summary)
    return summary


def eval_parity(
    worker,
    datasets=("COCO", "LVIS"),
    evaldata="/content/EvalData",
    limit=50,
    batch_size=8,
    generation_mode="hybrid",
    max_new_tokens=2048,
    out_root="./parity_out",
    download=True,
    eject_ab=True,
):
    """Task-metric parity on real data, **reusing an already-loaded worker** (no
    second model load). Downloads the COCO/LVIS eval data (cached), then for each
    dataset runs the repo's detection inference once one-image-at-a-time (B=1, what
    `main` does) and once batched, on ``limit`` samples.

    Returns a list of per-dataset summary dicts (timing, speedup, GT-free
    detection parity, output-length distribution, and the early-eject A/B — see
    ``run_dataset``) ready for ``repro.results_table.to_markdown``. Pred files
    are written under ``out_root/<dataset>/`` for offline AP scoring with the
    repo metric.
    """
    if download:
        from .eval_data import download_eval_data

        download_eval_data(datasets, dest=evaldata)

    summaries = []
    for ds in datasets:
        jsonl = os.path.join(evaldata, "_annotations", "box_eval", f"{ds}.jsonl")
        if not os.path.exists(jsonl):
            logger.warning("missing %s; skipping %s", jsonl, ds)
            continue
        samples = load_samples(jsonl, evaldata, limit)
        if not samples:
            logger.warning("no usable samples for %s; skipping", ds)
            continue
        logger.info(
            "%s: %d samples; mode=%s (greedy)", ds, len(samples), generation_mode
        )
        summaries.append(
            run_dataset(
                worker,
                samples,
                ds,
                batch_size,
                generation_mode,
                max_new_tokens,
                os.path.join(out_root, ds.lower()),
                eject_ab=eject_ab,
            )
        )
    return summaries


def _group_queries_by_image(samples, max_per_image):
    """Collapse samples into one-image -> many-queries groups by ``image_path``,
    each distinct query (referring expression / category) becoming one prompt.

    This is the workload where PR2 (encode the shared image once) helps. For a
    referring-expression split like ``RefCOCOg_val`` the queries are *real* — each
    record is a distinct natural-language expression on the same image; for a plain
    detection split it falls back to per-category queries. Returns
    ``[(image, image_path, [query, ...]), ...]`` with >= 2 queries each."""
    by_path = {}
    for s in samples:
        key = s["image_path"]
        if key not in by_path:
            by_path[key] = {"image": s["image"], "cats": []}
        for c in s["categories"]:
            if c not in by_path[key]["cats"]:
                by_path[key]["cats"].append(c)
    groups = []
    for path, g in by_path.items():
        cats = g["cats"][:max_per_image]
        if len(cats) < 2:
            continue
        qs = [
            "Locate all the instances that matches the following description: " f"{c}."
            for c in cats
        ]
        groups.append((g["image"], path, qs))
    return groups


@torch.inference_mode()
def grounded_eval(
    worker,
    datasets=("RefCOCOg_val",),
    evaldata="/content/EvalData",
    limit=400,
    max_per_image=8,
    generation_mode="hybrid",
    max_new_tokens=512,
    out_root="./grounded_out",
    download=True,
):
    """One image, **many** queries — the workload PR1's batched decode and PR2's
    encode-once both target. Defaults to the **RefCOCOg** referring-expression
    split (each image has several distinct natural-language expressions, each
    localizing an object — a genuine one-image-many-queries workload). Groups a
    dataset by image (each category -> its own grounding query), then times the
    same work two ways per image:

    * **single** - sequential ``worker.predict`` per query (what `main` does).
    * **batch**  - ``worker.predict_batch([img] * len(qs), qs)`` — ``len(qs)``
      references to the *same* image object, paired with ``len(qs)`` distinct
      prompts: this branch's own ``predict_batch``, called the way a real
      one-image-many-queries caller naturally would.

    This is **branch-isolated**: on a flat (PR1) worker, ``predict_batch`` has no
    vision dedup, so the repeated object is re-encoded once per row (still
    batched); on a grouped (PR2) worker, ``feat_cache`` keys on ``id(img)``, so
    the repeated object is encoded once and reused across all rows. Same call,
    each branch's real code path — ``combine`` puts PR1's ``speedup_x`` and PR2's
    ``speedup_x`` for this identical call side by side.
    """
    import time as _time

    if download:
        from .eval_data import download_eval_data

        download_eval_data(datasets, dest=evaldata)

    summaries = []
    for ds in datasets:
        jsonl = os.path.join(evaldata, "_annotations", "box_eval", f"{ds}.jsonl")
        if not os.path.exists(jsonl):
            logger.warning("missing %s; skipping %s", jsonl, ds)
            continue
        groups = _group_queries_by_image(
            load_samples(jsonl, evaldata, limit), max_per_image
        )
        if not groups:
            logger.warning("%s: no multi-query images; skipping", ds)
            continue
        logger.info(
            "%s: %d one-image-many-query groups (mean %.1f q/img)",
            ds,
            len(groups),
            sum(len(q) for _, _, q in groups) / len(groups),
        )

        # warmup
        warmup_qs = groups[0][2][:2]
        worker.predict_batch(
            [groups[0][0]] * len(warmup_qs), warmup_qs, max_new_tokens=32, temperature=0.0
        )
        t_single = t_batch = 0.0
        for img, _, qs in tqdm(groups, desc=f"{ds} grounded (single vs batch)"):
            t0 = _time.time()
            for q in qs:
                worker.predict(
                    img,
                    q,
                    generation_mode=generation_mode,
                    max_new_tokens=max_new_tokens,
                    temperature=0.0,
                    verbose=False,
                )
            t_single += _time.time() - t0
            t1 = _time.time()
            worker.predict_batch(
                [img] * len(qs),
                qs,
                generation_mode=generation_mode,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
            )
            t_batch += _time.time() - t1

        summary = {
            "dataset": ds,
            "task": "grounded_one_image_many_queries",
            "images": len(groups),
            "mean_queries_per_image": round(
                sum(len(q) for _, _, q in groups) / len(groups), 1
            ),
            "single_s": round(t_single, 1),
            "batch_s": round(t_batch, 1),
            "speedup_x": round(t_single / t_batch, 2) if t_batch else None,
        }
        os.makedirs(os.path.join(out_root, ds.lower()), exist_ok=True)
        with open(
            os.path.join(out_root, ds.lower(), "grounded_summary.json"), "w"
        ) as f:
            json.dump(summary, f, indent=2)
        logger.info("%s grounded summary: %s", ds, summary)
        summaries.append(summary)
    return summaries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--test_jsonl", required=True)
    ap.add_argument("--image_root", default="")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument(
        "--generation_mode", default="hybrid", choices=["fast", "slow", "hybrid"]
    )
    ap.add_argument("--max_new_tokens", type=int, default=2048)
    ap.add_argument("--out_dir", default="./parity_out")
    ap.add_argument(
        "--dataset", default="", help="label for the summary (default: jsonl stem)"
    )
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(
        level=args.log_level.upper(), format="%(levelname)s %(message)s"
    )
    dataset = args.dataset or os.path.splitext(os.path.basename(args.test_jsonl))[0]

    embodied = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    if embodied not in sys.path:
        sys.path.insert(0, embodied)
    from locateanything_worker import LocateAnythingWorker

    worker = LocateAnythingWorker(args.model_path, device="cuda")

    samples = load_samples(args.test_jsonl, args.image_root, args.limit)
    logger.info(
        "loaded %d samples; mode=%s (greedy)", len(samples), args.generation_mode
    )
    run_dataset(
        worker,
        samples,
        dataset,
        args.batch_size,
        args.generation_mode,
        args.max_new_tokens,
        args.out_dir,
    )
    logger.info(
        "For AP: score preds_before/after.jsonl with the repo metric "
        "(convert_coco_lvis_to_standard_format.py then coco_lvis_metric.py)."
    )


if __name__ == "__main__":
    main()
