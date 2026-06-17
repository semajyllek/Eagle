# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
"""Combine the per-PR benchmark records (``results_<tag>.json`` from
``repro.benchmark.run_benchmark``, one per notebook) into one evidence doc:
detection speedup/parity, the output-length distribution + the early-eject A/B,
and the PR2 one-image-many-queries showcase — each PR's own run, side by side.

    python -m repro.combine results_pr1.json results_pr2.json -o EVIDENCE.md
"""
import argparse
import json
import logging

logger = logging.getLogger(__name__)


def load_results(paths):
    out = {}
    for p in paths:
        with open(p) as f:
            r = json.load(f)
        out[r.get("tag") or r.get("ref") or p] = r
    return out


def _md_table(headers, rows):
    out = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(out)


def detection_table(results):
    """Per (ref, dataset): batched speedup over the B=1 baseline + structural
    parity. (Baseline `before` is the B=1 path, identical across refs = `main`.)"""
    headers = [
        "ref",
        "dataset",
        "samples",
        "before_s (main)",
        "after_s",
        "speedup",
        "parity",
        "median_coord_diff_px",
    ]
    rows = []
    for ref, r in results.items():
        for d in r.get("eval", []):
            rows.append(
                [
                    ref,
                    d.get("dataset", "—"),
                    d.get("samples", "—"),
                    d.get("before_s", "—"),
                    d.get("after_s", "—"),
                    f"{d.get('speedup_x', '—')}x",
                    f"{d.get('parity_same', '—')}/{d.get('parity_total', '—')}",
                    d.get("median_coord_diff_px", "—"),
                ]
            )
    return _md_table(headers, rows)


def eject_table(results):
    """Output-length spread + the early-eject A/B per (ref, dataset). Eject only
    helps when lengths vary, so the spread is shown alongside the win."""
    headers = [
        "ref",
        "dataset",
        "out_len median",
        "p90",
        "max",
        "max/median",
        "eject_off_s",
        "eject_on_s",
        "eject_speedup",
    ]
    rows = []
    for ref, r in results.items():
        for d in r.get("eval", []):
            st = d.get("output_len", {}).get("stats", {})
            rows.append(
                [
                    ref,
                    d["dataset"],
                    st.get("median", "—"),
                    st.get("p90", "—"),
                    st.get("max", "—"),
                    st.get("max_over_median", "—"),
                    d.get("after_noeject_s", "—"),
                    d["after_s"],
                    f"{d.get('eject_speedup_x', '—')}x",
                ]
            )
    return _md_table(headers, rows)


def length_buckets_table(results):
    """The raw output-length histogram (motivates the straggler/eject argument)."""
    # union of bucket labels (stable order from first seen)
    labels, rows = [], []
    for ref, r in results.items():
        for d in r.get("eval", []):
            b = d.get("output_len", {}).get("buckets", {})
            for k in b:
                if k not in labels:
                    labels.append(k)
    for ref, r in results.items():
        for d in r.get("eval", []):
            b = d.get("output_len", {}).get("buckets", {})
            rows.append([ref, d["dataset"]] + [b.get(lab, 0) for lab in labels])
    return _md_table(["ref", "dataset"] + labels, rows)


def grounded_table(results):
    """One image, many queries. ``single_s`` is sequential ``worker.predict``
    per query (what `main` does); ``batch_s`` is one
    ``worker.predict_batch([img]*K, qs)`` call — the real shipped API, on
    whichever branch's worker produced this record. ``speedup_x`` is
    ``single_s / batch_s``: each ref reports its own number for the identical
    call, so PR1's and PR2's rows are directly comparable."""
    headers = [
        "ref",
        "dataset",
        "images",
        "mean q/img",
        "single_s",
        "batch_s",
        "speedup_x",
    ]
    rows = []
    for ref, r in results.items():
        for g in r.get("grounded", []):
            rows.append(
                [
                    ref,
                    g.get("dataset", "—"),
                    g.get("images", "—"),
                    g.get("mean_queries_per_image", "—"),
                    g.get("single_s", "—"),
                    g.get("batch_s", "—"),
                    f"{g.get('speedup_x', '—')}x",
                ]
            )
    return _md_table(headers, rows)


def profile_table(results):
    """Per-step decode-loop profile from ``profile_sweep`` (B=16/32, hybrid/fast,
    averaged over repeats). ``step3_forward_s`` is the model forward itself
    (PR3 doesn't change its algorithm -- a control that shouldn't move);
    ``bookkeeping_s``/``bookkeeping_pct`` is steps 0/1/2/4/5, the per-row Python
    bookkeeping PR3 replaces with closed-form `[A, *]` tensor ops. Refs at the
    same (batch_size, mode) are directly comparable."""
    headers = [
        "ref",
        "batch_size",
        "mode",
        "n_steps",
        "total_s",
        "step3_forward_s",
        "bookkeeping_s",
        "bookkeeping_pct",
    ]
    rows = []
    for ref, r in results.items():
        for p in r.get("profile", []):
            if p.get("oom"):
                rows.append([ref, p.get("batch_size", "—"), p.get("generation_mode", "—"),
                              "OOM", "OOM", "OOM", "OOM", "OOM"])
                continue
            mode = p.get("generation_mode", "—")
            if p.get("partial_oom"):
                mode = f"{mode} (partial OOM, {p.get('repeats', '—')} reps)"
            rows.append(
                [
                    ref,
                    p.get("batch_size", "—"),
                    mode,
                    p.get("n_steps", "—"),
                    p.get("total_s", "—"),
                    p.get("step3_forward_s", "—"),
                    p.get("bookkeeping_s", "—"),
                    f"{p.get('bookkeeping_pct', '—')}%",
                ]
            )
    if not rows:
        return "_(no profile_sweep data)_"
    return _md_table(headers, rows)


def compile_probe_table(results):
    """``torch.compile(mode="reduce-overhead")`` feasibility probe on one
    steady-state decode-step ``lm(...)`` call (see ``compile_probe``).
    ``eager_s``/``compiled_s`` are per-call; ``speedup_x = eager_s /
    compiled_s`` is each ref's own win on the identical isolated call (``A``,
    ``W``, ``Ckv`` -- shapes can differ slightly per ref since they come from
    that ref's own decode loop). ``compile_error``/``error`` mean compilation
    (or capture) failed for that cell on that ref."""
    headers = ["ref", "batch_size", "A", "W", "Ckv", "eager_s", "compiled_s", "speedup_x"]
    rows = []
    for ref, r in results.items():
        for p in r.get("compile_probe", []):
            if p.get("error"):
                rows.append([ref, p.get("batch_size", "—"), "—", "—", "—", "—", "—",
                              f"error: {p['error']}"])
                continue
            if p.get("compile_error"):
                rows.append([ref, p.get("batch_size", "—"), p.get("A", "—"),
                              p.get("W", "—"), p.get("Ckv", "—"),
                              p.get("eager_s", "—"), "—",
                              f"compile_error: {p['compile_error']}"])
                continue
            rows.append([
                ref,
                p.get("batch_size", "—"),
                p.get("A", "—"),
                p.get("W", "—"),
                p.get("Ckv", "—"),
                p.get("eager_s", "—"),
                p.get("compiled_s", "—"),
                f"{p.get('speedup_x', '—')}x",
            ])
    if not rows:
        return "_(no compile_probe data)_"
    return _md_table(headers, rows)


def compile_loop_table(results):
    """Real ragged-decode-loop ``torch.compile`` A/B from ``compile_loop_probe``:
    a full ``predict_batch`` with the language model compiled, under the genuine
    dynamic shapes (``Ckv`` grows every step; ``A`` shrinks on early-eject) that the
    single-step ``compile_probe`` froze out. ``recompiles`` is compilations during
    the timed runs -- near 0 means the dynamic-shape graph generalized and the win
    holds; growing with steps means it recompiles per step and the static-probe
    speedup does NOT survive. ``parity`` is DETECTION-level vs eager (same #boxes +
    ``<ref>`` labels, coords within 5/1000 -- the eval's gate, NOT byte-identity,
    which is expected to be low on long greedy sequences). ``config``
    ``default/dyn`` is inductor+dynamic (the candidate); ``+vit`` also compiled the
    vision tower. ``recompiles`` is compiles during the timed runs (0 = shape-stable).
    ``parity`` is DETECTION-level vs eager (same #boxes + labels, coords within
    5/1000).

    The ladder is MEASURED on ONE workload/run (not a product of probes):
    ``B=1 s`` (sequential ``predict``) -> ``batched s`` (eager ``predict_batch``) ->
    ``compiled s``. ``B=1 s`` is the PRISTINE original model (reloaded from the
    ``.orig`` remote code, original ``generate()``) when available, so ``batching x``
    and ``stacked x`` are vs the UNMODIFIED model -- not our own B=1 path. ``batching
    x = seq/batched``, ``compile x = batched/compiled``, headline ``stacked x =
    seq/compiled`` = the real B=1->batched+compiled win as a single ratio."""
    headers = [
        "ref", "config", "B=1 s", "batched s", "batching x", "compiled s",
        "compile x", "STACKED x", "recompiles", "parity",
    ]
    rows = []
    notes = []
    for ref, r in results.items():
        for e in r.get("compile_loop", []):
            seq = e.get("seq_wall_s", "—")
            batched = e.get("eager_wall_s", "—")
            bx = e.get("batching_wall_x", "—")
            src = e.get("seq_source", "?")
            pr, ours = e.get("pristine_seq_wall_s"), e.get("our_seq_wall_s")
            notes.append(
                f"_{ref}: B=1 baseline = **{src}** "
                f"(pristine {pr}s vs our-checkout {ours}s -- a sanity check that our "
                f"changes didn't alter the B=1 path)._"
            )
            for c in e.get("configs", []):
                cfg = f"{c.get('mode', '—')}/{'dyn' if c.get('dynamic') else 'static'}"
                if c.get("compile_vision"):
                    cfg += "+vit"
                if c.get("error"):
                    rows.append([ref, cfg, seq, batched, f"{bx}x", "—", "—", "—",
                                 "—", f"error: {c['error']}"])
                    continue
                rows.append([
                    ref, cfg, seq, batched, f"{bx}x",
                    c.get("compiled_wall_s", "—"),
                    f"{c.get('wall_speedup_x', '—')}x",
                    f"{c.get('stacked_wall_x', '—')}x",
                    c.get("recompiles_steady", "—"),
                    f"{c.get('parity_same', '—')}/{c.get('parity_total', '—')}",
                ])
    if not rows:
        return "_(no compile_loop data)_"
    table = _md_table(headers, rows)
    if notes:
        table += "\n\n" + "\n\n".join(notes)
    return table


def stacked_eval_table(results):
    """The real-eval-workload speedup ladder (``stacked_eval_probe``): N eval images,
    multi-batch, 2048 tokens, vs the PRISTINE model -- same columns as
    ``compile_loop_table`` but the realistic production regime (compile_loop's
    short-workload ladder understates compile's share). ``record['stacked_eval']`` is
    a single result dict."""
    wrapped = {
        ref: {"compile_loop": [r["stacked_eval"]]}
        for ref, r in results.items() if r.get("stacked_eval")
    }
    return compile_loop_table(wrapped) if wrapped else "_(no stacked_eval data)_"


def compile_parity_table(results):
    """Eval-scale compiled-vs-eager DETECTION parity from ``compile_parity_probe``
    (``n`` samples, vs ``compile_loop``'s n=8). ``within`` = same #boxes + ``<ref>``
    labels AND coords within tol/1000 (the eval's gate); ``struct`` = boxes+labels
    only; ``exact`` = byte-identical (informational). Compare the rate to the eval's
    own batched-vs-B1 parity (e.g. 45/50): a similar rate means compile adds
    divergence of the same bf16-equivalence class, not a new failure mode."""
    headers = ["ref", "config", "n", "within", "struct", "exact (info)"]
    rows = []
    for ref, r in results.items():
        cp = r.get("compile_parity")
        if not cp:
            continue
        n = cp.get("n", "—")
        for c in cp.get("configs", []):
            cfg = f"{c.get('mode', '—')}/{'dyn' if c.get('dynamic') else 'static'}"
            if c.get("compile_vision"):
                cfg += "+vit"
            if c.get("error"):
                rows.append([ref, cfg, n, "—", "—", f"error: {c['error']}"])
                continue
            tot = c.get("parity_total", n)
            rows.append([
                ref, cfg, n,
                f"{c.get('parity_within', '—')}/{tot}",
                f"{c.get('parity_struct', '—')}/{tot}",
                f"{c.get('parity_exact', '—')}/{tot}",
            ])
    if not rows:
        return "_(no compile_parity data)_"
    return _md_table(headers, rows)


def _ap(metrics):
    """Format a scored-metrics dict's headline AP, or its error/—."""
    if not isinstance(metrics, dict):
        return "—"
    if "error" in metrics:
        return f"err: {str(metrics['error'])[:40]}"
    v = metrics.get("avg_precision")
    return f"{v:.3f}" if isinstance(v, (int, float)) else "—"


def compile_ap_table(results):
    """The compile mAP gate: per dataset, ``mAP(eager)`` vs ``mAP(compiled)`` over
    ``n`` samples, scored inline with the repo metric (``score_compile_ap``).
    ``ΔAP`` is compiled minus eager -- within noise means compile's parity
    divergence is near-ties and it ships; a real drop means it loses detections and
    needs fp32 logits. ``within``/``struct`` are the detection parity for context.
    ``compile_ap`` is ``{dataset: probe_result}``."""
    headers = ["ref", "dataset", "config", "n", "within", "struct", "AP", "ΔAP vs eager"]
    rows = []
    for ref, r in results.items():
        ap = r.get("compile_ap")
        if not isinstance(ap, dict):
            continue
        for ds, res in ap.items():
            if not isinstance(res, dict):
                continue
            n = res.get("n", "—")
            base = (res.get("map_eager") or {}).get("avg_precision")
            note = res.get("score_error")
            rows.append([ref, ds, "eager", n, "—", "—",
                         _ap(res.get("map_eager")) if not note else f"unscored ({note})", "—"])
            for c in res.get("configs", []):
                cfg = f"{c.get('mode', '—')}/{'dyn' if c.get('dynamic') else 'static'}"
                if c.get("compile_vision"):
                    cfg += "+vit"
                if c.get("error"):
                    rows.append([ref, ds, cfg, n, "—", "—", f"error: {c['error']}", "—"])
                    continue
                tot = c.get("parity_total", n)
                cap = (c.get("map") or {}).get("avg_precision")
                delta = (f"{cap - base:+.3f}"
                         if isinstance(cap, (int, float)) and isinstance(base, (int, float))
                         else "—")
                rows.append([
                    ref, ds, cfg, n,
                    f"{c.get('parity_within', '—')}/{tot}",
                    f"{c.get('parity_struct', '—')}/{tot}",
                    _ap(c.get("map")), delta,
                ])
    if not rows:
        return "_(no compile_ap data)_"
    return _md_table(headers, rows)


def _img(b64, alt):
    return f'<img alt="{alt}" src="data:image/png;base64,{b64}">'


def speed_plot(results):
    """Render each ref's ``speed_sweep`` (sequential vs batched, N distinct
    images) as a base64-embedded plot. Skips refs without ``speed`` data (older
    records) and degrades to a note if matplotlib (or this module's package
    context) isn't available."""
    out = []
    for ref, r in results.items():
        rows = r.get("speed")
        if not rows:
            continue
        try:
            from .plots import plot_speed_sweep

            b64 = plot_speed_sweep(
                rows, title=f"{ref}: sequential vs batched", as_base64=True
            )
        except ImportError as e:
            return f"_(plot skipped: {e})_"
        out.append(_img(b64, f"{ref} speed sweep"))
    return "\n\n".join(out) if out else "_(no speed_sweep data)_"


def write_doc(results, path="EVIDENCE.md"):
    from . import evidence_narrative as nar

    refs = ", ".join(results)
    lines = [
        nar.overview(),
        "",
        nar.code_changes(),
        "",
        nar.correctness_methodology(),
        "",
        "# Performance (measured)",
        "",
        f"_Each section below is each ref's **own** isolated run ({refs}); combined "
        "by `repro.combine`. Baseline (`before`/`single`/`B=1`) is the original "
        "single-sequence path._",
        "",
        "## Detection: batched speedup + parity",
        "",
        "`before` is `worker.predict` (B=1, what `main` does today); `after` is "
        "`worker.predict_batch` on distinct real images — the actual shipped "
        "API, called the way a real caller would. What `after` costs depends on "
        "this checkout's `predict_batch`: PR1 batches the vision tower across "
        "all N images, PR2's unified `predict_batch` encodes each distinct "
        "image separately. Each ref's row reflects its own checkout.",
        "",
        detection_table(results),
        "",
        "### Throughput sweep: sequential vs batched",
        "",
        "N distinct images (`image.copy()`): `sequential_s` is N x "
        "`predict_batch([img],[q])`, `batched_s` is one `predict_batch(imgs, "
        "qs)` call on this checkout. `endtoend_speedup = sequential_s / "
        "batched_s` is the real wall-clock win for that call.",
        "",
        speed_plot(results),
        "",
        "## Output-length spread + early-eject A/B",
        "",
        "Early-eject only helps when output lengths vary within a batch; the spread "
        "(`max/median`) shows how much they do, and `eject_speedup` is the measured "
        "straggler-removal win on that spread.",
        "",
        eject_table(results),
        "",
        "### Output-length histogram",
        "",
        length_buckets_table(results),
        "",
        "## One image, many queries",
        "",
        "**RefCOCOg** referring expressions grouped by image — each image carries "
        "several distinct natural-language queries, the workload standard detection "
        "can't exercise. `single_s` is sequential `worker.predict` per query "
        "(what `main` does); `batch_s` is one `worker.predict_batch([img]*K, "
        "qs)` call -- the real shipped API, on whichever branch's worker "
        "produced this record.",
        "",
        "`speedup_x = single_s / batch_s` is each ref's own win for the "
        "identical call: PR1's `predict_batch` re-encodes the (literally "
        "same-object) image K times but batches decode; PR2's also dedups the "
        "vision encode via `id()`-keyed caching. The two refs' `speedup_x` are "
        "directly comparable.",
        "",
        grounded_table(results),
        "",
        "## Decode-loop bookkeeping profile",
        "",
        "Per-step timing from `profile_sweep` (B=16/32, `hybrid`/`fast`, averaged "
        "over repeats). `step3_forward_s` is the model forward itself -- a control "
        "that PR3 does not change algorithmically and should not move. "
        "`bookkeeping_s`/`bookkeeping_pct` is steps 0/1/2/4/5: the per-row Python "
        "loops (input-block assembly, mask construction, cache compaction) that "
        "PR3 replaces with closed-form `[A, *]` tensor ops. A ref with PR3 should "
        "show a lower `bookkeeping_s` at the same `(batch_size, mode)`, growing "
        "with batch size and most visible in `fast` mode (every step hits the "
        "mask dispatch, vs. only MTP steps in `hybrid`).",
        "",
        profile_table(results),
        "",
        "## torch.compile feasibility probe",
        "",
        "One steady-state decode-step `lm(...)` call (real `[A, W]` input_ids / "
        "`[A, Ckv+W]` attention_mask / `past_key_values`, captured from a "
        "`fast`-mode `predict_batch` after prefill), timed eager vs. "
        "`torch.compile(mode=\"reduce-overhead\")`, isolated from the rest of "
        "the decode loop. PR3's Stage A removed the B per-row "
        "`input_ids[b,-1].item()` host syncs (one `torch._dynamo` graph break "
        "each) from this call's mask dispatch; a ref without that change pays "
        "B extra graph breaks per call here. `speedup_x` is each ref's own "
        "`eager_s / compiled_s` -- a large gap between refs at the same "
        "`(batch_size, A, W, Ckv)` is the signal that PR3's design, not just "
        "its current wall-clock, is what unlocks this.",
        "",
        compile_probe_table(results),
        "",
        "### Real decode-loop torch.compile A/B",
        "",
        "`compile_probe` above freezes one step; this runs a full `predict_batch` "
        "with the language model `torch.compile`d, under the genuine ragged shapes "
        "(`Ckv` grows every step; `A` shrinks on early-eject). `recompiles` is "
        "compilations during the timed runs -- near 0 means the dynamic-shape graph "
        "generalized and the static-step speedup survives; growing with steps means "
        "it recompiles per step and the win evaporates. `parity` is compiled vs. "
        "eager output (the correctness gate). `default/dyn` is inductor + dynamic "
        "shapes (the realistic candidate); `reduce-overhead/static` is cudagraphs -- "
        "what the static probe used -- expected to thrash on a growing-`Ckv` loop. "
        "A ref with PR3's sync-free dispatch should recompile less and hold more of "
        "its speedup here than one still paying per-step `.item()` graph breaks. "
        "Inputs are real decode-bound eval images (high `steps`) so `wall_x` reflects "
        "the regime where compile pays off, not a prefill-dominated one. `+vit` "
        "configs also compile the vision tower (most of the non-decode wall), "
        "measuring the additional end-to-end gain.",
        "",
        compile_loop_table(results),
        "",
        "### Compile mAP gate (eager vs compiled)",
        "",
        "Parity says how often compiled output *differs* from eager; this is the "
        "gate that says whether the difference *costs detections*. `compile_ap` runs "
        "100 samples eager and compiled with the same shipped settings (eject on, "
        "greedy) and writes paired `preds_*.jsonl`. Score each with the repo metric "
        "(`convert_coco_lvis_to_standard_format.py` then `coco_lvis_metric.py`, same "
        "GT as the eval) for mAP(eager) vs mAP(compiled): if mAP is within noise the "
        "parity divergence is near-ties and compile ships; if mAP drops, compile "
        "loses real detections and needs fp32 logits.",
        "",
        compile_ap_table(results),
        "",
        "## End-to-end stacked speedup (real eval workload)",
        "",
        "The headline `B=1 -> batched -> batched+compiled` win as a SINGLE measured "
        "ratio (not a product of probes), over N real eval images, multi-batch, at "
        "the eval's 2048 tokens, **vs the PRISTINE original model** (reloaded from the "
        "`.orig` remote code). `batching x = pristine_seq / batched`, `compile x = "
        "batched / compiled`, and `STACKED x = pristine_seq / compiled` is the real "
        "production speedup against the unmodified model. The note under the table "
        "shows pristine-B=1 vs our-checkout-B=1 -- a sanity check that our changes "
        "didn't alter (or game) the baseline.",
        "",
        stacked_eval_table(results),
        "",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    logger.info("wrote %s", path)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "results", nargs="+", help="results_<ref>.json files from run_branch"
    )
    ap.add_argument("-o", "--out", default="EVIDENCE.md")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    write_doc(load_results(args.results), args.out)


if __name__ == "__main__":
    main()
