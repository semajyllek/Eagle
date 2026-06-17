# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
"""``torch.compile`` feasibility probe for the decode-step forward.

PR3's Stage A replaced the per-row ``input_ids[b, -1].item()`` mask dispatch
(``apply_per_row_generation_window``, B host syncs per decode step) with a
tensor-only ``torch.where`` dispatch (``apply_per_row_generation_window_vectorized``,
0 syncs). Every ``.item()`` is a ``torch._dynamo`` graph break / CUDA-graph-breaking
host sync, so this is the one difference relevant to graph capture: PR1 forces B
extra breaks per decode step that PR3 does not.

This probe captures one steady-state decode step's ``lm(**model_kwargs)`` call --
the real ``[A, W]`` input_ids / ``[A, Ckv+W]`` attention_mask / ``[A, W]``
position_ids / ``past_key_values``, taken from a real ``fast``-mode
``predict_batch`` after prefill -- and times it eager vs.
``torch.compile(mode="reduce-overhead")``, isolated from the rest of the decode
loop. It does not change what either PR ships; it answers "does removing those B
syncs make this call compile/cudagraph-friendly in a way the un-vectorized version
can't be."

``compile_probe`` (below) answers that for ONE frozen step. ``compile_loop_probe``
answers the harder, decisive question: does that compile win survive the **real
ragged decode loop**, where ``Ckv`` grows every step and ``A`` shrinks on
early-eject? It swaps ``worker.model.language_model`` for a ``torch.compile``d
wrapper, runs a real ``predict_batch``, and measures decode wall/forward time eager
vs. compiled AND counts recompiles -- the make-or-break metric, since per-step
shape changes can force a recompile (or cudagraph re-record) every step and erase
the gain. ``dynamic=True`` (inductor) is the candidate that should generalize;
``reduce-overhead`` (cudagraphs, static) is what the single-step probe used and is
expected to thrash here.

Run from each PR's own notebook (same worker each PR's ``benchmark_pr*.ipynb``
already loaded)::

    from repro.compile_probe import compile_probe, compile_loop_probe
    compile_probe(worker, imgs[0], prompts[0])                 # one static step
    compile_loop_probe(worker, imgs, prompts, batch_size=8)    # the real loop
"""
import json
import logging
import os
import shutil
import time
from statistics import median

import torch

from .parity import coord_diff, parse_refs
from .perf import free

logger = logging.getLogger(__name__)


def _clone_legacy_cache(past_key_values):
    if past_key_values is None:
        return None
    return tuple((k.clone(), v.clone()) for k, v in past_key_values)


def _clone_kwargs(kwargs):
    out = {}
    for key, val in kwargs.items():
        if torch.is_tensor(val):
            out[key] = val.clone()
        elif key == "past_key_values":
            out[key] = _clone_legacy_cache(val)
        else:
            out[key] = val
    return out


def _capture_decode_step_kwargs(worker, image, prompt, batch_size):
    """Run one short ``fast``-mode ``predict_batch`` and capture the
    ``lm(**model_kwargs)`` call for the first true decode step -- the call right
    after prefill (real ``[A, W]`` MTP-window input, ``past_key_values`` from
    prefill, every step hits the MTP mask dispatch in ``fast`` mode)."""
    lm = worker.model.language_model
    imgs = [image.copy() for _ in range(batch_size)]
    qs = [prompt] * batch_size

    captured = {}
    orig_forward = lm.forward
    state = {"n": 0}

    def hooked_forward(*args, **kwargs):
        state["n"] += 1
        if state["n"] == 2 and "kwargs" not in captured:
            captured["kwargs"] = _clone_kwargs(kwargs)
        return orig_forward(*args, **kwargs)

    lm.forward = hooked_forward
    try:
        worker.predict_batch(
            imgs, qs, generation_mode="fast", max_new_tokens=64, temperature=0.0,
        )
    finally:
        lm.forward = orig_forward

    if "kwargs" not in captured:
        raise RuntimeError(
            "decode step was never reached (batch finished after prefill)"
        )
    return lm, captured["kwargs"]


def _run_calls(fn, kwargs, n):
    for _ in range(n):
        call_kwargs = _clone_kwargs(kwargs)
        with torch.no_grad():
            fn(**call_kwargs)


def _timed_calls(fn, kwargs, n_warmup, n_repeats):
    _run_calls(fn, kwargs, n_warmup)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    _run_calls(fn, kwargs, n_repeats)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_repeats


def compile_probe(worker, image, prompt, batch_sizes=(16, 32), n_warmup=8, n_repeats=20):
    """For each ``batch_size``: capture one steady-state decode-step call and
    time it eager vs. ``torch.compile(mode="reduce-overhead")``, isolated.

    Returns a list of per-``batch_size`` dicts with the captured call shape
    (``A``, ``W``, ``Ckv``), ``eager_s``/``compiled_s`` (per-call, averaged over
    ``n_repeats`` after ``n_warmup`` untimed calls -- ``n_warmup`` also covers
    ``torch.compile``'s first-call compilation and cudagraph capture), and
    ``speedup_x = eager_s / compiled_s``. If capture, compilation, or the
    compiled call raises, that cell is recorded with ``error`` instead of
    raising, so one bad cell can't take the whole record down.
    """
    rows = []
    for n in batch_sizes:
        free()
        row = dict(batch_size=n)
        try:
            lm, kwargs = _capture_decode_step_kwargs(worker, image, prompt, n)
            A, W = kwargs["input_ids"].shape
            Ckv = (
                kwargs["past_key_values"][0][0].shape[2]
                if kwargs["past_key_values"]
                else 0
            )
            row.update(A=A, W=W, Ckv=Ckv)

            eager_s = _timed_calls(lm, kwargs, n_warmup, n_repeats)
            row["eager_s"] = round(eager_s, 5)

            torch._dynamo.reset()
            try:
                compiled_lm = torch.compile(lm, mode="reduce-overhead")
                compiled_s = _timed_calls(compiled_lm, kwargs, n_warmup, n_repeats)
                row["compiled_s"] = round(compiled_s, 5)
                row["speedup_x"] = (
                    round(eager_s / compiled_s, 3) if compiled_s else None
                )
            except Exception as e:
                row["compile_error"] = f"{type(e).__name__}: {e}".strip().replace("\n", " ")
            finally:
                torch._dynamo.reset()
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {e}".strip().replace("\n", " ")
        rows.append(row)
    free()
    return rows


# ---------------------------------------------------------------------------
# Real-loop probe: compile the language model and run the actual ragged decode.
# ---------------------------------------------------------------------------
def _set_recompile_limit(limit):
    """Raise dynamo's recompile cap (default 8) so it compiles EVERY decode shape
    instead of bailing to eager after 8 -- otherwise the compiled timings include a
    silent eager fallback and ``recompiles_steady`` reads a falsely-reassuring 0
    (it counts new compiles, but after the cap is hit there are none because dynamo
    gave up). Returns the saved values for ``_restore_recompile_limit``."""
    cfg = torch._dynamo.config
    saved = {n: getattr(cfg, n) for n in ("recompile_limit", "cache_size_limit")
             if hasattr(cfg, n)}
    for n in saved:
        setattr(cfg, n, limit)
    return saved


def _restore_recompile_limit(saved):
    cfg = torch._dynamo.config
    for n, v in saved.items():
        setattr(cfg, n, v)


def _frames_ok():
    """Number of frames dynamo has compiled so far (initial compiles + recompiles).
    Delta over a run == compilations triggered by that run. ``None`` if the
    counter API isn't available on this torch."""
    try:
        return int(torch._dynamo.utils.counters["frames"]["ok"])
    except Exception:
        return None


def _run_predict_batch(worker, imgs, qs, max_new_tokens):
    """One real ``predict_batch`` decode; returns (outputs, wall_s, profile dict).
    ``profile`` carries the per-step breakdown (``step3_forward`` == pure lm time)."""
    prof = {}
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = worker.predict_batch(
        imgs, qs, generation_mode="hybrid", max_new_tokens=max_new_tokens,
        temperature=0.0, profile=prof,
    )
    torch.cuda.synchronize()
    return out, time.perf_counter() - t0, prof


def _snapshot_dir(worker):
    """Recover the local snapshot dir the worker was loaded from (carries the
    overlaid remote code + the .orig pristine backups)."""
    m = getattr(worker, "model", None)
    cand = getattr(getattr(m, "config", None), "_name_or_path", None)
    if not cand:
        cand = getattr(getattr(worker, "tokenizer", None), "name_or_path", None)
    return cand if cand and os.path.isdir(cand) else None


def _load_pristine_worker(worker, dest):
    """Load a worker from the model's PRISTINE remote code -- the original
    ``generate()`` (which still asserts ``batch_size==1``), reconstructed from the
    ``.orig`` backups the overlay saved: real ``.py`` from ``<name>.orig`` (else the
    file), symlinked weights. This is the true 'unmodified model' B=1 baseline.
    Returns a worker, or ``None`` if there are no ``.orig`` backups (model was never
    overlaid) or the snapshot can't be located."""
    snap = _snapshot_dir(worker)
    if not snap or not any(f.endswith(".orig") for f in os.listdir(snap)):
        return None
    if os.path.exists(dest):
        shutil.rmtree(dest)
    os.makedirs(dest)
    for entry in os.listdir(snap):
        if entry.endswith(".orig"):
            continue
        src = os.path.join(snap, entry)
        out = os.path.join(dest, entry)
        if entry.endswith(".py"):
            pristine = src + ".orig" if os.path.exists(src + ".orig") else src
            shutil.copy2(os.path.realpath(pristine), out)  # real file for trust_remote_code
        else:
            os.symlink(os.path.realpath(src), out)
    from locateanything_worker import LocateAnythingWorker

    return LocateAnythingWorker(dest, device=worker.device, dtype=worker.dtype)


def _seq_wall(worker, imgs, qs, max_new_tokens):
    """Wall time of the B=1 SEQUENTIAL path -- ``worker.predict`` once per image,
    i.e. exactly what ``main`` does today. The base of the stacked-speedup ladder so
    the full ``B=1 -> batched -> batched+compiled`` win is one MEASURED ratio rather
    than a product of two probes."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for img, q in zip(imgs, qs):
        worker.predict(
            img, q, generation_mode="hybrid", max_new_tokens=max_new_tokens,
            temperature=0.0, verbose=False,
        )
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def _detection_parity(eager_out, comp_out, tol=5):
    """Detection-level compiled-vs-eager parity over aligned answer lists. Returns
    ``(exact, struct, within, n)``: byte-identical, same #boxes + ``<ref>`` labels,
    and struct AND coords within ``tol``/1000 (the eval's gate)."""
    n = min(len(eager_out), len(comp_out))
    exact = struct = within = 0
    for a, b in zip(eager_out[:n], comp_out[:n]):
        exact += int(a == b)
        cd = coord_diff(a, b)  # None if box structure differs
        if cd is not None and parse_refs(a) == parse_refs(b):
            struct += 1
            within += int(cd <= tol)
    return exact, struct, within, n


def _predict_all(worker, images, prompts, batch_size, max_new_tokens):
    """``predict_batch`` over all samples in chunks of ``batch_size``; returns the
    flat list of answer strings."""
    out = []
    for i in range(0, len(images), batch_size):
        out.extend(
            worker.predict_batch(
                images[i:i + batch_size], prompts[i:i + batch_size],
                generation_mode="hybrid", max_new_tokens=max_new_tokens,
                temperature=0.0,
            )
        )
    return out


def compile_loop_probe(
    worker,
    images,
    prompts,
    batch_size=8,
    max_new_tokens=128,
    configs=(("default", True), ("reduce-overhead", False)),
    n_repeats=3,
):
    """Time the REAL ``batched_generate`` decode loop eager vs. ``torch.compile``.

    Unlike ``compile_probe`` (one frozen ``[A, W]/[A, Ckv+W]`` step), this runs a
    full ``predict_batch`` so the lm forward sees the genuine ragged shapes: ``Ckv``
    grows every step, and with ``early_eject`` the batch dim ``A`` shrinks too. It
    swaps ``worker.model.language_model`` for a compiled wrapper (restored in a
    ``finally``), so only the decoder is compiled -- the vision tower and prefill
    embedding-merge ride along unchanged, unless a config opts into
    ``compile_vision`` (below).

    Feed it **real decode-bound images** (long outputs, ~100+ steps) to measure the
    end-to-end win in the regime where it matters; ``sample_images`` generate only
    ~9 tokens, so the lm forward is a small slice of wall and ``wall_speedup_x``
    understates the payoff (use the ``n_steps`` field to see which regime you got).

    ``early_eject`` defaults to ``False`` for a clean compiled-vs-eager parity check
    (eject perturbs output via bf16 batch-variance) and to isolate the seq-len
    dynamism (``A`` fixed, only ``Ckv`` grows -- the canonical hard case for
    cudagraphs). Set ``True`` to add batch-dim dynamism.

    Each entry in ``configs`` is ``(mode, dynamic)`` or ``(mode, dynamic,
    compile_vision)``; ``compile_vision=True`` *also* ``torch.compile``s
    ``worker.model.vision_model`` (the ViT is the bulk of the non-decode wall, so
    this measures the additional end-to-end gain from compiling it too). For each
    config it does one warmup ``predict_batch`` (compilation + dynamo auto-dynamic
    learning), then ``n_repeats`` timed runs.
    Returns a dict with the eager baseline (median ``eager_wall_s``/``eager_fwd_s``,
    ``n_steps``, ``A_min``/``A_max``) and a ``configs`` list, each with
    ``compiled_wall_s``/``compiled_fwd_s``, ``wall_speedup_x``/``fwd_speedup_x``,
    ``warmup_compiles``, ``recompiles_steady`` (compiles during the timed runs --
    near 0 means dynamic shapes generalized; growing with steps means it's
    recompiling every step and the gain won't hold), and parity vs eager:
    ``parity_same``/``parity_total`` is DETECTION-level (``parity_within``: same
    #boxes + ``<ref>`` labels, coords within 5/1000 -- the eval's gate), with
    ``parity_struct`` (boxes+labels only) and ``parity_exact`` (byte-identical,
    informational) alongside. Byte-exact is expected to be low on long greedy
    sequences -- inductor fusion changes accumulation order, flipping a token that
    cascades -- so detection-level parity is what says whether compile is safe. A
    failing config records ``error`` instead of raising.
    """
    orig_lm = worker.model.language_model
    orig_vit = getattr(worker.model, "vision_model", None)
    imgs = [images[i % len(images)].copy() for i in range(batch_size)]
    qs = [prompts[i % len(prompts)] for i in range(batch_size)]

    free()
    # ---- eager baseline (median over n_repeats) ----
    walls, fwds, eager_out = [], [], None
    eager_prof = None
    for _ in range(n_repeats):
        out, wall, prof = _run_predict_batch(worker, imgs, qs, max_new_tokens)
        walls.append(wall)
        fwds.append(prof.get("step3_forward", 0.0))
        if eager_out is None:
            eager_out, eager_prof = out, prof
    a_hist = eager_prof.get("A_history", []) if eager_prof else []
    eager_wall = median(walls)
    reps = max(1, n_repeats // 2)
    # ---- B=1 sequential baselines: our checkout AND the PRISTINE original model ----
    our_seq = median([_seq_wall(worker, imgs, qs, max_new_tokens) for _ in range(reps)])
    pristine_seq = None
    try:
        pw = _load_pristine_worker(worker, os.path.join("/tmp", "pristine_model"))
        if pw is not None:
            pristine_seq = median([_seq_wall(pw, imgs, qs, max_new_tokens) for _ in range(reps)])
            del pw
            free()
    except Exception as e:
        logger.warning("pristine B=1 baseline unavailable (%s); using our-checkout B=1", e)
    # The true baseline is the unmodified model; fall back to our B=1 path if the
    # pristine model can't be reconstructed (no .orig backups).
    seq_wall = pristine_seq if pristine_seq is not None else our_seq
    result = {
        "batch_size": batch_size,
        "max_new_tokens": max_new_tokens,
        "n_steps": eager_prof.get("n_steps") if eager_prof else None,
        "A_min": min(a_hist) if a_hist else None,
        "A_max": max(a_hist) if a_hist else None,
        "seq_source": "pristine" if pristine_seq is not None else "our_checkout_B1",
        "pristine_seq_wall_s": round(pristine_seq, 4) if pristine_seq is not None else None,
        "our_seq_wall_s": round(our_seq, 4),
        "seq_wall_s": round(seq_wall, 4),
        "eager_wall_s": round(eager_wall, 4),
        "eager_fwd_s": round(median(fwds), 4),
        # MEASURED batching win (B=1 seq -> batched eager), same workload.
        "batching_wall_x": round(seq_wall / eager_wall, 3) if eager_wall else None,
        "configs": [],
    }

    # Compile every decode shape (no eager fallback) so fwd_x + recompile counts are
    # truthful; restored before return.
    saved_limits = _set_recompile_limit(256)
    for spec in configs:
        mode, dynamic = spec[0], spec[1]
        compile_vision = spec[2] if len(spec) > 2 else False
        cfg = {"mode": mode, "dynamic": dynamic, "compile_vision": compile_vision}
        try:
            torch._dynamo.reset()
            try:
                torch._dynamo.utils.counters.clear()
            except Exception:
                pass
            worker.model.language_model = torch.compile(orig_lm, mode=mode, dynamic=dynamic)
            if compile_vision and orig_vit is not None:
                worker.model.vision_model = torch.compile(orig_vit, mode=mode, dynamic=dynamic)

            # warmup: first-call compilation + dynamo's auto-dynamic learning.
            _run_predict_batch(worker, imgs, qs, max_new_tokens)
            cfg["warmup_compiles"] = _frames_ok()

            cwalls, cfwds, c_out, recompiles = [], [], None, 0
            for _ in range(n_repeats):
                before = _frames_ok()
                out, wall, prof = _run_predict_batch(
                    worker, imgs, qs, max_new_tokens
                )
                after = _frames_ok()
                if before is not None and after is not None:
                    recompiles += after - before
                cwalls.append(wall)
                cfwds.append(prof.get("step3_forward", 0.0))
                if c_out is None:
                    c_out = out

            cw, cf = median(cwalls), median(cfwds)
            cfg["compiled_wall_s"] = round(cw, 4)
            cfg["compiled_fwd_s"] = round(cf, 4)
            cfg["wall_speedup_x"] = round(result["eager_wall_s"] / cw, 3) if cw else None
            cfg["fwd_speedup_x"] = round(result["eager_fwd_s"] / cf, 3) if cf else None
            # THE end-to-end number: B=1 sequential -> batched+compiled, one workload.
            cfg["stacked_wall_x"] = round(seq_wall / cw, 3) if cw else None
            cfg["recompiles_steady"] = recompiles
            # Parity is DETECTION-level, not byte-identity: greedy decode amplifies
            # tiny inductor-fusion numerical diffs into divergent token streams, so
            # the question is whether the *detections* still match (same #boxes +
            # <ref> labels, coords within 5/1000) -- the same gate the eval uses.
            if c_out:
                exact, struct, within, _ = _detection_parity(eager_out, c_out)
                cfg["parity_exact"] = exact      # byte-identical (informational)
                cfg["parity_struct"] = struct    # same #boxes + same labels
                cfg["parity_within"] = within    # struct AND coords within 5/1000
                cfg["parity_same"] = within      # headline = detection-equivalent
            else:
                cfg["parity_exact"] = cfg["parity_struct"] = cfg["parity_within"] = None
                cfg["parity_same"] = None
            cfg["parity_total"] = len(eager_out)
        except Exception as e:
            cfg["error"] = f"{type(e).__name__}: {e}".strip().replace("\n", " ")
        finally:
            worker.model.language_model = orig_lm
            if compile_vision and orig_vit is not None:
                worker.model.vision_model = orig_vit
            torch._dynamo.reset()
        result["configs"].append(cfg)
        free()
    _restore_recompile_limit(saved_limits)
    return result


def compile_parity_probe(
    worker,
    images,
    prompts,
    batch_size=8,
    max_new_tokens=512,
    configs=(("default", True, False), ("default", True, True)),
    tol=5,
):
    """Eval-scale DETECTION parity: compiled-vs-eager over MANY distinct samples.

    ``compile_loop_probe`` measures parity on one ``batch_size`` batch (n=8 -- too
    small to tell a real divergence rate from noise). This runs ``predict_batch``
    over all of ``images``/``prompts`` (in chunks of ``batch_size``) eager, then
    once per compiled config, and scores compiled-vs-eager detections the same way
    the eval scores batched-vs-B1 -- so the resulting rate is directly comparable to
    the eval's batched parity (e.g. 45/50). Both sides are batched + ``early_eject``
    OFF, so the ONLY difference is ``torch.compile``: this isolates compile's
    marginal correctness cost, not batching's or eject's.

    Each ``configs`` entry is ``(mode, dynamic)`` or ``(mode, dynamic,
    compile_vision)``. The eager pass is run once and shared across configs.
    Returns ``{n, batch_size, max_new_tokens, configs: [...]}`` where each config
    has ``parity_within``/``parity_struct``/``parity_exact``/``parity_total`` (see
    ``_detection_parity``). A failing config records ``error``.
    """
    orig_lm = worker.model.language_model
    orig_vit = getattr(worker.model, "vision_model", None)

    free()
    eager = _predict_all(worker, images, prompts, batch_size, max_new_tokens)
    result = {
        "n": len(eager),
        "batch_size": batch_size,
        "max_new_tokens": max_new_tokens,
        "tol": tol,
        "configs": [],
    }

    saved_limits = _set_recompile_limit(256)
    for spec in configs:
        mode, dynamic = spec[0], spec[1]
        compile_vision = spec[2] if len(spec) > 2 else False
        cfg = {"mode": mode, "dynamic": dynamic, "compile_vision": compile_vision}
        try:
            torch._dynamo.reset()
            worker.model.language_model = torch.compile(orig_lm, mode=mode, dynamic=dynamic)
            if compile_vision and orig_vit is not None:
                worker.model.vision_model = torch.compile(orig_vit, mode=mode, dynamic=dynamic)
            comp = _predict_all(worker, images, prompts, batch_size, max_new_tokens)
            exact, struct, within, n = _detection_parity(eager, comp, tol=tol)
            cfg["parity_exact"] = exact
            cfg["parity_struct"] = struct
            cfg["parity_within"] = within
            cfg["parity_same"] = within
            cfg["parity_total"] = n
        except Exception as e:
            cfg["error"] = f"{type(e).__name__}: {e}".strip().replace("\n", " ")
        finally:
            worker.model.language_model = orig_lm
            if compile_vision and orig_vit is not None:
                worker.model.vision_model = orig_vit
            torch._dynamo.reset()
        result["configs"].append(cfg)
        free()
    _restore_recompile_limit(saved_limits)
    return result


def compile_ap_probe(
    worker,
    samples,
    out_dir,
    batch_size=8,
    max_new_tokens=1024,
    configs=(("default", True, False), ("default", True, True)),
):
    """Write paired eager/compiled prediction files for OFFLINE mAP scoring.

    Parity says how often compiled output *differs* from eager; this says whether
    that difference *costs detections*. Runs ``predict_batch`` over ``samples`` eager
    then once per compiled config -- all with the SAME shipped settings (``early_eject``
    on, greedy) so the only difference is ``torch.compile`` -- and writes each pass's
    predictions to ``out_dir/preds_<tag>.jsonl`` in the eval's ``to_record`` format.
    Score each with the repo metric (``convert_coco_lvis_to_standard_format.py`` then
    ``coco_lvis_metric.py``) to get ``mAP(eager)`` vs ``mAP(compiled)``; the GT json is
    the same one the eval's own AP uses. Also reports detection parity per config.

    Returns ``{n, out_dir, max_new_tokens, early_eject, files, configs:[...]}``.
    """
    from .eval_parity import build_question, to_record

    os.makedirs(out_dir, exist_ok=True)
    orig_lm = worker.model.language_model
    orig_vit = getattr(worker.model, "vision_model", None)
    images = [s["image"] for s in samples]
    prompts = [build_question(s["categories"]) for s in samples]

    def _write(tag, outputs):
        path = os.path.join(out_dir, f"preds_{tag}.jsonl")
        # ensure_ascii=True: the convert script reads the pred jsonl as utf-8, and
        # LVIS outputs can carry lone byte-fallback surrogates (Qwen's byte-level
        # tokenizer emits them for partial multibyte tokens) that ensure_ascii=False
        # can't serialize. ascii-escaping (\\udcXX) yields a pure-ASCII file that is
        # codec- and surrogate-proof and round-trips via json.loads. (COCO is clean
        # ASCII so it never hit this.)
        with open(path, "w", encoding="utf-8") as f:
            for s, o in zip(samples, outputs):
                f.write(json.dumps(to_record(s, o)) + "\n")
        return path

    free()
    eager = _predict_all(worker, images, prompts, batch_size, max_new_tokens)
    files = {"eager": _write("eager", eager)}
    result = {
        "n": len(eager),
        "out_dir": out_dir,
        "max_new_tokens": max_new_tokens,
        "files": files,
        "configs": [],
    }

    saved_limits = _set_recompile_limit(256)  # no eager fallback -> truly-compiled preds
    for spec in configs:
        mode, dynamic = spec[0], spec[1]
        compile_vision = spec[2] if len(spec) > 2 else False
        tag = f"{mode}_{'dyn' if dynamic else 'static'}" + ("_vit" if compile_vision else "")
        cfg = {"mode": mode, "dynamic": dynamic, "compile_vision": compile_vision, "tag": tag}
        try:
            torch._dynamo.reset()
            worker.model.language_model = torch.compile(orig_lm, mode=mode, dynamic=dynamic)
            if compile_vision and orig_vit is not None:
                worker.model.vision_model = torch.compile(orig_vit, mode=mode, dynamic=dynamic)
            comp = _predict_all(
                worker, images, prompts, batch_size, max_new_tokens
            )
            exact, struct, within, n = _detection_parity(eager, comp)
            cfg["parity_exact"] = exact
            cfg["parity_struct"] = struct
            cfg["parity_within"] = within
            cfg["parity_total"] = n
            cfg["pred_file"] = _write(tag, comp)
            files[tag] = cfg["pred_file"]
        except Exception as e:
            cfg["error"] = f"{type(e).__name__}: {e}".strip().replace("\n", " ")
        finally:
            worker.model.language_model = orig_lm
            if compile_vision and orig_vit is not None:
                worker.model.vision_model = orig_vit
            torch._dynamo.reset()
        result["configs"].append(cfg)
        free()
    _restore_recompile_limit(saved_limits)
    return result


def _timed_predict_all(worker, images, prompts, batch_size, max_new_tokens):
    """Wall time of a multi-batch ``predict_batch`` sweep (eject on = shipped path)."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    _predict_all(worker, images, prompts, batch_size, max_new_tokens)
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def stacked_eval_probe(
    worker,
    samples,
    batch_size=8,
    max_new_tokens=2048,
    configs=(("default", True, False), ("default", True, True)),
):
    """Stacked speedup ladder on the REAL EVAL workload, vs the PRISTINE model.

    Unlike ``compile_loop_probe`` (one batch of ``batch_size`` at short
    ``max_new_tokens``), this runs ``N`` real eval images through the *multi-batch*
    shipped path (groups of ``batch_size``, ``early_eject`` on, eval-length
    ``max_new_tokens``) -- the real length spread, where compile's share is larger --
    so the headline production number isn't understated. Times pristine B=1 seq (the
    unmodified model), our B=1 seq (sanity), batched-eager, and each compiled config,
    and returns the same shape ``compile_loop_table`` renders (``seq_wall_s`` /
    ``eager_wall_s`` / ``batching_wall_x`` + per-config ``compiled_wall_s`` /
    ``wall_speedup_x`` / ``stacked_wall_x`` = pristine_seq/compiled).
    """
    from .eval_parity import build_question

    images = [s["image"] for s in samples]
    prompts = [build_question(s["categories"]) for s in samples]

    free()
    our_seq = _seq_wall(worker, images, prompts, max_new_tokens)
    pristine_seq = None
    try:
        pw = _load_pristine_worker(worker, os.path.join("/tmp", "pristine_model"))
        if pw is not None:
            pristine_seq = _seq_wall(pw, images, prompts, max_new_tokens)
            del pw
            free()
    except Exception as e:
        logger.warning("pristine B=1 baseline unavailable (%s); using our-checkout B=1", e)
    seq_wall = pristine_seq if pristine_seq is not None else our_seq
    eager_wall = _timed_predict_all(worker, images, prompts, batch_size, max_new_tokens)
    result = {
        "n": len(samples),
        "batch_size": batch_size,
        "max_new_tokens": max_new_tokens,
        "seq_source": "pristine" if pristine_seq is not None else "our_checkout_B1",
        "pristine_seq_wall_s": round(pristine_seq, 3) if pristine_seq is not None else None,
        "our_seq_wall_s": round(our_seq, 3),
        "seq_wall_s": round(seq_wall, 3),
        "eager_wall_s": round(eager_wall, 3),
        "batching_wall_x": round(seq_wall / eager_wall, 3) if eager_wall else None,
        "configs": [],
    }

    saved_limits = _set_recompile_limit(256)
    orig_lm = worker.model.language_model
    orig_vit = getattr(worker.model, "vision_model", None)
    for spec in configs:
        mode, dynamic = spec[0], spec[1]
        compile_vision = spec[2] if len(spec) > 2 else False
        cfg = {"mode": mode, "dynamic": dynamic, "compile_vision": compile_vision}
        try:
            torch._dynamo.reset()
            worker.model.language_model = torch.compile(orig_lm, mode=mode, dynamic=dynamic)
            if compile_vision and orig_vit is not None:
                worker.model.vision_model = torch.compile(orig_vit, mode=mode, dynamic=dynamic)
            _timed_predict_all(worker, images, prompts, batch_size, max_new_tokens)  # warmup
            cw = _timed_predict_all(worker, images, prompts, batch_size, max_new_tokens)
            cfg["compiled_wall_s"] = round(cw, 3)
            cfg["wall_speedup_x"] = round(eager_wall / cw, 3) if cw else None
            cfg["stacked_wall_x"] = round(seq_wall / cw, 3) if cw else None
        except Exception as e:
            cfg["error"] = f"{type(e).__name__}: {e}".strip().replace("\n", " ")
        finally:
            worker.model.language_model = orig_lm
            if compile_vision and orig_vit is not None:
                worker.model.vision_model = orig_vit
            torch._dynamo.reset()
        result["configs"].append(cfg)
        free()
    _restore_recompile_limit(saved_limits)
    return result
