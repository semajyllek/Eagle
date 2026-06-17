# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
"""The benchmark: run the diagnostic suite against an already-loaded worker and
save a single tagged record (``results_<tag>.json``).

Each PR has its own notebook that (1) checks out *that PR's* branch, (2) loads the
worker from it, and (3) calls ``run_benchmark`` with metadata for the record. The
model under test is whatever the notebook loaded — there is no code-swapping here,
so a reviewer can trust that PR1's notebook measured PR1's code. ``repro.combine``
then merges the saved records into the comparison artifacts.
"""
import json
import logging
import os
import time

logger = logging.getLogger(__name__)


def _decode_bound_inputs(datasets, evaldata, n):
    """``n`` *decode-bound* eval images (long outputs -> the lm forward is the bulk
    of wall, where compile pays off). Falls back to short synthetic ``sample_images``
    if the eval data isn't reachable. Returns ``(images, prompts, max_new_tokens)``."""
    ds = (datasets or ("COCO",))[0]
    try:
        from .eval_data import download_eval_data
        from .eval_parity import build_question, load_samples

        download_eval_data((ds,), dest=evaldata)
        jsonl = os.path.join(evaldata, "_annotations", "box_eval", f"{ds}.jsonl")
        samples = load_samples(jsonl, evaldata, max(n, 8))
        if len(samples) >= n:
            sel = samples[:n]
            imgs = [s["image"] for s in sel]
            prompts = [build_question(s["categories"]) for s in sel]
            logger.info("decode-bound inputs from %s (%d imgs)", ds, len(imgs))
            return imgs, prompts, 512
        logger.warning("only %d %s samples; using sample_images", len(samples), ds)
    except Exception as e:
        logger.warning(
            "decode-bound inputs unavailable (%s); falling back to sample_images", e
        )
    from .model import sample_images

    imgs, prompts = sample_images(n)
    return imgs, prompts, 128


def run_benchmark(worker, cfg):
    """Run ``cfg.tasks`` against ``worker``, write ``results_<tag>_bs<bs>.json``, return the record.

    Everything the run needs comes off one config object (the notebook's ``CFG``
    ``SimpleNamespace``, or any object exposing these attributes) -- no scattered
    call-site globals:

      ``tag``         base run label, e.g. "PR3"     (REQUIRED)
      ``tasks``       which tasks to run             (default: eval + grounded)
      ``datasets``    detection / mAP datasets       (default: COCO + LVIS)
      ``n_limit``     images per detection dataset   (default: 50)
      ``batch_size``  batch size                     (default: 8)
      ``evaldata``    EvalData root dir              (default: /content/EvalData)
      ``model_branch`` / ``commit``  provenance only, stored in metadata (optional)

    ``run_tag`` (``tag`` at bs8, else ``tag@bs<N>``), the metadata block, and the output
    path are derived here and written back onto ``cfg`` (``cfg.run_tag``, ``cfg.out``) so
    the caller can reference them (e.g. to download the file). Every task calls only
    ``worker.predict`` /
    ``worker.predict_batch`` — the real shipped API — so each record reflects
    whatever this checkout's code actually does, with no special-casing between
    branches. ``speed`` adds ``speed_sweep`` (sequential vs batched, N distinct
    images); ``grounded`` adds ``grounded_eval`` (one image, many queries, via
    ``predict_batch([img]*K, qs)``); ``profile`` adds ``profile_sweep`` (per-step
    decode-loop timing breakdown at larger batch sizes / ``fast`` mode, isolating
    bookkeeping overhead from the model forward); ``compile_probe`` adds
    ``compile_probe`` (one steady-state decode-step call, eager vs.
    ``torch.compile(mode="reduce-overhead")``, isolated); ``compile_loop`` adds
    ``compile_loop_probe`` (the same eager-vs-compile comparison over the REAL
    ragged decode loop -- ``Ckv`` grows every step, ``A`` shrinks on early-eject --
    reporting recompiles and compiled-vs-eager parity, i.e. whether the static
    probe's win survives dynamic shapes); ``compile_parity`` adds
    ``compile_parity_probe`` (eval-scale compiled-vs-eager DETECTION parity over
    ``limit`` samples, so compile's divergence rate is comparable to the eval's
    batched parity rather than ``compile_loop``'s n=8); ``compile_ap`` adds
    ``compile_ap_probe`` (writes paired eager/compiled prediction files over 100
    samples for OFFLINE mAP scoring -> mAP(eager) vs mAP(compiled), i.e. whether
    compile's parity divergence costs real detections); ``stacked_eval`` adds
    ``stacked_eval_probe`` (the real-eval-workload speedup ladder -- N eval images,
    multi-batch, 2048 tokens -- measuring B=1-pristine -> batched -> batched+compiled
    as one ratio vs the unmodified model). ``repro.combine`` then puts each branch's
    numbers for the same call side by side.
    """
    if not getattr(cfg, "tag", None):
        raise ValueError("cfg.tag is required (the run label, e.g. 'PR3')")
    tasks = getattr(cfg, "tasks", ("eval", "grounded"))
    datasets = getattr(cfg, "datasets", ("COCO", "LVIS"))
    limit = getattr(cfg, "n_limit", 50)
    batch_size = getattr(cfg, "batch_size", 8)
    evaldata = getattr(cfg, "evaldata", "/content/EvalData")
    run_tag = cfg.tag if batch_size == 8 else f"{cfg.tag}@bs{batch_size}"  # bs8 = canonical
    out = f"results_{cfg.tag.lower()}_bs{batch_size}.json"
    try:                                    # surface derived values back on the config
        cfg.run_tag, cfg.out = run_tag, out
    except Exception:
        pass
    record = {
        "tag": run_tag,
        "metadata": {
            "branch": getattr(cfg, "model_branch", None),
            "commit": getattr(cfg, "commit", None),
            "batch_size": batch_size,
            "generated": time.strftime("%Y-%m-%d %H:%M"),
        },
    }

    if "parity" in tasks:
        from .model import sample_images
        from .parity import semantic_parity

        imgs, prompts = sample_images(4)
        record["semantic_parity"] = semantic_parity(worker, imgs, prompts)["rows"]

    if "speed" in tasks:
        from .model import sample_images
        from .perf import speed_sweep

        imgs, prompts = sample_images(4)
        record["speed"] = speed_sweep(
            worker, imgs[0], prompts[0], batch_sizes=(1, 2, 4, 8, 16)
        )

    if "profile" in tasks:
        from .model import sample_images
        from .perf import profile_sweep

        imgs, prompts = sample_images(4)
        record["profile"] = profile_sweep(worker, imgs[0], prompts[0])

    if "compile_probe" in tasks:
        from .compile_probe import compile_probe
        from .model import sample_images

        imgs, prompts = sample_images(4)
        record["compile_probe"] = compile_probe(worker, imgs[0], prompts[0])

    if "compile_loop" in tasks:
        from .compile_probe import compile_loop_probe

        cl_imgs, cl_prompts, cl_mnt = _decode_bound_inputs(datasets, evaldata, batch_size)
        record["compile_loop"] = [
            compile_loop_probe(
                worker, cl_imgs, cl_prompts, batch_size=batch_size,
                max_new_tokens=cl_mnt,
                configs=(("default", True, False), ("default", True, True)),
            )
        ]

    if "compile_parity" in tasks:
        from .compile_probe import compile_parity_probe

        # Eval-scale (`limit` samples) compiled-vs-eager DETECTION parity, so the
        # divergence rate is comparable to the eval's batched-vs-B1 parity (n/50),
        # not the n=8 of compile_loop. Decoder-only and decoder+ViT.
        cp_imgs, cp_prompts, cp_mnt = _decode_bound_inputs(datasets, evaldata, limit)
        record["compile_parity"] = compile_parity_probe(
            worker, cp_imgs, cp_prompts, batch_size=batch_size, max_new_tokens=cp_mnt,
            configs=(("default", True, False), ("default", True, True)),
        )

    if "compile_ap" in tasks:
        from .ap_score import score_compile_ap
        from .compile_probe import compile_ap_probe
        from .eval_data import download_eval_data, ensure_gt_json
        from .eval_parity import load_samples

        # For each dataset: 100 real samples through eager vs compiled (shipped
        # settings), write paired preds_*.jsonl, then score mAP(eager) vs
        # mAP(compiled) inline via the repo metric -- the definitive "does compile
        # cost real detections" gate (parity only says how often outputs differ).
        record["compile_ap"] = {}
        for ds in (datasets or ("COCO",)):
            try:
                download_eval_data((ds,), dest=evaldata)
                jsonl = os.path.join(evaldata, "_annotations", "box_eval", f"{ds}.jsonl")
                ap_samples = load_samples(jsonl, evaldata, 100)
            except Exception as e:
                logger.warning("compile_ap[%s]: samples unavailable (%s); skipping", ds, e)
                continue
            if not ap_samples:
                continue
            out_dir = os.path.join(evaldata, "compile_ap_out", ds)
            res = compile_ap_probe(
                worker, ap_samples, out_dir=out_dir, batch_size=batch_size
            )
            res = score_compile_ap(
                res, ensure_gt_json(ds, evaldata), ds.lower(),
                os.path.join(out_dir, "tsv"),
            )
            record["compile_ap"][ds] = res

    if "stacked_eval" in tasks:
        from .compile_probe import stacked_eval_probe
        from .eval_data import download_eval_data
        from .eval_parity import load_samples

        # The REAL production speedup as one measured ratio: N real eval images
        # through the multi-batch shipped path at eval-length tokens (2048), vs the
        # PRISTINE original model. N=16 keeps the 2N sequential B=1 generates
        # tractable; compile_loop's short-workload ladder understates compile's share.
        ds = (datasets or ("COCO",))[0]
        se_samples = []
        try:
            download_eval_data((ds,), dest=evaldata)
            jsonl = os.path.join(evaldata, "_annotations", "box_eval", f"{ds}.jsonl")
            se_samples = load_samples(jsonl, evaldata, 16)
        except Exception as e:
            logger.warning("stacked_eval: samples unavailable (%s); skipping", e)
        if se_samples:
            record["stacked_eval"] = stacked_eval_probe(
                worker, se_samples, batch_size=batch_size, max_new_tokens=2048,
            )

    if "eval" in tasks:
        from .eval_parity import eval_parity

        record["eval"] = eval_parity(
            worker,
            datasets=datasets,
            evaldata=evaldata,
            limit=limit,
            batch_size=batch_size,
            download=True,
        )

    if "grounded" in tasks:
        from .eval_parity import grounded_eval

        # RefCOCOg = real one-image-many-referring-expressions (its images are
        # the COCO archive already fetched by the detection step; only the small
        # jsonl is new, so download=True is cheap).
        record["grounded"] = grounded_eval(
            worker,
            datasets=("RefCOCOg_val",),
            evaldata=evaldata,
            limit=max(limit * 8, 400),
            download=True,
        )

    with open(out, "w") as f:
        json.dump(record, f, indent=2)
    logger.info("wrote %s (tag=%s)", out, run_tag)
    return record
