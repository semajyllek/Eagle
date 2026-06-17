# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
"""
Overlay our patched LocateAnything modeling code onto a downloaded HF snapshot.

WHY THIS EXISTS
---------------
``AutoModel.from_pretrained("nvidia/LocateAnything-3B", trust_remote_code=True)``
executes the ``.py`` files **bundled in the HF model repo**, not the files in this
GitHub checkout. So to test our batched-decode changes we must:

  1. download the model snapshot (weights + config + tokenizer + bundled code),
  2. overwrite the bundled modeling/util ``.py`` files with our patched copies
     (and add the new ``batched_generate.py``),
  3. load the model from that *local* directory with ``trust_remote_code=True``.

This script does step 2. The bundled remote code lives flat at the snapshot root
(referenced by ``auto_map`` in ``config.json``); our source lives under
``eaglevl/utils/locany/``. We copy by matching basename, overwriting the files
that exist in the snapshot and adding any new ones (e.g. ``batched_generate.py``)
that our patched code imports.

USAGE
-----
    from huggingface_hub import snapshot_download
    snap = snapshot_download("nvidia/LocateAnything-3B")
    python overlay_patched_code.py --snapshot $snap          # CLI, or
    overlay(snap)                                            # importable
"""
import argparse
import logging
import os
import shutil

logger = logging.getLogger(__name__)

HERE = os.path.dirname(os.path.abspath(__file__))
LOCANY_DIR = os.path.normpath(os.path.join(HERE, "..", "eaglevl", "utils", "locany"))

# Files our patched code touches or newly requires. We overlay the full coherent
# set so the snapshot never mixes our versions with bundled ones of different
# vintage (which could desync the relative imports / config field names).
OVERLAY_FILES = [
    "modeling_locateanything.py",  # patched: assert removed + batched dispatch
    "modeling_qwen2.py",  # patched: per-row mask builder
    "batched_generate.py",  # new: batched driver
    "generate_utils.py",
    "mask_sdpa_utils.py",
    "mask_magi_utils.py",
    "attn_mask_utils.py",
    "configuration_locateanything.py",
    "configuration_qwen2.py",
    "image_processing_locateanything.py",
    "processing_locateanything.py",
    "modeling_vit.py",
]


def _auto_map_modules(snapshot_dir):
    """Return the set of module filenames the model actually loads, read from
    ``config.json``'s ``auto_map`` (e.g. 'modeling_locateanything.py'). Empty if
    none found."""
    import json

    cfg_path = os.path.join(snapshot_dir, "config.json")
    if not os.path.exists(cfg_path):
        return set()
    cfg = json.load(open(cfg_path))
    mods = set()
    for v in (cfg.get("auto_map") or {}).values():
        vals = v if isinstance(v, list) else [v]
        for entry in vals:
            mods.add(entry.split(".")[0] + ".py")
    return mods


def overlay(
    snapshot_dir, files=OVERLAY_FILES, backup=True, verbose=True, src_dir=LOCANY_DIR
):
    """Copy patched ``.py`` files from ``src_dir`` into ``snapshot_dir`` (flat).
    Returns the list of (src, dst, action) it performed.

    ``src_dir`` defaults to this checkout's ``eaglevl/utils/locany``. Pass a dir
    materialized from a specific git ref (see ``branch_code.materialize_ref``) to
    measure another branch's code in isolation — no frankenstein checkout. Only
    files present in both ``files`` and ``src_dir`` are overlaid.
    """
    if not os.path.isdir(snapshot_dir):
        raise FileNotFoundError(f"snapshot dir not found: {snapshot_dir}")
    files = [f for f in files if os.path.exists(os.path.join(src_dir, f))]

    # Guard against HF layout drift: every module the model loads via auto_map
    # MUST be one we overlay AND must already exist (so we replace the file that
    # is actually executed, not silently add a dead one alongside it).
    auto_mods = _auto_map_modules(snapshot_dir)
    missing_overlay = sorted(m for m in auto_mods if m not in set(files))
    if missing_overlay:
        raise RuntimeError(
            "auto_map references modules we do NOT overlay: "
            f"{missing_overlay}. The HF repo layout differs from this checkout; "
            "add these to OVERLAY_FILES (and reconcile any code differences) "
            "before trusting the results."
        )

    actions = []
    for name in files:
        src = os.path.join(src_dir, name)
        if not os.path.exists(src):
            raise FileNotFoundError(f"patched source missing: {src}")
        dst = os.path.join(snapshot_dir, name)
        existed = os.path.exists(dst) or os.path.islink(dst)
        if existed and backup and not os.path.exists(dst + ".orig"):
            shutil.copy2(dst, dst + ".orig")
        # HF snapshots are trees of symlinks into a blobs/ store. Writing through
        # a symlink keeps it a symlink, and trust_remote_code then realpath()s the
        # entry module into blobs/ and can't find its siblings (FileNotFoundError
        # on blobs/<module>.py). Remove the link first so we write a REAL file and
        # the whole import graph resolves within the snapshot dir.
        if os.path.islink(dst) or os.path.exists(dst):
            os.remove(dst)
        shutil.copy2(src, dst)
        action = "overwrote" if existed else "added (NEW)"
        actions.append((src, dst, action))
        if verbose:
            logger.info("%-14s %s", action, name)
    # Every auto_map module must have been REPLACED (existed before), proving we
    # patched the code that actually runs rather than adding an ignored copy.
    added_new = {os.path.basename(d) for _, d, a in actions if a.startswith("added")}
    auto_added = sorted(auto_mods & added_new)
    if auto_added:
        raise RuntimeError(
            f"auto_map modules were ADDED, not overwritten: {auto_added}. They did "
            "not exist in the snapshot, so the model may load different code. Aborting."
        )

    # Sanity: confirm the assert is gone and the dispatch is present.
    target = os.path.join(snapshot_dir, "modeling_locateanything.py")
    text = open(target).read()
    assert (
        "assert batch_size == 1" not in text
    ), "overlay failed: stale batch_size==1 assert still present"
    assert "batched_generate" in text, "overlay failed: batched dispatch not found"
    if verbose:
        if auto_mods:
            logger.info("auto_map modules replaced: %s", sorted(auto_mods))
        logger.info("Overlay verified: batch_size==1 assert removed, dispatch wired.")
    return actions


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--snapshot", required=True, help="path from snapshot_download(...)"
    )
    ap.add_argument("--no-backup", action="store_true")
    args = ap.parse_args()
    overlay(args.snapshot, backup=not args.no_backup)
