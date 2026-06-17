# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
"""Download the COCO/LVIS evaluation data used by ``eval_parity``.

Pulls just the requested datasets' annotation JSONLs and image archives from the
``Rex-Omni-EvalData`` HF dataset (not the whole thing) and extracts them. Public
dataset — no token required.
"""
import logging
import os

logger = logging.getLogger(__name__)

DEFAULT_REPO = "Mountchicken/Rex-Omni-EvalData"

# Which image tarball each split's `image_path` actually lives in. Several splits
# share the COCO images (LVIS is COCO-image annotations; RefCOCOg references COCO
# train2014) so there is no `lvis.tar.gz` / `refcocog_*.tar.gz` — they need
# `coco.tar.gz`. Anything not listed defaults to ``<dataset>.tar.gz``.
_ARCHIVE = {
    "COCO": "coco.tar.gz",
    "LVIS": "coco.tar.gz",
    "RefCOCOg_val": "coco.tar.gz",
    "RefCOCOg_test": "coco.tar.gz",
    "HumanRef": "humanref.tar.gz",
}


def download_eval_data(
    datasets=("COCO", "LVIS"), dest="/content/EvalData", repo=DEFAULT_REPO
):
    """Download + extract the eval data for ``datasets`` into ``dest``.

    For each dataset it fetches ``<DATASET>.jsonl`` (annotations) and the image
    archive it needs (``_ARCHIVE`` map; default ``<dataset>.tar.gz``) by matching
    the repo file list, then unpacks. Returns ``dest``. Skips files that don't
    exist in the repo.
    """
    import tarfile

    from huggingface_hub import hf_hub_download, list_repo_files

    os.makedirs(dest, exist_ok=True)
    jsonls = tuple(f"{d}.jsonl" for d in datasets)
    archives = {_ARCHIVE.get(d, f"{d.lower()}.tar.gz") for d in datasets}
    want = [
        f
        for f in list_repo_files(repo, repo_type="dataset")
        if f.endswith(jsonls) or os.path.basename(f).lower() in archives
    ]
    logger.info("downloading %d file(s) from %s: %s", len(want), repo, want)
    for f in want:
        p = hf_hub_download(repo, f, repo_type="dataset", local_dir=dest)
        if p.endswith(".tar.gz"):
            logger.info("extracting %s", os.path.basename(p))
            with tarfile.open(p) as t:
                t.extractall(dest)
    logger.info("eval data ready at %s", dest)
    return dest


# GT json each dataset's metric needs. The two sources differ: COCO's
# instances_val2017.json is NOT a loose repo file -- it ships inside coco.tar.gz and
# lands at coco/ after extraction -- so it's local-only. LVIS's GT IS a loose repo
# file, but under ``missing_annotaitons/`` (sic, repo's typo), not coco/. Each entry
# is ``(local candidates to check first, repo path to download or None)``.
_GT_JSON = {
    "COCO": (["coco/instances_val2017.json", "instances_val2017.json"], None),
    "LVIS": (
        ["missing_annotaitons/lvis_v1_val_with_filename2.json",
         "coco/lvis_v1_val_with_filename2.json"],
        "missing_annotaitons/lvis_v1_val_with_filename2.json",
    ),
}


def ensure_gt_json(dataset, dest="/content/EvalData", repo=DEFAULT_REPO):
    """Return the local path to ``dataset``'s GT json -- checking where the image
    archive may have extracted it, then downloading the loose repo file if there is
    one. ``None`` if the dataset has no known GT or it can't be obtained (caller
    degrades to pred-files-only)."""
    entry = _GT_JSON.get(dataset)
    if not entry:
        return None
    candidates, repo_path = entry
    for c in candidates:
        local = os.path.join(dest, c)
        if os.path.exists(local):
            return local
    if not repo_path:
        logger.warning("GT json for %s not found locally (expected in archive): %s",
                       dataset, candidates[0])
        return None
    try:
        from huggingface_hub import hf_hub_download

        return hf_hub_download(repo, repo_path, repo_type="dataset", local_dir=dest)
    except Exception as e:
        logger.warning("GT json %s unavailable (%s)", repo_path, e)
        return None
