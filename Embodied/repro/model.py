# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
"""Model loading for reproducibility runs.

`trust_remote_code` executes the code bundled in the HF model repo, so to test
*this checkout's* changes we download the snapshot, overlay our patched files
onto it, and load from that local directory. `load_overlaid_worker` does all of
that and returns a ready `LocateAnythingWorker`.
"""
import io
import logging
import os
import sys

import torch

logger = logging.getLogger(__name__)

_HERE = os.path.dirname(os.path.abspath(__file__))
_EMBODIED = os.path.join(_HERE, "..")
sys.path.insert(0, _EMBODIED)  # for locateanything_worker
sys.path.insert(0, _HERE)  # for overlay

from overlay import overlay  # noqa: E402

DEFAULT_MODEL = "nvidia/LocateAnything-3B"


def load_overlaid_worker(
    model_id=DEFAULT_MODEL,
    device="cuda",
    dtype=torch.bfloat16,
    snapshot_dir=None,
    src_dir=None,
    verbose=True,
):
    """Download `model_id`, overlay patched code, and load a worker.

    By default overlays *this* checkout's code. Pass ``src_dir`` (a
    ``.../eaglevl/utils/locany`` dir from a specific branch checkout) to overlay
    that branch's code instead — this is how each per-PR notebook measures its own
    branch: clone the PR, point ``src_dir`` at its locany, and put the PR's
    ``Embodied`` on ``sys.path`` so its worker is the one imported here.

    Returns ``(worker, snapshot_dir)``. The overlay refuses to proceed unless it
    replaces the modules the model actually loads (see overlay.py).
    """
    from huggingface_hub import snapshot_download

    if src_dir:
        # the matching worker lives at <Embodied> = three levels above locany;
        # put it first so we import the PR's worker, not the tooling clone's.
        emb = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(src_dir)))
        )
        sys.path.insert(0, emb)
    from locateanything_worker import LocateAnythingWorker

    snap = snapshot_dir or snapshot_download(model_id)
    overlay(snap, verbose=verbose, **({"src_dir": src_dir} if src_dir else {}))
    worker = LocateAnythingWorker(snap, device=device, dtype=dtype)
    if verbose:
        logger.info(
            "loaded overlaid model from %s (src_dir=%s)",
            snap,
            src_dir or "this checkout",
        )
    return worker, snap


# ---------------------------------------------------------------------------
# Sample inputs (real images with a synthetic fallback if offline).
# ---------------------------------------------------------------------------
SAMPLE_URLS = [
    "https://ultralytics.com/images/bus.jpg",
    "https://ultralytics.com/images/zidane.jpg",
    "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg",
]

SAMPLE_PROMPTS = [
    "Locate all the instances that matches the following description: person.",
    "Detect all the text in box format.",
    "Locate all the instances that matches the following description: bus</c>car</c>person.",
    "Point to: the main subject.",
    "Locate a single instance that matches the following description: dog.",
    "Locate all the instances that match the following description: vehicle.",
    "Locate all the instances that matches the following description: object.",
    "Point to: the largest object.",
]


def _synthetic(seed):
    import random

    from PIL import Image, ImageDraw

    random.seed(seed)
    im = Image.new("RGB", (640, 480), (235, 235, 235))
    d = ImageDraw.Draw(im)
    for _ in range(4):
        x0, y0 = random.randint(0, 500), random.randint(0, 360)
        d.rectangle(
            [x0, y0, x0 + random.randint(40, 120), y0 + random.randint(40, 100)],
            fill=tuple(random.randint(0, 255) for _ in range(3)),
        )
    d.text((20, 20), f"synthetic {seed}", fill=(0, 0, 0))
    return im


def sample_images(n):
    """Return ``(images, prompts)`` of length ``n`` — real images when reachable,
    synthetic fallbacks otherwise."""
    import requests
    from PIL import Image

    images = []
    for i in range(n):
        try:
            r = requests.get(SAMPLE_URLS[i % len(SAMPLE_URLS)], timeout=10)
            r.raise_for_status()
            images.append(Image.open(io.BytesIO(r.content)).convert("RGB"))
        except Exception as e:
            logger.warning("image url %d failed (%s); using synthetic", i, e)
            images.append(_synthetic(i))
    prompts = [SAMPLE_PROMPTS[i % len(SAMPLE_PROMPTS)] for i in range(n)]
    return images, prompts
