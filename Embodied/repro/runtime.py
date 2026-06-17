# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
"""Runtime housekeeping for the benchmark notebook — switch PR/batch size without
losing the GPU runtime (so compile/stacked numbers stay on the same hardware) or the
cached 7 GB weights.

``reset_env`` is what the notebook's "switch" cell calls between branches. The one
subtle part is the ``transformers_modules`` purge: the overlay overwrites the
snapshot's remote code on disk, but ``trust_remote_code`` caches the *imported* module
in ``sys.modules`` and on disk under ``~/.cache``. Without dropping both, the next run
silently keeps executing the previous branch's code.
"""
import gc
import logging
import os
import shutil
import sys

logger = logging.getLogger(__name__)


def reset_env(namespace=None, free=("worker", "snap"), model_dir="/content/model"):
    """Free the loaded model and clear caches so the next branch loads cleanly,
    while keeping the GPU runtime and downloaded weights.

    ``namespace``: pass ``globals()`` from the notebook so the model references it
    holds (``worker``/``snap`` by default — see ``free``) are actually dropped; a
    function can't release the *caller's* globals otherwise, and without that the model
    stays alive and ``empty_cache`` reclaims nothing. ``model_dir``: the cloned model
    checkout to remove so setup re-clones the new branch (pass ``None`` locally, where
    the "model dir" is your real repo and must not be deleted).

    Returns a short status string (handy to ``print``).
    """
    if namespace is not None:
        for name in free:
            namespace.pop(name, None)
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    # drop cached trust_remote_code modules (sys.modules + on-disk) so the next
    # overlay's code is actually re-imported rather than served from cache.
    for _k in [k for k in sys.modules if k.startswith("transformers_modules")]:
        del sys.modules[_k]
    shutil.rmtree(
        os.path.expanduser("~/.cache/huggingface/modules/transformers_modules"),
        ignore_errors=True,
    )
    if model_dir:
        shutil.rmtree(model_dir, ignore_errors=True)

    msg = "ready — edit the CONFIG cell, then re-run Setup + Run (GPU kept)"
    logger.info(msg)
    return msg
