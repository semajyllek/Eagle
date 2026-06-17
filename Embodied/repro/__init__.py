# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
"""Reproducibility & verification tooling for the batched-inference change.

Private tooling (NOT part of the upstream PRs): used on the working branches to
(1) test/verify changes and (2) generate the evidence cited in PR descriptions.
Every public function is small and importable so snippets are easy to share.

Typical use::

    from repro import load_overlaid_worker, gpu_evidence
    worker, snap = load_overlaid_worker()      # download + overlay this checkout + load
    gpu_evidence(worker)                        # logit/parity/speed tables

CPU-only (no GPU/checkpoint)::

    from repro import cpu_evidence
    cpu_evidence()                              # 144x3 integer-exact checks

Logs via the standard ``logging`` module under the ``repro``
logger; call ``logging.basicConfig(level=logging.INFO)`` (or use
``python -m repro``) to see its messages.
"""
import logging as _logging

_logging.getLogger(__name__).addHandler(_logging.NullHandler())

from .benchmark import run_benchmark
from .compile_probe import compile_probe
from .cpu_evidence import cpu_evidence
from .equivalence import (
    bf16_gemm_demo,
    build_batched_inputs,
    logit_equivalence,
    prefill_logits,
)
from .eval_data import download_eval_data
from .eval_parity import eval_parity, grounded_eval
from .model import load_overlaid_worker, sample_images
from .parity import (
    coord_diff,
    diagnose_row,
    fast_mode_diagnostic,
    grouped_parity,
    parse_boxes,
    parse_refs,
    semantic_parity,
)
from .perf import free, profile_sweep, speed_sweep
from .plots import plot_speed_sweep
from .report import gpu_evidence, write_report
from .runtime import reset_env

__all__ = [
    "load_overlaid_worker",
    "sample_images",
    "cpu_evidence",
    "bf16_gemm_demo",
    "build_batched_inputs",
    "prefill_logits",
    "logit_equivalence",
    "parse_boxes",
    "parse_refs",
    "coord_diff",
    "semantic_parity",
    "diagnose_row",
    "fast_mode_diagnostic",
    "grouped_parity",
    "free",
    "speed_sweep",
    "profile_sweep",
    "compile_probe",
    "plot_speed_sweep",
    "download_eval_data",
    "eval_parity",
    "grounded_eval",
    "run_benchmark",
    "gpu_evidence",
    "write_report",
    "reset_env",
]
