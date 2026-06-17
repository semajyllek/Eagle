# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
"""Generate the two repro notebooks:

- ``benchmark.ipynb`` — a single, parameterized notebook: set ``MODEL_BRANCH``, run, and it
  evaluates *that* PR branch against the **pristine** model and saves ``results_<tag>.json``.
- ``combine_results.ipynb`` — merge the per-branch records into ``repro/evidence/EVIDENCE.md``.

All explanation/diffs/methodology live in ``repro/evidence/EVIDENCE.md`` (via
``evidence_narrative.py`` + ``evidence/code_changes.md``), so these notebooks are thin.

    python repro/build_notebook.py     # (re)writes benchmark.ipynb + combine_results.ipynb
"""
import json
import os

REPO_URL = "github.com/semajyllek/LocateAnything-batched.git"
HERE = os.path.dirname(os.path.abspath(__file__))

cells = []


def _reset():
    global cells
    cells = []


def md(t):
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": t.strip("\n").splitlines(keepends=True)})


def code(t):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": t.strip("\n").splitlines(keepends=True)})


def _write(path):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
            "colab": {"provenance": []},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(nb, f, indent=1)
    # sanity: every code cell must be valid Python
    for c in cells:
        if c["cell_type"] == "code":
            compile("".join(c["source"]), path, "exec")
    print("wrote", path, "with", len(cells), "cells")


_PIP = r"""subprocess.run([sys.executable, "-m", "pip", "-q", "install",
                "transformers==4.57.1", "tokenizers==0.22.0", "accelerate",
                "peft==0.12.0", "timm", "huggingface_hub", "pandas", "tqdm",
                "matplotlib", "decord", "lmdb", "requests"], check=False)"""


def benchmark_notebook():
    """The single parameterized benchmark notebook (any PR branch vs pristine)."""
    _reset()
    md("""
# LocateAnything — batched/compiled benchmark

One notebook to evaluate **any** PR branch against the **pristine** (unmodified) model and
produce its `results_<tag>.json`. Set `MODEL_BRANCH` below, run all cells; then feed the
result(s) to `combine_results.ipynb` -> `repro/evidence/EVIDENCE.md`.

- **Tooling** is always cloned from `feature/batched-generation` (the dev branch that holds
  `repro/`); the **model** is cloned from `MODEL_BRANCH` and overlaid onto the HF snapshot,
  so the run measures that branch's own code (no swapping).
- The pristine baseline is loaded *inside* the benchmark (the stacked/compile tasks), so the
  speedups are vs the unmodified model.

**Code changes, exact diffs, testing methodology, and per-change speed are all in the
deep-dive `repro/evidence/EVIDENCE.md`** — this notebook is just setup + run.
""")
    md("""
## 0 · Config — everything you'd tweak, in one place

Edit this cell, then **Run all**. Every knob the run cell needs lives here and is passed
through unchanged — nothing to hunt for further down. `DATASETS` is the *detection* (mAP)
set; grouping is measured by the `grounded` task on RefCOCOg, which is pinned there and
ignores `DATASETS` (one-image-many-queries is a different workload).
""")
    code(r"""
# === CONFIG — edit, then Run all ===
from types import SimpleNamespace
CFG = SimpleNamespace(
    model_branch = "pr3/vectorized-decode",  # pr/batched-generation | pr2/grouped-batch | pr3/vectorized-decode | pr4/grouped-vectorized
    batch_size   = 8,                        # rerun at 16/32/... (same runtime, see the switch cell) for the compile/stacked batch-size curve
    n_limit      = 50,                       # images per detection dataset (eval / compile tasks)
    datasets     = ("COCO", "LVIS"),         # detection mAP datasets; grouping uses RefCOCOg (pinned in the grounded task)
    tasks        = ("eval", "grounded", "speed", "profile",
                    "compile_probe", "compile_loop", "compile_ap", "stacked_eval"),
    evaldata     = "/content/EvalData",      # EvalData root (download dest + image root)
)   # setup adds CFG.tag / CFG.commit; the run writes back CFG.run_tag / CFG.out
""")
    md("## 1 · Setup — clone tooling + model, prepare the runtime")
    code(rf"""
import os, sys, getpass, logging, subprocess, shutil
logging.basicConfig(level=logging.INFO, format="%(message)s")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["GIT_LFS_SKIP_SMUDGE"] = "1"   # fork lacks upstream LFS blobs; code only
_TAG = {{"pr/batched-generation": "PR1", "pr2/grouped-batch": "PR2",
        "pr3/vectorized-decode": "PR3", "pr4/grouped-vectorized": "PR4"}}
CFG.tag = _TAG.get(CFG.model_branch, CFG.model_branch.split("/")[-1])
try:
    import google.colab  # noqa
    IN_COLAB = True
except Exception:
    IN_COLAB = False
TOOL, MODEL = "/content/tool", "/content/model"
if IN_COLAB:
    {_PIP}
    tok = getpass.getpass("GitHub token (Contents: read): ").strip()
    for d, branch in [(TOOL, "feature/batched-generation"), (MODEL, CFG.model_branch)]:
        if os.path.isdir(os.path.join(d, "Embodied")):
            continue
        if os.path.exists(d):
            shutil.rmtree(d)
        res = subprocess.run(
            ["git", "clone", "-b", branch, f"https://{{tok}}@{REPO_URL}", d],
            capture_output=True, text=True, env={{**os.environ}})
        if res.returncode != 0 or not os.path.isdir(os.path.join(d, "Embodied")):
            raise RuntimeError("clone of %s failed:\\n%s" % (branch, (res.stderr or res.stdout).replace(tok, "***")))
else:
    TOOL = MODEL = subprocess.run(["git", "rev-parse", "--show-toplevel"],
                                  capture_output=True, text=True).stdout.strip()
sys.path.insert(0, os.path.join(TOOL, "Embodied"))      # repro tooling
MODEL_EMB = os.path.join(MODEL, "Embodied")
MODEL_LOCANY = os.path.join(MODEL_EMB, "eaglevl", "utils", "locany")
CFG.commit = subprocess.run(["git", "-C", MODEL, "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
import torch
RUN_GPU = torch.cuda.is_available()
logging.getLogger(__name__).info("tag=%s  model=%s@%s  GPU=%s", CFG.tag, CFG.model_branch, CFG.commit, RUN_GPU)
""")
    md("""
## Code changes & deep dive

The exact code changes (motivation + diffs + how-tested + per-change speed), the full
correctness methodology (CPU tests, semantic parity, the mAP gate), and all measured
performance live in the self-contained deep-dive **`repro/evidence/EVIDENCE.md`** (built by
`repro.combine` from the `results_*.json` this notebook produces).
""")
    md("""
## 2 · Run the benchmark (GPU) and save the record

Loads `MODEL_BRANCH`'s code overlaid onto the snapshot and runs the shared suite, all via
the real shipped API (`worker.predict` / `worker.predict_batch`): detection (batched-vs-B=1
+ parity + length buckets + early-eject A/B), grounded (one-image-many-queries), the speed
sweep, the per-step profile, the compile probes (feasibility, real-loop A/B, mAP gate vs
eager), and the stacked ladder (B=1-pristine -> batched -> batched+compiled, vs the
unmodified model). Writes `results_<tag>.json`. See `EVIDENCE.md` for what each table means.
""")
    code(r"""
import pandas as pd
from IPython.display import Markdown, display
if RUN_GPU:
    from repro import load_overlaid_worker, run_benchmark
    from repro.combine import (detection_table, eject_table, length_buckets_table,
                               grounded_table, speed_plot, profile_table,
                               compile_probe_table, compile_loop_table,
                               compile_ap_table, stacked_eval_table)
    worker, snap = load_overlaid_worker(src_dir=MODEL_LOCANY)
    rec = run_benchmark(worker, CFG)   # reads CFG.tasks/datasets/n_limit/batch_size/evaldata/tag; writes CFG.run_tag, CFG.out
    res = {rec["tag"]: rec}
    print("Detection (batched vs B=1):"); display(Markdown(detection_table(res)))
    print("Throughput sweep (sequential vs batched):"); display(Markdown(speed_plot(res)))
    print("Output-length spread + early-eject A/B:"); display(Markdown(eject_table(res)))
    display(Markdown(length_buckets_table(res)))
    print("One image, many queries:"); display(Markdown(grounded_table(res)))
    print("Decode-loop bookkeeping profile:"); display(Markdown(profile_table(res)))
    print("torch.compile feasibility probe:"); display(Markdown(compile_probe_table(res)))
    print("Real decode-loop torch.compile A/B:"); display(Markdown(compile_loop_table(res)))
    print("Compile mAP gate:"); display(Markdown(compile_ap_table(res)))
    print("Real-eval-workload stacked speedup (vs pristine):"); display(Markdown(stacked_eval_table(res)))
    try:
        from google.colab import files; files.download(CFG.out)
    except Exception as e:
        print("download skipped:", e)
else:
    print("No GPU — run on a GPU runtime to produce the record.")
""")
    md("""
## 3 · Visualize parity (optional, post-hoc — no effect on the benchmark)

Draws the detected boxes + `<ref>` labels **side by side** for two passes, reading the
prediction files the run already wrote (`compile_ap`'s `preds_eager.jsonl` vs
`preds_<compiled>.jsonl`). This touches *no* timed code — pure post-processing of saved
outputs. `only_diffs=True` shows only the images where eager and compiled disagree at the
detection gate (the near-ties behind the ~1% mAP delta); flip it to `False` to eyeball
matched detections. Point it at `preds_before.jsonl`/`preds_after.jsonl` for B=1-vs-batched.
""")
    code(r"""
import os
_d = os.path.join(CFG.evaldata, "compile_ap_out", CFG.datasets[0])   # compile_ap wrote preds_<tag>.jsonl here
if os.path.exists(os.path.join(_d, "preds_eager.jsonl")):
    from repro.viz import parity_overlay
    import matplotlib.pyplot as plt
    fig = parity_overlay(os.path.join(_d, "preds_eager.jsonl"),
                         os.path.join(_d, "preds_default_dyn.jsonl"),
                         image_root=CFG.evaldata, labels=("eager", "compiled"),
                         n=4, only_diffs=False)     # only_diffs=True -> just the disagreements
    if fig is not None:
        plt.show()
else:
    print("Run the benchmark first — compile_ap writes the preds this reads.")
""")
    md("""
## 🔄 Switch PR / batch size — keep the runtime (same GPU)

Run this cell, then edit the **CONFIG** cell (`model_branch` / `batch_size`) and re-run setup
+ run. Keeps the exact GPU (compile/stacked numbers vary by hardware, so same-GPU comparison
matters) and the cached weights (no 7 GB re-download). `repro.reset_env` frees the model, clears
the CUDA cache, and purges the cached `trust_remote_code` modules — that purge is the key step:
the overlay overwrites the snapshot's code, but `trust_remote_code` caches the *imported* module,
so without it you'd silently keep running the previous branch's code. Passing `globals()` lets it
drop this notebook's `worker`/`snap` references (a function can't free the caller's globals
otherwise); `model_dir` is removed so setup re-clones the new branch.
""")
    code(r"""
from repro import reset_env
print(reset_env(globals(), model_dir="/content/model" if IN_COLAB else None))
""")
    _write(os.path.join(HERE, "notebooks", "benchmark.ipynb"))


def combine_notebook():
    """Merge the per-branch records (run benchmark.ipynb once per branch) into EVIDENCE.md."""
    _reset()
    md("""
# Combine PR benchmark records → EVIDENCE.md

Upload the `results_<tag>.json` files produced by `benchmark.ipynb` (run once per branch:
set `MODEL_BRANCH`, run, download — repeat for PR1..PR4). This reads them, writes
`EVIDENCE.md` (the deep-dive, prose + tables), and shows the comparison tables. No model is
loaded — a pure merge of the saved records, so it runs anywhere (CPU is fine).
""")
    md("## 0 · Setup (tooling only)")
    code(rf"""
import os, sys, getpass, logging, subprocess
logging.basicConfig(level=logging.INFO, format="%(message)s")
os.environ["GIT_LFS_SKIP_SMUDGE"] = "1"
try:
    import google.colab  # noqa
    IN_COLAB = True
except Exception:
    IN_COLAB = False
if IN_COLAB:
    subprocess.run([sys.executable, "-m", "pip", "-q", "install", "pandas"], check=False)
    TOOL = "/content/tool"
    if not os.path.isdir(os.path.join(TOOL, "Embodied")):
        tok = getpass.getpass("GitHub token (Contents: read): ").strip()
        res = subprocess.run(
            ["git", "clone", "-b", "feature/batched-generation",
             f"https://{{tok}}@{REPO_URL}", TOOL],
            capture_output=True, text=True, env={{**os.environ}})
        if res.returncode != 0:
            raise RuntimeError((res.stderr or res.stdout).replace(tok, "***"))
else:
    TOOL = subprocess.run(["git", "rev-parse", "--show-toplevel"],
                          capture_output=True, text=True).stdout.strip()
sys.path.insert(0, os.path.join(TOOL, "Embodied"))
""")
    md("## 1 · Upload the per-branch records")
    code(r"""
paths = []
try:
    from google.colab import files
    up = files.upload()                 # select results_pr1.json, results_pr2.json, ...
    paths = list(up.keys())
except Exception:
    import glob
    paths = sorted(glob.glob("results_*.json"))
print("records:", paths)
""")
    md("## 2 · Combine → EVIDENCE.md + comparison tables")
    code(r"""
from IPython.display import Markdown, display
from repro.combine import (load_results, write_doc, detection_table, eject_table,
                           length_buckets_table, grounded_table, speed_plot,
                           compile_loop_table, compile_ap_table, stacked_eval_table)
res = load_results(paths)
write_doc(res, "EVIDENCE.md")   # full deep-dive (prose + every table); place under repro/evidence/
print("Detection: batched speedup + parity"); display(Markdown(detection_table(res)))
print("Throughput sweep (sequential vs batched)"); display(Markdown(speed_plot(res)))
print("Output-length spread + early-eject A/B"); display(Markdown(eject_table(res)))
display(Markdown(length_buckets_table(res)))
print("One image, many queries"); display(Markdown(grounded_table(res)))
print("Real decode-loop torch.compile A/B"); display(Markdown(compile_loop_table(res)))
print("Compile mAP gate"); display(Markdown(compile_ap_table(res)))
print("End-to-end stacked speedup (vs pristine)"); display(Markdown(stacked_eval_table(res)))
try:
    from google.colab import files; files.download("EVIDENCE.md")
except Exception:
    print("wrote EVIDENCE.md")
""")
    _write(os.path.join(HERE, "notebooks", "combine_results.ipynb"))


if __name__ == "__main__":
    benchmark_notebook()
    combine_notebook()
