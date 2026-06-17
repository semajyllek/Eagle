# `repro/evidence/` — deep-dive evidence (dev-branch only)

This directory holds the **evidence artifacts** for the batched + compiled
LocateAnything work. It lives under `repro/`, which is **not part of any upstream
PR** (`pr/…`, `pr2/…`, `pr3/…`, `pr4/…`) — so everything here is tracked in the
personal dev branch (`feature/batched-generation`) and never ships in a PR.

Contents:

- **`code_changes.md`** — the *source* narrative for `EVIDENCE.md`'s "Code changes"
  section (motivation + exact diffs + how-tested + speed, per change). Ported out of
  the per-PR benchmark notebooks so the deep dive is self-contained. Edit this;
  `repro.evidence_narrative.code_changes()` injects it into the generated doc.
- **`EVIDENCE.md`** — the generated deep-dive: overview → code changes → correctness
  methodology → measured performance. Built by
  `python -m repro.combine results_pr1.json … results_pr4.json -o repro/evidence/EVIDENCE.md`
  (the static prose comes from `evidence_narrative.py` + `code_changes.md`; the
  tables come from the per-PR `results_*.json`).
- **`results_pr*.json`** — each PR's own benchmark record (downloaded from the Colab
  run of `benchmark_pr*.ipynb`). Kept here for provenance; the combine step reads
  them to fill the performance tables.

The per-PR notebooks (`repro/notebooks/benchmark_pr*.ipynb`) are now lean — setup +
run the benchmark — with the code-change narrative and testing methodology moved
here into `EVIDENCE.md`.
