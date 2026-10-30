# Rad-DAES: a guarded, drift-aware background tracker for radiation-portal count streams

**Status: first-draft research prototype (28 October 2026).** This repository contains a
small, reproducible simulation study. It does **not** contain real radiation measurements,
field validation, or a novelty claim. The headline result is **negative**: the candidate
pipeline does not improve the detection/false-alarm trade-off on these streams. The value of
the package is the auditable benchmark and the documented failure modes.

Rad-DAES is a reliability-aware background-estimation pipeline for six-window gamma count
streams. It separates three decisions that are often conflated:

1. **Is the telemetry trustworthy?** A four-condition telemetry-integrity gate decides
   whether an observation may update the background model.
2. **Is the observation suspicious?** A pre-update Mahalanobis score is compared with a
   calibrated alarm threshold. Detection is performed *before* any model update.
3. **How large should an eligible update be?** The original DAES scalar step-size heuristic
   (drift-aware eigenvalue scaling) sets the step from an online estimate of the largest
   curvature eigenvalue.

## Headline result

The held-out matrix is 4 background regimes x 8 telemetry-fault conditions x 5 seeds x
9 methods = **1,440 runs**. The recurring regime is held out from tuning.

| Method | Mean F1 | False alarms/h | Event detection |
|---|---|---|---|
| Oracle gate (non-deployable) | 0.242 | 7.3 | 0.736 |
| Static | 0.148 | 472.4 | 0.821 |
| Gated EWMA | 0.095 | 102.7 | 0.406 |
| **Rad-DAES** | **0.023** | **18.5** | **0.105** |
| EWMA / RLS / Adam / DAES | <= 0.003 | 2.0-2.5 | <= 0.060 |

No deployable method is both sensitive and quiet. The four methods whose development
configurations met the false-alarm constraint are nearly blind; the methods with visible
detection violate the constraint. The saved traces show two mechanisms: the innovation guard
freezes the model during drift (the 90-sample release counter peaks at 89 in the
representative trace), and the adaptive background absorbs source energy during the ramp.
See `docs/results_summary.md` and `paper/main.pdf` for the full account.

## What is and is not implemented

| Component | State |
|---|---|
| Six-window Poisson stream generator with labelled background regimes, source events, and telemetry faults | Implemented and tested |
| Telemetry-integrity gate (validity, timing, Fano, innovation) | Implemented and tested |
| Background model (mean + covariance, Gaussian quasi-likelihood) | Implemented and tested |
| Original DAES step-size tracker (power iteration, Rayleigh quotient, drift-responsive EMA, upward correction) | Implemented and tested; interface documented in `docs/daes_interface.md` |
| Baselines: static, EWMA, gated EWMA, RLS, Adam, ungated DAES, fixed-step gated DAES, oracle gate | Implemented and run |
| Detection, calibration, metrics, seeds, raw CSV outputs, figures | Implemented and run |
| Reproducibility audit (`scripts/audit_results.py`) | Implemented; 22 checks pass on the saved artifacts |
| Real RADAI / PyRIID / JRC data ingestion | **Not implemented**; no data files were available and formats were not verified |
| Covariance-frozen ablation and matched-FAR re-analysis | **Not run**; listed as future work |
| Field validation, hardware latency, operational safety | **Not attempted** |

See `docs/dataset_audit.md` for the data blocker. The experiments use **clearly labelled
synthetic streams only**. Synthetic fault injection is not real-world validation.

## Quick start

```powershell
# from this directory
python -m pip install -e .
python -m pytest -q

# regenerate the development-tuning and held-out benchmark (writes results/raw/)
python scripts/run_benchmarks.py --mode full --output results/raw

# rebuild tables and figures from saved CSV files
python scripts/aggregate_results.py
python scripts/make_figures.py

# verify matrix completeness, frozen selection, determinism, and manuscript inputs
python scripts/audit_results.py
```

The benchmark is deterministic given the seeds in `configs/experiments.yaml`. The runner
writes raw per-run metrics, per-step traces for representative runs, a manifest with
environment versions, and a `results/raw/run_index.csv`.

A smoke run is available for an end-to-end check only. It uses 1,800 s streams that contain
no source events, so its all-zero F1 values are **not** a scientific result:

```powershell
python scripts/run_benchmarks.py --mode smoke --workers 4 --output results_smoke
python scripts/aggregate_results.py --raw results_smoke --tables results_smoke/tables
python scripts/make_figures.py --raw results_smoke --tables results_smoke/tables --figures results_smoke/figures
```

## Repository layout

```text
rad-daes/
  README.md
  pyproject.toml
  requirements.txt
  configs/                 data.yaml, model.yaml, experiments.yaml
  docs/                    research_gap.md, mathematical_specification.md,
                           daes_interface.md, dataset_audit.md, protocol.md,
                           decisions.md, results_summary.md, submission_checklist.md
  src/rad_daes/
    data/                  synthetic six-window stream generator
    preprocessing/         window definitions and validation
    telemetry/             integrity gate
    background/            mean/covariance tracker
    adaptation/            original DAES step-size controller and adapters
    detection/             pre-update Mahalanobis detector
    evaluation/            metrics, calibration, benchmark runner
  tests/                   unit and integration tests
  scripts/                 inspect_data.py, generate_streams.py,
                           run_benchmarks.py, aggregate_results.py,
                           make_figures.py, audit_results.py
  results/
    raw/                   raw CSVs, traces, manifest, layout note
    tables/                generated tables and manuscript macros
    figures/               generated figures
    audit_report.md        reproducibility audit
  paper/                   main.tex, main.pdf, references.bib, supplementary/
```

## Separation from other projects

This repository is self-contained. It does not import, modify, or write to other projects
that share the workspace, including the lunar-image correspondence project and the unrelated
standalone script at the workspace root. No Mark-LVI/JARVIS material is used.

The original DAES implementation was read from an external snapshot for interface
documentation only. It is not vendored here and is not modified. See
`docs/daes_interface.md` for the exact source path and the equations that were preserved.

## Evidence boundary

- All reported results are from a synthetic, offline, 1 Hz stream simulation.
- No real dataset was downloaded, parsed, or used.
- No claim of statistical significance is made from a handful of seeds. Five seeds per
  scenario are used; intervals are descriptive Student-t intervals and are wide.
- The original DAES heuristic was proposed for deterministic strongly convex quadratics.
  Applying its scalar step-size rule to a stochastic background-tracking loss is an
  **adaptation**, not a theorem. This is stated in the mathematical specification.
- The Gemini-generated planning PDFs contain a multi-direction `V Phi V^T` update and a
  Fisher/curvature formulation that are inconsistent with the original DAES code. Those
  proposals are **not** implemented as DAES; see `docs/decisions.md`.
- The covariance-contamination mechanism is inferred from the static-versus-adaptive
  contrast and saved traces; it was not isolated by a covariance-frozen ablation.

## Submission package

- `paper/main.pdf` --- compiled manuscript (15 pages).
- `paper/cover_letter.pdf` and `.docx` --- venue-neutral draft cover letter.
- `submission/Rad-DAES_submission_2026-10-28.zip` --- manuscript, cover letter,
  supplementary note, and a full repository mirror with SHA-256 checksums; rebuilt by
  `python scripts/build_submission.py`.

Author details and the target venue are placeholders; nothing has been submitted externally.

## License and data

No license is asserted for third-party material. The synthetic generator is original code
for this study. RADAI and PyRIID are referenced only; no data or code from them is
redistributed here.
