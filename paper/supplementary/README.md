# Supplementary material and artifact index

This directory is intentionally thin: the primary artifacts are the saved CSV files, the
generated tables, the figures, the trace files, the audit report, and the manuscript PDF in
`paper/`. Nothing here is a hand-entered result.

## Evidence boundary

- All experiments are synthetic, offline, 1 Hz simulations.
- No real radiation data were downloaded, parsed, or used.
- The recurring background regime was held out from tuning.
- The oracle gate uses ground-truth labels and is not deployable.
- No statistical-significance claim is made.
- The headline result is negative; see `../docs/results_summary.md`.

## Artifacts

| Path | Contents |
|---|---|
| `results/raw/manifest.json` | Environment, package versions, config hashes, seed lists, run counts, selected candidates |
| `results/raw/calibration.csv` | Per-method, per-candidate alarm/gate thresholds |
| `results/raw/tuning_runs.csv` | Development tuning runs |
| `results/raw/tuning_selection.csv` | All candidate summaries, selected configuration, fallback status |
| `results/raw/test_runs.csv` | Held-out matrix, one row per method/stream |
| `results/raw/trace_index.csv` | Representative trace index and thresholds |
| `results/raw/traces/*.npz` | Per-sample score, alarm, gate, and tracked mean traces |
| `results/raw/README.md` | Layout note and run provenance |
| `results/tables/primary_results.csv` | Primary and secondary estimands |
| `results/tables/discrimination_results.csv` | Threshold-free AUC, event detection, gate behaviour |
| `results/tables/by_regime.csv` | F1 by regime |
| `results/tables/by_fault_condition.csv` | F1 by fault condition |
| `results/tables/paired_differences.csv` | Paired Rad-DAES-minus-baseline differences |
| `results/tables/gate_diagnostics.csv` | Gate rejection and release rates |
| `results/tables/table_*.tex` | Generated LaTeX tables |
| `results/figures/fig_*.png` | Generated figures |
| `results/audit_report.md` | Reproducibility audit (22 checks) |
| `paper/main.pdf` | Compiled manuscript |

## Regeneration

From the repository root:

```powershell
python -m pip install -e .
python -m pytest -q
python scripts/run_benchmarks.py --mode full --output results/raw
python scripts/aggregate_results.py
python scripts/make_figures.py
python scripts/audit_results.py
```

The benchmark is deterministic given the seeds and configuration files. A smoke run is
available for a quick end-to-end check only; it uses 1,800 s streams with no source events, so
its all-zero F1 is not a scientific result:

```powershell
python scripts/run_benchmarks.py --mode smoke --workers 4 --output results_smoke
python scripts/aggregate_results.py --raw results_smoke --tables results_smoke/tables
python scripts/make_figures.py --raw results_smoke --tables results_smoke/tables --figures results_smoke/figures
```

## Original DAES provenance

The original scalar DAES reference implementation was read from an external snapshot for
interface documentation. It is not vendored or modified here. The file and archive SHA-256
values are recorded in `docs/daes_interface.md`. The Gemini-style matrix update is not
implemented as DAES; see `docs/decisions.md`.
