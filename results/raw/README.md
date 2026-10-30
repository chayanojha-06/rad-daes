# Raw benchmark artifacts

These files are the unmodified outputs of a single run of

```text
python scripts/run_benchmarks.py --mode full --workers 10 --output results
```

with `configs/experiments.yaml` at the hashes recorded in `manifest.json`.

The run was launched with `--output results`, so the runner wrote its CSVs and the
`traces/` directory directly under `results/`. They were then moved into `results/raw/`
to match the layout used by `scripts/aggregate_results.py`, `scripts/make_figures.py`, and
`scripts/audit_results.py`. The only edit to a saved artifact was the `trace_path` column
of `trace_index.csv`, which was rewritten to point at the new `results/raw/traces/`
location; no measurement, seed, threshold, or label was changed.

| File | Contents |
|---|---|
| `candidates.csv` | 69 tuning candidates (one row per method/configuration) |
| `calibration.csv` | alarm and gate thresholds for every candidate |
| `tuning_runs.csv` | 1,035 development runs (3 regimes x mixed_10 x 5 seeds x 69 candidates) |
| `tuning_selection.csv` | all candidate summaries plus the selected configuration per method |
| `test_runs.csv` | 1,440 held-out runs (4 regimes x 8 fault conditions x 5 seeds x 9 methods) |
| `trace_index.csv` | index of the four representative traces |
| `traces/` | compressed per-step traces (mean, score, alarm, gate flags) |
| `manifest.json` | environment, config hashes, seed lists, run counts, elapsed time |
| `run_index.csv` | phase-level row counts |

`test_runs.csv` is the primary artifact. Everything in `results/tables/` and
`results/figures/` is derived from it by the aggregation and figure scripts.
