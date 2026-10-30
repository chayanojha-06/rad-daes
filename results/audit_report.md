# Reproducibility audit

Generated: 2026-10-28T13:49:02.651388+00:00

Checks: 22 total, 0 failed.

| Check | Result | Detail |
|---|---|---|
| `test_methods_match_config` | PASS | methods=['adam', 'daes', 'ewma', 'gated_daes_fixed', 'gated_ewma', 'oracle_gate', 'rad_daes', 'rls', 'static'] |
| `test_regimes_match_config` | PASS | regimes=['abrupt', 'gradual', 'recurring', 'stable'] |
| `test_faults_match_config` | PASS | faults=['clean', 'dropout_10', 'jitter_10', 'mixed_10', 'mixed_15', 'mixed_5', 'spike_10', 'stuck_10'] |
| `test_seeds_match_config` | PASS | seeds=[11, 12, 13, 14, 15] |
| `test_rows_unique` | PASS | 0 duplicate keys |
| `test_scenario_cells_complete` | PASS | 288 cells, expected 288; min=5, max=5 |
| `test_key_metrics_finite` | PASS | f1=ok, far_per_hour=ok, bg_rel_error=ok |
| `one_selected_candidate_per_method` | PASS | 9 selected rows for 9 methods |
| `test_uses_frozen_selection_and_thresholds` | PASS | consistent |
| `manifest_present` | PASS | elapsed=2011.131097799982s |
| `manifest_run_counts_match_protocol` | PASS | observed={'candidates': 69, 'calibration': 69, 'development': 1035, 'test': 1440}, expected={'candidates': 69, 'calibration': 69, 'development': 1035, 'test': 1440} |
| `manifest_test_rows_match_csv` | PASS | manifest=1440, csv=1440 |
| `manifest_declares_synthetic_only` | PASS | synthetic_only=True, real_data_used=False |
| `config_hashes_match_run` | PASS | all config hashes match |
| `saved_rows_reproduce_from_seeds` | PASS | 2 runs recomputed |
| `trace_files_present` | PASS | 4 traces, 0 missing |
| `required_tables_present` | PASS | all present |
| `required_figures_present` | PASS | all present |
| `manuscript_present` | PASS | 354 lines |
| `manuscript_inputs_exist` | PASS | 5 inputs |
| `manuscript_figures_exist` | PASS | 7 figures |
| `manuscript_macros_defined` | PASS | 12 generated macros used |
