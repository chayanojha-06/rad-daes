# Experiment protocol (prespecified before held-out runs)

**Frozen:** 28 October 2026, before the held-out test matrix was executed.
**Deviations:** any change after this date must be recorded in §9 with a reason and the
affected results.

## 1. Hypotheses

- **H1 (integrity gate).** Under telemetry corruption, the gated methods have lower
  background error and fewer corruption-induced false alarms than their ungated counterparts,
  without a large loss in source-event recall.
- **H2 (DAES contribution).** Under background drift, gated DAES has a better
  detection/false-alarm trade-off than gated EWMA at matched false-alarm burden.
- **H3 (drift-responsive control).** Gated DAES with the drift-responsive tracker has lower
  recovery time or background error than gated DAES with a step fixed at initialization.
- **H4 (source suppression risk).** The innovation guard blocks a non-trivial fraction of
  genuine source samples; the gate's recall cost is measurable and is reported.

These are directional hypotheses. The analysis reports the observed direction and
uncertainty, including cases where the direction reverses.

## 2. Streams and scenarios

- Duration: 21,600 s (6 h) at 1 Hz; 600 s warm-up excluded from metrics.
- Six energy windows, nominal means from `configs/data.yaml`.
- Four background regimes: `stable`, `gradual`, `abrupt`, `recurring`.
- Six source events per stream, 20–60 s, amplitudes 3–8 baseline σ, three templates.
- Fault conditions: `clean`, `mixed_5`, `mixed_10`, `mixed_15`, `dropout_10`, `jitter_10`,
  `stuck_10`, `spike_10`.
- Test matrix: 4 regimes × 8 fault conditions × 5 seeds = 160 streams. Each stream is
  processed by 9 methods, giving 1,440 runs.
- The `recurring` regime is **not** used for tuning. It is an unseen regime at test time.

## 3. Methods

| Method | Gate | Update | Purpose |
|---|---|---|---|
| `static` | no | none; initial mean/covariance frozen | value of adaptation |
| `ewma` | no | fixed α EWMA mean; covariance EWMA | simple background tracker |
| `gated_ewma` | yes | fixed α EWMA mean; covariance EWMA | isolate integrity-aware updating |
| `rls` | no | recursive least squares with forgetting | established adaptive estimator |
| `adam` | no | Adam on the Gaussian quasi-likelihood | optimizer baseline |
| `daes` | no | original DAES scalar step | isolate the gate's contribution |
| `gated_daes_fixed` | yes | DAES step frozen at initialization | isolate drift-responsive control |
| `rad_daes` | yes | original DAES scalar step | full pipeline |
| `oracle_gate` | oracle | original DAES scalar step | non-deployable reference using true labels |

The oracle gate passes an update only when the sample is neither a source nor a fault. It is
an upper bound on what a perfect integrity decision could do; it is **not** deployable.

All methods use the same detector, the same six windows, the same covariance update rule
(where applicable), and the same threshold-calibration procedure. The optimizer and the
detector are not treated as interchangeable competitors.

The gated methods use a sustained-exceedance release: after 90 consecutive samples that pass
validity, timing, and Poisson dispersion but exceed the innovation threshold, the innovation
condition is released until the score falls below threshold or another telemetry condition
fails. The patience is longer than the longest synthetic source event (60 s). This prevents
a persistent background step from deadlocking the gate. It is a design choice, not a
planning-document requirement, and is reported as a limitation.

## 4. Hyperparameter tuning

Tuning uses only:

- development seeds `1001–1005`;
- regimes `stable`, `gradual`, `abrupt` (not `recurring`);
- fault condition `mixed_10`;
- 6 h streams.

Grids are listed in `configs/experiments.yaml`. For each method, the selected configuration
maximises mean development F1 across the three regimes, subject to mean background relative
error ≤ 0.10 and mean false alarms per hour ≤ 2.0. Ties are broken by lower background error,
then by the smaller learning-rate/gain.

The selected values are frozen in `results/tables/tuning_selection.csv` before the test
matrix is run. If a method has no feasible configuration, the least-violating configuration
is used and the violation is reported.

## 5. Detector calibration

For each method, the alarm threshold is the empirical `1 − 1/3600` quantile of pre-update
scores on a clean, stable development stream, after warm-up. The five development seeds are
pooled. The threshold is frozen before test. The alarm rule requires two consecutive
exceedances. The realised false-alarm rate is reported; it may be below the 1/h target.

The gate innovation threshold is the `0.999` quantile of the same clean stable scores, with a
floor of 10.0. It is also frozen before test.

## 6. Metrics and estimands

Primary estimand: **mean sample-level F1 across the 32 test scenarios**, each scenario
weighted equally. Secondary estimands:

- false alarms per hour over non-source time;
- background relative error on non-source, non-fault samples;
- recovery time after abrupt steps;
- detection latency for detected events;
- update stability;
- runtime per sample;
- gate rejection rate and source-sample rejection rate (H4 diagnostic).

Each scenario is run with five seeds; the five seed-level values are averaged within each
scenario first. The reported interval is a two-sided 95% Student-t interval across the 32
scenario means (df = 31), with the scenario as the observational unit. No multiplicity
correction is applied in this draft; scenario-level results are descriptive. A claim of
statistical significance is **not** made.

## 7. Prespecified success gates

Adapted from the research plan:

- **Gate 1 — Feasibility:** stream generation, labels, seeds, and the original DAES interface
  are verified. Real data access is **not** verified; this is a partial failure and is
  reported.
- **Gate 2 — Correctness:** unit tests, numerical checks, matched-stream reproducibility, and
  edge cases pass. If not, fix the implementation before interpreting results.
- **Gate 3 — Scientific value:** the full system improves the prespecified primary estimand
  over strong baselines with uncertainty intervals that do not span a practically
  meaningless difference. If it does not, report a negative or mixed result.
- **Gate 4 — Novelty/publication:** a systematic literature audit establishes a distinct
  contribution. This draft does **not** pass Gate 4; it makes no novelty claim.

## 8. Analysis and reporting rules

- Report all methods and all scenarios, including failures and losses.
- Do not select only scenarios where Rad-DAES wins.
- Report the oracle as non-deployable and never as a competitor.
- Report synthetic-only status in the abstract, results, and limitations.
- Do not describe fault injection as real-world validation.
- Do not claim field readiness, operational safety, or regulatory suitability.
- Save every raw run record, seed, configuration, and environment version.
- Figures and tables are generated from saved CSV/JSON, not hand-entered.

## 9. Pre-test design revisions and deviations

The protocol was revised on 28 October 2026 **before the held-out test matrix was run**.
No held-out test result was inspected before these changes.

| Date | Revision | Reason | Affected results |
|---|---|---|---|
| 2026-10-28 | Source-event placement fixed to avoid a double-gap cursor bug; six events are now attempted and realised on 6 h streams. | The original placement produced only four events on some streams, contradicting the stated design. | All synthetic runs; development and test. |
| 2026-10-28 | Added a 90 s sustained-exceedance release to the gate. | Without it, a persistent background step keeps the innovation score above threshold and the model never adapts; the gate deadlocks and produces a long false-alarm burst. | All gated methods; H1, H2, H4. |
| 2026-10-28 | Covariance diagonal floor set to `max(1e-6, 0.25 * nominal)` per window. | A count covariance below a fraction of the Poisson variance is not physically meaningful and can make scores unstable. | All adaptive methods. |
| 2026-10-28 | Source profile sampled at interval midpoints. | The previous profile was exactly zero at the first labelled sample, so a labelled sample could contain no source. | All synthetic runs. |

After the test matrix starts, any further change is recorded as a post-test deviation with
the affected results.

## 10. Deviations log after freezing

| Date | Deviation | Reason | Affected results |
|---|---|---|---|
| 2026-10-28 | Added threshold-free ROC/PR AUC and event-level detection as secondary summaries after the frozen test matrix completed. | The primary F1 estimand depends on one calibrated threshold, and the frozen results showed that the threshold dominated the ranking. Both summaries are computed from columns (`roc_auc`, `pr_auc`, `n_detected`, `n_events`) that the frozen run already saved. | Secondary analysis only; the primary estimand, the selection rule, and the test matrix were not changed. |
| 2026-10-28 | Raw artifacts were moved from `results/` to `results/raw/`, and the `trace_path` column of `trace_index.csv` was rewritten to the new location. | The run was launched with `--output results`, but the aggregation and audit scripts expect `results/raw/`. No measurement, seed, threshold, label, or candidate was changed. | File paths only; documented in `results/raw/README.md`. |

No method, hyperparameter, threshold, or seed was changed after the test matrix was inspected.
The negative result is reported as observed; no rerun was performed to improve it.

## 11. Stopping rule

The draft stops at a small, complete, reproducible prototype: one synthetic benchmark,
nine methods, five seeds, unit tests, saved outputs, and a manuscript. It does not expand to
real-data ingestion or a full hyperparameter search within the deadline. Any unfinished item
is listed in `docs/decisions.md` and the manuscript's limitations.
