# Results summary (frozen run of 28 October 2026)

This document records the outcome of the frozen held-out matrix. It is a narrative index to
`results/raw/test_runs.csv`, `results/tables/`, and `results/figures/`; the numbers below are
copied from those generated files, not recomputed by hand.

## Run inventory

| Item | Value |
|---|---|
| Command | `python scripts/run_benchmarks.py --mode full --workers 10 --output results` |
| Elapsed | 2,011 s |
| Tuning candidates | 69 |
| Calibration runs | 69 (one per candidate; 5 seeds pooled per candidate) |
| Development runs | 1,035 (69 candidates x 3 regimes x 5 seeds) |
| Held-out runs | 1,440 (9 methods x 4 regimes x 8 fault conditions x 5 seeds) |
| Traces | 4 representative runs |
| Environment | Python 3.14, numpy 2.5.4, scipy 1.18.1, pandas 3.1.0, matplotlib 3.11.2, scikit-learn 1.9.1 |
| Config hashes | recorded in `results/raw/manifest.json`; verified by `scripts/audit_results.py` |

The recurring regime was never used for tuning. Alarm and gate thresholds were calibrated on
clean stable development streams and frozen before the test matrix.

## Primary results

F1 is the mean of 32 scenario means (each over 5 seeds); the interval is a two-sided 95%
Student-t interval across scenario means. FAR is false alarms per hour over non-source time.
Event detection is the fraction of the 960 source events with at least one alarm.

| Method | F1 [95% CI] | FAR/h | BG rel. | Event det. | ROC AUC | PR AUC |
|---|---|---|---|---|---|---|
| Static | 0.148 [0.109, 0.188] | 472.40 | 0.0546 | 0.821 | 0.728 | 0.128 |
| EWMA | 0.003 [0.002, 0.004] | 2.48 | 0.0153 | 0.060 | 0.622 | 0.031 |
| Gated EWMA | 0.095 [0.082, 0.108] | 102.75 | 0.0174 | 0.406 | 0.691 | 0.089 |
| RLS | 0.003 [0.002, 0.004] | 2.48 | 0.0153 | 0.060 | 0.622 | 0.031 |
| Adam | 0.003 [0.002, 0.004] | 2.50 | 0.0278 | 0.056 | 0.637 | 0.033 |
| DAES | 0.002 [0.002, 0.003] | 2.03 | 0.0375 | 0.054 | 0.600 | 0.024 |
| Gated DAES (fixed) | 0.006 [0.004, 0.008] | 6.65 | 0.0689 | 0.059 | 0.558 | 0.019 |
| **Rad-DAES** | **0.023 [0.018, 0.028]** | **18.51** | **0.0254** | **0.105** | **0.629** | **0.036** |
| Oracle gate (non-deployable) | 0.242 [0.230, 0.254] | 7.29 | 0.0143 | 0.736 | 0.773 | 0.223 |

The oracle gate uses the true source and fault labels and is not a deployable method. It is an
upper bound on this pipeline family, not a competitor.

## Main findings

1. **No deployable method is both sensitive and quiet.** The four methods with feasible
   development configurations (EWMA, RLS, Adam, ungated DAES) have test F1 at most 0.003 and
   detect at most 6% of source events, at 2.0--2.5 false alarms/h. Gated EWMA reaches F1
   0.095 but at 102.7/h; Rad-DAES reaches 0.023 at 18.5/h; static reaches 0.148 at 472.4/h.
2. **The ranking is not a threshold artefact.** Threshold-free ROC AUC orders the methods in
   the same way: oracle 0.773, static 0.728, gated EWMA 0.691, Rad-DAES 0.629, ungated DAES
   0.600. PR AUC shows the same ordering.
3. **The gate deadlocks under drift.** Rad-DAES rejects 26.1% of clean samples and 33.8% of
   source samples. The sustained release is active on about 3e-6 of samples. In the
   representative abrupt-drift trace the consecutive-exceedance counter reaches 89 and resets
   one sample before the 90-sample patience, so the mean stays frozen for roughly 350 s while
   the true background moves by about 170 counts/s.
4. **The adaptive background absorbs source energy.** The mean and covariance both update on a
   50-sample time constant, comparable to the source duration (20--60 s). During the source
   ramp the score is still below the gate threshold, so the gate passes those samples; the mean
   moves toward the source and the covariance learns its outer product, and the score for the
   peak falls. This is consistent with static (updates neither, ROC AUC 0.728) outperforming
   every adaptive method on separability, but the mean and covariance channels were not
   separated by an ablation.
5. **The selection rule was infeasible.** Four of nine methods used the fallback rule. Rad-DAES
   selected the largest gain in the grid (0.5) with a development FAR of 28.6/h; gated EWMA
   selected the smallest EWMA weight but still had 99.4/h; the oracle selected 0.05 and had
   12.5/h. The oracle's advantage combines perfect labelling with a ten-fold gentler step, so
   the gate effect cannot be separated from the gain effect in this matrix.
6. **Thresholds do not transfer across regimes.** Thresholds calibrated on clean stable
   streams at a 1/h target yield 7.3/h for the oracle on test, and 7.0--7.3/h for Rad-DAES on
   stable, gradual, and recurring regimes. Faults and drift, not the source label, dominate the
   false-alarm count.

## Regime and fault structure

FAR by regime (mean over faults and seeds):

| Method | stable | gradual | abrupt | recurring |
|---|---|---|---|---|
| Static | 7.26 | 498.64 | 1323.27 | 60.42 |
| Gated EWMA | 7.04 | 7.25 | 389.65 | 7.06 |
| Rad-DAES | 7.01 | 7.01 | 52.89 | 7.12 |
| Oracle gate | 7.24 | 7.25 | 7.44 | 7.24 |
| EWMA / RLS | 2.43 | 2.43 | 2.62 | 2.42 |
| DAES | 1.99 | 1.98 | 2.16 | 1.99 |

The abrupt regime is the stress case for every method that updates its mean: freezing during
the step produces a long alarm burst. The oracle gate is flat across regimes because it never
updates on a source or fault, but it still pays 7.2--7.4/h from drift alone.

The `spike_10` condition is the worst for false alarms: Rad-DAES reaches 68.8/h and gated EWMA
285.3/h, because a spike both alarms and freezes the update. Clean, dropout, jitter, and stuck
conditions are much quieter for the ungated methods.

## Recovery

Recovery is the first sample at which every window's relative mean error stays below 5% for 30
consecutive samples; an event that never meets the condition is not counted as recovered.

| Method | Mean recovery (s) | Events recovered |
|---|---|---|
| EWMA / RLS | 399 | 80 / 80 |
| Gated EWMA | 1,734 | 80 / 80 |
| Rad-DAES | 4,198 | 55 / 80 |
| Oracle gate | 4,213 | 80 / 80 |
| DAES | 4,564 | 69 / 80 |
| Gated DAES (fixed) | 5,196 | 23 / 80 |
| Adam | 5,286 | 80 / 80 |
| Static | --- | 0 / 80 |

Fast recovery and source sensitivity pull in opposite directions in this matrix. EWMA recovers
quickly because it tracks aggressively, which is also why it absorbs sources.

## What was not done

- No real radiation data: RADAI requires an account and a large download; PyRIID requires a
  compatible Python environment and spectra conversion; JRC reference spectra are not a
  continuous portal stream.
- No covariance-frozen ablation, so the contamination mechanism is inferred, not isolated.
- No matched-FAR re-analysis or threshold sweep; ROC/PR AUC is reported as a threshold-free
  secondary summary, but the selection rule was not based on it.
- No sensitivity analysis over the 90 s patience, the covariance time constant, the Fano
  bounds, the covariance floor, or the source amplitude.
- No field, hardware, or regulatory validation.

## What would change the conclusion

A follow-up would need at least: (i) a drift detector separate from the source detector;
(ii) a two-timescale or robust covariance that is frozen during suspected sources; (iii) a
matched-FAR evaluation so methods are compared at a common, achievable operating point; and
(iv) a real or realistically calibrated stream. Until then, the negative result stands for
this synthetic matrix.
