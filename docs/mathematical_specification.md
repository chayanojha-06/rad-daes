# Mathematical specification (frozen 28 October 2026)

This document is the implementation contract for the first-draft prototype. It separates
**the original DAES method** from **the Rad-DAES wrapper** and from **design choices that are
not established by the source documents**. Where the supplied planning PDFs disagree, the
original code is treated as authoritative for DAES; the disagreements are recorded in
`docs/decisions.md`.

## 1. Symbols

| Symbol | Meaning |
|---|---|
| `t` | discrete 1 s sample index |
| `x_t ∈ R^D`, `D = 6` | observed count vector in the six energy windows |
| `μ_t ∈ R^D` | background mean after processing sample `t` |
| `Σ_t ∈ R^{D×D}` | background covariance after processing sample `t` |
| `r_t = x_t − μ_{t−1}` | pre-update innovation |
| `s_t` | pre-update Mahalanobis score |
| `q_t ∈ {0,1}` | telemetry-integrity gate output |
| `H_t` | curvature matrix used by DAES; here `H_t = Σ_{t−1}^{−1}` |
| `λ̂_t` | DAES estimate of `λ_max(H_t)` |
| `η_t` | DAES scalar step size |
| `a_t ∈ {0,1}` | alarm output |
| `y_t ∈ {0,1}` | ground-truth source-event label (synthetic only) |

The six windows are a **design hypothesis**, not a universal detector configuration. Their
boundaries are listed in `configs/data.yaml`; they follow the planning document but were not
validated against a calibrated spectrum in this study.

## 2. Observation model

The synthetic generator produces, for each window `j`,

```
x_t,j ~ Poisson( μ_true,t,j + source_t,j )
```

where `μ_true,t,j` is the labelled background mean (stable, gradual, abrupt, or recurring)
and `source_t,j ≥ 0` is a labelled source excess. Poisson noise is applied after the
background and source terms are combined. Telemetry faults are then applied to `x_t` (and,
for jitter, to the timestamp). The generator returns `μ_true,t`, `y_t`, the fault label, and
the source event id. No real data are used.

## 3. Pre-update detection

Detection uses only information available before the update:

```
r_t = x_t − μ_{t−1}
s_t = r_t^T ( Σ_{t−1} + ε I )^{−1} r_t,     ε = 1e−6
a_t = 1[ s_t > τ_alarm ]  AND  a_{t−1} = 1     (persistence L = 2)
```

The persistence rule is the planning document's `L ≥ 2` condition. `τ_alarm` is calibrated
per method on a clean, stable development stream at a target false-alarm rate of 1 per hour
(see §7). A sample with a non-finite or non-positive observation cannot be scored; it is
treated as no alarm and no update.

This is a **starting detector**, not a validated one. The score distribution is not exactly
chi-square because `Σ` is estimated and the counts are Poisson; the empirical false-alarm
rate is reported rather than assumed.

## 4. Background model and loss

Rad-DAES uses a Gaussian quasi-likelihood for the mean with the current covariance treated
as fixed during the step:

```
L_t(θ) = 0.5 (x_t − θ)^T Σ_{t−1}^{−1} (x_t − θ)
∇L_t(θ) = − Σ_{t−1}^{−1} (x_t − θ)
∇²L_t(θ) = Σ_{t−1}^{−1} = H_t
```

`H_t` is therefore a **Hessian** (equivalently, the Fisher information for a Gaussian mean
with known covariance), not the covariance of the counts. The count covariance is `Σ_t` and
is tracked separately. This distinction is required by the planning document and is enforced
in the code.

The Gaussian quasi-likelihood is an approximation to the Poisson data. It is used because
the detector is already a covariance-weighted Mahalanobis rule and because it gives a
well-defined curvature matrix. The approximation is a limitation, not a claim that the
counts are Gaussian.

## 5. Original DAES step-size rule (preserved)

The original DAES implementation is a scalar step-size heuristic for gradient descent on a
strongly convex quadratic with a drifting Hessian. Its update is

```
x_{t+1} = x_t − η_t ∇f_t(x_t)
η_t = γ / ( (1 + s) λ̂_t )
```

where `λ̂_t` is an online estimate of `λ_max(A_t)`, `γ` is a gain, and `s` is a safety
margin. The estimator uses `k` power iterations on `A_t`, a Rayleigh quotient `r_t`, a
relative drift score `d_t`, a drift-responsive smoothing coefficient `α_t`, and an upward
correction:

```
v ← A_t v / ||A_t v||                     (k times)
r_t = v^T A_t v
d_t = |r_t − r_{t−1}| / max(|r_{t−1}|, 1e−12)
α_t = clip( α_0 + κ d_t, α_0, α_max )
λ̂_t = (1 − α_t) λ̂_{t−1} + α_t r_t
if r_t > λ̂_{t−1}:  λ̂_t = max(λ̂_t, r_t)
```

The exact source code is `drift_aware_eigen_scaling.py` in the external
`drift-aware-eigen-scaling-lab-main` snapshot (see `docs/daes_interface.md`). The defaults
are `γ = 1.78`, `s = 0.10`, `k = 7`, `α_0 = 0.16`, `α_max = 0.82`, `κ = 7.0`. In Rad-DAES,
`γ` is tuned on development data; the **equation form is unchanged**.

The fixed-objective stability condition `0 < η < 2/λ_max(A)` is a pointwise condition. It
does not become a theorem for the stochastic, time-varying Rad-DAES loss. Safety is measured
empirically, not assumed.

## 6. Rad-DAES update (the wrapper)

When `q_t = 1`, Rad-DAES applies the original DAES scalar step to the Gaussian
quasi-likelihood:

```
η_t = γ / ( (1 + s) λ̂_t ),      λ̂_t ≈ λ_max(Σ_{t−1}^{−1})
μ_t = μ_{t−1} + η_t Σ_{t−1}^{−1} r_t
```

This is gradient descent on `L_t` with step `η_t`: since `∇L_t = −Σ^{−1}r_t`,
`μ_t = μ_{t−1} − η_t ∇L_t`. The step is scalar and shared across windows, exactly as in the
original DAES rule. It is **not** a matrix preconditioner and **not** the `V Φ V^T` update
proposed in the Gemini PDFs.

When `q_t = 0`, the model is frozen:

```
μ_t = μ_{t−1},   Σ_t = Σ_{t−1},   λ̂_t = λ̂_{t−1}
```

The covariance is updated only on eligible samples, using the post-update residual
`u_t = x_t − μ_t`:

```
Σ̃_t = (1 − β) Σ_{t−1} + β u_t u_t^T
Σ_t = (1 − ρ) Σ̃_t + ρ diag(Σ̃_t)
diag(Σ_t) ← max(diag(Σ_t), max(1e−6, 0.25 · μ_0))
```

with `β = 0.02` and `ρ = 0.10` (development defaults). The shrinkage and floor are
regularisation choices, not part of DAES. The per-window floor keeps the diagonal at or
above a fraction of the nominal Poisson variance; it is not a claim that the true
covariance equals the Poisson variance. The covariance update is the same for all adaptive
methods that use a covariance; this keeps the detector pipeline common.

## 7. Telemetry-integrity gate

The gate is a conjunction of four binary conditions evaluated on the raw observation and a
length-`W = 10` sliding buffer:

1. **Validity** `g_valid`: every channel is finite and strictly positive.
2. **Timing** `g_time`: `0.5 ≤ Δt_t ≤ 1.5` s, where `Δt_t` is the timestamp difference from
   the previous accepted arrival. The planning document gives `0 < Δt ≤ Δt_max`; the lower
   bound is added so that short-interval jitter is detectable. This is a documented design
   choice.
3. **Poisson dispersion** `g_poisson`: for every window `j`,
   `0.25 ≤ Var_W(x_j) / Mean_W(x_j) ≤ 3.5`. A stuck sensor gives Fano ≈ 0; a spike burst
   gives Fano > 3.5. A constant elevated source remains approximately Poisson and is not
   rejected by this condition alone.
4. **Innovation guard** `g_innov`: `s_t ≤ τ_gate`, **or** a sustained-exceedance release
   is active. `τ_gate` is the `0.999` quantile of `s_t` on a clean stable development stream,
   with a floor of 10.0. The release activates after `P = 90` consecutive samples that pass
   validity, timing, and Poisson dispersion but exceed `τ_gate`; it resets on any other
   telemetry failure or on a score below threshold. It exists because a persistent background
   step otherwise deadlocks the gate: the model cannot adapt, so the innovation score never
   falls. The patience is longer than the longest synthetic source event (60 s), so a single
   source event cannot trigger it. It is a design choice, not part of the original planning
   document; its sensitivity is a limitation.

`q_t = g_valid · g_time · g_poisson · (g_innov OR sustained_release)`.

**Critical warning.** An unusual count pattern is not automatically a telemetry fault. A
real source is unusual too. The innovation guard can block updates during a source event;
this is intentional for model protection, but it can also suppress legitimate adaptation.
The sustained release limits the deadlock, but a source longer than 90 s would be treated as
drift. The evaluation reports source-detection performance separately from gate decisions,
includes a gate-suppression diagnostic, and reports the release rate.

## 8. Detection threshold calibration

For each method, `τ_alarm` is the `1 − r·Δt` empirical quantile of `s_t` on a clean, stable
development stream, after the 600 s warm-up, where `r = 1` false alarm per hour and
`Δt = 1` s. The same five development seeds are used for every method. The threshold is
frozen before the held-out test runs. Because the alarm rule requires two consecutive
exceedances, the realised false-alarm rate can be lower than the target; the realised rate
is reported.

## 9. Metrics

All metrics are computed after the warm-up and only on samples with a finite score, except
where noted.

- **Detection:** precision, recall, F1, PR-AUC, ROC-AUC at the sample level.
- **False-alarm burden:** false alarms per hour over non-source time.
- **Detection latency:** seconds from source-event start to the first alarm, for detected
  events; undetected events are reported separately.
- **Recovery time:** after an abrupt background step, the first time the relative mean error
  stays below 0.05 for 30 consecutive samples. Reported per drift event.
- **Background error:** mean `|μ_t,j − μ_true,t,j| / μ_true,t,j` over non-source,
  non-fault samples.
- **Update stability:** mean squared relative update `||μ_t − μ_{t−1}||² / ||μ_ref||²`.
- **Fault resilience:** the same detection and background metrics grouped by fault type and
  rate.
- **Runtime:** wall-clock seconds per sample, excluding file I/O.

## 10. Estimands and uncertainty

The primary estimand is the **mean sample-level F1 across the 32 held-out test scenarios**
(4 regimes × 8 fault conditions), with each scenario contributing equally. Secondary
estimands are false alarms per hour, background relative error, and recovery time. Each
scenario is run with five seeds; seed-level values are averaged within each scenario, and the
reported interval is a two-sided 95% Student-t interval across the 32 scenario means (df = 31).
The scenario, not the individual seed, is the observational unit. These intervals describe
between-scenario variation in this frozen matrix; they do not cover model misspecification or
sampling variability beyond the five seeds per scenario. No multiplicity correction is applied
in this draft.
Scenario-level results are reported, including failures and cases where Rad-DAES loses.

## 11. What is deliberately not specified

- No claim that the six windows preserve source information.
- No claim that the Poisson Fano bounds are optimal or physically universal.
- No claim that the Gaussian quasi-likelihood is the correct data model.
- No claim that DAES is a new optimizer or that Rad-DAES is novel.
- No field-readiness, safety, or regulatory claim.

These are recorded as limitations and future work, not silently assumed.
