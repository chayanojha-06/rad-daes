# Decisions, assumptions, and unresolved questions

**Status:** living document, last updated 28 October 2026.

## D1. DAES definition: original scalar rule, not the Gemini matrix rule

**Decision.** DAES is implemented as the original scalar step-size heuristic:
`η_t = γ / ((1+s) λ̂_max(H_t))`, with power iteration, a Rayleigh quotient, a
drift-responsive EMA, and an upward correction. The caller applies the scalar step.

**Why.** The supplied planning PDFs contain a different multi-direction rule
(`H_t = βH_{t−1} + (1−β)∇ℓ∇ℓ^T`, `Φ_ii = η_0/(sqrt(λ_i)+δ_0 exp(−κρ_t))`,
`θ_t = θ_{t−1} + VΦV^T r_t`). That rule is not in the original DAES code, uses a different
curvature recursion and damping law, and would silently change the method. The research plan
explicitly says to preserve the original method and document its interface. The matrix rule
is therefore **not** implemented as DAES.

**Consequence.** The manuscript describes DAES as a scalar step-size heuristic. It does not
claim a matrix preconditioner, a new optimizer, or the Gemini formulation. The matrix rule is
listed as future work and as an unverified proposal.

## D2. Curvature matrix: Gaussian quasi-likelihood Hessian

**Decision.** `H_t = Σ_{t−1}^{−1}`, the Hessian of
`L_t(θ) = 0.5 (x_t−θ)^T Σ_{t−1}^{−1} (x_t−θ)`.

**Why.** The detector already uses a covariance-weighted Mahalanobis score. A Gaussian
quasi-likelihood gives a well-defined Hessian and keeps the detector and the adaptation
mechanism consistent. A Poisson negative-log-likelihood would give a different Hessian
(`diag(x/θ²)` or `diag(1/θ)` under the expected Fisher information) and would require a
separate justification.

**Assumption.** The Gaussian quasi-likelihood is an approximation to Poisson counts. It is
not claimed to be the correct data model. A Poisson-likelihood variant is future work.

## D3. Covariance update

**Decision.** On eligible samples, update
`Σ̃ = (1−β)Σ + β u u^T`, shrink toward its diagonal with `ρ = 0.10`, and add a `1e−6`
floor. `β = 0.02`.

**Why.** A full covariance captures inter-window correlations that a diagonal Poisson
variance ignores. Shrinkage and a floor keep the matrix invertible with six windows and
finite samples.

**Unresolved.** The covariance is estimated from the same data used for detection. A
source event that passes the gate would inflate the covariance and reduce future scores.
The gate is intended to prevent this, but the interaction is not analysed theoretically.
Low-count windows may also make the covariance ill-conditioned; the shrinkage is a
practical, not a statistically optimal, remedy.

## D4. Gate timing bound

**Decision.** Accept `0.5 ≤ Δt ≤ 1.5` s. The planning document gives `0 < Δt ≤ Δt_max`.

**Why.** A lower bound is needed to detect short-interval timestamp jitter. Without it, a
jittered timestamp of 0.1 s would pass the original condition. This is a design choice, not
a claim about the true sampling tolerance of any detector.

## D5. Alarm persistence

**Decision.** Require two consecutive threshold exceedances (`L = 2`).

**Why.** The planning document specifies `L ≥ 2`. Persistence reduces isolated false alarms
but delays detection by one sample. The realised false-alarm rate is reported; it is not
assumed to equal the target.

## D6. Six energy windows

**Decision.** Use six windows with the boundaries in `configs/data.yaml`, taken from the
planning document.

**Why.** The planning document and the original project seed specify `D = 6`. The
eigen-decomposition is cheap and the detector is low-dimensional.

**Unresolved.** The boundaries are not validated against a calibrated detector. The
planning document's nominal count rates are illustrative. No claim is made that the windows
preserve source information or that they are optimal.

## D7. Real data

**Decision.** Do not use real data in this draft.

**Why.** No real radiation stream was available locally. RADAI requires a BDC account and a
large list-mode download; PyRIID requires Python 3.10–3.12 and a spectra-to-window
conversion; JRC UPu-Gamma is a reference-spectra collection, not a continuous portal stream.
No credentials were used. See `docs/dataset_audit.md`.

**Consequence.** Gate 1 is a partial failure. All results are synthetic. The manuscript
states this in the abstract and limitations.

## D8. Tuning budget and seeds

**Decision.** Five development seeds and five test seeds; a small grid; no per-regime
hyperparameter selection.

**Why.** The deadline is 30 October 2026 and the priority is a complete, reproducible
prototype. A larger search would increase the risk of overfitting the development scenarios
and would not fix the absence of real data.

**Consequence.** Confidence intervals are wide and describe seed variation only. No
significance claim is made.

## D9. Primary estimand

**Decision.** Mean sample-level F1 across the 32 held-out scenarios, each weighted equally.
Secondary: false alarms per hour, background relative error, recovery time.

**Why.** A single primary estimand prevents selecting the scenario where the method wins.
F1 is imperfect for rare events, so PR-AUC, recall, precision, and false-alarm burden are
also reported. The protocol is frozen in `docs/protocol.md`.

## D10. No novelty claim

**Decision.** The manuscript uses "candidate contribution" and "hypothesis" language and
does not claim priority or novelty.

**Why.** The literature audit is rapid and non-systematic. The research plan's Gate 4 is not
met. A novelty claim would require a protocol-driven review and a direct comparison on
public data.

## D11. Sustained-exceedance gate release

**Decision.** After 90 consecutive samples that pass validity, timing, and Poisson dispersion
but exceed the innovation threshold, the innovation condition is released. The release
resets when the score falls below threshold or another telemetry condition fails.

**Why.** A persistent background step makes every innovation exceed the threshold, so a pure
conjunction gate freezes the mean and the score never falls. This is a deadlock, not a
detection success. The 90 s patience is longer than the longest synthetic source event (60 s),
so a single source event cannot trigger it.

**Unresolved.** A source longer than 90 s, or a sequence of overlapping anomalies, would be
treated as drift. The release rate and its effect on source recall are reported. The patience
is a design parameter; the draft does not claim it is optimal.

## D12. Covariance floor

**Decision.** Floor each covariance diagonal at `max(1e-6, 0.25 * nominal_j)`, where
`nominal_j` is the nominal Poisson variance for window `j`.

**Why.** A count covariance below a fraction of the Poisson variance is not physically
meaningful and can make the Mahalanobis score unstable. The floor is a regularisation
choice, not a claim that the true covariance is Poisson.

**Unresolved.** The 0.25 fraction is arbitrary. A sensitivity analysis over the floor is
future work.

## D13. Synthetic source placement and profile

**Decision.** Place up to six events using a single gap between consecutive events, and
sample the raised-sine profile at interval midpoints so every labelled sample contains
positive source excess.

**Why.** The initial implementation advanced the cursor by two gaps and used a profile that
was zero at the first sample. Both were implementation bugs, not scientific choices. They
were fixed before the held-out matrix.

## D14. Report the frozen negative result without a rescue rerun

**Decision.** The held-out matrix was analysed as frozen. No method, hyperparameter,
threshold, seed, or source amplitude was changed after the test rows were inspected, and the
candidate was not re-tuned on test. The manuscript reports the negative result.

**Why.** Changing the design after seeing test performance would turn a prespecified comparison
into a tuning exercise. The primary estimand and the selection rule were fixed in
`docs/protocol.md` before the run; the observed outcome is that the candidate does not improve
the trade-off.

**Consequence.** The paper is a negative-result draft. Its contribution is the auditable
benchmark and the two documented failure modes, not a claim that Rad-DAES works.

## D15. Secondary threshold-free summaries are post-test additions

**Decision.** Add ROC/PR AUC and event-level detection as secondary summaries after the frozen
run, using columns already saved in `test_runs.csv`.

**Why.** F1 at a single calibrated threshold conflates separability with operating-point choice.
The frozen results showed that the threshold dominated the ranking, so a threshold-free summary
was needed to interpret the failure. The summaries do not change the primary estimand or the
selection rule, and they are labelled as post-test additions in `docs/protocol.md` and the
manuscript appendix.

**Unresolved.** No matched-FAR threshold sweep was run; that would be a new experiment.

## Open questions

1. Does a Poisson-likelihood curvature (instead of the Gaussian quasi-likelihood) change the
   DAES ranking?
2. Does the gate's innovation guard suppress genuine sources enough to erase the benefit of
   guarding? (H4 is measured, not assumed.)
3. Is the full covariance identifiable at the lowest-count window (nominal 55 cps)?
4. Would a per-direction matrix scaling (the Gemini proposal) outperform the original scalar
   rule? This draft does not test it.
5. How sensitive are the results to the six window boundaries, the fault rate, the source
   amplitude, the 90 s release patience, and the 50-sample covariance time constant? No
   sensitivity analysis was run in this draft.
6. Can the pipeline ingest RADAI list-mode data after a documented calibration and windowing
   step? Not attempted.
7. Is the synthetic drift model realistic enough to justify a follow-up? The current regimes
   are illustrative, not fitted to measured background.
8. Would a covariance-frozen ablation confirm the contamination mechanism inferred from the
   static-versus-adaptive contrast? Not run.
