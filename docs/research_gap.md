# Research gap and related-work audit (first draft)

**Date:** 28 October 2026
**Status:** rapid, non-systematic audit. It is not a full systematic review and does not
establish novelty or priority.

## 1. The question this draft can actually answer

The research plan asks whether a radiation-monitoring system can combine measurement-integrity
verification with drift-aware spectral adaptation and maintain reliable detection under
changing backgrounds and corrupted telemetry, compared with established baselines.

This draft narrows that to a controlled simulation question:

> Under matched synthetic six-window Poisson streams with labelled background regimes,
> source events, and telemetry faults, does a four-condition integrity gate plus the original
> DAES scalar step-size rule improve detection/false-alarm/stability outcomes relative to
> static, EWMA, gated EWMA, RLS, Adam, and ungated DAES baselines?

This is a mechanism test, not a field study.

## 2. Closest literatures

### 2.1 Radiation portal monitoring and background variability

Radiation portal monitors (RPMs) compare a measured count or spectrum against a background
estimate and an alarm threshold. Operational practice must cope with naturally occurring
radioactive material (NORM), rain/radon transients, cosmic rays, vehicle shadow-shielding,
and cargo-induced background suppression.

Verified sources:

- Ghawaly et al., "RADAI: A Large-Scale Realistic Dataset for Radiation Detection Algorithm
  Development," *IEEE Transactions on Nuclear Science* 73(5):2045–2066, 2026.
  DOI: [10.1109/TNS.2026.3682654](https://doi.org/10.1109/TNS.2026.3682654). RADAI is a
  large-scale **synthetic** list-mode dataset with urban background variability, rain
  transients, and source signatures; it is not a ready-made 1 Hz six-window portal stream.
- Ghawaly et al., "Data for training and testing radiation detection algorithms in an urban
  environment," *Scientific Data* 7:328, 2020.
  DOI: [10.1038/s41597-020-00672-2](https://doi.org/10.1038/s41597-020-00672-2). The public
  urban dataset is list-mode (time-since-last-event in microseconds, energy in keV), CC BY
  4.0, and requires Globus access.
- Coulon, Kondrasovs, Boudergui, and Normand, "Moving Sources Detection Algorithm for
  Radiation Portal Monitors Used in a Linear Network," *IEEE Transactions on Nuclear
  Science* 61(4):2189–2194, 2014.
  DOI: [10.1109/TNS.2014.2299872](https://doi.org/10.1109/TNS.2014.2299872). The paper
  describes conventional threshold detection under Poisson background assumptions and a
  correlation-based multi-detector alternative.
- "Suppression of Background Count by Cargo Containers in Radiation Portal Monitor,"
  *IEEE Transactions on Nuclear Science*, 2023.
  DOI: [10.1109/TNS.2023.3333353](https://doi.org/10.1109/TNS.2023.3333353). Monte Carlo
  study reporting 20–30% average background suppression for an empty container and larger
  variation for some cargoes. This motivates modelling background suppression as a distinct
  regime, not as a source.
- "Radiation portal monitor data file format for comprehensive background radiation
  monitoring," *Nuclear Instruments and Methods in Physics Research A*, 2025.
  DOI: [10.1016/j.nima.2025.170516](https://doi.org/10.1016/j.nima.2025.170516). Argues for
  continuous 24 h background logging to diagnose environmental fluctuations and hardware
  faults. This supports the separation of telemetry integrity from source detection.
- "Improved Performance of Energy Window Ratio Criteria Obtained Using Multiple Windows at
  Radiation Portal Monitoring Sites," PNNL conference paper, 2006.
  <https://www.pnnl.gov/publications/improved-performance-energy-window-ratio-criteria-obtained-using-multiple-windows>
  (accessed 28 Oct 2026). Discusses multi-window criteria as a way to increase sensitivity
  while controlling nuisance alarms. It does not validate the specific six windows used here.

**What this literature supplies:** the operational problem, the importance of background
non-stationarity and energy windows, and evidence that background suppression and nuisance
alarms are real concerns. **What it does not supply:** a direct comparison of a
telemetry-integrity gate with a curvature-based scalar step-size rule. This draft therefore
treats that combination as a hypothesis, not an established gap.

### 2.2 Online adaptation and concept drift

- Gama, Žliobaitė, Bifet, Pechenizkiy, and Bouchachia, "A Survey on Concept Drift
  Adaptation," *ACM Computing Surveys* 46(4):44, 2014.
  DOI: [10.1145/2523813](https://doi.org/10.1145/2523813). Taxonomy of drift types,
  adaptation strategies, and evaluation methodology.
- Roberts, "Control Chart Tests Based on Geometric Moving Averages," *Technometrics*
  1(3):239–250, 1959.
  DOI: [10.1080/00401706.1959.10489860](https://doi.org/10.1080/00401706.1959.10489860).
  The EWMA baseline.
- Page, "Continuous Inspection Schemes," *Biometrika* 41(1–2):100–115, 1954.
  DOI: [10.1093/biomet/41.1-2.100](https://doi.org/10.1093/biomet/41.1-2.100). The CUSUM
  change-detection baseline; not implemented in this draft.
- Huber, "Robust Estimation of a Location Parameter," *Annals of Mathematical Statistics*
  35(1):73–101, 1964.
  DOI: [10.1214/aoms/1177703732](https://doi.org/10.1214/aoms/1177703732). Robust
  location estimation; motivates the robust/gated baseline family, though this draft uses a
  gate rather than a Huber loss.
- Isermann, "Model-based fault-detection and diagnosis – status and applications," *Annual
  Reviews in Control* 29(1):71–85, 2005.
  DOI: [10.1016/j.arcontrol.2004.12.002](https://doi.org/10.1016/j.arcontrol.2004.12.002).
  General fault-detection framing; the four gate conditions are domain-specific, not a
  general fault-diagnosis method.

**What this literature supplies:** the vocabulary of drift, EWMA/CUSUM baselines, robust
estimation, and fault detection. **What it does not supply:** a validated recipe for
distinguishing a genuine source from corrupted telemetry in a radiation portal. That
distinction is the hard part of the present problem.

### 2.3 Step-size adaptation and the original DAES

- Polyak, "Gradient methods for the minimisation of functionals," *USSR Computational
  Mathematics and Mathematical Physics* 3:864–878, 1963.
  DOI: [10.1016/0041-5553(63)90382-3](https://doi.org/10.1016/0041-5553(63)90382-3).
  Classical gradient-method stability and step-size analysis.
- Barzilai and Borwein, "Two-Point Step Size Gradient Methods," *IMA Journal of Numerical
  Analysis* 8(1):141–148, 1988.
  DOI: [10.1093/imanum/8.1.141](https://doi.org/10.1093/imanum/8.1.141). A standard
  adaptive step-size alternative.
- Kingma and Ba, "Adam: A Method for Stochastic Optimization," *ICLR* 2015,
  arXiv:1412.6980. <https://arxiv.org/abs/1412.6980>. The Adam baseline.
- Ljung, *System Identification: Theory for the User*, 2nd ed., Prentice Hall, 1999.
  Standard recursive least squares (RLS) reference; the RLS baseline uses a forgetting
  factor.
- The original DAES implementation and its companion paper were supplied separately. The
  implementation is a **scalar step-size heuristic** for a drifting strongly convex
  quadratic: power iteration on the Hessian, a Rayleigh quotient, a drift-responsive EMA,
  an upward correction, and `η = γ / ((1+s) λ̂_max)`. It is not a matrix preconditioner and
  not a new optimizer. See `docs/daes_interface.md`.

**What this literature supplies:** the step-size rule and its fixed-objective stability
condition. **What it does not supply:** a stochastic, non-stationary convergence guarantee
for the Rad-DAES loss, or evidence that DAES helps in radiation background tracking.

## 3. Candidate contributions (hypotheses, not claims)

| Candidate | Status in this draft |
|---|---|
| **A. Fault-aware adaptation control** — separate trustworthy background change, source events, and unreliable telemetry; test whether the gate prevents corrupted updates without suppressing real sources. | Primary hypothesis. Implemented and evaluated; results are mixed and scenario-dependent. |
| **B. Drift-conditioned DAES integration** — use the original DAES scalar step-size rule inside the guarded pipeline; compare with fixed-step, EWMA, RLS, and Adam. | Mechanism under test. Implemented faithfully; the original equations are preserved. |
| **C. Reproducible stress-test benchmark** — documented synthetic regimes, source events, fault modes, seeds, and metrics. | Supporting contribution. Implemented as a small benchmark, not a community standard. |

No claim is made that A, B, or C is novel. The rapid audit found adjacent work but did not
perform the systematic, protocol-driven review needed to establish a research gap. The
manuscript therefore uses "candidate" and "hypothesis" language throughout.

## 4. What would be needed before a novelty claim

1. A protocol-driven search across IEEE Xplore, Scopus/Web of Science, INIS, and arXiv with
   inclusion/exclusion criteria and a documented search string.
2. A related-work matrix comparing: radiation context, data, background model, integrity
   handling, adaptation rule, baselines, metrics, and limitations.
3. A direct comparison against at least one published RPM background-adaptation method on
   the same public data.
4. A check that the six-window representation and the gate conditions are not already
   standard practice in the cited literature.
5. A negative-result analysis if the full pipeline does not beat the baselines.

Until then, the defensible statement is: *this draft tests a plausible reliability-aware
pipeline in a controlled simulation and reports where it helps and where it does not.*

## 5. Search log (non-systematic)

Queries run on 28 October 2026 included:

- "RADAI radiological anomaly detection dataset Oak Ridge National Laboratory paper"
- "PyRIID radiation portal monitor synthetic data Python documentation"
- "radiation portal monitor background estimation false alarm drift paper"
- "concept drift adaptation survey Gama 2014"
- "JRC UPu-Gamma dataset gamma spectrometry uranium plutonium reference materials"
- "RADAI dataset access ORNL three datasets training developer testing list mode"

Results were used only when the source page or DOI metadata could be checked. No citation
was generated from memory alone. This is not an exhaustive search.
