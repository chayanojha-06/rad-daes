# Rad-DAES: Interactive Research Demonstration Prototype

**Project:** Rad-DAES (Guarded Drift-Aware Background Tracking for Synthetic Radiation Count Streams)  
**Authors:** Tripti Kumari, Chayan Ojha, Mriganko Dey, Ankita Polley  
**Affiliation:** Rad-DAES Research Group  
**Target Environment:** Python 3.10+ (Tested on Python 3.14.7, Windows x64)

---

## 1. Overview

This interactive software prototype provides a real-time scientific demonstration environment for the **Rad-DAES** algorithm, built using Python, Streamlit, and Plotly.

The prototype simulates 1 Hz multi-channel gamma-ray radiation portal monitoring (RPM) count streams under non-stationary physical background variations and corrupted telemetry channels. It allows faculty reviewers and researchers to observe step-by-step background adaptation, state freezing during sensor corruption, Mahalanobis anomaly detection, and head-to-head benchmarking against all nine published research methods.

> **Research Integrity Disclosure:** All radiation data, background regimes, source events, and telemetry faults generated in this prototype are strictly **synthetic**, following the verified physical design parameters of the Rad-DAES benchmark suite.

---

## 2. Prototype Architecture

All prototype components reside exclusively inside `prototype/`, preserving the research source tree and historical benchmarks untouched:

```
prototype/
├── __init__.py           # Package marker
├── adapter.py            # InteractiveStreamEngine: Step-by-step streaming adapter
├── batch.py              # Multi-method batch runner evaluating all 9 baseline algorithms
├── plots.py              # Dark-navy scientific Plotly visualization suite
├── app.py                # Multi-tab interactive Streamlit research laboratory
├── test_prototype.py     # Rigorous integration, parity, and state-freezing tests
├── requirements.txt      # Prototype runtime dependencies
└── README.md             # Architecture documentation and launch instructions
```

---

## 3. Setup and Launch Instructions

### A. Environment Prerequisites
Ensure Python is installed with the required libraries:

```powershell
pip install -r prototype/requirements.txt
```

### B. Launch the Interactive Dashboard
From the repository root (`C:\Users\chaya\OneDrive\Documents\Default Project\rad-daes`), execute:

```powershell
streamlit run prototype/app.py
```

Streamlit will launch a local web server (typically at `http://localhost:8501`) and automatically open the application in your default web browser.

---

## 4. Key Scientific Capabilities

### 1. Interactive Streaming Laboratory (Tab 1)
* **Stepping Controls:** Step +1, +10, +100, +500 samples, or jump directly to the onset of the next injected source event or telemetry fault.
* **Live Telemetry HUD:** Displays real-time gate integrity decisions, sub-test verification flags, active Mahalanobis score $s_t$, alarm status, cumulative F1, FAR/hour, and background relative error.
* **Multichannel Spectral Visualizer:** Shows count rates across 6 energy windows ($W_1$ to $W_6$), true background mean $\mu^*$, estimated mean $\hat{\mu}$, and source injection bands.
* **Sequential Persistence Anomaly Score:** Visualizes the pre-update Mahalanobis distance vs. alarm threshold $\tau_{\text{alarm}}$ and demonstrates the two-consecutive-sample filter suppressing transient single-sample spikes.
* **Telemetry Integrity Sieve:** Real-time diagnostics for validity, timestamp jitter, rolling 10-sample Poisson Fano factor boundaries ($[0.25, 3.50]$), and the 90-step sustained drift override.
* **DAES Adaptive Controller Dynamics:** Plots the maximum curvature eigenvalue $\hat{\lambda}_t$ (via 7 power iterations) and adaptive step size $\eta_t = \frac{\gamma}{(1+s)\hat{\lambda}_t}$, explicitly showing frozen states on rejected observations.

### 2. Nine-Method Comparative Benchmark (Tab 2)
Evaluates and contrasts all nine research baseline methods on the exact same synthetic count stream:
1. `static`: Ineligible for updates; fixed background.
2. `ewma`: Standard ungated Exponentially Weighted Moving Average ($\alpha = 0.02$).
3. `gated_ewma`: Telemetry-gated EWMA ($\alpha = 0.005$).
4. `rls`: Recursive Least Squares ($1 - \lambda = 0.02$).
5. `adam`: First and second moment adaptive gradient on precision residuals ($\eta = 0.02$).
6. `daes`: Ungated DAES with dynamic learning rate ($\gamma = 0.20$).
7. `gated_daes_fixed`: Telemetry-gated DAES with fixed step size ($\gamma = 0.50$).
8. `rad_daes`: Fully adaptive, curvature-constrained, telemetry-gated DAES ($\gamma = 0.50$, quantile = 0.999).
9. `oracle_gate`: Ideal ground-truth gate ($y=0 \land \text{fault}=\text{none}$).

Metrics computed:
* Detection F1 Score, Precision, Recall
* False Alarms per Hour (FAR/h)
* Detection Latency (mean & median seconds)
* Recovery Time (mean & median seconds)
* Background Relative $L_2$ Error
* Update Stability & Number of Updates

### 3. Mathematical Foundations & Invariants (Tab 3)
Complete LaTeX mathematical specifications detailing Rayleigh quotient optimization, covariance regularization with Ledoit-Wolf shrinkage, 4-stage telemetry gating, and state-freezing invariants.

### 4. Data & Trace Export (Tab 4)
One-click CSV download for:
* Full sample-by-sample simulation telemetry (6 raw channels, true means, estimated means, scores, alarms, gate sub-checks).
* 9-method comparative benchmark results.

---

## 5. Verification & Testing

To run the automated integration and mathematical parity test suite:

```powershell
pytest prototype/test_prototype.py -v
```

### Verified Test Cases:
1. `test_synthetic_stream_generation`: Validates stream shapes, sampling rates, and metadata.
2. `test_adapter_runner_exact_parity`: Verifies bit-level parity between `InteractiveStreamEngine` and `runner.run_stream()` across `rad_daes`, `gated_ewma`, and `static`.
3. `test_gate_rejection_freezes_adaptive_state`: Proves that when the gate rejects an observation, model mean, covariance, precision, and DAES controller state remain strictly frozen.
4. `test_alarm_persistence_two_consecutive_rule`: Confirms that isolated spikes do not trigger alarms, while two consecutive exceedances trigger alarms.
5. `test_compare_all_nine_methods`: Confirms all 9 baseline methods execute without error and produce valid metrics.
6. `test_dataframe_trace_export`: Validates tabular export formats and CSV byte encoding.
7. `test_scientific_plots_generation`: Validates all 6 Plotly laboratory figures build cleanly without rendering errors.

---

## 6. Research Preservation Notice

This prototype has been developed with strict non-interference guarantees:
* Original source files (`src/rad_daes/*`) were inspected but **never modified**.
* Existing configuration files (`configs/*`) remain intact.
* Raw benchmark data (`results/raw/test_runs.csv`) and publication figures were preserved.
* The frozen submission archive (`submission/rad-daes-submission.zip`) and checksum manifest (`submission/CHECKSUMS.sha256`) remain untouched.
