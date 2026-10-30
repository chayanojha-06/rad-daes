"""Rad-DAES: Interactive Scientific Research Demonstration Prototype.

Authors: Tripti Kumari, Chayan Ojha, Mriganko Dey, Ankita Polley
Affiliation: Rad-DAES Research Group

Features:
- Dark navy scientific laboratory interface.
- Real-time 6-channel synthetic radiation stream simulator.
- Sample-by-sample InteractiveStreamEngine with play/step/jump controls.
- 4-Stage telemetry integrity gate monitoring & state freezing verification.
- DAES maximum eigenvalue tracking & dynamic learning rate visualization.
- Mahalanobis anomaly detection with two-consecutive-sample persistence.
- Full 9-method comparative benchmark engine (F1, FAR/h, Latency, Recovery, BG Error).
- CSV trace and benchmark result export.

Strict Research Disclosure: All generated radiation count streams are synthetic.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Ensure repository 'src' and 'prototype' are discoverable
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"
_PROT_DIR = _REPO_ROOT / "prototype"
for p in [str(_SRC_DIR), str(_PROT_DIR), str(_REPO_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from prototype.adapter import InteractiveStreamEngine
from prototype.batch import (
    ALL_METHODS,
    DEFAULT_OVERRIDES,
    DEFAULT_THRESHOLDS,
    compare_all_methods,
    load_prototype_configs,
)
from prototype.plots import (
    CHANNELS,
    plot_anomaly_detection_timeline,
    plot_daes_controller_dynamics,
    plot_method_comparisons,
    plot_multichannel_stream,
    plot_telemetry_gate_diagnostics,
    plot_tradeoff_scatter,
)
from rad_daes.data.synthetic import REGIMES, Stream, generate_stream
from rad_daes.evaluation.runner import make_spec

# Configure Streamlit page
st.set_page_config(
    page_title="Rad-DAES Research Prototype",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-End Laboratory Dark Styling
CUSTOM_CSS = """
<style>
    /* Dark laboratory palette */
    .stApp {
        background-color: #070a12;
        color: #e2e8f0;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0b101d;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    /* Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #0e1526;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 12px 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    div[data-testid="stMetricLabel"] {
        color: #94a3b8;
        font-size: 0.82rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    div[data-testid="stMetricValue"] {
        color: #00f2fe;
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.6rem;
        font-weight: 700;
    }
    
    /* Status Badges */
    .badge-normal {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }
    .badge-alarm {
        background-color: rgba(239, 68, 68, 0.2);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.4);
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        animation: pulse 2s infinite;
    }
    .badge-rejected {
        background-color: rgba(245, 158, 11, 0.15);
        color: #f59e0b;
        border: 1px solid rgba(245, 158, 11, 0.3);
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }
    .badge-synthetic {
        background-color: rgba(56, 189, 248, 0.1);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.3);
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 500;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: #94a3b8;
        font-weight: 500;
        border-radius: 4px 4px 0 0;
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        color: #00f2fe !important;
        border-bottom: 2px solid #00f2fe !important;
        background-color: rgba(0, 242, 254, 0.05) !important;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 6px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        border-color: #00f2fe;
        color: #00f2fe;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource
def get_cached_configs():
    """Load configuration files once."""
    return load_prototype_configs()


data_cfg, model_cfg, exp_cfg = get_cached_configs()

# Initialize Session State
if "stream" not in st.session_state:
    st.session_state.stream = None
if "engine" not in st.session_state:
    st.session_state.engine = None
if "comp_df" not in st.session_state:
    st.session_state.comp_df = None
if "selected_method" not in st.session_state:
    st.session_state.selected_method = "rad_daes"


def init_engine(stream: Stream, method_name: str, thresh: dict[str, float], override: dict):
    """Instantiate and prime a new InteractiveStreamEngine."""
    spec = make_spec(method_name, model_cfg, override=override)
    engine = InteractiveStreamEngine(
        stream=stream,
        spec=spec,
        thresholds=thresh,
        model_cfg=model_cfg,
        warmup=600 if stream.duration_seconds > 1200 else int(stream.duration_seconds * 0.1),
    )
    st.session_state.engine = engine


# ---------------------------------------------------------------------------
# SIDEBAR CONTROLS
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚛️ Rad-DAES Simulator")
    st.markdown('<span class="badge-synthetic">SYNTHETIC EXPERIMENTAL DATA</span>', unsafe_allow_html=True)
    st.markdown("---")

    st.subheader("1. Stream Configuration")
    regime = st.selectbox(
        "Background Regime",
        options=["stable", "gradual", "abrupt", "recurring"],
        index=2,  # Abrupt by default
        help="Simulated macroscopic background evolution dynamics.",
    )

    fault_condition = st.selectbox(
        "Telemetry Fault Condition",
        options=[
            "clean",
            "mixed_5",
            "mixed_10",
            "mixed_15",
            "dropout_10",
            "jitter_10",
            "stuck_10",
            "spike_10",
        ],
        index=2,  # mixed_10 by default
        help="Simulated sensor and communication channel failure modes.",
    )

    duration = st.slider(
        "Stream Duration (seconds)",
        min_value=300,
        max_value=7200,
        value=1800,
        step=300,
        help="Length of the synthetic count stream (1 Hz sampling rate).",
    )

    col_seed1, col_seed2 = st.columns([2, 1])
    with col_seed1:
        seed = st.number_input("RNG Seed", value=1001, min_value=1, max_value=99999, step=1)
    with col_seed2:
        st.write("")
        st.write("")
        preset_seed = st.selectbox("Presets", options=[1001, 1002, 11, 12, 13], index=0)
        if preset_seed != seed:
            seed = preset_seed

    include_sources = st.checkbox("Inject Radioactive Sources", value=True)
    n_sources = st.number_input("Source Events", value=6, min_value=1, max_value=20) if include_sources else 0

    if st.button("Generate Synthetic Stream", type="primary", use_container_width=True):
        with st.spinner("Generating 6-channel Poisson stream..."):
            st.session_state.stream = generate_stream(
                seed=int(seed),
                regime=regime,
                fault_condition=fault_condition,
                duration_seconds=int(duration),
                include_sources=include_sources,
                n_source_events=int(n_sources),
            )
            st.session_state.comp_df = None
            # Prime engine with selected method
            m_name = st.session_state.selected_method
            thresh = DEFAULT_THRESHOLDS.get(m_name, {"alarm": 65.0, "gate": 55.0})
            ovr = DEFAULT_OVERRIDES.get(m_name, {})
            init_engine(st.session_state.stream, m_name, thresh, ovr)
            st.success(f"Stream generated ({duration}s, {regime}, {fault_condition}).")

    st.markdown("---")
    st.subheader("2. Algorithm Configuration")
    method_name = st.selectbox(
        "Active Adaptation Method",
        options=ALL_METHODS,
        index=ALL_METHODS.index(st.session_state.selected_method),
        help="Select any of the 9 benchmarked background adaptation algorithms.",
    )
    if method_name != st.session_state.selected_method:
        st.session_state.selected_method = method_name
        if st.session_state.stream is not None:
            thresh = DEFAULT_THRESHOLDS.get(method_name, {"alarm": 30.0, "gate": 25.0})
            ovr = DEFAULT_OVERRIDES.get(method_name, {})
            init_engine(st.session_state.stream, method_name, thresh, ovr)

    # Threshold overrides
    std_thresh = DEFAULT_THRESHOLDS.get(method_name, {"alarm": 30.0, "gate": 25.0})
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        alarm_th = st.number_input("Alarm Threshold τ", value=float(std_thresh["alarm"]), format="%.2f")
    with col_t2:
        gate_th = st.number_input("Gate Threshold τ_g", value=float(std_thresh["gate"]), format="%.2f")

    if st.button("Apply Parameters & Reset Engine", use_container_width=True):
        if st.session_state.stream is not None:
            thresh = {"alarm": float(alarm_th), "gate": float(gate_th)}
            ovr = DEFAULT_OVERRIDES.get(method_name, {})
            init_engine(st.session_state.stream, method_name, thresh, ovr)
            st.info("Engine re-initialized with updated parameters.")

    st.markdown("---")
    st.caption("Rad-DAES Research Group • College Review Release")


# If stream not created yet, create default clean stream
if st.session_state.stream is None:
    st.session_state.stream = generate_stream(
        seed=1001,
        regime="abrupt",
        fault_condition="mixed_10",
        duration_seconds=1800,
        include_sources=True,
        n_source_events=6,
    )
    thresh = DEFAULT_THRESHOLDS["rad_daes"]
    ovr = DEFAULT_OVERRIDES["rad_daes"]
    init_engine(st.session_state.stream, "rad_daes", thresh, ovr)

engine: InteractiveStreamEngine = st.session_state.engine
stream: Stream = st.session_state.stream

# ---------------------------------------------------------------------------
# MAIN HEADER & RESEARCH DISCLOSURE
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 12px; margin-bottom: 16px;">
        <div>
            <h2 style="margin: 0; font-weight: 700; color: #ffffff;">Rad-DAES: Drift-Aware Background Estimation & Anomaly Detection</h2>
            <p style="margin: 4px 0 0 0; color: #94a3b8; font-size: 0.95rem;">
                Interactive Research Demonstration Prototype • Authors: <b>Tripti Kumari, Chayan Ojha, Mriganko Dey, Ankita Polley</b>
            </p>
        </div>
        <div style="text-align: right;">
            <span class="badge-synthetic">100% SYNTHETIC RAD STREAM</span>
            <div style="color: #64748b; font-size: 0.8rem; margin-top: 4px;">Sampling rate: 1.0 Hz • 6 Spectral Windows</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# TABS
# ---------------------------------------------------------------------------
tab_streaming, tab_benchmarks, tab_math, tab_export = st.tabs(
    [
        "🔬 Interactive Streaming Lab",
        "📊 9-Method Comparative Benchmark",
        "📐 Mathematical Foundations & Gating",
        "💾 Data & Trace CSV Export",
    ]
)

# ===========================================================================
# TAB 1: INTERACTIVE STREAMING LAB
# ===========================================================================
with tab_streaming:
    # Playback and Step Controls
    st.markdown("#### Simulation Stepping Controls")
    ctrl1, ctrl2, ctrl3, ctrl4, ctrl5, ctrl6, ctrl7 = st.columns([1, 1, 1, 1, 2, 1.5, 1])

    with ctrl1:
        if st.button("▶ +1 Step", use_container_width=True):
            engine.step()
    with ctrl2:
        if st.button("⏩ +10 Steps", use_container_width=True):
            engine.step_n(10)
    with ctrl3:
        if st.button("⏭ +100 Steps", use_container_width=True):
            engine.step_n(100)
    with ctrl4:
        if st.button("🚀 +500 Steps", use_container_width=True):
            engine.step_n(500)
    with ctrl5:
        if st.button("🎯 Advance to Next Event", use_container_width=True):
            with st.spinner("Stepping to event/fault..."):
                engine.step_until_event_or_fault()
    with ctrl6:
        if st.button("⚡ Run Full Stream", use_container_width=True):
            with st.spinner("Processing full stream..."):
                engine.run_all()
    with ctrl7:
        if st.button("↺ Reset", use_container_width=True):
            engine.reset()

    # Progress bar
    t_curr = engine.current_t
    t_total = engine.duration
    st.progress(float(t_curr) / float(t_total), text=f"Observation Step: t = {t_curr} / {t_total} seconds ({t_curr/t_total*100:.1f}%)")

    # Real-Time Telemetry & Status HUD
    snap = engine.last_snapshot
    res = engine.get_run_results()

    kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)

    # 1. Gate Status
    with kpi1:
        if snap is None:
            gate_html = '<span class="badge-normal">INITIALIZED</span>'
        elif snap.decision.sustained_override:
            gate_html = '<span class="badge-rejected">SUSTAINED OVERRIDE</span>'
        elif snap.decision.passed:
            gate_html = '<span class="badge-normal">GATE PASSED</span>'
        else:
            gate_html = '<span class="badge-alarm">GATE REJECTED</span>'
        st.markdown(f"**Gate Status**<br>{gate_html}", unsafe_allow_html=True)
        if snap:
            st.caption(f"Valid: {snap.decision.valid} | Time: {snap.decision.timing}")

    # 2. Alarm Status
    with kpi2:
        if snap is None or snap.alarm == 0:
            alarm_html = '<span class="badge-normal">NORMAL (NO ALARM)</span>'
        else:
            alarm_html = '<span class="badge-alarm">🚨 ANOMALY ALARM</span>'
        st.markdown(f"**Detector State**<br>{alarm_html}", unsafe_allow_html=True)
        if snap:
            st.caption(f"Score s_t: {snap.score:.2f} (τ={engine.thresholds['alarm']:.1f})")

    # 3. Model Updates
    with kpi3:
        st.metric("Updates Applied", f"{engine.updates}", delta=f"{engine.updates/(max(1, t_curr))*100:.1f}% pass rate")

    # 4. F1 Score
    with kpi4:
        f1_val = res.get("f1", float("nan"))
        st.metric("Detection F1", f"{f1_val:.3f}" if np.isfinite(f1_val) else "N/A")

    # 5. False Alarms per Hour
    with kpi5:
        far_val = res.get("far_per_hour", float("nan"))
        st.metric("FAR / hour", f"{far_val:.2f}" if np.isfinite(far_val) else "0.00", delta="-Target ≤ 1.0" if far_val > 1.0 else "In spec", delta_color="inverse")

    # 6. Background Relative Error
    with kpi6:
        bg_err = res.get("bg_rel_error", float("nan"))
        st.metric("BG Rel Error", f"{bg_err*100:.2f}%" if np.isfinite(bg_err) else "N/A")

    st.markdown("---")

    # Channel Display Selectors
    col_v1, col_v2 = st.columns([3, 1])
    with col_v1:
        st.markdown("#### Live Telemetry & Multichannel Spectral Count Traces")
    with col_v2:
        selected_channels = st.multiselect(
            "Visible Windows",
            options=CHANNELS,
            default=["W1", "W2", "W3"],
            help="Toggle spectral energy bins (W1: 30-120 keV, W2: 120-450 keV, etc.).",
        )

    # Traces DataFrame
    df_trace = engine.to_dataframe()

    if df_trace.empty:
        st.info("Advance the stream using the controls above to start plotting telemetry.")
    else:
        # Plot 1: Multichannel Count Stream
        fig_stream = plot_multichannel_stream(
            df_trace,
            selected_channels=selected_channels,
            show_true_mean=True,
            show_est_mean=True,
        )
        st.plotly_chart(fig_stream, use_container_width=True)

        col_p1, col_p2 = st.columns(2)

        with col_p1:
            # Plot 2: Anomaly Scoring & Alarm Persistence
            fig_score = plot_anomaly_detection_timeline(
                df_trace,
                alarm_threshold=float(engine.thresholds["alarm"]),
                gate_threshold=float(engine.thresholds["gate"]),
            )
            st.plotly_chart(fig_score, use_container_width=True)

        with col_p2:
            # Plot 3: Gate Integrity Verification Timeline
            fig_gate = plot_telemetry_gate_diagnostics(df_trace)
            st.plotly_chart(fig_gate, use_container_width=True)

        # Plot 4: DAES Controller Dynamics (only shown for DAES variants)
        if engine.spec.update_kind in {"daes", "daes_fixed"}:
            st.markdown("#### DAES Adaptive Controller Curvature & Learning Rate Dynamics")
            fig_daes = plot_daes_controller_dynamics(df_trace)
            st.plotly_chart(fig_daes, use_container_width=True)


# ===========================================================================
# TAB 2: 9-METHOD COMPARATIVE BENCHMARK
# ===========================================================================
with tab_benchmarks:
    st.markdown("### Rigorous Benchmark Evaluation Across All 9 Baseline Methods")
    st.markdown(
        """
        Evaluate and compare all nine research methods on the **exact same synthetic radiation stream**
        using the identical `run_stream()` execution engine and calibrated development parameters.
        """
    )

    col_b1, col_b2 = st.columns([1, 3])
    with col_b1:
        st.write("")
        if st.button("🚀 Run All 9 Methods Benchmark", type="primary", use_container_width=True):
            with st.spinner("Evaluating all 9 methods on current scenario..."):
                st.session_state.comp_df = compare_all_methods(
                    stream=stream,
                    model_cfg=model_cfg,
                    warmup=engine.warmup,
                )
            st.success("Benchmark evaluation completed across all 9 methods!")

    with col_b2:
        st.caption(
            f"Scenario: **{stream.regime.upper()}** background | **{stream.fault_condition}** telemetry | "
            f"Duration: **{stream.duration_seconds} s** | Seed: **{stream.seed}**"
        )

    if st.session_state.comp_df is not None:
        comp_df: pd.DataFrame = st.session_state.comp_df

        # Display Comparison Table with formatting
        st.markdown("#### Benchmark Metric Summary Table")
        formatted_df = comp_df.copy()
        formatted_df["f1"] = formatted_df["f1"].map(lambda v: f"{v:.3f}" if np.isfinite(v) else "—")
        formatted_df["precision"] = formatted_df["precision"].map(lambda v: f"{v:.3f}" if np.isfinite(v) else "—")
        formatted_df["recall"] = formatted_df["recall"].map(lambda v: f"{v:.3f}" if np.isfinite(v) else "—")
        formatted_df["far_per_hour"] = formatted_df["far_per_hour"].map(lambda v: f"{v:.2f}" if np.isfinite(v) else "—")
        formatted_df["bg_rel_error"] = formatted_df["bg_rel_error"].map(lambda v: f"{v*100:.2f}%" if np.isfinite(v) else "—")
        formatted_df["latency_mean"] = formatted_df["latency_mean"].map(lambda v: f"{v:.1f} s" if np.isfinite(v) else "—")
        formatted_df["recovery_mean"] = formatted_df["recovery_mean"].map(lambda v: f"{v:.1f} s" if np.isfinite(v) else "—")
        formatted_df["gate_pass_rate"] = formatted_df["gate_pass_rate"].map(lambda v: f"{v*100:.1f}%" if np.isfinite(v) else "—")

        st.dataframe(
            formatted_df.rename(
                columns={
                    "method": "Method",
                    "f1": "F1 Score",
                    "precision": "Precision",
                    "recall": "Recall",
                    "far_per_hour": "FAR / h",
                    "bg_rel_error": "BG Rel Error",
                    "latency_mean": "Latency (s)",
                    "recovery_mean": "Recovery (s)",
                    "n_updates": "Updates",
                    "gate_pass_rate": "Gate Pass",
                    "runtime_seconds": "Runtime (s)",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        # Plotly Bar Comparisons
        st.markdown("#### Metric Visual Breakdown")
        fig_comp = plot_method_comparisons(comp_df)
        st.plotly_chart(fig_comp, use_container_width=True)

        # Operating Trade-Off
        st.markdown("#### Operating Trade-Off: F1 Detection vs False Alarms")
        fig_tradeoff = plot_tradeoff_scatter(comp_df)
        st.plotly_chart(fig_tradeoff, use_container_width=True)
    else:
        st.info("Click 'Run All 9 Methods Benchmark' above to execute the multi-method comparative evaluation.")


# ===========================================================================
# TAB 3: MATHEMATICAL FOUNDATIONS & GATING
# ===========================================================================
with tab_math:
    st.markdown("### Rad-DAES Mathematical Specifications & Invariants")
    st.markdown(
        r"""
        Rad-DAES addresses two failure modes of dynamic radiation portal monitoring:
        1. **Noise amplification & divergence** caused by unconstrained learning rates during abrupt non-stationary shifts.
        2. **Contamination and poisoning** caused by corrupt telemetry, packet loss, and high-flux radioactive source events.

        ---
        #### 1. Telemetry Integrity Gate $g_t$
        Every incoming count vector $x_t \in \mathbb{R}^6$ at timestamp $t_{\text{obs}}$ passes through a four-stage sieve before model updating:
        """
    )
    st.latex(
        r"g_t = \text{Valid}(x_t) \land \text{Timing}(\Delta t) \land \text{Poisson}(x_t) \land \left[\text{Innovation}(s_t) \lor \text{Override}(t)\right]"
    )
    st.markdown(
        r"""
        * **Validity:** $x_{t, k} \ge 0$ and all elements finite.
        * **Timing:** Interval $\Delta t = t_{\text{obs}, t} - t_{\text{obs}, t-1} \in [0.5, 1.5]$ seconds.
        * **Poisson Dispersion:** Rolling 10-sample Fano factor $\text{Var}(x_k)/\mathbb{E}[x_k] \in [0.25, 3.50]$ across each window.
        * **Pre-Update Innovation:** Mahalanobis score $s_t = (x_t - \mu_{t-1})^\top \Sigma_{t-1}^{-1} (x_t - \mu_{t-1}) \le \tau_{\text{gate}}$.
        * **Sustained Override:** Released after 90 consecutive innovation rejections to prevent permanent deadlock during unmodeled physical shifts.

        > **Invariant:** When $g_t = 0$, both the background model $(\mu, \Sigma, \Sigma^{-1})$ and the DAES controller state remain **strictly frozen**.
        """
    )

    st.markdown("---")
    st.markdown(r"#### 2. Dynamic Curvature-Constrained Learning Rate $\eta_t$")
    st.latex(
        r"\eta_t = \frac{\gamma}{(1 + \text{safety}) \cdot \hat{\lambda}_{\max, t}}"
    )
    st.markdown(
        r"""
        * $\gamma$: Master adaptation gain (`gain = 0.50` in default calibrated Rad-DAES).
        * $\text{safety}$: Curvature safety headroom (`safety = 0.10`).
        * $\hat{\lambda}_{\max, t}$: Filtered estimate of the maximum eigenvalue of regularized curvature $\Sigma_{t-1}^{-1}$, tracking Rayleigh quotient $q_t$.
        
        *Source Implementation Check (`src/rad_daes/adaptation/daes.py#L124`):*
        ```python
        eta = self.gain / ((1.0 + self.safety) * max(float(self._estimate), 1e-12))
        ```
        """
    )
    st.latex(
        r"v_{k+1} = \frac{\Sigma^{-1} v_k}{\|\Sigma^{-1} v_k\|_2}, \quad q_t = \frac{v^\top \Sigma^{-1} v}{v^\top v}"
    )

    st.markdown("---")
    st.markdown("#### 3. Anomaly Scoring & Two-Sample Alarm Persistence")
    st.latex(
        r"a_t = \mathbb{I}\left(s_t > \tau_{\text{alarm}} \;\land\; s_{t-1} > \tau_{\text{alarm}}\right)"
    )
    st.markdown(
        """
        Isolated single-sample statistical fluctuations exceeding the threshold do **not** trigger an alarm.
        An alarm requires two consecutive samples exceeding $\tau_{\text{alarm}}$, drastically suppressing false alarms per hour.
        """
    )


# ===========================================================================
# TAB 4: DATA & TRACE CSV EXPORT
# ===========================================================================
with tab_export:
    st.markdown("### Export Simulation Traces and Benchmark Metrics")
    st.markdown("Download full research telemetry logs for offline post-processing in Python, R, or MATLAB.")

    col_e1, col_e2 = st.columns(2)

    with col_e1:
        st.markdown("#### 1. Interactive Trace Export")
        if df_trace.empty:
            st.info("No trace data available. Advance the stream in Tab 1 first.")
        else:
            csv_trace = df_trace.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Trace CSV (All Channels & Flags)",
                data=csv_trace,
                file_name=f"rad_daes_trace_{stream.regime}_{stream.fault_condition}_seed{stream.seed}.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.caption(f"Includes {len(df_trace)} time-steps, 6 raw channels, true means, estimated means, scores, alarms, and gate sub-checks.")

    with col_e2:
        st.markdown("#### 2. Benchmark Comparison Export")
        if st.session_state.comp_df is None:
            st.info("No benchmark comparison executed yet. Run the benchmark in Tab 2 first.")
        else:
            csv_comp = st.session_state.comp_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download 9-Method Benchmark CSV",
                data=csv_comp,
                file_name=f"rad_daes_9methods_comparison_{stream.regime}_{stream.fault_condition}_seed{stream.seed}.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.caption("Includes F1, Precision, Recall, FAR/h, Latency, Recovery Time, and Background Relative Error for all 9 methods.")
