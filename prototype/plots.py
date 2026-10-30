"""Rad-DAES: High-Performance Scientific Plotly Visualizers.

Design Philosophy:
- Dark navy / near-black research-laboratory aesthetic (#0a0e17 background, #0e1526 canvas).
- Electric cyan (#00f2fe) for signals, estimates, and Rad-DAES primary metrics.
- Restrained crimson (#ef4444) for alarms, fault injections, and threshold exceedances.
- Amber / gold (#f59e0b) for telemetry gate rejections and Fano dispersion boundaries.
- Emerald (#10b981) for ground truth background mean.
- Clean typography, high DPI responsiveness, interactive hover tooltips.

All generated radiation data are strictly synthetic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Scientific Dark Palette
COLOR_BG = "#0a0e17"
COLOR_PAPER = "#0d1322"
COLOR_GRID = "rgba(255, 255, 255, 0.08)"
COLOR_TEXT = "#e2e8f0"
COLOR_TEXT_MUTED = "#94a3b8"
COLOR_CYAN = "#00f2fe"
COLOR_CRIMSON = "#ef4444"
COLOR_AMBER = "#f59e0b"
COLOR_EMERALD = "#10b981"
COLOR_PURPLE = "#a855f7"
COLOR_BLUE = "#3b82f6"

CHANNELS = ["W1", "W2", "W3", "W4", "W5", "W6"]
CHANNEL_COLORS = [
    "#38bdf8",  # W1: Light Sky Blue
    "#818cf8",  # W2: Indigo
    "#c084fc",  # W3: Purple
    "#fb7185",  # W4: Rose
    "#fbbf24",  # W5: Amber
    "#34d399",  # W6: Emerald
]


def _apply_dark_theme(fig: go.Figure, title: str = "") -> go.Figure:
    """Apply consistent laboratory dark theme to a Plotly figure."""
    fig.update_layout(
        title={
            "text": title,
            "y": 0.96,
            "x": 0.02,
            "xanchor": "left",
            "yanchor": "top",
            "font": {"size": 15, "color": COLOR_TEXT, "family": "Inter, sans-serif"},
        },
        paper_bgcolor=COLOR_BG,
        plot_bgcolor=COLOR_PAPER,
        font={"color": COLOR_TEXT, "family": "Inter, sans-serif"},
        hoverlabel={"bgcolor": "#1e293b", "font_color": "#ffffff", "font_size": 12},
        margin={"l": 55, "r": 35, "t": 55, "b": 45},
        legend={
            "bgcolor": "rgba(13, 19, 34, 0.8)",
            "bordercolor": "rgba(255, 255, 255, 0.12)",
            "borderwidth": 1,
            "font": {"size": 11, "color": COLOR_TEXT_MUTED},
        },
    )
    fig.update_xaxes(
        gridcolor=COLOR_GRID,
        zerolinecolor=COLOR_GRID,
        showline=True,
        linecolor="rgba(255, 255, 255, 0.15)",
        tickfont={"size": 10, "color": COLOR_TEXT_MUTED},
    )
    fig.update_yaxes(
        gridcolor=COLOR_GRID,
        zerolinecolor=COLOR_GRID,
        showline=True,
        linecolor="rgba(255, 255, 255, 0.15)",
        tickfont={"size": 10, "color": COLOR_TEXT_MUTED},
    )
    return fig


def plot_multichannel_stream(
    df: pd.DataFrame,
    selected_channels: list[str] | None = None,
    show_true_mean: bool = True,
    show_est_mean: bool = True,
    max_points: int = 4000,
) -> go.Figure:
    """Plot six-channel count stream with source and background overlays."""
    if df.empty:
        fig = go.Figure()
        return _apply_dark_theme(fig, "No stream data available")

    # Downsample if needed for smooth interactive rendering
    n_total = len(df)
    if n_total > max_points:
        step = int(np.ceil(n_total / max_points))
        plot_df = df.iloc[::step].copy()
    else:
        plot_df = df

    if not selected_channels:
        selected_channels = ["W1", "W2", "W3"]

    fig = go.Figure()

    # Highlight source injection periods (y==1)
    if "y_true" in plot_df.columns:
        source_spans = []
        in_event = False
        start_t = 0
        for idx, row in plot_df.iterrows():
            if row["y_true"] == 1 and not in_event:
                in_event = True
                start_t = row["t_obs"]
            elif row["y_true"] == 0 and in_event:
                in_event = False
                source_spans.append((start_t, row["t_obs"]))
        if in_event:
            source_spans.append((start_t, plot_df["t_obs"].iloc[-1]))

        for i, (t0, t1) in enumerate(source_spans):
            fig.add_vrect(
                x0=t0,
                x1=t1,
                fillcolor="rgba(239, 68, 68, 0.15)",
                layer="below",
                line_width=0,
                annotation_text="Source Injection" if i == 0 else None,
                annotation_position="top left",
                annotation_font={"size": 9, "color": "#fca5a5"},
            )

    # Highlight telemetry fault periods (fault != "none")
    if "fault" in plot_df.columns:
        fault_mask = plot_df["fault"] != "none"
        if fault_mask.any():
            fault_df = plot_df[fault_mask]
            fig.add_trace(
                go.Scatter(
                    x=fault_df["t_obs"],
                    y=np.zeros(len(fault_df)),
                    mode="markers",
                    marker={"symbol": "triangle-up", "size": 7, "color": COLOR_AMBER},
                    name="Fault Injected",
                    hoverinfo="text",
                    hovertext=[f"Fault: {f} at t={t:.1f}s" for f, t in zip(fault_df["fault"], fault_df["t_obs"])],
                )
            )

    # Plot channel count streams and background means
    for ch_name in selected_channels:
        if f"{ch_name}_counts" not in plot_df.columns:
            continue
        ch_idx = CHANNELS.index(ch_name) if ch_name in CHANNELS else 0
        color = CHANNEL_COLORS[ch_idx % len(CHANNEL_COLORS)]

        # Raw counts
        fig.add_trace(
            go.Scatter(
                x=plot_df["t_obs"],
                y=plot_df[f"{ch_name}_counts"],
                mode="lines",
                line={"color": color, "width": 1.2},
                name=f"{ch_name} Counts (cps)",
                opacity=0.75,
            )
        )

        # Ground truth mean
        if show_true_mean and f"{ch_name}_mu_true" in plot_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=plot_df["t_obs"],
                    y=plot_df[f"{ch_name}_mu_true"],
                    mode="lines",
                    line={"color": color, "width": 1.8, "dash": "dot"},
                    name=f"{ch_name} True Mean μ*",
                    opacity=0.6,
                )
            )

        # Estimated mean
        if show_est_mean and f"{ch_name}_mu_est" in plot_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=plot_df["t_obs"],
                    y=plot_df[f"{ch_name}_mu_est"],
                    mode="lines",
                    line={"color": "#ffffff", "width": 2.0},
                    name=f"{ch_name} Estimated μ",
                    opacity=0.9,
                )
            )

    fig.update_xaxes(title_text="Observation Time t (seconds)")
    fig.update_yaxes(title_text="Count Rate (counts / second)")
    return _apply_dark_theme(fig, "Synthetic Multichannel Radiation Stream & Background Tracking")


def plot_anomaly_detection_timeline(
    df: pd.DataFrame,
    alarm_threshold: float,
    gate_threshold: float | None = None,
    max_points: int = 4000,
) -> go.Figure:
    """Plot Mahalanobis anomaly score, alarm persistence, and thresholds."""
    if df.empty or "score" not in df.columns:
        fig = go.Figure()
        return _apply_dark_theme(fig, "No score data available")

    n_total = len(df)
    if n_total > max_points:
        step = int(np.ceil(n_total / max_points))
        plot_df = df.iloc[::step].copy()
    else:
        plot_df = df

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.75, 0.25],
        subplot_titles=(
            "Pre-Update Mahalanobis Innovation Score s_t & Detection Thresholds",
            "Detection Alarm State (Two-Sample Persistence Requirement)",
        ),
    )

    # 1. Mahalanobis Score Trace
    valid_scores = plot_df["score"].replace([np.inf, -np.inf], np.nan)
    fig.add_trace(
        go.Scatter(
            x=plot_df["t_obs"],
            y=valid_scores,
            mode="lines",
            line={"color": COLOR_CYAN, "width": 1.4},
            name="Mahalanobis Score s_t",
        ),
        row=1,
        col=1,
    )

    # Alarm Threshold Line
    t_min = plot_df["t_obs"].iloc[0]
    t_max = plot_df["t_obs"].iloc[-1]
    fig.add_trace(
        go.Scatter(
            x=[t_min, t_max],
            y=[alarm_threshold, alarm_threshold],
            mode="lines",
            line={"color": COLOR_CRIMSON, "width": 1.8, "dash": "dash"},
            name=f"Alarm Threshold (τ = {alarm_threshold:.2f})",
        ),
        row=1,
        col=1,
    )

    # Gate Innovation Threshold Line
    if gate_threshold is not None:
        fig.add_trace(
            go.Scatter(
                x=[t_min, t_max],
                y=[gate_threshold, gate_threshold],
                mode="lines",
                line={"color": COLOR_AMBER, "width": 1.5, "dash": "dot"},
                name=f"Gate Threshold (τ_gate = {gate_threshold:.2f})",
            ),
            row=1,
            col=1,
        )

    # Highlight Alarm Points
    alarm_df = plot_df[plot_df["alarm"] == 1]
    if not alarm_df.empty:
        fig.add_trace(
            go.Scatter(
                x=alarm_df["t_obs"],
                y=alarm_df["score"],
                mode="markers",
                marker={"color": COLOR_CRIMSON, "size": 6, "symbol": "circle"},
                name="Confirmed Alarm",
            ),
            row=1,
            col=1,
        )

    # 2. Binary Alarm State & Ground Truth Source
    fig.add_trace(
        go.Scatter(
            x=plot_df["t_obs"],
            y=plot_df["alarm"],
            mode="lines",
            line={"color": COLOR_CRIMSON, "width": 1.6, "shape": "hv"},
            name="Alarm Output a_t ∈ {0, 1}",
        ),
        row=2,
        col=1,
    )

    if "y_true" in plot_df.columns:
        fig.add_trace(
            go.Scatter(
                x=plot_df["t_obs"],
                y=plot_df["y_true"],
                mode="lines",
                line={"color": COLOR_EMERALD, "width": 1.2, "dash": "dot", "shape": "hv"},
                name="Ground Truth Source y_t",
            ),
            row=2,
            col=1,
        )

    fig.update_xaxes(title_text="Observation Time t (seconds)", row=2, col=1)
    fig.update_yaxes(title_text="Mahalanobis Distance", type="log", row=1, col=1)
    fig.update_yaxes(title_text="State {0, 1}", range=[-0.1, 1.2], row=2, col=1)
    return _apply_dark_theme(fig, "Anomaly Detection & Sequential Persistence Diagnostics")


def plot_telemetry_gate_diagnostics(
    df: pd.DataFrame,
    max_points: int = 4000,
) -> go.Figure:
    """Visualize 4-stage telemetry integrity gate decisions and failure modes."""
    if df.empty or "gate_passed" not in df.columns:
        fig = go.Figure()
        return _apply_dark_theme(fig, "No gate telemetry available")

    n_total = len(df)
    if n_total > max_points:
        step = int(np.ceil(n_total / max_points))
        plot_df = df.iloc[::step].copy()
    else:
        plot_df = df

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            "Telemetry Gate Integrity Decision (Passed vs Rejected)",
            "Sub-Test Breakdown: Validity, Timing, Poisson Fano, Innovation",
        ),
    )

    # 1. Gate Passed / Rejected timeline
    passed_df = plot_df[plot_df["gate_passed"]]
    rejected_df = plot_df[~plot_df["gate_passed"]]

    if not passed_df.empty:
        fig.add_trace(
            go.Scatter(
                x=passed_df["t_obs"],
                y=np.ones(len(passed_df)),
                mode="markers",
                marker={"color": COLOR_EMERALD, "size": 4, "symbol": "circle"},
                name="Gate Passed (Eligible)",
            ),
            row=1,
            col=1,
        )

    if not rejected_df.empty:
        fig.add_trace(
            go.Scatter(
                x=rejected_df["t_obs"],
                y=np.zeros(len(rejected_df)),
                mode="markers",
                marker={"color": COLOR_CRIMSON, "size": 5, "symbol": "x"},
                name="Gate Rejected (Frozen)",
            ),
            row=1,
            col=1,
        )

    # Sustained Override Indicator
    if "gate_override" in plot_df.columns:
        override_df = plot_df[plot_df["gate_override"]]
        if not override_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=override_df["t_obs"],
                    y=np.full(len(override_df), 0.5),
                    mode="markers",
                    marker={"color": COLOR_AMBER, "size": 6, "symbol": "diamond"},
                    name="Sustained Drift Override (90-step)",
                ),
                row=1,
                col=1,
            )

    # 2. Sub-Test Statuses
    flags = [("valid", "Data Validity", "#38bdf8"),
             ("timing", "Sample Timing", "#818cf8"),
             ("poisson", "Poisson Fano Factor", "#fbbf24"),
             ("innovation", "Innovation Check", "#f43f5e")]

    for i, (col_name, label, color) in enumerate(flags):
        if col_name in plot_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=plot_df["t_obs"],
                    y=plot_df[col_name].astype(int) * 0.2 + (i * 0.25),
                    mode="lines",
                    line={"color": color, "width": 1.4, "shape": "hv"},
                    name=label,
                ),
                row=2,
                col=1,
            )

    fig.update_xaxes(title_text="Observation Time t (seconds)", row=2, col=1)
    fig.update_yaxes(title_text="Status", tickvals=[0, 0.5, 1], ticktext=["Reject", "Override", "Pass"], row=1, col=1)
    fig.update_yaxes(title_text="Sub-Test Flags", showticklabels=False, row=2, col=1)
    return _apply_dark_theme(fig, "Telemetry Integrity Gate Verification & Override Timeline")


def plot_daes_controller_dynamics(
    df: pd.DataFrame,
    max_points: int = 4000,
) -> go.Figure:
    """Plot DAES maximum eigenvalue estimation and adaptive learning rate."""
    if df.empty or "eta" not in df.columns:
        fig = go.Figure()
        return _apply_dark_theme(fig, "No DAES controller dynamics available")

    n_total = len(df)
    if n_total > max_points:
        step = int(np.ceil(n_total / max_points))
        plot_df = df.iloc[::step].copy()
    else:
        plot_df = df

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            "Estimated Curvature Maximum Eigenvalue λ_hat via Power Iteration",
            "Adaptive Learning Rate η_t = γ / ((1 + s) λ_hat) & State Freezing",
        ),
    )

    # 1. Lambda hat trace
    valid_lambda = plot_df["lambda_hat"].replace([np.inf, -np.inf], np.nan)
    fig.add_trace(
        go.Scatter(
            x=plot_df["t_obs"],
            y=valid_lambda,
            mode="lines",
            line={"color": COLOR_PURPLE, "width": 1.6},
            name="Max Eigenvalue λ_hat",
        ),
        row=1,
        col=1,
    )

    # 2. Adaptive learning rate eta trace
    valid_eta = plot_df["eta"].replace([np.inf, -np.inf], np.nan)
    fig.add_trace(
        go.Scatter(
            x=plot_df["t_obs"],
            y=valid_eta,
            mode="lines",
            line={"color": COLOR_CYAN, "width": 1.8},
            name="Adaptive Step Size η_t",
        ),
        row=2,
        col=1,
    )

    # Mark frozen points (where eligible == False)
    if "eligible" in plot_df.columns:
        frozen_df = plot_df[~plot_df["eligible"]]
        if not frozen_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=frozen_df["t_obs"],
                    y=np.zeros(len(frozen_df)),
                    mode="markers",
                    marker={"color": COLOR_AMBER, "size": 3, "symbol": "line-ns"},
                    name="Controller Frozen",
                ),
                row=2,
                col=1,
            )

    fig.update_xaxes(title_text="Observation Time t (seconds)", row=2, col=1)
    fig.update_yaxes(title_text="Curvature λ_hat", row=1, col=1)
    fig.update_yaxes(title_text="Step Size η_t", row=2, col=1)
    return _apply_dark_theme(fig, "DAES Adaptive Controller Dynamics & Numerical Stability")


def plot_method_comparisons(df_comparison: pd.DataFrame) -> go.Figure:
    """Plot multi-method comparison across F1, FAR/h, and background relative error."""
    if df_comparison.empty:
        fig = go.Figure()
        return _apply_dark_theme(fig, "No comparison data available")

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(
            "Detection F1 Score (↑ higher is better)",
            "False Alarms per Hour (↓ lower is better)",
            "Background Relative Error (↓ lower is better)",
        ),
        horizontal_spacing=0.08,
    )

    # Color code: highlight rad_daes in cyan, baselines in muted blues, oracle in green
    colors = []
    for m in df_comparison["method"]:
        if m == "rad_daes":
            colors.append(COLOR_CYAN)
        elif m == "oracle_gate":
            colors.append(COLOR_EMERALD)
        elif "gated" in m:
            colors.append("#60a5fa")
        else:
            colors.append("#64748b")

    # 1. F1 Score
    fig.add_trace(
        go.Bar(
            x=df_comparison["method"],
            y=df_comparison["f1"],
            marker_color=colors,
            name="F1 Score",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # 2. FAR/h
    fig.add_trace(
        go.Bar(
            x=df_comparison["method"],
            y=df_comparison["far_per_hour"],
            marker_color=colors,
            name="FAR / h",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    # Add target line of 1.0 FAR/h
    fig.add_hline(y=1.0, line_dash="dash", line_color=COLOR_AMBER, row=1, col=2)

    # 3. Background Relative Error
    fig.add_trace(
        go.Bar(
            x=df_comparison["method"],
            y=df_comparison["bg_rel_error"],
            marker_color=colors,
            name="BG Rel Error",
            showlegend=False,
        ),
        row=1,
        col=3,
    )

    fig.update_xaxes(tickangle=-45)
    fig.update_yaxes(title_text="F1 Score", range=[0, 1.05], row=1, col=1)
    fig.update_yaxes(title_text="False Alarms / Hour", row=1, col=2)
    fig.update_yaxes(title_text="Relative L2 Error", row=1, col=3)
    return _apply_dark_theme(fig, "Benchmark Comparison: All 9 Methods on Identical Stream")


def plot_tradeoff_scatter(df_comparison: pd.DataFrame) -> go.Figure:
    """Plot F1 score vs False Alarms per hour trade-off scatter plot."""
    if df_comparison.empty:
        fig = go.Figure()
        return _apply_dark_theme(fig, "No trade-off data available")

    fig = go.Figure()

    for _, row in df_comparison.iterrows():
        m = row["method"]
        if m == "rad_daes":
            color = COLOR_CYAN
            size = 16
            symbol = "star"
        elif m == "oracle_gate":
            color = COLOR_EMERALD
            size = 14
            symbol = "diamond"
        elif "gated" in m:
            color = "#38bdf8"
            size = 12
            symbol = "circle"
        else:
            color = "#94a3b8"
            size = 10
            symbol = "square"

        fig.add_trace(
            go.Scatter(
                x=[row["far_per_hour"]],
                y=[row["f1"]],
                mode="markers+text",
                marker={"color": color, "size": size, "symbol": symbol, "line": {"color": "#ffffff", "width": 1.2}},
                text=[m],
                textposition="top center",
                name=m,
                hoverinfo="text",
                hovertext=(
                    f"<b>{m}</b><br>"
                    f"F1 Score: {row['f1']:.3f}<br>"
                    f"FAR / h: {row['far_per_hour']:.2f}<br>"
                    f"BG Rel Error: {row['bg_rel_error']:.4f}<br>"
                    f"Updates: {row['n_updates']}"
                ),
            )
        )

    # Reference target lines
    fig.add_vline(x=1.0, line_dash="dash", line_color=COLOR_AMBER, annotation_text="Target FAR (1.0/h)", annotation_position="top right")

    fig.update_xaxes(title_text="False Alarms per Hour (FAR/h) [Target ≤ 1.0]", type="log")
    fig.update_yaxes(title_text="Detection F1 Score [Target ≥ 0.90]", range=[0, 1.05])
    return _apply_dark_theme(fig, "Operating Trade-Off: Detection F1 vs False-Alarm Rate")
