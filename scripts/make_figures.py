"""Generate figures from saved benchmark tables and traces."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rad_daes.evaluation.runner import METHODS  # noqa: E402

DISPLAY = {
    "static": "Static",
    "ewma": "EWMA",
    "gated_ewma": "Gated EWMA",
    "rls": "RLS",
    "adam": "Adam",
    "daes": "DAES",
    "gated_daes_fixed": "Gated DAES\n(fixed)",
    "rad_daes": "Rad-DAES",
    "oracle_gate": "Oracle gate",
}
COLORS = {
    "static": "#9e9e9e",
    "ewma": "#1f77b4",
    "gated_ewma": "#17becf",
    "rls": "#2ca02c",
    "adam": "#9467bd",
    "daes": "#ff7f0e",
    "gated_daes_fixed": "#bcbd22",
    "rad_daes": "#d62728",
    "oracle_gate": "#7f7f7f",
}


def _method_order(values) -> list[str]:
    present = set(values)
    return [m for m in METHODS if m in present]


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def fig_primary(tables: Path, figures: Path) -> None:
    table = pd.read_csv(tables / "primary_results.csv")
    order = _method_order(table["method"])
    table = table.set_index("method").loc[order].reset_index()
    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    x = np.arange(len(table))
    colors = [COLORS.get(m, "#333333") for m in table["method"]]
    ax.bar(x, table["f1"], color=colors, alpha=0.88)
    low = table["f1"] - table["f1_ci_low"]
    high = table["f1_ci_high"] - table["f1"]
    ax.errorbar(x, table["f1"], yerr=[low.fillna(0), high.fillna(0)], fmt="none", ecolor="black", capsize=3, lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY.get(m, m) for m in table["method"]], rotation=25, ha="right")
    ax.set_ylabel("Mean sample-level F1")
    ax.set_title("Held-out detection performance (scenario means, 95% t interval)")
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylim(bottom=0)
    _save(fig, figures / "fig_primary_f1.png")


def fig_heatmap(table: pd.DataFrame, columns: list[str], title: str, ylabel: str, path: Path) -> None:
    order = _method_order(table["method"])
    table = table.set_index("method").loc[order]
    data = table[columns].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(max(6.5, 0.75 * len(columns) + 3.0), 4.0))
    image = ax.imshow(data, aspect="auto", cmap="viridis", vmin=0.0)
    ax.set_xticks(np.arange(len(columns)))
    ax.set_xticklabels(columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(table)))
    ax.set_yticklabels([DISPLAY.get(m, m) for m in table.index])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="white" if value < 0.55 * np.nanmax(data) else "black", fontsize=7)
    fig.colorbar(image, ax=ax, label="Mean F1")
    _save(fig, path)


def fig_far_vs_f1(tables: Path, figures: Path) -> None:
    table = pd.read_csv(tables / "primary_results.csv")
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    for _, row in table.iterrows():
        method = row["method"]
        ax.scatter(
            max(float(row["far_per_hour"]), 1e-3),
            float(row["f1"]),
            s=60,
            color=COLORS.get(method, "#333333"),
            label=DISPLAY.get(method, method).replace("\n", " "),
            edgecolor="black",
            linewidth=0.4,
        )
    ax.set_xscale("symlog", linthresh=1.0)
    ax.set_xlabel("False alarms per hour (symlog)")
    ax.set_ylabel("Mean sample-level F1")
    ax.set_title("Detection/false-alarm trade-off")
    ax.grid(alpha=0.25)
    # The four false-alarm-compliant methods overlap almost exactly near 2-2.5/h and
    # F1 ~ 0.003, so a legend is used instead of point annotations.
    ax.legend(
        fontsize=7,
        ncol=2,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
    )
    _save(fig, figures / "fig_far_vs_f1.png")


def fig_gate(tables: Path, figures: Path) -> None:
    path = tables / "gate_diagnostics.csv"
    if not path.exists():
        return
    table = pd.read_csv(path)
    order = _method_order(table["method"])
    table = table.set_index("method").loc[order].reset_index()
    columns = [c for c in ("gate_reject_clean", "gate_reject_source", "gate_reject_fault", "gate_override_rate") if c in table.columns]
    labels = ["clean", "source", "fault", "override (approx. 0)"]
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    width = 0.8 / max(len(columns), 1)
    x = np.arange(len(table))
    for k, column in enumerate(columns):
        ax.bar(x + (k - (len(columns) - 1) / 2) * width, table[column], width=width, label=labels[k])
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY.get(m, m) for m in table["method"]], rotation=25, ha="right")
    ax.set_ylabel("Fraction of samples")
    ax.set_title("Gate rejection and sustained-exceedance release")
    ax.legend(ncol=4, fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    _save(fig, figures / "fig_gate.png")


def fig_trace(raw: Path, figures: Path) -> None:
    path = raw / "traces" / "rad_daes_abrupt_mixed_10_seed11.npz"
    if not path.exists():
        return
    threshold = 40.0
    index_path = raw / "trace_index.csv"
    if index_path.exists():
        index = pd.read_csv(index_path)
        row = index[(index["method"] == "rad_daes") & (index["regime"] == "abrupt")]
        if not row.empty:
            threshold = float(row.iloc[0]["threshold_alarm"])
    with np.load(path, allow_pickle=True) as data:
        score = np.asarray(data["score"], dtype=float)
        gate = np.asarray(data["gate"], dtype=bool)
        alarm = np.asarray(data["alarm"], dtype=int)
        mu = np.asarray(data["mu"], dtype=float)
        mu_true = np.asarray(data["mu_true"], dtype=float)
        y = np.asarray(data["y"], dtype=int)
        fault = np.asarray(data["fault"]).astype(str)
        flags = np.asarray(data["gate_flags"], dtype=bool)
        drift = json.loads(str(data["drift_events"]))
        t = np.arange(len(score))

    # Reconstruct the gate's consecutive-innovation-exceedance counter from the saved
    # per-condition flags. This is the quantity compared with the 90-sample patience.
    valid, timing, poisson, innovation = (flags[:, j] for j in range(4))
    exceedance_run = np.zeros(len(score), dtype=int)
    count = 0
    for i in range(len(score)):
        if not (valid[i] and timing[i] and poisson[i]) or innovation[i]:
            count = 0
        else:
            count += 1
        exceedance_run[i] = count

    start, end = 6800, min(8200, len(t))
    sl = slice(start, end)
    fig, axes = plt.subplots(3, 1, figsize=(8.4, 7.2), sharex=True)
    ax = axes[0]
    ax.plot(t[sl], score[sl], color="#1f77b4", lw=0.8, label="Mahalanobis score")
    ax.set_yscale("log")
    ax.axhline(threshold, color="black", ls="--", lw=0.8, label="alarm threshold")
    ax.scatter(t[sl][alarm[sl] == 1], score[sl][alarm[sl] == 1], s=4, color="#d62728", label="alarm")
    ax.scatter(t[sl][~gate[sl]], score[sl][~gate[sl]], s=4, color="#ff7f0e", alpha=0.4, label="gate rejected")
    for event in drift:
        ax.axvline(event["time"], color="grey", ls=":", lw=0.8)
    ax.set_ylabel("Score")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.2)
    ax.set_title("Representative abrupt-drift trace (synthetic, seed 11, mixed_10)")

    ax = axes[1]
    ax.plot(t[sl], exceedance_run[sl], color="#9467bd", lw=1.0, label="consecutive exceedances")
    ax.axhline(90, color="black", ls="--", lw=0.8, label="release patience (90 samples)")
    ax.set_ylabel("Samples")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.2)
    ax.set_title("Innovation-exceedance run versus sustained-release patience")

    ax = axes[2]
    ax.plot(t[sl], mu_true[sl, 0], color="black", lw=1.2, label="true W1 background")
    ax.plot(t[sl], mu[sl, 0], color="#d62728", lw=1.0, label="tracked W1 mean")
    ax.fill_between(t[sl], 0, 1, where=y[sl] == 1, transform=ax.get_xaxis_transform(), color="#2ca02c", alpha=0.15, label="source label")
    fault_mask = fault[sl] != "none"
    ax.fill_between(t[sl], 0, 1, where=fault_mask, transform=ax.get_xaxis_transform(), color="#9467bd", alpha=0.12, label="fault label")
    ax.set_ylabel("Counts / s")
    ax.set_xlabel("Time (s)")
    ax.legend(fontsize=7, ncol=4)
    ax.grid(alpha=0.2)
    _save(fig, figures / "fig_trace.png")


def fig_recovery(tables: Path, figures: Path) -> None:
    path = tables / "primary_results.csv"
    table = pd.read_csv(path)
    if "recovery_mean" not in table.columns or table["recovery_mean"].notna().sum() == 0:
        return
    order = _method_order(table["method"])
    table = table.set_index("method").loc[order].reset_index()
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    values = table["recovery_mean"].fillna(0.0)
    colors = [COLORS.get(m, "#333333") for m in table["method"]]
    ax.bar(np.arange(len(table)), values, color=colors, alpha=0.85)
    ax.set_xticks(np.arange(len(table)))
    ax.set_xticklabels([DISPLAY.get(m, m) for m in table["method"]], rotation=25, ha="right")
    ax.set_ylabel("Mean recovery time (s)")
    ax.set_title("Recovery after abrupt steps (NaN shown as zero)")
    ax.grid(axis="y", alpha=0.25)
    _save(fig, figures / "fig_recovery.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=ROOT / "results" / "raw")
    parser.add_argument("--tables", type=Path, default=ROOT / "results" / "tables")
    parser.add_argument("--figures", type=Path, default=ROOT / "results" / "figures")
    args = parser.parse_args()
    raw, tables, figures = args.raw.resolve(), args.tables.resolve(), args.figures.resolve()
    figures.mkdir(parents=True, exist_ok=True)
    fig_primary(tables, figures)
    by_regime = pd.read_csv(tables / "by_regime.csv")
    fig_heatmap(
        by_regime,
        [c for c in by_regime.columns if c != "method"],
        "Mean F1 by background regime",
        "Method",
        figures / "fig_f1_by_regime.png",
    )
    by_fault = pd.read_csv(tables / "by_fault_condition.csv")
    fig_heatmap(
        by_fault,
        [c for c in by_fault.columns if c != "method"],
        "Mean F1 by telemetry-fault condition",
        "Method",
        figures / "fig_f1_by_fault.png",
    )
    fig_far_vs_f1(tables, figures)
    fig_gate(tables, figures)
    fig_trace(raw, figures)
    fig_recovery(tables, figures)
    print(f"wrote figures to {figures}")


if __name__ == "__main__":
    main()
