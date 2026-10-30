"""Aggregate raw benchmark runs into tables and manuscript macros.

Every number in the manuscript tables is generated here from ``results/raw/*.csv``. The
script does not read or write hand-entered measurements.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rad_daes.evaluation.metrics import student_t_interval  # noqa: E402
from rad_daes.evaluation.runner import METHODS  # noqa: E402

DISPLAY = {
    "static": "Static",
    "ewma": "EWMA",
    "gated_ewma": "Gated EWMA",
    "rls": "RLS",
    "adam": "Adam",
    "daes": "DAES",
    "gated_daes_fixed": "Gated DAES (fixed)",
    "rad_daes": "Rad-DAES",
    "oracle_gate": "Oracle gate",
}
MACRO = {
    "static": "Static",
    "ewma": "EWMA",
    "gated_ewma": "GatedEWMA",
    "rls": "RLS",
    "adam": "Adam",
    "daes": "DAES",
    "gated_daes_fixed": "GatedDAESFixed",
    "rad_daes": "RadDAES",
    "oracle_gate": "OracleGate",
}
METRICS = ("f1", "recall", "precision", "far_per_hour", "bg_rel_error", "latency_mean", "recovery_mean", "runtime_per_sample_ms")


def _fmt(value: float, digits: int = 3) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "---"
    if not math.isfinite(numeric):
        return "---"
    if abs(numeric) >= 1000:
        return f"{numeric:.0f}"
    if abs(numeric) >= 100:
        return f"{numeric:.1f}"
    return f"{numeric:.{digits}f}"


def _tex(value: str) -> str:
    return value.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def _scenario_means(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    return (
        df.groupby(["method", "regime", "fault_condition"], as_index=False)[metric]
        .mean()
        .rename(columns={metric: "scenario_mean"})
    )


def _primary_table(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    scenario_rows = []
    for metric in METRICS:
        if metric not in df.columns:
            continue
        means = _scenario_means(df, metric)
        for method, group in means.groupby("method"):
            values = group["scenario_mean"].to_numpy(dtype=float)
            mean, low, high, n = student_t_interval(values)
            scenario_rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "mean": mean,
                    "ci_low": low,
                    "ci_high": high,
                    "n_scenarios": n,
                }
            )
    long = pd.DataFrame(scenario_rows)
    wide = long.pivot(index="method", columns="metric", values="mean").reset_index()
    wide["n_scenarios"] = wide["method"].map(
        long.groupby("method")["n_scenarios"].max().to_dict()
    )
    for metric in METRICS:
        ci = long[long["metric"] == metric].set_index("method")
        wide[f"{metric}_ci_low"] = wide["method"].map(ci["ci_low"].to_dict())
        wide[f"{metric}_ci_high"] = wide["method"].map(ci["ci_high"].to_dict())
    wide["n_runs"] = wide["method"].map(df.groupby("method").size().to_dict())
    order = {m: i for i, m in enumerate(METHODS)}
    wide = wide.sort_values("method", key=lambda s: s.map(order))
    return wide, long


def _by_factor(df: pd.DataFrame, factor: str) -> pd.DataFrame:
    scenario = _scenario_means(df, "f1")
    pivot = scenario.pivot_table(
        index="method", columns=factor, values="scenario_mean", aggfunc="mean"
    )
    order = {m: i for i, m in enumerate(METHODS)}
    pivot = pivot.reindex(sorted(pivot.index, key=lambda m: order.get(m, 99)))
    return pivot.reset_index()


def _paired(df: pd.DataFrame) -> pd.DataFrame:
    if "rad_daes" not in set(df["method"]):
        return pd.DataFrame()
    keys = ["regime", "fault_condition", "seed"]
    base = df[df["method"] == "rad_daes"][keys + ["f1", "far_per_hour", "bg_rel_error"]].rename(
        columns={"f1": "rad_f1", "far_per_hour": "rad_far", "bg_rel_error": "rad_bg"}
    )
    rows = []
    for method in METHODS:
        if method == "rad_daes":
            continue
        other = df[df["method"] == method][keys + ["f1", "far_per_hour", "bg_rel_error"]].rename(
            columns={"f1": "other_f1", "far_per_hour": "other_far", "bg_rel_error": "other_bg"}
        )
        merged = base.merge(other, on=keys, how="inner")
        if merged.empty:
            continue
        for metric, a, b in (
            ("f1", "rad_f1", "other_f1"),
            ("far_per_hour", "rad_far", "other_far"),
            ("bg_rel_error", "rad_bg", "other_bg"),
        ):
            diff = merged[a] - merged[b]
            mean, low, high, n = student_t_interval(diff)
            rows.append(
                {
                    "comparison": f"Rad-DAES minus {DISPLAY.get(method, method)}",
                    "metric": metric,
                    "mean_difference": mean,
                    "ci_low": low,
                    "ci_high": high,
                    "n_pairs": n,
                }
            )
    return pd.DataFrame(rows)


def _gate_table(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "gate_reject_clean",
        "gate_reject_source",
        "gate_reject_fault",
        "gate_override_rate",
        "gate_override_source",
    ]
    available = [c for c in columns if c in df.columns]
    table = df.groupby("method")[available].mean().reset_index()
    order = {m: i for i, m in enumerate(METHODS)}
    return table.sort_values("method", key=lambda s: s.map(order))


def _discrimination_table(df: pd.DataFrame) -> pd.DataFrame:
    """Threshold-free discrimination and gate behaviour, one row per method.

    ROC/PR AUC are computed per run from the pre-update score, so they do not depend on
    the calibrated alarm threshold. Event detection counts a source event as detected if
    any sample in the event alarms.
    """

    rows = []
    for method, group in df.groupby("method"):
        events = float(group["n_events"].sum())
        detected = float(group["n_detected"].sum())
        alarms = float((group["tp"] + group["fp"]).sum())
        samples = float((group["tp"] + group["fp"] + group["tn"] + group["fn"]).sum())
        rows.append(
            {
                "method": method,
                "roc_auc": float(group["roc_auc"].mean()),
                "pr_auc": float(group["pr_auc"].mean()),
                "event_detection": detected / events if events else float("nan"),
                "alarm_fraction": alarms / samples if samples else float("nan"),
                "gate_reject_clean": float(group["gate_reject_clean"].mean()),
                "gate_reject_source": float(group["gate_reject_source"].mean()),
                "gate_override_rate": float(group["gate_override_rate"].mean()),
                "mean_updates": float(group["n_updates"].mean()),
            }
        )
    table = pd.DataFrame(rows)
    order = {m: i for i, m in enumerate(METHODS)}
    return table.sort_values("method", key=lambda s: s.map(order)).reset_index(drop=True)


def _write_discrimination_tex(table: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{tabular}{lccccccc}",
        r"\toprule",
        r"Method & ROC AUC & PR AUC & Event det. & Alarm frac. & Gate rej.\ clean & Gate rej.\ source & Release rate \\",
        r"\midrule",
    ]
    for _, row in table.iterrows():
        lines.append(
            f"{_tex(DISPLAY.get(row['method'], row['method']))} & "
            f"{_fmt(row['roc_auc'])} & {_fmt(row['pr_auc'])} & "
            f"{_fmt(row['event_detection'])} & {_fmt(row['alarm_fraction'], 4)} & "
            f"{_fmt(row['gate_reject_clean'])} & {_fmt(row['gate_reject_source'])} & "
            f"{_fmt(row['gate_override_rate'], 4)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _tuning_table(raw: Path) -> pd.DataFrame:
    path = raw / "tuning_selection.csv"
    if not path.exists():
        return pd.DataFrame()
    table = pd.read_csv(path)
    if "selected" in table.columns:
        table = table[table["selected"].astype(str).str.lower() == "true"].copy()
    return table


def _write_primary_tex(table: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{tabular}{lccccccc}",
        r"\toprule",
        r"Method & F1 (95\% CI) & FAR/h & BG rel. & Recall & Precision & Recovery (s) & ms/sample \\",
        r"\midrule",
    ]
    for _, row in table.iterrows():
        method = DISPLAY.get(row["method"], row["method"])
        f1 = _fmt(row.get("f1", float("nan")))
        low = _fmt(row.get("f1_ci_low", float("nan")))
        high = _fmt(row.get("f1_ci_high", float("nan")))
        lines.append(
            f"{_tex(method)} & {f1} [{low}, {high}] & {_fmt(row.get('far_per_hour', float('nan')), 2)} & "
            f"{_fmt(row.get('bg_rel_error', float('nan')), 4)} & {_fmt(row.get('recall', float('nan')))} & "
            f"{_fmt(row.get('precision', float('nan')))} & {_fmt(row.get('recovery_mean', float('nan')), 1)} & "
            f"{_fmt(row.get('runtime_per_sample_ms', float('nan')), 3)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_factor_tex(table: pd.DataFrame, factor: str, path: Path) -> None:
    columns = [c for c in table.columns if c != "method"]
    header = "Method & " + " & ".join(_tex(str(c)) for c in columns) + r" \\"
    lines = [
        r"\begin{tabular}{l" + "c" * len(columns) + "}",
        r"\toprule",
        header,
        r"\midrule",
    ]
    for _, row in table.iterrows():
        values = " & ".join(_fmt(row[c]) for c in columns)
        lines.append(f"{_tex(DISPLAY.get(row['method'], row['method']))} & {values} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_paired_tex(table: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Comparison & Metric & Mean difference & 95\% CI & Pairs \\",
        r"\midrule",
    ]
    for _, row in table.iterrows():
        lines.append(
            f"{_tex(str(row['comparison']))} & {_tex(str(row['metric']))} & "
            f"{_fmt(row['mean_difference'])} & [{_fmt(row['ci_low'])}, {_fmt(row['ci_high'])}] & "
            f"{int(row['n_pairs'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_tuning_tex(table: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Method & Selected configuration & Status & Dev F1 & Dev BG & Dev FAR/h \\",
        r"\midrule",
    ]
    for _, row in table.iterrows():
        lines.append(
            f"{_tex(DISPLAY.get(row['method'], str(row['method'])))} & "
            f"\\texttt{{{_tex(str(row.get('candidate_id', 'default')))}}} & "
            f"{_tex(str(row.get('selection_status', '---')))} & "
            f"{_fmt(row.get('mean_f1', float('nan')))} & "
            f"{_fmt(row.get('mean_bg_rel_error', float('nan')), 4)} & "
            f"{_fmt(row.get('mean_far_per_hour', float('nan')), 2)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_macros(
    primary: pd.DataFrame,
    path: Path,
    paired: pd.DataFrame | None = None,
    discrimination: pd.DataFrame | None = None,
) -> None:
    lines = ["% Generated by scripts/aggregate_results.py. Do not edit by hand."]
    metric_names = {
        "f1": "Fone",
        "far_per_hour": "FarPerHour",
        "bg_rel_error": "BgRelError",
        "recall": "Recall",
        "precision": "Precision",
        "recovery_mean": "RecoveryMean",
    }
    for _, row in primary.iterrows():
        name = MACRO.get(row["method"], str(row["method"]))
        for metric, suffix in metric_names.items():
            value = row.get(metric, float("nan"))
            lines.append(f"\\newcommand{{\\{name}{suffix}}}{{{_fmt(value)}}}")
    best = primary.sort_values("f1", ascending=False).iloc[0]
    lines.append(f"\\newcommand{{\\BestMethod}}{{{_tex(DISPLAY.get(best['method'], best['method']))}}}")
    lines.append(f"\\newcommand{{\\BestValue}}{{{_fmt(best['f1'])}}}")
    lines.append(f"\\newcommand{{\\NScenarios}}{{{int(primary['n_scenarios'].max())}}}")
    if discrimination is not None and not discrimination.empty:
        suffixes = {
            "roc_auc": "RocAuc",
            "pr_auc": "PrAuc",
            "event_detection": "EventDetection",
            "alarm_fraction": "AlarmFraction",
            "gate_reject_clean": "GateRejectClean",
            "gate_reject_source": "GateRejectSource",
            "gate_override_rate": "GateOverrideRate",
        }
        for _, row in discrimination.iterrows():
            name = MACRO.get(row["method"], str(row["method"]))
            for column, suffix in suffixes.items():
                lines.append(f"\\newcommand{{\\{name}{suffix}}}{{{_fmt(row[column])}}}")
    if paired is not None and not paired.empty:
        suffix = {"f1": "Fone", "far_per_hour": "FarPerHour", "bg_rel_error": "BgRelError"}
        for _, row in paired.iterrows():
            method = str(row["comparison"]).replace("Rad-DAES minus ", "")
            method_key = next((k for k, v in DISPLAY.items() if v == method), None)
            if method_key is None:
                continue
            metric = str(row["metric"])
            if metric not in suffix:
                continue
            macro_name = f"Paired{MACRO.get(method_key, method_key)}{suffix[metric]}"
            lines.append(
                f"\\newcommand{{\\{macro_name}}}{{{_fmt(row['mean_difference'])}}}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=ROOT / "results" / "raw")
    parser.add_argument("--tables", type=Path, default=ROOT / "results" / "tables")
    args = parser.parse_args()
    raw = args.raw.resolve()
    tables = args.tables.resolve()
    tables.mkdir(parents=True, exist_ok=True)

    test_path = raw / "test_runs.csv"
    if not test_path.exists():
        raise SystemExit(f"missing {test_path}; run scripts/run_benchmarks.py first")
    df = pd.read_csv(test_path)
    if df.empty:
        raise SystemExit("test_runs.csv is empty")

    primary, long = _primary_table(df)
    primary.to_csv(tables / "primary_results.csv", index=False)
    long.to_csv(tables / "primary_long.csv", index=False)
    _write_primary_tex(primary, tables / "table_primary.tex")

    for factor in ("regime", "fault_condition"):
        table = _by_factor(df, factor)
        table.to_csv(tables / f"by_{factor}.csv", index=False)
        _write_factor_tex(table, factor, tables / f"table_by_{factor}.tex")

    paired = _paired(df)
    if not paired.empty:
        paired.to_csv(tables / "paired_differences.csv", index=False)
        _write_paired_tex(paired, tables / "table_paired.tex")

    gate = _gate_table(df)
    gate.to_csv(tables / "gate_diagnostics.csv", index=False)

    discrimination = _discrimination_table(df)
    discrimination.to_csv(tables / "discrimination_results.csv", index=False)
    _write_discrimination_tex(discrimination, tables / "table_discrimination.tex")

    tuning = _tuning_table(raw)
    if not tuning.empty:
        tuning.to_csv(tables / "tuning_selection.csv", index=False)
        _write_tuning_tex(tuning, tables / "table_tuning.tex")

    _write_macros(
        primary,
        tables / "manuscript_numbers.tex",
        paired if not paired.empty else None,
        discrimination,
    )

    summary = {
        "n_runs": int(len(df)),
        "n_methods": int(df["method"].nunique()),
        "n_scenarios": int(df.groupby(["regime", "fault_condition"]).ngroups),
        "n_seeds": int(df["seed"].nunique()),
        "methods": sorted(df["method"].unique().tolist()),
        "best_f1_method": str(primary.sort_values("f1", ascending=False).iloc[0]["method"]),
        "best_f1": float(primary.sort_values("f1", ascending=False).iloc[0]["f1"]),
    }
    (tables / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote tables to {tables}")


if __name__ == "__main__":
    main()
