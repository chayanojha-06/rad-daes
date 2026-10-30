"""Audit the saved benchmark artifacts before the manuscript is finalised.

The audit answers four questions:

1. Are all prespecified runs present, with the expected matrix shape?
2. Do the test rows use the configurations and thresholds selected before test?
3. Can representative rows be reproduced from the saved configuration and seeds?
4. Are every table, figure, and macro used by the manuscript actually generated?

The script only reads saved artifacts, except for the determinism spot-check, which
re-runs a small number of streams from the recorded seeds. It writes
``results/audit_report.json`` and ``results/audit_report.md`` and exits non-zero if
any check fails.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rad_daes.data.synthetic import generate_stream  # noqa: E402
from rad_daes.evaluation.runner import (  # noqa: E402
    load_configs,
    make_spec,
    run_stream,
)

METRIC_TOLERANCE = 1e-9
KEY_METRICS = ("f1", "far_per_hour", "bg_rel_error")


class Audit:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []

    def check(self, name: str, passed: bool, detail: str) -> None:
        self.checks.append({"check": name, "passed": bool(passed), "detail": detail})
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {name}: {detail}", flush=True)

    @property
    def failed(self) -> list[dict[str, Any]]:
        return [c for c in self.checks if not c["passed"]]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _expected_counts(root: Path, exp_cfg: dict, methods: list[str]) -> dict[str, int]:
    """Recompute the prespecified run counts without importing the runner script."""

    grid = exp_cfg["tuning_grid"]
    n_ewma = len(grid["ewma_alpha"])
    n_gain = len(grid["daes_gain"])
    n_rls = len(grid["rls_forgetting"])
    n_adam = len(grid["adam_learning_rate"])
    n_gate = len(grid["gate_innovation_quantile"])
    per_method = {
        "static": 1,
        "ewma": n_ewma,
        "gated_ewma": n_ewma * n_gate,
        "rls": n_rls,
        "adam": n_adam,
        "daes": n_gain,
        "gated_daes_fixed": n_gain * n_gate,
        "rad_daes": n_gain * n_gate,
        "oracle_gate": n_gain,
    }
    n_candidates = sum(per_method[m] for m in methods)
    n_dev = n_candidates * len(exp_cfg["dev"]["seeds"]) * len(exp_cfg["dev"]["regimes"])
    n_test = (
        len(methods)
        * len(exp_cfg["test"]["seeds"])
        * len(exp_cfg["test"]["regimes"])
        * len(exp_cfg["test"]["fault_conditions"])
    )
    return {
        "candidates": n_candidates,
        "calibration": n_candidates,
        "development": n_dev,
        "test": n_test,
    }


def _check_matrix(audit: Audit, test: pd.DataFrame, exp_cfg: dict) -> None:
    methods = list(exp_cfg["methods"])
    regimes = list(exp_cfg["test"]["regimes"])
    faults = list(exp_cfg["test"]["fault_conditions"])
    seeds = [int(s) for s in exp_cfg["test"]["seeds"]]

    audit.check(
        "test_methods_match_config",
        set(test["method"]) == set(methods),
        f"methods={sorted(set(test['method']))}",
    )
    audit.check(
        "test_regimes_match_config",
        set(test["regime"]) == set(regimes),
        f"regimes={sorted(set(test['regime']))}",
    )
    audit.check(
        "test_faults_match_config",
        set(test["fault_condition"]) == set(faults),
        f"faults={sorted(set(test['fault_condition']))}",
    )
    audit.check(
        "test_seeds_match_config",
        set(int(s) for s in test["seed"]) == set(seeds),
        f"seeds={sorted(set(int(s) for s in test['seed']))}",
    )

    duplicates = test.duplicated(subset=["method", "regime", "fault_condition", "seed"]).sum()
    audit.check("test_rows_unique", duplicates == 0, f"{int(duplicates)} duplicate keys")

    sizes = test.groupby(["method", "regime", "fault_condition"]).size()
    complete = bool((sizes == len(seeds)).all())
    audit.check(
        "test_scenario_cells_complete",
        complete,
        f"{len(sizes)} cells, expected {len(methods) * len(regimes) * len(faults)}; "
        f"min={int(sizes.min())}, max={int(sizes.max())}",
    )

    finite = {m: bool(np.isfinite(test[m].to_numpy(dtype=float)).all()) for m in KEY_METRICS}
    audit.check(
        "test_key_metrics_finite",
        all(finite.values()),
        ", ".join(f"{k}={'ok' if v else 'non-finite'}" for k, v in finite.items()),
    )


def _check_selection(
    audit: Audit, root: Path, test: pd.DataFrame, raw: Path, exp_cfg: dict
) -> None:
    selection_path = raw / "tuning_selection.csv"
    calibration_path = raw / "calibration.csv"
    if not selection_path.exists() or not calibration_path.exists():
        audit.check("selection_files_present", False, "tuning_selection.csv or calibration.csv missing")
        return

    selection = pd.read_csv(selection_path)
    selected = selection[selection["selected"].astype(str).str.lower() == "true"]
    audit.check(
        "one_selected_candidate_per_method",
        len(selected) == len(set(exp_cfg["methods"])),
        f"{len(selected)} selected rows for {len(set(exp_cfg['methods']))} methods",
    )

    calibration = pd.read_csv(calibration_path)
    mismatches: list[str] = []
    for method in exp_cfg["methods"]:
        rows = test[test["method"] == method]
        if rows.empty:
            mismatches.append(f"{method}: no test rows")
            continue
        candidates = set(rows["candidate_id"].astype(str))
        if len(candidates) != 1:
            mismatches.append(f"{method}: multiple candidates {sorted(candidates)}")
            continue
        candidate_id = candidates.pop()
        chosen = selected[selected["method"] == method]
        if chosen.empty or str(chosen.iloc[0]["candidate_id"]) != candidate_id:
            mismatches.append(f"{method}: test uses {candidate_id}, selection differs")
            continue
        threshold_rows = calibration[
            (calibration["method"] == method) & (calibration["candidate_id"] == candidate_id)
        ]
        if threshold_rows.empty:
            mismatches.append(f"{method}: no calibration row for {candidate_id}")
            continue
        threshold_row = threshold_rows.iloc[0]
        for column, source in (("threshold_alarm", "alarm"), ("threshold_gate", "gate")):
            if not np.isclose(
                rows[column].to_numpy(dtype=float),
                float(threshold_row[source]),
                rtol=0.0,
                atol=1e-9,
            ).all():
                mismatches.append(f"{method}: {column} differs from calibration")

    audit.check(
        "test_uses_frozen_selection_and_thresholds",
        not mismatches,
        "consistent" if not mismatches else "; ".join(mismatches),
    )


def _check_reproduction(
    audit: Audit, root: Path, test: pd.DataFrame, model_cfg: dict, exp_cfg: dict
) -> None:
    cases = [
        ("rad_daes", "abrupt", "mixed_10"),
        ("gated_ewma", "recurring", "clean"),
    ]
    seed = int(exp_cfg["test"]["seeds"][0])
    problems: list[str] = []
    for method, regime, fault in cases:
        row = test[
            (test["method"] == method)
            & (test["regime"] == regime)
            & (test["fault_condition"] == fault)
            & (test["seed"] == seed)
        ]
        if len(row) != 1:
            problems.append(f"{method}/{regime}/{fault}: expected one row, found {len(row)}")
            continue
        row = row.iloc[0]
        stream = generate_stream(
            seed=seed,
            regime=regime,
            fault_condition=fault,
            duration_seconds=int(row["duration_seconds"]),
        )
        spec = make_spec(method, model_cfg, json.loads(row["candidate_json"]))
        recomputed = run_stream(
            stream,
            spec,
            thresholds={"alarm": float(row["threshold_alarm"]), "gate": float(row["threshold_gate"])},
            model_cfg=model_cfg,
            warmup=int(row["warmup_seconds"]),
            collect_trace=False,
        )
        for metric in KEY_METRICS + ("recall", "precision"):
            saved = float(row[metric])
            again = float(recomputed[metric])
            if not np.isclose(saved, again, rtol=0.0, atol=METRIC_TOLERANCE):
                problems.append(f"{method}/{regime}/{fault}/{metric}: {saved!r} != {again!r}")

    audit.check(
        "saved_rows_reproduce_from_seeds",
        not problems,
        f"{len(cases)} runs recomputed" if not problems else "; ".join(problems),
    )


def _check_manifest(audit: Audit, root: Path, raw: Path, exp_cfg: dict, test: pd.DataFrame) -> None:
    path = raw / "manifest.json"
    if not path.exists():
        audit.check("manifest_present", False, "manifest.json missing")
        return
    manifest = json.loads(path.read_text(encoding="utf-8"))
    audit.check("manifest_present", True, f"elapsed={manifest.get('elapsed_seconds', 'n/a')}s")

    expected = _expected_counts(root, exp_cfg, list(exp_cfg["methods"]))
    observed = {
        "candidates": manifest.get("n_candidates"),
        "calibration": manifest.get("n_calibration_runs"),
        "development": manifest.get("n_development_runs"),
        "test": manifest.get("n_test_runs"),
    }
    audit.check(
        "manifest_run_counts_match_protocol",
        observed == expected,
        f"observed={observed}, expected={expected}",
    )
    audit.check(
        "manifest_test_rows_match_csv",
        int(manifest.get("n_test_runs", -1)) == len(test),
        f"manifest={manifest.get('n_test_runs')}, csv={len(test)}",
    )
    audit.check(
        "manifest_declares_synthetic_only",
        manifest.get("synthetic_only") is True and manifest.get("real_data_used") is False,
        f"synthetic_only={manifest.get('synthetic_only')}, real_data_used={manifest.get('real_data_used')}",
    )

    hashes = manifest.get("config_sha256", {})
    drift = [
        name
        for name, digest in hashes.items()
        if not (root / "configs" / name).exists()
        or _sha256(root / "configs" / name) != digest
    ]
    audit.check(
        "config_hashes_match_run",
        bool(hashes) and not drift,
        "all config hashes match" if hashes and not drift else f"changed: {drift or 'no hashes'}",
    )


def _check_artifacts(audit: Audit, root: Path, raw: Path, tables: Path, figures: Path) -> None:
    trace_index_path = raw / "trace_index.csv"
    if trace_index_path.exists():
        index = pd.read_csv(trace_index_path)
        missing = [p for p in index["trace_path"] if not (root / p).exists()]
        audit.check(
            "trace_files_present",
            not missing,
            f"{len(index)} traces, {len(missing)} missing",
        )
    else:
        audit.check("trace_files_present", False, "trace_index.csv missing")

    required_tables = [
        "table_primary.tex",
        "table_paired.tex",
        "table_tuning.tex",
        "manuscript_numbers.tex",
        "primary_results.csv",
        "by_regime.csv",
        "by_fault_condition.csv",
        "gate_diagnostics.csv",
    ]
    missing_tables = [name for name in required_tables if not (tables / name).exists()]
    audit.check(
        "required_tables_present",
        not missing_tables,
        f"{len(required_tables)} required, missing {missing_tables}" if missing_tables else "all present",
    )

    required_figures = [
        "fig_primary_f1.png",
        "fig_f1_by_regime.png",
        "fig_f1_by_fault.png",
        "fig_far_vs_f1.png",
        "fig_gate.png",
        "fig_trace.png",
        "fig_recovery.png",
    ]
    missing_figures = [name for name in required_figures if not (figures / name).exists()]
    audit.check(
        "required_figures_present",
        not missing_figures,
        f"{len(required_figures)} required, missing {missing_figures}" if missing_figures else "all present",
    )


def _check_manuscript(audit: Audit, root: Path, tables: Path, figures: Path) -> None:
    tex_path = root / "paper" / "main.tex"
    if not tex_path.exists():
        audit.check("manuscript_present", False, "paper/main.tex missing")
        return
    text = tex_path.read_text(encoding="utf-8")
    audit.check("manuscript_present", True, f"{len(text.splitlines())} lines")

    inputs = re.findall(r"\\input\{([^}]+)\}", text)
    missing_inputs = [
        value
        for value in inputs
        if not (root / "paper" / value).resolve().exists()
    ]
    audit.check(
        "manuscript_inputs_exist",
        not missing_inputs,
        f"{len(inputs)} inputs" if not missing_inputs else f"missing {missing_inputs}",
    )

    graphics = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", text)
    missing_graphics = [
        value for value in graphics if not (root / "paper" / value).resolve().exists()
    ]
    audit.check(
        "manuscript_figures_exist",
        not missing_graphics,
        f"{len(graphics)} figures" if not missing_graphics else f"missing {missing_graphics}",
    )

    macro_path = tables / "manuscript_numbers.tex"
    if not macro_path.exists():
        audit.check("manuscript_macros_defined", False, "manuscript_numbers.tex missing")
        return
    defined = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", macro_path.read_text(encoding="utf-8")))
    used = set(
        re.findall(
            r"\\(NScenarios|BestMethod|BestValue|"
            r"(?:Static|EWMA|GatedEWMA|RLS|Adam|DAES|GatedDAESFixed|RadDAES|OracleGate)"
            r"(?:Fone|FarPerHour|BgRelError|Recall|Precision|RecoveryMean)|"
            r"Paired[A-Za-z]+(?:Fone|FarPerHour|BgRelError))",
            text,
        )
    )
    undefined = sorted(used - defined)
    audit.check(
        "manuscript_macros_defined",
        not undefined,
        f"{len(used)} generated macros used" if not undefined else f"undefined {undefined}",
    )


def _write_report(audit: Audit, root: Path) -> None:
    payload = {
        "generated_at_utc": pd.Timestamp.now("UTC").isoformat(),
        "n_checks": len(audit.checks),
        "n_failed": len(audit.failed),
        "checks": audit.checks,
    }
    (root / "results").mkdir(parents=True, exist_ok=True)
    (root / "results" / "audit_report.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    lines = [
        "# Reproducibility audit",
        "",
        f"Generated: {payload['generated_at_utc']}",
        "",
        f"Checks: {payload['n_checks']} total, {payload['n_failed']} failed.",
        "",
        "| Check | Result | Detail |",
        "|---|---|---|",
    ]
    for check in audit.checks:
        detail = str(check["detail"]).replace("|", "\\|")
        lines.append(
            f"| `{check['check']}` | {'PASS' if check['passed'] else 'FAIL'} | {detail} |"
        )
    (root / "results" / "audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--raw", type=Path, default=None)
    parser.add_argument("--tables", type=Path, default=None)
    parser.add_argument("--figures", type=Path, default=None)
    args = parser.parse_args()

    root = args.root.resolve()
    raw = (args.raw or root / "results" / "raw").resolve()
    tables = (args.tables or root / "results" / "tables").resolve()
    figures = (args.figures or root / "results" / "figures").resolve()

    data_cfg, model_cfg, exp_cfg = load_configs(root)
    del data_cfg

    test_path = raw / "test_runs.csv"
    if not test_path.exists():
        raise SystemExit(f"missing {test_path}; run scripts/run_benchmarks.py first")
    test = pd.read_csv(test_path)

    audit = Audit()
    _check_matrix(audit, test, exp_cfg)
    _check_selection(audit, root, test, raw, exp_cfg)
    _check_manifest(audit, root, raw, exp_cfg, test)
    _check_reproduction(audit, root, test, model_cfg, exp_cfg)
    _check_artifacts(audit, root, raw, tables, figures)
    _check_manuscript(audit, root, tables, figures)
    _write_report(audit, root)

    print(
        f"\n{len(audit.checks)} checks, {len(audit.failed)} failed. "
        f"Report: {root / 'results' / 'audit_report.md'}",
        flush=True,
    )
    if audit.failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
