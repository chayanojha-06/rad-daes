"""Run development tuning, threshold calibration, and the held-out benchmark.

The script is deliberately explicit about its phases:

1. ``calibrate``: per-method, per-candidate alarm and gate thresholds on clean stable
   development streams.
2. ``tune``: development scenarios only (three regimes, mixed_10 faults).
3. ``test``: the prespecified 4 x 8 x 5 held-out matrix.

All raw run records are written as CSV. No result is typed into the manuscript by hand.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rad_daes.data.synthetic import generate_stream  # noqa: E402
from rad_daes.evaluation.runner import (  # noqa: E402
    METHODS,
    calibrate_method,
    environment_manifest,
    load_configs,
    make_spec,
    run_stream,
    select_candidate,
)

PHASES = ("calibrate", "tune", "test")


@lru_cache(maxsize=None)
def _configs(root: str) -> tuple[dict, dict, dict]:
    return load_configs(root)


def _candidate_id(candidate: dict[str, Any]) -> str:
    if not candidate:
        return "default"
    return ";".join(f"{key}={candidate[key]}" for key in sorted(candidate))


def _candidate_grid(method: str, exp_cfg: dict) -> list[dict[str, Any]]:
    grid = exp_cfg["tuning_grid"]
    alphas = list(grid["ewma_alpha"])
    gains = list(grid["daes_gain"])
    forgettings = list(grid["rls_forgetting"])
    learning_rates = list(grid["adam_learning_rate"])
    gate_quantiles = list(grid["gate_innovation_quantile"])

    if method == "static":
        return [{}]
    if method == "ewma":
        return [{"ewma_alpha": a} for a in alphas]
    if method == "gated_ewma":
        return [
            {"ewma_alpha": a, "gate_quantile": q}
            for a in alphas
            for q in gate_quantiles
        ]
    if method == "rls":
        return [{"rls_forgetting": f} for f in forgettings]
    if method == "adam":
        return [{"adam_learning_rate": lr} for lr in learning_rates]
    if method == "daes":
        return [{"daes_gain": g} for g in gains]
    if method == "gated_daes_fixed":
        return [
            {"daes_gain": g, "gate_quantile": q}
            for g in gains
            for q in gate_quantiles
        ]
    if method == "rad_daes":
        return [
            {"daes_gain": g, "gate_quantile": q}
            for g in gains
            for q in gate_quantiles
        ]
    if method == "oracle_gate":
        return [{"daes_gain": g} for g in gains]
    raise ValueError(f"unknown method: {method}")


def _thresholds_for_candidate(
    root: str,
    method: str,
    candidate: dict[str, Any],
    seeds: list[int],
    duration: int,
    warmup: int,
) -> dict[str, Any]:
    data_cfg, model_cfg, _ = _configs(root)
    spec = make_spec(method, model_cfg, candidate)
    gate_quantile = float(
        candidate.get("gate_quantile", model_cfg["gate"].get("innovation_quantile", 0.999))
    )
    target_far = float(model_cfg["detector"].get("target_far_per_hour", 1.0))
    thresholds = calibrate_method(
        spec,
        data_cfg,
        model_cfg,
        seeds=seeds,
        gate_quantile=gate_quantile,
        target_far_per_hour=target_far,
        duration_seconds=duration,
        warmup=warmup,
    )
    return {
        "method": method,
        "candidate_id": _candidate_id(candidate),
        "candidate_json": json.dumps(candidate, sort_keys=True),
        "gate_quantile": gate_quantile,
        "alarm": thresholds["alarm"],
        "gate": thresholds["gate"],
        "n_calibration_seeds": len(seeds),
        "calibration_duration_seconds": duration,
        "calibration_warmup_seconds": warmup,
    }


def _run_one(
    root: str,
    phase: str,
    method: str,
    candidate: dict[str, Any],
    thresholds: dict[str, float],
    seed: int,
    regime: str,
    fault_condition: str,
    duration: int,
    warmup: int,
) -> dict[str, Any]:
    _, model_cfg, _ = _configs(root)
    stream = generate_stream(
        seed=seed,
        regime=regime,
        fault_condition=fault_condition,
        duration_seconds=duration,
        include_sources=fault_condition != "calibration_clean",
    )
    spec = make_spec(method, model_cfg, candidate)
    result = run_stream(
        stream,
        spec,
        thresholds={"alarm": float(thresholds["alarm"]), "gate": float(thresholds["gate"])},
        model_cfg=model_cfg,
        warmup=warmup,
        collect_trace=False,
    )
    result.update(
        {
            "phase": phase,
            "method": method,
            "candidate_id": _candidate_id(candidate),
            "candidate_json": json.dumps(candidate, sort_keys=True),
            "threshold_alarm": float(thresholds["alarm"]),
            "threshold_gate": float(thresholds["gate"]),
            "duration_seconds": duration,
            "warmup_seconds": warmup,
            "run_id": f"{phase}|{method}|{_candidate_id(candidate)}|{regime}|{fault_condition}|{seed}",
        }
    )
    return result


def _trace_one(
    root: str,
    method: str,
    candidate: dict[str, Any],
    thresholds: dict[str, float],
    seed: int,
    regime: str,
    fault_condition: str,
    duration: int,
    warmup: int,
    trace_dir: Path,
) -> dict[str, Any]:
    _, model_cfg, _ = _configs(root)
    stream = generate_stream(
        seed=seed,
        regime=regime,
        fault_condition=fault_condition,
        duration_seconds=duration,
    )
    spec = make_spec(method, model_cfg, candidate)
    result = run_stream(
        stream,
        spec,
        thresholds={"alarm": float(thresholds["alarm"]), "gate": float(thresholds["gate"])},
        model_cfg=model_cfg,
        warmup=warmup,
        collect_trace=True,
    )
    trace = result.pop("_trace")
    trace_dir.mkdir(parents=True, exist_ok=True)
    path = trace_dir / f"{method}_{regime}_{fault_condition}_seed{seed}.npz"
    np.savez_compressed(
        path,
        x=stream.x,
        t_obs=stream.t_obs,
        mu_true=stream.mu_true,
        source=stream.source,
        y=stream.y,
        fault=stream.fault.astype(str),
        event_id=stream.event_id,
        drift_events=np.array(json.dumps(stream.drift_events)),
        **trace,
    )
    result.update(
        {
            "method": method,
            "seed": seed,
            "regime": regime,
            "fault_condition": fault_condition,
            "trace_path": str(path.relative_to(ROOT)),
            "threshold_alarm": float(thresholds["alarm"]),
            "threshold_gate": float(thresholds["gate"]),
        }
    )
    return result


def _parallel_map(executor: ProcessPoolExecutor, func, tasks: Iterable[dict], label: str) -> list:
    futures = [executor.submit(func, **task) for task in tasks]
    results = []
    started = time.perf_counter()
    for i, future in enumerate(as_completed(futures), start=1):
        results.append(future.result())
        if i % 50 == 0 or i == len(futures):
            elapsed = time.perf_counter() - started
            print(f"  {label}: {i}/{len(futures)} complete ({elapsed:.1f}s)", flush=True)
    return results


def _write_csv(rows: list[dict[str, Any]], path: Path) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


def _selection_table(
    dev: pd.DataFrame,
    methods: list[str],
    exp_cfg: dict,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    selected: dict[str, dict[str, Any]] = {}
    for method in methods:
        group = dev[dev["method"] == method].copy()
        if group.empty:
            raise RuntimeError(f"no development rows for {method}")
        group = group.rename(columns={"candidate_id": "candidate"})
        choice = select_candidate(group)
        candidate_id = str(choice["candidate"])
        candidate = json.loads(
            group.loc[group["candidate"] == candidate_id, "candidate_json"].iloc[0]
        )
        selected[method] = candidate
        for candidate_value, candidate_group in group.groupby("candidate", sort=False):
            summary = {
                "method": method,
                "candidate_id": str(candidate_value),
                "candidate_json": candidate_group["candidate_json"].iloc[0],
                "mean_f1": float(candidate_group["f1"].mean()),
                "mean_bg_rel_error": float(candidate_group["bg_rel_error"].mean()),
                "mean_far_per_hour": float(candidate_group["far_per_hour"].mean()),
                "mean_recall": float(candidate_group["recall"].mean()),
                "mean_precision": float(candidate_group["precision"].mean()),
                "n_runs": int(len(candidate_group)),
                "selected": str(candidate_value) == candidate_id,
            }
            summary["feasible"] = bool(
                np.isfinite(summary["mean_bg_rel_error"])
                and np.isfinite(summary["mean_far_per_hour"])
                and summary["mean_bg_rel_error"] <= 0.10
                and summary["mean_far_per_hour"] <= 2.0
                and np.isfinite(summary["mean_f1"])
            )
            summary["selection_status"] = (
                "fixed"
                if method == "static"
                else ("feasible" if summary["feasible"] else "fallback")
            )
            rows.append(summary)
    table = pd.DataFrame(rows)
    return table, selected


def _manifest(root: Path, args: argparse.Namespace, phases: list[str]) -> dict[str, Any]:
    manifest = environment_manifest()
    manifest.update(
        {
            "package": "rad-daes",
            "package_root": str(root),
            "command": " ".join(sys.argv),
            "phases": phases,
            "synthetic_only": True,
            "real_data_used": False,
            "protocol_date": "2026-10-28",
            "duration_seconds": args.duration,
            "warmup_seconds": args.warmup,
            "workers": args.workers,
            "seed_dev": args.dev_seeds,
            "seed_test": args.test_seeds,
            "config_sha256": {},
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
    )
    for name in ("data.yaml", "model.yaml", "experiments.yaml"):
        path = root / "configs" / name
        manifest["config_sha256"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    parser.add_argument("--workers", type=int, default=max(1, min(10, (os.cpu_count() or 2) - 2)))
    parser.add_argument("--duration", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--dev-seeds", type=int, nargs="*", default=None)
    parser.add_argument("--test-seeds", type=int, nargs="*", default=None)
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--skip-tuning", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    root = args.root.resolve()
    data_cfg, model_cfg, exp_cfg = _configs(str(root))
    args.duration = int(args.duration or exp_cfg["duration_seconds"])
    args.warmup = int(args.warmup or exp_cfg["warmup_seconds"])
    if args.duration <= args.warmup:
        raise SystemExit("duration must exceed warmup")
    if args.mode == "smoke":
        args.dev_seeds = args.dev_seeds or [1001]
        args.test_seeds = args.test_seeds or [11]
        if args.duration == int(exp_cfg["duration_seconds"]):
            args.duration = 1800
        if args.warmup >= args.duration:
            args.warmup = min(300, args.duration // 4)
    else:
        args.dev_seeds = args.dev_seeds or list(exp_cfg["dev"]["seeds"])
        args.test_seeds = args.test_seeds or list(exp_cfg["test"]["seeds"])
    methods = list(args.methods or exp_cfg["methods"])
    for method in methods:
        if method not in METHODS:
            raise SystemExit(f"unknown method: {method}")

    output = (args.output or (root / "results" / "raw")).resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    print(f"Rad-DAES benchmark: mode={args.mode}, methods={len(methods)}, duration={args.duration}s", flush=True)
    print(f"Output: {output}", flush=True)

    grids = {method: _candidate_grid(method, exp_cfg) for method in methods}
    candidate_rows = [
        {
            "method": method,
            "candidate_id": _candidate_id(candidate),
            "candidate_json": json.dumps(candidate, sort_keys=True),
            "gate_quantile": candidate.get(
                "gate_quantile", model_cfg["gate"].get("innovation_quantile", 0.999)
            ),
        }
        for method in methods
        for candidate in grids[method]
    ]
    _write_csv(candidate_rows, output / "candidates.csv")

    phases = ["calibrate", "tune", "test"] if not args.skip_tuning else ["calibrate", "test"]
    calibration_rows: list[dict[str, Any]] = []
    dev_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        calibration_tasks = [
            {
                "root": str(root),
                "method": row["method"],
                "candidate": json.loads(row["candidate_json"]),
                "seeds": args.dev_seeds,
                "duration": args.duration,
                "warmup": args.warmup,
            }
            for row in candidate_rows
        ]
        calibration_rows = _parallel_map(
            executor, _thresholds_for_candidate, calibration_tasks, "calibration"
        )
        calibration = pd.DataFrame(calibration_rows)
        _write_csv(calibration_rows, output / "calibration.csv")

        thresholds_by_key = {
            (row["method"], row["candidate_id"]): {
                "alarm": row["alarm"],
                "gate": row["gate"],
            }
            for row in calibration_rows
        }

        if not args.skip_tuning:
            dev_tasks = []
            for method in methods:
                for candidate in grids[method]:
                    cid = _candidate_id(candidate)
                    thresholds = thresholds_by_key[(method, cid)]
                    for seed in args.dev_seeds:
                        for regime in exp_cfg["dev"]["regimes"]:
                            dev_tasks.append(
                                {
                                    "root": str(root),
                                    "phase": "tune",
                                    "method": method,
                                    "candidate": candidate,
                                    "thresholds": thresholds,
                                    "seed": int(seed),
                                    "regime": regime,
                                    "fault_condition": exp_cfg["dev"]["fault_condition"],
                                    "duration": args.duration,
                                    "warmup": args.warmup,
                                }
                            )
            dev_rows = _parallel_map(executor, _run_one, dev_tasks, "development")
            _write_csv(dev_rows, output / "tuning_runs.csv")
            dev = pd.DataFrame(dev_rows)
            selection, selected = _selection_table(dev, methods, exp_cfg)
            _write_csv(selection.to_dict("records"), output / "tuning_selection.csv")
        else:
            selection = pd.DataFrame(
                [
                    {
                        "method": method,
                        "candidate_id": "default",
                        "candidate_json": "{}",
                        "selected": True,
                        "selection_status": "default",
                    }
                    for method in methods
                ]
            )
            _write_csv(selection.to_dict("records"), output / "tuning_selection.csv")
            selected = {method: {} for method in methods}

        test_tasks = []
        for method in methods:
            candidate = selected[method]
            cid = _candidate_id(candidate)
            thresholds = thresholds_by_key[(method, cid)]
            for seed in args.test_seeds:
                for regime in exp_cfg["test"]["regimes"]:
                    for fault_condition in exp_cfg["test"]["fault_conditions"]:
                        test_tasks.append(
                            {
                                "root": str(root),
                                "phase": "test",
                                "method": method,
                                "candidate": candidate,
                                "thresholds": thresholds,
                                "seed": int(seed),
                                "regime": regime,
                                "fault_condition": fault_condition,
                                "duration": args.duration,
                                "warmup": args.warmup,
                            }
                        )
        test_rows = _parallel_map(executor, _run_one, test_tasks, "test")
        _write_csv(test_rows, output / "test_runs.csv")

        # Representative traces, one per method family, for figure generation and inspection.
        trace_tasks = []
        for method in [m for m in ("static", "gated_ewma", "rad_daes", "oracle_gate") if m in methods]:
            candidate = selected[method]
            thresholds = thresholds_by_key[(method, _candidate_id(candidate))]
            trace_tasks.append(
                {
                    "root": str(root),
                    "method": method,
                    "candidate": candidate,
                    "thresholds": thresholds,
                    "seed": int(args.test_seeds[0]),
                    "regime": "abrupt",
                    "fault_condition": "mixed_10",
                    "duration": args.duration,
                    "warmup": args.warmup,
                    "trace_dir": output / "traces",
                }
            )
        trace_rows = _parallel_map(executor, _trace_one, trace_tasks, "traces")
        _write_csv(trace_rows, output / "trace_index.csv")

    manifest = _manifest(root, args, phases)
    manifest["elapsed_seconds"] = float(time.perf_counter() - started)
    manifest["n_candidates"] = int(len(candidate_rows))
    manifest["n_calibration_runs"] = int(len(calibration_rows))
    manifest["n_development_runs"] = int(len(dev_rows))
    manifest["n_test_runs"] = int(len(test_rows))
    manifest["methods"] = methods
    manifest["selected_candidates"] = selected
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    index_rows = [
        {
            "phase": "calibrate",
            "methods": len(methods),
            "candidates": len(candidate_rows),
            "runs": len(calibration_rows),
            "path": "calibration.csv",
        },
        {
            "phase": "tune",
            "methods": len(methods),
            "candidates": len(candidate_rows),
            "runs": len(dev_rows),
            "path": "tuning_runs.csv",
        },
        {
            "phase": "test",
            "methods": len(methods),
            "candidates": len(selected),
            "runs": len(test_rows),
            "path": "test_runs.csv",
        },
    ]
    _write_csv(index_rows, output / "run_index.csv")
    print(f"Done in {time.perf_counter() - started:.1f}s. Wrote {output}", flush=True)


if __name__ == "__main__":
    main()
