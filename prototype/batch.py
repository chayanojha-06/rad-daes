"""Rad-DAES: Multi-Method Batch Evaluation Engine.

Provides unified batch evaluation across all 9 research baseline methods:
  - static
  - ewma
  - gated_ewma
  - rls
  - adam
  - daes
  - gated_daes_fixed
  - rad_daes
  - oracle_gate

Strictly reuses `rad_daes.evaluation.runner.run_stream()` to preserve
research integrity and exact algorithmic comparability.
All generated radiation data are synthetic.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

# Ensure repository 'src' is discoverable
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from rad_daes.data.synthetic import Stream, generate_stream
from rad_daes.evaluation.runner import (
    METHODS,
    MethodSpec,
    load_configs,
    make_spec,
    run_stream,
)

# Standard calibrated thresholds from the verified benchmark tuning runs
DEFAULT_THRESHOLDS: dict[str, dict[str, float]] = {
    "static": {"alarm": 27.0754, "gate": 23.7002},
    "ewma": {"alarm": 28.8545, "gate": 24.9798},
    "gated_ewma": {"alarm": 27.7201, "gate": 23.7321},
    "rls": {"alarm": 28.8545, "gate": 24.9798},
    "adam": {"alarm": 27.0729, "gate": 23.7536},
    "daes": {"alarm": 33.6515, "gate": 29.5982},
    "gated_daes_fixed": {"alarm": 206.2735, "gate": 149.8127},
    "rad_daes": {"alarm": 65.0942, "gate": 55.4837},
    "oracle_gate": {"alarm": 28.7972, "gate": 25.2995},
}

# Selected candidate hyperparameter overrides matching the benchmark
DEFAULT_OVERRIDES: dict[str, dict[str, Any]] = {
    "static": {},
    "ewma": {"ewma_alpha": 0.02},
    "gated_ewma": {"ewma_alpha": 0.005},
    "rls": {"rls_forgetting": 0.98},
    "adam": {"adam_learning_rate": 0.02},
    "daes": {"daes_gain": 0.20},
    "gated_daes_fixed": {"daes_gain": 0.50},
    "rad_daes": {"daes_gain": 0.50},
    "oracle_gate": {"daes_gain": 0.05},
}

ALL_METHODS: tuple[str, ...] = (
    "static",
    "ewma",
    "gated_ewma",
    "rls",
    "adam",
    "daes",
    "gated_daes_fixed",
    "rad_daes",
    "oracle_gate",
)


def load_prototype_configs() -> tuple[dict, dict, dict]:
    """Load data, model, and experiment YAML configuration files."""
    return load_configs(_REPO_ROOT)


def evaluate_single_method(
    stream: Stream,
    method_name: str,
    thresholds: dict[str, float] | None = None,
    override: dict[str, Any] | None = None,
    model_cfg: dict | None = None,
    warmup: int = 600,
    collect_trace: bool = False,
) -> dict[str, Any]:
    """Run one method on a stream using the official research runner."""
    if model_cfg is None:
        _, model_cfg, _ = load_prototype_configs()

    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS.get(
            method_name, {"alarm": 25.0, "gate": 20.0}
        )

    if override is None:
        override = DEFAULT_OVERRIDES.get(method_name, {})

    spec = make_spec(method_name, model_cfg, override=override)
    result = run_stream(
        stream=stream,
        spec=spec,
        thresholds=thresholds,
        model_cfg=model_cfg,
        warmup=warmup,
        collect_trace=collect_trace,
    )
    return result


def compare_all_methods(
    stream: Stream,
    methods: Sequence[str] | None = None,
    model_cfg: dict | None = None,
    warmup: int = 600,
) -> pd.DataFrame:
    """Evaluate multiple methods on the same stream and return comparison table."""
    if methods is None:
        methods = ALL_METHODS

    if model_cfg is None:
        _, model_cfg, _ = load_prototype_configs()

    rows: list[dict[str, Any]] = []
    for method_name in methods:
        res = evaluate_single_method(
            stream=stream,
            method_name=method_name,
            model_cfg=model_cfg,
            warmup=warmup,
            collect_trace=False,
        )
        rows.append(
            {
                "method": method_name,
                "f1": res.get("f1", float("nan")),
                "precision": res.get("precision", float("nan")),
                "recall": res.get("recall", float("nan")),
                "far_per_hour": res.get("far_per_hour", float("nan")),
                "bg_rel_error": res.get("bg_rel_error", float("nan")),
                "latency_mean": res.get("latency_mean", float("nan")),
                "recovery_mean": res.get("recovery_mean", float("nan")),
                "update_stability": res.get("update_stability", float("nan")),
                "n_updates": res.get("n_updates", 0),
                "runtime_seconds": res.get("runtime_seconds", float("nan")),
                "gate_pass_rate": res.get("gate_pass_all", float("nan")),
            }
        )

    df = pd.DataFrame(rows)
    # Order methods consistently
    method_order = {name: i for i, name in enumerate(methods)}
    df["sort_key"] = df["method"].map(method_order)
    df = df.sort_values("sort_key").drop(columns=["sort_key"]).reset_index(drop=True)
    return df
