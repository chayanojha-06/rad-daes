"""Tests for the benchmark runner and calibration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rad_daes.data.synthetic import generate_stream
from rad_daes.evaluation.runner import (
    METHODS,
    calibrate_method,
    load_configs,
    make_spec,
    run_stream,
    select_candidate,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def configs():
    return load_configs(ROOT)


@pytest.fixture(scope="module")
def short_stream():
    return generate_stream(
        seed=11,
        regime="abrupt",
        fault_condition="mixed_10",
        duration_seconds=2400,
    )


def test_all_method_specs_are_available(configs) -> None:
    _, model_cfg, _ = configs
    for method in METHODS:
        spec = make_spec(method, model_cfg)
        assert spec.name == method
        assert spec.gate_mode in {"none", "gate", "oracle"}
        assert spec.update_kind in {"none", "ewma", "rls", "adam", "daes", "daes_fixed"}


def test_candidate_override_changes_parameter(configs) -> None:
    _, model_cfg, _ = configs
    spec = make_spec("rad_daes", model_cfg, {"daes_gain": 0.2})
    assert spec.params["gain"] == pytest.approx(0.2)
    ewma = make_spec("ewma", model_cfg, {"ewma_alpha": 0.05})
    assert ewma.params["alpha"] == pytest.approx(0.05)


def test_run_stream_returns_expected_fields(configs, short_stream) -> None:
    _, model_cfg, _ = configs
    result = run_stream(
        short_stream,
        make_spec("rad_daes", model_cfg),
        {"alarm": 30.0, "gate": 20.0},
        model_cfg,
        warmup=100,
    )
    for key in (
        "f1",
        "far_per_hour",
        "bg_rel_error",
        "runtime_per_sample_ms",
        "n_updates",
        "gate_reject_source",
        "gate_override_rate",
    ):
        assert key in result
    assert 0.0 <= result["f1"] <= 1.0
    assert result["n_updates"] > 0
    assert result["runtime_per_sample_ms"] > 0.0


def test_run_stream_is_deterministic(configs, short_stream) -> None:
    _, model_cfg, _ = configs
    spec = make_spec("gated_ewma", model_cfg)
    thresholds = {"alarm": 30.0, "gate": 20.0}
    first = run_stream(short_stream, spec, thresholds, model_cfg, warmup=100)
    second = run_stream(short_stream, spec, thresholds, model_cfg, warmup=100)
    for key in ("f1", "far_per_hour", "bg_rel_error", "n_updates"):
        assert first[key] == second[key]


def test_static_never_updates(configs, short_stream) -> None:
    _, model_cfg, _ = configs
    result = run_stream(
        short_stream,
        make_spec("static", model_cfg),
        {"alarm": 30.0, "gate": 20.0},
        model_cfg,
        warmup=100,
    )
    assert result["n_updates"] == 0


def test_oracle_gate_only_updates_clean_samples(configs, short_stream) -> None:
    _, model_cfg, _ = configs
    result = run_stream(
        short_stream,
        make_spec("oracle_gate", model_cfg),
        {"alarm": 30.0, "gate": 20.0},
        model_cfg,
        warmup=100,
    )
    eligible = int(
        np.sum(
            (short_stream.y == 0)
            & (short_stream.fault == "none")
            & np.all(np.isfinite(short_stream.x), axis=1)
        )
    )
    assert result["n_updates"] <= eligible
    assert result["n_updates"] > 0


def test_calibration_returns_positive_thresholds(configs) -> None:
    _, model_cfg, _ = configs
    spec = make_spec("ewma", model_cfg)
    thresholds = calibrate_method(
        spec,
        {},
        model_cfg,
        seeds=[1001, 1002],
        duration_seconds=1200,
        warmup=100,
    )
    assert thresholds["alarm"] > 0.0
    assert thresholds["gate"] >= 10.0
    assert thresholds["alarm"] < thresholds["gate"] * 10.0


def test_select_candidate_prefers_feasible_high_f1() -> None:
    rows = []
    for candidate, f1, bg, far in [
        ("a", 0.20, 0.05, 1.0),
        ("b", 0.30, 0.20, 1.0),
        ("c", 0.25, 0.06, 1.5),
    ]:
        for _ in range(3):
            rows.append(
                {"candidate": candidate, "f1": f1, "bg_rel_error": bg, "far_per_hour": far}
            )
    selected = select_candidate(pd.DataFrame(rows))
    assert selected["candidate"] == "c"


def test_select_candidate_fallback_is_deterministic() -> None:
    rows = []
    for candidate, f1, bg, far in [
        ("a", 0.10, 0.50, 50.0),
        ("b", 0.05, 0.60, 40.0),
    ]:
        for _ in range(2):
            rows.append(
                {"candidate": candidate, "f1": f1, "bg_rel_error": bg, "far_per_hour": far}
            )
    selected = select_candidate(pd.DataFrame(rows))
    assert selected["candidate"] in {"a", "b"}
    assert np.isfinite(selected["mean_f1"])
