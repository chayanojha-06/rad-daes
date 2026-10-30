"""Matched-stream reproducibility and environment-manifest checks."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from rad_daes.data.synthetic import generate_stream
from rad_daes.evaluation.runner import (
    environment_manifest,
    load_configs,
    make_spec,
    run_stream,
)

ROOT = Path(__file__).resolve().parents[1]


def test_matched_stream_and_spec_reproduce_trace() -> None:
    _, model_cfg, _ = load_configs(ROOT)
    stream = generate_stream(
        seed=11,
        regime="gradual",
        fault_condition="dropout_10",
        duration_seconds=1800,
    )
    spec = make_spec("rad_daes", model_cfg)
    thresholds = {"alarm": 30.0, "gate": 20.0}
    first = run_stream(stream, spec, thresholds, model_cfg, warmup=100, collect_trace=True)
    second = run_stream(stream, spec, thresholds, model_cfg, warmup=100, collect_trace=True)
    for key in ("mu", "score", "alarm", "gate", "gate_flags"):
        assert np.array_equal(first["_trace"][key], second["_trace"][key], equal_nan=True)


def test_stream_generation_is_independent_of_method() -> None:
    # The runner regenerates a stream inside each task; this check makes the matched-stream
    # assumption explicit.
    a = generate_stream(seed=11, regime="recurring", fault_condition="mixed_10", duration_seconds=1800)
    b = generate_stream(seed=11, regime="recurring", fault_condition="mixed_10", duration_seconds=1800)
    assert np.array_equal(a.x, b.x, equal_nan=True)
    assert np.array_equal(a.mu_true, b.mu_true)
    assert np.array_equal(a.y, b.y)


def test_environment_manifest_has_versions() -> None:
    manifest = environment_manifest()
    for key in ("python", "platform", "numpy", "scipy", "pandas", "matplotlib"):
        assert key in manifest
        assert manifest[key]
