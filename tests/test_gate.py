"""Tests for the telemetry-integrity gate."""

from __future__ import annotations

import numpy as np

from rad_daes.telemetry.gate import TelemetryGate


CLEAN_SEQUENCE = np.random.default_rng(42).poisson(5.0, size=(12, 6)).astype(float)
CLEAN_X = CLEAN_SEQUENCE[0]


def _feed_clean(gate: TelemetryGate, n: int = 12, score: float = 1.0) -> list:
    decisions = []
    for t in range(n):
        decisions.append(gate.evaluate(CLEAN_SEQUENCE[t], float(t), score))
    return decisions


def test_clean_observation_passes_after_buffer_fills() -> None:
    gate = TelemetryGate(window=10, innovation_threshold=10.0, drift_patience=5)
    decisions = _feed_clean(gate, n=12)
    assert not decisions[0].poisson
    assert decisions[-1].poisson
    assert decisions[-1].valid
    assert decisions[-1].timing
    assert decisions[-1].innovation
    assert decisions[-1].passed
    assert decisions[-1].sustained_override is False


def test_invalid_observation_fails_and_resets_override() -> None:
    gate = TelemetryGate(window=5, innovation_threshold=1.0, drift_patience=2)
    _feed_clean(gate, n=6)
    gate.evaluate(np.array([np.nan, 1.0, 1.0, 1.0, 1.0, 1.0]), 6.0, 99.0)
    decision = gate.evaluate(np.array([1.0, 2.0, 1.0, 2.0, 1.0, 2.0]), 7.0, 99.0)
    assert decision.valid is False or decision.passed is False
    assert decision.sustained_override is False


def test_timing_rejects_short_and_long_intervals() -> None:
    gate = TelemetryGate(window=3, max_dt_seconds=1.5, min_dt_seconds=0.5)
    x = np.array([1.0, 2.0, 1.0, 2.0, 1.0, 2.0])
    first = gate.evaluate(x, 0.0, 1.0)
    short = gate.evaluate(x, 0.1, 1.0)
    long = gate.evaluate(x, 3.0, 1.0)
    assert first.timing is True
    assert short.timing is False
    assert long.timing is False


def test_fano_rejects_constant_stuck_channel() -> None:
    gate = TelemetryGate(window=5, fano_min=0.25, fano_max=3.5)
    x = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
    decision = None
    for t in range(6):
        decision = gate.evaluate(x, float(t), 1.0)
    assert decision is not None
    assert decision.poisson is False
    assert decision.passed is False


def test_sustained_exceedance_releases_innovation_guard() -> None:
    gate = TelemetryGate(window=5, innovation_threshold=2.0, drift_patience=3)
    decisions = [gate.evaluate(CLEAN_SEQUENCE[t], float(t), 99.0) for t in range(8)]
    assert decisions[-1].innovation is False
    assert decisions[-1].sustained_override is True
    assert decisions[-1].passed is True


def test_override_requires_other_telemetry_conditions() -> None:
    gate = TelemetryGate(window=5, innovation_threshold=2.0, drift_patience=3)
    for t in range(8):
        gate.evaluate(CLEAN_SEQUENCE[t], float(t), 99.0)
    bad_x = CLEAN_SEQUENCE[8].copy()
    bad_x[0] = np.nan
    bad = gate.evaluate(bad_x, 8.0, 99.0)
    assert bad.sustained_override is False
    assert bad.passed is False
