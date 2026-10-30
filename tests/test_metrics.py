"""Tests for evaluation metrics."""

from __future__ import annotations

import numpy as np
import pytest

from rad_daes.evaluation import metrics as M


def test_binary_detection_metrics() -> None:
    y = np.array([0, 0, 1, 1, 1, 0])
    alarm = np.array([0, 1, 1, 0, 1, 0])
    out = M.binary_detection_metrics(y, alarm, start=0)
    assert out["tp"] == 2
    assert out["fp"] == 1
    assert out["fn"] == 1
    assert out["tn"] == 2
    assert out["precision"] == pytest.approx(2 / 3)
    assert out["recall"] == pytest.approx(2 / 3)
    assert out["f1"] == pytest.approx(2 / 3)


def test_false_alarms_per_hour() -> None:
    y = np.array([0] * 3600 + [1] * 10)
    alarm = np.array([0] * 3599 + [1] + [0] * 10)
    assert M.false_alarms_per_hour(y, alarm, start=0) == pytest.approx(1.0)


def test_background_relative_error_and_stability() -> None:
    mu_true = np.array([[10.0, 20.0], [10.0, 20.0], [10.0, 20.0]])
    mu_hist = np.array([[11.0, 18.0], [10.0, 20.0], [9.0, 22.0]])
    mask = np.array([True, True, False])
    value = M.background_relative_error(mu_hist, mu_true, mask)
    assert value == pytest.approx((0.1 + 0.1 + 0.0 + 0.0) / 4.0)
    stability = M.update_stability(mu_hist, np.array([10.0, 20.0]), start=0)
    assert stability >= 0.0


def test_latency_and_recovery() -> None:
    y = np.array([0, 1, 1, 0, 1, 1])
    alarm = np.array([0, 0, 1, 0, 1, 1])
    event_id = np.array([0, 1, 1, 0, 2, 2])
    latencies = M.detection_latencies(y, alarm, event_id)
    assert latencies == [1.0, 0.0]

    mu_true = np.ones((10, 2)) * 10.0
    mu_hist = mu_true.copy()
    mu_hist[3:] = 12.0
    mu_hist[6:] = 10.0
    recoveries = M.recovery_times(
        mu_hist, mu_true, [{"time": 3}], threshold=0.05, hold=2
    )
    assert recoveries == [3.0]


def test_gate_diagnostics_include_override() -> None:
    gate = np.array([False, True, True, False, True])
    flags = np.array(
        [
            [True, True, True, False],
            [True, True, True, True],
            [True, True, True, True],
            [False, True, True, False],
            [True, True, True, True],
        ]
    )
    y = np.array([0, 0, 1, 0, 1])
    fault = np.array(["none", "none", "none", "dropout", "none"])
    override = np.array([True, False, False, False, False])
    out = M.gate_diagnostics(gate, flags, y, fault, start=0, override=override)
    assert out["gate_reject_source"] == pytest.approx(0.0)
    assert out["gate_reject_fault"] == pytest.approx(1.0)
    assert out["gate_override_rate"] == pytest.approx(0.2)


def test_student_t_interval() -> None:
    mean, low, high, n = M.student_t_interval([1.0, 2.0, 3.0])
    assert n == 3
    assert low < mean < high
    assert mean == pytest.approx(2.0)
