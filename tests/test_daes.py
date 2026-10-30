"""Tests for the original DAES scalar step-size controller."""

from __future__ import annotations

import numpy as np
import pytest

from rad_daes.adaptation.daes import DAESController


def test_identity_curvature_has_expected_step() -> None:
    controller = DAESController(gain=1.78, safety=0.10, power_iterations=7)
    controller.reset(np.eye(3))
    step = controller.step(np.eye(3))
    assert step.raw_rayleigh == pytest.approx(1.0)
    assert step.filtered_lambda == pytest.approx(1.0)
    assert step.eta == pytest.approx(1.78 / 1.10)
    assert step.drift_score == 0.0
    assert step.alpha == pytest.approx(0.16)


def test_power_iteration_finds_dominant_eigenvalue() -> None:
    curvature = np.diag([2.0, 1.0, 0.5])
    controller = DAESController(gain=1.0, power_iterations=50, initial_vector=[1.0, 1.0, 1.0])
    controller.reset(curvature)
    step = controller.step(curvature)
    assert step.raw_rayleigh == pytest.approx(2.0, rel=1e-8)
    assert step.eta == pytest.approx(1.0 / (1.10 * 2.0))


def test_upward_rayleigh_is_not_smoothed_down() -> None:
    controller = DAESController(gain=1.0, power_iterations=30, initial_vector=[1.0, 1.0])
    controller.reset(np.eye(2))
    first = controller.step(np.eye(2))
    second = controller.step(np.diag([4.0, 1.0]))
    assert first.filtered_lambda == pytest.approx(1.0)
    assert second.raw_rayleigh == pytest.approx(4.0, rel=1e-8)
    assert second.filtered_lambda >= second.raw_rayleigh - 1e-10
    assert second.eta <= first.eta


def test_drift_score_increases_with_change() -> None:
    controller = DAESController(gain=1.0, power_iterations=30, initial_vector=[1.0, 1.0])
    controller.reset(np.eye(2))
    controller.step(np.eye(2))
    changed = controller.step(np.diag([9.0, 1.0]))
    assert changed.drift_score > 0.0
    assert changed.alpha > 0.16


def test_curvature_validation_and_reset() -> None:
    controller = DAESController(initial_vector=[1.0, 2.0])
    with pytest.raises(ValueError):
        controller.reset(np.eye(3))
    with pytest.raises(ValueError):
        controller.reset(np.array([[1.0, np.nan], [0.0, 1.0]]))
    controller.reset(np.eye(2))
    assert controller.estimate is None
    controller.step(np.eye(2))
    assert controller.estimate == pytest.approx(1.0)
