"""Tests for the background mean/covariance model."""

from __future__ import annotations

import numpy as np
import pytest

from rad_daes.background.model import BackgroundModel


def test_precision_and_score_on_diagonal_covariance() -> None:
    model = BackgroundModel(mean=np.array([10.0, 20.0]), covariance=np.diag([4.0, 25.0]))
    precision = model.precision(epsilon=0.0)
    assert np.allclose(precision, np.diag([0.25, 0.04]))
    score = model.score(np.array([12.0, 25.0]), epsilon=0.0)
    assert score == pytest.approx(1.0 + 1.0)


def test_score_nan_for_nonfinite_observation() -> None:
    model = BackgroundModel(mean=np.zeros(2), covariance=np.eye(2))
    assert np.isnan(model.score(np.array([1.0, np.nan])))


def test_covariance_update_is_symmetric_and_floored() -> None:
    model = BackgroundModel(mean=np.array([10.0, 20.0]), covariance=np.eye(2))
    model.update_covariance(
        np.array([12.0, 18.0]),
        beta=0.5,
        shrinkage=0.0,
        floor=np.array([5.0, 7.0]),
    )
    assert np.allclose(model.covariance, model.covariance.T)
    assert model.covariance[0, 0] >= 5.0
    assert model.covariance[1, 1] >= 7.0
    assert model.covariance[0, 0] == pytest.approx(5.0)
    assert model.covariance[1, 1] == pytest.approx(7.0)


def test_shrinkage_reduces_off_diagonal_magnitude() -> None:
    model = BackgroundModel(
        mean=np.zeros(2),
        covariance=np.array([[4.0, 3.0], [3.0, 9.0]]),
    )
    model.update_covariance(np.array([10.0, 10.0]), beta=1.0, shrinkage=1.0, floor=1e-6)
    assert model.covariance[0, 1] == pytest.approx(0.0)
    assert np.all(np.diag(model.covariance) > 0.0)


def test_invalid_inputs_raise() -> None:
    model = BackgroundModel(mean=np.zeros(2), covariance=np.eye(2))
    with pytest.raises(ValueError):
        model.update_covariance(np.array([1.0, 2.0]), beta=0.0)
    with pytest.raises(ValueError):
        model.update_covariance(np.array([1.0, 2.0]), shrinkage=1.1)
    with pytest.raises(ValueError):
        model.update_covariance(np.array([1.0, 2.0]), floor=np.array([1.0]))
    with pytest.raises(ValueError):
        model.update_mean(np.array([1.0]))
