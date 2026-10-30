"""Tests for the labelled synthetic stream generator."""

from __future__ import annotations

import numpy as np
import pytest

from rad_daes.data.synthetic import FAULT_MODES, generate_stream


def test_deterministic_for_same_seed() -> None:
    a = generate_stream(seed=7, regime="gradual", fault_condition="mixed_10", duration_seconds=2000)
    b = generate_stream(seed=7, regime="gradual", fault_condition="mixed_10", duration_seconds=2000)
    assert np.array_equal(a.x, b.x, equal_nan=True)
    assert np.array_equal(a.t_obs, b.t_obs)
    assert np.array_equal(a.y, b.y)
    assert np.array_equal(a.fault, b.fault)
    assert np.array_equal(a.mu_true, b.mu_true)


def test_different_seeds_change_stream() -> None:
    a = generate_stream(seed=7, regime="stable", fault_condition="clean", duration_seconds=1000)
    b = generate_stream(seed=8, regime="stable", fault_condition="clean", duration_seconds=1000)
    assert not np.array_equal(a.x, b.x)


def test_default_six_hour_stream_has_six_events() -> None:
    stream = generate_stream(seed=11, regime="stable", fault_condition="clean", duration_seconds=21600)
    event_ids = np.unique(stream.event_id)
    assert list(event_ids) == [0, 1, 2, 3, 4, 5, 6]
    assert stream.y.sum() > 0
    assert np.all(stream.source[stream.y == 1].sum(axis=1) > 0.0)
    assert np.all(stream.y[stream.event_id == 0] == 0)


def test_source_free_stream_has_no_labels() -> None:
    stream = generate_stream(
        seed=5,
        regime="stable",
        fault_condition="clean",
        duration_seconds=1000,
        include_sources=False,
    )
    assert stream.y.sum() == 0
    assert np.all(stream.source == 0.0)
    assert np.all(stream.event_id == 0)


@pytest.mark.parametrize("mode", FAULT_MODES)
def test_single_fault_mode_labels_and_effects(mode: str) -> None:
    stream = generate_stream(
        seed=3,
        regime="stable",
        fault_condition=f"{mode}_10",
        duration_seconds=4000,
    )
    affected = stream.fault != "none"
    assert 0.05 < affected.mean() < 0.15
    assert set(np.unique(stream.fault)) <= {"none", mode}
    if mode == "dropout":
        assert np.isnan(stream.x).any()
    if mode == "jitter":
        assert not np.allclose(stream.t_obs, np.arange(len(stream.t_obs)))
    if mode == "stuck":
        assert np.any(np.diff(stream.x, axis=0) == 0.0)


def test_mixed_fault_uses_multiple_modes() -> None:
    stream = generate_stream(
        seed=4,
        regime="stable",
        fault_condition="mixed_15",
        duration_seconds=6000,
    )
    modes = set(np.unique(stream.fault)) - {"none"}
    assert len(modes) >= 2
    assert 0.10 < (stream.fault != "none").mean() < 0.20


def test_invalid_arguments_raise() -> None:
    with pytest.raises(ValueError):
        generate_stream(seed=1, regime="unknown")
    with pytest.raises(ValueError):
        generate_stream(seed=1, fault_condition="unknown")
    with pytest.raises(ValueError):
        generate_stream(seed=1, duration_seconds=99)
    with pytest.raises(ValueError):
        generate_stream(seed=1, nominal_cps=np.ones(5))
