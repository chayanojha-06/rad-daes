"""Rad-DAES: Prototype Integration and Parity Tests.

Validates:
1. Exact mathematical parity between `InteractiveStreamEngine` and `runner.run_stream()`.
2. State freezing of background model and DAES controller on gate rejection.
3. Two-consecutive-sample alarm persistence requirement.
4. Batch comparison of all 9 baseline methods.
5. Tabular export integrity.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
_PROTOTYPE_DIR = _REPO_ROOT / "prototype"
if str(_PROTOTYPE_DIR) not in sys.path:
    sys.path.insert(0, str(_PROTOTYPE_DIR))

from prototype.adapter import InteractiveStreamEngine
from prototype.batch import (
    ALL_METHODS,
    DEFAULT_OVERRIDES,
    DEFAULT_THRESHOLDS,
    compare_all_methods,
    evaluate_single_method,
    load_prototype_configs,
)
from rad_daes.data.synthetic import generate_stream
from rad_daes.evaluation.runner import make_spec, run_stream


@pytest.fixture(scope="module")
def configs():
    return load_prototype_configs()


def test_synthetic_stream_generation():
    """Verify synthetic streams generate expected dimensions and fields."""
    stream = generate_stream(
        seed=101,
        regime="abrupt",
        fault_condition="mixed_10",
        duration_seconds=1200,
    )
    assert stream.x.shape == (1200, 6)
    assert stream.t_obs.shape == (1200,)
    assert stream.mu_true.shape == (1200, 6)
    assert stream.y.shape == (1200,)
    assert stream.fault.shape == (1200,)
    assert stream.regime == "abrupt"
    assert stream.fault_condition == "mixed_10"
    assert "nominal_cps" in stream.metadata


@pytest.mark.parametrize("method_name", ALL_METHODS)
def test_adapter_runner_exact_parity(configs, method_name):
    """Verify sample-by-sample parity between InteractiveStreamEngine and run_stream across all 9 methods."""
    data_cfg, model_cfg, _ = configs
    stream = generate_stream(
        seed=1001,
        regime="gradual",
        fault_condition="mixed_10",
        duration_seconds=1000,
    )
    thresholds = DEFAULT_THRESHOLDS[method_name]
    override = DEFAULT_OVERRIDES[method_name]
    spec = make_spec(method_name, model_cfg, override=override)

    # 1. Run official runner
    runner_res = run_stream(
        stream=stream,
        spec=spec,
        thresholds=thresholds,
        model_cfg=model_cfg,
        warmup=200,
        collect_trace=True,
    )
    runner_trace = runner_res["_trace"]

    # 2. Run interactive adapter sample-by-sample
    engine = InteractiveStreamEngine(
        stream=stream,
        spec=spec,
        thresholds=thresholds,
        model_cfg=model_cfg,
        warmup=200,
    )
    engine.run_all()
    adapter_trace = engine.get_traces()

    # 3. Assert exact equivalence across all traces
    # Mean history
    assert np.allclose(
        adapter_trace["mu"], runner_trace["mu"], atol=1e-5, equal_nan=True
    ), f"{method_name}: mu history mismatch"

    # Score history (handle NaNs identical)
    mask_fin = np.isfinite(runner_trace["score"])
    assert np.all(np.isnan(adapter_trace["score"]) == np.isnan(runner_trace["score"]))
    assert np.allclose(
        adapter_trace["score"][mask_fin],
        runner_trace["score"][mask_fin],
        atol=1e-5,
    ), f"{method_name}: score history mismatch"

    # Alarm history
    assert np.array_equal(
        adapter_trace["alarm"], runner_trace["alarm"]
    ), f"{method_name}: alarm history mismatch"

    # Gate decision history
    assert np.array_equal(
        adapter_trace["gate"], runner_trace["gate"]
    ), f"{method_name}: gate history mismatch"

    # Gate flags history
    assert np.array_equal(
        adapter_trace["gate_flags"], runner_trace["gate_flags"]
    ), f"{method_name}: gate flags mismatch"

    # Gate sustained override history
    assert np.array_equal(
        adapter_trace["gate_override"], runner_trace["gate_override"]
    ), f"{method_name}: gate override mismatch"

    # 4. Compare run metrics
    adapter_metrics = engine.get_run_results()
    for key in ["f1", "precision", "recall", "far_per_hour", "bg_rel_error", "n_updates"]:
        r_val = runner_res[key]
        a_val = adapter_metrics[key]
        if np.isnan(r_val):
            assert np.isnan(a_val), f"Metric {key} NaN mismatch"
        else:
            assert np.isclose(r_val, a_val, atol=1e-5), f"Metric {key} value mismatch: runner={r_val}, adapter={a_val}"


def test_gate_rejection_freezes_adaptive_state(configs):
    """Verify that when the gate rejects an observation, the model and controller freeze."""
    data_cfg, model_cfg, _ = configs
    stream = generate_stream(
        seed=1002,
        regime="abrupt",
        fault_condition="dropout_10",
        duration_seconds=500,
    )
    spec = make_spec("rad_daes", model_cfg, override=DEFAULT_OVERRIDES["rad_daes"])
    engine = InteractiveStreamEngine(
        stream=stream,
        spec=spec,
        thresholds=DEFAULT_THRESHOLDS["rad_daes"],
        model_cfg=model_cfg,
        warmup=100,
    )

    prev_mean = None
    prev_cov = None
    prev_precision = None
    rejections_seen = 0

    while not engine.is_finished:
        t = engine.current_t
        # Pre-step state
        pre_mean = engine.model.mean.copy()
        pre_cov = engine.model.covariance.copy()
        pre_prec = engine.precision.copy()

        snap = engine.step()
        assert snap is not None

        if not snap.eligible:
            rejections_seen += 1
            # Model mean, covariance, and precision must remain identical
            assert np.array_equal(engine.model.mean, pre_mean), f"Mean changed at t={t} despite rejection"
            assert np.array_equal(engine.model.covariance, pre_cov), f"Covariance changed at t={t} despite rejection"
            assert np.array_equal(engine.precision, pre_prec), f"Precision changed at t={t} despite rejection"
            # Controller step must not have been taken on rejected step
            assert snap.daes_step is None

    assert rejections_seen > 0, "Test stream had no gate rejections; dropout_10 should induce rejections"


def test_alarm_persistence_two_consecutive_rule(configs):
    """Verify the detector enforces two consecutive samples exceeding threshold to trigger alarm."""
    data_cfg, model_cfg, _ = configs
    stream = generate_stream(
        seed=1003,
        regime="stable",
        fault_condition="clean",
        duration_seconds=300,
        include_sources=False,
    )
    spec = make_spec("rad_daes", model_cfg, override=DEFAULT_OVERRIDES["rad_daes"])
    engine = InteractiveStreamEngine(
        stream=stream,
        spec=spec,
        thresholds=DEFAULT_THRESHOLDS["rad_daes"],
        model_cfg=model_cfg,
        warmup=50,
    )

    # Artificially inject scores to test persistence logic directly through detector
    det = engine.detector
    thresh = engine.thresholds["alarm"]

    # 1. Under threshold
    assert det.update(thresh - 5.0) == 0
    assert det.update(thresh - 2.0) == 0

    # 2. Single isolated spike above threshold: MUST NOT alarm (persistence=2)
    assert det.update(thresh + 10.0) == 0

    # 3. Drop back under threshold
    assert det.update(thresh - 1.0) == 0

    # 4. First sample above threshold: MUST NOT alarm yet
    assert det.update(thresh + 5.0) == 0

    # 5. Second consecutive sample above threshold: MUST alarm!
    assert det.update(thresh + 6.0) == 1

    # 6. Third consecutive sample: continues alarm
    assert det.update(thresh + 7.0) == 1

    # 7. Drop below: alarm clears immediately
    assert det.update(thresh - 0.5) == 0


def test_compare_all_nine_methods(configs):
    """Verify batch comparison successfully executes all 9 research baselines."""
    data_cfg, model_cfg, _ = configs
    stream = generate_stream(
        seed=11,
        regime="stable",
        fault_condition="clean",
        duration_seconds=800,
        n_source_events=2,
    )

    df = compare_all_methods(stream, model_cfg=model_cfg, warmup=200)

    assert len(df) == 9
    assert list(df["method"]) == list(ALL_METHODS)
    expected_cols = [
        "method",
        "f1",
        "precision",
        "recall",
        "far_per_hour",
        "bg_rel_error",
        "latency_mean",
        "recovery_mean",
        "update_stability",
        "n_updates",
        "runtime_seconds",
        "gate_pass_rate",
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing column {col}"

    # Verify rad_daes updates are positive
    rad_row = df[df["method"] == "rad_daes"].iloc[0]
    assert rad_row["n_updates"] > 0
    # Static method must have zero updates
    static_row = df[df["method"] == "static"].iloc[0]
    assert static_row["n_updates"] == 0


def test_dataframe_trace_export(configs):
    """Verify DataFrame trace export contains all required channels and flags."""
    data_cfg, model_cfg, _ = configs
    stream = generate_stream(
        seed=12,
        regime="gradual",
        fault_condition="mixed_5",
        duration_seconds=400,
    )
    spec = make_spec("rad_daes", model_cfg, override=DEFAULT_OVERRIDES["rad_daes"])
    engine = InteractiveStreamEngine(
        stream=stream,
        spec=spec,
        thresholds=DEFAULT_THRESHOLDS["rad_daes"],
        model_cfg=model_cfg,
        warmup=100,
    )
    engine.step_n(250)

    df = engine.to_dataframe()
    assert len(df) == 250
    # Core columns
    for col in ["t", "t_obs", "score", "alarm", "gate_passed", "eligible", "y_true", "fault"]:
        assert col in df.columns

    # 6 channels counts, estimates, and true
    for ch in range(1, 7):
        assert f"W{ch}_counts" in df.columns
        assert f"W{ch}_mu_est" in df.columns
        assert f"W{ch}_mu_true" in df.columns

    # Verify CSV export
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    assert len(csv_bytes) > 0


def test_scientific_plots_generation(configs):
    """Verify all Plotly scientific figures build cleanly from trace data."""
    from prototype.plots import (
        plot_anomaly_detection_timeline,
        plot_daes_controller_dynamics,
        plot_method_comparisons,
        plot_multichannel_stream,
        plot_telemetry_gate_diagnostics,
        plot_tradeoff_scatter,
    )
    import plotly.graph_objects as go

    data_cfg, model_cfg, _ = configs
    stream = generate_stream(
        seed=13,
        regime="abrupt",
        fault_condition="mixed_10",
        duration_seconds=300,
    )
    spec = make_spec("rad_daes", model_cfg, override=DEFAULT_OVERRIDES["rad_daes"])
    engine = InteractiveStreamEngine(
        stream=stream,
        spec=spec,
        thresholds=DEFAULT_THRESHOLDS["rad_daes"],
        model_cfg=model_cfg,
        warmup=50,
    )
    engine.step_n(200)
    df = engine.to_dataframe()

    f1 = plot_multichannel_stream(df)
    assert isinstance(f1, go.Figure)

    f2 = plot_anomaly_detection_timeline(df, alarm_threshold=65.0, gate_threshold=55.0)
    assert isinstance(f2, go.Figure)

    f3 = plot_telemetry_gate_diagnostics(df)
    assert isinstance(f3, go.Figure)

    f4 = plot_daes_controller_dynamics(df)
    assert isinstance(f4, go.Figure)

    df_comp = compare_all_methods(stream, methods=["static", "rad_daes"], model_cfg=model_cfg, warmup=50)
    f5 = plot_method_comparisons(df_comp)
    assert isinstance(f5, go.Figure)

    f6 = plot_tradeoff_scatter(df_comp)
    assert isinstance(f6, go.Figure)


def test_engine_session_state_lifecycle(configs):
    """Verify engine lifecycle transitions: step, step_n, jump-to-event, reset, run_all."""
    data_cfg, model_cfg, _ = configs
    stream = generate_stream(
        seed=42,
        regime="abrupt",
        fault_condition="mixed_10",
        duration_seconds=500,
        n_source_events=2,
    )
    spec = make_spec("rad_daes", model_cfg, override=DEFAULT_OVERRIDES["rad_daes"])
    engine = InteractiveStreamEngine(
        stream=stream,
        spec=spec,
        thresholds=DEFAULT_THRESHOLDS["rad_daes"],
        model_cfg=model_cfg,
        warmup=50,
    )

    # Initial state
    assert engine.current_t == 0
    assert not engine.is_finished

    # 1. Step +1
    s1 = engine.step()
    assert s1 is not None
    assert engine.current_t == 1

    # 2. Step +10
    s10 = engine.step_n(10)
    assert len(s10) == 10
    assert engine.current_t == 11

    # 3. Jump to next event or fault
    s_jump = engine.step_until_event_or_fault()
    assert len(s_jump) > 0
    assert engine.current_t > 11
    last_idx = engine.current_t - 1
    assert stream.y[last_idx] == 1 or stream.fault[last_idx] != "none"

    # 4. Reset
    engine.reset()
    assert engine.current_t == 0
    assert engine.updates == 0
    assert engine.last_snapshot is None
    assert not engine.is_finished

    # 5. Run all
    engine.run_all()
    assert engine.is_finished
    assert engine.current_t == 500
    assert engine.step() is None


def test_csv_download_payloads_validity(configs):
    """Verify both trace and benchmark CSV export payloads have expected columns and valid data."""
    import io

    data_cfg, model_cfg, _ = configs
    stream = generate_stream(
        seed=14,
        regime="gradual",
        fault_condition="mixed_5",
        duration_seconds=400,
        n_source_events=2,
    )
    spec = make_spec("rad_daes", model_cfg, override=DEFAULT_OVERRIDES["rad_daes"])
    engine = InteractiveStreamEngine(
        stream=stream,
        spec=spec,
        thresholds=DEFAULT_THRESHOLDS["rad_daes"],
        model_cfg=model_cfg,
        warmup=50,
    )
    engine.step_n(200)

    # 1. Trace CSV
    df_trace = engine.to_dataframe()
    csv_trace_str = df_trace.to_csv(index=False)
    loaded_trace = pd.read_csv(io.StringIO(csv_trace_str))

    assert len(loaded_trace) == 200
    required_trace_cols = [
        "t", "t_obs", "score", "alarm", "gate_passed", "gate_override",
        "valid", "timing", "poisson", "innovation", "eligible",
        "y_true", "fault", "event_id", "eta", "lambda_hat"
    ] + [f"W{i}_counts" for i in range(1, 7)] + [f"W{i}_mu_est" for i in range(1, 7)]
    for col in required_trace_cols:
        assert col in loaded_trace.columns, f"Missing trace column: {col}"

    # 2. Benchmark Comparison CSV
    df_comp = compare_all_methods(stream, model_cfg=model_cfg, warmup=50)
    csv_comp_str = df_comp.to_csv(index=False)
    loaded_comp = pd.read_csv(io.StringIO(csv_comp_str))

    assert len(loaded_comp) == 9
    required_comp_cols = [
        "method", "f1", "precision", "recall", "far_per_hour", "bg_rel_error",
        "latency_mean", "recovery_mean", "update_stability", "n_updates",
        "runtime_seconds", "gate_pass_rate"
    ]
    for col in required_comp_cols:
        assert col in loaded_comp.columns, f"Missing benchmark column: {col}"


def test_benchmark_preserves_frozen_benchmark_files(configs):
    """Verify that multi-method batch comparison strictly leaves frozen benchmark results untouched."""
    import hashlib

    data_cfg, model_cfg, _ = configs
    protected_files = [
        _REPO_ROOT / "results" / "raw" / "test_runs.csv",
        _REPO_ROOT / "submission" / "CHECKSUMS.sha256",
        _REPO_ROOT / "paper" / "main.pdf",
    ]

    # Compute hashes before
    hashes_before = {}
    for p in protected_files:
        if p.exists():
            hashes_before[p] = hashlib.sha256(p.read_bytes()).hexdigest()

    # Run multi-method comparison on a stream
    stream = generate_stream(
        seed=15,
        regime="stable",
        fault_condition="clean",
        duration_seconds=300,
        n_source_events=1,
    )
    df_comp = compare_all_methods(stream, model_cfg=model_cfg, warmup=30)
    assert len(df_comp) == 9

    # Verify hashes after
    for p, h_before in hashes_before.items():
        assert p.exists(), f"File vanished: {p}"
        h_after = hashlib.sha256(p.read_bytes()).hexdigest()
        assert h_before == h_after, f"Protected benchmark file modified! {p}"


def test_streamlit_apptest_lifecycle():
    """Verify Streamlit app rendering and button interactions via streamlit.testing.v1.AppTest."""
    from streamlit.testing.v1 import AppTest

    app_path = str(_PROTOTYPE_DIR / "app.py")
    at = AppTest.from_file(app_path, default_timeout=30)
    at.run()

    # 1. Initial render must produce 0 exceptions
    assert len(at.exception) == 0, f"App render threw exceptions: {at.exception}"

    # 2. Check 4 tabs are present
    assert len(at.tabs) == 4

    # 3. Step +1 button interaction
    step_btn = next((b for b in at.button if "Step" in b.label and "+1" in b.label), None)
    if step_btn is not None:
        step_btn.click()
        at.run()
        assert len(at.exception) == 0, f"Step +1 click threw exception: {at.exception}"

    # 4. Reset button interaction
    reset_btn = next((b for b in at.button if "Reset" in b.label), None)
    if reset_btn is not None:
        reset_btn.click()
        at.run()
        assert len(at.exception) == 0, f"Reset click threw exception: {at.exception}"


