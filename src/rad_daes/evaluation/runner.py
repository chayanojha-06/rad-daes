"""Benchmark runner: matched streams, common detector, method-specific adaptation."""

from __future__ import annotations

import json
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml
from scipy import stats

from rad_daes.adaptation.daes import DAESController
from rad_daes.background.model import BackgroundModel
from rad_daes.data.synthetic import Stream, generate_stream
from rad_daes.detection.detector import MahalanobisDetector
from rad_daes.evaluation import metrics as M
from rad_daes.preprocessing.windows import NOMINAL_CPS
from rad_daes.telemetry.gate import TelemetryGate

METHODS: tuple[str, ...] = (
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


@dataclass(frozen=True)
class MethodSpec:
    name: str
    gate_mode: str          # "none", "gate", "oracle"
    update_kind: str        # "none", "ewma", "rls", "adam", "daes", "daes_fixed"
    params: dict[str, Any] = field(default_factory=dict)


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_configs(root: str | Path) -> tuple[dict, dict, dict]:
    root = Path(root)
    return (
        load_yaml(root / "configs" / "data.yaml"),
        load_yaml(root / "configs" / "model.yaml"),
        load_yaml(root / "configs" / "experiments.yaml"),
    )


def make_spec(method: str, model_cfg: dict, override: dict | None = None) -> MethodSpec:
    """Build a method specification from the model config plus one candidate override."""

    override = dict(override or {})
    daes_cfg = dict(model_cfg["daes"])
    base = dict(model_cfg.get("baselines", {}))
    daes_cfg.update({k: v for k, v in override.items() if k.startswith("daes_") or k in {
        "gain", "safety", "power_iterations", "tracker_alpha", "tracker_alpha_max",
        "drift_sensitivity", "initial_vector",
    }})
    base.update({k: v for k, v in override.items() if k not in daes_cfg})

    common_daes = {
        "gain": float(override.get("daes_gain", daes_cfg.get("gain", 0.35))),
        "safety": float(daes_cfg.get("safety", 0.10)),
        "power_iterations": int(daes_cfg.get("power_iterations", 7)),
        "tracker_alpha": float(daes_cfg.get("tracker_alpha", 0.16)),
        "tracker_alpha_max": float(daes_cfg.get("tracker_alpha_max", 0.82)),
        "drift_sensitivity": float(daes_cfg.get("drift_sensitivity", 7.0)),
        "initial_vector": list(daes_cfg.get("initial_vector", [1.0] * 6)),
    }

    if method == "static":
        return MethodSpec(method, "none", "none", {})
    if method == "ewma":
        return MethodSpec(method, "none", "ewma", {"alpha": float(override.get("ewma_alpha", base.get("ewma_alpha", 0.02)))})
    if method == "gated_ewma":
        return MethodSpec(method, "gate", "ewma", {"alpha": float(override.get("ewma_alpha", base.get("ewma_alpha", 0.02)))})
    if method == "rls":
        return MethodSpec(method, "none", "rls", {"forgetting": float(override.get("rls_forgetting", base.get("rls_forgetting", 0.995)))})
    if method == "adam":
        return MethodSpec(
            method,
            "none",
            "adam",
            {
                "learning_rate": float(override.get("adam_learning_rate", base.get("adam_learning_rate", 0.01))),
                "beta1": float(base.get("adam_beta1", 0.9)),
                "beta2": float(base.get("adam_beta2", 0.999)),
                "epsilon": float(base.get("adam_epsilon", 1e-8)),
            },
        )
    if method == "daes":
        return MethodSpec(method, "none", "daes", common_daes)
    if method == "gated_daes_fixed":
        return MethodSpec(method, "gate", "daes_fixed", common_daes)
    if method == "rad_daes":
        return MethodSpec(method, "gate", "daes", common_daes)
    if method == "oracle_gate":
        return MethodSpec(method, "oracle", "daes", common_daes)
    raise ValueError(f"unknown method: {method}")


def _initial_model(nominal: np.ndarray) -> BackgroundModel:
    return BackgroundModel(
        mean=nominal.copy(),
        covariance=np.diag(nominal.copy()),
    )


def run_stream(
    stream: Stream,
    spec: MethodSpec,
    thresholds: dict[str, float],
    model_cfg: dict,
    warmup: int = 600,
    collect_trace: bool = False,
) -> dict[str, Any]:
    """Run one method on one stream and return run-level metrics."""

    if warmup < 0 or warmup >= stream.duration_seconds:
        raise ValueError("warmup must be in [0, duration)")

    nominal = np.asarray(stream.metadata["nominal_cps"], dtype=float)
    model = _initial_model(nominal)
    eps = float(model_cfg["detector"].get("epsilon", 1e-6))
    cov_beta = float(model_cfg["background"].get("covariance_forgetting", 0.02))
    cov_shrink = float(model_cfg["background"].get("covariance_shrinkage", 0.10))
    cov_floor = float(model_cfg["background"].get("covariance_floor", 1e-6))
    # A covariance diagonal below the Poisson variance is not physically meaningful for
    # counts. The fraction is a regularisation choice, not a claim about the data.
    floor_fraction = float(model_cfg["background"].get("covariance_floor_fraction", 0.0))
    cov_floor_vector = np.maximum(cov_floor, floor_fraction * nominal)

    gate_cfg = model_cfg["gate"]
    gate = TelemetryGate(
        window=int(gate_cfg.get("window", 10)),
        fano_min=float(gate_cfg.get("fano_min", 0.25)),
        fano_max=float(gate_cfg.get("fano_max", 3.50)),
        max_dt_seconds=float(gate_cfg.get("max_dt_seconds", 1.50)),
        innovation_threshold=float(thresholds["gate"]),
        drift_patience=int(gate_cfg.get("drift_patience", 90)),
    )
    detector = MahalanobisDetector(
        threshold=float(thresholds["alarm"]),
        persistence=2,
        epsilon=eps,
    )

    controller: DAESController | None = None
    eta_fixed = float("nan")
    if spec.update_kind in {"daes", "daes_fixed"}:
        p = spec.params
        controller = DAESController(
            gain=float(p["gain"]),
            safety=float(p["safety"]),
            power_iterations=int(p["power_iterations"]),
            tracker_alpha=float(p["tracker_alpha"]),
            tracker_alpha_max=float(p["tracker_alpha_max"]),
            drift_sensitivity=float(p["drift_sensitivity"]),
            initial_vector=p["initial_vector"],
        )
        h0 = model.precision(eps)
        controller.reset(h0)
        if spec.update_kind == "daes_fixed":
            eta_fixed = controller.step(h0).eta

    alpha = float(spec.params.get("alpha", 0.02))
    rls_forgetting = float(spec.params.get("forgetting", 0.995))
    rls_p = np.ones(model.dimension, dtype=float)
    adam_lr = float(spec.params.get("learning_rate", 0.01))
    adam_b1 = float(spec.params.get("beta1", 0.9))
    adam_b2 = float(spec.params.get("beta2", 0.999))
    adam_eps = float(spec.params.get("epsilon", 1e-8))
    adam_m = np.zeros(model.dimension, dtype=float)
    adam_v = np.zeros(model.dimension, dtype=float)
    adam_t = 0

    n = stream.duration_seconds
    mu_hist = np.empty((n, model.dimension), dtype=np.float32)
    score_hist = np.empty(n, dtype=np.float64)
    alarm_hist = np.zeros(n, dtype=np.int8)
    gate_hist = np.zeros(n, dtype=bool)
    gate_override_hist = np.zeros(n, dtype=bool)
    gate_flags = np.zeros((n, 4), dtype=bool)
    updates = 0

    precision = model.precision(eps)
    started = time.perf_counter()

    for t in range(n):
        x = stream.x[t]
        residual = x - model.mean
        if np.all(np.isfinite(x)):
            score = float(residual @ precision @ residual)
        else:
            score = float("nan")

        decision = gate.evaluate(x, float(stream.t_obs[t]), score)
        alarm = detector.update(score)

        if spec.update_kind == "none":
            eligible = False
        elif spec.gate_mode == "gate":
            eligible = decision.passed
        elif spec.gate_mode == "none":
            eligible = bool(np.all(np.isfinite(x)))
        elif spec.gate_mode == "oracle":
            eligible = bool(
                stream.y[t] == 0
                and stream.fault[t] == "none"
                and np.all(np.isfinite(x))
            )
        else:
            raise ValueError(f"unknown gate mode: {spec.gate_mode}")

        mu_hist[t] = model.mean.astype(np.float32)
        score_hist[t] = score
        alarm_hist[t] = alarm
        gate_hist[t] = decision.passed
        gate_override_hist[t] = decision.sustained_override
        gate_flags[t] = (decision.valid, decision.timing, decision.poisson, decision.innovation)

        if eligible and spec.update_kind != "none":
            if spec.update_kind == "ewma":
                delta = alpha * residual
            elif spec.update_kind == "rls":
                gain = rls_p / (rls_forgetting + rls_p)
                delta = gain * residual
                rls_p = (rls_p - gain * rls_p) / rls_forgetting
                rls_p = np.maximum(rls_p, 1e-12)
            elif spec.update_kind == "adam":
                direction = precision @ residual
                adam_t += 1
                adam_m = adam_b1 * adam_m + (1.0 - adam_b1) * direction
                adam_v = adam_b2 * adam_v + (1.0 - adam_b2) * (direction * direction)
                m_hat = adam_m / (1.0 - adam_b1**adam_t)
                v_hat = adam_v / (1.0 - adam_b2**adam_t)
                delta = adam_lr * m_hat / (np.sqrt(v_hat) + adam_eps)
            elif spec.update_kind == "daes":
                assert controller is not None
                step = controller.step(precision)
                delta = step.eta * (precision @ residual)
            elif spec.update_kind == "daes_fixed":
                delta = eta_fixed * (precision @ residual)
            else:
                raise ValueError(f"unknown update kind: {spec.update_kind}")

            model.update_mean(delta)
            model.update_covariance(
                x, beta=cov_beta, shrinkage=cov_shrink, floor=cov_floor_vector
            )
            precision = model.precision(eps)
            updates += 1

    elapsed = time.perf_counter() - started

    y = stream.y
    fault = stream.fault
    bg_mask = (y == 0) & (fault == "none")
    bg_mask[:warmup] = False

    detection = M.binary_detection_metrics(y, alarm_hist, score_hist, start=warmup)
    far = M.false_alarms_per_hour(y, alarm_hist, start=warmup)
    bg_error = M.background_relative_error(mu_hist, stream.mu_true, bg_mask)
    stability = M.update_stability(mu_hist, nominal, start=warmup)
    latencies = M.detection_latencies(y, alarm_hist, stream.event_id)
    finite_latencies = [v for v in latencies if np.isfinite(v)]
    recoveries = M.recovery_times(mu_hist, stream.mu_true, stream.drift_events)
    finite_recoveries = [v for v in recoveries if np.isfinite(v)]
    gate_diag = M.gate_diagnostics(
        gate_hist, gate_flags, y, fault, start=warmup, override=gate_override_hist
    )

    out: dict[str, Any] = {
        "method": spec.name,
        "seed": int(stream.seed),
        "regime": stream.regime,
        "fault_condition": stream.fault_condition,
        **detection,
        "far_per_hour": far,
        "bg_rel_error": bg_error,
        "update_stability": stability,
        "latency_mean": float(np.mean(finite_latencies)) if finite_latencies else float("nan"),
        "latency_median": float(np.median(finite_latencies)) if finite_latencies else float("nan"),
        "n_events": int(len(latencies)),
        "n_detected": int(len(finite_latencies)),
        "recovery_mean": float(np.mean(finite_recoveries)) if finite_recoveries else float("nan"),
        "recovery_median": float(np.median(finite_recoveries)) if finite_recoveries else float("nan"),
        "n_recovery_events": int(len(recoveries)),
        "n_recovered": int(len(finite_recoveries)),
        "runtime_seconds": float(elapsed),
        "runtime_per_sample_ms": float(elapsed / n * 1000.0),
        "n_updates": int(updates),
        **gate_diag,
    }
    if collect_trace:
        out["_trace"] = {
            "mu": mu_hist.copy(),
            "score": score_hist.copy(),
            "alarm": alarm_hist.copy(),
            "gate": gate_hist.copy(),
            "gate_override": gate_override_hist.copy(),
            "gate_flags": gate_flags.copy(),
        }
    return out


def calibrate_method(
    spec: MethodSpec,
    data_cfg: dict,
    model_cfg: dict,
    seeds: Iterable[int],
    gate_quantile: float = 0.999,
    target_far_per_hour: float = 1.0,
    duration_seconds: int = 21600,
    warmup: int = 600,
) -> dict[str, float]:
    """Calibrate alarm and gate thresholds on clean stable development streams."""

    provisional_alarm = float(stats.chi2.ppf(1.0 - target_far_per_hour / 3600.0, df=6))
    provisional_gate = max(10.0, float(stats.chi2.ppf(gate_quantile, df=6)))
    pooled: list[np.ndarray] = []

    for seed in seeds:
        stream = generate_stream(
            seed=seed,
            regime="stable",
            fault_condition="clean",
            duration_seconds=duration_seconds,
            include_sources=False,
        )
        result = run_stream(
            stream,
            spec,
            thresholds={"alarm": provisional_alarm, "gate": provisional_gate},
            model_cfg=model_cfg,
            warmup=warmup,
            collect_trace=True,
        )
        scores = np.asarray(result["_trace"]["score"], dtype=float)[warmup:]
        pooled.append(scores[np.isfinite(scores)])

    values = np.concatenate(pooled) if pooled else np.array([], dtype=float)
    if values.size == 0:
        raise RuntimeError(f"no finite calibration scores for {spec.name}")
    alarm = float(np.quantile(values, 1.0 - target_far_per_hour / 3600.0))
    gate = max(10.0, float(np.quantile(values, gate_quantile)))
    return {"alarm": alarm, "gate": gate}


def _mean(values: pd.Series) -> float:
    arr = values.to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def select_candidate(df: pd.DataFrame) -> dict[str, Any]:
    """Select a tuning candidate by the prespecified rule."""

    rows = []
    for candidate, group in df.groupby("candidate", dropna=False):
        rows.append(
            {
                "candidate": candidate,
                "mean_f1": _mean(group["f1"]),
                "mean_bg_rel_error": _mean(group["bg_rel_error"]),
                "mean_far_per_hour": _mean(group["far_per_hour"]),
                "n_runs": int(len(group)),
            }
        )
    summary = pd.DataFrame(rows)
    feasible = summary[
        (summary["mean_bg_rel_error"] <= 0.10)
        & (summary["mean_far_per_hour"] <= 2.0)
        & summary["mean_f1"].notna()
    ]
    if len(feasible):
        selected = feasible.sort_values(
            ["mean_f1", "mean_bg_rel_error"],
            ascending=[False, True],
        ).iloc[0]
    else:
        # Least-violating fallback; recorded as a protocol deviation by the caller.
        penalty = (
            np.maximum(0.0, summary["mean_bg_rel_error"] - 0.10)
            + 0.1 * np.maximum(0.0, summary["mean_far_per_hour"] - 2.0)
            - summary["mean_f1"].fillna(-1.0)
        )
        selected = summary.assign(penalty=penalty).sort_values("penalty").iloc[0]
    return {k: (v.item() if hasattr(v, "item") else v) for k, v in selected.to_dict().items()}


def environment_manifest() -> dict[str, Any]:
    import matplotlib
    import scipy
    import sklearn

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
        "matplotlib": matplotlib.__version__,
        "scikit_learn": sklearn.__version__,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
