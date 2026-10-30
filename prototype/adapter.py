"""Rad-DAES: Interactive Streaming Adapter.

Provides `InteractiveStreamEngine`, a stateful, sample-by-sample execution
adapter that reproduces `rad_daes.evaluation.runner.run_stream()` with exact
mathematical parity.

All generated radiation data are strictly synthetic.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

# Ensure repository 'src' is discoverable
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from rad_daes.adaptation.daes import DAESController, DAESStep
from rad_daes.background.model import BackgroundModel
from rad_daes.data.synthetic import Stream, generate_stream
from rad_daes.detection.detector import MahalanobisDetector
from rad_daes.evaluation.runner import MethodSpec, _initial_model, make_spec
import rad_daes.evaluation.metrics as M
from rad_daes.telemetry.gate import GateDecision, TelemetryGate


@dataclass
class StepSnapshot:
    """Detailed observation and diagnostic state captured at sample t."""

    t: int
    t_obs: float
    x: np.ndarray
    residual: np.ndarray
    score: float
    decision: GateDecision
    alarm: int
    eligible: bool
    mean_estimate: np.ndarray
    mu_true: np.ndarray
    y_true: int
    fault_true: str
    event_id: int
    daes_step: DAESStep | None = None
    step_size_eta: float = float("nan")
    lambda_hat: float = float("nan")
    rayleigh_quotient: float = float("nan")
    drift_score: float = float("nan")
    cov_diag: np.ndarray | None = None


class InteractiveStreamEngine:
    """Stateful streaming execution engine matching `run_stream()` exactly.

    Maintains step-by-step state, allowing interactive play, pause, single-stepping,
    event-seeking, and live diagnostics during research demonstrations.
    """

    def __init__(
        self,
        stream: Stream,
        spec: MethodSpec,
        thresholds: dict[str, float],
        model_cfg: dict,
        warmup: int = 600,
    ) -> None:
        if warmup < 0 or warmup >= stream.duration_seconds:
            raise ValueError("warmup must be in [0, duration)")

        self.stream = stream
        self.spec = spec
        self.thresholds = dict(thresholds)
        self.model_cfg = dict(model_cfg)
        self.warmup = int(warmup)
        self.duration: int = int(stream.duration_seconds)
        self.nominal = np.asarray(stream.metadata["nominal_cps"], dtype=float)

        # Configuration parameters matching run_stream()
        self.eps = float(model_cfg["detector"].get("epsilon", 1e-6))
        self.cov_beta = float(model_cfg["background"].get("covariance_forgetting", 0.02))
        self.cov_shrink = float(model_cfg["background"].get("covariance_shrinkage", 0.10))
        self.cov_floor = float(model_cfg["background"].get("covariance_floor", 1e-6))
        self.floor_fraction = float(model_cfg["background"].get("covariance_floor_fraction", 0.0))
        self.cov_floor_vector = np.maximum(self.cov_floor, self.floor_fraction * self.nominal)

        # Reset and initialize runtime state
        self.reset()

    def reset(self) -> None:
        """Reset all internal models, controllers, detectors, and histories."""
        self.model: BackgroundModel = _initial_model(self.nominal)
        self.precision: np.ndarray = self.model.precision(self.eps)

        gate_cfg = self.model_cfg["gate"]
        self.gate = TelemetryGate(
            window=int(gate_cfg.get("window", 10)),
            fano_min=float(gate_cfg.get("fano_min", 0.25)),
            fano_max=float(gate_cfg.get("fano_max", 3.50)),
            max_dt_seconds=float(gate_cfg.get("max_dt_seconds", 1.50)),
            min_dt_seconds=float(gate_cfg.get("min_dt_seconds", 0.50)),
            innovation_threshold=float(self.thresholds["gate"]),
            drift_patience=int(gate_cfg.get("drift_patience", 90)),
        )

        self.detector = MahalanobisDetector(
            threshold=float(self.thresholds["alarm"]),
            persistence=2,
            epsilon=self.eps,
        )

        self.controller: DAESController | None = None
        self.eta_fixed = float("nan")
        if self.spec.update_kind in {"daes", "daes_fixed"}:
            p = self.spec.params
            self.controller = DAESController(
                gain=float(p["gain"]),
                safety=float(p["safety"]),
                power_iterations=int(p["power_iterations"]),
                tracker_alpha=float(p["tracker_alpha"]),
                tracker_alpha_max=float(p["tracker_alpha_max"]),
                drift_sensitivity=float(p["drift_sensitivity"]),
                initial_vector=p["initial_vector"],
            )
            h0 = self.model.precision(self.eps)
            self.controller.reset(h0)
            if self.spec.update_kind == "daes_fixed":
                self.eta_fixed = self.controller.step(h0).eta

        # Baseline parameters
        self.alpha = float(self.spec.params.get("alpha", 0.02))
        self.rls_forgetting = float(self.spec.params.get("forgetting", 0.995))
        self.rls_p = np.ones(self.model.dimension, dtype=float)
        self.adam_lr = float(self.spec.params.get("learning_rate", 0.01))
        self.adam_b1 = float(self.spec.params.get("beta1", 0.9))
        self.adam_b2 = float(self.spec.params.get("beta2", 0.999))
        self.adam_eps = float(self.spec.params.get("epsilon", 1e-8))
        self.adam_m = np.zeros(self.model.dimension, dtype=float)
        self.adam_v = np.zeros(self.model.dimension, dtype=float)
        self.adam_t = 0

        # Trace buffers matching run_stream()
        n = self.duration
        d = self.model.dimension
        self.mu_hist = np.empty((n, d), dtype=np.float32)
        self.score_hist = np.empty(n, dtype=np.float64)
        self.alarm_hist = np.zeros(n, dtype=np.int8)
        self.gate_hist = np.zeros(n, dtype=bool)
        self.gate_override_hist = np.zeros(n, dtype=bool)
        self.gate_flags = np.zeros((n, 4), dtype=bool)
        self.eta_hist = np.full(n, np.nan, dtype=np.float64)
        self.lambda_hist = np.full(n, np.nan, dtype=np.float64)
        self.rayleigh_hist = np.full(n, np.nan, dtype=np.float64)
        self.drift_hist = np.full(n, np.nan, dtype=np.float64)
        self.eligible_hist = np.zeros(n, dtype=bool)

        self.current_t: int = 0
        self.updates: int = 0
        self.last_snapshot: StepSnapshot | None = None

    @property
    def is_finished(self) -> bool:
        """Whether the stream has completed all samples."""
        return self.current_t >= self.duration

    def step(self) -> StepSnapshot | None:
        """Process exactly one sample t following run_stream() sequence."""
        if self.is_finished:
            return None

        t = self.current_t
        x = self.stream.x[t]
        t_obs = float(self.stream.t_obs[t])

        # 1. Residual and Pre-Update Mahalanobis distance
        residual = x - self.model.mean
        if np.all(np.isfinite(x)):
            score = float(residual @ self.precision @ residual)
        else:
            score = float("nan")

        # 2. Gate evaluation and Detector evaluation
        decision = self.gate.evaluate(x, t_obs, score)
        alarm = self.detector.update(score)

        # 3. Update eligibility
        if self.spec.update_kind == "none":
            eligible = False
        elif self.spec.gate_mode == "gate":
            eligible = decision.passed
        elif self.spec.gate_mode == "none":
            eligible = bool(np.all(np.isfinite(x)))
        elif self.spec.gate_mode == "oracle":
            eligible = bool(
                self.stream.y[t] == 0
                and self.stream.fault[t] == "none"
                and np.all(np.isfinite(x))
            )
        else:
            raise ValueError(f"unknown gate mode: {self.spec.gate_mode}")

        # 4. Record history arrays
        self.mu_hist[t] = self.model.mean.astype(np.float32)
        self.score_hist[t] = score
        self.alarm_hist[t] = alarm
        self.gate_hist[t] = decision.passed
        self.gate_override_hist[t] = decision.sustained_override
        self.gate_flags[t] = (
            decision.valid,
            decision.timing,
            decision.poisson,
            decision.innovation,
        )
        self.eligible_hist[t] = eligible

        # DAES diagnostics
        daes_step: DAESStep | None = None
        step_eta = float("nan")
        lambda_hat = float("nan")
        rayleigh = float("nan")
        drift = float("nan")

        # 5. Conditional Update or State Freeze
        if eligible and self.spec.update_kind != "none":
            if self.spec.update_kind == "ewma":
                delta = self.alpha * residual
            elif self.spec.update_kind == "rls":
                gain = self.rls_p / (self.rls_forgetting + self.rls_p)
                delta = gain * residual
                self.rls_p = (self.rls_p - gain * self.rls_p) / self.rls_forgetting
                self.rls_p = np.maximum(self.rls_p, 1e-12)
            elif self.spec.update_kind == "adam":
                direction = self.precision @ residual
                self.adam_t += 1
                self.adam_m = self.adam_b1 * self.adam_m + (1.0 - self.adam_b1) * direction
                self.adam_v = self.adam_b2 * self.adam_v + (1.0 - self.adam_b2) * (direction * direction)
                m_hat = self.adam_m / (1.0 - self.adam_b1**self.adam_t)
                v_hat = self.adam_v / (1.0 - self.adam_b2**self.adam_t)
                delta = self.adam_lr * m_hat / (np.sqrt(v_hat) + self.adam_eps)
            elif self.spec.update_kind == "daes":
                assert self.controller is not None
                daes_step = self.controller.step(self.precision)
                step_eta = daes_step.eta
                lambda_hat = daes_step.filtered_lambda
                rayleigh = daes_step.raw_rayleigh
                drift = daes_step.drift_score
                delta = step_eta * (self.precision @ residual)
            elif self.spec.update_kind == "daes_fixed":
                step_eta = self.eta_fixed
                delta = self.eta_fixed * (self.precision @ residual)
            else:
                raise ValueError(f"unknown update kind: {self.spec.update_kind}")

            # Apply background update
            self.model.update_mean(delta)
            self.model.update_covariance(
                x,
                beta=self.cov_beta,
                shrinkage=self.cov_shrink,
                floor=self.cov_floor_vector,
            )
            self.precision = self.model.precision(self.eps)
            self.updates += 1

        # Store controller diagnostics in histories
        self.eta_hist[t] = step_eta
        self.lambda_hist[t] = lambda_hat
        self.rayleigh_hist[t] = rayleigh
        self.drift_hist[t] = drift

        snapshot = StepSnapshot(
            t=t,
            t_obs=t_obs,
            x=x.copy(),
            residual=residual.copy(),
            score=score,
            decision=decision,
            alarm=alarm,
            eligible=eligible,
            mean_estimate=self.model.mean.copy(),
            mu_true=self.stream.mu_true[t].copy(),
            y_true=int(self.stream.y[t]),
            fault_true=str(self.stream.fault[t]),
            event_id=int(self.stream.event_id[t]),
            daes_step=daes_step,
            step_size_eta=step_eta,
            lambda_hat=lambda_hat,
            rayleigh_quotient=rayleigh,
            drift_score=drift,
            cov_diag=np.diag(self.model.covariance).copy(),
        )

        self.last_snapshot = snapshot
        self.current_t += 1
        return snapshot

    def step_n(self, n_steps: int) -> list[StepSnapshot]:
        """Step the engine by up to `n_steps` samples."""
        snapshots: list[StepSnapshot] = []
        for _ in range(n_steps):
            if self.is_finished:
                break
            snap = self.step()
            if snap is not None:
                snapshots.append(snap)
        return snapshots

    def step_until_event_or_fault(self) -> list[StepSnapshot]:
        """Advance until the start of a source event (y==1) or injected fault."""
        snapshots: list[StepSnapshot] = []
        while not self.is_finished:
            t = self.current_t
            if self.stream.y[t] == 1 or self.stream.fault[t] != "none":
                # Advance one step into the event and stop
                snap = self.step()
                if snap is not None:
                    snapshots.append(snap)
                break
            snap = self.step()
            if snap is not None:
                snapshots.append(snap)
        return snapshots

    def run_all(self) -> None:
        """Run all remaining steps to completion."""
        while not self.is_finished:
            self.step()

    def get_run_results(self) -> dict[str, Any]:
        """Compute evaluation metrics matching run_stream() return structure."""
        t_max = self.current_t
        if t_max == 0:
            return {}

        y = self.stream.y[:t_max]
        fault = self.stream.fault[:t_max]
        bg_mask = (y == 0) & (fault == "none")

        effective_warmup = self.warmup if t_max > self.warmup else 0
        bg_mask[:effective_warmup] = False

        detection = M.binary_detection_metrics(
            y,
            self.alarm_hist[:t_max],
            self.score_hist[:t_max],
            start=effective_warmup,
        )
        far = M.false_alarms_per_hour(
            y,
            self.alarm_hist[:t_max],
            start=effective_warmup,
        )
        bg_error = M.background_relative_error(
            self.mu_hist[:t_max],
            self.stream.mu_true[:t_max],
            bg_mask,
        )
        stability = M.update_stability(
            self.mu_hist[:t_max],
            self.nominal,
            start=effective_warmup,
        )
        latencies = M.detection_latencies(
            y,
            self.alarm_hist[:t_max],
            self.stream.event_id[:t_max],
        )
        finite_latencies = [v for v in latencies if np.isfinite(v)]
        recoveries = M.recovery_times(
            self.mu_hist[:t_max],
            self.stream.mu_true[:t_max],
            self.stream.drift_events,
        )
        finite_recoveries = [v for v in recoveries if np.isfinite(v)]
        gate_diag = M.gate_diagnostics(
            self.gate_hist[:t_max],
            self.gate_flags[:t_max],
            y,
            fault,
            start=effective_warmup,
            override=self.gate_override_hist[:t_max],
        )

        return {
            "method": self.spec.name,
            "seed": int(self.stream.seed),
            "regime": self.stream.regime,
            "fault_condition": self.stream.fault_condition,
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
            "n_updates": int(self.updates),
            "current_t": int(t_max),
            "duration": int(self.duration),
            **gate_diag,
        }

    def get_traces(self, up_to_t: int | None = None) -> dict[str, np.ndarray]:
        """Extract recorded traces up to a given index or current t."""
        t_end = self.current_t if up_to_t is None else min(up_to_t, self.current_t)
        return {
            "mu": self.mu_hist[:t_end].copy(),
            "score": self.score_hist[:t_end].copy(),
            "alarm": self.alarm_hist[:t_end].copy(),
            "gate": self.gate_hist[:t_end].copy(),
            "gate_override": self.gate_override_hist[:t_end].copy(),
            "gate_flags": self.gate_flags[:t_end].copy(),
            "eta": self.eta_hist[:t_end].copy(),
            "lambda_hat": self.lambda_hist[:t_end].copy(),
            "rayleigh": self.rayleigh_hist[:t_end].copy(),
            "drift": self.drift_hist[:t_end].copy(),
            "eligible": self.eligible_hist[:t_end].copy(),
        }

    def to_dataframe(self, up_to_t: int | None = None) -> pd.DataFrame:
        """Export tabular trace data for analysis or CSV download."""
        t_end = self.current_t if up_to_t is None else min(up_to_t, self.current_t)
        if t_end == 0:
            return pd.DataFrame()

        rows = {
            "t": np.arange(t_end),
            "t_obs": self.stream.t_obs[:t_end],
            "score": self.score_hist[:t_end],
            "alarm": self.alarm_hist[:t_end],
            "gate_passed": self.gate_hist[:t_end],
            "gate_override": self.gate_override_hist[:t_end],
            "valid": self.gate_flags[:t_end, 0],
            "timing": self.gate_flags[:t_end, 1],
            "poisson": self.gate_flags[:t_end, 2],
            "innovation": self.gate_flags[:t_end, 3],
            "eligible": self.eligible_hist[:t_end],
            "y_true": self.stream.y[:t_end],
            "fault": self.stream.fault[:t_end],
            "event_id": self.stream.event_id[:t_end],
            "eta": self.eta_hist[:t_end],
            "lambda_hat": self.lambda_hist[:t_end],
        }

        # Add 6 channel counts and estimated means
        for ch in range(self.model.dimension):
            w_name = f"W{ch+1}"
            rows[f"{w_name}_counts"] = self.stream.x[:t_end, ch]
            rows[f"{w_name}_mu_est"] = self.mu_hist[:t_end, ch]
            rows[f"{w_name}_mu_true"] = self.stream.mu_true[:t_end, ch]

        return pd.DataFrame(rows)
