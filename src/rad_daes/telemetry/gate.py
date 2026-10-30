"""Four-condition telemetry-integrity gate."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GateDecision:
    passed: bool
    valid: bool
    timing: bool
    poisson: bool
    innovation: bool
    sustained_override: bool
    fano: np.ndarray
    dt: float


class TelemetryGate:
    """Validity, timing, Poisson-dispersion, and innovation checks.

    The gate is deliberately separate from the detector: an observation can be a genuine
    source (alarm) and still be rejected for model updating. The evaluation reports both
    decisions separately.

    A sustained-exceedance release prevents a deadlock under a persistent background step:
    after ``drift_patience`` consecutive innovation exceedances, the innovation condition is
    released while the other three telemetry conditions still apply. This is a design choice,
    not part of the original planning document. The patience is deliberately longer than the
    longest synthetic source event (60 s) so a single source event cannot trigger it.
    """

    def __init__(
        self,
        window: int = 10,
        fano_min: float = 0.25,
        fano_max: float = 3.50,
        max_dt_seconds: float = 1.50,
        min_dt_seconds: float = 0.50,
        innovation_threshold: float = 22.46,
        drift_patience: int = 90,
    ) -> None:
        if window < 2:
            raise ValueError("window must be at least 2")
        if not (0.0 <= fano_min < fano_max):
            raise ValueError("require 0 <= fano_min < fano_max")
        if min_dt_seconds <= 0 or max_dt_seconds <= min_dt_seconds:
            raise ValueError("require 0 < min_dt < max_dt")
        if innovation_threshold <= 0:
            raise ValueError("innovation_threshold must be positive")
        if drift_patience < 1:
            raise ValueError("drift_patience must be at least 1")

        self.window = int(window)
        self.fano_min = float(fano_min)
        self.fano_max = float(fano_max)
        self.max_dt_seconds = float(max_dt_seconds)
        self.min_dt_seconds = float(min_dt_seconds)
        self.innovation_threshold = float(innovation_threshold)
        self.drift_patience = int(drift_patience)
        self._buffer: deque[np.ndarray] = deque(maxlen=self.window)
        self._previous_t: float | None = None
        self._exceedance_run = 0

    def reset(self, innovation_threshold: float | None = None) -> None:
        self._buffer.clear()
        self._previous_t = None
        self._exceedance_run = 0
        if innovation_threshold is not None:
            self.innovation_threshold = float(innovation_threshold)

    def set_threshold(self, innovation_threshold: float) -> None:
        if innovation_threshold <= 0:
            raise ValueError("innovation_threshold must be positive")
        self.innovation_threshold = float(innovation_threshold)

    def evaluate(self, x: np.ndarray, t_obs: float, score: float) -> GateDecision:
        x = np.asarray(x, dtype=float).ravel()
        if x.ndim != 1:
            raise ValueError("observation must be a vector")

        valid = bool(np.all(np.isfinite(x)) and np.all(x > 0.0))

        if self._previous_t is None:
            timing = True
            dt = float("nan")
        else:
            dt = float(t_obs - self._previous_t)
            timing = bool(
                np.isfinite(dt)
                and self.min_dt_seconds <= dt <= self.max_dt_seconds
            )
        self._previous_t = float(t_obs)

        self._buffer.append(x.copy())
        if len(self._buffer) < self.window:
            poisson = False
            fano = np.full(x.shape, np.nan)
        else:
            buffer = np.vstack(self._buffer)
            with np.errstate(invalid="ignore", divide="ignore"):
                means = buffer.mean(axis=0)
                variances = buffer.var(axis=0, ddof=1)
                fano = variances / means
            poisson = bool(
                np.all(np.isfinite(fano))
                and np.all(fano >= self.fano_min)
                and np.all(fano <= self.fano_max)
            )

        innovation = bool(np.isfinite(score) and score <= self.innovation_threshold)
        if not (valid and timing and poisson):
            # A telemetry fault is not evidence of a persistent background shift.
            self._exceedance_run = 0
        elif innovation:
            self._exceedance_run = 0
        else:
            self._exceedance_run += 1
        sustained_override = bool(self._exceedance_run >= self.drift_patience)
        innovation_ok = bool(innovation or sustained_override)
        passed = bool(valid and timing and poisson and innovation_ok)
        return GateDecision(
            passed=passed,
            valid=valid,
            timing=timing,
            poisson=poisson,
            innovation=innovation,
            sustained_override=sustained_override,
            fano=np.asarray(fano, dtype=float),
            dt=dt,
        )
