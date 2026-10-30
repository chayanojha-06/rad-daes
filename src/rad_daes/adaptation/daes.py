"""Original DAES scalar step-size controller.

The equations here are a faithful re-implementation of the supplied
``drift_aware_eigen_scaling.py`` tracker. See ``docs/daes_interface.md``.

DAES does not update a matrix. It returns a scalar step size for a caller-supplied
curvature matrix. Rad-DAES supplies ``H = Sigma^{-1}`` and applies the step to the
background mean.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DAESStep:
    eta: float
    raw_rayleigh: float
    filtered_lambda: float
    drift_score: float
    alpha: float


class DAESController:
    """Drift-aware eigenvalue step-size heuristic.

    Parameters mirror the original implementation. ``gain`` is the original ``gamma``;
    ``safety`` is the original ``s``.
    """

    def __init__(
        self,
        gain: float = 1.78,
        safety: float = 0.10,
        power_iterations: int = 7,
        tracker_alpha: float = 0.16,
        tracker_alpha_max: float = 0.82,
        drift_sensitivity: float = 7.0,
        initial_vector: np.ndarray | list[float] | None = None,
    ) -> None:
        if gain <= 0:
            raise ValueError("gain must be positive")
        if safety < 0:
            raise ValueError("safety must be non-negative")
        if power_iterations < 1:
            raise ValueError("power_iterations must be at least 1")
        if not (0.0 < tracker_alpha <= tracker_alpha_max <= 1.0):
            raise ValueError("require 0 < tracker_alpha <= tracker_alpha_max <= 1")
        if drift_sensitivity < 0:
            raise ValueError("drift_sensitivity must be non-negative")

        self.gain = float(gain)
        self.safety = float(safety)
        self.power_iterations = int(power_iterations)
        self.tracker_alpha = float(tracker_alpha)
        self.tracker_alpha_max = float(tracker_alpha_max)
        self.drift_sensitivity = float(drift_sensitivity)
        self._initial_vector = None if initial_vector is None else np.asarray(initial_vector, dtype=float)
        self._vector: np.ndarray | None = None
        self._estimate: float | None = None
        self._previous_raw: float | None = None

    def reset(self, curvature: np.ndarray, initial_vector: np.ndarray | None = None) -> None:
        """Reset the tracker. The vector is normalized and stored."""

        h = self._validate_curvature(curvature)
        vector = initial_vector if initial_vector is not None else self._initial_vector
        if vector is None:
            vector = np.ones(h.shape[0], dtype=float)
        vector = np.asarray(vector, dtype=float).ravel()
        if vector.shape != (h.shape[0],):
            raise ValueError("initial_vector has the wrong dimension")
        norm = float(np.linalg.norm(vector))
        if norm <= 0:
            raise ValueError("initial_vector must be non-zero")
        self._vector = vector / norm
        self._estimate = None
        self._previous_raw = None

    def step(self, curvature: np.ndarray) -> DAESStep:
        """Consume one curvature matrix and return the scalar step and diagnostics."""

        h = self._validate_curvature(curvature)
        if self._vector is None or self._vector.shape != (h.shape[0],):
            self.reset(h)

        assert self._vector is not None
        for _ in range(self.power_iterations):
            candidate = h @ self._vector
            norm = float(np.linalg.norm(candidate))
            if norm <= 0.0:
                break
            self._vector = candidate / norm

        raw = float(self._vector @ h @ self._vector)

        if self._estimate is None:
            self._estimate = raw
            self._previous_raw = raw
            drift = 0.0
            alpha = self.tracker_alpha
            filtered = raw
        else:
            assert self._previous_raw is not None
            drift = abs(raw - self._previous_raw) / max(abs(self._previous_raw), 1e-12)
            alpha = float(
                np.clip(
                    self.tracker_alpha + self.drift_sensitivity * drift,
                    self.tracker_alpha,
                    self.tracker_alpha_max,
                )
            )
            filtered = (1.0 - alpha) * self._estimate + alpha * raw
            # The original implementation never lets the filtered estimate lag below a
            # newly observed upward Rayleigh quotient.
            if raw > self._estimate:
                filtered = max(filtered, raw)
            self._estimate = float(filtered)
            self._previous_raw = raw

        eta = self.gain / ((1.0 + self.safety) * max(float(self._estimate), 1e-12))
        return DAESStep(
            eta=float(eta),
            raw_rayleigh=raw,
            filtered_lambda=float(self._estimate),
            drift_score=float(drift),
            alpha=float(alpha),
        )

    @property
    def estimate(self) -> float | None:
        return self._estimate

    @staticmethod
    def _validate_curvature(curvature: np.ndarray) -> np.ndarray:
        h = np.asarray(curvature, dtype=float)
        if h.ndim != 2 or h.shape[0] != h.shape[1]:
            raise ValueError("curvature must be a square matrix")
        if not np.all(np.isfinite(h)):
            raise ValueError("curvature contains non-finite values")
        # Symmetrize to protect the Rayleigh quotient from tiny numerical asymmetry.
        return 0.5 * (h + h.T)
