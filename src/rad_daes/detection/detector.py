"""Pre-update Mahalanobis detector with a two-sample persistence rule."""

from __future__ import annotations

import numpy as np

from rad_daes.background.model import BackgroundModel


class MahalanobisDetector:
    """Score first, update later.

    The detector never sees the updated background for the current sample. This preserves
    the planning document's "detect first, adapt second" ordering.
    """

    def __init__(self, threshold: float = 22.46, persistence: int = 2, epsilon: float = 1e-6) -> None:
        if threshold <= 0:
            raise ValueError("threshold must be positive")
        if persistence < 1:
            raise ValueError("persistence must be at least 1")
        if epsilon < 0:
            raise ValueError("epsilon must be non-negative")
        self.threshold = float(threshold)
        self.persistence = int(persistence)
        self.epsilon = float(epsilon)
        self._consecutive = 0

    def reset(self, threshold: float | None = None) -> None:
        self._consecutive = 0
        if threshold is not None:
            self.set_threshold(threshold)

    def set_threshold(self, threshold: float) -> None:
        if threshold <= 0:
            raise ValueError("threshold must be positive")
        self.threshold = float(threshold)

    def score(self, x: np.ndarray, model: BackgroundModel) -> float:
        return model.score(x, epsilon=self.epsilon)

    def update(self, score: float) -> int:
        if not np.isfinite(score):
            self._consecutive = 0
            return 0
        if score > self.threshold:
            self._consecutive += 1
        else:
            self._consecutive = 0
        return int(self._consecutive >= self.persistence)

    @staticmethod
    def threshold_for_far(scores: np.ndarray, target_far_per_hour: float, dt_seconds: float = 1.0) -> float:
        """Empirical threshold for a target per-sample false-alarm rate.

        The persistence rule can only reduce the realised rate; this threshold is the
        single-sample quantile, not a guarantee for the two-sample alarm.
        """

        if target_far_per_hour <= 0:
            raise ValueError("target_far_per_hour must be positive")
        if dt_seconds <= 0:
            raise ValueError("dt_seconds must be positive")
        values = np.asarray(scores, dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            raise ValueError("no finite scores to calibrate")
        alpha = min(max(target_far_per_hour * dt_seconds / 3600.0, 0.0), 1.0)
        quantile = 1.0 - alpha
        return float(np.quantile(values, quantile))
