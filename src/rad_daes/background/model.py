"""Background mean/covariance state and its numerical helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BackgroundModel:
    """A Gaussian background state with a full covariance matrix."""

    mean: np.ndarray
    covariance: np.ndarray

    def __post_init__(self) -> None:
        self.mean = np.asarray(self.mean, dtype=float).copy()
        self.covariance = np.asarray(self.covariance, dtype=float).copy()
        if self.mean.ndim != 1:
            raise ValueError("mean must be a vector")
        d = self.mean.shape[0]
        if self.covariance.shape != (d, d):
            raise ValueError("covariance must be D x D")
        if not np.all(np.isfinite(self.mean)) or not np.all(np.isfinite(self.covariance)):
            raise ValueError("background state contains non-finite values")
        self.covariance = 0.5 * (self.covariance + self.covariance.T)

    @property
    def dimension(self) -> int:
        return int(self.mean.shape[0])

    def copy(self) -> "BackgroundModel":
        return BackgroundModel(self.mean.copy(), self.covariance.copy())

    def precision(self, epsilon: float = 1e-6) -> np.ndarray:
        """Return ``(covariance + epsilon I)^{-1}``."""

        if epsilon < 0:
            raise ValueError("epsilon must be non-negative")
        d = self.dimension
        matrix = self.covariance + epsilon * np.eye(d)
        # Symmetrize before inversion; the covariance update already does this.
        matrix = 0.5 * (matrix + matrix.T)
        return np.linalg.inv(matrix)

    def score(self, x: np.ndarray, epsilon: float = 1e-6) -> float:
        """Pre-update Mahalanobis score ``r^T (Sigma + eps I)^{-1} r``."""

        x = np.asarray(x, dtype=float).ravel()
        if x.shape != self.mean.shape:
            raise ValueError("observation has the wrong dimension")
        if not np.all(np.isfinite(x)):
            return float("nan")
        residual = x - self.mean
        precision = self.precision(epsilon)
        return float(residual @ precision @ residual)

    def update_mean(self, delta: np.ndarray) -> None:
        delta = np.asarray(delta, dtype=float).ravel()
        if delta.shape != self.mean.shape:
            raise ValueError("mean update has the wrong dimension")
        if not np.all(np.isfinite(delta)):
            raise ValueError("mean update contains non-finite values")
        self.mean = self.mean + delta

    def update_covariance(
        self,
        x: np.ndarray,
        beta: float = 0.02,
        shrinkage: float = 0.10,
        floor: float | np.ndarray = 1e-6,
    ) -> None:
        """EWMA covariance update with diagonal shrinkage and a positive floor.

        Uses the post-update residual ``u = x - mean``. ``floor`` may be a scalar or a
        per-window vector; a vector floor can keep the diagonal at or above a physically
        motivated Poisson variance.
        """

        if not (0.0 < beta <= 1.0):
            raise ValueError("beta must be in (0, 1]")
        if not (0.0 <= shrinkage <= 1.0):
            raise ValueError("shrinkage must be in [0, 1]")
        x = np.asarray(x, dtype=float).ravel()
        if x.shape != self.mean.shape:
            raise ValueError("observation has the wrong dimension")
        if not np.all(np.isfinite(x)):
            raise ValueError("covariance update requires a finite observation")
        floor_array = np.asarray(floor, dtype=float)
        if floor_array.ndim == 0:
            floor_array = np.full(x.shape, float(floor_array))
        if floor_array.shape != x.shape:
            raise ValueError("floor must be a scalar or have one entry per window")
        if not np.all(np.isfinite(floor_array)) or np.any(floor_array <= 0.0):
            raise ValueError("floor entries must be positive and finite")

        residual = x - self.mean
        empirical = np.outer(residual, residual)
        updated = (1.0 - beta) * self.covariance + beta * empirical
        updated = (1.0 - shrinkage) * updated + shrinkage * np.diag(np.diag(updated))
        updated = 0.5 * (updated + updated.T)
        # Floor the diagonal rather than adding a constant at every update; repeated
        # addition would slowly inflate the covariance over long streams.
        diag = np.maximum(np.diag(updated), floor_array)
        np.fill_diagonal(updated, diag)
        self.covariance = updated

    def relative_error(self, true_mean: np.ndarray) -> np.ndarray:
        true_mean = np.asarray(true_mean, dtype=float).ravel()
        if true_mean.shape != self.mean.shape:
            raise ValueError("true_mean has the wrong dimension")
        return np.abs(self.mean - true_mean) / np.maximum(np.abs(true_mean), 1e-12)
