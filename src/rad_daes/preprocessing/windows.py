"""Energy-window definitions and basic count aggregation.

The six windows are a design hypothesis from the supplied planning document. They are
not a validated detector configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

WINDOW_NAMES: tuple[str, ...] = ("W1", "W2", "W3", "W4", "W5", "W6")
LOWER_KEV: tuple[float, ...] = (30.0, 120.0, 450.0, 900.0, 1350.0, 1600.0)
UPPER_KEV: tuple[float, ...] = (120.0, 450.0, 900.0, 1350.0, 1600.0, 3000.0)
NOMINAL_CPS: tuple[float, ...] = (450.0, 320.0, 180.0, 95.0, 70.0, 55.0)


@dataclass(frozen=True)
class WindowSet:
    names: Sequence[str] = WINDOW_NAMES
    lower_keV: Sequence[float] = LOWER_KEV
    upper_keV: Sequence[float] = UPPER_KEV
    nominal_cps: Sequence[float] = NOMINAL_CPS

    def __post_init__(self) -> None:
        if not (len(self.names) == len(self.lower_keV) == len(self.upper_keV) == len(self.nominal_cps)):
            raise ValueError("window arrays must have the same length")
        if any(lo >= hi for lo, hi in zip(self.lower_keV, self.upper_keV)):
            raise ValueError("window lower bounds must be below upper bounds")
        if any(c <= 0 for c in self.nominal_cps):
            raise ValueError("nominal count rates must be positive")

    @property
    def dimension(self) -> int:
        return len(self.names)


def counts_in_windows(
    energies_keV: np.ndarray,
    lower_keV: Sequence[float] = LOWER_KEV,
    upper_keV: Sequence[float] = UPPER_KEV,
) -> np.ndarray:
    """Count events in half-open windows ``[lower, upper)``.

    This helper is for future ingestion of list-mode data. It is not used by the synthetic
    benchmark, which generates window counts directly.
    """

    energies = np.asarray(energies_keV, dtype=float).ravel()
    out = np.zeros(len(lower_keV), dtype=float)
    for j, (lo, hi) in enumerate(zip(lower_keV, upper_keV)):
        out[j] = float(np.count_nonzero((energies >= lo) & (energies < hi)))
    return out
