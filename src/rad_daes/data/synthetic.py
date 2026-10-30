"""Synthetic six-window Poisson streams with labelled background, sources, and faults.

This module is the only data source for the first-draft experiments. It is synthetic by
construction and must never be described as measured radiation data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from rad_daes.preprocessing.windows import NOMINAL_CPS, WINDOW_NAMES

REGIMES = ("stable", "gradual", "abrupt", "recurring")
FAULT_MODES = ("dropout", "jitter", "stuck", "spike")

# Relative per-window excess templates. The maximum entry is 1.0, so the
# amplitude parameter is approximately the peak z-score in the dominant window.
SOURCE_TEMPLATES: dict[str, np.ndarray] = {
    "cs137": np.array([0.05, 0.15, 1.00, 0.25, 0.05, 0.02]),
    "co60": np.array([0.03, 0.08, 0.20, 1.00, 0.15, 0.05]),
    "pu239": np.array([1.00, 0.40, 0.10, 0.05, 0.02, 0.01]),
}


@dataclass
class Stream:
    """A labelled synthetic stream."""

    seed: int
    regime: str
    fault_condition: str
    x: np.ndarray                 # (T, D) observed counts, possibly NaN
    t_obs: np.ndarray             # (T,) observed timestamps in seconds
    mu_true: np.ndarray           # (T, D) noise-free background mean
    source: np.ndarray            # (T, D) source excess
    y: np.ndarray                 # (T,) source label
    fault: np.ndarray             # (T,) fault mode string
    event_id: np.ndarray          # (T,) source event id, 0 = none
    drift_events: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def duration_seconds(self) -> int:
        return int(len(self.y))


def _ar1(rng: np.random.Generator, n: int, rho: float = 0.999) -> np.ndarray:
    """Unit-variance AR(1) sequence with a burn-in."""

    eps = rng.normal(size=n + 200)
    z = np.empty_like(eps)
    z[0] = eps[0]
    scale = np.sqrt(max(1.0 - rho * rho, 1e-12))
    for i in range(1, len(eps)):
        z[i] = rho * z[i - 1] + scale * eps[i]
    return z[200:]


def _background(
    rng: np.random.Generator,
    regime: str,
    n: int,
    nominal: np.ndarray,
) -> tuple[np.ndarray, list[dict]]:
    """Return (T, D) background mean and a list of discrete drift events."""

    t = np.arange(n, dtype=float)
    base = np.tile(nominal, (n, 1))
    drift_events: list[dict] = []

    if regime == "stable":
        sigma = 0.015
        slow = np.sin(2.0 * np.pi * t / 86400.0)[:, None]
        mu = base * (1.0 + 0.005 * slow + sigma * _ar1(rng, n)[:, None])
    elif regime == "gradual":
        sigma = 0.025
        phase = rng.uniform(0.0, 2.0 * np.pi)
        slow = np.sin(2.0 * np.pi * t / n + phase)[:, None]
        trend = (t / max(n - 1, 1))[:, None]
        mu = base * (1.0 + 0.12 * slow + 0.08 * trend + sigma * _ar1(rng, n)[:, None])
    elif regime == "abrupt":
        sigma = 0.020
        mu = base * (1.0 + sigma * _ar1(rng, n)[:, None])
        first = int(0.33 * n)
        second = int(0.66 * n)
        factor1 = np.array([1.35, 1.25, 1.10, 0.95, 0.80, 0.70])
        factor2 = np.array([0.85, 0.90, 1.00, 1.10, 1.20, 1.30])
        mu[first:] *= factor1
        mu[second:] *= factor2
        drift_events = [
            {"time": first, "factor": factor1.tolist(), "kind": "abrupt_step"},
            {"time": second, "factor": factor2.tolist(), "kind": "abrupt_step"},
        ]
    elif regime == "recurring":
        sigma = 0.020
        mu = base * (1.0 + sigma * _ar1(rng, n)[:, None])
        # Rain/radon-like bumps, strongest in the low-energy and high-energy windows.
        bump_shape = np.array([0.35, 0.10, 0.05, 0.05, 0.15, 0.55])
        for center in (int(0.18 * n), int(0.52 * n), int(0.82 * n)):
            width = rng.uniform(180.0, 420.0)
            amplitude = rng.uniform(0.35, 0.65)
            bump = amplitude * np.exp(-0.5 * ((t - center) / width) ** 2)
            mu += bump[:, None] * bump_shape[None, :] * nominal[None, :]
    else:
        raise ValueError(f"unknown regime: {regime}")

    return np.maximum(mu, 1.0), drift_events


def _sources(
    rng: np.random.Generator,
    n: int,
    nominal: np.ndarray,
    n_events: int = 6,
    duration_range: tuple[float, float] = (20.0, 60.0),
    amplitude_range: tuple[float, float] = (3.0, 8.0),
    gap_range: tuple[float, float] = (900.0, 2700.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (source excess, labels, event ids)."""

    source = np.zeros((n, len(nominal)), dtype=float)
    y = np.zeros(n, dtype=np.int8)
    event_id = np.zeros(n, dtype=np.int16)

    if n_events <= 0:
        return source, y, event_id

    templates = list(SOURCE_TEMPLATES.values())
    # Place events with minimum gaps, leaving room at both ends. The cursor advances by
    # the realised duration plus one gap, not by two gaps.
    placements: list[tuple[int, int]] = []
    cursor = int(0.10 * n)
    for _ in range(n_events):
        gap = int(rng.uniform(*gap_range))
        start = cursor + gap
        if start >= int(0.95 * n) - int(duration_range[1]):
            break
        duration = int(rng.uniform(*duration_range))
        placements.append((start, duration))
        cursor = start + duration

    for k, (start, duration) in enumerate(placements, start=1):
        end = min(start + duration, n)
        # Midpoint sampling keeps both endpoints strictly positive, so every labelled
        # sample actually contains source excess.
        u = (np.arange(end - start, dtype=float) + 0.5) / float(end - start)
        profile = np.sin(np.pi * u) ** 2
        template = templates[int(rng.integers(0, len(templates)))]
        amplitude = float(rng.uniform(*amplitude_range))
        excess = amplitude * profile[:, None] * template[None, :] * np.sqrt(nominal)[None, :]
        source[start:end] += excess
        y[start:end] = 1
        event_id[start:end] = k

    return source, y, event_id


def _fault_mask(
    rng: np.random.Generator,
    n: int,
    rate: float,
    modes: Iterable[str],
) -> np.ndarray:
    """Return an object array of fault labels with approximately ``rate * n`` affected samples."""

    fault = np.full(n, "none", dtype=object)
    if rate <= 0.0:
        return fault
    modes = tuple(modes)
    if not modes:
        raise ValueError("at least one fault mode is required")
    target = int(round(rate * n))
    affected = np.zeros(n, dtype=bool)
    count = 0
    guard = 0
    while count < target and guard < 100 * target + 1000:
        guard += 1
        t0 = int(rng.integers(0, n))
        mode = modes[int(rng.integers(0, len(modes)))]
        if mode == "stuck":
            length = int(rng.integers(5, 31))
            idx = np.arange(t0, min(t0 + length, n))
        else:
            idx = np.array([t0], dtype=int)
        idx = idx[~affected[idx]]
        if len(idx) == 0:
            continue
        if count + len(idx) > target:
            idx = idx[: target - count]
        fault[idx] = mode
        affected[idx] = True
        count += len(idx)
    return fault


def _apply_faults(
    rng: np.random.Generator,
    x: np.ndarray,
    t_obs: np.ndarray,
    mu_true: np.ndarray,
    fault: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the labelled faults in place and return (x, t_obs)."""

    x = x.copy()
    t_obs = t_obs.copy()
    n, d = x.shape
    last_valid = x[0].copy()

    for t in range(n):
        mode = fault[t]
        if mode == "none":
            if np.all(np.isfinite(x[t])):
                last_valid = x[t].copy()
            continue
        if mode == "dropout":
            k = int(rng.integers(1, min(3, d) + 1))
            channels = rng.choice(d, size=k, replace=False)
            x[t, channels] = np.nan
        elif mode == "jitter":
            if t > 0:
                offset = float(rng.uniform(0.05, 0.49))
                if rng.random() < 0.5:
                    offset = -offset
                t_obs[t] = t_obs[t] + offset
        elif mode == "stuck":
            x[t] = last_valid
        elif mode == "spike":
            channel = int(rng.integers(0, d))
            magnitude = float(rng.uniform(10.0, 50.0))
            x[t, channel] += magnitude * np.sqrt(max(mu_true[t, channel], 1.0))
        else:
            raise ValueError(f"unknown fault mode: {mode}")

        if np.all(np.isfinite(x[t])):
            last_valid = x[t].copy()

    return x, t_obs


def _parse_fault_condition(condition: str) -> tuple[float, tuple[str, ...]]:
    if condition == "clean":
        return 0.0, ()
    if condition.startswith("mixed_"):
        rate = float(condition.split("_", 1)[1]) / 100.0
        return rate, FAULT_MODES
    for mode in FAULT_MODES:
        prefix = f"{mode}_"
        if condition.startswith(prefix):
            rate = float(condition[len(prefix):]) / 100.0
            return rate, (mode,)
    raise ValueError(f"unknown fault condition: {condition}")


def generate_stream(
    seed: int,
    regime: str = "stable",
    fault_condition: str = "clean",
    duration_seconds: int = 21600,
    nominal_cps: np.ndarray | None = None,
    include_sources: bool = True,
    n_source_events: int = 6,
) -> Stream:
    """Generate one labelled synthetic stream.

    Parameters
    ----------
    seed:
        Master seed. Independent child seeds are derived for background, sources, and
        faults so that changing one component does not silently change another.
    regime:
        One of ``stable``, ``gradual``, ``abrupt``, ``recurring``.
    fault_condition:
        ``clean``, ``mixed_5``/``mixed_10``/``mixed_15``, or
        ``dropout_10``/``jitter_10``/``stuck_10``/``spike_10``.
    duration_seconds:
        Number of 1 Hz samples.
    nominal_cps:
        Optional override for the six nominal background means.
    include_sources:
        If False, no source events are injected (used for threshold calibration).
    n_source_events:
        Number of source events to attempt.
    """

    if regime not in REGIMES:
        raise ValueError(f"unknown regime: {regime}")
    nominal = np.asarray(NOMINAL_CPS if nominal_cps is None else nominal_cps, dtype=float)
    if nominal.shape != (len(WINDOW_NAMES),):
        raise ValueError("nominal_cps must have six entries")

    n = int(duration_seconds)
    if n < 100:
        raise ValueError("duration_seconds must be at least 100")

    rng_bg = np.random.default_rng((seed * 1009 + 1) % (2**63 - 1))
    rng_src = np.random.default_rng((seed * 1009 + 2) % (2**63 - 1))
    rng_fault = np.random.default_rng((seed * 1009 + 3) % (2**63 - 1))

    mu_true, drift_events = _background(rng_bg, regime, n, nominal)
    if include_sources:
        source, y, event_id = _sources(rng_src, n, nominal, n_events=n_source_events)
    else:
        source = np.zeros((n, len(nominal)), dtype=float)
        y = np.zeros(n, dtype=np.int8)
        event_id = np.zeros(n, dtype=np.int16)

    rate, modes = _parse_fault_condition(fault_condition)
    fault = _fault_mask(rng_fault, n, rate, modes)

    # Poisson draws. The background and source are combined before sampling.
    x = rng_bg.poisson(np.maximum(mu_true + source, 0.0)).astype(float)
    t_obs = np.arange(n, dtype=float)
    x, t_obs = _apply_faults(rng_fault, x, t_obs, mu_true, fault)

    return Stream(
        seed=seed,
        regime=regime,
        fault_condition=fault_condition,
        x=x,
        t_obs=t_obs,
        mu_true=mu_true,
        source=source,
        y=y,
        fault=fault,
        event_id=event_id,
        drift_events=drift_events,
        metadata={
            "nominal_cps": nominal.tolist(),
            "fault_rate": rate,
            "fault_modes": list(modes),
            "include_sources": include_sources,
            "n_source_events": int(n_source_events),
            "generator": "rad_daes.data.synthetic.generate_stream",
        },
    )
