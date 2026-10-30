"""Evaluation metrics and uncertainty summaries."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy import stats

try:  # scikit-learn is convenient but not required for the core gate/model.
    from sklearn.metrics import average_precision_score, roc_auc_score
except Exception:  # pragma: no cover - exercised only in minimal environments
    average_precision_score = None
    roc_auc_score = None


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def binary_detection_metrics(
    y: np.ndarray,
    alarm: np.ndarray,
    score: np.ndarray | None = None,
    start: int = 0,
) -> dict[str, float]:
    """Sample-level detection metrics after ``start``."""

    y = np.asarray(y, dtype=int).ravel()[start:]
    alarm = np.asarray(alarm, dtype=int).ravel()[start:]
    if y.shape != alarm.shape:
        raise ValueError("y and alarm must have the same shape")

    tp = int(np.sum((y == 1) & (alarm == 1)))
    fp = int(np.sum((y == 0) & (alarm == 1)))
    fn = int(np.sum((y == 1) & (alarm == 0)))
    tn = int(np.sum((y == 0) & (alarm == 0)))

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)

    pr_auc = float("nan")
    roc_auc = float("nan")
    if score is not None and average_precision_score is not None:
        s = np.asarray(score, dtype=float).ravel()[start:]
        if s.shape != y.shape:
            raise ValueError("score and y must have the same shape")
        mask = np.isfinite(s)
        if mask.sum() > 0 and len(np.unique(y[mask])) == 2:
            pr_auc = float(average_precision_score(y[mask], s[mask]))
            roc_auc = float(roc_auc_score(y[mask], s[mask]))

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
    }


def false_alarms_per_hour(
    y: np.ndarray,
    alarm: np.ndarray,
    start: int = 0,
    dt_seconds: float = 1.0,
) -> float:
    """False alarms over non-source time, in alarms per hour."""

    y = np.asarray(y, dtype=int).ravel()[start:]
    alarm = np.asarray(alarm, dtype=int).ravel()[start:]
    non_source = y == 0
    duration_hours = float(non_source.sum() * dt_seconds / 3600.0)
    if duration_hours <= 0:
        return float("nan")
    return float(np.sum(alarm[non_source] == 1) / duration_hours)


def background_relative_error(
    mu_hist: np.ndarray,
    mu_true: np.ndarray,
    mask: np.ndarray,
) -> float:
    """Mean relative background error over a boolean mask."""

    mu_hist = np.asarray(mu_hist, dtype=float)
    mu_true = np.asarray(mu_true, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if mu_hist.shape != mu_true.shape:
        raise ValueError("mu_hist and mu_true must have the same shape")
    if mask.shape != (mu_hist.shape[0],):
        raise ValueError("mask must be one-dimensional over time")
    if not np.any(mask):
        return float("nan")
    rel = np.abs(mu_hist[mask] - mu_true[mask]) / np.maximum(np.abs(mu_true[mask]), 1e-12)
    return float(np.mean(rel))


def update_stability(mu_hist: np.ndarray, nominal: np.ndarray, start: int = 0) -> float:
    """Mean squared relative update, normalised by the nominal count vector."""

    mu_hist = np.asarray(mu_hist, dtype=float)
    nominal = np.asarray(nominal, dtype=float)
    if mu_hist.ndim != 2:
        raise ValueError("mu_hist must be two-dimensional")
    if mu_hist.shape[0] < start + 2:
        return float("nan")
    deltas = np.diff(mu_hist[start:], axis=0) / np.maximum(nominal[None, :], 1e-12)
    return float(np.mean(deltas**2))


def detection_latencies(
    y: np.ndarray,
    alarm: np.ndarray,
    event_id: np.ndarray,
    dt_seconds: float = 1.0,
) -> list[float]:
    """Seconds from each event start to its first alarm, or NaN if undetected."""

    y = np.asarray(y, dtype=int).ravel()
    alarm = np.asarray(alarm, dtype=int).ravel()
    event_id = np.asarray(event_id, dtype=int).ravel()
    out: list[float] = []
    for eid in sorted(int(v) for v in np.unique(event_id) if v > 0):
        idx = np.flatnonzero(event_id == eid)
        if idx.size == 0:
            continue
        start = int(idx[0])
        end = int(idx[-1]) + 1
        hits = np.flatnonzero(alarm[start:end] == 1)
        out.append(float(hits[0] * dt_seconds) if hits.size else float("nan"))
    return out


def recovery_times(
    mu_hist: np.ndarray,
    mu_true: np.ndarray,
    events: Iterable[dict],
    threshold: float = 0.05,
    hold: int = 30,
    dt_seconds: float = 1.0,
) -> list[float]:
    """Recovery time after each discrete drift event.

    Recovery is the first sample at which the relative error over all windows stays below
    ``threshold`` for ``hold`` consecutive samples. Returns NaN if it never does.
    """

    mu_hist = np.asarray(mu_hist, dtype=float)
    mu_true = np.asarray(mu_true, dtype=float)
    rel = np.abs(mu_hist - mu_true) / np.maximum(np.abs(mu_true), 1e-12)
    within = np.all(rel <= threshold, axis=1)
    out: list[float] = []
    n = len(within)
    for event in events:
        start = int(event["time"])
        if start >= n:
            out.append(float("nan"))
            continue
        found = float("nan")
        run = 0
        for t in range(start, n):
            run = run + 1 if within[t] else 0
            if run >= hold:
                found = float((t - hold + 1 - start) * dt_seconds)
                break
        out.append(found)
    return out


def gate_diagnostics(
    gate_passed: np.ndarray,
    gate_flags: np.ndarray,
    y: np.ndarray,
    fault: np.ndarray,
    start: int = 0,
    override: np.ndarray | None = None,
) -> dict[str, float]:
    """Rejection rates by source/fault status and by individual gate condition.

    ``gate_flags`` has columns ``valid, timing, poisson, innovation``. ``override`` is the
    optional sustained-exceedance release flag.
    """

    gate_passed = np.asarray(gate_passed, dtype=bool).ravel()[start:]
    gate_flags = np.asarray(gate_flags, dtype=bool)
    if gate_flags.ndim != 2 or gate_flags.shape[1] != 4:
        raise ValueError("gate_flags must have four columns")
    gate_flags = gate_flags[start:]
    y = np.asarray(y, dtype=int).ravel()[start:]
    fault = np.asarray(fault).ravel()[start:]
    override = None if override is None else np.asarray(override, dtype=bool).ravel()[start:]

    source = y == 1
    faulted = fault != "none"
    clean = ~source & ~faulted

    def rate(mask: np.ndarray, value: bool = False) -> float:
        if not np.any(mask):
            return float("nan")
        return float(np.mean(gate_passed[mask] == value))

    out = {
        "gate_reject_source": rate(source, False),
        "gate_reject_fault": rate(faulted, False),
        "gate_reject_clean": rate(clean, False),
        "gate_pass_all": float(np.mean(gate_passed)),
    }
    names = ("valid", "timing", "poisson", "innovation")
    for j, name in enumerate(names):
        out[f"gate_fail_{name}_source"] = float(np.mean(~gate_flags[source, j])) if np.any(source) else float("nan")
        out[f"gate_fail_{name}_fault"] = float(np.mean(~gate_flags[faulted, j])) if np.any(faulted) else float("nan")
    if override is not None:
        if override.shape != gate_passed.shape:
            raise ValueError("override must have the same shape as gate_passed")
        out["gate_override_rate"] = float(np.mean(override))
        out["gate_override_source"] = float(np.mean(override[source])) if np.any(source) else float("nan")
        out["gate_override_fault"] = float(np.mean(override[faulted])) if np.any(faulted) else float("nan")
        out["gate_override_clean"] = float(np.mean(override[clean])) if np.any(clean) else float("nan")
    return out


def student_t_interval(values: Iterable[float], confidence: float = 0.95) -> tuple[float, float, float, int]:
    """Return (mean, low, high, n) using a two-sided Student-t interval."""

    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    mean = float(np.mean(arr))
    if n < 2:
        return mean, float("nan"), float("nan"), n
    sem = float(np.std(arr, ddof=1) / np.sqrt(n))
    crit = float(stats.t.ppf(0.5 + confidence / 2.0, df=n - 1))
    return mean, mean - crit * sem, mean + crit * sem, n
