"""Metrics, baselines and bootstrap intervals used by evaluation and the gate."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

Arr = NDArray[np.floating[Any]]
MetricFn = Callable[[Arr, Arr], float]


def mape(y: Arr, p: Arr) -> float:
    return float(np.mean(np.abs(p - y) / np.abs(y)))


def median_ape(y: Arr, p: Arr) -> float:
    return float(np.median(np.abs(p - y) / np.abs(y)))


def within_10pct(y: Arr, p: Arr) -> float:
    return float(np.mean(np.abs(p - y) / np.abs(y) <= 0.10))


def mae(y: Arr, p: Arr) -> float:
    return float(np.mean(np.abs(p - y)))


def auc(y: Arr, p: Arr) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def brier(y: Arr, p: Arr) -> float:
    return float(brier_score_loss(y, np.clip(p, 0, 1)))


def logloss(y: Arr, p: Arr) -> float:
    return float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1]))


METRICS: dict[str, tuple[MetricFn, str]] = {
    "mape": (mape, "lower"),
    "median_ape": (median_ape, "lower"),
    "within_10pct": (within_10pct, "higher"),
    "mae": (mae, "lower"),
    "auc": (auc, "higher"),
    "brier": (brier, "lower"),
    "logloss": (logloss, "lower"),
}


# Value a useless model scores. Relative improvements are measured from here:
# going from AUC 0.745 to 0.796 is 21% more lift over chance, not 7%.
CHANCE: dict[str, float] = {"auc": 0.5}


def relative_improvement(name: str, candidate: float, reference: float) -> float:
    span = abs(reference - CHANCE.get(name, 0.0))
    return improvement(name, candidate, reference) / span if span else 0.0


def compute(names: list[str], y: Arr, p: Arr) -> dict[str, float]:
    return {n: METRICS[n][0](y, p) for n in names}


def better(name: str, a: float, b: float) -> bool:
    """True if value ``a`` is better than ``b`` for metric ``name``."""
    return a < b if METRICS[name][1] == "lower" else a > b


def improvement(name: str, candidate: float, reference: float) -> float:
    """Signed improvement of candidate over reference (positive = better)."""
    return reference - candidate if METRICS[name][1] == "lower" else candidate - reference


def bootstrap_ci(name: str, y: Arr, p: Arr, n: int = 400, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    fn = METRICS[name][0]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), (n, len(y)))
    vals = np.array([fn(y[i], p[i]) for i in idx])
    lo, hi = np.nanquantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def paired_bootstrap(name: str, y: Arr, p_new: Arr, p_old: Arr, n: int = 400, seed: int = 0, alpha: float = 0.05) -> dict[str, float]:
    """CI of the improvement of p_new over p_old on the SAME resampled rows."""
    fn = METRICS[name][0]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), (n, len(y)))
    deltas = np.array([improvement(name, fn(y[i], p_new[i]), fn(y[i], p_old[i])) for i in idx])
    lo, hi = np.nanquantile(deltas, [alpha / 2, 1 - alpha / 2])
    return {
        "delta": improvement(name, fn(y, p_new), fn(y, p_old)),
        "ci_low": float(lo),
        "ci_high": float(hi),
    }
