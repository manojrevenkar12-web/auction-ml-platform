"""Reference profiles, PSI drift scores and empirically calibrated thresholds.

Thresholds are not the folklore 0.1/0.25: they are set from how much PSI moves
week to week while nothing is wrong (the training period's own weeks scored
against the reference profile). That keeps the false-alarm rate measured rather
than assumed.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

EPS = 1e-4
OTHER = "__other__"
UNSEEN = "__unseen__"
MISSING = "__missing__"
PREDICTION = "__prediction__"


def _num_profile(x: pd.Series, bins: int = 10) -> dict[str, Any]:
    v = pd.to_numeric(x, errors="coerce").astype(float)
    finite = v.dropna().to_numpy()
    edges = np.unique(np.quantile(finite, np.linspace(0, 1, bins + 1))) if len(finite) else np.array([0.0, 1.0])
    edges = edges[1:-1]  # inner edges; outer bins are open-ended
    return {"kind": "numeric", "edges": edges.tolist(), "freq": _num_freq(v, edges).tolist(), "null_rate": float(v.isna().mean())}


def _num_freq(v: pd.Series, edges: np.ndarray) -> np.ndarray:
    vals = v.to_numpy(dtype=float)
    idx = np.where(np.isnan(vals), len(edges) + 1, np.searchsorted(edges, vals, side="right"))
    counts = np.bincount(idx.astype(int), minlength=len(edges) + 2).astype(float)
    return counts / max(len(vals), 1)


def _cat_profile(x: pd.Series, min_share: float = 0.01) -> dict[str, Any]:
    s = x.astype(object).where(x.notna(), MISSING).astype(str)
    share = s.value_counts(normalize=True)
    keep = [str(k) for k, v in share.items() if v >= min_share]
    return {"kind": "categorical", "categories": keep, "known": sorted(map(str, share.index)), "freq": _cat_freq(s, keep, None).tolist(), "null_rate": float(x.isna().mean())}


def _cat_freq(s: pd.Series, keep: list[str], known: set[str] | None) -> np.ndarray:
    """Share per kept category, then OTHER (seen in reference, rare) and UNSEEN."""
    labels = keep + [OTHER, UNSEEN]
    if known is None:
        mapped = s.where(s.isin(keep), OTHER)
    else:
        mapped = pd.Series(np.select([s.isin(keep), s.isin(known)], [s, OTHER], UNSEEN), index=s.index)
    counts = mapped.value_counts().reindex(labels, fill_value=0).to_numpy(dtype=float)
    return counts / max(len(s), 1)


def build_profile(frame: pd.DataFrame, numeric: list[str], categorical: list[str], predictions: np.ndarray) -> dict[str, Any]:
    prof: dict[str, Any] = {c: _num_profile(frame[c]) for c in numeric}
    prof.update({c: _cat_profile(frame[c]) for c in categorical})
    prof[PREDICTION] = _num_profile(pd.Series(predictions))
    return prof


def window_freq(profile: dict[str, Any], col: str, x: pd.Series) -> np.ndarray:
    p = profile[col]
    if p["kind"] == "numeric":
        return _num_freq(pd.to_numeric(x, errors="coerce").astype(float), np.asarray(p["edges"]))
    s = x.astype(object).where(x.notna(), MISSING).astype(str)
    return _cat_freq(s, p["categories"], set(p["known"]))


def psi(expected: np.ndarray, actual: np.ndarray) -> float:
    e = np.clip(np.asarray(expected, dtype=float), EPS, None)
    a = np.clip(np.asarray(actual, dtype=float), EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def score_window(profile: dict[str, Any], frame: pd.DataFrame, predictions: np.ndarray) -> dict[str, float]:
    out = {c: psi(np.asarray(profile[c]["freq"]), window_freq(profile, c, frame[c])) for c in profile if c != PREDICTION}
    out[PREDICTION] = psi(np.asarray(profile[PREDICTION]["freq"]), window_freq(profile, PREDICTION, pd.Series(predictions)))
    return out


def unseen_rate(profile: dict[str, Any], col: str, x: pd.Series) -> float:
    p = profile[col]
    if p["kind"] != "categorical":
        return 0.0
    s = x.astype(object).where(x.notna(), MISSING).astype(str)
    return float((~s.isin(set(p["known"]))).mean())


def calibrate_backtest(
    frame: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
    predict: Callable[[pd.DataFrame], np.ndarray],
    ref_weeks: int = 8,
    horizon_weeks: int = 8,
    step_weeks: int = 2,
    warn_mult: float = 1.5,
    crit_mult: float = 3.0,
) -> tuple[dict[str, dict[str, float]], int]:
    """Per-column thresholds from a rolling backtest over the training period.

    Slide an ``ref_weeks`` reference window through the (healthy) training data,
    score each of the next ``horizon_weeks`` weeks against it, and set thresholds
    from the largest PSI seen. This captures the drift a model normally sees
    between retrains, so alerts mean "more than usual", not "anything moved".
    """
    as_of = pd.to_datetime(frame["as_of"])
    week = pd.Timedelta(weeks=1)
    end = as_of.max()
    ref_end = as_of.min().normalize() + ref_weeks * week
    rows: list[dict[str, float]] = []
    while ref_end + week <= end:
        ref = frame[(as_of >= ref_end - ref_weeks * week) & (as_of < ref_end)]
        prof = build_profile(ref, numeric, categorical, predict(ref))
        for h in range(horizon_weeks):
            lo = ref_end + h * week
            if lo + week > end:
                break
            w = frame[(as_of >= lo) & (as_of < lo + week)]
            if len(w) >= 50:
                rows.append(score_window(prof, w, predict(w)))
        ref_end += step_weeks * week
    scores = pd.DataFrame(rows)
    return _thresholds(scores, warn_mult, crit_mult), len(rows)


def _thresholds(scores: pd.DataFrame, warn_mult: float, crit_mult: float) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for c in scores.columns:
        floor = float(scores[c].max())
        out[c] = {
            "noise_max": floor,
            "noise_median": float(scores[c].median()),
            "warn": max(0.05, warn_mult * floor),
            "critical": max(0.15, crit_mult * floor),
        }
    return out
