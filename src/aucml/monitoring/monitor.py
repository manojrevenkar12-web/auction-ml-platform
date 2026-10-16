"""Score one monitoring window of captured inference traffic.

Inputs are what production actually has: the data-capture log (request,
assembled features, prediction) and, once they arrive, the delayed labels.
Thresholds come from the model's manifest, so every model is monitored by the
same code with its own calibrated limits.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aucml import metrics as M
from aucml.monitoring.alerts import Alert
from aucml.monitoring.drift import PREDICTION, score_window, unseen_rate

UNSEEN_WARN, UNSEEN_CRIT = 0.02, 0.10
NULL_SHIFT_WARN = 0.10
PERF_WARN, PERF_CRIT = 0.25, 0.50  # relative degradation of the primary metric
MIN_LABELLED = 100


def read_capture(paths: list[Path], start: pd.Timestamp, end: pd.Timestamp, model: str) -> pd.DataFrame:
    rows = []
    for p in paths:
        with p.open() as f:
            for line in f:
                r = json.loads(line)
                if r["model"] != model:
                    continue
                ts = pd.Timestamp(r["as_of"])
                if start <= ts < end:
                    rows.append({"event_id": r["event_id"], "entity_id": r["request"].get("request_id"), "as_of": ts, "version": r["version"], "prediction": r["prediction"], **r["features"]})
    return pd.DataFrame(rows)


@dataclass
class WindowReport:
    window: str
    rows: int
    labelled: int
    psi: dict[str, float]
    unseen: dict[str, float]
    performance: dict[str, Any]
    alerts: list[Alert] = field(default_factory=list)


def _sev(value: float, warn: float, crit: float) -> str | None:
    if value >= crit:
        return "critical"
    if value >= warn:
        return "warn"
    return None


def score(
    manifest: dict[str, Any],
    window: str,
    captured: pd.DataFrame,
    labels: pd.DataFrame | None = None,
    slice_columns: list[str] | None = None,
) -> WindowReport:
    """``labels``: columns entity_id, label (delayed ground truth for this window)."""
    name, version = manifest["name"], manifest["version"]
    mon = manifest["monitoring"]
    profile, thr = mon["profile"], mon["thresholds"]
    alerts: list[Alert] = []
    preds = captured["prediction"].to_numpy(dtype=float)
    psi = score_window(profile, captured, preds)
    for col, val in psi.items():
        t = thr[col]
        sev = _sev(val, t["warn"], t["critical"])
        if sev:
            check = "prediction_drift" if col == PREDICTION else "drift"
            alerts.append(Alert(name, version, window, check, col, sev, val, t[sev], f"PSI {val:.3f} (healthy max {t['noise_max']:.3f}, {sev} at {t[sev]:.3f})"))  # type: ignore[arg-type]

    unseen = {c: unseen_rate(profile, c, captured[c]) for c in manifest["categorical"]}
    for col, val in unseen.items():
        sev = _sev(val, UNSEEN_WARN, UNSEEN_CRIT)
        if sev:
            alerts.append(Alert(name, version, window, "data_quality", col, sev, val, UNSEEN_WARN if sev == "warn" else UNSEEN_CRIT, f"{100 * val:.1f}% of requests carry a category never seen in training"))  # type: ignore[arg-type]
    for col in manifest["columns"]:
        ref_null = profile[col]["null_rate"]
        now = float(captured[col].isna().mean())
        if now - ref_null >= NULL_SHIFT_WARN:
            alerts.append(Alert(name, version, window, "data_quality", col, "warn", now, ref_null + NULL_SHIFT_WARN, f"null rate {100 * now:.1f}% vs {100 * ref_null:.1f}% in training"))

    perf: dict[str, Any] = {}
    if labels is not None and len(labels):
        pm = manifest["primary_metric"]
        joined = captured.merge(labels.dropna(subset=["label"]), on="entity_id", how="inner")
        perf["labelled"] = len(joined)
        if len(joined) >= MIN_LABELLED:
            fn, direction = M.METRICS[pm]
            y = joined["label"].to_numpy(dtype=float)
            p = joined["prediction"].to_numpy(dtype=float)
            ref = float(mon["reference_performance"])
            val = fn(y, p)
            perf[pm] = val
            degr = (val - ref) / ref if direction == "lower" else (ref - val) / ref
            sev = _sev(degr, PERF_WARN, PERF_CRIT)
            if sev:
                alerts.append(Alert(name, version, window, "performance", "overall", sev, val, ref, f"{pm} {val:.4f} vs {ref:.4f} at release ({100 * degr:+.0f}%)"))  # type: ignore[arg-type]
            perf["slices"] = {}
            for col in slice_columns or []:
                ref_rows = {r["value"]: r["model"] for r in manifest.get("slice_performance", {}).get(col, []) if r["model"] is not None}
                for v, grp in joined.groupby(joined[col].astype(str)):
                    if len(grp) < MIN_LABELLED:
                        continue
                    sv = fn(grp["label"].to_numpy(dtype=float), grp["prediction"].to_numpy(dtype=float))
                    perf["slices"][f"{col}={v}"] = {"n": len(grp), pm: sv}
                    sref = ref_rows.get(str(v), ref)
                    d = (sv - sref) / sref if direction == "lower" else (sref - sv) / sref
                    sev = _sev(d, PERF_WARN, PERF_CRIT)
                    if sev:
                        alerts.append(Alert(name, version, window, "performance", f"{col}={v}", sev, sv, sref, f"{pm} {sv:.4f} vs {sref:.4f} at release ({100 * d:+.0f}%, n={len(grp)})"))  # type: ignore[arg-type]
    return WindowReport(window, len(captured), int(perf.get("labelled", 0)), psi, unseen, perf, alerts)


class Incidents:
    """Turns repeated alerts into incidents so on-call is paged once, not weekly.

    An alert key (model, check, subject) opens an incident on first sight and
    pages; while it keeps firing it stays open without paging again unless it
    escalates warn -> critical; two quiet windows close it.
    """

    def __init__(self, close_after: int = 2) -> None:
        self.close_after = close_after
        self.open: dict[str, dict[str, Any]] = {}

    def update(self, alerts: list[Alert]) -> tuple[list[Alert], dict[str, dict[str, Any]]]:
        pages: list[Alert] = []
        seen = set()
        for a in alerts:
            k = a.key()
            seen.add(k)
            inc = self.open.get(k)
            if inc is None:
                self.open[k] = {"opened": a.window, "severity": a.severity, "windows": 1, "quiet": 0}
                pages.append(a)
            else:
                inc["windows"] += 1
                inc["quiet"] = 0
                if a.severity == "critical" and inc["severity"] == "warn":
                    inc["severity"] = "critical"
                    pages.append(a)
        for k in list(self.open):
            if k not in seen:
                self.open[k]["quiet"] += 1
                if self.open[k]["quiet"] >= self.close_after:
                    del self.open[k]
        return pages, self.open

    def age(self, key: str) -> int:
        inc = self.open.get(key)
        return int(inc["windows"]) if inc else 0

    def resolve(self, prefix: str) -> None:
        """Close incidents for a model after a new version is deployed."""
        for k in [k for k in self.open if k.startswith(prefix)]:
            del self.open[k]


def summarize(reports: list[WindowReport]) -> pd.DataFrame:
    rows = []
    for r in reports:
        rows.append(
            {
                "window": r.window,
                "rows": r.rows,
                "labelled": int(r.performance.get("labelled", r.labelled)) if r.performance else r.labelled,
                "pred_psi": round(r.psi.get(PREDICTION, np.nan), 3),
                "warn": sum(a.severity == "warn" for a in r.alerts),
                "critical": sum(a.severity == "critical" for a in r.alerts),
                "alerts": "; ".join(sorted({f"{a.check}/{a.subject}" for a in r.alerts})),
            }
        )
    return pd.DataFrame(rows)
