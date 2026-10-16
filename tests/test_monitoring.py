from __future__ import annotations

import io
from typing import Any

import numpy as np
import pandas as pd

from aucml.monitoring.alerts import Alert, ConsoleSink, SlackWebhookSink, emf_record
from aucml.monitoring.drift import PREDICTION, build_profile, calibrate_backtest, psi, score_window, unseen_rate
from aucml.monitoring.monitor import Incidents, score


def _frame(n: int, shift: float = 0.0, seed: int = 0, countries: tuple[str, ...] = ("DE", "FR", "IT")) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "as_of": pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.uniform(0, 280, n), unit="D"),
            "km": rng.normal(80_000 + shift, 20_000, n),
            "country": rng.choice(countries, n),
        }
    )


def _profile(df: pd.DataFrame) -> dict[str, Any]:
    return build_profile(df, ["km"], ["country"], df.km.to_numpy() / 10)


def test_psi_is_zero_for_identical_and_grows_with_shift() -> None:
    ref = _frame(5000)
    prof = _profile(ref)
    same = score_window(prof, ref, ref.km.to_numpy() / 10)
    assert same["km"] < 1e-9 and same[PREDICTION] < 1e-9
    small = score_window(prof, _frame(5000, 5_000, seed=1), np.zeros(5000))["km"]
    big = score_window(prof, _frame(5000, 30_000, seed=1), np.zeros(5000))["km"]
    assert small < big and big > 0.5
    assert psi(np.array([0.5, 0.5]), np.array([0.5, 0.5])) == 0.0


def test_unseen_categories_are_counted_separately() -> None:
    prof = _profile(_frame(3000))
    new = _frame(1000, countries=("DE", "PL"))
    assert 0.4 < unseen_rate(prof, "country", new.country) < 0.6
    assert unseen_rate(prof, "country", _frame(1000).country) == 0.0


def test_backtest_thresholds_do_not_fire_on_healthy_weeks() -> None:
    df = _frame(20_000)
    thr, windows = calibrate_backtest(df, ["km"], ["country"], lambda f: f.km.to_numpy() / 10)
    assert windows > 20
    healthy = _frame(600, seed=99)
    prof = _profile(df[df.as_of >= df.as_of.max() - pd.Timedelta(weeks=8)])
    s = score_window(prof, healthy, healthy.km.to_numpy() / 10)
    assert s["km"] < thr["km"]["warn"]
    shifted = _frame(600, 25_000, seed=99)
    assert score_window(prof, shifted, shifted.km.to_numpy() / 10)["km"] > thr["km"]["critical"]


def _manifest() -> dict[str, Any]:
    ref = _frame(4000)
    prof = _profile(ref)
    thr = {c: {"noise_max": 0.02, "noise_median": 0.01, "warn": 0.05, "critical": 0.15} for c in prof}
    return {
        "name": "m",
        "version": "v1",
        "columns": ["km", "country"],
        "categorical": ["country"],
        "primary_metric": "mape",
        "monitoring": {"profile": prof, "thresholds": thr, "reference_performance": 0.08},
        "slice_performance": {"country": [{"value": "DE", "model": 0.08}]},
    }


def test_score_raises_drift_quality_and_performance_alerts() -> None:
    m = _manifest()
    cur = _frame(800, 30_000, seed=3, countries=("DE", "PL"))
    cur["entity_id"] = [f"e{i}" for i in range(len(cur))]
    cur["prediction"] = 100.0
    labels = pd.DataFrame({"entity_id": cur.entity_id, "label": 100.0 / 1.3})  # 30% error
    rep = score(m, "w1", cur, labels, ["country"])
    kinds = {(a.check, a.subject, a.severity) for a in rep.alerts}
    assert ("drift", "km", "critical") in kinds
    assert ("data_quality", "country", "critical") in kinds
    assert ("performance", "overall", "critical") in kinds
    assert any(a.subject == "country=DE" for a in rep.alerts)


def _alert(sev: str, subject: str = "km") -> Alert:
    return Alert("m", "v1", "w", "drift", subject, sev, 1.0, 0.5, "x")  # type: ignore[arg-type]


def test_incidents_page_once_escalate_and_close() -> None:
    inc = Incidents(close_after=2)
    assert len(inc.update([_alert("warn")])[0]) == 1
    assert len(inc.update([_alert("warn")])[0]) == 0  # still open, no new page
    assert len(inc.update([_alert("critical")])[0]) == 1  # escalation pages
    assert inc.age("m:drift:km") == 3
    inc.update([])
    inc.update([])
    assert inc.age("m:drift:km") == 0  # closed after two quiet windows
    assert len(inc.update([_alert("warn")])[0]) == 1  # reopens


def test_sinks_and_emf() -> None:
    buf = io.StringIO()
    ConsoleSink(buf).send([_alert("critical")])
    assert "CRITICAL" in buf.getvalue()
    slack = SlackWebhookSink("https://hooks.example/x")
    assert slack.payload([_alert("warn")]) is None
    assert "1 model alert" in str(slack.payload([_alert("critical")]))
    rec = emf_record("aucml/monitoring", "m", {"psi.km": 0.2}, 1_700_000_000_000)
    assert rec["_aws"]["CloudWatchMetrics"][0]["Metrics"] == [{"Name": "psi.km", "Unit": "None"}]  # type: ignore[index]
    assert rec["psi.km"] == 0.2 and rec["Model"] == "m"
