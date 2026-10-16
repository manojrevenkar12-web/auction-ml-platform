"""Replay six months of production for every model on the platform.

1. Release v1 of each model (cutoff 2025-11-01) through review -> gate -> prod.
2. For every day from 2026-01-05 to 2026-06-28: refresh the online store at
   midnight, send that day's requests through the prod model's serving path
   (data capture on).
3. Every Sunday night: score the week's captured traffic for drift and data
   quality, and the previous week's traffic against labels that have arrived.
4. Alerts become incidents (one page per new problem, not one per week).
5. Retrain on a critical performance alert, or on a critical data-quality
   incident open for 5 weeks: rolling 8-month window, gate the challenger
   against the prod champion on the same recent rows, promote if it passes.

Everything written to docs/results/ comes from this script.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from aucml.cli import release
from aucml.config import Catalog, ModelSpec, discover_specs
from aucml.featurestore import OfflineStore, OnlineStore, materialize_offline
from aucml.gate import to_markdown
from aucml.monitoring.alerts import EmfSink, JsonlSink
from aucml.monitoring.monitor import Incidents, read_capture, score, summarize
from aucml.packaging import unpack
from aucml.pipeline import Workspace, with_window
from aucml.registry import Registry
from aucml.serving.inference import handle, model_fn

ROOT = Path(__file__).resolve().parents[1]
START, END = date(2026, 1, 5), date(2026, 6, 29)
APPROVER = "on-call (simulated)"
PERSISTENT_WEEKS = 5
BACKOFF_WEEKS = 4  # after a rejected retrain, wait before retrying on the same trigger
OUT = ROOT / "docs" / "results"


def requests_for(spec: ModelSpec, ws: Workspace, day: date) -> pd.DataFrame:
    con = duckdb.connect(str(ws.warehouse), read_only=True)
    try:
        df = con.execute(f"SELECT * FROM ({spec.sql()}) q WHERE as_of >= ? AND as_of < ?", [day, day + timedelta(days=1)]).df()
    finally:
        con.close()
    return df


def labels_for(spec: ModelSpec, ws: Workspace, start: date, end: date, known_by: datetime) -> pd.DataFrame:
    con = duckdb.connect(str(ws.warehouse), read_only=True)
    try:
        df = con.execute(
            f"SELECT entity_id, label FROM ({spec.sql()}) q WHERE as_of >= ? AND as_of < ? AND label_known_at < ? AND label IS NOT NULL",
            [start, end, known_by],
        ).df()
    finally:
        con.close()
    df["label"] = df["label"].astype(float)
    return df


class Deployed:
    """The prod model of one spec, unpacked and loaded like a serving container."""

    def __init__(self, spec: ModelSpec, ws: Workspace, online: OnlineStore) -> None:
        self.spec, self.ws, self.online = spec, ws, online
        self.manifests: dict[str, dict[str, Any]] = {}
        self.load()

    def load(self) -> None:
        reg = Registry(self.ws.registry)
        version = reg.lanes(self.spec.name)["prod"]
        d = unpack(reg.vdir(self.spec.name, version) / "model.tar.gz", self.ws.root / ".serving" / self.spec.name)
        self.manifests[version] = json.loads((d / "manifest.json").read_text())
        self.handler = model_fn(d, lookup=self.online)
        self.handler.capture_dir = self.ws.root / "capture"
        self.version = version

    def serve(self, raw: pd.DataFrame) -> int:
        if raw.empty:
            return 0
        fields = self.handler.manifest["request_schema"]["required"] + self.handler.manifest["request_schema"]["optional"]
        req = raw[[f for f in fields if f in raw]].copy()
        req["request_id"] = raw["entity_id"].to_numpy()
        body = json.dumps({"instances": json.loads(req.to_json(orient="records", date_format="iso"))})
        handle(body, self.handler)
        return len(req)


def main() -> None:
    t0 = time.perf_counter()
    ws = Workspace(ROOT)
    cat = Catalog.load(ROOT / "feature_repo")
    for d in ("runs", "registry", "capture", ".serving", "monitoring"):
        shutil.rmtree(ROOT / d, ignore_errors=True)
    OUT.mkdir(parents=True, exist_ok=True)
    for stale in [*OUT.glob("gate_*.md"), *OUT.glob("monitoring_*.md")]:
        stale.unlink()
    print(materialize_offline(cat, ws.warehouse, ws.store, "2025-01-01", "2026-07-01"))

    specs = discover_specs(ROOT / "models")
    reg = Registry(ws.registry)
    summary: dict[str, Any] = {"releases": [], "retrains": [], "weeks": {}}
    for spec in specs:
        out = release(spec, cat, ws, lane="prod", approved_by=APPROVER)
        summary["releases"].append({"model": spec.name, "version": out["version"], "decision": out["decision"]["decision"], "lanes": out["promoted"]})
        (OUT / f"gate_{spec.name}_v1.md").write_text(to_markdown(out["decision"]))
        print(spec.name, out["decision"]["decision"], out["promoted"])
        if "prod" not in out["promoted"]:
            sys.exit(f"{spec.name} v1 did not reach prod; see docs/results/gate_{spec.name}_v1.md")

    online = OnlineStore(ws.store / "online.sqlite")
    batch_groups = [g for g, v in cat.groups.items() if v.kind == "batch"]
    deployed = {s.name: Deployed(s, ws, online) for s in specs}
    alerts_sink = JsonlSink(ROOT / "monitoring" / "alerts.jsonl")
    pages_sink = JsonlSink(ROOT / "monitoring" / "pages.jsonl")
    incidents = {s.name: Incidents() for s in specs}  # one tracker per model
    pages_per_week: dict[str, list[int]] = {s.name: [] for s in specs}
    rejected_at: dict[str, date] = {}
    emf = EmfSink(ROOT / "monitoring" / "emf.jsonl")
    reports: dict[str, list[Any]] = {s.name: [] for s in specs}
    served = dict.fromkeys(deployed, 0)

    day = START
    while day < END:
        online.materialize(OfflineStore(ws.store), batch_groups, datetime.combine(day, datetime.min.time()))
        for name, dep in deployed.items():
            served[name] += dep.serve(requests_for(dep.spec, ws, day))
        if day.weekday() == 6:  # Sunday night: weekly monitoring
            week_start = day - timedelta(days=6)
            now = datetime.combine(day + timedelta(days=1), datetime.min.time())
            for name, dep in deployed.items():
                cap_paths = sorted((ROOT / "capture").glob(f"{name}-*.jsonl"))
                cur = read_capture(cap_paths, pd.Timestamp(week_start), pd.Timestamp(now), name)
                prev_start = week_start - timedelta(days=7)
                prev = read_capture(cap_paths, pd.Timestamp(prev_start), pd.Timestamp(week_start), name)
                cols = [c.replace(".", "__", 1) for c in dep.spec.slices]
                drift = score(dep.handler.manifest, f"{week_start:%Y-%m-%d}", cur)
                perf_alerts: list[Any] = []
                perf: dict[str, Any] = {}
                served_by = None
                if not prev.empty:
                    labels = labels_for(dep.spec, ws, prev_start, week_start, now)
                    # judge last week's traffic against the version that actually served it
                    served_by = str(prev["version"].mode().iloc[0])
                    pr = score(dep.manifests[served_by], f"{prev_start:%Y-%m-%d}", prev, labels, cols)
                    perf_alerts = [a for a in pr.alerts if a.check == "performance"]
                    perf = pr.performance
                alerts = drift.alerts + perf_alerts
                alerts_sink.send(alerts)
                pages, _ = incidents[name].update(alerts)
                pages_sink.send(pages)
                pages_per_week[name].append(len(pages))
                metrics = {f"psi.{k}": v for k, v in drift.psi.items()}
                if dep.spec.gate.primary_metric in perf:
                    metrics[f"perf.{dep.spec.gate.primary_metric}"] = perf[dep.spec.gate.primary_metric]
                emf.write_scores(name, metrics, int(now.timestamp() * 1000))
                drift.alerts = alerts
                drift.performance = perf
                reports[name].append(drift)
                # Retrain policy:
                #  * a critical performance alert on traffic the current version served, or
                #  * a critical data-quality incident open for 5 weeks (long enough that the
                #    new population is inside the next training window)
                crit = [a for a in perf_alerts if a.severity == "critical"] if served_by == dep.version else []
                crit += [
                    a
                    for a in drift.alerts
                    if a.check == "data_quality" and a.severity == "critical" and incidents[name].age(a.key()) >= PERSISTENT_WEEKS
                ]
                backing_off = name in rejected_at and (week_start - rejected_at[name]).days < 7 * BACKOFF_WEEKS
                if crit and not backing_off:
                    cutoff = week_start - timedelta(days=21)
                    challenger = with_window(dep.spec, cutoff - timedelta(days=243), cutoff, week_start, now)
                    res = release(challenger, cat, ws, lane="prod", approved_by=APPROVER)
                    tag = f"{name}_{week_start:%Y%m%d}"
                    (OUT / f"gate_{tag}.md").write_text(to_markdown(res["decision"]))
                    summary["retrains"].append(
                        {
                            "model": name,
                            "triggered_by": [f"{a.subject}: {a.message}" for a in crit],
                            "week": f"{week_start:%Y-%m-%d}",
                            "train_window": [str(cutoff - timedelta(days=243)), str(cutoff)],
                            "test_window": [str(cutoff), str(week_start)],
                            "challenger": res["version"],
                            "champion": res["champion"],
                            "decision": res["decision"]["decision"],
                            "lanes": res["promoted"],
                            "checks": {c["check"]: c["detail"] for c in res["decision"]["checks"]},
                        }
                    )
                    print(f"{week_start} {name}: retrain -> {res['decision']['decision']} {res['promoted']}")
                    if res["decision"]["decision"] != "approve":
                        rejected_at[name] = week_start
                    if "prod" in res["promoted"]:
                        rejected_at.pop(name, None)
                        dep.load()
                        incidents[name].resolve(f"{name}:")
        day += timedelta(days=1)

    for name, reps in reports.items():
        table = summarize(reps)
        perf_col = []
        pm = deployed[name].spec.gate.primary_metric
        for r in reps:
            perf_col.append(round(r.performance.get(pm, float("nan")), 4) if r.performance else float("nan"))
        table.insert(3, f"prev_week_{pm}", perf_col)
        table.insert(table.columns.get_loc("critical") + 1, "pages", pages_per_week[name])
        summary["weeks"][name] = table.to_dict(orient="records")
        (OUT / f"monitoring_{name}.md").write_text(f"# Weekly monitoring: `{name}`\n\n" + table.to_markdown(index=False) + "\n")
    summary["requests_served"] = served
    summary["lanes"] = {s.name: reg.lanes(s.name) for s in specs}
    summary["runtime_s"] = round(time.perf_counter() - t0, 1)
    (OUT / "production_replay.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: summary[k] for k in ("requests_served", "runtime_s")}, indent=2))


if __name__ == "__main__":
    main()
