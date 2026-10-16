"""Fit-to-serve gate: the decision on whether a model may take traffic.

A candidate is approved only if every check passes:

1. pre-ship review does not block
2. primary metric meets the spec threshold
3. beats the spec baseline by the required relative margin
4. not worse than the current champion on the same rows (paired bootstrap)
5. no slice regresses beyond tolerance (vs champion, or vs threshold)
6. the packaged artifact, loaded in a fresh process, predicts bit-identically
7. training/serving parity: features assembled through the serving path from
   the online store match the training-time features
8. p95 single-request latency within budget
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from aucml import metrics as M
from aucml.config import Catalog, ModelSpec, column
from aucml.featurestore import OfflineStore, OnlineStore, assemble
from aucml.packaging import unpack
from aucml.pipeline import BASE_COLUMNS, FittedModel, Workspace
from aucml.review import Finding, review, verdict
from aucml.serving.inference import handle, model_fn, parse_records

SMOKE = """
import json, sys
import joblib, pandas as pd, numpy as np
m = joblib.load(sys.argv[1] + "/model.joblib")
x = pd.read_parquet(sys.argv[2])
np.save(sys.argv[3], m.predict(x))
"""


def _check(name: str, passed: bool, detail: str, **extra: Any) -> dict[str, Any]:
    return {"check": name, "passed": bool(passed), "detail": detail, **extra}


def _slice_table(spec: ModelSpec, test: pd.DataFrame, y: np.ndarray, p: np.ndarray) -> dict[str, dict[str, tuple[int, float]]]:
    fn = M.METRICS[spec.gate.primary_metric][0]
    out: dict[str, dict[str, tuple[int, float]]] = {}
    for ref in spec.slices:
        col = test[column(ref)].astype(str).to_numpy()
        out[ref] = {}
        for v in np.unique(col):
            ii = np.where(col == v)[0]
            if len(ii) >= spec.gate.min_slice_rows:
                out[ref][str(v)] = (len(ii), fn(y[ii], p[ii]))
    return out


def champion_predictions(champion_dir: Path, catalog: Catalog, ws: Workspace, test_raw: pd.DataFrame) -> np.ndarray:
    manifest = json.loads((champion_dir / "manifest.json").read_text())
    model: FittedModel = joblib.load(champion_dir / "model.joblib")
    feats = assemble(catalog, test_raw.reset_index(drop=True), manifest["features"], OfflineStore(ws.store))
    return model.predict(feats)


def parity_check(
    spec: ModelSpec, catalog: Catalog, ws: Workspace, model_dir: Path, test_raw: pd.DataFrame, test: pd.DataFrame, days: int = 5
) -> dict[str, Any]:
    """Replay test-window requests through the serving path against the online store."""
    as_of = pd.to_datetime(test_raw.as_of).reset_index(drop=True)
    raw = test_raw.reset_index(drop=True)
    day_list = sorted(as_of.dt.normalize().unique())
    picks = [day_list[int(i)] for i in np.linspace(0, len(day_list) - 1, min(days, len(day_list)))]
    online_path = ws.store / "parity_online.sqlite"
    handler = model_fn(model_dir, lookup=OnlineStore(online_path))
    _, batch = handler.manifest["on_demand_groups"], handler.manifest["batch_groups"]
    cols = handler.manifest["columns"]
    fields = handler.manifest["request_schema"]["required"] + handler.manifest["request_schema"]["optional"]
    rows = mismatched = pred_mismatch = 0
    examples: list[str] = []
    for d in picks:
        OnlineStore(online_path).materialize(OfflineStore(ws.store), batch, pd.Timestamp(d).to_pydatetime())
        idx = np.where(as_of.dt.normalize() == d)[0]
        req = raw.iloc[idx][[f for f in fields if f in raw]].copy()
        records = json.loads(req.to_json(orient="records", date_format="iso"))
        parsed = parse_records(records, handler.manifest)
        served = assemble(handler.catalog, parsed, handler.manifest["features"], handler.lookup, handler.transforms)[cols]
        offline = test.iloc[idx][cols].reset_index(drop=True)
        for c in cols:
            a, b = served[c].reset_index(drop=True), offline[c]
            if c in handler.model.categorical:
                eq = (a.astype(str) == b.astype(str)) | (a.isna() & b.isna())
            else:
                af, bf = a.astype(float).to_numpy(), b.astype(float).to_numpy()
                eq = pd.Series(np.isclose(af, bf, rtol=1e-9, atol=1e-9) | (np.isnan(af) & np.isnan(bf)))
            if not eq.all() and len(examples) < 5:
                examples.append(f"{c}: {int((~eq).sum())} rows differ on {pd.Timestamp(d):%Y-%m-%d}")
        row_eq = np.ones(len(idx), dtype=bool)
        for c in cols:
            a, b = served[c].reset_index(drop=True), offline[c]
            if c in handler.model.categorical:
                row_eq &= ((a.astype(str) == b.astype(str)) | (a.isna() & b.isna())).to_numpy()
            else:
                af, bf = a.astype(float).to_numpy(), b.astype(float).to_numpy()
                row_eq &= np.isclose(af, bf, rtol=1e-9, atol=1e-9) | (np.isnan(af) & np.isnan(bf))
        body = json.dumps({"instances": records})
        served_pred = np.array(json.loads(handle(body, handler)[0])["predictions"])
        offline_pred = handler.model.predict(test.iloc[idx])
        pred_mismatch += int((~np.isclose(served_pred, offline_pred, rtol=1e-9)).sum())
        rows += len(idx)
        mismatched += int((~row_eq).sum())
    return {
        "rows": rows,
        "days": [f"{pd.Timestamp(d):%Y-%m-%d}" for d in picks],
        "feature_parity": 1 - mismatched / max(rows, 1),
        "prediction_mismatches": pred_mismatch,
        "examples": examples,
    }


def latency(model_dir: Path, ws: Workspace, test_raw: pd.DataFrame, n: int = 200) -> dict[str, float]:
    handler = model_fn(model_dir, lookup=OnlineStore(ws.store / "parity_online.sqlite"))
    fields = handler.manifest["request_schema"]["required"] + handler.manifest["request_schema"]["optional"]
    sample = test_raw.sample(min(n, len(test_raw)), random_state=0)[[f for f in fields if f in test_raw]]
    records = json.loads(sample.to_json(orient="records", date_format="iso"))
    times = []
    for r in records:
        t = time.perf_counter()
        handle(json.dumps({"instances": [r]}), handler)
        times.append((time.perf_counter() - t) * 1000)
    arr = np.array(times[5:])  # drop warm-up
    return {"p50_ms": float(np.percentile(arr, 50)), "p95_ms": float(np.percentile(arr, 95)), "n": float(len(arr))}


def decide(
    spec: ModelSpec,
    catalog: Catalog,
    ws: Workspace,
    run_dir: Path,
    package: Path,
    champion_package: Path | None = None,
) -> tuple[dict[str, Any], list[Finding]]:
    g = spec.gate
    train = pd.read_parquet(run_dir / "train.parquet")
    test = pd.read_parquet(run_dir / "test.parquet")
    test_raw = pd.read_parquet(run_dir / "test_raw.parquet")
    model: FittedModel = joblib.load(run_dir / "model.joblib")
    evaluation = json.loads((run_dir / "evaluation.json").read_text())
    pm = g.primary_metric
    y = test.label.astype(float).to_numpy()
    p = model.predict(test)
    checks: list[dict[str, Any]] = []

    findings = review(spec, catalog, train, test)
    checks.append(_check("pre_ship_review", verdict(findings) != "block", f"verdict={verdict(findings)}"))

    value = float(evaluation["metrics"][pm])
    ok = value <= g.threshold if g.direction == "lower" else value >= g.threshold
    checks.append(_check("threshold", ok, f"{pm}={value:.4f} vs threshold {g.threshold}", value=value))

    base = float(evaluation["baseline"][pm])
    rel = M.relative_improvement(pm, value, base)
    checks.append(
        _check(
            "beats_baseline",
            rel >= g.min_rel_improvement_vs_baseline,
            f"{pm} {value:.4f} vs baseline {base:.4f}: {100 * rel:.1f}% better"
            + (f" above chance ({M.CHANCE[pm]})" if pm in M.CHANCE else "")
            + f" (need {100 * g.min_rel_improvement_vs_baseline:.0f}%)",
        )
    )

    with tempfile.TemporaryDirectory() as tmp:
        cand_dir = unpack(package, Path(tmp) / "candidate")
        if champion_package is not None:
            champ_dir = unpack(champion_package, Path(tmp) / "champion")
            cm = json.loads((champ_dir / "manifest.json").read_text())
            pc = champion_predictions(champ_dir, catalog, ws, test_raw)
            pb = M.paired_bootstrap(pm, y, p, pc)
            half_width = (pb["ci_high"] - pb["ci_low"]) / 2
            power_note = (
                f"; underpowered: CI half-width {half_width:.4f} exceeds the margin, so non-inferiority can only be shown "
                "by a clearly better challenger or a longer test window"
                if half_width > g.champion_margin
                else ""
            )
            checks.append(
                _check(
                    "vs_champion",
                    pb["ci_low"] >= -g.champion_margin,
                    f"vs {cm['version']}: improvement {pb['delta']:+.4f} {pm} (95% CI {pb['ci_low']:+.4f} .. {pb['ci_high']:+.4f}); "
                    f"non-inferiority margin {g.champion_margin}{power_note}",
                    champion=cm["version"],
                    champion_metric=M.METRICS[pm][0](y, pc),
                    ci_half_width=half_width,
                    **pb,
                )
            )
            cand_s, champ_s = _slice_table(spec, test, y, p), _slice_table(spec, test, y, pc)
            worse = []
            for ref, vals in cand_s.items():
                for v, (n, m) in vals.items():
                    if v in champ_s.get(ref, {}):
                        cm_val = champ_s[ref][v][1]
                        if M.improvement(pm, m, cm_val) < -g.slice_tolerance * abs(cm_val):
                            worse.append(f"{ref}={v} {m:.4f} vs {cm_val:.4f} (n={n})")
            checks.append(_check("slices_vs_champion", not worse, f"regressed slices: {worse}" if worse else "no slice regressed beyond tolerance"))
        else:
            checks.append(_check("vs_champion", True, "no champion in this lane: first deployment"))
            limit = g.threshold * (1 + g.slice_tolerance) if g.direction == "lower" else g.threshold * (1 - g.slice_tolerance)
            worse = [
                f"{ref}={v} {m:.4f} (n={n})"
                for ref, vals in _slice_table(spec, test, y, p).items()
                for v, (n, m) in vals.items()
                if (m > limit if g.direction == "lower" else m < limit)
            ]
            checks.append(
                _check("slices_vs_threshold", not worse, f"slices past {limit:.4f}: {worse}" if worse else f"all slices within {limit:.4f}")
            )

        x_path = Path(tmp) / "x.parquet"
        test.to_parquet(x_path)
        out = Path(tmp) / "pred.npy"
        subprocess.run([sys.executable, "-c", SMOKE, str(cand_dir), str(x_path), str(out)], check=True)
        diff = float(np.max(np.abs(np.load(out) - p)))
        checks.append(_check("package_reproduces", diff == 0.0, f"max |packaged - in-memory| = {diff:.3g} over {len(p)} rows"))

        par = parity_check(spec, catalog, ws, cand_dir, test_raw, test)
        checks.append(
            _check(
                "serving_parity",
                par["feature_parity"] >= g.min_parity and par["prediction_mismatches"] == 0,
                f"{100 * par['feature_parity']:.2f}% of {par['rows']} replayed requests match training features; "
                f"{par['prediction_mismatches']} prediction mismatches",
                **par,
            )
        )
        lat = latency(cand_dir, ws, test_raw)
        checks.append(
            _check(
                "latency",
                lat["p95_ms"] <= g.max_latency_p95_ms,
                f"p50 {lat['p50_ms']:.1f} ms, p95 {lat['p95_ms']:.1f} ms (budget {g.max_latency_p95_ms} ms)",
                **lat,
            )
        )

    decision = {
        "model": spec.name,
        "version": json.loads((run_dir / "run.json").read_text())["run_id"],
        "decided_at": datetime.now().isoformat(timespec="seconds"),
        "decision": "approve" if all(c["passed"] for c in checks) else "reject",
        "checks": checks,
        "rows": {"train": len(train), "test": len(test)},
        "base_columns": BASE_COLUMNS,
    }
    return decision, findings


def to_markdown(decision: dict[str, Any]) -> str:
    rows = decision["rows"]
    lines = [
        f"# Fit-to-serve decision: `{decision['model']}` {decision['version']}",
        "",
        f"**{decision['decision'].upper()}** ({decision['decided_at']}; train {rows['train']:,} rows, test {rows['test']:,} rows)",
        "",
        "| check | result | detail |",
        "|---|---|---|",
    ]
    for c in decision["checks"]:
        lines.append(f"| {c['check']} | {'PASS' if c['passed'] else 'FAIL'} | {c['detail']} |")
    return "\n".join(lines) + "\n"
