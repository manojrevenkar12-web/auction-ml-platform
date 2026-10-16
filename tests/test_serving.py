from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from aucml.config import Catalog
from aucml.featurestore import OfflineStore, OnlineStore
from aucml.gate import parity_check
from aucml.packaging import package, unpack
from aucml.pipeline import RunResult, Workspace
from aucml.serving.app import create_app
from aucml.serving.inference import handle, model_fn


@pytest.fixture(scope="module")
def model_dir(ws: Workspace, catalog: Catalog, price_run: RunResult) -> Path:
    tar, _ = package(price_run.dir, price_run.spec, catalog, ws.root / "pkg_serving")
    return unpack(tar, ws.root / "unpacked_serving")


@pytest.fixture(scope="module")
def online(ws: Workspace, catalog: Catalog) -> OnlineStore:
    store = OnlineStore(ws.store / "serving_online.sqlite")
    groups = [g for g, v in catalog.groups.items() if v.kind == "batch"]
    store.materialize(OfflineStore(ws.store), groups, datetime(2025, 12, 1))
    return store


def _payload(price_run: RunResult, n: int = 3) -> list[dict[str, object]]:
    raw = price_run.test_raw.head(n)
    fields = [
        "make",
        "model",
        "fuel",
        "segment",
        "first_registration",
        "mileage_km",
        "damage_score",
        "power_kw",
        "seller_id",
        "seller_country",
        "as_of",
    ]
    recs: list[dict[str, object]] = json.loads(raw[fields].to_json(orient="records", date_format="iso"))
    return recs


def test_http_contract(model_dir: Path, online: OnlineStore, price_run: RunResult, tmp_path: Path) -> None:
    handler = model_fn(model_dir, lookup=online)
    handler.capture_dir = tmp_path / "capture"
    client = TestClient(create_app(handler=handler))
    assert client.get("/ping").status_code == 200
    recs = _payload(price_run)
    recs[0]["request_id"] = "r-1"
    r = client.post("/invocations", json={"instances": recs})
    assert r.status_code == 200
    body = r.json()
    assert len(body["predictions"]) == 3 and all(p > 0 for p in body["predictions"])
    lines = (next((tmp_path / "capture").glob("*.jsonl"))).read_text().splitlines()
    first = json.loads(lines[0])
    assert first["request"]["request_id"] == "r-1" and "listing__age_months" in first["features"]


def test_bad_requests_get_400(model_dir: Path, online: OnlineStore, price_run: RunResult) -> None:
    client = TestClient(create_app(handler=model_fn(model_dir, lookup=online)))
    rec = _payload(price_run, 1)[0]
    del rec["mileage_km"]
    r = client.post("/invocations", json={"instances": [rec]})
    assert r.status_code == 400 and "mileage_km" in r.text
    rec = _payload(price_run, 1)[0] | {"mileage_km": "lots"}
    assert client.post("/invocations", json={"instances": [rec]}).status_code == 400
    assert client.post("/invocations", content=b"not json", headers={"content-type": "application/json"}).status_code == 400


def test_parity_check_catches_training_serving_skew(ws: Workspace, catalog: Catalog, price_run: RunResult, model_dir: Path) -> None:
    clean = parity_check(price_run.spec, catalog, ws, model_dir, price_run.test_raw, price_run.test, days=2)
    assert clean["feature_parity"] == 1.0 and clean["prediction_mismatches"] == 0
    # Simulate a serving image whose transform drifted from training: age in years, not months.
    skewed = ws.root / "skewed_model"
    shutil.rmtree(skewed, ignore_errors=True)
    shutil.copytree(model_dir, skewed)
    t = skewed / "feature_repo" / "transforms.py"
    t.write_text(t.read_text().replace('"age_months": age,', '"age_months": age / 12,'))
    bad = parity_check(price_run.spec, catalog, ws, skewed, price_run.test_raw, price_run.test, days=2)
    assert bad["feature_parity"] < 0.05
    assert bad["prediction_mismatches"] > 0
    assert any("listing__age_months" in e for e in bad["examples"])


def test_features_for_unknown_seller_degrade_gracefully(model_dir: Path, online: OnlineStore, price_run: RunResult) -> None:
    handler = model_fn(model_dir, lookup=online)
    rec = _payload(price_run, 1)[0] | {"seller_id": "S9999", "seller_country": "PL"}
    out = json.loads(handle(json.dumps({"instances": [rec]}), handler)[0])
    assert out["predictions"][0] > 0
    assert out["feature_completeness"][0] < 1.0
