from __future__ import annotations

import shutil

import numpy as np
import pandas as pd
import pytest
import yaml

from aucml import review
from aucml.config import Catalog, ModelSpec
from aucml.packaging import package
from aucml.pipeline import RunResult, Workspace, extract, run, split, validate

from .conftest import spec_of


def test_time_split_respects_cutoff_and_label_arrival(ws: Workspace) -> None:
    spec = spec_of(ws, "transport_cost/model.yaml")
    train, test = split(spec, extract(spec, ws))
    cutoff = pd.Timestamp(spec.split.cutoff)
    assert (pd.to_datetime(train.as_of) < cutoff).all()
    # transport labels arrive days after the quote: late ones must be dropped
    assert (pd.to_datetime(train.label_known_at) < cutoff).all()
    assert (pd.to_datetime(test.as_of) >= cutoff).all()


def test_validation_blocks_bad_rows(ws: Workspace, catalog: Catalog) -> None:
    spec = spec_of(ws, "price_estimator/model.yaml")
    df = extract(spec, ws)
    assert all(c.passed for c in validate(spec, catalog, df))
    bad = df.copy()
    bad.loc[bad.index[:1000], "mileage_km"] = np.nan
    bad.loc[bad.index[0], "label_known_at"] = bad.loc[bad.index[0], "as_of"] - pd.Timedelta(days=1)
    failed = {c.name for c in validate(spec, catalog, bad) if not c.passed}
    assert {"null_rate:mileage_km", "label_after_prediction"} <= failed


def test_runs_are_reproducible(ws: Workspace, catalog: Catalog, price_run: RunResult, tmp_path: object) -> None:
    again = run(price_run.spec, catalog, ws)
    assert again.run_id == price_run.run_id
    np.testing.assert_array_equal(again.model.predict(again.test), price_run.model.predict(price_run.test))


def test_package_is_byte_reproducible(ws: Workspace, catalog: Catalog, price_run: RunResult, tmp_path: pytest.TempPathFactory) -> None:
    _, m1 = package(price_run.dir, price_run.spec, catalog, ws.root / "pkgA")
    _, m2 = package(price_run.dir, price_run.spec, catalog, ws.root / "pkgB")
    assert m1["package_sha256"] == m2["package_sha256"]
    assert m1["request_schema"]["required"]
    assert "thresholds" in m1["monitoring"]


def test_new_model_is_config_only(ws: Workspace, catalog: Catalog) -> None:
    """A fourth model from YAML + SQL alone: the platform code is untouched."""
    d = ws.root / "models" / "relist_risk"
    d.mkdir(exist_ok=True)
    (d / "entities.sql").write_text(
        """
        SELECT a.auction_id AS entity_id, a.listed_at AS as_of,
               CASE WHEN a.ended_at IS NOT NULL THEN (NOT a.sold AND a.relist_no = 0)::INT END AS label,
               a.ended_at AS label_known_at,
               v.make, v.model, v.fuel, v.segment, v.first_registration, a.mileage_km, a.damage_score,
               v.power_kw, a.start_price, a.seller_id, s.country AS seller_country, a.relist_no
        FROM auctions a JOIN vehicles v USING (vin) JOIN sellers s USING (seller_id)
        """
    )
    base = yaml.safe_load((ws.root / "models" / "sale_probability" / "model.yaml").read_text())
    base.update(name="relist_risk", description="first listing fails to sell", owner="marketplace")
    (d / "model.yaml").write_text(yaml.safe_dump(base))
    spec = ModelSpec.load(d / "model.yaml")
    r = run(spec, catalog, ws)
    assert r.evaluation["metrics"]["auc"] > 0.7
    assert review.verdict(review.review(spec, catalog, r.train, r.test)) != "block"
    shutil.rmtree(d)


def test_extract_drops_rows_without_labels(ws: Workspace) -> None:
    spec = spec_of(ws, "price_estimator/model.yaml")
    df = extract(spec, ws)
    assert df.label.notna().all()
