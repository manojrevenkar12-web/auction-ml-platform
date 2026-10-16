from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from aucml.config import Catalog
from aucml.featurestore import OfflineStore, OnlineStore
from aucml.pipeline import Workspace


def _requests(ws: Workspace, group: str, n: int = 400) -> tuple[pd.Series, pd.Series]:
    snap = pd.read_parquet(OfflineStore(ws.store).path(group))
    rng = np.random.default_rng(0)
    keys = pd.Series(rng.choice(snap.entity_key.unique(), n))
    as_of = pd.Series(pd.Timestamp("2025-06-01") + pd.to_timedelta(rng.uniform(0, 200, n), unit="D"))
    return keys, as_of


def test_point_in_time_join_never_returns_future_snapshots(ws: Workspace) -> None:
    keys, as_of = _requests(ws, "seller_90d")
    got = OfflineStore(ws.store).get("seller_90d", keys, as_of)
    assert got.event_time.notna().mean() > 0.8  # PL sellers have no history before Apr 2026
    assert (got.event_time.dropna() <= as_of[got.event_time.notna()]).all()
    assert (got.known_at.dropna() <= as_of[got.known_at.notna()]).all()


def test_point_in_time_join_picks_the_latest_eligible_snapshot(ws: Workspace) -> None:
    keys, as_of = _requests(ws, "segment_market_28d", 50)
    got = OfflineStore(ws.store).get("segment_market_28d", keys, as_of)
    snap = pd.read_parquet(OfflineStore(ws.store).path("segment_market_28d"))
    for k, t, e in zip(keys, as_of, got.event_time, strict=True):
        eligible = snap[(snap.entity_key == k) & (snap.event_time <= t)]
        assert e == eligible.event_time.max()


def test_naive_join_leaks_future_values(ws: Workspace) -> None:
    keys, as_of = _requests(ws, "seller_90d")
    got = OfflineStore(ws.store, point_in_time=False).get("seller_90d", keys, as_of)
    assert (got.event_time > as_of).mean() > 0.9


def test_booking_window_aggregates_unknown_costs(ws: Workspace) -> None:
    good = pd.read_parquet(OfflineStore(ws.store).path("lane_cost_56d"))
    bad = pd.read_parquet(OfflineStore(ws.store).path("lane_cost_56d_by_booking"))
    assert (good.known_at <= good.event_time).all()
    assert (bad.known_at > bad.event_time).mean() > 0.5


def test_online_store_matches_offline_as_of_materialisation(ws: Workspace, tmp_path: object) -> None:
    at = datetime(2025, 12, 1)
    online = OnlineStore(ws.store / "test_online.sqlite")
    online.materialize(OfflineStore(ws.store), ["seller_90d", "model_market_90d"], at)
    for g in ("seller_90d", "model_market_90d"):
        keys, _ = _requests(ws, g, 200)
        as_of = pd.Series([pd.Timestamp(at) + pd.Timedelta(hours=9)] * len(keys))
        off = OfflineStore(ws.store).get(g, keys, as_of)
        on = online.get(g, keys, as_of)
        cols = [c for c in off.columns if c not in ("event_time", "known_at")]
        pd.testing.assert_frame_equal(
            off[cols].astype(float).reset_index(drop=True), on[cols].astype(float).reset_index(drop=True), check_exact=False, rtol=1e-12
        )
        assert (off.event_time.fillna(pd.Timestamp(0)).to_numpy() == on.event_time.fillna(pd.Timestamp(0)).to_numpy()).all()


def test_unknown_entities_come_back_empty_not_wrong(ws: Workspace, catalog: Catalog) -> None:
    online = OnlineStore(ws.store / "test_online.sqlite")
    online.materialize(OfflineStore(ws.store), ["seller_90d"], datetime(2025, 12, 1))
    got = online.get("seller_90d", pd.Series(["NOPE"]), pd.Series([pd.Timestamp("2025-12-02")]))
    assert got.reindex(columns=list(catalog.groups["seller_90d"].features)).isna().all().all()
