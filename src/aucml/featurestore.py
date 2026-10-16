"""Feature store: offline (Parquet + point-in-time joins) and online (latest values).

One assembly function, :func:`assemble`, builds model inputs for both training
(``OfflineStore``) and serving (``OnlineStore``). Only the lookup backend differs,
so parity between the two is a property that can be tested, not hoped for.
"""

from __future__ import annotations

import importlib.util
import json
import sqlite3
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Protocol

import duckdb
import numpy as np
import pandas as pd

from aucml.config import Catalog, column

AUDIT_EVENT = "__event_time__"
AUDIT_KNOWN = "__known_at__"


def load_transforms(catalog: Catalog) -> ModuleType:
    path = catalog.root / "transforms.py"
    spec = importlib.util.spec_from_file_location("feature_repo_transforms", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Lookup(Protocol):
    """Backend that returns batch feature values for (key, as_of) pairs."""

    def get(self, group: str, keys: pd.Series, as_of: pd.Series) -> pd.DataFrame: ...


def groups_for(catalog: Catalog, refs: list[str]) -> tuple[list[str], list[str]]:
    """Split the groups referenced by ``refs`` into (on_demand, batch)."""
    seen = sorted({catalog.resolve(r)[0] for r in refs})
    on_demand = [g for g in seen if catalog.groups[g].kind == "on_demand"]
    batch = [g for g in seen if catalog.groups[g].kind == "batch"]
    # batch groups need keys produced by an on-demand group
    needed_keys = {catalog.groups[g].entity for g in batch}
    for name, g in catalog.groups.items():
        if g.kind == "on_demand" and name not in on_demand and needed_keys & set(g.keys):
            on_demand.append(name)
    return sorted(on_demand), batch


def request_fields(catalog: Catalog, refs: list[str]) -> list[str]:
    """Fields a client must send for a model using ``refs``."""
    return request_schema(catalog, refs)[0]


def request_schema(catalog: Catalog, refs: list[str]) -> tuple[list[str], list[str]]:
    """(required, optional) request fields for a model using ``refs``."""
    on_demand, _ = groups_for(catalog, refs)
    used = {catalog.resolve(r)[:2] for r in refs}
    required: list[str] = []
    optional: list[str] = []
    for g in on_demand:
        grp = catalog.groups[g]
        for f in grp.request_fields:
            needed_by = grp.optional_fields.get(f)
            is_optional = needed_by is not None and not any((g, feat) in used for feat in needed_by)
            target = optional if is_optional else required
            if f not in required and f not in optional:
                target.append(f)
    return required, optional


def assemble(
    catalog: Catalog,
    raw: pd.DataFrame,
    refs: list[str],
    lookup: Lookup,
    transforms: ModuleType | None = None,
) -> pd.DataFrame:
    """Raw request/entity rows -> model input frame (+ audit timestamp columns)."""
    transforms = transforms or load_transforms(catalog)
    on_demand, batch = groups_for(catalog, refs)
    cols: dict[str, object] = {}
    keys: dict[str, pd.Series] = {}
    for g in on_demand:
        fn: Callable[[pd.DataFrame], pd.DataFrame] = getattr(transforms, catalog.groups[g].transform or "")
        res = fn(raw)
        for feat in catalog.groups[g].features:
            cols[column(f"{g}.{feat}")] = res[feat].to_numpy()
        for k in catalog.groups[g].keys:
            keys[k] = res[k]
    as_of = pd.to_datetime(raw["as_of"])
    audit: dict[str, object] = {}
    for g in batch:
        entity = catalog.groups[g].entity or ""
        if entity not in keys:
            raise KeyError(f"no on-demand group produces key {entity!r} for group {g!r}")
        feats = list(catalog.groups[g].features)
        vals = lookup.get(g, keys[entity], as_of).reindex(columns=[*feats, "event_time", "known_at"])
        for feat in feats:
            cols[column(f"{g}.{feat}")] = pd.to_numeric(vals[feat], errors="coerce").to_numpy(dtype="float64")
        audit[AUDIT_EVENT + g] = pd.to_datetime(vals["event_time"]).to_numpy()
        audit[AUDIT_KNOWN + g] = pd.to_datetime(vals["known_at"]).to_numpy()
    ordered = {column(r): cols[column(r)] for r in refs}
    return pd.DataFrame({**ordered, **audit}, index=raw.index)


def materialize_offline(catalog: Catalog, warehouse: Path, store: Path, start: str, end: str) -> dict[str, int]:
    """Run every batch group's SQL against the warehouse into ``store/offline``."""
    off = store / "offline"
    off.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    con = duckdb.connect(str(warehouse), read_only=True)
    try:
        for name, g in catalog.groups.items():
            if g.kind != "batch":
                continue
            sql = (catalog.root / (g.sql or "")).read_text().format(start=start, end=end)
            df = con.execute(sql).df()
            expected = ["entity_key", "event_time", "known_at", *g.features]
            missing = set(expected) - set(df.columns)
            if missing:
                raise ValueError(f"{name}: SQL is missing columns {sorted(missing)}")
            df = df[expected].sort_values(["entity_key", "event_time"]).reset_index(drop=True)
            df.to_parquet(off / f"{name}.parquet", index=False)
            counts[name] = len(df)
    finally:
        con.close()
    return counts


class OfflineStore:
    """Point-in-time lookups over the materialised Parquet snapshots."""

    def __init__(self, store: Path, point_in_time: bool = True) -> None:
        self.dir = store / "offline"
        self.point_in_time = point_in_time

    def path(self, group: str) -> Path:
        return self.dir / f"{group}.parquet"

    def get(self, group: str, keys: pd.Series, as_of: pd.Series) -> pd.DataFrame:
        req = pd.DataFrame({"row_id": np.arange(len(keys)), "entity_key": keys.to_numpy(), "as_of": as_of.to_numpy()})
        con = duckdb.connect()
        try:
            con.register("req", req)
            src = f"read_parquet('{self.path(group)}')"
            if self.point_in_time:
                sql = f"""
                    SELECT req.row_id, f.* EXCLUDE (entity_key)
                    FROM req ASOF LEFT JOIN {src} f
                      ON req.entity_key = f.entity_key AND req.as_of >= f.event_time
                """
            else:  # the classic mistake: join the latest known value, ignoring time
                sql = f"""
                    SELECT req.row_id, f.* EXCLUDE (entity_key, rn)
                    FROM req LEFT JOIN (
                      SELECT *, row_number() OVER (PARTITION BY entity_key ORDER BY event_time DESC) rn
                      FROM {src}
                    ) f ON req.entity_key = f.entity_key AND f.rn = 1
                """
            res = con.execute(sql).df().sort_values("row_id")
        finally:
            con.close()
        res.index = keys.index
        return res.drop(columns="row_id")


class OnlineStore:
    """Latest value per entity, as served at inference time.

    SQLite stands in for the SageMaker Feature Store online store; see
    :mod:`aucml.aws` for the boto3 adapter with the same ``get`` signature.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._reader: sqlite3.Connection | None = None

    def _con(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(self.path)

    def materialize(self, offline: OfflineStore, groups: list[str], at: datetime) -> dict[str, int]:
        """Load the latest snapshot at or before ``at`` for each entity."""
        counts: dict[str, int] = {}
        if self._reader is not None:
            self._reader.close()
            self._reader = None
        with self._con() as con:
            for g in groups:
                df = pd.read_parquet(offline.path(g))
                df = df[df.event_time <= pd.Timestamp(at)]
                df = df.sort_values("event_time").groupby("entity_key", as_index=False).tail(1)
                con.execute(f'DROP TABLE IF EXISTS "{g}"')
                con.execute(f'CREATE TABLE "{g}" (entity_key TEXT PRIMARY KEY, record TEXT NOT NULL)')
                feats = [c for c in df.columns if c != "entity_key"]
                rows = [
                    (str(k), json.dumps({c: _jsonable(v) for c, v in zip(feats, vals, strict=True)}))
                    for k, *vals in df[["entity_key", *feats]].itertuples(index=False)
                ]
                con.executemany(f'INSERT INTO "{g}" VALUES (?, ?)', rows)
                counts[g] = len(rows)
        return counts

    def get(self, group: str, keys: pd.Series, as_of: pd.Series) -> pd.DataFrame:
        del as_of  # online: whatever is current
        uniq = [str(k) for k in pd.unique(keys.astype(str))]
        if self._reader is None:
            self._reader = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True, check_same_thread=False)
        found: dict[str, dict[str, object]] = {}
        for i in range(0, len(uniq), 500):
            chunk = uniq[i : i + 500]
            q = f'SELECT entity_key, record FROM "{group}" WHERE entity_key IN ({",".join("?" * len(chunk))})'
            for k, rec in self._reader.execute(q, chunk):
                found[k] = json.loads(rec)
        records = [found.get(str(k), {}) for k in keys]
        names = list(records[0].keys()) if records and records[0] else []
        for r in records:
            if r and len(r) != len(names):
                names = list(dict.fromkeys([*names, *r]))
        data = {n: [r.get(n) for r in records] for n in names}
        df = pd.DataFrame(data, index=keys.index)
        for c in ("event_time", "known_at"):
            df[c] = pd.to_datetime(df[c], format="ISO8601") if c in df else pd.NaT
        return df


def _jsonable(v: object) -> object:
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    if isinstance(v, float) and np.isnan(v):
        return None
    if isinstance(v, np.floating):
        return None if np.isnan(v) else float(v)
    return v
