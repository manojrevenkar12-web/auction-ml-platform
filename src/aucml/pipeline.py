"""The shared five-stage pipeline: extract -> validate -> transform -> train -> evaluate.

Every model runs through exactly this code. A model is a YAML spec plus an
entities SQL file; adding one requires no change here.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import duckdb
import joblib
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from aucml import metrics as M
from aucml.config import Catalog, ModelSpec, column
from aucml.featurestore import AUDIT_EVENT, AUDIT_KNOWN, OfflineStore, assemble, request_fields

BASE_COLUMNS = ["entity_id", "as_of", "label", "label_known_at"]


@dataclass(frozen=True)
class Workspace:
    root: Path

    @property
    def warehouse(self) -> Path:
        return self.root / "data" / "warehouse.duckdb"

    @property
    def store(self) -> Path:
        return self.root / "store"

    @property
    def runs(self) -> Path:
        return self.root / "runs"

    @property
    def registry(self) -> Path:
        return self.root / "registry"

    def warehouse_fingerprint(self) -> str:
        con = duckdb.connect(str(self.warehouse), read_only=True)
        try:
            row = con.execute("SELECT * FROM _meta").fetchone()
        finally:
            con.close()
        return hashlib.sha256(repr(row).encode()).hexdigest()[:12]


class StageError(RuntimeError):
    """A blocking failure in a pipeline stage."""


# --------------------------------------------------------------------- extract
def extract(spec: ModelSpec, ws: Workspace) -> pd.DataFrame:
    """Labelled entity rows in the spec window.

    Entities SQL returns the whole request population (what production scores);
    rows whose label has not arrived (or never will) are not training data.
    """
    s = spec.split
    sql = f"SELECT * FROM ({spec.sql()}) q WHERE as_of >= ? AND as_of < ? AND label IS NOT NULL"
    con = duckdb.connect(str(ws.warehouse), read_only=True)
    try:
        df = con.execute(sql, [s.train_start, s.test_end]).df()
    finally:
        con.close()
    return df.sort_values(["as_of", "entity_id"], kind="stable").reset_index(drop=True)


# -------------------------------------------------------------------- validate
@dataclass
class Check:
    name: str
    passed: bool
    detail: str
    blocking: bool = True


def validate(spec: ModelSpec, catalog: Catalog, df: pd.DataFrame) -> list[Check]:
    """Data contract on the extracted entity frame."""
    checks: list[Check] = []
    need = BASE_COLUMNS + request_fields(catalog, spec.features)
    missing = [c for c in need if c not in df.columns and c != "as_of"]
    checks.append(Check("required_columns", not missing, f"missing={missing}"))
    if missing:
        return checks
    checks.append(Check("min_rows", len(df) >= 2 * spec.gate.min_test_rows, f"rows={len(df)}"))
    dup = int(df.entity_id.duplicated().sum())
    checks.append(Check("unique_entity_id", dup == 0, f"duplicates={dup}"))
    nl = int(df.label.isna().sum())
    checks.append(Check("label_not_null", nl == 0, f"null_labels={nl}"))
    if spec.task == "classification":
        bad = int((~df.label.isin([0, 1, True, False])).sum())
        checks.append(Check("label_binary", bad == 0, f"non_binary={bad}"))
        rate = float(df.label.astype(float).mean())
        checks.append(Check("label_rate_sane", 0.02 < rate < 0.98, f"positive_rate={rate:.3f}"))
    else:
        bad = int((df.label <= 0).sum())
        checks.append(Check("label_positive", bad == 0, f"non_positive={bad}"))
    order = int((pd.to_datetime(df.label_known_at) < pd.to_datetime(df.as_of)).sum())
    checks.append(Check("label_after_prediction", order == 0, f"label_known_before_as_of={order}"))
    for f in request_fields(catalog, spec.features):
        if f in ("as_of", "auction_id"):
            continue
        rate = float(df[f].isna().mean())
        checks.append(Check(f"null_rate:{f}", rate <= 0.01, f"null_rate={rate:.4f}"))
    return checks


# ------------------------------------------------------------------- transform
def split(spec: ModelSpec, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    s = spec.split
    as_of = pd.to_datetime(df.as_of)
    if s.kind == "random":
        rng = np.random.default_rng(0)
        in_test = rng.random(len(df)) < s.test_fraction
        return df[~in_test].copy(), df[in_test].copy()
    cutoff = pd.Timestamp(s.cutoff)
    known = pd.to_datetime(df.label_known_at)
    train = df[(as_of < cutoff) & (known < cutoff)]
    test_mask = (as_of >= cutoff + pd.Timedelta(days=s.embargo_days)) & (as_of < pd.Timestamp(s.test_end))
    if s.labels_known_before is not None:
        test_mask &= known < pd.Timestamp(s.labels_known_before)
    return train.copy(), df[test_mask].copy()


def features_for(spec: ModelSpec, catalog: Catalog, ws: Workspace, raw: pd.DataFrame) -> pd.DataFrame:
    store = OfflineStore(ws.store, point_in_time=spec.point_in_time)
    feats = assemble(catalog, raw, spec.features, store)
    return pd.concat([raw[BASE_COLUMNS].reset_index(drop=True), feats.reset_index(drop=True)], axis=1)


# ----------------------------------------------------------------------- train
class FittedModel:
    """A fitted sklearn pipeline plus the contract it was trained under."""

    def __init__(self, spec: ModelSpec, catalog: Catalog) -> None:
        self.task = spec.task
        self.refs = list(spec.features)
        self.columns = spec.columns()
        self.categorical = [column(r) for r in self.refs if catalog.resolve(r)[2] == "categorical"]
        self.numeric = [c for c in self.columns if c not in self.categorical]
        e = spec.estimator
        pre = ColumnTransformer(
            [
                (
                    "cat",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=np.nan,
                        encoded_missing_value=np.nan,
                    ),
                    self.categorical,
                ),
                ("num", "passthrough", self.numeric),
            ],
            verbose_feature_names_out=False,
        )
        common: dict[str, Any] = {
            "max_iter": e.max_iter,
            "learning_rate": e.learning_rate,
            "max_leaf_nodes": e.max_leaf_nodes,
            "min_samples_leaf": e.min_samples_leaf,
            "l2_regularization": e.l2_regularization,
            "categorical_features": list(range(len(self.categorical))),
            "early_stopping": False,
            "random_state": 0,
        }
        est: Any
        if spec.task == "classification":
            est = HistGradientBoostingClassifier(**common)
        elif e.log_target:
            est = TransformedTargetRegressor(regressor=HistGradientBoostingRegressor(**common), func=np.log, inverse_func=np.exp)
        else:
            est = HistGradientBoostingRegressor(**common)
        self.pipeline = Pipeline([("pre", pre), ("model", est)])

    def _x(self, frame: pd.DataFrame) -> pd.DataFrame:
        data: dict[str, Any] = {}
        for c in self.categorical:
            v = frame[c]
            data[c] = v.astype(object).where(v.notna(), np.nan).to_numpy()
        for c in self.numeric:
            data[c] = pd.to_numeric(frame[c], errors="coerce").to_numpy(dtype="float64")
        return pd.DataFrame({c: data[c] for c in self.columns}, index=frame.index)

    def fit(self, frame: pd.DataFrame) -> FittedModel:
        y = frame["label"].astype(float).to_numpy()
        self.pipeline.fit(self._x(frame), y)
        return self

    def predict(self, frame: pd.DataFrame) -> NDArray[np.float64]:
        x = self._x(frame)
        if self.task == "classification":
            return np.asarray(self.pipeline.predict_proba(x)[:, 1], dtype=np.float64)
        return np.asarray(self.pipeline.predict(x), dtype=np.float64)


def train(spec: ModelSpec, catalog: Catalog, frame: pd.DataFrame) -> FittedModel:
    return FittedModel(spec, catalog).fit(frame)


# -------------------------------------------------------------------- evaluate
def baseline_predictions(spec: ModelSpec, train_df: pd.DataFrame, test_df: pd.DataFrame) -> NDArray[np.float64]:
    fill = float(train_df.label.astype(float).median() if spec.task == "regression" else train_df.label.astype(float).mean())
    if spec.baseline is None:
        return np.full(len(test_df), fill)
    col = column(spec.baseline)
    vals = pd.to_numeric(test_df[col], errors="coerce").astype("float64").fillna(fill)
    return np.asarray(vals, dtype=np.float64)


def evaluate(spec: ModelSpec, model: FittedModel, train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, Any]:
    y = test_df.label.astype(float).to_numpy()
    p = model.predict(test_df)
    b = baseline_predictions(spec, train_df, test_df)
    pm = spec.gate.primary_metric
    report: dict[str, Any] = {
        "rows": {"train": len(train_df), "test": len(test_df)},
        "metrics": M.compute(spec.metrics, y, p),
        "baseline": M.compute(spec.metrics, y, b),
        "primary_ci": M.bootstrap_ci(pm, y, p),
        "slices": {},
    }
    for ref in spec.slices:
        col = column(ref)
        rows = []
        for val, idx in test_df.groupby(test_df[col].astype(str)).groups.items():
            ii = test_df.index.get_indexer(idx)
            if len(ii) < spec.gate.min_slice_rows:
                rows.append({"value": val, "n": len(ii), "model": None, "baseline": None})
                continue
            rows.append(
                {
                    "value": val,
                    "n": len(ii),
                    "model": M.METRICS[pm][0](y[ii], p[ii]),
                    "baseline": M.METRICS[pm][0](y[ii], b[ii]),
                }
            )
        report["slices"][ref] = rows
    return report


# --------------------------------------------------------------------- run all
@dataclass
class RunResult:
    run_id: str
    dir: Path
    spec: ModelSpec
    model: FittedModel
    train: pd.DataFrame
    test: pd.DataFrame
    test_raw: pd.DataFrame
    validation: list[Check]
    evaluation: dict[str, Any]
    timings: dict[str, float] = field(default_factory=dict)


def run_id_for(spec: ModelSpec, catalog: Catalog, ws: Workspace) -> str:
    h = hashlib.sha256()
    h.update(spec.model_dump_json(exclude={"path"}).encode())
    h.update(spec.sql().encode())
    h.update((catalog.root / "catalog.yaml").read_bytes())
    h.update((catalog.root / "transforms.py").read_bytes())
    for g in sorted(catalog.groups):
        if catalog.groups[g].sql:
            h.update((catalog.root / (catalog.groups[g].sql or "")).read_bytes())
    h.update(ws.warehouse_fingerprint().encode())
    return f"{spec.name}-{spec.split.cutoff:%Y%m%d}-{h.hexdigest()[:8]}"


def with_window(spec: ModelSpec, train_start: date, cutoff: date, test_end: date, labels_known_before: datetime | None = None) -> ModelSpec:
    split = spec.split.model_copy(
        update={"train_start": train_start, "cutoff": cutoff, "test_end": test_end, "labels_known_before": labels_known_before}
    )
    return spec.model_copy(update={"split": split})


def run(spec: ModelSpec, catalog: Catalog, ws: Workspace) -> RunResult:
    timings: dict[str, float] = {}
    t = time.perf_counter()
    raw = extract(spec, ws)
    timings["extract"] = time.perf_counter() - t

    t = time.perf_counter()
    checks = validate(spec, catalog, raw)
    timings["validate"] = time.perf_counter() - t
    failed = [c for c in checks if c.blocking and not c.passed]
    if failed:
        raise StageError("validation failed: " + "; ".join(f"{c.name} ({c.detail})" for c in failed))

    t = time.perf_counter()
    train_raw, test_raw = split(spec, raw)
    train_df = features_for(spec, catalog, ws, train_raw)
    test_df = features_for(spec, catalog, ws, test_raw)
    timings["transform"] = time.perf_counter() - t

    t = time.perf_counter()
    model = train(spec, catalog, train_df)
    timings["train"] = time.perf_counter() - t

    t = time.perf_counter()
    report = evaluate(spec, model, train_df, test_df)
    timings["evaluate"] = time.perf_counter() - t

    rid = run_id_for(spec, catalog, ws)
    out = ws.runs / spec.name / rid
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / "model.joblib")
    train_df.to_parquet(out / "train.parquet", index=False)
    test_df.to_parquet(out / "test.parquet", index=False)
    test_raw.reset_index(drop=True).to_parquet(out / "test_raw.parquet", index=False)
    (out / "validation.json").write_text(json.dumps([c.__dict__ for c in checks], indent=2))
    (out / "evaluation.json").write_text(json.dumps(report, indent=2, default=str))
    (out / "run.json").write_text(
        json.dumps(
            {
                "run_id": rid,
                "model": spec.name,
                "spec": spec.model_dump(mode="json", exclude={"path"}),
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "timings_s": {k: round(v, 3) for k, v in timings.items()},
                "audit_columns": [c for c in train_df.columns if c.startswith((AUDIT_EVENT, AUDIT_KNOWN))],
            },
            indent=2,
        )
    )
    return RunResult(rid, out, spec, model, train_df, test_df, test_raw, checks, report, timings)
