"""SageMaker inference contract: model_fn, input_fn, predict_fn, output_fn.

The same four functions back the local FastAPI container (``aucml serve``), the
batch scorer used in monitoring, and a SageMaker endpoint. Request-time features
go through the feature repo's on-demand transforms (shared with training) and
batch features come from the online store, so the serving vector is assembled by
the same code that built the training set.

Clients may send a ``request_id`` per instance; it is echoed in the capture log
so delayed labels can be joined back for performance monitoring.

Monitoring hook: every call can append a JSON line per row (request, assembled
features, prediction, model version) to ``AUCML_CAPTURE_DIR``, the local
equivalent of SageMaker endpoint data capture.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import joblib
import numpy as np
import pandas as pd

from aucml.config import Catalog
from aucml.featurestore import Lookup, OnlineStore, assemble, load_transforms
from aucml.pipeline import FittedModel


class RequestError(ValueError):
    """Client error: the request does not match the model's request schema."""


@dataclass
class Handler:
    model_dir: Path
    manifest: dict[str, Any]
    model: FittedModel
    catalog: Catalog
    transforms: ModuleType
    lookup: Lookup
    capture_dir: Path | None = None
    timings_ms: list[float] = field(default_factory=list)

    @property
    def version(self) -> str:
        return str(self.manifest["version"])


def _lookup_from_env() -> Lookup:
    backend = os.environ.get("AUCML_FEATURE_STORE", "sqlite")
    if backend == "sagemaker":
        from aucml.aws import SageMakerOnlineStore

        return SageMakerOnlineStore.from_env()
    return OnlineStore(Path(os.environ.get("AUCML_ONLINE_STORE", "store/online.sqlite")))


def model_fn(model_dir: str | Path, lookup: Lookup | None = None) -> Handler:
    d = Path(model_dir)
    manifest = json.loads((d / "manifest.json").read_text())
    model: FittedModel = joblib.load(d / "model.joblib")
    catalog = Catalog.load(d / "feature_repo")
    capture = os.environ.get("AUCML_CAPTURE_DIR")
    return Handler(
        model_dir=d,
        manifest=manifest,
        model=model,
        catalog=catalog,
        transforms=load_transforms(catalog),
        lookup=lookup or _lookup_from_env(),
        capture_dir=Path(capture) if capture else None,
    )


def parse_records(records: list[dict[str, Any]], manifest: dict[str, Any]) -> pd.DataFrame:
    schema = manifest["request_schema"]
    if not records:
        raise RequestError("no instances in request")
    errors: list[str] = []
    for i, r in enumerate(records):
        miss = [f for f in schema["required"] if r.get(f) is None]
        if miss:
            errors.append(f"instance {i}: missing {miss}")
    if errors:
        raise RequestError("; ".join(errors[:10]))
    df = pd.DataFrame.from_records(records)
    for f, t in schema["types"].items():
        if f not in df:
            df[f] = np.nan
            continue
        try:
            if t == "datetime":
                df[f] = pd.to_datetime(df[f])
            elif t == "number":
                df[f] = pd.to_numeric(df[f], errors="raise").astype("float64")
            else:
                df[f] = df[f].astype(object)
        except (ValueError, TypeError) as exc:
            raise RequestError(f"field {f!r} is not a valid {t}: {exc}") from exc
    return df


def input_fn(body: str | bytes, content_type: str = "application/json", handler: Handler | None = None) -> pd.DataFrame:
    if content_type != "application/json":
        raise RequestError(f"unsupported content type {content_type!r}")
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RequestError(f"invalid JSON: {exc}") from exc
    records = payload.get("instances") if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise RequestError("expected a list of instances or {'instances': [...]}")
    if handler is None:
        return pd.DataFrame.from_records(records)
    return parse_records(records, handler.manifest)


def predict_fn(data: pd.DataFrame, handler: Handler) -> dict[str, Any]:
    t0 = time.perf_counter()
    if not {"as_of"} <= set(data.columns):
        raise RequestError("as_of is required")
    feats = assemble(handler.catalog, data, handler.manifest["features"], handler.lookup, handler.transforms)
    preds = handler.model.predict(feats)
    elapsed = (time.perf_counter() - t0) * 1000
    handler.timings_ms.append(elapsed)
    cols = handler.manifest["columns"]
    completeness = 1.0 - feats[cols].isna().mean(axis=1).to_numpy()
    if handler.capture_dir is not None:
        _capture(handler, data, feats[cols], preds)
    return {
        "model": handler.manifest["name"],
        "version": handler.version,
        "predictions": [float(p) for p in preds],
        "feature_completeness": [round(float(c), 4) for c in completeness],
    }


def output_fn(prediction: dict[str, Any], accept: str = "application/json") -> tuple[str, str]:
    if accept not in ("application/json", "*/*"):
        raise RequestError(f"unsupported accept type {accept!r}")
    return json.dumps(prediction), "application/json"


def _capture(handler: Handler, raw: pd.DataFrame, feats: pd.DataFrame, preds: np.ndarray) -> None:
    assert handler.capture_dir is not None
    handler.capture_dir.mkdir(parents=True, exist_ok=True)
    day = pd.Timestamp.now().strftime("%Y%m%d")
    path = handler.capture_dir / f"{handler.manifest['name']}-{day}.jsonl"
    req = raw.astype(object).where(raw.notna(), None)
    fv = feats.astype(object).where(feats.notna(), None)
    with path.open("a") as f:
        for i in range(len(raw)):
            f.write(
                json.dumps(
                    {
                        "event_id": str(uuid.uuid4()),
                        "model": handler.manifest["name"],
                        "version": handler.version,
                        "as_of": str(raw["as_of"].iloc[i]),
                        "request": {k: _plain(v) for k, v in req.iloc[i].items()},
                        "features": {k: _plain(v) for k, v in fv.iloc[i].items()},
                        "prediction": float(preds[i]),
                    }
                )
                + "\n"
            )


def _plain(v: object) -> object:
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    if isinstance(v, np.generic):
        return v.item()
    return v


def handle(body: str | bytes, handler: Handler, content_type: str = "application/json") -> tuple[str, str]:
    """Full request path, as SageMaker would chain it."""
    return output_fn(predict_fn(input_fn(body, content_type, handler), handler))
