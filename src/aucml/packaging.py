"""Package a pipeline run into a SageMaker-style ``model.tar.gz``.

Layout (what SageMaker unpacks into /opt/ml/model)::

    model.joblib            fitted sklearn pipeline (preprocessing + estimator)
    manifest.json           contract: request schema, features, metrics, lineage,
                            drift reference profile and calibrated thresholds
    feature_repo/           catalog + on-demand transforms used at inference
    code/inference.py       SageMaker entry point (model_fn/input_fn/predict_fn/output_fn)

The archive is byte-reproducible: same run -> same sha256.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import shutil
import tarfile
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from aucml import __version__
from aucml.config import Catalog, ModelSpec, column
from aucml.featurestore import groups_for, request_schema
from aucml.monitoring.drift import build_profile, calibrate_backtest
from aucml.pipeline import FittedModel

ENTRY_POINT = '''"""SageMaker inference entry point. Logic lives in aucml.serving.inference."""
from aucml.serving.inference import input_fn, model_fn, output_fn, predict_fn  # noqa: F401
'''


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _schema(test_raw: pd.DataFrame, fields: list[str]) -> dict[str, str]:
    out = {}
    for f in fields:
        if f not in test_raw:
            out[f] = "string"
        elif pd.api.types.is_datetime64_any_dtype(test_raw[f]) or f == "as_of" or f.endswith(("_at", "_registration")):
            out[f] = "datetime"
        elif pd.api.types.is_numeric_dtype(test_raw[f]):
            out[f] = "number"
        else:
            out[f] = "string"
    return out


def reference_frame(spec: ModelSpec, train: pd.DataFrame, reference_weeks: int = 8) -> pd.DataFrame:
    """The last ``reference_weeks`` of training data: what production is compared to."""
    as_of = pd.to_datetime(train.as_of)
    return train[as_of >= pd.Timestamp(spec.split.cutoff) - pd.Timedelta(weeks=reference_weeks)]


def build_manifest(
    run_dir: Path, spec: ModelSpec, catalog: Catalog, model: FittedModel, train: pd.DataFrame, test_raw: pd.DataFrame
) -> dict[str, Any]:
    run = json.loads((run_dir / "run.json").read_text())
    evaluation = json.loads((run_dir / "evaluation.json").read_text())
    ref = reference_frame(spec, train)
    profile = build_profile(ref, model.numeric, model.categorical, model.predict(ref))
    thresholds, windows = calibrate_backtest(train, model.numeric, model.categorical, model.predict)
    on_demand, batch = groups_for(catalog, spec.features)
    required, optional = request_schema(catalog, spec.features)
    fields = required + optional
    return {
        "name": spec.name,
        "version": run["run_id"],
        "platform_version": __version__,
        "task": spec.task,
        "prediction_point": spec.prediction_point,
        "features": spec.features,
        "columns": model.columns,
        "categorical": model.categorical,
        "on_demand_groups": on_demand,
        "batch_groups": batch,
        "request_schema": {"required": required, "optional": optional, "types": _schema(test_raw, fields)},
        "training_window": {"start": str(spec.split.train_start), "cutoff": str(spec.split.cutoff), "test_end": str(spec.split.test_end)},
        "rows": evaluation["rows"],
        "metrics": evaluation["metrics"],
        "baseline_metrics": evaluation["baseline"],
        "slice_performance": {column(ref): rows for ref, rows in evaluation["slices"].items()},
        "primary_metric": spec.gate.primary_metric,
        "monitoring": {
            "reference_rows": len(ref),
            "calibration_windows": windows,
            "profile": profile,
            "thresholds": thresholds,
            "reference_performance": evaluation["metrics"][spec.gate.primary_metric],
        },
    }


def _add_bytes(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mtime = 0
    info.mode = 0o644
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    tar.addfile(info, io.BytesIO(data))


def package(run_dir: Path, spec: ModelSpec, catalog: Catalog, out_dir: Path) -> tuple[Path, dict[str, Any]]:
    model: FittedModel = joblib.load(run_dir / "model.joblib")
    train = pd.read_parquet(run_dir / "train.parquet")
    test_raw = pd.read_parquet(run_dir / "test_raw.parquet")
    manifest = build_manifest(run_dir, spec, catalog, model, train, test_raw)
    manifest["model_sha256"] = sha256(run_dir / "model.joblib")
    files: dict[str, bytes] = {
        "model.joblib": (run_dir / "model.joblib").read_bytes(),
        "manifest.json": json.dumps(manifest, indent=2, sort_keys=True, default=_json_default).encode(),
        "feature_repo/catalog.yaml": (catalog.root / "catalog.yaml").read_bytes(),
        "feature_repo/transforms.py": (catalog.root / "transforms.py").read_bytes(),
        "code/inference.py": ENTRY_POINT.encode(),
        "code/requirements.txt": b"# the serving image installs aucml and its pinned dependencies\n",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    tar_path = out_dir / "model.tar.gz"
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.USTAR_FORMAT) as tar:
        for name in sorted(files):
            _add_bytes(tar, name, files[name])
    with tar_path.open("wb") as f, gzip.GzipFile(fileobj=f, mode="wb", mtime=0, filename="") as gz:
        gz.write(raw.getvalue())
    (out_dir / "manifest.json").write_bytes(files["manifest.json"])
    manifest["package_sha256"] = sha256(tar_path)
    return tar_path, manifest


def unpack(tar_path: Path, dest: Path) -> Path:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    with tarfile.open(tar_path) as tar:
        tar.extractall(dest, filter="data")
    return dest


def _json_default(o: object) -> object:
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, pd.Timestamp):
        return o.isoformat()
    raise TypeError(type(o))
