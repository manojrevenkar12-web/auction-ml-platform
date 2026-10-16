from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from aucml.config import Catalog, ModelSpec
from aucml.featurestore import materialize_offline
from aucml.pipeline import RunResult, Workspace, run
from aucsim.market import write_warehouse

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def ws(tmp_path_factory: pytest.TempPathFactory) -> Workspace:
    root = tmp_path_factory.mktemp("ws")
    shutil.copytree(REPO / "feature_repo", root / "feature_repo")
    shutil.copytree(REPO / "models", root / "models")
    w = Workspace(root)
    write_warehouse(w.warehouse, seed=7)
    materialize_offline(Catalog.load(root / "feature_repo"), w.warehouse, w.store, "2025-01-01", "2026-07-01")
    return w


@pytest.fixture(scope="session")
def catalog(ws: Workspace) -> Catalog:
    return Catalog.load(ws.root / "feature_repo")


def spec_of(ws: Workspace, rel: str) -> ModelSpec:
    return ModelSpec.load(ws.root / "models" / rel)


@pytest.fixture(scope="session")
def price_run(ws: Workspace, catalog: Catalog) -> RunResult:
    return run(spec_of(ws, "price_estimator/model.yaml"), catalog, ws)


@pytest.fixture(scope="session")
def transport_run(ws: Workspace, catalog: Catalog) -> RunResult:
    return run(spec_of(ws, "transport_cost/model.yaml"), catalog, ws)
