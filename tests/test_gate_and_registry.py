from __future__ import annotations

import json
from pathlib import Path

import pytest

from aucml.config import Catalog
from aucml.gate import decide
from aucml.packaging import package
from aucml.pipeline import RunResult, Workspace
from aucml.registry import PromotionError, Registry
from aucml.review import as_dict, review


@pytest.fixture(scope="module")
def registered(ws: Workspace, catalog: Catalog, price_run: RunResult) -> tuple[Registry, str, Path]:
    reg = Registry(ws.root / "test_registry")
    tar, manifest = package(price_run.dir, price_run.spec, catalog, ws.root / "pkg_gate")
    reg.register(price_run.spec.name, price_run.run_id, tar, manifest, as_dict(review(price_run.spec, catalog, price_run.train, price_run.test)))
    return reg, price_run.run_id, tar


def test_gate_approves_a_sound_model(ws: Workspace, catalog: Catalog, price_run: RunResult, registered: tuple[Registry, str, Path]) -> None:
    reg, version, tar = registered
    decision, _ = decide(price_run.spec, catalog, ws, price_run.dir, tar)
    checks = {c["check"]: c for c in decision["checks"]}
    assert decision["decision"] == "approve", json.dumps(decision["checks"], indent=1)
    assert checks["serving_parity"]["feature_parity"] == 1.0
    assert checks["package_reproduces"]["passed"]
    reg.record_gate(price_run.spec.name, version, decision, "")


def test_gate_against_itself_is_non_inferior(ws: Workspace, catalog: Catalog, price_run: RunResult, registered: tuple[Registry, str, Path]) -> None:
    _, _, tar = registered
    decision, _ = decide(price_run.spec, catalog, ws, price_run.dir, tar, champion_package=tar)
    vs = next(c for c in decision["checks"] if c["check"] == "vs_champion")
    assert vs["passed"] and vs["delta"] == 0.0


def test_gate_rejects_when_bar_is_not_met(ws: Workspace, catalog: Catalog, price_run: RunResult, registered: tuple[Registry, str, Path]) -> None:
    _, _, tar = registered
    strict = price_run.spec.model_copy(update={"gate": price_run.spec.gate.model_copy(update={"threshold": 0.01})})
    decision, _ = decide(strict, catalog, ws, price_run.dir, tar)
    assert decision["decision"] == "reject"
    assert not next(c for c in decision["checks"] if c["check"] == "threshold")["passed"]


def test_promotion_lanes_are_enforced(registered: tuple[Registry, str, Path], price_run: RunResult) -> None:
    reg, version, _ = registered
    name = price_run.spec.name
    with pytest.raises(PromotionError, match="staging before prod"):
        reg.promote(name, version, "prod", approved_by="someone")
    reg.promote(name, version, "dev")
    reg.promote(name, version, "staging")
    with pytest.raises(PromotionError, match="named approver"):
        reg.promote(name, version, "prod")
    state = reg.promote(name, version, "prod", approved_by="reviewer")
    assert state["prod"] == version


def test_staging_needs_an_approved_gate(registered: tuple[Registry, str, Path], price_run: RunResult) -> None:
    reg, version, _ = registered
    name = price_run.spec.name
    gate_file = reg.vdir(name, version) / "gate.json"
    saved = gate_file.read_text()
    gate_file.write_text(json.dumps({"decision": "reject"}))
    try:
        with pytest.raises(PromotionError, match="fit-to-serve"):
            reg.promote(name, version, "staging")
    finally:
        gate_file.write_text(saved)


def test_rollback_restores_previous_version(registered: tuple[Registry, str, Path], price_run: RunResult) -> None:
    reg, version, tar = registered
    name = price_run.spec.name
    reg.register(name, "v-next", tar, {}, {"verdict": "pass"})
    reg.record_gate(name, "v-next", {"decision": "approve"}, "")
    reg.promote(name, "v-next", "staging")
    reg.promote(name, "v-next", "prod", approved_by="reviewer")
    state = reg.rollback(name, "prod", approved_by="on-call", reason="5xx after deploy")
    assert state["prod"] == version
    assert state["history"][-1]["reason"].startswith("rollback")
