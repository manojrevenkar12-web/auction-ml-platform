"""Model registry with governed promotion lanes: dev -> staging -> prod.

Rules (enforced here, mirrored by the GitHub Actions promotion workflow):

* dev      any packaged version whose pre-ship review did not block
* staging  requires an approved fit-to-serve gate decision for that version
* prod     requires the version to be in staging first, an approved gate, and a
           named human approver
Every lane change is appended to an audit history; rollback restores the
previous version of a lane.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

LANES = ("dev", "staging", "prod")


class PromotionError(RuntimeError):
    pass


@dataclass(frozen=True)
class Registry:
    root: Path

    def vdir(self, model: str, version: str) -> Path:
        return self.root / model / version

    def _lanes_path(self, model: str) -> Path:
        return self.root / model / "lanes.json"

    def lanes(self, model: str) -> dict[str, Any]:
        p = self._lanes_path(model)
        if not p.exists():
            return {"dev": None, "staging": None, "prod": None, "history": []}
        data: dict[str, Any] = json.loads(p.read_text())
        return data

    def register(self, model: str, version: str, package: Path, manifest: dict[str, Any], review: dict[str, Any]) -> Path:
        d = self.vdir(model, version)
        d.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(package, d / "model.tar.gz")
        (d / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str))
        (d / "review.json").write_text(json.dumps(review, indent=2))
        return d

    def record_gate(self, model: str, version: str, decision: dict[str, Any], markdown: str) -> None:
        d = self.vdir(model, version)
        (d / "gate.json").write_text(json.dumps(decision, indent=2, default=str))
        (d / "gate.md").write_text(markdown)

    def gate(self, model: str, version: str) -> dict[str, Any] | None:
        p = self.vdir(model, version) / "gate.json"
        if not p.exists():
            return None
        data: dict[str, Any] = json.loads(p.read_text())
        return data

    def promote(self, model: str, version: str, lane: str, approved_by: str | None = None, reason: str = "") -> dict[str, Any]:
        if lane not in LANES:
            raise PromotionError(f"unknown lane {lane!r}")
        d = self.vdir(model, version)
        if not (d / "model.tar.gz").exists():
            raise PromotionError(f"{model}/{version} is not registered")
        review = json.loads((d / "review.json").read_text())
        if review.get("verdict") == "block":
            raise PromotionError("pre-ship review blocked this version")
        state = self.lanes(model)
        if lane in ("staging", "prod"):
            g = self.gate(model, version)
            if not g or g.get("decision") != "approve":
                raise PromotionError(f"no approved fit-to-serve decision for {version}")
        if lane == "prod":
            if state.get("staging") != version:
                raise PromotionError("a version must be in staging before prod")
            if not approved_by:
                raise PromotionError("prod promotion needs a named approver")
        prev = state.get(lane)
        state[lane] = version
        state["history"].append(
            {
                "at": datetime.now(UTC).isoformat(timespec="seconds"),
                "lane": lane,
                "from": prev,
                "to": version,
                "approved_by": approved_by,
                "reason": reason,
            }
        )
        self._lanes_path(model).parent.mkdir(parents=True, exist_ok=True)
        self._lanes_path(model).write_text(json.dumps(state, indent=2))
        return state

    def rollback(self, model: str, lane: str, approved_by: str, reason: str) -> dict[str, Any]:
        state = self.lanes(model)
        moves = [h for h in state["history"] if h["lane"] == lane]
        if not moves or moves[-1]["from"] is None:
            raise PromotionError(f"nothing to roll back to in {lane}")
        target = moves[-1]["from"]
        prev = state[lane]
        state[lane] = target
        state["history"].append(
            {
                "at": datetime.now(UTC).isoformat(timespec="seconds"),
                "lane": lane,
                "from": prev,
                "to": target,
                "approved_by": approved_by,
                "reason": f"rollback: {reason}",
            }
        )
        self._lanes_path(model).write_text(json.dumps(state, indent=2))
        return state

    def package_path(self, model: str, lane: str) -> Path:
        v = self.lanes(model).get(lane)
        if not v:
            raise PromotionError(f"nothing deployed to {model}/{lane}")
        return self.vdir(model, v) / "model.tar.gz"
