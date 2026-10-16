"""Typed configuration: the feature catalog and per-model specs."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

Task = Literal["regression", "classification"]
Direction = Literal["lower", "higher"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FeatureGroup(_Strict):
    kind: Literal["on_demand", "batch"]
    available_at: str
    features: dict[str, Literal["numeric", "categorical"]]
    # on_demand
    transform: str | None = None
    request_fields: list[str] = Field(default_factory=list)
    optional_fields: dict[str, list[str]] = Field(default_factory=dict)
    keys: list[str] = Field(default_factory=list)
    # batch
    entity: str | None = None
    sql: str | None = None

    @model_validator(mode="after")
    def _check_kind(self) -> FeatureGroup:
        if self.kind == "on_demand" and not self.transform:
            raise ValueError("on_demand groups need a transform")
        if self.kind == "batch" and not (self.entity and self.sql):
            raise ValueError("batch groups need entity and sql")
        if self.kind == "batch" and set(self.features.values()) != {"numeric"}:
            raise ValueError("batch features must be numeric; categoricals belong in on-demand groups")
        unknown = set(self.optional_fields) - set(self.request_fields)
        if unknown:
            raise ValueError(f"optional_fields not in request_fields: {sorted(unknown)}")
        return self


class Catalog(_Strict):
    prediction_points: list[str]
    groups: dict[str, FeatureGroup]
    root: Path

    @classmethod
    def load(cls, root: Path) -> Catalog:
        raw = yaml.safe_load((root / "catalog.yaml").read_text())
        return cls(root=root, **raw)

    def point_rank(self, point: str) -> int:
        return self.prediction_points.index(point)

    def resolve(self, ref: str) -> tuple[str, str, str]:
        """``group.feature`` -> (group, feature, kind)."""
        group, _, feat = ref.partition(".")
        if group not in self.groups or feat not in self.groups[group].features:
            raise KeyError(f"unknown feature {ref!r}")
        return group, feat, self.groups[group].features[feat]


def column(ref: str) -> str:
    """Column name used in feature frames for ``group.feature``."""
    return ref.replace(".", "__", 1)


class EstimatorSpec(_Strict):
    max_iter: int = 400
    learning_rate: float = 0.06
    max_leaf_nodes: int = 31
    min_samples_leaf: int = 40
    l2_regularization: float = 0.0
    log_target: bool = False


class SplitSpec(_Strict):
    kind: Literal["time", "random"] = "time"
    train_start: date
    cutoff: date
    test_end: date
    embargo_days: int = 0
    test_fraction: float = 0.2  # random split only
    # when retraining "now": only use test labels that have actually arrived
    labels_known_before: datetime | None = None


class GateSpec(_Strict):
    primary_metric: str
    direction: Direction
    threshold: float
    min_rel_improvement_vs_baseline: float = 0.10
    champion_margin: float = 0.0
    slice_tolerance: float = 0.05
    min_slice_rows: int = 150
    min_test_rows: int = 1000
    max_latency_p95_ms: float = 100.0
    min_parity: float = 0.999


class ModelSpec(_Strict):
    name: str
    description: str
    owner: str
    task: Task
    prediction_point: str
    entities_sql: str
    features: list[str]
    point_in_time: bool = True
    estimator: EstimatorSpec = EstimatorSpec()
    split: SplitSpec
    metrics: list[str]
    baseline: str | None = None
    slices: list[str] = Field(default_factory=list)
    gate: GateSpec
    path: Path

    @classmethod
    def load(cls, path: Path) -> ModelSpec:
        raw = yaml.safe_load(path.read_text())
        return cls(path=path, **raw)

    def sql(self) -> str:
        return (self.path.parent / self.entities_sql).read_text()

    def columns(self) -> list[str]:
        return [column(f) for f in self.features]


def discover_specs(models_dir: Path, include_experiments: bool = False) -> list[ModelSpec]:
    specs = []
    for p in sorted(models_dir.glob("*/model.yaml")):
        if p.parent.name.startswith("_") and not include_experiments:
            continue
        specs.append(ModelSpec.load(p))
    if include_experiments:
        specs += [ModelSpec.load(p) for p in sorted(models_dir.glob("_*/*.yaml"))]
    return specs
