"""The shared platform must stay model-agnostic: no model or feature-group names in it."""

from __future__ import annotations

import re
from pathlib import Path

from aucml.config import Catalog, discover_specs

REPO = Path(__file__).resolve().parents[1]


def test_platform_code_names_no_model_or_feature_group() -> None:
    catalog = Catalog.load(REPO / "feature_repo")
    names = {s.name for s in discover_specs(REPO / "models", include_experiments=True)} | set(catalog.groups)
    offenders = []
    for path in (REPO / "src" / "aucml").rglob("*.py"):
        text = path.read_text()
        for n in names:
            if re.search(rf"\b{re.escape(n)}\b", text):
                offenders.append(f"{path.relative_to(REPO)}: {n}")
    assert not offenders, offenders


def test_every_spec_resolves_against_the_catalog() -> None:
    catalog = Catalog.load(REPO / "feature_repo")
    for spec in discover_specs(REPO / "models", include_experiments=True):
        for ref in [*spec.features, *spec.slices, *([spec.baseline] if spec.baseline else [])]:
            catalog.resolve(ref)
        assert spec.prediction_point in catalog.prediction_points
        assert spec.sql().strip()


def test_batch_groups_are_numeric() -> None:
    """Batch features travel through float-typed stores; categoricals belong in on-demand groups."""
    catalog = Catalog.load(REPO / "feature_repo")
    for name, g in catalog.groups.items():
        if g.kind == "batch":
            assert set(g.features.values()) == {"numeric"}, name
