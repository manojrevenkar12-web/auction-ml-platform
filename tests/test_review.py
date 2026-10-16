from __future__ import annotations

from aucml.config import Catalog
from aucml.pipeline import RunResult, Workspace, run
from aucml.review import review, verdict

from .conftest import spec_of


def _failed(findings: list) -> set[str]:  # type: ignore[type-arg]
    return {f.check for f in findings if not f.passed and f.severity == "error"}


def test_production_specs_pass(price_run: RunResult, transport_run: RunResult, catalog: Catalog) -> None:
    for r in (price_run, transport_run):
        assert verdict(review(r.spec, catalog, r.train, r.test)) in ("pass", "pass_with_warnings")


def test_leaky_price_model_is_blocked_for_every_reason(ws: Workspace, catalog: Catalog, price_run: RunResult) -> None:
    spec = spec_of(ws, "_experiments/price_estimator_leaky.yaml")
    r = run(spec, catalog, ws)
    findings = review(spec, catalog, r.train, r.test)
    assert verdict(findings) == "block"
    assert _failed(findings) == {"temporal_split", "label_availability", "post_event_features", "point_in_time_join"}
    # ...and it is exactly the model that would have looked best offline
    assert r.evaluation["metrics"]["mape"] < price_run.evaluation["metrics"]["mape"]


def test_backward_window_error_is_caught_even_when_metrics_look_fine(ws: Workspace, catalog: Catalog, transport_run: RunResult) -> None:
    spec = spec_of(ws, "_experiments/transport_cost_by_booking.yaml")
    r = run(spec, catalog, ws)
    findings = review(spec, catalog, r.train, r.test)
    assert _failed(findings) == {"backward_window"}
    gap = abs(r.evaluation["metrics"]["mape"] - transport_run.evaluation["metrics"]["mape"])
    assert gap < 0.01  # a metric-based sanity check would not have noticed
