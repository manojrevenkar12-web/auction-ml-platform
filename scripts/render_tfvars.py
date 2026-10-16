"""Render the models deployed in a registry lane as Terraform variables.

    python scripts/render_tfvars.py prod --artifacts-bucket aucml-prod-artifacts-123456789012

Alarm thresholds come from each version's manifest, so the CloudWatch alarms
track the same calibrated limits as the platform's own monitoring.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aucml.monitoring.monitor import PERF_CRIT
from aucml.registry import Registry


def render(registry: Path, lane: str, bucket: str) -> dict[str, object]:
    reg = Registry(registry)
    models: dict[str, object] = {}
    for model_dir in sorted(p for p in registry.iterdir() if p.is_dir()):
        version = reg.lanes(model_dir.name).get(lane)
        if not version:
            continue
        m = json.loads((reg.vdir(model_dir.name, version) / "manifest.json").read_text())
        pm = m["primary_metric"]
        ref = float(m["monitoring"]["reference_performance"])
        lower = pm in ("mape", "median_ape", "mae", "brier", "logloss")
        models[model_dir.name] = {
            "version": version,
            "model_data_url": f"s3://{bucket}/models/{model_dir.name}/{version}/model.tar.gz",
            "primary_metric": pm,
            "metric_direction": "lower" if lower else "higher",
            "performance_critical": round(ref * (1 + PERF_CRIT) if lower else ref * (1 - PERF_CRIT), 6),
            "prediction_psi_critical": round(float(m["monitoring"]["thresholds"]["__prediction__"]["critical"]), 6),
            "latency_p95_ms": 100,
        }
    return {"models": models}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("lane", choices=["dev", "staging", "prod"])
    p.add_argument("--artifacts-bucket", required=True)
    p.add_argument("--registry", default="registry")
    a = p.parse_args()
    print(json.dumps(render(Path(a.registry), a.lane, a.artifacts_bucket), indent=2))


if __name__ == "__main__":
    main()
