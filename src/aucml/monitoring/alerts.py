"""Alerts and where they go: JSON lines, console, Slack webhook, CloudWatch EMF."""

from __future__ import annotations

import json
import sys
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Protocol, TextIO

Severity = Literal["warn", "critical"]


@dataclass(frozen=True)
class Alert:
    model: str
    version: str
    window: str
    check: str  # drift | prediction_drift | data_quality | performance
    subject: str  # column or slice
    severity: Severity
    value: float
    threshold: float
    message: str

    def key(self) -> str:
        return f"{self.model}:{self.check}:{self.subject}"


class Sink(Protocol):
    def send(self, alerts: list[Alert]) -> None: ...


class JsonlSink:
    def __init__(self, path: Path) -> None:
        self.path = path

    def send(self, alerts: list[Alert]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            for a in alerts:
                f.write(json.dumps(asdict(a)) + "\n")


class ConsoleSink:
    def __init__(self, stream: TextIO | None = None) -> None:
        self.stream = stream or sys.stdout

    def send(self, alerts: list[Alert]) -> None:
        for a in alerts:
            self.stream.write(f"[{a.severity.upper():8s}] {a.window} {a.model} {a.check}/{a.subject}: {a.message}\n")


class SlackWebhookSink:
    """Posts critical alerts to a Slack incoming webhook (one message per batch)."""

    def __init__(self, url: str, min_severity: Severity = "critical") -> None:
        self.url = url
        self.min_severity = min_severity

    def payload(self, alerts: list[Alert]) -> dict[str, str] | None:
        keep = [a for a in alerts if self.min_severity == "warn" or a.severity == "critical"]
        if not keep:
            return None
        lines = [f"*{a.model}* `{a.check}/{a.subject}` {a.message}" for a in keep]
        return {"text": f":rotating_light: {len(keep)} model alert(s)\n" + "\n".join(lines)}

    def send(self, alerts: list[Alert]) -> None:
        body = self.payload(alerts)
        if body is None:
            return
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=5).close()  # noqa: S310 - URL comes from operator config


def emf_record(namespace: str, model: str, metrics: dict[str, float], timestamp_ms: int) -> dict[str, object]:
    """CloudWatch Embedded Metric Format record; CloudWatch alarms are defined in Terraform."""
    return {
        "_aws": {
            "Timestamp": timestamp_ms,
            "CloudWatchMetrics": [
                {
                    "Namespace": namespace,
                    "Dimensions": [["Model"]],
                    "Metrics": [{"Name": k, "Unit": "None"} for k in sorted(metrics)],
                }
            ],
        },
        "Model": model,
        **metrics,
    }


class EmfSink:
    """Writes one EMF line per alert window so CloudWatch can alarm on the raw scores."""

    def __init__(self, path: Path, namespace: str = "aucml/monitoring") -> None:
        self.path = path
        self.namespace = namespace

    def write_scores(self, model: str, scores: dict[str, float], timestamp_ms: int) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(emf_record(self.namespace, model, scores, timestamp_ms)) + "\n")

    def send(self, alerts: list[Alert]) -> None:
        del alerts  # EMF carries scores, not alert decisions
