"""Pre-ship methodology review.

Catches the mistakes that make offline metrics lie, while they are still cheap
to fix: post-event features, non point-in-time joins, backward-window errors
(aggregates over values not yet known), random splits on temporal data, labels
that would not have existed at training time, and evaluation too thin to trust.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from aucml.config import Catalog, ModelSpec, column
from aucml.featurestore import AUDIT_EVENT, AUDIT_KNOWN

Severity = Literal["error", "warn", "info"]


@dataclass
class Finding:
    check: str
    severity: Severity
    passed: bool
    detail: str


def _pct(x: float) -> str:
    return f"{100 * x:.2f}%"


def review(spec: ModelSpec, catalog: Catalog, train: pd.DataFrame, test: pd.DataFrame) -> list[Finding]:
    out: list[Finding] = []
    s = spec.split

    # 1. temporal split
    out.append(
        Finding(
            "temporal_split",
            "error",
            s.kind == "time",
            "time-based split"
            if s.kind == "time"
            else "random split on time-ordered data: test rows sit "
            "between training rows, so the model is evaluated on the past it was trained around",
        )
    )

    # 2. label availability: no training label may be known after the earliest test prediction
    first_test = pd.to_datetime(test.as_of).min()
    late = float((pd.to_datetime(train.label_known_at) >= first_test).mean())
    out.append(
        Finding(
            "label_availability",
            "error",
            late == 0.0,
            f"{_pct(late)} of training labels were only known after the first test prediction ({first_test:%Y-%m-%d})",
        )
    )

    # 3. post-event features
    rank = catalog.point_rank(spec.prediction_point)
    post = [r for r in spec.features if catalog.point_rank(catalog.groups[catalog.resolve(r)[0]].available_at) > rank]
    out.append(
        Finding(
            "post_event_features",
            "error",
            not post,
            f"features not available at '{spec.prediction_point}': {post}" if post else "all features exist at prediction time",
        )
    )

    # 4. point-in-time join (event_time audit) and 5. backward windows (known_at audit)
    frame = pd.concat([train, test])
    as_of = pd.to_datetime(frame.as_of)
    future_evt: dict[str, float] = {}
    future_known: dict[str, float] = {}
    for c in frame.columns:
        if c.startswith(AUDIT_EVENT):
            ts = pd.to_datetime(frame[c])
            future_evt[c.removeprefix(AUDIT_EVENT)] = float((ts > as_of).mean())
        elif c.startswith(AUDIT_KNOWN):
            ts = pd.to_datetime(frame[c])
            future_known[c.removeprefix(AUDIT_KNOWN)] = float((ts > as_of).mean())
    bad_evt = {g: v for g, v in future_evt.items() if v > 0}
    out.append(
        Finding(
            "point_in_time_join",
            "error",
            spec.point_in_time and not bad_evt,
            ("point_in_time disabled in spec; " if not spec.point_in_time else "")
            + (
                "rows joined to snapshots taken after the prediction: " + ", ".join(f"{g}={_pct(v)}" for g, v in bad_evt.items())
                if bad_evt
                else "every joined snapshot predates its prediction"
            ),
        )
    )
    bad_known = {g: v for g, v in future_known.items() if v > 0 and g not in bad_evt}
    out.append(
        Finding(
            "backward_window",
            "error",
            not bad_known,
            "snapshots aggregate values that were not yet known at prediction time: " + ", ".join(f"{g}={_pct(v)}" for g, v in bad_known.items())
            if bad_known
            else "every aggregated value was known before prediction time",
        )
    )

    # 6. univariate leakage probe
    y = train.label.astype(float).to_numpy()
    suspicious: list[str] = []
    for ref in spec.features:
        if catalog.resolve(ref)[2] != "numeric":
            continue
        x = pd.to_numeric(train[column(ref)], errors="coerce").to_numpy(dtype=float)
        ok = ~np.isnan(x)
        if ok.sum() < 100 or np.nanstd(x) == 0:
            continue
        if spec.task == "classification":
            score = abs(float(roc_auc_score(y[ok], x[ok])) - 0.5) * 2
        else:
            score = abs(float(spearmanr(x[ok], y[ok]).statistic))
        if score > 0.95:
            suspicious.append(f"{ref} ({score:.3f})")
    out.append(
        Finding(
            "univariate_leakage_probe",
            "warn",
            not suspicious,
            f"single features that almost determine the label: {suspicious}" if suspicious else "no single feature determines the label",
        )
    )

    # 7. training/serving feature coverage
    gaps = []
    for ref in spec.features:
        c = column(ref)
        d = abs(float(train[c].isna().mean()) - float(test[c].isna().mean()))
        if d > 0.10:
            gaps.append(f"{ref} ({_pct(float(train[c].isna().mean()))} vs {_pct(float(test[c].isna().mean()))})")
    out.append(
        Finding(
            "missingness_shift",
            "warn",
            not gaps,
            f"feature missing far more often in one split: {gaps}" if gaps else "missing-value rates consistent across splits",
        )
    )

    # 8. evaluation adequacy
    g = spec.gate
    out.append(Finding("test_size", "error", len(test) >= g.min_test_rows, f"test rows={len(test)} (min {g.min_test_rows})"))
    out.append(
        Finding(
            "baseline_defined",
            "warn",
            spec.baseline is not None,
            f"baseline={spec.baseline}" if spec.baseline else "no baseline: improvement cannot be judged",
        )
    )
    weeks = (pd.to_datetime(test.as_of).max() - pd.to_datetime(test.as_of).min()).days / 7
    out.append(Finding("test_window", "warn", weeks >= 4, f"test window spans {weeks:.1f} weeks"))
    small = []
    for ref in spec.slices:
        counts = test[column(ref)].astype(str).value_counts()
        small += [f"{ref}={v} (n={n})" for v, n in counts.items() if n < g.min_slice_rows]
    out.append(
        Finding(
            "slice_coverage",
            "info",
            not small,
            f"slices too small to gate on: {small}" if small else "every slice large enough to gate on",
        )
    )
    return out


def verdict(findings: list[Finding]) -> str:
    if any(f.severity == "error" and not f.passed for f in findings):
        return "block"
    if any(f.severity == "warn" and not f.passed for f in findings):
        return "pass_with_warnings"
    return "pass"


def as_dict(findings: list[Finding]) -> dict[str, Any]:
    return {"verdict": verdict(findings), "findings": [asdict(f) for f in findings]}


def to_markdown(spec: ModelSpec, findings: list[Finding]) -> str:
    icon = {True: "PASS", False: "FAIL"}
    lines = [
        f"# Pre-ship review: `{spec.name}`",
        "",
        f"**Verdict: {verdict(findings)}**",
        "",
        "| check | severity | result | detail |",
        "|---|---|---|---|",
    ]
    for f in findings:
        lines.append(f"| {f.check} | {f.severity} | {icon[f.passed]} | {f.detail} |")
    return "\n".join(lines) + "\n"
