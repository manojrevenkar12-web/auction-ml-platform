"""Run the pre-ship review on every spec (including the deliberately broken ones)
and write docs/results/review_*.md plus a summary JSON."""

from __future__ import annotations

import json
from pathlib import Path

from aucml.config import Catalog, discover_specs
from aucml.pipeline import Workspace, run
from aucml.review import review, to_markdown, verdict

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "results"


def main() -> None:
    ws = Workspace(ROOT)
    cat = Catalog.load(ROOT / "feature_repo")
    summary = {}
    for spec in discover_specs(ROOT / "models", include_experiments=True):
        r = run(spec, cat, ws)
        findings = review(spec, cat, r.train, r.test)
        (OUT / f"review_{spec.name}.md").write_text(to_markdown(spec, findings))
        summary[spec.name] = {
            "verdict": verdict(findings),
            "errors": [f.check for f in findings if f.severity == "error" and not f.passed],
            "metrics": r.evaluation["metrics"],
            "baseline": r.evaluation["baseline"],
            "rows": r.evaluation["rows"],
            "timings_s": {k: round(v, 2) for k, v in r.timings.items()},
        }
        print(spec.name, summary[spec.name]["verdict"], summary[spec.name]["errors"])
    (OUT / "reviews.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
