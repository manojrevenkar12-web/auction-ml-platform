"""Plot weekly delayed-label performance from docs/results/production_replay.json."""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SURFACE, INK, MUTED, GRID, SERIES = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"
PANELS = [
    ("price_estimator", "mape", "Price estimator: weekly MAPE (lower is better)", "{:.0%}"),
    ("sale_probability", "auc", "Sale probability: weekly AUC (higher is better)", "{:.2f}"),
    ("transport_cost", "mape", "Transport cost: weekly MAPE (lower is better)", "{:.0%}"),
]


def main() -> None:
    s = json.loads((ROOT / "docs" / "results" / "production_replay.json").read_text())
    releases = {r["model"]: r for r in s["releases"]}
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})
    fig, axes = plt.subplots(3, 1, figsize=(9, 8.4), sharex=True, facecolor=SURFACE)
    for ax, (model, metric, title, fmt) in zip(axes, PANELS, strict=True):
        rows = pd.DataFrame(s["weeks"][model])
        col = f"prev_week_{metric}"
        # each row reports the previous week's traffic: plot it at that week
        rows["traffic_week"] = pd.to_datetime(rows["window"]) - timedelta(days=7)
        rows = rows.dropna(subset=[col])
        ax.set_facecolor(SURFACE)
        ax.axvspan(pd.Timestamp("2026-03-02"), pd.Timestamp("2026-03-23"), color=GRID, alpha=0.7, lw=0)
        ax.axvline(pd.Timestamp("2026-04-13"), color=MUTED, lw=1, ls=":")
        release_value = None
        for line in (ROOT / "registry" / model).glob("*/manifest.json"):
            m = json.loads(line.read_text())
            if m["version"] == releases[model]["version"]:
                release_value = m["metrics"][metric]
        if release_value is not None:
            ax.axhline(release_value, color=MUTED, lw=1, ls="--")
            ax.text(rows.traffic_week.min(), release_value, "at release ", color=MUTED, va="bottom", ha="left", fontsize=8)
        ax.plot(rows.traffic_week, rows[col], color=SERIES, lw=2, marker="o", ms=4, mec=SURFACE, mew=1)
        for r in s["retrains"]:
            if r["model"] == model and "prod" in r["lanes"]:
                deployed = pd.Timestamp(r["week"]) + timedelta(days=7)
                ax.axvline(deployed, color=SERIES, lw=1.2, alpha=0.45)
        ax.set_title(title, loc="left", color=INK, fontsize=10, pad=16)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _p, f=fmt: f.format(v)))
        ax.grid(axis="y", color=GRID, lw=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.suptitle("Six months of replayed production traffic, scored on labels as they arrived", x=0.01, y=0.995, ha="left", color=INK, fontsize=11)
    fig.text(
        0.01,
        0.962,
        "Grey band: EV price shock.  Dotted line: PL sellers onboarded.  "
        "Blue lines: a retrained version reached prod.  Dashed: metric at release.",
        color=MUTED,
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    out = ROOT / "docs" / "replay_timeline.png"
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print(out)


if __name__ == "__main__":
    main()
