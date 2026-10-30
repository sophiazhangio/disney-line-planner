"""Rewrite the model results section of README.md from results/metrics.json.

Run after `python -m dlp.model`. Does nothing if there are no metrics yet.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
METRICS = ROOT / "results" / "metrics.json"
START, END = "<!-- model-results:start -->", "<!-- model-results:end -->"


def render(m: dict) -> str:
    lw, ha = m["baseline_last_week"], m["baseline_ride_hour_avg"]

    def row(name: str, b: dict) -> str:
        if not b["rows"]:
            return f"| {name} | n/a | n/a | 0 |"
        better = 100 * (b["baseline_mae"] - b["model_mae"]) / b["baseline_mae"]
        change = f"{abs(better):.0f}% {'lower' if better >= 0 else 'higher'}"
        return f"| {name} | {b['baseline_mae']:.1f} | {b['model_mae']:.1f} ({change}) | {b['rows']:,} |"

    return "\n".join([
        f"Trained on {m['data_from']} to {m['data_to']} ({m.get('days_of_data', '?')} days, "
        f"{m['rides']} rides). Tested on the last 7 days, which the model did not see.",
        "",
        "| baseline | baseline MAE (min) | model MAE (min) | test rows |",
        "|---|---|---|---|",
        row("same time last week", lw),
        row("ride × hour average", ha),
        "",
        f"Model MAE on all {m['test_rows']:,} test rows: {m['model_mae_all']:.1f} min. "
        f"Each comparison uses only the test rows where that baseline has a value. "
        f"Last updated {m['trained_at'][:10]}.",
    ])


def main() -> int:
    if not METRICS.exists():
        print("no metrics yet; README unchanged")
        return 0
    text = README.read_text(encoding="utf-8")
    pattern = re.compile(re.escape(START) + r".*?" + re.escape(END), re.S)
    if not pattern.search(text):
        print("README markers not found", file=sys.stderr)
        return 1
    body = render(json.loads(METRICS.read_text()))
    README.write_text(pattern.sub(lambda _: f"{START}\n{body}\n{END}", text), encoding="utf-8")
    print("README results updated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
