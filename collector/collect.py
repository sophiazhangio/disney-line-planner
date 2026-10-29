"""Pull live wait times from Queue-Times.com and append them to daily CSV files.

Data source: Queue-Times.com (https://queue-times.com/) - free API, attribution required.
Run: python collector/collect.py   (writes data/raw/YYYY-MM-DD.csv, UTC date)
"""
from __future__ import annotations

import csv
import json
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

PARKS = {16: "Disneyland", 17: "Disney California Adventure"}
URL = "https://queue-times.com/parks/{park_id}/queue_times.json"
FIELDS = ["timestamp_utc", "park_id", "park", "ride_id", "ride", "land",
          "is_open", "wait_time", "last_updated"]
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def fetch(park_id: int, retries: int = 3) -> dict:
    req = urllib.request.Request(
        URL.format(park_id=park_id),
        headers={"User-Agent": "disney-line-planner (github.com/sophiazhangio/disney-line-planner)"},
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                return json.load(resp)
        except Exception as exc:  # network hiccup: back off and retry
            if attempt == retries - 1:
                raise
            print(f"park {park_id}: {exc}, retrying", file=sys.stderr)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("unreachable")


def flatten(park_id: int, payload: dict, now: str) -> list[dict]:
    rows = []
    groups = [(land["name"], land["rides"]) for land in payload.get("lands", [])]
    groups.append(("", payload.get("rides", [])))  # rides not assigned to a land
    for land, rides in groups:
        for r in rides:
            rows.append({
                "timestamp_utc": now,
                "park_id": park_id,
                "park": PARKS[park_id],
                "ride_id": r["id"],
                "ride": r["name"],
                "land": land,
                "is_open": int(bool(r["is_open"])),
                "wait_time": r["wait_time"],
                "last_updated": r.get("last_updated", ""),
            })
    return rows


def main() -> int:
    now_dt = datetime.now(timezone.utc).replace(microsecond=0)
    now = now_dt.isoformat().replace("+00:00", "Z")
    rows, failures = [], 0
    for park_id in PARKS:
        try:
            rows += flatten(park_id, fetch(park_id), now)
        except Exception as exc:
            failures += 1
            print(f"park {park_id} failed: {exc}", file=sys.stderr)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = DATA_DIR / f"{now_dt.date().isoformat()}.csv"
    new_file = not out.exists()
    with out.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerows(rows)
    print(f"{now}: wrote {len(rows)} rows to {out.name}")
    return 1 if failures == len(PARKS) else 0


if __name__ == "__main__":
    sys.exit(main())
