"""Compare A*, UCS and greedy orders on one saved snapshot of real waits.

Uses the first snapshot in data/raw/2026-10-29.csv so the numbers are reproducible.
Run: python scripts/benchmark.py
"""
from __future__ import annotations

import math
import random
import statistics
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dlp.park import load_rides  # noqa: E402
from dlp.search import plan_astar, plan_greedy, plan_ucs  # noqa: E402
from dlp.waits import constant_profile  # noqa: E402

SNAPSHOT = Path(__file__).resolve().parent.parent / "data" / "raw" / "2026-10-29.csv"


def main(trials: int = 40, k: int = 8, start: float = 9 * 60) -> None:
    df = pd.read_csv(SNAPSHOT)
    snap = df[df["timestamp_utc"] == df["timestamp_utc"].min()]
    waits = {int(r.ride_id): (float(r.wait_time) if r.is_open else math.inf) for r in snap.itertuples()}
    profile = constant_profile(waits, close_minute=24 * 60)
    rides = load_rides()
    pool = [rid for rid, w in waits.items() if rid in rides and not rides[rid].single_rider and 10 <= w < math.inf]

    rng = random.Random(0)
    rows = []
    for _ in range(trials):
        picks = rng.sample(pool, k)
        t0 = time.perf_counter(); a = plan_astar(picks, start, profile); ta = time.perf_counter() - t0
        t0 = time.perf_counter(); u = plan_ucs(picks, start, profile); tu = time.perf_counter() - t0
        g1 = plan_greedy(picks, start, profile, "nearest")
        g2 = plan_greedy(picks, start, profile, "shortest-wait")
        assert abs(a.finish - u.finish) < 1e-6
        rows.append((a.total_minutes, g1.total_minutes, g2.total_minutes, a.total_walk, g1.total_walk,
                     g2.total_walk, a.expanded, u.expanded, ta, tu))
    col = list(zip(*rows))
    mean = lambda i: statistics.mean(col[i])
    print(f"snapshot {snap['timestamp_utc'].iloc[0]}, {trials} random sets of {k} rides (wait >= 10 min), start 09:00, waits held constant")
    print(f"| order | total time (min) | walking (min) |")
    print(f"|---|---|---|")
    print(f"| A* (optimal) | {mean(0):.0f} | {mean(3):.0f} |")
    print(f"| greedy: nearest ride next | {mean(1):.0f} | {mean(4):.0f} |")
    print(f"| greedy: shortest line next | {mean(2):.0f} | {mean(5):.0f} |")
    print(f"\nstates expanded: A* {mean(6):.0f} vs UCS {mean(7):.0f} "
          f"({mean(7) / mean(6):.1f}x fewer); time A* {1000 * mean(8):.0f} ms vs UCS {1000 * mean(9):.0f} ms")


if __name__ == "__main__":
    main()
