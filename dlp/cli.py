"""Command line: plan a ride order.

  python -m dlp.cli --rides "space mountain, matterhorn, indiana jones" --start 09:00
  python -m dlp.cli --rides ... --start 09:00 --date 2026-11-14 --mode predicted
"""
from __future__ import annotations

import argparse
from datetime import date

from .park import load_rides
from .search import Infeasible, Plan, plan_astar, plan_greedy, plan_ucs
from .waits import live_profile


def clock(m: float) -> str:
    m = int(round(m))
    h, mm = divmod(m, 60)
    return f"{h % 24:02d}:{mm:02d}"


def parse_clock(s: str) -> float:
    h, m = s.split(":")
    return int(h) * 60 + int(m)


def find_rides(query: str) -> list[int]:
    rides = load_rides()
    out = []
    for q in [x.strip().lower() for x in query.split(",") if x.strip()]:
        hits = [r for r in rides.values() if q in r.name.lower() and not r.single_rider]
        if not hits:
            raise SystemExit(f"no ride matches '{q}'")
        out.append(min(hits, key=lambda r: len(r.name)).id)
    return out


def print_plan(p: Plan) -> None:
    print(f"{'walk':>5} {'arrive':>6} {'line':>5} {'wait':>5} {'board':>5} {'done':>5}  ride")
    for s in p.steps:
        rest = f" (rest {s.join_line - s.arrive:.0f} min first)" if s.join_line > s.arrive else ""
        print(f"{s.walk_minutes:5.1f} {clock(s.arrive):>6} {clock(s.join_line):>5} "
              f"{s.wait_minutes:5.0f} {clock(s.board):>5} {clock(s.ride_end):>5}  {s.ride}{rest}")
    print(f"finish {clock(p.finish)}  total {p.total_minutes:.0f} min "
          f"(wait {p.total_wait:.0f}, walk {p.total_walk:.0f}, rest {p.total_rest:.0f})  "
          f"[{p.algorithm}, {p.expanded} states expanded]")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rides", required=True, help="comma-separated ride names (partial is fine)")
    ap.add_argument("--start", default="09:00", help="start time HH:MM (park local time)")
    ap.add_argument("--close", default="23:59", help="park closing time HH:MM")
    ap.add_argument("--mode", choices=["live", "predicted"], default="live")
    ap.add_argument("--date", default=date.today().isoformat(), help="visit date for predicted mode")
    ap.add_argument("--compare", action="store_true", help="also show greedy baselines and UCS")
    a = ap.parse_args()

    ids = find_rides(a.rides)
    start, close = parse_clock(a.start), parse_clock(a.close)
    if a.mode == "live":
        profile = live_profile(close)
    else:
        from .model import predicted_profile
        profile = predicted_profile(date.fromisoformat(a.date), ids, close)
    print(f"waits: {profile.source}\n")
    best = plan_astar(ids, start, profile)
    print_plan(best)
    if a.compare:
        for name, fn in (("ucs", lambda: plan_ucs(ids, start, profile)),
                         ("greedy-nearest", lambda: plan_greedy(ids, start, profile, "nearest")),
                         ("greedy-shortest-wait", lambda: plan_greedy(ids, start, profile, "shortest-wait"))):
            print()
            try:
                print_plan(fn())
            except Infeasible as exc:
                print(f"[{name}] cannot finish: {exc}")
    print("\nWait times: Queue-Times.com (https://queue-times.com/). Map data: (c) OpenStreetMap contributors.")


if __name__ == "__main__":
    main()
