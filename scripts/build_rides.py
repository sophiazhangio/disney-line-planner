"""Build data/rides.csv: every Queue-Times ride with coordinates and a ride length.

Coordinates come from OpenStreetMap (snapshot in data/osm/attractions.json,
(c) OpenStreetMap contributors, ODbL). Ride names are matched to OSM names by
normalising both and using difflib, with a short alias table for the ones that
differ (seasonal overlays, single-rider lines, shows inside other buildings).

Run: python scripts/build_rides.py
"""
from __future__ import annotations

import csv
import difflib
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OSM_FILE = ROOT / "data" / "osm" / "attractions.json"
RAW_DIR = ROOT / "data" / "raw"
OUT = ROOT / "data" / "rides.csv"

# Queue-Times name (normalised) -> OSM name (normalised) when they differ a lot.
ALIASES = {
    "haunted mansion holiday": "the haunted mansion",
    "haunted mansion": "the haunted mansion",
    "guardians of the galaxy monsters after dark": "guardians of the galaxy mission breakout",
    "maters graveyard jamboree": "maters junkyard jamboree",
    "luigis honkin haulowee": "luigis rollickin roadsters",
    "luigis honkin haulowen": "luigis rollickin roadsters",
    "meet disney princesses at royal hall": "royal hall",
    "opera house walt disney a magical life": "main street opera house",
    "walt disney a magical life": "main street opera house",
    "great moments with mr lincoln": "main street opera house",
    "mark twain riverboat": "frontierland landing",
    "sailing ship columbia": "frontierland landing",
    "pirates lair on tom sawyer island": "rafts to tom sawyer island",
    "disneyland monorail": "tomorrowland monorail station",
    "pixar short film spotlight": "magic eye theater",
    "main street vehicles": "town square",
    "the disney gallery": "town square",
    "turtle talk with crush": "disney animation",
    "soarin across america": "soarin around the world",
    "mickeys house and meet mickey mouse": "mickeys house",
    "frontierland shootin exposition": "shooting exposition",
    "pixar palaround nonswinging": "pixar palaround",
    "pixar palaround swinging": "pixar palaround",
    "world of color one": "world of color one",
    "adventureland treehouse inspired by walt disneys swiss family robinson": "adventureland treehouse",
}

# Approximate ride length in minutes (time from boarding to exit). Default 5.
RIDE_MINUTES = {
    "pirates of the caribbean": 15, "jungle cruise": 10, "indiana jones adventure": 10,
    "the haunted mansion": 9, "haunted mansion holiday": 9, "its a small world": 14,
    "matterhorn bobsleds": 4, "space mountain": 4, "big thunder mountain railroad": 4,
    "star wars rise of the resistance": 18, "millennium falcon smugglers run": 6,
    "mickey minnies runaway railway": 6, "star tours the adventures continue": 7,
    "buzz lightyear astro blasters": 5, "finding nemo submarine voyage": 13,
    "autopia": 6, "tianas bayou adventure": 11, "the many adventures of winnie the pooh": 4,
    "storybook land canal boats": 10, "peter pans flight": 3, "alice in wonderland": 4,
    "mr toads wild ride": 2, "snow whites enchanted wish": 2, "pinocchios daring journey": 3,
    "casey jr circus train": 4, "disneyland railroad": 20, "mark twain riverboat": 14,
    "sailing ship columbia": 14, "davy crocketts explorer canoes": 10,
    "walt disneys enchanted tiki room": 15, "great moments with mr lincoln": 15,
    "radiator springs racers": 5, "incredicoaster": 3, "guardians of the galaxy mission breakout": 3,
    "guardians of the galaxy monsters after dark": 3, "web slingers a spiderman adventure": 5,
    "soarin across america": 9, "grizzly river run": 6, "toy story midway mania": 7,
    "the little mermaid ariels undersea adventure": 7, "monsters inc mike sulley to the rescue": 4,
    "goofys sky school": 2, "turtle talk with crush": 15, "mickeys philharmagic": 12,
    "pixar short film spotlight": 15, "world of color one": 25, "the bakery tour": 10,
    "disneyland monorail": 10, "redwood creek challenge trail": 20,
    "opera house walt disney a magical life": 15, "walt disney a magical life": 15,
}


def norm(name: str) -> str:
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    s = s.lower().replace("&", " ").replace("-", "")
    s = re.sub(r"single rider", "", s)
    s = re.sub(r"[^a-z0-9 ]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def load_osm() -> dict[str, tuple[float, float]]:
    """OSM name -> mean (lat, lon) of all elements with that name."""
    pts: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for el in json.loads(OSM_FILE.read_text())["elements"]:
        name = el.get("tags", {}).get("name")
        c = el.get("center", el)
        if name and "lat" in c:
            pts[norm(name)].append((c["lat"], c["lon"]))
    return {k: (sum(p[0] for p in v) / len(v), sum(p[1] for p in v) / len(v)) for k, v in pts.items()}


def load_rides() -> dict[int, dict]:
    rides: dict[int, dict] = {}
    for f in sorted(RAW_DIR.glob("*.csv")):
        with f.open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                rides[int(r["ride_id"])] = r  # latest name/land wins
    return rides


def main() -> None:
    osm = load_osm()
    rows, unmatched = [], []
    for ride_id, r in sorted(load_rides().items()):
        key = norm(r["ride"])
        target = ALIASES.get(key, key)
        match = target if target in osm else None
        if match is None:
            close = difflib.get_close_matches(target, osm.keys(), n=1, cutoff=0.8)
            match = close[0] if close else None
        if match is None:
            unmatched.append(r["ride"])
            continue
        lat, lon = osm[match]
        rows.append({
            "ride_id": ride_id, "park_id": r["park_id"], "park": r["park"],
            "ride": r["ride"], "land": r["land"], "lat": round(lat, 6), "lon": round(lon, 6),
            "osm_name": match, "ride_minutes": RIDE_MINUTES.get(key, RIDE_MINUTES.get(match, 5)),
            "single_rider": int("single rider" in r["ride"].lower()),
        })

    # Rides with no OSM match: place them at the centre of their land.
    by_land: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_land[row["land"]].append(row)
    for name in unmatched:
        r = next(v for v in load_rides().values() if v["ride"] == name)
        peers = by_land.get(r["land"])
        if not peers:
            print(f"skip (no coordinates): {name}")
            continue
        rows.append({
            "ride_id": int(r["ride_id"]), "park_id": r["park_id"], "park": r["park"],
            "ride": name, "land": r["land"],
            "lat": round(sum(p["lat"] for p in peers) / len(peers), 6),
            "lon": round(sum(p["lon"] for p in peers) / len(peers), 6),
            "osm_name": "(land centre)", "ride_minutes": RIDE_MINUTES.get(norm(name), 5),
            "single_rider": 0,
        })
        print(f"land centre: {name} ({r['land']})")

    rows.sort(key=lambda x: (x["park_id"], x["land"], x["ride"]))
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rides to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
