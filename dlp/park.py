"""Ride locations and walking times for the Disneyland Resort."""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RIDES_CSV = ROOT / "data" / "rides.csv"

DISNEYLAND, DCA = 16, 17
# Park gates face each other across the Esplanade. Walking between parks goes
# through both gates, plus a few minutes for the re-entry scan.
GATES = {DISNEYLAND: (33.80985, -117.91897), DCA: (33.80795, -117.91898)}
ESPLANADE = (33.80890, -117.91897)
PARK_HOP_MINUTES = 5.0

WALK_METERS_PER_MIN = 70.0   # ~4.2 km/h, a crowded-park pace
DETOUR_FACTOR = 1.35         # paths are not straight lines

START = 0  # node id for the Esplanade (between the two gates)


@dataclass(frozen=True)
class Ride:
    id: int
    park_id: int
    name: str
    land: str
    lat: float
    lon: float
    ride_minutes: float
    single_rider: bool


@lru_cache(maxsize=1)
def load_rides() -> dict[int, Ride]:
    with RIDES_CSV.open(encoding="utf-8") as f:
        return {
            int(r["ride_id"]): Ride(
                id=int(r["ride_id"]), park_id=int(r["park_id"]), name=r["ride"],
                land=r["land"], lat=float(r["lat"]), lon=float(r["lon"]),
                ride_minutes=float(r["ride_minutes"]), single_rider=r["single_rider"] == "1",
            )
            for r in csv.DictReader(f)
        }


def haversine_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (*a, *b))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6_371_000 * math.asin(math.sqrt(h))


def _walk(a: tuple[float, float], b: tuple[float, float]) -> float:
    return haversine_m(a, b) * DETOUR_FACTOR / WALK_METERS_PER_MIN


def walk_minutes(src: int, dst: int) -> float:
    """Walking time between two nodes (START or ride ids)."""
    if src == dst:
        return 0.0
    rides = load_rides()
    if src == START or dst == START:
        ride = rides[dst if src == START else src]
        gate = GATES[ride.park_id]
        return _walk(ESPLANADE, gate) + _walk(gate, (ride.lat, ride.lon))
    a, b = rides[src], rides[dst]
    if a.park_id == b.park_id:
        return _walk((a.lat, a.lon), (b.lat, b.lon))
    ga, gb = GATES[a.park_id], GATES[b.park_id]
    return _walk((a.lat, a.lon), ga) + _walk(ga, gb) + _walk(gb, (b.lat, b.lon)) + PARK_HOP_MINUTES


def walk_matrix(nodes: list[int]) -> dict[tuple[int, int], float]:
    return {(a, b): walk_minutes(a, b) for a in nodes for b in nodes}
