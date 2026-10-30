"""Wait-time sources for the planner.

A wait source answers "how long is the line for ride r if I join at minute t?"
Times are minutes after local midnight (America/Los_Angeles). Waits are a step
function: constant within each slot (e.g. 15 minutes). `math.inf` = closed.
"""
from __future__ import annotations

import json
import math
import time
import urllib.request
from dataclasses import dataclass, field

QUEUE_TIMES_URL = "https://queue-times.com/parks/{park_id}/queue_times.json"
DAY = 24 * 60


@dataclass
class WaitProfile:
    """Per-ride waits by time slot, plus the park's closing time."""
    slots: dict[int, list[float]]          # ride id -> wait per slot (minutes or inf)
    slot_minutes: int = 15
    close_minute: float = DAY              # nobody can join a line after this
    source: str = "profile"
    _suffix_min: dict[int, list[float]] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        for rid, w in self.slots.items():
            suf, best = [0.0] * len(w), math.inf
            for i in range(len(w) - 1, -1, -1):
                best = min(best, w[i])
                suf[i] = best
            self._suffix_min[rid] = suf

    def _slot(self, t: float) -> int:
        return int(t // self.slot_minutes)

    def wait(self, ride_id: int, t: float) -> float:
        if t >= self.close_minute or ride_id not in self.slots:
            return math.inf
        w = self.slots[ride_id]
        i = self._slot(t)
        return w[i] if 0 <= i < len(w) else math.inf

    def min_wait_from(self, ride_id: int, t: float) -> float:
        """Smallest wait at any time >= t (used by the A* heuristic)."""
        suf = self._suffix_min.get(ride_id)
        i = self._slot(t)
        if suf is None or i >= len(suf) or t >= self.close_minute:
            return math.inf
        return suf[max(i, 0)]

    def best_boarding(self, ride_id: int, arrive: float) -> tuple[float, float]:
        """Earliest time you can board if you reach the ride at `arrive`.

        You may join the line right away or rest first and join later when the
        line is shorter. Returns (join_time, board_time). Because the wait is a
        step function, the best join time is `arrive` or the start of a later
        slot. Allowing this choice makes the travel time "first in, first out"
        (arriving earlier never makes you board later), which is what lets the
        search keep only the earliest arrival per state.
        """
        best_join, best_board = arrive, arrive + self.wait(ride_id, arrive)
        t = (self._slot(arrive) + 1) * self.slot_minutes
        while t < self.close_minute and t - arrive < best_board - arrive:
            board = t + self.wait(ride_id, t)
            if board < best_board:
                best_join, best_board = t, board
            t += self.slot_minutes
        return best_join, best_board


def constant_profile(waits: dict[int, float], close_minute: float = DAY, source: str = "constant") -> WaitProfile:
    n = DAY // 15
    return WaitProfile({rid: [w] * n for rid, w in waits.items()}, 15, close_minute, source)


_live_cache: dict[str, object] = {"at": 0.0, "data": None}


def fetch_live_waits(max_age_s: float = 300) -> dict[int, dict]:
    """Current waits from Queue-Times.com, cached for 5 minutes.

    Returns ride id -> {"wait": minutes, "is_open": bool, "updated": iso time}.
    """
    if _live_cache["data"] is not None and time.time() - float(_live_cache["at"]) < max_age_s:
        return _live_cache["data"]  # type: ignore[return-value]
    out: dict[int, dict] = {}
    for park_id in (16, 17):
        req = urllib.request.Request(QUEUE_TIMES_URL.format(park_id=park_id),
                                     headers={"User-Agent": "disney-line-planner"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            payload = json.load(resp)
        groups = [land["rides"] for land in payload.get("lands", [])] + [payload.get("rides", [])]
        for rides in groups:
            for r in rides:
                out[r["id"]] = {"wait": float(r["wait_time"]), "is_open": bool(r["is_open"]),
                                "updated": r.get("last_updated", "")}
    _live_cache.update(at=time.time(), data=out)
    return out


def live_profile(close_minute: float = DAY) -> WaitProfile:
    """Treat the current posted wait as the wait for the rest of the day."""
    live = fetch_live_waits()
    waits = {rid: (v["wait"] if v["is_open"] else math.inf) for rid, v in live.items()}
    return constant_profile(waits, close_minute, source="live")


def blend_profile(pred: WaitProfile, live: dict[int, dict], now: float,
                  tau: float = 60.0, closed_minutes: float = 60.0) -> WaitProfile:
    """Correct a predicted day using the waits posted right now.

    Live waits are the best guess for the next half hour, predictions for later
    in the day. So each ride's forecast is shifted by today's error
    (live - predicted, at `now`), and that shift fades out with time:

        wait(t) = predicted(t) + (live - predicted(now)) * exp(-(t - now) / tau)

    A ride that is closed right now is treated as closed for `closed_minutes`,
    then follows the prediction again. Slots before `now` are left unchanged.
    """
    slots: dict[int, list[float]] = {}
    size = pred.slot_minutes
    for rid, waits in pred.slots.items():
        lv = live.get(rid)
        if lv is None:
            slots[rid] = list(waits)
            continue
        out = list(waits)
        p_now = pred.wait(rid, now)
        for i, p in enumerate(waits):
            start, mid = i * size, i * size + size / 2
            if start + size <= now:
                continue  # already in the past
            dt = max(0.0, mid - now)
            if not lv["is_open"]:
                out[i] = math.inf if start < now + closed_minutes else p
            elif math.isinf(p_now) or math.isinf(p):
                # The model expected it closed but it is open: use the live wait
                # for the next hour, then trust the model's opening hours.
                out[i] = lv["wait"] if start < now + 60 else p
            else:
                out[i] = max(0.0, p + (lv["wait"] - p_now) * math.exp(-dt / tau))
            if start <= now < start + size and lv["is_open"]:
                out[i] = lv["wait"]  # the slot we are in now: exactly what is posted
        slots[rid] = out
    return WaitProfile(slots, size, pred.close_minute, source=f"live now, {pred.source} later")
