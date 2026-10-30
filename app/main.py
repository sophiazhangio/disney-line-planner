"""FastAPI app: pick rides and a start time, get the fastest order.

  uvicorn app.main:app --reload
"""
from __future__ import annotations

import math
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from dlp.park import ESPLANADE, START, load_rides
from dlp.search import Infeasible, Plan, plan_astar, plan_greedy
from dlp.waits import blend_profile, fetch_live_waits, live_profile

PARK_TZ = ZoneInfo("America/Los_Angeles")

STATIC = Path(__file__).parent / "static"
MAX_RIDES = 12

app = FastAPI(title="Disneyland Line Planner")


class PlanRequest(BaseModel):
    ride_ids: list[int] = Field(min_length=1, max_length=MAX_RIDES)
    start: str = "09:00"
    close: str = "23:59"
    mode: str = "blend"         # "live", "predicted", or "blend" (live now, predicted later)
    date: str | None = None     # visit date (YYYY-MM-DD); defaults to today at the park
    from_ride_id: int | None = None  # where you are now; None = park entrance


def _minutes(hhmm: str) -> float:
    try:
        h, m = hhmm.split(":")
        return int(h) * 60 + int(m)
    except ValueError:
        raise HTTPException(422, f"bad time '{hhmm}', use HH:MM")


def _plan_json(p: Plan) -> dict:
    rides = load_rides()
    return {
        "algorithm": p.algorithm, "expanded": p.expanded,
        "start": p.start, "finish": p.finish, "total_minutes": round(p.total_minutes, 1),
        "total_wait": round(p.total_wait, 1), "total_walk": round(p.total_walk, 1),
        "total_rest": round(p.total_rest, 1),
        "steps": [{
            "ride_id": s.ride_id, "ride": s.ride, "land": rides[s.ride_id].land,
            "lat": rides[s.ride_id].lat, "lon": rides[s.ride_id].lon,
            "walk_minutes": s.walk_minutes, "arrive": round(s.arrive, 1), "join_line": round(s.join_line, 1),
            "wait_minutes": round(s.wait_minutes, 1), "board": round(s.board, 1), "ride_end": round(s.ride_end, 1),
        } for s in p.steps],
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/rides")
def rides() -> dict:
    try:
        live = fetch_live_waits()
    except Exception:
        live = {}
    out = []
    for r in load_rides().values():
        lv = live.get(r.id, {})
        out.append({"id": r.id, "name": r.name, "park_id": r.park_id, "land": r.land,
                    "lat": r.lat, "lon": r.lon, "single_rider": r.single_rider,
                    "is_open": lv.get("is_open"), "wait": lv.get("wait"), "updated": lv.get("updated")})
    out.sort(key=lambda x: (x["park_id"], x["land"], x["name"]))
    return {"rides": out, "start": {"lat": ESPLANADE[0], "lon": ESPLANADE[1]}, "max_rides": MAX_RIDES,
            "model_ready": _model_ready(), "park_now": _park_now().strftime("%Y-%m-%d %H:%M")}


def _park_now() -> datetime:
    return datetime.now(PARK_TZ)


def _model_ready() -> bool:
    from dlp.model import MODEL_PATH
    return MODEL_PATH.exists()


def _profile(req: PlanRequest, close: float):
    """Pick the wait source. Falls back to live waits when there is no model yet."""
    now = _park_now()
    day = date.fromisoformat(req.date) if req.date else now.date()
    is_today = day == now.date()
    if req.mode == "live" or not _model_ready():
        if req.mode != "live" and not is_today:
            raise HTTPException(409, "Predictions for other days need the model, which is not trained yet.")
        prof = live_profile(close)
        if req.mode != "live":
            prof.source = "live (prediction model not trained yet)"
        return prof
    from dlp.model import predicted_profile
    pred = predicted_profile(day, req.ride_ids + ([req.from_ride_id] if req.from_ride_id else []), close)
    if req.mode == "blend" and is_today:
        return blend_profile(pred, fetch_live_waits(), now.hour * 60 + now.minute)
    return pred


@app.post("/api/plan")
def plan(req: PlanRequest) -> dict:
    known = load_rides()
    bad = [r for r in req.ride_ids + ([req.from_ride_id] if req.from_ride_id else []) if r not in known]
    if bad:
        raise HTTPException(422, f"unknown ride ids {bad}")
    if req.mode not in ("live", "predicted", "blend"):
        raise HTTPException(422, "mode must be live, predicted or blend")
    start, close = _minutes(req.start), _minutes(req.close)
    here = req.from_ride_id or START
    todo = [r for r in req.ride_ids if r != here]
    if not todo:
        raise HTTPException(422, "Nothing left to plan.")
    profile = _profile(req, close)

    closed = [known[r].name for r in todo if math.isinf(profile.min_wait_from(r, start))]
    if closed:
        raise HTTPException(409, "Closed for the rest of the day: " + ", ".join(closed))
    try:
        best = plan_astar(todo, start, profile, start_node=here)
    except Infeasible as exc:
        raise HTTPException(409, f"Can't fit all of these before closing: {exc}")
    baseline = None
    try:
        baseline = _plan_json(plan_greedy(todo, start, profile, "nearest", start_node=here))
    except Infeasible:
        pass
    lat, lon = (known[here].lat, known[here].lon) if here != START else ESPLANADE
    name = known[here].name if here != START else "Park entrance"
    return {"waits": profile.source, "plan": _plan_json(best), "baseline_nearest": baseline,
            "start_point": {"id": here, "name": name, "lat": lat, "lon": lon}}
