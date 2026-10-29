"""FastAPI app: pick rides and a start time, get the fastest order.

  uvicorn app.main:app --reload
"""
from __future__ import annotations

import math
from datetime import date
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from dlp.park import ESPLANADE, START, load_rides
from dlp.search import Infeasible, Plan, plan_astar, plan_greedy
from dlp.waits import fetch_live_waits, live_profile

STATIC = Path(__file__).parent / "static"
MAX_RIDES = 12

app = FastAPI(title="Disneyland Line Planner")


class PlanRequest(BaseModel):
    ride_ids: list[int] = Field(min_length=1, max_length=MAX_RIDES)
    start: str = "09:00"
    close: str = "23:59"
    mode: str = "live"          # "live" or "predicted"
    date: str | None = None     # visit date for predicted mode (YYYY-MM-DD)


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
    return {"rides": out, "start": {"lat": ESPLANADE[0], "lon": ESPLANADE[1]}, "max_rides": MAX_RIDES}


@app.post("/api/plan")
def plan(req: PlanRequest) -> dict:
    known = load_rides()
    bad = [r for r in req.ride_ids if r not in known]
    if bad:
        raise HTTPException(422, f"unknown ride ids {bad}")
    start, close = _minutes(req.start), _minutes(req.close)
    if req.mode == "predicted":
        from dlp.model import predicted_profile
        try:
            day = date.fromisoformat(req.date) if req.date else date.today()
            profile = predicted_profile(day, req.ride_ids, close)
        except FileNotFoundError:
            raise HTTPException(409, "The prediction model is not trained yet. Use live waits for now.")
    else:
        profile = live_profile(close)

    closed = [known[r].name for r in req.ride_ids if math.isinf(profile.min_wait_from(r, start))]
    if closed:
        raise HTTPException(409, "Closed for the rest of the day: " + ", ".join(closed))
    try:
        best = plan_astar(req.ride_ids, start, profile)
    except Infeasible as exc:
        raise HTTPException(409, f"Can't fit all of these before closing: {exc}")
    baseline = None
    try:
        baseline = _plan_json(plan_greedy(req.ride_ids, start, profile, "nearest"))
    except Infeasible:
        pass
    return {"waits": profile.source, "plan": _plan_json(best), "baseline_nearest": baseline,
            "start_point": {"id": START, "lat": ESPLANADE[0], "lon": ESPLANADE[1]}}
