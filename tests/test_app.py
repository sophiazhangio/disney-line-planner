from fastapi.testclient import TestClient

import app.main as main
from dlp.waits import constant_profile

client = TestClient(main.app)


def test_plan_endpoint(monkeypatch):
    monkeypatch.setattr(main, "live_profile", lambda close: constant_profile({284: 30, 279: 20, 289: 10}, close))
    r = client.post("/api/plan", json={"ride_ids": [284, 279, 289], "start": "09:00"})
    assert r.status_code == 200
    body = r.json()
    assert sorted(s["ride_id"] for s in body["plan"]["steps"]) == [279, 284, 289]
    assert body["plan"]["finish"] <= body["baseline_nearest"]["finish"]


def test_closed_ride_message(monkeypatch):
    monkeypatch.setattr(main, "live_profile", lambda close: constant_profile({284: 30, 279: float("inf")}, close))
    r = client.post("/api/plan", json={"ride_ids": [284, 279]})
    assert r.status_code == 409 and "Matterhorn" in r.json()["detail"]


def test_index_has_attribution():
    html = client.get("/").text
    assert 'href="https://queue-times.com/"' in html and "Powered by" in html
