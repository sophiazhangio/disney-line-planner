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


def test_replan_from_a_ride(monkeypatch):
    monkeypatch.setattr(main, "live_profile", lambda close: constant_profile({284: 30, 279: 20, 289: 10}, close))
    r = client.post("/api/plan", json={"ride_ids": [284, 279, 289], "start": "11:00", "from_ride_id": 284})
    assert r.status_code == 200
    body = r.json()
    assert sorted(s["ride_id"] for s in body["plan"]["steps"]) == [279, 289]   # 284 is where we are
    assert body["start_point"]["name"] == "Space Mountain"


def test_blend_mode_uses_live_now_and_prediction_later(monkeypatch):
    import dlp.model
    from dlp.waits import WaitProfile

    monkeypatch.setattr(main, "_model_ready", lambda: True)
    monkeypatch.setattr(dlp.model, "predicted_profile",
                        lambda day, ids, close: WaitProfile({i: [30.0] * 96 for i in ids}, 15, close, "predicted"))
    monkeypatch.setattr(main, "fetch_live_waits", lambda: {284: {"wait": 90.0, "is_open": True}})
    r = client.post("/api/plan", json={"ride_ids": [284, 279], "mode": "blend"})
    assert r.status_code == 200
    assert r.json()["waits"].startswith("live now")


def test_blend_without_model_falls_back_to_live(monkeypatch):
    monkeypatch.setattr(main, "_model_ready", lambda: False)
    monkeypatch.setattr(main, "live_profile", lambda close: constant_profile({284: 30, 279: 20}, close))
    r = client.post("/api/plan", json={"ride_ids": [284, 279], "mode": "blend"})
    assert r.status_code == 200 and "not trained yet" in r.json()["waits"]
