from datetime import date

import numpy as np
import pandas as pd

from dlp.model import TZ, add_features, predicted_profile, open_slots, train_and_evaluate
from dlp.search import plan_astar


def synthetic_history(days=21, rides=(284, 279, 6340), seed=0):
    rng = np.random.default_rng(seed)
    times = pd.date_range("2026-10-01 08:00", periods=days * 24 * 6, freq="10min", tz=TZ)
    times = times[(times.hour >= 8) & (times.hour < 23)]
    rows = []
    for i, rid in enumerate(rides):
        base = 20 + 20 * i
        hour = times.hour + times.minute / 60
        shape = np.exp(-((hour - 14) ** 2) / 18)  # busiest mid-afternoon
        weekend = np.where(times.day_of_week >= 5, 1.5, 1.0)
        wait = base * (0.3 + shape) * weekend + rng.normal(0, 6, len(times))
        rows.append(pd.DataFrame({"ride_id": rid, "local": times,
                                  "wait_time": np.clip(np.round(wait / 5) * 5, 0, None), "is_open": 1}))
    return add_features(pd.concat(rows, ignore_index=True))


def test_model_beats_last_week_baseline():
    df = synthetic_history()
    model, m = train_and_evaluate(df)
    b = m["baseline_last_week"]
    assert b["rows"] > 0
    assert b["model_mae"] < b["baseline_mae"]


def test_predicted_profile_feeds_planner():
    df = synthetic_history()
    model, _ = train_and_evaluate(df)
    bundle = {"model": model, "open": open_slots(df)}
    prof = predicted_profile(date(2026, 10, 24), [284, 279, 6340], bundle=bundle)
    assert prof.wait(284, 6 * 60) == float("inf")      # never seen open at 6am
    assert prof.wait(6340, 14 * 60) > prof.wait(6340, 9 * 60)
    plan = plan_astar([284, 279, 6340], 9 * 60, prof)
    assert len(plan.steps) == 3
