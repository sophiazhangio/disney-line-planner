"""Wait-time model: gradient boosting on ride, weekday, time of day and holidays.

  python -m dlp.model            # train, evaluate, save model + results/metrics.json

Evaluation holds out the most recent 7 days and compares against a simple
baseline: "the wait for this ride at the same time last week".
"""
from __future__ import annotations

import json
import math
import os

# HistGradientBoosting's OpenMP threads thrash on small data; one thread is faster here.
os.environ.setdefault("OMP_NUM_THREADS", "1")
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path

import holidays
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .waits import DAY, WaitProfile

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
MODEL_PATH = ROOT / "model" / "wait_model.joblib"
METRICS_PATH = ROOT / "results" / "metrics.json"
TZ = "America/Los_Angeles"
SLOT = 15
FEATURES = ["ride_code", "dow", "minute", "is_weekend", "is_holiday", "near_holiday", "month"]
CATEGORICAL = ["ride_code"]


@lru_cache(maxsize=4)
def _holiday_dates(years: tuple[int, ...]) -> frozenset[date]:
    return frozenset(holidays.country_holidays("US", subdiv="CA", years=list(years)).keys())


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Expects columns ride_id and local (tz-aware local timestamp)."""
    out = df.copy()
    local = out["local"]
    out["date"] = local.dt.date
    out["dow"] = local.dt.day_of_week
    out["minute"] = local.dt.hour * 60 + local.dt.minute
    out["is_weekend"] = (out["dow"] >= 5).astype(int)
    out["month"] = local.dt.month
    years = tuple(sorted(set(local.dt.year) | {y + 1 for y in set(local.dt.year)} | {y - 1 for y in set(local.dt.year)}))
    hol = _holiday_dates(years)
    out["is_holiday"] = out["date"].map(lambda d: int(d in hol))
    # Long weekends: within 2 days of a holiday.
    out["near_holiday"] = out["date"].map(lambda d: int(any(d + timedelta(k) in hol for k in range(-2, 3))))
    out["ride_code"] = out["ride_id"].astype(int)
    return out


def load_history(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    files = sorted(raw_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no data in {raw_dir}")
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    df["local"] = pd.to_datetime(df["timestamp_utc"], utc=True).dt.tz_convert(TZ)
    df = df.drop_duplicates(["ride_id", "timestamp_utc"])
    return add_features(df)


def same_time_last_week(df: pd.DataFrame) -> pd.Series:
    """Baseline: wait for the same ride, 7 days earlier, nearest 10-minute mark."""
    key = df["local"].dt.round("10min")
    lookup = pd.Series(df["wait_time"].to_numpy(), index=pd.MultiIndex.from_arrays([df["ride_id"], key]))
    lookup = lookup[~lookup.index.duplicated()]
    prev = pd.MultiIndex.from_arrays([df["ride_id"], key - pd.Timedelta(days=7)])
    return pd.Series(lookup.reindex(prev).to_numpy(), index=df.index)


def make_model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="absolute_error", max_iter=400, learning_rate=0.08, max_leaf_nodes=63,
        min_samples_leaf=20, categorical_features=CATEGORICAL, random_state=0,
    )


def train_and_evaluate(df: pd.DataFrame, test_days: int = 7) -> tuple[HistGradientBoostingRegressor, dict]:
    open_rows = df[df["is_open"] == 1].copy()
    baseline = same_time_last_week(open_rows)
    cutoff = open_rows["local"].max().normalize() - pd.Timedelta(days=test_days - 1)
    train, test = open_rows[open_rows["local"] < cutoff], open_rows[open_rows["local"] >= cutoff]
    if train.empty or test.empty:
        raise ValueError("need more than test_days of data to evaluate")

    model = make_model().fit(train[FEATURES], train["wait_time"])
    pred = np.clip(model.predict(test[FEATURES]), 0, None)
    err = np.abs(pred - test["wait_time"].to_numpy())

    base = baseline.loc[test.index]
    has_base = base.notna().to_numpy()
    # Second baseline: average wait for that ride and hour in the training data.
    hour_avg = train.groupby(["ride_id", train["minute"] // 60])["wait_time"].mean()
    ha = hour_avg.reindex(pd.MultiIndex.from_arrays([test["ride_id"], test["minute"] // 60])).to_numpy()
    has_ha = ~np.isnan(ha)

    metrics = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "data_from": str(open_rows["local"].min().date()), "data_to": str(open_rows["local"].max().date()),
        "train_rows": int(len(train)), "test_rows": int(len(test)), "rides": int(open_rows["ride_id"].nunique()),
        "model_mae_all": round(float(err.mean()), 2),
        "baseline_last_week": {
            "rows": int(has_base.sum()),
            "model_mae": round(float(err[has_base].mean()), 2) if has_base.any() else None,
            "baseline_mae": round(float(np.abs(base[has_base] - test["wait_time"][has_base]).mean()), 2) if has_base.any() else None,
        },
        "baseline_ride_hour_avg": {
            "rows": int(has_ha.sum()),
            "model_mae": round(float(err[has_ha].mean()), 2) if has_ha.any() else None,
            "baseline_mae": round(float(np.abs(ha[has_ha] - test["wait_time"].to_numpy()[has_ha]).mean()), 2) if has_ha.any() else None,
        },
    }
    # Refit on everything for use in the app.
    final = make_model().fit(open_rows[FEATURES], open_rows["wait_time"])
    return final, metrics


def open_slots(df: pd.DataFrame) -> dict[tuple[int, int, int], float]:
    """Share of observations a ride was open, by (ride, weekend?, 15-min slot)."""
    g = df.groupby(["ride_id", "is_weekend", df["minute"] // SLOT])["is_open"].mean()
    return {tuple(int(x) for x in k): float(v) for k, v in g.items()}


def save(model, df: pd.DataFrame, path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "open": open_slots(df), "features": FEATURES}, path)


@lru_cache(maxsize=1)
def load_bundle(path: Path = MODEL_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError("no trained model yet: run `python -m dlp.model`")
    return joblib.load(path)


def predicted_profile(day: date, ride_ids: list[int], close_minute: float = DAY,
                      bundle: dict | None = None) -> WaitProfile:
    bundle = bundle or load_bundle()
    model, open_share = bundle["model"], bundle["open"]
    n = DAY // SLOT
    starts = pd.date_range(pd.Timestamp(day, tz=TZ), periods=n, freq=f"{SLOT}min")
    weekend = int(day.weekday() >= 5)
    slots: dict[int, list[float]] = {}
    for rid in ride_ids:
        frame = add_features(pd.DataFrame({"ride_id": rid, "local": starts + pd.Timedelta(minutes=SLOT / 2)}))
        pred = np.clip(model.predict(frame[FEATURES]), 0, None)
        # Closed if the ride was usually closed at that time (or never seen).
        slots[rid] = [
            round(float(p), 1) if open_share.get((rid, weekend, i), 0.0) >= 0.5 else math.inf
            for i, p in enumerate(pred)
        ]
    return WaitProfile(slots, SLOT, close_minute, source=f"predicted for {day.isoformat()}")


MIN_DAYS = 14  # one week to train on + one held-out week with a "last week" to compare to


def main() -> int:
    df = load_history()
    days = df["date"].nunique()
    if days < MIN_DAYS:
        print(f"days of data: {days}; need {MIN_DAYS} to train and evaluate. Skipping.")
        return 0
    model, metrics = train_and_evaluate(df)
    metrics["days_of_data"] = int(days)
    save(model, df)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
