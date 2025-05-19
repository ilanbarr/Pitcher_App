#!/usr/bin/env python3
"""
Pitcher Velocity Drop Detector – **pandas edition** (fixed for current ``pybaseball``)

The previous revision passed ``player_id`` to ``pybaseball.statcast`` but recent
versions (≥ 2.2) route individual‐player queries through
``pybaseball.statcast_pitcher`` or ``statcast_batter``. This update switches to
``statcast_pitcher`` and drops the incompatible keyword argument.

Usage
-----
::

    pip install pandas requests "pybaseball>=2.2"

    python pitcher_velocity_app.py            # today (Europe/Dublin)
    python pitcher_velocity_app.py --date 2025-05-19

"""
from __future__ import annotations

import argparse
import datetime as dt
from typing import Dict, List

import pandas as pd
import requests
from pybaseball import statcast_pitcher  # ← dedicated pitcher endpoint

MLB_SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"


def get_probable_pitchers(date: dt.date) -> List[Dict]:
    """Return a list of probable starters {{id, name}} for *date*."""
    params = {
        "sportId": 1,  # MLB
        "hydrate": "probablePitcher",
        "date": date.isoformat(),
    }
    r = requests.get(MLB_SCHEDULE_URL, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()

    pitchers: list[dict] = []
    for d in data.get("dates", []):
        for g in d["games"]:
            for side in ("home", "away"):
                entry = g["teams"][side].get("probablePitcher")
                if entry:
                    pitchers.append({"id": entry["id"], "name": entry["fullName"]})

    # deduplicate by id (keep latest name spelling)
    return [{"id": k, "name": v} for k, v in {p["id"]: p["name"] for p in pitchers}.items()]


def fetch_pitcher_statcast(pid: int, start: dt.date, end: dt.date) -> pd.DataFrame:
    """Pitch-level Statcast for *pid* between *start* and *end* inclusive."""
    try:
        df = statcast_pitcher(start.isoformat(), end.isoformat(), pid)
    except Exception as exc:  # network, maintenance, etc.
        print(f"⚠️  Statcast query failed for {pid}: {exc}")
        return pd.DataFrame()

    return df if not df.empty else pd.DataFrame()


def compute_velocity_changes(df: pd.DataFrame) -> pd.DataFrame:
    """Compute ΔV for each pitch type in the latest appearance of *df*."""
    if df.empty:
        return df

    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date

    appearance = (
        df.groupby(["game_date", "pitch_type"], as_index=False)["release_speed"].mean()
        .rename(columns={"release_speed": "avg_v"})
        .sort_values("game_date")
    )

    # rolling baseline (three prior appearances) per pitch type
    appearance["baseline_v"] = (
        appearance.groupby("pitch_type")["avg_v"].transform(lambda s: s.shift(1).rolling(3, min_periods=3).mean())
    )

    latest_date = appearance["game_date"].max()
    latest = appearance[appearance["game_date"] == latest_date].copy()
    latest["delta_v"] = latest["avg_v"] - latest["baseline_v"]
    return latest


def summarize_pitcher(pid: int, name: str, today: dt.date, lookback_days: int = 35):
    start = today - dt.timedelta(days=lookback_days)
    stats = fetch_pitcher_statcast(pid, start, today)
    if stats.empty:
        return None
    summary = compute_velocity_changes(stats)
    if summary.empty:
        return None

    summary["pitcher"] = name
    summary["mlbam_id"] = pid
    return summary[[
        "pitcher",
        "mlbam_id",
        "game_date",
        "pitch_type",
        "avg_v",
        "baseline_v",
        "delta_v",
    ]]


def main(date_iso: str | None = None):
    today = dt.date.fromisoformat(date_iso) if date_iso else dt.date.today()

    pitchers = get_probable_pitchers(today)
    if not pitchers:
        print(f"⚠️  No probable pitchers returned for {today}")
        return

    frames: list[pd.DataFrame] = []
    for p in pitchers:
        res = summarize_pitcher(p["id"], p["name"], today)
        if res is not None:
            frames.append(res)

    if not frames:
        print("⚠️  No Statcast data within look-back window for probable starters.")
        return

    report = pd.concat(frames).sort_values("delta_v").reset_index(drop=True)
    report[["avg_v", "baseline_v", "delta_v"]] = report[["avg_v", "baseline_v", "delta_v"]].round(2)

    pd.set_option("display.max_rows", None)
    print(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect velocity drops for today’s probable pitchers (pandas)")
    parser.add_argument("--date", help="Date YYYY-MM-DD (default = today)")
    args = parser.parse_args()

    main(args.date)