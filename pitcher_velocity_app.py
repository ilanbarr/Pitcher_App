#!/usr/bin/env python3
"""
Pitcher Velocity Drop Detector + Streamlit Dashboard (robust version)
====================================================================

**Important note:**
Do **NOT** name this file `streamlit.py` or place it in a directory that
already contains a module named `streamlit.py`; doing so shadows the actual
*Streamlit* package and triggers obscure attribute errors (like the one you
just saw).  Stick with `pitcher_velocity_app.py` or another non‑conflicting
filename.

Run options
-----------
```bash
# CLI
python pitcher_velocity_app.py --date 2025-05-19

# Streamlit UI
streamlit run pitcher_velocity_app.py
```

Dependencies
------------
```
pip install pandas requests "pybaseball>=2.2" streamlit
```
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from typing import Dict, List

import pandas as pd
import requests

try:
    import streamlit as st  # pylint: disable=import-error
except ModuleNotFoundError:  # Streamlit not installed – fine for CLI mode
    st = None  # type: ignore

from pybaseball import statcast_pitcher

MLB_SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
HEADSHOT_TEMPLATE = (
    "https://img.mlbstatic.com/mlb-photos/image/upload/"
    "w_{size},q_auto:best,f_auto/v1/people/{pid}/headshot/67/current"
)

# ---------------------------------------------------------------------------
# Helper: detect if we’re running under `streamlit run`
# ---------------------------------------------------------------------------

def in_streamlit() -> bool:
    """True when executed via `streamlit run …` (any recent Streamlit version)."""
    if st is None:
        return False
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx  # type: ignore

        return get_script_run_ctx() is not None  # running inside Streamlit server
    except Exception:  # pragma: no cover
        return False


# ---------------------------------------------------------------------------
# Backend – same analytical routines as before
# ---------------------------------------------------------------------------

def get_probable_pitchers(date: dt.date) -> List[Dict]:
    params = {"sportId": 1, "hydrate": "probablePitcher", "date": date.isoformat()}
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
    # dedupe
    return [
        {"id": k, "name": v} for k, v in {p["id"]: p["name"] for p in pitchers}.items()
    ]


def fetch_pitcher_statcast(pid: int, start: dt.date, end: dt.date) -> pd.DataFrame:
    try:
        df = statcast_pitcher(start.isoformat(), end.isoformat(), pid)
    except Exception as exc:  # network hiccup, maintenance window, etc.
        print(f"⚠️  Statcast query failed for {pid}: {exc}")
        return pd.DataFrame()
    return df if not df.empty else pd.DataFrame()


def compute_velocity_changes(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
    appearance = (
        df.groupby(["game_date", "pitch_type"], as_index=False)["release_speed"].mean()
        .rename(columns={"release_speed": "avg_v"})
        .sort_values("game_date")
    )
    appearance["baseline_v"] = (
        appearance.groupby("pitch_type")["avg_v"].transform(
            lambda s: s.shift(1).rolling(3, min_periods=3).mean()
        )
    )
    latest_date = appearance["game_date"].max()
    latest = appearance[appearance["game_date"] == latest_date].copy()
    latest["delta_v"] = latest["avg_v"] - latest["baseline_v"]
    return latest


def summarize_pitcher(pid: int, name: str, today: dt.date, lookback: int = 35):
    start = today - dt.timedelta(days=lookback)
    stats = fetch_pitcher_statcast(pid, start, today)
    if stats.empty:
        return None
    summary = compute_velocity_changes(stats)
    if summary.empty:
        return None

    summary["pitcher"] = name
    summary["mlbam_id"] = pid
    cols = [
        "pitcher",
        "mlbam_id",
        "game_date",
        "pitch_type",
        "avg_v",
        "baseline_v",
        "delta_v",
    ]
    return summary[cols]

# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

if in_streamlit():
    st.set_page_config(
        page_title="Pitcher Velo Drops", page_icon="⚾", layout="wide"
    )
    st.title("⚾ Pitcher Velocity Drop Dashboard")

    # Sidebar controls
    with st.sidebar:
        st.header("Settings")
        sel_date: dt.date = st.date_input("Game Date", dt.date.today())
        lookback = st.number_input("Look‑back window (days)", 7, 60, 35)
        fetch_btn = st.button("Fetch Probable Pitchers ✈️", use_container_width=True)

    if fetch_btn:
        with st.status("Fetching probable pitchers…", expanded=False):
            pitchers = get_probable_pitchers(sel_date)
        if not pitchers:
            st.warning("No probable starters found for that date.")
        else:
            st.success(f"Found {len(pitchers)} probable starters.")
            for p in pitchers:
                summary_df = summarize_pitcher(p["id"], p["name"], sel_date, lookback)
                if summary_df is None:
                    st.info(f"No recent Statcast data for {p['name']}.")
                    continue

                headshot_url = HEADSHOT_TEMPLATE.format(pid=p["id"], size=150)
                pitch_types = ", ".join(summary_df["pitch_type"].unique())
                delta_min = summary_df["delta_v"].min()

                card = st.container(border=True)
                with card:
                    c1, c2 = st.columns([1, 3])
                    with c1:
                        st.image(headshot_url, width=120)
                    with c2:
                        st.subheader(p["name"])
                        st.markdown(f"**Pitch types:** {pitch_types}")
                        st.markdown(f"**Largest ΔV:** {delta_min:.2f} mph")

                        styled = (
                            summary_df.style
                            .format(
                                {
                                    "avg_v": "{:.2f}",
                                    "baseline_v": "{:.2f}",
                                    "delta_v": "{:.2f}",
                                }
                            )
                            .applymap(
                                lambda v: (
                                    "background-color:#ffe6e6"
                                    if isinstance(v, (float, int)) and v < -1
                                    else ""
                                )
                                if pd.notna(v)
                                else "",
                                subset=["delta_v"],
                            )
                        )
                        st.dataframe(styled, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# CLI fallback
# ---------------------------------------------------------------------------

def cli_main(date_iso: str | None = None):
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
        print("⚠️  No Statcast data within look‑back window for probable starters.")
        return

    report = pd.concat(frames).sort_values("delta_v").reset_index(drop=True)
    report[["avg_v", "baseline_v", "delta_v"]] = report[["avg_v", "baseline_v", "delta_v"]].round(
        2
    )

    pd.set_option("display.max_rows", None)
    print(report)


if __name__ == "__main__" and not in_streamlit():
    parser = argparse.ArgumentParser(
        description="Detect velocity drops for probable pitchers (CLI)"
    )
    parser.add_argument("--date", help="Date YYYY‑MM‑DD (default = today)")
    args = parser.parse_args()
    cli_main(args.date)
