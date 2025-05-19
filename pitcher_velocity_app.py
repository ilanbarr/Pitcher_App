#!/usr/bin/env python3
"""
Pitcher Performance Dashboard (v3)
=================================

**What’s new**
--------------
* **Trend lines** added for
  * Whiff % over time
  * Barrel % over time
  * Horizontal & vertical movement (inches) over time
  * xwOBA / wOBA over time
* Aggregate logic now converts movement to inches (`h_in`, `v_in`) so they’re
  easier to read and chart.
* UI stays single‑page: each metric gets its own simple line chart (no subplots).
* CLI unchanged.

Run UI:
```bash
streamlit run pitcher_app.py
```
"""
from __future__ import annotations

import argparse
import datetime as dt
import urllib.parse
from typing import Dict, List, Tuple

import pandas as pd
import requests

try:
    import streamlit as st
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    st = None  # type: ignore

from pybaseball import statcast_pitcher

MLB_SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
HEADSHOT_TEMPLATE = (
    "https://img.mlbstatic.com/mlb-photos/image/upload/"
    "w_{size},q_auto:best,f_auto/v1/people/{pid}/headshot/67/current"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def in_streamlit() -> bool:
    return st is not None and get_script_run_ctx() is not None  # type: ignore[arg-type]

# ---------------------------------------------------------------------------
# Data fetch
# ---------------------------------------------------------------------------

def get_probable_pitchers(date: dt.date) -> List[Dict]:
    params = {"sportId": 1, "hydrate": "probablePitcher", "date": date.isoformat()}
    r = requests.get(MLB_SCHEDULE_URL, params=params, timeout=20)
    r.raise_for_status()
    pitchers: list[dict] = []
    for d in r.json().get("dates", []):
        for g in d.get("games", []):
            for side in ("home", "away"):
                if (pp := g["teams"][side].get("probablePitcher")):
                    pitchers.append({"id": pp["id"], "name": pp["fullName"]})
    return [{"id": k, "name": v} for k, v in {p["id"]: p["name"] for p in pitchers}.items()]


if st is not None:
    @st.cache_data(show_spinner=False)
    def fetch_pitcher_statcast(pid: int, start: dt.date, end: dt.date) -> pd.DataFrame:  # type: ignore
        try:
            return statcast_pitcher(start.isoformat(), end.isoformat(), pid)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Statcast query failed for {pid}: {exc}")
            return pd.DataFrame()
else:
    def fetch_pitcher_statcast(pid: int, start: dt.date, end: dt.date) -> pd.DataFrame:  # type: ignore
        try:
            return statcast_pitcher(start.isoformat(), end.isoformat(), pid)
        except Exception:
            return pd.DataFrame()

# ---------------------------------------------------------------------------
# Metric calculations
# ---------------------------------------------------------------------------

def add_pitch_flags(df: pd.DataFrame) -> pd.DataFrame:
    miss_set = {
        "swinging_strike",
        "swinging_strike_blocked",
        "swinging_pitchout",
        "missed_bunt",
    }
    df["is_swing"] = df["description"].str.contains("swing", na=False) | df["description"].isin(
        [
            "foul",
            "foul_tip",
            "foul_bunt",
            "hit_into_play",
            "hit_into_play_score",
            "hit_into_play_no_out",
        ]
    )
    df["is_miss"] = df["description"].isin(miss_set)
    df["is_barrel"] = (
        (df["launch_speed"] >= 98) & df["launch_angle"].between(8, 26)
    )
    return df


def aggregate_metrics(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    df = add_pitch_flags(df)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date

    grp = df.groupby(["game_date", "pitch_type"], as_index=False)
    agg = grp.agg(
        avg_v=("release_speed", "mean"),
        avg_spin=("release_spin_rate", "mean"),
        avg_hmov=("pfx_x", "mean"),
        avg_vmov=("pfx_z", "mean"),
        swings=("is_swing", "sum"),
        misses=("is_miss", "sum"),
        barrels=("is_barrel", "sum"),
        pitches=("pitch_type", "size"),
        woba_value=("woba_value", "mean"),
        xwoba=("estimated_woba_using_speedangle", "mean"),
    )
    agg["whiff%"] = (agg["misses"] / agg["swings"]).round(3)
    agg["barrel%"] = (agg["barrels"] / agg["pitches"]).round(3)
    agg["wOBA"] = agg["woba_value"].round(3)
    agg["xwOBA"] = agg["xwoba"].round(3)

    # movement to inches (more intuitive)
    agg["h_in"] = (agg["avg_hmov"] * 12).round(2)
    agg["v_in"] = (agg["avg_vmov"] * 12).round(2)

    # ΔV baseline
    agg = agg.sort_values("game_date")
    agg["baseline_v"] = (
        agg.groupby("pitch_type")["avg_v"].transform(lambda s: s.shift(1).rolling(3, min_periods=3).mean())
    )
    agg["delta_v"] = agg["avg_v"] - agg["baseline_v"]

    latest = agg[agg["game_date"] == agg["game_date"].max()].copy()
    return agg, latest


def summarize_pitcher(pid: int, name: str, today: dt.date, lookback: int = 35):
    start = today - dt.timedelta(days=lookback)
    raw = fetch_pitcher_statcast(pid, start, today)
    if raw.empty:
        return None, None, None
    agg, latest = aggregate_metrics(raw)
    return agg, latest, raw

# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

if in_streamlit():
    st.set_page_config("Pitcher Dashboard", "⚾", layout="wide")

    # ------ Query‑param helpers ------
    def qp_get(key: str, default: str | None = None):
        if hasattr(st, "query_params"):
            proxy = st.query_params
        else:
            proxy = st.experimental_get_query_params()  # type: ignore[attr-defined]
        if key in proxy:
            v = proxy[key]
            return v[0] if isinstance(v, list) else v
        return default

    def qp_clear():
        if hasattr(st, "query_params"):
            st.query_params.clear()  # type: ignore[attr-defined]
        else:
            st.experimental_set_query_params()

    # DETAIL PAGE --------------------------------------------------------
    if qp_get("pid") is not None:
        pid = int(qp_get("pid"))
        name = qp_get("name", "Unknown")
        sel_date = dt.date.fromisoformat(qp_get("date", dt.date.today().isoformat()))
        lookback = int(qp_get("lookback", "35"))

        st.markdown(f"## {name} — Trends (last {lookback} days)")
        st.image(HEADSHOT_TEMPLATE.format(pid=pid, size=200), width=150)

        agg_df, latest_df, raw_df = summarize_pitcher(pid, name, sel_date, lookback)
        if agg_df is None:
            st.error("No Statcast data available.")
        else:
            # --- helper to make clean line charts ---
            def line_chart(metric: str, ylabel: str):
                plt.figure(figsize=(9, 4))
                for pt, grp in agg_df.groupby("pitch_type"):
                    plt.plot(grp["game_date"], grp[metric], marker="o", label=pt)
                plt.xlabel("Game Date"); plt.ylabel(ylabel)
                plt.title(f"{name} — {ylabel} by Pitch Type")
                plt.xticks(rotation=45); plt.legend(); st.pyplot(plt.gcf()); plt.clf()

            # velocity / spin
            line_chart("avg_v", "Velocity (mph)")
            line_chart("avg_spin", "Spin Rate (RPM)")
            # whiff / barrel
            line_chart("whiff%", "Whiff Rate")
            line_chart("barrel%", "Barrel Rate")
            # movement
            line_chart("h_in", "Horizontal Break (in)")
            line_chart("v_in", "Vertical Break (in)")
            # xwOBA / wOBA
            line_chart("xwOBA", "xwOBA")
            line_chart("wOBA", "wOBA")

            # Latest metrics table
            if latest_df is not None and not latest_df.empty:
                st.markdown("### Latest Appearance Metrics")
                cols = [
                    "pitch_type",
                    "avg_v",
                    "delta_v",
                    "avg_spin",
                    "whiff%",
                    "barrel%",
                    "wOBA",
                    "xwOBA",
                    "h_in",
                    "v_in",
                ]
                st.dataframe(latest_df[cols].round(3), use_container_width=True)

            with st.expander("Raw Statcast (recent)"):
                st.dataframe(raw_df, use_container_width=True)

        if st.button("← Back to list"):
            qp_clear()

    # MAIN DASHBOARD -----------------------------------------------------
    else:
        st.title("⚾ Pitcher Performance Dashboard")
        with st.sidebar:
            sel_date: dt.date = st.date_input("Game Date", dt.date.today())
            lookback = st.number_input("Look‑back window (days)", 7, 60, 35)
            fetch = st.button("Fetch Probables ✈️", use_container_width=True)

        if fetch:
            with st.status("Fetching probable pitchers…", expanded=False):
                probables = get_probable_pitchers(sel_date)
            if not probables:
                st.warning("No probable starters found.")
            else:
                for p in probables:
                    _, latest_df, _ = summarize_pitcher(p["id"], p["name"], sel_date, lookback)
                    if latest_df is None or latest_df.empty:
                        continue

                    delta_min = latest_df["delta_v"].min()
                    whiff_max = latest_df["whiff%"].max()
                    barrel_max = latest_df["barrel%"].max()
                    xwoba_mean = latest_df["xwOBA"].mean()
                    headshot = HEADSHOT_TEMPLATE.format(pid=p["id"], size=150)

                    with st.container(border=True):
                        c1, c2 = st.columns([1, 3])
                        with c1:
                            st.image(headshot, width=120)
                        with c2:
                            st.subheader(p["name"])
                            st.markdown(
                                f"**ΔV (min):** {delta_min:+.2f} mph | "
                                f"**Whiff% (max):** {whiff_max:.0%} | "
                                f"**Barrel% (max):** {barrel_max:.0%} | "
                                f"**xwOBA (avg):** {xwoba_mean:.3f}"
                            )
                            url = "?" + urllib.parse.urlencode(
                                {
                                    "pid": p["id"],
                                    "name": p["name"],
                                    "date": sel_date.isoformat(),
                                    "lookback": lookback,
                                }
                            )
                            st.link_button("Details ➜", url)

# ---------------------------------------------------------------------------
# CLI fallback
# ---------------------------------------------------------------------------

def cli_main(date_iso: str | None = None):
    date = dt.date.fromisoformat(date_iso) if date_iso else dt.date.today()
    probables = get_probable_pitchers(date)
    frames = []
    for p in probables:
        _, latest_df, _ = summarize_pitcher(p["id"], p["name"], date)
        if latest_df is not None:
            latest_df["pitcher"] = p["name"]
            frames.append(latest_df)

    if not frames:
        print("No recent Statcast data for probable starters.")
        return

    report = (
        pd.concat(frames)
            [
                "pitcher",
                "pitch_type",
                "avg_v",
                "delta_v",
                "avg_spin",
                "whiff%",
                "barrel%",
                "wOBA",
                "xwOBA",
            ]
            .round(3)
            .sort_values(["pitcher", "delta_v"])
        )
    
    print(report.to_string(index=False))


if __name__ == "__main__" and not in_streamlit():
    argp = argparse.ArgumentParser("Pitcher performance CLI")
    argp.add_argument("--date", help="YYYY-MM-DD (default today)")
    args = argp.parse_args()
    cli_main(args.date)
