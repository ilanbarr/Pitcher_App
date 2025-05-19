#!/usr/bin/env python3
"""
Pitcher Performance Dashboard
============================

* Streamlit UI with drill‑down charts **and** CLI fallback
* Metrics: **velocity, movement (pfx_x / pfx_z), spin‑rate, whiff%, barrel%**
* Compatible with both old and new Streamlit APIs (no deprecated attributes)

---
CLI:
```bash
python pitcher_app.py --date 2025-05-19
```
Streamlit:
```bash
streamlit run pitcher_app.py
```
Dependencies:
```
pip install pandas requests "pybaseball>=2.2" streamlit matplotlib
```
"""
from __future__ import annotations

import argparse
import datetime as dt
import urllib.parse
from typing import Dict, List, Tuple

import pandas as pd
import requests

# Optional heavy deps (only loaded when UI)
try:
    import streamlit as st  # type: ignore
    from streamlit.runtime.scriptrunner import get_script_run_ctx  # type: ignore
    import matplotlib.pyplot as plt  # type: ignore
except ModuleNotFoundError:  # running in CLI‑only environment
    st = None  # type: ignore

from pybaseball import statcast_pitcher

MLB_SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
HEADSHOT_TEMPLATE = (
    "https://img.mlbstatic.com/mlb-photos/image/upload/"
    "w_{size},q_auto:best,f_auto/v1/people/{pid}/headshot/67/current"
)

# ---------------------------------------------------------------------------
# Helper – detect Streamlit context
# ---------------------------------------------------------------------------

def in_streamlit() -> bool:
    return st is not None and get_script_run_ctx() is not None  # type: ignore[arg-type]

# ---------------------------------------------------------------------------
# Data fetch / caching
# ---------------------------------------------------------------------------

def get_probable_pitchers(date: dt.date) -> List[Dict]:
    """Return list[{id,name}] of probable starters for *date*."""
    params = {"sportId": 1, "hydrate": "probablePitcher", "date": date.isoformat()}
    r = requests.get(MLB_SCHEDULE_URL, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    pitchers = []
    for d in data.get("dates", []):
        for g in d.get("games", []):
            for side in ("home", "away"):
                entry = g["teams"][side].get("probablePitcher")
                if entry:
                    pitchers.append({"id": entry["id"], "name": entry["fullName"]})
    # dedupe
    return [{"id": k, "name": v} for k, v in {p["id"]: p["name"] for p in pitchers}.items()]


if st is not None:
    @st.cache_data(show_spinner=False)
    def fetch_pitcher_statcast(pid: int, start: dt.date, end: dt.date) -> pd.DataFrame:  # type: ignore
        try:
            return statcast_pitcher(start.isoformat(), end.isoformat(), pid)
        except Exception as exc:
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
    """Add helper boolean columns: swing, miss, barrel."""
    descriptions_miss = {
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
    df["is_miss"] = df["description"].isin(descriptions_miss)

    # Barrel according to simple rule of launch_speed ≥ 98 & 26° ≥ launch_angle ≥ 8°
    df["is_barrel"] = (
        (df["launch_speed"] >= 98)
        & (df["launch_angle"] >= 8)
        & (df["launch_angle"] <= 26)
    )
    return df


def aggregate_metrics(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return appearance‑level aggregated metrics & latest‑game deltas."""
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
    )
    agg["whiff%"] = (agg["misses"] / agg["swings"]).round(3)
    agg["barrel%"] = (agg["barrels"] / agg["pitches"]).round(3)

    # velocity deltas (keep for convenience)
    agg = agg.sort_values("game_date")
    agg["baseline_v"] = (
        agg.groupby("pitch_type")["avg_v"].transform(lambda s: s.shift(1).rolling(3, min_periods=3).mean())
    )
    agg["delta_v"] = agg["avg_v"] - agg["baseline_v"]

    latest_date = agg["game_date"].max()
    latest = agg[agg["game_date"] == latest_date].copy()
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

    # unified query‑param helpers (list→str)
    def qp_get(key: str, default: str | None = None):
        if hasattr(st, "query_params"):
            qp = st.query_params  # proxy
        else:
            qp = st.experimental_get_query_params()  # type: ignore[attr-defined]
        if key in qp:
            v = qp[key]
            return v[0] if isinstance(v, list) else v
        return default

    def qp_set(**kwargs):
        if hasattr(st, "query_params"):
            st.query_params.update(kwargs)  # type: ignore[attr-defined]
        else:
            st.experimental_set_query_params(**kwargs)  # type: ignore[attr-defined]

    # DETAIL VIEW -----------------------------------------------------------
    if qp_get("pid") is not None:
        pid = int(qp_get("pid"))
        name = qp_get("name", "Unknown")
        sel_date = dt.date.fromisoformat(qp_get("date", dt.date.today().isoformat()))
        lookback = int(qp_get("lookback", "35"))

        st.markdown(f"## {name} – Recent Trends")
        st.image(HEADSHOT_TEMPLATE.format(pid=pid, size=200), width=150)

        agg_df, latest_df, raw_df = summarize_pitcher(pid, name, sel_date, lookback)
        if agg_df is None:
            st.error("No Statcast data available.")
        else:
            # --- Line charts (velocity & spin)
            for metric, ylabel in [("avg_v", "Velocity (mph)"), ("avg_spin", "Spin Rate (RPM)")]:
                plt.figure(figsize=(9, 4))
                for pt, grp in agg_df.groupby("pitch_type"):
                    plt.plot(grp["game_date"], grp[metric], marker="o", label=pt)
                plt.xlabel("Game Date")
                plt.ylabel(ylabel)
                plt.title(f"{name} – {ylabel.split()[0]} by Pitch Type")
                plt.xticks(rotation=45)
                plt.legend()
                st.pyplot(plt.gcf())
                plt.clf()

            # --- Movement scatter (latest appearance)
            st.markdown("### Latest Game Movement vs Spin")
            if not latest_df.empty:
                plt.figure(figsize=(6, 5))
                plt.scatter(latest_df["avg_hmov"], latest_df["avg_vmov"], s=100)
                for _, row in latest_df.iterrows():
                    plt.text(row["avg_hmov"], row["avg_vmov"], row["pitch_type"])
                plt.xlabel("Horiz. Movement (pfx_x, ft)")
                plt.ylabel("Vert. Movement (pfx_z, ft)")
                plt.axhline(0, ls=":", lw=0.5)
                plt.axvline(0, ls=":", lw=0.5)
                st.pyplot(plt.gcf())
                plt.clf()

            # --- Latest metrics table
            st.markdown("### Latest Appearance Summary")
            latest_cols = [
                "pitch_type",
                "avg_v",
                "delta_v",
                "avg_spin",
                "whiff%",
                "barrel%",
            ]
            st.dataframe(latest_df[latest_cols].round(3), use_container_width=True)

            # Raw data
            with st.expander("Raw Statcast (recent)"):
                st.dataframe(raw_df, use_container_width=True)

        # Back button resets query params
        if st.button("← Back to list"):
            qp_set()  # clears all

    # MAIN DASHBOARD --------------------------------------------------------
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
                st.success(f"Found {len(probables)} pitchers.")
                for p in probables:
                    agg_df, latest_df, _ = summarize_pitcher(p["id"], p["name"], sel_date, lookback)
                    if latest_df is None or latest_df.empty:
                        st.info(f"No Statcast data for {p['name']}")
                        continue
                    delta_min = latest_df["delta_v"].min()
                    whiff_max = latest_df["whiff%"].max()
                    barrel_max = latest_df["barrel%"].max()
                    headshot = HEADSHOT_TEMPLATE.format(pid=p["id"], size=150)

                    card = st.container(border=True)
                    with card:
                        c1, c2 = st.columns([1, 3])
                        with c1:
                            st.image(headshot, width=120)
                        with c2:
                            st.subheader(p["name"])
                            st.markdown(
                                f"**Largest ΔV:** {delta_min:+.2f} mph   |  "
                                f"**Peak Whiff%:** {whiff_max:.0%}   |  "
                                f"**Peak Barrel%:** {barrel_max:.0%}"
                            )
                            params = {
                                "pid": p["id"],
                                "name": p["name"],
                                "date": sel_date.isoformat(),
                                "lookback": lookback,
                            }
                            url = f"?{urllib.parse.urlencode(params)}"
                            st.link_button("Details ➜", url)

# ---------------------------------------------------------------------------
# CLI fallback
# ---------------------------------------------------------------------------

def cli_main(date_iso: str | None = None):
    today = dt.date.fromisoformat(date_iso) if date_iso else dt.date.today()
    probables = get_probable_pitchers(today)
    if not probables:
        print("No probable starters.")
        return
    frames = []
    for p in probables:
        _, latest_df, _ = summarize_pitcher(p["id"], p["name"], today)
        if latest_df is not None:
            latest_df["pitcher"] = p["name"]
            frames.append(latest_df)
    if not frames:
        print("No recent Statcast data.")
        return
    report = pd.concat(frames)[
        [
            "pitcher",
            "pitch_type",
            "avg_v",
            "delta_v",
            "avg_spin",
            "whiff%",
            "barrel%",
        ]
    ].round(2)
    print(report.sort_values(["pitcher", "delta_v"]))


if __name__ == "__main__" and not in_streamlit():
    parser = argparse.ArgumentParser("Pitcher performance CLI")
    parser.add_argument("--date", help="YYYY-MM-DD (default today)")
    args = parser.parse_args()
    cli_main(args.date)
