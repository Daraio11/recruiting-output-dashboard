import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# -----------------------
# Config
# -----------------------
st.set_page_config(page_title="Recruiting Output Dashboard", layout="wide")
DATA_PATH = "data/PFF_Recruiting_Performance_Delta.csv"

# -----------------------
# Helpers
# -----------------------
def fail_if_missing_data():
    if not Path(DATA_PATH).exists():
        st.error(f"Data file not found: {DATA_PATH}")
        st.stop()

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)

    # Ensure numeric columns
    num_cols = [
        "stars", "industry_rating", "recruiting_pct",
        "performance_pct", "perf_minus_recruit", "nil_value_usd",
        "pass_grade", "pass_grade_pct",
        "rec_grade", "rec_grade_pct",
        "rush_grade", "rush_grade_pct",
        "def_grade", "def_grade_pct",
        "player_game_count"
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Clean categoricals
    for c in ["team_name", "position", "class_year", "player"]:
        if c in df.columns:
            df[c] = df[c].fillna("UNKNOWN").astype(str)

    return df

def apply_chat_query(d: pd.DataFrame, q: str) -> pd.DataFrame:
    """Rules-based parser to translate simple natural language into filters."""
    q = (q or "").lower().strip()
    if not q:
        return d

    # Position shortcuts
    pos_map = {
        "qb": "QB", "rb": "RB", "wr": "WR", "te": "TE",
        "ol": "OL", "dl": "DL", "lb": "LB", "db": "DB",
        "cb": "CB", "s": "S", "edge": "EDGE", "de": "DE", "dt": "DT"
    }
    if "position" in d.columns:
        for k, v in pos_map.items():
            if re.search(rf"\b{k}\b", q):
                d = d[d["position"].str.upper() == v]

    # Stars: "3 star", "4-star"
    m = re.search(r"(\d)\s*[- ]?star", q)
    if m and "stars" in d.columns:
        star = int(m.group(1))
        d = d[d["stars"] == star]

    # performance >= N
    m = re.search(r"(performance|perf)\s*(>=|>|at least)\s*(\d+)", q)
    if m and "performance_pct" in d.columns:
        val = float(m.group(3))
        d = d[d["performance_pct"] >= val]

    # recruiting >= N
    m = re.search(r"(recruiting|recruit)\s*(>=|>|at least)\s*(\d+)", q)
    if m and "recruiting_pct" in d.columns:
        val = float(m.group(3))
        d = d[d["recruiting_pct"] >= val]

    # delta > N or < N
    m = re.search(r"(delta)\s*(>=|>|<=|<)\s*(-?\d+)", q)
    if m and "perf_minus_recruit" in d.columns:
        op = m.group(2)
        val = float(m.group(3))
        if op in (">", ">="):
            d = d[d["perf_minus_recruit"] >= val]
        else:
            d = d[d["perf_minus_recruit"] <= val]

    # Team mention (simple contains)
    if "team_name" in d.columns:
        teams = d["team_name"].dropna().unique().tolist()
        for t in teams:
            tlow = str(t).lower()
            if tlow and tlow in q:
                d = d[d["team_name"].str.lower() == tlow]
                break

    return d

def bytes_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

def report_preset(df: pd.DataFrame, preset: str) -> pd.DataFrame:
    """Saved views."""
    x = df.copy()
    if preset == "Custom (filters)":
        return x

    if preset == "Late Bloomers (delta ≥ +20)":
        return x[x["perf_minus_recruit"] >= 20]

    if preset == "Blue-Chip Underperformers (recruit ≥ 80, delta ≤ -15)":
        return x[(x["recruiting_pct"] >= 80) & (x["perf_minus_recruit"] <= -15)]

    if preset == "Elite Performers (performance ≥ 90)":
        return x[x["performance_pct"] >= 90]

    if preset == "3-Star Values (3★, performance ≥ 85)":
        return x[(x["stars"] == 3) & (x["performance_pct"] >= 85)]

    if preset == "NIL Inefficiencies (performance ≥ 85, NIL bottom quartile)":
        if "nil_value_usd" not in x.columns:
            return x.iloc[0:0]
        nil = x["nil_value_usd"].fillna(0)
        cutoff = nil.quantile(0.25)
        return x[(x["performance_pct"] >= 85) & (nil <= cutoff)]

    return x

def apply_sidebar_filters(x: pd.DataFrame, position, team, class_year, stars_range, perf_min, rec_min, delta_range):
    if position != "ALL":
        x = x[x["position"] == position]
    if team != "ALL":
        x = x[x["team_name"] == team]
    if class_year != "ALL":
        x = x[x["class_year"] == class_year]

    if "stars" in x.columns and stars_range is not None:
        x = x[(x["stars"].fillna(-1) >= stars_range[0]) & (x["stars"].fillna(-1) <= stars_range[1])]

    x = x[x["performance_pct"].fillna(-1) >= perf_min]
    x = x[x["recruiting_pct"].fillna(-1) >= rec_min]
    x = x[(x["perf_minus_recruit"].fillna(-999) >= delta_range[0]) & (x["perf_minus_recruit"].fillna(999) <= delta_range[1])]

    return x

def choose_columns(view: pd.DataFrame, mode: str) -> list[str]:
    base = [
        "player", "position", "team_name", "class_year",
        "stars", "industry_rating", "recruiting_pct",
        "performance_pct", "perf_minus_recruit",
        "nil_value_usd", "profile_url"
    ]

    if mode == "QB":
        base += ["pass_grade", "pass_grade_pct", "pass_attempts", "accuracy_percent", "avg_depth_of_target",
                 "avg_time_to_throw", "big_time_throws", "turnover_worthy_plays"]
    elif mode == "SKILL":
        base += ["rec_grade", "rec_grade_pct", "targets", "receptions", "yards", "yards_per_reception", "drops",
                 "rush_grade", "rush_grade_pct", "rush_attempts", "yards_after_contact"]
    elif mode == "DEF":
        base += ["def_grade", "def_grade_pct", "coverage_grade", "pass_rush_grade", "run_defense_grade", "tackle_grade",
                 "tackles", "stops", "pressures", "sacks"]

    # keep only existing
    return [c for c in base if c in view.columns]

def team_roi_table(df: pd.DataFrame) -> pd.DataFrame:
    """Team-level ROI: how much performance exceeds recruiting."""
    x = df.copy()
    grp = x.groupby("team_name", dropna=False).agg(
        players=("player", "count"),
        avg_recruit=("recruiting_pct", "mean"),
        avg_perf=("performance_pct", "mean"),
        avg_delta=("perf_minus_recruit", "mean"),
        med_delta=("perf_minus_recruit", "median"),
        pct_elite=("performance_pct", lambda s: float((s >= 90).mean() * 100)),
        pct_late_bloom=("perf_minus_recruit", lambda s: float((s >= 20).mean() * 100)),
    ).reset_index()

    # Require some sample size so we don't overreact
    grp = grp.sort_values(["avg_delta", "players"], ascending=[False, False])
    return grp

# -----------------------
# App start
# -----------------------
fail_if_missing_data()
df = load_data()

st.title("Recruiting Output Dashboard")
st.caption("PFF performance + recruiting baseline (industry_rating) + delta metrics (HIGH_CONF matches)")

# -----------------------
# Sidebar global controls
# -----------------------
with st.sidebar:
    st.header("Saved Views")
    preset = st.selectbox(
        "Choose a report",
        [
            "Custom (filters)",
            "Late Bloomers (delta ≥ +20)",
            "Blue-Chip Underperformers (recruit ≥ 80, delta ≤ -15)",
            "Elite Performers (performance ≥ 90)",
            "3-Star Values (3★, performance ≥ 85)",
            "NIL Inefficiencies (performance ≥ 85, NIL bottom quartile)"
        ],
        index=0
    )

    st.divider()
    st.header("Filters")

    pos_options = ["ALL"] + sorted(df["position"].unique().tolist())
    team_options = ["ALL"] + sorted(df["team_name"].unique().tolist())
    year_options = ["ALL"] + sorted(df["class_year"].unique().tolist())

    position = st.selectbox("Position", pos_options, index=0)
    team = st.selectbox("Team", team_options, index=0)
    class_year = st.selectbox("Class Year", year_options, index=0)

    stars_range = st.slider("Stars", 0, 5, (0, 5)) if "stars" in df.columns else None
    perf_min = st.slider("Min Performance %", 0, 100, 0)
    rec_min = st.slider("Min Recruiting %", 0, 100, 0)
    delta_range = st.slider("Delta (Perf - Recruit)", -100, 100, (-100, 100))

    st.divider()
    st.header("Ask (optional)")
    chat = st.text_input('Examples: "3 star QBs perf >= 90" | "WR delta > 25" | "Alabama DB delta > 15"')

# Apply preset + filters + chat
view = report_preset(df, preset)
view = apply_sidebar_filters(view, position, team, class_year, stars_range, perf_min, rec_min, delta_range)
view = apply_chat_query(view, chat)

# Default sort
if "perf_minus_recruit" in view.columns:
    view = view.sort_values("perf_minus_recruit", ascending=False)

# -----------------------
# Tabs (multi-page feel)
# -----------------------
tab_overview, tab_qb, tab_skill, tab_def, tab_team = st.tabs(
    ["Overview", "QB", "Skill (RB/WR/TE)", "Defense", "Team ROI"]
)

def render_table(title: str, data: pd.DataFrame, cols: list[str], download_name: str):
    st.subheader(title)
    st.write(f"Rows: {len(data):,}")

    c1, c2 = st.columns([1, 2])
    with c1:
        st.download_button(
            label="⬇️ Download filtered CSV",
            data=bytes_csv(data[cols] if cols else data),
            file_name=download_name,
            mime="text/csv",
            use_container_width=True
        )
    with c2:
        st.caption("Download exports exactly what you're currently viewing (after preset + filters + chat).")

    if cols:
        st.dataframe(data[cols], use_container_width=True, hide_index=True)
    else:
        st.dataframe(data, use_container_width=True, hide_index=True)

with tab_overview:
    cols = choose_columns(view, mode="BASE")
    render_table(
        title=f"Results — {preset}",
        data=view,
        cols=cols,
        download_name="RecruitingOutput_Filtered.csv"
    )

with tab_qb:
    qb = view[view["position"].str.upper() == "QB"].copy()
    cols = choose_columns(qb, mode="QB")
    render_table(
        title="QB Board",
        data=qb,
        cols=cols,
        download_name="RecruitingOutput_QB_Filtered.csv"
    )

with tab_skill:
    skill_positions = {"RB", "WR", "TE"}
    skill = view[view["position"].str.upper().isin(skill_positions)].copy()
    cols = choose_columns(skill, mode="SKILL")
    render_table(
        title="Skill Board (RB/WR/TE)",
        data=skill,
        cols=cols,
        download_name="RecruitingOutput_Skill_Filtered.csv"
    )

with tab_def:
    # Defense tab: anyone with a def_grade_pct (or common defensive positions)
    if "def_grade_pct" in view.columns:
        defense = view[view["def_grade_pct"].notna()].copy()
    else:
        defense = view.iloc[0:0].copy()

    cols = choose_columns(defense, mode="DEF")
    render_table(
        title="Defense Board",
        data=defense,
        cols=cols,
        download_name="RecruitingOutput_Defense_Filtered.csv"
    )

with tab_team:
    st.subheader("Team Development / ROI")
    st.caption("Team-level view: how much on-field performance exceeds recruiting expectation (based on matched players).")

    roi = team_roi_table(view)

    # Optional sample size filter for credibility
    min_players = st.slider("Min matched players per team", 1, 50, 10)
    roi2 = roi[roi["players"] >= min_players].copy()

    st.write(f"Teams shown: {len(roi2):,}")
    st.download_button(
        label="⬇️ Download Team ROI CSV",
        data=bytes_csv(roi2),
        file_name="RecruitingOutput_Team_ROI.csv",
        mime="text/csv"
    )

    st.dataframe(roi2, use_container_width=True, hide_index=True))
