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
def fail_if_missing_data() -> None:
    if not Path(DATA_PATH).exists():
        st.error(f"Data file not found: {DATA_PATH}")
        st.stop()


def bytes_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)

    def to_num(series: pd.Series) -> pd.Series:
        s = series.copy().astype("string")

        s = s.replace(
            {
                "nan": pd.NA,
                "None": pd.NA,
                "N/A": pd.NA,
                "NA": pd.NA,
                "—": pd.NA,
                "–": pd.NA,
                "": pd.NA,
            }
        )

        s = (
            s.str.replace("%", "", regex=False)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
        )

        return pd.to_numeric(s, errors="coerce")

    # Robust numeric conversion if present
    num_cols = [
        "stars",
        "industry_rating",
        "recruiting_pct",
        "performance_pct",
        "perf_minus_recruit",
        "nil_value_usd",
        "pass_grade",
        "pass_grade_pct",
        "rec_grade",
        "rec_grade_pct",
        "rush_grade",
        "rush_grade_pct",
        "def_grade",
        "def_grade_pct",
        "player_game_count",
        "pass_attempts",
        "accuracy_percent",
        "avg_depth_of_target",
        "avg_time_to_throw",
        "big_time_throws",
        "turnover_worthy_plays",
        "targets",
        "receptions",
        "yards",
        "yards_per_reception",
        "drops",
        "rush_attempts",
        "yards_after_contact",
        "tackles",
        "stops",
        "pressures",
        "sacks",
        "coverage_grade",
        "pass_rush_grade",
        "run_defense_grade",
        "tackle_grade",
        "interceptions",
        "pass_break_ups",
        "hits",
        "hurries",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = to_num(df[c])

    # Clean categoricals
    for c in ["team_name", "position", "class_year", "player", "profile_url"]:
        if c in df.columns:
            df[c] = df[c].fillna("UNKNOWN").astype(str).str.strip()

    return df


def report_preset(df: pd.DataFrame, preset: str) -> pd.DataFrame:
    x = df.copy()

    if preset == "All Data (no preset)":
        return x

    if preset == "Late Bloomers (delta ≥ +20)":
        if "perf_minus_recruit" in x.columns:
            return x[x["perf_minus_recruit"] >= 20]
        return x.iloc[0:0]

    if preset == "Blue-Chip Underperformers (recruit ≥ 80, delta ≤ -15)":
        if {"recruiting_pct", "perf_minus_recruit"}.issubset(x.columns):
            return x[(x["recruiting_pct"] >= 80) & (x["perf_minus_recruit"] <= -15)]
        return x.iloc[0:0]

    if preset == "Elite Performers (performance ≥ 90)":
        if "performance_pct" in x.columns:
            return x[x["performance_pct"] >= 90]
        return x.iloc[0:0]

    if preset == "3-Star Values (3★, performance ≥ 85)":
        if {"stars", "performance_pct"}.issubset(x.columns):
            return x[(x["stars"] == 3) & (x["performance_pct"] >= 85)]
        return x.iloc[0:0]

    if preset == "NIL Inefficiencies (performance ≥ 85, NIL bottom quartile)":
        if {"performance_pct", "nil_value_usd"}.issubset(x.columns):
            cutoff = x["nil_value_usd"].fillna(0).quantile(0.25)
            return x[(x["performance_pct"] >= 85) & (x["nil_value_usd"].fillna(0) <= cutoff)]
        return x.iloc[0:0]

    return x


def apply_sidebar_filters(
    x: pd.DataFrame,
    position: str,
    team: str,
    class_year: str,
    stars_range,
    perf_min: int,
    rec_min: int,
    delta_range,
) -> pd.DataFrame:
    if position != "ALL" and "position" in x.columns:
        x = x[x["position"] == position]
    if team != "ALL" and "team_name" in x.columns:
        x = x[x["team_name"] == team]
    if class_year != "ALL" and "class_year" in x.columns:
        x = x[x["class_year"] == class_year]

    if "stars" in x.columns and stars_range is not None and x["stars"].notna().any():
        x = x[(x["stars"] >= stars_range[0]) & (x["stars"] <= stars_range[1])]

    if "performance_pct" in x.columns and x["performance_pct"].notna().any():
        x = x[x["performance_pct"] >= perf_min]

    if "recruiting_pct" in x.columns and x["recruiting_pct"].notna().any():
        x = x[x["recruiting_pct"] >= rec_min]

    if "perf_minus_recruit" in x.columns and x["perf_minus_recruit"].notna().any():
        x = x[(x["perf_minus_recruit"] >= delta_range[0]) & (x["perf_minus_recruit"] <= delta_range[1])]

    return x


def apply_chat_query(d: pd.DataFrame, q: str) -> pd.DataFrame:
    q = (q or "").lower().strip()
    if not q:
        return d

    pos_map = {
        "qb": "QB",
        "rb": "RB",
        "wr": "WR",
        "te": "TE",
        "ol": "OL",
        "dl": "DL",
        "lb": "LB",
        "db": "DB",
        "cb": "CB",
        "s": "S",
        "edge": "EDGE",
        "de": "DE",
        "dt": "DT",
    }
    if "position" in d.columns:
        for k, v in pos_map.items():
            if re.search(rf"\b{k}\b", q):
                d = d[d["position"].astype(str).str.upper() == v]

    m = re.search(r"(\d)\s*[- ]?star", q)
    if m and "stars" in d.columns:
        d = d[d["stars"] == int(m.group(1))]

    m = re.search(r"(performance|perf)\s*(>=|>|at least)\s*(\d+)", q)
    if m and "performance_pct" in d.columns:
        d = d[d["performance_pct"] >= float(m.group(3))]

    m = re.search(r"(recruiting|recruit)\s*(>=|>|at least)\s*(\d+)", q)
    if m and "recruiting_pct" in d.columns:
        d = d[d["recruiting_pct"] >= float(m.group(3))]

    m = re.search(r"(delta)\s*(>=|>|<=|<)\s*(-?\d+)", q)
    if m and "perf_minus_recruit" in d.columns:
        op = m.group(2)
        val = float(m.group(3))
        if op in (">", ">="):
            d = d[d["perf_minus_recruit"] >= val]
        else:
            d = d[d["perf_minus_recruit"] <= val]

    if "team_name" in d.columns:
        teams = d["team_name"].dropna().astype(str).unique().tolist()
        for t in teams:
            tlow = t.lower()
            if tlow and tlow in q:
                d = d[d["team_name"].astype(str).str.lower() == tlow]
                break

    return d


# Friendly display names (only for UI)
FRIENDLY = {
    "player": "Player",
    "position": "Position",
    "team_name": "Team",
    "class_year": "Class Year",
    "player_game_count": "Games",
    "stars": "Stars",
    "industry_rating": "Industry Rating",
    "recruiting_pct": "Recruiting %ile",
    "performance_pct": "Performance %ile",
    "perf_minus_recruit": "Delta (Perf - Recruit)",
    "nil_value_usd": "NIL ($)",
    "pass_grade": "Pass Grade",
    "pass_grade_pct": "Pass %ile",
    "rec_grade": "Receiving Grade",
    "rec_grade_pct": "Receiving %ile",
    "rush_grade": "Rushing Grade",
    "rush_grade_pct": "Rushing %ile",
    "def_grade": "Defense Grade",
    "def_grade_pct": "Defense %ile",
    "coverage_grade": "Coverage Grade",
    "pass_rush_grade": "Pass Rush Grade",
    "run_defense_grade": "Run Defense Grade",
    "tackle_grade": "Tackle Grade",
    "pass_attempts": "Pass Att",
    "accuracy_percent": "Accuracy %",
    "avg_depth_of_target": "aDOT",
    "avg_time_to_throw": "Time to Throw",
    "big_time_throws": "Big Time Throws",
    "turnover_worthy_plays": "TWP",
    "targets": "Targets",
    "receptions": "Receptions",
    "yards": "Yards",
    "yards_per_reception": "Yds/Rec",
    "drops": "Drops",
    "rush_attempts": "Rush Att",
    "yards_after_contact": "YAC (Contact)",
    "tackles": "Tackles",
    "stops": "Stops",
    "pressures": "Pressures",
    "sacks": "Sacks",
    "interceptions": "INT",
    "pass_break_ups": "PBU",
    "hits": "Hits",
    "hurries": "Hurries",
    "profile_url": "Profile URL",
}


def prettify_for_display(df_in: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, dict]:
    """
    Returns:
      - display_df: subset with renamed columns for UI
      - column_config: streamlit column config (for clickable links, etc.)
    """
    df_out = df_in.copy()

    # Keep only cols (if provided)
    if cols:
        keep = [c for c in cols if c in df_out.columns]
        df_out = df_out[keep]

    # Rename columns for display
    rename_map = {c: FRIENDLY.get(c, c.replace("_", " ").title()) for c in df_out.columns}
    df_out = df_out.rename(columns=rename_map)

    # Make Profile URL clickable (if present)
    col_config = {}
    profile_label = FRIENDLY.get("profile_url", "Profile URL")
    if profile_label in df_out.columns:
        col_config[profile_label] = st.column_config.LinkColumn(
            label=profile_label,
            display_text="Open",
        )

    return df_out, col_config


def choose_columns(view: pd.DataFrame, mode: str) -> list[str]:
    # IMPORTANT: profile_url is placed at END by design.
    base = [
        "player",
        "position",
        "team_name",
        "class_year",
        "player_game_count",
        "stars",
        "industry_rating",
        "recruiting_pct",
        "performance_pct",
        "perf_minus_recruit",
        "nil_value_usd",
    ]

    if mode == "QB":
        base += [
            "pass_grade",
            "pass_grade_pct",
            "pass_attempts",
            "accuracy_percent",
            "avg_depth_of_target",
            "avg_time_to_throw",
            "big_time_throws",
            "turnover_worthy_plays",
        ]
    elif mode == "SKILL":
        base += [
            "rec_grade",
            "rec_grade_pct",
            "targets",
            "receptions",
            "yards",
            "yards_per_reception",
            "drops",
            "rush_grade",
            "rush_grade_pct",
            "rush_attempts",
            "yards_after_contact",
        ]
    elif mode == "DEF":
        base += [
            "def_grade",
            "def_grade_pct",
            "coverage_grade",
            "pass_rush_grade",
            "run_defense_grade",
            "tackle_grade",
            "tackles",
            "stops",
            "pressures",
            "sacks",
            "interceptions",
            "pass_break_ups",
            "hits",
            "hurries",
        ]

    # profile_url always last
    base += ["profile_url"]

    # keep only existing
    return [c for c in base if c in view.columns]


def team_roi_table(df_in: pd.DataFrame) -> pd.DataFrame:
    if "team_name" not in df_in.columns:
        return pd.DataFrame()

    x = df_in.copy()

    if not {"recruiting_pct", "performance_pct", "perf_minus_recruit"}.issubset(x.columns):
        return x.groupby("team_name", dropna=False).agg(players=("player", "count")).reset_index()

    grp = x.groupby("team_name", dropna=False).agg(
        players=("player", "count"),
        avg_recruit=("recruiting_pct", "mean"),
        avg_perf=("performance_pct", "mean"),
        avg_delta=("perf_minus_recruit", "mean"),
        med_delta=("perf_minus_recruit", "median"),
        pct_elite=("performance_pct", lambda s: float((s >= 90).mean() * 100)),
        pct_late_bloom=("perf_minus_recruit", lambda s: float((s >= 20).mean() * 100)),
    ).reset_index()

    return grp.sort_values(["avg_delta", "players"], ascending=[False, False])


def render_table(title: str, data: pd.DataFrame, cols: list[str], download_name: str) -> None:
    st.subheader(title)
    st.write(f"Rows: {len(data):,}")

    # Display dataframe (pretty headers + clickable links)
    display_df, col_config = prettify_for_display(data, cols)

    c1, c2 = st.columns([1, 2])
    with c1:
        st.download_button(
            label="⬇️ Download filtered CSV",
            data=bytes_csv(display_df),
            file_name=download_name,
            mime="text/csv",
            use_container_width=True,
        )
    with c2:
        st.caption("Exports exactly what you're viewing (after preset + filters + chat).")

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config=col_config,
    )


def render_about_tab() -> None:
    st.title("About")
    st.caption("What these columns mean, how they were derived, and how to interpret the outputs.")

    st.markdown(
        """
### Data Sources
- **Recruiting (On3 / Rivals export):** provides recruit baseline attributes like `industry_rating`, stars, and class year.
- **PFF:** provides on-field performance grades across passing/receiving/rushing/defense.

### Industry Rating
**`industry_rating`** is a *recruiting* baseline score from the On3/Rivals “industry” feed.  
Think: **pre-college expectation** for a player. Higher = more highly regarded prospect.

### Recruiting Percentile
**`recruiting_pct`** is the player’s percentile rank (0–100) within the recruiting dataset based on `industry_rating`.  
- 90 = top 10% recruit  
- 50 = average recruit  
- 20 = bottom 20%

### Performance Percentile
**`performance_pct`** is the player’s percentile rank (0–100) derived from PFF performance measures included in this dataset.
It represents **how the player performed on-field relative to peers** in the current dataset.

### Delta (Performance minus Recruiting)
**`perf_minus_recruit`** is the main “development signal”:

`perf_minus_recruit = performance_pct − recruiting_pct`

- Positive = player is outperforming recruiting expectation (late bloomer / development win)
- Negative = underperforming expectation

### Team ROI (Return on Investment)
Team ROI aggregates player deltas at the team level:
- **Avg Delta** = average `perf_minus_recruit` across matched players on the team  
Higher Avg Delta implies the program is **getting more on-field performance than recruiting profiles predicted**.

### Notes / Caveats
- This dashboard relies on **matched player records** across sources (PFF × recruiting). Not every player will match.
- Percentiles are computed within the current dataset scope (as exported), so reruns with different data can shift percentile values.
"""
    )


# -----------------------
# App start
# -----------------------
fail_if_missing_data()

# Clear cache via a button (safer than always clearing)
with st.sidebar:
    if st.button("Reload data"):
        st.cache_data.clear()

df = load_data()

st.title("Recruiting Output Dashboard")
st.caption("PFF performance + recruiting baseline (industry_rating) + delta metrics (HIGH_CONF matches)")

# -----------------------
# Sidebar controls
# -----------------------
with st.sidebar:
    st.header("Saved Views")
    preset = st.selectbox(
        "Choose a report",
        [
            "All Data (no preset)",
            "Late Bloomers (delta ≥ +20)",
            "Blue-Chip Underperformers (recruit ≥ 80, delta ≤ -15)",
            "Elite Performers (performance ≥ 90)",
            "3-Star Values (3★, performance ≥ 85)",
            "NIL Inefficiencies (performance ≥ 85, NIL bottom quartile)",
        ],
        index=0,
    )

    st.divider()
    st.header("Filters")

    pos_options = ["ALL"] + sorted(df["position"].unique().tolist()) if "position" in df.columns else ["ALL"]
    team_options = ["ALL"] + sorted(df["team_name"].unique().tolist()) if "team_name" in df.columns else ["ALL"]
    year_options = ["ALL"] + sorted(df["class_year"].unique().tolist()) if "class_year" in df.columns else ["ALL"]

    position = st.selectbox("Position", pos_options, index=0)
    team = st.selectbox("Team", team_options, index=0)
    class_year = st.selectbox("Class Year", year_options, index=0)

    stars_range = st.slider("Stars", 0, 5, (0, 5)) if "stars" in df.columns else None
    perf_min = st.slider("Min Performance %ile", 0, 100, 0)
    rec_min = st.slider("Min Recruiting %ile", 0, 100, 0)
    delta_range = st.slider("Delta (Perf - Recruit)", -100, 100, (-100, 100))

    st.divider()
    st.header("Ask (optional)")
    chat = st.text_input('Examples: "3 star QBs perf >= 90" | "WR delta > 25" | "Alabama DB delta > 15"')

# Apply preset + filters + chat
view = report_preset(df, preset)
view = apply_sidebar_filters(view, position, team, class_year, stars_range, perf_min, rec_min, delta_range)
view = apply_chat_query(view, chat)

# Default sort
if "perf_minus_recruit" in view.columns and view["perf_minus_recruit"].notna().any():
    view = view.sort_values("perf_minus_recruit", ascending=False)

# -----------------------
# Tabs
# -----------------------
tab_about, tab_overview, tab_qb, tab_skill, tab_def, tab_team = st.tabs(
    ["About", "All Data", "QB", "Skill (RB/WR/TE)", "Defense", "Team ROI"]
)

with tab_about:
    render_about_tab()

with tab_overview:
    cols = choose_columns(view, mode="BASE")
    render_table("Results", view, cols, "RecruitingOutput_Filtered.csv")

with tab_qb:
    qb = view[view["position"].astype(str).str.upper() == "QB"].copy() if "position" in view.columns else view.iloc[0:0]
    cols = choose_columns(qb, mode="QB")
    render_table("QB Board", qb, cols, "RecruitingOutput_QB_Filtered.csv")

with tab_skill:
    skill_positions = {"RB", "WR", "TE"}
    if "position" in view.columns:
        skill = view[view["position"].astype(str).str.upper().isin(skill_positions)].copy()
    else:
        skill = view.iloc[0:0].copy()
    cols = choose_columns(skill, mode="SKILL")
    render_table("Skill Board (RB/WR/TE)", skill, cols, "RecruitingOutput_Skill_Filtered.csv")

with tab_def:
    if "def_grade_pct" in view.columns and view["def_grade_pct"].notna().any():
        defense = view[view["def_grade_pct"].notna()].copy()
    else:
        defense = view.iloc[0:0].copy()
    cols = choose_columns(defense, mode="DEF")
    render_table("Defense Board", defense, cols, "RecruitingOutput_Defense_Filtered.csv")

with tab_team:
    st.subheader("Team Development / ROI")
    st.caption("Team-level view: performance vs recruiting expectation (based on matched players).")

    roi = team_roi_table(view)
    if roi.empty:
        st.info("Team ROI unavailable (missing columns).")
    else:
        min_players = st.slider("Min matched players per team", 1, 50, 10)
        roi2 = roi[roi["players"] >= min_players].copy() if "players" in roi.columns else roi.copy()

        display_roi = roi2.rename(
            columns={
                "team_name": "Team",
                "players": "Players",
                "avg_recruit": "Avg Recruiting %ile",
                "avg_perf": "Avg Performance %ile",
                "avg_delta": "Avg Delta",
                "med_delta": "Median Delta",
                "pct_elite": "% Elite (≥90 %ile)",
                "pct_late_bloom": "% Late Bloomers (≥+20)",
            }
        )

        st.write(f"Teams shown: {len(display_roi):,}")
        st.download_button(
            label="⬇️ Download Team ROI CSV",
            data=bytes_csv(display_roi),
            file_name="RecruitingOutput_Team_ROI.csv",
            mime="text/csv",
        )
        st.dataframe(display_roi, use_container_width=True, hide_index=True)
