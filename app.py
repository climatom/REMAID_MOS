"""
REMAID GFS point-forecast viewer (Tindouf region).

Data sources, in order of precedence:
  1. REMAID_DATA_DIR environment variable -> local <data_dir>/points/... files
     (e.g. running on JASMIN:  REMAID_DATA_DIR=/gws/.../REMAID/Data/GFS streamlit run app.py)
  2. Google Drive file IDs below (Streamlit Community Cloud; files synced by rclone)

Run locally:
  streamlit run app.py
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import altair as alt
import pandas as pd
import requests
import streamlit as st

# ============================================================
# Data sources
# ============================================================
# Replace after the first rclone sync (right-click file in Drive -> Share -> copy link;
# the ID is the long token in the URL). The files must be shared "anyone with the link".
GDRIVE_FILE_IDS = {
    "latest": "1gQd596liUxcUOTWdAsg3chH5V5ixvpqn",           # gfs0p25_points_latest.csv
    "recent_runs": "1GvbHlWAiQ4rICNJ-0Tgq_846ve-FPann",  # recent_runs.csv
}
GDRIVE_BASE = "https://drive.google.com/uc?export=download&id="

LOCAL_FILES = {
    "latest": Path("points/latest/gfs0p25_points_latest.csv"),
    "recent_runs": Path("points/postprocessed/recent_runs.csv"),
}

LOCAL_TZ = "Africa/Algiers"  # UTC+1, no DST

# Reference categorical palette (validated slots, light mode) - fixed order
C_BLUE, C_ORANGE, C_AQUA = "#2a78d6", "#eb6834", "#1baf7a"
# Sequential blue ramp for run age (oldest light -> newest dark, ordinal from step 250)
RUN_RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#104281"]

VARIABLES = {
    # label: (column, unit)
    "2 m temperature": ("t2m_C", "°C"),
    "2 m dewpoint": ("td2m_C", "°C"),
    "2 m wet-bulb temperature": ("tw2m_C", "°C"),
    "10 m wind speed": ("wspd10_ms", "m/s"),
    "Wind gust": ("gust_ms", "m/s"),
    "Precipitation (interval)": ("precip_mm", "mm"),
    "Accumulated precipitation": ("precip_acc_mm", "mm"),
}

st.set_page_config(page_title="REMAID GFS forecast", layout="wide")


# ============================================================
# Loading
# ============================================================

@st.cache_data(ttl=900, show_spinner="Loading forecast data…")
def load(kind: str) -> pd.DataFrame:
    local_dir = os.environ.get("REMAID_DATA_DIR")
    if local_dir:
        src = Path(local_dir) / LOCAL_FILES[kind]
        df = pd.read_csv(src)
    else:
        file_id = GDRIVE_FILE_IDS[kind]
        if file_id.startswith("TODO_"):
            raise RuntimeError(
                f"No data source for '{kind}': set REMAID_DATA_DIR or fill GDRIVE_FILE_IDS in app.py"
            )
        r = requests.get(GDRIVE_BASE + file_id, timeout=60)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))

    for c in ["init_time_utc", "valid_time_utc"]:
        df[c] = pd.to_datetime(df[c], utc=True)
    return df


def add_display_time(df: pd.DataFrame, tz: str) -> pd.DataFrame:
    df = df.copy()
    t = df["valid_time_utc"] if tz == "UTC" else df["valid_time_utc"].dt.tz_convert(LOCAL_TZ)
    df["time"] = t.dt.tz_localize(None)  # Altair renders naive times as given
    return df


# ============================================================
# Chart helpers
# ============================================================

HEIGHT = 320  # default panel height (px); overridden by the sidebar slider
X_AXIS = alt.Axis(format="%a %d %b", tickCount={"interval": "day", "step": 1}, labelAngle=0)


def crosshair_layer(base: alt.Chart, df: pd.DataFrame, tooltip: list) -> alt.LayerChart:
    """Invisible nearest-x selector + vertical rule + tooltip (hover layer)."""
    hover = alt.selection_point(fields=["time"], nearest=True, on="pointerover", empty=False)
    selectors = base.mark_rule(opacity=0, strokeWidth=12).encode(
        x="time:T", tooltip=tooltip
    ).add_params(hover)
    rule = base.mark_rule(color="#8a8984", strokeWidth=1).encode(x="time:T").transform_filter(hover)
    return alt.layer(selectors, rule)


def line_panel(df: pd.DataFrame, series: dict[str, tuple[str, str]], title: str, unit: str,
               x_title: str, zero: bool = False) -> alt.LayerChart:
    """Line chart of 1-3 series (label -> (column, colour)) with legend + hover."""
    cols = {lab: col for lab, (col, _) in series.items()}
    long = df.melt(id_vars=["time"], value_vars=list(cols.values()), var_name="col", value_name="value")
    long["series"] = long["col"].map({v: k for k, v in cols.items()})

    color = alt.Color(
        "series:N",
        scale=alt.Scale(domain=list(series), range=[c for _, c in series.values()]),
        legend=alt.Legend(orient="top", title=None) if len(series) > 1 else None,
    )
    base = alt.Chart(long).encode(x=alt.X("time:T", title=x_title, axis=X_AXIS))
    lines = base.mark_line(strokeWidth=2, interpolate="monotone").encode(
        y=alt.Y("value:Q", title=unit, scale=alt.Scale(zero=zero)), color=color
    )

    wide = df[["time"] + list(cols.values())]
    tooltip = [alt.Tooltip("time:T", title="Time", format="%a %d %b %H:%M")] + [
        alt.Tooltip(f"{col}:Q", title=lab, format=".1f") for lab, col in cols.items()
    ]
    hover = crosshair_layer(alt.Chart(wide), wide, tooltip)
    return alt.layer(lines, hover).properties(title=title, height=HEIGHT)


def precip_panel(df: pd.DataFrame, x_title: str) -> alt.LayerChart:
    d = df[df["fxx"] > 0][["time", "precip_mm", "precip_period_h", "conv_precip_mm"]].copy()
    d["start"] = d["time"] - pd.to_timedelta(d["precip_period_h"], unit="h")
    base = alt.Chart(d)
    bars = base.mark_rect(color=C_BLUE, cornerRadiusTopLeft=2, cornerRadiusTopRight=2).encode(
        x=alt.X("start:T", title=x_title, axis=X_AXIS), x2="time:T",
        y=alt.Y("precip_mm:Q", title="mm per interval", scale=alt.Scale(domainMin=0, nice=True)),
        tooltip=[
            alt.Tooltip("start:T", title="From", format="%a %d %b %H:%M"),
            alt.Tooltip("time:T", title="To", format="%a %d %b %H:%M"),
            alt.Tooltip("precip_mm:Q", title="Total (mm)", format=".2f"),
            alt.Tooltip("conv_precip_mm:Q", title="Convective (mm)", format=".2f"),
        ],
    )
    total = df["precip_mm"].sum(skipna=True)
    return bars.properties(
        title=f"Precipitation — 1 h bars to +120 h, 3 h after (total {total:.1f} mm)", height=HEIGHT
    )


def runs_panel(runs: pd.DataFrame, col: str, label: str, unit: str, x_title: str) -> alt.LayerChart:
    inits = sorted(runs["init_time_utc"].unique())
    runs = runs.copy()
    runs["run"] = runs["init_time_utc"].dt.strftime("%d %b %HZ")
    domain = [pd.Timestamp(i).strftime("%d %b %HZ") for i in inits]
    ramp = RUN_RAMP[-len(domain):]

    base = alt.Chart(runs).encode(x=alt.X("time:T", title=x_title, axis=X_AXIS))
    lines = base.mark_line(strokeWidth=2, interpolate="monotone").encode(
        y=alt.Y(f"{col}:Q", title=unit, scale=alt.Scale(zero=False)),
        color=alt.Color("run:N", scale=alt.Scale(domain=domain, range=ramp),
                        legend=alt.Legend(orient="top", title="GFS run (darker = newer)")),
        strokeWidth=alt.condition(alt.datum.run == domain[-1], alt.value(3), alt.value(1.5)),
        tooltip=[
            alt.Tooltip("run:N", title="Run"),
            alt.Tooltip("time:T", title="Valid", format="%a %d %b %H:%M"),
            alt.Tooltip("fxx:Q", title="Lead (h)"),
            alt.Tooltip(f"{col}:Q", title=label, format=".1f"),
        ],
    )
    return lines.properties(title=f"{label}: last {len(domain)} runs", height=int(HEIGHT * 1.4))


# ============================================================
# Page
# ============================================================

st.title("REMAID — GFS (0.25°) forecast")

HEIGHT = st.sidebar.slider("Chart height (px)", 200, 600, HEIGHT, step=20)

try:
    latest = load("latest")
except Exception as e:
    st.error(f"Could not load forecast data: {e}")
    st.stop()

sites = latest.drop_duplicates("site").set_index("site")
c1, c2, c3 = st.columns([2, 1, 1])
site = c1.selectbox("Site", sites.index, format_func=lambda s: sites.loc[s, "site_name"])
interp = c2.radio("Grid → point", ["bilinear", "nearest"], horizontal=True,
                  help="Bilinear: interpolated to the site. Nearest: closest 0.25° grid point.")
tz = c3.radio("Time zone", ["UTC", "Local (UTC+1)"], horizontal=True)
x_title = "Time (UTC)" if tz == "UTC" else "Local time (UTC+1)"

df = latest[(latest["site"] == site) & (latest["interp"] == interp)].sort_values("fxx")
df = add_display_time(df, tz)
meta = df.iloc[0]
init = meta["init_time_utc"]

st.caption(
    f"GFS run **{init:%Y-%m-%d %H}Z** · site {meta['site_lat']:.3f}°N, {meta['site_lon']:.3f}°E, "
    f"{meta['site_elev_m']:.0f} m · model orography {meta['model_orog_m']:.0f} m "
    f"(grid point {meta['grid_lat']:.2f}, {meta['grid_lon']:.2f}) · "
    "raw model output, no MOS correction yet"
)

# ---- Next-24 h headline tiles
next24 = df[(df["fxx"] >= 1) & (df["fxx"] <= 24)]
k = st.columns(4)
k[0].metric("Max temp, next 24 h", f"{next24['t2m_C'].max():.1f} °C")
k[1].metric("Min temp, next 24 h", f"{next24['t2m_C'].min():.1f} °C")
k[2].metric("Max gust, next 24 h", f"{next24['gust_ms'].max():.1f} m/s")
k[3].metric("Precip, next 24 h", f"{next24['precip_mm'].sum():.1f} mm")

tab_fc, tab_runs, tab_table = st.tabs(["Forecast", "Run-to-run comparison", "Table"])

with tab_fc:
    max_h = int(df["fxx"].max())
    horizon = st.slider("Forecast horizon (hours)", 24, max_h, min(120, max_h), step=24)
    d = df[df["fxx"] <= horizon]

    st.altair_chart(line_panel(d, {"Temperature": ("t2m_C", C_ORANGE), "Wet-bulb (Romps)": ("tw2m_C", C_AQUA),
                                   "Dewpoint": ("td2m_C", C_BLUE)},
                               "2 m temperature, wet-bulb and dewpoint", "°C", x_title), width="stretch")
    st.altair_chart(precip_panel(d, x_title), width="stretch")
    st.altair_chart(line_panel(d, {"10 m wind": ("wspd10_ms", C_BLUE), "Gust": ("gust_ms", C_ORANGE),
                                   "100 m wind": ("wspd100_ms", C_AQUA)},
                               "Wind speed", "m/s", x_title, zero=True), width="stretch")

with tab_runs:
    try:
        runs = load("recent_runs")
    except Exception as e:
        st.info(f"Recent runs not available: {e}")
    else:
        label = st.selectbox("Variable", list(VARIABLES), index=0)
        col, unit = VARIABLES[label]
        r = runs[(runs["site"] == site) & (runs["interp"] == interp)]
        r = add_display_time(r, tz)
        st.altair_chart(runs_panel(r, col, label, unit, x_title), width="stretch")
        st.caption("Consistency between successive runs indicates forecast confidence; "
                   "large jumps suggest low predictability.")

with tab_table:
    show = ["time", "fxx", "t2m_C", "tw2m_C", "td2m_C", "wspd10_ms", "wdir10_deg", "gust_ms",
            "wspd100_ms", "precip_mm", "precip_acc_mm", "sp_hPa", "mslp_hPa"]
    st.dataframe(df[show].rename(columns={"time": x_title}), hide_index=True, width="stretch")
    st.download_button("Download CSV (this site/run)", df.drop(columns="time").to_csv(index=False),
                       file_name=f"gfs_{site}_{init:%Y%m%d%H}_{interp}.csv", mime="text/csv")
