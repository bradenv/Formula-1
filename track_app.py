### Interactive track comparison
#
# Run with:  .venv/bin/streamlit run track_app.py
#
# Draws the circuit from OpenF1 car position data, then compares up to 3 drivers' laps:
#   - track map coloured by which driver was fastest through each mini-sector
#   - official sector times and gaps
#   - time delta and speed over the lap

import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

OPENF1_URL = "https://api.openf1.org/v1"

# Validated categorical palette: distinct for colour-blind viewers when all 3 are on the map together.
# Colours go by selection slot, not team, so teammates are still distinguishable.
DRIVER_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
MAX_DRIVERS = len(DRIVER_COLORS)
TRACK_GRAY = "#d4d3cf"
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#ecebe8"

MINI_SECTORS_PER_SECTOR = 8
GRID_POINTS = 1500
FROZEN_SECONDS = 1.2


@st.cache_data(show_spinner=False, ttl=3600)
def openf1(endpoint, **params):
    for attempt in range(6):
        resp = requests.get(f"{OPENF1_URL}/{endpoint}", params=params, timeout=60)
        if resp.status_code == 429:  # rate limited - back off and retry
            time.sleep(float(resp.headers.get("Retry-After", 2 ** attempt)))
            continue
        resp.raise_for_status()
        return pd.DataFrame(resp.json())
    st.error("OpenF1 is rate-limiting requests right now. Wait a minute, then change any filter to retry.")
    st.stop()


def fmt_lap(seconds):
    if pd.isna(seconds):
        return "-"
    minutes, secs = divmod(seconds, 60)
    return f"{int(minutes)}:{secs:06.3f}" if minutes else f"{secs:.3f}"


def fmt_delta(seconds):
    return "-" if pd.isna(seconds) else f"{seconds:+.3f}"


# --- Lap data processing ---

def fetch_lap_trace(session_key, driver_number, lap):
    """Telemetry and position for one lap, with time `t` (s) from lap start and fraction `frac` of the lap."""
    start = pd.Timestamp(lap["date_start"])
    duration = float(lap["lap_duration"])
    window = {"date>": start.isoformat(), "date<": (start + pd.to_timedelta(duration, unit="s")).isoformat()}

    car = openf1("car_data", session_key=session_key, driver_number=driver_number, **window)
    loc = openf1("location", session_key=session_key, driver_number=driver_number, **window)
    if car.empty or loc.empty:
        return None

    car["t"] = (pd.to_datetime(car["date"], format="ISO8601") - start).dt.total_seconds()
    car = car.sort_values("t", ignore_index=True)
    loc["t"] = (pd.to_datetime(loc["date"], format="ISO8601") - start).dt.total_seconds()
    loc = loc.sort_values("t", ignore_index=True)

    # The live feed sometimes freezes, repeating one speed for seconds (e.g. "308 km/h" through a
    # braking zone). Treat repeats lasting longer than FROZEN_SECONDS as missing and fill them in.
    run_id = (car["speed"] != car["speed"].shift()).cumsum()
    run_length = car.groupby(run_id)["t"].transform(lambda t: t.max() - t.min())
    frozen = (run_length > FROZEN_SECONDS) & run_id.duplicated()
    in_frozen_run = run_length > FROZEN_SECONDS
    frozen_spans = car.loc[in_frozen_run].groupby(run_id[in_frozen_run])["t"].agg(["min", "max"]).to_numpy()
    repaired_seconds = float((frozen_spans[:, 1] - frozen_spans[:, 0]).sum()) if len(frozen_spans) else 0.0
    car["speed"] = car["speed"].astype(float).mask(frozen).interpolate()

    # Anchor the trace to exactly 0 and the official lap time so deltas add up to the real gap
    path = pd.concat([
        pd.DataFrame({"t": [0.0], "speed": [car["speed"].iloc[0]]}),
        car[["t", "speed"]],
        pd.DataFrame({"t": [duration], "speed": [car["speed"].iloc[-1]]}),
    ], ignore_index=True)
    mean_speed = (path["speed"] + path["speed"].shift()) / 2 / 3.6
    path["distance"] = (mean_speed * path["t"].diff()).fillna(0).cumsum()
    path["frac"] = path["distance"] / path["distance"].iloc[-1]

    car["frac"] = np.interp(car["t"], path["t"], path["frac"])
    loc["frac"] = np.interp(loc["t"], path["t"], path["frac"])
    return {"car": car, "loc": loc, "path": path, "length": path["distance"].iloc[-1],
            "repaired_seconds": repaired_seconds, "frozen_spans": frozen_spans}


def sector_end_times(lap):
    s1, s2 = lap["duration_sector_1"], lap["duration_sector_2"]
    if pd.isna(s1) or pd.isna(s2):
        return None
    return np.array([s1, s1 + s2])


def build_comparison(traces, laps):
    """Put every driver on the reference driver's lap distance, aligned at the official sector lines."""
    ref = traces[0]
    ref_bounds_t = sector_end_times(laps[0])
    ref_bounds = (np.interp(ref_bounds_t, ref["path"]["t"], ref["path"]["frac"])
                  if ref_bounds_t is not None else np.array([1 / 3, 2 / 3]))

    grid = np.linspace(0, 1, GRID_POINTS)
    times, speeds = [], []
    for trace, lap in zip(traces, laps):
        path, car = trace["path"], trace["car"]
        path_frac, car_frac = path["frac"].to_numpy(), car["frac"].to_numpy()
        bounds_t = sector_end_times(lap)
        if bounds_t is not None:
            # Stretch each sector so everyone crosses the sector lines at the same point on the map
            bounds = np.interp(bounds_t, path["t"], path_frac)
            path_frac = np.interp(path_frac, [0, *bounds, 1], [0, *ref_bounds, 1])
            car_frac = np.interp(car_frac, [0, *bounds, 1], [0, *ref_bounds, 1])
        times.append(np.interp(grid, path_frac, path["t"]))
        speeds.append(np.interp(grid, car_frac, car["speed"]))

    # Mini-sectors evenly split within each official sector
    edges = np.concatenate([
        np.linspace(a, b, MINI_SECTORS_PER_SECTOR + 1)[:-1]
        for a, b in zip([0, *ref_bounds], [*ref_bounds, 1])
    ] + [[1.0]])
    edge_times = np.array([np.interp(edges, grid, t) for t in times])
    mini_times = np.diff(edge_times, axis=1)

    # Where on the lap each driver's speed was filled in, in metres
    estimated = [(d, *np.interp(span, t, grid) * ref["length"])
                 for d, (trace, t) in enumerate(zip(traces, times)) for span in trace["frozen_spans"]]

    return {
        "estimated": estimated,
        "grid": grid,
        "distance": grid * ref["length"],
        "times": np.array(times),
        "speeds": np.array(speeds),
        "sector_bounds": ref_bounds,
        "mini_edges": edges,
        "mini_times": mini_times,
        "mini_winner": mini_times.argmin(axis=0),
    }


# --- Charts ---

def base_layout(fig, height):
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_MUTED, size=13),
        legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, font=dict(color=TEXT)),
        hoverlabel=dict(bgcolor="white", font_color=TEXT),
    )


def track_map(comp, traces, names, sector_rows):
    loc = traces[0]["loc"]
    grid = comp["grid"]
    x = np.interp(grid, loc["frac"], loc["x"])
    y = np.interp(grid, loc["frac"], loc["y"])

    fig = go.Figure()
    # Outline underneath so the circuit reads as one shape
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color=TRACK_GRAY, width=14),
                             hoverinfo="skip", showlegend=False))

    edges, winners, mini_times = comp["mini_edges"], comp["mini_winner"], comp["mini_times"]
    for i, winner in enumerate(winners):
        mask = (grid >= edges[i]) & (grid <= edges[i + 1])
        times_here = mini_times[:, i]
        lines = [f"<b>Mini-sector {i + 1}</b>"] + [
            f"{names[d]}: {times_here[d]:.3f}s" + ("" if d == winner else f"  (+{times_here[d] - times_here[winner]:.3f})")
            for d in np.argsort(times_here)
        ]
        fig.add_trace(go.Scatter(
            x=x[mask], y=y[mask], mode="lines", line=dict(color=DRIVER_COLORS[winner], width=7),
            hovertemplate="<br>".join(lines) + "<extra></extra>", showlegend=False,
        ))

    # Legend entries: one per driver, with how many mini-sectors they won
    for d, name in enumerate(names):
        won = int((winners == d).sum())
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color=DRIVER_COLORS[d], width=7),
                                 name=f"{name} - fastest in {won}/{len(winners)} mini-sectors"))

    # Start/finish and sector lines
    for frac, label in [(0.0, "Start / Finish"), *[(b, f"S{i + 1} | S{i + 2}") for i, b in enumerate(comp["sector_bounds"])]]:
        px_, py_ = np.interp(frac, grid, x), np.interp(frac, grid, y)
        fig.add_trace(go.Scatter(x=[px_], y=[py_], mode="markers",
                                 marker=dict(size=13, color=TEXT, line=dict(color="white", width=2)),
                                 hovertemplate=f"{label}<extra></extra>", showlegend=False))

    # Sector labels at the middle of each sector: who was fastest and by how much
    bounds = [0, *comp["sector_bounds"], 1]
    # Rough label box size in track units (label is ~180x30 px on a ~1000x570 px plot area)
    units_per_px = max(np.ptp(x) / 1000, np.ptp(y) / 570)
    half_w, half_h = 95 * units_per_px, 18 * units_per_px
    angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    radii = np.array([0.6, 1.0, 1.5]) * half_w * 2
    placed = []
    for s, row in enumerate(sector_rows):
        mid = (bounds[s] + bounds[s + 1]) / 2
        mx, my = np.interp(mid, grid, x), np.interp(mid, grid, y)
        # Try spots around the sector midpoint and keep the one whose box is furthest from the
        # track and the labels already placed, preferring spots close to the sector
        cand_x = (mx + radii[:, None] * np.cos(angles)).ravel()
        cand_y = (my + radii[:, None] * np.sin(angles)).ravel()
        obstacles_x = np.concatenate([x[::5], [p[0] for p in placed]])
        obstacles_y = np.concatenate([y[::5], [p[1] for p in placed]])
        gap_x = np.maximum(np.abs(cand_x[:, None] - obstacles_x[None, :]) - half_w, 0)
        gap_y = np.maximum(np.abs(cand_y[:, None] - obstacles_y[None, :]) - half_h, 0)
        clearance = np.hypot(gap_x, gap_y).min(axis=1)
        clear_enough = clearance >= clearance.max() * 0.8
        best = np.flatnonzero(clear_enough)[0]  # candidates are ordered nearest radius first
        placed.append((cand_x[best], cand_y[best]))
        fig.add_annotation(x=mx, y=my, ax=cand_x[best], ay=cand_y[best], axref="x", ayref="y", text=row["label"],
                           showarrow=True, arrowcolor=TEXT_MUTED, arrowwidth=1, arrowhead=0,
                           font=dict(color=TEXT, size=13), align="left",
                           bgcolor="rgba(255,255,255,0.9)", bordercolor=GRID, borderpad=5)

    base_layout(fig, 620)
    # Fit the track plus the labels, which Plotly leaves out of its automatic range
    px_, py_ = [p[0] for p in placed], [p[1] for p in placed]
    fig.update_xaxes(visible=False, range=[min(x.min(), *px_) - half_w * 1.2, max(x.max(), *px_) + half_w * 1.2])
    fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1,
                     range=[min(y.min(), *py_) - half_h * 2, max(y.max(), *py_) + half_h * 2])
    return fig


def add_estimated_spans(fig, comp, names):
    for d, x0, x1 in comp["estimated"]:
        fig.add_vrect(x0=x0, x1=x1, fillcolor=TEXT_MUTED, opacity=0.08, line_width=0,
                      annotation_text=f"{names[d]} estimated", annotation_position="bottom left",
                      annotation_font=dict(color=TEXT_MUTED, size=11))


def add_sector_lines(fig, comp):
    for i, b in enumerate(comp["sector_bounds"]):
        fig.add_vline(x=b * comp["distance"][-1], line=dict(color=TEXT_MUTED, width=1, dash="dot"),
                      annotation_text=f"S{i + 1} | S{i + 2}", annotation_font_color=TEXT_MUTED,
                      annotation_position="top")


def delta_chart(comp, names):
    fig = go.Figure()
    fig.add_hline(y=0, line=dict(color=DRIVER_COLORS[0], width=2))
    for d in range(1, len(names)):
        delta = comp["times"][d] - comp["times"][0]
        fig.add_trace(go.Scatter(x=comp["distance"], y=delta, mode="lines", name=names[d],
                                 line=dict(color=DRIVER_COLORS[d], width=2),
                                 hovertemplate=f"{names[d]}: %{{y:+.3f}}s<extra></extra>"))
    add_estimated_spans(fig, comp, names)
    add_sector_lines(fig, comp)
    base_layout(fig, 300)
    fig.update_layout(margin_l=70, hovermode="x unified", showlegend=True)
    fig.update_xaxes(title="Distance (m)", gridcolor=GRID, zeroline=False)
    fig.update_yaxes(title=f"Gap to {names[0]} (s)", gridcolor=GRID, zeroline=False)
    return fig


def speed_chart(comp, names):
    fig = go.Figure()
    for d, name in enumerate(names):
        fig.add_trace(go.Scatter(x=comp["distance"], y=comp["speeds"][d], mode="lines", name=name,
                                 line=dict(color=DRIVER_COLORS[d], width=2),
                                 hovertemplate=f"{name}: %{{y:.0f}} km/h<extra></extra>"))
    add_estimated_spans(fig, comp, names)
    add_sector_lines(fig, comp)
    base_layout(fig, 300)
    fig.update_layout(margin_l=70, hovermode="x unified")
    fig.update_xaxes(title="Distance (m)", gridcolor=GRID, zeroline=False)
    fig.update_yaxes(title="Speed (km/h)", gridcolor=GRID, zeroline=False)
    return fig


# --- App ---

st.set_page_config(page_title="F1 Track Comparison", layout="wide")
st.title("F1 Track Comparison")

with st.sidebar:
    st.header("Session")
    year = st.selectbox("Year", [2026, 2025, 2024, 2023])
    meetings = openf1("meetings", year=year)
    meetings = meetings[~meetings["meeting_name"].str.contains("Testing")]
    meeting_labels = meetings["meeting_name"].tolist()
    default_meeting = next((i for i, m in enumerate(meeting_labels) if "Ital" in m), 0)
    meeting_name = st.selectbox("Grand Prix", meeting_labels, index=default_meeting)
    meeting_key = int(meetings.loc[meetings["meeting_name"] == meeting_name, "meeting_key"].iloc[0])

    sessions = openf1("sessions", meeting_key=meeting_key)
    session_names = sessions["session_name"].tolist()
    session_name = st.selectbox("Session", session_names,
                                index=session_names.index("Qualifying") if "Qualifying" in session_names else 0)
    session_key = int(sessions.loc[sessions["session_name"] == session_name, "session_key"].iloc[0])

drivers = openf1("drivers", session_key=session_key)
all_laps = openf1("laps", session_key=session_key)
if drivers.empty or all_laps.empty:
    st.info("No timing data for this session yet.")
    st.stop()

drivers = drivers.drop_duplicates("driver_number").set_index("driver_number")
valid_laps = all_laps.dropna(subset=["lap_duration"])
best = valid_laps.groupby("driver_number")["lap_duration"].min().sort_values()

with st.sidebar:
    st.header("Drivers")
    teams = sorted(drivers["team_name"].dropna().unique())
    team_filter = st.multiselect("Filter by team", teams, placeholder="All teams")
    candidates = [n for n in best.index if n in drivers.index
                  and (not team_filter or drivers.loc[n, "team_name"] in team_filter)]

    def driver_label(n):
        return f"{drivers.loc[n, 'name_acronym']} - {drivers.loc[n, 'team_name']}"

    selected = st.multiselect(f"Compare up to {MAX_DRIVERS} drivers", candidates, default=candidates[:2],
                              format_func=driver_label, max_selections=MAX_DRIVERS)
    if selected:
        st.caption(f"Gaps are measured against **{drivers.loc[selected[0], 'name_acronym']}** (first selected).")

    st.header("Laps")
    chosen_laps = []
    for n in selected:
        driver_laps = valid_laps[valid_laps["driver_number"] == n].sort_values("lap_number")
        fastest_idx = driver_laps["lap_duration"].idxmin()
        options = driver_laps.index.tolist()

        def lap_label(i, laps=driver_laps, fastest=fastest_idx):
            lap = laps.loc[i]
            tags = [" (fastest)" if i == fastest else "", " (pit out)" if lap["is_pit_out_lap"] else ""]
            return f"Lap {int(lap['lap_number'])} - {fmt_lap(lap['lap_duration'])}{''.join(tags)}"

        idx = st.selectbox(drivers.loc[n, "name_acronym"], options, index=options.index(fastest_idx),
                           format_func=lap_label, key=f"lap_{session_key}_{n}")
        chosen_laps.append(driver_laps.loc[idx])

if not selected:
    st.info("Pick at least one driver in the sidebar.")
    st.stop()

with st.spinner("Loading telemetry..."):
    traces = [fetch_lap_trace(session_key, n, lap) for n, lap in zip(selected, chosen_laps)]
missing = [drivers.loc[n, "name_acronym"] for n, t in zip(selected, traces) if t is None]
if missing:
    st.error(f"No telemetry available for: {', '.join(missing)}")
    st.stop()

names = [f"{drivers.loc[n, 'name_acronym']} L{int(lap['lap_number'])}" for n, lap in zip(selected, chosen_laps)]
comp = build_comparison(traces, chosen_laps)
repaired = [f"{name} ({t['repaired_seconds']:.1f}s)" for name, t in zip(names, traces) if t["repaired_seconds"]]
if repaired:
    st.caption(f"Frozen speed readings in the live feed were filled in for: {', '.join(repaired)}. "
               "Official sector and lap times are unaffected.")

# Official sector times
sector_cols = ["duration_sector_1", "duration_sector_2", "duration_sector_3"]
sector_times = np.array([[lap[c] for c in sector_cols] for lap in chosen_laps], dtype=float)
lap_times = np.array([lap["lap_duration"] for lap in chosen_laps], dtype=float)

sector_rows = []
for s in range(3):
    col = sector_times[:, s]
    if np.isnan(col).all():
        sector_rows.append({"label": f"<b>S{s + 1}</b>  no timing"})
        continue
    order = np.argsort(np.where(np.isnan(col), np.inf, col))
    w = order[0]
    margin = (f"  by {col[order[1]] - col[w]:.3f}s" if len(names) > 1 and not np.isnan(col[order[1]]) else "")
    sector_rows.append({"label": f"<b>S{s + 1}</b>  {names[w]}{margin}", "winner": w,
                        "time": col[w], "margin": margin})

# Header: who was fastest in each sector and overall
cols = st.columns(4)
for s, row in enumerate(sector_rows):
    if "winner" in row:
        cols[s].metric(f"Sector {s + 1}", f"{row['time']:.3f}s", f"{names[row['winner']]}{row['margin']}",
                       delta_color="off", delta_arrow="off")
overall = int(np.nanargmin(lap_times))
cols[3].metric("Lap", fmt_lap(lap_times[overall]), names[overall], delta_color="off", delta_arrow="off")

st.subheader("Track map")
st.caption("Each mini-sector is coloured by the driver who was quickest through it. Hover the track for times.")
st.plotly_chart(track_map(comp, traces, names, sector_rows), width="stretch")

st.subheader("Sector times")
table = pd.DataFrame({
    "Driver": names,
    "Team": [drivers.loc[n, "team_name"] for n in selected],
    **{f"S{s + 1}": [f"{t:.3f}" if not np.isnan(t) else "-" for t in sector_times[:, s]] for s in range(3)},
    **{f"S{s + 1} gap": [fmt_delta(t - sector_times[0, s]) for t in sector_times[:, s]] for s in range(3)},
    "Lap time": [fmt_lap(t) for t in lap_times],
    "Lap gap": [fmt_delta(t - lap_times[0]) for t in lap_times],
})
st.dataframe(table[["Driver", "Team", "S1", "S1 gap", "S2", "S2 gap", "S3", "S3 gap", "Lap time", "Lap gap"]],
             hide_index=True, width="stretch")

if len(names) > 1:
    st.subheader(f"Gap to {names[0]} around the lap")
    st.caption(f"Above zero = behind {names[0]}. A rising line means losing time on that stretch.")
    st.plotly_chart(delta_chart(comp, names), width="stretch")

st.subheader("Speed")
st.plotly_chart(speed_chart(comp, names), width="stretch")
