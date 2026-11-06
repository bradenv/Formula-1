### Connect to Formula 1 API
#
# Data sources (all free, no login):
#   1. Jolpica API (successor to Ergast) - historical results, standings, drivers, teams (1950-present)
#   2. OpenF1 API - lap times and car telemetry (speed, throttle, brake, gear, RPM, DRS) from 2023+
#
# FastF1 is also installed. Its timing/telemetry now requires an F1 TV subscription
# (run `.venv/bin/python -m fastf1 auth f1tv`), but schedules and results work without one.

import matplotlib.pyplot as plt
import pandas as pd
import requests

JOLPICA_URL = "https://api.jolpi.ca/ergast/f1"
OPENF1_URL = "https://api.openf1.org/v1"


def openf1(endpoint, **params):
    resp = requests.get(f"{OPENF1_URL}/{endpoint}", params=params, timeout=60)
    resp.raise_for_status()
    return pd.DataFrame(resp.json())


# --- 1. Generic driver / team data (Jolpica) ---

def get_driver_standings(season="current"):
    data = requests.get(f"{JOLPICA_URL}/{season}/driverStandings.json", timeout=30).json()
    standings = data["MRData"]["StandingsTable"]["StandingsLists"][0]["DriverStandings"]
    return pd.DataFrame([
        {
            "position": int(s["position"]),
            "driver": f"{s['Driver']['givenName']} {s['Driver']['familyName']}",
            "team": s["Constructors"][0]["name"],
            "points": float(s["points"]),
            "wins": int(s["wins"]),
        }
        for s in standings
    ])


# --- 2. Real car data (OpenF1) ---

def get_session_key(year, location, session_name="Qualifying"):
    """session_name: 'Practice 1', 'Practice 2', 'Practice 3', 'Qualifying', 'Sprint', 'Race'"""
    sessions = openf1("sessions", year=year, location=location, session_name=session_name)
    return int(sessions.iloc[0]["session_key"])


def get_fastest_lap_telemetry(session_key, driver_number):
    laps = openf1("laps", session_key=session_key, driver_number=driver_number).dropna(subset=["lap_duration"])
    lap = laps.loc[laps["lap_duration"].idxmin()]
    start = pd.Timestamp(lap["date_start"])
    end = start + pd.Timedelta(seconds=lap["lap_duration"])
    tel = openf1("car_data", session_key=session_key, driver_number=driver_number,
                 **{"date>": start.isoformat(), "date<": end.isoformat()})
    tel["date"] = pd.to_datetime(tel["date"], format="ISO8601")
    # Integrate speed over time to get distance around the lap
    dt = tel["date"].diff().dt.total_seconds().fillna(0)
    tel["distance"] = (tel["speed"] / 3.6 * dt).cumsum()
    return lap, tel


def plot_fastest_lap_speed(session_key, driver_numbers):
    drivers = openf1("drivers", session_key=session_key).set_index("driver_number")
    fig, ax = plt.subplots(figsize=(12, 5))
    for num in driver_numbers:
        lap, tel = get_fastest_lap_telemetry(session_key, num)
        info = drivers.loc[num]
        ax.plot(tel["distance"], tel["speed"], color=f"#{info['team_colour']}",
                label=f"{info['name_acronym']} ({lap['lap_duration']:.3f}s)")
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Speed (km/h)")
    ax.set_title("Fastest Lap Speed Comparison")
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print(get_driver_standings().head(10).to_string(index=False))

    session_key = get_session_key(2026, "Monza", "Qualifying")
    plot_fastest_lap_speed(session_key, [12, 63])  # Antonelli v Russel
