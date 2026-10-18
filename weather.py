"""Wind weather feed for NFL totals (Open-Meteo, free, no key).

Research: each mph of sustained wind ≈ -0.17 pts on the total,
meaningful at 15+ mph, accelerating past 20 mph. Only valuable vs the
book when the market hasn't moved yet — so we apply it to the fair
total and let the edge-vs-book comparison do its job.
"""
import json
import os
import time
import urllib.request
from datetime import datetime, timedelta

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "cache")
CACHE_TTL = 6 * 3600  # 6 hours

# stadium coordinates (lat, lon)
STADIUMS = {
    "ARI": (33.5277, -112.2626), "ATL": (33.7554, -84.4008),
    "BAL": (39.2780, -76.6227), "BUF": (42.7738, -78.7869),
    "CAR": (35.2258, -80.8529), "CHI": (41.8623, -87.6167),
    "CIN": (39.0955, -84.5161), "CLE": (41.5061, -81.6996),
    "DAL": (32.7473, -97.0945), "DEN": (39.7439, -105.0201),
    "DET": (42.3400, -83.0456), "GB": (44.5013, -88.0622),
    "HOU": (29.6847, -95.4107), "IND": (39.7601, -86.1639),
    "JAX": (30.3239, -81.6373), "KC": (39.0489, -94.4839),
    "LAC": (33.9535, -118.3392), "LAR": (33.9535, -118.3392),
    "LV": (36.0908, -115.1838), "MIA": (25.9580, -80.2389),
    "MIN": (44.9738, -93.2577), "NE": (42.0909, -71.2643),
    "NO": (29.9511, -90.0812), "NYG": (40.8135, -74.0745),
    "NYJ": (40.8135, -74.0745), "PHI": (39.9008, -75.1675),
    "PIT": (40.4468, -79.9921), "SEA": (47.5952, -122.3316),
    "SF": (37.4030, -121.9700), "TB": (27.9759, -82.5033),
    "TEN": (36.1665, -86.7713), "WSH": (38.9076, -76.8644),
}
# domes / retractable roofs typically closed in bad weather — wind N/A
DOMES = {"ARI", "ATL", "DET", "HOU", "IND", "LAC", "LAR", "LV",
         "MIN", "NO", "DAL"}

WIND_PTS_PER_MPH = 0.17
WIND_THRESHOLD = 15.0  # mph — below this, effect is negligible


def _cache_path(key):
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"wind_{key}.json")


def game_wind(home, game_time_iso):
    """Return (wind_mph, pts_adjustment) for a game, or (None, 0.0).

    game_time_iso: ISO datetime string (UTC) of kickoff.
    """
    home = home.upper()
    if home in DOMES:
        return None, 0.0
    lat, lon = STADIUMS.get(home, (None, None))
    if lat is None:
        return None, 0.0
    try:
        dt = datetime.fromisoformat(game_time_iso.replace("Z", "+00:00"))
    except Exception:
        return None, 0.0
    # only forecastable ~16 days out
    if dt > datetime.now(dt.tzinfo) + timedelta(days=15):
        return None, 0.0
    if dt < datetime.now(dt.tzinfo) - timedelta(hours=3):
        return None, 0.0  # game already started

    day = dt.strftime("%Y-%m-%d")
    ck = f"{home}_{day}"
    cp = _cache_path(ck)
    data = None
    if os.path.exists(cp) and time.time() - os.path.getmtime(cp) < CACHE_TTL:
        try:
            data = json.load(open(cp))
        except Exception:
            data = None
    if data is None:
        url = (f"https://api.open-meteo.com/v1/forecast?latitude={lat}"
               f"&longitude={lon}&hourly=wind_speed_10m"
               f"&start_date={day}&end_date={day}"
               f"&timezone=UTC&wind_speed_unit=mph")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "nfl-model"})
            data = json.load(urllib.request.urlopen(req, timeout=15))
            json.dump(data, open(cp, "w"))
        except Exception:
            return None, 0.0
    try:
        hours = data["hourly"]["time"]
        winds = data["hourly"]["wind_speed_10m"]
        # wind at kickoff hour (and the next 3 hours, take max)
        target = dt.replace(minute=0, second=0, microsecond=0)
        best = 0.0
        for i, t in enumerate(hours):
            ht = datetime.fromisoformat(t.replace("Z", "+00:00"))
            if ht.tzinfo is None:
                ht = ht.replace(tzinfo=dt.tzinfo)
            if target <= ht <= target + timedelta(hours=3):
                if winds[i] is not None:
                    best = max(best, winds[i])
        if best < WIND_THRESHOLD:
            return round(best, 1), 0.0
        # accelerating past 20 mph
        adj = best * WIND_PTS_PER_MPH
        if best >= 20:
            adj *= 1.3
        return round(best, 1), round(adj, 1)
    except Exception:
        return None, 0.0
