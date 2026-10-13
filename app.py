"""Model Sim web app: FastAPI backend serving the NFL Monte Carlo model.

Endpoints:
  GET /api/health
  GET /api/slate?week=N        -> every game that week with predictions
  GET /api/predict?away=X&home=Y&sims=N&seed=S -> full prediction + TD boards
  POST /api/refresh            -> clear caches, reload data
  GET /                        -> the frontend
"""
import os
import threading

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from fetch import all_games, PFR_ABBR
from model import SeasonData, predict, MAX_DONE_WEEK, SEASON

app = FastAPI(title="Model Sim")
BASE = os.path.dirname(os.path.abspath(__file__))
app.mount("/static", StaticFiles(directory=os.path.join(BASE, "static")),
          name="static")

DATA = None
DATA_LOCK = threading.Lock()
SLATE_CACHE = {}

# nflverse play-by-play is too big for git -- download on first boot.
PBP_URLS = {
    2026: "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_2026.csv",
    2025: "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_2025.csv",
}


def ensure_pbp():
    import urllib.request
    os.makedirs(os.path.join(BASE, "data"), exist_ok=True)
    for season, url in PBP_URLS.items():
        path = os.path.join(BASE, "data", f"pbp_{season}.csv")
        if not os.path.exists(path):
            print(f"downloading play-by-play {season}...", flush=True)
            urllib.request.urlretrieve(url, path)
            print(f"saved {path}", flush=True)


def get_data():
    global DATA
    if DATA is None:
        with DATA_LOCK:
            if DATA is None:
                ensure_pbp()
                globals()["DATA"] = SeasonData()
    return DATA


def _slim(r):
    return {
        "away": r["away"], "home": r["home"],
        "p_home": round(r["p_home"], 4), "p_away": round(r["p_away"], 4),
        "proj_away": round(r["proj_away"], 1),
        "proj_home": round(r["proj_home"], 1),
        "fair_spread": round(r["fair_spread"], 1),
        "fair_total": round(r["fair_total"], 1),
    }


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/slate")
def slate(week: int = 5):
    """All games in a week. Predictions use data through the prior week."""
    if week in SLATE_CACHE:
        return SLATE_CACHE[week]
    data = get_data()
    games = [g for g in data.cur if g["week"] == week]
    out = []
    for g in games:
        r = predict(g["away"], g["home"], data, max_week=week - 1,
                    sims=10000, include_td=False)
        row = _slim(r)
        row["date"] = g["date"]
        row["done"] = g["done"]
        if g["done"]:
            row["final"] = f"{g['as']}-{g['hs']}"
            row["winner"] = g["away"] if g["as"] > g["hs"] else (
                g["home"] if g["hs"] > g["as"] else "TIE")
        out.append(row)
    res = {"week": week, "games": out,
           "as_of_week": MAX_DONE_WEEK,
           "backtest": {"brier": 0.2235, "accuracy": 0.661, "games": 62,
                        "note": "walk-forward, 2026 weeks 2-5"}}
    SLATE_CACHE[week] = res
    return res


@app.get("/api/predict")
def predict_game(away: str, home: str, sims: int = 20000, seed: int = None):
    away, home = away.upper(), home.upper()
    if away not in PFR_ABBR or home not in PFR_ABBR:
        raise HTTPException(400, "unknown team abbreviation")
    if away == home:
        raise HTTPException(400, "teams must differ")
    sims = max(1000, min(100000, sims))
    r = predict(away, home, get_data(), sims=sims, seed=seed)
    return r


@app.post("/api/refresh")
def refresh():
    global DATA
    with DATA_LOCK:
        DATA = None
        SLATE_CACHE.clear()
    get_data()
    return {"ok": True, "as_of_week": MAX_DONE_WEEK}


@app.get("/")
def index():
    return FileResponse(os.path.join(BASE, "static", "index.html"))
