"""Turnover-luck diagnostic.

Turnover margin barely repeats (fumble recoveries are ~50/50), so teams
with extreme margins are usually luckier/worse than their record.
Flag them; don't auto-adjust the model (EPA already handles it).
"""
import os
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
_NV_TO_ESPN = {"LA": "LAR", "ARZ": "ARI", "BLT": "BAL", "CLV": "CLE",
               "HST": "HOU", "JAX": "JAX", "WAS": "WSH"}

_cache = {}


def turnover_margins(season=2026, max_week=None):
    """Return {abbr: avg_turnover_margin_per_game}.

    Positive = team takes away more than it gives (likely lucky).
    """
    key = (season, max_week)
    if key in _cache:
        return _cache[key]
    path = os.path.join(BASE, "data", f"pbp_{season}.csv")
    if not os.path.exists(path):
        return {}
    d = pd.read_csv(path, low_memory=False,
                    usecols=["game_id", "week", "posteam", "defteam",
                             "interception", "fumble_lost"])
    if max_week is not None:
        d = d[d.week <= max_week]
    d = d[d["posteam"].notna()]
    to = ((d["interception"] == 1) | (d["fumble_lost"] == 1)).astype(int)
    d = d.assign(_to=to)
    # giveaways by posteam, takeaways by defteam
    gave = d.groupby("posteam")["_to"].sum()
    took = d.groupby("defteam")["_to"].sum()
    games = d.groupby("posteam")["game_id"].nunique()
    out = {}
    for tm in set(gave.index) | set(took.index):
        abbr = _NV_TO_ESPN.get(tm, tm)
        g = games.get(tm, 1)
        margin = (took.get(tm, 0) - gave.get(tm, 0)) / max(g, 1)
        out[abbr] = round(float(margin), 2)
    _cache[key] = out
    return out


def luck_flag(margin):
    """Human-readable flag for extreme turnover margins."""
    if margin >= 1.0:
        return f"turnover-lucky (+{margin}/gm — record flatters them)"
    if margin <= -1.0:
        return f"turnover-unlucky ({margin}/gm — better than record)"
    return None
