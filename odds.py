#!/usr/bin/env python3
"""Free book lines from ESPN's scoreboard API (DraftKings odds).

No API key needed. Odds only exist for upcoming games — ESPN strips them
once games go final.
"""
import re
import requests

SCOREBOARD = ("https://site.api.espn.com/apis/site/v2/sports/football/nfl/"
              "scoreboard")


def _to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _parse_moneyline(s):
    """'SEA -145' -> -145 ; returns None if unparseable."""
    if not s:
        return None
    m = re.search(r"([+-]?\d+)\s*$", str(s))
    return int(m.group(1)) if m else None


def key_numbers_crossed(fair_spread, spread_home):
    """Which key numbers (3, 7) the model-vs-market gap straddles.

    fair_spread = model's E[home - away]; spread_home = book handicap
    (negative = home favored). Market-implied home margin = -spread_home.
    An edge crossing 3 or 7 is ~2-3x more valuable (15.1% / 9.0% of
    games land on those margins).
    """
    if fair_spread is None or spread_home is None:
        return []
    mkt = -spread_home
    out = []
    for k in (3, 7):
        if (fair_spread - k) * (mkt - k) < 0 or (fair_spread + k) * (mkt + k) < 0:
            out.append(k)
    return out

def book_lines(week, season=2026):
    """Return {(away_abbr, home_abbr): dict} for the given week.

    dict: spread_home (negative = home favored), total, ml_home, ml_away,
          favorite ('home'/'away'/None), provider.
    """
    params = {"week": week}
    if season != 2026:
        params["season"] = season
    try:
        d = requests.get(SCOREBOARD, params=params, timeout=20).json()
    except Exception:
        return {}
    out = {}
    for e in d.get("events", []):
        comp = (e.get("competitions") or [{}])[0]
        teams = comp.get("competitors", [])
        if len(teams) < 2:
            continue
        home = next((c for c in teams if c.get("homeAway") == "home"), None)
        away = next((c for c in teams if c.get("homeAway") == "away"), None)
        if not home or not away:
            continue
        ha = home["team"]["abbreviation"]
        aa = away["team"]["abbreviation"]
        odds = comp.get("odds") or []
        if not odds:
            continue
        o = odds[0]
        spread_home = _to_float(o.get("spread"))
        # ESPN's spread sign: negative = home favored (verified HOU@JAX -3).
        # Sanity-check against favorite flags and flip if inconsistent.
        fav_home = (o.get("homeTeamOdds") or {}).get("favorite")
        fav_away = (o.get("awayTeamOdds") or {}).get("favorite")
        if spread_home is not None:
            if fav_home and spread_home > 0:
                spread_home = -spread_home
            elif fav_away and spread_home < 0:
                spread_home = -spread_home
        out[(aa, ha)] = {
            "spread_home": spread_home,
            "total": _to_float(o.get("overUnder")),
            "ml_home": _parse_moneyline(o.get("moneyline")),
            "favorite": ("home" if fav_home else
                         "away" if fav_away else None),
            "provider": (o.get("provider") or {}).get("name", "ESPN"),
        }
    return out


if __name__ == "__main__":
    import sys
    wk = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    for (a, h), v in book_lines(wk).items():
        print(f"{a} @ {h}: spread_home={v['spread_home']} total={v['total']} "
              f"fav={v['favorite']} ({v['provider']})")
