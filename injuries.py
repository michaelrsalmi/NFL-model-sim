#!/usr/bin/env python3
"""ESPN injury feed + EPA-based point adjustments.

For each team, players listed as Out / Injured Reserve / Suspension
(full weight) or Doubtful (half weight) get their expected contribution
replaced with a replacement-level value, using nflverse play-by-play EPA:

    points_lost = (shrunk_player_epa_per_play - replacement_epa) * plays_per_game

- Player EPA/play is shrunk toward the positional mean (tames small samples).
- QBs use their own backup's EPA when he has >=15 dropbacks; otherwise the
  10th-percentile QB (unknown backups are usually bad).
- Skill players use the 25th-percentile EPA/play at their position.
- Team total is clipped to [-2, +10]: we never gift a team more than 2
  points for losing a player, no matter what the numbers say.

Apply by subtracting points_lost from the team's expected score (lambda).
"""
import json
import os
import re
import time
import unicodedata

import numpy as np
import pandas as pd
import requests

BASE = os.path.dirname(os.path.abspath(__file__))
INJ_URL = ("https://site.api.espn.com/apis/site/v2/sports/football/nfl/"
           "injuries")
CACHE = os.path.join(BASE, "data", "cache", "espn_injuries.json")

TEAM_MAP = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL",
    "Baltimore Ravens": "BAL", "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL", "Denver Broncos": "DEN",
    "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX", "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LAR", "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN", "New England Patriots": "NE",
    "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT", "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
}

FULL = {"Out", "Injured Reserve", "Suspension"}
HALF = {"Doubtful"}
MIN_PLAYS = {"QB": 30, "RB": 15, "WR": 15, "TE": 15}
SHRINK_W = {"QB": 80, "RB": 40, "WR": 40, "TE": 40}


def _key(name):
    """(first_initial, last_name) — matches 'J.Hurts' to 'Jalen Hurts'."""
    n = unicodedata.normalize("NFKD", name or "").lower()
    n = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b\.?", "", n)
    n = re.sub(r"[^a-z .]", "", n).replace(".", " ").strip()
    parts = re.sub(r"\s+", " ", n).split()
    if not parts:
        return ("", "")
    return (parts[0][0], parts[-1])


def fetch_injuries(max_age_hours=6):
    """Return {abbr: [(display_name, status)]} for actionable statuses."""
    if os.path.exists(CACHE):
        age = (time.time() - os.path.getmtime(CACHE)) / 3600
        if age < max_age_hours:
            return {k: [tuple(x) for x in v]
                    for k, v in json.load(open(CACHE)).items()}
    d = requests.get(INJ_URL, timeout=20).json()
    out = {}
    for t in d.get("injuries", []):
        abbr = TEAM_MAP.get(t.get("displayName", ""))
        if not abbr:
            continue
        for p in t.get("injuries", []):
            st = p.get("status", "")
            if st in FULL or st in HALF:
                nm = (p.get("athlete") or {}).get("displayName", "")
                if nm:
                    out.setdefault(abbr, []).append((nm, st))
    json.dump(out, open(CACHE, "w"))
    return out


_player_values = None


def player_values(season=2026):
    """Per-player offensive value. Columns: key, name, team, pos, plays,
    epa_per_play (shrunk), plays_per_game."""
    global _player_values
    if _player_values is not None:
        return _player_values
    d = pd.read_csv(os.path.join(BASE, "data", f"pbp_{season}.csv"),
                    low_memory=False, usecols=[
                        "epa", "qb_dropback", "passer_player_name",
                        "rusher_player_name", "receiver_player_name",
                        "posteam", "game_id", "pass_attempt", "rush_attempt"])
    d = d[d["epa"].notna()]
    team_games = d.groupby("posteam")["game_id"].nunique().to_dict()

    pos_map = {}
    try:
        from fetch import player_tds
        for abbr in TEAM_MAP.values():
            try:
                for p in player_tds(abbr, season):
                    pos_map[_key(p["name"])] = p.get("pos", "")
            except Exception:
                pass
    except Exception:
        pass

    rows = []
    qb = d[(d["qb_dropback"] == 1) & d["passer_player_name"].notna()]
    for (tm, nm), g in qb.groupby(["posteam", "passer_player_name"]):
        rows.append({"name": nm, "team": tm, "pos": "QB",
                     "plays": len(g), "epa": g["epa"].sum()})
    ru = d[(d["rush_attempt"] == 1) & d["rusher_player_name"].notna()]
    for (tm, nm), g in ru.groupby(["posteam", "rusher_player_name"]):
        rows.append({"name": nm, "team": tm,
                     "pos": pos_map.get(_key(nm), "RB"),
                     "plays": len(g), "epa": g["epa"].sum()})
    rc = d[(d["pass_attempt"] == 1) & d["receiver_player_name"].notna()]
    for (tm, nm), g in rc.groupby(["posteam", "receiver_player_name"]):
        rows.append({"name": nm, "team": tm,
                     "pos": pos_map.get(_key(nm), "WR"),
                     "plays": len(g), "epa": g["epa"].sum()})
    pdf = pd.DataFrame(rows)
    pdf["key"] = pdf["name"].map(_key)
    pdf = pdf.sort_values("plays", ascending=False).drop_duplicates("key")
    pdf["pos"] = pdf["pos"].replace({"FB": "RB"})
    pdf["pos"] = pdf["pos"].apply(lambda p: p if p in ("QB", "RB", "WR", "TE")
                                  else "WR")
    raw = pdf["epa"] / pdf["plays"]
    # shrink toward positional mean
    pos_mean = pdf.groupby("pos").apply(
        lambda g: np.average(g["epa"] / g["plays"], weights=g["plays"]))
    w = pdf["pos"].map(SHRINK_W)
    pdf["epa_per_play"] = ((pdf["plays"] * raw + w * pdf["pos"].map(pos_mean))
                           / (pdf["plays"] + w))
    pdf["plays_per_game"] = pdf.apply(
        lambda r: r["plays"] / max(1, team_games.get(r["team"], 5)), axis=1)
    _player_values = pdf.reset_index(drop=True)
    return _player_values


def _replacement_levels(pdf):
    repl, q10 = {}, {}
    for pos in ("QB", "RB", "WR", "TE"):
        sub = pdf[(pdf["pos"] == pos) & (pdf["plays"] >= 40)]
        if len(sub):
            repl[pos] = float(sub["epa_per_play"].quantile(0.25))
            q10[pos] = float(sub["epa_per_play"].quantile(0.10))
    repl.setdefault("QB", -0.08)
    q10.setdefault("QB", -0.19)
    repl.setdefault("RB", -0.10)
    repl.setdefault("WR", 0.10)
    repl.setdefault("TE", 0.05)
    return repl, q10


def team_injury_adjustment(abbr, injuries=None, season=2026):
    """Points `abbr` loses from inactive players.

    Returns (points_lost, [(name, status, pts_lost, pos)]).
    Positive = team is worse without them.
    """
    if injuries is None:
        injuries = fetch_injuries()
    lst = injuries.get(abbr, [])
    if not lst:
        return 0.0, []
    pdf = player_values(season)
    repl, q10 = _replacement_levels(pdf)
    team = pdf[pdf["team"] == abbr]
    qbs = team[team["pos"] == "QB"].sort_values("plays", ascending=False)
    backup_epa = None
    if len(qbs) >= 2 and qbs.iloc[1]["plays"] >= 15:
        backup_epa = float(qbs.iloc[1]["epa_per_play"])

    total, details = 0.0, []
    for name, status in lst:
        k = _key(name)
        # 1) same-team match first (avoids cross-team collisions like
        #    ARI's Jeremiyah Love vs GB's QB J.Love)
        m = team[team["key"] == k]
        if len(m) == 0:
            # 2) league-wide only if unambiguous
            m = pdf[pdf["key"] == k]
            if len(m) != 1:
                continue
            # 3) never borrow a different team's starter
            if m.iloc[0]["team"] != abbr and m.iloc[0]["plays"] >= 50:
                continue
        if len(m) == 0:
            continue
        r = m.iloc[0]
        pos = r["pos"]
        if r["plays"] < MIN_PLAYS[pos]:
            continue
        is_starter_qb = (pos == "QB" and len(qbs) and
                         r["key"] == qbs.iloc[0]["key"])
        if is_starter_qb and backup_epa is not None:
            rep = backup_epa
        elif pos == "QB":
            rep = q10["QB"]
        else:
            rep = repl[pos]
        pts = (float(r["epa_per_play"]) - rep) * float(r["plays_per_game"])
        pts *= 0.5 if status in HALF else 1.0
        if abs(pts) >= 0.3:
            details.append((name, status, round(pts, 1), pos))
            total += pts
    total = float(np.clip(total, -2.0, 10.0))
    details.sort(key=lambda x: -x[2])
    return round(total, 1), details


def out_keys(abbr, injuries=None):
    """Name-keys of players fully out (for TD-board filtering)."""
    if injuries is None:
        injuries = fetch_injuries()
    return {_key(nm) for nm, st in injuries.get(abbr, []) if st in FULL}


if __name__ == "__main__":
    inj = fetch_injuries()
    n = sum(len(v) for v in inj.values())
    print(f"{n} actionable injuries across {len(inj)} teams\n")
    for abbr in ("PHI", "TB", "KC", "LAC", "GB", "CHI", "SF", "DEN", "BAL",
                 "CLE", "DAL", "NYJ", "NE"):
        tot, det = team_injury_adjustment(abbr, inj)
        if tot or det:
            print(f"{abbr}: -{tot} pts")
            for name, st, pts, pos in det[:5]:
                print(f"   {name} ({pos}, {st}): {pts:+.1f}")
