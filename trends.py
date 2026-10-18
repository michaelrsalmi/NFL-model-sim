"""Situational trend spotter.

Computes betting-relevant trends from completed games:
- Primetime underdogs/favorites (Thu/Sun/Mon night)
- Home underdogs, divisional underdogs
- Team ATS (vs model fair) and O/U streaks
- Favorites off a loss, etc.

"Underdog" = team our fair spread has as dog (we lack historical
closing lines for weeks 1-5; book_archive builds true ATS going forward).
"""
import json
import os
from datetime import datetime

BASE = os.path.dirname(os.path.abspath(__file__))


def _load_weeks():
    out = {}
    dd = os.path.join(BASE, "docs", "data")
    for fn in os.listdir(dd):
        if fn.startswith("week") and fn.endswith(".json") and fn[4].isdigit():
            w = int(fn[4:].split(".")[0])
            out[w] = json.load(open(os.path.join(dd, fn)))
    return out


def _is_primetime(date_iso):
    """Thu night, Sun night, Mon night (ET)."""
    try:
        dt = datetime.fromisoformat(date_iso.replace("Z", "+00:00"))
        # convert to ET roughly (UTC-4 in Oct)
        hr = (dt.hour - 4) % 24
        wd = dt.weekday()  # Mon=0
        # Thu 8pm+, Sun 8pm+, Mon 8pm+ ET
        if wd == 3 and hr >= 20:
            return True
        if wd == 6 and hr >= 20:
            return True
        if wd == 0 and hr >= 20:
            return True
        return False
    except Exception:
        return False


def compute_trends():
    weeks = _load_weeks()
    games = []
    for w, gs in weeks.items():
        for g in gs:
            if g.get("done") and g.get("winner") and g.get("fair_spread") is not None:
                games.append(g)

    trends = []

    def _rec(filter_fn, label):
        sel = [g for g in games if filter_fn(g)]
        if len(sel) < 5:
            return
        # ATS vs our fair spread (dog covers if they beat the fair number)
        ats_w = 0
        for g in sel:
            # our pick: team we favored vs fair
            # dog = team with fair_spread working against them
            # SU winner vs our SU pick
            pick_home = g["p_home"] > 0.5
            won = (g["winner"] == "home") == pick_home
            # for ATS we use margin vs fair spread
            margin = g["hs"] - g["as"]
            # did the dog cover our fair spread? dog covers if margin beats fair
            # (fair is E[home-away]; dog covers if actual is better than fair for them)
            if g["fair_spread"] > 0:  # we favored home, away is dog
                cover = margin < g["fair_spread"]  # away beats expectation
                dog_won_su = g["winner"] == "away"
            else:  # we favored away, home is dog
                cover = margin > g["fair_spread"]
                dog_won_su = g["winner"] == "home"
            ats_w += 1 if cover else 0
        trends.append({
            "label": label,
            "n": len(sel),
            "ats": f"{ats_w}-{len(sel)-ats_w}",
            "pct": round(100 * ats_w / len(sel)),
        })

    # situational
    _rec(lambda g: _is_primetime(g["date"]) and g["fair_spread"] > 1.5,
         "Primetime home favorites")
    _rec(lambda g: _is_primetime(g["date"]) and g["fair_spread"] < -1.5,
         "Primetime road favorites")
    # underdogs in primetime (either side getting points from our fair)
    def _primetime_dog(g):
        if not _is_primetime(g["date"]):
            return False
        return abs(g["fair_spread"]) > 1.5
    _rec(_primetime_dog, "Primetime games (dog covers vs fair)")
    _rec(lambda g: g.get("divisional") and abs(g["fair_spread"]) > 1.5,
         "Divisional dogs vs fair")
    _rec(lambda g: g["fair_spread"] > 3, "Home favorites (-3+) vs fair")
    _rec(lambda g: g["fair_spread"] < -3, "Road favorites (-3+) vs fair")

    # team streaks (SU and vs fair)
    teams = {}
    for g in sorted(games, key=lambda x: x["date"]):
        for side, tm in (("away", g["away"]), ("home", g["home"])):
            t = teams.setdefault(tm, {"su": [], "ats": []})
            won = (g["winner"] == side)
            t["su"].append("W" if won else "L")
            margin = g["hs"] - g["as"]
            if side == "home":
                cover = (margin > 0) == (g["p_home"] > 0.5)
            else:
                cover = (margin < 0) == (g["p_home"] < 0.5)
            # simpler: did they beat our fair expectation?
            if side == "home":
                beat = margin > g["fair_spread"]
            else:
                beat = margin < g["fair_spread"]
            t["ats"].append("W" if beat else "L")

    streaks = []
    for tm, t in teams.items():
        for key, label in (("su", "SU"), ("ats", "vs fair")):
            s = t[key]
            if len(s) >= 3:
                last = s[-1]
                n = 1
                for r in reversed(s[:-1]):
                    if r == last:
                        n += 1
                    else:
                        break
                if n >= 3:
                    streaks.append({"team": tm, "type": label,
                                    "streak": f"{n}{last}"})
    streaks.sort(key=lambda x: -int(x["streak"][:-1]))

    return {"situational": trends, "streaks": streaks[:12]}


if __name__ == "__main__":
    t = compute_trends()
    print("=== SITUATIONAL ===")
    for s in t["situational"]:
        print(f'{s["label"]}: {s["ats"]} ({s["pct"]}%) n={s["n"]}')
    print("\n=== STREAKS ===")
    for s in t["streaks"]:
        print(f'{s["team"]} {s["type"]}: {s["streak"]}')
