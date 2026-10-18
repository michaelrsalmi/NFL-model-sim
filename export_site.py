#!/usr/bin/env python3
"""Export the full static site: walk-forward predictions for every game
of weeks 1-18, with final scores + hit/miss for completed games.

Output: docs/index.html (self-contained, served by GitHub Pages).
"""
import json
import os

from fetch import all_games, player_tds
from model import SeasonData, predict, SEASON
from odds import book_lines
from ratings import adjusted_scoring
from epa import team_epa

BASE = os.path.dirname(os.path.abspath(__file__))


def main():
    # Scores/results change after every game: never trust cached scoreboards
    # for the current season (PFR player caches are fine to reuse intraday).
    import glob
    for f in glob.glob(os.path.join(BASE, "data", "cache", "espn_sb_2026_*.json")):
        os.remove(f)
    data = SeasonData()
    _done = [w for w in range(1, 19)
             if [g for g in data.cur if g["week"] == w]
             and all(g["done"] for g in data.cur if g["week"] == w)]
    MAXDW = max(_done) if _done else 0  # last fully completed week

    # ---- team params for the JS sim lab (as of latest completed week) ----
    from model import _run_elo
    from ratings import regress_offseason, win_prob, ELO_INIT, ELO_K, _mov_mult
    cur_done = data.done_through(MAXDW)
    off, deff, lg = adjusted_scoring(cur_done)
    off25, deff25, _ = adjusted_scoring(data.prior)
    off25 = {t: v * 0.6 for t, v in off25.items()}
    deff25 = {t: v * 0.6 for t, v in deff25.items()}
    gp = {}
    for g in cur_done:
        gp[g["away"]] = gp.get(g["away"], 0) + 1
        gp[g["home"]] = gp.get(g["home"], 0) + 1

    def shrink(t, v):
        n = gp.get(t, 4)
        return v * n / (n + 8.0)

    teams = set(off) | set(off25)
    off_b = {t: (gp.get(t, 0) / (gp.get(t, 0) + 10.0)) * off.get(t, 0.0) +
                (10.0 / (gp.get(t, 0) + 10.0)) * off25.get(t, 0.0) for t in teams}
    deff_b = {t: (gp.get(t, 0) / (gp.get(t, 0) + 10.0)) * deff.get(t, 0.0) +
                 (10.0 / (gp.get(t, 0) + 10.0)) * deff25.get(t, 0.0) for t in teams}
    off_e, def_e, lg_e, ppg = team_epa(SEASON, MAXDW)
    off_e25, def_e25, _, _ = team_epa(2025)

    def eblend(cur, prior, t):
        n = gp.get(t, 4)
        w = n / (n + 10.0)
        return w * cur.get(t, lg_e) + (1 - w) * (
            lg_e + (prior.get(t, lg_e) - lg_e) * 0.6)

    elo = regress_offseason(_run_elo(data.prior))
    for g in cur_done:
        a, h = g["away"], g["home"]
        ea, eh = elo.get(a, ELO_INIT), elo.get(h, ELO_INIT)
        pa = win_prob(ea, eh, home_team="b")
        sa = 0.5 if g["as"] == g["hs"] else (1.0 if g["as"] > g["hs"] else 0.0)
        k = ELO_K * _mov_mult(abs(g["as"] - g["hs"]), ea - eh)
        sh = k * (sa - pa)
        elo[a], elo[h] = ea + sh, eh - sh

    teams_out = {}
    for t in teams:
        players = []
        try:
            for p in player_tds(t, SEASON):
                players.append({"name": p["name"], "pos": p["pos"],
                                "td": p["td"], "touches": p["touches"]})
        except Exception:
            pass
        teams_out[t] = {
            "srs_off": shrink(t, off_b.get(t, 0.0)),
            "srs_def": shrink(t, deff_b.get(t, 0.0)),
            "epa_off": eblend(off_e, off_e25, t),
            "epa_def": eblend(def_e, def_e25, t),
            "elo": elo.get(t, ELO_INIT),
            "players": players,
        }

    # ---- book archive: once a line is captured, keep it forever ----
    # (ESPN strips odds after games go final)
    arc_path = os.path.join(BASE, "docs", "data", "book_archive.json")
    try:
        archive = json.load(open(arc_path))
    except Exception:
        archive = {}

    # ---- walk-forward slate for every week ----
    weeks = {}
    for week in range(1, 19):
        games = [g for g in data.cur if g["week"] == week]
        if not games:
            continue
        books = book_lines(week)  # free DraftKings lines via ESPN (upcoming only)
        for (a, h), b in books.items():
            if b and b["spread_home"] is not None:
                archive[f"{a}@{h}_{week}"] = {
                    "spread": b["spread_home"], "total": b["total"],
                    "fav": b["favorite"], "by": b["provider"]}
        out = []
        for i, g in enumerate(games):
            key = f"{g['away']}@{g['home']}_{week}"
            ab = archive.get(key)  # archived line (survives after final)
            r = predict(g["away"], g["home"], data, max_week=week - 1,
                        sims=10000, seed=week * 100 + i,
                        include_td=not g["done"], use_injuries=not g["done"])
            row = {
                "away": r["away"], "home": r["home"], "date": g["date"],
                "done": g["done"],
                "p_home": round(r["p_home"], 3), "p_away": round(r["p_away"], 3),
                "proj_away": round(r["proj_away"], 1),
                "proj_home": round(r["proj_home"], 1),
                "fair_spread": round(r["fair_spread"], 1),
                "fair_total": round(r["fair_total"], 1),
                "inj_adj": r["inj_adj"],
                "injuries": r["injuries"],
            }
            if not g["done"]:
                # full TD boards only for upcoming games (keeps bundle small)
                row["anytime"] = r["anytime"]
                row["first_td"] = r["first_td"]
                b = books.get((g["away"], g["home"]))
                if b and b["spread_home"] is not None:
                    # fair_spread = E[home pts - away pts]; book spread_home is
                    # the handicap (negative = home favored), so flip its sign
                    edge_s = round(r["fair_spread"] + b["spread_home"], 1)
                    edge_t = (round(r["fair_total"] - b["total"], 1)
                              if b["total"] is not None else None)
                    row["book"] = {
                        "spread": b["spread_home"], "total": b["total"],
                        "fav": b["favorite"], "by": b["provider"],
                        "edge_spread": edge_s, "edge_total": edge_t,
                        # model minus book: +spread edge => model likes home more
                        "flag_spread": abs(edge_s) >= 2.0,
                        "flag_total": edge_t is not None and abs(edge_t) >= 3.0,
                    }
                elif ab:
                    row["book"] = ab
            else:
                row["anytime"] = []
                row["first_td"] = []
                if ab:
                    # keep the archived line so ATS/O/U can be marked
                    row["book"] = ab
            if g["done"]:
                row["final"] = f"{g['as']}-{g['hs']}"
                row["as"], row["hs"] = g["as"], g["hs"]
                row["winner"] = ("away" if g["as"] > g["hs"]
                                 else "home" if g["hs"] > g["as"] else "tie")
            out.append(row)
        weeks[str(week)] = out
        print(f"week {week}: {len(out)} games", flush=True)

    # ---- write docs/ data files (per-week JSON keeps each file small) ----
    docs = os.path.join(BASE, "docs")
    datadir = os.path.join(docs, "data")
    os.makedirs(datadir, exist_ok=True)
    for week, games in weeks.items():
        with open(os.path.join(datadir, f"week{week}.json"), "w") as f:
            json.dump(games, f)
    with open(os.path.join(datadir, "teams.json"), "w") as f:
        json.dump(teams_out, f)
    with open(os.path.join(datadir, "meta.json"), "w") as f:
        json.dump({"lg_pts": lg, "lg_epa": lg_e, "ppg": ppg, "hfa_pts": 1.0,
                   "as_of_week": MAXDW,
                   "backtest": {"brier": 0.2198, "accuracy": 0.619,
                                "games": 63}}, f)
    with open(os.path.join(datadir, "weeks.json"), "w") as f:
        json.dump(sorted(int(w) for w in weeks), f)
    with open(arc_path, "w") as f:
        json.dump(archive, f)

    # ---- docs/index.html from the fetch-based template ----
    import shutil
    shutil.copy(os.path.join(BASE, "site_template.html"),
                os.path.join(docs, "index.html"))
    kb = os.path.getsize(os.path.join(docs, "index.html")) // 1024
    nfiles = len(os.listdir(datadir))
    print(f"docs/index.html ({kb} KB) + {nfiles} data files written")


if __name__ == "__main__":
    main()
