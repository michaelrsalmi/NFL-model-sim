#!/usr/bin/env python3
"""NFL Monte Carlo game simulator + TD-scorer model (CLI).

    python3 simulate.py --away BUF --home LAR --sims 20000
    python3 simulate.py --away BUF --home LAR --line -2.5 --total 49.5
    python3 simulate.py --away KC --home PHI --sims 50000 --refresh
    python3 simulate.py --away BUF --home LAR --out "Dalton Kincaid"

Each run executes N full-game simulations and aggregates them, so one run
already gives the stable, averaged "best result".
"""
import argparse
import sys

from fetch import PFR_ABBR
from model import SeasonData, predict, MAX_DONE_WEEK


def fmt_odds(o):
    if o is None:
        return "  -- "
    return f"{o:+d}" if o < 0 else f"+{o}"


def main():
    ap = argparse.ArgumentParser(description="NFL Monte Carlo simulator")
    ap.add_argument("--away", default="BUF", help="away team abbreviation")
    ap.add_argument("--home", default="LAR", help="home team abbreviation")
    ap.add_argument("--sims", type=int, default=20000)
    ap.add_argument("--line", type=float, default=None,
                    help="home spread for cover prob, e.g. -2.5")
    ap.add_argument("--total", type=float, default=None,
                    help="game total for over/under prob, e.g. 49.5")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--out", action="append", default=[],
                    help="scratched player (repeatable)")
    ap.add_argument("--week", type=int, default=None,
                    help="predict as of this completed week (backtest use)")
    args = ap.parse_args()

    away, home = args.away.upper(), args.home.upper()
    for t in (away, home):
        if t not in PFR_ABBR:
            sys.exit(f"unknown team {t}")

    print(f"loading data ({away} @ {home})...")
    data = SeasonData(refresh=args.refresh)
    r = predict(away, home, data,
                max_week=args.week or MAX_DONE_WEEK,
                sims=args.sims, seed=args.seed, refresh=args.refresh)

    if args.out:
        # re-run TD board minus scratched players
        from fetch import player_tds
        from model import SEASON
        from td_model import build_td_board
        ap_ = [p for p in player_tds(away, SEASON) if p["name"] not in args.out]
        hp_ = [p for p in player_tds(home, SEASON) if p["name"] not in args.out]
        anytime, first = build_td_board(ap_, hp_, r["lam_away"], r["lam_home"])
        r["anytime"] = [{"name": p["name"], "team": p["team"],
                         "prob": p["p_anytime"], "odds": p["any_odds"]}
                        for p in anytime[:12]]
        r["first_td"] = [{"name": p["name"], "team": p["team"],
                          "prob": p["p_first"], "odds": p["first_odds"]}
                         for p in first[:8]]

    n = r["sims"]
    print()
    print(f"===== {away} @ {home}  |  {n:,} simulations =====")
    print(f"win prob:  {home} {r['p_home']:.1%}  |  {away} {r['p_away']:.1%}  "
          f"(sim only: {r['p_sim_home']:.1%}, Elo only: {r['p_elo_home']:.1%})")
    print(f"projected: {away} {r['proj_away']:.1f} - {home} {r['proj_home']:.1f}")
    print(f"fair spread (home view): {r['fair_spread']:+.1f}   "
          f"fair total: {r['fair_total']:.1f}")
    if args.line is not None:
        import numpy as np
        rng = np.random.default_rng(args.seed)
        sa = rng.poisson(r["lam_away"], n)
        sh = rng.poisson(r["lam_home"], n)
        cover = ((sh - sa + args.line) > 0).mean()
        over = ((sa + sh) > args.total).mean() if args.total else None
        print(f"vs line {args.line:+}: {home} covers {cover:.1%}")
        if over is not None:
            print(f"vs total {args.total}: over {over:.1%} / under {1-over:.1%}")
    elif args.total is not None:
        import numpy as np
        rng = np.random.default_rng(args.seed)
        sa = rng.poisson(r["lam_away"], n)
        sh = rng.poisson(r["lam_home"], n)
        over = ((sa + sh) > args.total).mean()
        print(f"vs total {args.total}: over {over:.1%} / under {1-over:.1%}")
    print(f"Elo ratings: {home} {r['elo_home']:.0f} vs {away} {r['elo_away']:.0f}")

    print()
    print("--- anytime TD (top 12) ---")
    for p in r["anytime"]:
        print(f"  {p['name']} ({p['team']})  {p['prob']:.1%}  {fmt_odds(p['odds'])}")
    print("--- first TD (top 8) ---")
    for p in r["first_td"]:
        print(f"  {p['name']} ({p['team']})  {p['prob']:.2%}  {fmt_odds(p['odds'])}")
    print()
    print("notes: Poisson score model; ties split; TD shares from season "
          "rushing+receiving TDs. Re-run any time -- results converge ~+/-0.3%.")


if __name__ == "__main__":
    main()
