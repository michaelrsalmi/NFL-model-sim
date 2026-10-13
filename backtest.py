#!/usr/bin/env python3
"""Walk-forward backtest: for each completed 2026 game in weeks 2-5,
predict using only data from earlier weeks, then score.

Metrics: Brier score, straight-up accuracy, spread MAE, total MAE.
"""
import math
import sys

from model import SeasonData, predict

TEST_WEEKS = [2, 3, 4, 5]
SIMS = 5000


def main():
    data = SeasonData()
    n = brier = correct = 0
    spread_err = total_err = 0.0
    rows = []
    for week in TEST_WEEKS:
        games = [g for g in data.cur if g["week"] == week and g["done"]]
        print(f"week {week}: {len(games)} games")
        for g in games:
            r = predict(g["away"], g["home"], data, max_week=week - 1,
                        sims=SIMS, seed=week * 100 + n, include_td=False)
            home_won = 1.0 if g["hs"] > g["as"] else (0.0 if g["hs"] < g["as"] else 0.5)
            brier += (r["p_home"] - home_won) ** 2
            correct += (r["p_home"] > 0.5) == (home_won == 1.0)
            spread_err += abs(r["fair_spread"] - (g["hs"] - g["as"]))
            total_err += abs(r["fair_total"] - (g["hs"] + g["as"]))
            n += 1
            rows.append((week, g["away"], g["home"], r["p_home"], home_won,
                         r["fair_spread"], g["hs"] - g["as"]))
    print()
    print(f"games: {n}")
    print(f"Brier score: {brier/n:.4f}  (null 0.2500; lower is better)")
    print(f"straight-up accuracy: {correct}/{n} = {correct/n:.1%}")
    print(f"spread MAE: {spread_err/n:.2f} pts")
    print(f"total MAE: {total_err/n:.2f} pts")
    print()
    print("biggest misses (model win prob vs result):")
    for w, a, h, p, won, fs, actual in sorted(rows, key=lambda r: -abs(r[3] - r[4]))[:8]:
        print(f"  wk{w} {a}@{h}: model {h} {p:.0%}, "
              f"{'won' if won == 1 else 'lost'}, spread err {abs(fs-actual):.1f}")


if __name__ == "__main__":
    main()
