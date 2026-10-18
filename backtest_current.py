"""Walk-forward backtest of the CURRENT model on completed weeks.

For each week W (2..6): predict using only data through week W-1,
compare to actual. Injuries/wind OFF (no historical data).
"""
import sys
sys.path.insert(0, "/home/hatch/workspace/nfl-model")
from model import predict, SeasonData

data = SeasonData()
results = []
for week in [2, 3, 4, 5, 6]:
    games = [g for g in data.cur if g["week"] == week and g.get("done")]
    for g in games:
        try:
            r = predict(g["away"], g["home"], data, max_week=week - 1,
                        sims=8000, seed=week * 100 + 7,
                        include_td=False, use_injuries=False)
        except Exception as e:
            print(f"skip {g['away']}@{g['home']} w{week}: {e}")
            continue
        actual_home = 1.0 if g["hs"] > g["as"] else 0.0
        results.append({
            "week": week,
            "away": g["away"], "home": g["home"],
            "p_home": r["p_home"],
            "fair_spread": r["fair_spread"],
            "fair_total": r["fair_total"],
            "actual_home": actual_home,
            "margin": g["hs"] - g["as"],
            "total": g["hs"] + g["as"],
        })

n = len(results)
brier = sum((r["p_home"] - r["actual_home"]) ** 2 for r in results) / n
su = sum(1 for r in results if (r["p_home"] > 0.5) == (r["actual_home"] == 1.0)) / n
mae_spread = sum(abs(r["fair_spread"] - r["margin"]) for r in results) / n
mae_total = sum(abs(r["fair_total"] - r["total"]) for r in results) / n

print(f"n={n}  Brier={brier:.4f}  SU={su:.1%}  spreadMAE={mae_spread:.2f}  totalMAE={mae_total:.2f}")
print("\nBy week:")
for w in [2, 3, 4, 5, 6]:
    wr = [r for r in results if r["week"] == w]
    if not wr:
        continue
    b = sum((r["p_home"] - r["actual_home"]) ** 2 for r in wr) / len(wr)
    s = sum(1 for r in wr if (r["p_home"] > 0.5) == (r["actual_home"] == 1.0)) / len(wr)
    print(f"  wk{w}: n={len(wr)} Brier={b:.4f} SU={s:.1%}")

# calibration buckets
print("\nCalibration:")
for lo, hi in [(0, 0.4), (0.4, 0.55), (0.55, 0.7), (0.7, 1.0)]:
    br = [r for r in results if lo <= r["p_home"] < hi]
    if br:
        mp = sum(r["p_home"] for r in br) / len(br)
        ap = sum(r["actual_home"] for r in br) / len(br)
        print(f"  {lo:.2f}-{hi:.2f}: model {mp:.3f} vs actual {ap:.3f} (n={len(br)})")
