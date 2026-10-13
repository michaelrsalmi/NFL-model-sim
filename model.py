"""Core prediction engine shared by the CLI, the backtester, and the web app.

Walk-forward safe: every rating is built only from games with
week <= max_week, so backtests never peek at the future.
"""
import math

import numpy as np

from fetch import all_games, team_summary, player_tds, PFR_ABBR
from ratings import (regress_offseason, win_prob, adjusted_scoring,
                     expected_points, ELO_INIT, ELO_K, _mov_mult, HOME_PTS)
from epa import team_epa, epa_expected_points
from td_model import build_td_board

SEASON = 2026
PRIOR_SEASON = 2025
PRIOR_WEEKS = list(range(1, 19))
CUR_WEEKS = list(range(1, 19))
MAX_DONE_WEEK = 4          # weeks fully completed (week 5 in progress)

# 2/3 Elo + 1/3 score-sim: Elo is the best-validated public signal,
# the sim adds matchup-specific scoring info.
ELO_WEIGHT = 2.0 / 3.0


class SeasonData:
    def __init__(self, refresh=False):
        self.prior = [g for g in all_games(PRIOR_SEASON, PRIOR_WEEKS,
                                          refresh=refresh) if g["done"]]
        self.cur = all_games(SEASON, CUR_WEEKS, refresh=refresh)

    def done_through(self, max_week):
        return [g for g in self.cur if g["done"] and g["week"] <= max_week]


def _run_elo(games):
    elo = {}
    for g in games:
        a, h = g["away"], g["home"]
        ea, eh = elo.get(a, ELO_INIT), elo.get(h, ELO_INIT)
        pa = win_prob(ea, eh, home_team="b")
        sa = 0.5 if g["as"] == g["hs"] else (1.0 if g["as"] > g["hs"] else 0.0)
        k = ELO_K * _mov_mult(abs(g["as"] - g["hs"]), ea - eh)
        sh = k * (sa - pa)
        elo[a], elo[h] = ea + sh, eh - sh
    return elo


def _blended_scoring(cur_done, prior):
    off, deff, lg = adjusted_scoring(cur_done)
    off25, deff25, _ = adjusted_scoring(prior)
    off25 = {t: v * 0.6 for t, v in off25.items()}
    deff25 = {t: v * 0.6 for t, v in deff25.items()}
    gp = {}
    for g in cur_done:
        gp[g["away"]] = gp.get(g["away"], 0) + 1
        gp[g["home"]] = gp.get(g["home"], 0) + 1
    off_b, deff_b = {}, {}
    for t in set(off) | set(off25):
        n = gp.get(t, 0)
        w = n / (n + 10.0)
        off_b[t] = w * off.get(t, 0.0) + (1 - w) * off25.get(t, 0.0)
        deff_b[t] = w * deff.get(t, 0.0) + (1 - w) * deff25.get(t, 0.0)
    return off_b, deff_b, lg, gp


def predict(away, home, data, max_week=MAX_DONE_WEEK, sims=20000,
            seed=None, include_td=True, refresh=False):
    """Full prediction for one matchup. Returns a results dict."""
    away, home = away.upper(), home.upper()
    rng = np.random.default_rng(seed)
    cur_done = data.done_through(max_week)

    elo = _run_elo(data.prior)
    elo = regress_offseason(elo)
    for g in cur_done:
        a, h = g["away"], g["home"]
        ea, eh = elo.get(a, ELO_INIT), elo.get(h, ELO_INIT)
        pa = win_prob(ea, eh, home_team="b")
        sa = 0.5 if g["as"] == g["hs"] else (1.0 if g["as"] > g["hs"] else 0.0)
        k = ELO_K * _mov_mult(abs(g["as"] - g["hs"]), ea - eh)
        sh = k * (sa - pa)
        elo[a], elo[h] = ea + sh, eh - sh

    off_b, deff_b, lg, gp = _blended_scoring(cur_done, data.prior)
    lam_a_srs, lam_h_srs = expected_points(away, home, off_b, deff_b, lg, gp)
    off_e, def_e, lg_e, ppg = team_epa(SEASON, max_week)
    off_e25, def_e25, _, _ = team_epa(PRIOR_SEASON)
    lam_a_epa, lam_h_epa = epa_expected_points(
        away, home, off_e, def_e, lg_e, ppg, lg, off_e25, def_e25, gp)
    lam_a = min(45.0, max(10.0, 0.5 * (lam_a_srs + lam_a_epa) - HOME_PTS))
    lam_h = min(45.0, max(10.0, 0.5 * (lam_h_srs + lam_h_epa) + HOME_PTS))

    n = sims
    sa = rng.poisson(lam_a, n)
    sh = rng.poisson(lam_h, n)
    home_w = int((sh > sa).sum())
    away_w = int((sa > sh).sum())
    ties = n - home_w - away_w
    p_sim_home = (home_w + 0.5 * ties) / n
    p_elo_home = win_prob(elo.get(home, ELO_INIT), elo.get(away, ELO_INIT),
                          home_team="a")
    p_home = ELO_WEIGHT * p_elo_home + (1 - ELO_WEIGHT) * p_sim_home

    res = {
        "away": away, "home": home, "sims": n,
        "p_home": p_home, "p_away": 1 - p_home,
        "p_sim_home": p_sim_home, "p_elo_home": p_elo_home,
        "proj_away": float(sa.mean()), "proj_home": float(sh.mean()),
        "fair_spread": float(sh.mean() - sa.mean()),
        "fair_total": float(sh.mean() + sa.mean()),
        "lam_away": lam_a, "lam_home": lam_h,
        "elo_home": elo.get(home, ELO_INIT),
        "elo_away": elo.get(away, ELO_INIT),
    }
    if include_td:
        try:
            ap_ = player_tds(away, SEASON, refresh=refresh)
            hp_ = player_tds(home, SEASON, refresh=refresh)
            anytime, first = build_td_board(ap_, hp_, lam_a, lam_h)
            res["anytime"] = [
                {"name": p["name"], "team": p["team"],
                 "prob": p["p_anytime"], "odds": p["any_odds"]}
                for p in anytime[:12]]
            res["first_td"] = [
                {"name": p["name"], "team": p["team"],
                 "prob": p["p_first"], "odds": p["first_odds"]}
                for p in first[:8]]
        except Exception:
            res["anytime"], res["first_td"] = [], []
    return res
