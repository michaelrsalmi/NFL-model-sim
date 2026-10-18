"""Core prediction engine shared by the CLI, the backtester, and the web app.

Walk-forward safe: every rating is built only from games with
week <= max_week, so backtests never peek at the future.
"""
import math

import numpy as np

from fetch import all_games, team_summary, player_tds, PFR_ABBR
from ratings import (regress_offseason, win_prob, adjusted_scoring,
                     expected_points, ELO_INIT, ELO_K, _mov_mult, HOME_PTS,
                     DIV_HFA_SHAVE, is_divisional)
from epa import team_epa, epa_expected_points, team_pace
from td_model import build_td_board

SEASON = 2026
PRIOR_SEASON = 2025
PRIOR_WEEKS = list(range(1, 19))
CUR_WEEKS = list(range(1, 19))
MAX_DONE_WEEK = 5          # last fully completed week (bump as season progresses)

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
            seed=None, include_td=True, refresh=False, use_injuries=True,
            game_time=None):
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
    if max_week >= 1:
        off_e, def_e, lg_e, ppg = team_epa(SEASON, max_week)
        off_e25, def_e25, _, _ = team_epa(PRIOR_SEASON)
    else:
        # week 1: no current-season plays yet -> regressed prior-season EPA
        off_e, def_e, lg_e, ppg = team_epa(PRIOR_SEASON)
        off_e25, def_e25 = {}, {}
    lam_a_epa, lam_h_epa = epa_expected_points(
        away, home, off_e, def_e, lg_e, ppg, lg, off_e25, def_e25, gp)
    # divisional games: familiarity shrinks the home edge (~0.75 pts)
    div = is_divisional(away, home)
    hfa = HOME_PTS - (DIV_HFA_SHAVE if div else 0.0)
    lam_a = min(45.0, max(10.0, 0.5 * (lam_a_srs + lam_a_epa) - hfa))
    lam_h = min(45.0, max(10.0, 0.5 * (lam_h_srs + lam_h_epa) + hfa))

    # injury adjustments (upcoming games only — no historical injury data)
    inj_a, inj_h, det_a, det_h = 0.0, 0.0, [], []
    out_a, out_h = set(), set()
    if use_injuries:
        try:
            from injuries import team_injury_adjustment, fetch_injuries, out_keys
            _inj = fetch_injuries()
            inj_a, det_a = team_injury_adjustment(away, _inj)
            inj_h, det_h = team_injury_adjustment(home, _inj)
            out_a, out_h = out_keys(away, _inj), out_keys(home, _inj)
            lam_a = min(45.0, max(10.0, lam_a - inj_a))
            lam_h = min(45.0, max(10.0, lam_h - inj_h))
        except Exception:
            pass

    # pace adjustment: faster games -> more plays -> more points.
    # Scale the total (not the margin) by a dampened pace factor.
    if max_week >= 1:
        try:
            pace, lg_pace = team_pace(SEASON, max_week)
            pace25, _ = team_pace(PRIOR_SEASON)
            def _pace(t):
                n = gp.get(t, 0)
                cur = pace.get(t, lg_pace)
                prior = pace25.get(t, lg_pace)
                # blend like EPA: current + 40% regressed prior by games played
                w = n / (n + 8.0)
                return w * cur + (1 - w) * (0.6 * prior + 0.4 * lg_pace)
            pr = (_pace(away) + _pace(home)) / 2.0 / lg_pace
            pm = 0.7 + 0.3 * pr
            tot = lam_a + lam_h
            mgn = lam_h - lam_a
            lam_a = min(45.0, max(10.0, (tot * pm - mgn) / 2.0))
            lam_h = min(45.0, max(10.0, (tot * pm + mgn) / 2.0))
        except Exception:
            pass

    # wind: sustained 15+ mph suppresses scoring (~0.17 pts/mph).
    # Outdoor games only; domes excluded in weather.py.
    wind_mph, wind_adj = None, 0.0
    if use_injuries and game_time:
        try:
            from weather import game_wind
            wind_mph, wind_adj = game_wind(home, game_time)
            if wind_adj:
                tot = lam_a + lam_h
                mgn = lam_h - lam_a
                lam_a = min(45.0, max(10.0, (tot - wind_adj - mgn) / 2.0))
                lam_h = min(45.0, max(10.0, (tot - wind_adj + mgn) / 2.0))
        except Exception:
            pass

    n = sims
    # Elo is backward-looking, so shift it by the injury adjustment
    # (~25 Elo per point, the standard NFL conversion) to keep the
    # win prob coherent with the injury-adjusted sims.
    elo_h_adj = elo.get(home, ELO_INIT) - inj_h * 25.0
    elo_a_adj = elo.get(away, ELO_INIT) - inj_a * 25.0
    if div:
        # divisional shave on the Elo-implied home edge too (~25 Elo per point)
        elo_h_adj -= DIV_HFA_SHAVE * 25.0
    p_elo_home = win_prob(elo_h_adj, elo_a_adj, home_team="a")
    # Win probability: blend Elo (well-calibrated) with sim.
    # The projected SCORES below stay pure (scoring model only) — they are
    # the genuine score prediction, shown separately from win probability.
    # In toss-ups the two can disagree; that's honest signal, not a bug.
    sa0 = rng.poisson(lam_a, 2000)
    sh0 = rng.poisson(lam_h, 2000)
    p_sim_home = float(((sh0 > sa0).sum() + 0.5 * (sh0 == sa0).sum()) / 2000)
    p_home = ELO_WEIGHT * p_elo_home + (1 - ELO_WEIGHT) * p_sim_home
    # Calibration shrinkage: walk-forward showed overconfidence in the
    # 55-70% band (model 63% vs actual 38%). Shrink toward 50% to correct.
    p_home = 0.5 + 0.9 * (p_home - 0.5)
    # Final sims for score/spread/total (pure scoring model).
    sa = rng.poisson(lam_a, n)
    sh = rng.poisson(lam_h, n)

    res = {
        "away": away, "home": home, "sims": n,
        "p_home": p_home, "p_away": 1 - p_home,
        "p_elo_home": p_elo_home,
        "proj_away": float(sa.mean()), "proj_home": float(sh.mean()),
        "fair_spread": float(sh.mean() - sa.mean()),
        "fair_total": float(sh.mean() + sa.mean()),
        "lam_away": lam_a, "lam_home": lam_h,
        "elo_home": elo.get(home, ELO_INIT),
        "elo_away": elo.get(away, ELO_INIT),
        "divisional": div,
        "wind_mph": wind_mph, "wind_adj": wind_adj,
        # do the scoring model and win prob agree on the winner?
        "split": ((float(sa.mean()) > float(sh.mean())) != (p_home < 0.5)
                  and abs(p_home - 0.5) < 0.08),
        "inj_adj": [inj_a, inj_h],
        "injuries": {"away": det_a, "home": det_h},
    }
    if include_td:
        try:
            from injuries import _key as _ikey
            ap_ = player_tds(away, SEASON, refresh=refresh)
            hp_ = player_tds(home, SEASON, refresh=refresh)
            ap_ = [p for p in ap_ if _ikey(p["name"]) not in out_a]
            hp_ = [p for p in hp_ if _ikey(p["name"]) not in out_h]
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
