"""Touchdown-scorer module.

Method: a team's expected TDs come from the game sim (lambda / ~7 pts per TD).
Each skill player's TD share (season rushing+receiving TDs, small prior so
zero-TD players get a nonzero dart price) sets their per-game TD expectation.

  P(anytime TD) = 1 - exp(-lambda_player)      (Poisson, at least one)
  P(first TD)   = share of all TD expectation x P(any TD is scored)

This mirrors how books price the market: team total -> player shares.
"""
import math

PTS_PER_TD = 7.0
PRIOR_TD = 0.25   # regression prior (TDs) per skill player


def player_expectations(players, team_exp_tds):
    """players: list of {name,pos,team,td,touches,...}.

    Share = 50% TD share + 50% touch share (both with small priors).
    Touch share predicts future TDs better than past TDs alone, which
    stops one hot streak (e.g. a QB's rushing TDs) from breaking the board.
    """
    n = len(players) or 1
    tot_td = sum(p["td"] for p in players)
    tot_touch = sum(p["touches"] for p in players)
    out = []
    for p in players:
        td_share = (p["td"] + 0.5) / (tot_td + 0.5 * n)
        touch_share = (p["touches"] + 5.0) / (tot_touch + 5.0 * n)
        share = 0.5 * td_share + 0.5 * touch_share
        lam = team_exp_tds * share
        out.append({**p, "share": share, "lam": lam,
                    "p_anytime": 1.0 - math.exp(-lam)})
    return out


def fair_odds(p):
    """American odds for probability p (None if p<=0)."""
    if p <= 0:
        return None
    if p >= 1:
        return -10000
    if p > 0.5:
        return int(round(-100 * p / (1 - p)))
    return int(round(100 * (1 - p) / p))


def first_td_probs(all_players):
    """all_players: expectations from both teams (with 'lam')."""
    total_lam = sum(p["lam"] for p in all_players)
    p_any_td = 1.0 - math.exp(-total_lam)
    out = []
    for p in all_players:
        w = p["lam"] / total_lam if total_lam else 0.0
        out.append({**p, "p_first": w * p_any_td})
    return out


def build_td_board(away_players, home_players, lam_a, lam_h, scratches=()):
    """Full anytime + first-TD board for a matchup."""
    exp_a = lam_a / PTS_PER_TD
    exp_h = lam_h / PTS_PER_TD
    ap = [p for p in player_expectations(away_players, exp_a)
          if p["name"] not in scratches]
    hp = [p for p in player_expectations(home_players, exp_h)
          if p["name"] not in scratches]
    board = first_td_probs(ap + hp)
    for p in board:
        p["any_odds"] = fair_odds(p["p_anytime"])
        p["first_odds"] = fair_odds(p["p_first"])
    anytime = sorted(board, key=lambda p: -p["p_anytime"])
    first = sorted(board, key=lambda p: -p["p_first"])
    return anytime, first
