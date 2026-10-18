"""Ratings engine.

Two complementary systems, mirroring the best public simulators:

1. ELO (FiveThirtyEight methodology, parameters validated out-of-sample
   2010-2025: K=40, home edge 42.2, 40% regression to mean each offseason).
   -> win probability, the backbone of the sim.

2. Adjusted scoring ratings (SRS-style): points scored/allowed per game,
   iteratively adjusted for opponent strength.
   -> expected points (lambda) for each team.
"""
import math

ELO_INIT = 1500.0
ELO_MEAN = 1503.1
ELO_K = 40.0
ELO_HFA = 42.2          # Elo points of home-field edge
ELO_REGRESS = 0.40      # offseason regression toward mean
HOME_PTS = 1.0          # points added to home team expectation (2.0 total edge)
DIV_HFA_SHAVE = 0.75    # divisional games: home edge is ~0.75 pts smaller
                        # (Greer: 1.36-pt gap; 0.75 is the conservative choice)

DIVISIONS = {
    "BUF": "AFCE", "MIA": "AFCE", "NE": "AFCE", "NYJ": "AFCE",
    "BAL": "AFCN", "CIN": "AFCN", "CLE": "AFCN", "PIT": "AFCN",
    "HOU": "AFCS", "IND": "AFCS", "JAX": "AFCS", "TEN": "AFCS",
    "DEN": "AFCW", "KC": "AFCW", "LAC": "AFCW", "LV": "AFCW",
    "DAL": "NFCE", "NYG": "NFCE", "PHI": "NFCE", "WSH": "NFCE",
    "CHI": "NFCN", "DET": "NFCN", "GB": "NFCN", "MIN": "NFCN",
    "ATL": "NFCS", "CAR": "NFCS", "NO": "NFCS", "TB": "NFCS",
    "ARI": "NFCW", "LAR": "NFCW", "SEA": "NFCW", "SF": "NFCW",
}

def is_divisional(away, home):
    return DIVISIONS.get(away.upper()) == DIVISIONS.get(home.upper())


def win_prob(elo_a, elo_b, home_team="a"):
    """P(A beats B) from Elo diff, 400-point scale, home edge applied."""
    diff = elo_a - elo_b
    if home_team == "a":
        diff += ELO_HFA
    elif home_team == "b":
        diff -= ELO_HFA
    return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))


def _mov_mult(margin, elo_diff):
    # FiveThirtyEight margin-of-victory multiplier
    return math.log(margin + 1.0) * 2.2 / (abs(elo_diff) * 0.001 + 2.2)


def run_elo(games):
    """games: list of dicts(away, home, as, hs, done). Returns {abbr: elo}."""
    elo = {}

    def get(t):
        return elo.setdefault(t, ELO_INIT)

    for g in games:
        if not g["done"]:
            continue
        a, h = g["away"], g["home"]
        ea, eh = get(a), get(h)
        pa = win_prob(ea, eh, home_team="b")
        if g["as"] == g["hs"]:
            sa = 0.5
        else:
            sa = 1.0 if g["as"] > g["hs"] else 0.0
        margin = abs(g["as"] - g["hs"])
        k = ELO_K * _mov_mult(margin, ea - eh)
        shift = k * (sa - pa)
        elo[a] = ea + shift
        elo[h] = eh - shift
    return elo


def regress_offseason(elo):
    return {t: ELO_MEAN + (e - ELO_MEAN) * (1.0 - ELO_REGRESS) for t, e in elo.items()}


def adjusted_scoring(games):
    """Iterative opponent-adjusted points for/against per game.

    Returns (off, deff, lg_avg): off[t] ~ points above avg scored,
    deff[t] ~ points above avg allowed (negative = good defense).
    """
    teams = sorted({g["away"] for g in games} | {g["home"] for g in games})
    done = [g for g in games if g["done"]]
    if not done:
        return {}, {}, 22.0
    total_pts = sum(g["as"] + g["hs"] for g in done)
    total_tm = sum(2 for _ in done)
    lg = total_pts / total_tm if total_tm else 22.0

    off = {t: 0.0 for t in teams}
    deff = {t: 0.0 for t in teams}
    for _ in range(50):
        noff, ndeff = {}, {}
        for t in teams:
            s_for, s_ag, n = 0.0, 0.0, 0
            for g in done:
                if g["away"] == t:
                    s_for += g["as"] - deff[g["home"]]
                    s_ag += g["hs"] - off[g["home"]]
                    n += 1
                elif g["home"] == t:
                    s_for += g["hs"] - deff[g["away"]]
                    s_ag += g["as"] - off[g["away"]]
                    n += 1
            noff[t] = (s_for / n - lg) if n else 0.0
            ndeff[t] = (s_ag / n - lg) if n else 0.0
        if max(abs(noff[t] - off[t]) for t in teams) < 1e-9:
            off, deff = noff, ndeff
            break
        off, deff = noff, ndeff
    return off, deff, lg


def expected_points(away, home, off, deff, lg, games_played):
    """Points-based expected points (SRS-style) for a matchup.

    Returns raw lambdas WITHOUT home-field (applied once by the caller,
    since the EPA-based lambda is blended in too).
    """
    def shrink(t, v):
        n = games_played.get(t, 4)
        return v * n / (n + 8.0)

    oa = shrink(away, off.get(away, 0.0))
    da = shrink(away, deff.get(away, 0.0))
    oh = shrink(home, off.get(home, 0.0))
    dh = shrink(home, deff.get(home, 0.0))
    # deff = points ABOVE avg allowed (positive = bad defense), so a bad
    # opposing defense ADDS to your expected score.
    la = min(45.0, max(10.0, lg + oa + dh))
    lh = min(45.0, max(10.0, lg + oh + da))
    return la, lh
