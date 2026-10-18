"""Play-by-play EPA ratings (nflverse data).

The gold-standard input the best public sims use: EPA per play is more
stable and predictive than raw points, especially early in the season.
"""
import os
from functools import lru_cache
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")

PLAYS = ("pass", "run")   # real scrimmage plays only


@lru_cache(maxsize=4)
def load_pbp(season):
    path = os.path.join(DATA, f"pbp_{season}.csv")
    if not os.path.exists(path):
        return None
    d = pd.read_csv(path, low_memory=True,
                    usecols=["game_id", "posteam", "defteam", "epa",
                             "play_type", "season_type", "week"])
    d = d[d.season_type == "REG"]
    d = d[d.play_type.isin(PLAYS) & d.epa.notna()]
    d = d[d.posteam.notna() & d.defteam.notna()]
    return d


# nflverse abbr -> ESPN abbr (nflverse calls the Rams "LA")
NV_TO_ESPN = {"LA": "LAR"}


def team_epa(season, max_week=None):
    """Returns (off_epa_pp, def_epa_pp, lg_epa_pp, plays_per_game).

    Keys are ESPN abbreviations. max_week filters to weeks <= max_week
    (for walk-forward backtesting).
    """
    d = load_pbp(season)
    if d is None or len(d) == 0:
        return {}, {}, 0.0, 63.0
    if max_week is not None:
        d = d[d.week <= max_week]
    if len(d) == 0:
        return {}, {}, 0.0, 63.0
    off = {NV_TO_ESPN.get(k, k): v
           for k, v in d.groupby("posteam").epa.mean().to_dict().items()}
    deff = {NV_TO_ESPN.get(k, k): v
            for k, v in d.groupby("defteam").epa.mean().to_dict().items()}
    lg = d.epa.mean()
    # plays per team-game
    g = d.groupby(["game_id", "posteam"]).size().reset_index()
    ppg = g[0].mean() if len(g) else 63.0
    return off, deff, float(lg), float(ppg)


def team_pace(season, max_week=None):
    """Returns (pace_dict, lg_pace): plays per game by team (ESPN abbr).

    Pace is a real driver of totals; blend with prior season like EPA.
    """
    d = load_pbp(season)
    if d is None or len(d) == 0:
        return {}, 63.0
    if max_week is not None:
        d = d[d.week <= max_week]
    if len(d) == 0:
        return {}, 63.0
    g = d.groupby(["game_id", "posteam"]).size().reset_index()
    g["tm"] = g["posteam"].map(lambda k: NV_TO_ESPN.get(k, k))
    pace = g.groupby("tm")[0].mean().to_dict()
    return pace, float(g[0].mean())


def epa_expected_points(away, home, off, deff, lg_epa, ppg, lg_pts,
                        off25=None, deff25=None, gp=None):
    """EPA-implied expected points for a matchup.

    edge (pts/game) = (off_epa_pp - lg) * plays  -> offense
    defense reduces opponent expectation when it allows less EPA than avg.
    Prior season (regressed 40%) blended by games played, same as the
    points-based ratings.
    """
    off25 = off25 or {}
    deff25 = deff25 or {}
    gp = gp or {}

    def blend(cur, prior, t):
        n = gp.get(t, 4)
        w = n / (n + 10.0)
        return w * cur.get(t, lg_epa) + (1 - w) * (
            lg_epa + (prior.get(t, lg_epa) - lg_epa) * 0.6)

    oa = blend(off, off25, away)
    da = blend(deff, deff25, away)
    oh = blend(off, off25, home)
    dh = blend(deff, deff25, home)
    # points above average each unit contributes per game.
    # deff = EPA/play ALLOWED (higher = worse defense), so facing a bad
    # defense adds (dh - lg_epa) * plays to your expected score.
    la = lg_pts + (oa - lg_epa) * ppg + (dh - lg_epa) * ppg
    lh = lg_pts + (oh - lg_epa) * ppg + (da - lg_epa) * ppg
    return la, lh
