"""Data layer: fetch + cache + parse ESPN scoreboard data and PFR team pages.

Sources (all free, no API key):
  - ESPN site API: full season schedules/scores (used for Elo ratings)
  - Pro-Football-Reference team pages: team stats, per-game EPA, player
    rushing/receiving TDs (used for expected points + TD shares)

Everything is cached under data/cache/ so re-runs are instant.
Use --refresh on the CLI to force fresh downloads.
"""
import json
import os
import re
import time
import urllib.request
from html.parser import HTMLParser

BASE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(BASE, "data", "cache")
os.makedirs(CACHE, exist_ok=True)

ESPN = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"

# ESPN abbr -> PFR abbr
PFR_ABBR = {
    "ARI": "crd", "ATL": "atl", "BAL": "rav", "BUF": "buf", "CAR": "car",
    "CHI": "chi", "CIN": "cin", "CLE": "cle", "DAL": "dal", "DEN": "den",
    "DET": "det", "GB": "gnb", "HOU": "htx", "IND": "clt", "JAX": "jax",
    "KC": "kan", "LV": "rai", "LAC": "sdg", "LAR": "ram", "MIA": "mia",
    "MIN": "min", "NE": "nwe", "NO": "nor", "NYG": "nyg", "NYJ": "nyj",
    "PHI": "phi", "PIT": "pit", "SEA": "sea", "SF": "sfo", "TB": "tam",
    "TEN": "oti", "WAS": "was",
}

HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) nfl-model/1.0"}


def _get(url, cache_name, max_age_hours=24, refresh=False):
    path = os.path.join(CACHE, cache_name)
    if not refresh and os.path.exists(path):
        age = (time.time() - os.path.getmtime(path)) / 3600
        if age < max_age_hours:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as r:
        text = r.read().decode("utf-8", "replace")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return text


# ---------------------------------------------------------------- ESPN
def espn_json(url, cache_name, refresh=False):
    return json.loads(_get(url, cache_name, refresh=refresh))


def scoreboard_games(season, week, refresh=False):
    """All games for a season/week. Each -> dict(away, home, as_, hs, done)."""
    url = f"{ESPN}/scoreboard?dates={season}&seasontype=2&week={week}"
    d = espn_json(url, f"espn_sb_{season}_w{week}.json", refresh=refresh)
    out = []
    for ev in d.get("events", []):
        comp = ev["competitions"][0]
        teams = {t["homeAway"]: t for t in comp["competitors"]}
        a, h = teams["away"], teams["home"]
        out.append({
            "away": a["team"]["abbreviation"],
            "home": h["team"]["abbreviation"],
            "as": int(a.get("score", 0)),
            "hs": int(h.get("score", 0)),
            "done": comp["status"]["type"].get("completed", False),
            "date": ev.get("date", ""),
            "week": week,
            "season": season,
        })
    return out


def all_games(season, weeks, refresh=False):
    games = []
    for w in weeks:
        try:
            games.extend(scoreboard_games(season, w, refresh=refresh))
        except Exception as e:
            print(f"  [warn] week {w} fetch failed: {e}")
    return games


# ---------------------------------------------------------------- PFR parsing
class _TableParser(HTMLParser):
    """Extract rows of one table by id as list of {data-stat: text} dicts."""

    def __init__(self, table_id):
        super().__init__()
        self.table_id = table_id
        self.in_table = False
        self.in_row = False
        self.in_cell = False
        self.cell_stat = None
        self.cell_text = []
        self.row = {}
        self.rows = []
        self.depth = 0

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "table" and a.get("id") == self.table_id:
            self.in_table = True
            self.depth = 1
            return
        if not self.in_table:
            return
        if tag == "table":
            self.depth += 1
        if self.depth > 1:
            return
        if tag == "tr":
            self.in_row = True
            self.row = {}
        elif tag in ("td", "th") and self.in_row:
            self.in_cell = True
            self.cell_stat = a.get("data-stat")
            self.cell_text = []

    def handle_endtag(self, tag):
        if not self.in_table:
            return
        if tag == "table":
            self.depth -= 1
            if self.depth == 0:
                self.in_table = False
            return
        if self.depth > 1:
            return
        if tag in ("td", "th") and self.in_cell:
            if self.cell_stat:
                self.row[self.cell_stat] = "".join(self.cell_text).strip()
            self.in_cell = False
        elif tag == "tr" and self.in_row:
            if self.row:
                self.rows.append(self.row)
            self.in_row = False

    def handle_data(self, data):
        if self.in_table and self.in_cell and self.depth == 1:
            self.cell_text.append(data)


def pfr_table(pfr_abbr, season, table_id, refresh=False):
    url = f"https://www.pro-football-reference.com/teams/{pfr_abbr}/{season}.htm"
    html = _get(url, f"pfr_{pfr_abbr}_{season}.html",
                max_age_hours=12, refresh=refresh)
    p = _TableParser(table_id)
    p.feed(html)
    return p.rows


def _f(x, default=0.0):
    try:
        return float(str(x).replace(",", ""))
    except (ValueError, TypeError):
        return default


def _rows_text(pfr_abbr, season, table_id, refresh=False):
    """Positional rows: list of cell-text lists (for tables without data-stats)."""
    url = f"https://www.pro-football-reference.com/teams/{pfr_abbr}/{season}.htm"
    html = _get(url, f"pfr_{pfr_abbr}_{season}.html",
                max_age_hours=12, refresh=refresh)

    class P(HTMLParser):
        def __init__(self):
            super().__init__()
            self.in_t = False
            self.in_row = False
            self.in_cell = False
            self.txt = []
            self.row = []
            self.rows = []

        def handle_starttag(self, tag, attrs):
            if tag == "table" and dict(attrs).get("id") == table_id:
                self.in_t = True
            elif self.in_t and tag == "tr":
                self.in_row = True
                self.row = []
            elif self.in_t and self.in_row and tag in ("td", "th"):
                self.in_cell = True
                self.txt = []

        def handle_endtag(self, tag):
            if tag == "table" and self.in_t:
                self.in_t = False
            elif self.in_t and tag in ("td", "th") and self.in_cell:
                self.row.append("".join(self.txt).strip())
                self.in_cell = False
            elif self.in_t and tag == "tr" and self.in_row:
                if any(c for c in self.row):
                    self.rows.append(self.row)
                self.in_row = False

        def handle_data(self, data):
            if self.in_cell:
                self.txt.append(data)

    p = P()
    p.feed(html)
    return p.rows


def team_summary(espn_abbr, season, refresh=False):
    """Team-level stats from PFR: points, EPA/game, red zone, games played."""
    pfr = PFR_ABBR[espn_abbr]
    tstats = {r.get("player", ""): r for r in pfr_table(pfr, season, "team_stats", refresh)}
    team = tstats.get("Team Stats", {})
    opp = tstats.get("Opp. Stats", {})
    games = [r for r in pfr_table(pfr, season, "games", refresh)
             if r.get("game_outcome") in ("W", "L", "T")]
    n = len(games) or 1
    off_epa = sum(_f(r.get("exp_pts_off")) for r in games) / n
    def_epa = sum(_f(r.get("exp_pts_def")) for r in games) / n
    conv = {r.get("player", ""): r for r in pfr_table(pfr, season, "team_conversions", refresh)}
    tconv = conv.get("Team Stats", {})
    rz_att = _f(tconv.get("red_zone_att"))
    rz_td = _f(tconv.get("red_zone_scores"))
    pf = _f(team.get("points"))
    pa = _f(opp.get("points"))
    return {
        "abbr": espn_abbr,
        "games": len(games),
        "pf": pf,
        "pa": pa,
        "pf_pg": pf / n,
        "pa_pg": pa / n,
        "off_epa_pg": off_epa,
        "def_epa_pg": def_epa,          # negative = good defense
        "rz_td_rate": (rz_td / rz_att) if rz_att else 0.0,
    }


def player_tds(espn_abbr, season, refresh=False):
    """Skill-player TD shares from PFR rushing+receiving table.

    Column positions (0-based): 1=name, 3=pos, 8=rush TD, 19=rec TD,
    32=total rush+rec TD, 29=touches, 15=targets.
    """
    rows = _rows_text(PFR_ABBR[espn_abbr], season, "rushing_and_receiving",
                      refresh)
    players = []
    for c in rows:
        if len(c) < 33 or not c[0].isdigit():
            continue
        name = c[1]
        if not name or "Team" in name:
            continue
        rush_td = _f(c[8])
        rec_td = _f(c[19])
        players.append({
            "name": name,
            "pos": c[3],
            "team": espn_abbr,
            "rush_td": rush_td,
            "rec_td": rec_td,
            "td": _f(c[32]) or (rush_td + rec_td),
            "touches": _f(c[29]),
            "targets": _f(c[15]),
        })
    return players
