# NFL Monte Carlo Simulator + TD-Scorer Model

Your own runnable version of the "model sim" concept: simulates any NFL
matchup thousands of times and reports win probability, projected score,
fair spread/total, and anytime/first-TD probabilities with fair odds.

## Run it

```bash
cd ~/workspace/nfl-model
python3 simulate.py --away BUF --home LAR                 # tonight's game
python3 simulate.py --away BUF --home LAR --sims 50000    # tighter numbers
python3 simulate.py --away KC --home PHI --line -3 --total 48.5
python3 simulate.py --away BUF --home LAR --out "Puka Nacua"   # scratch a player
python3 simulate.py --away BUF --home LAR --refresh       # force fresh data
python3 simulate.py --away BUF --home LAR --seed 7        # reproducible run
```

Each run executes N full-game simulations (default 20,000) and aggregates
them, so a single run already gives the stable averaged result. Re-running
moves the numbers by ~+/-0.3%.

## What it outputs

- **Win probability** -- blended 2/3 Elo + 1/3 score simulation
- **Projected score, fair spread, fair total**
- **Cover / over-under probability** vs any line you pass in
- **Anytime-TD board** (top 12) and **first-TD board** (top 8) with fair
  American odds, built from each team's expected TDs x player TD/touch shares

## How it works (built from the best public simulators' validated ideas)

1. **Elo ratings** (FiveThirtyEight methodology): K=40, 42.2-pt home edge,
   margin-of-victory multiplier, 40% offseason regression. Run over the full
   prior season, regressed, then the current season to date. Parameters were
   validated out-of-sample on 2010-2025 (Brier ~0.22, ~64% accuracy).
2. **Scoring ratings** (SRS-style): points scored/allowed per game, adjusted
   for opponent strength over 50 iterations, blended with last season's
   regressed ratings (a 4-game sample never dominates) and shrunk toward
   league average.
3. **EPA ratings** (nflverse play-by-play, the gold standard): EPA per play
   for every team's offense and defense, converted to points via plays per
   game, blended with last season's regressed EPA the same way. This is the
   same input the most accurate public sims are built on.
4. **Expected points** = 50% points-based rating + 50% EPA-implied, +/-1.0
   for home field.
5. **Monte Carlo**: each team's score ~ Poisson(expected points); 20k draws
   give win%, spread, total. Ties split 50/50.
6. **TD module**: team expected TDs (lambda/7) x player share (50% TD share +
   50% touch share, both regressed). Anytime = 1 - e^-lambda; first TD =
   share of total TD expectation.

## Data (all free, no API key, cached in data/cache/)

- ESPN API: schedules + scores (Elo + scoring ratings)
- Pro-Football-Reference: team stats, per-game EPA, player rushing/receiving

## Honest limitations

- No injury feed: use `--out "Player Name"` to scratch someone manually.
- No weather adjustment (dome/good-weather games unaffected anyway).
- Poisson slightly overstates tie probability; win% splits ties evenly.
- Like every public model, it won't beat the closing line consistently --
  use it to find your own number, then compare against the book's.
