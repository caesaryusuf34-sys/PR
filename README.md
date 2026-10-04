# sportsai — a self-learning sports prediction system

Ask it `Predict Miami vs Clemson` and it will:

1. **update itself**: ingest new schedules and results, collect the outcomes of its own earlier
   predictions, score them, and run a learning cycle if one is due;
2. **load the latest validated model** (the *champion*) from the versioned registry;
3. **collect current pre-game data** and rebuild features **as of the moment of prediction** —
   only results and box scores of games that were already over are visible;
4. **predict** win probabilities, margin, total and projected score, and **explain** them;
5. **log** the prediction (with every input feature, the data cutoff, the model version and a
   SHA-256 snapshot hash) in an append-only database, to be graded once the game is played.

You never retrain it by hand: every completed game becomes training and evaluation data, and the
learner only replaces the champion when an out-of-sample, statistically gated comparison says it should.

```
NEW QUERY ─► pre-game data ─► point-in-time features ─► champion model ─► prediction + explanation
                                                                              │ (immutable log)
     ┌──────────────────────── next prediction uses the new champion ◄───┐    ▼
     │                                                                   │  game ends
     ▼                                                                   │    ▼
  model registry  ◄── promote / refit ◄── promotion gates ◄── walk-forward│ result collection
  (v1.0, v1.1, v2.0…)                    (effect size,       validation   │    ▼
                                          bootstrap p,       of candidates│ evaluation (Brier, log loss,
                                          Brier, ECE, n)          ▲       │ calibration, MAE)
                                                                  │       │    ▼
                                         diagnosis-driven candidates ◄── error analysis (biases,
                                         (calibration, stacking,          calibration, residual
                                          features, members, params)      signal, importance, drift)
```

## Quick start

```bash
pip install -r requirements.txt
python -m sportsai bootstrap                 # build the database (ESPN 2015→now) and model v1.0
python -m sportsai predict "Ohio State at Michigan"
python -m sportsai slate --date 20261010     # every unstarted game on a date
python -m sportsai update                    # sync → collect results → evaluate → learn if due
python -m sportsai status                    # champion, version history, live record, pending picks
python -m sportsai history                   # graded predictions
python -m sportsai audit                     # leakage audit of logged predictions
python -m sportsai learn --force             # run a learning cycle now
python -m sportsai daemon --every-hours 6    # keep the loop running unattended
```

`predict` runs `update` first, so a learning cycle triggers automatically whenever enough new
games have finished. For unattended operation, see `deploy/cron.example` or the GitHub Actions
template `deploy/github-actions-sportsai.yml`. Both commit `store/` after each cycle.

Query syntax: `A vs B` means the scheduled game between them. If no game is scheduled in the next 21 days, it
becomes a hypothetical neutral-site matchup. `A @ B` / `A at B` means a hypothetical with B at home.
Hypotheticals are logged but never graded.

## Architecture

```
sportsai/
  config.py                 storage paths + LearningPolicy (all thresholds in one place)
  cli.py, __main__.py       command-line interface
  core/                     ── sport-agnostic ──
    db.py                   SQLite store; pre-game vs post-game tables; immutable predictions
    leakage.py              TemporalGuard (availability rule) + snapshot hashing
    featurestore.py         cached point-in-time training features, auto-invalidated
    engine.py               query → predict → explain → log; results; evaluation; auto_update
    analysis.py             error analysis (calibration, biases, residual signal, importance, drift)
    learner.py              candidates → walk-forward → promotion gates → versioning → reports
    registry.py             model versions on disk + metadata/status in the database
    metrics.py              accuracy, Brier, log loss, ECE, calibration slope, MAE, BH correction
    audit.py                re-derives logged features at their cutoff to prove no leakage
    legacy.py               imports the original 2026-10-03 slate predictions (commit 82c7a54)
    sport.py                SportAdapter / FeatureBuilder interfaces + adapter registry
  models/ensemble.py        Elo, adjusted-scoring, logistic, LightGBM, margin ridge, GBM margin;
                            simplex or stacked ensemble; none/Platt/isotonic calibration
  sports/ncaaf/             ── NCAA football adapter ──
    espn.py, parse.py       ESPN public JSON API → rows (schedule / results / box score / plays)
    stats.py                expected-points model (versioned) → per team-game efficiency stats
    ratings.py              MOV Elo engine + opponent-adjusted ridge ratings
    features.py             point-in-time feature builder
    adapter.py              ingestion, team resolution, game lookup, explanations, subgroups
store/                      PERSISTENT MEMORY (commit it): sportsai.db, models/<sport>/<version>/,
                            reports/<sport>/<learning run>.md|json, ncaaf_ep_model.json
cache/                      rebuildable: raw ESPN summaries, scoreboards, feature cache
tests/                      unit tests + leakage integration tests
src/, *.csv, PREDICTIONS.md the original one-off pipeline and its 2026-10-03 deliverables (v0-legacy)
```

### Historical database (`store/sportsai.db`)

| table | when the information exists | content |
|---|---|---|
| `games`, `teams` | pre-game | schedule, venue, neutral site, conference game. **Scores are rejected here.** |
| `pregame_context` | pre-game, timestamped | forecast, records, ranks captured before kickoff |
| `predictions` | pre-game, **append-only** | p(home), margin, total, projected score, confidence, member predictions, all input features, feature names, explanation, `created_utc`, `data_cutoff_utc`, `kickoff_utc`, `is_blind`, origin, model + feature version, SHA-256 snapshot hash |
| `results` | post-game | final scores + collection timestamp |
| `team_game_stats` | post-game | EPA, success rate, explosiveness, rush/pass splits, sack and turnover rates, drives, red zone, field position, FG, box score, starting QB |
| `evaluations` | derived | per-prediction correctness, Brier, log loss, margin/total error, high-confidence miss |
| `prediction_annotations` | append-only | invalidation notes (a bad prediction is flagged, never edited) |
| `model_versions` | — | version, parent, status, kind, config, features, training cutoff, n, validation metrics, reason, artifact, code revision |
| `learning_runs` | — | every cycle: trigger, decision, reason, candidates, tests, report path |

## Leakage prevention: before vs after kickoff

* **Availability rule.** A result or box score from game X becomes visible at
  `kickoff(X) + 4.5 h`. It can feed a prediction only if that time is ≤ the prediction's cutoff, and
  the cutoff is ≤ kickoff. Every feature builder takes an explicit cutoff and reads data only through
  `TemporalGuard`.
* **Schema separation.** Pre-game tables cannot hold scores (enforced in code). Post-game tables are
  joined only at evaluation time.
* **Immutable, hashed predictions.** SQLite triggers abort any `UPDATE` or `DELETE` on
  `predictions`. A CHECK constraint rejects `data_cutoff > kickoff`. `is_blind = 1` only when the
  prediction was written before kickoff. A request after kickoff is logged as a non-blind `backtest`
  with a cutoff at kickoff.
* **Truncation-invariance tests.** Features for a game are identical whether or not everything after
  its cutoff has been deleted from the data (`tests/test_leakage_ncaaf.py`).
* **Audit.** `python -m sportsai audit` rebuilds each logged prediction's features at its stored
  cutoff, using today's database (which now includes the game's own result), and compares them with
  the frozen snapshot and hash.
* **Walk-forward only.** Every model comparison trains strictly on games that finished before the
  test block starts. There are no random splits.
* Betting lines are never model inputs. They are stored only as a labelled reference in `context_json`.

## How the system learns (and how it avoids fooling itself)

A learning cycle runs when ≥ `min_new_games` (100) games have completed since the champion was trained.
It also runs when the champion's blind live record is significantly worse than its validated
expectation (z > 3, at least 150 games).

1. **Data.** Point-in-time features for every completed game since 2022, with the newest games included.
2. **Champion walk-forward.** The champion's configuration is re-estimated before each of 6 time
   blocks and predicts that block. That gives 3,000 out-of-sample predictions.
3. **Error analysis** on that large sample, plus the live record:
   * calibration table, ECE, calibration slope ± SE (over- or under-confidence);
   * high-confidence misses, observed vs expected;
   * bias in about 12 subgroups (home/road favourites, FBS-vs-FCS, early season, QB changes, big
     favourites, toss-ups, neutral site…), with Benjamini–Hochberg correction;
   * features still correlated with the residuals (signal the model misses);
   * permutation importance on the newest block (features with no value);
   * member performance and drift.
4. **Candidates.** At most 8 per cycle, each tied to a finding:
   * a recalibration (Platt or isotonic);
   * a stacked meta-learner given the biased subgroups' indicators;
   * extra GBM capacity for residual signal;
   * dropping useless features;
   * adding residual-correlated features;
   * dropping dead ensemble members;
   * recency weighting when drift is detected;
   * one rotating local hyper-parameter move.
5. **Two-stage test.** Candidates are ranked on the older *selection* blocks. The winner is then
   compared with the champion on the newer *confirmation* blocks, which played no part in choosing
   it. Promotion requires all of:
   * a mean log-loss gain ≥ 0.002;
   * a one-sided block-bootstrap (by calendar week) p < 0.05;
   * Brier no worse;
   * ECE no worse than +0.01;
   * at least 300 games.
6. **Decision.**
   * *structural*: the candidate passes every gate, gets a major version and becomes champion;
   * *refit*: otherwise, the validated configuration is re-estimated on all data and gets a minor
     version;
   * *no change*: not enough new data.

   Every version keeps its artifact, configuration, validation metrics and the reason it was created.
   The full reasoning goes to `store/reports/<sport>/`.

The system therefore learns *statistically*. One upset never changes the model. A pattern only does
when it shows up across thousands of out-of-sample games and survives an independent confirmation
window. Individual misses are listed in the report as context, never fitted.

## NFL

`--sport nfl` uses the same engine. ESPN serves the NFL in the same JSON format, so the NFL adapter
(`sportsai/sports/nfl/adapter.py`) reuses ingestion, the expected-points pipeline and the
point-in-time feature builder. Its settings:

* one league;
* Elo K = 20, home bonus 48, one-third regression to the mean between seasons;
* a stronger preseason prior;
* weeks 1–18 plus playoffs, with the Pro Bowl excluded;
* full team names;
* the ESPN injury report (Out / Doubtful / Questionable) captured before kickoff and shown as risk
  context. It is not a model input, because there is no historical injury archive to train on.

It has its own expected-points model (`store/nfl_ep_model.json`) and its own learning policy: a
cycle runs every 48 new games, i.e. about every 3 weeks.

```bash
python -m sportsai --sport nfl week --report reports/NFL_week.md   # all unstarted games this week
python -m sportsai --sport nfl predict "Chiefs at Raiders"
python -m sportsai --sport nfl update                              # grade + learn when due
```

NFL v1.0 (bootstrapped 2026-10-04): walk-forward on 1,140 out-of-sample games (2022–2026) gives
64.0% accuracy, log loss 0.637, Brier 0.224 and margin MAE 10.1. The 2026 Week 4 picks were frozen at
10:43:44Z, before the first kickoff (13:30Z), in commit 48557b4. See `reports/NFL_2026_W4_PREDICTIONS.md`.

## Adding another sport

Implement `SportAdapter` and `FeatureBuilder` (see `core/sport.py`) and register the adapter in
`get_adapter`. The database, registry, engine, evaluation, error analysis and learner are reused
unchanged. ESPN's JSON endpoints share one format across leagues, so `sports/ncaaf/espn.py` and
`parse.py` cover most of an NFL or NBA adapter. The sport-specific work is the stats, the features
and the explanations.

## Data sources and known gaps

* ESPN public JSON API only. CollegeFootballData needs an API key; Sports-Reference refuses
  automated access. Neither is bypassed.
* Injuries: **data unavailable**. There is no public college injury feed. QB availability is proxied
  by starter continuity.
* Weather: the ESPN/AccuWeather forecast is captured for display. It is not a model input, because
  there is no historical forecast archive to train on.
* Travel distance: not modelled (no venue coordinates in the source).

## Current state (2026-10-04)

| version | kind | training cutoff | games | why |
|---|---|---|---|---|
| v0-legacy | legacy | 2026-10-03 12:00Z | — | the original one-off pipeline; its 42 blind picks for 2026-10-03 were imported from commit 82c7a54 |
| v1.0 | bootstrap | 2026-10-03 09:00Z | 7,146 | default config. Walk-forward on 3,000 games: 76.4% accuracy, log loss 0.474, Brier 0.157, margin MAE 12.5 |
| **v1.1** (champion) | refit | 2026-10-04 | 7,254 | first learning cycle (+108 games from Oct 3). 8 diagnosis-driven challengers were tested. The best (`drop_useless_features`) gained +0.0013 log loss, p = 0.10: below the 0.002 / 0.05 gates, so it was rejected and the validated config was re-estimated instead |

First learning cycle, error analysis of v1.0 (`store/reports/ncaaf/20261004T100805Z_66e7eb8d7bb0.md`):
* **Under-confident.** Calibration slope 1.11 ± 0.05. High-confidence misses: 136 observed vs 168 expected.
* **Margins too small** for big favourites, FBS-vs-FCS games and early-season games (+2.4 to +3.9 pts,
  BH q < 0.02).
* QB-continuity features, three passing/success matchups and turnover rate add nothing out of sample.

Neither recalibration nor stacking beat the champion on held-out games, so the system logged these
findings and kept the structure. The same tests rerun automatically every cycle as data accumulates.

Graded record so far, Oct 3 slate:
* v0-legacy (blind): 42 games, 76.2% accuracy, log loss 0.491.
* v1.0 (point-in-time backtest, not blind): 108 games, 75.9% accuracy, log loss 0.538.
* Same 42 games: v1.0 log loss 0.487 vs v0 0.491.

Pending blind picks will be graded automatically by the next `update` after their games:
Georgia @ Alabama, Texas vs Oklahoma, USC @ Penn State.
