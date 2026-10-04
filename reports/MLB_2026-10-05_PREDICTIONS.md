# MLB — predictions for 2026-10-05

- Model **v1.0** · predictions frozen 2026-10-04T16:19:05Z (before every kickoff listed) · all inputs are pre-game; results/injuries after the cutoff never enter
- Walk-forward validation (3000 out-of-sample games): accuracy 55.8%, log loss 0.680, Brier 0.243, margin MAE 3.5
- Betting lines are not model inputs. Starting pitchers are the announced probables as captured at the freeze (TBD = the team's recent rotation on average). A tied game (NPB/KBO/CPBL) voids the pick.

## Games

```
Chicago White Sox @ Cleveland Guardians
[BLIND pre-game prediction · model v1.0 · data cutoff 2026-10-04T16:19:05Z · first pitch 2026-10-05T21:00:00Z]

Prediction: Cleveland Guardians
Win Probability: Chicago White Sox 45% / Cleveland Guardians 55%
Projected Score: Chicago White Sox 3.9–4.2 Cleveland Guardians
Projected Margin: 0.3
Upset Probability: 45%
Confidence: 1.0/10

Key Factors:

1. Form (last 10, runs vs Elo expectation): Cleveland Guardians +1.32, Chicago White Sox +2.80 → edge Chicago White Sox (0.37 runs)
2. Lineup & defence: opponent-adjusted runs scored/allowed per game (ex-starter) — Cleveland Guardians -0.34/-0.29, Chicago White Sox +0.25/-0.05 → edge Chicago White Sox (0.35 runs)
3. Bullpen: FIP-type vs league Cleveland Guardians -0.53, Chicago White Sox +0.01 per 9; relief innings last 3 days Cleveland Guardians 4.7, Chicago White Sox 6.3 → edge Cleveland Guardians (0.18 runs)

Main Risk: starter not announced for Chicago White Sox — rotation average used; ensemble members disagree on the winner; Chicago White Sox's best counter: recent form (0.37 runs)

Model drivers (gradient-boosting SHAP, log-odds toward):
  - Starter strikeout-minus-walk rate: +0.17 → Cleveland Guardians
  - Starter runs allowed per 9: +0.05 → Cleveland Guardians
  - Recent form vs expectation (last 10): -0.05 → Chicago White Sox
  - Bullpen workload, last 3 days: -0.04 → Chicago White Sox
  - Opponent-adjusted run model (incl. starters): -0.04 → Chicago White Sox

Member P(Cleveland Guardians win): elo 0.521, scoring 0.483, logistic 0.556, gbm 0.559, margin_ridge 0.518 · ensemble 0.546
  starters: Chicago White Sox: TBD (rotation average) (rest 5 d) · Cleveland Guardians: Gavin Williams (rest 10 d)
  home_field: Cleveland Guardians at home
  rest: team rest days: Cleveland Guardians 2, Chicago White Sox 2; games in last 7 days: Cleveland Guardians 1, Chicago White Sox 3
  park: park run factor vs league -0.52 runs/game

already logged as prediction 343702dcb8624ffd90703b7ac182e1e2 (snapshot sha256 81d038312848735d…)
```

```
New York Yankees @ Tampa Bay Rays
[BLIND pre-game prediction · model v1.0 · data cutoff 2026-10-04T16:19:05Z · first pitch 2026-10-06T00:00:00Z]

Prediction: New York Yankees
Win Probability: New York Yankees 56% / Tampa Bay Rays 44%
Projected Score: New York Yankees 4.5–3.7 Tampa Bay Rays
Projected Margin: 0.8
Upset Probability: 44%
Confidence: 2.2/10

Key Factors:

1. Starting pitching: Cam Schlittler (New York Yankees) vs Freddy Peralta (Tampa Bay Rays) — FIP-type vs league -1.31 vs -0.08 per 9, K-BB/9 +2.85 vs +0.39, starts this season 34 vs 32; run-prevention rating -0.44 vs -0.14 runs/game (lower is better) → edge New York Yankees (0.56 runs)
2. Lineup & defence: opponent-adjusted runs scored/allowed per game (ex-starter) — Tampa Bay Rays +0.06/-0.38, New York Yankees +0.25/-0.61 → edge New York Yankees (0.42 runs)
3. Bullpen: FIP-type vs league Tampa Bay Rays +0.12, New York Yankees -0.09 per 9; relief innings last 3 days Tampa Bay Rays 1.0, New York Yankees 3.0 → edge New York Yankees (0.07 runs)

Main Risk: ensemble members disagree on the winner; single-game variance is large in baseball (run-margin sigma ≈ 4.5)

Model drivers (gradient-boosting SHAP, log-odds toward):
  - Starter strikeout-minus-walk rate: -0.10 → New York Yankees
  - Starting pitcher run-prevention rating: -0.07 → New York Yankees
  - Opponent-adjusted run model (incl. starters): -0.06 → New York Yankees
  - Run environment: +0.04 → Tampa Bay Rays
  - Home starter rest: -0.04 → New York Yankees

Member P(Tampa Bay Rays win): elo 0.514, scoring 0.453, logistic 0.430, gbm 0.439, margin_ridge 0.400 · ensemble 0.435
  starters: New York Yankees: Cam Schlittler (rest 6 d) · Tampa Bay Rays: Freddy Peralta (rest 10 d)
  home_field: Tampa Bay Rays at home
  rest: team rest days: Tampa Bay Rays 2, New York Yankees 2; games in last 7 days: Tampa Bay Rays 1, New York Yankees 3
  park: park run factor vs league -0.26 runs/game

already logged as prediction 884a928034464505b9450b6498147fa3 (snapshot sha256 52aeb220542248bb…)
```

## Summary

| Game | First pitch (UTC) | Pick | Win % | Projected score | Upset % | Confidence | Flags |
|---|---|---|---|---|---|---|---|
| Chicago White Sox @ Cleveland Guardians | 10-05 21:00 | Cleveland Guardians | 55% | Chicago White Sox 3.9–4.2 Cleveland Guardians | 45% | 1.0 | starter TBD |
| New York Yankees @ Tampa Bay Rays | 10-06 00:00 | New York Yankees | 56% | New York Yankees 4.5–3.7 Tampa Bay Rays | 44% | 2.2 |  |

**Most confident:** New York Yankees (56%), Cleveland Guardians (55%)

**Closest:** Chicago White Sox @ Cleveland Guardians (55%), New York Yankees @ Tampa Bay Rays (56%)

**Upset candidates (underdog ≥ 35%):** Chicago White Sox 45%, Tampa Bay Rays 44%

**Highest model disagreement:** New York Yankees @ Tampa Bay Rays, Chicago White Sox @ Cleveland Guardians