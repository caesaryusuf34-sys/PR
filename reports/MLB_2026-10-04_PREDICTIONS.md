# MLB — predictions for 2026-10-04

- Model **v1.0** · predictions frozen 2026-10-04T12:38:44Z (before every kickoff listed) · all inputs are pre-game; results/injuries after the cutoff never enter
- Walk-forward validation (3000 out-of-sample games): accuracy 55.8%, log loss 0.680, Brier 0.243, margin MAE 3.5
- Betting lines are not model inputs. Starting pitchers are the announced probables as captured at the freeze (TBD = the team's recent rotation on average). A tied game (NPB/KBO/CPBL) voids the pick.

## Games

```
San Diego Padres @ Milwaukee Brewers
[BLIND pre-game prediction · model v1.0 · data cutoff 2026-10-04T12:38:44Z · first pitch 2026-10-04T20:00:00Z]

Prediction: Milwaukee Brewers
Win Probability: San Diego Padres 40% / Milwaukee Brewers 60%
Projected Score: San Diego Padres 3.7–4.7 Milwaukee Brewers
Projected Margin: 1.0
Upset Probability: 40%
Confidence: 5.2/10

Key Factors:

1. Lineup & defence: opponent-adjusted runs scored/allowed per game (ex-starter) — Milwaukee Brewers +0.58/-0.61, San Diego Padres -0.05/-0.32 → edge Milwaukee Brewers (0.92 runs)
2. Starting pitching: Michael King (San Diego Padres) vs Logan Henderson (Milwaukee Brewers) — FIP-type vs league -0.12 vs -0.59 per 9, K-BB/9 -0.50 vs +2.29, starts this season 33 vs 18; run-prevention rating -0.27 vs -0.53 runs/game (lower is better) → edge Milwaukee Brewers (0.28 runs)
3. Home field: Milwaukee Brewers (league home edge +0.06 runs/game) → edge Milwaukee Brewers (0.06 runs)

Main Risk: single-game variance is large in baseball (run-margin sigma ≈ 4.5)

Model drivers (gradient-boosting SHAP, log-odds toward):
  - Starter strikeout-minus-walk rate: +0.17 → Milwaukee Brewers
  - Opponent-adjusted run model (incl. starters): +0.11 → Milwaukee Brewers
  - Elo rating edge (incl. home field): +0.08 → Milwaukee Brewers
  - Run environment: +0.04 → Milwaukee Brewers
  - Bullpen workload, last 3 days: -0.03 → San Diego Padres

Member P(Milwaukee Brewers win): elo 0.580, scoring 0.587, logistic 0.599, gbm 0.622, margin_ridge 0.574 · ensemble 0.602
  starters: San Diego Padres: Michael King (rest 5 d) · Milwaukee Brewers: Logan Henderson (rest 10 d)
  home_field: Milwaukee Brewers at home
  rest: team rest days: Milwaukee Brewers 0, San Diego Padres 0; games in last 7 days: Milwaukee Brewers 1, San Diego Padres 3
  park: park run factor vs league -0.13 runs/game

already logged as prediction ff1acab64d414dbcb7650e8d12490022 (snapshot sha256 7ec6672fba0fbe82…)
```

```
Atlanta Braves @ Los Angeles Dodgers
[BLIND pre-game prediction · model v1.0 · data cutoff 2026-10-04T12:38:44Z · first pitch 2026-10-05T00:00:00Z]

Prediction: Los Angeles Dodgers
Win Probability: Atlanta Braves 41% / Los Angeles Dodgers 59%
Projected Score: Atlanta Braves 3.7–4.5 Los Angeles Dodgers
Projected Margin: 0.8
Upset Probability: 41%
Confidence: 3.1/10

Key Factors:

1. Lineup & defence: opponent-adjusted runs scored/allowed per game (ex-starter) — Los Angeles Dodgers +0.40/-0.60, Atlanta Braves +0.08/-0.34 → edge Los Angeles Dodgers (0.58 runs)
2. Form (last 10, runs vs Elo expectation): Los Angeles Dodgers +1.76, Atlanta Braves -0.37 → edge Los Angeles Dodgers (0.53 runs)
3. Starting pitching: TBD (rotation average) (Atlanta Braves) vs Blake Snell (Los Angeles Dodgers) — FIP-type vs league -0.27 vs -1.25 per 9, K-BB/9 +0.37 vs +1.60, starts this season 14 vs 9; run-prevention rating -0.29 vs -0.36 runs/game (lower is better) → edge Los Angeles Dodgers (0.34 runs)

Main Risk: starter not announced for Atlanta Braves — rotation average used; Atlanta Braves bullpen heavily used in the last 3 days (10.7 relief innings); single-game variance is large in baseball (run-margin sigma ≈ 4.5)

Model drivers (gradient-boosting SHAP, log-odds toward):
  - Opponent-adjusted run model (incl. starters): +0.12 → Los Angeles Dodgers
  - Starter strikeout-minus-walk rate: +0.10 → Los Angeles Dodgers
  - Elo rating edge (incl. home field): +0.08 → Los Angeles Dodgers
  - Run environment: +0.05 → Los Angeles Dodgers
  - Starter runs allowed per 9: +0.05 → Los Angeles Dodgers

Member P(Los Angeles Dodgers win): elo 0.583, scoring 0.549, logistic 0.580, gbm 0.656, margin_ridge 0.548 · ensemble 0.590
  starters: Atlanta Braves: TBD (rotation average) (rest 5 d) · Los Angeles Dodgers: Blake Snell (rest 8 d)
  home_field: Los Angeles Dodgers at home
  rest: team rest days: Los Angeles Dodgers 1, Atlanta Braves 1; games in last 7 days: Los Angeles Dodgers 1, Atlanta Braves 4
  park: park run factor vs league +0.04 runs/game

already logged as prediction 4ef130d463cb4a9a8b6d0260de21776a (snapshot sha256 5330947f7ad3d27b…)
```

## Summary

| Game | First pitch (UTC) | Pick | Win % | Projected score | Upset % | Confidence | Flags |
|---|---|---|---|---|---|---|---|
| San Diego Padres @ Milwaukee Brewers | 10-04 20:00 | Milwaukee Brewers | 60% | San Diego Padres 3.7–4.7 Milwaukee Brewers | 40% | 5.2 |  |
| Atlanta Braves @ Los Angeles Dodgers | 10-05 00:00 | Los Angeles Dodgers | 59% | Atlanta Braves 3.7–4.5 Los Angeles Dodgers | 41% | 3.1 | starter TBD |

**Most confident:** Milwaukee Brewers (60%), Los Angeles Dodgers (59%)

**Closest:** Atlanta Braves @ Los Angeles Dodgers (59%), San Diego Padres @ Milwaukee Brewers (60%)

**Upset candidates (underdog ≥ 35%):** Atlanta Braves 41%, San Diego Padres 40%

**Highest model disagreement:** Atlanta Braves @ Los Angeles Dodgers, San Diego Padres @ Milwaukee Brewers