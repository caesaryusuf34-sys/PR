# NCAAF — predictions for Wed 2026-10-07 and Thu 2026-10-08 (US Eastern)

- Model **v1.1** · predictions frozen 2026-10-07T20:48:24Z (before every kickoff listed) · all inputs are pre-game; results/injuries after the cutoff never enter
- Walk-forward validation (3000 out-of-sample games): accuracy 76.3%, log loss 0.476, Brier 0.158, margin MAE 12.6
- Betting lines are not model inputs. Injury reports are shown as risk context only (no historical injury data to train on).

## Games

```
Jacksonville State @ Kennesaw State
[BLIND pre-game prediction · model v1.1 · data cutoff 2026-10-07T20:48:24Z · kickoff 2026-10-07T23:00:00Z]

Prediction: Kennesaw State
Win Probability: Jacksonville State 42% / Kennesaw State 58%
Projected Score: Jacksonville State 24–26 Kennesaw State
Projected Margin: 2.7
Upset Probability: 42%
Confidence: 1.0/10

Key Factors:

1. Pass game vs pass defense (opp-adj pass EPA/play): edge Jacksonville State (1.19 SD) — vs-average expectation: Kennesaw State offense vs Jacksonville State defense -0.18 EPA/play, Jacksonville State offense vs Kennesaw State defense +0.10 EPA/play
2. Points-per-drive matchup: edge Jacksonville State (0.76 SD) — vs-average expectation: Kennesaw State offense vs Jacksonville State defense -0.32 pts, Jacksonville State offense vs Kennesaw State defense +0.16 pts
3. Finishing drives (pts per scoring opportunity): edge Jacksonville State (0.76 SD) — vs-average expectation: Kennesaw State offense vs Jacksonville State defense -0.37 pts, Jacksonville State offense vs Kennesaw State defense +0.17 pts

Main Risk: ensemble members disagree on the winner; Jacksonville State's best counter: Pass game vs pass defense (opp-adj pass EPA/play) (1.19 SD)

Model drivers (gradient-boosting SHAP, log-odds toward):
  - Opponent-adjusted scoring margin: -0.58 → Jacksonville State
  - Elo power-rating edge (incl. home field): -0.19 → Jacksonville State
  - Points-per-drive matchup: +0.11 → Kennesaw State
  - Finishing drives (pts per scoring opportunity): +0.08 → Kennesaw State
  - Turnover tendencies: -0.08 → Jacksonville State

Member P(Kennesaw State win): elo 0.384, scoring 0.605, logistic 0.548, gbm 0.597, margin_ridge 0.561 · ensemble 0.578
  home_field: Kennesaw State at home (scoring-model home coef 5.9 pts)
  form: last-3 margin vs expectation: Kennesaw State -15.2, Jacksonville State +2.0
  rest: rest days: Kennesaw State 11, Jacksonville State 11
  qb_continuity: starter share of season attempts: Kennesaw State 100%, Jacksonville State 100%
  injuries: Data unavailable (no public college injury feed)

logged as prediction 2682404f95bd4581b79c2ab82c766295 (snapshot sha256 5265f28aa81b9c07…)
```

```
New Mexico State @ Florida International
[BLIND pre-game prediction · model v1.1 · data cutoff 2026-10-07T20:48:24Z · kickoff 2026-10-07T23:30:00Z]

Prediction: Florida International
Win Probability: New Mexico State 37% / Florida International 63%
Projected Score: New Mexico State 19–25 Florida International
Projected Margin: 5.5
Upset Probability: 37%
Confidence: 3.2/10

Key Factors:

1. Passing success matchup: edge Florida International (0.62 SD) — vs-average expectation: Florida International offense vs New Mexico State defense +2.67 pp, New Mexico State offense vs Florida International defense -4.17 pp
2. Yards-per-play matchup: edge Florida International (0.49 SD) — vs-average expectation: Florida International offense vs New Mexico State defense -0.08 yds, New Mexico State offense vs Florida International defense -0.98 yds
3. Pass protection vs pass rush (sack rate): edge Florida International (0.44 SD) — vs-average expectation: Florida International offense vs New Mexico State defense -2.39 pp, New Mexico State offense vs Florida International defense -0.57 pp

Main Risk: New Mexico State's best counter: Turnover tendencies (0.32 SD)

Model drivers (gradient-boosting SHAP, log-odds toward):
  - Opponent-adjusted scoring margin: -0.56 → New Mexico State
  - Elo power-rating edge (incl. home field): +0.20 → Florida International
  - Passing success matchup: -0.11 → New Mexico State
  - Scoring environment: +0.07 → Florida International
  - Rest advantage: +0.06 → Florida International

Member P(Florida International win): elo 0.740, scoring 0.630, logistic 0.718, gbm 0.648, margin_ridge 0.630 · ensemble 0.632
  home_field: Florida International at home (scoring-model home coef 5.9 pts)
  form: last-3 margin vs expectation: Florida International +1.8, New Mexico State +3.3
  rest: rest days: Florida International 11, New Mexico State 5
  qb_continuity: starter share of season attempts: Florida International 100%, New Mexico State 32%
  injuries: Data unavailable (no public college injury feed)

logged as prediction 7f30d8dc9f5241a29d358109c79cf1a6 (snapshot sha256 2f0e9ba017cd6e82…)
```

```
Sam Houston @ Liberty
[BLIND pre-game prediction · model v1.1 · data cutoff 2026-10-07T20:48:24Z · kickoff 2026-10-08T23:00:00Z]

Prediction: Liberty
Win Probability: Sam Houston 16% / Liberty 84%
Projected Score: Sam Houston 19–35 Liberty
Projected Margin: 16.0
Upset Probability: 16%
Confidence: 7.3/10

Key Factors:

1. Run game vs run defense (opp-adj rush EPA/play): edge Liberty (1.39 SD) — vs-average expectation: Liberty offense vs Sam Houston defense +0.23 EPA/play, Sam Houston offense vs Liberty defense -0.10 EPA/play
2. Rushing success matchup: edge Liberty (1.30 SD) — vs-average expectation: Liberty offense vs Sam Houston defense +12.00 pp, Sam Houston offense vs Liberty defense -1.42 pp
3. Field position / special teams: edge Liberty (1.10 SD) — vs-average expectation: Liberty offense vs Sam Houston defense +2.12 yds, Sam Houston offense vs Liberty defense -2.83 yds

Main Risk: Sam Houston's best counter: Pass protection vs pass rush (sack rate) (1.09 SD)

Model drivers (gradient-boosting SHAP, log-odds toward):
  - Opponent-adjusted scoring margin: +0.53 → Liberty
  - Elo power-rating edge (incl. home field): +0.35 → Liberty
  - Run game vs run defense (opp-adj rush EPA/play): +0.15 → Liberty
  - Points-per-drive matchup: -0.11 → Sam Houston
  - Pass protection vs pass rush (sack rate): -0.07 → Sam Houston

Member P(Liberty win): elo 0.881, scoring 0.833, logistic 0.848, gbm 0.874, margin_ridge 0.841 · ensemble 0.842
  home_field: Liberty at home (scoring-model home coef 5.9 pts)
  form: last-3 margin vs expectation: Liberty +21.8, Sam Houston +13.6
  rest: rest days: Liberty 6, Sam Houston 12
  qb_continuity: starter share of season attempts: Liberty 100%, Sam Houston 100%
  injuries: Data unavailable (no public college injury feed)

logged as prediction 9db3c839b8914f9bbc55f5a1be6aec95 (snapshot sha256 95ca9d884e2a5866…)
```

```
Missouri State @ Western Kentucky
[BLIND pre-game prediction · model v1.1 · data cutoff 2026-10-07T20:48:24Z · kickoff 2026-10-08T23:00:00Z]

Prediction: Missouri State
Win Probability: Missouri State 52% / Western Kentucky 48%
Projected Score: Missouri State 28–27 Western Kentucky
Projected Margin: 0.8
Upset Probability: 48%
Confidence: 1.0/10

Key Factors:

1. Turnover tendencies: edge Missouri State (1.18 SD) — vs-average expectation: Western Kentucky offense vs Missouri State defense +0.61 pp, Missouri State offense vs Western Kentucky defense -0.37 pp
2. Run game vs run defense (opp-adj rush EPA/play): edge Missouri State (0.86 SD) — vs-average expectation: Western Kentucky offense vs Missouri State defense +0.08 EPA/play, Missouri State offense vs Western Kentucky defense +0.23 EPA/play
3. Yards-per-play matchup: edge Missouri State (0.82 SD) — vs-average expectation: Western Kentucky offense vs Missouri State defense -0.18 yds, Missouri State offense vs Western Kentucky defense +0.63 yds

Main Risk: ensemble members disagree on the winner; Western Kentucky's best counter: Pass game vs pass defense (opp-adj pass EPA/play) (0.80 SD)

Model drivers (gradient-boosting SHAP, log-odds toward):
  - Opponent-adjusted scoring margin: -0.75 → Missouri State
  - Elo power-rating edge (incl. home field): -0.16 → Missouri State
  - Yards-per-play matchup: -0.08 → Missouri State
  - Success-rate matchup: +0.08 → Western Kentucky
  - Home QB continuity: -0.04 → Missouri State

Member P(Western Kentucky win): elo 0.551, scoring 0.560, logistic 0.447, gbm 0.434, margin_ridge 0.458 · ensemble 0.484
  home_field: Western Kentucky at home (scoring-model home coef 5.9 pts)
  form: last-3 margin vs expectation: Western Kentucky -3.0, Missouri State +8.3
  rest: rest days: Western Kentucky 6, Missouri State 11
  qb_continuity: starter share of season attempts: Western Kentucky 65%, Missouri State 100%
  injuries: Data unavailable (no public college injury feed)

logged as prediction 0185d4129320479c997e5085497f6aeb (snapshot sha256 d63d35a65721123b…)
```

```
South Florida @ UTSA
[BLIND pre-game prediction · model v1.1 · data cutoff 2026-10-07T20:48:24Z · kickoff 2026-10-08T23:30:00Z]

Prediction: UTSA
Win Probability: South Florida 40% / UTSA 60%
Projected Score: South Florida 24–28 UTSA
Projected Margin: 4.1
Upset Probability: 40%
Confidence: 2.8/10

Key Factors:

1. Finishing drives (pts per scoring opportunity): edge South Florida (1.14 SD) — vs-average expectation: UTSA offense vs South Florida defense -0.10 pts, South Florida offense vs UTSA defense +0.79 pts
2. Pass protection vs pass rush (sack rate): edge UTSA (0.65 SD) — vs-average expectation: UTSA offense vs South Florida defense -1.14 pp, South Florida offense vs UTSA defense +1.37 pp
3. Turnover tendencies: edge South Florida (0.62 SD) — vs-average expectation: UTSA offense vs South Florida defense -0.02 pp, South Florida offense vs UTSA defense -0.50 pp

Main Risk: South Florida's best counter: Finishing drives (pts per scoring opportunity) (1.14 SD)

Model drivers (gradient-boosting SHAP, log-odds toward):
  - Opponent-adjusted scoring margin: -0.63 → South Florida
  - Elo power-rating edge (incl. home field): +0.13 → UTSA
  - Pass protection vs pass rush (sack rate): +0.08 → UTSA
  - Yards-per-play matchup: +0.06 → UTSA
  - Field position / special teams: +0.05 → UTSA

Member P(UTSA win): elo 0.638, scoring 0.588, logistic 0.608, gbm 0.637, margin_ridge 0.600 · ensemble 0.601
  home_field: UTSA at home (scoring-model home coef 5.9 pts)
  form: last-3 margin vs expectation: UTSA -6.2, South Florida +0.0
  rest: rest days: UTSA 5, South Florida 5
  qb_continuity: starter share of season attempts: UTSA 100%, South Florida 100%
  injuries: Data unavailable (no public college injury feed)

logged as prediction 11e394b1d38546d48e5d62565e67fe19 (snapshot sha256 3dd02d14dd155e3a…)
```

```
South Alabama @ Arkansas State
[BLIND pre-game prediction · model v1.1 · data cutoff 2026-10-07T20:48:24Z · kickoff 2026-10-08T23:30:00Z]

Prediction: Arkansas State
Win Probability: South Alabama 39% / Arkansas State 61%
Projected Score: South Alabama 28–32 Arkansas State
Projected Margin: 4.2
Upset Probability: 39%
Confidence: 2.8/10

Key Factors:

1. Field position / special teams: edge Arkansas State (0.65 SD) — vs-average expectation: Arkansas State offense vs South Alabama defense +1.45 yds, South Alabama offense vs Arkansas State defense -1.70 yds
2. Pass game vs pass defense (opp-adj pass EPA/play): edge South Alabama (0.51 SD) — vs-average expectation: Arkansas State offense vs South Alabama defense +0.11 EPA/play, South Alabama offense vs Arkansas State defense +0.20 EPA/play
3. Run game vs run defense (opp-adj rush EPA/play): edge Arkansas State (0.51 SD) — vs-average expectation: Arkansas State offense vs South Alabama defense +0.16 EPA/play, South Alabama offense vs Arkansas State defense +0.02 EPA/play

Main Risk: South Alabama's best counter: Pass game vs pass defense (opp-adj pass EPA/play) (0.51 SD)

Model drivers (gradient-boosting SHAP, log-odds toward):
  - Opponent-adjusted scoring margin: -0.56 → South Alabama
  - Away QB continuity: -0.09 → South Alabama
  - Elo power-rating edge (incl. home field): -0.08 → South Alabama
  - Field position / special teams: +0.06 → Arkansas State
  - Run game vs run defense (opp-adj rush EPA/play): -0.06 → South Alabama

Member P(Arkansas State win): elo 0.579, scoring 0.636, logistic 0.633, gbm 0.553, margin_ridge 0.605 · ensemble 0.608
  home_field: Arkansas State at home (scoring-model home coef 5.9 pts)
  form: last-3 margin vs expectation: Arkansas State -5.3, South Alabama -1.3
  rest: rest days: Arkansas State 4, South Alabama 5
  qb_continuity: starter share of season attempts: Arkansas State 100%, South Alabama 58%
  injuries: Data unavailable (no public college injury feed)

logged as prediction ee6dfe5a2057470fa70ef6c803003ecf (snapshot sha256 4a93383664db0eaa…)
```

## Summary

| Game | Kickoff (UTC) | Pick | Win % | Projected score | Upset % | Confidence | Flags |
|---|---|---|---|---|---|---|---|
| Jacksonville State @ Kennesaw State | 10-07 23:00 | Kennesaw State | 58% | Jacksonville State 24–26 Kennesaw State | 42% | 1.0 |  |
| New Mexico State @ Florida International | 10-07 23:30 | Florida International | 63% | New Mexico State 19–25 Florida International | 37% | 3.2 |  |
| Sam Houston @ Liberty | 10-08 23:00 | Liberty | 84% | Sam Houston 19–35 Liberty | 16% | 7.3 |  |
| Missouri State @ Western Kentucky | 10-08 23:00 | Missouri State | 52% | Missouri State 28–27 Western Kentucky | 48% | 1.0 |  |
| South Florida @ UTSA | 10-08 23:30 | UTSA | 60% | South Florida 24–28 UTSA | 40% | 2.8 |  |
| South Alabama @ Arkansas State | 10-08 23:30 | Arkansas State | 61% | South Alabama 28–32 Arkansas State | 39% | 2.8 |  |

**Most confident:** Liberty (84%), Florida International (63%), Arkansas State (61%), UTSA (60%)

**Closest:** Missouri State @ Western Kentucky (52%), Jacksonville State @ Kennesaw State (58%), South Florida @ UTSA (60%), South Alabama @ Arkansas State (61%)

**Upset candidates (underdog ≥ 35%):** Western Kentucky 48%, Jacksonville State 42%, South Florida 40%, South Alabama 39%, New Mexico State 37%

**Highest model disagreement:** Jacksonville State @ Kennesaw State, Missouri State @ Western Kentucky, New Mexico State @ Florida International

> Data note: ESPN was unreachable from this environment (network policy) when these were made, so the database was last synced 2026-10-04 10:11Z. All completed games through Saturday 2026-10-03 are included; the Tuesday 2026-10-06 Southern Miss @ Troy result is not. Kickoff times are as scheduled on 2026-10-04 and no weather forecast was captured.
