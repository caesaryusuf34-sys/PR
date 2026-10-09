# NCAAF — Friday 2026-10-09 (US) slate

- Model **v1.1** · predictions frozen 2026-10-09T19:13:24Z (before every kickoff listed) · all inputs are pre-game; results/injuries after the cutoff never enter
- Walk-forward validation (3000 out-of-sample games): accuracy 76.3%, log loss 0.476, Brier 0.158, margin MAE 12.6
- Betting lines are not model inputs. Injury reports are shown as risk context only (no historical injury data to train on).

## Games

```
Florida State @ Louisville
[BLIND pre-game prediction · model v1.1 · data cutoff 2026-10-09T19:13:24Z · kickoff 2026-10-09T23:00:00Z]

Prediction: Louisville
Win Probability: Florida State 37% / Louisville 63%
Projected Score: Florida State 28–33 Louisville
Projected Margin: 5.4
Upset Probability: 37%
Confidence: 3.1/10

Key Factors:

1. Passing success matchup: edge Louisville (0.87 SD) — vs-average expectation: Louisville offense vs Florida State defense +5.88 pp, Florida State offense vs Louisville defense -3.19 pp
2. Pass game vs pass defense (opp-adj pass EPA/play): edge Louisville (0.59 SD) — vs-average expectation: Louisville offense vs Florida State defense +0.30 EPA/play, Florida State offense vs Louisville defense +0.08 EPA/play
3. Finishing drives (pts per scoring opportunity): edge Florida State (0.49 SD) — vs-average expectation: Louisville offense vs Florida State defense +0.10 pts, Florida State offense vs Louisville defense +0.39 pts

Main Risk: Florida State's best counter: Finishing drives (pts per scoring opportunity) (0.49 SD)

Model drivers (gradient-boosting SHAP, log-odds toward):
  - Opponent-adjusted scoring margin: -0.63 → Florida State
  - Elo power-rating edge (incl. home field): +0.25 → Louisville
  - Success-rate matchup: +0.09 → Louisville
  - Yards-per-play matchup: +0.09 → Louisville
  - Passing success matchup: +0.07 → Louisville

Member P(Louisville win): elo 0.730, scoring 0.600, logistic 0.648, gbm 0.718, margin_ridge 0.627 · ensemble 0.630
  home_field: Louisville at home (scoring-model home coef 5.9 pts)
  form: last-3 margin vs expectation: Louisville -1.4, Florida State +15.8
  rest: rest days: Louisville 6, Florida State 6
  qb_continuity: starter share of season attempts: Louisville 100%, Florida State 100%
  injuries: Data unavailable (no public college injury feed)
  weather (forecast): {'temperature': 81, 'gust': 13, 'precipitation': 0, 'conditionId': '2'}

logged as prediction 20ae04e3157e43638e7dbf1a5f5c35c4 (snapshot sha256 7d0c53bad1747d52…)
```

```
Columbia @ Marist
[BLIND pre-game prediction · model v1.1 · data cutoff 2026-10-09T19:13:24Z · kickoff 2026-10-09T23:00:00Z]

Prediction: Marist
Win Probability: Columbia 29% / Marist 71%
Projected Score: Columbia 19–27 Marist
Projected Margin: 8.8
Upset Probability: 29%
Confidence: 4.9/10

Key Factors:

1. Finishing drives (pts per scoring opportunity): edge Marist (1.46 SD) — vs-average expectation: Marist offense vs Columbia defense +0.30 pts, Columbia offense vs Marist defense -1.19 pts
2. Run game vs run defense (opp-adj rush EPA/play): edge Marist (0.93 SD) — vs-average expectation: Marist offense vs Columbia defense +0.13 EPA/play, Columbia offense vs Marist defense -0.10 EPA/play
3. Field position / special teams: edge Columbia (0.71 SD) — vs-average expectation: Marist offense vs Columbia defense -1.90 yds, Columbia offense vs Marist defense +0.32 yds

Main Risk: Columbia's best counter: Field position / special teams (0.71 SD)

Model drivers (gradient-boosting SHAP, log-odds toward):
  - Opponent-adjusted scoring margin: -0.16 → Columbia
  - Elo power-rating edge (incl. home field): +0.13 → Marist
  - Rushing success matchup: -0.11 → Columbia
  - Scoring environment: +0.11 → Marist
  - Yards-per-play matchup: +0.06 → Marist

Member P(Marist win): elo 0.659, scoring 0.727, logistic 0.683, gbm 0.736, margin_ridge 0.698 · ensemble 0.710
  home_field: Marist at home (scoring-model home coef 5.9 pts)
  form: last-3 margin vs expectation: Marist +16.5, Columbia -7.9
  rest: rest days: Marist 6, Columbia 6
  qb_continuity: starter share of season attempts: Marist 100%, Columbia 100%
  injuries: Data unavailable (no public college injury feed)
  weather (forecast): {'temperature': 63, 'gust': 10, 'precipitation': 0, 'conditionId': '34'}

logged as prediction d40d90d16a65407983bc78544fd3da5a (snapshot sha256 e607e5d4ceb5bdda…)
```

```
Iowa @ Washington
[BLIND pre-game prediction · model v1.1 · data cutoff 2026-10-09T19:13:24Z · kickoff 2026-10-10T01:00:00Z]

Prediction: Iowa
Win Probability: Iowa 63% / Washington 37%
Projected Score: Iowa 25–20 Washington
Projected Margin: 4.7
Upset Probability: 37%
Confidence: 3.1/10

Key Factors:

1. Pass protection vs pass rush (sack rate): edge Iowa (1.18 SD) — vs-average expectation: Washington offense vs Iowa defense +2.42 pp, Iowa offense vs Washington defense -1.16 pp
2. Field position / special teams: edge Iowa (1.13 SD) — vs-average expectation: Washington offense vs Iowa defense -3.07 yds, Iowa offense vs Washington defense +0.82 yds
3. Finishing drives (pts per scoring opportunity): edge Iowa (0.92 SD) — vs-average expectation: Washington offense vs Iowa defense -0.40 pts, Iowa offense vs Washington defense +0.28 pts

Main Risk: Washington's best counter: Pass game vs pass defense (opp-adj pass EPA/play) (0.06 SD)

Model drivers (gradient-boosting SHAP, log-odds toward):
  - Opponent-adjusted scoring margin: -0.92 → Iowa
  - Elo power-rating edge (incl. home field): -0.16 → Iowa
  - Pass protection vs pass rush (sack rate): -0.15 → Iowa
  - Scoring environment: +0.11 → Washington
  - Field position / special teams: -0.08 → Iowa

Member P(Washington win): elo 0.483, scoring 0.399, logistic 0.352, gbm 0.397, margin_ridge 0.349 · ensemble 0.368
  home_field: Washington at home (scoring-model home coef 5.9 pts)
  form: last-3 margin vs expectation: Washington -1.7, Iowa +5.2
  rest: rest days: Washington 6, Iowa 6
  qb_continuity: starter share of season attempts: Washington 100%, Iowa 100%
  injuries: Data unavailable (no public college injury feed)
  weather (forecast): {'temperature': 53, 'gust': 24, 'precipitation': 34, 'conditionId': '3'}

logged as prediction 426d9ae87fe14111ad00bbe738f6aa7c (snapshot sha256 10e0415c677e87a1…)
```

```
Washington State @ Utah State
[BLIND pre-game prediction · model v1.1 · data cutoff 2026-10-09T19:13:24Z · kickoff 2026-10-10T01:00:00Z]

Prediction: Utah State
Win Probability: Washington State 40% / Utah State 60%
Projected Score: Washington State 19–23 Utah State
Projected Margin: 4.1
Upset Probability: 40%
Confidence: 2.6/10

Key Factors:

1. Field position / special teams: edge Utah State (1.22 SD) — vs-average expectation: Utah State offense vs Washington State defense +1.49 yds, Washington State offense vs Utah State defense -3.91 yds
2. Explosive-play matchup: edge Utah State (1.00 SD) — vs-average expectation: Utah State offense vs Washington State defense +0.65 pp, Washington State offense vs Utah State defense -3.73 pp
3. Run game vs run defense (opp-adj rush EPA/play): edge Utah State (0.72 SD) — vs-average expectation: Utah State offense vs Washington State defense +0.09 EPA/play, Washington State offense vs Utah State defense -0.10 EPA/play

Main Risk: Washington State's best counter: Passing success matchup (0.44 SD)

Model drivers (gradient-boosting SHAP, log-odds toward):
  - Opponent-adjusted scoring margin: -0.65 → Washington State
  - Elo power-rating edge (incl. home field): -0.14 → Washington State
  - Field position / special teams: +0.10 → Utah State
  - Yards-per-play matchup: +0.09 → Utah State
  - Explosive-play matchup: +0.07 → Utah State

Member P(Utah State win): elo 0.567, scoring 0.606, logistic 0.629, gbm 0.645, margin_ridge 0.581 · ensemble 0.595
  home_field: Utah State at home (scoring-model home coef 5.9 pts)
  form: last-3 margin vs expectation: Utah State -2.1, Washington State +0.6
  rest: rest days: Utah State 6, Washington State 5
  qb_continuity: starter share of season attempts: Utah State 77%, Washington State 100%
  injuries: Data unavailable (no public college injury feed)
  weather (forecast): {'temperature': 76, 'gust': 10, 'precipitation': 51, 'conditionId': '15'}

logged as prediction 51738c1401214d2f8690f0ac88c943e6 (snapshot sha256 b49cf2082fb119b6…)
```

```
Wyoming @ San José State
[BLIND pre-game prediction · model v1.1 · data cutoff 2026-10-09T19:13:24Z · kickoff 2026-10-10T01:00:00Z]

Prediction: San José State
Win Probability: Wyoming 34% / San José State 66%
Projected Score: Wyoming 19–25 San José State
Projected Margin: 6.1
Upset Probability: 34%
Confidence: 3.9/10

Key Factors:

1. Pass protection vs pass rush (sack rate): edge San José State (0.89 SD) — vs-average expectation: San José State offense vs Wyoming defense -2.91 pp, Wyoming offense vs San José State defense +0.40 pp
2. Rushing success matchup: edge Wyoming (0.68 SD) — vs-average expectation: San José State offense vs Wyoming defense -1.81 pp, Wyoming offense vs San José State defense +2.51 pp
3. Field position / special teams: edge Wyoming (0.47 SD) — vs-average expectation: San José State offense vs Wyoming defense -2.07 yds, Wyoming offense vs San José State defense -0.80 yds

Main Risk: Wyoming's best counter: Rushing success matchup (0.68 SD)

Model drivers (gradient-boosting SHAP, log-odds toward):
  - Rushing success matchup: -0.11 → Wyoming
  - Yards-per-play matchup: +0.09 → San José State
  - Opponent-adjusted scoring margin: -0.09 → Wyoming
  - Elo power-rating edge (incl. home field): +0.08 → San José State
  - Scoring environment: +0.07 → San José State

Member P(San José State win): elo 0.637, scoring 0.720, logistic 0.649, gbm 0.719, margin_ridge 0.628 · ensemble 0.664
  home_field: San José State at home (scoring-model home coef 5.9 pts)
  form: last-3 margin vs expectation: San José State -0.7, Wyoming -0.1
  rest: rest days: San José State 5, Wyoming 6
  qb_continuity: starter share of season attempts: San José State 100%, Wyoming 100%
  injuries: Data unavailable (no public college injury feed)
  weather (forecast): {'temperature': 76, 'gust': 8, 'precipitation': 0, 'conditionId': '2'}

logged as prediction 7b642e5cdf33465082d73312c9ed2d66 (snapshot sha256 9061585395b1063d…)
```

```
Iowa State @ BYU
[BLIND pre-game prediction · model v1.1 · data cutoff 2026-10-09T19:13:24Z · kickoff 2026-10-10T02:15:00Z]

Prediction: BYU
Win Probability: Iowa State 23% / BYU 77%
Projected Score: Iowa State 22–33 BYU
Projected Margin: 11.8
Upset Probability: 23%
Confidence: 6.1/10

Key Factors:

1. Rushing success matchup: edge BYU (1.34 SD) — vs-average expectation: BYU offense vs Iowa State defense +9.09 pp, Iowa State offense vs BYU defense -4.71 pp
2. Overall offense vs defense (opp-adj EPA/play): edge BYU (1.03 SD) — vs-average expectation: BYU offense vs Iowa State defense +0.19 EPA/play, Iowa State offense vs BYU defense -0.06 EPA/play
3. Success-rate matchup: edge BYU (0.96 SD) — vs-average expectation: BYU offense vs Iowa State defense +6.06 pp, Iowa State offense vs BYU defense -3.33 pp

Main Risk: Iowa State's best counter: Finishing drives (pts per scoring opportunity) (0.32 SD)

Model drivers (gradient-boosting SHAP, log-odds toward):
  - Yards-per-play matchup: +0.20 → BYU
  - Rushing success matchup: +0.14 → BYU
  - Elo power-rating edge (incl. home field): +0.13 → BYU
  - Pass game vs pass defense (opp-adj pass EPA/play): -0.08 → Iowa State
  - Overall offense vs defense (opp-adj EPA/play): +0.07 → BYU

Member P(BYU win): elo 0.829, scoring 0.753, logistic 0.815, gbm 0.783, margin_ridge 0.781 · ensemble 0.774
  home_field: BYU at home (scoring-model home coef 5.9 pts)
  form: last-3 margin vs expectation: BYU +4.9, Iowa State +7.0
  rest: rest days: BYU 6, Iowa State 6
  qb_continuity: starter share of season attempts: BYU 100%, Iowa State 100%
  injuries: Data unavailable (no public college injury feed)
  weather (forecast): {'temperature': 74, 'gust': 9, 'precipitation': 34, 'conditionId': '38'}

logged as prediction c08bc6bd4b734c86b7409f4e79190fcf (snapshot sha256 4d0f47c789b2d44a…)
```

## Summary

| Game | Kickoff (UTC) | Pick | Win % | Projected score | Upset % | Confidence | Flags |
|---|---|---|---|---|---|---|---|
| Florida State @ Louisville | 10-09 23:00 | Louisville | 63% | Florida State 28–33 Louisville | 37% | 3.1 |  |
| Columbia @ Marist | 10-09 23:00 | Marist | 71% | Columbia 19–27 Marist | 29% | 4.9 |  |
| Iowa @ Washington | 10-10 01:00 | Iowa | 63% | Iowa 25–20 Washington | 37% | 3.1 |  |
| Washington State @ Utah State | 10-10 01:00 | Utah State | 60% | Washington State 19–23 Utah State | 40% | 2.6 |  |
| Wyoming @ San José State | 10-10 01:00 | San José State | 66% | Wyoming 19–25 San José State | 34% | 3.9 |  |
| Iowa State @ BYU | 10-10 02:15 | BYU | 77% | Iowa State 22–33 BYU | 23% | 6.1 |  |

**Most confident:** BYU (77%), Marist (71%), San José State (66%), Iowa (63%)

**Closest:** Washington State @ Utah State (60%), Florida State @ Louisville (63%), Iowa @ Washington (63%), Wyoming @ San José State (66%)

**Upset candidates (underdog ≥ 35%):** Washington State 40%, Florida State 37%, Washington 37%

**Highest model disagreement:** Florida State @ Louisville, Iowa @ Washington, Wyoming @ San José State