# NFL — predictions for the current week

- Model **v1.0** · predictions frozen 2026-10-04T23:56:03Z (before every kickoff listed) · all inputs are pre-game; results/injuries after the cutoff never enter
- Walk-forward validation (1140 out-of-sample games): accuracy 63.9%, log loss 0.637, Brier 0.224, margin MAE 10.1
- Not predicted (already started or final at freeze): Pittsburgh Steelers @ Cleveland Browns; Indianapolis Colts @ Washington Commanders; Arizona Cardinals @ New York Giants; Dallas Cowboys @ Houston Texans; Green Bay Packers @ Tampa Bay Buccaneers; Jacksonville Jaguars @ Cincinnati Bengals; Los Angeles Rams @ Philadelphia Eagles; New England Patriots @ Buffalo Bills; New York Jets @ Chicago Bears; Tennessee Titans @ Baltimore Ravens; Miami Dolphins @ Minnesota Vikings; Denver Broncos @ San Francisco 49ers; Kansas City Chiefs @ Las Vegas Raiders; Los Angeles Chargers @ Seattle Seahawks
- Betting lines are not model inputs. Injury reports are shown as risk context only (no historical injury data to train on).

## Games

```
Detroit Lions @ Carolina Panthers
[BLIND pre-game prediction · model v1.0 · data cutoff 2026-10-04T23:56:03Z · kickoff 2026-10-05T00:20:00Z]

Prediction: Detroit Lions
Win Probability: Detroit Lions 58% / Carolina Panthers 42%
Projected Score: Detroit Lions 26–23 Carolina Panthers
Projected Margin: 2.3
Upset Probability: 42%
Confidence: 1.0/10

Key Factors:

1. Yards-per-play matchup: edge Detroit Lions (0.58 SD) — vs-average expectation: Carolina Panthers offense vs Detroit Lions defense +0.17 yds, Detroit Lions offense vs Carolina Panthers defense +0.54 yds
2. Explosive-play matchup: edge Carolina Panthers (0.55 SD) — vs-average expectation: Carolina Panthers offense vs Detroit Lions defense +1.27 pp, Detroit Lions offense vs Carolina Panthers defense +0.02 pp
3. Success-rate matchup: edge Detroit Lions (0.55 SD) — vs-average expectation: Carolina Panthers offense vs Detroit Lions defense +0.72 pp, Detroit Lions offense vs Carolina Panthers defense +2.91 pp

Main Risk: CAR QB Haynes King OUT; CAR: 5 players out/doubtful; ensemble members disagree on the winner; Carolina Panthers's best counter: Explosive-play matchup (0.55 SD)

Model drivers (gradient-boosting SHAP, log-odds toward):
  - Elo power-rating edge (incl. home field): -0.43 → Detroit Lions
  - Scoring environment: -0.16 → Detroit Lions
  - Pass protection vs pass rush (sack rate): -0.09 → Detroit Lions
  - Recent form vs expectation: -0.08 → Detroit Lions
  - Opponent-adjusted scoring margin: +0.08 → Carolina Panthers

Member P(Carolina Panthers win): elo 0.401, scoring 0.508, logistic 0.414, gbm 0.357, margin_ridge 0.469 · ensemble 0.419
  home_field: Carolina Panthers at home (scoring-model home coef 0.4 pts)
  form: last-3 margin vs expectation: Carolina Panthers +4.4, Detroit Lions -4.1
  rest: rest days: Carolina Panthers 7, Detroit Lions 7
  qb_continuity: starter share of season attempts: Carolina Panthers 100%, Detroit Lions 100%
  injuries: see injury report in context (pre-game ESPN feed; not a model input)
  injury_report: CAR: Cam Jackson (DT, Out), Haynes King (QB, Out), Albert Reese IV (OT, Out), Ja'Tavion Sanders (TE, Out), Damien Lewis (G, Out); DET: Mekhi Wingo (DT, Out), Devin White (LB, Out), Tyler Conklin (TE, Out), Bishop Fitzgerald (S, Out), Ahmed Hassanein (DE, Out)
  weather (forecast): {'temperature': 69, 'gust': 4, 'precipitation': 34, 'conditionId': '7'}

logged as prediction 5e4cc52f42094380bd29034154e3c3c1 (snapshot sha256 a17e3cc0d207abd3…)
```

```
Atlanta Falcons @ New Orleans Saints
[BLIND pre-game prediction · model v1.0 · data cutoff 2026-10-04T23:56:03Z · kickoff 2026-10-06T00:15:00Z]

Prediction: New Orleans Saints
Win Probability: Atlanta Falcons 47% / New Orleans Saints 53%
Projected Score: Atlanta Falcons 21–22 New Orleans Saints
Projected Margin: 1.3
Upset Probability: 47%
Confidence: 1.0/10

Key Factors:

1. Rushing success matchup: edge Atlanta Falcons (1.51 SD) — vs-average expectation: New Orleans Saints offense vs Atlanta Falcons defense -2.62 pp, Atlanta Falcons offense vs New Orleans Saints defense +5.63 pp
2. Turnover tendencies: edge New Orleans Saints (1.42 SD) — vs-average expectation: New Orleans Saints offense vs Atlanta Falcons defense -0.34 pp, Atlanta Falcons offense vs New Orleans Saints defense +0.87 pp
3. Success-rate matchup: edge Atlanta Falcons (0.98 SD) — vs-average expectation: New Orleans Saints offense vs Atlanta Falcons defense -0.89 pp, Atlanta Falcons offense vs New Orleans Saints defense +3.08 pp

Main Risk: Atlanta Falcons's best counter: Rushing success matchup (1.51 SD); starting QB changed last game (Atlanta Falcons) — availability unconfirmed

Model drivers (gradient-boosting SHAP, log-odds toward):
  - Rushing success matchup: -0.29 → Atlanta Falcons
  - Away QB continuity: +0.16 → New Orleans Saints
  - Elo power-rating edge (incl. home field): -0.11 → Atlanta Falcons
  - Pass protection vs pass rush (sack rate): -0.11 → Atlanta Falcons
  - Recent form vs expectation: +0.10 → New Orleans Saints

Member P(New Orleans Saints win): elo 0.525, scoring 0.609, logistic 0.522, gbm 0.525, margin_ridge 0.540 · ensemble 0.530
  home_field: New Orleans Saints at home (scoring-model home coef 0.4 pts)
  form: last-3 margin vs expectation: New Orleans Saints +2.7, Atlanta Falcons -3.8
  rest: rest days: New Orleans Saints 8, Atlanta Falcons 11
  qb_continuity: starter share of season attempts: New Orleans Saints 100%, Atlanta Falcons 39%
  injuries: see injury report in context (pre-game ESPN feed; not a model input)
  injury_report: NO: Anfernee Jennings (LB, Out), Carl Granderson (DE, Out), Kaden Elliss (LB, Out); ATL: no outs
  weather (forecast): {'temperature': 78, 'gust': 9, 'precipitation': 40, 'conditionId': '7'}

logged as prediction c3bc85c147c841ff96a68abcba678113 (snapshot sha256 1fabd30a5a1571f7…)
```

## Summary

| Game | Kickoff (UTC) | Pick | Win % | Projected score | Upset % | Confidence | Flags |
|---|---|---|---|---|---|---|---|
| Detroit Lions @ Carolina Panthers | 10-05 00:20 | Detroit Lions | 58% | Detroit Lions 26–23 Carolina Panthers | 42% | 1.0 | injuries: CAR QB Haynes King OUT, CAR: 5 players out/doubtful |
| Atlanta Falcons @ New Orleans Saints | 10-06 00:15 | New Orleans Saints | 53% | Atlanta Falcons 21–22 New Orleans Saints | 47% | 1.0 |  |

**Most confident:** Detroit Lions (58%), New Orleans Saints (53%)

**Closest:** Atlanta Falcons @ New Orleans Saints (53%), Detroit Lions @ Carolina Panthers (58%)

**Upset candidates (underdog ≥ 35%):** Atlanta Falcons 47%, Carolina Panthers 42%

**Highest model disagreement:** Detroit Lions @ Carolina Panthers, Atlanta Falcons @ New Orleans Saints