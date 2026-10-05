# NFL — predictions for the current week

- Model **v1.0** · predictions frozen 2026-10-05T14:00:58Z (before every kickoff listed) · all inputs are pre-game; results/injuries after the cutoff never enter
- Walk-forward validation (1140 out-of-sample games): accuracy 63.9%, log loss 0.637, Brier 0.224, margin MAE 10.1
- Not predicted (already started or final at freeze): Pittsburgh Steelers @ Cleveland Browns; Indianapolis Colts @ Washington Commanders; Arizona Cardinals @ New York Giants; Dallas Cowboys @ Houston Texans; Green Bay Packers @ Tampa Bay Buccaneers; Jacksonville Jaguars @ Cincinnati Bengals; Los Angeles Rams @ Philadelphia Eagles; New England Patriots @ Buffalo Bills; New York Jets @ Chicago Bears; Tennessee Titans @ Baltimore Ravens; Miami Dolphins @ Minnesota Vikings; Denver Broncos @ San Francisco 49ers; Kansas City Chiefs @ Las Vegas Raiders; Los Angeles Chargers @ Seattle Seahawks; Detroit Lions @ Carolina Panthers
- Betting lines are not model inputs. Injury reports are shown as risk context only (no historical injury data to train on).

## Games

```
Atlanta Falcons @ New Orleans Saints
[BLIND pre-game prediction · model v1.0 · data cutoff 2026-10-05T14:00:58Z · kickoff 2026-10-06T00:15:00Z]

Prediction: New Orleans Saints
Win Probability: Atlanta Falcons 47% / New Orleans Saints 53%
Projected Score: Atlanta Falcons 21–22 New Orleans Saints
Projected Margin: 1.3
Upset Probability: 47%
Confidence: 1.0/10

Key Factors:

1. Rushing success matchup: edge Atlanta Falcons (1.59 SD) — vs-average expectation: New Orleans Saints offense vs Atlanta Falcons defense -2.93 pp, Atlanta Falcons offense vs New Orleans Saints defense +5.79 pp
2. Turnover tendencies: edge New Orleans Saints (1.43 SD) — vs-average expectation: New Orleans Saints offense vs Atlanta Falcons defense -0.30 pp, Atlanta Falcons offense vs New Orleans Saints defense +0.93 pp
3. Success-rate matchup: edge Atlanta Falcons (1.04 SD) — vs-average expectation: New Orleans Saints offense vs Atlanta Falcons defense -1.15 pp, Atlanta Falcons offense vs New Orleans Saints defense +3.07 pp

Main Risk: Atlanta Falcons's best counter: Rushing success matchup (1.59 SD); starting QB changed last game (Atlanta Falcons) — availability unconfirmed

Model drivers (gradient-boosting SHAP, log-odds toward):
  - Rushing success matchup: -0.29 → Atlanta Falcons
  - Away QB continuity: +0.16 → New Orleans Saints
  - Elo power-rating edge (incl. home field): -0.11 → Atlanta Falcons
  - Pass protection vs pass rush (sack rate): -0.11 → Atlanta Falcons
  - Recent form vs expectation: +0.10 → New Orleans Saints

Member P(New Orleans Saints win): elo 0.525, scoring 0.606, logistic 0.519, gbm 0.528, margin_ridge 0.537 · ensemble 0.529
  home_field: New Orleans Saints at home (scoring-model home coef 0.5 pts)
  form: last-3 margin vs expectation: New Orleans Saints +2.7, Atlanta Falcons -3.8
  rest: rest days: New Orleans Saints 8, Atlanta Falcons 11
  qb_continuity: starter share of season attempts: New Orleans Saints 100%, Atlanta Falcons 39%
  injuries: see injury report in context (pre-game ESPN feed; not a model input)
  injury_report: NO: Anfernee Jennings (LB, Out), Carl Granderson (DE, Out), Kaden Elliss (LB, Out); ATL: no outs
  weather (forecast): {'temperature': 81, 'gust': 9, 'precipitation': 40, 'conditionId': '7'}

logged as prediction 83d8f91d0a3b44f88180b3c19f53c667 (snapshot sha256 5f4787de218ea294…)
```

## Summary

| Game | Kickoff (UTC) | Pick | Win % | Projected score | Upset % | Confidence | Flags |
|---|---|---|---|---|---|---|---|
| Atlanta Falcons @ New Orleans Saints | 10-06 00:15 | New Orleans Saints | 53% | Atlanta Falcons 21–22 New Orleans Saints | 47% | 1.0 |  |

**Most confident:** New Orleans Saints (53%)

**Closest:** Atlanta Falcons @ New Orleans Saints (53%)

**Upset candidates (underdog ≥ 35%):** Atlanta Falcons 47%

**Highest model disagreement:** Atlanta Falcons @ New Orleans Saints