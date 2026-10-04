# NFL — predictions for the current week

- Model **v1.0** · predictions frozen 2026-10-04T10:43:44Z (before every kickoff listed) · all inputs are pre-game; results/injuries after the cutoff never enter
- Walk-forward validation (1140 out-of-sample games): accuracy 63.9%, log loss 0.637, Brier 0.224, margin MAE 10.1
- Not predicted (already started or final at freeze): Pittsburgh Steelers @ Cleveland Browns
- Betting lines are not model inputs. Injury reports are shown as risk context only (no historical injury data to train on).

## Games

```
Indianapolis Colts @ Washington Commanders (neutral site)
[BLIND pre-game prediction · model v1.0 · data cutoff 2026-10-04T10:43:44Z · kickoff 2026-10-04T13:30:00Z]

Prediction: Indianapolis Colts
Win Probability: Indianapolis Colts 55% / Washington Commanders 45%
Projected Score: Indianapolis Colts 26–24 Washington Commanders
Projected Margin: 1.8
Upset Probability: 45%
Confidence: 1.0/10

Key Factors:

1. Turnover tendencies: edge Washington Commanders (0.81 SD) — vs-average expectation: Washington Commanders offense vs Indianapolis Colts defense -0.73 pp, Indianapolis Colts offense vs Washington Commanders defense -0.02 pp
2. Finishing drives (pts per scoring opportunity): edge Indianapolis Colts (0.80 SD) — vs-average expectation: Washington Commanders offense vs Indianapolis Colts defense +0.09 pts, Indianapolis Colts offense vs Washington Commanders defense +0.58 pts
3. Passing success matchup: edge Indianapolis Colts (0.72 SD) — vs-average expectation: Washington Commanders offense vs Indianapolis Colts defense -0.31 pp, Indianapolis Colts offense vs Washington Commanders defense +3.49 pp

Main Risk: WSH QB Jayden Daniels OUT; WSH: 4 players out/doubtful; Washington Commanders's best counter: Turnover tendencies (0.81 SD); starting QB changed last game (Washington Commanders) — availability unconfirmed

Model drivers (gradient-boosting SHAP, log-odds toward):
  - Elo power-rating edge (incl. home field): -0.29 → Indianapolis Colts
  - Home QB continuity: -0.22 → Indianapolis Colts
  - Scoring environment: -0.21 → Indianapolis Colts
  - Pass protection vs pass rush (sack rate): -0.09 → Indianapolis Colts
  - Recent form vs expectation: +0.08 → Washington Commanders

Member P(Washington Commanders win): elo 0.487, scoring 0.473, logistic 0.381, gbm 0.345, margin_ridge 0.398 · ensemble 0.452
  home_field: neutral site
  form: last-3 margin vs expectation: Washington Commanders -0.4, Indianapolis Colts -3.3
  rest: rest days: Washington Commanders 6, Indianapolis Colts 6
  qb_continuity: starter share of season attempts: Washington Commanders 38%, Indianapolis Colts 100%
  injuries: see injury report in context (pre-game ESPN feed; not a model input)
  injury_report: WSH: Rachaad White (RB, Out), Jayden Daniels (QB, Out), Nick Cross (S, Out), Sam Cosmi (G, Out); IND: Keenan Allen (WR, Out)

logged as prediction ecc1e2e04fa9419ab069ab1ae23a7b61 (snapshot sha256 30f1c0277e335bd5…)
```

```
Arizona Cardinals @ New York Giants
[BLIND pre-game prediction · model v1.0 · data cutoff 2026-10-04T10:43:44Z · kickoff 2026-10-04T17:00:00Z]

Prediction: New York Giants
Win Probability: Arizona Cardinals 45% / New York Giants 55%
Projected Score: Arizona Cardinals 22–23 New York Giants
Projected Margin: 1.0
Upset Probability: 45%
Confidence: 1.0/10

Key Factors:

1. Finishing drives (pts per scoring opportunity): edge Arizona Cardinals (1.28 SD) — vs-average expectation: New York Giants offense vs Arizona Cardinals defense -0.22 pts, Arizona Cardinals offense vs New York Giants defense +0.57 pts
2. Rushing success matchup: edge Arizona Cardinals (1.12 SD) — vs-average expectation: New York Giants offense vs Arizona Cardinals defense -1.65 pp, Arizona Cardinals offense vs New York Giants defense +4.45 pp
3. Pass protection vs pass rush (sack rate): edge Arizona Cardinals (1.10 SD) — vs-average expectation: New York Giants offense vs Arizona Cardinals defense +1.04 pp, Arizona Cardinals offense vs New York Giants defense -1.86 pp

Main Risk: ensemble members disagree on the winner; Arizona Cardinals's best counter: Finishing drives (pts per scoring opportunity) (1.28 SD)

Model drivers (gradient-boosting SHAP, log-odds toward):
  - Rushing success matchup: -0.21 → Arizona Cardinals
  - Home QB continuity: -0.14 → Arizona Cardinals
  - Pass protection vs pass rush (sack rate): -0.08 → Arizona Cardinals
  - Elo power-rating edge (incl. home field): +0.07 → New York Giants
  - Away QB continuity: -0.07 → Arizona Cardinals

Member P(New York Giants win): elo 0.585, scoring 0.526, logistic 0.476, gbm 0.420, margin_ridge 0.511 · ensemble 0.553
  home_field: New York Giants at home (scoring-model home coef 0.7 pts)
  form: last-3 margin vs expectation: New York Giants -3.0, Arizona Cardinals +2.2
  rest: rest days: New York Giants 7, Arizona Cardinals 6
  qb_continuity: starter share of season attempts: New York Giants 63%, Arizona Cardinals 100%
  injuries: see injury report in context (pre-game ESPN feed; not a model input)
  injury_report: NYG: no outs; ARI: Dadrion Taylor-Demerson (S, Out)
  weather (forecast): {'temperature': 64, 'gust': 10, 'precipitation': 37, 'conditionId': '7'}

logged as prediction bcc632e87b30441197c2c4f48063420b (snapshot sha256 2f68800ff500a079…)
```

```
Dallas Cowboys @ Houston Texans
[BLIND pre-game prediction · model v1.0 · data cutoff 2026-10-04T10:43:44Z · kickoff 2026-10-04T17:00:00Z]

Prediction: Houston Texans
Win Probability: Dallas Cowboys 35% / Houston Texans 65%
Projected Score: Dallas Cowboys 21–26 Houston Texans
Projected Margin: 4.9
Upset Probability: 35%
Confidence: 3.6/10

Key Factors:

1. Passing success matchup: edge Houston Texans (1.20 SD) — vs-average expectation: Houston Texans offense vs Dallas Cowboys defense +4.48 pp, Dallas Cowboys offense vs Houston Texans defense -2.00 pp
2. Success-rate matchup: edge Houston Texans (0.95 SD) — vs-average expectation: Houston Texans offense vs Dallas Cowboys defense +2.27 pp, Dallas Cowboys offense vs Houston Texans defense -1.67 pp
3. Explosive-play matchup: edge Houston Texans (0.67 SD) — vs-average expectation: Houston Texans offense vs Dallas Cowboys defense +0.81 pp, Dallas Cowboys offense vs Houston Texans defense -0.71 pp

Main Risk: Dallas Cowboys's best counter: Field position / special teams (0.58 SD)

Model drivers (gradient-boosting SHAP, log-odds toward):
  - Elo power-rating edge (incl. home field): +0.14 → Houston Texans
  - Pass protection vs pass rush (sack rate): +0.11 → Houston Texans
  - Scoring environment: +0.09 → Houston Texans
  - Home QB continuity: +0.06 → Houston Texans
  - Away QB continuity: -0.06 → Dallas Cowboys

Member P(Houston Texans win): elo 0.651, scoring 0.538, logistic 0.656, gbm 0.646, margin_ridge 0.659 · ensemble 0.653
  home_field: Houston Texans at home (scoring-model home coef 0.7 pts)
  form: last-3 margin vs expectation: Houston Texans -10.3, Dallas Cowboys +2.8
  rest: rest days: Houston Texans 7, Dallas Cowboys 6
  qb_continuity: starter share of season attempts: Houston Texans 100%, Dallas Cowboys 100%
  injuries: see injury report in context (pre-game ESPN feed; not a model input)
  injury_report: HOU: Azeez Al-Shaair (LB, Out); DAL: Cobie Durant (CB, Out), DeMarvion Overshown (LB, Out)
  weather (forecast): {'temperature': 79, 'gust': 17, 'precipitation': 47, 'conditionId': '7'}

logged as prediction e77a58a96a374257981243d4ccce2bf2 (snapshot sha256 2cc105e1ca9616e8…)
```

```
Green Bay Packers @ Tampa Bay Buccaneers
[BLIND pre-game prediction · model v1.0 · data cutoff 2026-10-04T10:43:44Z · kickoff 2026-10-04T17:00:00Z]

Prediction: Green Bay Packers
Win Probability: Green Bay Packers 52% / Tampa Bay Buccaneers 48%
Projected Score: Green Bay Packers 24–23 Tampa Bay Buccaneers
Projected Margin: 0.7
Upset Probability: 48%
Confidence: 1.0/10

Key Factors:

1. Finishing drives (pts per scoring opportunity): edge Tampa Bay Buccaneers (1.33 SD) — vs-average expectation: Tampa Bay Buccaneers offense vs Green Bay Packers defense +0.52 pts, Green Bay Packers offense vs Tampa Bay Buccaneers defense -0.33 pts
2. Explosive-play matchup: edge Green Bay Packers (1.19 SD) — vs-average expectation: Tampa Bay Buccaneers offense vs Green Bay Packers defense -1.62 pp, Green Bay Packers offense vs Tampa Bay Buccaneers defense +1.11 pp
3. Rushing success matchup: edge Tampa Bay Buccaneers (0.97 SD) — vs-average expectation: Tampa Bay Buccaneers offense vs Green Bay Packers defense -0.10 pp, Green Bay Packers offense vs Tampa Bay Buccaneers defense -5.45 pp

Main Risk: ensemble members disagree on the winner; Tampa Bay Buccaneers's best counter: Finishing drives (pts per scoring opportunity) (1.33 SD)

Model drivers (gradient-boosting SHAP, log-odds toward):
  - Elo power-rating edge (incl. home field): -0.31 → Green Bay Packers
  - Scoring environment: -0.17 → Green Bay Packers
  - Pass protection vs pass rush (sack rate): -0.10 → Green Bay Packers
  - Rushing success matchup: +0.10 → Tampa Bay Buccaneers
  - Explosive-play matchup: +0.09 → Tampa Bay Buccaneers

Member P(Tampa Bay Buccaneers win): elo 0.494, scoring 0.603, logistic 0.479, gbm 0.390, margin_ridge 0.476 · ensemble 0.482
  home_field: Tampa Bay Buccaneers at home (scoring-model home coef 0.7 pts)
  form: last-3 margin vs expectation: Tampa Bay Buccaneers -4.9, Green Bay Packers -13.6
  rest: rest days: Tampa Bay Buccaneers 6, Green Bay Packers 9
  qb_continuity: starter share of season attempts: Tampa Bay Buccaneers 100%, Green Bay Packers 100%
  injuries: see injury report in context (pre-game ESPN feed; not a model input)
  injury_report: TB: Benjamin Morrison (CB, Out), Ko Kieft (TE, Out), Rueben Bain Jr. (LB, Out); GB: Jacob Monk (C, Out), Aaron Banks (G, Out), Anthony Campbell (DT, Out)
  weather (forecast): {'temperature': 88, 'gust': 10, 'precipitation': 18, 'conditionId': '4'}

logged as prediction c4cf20c94bd04cd1b3d15b3562c777f7 (snapshot sha256 490b30673347e2d0…)
```

```
Jacksonville Jaguars @ Cincinnati Bengals
[BLIND pre-game prediction · model v1.0 · data cutoff 2026-10-04T10:43:44Z · kickoff 2026-10-04T17:00:00Z]

Prediction: Jacksonville Jaguars
Win Probability: Jacksonville Jaguars 56% / Cincinnati Bengals 44%
Projected Score: Jacksonville Jaguars 25–24 Cincinnati Bengals
Projected Margin: 2.0
Upset Probability: 44%
Confidence: 1.3/10

Key Factors:

1. Finishing drives (pts per scoring opportunity): edge Jacksonville Jaguars (1.59 SD) — vs-average expectation: Cincinnati Bengals offense vs Jacksonville Jaguars defense -0.42 pts, Jacksonville Jaguars offense vs Cincinnati Bengals defense +0.57 pts
2. Turnover tendencies: edge Jacksonville Jaguars (1.41 SD) — vs-average expectation: Cincinnati Bengals offense vs Jacksonville Jaguars defense +1.17 pp, Jacksonville Jaguars offense vs Cincinnati Bengals defense -0.01 pp
3. Passing success matchup: edge Jacksonville Jaguars (1.37 SD) — vs-average expectation: Cincinnati Bengals offense vs Jacksonville Jaguars defense +1.18 pp, Jacksonville Jaguars offense vs Cincinnati Bengals defense +8.46 pp

Main Risk: Cincinnati Bengals's best counter: Field position / special teams (0.88 SD)

Model drivers (gradient-boosting SHAP, log-odds toward):
  - Elo power-rating edge (incl. home field): -0.27 → Jacksonville Jaguars
  - Opponent-adjusted scoring margin: -0.22 → Jacksonville Jaguars
  - Scoring environment: -0.11 → Jacksonville Jaguars
  - Overall offense vs defense (opp-adj EPA/play): -0.08 → Jacksonville Jaguars
  - Away QB continuity: -0.06 → Jacksonville Jaguars

Member P(Cincinnati Bengals win): elo 0.448, scoring 0.358, logistic 0.445, gbm 0.315, margin_ridge 0.463 · ensemble 0.445
  home_field: Cincinnati Bengals at home (scoring-model home coef 0.7 pts)
  form: last-3 margin vs expectation: Cincinnati Bengals +7.1, Jacksonville Jaguars +13.1
  rest: rest days: Cincinnati Bengals 7, Jacksonville Jaguars 7
  qb_continuity: starter share of season attempts: Cincinnati Bengals 100%, Jacksonville Jaguars 100%
  injuries: see injury report in context (pre-game ESPN feed; not a model input)
  injury_report: CIN: Kyle Dugger (S, Out); JAX: no outs
  weather (forecast): {'temperature': 71, 'gust': 7, 'precipitation': 0, 'conditionId': '2'}

logged as prediction 94e69477a244417f95ad5abec9fea4d4 (snapshot sha256 e553e694941b7e28…)
```

```
Los Angeles Rams @ Philadelphia Eagles
[BLIND pre-game prediction · model v1.0 · data cutoff 2026-10-04T10:43:44Z · kickoff 2026-10-04T17:00:00Z]

Prediction: Los Angeles Rams
Win Probability: Los Angeles Rams 53% / Philadelphia Eagles 47%
Projected Score: Los Angeles Rams 23–21 Philadelphia Eagles
Projected Margin: 2.0
Upset Probability: 47%
Confidence: 1.0/10

Key Factors:

1. Success-rate matchup: edge Los Angeles Rams (1.46 SD) — vs-average expectation: Philadelphia Eagles offense vs Los Angeles Rams defense -3.67 pp, Los Angeles Rams offense vs Philadelphia Eagles defense +2.26 pp
2. Yards-per-play matchup: edge Los Angeles Rams (1.43 SD) — vs-average expectation: Philadelphia Eagles offense vs Los Angeles Rams defense -0.50 yds, Los Angeles Rams offense vs Philadelphia Eagles defense +0.41 yds
3. Passing success matchup: edge Los Angeles Rams (1.21 SD) — vs-average expectation: Philadelphia Eagles offense vs Los Angeles Rams defense -5.32 pp, Los Angeles Rams offense vs Philadelphia Eagles defense +1.11 pp

Main Risk: PHI: 5 players out/doubtful; ensemble members disagree on the winner; Philadelphia Eagles's best counter: Field position / special teams (0.48 SD)

Model drivers (gradient-boosting SHAP, log-odds toward):
  - Rushing success matchup: -0.26 → Los Angeles Rams
  - Elo power-rating edge (incl. home field): -0.24 → Los Angeles Rams
  - Pass protection vs pass rush (sack rate): -0.08 → Los Angeles Rams
  - Away QB continuity: -0.07 → Los Angeles Rams
  - Overall offense vs defense (opp-adj EPA/play): -0.07 → Los Angeles Rams

Member P(Philadelphia Eagles win): elo 0.506, scoring 0.422, logistic 0.427, gbm 0.322, margin_ridge 0.413 · ensemble 0.468
  home_field: Philadelphia Eagles at home (scoring-model home coef 0.7 pts)
  form: last-3 margin vs expectation: Philadelphia Eagles -8.9, Los Angeles Rams -3.0
  rest: rest days: Philadelphia Eagles 5, Los Angeles Rams 6
  qb_continuity: starter share of season attempts: Philadelphia Eagles 100%, Los Angeles Rams 100%
  injuries: see injury report in context (pre-game ESPN feed; not a model input)
  injury_report: PHI: Fred Johnson (OT, Out), Marcus Epps (S, Out), Zack Baun (LB, Out), Hollywood Brown (WR, Out), Dallas Goedert (TE, Out); LAR: Jaylen Watson (CB, Out), Aaron Donald (DT, Out)
  weather (forecast): {'temperature': 62, 'gust': 15, 'precipitation': 34, 'conditionId': '7'}

logged as prediction 168b2dcee29144acad02d8587296daf6 (snapshot sha256 e2ce9dd550f22001…)
```

```
New England Patriots @ Buffalo Bills
[BLIND pre-game prediction · model v1.0 · data cutoff 2026-10-04T10:43:44Z · kickoff 2026-10-04T17:00:00Z]

Prediction: Buffalo Bills
Win Probability: New England Patriots 37% / Buffalo Bills 63%
Projected Score: New England Patriots 23–26 Buffalo Bills
Projected Margin: 3.9
Upset Probability: 37%
Confidence: 3.4/10

Key Factors:

1. Points-per-drive matchup: edge Buffalo Bills (0.77 SD) — vs-average expectation: Buffalo Bills offense vs New England Patriots defense +0.31 pts, New England Patriots offense vs Buffalo Bills defense -0.10 pts
2. Field position / special teams: edge Buffalo Bills (0.65 SD) — vs-average expectation: Buffalo Bills offense vs New England Patriots defense +0.19 yds, New England Patriots offense vs Buffalo Bills defense -1.72 yds
3. Pass protection vs pass rush (sack rate): edge Buffalo Bills (0.60 SD) — vs-average expectation: Buffalo Bills offense vs New England Patriots defense +0.15 pp, New England Patriots offense vs Buffalo Bills defense +1.76 pp

Main Risk: New England Patriots's best counter: Explosive-play matchup (0.35 SD)

Model drivers (gradient-boosting SHAP, log-odds toward):
  - Elo power-rating edge (incl. home field): +0.14 → Buffalo Bills
  - Recent form vs expectation: -0.11 → New England Patriots
  - Scoring environment: +0.09 → Buffalo Bills
  - Overall offense vs defense (opp-adj EPA/play): -0.08 → New England Patriots
  - Away QB continuity: -0.08 → New England Patriots

Member P(Buffalo Bills win): elo 0.644, scoring 0.609, logistic 0.627, gbm 0.556, margin_ridge 0.627 · ensemble 0.634
  home_field: Buffalo Bills at home (scoring-model home coef 0.7 pts)
  form: last-3 margin vs expectation: Buffalo Bills +3.6, New England Patriots -4.7
  rest: rest days: Buffalo Bills 7, New England Patriots 7
  qb_continuity: starter share of season attempts: Buffalo Bills 100%, New England Patriots 100%
  injuries: see injury report in context (pre-game ESPN feed; not a model input)
  injury_report: BUF: T.J. Sanders (DT, Doubtful), Tyrell Shavers (WR, Out); NE: Channing Canada (CB, Out)
  weather (forecast): {'temperature': 49, 'gust': 4, 'precipitation': 25, 'conditionId': 'Clear'}

logged as prediction f8fa5b486c694755803354633c4ef20a (snapshot sha256 2399385b9324df77…)
```

```
New York Jets @ Chicago Bears
[BLIND pre-game prediction · model v1.0 · data cutoff 2026-10-04T10:43:44Z · kickoff 2026-10-04T17:00:00Z]

Prediction: Chicago Bears
Win Probability: New York Jets 26% / Chicago Bears 74%
Projected Score: New York Jets 19–27 Chicago Bears
Projected Margin: 8.1
Upset Probability: 26%
Confidence: 4.5/10

Key Factors:

1. Finishing drives (pts per scoring opportunity): edge Chicago Bears (1.72 SD) — vs-average expectation: Chicago Bears offense vs New York Jets defense +0.42 pts, New York Jets offense vs Chicago Bears defense -0.67 pts
2. Points-per-drive matchup: edge Chicago Bears (1.51 SD) — vs-average expectation: Chicago Bears offense vs New York Jets defense +0.34 pts, New York Jets offense vs Chicago Bears defense -0.45 pts
3. Turnover tendencies: edge Chicago Bears (1.34 SD) — vs-average expectation: Chicago Bears offense vs New York Jets defense -0.96 pp, New York Jets offense vs Chicago Bears defense +0.19 pp

Main Risk: CHI QB Caleb Williams OUT; NYJ: 4 players out/doubtful; New York Jets's best counter: Explosive-play matchup (0.67 SD); starting QB changed last game (Chicago Bears) — availability unconfirmed

Model drivers (gradient-boosting SHAP, log-odds toward):
  - Elo power-rating edge (incl. home field): +0.75 → Chicago Bears
  - Pass protection vs pass rush (sack rate): +0.19 → Chicago Bears
  - Home QB continuity: -0.16 → New York Jets
  - Scoring environment: +0.14 → Chicago Bears
  - Points-per-drive matchup: +0.14 → Chicago Bears

Member P(Chicago Bears win): elo 0.771, scoring 0.758, logistic 0.669, gbm 0.828, margin_ridge 0.655 · ensemble 0.740
  home_field: Chicago Bears at home (scoring-model home coef 0.7 pts)
  form: last-3 margin vs expectation: Chicago Bears +10.7, New York Jets +5.2
  rest: rest days: Chicago Bears 5, New York Jets 7
  qb_continuity: starter share of season attempts: Chicago Bears 38%, New York Jets 100%
  injuries: see injury report in context (pre-game ESPN feed; not a model input)
  injury_report: CHI: Cam Lewis (CB, Doubtful), Caleb Williams (QB, Out), Kyler Gordon (CB, Out); NYJ: Dylan Parham (G, Out), Adonai Mitchell (WR, Out), Kiko Mauigoa (LB, Out), Breece Hall (RB, Out)
  weather (forecast): {'temperature': 69, 'gust': 13, 'precipitation': 0, 'conditionId': '2'}

logged as prediction b0897ab066d34c8bb1088aab5a23a369 (snapshot sha256 43cc51ba9abc286d…)
```

```
Tennessee Titans @ Baltimore Ravens
[BLIND pre-game prediction · model v1.0 · data cutoff 2026-10-04T10:43:44Z · kickoff 2026-10-04T17:00:00Z]

Prediction: Baltimore Ravens
Win Probability: Tennessee Titans 19% / Baltimore Ravens 81%
Projected Score: Tennessee Titans 18–29 Baltimore Ravens
Projected Margin: 11.0
Upset Probability: 19%
Confidence: 6.6/10

Key Factors:

1. Success-rate matchup: edge Baltimore Ravens (2.10 SD) — vs-average expectation: Baltimore Ravens offense vs Tennessee Titans defense +3.96 pp, Tennessee Titans offense vs Baltimore Ravens defense -4.66 pp
2. Passing success matchup: edge Baltimore Ravens (1.99 SD) — vs-average expectation: Baltimore Ravens offense vs Tennessee Titans defense +4.46 pp, Tennessee Titans offense vs Baltimore Ravens defense -6.27 pp
3. Yards-per-play matchup: edge Baltimore Ravens (1.96 SD) — vs-average expectation: Baltimore Ravens offense vs Tennessee Titans defense +0.67 yds, Tennessee Titans offense vs Baltimore Ravens defense -0.59 yds

Main Risk: single-game variance (margin sigma ~ 16 pts)

Model drivers (gradient-boosting SHAP, log-odds toward):
  - Elo power-rating edge (incl. home field): +0.77 → Baltimore Ravens
  - Points-per-drive matchup: +0.14 → Baltimore Ravens
  - Scoring environment: +0.13 → Baltimore Ravens
  - Recent form vs expectation: +0.12 → Baltimore Ravens
  - Overall offense vs defense (opp-adj EPA/play): +0.11 → Baltimore Ravens

Member P(Baltimore Ravens win): elo 0.811, scoring 0.746, logistic 0.798, gbm 0.841, margin_ridge 0.797 · ensemble 0.808
  home_field: Baltimore Ravens at home (scoring-model home coef 0.7 pts)
  form: last-3 margin vs expectation: Baltimore Ravens +0.9, Tennessee Titans -3.3
  rest: rest days: Baltimore Ravens 6, Tennessee Titans 7
  qb_continuity: starter share of season attempts: Baltimore Ravens 100%, Tennessee Titans 100%
  injuries: see injury report in context (pre-game ESPN feed; not a model input)
  injury_report: BAL: no outs; TEN: no outs
  weather (forecast): {'temperature': 62, 'gust': 10, 'precipitation': 46, 'conditionId': '7'}

logged as prediction 881b0955afcd44bf8583d6341271952b (snapshot sha256 44bc339451caffac…)
```

```
Miami Dolphins @ Minnesota Vikings
[BLIND pre-game prediction · model v1.0 · data cutoff 2026-10-04T10:43:44Z · kickoff 2026-10-04T20:05:00Z]

Prediction: Minnesota Vikings
Win Probability: Miami Dolphins 22% / Minnesota Vikings 78%
Projected Score: Miami Dolphins 17–26 Minnesota Vikings
Projected Margin: 9.3
Upset Probability: 22%
Confidence: 5.7/10

Key Factors:

1. Finishing drives (pts per scoring opportunity): edge Minnesota Vikings (2.34 SD) — vs-average expectation: Minnesota Vikings offense vs Miami Dolphins defense +0.24 pts, Miami Dolphins offense vs Minnesota Vikings defense -1.24 pts
2. Success-rate matchup: edge Minnesota Vikings (2.23 SD) — vs-average expectation: Minnesota Vikings offense vs Miami Dolphins defense -0.24 pp, Miami Dolphins offense vs Minnesota Vikings defense -9.39 pp
3. Passing success matchup: edge Minnesota Vikings (2.04 SD) — vs-average expectation: Minnesota Vikings offense vs Miami Dolphins defense +0.28 pp, Miami Dolphins offense vs Minnesota Vikings defense -10.73 pp

Main Risk: starting QB changed last game (Minnesota Vikings) — availability unconfirmed

Model drivers (gradient-boosting SHAP, log-odds toward):
  - Elo power-rating edge (incl. home field): +0.79 → Minnesota Vikings
  - Pass protection vs pass rush (sack rate): +0.24 → Minnesota Vikings
  - Home QB continuity: -0.16 → Miami Dolphins
  - Points-per-drive matchup: +0.14 → Minnesota Vikings
  - Rushing success matchup: +0.13 → Minnesota Vikings

Member P(Minnesota Vikings win): elo 0.795, scoring 0.787, logistic 0.771, gbm 0.792, margin_ridge 0.738 · ensemble 0.778
  home_field: Minnesota Vikings at home (scoring-model home coef 0.7 pts)
  form: last-3 margin vs expectation: Minnesota Vikings +8.0, Miami Dolphins -13.6
  rest: rest days: Minnesota Vikings 7, Miami Dolphins 7
  qb_continuity: starter share of season attempts: Minnesota Vikings 43%, Miami Dolphins 100%
  injuries: see injury report in context (pre-game ESPN feed; not a model input)
  injury_report: MIN: Justin Jefferson (WR, Out), Brett Thorson (P, Out), Charles Demmings (CB, Out); MIA: Robert Beal Jr. (DE, Out), Caleb Douglas (WR, Out), Storm Duck (CB, Out)
  weather (forecast): {'temperature': 67, 'gust': 21, 'precipitation': 0, 'conditionId': '2'}

logged as prediction a2ea4062d5304ea1aa07c6ce163595f1 (snapshot sha256 02d576b87488e137…)
```

```
Denver Broncos @ San Francisco 49ers
[BLIND pre-game prediction · model v1.0 · data cutoff 2026-10-04T10:43:44Z · kickoff 2026-10-04T20:25:00Z]

Prediction: San Francisco 49ers
Win Probability: Denver Broncos 41% / San Francisco 49ers 59%
Projected Score: Denver Broncos 22–25 San Francisco 49ers
Projected Margin: 3.1
Upset Probability: 41%
Confidence: 2.3/10

Key Factors:

1. Yards-per-play matchup: edge San Francisco 49ers (1.84 SD) — vs-average expectation: San Francisco 49ers offense vs Denver Broncos defense +0.67 yds, Denver Broncos offense vs San Francisco 49ers defense -0.52 yds
2. Passing success matchup: edge San Francisco 49ers (1.70 SD) — vs-average expectation: San Francisco 49ers offense vs Denver Broncos defense +5.79 pp, Denver Broncos offense vs San Francisco 49ers defense -3.40 pp
3. Success-rate matchup: edge San Francisco 49ers (1.24 SD) — vs-average expectation: San Francisco 49ers offense vs Denver Broncos defense +4.41 pp, Denver Broncos offense vs San Francisco 49ers defense -0.69 pp

Main Risk: Denver Broncos's best counter: Pass protection vs pass rush (sack rate) (0.39 SD)

Model drivers (gradient-boosting SHAP, log-odds toward):
  - Scoring environment: +0.16 → San Francisco 49ers
  - Points-per-drive matchup: +0.15 → San Francisco 49ers
  - Recent form vs expectation: -0.12 → Denver Broncos
  - Overall offense vs defense (opp-adj EPA/play): +0.12 → San Francisco 49ers
  - Pass protection vs pass rush (sack rate): -0.12 → Denver Broncos

Member P(San Francisco 49ers win): elo 0.564, scoring 0.661, logistic 0.600, gbm 0.663, margin_ridge 0.624 · ensemble 0.588
  home_field: San Francisco 49ers at home (scoring-model home coef 0.7 pts)
  form: last-3 margin vs expectation: San Francisco 49ers +10.7, Denver Broncos -5.4
  rest: rest days: San Francisco 49ers 7, Denver Broncos 6
  qb_continuity: starter share of season attempts: San Francisco 49ers 100%, Denver Broncos 100%
  injuries: see injury report in context (pre-game ESPN feed; not a model input)
  injury_report: SF: Mykel Williams (DE, Out); DEN: Dondrea Tillman (LB, Out), Jonathon Cooper (LB, Out), Nick Gargiulo (G, Out)
  weather (forecast): {'temperature': 92, 'gust': 8, 'precipitation': 0, 'conditionId': '2'}

logged as prediction f676cceb58314f67ae3d3f8786b38335 (snapshot sha256 bd41706a098733b4…)
```

```
Kansas City Chiefs @ Las Vegas Raiders
[BLIND pre-game prediction · model v1.0 · data cutoff 2026-10-04T10:43:44Z · kickoff 2026-10-04T20:25:00Z]

Prediction: Kansas City Chiefs
Win Probability: Kansas City Chiefs 62% / Las Vegas Raiders 38%
Projected Score: Kansas City Chiefs 24–19 Las Vegas Raiders
Projected Margin: 5.4
Upset Probability: 38%
Confidence: 2.6/10

Key Factors:

1. Rushing success matchup: edge Kansas City Chiefs (2.36 SD) — vs-average expectation: Las Vegas Raiders offense vs Kansas City Chiefs defense -8.86 pp, Kansas City Chiefs offense vs Las Vegas Raiders defense +4.08 pp
2. Success-rate matchup: edge Kansas City Chiefs (2.28 SD) — vs-average expectation: Las Vegas Raiders offense vs Kansas City Chiefs defense -5.14 pp, Kansas City Chiefs offense vs Las Vegas Raiders defense +4.15 pp
3. Explosive-play matchup: edge Kansas City Chiefs (2.09 SD) — vs-average expectation: Las Vegas Raiders offense vs Kansas City Chiefs defense -4.11 pp, Kansas City Chiefs offense vs Las Vegas Raiders defense +0.67 pp

Main Risk: Las Vegas Raiders's best counter: Field position / special teams (0.42 SD)

Model drivers (gradient-boosting SHAP, log-odds toward):
  - Elo power-rating edge (incl. home field): -0.50 → Kansas City Chiefs
  - Explosive-play matchup: -0.30 → Kansas City Chiefs
  - Rushing success matchup: -0.29 → Kansas City Chiefs
  - Recent form vs expectation: +0.08 → Las Vegas Raiders
  - Pass game vs pass defense (opp-adj pass EPA/play): -0.08 → Kansas City Chiefs

Member P(Las Vegas Raiders win): elo 0.409, scoring 0.431, logistic 0.328, gbm 0.204, margin_ridge 0.337 · ensemble 0.376
  home_field: Las Vegas Raiders at home (scoring-model home coef 0.7 pts)
  form: last-3 margin vs expectation: Las Vegas Raiders +15.4, Kansas City Chiefs +10.8
  rest: rest days: Las Vegas Raiders 7, Kansas City Chiefs 7
  qb_continuity: starter share of season attempts: Las Vegas Raiders 100%, Kansas City Chiefs 100%
  injuries: see injury report in context (pre-game ESPN feed; not a model input)
  injury_report: LV: Jackson Powers-Johnson (G, Out); KC: Josh Simmons (OT, Out), Omarr Norman-Lott (DT, Out)
  weather (forecast): {'temperature': 92, 'gust': 6, 'precipitation': 0, 'conditionId': '1'}

logged as prediction 94346dfc6e594312968e36104433029e (snapshot sha256 064cee7bae545a93…)
```

```
Los Angeles Chargers @ Seattle Seahawks
[BLIND pre-game prediction · model v1.0 · data cutoff 2026-10-04T10:43:44Z · kickoff 2026-10-04T20:25:00Z]

Prediction: Seattle Seahawks
Win Probability: Los Angeles Chargers 21% / Seattle Seahawks 79%
Projected Score: Los Angeles Chargers 15–26 Seattle Seahawks
Projected Margin: 10.3
Upset Probability: 21%
Confidence: 5.9/10

Key Factors:

1. Explosive-play matchup: edge Seattle Seahawks (1.61 SD) — vs-average expectation: Seattle Seahawks offense vs Los Angeles Chargers defense +0.31 pp, Los Angeles Chargers offense vs Seattle Seahawks defense -3.35 pp
2. Yards-per-play matchup: edge Seattle Seahawks (1.54 SD) — vs-average expectation: Seattle Seahawks offense vs Los Angeles Chargers defense +0.40 yds, Los Angeles Chargers offense vs Seattle Seahawks defense -0.60 yds
3. Points-per-drive matchup: edge Seattle Seahawks (1.47 SD) — vs-average expectation: Seattle Seahawks offense vs Los Angeles Chargers defense -0.10 pts, Los Angeles Chargers offense vs Seattle Seahawks defense -0.87 pts

Main Risk: starting QB changed last game (Seattle Seahawks) — availability unconfirmed

Model drivers (gradient-boosting SHAP, log-odds toward):
  - Elo power-rating edge (incl. home field): +0.74 → Seattle Seahawks
  - Pass protection vs pass rush (sack rate): +0.23 → Seattle Seahawks
  - Home QB continuity: -0.15 → Los Angeles Chargers
  - Points-per-drive matchup: +0.14 → Seattle Seahawks
  - Opponent-adjusted scoring margin: +0.09 → Seattle Seahawks

Member P(Seattle Seahawks win): elo 0.813, scoring 0.774, logistic 0.755, gbm 0.781, margin_ridge 0.746 · ensemble 0.791
  home_field: Seattle Seahawks at home (scoring-model home coef 0.7 pts)
  form: last-3 margin vs expectation: Seattle Seahawks +1.1, Los Angeles Chargers -11.9
  rest: rest days: Seattle Seahawks 7, Los Angeles Chargers 7
  qb_continuity: starter share of season attempts: Seattle Seahawks 48%, Los Angeles Chargers 100%
  injuries: see injury report in context (pre-game ESPN feed; not a model input)
  injury_report: SEA: Zach Charbonnet (RB, Out); LAC: Dalvin Tomlinson (DT, Out), Brenen Thompson (WR, Out), Charlie Kolar (TE, Out)
  weather (forecast): {'temperature': 67, 'gust': 7, 'precipitation': 0, 'conditionId': '11'}

logged as prediction b72e707a8f3f44a8b2b064bb8f4c8f3f (snapshot sha256 7c684ea51d41238e…)
```

```
Detroit Lions @ Carolina Panthers
[BLIND pre-game prediction · model v1.0 · data cutoff 2026-10-04T10:43:44Z · kickoff 2026-10-05T00:20:00Z]

Prediction: Detroit Lions
Win Probability: Detroit Lions 58% / Carolina Panthers 42%
Projected Score: Detroit Lions 26–23 Carolina Panthers
Projected Margin: 2.3
Upset Probability: 42%
Confidence: 1.0/10

Key Factors:

1. Passing success matchup: edge Detroit Lions (0.68 SD) — vs-average expectation: Carolina Panthers offense vs Detroit Lions defense +0.91 pp, Detroit Lions offense vs Carolina Panthers defense +4.52 pp
2. Yards-per-play matchup: edge Detroit Lions (0.56 SD) — vs-average expectation: Carolina Panthers offense vs Detroit Lions defense +0.18 yds, Detroit Lions offense vs Carolina Panthers defense +0.54 yds
3. Turnover tendencies: edge Carolina Panthers (0.53 SD) — vs-average expectation: Carolina Panthers offense vs Detroit Lions defense -0.39 pp, Detroit Lions offense vs Carolina Panthers defense +0.07 pp

Main Risk: ensemble members disagree on the winner; Carolina Panthers's best counter: Turnover tendencies (0.53 SD)

Model drivers (gradient-boosting SHAP, log-odds toward):
  - Elo power-rating edge (incl. home field): -0.43 → Detroit Lions
  - Scoring environment: -0.14 → Detroit Lions
  - Pass protection vs pass rush (sack rate): -0.09 → Detroit Lions
  - Opponent-adjusted scoring margin: +0.08 → Carolina Panthers
  - Recent form vs expectation: -0.08 → Detroit Lions

Member P(Carolina Panthers win): elo 0.401, scoring 0.514, logistic 0.415, gbm 0.362, margin_ridge 0.469 · ensemble 0.419
  home_field: Carolina Panthers at home (scoring-model home coef 0.7 pts)
  form: last-3 margin vs expectation: Carolina Panthers +4.4, Detroit Lions -4.1
  rest: rest days: Carolina Panthers 7, Detroit Lions 7
  qb_continuity: starter share of season attempts: Carolina Panthers 100%, Detroit Lions 100%
  injuries: see injury report in context (pre-game ESPN feed; not a model input)
  injury_report: CAR: Damien Lewis (G, Out); DET: Ben Bartch (G, Out), Brian Branch (S, Out)
  weather (forecast): {'temperature': 66, 'gust': 6, 'precipitation': 60, 'conditionId': '18'}

logged as prediction f5167f050f394ea590e51ff9f1d9fa0f (snapshot sha256 3832299c3e76d5f6…)
```

```
Atlanta Falcons @ New Orleans Saints
[BLIND pre-game prediction · model v1.0 · data cutoff 2026-10-04T10:43:44Z · kickoff 2026-10-06T00:15:00Z]

Prediction: New Orleans Saints
Win Probability: Atlanta Falcons 47% / New Orleans Saints 53%
Projected Score: Atlanta Falcons 21–22 New Orleans Saints
Projected Margin: 1.3
Upset Probability: 47%
Confidence: 1.0/10

Key Factors:

1. Rushing success matchup: edge Atlanta Falcons (1.57 SD) — vs-average expectation: New Orleans Saints offense vs Atlanta Falcons defense -3.24 pp, Atlanta Falcons offense vs New Orleans Saints defense +5.35 pp
2. Turnover tendencies: edge New Orleans Saints (1.37 SD) — vs-average expectation: New Orleans Saints offense vs Atlanta Falcons defense -0.30 pp, Atlanta Falcons offense vs New Orleans Saints defense +0.88 pp
3. Success-rate matchup: edge Atlanta Falcons (1.01 SD) — vs-average expectation: New Orleans Saints offense vs Atlanta Falcons defense -1.23 pp, Atlanta Falcons offense vs New Orleans Saints defense +2.86 pp

Main Risk: Atlanta Falcons's best counter: Rushing success matchup (1.57 SD); starting QB changed last game (Atlanta Falcons) — availability unconfirmed

Model drivers (gradient-boosting SHAP, log-odds toward):
  - Rushing success matchup: -0.29 → Atlanta Falcons
  - Away QB continuity: +0.16 → New Orleans Saints
  - Elo power-rating edge (incl. home field): -0.11 → Atlanta Falcons
  - Pass protection vs pass rush (sack rate): -0.11 → Atlanta Falcons
  - Recent form vs expectation: +0.10 → New Orleans Saints

Member P(New Orleans Saints win): elo 0.525, scoring 0.617, logistic 0.523, gbm 0.519, margin_ridge 0.541 · ensemble 0.530
  home_field: New Orleans Saints at home (scoring-model home coef 0.7 pts)
  form: last-3 margin vs expectation: New Orleans Saints +2.7, Atlanta Falcons -3.8
  rest: rest days: New Orleans Saints 8, Atlanta Falcons 11
  qb_continuity: starter share of season attempts: New Orleans Saints 100%, Atlanta Falcons 39%
  injuries: see injury report in context (pre-game ESPN feed; not a model input)
  injury_report: NO: Anfernee Jennings (LB, Out), Carl Granderson (DE, Out), Kaden Elliss (LB, Out); ATL: no outs
  weather (forecast): {'temperature': 79, 'gust': 9, 'precipitation': 37, 'conditionId': '7'}

logged as prediction ee72908a33434b14b86c70ede7eb1590 (snapshot sha256 fe0292ebec8ab582…)
```

## Summary

| Game | Kickoff (UTC) | Pick | Win % | Projected score | Upset % | Confidence | Flags |
|---|---|---|---|---|---|---|---|
| Indianapolis Colts @ Washington Commanders (N) | 10-04 13:30 | Indianapolis Colts | 55% | Indianapolis Colts 26–24 Washington Commanders | 45% | 1.0 | injuries: WSH QB Jayden Daniels OUT, WSH: 4 players out/doubtful |
| Arizona Cardinals @ New York Giants | 10-04 17:00 | New York Giants | 55% | Arizona Cardinals 22–23 New York Giants | 45% | 1.0 |  |
| Dallas Cowboys @ Houston Texans | 10-04 17:00 | Houston Texans | 65% | Dallas Cowboys 21–26 Houston Texans | 35% | 3.6 |  |
| Green Bay Packers @ Tampa Bay Buccaneers | 10-04 17:00 | Green Bay Packers | 52% | Green Bay Packers 24–23 Tampa Bay Buccaneers | 48% | 1.0 |  |
| Jacksonville Jaguars @ Cincinnati Bengals | 10-04 17:00 | Jacksonville Jaguars | 56% | Jacksonville Jaguars 25–24 Cincinnati Bengals | 44% | 1.3 |  |
| Los Angeles Rams @ Philadelphia Eagles | 10-04 17:00 | Los Angeles Rams | 53% | Los Angeles Rams 23–21 Philadelphia Eagles | 47% | 1.0 | injuries: PHI: 5 players out/doubtful |
| New England Patriots @ Buffalo Bills | 10-04 17:00 | Buffalo Bills | 63% | New England Patriots 23–26 Buffalo Bills | 37% | 3.4 |  |
| New York Jets @ Chicago Bears | 10-04 17:00 | Chicago Bears | 74% | New York Jets 19–27 Chicago Bears | 26% | 4.5 | injuries: CHI QB Caleb Williams OUT, NYJ: 4 players out/doubtful |
| Tennessee Titans @ Baltimore Ravens | 10-04 17:00 | Baltimore Ravens | 81% | Tennessee Titans 18–29 Baltimore Ravens | 19% | 6.6 |  |
| Miami Dolphins @ Minnesota Vikings | 10-04 20:05 | Minnesota Vikings | 78% | Miami Dolphins 17–26 Minnesota Vikings | 22% | 5.7 |  |
| Denver Broncos @ San Francisco 49ers | 10-04 20:25 | San Francisco 49ers | 59% | Denver Broncos 22–25 San Francisco 49ers | 41% | 2.3 |  |
| Kansas City Chiefs @ Las Vegas Raiders | 10-04 20:25 | Kansas City Chiefs | 62% | Kansas City Chiefs 24–19 Las Vegas Raiders | 38% | 2.6 |  |
| Los Angeles Chargers @ Seattle Seahawks | 10-04 20:25 | Seattle Seahawks | 79% | Los Angeles Chargers 15–26 Seattle Seahawks | 21% | 5.9 |  |
| Detroit Lions @ Carolina Panthers | 10-05 00:20 | Detroit Lions | 58% | Detroit Lions 26–23 Carolina Panthers | 42% | 1.0 |  |
| Atlanta Falcons @ New Orleans Saints | 10-06 00:15 | New Orleans Saints | 53% | Atlanta Falcons 21–22 New Orleans Saints | 47% | 1.0 |  |

**Most confident:** Baltimore Ravens (81%), Seattle Seahawks (79%), Minnesota Vikings (78%), Chicago Bears (74%)

**Closest:** Green Bay Packers @ Tampa Bay Buccaneers (52%), Atlanta Falcons @ New Orleans Saints (53%), Los Angeles Rams @ Philadelphia Eagles (53%), Indianapolis Colts @ Washington Commanders (55%)

**Upset candidates (underdog ≥ 35%):** Tampa Bay Buccaneers 48%, Atlanta Falcons 47%, Philadelphia Eagles 47%, Washington Commanders 45%, Arizona Cardinals 45%, Cincinnati Bengals 44%, Carolina Panthers 42%, Denver Broncos 41%, Las Vegas Raiders 38%, New England Patriots 37%

**Highest model disagreement:** Kansas City Chiefs @ Las Vegas Raiders, Green Bay Packers @ Tampa Bay Buccaneers, New York Jets @ Chicago Bears