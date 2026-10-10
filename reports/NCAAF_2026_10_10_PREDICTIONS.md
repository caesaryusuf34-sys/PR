# NCAAF — Saturday 2026-10-10 (games not yet started)

- Model **v1.1** · predictions frozen 2026-10-10T19:03:32Z (before every kickoff listed) · all inputs are pre-game; results/injuries after the cutoff never enter
- Walk-forward validation (3000 out-of-sample games): accuracy 76.3%, log loss 0.476, Brier 0.158, margin MAE 12.6
- Not predicted (already started or final at freeze): New Hampshire @ North Carolina A&T; Stonehill @ Sacred Heart; Bryant @ Brown; Dartmouth @ Yale; Robert Morris @ Morgan State; Princeton @ Wagner; Morehead State @ Dayton; East Tennessee State @ The Citadel; Davidson @ Presbyterian; Indiana @ Nebraska; UCF @ Oklahoma State; Texas A&M @ Missouri; Arizona @ West Virginia; North Carolina @ Pittsburgh; Wake Forest @ NC State; Tulane @ Army; Sacramento State @ Bowling Green; Bucknell @ Georgetown; Ball State @ Northwestern; South Carolina @ Florida; Harvard @ Cornell; Youngstown State @ Indiana State; American International @ Central Connecticut; Long Island University @ Duquesne; Franklin Pierce @ Delaware State; Howard @ Pennsylvania; William & Mary @ North Carolina Central; San Diego @ Butler; Samford @ Chattanooga; South Carolina State @ Charleston Southern; Bethune-Cookman @ Alabama A&M; Old Dominion @ App State; Rice @ East Carolina; Wofford @ Elon; Southern Illinois @ South Dakota; Virginia Lynchburg @ Norfolk State; Drake @ Valparaiso; Stetson @ St. Thomas; Southeast Missouri State @ Lindenwood; Miami (OH) @ Massachusetts; Idaho @ Weber State; Arkansas Baptist @ Alcorn State; Western Carolina @ Mercer; Mississippi Valley State @ Texas Southern
- Betting lines are not model inputs. Injury reports are shown as risk context only (no historical injury data to train on).

## Games

```
Texas @ Oklahoma (neutral site)
[BLIND pre-game prediction · model v1.1 · data cutoff 2026-10-10T19:03:32Z · kickoff 2026-10-10T19:30:00Z]

Prediction: Texas
Win Probability: Texas 72% / Oklahoma 28%
Projected Score: Texas 24–16 Oklahoma
Projected Margin: 8.1
Upset Probability: 28%
Confidence: 4.8/10

Key Factors:

1. Field position / special teams: edge Texas (1.72 SD) — vs-average expectation: Oklahoma offense vs Texas defense -2.31 yds, Texas offense vs Oklahoma defense +3.89 yds
2. Points-per-drive matchup: edge Texas (1.36 SD) — vs-average expectation: Oklahoma offense vs Texas defense -1.19 pts, Texas offense vs Oklahoma defense -0.19 pts
3. Turnover tendencies: edge Texas (1.28 SD) — vs-average expectation: Oklahoma offense vs Texas defense +0.81 pp, Texas offense vs Oklahoma defense -0.26 pp

Main Risk: Oklahoma's best counter: Pass protection vs pass rush (sack rate) (0.71 SD)

Model drivers (gradient-boosting SHAP, log-odds toward):
  - Opponent-adjusted scoring margin: -1.16 → Texas
  - Elo power-rating edge (incl. home field): -0.21 → Texas
  - Scoring environment: -0.12 → Texas
  - Field position / special teams: -0.11 → Texas
  - Yards-per-play matchup: +0.09 → Oklahoma

Member P(Oklahoma win): elo 0.249, scoring 0.221, logistic 0.300, gbm 0.353, margin_ridge 0.301 · ensemble 0.284
  home_field: neutral site
  form: last-3 margin vs expectation: Oklahoma -7.5, Texas +4.6
  rest: rest days: Oklahoma 14, Texas 14
  qb_continuity: starter share of season attempts: Oklahoma 100%, Texas 100%
  injuries: Data unavailable (no public college injury feed)
  weather (forecast): {'temperature': 89, 'gust': 14, 'precipitation': 0, 'conditionId': '1'}

logged as prediction f64e5211c2444fab97ee71c7e68ab0cf (snapshot sha256 5fe49dff158fa43d…)
```

```
Ole Miss @ Vanderbilt
[BLIND pre-game prediction · model v1.1 · data cutoff 2026-10-10T19:03:32Z · kickoff 2026-10-10T19:30:00Z]

Prediction: Ole Miss
Win Probability: Ole Miss 53% / Vanderbilt 47%
Projected Score: Ole Miss 29–28 Vanderbilt
Projected Margin: 1.1
Upset Probability: 47%
Confidence: 1.0/10

Key Factors:

1. Finishing drives (pts per scoring opportunity): edge Ole Miss (1.07 SD) — vs-average expectation: Vanderbilt offense vs Ole Miss defense +0.11 pts, Ole Miss offense vs Vanderbilt defense +0.93 pts
2. Pass game vs pass defense (opp-adj pass EPA/play): edge Ole Miss (0.91 SD) — vs-average expectation: Vanderbilt offense vs Ole Miss defense +0.00 EPA/play, Ole Miss offense vs Vanderbilt defense +0.21 EPA/play
3. Field position / special teams: edge Vanderbilt (0.89 SD) — vs-average expectation: Vanderbilt offense vs Ole Miss defense +0.28 yds, Ole Miss offense vs Vanderbilt defense -3.84 yds

Main Risk: ensemble members disagree on the winner; Vanderbilt's best counter: Field position / special teams (0.89 SD); starting QB changed last game (Vanderbilt) — availability unconfirmed

Model drivers (gradient-boosting SHAP, log-odds toward):
  - Opponent-adjusted scoring margin: -0.77 → Ole Miss
  - Field position / special teams: +0.12 → Vanderbilt
  - Elo power-rating edge (incl. home field): -0.10 → Ole Miss
  - Overall offense vs defense (opp-adj EPA/play): -0.07 → Ole Miss
  - Rest advantage: -0.07 → Ole Miss

Member P(Vanderbilt win): elo 0.469, scoring 0.503, logistic 0.460, gbm 0.463, margin_ridge 0.463 · ensemble 0.474
  home_field: Vanderbilt at home (scoring-model home coef 5.8 pts)
  form: last-3 margin vs expectation: Vanderbilt -10.3, Ole Miss -8.6
  rest: rest days: Vanderbilt 7, Ole Miss 14
  qb_continuity: starter share of season attempts: Vanderbilt 30%, Ole Miss 100%
  injuries: Data unavailable (no public college injury feed)
  weather (forecast): {'temperature': 63, 'gust': 18, 'precipitation': 57, 'conditionId': '18'}

logged as prediction d3fdd848e1a040ada3319e5bfdbba5d6 (snapshot sha256 59284971643b7ab2…)
```

```
Houston @ Kansas State
[BLIND pre-game prediction · model v1.1 · data cutoff 2026-10-10T19:03:32Z · kickoff 2026-10-10T19:30:00Z]

Prediction: Kansas State
Win Probability: Houston 47% / Kansas State 53%
Projected Score: Houston 28–30 Kansas State
Projected Margin: 1.6
Upset Probability: 47%
Confidence: 1.0/10

Key Factors:

1. Turnover tendencies: edge Houston (1.55 SD) — vs-average expectation: Kansas State offense vs Houston defense +0.77 pp, Houston offense vs Kansas State defense -0.54 pp
2. Field position / special teams: edge Houston (1.53 SD) — vs-average expectation: Kansas State offense vs Houston defense -3.68 yds, Houston offense vs Kansas State defense +1.77 yds
3. Finishing drives (pts per scoring opportunity): edge Kansas State (0.89 SD) — vs-average expectation: Kansas State offense vs Houston defense +1.01 pts, Houston offense vs Kansas State defense +0.04 pts

Main Risk: ensemble members disagree on the winner; Houston's best counter: Turnover tendencies (1.55 SD)

Model drivers (gradient-boosting SHAP, log-odds toward):
  - Opponent-adjusted scoring margin: -0.74 → Houston
  - Finishing drives (pts per scoring opportunity): -0.14 → Houston
  - Rushing success matchup: -0.13 → Houston
  - Rest advantage: +0.09 → Kansas State
  - Field position / special teams: -0.08 → Houston

Member P(Kansas State win): elo 0.586, scoring 0.546, logistic 0.522, gbm 0.435, margin_ridge 0.534 · ensemble 0.526
  home_field: Kansas State at home (scoring-model home coef 5.8 pts)
  form: last-3 margin vs expectation: Kansas State +7.2, Houston +7.3
  rest: rest days: Kansas State 13, Houston 7
  qb_continuity: starter share of season attempts: Kansas State 100%, Houston 100%
  injuries: Data unavailable (no public college injury feed)
  weather (forecast): {'temperature': 84, 'gust': 30, 'precipitation': 0, 'conditionId': '1'}

logged as prediction 4f364b8dcf9e4e0cb014c2e110182cdf (snapshot sha256 8cbab8623b0aaed1…)
```

```
Duke @ Georgia Tech
[BLIND pre-game prediction · model v1.1 · data cutoff 2026-10-10T19:03:32Z · kickoff 2026-10-10T19:30:00Z]

Prediction: Duke
Win Probability: Duke 65% / Georgia Tech 35%
Projected Score: Duke 31–24 Georgia Tech
Projected Margin: 6.6
Upset Probability: 35%
Confidence: 3.5/10

Key Factors:

1. Field position / special teams: edge Duke (1.52 SD) — vs-average expectation: Georgia Tech offense vs Duke defense -3.23 yds, Duke offense vs Georgia Tech defense +2.18 yds
2. Turnover tendencies: edge Duke (1.50 SD) — vs-average expectation: Georgia Tech offense vs Duke defense +0.26 pp, Duke offense vs Georgia Tech defense -1.01 pp
3. Points-per-drive matchup: edge Duke (1.45 SD) — vs-average expectation: Georgia Tech offense vs Duke defense -0.30 pts, Duke offense vs Georgia Tech defense +0.78 pts

Main Risk: Georgia Tech's best counter: Passing success matchup (0.43 SD)

Model drivers (gradient-boosting SHAP, log-odds toward):
  - Opponent-adjusted scoring margin: -1.13 → Duke
  - Elo power-rating edge (incl. home field): -0.43 → Duke
  - Pass protection vs pass rush (sack rate): -0.18 → Duke
  - Field position / special teams: -0.10 → Duke
  - Points-per-drive matchup: +0.09 → Georgia Tech

Member P(Georgia Tech win): elo 0.223, scoring 0.355, logistic 0.317, gbm 0.319, margin_ridge 0.356 · ensemble 0.352
  home_field: Georgia Tech at home (scoring-model home coef 5.8 pts)
  form: last-3 margin vs expectation: Georgia Tech -0.9, Duke +16.3
  rest: rest days: Georgia Tech 13, Duke 14
  qb_continuity: starter share of season attempts: Georgia Tech 100%, Duke 100%
  injuries: Data unavailable (no public college injury feed)
  weather (forecast): {'temperature': 65, 'gust': 29, 'precipitation': 70, 'conditionId': '18'}

logged as prediction 703d759b669646119cdd75bd353f786b (snapshot sha256 ac79d675b906ff0a…)
```

```
Stanford @ Notre Dame
[BLIND pre-game prediction · model v1.1 · data cutoff 2026-10-10T19:03:32Z · kickoff 2026-10-10T19:30:00Z]

Prediction: Notre Dame
Win Probability: Stanford 1% / Notre Dame 99%
Projected Score: Stanford 8–51 Notre Dame
Projected Margin: 42.4
Upset Probability: 1%
Confidence: 9.7/10

Key Factors:

1. Success-rate matchup: edge Notre Dame (3.09 SD) — vs-average expectation: Notre Dame offense vs Stanford defense +15.56 pp, Stanford offense vs Notre Dame defense -10.82 pp
2. Points-per-drive matchup: edge Notre Dame (3.04 SD) — vs-average expectation: Notre Dame offense vs Stanford defense +1.61 pts, Stanford offense vs Notre Dame defense -1.26 pts
3. Passing success matchup: edge Notre Dame (2.99 SD) — vs-average expectation: Notre Dame offense vs Stanford defense +16.29 pp, Stanford offense vs Notre Dame defense -11.18 pp

Main Risk: single-game variance (margin sigma ~ 16 pts)

Model drivers (gradient-boosting SHAP, log-odds toward):
  - Opponent-adjusted scoring margin: +2.61 → Notre Dame
  - Elo power-rating edge (incl. home field): +0.37 → Notre Dame
  - Rushing success matchup: +0.30 → Notre Dame
  - Pass protection vs pass rush (sack rate): +0.22 → Notre Dame
  - Success-rate matchup: +0.22 → Notre Dame

Member P(Notre Dame win): elo 0.954, scoring 0.996, logistic 0.989, gbm 0.994, margin_ridge 0.994 · ensemble 0.995
  home_field: Notre Dame at home (scoring-model home coef 5.8 pts)
  form: last-3 margin vs expectation: Notre Dame +6.0, Stanford -19.0
  rest: rest days: Notre Dame 7, Stanford 7
  qb_continuity: starter share of season attempts: Notre Dame 100%, Stanford 100%
  injuries: Data unavailable (no public college injury feed)
  weather (forecast): {'temperature': 77, 'gust': 20, 'precipitation': 7, 'conditionId': '3'}

logged as prediction 946f7377137941e9a1e6ac2ee8310069 (snapshot sha256 764435bdef065cc8…)
```

```
Virginia Tech @ California
[BLIND pre-game prediction · model v1.1 · data cutoff 2026-10-10T19:03:32Z · kickoff 2026-10-10T19:30:00Z]

Prediction: Virginia Tech
Win Probability: Virginia Tech 78% / California 22%
Projected Score: Virginia Tech 34–21 California
Projected Margin: 12.8
Upset Probability: 22%
Confidence: 5.0/10

Key Factors:

1. Pass protection vs pass rush (sack rate): edge Virginia Tech (2.96 SD) — vs-average expectation: California offense vs Virginia Tech defense +8.04 pp, Virginia Tech offense vs California defense -1.46 pp
2. Run game vs run defense (opp-adj rush EPA/play): edge Virginia Tech (1.98 SD) — vs-average expectation: California offense vs Virginia Tech defense -0.12 EPA/play, Virginia Tech offense vs California defense +0.26 EPA/play
3. Rushing success matchup: edge Virginia Tech (1.87 SD) — vs-average expectation: California offense vs Virginia Tech defense -7.84 pp, Virginia Tech offense vs California defense +7.21 pp

Main Risk: California's best counter: Turnover tendencies (0.17 SD); starting QB changed last game (California) — availability unconfirmed

Model drivers (gradient-boosting SHAP, log-odds toward):
  - Opponent-adjusted scoring margin: -1.27 → Virginia Tech
  - Rushing success matchup: -0.20 → Virginia Tech
  - Pass protection vs pass rush (sack rate): -0.18 → Virginia Tech
  - Elo power-rating edge (incl. home field): -0.15 → Virginia Tech
  - Yards-per-play matchup: -0.10 → Virginia Tech

Member P(California win): elo 0.338, scoring 0.346, logistic 0.154, gbm 0.201, margin_ridge 0.164 · ensemble 0.219
  home_field: California at home (scoring-model home coef 5.8 pts)
  form: last-3 margin vs expectation: California +5.6, Virginia Tech +4.6
  rest: rest days: California 7, Virginia Tech 7
  qb_continuity: starter share of season attempts: California 19%, Virginia Tech 100%
  injuries: Data unavailable (no public college injury feed)
  weather (forecast): {'temperature': 63, 'gust': 23, 'precipitation': 0, 'conditionId': '6'}

logged as prediction 6f02ded51ece44c8b54bcf625c5150ca (snapshot sha256 e9aa9d1f6aa28241…)
```

```
Illinois @ Michigan State
[BLIND pre-game prediction · model v1.1 · data cutoff 2026-10-10T19:03:32Z · kickoff 2026-10-10T19:30:00Z]

Prediction: Michigan State
Win Probability: Illinois 50% / Michigan State 50%
Projected Score: Illinois 27–26 Michigan State
Projected Margin: 0.7
Upset Probability: 50%
Confidence: 1.0/10

Key Factors:

1. Explosive-play matchup: edge Illinois (1.06 SD) — vs-average expectation: Michigan State offense vs Illinois defense -3.05 pp, Illinois offense vs Michigan State defense +0.22 pp
2. Yards-per-play matchup: edge Illinois (0.88 SD) — vs-average expectation: Michigan State offense vs Illinois defense -0.72 yds, Illinois offense vs Michigan State defense +0.17 yds
3. Pass game vs pass defense (opp-adj pass EPA/play): edge Illinois (0.61 SD) — vs-average expectation: Michigan State offense vs Illinois defense +0.07 EPA/play, Illinois offense vs Michigan State defense +0.19 EPA/play

Main Risk: ensemble members disagree on the winner; Illinois's best counter: Explosive-play matchup (1.06 SD)

Model drivers (gradient-boosting SHAP, log-odds toward):
  - Opponent-adjusted scoring margin: -0.75 → Illinois
  - Elo power-rating edge (incl. home field): -0.18 → Illinois
  - Yards-per-play matchup: -0.15 → Illinois
  - Explosive-play matchup: -0.08 → Illinois
  - Field position / special teams: +0.04 → Michigan State

Member P(Michigan State win): elo 0.455, scoring 0.556, logistic 0.479, gbm 0.458, margin_ridge 0.484 · ensemble 0.502
  home_field: Michigan State at home (scoring-model home coef 5.8 pts)
  form: last-3 margin vs expectation: Michigan State -12.6, Illinois -2.4
  rest: rest days: Michigan State 7, Illinois 6
  qb_continuity: starter share of season attempts: Michigan State 100%, Illinois 100%
  injuries: Data unavailable (no public college injury feed)
  weather (forecast): {'temperature': 77, 'gust': 23, 'precipitation': 0, 'conditionId': '3'}

logged as prediction 18b6fc92a8fc44dfb3f69d10a1caf84c (snapshot sha256 f6cd4f1abc7166a3…)
```

```
UCLA @ Oregon
[BLIND pre-game prediction · model v1.1 · data cutoff 2026-10-10T19:03:32Z · kickoff 2026-10-10T19:30:00Z]

Prediction: Oregon
Win Probability: UCLA 30% / Oregon 70%
Projected Score: UCLA 26–34 Oregon
Projected Margin: 7.6
Upset Probability: 30%
Confidence: 4.0/10

Key Factors:

1. Run game vs run defense (opp-adj rush EPA/play): edge UCLA (1.70 SD) — vs-average expectation: Oregon offense vs UCLA defense -0.05 EPA/play, UCLA offense vs Oregon defense +0.28 EPA/play
2. Pass game vs pass defense (opp-adj pass EPA/play): edge Oregon (1.42 SD) — vs-average expectation: Oregon offense vs UCLA defense +0.21 EPA/play, UCLA offense vs Oregon defense -0.23 EPA/play
3. Passing success matchup: edge Oregon (1.35 SD) — vs-average expectation: Oregon offense vs UCLA defense +4.01 pp, UCLA offense vs Oregon defense -9.17 pp

Main Risk: UCLA's best counter: Run game vs run defense (opp-adj rush EPA/play) (1.70 SD); starting QB changed last game (Oregon) — availability unconfirmed

Model drivers (gradient-boosting SHAP, log-odds toward):
  - Yards-per-play matchup: +0.16 → Oregon
  - Finishing drives (pts per scoring opportunity): -0.15 → UCLA
  - Elo power-rating edge (incl. home field): +0.14 → Oregon
  - Home QB continuity: -0.09 → UCLA
  - Pass game vs pass defense (opp-adj pass EPA/play): -0.09 → UCLA

Member P(Oregon win): elo 0.814, scoring 0.758, logistic 0.698, gbm 0.715, margin_ridge 0.675 · ensemble 0.703
  home_field: Oregon at home (scoring-model home coef 5.8 pts)
  form: last-3 margin vs expectation: Oregon +12.6, UCLA +23.0
  rest: rest days: Oregon 13, UCLA 14
  qb_continuity: starter share of season attempts: Oregon 22%, UCLA 100%
  injuries: Data unavailable (no public college injury feed)
  weather (forecast): {'temperature': 54, 'gust': 6, 'precipitation': 20, 'conditionId': '3'}

logged as prediction 80e19f3b84b4407396b27c05fd6d29fb (snapshot sha256 191bba467690a9cb…)
```

```
Charlotte @ North Texas
[BLIND pre-game prediction · model v1.1 · data cutoff 2026-10-10T19:03:32Z · kickoff 2026-10-10T19:30:00Z]

Prediction: North Texas
Win Probability: Charlotte 2% / North Texas 98%
Projected Score: Charlotte 15–50 North Texas
Projected Margin: 34.6
Upset Probability: 2%
Confidence: 9.6/10

Key Factors:

1. Run game vs run defense (opp-adj rush EPA/play): edge North Texas (2.80 SD) — vs-average expectation: North Texas offense vs Charlotte defense +0.21 EPA/play, Charlotte offense vs North Texas defense -0.43 EPA/play
2. Turnover tendencies: edge North Texas (2.68 SD) — vs-average expectation: North Texas offense vs Charlotte defense -0.29 pp, Charlotte offense vs North Texas defense +2.18 pp
3. Finishing drives (pts per scoring opportunity): edge North Texas (2.57 SD) — vs-average expectation: North Texas offense vs Charlotte defense +1.59 pts, Charlotte offense vs North Texas defense -0.91 pts

Main Risk: single-game variance (margin sigma ~ 16 pts)

Model drivers (gradient-boosting SHAP, log-odds toward):
  - Opponent-adjusted scoring margin: +2.60 → North Texas
  - Elo power-rating edge (incl. home field): +0.37 → North Texas
  - Rushing success matchup: +0.31 → North Texas
  - Success-rate matchup: +0.24 → North Texas
  - Explosive-play matchup: +0.18 → North Texas

Member P(North Texas win): elo 0.960, scoring 0.989, logistic 0.976, gbm 0.993, margin_ridge 0.981 · ensemble 0.985
  home_field: North Texas at home (scoring-model home coef 5.8 pts)
  form: last-3 margin vs expectation: North Texas +0.0, Charlotte -16.2
  rest: rest days: North Texas 8, Charlotte 7
  qb_continuity: starter share of season attempts: North Texas 100%, Charlotte 100%
  injuries: Data unavailable (no public college injury feed)
  weather (forecast): {'temperature': 91, 'gust': 15, 'precipitation': 0, 'conditionId': '1'}

logged as prediction 0a0d3e65aa3f47a78ad61a162e83789c (snapshot sha256 5b67f3fa800219ad…)
```

```
Tulsa @ Navy
[BLIND pre-game prediction · model v1.1 · data cutoff 2026-10-10T19:03:32Z · kickoff 2026-10-10T19:30:00Z]

Prediction: Navy
Win Probability: Tulsa 36% / Navy 64%
Projected Score: Tulsa 20–26 Navy
Projected Margin: 5.7
Upset Probability: 36%
Confidence: 2.6/10

Key Factors:

1. Passing success matchup: edge Tulsa (1.09 SD) — vs-average expectation: Navy offense vs Tulsa defense -9.49 pp, Tulsa offense vs Navy defense -1.51 pp
2. Rushing success matchup: edge Navy (0.98 SD) — vs-average expectation: Navy offense vs Tulsa defense +2.66 pp, Tulsa offense vs Navy defense -7.94 pp
3. Finishing drives (pts per scoring opportunity): edge Tulsa (0.85 SD) — vs-average expectation: Navy offense vs Tulsa defense -0.90 pts, Tulsa offense vs Navy defense -0.28 pts

Main Risk: Tulsa's best counter: Passing success matchup (1.09 SD); starting QB changed last game (Tulsa) — availability unconfirmed

Model drivers (gradient-boosting SHAP, log-odds toward):
  - Opponent-adjusted scoring margin: -0.62 → Tulsa
  - Elo power-rating edge (incl. home field): +0.30 → Navy
  - Rushing success matchup: +0.10 → Navy
  - Success-rate matchup: +0.09 → Navy
  - Away QB continuity: -0.08 → Tulsa

Member P(Navy win): elo 0.806, scoring 0.614, logistic 0.703, gbm 0.644, margin_ridge 0.651 · ensemble 0.640
  home_field: Navy at home (scoring-model home coef 5.8 pts)
  form: last-3 margin vs expectation: Navy -12.7, Tulsa +3.3
  rest: rest days: Navy 7, Tulsa 8
  qb_continuity: starter share of season attempts: Navy 87%, Tulsa 62%
  injuries: Data unavailable (no public college injury feed)
  weather (forecast): {'temperature': 73, 'gust': 12, 'precipitation': 44, 'conditionId': '7'}

logged as prediction dec8a535df184664961a47b5c995a03c (snapshot sha256 580e1c4e1619cc4a…)
```

```
Eastern Michigan @ Akron
[BLIND pre-game prediction · model v1.1 · data cutoff 2026-10-10T19:03:32Z · kickoff 2026-10-10T19:30:00Z]

Prediction: Eastern Michigan
Win Probability: Eastern Michigan 55% / Akron 45%
Projected Score: Eastern Michigan 27–25 Akron
Projected Margin: 2.4
Upset Probability: 45%
Confidence: 1.4/10

Key Factors:

1. Pass protection vs pass rush (sack rate): edge Eastern Michigan (1.74 SD) — vs-average expectation: Akron offense vs Eastern Michigan defense +0.36 pp, Eastern Michigan offense vs Akron defense -5.09 pp
2. Overall offense vs defense (opp-adj EPA/play): edge Eastern Michigan (0.99 SD) — vs-average expectation: Akron offense vs Eastern Michigan defense -0.09 EPA/play, Eastern Michigan offense vs Akron defense +0.07 EPA/play
3. Finishing drives (pts per scoring opportunity): edge Eastern Michigan (0.95 SD) — vs-average expectation: Akron offense vs Eastern Michigan defense -0.19 pts, Eastern Michigan offense vs Akron defense +0.52 pts

Main Risk: Akron's best counter: Field position / special teams (0.24 SD)

Model drivers (gradient-boosting SHAP, log-odds toward):
  - Opponent-adjusted scoring margin: -0.81 → Eastern Michigan
  - Pass protection vs pass rush (sack rate): -0.20 → Eastern Michigan
  - Elo power-rating edge (incl. home field): -0.16 → Eastern Michigan
  - Overall offense vs defense (opp-adj EPA/play): -0.14 → Eastern Michigan
  - Field position / special teams: +0.08 → Akron

Member P(Akron win): elo 0.401, scoring 0.491, logistic 0.402, gbm 0.386, margin_ridge 0.438 · ensemble 0.447
  home_field: Akron at home (scoring-model home coef 5.8 pts)
  form: last-3 margin vs expectation: Akron -21.8, Eastern Michigan +3.9
  rest: rest days: Akron 7, Eastern Michigan 7
  qb_continuity: starter share of season attempts: Akron 61%, Eastern Michigan 100%
  injuries: Data unavailable (no public college injury feed)
  weather (forecast): {'temperature': 73, 'gust': 17, 'precipitation': 0, 'conditionId': '3'}

logged as prediction f911e2ac53e1439998bb209854fbaf0e (snapshot sha256 c630c4fff28e0d67…)
```

```
Buffalo @ Toledo
[BLIND pre-game prediction · model v1.1 · data cutoff 2026-10-10T19:03:32Z · kickoff 2026-10-10T19:30:00Z]

Prediction: Toledo
Win Probability: Buffalo 3% / Toledo 97%
Projected Score: Buffalo 15–44 Toledo
Projected Margin: 29.8
Upset Probability: 3%
Confidence: 9.1/10

Key Factors:

1. Yards-per-play matchup: edge Toledo (2.11 SD) — vs-average expectation: Toledo offense vs Buffalo defense +1.42 yds, Buffalo offense vs Toledo defense -1.58 yds
2. Explosive-play matchup: edge Toledo (1.94 SD) — vs-average expectation: Toledo offense vs Buffalo defense +2.48 pp, Buffalo offense vs Toledo defense -5.35 pp
3. Rushing success matchup: edge Toledo (1.90 SD) — vs-average expectation: Toledo offense vs Buffalo defense +3.08 pp, Buffalo offense vs Toledo defense -15.79 pp

Main Risk: single-game variance (margin sigma ~ 16 pts)

Model drivers (gradient-boosting SHAP, log-odds toward):
  - Opponent-adjusted scoring margin: +2.61 → Toledo
  - Elo power-rating edge (incl. home field): +0.37 → Toledo
  - Rushing success matchup: +0.30 → Toledo
  - Success-rate matchup: +0.18 → Toledo
  - Explosive-play matchup: +0.14 → Toledo

Member P(Toledo win): elo 0.877, scoring 0.960, logistic 0.960, gbm 0.991, margin_ridge 0.968 · ensemble 0.968
  home_field: Toledo at home (scoring-model home coef 5.8 pts)
  form: last-3 margin vs expectation: Toledo +6.5, Buffalo -9.6
  rest: rest days: Toledo 7, Buffalo 7
  qb_continuity: starter share of season attempts: Toledo 100%, Buffalo 68%
  injuries: Data unavailable (no public college injury feed)
  weather (forecast): {'temperature': 75, 'gust': 15, 'precipitation': 0, 'conditionId': '2'}

logged as prediction 479b5b5f649b4145a4a5bd76e2a5b5de (snapshot sha256 4f841639c89e77e4…)
```

```
Central Michigan @ Ohio
[BLIND pre-game prediction · model v1.1 · data cutoff 2026-10-10T19:03:32Z · kickoff 2026-10-10T19:30:00Z]

Prediction: Ohio
Win Probability: Central Michigan 26% / Ohio 74%
Projected Score: Central Michigan 20–30 Ohio
Projected Margin: 10.3
Upset Probability: 26%
Confidence: 5.0/10

Key Factors:

1. Pass game vs pass defense (opp-adj pass EPA/play): edge Central Michigan (1.15 SD) — vs-average expectation: Ohio offense vs Central Michigan defense -0.31 EPA/play, Central Michigan offense vs Ohio defense -0.03 EPA/play
2. Finishing drives (pts per scoring opportunity): edge Central Michigan (0.99 SD) — vs-average expectation: Ohio offense vs Central Michigan defense -0.31 pts, Central Michigan offense vs Ohio defense +0.44 pts
3. Turnover tendencies: edge Central Michigan (0.91 SD) — vs-average expectation: Ohio offense vs Central Michigan defense +0.42 pp, Central Michigan offense vs Ohio defense -0.32 pp

Main Risk: Central Michigan's best counter: Pass game vs pass defense (opp-adj pass EPA/play) (1.15 SD); starting QB changed last game (Central Michigan) — availability unconfirmed

Model drivers (gradient-boosting SHAP, log-odds toward):
  - Elo power-rating edge (incl. home field): +0.18 → Ohio
  - Opponent-adjusted scoring margin: -0.16 → Central Michigan
  - Rushing success matchup: +0.10 → Ohio
  - Turnover tendencies: +0.08 → Ohio
  - Passing success matchup: -0.08 → Central Michigan

Member P(Ohio win): elo 0.718, scoring 0.712, logistic 0.772, gbm 0.725, margin_ridge 0.750 · ensemble 0.737
  home_field: Ohio at home (scoring-model home coef 5.8 pts)
  form: last-3 margin vs expectation: Ohio -1.0, Central Michigan -1.2
  rest: rest days: Ohio 7, Central Michigan 7
  qb_continuity: starter share of season attempts: Ohio 65%, Central Michigan 58%
  injuries: Data unavailable (no public college injury feed)
  weather (forecast): {'temperature': 65, 'gust': 13, 'precipitation': 81, 'conditionId': '12'}

logged as prediction c061bdbcc4134c27a59ff4bb74878e55 (snapshot sha256 851b5ff5a7c1d552…)
```

```
Kent State @ Western Michigan
[BLIND pre-game prediction · model v1.1 · data cutoff 2026-10-10T19:03:32Z · kickoff 2026-10-10T19:30:00Z]

Prediction: Western Michigan
Win Probability: Kent State 11% / Western Michigan 89%
Projected Score: Kent State 14–34 Western Michigan
Projected Margin: 19.9
Upset Probability: 11%
Confidence: 8.1/10

Key Factors:

1. Success-rate matchup: edge Western Michigan (1.29 SD) — vs-average expectation: Western Michigan offense vs Kent State defense +6.04 pp, Kent State offense vs Western Michigan defense -5.98 pp
2. Points-per-drive matchup: edge Western Michigan (1.21 SD) — vs-average expectation: Western Michigan offense vs Kent State defense +0.34 pts, Kent State offense vs Western Michigan defense -0.92 pts
3. Rushing success matchup: edge Western Michigan (1.14 SD) — vs-average expectation: Western Michigan offense vs Kent State defense +6.10 pp, Kent State offense vs Western Michigan defense -5.89 pp

Main Risk: Kent State's best counter: Pass game vs pass defense (opp-adj pass EPA/play) (0.33 SD)

Model drivers (gradient-boosting SHAP, log-odds toward):
  - Opponent-adjusted scoring margin: +1.37 → Western Michigan
  - Elo power-rating edge (incl. home field): +0.33 → Western Michigan
  - Finishing drives (pts per scoring opportunity): -0.16 → Kent State
  - Points-per-drive matchup: -0.12 → Kent State
  - Rushing success matchup: +0.10 → Western Michigan

Member P(Western Michigan win): elo 0.855, scoring 0.913, logistic 0.885, gbm 0.913, margin_ridge 0.883 · ensemble 0.895
  home_field: Western Michigan at home (scoring-model home coef 5.8 pts)
  form: last-3 margin vs expectation: Western Michigan -10.7, Kent State -8.5
  rest: rest days: Western Michigan 7, Kent State 7
  qb_continuity: starter share of season attempts: Western Michigan 100%, Kent State 100%
  injuries: Data unavailable (no public college injury feed)
  weather (forecast): {'temperature': 79, 'gust': 21, 'precipitation': 7, 'conditionId': '2'}

logged as prediction 2c1b8a3154704eeea45bbb6cb93f4f7f (snapshot sha256 cb0551ac8946bfab…)
```

```
Richmond @ Fordham
[BLIND pre-game prediction · model v1.1 · data cutoff 2026-10-10T19:03:32Z · kickoff 2026-10-10T19:30:00Z]

Prediction: Richmond
Win Probability: Richmond 54% / Fordham 46%
Projected Score: Richmond 24–22 Fordham
Projected Margin: 2.2
Upset Probability: 46%
Confidence: 1.0/10

Key Factors:

1. Field position / special teams: edge Richmond (2.38 SD) — vs-average expectation: Fordham offense vs Richmond defense -4.61 yds, Richmond offense vs Fordham defense +4.21 yds
2. Pass protection vs pass rush (sack rate): edge Richmond (1.89 SD) — vs-average expectation: Fordham offense vs Richmond defense +5.18 pp, Richmond offense vs Fordham defense -0.76 pp
3. Finishing drives (pts per scoring opportunity): edge Fordham (1.01 SD) — vs-average expectation: Fordham offense vs Richmond defense +0.54 pts, Richmond offense vs Fordham defense -0.54 pts

Main Risk: ensemble members disagree on the winner; Fordham's best counter: Finishing drives (pts per scoring opportunity) (1.01 SD)

Model drivers (gradient-boosting SHAP, log-odds toward):
  - Opponent-adjusted scoring margin: -0.79 → Richmond
  - Elo power-rating edge (incl. home field): -0.54 → Richmond
  - Finishing drives (pts per scoring opportunity): -0.13 → Richmond
  - Turnover tendencies: +0.11 → Fordham
  - Field position / special teams: -0.11 → Richmond

Member P(Fordham win): elo 0.177, scoring 0.557, logistic 0.368, gbm 0.390, margin_ridge 0.431 · ensemble 0.462
  home_field: Fordham at home (scoring-model home coef 5.8 pts)
  form: last-3 margin vs expectation: Fordham +6.5, Richmond -1.0
  rest: rest days: Fordham 7, Richmond 7
  qb_continuity: starter share of season attempts: Fordham 100%, Richmond 100%
  injuries: Data unavailable (no public college injury feed)
  weather (forecast): {'temperature': 69, 'gust': 7, 'precipitation': 0, 'conditionId': '2'}

logged as prediction 3b4cfa5c31e84350a07e6707aefac0c3 (snapshot sha256 6d413142b1393a31…)
```

```
Eastern Illinois @ Gardner-Webb
[BLIND pre-game prediction · model v1.1 · data cutoff 2026-10-10T19:03:32Z · kickoff 2026-10-10T19:30:00Z]

Prediction: Gardner-Webb
Win Probability: Eastern Illinois 7% / Gardner-Webb 93%
Projected Score: Eastern Illinois 16–39 Gardner-Webb
Projected Margin: 23.4
Upset Probability: 7%
Confidence: 8.4/10

Key Factors:

1. Pass protection vs pass rush (sack rate): edge Gardner-Webb (2.40 SD) — vs-average expectation: Gardner-Webb offense vs Eastern Illinois defense -2.65 pp, Eastern Illinois offense vs Gardner-Webb defense +5.67 pp
2. Passing success matchup: edge Gardner-Webb (1.70 SD) — vs-average expectation: Gardner-Webb offense vs Eastern Illinois defense +5.06 pp, Eastern Illinois offense vs Gardner-Webb defense -11.16 pp
3. Success-rate matchup: edge Gardner-Webb (1.44 SD) — vs-average expectation: Gardner-Webb offense vs Eastern Illinois defense +2.96 pp, Eastern Illinois offense vs Gardner-Webb defense -10.23 pp

Main Risk: Eastern Illinois's best counter: Turnover tendencies (0.63 SD)

Model drivers (gradient-boosting SHAP, log-odds toward):
  - Opponent-adjusted scoring margin: +1.30 → Gardner-Webb
  - Success-rate matchup: +0.27 → Gardner-Webb
  - Elo power-rating edge (incl. home field): +0.26 → Gardner-Webb
  - Pass protection vs pass rush (sack rate): +0.24 → Gardner-Webb
  - Rushing success matchup: +0.15 → Gardner-Webb

Member P(Gardner-Webb win): elo 0.835, scoring 0.927, logistic 0.918, gbm 0.953, margin_ridge 0.923 · ensemble 0.928
  home_field: Gardner-Webb at home (scoring-model home coef 5.8 pts)
  form: last-3 margin vs expectation: Gardner-Webb +13.3, Eastern Illinois -15.2
  rest: rest days: Gardner-Webb 6, Eastern Illinois 13
  qb_continuity: starter share of season attempts: Gardner-Webb 100%, Eastern Illinois 100%
  injuries: Data unavailable (no public college injury feed)
  weather (forecast): {'temperature': 59, 'gust': 22, 'precipitation': 83, 'conditionId': '18'}

logged as prediction df878dff8f2d43239c2409e9d2df22d8 (snapshot sha256 3a09b43421f72cf4…)
```

```
UT Martin @ Tennessee State
[BLIND pre-game prediction · model v1.1 · data cutoff 2026-10-10T19:03:32Z · kickoff 2026-10-10T19:30:00Z]

Prediction: UT Martin
Win Probability: UT Martin 71% / Tennessee State 29%
Projected Score: UT Martin 29–20 Tennessee State
Projected Margin: 9.4
Upset Probability: 29%
Confidence: 4.4/10

Key Factors:

1. Rushing success matchup: edge UT Martin (1.58 SD) — vs-average expectation: Tennessee State offense vs UT Martin defense -10.78 pp, UT Martin offense vs Tennessee State defense +1.59 pp
2. Run game vs run defense (opp-adj rush EPA/play): edge UT Martin (1.55 SD) — vs-average expectation: Tennessee State offense vs UT Martin defense -0.21 EPA/play, UT Martin offense vs Tennessee State defense +0.08 EPA/play
3. Explosive-play matchup: edge UT Martin (1.00 SD) — vs-average expectation: Tennessee State offense vs UT Martin defense -1.63 pp, UT Martin offense vs Tennessee State defense +1.42 pp

Main Risk: Tennessee State's best counter: Passing success matchup (0.91 SD)

Model drivers (gradient-boosting SHAP, log-odds toward):
  - Opponent-adjusted scoring margin: -0.87 → UT Martin
  - Elo power-rating edge (incl. home field): -0.23 → UT Martin
  - Rushing success matchup: -0.20 → UT Martin
  - Yards-per-play matchup: -0.12 → UT Martin
  - Rest advantage: -0.10 → UT Martin

Member P(Tennessee State win): elo 0.335, scoring 0.419, logistic 0.240, gbm 0.327, margin_ridge 0.226 · ensemble 0.292
  home_field: Tennessee State at home (scoring-model home coef 5.8 pts)
  form: last-3 margin vs expectation: Tennessee State -5.5, UT Martin -18.9
  rest: rest days: Tennessee State 6, UT Martin 14
  qb_continuity: starter share of season attempts: Tennessee State 100%, UT Martin 82%
  injuries: Data unavailable (no public college injury feed)
  weather (forecast): {'temperature': 64, 'gust': 14, 'precipitation': 65, 'conditionId': '18'}

logged as prediction 8c825496a2eb490fbd10f836d7bb4aab (snapshot sha256 fafcfbfbbfbe0127…)
```

```
Furman @ Tennessee Tech
[BLIND pre-game prediction · model v1.1 · data cutoff 2026-10-10T19:03:32Z · kickoff 2026-10-10T19:30:00Z]

Prediction: Tennessee Tech
Win Probability: Furman 15% / Tennessee Tech 85%
Projected Score: Furman 16–32 Tennessee Tech
Projected Margin: 15.8
Upset Probability: 15%
Confidence: 7.5/10

Key Factors:

1. Pass game vs pass defense (opp-adj pass EPA/play): edge Tennessee Tech (1.89 SD) — vs-average expectation: Tennessee Tech offense vs Furman defense +0.14 EPA/play, Furman offense vs Tennessee Tech defense -0.43 EPA/play
2. Passing success matchup: edge Tennessee Tech (1.23 SD) — vs-average expectation: Tennessee Tech offense vs Furman defense +1.85 pp, Furman offense vs Tennessee Tech defense -10.30 pp
3. Run game vs run defense (opp-adj rush EPA/play): edge Furman (1.05 SD) — vs-average expectation: Tennessee Tech offense vs Furman defense -0.20 EPA/play, Furman offense vs Tennessee Tech defense -0.01 EPA/play

Main Risk: Furman's best counter: Run game vs run defense (opp-adj rush EPA/play) (1.05 SD)

Model drivers (gradient-boosting SHAP, log-odds toward):
  - Opponent-adjusted scoring margin: +0.64 → Tennessee Tech
  - Rushing success matchup: +0.18 → Tennessee Tech
  - Elo power-rating edge (incl. home field): +0.14 → Tennessee Tech
  - Run game vs run defense (opp-adj rush EPA/play): -0.14 → Furman
  - Yards-per-play matchup: +0.11 → Tennessee Tech

Member P(Tennessee Tech win): elo 0.827, scoring 0.853, logistic 0.856, gbm 0.867, margin_ridge 0.843 · ensemble 0.849
  home_field: Tennessee Tech at home (scoring-model home coef 5.8 pts)
  form: last-3 margin vs expectation: Tennessee Tech -9.0, Furman -2.6
  rest: rest days: Tennessee Tech 6, Furman 7
  qb_continuity: starter share of season attempts: Tennessee Tech 100%, Furman 43%
  injuries: Data unavailable (no public college injury feed)
  weather (forecast): {'temperature': 66, 'gust': 17, 'precipitation': 69, 'conditionId': '18'}

logged as prediction e17d753b32b846a09ca3780df06900d4 (snapshot sha256 557530ce152d3331…)
```

```
UConn @ Temple
[BLIND pre-game prediction · model v1.1 · data cutoff 2026-10-10T19:03:32Z · kickoff 2026-10-10T19:45:00Z]

Prediction: Temple
Win Probability: UConn 40% / Temple 60%
Projected Score: UConn 26–30 Temple
Projected Margin: 3.7
Upset Probability: 40%
Confidence: 2.5/10

Key Factors:

1. Explosive-play matchup: edge UConn (1.10 SD) — vs-average expectation: Temple offense vs UConn defense -1.29 pp, UConn offense vs Temple defense +2.11 pp
2. Finishing drives (pts per scoring opportunity): edge UConn (0.90 SD) — vs-average expectation: Temple offense vs UConn defense -0.04 pts, UConn offense vs Temple defense +0.62 pts
3. Yards-per-play matchup: edge UConn (0.74 SD) — vs-average expectation: Temple offense vs UConn defense -0.32 yds, UConn offense vs Temple defense +0.39 yds

Main Risk: UConn's best counter: Explosive-play matchup (1.10 SD)

Model drivers (gradient-boosting SHAP, log-odds toward):
  - Opponent-adjusted scoring margin: -0.63 → UConn
  - Explosive-play matchup: -0.11 → UConn
  - Yards-per-play matchup: -0.06 → UConn
  - Turnover tendencies: -0.06 → UConn
  - Pass protection vs pass rush (sack rate): +0.06 → Temple

Member P(Temple win): elo 0.578, scoring 0.615, logistic 0.558, gbm 0.518, margin_ridge 0.601 · ensemble 0.596
  home_field: Temple at home (scoring-model home coef 5.8 pts)
  form: last-3 margin vs expectation: Temple +7.4, UConn +4.7
  rest: rest days: Temple 6, UConn 7
  qb_continuity: starter share of season attempts: Temple 80%, UConn 100%
  injuries: Data unavailable (no public college injury feed)
  weather (forecast): {'temperature': 74, 'gust': 12, 'precipitation': 0, 'conditionId': '4'}

logged as prediction df8ffd7293d8433ba4e0843ea6352a8a (snapshot sha256 deac36a801159fd2…)
```

```
Stony Brook @ Towson
[BLIND pre-game prediction · model v1.1 · data cutoff 2026-10-10T19:03:32Z · kickoff 2026-10-10T20:00:00Z]

Prediction: Stony Brook
Win Probability: Stony Brook 66% / Towson 34%
Projected Score: Stony Brook 33–26 Towson
Projected Margin: 6.8
Upset Probability: 34%
Confidence: 3.5/10

Key Factors:

1. Pass protection vs pass rush (sack rate): edge Stony Brook (2.04 SD) — vs-average expectation: Towson offense vs Stony Brook defense +3.37 pp, Stony Brook offense vs Towson defense -3.08 pp
2. Field position / special teams: edge Stony Brook (0.97 SD) — vs-average expectation: Towson offense vs Stony Brook defense -0.81 yds, Stony Brook offense vs Towson defense +2.44 yds
3. Rushing success matchup: edge Stony Brook (0.91 SD) — vs-average expectation: Towson offense vs Stony Brook defense -3.70 pp, Stony Brook offense vs Towson defense +2.72 pp

Main Risk: Towson's best counter: Finishing drives (pts per scoring opportunity) (0.64 SD)

Model drivers (gradient-boosting SHAP, log-odds toward):
  - Opponent-adjusted scoring margin: -0.95 → Stony Brook
  - Elo power-rating edge (incl. home field): -0.30 → Stony Brook
  - Pass protection vs pass rush (sack rate): -0.16 → Stony Brook
  - Rushing success matchup: -0.12 → Stony Brook
  - Yards-per-play matchup: +0.07 → Towson

Member P(Towson win): elo 0.267, scoring 0.422, logistic 0.247, gbm 0.364, margin_ridge 0.301 · ensemble 0.342
  home_field: Towson at home (scoring-model home coef 5.8 pts)
  form: last-3 margin vs expectation: Towson -2.8, Stony Brook +8.8
  rest: rest days: Towson 7, Stony Brook 14
  qb_continuity: starter share of season attempts: Towson 100%, Stony Brook 100%
  injuries: Data unavailable (no public college injury feed)
  weather (forecast): {'temperature': 69, 'gust': 14, 'precipitation': 20, 'conditionId': '4'}

logged as prediction 007fb6f31cf64661a22aa06c687c448d (snapshot sha256 208ea745bd6aca0a…)
```

```
Utah Tech @ Portland State
[BLIND pre-game prediction · model v1.1 · data cutoff 2026-10-10T19:03:32Z · kickoff 2026-10-10T20:00:00Z]

Prediction: Portland State
Win Probability: Utah Tech 38% / Portland State 62%
Projected Score: Utah Tech 27–31 Portland State
Projected Margin: 4.2
Upset Probability: 38%
Confidence: 2.3/10

Key Factors:

1. Turnover tendencies: edge Portland State (1.10 SD) — vs-average expectation: Portland State offense vs Utah Tech defense -0.62 pp, Utah Tech offense vs Portland State defense +0.45 pp
2. Passing success matchup: edge Portland State (0.63 SD) — vs-average expectation: Portland State offense vs Utah Tech defense +3.30 pp, Utah Tech offense vs Portland State defense -3.69 pp
3. Rushing success matchup: edge Utah Tech (0.51 SD) — vs-average expectation: Portland State offense vs Utah Tech defense +2.89 pp, Utah Tech offense vs Portland State defense +5.67 pp

Main Risk: Utah Tech's best counter: Rushing success matchup (0.51 SD); starting QB changed last game (Portland State) — availability unconfirmed

Model drivers (gradient-boosting SHAP, log-odds toward):
  - Opponent-adjusted scoring margin: -0.29 → Utah Tech
  - Elo power-rating edge (incl. home field): +0.20 → Portland State
  - Rushing success matchup: -0.15 → Utah Tech
  - Home QB continuity: -0.11 → Utah Tech
  - Rest advantage: +0.09 → Portland State

Member P(Portland State win): elo 0.740, scoring 0.687, logistic 0.616, gbm 0.627, margin_ridge 0.585 · ensemble 0.619
  home_field: Portland State at home (scoring-model home coef 5.8 pts)
  form: last-3 margin vs expectation: Portland State -16.8, Utah Tech -8.3
  rest: rest days: Portland State 13, Utah Tech 6
  qb_continuity: starter share of season attempts: Portland State 14%, Utah Tech 100%
  injuries: Data unavailable (no public college injury feed)
  weather (forecast): {'temperature': 57, 'gust': 6, 'precipitation': 49, 'conditionId': '3'}

logged as prediction b0f75ea746064455832d4f94bf8978e2 (snapshot sha256 67c9285cc538a913…)
```

```
Southern @ Prairie View A&M
[BLIND pre-game prediction · model v1.1 · data cutoff 2026-10-10T19:03:32Z · kickoff 2026-10-10T20:00:00Z]

Prediction: Prairie View A&M
Win Probability: Southern 9% / Prairie View A&M 91%
Projected Score: Southern 16–38 Prairie View A&M
Projected Margin: 21.1
Upset Probability: 9%
Confidence: 8.0/10

Key Factors:

1. Field position / special teams: edge Prairie View A&M (1.70 SD) — vs-average expectation: Prairie View A&M offense vs Southern defense +5.52 yds, Southern offense vs Prairie View A&M defense -1.76 yds
2. Turnover tendencies: edge Prairie View A&M (1.38 SD) — vs-average expectation: Prairie View A&M offense vs Southern defense -0.86 pp, Southern offense vs Prairie View A&M defense +0.45 pp
3. Run game vs run defense (opp-adj rush EPA/play): edge Prairie View A&M (1.25 SD) — vs-average expectation: Prairie View A&M offense vs Southern defense +0.21 EPA/play, Southern offense vs Prairie View A&M defense -0.09 EPA/play

Main Risk: Southern's best counter: Pass protection vs pass rush (sack rate) (0.50 SD); starting QB changed last game (Southern) — availability unconfirmed

Model drivers (gradient-boosting SHAP, log-odds toward):
  - Opponent-adjusted scoring margin: +1.30 → Prairie View A&M
  - Elo power-rating edge (incl. home field): +0.31 → Prairie View A&M
  - Away QB continuity: +0.21 → Prairie View A&M
  - Points-per-drive matchup: -0.13 → Southern
  - Run game vs run defense (opp-adj rush EPA/play): +0.12 → Prairie View A&M

Member P(Prairie View A&M win): elo 0.900, scoring 0.919, logistic 0.906, gbm 0.948, margin_ridge 0.903 · ensemble 0.913
  home_field: Prairie View A&M at home (scoring-model home coef 5.8 pts)
  form: last-3 margin vs expectation: Prairie View A&M -7.8, Southern -6.7
  rest: rest days: Prairie View A&M 7, Southern 6
  qb_continuity: starter share of season attempts: Prairie View A&M 100%, Southern 17%
  injuries: Data unavailable (no public college injury feed)
  weather (forecast): {'temperature': 89, 'gust': 9, 'precipitation': 0, 'conditionId': '2'}

logged as prediction 76e369d3cd124d349443fe5af4801318 (snapshot sha256 a6450b849e8d19c8…)
```

```
Incarnate Word @ Lamar
[BLIND pre-game prediction · model v1.1 · data cutoff 2026-10-10T19:03:32Z · kickoff 2026-10-10T20:00:00Z]

Prediction: Lamar
Win Probability: Incarnate Word 39% / Lamar 61%
Projected Score: Incarnate Word 24–29 Lamar
Projected Margin: 4.3
Upset Probability: 39%
Confidence: 2.4/10

Key Factors:

1. Turnover tendencies: edge Incarnate Word (1.22 SD) — vs-average expectation: Lamar offense vs Incarnate Word defense +0.43 pp, Incarnate Word offense vs Lamar defense -0.59 pp
2. Pass protection vs pass rush (sack rate): edge Incarnate Word (0.93 SD) — vs-average expectation: Lamar offense vs Incarnate Word defense +0.94 pp, Incarnate Word offense vs Lamar defense -1.81 pp
3. Run game vs run defense (opp-adj rush EPA/play): edge Incarnate Word (0.82 SD) — vs-average expectation: Lamar offense vs Incarnate Word defense -0.09 EPA/play, Incarnate Word offense vs Lamar defense +0.06 EPA/play

Main Risk: Incarnate Word's best counter: Turnover tendencies (1.22 SD)

Model drivers (gradient-boosting SHAP, log-odds toward):
  - Opponent-adjusted scoring margin: -0.47 → Incarnate Word
  - Elo power-rating edge (incl. home field): +0.15 → Lamar
  - Pass protection vs pass rush (sack rate): -0.13 → Incarnate Word
  - Rushing success matchup: -0.12 → Incarnate Word
  - Turnover tendencies: -0.07 → Incarnate Word

Member P(Lamar win): elo 0.753, scoring 0.644, logistic 0.592, gbm 0.516, margin_ridge 0.615 · ensemble 0.612
  home_field: Lamar at home (scoring-model home coef 5.8 pts)
  form: last-3 margin vs expectation: Lamar +12.4, Incarnate Word -8.5
  rest: rest days: Lamar 6, Incarnate Word 6
  qb_continuity: starter share of season attempts: Lamar 100%, Incarnate Word 100%
  injuries: Data unavailable (no public college injury feed)
  weather (forecast): {'temperature': 87, 'gust': 12, 'precipitation': 0, 'conditionId': '3'}

logged as prediction b55e8821debf40d8bd635c1e41b4639f (snapshot sha256 62a5a96696e391e9…)
```

```
Tennessee @ Arkansas
[BLIND pre-game prediction · model v1.1 · data cutoff 2026-10-10T19:03:32Z · kickoff 2026-10-10T20:15:00Z]

Prediction: Tennessee
Win Probability: Tennessee 78% / Arkansas 22%
Projected Score: Tennessee 34–22 Arkansas
Projected Margin: 12.5
Upset Probability: 22%
Confidence: 5.8/10

Key Factors:

1. Overall offense vs defense (opp-adj EPA/play): edge Tennessee (1.70 SD) — vs-average expectation: Arkansas offense vs Tennessee defense -0.16 EPA/play, Tennessee offense vs Arkansas defense +0.15 EPA/play
2. Points-per-drive matchup: edge Tennessee (1.58 SD) — vs-average expectation: Arkansas offense vs Tennessee defense -0.31 pts, Tennessee offense vs Arkansas defense +0.90 pts
3. Finishing drives (pts per scoring opportunity): edge Tennessee (1.58 SD) — vs-average expectation: Arkansas offense vs Tennessee defense -0.41 pts, Tennessee offense vs Arkansas defense +0.88 pts

Main Risk: Arkansas's best counter: Passing success matchup (0.05 SD); starting QB changed last game (Tennessee) — availability unconfirmed

Model drivers (gradient-boosting SHAP, log-odds toward):
  - Opponent-adjusted scoring margin: -1.25 → Tennessee
  - Elo power-rating edge (incl. home field): -0.50 → Tennessee
  - Away QB continuity: +0.21 → Arkansas
  - Rushing success matchup: -0.14 → Tennessee
  - Pass protection vs pass rush (sack rate): -0.13 → Tennessee

Member P(Arkansas win): elo 0.230, scoring 0.223, logistic 0.214, gbm 0.246, margin_ridge 0.214 · ensemble 0.220
  home_field: Arkansas at home (scoring-model home coef 5.8 pts)
  form: last-3 margin vs expectation: Arkansas -2.7, Tennessee +9.1
  rest: rest days: Arkansas 6, Tennessee 7
  qb_continuity: starter share of season attempts: Arkansas 100%, Tennessee 10%
  injuries: Data unavailable (no public college injury feed)
  weather (forecast): {'temperature': 88, 'gust': 14, 'precipitation': 0, 'conditionId': '1'}

logged as prediction 73747b1cdb7a46d190d6d4d83cd9696c (snapshot sha256 c7156f9d57158082…)
```

```
Maryland @ Ohio State
[BLIND pre-game prediction · model v1.1 · data cutoff 2026-10-10T19:03:32Z · kickoff 2026-10-10T20:15:00Z]

Prediction: Ohio State
Win Probability: Maryland 3% / Ohio State 97%
Projected Score: Maryland 13–44 Ohio State
Projected Margin: 30.6
Upset Probability: 3%
Confidence: 9.5/10

Key Factors:

1. Passing success matchup: edge Ohio State (2.40 SD) — vs-average expectation: Ohio State offense vs Maryland defense +15.20 pp, Maryland offense vs Ohio State defense -7.08 pp
2. Points-per-drive matchup: edge Ohio State (2.01 SD) — vs-average expectation: Ohio State offense vs Maryland defense +1.08 pts, Maryland offense vs Ohio State defense -0.88 pts
3. Success-rate matchup: edge Ohio State (2.01 SD) — vs-average expectation: Ohio State offense vs Maryland defense +9.68 pp, Maryland offense vs Ohio State defense -8.09 pp

Main Risk: Maryland's best counter: Pass protection vs pass rush (sack rate) (1.70 SD)

Model drivers (gradient-boosting SHAP, log-odds toward):
  - Opponent-adjusted scoring margin: +2.72 → Ohio State
  - Elo power-rating edge (incl. home field): +0.39 → Ohio State
  - Success-rate matchup: +0.26 → Ohio State
  - Pass protection vs pass rush (sack rate): -0.17 → Maryland
  - Finishing drives (pts per scoring opportunity): +0.16 → Ohio State

Member P(Ohio State win): elo 0.975, scoring 0.983, logistic 0.962, gbm 0.985, margin_ridge 0.968 · ensemble 0.974
  home_field: Ohio State at home (scoring-model home coef 5.8 pts)
  form: last-3 margin vs expectation: Ohio State +20.8, Maryland -23.8
  rest: rest days: Ohio State 7, Maryland 7
  qb_continuity: starter share of season attempts: Ohio State 100%, Maryland 100%
  injuries: Data unavailable (no public college injury feed)
  weather (forecast): {'temperature': 71, 'gust': 16, 'precipitation': 80, 'conditionId': '12'}

logged as prediction 0f2cfc977a04426399894ad70b5d3f78 (snapshot sha256 50165563fe49b2e7…)
```

```
North Dakota @ Northern Iowa
[BLIND pre-game prediction · model v1.1 · data cutoff 2026-10-10T19:03:32Z · kickoff 2026-10-10T21:00:00Z]

Prediction: North Dakota
Win Probability: North Dakota 77% / Northern Iowa 23%
Projected Score: North Dakota 36–23 Northern Iowa
Projected Margin: 12.3
Upset Probability: 23%
Confidence: 5.4/10

Key Factors:

1. Rushing success matchup: edge North Dakota (2.16 SD) — vs-average expectation: Northern Iowa offense vs North Dakota defense -3.86 pp, North Dakota offense vs Northern Iowa defense +13.79 pp
2. Success-rate matchup: edge North Dakota (2.14 SD) — vs-average expectation: Northern Iowa offense vs North Dakota defense -4.37 pp, North Dakota offense vs Northern Iowa defense +11.07 pp
3. Explosive-play matchup: edge North Dakota (1.96 SD) — vs-average expectation: Northern Iowa offense vs North Dakota defense -1.76 pp, North Dakota offense vs Northern Iowa defense +4.83 pp

Main Risk: Northern Iowa's best counter: Finishing drives (pts per scoring opportunity) (0.03 SD)

Model drivers (gradient-boosting SHAP, log-odds toward):
  - Opponent-adjusted scoring margin: -1.28 → North Dakota
  - Success-rate matchup: -0.37 → North Dakota
  - Elo power-rating edge (incl. home field): -0.24 → North Dakota
  - Rushing success matchup: -0.23 → North Dakota
  - Yards-per-play matchup: -0.17 → North Dakota

Member P(Northern Iowa win): elo 0.316, scoring 0.346, logistic 0.167, gbm 0.163, margin_ridge 0.192 · ensemble 0.232
  home_field: Northern Iowa at home (scoring-model home coef 5.8 pts)
  form: last-3 margin vs expectation: Northern Iowa +11.8, North Dakota -4.1
  rest: rest days: Northern Iowa 14, North Dakota 7
  qb_continuity: starter share of season attempts: Northern Iowa 100%, North Dakota 100%
  injuries: Data unavailable (no public college injury feed)
  weather (forecast): {'temperature': 83, 'gust': 35, 'precipitation': 0, 'conditionId': '1'}

logged as prediction a67a626c90fd492d8dba651f5737ef18 (snapshot sha256 538922fa26d1dd92…)
```

```
Montana @ Northern Arizona
[BLIND pre-game prediction · model v1.1 · data cutoff 2026-10-10T19:03:32Z · kickoff 2026-10-10T21:00:00Z]

Prediction: Montana
Win Probability: Montana 61% / Northern Arizona 39%
Projected Score: Montana 30–25 Northern Arizona
Projected Margin: 5.2
Upset Probability: 39%
Confidence: 2.6/10

Key Factors:

1. Yards-per-play matchup: edge Montana (1.09 SD) — vs-average expectation: Northern Arizona offense vs Montana defense -0.24 yds, Montana offense vs Northern Arizona defense +0.92 yds
2. Explosive-play matchup: edge Montana (1.05 SD) — vs-average expectation: Northern Arizona offense vs Montana defense -1.65 pp, Montana offense vs Northern Arizona defense +1.58 pp
3. Pass protection vs pass rush (sack rate): edge Montana (1.02 SD) — vs-average expectation: Northern Arizona offense vs Montana defense -0.24 pp, Montana offense vs Northern Arizona defense -3.30 pp

Main Risk: Northern Arizona's best counter: Turnover tendencies (1.01 SD)

Model drivers (gradient-boosting SHAP, log-odds toward):
  - Opponent-adjusted scoring margin: -0.90 → Montana
  - Elo power-rating edge (incl. home field): -0.26 → Montana
  - Yards-per-play matchup: -0.15 → Montana
  - Rushing success matchup: -0.11 → Montana
  - Pass protection vs pass rush (sack rate): -0.10 → Montana

Member P(Northern Arizona win): elo 0.374, scoring 0.468, logistic 0.321, gbm 0.295, margin_ridge 0.364 · ensemble 0.386
  home_field: Northern Arizona at home (scoring-model home coef 5.8 pts)
  form: last-3 margin vs expectation: Northern Arizona +6.0, Montana +2.0
  rest: rest days: Northern Arizona 6, Montana 7
  qb_continuity: starter share of season attempts: Northern Arizona 100%, Montana 100%
  injuries: Data unavailable (no public college injury feed)
  weather (forecast): {'temperature': 64, 'gust': 25, 'precipitation': 24, 'conditionId': '7'}

logged as prediction ad10cc454ff84c43b63e2ed07dcf5596 (snapshot sha256 e977e5a41a445789…)
```

```
Jackson State @ Grambling
[BLIND pre-game prediction · model v1.1 · data cutoff 2026-10-10T19:03:32Z · kickoff 2026-10-10T21:00:00Z]

Prediction: Jackson State
Win Probability: Jackson State 69% / Grambling 31%
Projected Score: Jackson State 34–25 Grambling
Projected Margin: 9.2
Upset Probability: 31%
Confidence: 3.9/10

Key Factors:

1. Finishing drives (pts per scoring opportunity): edge Jackson State (1.45 SD) — vs-average expectation: Grambling offense vs Jackson State defense -0.43 pts, Jackson State offense vs Grambling defense +0.74 pts
2. Rushing success matchup: edge Jackson State (1.38 SD) — vs-average expectation: Grambling offense vs Jackson State defense -3.78 pp, Jackson State offense vs Grambling defense +6.84 pp
3. Success-rate matchup: edge Jackson State (1.20 SD) — vs-average expectation: Grambling offense vs Jackson State defense -3.27 pp, Jackson State offense vs Grambling defense +4.63 pp

Main Risk: Grambling's best counter: Turnover tendencies (0.25 SD); starting QB changed last game (Grambling) — availability unconfirmed

Model drivers (gradient-boosting SHAP, log-odds toward):
  - Opponent-adjusted scoring margin: -1.15 → Jackson State
  - Elo power-rating edge (incl. home field): -0.25 → Jackson State
  - Rushing success matchup: -0.21 → Jackson State
  - Success-rate matchup: -0.12 → Jackson State
  - Home QB continuity: -0.10 → Jackson State

Member P(Grambling win): elo 0.277, scoring 0.366, logistic 0.251, gbm 0.260, margin_ridge 0.288 · ensemble 0.307
  home_field: Grambling at home (scoring-model home coef 5.8 pts)
  form: last-3 margin vs expectation: Grambling -13.4, Jackson State +3.7
  rest: rest days: Grambling 7, Jackson State 7
  qb_continuity: starter share of season attempts: Grambling 18%, Jackson State 100%
  injuries: Data unavailable (no public college injury feed)
  weather (forecast): {'temperature': 88, 'gust': 8, 'precipitation': 0, 'conditionId': '3'}

logged as prediction f03d73daf51740d5a0c0bd8794e58ab4 (snapshot sha256 dac76c000ae33078…)
```

```
Northeastern State @ West Florida
[BLIND pre-game prediction · model v1.1 · data cutoff 2026-10-10T19:03:32Z · kickoff 2026-10-10T21:00:00Z]

Prediction: West Florida
Win Probability: Northeastern State 3% / West Florida 97%
Projected Score: Northeastern State 13–44 West Florida
Projected Margin: 30.3
Upset Probability: 3%
Confidence: 8.4/10

Key Factors:

1. Pass protection vs pass rush (sack rate): edge West Florida (1.41 SD) — vs-average expectation: West Florida offense vs Northeastern State defense -3.32 pp, Northeastern State offense vs West Florida defense +1.71 pp
2. Field position / special teams: edge West Florida (1.38 SD) — vs-average expectation: West Florida offense vs Northeastern State defense +3.70 yds, Northeastern State offense vs West Florida defense -2.34 yds
3. Pass game vs pass defense (opp-adj pass EPA/play): edge West Florida (1.32 SD) — vs-average expectation: West Florida offense vs Northeastern State defense +0.14 EPA/play, Northeastern State offense vs West Florida defense -0.27 EPA/play

Main Risk: starting QB changed last game (West Florida) — availability unconfirmed

Model drivers (gradient-boosting SHAP, log-odds toward):
  - Opponent-adjusted scoring margin: +2.78 → West Florida
  - Rushing success matchup: +0.16 → West Florida
  - Pass protection vs pass rush (sack rate): +0.14 → West Florida
  - Elo power-rating edge (incl. home field): +0.14 → West Florida
  - Overall offense vs defense (opp-adj EPA/play): +0.08 → West Florida

Member P(West Florida win): elo 0.823, scoring 0.969, logistic 0.940, gbm 0.986, margin_ridge 0.966 · ensemble 0.969
  home_field: West Florida at home (scoring-model home coef 5.8 pts)
  form: last-3 margin vs expectation: West Florida -10.3, Northeastern State -11.3
  rest: rest days: West Florida 6, Northeastern State 13
  qb_continuity: starter share of season attempts: West Florida 69%, Northeastern State 100%
  injuries: Data unavailable (no public college injury feed)

logged as prediction 135ca0bc5afc4426873cdded6bbfe1f7 (snapshot sha256 bb542c2c2252daf9…)
```

```
San Diego State @ Oregon State
[BLIND pre-game prediction · model v1.1 · data cutoff 2026-10-10T19:03:32Z · kickoff 2026-10-10T22:00:00Z]

Prediction: Oregon State
Win Probability: San Diego State 26% / Oregon State 74%
Projected Score: San Diego State 19–30 Oregon State
Projected Margin: 10.3
Upset Probability: 26%
Confidence: 5.3/10

Key Factors:

1. Passing success matchup: edge Oregon State (1.66 SD) — vs-average expectation: Oregon State offense vs San Diego State defense +6.16 pp, San Diego State offense vs Oregon State defense -9.75 pp
2. Pass game vs pass defense (opp-adj pass EPA/play): edge Oregon State (1.33 SD) — vs-average expectation: Oregon State offense vs San Diego State defense +0.08 EPA/play, San Diego State offense vs Oregon State defense -0.33 EPA/play
3. Success-rate matchup: edge Oregon State (1.24 SD) — vs-average expectation: Oregon State offense vs San Diego State defense +5.02 pp, San Diego State offense vs Oregon State defense -6.60 pp

Main Risk: San Diego State's best counter: Field position / special teams (1.20 SD)

Model drivers (gradient-boosting SHAP, log-odds toward):
  - Rushing success matchup: +0.20 → Oregon State
  - Elo power-rating edge (incl. home field): +0.13 → Oregon State
  - Points-per-drive matchup: -0.11 → San Diego State
  - Recent form vs expectation: -0.09 → San Diego State
  - Passing success matchup: -0.08 → San Diego State

Member P(Oregon State win): elo 0.653, scoring 0.792, logistic 0.774, gbm 0.727, margin_ridge 0.726 · ensemble 0.745
  home_field: Oregon State at home (scoring-model home coef 5.8 pts)
  form: last-3 margin vs expectation: Oregon State +31.9, San Diego State -6.4
  rest: rest days: Oregon State 7, San Diego State 6
  qb_continuity: starter share of season attempts: Oregon State 100%, San Diego State 100%
  injuries: Data unavailable (no public college injury feed)
  weather (forecast): {'temperature': 62, 'gust': 8, 'precipitation': 49, 'conditionId': '4'}

logged as prediction 199f85c15ed44c71bd5e717ecff86ae0 (snapshot sha256 31069c8bc7f1df5e…)
```

```
Cal Poly @ Idaho State
[BLIND pre-game prediction · model v1.1 · data cutoff 2026-10-10T19:03:32Z · kickoff 2026-10-10T22:00:00Z]

Prediction: Idaho State
Win Probability: Cal Poly 16% / Idaho State 84%
Projected Score: Cal Poly 21–36 Idaho State
Projected Margin: 15.3
Upset Probability: 16%
Confidence: 7.3/10

Key Factors:

1. Finishing drives (pts per scoring opportunity): edge Idaho State (1.37 SD) — vs-average expectation: Idaho State offense vs Cal Poly defense +0.42 pts, Cal Poly offense vs Idaho State defense -0.99 pts
2. Turnover tendencies: edge Idaho State (1.07 SD) — vs-average expectation: Idaho State offense vs Cal Poly defense -0.47 pp, Cal Poly offense vs Idaho State defense +0.57 pp
3. Pass protection vs pass rush (sack rate): edge Idaho State (0.87 SD) — vs-average expectation: Idaho State offense vs Cal Poly defense -1.73 pp, Cal Poly offense vs Idaho State defense +1.51 pp

Main Risk: Cal Poly's best counter: Explosive-play matchup (0.43 SD)

Model drivers (gradient-boosting SHAP, log-odds toward):
  - Opponent-adjusted scoring margin: +0.63 → Idaho State
  - Elo power-rating edge (incl. home field): +0.34 → Idaho State
  - Points-per-drive matchup: -0.10 → Cal Poly
  - Rushing success matchup: -0.09 → Cal Poly
  - Yards-per-play matchup: +0.07 → Idaho State

Member P(Idaho State win): elo 0.870, scoring 0.881, logistic 0.821, gbm 0.832, margin_ridge 0.829 · ensemble 0.844
  home_field: Idaho State at home (scoring-model home coef 5.8 pts)
  form: last-3 margin vs expectation: Idaho State +7.4, Cal Poly -2.6
  rest: rest days: Idaho State 6, Cal Poly 6
  qb_continuity: starter share of season attempts: Idaho State 100%, Cal Poly 100%
  injuries: Data unavailable (no public college injury feed)
  weather (forecast): {'temperature': 72, 'gust': 18, 'precipitation': 49, 'conditionId': '7'}

logged as prediction 986b634a8b7a484a9acd6639d77485d6 (snapshot sha256 49d396f07c872cd4…)
```

```
West Georgia @ Eastern Kentucky
[BLIND pre-game prediction · model v1.1 · data cutoff 2026-10-10T19:03:32Z · kickoff 2026-10-10T22:00:00Z]

Prediction: Eastern Kentucky
Win Probability: West Georgia 45% / Eastern Kentucky 55%
Projected Score: West Georgia 23–25 Eastern Kentucky
Projected Margin: 2.0
Upset Probability: 45%
Confidence: 1.0/10

Key Factors:

1. Pass protection vs pass rush (sack rate): edge West Georgia (1.42 SD) — vs-average expectation: Eastern Kentucky offense vs West Georgia defense +2.73 pp, West Georgia offense vs Eastern Kentucky defense -1.64 pp
2. Finishing drives (pts per scoring opportunity): edge West Georgia (1.13 SD) — vs-average expectation: Eastern Kentucky offense vs West Georgia defense -0.47 pts, West Georgia offense vs Eastern Kentucky defense +0.41 pts
3. Yards-per-play matchup: edge West Georgia (0.81 SD) — vs-average expectation: Eastern Kentucky offense vs West Georgia defense -0.76 yds, West Georgia offense vs Eastern Kentucky defense +0.03 yds

Main Risk: ensemble members disagree on the winner; West Georgia's best counter: Pass protection vs pass rush (sack rate) (1.42 SD)

Model drivers (gradient-boosting SHAP, log-odds toward):
  - Opponent-adjusted scoring margin: -0.68 → West Georgia
  - Pass protection vs pass rush (sack rate): -0.14 → West Georgia
  - Yards-per-play matchup: -0.12 → West Georgia
  - Elo power-rating edge (incl. home field): -0.10 → West Georgia
  - Field position / special teams: +0.07 → Eastern Kentucky

Member P(Eastern Kentucky win): elo 0.569, scoring 0.600, logistic 0.530, gbm 0.441, margin_ridge 0.551 · ensemble 0.553
  home_field: Eastern Kentucky at home (scoring-model home coef 5.8 pts)
  form: last-3 margin vs expectation: Eastern Kentucky -1.6, West Georgia -1.4
  rest: rest days: Eastern Kentucky 7, West Georgia 14
  qb_continuity: starter share of season attempts: Eastern Kentucky 48%, West Georgia 71%
  injuries: Data unavailable (no public college injury feed)
  weather (forecast): {'temperature': 59, 'gust': 14, 'precipitation': 90, 'conditionId': '18'}

logged as prediction b2462c9936b9460aa7013d9407e443cb (snapshot sha256 466997be4f53d5ce…)
```

```
LSU @ Kentucky
[BLIND pre-game prediction · model v1.1 · data cutoff 2026-10-10T19:03:32Z · kickoff 2026-10-10T23:00:00Z]

Prediction: LSU
Win Probability: LSU 66% / Kentucky 34%
Projected Score: LSU 29–22 Kentucky
Projected Margin: 7.0
Upset Probability: 34%
Confidence: 3.5/10

Key Factors:

1. Pass protection vs pass rush (sack rate): edge LSU (2.68 SD) — vs-average expectation: Kentucky offense vs LSU defense +7.33 pp, LSU offense vs Kentucky defense -1.23 pp
2. Passing success matchup: edge LSU (1.81 SD) — vs-average expectation: Kentucky offense vs LSU defense -5.83 pp, LSU offense vs Kentucky defense +8.38 pp
3. Success-rate matchup: edge LSU (1.25 SD) — vs-average expectation: Kentucky offense vs LSU defense -3.40 pp, LSU offense vs Kentucky defense +4.90 pp

Main Risk: Kentucky's best counter: Turnover tendencies (0.29 SD)

Model drivers (gradient-boosting SHAP, log-odds toward):
  - Opponent-adjusted scoring margin: -1.06 → LSU
  - Pass protection vs pass rush (sack rate): -0.25 → LSU
  - Success-rate matchup: -0.14 → LSU
  - Yards-per-play matchup: -0.13 → LSU
  - Elo power-rating edge (incl. home field): -0.10 → LSU

Member P(Kentucky win): elo 0.467, scoring 0.379, logistic 0.313, gbm 0.261, margin_ridge 0.335 · ensemble 0.339
  home_field: Kentucky at home (scoring-model home coef 5.8 pts)
  form: last-3 margin vs expectation: Kentucky +12.1, LSU +17.0
  rest: rest days: Kentucky 7, LSU 6
  qb_continuity: starter share of season attempts: Kentucky 100%, LSU 100%
  injuries: Data unavailable (no public college injury feed)
  weather (forecast): {'temperature': 63, 'gust': 18, 'precipitation': 75, 'conditionId': '18'}

logged as prediction 0e260027b6ee4145a479fa784a06c63d (snapshot sha256 3e62a0ec417c7b8c…)
```

```
UAB @ Memphis
[BLIND pre-game prediction · model v1.1 · data cutoff 2026-10-10T19:03:32Z · kickoff 2026-10-10T23:00:00Z]

Prediction: Memphis
Win Probability: UAB 11% / Memphis 89%
Projected Score: UAB 20–40 Memphis
Projected Margin: 19.5
Upset Probability: 11%
Confidence: 7.9/10

Key Factors:

1. Passing success matchup: edge Memphis (1.27 SD) — vs-average expectation: Memphis offense vs UAB defense +6.42 pp, UAB offense vs Memphis defense -6.07 pp
2. Success-rate matchup: edge Memphis (1.16 SD) — vs-average expectation: Memphis offense vs UAB defense +8.92 pp, UAB offense vs Memphis defense -2.01 pp
3. Finishing drives (pts per scoring opportunity): edge Memphis (1.12 SD) — vs-average expectation: Memphis offense vs UAB defense +0.98 pts, UAB offense vs Memphis defense -0.20 pts

Main Risk: UAB's best counter: Field position / special teams (0.46 SD)

Model drivers (gradient-boosting SHAP, log-odds toward):
  - Opponent-adjusted scoring margin: +1.35 → Memphis
  - Finishing drives (pts per scoring opportunity): -0.24 → UAB
  - Rushing success matchup: +0.22 → Memphis
  - Elo power-rating edge (incl. home field): +0.12 → Memphis
  - Yards-per-play matchup: +0.11 → Memphis

Member P(Memphis win): elo 0.805, scoring 0.914, logistic 0.882, gbm 0.900, margin_ridge 0.883 · ensemble 0.894
  home_field: Memphis at home (scoring-model home coef 5.8 pts)
  form: last-3 margin vs expectation: Memphis +9.9, UAB +3.5
  rest: rest days: Memphis 7, UAB 7
  qb_continuity: starter share of season attempts: Memphis 100%, UAB 100%
  injuries: Data unavailable (no public college injury feed)
  weather (forecast): {'temperature': 70, 'gust': 28, 'precipitation': 66, 'conditionId': '18'}

logged as prediction 8310adc2186145c89d11765524ae366e (snapshot sha256 c9fcd059a6d2237d…)
```

```
Nevada @ UTEP
[BLIND pre-game prediction · model v1.1 · data cutoff 2026-10-10T19:03:32Z · kickoff 2026-10-10T23:00:00Z]

Prediction: Nevada
Win Probability: Nevada 72% / UTEP 28%
Projected Score: Nevada 30–21 UTEP
Projected Margin: 9.2
Upset Probability: 28%
Confidence: 4.7/10

Key Factors:

1. Pass protection vs pass rush (sack rate): edge Nevada (2.70 SD) — vs-average expectation: UTEP offense vs Nevada defense +5.65 pp, Nevada offense vs UTEP defense -2.98 pp
2. Finishing drives (pts per scoring opportunity): edge Nevada (1.24 SD) — vs-average expectation: UTEP offense vs Nevada defense -0.70 pts, Nevada offense vs UTEP defense +0.28 pts
3. Points-per-drive matchup: edge Nevada (1.23 SD) — vs-average expectation: UTEP offense vs Nevada defense -0.61 pts, Nevada offense vs UTEP defense +0.28 pts

Main Risk: UTEP's best counter: Turnover tendencies (0.79 SD)

Model drivers (gradient-boosting SHAP, log-odds toward):
  - Opponent-adjusted scoring margin: -1.33 → Nevada
  - Pass protection vs pass rush (sack rate): -0.25 → Nevada
  - Elo power-rating edge (incl. home field): -0.12 → Nevada
  - Yards-per-play matchup: -0.11 → Nevada
  - Rushing success matchup: -0.08 → Nevada

Member P(UTEP win): elo 0.405, scoring 0.311, logistic 0.259, gbm 0.208, margin_ridge 0.276 · ensemble 0.278
  home_field: UTEP at home (scoring-model home coef 5.8 pts)
  form: last-3 margin vs expectation: UTEP -26.0, Nevada -7.1
  rest: rest days: UTEP 7, Nevada 13
  qb_continuity: starter share of season attempts: UTEP 100%, Nevada 78%
  injuries: Data unavailable (no public college injury feed)
  weather (forecast): {'temperature': 86, 'gust': 16, 'precipitation': 0, 'conditionId': '2'}

logged as prediction 40b848385ae34202a665b3801b8e0ade (snapshot sha256 e6c9914b51e11aa3…)
```

```
North Dakota State @ UNLV
[BLIND pre-game prediction · model v1.1 · data cutoff 2026-10-10T19:03:32Z · kickoff 2026-10-10T23:00:00Z]

Prediction: North Dakota State
Win Probability: North Dakota State 58% / UNLV 42%
Projected Score: North Dakota State 25–22 UNLV
Projected Margin: 2.9
Upset Probability: 42%
Confidence: 2.2/10

Key Factors:

1. Turnover tendencies: edge North Dakota State (2.11 SD) — vs-average expectation: UNLV offense vs North Dakota State defense +1.46 pp, North Dakota State offense vs UNLV defense -0.36 pp
2. Finishing drives (pts per scoring opportunity): edge North Dakota State (1.82 SD) — vs-average expectation: UNLV offense vs North Dakota State defense -1.38 pts, North Dakota State offense vs UNLV defense +0.13 pts
3. Run game vs run defense (opp-adj rush EPA/play): edge North Dakota State (1.43 SD) — vs-average expectation: UNLV offense vs North Dakota State defense -0.12 EPA/play, North Dakota State offense vs UNLV defense +0.15 EPA/play

Main Risk: single-game variance (margin sigma ~ 16 pts)

Model drivers (gradient-boosting SHAP, log-odds toward):
  - Opponent-adjusted scoring margin: -0.88 → North Dakota State
  - Pass protection vs pass rush (sack rate): -0.15 → North Dakota State
  - Elo power-rating edge (incl. home field): -0.14 → North Dakota State
  - Field position / special teams: +0.10 → UNLV
  - Rushing success matchup: -0.09 → North Dakota State

Member P(UNLV win): elo 0.439, scoring 0.420, logistic 0.386, gbm 0.400, margin_ridge 0.429 · ensemble 0.423
  home_field: UNLV at home (scoring-model home coef 5.8 pts)
  form: last-3 margin vs expectation: UNLV -4.2, North Dakota State +7.0
  rest: rest days: UNLV 7, North Dakota State 7
  qb_continuity: starter share of season attempts: UNLV 100%, North Dakota State 100%
  injuries: Data unavailable (no public college injury feed)
  weather (forecast): {'temperature': 88, 'gust': 28, 'precipitation': 19, 'conditionId': '4'}

logged as prediction 5c647aac098845bd91147dd1f369589f (snapshot sha256 7b9a0a483cd59dbb…)
```

```
Tarleton State @ Austin Peay
[BLIND pre-game prediction · model v1.1 · data cutoff 2026-10-10T19:03:32Z · kickoff 2026-10-10T23:00:00Z]

Prediction: Austin Peay
Win Probability: Tarleton State 39% / Austin Peay 61%
Projected Score: Tarleton State 28–32 Austin Peay
Projected Margin: 4.0
Upset Probability: 39%
Confidence: 1.3/10

Key Factors:

1. Explosive-play matchup: edge Austin Peay (1.40 SD) — vs-average expectation: Austin Peay offense vs Tarleton State defense +5.93 pp, Tarleton State offense vs Austin Peay defense +0.07 pp
2. Turnover tendencies: edge Tarleton State (1.04 SD) — vs-average expectation: Austin Peay offense vs Tarleton State defense -0.02 pp, Tarleton State offense vs Austin Peay defense -0.88 pp
3. Pass protection vs pass rush (sack rate): edge Tarleton State (1.00 SD) — vs-average expectation: Austin Peay offense vs Tarleton State defense +1.51 pp, Tarleton State offense vs Austin Peay defense -1.47 pp

Main Risk: ensemble members disagree on the winner; Tarleton State's best counter: Turnover tendencies (1.04 SD)

Model drivers (gradient-boosting SHAP, log-odds toward):
  - Opponent-adjusted scoring margin: -0.61 → Tarleton State
  - Yards-per-play matchup: +0.22 → Austin Peay
  - Explosive-play matchup: +0.16 → Austin Peay
  - Elo power-rating edge (incl. home field): -0.16 → Tarleton State
  - Pass protection vs pass rush (sack rate): -0.15 → Tarleton State

Member P(Austin Peay win): elo 0.411, scoring 0.635, logistic 0.563, gbm 0.577, margin_ridge 0.601 · ensemble 0.608
  home_field: Austin Peay at home (scoring-model home coef 5.8 pts)
  form: last-3 margin vs expectation: Austin Peay +12.6, Tarleton State +4.2
  rest: rest days: Austin Peay 7, Tarleton State 7
  qb_continuity: starter share of season attempts: Austin Peay 100%, Tarleton State 100%
  injuries: Data unavailable (no public college injury feed)
  weather (forecast): {'temperature': 64, 'gust': 18, 'precipitation': 69, 'conditionId': '18'}

logged as prediction b74c1e7aee434dd8ac354c5784730491 (snapshot sha256 6830fd31608f9e45…)
```

```
Northern Colorado @ Eastern Washington
[BLIND pre-game prediction · model v1.1 · data cutoff 2026-10-10T19:03:32Z · kickoff 2026-10-10T23:00:00Z]

Prediction: Eastern Washington
Win Probability: Northern Colorado 33% / Eastern Washington 67%
Projected Score: Northern Colorado 24–30 Eastern Washington
Projected Margin: 6.4
Upset Probability: 33%
Confidence: 4.0/10

Key Factors:

1. Run game vs run defense (opp-adj rush EPA/play): edge Northern Colorado (0.67 SD) — vs-average expectation: Eastern Washington offense vs Northern Colorado defense +0.01 EPA/play, Northern Colorado offense vs Eastern Washington defense +0.12 EPA/play
2. Pass game vs pass defense (opp-adj pass EPA/play): edge Eastern Washington (0.67 SD) — vs-average expectation: Eastern Washington offense vs Northern Colorado defense +0.04 EPA/play, Northern Colorado offense vs Eastern Washington defense -0.19 EPA/play
3. Pass protection vs pass rush (sack rate): edge Northern Colorado (0.66 SD) — vs-average expectation: Eastern Washington offense vs Northern Colorado defense +0.51 pp, Northern Colorado offense vs Eastern Washington defense -1.32 pp

Main Risk: Northern Colorado's best counter: Run game vs run defense (opp-adj rush EPA/play) (0.67 SD)

Model drivers (gradient-boosting SHAP, log-odds toward):
  - Pass game vs pass defense (opp-adj pass EPA/play): +0.15 → Eastern Washington
  - Opponent-adjusted scoring margin: -0.12 → Northern Colorado
  - Elo power-rating edge (incl. home field): -0.08 → Northern Colorado
  - Passing success matchup: -0.06 → Northern Colorado
  - Points-per-drive matchup: +0.06 → Eastern Washington

Member P(Eastern Washington win): elo 0.603, scoring 0.703, logistic 0.643, gbm 0.700, margin_ridge 0.644 · ensemble 0.667
  home_field: Eastern Washington at home (scoring-model home coef 5.8 pts)
  form: last-3 margin vs expectation: Eastern Washington -5.0, Northern Colorado +4.1
  rest: rest days: Eastern Washington 6, Northern Colorado 7
  qb_continuity: starter share of season attempts: Eastern Washington 100%, Northern Colorado 51%
  injuries: Data unavailable (no public college injury feed)
  weather (forecast): {'temperature': 56, 'gust': 6, 'precipitation': 47, 'conditionId': '4'}

logged as prediction c5e04719852047a5ab1cce94a179de74 (snapshot sha256 a4eea9a14d741f36…)
```

```
Coastal Carolina @ Marshall
[BLIND pre-game prediction · model v1.1 · data cutoff 2026-10-10T19:03:32Z · kickoff 2026-10-10T23:00:00Z]

Prediction: Marshall
Win Probability: Coastal Carolina 44% / Marshall 56%
Projected Score: Coastal Carolina 26–29 Marshall
Projected Margin: 2.5
Upset Probability: 44%
Confidence: 1.0/10

Key Factors:

1. Passing success matchup: edge Marshall (0.95 SD) — vs-average expectation: Marshall offense vs Coastal Carolina defense +8.72 pp, Coastal Carolina offense vs Marshall defense -1.02 pp
2. Finishing drives (pts per scoring opportunity): edge Coastal Carolina (0.93 SD) — vs-average expectation: Marshall offense vs Coastal Carolina defense -0.70 pts, Coastal Carolina offense vs Marshall defense -0.01 pts
3. Explosive-play matchup: edge Coastal Carolina (0.87 SD) — vs-average expectation: Marshall offense vs Coastal Carolina defense -0.32 pp, Coastal Carolina offense vs Marshall defense +2.25 pp

Main Risk: Coastal Carolina's best counter: Finishing drives (pts per scoring opportunity) (0.93 SD)

Model drivers (gradient-boosting SHAP, log-odds toward):
  - Opponent-adjusted scoring margin: -0.69 → Coastal Carolina
  - Elo power-rating edge (incl. home field): +0.35 → Marshall
  - Rushing success matchup: -0.11 → Coastal Carolina
  - Success-rate matchup: +0.08 → Marshall
  - Pass protection vs pass rush (sack rate): +0.06 → Marshall

Member P(Marshall win): elo 0.840, scoring 0.569, logistic 0.594, gbm 0.615, margin_ridge 0.552 · ensemble 0.564
  home_field: Marshall at home (scoring-model home coef 5.8 pts)
  form: last-3 margin vs expectation: Marshall -7.6, Coastal Carolina -9.1
  rest: rest days: Marshall 7, Coastal Carolina 7
  qb_continuity: starter share of season attempts: Marshall 93%, Coastal Carolina 100%
  injuries: Data unavailable (no public college injury feed)
  weather (forecast): {'temperature': 61, 'gust': 12, 'precipitation': 75, 'conditionId': '18'}

logged as prediction 6db6665533a545f3882b8daf3ceefb11 (snapshot sha256 becaf2846fa13587…)
```

```
Georgia @ Alabama
[BLIND pre-game prediction · model v1.1 · data cutoff 2026-10-10T19:03:32Z · kickoff 2026-10-10T23:30:00Z]

Prediction: Alabama
Win Probability: Georgia 41% / Alabama 59%
Projected Score: Georgia 30–33 Alabama
Projected Margin: 2.9
Upset Probability: 41%
Confidence: 1.0/10

Key Factors:

1. Pass game vs pass defense (opp-adj pass EPA/play): edge Alabama (1.10 SD) — vs-average expectation: Alabama offense vs Georgia defense +0.22 EPA/play, Georgia offense vs Alabama defense -0.13 EPA/play
2. Turnover tendencies: edge Alabama (0.89 SD) — vs-average expectation: Alabama offense vs Georgia defense -0.00 pp, Georgia offense vs Alabama defense +0.87 pp
3. Run game vs run defense (opp-adj rush EPA/play): edge Georgia (0.83 SD) — vs-average expectation: Alabama offense vs Georgia defense -0.14 EPA/play, Georgia offense vs Alabama defense +0.00 EPA/play

Main Risk: ensemble members disagree on the winner; Georgia's best counter: Run game vs run defense (opp-adj rush EPA/play) (0.83 SD)

Model drivers (gradient-boosting SHAP, log-odds toward):
  - Opponent-adjusted scoring margin: -0.39 → Georgia
  - Elo power-rating edge (incl. home field): -0.13 → Georgia
  - Rushing success matchup: -0.10 → Georgia
  - Pass game vs pass defense (opp-adj pass EPA/play): -0.10 → Georgia
  - Turnover tendencies: -0.09 → Georgia

Member P(Alabama win): elo 0.533, scoring 0.678, logistic 0.535, gbm 0.490, margin_ridge 0.562 · ensemble 0.587
  home_field: Alabama at home (scoring-model home coef 5.8 pts)
  form: last-3 margin vs expectation: Alabama +14.8, Georgia +16.0
  rest: rest days: Alabama 7, Georgia 7
  qb_continuity: starter share of season attempts: Alabama 100%, Georgia 100%
  injuries: Data unavailable (no public college injury feed)
  weather (forecast): {'temperature': 64, 'gust': 27, 'precipitation': 30, 'conditionId': '7'}

logged as prediction e5e4259766af4d25811d2c845bbabf7e (snapshot sha256 6e41220d5f879104…)
```

```
Syracuse @ Virginia
[BLIND pre-game prediction · model v1.1 · data cutoff 2026-10-10T19:03:32Z · kickoff 2026-10-10T23:30:00Z]

Prediction: Virginia
Win Probability: Syracuse 26% / Virginia 74%
Projected Score: Syracuse 20–30 Virginia
Projected Margin: 10.5
Upset Probability: 26%
Confidence: 5.0/10

Key Factors:

1. Run game vs run defense (opp-adj rush EPA/play): edge Virginia (1.05 SD) — vs-average expectation: Virginia offense vs Syracuse defense +0.24 EPA/play, Syracuse offense vs Virginia defense -0.02 EPA/play
2. Turnover tendencies: edge Virginia (0.90 SD) — vs-average expectation: Virginia offense vs Syracuse defense -0.51 pp, Syracuse offense vs Virginia defense +0.37 pp
3. Finishing drives (pts per scoring opportunity): edge Virginia (0.80 SD) — vs-average expectation: Virginia offense vs Syracuse defense +0.02 pts, Syracuse offense vs Virginia defense -0.87 pts

Main Risk: Syracuse's best counter: Explosive-play matchup (0.11 SD); starting QB changed last game (Syracuse) — availability unconfirmed

Model drivers (gradient-boosting SHAP, log-odds toward):
  - Away QB continuity: +0.39 → Virginia
  - Elo power-rating edge (incl. home field): +0.18 → Virginia
  - Rushing success matchup: +0.16 → Virginia
  - Points-per-drive matchup: -0.16 → Syracuse
  - Opponent-adjusted scoring margin: -0.09 → Syracuse

Member P(Virginia win): elo 0.818, scoring 0.760, logistic 0.772, gbm 0.783, margin_ridge 0.731 · ensemble 0.745
  home_field: Virginia at home (scoring-model home coef 5.8 pts)
  form: last-3 margin vs expectation: Virginia -7.6, Syracuse +0.6
  rest: rest days: Virginia 7, Syracuse 7
  qb_continuity: starter share of season attempts: Virginia 100%, Syracuse 26%
  injuries: Data unavailable (no public college injury feed)
  weather (forecast): {'temperature': 63, 'gust': 8, 'precipitation': 39, 'conditionId': '36'}

logged as prediction d921d751d11d4525adb559376a170dcc (snapshot sha256 5a4efb2d380279c5…)
```

```
USC @ Penn State
[BLIND pre-game prediction · model v1.1 · data cutoff 2026-10-10T19:03:32Z · kickoff 2026-10-10T23:30:00Z]

Prediction: Penn State
Win Probability: USC 40% / Penn State 60%
Projected Score: USC 27–30 Penn State
Projected Margin: 3.3
Upset Probability: 40%
Confidence: 1.2/10

Key Factors:

1. Turnover tendencies: edge Penn State (0.93 SD) — vs-average expectation: Penn State offense vs USC defense -0.80 pp, USC offense vs Penn State defense +0.10 pp
2. Field position / special teams: edge Penn State (0.74 SD) — vs-average expectation: Penn State offense vs USC defense +0.23 yds, USC offense vs Penn State defense -3.28 yds
3. Passing success matchup: edge USC (0.72 SD) — vs-average expectation: Penn State offense vs USC defense +1.45 pp, USC offense vs Penn State defense +6.18 pp

Main Risk: ensemble members disagree on the winner; USC's best counter: Passing success matchup (0.72 SD)

Model drivers (gradient-boosting SHAP, log-odds toward):
  - Opponent-adjusted scoring margin: -0.39 → USC
  - Elo power-rating edge (incl. home field): -0.15 → USC
  - Field position / special teams: +0.07 → Penn State
  - Turnover tendencies: -0.06 → USC
  - Conference game: -0.04 → USC

Member P(Penn State win): elo 0.477, scoring 0.679, logistic 0.553, gbm 0.561, margin_ridge 0.564 · ensemble 0.596
  home_field: Penn State at home (scoring-model home coef 5.8 pts)
  form: last-3 margin vs expectation: Penn State -4.7, USC -6.6
  rest: rest days: Penn State 7, USC 7
  qb_continuity: starter share of season attempts: Penn State 100%, USC 100%
  injuries: Data unavailable (no public college injury feed)
  weather (forecast): {'temperature': 62, 'gust': 5, 'precipitation': 1, 'conditionId': '38'}

logged as prediction c16172f9f2fd4d77b9ca0eb7624b9b10 (snapshot sha256 cd859720478365c6…)
```

```
Air Force @ Northern Illinois
[BLIND pre-game prediction · model v1.1 · data cutoff 2026-10-10T19:03:32Z · kickoff 2026-10-10T23:30:00Z]

Prediction: Air Force
Win Probability: Air Force 64% / Northern Illinois 36%
Projected Score: Air Force 28–23 Northern Illinois
Projected Margin: 5.5
Upset Probability: 36%
Confidence: 3.2/10

Key Factors:

1. Finishing drives (pts per scoring opportunity): edge Air Force (1.89 SD) — vs-average expectation: Northern Illinois offense vs Air Force defense -0.56 pts, Air Force offense vs Northern Illinois defense +1.01 pts
2. Success-rate matchup: edge Air Force (1.17 SD) — vs-average expectation: Northern Illinois offense vs Air Force defense -4.30 pp, Air Force offense vs Northern Illinois defense +3.41 pp
3. Turnover tendencies: edge Air Force (1.10 SD) — vs-average expectation: Northern Illinois offense vs Air Force defense +0.36 pp, Air Force offense vs Northern Illinois defense -0.56 pp

Main Risk: Northern Illinois's best counter: Yards-per-play matchup (0.06 SD)

Model drivers (gradient-boosting SHAP, log-odds toward):
  - Opponent-adjusted scoring margin: -1.18 → Air Force
  - Elo power-rating edge (incl. home field): -0.22 → Air Force
  - Success-rate matchup: -0.15 → Air Force
  - Rushing success matchup: -0.10 → Air Force
  - Field position / special teams: +0.08 → Northern Illinois

Member P(Northern Illinois win): elo 0.245, scoring 0.359, logistic 0.338, gbm 0.319, margin_ridge 0.376 · ensemble 0.365
  home_field: Northern Illinois at home (scoring-model home coef 5.8 pts)
  form: last-3 margin vs expectation: Northern Illinois -8.6, Air Force +2.9
  rest: rest days: Northern Illinois 14, Air Force 7
  qb_continuity: starter share of season attempts: Northern Illinois 100%, Air Force 100%
  injuries: Data unavailable (no public college injury feed)
  weather (forecast): {'temperature': 72, 'gust': 23, 'precipitation': 0, 'conditionId': '3'}

logged as prediction 57baa5c1051e4931bd624448d4b69734 (snapshot sha256 c61a8ac21e00403a…)
```

```
James Madison @ Georgia Southern
[BLIND pre-game prediction · model v1.1 · data cutoff 2026-10-10T19:03:32Z · kickoff 2026-10-10T23:30:00Z]

Prediction: James Madison
Win Probability: James Madison 85% / Georgia Southern 15%
Projected Score: James Madison 35–19 Georgia Southern
Projected Margin: 16.2
Upset Probability: 15%
Confidence: 7.5/10

Key Factors:

1. Field position / special teams: edge James Madison (1.65 SD) — vs-average expectation: Georgia Southern offense vs James Madison defense -3.76 yds, James Madison offense vs Georgia Southern defense +2.16 yds
2. Points-per-drive matchup: edge James Madison (1.62 SD) — vs-average expectation: Georgia Southern offense vs James Madison defense -0.37 pts, James Madison offense vs Georgia Southern defense +0.86 pts
3. Rushing success matchup: edge James Madison (1.60 SD) — vs-average expectation: Georgia Southern offense vs James Madison defense -5.96 pp, James Madison offense vs Georgia Southern defense +6.61 pp

Main Risk: single-game variance (margin sigma ~ 16 pts)

Model drivers (gradient-boosting SHAP, log-odds toward):
  - Opponent-adjusted scoring margin: -1.56 → James Madison
  - Elo power-rating edge (incl. home field): -0.61 → James Madison
  - Rushing success matchup: -0.26 → James Madison
  - Success-rate matchup: -0.16 → James Madison
  - Field position / special teams: -0.08 → James Madison

Member P(Georgia Southern win): elo 0.148, scoring 0.176, logistic 0.148, gbm 0.153, margin_ridge 0.144 · ensemble 0.154
  home_field: Georgia Southern at home (scoring-model home coef 5.8 pts)
  form: last-3 margin vs expectation: Georgia Southern -2.6, James Madison +13.3
  rest: rest days: Georgia Southern 7, James Madison 7
  qb_continuity: starter share of season attempts: Georgia Southern 100%, James Madison 93%
  injuries: Data unavailable (no public college injury feed)
  weather (forecast): {'temperature': 74, 'gust': 30, 'precipitation': 49, 'conditionId': '7'}

logged as prediction a1e29c47e7f642d99216185277f9d909 (snapshot sha256 67f34333068bd5e6…)
```

```
Louisiana @ Louisiana Tech
[BLIND pre-game prediction · model v1.1 · data cutoff 2026-10-10T19:03:32Z · kickoff 2026-10-10T23:30:00Z]

Prediction: Louisiana Tech
Win Probability: Louisiana 33% / Louisiana Tech 67%
Projected Score: Louisiana 23–30 Louisiana Tech
Projected Margin: 6.9
Upset Probability: 33%
Confidence: 2.8/10

Key Factors:

1. Finishing drives (pts per scoring opportunity): edge Louisiana (1.41 SD) — vs-average expectation: Louisiana Tech offense vs Louisiana defense -0.75 pts, Louisiana offense vs Louisiana Tech defense +0.38 pts
2. Run game vs run defense (opp-adj rush EPA/play): edge Louisiana (0.95 SD) — vs-average expectation: Louisiana Tech offense vs Louisiana defense -0.17 EPA/play, Louisiana offense vs Louisiana Tech defense +0.00 EPA/play
3. Points-per-drive matchup: edge Louisiana (0.88 SD) — vs-average expectation: Louisiana Tech offense vs Louisiana defense -0.31 pts, Louisiana offense vs Louisiana Tech defense +0.27 pts

Main Risk: ensemble members disagree on the winner; Louisiana's best counter: Finishing drives (pts per scoring opportunity) (1.41 SD)

Model drivers (gradient-boosting SHAP, log-odds toward):
  - Opponent-adjusted scoring margin: -0.22 → Louisiana
  - Points-per-drive matchup: +0.13 → Louisiana Tech
  - Elo power-rating edge (incl. home field): -0.08 → Louisiana
  - Scoring environment: -0.05 → Louisiana
  - Rushing success matchup: +0.04 → Louisiana Tech

Member P(Louisiana Tech win): elo 0.499, scoring 0.688, logistic 0.644, gbm 0.660, margin_ridge 0.670 · ensemble 0.674
  home_field: Louisiana Tech at home (scoring-model home coef 5.8 pts)
  form: last-3 margin vs expectation: Louisiana Tech -10.3, Louisiana +6.2
  rest: rest days: Louisiana Tech 7, Louisiana 6
  qb_continuity: starter share of season attempts: Louisiana Tech 65%, Louisiana 100%
  injuries: Data unavailable (no public college injury feed)
  weather (forecast): {'temperature': 83, 'gust': 7, 'precipitation': 0, 'conditionId': '3'}

logged as prediction 91ca18e9321046b4b650063011f2eb3f (snapshot sha256 bd68a6134ed4786d…)
```

```
Kansas @ Utah
[BLIND pre-game prediction · model v1.1 · data cutoff 2026-10-10T19:03:32Z · kickoff 2026-10-11T00:00:00Z]

Prediction: Utah
Win Probability: Kansas 16% / Utah 84%
Projected Score: Kansas 21–36 Utah
Projected Margin: 15.3
Upset Probability: 16%
Confidence: 7.1/10

Key Factors:

1. Pass game vs pass defense (opp-adj pass EPA/play): edge Utah (1.98 SD) — vs-average expectation: Utah offense vs Kansas defense +0.30 EPA/play, Kansas offense vs Utah defense -0.29 EPA/play
2. Turnover tendencies: edge Utah (1.84 SD) — vs-average expectation: Utah offense vs Kansas defense -0.86 pp, Kansas offense vs Utah defense +0.85 pp
3. Field position / special teams: edge Utah (1.59 SD) — vs-average expectation: Utah offense vs Kansas defense +3.06 yds, Kansas offense vs Utah defense -3.81 yds

Main Risk: Kansas's best counter: Rushing success matchup (0.82 SD)

Model drivers (gradient-boosting SHAP, log-odds toward):
  - Opponent-adjusted scoring margin: +0.85 → Utah
  - Elo power-rating edge (incl. home field): +0.32 → Utah
  - Rushing success matchup: -0.15 → Kansas
  - Explosive-play matchup: -0.11 → Kansas
  - Turnover tendencies: +0.10 → Utah

Member P(Utah win): elo 0.927, scoring 0.896, logistic 0.834, gbm 0.836, margin_ridge 0.822 · ensemble 0.844
  home_field: Utah at home (scoring-model home coef 5.8 pts)
  form: last-3 margin vs expectation: Utah +14.5, Kansas +10.6
  rest: rest days: Utah 14, Kansas 7
  qb_continuity: starter share of season attempts: Utah 100%, Kansas 100%
  injuries: Data unavailable (no public college injury feed)
  weather (forecast): {'temperature': 68, 'gust': 20, 'precipitation': 49, 'conditionId': '7'}

logged as prediction 2c525c5866224edeb30c8fedfeefd6a5 (snapshot sha256 690794a79aa6944e…)
```

```
Minnesota @ Purdue
[BLIND pre-game prediction · model v1.1 · data cutoff 2026-10-10T19:03:32Z · kickoff 2026-10-11T00:00:00Z]

Prediction: Minnesota
Win Probability: Minnesota 61% / Purdue 39%
Projected Score: Minnesota 30–25 Purdue
Projected Margin: 5.5
Upset Probability: 39%
Confidence: 2.4/10

Key Factors:

1. Field position / special teams: edge Minnesota (2.00 SD) — vs-average expectation: Purdue offense vs Minnesota defense -5.74 yds, Minnesota offense vs Purdue defense +1.59 yds
2. Turnover tendencies: edge Minnesota (1.36 SD) — vs-average expectation: Purdue offense vs Minnesota defense -0.02 pp, Minnesota offense vs Purdue defense -1.16 pp
3. Pass protection vs pass rush (sack rate): edge Minnesota (1.18 SD) — vs-average expectation: Purdue offense vs Minnesota defense +1.67 pp, Minnesota offense vs Purdue defense -1.92 pp

Main Risk: Purdue's best counter: Finishing drives (pts per scoring opportunity) (0.57 SD)

Model drivers (gradient-boosting SHAP, log-odds toward):
  - Opponent-adjusted scoring margin: -0.85 → Minnesota
  - Elo power-rating edge (incl. home field): -0.23 → Minnesota
  - Pass protection vs pass rush (sack rate): -0.12 → Minnesota
  - Turnover tendencies: -0.11 → Minnesota
  - Rushing success matchup: -0.10 → Minnesota

Member P(Purdue win): elo 0.268, scoring 0.473, logistic 0.313, gbm 0.315, margin_ridge 0.368 · ensemble 0.392
  home_field: Purdue at home (scoring-model home coef 5.8 pts)
  form: last-3 margin vs expectation: Purdue -1.4, Minnesota +15.5
  rest: rest days: Purdue 7, Minnesota 7
  qb_continuity: starter share of season attempts: Purdue 100%, Minnesota 100%
  injuries: Data unavailable (no public college injury feed)
  weather (forecast): {'temperature': 70, 'gust': 8, 'precipitation': 7, 'conditionId': '7'}

logged as prediction 76665c03d30e481398bc7d5b2bee9310 (snapshot sha256 e6ac2451badc3915…)
```

```
Central Arkansas @ Abilene Christian
[BLIND pre-game prediction · model v1.1 · data cutoff 2026-10-10T19:03:32Z · kickoff 2026-10-11T00:00:00Z]

Prediction: Abilene Christian
Win Probability: Central Arkansas 21% / Abilene Christian 79%
Projected Score: Central Arkansas 21–33 Abilene Christian
Projected Margin: 11.9
Upset Probability: 21%
Confidence: 6.4/10

Key Factors:

1. Field position / special teams: edge Abilene Christian (2.06 SD) — vs-average expectation: Abilene Christian offense vs Central Arkansas defense +3.68 yds, Central Arkansas offense vs Abilene Christian defense -5.04 yds
2. Passing success matchup: edge Abilene Christian (1.10 SD) — vs-average expectation: Abilene Christian offense vs Central Arkansas defense +8.72 pp, Central Arkansas offense vs Abilene Christian defense -2.32 pp
3. Explosive-play matchup: edge Abilene Christian (0.94 SD) — vs-average expectation: Abilene Christian offense vs Central Arkansas defense +1.84 pp, Central Arkansas offense vs Abilene Christian defense -2.30 pp

Main Risk: Central Arkansas's best counter: Pass protection vs pass rush (sack rate) (0.39 SD)

Model drivers (gradient-boosting SHAP, log-odds toward):
  - Opponent-adjusted scoring margin: +0.39 → Abilene Christian
  - Elo power-rating edge (incl. home field): +0.17 → Abilene Christian
  - Field position / special teams: -0.15 → Central Arkansas
  - Pass game vs pass defense (opp-adj pass EPA/play): +0.12 → Abilene Christian
  - Passing success matchup: +0.09 → Abilene Christian

Member P(Abilene Christian win): elo 0.830, scoring 0.808, logistic 0.777, gbm 0.825, margin_ridge 0.777 · ensemble 0.791
  home_field: Abilene Christian at home (scoring-model home coef 5.8 pts)
  form: last-3 margin vs expectation: Abilene Christian +1.0, Central Arkansas +2.0
  rest: rest days: Abilene Christian 7, Central Arkansas 14
  qb_continuity: starter share of season attempts: Abilene Christian 41%, Central Arkansas 100%
  injuries: Data unavailable (no public college injury feed)
  weather (forecast): {'temperature': 84, 'gust': 18, 'precipitation': 0, 'conditionId': '1'}

logged as prediction e146a134e3254c98882a9b5bd643e9fc (snapshot sha256 7841df3216612b81…)
```

```
SE Louisiana @ UT Rio Grande Valley
[BLIND pre-game prediction · model v1.1 · data cutoff 2026-10-10T19:03:32Z · kickoff 2026-10-11T00:00:00Z]

Prediction: UT Rio Grande Valley
Win Probability: SE Louisiana 45% / UT Rio Grande Valley 55%
Projected Score: SE Louisiana 25–26 UT Rio Grande Valley
Projected Margin: 1.7
Upset Probability: 45%
Confidence: 1.0/10

Key Factors:

1. Field position / special teams: edge SE Louisiana (1.51 SD) — vs-average expectation: UT Rio Grande Valley offense vs SE Louisiana defense -2.84 yds, SE Louisiana offense vs UT Rio Grande Valley defense +2.55 yds
2. Turnover tendencies: edge SE Louisiana (1.47 SD) — vs-average expectation: UT Rio Grande Valley offense vs SE Louisiana defense +0.96 pp, SE Louisiana offense vs UT Rio Grande Valley defense -0.28 pp
3. Explosive-play matchup: edge UT Rio Grande Valley (0.70 SD) — vs-average expectation: UT Rio Grande Valley offense vs SE Louisiana defense +0.46 pp, SE Louisiana offense vs UT Rio Grande Valley defense -2.81 pp

Main Risk: ensemble members disagree on the winner; SE Louisiana's best counter: Field position / special teams (1.51 SD)

Model drivers (gradient-boosting SHAP, log-odds toward):
  - Opponent-adjusted scoring margin: -0.71 → SE Louisiana
  - Yards-per-play matchup: +0.11 → UT Rio Grande Valley
  - Turnover tendencies: -0.10 → SE Louisiana
  - Rushing success matchup: +0.06 → UT Rio Grande Valley
  - Field position / special teams: -0.05 → SE Louisiana

Member P(UT Rio Grande Valley win): elo 0.578, scoring 0.590, logistic 0.526, gbm 0.475, margin_ridge 0.540 · ensemble 0.547
  home_field: UT Rio Grande Valley at home (scoring-model home coef 5.8 pts)
  form: last-3 margin vs expectation: UT Rio Grande Valley -2.1, SE Louisiana +7.1
  rest: rest days: UT Rio Grande Valley 7, SE Louisiana 14
  qb_continuity: starter share of season attempts: UT Rio Grande Valley 53%, SE Louisiana 100%
  injuries: Data unavailable (no public college injury feed)

logged as prediction 181260f292694cf5a4836c014aa86ded (snapshot sha256 fb938460798dce65…)
```

```
Hawai'i @ Arizona State
[BLIND pre-game prediction · model v1.1 · data cutoff 2026-10-10T19:03:32Z · kickoff 2026-10-11T02:30:00Z]

Prediction: Arizona State
Win Probability: Hawai'i 12% / Arizona State 88%
Projected Score: Hawai'i 16–36 Arizona State
Projected Margin: 19.5
Upset Probability: 12%
Confidence: 7.7/10

Key Factors:

1. Explosive-play matchup: edge Arizona State (1.76 SD) — vs-average expectation: Arizona State offense vs Hawai'i defense +3.74 pp, Hawai'i offense vs Arizona State defense -3.44 pp
2. Success-rate matchup: edge Arizona State (1.31 SD) — vs-average expectation: Arizona State offense vs Hawai'i defense +7.01 pp, Hawai'i offense vs Arizona State defense -5.15 pp
3. Passing success matchup: edge Arizona State (1.25 SD) — vs-average expectation: Arizona State offense vs Hawai'i defense +6.98 pp, Hawai'i offense vs Arizona State defense -5.33 pp

Main Risk: Hawai'i's best counter: Turnover tendencies (0.27 SD)

Model drivers (gradient-boosting SHAP, log-odds toward):
  - Opponent-adjusted scoring margin: +0.67 → Arizona State
  - Explosive-play matchup: +0.27 → Arizona State
  - Rushing success matchup: +0.12 → Arizona State
  - Elo power-rating edge (incl. home field): +0.12 → Arizona State
  - Success-rate matchup: +0.11 → Arizona State

Member P(Arizona State win): elo 0.801, scoring 0.872, logistic 0.888, gbm 0.920, margin_ridge 0.880 · ensemble 0.882
  home_field: Arizona State at home (scoring-model home coef 5.8 pts)
  form: last-3 margin vs expectation: Arizona State -19.0, Hawai'i -9.5
  rest: rest days: Arizona State 7, Hawai'i 6
  qb_continuity: starter share of season attempts: Arizona State 100%, Hawai'i 100%
  injuries: Data unavailable (no public college injury feed)
  weather (forecast): {'temperature': 85, 'gust': 12, 'precipitation': 20, 'conditionId': '38'}

logged as prediction 8db8a872ff424b3c981abade4ac88e36 (snapshot sha256 86af10de45a9928e…)
```

```
Boise State @ Fresno State
[BLIND pre-game prediction · model v1.1 · data cutoff 2026-10-10T19:03:32Z · kickoff 2026-10-11T02:30:00Z]

Prediction: Boise State
Win Probability: Boise State 62% / Fresno State 38%
Projected Score: Boise State 26–21 Fresno State
Projected Margin: 5.3
Upset Probability: 38%
Confidence: 2.9/10

Key Factors:

1. Rushing success matchup: edge Boise State (2.20 SD) — vs-average expectation: Fresno State offense vs Boise State defense -13.34 pp, Boise State offense vs Fresno State defense +4.65 pp
2. Field position / special teams: edge Fresno State (2.12 SD) — vs-average expectation: Fresno State offense vs Boise State defense +3.10 yds, Boise State offense vs Fresno State defense -5.87 yds
3. Run game vs run defense (opp-adj rush EPA/play): edge Boise State (1.89 SD) — vs-average expectation: Fresno State offense vs Boise State defense -0.29 EPA/play, Boise State offense vs Fresno State defense +0.08 EPA/play

Main Risk: Fresno State's best counter: Field position / special teams (2.12 SD)

Model drivers (gradient-boosting SHAP, log-odds toward):
  - Opponent-adjusted scoring margin: -0.89 → Boise State
  - Rushing success matchup: -0.25 → Boise State
  - Success-rate matchup: -0.19 → Boise State
  - Overall offense vs defense (opp-adj EPA/play): -0.07 → Boise State
  - Elo power-rating edge (incl. home field): -0.07 → Boise State

Member P(Fresno State win): elo 0.462, scoring 0.431, logistic 0.366, gbm 0.318, margin_ridge 0.361 · ensemble 0.376
  home_field: Fresno State at home (scoring-model home coef 5.8 pts)
  form: last-3 margin vs expectation: Fresno State +10.6, Boise State +13.0
  rest: rest days: Fresno State 7, Boise State 7
  qb_continuity: starter share of season attempts: Fresno State 100%, Boise State 100%
  injuries: Data unavailable (no public college injury feed)
  weather (forecast): {'temperature': 80, 'gust': 12, 'precipitation': 0, 'conditionId': '35'}

logged as prediction 757252cad7ec4d5790ec5fd011bf7ec7 (snapshot sha256 87901c7e9bdb22a3…)
```

## Summary

| Game | Kickoff (UTC) | Pick | Win % | Projected score | Upset % | Confidence | Flags |
|---|---|---|---|---|---|---|---|
| Texas @ Oklahoma (N) | 10-10 19:30 | Texas | 72% | Texas 24–16 Oklahoma | 28% | 4.8 |  |
| Ole Miss @ Vanderbilt | 10-10 19:30 | Ole Miss | 53% | Ole Miss 29–28 Vanderbilt | 47% | 1.0 |  |
| Houston @ Kansas State | 10-10 19:30 | Kansas State | 53% | Houston 28–30 Kansas State | 47% | 1.0 |  |
| Duke @ Georgia Tech | 10-10 19:30 | Duke | 65% | Duke 31–24 Georgia Tech | 35% | 3.5 |  |
| Stanford @ Notre Dame | 10-10 19:30 | Notre Dame | 99% | Stanford 8–51 Notre Dame | 1% | 9.7 |  |
| Virginia Tech @ California | 10-10 19:30 | Virginia Tech | 78% | Virginia Tech 34–21 California | 22% | 5.0 |  |
| Illinois @ Michigan State | 10-10 19:30 | Michigan State | 50% | Illinois 27–26 Michigan State | 50% | 1.0 |  |
| UCLA @ Oregon | 10-10 19:30 | Oregon | 70% | UCLA 26–34 Oregon | 30% | 4.0 |  |
| Charlotte @ North Texas | 10-10 19:30 | North Texas | 98% | Charlotte 15–50 North Texas | 2% | 9.6 |  |
| Tulsa @ Navy | 10-10 19:30 | Navy | 64% | Tulsa 20–26 Navy | 36% | 2.6 |  |
| Eastern Michigan @ Akron | 10-10 19:30 | Eastern Michigan | 55% | Eastern Michigan 27–25 Akron | 45% | 1.4 |  |
| Buffalo @ Toledo | 10-10 19:30 | Toledo | 97% | Buffalo 15–44 Toledo | 3% | 9.1 |  |
| Central Michigan @ Ohio | 10-10 19:30 | Ohio | 74% | Central Michigan 20–30 Ohio | 26% | 5.0 |  |
| Kent State @ Western Michigan | 10-10 19:30 | Western Michigan | 89% | Kent State 14–34 Western Michigan | 11% | 8.1 |  |
| Richmond @ Fordham | 10-10 19:30 | Richmond | 54% | Richmond 24–22 Fordham | 46% | 1.0 |  |
| Eastern Illinois @ Gardner-Webb | 10-10 19:30 | Gardner-Webb | 93% | Eastern Illinois 16–39 Gardner-Webb | 7% | 8.4 |  |
| UT Martin @ Tennessee State | 10-10 19:30 | UT Martin | 71% | UT Martin 29–20 Tennessee State | 29% | 4.4 |  |
| Furman @ Tennessee Tech | 10-10 19:30 | Tennessee Tech | 85% | Furman 16–32 Tennessee Tech | 15% | 7.5 |  |
| UConn @ Temple | 10-10 19:45 | Temple | 60% | UConn 26–30 Temple | 40% | 2.5 |  |
| Stony Brook @ Towson | 10-10 20:00 | Stony Brook | 66% | Stony Brook 33–26 Towson | 34% | 3.5 |  |
| Utah Tech @ Portland State | 10-10 20:00 | Portland State | 62% | Utah Tech 27–31 Portland State | 38% | 2.3 |  |
| Southern @ Prairie View A&M | 10-10 20:00 | Prairie View A&M | 91% | Southern 16–38 Prairie View A&M | 9% | 8.0 |  |
| Incarnate Word @ Lamar | 10-10 20:00 | Lamar | 61% | Incarnate Word 24–29 Lamar | 39% | 2.4 |  |
| Tennessee @ Arkansas | 10-10 20:15 | Tennessee | 78% | Tennessee 34–22 Arkansas | 22% | 5.8 |  |
| Maryland @ Ohio State | 10-10 20:15 | Ohio State | 97% | Maryland 13–44 Ohio State | 3% | 9.5 |  |
| North Dakota @ Northern Iowa | 10-10 21:00 | North Dakota | 77% | North Dakota 36–23 Northern Iowa | 23% | 5.4 |  |
| Montana @ Northern Arizona | 10-10 21:00 | Montana | 61% | Montana 30–25 Northern Arizona | 39% | 2.6 |  |
| Jackson State @ Grambling | 10-10 21:00 | Jackson State | 69% | Jackson State 34–25 Grambling | 31% | 3.9 |  |
| Northeastern State @ West Florida | 10-10 21:00 | West Florida | 97% | Northeastern State 13–44 West Florida | 3% | 8.4 |  |
| San Diego State @ Oregon State | 10-10 22:00 | Oregon State | 74% | San Diego State 19–30 Oregon State | 26% | 5.3 |  |
| Cal Poly @ Idaho State | 10-10 22:00 | Idaho State | 84% | Cal Poly 21–36 Idaho State | 16% | 7.3 |  |
| West Georgia @ Eastern Kentucky | 10-10 22:00 | Eastern Kentucky | 55% | West Georgia 23–25 Eastern Kentucky | 45% | 1.0 |  |
| LSU @ Kentucky | 10-10 23:00 | LSU | 66% | LSU 29–22 Kentucky | 34% | 3.5 |  |
| UAB @ Memphis | 10-10 23:00 | Memphis | 89% | UAB 20–40 Memphis | 11% | 7.9 |  |
| Nevada @ UTEP | 10-10 23:00 | Nevada | 72% | Nevada 30–21 UTEP | 28% | 4.7 |  |
| North Dakota State @ UNLV | 10-10 23:00 | North Dakota State | 58% | North Dakota State 25–22 UNLV | 42% | 2.2 |  |
| Tarleton State @ Austin Peay | 10-10 23:00 | Austin Peay | 61% | Tarleton State 28–32 Austin Peay | 39% | 1.3 |  |
| Northern Colorado @ Eastern Washington | 10-10 23:00 | Eastern Washington | 67% | Northern Colorado 24–30 Eastern Washington | 33% | 4.0 |  |
| Coastal Carolina @ Marshall | 10-10 23:00 | Marshall | 56% | Coastal Carolina 26–29 Marshall | 44% | 1.0 |  |
| Georgia @ Alabama | 10-10 23:30 | Alabama | 59% | Georgia 30–33 Alabama | 41% | 1.0 |  |
| Syracuse @ Virginia | 10-10 23:30 | Virginia | 74% | Syracuse 20–30 Virginia | 26% | 5.0 |  |
| USC @ Penn State | 10-10 23:30 | Penn State | 60% | USC 27–30 Penn State | 40% | 1.2 |  |
| Air Force @ Northern Illinois | 10-10 23:30 | Air Force | 64% | Air Force 28–23 Northern Illinois | 36% | 3.2 |  |
| James Madison @ Georgia Southern | 10-10 23:30 | James Madison | 85% | James Madison 35–19 Georgia Southern | 15% | 7.5 |  |
| Louisiana @ Louisiana Tech | 10-10 23:30 | Louisiana Tech | 67% | Louisiana 23–30 Louisiana Tech | 33% | 2.8 |  |
| Kansas @ Utah | 10-11 00:00 | Utah | 84% | Kansas 21–36 Utah | 16% | 7.1 |  |
| Minnesota @ Purdue | 10-11 00:00 | Minnesota | 61% | Minnesota 30–25 Purdue | 39% | 2.4 |  |
| Central Arkansas @ Abilene Christian | 10-11 00:00 | Abilene Christian | 79% | Central Arkansas 21–33 Abilene Christian | 21% | 6.4 |  |
| SE Louisiana @ UT Rio Grande Valley | 10-11 00:00 | UT Rio Grande Valley | 55% | SE Louisiana 25–26 UT Rio Grande Valley | 45% | 1.0 |  |
| Hawai'i @ Arizona State | 10-11 02:30 | Arizona State | 88% | Hawai'i 16–36 Arizona State | 12% | 7.7 |  |
| Boise State @ Fresno State | 10-11 02:30 | Boise State | 62% | Boise State 26–21 Fresno State | 38% | 2.9 |  |

**Most confident:** Notre Dame (99%), North Texas (98%), Ohio State (97%), West Florida (97%)

**Closest:** Illinois @ Michigan State (50%), Ole Miss @ Vanderbilt (53%), Houston @ Kansas State (53%), Richmond @ Fordham (54%)

**Upset candidates (underdog ≥ 35%):** Illinois 50%, Vanderbilt 47%, Houston 47%, Fordham 46%, SE Louisiana 45%, West Georgia 45%, Akron 45%, Coastal Carolina 44%, UNLV 42%, Georgia 41%, USC 40%, UConn 40%, Tarleton State 39%, Purdue 39%, Incarnate Word 39%, Northern Arizona 39%, Utah Tech 38%, Fresno State 38%, Northern Illinois 36%, Tulsa 36%, Georgia Tech 35%

**Highest model disagreement:** Richmond @ Fordham, Coastal Carolina @ Marshall, Virginia Tech @ California