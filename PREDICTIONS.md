# NCAA Football — Blind Pre-Match Predictions, Saturday 2026-10-03

- **Data cutoff (training/ratings):** completed games kicking off before 2026-10-03T12:00:00+00:00 (no slate-day results used)
- **Slate collected:** 2026-10-03T20:20:09+00:00 UTC; **Predictions frozen:** 2026-10-03T20:35:20.283773+00:00 (git commit 82c7a54, pushed 20:35:40 UTC)
- **Games predicted:** 42 (all FBS/FCS games not yet kicked off at freeze). Excluded: Kentucky @ South Carolina (2026-10-03 04:15 PM ET, kicked off before freeze); plus 65 games already in progress/final at collection.
- **Sources:** ESPN public JSON APIs (scoreboard, game summary/box score/play-by-play, standings, venue + AccuWeather forecast embedded in ESPN game data). CollegeFootballData API (requires key → 401), Sports-Reference (403) and Open-Meteo (timeout) were not reachable and were **not** bypassed.
- **Betting odds:** not used as a model input. The DraftKings line is stored in `games_today.csv` only as a reference column (`*_REFERENCE_ONLY`).
- **Injuries:** *Data unavailable* — ESPN's college injury feed returned no entries and no depth charts are published. QB availability is proxied by box-score starter continuity (flagged when the starter changed last game).

## Model & walk-forward validation

Ensemble members (trained on 2022→ seasons, features computed strictly from games before each game's week):
1. **Elo** (MOV-weighted, tuned K/HFA/season-regression on 2017–22)
2. **Opponent-adjusted scoring model** (weekly ridge regression, off/def points with home term and preseason priors)
3. **Logistic regression** and 4. **LightGBM gradient boosting** on 29 features (Elo, adjusted scoring, opponent-adjusted EPA/success/explosiveness/rush/pass/sack/turnover/drive/field-position matchups, form, rest, QB continuity, site, division)
5. **Point-differential regression** (ridge) + a LightGBM margin regressor; totals from a separate ridge model.

Ensemble weights were fit on out-of-sample walk-forward predictions for 2023–2025 (train on seasons < S, test on S); 2026 (weeks 1–5, pre-today) is a pure holdout.

| Probability model | weight |  | Margin model | weight |
|---|---|---|---|---|
| Elo | 0.002 | | m_elo | 0.000 |
| Adj. scoring model | 0.249 | | m_pts | 0.265 |
| Logistic reg. | 0.263 | | m_reg | 0.518 |
| Gradient boosting | 0.069 | | m_gbm | 0.218 |
| Point-diff regression | 0.418 | |  |  |

| Season (OOS) | Games | Accuracy | Brier | Log loss | Margin MAE | Total MAE | Elo-only log loss |
|---|---|---|---|---|---|---|---|
| 2023 | 1602 | 0.740 | 0.171 | 0.508 | 13.10 | 13.31 | 0.520 |
| 2024 | 1665 | 0.733 | 0.170 | 0.505 | 12.77 | 12.98 | 0.527 |
| 2025 | 1686 | 0.768 | 0.156 | 0.472 | 12.39 | 12.74 | 0.501 |
| 2026 (holdout) | 596 | 0.805 | 0.132 | 0.407 | 13.85 | 12.45 | 0.445 |

Calibration (ensemble, all OOS seasons 2023–2026):

| Predicted bin | n | mean predicted | actual win rate |
|---|---|---|---|
| (0.0, 0.1] | 123 | 0.067 | 0.089 |
| (0.1, 0.2] | 248 | 0.154 | 0.125 |
| (0.2, 0.3] | 390 | 0.255 | 0.221 |
| (0.3, 0.4] | 464 | 0.351 | 0.358 |
| (0.4, 0.5] | 549 | 0.452 | 0.446 |
| (0.5, 0.6] | 614 | 0.550 | 0.520 |
| (0.6, 0.7] | 660 | 0.649 | 0.662 |
| (0.7, 0.8] | 723 | 0.749 | 0.755 |
| (0.8, 0.9] | 774 | 0.850 | 0.866 |
| (0.9, 1.0] | 1004 | 0.955 | 0.969 |

Ensemble margin residual σ = 16.1 pts. Note: 2023–2026 OOS sets include FBS-vs-FCS and FCS games (easier to call); FBS-involved-only 2026 log loss = 0.347.

Definitions: **Upset probability** = win probability of the model's underdog. **Confidence** = 10·|2p−1|^0.75 minus penalties for model spread (10·SD of member probabilities), models disagreeing on the winner (−1), thin 2026 sample (−0.75) and a QB change last game (−0.5); clipped to 1–10.

## Game-by-game predictions

### Youngstown State @ Southern Illinois
*2026-10-03 05:00 PM ET · Saluki Stadium (Carbondale, IL) · Missouri Valley Football · records: Youngstown State 3-2, Southern Illinois 3-2*

**Prediction: Youngstown State**  
Win Probability: Youngstown State 57% / Southern Illinois 43%  
Projected Score: Youngstown State 32–29 Southern Illinois  
Projected Margin: 2.3 (Youngstown State)  
Upset Probability: 43%  
Confidence: 1.0/10  ⚠️ HIGH MODEL DISAGREEMENT

Key Factors:

1. Overall offense vs defense (opp-adj EPA/play): edge **Youngstown State** (1.23 SD); vs-average expectation — Southern Illinois offense vs Youngstown State defense +0.01 EPA/play, Youngstown State offense vs Southern Illinois defense +0.31 EPA/play
2. Run game vs run defense (opp-adj rush EPA/play): edge **Youngstown State** (1.16 SD); vs-average expectation — Southern Illinois offense vs Youngstown State defense -0.01 EPA/play, Youngstown State offense vs Southern Illinois defense +0.28 EPA/play
3. Pass game vs pass defense (opp-adj pass EPA/play): edge **Youngstown State** (0.92 SD); vs-average expectation — Southern Illinois offense vs Youngstown State defense +0.05 EPA/play, Youngstown State offense vs Southern Illinois defense +0.33 EPA/play

Main Risk: models disagree on the winner; Southern Illinois's best counter is OL protection vs pass rush (0.57 SD edge)

<details><summary>Model breakdown & matchup detail</summary>

| Model | Elo | Adj. scoring model | Logistic reg. | Gradient boosting | Point-diff regression | **Ensemble** |
|---|---|---|---|---|---|---|
| P(Southern Illinois win) | 0.497 | 0.442 | 0.389 | 0.531 | 0.435 | **0.431** |
| Home margin | +0.6 | -2.5 | — | +1.2 (GBM reg.) | -2.6 | **-2.3** |

- Elo: Youngstown State 1387, Southern Illinois 1321; adj. net points rating: Youngstown State +10.5, Southern Illinois +3.5
- Success-rate matchup: edge **Youngstown State** (0.73 SD); vs-average expectation — Southern Illinois offense vs Youngstown State defense +3.59 pp, Youngstown State offense vs Southern Illinois defense +11.45 pp
- OL protection vs pass rush (sack rate): edge **Southern Illinois** (0.57 SD); vs-average expectation — Southern Illinois offense vs Youngstown State defense -3.35 pp sack rate, Youngstown State offense vs Southern Illinois defense -1.26 pp sack rate
- Turnover tendencies (giveaways vs takeaways): edge **Youngstown State** (0.34 SD); vs-average expectation — Southern Illinois offense vs Youngstown State defense +0.43 pp turnover rate, Youngstown State offense vs Southern Illinois defense +0.10 pp turnover rate
- Finishing drives (pts per scoring opportunity): edge **Youngstown State** (0.30 SD); vs-average expectation — Southern Illinois offense vs Youngstown State defense +0.59 pts, Youngstown State offense vs Southern Illinois defense +0.94 pts
- Field position / special teams (start field pos.): edge **Southern Illinois** (0.10 SD); vs-average expectation — Southern Illinois offense vs Youngstown State defense +0.51 yds, Youngstown State offense vs Southern Illinois defense +0.08 yds
- Explosive-play rate matchup: edge **Youngstown State** (0.03 SD); vs-average expectation — Southern Illinois offense vs Youngstown State defense +2.95 pp, Youngstown State offense vs Southern Illinois defense +3.07 pp
- QB: Southern Illinois DJ Williams (7.8 YPA, 4 TD/2 INT) vs Youngstown State Beau Brungard (8.7 YPA, 13 TD/2 INT)
- Home field: Southern Illinois (Elo home bonus 45 rating pts; scoring-model home coefficient 6.4 pts on 2026 data)
- Recent form (last-3 margin vs expectation): Southern Illinois +6.8, Youngstown State +18.0
- Strength of schedule (avg opp net rating): Southern Illinois -2.1, Youngstown State +4.3
- Rest: Youngstown State 7 days, Southern Illinois 6 days; travel distance: Data unavailable
- Weather: Forecast: Mostly sunny, 76.0°F, gusts 16.0 mph, precip 0.0%
- Injuries/availability: Data unavailable (no public CFB injury feed); QB continuity — Youngstown State starter share 100%, Southern Illinois 60%

</details>

### New Haven @ Austin Peay
*2026-10-03 05:00 PM ET · Fortera Stadium (Clarksville, TN) · Northeast vs United Athletic · records: New Haven 1-4, Austin Peay 4-1*

**Prediction: Austin Peay**  
Win Probability: New Haven 1% / Austin Peay 99%  
Projected Score: New Haven 8–45 Austin Peay  
Projected Margin: 37.7 (Austin Peay)  
Upset Probability: 1%  
Confidence: 9.2/10

Key Factors:

1. Run game vs run defense (opp-adj rush EPA/play): edge **Austin Peay** (2.42 SD); vs-average expectation — Austin Peay offense vs New Haven defense +0.38 EPA/play, New Haven offense vs Austin Peay defense -0.24 EPA/play
2. Overall offense vs defense (opp-adj EPA/play): edge **Austin Peay** (2.30 SD); vs-average expectation — Austin Peay offense vs New Haven defense +0.30 EPA/play, New Haven offense vs Austin Peay defense -0.26 EPA/play
3. Pass game vs pass defense (opp-adj pass EPA/play): edge **Austin Peay** (2.24 SD); vs-average expectation — Austin Peay offense vs New Haven defense +0.38 EPA/play, New Haven offense vs Austin Peay defense -0.32 EPA/play

Main Risk: New Haven's best counter is OL protection vs pass rush (1.55 SD edge); QB change last game (New Haven) — availability unconfirmed

<details><summary>Model breakdown & matchup detail</summary>

| Model | Elo | Adj. scoring model | Logistic reg. | Gradient boosting | Point-diff regression | **Ensemble** |
|---|---|---|---|---|---|---|
| P(Austin Peay win) | 0.955 | 0.991 | 0.982 | 0.991 | 0.990 | **0.988** |
| Home margin | +31.0 | +39.4 | — | +40.4 (GBM reg.) | +38.0 | **+37.7** |

- Elo: New Haven 956, Austin Peay 1374; adj. net points rating: New Haven -19.0, Austin Peay +10.7
- Explosive-play rate matchup: edge **Austin Peay** (1.95 SD); vs-average expectation — Austin Peay offense vs New Haven defense +5.16 pp, New Haven offense vs Austin Peay defense -4.00 pp
- Field position / special teams (start field pos.): edge **Austin Peay** (1.72 SD); vs-average expectation — Austin Peay offense vs New Haven defense +1.59 yds, New Haven offense vs Austin Peay defense -5.85 yds
- OL protection vs pass rush (sack rate): edge **New Haven** (1.55 SD); vs-average expectation — Austin Peay offense vs New Haven defense +3.86 pp sack rate, New Haven offense vs Austin Peay defense -1.80 pp sack rate
- Turnover tendencies (giveaways vs takeaways): edge **Austin Peay** (1.44 SD); vs-average expectation — Austin Peay offense vs New Haven defense -1.47 pp turnover rate, New Haven offense vs Austin Peay defense -0.10 pp turnover rate
- Success-rate matchup: edge **Austin Peay** (1.23 SD); vs-average expectation — Austin Peay offense vs New Haven defense +3.84 pp, New Haven offense vs Austin Peay defense -9.40 pp
- Finishing drives (pts per scoring opportunity): edge **Austin Peay** (0.27 SD); vs-average expectation — Austin Peay offense vs New Haven defense -0.03 pts, New Haven offense vs Austin Peay defense -0.34 pts
- QB: Austin Peay Chris Parson (11.6 YPA, 7 TD/1 INT) vs New Haven Jake Tripptree (7.0 YPA, 0 TD/0 INT, NEW STARTER last game)
- Home field: Austin Peay (Elo home bonus 45 rating pts; scoring-model home coefficient 6.4 pts on 2026 data)
- Recent form (last-3 margin vs expectation): Austin Peay +16.8, New Haven -2.2
- Strength of schedule (avg opp net rating): Austin Peay -1.2, New Haven -7.3
- Rest: New Haven 7 days, Austin Peay 7 days; travel distance: Data unavailable
- Weather: Forecast: Intermittent clouds, 72.0°F, gusts 14.0 mph, precip 7.0%
- Injuries/availability: Data unavailable (no public CFB injury feed); QB continuity — New Haven starter share 11%, Austin Peay 100%

</details>

### Alabama A&M @ Jackson State
*2026-10-03 05:00 PM ET · Ladd-Peebles Stadium (Mobile, AL) · neutral site · nan vs nan · records: Alabama A&M 4-1, Jackson State 4-0*

**Prediction: Jackson State**  
Win Probability: Alabama A&M 41% / Jackson State 59%  
Projected Score: Alabama A&M 25–29 Jackson State  
Projected Margin: 3.8 (Jackson State)  
Upset Probability: 41%  
Confidence: 1.7/10  ⚠️ HIGH MODEL DISAGREEMENT

Key Factors:

1. Finishing drives (pts per scoring opportunity): edge **Jackson State** (0.65 SD); vs-average expectation — Jackson State offense vs Alabama A&M defense +0.24 pts, Alabama A&M offense vs Jackson State defense -0.51 pts
2. Turnover tendencies (giveaways vs takeaways): edge **Alabama A&M** (0.53 SD); vs-average expectation — Jackson State offense vs Alabama A&M defense -0.51 pp turnover rate, Alabama A&M offense vs Jackson State defense -1.01 pp turnover rate
3. Pass game vs pass defense (opp-adj pass EPA/play): edge **Jackson State** (0.45 SD); vs-average expectation — Jackson State offense vs Alabama A&M defense +0.07 EPA/play, Alabama A&M offense vs Jackson State defense -0.07 EPA/play

Main Risk: Alabama A&M's best counter is Turnover tendencies (0.53 SD edge)

<details><summary>Model breakdown & matchup detail</summary>

| Model | Elo | Adj. scoring model | Logistic reg. | Gradient boosting | Point-diff regression | **Ensemble** |
|---|---|---|---|---|---|---|
| P(Jackson State win) | 0.810 | 0.627 | 0.568 | 0.601 | 0.573 | **0.588** |
| Home margin | +15.1 | +5.4 | — | +4.7 (GBM reg.) | +3.0 | **+3.8** |

- Elo: Alabama A&M 1165, Jackson State 1376; adj. net points rating: Alabama A&M -8.7, Jackson State -2.4
- Explosive-play rate matchup: edge **Jackson State** (0.28 SD); vs-average expectation — Jackson State offense vs Alabama A&M defense +0.72 pp, Alabama A&M offense vs Jackson State defense -0.62 pp
- Field position / special teams (start field pos.): edge **Jackson State** (0.26 SD); vs-average expectation — Jackson State offense vs Alabama A&M defense +0.64 yds, Alabama A&M offense vs Jackson State defense -0.50 yds
- Run game vs run defense (opp-adj rush EPA/play): edge **Alabama A&M** (0.20 SD); vs-average expectation — Jackson State offense vs Alabama A&M defense -0.13 EPA/play, Alabama A&M offense vs Jackson State defense -0.08 EPA/play
- OL protection vs pass rush (sack rate): edge **Jackson State** (0.15 SD); vs-average expectation — Jackson State offense vs Alabama A&M defense -0.80 pp sack rate, Alabama A&M offense vs Jackson State defense -0.25 pp sack rate
- Overall offense vs defense (opp-adj EPA/play): edge **Jackson State** (0.06 SD); vs-average expectation — Jackson State offense vs Alabama A&M defense -0.00 EPA/play, Alabama A&M offense vs Jackson State defense -0.02 EPA/play
- Success-rate matchup: edge **Alabama A&M** (0.01 SD); vs-average expectation — Jackson State offense vs Alabama A&M defense +0.20 pp, Alabama A&M offense vs Jackson State defense +0.33 pp
- QB: Jackson State Jared Lockhart (11.6 YPA, 11 TD/0 INT) vs Alabama A&M Cornelious Brown IV (8.3 YPA, 11 TD/3 INT)
- Neutral site
- Recent form (last-3 margin vs expectation): Jackson State +4.7, Alabama A&M +21.0
- Strength of schedule (avg opp net rating): Jackson State -24.9, Alabama A&M -21.9
- Rest: Alabama A&M 6 days, Jackson State 6 days; travel distance: Data unavailable
- Weather: Forecast: Cloudy, 79.0°F, gusts 15.0 mph, precip 49.0%
- Injuries/availability: Data unavailable (no public CFB injury feed); QB continuity — Alabama A&M starter share 100%, Jackson State 100%

</details>

### Oregon State @ Colorado State
*2026-10-03 06:00 PM ET · Canvas Stadium (Fort Collins, CO) · Pac-12 · records: Oregon State 2-2, Colorado State 2-2*

**Prediction: Oregon State**  
Win Probability: Oregon State 63% / Colorado State 37%  
Projected Score: Oregon State 32–26 Colorado State  
Projected Margin: 5.1 (Oregon State)  
Upset Probability: 37%  
Confidence: 1.6/10  ⚠️ HIGH MODEL DISAGREEMENT

Key Factors:

1. Explosive-play rate matchup: edge **Oregon State** (1.71 SD); vs-average expectation — Colorado State offense vs Oregon State defense -2.85 pp, Oregon State offense vs Colorado State defense +5.18 pp
2. Turnover tendencies (giveaways vs takeaways): edge **Colorado State** (1.48 SD); vs-average expectation — Colorado State offense vs Oregon State defense -0.66 pp turnover rate, Oregon State offense vs Colorado State defense +0.75 pp turnover rate
3. Success-rate matchup: edge **Oregon State** (0.95 SD); vs-average expectation — Colorado State offense vs Oregon State defense -4.09 pp, Oregon State offense vs Colorado State defense +6.19 pp

Main Risk: models disagree on the winner; Colorado State's best counter is Turnover tendencies (1.48 SD edge); weather (wind/storms) adds variance

<details><summary>Model breakdown & matchup detail</summary>

| Model | Elo | Adj. scoring model | Logistic reg. | Gradient boosting | Point-diff regression | **Ensemble** |
|---|---|---|---|---|---|---|
| P(Colorado State win) | 0.519 | 0.469 | 0.306 | 0.250 | 0.363 | **0.367** |
| Home margin | +1.5 | -1.3 | — | -6.4 (GBM reg.) | -5.7 | **-5.1** |

- Elo: Oregon State 1437, Colorado State 1386; adj. net points rating: Oregon State +20.5, Colorado State +14.6
- Pass game vs pass defense (opp-adj pass EPA/play): edge **Oregon State** (0.94 SD); vs-average expectation — Colorado State offense vs Oregon State defense -0.03 EPA/play, Oregon State offense vs Colorado State defense +0.26 EPA/play
- Overall offense vs defense (opp-adj EPA/play): edge **Oregon State** (0.85 SD); vs-average expectation — Colorado State offense vs Oregon State defense -0.02 EPA/play, Oregon State offense vs Colorado State defense +0.19 EPA/play
- OL protection vs pass rush (sack rate): edge **Oregon State** (0.85 SD); vs-average expectation — Colorado State offense vs Oregon State defense +1.94 pp sack rate, Oregon State offense vs Colorado State defense -1.15 pp sack rate
- Field position / special teams (start field pos.): edge **Colorado State** (0.76 SD); vs-average expectation — Colorado State offense vs Oregon State defense -0.02 yds, Oregon State offense vs Colorado State defense -3.32 yds
- Finishing drives (pts per scoring opportunity): edge **Oregon State** (0.55 SD); vs-average expectation — Colorado State offense vs Oregon State defense +0.22 pts, Oregon State offense vs Colorado State defense +0.85 pts
- Run game vs run defense (opp-adj rush EPA/play): edge **Oregon State** (0.44 SD); vs-average expectation — Colorado State offense vs Oregon State defense +0.00 EPA/play, Oregon State offense vs Colorado State defense +0.11 EPA/play
- QB: Colorado State K'saan Farrar (6.0 YPA, 2 TD/2 INT) vs Oregon State Braden Atkinson (8.7 YPA, 11 TD/2 INT)
- Home field: Colorado State (Elo home bonus 45 rating pts; scoring-model home coefficient 6.4 pts on 2026 data)
- Recent form (last-3 margin vs expectation): Colorado State -0.6, Oregon State +11.4
- Strength of schedule (avg opp net rating): Colorado State +14.2, Oregon State +17.4
- Rest: Oregon State 6 days, Colorado State 7 days; travel distance: Data unavailable
- Weather: Forecast: Mostly sunny, 84.0°F, gusts 28.0 mph, precip 0.0% — flagged (wind/storms)
- Injuries/availability: Data unavailable (no public CFB injury feed); QB continuity — Oregon State starter share 100%, Colorado State 52%

</details>

### Texas Southern @ Florida Atlantic
*2026-10-03 06:00 PM ET · Flagler Credit Union Stadium (Boca Raton, FL) · nan vs American · records: Texas Southern 1-3, Florida Atlantic 3-1*

**Prediction: Florida Atlantic**  
Win Probability: Texas Southern 1% / Florida Atlantic 99%  
Projected Score: Texas Southern 13–52 Florida Atlantic  
Projected Margin: 39.2 (Florida Atlantic)  
Upset Probability: 1%  
Confidence: 9.7/10

Key Factors:

1. OL protection vs pass rush (sack rate): edge **Florida Atlantic** (2.13 SD); vs-average expectation — Florida Atlantic offense vs Texas Southern defense -4.01 pp sack rate, Texas Southern offense vs Florida Atlantic defense +3.74 pp sack rate
2. Field position / special teams (start field pos.): edge **Florida Atlantic** (2.10 SD); vs-average expectation — Florida Atlantic offense vs Texas Southern defense +4.47 yds, Texas Southern offense vs Florida Atlantic defense -4.62 yds
3. Run game vs run defense (opp-adj rush EPA/play): edge **Florida Atlantic** (1.70 SD); vs-average expectation — Florida Atlantic offense vs Texas Southern defense +0.36 EPA/play, Texas Southern offense vs Florida Atlantic defense -0.07 EPA/play

Main Risk: single-game variance (σ ≈ 16 pts on the margin)

<details><summary>Model breakdown & matchup detail</summary>

| Model | Elo | Adj. scoring model | Logistic reg. | Gradient boosting | Point-diff regression | **Ensemble** |
|---|---|---|---|---|---|---|
| P(Florida Atlantic win) | 0.947 | 0.992 | 0.988 | 0.992 | 0.993 | **0.991** |
| Home margin | +29.2 | +40.2 | — | +41.6 (GBM reg.) | +39.6 | **+39.2** |

- Elo: Texas Southern 1086, Florida Atlantic 1478; adj. net points rating: Texas Southern -16.7, Florida Atlantic +13.7
- Success-rate matchup: edge **Florida Atlantic** (1.69 SD); vs-average expectation — Florida Atlantic offense vs Texas Southern defense +10.15 pp, Texas Southern offense vs Florida Atlantic defense -8.06 pp
- Overall offense vs defense (opp-adj EPA/play): edge **Florida Atlantic** (1.42 SD); vs-average expectation — Florida Atlantic offense vs Texas Southern defense +0.26 EPA/play, Texas Southern offense vs Florida Atlantic defense -0.09 EPA/play
- Pass game vs pass defense (opp-adj pass EPA/play): edge **Florida Atlantic** (0.81 SD); vs-average expectation — Florida Atlantic offense vs Texas Southern defense +0.19 EPA/play, Texas Southern offense vs Florida Atlantic defense -0.07 EPA/play
- Finishing drives (pts per scoring opportunity): edge **Florida Atlantic** (0.77 SD); vs-average expectation — Florida Atlantic offense vs Texas Southern defense +0.63 pts, Texas Southern offense vs Florida Atlantic defense -0.27 pts
- Turnover tendencies (giveaways vs takeaways): edge **Florida Atlantic** (0.72 SD); vs-average expectation — Florida Atlantic offense vs Texas Southern defense -0.71 pp turnover rate, Texas Southern offense vs Florida Atlantic defense -0.02 pp turnover rate
- Explosive-play rate matchup: edge **Florida Atlantic** (0.03 SD); vs-average expectation — Florida Atlantic offense vs Texas Southern defense +1.28 pp, Texas Southern offense vs Florida Atlantic defense +1.15 pp
- QB: Florida Atlantic Caden Veltkamp (7.4 YPA, 8 TD/2 INT) vs Texas Southern Cam'Ron McCoy (7.8 YPA, 5 TD/2 INT)
- Home field: Florida Atlantic (Elo home bonus 45 rating pts; scoring-model home coefficient 6.4 pts on 2026 data)
- Recent form (last-3 margin vs expectation): Florida Atlantic +14.3, Texas Southern -12.0
- Strength of schedule (avg opp net rating): Florida Atlantic +15.4, Texas Southern -6.5
- Rest: Texas Southern 7 days, Florida Atlantic 6 days; travel distance: Data unavailable
- Weather: Forecast: Intermittent clouds, 86.0°F, gusts 18.0 mph, precip 33.0%
- Injuries/availability: Data unavailable (no public CFB injury feed); QB continuity — Texas Southern starter share 100%, Florida Atlantic 100%

</details>

### North Carolina A&T @ Bryant
*2026-10-03 06:00 PM ET · Beirne Stadium (Smithfield, RI) · Coastal Athletic Association · records: North Carolina A&T 1-4, Bryant 3-1*

**Prediction: Bryant**  
Win Probability: North Carolina A&T 16% / Bryant 84%  
Projected Score: North Carolina A&T 18–34 Bryant  
Projected Margin: 15.9 (Bryant)  
Upset Probability: 16%  
Confidence: 6.7/10

Key Factors:

1. Finishing drives (pts per scoring opportunity): edge **Bryant** (1.38 SD); vs-average expectation — Bryant offense vs North Carolina A&T defense +0.57 pts, North Carolina A&T offense vs Bryant defense -1.02 pts
2. Pass game vs pass defense (opp-adj pass EPA/play): edge **Bryant** (0.97 SD); vs-average expectation — Bryant offense vs North Carolina A&T defense +0.14 EPA/play, North Carolina A&T offense vs Bryant defense -0.16 EPA/play
3. OL protection vs pass rush (sack rate): edge **Bryant** (0.89 SD); vs-average expectation — Bryant offense vs North Carolina A&T defense -2.82 pp sack rate, North Carolina A&T offense vs Bryant defense +0.43 pp sack rate

Main Risk: North Carolina A&T's best counter is Explosive-play rate matchup (0.33 SD edge); QB change last game (North Carolina A&T) — availability unconfirmed

<details><summary>Model breakdown & matchup detail</summary>

| Model | Elo | Adj. scoring model | Logistic reg. | Gradient boosting | Point-diff regression | **Ensemble** |
|---|---|---|---|---|---|---|
| P(Bryant win) | 0.862 | 0.870 | 0.826 | 0.891 | 0.828 | **0.842** |
| Home margin | +18.9 | +18.8 | — | +12.9 (GBM reg.) | +15.3 | **+15.9** |

- Elo: North Carolina A&T 957, Bryant 1183; adj. net points rating: North Carolina A&T -18.3, Bryant -6.7
- Field position / special teams (start field pos.): edge **Bryant** (0.61 SD); vs-average expectation — Bryant offense vs North Carolina A&T defense +0.84 yds, North Carolina A&T offense vs Bryant defense -1.81 yds
- Success-rate matchup: edge **Bryant** (0.55 SD); vs-average expectation — Bryant offense vs North Carolina A&T defense +3.39 pp, North Carolina A&T offense vs Bryant defense -2.55 pp
- Overall offense vs defense (opp-adj EPA/play): edge **Bryant** (0.54 SD); vs-average expectation — Bryant offense vs North Carolina A&T defense +0.13 EPA/play, North Carolina A&T offense vs Bryant defense -0.00 EPA/play
- Explosive-play rate matchup: edge **North Carolina A&T** (0.33 SD); vs-average expectation — Bryant offense vs North Carolina A&T defense -1.52 pp, North Carolina A&T offense vs Bryant defense +0.02 pp
- Run game vs run defense (opp-adj rush EPA/play): edge **Bryant** (0.30 SD); vs-average expectation — Bryant offense vs North Carolina A&T defense +0.11 EPA/play, North Carolina A&T offense vs Bryant defense +0.03 EPA/play
- Turnover tendencies (giveaways vs takeaways): edge **North Carolina A&T** (0.11 SD); vs-average expectation — Bryant offense vs North Carolina A&T defense -0.71 pp turnover rate, North Carolina A&T offense vs Bryant defense -0.81 pp turnover rate
- QB: Bryant Brennan Myer (6.3 YPA, 2 TD/1 INT) vs North Carolina A&T Kevin White (6.2 YPA, 4 TD/3 INT, NEW STARTER last game)
- Home field: Bryant (Elo home bonus 45 rating pts; scoring-model home coefficient 6.4 pts on 2026 data)
- Recent form (last-3 margin vs expectation): Bryant -3.7, North Carolina A&T +8.3
- Strength of schedule (avg opp net rating): Bryant -5.6, North Carolina A&T -1.5
- Rest: North Carolina A&T 7 days, Bryant 14 days; travel distance: Data unavailable
- Weather: Forecast: Mostly sunny, 63.0°F, gusts 14.0 mph, precip 0.0%
- Injuries/availability: Data unavailable (no public CFB injury feed); QB continuity — North Carolina A&T starter share 61%, Bryant 100%

</details>

### Campbell @ North Carolina Central
*2026-10-03 06:00 PM ET · O'Kelly-Riddick Stadium (Durham, NC) · Coastal Athletic Association vs Mid-Eastern Athletic · records: Campbell 4-1, North Carolina Central 2-3*

**Prediction: Campbell**  
Win Probability: Campbell 68% / North Carolina Central 32%  
Projected Score: Campbell 31–23 North Carolina Central  
Projected Margin: 7.7 (Campbell)  
Upset Probability: 32%  
Confidence: 2.1/10  ⚠️ HIGH MODEL DISAGREEMENT

Key Factors:

1. OL protection vs pass rush (sack rate): edge **Campbell** (1.70 SD); vs-average expectation — North Carolina Central offense vs Campbell defense +4.88 pp sack rate, Campbell offense vs North Carolina Central defense -1.33 pp sack rate
2. Turnover tendencies (giveaways vs takeaways): edge **North Carolina Central** (1.07 SD); vs-average expectation — North Carolina Central offense vs Campbell defense +0.28 pp turnover rate, Campbell offense vs North Carolina Central defense +1.29 pp turnover rate
3. Explosive-play rate matchup: edge **Campbell** (0.63 SD); vs-average expectation — North Carolina Central offense vs Campbell defense +0.27 pp, Campbell offense vs North Carolina Central defense +3.20 pp

Main Risk: models disagree on the winner; North Carolina Central's best counter is Turnover tendencies (1.07 SD edge); QB change last game (North Carolina Central) — availability unconfirmed

<details><summary>Model breakdown & matchup detail</summary>

| Model | Elo | Adj. scoring model | Logistic reg. | Gradient boosting | Point-diff regression | **Ensemble** |
|---|---|---|---|---|---|---|
| P(North Carolina Central win) | 0.507 | 0.416 | 0.283 | 0.342 | 0.291 | **0.324** |
| Home margin | +1.0 | -3.5 | — | -11.5 (GBM reg.) | -8.9 | **-7.7** |

- Elo: Campbell 1213, North Carolina Central 1155; adj. net points rating: Campbell -3.4, North Carolina Central -11.3
- Success-rate matchup: edge **Campbell** (0.59 SD); vs-average expectation — North Carolina Central offense vs Campbell defense -1.31 pp, Campbell offense vs North Carolina Central defense +5.10 pp
- Overall offense vs defense (opp-adj EPA/play): edge **Campbell** (0.39 SD); vs-average expectation — North Carolina Central offense vs Campbell defense -0.03 EPA/play, Campbell offense vs North Carolina Central defense +0.07 EPA/play
- Run game vs run defense (opp-adj rush EPA/play): edge **Campbell** (0.36 SD); vs-average expectation — North Carolina Central offense vs Campbell defense -0.17 EPA/play, Campbell offense vs North Carolina Central defense -0.07 EPA/play
- Pass game vs pass defense (opp-adj pass EPA/play): edge **Campbell** (0.32 SD); vs-average expectation — North Carolina Central offense vs Campbell defense +0.05 EPA/play, Campbell offense vs North Carolina Central defense +0.14 EPA/play
- Finishing drives (pts per scoring opportunity): edge **North Carolina Central** (0.16 SD); vs-average expectation — North Carolina Central offense vs Campbell defense +0.31 pts, Campbell offense vs North Carolina Central defense +0.12 pts
- Field position / special teams (start field pos.): edge **North Carolina Central** (0.05 SD); vs-average expectation — North Carolina Central offense vs Campbell defense +0.87 yds, Campbell offense vs North Carolina Central defense +0.65 yds
- QB: North Carolina Central Cobey Thompkins (7.2 YPA, 1 TD/2 INT, NEW STARTER last game) vs Campbell Kamden Sixkiller (8.4 YPA, 13 TD/4 INT)
- Home field: North Carolina Central (Elo home bonus 45 rating pts; scoring-model home coefficient 6.4 pts on 2026 data)
- Recent form (last-3 margin vs expectation): North Carolina Central -4.1, Campbell +4.2
- Strength of schedule (avg opp net rating): North Carolina Central -8.5, Campbell -3.7
- Rest: Campbell 7 days, North Carolina Central 7 days; travel distance: Data unavailable
- Weather: Forecast: Cloudy, 67.0°F, gusts 20.0 mph, precip 44.0% — flagged (wind/storms)
- Injuries/availability: Data unavailable (no public CFB injury feed); QB continuity — Campbell starter share 100%, North Carolina Central 15%

</details>

### North Alabama @ Eastern Kentucky
*2026-10-03 06:00 PM ET · Roy Kidd Stadium (Richmond, KY) · United Athletic · records: North Alabama 3-2, Eastern Kentucky 1-3*

**Prediction: North Alabama**  
Win Probability: North Alabama 60% / Eastern Kentucky 40%  
Projected Score: North Alabama 25–21 Eastern Kentucky  
Projected Margin: 4.1 (North Alabama)  
Upset Probability: 40%  
Confidence: 1.0/10  ⚠️ HIGH MODEL DISAGREEMENT

Key Factors:

1. Turnover tendencies (giveaways vs takeaways): edge **North Alabama** (1.60 SD); vs-average expectation — Eastern Kentucky offense vs North Alabama defense +0.62 pp turnover rate, North Alabama offense vs Eastern Kentucky defense -0.90 pp turnover rate
2. OL protection vs pass rush (sack rate): edge **North Alabama** (1.30 SD); vs-average expectation — Eastern Kentucky offense vs North Alabama defense +2.77 pp sack rate, North Alabama offense vs Eastern Kentucky defense -1.99 pp sack rate
3. Finishing drives (pts per scoring opportunity): edge **North Alabama** (0.89 SD); vs-average expectation — Eastern Kentucky offense vs North Alabama defense -0.67 pts, North Alabama offense vs Eastern Kentucky defense +0.35 pts

Main Risk: models disagree on the winner; Eastern Kentucky's best counter is Explosive-play rate matchup (0.27 SD edge); QB change last game (Eastern Kentucky) — availability unconfirmed

<details><summary>Model breakdown & matchup detail</summary>

| Model | Elo | Adj. scoring model | Logistic reg. | Gradient boosting | Point-diff regression | **Ensemble** |
|---|---|---|---|---|---|---|
| P(Eastern Kentucky win) | 0.591 | 0.405 | 0.399 | 0.384 | 0.398 | **0.400** |
| Home margin | +4.4 | -4.0 | — | -4.0 (GBM reg.) | -4.2 | **-4.1** |

- Elo: North Alabama 1173, Eastern Kentucky 1168; adj. net points rating: North Alabama -3.7, Eastern Kentucky -12.1
- Pass game vs pass defense (opp-adj pass EPA/play): edge **North Alabama** (0.43 SD); vs-average expectation — Eastern Kentucky offense vs North Alabama defense -0.10 EPA/play, North Alabama offense vs Eastern Kentucky defense +0.04 EPA/play
- Run game vs run defense (opp-adj rush EPA/play): edge **North Alabama** (0.41 SD); vs-average expectation — Eastern Kentucky offense vs North Alabama defense -0.23 EPA/play, North Alabama offense vs Eastern Kentucky defense -0.13 EPA/play
- Success-rate matchup: edge **North Alabama** (0.29 SD); vs-average expectation — Eastern Kentucky offense vs North Alabama defense -5.19 pp, North Alabama offense vs Eastern Kentucky defense -2.10 pp
- Explosive-play rate matchup: edge **Eastern Kentucky** (0.27 SD); vs-average expectation — Eastern Kentucky offense vs North Alabama defense -0.14 pp, North Alabama offense vs Eastern Kentucky defense -1.42 pp
- Field position / special teams (start field pos.): edge **North Alabama** (0.23 SD); vs-average expectation — Eastern Kentucky offense vs North Alabama defense -2.22 yds, North Alabama offense vs Eastern Kentucky defense -1.22 yds
- Overall offense vs defense (opp-adj EPA/play): edge **North Alabama** (0.18 SD); vs-average expectation — Eastern Kentucky offense vs North Alabama defense -0.13 EPA/play, North Alabama offense vs Eastern Kentucky defense -0.08 EPA/play
- QB: Eastern Kentucky Jordyn Potts (7.6 YPA, 1 TD/0 INT, NEW STARTER last game) vs North Alabama Destin Wade (8.7 YPA, 5 TD/2 INT)
- Home field: Eastern Kentucky (Elo home bonus 45 rating pts; scoring-model home coefficient 6.4 pts on 2026 data)
- Recent form (last-3 margin vs expectation): Eastern Kentucky -12.5, North Alabama +1.2
- Strength of schedule (avg opp net rating): Eastern Kentucky -1.0, North Alabama -1.7
- Rest: North Alabama 7 days, Eastern Kentucky 14 days; travel distance: Data unavailable
- Weather: Forecast: Intermittent clouds, 70.0°F, gusts 8.0 mph, precip 0.0%
- Injuries/availability: Data unavailable (no public CFB injury feed); QB continuity — North Alabama starter share 100%, Eastern Kentucky 32%

</details>

### Gardner-Webb @ Charleston Southern
*2026-10-03 06:00 PM ET · Buccaneer Field (Charleston, SC) · Ohio Valley · records: Gardner-Webb 1-4, Charleston Southern 2-2*

**Prediction: Gardner-Webb**  
Win Probability: Gardner-Webb 56% / Charleston Southern 44%  
Projected Score: Gardner-Webb 24–22 Charleston Southern  
Projected Margin: 2.3 (Gardner-Webb)  
Upset Probability: 44%  
Confidence: 1.0/10  ⚠️ HIGH MODEL DISAGREEMENT

Key Factors:

1. OL protection vs pass rush (sack rate): edge **Gardner-Webb** (2.70 SD); vs-average expectation — Charleston Southern offense vs Gardner-Webb defense +7.84 pp sack rate, Gardner-Webb offense vs Charleston Southern defense -2.00 pp sack rate
2. Turnover tendencies (giveaways vs takeaways): edge **Gardner-Webb** (1.31 SD); vs-average expectation — Charleston Southern offense vs Gardner-Webb defense +0.62 pp turnover rate, Gardner-Webb offense vs Charleston Southern defense -0.62 pp turnover rate
3. Overall offense vs defense (opp-adj EPA/play): edge **Gardner-Webb** (0.79 SD); vs-average expectation — Charleston Southern offense vs Gardner-Webb defense -0.24 EPA/play, Gardner-Webb offense vs Charleston Southern defense -0.05 EPA/play

Main Risk: models disagree on the winner; Charleston Southern's best counter is Field position / special teams (0.30 SD edge); QB change last game (Charleston Southern) — availability unconfirmed

<details><summary>Model breakdown & matchup detail</summary>

| Model | Elo | Adj. scoring model | Logistic reg. | Gradient boosting | Point-diff regression | **Ensemble** |
|---|---|---|---|---|---|---|
| P(Charleston Southern win) | 0.537 | 0.460 | 0.414 | 0.464 | 0.439 | **0.440** |
| Home margin | +2.2 | -1.7 | — | -1.8 (GBM reg.) | -2.5 | **-2.3** |

- Elo: Gardner-Webb 1212, Charleston Southern 1172; adj. net points rating: Gardner-Webb +3.2, Charleston Southern -3.1
- Pass game vs pass defense (opp-adj pass EPA/play): edge **Gardner-Webb** (0.66 SD); vs-average expectation — Charleston Southern offense vs Gardner-Webb defense -0.16 EPA/play, Gardner-Webb offense vs Charleston Southern defense +0.05 EPA/play
- Run game vs run defense (opp-adj rush EPA/play): edge **Gardner-Webb** (0.61 SD); vs-average expectation — Charleston Southern offense vs Gardner-Webb defense -0.29 EPA/play, Gardner-Webb offense vs Charleston Southern defense -0.13 EPA/play
- Success-rate matchup: edge **Gardner-Webb** (0.33 SD); vs-average expectation — Charleston Southern offense vs Gardner-Webb defense -7.11 pp, Gardner-Webb offense vs Charleston Southern defense -3.50 pp
- Field position / special teams (start field pos.): edge **Charleston Southern** (0.30 SD); vs-average expectation — Charleston Southern offense vs Gardner-Webb defense +1.25 yds, Gardner-Webb offense vs Charleston Southern defense -0.05 yds
- Explosive-play rate matchup: edge **Charleston Southern** (0.29 SD); vs-average expectation — Charleston Southern offense vs Gardner-Webb defense -0.13 pp, Gardner-Webb offense vs Charleston Southern defense -1.47 pp
- Finishing drives (pts per scoring opportunity): edge **Gardner-Webb** (0.03 SD); vs-average expectation — Charleston Southern offense vs Gardner-Webb defense -1.02 pts, Gardner-Webb offense vs Charleston Southern defense -0.99 pts
- QB: Charleston Southern Jared Echols (8.1 YPA, 0 TD/0 INT, NEW STARTER last game) vs Gardner-Webb Cole Pennington (7.0 YPA, 7 TD/4 INT)
- Home field: Charleston Southern (Elo home bonus 45 rating pts; scoring-model home coefficient 6.4 pts on 2026 data)
- Recent form (last-3 margin vs expectation): Charleston Southern +12.3, Gardner-Webb +16.7
- Strength of schedule (avg opp net rating): Charleston Southern -11.2, Gardner-Webb +4.9
- Rest: Gardner-Webb 7 days, Charleston Southern 13 days; travel distance: Data unavailable
- Weather: Forecast: Thunderstorms, 78.0°F, gusts 10.0 mph, precip 75.0% — flagged (wind/storms)
- Injuries/availability: Data unavailable (no public CFB injury feed); QB continuity — Gardner-Webb starter share 100%, Charleston Southern 13%

</details>

### Incarnate Word @ Stephen F. Austin
*2026-10-03 06:00 PM ET · Homer Bryce Stadium (Nacogdoches, TX) · Southland · records: Incarnate Word 2-2, Stephen F. Austin 5-0*

**Prediction: Stephen F. Austin**  
Win Probability: Incarnate Word 10% / Stephen F. Austin 90%  
Projected Score: Incarnate Word 18–39 Stephen F. Austin  
Projected Margin: 20.3 (Stephen F. Austin)  
Upset Probability: 10%  
Confidence: 8.1/10

Key Factors:

1. Field position / special teams (start field pos.): edge **Stephen F. Austin** (1.36 SD); vs-average expectation — Stephen F. Austin offense vs Incarnate Word defense +4.87 yds, Incarnate Word offense vs Stephen F. Austin defense -1.02 yds
2. Turnover tendencies (giveaways vs takeaways): edge **Stephen F. Austin** (1.01 SD); vs-average expectation — Stephen F. Austin offense vs Incarnate Word defense -0.05 pp turnover rate, Incarnate Word offense vs Stephen F. Austin defense +0.91 pp turnover rate
3. Finishing drives (pts per scoring opportunity): edge **Stephen F. Austin** (0.92 SD); vs-average expectation — Stephen F. Austin offense vs Incarnate Word defense +0.60 pts, Incarnate Word offense vs Stephen F. Austin defense -0.46 pts

Main Risk: Incarnate Word's best counter is Explosive-play rate matchup (0.45 SD edge); weather (wind/storms) adds variance

<details><summary>Model breakdown & matchup detail</summary>

| Model | Elo | Adj. scoring model | Logistic reg. | Gradient boosting | Point-diff regression | **Ensemble** |
|---|---|---|---|---|---|---|
| P(Stephen F. Austin win) | 0.900 | 0.935 | 0.867 | 0.871 | 0.894 | **0.896** |
| Home margin | +22.5 | +25.2 | — | +14.6 (GBM reg.) | +20.3 | **+20.3** |

- Elo: Incarnate Word 1214, Stephen F. Austin 1499; adj. net points rating: Incarnate Word -4.3, Stephen F. Austin +13.0
- OL protection vs pass rush (sack rate): edge **Stephen F. Austin** (0.89 SD); vs-average expectation — Stephen F. Austin offense vs Incarnate Word defense -0.81 pp sack rate, Incarnate Word offense vs Stephen F. Austin defense +2.42 pp sack rate
- Pass game vs pass defense (opp-adj pass EPA/play): edge **Stephen F. Austin** (0.56 SD); vs-average expectation — Stephen F. Austin offense vs Incarnate Word defense +0.25 EPA/play, Incarnate Word offense vs Stephen F. Austin defense +0.07 EPA/play
- Explosive-play rate matchup: edge **Incarnate Word** (0.45 SD); vs-average expectation — Stephen F. Austin offense vs Incarnate Word defense +0.44 pp, Incarnate Word offense vs Stephen F. Austin defense +2.58 pp
- Success-rate matchup: edge **Stephen F. Austin** (0.21 SD); vs-average expectation — Stephen F. Austin offense vs Incarnate Word defense +0.02 pp, Incarnate Word offense vs Stephen F. Austin defense -2.24 pp
- Overall offense vs defense (opp-adj EPA/play): edge **Stephen F. Austin** (0.16 SD); vs-average expectation — Stephen F. Austin offense vs Incarnate Word defense +0.08 EPA/play, Incarnate Word offense vs Stephen F. Austin defense +0.04 EPA/play
- Run game vs run defense (opp-adj rush EPA/play): edge **Incarnate Word** (0.15 SD); vs-average expectation — Stephen F. Austin offense vs Incarnate Word defense -0.12 EPA/play, Incarnate Word offense vs Stephen F. Austin defense -0.08 EPA/play
- QB: Stephen F. Austin Gavin Rutherford (9.3 YPA, 12 TD/5 INT) vs Incarnate Word Emmett Brown (7.2 YPA, 5 TD/5 INT)
- Home field: Stephen F. Austin (Elo home bonus 45 rating pts; scoring-model home coefficient 6.4 pts on 2026 data)
- Recent form (last-3 margin vs expectation): Stephen F. Austin +12.9, Incarnate Word -13.6
- Strength of schedule (avg opp net rating): Stephen F. Austin -9.8, Incarnate Word -6.0
- Rest: Incarnate Word 7 days, Stephen F. Austin 7 days; travel distance: Data unavailable
- Weather: Forecast: Mostly cloudy w/ t-storms, 82.0°F, gusts 14.0 mph, precip 52.0% — flagged (wind/storms)
- Injuries/availability: Data unavailable (no public CFB injury feed); QB continuity — Incarnate Word starter share 100%, Stephen F. Austin 100%

</details>

### Tennessee Tech @ Western Carolina
*2026-10-03 06:00 PM ET · E.J. Whitmire Stadium (Cullowhee, NC) · Southern · records: Tennessee Tech 4-0, Western Carolina 2-3*

**Prediction: Tennessee Tech**  
Win Probability: Tennessee Tech 71% / Western Carolina 29%  
Projected Score: Tennessee Tech 34–25 Western Carolina  
Projected Margin: 8.5 (Tennessee Tech)  
Upset Probability: 29%  
Confidence: 4.9/10

Key Factors:

1. Pass game vs pass defense (opp-adj pass EPA/play): edge **Tennessee Tech** (1.60 SD); vs-average expectation — Western Carolina offense vs Tennessee Tech defense -0.27 EPA/play, Tennessee Tech offense vs Western Carolina defense +0.23 EPA/play
2. Field position / special teams (start field pos.): edge **Tennessee Tech** (1.29 SD); vs-average expectation — Western Carolina offense vs Tennessee Tech defense -1.38 yds, Tennessee Tech offense vs Western Carolina defense +4.20 yds
3. Overall offense vs defense (opp-adj EPA/play): edge **Tennessee Tech** (1.24 SD); vs-average expectation — Western Carolina offense vs Tennessee Tech defense -0.05 EPA/play, Tennessee Tech offense vs Western Carolina defense +0.25 EPA/play

Main Risk: Western Carolina's best counter is Turnover tendencies (0.32 SD edge)

<details><summary>Model breakdown & matchup detail</summary>

| Model | Elo | Adj. scoring model | Logistic reg. | Gradient boosting | Point-diff regression | **Ensemble** |
|---|---|---|---|---|---|---|
| P(Western Carolina win) | 0.307 | 0.308 | 0.243 | 0.295 | 0.312 | **0.292** |
| Home margin | -7.3 | -8.4 | — | -8.8 (GBM reg.) | -8.0 | **-8.5** |

- Elo: Tennessee Tech 1420, Western Carolina 1229; adj. net points rating: Tennessee Tech +7.5, Western Carolina -4.7
- Run game vs run defense (opp-adj rush EPA/play): edge **Tennessee Tech** (0.64 SD); vs-average expectation — Western Carolina offense vs Tennessee Tech defense +0.11 EPA/play, Tennessee Tech offense vs Western Carolina defense +0.27 EPA/play
- Success-rate matchup: edge **Tennessee Tech** (0.58 SD); vs-average expectation — Western Carolina offense vs Tennessee Tech defense -1.98 pp, Tennessee Tech offense vs Western Carolina defense +4.27 pp
- OL protection vs pass rush (sack rate): edge **Tennessee Tech** (0.57 SD); vs-average expectation — Western Carolina offense vs Tennessee Tech defense +2.12 pp sack rate, Tennessee Tech offense vs Western Carolina defense +0.03 pp sack rate
- Explosive-play rate matchup: edge **Tennessee Tech** (0.57 SD); vs-average expectation — Western Carolina offense vs Tennessee Tech defense -1.11 pp, Tennessee Tech offense vs Western Carolina defense +1.57 pp
- Turnover tendencies (giveaways vs takeaways): edge **Western Carolina** (0.32 SD); vs-average expectation — Western Carolina offense vs Tennessee Tech defense -0.33 pp turnover rate, Tennessee Tech offense vs Western Carolina defense -0.02 pp turnover rate
- Finishing drives (pts per scoring opportunity): edge **Tennessee Tech** (0.23 SD); vs-average expectation — Western Carolina offense vs Tennessee Tech defense +0.58 pts, Tennessee Tech offense vs Western Carolina defense +0.84 pts
- QB: Western Carolina Lex Thomas (7.1 YPA, 6 TD/0 INT) vs Tennessee Tech Jax Leatherwood (8.1 YPA, 13 TD/2 INT)
- Home field: Western Carolina (Elo home bonus 45 rating pts; scoring-model home coefficient 6.4 pts on 2026 data)
- Recent form (last-3 margin vs expectation): Western Carolina -8.2, Tennessee Tech +0.7
- Strength of schedule (avg opp net rating): Western Carolina -4.0, Tennessee Tech -4.8
- Rest: Tennessee Tech 7 days, Western Carolina 7 days; travel distance: Data unavailable
- Weather: Forecast: Intermittent clouds, 72.0°F, gusts 4.0 mph, precip 49.0%
- Injuries/availability: Data unavailable (no public CFB injury feed); QB continuity — Tennessee Tech starter share 100%, Western Carolina 100%

</details>

### Arkansas @ Texas A&M
*2026-10-03 07:00 PM ET · Kyle Field (College Station, TX) · Southeastern · records: Arkansas 2-2, Texas A&M 2-2*

**Prediction: Texas A&M**  
Win Probability: Arkansas 17% / Texas A&M 83%  
Projected Score: Arkansas 21–36 Texas A&M  
Projected Margin: 15.2 (Texas A&M)  
Upset Probability: 17%  
Confidence: 7.0/10

Key Factors:

1. Field position / special teams (start field pos.): edge **Texas A&M** (1.01 SD); vs-average expectation — Texas A&M offense vs Arkansas defense +2.19 yds, Arkansas offense vs Texas A&M defense -2.16 yds
2. Run game vs run defense (opp-adj rush EPA/play): edge **Texas A&M** (0.97 SD); vs-average expectation — Texas A&M offense vs Arkansas defense +0.09 EPA/play, Arkansas offense vs Texas A&M defense -0.15 EPA/play
3. Finishing drives (pts per scoring opportunity): edge **Texas A&M** (0.94 SD); vs-average expectation — Texas A&M offense vs Arkansas defense +1.05 pts, Arkansas offense vs Texas A&M defense -0.03 pts

Main Risk: Arkansas's best counter is OL protection vs pass rush (0.48 SD edge)

<details><summary>Model breakdown & matchup detail</summary>

| Model | Elo | Adj. scoring model | Logistic reg. | Gradient boosting | Point-diff regression | **Ensemble** |
|---|---|---|---|---|---|---|
| P(Texas A&M win) | 0.789 | 0.867 | 0.806 | 0.837 | 0.813 | **0.826** |
| Home margin | +13.8 | +18.5 | — | +13.3 (GBM reg.) | +14.4 | **+15.2** |

- Elo: Arkansas 1477, Texas A&M 1622; adj. net points rating: Arkansas +18.2, Texas A&M +29.6
- Overall offense vs defense (opp-adj EPA/play): edge **Texas A&M** (0.81 SD); vs-average expectation — Texas A&M offense vs Arkansas defense +0.11 EPA/play, Arkansas offense vs Texas A&M defense -0.08 EPA/play
- Turnover tendencies (giveaways vs takeaways): edge **Texas A&M** (0.66 SD); vs-average expectation — Texas A&M offense vs Arkansas defense -0.35 pp turnover rate, Arkansas offense vs Texas A&M defense +0.28 pp turnover rate
- Success-rate matchup: edge **Texas A&M** (0.57 SD); vs-average expectation — Texas A&M offense vs Arkansas defense +4.17 pp, Arkansas offense vs Texas A&M defense -1.97 pp
- OL protection vs pass rush (sack rate): edge **Arkansas** (0.48 SD); vs-average expectation — Texas A&M offense vs Arkansas defense +1.20 pp sack rate, Arkansas offense vs Texas A&M defense -0.53 pp sack rate
- Pass game vs pass defense (opp-adj pass EPA/play): edge **Texas A&M** (0.30 SD); vs-average expectation — Texas A&M offense vs Arkansas defense +0.10 EPA/play, Arkansas offense vs Texas A&M defense +0.01 EPA/play
- Explosive-play rate matchup: edge **Arkansas** (0.03 SD); vs-average expectation — Texas A&M offense vs Arkansas defense +0.29 pp, Arkansas offense vs Texas A&M defense +0.45 pp
- QB: Texas A&M Marcel Reed (5.7 YPA, 6 TD/4 INT) vs Arkansas KJ Jackson (7.0 YPA, 5 TD/3 INT)
- Home field: Texas A&M (Elo home bonus 45 rating pts; scoring-model home coefficient 6.4 pts on 2026 data)
- Recent form (last-3 margin vs expectation): Texas A&M -14.1, Arkansas -0.6
- Strength of schedule (avg opp net rating): Texas A&M +24.0, Arkansas +22.7
- Rest: Arkansas 6 days, Texas A&M 6 days; travel distance: Data unavailable
- Weather: Forecast: Intermittent clouds, 83.0°F, gusts 15.0 mph, precip 25.0%
- Injuries/availability: Data unavailable (no public CFB injury feed); QB continuity — Arkansas starter share 100%, Texas A&M 100%

</details>

### BYU @ TCU
*2026-10-03 07:00 PM ET · Amon G. Carter Stadium (Fort Worth, TX) · Big 12 · records: BYU 3-0, TCU 2-2*

**Prediction: BYU**  
Win Probability: BYU 58% / TCU 42%  
Projected Score: BYU 27–24 TCU  
Projected Margin: 3.0 (BYU)  
Upset Probability: 42%  
Confidence: 1.0/10  ⚠️ HIGH MODEL DISAGREEMENT

Key Factors:

1. Turnover tendencies (giveaways vs takeaways): edge **BYU** (2.10 SD); vs-average expectation — TCU offense vs BYU defense +1.17 pp turnover rate, BYU offense vs TCU defense -0.83 pp turnover rate
2. Run game vs run defense (opp-adj rush EPA/play): edge **BYU** (1.09 SD); vs-average expectation — TCU offense vs BYU defense -0.18 EPA/play, BYU offense vs TCU defense +0.09 EPA/play
3. Overall offense vs defense (opp-adj EPA/play): edge **BYU** (0.92 SD); vs-average expectation — TCU offense vs BYU defense -0.09 EPA/play, BYU offense vs TCU defense +0.13 EPA/play

Main Risk: models disagree on the winner; TCU's best counter is OL protection vs pass rush (0.35 SD edge); weather (wind/storms) adds variance

<details><summary>Model breakdown & matchup detail</summary>

| Model | Elo | Adj. scoring model | Logistic reg. | Gradient boosting | Point-diff regression | **Ensemble** |
|---|---|---|---|---|---|---|
| P(TCU win) | 0.316 | 0.508 | 0.349 | 0.310 | 0.419 | **0.415** |
| Home margin | -6.9 | +0.3 | — | -3.8 (GBM reg.) | -3.3 | **-3.0** |

- Elo: BYU 1769, TCU 1584; adj. net points rating: BYU +30.5, TCU +25.9
- Success-rate matchup: edge **BYU** (0.78 SD); vs-average expectation — TCU offense vs BYU defense -1.96 pp, BYU offense vs TCU defense +6.49 pp
- Pass game vs pass defense (opp-adj pass EPA/play): edge **BYU** (0.63 SD); vs-average expectation — TCU offense vs BYU defense +0.01 EPA/play, BYU offense vs TCU defense +0.20 EPA/play
- OL protection vs pass rush (sack rate): edge **TCU** (0.35 SD); vs-average expectation — TCU offense vs BYU defense -1.89 pp sack rate, BYU offense vs TCU defense -0.60 pp sack rate
- Explosive-play rate matchup: edge **BYU** (0.32 SD); vs-average expectation — TCU offense vs BYU defense -0.99 pp, BYU offense vs TCU defense +0.53 pp
- Finishing drives (pts per scoring opportunity): edge **BYU** (0.31 SD); vs-average expectation — TCU offense vs BYU defense -0.02 pts, BYU offense vs TCU defense +0.33 pts
- Field position / special teams (start field pos.): edge **TCU** (0.08 SD); vs-average expectation — TCU offense vs BYU defense -0.45 yds, BYU offense vs TCU defense -0.81 yds
- QB: TCU Jaden Craig (9.2 YPA, 7 TD/3 INT) vs BYU Bear Bachmeier (10.4 YPA, 6 TD/1 INT)
- Home field: TCU (Elo home bonus 45 rating pts; scoring-model home coefficient 6.4 pts on 2026 data)
- Recent form (last-3 margin vs expectation): TCU +8.1, BYU +9.7
- Strength of schedule (avg opp net rating): TCU +12.3, BYU +6.9
- Rest: BYU 13 days, TCU 7 days; travel distance: Data unavailable
- Weather: Forecast: Rain, 71.0°F, gusts 13.0 mph, precip 85.0% — flagged (wind/storms)
- Injuries/availability: Data unavailable (no public CFB injury feed); QB continuity — BYU starter share 100%, TCU 100%

</details>

### UTSA @ Rice
*2026-10-03 07:00 PM ET · First Community Stadium (Houston, TX) · American · records: UTSA 3-1, Rice 1-3*

**Prediction: UTSA**  
Win Probability: UTSA 80% / Rice 20%  
Projected Score: UTSA 34–20 Rice  
Projected Margin: 13.5 (UTSA)  
Upset Probability: 20%  
Confidence: 6.4/10

Key Factors:

1. Field position / special teams (start field pos.): edge **UTSA** (1.50 SD); vs-average expectation — Rice offense vs UTSA defense -3.96 yds, UTSA offense vs Rice defense +2.51 yds
2. Turnover tendencies (giveaways vs takeaways): edge **UTSA** (1.45 SD); vs-average expectation — Rice offense vs UTSA defense +0.44 pp turnover rate, UTSA offense vs Rice defense -0.95 pp turnover rate
3. Pass game vs pass defense (opp-adj pass EPA/play): edge **UTSA** (1.06 SD); vs-average expectation — Rice offense vs UTSA defense -0.29 EPA/play, UTSA offense vs Rice defense +0.05 EPA/play

Main Risk: weather (wind/storms) adds variance

<details><summary>Model breakdown & matchup detail</summary>

| Model | Elo | Adj. scoring model | Logistic reg. | Gradient boosting | Point-diff regression | **Ensemble** |
|---|---|---|---|---|---|---|
| P(Rice win) | 0.192 | 0.273 | 0.160 | 0.147 | 0.183 | **0.197** |
| Home margin | -13.5 | -10.1 | — | -14.0 (GBM reg.) | -14.7 | **-13.5** |

- Elo: UTSA 1629, Rice 1339; adj. net points rating: UTSA +22.7, Rice +9.0
- Overall offense vs defense (opp-adj EPA/play): edge **UTSA** (1.05 SD); vs-average expectation — Rice offense vs UTSA defense -0.08 EPA/play, UTSA offense vs Rice defense +0.17 EPA/play
- Explosive-play rate matchup: edge **UTSA** (0.91 SD); vs-average expectation — Rice offense vs UTSA defense -3.47 pp, UTSA offense vs Rice defense +0.83 pp
- Run game vs run defense (opp-adj rush EPA/play): edge **UTSA** (0.83 SD); vs-average expectation — Rice offense vs UTSA defense +0.07 EPA/play, UTSA offense vs Rice defense +0.28 EPA/play
- OL protection vs pass rush (sack rate): edge **UTSA** (0.74 SD); vs-average expectation — Rice offense vs UTSA defense -0.27 pp sack rate, UTSA offense vs Rice defense -2.96 pp sack rate
- Success-rate matchup: edge **UTSA** (0.73 SD); vs-average expectation — Rice offense vs UTSA defense -2.94 pp, UTSA offense vs Rice defense +4.93 pp
- Finishing drives (pts per scoring opportunity): edge **UTSA** (0.39 SD); vs-average expectation — Rice offense vs UTSA defense -0.06 pts, UTSA offense vs Rice defense +0.38 pts
- QB: Rice Jacurri Brown (6.1 YPA, 3 TD/1 INT) vs UTSA Owen McCown (7.1 YPA, 7 TD/1 INT)
- Home field: Rice (Elo home bonus 45 rating pts; scoring-model home coefficient 6.4 pts on 2026 data)
- Recent form (last-3 margin vs expectation): Rice -4.5, UTSA -0.3
- Strength of schedule (avg opp net rating): Rice +19.4, UTSA +18.0
- Rest: UTSA 7 days, Rice 6 days; travel distance: Data unavailable
- Weather: Forecast: Thunderstorms, 83.0°F, gusts 14.0 mph, precip 57.0% — flagged (wind/storms)
- Injuries/availability: Data unavailable (no public CFB injury feed); QB continuity — UTSA starter share 100%, Rice 100%

</details>

### Chicago State @ Tarleton State
*2026-10-03 07:00 PM ET · Tarleton Memorial Stadium (Stephenville, TX) · FCS Independents vs United Athletic · records: Chicago State 1-4, Tarleton State 4-0*

**Prediction: Tarleton State**  
Win Probability: Chicago State 2% / Tarleton State 98%  
Projected Score: Chicago State 13–48 Tarleton State  
Projected Margin: 35.0 (Tarleton State)  
Upset Probability: 2%  
Confidence: 9.7/10

Key Factors:

1. Field position / special teams (start field pos.): edge **Tarleton State** (1.60 SD); vs-average expectation — Tarleton State offense vs Chicago State defense +3.64 yds, Chicago State offense vs Tarleton State defense -3.30 yds
2. Overall offense vs defense (opp-adj EPA/play): edge **Tarleton State** (1.47 SD); vs-average expectation — Tarleton State offense vs Chicago State defense +0.27 EPA/play, Chicago State offense vs Tarleton State defense -0.10 EPA/play
3. Pass game vs pass defense (opp-adj pass EPA/play): edge **Tarleton State** (1.28 SD); vs-average expectation — Tarleton State offense vs Chicago State defense +0.37 EPA/play, Chicago State offense vs Tarleton State defense -0.03 EPA/play

Main Risk: single-game variance (σ ≈ 16 pts on the margin)

<details><summary>Model breakdown & matchup detail</summary>

| Model | Elo | Adj. scoring model | Logistic reg. | Gradient boosting | Point-diff regression | **Ensemble** |
|---|---|---|---|---|---|---|
| P(Tarleton State win) | 0.975 | 0.985 | 0.973 | 0.989 | 0.985 | **0.982** |
| Home margin | +36.8 | +36.2 | — | +38.9 (GBM reg.) | +35.1 | **+35.0** |

- Elo: Chicago State 985, Tarleton State 1497; adj. net points rating: Chicago State -19.1, Tarleton State +7.8
- Turnover tendencies (giveaways vs takeaways): edge **Tarleton State** (1.25 SD); vs-average expectation — Tarleton State offense vs Chicago State defense -0.69 pp turnover rate, Chicago State offense vs Tarleton State defense +0.50 pp turnover rate
- Run game vs run defense (opp-adj rush EPA/play): edge **Tarleton State** (1.22 SD); vs-average expectation — Tarleton State offense vs Chicago State defense +0.15 EPA/play, Chicago State offense vs Tarleton State defense -0.16 EPA/play
- Success-rate matchup: edge **Tarleton State** (1.10 SD); vs-average expectation — Tarleton State offense vs Chicago State defense +7.47 pp, Chicago State offense vs Tarleton State defense -4.40 pp
- Finishing drives (pts per scoring opportunity): edge **Tarleton State** (1.00 SD); vs-average expectation — Tarleton State offense vs Chicago State defense +1.01 pts, Chicago State offense vs Tarleton State defense -0.14 pts
- Explosive-play rate matchup: edge **Tarleton State** (0.95 SD); vs-average expectation — Tarleton State offense vs Chicago State defense +4.20 pp, Chicago State offense vs Tarleton State defense -0.27 pp
- OL protection vs pass rush (sack rate): edge **Tarleton State** (0.22 SD); vs-average expectation — Tarleton State offense vs Chicago State defense -2.74 pp sack rate, Chicago State offense vs Tarleton State defense -1.94 pp sack rate
- QB: Tarleton State Kaden Anderson (9.6 YPA, 8 TD/0 INT) vs Chicago State Ashton Pannell (6.7 YPA, 10 TD/2 INT)
- Home field: Tarleton State (Elo home bonus 45 rating pts; scoring-model home coefficient 6.4 pts on 2026 data)
- Recent form (last-3 margin vs expectation): Tarleton State +1.5, Chicago State +3.1
- Strength of schedule (avg opp net rating): Tarleton State -5.1, Chicago State -18.9
- Rest: Chicago State 7 days, Tarleton State 14 days; travel distance: Data unavailable
- Weather: Forecast: Cloudy, 71.0°F, gusts 10.0 mph, precip 49.0%
- Injuries/availability: Data unavailable (no public CFB injury feed); QB continuity — Chicago State starter share 100%, Tarleton State 100%

</details>

### Colgate @ Harvard
*2026-10-03 07:00 PM ET · Harvard Stadium (Cambridge, MA) · Patriot League vs Ivy League · records: Colgate 4-1, Harvard 2-0*

**Prediction: Harvard**  
Win Probability: Colgate 43% / Harvard 57%  
Projected Score: Colgate 22–24 Harvard  
Projected Margin: 2.6 (Harvard)  
Upset Probability: 43%  
Confidence: 1.0/10

Key Factors:

1. Turnover tendencies (giveaways vs takeaways): edge **Harvard** (1.01 SD); vs-average expectation — Harvard offense vs Colgate defense +0.20 pp turnover rate, Colgate offense vs Harvard defense +1.17 pp turnover rate
2. Field position / special teams (start field pos.): edge **Colgate** (0.72 SD); vs-average expectation — Harvard offense vs Colgate defense +0.12 yds, Colgate offense vs Harvard defense +3.23 yds
3. Explosive-play rate matchup: edge **Harvard** (0.47 SD); vs-average expectation — Harvard offense vs Colgate defense -1.04 pp, Colgate offense vs Harvard defense -3.25 pp

Main Risk: Colgate's best counter is Field position / special teams (0.72 SD edge); thin 2026 sample for at least one team

<details><summary>Model breakdown & matchup detail</summary>

| Model | Elo | Adj. scoring model | Logistic reg. | Gradient boosting | Point-diff regression | **Ensemble** |
|---|---|---|---|---|---|---|
| P(Harvard win) | 0.740 | 0.567 | 0.607 | 0.620 | 0.542 | **0.571** |
| Home margin | +11.1 | +2.8 | — | +3.1 (GBM reg.) | +1.7 | **+2.6** |

- Elo: Colgate 1312, Harvard 1414; adj. net points rating: Colgate -0.2, Harvard -2.6
- Run game vs run defense (opp-adj rush EPA/play): edge **Harvard** (0.42 SD); vs-average expectation — Harvard offense vs Colgate defense +0.02 EPA/play, Colgate offense vs Harvard defense -0.09 EPA/play
- Overall offense vs defense (opp-adj EPA/play): edge **Harvard** (0.40 SD); vs-average expectation — Harvard offense vs Colgate defense -0.05 EPA/play, Colgate offense vs Harvard defense -0.14 EPA/play
- Success-rate matchup: edge **Colgate** (0.24 SD); vs-average expectation — Harvard offense vs Colgate defense -1.77 pp, Colgate offense vs Harvard defense +0.81 pp
- Pass game vs pass defense (opp-adj pass EPA/play): edge **Colgate** (0.09 SD); vs-average expectation — Harvard offense vs Colgate defense -0.14 EPA/play, Colgate offense vs Harvard defense -0.11 EPA/play
- OL protection vs pass rush (sack rate): edge **Harvard** (0.04 SD); vs-average expectation — Harvard offense vs Colgate defense -0.57 pp sack rate, Colgate offense vs Harvard defense -0.41 pp sack rate
- Finishing drives (pts per scoring opportunity): edge **Colgate** (0.03 SD); vs-average expectation — Harvard offense vs Colgate defense -0.59 pts, Colgate offense vs Harvard defense -0.55 pts
- QB: Harvard Charlie Smith (4.9 YPA, 3 TD/2 INT) vs Colgate Jake Stearney (7.4 YPA, 8 TD/6 INT)
- Home field: Harvard (Elo home bonus 45 rating pts; scoring-model home coefficient 6.4 pts on 2026 data)
- Recent form (last-3 margin vs expectation): Harvard +14.9, Colgate +4.9
- Strength of schedule (avg opp net rating): Harvard -13.9, Colgate -7.3
- Rest: Colgate 7 days, Harvard 8 days; travel distance: Data unavailable
- Weather: Forecast: Mostly clear, 59.0°F, gusts 13.0 mph, precip 0.0%
- Injuries/availability: Data unavailable (no public CFB injury feed); QB continuity — Colgate starter share 100%, Harvard 100%

</details>

### South Dakota State @ Illinois State
*2026-10-03 07:00 PM ET · Hancock Stadium (Normal, IL) · Missouri Valley Football · records: South Dakota State 4-1, Illinois State 4-1*

**Prediction: Illinois State**  
Win Probability: South Dakota State 46% / Illinois State 54%  
Projected Score: South Dakota State 26–28 Illinois State  
Projected Margin: 1.8 (Illinois State)  
Upset Probability: 46%  
Confidence: 1.0/10  ⚠️ HIGH MODEL DISAGREEMENT

Key Factors:

1. Run game vs run defense (opp-adj rush EPA/play): edge **South Dakota State** (1.52 SD); vs-average expectation — Illinois State offense vs South Dakota State defense -0.20 EPA/play, South Dakota State offense vs Illinois State defense +0.19 EPA/play
2. Turnover tendencies (giveaways vs takeaways): edge **South Dakota State** (1.11 SD); vs-average expectation — Illinois State offense vs South Dakota State defense +0.70 pp turnover rate, South Dakota State offense vs Illinois State defense -0.36 pp turnover rate
3. Overall offense vs defense (opp-adj EPA/play): edge **South Dakota State** (0.70 SD); vs-average expectation — Illinois State offense vs South Dakota State defense -0.06 EPA/play, South Dakota State offense vs Illinois State defense +0.11 EPA/play

Main Risk: models disagree on the winner; South Dakota State's best counter is Run game vs run defense (1.52 SD edge); QB change last game (South Dakota State) — availability unconfirmed

<details><summary>Model breakdown & matchup detail</summary>

| Model | Elo | Adj. scoring model | Logistic reg. | Gradient boosting | Point-diff regression | **Ensemble** |
|---|---|---|---|---|---|---|
| P(Illinois State win) | 0.678 | 0.626 | 0.487 | 0.540 | 0.509 | **0.535** |
| Home margin | +8.1 | +5.4 | — | +2.4 (GBM reg.) | +0.4 | **+1.8** |

- Elo: South Dakota State 1400, Illinois State 1454; adj. net points rating: South Dakota State +12.8, Illinois State +12.7
- Finishing drives (pts per scoring opportunity): edge **Illinois State** (0.60 SD); vs-average expectation — Illinois State offense vs South Dakota State defense +0.43 pts, South Dakota State offense vs Illinois State defense -0.27 pts
- Field position / special teams (start field pos.): edge **South Dakota State** (0.56 SD); vs-average expectation — Illinois State offense vs South Dakota State defense -1.99 yds, South Dakota State offense vs Illinois State defense +0.45 yds
- Explosive-play rate matchup: edge **South Dakota State** (0.50 SD); vs-average expectation — Illinois State offense vs South Dakota State defense -2.19 pp, South Dakota State offense vs Illinois State defense +0.15 pp
- Success-rate matchup: edge **South Dakota State** (0.35 SD); vs-average expectation — Illinois State offense vs South Dakota State defense -0.69 pp, South Dakota State offense vs Illinois State defense +3.11 pp
- OL protection vs pass rush (sack rate): edge **South Dakota State** (0.16 SD); vs-average expectation — Illinois State offense vs South Dakota State defense -0.65 pp sack rate, South Dakota State offense vs Illinois State defense -1.24 pp sack rate
- Pass game vs pass defense (opp-adj pass EPA/play): edge **Illinois State** (0.02 SD); vs-average expectation — Illinois State offense vs South Dakota State defense +0.08 EPA/play, South Dakota State offense vs Illinois State defense +0.07 EPA/play
- QB: Illinois State Gage Roy (6.3 YPA, 6 TD/0 INT) vs South Dakota State Josh Holst (8.0 YPA, 1 TD/0 INT, NEW STARTER last game)
- Home field: Illinois State (Elo home bonus 45 rating pts; scoring-model home coefficient 6.4 pts on 2026 data)
- Recent form (last-3 margin vs expectation): Illinois State -3.7, South Dakota State +1.4
- Strength of schedule (avg opp net rating): Illinois State -9.9, South Dakota State -3.6
- Rest: South Dakota State 7 days, Illinois State 7 days; travel distance: Data unavailable
- Weather: Forecast: Sunny, 65.0°F, gusts 8.0 mph, precip 0.0%
- Injuries/availability: Data unavailable (no public CFB injury feed); QB continuity — South Dakota State starter share 20%, Illinois State 79%

</details>

### UT Rio Grande Valley @ East Texas A&M
*2026-10-03 07:00 PM ET · Memorial Stadium (Commerce, TX) · Southland · records: UT Rio Grande Valley 2-2, East Texas A&M 1-4*

**Prediction: UT Rio Grande Valley**  
Win Probability: UT Rio Grande Valley 68% / East Texas A&M 32%  
Projected Score: UT Rio Grande Valley 28–20 East Texas A&M  
Projected Margin: 7.6 (UT Rio Grande Valley)  
Upset Probability: 32%  
Confidence: 3.7/10

Key Factors:

1. Turnover tendencies (giveaways vs takeaways): edge **UT Rio Grande Valley** (1.29 SD); vs-average expectation — East Texas A&M offense vs UT Rio Grande Valley defense +0.97 pp turnover rate, UT Rio Grande Valley offense vs East Texas A&M defense -0.26 pp turnover rate
2. Finishing drives (pts per scoring opportunity): edge **UT Rio Grande Valley** (0.86 SD); vs-average expectation — East Texas A&M offense vs UT Rio Grande Valley defense -1.13 pts, UT Rio Grande Valley offense vs East Texas A&M defense -0.13 pts
3. Pass game vs pass defense (opp-adj pass EPA/play): edge **UT Rio Grande Valley** (0.60 SD); vs-average expectation — East Texas A&M offense vs UT Rio Grande Valley defense -0.21 EPA/play, UT Rio Grande Valley offense vs East Texas A&M defense -0.02 EPA/play

Main Risk: East Texas A&M's best counter is Run game vs run defense (0.04 SD edge); QB change last game (East Texas A&M, UT Rio Grande Valley) — availability unconfirmed

<details><summary>Model breakdown & matchup detail</summary>

| Model | Elo | Adj. scoring model | Logistic reg. | Gradient boosting | Point-diff regression | **Ensemble** |
|---|---|---|---|---|---|---|
| P(East Texas A&M win) | 0.242 | 0.361 | 0.306 | 0.242 | 0.311 | **0.317** |
| Home margin | -10.6 | -5.9 | — | -8.2 (GBM reg.) | -8.0 | **-7.6** |

- Elo: UT Rio Grande Valley 1287, East Texas A&M 1043; adj. net points rating: UT Rio Grande Valley -0.6, East Texas A&M -10.7
- OL protection vs pass rush (sack rate): edge **UT Rio Grande Valley** (0.53 SD); vs-average expectation — East Texas A&M offense vs UT Rio Grande Valley defense +0.78 pp sack rate, UT Rio Grande Valley offense vs East Texas A&M defense -1.16 pp sack rate
- Overall offense vs defense (opp-adj EPA/play): edge **UT Rio Grande Valley** (0.38 SD); vs-average expectation — East Texas A&M offense vs UT Rio Grande Valley defense -0.15 EPA/play, UT Rio Grande Valley offense vs East Texas A&M defense -0.05 EPA/play
- Field position / special teams (start field pos.): edge **UT Rio Grande Valley** (0.31 SD); vs-average expectation — East Texas A&M offense vs UT Rio Grande Valley defense -0.37 yds, UT Rio Grande Valley offense vs East Texas A&M defense +0.96 yds
- Explosive-play rate matchup: edge **UT Rio Grande Valley** (0.08 SD); vs-average expectation — East Texas A&M offense vs UT Rio Grande Valley defense -1.58 pp, UT Rio Grande Valley offense vs East Texas A&M defense -1.21 pp
- Run game vs run defense (opp-adj rush EPA/play): edge **East Texas A&M** (0.04 SD); vs-average expectation — East Texas A&M offense vs UT Rio Grande Valley defense -0.07 EPA/play, UT Rio Grande Valley offense vs East Texas A&M defense -0.09 EPA/play
- Success-rate matchup: edge **East Texas A&M** (0.04 SD); vs-average expectation — East Texas A&M offense vs UT Rio Grande Valley defense +1.37 pp, UT Rio Grande Valley offense vs East Texas A&M defense +0.92 pp
- QB: East Texas A&M Eric Rodriguez (8.3 YPA, 4 TD/6 INT, NEW STARTER last game) vs UT Rio Grande Valley Aidan Jakobsohn (8.8 YPA, 2 TD/1 INT, NEW STARTER last game)
- Home field: East Texas A&M (Elo home bonus 45 rating pts; scoring-model home coefficient 6.4 pts on 2026 data)
- Recent form (last-3 margin vs expectation): East Texas A&M -0.1, UT Rio Grande Valley -3.1
- Strength of schedule (avg opp net rating): East Texas A&M -1.2, UT Rio Grande Valley -8.3
- Rest: UT Rio Grande Valley 13 days, East Texas A&M 7 days; travel distance: Data unavailable
- Weather: Forecast: Mostly cloudy w/ showers, 72.0°F, gusts 8.0 mph, precip 80.0%
- Injuries/availability: Data unavailable (no public CFB injury feed); QB continuity — UT Rio Grande Valley starter share 38%, East Texas A&M 62%

</details>

### Arkansas-Pine Bluff @ Southern
*2026-10-03 07:00 PM ET · A.W. Mumford Stadium (Baton Rouge, LA) · nan vs nan · records: Arkansas-Pine Bluff 2-2, Southern 2-3*

**Prediction: Arkansas-Pine Bluff**  
Win Probability: Arkansas-Pine Bluff 51% / Southern 49%  
Projected Score: Arkansas-Pine Bluff 28–29 Southern  
Projected Margin: 0.2 (Arkansas-Pine Bluff)  
Upset Probability: 49%  
Confidence: 1.0/10  ⚠️ HIGH MODEL DISAGREEMENT

Key Factors:

1. Turnover tendencies (giveaways vs takeaways): edge **Arkansas-Pine Bluff** (0.99 SD); vs-average expectation — Southern offense vs Arkansas-Pine Bluff defense +0.26 pp turnover rate, Arkansas-Pine Bluff offense vs Southern defense -0.69 pp turnover rate
2. Field position / special teams (start field pos.): edge **Arkansas-Pine Bluff** (0.98 SD); vs-average expectation — Southern offense vs Arkansas-Pine Bluff defense -1.32 yds, Arkansas-Pine Bluff offense vs Southern defense +2.90 yds
3. Run game vs run defense (opp-adj rush EPA/play): edge **Arkansas-Pine Bluff** (0.80 SD); vs-average expectation — Southern offense vs Arkansas-Pine Bluff defense -0.18 EPA/play, Arkansas-Pine Bluff offense vs Southern defense +0.02 EPA/play

Main Risk: models disagree on the winner; Southern's best counter is Pass game vs pass defense (0.75 SD edge); weather (wind/storms) adds variance

<details><summary>Model breakdown & matchup detail</summary>

| Model | Elo | Adj. scoring model | Logistic reg. | Gradient boosting | Point-diff regression | **Ensemble** |
|---|---|---|---|---|---|---|
| P(Southern win) | 0.570 | 0.556 | 0.450 | 0.544 | 0.470 | **0.492** |
| Home margin | +3.5 | +2.3 | — | +3.0 (GBM reg.) | -1.2 | **+0.2** |

- Elo: Arkansas-Pine Bluff 1042, Southern 1023; adj. net points rating: Arkansas-Pine Bluff -16.3, Southern -19.1
- Pass game vs pass defense (opp-adj pass EPA/play): edge **Southern** (0.75 SD); vs-average expectation — Southern offense vs Arkansas-Pine Bluff defense +0.20 EPA/play, Arkansas-Pine Bluff offense vs Southern defense -0.03 EPA/play
- Success-rate matchup: edge **Southern** (0.50 SD); vs-average expectation — Southern offense vs Arkansas-Pine Bluff defense +1.45 pp, Arkansas-Pine Bluff offense vs Southern defense -3.89 pp
- OL protection vs pass rush (sack rate): edge **Southern** (0.14 SD); vs-average expectation — Southern offense vs Arkansas-Pine Bluff defense -1.16 pp sack rate, Arkansas-Pine Bluff offense vs Southern defense -0.63 pp sack rate
- Overall offense vs defense (opp-adj EPA/play): edge **Southern** (0.12 SD); vs-average expectation — Southern offense vs Arkansas-Pine Bluff defense +0.02 EPA/play, Arkansas-Pine Bluff offense vs Southern defense -0.01 EPA/play
- Finishing drives (pts per scoring opportunity): edge **Southern** (0.10 SD); vs-average expectation — Southern offense vs Arkansas-Pine Bluff defense +0.81 pts, Arkansas-Pine Bluff offense vs Southern defense +0.70 pts
- Explosive-play rate matchup: edge **Arkansas-Pine Bluff** (0.10 SD); vs-average expectation — Southern offense vs Arkansas-Pine Bluff defense -0.84 pp, Arkansas-Pine Bluff offense vs Southern defense -0.39 pp
- QB: Southern Christian Johnson (6.2 YPA, 6 TD/1 INT) vs Arkansas-Pine Bluff Garrison Davis (7.7 YPA, 7 TD/3 INT)
- Home field: Southern (Elo home bonus 45 rating pts; scoring-model home coefficient 6.4 pts on 2026 data)
- Recent form (last-3 margin vs expectation): Southern -5.4, Arkansas-Pine Bluff -3.7
- Strength of schedule (avg opp net rating): Southern -7.3, Arkansas-Pine Bluff -11.4
- Rest: Arkansas-Pine Bluff 14 days, Southern 7 days; travel distance: Data unavailable
- Weather: Forecast: Thunderstorms, 80.0°F, gusts 12.0 mph, precip 75.0% — flagged (wind/storms)
- Injuries/availability: Data unavailable (no public CFB injury feed); QB continuity — Arkansas-Pine Bluff starter share 100%, Southern 100%

</details>

### Northwestern State @ Lamar
*2026-10-03 07:00 PM ET · Provost Umphrey Stadium (Beaumont, TX) · Southland · records: Northwestern State 0-5, Lamar 3-1*

**Prediction: Lamar**  
Win Probability: Northwestern State 1% / Lamar 99%  
Projected Score: Northwestern State 10–48 Lamar  
Projected Margin: 38.3 (Lamar)  
Upset Probability: 1%  
Confidence: 9.8/10

Key Factors:

1. Run game vs run defense (opp-adj rush EPA/play): edge **Lamar** (2.75 SD); vs-average expectation — Lamar offense vs Northwestern State defense +0.38 EPA/play, Northwestern State offense vs Lamar defense -0.32 EPA/play
2. Overall offense vs defense (opp-adj EPA/play): edge **Lamar** (2.18 SD); vs-average expectation — Lamar offense vs Northwestern State defense +0.31 EPA/play, Northwestern State offense vs Lamar defense -0.23 EPA/play
3. Field position / special teams (start field pos.): edge **Lamar** (2.03 SD); vs-average expectation — Lamar offense vs Northwestern State defense +6.21 yds, Northwestern State offense vs Lamar defense -2.58 yds

Main Risk: weather (wind/storms) adds variance

<details><summary>Model breakdown & matchup detail</summary>

| Model | Elo | Adj. scoring model | Logistic reg. | Gradient boosting | Point-diff regression | **Ensemble** |
|---|---|---|---|---|---|---|
| P(Lamar win) | 0.972 | 0.995 | 0.984 | 0.993 | 0.991 | **0.990** |
| Home margin | +35.9 | +42.5 | — | +37.5 (GBM reg.) | +38.2 | **+38.3** |

- Elo: Northwestern State 821, Lamar 1319; adj. net points rating: Northwestern State -31.5, Lamar +0.9
- Success-rate matchup: edge **Lamar** (1.50 SD); vs-average expectation — Lamar offense vs Northwestern State defense +7.26 pp, Northwestern State offense vs Lamar defense -8.94 pp
- Pass game vs pass defense (opp-adj pass EPA/play): edge **Lamar** (1.46 SD); vs-average expectation — Lamar offense vs Northwestern State defense +0.32 EPA/play, Northwestern State offense vs Lamar defense -0.14 EPA/play
- Explosive-play rate matchup: edge **Lamar** (1.38 SD); vs-average expectation — Lamar offense vs Northwestern State defense +3.98 pp, Northwestern State offense vs Lamar defense -2.53 pp
- Finishing drives (pts per scoring opportunity): edge **Lamar** (1.32 SD); vs-average expectation — Lamar offense vs Northwestern State defense +0.24 pts, Northwestern State offense vs Lamar defense -1.28 pts
- Turnover tendencies (giveaways vs takeaways): edge **Lamar** (1.19 SD); vs-average expectation — Lamar offense vs Northwestern State defense -0.90 pp turnover rate, Northwestern State offense vs Lamar defense +0.24 pp turnover rate
- OL protection vs pass rush (sack rate): edge **Lamar** (1.09 SD); vs-average expectation — Lamar offense vs Northwestern State defense -1.75 pp sack rate, Northwestern State offense vs Lamar defense +2.24 pp sack rate
- QB: Lamar Aiden McCown (7.1 YPA, 7 TD/4 INT) vs Northwestern State Zach Wilcke (5.7 YPA, 6 TD/2 INT)
- Home field: Lamar (Elo home bonus 45 rating pts; scoring-model home coefficient 6.4 pts on 2026 data)
- Recent form (last-3 margin vs expectation): Lamar +9.1, Northwestern State -0.5
- Strength of schedule (avg opp net rating): Lamar -1.4, Northwestern State -6.7
- Rest: Northwestern State 7 days, Lamar 7 days; travel distance: Data unavailable
- Weather: Forecast: Thunderstorms, 80.0°F, gusts 10.0 mph, precip 66.0% — flagged (wind/storms)
- Injuries/availability: Data unavailable (no public CFB injury feed); QB continuity — Northwestern State starter share 65%, Lamar 100%

</details>

### Nicholls @ Houston Christian
*2026-10-03 07:00 PM ET · Husky Stadium (TX) (Houston, TX) · Southland · records: Nicholls 2-3, Houston Christian 2-3*

**Prediction: Houston Christian**  
Win Probability: Nicholls 15% / Houston Christian 85%  
Projected Score: Nicholls 19–35 Houston Christian  
Projected Margin: 16.9 (Houston Christian)  
Upset Probability: 15%  
Confidence: 6.3/10  ⚠️ HIGH MODEL DISAGREEMENT

Key Factors:

1. Explosive-play rate matchup: edge **Houston Christian** (1.25 SD); vs-average expectation — Houston Christian offense vs Nicholls defense +3.47 pp, Nicholls offense vs Houston Christian defense -2.40 pp
2. Field position / special teams (start field pos.): edge **Houston Christian** (0.82 SD); vs-average expectation — Houston Christian offense vs Nicholls defense +3.85 yds, Nicholls offense vs Houston Christian defense +0.28 yds
3. Turnover tendencies (giveaways vs takeaways): edge **Nicholls** (0.68 SD); vs-average expectation — Houston Christian offense vs Nicholls defense +0.60 pp turnover rate, Nicholls offense vs Houston Christian defense -0.05 pp turnover rate

Main Risk: Nicholls's best counter is Turnover tendencies (0.68 SD edge)

<details><summary>Model breakdown & matchup detail</summary>

| Model | Elo | Adj. scoring model | Logistic reg. | Gradient boosting | Point-diff regression | **Ensemble** |
|---|---|---|---|---|---|---|
| P(Houston Christian win) | 0.542 | 0.889 | 0.821 | 0.805 | 0.852 | **0.849** |
| Home margin | +2.4 | +20.3 | — | +14.0 (GBM reg.) | +16.9 | **+16.9** |

- Elo: Nicholls 1093, Houston Christian 1056; adj. net points rating: Nicholls -17.4, Houston Christian -4.4
- Finishing drives (pts per scoring opportunity): edge **Houston Christian** (0.52 SD); vs-average expectation — Houston Christian offense vs Nicholls defense +0.13 pts, Nicholls offense vs Houston Christian defense -0.47 pts
- Pass game vs pass defense (opp-adj pass EPA/play): edge **Houston Christian** (0.41 SD); vs-average expectation — Houston Christian offense vs Nicholls defense +0.00 EPA/play, Nicholls offense vs Houston Christian defense -0.13 EPA/play
- Run game vs run defense (opp-adj rush EPA/play): edge **Houston Christian** (0.39 SD); vs-average expectation — Houston Christian offense vs Nicholls defense +0.10 EPA/play, Nicholls offense vs Houston Christian defense +0.01 EPA/play
- Success-rate matchup: edge **Houston Christian** (0.28 SD); vs-average expectation — Houston Christian offense vs Nicholls defense +1.80 pp, Nicholls offense vs Houston Christian defense -1.23 pp
- Overall offense vs defense (opp-adj EPA/play): edge **Houston Christian** (0.22 SD); vs-average expectation — Houston Christian offense vs Nicholls defense +0.00 EPA/play, Nicholls offense vs Houston Christian defense -0.05 EPA/play
- OL protection vs pass rush (sack rate): edge **Houston Christian** (0.01 SD); vs-average expectation — Houston Christian offense vs Nicholls defense -3.43 pp sack rate, Nicholls offense vs Houston Christian defense -3.38 pp sack rate
- QB: Houston Christian Cutter Stewart (7.5 YPA, 7 TD/2 INT) vs Nicholls Ean Rodrigue (5.8 YPA, 9 TD/4 INT)
- Home field: Houston Christian (Elo home bonus 45 rating pts; scoring-model home coefficient 6.4 pts on 2026 data)
- Recent form (last-3 margin vs expectation): Houston Christian +11.3, Nicholls -18.5
- Strength of schedule (avg opp net rating): Houston Christian -4.7, Nicholls +1.8
- Rest: Nicholls 7 days, Houston Christian 6 days; travel distance: Data unavailable
- Weather: Forecast: Cloudy, 82.0°F, gusts 9.0 mph, precip 49.0%
- Injuries/availability: Data unavailable (no public CFB injury feed); QB continuity — Nicholls starter share 100%, Houston Christian 100%

</details>

### Georgia Southern @ Coastal Carolina
*2026-10-03 07:00 PM ET · Brooks Stadium (SC) (Conway, SC) · nan vs nan · records: Georgia Southern 1-3, Coastal Carolina 1-3*

**Prediction: Georgia Southern**  
Win Probability: Georgia Southern 52% / Coastal Carolina 48%  
Projected Score: Georgia Southern 26–25 Coastal Carolina  
Projected Margin: 0.9 (Georgia Southern)  
Upset Probability: 48%  
Confidence: 1.0/10  ⚠️ HIGH MODEL DISAGREEMENT

Key Factors:

1. Success-rate matchup: edge **Georgia Southern** (0.73 SD); vs-average expectation — Coastal Carolina offense vs Georgia Southern defense -3.10 pp, Georgia Southern offense vs Coastal Carolina defense +4.75 pp
2. Turnover tendencies (giveaways vs takeaways): edge **Coastal Carolina** (0.60 SD); vs-average expectation — Coastal Carolina offense vs Georgia Southern defense -0.80 pp turnover rate, Georgia Southern offense vs Coastal Carolina defense -0.23 pp turnover rate
3. Pass game vs pass defense (opp-adj pass EPA/play): edge **Georgia Southern** (0.58 SD); vs-average expectation — Coastal Carolina offense vs Georgia Southern defense +0.02 EPA/play, Georgia Southern offense vs Coastal Carolina defense +0.20 EPA/play

Main Risk: models disagree on the winner; Coastal Carolina's best counter is Turnover tendencies (0.60 SD edge)

<details><summary>Model breakdown & matchup detail</summary>

| Model | Elo | Adj. scoring model | Logistic reg. | Gradient boosting | Point-diff regression | **Ensemble** |
|---|---|---|---|---|---|---|
| P(Coastal Carolina win) | 0.447 | 0.540 | 0.466 | 0.399 | 0.471 | **0.482** |
| Home margin | -1.4 | +1.7 | — | -3.9 (GBM reg.) | -1.2 | **-0.9** |

- Elo: Georgia Southern 1423, Coastal Carolina 1326; adj. net points rating: Georgia Southern +14.7, Coastal Carolina +11.3
- OL protection vs pass rush (sack rate): edge **Georgia Southern** (0.49 SD); vs-average expectation — Coastal Carolina offense vs Georgia Southern defense +1.19 pp sack rate, Georgia Southern offense vs Coastal Carolina defense -0.59 pp sack rate
- Explosive-play rate matchup: edge **Coastal Carolina** (0.38 SD); vs-average expectation — Coastal Carolina offense vs Georgia Southern defense +0.23 pp, Georgia Southern offense vs Coastal Carolina defense -1.55 pp
- Field position / special teams (start field pos.): edge **Coastal Carolina** (0.35 SD); vs-average expectation — Coastal Carolina offense vs Georgia Southern defense -0.74 yds, Georgia Southern offense vs Coastal Carolina defense -2.24 yds
- Overall offense vs defense (opp-adj EPA/play): edge **Georgia Southern** (0.32 SD); vs-average expectation — Coastal Carolina offense vs Georgia Southern defense -0.01 EPA/play, Georgia Southern offense vs Coastal Carolina defense +0.07 EPA/play
- Finishing drives (pts per scoring opportunity): edge **Coastal Carolina** (0.11 SD); vs-average expectation — Coastal Carolina offense vs Georgia Southern defense +0.11 pts, Georgia Southern offense vs Coastal Carolina defense -0.02 pts
- Run game vs run defense (opp-adj rush EPA/play): edge **Georgia Southern** (0.03 SD); vs-average expectation — Coastal Carolina offense vs Georgia Southern defense -0.02 EPA/play, Georgia Southern offense vs Coastal Carolina defense -0.01 EPA/play
- QB: Coastal Carolina Deuce Bailey (8.5 YPA, 10 TD/2 INT) vs Georgia Southern Max Johnson (6.3 YPA, 5 TD/1 INT)
- Home field: Coastal Carolina (Elo home bonus 45 rating pts; scoring-model home coefficient 6.4 pts on 2026 data)
- Recent form (last-3 margin vs expectation): Coastal Carolina -5.7, Georgia Southern -2.5
- Strength of schedule (avg opp net rating): Coastal Carolina +10.1, Georgia Southern +15.8
- Rest: Georgia Southern 7 days, Coastal Carolina 8 days; travel distance: Data unavailable
- Weather: Forecast: Cloudy, 78.0°F, gusts 8.0 mph, precip 49.0%
- Injuries/availability: Data unavailable (no public CFB injury feed); QB continuity — Georgia Southern starter share 100%, Coastal Carolina 100%

</details>

### UL Monroe @ South Alabama
*2026-10-03 07:00 PM ET · Hancock Whitney Stadium (Mobile, AL) · nan vs nan · records: UL Monroe 0-4, South Alabama 2-2*

**Prediction: South Alabama**  
Win Probability: UL Monroe 12% / South Alabama 88%  
Projected Score: UL Monroe 21–40 South Alabama  
Projected Margin: 19.1 (South Alabama)  
Upset Probability: 12%  
Confidence: 7.5/10

Key Factors:

1. Turnover tendencies (giveaways vs takeaways): edge **South Alabama** (1.39 SD); vs-average expectation — South Alabama offense vs UL Monroe defense -1.26 pp turnover rate, UL Monroe offense vs South Alabama defense +0.06 pp turnover rate
2. Pass game vs pass defense (opp-adj pass EPA/play): edge **South Alabama** (0.96 SD); vs-average expectation — South Alabama offense vs UL Monroe defense +0.19 EPA/play, UL Monroe offense vs South Alabama defense -0.11 EPA/play
3. Field position / special teams (start field pos.): edge **South Alabama** (0.71 SD); vs-average expectation — South Alabama offense vs UL Monroe defense +0.75 yds, UL Monroe offense vs South Alabama defense -2.30 yds

Main Risk: UL Monroe's best counter is Explosive-play rate matchup (0.01 SD edge); QB change last game (UL Monroe) — availability unconfirmed; weather (wind/storms) adds variance

<details><summary>Model breakdown & matchup detail</summary>

| Model | Elo | Adj. scoring model | Logistic reg. | Gradient boosting | Point-diff regression | **Ensemble** |
|---|---|---|---|---|---|---|
| P(South Alabama win) | 0.890 | 0.918 | 0.874 | 0.884 | 0.868 | **0.883** |
| Home margin | +21.4 | +23.2 | — | +16.3 (GBM reg.) | +18.1 | **+19.1** |

- Elo: UL Monroe 1175, South Alabama 1441; adj. net points rating: UL Monroe -1.1, South Alabama +14.4
- Overall offense vs defense (opp-adj EPA/play): edge **South Alabama** (0.60 SD); vs-average expectation — South Alabama offense vs UL Monroe defense +0.17 EPA/play, UL Monroe offense vs South Alabama defense +0.02 EPA/play
- Success-rate matchup: edge **South Alabama** (0.52 SD); vs-average expectation — South Alabama offense vs UL Monroe defense +4.45 pp, UL Monroe offense vs South Alabama defense -1.17 pp
- OL protection vs pass rush (sack rate): edge **South Alabama** (0.48 SD); vs-average expectation — South Alabama offense vs UL Monroe defense -2.11 pp sack rate, UL Monroe offense vs South Alabama defense -0.36 pp sack rate
- Finishing drives (pts per scoring opportunity): edge **South Alabama** (0.14 SD); vs-average expectation — South Alabama offense vs UL Monroe defense +0.32 pts, UL Monroe offense vs South Alabama defense +0.16 pts
- Run game vs run defense (opp-adj rush EPA/play): edge **South Alabama** (0.07 SD); vs-average expectation — South Alabama offense vs UL Monroe defense +0.18 EPA/play, UL Monroe offense vs South Alabama defense +0.16 EPA/play
- Explosive-play rate matchup: edge **UL Monroe** (0.01 SD); vs-average expectation — South Alabama offense vs UL Monroe defense +1.00 pp, UL Monroe offense vs South Alabama defense +1.05 pp
- QB: South Alabama Jared Hollins (7.3 YPA, 3 TD/2 INT) vs UL Monroe Aidan Armenta (7.7 YPA, 2 TD/0 INT, NEW STARTER last game)
- Home field: South Alabama (Elo home bonus 45 rating pts; scoring-model home coefficient 6.4 pts on 2026 data)
- Recent form (last-3 margin vs expectation): South Alabama +0.0, UL Monroe -12.3
- Strength of schedule (avg opp net rating): South Alabama +14.5, UL Monroe +13.8
- Rest: UL Monroe 6 days, South Alabama 7 days; travel distance: Data unavailable
- Weather: Forecast: Thunderstorms, 77.0°F, gusts 10.0 mph, precip 79.0% — flagged (wind/storms)
- Injuries/availability: Data unavailable (no public CFB injury feed); QB continuity — UL Monroe starter share 51%, South Alabama 52%

</details>

### Texas Tech @ Colorado
*2026-10-03 07:30 PM ET · Folsom Field (Boulder, CO) · Big 12 · records: Texas Tech 4-0, Colorado 2-2*

**Prediction: Texas Tech**  
Win Probability: Texas Tech 83% / Colorado 17%  
Projected Score: Texas Tech 30–15 Colorado  
Projected Margin: 15.1 (Texas Tech)  
Upset Probability: 17%  
Confidence: 6.2/10

Key Factors:

1. Turnover tendencies (giveaways vs takeaways): edge **Texas Tech** (1.58 SD); vs-average expectation — Colorado offense vs Texas Tech defense +0.96 pp turnover rate, Texas Tech offense vs Colorado defense -0.55 pp turnover rate
2. Pass game vs pass defense (opp-adj pass EPA/play): edge **Texas Tech** (1.22 SD); vs-average expectation — Colorado offense vs Texas Tech defense -0.17 EPA/play, Texas Tech offense vs Colorado defense +0.21 EPA/play
3. Finishing drives (pts per scoring opportunity): edge **Texas Tech** (1.19 SD); vs-average expectation — Colorado offense vs Texas Tech defense -1.36 pts, Texas Tech offense vs Colorado defense +0.01 pts

Main Risk: QB change last game (Colorado) — availability unconfirmed

<details><summary>Model breakdown & matchup detail</summary>

| Model | Elo | Adj. scoring model | Logistic reg. | Gradient boosting | Point-diff regression | **Ensemble** |
|---|---|---|---|---|---|---|
| P(Colorado win) | 0.102 | 0.256 | 0.132 | 0.138 | 0.151 | **0.171** |
| Home margin | -20.8 | -10.9 | — | -15.6 (GBM reg.) | -16.7 | **-15.1** |

- Elo: Texas Tech 1801, Colorado 1395; adj. net points rating: Texas Tech +30.8, Colorado +16.4
- Field position / special teams (start field pos.): edge **Texas Tech** (1.09 SD); vs-average expectation — Colorado offense vs Texas Tech defense -3.09 yds, Texas Tech offense vs Colorado defense +1.62 yds
- Overall offense vs defense (opp-adj EPA/play): edge **Texas Tech** (0.96 SD); vs-average expectation — Colorado offense vs Texas Tech defense -0.15 EPA/play, Texas Tech offense vs Colorado defense +0.08 EPA/play
- Success-rate matchup: edge **Texas Tech** (0.89 SD); vs-average expectation — Colorado offense vs Texas Tech defense -6.75 pp, Texas Tech offense vs Colorado defense +2.86 pp
- OL protection vs pass rush (sack rate): edge **Texas Tech** (0.80 SD); vs-average expectation — Colorado offense vs Texas Tech defense +2.24 pp sack rate, Texas Tech offense vs Colorado defense -0.67 pp sack rate
- Explosive-play rate matchup: edge **Texas Tech** (0.61 SD); vs-average expectation — Colorado offense vs Texas Tech defense -1.93 pp, Texas Tech offense vs Colorado defense +0.92 pp
- Run game vs run defense (opp-adj rush EPA/play): edge **Texas Tech** (0.45 SD); vs-average expectation — Colorado offense vs Texas Tech defense -0.09 EPA/play, Texas Tech offense vs Colorado defense +0.03 EPA/play
- QB: Colorado Isaac Wilson (6.2 YPA, 1 TD/0 INT, NEW STARTER last game) vs Texas Tech Will Hammond (8.3 YPA, 4 TD/5 INT)
- Home field: Colorado (Elo home bonus 45 rating pts; scoring-model home coefficient 6.4 pts on 2026 data)
- Recent form (last-3 margin vs expectation): Colorado -7.1, Texas Tech +1.8
- Strength of schedule (avg opp net rating): Colorado +15.6, Texas Tech +14.8
- Rest: Texas Tech 7 days, Colorado 7 days; travel distance: Data unavailable
- Weather: Forecast: Mostly sunny, 81.0°F, gusts 15.0 mph, precip 0.0%
- Injuries/availability: Data unavailable (no public CFB injury feed); QB continuity — Texas Tech starter share 100%, Colorado 24%

</details>

### Miami @ Clemson
*2026-10-03 07:30 PM ET · Memorial Stadium (Clemson, SC) · Atlantic Coast · records: Miami 4-0, Clemson 3-1*

**Prediction: Miami**  
Win Probability: Miami 88% / Clemson 12%  
Projected Score: Miami 35–16 Clemson  
Projected Margin: 19.2 (Miami)  
Upset Probability: 12%  
Confidence: 7.5/10

Key Factors:

1. Success-rate matchup: edge **Miami** (1.57 SD); vs-average expectation — Clemson offense vs Miami defense -7.49 pp, Miami offense vs Clemson defense +9.40 pp
2. Turnover tendencies (giveaways vs takeaways): edge **Miami** (1.54 SD); vs-average expectation — Clemson offense vs Miami defense +0.77 pp turnover rate, Miami offense vs Clemson defense -0.69 pp turnover rate
3. Run game vs run defense (opp-adj rush EPA/play): edge **Miami** (1.44 SD); vs-average expectation — Clemson offense vs Miami defense -0.16 EPA/play, Miami offense vs Clemson defense +0.20 EPA/play

Main Risk: weather (wind/storms) adds variance

<details><summary>Model breakdown & matchup detail</summary>

| Model | Elo | Adj. scoring model | Logistic reg. | Gradient boosting | Point-diff regression | **Ensemble** |
|---|---|---|---|---|---|---|
| P(Clemson win) | 0.251 | 0.151 | 0.104 | 0.106 | 0.109 | **0.118** |
| Home margin | -10.1 | -17.2 | — | -20.7 (GBM reg.) | -20.0 | **-19.2** |

- Elo: Miami 1881, Clemson 1645; adj. net points rating: Miami +41.5, Clemson +21.6
- Overall offense vs defense (opp-adj EPA/play): edge **Miami** (1.29 SD); vs-average expectation — Clemson offense vs Miami defense -0.19 EPA/play, Miami offense vs Clemson defense +0.12 EPA/play
- Explosive-play rate matchup: edge **Miami** (1.09 SD); vs-average expectation — Clemson offense vs Miami defense -2.97 pp, Miami offense vs Clemson defense +2.18 pp
- OL protection vs pass rush (sack rate): edge **Miami** (0.98 SD); vs-average expectation — Clemson offense vs Miami defense +1.15 pp sack rate, Miami offense vs Clemson defense -2.41 pp sack rate
- Pass game vs pass defense (opp-adj pass EPA/play): edge **Miami** (0.77 SD); vs-average expectation — Clemson offense vs Miami defense -0.18 EPA/play, Miami offense vs Clemson defense +0.07 EPA/play
- Field position / special teams (start field pos.): edge **Miami** (0.31 SD); vs-average expectation — Clemson offense vs Miami defense -4.11 yds, Miami offense vs Clemson defense -2.78 yds
- Finishing drives (pts per scoring opportunity): edge **Miami** (0.18 SD); vs-average expectation — Clemson offense vs Miami defense -0.32 pts, Miami offense vs Clemson defense -0.11 pts
- QB: Clemson Tait Reynolds (6.4 YPA, 1 TD/2 INT) vs Miami Darian Mensah (11.4 YPA, 14 TD/0 INT)
- Home field: Clemson (Elo home bonus 45 rating pts; scoring-model home coefficient 6.4 pts on 2026 data)
- Recent form (last-3 margin vs expectation): Clemson +7.0, Miami +7.1
- Strength of schedule (avg opp net rating): Clemson +21.1, Miami +5.8
- Rest: Miami 7 days, Clemson 7 days; travel distance: Data unavailable
- Weather: Forecast: Mostly cloudy w/ t-storms, 75.0°F, gusts 8.0 mph, precip 75.0% — flagged (wind/storms)
- Injuries/availability: Data unavailable (no public CFB injury feed); QB continuity — Miami starter share 100%, Clemson 83%

</details>

### Washington @ USC
*2026-10-03 07:30 PM ET · Los Angeles Memorial Coliseum · Big Ten · records: Washington 3-1, USC 4-1*

**Prediction: USC**  
Win Probability: Washington 24% / USC 76%  
Projected Score: Washington 24–36 USC  
Projected Margin: 11.4 (USC)  
Upset Probability: 24%  
Confidence: 5.5/10

Key Factors:

1. Turnover tendencies (giveaways vs takeaways): edge **Washington** (0.81 SD); vs-average expectation — USC offense vs Washington defense +0.32 pp turnover rate, Washington offense vs USC defense -0.45 pp turnover rate
2. Field position / special teams (start field pos.): edge **USC** (0.70 SD); vs-average expectation — USC offense vs Washington defense +0.28 yds, Washington offense vs USC defense -2.75 yds
3. Success-rate matchup: edge **USC** (0.54 SD); vs-average expectation — USC offense vs Washington defense +6.79 pp, Washington offense vs USC defense +0.92 pp

Main Risk: Washington's best counter is Turnover tendencies (0.81 SD edge)

<details><summary>Model breakdown & matchup detail</summary>

| Model | Elo | Adj. scoring model | Logistic reg. | Gradient boosting | Point-diff regression | **Ensemble** |
|---|---|---|---|---|---|---|
| P(USC win) | 0.620 | 0.802 | 0.739 | 0.753 | 0.754 | **0.761** |
| Home margin | +5.6 | +14.1 | — | +8.9 (GBM reg.) | +11.1 | **+11.4** |

- Elo: Washington 1674, USC 1689; adj. net points rating: Washington +21.7, USC +29.2
- OL protection vs pass rush (sack rate): edge **USC** (0.53 SD); vs-average expectation — USC offense vs Washington defense -1.84 pp sack rate, Washington offense vs USC defense +0.09 pp sack rate
- Explosive-play rate matchup: edge **USC** (0.49 SD); vs-average expectation — USC offense vs Washington defense +2.25 pp, Washington offense vs USC defense -0.06 pp
- Pass game vs pass defense (opp-adj pass EPA/play): edge **USC** (0.46 SD); vs-average expectation — USC offense vs Washington defense +0.22 EPA/play, Washington offense vs USC defense +0.08 EPA/play
- Run game vs run defense (opp-adj rush EPA/play): edge **Washington** (0.29 SD); vs-average expectation — USC offense vs Washington defense -0.13 EPA/play, Washington offense vs USC defense -0.06 EPA/play
- Finishing drives (pts per scoring opportunity): edge **USC** (0.18 SD); vs-average expectation — USC offense vs Washington defense +0.74 pts, Washington offense vs USC defense +0.53 pts
- Overall offense vs defense (opp-adj EPA/play): edge **USC** (0.07 SD); vs-average expectation — USC offense vs Washington defense +0.03 EPA/play, Washington offense vs USC defense +0.01 EPA/play
- QB: USC Jayden Maiava (9.5 YPA, 14 TD/2 INT) vs Washington Demond Williams Jr. (8.2 YPA, 7 TD/3 INT)
- Home field: USC (Elo home bonus 45 rating pts; scoring-model home coefficient 6.4 pts on 2026 data)
- Recent form (last-3 margin vs expectation): USC -8.4, Washington -13.6
- Strength of schedule (avg opp net rating): USC +20.1, Washington +11.8
- Rest: Washington 6 days, USC 7 days; travel distance: Data unavailable
- Weather: Forecast: Sunny, 98.0°F, gusts 12.0 mph, precip 0.0%
- Injuries/availability: Data unavailable (no public CFB injury feed); QB continuity — Washington starter share 100%, USC 100%

</details>

### Utah State @ Boise State
*2026-10-03 07:30 PM ET · Albertsons Stadium (Boise, ID) · Pac-12 · records: Utah State 1-3, Boise State 3-1*

**Prediction: Boise State**  
Win Probability: Utah State 10% / Boise State 90%  
Projected Score: Utah State 15–35 Boise State  
Projected Margin: 20.6 (Boise State)  
Upset Probability: 10%  
Confidence: 8.4/10

Key Factors:

1. OL protection vs pass rush (sack rate): edge **Boise State** (1.13 SD); vs-average expectation — Boise State offense vs Utah State defense -2.46 pp sack rate, Utah State offense vs Boise State defense +1.68 pp sack rate
2. Success-rate matchup: edge **Boise State** (1.12 SD); vs-average expectation — Boise State offense vs Utah State defense +4.22 pp, Utah State offense vs Boise State defense -7.87 pp
3. Run game vs run defense (opp-adj rush EPA/play): edge **Boise State** (0.87 SD); vs-average expectation — Boise State offense vs Utah State defense +0.06 EPA/play, Utah State offense vs Boise State defense -0.16 EPA/play

Main Risk: Utah State's best counter is Field position / special teams (0.66 SD edge)

<details><summary>Model breakdown & matchup detail</summary>

| Model | Elo | Adj. scoring model | Logistic reg. | Gradient boosting | Point-diff regression | **Ensemble** |
|---|---|---|---|---|---|---|
| P(Boise State win) | 0.899 | 0.908 | 0.898 | 0.894 | 0.900 | **0.901** |
| Home margin | +22.3 | +22.1 | — | +17.8 (GBM reg.) | +20.8 | **+20.6** |

- Elo: Utah State 1425, Boise State 1706; adj. net points rating: Utah State +14.0, Boise State +28.6
- Turnover tendencies (giveaways vs takeaways): edge **Boise State** (0.85 SD); vs-average expectation — Boise State offense vs Utah State defense -0.43 pp turnover rate, Utah State offense vs Boise State defense +0.37 pp turnover rate
- Finishing drives (pts per scoring opportunity): edge **Boise State** (0.83 SD); vs-average expectation — Boise State offense vs Utah State defense -0.15 pts, Utah State offense vs Boise State defense -1.12 pts
- Field position / special teams (start field pos.): edge **Utah State** (0.66 SD); vs-average expectation — Boise State offense vs Utah State defense -2.64 yds, Utah State offense vs Boise State defense +0.20 yds
- Explosive-play rate matchup: edge **Utah State** (0.46 SD); vs-average expectation — Boise State offense vs Utah State defense -1.69 pp, Utah State offense vs Boise State defense +0.49 pp
- Overall offense vs defense (opp-adj EPA/play): edge **Boise State** (0.46 SD); vs-average expectation — Boise State offense vs Utah State defense -0.01 EPA/play, Utah State offense vs Boise State defense -0.13 EPA/play
- Pass game vs pass defense (opp-adj pass EPA/play): edge **Boise State** (0.17 SD); vs-average expectation — Boise State offense vs Utah State defense -0.09 EPA/play, Utah State offense vs Boise State defense -0.14 EPA/play
- QB: Boise State Maddux Madsen (7.3 YPA, 9 TD/0 INT) vs Utah State Grady Brosterhous (6.7 YPA, 4 TD/5 INT)
- Home field: Boise State (Elo home bonus 45 rating pts; scoring-model home coefficient 6.4 pts on 2026 data)
- Recent form (last-3 margin vs expectation): Boise State +8.9, Utah State +8.0
- Strength of schedule (avg opp net rating): Boise State +21.2, Utah State +20.1
- Rest: Utah State 7 days, Boise State 7 days; travel distance: Data unavailable
- Weather: Forecast: Sunny, 85.0°F, gusts 6.0 mph, precip 0.0%
- Injuries/availability: Data unavailable (no public CFB injury feed); QB continuity — Utah State starter share 74%, Boise State 100%

</details>

### Temple @ South Florida
*2026-10-03 07:30 PM ET · Raymond James Stadium (Tampa, FL) · American · records: Temple 1-3, South Florida 4-0*

**Prediction: South Florida**  
Win Probability: Temple 17% / South Florida 83%  
Projected Score: Temple 21–36 South Florida  
Projected Margin: 14.9 (South Florida)  
Upset Probability: 17%  
Confidence: 6.8/10

Key Factors:

1. Finishing drives (pts per scoring opportunity): edge **South Florida** (1.31 SD); vs-average expectation — South Florida offense vs Temple defense +1.34 pts, Temple offense vs South Florida defense -0.16 pts
2. Run game vs run defense (opp-adj rush EPA/play): edge **South Florida** (0.98 SD); vs-average expectation — South Florida offense vs Temple defense +0.22 EPA/play, Temple offense vs South Florida defense -0.03 EPA/play
3. Field position / special teams (start field pos.): edge **South Florida** (0.67 SD); vs-average expectation — South Florida offense vs Temple defense -1.04 yds, Temple offense vs South Florida defense -3.93 yds

Main Risk: Temple's best counter is Pass game vs pass defense (0.37 SD edge); weather (wind/storms) adds variance

<details><summary>Model breakdown & matchup detail</summary>

| Model | Elo | Adj. scoring model | Logistic reg. | Gradient boosting | Point-diff regression | **Ensemble** |
|---|---|---|---|---|---|---|
| P(South Florida win) | 0.893 | 0.826 | 0.840 | 0.905 | 0.806 | **0.827** |
| Home margin | +21.8 | +15.6 | — | +14.6 (GBM reg.) | +14.0 | **+14.9** |

- Elo: Temple 1395, South Florida 1667; adj. net points rating: Temple +15.5, South Florida +24.4
- Success-rate matchup: edge **South Florida** (0.65 SD); vs-average expectation — South Florida offense vs Temple defense +9.22 pp, Temple offense vs South Florida defense +2.24 pp
- Explosive-play rate matchup: edge **South Florida** (0.43 SD); vs-average expectation — South Florida offense vs Temple defense +1.56 pp, Temple offense vs South Florida defense -0.44 pp
- Overall offense vs defense (opp-adj EPA/play): edge **South Florida** (0.41 SD); vs-average expectation — South Florida offense vs Temple defense +0.15 EPA/play, Temple offense vs South Florida defense +0.05 EPA/play
- Pass game vs pass defense (opp-adj pass EPA/play): edge **Temple** (0.37 SD); vs-average expectation — South Florida offense vs Temple defense +0.12 EPA/play, Temple offense vs South Florida defense +0.23 EPA/play
- OL protection vs pass rush (sack rate): edge **South Florida** (0.26 SD); vs-average expectation — South Florida offense vs Temple defense -0.05 pp sack rate, Temple offense vs South Florida defense +0.89 pp sack rate
- Turnover tendencies (giveaways vs takeaways): edge **Temple** (0.15 SD); vs-average expectation — South Florida offense vs Temple defense -0.38 pp turnover rate, Temple offense vs South Florida defense -0.52 pp turnover rate
- QB: South Florida Michael Van Buren Jr. (7.6 YPA, 3 TD/0 INT) vs Temple Ajani Sheppard (9.4 YPA, 5 TD/0 INT)
- Home field: South Florida (Elo home bonus 45 rating pts; scoring-model home coefficient 6.4 pts on 2026 data)
- Recent form (last-3 margin vs expectation): South Florida +0.4, Temple +0.4
- Strength of schedule (avg opp net rating): South Florida +6.9, Temple +18.8
- Rest: Temple 8 days, South Florida 7 days; travel distance: Data unavailable
- Weather: Forecast: Mostly cloudy w/ t-storms, 81.0°F, gusts 8.0 mph, precip 61.0% — flagged (wind/storms)
- Injuries/availability: Data unavailable (no public CFB injury feed); QB continuity — Temple starter share 76%, South Florida 100%

</details>

### Army @ Louisiana Tech
*2026-10-03 07:30 PM ET · Joe Aillet Stadium (Ruston, LA) · American vs nan · records: Army 2-1, Louisiana Tech 1-2*

**Prediction: Louisiana Tech**  
Win Probability: Army 46% / Louisiana Tech 54%  
Projected Score: Army 26–28 Louisiana Tech  
Projected Margin: 1.5 (Louisiana Tech)  
Upset Probability: 46%  
Confidence: 1.0/10  ⚠️ HIGH MODEL DISAGREEMENT

Key Factors:

1. Finishing drives (pts per scoring opportunity): edge **Army** (1.30 SD); vs-average expectation — Louisiana Tech offense vs Army defense -0.84 pts, Army offense vs Louisiana Tech defense +0.65 pts
2. Field position / special teams (start field pos.): edge **Army** (0.66 SD); vs-average expectation — Louisiana Tech offense vs Army defense -1.23 yds, Army offense vs Louisiana Tech defense +1.62 yds
3. Run game vs run defense (opp-adj rush EPA/play): edge **Army** (0.62 SD); vs-average expectation — Louisiana Tech offense vs Army defense +0.02 EPA/play, Army offense vs Louisiana Tech defense +0.18 EPA/play

Main Risk: models disagree on the winner; Army's best counter is Finishing drives (1.30 SD edge); QB change last game (Louisiana Tech) — availability unconfirmed

<details><summary>Model breakdown & matchup detail</summary>

| Model | Elo | Adj. scoring model | Logistic reg. | Gradient boosting | Point-diff regression | **Ensemble** |
|---|---|---|---|---|---|---|
| P(Louisiana Tech win) | 0.365 | 0.497 | 0.555 | 0.417 | 0.566 | **0.535** |
| Home margin | -4.8 | -0.1 | — | +0.8 (GBM reg.) | +2.7 | **+1.5** |

- Elo: Army 1587, Louisiana Tech 1437; adj. net points rating: Army +23.7, Louisiana Tech +18.7
- Explosive-play rate matchup: edge **Louisiana Tech** (0.43 SD); vs-average expectation — Louisiana Tech offense vs Army defense +1.97 pp, Army offense vs Louisiana Tech defense -0.06 pp
- Pass game vs pass defense (opp-adj pass EPA/play): edge **Army** (0.38 SD); vs-average expectation — Louisiana Tech offense vs Army defense +0.10 EPA/play, Army offense vs Louisiana Tech defense +0.22 EPA/play
- Turnover tendencies (giveaways vs takeaways): edge **Louisiana Tech** (0.35 SD); vs-average expectation — Louisiana Tech offense vs Army defense -0.28 pp turnover rate, Army offense vs Louisiana Tech defense +0.06 pp turnover rate
- OL protection vs pass rush (sack rate): edge **Louisiana Tech** (0.26 SD); vs-average expectation — Louisiana Tech offense vs Army defense -0.87 pp sack rate, Army offense vs Louisiana Tech defense +0.08 pp sack rate
- Overall offense vs defense (opp-adj EPA/play): edge **Army** (0.15 SD); vs-average expectation — Louisiana Tech offense vs Army defense +0.06 EPA/play, Army offense vs Louisiana Tech defense +0.10 EPA/play
- Success-rate matchup: edge **Louisiana Tech** (0.00 SD); vs-average expectation — Louisiana Tech offense vs Army defense -1.04 pp, Army offense vs Louisiana Tech defense -1.04 pp
- QB: Louisiana Tech Trey Kukuk (5.1 YPA, 1 TD/1 INT, NEW STARTER last game) vs Army Cale Hellums (7.4 YPA, 2 TD/0 INT)
- Home field: Louisiana Tech (Elo home bonus 45 rating pts; scoring-model home coefficient 6.4 pts on 2026 data)
- Recent form (last-3 margin vs expectation): Louisiana Tech +5.6, Army +6.0
- Strength of schedule (avg opp net rating): Louisiana Tech +9.3, Army +11.1
- Rest: Army 8 days, Louisiana Tech 14 days; travel distance: Data unavailable
- Weather: Forecast: Intermittent clouds, 80.0°F, gusts 12.0 mph, precip 34.0%
- Injuries/availability: Data unavailable (no public CFB injury feed); QB continuity — Army starter share 100%, Louisiana Tech 56%

</details>

### McNeese @ LSU
*2026-10-03 07:45 PM ET · Tiger Stadium (LA) (Baton Rouge, LA) · Southland vs Southeastern · records: McNeese 2-3, LSU 3-1*

**Prediction: LSU**  
Win Probability: McNeese 0% / LSU 100%  
Projected Score: McNeese 5–54 LSU  
Projected Margin: 49.8 (LSU)  
Upset Probability: 0%  
Confidence: 9.4/10

Key Factors:

1. Overall offense vs defense (opp-adj EPA/play): edge **LSU** (2.64 SD); vs-average expectation — LSU offense vs McNeese defense +0.31 EPA/play, McNeese offense vs LSU defense -0.33 EPA/play
2. Success-rate matchup: edge **LSU** (2.59 SD); vs-average expectation — LSU offense vs McNeese defense +10.78 pp, McNeese offense vs LSU defense -17.18 pp
3. Field position / special teams (start field pos.): edge **LSU** (2.44 SD); vs-average expectation — LSU offense vs McNeese defense +2.02 yds, McNeese offense vs LSU defense -8.55 yds

Main Risk: QB change last game (McNeese) — availability unconfirmed; weather (wind/storms) adds variance

<details><summary>Model breakdown & matchup detail</summary>

| Model | Elo | Adj. scoring model | Logistic reg. | Gradient boosting | Point-diff regression | **Ensemble** |
|---|---|---|---|---|---|---|
| P(LSU win) | 0.984 | 0.999 | 0.998 | 0.995 | 0.999 | **0.999** |
| Home margin | +41.3 | +52.8 | — | +49.3 (GBM reg.) | +51.8 | **+49.8** |

- Elo: McNeese 1138, LSU 1722; adj. net points rating: McNeese -5.4, LSU +36.0
- Pass game vs pass defense (opp-adj pass EPA/play): edge **LSU** (2.42 SD); vs-average expectation — LSU offense vs McNeese defense +0.29 EPA/play, McNeese offense vs LSU defense -0.47 EPA/play
- OL protection vs pass rush (sack rate): edge **LSU** (2.34 SD); vs-average expectation — LSU offense vs McNeese defense -2.08 pp sack rate, McNeese offense vs LSU defense +6.45 pp sack rate
- Run game vs run defense (opp-adj rush EPA/play): edge **LSU** (2.23 SD); vs-average expectation — LSU offense vs McNeese defense +0.37 EPA/play, McNeese offense vs LSU defense -0.20 EPA/play
- Turnover tendencies (giveaways vs takeaways): edge **LSU** (2.21 SD); vs-average expectation — LSU offense vs McNeese defense -0.60 pp turnover rate, McNeese offense vs LSU defense +1.50 pp turnover rate
- Finishing drives (pts per scoring opportunity): edge **LSU** (1.94 SD); vs-average expectation — LSU offense vs McNeese defense +1.50 pts, McNeese offense vs LSU defense -0.74 pts
- Explosive-play rate matchup: edge **LSU** (1.82 SD); vs-average expectation — LSU offense vs McNeese defense +5.53 pp, McNeese offense vs LSU defense -3.00 pp
- QB: LSU Sam Leavitt (9.0 YPA, 5 TD/6 INT) vs McNeese Jake Strong (7.5 YPA, 1 TD/0 INT, NEW STARTER last game)
- Home field: LSU (Elo home bonus 45 rating pts; scoring-model home coefficient 6.4 pts on 2026 data)
- Recent form (last-3 margin vs expectation): LSU +10.0, McNeese -3.8
- Strength of schedule (avg opp net rating): LSU +24.6, McNeese -5.3
- Rest: McNeese 7 days, LSU 7 days; travel distance: Data unavailable
- Weather: Forecast: Thunderstorms, 81.0°F, gusts 12.0 mph, precip 75.0% — flagged (wind/storms)
- Injuries/availability: Data unavailable (no public CFB injury feed); QB continuity — McNeese starter share 19%, LSU 100%

</details>

### Indiana @ Rutgers
*2026-10-03 08:00 PM ET · SHI Stadium (Piscataway, NJ) · Big Ten · records: Indiana 4-0, Rutgers 1-3*

**Prediction: Indiana**  
Win Probability: Indiana 93% / Rutgers 7%  
Projected Score: Indiana 43–19 Rutgers  
Projected Margin: 24.0 (Indiana)  
Upset Probability: 7%  
Confidence: 8.7/10

Key Factors:

1. Finishing drives (pts per scoring opportunity): edge **Indiana** (2.00 SD); vs-average expectation — Rutgers offense vs Indiana defense -1.27 pts, Indiana offense vs Rutgers defense +1.04 pts
2. Pass game vs pass defense (opp-adj pass EPA/play): edge **Indiana** (1.78 SD); vs-average expectation — Rutgers offense vs Indiana defense -0.03 EPA/play, Indiana offense vs Rutgers defense +0.52 EPA/play
3. Explosive-play rate matchup: edge **Indiana** (1.73 SD); vs-average expectation — Rutgers offense vs Indiana defense -1.39 pp, Indiana offense vs Rutgers defense +6.74 pp

Main Risk: single-game variance (σ ≈ 16 pts on the margin)

<details><summary>Model breakdown & matchup detail</summary>

| Model | Elo | Adj. scoring model | Logistic reg. | Gradient boosting | Point-diff regression | **Ensemble** |
|---|---|---|---|---|---|---|
| P(Rutgers win) | 0.033 | 0.093 | 0.053 | 0.103 | 0.054 | **0.067** |
| Home margin | -32.6 | -22.0 | — | -20.9 (GBM reg.) | -26.1 | **-24.0** |

- Elo: Indiana 1968, Rutgers 1374; adj. net points rating: Indiana +39.3, Rutgers +15.2
- Turnover tendencies (giveaways vs takeaways): edge **Indiana** (1.59 SD); vs-average expectation — Rutgers offense vs Indiana defense +0.90 pp turnover rate, Indiana offense vs Rutgers defense -0.62 pp turnover rate
- Overall offense vs defense (opp-adj EPA/play): edge **Indiana** (1.37 SD); vs-average expectation — Rutgers offense vs Indiana defense -0.04 EPA/play, Indiana offense vs Rutgers defense +0.30 EPA/play
- Success-rate matchup: edge **Indiana** (1.15 SD); vs-average expectation — Rutgers offense vs Indiana defense -1.86 pp, Indiana offense vs Rutgers defense +10.59 pp
- Field position / special teams (start field pos.): edge **Indiana** (1.15 SD); vs-average expectation — Rutgers offense vs Indiana defense -2.66 yds, Indiana offense vs Rutgers defense +2.33 yds
- Run game vs run defense (opp-adj rush EPA/play): edge **Indiana** (0.95 SD); vs-average expectation — Rutgers offense vs Indiana defense -0.06 EPA/play, Indiana offense vs Rutgers defense +0.18 EPA/play
- OL protection vs pass rush (sack rate): edge **Indiana** (0.51 SD); vs-average expectation — Rutgers offense vs Indiana defense +0.85 pp sack rate, Indiana offense vs Rutgers defense -1.02 pp sack rate
- QB: Rutgers Dylan Lonergan (8.6 YPA, 7 TD/5 INT) vs Indiana Josh Hoover (10.6 YPA, 11 TD/0 INT)
- Home field: Rutgers (Elo home bonus 45 rating pts; scoring-model home coefficient 6.4 pts on 2026 data)
- Recent form (last-3 margin vs expectation): Rutgers +9.1, Indiana -9.8
- Strength of schedule (avg opp net rating): Rutgers +9.8, Indiana +12.5
- Rest: Indiana 8 days, Rutgers 8 days; travel distance: Data unavailable
- Weather: Forecast: Mostly clear, 62.0°F, gusts 9.0 mph, precip 0.0%
- Injuries/availability: Data unavailable (no public CFB injury feed); QB continuity — Indiana starter share 100%, Rutgers 75%

</details>

### West Florida @ Abilene Christian
*2026-10-03 08:00 PM ET · Wildcat Stadium (TX) (Abilene, TX) · United Athletic · records: West Florida 3-1, Abilene Christian 1-4*

**Prediction: Abilene Christian**  
Win Probability: West Florida 37% / Abilene Christian 63%  
Projected Score: West Florida 23–28 Abilene Christian  
Projected Margin: 5.1 (Abilene Christian)  
Upset Probability: 37%  
Confidence: 2.7/10

Key Factors:

1. Turnover tendencies (giveaways vs takeaways): edge **West Florida** (1.60 SD); vs-average expectation — Abilene Christian offense vs West Florida defense +1.49 pp turnover rate, West Florida offense vs Abilene Christian defense -0.03 pp turnover rate
2. OL protection vs pass rush (sack rate): edge **West Florida** (0.93 SD); vs-average expectation — Abilene Christian offense vs West Florida defense +1.55 pp sack rate, West Florida offense vs Abilene Christian defense -1.85 pp sack rate
3. Explosive-play rate matchup: edge **Abilene Christian** (0.84 SD); vs-average expectation — Abilene Christian offense vs West Florida defense +1.50 pp, West Florida offense vs Abilene Christian defense -2.47 pp

Main Risk: West Florida's best counter is Turnover tendencies (1.60 SD edge); QB change last game (West Florida) — availability unconfirmed

<details><summary>Model breakdown & matchup detail</summary>

| Model | Elo | Adj. scoring model | Logistic reg. | Gradient boosting | Point-diff regression | **Ensemble** |
|---|---|---|---|---|---|---|
| P(Abilene Christian win) | 0.724 | 0.603 | 0.639 | 0.630 | 0.643 | **0.631** |
| Home margin | +10.3 | +4.4 | — | +3.1 (GBM reg.) | +5.9 | **+5.1** |

- Elo: West Florida 1125, Abilene Christian 1214; adj. net points rating: West Florida -1.0, Abilene Christian -2.0
- Field position / special teams (start field pos.): edge **West Florida** (0.81 SD); vs-average expectation — Abilene Christian offense vs West Florida defense -2.36 yds, West Florida offense vs Abilene Christian defense +1.14 yds
- Pass game vs pass defense (opp-adj pass EPA/play): edge **West Florida** (0.80 SD); vs-average expectation — Abilene Christian offense vs West Florida defense -0.20 EPA/play, West Florida offense vs Abilene Christian defense +0.05 EPA/play
- Run game vs run defense (opp-adj rush EPA/play): edge **Abilene Christian** (0.72 SD); vs-average expectation — Abilene Christian offense vs West Florida defense +0.00 EPA/play, West Florida offense vs Abilene Christian defense -0.18 EPA/play
- Finishing drives (pts per scoring opportunity): edge **West Florida** (0.50 SD); vs-average expectation — Abilene Christian offense vs West Florida defense -0.39 pts, West Florida offense vs Abilene Christian defense +0.19 pts
- Success-rate matchup: edge **Abilene Christian** (0.28 SD); vs-average expectation — Abilene Christian offense vs West Florida defense +1.23 pp, West Florida offense vs Abilene Christian defense -1.76 pp
- Overall offense vs defense (opp-adj EPA/play): edge **West Florida** (0.03 SD); vs-average expectation — Abilene Christian offense vs West Florida defense -0.10 EPA/play, West Florida offense vs Abilene Christian defense -0.09 EPA/play
- QB: Abilene Christian Tucker Parks (7.0 YPA, 2 TD/3 INT) vs West Florida Donovan Leary (4.3 YPA, 1 TD/1 INT, NEW STARTER last game)
- Home field: Abilene Christian (Elo home bonus 45 rating pts; scoring-model home coefficient 6.4 pts on 2026 data)
- Recent form (last-3 margin vs expectation): Abilene Christian -13.9, West Florida +5.2
- Strength of schedule (avg opp net rating): Abilene Christian +8.3, West Florida +0.7
- Rest: West Florida 7 days, Abilene Christian 7 days; travel distance: Data unavailable
- Weather: Forecast: Intermittent clouds, 73.0°F, gusts 12.0 mph, precip 44.0%
- Injuries/availability: Data unavailable (no public CFB injury feed); QB continuity — West Florida starter share 43%, Abilene Christian 34%

</details>

### Weber State @ Cal Poly
*2026-10-03 08:00 PM ET · Mustang Memorial Field at Spanos Stadium (San Luis Obispo, CA) · Big Sky · records: Weber State 2-3, Cal Poly 3-2*

**Prediction: Cal Poly**  
Win Probability: Weber State 37% / Cal Poly 63%  
Projected Score: Weber State 27–32 Cal Poly  
Projected Margin: 5.5 (Cal Poly)  
Upset Probability: 37%  
Confidence: 3.4/10

Key Factors:

1. Field position / special teams (start field pos.): edge **Cal Poly** (1.00 SD); vs-average expectation — Cal Poly offense vs Weber State defense +3.19 yds, Weber State offense vs Cal Poly defense -1.12 yds
2. Explosive-play rate matchup: edge **Cal Poly** (0.72 SD); vs-average expectation — Cal Poly offense vs Weber State defense +3.28 pp, Weber State offense vs Cal Poly defense -0.09 pp
3. Finishing drives (pts per scoring opportunity): edge **Weber State** (0.46 SD); vs-average expectation — Cal Poly offense vs Weber State defense -0.15 pts, Weber State offense vs Cal Poly defense +0.38 pts

Main Risk: Weber State's best counter is Finishing drives (0.46 SD edge)

<details><summary>Model breakdown & matchup detail</summary>

| Model | Elo | Adj. scoring model | Logistic reg. | Gradient boosting | Point-diff regression | **Ensemble** |
|---|---|---|---|---|---|---|
| P(Cal Poly win) | 0.657 | 0.674 | 0.624 | 0.632 | 0.608 | **0.631** |
| Home margin | +7.2 | +7.5 | — | +5.9 (GBM reg.) | +4.5 | **+5.5** |

- Elo: Weber State 1130, Cal Poly 1169; adj. net points rating: Weber State -9.0, Cal Poly -7.3
- Turnover tendencies (giveaways vs takeaways): edge **Cal Poly** (0.38 SD); vs-average expectation — Cal Poly offense vs Weber State defense -0.28 pp turnover rate, Weber State offense vs Cal Poly defense +0.09 pp turnover rate
- Pass game vs pass defense (opp-adj pass EPA/play): edge **Cal Poly** (0.37 SD); vs-average expectation — Cal Poly offense vs Weber State defense +0.09 EPA/play, Weber State offense vs Cal Poly defense -0.02 EPA/play
- Overall offense vs defense (opp-adj EPA/play): edge **Cal Poly** (0.30 SD); vs-average expectation — Cal Poly offense vs Weber State defense +0.04 EPA/play, Weber State offense vs Cal Poly defense -0.04 EPA/play
- Run game vs run defense (opp-adj rush EPA/play): edge **Weber State** (0.15 SD); vs-average expectation — Cal Poly offense vs Weber State defense -0.06 EPA/play, Weber State offense vs Cal Poly defense -0.02 EPA/play
- Success-rate matchup: edge **Weber State** (0.10 SD); vs-average expectation — Cal Poly offense vs Weber State defense +1.47 pp, Weber State offense vs Cal Poly defense +2.56 pp
- OL protection vs pass rush (sack rate): edge **Weber State** (0.08 SD); vs-average expectation — Cal Poly offense vs Weber State defense -1.59 pp sack rate, Weber State offense vs Cal Poly defense -1.87 pp sack rate
- QB: Cal Poly Anthony Grigsby Jr. (7.6 YPA, 13 TD/4 INT) vs Weber State Nate Dahle (7.2 YPA, 12 TD/6 INT)
- Home field: Cal Poly (Elo home bonus 45 rating pts; scoring-model home coefficient 6.4 pts on 2026 data)
- Recent form (last-3 margin vs expectation): Cal Poly +6.0, Weber State -8.1
- Strength of schedule (avg opp net rating): Cal Poly -8.0, Weber State -9.0
- Rest: Weber State 7 days, Cal Poly 7 days; travel distance: Data unavailable
- Weather: Forecast: Sunny, 85.0°F, gusts 12.0 mph, precip 0.0%
- Injuries/availability: Data unavailable (no public CFB injury feed); QB continuity — Weber State starter share 88%, Cal Poly 100%

</details>

### Northern Arizona @ Idaho State
*2026-10-03 08:00 PM ET · ICCU Dome (Pocatello, ID) · Big Sky · records: Northern Arizona 3-2, Idaho State 4-0*

**Prediction: Idaho State**  
Win Probability: Northern Arizona 18% / Idaho State 82%  
Projected Score: Northern Arizona 19–34 Idaho State  
Projected Margin: 14.4 (Idaho State)  
Upset Probability: 18%  
Confidence: 6.8/10

Key Factors:

1. OL protection vs pass rush (sack rate): edge **Idaho State** (1.33 SD); vs-average expectation — Idaho State offense vs Northern Arizona defense -3.15 pp sack rate, Northern Arizona offense vs Idaho State defense +1.72 pp sack rate
2. Explosive-play rate matchup: edge **Idaho State** (0.70 SD); vs-average expectation — Idaho State offense vs Northern Arizona defense +2.59 pp, Northern Arizona offense vs Idaho State defense -0.72 pp
3. Success-rate matchup: edge **Idaho State** (0.61 SD); vs-average expectation — Idaho State offense vs Northern Arizona defense +6.81 pp, Northern Arizona offense vs Idaho State defense +0.20 pp

Main Risk: Northern Arizona's best counter is Run game vs run defense (0.43 SD edge)

<details><summary>Model breakdown & matchup detail</summary>

| Model | Elo | Adj. scoring model | Logistic reg. | Gradient boosting | Point-diff regression | **Ensemble** |
|---|---|---|---|---|---|---|
| P(Idaho State win) | 0.762 | 0.815 | 0.819 | 0.817 | 0.815 | **0.816** |
| Home margin | +12.2 | +14.9 | — | +13.1 (GBM reg.) | +14.6 | **+14.4** |

- Elo: Northern Arizona 1294, Idaho State 1414; adj. net points rating: Northern Arizona -1.0, Idaho State +7.3
- Pass game vs pass defense (opp-adj pass EPA/play): edge **Idaho State** (0.56 SD); vs-average expectation — Idaho State offense vs Northern Arizona defense +0.03 EPA/play, Northern Arizona offense vs Idaho State defense -0.14 EPA/play
- Run game vs run defense (opp-adj rush EPA/play): edge **Northern Arizona** (0.43 SD); vs-average expectation — Idaho State offense vs Northern Arizona defense -0.01 EPA/play, Northern Arizona offense vs Idaho State defense +0.10 EPA/play
- Field position / special teams (start field pos.): edge **Idaho State** (0.39 SD); vs-average expectation — Idaho State offense vs Northern Arizona defense -0.44 yds, Northern Arizona offense vs Idaho State defense -2.13 yds
- Finishing drives (pts per scoring opportunity): edge **Idaho State** (0.33 SD); vs-average expectation — Idaho State offense vs Northern Arizona defense -0.33 pts, Northern Arizona offense vs Idaho State defense -0.71 pts
- Overall offense vs defense (opp-adj EPA/play): edge **Idaho State** (0.27 SD); vs-average expectation — Idaho State offense vs Northern Arizona defense +0.03 EPA/play, Northern Arizona offense vs Idaho State defense -0.03 EPA/play
- Turnover tendencies (giveaways vs takeaways): edge **Idaho State** (0.26 SD); vs-average expectation — Idaho State offense vs Northern Arizona defense +0.38 pp turnover rate, Northern Arizona offense vs Idaho State defense +0.63 pp turnover rate
- QB: Idaho State Jordan Cooke (8.9 YPA, 10 TD/2 INT) vs Northern Arizona Ty Pennington (7.3 YPA, 7 TD/3 INT)
- Home field: Idaho State (Elo home bonus 45 rating pts; scoring-model home coefficient 6.4 pts on 2026 data)
- Recent form (last-3 margin vs expectation): Idaho State +14.8, Northern Arizona +6.1
- Strength of schedule (avg opp net rating): Idaho State -8.3, Northern Arizona +1.5
- Rest: Northern Arizona 6 days, Idaho State 7 days; travel distance: Data unavailable
- Weather: Indoor venue
- Injuries/availability: Data unavailable (no public CFB injury feed); QB continuity — Northern Arizona starter share 100%, Idaho State 100%

</details>

### Southern Utah @ Utah Tech
*2026-10-03 08:00 PM ET · Greater Zion Stadium (Saint George, UT) · Big Sky · records: Southern Utah 1-4, Utah Tech 0-5*

**Prediction: Southern Utah**  
Win Probability: Southern Utah 79% / Utah Tech 21%  
Projected Score: Southern Utah 34–21 Utah Tech  
Projected Margin: 12.5 (Southern Utah)  
Upset Probability: 21%  
Confidence: 6.3/10

Key Factors:

1. Finishing drives (pts per scoring opportunity): edge **Southern Utah** (1.40 SD); vs-average expectation — Utah Tech offense vs Southern Utah defense -0.93 pts, Southern Utah offense vs Utah Tech defense +0.69 pts
2. Explosive-play rate matchup: edge **Southern Utah** (1.34 SD); vs-average expectation — Utah Tech offense vs Southern Utah defense -1.97 pp, Southern Utah offense vs Utah Tech defense +4.35 pp
3. Field position / special teams (start field pos.): edge **Southern Utah** (1.28 SD); vs-average expectation — Utah Tech offense vs Southern Utah defense -2.77 yds, Southern Utah offense vs Utah Tech defense +2.75 yds

Main Risk: single-game variance (σ ≈ 16 pts on the margin)

<details><summary>Model breakdown & matchup detail</summary>

| Model | Elo | Adj. scoring model | Logistic reg. | Gradient boosting | Point-diff regression | **Ensemble** |
|---|---|---|---|---|---|---|
| P(Utah Tech win) | 0.192 | 0.223 | 0.178 | 0.150 | 0.234 | **0.211** |
| Home margin | -13.5 | -12.7 | — | -11.7 (GBM reg.) | -11.8 | **-12.5** |

- Elo: Southern Utah 1248, Utah Tech 958; adj. net points rating: Southern Utah -3.5, Utah Tech -19.5
- Pass game vs pass defense (opp-adj pass EPA/play): edge **Southern Utah** (1.13 SD); vs-average expectation — Utah Tech offense vs Southern Utah defense -0.30 EPA/play, Southern Utah offense vs Utah Tech defense +0.06 EPA/play
- Overall offense vs defense (opp-adj EPA/play): edge **Southern Utah** (1.05 SD); vs-average expectation — Utah Tech offense vs Southern Utah defense -0.18 EPA/play, Southern Utah offense vs Utah Tech defense +0.07 EPA/play
- Success-rate matchup: edge **Southern Utah** (0.95 SD); vs-average expectation — Utah Tech offense vs Southern Utah defense -5.03 pp, Southern Utah offense vs Utah Tech defense +5.26 pp
- Turnover tendencies (giveaways vs takeaways): edge **Southern Utah** (0.84 SD); vs-average expectation — Utah Tech offense vs Southern Utah defense +0.95 pp turnover rate, Southern Utah offense vs Utah Tech defense +0.15 pp turnover rate
- Run game vs run defense (opp-adj rush EPA/play): edge **Southern Utah** (0.62 SD); vs-average expectation — Utah Tech offense vs Southern Utah defense -0.05 EPA/play, Southern Utah offense vs Utah Tech defense +0.11 EPA/play
- OL protection vs pass rush (sack rate): edge **Southern Utah** (0.17 SD); vs-average expectation — Utah Tech offense vs Southern Utah defense -1.09 pp sack rate, Southern Utah offense vs Utah Tech defense -1.71 pp sack rate
- QB: Utah Tech Bronson Barben (6.0 YPA, 2 TD/5 INT) vs Southern Utah Will Burns (7.4 YPA, 6 TD/5 INT)
- Home field: Utah Tech (Elo home bonus 45 rating pts; scoring-model home coefficient 6.4 pts on 2026 data)
- Recent form (last-3 margin vs expectation): Utah Tech -0.6, Southern Utah -2.9
- Strength of schedule (avg opp net rating): Utah Tech +7.4, Southern Utah +2.5
- Rest: Southern Utah 7 days, Utah Tech 7 days; travel distance: Data unavailable
- Weather: Forecast: Mostly sunny, 94.0°F, gusts 2.0 mph, precip 0.0%
- Injuries/availability: Data unavailable (no public CFB injury feed); QB continuity — Southern Utah starter share 100%, Utah Tech 100%

</details>

### Arkansas State @ Louisiana
*2026-10-03 08:00 PM ET · Our Lady of Lourdes Stadium (Lafayette, LA) · nan vs nan · records: Arkansas State 2-2, Louisiana 3-1*

**Prediction: Louisiana**  
Win Probability: Arkansas State 33% / Louisiana 67%  
Projected Score: Arkansas State 22–29 Louisiana  
Projected Margin: 7.1 (Louisiana)  
Upset Probability: 33%  
Confidence: 3.8/10

Key Factors:

1. Finishing drives (pts per scoring opportunity): edge **Louisiana** (1.69 SD); vs-average expectation — Louisiana offense vs Arkansas State defense +1.01 pts, Arkansas State offense vs Louisiana defense -0.94 pts
2. Explosive-play rate matchup: edge **Louisiana** (0.73 SD); vs-average expectation — Louisiana offense vs Arkansas State defense +0.35 pp, Arkansas State offense vs Louisiana defense -3.10 pp
3. Run game vs run defense (opp-adj rush EPA/play): edge **Louisiana** (0.52 SD); vs-average expectation — Louisiana offense vs Arkansas State defense -0.07 EPA/play, Arkansas State offense vs Louisiana defense -0.20 EPA/play

Main Risk: Arkansas State's best counter is Success-rate matchup (0.17 SD edge); weather (wind/storms) adds variance

<details><summary>Model breakdown & matchup detail</summary>

| Model | Elo | Adj. scoring model | Logistic reg. | Gradient boosting | Point-diff regression | **Ensemble** |
|---|---|---|---|---|---|---|
| P(Louisiana win) | 0.672 | 0.746 | 0.649 | 0.759 | 0.619 | **0.668** |
| Home margin | +7.9 | +11.0 | — | +7.7 (GBM reg.) | +4.9 | **+7.1** |

- Elo: Arkansas State 1459, Louisiana 1510; adj. net points rating: Arkansas State +13.7, Louisiana +18.5
- Pass game vs pass defense (opp-adj pass EPA/play): edge **Louisiana** (0.50 SD); vs-average expectation — Louisiana offense vs Arkansas State defense +0.04 EPA/play, Arkansas State offense vs Louisiana defense -0.12 EPA/play
- Overall offense vs defense (opp-adj EPA/play): edge **Louisiana** (0.47 SD); vs-average expectation — Louisiana offense vs Arkansas State defense -0.02 EPA/play, Arkansas State offense vs Louisiana defense -0.14 EPA/play
- Field position / special teams (start field pos.): edge **Louisiana** (0.44 SD); vs-average expectation — Louisiana offense vs Arkansas State defense +0.29 yds, Arkansas State offense vs Louisiana defense -1.63 yds
- Turnover tendencies (giveaways vs takeaways): edge **Louisiana** (0.27 SD); vs-average expectation — Louisiana offense vs Arkansas State defense +0.38 pp turnover rate, Arkansas State offense vs Louisiana defense +0.63 pp turnover rate
- Success-rate matchup: edge **Arkansas State** (0.17 SD); vs-average expectation — Louisiana offense vs Arkansas State defense +1.08 pp, Arkansas State offense vs Louisiana defense +2.93 pp
- OL protection vs pass rush (sack rate): edge **Arkansas State** (0.01 SD); vs-average expectation — Louisiana offense vs Arkansas State defense -1.12 pp sack rate, Arkansas State offense vs Louisiana defense -1.15 pp sack rate
- QB: Louisiana Lunch Winfield (7.4 YPA, 2 TD/2 INT) vs Arkansas State Trey Owens (8.2 YPA, 4 TD/4 INT)
- Home field: Louisiana (Elo home bonus 45 rating pts; scoring-model home coefficient 6.4 pts on 2026 data)
- Recent form (last-3 margin vs expectation): Louisiana +7.0, Arkansas State -0.3
- Strength of schedule (avg opp net rating): Louisiana +10.6, Arkansas State +12.4
- Rest: Arkansas State 7 days, Louisiana 7 days; travel distance: Data unavailable
- Weather: Forecast: Thunderstorms, 81.0°F, gusts 6.0 mph, precip 79.0% — flagged (wind/storms)
- Injuries/availability: Data unavailable (no public CFB injury feed); QB continuity — Arkansas State starter share 100%, Louisiana 100%

</details>

### Fresno State @ Washington State
*2026-10-03 09:30 PM ET · Martin Stadium (Pullman, WA) · Pac-12 · records: Fresno State 3-1, Washington State 1-3*

**Prediction: Washington State**  
Win Probability: Fresno State 49% / Washington State 51%  
Projected Score: Fresno State 22–23 Washington State  
Projected Margin: 0.4 (Washington State)  
Upset Probability: 49%  
Confidence: 1.0/10  ⚠️ HIGH MODEL DISAGREEMENT

Key Factors:

1. Field position / special teams (start field pos.): edge **Fresno State** (1.97 SD); vs-average expectation — Washington State offense vs Fresno State defense -5.75 yds, Fresno State offense vs Washington State defense +2.78 yds
2. Turnover tendencies (giveaways vs takeaways): edge **Fresno State** (1.57 SD); vs-average expectation — Washington State offense vs Fresno State defense +1.01 pp turnover rate, Fresno State offense vs Washington State defense -0.48 pp turnover rate
3. OL protection vs pass rush (sack rate): edge **Fresno State** (0.85 SD); vs-average expectation — Washington State offense vs Fresno State defense +1.23 pp sack rate, Fresno State offense vs Washington State defense -1.87 pp sack rate

Main Risk: models disagree on the winner; Fresno State's best counter is Field position / special teams (1.97 SD edge)

<details><summary>Model breakdown & matchup detail</summary>

| Model | Elo | Adj. scoring model | Logistic reg. | Gradient boosting | Point-diff regression | **Ensemble** |
|---|---|---|---|---|---|---|
| P(Washington State win) | 0.423 | 0.547 | 0.464 | 0.395 | 0.525 | **0.505** |
| Home margin | -2.4 | +2.0 | — | -2.1 (GBM reg.) | +1.0 | **+0.4** |

- Elo: Fresno State 1588, Washington State 1475; adj. net points rating: Fresno State +19.8, Washington State +16.7
- Explosive-play rate matchup: edge **Fresno State** (0.40 SD); vs-average expectation — Washington State offense vs Fresno State defense -2.14 pp, Fresno State offense vs Washington State defense -0.26 pp
- Success-rate matchup: edge **Washington State** (0.39 SD); vs-average expectation — Washington State offense vs Fresno State defense +1.39 pp, Fresno State offense vs Washington State defense -2.83 pp
- Pass game vs pass defense (opp-adj pass EPA/play): edge **Fresno State** (0.35 SD); vs-average expectation — Washington State offense vs Fresno State defense -0.13 EPA/play, Fresno State offense vs Washington State defense -0.02 EPA/play
- Overall offense vs defense (opp-adj EPA/play): edge **Fresno State** (0.21 SD); vs-average expectation — Washington State offense vs Fresno State defense -0.10 EPA/play, Fresno State offense vs Washington State defense -0.04 EPA/play
- Finishing drives (pts per scoring opportunity): edge **Fresno State** (0.12 SD); vs-average expectation — Washington State offense vs Fresno State defense -1.22 pts, Fresno State offense vs Washington State defense -1.08 pts
- Run game vs run defense (opp-adj rush EPA/play): edge **Washington State** (0.08 SD); vs-average expectation — Washington State offense vs Fresno State defense -0.03 EPA/play, Fresno State offense vs Washington State defense -0.05 EPA/play
- QB: Washington State Caden Pinnick (6.0 YPA, 3 TD/4 INT) vs Fresno State Jayden Mandal (7.6 YPA, 6 TD/1 INT)
- Home field: Washington State (Elo home bonus 45 rating pts; scoring-model home coefficient 6.4 pts on 2026 data)
- Recent form (last-3 margin vs expectation): Washington State -7.4, Fresno State +15.5
- Strength of schedule (avg opp net rating): Washington State +16.1, Fresno State +11.8
- Rest: Fresno State 6 days, Washington State 7 days; travel distance: Data unavailable
- Weather: Forecast: Sunny, 70.0°F, gusts 7.0 mph, precip 0.0%
- Injuries/availability: Data unavailable (no public CFB injury feed); QB continuity — Fresno State starter share 100%, Washington State 100%

</details>

### Baylor @ Arizona State
*2026-10-03 10:30 PM ET · Mountain America Stadium (Tempe, AZ) · Big 12 · records: Baylor 3-1, Arizona State 2-1*

**Prediction: Arizona State**  
Win Probability: Baylor 40% / Arizona State 60%  
Projected Score: Baylor 22–26 Arizona State  
Projected Margin: 3.7 (Arizona State)  
Upset Probability: 40%  
Confidence: 2.0/10

Key Factors:

1. Explosive-play rate matchup: edge **Arizona State** (0.67 SD); vs-average expectation — Arizona State offense vs Baylor defense +0.86 pp, Baylor offense vs Arizona State defense -2.31 pp
2. Field position / special teams (start field pos.): edge **Baylor** (0.64 SD); vs-average expectation — Arizona State offense vs Baylor defense +0.79 yds, Baylor offense vs Arizona State defense +3.56 yds
3. OL protection vs pass rush (sack rate): edge **Arizona State** (0.35 SD); vs-average expectation — Arizona State offense vs Baylor defense -1.66 pp sack rate, Baylor offense vs Arizona State defense -0.37 pp sack rate

Main Risk: Baylor's best counter is Field position / special teams (0.64 SD edge); QB change last game (Baylor) — availability unconfirmed

<details><summary>Model breakdown & matchup detail</summary>

| Model | Elo | Adj. scoring model | Logistic reg. | Gradient boosting | Point-diff regression | **Ensemble** |
|---|---|---|---|---|---|---|
| P(Arizona State win) | 0.637 | 0.663 | 0.607 | 0.597 | 0.549 | **0.596** |
| Home margin | +6.3 | +7.0 | — | +2.7 (GBM reg.) | +2.0 | **+3.7** |

- Elo: Baylor 1566, Arizona State 1592; adj. net points rating: Baylor +23.4, Arizona State +24.7
- Finishing drives (pts per scoring opportunity): edge **Baylor** (0.32 SD); vs-average expectation — Arizona State offense vs Baylor defense -0.22 pts, Baylor offense vs Arizona State defense +0.16 pts
- Overall offense vs defense (opp-adj EPA/play): edge **Arizona State** (0.28 SD); vs-average expectation — Arizona State offense vs Baylor defense -0.06 EPA/play, Baylor offense vs Arizona State defense -0.13 EPA/play
- Pass game vs pass defense (opp-adj pass EPA/play): edge **Arizona State** (0.28 SD); vs-average expectation — Arizona State offense vs Baylor defense -0.06 EPA/play, Baylor offense vs Arizona State defense -0.15 EPA/play
- Turnover tendencies (giveaways vs takeaways): edge **Baylor** (0.25 SD); vs-average expectation — Arizona State offense vs Baylor defense +0.43 pp turnover rate, Baylor offense vs Arizona State defense +0.19 pp turnover rate
- Success-rate matchup: edge **Arizona State** (0.18 SD); vs-average expectation — Arizona State offense vs Baylor defense +0.37 pp, Baylor offense vs Arizona State defense -1.54 pp
- Run game vs run defense (opp-adj rush EPA/play): edge **Arizona State** (0.09 SD); vs-average expectation — Arizona State offense vs Baylor defense -0.10 EPA/play, Baylor offense vs Arizona State defense -0.12 EPA/play
- QB: Arizona State Cutter Boley (8.8 YPA, 9 TD/2 INT) vs Baylor DJ Lagway (6.0 YPA, 4 TD/2 INT, NEW STARTER last game)
- Home field: Arizona State (Elo home bonus 45 rating pts; scoring-model home coefficient 6.4 pts on 2026 data)
- Recent form (last-3 margin vs expectation): Arizona State +5.3, Baylor +4.2
- Strength of schedule (avg opp net rating): Arizona State +13.2, Baylor +14.3
- Rest: Baylor 7 days, Arizona State 14 days; travel distance: Data unavailable
- Weather: Forecast: Clear, 93.0°F, gusts 12.0 mph, precip 0.0%
- Injuries/availability: Data unavailable (no public CFB injury feed); QB continuity — Baylor starter share 61%, Arizona State 100%

</details>

### Texas State @ San Diego State
*2026-10-03 10:30 PM ET · Snapdragon Stadium (San Diego, CA) · Pac-12 · records: Texas State 2-2, San Diego State 1-3*

**Prediction: Texas State**  
Win Probability: Texas State 61% / San Diego State 39%  
Projected Score: Texas State 29–24 San Diego State  
Projected Margin: 4.8 (Texas State)  
Upset Probability: 39%  
Confidence: 1.4/10  ⚠️ HIGH MODEL DISAGREEMENT

Key Factors:

1. Pass game vs pass defense (opp-adj pass EPA/play): edge **Texas State** (0.95 SD); vs-average expectation — San Diego State offense vs Texas State defense -0.09 EPA/play, Texas State offense vs San Diego State defense +0.21 EPA/play
2. Success-rate matchup: edge **Texas State** (0.68 SD); vs-average expectation — San Diego State offense vs Texas State defense -1.74 pp, Texas State offense vs San Diego State defense +5.62 pp
3. Turnover tendencies (giveaways vs takeaways): edge **Texas State** (0.62 SD); vs-average expectation — San Diego State offense vs Texas State defense +0.11 pp turnover rate, Texas State offense vs San Diego State defense -0.49 pp turnover rate

Main Risk: models disagree on the winner; San Diego State's best counter is Finishing drives (0.55 SD edge)

<details><summary>Model breakdown & matchup detail</summary>

| Model | Elo | Adj. scoring model | Logistic reg. | Gradient boosting | Point-diff regression | **Ensemble** |
|---|---|---|---|---|---|---|
| P(San Diego State win) | 0.346 | 0.529 | 0.320 | 0.389 | 0.349 | **0.389** |
| Home margin | -5.5 | +1.2 | — | -9.5 (GBM reg.) | -6.3 | **-4.8** |

- Elo: Texas State 1596, San Diego State 1432; adj. net points rating: Texas State +19.7, San Diego State +15.9
- Explosive-play rate matchup: edge **Texas State** (0.57 SD); vs-average expectation — San Diego State offense vs Texas State defense -2.40 pp, Texas State offense vs San Diego State defense +0.27 pp
- Finishing drives (pts per scoring opportunity): edge **San Diego State** (0.55 SD); vs-average expectation — San Diego State offense vs Texas State defense +0.32 pts, Texas State offense vs San Diego State defense -0.32 pts
- Run game vs run defense (opp-adj rush EPA/play): edge **Texas State** (0.51 SD); vs-average expectation — San Diego State offense vs Texas State defense -0.19 EPA/play, Texas State offense vs San Diego State defense -0.06 EPA/play
- Overall offense vs defense (opp-adj EPA/play): edge **Texas State** (0.43 SD); vs-average expectation — San Diego State offense vs Texas State defense -0.09 EPA/play, Texas State offense vs San Diego State defense +0.02 EPA/play
- OL protection vs pass rush (sack rate): edge **Texas State** (0.35 SD); vs-average expectation — San Diego State offense vs Texas State defense -1.40 pp sack rate, Texas State offense vs San Diego State defense -2.69 pp sack rate
- Field position / special teams (start field pos.): edge **San Diego State** (0.33 SD); vs-average expectation — San Diego State offense vs Texas State defense -0.40 yds, Texas State offense vs San Diego State defense -1.81 yds
- QB: San Diego State Jayden Denegal (6.1 YPA, 5 TD/5 INT) vs Texas State Brad Jackson (9.0 YPA, 10 TD/3 INT)
- Home field: San Diego State (Elo home bonus 45 rating pts; scoring-model home coefficient 6.4 pts on 2026 data)
- Recent form (last-3 margin vs expectation): San Diego State -9.1, Texas State +16.1
- Strength of schedule (avg opp net rating): San Diego State +18.1, Texas State +20.8
- Rest: Texas State 7 days, San Diego State 7 days; travel distance: Data unavailable
- Weather: Forecast: Clear, 92.0°F, gusts 6.0 mph, precip 0.0%
- Injuries/availability: Data unavailable (no public CFB injury feed); QB continuity — Texas State starter share 100%, San Diego State 100%

</details>

### Eastern Washington @ UC Davis
*2026-10-03 10:30 PM ET · UC Davis Health Stadium · Big Sky · records: Eastern Washington 1-4, UC Davis 3-2*

**Prediction: UC Davis**  
Win Probability: Eastern Washington 28% / UC Davis 72%  
Projected Score: Eastern Washington 24–33 UC Davis  
Projected Margin: 9.6 (UC Davis)  
Upset Probability: 28%  
Confidence: 4.7/10

Key Factors:

1. Run game vs run defense (opp-adj rush EPA/play): edge **UC Davis** (1.30 SD); vs-average expectation — UC Davis offense vs Eastern Washington defense +0.25 EPA/play, Eastern Washington offense vs UC Davis defense -0.08 EPA/play
2. Pass game vs pass defense (opp-adj pass EPA/play): edge **Eastern Washington** (0.96 SD); vs-average expectation — UC Davis offense vs Eastern Washington defense -0.23 EPA/play, Eastern Washington offense vs UC Davis defense +0.07 EPA/play
3. OL protection vs pass rush (sack rate): edge **UC Davis** (0.57 SD); vs-average expectation — UC Davis offense vs Eastern Washington defense -0.80 pp sack rate, Eastern Washington offense vs UC Davis defense +1.27 pp sack rate

Main Risk: Eastern Washington's best counter is Pass game vs pass defense (0.96 SD edge)

<details><summary>Model breakdown & matchup detail</summary>

| Model | Elo | Adj. scoring model | Logistic reg. | Gradient boosting | Point-diff regression | **Ensemble** |
|---|---|---|---|---|---|---|
| P(UC Davis win) | 0.863 | 0.676 | 0.732 | 0.788 | 0.728 | **0.720** |
| Home margin | +19.0 | +7.6 | — | +12.5 (GBM reg.) | +9.8 | **+9.6** |

- Elo: Eastern Washington 1124, UC Davis 1352; adj. net points rating: Eastern Washington -5.7, UC Davis -3.9
- Field position / special teams (start field pos.): edge **UC Davis** (0.38 SD); vs-average expectation — UC Davis offense vs Eastern Washington defense +1.22 yds, Eastern Washington offense vs UC Davis defense -0.42 yds
- Success-rate matchup: edge **UC Davis** (0.37 SD); vs-average expectation — UC Davis offense vs Eastern Washington defense +4.39 pp, Eastern Washington offense vs UC Davis defense +0.42 pp
- Explosive-play rate matchup: edge **UC Davis** (0.29 SD); vs-average expectation — UC Davis offense vs Eastern Washington defense +1.74 pp, Eastern Washington offense vs UC Davis defense +0.40 pp
- Finishing drives (pts per scoring opportunity): edge **Eastern Washington** (0.22 SD); vs-average expectation — UC Davis offense vs Eastern Washington defense +0.18 pts, Eastern Washington offense vs UC Davis defense +0.44 pts
- Turnover tendencies (giveaways vs takeaways): edge **Eastern Washington** (0.10 SD); vs-average expectation — UC Davis offense vs Eastern Washington defense -0.09 pp turnover rate, Eastern Washington offense vs UC Davis defense -0.19 pp turnover rate
- Overall offense vs defense (opp-adj EPA/play): edge **Eastern Washington** (0.04 SD); vs-average expectation — UC Davis offense vs Eastern Washington defense +0.02 EPA/play, Eastern Washington offense vs UC Davis defense +0.03 EPA/play
- QB: UC Davis Treynor Cleeland (7.2 YPA, 10 TD/3 INT) vs Eastern Washington Nate Bell (6.7 YPA, 8 TD/4 INT)
- Home field: UC Davis (Elo home bonus 45 rating pts; scoring-model home coefficient 6.4 pts on 2026 data)
- Recent form (last-3 margin vs expectation): UC Davis -21.6, Eastern Washington -4.4
- Strength of schedule (avg opp net rating): UC Davis -5.6, Eastern Washington +3.6
- Rest: Eastern Washington 7 days, UC Davis 7 days; travel distance: Data unavailable
- Weather: Forecast: Clear, 91.0°F, gusts 5.0 mph, precip 0.0%
- Injuries/availability: Data unavailable (no public CFB injury feed); QB continuity — Eastern Washington starter share 100%, UC Davis 100%

</details>

### Cincinnati @ Arizona
*2026-10-03 11:00 PM ET · Casino Del Sol Stadium (Tucson, AZ) · Big 12 · records: Cincinnati 4-0, Arizona 3-1*

**Prediction: Arizona**  
Win Probability: Cincinnati 30% / Arizona 70%  
Projected Score: Cincinnati 24–32 Arizona  
Projected Margin: 8.3 (Arizona)  
Upset Probability: 30%  
Confidence: 4.8/10

Key Factors:

1. Success-rate matchup: edge **Arizona** (0.77 SD); vs-average expectation — Arizona offense vs Cincinnati defense +8.20 pp, Cincinnati offense vs Arizona defense -0.07 pp
2. Explosive-play rate matchup: edge **Arizona** (0.40 SD); vs-average expectation — Arizona offense vs Cincinnati defense +1.22 pp, Cincinnati offense vs Arizona defense -0.64 pp
3. OL protection vs pass rush (sack rate): edge **Cincinnati** (0.36 SD); vs-average expectation — Arizona offense vs Cincinnati defense +0.17 pp sack rate, Cincinnati offense vs Arizona defense -1.13 pp sack rate

Main Risk: Cincinnati's best counter is OL protection vs pass rush (0.36 SD edge)

<details><summary>Model breakdown & matchup detail</summary>

| Model | Elo | Adj. scoring model | Logistic reg. | Gradient boosting | Point-diff regression | **Ensemble** |
|---|---|---|---|---|---|---|
| P(Arizona win) | 0.700 | 0.673 | 0.719 | 0.661 | 0.708 | **0.699** |
| Home margin | +9.1 | +7.5 | — | +7.7 (GBM reg.) | +8.9 | **+8.3** |

- Elo: Cincinnati 1608, Arizona 1679; adj. net points rating: Cincinnati +23.8, Arizona +25.6
- Pass game vs pass defense (opp-adj pass EPA/play): edge **Arizona** (0.35 SD); vs-average expectation — Arizona offense vs Cincinnati defense +0.20 EPA/play, Cincinnati offense vs Arizona defense +0.09 EPA/play
- Field position / special teams (start field pos.): edge **Cincinnati** (0.28 SD); vs-average expectation — Arizona offense vs Cincinnati defense -2.56 yds, Cincinnati offense vs Arizona defense -1.34 yds
- Run game vs run defense (opp-adj rush EPA/play): edge **Cincinnati** (0.25 SD); vs-average expectation — Arizona offense vs Cincinnati defense +0.08 EPA/play, Cincinnati offense vs Arizona defense +0.15 EPA/play
- Overall offense vs defense (opp-adj EPA/play): edge **Arizona** (0.20 SD); vs-average expectation — Arizona offense vs Cincinnati defense +0.15 EPA/play, Cincinnati offense vs Arizona defense +0.11 EPA/play
- Turnover tendencies (giveaways vs takeaways): edge **Arizona** (0.12 SD); vs-average expectation — Arizona offense vs Cincinnati defense -0.68 pp turnover rate, Cincinnati offense vs Arizona defense -0.56 pp turnover rate
- Finishing drives (pts per scoring opportunity): edge **Cincinnati** (0.04 SD); vs-average expectation — Arizona offense vs Cincinnati defense -0.00 pts, Cincinnati offense vs Arizona defense +0.05 pts
- QB: Arizona Noah Fifita (8.6 YPA, 11 TD/2 INT) vs Cincinnati JC French IV (8.5 YPA, 6 TD/0 INT)
- Home field: Arizona (Elo home bonus 45 rating pts; scoring-model home coefficient 6.4 pts on 2026 data)
- Recent form (last-3 margin vs expectation): Arizona +4.3, Cincinnati +2.1
- Strength of schedule (avg opp net rating): Arizona +13.2, Cincinnati +12.3
- Rest: Cincinnati 7 days, Arizona 7 days; travel distance: Data unavailable
- Weather: Forecast: Clear, 81.0°F, gusts 12.0 mph, precip 0.0%
- Injuries/availability: Data unavailable (no public CFB injury feed); QB continuity — Cincinnati starter share 100%, Arizona 100%

</details>

### San José State @ Hawai'i
*2026-10-03 11:59 PM ET · Clarence T.C. Ching Athletics Complex (Honolulu, HI) · Mountain West · records: San José State 2-2, Hawai'i 1-3*

**Prediction: Hawai'i**  
Win Probability: San José State 44% / Hawai'i 56%  
Projected Score: San José State 25–27 Hawai'i  
Projected Margin: 2.2 (Hawai'i)  
Upset Probability: 44%  
Confidence: 1.2/10

Key Factors:

1. Explosive-play rate matchup: edge **San José State** (1.08 SD); vs-average expectation — Hawai'i offense vs San José State defense -3.27 pp, San José State offense vs Hawai'i defense +1.83 pp
2. OL protection vs pass rush (sack rate): edge **San José State** (0.56 SD); vs-average expectation — Hawai'i offense vs San José State defense -0.62 pp sack rate, San José State offense vs Hawai'i defense -2.67 pp sack rate
3. Field position / special teams (start field pos.): edge **Hawai'i** (0.56 SD); vs-average expectation — Hawai'i offense vs San José State defense -0.53 yds, San José State offense vs Hawai'i defense -2.94 yds

Main Risk: San José State's best counter is Explosive-play rate matchup (1.08 SD edge); weather (wind/storms) adds variance

<details><summary>Model breakdown & matchup detail</summary>

| Model | Elo | Adj. scoring model | Logistic reg. | Gradient boosting | Point-diff regression | **Ensemble** |
|---|---|---|---|---|---|---|
| P(Hawai'i win) | 0.675 | 0.640 | 0.520 | 0.525 | 0.532 | **0.555** |
| Home margin | +8.0 | +6.0 | — | -0.7 (GBM reg.) | +1.3 | **+2.2** |

- Elo: San José State 1339, Hawai'i 1392; adj. net points rating: San José State +9.1, Hawai'i +9.5
- Turnover tendencies (giveaways vs takeaways): edge **Hawai'i** (0.55 SD); vs-average expectation — Hawai'i offense vs San José State defense -0.52 pp turnover rate, San José State offense vs Hawai'i defense +0.00 pp turnover rate
- Finishing drives (pts per scoring opportunity): edge **Hawai'i** (0.53 SD); vs-average expectation — Hawai'i offense vs San José State defense +0.45 pts, San José State offense vs Hawai'i defense -0.17 pts
- Pass game vs pass defense (opp-adj pass EPA/play): edge **Hawai'i** (0.32 SD); vs-average expectation — Hawai'i offense vs San José State defense -0.03 EPA/play, San José State offense vs Hawai'i defense -0.13 EPA/play
- Overall offense vs defense (opp-adj EPA/play): edge **Hawai'i** (0.27 SD); vs-average expectation — Hawai'i offense vs San José State defense +0.02 EPA/play, San José State offense vs Hawai'i defense -0.05 EPA/play
- Run game vs run defense (opp-adj rush EPA/play): edge **Hawai'i** (0.14 SD); vs-average expectation — Hawai'i offense vs San José State defense +0.05 EPA/play, San José State offense vs Hawai'i defense +0.01 EPA/play
- Success-rate matchup: edge **San José State** (0.12 SD); vs-average expectation — Hawai'i offense vs San José State defense -0.46 pp, San José State offense vs Hawai'i defense +0.84 pp
- QB: Hawai'i Micah Alejado (5.6 YPA, 3 TD/1 INT) vs San José State Luke Weaver (7.2 YPA, 9 TD/5 INT)
- Home field: Hawai'i (Elo home bonus 45 rating pts; scoring-model home coefficient 6.4 pts on 2026 data)
- Recent form (last-3 margin vs expectation): Hawai'i -7.8, San José State -11.4
- Strength of schedule (avg opp net rating): Hawai'i +12.2, San José State +11.5
- Rest: San José State 14 days, Hawai'i 7 days; travel distance: Data unavailable
- Weather: Forecast: Partly sunny, 82.0°F, gusts 23.0 mph, precip 0.0% — flagged (wind/storms)
- Injuries/availability: Data unavailable (no public CFB injury feed); QB continuity — San José State starter share 100%, Hawai'i 100%

</details>

## Final slate summary

| Game | Kickoff | Predicted Winner | Win % | Projected Score | Upset % | Confidence | Flags |
|---|---|---|---|---|---|---|---|
| Youngstown State @ Southern Illinois | 05:00 PM ET | Youngstown State | 57% | Youngstown State 32–29 Southern Illinois | 43% | 1.0 | disagree |
| New Haven @ Austin Peay | 05:00 PM ET | Austin Peay | 99% | New Haven 8–45 Austin Peay | 1% | 9.2 | QB chg |
| Alabama A&M @ Jackson State | 05:00 PM ET | Jackson State | 59% | Alabama A&M 25–29 Jackson State | 41% | 1.7 | disagree |
| Oregon State @ Colorado State | 06:00 PM ET | Oregon State | 63% | Oregon State 32–26 Colorado State | 37% | 1.6 | disagree |
| Texas Southern @ Florida Atlantic | 06:00 PM ET | Florida Atlantic | 99% | Texas Southern 13–52 Florida Atlantic | 1% | 9.7 |  |
| North Carolina A&T @ Bryant | 06:00 PM ET | Bryant | 84% | North Carolina A&T 18–34 Bryant | 16% | 6.7 | QB chg |
| Campbell @ North Carolina Central | 06:00 PM ET | Campbell | 68% | Campbell 31–23 North Carolina Central | 32% | 2.1 | disagree, QB chg |
| North Alabama @ Eastern Kentucky | 06:00 PM ET | North Alabama | 60% | North Alabama 25–21 Eastern Kentucky | 40% | 1.0 | disagree, QB chg |
| Gardner-Webb @ Charleston Southern | 06:00 PM ET | Gardner-Webb | 56% | Gardner-Webb 24–22 Charleston Southern | 44% | 1.0 | disagree, QB chg |
| Incarnate Word @ Stephen F. Austin | 06:00 PM ET | Stephen F. Austin | 90% | Incarnate Word 18–39 Stephen F. Austin | 10% | 8.1 |  |
| Tennessee Tech @ Western Carolina | 06:00 PM ET | Tennessee Tech | 71% | Tennessee Tech 34–25 Western Carolina | 29% | 4.9 |  |
| Arkansas @ Texas A&M | 07:00 PM ET | Texas A&M | 83% | Arkansas 21–36 Texas A&M | 17% | 7.0 |  |
| BYU @ TCU | 07:00 PM ET | BYU | 58% | BYU 27–24 TCU | 42% | 1.0 | disagree |
| UTSA @ Rice | 07:00 PM ET | UTSA | 80% | UTSA 34–20 Rice | 20% | 6.4 |  |
| Chicago State @ Tarleton State | 07:00 PM ET | Tarleton State | 98% | Chicago State 13–48 Tarleton State | 2% | 9.7 |  |
| Colgate @ Harvard | 07:00 PM ET | Harvard | 57% | Colgate 22–24 Harvard | 43% | 1.0 | thin data |
| South Dakota State @ Illinois State | 07:00 PM ET | Illinois State | 54% | South Dakota State 26–28 Illinois State | 46% | 1.0 | disagree, QB chg |
| UT Rio Grande Valley @ East Texas A&M | 07:00 PM ET | UT Rio Grande Valley | 68% | UT Rio Grande Valley 28–20 East Texas A&M | 32% | 3.7 | QB chg |
| Arkansas-Pine Bluff @ Southern | 07:00 PM ET | Arkansas-Pine Bluff | 51% | Arkansas-Pine Bluff 28–29 Southern | 49% | 1.0 | disagree |
| Northwestern State @ Lamar | 07:00 PM ET | Lamar | 99% | Northwestern State 10–48 Lamar | 1% | 9.8 |  |
| Nicholls @ Houston Christian | 07:00 PM ET | Houston Christian | 85% | Nicholls 19–35 Houston Christian | 15% | 6.3 | disagree |
| Georgia Southern @ Coastal Carolina | 07:00 PM ET | Georgia Southern | 52% | Georgia Southern 26–25 Coastal Carolina | 48% | 1.0 | disagree |
| UL Monroe @ South Alabama | 07:00 PM ET | South Alabama | 88% | UL Monroe 21–40 South Alabama | 12% | 7.5 | QB chg |
| Texas Tech @ Colorado | 07:30 PM ET | Texas Tech | 83% | Texas Tech 30–15 Colorado | 17% | 6.2 | QB chg |
| Miami @ Clemson | 07:30 PM ET | Miami | 88% | Miami 35–16 Clemson | 12% | 7.5 |  |
| Washington @ USC | 07:30 PM ET | USC | 76% | Washington 24–36 USC | 24% | 5.5 |  |
| Utah State @ Boise State | 07:30 PM ET | Boise State | 90% | Utah State 15–35 Boise State | 10% | 8.4 |  |
| Temple @ South Florida | 07:30 PM ET | South Florida | 83% | Temple 21–36 South Florida | 17% | 6.8 |  |
| Army @ Louisiana Tech | 07:30 PM ET | Louisiana Tech | 54% | Army 26–28 Louisiana Tech | 46% | 1.0 | disagree, QB chg |
| McNeese @ LSU | 07:45 PM ET | LSU | 100% | McNeese 5–54 LSU | 0% | 9.4 | QB chg |
| Indiana @ Rutgers | 08:00 PM ET | Indiana | 93% | Indiana 43–19 Rutgers | 7% | 8.7 |  |
| West Florida @ Abilene Christian | 08:00 PM ET | Abilene Christian | 63% | West Florida 23–28 Abilene Christian | 37% | 2.7 | QB chg |
| Weber State @ Cal Poly | 08:00 PM ET | Cal Poly | 63% | Weber State 27–32 Cal Poly | 37% | 3.4 |  |
| Northern Arizona @ Idaho State | 08:00 PM ET | Idaho State | 82% | Northern Arizona 19–34 Idaho State | 18% | 6.8 |  |
| Southern Utah @ Utah Tech | 08:00 PM ET | Southern Utah | 79% | Southern Utah 34–21 Utah Tech | 21% | 6.3 |  |
| Arkansas State @ Louisiana | 08:00 PM ET | Louisiana | 67% | Arkansas State 22–29 Louisiana | 33% | 3.8 |  |
| Fresno State @ Washington State | 09:30 PM ET | Washington State | 51% | Fresno State 22–23 Washington State | 49% | 1.0 | disagree |
| Baylor @ Arizona State | 10:30 PM ET | Arizona State | 60% | Baylor 22–26 Arizona State | 40% | 2.0 | QB chg |
| Texas State @ San Diego State | 10:30 PM ET | Texas State | 61% | Texas State 29–24 San Diego State | 39% | 1.4 | disagree |
| Eastern Washington @ UC Davis | 10:30 PM ET | UC Davis | 72% | Eastern Washington 24–33 UC Davis | 28% | 4.7 |  |
| Cincinnati @ Arizona | 11:00 PM ET | Arizona | 70% | Cincinnati 24–32 Arizona | 30% | 4.8 |  |
| San José State @ Hawai'i | 11:59 PM ET | Hawai'i | 56% | San José State 25–27 Hawai'i | 44% | 1.2 |  |

### Highest-confidence games
- Lamar over Northwestern State — 99%, conf 9.8
- Florida Atlantic over Texas Southern — 99%, conf 9.7
- Tarleton State over Chicago State — 98%, conf 9.7
- LSU over McNeese — 100%, conf 9.4
- Austin Peay over New Haven — 99%, conf 9.2
- Indiana over Rutgers — 93%, conf 8.7

### Closest games
- Fresno State @ Washington State — Washington State 51%, margin 0.4
- Arkansas-Pine Bluff @ Southern — Arkansas-Pine Bluff 51%, margin 0.2
- Georgia Southern @ Coastal Carolina — Georgia Southern 52%, margin 0.9
- South Dakota State @ Illinois State — Illinois State 54%, margin 1.8
- Army @ Louisiana Tech — Louisiana Tech 54%, margin 1.5
- San José State @ Hawai'i — Hawai'i 56%, margin 2.2

### Best upset candidates (model underdog ≥ 25%)
- Fresno State over Washington State — 49% (reference market line, not a model input: WSU -2.5)
- Southern over Arkansas-Pine Bluff — 49% (reference market line, not a model input: SOU -2.5)
- Coastal Carolina over Georgia Southern — 48% (reference market line, not a model input: GASO -2.5)
- South Dakota State over Illinois State — 46% (reference market line, not a model input: SDST -2.5)
- Army over Louisiana Tech — 46% (reference market line, not a model input: LT -1.5)
- San José State over Hawai'i — 44% (reference market line, not a model input: HAW -2.5)
- Charleston Southern over Gardner-Webb — 44% (reference market line, not a model input: GWEB -7)
- Southern Illinois over Youngstown State — 43% (reference market line, not a model input: YSU -4)

### Highest model disagreement
- Nicholls @ Houston Christian — member P(Houston Christian) range 0.54–0.89 (SD 0.138)
- Oregon State @ Colorado State — member P(Colorado State) range 0.25–0.52 (SD 0.112); models split on winner
- Alabama A&M @ Jackson State — member P(Jackson State) range 0.57–0.81 (SD 0.100)
- Campbell @ North Carolina Central — member P(North Carolina Central) range 0.28–0.51 (SD 0.094); models split on winner
- Army @ Louisiana Tech — member P(Louisiana Tech) range 0.36–0.57 (SD 0.087); models split on winner
- North Alabama @ Eastern Kentucky — member P(Eastern Kentucky) range 0.38–0.59 (SD 0.087); models split on winner

### Major injury / availability factors

- Official injury reports: **Data unavailable** (no programmatic public source reachable; ESPN college injury feed empty).
- North Carolina Central: starting QB changed in most recent game (starter sequence: Nelson Layne > Nelson Layne > Nelson Layne > Nelson Layne > Cobey Thompkins). Treat as possible injury/depth-chart change — unconfirmed.
- Eastern Kentucky: starting QB changed in most recent game (starter sequence: Tyler Travers > Tyler Travers > Tyler Travers > Jordyn Potts). Treat as possible injury/depth-chart change — unconfirmed.
- Charleston Southern: starting QB changed in most recent game (starter sequence: Lek Powell > Lek Powell > Lek Powell > Jared Echols). Treat as possible injury/depth-chart change — unconfirmed.
- East Texas A&M: starting QB changed in most recent game (starter sequence: Eric Rodriguez > Eric Rodriguez > Jack Jacobs > Jack Jacobs > Eric Rodriguez). Treat as possible injury/depth-chart change — unconfirmed.
- Colorado: starting QB changed in most recent game (starter sequence: Julian Lewis > Julian Lewis > Julian Lewis > Isaac Wilson). Treat as possible injury/depth-chart change — unconfirmed.
- Louisiana Tech: starting QB changed in most recent game (starter sequence: Blake Baker > Blake Baker > Trey Kukuk). Treat as possible injury/depth-chart change — unconfirmed.
- New Haven: starting QB changed in most recent game (starter sequence: AJ Duffy > AJ Duffy > AJ Duffy > AJ Duffy > Jake Tripptree). Treat as possible injury/depth-chart change — unconfirmed.
- North Carolina A&T: starting QB changed in most recent game (starter sequence: Braxton Thomas > Kevin White > Kevin White > Champ Long > Kevin White). Treat as possible injury/depth-chart change — unconfirmed.
- South Dakota State: starting QB changed in most recent game (starter sequence: Chase Mason > Chase Mason > Chase Mason > Chase Mason > Josh Holst). Treat as possible injury/depth-chart change — unconfirmed.
- UT Rio Grande Valley: starting QB changed in most recent game (starter sequence: Aidan Jakobsohn > Garret Rangel > Garret Rangel > Aidan Jakobsohn). Treat as possible injury/depth-chart change — unconfirmed.
- UL Monroe: starting QB changed in most recent game (starter sequence: Aidan Armenta > Austin Carlisle > Austin Carlisle > Aidan Armenta). Treat as possible injury/depth-chart change — unconfirmed.
- McNeese: starting QB changed in most recent game (starter sequence: Devin Lippold > Devin Lippold > Devin Lippold > Devin Lippold > Jake Strong). Treat as possible injury/depth-chart change — unconfirmed.
- West Florida: starting QB changed in most recent game (starter sequence: Ty'ray Davis > Ty'ray Davis > Ty'ray Davis > Donovan Leary). Treat as possible injury/depth-chart change — unconfirmed.
- Baylor: starting QB changed in most recent game (starter sequence: DJ Lagway > Nate Bennett > Nate Bennett > DJ Lagway). Treat as possible injury/depth-chart change — unconfirmed.

### Weather watch
- Oregon State @ Colorado State: Mostly sunny, gusts 28.0 mph, precip 0.0% (not adjusted in model; adds variance)
- Campbell @ North Carolina Central: Cloudy, gusts 20.0 mph, precip 44.0% (not adjusted in model; adds variance)
- Gardner-Webb @ Charleston Southern: Thunderstorms, gusts 10.0 mph, precip 75.0% (not adjusted in model; adds variance)
- Incarnate Word @ Stephen F. Austin: Mostly cloudy w/ t-storms, gusts 14.0 mph, precip 52.0% (not adjusted in model; adds variance)
- BYU @ TCU: Rain, gusts 13.0 mph, precip 85.0% (not adjusted in model; adds variance)
- UTSA @ Rice: Thunderstorms, gusts 14.0 mph, precip 57.0% (not adjusted in model; adds variance)
- Arkansas-Pine Bluff @ Southern: Thunderstorms, gusts 12.0 mph, precip 75.0% (not adjusted in model; adds variance)
- Northwestern State @ Lamar: Thunderstorms, gusts 10.0 mph, precip 66.0% (not adjusted in model; adds variance)
- UL Monroe @ South Alabama: Thunderstorms, gusts 10.0 mph, precip 79.0% (not adjusted in model; adds variance)
- Miami @ Clemson: Mostly cloudy w/ t-storms, gusts 8.0 mph, precip 75.0% (not adjusted in model; adds variance)
- Temple @ South Florida: Mostly cloudy w/ t-storms, gusts 8.0 mph, precip 61.0% (not adjusted in model; adds variance)
- McNeese @ LSU: Thunderstorms, gusts 12.0 mph, precip 75.0% (not adjusted in model; adds variance)
- Arkansas State @ Louisiana: Thunderstorms, gusts 6.0 mph, precip 79.0% (not adjusted in model; adds variance)
- San José State @ Hawai'i: Partly sunny, gusts 23.0 mph, precip 0.0% (not adjusted in model; adds variance)

## Reproducibility

```
pip install pandas numpy scikit-learn scipy lightgbm requests
python src/collect_scores.py 2015 ... 2026   # results
python src/collect_boxscores.py 2026 2025 2024 2023 2022 2021   # box scores + play-by-play
python src/collect_slate.py   # today's slate
cd src && python pipeline.py && python model.py && python report.py
```
Files: `games_today.csv`, `team_stats.csv`, `features.csv`, `predictions.csv` (repo root); validation in `output/validation.json`, walk-forward OOS predictions in `output/walkforward_oos_predictions.csv.gz`.