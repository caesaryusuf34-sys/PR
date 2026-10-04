# KBO — predictions for 2026-10-05

- Model **v1.0** · predictions frozen 2026-10-04T12:38:50Z (before every kickoff listed) · all inputs are pre-game; results/injuries after the cutoff never enter
- Walk-forward validation (1500 out-of-sample games): accuracy 56.0%, log loss 0.680, Brier 0.244, margin MAE 3.8
- Betting lines are not model inputs. Starting pitchers are the announced probables as captured at the freeze (TBD = the team's recent rotation on average). A tied game (NPB/KBO/CPBL) voids the pick.

## Games

```
KIA Tigers @ LG Twins
[BLIND pre-game prediction · model v1.0 · data cutoff 2026-10-04T12:38:50Z · first pitch 2026-10-05T05:00:00Z]

Prediction: KIA Tigers
Win Probability: KIA Tigers 54% / LG Twins 46%
Projected Score: KIA Tigers 4.9–4.2 LG Twins
Projected Margin: 0.7
Upset Probability: 46%
Confidence: 2.1/10

Key Factors:

1. Starting pitching: 올러 (KIA Tigers) vs 톨허스트 (LG Twins) — run-prevention rating -1.01 vs -0.30 runs/game (lower is better) → edge KIA Tigers (0.72 runs)
2. Lineup & defence: opponent-adjusted runs scored/allowed per game (ex-starter) — LG Twins +0.09/-0.02, KIA Tigers +0.18/-0.24 → edge KIA Tigers (0.31 runs)
3. Home field: LG Twins (league home edge -0.28 runs/game) → edge KIA Tigers (0.28 runs)

Main Risk: single-game variance is large in baseball (run-margin sigma ≈ 4.9)

Model drivers (gradient-boosting SHAP, log-odds toward):
  - Starting pitcher run-prevention rating: -0.20 → KIA Tigers
  - Home games played: +0.13 → LG Twins
  - Run environment: +0.10 → LG Twins
  - Elo rating edge (incl. home field): -0.10 → KIA Tigers
  - Opponent-adjusted run model (incl. starters): -0.09 → KIA Tigers

Member P(LG Twins win): elo 0.485, scoring 0.419, logistic 0.427, gbm 0.483, margin_ridge 0.413 · ensemble 0.461
  starters: KIA Tigers: 올러 (rest 6 d) · LG Twins: 톨허스트 (rest 6 d)
  home_field: LG Twins at home
  rest: team rest days: LG Twins 0, KIA Tigers 0; games in last 7 days: LG Twins 5, KIA Tigers 5
  park: park run factor vs league -0.39 runs/game
  ties: KBO regular-season games can end tied at the extra-inning limit; P(win) is for a decided game and a tie voids the pick

already logged as prediction 3265081c500149c7af7503513465c733 (snapshot sha256 c82209810caeebd7…)
```

```
Lotte Giants @ KT Wiz
[BLIND pre-game prediction · model v1.0 · data cutoff 2026-10-04T12:38:50Z · first pitch 2026-10-05T05:00:00Z]

Prediction: KT Wiz
Win Probability: Lotte Giants 29% / KT Wiz 71%
Projected Score: Lotte Giants 4.3–6.4 KT Wiz
Projected Margin: 2.2
Upset Probability: 29%
Confidence: 8.7/10

Key Factors:

1. Lineup & defence: opponent-adjusted runs scored/allowed per game (ex-starter) — KT Wiz +0.39/-0.39, Lotte Giants -0.30/+0.01 → edge KT Wiz (1.10 runs)
2. Form (last 10, runs vs Elo expectation): KT Wiz +0.24, Lotte Giants +1.89 → edge Lotte Giants (0.41 runs)
3. Home field: KT Wiz (league home edge -0.28 runs/game) → edge Lotte Giants (0.28 runs)

Main Risk: Lotte Giants's best counter: recent form (0.41 runs); single-game variance is large in baseball (run-margin sigma ≈ 4.9)

Model drivers (gradient-boosting SHAP, log-odds toward):
  - Elo rating edge (incl. home field): +0.27 → KT Wiz
  - Opponent-adjusted run model (incl. starters): +0.19 → KT Wiz
  - Away starter rest: +0.18 → KT Wiz
  - Home games played: +0.13 → KT Wiz
  - Starting pitcher run-prevention rating: +0.12 → KT Wiz

Member P(KT Wiz win): elo 0.618, scoring 0.550, logistic 0.680, gbm 0.747, margin_ridge 0.620 · ensemble 0.710
  starters: Lotte Giants: 김진욱 (rest 10 d) · KT Wiz: 로건 (rest 6 d)
  home_field: KT Wiz at home
  rest: team rest days: KT Wiz 0, Lotte Giants 0; games in last 7 days: KT Wiz 5, Lotte Giants 4
  park: park run factor vs league +0.16 runs/game
  ties: KBO regular-season games can end tied at the extra-inning limit; P(win) is for a decided game and a tie voids the pick

already logged as prediction 7e5de86024f84d00a704f562a8afeea4 (snapshot sha256 50b33aa1cf8f0cc1…)
```

```
Doosan Bears @ Samsung Lions
[BLIND pre-game prediction · model v1.0 · data cutoff 2026-10-04T12:38:50Z · first pitch 2026-10-05T05:00:00Z]

Prediction: Samsung Lions
Win Probability: Doosan Bears 36% / Samsung Lions 64%
Projected Score: Doosan Bears 4.3–5.6 Samsung Lions
Projected Margin: 1.2
Upset Probability: 36%
Confidence: 6.2/10

Key Factors:

1. Lineup & defence: opponent-adjusted runs scored/allowed per game (ex-starter) — Samsung Lions +0.50/-0.35, Doosan Bears -0.32/-0.61 → edge Samsung Lions (0.56 runs)
2. Starting pitching: 잭로그 (Doosan Bears) vs 후라도 (Samsung Lions) — run-prevention rating -0.24 vs -0.76 runs/game (lower is better) → edge Samsung Lions (0.52 runs)
3. Home field: Samsung Lions (league home edge -0.28 runs/game) → edge Doosan Bears (0.28 runs)

Main Risk: Doosan Bears's best counter: home field (0.28 runs); single-game variance is large in baseball (run-margin sigma ≈ 4.9)

Model drivers (gradient-boosting SHAP, log-odds toward):
  - Elo rating edge (incl. home field): +0.22 → Samsung Lions
  - Home games played: +0.20 → Samsung Lions
  - Opponent-adjusted run model (incl. starters): +0.17 → Samsung Lions
  - Starting pitcher run-prevention rating: +0.09 → Samsung Lions
  - Away starter rest: -0.08 → Doosan Bears

Member P(Samsung Lions win): elo 0.552, scoring 0.542, logistic 0.596, gbm 0.683, margin_ridge 0.550 · ensemble 0.638
  starters: Doosan Bears: 잭로그 (rest 5 d) · Samsung Lions: 후라도 (rest 5 d)
  home_field: Samsung Lions at home
  rest: team rest days: Samsung Lions 0, Doosan Bears 0; games in last 7 days: Samsung Lions 5, Doosan Bears 5
  park: park run factor vs league +0.40 runs/game
  ties: KBO regular-season games can end tied at the extra-inning limit; P(win) is for a decided game and a tie voids the pick

already logged as prediction 85b064eb15374fd8b3110d2053f686d5 (snapshot sha256 6c0cae6cef004d14…)
```

```
SSG Landers @ NC Dinos
[BLIND pre-game prediction · model v1.0 · data cutoff 2026-10-04T12:38:50Z · first pitch 2026-10-05T05:00:00Z]

Prediction: SSG Landers
Win Probability: SSG Landers 58% / NC Dinos 42%
Projected Score: SSG Landers 5.5–4.4 NC Dinos
Projected Margin: 1.1
Upset Probability: 42%
Confidence: 3.8/10

Key Factors:

1. Starting pitching: 아빌라 (SSG Landers) vs 토다 (NC Dinos) — run-prevention rating -1.10 vs -0.20 runs/game (lower is better) → edge SSG Landers (0.91 runs)
2. Home field: NC Dinos (league home edge -0.28 runs/game) → edge SSG Landers (0.28 runs)
3. Form (last 10, runs vs Elo expectation): NC Dinos +0.76, SSG Landers +0.25 → edge NC Dinos (0.13 runs)

Main Risk: NC Dinos's best counter: recent form (0.13 runs); single-game variance is large in baseball (run-margin sigma ≈ 4.9)

Model drivers (gradient-boosting SHAP, log-odds toward):
  - Starting pitcher run-prevention rating: -0.18 → SSG Landers
  - Elo rating edge (incl. home field): -0.14 → SSG Landers
  - Home games played: +0.13 → NC Dinos
  - Home starter rest: -0.11 → SSG Landers
  - Team run differential (ex-starter, opp-adjusted): -0.09 → SSG Landers

Member P(NC Dinos win): elo 0.496, scoring 0.431, logistic 0.387, gbm 0.437, margin_ridge 0.389 · ensemble 0.422
  starters: SSG Landers: 아빌라 (rest 6 d) · NC Dinos: 토다 (rest 9 d)
  home_field: NC Dinos at home
  rest: team rest days: NC Dinos 0, SSG Landers 0; games in last 7 days: NC Dinos 5, SSG Landers 5
  park: park run factor vs league +0.37 runs/game
  ties: KBO regular-season games can end tied at the extra-inning limit; P(win) is for a decided game and a tie voids the pick

already logged as prediction b1a498ebae764e848b17f54ec8b6fb2c (snapshot sha256 c88ecf2bd6a8cf7a…)
```

```
Kiwoom Heroes @ Hanwha Eagles
[BLIND pre-game prediction · model v1.0 · data cutoff 2026-10-04T12:38:50Z · first pitch 2026-10-05T05:00:00Z]

Prediction: Hanwha Eagles
Win Probability: Kiwoom Heroes 44% / Hanwha Eagles 56%
Projected Score: Kiwoom Heroes 5.6–6.1 Hanwha Eagles
Projected Margin: 0.5
Upset Probability: 44%
Confidence: 3.2/10

Key Factors:

1. Lineup & defence: opponent-adjusted runs scored/allowed per game (ex-starter) — Hanwha Eagles +0.59/+0.57, Kiwoom Heroes -0.98/+0.52 → edge Hanwha Eagles (1.52 runs)
2. Form (last 10, runs vs Elo expectation): Hanwha Eagles -2.79, Kiwoom Heroes +2.03 → edge Kiwoom Heroes (1.21 runs)
3. Home field: Hanwha Eagles (league home edge -0.28 runs/game) → edge Kiwoom Heroes (0.28 runs)

Main Risk: Kiwoom Heroes's best counter: recent form (1.21 runs); single-game variance is large in baseball (run-margin sigma ≈ 4.9)

Model drivers (gradient-boosting SHAP, log-odds toward):
  - Run environment: -0.18 → Kiwoom Heroes
  - Elo rating edge (incl. home field): +0.17 → Hanwha Eagles
  - Home games played: +0.16 → Hanwha Eagles
  - Opponent-adjusted run model (incl. starters): +0.12 → Hanwha Eagles
  - Starting pitcher run-prevention rating: +0.09 → Hanwha Eagles

Member P(Hanwha Eagles win): elo 0.561, scoring 0.567, logistic 0.510, gbm 0.603, margin_ridge 0.503 · ensemble 0.563
  starters: Kiwoom Heroes: 전준표 (rest 6 d) · Hanwha Eagles: 강건우 (rest 10 d)
  home_field: Hanwha Eagles at home
  rest: team rest days: Hanwha Eagles 0, Kiwoom Heroes 0; games in last 7 days: Hanwha Eagles 5, Kiwoom Heroes 4
  park: park run factor vs league +0.89 runs/game
  ties: KBO regular-season games can end tied at the extra-inning limit; P(win) is for a decided game and a tie voids the pick

already logged as prediction b913684c11ab4e49989059f884769e8a (snapshot sha256 11af5f0933200da5…)
```

## Summary

| Game | First pitch (UTC) | Pick | Win % | Projected score | Upset % | Confidence | Flags |
|---|---|---|---|---|---|---|---|
| KIA Tigers @ LG Twins | 10-05 05:00 | KIA Tigers | 54% | KIA Tigers 4.9–4.2 LG Twins | 46% | 2.1 |  |
| Lotte Giants @ KT Wiz | 10-05 05:00 | KT Wiz | 71% | Lotte Giants 4.3–6.4 KT Wiz | 29% | 8.7 |  |
| Doosan Bears @ Samsung Lions | 10-05 05:00 | Samsung Lions | 64% | Doosan Bears 4.3–5.6 Samsung Lions | 36% | 6.2 |  |
| SSG Landers @ NC Dinos | 10-05 05:00 | SSG Landers | 58% | SSG Landers 5.5–4.4 NC Dinos | 42% | 3.8 |  |
| Kiwoom Heroes @ Hanwha Eagles | 10-05 05:00 | Hanwha Eagles | 56% | Kiwoom Heroes 5.6–6.1 Hanwha Eagles | 44% | 3.2 |  |

**Most confident:** KT Wiz (71%), Samsung Lions (64%), SSG Landers (58%), Hanwha Eagles (56%)

**Closest:** KIA Tigers @ LG Twins (54%), Kiwoom Heroes @ Hanwha Eagles (56%), SSG Landers @ NC Dinos (58%), Doosan Bears @ Samsung Lions (64%)

**Upset candidates (underdog ≥ 35%):** LG Twins 46%, Kiwoom Heroes 44%, NC Dinos 42%, Doosan Bears 36%

**Highest model disagreement:** Lotte Giants @ KT Wiz, Doosan Bears @ Samsung Lions, SSG Landers @ NC Dinos