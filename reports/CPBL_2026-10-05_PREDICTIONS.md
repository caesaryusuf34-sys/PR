# CPBL — predictions for 2026-10-05

- Model **v1.0** · predictions frozen 2026-10-04T12:38:59Z (before every kickoff listed) · all inputs are pre-game; results/injuries after the cutoff never enter
- Walk-forward validation (756 out-of-sample games): accuracy 53.7%, log loss 0.687, Brier 0.247, margin MAE 3.3
- Betting lines are not model inputs. Starting pitchers are the announced probables as captured at the freeze (TBD = the team's recent rotation on average). A tied game (NPB/KBO/CPBL) voids the pick.

## Games

```
Uni-President Lions @ CTBC Brothers
[BLIND pre-game prediction · model v1.0 · data cutoff 2026-10-04T12:38:59Z · first pitch 2026-10-05T10:35:00Z]

Prediction: CTBC Brothers
Win Probability: Uni-President Lions 49% / CTBC Brothers 51%
Projected Score: Uni-President Lions 4.2–4.3 CTBC Brothers
Projected Margin: 0.0
Upset Probability: 49%
Confidence: 1.0/10

Key Factors:

1. Form (last 10, runs vs Elo expectation): CTBC Brothers +1.46, Uni-President Lions -0.50 → edge CTBC Brothers (0.49 runs)
2. Lineup & defence: opponent-adjusted runs scored/allowed per game (ex-starter) — CTBC Brothers +0.57/+0.56, Uni-President Lions -0.01/-0.33 → edge Uni-President Lions (0.32 runs)
3. Home field: CTBC Brothers (league home edge -0.25 runs/game) → edge Uni-President Lions (0.25 runs)

Main Risk: starter not announced for CTBC Brothers, Uni-President Lions — rotation average used; ensemble members disagree on the winner; Uni-President Lions's best counter: lineup & defence (0.32 runs)

Model drivers (gradient-boosting SHAP, log-odds toward):
  - Starter innings per start: +0.18 → CTBC Brothers
  - Bullpen FIP-type rate: +0.15 → CTBC Brothers
  - Recent form vs expectation (last 10): -0.13 → Uni-President Lions
  - Starter FIP-type rate: +0.11 → CTBC Brothers
  - Starting pitcher run-prevention rating: +0.11 → CTBC Brothers

Member P(CTBC Brothers win): elo 0.527, scoring 0.475, logistic 0.531, gbm 0.588, margin_ridge 0.507 · ensemble 0.506
  starters: Uni-President Lions: TBD (rotation average) (rest 5 d) · CTBC Brothers: TBD (rotation average) (rest 5 d)
  home_field: CTBC Brothers at home
  rest: team rest days: CTBC Brothers 2, Uni-President Lions 2; games in last 7 days: CTBC Brothers 2, Uni-President Lions 3
  park: park run factor vs league +0.54 runs/game
  ties: CPBL games can end tied after 12 innings; P(win) is for a decided game and a tie voids the pick

already logged as prediction a8f18d5113de4c9a9e796eb4051c1474 (snapshot sha256 bd405aa27b1a0421…)
```

```
Fubon Guardians @ Rakuten Monkeys
[BLIND pre-game prediction · model v1.0 · data cutoff 2026-10-04T12:38:59Z · first pitch 2026-10-05T10:35:00Z]

Prediction: Fubon Guardians
Win Probability: Fubon Guardians 51% / Rakuten Monkeys 49%
Projected Score: Fubon Guardians 3.7–3.7 Rakuten Monkeys
Projected Margin: 0.1
Upset Probability: 49%
Confidence: 1.0/10

Key Factors:

1. Home field: Rakuten Monkeys (league home edge -0.25 runs/game) → edge Fubon Guardians (0.25 runs)
2. Form (last 10, runs vs Elo expectation): Rakuten Monkeys +0.52, Fubon Guardians -0.26 → edge Rakuten Monkeys (0.19 runs)
3. Lineup & defence: opponent-adjusted runs scored/allowed per game (ex-starter) — Rakuten Monkeys -0.49/-0.22, Fubon Guardians -0.13/+0.08 → edge Fubon Guardians (0.06 runs)

Main Risk: starter not announced for Rakuten Monkeys, Fubon Guardians — rotation average used; ensemble members disagree on the winner; Rakuten Monkeys's best counter: recent form (0.19 runs)

Model drivers (gradient-boosting SHAP, log-odds toward):
  - Bullpen FIP-type rate: -0.13 → Fubon Guardians
  - Starter runs allowed per 9: -0.10 → Fubon Guardians
  - Starting pitcher run-prevention rating: -0.09 → Fubon Guardians
  - Starter FIP-type rate: +0.09 → Rakuten Monkeys
  - Recent form vs expectation (last 10): +0.09 → Rakuten Monkeys

Member P(Rakuten Monkeys win): elo 0.541, scoring 0.475, logistic 0.515, gbm 0.520, margin_ridge 0.493 · ensemble 0.487
  starters: Fubon Guardians: TBD (rotation average) (rest 5 d) · Rakuten Monkeys: TBD (rotation average) (rest 5 d)
  home_field: Rakuten Monkeys at home
  rest: team rest days: Rakuten Monkeys 2, Fubon Guardians 2; games in last 7 days: Rakuten Monkeys 4, Fubon Guardians 3
  park: park run factor vs league +0.22 runs/game
  ties: CPBL games can end tied after 12 innings; P(win) is for a decided game and a tie voids the pick

already logged as prediction cce934a452d34e50aceb4e3c040d1104 (snapshot sha256 34dcd0a1a7feee65…)
```

## Summary

| Game | First pitch (UTC) | Pick | Win % | Projected score | Upset % | Confidence | Flags |
|---|---|---|---|---|---|---|---|
| Uni-President Lions @ CTBC Brothers | 10-05 10:35 | CTBC Brothers | 51% | Uni-President Lions 4.2–4.3 CTBC Brothers | 49% | 1.0 | starter TBD |
| Fubon Guardians @ Rakuten Monkeys | 10-05 10:35 | Fubon Guardians | 51% | Fubon Guardians 3.7–3.7 Rakuten Monkeys | 49% | 1.0 | starter TBD |

**Most confident:** Fubon Guardians (51%), CTBC Brothers (51%)

**Closest:** Uni-President Lions @ CTBC Brothers (51%), Fubon Guardians @ Rakuten Monkeys (51%)

**Upset candidates (underdog ≥ 35%):** Uni-President Lions 49%, Rakuten Monkeys 49%

**Highest model disagreement:** Uni-President Lions @ CTBC Brothers, Fubon Guardians @ Rakuten Monkeys