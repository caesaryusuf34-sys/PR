# NPB — predictions for 2026-10-05

- Model **v1.0** · predictions frozen 2026-10-04T12:38:54Z (before every kickoff listed) · all inputs are pre-game; results/injuries after the cutoff never enter
- Walk-forward validation (1500 out-of-sample games): accuracy 56.4%, log loss 0.678, Brier 0.242, margin MAE 3.0
- Betting lines are not model inputs. Starting pitchers are the announced probables as captured at the freeze (TBD = the team's recent rotation on average). A tied game (NPB/KBO/CPBL) voids the pick.

## Games

```
Fukuoka SoftBank Hawks @ Tohoku Rakuten Golden Eagles
[BLIND pre-game prediction · model v1.0 · data cutoff 2026-10-04T12:38:54Z · first pitch 2026-10-05T09:00:00Z]

Prediction: Fukuoka SoftBank Hawks
Win Probability: Fukuoka SoftBank Hawks 63% / Tohoku Rakuten Golden Eagles 37%
Projected Score: Fukuoka SoftBank Hawks 4.7–3.2 Tohoku Rakuten Golden Eagles
Projected Margin: 1.4
Upset Probability: 37%
Confidence: 6.2/10

Key Factors:

1. Lineup & defence: opponent-adjusted runs scored/allowed per game (ex-starter) — Tohoku Rakuten Golden Eagles -0.21/+0.04, Fukuoka SoftBank Hawks +1.15/-0.39 → edge Fukuoka SoftBank Hawks (1.79 runs)
2. Starting pitching: 上茶谷 大河 (Fukuoka SoftBank Hawks) vs 前田 健太 (Tohoku Rakuten Golden Eagles) — FIP-type vs league +0.43 vs -0.32 per 9, K-BB/9 +1.12 vs +0.18, starts this season 9 vs 14; run-prevention rating -0.00 vs -0.08 runs/game (lower is better) → edge Tohoku Rakuten Golden Eagles (0.27 runs)
3. Form (last 10, runs vs Elo expectation): Tohoku Rakuten Golden Eagles +0.93, Fukuoka SoftBank Hawks +1.79 → edge Fukuoka SoftBank Hawks (0.22 runs)

Main Risk: Tohoku Rakuten Golden Eagles's best counter: starting pitching (0.27 runs); Tohoku Rakuten Golden Eagles bullpen heavily used in the last 3 days (8.7 relief innings); single-game variance is large in baseball (run-margin sigma ≈ 4.0)

Model drivers (gradient-boosting SHAP, log-odds toward):
  - Elo rating edge (incl. home field): -0.17 → Fukuoka SoftBank Hawks
  - Team run differential (ex-starter, opp-adjusted): -0.09 → Fukuoka SoftBank Hawks
  - Run environment: -0.06 → Fukuoka SoftBank Hawks
  - Recent form vs expectation (last 10): -0.05 → Fukuoka SoftBank Hawks
  - Bullpen FIP-type rate: -0.04 → Fukuoka SoftBank Hawks

Member P(Tohoku Rakuten Golden Eagles win): elo 0.368, scoring 0.408, logistic 0.359, gbm 0.439, margin_ridge 0.344 · ensemble 0.368
  starters: Fukuoka SoftBank Hawks: 上茶谷 大河 (rest 10 d) · Tohoku Rakuten Golden Eagles: 前田 健太 (rest 10 d)
  home_field: Tohoku Rakuten Golden Eagles at home
  rest: team rest days: Tohoku Rakuten Golden Eagles 2, Fukuoka SoftBank Hawks 2; games in last 7 days: Tohoku Rakuten Golden Eagles 5, Fukuoka SoftBank Hawks 3
  park: park run factor vs league +0.06 runs/game
  ties: NPB games can end tied after 12 innings; P(win) is for a decided game and a tie voids the pick

already logged as prediction 1bd0331709bc492c96b033e8bab123ef (snapshot sha256 102617acf25826d9…)
```

```
Saitama Seibu Lions @ Chiba Lotte Marines
[BLIND pre-game prediction · model v1.0 · data cutoff 2026-10-04T12:38:54Z · first pitch 2026-10-05T09:00:00Z]

Prediction: Saitama Seibu Lions
Win Probability: Saitama Seibu Lions 52% / Chiba Lotte Marines 48%
Projected Score: Saitama Seibu Lions 3.9–3.6 Chiba Lotte Marines
Projected Margin: 0.3
Upset Probability: 48%
Confidence: 1.0/10

Key Factors:

1. Lineup & defence: opponent-adjusted runs scored/allowed per game (ex-starter) — Chiba Lotte Marines -0.07/+0.22, Saitama Seibu Lions -0.21/-0.42 → edge Saitama Seibu Lions (0.50 runs)
2. Form (last 10, runs vs Elo expectation): Chiba Lotte Marines +0.49, Saitama Seibu Lions -1.35 → edge Chiba Lotte Marines (0.46 runs)
3. Home field: Chiba Lotte Marines (league home edge +0.17 runs/game) → edge Chiba Lotte Marines (0.17 runs)

Main Risk: Chiba Lotte Marines's best counter: recent form (0.46 runs); Chiba Lotte Marines bullpen heavily used in the last 3 days (11.7 relief innings); single-game variance is large in baseball (run-margin sigma ≈ 4.0)

Model drivers (gradient-boosting SHAP, log-odds toward):
  - Elo rating edge (incl. home field): -0.17 → Saitama Seibu Lions
  - Starter FIP-type rate: -0.06 → Saitama Seibu Lions
  - Bullpen workload, last 3 days: -0.05 → Saitama Seibu Lions
  - Starter innings per start: -0.05 → Saitama Seibu Lions
  - Home starter rest: +0.04 → Chiba Lotte Marines

Member P(Chiba Lotte Marines win): elo 0.483, scoring 0.479, logistic 0.426, gbm 0.444, margin_ridge 0.445 · ensemble 0.482
  starters: Saitama Seibu Lions: 髙橋 光成 (rest 10 d) · Chiba Lotte Marines: 西野 勇士 (rest 10 d)
  home_field: Chiba Lotte Marines at home
  rest: team rest days: Chiba Lotte Marines 2, Saitama Seibu Lions 4; games in last 7 days: Chiba Lotte Marines 6, Saitama Seibu Lions 2
  park: park run factor vs league +0.33 runs/game
  ties: NPB games can end tied after 12 innings; P(win) is for a decided game and a tie voids the pick

already logged as prediction 10d6bcc89b2749168bd8525feecfa024 (snapshot sha256 3cb82c6e4976893a…)
```

## Summary

| Game | First pitch (UTC) | Pick | Win % | Projected score | Upset % | Confidence | Flags |
|---|---|---|---|---|---|---|---|
| Fukuoka SoftBank Hawks @ Tohoku Rakuten Golden Eagles | 10-05 09:00 | Fukuoka SoftBank Hawks | 63% | Fukuoka SoftBank Hawks 4.7–3.2 Tohoku Rakuten Golden Eagles | 37% | 6.2 |  |
| Saitama Seibu Lions @ Chiba Lotte Marines | 10-05 09:00 | Saitama Seibu Lions | 52% | Saitama Seibu Lions 3.9–3.6 Chiba Lotte Marines | 48% | 1.0 |  |

**Most confident:** Fukuoka SoftBank Hawks (63%), Saitama Seibu Lions (52%)

**Closest:** Saitama Seibu Lions @ Chiba Lotte Marines (52%), Fukuoka SoftBank Hawks @ Tohoku Rakuten Golden Eagles (63%)

**Upset candidates (underdog ≥ 35%):** Chiba Lotte Marines 48%, Tohoku Rakuten Golden Eagles 37%

**Highest model disagreement:** Fukuoka SoftBank Hawks @ Tohoku Rakuten Golden Eagles, Saitama Seibu Lions @ Chiba Lotte Marines