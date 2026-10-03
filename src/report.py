"""Build deliverables from the FROZEN slate snapshot (output/slate_features.pkl). No model is
re-fit here and no post-freeze data is fetched: this only formats the frozen numbers."""
import os, json
import numpy as np, pandas as pd
from features import ROOT, CUTOFF, load_games, division_for

EFF = ["epa_pp", "sr", "ypp", "expl", "rush_epa", "pass_epa", "rush_sr", "pass_sr", "sack_rate", "to_rate",
       "pts_per_drive", "ppo", "start_fp"]
PCOLS = ["p_elo", "p_pts", "p_lr", "p_gbm", "p_reg"]
PNAMES = {"p_elo": "Elo", "p_pts": "Adj. scoring model", "p_lr": "Logistic reg.", "p_gbm": "Gradient boosting",
          "p_reg": "Point-diff regression"}


def team_stats(slate, R, m, g, seq, divs, elo_final):
    tids = pd.unique(pd.concat([slate.home_id, slate.away_id]))
    rt = R[(R.season == 2026) & (R.wk == "today")]
    piv = rt.pivot_table(index="tid", columns="metric", values=["o", "d"]); piv.columns = [f"adj_{b}_{'off' if a=='o' else 'def'}" for a, b in piv.columns]
    net = (rt[rt.metric == "pts"].set_index("tid").o - rt[rt.metric == "pts"].set_index("tid").d)
    m26 = m[(m.season == 2026)]
    g26 = g[g.season == 2026]
    rows = []
    for t in tids:
        tm = m26[m26.tid == t]
        name = slate.loc[slate.home_id == t, "home"].tolist() + slate.loc[slate.away_id == t, "away"].tolist()
        gg = g26[(g26.home_id == t) | (g26.away_id == t)].sort_values("date")
        is_home = gg.home_id == t
        pf = np.where(is_home, gg.home_score, gg.away_score); pa = np.where(is_home, gg.away_score, gg.home_score)
        opp = np.where(is_home, gg.away_id, gg.home_id)
        marg = pf - pa
        home_m = marg[(is_home & ~gg.neutral.astype(bool)).values]; away_m = marg[(~is_home & ~gg.neutral.astype(bool)).values]
        opp_tm = m26.set_index(["event_id", "tid"])
        # opponent box (for turnover margin / sacks allowed)
        to_for, to_against = [], []
        for eid, o in zip(gg.event_id, opp):
            if (eid, o) in opp_tm.index:
                to_for.append(opp_tm.loc[(eid, o)].turnovers)
        qb = tm.dropna(subset=["qb_id"])
        if len(qb):
            lastqb = qb.sort_values("date").iloc[-1]
            qq = qb[qb.qb_id == lastqb.qb_id]
            qb_att, qb_cmp, qb_yds, qb_td, qb_int = qq.qb_att.sum(), qq.qb_cmp.sum(), qq.qb_yds.sum(), qq.qb_td.sum(), qq.qb_int.sum()
            qb_name = lastqb.qb_name; qb_starts = len(qq); share = qb_att / max(qb.qb_att.sum(), 1)
            starters = qb.sort_values("date").qb_name.tolist()
            qb_change = len(starters) >= 2 and starters[-1] != starters[-2]
        else:
            qb_att = qb_cmp = qb_yds = qb_td = qb_int = np.nan; qb_name = "Data unavailable"; qb_starts = 0; share = np.nan; qb_change = False; starters = []
        s_t = seq[(seq.tid == t) & (seq.season == 2026)].sort_values("date")
        res = dict(
            tid=t, team=name[0] if name else t, division_2026=division_for(divs, 2026, t),
            games_played=len(gg), wins=int((marg > 0).sum()), losses=int((marg < 0).sum()),
            ppg=pf.mean() if len(gg) else np.nan, opp_ppg=pa.mean() if len(gg) else np.nan,
            avg_margin=marg.mean() if len(gg) else np.nan,
            home_avg_margin=home_m.mean() if len(home_m) else np.nan, n_home=len(home_m),
            away_avg_margin=away_m.mean() if len(away_m) else np.nan, n_away=len(away_m),
            last3_margins=",".join(str(int(x)) for x in marg[-3:]), last5_margins=",".join(str(int(x)) for x in marg[-5:]),
            last3_vs_expect=s_t.resid.tail(3).mean() if len(s_t) else np.nan,
            sos_opp_net_rating=np.mean([net.get(o, np.nan) for o in opp]) if len(opp) else np.nan,
            sos_opp_elo=np.mean([elo_final.get(o, np.nan) for o in opp]) if len(opp) else np.nan,
            elo=elo_final.get(t, np.nan), net_pts_rating=net.get(t, np.nan),
            raw_ypp=tm.ypp.mean(), raw_epa_pp=tm.epa_pp.mean(), raw_success_rate=tm.sr.mean(), raw_explosive_rate=tm.expl.mean(),
            third_down_pct=tm.third_conv.sum() / tm.third_att.sum() if tm.third_att.sum() > 0 else np.nan,
            rz_td_rate=(tm.rz_td_rate * tm.rz_trips).sum() / tm.rz_trips.sum() if tm.rz_trips.sum() > 0 else np.nan,
            pts_per_scoring_opp=tm.ppo.mean(),
            turnovers_lost_pg=tm.turnovers.mean(),
            takeaways_pg=np.nanmean(to_for) if len(to_for) else np.nan,
            sacks_pg=tm["def_SACKS"].mean(), tfl_pg=tm["def_TFL"].mean(),
            fg_made=tm.fg_made.sum(), fg_att=tm.fg_att.sum(), avg_start_field_pos=tm.start_fp.mean(),
            penalty_yds_pg=tm.pen_yds.mean(),
            qb_recent_starter=qb_name, qb_starts=qb_starts, qb_att=qb_att,
            qb_cmp_pct=qb_cmp / qb_att if qb_att and qb_att > 0 else np.nan,
            qb_ypa=qb_yds / qb_att if qb_att and qb_att > 0 else np.nan, qb_td=qb_td, qb_int=qb_int,
            qb_share_of_team_att=share, qb_changed_last_game=qb_change, qb_starter_sequence=" > ".join(map(str, starters)),
            injuries="Data unavailable (ESPN college injury feed returned no entries; no depth chart published)",
        )
        res["turnover_margin_pg"] = res["takeaways_pg"] - res["turnovers_lost_pg"] if len(to_for) else np.nan
        rows.append(res)
    ts = pd.DataFrame(rows).set_index("tid").join(piv)
    return ts


def fmt_pct(p):
    return f"{100*p:.0f}%"


def analyze(r, ts, sd, hfa_pts):
    """Return (key factors list, main risk, matchup dict) from frozen ratings."""
    H, A = r.home, r.away
    th, ta = ts.loc[r.home_id], ts.loc[r.away_id]
    fav_home = r.p_ens >= 0.5
    fav, dog = (H, A) if fav_home else (A, H)
    sgn = 1 if fav_home else -1
    edges = []  # (value in SD units from favourite's perspective, text)
    def unit(off_t, def_t, k):
        return (off_t[f"adj_{k}_off"] + def_t[f"adj_{k}_def"])
    # net efficiency matchups: positive = home edge
    for k, label, good_high in [("epa_pp", "Overall offense vs defense (opp-adj EPA/play)", True),
                                ("rush_epa", "Run game vs run defense (opp-adj rush EPA/play)", True),
                                ("pass_epa", "Pass game vs pass defense (opp-adj pass EPA/play)", True),
                                ("expl", "Explosive-play rate matchup", True),
                                ("sr", "Success-rate matchup", True),
                                ("ppo", "Finishing drives (pts per scoring opportunity)", True),
                                ("start_fp", "Field position / special teams (start field pos.)", True)]:
        hv, av = unit(th, ta, k), unit(ta, th, k)
        if np.isnan(hv) or np.isnan(av): continue
        edges.append(((hv - av) / sd[k] * sgn, label, k, hv, av))
    # pressure: home OL sack rate allowed + away pass rush generated ... (higher = worse for that offense)
    for k, label in [("sack_rate", "OL protection vs pass rush (sack rate)"), ("to_rate", "Turnover tendencies (giveaways vs takeaways)")]:
        hv, av = unit(th, ta, k), unit(ta, th, k)
        if np.isnan(hv) or np.isnan(av): continue
        edges.append(((av - hv) / sd[k] * sgn, label, k, hv, av))
    edges.sort(key=lambda e: -abs(e[0]))
    factors = []
    for v, label, k, hv, av in edges:
        who = fav if v > 0 else dog
        unitlbl = {"sack_rate": "pp sack rate", "to_rate": "pp turnover rate", "start_fp": "yds", "ppo": "pts", "sr": "pp",
                   "expl": "pp"}.get(k, "EPA/play")
        scale = 100 if k in ("sack_rate", "to_rate", "sr", "expl") else 1
        txt = (f"{label}: edge **{who}** ({abs(v):.2f} SD); vs-average expectation — {H} offense vs {A} defense "
               f"{hv*scale:+.2f} {unitlbl}, {A} offense vs {H} defense {av*scale:+.2f} {unitlbl}")
        factors.append((abs(v), txt, v, label))
    # QB
    qb_txt = (f"QB: {H} {th.qb_recent_starter} ({th.qb_ypa:.1f} YPA, {int(th.qb_td or 0)} TD/{int(th.qb_int or 0)} INT"
              f"{', NEW STARTER last game' if th.qb_changed_last_game else ''}) vs {A} {ta.qb_recent_starter} "
              f"({ta.qb_ypa:.1f} YPA, {int(ta.qb_td or 0)} TD/{int(ta.qb_int or 0)} INT{', NEW STARTER last game' if ta.qb_changed_last_game else ''})")
    # home field / form / SOS
    ctx = []
    if not r.neutral:
        ctx.append(f"Home field: {H} (Elo home bonus 45 rating pts; scoring-model home coefficient {hfa_pts:.1f} pts on 2026 data)")
    else:
        ctx.append("Neutral site")
    fd = r.form3_h - r.form3_a
    ctx.append(f"Recent form (last-3 margin vs expectation): {H} {r.form3_h:+.1f}, {A} {r.form3_a:+.1f}")
    ctx.append(f"Strength of schedule (avg opp net rating): {H} {th.sos_opp_net_rating:+.1f}, {A} {ta.sos_opp_net_rating:+.1f}")
    # weather
    wx = []
    if r.indoor is True:
        wx.append("Indoor venue")
    else:
        cond = str(r.wx_condition) if pd.notna(r.wx_condition) else "Data unavailable"
        wx.append(f"Forecast: {cond}, {r.wx_temp_f if pd.notna(r.wx_temp_f) else 'n/a'}°F, gusts {r.wx_gust_mph if pd.notna(r.wx_gust_mph) else 'n/a'} mph, precip {r.wx_precip_pct if pd.notna(r.wx_precip_pct) else 'n/a'}%")
    wx_flag = (pd.notna(r.wx_gust_mph) and r.wx_gust_mph >= 20) or ("storm" in str(r.wx_condition).lower()) or ("rain" in str(r.wx_condition).lower())
    # main risk
    dog_edges = [f for f in factors if f[2] < 0]
    risks = []
    if r.models_split: risks.append("models disagree on the winner")
    if dog_edges: risks.append(f"{dog}'s best counter is {dog_edges[0][3].split(' (')[0]} ({abs(dog_edges[0][2]):.2f} SD edge)")
    if th.qb_changed_last_game or ta.qb_changed_last_game:
        who = ", ".join([x for x, t in ((H, th), (A, ta)) if t.qb_changed_last_game])
        risks.append(f"QB change last game ({who}) — availability unconfirmed")
    if min(th.games_played, ta.games_played) <= 2 or "OTHER" in (th.division_2026, ta.division_2026):
        risks.append("thin 2026 sample for at least one team")
    if wx_flag: risks.append("weather (wind/storms) adds variance")
    if not risks: risks.append("single-game variance (σ ≈ 16 pts on the margin)")
    return factors, qb_txt, ctx, wx, wx_flag, "; ".join(risks[:3])


def main():
    out = os.path.join(ROOT, "output")
    x = pd.read_pickle(os.path.join(out, "slate_features.pkl"))
    meta = json.load(open(os.path.join(out, "freeze_meta.json")))
    val = json.load(open(os.path.join(out, "validation.json")))
    g, divs = load_games()
    R = pd.read_csv(os.path.join(ROOT, "data", "weekly_ratings.csv.gz"), dtype={"tid": str, "wk": str})
    m = pd.read_csv(os.path.join(ROOT, "data", "team_game_metrics.csv.gz"), dtype={"event_id": str, "tid": str, "qb_id": str})
    m["date"] = pd.to_datetime(m.date, utc=True)
    seq = pd.read_csv(os.path.join(ROOT, "data", "team_sequence.csv.gz"), dtype={"event_id": str, "tid": str})
    elo_final = json.load(open(os.path.join(ROOT, "data", "elo_final.json")))["elo_final"]
    slate_all = pd.read_csv(os.path.join(ROOT, "data", "slate_all_today.csv"), dtype={"event_id": str, "home_id": str, "away_id": str})

    ts = team_stats(x, R, m, g, seq, divs, elo_final)
    rt = R[(R.season == 2026) & (R.wk == "today")]
    d1 = [t for t in rt.tid.unique() if division_for(divs, 2026, t) in ("FBS", "FCS")]
    sd = {}
    for k in EFF:
        rk = rt[(rt.metric == k) & rt.tid.isin(d1)]
        # SD of a matchup differential (o_h + d_a) - (o_a + d_h) between two random D1 teams
        sd[k] = float(np.sqrt(2 * (np.var(rk.o) + np.var(rk.d)))) or 1.0
    hfa = float(rt[rt.metric == "pts"].h.iloc[0]) * 2

    x = x.copy()
    x["models_split"] = (x[PCOLS].min(axis=1) < 0.5) & (x[PCOLS].max(axis=1) > 0.5)
    x["home_win_prob"] = x.p_ens; x["away_win_prob"] = 1 - x.p_ens
    x["pred_winner"] = np.where(x.p_ens >= 0.5, x.home, x.away)
    x["win_prob_pred"] = np.maximum(x.p_ens, 1 - x.p_ens)
    x["upset_prob"] = 1 - x.win_prob_pred
    x["proj_home"] = ((x.total + x.m_final) / 2).clip(lower=0)
    x["proj_away"] = ((x.total - x.m_final) / 2).clip(lower=0)
    # if the clipped side hits zero keep the projected margin
    neg = x.proj_away == 0; x.loc[neg, "proj_home"] = x.loc[neg, "m_final"]
    neg = x.proj_home == 0; x.loc[neg, "proj_away"] = -x.loc[neg, "m_final"]
    x["proj_margin"] = x.m_final.abs()
    thin = [min(ts.loc[h].games_played, ts.loc[a].games_played) <= 2 or "OTHER" in (ts.loc[h].division_2026, ts.loc[a].division_2026)
            for h, a in zip(x.home_id, x.away_id)]
    qbchg = [bool(ts.loc[h].qb_changed_last_game or ts.loc[a].qb_changed_last_game) for h, a in zip(x.home_id, x.away_id)]
    base = 10 * (2 * x.win_prob_pred - 1).clip(0, 1) ** 0.75
    conf = base - 10 * x.p_std - 0.75 * np.array(thin) - 0.5 * np.array(qbchg) - 1.0 * x.models_split
    x["confidence"] = conf.clip(1, 10).round(1)
    x["high_disagreement"] = (x.p_std >= 0.08) | x.models_split | (x.m_std >= 6)
    x["thin_sample"] = thin; x["qb_change_flag"] = qbchg

    # ---------------- CSV deliverables
    games_today = slate_all.copy()
    games_today["conference"] = np.where(games_today.away_conf == games_today.home_conf, games_today.home_conf,
                                         games_today.away_conf.fillna("?") + " vs " + games_today.home_conf.fillna("?"))
    games_today["predicted"] = games_today.event_id.isin(x.event_id)
    games_today["status_at_freeze"] = np.where(games_today.predicted, "Not started (predicted)",
                                               np.where(games_today.state == "pre", "Kicked off before freeze (excluded)",
                                                        np.where(games_today.state == "in", "In progress at collection (excluded)", "Final at collection (excluded)")))
    games_today["freeze_utc"] = meta["freeze_utc"]
    gcols = ["event_id", "kickoff_et", "date_utc", "away", "home", "venue", "venue_city", "venue_state", "indoor", "neutral",
             "away_conf", "home_conf", "conference", "conf_game", "away_record", "home_record", "away_rank", "home_rank",
             "status", "state", "status_at_freeze", "predicted", "wx_condition", "wx_temp_f", "wx_gust_mph", "wx_precip_pct",
             "broadcast", "market_spread_REFERENCE_ONLY", "market_total_REFERENCE_ONLY", "data_pulled_utc", "freeze_utc"]
    games_today[gcols].to_csv(os.path.join(ROOT, "games_today.csv"), index=False)
    ts.reset_index().round(4).to_csv(os.path.join(ROOT, "team_stats.csv"), index=False)
    fcols = ["event_id", "away", "home", "elo_h", "elo_a", "elo_diff", "pp_home", "pp_away", "pts_margin_pred", "pts_total_pred"] + \
        [f"mx_{k}" for k in EFF] + [f"{s}_{u}_{k}" for k in EFF for s in ("h", "a") for u in ("o", "d")] + \
        ["form3_h", "form3_a", "form_diff", "rest_h", "rest_a", "rest_diff", "qb_share_h", "qb_share_a", "qb_change_h", "qb_change_a",
         "n_prior_h", "n_prior_a", "neutral", "conf_game", "fbs_h", "fbs_a", "early"]
    x[fcols].round(4).to_csv(os.path.join(ROOT, "features.csv"), index=False)
    pcols = ["event_id", "kickoff_et", "away", "home", "pred_winner", "away_win_prob", "home_win_prob", "proj_away", "proj_home",
             "proj_margin", "total", "upset_prob", "confidence", "p_elo", "p_pts", "p_lr", "p_gbm", "p_reg", "p_ens",
             "m_elo", "m_pts", "m_reg", "m_gbm", "m_ens", "m_final", "p_std", "m_std", "models_split", "high_disagreement",
             "thin_sample", "qb_change_flag"]
    P = x[pcols].rename(columns={"total": "proj_total"}).copy()
    P["data_cutoff_utc"] = CUTOFF.isoformat(); P["prediction_freeze_utc"] = meta["freeze_utc"]
    P.round(4).to_csv(os.path.join(ROOT, "predictions.csv"), index=False)

    # ---------------- markdown report
    L = []
    L.append("# NCAA Football — Blind Pre-Match Predictions, Saturday 2026-10-03\n")
    L.append(f"- **Data cutoff (training/ratings):** completed games kicking off before {CUTOFF.isoformat()} (no slate-day results used)")
    L.append(f"- **Slate collected:** {slate_all.data_pulled_utc.iloc[0]} UTC; **Predictions frozen:** {meta['freeze_utc']} (git commit 82c7a54, pushed 20:35:40 UTC)")
    L.append(f"- **Games predicted:** {len(x)} (all FBS/FCS games not yet kicked off at freeze). Excluded: " +
             "; ".join(f"{s['away']} @ {s['home']} ({s['kickoff_et']}, kicked off before freeze)" for s in meta["skipped_started_before_freeze"]) +
             f"; plus {int((slate_all.state != 'pre').sum())} games already in progress/final at collection.")
    L.append("- **Sources:** ESPN public JSON APIs (scoreboard, game summary/box score/play-by-play, standings, venue + AccuWeather forecast embedded in ESPN game data). "
             "CollegeFootballData API (requires key → 401), Sports-Reference (403) and Open-Meteo (timeout) were not reachable and were **not** bypassed.")
    L.append("- **Betting odds:** not used as a model input. The DraftKings line is stored in `games_today.csv` only as a reference column (`*_REFERENCE_ONLY`).")
    L.append("- **Injuries:** *Data unavailable* — ESPN's college injury feed returned no entries and no depth charts are published. QB availability is proxied by box-score starter continuity (flagged when the starter changed last game).\n")

    wp = val["weights_prob"]; wm = val["weights_margin"]
    L.append("## Model & walk-forward validation\n")
    L.append("Ensemble members (trained on 2022→ seasons, features computed strictly from games before each game's week):")
    L.append("1. **Elo** (MOV-weighted, tuned K/HFA/season-regression on 2017–22)\n2. **Opponent-adjusted scoring model** (weekly ridge regression, off/def points with home term and preseason priors)\n"
             "3. **Logistic regression** and 4. **LightGBM gradient boosting** on 29 features (Elo, adjusted scoring, opponent-adjusted EPA/success/explosiveness/rush/pass/sack/turnover/drive/field-position matchups, form, rest, QB continuity, site, division)\n"
             "5. **Point-differential regression** (ridge) + a LightGBM margin regressor; totals from a separate ridge model.\n")
    L.append("Ensemble weights were fit on out-of-sample walk-forward predictions for 2023–2025 (train on seasons < S, test on S); 2026 (weeks 1–5, pre-today) is a pure holdout.\n")
    L.append("| Probability model | weight |  | Margin model | weight |\n|---|---|---|---|---|")
    for (a, b), (c, d) in zip(list(wp.items()) + [("", "")] * 0, list(wm.items()) + [("", "")]):
        L.append(f"| {PNAMES.get(a, a)} | {b:.3f} | | {c} | {d if d == '' else f'{d:.3f}'} |")
    L.append("\n| Season (OOS) | Games | Accuracy | Brier | Log loss | Margin MAE | Total MAE | Elo-only log loss |\n|---|---|---|---|---|---|---|---|")
    for S, r in val["validation"].items():
        e = r["p_ens"]
        L.append(f"| {S}{' (holdout)' if S == '2026' else ''} | {e['n']} | {e['accuracy']:.3f} | {e['brier']:.3f} | {e['log_loss']:.3f} | {r['m_ens']['margin_MAE']:.2f} | {r['total_MAE']:.2f} | {r['p_elo']['log_loss']:.3f} |")
    L.append("\nCalibration (ensemble, all OOS seasons 2023–2026):\n\n| Predicted bin | n | mean predicted | actual win rate |\n|---|---|---|---|")
    for c in val["calibration_2023_2026"]:
        L.append(f"| {c['bin']} | {c['n']} | {c['mean_pred']:.3f} | {c['actual']:.3f} |")
    L.append(f"\nEnsemble margin residual σ = {val['ensemble_margin_sigma']:.1f} pts. Note: 2023–2026 OOS sets include FBS-vs-FCS and FCS games (easier to call); FBS-involved-only 2026 log loss = {val['validation']['2026']['p_ens_FBS_only']['log_loss']:.3f}.\n")
    L.append("Definitions: **Upset probability** = win probability of the model's underdog. **Confidence** = 10·|2p−1|^0.75 minus penalties for model spread (10·SD of member probabilities), models disagreeing on the winner (−1), thin 2026 sample (−0.75) and a QB change last game (−0.5); clipped to 1–10.\n")

    L.append("## Game-by-game predictions\n")
    x = x.sort_values(["date_utc", "event_id"])
    rows_summary = []
    for r in x.itertuples():
        factors, qb_txt, ctx, wx, wx_flag, risk = analyze(r, ts, sd, hfa)
        th, ta = ts.loc[r.home_id], ts.loc[r.away_id]
        conf_lbl = r.home_conf if r.home_conf == r.away_conf else f"{r.away_conf} vs {r.home_conf}"
        L.append(f"### {r.away} @ {r.home}")
        L.append(f"*{r.kickoff_et} · {r.venue}{'' if str(r.venue_city) in str(r.venue) else f' ({r.venue_city}, {r.venue_state})'}{' · neutral site' if r.neutral else ''} · {conf_lbl} · records: {r.away} {th.name if False else ta.wins}-{ta.losses}, {r.home} {th.wins}-{th.losses}*\n")
        L.append(f"**Prediction: {r.pred_winner}**  ")
        L.append(f"Win Probability: {r.away} {fmt_pct(r.away_win_prob)} / {r.home} {fmt_pct(r.home_win_prob)}  ")
        L.append(f"Projected Score: {r.away} {r.proj_away:.0f}–{r.proj_home:.0f} {r.home}  ")
        L.append(f"Projected Margin: {r.proj_margin:.1f} ({r.pred_winner})  ")
        L.append(f"Upset Probability: {fmt_pct(r.upset_prob)}  ")
        L.append(f"Confidence: {r.confidence:.1f}/10{'  ⚠️ HIGH MODEL DISAGREEMENT' if r.high_disagreement else ''}\n")
        L.append("Key Factors:\n")
        top = [f for f in factors[:3]]
        for i, f in enumerate(top, 1):
            L.append(f"{i}. {f[1]}")
        L.append(f"\nMain Risk: {risk}\n")
        L.append("<details><summary>Model breakdown & matchup detail</summary>\n")
        L.append("| Model | " + " | ".join(PNAMES[c] for c in PCOLS) + " | **Ensemble** |\n|---|" + "---|" * (len(PCOLS) + 1))
        L.append(f"| P({r.home} win) | " + " | ".join(f"{getattr(r, c):.3f}" for c in PCOLS) + f" | **{r.p_ens:.3f}** |")
        L.append(f"| Home margin | {r.m_elo:+.1f} | {r.m_pts:+.1f} | — | {r.m_gbm:+.1f} (GBM reg.) | {r.m_reg:+.1f} | **{r.m_final:+.1f}** |\n")
        L.append(f"- Elo: {r.away} {r.elo_a:.0f}, {r.home} {r.elo_h:.0f}; adj. net points rating: {r.away} {ta.net_pts_rating:+.1f}, {r.home} {th.net_pts_rating:+.1f}")
        for f in factors[3:]:
            L.append(f"- {f[1]}")
        L.append(f"- {qb_txt}")
        for c in ctx: L.append(f"- {c}")
        L.append(f"- Rest: {r.away} {r.rest_a:.0f} days, {r.home} {r.rest_h:.0f} days; travel distance: Data unavailable")
        L.append(f"- Weather: {wx[0]}{' — flagged (wind/storms)' if wx_flag else ''}")
        L.append(f"- Injuries/availability: Data unavailable (no public CFB injury feed); QB continuity — {r.away} starter share {r.qb_share_a:.0%}, {r.home} {r.qb_share_h:.0%}")
        L.append("\n</details>\n")
        rows_summary.append(r)

    L.append("## Final slate summary\n")
    L.append("| Game | Kickoff | Predicted Winner | Win % | Projected Score | Upset % | Confidence | Flags |\n|---|---|---|---|---|---|---|---|")
    for r in x.itertuples():
        flags = []
        if r.high_disagreement: flags.append("disagree")
        if r.thin_sample: flags.append("thin data")
        if r.qb_change_flag: flags.append("QB chg")
        L.append(f"| {r.away} @ {r.home} | {r.kickoff_et.split(' ', 1)[1]} | {r.pred_winner} | {fmt_pct(r.win_prob_pred)} | {r.away} {r.proj_away:.0f}–{r.proj_home:.0f} {r.home} | {fmt_pct(r.upset_prob)} | {r.confidence:.1f} | {', '.join(flags)} |")
    def lst(df, f):
        return "\n".join(f"- {f(r)}" for r in df.itertuples())
    hc = x.sort_values("confidence", ascending=False).head(6)
    L.append("\n### Highest-confidence games\n" + lst(hc, lambda r: f"{r.pred_winner} over {r.away if r.pred_winner == r.home else r.home} — {fmt_pct(r.win_prob_pred)}, conf {r.confidence}"))
    cl = x.assign(c=(x.p_ens - 0.5).abs()).sort_values("c").head(6)
    L.append("\n### Closest games\n" + lst(cl, lambda r: f"{r.away} @ {r.home} — {r.pred_winner} {fmt_pct(r.win_prob_pred)}, margin {r.proj_margin:.1f}"))
    fbs_x = x[(x.fbs_h + x.fbs_a) > 0]
    up = x[(x.upset_prob >= 0.25)].sort_values("upset_prob", ascending=False).head(8)
    L.append("\n### Best upset candidates (model underdog ≥ 25%)\n" + lst(up, lambda r: f"{r.away if r.pred_winner == r.home else r.home} over {r.pred_winner} — {fmt_pct(r.upset_prob)}"
                                                                          + (f" (reference market line, not a model input: {r.market_spread_REFERENCE_ONLY})" if isinstance(r.market_spread_REFERENCE_ONLY, str) else "")))
    dis = x.sort_values("p_std", ascending=False).head(6)
    L.append("\n### Highest model disagreement\n" + lst(dis, lambda r: f"{r.away} @ {r.home} — member P({r.home}) range {r[0] if False else min(getattr(r, c) for c in PCOLS):.2f}–{max(getattr(r, c) for c in PCOLS):.2f} (SD {r.p_std:.3f}){'; models split on winner' if r.models_split else ''}"))
    qb = [(t.team, t.qb_starter_sequence) for _, t in ts.iterrows() if t.qb_changed_last_game]
    L.append("\n### Major injury / availability factors\n")
    L.append("- Official injury reports: **Data unavailable** (no programmatic public source reachable; ESPN college injury feed empty).")
    for team, s in qb:
        L.append(f"- {team}: starting QB changed in most recent game (starter sequence: {s}). Treat as possible injury/depth-chart change — unconfirmed.")
    wxg = x[(x.wx_gust_mph >= 20) | x.wx_condition.astype(str).str.contains("storm|Rain", case=False)]
    L.append("\n### Weather watch\n" + lst(wxg, lambda r: f"{r.away} @ {r.home}: {r.wx_condition}, gusts {r.wx_gust_mph} mph, precip {r.wx_precip_pct}% (not adjusted in model; adds variance)"))
    L.append("\n## Reproducibility\n")
    L.append("```\npip install pandas numpy scikit-learn scipy lightgbm requests\npython src/collect_scores.py 2015 ... 2026   # results\n"
             "python src/collect_boxscores.py 2026 2025 2024 2023 2022 2021   # box scores + play-by-play\npython src/collect_slate.py   # today's slate\n"
             "cd src && python pipeline.py && python model.py && python report.py\n```")
    L.append("Files: `games_today.csv`, `team_stats.csv`, `features.csv`, `predictions.csv` (repo root); validation in `output/validation.json`, walk-forward OOS predictions in `output/walkforward_oos_predictions.csv.gz`.")
    open(os.path.join(ROOT, "PREDICTIONS.md"), "w").write("\n".join(L))
    print("ok", len(x))


if __name__ == "__main__":
    main()
