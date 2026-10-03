"""Build walk-forward ratings + features for every historical game (2016-2026) and for today's slate."""
import os, sys, time, json
import numpy as np, pandas as pd
from features import (ROOT, CUTOFF, load_games, run_elo, load_parsed, add_epa, team_game_metrics,
                      ridge_ratings, division_for)

T0 = time.time()
def log(*a):
    print(f"[{time.time()-T0:6.1f}s]", *a, flush=True)

EFF = ["epa_pp", "sr", "ypp", "expl", "rush_epa", "pass_epa", "rush_sr", "pass_sr", "sack_rate", "to_rate",
       "pts_per_drive", "ppo", "start_fp"]
LAM = {"pts": 3.0}; LAM.update({k: 4.0 for k in EFF})
SHRINK = 0.6  # weight on last season's final rating in the preseason prior


def main():
    g, divs = load_games()
    log("games", len(g))
    # ---- Elo: small grid on 2017-2022 (log loss), applied to all
    best = None
    for K in (20, 28, 36):
        for HFA in (45, 65):
            for rev in (0.25, 0.4):
                ph, pa, _ = run_elo(g, divs, K, HFA, rev)
                sel = (g.season >= 2017) & (g.season <= 2022)
                d = ph - pa + np.where(g.neutral, 0, HFA)
                p = 1 / (1 + 10 ** (-d / 400)); y = (g.home_score > g.away_score).astype(float)
                ll = -np.mean((y * np.log(p) + (1 - y) * np.log(1 - p))[sel])
                if best is None or ll < best[0]:
                    best = (ll, K, HFA, rev)
    _, K, HFA, REV = best
    log("elo params", best)
    g["elo_h"], g["elo_a"], elo_final = run_elo(g, divs, K, HFA, REV)
    g["elo_hfa"] = np.where(g.neutral, 0, HFA)
    json.dump({"K": K, "HFA": HFA, "revert": REV, "train_logloss_2017_2022": best[0]},
              open(os.path.join(ROOT, "data", "elo_params.json"), "w"))

    # ---- play-level EPA + team-game metrics (2021-2026)
    tb, pl = load_parsed()
    log("parsed", len(tb), len(pl))
    s = add_epa(pl, g)
    log("scrimmage plays", len(s))
    m = team_game_metrics(s, pl, tb)
    m = m.merge(g[["event_id", "season", "wk", "date", "home_id", "away_id", "neutral", "home_score", "away_score"]],
                on="event_id", how="inner")
    m["is_home"] = m.tid == m.home_id
    m["opp"] = np.where(m.is_home, m.away_id, m.home_id)
    m["pts"] = np.where(m.is_home, m.home_score, m.away_score)
    m["pts_allowed"] = np.where(m.is_home, m.away_score, m.home_score)
    m["loc"] = np.where(m.neutral, 0, np.where(m.is_home, 1, -1))
    m.to_csv(os.path.join(ROOT, "data", "team_game_metrics.csv.gz"), index=False, compression="gzip")
    log("team-game metrics", m.shape)

    # ---- points observations for all seasons (score-based) ---------------------------------
    ph = g.assign(off=g.home_id, dfn=g.away_id, y=g.home_score, loc=np.where(g.neutral, 0, 1))
    pa = g.assign(off=g.away_id, dfn=g.home_id, y=g.away_score, loc=np.where(g.neutral, 0, -1))
    pts_obs = pd.concat([ph, pa])[["event_id", "season", "wk", "off", "dfn", "y", "loc"]]
    eff_obs = m.rename(columns={"tid": "off", "opp": "dfn"})

    # ---- weekly walk-forward ratings ----------------------------------------------------
    rating_rows = []  # (season, wk, tid, metric, o, d, mu, h)
    final = {}       # metric -> (o dict, d dict) at season end
    seasons = sorted(g.season.unique())
    for season in seasons:
        if season < 2016: continue
        sg = g[g.season == season]
        teams = sorted(set(sg.home_id) | set(sg.away_id))
        weeks = sorted(sg.wk.unique()) + (["today"] if season == 2026 else [])
        metrics = ["pts"] + (EFF if season >= 2021 else [])
        for metric in metrics:
            prev = final.get((metric, season - 1))
            if metric == "pts" and prev is None and season == 2016:
                # bootstrap 2015 finals as prior for 2016
                prev = None
            prior_o, prior_d = {}, {}
            if prev:
                po, pd_ = prev
                # division mean of last-season ratings
                dm = {}
                for t in po:
                    dm.setdefault(division_for(divs, season - 1, t), []).append((po[t], pd_[t]))
                dmean = {k: (np.mean([a for a, _ in v]), np.mean([b for _, b in v])) for k, v in dm.items()}
                for t in teams:
                    dv = division_for(divs, season, t)
                    mo, md = dmean.get(dv, dmean.get("OTHER", (0, 0)))
                    if t in po:
                        prior_o[t] = SHRINK * po[t] + (1 - SHRINK) * mo; prior_d[t] = SHRINK * pd_[t] + (1 - SHRINK) * md
                    else:
                        prior_o[t], prior_d[t] = mo, md
            obs_all = (pts_obs if metric == "pts" else eff_obs.rename(columns={metric: "y"}))
            obs_all = obs_all[obs_all.season == season]
            if metric != "pts":
                obs_all = obs_all[["off", "dfn", "y", "loc", "wk"]].dropna()
            # include last season as weak observations? no - prior only
            for w in weeks:
                ob = obs_all if w == "today" else obs_all[obs_all.wk < w]
                mu, h, o, d = ridge_ratings(ob, teams, prior_o, prior_d, LAM[metric], LAM[metric])
                for t in teams:
                    rating_rows.append((season, w, t, metric, o[t], d[t], mu, h))
            # end-of-season fit
            mu, h, o, d = ridge_ratings(obs_all, teams, prior_o, prior_d, LAM[metric], LAM[metric])
            final[(metric, season)] = (o, d)
        log("season", season, "done")
    R = pd.DataFrame(rating_rows, columns=["season", "wk", "tid", "metric", "o", "d", "mu", "h"])
    R["wk"] = R.wk.astype(str)
    R.to_csv(os.path.join(ROOT, "data", "weekly_ratings.csv.gz"), index=False, compression="gzip")
    log("ratings", R.shape)

    # ---- form, rest, QB continuity (team-game sequence) -----------------------------------
    Rp = R[R.metric == "pts"].set_index(["season", "wk", "tid"])
    def pts_pred(season, wk, home, away, loc):
        a = Rp.loc[(season, str(wk), home)]; b = Rp.loc[(season, str(wk), away)]
        hp = a.mu + a.o + b.d + a.h * loc; ap = a.mu + b.o + a.d - a.h * loc
        return hp, ap
    hp, ap = [], []
    for r in g.itertuples(index=False):
        if r.season < 2016: hp.append(np.nan); ap.append(np.nan); continue
        x, y = pts_pred(r.season, r.wk, r.home_id, r.away_id, 0 if r.neutral else 1)
        hp.append(x); ap.append(y)
    g["pp_home"], g["pp_away"] = hp, ap
    seq = pd.concat([
        g.assign(tid=g.home_id, margin=g.home_score - g.away_score, pmargin=g.pp_home - g.pp_away),
        g.assign(tid=g.away_id, margin=g.away_score - g.home_score, pmargin=g.pp_away - g.pp_home)])
    seq = seq[["event_id", "season", "date", "tid", "margin", "pmargin"]].merge(
        m[["event_id", "tid", "qb_id", "qb_name", "qb_att"]], on=["event_id", "tid"], how="left")
    seq = seq.sort_values(["tid", "date"]).reset_index(drop=True)
    seq["resid"] = seq.margin - seq.pmargin
    gb = seq.groupby(["tid", "season"])
    seq["form3"] = gb.resid.transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean()).fillna(0)
    seq["form5_margin"] = gb.margin.transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    seq["rest"] = gb.date.transform(lambda x: x.diff().dt.days).fillna(14).clip(upper=21)
    seq["n_prior"] = gb.cumcount()
    # QB continuity: share of season-to-date attempts thrown by the most recent game's starter
    shares, chg = [], []
    for (tid, season), grp in seq.groupby(["tid", "season"], sort=False):
        att = {}; last = None; prevlast = None
        for r in grp.itertuples():
            tot = sum(att.values())
            shares.append((att.get(last, 0) / tot) if (tot > 0 and last is not None) else 1.0)
            chg.append(float(last is not None and prevlast is not None and last != prevlast))
            if isinstance(r.qb_id, str) and r.qb_id:
                att[r.qb_id] = att.get(r.qb_id, 0) + float(np.nan_to_num(r.qb_att))
                prevlast, last = last, r.qb_id
    seq["qb_share"] = shares; seq["qb_change"] = chg
    seq.to_csv(os.path.join(ROOT, "data", "team_sequence.csv.gz"), index=False, compression="gzip")
    g.to_csv(os.path.join(ROOT, "data", "games_rated.csv.gz"), index=False, compression="gzip")
    json.dump({"elo_final": elo_final}, open(os.path.join(ROOT, "data", "elo_final.json"), "w"))
    log("done")

if __name__ == "__main__":
    main()
