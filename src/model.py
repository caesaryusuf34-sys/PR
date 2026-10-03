"""Walk-forward ensemble: Elo, opponent-adjusted scoring model, logistic regression,
gradient boosting and point-differential regression. Predicts today's unstarted games."""
import os, json, datetime as dt
import numpy as np, pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score
import lightgbm as lgb
from features import ROOT, CUTOFF, load_games, division_for

EFF = ["epa_pp", "sr", "ypp", "expl", "rush_epa", "pass_epa", "rush_sr", "pass_sr", "sack_rate", "to_rate",
       "pts_per_drive", "ppo", "start_fp"]
FEATS = (["elo_diff", "pts_margin_pred", "pts_total_pred"] + [f"mx_{k}" for k in EFF] +
         ["form_diff", "rest_diff", "qb_share_h", "qb_share_a", "qb_change_h", "qb_change_a",
          "neutral", "conf_game", "fbs_h", "fbs_a", "early", "n_prior_h", "n_prior_a"])


def build(g_rows, R, seq, divs, elo_hfa):
    """g_rows: DataFrame with event_id, season, wk(str), home_id, away_id, neutral, conf_game, elo_h, elo_a."""
    piv = R.pivot_table(index=["season", "wk", "tid"], columns="metric", values=["o", "d"])
    piv.columns = [f"{a}_{b}" for a, b in piv.columns]
    mh = R[R.metric == "pts"].set_index(["season", "wk", "tid"])[["mu", "h"]]
    X = g_rows.copy()
    X["wk"] = X.wk.astype(str)
    H = piv.reindex(pd.MultiIndex.from_frame(X[["season", "wk", "home_id"]])).reset_index(drop=True)
    A = piv.reindex(pd.MultiIndex.from_frame(X[["season", "wk", "away_id"]])).reset_index(drop=True)
    MU = mh.reindex(pd.MultiIndex.from_frame(X[["season", "wk", "home_id"]])).reset_index(drop=True)
    X = X.reset_index(drop=True)
    loc = np.where(X.neutral.astype(bool), 0, 1)
    X["elo_diff"] = X.elo_h - X.elo_a + loc * elo_hfa
    X["pp_home"] = MU.mu + H.o_pts + A.d_pts + MU.h * loc
    X["pp_away"] = MU.mu + A.o_pts + H.d_pts - MU.h * loc
    X["pts_margin_pred"] = X.pp_home - X.pp_away
    X["pts_total_pred"] = X.pp_home + X.pp_away
    for k in EFF:
        X[f"mx_{k}"] = (H[f"o_{k}"] + A[f"d_{k}"]) - (A[f"o_{k}"] + H[f"d_{k}"])
        for side, T in (("h", H), ("a", A)):
            X[f"{side}_o_{k}"] = T[f"o_{k}"].values; X[f"{side}_d_{k}"] = T[f"d_{k}"].values
    X["h_o_pts"], X["h_d_pts"], X["a_o_pts"], X["a_d_pts"] = H.o_pts, H.d_pts, A.o_pts, A.d_pts
    s = seq.set_index(["event_id", "tid"])
    for side, col in (("h", "home_id"), ("a", "away_id")):
        S = s.reindex(pd.MultiIndex.from_frame(X[["event_id", col]].astype(str))).reset_index(drop=True)
        for c in ("form3", "rest", "qb_share", "qb_change", "n_prior"):
            X[f"{c}_{side}"] = S[c].values
    X["form_diff"] = X.form3_h - X.form3_a
    X["rest_diff"] = X.rest_h - X.rest_a
    X["fbs_h"] = [float(division_for(divs, se, t) == "FBS") for se, t in zip(X.season, X.home_id)]
    X["fbs_a"] = [float(division_for(divs, se, t) == "FBS") for se, t in zip(X.season, X.away_id)]
    X["early"] = (pd.to_numeric(X.n_prior_h, errors="coerce").fillna(0) + pd.to_numeric(X.n_prior_a, errors="coerce").fillna(0) < 4).astype(float)
    X["neutral"] = X.neutral.astype(float); X["conf_game"] = X.conf_game.astype(float)
    return X


def tot_X(X):
    """Totals model inputs: scoring-model total plus opponent-adjusted scoring/efficiency environment."""
    cols = [X.pts_total_pred, X.pts_margin_pred.abs(), X.pp_home.clip(0, 80), X.pp_away.clip(0, 80),
            X.h_o_pts_per_drive + X.a_d_pts_per_drive, X.a_o_pts_per_drive + X.h_d_pts_per_drive,
            X.h_o_epa_pp + X.a_d_epa_pp, X.a_o_epa_pp + X.h_d_epa_pp, X.h_o_expl + X.a_d_expl, X.a_o_expl + X.h_d_expl,
            X.early, X.fbs_h, X.fbs_a, X.fbs_h * X.fbs_a, X.conf_game]
    return pd.concat(cols, axis=1).fillna(0).values


def fit_models(tr):
    y = (tr.margin > 0).astype(int).values; ym = tr.margin.values
    Xt = tr[FEATS].fillna(0).values
    mods = {}
    # M1 Elo: logistic on elo_diff/400 scale (calibrated slope) + margin ~ elo_diff
    mods["elo_p"] = LogisticRegression(C=1e6).fit(tr[["elo_diff"]].values / 400, y)
    mods["elo_m"] = Ridge(alpha=1e-6).fit(tr[["elo_diff"]].values, ym)
    # M2 opponent-adjusted scoring model: margin = pts_margin_pred (linear-calibrated); p = Phi(m/sigma)
    mods["pts_m"] = Ridge(alpha=1e-6).fit(tr[["pts_margin_pred"]].values, ym)
    mods["pts_sigma"] = np.std(ym - mods["pts_m"].predict(tr[["pts_margin_pred"]].values))
    mods["tot"] = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(tot_X(tr), tr.total.values)
    # M3 logistic regression on full feature set
    mods["lr"] = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)).fit(Xt, y)
    # M4 gradient boosting (LightGBM)
    mods["gbm"] = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=40,
                                     subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0,
                                     verbose=-1).fit(Xt, y)
    # M5 point-differential regression on full feature set
    mods["reg"] = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(Xt, ym)
    mods["reg_sigma"] = np.std(ym - mods["reg"].predict(Xt))
    mods["gbm_m"] = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=40,
                                      subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0,
                                      verbose=-1).fit(Xt, ym)
    return mods


def predict(mods, X):
    Xt = X[FEATS].fillna(0).values
    out = pd.DataFrame(index=X.index)
    out["p_elo"] = mods["elo_p"].predict_proba(X[["elo_diff"]].values / 400)[:, 1]
    out["m_elo"] = mods["elo_m"].predict(X[["elo_diff"]].values)
    out["m_pts"] = mods["pts_m"].predict(X[["pts_margin_pred"]].values)
    out["p_pts"] = norm.cdf(out.m_pts / mods["pts_sigma"])
    out["p_lr"] = mods["lr"].predict_proba(Xt)[:, 1]
    out["p_gbm"] = mods["gbm"].predict_proba(Xt)[:, 1]
    out["m_reg"] = mods["reg"].predict(Xt)
    out["p_reg"] = norm.cdf(out.m_reg / mods["reg_sigma"])
    out["m_gbm"] = mods["gbm_m"].predict(Xt)
    out["total"] = mods["tot"].predict(tot_X(X))
    return out


PCOLS = ["p_elo", "p_pts", "p_lr", "p_gbm", "p_reg"]
MCOLS = ["m_elo", "m_pts", "m_reg", "m_gbm"]


def simplex_weights(P, y, loss):
    k = P.shape[1]
    def f(w):
        w = np.abs(w) / np.abs(w).sum()
        return loss(P @ w, y)
    r = minimize(f, np.ones(k) / k, method="Nelder-Mead", options={"maxiter": 4000, "xatol": 1e-5, "fatol": 1e-7})
    w = np.abs(r.x) / np.abs(r.x).sum()
    return w


def ll(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4); return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def metrics(p, y, m=None, margin=None):
    d = {"n": len(y), "accuracy": accuracy_score(y, p > 0.5), "brier": brier_score_loss(y, p), "log_loss": ll(p, y)}
    if m is not None:
        d["margin_MAE"] = float(np.mean(np.abs(m - margin)))
    return d


def main():
    g, divs = load_games()
    R = pd.read_csv(os.path.join(ROOT, "data", "weekly_ratings.csv.gz"), dtype={"tid": str, "wk": str})
    seq = pd.read_csv(os.path.join(ROOT, "data", "team_sequence.csv.gz"), dtype={"event_id": str, "tid": str})
    gr = pd.read_csv(os.path.join(ROOT, "data", "games_rated.csv.gz"), dtype={"event_id": str, "home_id": str, "away_id": str})
    elo_params = json.load(open(os.path.join(ROOT, "data", "elo_params.json")))
    HFA = elo_params["HFA"]
    gr["wk"] = gr.wk.astype(int).astype(str)
    hist = gr[gr.season >= 2022].copy()
    X = build(hist, R, seq, divs, HFA)
    X["margin"] = X.home_score - X.away_score; X["total"] = X.home_score + X.away_score
    X = X[X.margin != 0]
    # ---- walk-forward validation: train seasons < S, test S
    oos = []
    for S in (2023, 2024, 2025, 2026):
        tr, te = X[X.season < S], X[X.season == S]
        mods = fit_models(tr)
        pr = predict(mods, te); pr["season"] = S; pr["y"] = (te.margin > 0).astype(int).values
        pr["margin"] = te.margin.values; pr["total_actual"] = te.total.values; pr["fbs_game"] = (te.fbs_h + te.fbs_a > 0).values
        oos.append(pr)
    O = pd.concat(oos)
    V = O[O.season <= 2025]  # ensemble-weight fitting set
    wp = simplex_weights(V[PCOLS].values, V.y.values, ll)
    wm = simplex_weights(V[MCOLS].values, V.margin.values, lambda p, y: np.mean((p - y) ** 2))
    O["p_ens"] = O[PCOLS].values @ wp; O["m_ens"] = O[MCOLS].values @ wm
    sig = float(np.std(V.margin - V[MCOLS].values @ wm))
    # validation report
    rep = {}
    for S in (2023, 2024, 2025, 2026):
        o = O[O.season == S]
        rep[S] = {c: metrics(o[c].values, o.y.values) for c in PCOLS + ["p_ens"]}
        for c in MCOLS + ["m_ens"]:
            rep[S][c] = {"margin_MAE": float(np.mean(np.abs(o[c] - o.margin)))}
        rep[S]["total_MAE"] = float(np.mean(np.abs(o.total - o.total_actual)))
        of = o[o.fbs_game]
        rep[S]["p_ens_FBS_only"] = metrics(of.p_ens.values, of.y.values)
    bins = pd.cut(O.p_ens, np.linspace(0, 1, 11))
    calib = O.groupby(bins, observed=True).agg(n=("y", "size"), mean_pred=("p_ens", "mean"), actual=("y", "mean")).reset_index()
    calib["bin"] = calib.p_ens.astype(str); calib = calib.drop(columns="p_ens")
    json.dump({"weights_prob": dict(zip(PCOLS, map(float, wp))), "weights_margin": dict(zip(MCOLS, map(float, wm))),
               "ensemble_margin_sigma": sig, "validation": rep, "calibration_2023_2026": calib.to_dict("records"),
               "elo_params": elo_params},
              open(os.path.join(ROOT, "output", "validation.json"), "w"), indent=1, default=float)
    O.to_csv(os.path.join(ROOT, "output", "walkforward_oos_predictions.csv.gz"), index=False, compression="gzip")
    print("weights p", dict(zip(PCOLS, wp.round(3))), "weights m", dict(zip(MCOLS, wm.round(3))), "sigma", round(sig, 2))
    for S in rep:
        print(S, {k: (round(v["log_loss"], 4), round(v["accuracy"], 3)) if "log_loss" in v else round(v.get("margin_MAE", 0), 2)
                  for k, v in rep[S].items() if isinstance(v, dict)}, "totMAE", round(rep[S]["total_MAE"], 2))
    print(calib.to_string())

    # ---- final model: train on all completed games 2022 .. pre-today
    mods = fit_models(X)
    slate = pd.read_csv(os.path.join(ROOT, "data", "slate_all_today.csv"), dtype={"event_id": str, "home_id": str, "away_id": str})
    now = pd.Timestamp.now(tz="UTC")
    slate["date_utc"] = pd.to_datetime(slate.date_utc, utc=True)
    todo = slate[(slate.state == "pre") & (slate.date_utc > now)].copy()
    skipped = slate[(slate.state == "pre") & (slate.date_utc <= now)]
    elo_final = json.load(open(os.path.join(ROOT, "data", "elo_final.json")))["elo_final"]
    def elo_now(t):
        if t in elo_final: return elo_final[t]
        from features import ELO_INIT
        return ELO_INIT[division_for(divs, 2026, t)]
    todo["season"] = 2026; todo["wk"] = "today"
    todo["elo_h"] = todo.home_id.map(elo_now); todo["elo_a"] = todo.away_id.map(elo_now)
    todo["neutral"] = todo.neutral.fillna(False).astype(bool); todo["conf_game"] = todo.conf_game.fillna(False).astype(bool)
    # today's team-sequence rows (form/rest/QB as of the last completed game)
    last = seq.sort_values("date").groupby("tid").tail(1).set_index("tid")
    seq_all = seq.sort_values("date")
    rows = []
    for r in todo.itertuples():
        for tid in (r.home_id, r.away_id):
            ts = seq_all[(seq_all.tid == tid) & (seq_all.season == 2026)]
            if len(ts):
                L = ts.iloc[-1]
                resid = ts.resid.tail(3).mean()
                rest = (r.date_utc - pd.Timestamp(L.date)).days
                att = ts.dropna(subset=["qb_id"]).groupby("qb_id").qb_att.sum()
                lastqb = ts.dropna(subset=["qb_id"]).qb_id.iloc[-1] if ts.qb_id.notna().any() else None
                share = float(att.get(lastqb, 0) / att.sum()) if lastqb is not None and att.sum() > 0 else 1.0
                qbs = ts.dropna(subset=["qb_id"]).qb_id.tolist()
                change = float(len(qbs) >= 2 and qbs[-1] != qbs[-2])
                rows.append(dict(event_id=r.event_id, tid=tid, form3=resid, rest=min(rest, 21), qb_share=share,
                                 qb_change=change, n_prior=len(ts)))
            else:
                rows.append(dict(event_id=r.event_id, tid=tid, form3=0, rest=14, qb_share=1.0, qb_change=0.0, n_prior=0))
    seq_today = pd.DataFrame(rows)
    XT = build(todo, R, seq_today, divs, HFA)
    PT = predict(mods, XT)
    XT = pd.concat([XT, PT], axis=1).copy()
    XT["p_ens"] = XT[PCOLS].values @ wp
    XT["m_ens"] = XT[MCOLS].values @ wm
    # reconcile margin with probability (blend ensemble margin with probability-implied margin)
    XT["m_final"] = 0.5 * XT.m_ens + 0.5 * sig * norm.ppf(XT.p_ens.clip(0.001, 0.999))
    XT["p_std"] = XT[PCOLS].std(axis=1)
    XT["m_std"] = XT[MCOLS].std(axis=1)
    XT.to_pickle(os.path.join(ROOT, "output", "slate_features.pkl"))
    json.dump({"skipped_started_before_freeze": skipped[["event_id", "away", "home", "kickoff_et"]].to_dict("records"),
               "freeze_utc": now.isoformat(), "sigma": sig},
              open(os.path.join(ROOT, "output", "freeze_meta.json"), "w"), indent=1, default=str)
    print(XT[["away", "home", "p_ens", "m_ens", "m_final", "total", "p_std"]].round(3).to_string())


if __name__ == "__main__":
    os.makedirs(os.path.join(ROOT, "output"), exist_ok=True)
    main()
