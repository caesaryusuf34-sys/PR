"""Error analysis: turns out-of-sample prediction errors into statistical findings.

All findings are computed on large out-of-sample samples (walk-forward predictions over thousands of
games, plus the live record) and corrected for multiple testing, so the learner reacts to systematic
patterns rather than to the last result."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats as sps

from .metrics import summarize, calibration_table, calibration_slope, logloss_vec, bh_adjust


def _pick_prob(p):
    return np.maximum(p, 1 - p)


def diagnose(oos: pd.DataFrame, features: list[str], subgroups: dict, policy, member_cols=None,
             perm_model=None, perm_frame: pd.DataFrame | None = None) -> dict:
    """oos: out-of-sample rows with p_home, margin (pred), total (pred), home_win, margin_actual, total_actual,
    fold, kickoff_utc and the feature columns."""
    y = oos.home_win.values.astype(float); p = oos.p_home.values
    D = {"n": int(len(oos))}
    D["overall"] = summarize(p, y, oos.margin.values, oos.margin_actual.values, oos.total.values, oos.total_actual.values)
    # ---------------- calibration
    tab = calibration_table(p, y)
    a, b, sa, sb = calibration_slope(p, y)
    cal_flag = None
    if abs(b - 1) > 2 * sb:
        cal_flag = "over-confident" if b < 1 else "under-confident"
    if abs(a) > 2 * sa and cal_flag is None:
        cal_flag = "biased toward " + ("home" if a < 0 else "away")
    D["calibration"] = {"table": tab.to_dict("records"), "ece": D["overall"]["ece"], "intercept": a, "slope": b,
                        "slope_se": sb, "intercept_se": sa, "flag": cal_flag,
                        "needs_recalibration": bool(cal_flag is not None or D["overall"]["ece"] > 0.02)}
    # ---------------- high-confidence misses
    pp = _pick_prob(p); hc = pp >= policy.high_conf
    correct = (p >= 0.5) == (y == 1)
    obs_miss = int((hc & ~correct).sum()); exp_miss = float((1 - pp[hc]).sum())
    var = float((pp[hc] * (1 - pp[hc])).sum())
    z = (obs_miss - exp_miss) / np.sqrt(var) if var > 0 else 0.0
    D["high_conf_misses"] = {"n_high_conf": int(hc.sum()), "observed_misses": obs_miss, "expected_misses": exp_miss,
                             "z": float(z), "p_value": float(1 - sps.norm.cdf(z)), "flag": bool(z > 2)}
    # ---------------- subgroup biases (calibration-in-the-large and margin bias), BH-corrected
    rows = []
    for name, mask in subgroups.items():
        m = np.asarray(mask, bool)
        n = int(m.sum())
        if n < policy.min_subgroup_n:
            continue
        r = y[m] - p[m]
        se = np.sqrt((p[m] * (1 - p[m])).sum()) / n
        zz = r.mean() / se if se > 0 else 0.0
        me = (oos.margin_actual.values[m] - oos.margin.values[m])
        tt = sps.ttest_1samp(me, 0.0)
        rows.append({"subgroup": name, "n": n, "win_bias": float(r.mean()), "win_bias_p": float(2 * (1 - sps.norm.cdf(abs(zz)))),
                     "margin_bias": float(me.mean()), "margin_bias_p": float(tt.pvalue),
                     "log_loss": float(logloss_vec(p[m], y[m]).mean())})
    sg = pd.DataFrame(rows)
    if len(sg):
        sg["win_bias_q"] = bh_adjust(sg.win_bias_p.values); sg["margin_bias_q"] = bh_adjust(sg.margin_bias_p.values)
        sg["significant"] = (sg.win_bias_q < policy.fdr) | (sg.margin_bias_q < policy.fdr)
    D["subgroups"] = sg.to_dict("records")
    D["significant_subgroups"] = sg[sg.significant].subgroup.tolist() if len(sg) else []
    # ---------------- feature / residual structure (missed signal), BH-corrected
    res_p = y - p; res_m = oos.margin_actual.values - oos.margin.values
    fr = []
    for f in features:
        if f not in oos:
            continue
        x = oos[f].astype(float).values
        ok = np.isfinite(x)
        if ok.sum() < 100 or np.nanstd(x[ok]) == 0:
            continue
        r1, p1 = sps.pearsonr(x[ok], res_p[ok]); r2, p2 = sps.pearsonr(x[ok], res_m[ok])
        fr.append({"feature": f, "r_win_resid": float(r1), "p_win": float(p1), "r_margin_resid": float(r2), "p_margin": float(p2)})
    fr = pd.DataFrame(fr)
    if len(fr):
        fr["q_win"] = bh_adjust(fr.p_win.values); fr["q_margin"] = bh_adjust(fr.p_margin.values)
        fr["significant"] = (fr.q_win < policy.fdr) | (fr.q_margin < policy.fdr)
        fr = fr.sort_values("p_win")
    D["feature_residuals"] = fr.to_dict("records")
    D["residual_signal_features"] = fr[fr.significant].feature.tolist() if len(fr) else []
    # ---------------- permutation importance on the newest fold
    if perm_model is not None and perm_frame is not None and len(perm_frame) > 50:
        rng = np.random.default_rng(0)
        yy = perm_frame.home_win.values
        base = logloss_vec(perm_model.predict(perm_frame).p_home.values, yy).mean()
        imp = []
        for f in perm_model.features:
            deltas = []
            for _ in range(3):
                Xp = perm_frame.copy(); Xp[f] = rng.permutation(Xp[f].values)
                deltas.append(logloss_vec(perm_model.predict(Xp).p_home.values, yy).mean() - base)
            imp.append({"feature": f, "delta_logloss": float(np.mean(deltas)), "sd": float(np.std(deltas))})
        imp = pd.DataFrame(imp).sort_values("delta_logloss", ascending=False)
        D["permutation_importance"] = imp.to_dict("records")
        D["useless_features"] = imp[imp.delta_logloss <= 0].feature.tolist()
    # ---------------- members
    if member_cols:
        D["members"] = {c: summarize(oos[c].values, y) for c in member_cols if c in oos}
    # ---------------- drift: is the newest fold worse than the earlier ones?
    ll = logloss_vec(p, y)
    by = pd.DataFrame({"fold": oos.fold.values, "ll": ll}).groupby("fold").ll.agg(["mean", "size", "std"])
    D["by_fold"] = by.reset_index().to_dict("records")
    if len(by) >= 3:
        last = by.iloc[-1]; prev = ll[oos.fold.values != by.index[-1]]
        zd = (last["mean"] - prev.mean()) / np.sqrt(last["std"] ** 2 / last["size"] + prev.var() / len(prev))
        D["drift"] = {"last_fold_ll": float(last["mean"]), "earlier_ll": float(prev.mean()), "z": float(zd), "flag": bool(zd > 2.5)}
    # ---------------- worst misses (for the report, not for fitting)
    o = oos.assign(ll=ll, pick=np.where(p >= 0.5, oos.home_name, oos.away_name))
    D["worst_misses"] = o.sort_values("ll", ascending=False).head(10)[
        ["kickoff_utc", "away_name", "home_name", "p_home", "margin", "margin_actual", "ll"]].astype({"kickoff_utc": str}).to_dict("records")
    return D


def live_record(evaluated: pd.DataFrame, policy) -> dict:
    """Performance of logged predictions (blind live picks are reported separately from backtests)."""
    if evaluated.empty:
        return {}
    # one graded pick per game / model / origin: the latest one written (re-asking never double-counts)
    evaluated = evaluated.sort_values("created_utc").drop_duplicates(["game_id", "model_version", "origin"], keep="last")
    out = {}
    for (origin, ver), g in evaluated.groupby(["origin", "model_version"]):
        s = summarize(g.p_home.values, g.home_win.values, g.pred_margin.values, g.actual_margin.values,
                      g.pred_total.values, g.actual_total.values)
        s["high_conf_misses"] = int(g.high_conf_miss.sum())
        s["blind_share"] = float(g.is_blind.mean())
        out[f"{origin}|{ver}"] = s
    return out


def render_markdown(D: dict, title: str) -> str:
    L = [f"## {title}", ""]
    o = D.get("overall", {})
    if o:
        L.append(f"- n={o.get('n')} · accuracy {o.get('accuracy', 0):.3f} · Brier {o.get('brier', 0):.4f} · log loss {o.get('log_loss', 0):.4f} · "
                 f"ECE {o.get('ece', 0):.4f} · margin MAE {o.get('margin_mae', float('nan')):.2f} · total MAE {o.get('total_mae', float('nan')):.2f}")
    c = D.get("calibration")
    if c:
        L.append(f"- Calibration slope {c['slope']:.3f} ± {c['slope_se']:.3f}, intercept {c['intercept']:+.3f}; flag: {c['flag'] or 'none'}")
    h = D.get("high_conf_misses")
    if h:
        L.append(f"- High-confidence (≥ pick prob threshold) misses: {h['observed_misses']} observed vs {h['expected_misses']:.1f} expected "
                 f"(z={h['z']:+.2f}){' → OVERCONFIDENT' if h['flag'] else ''}")
    if D.get("drift"):
        d = D["drift"]; L.append(f"- Drift: newest fold log loss {d['last_fold_ll']:.4f} vs earlier {d['earlier_ll']:.4f} (z={d['z']:+.2f})")
    if D.get("subgroups"):
        L += ["", "| Subgroup | n | win bias (actual−pred) | q | margin bias (pts) | q | log loss |", "|---|---|---|---|---|---|---|"]
        for r in D["subgroups"]:
            star = " **" if r.get("significant") else ""
            L.append(f"| {r['subgroup']}{star} | {r['n']} | {r['win_bias']:+.3f} | {r['win_bias_q']:.3f} | {r['margin_bias']:+.2f} | {r['margin_bias_q']:.3f} | {r['log_loss']:.4f} |")
    sig = [r for r in D.get("feature_residuals", []) if r.get("significant")]
    L.append("")
    L.append("- Features still correlated with residuals (BH-significant): " +
             (", ".join(f"{r['feature']} (r={r['r_win_resid']:+.3f})" for r in sig[:8]) if sig else "none"))
    if D.get("permutation_importance"):
        top = D["permutation_importance"][:8]
        L.append("- Permutation importance (Δ log loss, newest fold): " + ", ".join(f"{r['feature']} {r['delta_logloss']:+.4f}" for r in top))
        L.append("- Features with no out-of-sample value: " + (", ".join(D.get("useless_features", [])) or "none"))
    return "\n".join(L)
