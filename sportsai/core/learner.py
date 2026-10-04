"""The self-learning loop.

    accumulated games ──► walk-forward OOS predictions of the champion ──► error analysis
            ▲                                                                  │
            │                                       diagnosis-driven candidate configurations
            │                                                                  │
    new version ◄── promotion gate (confirmation window, bootstrap) ◄── walk-forward OOS of candidates

Anti-overfitting design
  * every comparison is out-of-sample and time-ordered (train strictly before each test block);
  * candidates are ranked on the older "selection" folds and the winner is then tested against the
    champion on the newer "confirmation" folds it was not chosen on;
  * a structural change needs a minimum effect size AND one-sided significance under a block
    bootstrap by calendar week AND no worse Brier / calibration AND a minimum sample size;
  * otherwise the champion's (already validated) configuration is simply re-estimated on all data;
  * the candidate set per cycle is small and deterministic (diagnosis-driven + one local hyper-
    parameter move), limiting the number of hypotheses tested.
"""
from __future__ import annotations
import copy, json, time, uuid
import numpy as np
import pandas as pd

from ..models.ensemble import EnsembleModel
from .analysis import diagnose, live_record, render_markdown
from .metrics import summarize, logloss_vec, brier_vec, ece
from .timeutil import iso, now

SUBGROUP_FEATURES = {"early_season": ["early", "n_prior_h", "n_prior_a"], "fbs_vs_fcs": ["fbs_h", "fbs_a"],
                     "fcs_or_lower_only": ["fbs_h", "fbs_a"], "both_fbs": ["fbs_h", "fbs_a"],
                     "conference_game": ["conf_game"], "neutral_site": ["neutral"],
                     "qb_change_any": ["qb_change_h", "qb_change_a"], "home_favorite": ["neutral"], "road_favorite": ["neutral"]}


class Learner:
    def __init__(self, engine):
        self.E = engine
        self.P = engine.S.policy

    # ------------------------------------------------------------------ trigger
    def new_games_since_champion(self) -> tuple[int, str | None]:
        ch = self.E.registry.champion_row()
        if ch is None:
            return 0, None
        done = self.E.store.completed_games(self.E.sport)
        cut = pd.Timestamp(ch.train_cutoff_utc)
        avail = done.kickoff_utc + pd.Timedelta(hours=self.E.adapter.game_duration_hours)
        seen = pd.to_datetime(done.result_collected_utc, utc=True)
        avail = avail.where((seen >= avail) | (seen <= done.kickoff_utc), seen)   # same availability rule as features
        return int(((avail > cut) & (avail <= now())).sum()), ch.version

    def degradation_z(self) -> float | None:
        """Is the champion's BLIND live record significantly worse than its validated walk-forward log loss?"""
        ch = self.E.registry.champion_row()
        if ch is None or not ch.validation_json:
            return None
        expected = json.loads(ch.validation_json).get("walk_forward", {}).get("log_loss")
        ev = self.E.store.evaluated(self.E.sport)
        ev = ev[(ev.model_version == ch.version) & (ev.origin == "live") & (ev.is_blind == 1)]
        if expected is None or len(ev) < 150:
            return None
        return float((ev.logloss.mean() - expected) / (ev.logloss.std() / np.sqrt(len(ev))))

    def due(self) -> tuple[bool, str]:
        n, v = self.new_games_since_champion()
        if v is None:
            return False, "no champion"
        z = self.degradation_z()
        if z is not None and z > 3:
            return True, f"live performance of {v} degraded vs its validation (z={z:+.1f})"
        if n >= self.P.min_new_games:
            return True, f"{n} completed games since {v} was trained (threshold {self.P.min_new_games})"
        return False, f"{n}/{self.P.min_new_games} new completed games since {v}"

    # ------------------------------------------------------------------ walk-forward
    def folds(self, data: pd.DataFrame):
        n = len(data); W = min(self.P.eval_window_games, int(n * 0.45))
        idx = np.arange(n - W, n)
        blocks = np.array_split(idx, self.P.n_folds)
        dur = pd.Timedelta(hours=self.E.adapter.game_duration_hours)
        avail = data.kickoff_utc + dur
        out = []
        for b, blk in enumerate(blocks):
            start = data.kickoff_utc.iloc[blk[0]]
            train = np.where((avail <= start).values)[0]          # strictly finished before the block starts
            out.append((b, train, blk))
        return out

    def walk_forward(self, config: dict, data: pd.DataFrame, folds) -> tuple[pd.DataFrame, EnsembleModel, pd.DataFrame]:
        fv = self.E.adapter.feature_builder_version
        parts, last = [], None
        for b, tr, te in folds:
            trd = data.iloc[tr]
            trd = trd[trd.season >= config.get("min_train_season", 0)]
            m = EnsembleModel(config, fv).fit(trd)
            ted = data.iloc[te].reset_index(drop=True)
            pr = m.predict(ted).reset_index(drop=True)
            pr["fold"] = b
            parts.append(pd.concat([ted.rename(columns={"margin": "margin_actual", "total": "total_actual"}), pr], axis=1))
            last = (m, ted)
        return pd.concat(parts, ignore_index=True), last[0], last[1]

    # ------------------------------------------------------------------ candidates
    def propose(self, cfg: dict, diag: dict, run_index: int) -> list[dict]:
        cands = []
        def add(name, why, fn):
            c = copy.deepcopy(cfg); fn(c)
            if c != cfg and all(c != x["config"] for x in cands):
                cands.append({"name": name, "why": why, "config": c})
        cal = diag.get("calibration", {})
        if cal.get("needs_recalibration") or diag.get("high_conf_misses", {}).get("flag"):
            why = f"calibration flag={cal.get('flag')}, ECE={cal.get('ece', 0):.4f}, high-conf misses z={diag['high_conf_misses']['z']:+.2f}"
            add("calibration_platt", why, lambda c: c.update(calibration="platt"))
            add("calibration_isotonic", why, lambda c: c.update(calibration="isotonic"))
        elif cfg.get("calibration", "none") != "none":
            add("calibration_none", "current calibrator may be unnecessary", lambda c: c.update(calibration="none"))
        else:
            add("calibration_platt", "routine calibration check", lambda c: c.update(calibration="platt"))
        sig = diag.get("significant_subgroups", [])
        ctx = sorted({f for s in sig for f in SUBGROUP_FEATURES.get(s, []) if f in self.available})
        if ctx:
            add("stack_subgroup_context", f"systematic bias in subgroups {sig}", lambda c: c.update(ensemble="stack", stack_context=ctx))
        resid = [f for f in diag.get("residual_signal_features", []) if f in self.available][:6]
        if resid:
            add("stack_residual_features", f"features still correlated with residuals: {resid}",
                lambda c: c.update(ensemble="stack", stack_context=sorted(set(c.get("stack_context", [])) | set(resid))))
            add("gbm_more_capacity", f"possible non-linear signal in {resid}",
                lambda c: [c["members"][k].update(num_leaves=31, min_child_samples=30) for k in ("gbm", "gbm_margin") if k in c["members"]])
        useless = [f for f in diag.get("useless_features", []) if f in cfg["features"]]
        if len(useless) >= 2:
            add("drop_useless_features", f"no out-of-sample value: {useless}", lambda c: c.update(features=[f for f in c["features"] if f not in useless]))
        extra = [f for f in resid if f not in cfg["features"]]
        if extra:
            add("add_residual_features", f"add {extra} as direct inputs", lambda c: c.update(features=c["features"] + extra))
        w = self.champion_weights or {}
        dead = [k for k in cfg["members"] if w.get("prob", {}).get(k, 1) < 0.02 and w.get("margin", {}).get(k, 1) < 0.02
                and isinstance(w.get("prob"), dict)]
        if dead and len(cfg["members"]) - len(dead) >= 3:
            add("drop_dead_members", f"members with ~0 ensemble weight: {dead}", lambda c: [c["members"].pop(k) for k in dead])
        if diag.get("drift", {}).get("flag"):
            add("recency_weighting", f"newest fold worse than earlier (z={diag['drift']['z']:+.2f})", lambda c: c.update(half_life_days=365))
        # one local hyper-parameter move per cycle, rotating through the neighbourhood over time
        moves = [("logistic_C_down", lambda c: c["members"].get("logistic", {}).update(C=round(c["members"]["logistic"].get("C", 0.1) / 3, 4)) if "logistic" in c["members"] else None),
                 ("ridge_alpha_up", lambda c: c["members"].get("margin_ridge", {}).update(alpha=c["members"]["margin_ridge"].get("alpha", 10) * 3) if "margin_ridge" in c["members"] else None),
                 ("gbm_slower_more_trees", lambda c: [c["members"][k].update(learning_rate=0.02, n_estimators=500) for k in ("gbm", "gbm_margin") if k in c["members"]]),
                 ("logistic_C_up", lambda c: c["members"].get("logistic", {}).update(C=round(c["members"]["logistic"].get("C", 0.1) * 3, 4)) if "logistic" in c["members"] else None),
                 ("ridge_alpha_down", lambda c: c["members"].get("margin_ridge", {}).update(alpha=c["members"]["margin_ridge"].get("alpha", 10) / 3) if "margin_ridge" in c["members"] else None),
                 ("gbm_more_regularized", lambda c: [c["members"][k].update(min_child_samples=80, reg_lambda=15.0) for k in ("gbm", "gbm_margin") if k in c["members"]]),
                 ("half_life_2y", lambda c: c.update(half_life_days=730))]
        nm, fn = moves[run_index % len(moves)]
        add(f"local_{nm}", "local hyper-parameter search (rotating neighbourhood)", fn)
        return cands[: self.P.max_candidates]

    # ------------------------------------------------------------------ statistics
    def _bootstrap(self, champ: pd.DataFrame, chal: pd.DataFrame) -> dict:
        y = champ.home_win.values
        d = logloss_vec(champ.p_home.values, y) - logloss_vec(chal.p_home.values, y)
        weeks = champ.kickoff_utc.dt.strftime("%G-%V").values
        uw, inv = np.unique(weeks, return_inverse=True)
        sums = np.bincount(inv, weights=d); cnts = np.bincount(inv)
        rng = np.random.default_rng(42)
        boots = []
        for _ in range(self.P.bootstrap_reps):
            s = rng.integers(0, len(uw), len(uw))
            boots.append(sums[s].sum() / cnts[s].sum())
        boots = np.array(boots)
        return {"n": int(len(d)), "weeks": int(len(uw)), "mean_gain": float(d.mean()),
                "ci90": [float(np.quantile(boots, 0.05)), float(np.quantile(boots, 0.95))],
                "p_value": float((boots <= 0).mean())}

    # ------------------------------------------------------------------ first model
    def bootstrap(self, as_of=None, config: dict | None = None, version: str = "v1.0") -> dict:
        E = self.E; as_of = as_of or now()
        cfg = config or E.adapter.default_config()
        data = E.fstore.training_frame(as_of=as_of, min_season=E.adapter.first_train_season)
        folds = self.folds(data)
        print(f"[bootstrap] {len(data)} games up to {iso(as_of)}; walk-forward over {sum(len(f[2]) for f in folds)} games", flush=True)
        oos, _, _ = self.walk_forward(cfg, data, folds)
        val = {"walk_forward": summarize(oos.p_home, oos.home_win, oos.margin, oos.margin_actual, oos.total, oos.total_actual),
               "by_fold": [summarize(g.p_home, g.home_win, g.margin, g.margin_actual) for _, g in oos.groupby("fold")],
               "eval_games": int(len(oos))}
        model = EnsembleModel(cfg, E.adapter.feature_builder_version).fit(data[data.season >= cfg.get("min_train_season", 0)])
        v = E.registry.register(model, kind="bootstrap", reason=f"initial model; default configuration validated walk-forward on {len(oos)} games",
                                train_cutoff=as_of, n_train=model.n_train, validation=val, parent=None, version=version)
        return {"version": v, "validation": val, "weights": model.weights()}

    # ------------------------------------------------------------------ main
    def run(self, trigger: str = "auto", force: bool = False) -> dict:
        E = self.E; t0 = time.time(); started = now()
        due, why = self.due()
        if not (due or force):
            return {"decision": "skipped", "reason": why}
        ch = E.registry.champion_row()
        champ_cfg = json.loads(ch.config_json)
        champ_model = E.registry.load(ch.version)
        self.champion_weights = champ_model.weights()
        n_new, _ = self.new_games_since_champion()
        data = E.fstore.training_frame(min_season=E.adapter.first_train_season)
        self.available = set(data.columns)
        folds = self.folds(data)
        print(f"[learn] {len(data)} games, eval window {sum(len(f[2]) for f in folds)} in {len(folds)} folds; {why}", flush=True)
        oos_c, last_m, last_frame = self.walk_forward(champ_cfg, data, folds)
        member_cols = [c for c in oos_c.columns if c.startswith("p_") and c not in ("p_home", "p_raw", "p_std")]
        diag = diagnose(oos_c, sorted(set(E.builder_candidates()) & set(data.columns)), E.adapter.subgroups(oos_c), self.P,
                        member_cols=member_cols, perm_model=last_m, perm_frame=last_frame)
        live = live_record(E.store.evaluated(E.sport), self.P)
        runs = int(E.store.df("SELECT COUNT(*) n FROM learning_runs WHERE sport=?", (E.sport,)).n.iloc[0])
        cands = self.propose(champ_cfg, diag, runs)
        sel_folds = set(range(self.P.n_folds - self.P.confirm_folds)); conf_folds = set(range(self.P.n_folds)) - sel_folds
        def split(o):
            return o[o.fold.isin(sel_folds)], o[o.fold.isin(conf_folds)]
        c_sel, c_conf = split(oos_c)
        champ_sel_ll = float(logloss_vec(c_sel.p_home, c_sel.home_win).mean())
        results = []
        for c in cands:
            print(f"[learn] candidate {c['name']}: {c['why']}", flush=True)
            try:
                o, _, _ = self.walk_forward(c["config"], data, folds)
            except Exception as e:  # a broken candidate is rejected, never fatal
                results.append({**c, "error": str(e)}); continue
            s, cf_ = split(o)
            results.append({**c, "oos": o, "sel_ll": float(logloss_vec(s.p_home, s.home_win).mean()),
                            "full": summarize(o.p_home, o.home_win, o.margin, o.margin_actual, o.total, o.total_actual)})
        ok = [r for r in results if "sel_ll" in r and r["sel_ll"] < champ_sel_ll]
        best = min(ok, key=lambda r: r["sel_ll"]) if ok else None
        test, decision, reason = None, None, None
        if best is not None:
            _, b_conf = split(best["oos"])
            test = self._bootstrap(c_conf.reset_index(drop=True), b_conf.reset_index(drop=True))
            test["brier_champion"] = float(brier_vec(c_conf.p_home, c_conf.home_win).mean())
            test["brier_challenger"] = float(brier_vec(b_conf.p_home, b_conf.home_win).mean())
            test["ece_champion"] = ece(c_conf.p_home, c_conf.home_win); test["ece_challenger"] = ece(b_conf.p_home, b_conf.home_win)
            gates = {"min_sample": test["n"] >= self.P.min_confirm_games,
                     "effect_size": test["mean_gain"] >= self.P.min_logloss_gain,
                     "significance": test["p_value"] < self.P.alpha,
                     "brier_not_worse": test["brier_challenger"] <= test["brier_champion"],
                     "calibration_not_worse": test["ece_challenger"] <= test["ece_champion"] + self.P.max_ece_worsening}
            test["gates"] = gates
            if all(gates.values()):
                decision = "structural"
                reason = (f"accepted challenger '{best['name']}' ({best['why']}): confirmation-window log-loss gain "
                          f"{test['mean_gain']:+.4f} (90% CI {test['ci90'][0]:+.4f}..{test['ci90'][1]:+.4f}, p={test['p_value']:.3f}, "
                          f"n={test['n']} games / {test['weeks']} weeks)")
        if decision is None:
            if n_new >= self.P.min_new_games or force:
                decision = "refit"
                failed = (f"best challenger '{best['name']}' failed gates {[k for k, v in test['gates'].items() if not v]}"
                          if best is not None else "no challenger beat the champion on the selection folds")
                reason = f"re-estimated validated configuration on +{n_new} new completed games; {failed}"
            else:
                decision = "no_change"; reason = "not enough new data"
        new_version = ch.version
        val_src = best["oos"] if decision == "structural" else oos_c
        validation = {"walk_forward": summarize(val_src.p_home, val_src.home_win, val_src.margin, val_src.margin_actual,
                                                val_src.total, val_src.total_actual),
                      "confirmation_test": test, "eval_games": int(len(val_src))}
        if decision in ("structural", "refit"):
            cfg = best["config"] if decision == "structural" else champ_cfg
            model = EnsembleModel(cfg, E.adapter.feature_builder_version).fit(data[data.season >= cfg.get("min_train_season", 0)])
            new_version = E.registry.register(model, kind=decision, reason=reason, train_cutoff=now(), n_train=model.n_train,
                                              validation=validation, parent=ch.version)
        run_id = uuid.uuid4().hex[:12]
        summary = {"trigger": trigger, "why_due": why, "n_new_games": n_new, "decision": decision, "reason": reason,
                   "champion_selection_ll": champ_sel_ll,
                   "champion_before": ch.version, "champion_after": new_version,
                   "champion_walk_forward": diag["overall"], "confirmation_test": test,
                   "candidates": [{k: v for k, v in r.items() if k not in ("oos", "config")} | {"config_diff": _diff(champ_cfg, r["config"])}
                                  for r in results], "live_record": live, "seconds": round(time.time() - t0, 1)}
        path = self._report(run_id, summary, diag, started)
        with E.store.tx() as c:
            c.execute("INSERT INTO learning_runs VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                      (run_id, E.sport, iso(started), iso(now()), trigger, n_new, ch.version, new_version, decision,
                       json.dumps(summary, default=str), str(path.relative_to(E.S.root))))
        return summary

    def _report(self, run_id, summary, diag, started):
        d = self.E.S.reports_dir / self.E.sport
        d.mkdir(parents=True, exist_ok=True)
        stem = f"{started.strftime('%Y%m%dT%H%M%SZ')}_{run_id}"
        (d / f"{stem}.json").write_text(json.dumps({"summary": summary, "diagnostics": diag}, indent=1, default=str))
        L = [f"# Learning run {run_id} — {iso(started)}", "",
             f"**Decision: {summary['decision']}** · champion {summary['champion_before']} → {summary['champion_after']}", "",
             f"Reason: {summary['reason']}", "", f"Trigger: {summary['trigger']} ({summary['why_due']})", ""]
        t = summary.get("confirmation_test")
        if t:
            L += ["### Champion vs best challenger (confirmation window)", "",
                  f"- mean log-loss gain {t['mean_gain']:+.4f}, 90% CI [{t['ci90'][0]:+.4f}, {t['ci90'][1]:+.4f}], p={t['p_value']:.3f}, n={t['n']}",
                  f"- Brier {t['brier_champion']:.4f} → {t['brier_challenger']:.4f}; ECE {t['ece_champion']:.4f} → {t['ece_challenger']:.4f}",
                  f"- gates: {t['gates']}", ""]
        L += ["### Candidates", "", "| candidate | why | selection log loss | walk-forward log loss | accuracy |", "|---|---|---|---|---|",
              f"| **champion {summary['champion_before']}** | baseline | {summary['champion_selection_ll']:.4f} | "
              f"{summary['champion_walk_forward']['log_loss']:.4f} | {summary['champion_walk_forward']['accuracy']:.3f} |"]
        for r in summary["candidates"]:
            if "error" in r:
                L.append(f"| {r['name']} | {r['why']} | error: {r['error']} | | |"); continue
            L.append(f"| {r['name']} | {r['why']} | {r['sel_ll']:.4f} | {r['full']['log_loss']:.4f} | {r['full']['accuracy']:.3f} |")
        L += ["", render_markdown(diag, "Error analysis of the champion (walk-forward, out-of-sample)"), ""]
        if summary.get("live_record"):
            L += ["### Logged predictions (live/blind and backtest)", "", "| origin / version | n | accuracy | Brier | log loss | margin MAE |", "|---|---|---|---|---|---|"]
            for k, s in summary["live_record"].items():
                L.append(f"| {k} | {s['n']} | {s.get('accuracy', 0):.3f} | {s.get('brier', 0):.4f} | {s.get('log_loss', 0):.4f} | {s.get('margin_mae', float('nan')):.2f} |")
        if diag.get("worst_misses"):
            L += ["", "### Largest out-of-sample misses (context only — never fitted individually)", ""]
            for w in diag["worst_misses"]:
                L.append(f"- {w['kickoff_utc'][:10]} {w['away_name']} @ {w['home_name']}: P(home)={w['p_home']:.2f}, "
                         f"pred margin {w['margin']:+.1f}, actual {w['margin_actual']:+.0f}")
        (d / f"{stem}.md").write_text("\n".join(L))
        return d / f"{stem}.md"


def _diff(a: dict, b: dict, prefix=""):
    out = {}
    for k in set(a) | set(b):
        va, vb = a.get(k), b.get(k)
        if isinstance(va, dict) and isinstance(vb, dict):
            out.update(_diff(va, vb, prefix + k + "."))
        elif va != vb:
            out[prefix + k] = {"from": va, "to": vb}
    return out
