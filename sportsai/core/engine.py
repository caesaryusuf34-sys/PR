"""Prediction engine: the single entry point used by the CLI (and by any future API / bot).

    engine.predict("Miami vs Clemson")
      1. auto_update(): ingest new schedule/results, collect results of pending predictions,
         evaluate them, and run a learning cycle if one is due (no manual retraining needed)
      2. load the latest validated (champion) model
      3. resolve teams + find the scheduled game (else a hypothetical neutral-site matchup)
      4. build point-in-time features with cutoff = min(now, kickoff)
      5. predict, explain, and store an immutable, hashed snapshot of everything used
"""
from __future__ import annotations
import json, re
import numpy as np
import pandas as pd

from ..config import Settings
from .db import Store
from .featurestore import FeatureStore
from .leakage import snapshot_hash
from .registry import ModelRegistry
from .sport import get_adapter
from .timeutil import iso, now, ts

QUERY = re.compile(r"^\s*(?:predict\s+)?(?P<a>.+?)\s+(?P<sep>vs\.?|versus|v\.?|@|at)\s+(?P<b>.+?)\s*[?.!]*\s*$", re.I)


def confidence_score(p, p_std, split, early, qb_change):
    base = 10 * np.clip(2 * np.maximum(p, 1 - p) - 1, 0, 1) ** 0.75
    return float(np.clip(base - 10 * p_std - 1.0 * split - 0.75 * early - 0.5 * qb_change, 1, 10))


class Engine:
    def __init__(self, sport: str = "ncaaf", settings: Settings | None = None):
        self.S = (settings or Settings()).ensure()
        self.store = Store(self.S.db_path)
        self.adapter = get_adapter(sport, self.S)
        self.sport = self.adapter.sport
        if getattr(self.adapter, "policy_overrides", None):     # sport-specific learning policy
            import dataclasses
            self.S.policy = dataclasses.replace(self.S.policy, **self.adapter.policy_overrides)
        self.registry = ModelRegistry(self.store, self.S.models_dir, self.sport, self.S.root)
        self.fstore = FeatureStore(self.S, self.store, self.adapter)

    def builder_candidates(self):
        return list(self.adapter.candidate_features)

    # ============================================================ data + outcomes
    def sync(self, full=False, seasons=None):
        return self.adapter.sync(self.store, full=full, seasons=seasons)

    def collect_results(self) -> int:
        """Fetch final scores for logged predictions whose games should be over."""
        p = self.store.df("""SELECT p.game_id, p.kickoff_utc FROM predictions p LEFT JOIN results r
                             ON r.sport=p.sport AND r.game_id=p.game_id
                             WHERE p.sport=? AND p.game_id IS NOT NULL AND r.game_id IS NULL""", (self.sport,))
        if p.empty:
            return 0
        p["kickoff_utc"] = pd.to_datetime(p.kickoff_utc, utc=True)
        over = p[p.kickoff_utc + pd.Timedelta(hours=self.adapter.game_duration_hours) <= now()]
        return self.adapter.refresh_results(self.store, sorted(set(over.game_id))) if len(over) else 0

    def evaluate(self) -> int:
        q = self.store.df("""SELECT p.pred_id, p.game_id, p.p_home, p.pred_margin, p.pred_total, r.home_score, r.away_score
                             FROM predictions p JOIN results r ON r.sport=p.sport AND r.game_id=p.game_id
                             WHERE p.sport=? AND p.origin IN ('live','backtest','legacy_import')
                               AND p.pred_id NOT IN (SELECT pred_id FROM prediction_annotations WHERE kind='invalid')""", (self.sport,))
        if q.empty:
            return 0
        y = (q.home_score > q.away_score).astype(int); m = q.home_score - q.away_score; t = q.home_score + q.away_score
        p = q.p_home.clip(1e-4, 1 - 1e-4)
        pick_p = np.maximum(p, 1 - p); correct = ((p >= 0.5) == (y == 1)).astype(int)
        rows = [dict(pred_id=r.pred_id, sport=self.sport, game_id=r.game_id, evaluated_utc=iso(now()),
                     home_win=int(y[i]), actual_margin=float(m[i]), actual_total=float(t[i]), correct=int(correct[i]),
                     brier=float((p[i] - y[i]) ** 2), logloss=float(-(y[i] * np.log(p[i]) + (1 - y[i]) * np.log(1 - p[i]))),
                     margin_error=float(r.pred_margin - m[i]) if pd.notna(r.pred_margin) else None,
                     abs_margin_error=float(abs(r.pred_margin - m[i])) if pd.notna(r.pred_margin) else None,
                     total_error=float(r.pred_total - t[i]) if pd.notna(r.pred_total) else None,
                     high_conf_miss=int(pick_p[i] >= self.S.policy.high_conf and not correct[i]))
                for i, r in enumerate(q.itertuples())]
        self.store.replace_evaluations(rows)
        return len(rows)

    def learner(self):
        from .learner import Learner
        return Learner(self)

    def auto_update(self, learn: bool = True, force_learn: bool = False, verbose: bool = True) -> dict:
        out = {"sync": self.sync()}
        out["results_collected"] = self.collect_results()
        out["evaluated"] = self.evaluate()
        if learn:
            L = self.learner()
            due, why = L.due()
            out["learning_due"] = why
            if due or force_learn:
                out["learning"] = L.run(trigger="forced" if force_learn else "auto", force=force_learn)
        if verbose:
            print(f"[update] {json.dumps({k: v for k, v in out.items() if k != 'learning'}, default=str)}", flush=True)
        return out

    # ============================================================ prediction
    def parse_query(self, text: str):
        m = QUERY.match(text)
        if not m:
            raise ValueError("expected 'Predict Team A vs Team B' (or 'Team A @ Team B' for A at B)")
        return m.group("a"), m.group("b"), m.group("sep").lower() in ("@", "at")

    def predict(self, text: str, *, update: bool = True, save: bool = True, learn: bool = True) -> dict:
        if update:
            self.auto_update(learn=learn)
        a_txt, b_txt, a_at_b = self.parse_query(text)
        a_id, a_name = self.adapter.resolve_team(self.store, a_txt)
        b_id, b_name = self.adapter.resolve_team(self.store, b_txt)
        t = now()
        g = self.adapter.find_game(self.store, a_id, b_id, t)
        if g is not None:
            game = dict(game_id=g.game_id, season=int(g.season), kickoff_utc=ts(g.kickoff_utc), home_id=g.home_id,
                        away_id=g.away_id, home_name=g.home_name, away_name=g.away_name, neutral=int(g.neutral),
                        conf_game=int(g.conf_game), venue=g.venue, seasontype=g.seasontype)
            origin = "live" if t < game["kickoff_utc"] else "backtest"
        else:  # hypothetical matchup: "A @ B" -> B hosts, "A vs B" -> neutral site
            home, away = (b_id, b_name), (a_id, a_name)
            from ..sports.ncaaf.adapter import current_season
            game = dict(game_id=None, season=current_season(t), kickoff_utc=t, home_id=home[0], away_id=away[0],
                        home_name=home[1], away_name=away[1], neutral=0 if a_at_b else 1, conf_game=0, venue=None, seasontype=2)
            origin = "hypothetical"
        return self.predict_games(pd.DataFrame([game]), origin=origin, query=text, save=save)[0]

    def predict_games(self, games: pd.DataFrame, *, origin: str, query: str | None = None, save: bool = True,
                      cutoff=None, context: bool = True) -> list[dict]:
        version, model, ch = self.registry.champion()
        created = now()
        g = games.copy()
        g["kickoff_utc"] = pd.to_datetime(g.kickoff_utc, utc=True)
        if cutoff is not None:
            g["cutoff_utc"] = ts(cutoff)
        else:
            g["cutoff_utc"] = g.kickoff_utc.where(g.kickoff_utc <= created, created)   # min(now, kickoff)
        if (g.cutoff_utc > g.kickoff_utc).any():
            raise ValueError("cutoff after kickoff")
        tg = g.copy(); tg["game_id"] = tg.game_id.fillna("hypothetical")
        fb = self.adapter.feature_builder(self.store)
        F = fb.build(tg[["game_id", "season", "kickoff_utc", "home_id", "away_id", "neutral", "conf_game", "cutoff_utc"]])
        # the builder returns rows sorted by cutoff; re-align them with the input game order
        F = F.set_index("game_id").loc[tg.game_id.values].reset_index()
        F = F.merge(tg[["game_id", "home_name", "away_name", "seasontype"]], on="game_id", how="left")
        assert (F.game_id.values == tg.game_id.values).all() and (F.home_id.values == tg.home_id.values).all()
        P = model.predict(F)
        out = []
        for i in range(len(F)):
            row, pr, gi = F.iloc[i], P.iloc[i], g.iloc[i]
            p = float(pr.p_home)
            conf = confidence_score(p, float(pr.p_std), int(pr.models_split), float(row.early), float(row.qb_change_h + row.qb_change_a > 0))
            total = float(pr.total) if pd.notna(pr.total) else np.nan
            margin = float(pr.margin)
            ph, pa = max((total + margin) / 2, 0), max((total - margin) / 2, 0)
            expl = self.adapter.explain(row, pr, model)
            ctx = self.adapter.pregame_context(gi.game_id) if (context and origin == "live" and gi.game_id) else {}
            if ctx and hasattr(self.adapter, "augment_explanation"):
                expl = self.adapter.augment_explanation(expl, ctx, gi)
            is_blind = int(origin == "live" and created < gi.kickoff_utc)
            feats = {f: (None if pd.isna(row.get(f)) else float(row.get(f))) for f in self.builder_candidates() + list(model.features) if f in row}
            members = {k: float(v) for k, v in pr.items() if (k.startswith("p_") or k.startswith("m_")) and pd.notna(v)}
            snap = {"features": feats, "cutoff_utc": iso(row.cutoff_utc), "model_version": version,
                    "feature_version": row.feature_version, "game_id": gi.game_id, "home_id": gi.home_id, "away_id": gi.away_id}
            rec = dict(sport=self.sport, game_id=gi.game_id, query=query, home_id=gi.home_id, away_id=gi.away_id,
                       home_name=gi.home_name, away_name=gi.away_name, neutral=int(gi.neutral),
                       kickoff_utc=iso(gi.kickoff_utc), created_utc=iso(created), data_cutoff_utc=iso(row.cutoff_utc),
                       origin=origin, is_blind=is_blind, model_version=version, feature_version=row.feature_version,
                       p_home=p, pred_margin=margin, pred_total=total, proj_home=ph, proj_away=pa, confidence=conf,
                       upset_prob=float(min(p, 1 - p)), member_preds_json=json.dumps(members),
                       features_json=json.dumps(feats), feature_names_json=json.dumps(model.features),
                       explanation_json=json.dumps(expl, default=str), context_json=json.dumps(ctx, default=str),
                       snapshot_sha256=snapshot_hash(snap))
            if save:
                dup = self.store.df("""SELECT pred_id FROM predictions WHERE sport=? AND game_id IS ? AND model_version=?
                                       AND origin=? AND features_json=? AND home_id=? AND away_id=?""",
                                    (self.sport, gi.game_id, version, origin, rec["features_json"], gi.home_id, gi.away_id))
                if len(dup):   # same game, model and identical inputs -> identical prediction; keep the original row
                    rec["pred_id"], rec["duplicate_of_existing"] = dup.pred_id.iloc[0], True
                else:
                    rec["pred_id"] = self.store.insert_prediction(rec)
            rec["explanation"], rec["context"], rec["members"] = expl, ctx, members
            rec["venue"] = gi.get("venue")
            out.append(rec)
        return out

    # ============================================================ reporting
    @staticmethod
    def format(rec: dict) -> str:
        H, A = rec["home_name"], rec["away_name"]; p = rec["p_home"]
        winner = H if p >= 0.5 else A
        e = rec["explanation"]
        tag = {"live": "BLIND pre-game prediction", "backtest": "backtest (game already started — not blind)",
               "hypothetical": "hypothetical matchup (no scheduled game found)"}.get(rec["origin"], rec["origin"])
        L = [f"{A} @ {H}" + (" (neutral site)" if rec["neutral"] else ""),
             f"[{tag} · model {rec['model_version']} · data cutoff {rec['data_cutoff_utc']} · kickoff {rec['kickoff_utc']}]", "",
             f"Prediction: {winner}",
             f"Win Probability: {A} {100 * (1 - p):.0f}% / {H} {100 * p:.0f}%",
             f"Projected Score: {A} {rec['proj_away']:.0f}–{rec['proj_home']:.0f} {H}",
             f"Projected Margin: {abs(rec['pred_margin']):.1f}",
             f"Upset Probability: {100 * rec['upset_prob']:.0f}%",
             f"Confidence: {rec['confidence']:.1f}/10", "", "Key Factors:", ""]
        L += [f"{i}. {f}" for i, f in enumerate(e["key_factors"], 1)]
        L += ["", f"Main Risk: {e['main_risk']}", "", "Model drivers (gradient-boosting SHAP, log-odds toward):"]
        L += [f"  - {d['label']}: {d['log_odds']:+.2f} → {d['favours']}" for d in e.get("drivers", [])[:5]]
        mp = {k: v for k, v in rec["members"].items() if k.startswith("p_") and k not in ("p_home", "p_raw", "p_std")}
        L += ["", "Member P(" + H + " win): " + ", ".join(f"{k[2:]} {v:.3f}" for k, v in mp.items()) + f" · ensemble {p:.3f}"]
        for k, v in e.get("notes", {}).items():
            L.append(f"  {k}: {v}")
        if rec.get("context"):
            c = rec["context"]
            if c.get("weather"):
                L.append(f"  weather (forecast): {c['weather']}")
        if rec.get("pred_id"):
            L.append(f"\n{'already logged' if rec.get('duplicate_of_existing') else 'logged'} as prediction {rec['pred_id']} "
                     f"(snapshot sha256 {rec['snapshot_sha256'][:16]}…)")
        return "\n".join(L)

    def predict_week(self, save: bool = True) -> tuple[list[dict], pd.DataFrame]:
        """Predict every not-yet-started game on the league's current-week scoreboard."""
        ids = self.adapter.current_week_game_ids()
        g = self.store.games(self.sport)
        g = g[g.game_id.isin(ids)].sort_values(["kickoff_utc", "game_id"])
        res = set(self.store.df("SELECT game_id FROM results WHERE sport=?", (self.sport,)).game_id)
        started = g[(g.kickoff_utc <= now()) | g.game_id.isin(res)]
        todo = g[~g.game_id.isin(started.game_id)]
        recs = self.predict_games(todo, origin="live", save=save) if len(todo) else []
        return recs, started

    @staticmethod
    def week_report(recs: list[dict], excluded: pd.DataFrame, title: str, validation: dict | None = None) -> str:
        L = [f"# {title}", ""]
        if recs:
            L.append(f"- Model **{recs[0]['model_version']}** · predictions frozen {recs[0]['created_utc']} (before every kickoff listed) · "
                     f"all inputs are pre-game; results/injuries after the cutoff never enter")
        if validation:
            w = validation.get("walk_forward", {})
            L.append(f"- Walk-forward validation ({w.get('n')} out-of-sample games): accuracy {w.get('accuracy', 0):.1%}, "
                     f"log loss {w.get('log_loss', 0):.3f}, Brier {w.get('brier', 0):.3f}, margin MAE {w.get('margin_mae', 0):.1f}")
        if len(excluded):
            L.append("- Not predicted (already started or final at freeze): " + "; ".join(f"{r.away_name} @ {r.home_name}" for r in excluded.itertuples()))
        L.append("- Betting lines are not model inputs. Injury reports are shown as risk context only (no historical injury data to train on).")
        L += ["", "## Games", ""]
        for r in recs:
            L += ["```", Engine.format(r), "```", ""]
        L += ["## Summary", "", "| Game | Kickoff (UTC) | Pick | Win % | Projected score | Upset % | Confidence | Flags |", "|---|---|---|---|---|---|---|---|"]
        for r in recs:
            p = r["p_home"]; pick = r["home_name"] if p >= 0.5 else r["away_name"]
            flags = []
            if int(r["members"].get("models_split", 0) if isinstance(r["members"], dict) else 0): flags.append("split")
            if r["explanation"].get("injury_flags"): flags.append("injuries: " + ", ".join(r["explanation"]["injury_flags"][:2]))
            L.append(f"| {r['away_name']} @ {r['home_name']}{' (N)' if r['neutral'] else ''} | {r['kickoff_utc'][5:16].replace('T', ' ')} | {pick} | "
                     f"{100 * max(p, 1 - p):.0f}% | {r['away_name']} {r['proj_away']:.0f}–{r['proj_home']:.0f} {r['home_name']} | "
                     f"{100 * r['upset_prob']:.0f}% | {r['confidence']:.1f} | {'; '.join(flags)} |")
        srt = sorted(recs, key=lambda r: -max(r["p_home"], 1 - r["p_home"]))
        L += ["", "**Most confident:** " + ", ".join(f"{(r['home_name'] if r['p_home'] >= .5 else r['away_name'])} ({100 * max(r['p_home'], 1 - r['p_home']):.0f}%)" for r in srt[:4])]
        L += ["", "**Closest:** " + ", ".join(f"{r['away_name']} @ {r['home_name']} ({100 * max(r['p_home'], 1 - r['p_home']):.0f}%)" for r in srt[::-1][:4])]
        ups = [r for r in recs if r["upset_prob"] >= 0.35]
        L += ["", "**Upset candidates (underdog ≥ 35%):** " + (", ".join(f"{(r['away_name'] if r['p_home'] >= .5 else r['home_name'])} {100 * r['upset_prob']:.0f}%" for r in sorted(ups, key=lambda r: -r['upset_prob'])) or "none")]
        dis = sorted(recs, key=lambda r: -np.std([v for k, v in r["members"].items() if k.startswith("p_") and k not in ("p_home", "p_raw", "p_std")]))
        L += ["", "**Highest model disagreement:** " + ", ".join(f"{r['away_name']} @ {r['home_name']}" for r in dis[:3])]
        return "\n".join(L)

    def scorecard(self, since=None, title=None) -> str:
        """Markdown scorecard: graded picks (latest blind pick per game & model), metrics, pending picks."""
        from .metrics import summarize
        P = self.store.predictions(self.sport)
        P = P[P.origin.isin(["live", "legacy_import"]) & P.game_id.notna()]
        if since is not None:
            P = P[pd.to_datetime(P.kickoff_utc, utc=True) >= ts(since)]
        P = P.sort_values("created_utc").drop_duplicates(["game_id", "model_version"], keep="last")
        R = self.store.df("SELECT game_id, home_score, away_score FROM results WHERE sport=?", (self.sport,))
        M = P.merge(R, on="game_id", how="left").sort_values("kickoff_utc")
        done, pend = M[M.home_score.notna()], M[M.home_score.isna()]
        L = [f"# {title or self.sport.upper() + ' scorecard'}", ""]
        if len(done):
            y = (done.home_score > done.away_score).astype(int).values
            s = summarize(done.p_home.values, y, done.pred_margin.values, (done.home_score - done.away_score).values,
                          done.pred_total.values, (done.home_score + done.away_score).values)
            exp = float(np.maximum(done.p_home, 1 - done.p_home).sum())
            L += [f"**Graded: {int(((done.p_home >= .5) == (y == 1)).sum())}/{len(done)} correct** "
                  f"(expected {exp:.1f} from the stated probabilities) · Brier {s['brier']:.3f} · log loss {s['log_loss']:.3f} · "
                  f"margin MAE {s['margin_mae']:.1f} · total MAE {s['total_mae']:.1f}", "",
                  "| Game | Final | Pick (prob) | Projected | Result |", "|---|---|---|---|---|"]
            for r in done.itertuples():
                pick = r.home_name if r.p_home >= .5 else r.away_name
                ok = (r.p_home >= .5) == (r.home_score > r.away_score)
                L.append(f"| {r.away_name} @ {r.home_name} | {int(r.away_score)}–{int(r.home_score)} | {pick} ({100 * max(r.p_home, 1 - r.p_home):.0f}%) | "
                         f"{r.proj_away:.0f}–{r.proj_home:.0f} | {'✅' if ok else '❌'} |")
        if len(pend):
            L += ["", "## Not final yet (latest blind prediction)", "", "| Game | Kickoff (UTC) | Pick | Win % | Projected | Predicted at (UTC) |", "|---|---|---|---|---|---|"]
            for r in pend.itertuples():
                pick = r.home_name if r.p_home >= .5 else r.away_name
                L.append(f"| {r.away_name} @ {r.home_name} | {r.kickoff_utc[5:16].replace('T', ' ')} | {pick} | {100 * max(r.p_home, 1 - r.p_home):.0f}% | "
                         f"{r.proj_away:.0f}–{r.proj_home:.0f} | {r.created_utc[5:16].replace('T', ' ')} |")
        return "\n".join(L)

    def status(self) -> dict:
        v = self.registry.versions()
        ev = self.store.evaluated(self.sport)
        pend = self.store.df("""SELECT COUNT(*) n FROM predictions p LEFT JOIN results r ON r.sport=p.sport AND r.game_id=p.game_id
                                WHERE p.sport=? AND p.game_id IS NOT NULL AND r.game_id IS NULL""", (self.sport,)).n.iloc[0]
        from .analysis import live_record
        due, why = self.learner().due()
        return {"versions": v[["version", "status", "kind", "created_utc", "train_cutoff_utc", "n_train", "reason"]].to_dict("records"),
                "pending_predictions": int(pend), "record": live_record(ev, self.S.policy), "learning_due": why}
