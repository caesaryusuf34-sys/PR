"""Baseball adapter shared by MLB, KBO, NPB and CPBL: ingestion through a league source, announced-starter
bookkeeping, team resolution, game lookup, explanations and confidence. League specifics (data source,
time zone, constants, names, learning policy) are class attributes set in leagues.py."""
from __future__ import annotations
import concurrent.futures as cf
import difflib, gzip, json, re, unicodedata
from datetime import timedelta
import numpy as np
import pandas as pd

from ...core.leakage import TemporalGuard
from ...core.sport import SportAdapter
from ...core.timeutil import iso, now
from .http import Http
from .features import BaseballFeatureBuilder

FEATURE_LABELS = {
    "elo_diff": "Elo rating edge (incl. home field)", "pts_margin_pred": "Opponent-adjusted run model (incl. starters)",
    "pts_total_pred": "Run environment", "team_diff": "Team run differential (ex-starter, opp-adjusted)",
    "sp_eff_diff": "Starting pitcher run-prevention rating", "sp_fip_diff": "Starter FIP-type rate",
    "sp_kbb_diff": "Starter strikeout-minus-walk rate", "sp_ra9_diff": "Starter runs allowed per 9",
    "sp_depth_diff": "Starter innings per start", "pen_fip_diff": "Bullpen FIP-type rate",
    "pen_load_diff": "Bullpen workload, last 3 days", "sp_known_h": "Home starter announced",
    "sp_known_a": "Away starter announced", "sp_rest_h": "Home starter rest", "sp_rest_a": "Away starter rest",
    "form_diff": "Recent form vs expectation (last 10)", "rest_diff": "Team rest", "early": "Early-season uncertainty",
    "n_prior_h": "Home games played", "n_prior_a": "Away games played", "conf_game": "Same-league game"}


EDGE_LABELS = {"sp": "starting pitching", "team": "lineup & defence", "pen": "bullpen", "home": "home field", "form": "recent form"}


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", str(s))
    return re.sub(r"[^\w&]+", " ", s.lower()).strip()


class BaseballAdapter(SportAdapter):
    sport = "baseball"
    league_name = "Baseball"
    source_cls = None
    builder_cls = BaseballFeatureBuilder
    local_tz = "America/New_York"
    game_duration_hours = 4.5
    first_train_season = 2022
    void_ties = True                    # a tied game voids a win/loss pick
    start_label = "first pitch"
    score_decimals = 1
    sync_back_days, sync_ahead_days = 4, 7
    team_aliases: dict = {}
    policy_overrides: dict = {}
    subgroup_features = {"early_season": ["early", "n_prior_h", "n_prior_a"], "starter_unknown_any": ["sp_known_h", "sp_known_a"],
                         "interleague": ["conf_game"], "tired_bullpen": ["pen_load_diff"]}
    has_ties = False
    tie_note = ""

    def __init__(self, settings):
        self.settings = settings
        self.http = Http(settings.user_agent)
        self.src = self.source_cls(self.http)
        self.guard = TemporalGuard(self.game_duration_hours)
        self.cache = settings.cache_dir / self.sport
        (self.cache / "box").mkdir(parents=True, exist_ok=True)
        (self.cache / "seasons").mkdir(parents=True, exist_ok=True)

    @property
    def db_file(self) -> str:
        return f"{self.sport}.db"

    @property
    def feature_builder_version(self):
        return self.builder_cls.feature_version

    @property
    def candidate_features(self):
        return self.builder_cls.candidate_features

    def current_season(self, t: pd.Timestamp) -> int:
        return pd.Timestamp(t).tz_convert(self.local_tz).year

    def local_today(self):
        return now().tz_convert(self.local_tz).date()

    # ================================================================== ingestion
    def _season_rows(self, season: int, t: pd.Timestamp) -> dict:
        """A whole season. Completed seasons are immutable and cached on disk."""
        fn = self.cache / "seasons" / f"{season}.json.gz"
        if season < self.current_season(t) and fn.exists():
            return json.load(gzip.open(fn, "rt"))
        rows = self.src.fetch_season(season, iso(t), t)
        if season < self.current_season(t):
            json.dump(rows, gzip.open(fn, "wt"))
        return rows

    def sync(self, store, *, full=False, seasons=None, stats=True) -> dict:
        t0 = now(); stamp = iso(t0)
        if full or seasons:
            seasons = seasons or list(range(self.builder_cls.FIRST_SEASON, self.current_season(t0) + 1))
            parts = []
            for s in seasons:
                print(f"[sync {self.sport}] season {s}", flush=True)
                parts.append(self._season_rows(s, t0))
            from .sources import merge_rows
            rows = merge_rows(parts)
        else:
            today = self.local_today()
            rows = self.src.fetch_range(today - timedelta(days=self.sync_back_days), today + timedelta(days=self.sync_ahead_days), stamp, t0)
        out = self.ingest(store, rows, stamp)
        out["seasons"] = seasons
        if stats:
            out["stats_new"] = self.sync_stats(store)
        store.log(self.sport, "sync", json.dumps(out, default=str))
        return out

    def ingest(self, store, rows: dict, stamp: str) -> dict:
        S = self.sport
        teams = list({r["team_id"]: dict(r, sport=S, updated_utc=stamp) for r in rows["teams"]}.values())
        games = list({r["game_id"]: dict(r, sport=S, feed_group=None, first_seen_utc=stamp, updated_utc=stamp)
                      for r in rows["games"]}.values())
        res = list({r["game_id"]: dict(r, sport=S, collected_utc=stamp, source=self.src.__class__.__name__) for r in rows["results"]}.values())
        known = set(store.df("SELECT game_id FROM results WHERE sport=?", (S,)).game_id)
        store.upsert_teams(teams)
        store.upsert_games(games)
        store.upsert_results([r for r in res if r["game_id"] not in known])
        with store.tx() as c:   # corrections keep the first collection timestamp
            c.executemany("UPDATE results SET home_score=?, away_score=?, status=? WHERE sport=? AND game_id=?",
                          [(r["home_score"], r["away_score"], r["status"], S, r["game_id"]) for r in res if r["game_id"] in known])
        st = self.resolve_starter_ids(store, rows["starters"])
        n_st = store.add_starters(S, [r for r in st if r["source"] == "backfill"], stamp)
        n_st += store.add_starters(S, [r for r in st if r["source"] == "probable"], stamp)
        if rows["stats"]:
            self._write_stats(store, rows["stats"], stamp)
        return {"games": len(games), "results_new": len([r for r in res if r["game_id"] not in known]), "starter_updates": n_st}

    def resolve_starter_ids(self, store, starters: list[dict]) -> list[dict]:
        """Sources that announce starters by name only (NPB) are mapped to player ids seen in the team's
        past box scores; an unmatched name gets a stable name-based id (a pitcher with no history)."""
        need = [r for r in starters if r.get("pitcher_id") is None and r.get("pitcher_name")]
        if not need:
            return starters
        st = store.team_game_stats(self.sport)
        names = {}
        if len(st):
            st = st.sort_values("kickoff_utc")
            for tid, sid, nm, staff in zip(st.team_id, st.sp_id, st.sp_name, st.get("staff", pd.Series([None] * len(st)))):
                if isinstance(staff, list):
                    for pid, pn in staff:
                        names[(tid, _norm(pn))] = str(pid)
                if sid is not None and sid == sid and nm:
                    names[(tid, _norm(nm))] = str(sid)
        out = []
        for r in starters:
            if r.get("pitcher_id") is None and r.get("pitcher_name"):
                r = dict(r, pitcher_id=names.get((r["team_id"], _norm(r["pitcher_name"]))) or f"name:{r['team_id']}:{r['pitcher_name']}")
            out.append(r)
        return out

    def _write_stats(self, store, rows, stamp):
        store.upsert_team_game_stats([{"sport": self.sport, "game_id": r["game_id"], "team_id": r["team_id"],
                                       "stats_json": json.dumps(r["stats"], default=float, ensure_ascii=False),
                                       "stats_version": "bb-s1", "collected_utc": stamp} for r in rows])

    def _box(self, game: dict):
        fn = self.cache / "box" / f"{re.sub(r'[^A-Za-z0-9._-]', '_', game['game_id'])}.json.gz"
        if fn.exists():
            return json.load(gzip.open(fn, "rt"))
        try:
            b = self.src.fetch_box(game)
        except RuntimeError:
            return None
        if b:
            json.dump(b, gzip.open(fn, "wt"), ensure_ascii=False)
        return b

    def sync_stats(self, store, game_ids=None, threads: int = 8) -> int:
        """Box scores (post-game) for completed games that have none yet; also records the actual starter
        as the 'backfill' starter where the source announces none historically (NPB, CPBL)."""
        if not self.src.needs_box:
            return 0
        done = store.completed_games(self.sport)
        have = set(store.df("SELECT DISTINCT game_id FROM team_game_stats WHERE sport=?", (self.sport,)).game_id)
        todo = done[(done.season >= self.builder_cls.FIRST_SEASON) & ~done.game_id.isin(have)]
        if game_ids is not None:
            todo = todo[todo.game_id.isin(game_ids)]
        if todo.empty:
            return 0
        recs = todo[["game_id", "season", "home_id", "away_id", "source_ref"]].to_dict("records")
        if len(recs) > 50:
            print(f"[sync {self.sport}] box scores for {len(recs)} games", flush=True)
        with cf.ThreadPoolExecutor(threads) as ex:
            boxes = list(ex.map(self._box, recs))
        stamp = iso(now()); rows, bf = [], []
        for g, b in zip(recs, boxes):
            if not b:
                continue
            for side, tid in (("home", g["home_id"]), ("away", g["away_id"])):
                rows.append({"game_id": g["game_id"], "team_id": tid, "stats": b[side]})
                if b[side].get("sp_id"):
                    bf.append(dict(game_id=g["game_id"], side=side, team_id=tid, pitcher_id=b[side]["sp_id"],
                                   pitcher_name=b[side].get("sp_name"), source="backfill"))
        self._write_stats(store, rows, stamp)
        store.add_starters(self.sport, bf, stamp)
        return len(rows)

    def refresh_results(self, store, game_ids) -> int:
        g = store.games(self.sport)
        g = g[g.game_id.isin(game_ids)]
        if g.empty:
            return 0
        days = g.kickoff_utc.dt.tz_convert(self.local_tz).dt.date
        t = now()
        rows = self.src.fetch_range(days.min(), days.max(), iso(t), t)
        self.ingest(store, rows, iso(t))
        self.sync_stats(store, game_ids=list(game_ids))
        res = set(store.df("SELECT game_id FROM results WHERE sport=?", (self.sport,)).game_id)
        return int(sum(gid in res for gid in game_ids))

    def before_predict(self, store, games: pd.DataFrame):
        """Freeze the announced-starter state used by a live prediction: every side of every game gets a
        'probable' row captured before the prediction (TBD if nothing is announced), so the logged features
        can be rebuilt exactly later even after the actual starter is known."""
        if games.empty:
            return
        cur = store.starters(self.sport)
        cur = cur[(cur.source == "probable") & cur.game_id.isin(set(games.game_id))]
        have = set(zip(cur.game_id, cur.side))
        rows = [dict(game_id=g.game_id, side=side, team_id=tid, pitcher_id=None, pitcher_name=None, source="probable")
                for g in games.itertuples() for side, tid in (("home", g.home_id), ("away", g.away_id)) if (g.game_id, side) not in have]
        if rows:
            store.add_starters(self.sport, rows, iso(now()))

    def playable(self, games: pd.DataFrame) -> pd.Series:
        return ~games.status.fillna("").str.lower().str.contains("cancel|postpon|reserve|suspend")

    # ================================================================== lookup
    def resolve_team(self, store, text: str) -> tuple[str, str]:
        teams = store.df("SELECT * FROM teams WHERE sport=?", (self.sport,))
        if teams.empty:
            raise LookupError(f"no {self.sport} teams in the database; run sync/bootstrap first")
        aliases = {_norm(k): v for k, v in self.team_aliases.items()}
        q = _norm(aliases.get(_norm(text), text))
        fields = ["display_name", "location", "name", "abbr", "short_name", "team_id"]
        pool = {}
        for r in teams.itertuples():
            for f in fields:
                v = getattr(r, f)
                if v:
                    pool.setdefault(_norm(v), set()).add((r.team_id, r.display_name))
        if q in pool and len(pool[q]) == 1:
            return next(iter(pool[q]))
        hits = {x for k, v in pool.items() if q and (q in k.split() or k.endswith(" " + q) or k == q) for x in v}
        if len(hits) == 1:
            return next(iter(hits))
        m = difflib.get_close_matches(q, list(pool), n=1, cutoff=0.6)
        if m and len(pool[m[0]]) == 1:
            return next(iter(pool[m[0]]))
        raise LookupError(f"team not found or ambiguous: {text!r}")

    def find_game(self, store, team_a, team_b, after, days=21):
        def search():
            g = store.games(self.sport)
            res = set(store.df("SELECT game_id FROM results WHERE sport=?", (self.sport,)).game_id)
            m = (((g.home_id == team_a) & (g.away_id == team_b)) | ((g.home_id == team_b) & (g.away_id == team_a)))
            m &= (g.kickoff_utc > after - pd.Timedelta(hours=self.game_duration_hours)) & (g.kickoff_utc <= after + pd.Timedelta(days=days))
            m &= ~g.game_id.isin(res)
            g = g[m]
            return g[self.playable(g)].sort_values("kickoff_utc")
        hit = search()
        return None if hit.empty else hit.iloc[0]

    def pregame_context(self, game_id: str) -> dict:
        return {"captured_utc": iso(now())}

    # ================================================================== modelling hooks
    def feature_builder(self, store):
        return self.builder_cls.from_store(store, self.guard, self.sport)

    def default_config(self) -> dict:
        gbm = dict(n_estimators=250, learning_rate=0.02, num_leaves=7, min_child_samples=80, reg_lambda=10.0)
        fb = self.builder_cls
        return {"features": list(fb.model_features), "total_features": list(fb.total_features),
                "explain_features": list(fb.model_features),
                "members": {"elo": {}, "scoring": {}, "logistic": {"C": 0.05}, "gbm": dict(gbm),
                            "margin_ridge": {"alpha": 30.0}, "gbm_margin": dict(gbm)},
                "ensemble": "simplex", "stack_context": [], "calibration": "none",
                "half_life_days": None, "min_train_season": self.first_train_season, "holdout_frac": 0.2}

    def subgroups(self, df: pd.DataFrame) -> dict:
        p = df.p_home; fav_home = p >= 0.5
        out = {"home_favorite": fav_home, "road_favorite": ~fav_home, "big_favorite_65": (p >= 0.65) | (p <= 0.35),
               "toss_up_45_55": (p > 0.45) & (p < 0.55), "early_season": df.early == 1,
               "starter_unknown_any": (df.sp_known_h + df.sp_known_a) < 2,
               "postseason": df.get("seasontype", pd.Series(2, index=df.index)) == 3}
        if df.conf_game.nunique() > 1:
            out["interleague"] = df.conf_game == 0
        if "pen_load_diff" in df and df.pen_load_diff.abs().sum() > 0:
            out["tired_bullpen"] = df.pen_load_diff.abs() >= 1.5 * df.pen_load_diff.std()
        return out

    def confidence(self, p: float, pr, row) -> float:
        """1-10 relative to what is achievable in baseball (a 70% pick is already a strong edge):
        lowered for member disagreement, an unannounced starter and early-season samples."""
        base = 10 * np.clip((max(p, 1 - p) - 0.5) / 0.22, 0, 1) ** 0.8
        unknown = 2 - float(row.get("sp_known_h", 1)) - float(row.get("sp_known_a", 1))
        return float(np.clip(base - 12 * float(pr.p_std) - 1.0 * int(pr.models_split) - 1.25 * unknown
                             - 0.75 * float(row.get("early", 0)), 1, 10))

    # ================================================================== explanations
    def _sp_label(self, row, side):
        nm = row.get(f"sp_name_{side}") or row.get(f"sp_id_{side}")
        if not row.get(f"sp_known_{side}"):
            return "TBD (rotation average)"
        nm = str(nm)
        return nm[5:].split(":", 1)[-1] if nm.startswith("name:") else nm

    def explain(self, row: pd.Series, pred: pd.Series, model) -> dict:
        H, A = row.home_name, row.away_name
        p = float(pred.p_home); sgn = 1 if p >= 0.5 else -1
        fav, dog = (H, A) if p >= 0.5 else (A, H)
        lines = self.builder_cls.HAS_LINES
        sph, spa = self._sp_label(row, "h"), self._sp_label(row, "a")
        # run-equivalent edges (+ = home), each translated into a sentence
        edges = []
        sp_runs = float(row.sp_eff_diff)
        if lines:
            sp_runs = 0.5 * sp_runs + 0.5 * float(row.sp_fip_diff) * float(5.5 + 0.5 * (row.sp_depth_h + row.sp_depth_a)) / 9
        edges.append({"key": "sp", "runs": sp_runs, "text": (
            f"Starting pitching: {spa} ({A}) vs {sph} ({H}) — "
            + (f"FIP-type vs league {row.sp_fip_a:+.2f} vs {row.sp_fip_h:+.2f} per 9, K-BB/9 {row.sp_kbb_a:+.2f} vs {row.sp_kbb_h:+.2f}, "
               f"starts this season {row.sp_starts_a:.0f} vs {row.sp_starts_h:.0f}; " if lines else "")
            + f"run-prevention rating {row.sp_eff_a:+.2f} vs {row.sp_eff_h:+.2f} runs/game (lower is better)")})
        edges.append({"key": "team", "runs": float(row.team_diff), "text": (
            f"Lineup & defence: opponent-adjusted runs scored/allowed per game (ex-starter) — {H} {row.off_h:+.2f}/{row.def_h:+.2f}, "
            f"{A} {row.off_a:+.2f}/{row.def_a:+.2f}")})
        if lines:
            pen_runs = float(row.pen_fip_diff) * 3.0 / 9
            edges.append({"key": "pen", "runs": pen_runs, "text": (
                f"Bullpen: FIP-type vs league {H} {row.pen_fip_h:+.2f}, {A} {row.pen_fip_a:+.2f} per 9; relief innings last 3 days "
                f"{H} {row.pen_load_h:.1f}, {A} {row.pen_load_a:.1f}")})
        if not row.neutral:
            edges.append({"key": "home", "runs": float(row.get("hfa_runs", 0.0)), "text": f"Home field: {H} (league home edge {row.get('hfa_runs', 0):+.2f} runs/game)"})
        edges.append({"key": "form", "runs": 0.25 * float(row.form_diff), "text": f"Form (last 10, runs vs Elo expectation): {H} {row.form10_h:+.2f}, {A} {row.form10_a:+.2f}"})
        for e in edges:
            e["edge_team"] = H if e["runs"] > 0 else A
            e["favours_pick"] = bool(e["runs"] * sgn > 0)
        edges.sort(key=lambda e: -abs(e["runs"]))
        factors = [f"{e['text']} → edge {e['edge_team']} ({abs(e['runs']):.2f} runs)" for e in edges[:3]]
        drivers = []
        C = model.contributions(row.to_frame().T)
        if C is not None:
            c = C.iloc[0].sort_values(key=np.abs, ascending=False)
            for f, v in c.head(6).items():
                if abs(v) >= 0.01:
                    drivers.append({"feature": f, "label": FEATURE_LABELS.get(f, f), "log_odds": float(v), "favours": H if v > 0 else A})
        risks = []
        tbd = [t for t, s in ((H, "h"), (A, "a")) if not row.get(f"sp_known_{s}")]
        if tbd:
            risks.append(f"starter not announced for {', '.join(tbd)} — rotation average used")
        if int(pred.models_split):
            risks.append("ensemble members disagree on the winner")
        counters = [e for e in edges if not e["favours_pick"] and abs(e["runs"]) >= 0.1]
        if counters:
            risks.append(f"{dog}'s best counter: {EDGE_LABELS[counters[0]['key']]} ({abs(counters[0]['runs']):.2f} runs)")
        if lines:
            mu, sd = model.feature_stats.get("pen_load_diff", (0.0, 3.5))
            z = (float(row.pen_load_diff) - mu) / (sd or 1.0)
            if abs(z) >= 1.5:   # one bullpen clearly more worked than the other over the last 3 days
                risks.append(f"{A if z > 0 else H} bullpen heavily used in the last 3 days "
                             f"({max(row.pen_load_a, row.pen_load_h):.1f} relief innings)")
        if row.get("early", 0):
            risks.append("early-season sample: ratings lean on last season")
        risks.append(f"single-game variance is large in baseball (run-margin sigma ≈ {getattr(model, 'sigma', 4.2):.1f})")
        notes = {
            "starters": f"{A}: {spa} (rest {row.sp_rest_a:.0f} d) · {H}: {sph} (rest {row.sp_rest_h:.0f} d)",
            "home_field": "neutral site" if row.neutral else f"{H} at home",
            "rest": f"team rest days: {H} {row.rest_h:.0f}, {A} {row.rest_a:.0f}; games in last 7 days: {H} {row.g7_h:.0f}, {A} {row.g7_a:.0f}",
            "park": f"park run factor vs league {row.park_h:+.2f} runs/game",
        }
        if self.has_ties:
            notes["ties"] = self.tie_note
        return {"key_factors": factors, "drivers": drivers, "matchups": edges, "main_risk": "; ".join(risks[:3]), "notes": notes}
