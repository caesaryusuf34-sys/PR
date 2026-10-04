"""NCAA football adapter: ESPN ingestion, team resolution, game lookup, features, explanations."""
from __future__ import annotations
import concurrent.futures as cf
import difflib, gzip, json, re, unicodedata
from pathlib import Path
import numpy as np
import pandas as pd

from ...core.leakage import TemporalGuard
from ...core.sport import SportAdapter
from ...core.timeutil import iso, now
from .espn import ESPN
from .features import EFF, MODEL_FEATURES, TOTAL_FEATURES, FIRST_SEASON, FIRST_STATS_SEASON, NCAAFFeatureBuilder
from .parse import compact_summary, scoreboard_rows, num
from .stats import STATS_VERSION, EPModel, plays_frame, team_game_stats

SPORT = "ncaaf"
ALIASES = {"pitt": "Pittsburgh", "ole miss": "Ole Miss", "miami fl": "Miami", "miami (fl)": "Miami", "umass": "UMass",
           "uconn": "UConn", "fau": "Florida Atlantic", "fiu": "Florida International", "ucf": "UCF", "smu": "SMU",
           "the u": "Miami", "bama": "Alabama", "a&m": "Texas A&M", "tamu": "Texas A&M", "osu": "Ohio State",
           "psu": "Penn State", "vt": "Virginia Tech", "gt": "Georgia Tech", "unc": "North Carolina",
           "nc state": "NC State", "wazzu": "Washington State", "sjsu": "San José State", "usf": "South Florida",
           "ulm": "UL Monroe", "louisiana-lafayette": "Louisiana", "ul lafayette": "Louisiana"}
LABELS = {"epa_pp": "Overall offense vs defense (opp-adj EPA/play)", "rush_epa": "Run game vs run defense (opp-adj rush EPA/play)",
          "pass_epa": "Pass game vs pass defense (opp-adj pass EPA/play)", "expl": "Explosive-play matchup",
          "sr": "Success-rate matchup", "ppo": "Finishing drives (pts per scoring opportunity)",
          "start_fp": "Field position / special teams", "sack_rate": "Pass protection vs pass rush (sack rate)",
          "to_rate": "Turnover tendencies", "ypp": "Yards-per-play matchup", "rush_sr": "Rushing success matchup",
          "pass_sr": "Passing success matchup", "pts_per_drive": "Points-per-drive matchup"}
FEATURE_LABELS = {"elo_diff": "Elo power-rating edge (incl. home field)", "pts_margin_pred": "Opponent-adjusted scoring margin",
                  "pts_total_pred": "Scoring environment", "form_diff": "Recent form vs expectation", "rest_diff": "Rest advantage",
                  "qb_share_h": "Home QB continuity", "qb_share_a": "Away QB continuity", "qb_change_h": "Home QB change",
                  "qb_change_a": "Away QB change", "neutral": "Neutral site", "conf_game": "Conference game",
                  "fbs_h": "Home team FBS", "fbs_a": "Away team FBS", "early": "Early-season uncertainty",
                  "n_prior_h": "Home games played", "n_prior_a": "Away games played",
                  **{f"mx_{k}": v for k, v in LABELS.items()}}
# metrics where a HIGHER offensive value is BAD for the offense
BAD_HIGH = {"sack_rate", "to_rate"}


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9&()]+", " ", s.lower()).strip()


def current_season(t: pd.Timestamp) -> int:
    return t.year if t.month >= 7 else t.year - 1


class NCAAFAdapter(SportAdapter):
    sport = SPORT
    first_train_season = 2022
    policy_overrides: dict = {}
    full_names = False
    league_path = "football/college-football"
    builder_cls = NCAAFFeatureBuilder
    game_duration_hours = 4.5
    @property
    def feature_builder_version(self):
        return self.builder_cls.feature_version

    @property
    def candidate_features(self):
        return self.builder_cls.candidate_features

    def __init__(self, settings):
        self.settings = settings
        self.espn = ESPN(self.league_path, settings.user_agent)
        self.guard = TemporalGuard(self.game_duration_hours)
        self.cache = settings.cache_dir / self.sport / "parsed"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.ep_path = settings.store_dir / f"{self.sport}_ep_model.json"

    # ================================================================== ingestion
    def scoreboard_jobs(self, seasons):
        return [(s, 2, w, g) for s in seasons for w in range(1, 17) for g in (80, 81)] + \
               [(s, 3, 1, g) for s in seasons for g in (80, 81)]

    def skip_event(self, e, h, a) -> bool:
        return False

    def current_week_game_ids(self) -> list[str]:
        """Game ids on ESPN's current-week scoreboard (pre-game schedule information)."""
        ids = []
        for g in (80, 81):
            ids += [str(e["id"]) for e in self.espn.scoreboard(groups=g).get("events", [])]
        return list(dict.fromkeys(ids))

    def sync(self, store, *, full=False, seasons=None, stats=True) -> dict:
        t0 = now(); cur = current_season(t0)
        seasons = seasons or (list(range(self.builder_cls.FIRST_SEASON, cur + 1)) if full else [cur])
        jobs = self.scoreboard_jobs(seasons)
        stamp = iso(t0)
        G, Rr, T = [], [], []
        sb_cache = self.settings.cache_dir / self.sport / "scoreboards"; sb_cache.mkdir(parents=True, exist_ok=True)
        def fetch(j):
            s, st, w, g = j
            fn = sb_cache / f"{s}_{st}_{w}_{g}.json"
            if s < cur and fn.exists():          # completed seasons are immutable -> cached
                return j, json.loads(fn.read_text())
            try:
                d = self.espn.scoreboard(dates=s, seasontype=st, week=w, groups=g)
            except RuntimeError:
                return j, None
            if s < cur:
                fn.write_text(json.dumps(d))
            return j, d
        failed = []
        for attempt in range(2):
            with cf.ThreadPoolExecutor(8) as ex:
                for (s, st, w, g), d in ex.map(fetch, jobs):
                    if d is None:
                        failed.append((s, st, w, g)); continue
                    a, b, c = scoreboard_rows(d, g, stamp, sport=self.sport, skip=self.skip_event, full_names=self.full_names)
                    G += a; Rr += b; T += c
            if not failed:
                break
            jobs, failed = failed, []
        # FBS feed first so feed_group=80 wins for games listed in both feeds
        G = list({r["game_id"]: r for r in sorted(G, key=lambda r: -(r["feed_group"] or 0))}.values())
        T = list({r["team_id"]: r for r in T}.values())
        Rr = list({r["game_id"]: r for r in Rr}.values())
        known = set(store.df("SELECT game_id FROM results WHERE sport=?", (self.sport,)).game_id)
        new_results = [r for r in Rr if r["game_id"] not in known]
        store.upsert_teams(T); store.upsert_games(G)
        # results keep their first collection timestamp; corrections update the score only
        store.upsert_results(new_results)
        for r in Rr:
            if r["game_id"] in known:
                with store.tx() as c:
                    c.execute("UPDATE results SET home_score=?, away_score=?, status=? WHERE sport=? AND game_id=?",
                              (r["home_score"], r["away_score"], r["status"], self.sport, r["game_id"]))
        out = {"seasons": seasons, "games": len(G), "results_new": len(new_results), "failed_requests": failed}
        if stats:
            out["stats_new"] = self.sync_stats(store)
        store.log(self.sport, "sync", json.dumps(out))
        return out

    def _summary_compact(self, eid: str, fetch: bool = True):
        fn = self.cache / f"{eid}.json.gz"
        if fn.exists():
            return json.load(gzip.open(fn, "rt"))
        if not fetch:
            return None
        s = self.espn.summary(eid)
        if not s:
            return None
        d = compact_summary(eid, s)
        json.dump(d, gzip.open(fn, "wt"))
        return d

    def ep_model(self, store) -> EPModel:
        if self.ep_path.exists():
            return EPModel.load(self.ep_path)
        done = store.completed_games(self.sport)
        fit_games = done[(done.season >= self.builder_cls.FIRST_STATS_SEASON) & (done.season <= current_season(now()) - 1)]
        parsed = [d for d in (self._summary_compact(e, fetch=False) for e in fit_games.game_id) if d]
        pl = plays_frame(parsed, dict(zip(fit_games.game_id, fit_games.home_id)))
        ep = EPModel().fit(pl)
        ep.meta.update({"fit_seasons": sorted(fit_games.season.unique().tolist()), "fitted_utc": iso(now())})
        ep.save(self.ep_path)
        return ep

    def sync_stats(self, store, game_ids=None, threads=16) -> int:
        done = store.completed_games(self.sport)
        have = set(store.df("SELECT game_id FROM team_game_stats WHERE sport=?", (self.sport,)).game_id)
        todo = done[(done.season >= self.builder_cls.FIRST_STATS_SEASON) & ~done.game_id.isin(have)]
        if game_ids is not None:
            todo = todo[todo.game_id.isin(game_ids)]
        if todo.empty:
            return 0
        with cf.ThreadPoolExecutor(threads) as ex:
            parsed = [d for d in ex.map(self._summary_compact, todo.game_id) if d]
        ep = self.ep_model(store)
        home_of = dict(zip(todo.game_id, todo.home_id))
        stamp = iso(now()); n = 0
        for i in range(0, len(parsed), 1500):
            rows = team_game_stats(parsed[i:i + 1500], home_of, ep)
            store.upsert_team_game_stats([{"sport": self.sport, "game_id": r["game_id"], "team_id": r["team_id"],
                                           "stats_json": json.dumps(r["stats"], default=float), "stats_version": STATS_VERSION,
                                           "collected_utc": stamp} for r in rows])
            n += len(rows)
        return n

    def refresh_results(self, store, game_ids) -> int:
        """Collect final scores (post-game) for specific games via the game summary."""
        stamp = iso(now()); rows = []
        for gid in game_ids:
            s = self.espn.summary(gid)
            try:
                comp = s["header"]["competitions"][0]
            except (KeyError, IndexError, TypeError):
                continue
            st = comp.get("status", {}).get("type", {})
            if not (st.get("completed") and st.get("state") == "post"):
                continue
            sc = {c["homeAway"]: num(c.get("score")) for c in comp.get("competitors", [])}
            if sc.get("home") is None or sc.get("away") is None:
                continue
            rows.append(dict(sport=self.sport, game_id=str(gid), home_score=sc["home"], away_score=sc["away"],
                             status=st.get("name"), collected_utc=stamp, source="espn_summary"))
            fn = self.cache / f"{gid}.json.gz"
            if not fn.exists():
                json.dump(compact_summary(gid, s), gzip.open(fn, "wt"))
        known = set(store.df("SELECT game_id FROM results WHERE sport=?", (self.sport,)).game_id)
        store.upsert_results([r for r in rows if r["game_id"] not in known])
        if rows:
            self.sync_stats(store, game_ids=[r["game_id"] for r in rows])
        return len(rows)

    # ================================================================== lookup
    def resolve_team(self, store, text: str) -> tuple[str, str]:
        teams = store.df("SELECT * FROM teams WHERE sport=?", (self.sport,))
        g = store.df("SELECT home_id, away_id, feed_group FROM games WHERE sport=? AND season>=?", (self.sport, current_season(now()) - 1))
        cnt = pd.concat([g[g.feed_group == 80].home_id, g[g.feed_group == 80].away_id]).value_counts()
        q = _norm(ALIASES.get(_norm(text), text))
        fields = ["location", "display_name", "abbr", "short_name", "name"]
        scored = []
        for r in teams.itertuples():
            names = {_norm(getattr(r, f)) for f in fields if getattr(r, f)}
            if q in names:
                rank = 0 if _norm(r.location or "") == q else 1
                scored.append((rank, -cnt.get(r.team_id, 0), r.team_id, r.location))
        if not scored:
            pool = {}
            for r in teams.itertuples():
                for f in fields:
                    v = getattr(r, f)
                    if v:
                        pool.setdefault(_norm(v), []).append(r)
            m = difflib.get_close_matches(q, list(pool), n=3, cutoff=0.75)
            if not m:
                raise LookupError(f"team not found: {text!r}")
            scored = [(1, -cnt.get(r.team_id, 0), r.team_id, r.location) for r in pool[m[0]]]
        scored.sort()
        return scored[0][2], scored[0][3]

    def find_game(self, store, team_a, team_b, after, days=21):
        def search():
            g = store.games(self.sport)
            res = set(store.df("SELECT game_id FROM results WHERE sport=?", (self.sport,)).game_id)
            m = (((g.home_id == team_a) & (g.away_id == team_b)) | ((g.home_id == team_b) & (g.away_id == team_a)))
            m &= (g.kickoff_utc > after - pd.Timedelta(hours=self.game_duration_hours)) & (g.kickoff_utc <= after + pd.Timedelta(days=days))
            m &= ~g.game_id.isin(res)
            return g[m].sort_values("kickoff_utc")
        hit = search()
        if hit.empty:  # pull the team's schedule (pre-game information) and look again
            d = self.espn.team_schedule(team_a, current_season(after))
            G, _, T = scoreboard_rows(d, None, iso(now()), sport=self.sport, skip=self.skip_event, full_names=self.full_names)
            if G:
                store.upsert_teams(T); store.upsert_games(G, only_new=True)
            hit = search()
        return None if hit.empty else hit.iloc[0]

    def pregame_context(self, game_id: str) -> dict:
        """Pre-kickoff context for display (forecast, records, ranks). The market line is recorded only
        as an external reference and is never a model input."""
        try:
            s = self.espn.summary(game_id)
        except Exception:
            return {}
        ctx = {"captured_utc": iso(now())}
        gi = s.get("gameInfo") or {}
        w = gi.get("weather") or {}
        ctx["weather"] = {k: w.get(k) for k in ("temperature", "gust", "precipitation", "conditionId")} if w else None
        ctx["venue"] = (gi.get("venue") or {}).get("fullName")
        try:
            comp = s["header"]["competitions"][0]
            st = comp.get("status", {}).get("type", {})
            ctx["status_state"] = st.get("state")
            for c in comp.get("competitors", []):
                side = c["homeAway"]
                ctx[f"{side}_record"] = (c.get("record") or [{}])[0].get("summary") if c.get("record") else None
                ctx[f"{side}_rank"] = c.get("rank")
        except (KeyError, IndexError, TypeError):
            pass
        pc = s.get("pickcenter") or []
        if pc:
            ctx["market_reference_not_model_input"] = {"details": pc[0].get("details"), "overUnder": pc[0].get("overUnder")}
        ctx["injuries"] = "Data unavailable (ESPN publishes no college injury feed)"
        return ctx

    def qb_names(self, store, team_id: str, before: pd.Timestamp):
        s = store.df("""SELECT t.stats_json, g.kickoff_utc FROM team_game_stats t JOIN games g USING (sport, game_id)
                        WHERE t.sport=? AND t.team_id=? AND g.kickoff_utc < ? ORDER BY g.kickoff_utc DESC LIMIT 6""",
                     (self.sport, team_id, iso(before - pd.Timedelta(hours=self.game_duration_hours))))
        names = [json.loads(x).get("qb_name") for x in s.stats_json]
        return [n for n in names if n]

    # ================================================================== modelling hooks
    def feature_builder(self, store):
        return self.builder_cls.from_store(store, self.guard, self.sport)

    def default_config(self) -> dict:
        gbm = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=40, reg_lambda=5.0)
        return {"features": list(MODEL_FEATURES), "total_features": list(TOTAL_FEATURES),
                "explain_features": list(MODEL_FEATURES),
                "members": {"elo": {}, "scoring": {}, "logistic": {"C": 0.1}, "gbm": dict(gbm),
                            "margin_ridge": {"alpha": 10.0}, "gbm_margin": dict(gbm)},
                "ensemble": "simplex", "stack_context": [], "calibration": "none",
                "half_life_days": None, "min_train_season": 2022, "holdout_frac": 0.2}

    def subgroups(self, df: pd.DataFrame) -> dict:
        p = df.p_home
        fav_home = p >= 0.5
        return {
            "home_favorite": fav_home & (df.neutral == 0), "road_favorite": ~fav_home & (df.neutral == 0),
            "neutral_site": df.neutral == 1, "conference_game": df.conf_game == 1,
            "fbs_vs_fcs": (df.fbs_h + df.fbs_a) == 1, "both_fbs": (df.fbs_h + df.fbs_a) == 2, "fcs_or_lower_only": (df.fbs_h + df.fbs_a) == 0,
            "early_season": df.early == 1, "qb_change_any": (df.qb_change_h + df.qb_change_a) > 0,
            "big_favorite_85": (p >= 0.85) | (p <= 0.15), "toss_up_40_60": (p > 0.4) & (p < 0.6),
            "postseason": df.get("seasontype", pd.Series(2, index=df.index)) == 3,
        }

    def explain(self, row: pd.Series, pred: pd.Series, model) -> dict:
        H, A = row.home_name, row.away_name
        p = float(pred.p_home); fav, dog = (H, A) if p >= 0.5 else (A, H); sgn = 1 if p >= 0.5 else -1
        # matchup edges, standardized by the training distribution of each matchup feature
        edges = []
        for k in EFF:
            c = f"mx_{k}"
            if c not in row or pd.isna(row[c]):
                continue
            mu, sd = model.feature_stats.get(c, (0.0, 1.0))
            z = (row[c] - mu) / (sd or 1.0) * (-1 if k in BAD_HIGH else 1)   # + = home edge
            ho, ad = row.get(f"h_o_{k}", np.nan), row.get(f"a_d_{k}", np.nan)
            ao, hd = row.get(f"a_o_{k}", np.nan), row.get(f"h_d_{k}", np.nan)
            sc, unit = {"sr": (100, "pp"), "expl": (100, "pp"), "sack_rate": (100, "pp"), "to_rate": (100, "pp"),
                        "rush_sr": (100, "pp"), "pass_sr": (100, "pp"), "start_fp": (1, "yds"), "ppo": (1, "pts"),
                        "pts_per_drive": (1, "pts"), "ypp": (1, "yds")}.get(k, (1, "EPA/play"))
            edges.append({"metric": k, "label": LABELS[k], "edge_team": H if z > 0 else A, "z": float(abs(z)),
                          "favours_pick": bool(z * sgn > 0),
                          "detail": f"vs-average expectation: {H} offense vs {A} defense {(ho + ad) * sc:+.2f} {unit}, "
                                    f"{A} offense vs {H} defense {(ao + hd) * sc:+.2f} {unit}"})
        edges.sort(key=lambda e: -e["z"])
        drivers = []
        C = model.contributions(row.to_frame().T)
        if C is not None:
            c = C.iloc[0].sort_values(key=np.abs, ascending=False)
            for f, v in c.head(6).items():
                if abs(v) < 0.01:
                    continue
                drivers.append({"feature": f, "label": FEATURE_LABELS.get(f, f), "log_odds": float(v),
                                "favours": H if v > 0 else A})
        factors = [f"{e['label']}: edge {e['edge_team']} ({e['z']:.2f} SD) — {e['detail']}" for e in edges[:3]]
        risks = []
        if int(pred.models_split):
            risks.append("ensemble members disagree on the winner")
        counters = [e for e in edges if not e["favours_pick"]]
        if counters:
            risks.append(f"{dog}'s best counter: {counters[0]['label']} ({counters[0]['z']:.2f} SD)")
        if row.get("qb_change_h", 0) or row.get("qb_change_a", 0):
            who = [n for n, k in ((H, "qb_change_h"), (A, "qb_change_a")) if row.get(k, 0)]
            risks.append(f"starting QB changed last game ({', '.join(who)}) — availability unconfirmed")
        if row.get("early", 0):
            risks.append("early-season sample: ratings lean on last season's prior")
        if not risks:
            risks.append("single-game variance (margin sigma ~ 16 pts)")
        notes = {
            "home_field": "neutral site" if row.neutral else f"{H} at home (scoring-model home coef {row.get('hfa_pts_coef', np.nan):.1f} pts)",
            "form": f"last-3 margin vs expectation: {H} {row.form3_h:+.1f}, {A} {row.form3_a:+.1f}",
            "rest": f"rest days: {H} {row.rest_h:.0f}, {A} {row.rest_a:.0f}",
            "qb_continuity": f"starter share of season attempts: {H} {row.qb_share_h:.0%}, {A} {row.qb_share_a:.0%}",
            "injuries": "Data unavailable (no public college injury feed)",
        }
        return {"key_factors": factors, "drivers": drivers, "matchups": edges, "main_risk": "; ".join(risks[:3]), "notes": notes}
