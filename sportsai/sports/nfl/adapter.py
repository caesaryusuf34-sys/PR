"""NFL adapter. ESPN serves the NFL in the same JSON format as college football, so ingestion, the
expected-points model, team-game statistics and the point-in-time feature builder are reused with
NFL-specific settings (single league, NFL Elo constants, stronger preseason regression, 18-week
regular season + playoffs, Pro Bowl excluded). Injury reports, which ESPN publishes for the NFL,
are captured as pre-game context and surfaced in explanations/risks."""
from __future__ import annotations
import numpy as np
import pandas as pd

from ...core.timeutil import iso, now
from ..ncaaf.adapter import NCAAFAdapter, LABELS
from ..ncaaf.features import NCAAFFeatureBuilder, EFF

SPORT = "nfl"
NFL_FEATURES = (["elo_diff", "pts_margin_pred", "pts_total_pred"] + [f"mx_{k}" for k in EFF] +
                ["form_diff", "rest_diff", "qb_share_h", "qb_share_a", "qb_change_h", "qb_change_a",
                 "neutral", "conf_game", "early", "n_prior_h", "n_prior_a"])
NFL_TOTAL_FEATURES = ["pts_total_pred", "abs_pts_margin_pred", "pp_home", "pp_away", "env_ppd_h", "env_ppd_a",
                      "env_epa_h", "env_epa_a", "env_expl_h", "env_expl_a", "early", "conf_game"]
ALIASES = {"niners": "49ers", "pats": "Patriots", "bucs": "Buccaneers", "jags": "Jaguars", "commies": "Commanders",
           "washington": "Commanders", "football team": "Commanders", "philly": "Eagles", "g-men": "Giants",
           "kc": "Chiefs", "sf": "49ers", "gb": "Packers", "ne": "Patriots", "no": "Saints", "tb": "Buccaneers",
           "lv": "Raiders", "lar": "Rams", "lac": "Chargers", "nyg": "Giants", "nyj": "Jets", "wsh": "Commanders"}


class NFLFeatureBuilder(NCAAFFeatureBuilder):
    feature_version = "nfl-f1"
    SPORT = SPORT
    FIRST_SEASON = 2010
    FIRST_STATS_SEASON = 2016
    SHRINK = 0.5                                  # NFL regresses harder to the mean between seasons
    LAM = {"pts": 4.0, **{k: 5.0 for k in EFF}}   # prior worth ~4-5 games of a 17-game season
    ELO = dict(K=20.0, HFA=48.0, revert=1 / 3)
    ELO_PTS = 25.0
    ELO_INIT = {"NFL": 1500.0}
    model_features = NFL_FEATURES

    def div_of(self, season, tid):
        return "NFL"


class NFLAdapter(NCAAFAdapter):
    sport = SPORT
    league_path = "football/nfl"
    builder_cls = NFLFeatureBuilder
    game_duration_hours = 4.0
    full_names = True
    first_train_season = 2017
    policy_overrides = {"min_new_games": 48, "min_confirm_games": 250, "min_subgroup_n": 40}

    def scoreboard_jobs(self, seasons):
        # regular season weeks 1-18 (17 before 2021) and playoffs; feed_group=None (single league)
        return [(s, 2, w, None) for s in seasons for w in range(1, 19)] + [(s, 3, w, None) for s in seasons for w in range(1, 6)]

    def skip_event(self, e, h, a) -> bool:
        ab = {h["team"].get("abbreviation"), a["team"].get("abbreviation")}
        return bool(ab & {"AFC", "NFC", "APR", "NPR"})          # Pro Bowl / all-star games

    def current_week_game_ids(self):
        return [str(e["id"]) for e in self.espn.scoreboard().get("events", [])]

    def resolve_team(self, store, text: str):
        return super().resolve_team(store, ALIASES.get(text.strip().lower(), text))

    def default_config(self) -> dict:
        gbm = dict(n_estimators=250, learning_rate=0.03, num_leaves=7, min_child_samples=40, reg_lambda=10.0)
        return {"features": list(NFL_FEATURES), "total_features": list(NFL_TOTAL_FEATURES),
                "explain_features": list(NFL_FEATURES),
                "members": {"elo": {}, "scoring": {}, "logistic": {"C": 0.05}, "gbm": dict(gbm),
                            "margin_ridge": {"alpha": 30.0}, "gbm_margin": dict(gbm)},
                "ensemble": "simplex", "stack_context": [], "calibration": "none",
                "half_life_days": None, "min_train_season": 2017, "holdout_frac": 0.2}

    def subgroups(self, df: pd.DataFrame) -> dict:
        p = df.p_home; fav_home = p >= 0.5
        return {"home_favorite": fav_home & (df.neutral == 0), "road_favorite": ~fav_home & (df.neutral == 0),
                "neutral_site": df.neutral == 1, "conference_game": df.conf_game == 1, "early_season": df.early == 1,
                "qb_change_any": (df.qb_change_h + df.qb_change_a) > 0, "big_favorite_75": (p >= 0.75) | (p <= 0.25),
                "toss_up_40_60": (p > 0.4) & (p < 0.6), "rest_edge_3plus": df.rest_diff.abs() >= 3,
                "postseason": df.get("seasontype", pd.Series(2, index=df.index)) == 3}

    def pregame_context(self, game_id: str) -> dict:
        ctx = super().pregame_context(game_id)
        try:
            s = self.espn.summary(game_id)
        except Exception:
            return ctx
        inj = {}
        for t in s.get("injuries") or []:
            ab = t.get("team", {}).get("abbreviation")
            rows = []
            for i in t.get("injuries", []):
                st = i.get("status")
                if st in ("Out", "Doubtful", "Questionable", "Injured Reserve"):
                    a = i.get("athlete", {}) or {}
                    rows.append({"player": a.get("displayName"), "pos": (a.get("position") or {}).get("abbreviation"), "status": st})
            inj[ab] = rows
        ctx["injuries"] = inj if inj else "none reported"
        return ctx

    def augment_explanation(self, expl: dict, ctx: dict, game) -> dict:
        """Surface the pre-game injury report as risk context (not a model input)."""
        inj = ctx.get("injuries")
        if not isinstance(inj, dict):
            return expl
        flags = []
        for team, rows in inj.items():
            key = [r for r in rows if r["status"] in ("Out", "Doubtful") and r["pos"] in ("QB",)]
            out = [r for r in rows if r["status"] in ("Out", "Doubtful")]
            for r in key:
                flags.append(f"{team} QB {r['player']} {r['status'].upper()}")
            if len(out) >= 4:
                flags.append(f"{team}: {len(out)} players out/doubtful")
        expl = dict(expl)
        expl["injury_flags"] = flags
        if flags:
            expl["main_risk"] = "; ".join(flags[:2]) + ("; " + expl["main_risk"] if expl.get("main_risk") else "")
        expl["notes"] = dict(expl.get("notes", {}))
        expl["notes"]["injury_report"] = "; ".join(
            f"{t}: " + (", ".join(f"{r['player']} ({r['pos']}, {r['status']})" for r in rows if r["status"] in ("Out", "Doubtful")) or "no outs")
            for t, rows in inj.items())
        return expl

    def explain(self, row, pred, model) -> dict:
        out = super().explain(row, pred, model)
        out["notes"]["injuries"] = "see injury report in context (pre-game ESPN feed; not a model input)"
        return out
