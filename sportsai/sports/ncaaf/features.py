"""Point-in-time feature builder for NCAA football.

For each target game the builder receives a cutoff (<= kickoff) and reconstructs the world as it was
at that instant: only results/box scores of games that had finished by the cutoff are visible
(TemporalGuard), ratings are re-fitted on exactly that information, and the season prior comes from
last season's final ratings. The schedule itself (who plays whom, where) is pre-game knowledge.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from ...core.leakage import TemporalGuard
from ...core.sport import FeatureBuilder
from ...core.timeutil import ns
from .ratings import EloEngine, ridge_ratings

EFF = ["epa_pp", "sr", "ypp", "expl", "rush_epa", "pass_epa", "rush_sr", "pass_sr", "sack_rate", "to_rate",
       "pts_per_drive", "ppo", "start_fp"]
LAM = {"pts": 3.0, **{k: 4.0 for k in EFF}}
SHRINK = 0.6           # weight of last season's final rating in the preseason prior
FIRST_STATS_SEASON = 2021
FIRST_SEASON = 2015
ELO = dict(K=36.0, HFA=45.0, revert=0.25)
ELO_PTS = 25.0         # Elo points per point of margin when converting Elo edges to expected margins

MODEL_FEATURES = (["elo_diff", "pts_margin_pred", "pts_total_pred"] + [f"mx_{k}" for k in EFF] +
                  ["form_diff", "rest_diff", "qb_share_h", "qb_share_a", "qb_change_h", "qb_change_a",
                   "neutral", "conf_game", "fbs_h", "fbs_a", "early", "n_prior_h", "n_prior_a"])
TOTAL_FEATURES = ["pts_total_pred", "abs_pts_margin_pred", "pp_home", "pp_away", "env_ppd_h", "env_ppd_a",
                  "env_epa_h", "env_epa_a", "env_expl_h", "env_expl_a", "early", "fbs_h", "fbs_a", "fbs_both", "conf_game"]
UNIT_FEATURES = [f"{s}_{u}_{k}" for k in ["pts"] + EFF for s in ("h", "a") for u in ("o", "d")]
EXTRA_FEATURES = ["pp_home", "pp_away", "elo_h", "elo_a", "form3_h", "form3_a", "rest_h", "rest_a"]


class NCAAFFeatureBuilder(FeatureBuilder):
    """Generic American-football point-in-time builder; sport specifics are class attributes."""
    feature_version = "ncaaf-f1"
    FIRST_SEASON, FIRST_STATS_SEASON, SHRINK, LAM, ELO, ELO_PTS = FIRST_SEASON, FIRST_STATS_SEASON, SHRINK, LAM, ELO, ELO_PTS
    ELO_INIT = None            # None -> ratings.ELO_INIT (FBS/FCS/OTHER)
    SPORT = "ncaaf"
    model_features = MODEL_FEATURES
    candidate_features = MODEL_FEATURES + EXTRA_FEATURES + [f for f in UNIT_FEATURES if f not in MODEL_FEATURES]

    def __init__(self, schedule: pd.DataFrame, completed: pd.DataFrame, stats: pd.DataFrame, guard: TemporalGuard):
        self.guard = guard
        self.sched = schedule.copy()
        self.done = completed[completed.season >= self.FIRST_SEASON].sort_values(["kickoff_utc", "game_id"]).reset_index(drop=True)
        # a result is usable once it was observed final: at the time it was recorded, and in any case no later
        # than kickoff + duration (historical backfills are recorded long after the fact)
        avail = guard.available_at(self.done.kickoff_utc)
        if "result_collected_utc" in self.done:
            seen = pd.to_datetime(self.done.result_collected_utc, utc=True)
            avail = avail.where(seen.isna() | (seen >= avail), seen)
            avail = avail.where(avail > self.done.kickoff_utc, guard.available_at(self.done.kickoff_utc))
        self.done["avail"] = avail
        self.stats = stats.copy() if len(stats) else pd.DataFrame(columns=["game_id", "team_id", "kickoff_utc", "season"])
        # division per team-season from the schedule (pre-game knowledge): share of games in the FBS feed
        long = pd.concat([self.sched[["season", "home_id", "feed_group"]].rename(columns={"home_id": "tid"}),
                          self.sched[["season", "away_id", "feed_group"]].rename(columns={"away_id": "tid"})])
        agg = long.groupby(["season", "tid"]).feed_group.agg(n="size", fbs=lambda s: (s == 80).mean())
        agg["div"] = np.where(agg.fbs > 0.5, "FBS", np.where(agg.n >= 4, "FCS", "OTHER"))
        self._div = agg["div"].to_dict()
        self._teams_by_season = {s: sorted(set(g.home_id) | set(g.away_id)) for s, g in self.sched.groupby("season")}
        self._finals = {}
        self._prep_obs()

    # ------------------------------------------------------------------ helpers
    @classmethod
    def from_store(cls, store, guard, sport=None):
        sport = sport or cls.SPORT
        return cls(store.games(sport), store.completed_games(sport), store.team_game_stats(sport), guard)

    def div_of(self, season, tid):
        for s in (season, season - 1, season - 2):
            v = self._div.get((s, tid))
            if v is not None:
                return v
        return "OTHER"

    def visible_counts(self, cutoffs: pd.Series):
        """(n completed games, n team-game stat rows) visible at each cutoff - used to validate cached features."""
        a = np.sort(ns(self.done.avail)); b = np.sort(ns(self.eff_obs.avail)) if len(self.eff_obs) else np.array([], dtype="int64")
        c = ns(cutoffs)
        return np.searchsorted(a, c, side="right"), np.searchsorted(b, c, side="right")

    def training_cutoff(self, kickoff: pd.Series) -> pd.Series:
        """Historical reconstruction cutoff: 05:00 US/Eastern on game day (never after kickoff)."""
        et = kickoff.dt.tz_convert("America/New_York")
        c = (et.dt.normalize() + pd.Timedelta(hours=5)).dt.tz_convert("UTC")
        return pd.concat([c, kickoff], axis=1).min(axis=1)

    def _prep_obs(self):
        d = self.done
        loc_h = np.where(d.neutral.astype(bool), 0, 1)
        self.pts_obs = pd.DataFrame({
            "season": np.r_[d.season, d.season], "avail": np.r_[d.avail, d.avail],
            "off": np.r_[d.home_id, d.away_id], "dfn": np.r_[d.away_id, d.home_id],
            "loc": np.r_[loc_h, -loc_h], "y": np.r_[d.home_score, d.away_score]})
        st = self.stats.merge(d[["game_id", "home_id", "away_id", "neutral", "avail"]], on="game_id", how="inner")
        st["is_home"] = st.team_id == st.home_id
        st["off"] = st.team_id; st["dfn"] = np.where(st.is_home, st.away_id, st.home_id)
        st["loc"] = np.where(st.neutral.astype(bool), 0, np.where(st.is_home, 1, -1))
        self.eff_obs = st
        # per team-season histories for sequence features
        th = pd.DataFrame({"game_id": np.r_[d.game_id, d.game_id], "tid": np.r_[d.home_id, d.away_id],
                           "season": np.r_[d.season, d.season], "kick": np.r_[d.kickoff_utc, d.kickoff_utc],
                           "avail": np.r_[d.avail, d.avail],
                           "margin": np.r_[d.home_score - d.away_score, d.away_score - d.home_score],
                           "is_home": np.r_[np.ones(len(d), bool), np.zeros(len(d), bool)],
                           "neutral": np.r_[d.neutral, d.neutral].astype(bool)})
        th = th.sort_values(["tid", "season", "avail"])
        self._hist = {k: g.reset_index(drop=True) for k, g in th.groupby(["tid", "season"])}
        q = st[["team_id", "season", "avail", "qb_id", "qb_att"]].copy() if "qb_id" in st else pd.DataFrame()
        self._qb = {k: g.sort_values("avail").reset_index(drop=True) for k, g in q.groupby(["team_id", "season"])} if len(q) else {}

    def _obs(self, metric, season, cutoff):
        if metric == "pts":
            o = self.pts_obs[(self.pts_obs.season == season) & (self.pts_obs.avail <= cutoff)]
            y = o.y.values
        else:
            o = self.eff_obs[(self.eff_obs.season == season) & (self.eff_obs.avail <= cutoff)]
            if metric not in o:
                return o.iloc[:0], np.array([])
            o = o[o[metric].notna()]
            y = o[metric].astype(float).values
        return o, y

    def _prior(self, metric, season, cutoff):
        prev = self._final(metric, season - 1, cutoff)
        if prev is None:
            return {}, {}
        po, pdv = prev
        groups = {}
        for t in po:
            groups.setdefault(self.div_of(season - 1, t), []).append((po[t], pdv[t]))
        dmean = {k: (np.mean([a for a, _ in v]), np.mean([b for _, b in v])) for k, v in groups.items()}
        prior_o, prior_d = {}, {}
        for t in self._teams_by_season.get(season, []):
            mo, md = dmean.get(self.div_of(season, t), dmean.get("OTHER", (0.0, 0.0)))
            if t in po:
                prior_o[t] = self.SHRINK * po[t] + (1 - self.SHRINK) * mo; prior_d[t] = self.SHRINK * pdv[t] + (1 - self.SHRINK) * md
            else:
                prior_o[t], prior_d[t] = mo, md
        return prior_o, prior_d

    def _final(self, metric, season, cutoff):
        first = self.FIRST_SEASON if metric == "pts" else self.FIRST_STATS_SEASON
        if season < first:
            return None
        o, y = self._obs(metric, season, cutoff)
        key = (metric, season, len(y))
        if key not in self._finals:
            prior_o, prior_d = self._prior(metric, season, cutoff)
            teams = sorted(set(self._teams_by_season.get(season, [])) | set(o.off) | set(o.dfn))
            _, _, ro, rd = ridge_ratings(o.off.values, o.dfn.values, o["loc"].values, y, teams, prior_o, prior_d, self.LAM[metric])
            self._finals[key] = (ro, rd)
        return self._finals[key]

    def ratings_at(self, season, cutoff, extra_teams=()):
        out = {}
        for metric in ["pts"] + (EFF if season >= self.FIRST_STATS_SEASON else []):
            o, y = self._obs(metric, season, cutoff)
            self.guard.check(pd.DataFrame({"kickoff_utc": o.avail - self.guard.duration}) if len(o) else pd.DataFrame({"kickoff_utc": []}),
                             cutoff, f"ratings[{metric}]")
            prior_o, prior_d = self._prior(metric, season, cutoff)
            teams = sorted(set(self._teams_by_season.get(season, [])) | set(o.off) | set(o.dfn) | set(extra_teams))
            mu, h, ro, rd = ridge_ratings(o.off.values, o.dfn.values, o["loc"].values, y, teams, prior_o, prior_d, self.LAM[metric])
            out[metric] = (mu, h, ro, rd)
        return out

    def _seq(self, tid, season, cutoff, kickoff, elo):
        h = self._hist.get((tid, season))
        res = {"form3": 0.0, "rest": 14.0, "n_prior": 0, "qb_share": 1.0, "qb_change": 0.0}
        if h is not None:
            k = int(np.searchsorted(ns(h.avail), ns(cutoff), side="right"))
            vis = h.iloc[:k]
            res["n_prior"] = k
            if k:
                resid = []
                for r in vis.tail(3).itertuples():
                    pre = elo.pre.get(r.game_id)
                    if pre is None:
                        continue
                    mine, theirs = (pre[0], pre[1]) if r.is_home else (pre[1], pre[0])
                    hfa = 0 if r.neutral else (self.ELO["HFA"] if r.is_home else -self.ELO["HFA"])
                    resid.append(r.margin - (mine - theirs + hfa) / self.ELO_PTS)
                res["form3"] = float(np.mean(resid)) if resid else 0.0
                res["rest"] = float(min((kickoff - vis.kick.iloc[-1]).days, 21))
        q = self._qb.get((tid, season))
        if q is not None:
            k = int(np.searchsorted(ns(q.avail), ns(cutoff), side="right"))
            vq = q.iloc[:k].dropna(subset=["qb_id"])
            if len(vq):
                att = vq.groupby("qb_id").qb_att.sum()
                last = vq.qb_id.iloc[-1]
                res["qb_share"] = float(att.get(last, 0) / att.sum()) if att.sum() > 0 else 1.0
                res["qb_change"] = float(len(vq) >= 2 and vq.qb_id.iloc[-1] != vq.qb_id.iloc[-2])
        return res

    # ------------------------------------------------------------------ main entry
    def build(self, targets: pd.DataFrame) -> pd.DataFrame:
        t = targets.copy()
        t["kickoff_utc"] = pd.to_datetime(t.kickoff_utc, utc=True)
        t["cutoff_utc"] = pd.to_datetime(t.cutoff_utc, utc=True)
        if (t.cutoff_utc > t.kickoff_utc).any():
            raise ValueError("cutoff after kickoff")
        t = t.sort_values(["cutoff_utc", "game_id"])
        elo = EloEngine(self.done, self.done.avail, self.div_of, **self.ELO, init=self.ELO_INIT)
        rows = []
        for (cutoff, season), grp in t.groupby(["cutoff_utc", "season"], sort=True):
            elo.advance(cutoff)
            elo.snapshot(season)
            R = self.ratings_at(season, cutoff, extra_teams=set(grp.home_id) | set(grp.away_id))
            n_used = int((self.done.avail <= cutoff).sum())
            n_stats = int((self.eff_obs.avail <= cutoff).sum()) if len(self.eff_obs) else 0
            for g in grp.itertuples():
                loc = 0 if bool(g.neutral) else 1
                f = {"game_id": g.game_id, "cutoff_utc": cutoff, "n_games_visible": n_used, "n_stats_visible": n_stats}
                eh, ea = elo.rating(season, g.home_id), elo.rating(season, g.away_id)
                f.update(elo_h=eh, elo_a=ea, elo_diff=eh - ea + loc * self.ELO["HFA"])
                for metric, (mu, h, ro, rd) in R.items():
                    f[f"h_o_{metric}"], f[f"h_d_{metric}"] = ro[g.home_id], rd[g.home_id]
                    f[f"a_o_{metric}"], f[f"a_d_{metric}"] = ro[g.away_id], rd[g.away_id]
                    if metric == "pts":
                        f["pp_home"] = mu + ro[g.home_id] + rd[g.away_id] + h * loc
                        f["pp_away"] = mu + ro[g.away_id] + rd[g.home_id] - h * loc
                        f["hfa_pts_coef"] = 2 * h
                    else:
                        f[f"mx_{metric}"] = (ro[g.home_id] + rd[g.away_id]) - (ro[g.away_id] + rd[g.home_id])
                for side, tid in (("h", g.home_id), ("a", g.away_id)):
                    s = self._seq(tid, season, cutoff, g.kickoff_utc, elo)
                    for k, v in s.items():
                        f[f"{k}_{side}"] = v
                    f[f"fbs_{side}"] = float(self.div_of(season, tid) == "FBS")
                rows.append(f)
        F = pd.DataFrame(rows)
        for k in EFF:
            for c in (f"mx_{k}", f"h_o_{k}", f"h_d_{k}", f"a_o_{k}", f"a_d_{k}"):
                if c not in F:
                    F[c] = np.nan
        F["pts_margin_pred"] = F.pp_home - F.pp_away
        F["pts_total_pred"] = F.pp_home + F.pp_away
        F["abs_pts_margin_pred"] = F.pts_margin_pred.abs()
        F["form_diff"] = F.form3_h - F.form3_a
        F["rest_diff"] = F.rest_h - F.rest_a
        F["early"] = ((F.n_prior_h + F.n_prior_a) < 4).astype(float)
        F["fbs_both"] = F.fbs_h * F.fbs_a
        for side, (a, b) in (("h", ("h", "a")), ("a", ("a", "h"))):
            F[f"env_ppd_{side}"] = F[f"{a}_o_pts_per_drive"] + F[f"{b}_d_pts_per_drive"]
            F[f"env_epa_{side}"] = F[f"{a}_o_epa_pp"] + F[f"{b}_d_epa_pp"]
            F[f"env_expl_{side}"] = F[f"{a}_o_expl"] + F[f"{b}_d_expl"]
        meta = t[["game_id", "season", "kickoff_utc", "home_id", "away_id", "neutral", "conf_game"]].copy()
        meta["neutral"] = meta.neutral.astype(float); meta["conf_game"] = meta.conf_game.astype(float)
        out = meta.merge(F, on="game_id", how="left")
        out["feature_version"] = self.feature_version
        return out
