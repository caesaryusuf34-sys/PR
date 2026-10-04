"""Point-in-time feature builder for baseball (MLB, KBO, NPB, CPBL).

For a target game with information cutoff c (<= first pitch) only results and box scores of games that
had finished by c are visible (TemporalGuard). Everything is re-estimated on exactly that information:

  * Elo                  margin-aware team Elo, regressed toward the mean between seasons
  * run ratings          joint ridge model of runs scored per team-game:
                             runs = mu + offense[team] + defense[opponent] + w * starter[opponent SP] + home
                         shrunk toward last season's final ratings (teams and pitchers separately)
  * starting pitchers    the ANNOUNCED starter as known at c (latest 'probable' captured <= c, else the
                         historical 'backfill' starter); his run effect from the ridge model, plus
                         FIP-type, runs-allowed, (K-BB)/9 and innings-per-start rates from his box-score
                         lines (current season + half-weighted previous season, regressed to the league),
                         and days of rest.  Unknown starter -> average of the team's last five starters.
  * bullpen              regressed FIP-type rate of the relievers, and relief innings in the last 3 days
  * context              home field, form vs Elo expectation, rest, games played, park run factor

Leagues without box-score pitching lines (KBO) use the same model minus the line-based features.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import scipy.sparse as sp

from ...core.leakage import TemporalGuard
from ...core.sport import FeatureBuilder
from ...core.timeutil import ns
from ..ncaaf.ratings import EloEngine

DAY = 86_400 * 10 ** 9


def _ok(p) -> bool:
    return p is not None and p == p
LINE = ["outs", "hr", "bb", "k", "r", "er"]

BASE_FEATURES = ["elo_diff", "pts_margin_pred", "pts_total_pred", "team_diff", "sp_eff_diff",
                 "sp_known_h", "sp_known_a", "sp_rest_h", "sp_rest_a", "form_diff", "rest_diff",
                 "early", "n_prior_h", "n_prior_a", "conf_game"]
LINE_FEATURES = ["sp_fip_diff", "sp_kbb_diff", "sp_ra9_diff", "sp_depth_diff", "pen_fip_diff", "pen_load_diff"]
TOTAL_FEATURES = ["pts_total_pred", "park_h", "off_sum", "def_sum", "sp_eff_sum", "early", "conf_game"]
TOTAL_LINE_FEATURES = ["sp_fip_sum", "sp_ra9_sum", "sp_depth_sum", "pen_fip_sum"]
EXTRA_FEATURES = ["elo_h", "elo_a", "pp_home", "pp_away", "off_h", "off_a", "def_h", "def_a", "sp_eff_h", "sp_eff_a",
                  "sp_fip_h", "sp_fip_a", "sp_kbb_h", "sp_kbb_a", "sp_ra9_h", "sp_ra9_a", "sp_depth_h", "sp_depth_a",
                  "sp_starts_h", "sp_starts_a", "pen_fip_h", "pen_fip_a", "pen_load_h", "pen_load_a",
                  "form10_h", "form10_a", "rest_h", "rest_a", "g7_h", "g7_a", "park_h"]


def ridge_runs(off, dfn, sp_idx, loc, y, n_t, n_p, prior_o, prior_d, prior_p, lam_t, lam_p, sp_w):
    """runs = mu + h*loc + o[off] + d[dfn] + sp_w * p[sp] (sp_idx = -1: starter unknown).
    Ridge-shrunk toward the priors. Returns (mu, h, o, d, p) as arrays."""
    P = 2 + 2 * n_t + n_p
    r = len(y)
    if r:
        has = sp_idx >= 0
        rows = np.concatenate([np.repeat(np.arange(r), 4), np.arange(r)[has]])
        cols = np.concatenate([np.column_stack([np.zeros(r, int), np.ones(r, int), 2 + off, 2 + n_t + dfn]).ravel(),
                               2 + 2 * n_t + sp_idx[has]])
        vals = np.concatenate([np.column_stack([np.ones(r), loc, np.ones(r), np.ones(r)]).ravel(), np.full(has.sum(), sp_w)])
        X = sp.csr_matrix((vals, (rows, cols)), shape=(r, P))
        A = (X.T @ X).toarray(); b = X.T @ np.asarray(y, float)
    else:
        A = np.zeros((P, P)); b = np.zeros(P)
    A[0, 0] += 1e-3; A[1, 1] += 1e-3 + (200.0 if r < 200 else 0.0)
    i0, i1, i2 = 2, 2 + n_t, 2 + 2 * n_t
    A[i0:i1, i0:i1] += np.eye(n_t) * lam_t; b[i0:i1] += lam_t * prior_o
    A[i1:i2, i1:i2] += np.eye(n_t) * lam_t; b[i1:i2] += lam_t * prior_d
    A[i2:, i2:] += np.eye(n_p) * lam_p; b[i2:] += lam_p * prior_p
    if not r:   # no games yet: the intercept is the league scoring level carried by the caller
        A[0, 0] += 1.0
    th = np.linalg.solve(A, b)
    return th[0], th[1], th[i0:i1], th[i1:i2], th[i2:]


class _Cum:
    """Cumulative sums of event rows keyed by availability time (ns), for O(log n) as-of queries."""
    def __init__(self, avail_ns, values: np.ndarray):
        o = np.argsort(avail_ns, kind="stable")
        self.t = np.asarray(avail_ns)[o]
        self.c = np.vstack([np.zeros((1, values.shape[1])), np.cumsum(values[o], axis=0)])

    def at(self, cutoff_ns) -> np.ndarray:
        return self.c[np.searchsorted(self.t, cutoff_ns, side="right")]


class BaseballFeatureBuilder(FeatureBuilder):
    feature_version = "bb-f1"
    SPORT = "mlb"
    TZ = "America/New_York"
    FIRST_SEASON = 2021
    HAS_LINES = True
    ELO = dict(K=5.0, HFA=24.0, revert=1 / 3)
    ELO_PTS = 130.0          # Elo points per run of margin (form residuals)
    SHRINK = 0.6             # weight of last season's final team ratings in the prior
    SP_SHRINK = 0.6          # ... and of a starter's last-season effect
    LAM_T, LAM_P, SP_W = 25.0, 12.0, 0.6
    PREV_W = 0.5             # weight of last season's pitching lines
    IP0_FIP, IP0_RA, IP0_PEN, ST0 = 40.0, 60.0, 60.0, 3.0
    EARLY_GAMES = 20         # combined games played below which a game counts as early-season
    model_features = BASE_FEATURES + LINE_FEATURES
    total_features = TOTAL_FEATURES + TOTAL_LINE_FEATURES
    candidate_features = model_features + EXTRA_FEATURES

    def __init__(self, schedule: pd.DataFrame, completed: pd.DataFrame, stats: pd.DataFrame, starters: pd.DataFrame,
                 guard: TemporalGuard):
        self.guard = guard
        self.sched = schedule.copy()
        self.done = completed[completed.season >= self.FIRST_SEASON].sort_values(["kickoff_utc", "game_id"]).reset_index(drop=True)
        self.done["avail"] = guard.available_at(self.done.kickoff_utc)
        self.stats = stats.copy() if len(stats) else pd.DataFrame(columns=["game_id", "team_id", "kickoff_utc", "season", "sp_id"])
        if "sp_id" not in self.stats:
            self.stats["sp_id"] = None
        self._teams_by_season = {s: sorted(set(g.home_id) | set(g.away_id)) for s, g in self.sched.groupby("season")}
        self._prep_starters(starters)
        self._prep()
        self._finals = {}

    @classmethod
    def from_store(cls, store, guard, sport=None):
        sport = sport or cls.SPORT
        return cls(store.games(sport), store.completed_games(sport), store.team_game_stats(sport), store.starters(sport), guard)

    # ------------------------------------------------------------------ announced starters
    def _prep_starters(self, st: pd.DataFrame):
        self._st = {}
        if st is None or st.empty:
            return
        st = st.sort_values("captured_utc")
        cap = ns(st.captured_utc)
        for (gid, side, src, c, pid, nm) in zip(st.game_id, st.side, st.source, cap, st.pitcher_id, st.pitcher_name):
            self._st.setdefault((gid, side), []).append((src, c, None if pid is None or pid != pid else str(pid), nm))

    def resolve_starter(self, game_id, side, cutoff_ns):
        """Announced starter as known at the cutoff: the latest 'probable' captured <= cutoff, else the
        historical 'backfill' record, else unknown. Returns (pitcher_id, name, source)."""
        rows = self._st.get((game_id, side), [])
        prob = [r for r in rows if r[0] == "probable" and r[1] <= cutoff_ns]
        if prob:
            return prob[-1][2], prob[-1][3], "probable"
        bf = [r for r in rows if r[0] == "backfill"]
        if bf:
            return bf[0][2], bf[0][3], "backfill"
        return None, None, "none"

    def row_signature(self, frame: pd.DataFrame) -> np.ndarray:
        cut = ns(frame.cutoff_utc)
        return np.array([f"{self.resolve_starter(g, 'home', c)[0]}|{self.resolve_starter(g, 'away', c)[0]}"
                         for g, c in zip(frame.game_id.values, cut)], dtype=object).astype(str)

    # ------------------------------------------------------------------ observations
    def _prep(self):
        d = self.done
        av = ns(d.avail)
        # starter of each completed game: box score (post-game) if present, else the announced backfill starter
        spmap = {}
        if len(self.stats):
            for gid, tid, sid in zip(self.stats.game_id, self.stats.team_id, self.stats.sp_id):
                if sid is not None and sid == sid:
                    spmap[(gid, tid)] = str(sid)
        sp_h = [spmap.get((g, h)) or self._backfill(g, "home") for g, h in zip(d.game_id, d.home_id)]
        sp_a = [spmap.get((g, a)) or self._backfill(g, "away") for g, a in zip(d.game_id, d.away_id)]
        loc_h = np.where(d.neutral.astype(bool), 0, 1)
        self.runs = pd.DataFrame({
            "season": np.r_[d.season, d.season].astype(int), "avail": np.r_[av, av],
            "off": np.r_[d.home_id, d.away_id], "dfn": np.r_[d.away_id, d.home_id],
            "sp": np.r_[np.array(sp_a, dtype=object), np.array(sp_h, dtype=object)],   # the DEFENSE's starter
            "loc": np.r_[loc_h, -loc_h], "y": np.r_[d.home_score, d.away_score].astype(float)}).sort_values("avail", kind="stable")
        self._runs_by_season = {s: g.reset_index(drop=True) for s, g in self.runs.groupby("season")}
        # starters' appearances (all leagues) and lines (leagues with box scores)
        s = self.stats.merge(d[["game_id", "avail", "home_id", "away_id"]], on="game_id", how="inner") if len(self.stats) else pd.DataFrame()
        self.sp_app = {}
        self.sp_cum, self.lg_cum, self.pen_cum, self.lg_pen_cum, self.pen_days = {}, {}, {}, {}, {}
        self.rot = {}
        if len(s):
            s = s[s.sp_id.notna()].copy()
            s["sp_id"] = s.sp_id.astype(str)
            s["av"] = ns(s.avail); s["ko"] = ns(s.kickoff_utc)
            s = s.sort_values("av", kind="stable")
            for pid, g in s.groupby("sp_id"):
                self.sp_app[pid] = g.ko.values
            for (tid, season), g in s.groupby(["team_id", "season"]):
                self.rot[(tid, int(season))] = (g.av.values, g.sp_id.values)
            if self.HAS_LINES and "sp_outs" in s:
                L = s[s.sp_outs.notna() & (s.sp_outs > 0)].copy()
                for c in LINE:
                    L[f"sp_{c}"] = L[f"sp_{c}"].astype(float).fillna(0.0)
                vals = np.column_stack([L[f"sp_{c}"].values for c in LINE] + [np.ones(len(L))])
                for (pid, season), idx in L.groupby(["sp_id", "season"]).indices.items():
                    self.sp_cum[(pid, int(season))] = _Cum(L.av.values[idx], vals[idx])
                for season, idx in L.groupby("season").indices.items():
                    self.lg_cum[int(season)] = _Cum(L.av.values[idx], vals[idx])
                P = s[s.tm_outs.notna() & s.sp_outs.notna()].copy()
                pv = np.column_stack([(P[f"tm_{c}"].astype(float).fillna(0) - P[f"sp_{c}"].astype(float).fillna(0)).clip(lower=0).values
                                      for c in LINE])
                for (tid, season), idx in P.groupby(["team_id", "season"]).indices.items():
                    self.pen_cum[(tid, int(season))] = _Cum(P.av.values[idx], pv[idx])
                for season, idx in P.groupby("season").indices.items():
                    self.lg_pen_cum[int(season)] = _Cum(P.av.values[idx], pv[idx])
                for tid, idx in P.groupby("team_id").indices.items():
                    o = np.argsort(P.ko.values[idx], kind="stable")
                    self.pen_days[tid] = (P.ko.values[idx][o], P.av.values[idx][o], np.r_[0.0, np.cumsum(pv[idx][o][:, 0])])
        self.stats_avail = np.sort(s.av.values) if len(s) else np.array([], dtype="int64")
        self.done_avail = np.sort(av)
        # team histories for sequence features
        th = pd.DataFrame({"game_id": np.r_[d.game_id, d.game_id], "tid": np.r_[d.home_id, d.away_id],
                           "season": np.r_[d.season, d.season].astype(int), "ko": np.r_[ns(d.kickoff_utc), ns(d.kickoff_utc)],
                           "av": np.r_[av, av], "margin": np.r_[d.home_score - d.away_score, d.away_score - d.home_score],
                           "is_home": np.r_[np.ones(len(d), bool), np.zeros(len(d), bool)],
                           "neutral": np.r_[d.neutral, d.neutral].astype(bool)}).sort_values(["tid", "season", "av"], kind="stable")
        self._hist = {k: g.reset_index(drop=True) for k, g in th.groupby(["tid", "season"])}
        # park: total runs in each team's home games vs the league
        pk = pd.DataFrame({"tid": d.home_id.values, "season": d.season.values.astype(int), "av": av,
                           "tot": (d.home_score + d.away_score).values.astype(float)})
        self.park_cum = {(t, int(se)): _Cum(g.av.values, np.column_stack([g.tot.values, np.ones(len(g))]))
                         for (t, se), g in pk.groupby(["tid", "season"])}
        self.lg_tot_cum = {int(se): _Cum(g.av.values, np.column_stack([g.tot.values, np.ones(len(g))])) for se, g in pk.groupby("season")}

    def _backfill(self, gid, side):
        rows = [r for r in self._st.get((gid, side), []) if r[0] == "backfill"]
        return rows[0][2] if rows else None

    # ------------------------------------------------------------------ helpers
    def visible_counts(self, cutoffs: pd.Series):
        c = ns(cutoffs)
        return np.searchsorted(self.done_avail, c, side="right"), np.searchsorted(self.stats_avail, c, side="right")

    def training_cutoff(self, kickoff: pd.Series) -> pd.Series:
        """Historical reconstruction cutoff: 05:00 local time on game day (never after first pitch)."""
        loc = kickoff.dt.tz_convert(self.TZ)
        c = (loc.dt.normalize() + pd.Timedelta(hours=5)).dt.tz_convert("UTC")
        return pd.concat([c, kickoff], axis=1).min(axis=1)

    def _obs(self, season, cutoff_ns):
        o = self._runs_by_season.get(season)
        if o is None:
            return self.runs.iloc[:0]
        return o.iloc[:np.searchsorted(o.avail.values, cutoff_ns, side="right")]

    def _fit(self, season, cutoff_ns, extra_teams=(), extra_pitchers=()):
        o = self._obs(season, cutoff_ns)
        prev = self._final(season - 1, cutoff_ns)
        teams = sorted(set(self._teams_by_season.get(season, [])) | set(o.off) | set(o.dfn) | set(extra_teams))
        pitchers = sorted({p for p in o.sp if _ok(p)} | {p for p in extra_pitchers if _ok(p)})
        ti = {t: i for i, t in enumerate(teams)}; pi = {p: i for i, p in enumerate(pitchers)}
        if prev is not None:
            pmu, po, pdf, pp = prev
            prior_o = np.array([self.SHRINK * po.get(t, 0.0) for t in teams])
            prior_d = np.array([self.SHRINK * pdf.get(t, 0.0) for t in teams])
            prior_p = np.array([self.SP_SHRINK * pp.get(p, 0.0) for p in pitchers])
        else:
            pmu = None
            prior_o = np.zeros(len(teams)); prior_d = np.zeros(len(teams)); prior_p = np.zeros(len(pitchers))
        mu, h, ro, rd, rp = ridge_runs(np.fromiter((ti[t] for t in o.off), int, len(o)), np.fromiter((ti[t] for t in o.dfn), int, len(o)),
                                       np.fromiter((pi.get(p, -1) if _ok(p) else -1 for p in o.sp), int, len(o)),
                                       o["loc"].values.astype(float), o.y.values, len(teams), len(pitchers),
                                       prior_o, prior_d, prior_p, self.LAM_T, self.LAM_P, self.SP_W)
        if not len(o) and pmu is not None:
            mu = pmu
        return mu, h, dict(zip(teams, ro)), dict(zip(teams, rd)), dict(zip(pitchers, rp)), len(o)

    def _final(self, season, cutoff_ns):
        if season < self.FIRST_SEASON or season not in self._runs_by_season:
            return None
        n = len(self._obs(season, cutoff_ns))
        key = (season, n)
        if key not in self._finals:
            mu, h, ro, rd, rp, _ = self._fit(season, cutoff_ns)
            self._finals[key] = (mu, ro, rd, rp)
        return self._finals[key]

    # ---- pitcher / bullpen rates as of the cutoff (deviation from league average; + = more runs allowed)
    def _lines(self, cums: dict, key_fn, season, c, width: int):
        """Current-season sums plus PREV_W x last season's, as of the cutoff."""
        cur = cums.get(key_fn(season)); prv = cums.get(key_fn(season - 1))
        v = np.zeros(width) if cur is None else cur.at(c).copy()
        if prv is not None:
            v += self.PREV_W * prv.at(c)
        return v

    def _league(self, season, c):
        v = self._lines(self.lg_cum, lambda s: s, season, c, len(LINE) + 1)
        ip = max(v[0] / 3, 1.0)
        return {"fip": (13 * v[1] + 3 * v[2] - 2 * v[3]) / ip, "ra9": 9 * v[4] / ip, "kbb": 9 * (v[3] - v[2]) / ip,
                "depth": v[0] / max(v[6], 1) / 3}

    def pitcher_rates(self, pid, season, c, lg):
        """Rates regressed to the league, as deviations from the league average (0 = average / no data):
        FIP-type (13HR+3BB-2K)/IP (ERA scale, no constant), runs allowed per 9, (K-BB) per 9, innings per start."""
        out = {"fip": 0.0, "ra9": 0.0, "kbb": 0.0, "depth": 0.0, "starts": 0.0}
        if not self.HAS_LINES or pid is None:
            return out
        v = self._lines(self.sp_cum, lambda s: (pid, s), season, c, len(LINE) + 1)
        cur = self.sp_cum.get((pid, season))
        out["starts"] = float(cur.at(c)[6]) if cur is not None else 0.0
        ip, outs, st = v[0] / 3, v[0], v[6]
        out["fip"] = ((13 * v[1] + 3 * v[2] - 2 * v[3]) + lg["fip"] * self.IP0_FIP) / (ip + self.IP0_FIP) - lg["fip"]
        out["ra9"] = (9 * v[4] + lg["ra9"] * self.IP0_RA) / (ip + self.IP0_RA) - lg["ra9"]
        out["kbb"] = (9 * (v[3] - v[2]) + lg["kbb"] * self.IP0_FIP) / (ip + self.IP0_FIP) - lg["kbb"]
        out["depth"] = (outs / 3 + lg["depth"] * self.ST0) / (st + self.ST0) - lg["depth"]
        return out

    def bullpen(self, tid, season, c, kick_ns):
        if not self.HAS_LINES:
            return 0.0, 0.0
        v = self._lines(self.pen_cum, lambda s: (tid, s), season, c, len(LINE))
        L = self._lines(self.lg_pen_cum, lambda s: s, season, c, len(LINE))
        lg_ip = max(L[0] / 3, 1.0)
        lg_fip = (13 * L[1] + 3 * L[2] - 2 * L[3]) / lg_ip
        fip = (13 * v[1] + 3 * v[2] - 2 * v[3] + lg_fip * self.IP0_PEN) / (v[0] / 3 + self.IP0_PEN) - lg_fip
        load = 0.0
        pdays = self.pen_days.get(tid)
        if pdays is not None:
            ko, av, cum = pdays
            lo = np.searchsorted(ko, kick_ns - 3 * DAY, side="left")
            hi = min(np.searchsorted(ko, kick_ns, side="left"), np.searchsorted(av, c, side="right"))
            if hi > lo:
                load = float(cum[hi] - cum[lo]) / 3
        return fip, load

    def rest_days(self, pid, kick_ns, c):
        if pid is None:
            return 5.0
        ap = self.sp_app.get(pid)
        if ap is None:
            return 10.0
        # last appearance whose box score was available at the cutoff
        prior = ap[(ap < kick_ns) & (ap + int(self.guard.duration.value) <= c)]
        return float(min((kick_ns - prior.max()) / DAY, 10.0)) if len(prior) else 10.0

    def rotation(self, tid, season, c, k=5):
        """Starters of the team's last k starts visible at the cutoff (this season, else last season)."""
        for s in (season, season - 1):
            r = self.rot.get((tid, s))
            if r is None:
                continue
            n = np.searchsorted(r[0], c, side="right")
            if n:
                return list(r[1][max(0, n - k):n])
        return []

    def _seq(self, tid, season, c, kick_ns, elo):
        res = {"form10": 0.0, "rest": 1.0, "n_prior": 0, "g7": 0.0}
        h = self._hist.get((tid, season))
        if h is not None:
            k = int(np.searchsorted(h.av.values, c, side="right"))
            res["n_prior"] = k
            if k:
                vis = h.iloc[max(0, k - 10):k]
                resid = []
                for r in vis.itertuples():
                    pre = elo.pre.get(r.game_id)
                    if pre is None:
                        continue
                    mine, theirs = (pre[0], pre[1]) if r.is_home else (pre[1], pre[0])
                    hfa = 0 if r.neutral else (self.ELO["HFA"] if r.is_home else -self.ELO["HFA"])
                    resid.append(r.margin - (mine - theirs + hfa) / self.ELO_PTS)
                res["form10"] = float(np.mean(resid)) if resid else 0.0
                last_ko = h.ko.values[k - 1]
                res["rest"] = float(min(max((kick_ns - last_ko) / DAY - 0.5, 0.0), 4.0))
                res["g7"] = float(((h.ko.values[:k] >= kick_ns - 7 * DAY)).sum())
        return res

    def _park(self, tid, season, c):
        v = np.zeros(2); L = np.zeros(2)
        for s, w in ((season, 1.0), (season - 1, self.PREV_W)):
            if (tid, s) in self.park_cum:
                v += w * self.park_cum[(tid, s)].at(c)
            if s in self.lg_tot_cum:
                L += w * self.lg_tot_cum[s].at(c)
        lg = L[0] / L[1] if L[1] else 0.0
        return (v[0] + lg * 40) / (v[1] + 40) - lg if L[1] else 0.0

    # ------------------------------------------------------------------ main entry
    def build(self, targets: pd.DataFrame) -> pd.DataFrame:
        t = targets.copy()
        t["kickoff_utc"] = pd.to_datetime(t.kickoff_utc, utc=True)
        t["cutoff_utc"] = pd.to_datetime(t.cutoff_utc, utc=True)
        if (t.cutoff_utc > t.kickoff_utc).any():
            raise ValueError("cutoff after kickoff")
        t["season"] = t.season.astype(int)
        t = t.sort_values(["cutoff_utc", "game_id"])
        elo = EloEngine(self.done, self.done.avail, lambda s, tid: "ALL", **self.ELO, init={"ALL": 1500.0})
        rows = []
        for (cutoff, season), grp in t.groupby(["cutoff_utc", "season"], sort=True):
            c = ns(cutoff)
            elo.advance(cutoff)
            elo.snapshot(season)
            res = {}
            for g in grp.itertuples():
                for side, tid in (("home", g.home_id), ("away", g.away_id)):
                    pid, nm, src = self.resolve_starter(g.game_id, side, c)
                    res[(g.game_id, side)] = (pid, nm, src, [] if pid is not None else self.rotation(tid, season, c))
            extra_p = {r[0] for r in res.values() if _ok(r[0])} | {p for r in res.values() for p in r[3]}
            mu, hfa, ro, rd, rp, n_obs = self._fit(season, c, extra_teams=set(grp.home_id) | set(grp.away_id), extra_pitchers=extra_p)
            o = self._obs(season, c)
            self.guard.check(pd.DataFrame({"kickoff_utc": pd.to_datetime(o.avail.values, utc=True) - self.guard.duration}), cutoff, "run ratings")
            lg = self._league(season, c) if self.HAS_LINES else None
            n_used = int(np.searchsorted(self.done_avail, c, side="right"))
            n_stats = int(np.searchsorted(self.stats_avail, c, side="right"))
            for g in grp.itertuples():
                kick = ns(g.kickoff_utc)
                loc = 0 if bool(g.neutral) else 1
                f = {"game_id": g.game_id, "cutoff_utc": cutoff, "n_games_visible": n_used, "n_stats_visible": n_stats}
                eh, ea = elo.rating(season, g.home_id), elo.rating(season, g.away_id)
                f.update(elo_h=eh, elo_a=ea, elo_diff=eh - ea + loc * self.ELO["HFA"])
                sig = []
                for side, tid in (("h", g.home_id), ("a", g.away_id)):
                    pid, nm, src, rot = res[(g.game_id, "home" if side == "h" else "away")]
                    sig.append(str(pid))
                    f[f"sp_id_{side}"], f[f"sp_name_{side}"], f[f"sp_src_{side}"] = pid, nm, src
                    f[f"sp_known_{side}"] = float(pid is not None)
                    if pid is not None:
                        f[f"sp_eff_{side}"] = self.SP_W * rp.get(pid, 0.0)
                        rates = self.pitcher_rates(pid, season, c, lg)
                        f[f"sp_rest_{side}"] = self.rest_days(pid, kick, c)
                    else:                                   # TBD: the team's recent rotation on average
                        f[f"sp_eff_{side}"] = self.SP_W * float(np.mean([rp.get(p, 0.0) for p in rot])) if rot else 0.0
                        rr = [self.pitcher_rates(p, season, c, lg) for p in rot] or [self.pitcher_rates(None, season, c, lg)]
                        rates = {k: float(np.mean([r[k] for r in rr])) for k in rr[0]}
                        f[f"sp_rest_{side}"] = 5.0
                    for k, v in rates.items():
                        f[f"sp_{k}_{side}"] = v
                    f[f"pen_fip_{side}"], f[f"pen_load_{side}"] = self.bullpen(tid, season, c, kick)
                    f[f"off_{side}"], f[f"def_{side}"] = ro.get(tid, 0.0), rd.get(tid, 0.0)
                    s = self._seq(tid, season, c, kick, elo)
                    for k, v in s.items():
                        f[f"{k}_{side}"] = v
                f["row_signature"] = "|".join(sig)
                f["pp_home"] = mu + f["off_h"] + f["def_a"] + f["sp_eff_a"] + hfa * loc
                f["pp_away"] = mu + f["off_a"] + f["def_h"] + f["sp_eff_h"] - hfa * loc
                f["hfa_runs"] = 2 * hfa
                f["park_h"] = self._park(g.home_id, season, c)
                rows.append(f)
        F = pd.DataFrame(rows)
        F["pts_margin_pred"] = F.pp_home - F.pp_away
        F["pts_total_pred"] = F.pp_home + F.pp_away
        F["team_diff"] = (F.off_h - F.def_h) - (F.off_a - F.def_a)
        F["sp_eff_diff"] = F.sp_eff_a - F.sp_eff_h
        F["sp_eff_sum"] = F.sp_eff_h + F.sp_eff_a
        F["off_sum"] = F.off_h + F.off_a
        F["def_sum"] = F.def_h + F.def_a
        F["sp_fip_diff"] = F.sp_fip_a - F.sp_fip_h
        F["sp_ra9_diff"] = F.sp_ra9_a - F.sp_ra9_h
        F["sp_kbb_diff"] = F.sp_kbb_h - F.sp_kbb_a
        F["sp_depth_diff"] = F.sp_depth_h - F.sp_depth_a
        F["pen_fip_diff"] = F.pen_fip_a - F.pen_fip_h
        F["pen_load_diff"] = F.pen_load_a - F.pen_load_h
        F["sp_fip_sum"] = F.sp_fip_h + F.sp_fip_a
        F["sp_ra9_sum"] = F.sp_ra9_h + F.sp_ra9_a
        F["sp_depth_sum"] = F.sp_depth_h + F.sp_depth_a
        F["pen_fip_sum"] = F.pen_fip_h + F.pen_fip_a
        F["form_diff"] = F.form10_h - F.form10_a
        F["rest_diff"] = F.rest_h - F.rest_a
        F["early"] = ((F.n_prior_h + F.n_prior_a) < self.EARLY_GAMES).astype(float)
        meta = t[["game_id", "season", "kickoff_utc", "home_id", "away_id", "neutral", "conf_game"]].copy()
        meta["neutral"] = meta.neutral.astype(float); meta["conf_game"] = meta.conf_game.astype(float)
        out = meta.merge(F, on="game_id", how="left")
        out["feature_version"] = self.feature_version
        return out
