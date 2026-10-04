"""Team-strength estimators used by the feature builder: margin-of-victory Elo and opponent-adjusted
ridge ratings (offense/defense with a home term and a preseason prior from last season)."""
from __future__ import annotations
import numpy as np
from ...core.timeutil import ns
import pandas as pd
import scipy.sparse as sp

ELO_INIT = {"FBS": 1500.0, "FCS": 1200.0, "OTHER": 950.0}


class EloEngine:
    """Processes completed games in kickoff order. `advance(cutoff, visible_mask_fn)` applies every
    game whose result is available at the cutoff; `snapshot(season)` returns ratings regressed for the
    target season if the season changed. Pre-game ratings of every processed game are recorded."""

    def __init__(self, games: pd.DataFrame, avail: pd.Series, div_of, K=36.0, HFA=45.0, revert=0.25, init=None):
        self.g = games.reset_index(drop=True)
        self.avail = ns(avail.reset_index(drop=True))
        self.div_of = div_of
        self.init = init or ELO_INIT
        self.K, self.HFA, self.revert = K, HFA, revert
        self.r: dict[str, float] = {}
        self.i = 0
        self.season = None
        self.pre = {}                     # game_id -> (elo_home_pre, elo_away_pre)
        self._h = self.g.home_id.values; self._a = self.g.away_id.values
        self._hs = self.g.home_score.values; self._as = self.g.away_score.values
        self._neu = self.g.neutral.values.astype(bool); self._sea = self.g.season.values
        self._gid = self.g.game_id.values

    def _new_season(self, season):
        if self.season is not None and season != self.season:
            for t in list(self.r):
                d = self.div_of(season - 1, t)
                self.r[t] = self.init[d] + (1 - self.revert) * (self.r[t] - self.init[d])
        self.season = season

    def rating(self, season, t):
        if t not in self.r:
            self.r[t] = self.init[self.div_of(season, t)]
        return self.r[t]

    def advance(self, cutoff):
        n = len(self.g); cutoff = ns(cutoff)
        while self.i < n and self.avail[self.i] <= cutoff:
            i = self.i
            s = int(self._sea[i])
            if self.season is None or s > self.season:
                self._new_season(s)
            h, a = self._h[i], self._a[i]
            rh, ra = self.rating(s, h), self.rating(s, a)
            self.pre[self._gid[i]] = (rh, ra)
            diff = rh + (0 if self._neu[i] else self.HFA) - ra
            exp = 1 / (1 + 10 ** (-diff / 400))
            m = self._hs[i] - self._as[i]
            res = 1.0 if m > 0 else (0.0 if m < 0 else 0.5)
            mult = np.log(abs(m) + 1) * 2.2 / ((diff if m > 0 else -diff) * 0.001 + 2.2)
            dlt = self.K * mult * (res - exp)
            self.r[h] = rh + dlt; self.r[a] = ra - dlt
            self.i += 1

    def snapshot(self, season):
        if self.season is None or season > self.season:
            self._new_season(season)
        return self


def ridge_ratings(off, dfn, loc, y, teams, prior_o, prior_d, lam):
    """y = mu + o[off] + d[dfn] + h*loc, ridge-shrunk toward priors. Returns (mu, h, o, d) with o, d dicts."""
    idx = {t: i for i, t in enumerate(teams)}; n = len(teams); P = 2 + 2 * n
    po = np.array([prior_o.get(t, 0.0) for t in teams]); pdv = np.array([prior_d.get(t, 0.0) for t in teams])
    if len(y) == 0:
        return 0.0, 0.0, dict(zip(teams, po)), dict(zip(teams, pdv))
    oi = np.fromiter((idx[t] for t in off), int, len(off)); di = np.fromiter((idx[t] for t in dfn), int, len(dfn))
    r = len(y); rows = np.repeat(np.arange(r), 4)
    cols = np.column_stack([np.zeros(r, int), np.ones(r, int), 2 + oi, 2 + n + di]).ravel()
    vals = np.column_stack([np.ones(r), loc, np.ones(r), np.ones(r)]).ravel()
    X = sp.csr_matrix((vals, (rows, cols)), shape=(r, P))
    A = (X.T @ X).toarray(); b = X.T @ np.asarray(y, float)
    A[0, 0] += 1e-3; A[1, 1] += 1e-3 + (50 if r < 50 else 0)
    A[2:2 + n, 2:2 + n] += np.eye(n) * lam; b[2:2 + n] += lam * po
    A[2 + n:, 2 + n:] += np.eye(n) * lam; b[2 + n:] += lam * pdv
    th = np.linalg.solve(A, b)
    return float(th[0]), float(th[1]), dict(zip(teams, th[2:2 + n])), dict(zip(teams, th[2 + n:]))
