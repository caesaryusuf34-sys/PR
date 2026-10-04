"""Configurable ensemble used by every sport.

config = {
  "features":       model inputs,
  "total_features": inputs of the totals model,
  "members":        {name: params} from MEMBER_TYPES,
  "ensemble":       "simplex" (convex weights minimising log loss) | "stack" (logistic meta-learner),
  "stack_context":  extra columns the stacker may use (lets the system learn subgroup corrections),
  "calibration":    "none" | "platt" | "isotonic",
  "half_life_days": recency weighting of training rows (None = equal weights),
  "min_train_season", "holdout_frac"
}
Fitting: members are fit on the older (1 - holdout_frac) of the training rows, the newest rows are
used to fit ensemble weights / stacker / calibrator out-of-sample, then members are refit on all rows.
"""
from __future__ import annotations
import copy
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

from ..core.metrics import logloss_vec

EPS = 1e-4


def _logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


# ------------------------------------------------------------------------------------------ members
class Member:
    has_prob = has_margin = False
    def __init__(self, features, **params):
        self.features, self.params = list(features), params
    def X(self, df):
        return df[self.features].astype(float).fillna(0.0).values


class EloMember(Member):
    """Win probability and margin from the Elo difference alone."""
    has_prob = has_margin = True
    def fit(self, df, y, m, w):
        x = df[["elo_diff"]].fillna(0).values / 400
        self.lr = LogisticRegression(C=1e6, max_iter=1000).fit(x, y, sample_weight=w)
        self.rg = Ridge(alpha=1e-6).fit(x, m, sample_weight=w); return self
    def predict(self, df):
        x = df[["elo_diff"]].fillna(0).values / 400
        return self.lr.predict_proba(x)[:, 1], self.rg.predict(x)


class ScoringMember(Member):
    """Opponent-adjusted scoring model: calibrated margin from the ridge points ratings, p = Phi(m / sigma)."""
    has_prob = has_margin = True
    def fit(self, df, y, m, w):
        x = df[["pts_margin_pred"]].fillna(0).values
        self.rg = Ridge(alpha=1e-6).fit(x, m, sample_weight=w)
        self.sigma = float(np.sqrt(np.average((m - self.rg.predict(x)) ** 2, weights=w))); return self
    def predict(self, df):
        mm = self.rg.predict(df[["pts_margin_pred"]].fillna(0).values)
        return norm.cdf(mm / self.sigma), mm


class LogisticMember(Member):
    has_prob = True
    def fit(self, df, y, m, w):
        self.model = make_pipeline(StandardScaler(), LogisticRegression(C=self.params.get("C", 0.1), max_iter=3000))
        self.model.fit(self.X(df), y, logisticregression__sample_weight=w); return self
    def predict(self, df):
        return self.model.predict_proba(self.X(df))[:, 1], None
    def contributions(self, df):
        sc, lr = self.model.named_steps["standardscaler"], self.model.named_steps["logisticregression"]
        z = (self.X(df) - sc.mean_) / np.where(sc.scale_ == 0, 1, sc.scale_)
        return pd.DataFrame(z * lr.coef_[0], columns=self.features, index=df.index)


def _lgb_params(p):
    d = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=40, subsample=0.8,
             subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0, verbose=-1, random_state=7)
    d.update(p); return d


class GBMMember(Member):
    has_prob = True
    def fit(self, df, y, m, w):
        self.model = lgb.LGBMClassifier(**_lgb_params(self.params)).fit(self.X(df), y, sample_weight=w); return self
    def predict(self, df):
        return self.model.predict_proba(self.X(df))[:, 1], None


class MarginRidgeMember(Member):
    has_prob = has_margin = True
    def fit(self, df, y, m, w):
        self.model = make_pipeline(StandardScaler(), Ridge(alpha=self.params.get("alpha", 10.0)))
        self.model.fit(self.X(df), m, ridge__sample_weight=w)
        self.sigma = float(np.sqrt(np.average((m - self.model.predict(self.X(df))) ** 2, weights=w))); return self
    def predict(self, df):
        mm = self.model.predict(self.X(df)); return norm.cdf(mm / self.sigma), mm


class GBMMarginMember(Member):
    has_margin = True
    def fit(self, df, y, m, w):
        self.model = lgb.LGBMRegressor(**_lgb_params(self.params)).fit(self.X(df), m, sample_weight=w); return self
    def predict(self, df):
        return None, self.model.predict(self.X(df))


MEMBER_TYPES = {"elo": EloMember, "scoring": ScoringMember, "logistic": LogisticMember, "gbm": GBMMember,
                "margin_ridge": MarginRidgeMember, "gbm_margin": GBMMarginMember}
MEMBER_LABELS = {"elo": "Elo", "scoring": "Adj. scoring model", "logistic": "Logistic regression",
                 "gbm": "Gradient boosting", "margin_ridge": "Point-diff regression", "gbm_margin": "GBM margin regression"}


def simplex_weights(P, target, loss):
    k = P.shape[1]
    if k == 1:
        return np.ones(1)
    f = lambda w: loss(P @ (np.abs(w) / np.abs(w).sum()), target)
    best = None
    for start in (np.ones(k) / k, *np.eye(k) * 0.9 + 0.1 / k):
        r = minimize(f, start, method="Nelder-Mead", options={"maxiter": 3000, "xatol": 1e-5, "fatol": 1e-8})
        if best is None or r.fun < best.fun:
            best = r
    return np.abs(best.x) / np.abs(best.x).sum()


# ------------------------------------------------------------------------------------------ ensemble
class EnsembleModel:
    def __init__(self, config: dict, feature_version: str):
        self.config = copy.deepcopy(config)
        self.features = list(config["features"])
        self.feature_version = feature_version

    # ---- helpers
    def _weights(self, df):
        hl = self.config.get("half_life_days")
        if not hl:
            return np.ones(len(df))
        age = (df.kickoff_utc.max() - df.kickoff_utc).dt.total_seconds() / 86400
        return 0.5 ** (age.values / hl)

    def _fit_members(self, df):
        y = df.home_win.values.astype(int); m = df.margin.values.astype(float); w = self._weights(df)
        mem = {}
        for name, params in self.config["members"].items():
            mem[name] = MEMBER_TYPES[name](self.features, **(params or {})).fit(df, y, m, w)
        return mem

    @staticmethod
    def _member_preds(members, df):
        P, M = {}, {}
        for name, mb in members.items():
            p, mm = mb.predict(df)
            if mb.has_prob and p is not None:
                P[name] = np.clip(p, EPS, 1 - EPS)
            if mb.has_margin and mm is not None:
                M[name] = mm
        return pd.DataFrame(P, index=df.index), pd.DataFrame(M, index=df.index)

    def _stack_X(self, P, df):
        cols = [_logit(P[c].values) for c in self.p_names]
        cols += [df[c].astype(float).fillna(0).values for c in self.config.get("stack_context", [])]
        return np.column_stack(cols)

    # ---- API
    def fit(self, df: pd.DataFrame):
        df = df.sort_values("kickoff_utc").reset_index(drop=True)
        n_hold = max(int(len(df) * self.config.get("holdout_frac", 0.2)), 200)
        early, hold = df.iloc[:-n_hold], df.iloc[-n_hold:]
        mem = self._fit_members(early)
        P, M = self._member_preds(mem, hold)
        self.p_names, self.m_names = list(P.columns), list(M.columns)
        y, m = hold.home_win.values, hold.margin.values
        if self.config.get("ensemble", "simplex") == "stack":
            self.stacker = LogisticRegression(C=1.0, max_iter=2000).fit(self._stack_X(P, hold), y)
            p_raw = self.stacker.predict_proba(self._stack_X(P, hold))[:, 1]
            self.wp = None
        else:
            self.stacker = None
            self.wp = simplex_weights(P.values, y, lambda p, t: logloss_vec(p, t).mean())
            p_raw = P.values @ self.wp
        self.wm = simplex_weights(M.values, m, lambda p, t: np.mean((p - t) ** 2))
        self.sigma = float(np.std(m - M.values @ self.wm))
        cal = self.config.get("calibration", "none")
        if cal == "platt":
            self.calibrator = LogisticRegression(C=1e6).fit(_logit(p_raw).reshape(-1, 1), y)
        elif cal == "isotonic":
            self.calibrator = IsotonicRegression(y_min=0.005, y_max=0.995, out_of_bounds="clip").fit(p_raw, y)
        else:
            self.calibrator = None
        # refit members on all rows; totals model on all rows
        self.members = self._fit_members(df)
        tf = self.config.get("total_features") or []
        if tf:
            self.total_model = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(df[tf].astype(float).fillna(0).values, df.total.values)
        else:
            self.total_model = None
        self.feature_stats = {c: (float(df[c].mean()), float(df[c].std() or 1.0)) for c in self.config.get("explain_features", self.features) if c in df}
        self.n_train = len(df)
        self.train_end = df.kickoff_utc.max()
        return self

    def _calibrate(self, p):
        if self.calibrator is None:
            return p
        if isinstance(self.calibrator, IsotonicRegression):
            return np.clip(self.calibrator.predict(p), EPS, 1 - EPS)
        return self.calibrator.predict_proba(_logit(p).reshape(-1, 1))[:, 1]

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        P, M = self._member_preds(self.members, df)
        out = pd.DataFrame(index=df.index)
        for c in P: out[f"p_{c}"] = P[c]
        for c in M: out[f"m_{c}"] = M[c]
        p_raw = self.stacker.predict_proba(self._stack_X(P, df))[:, 1] if self.stacker is not None else P[self.p_names].values @ self.wp
        out["p_raw"] = p_raw
        out["p_home"] = self._calibrate(p_raw)
        out["m_ens"] = M[self.m_names].values @ self.wm
        out["margin"] = 0.5 * out.m_ens + 0.5 * self.sigma * norm.ppf(out.p_home.clip(0.001, 0.999))
        tf = self.config.get("total_features") or []
        out["total"] = self.total_model.predict(df[tf].astype(float).fillna(0).values) if self.total_model is not None else np.nan
        out["p_std"] = P.std(axis=1) if P.shape[1] > 1 else 0.0
        out["models_split"] = ((P.min(axis=1) < 0.5) & (P.max(axis=1) > 0.5)).astype(int)
        return out

    def weights(self) -> dict:
        return {"prob": dict(zip(self.p_names, map(float, self.wp))) if self.wp is not None else "stacked",
                "margin": dict(zip(self.m_names, map(float, self.wm))), "sigma": self.sigma}

    def contributions(self, df) -> pd.DataFrame | None:
        """Per-feature log-odds contributions: TreeSHAP of the gradient-boosting member when present
        (robust to correlated inputs), else the standardized logistic-regression terms."""
        gb = self.members.get("gbm")
        if gb is not None:
            c = gb.model.booster_.predict(gb.X(df), pred_contrib=True)
            return pd.DataFrame(c[:, :-1], columns=gb.features, index=df.index)
        lr = self.members.get("logistic")
        return lr.contributions(df) if lr is not None else None
