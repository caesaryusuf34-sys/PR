"""Probabilistic and point-forecast evaluation metrics."""
from __future__ import annotations
import numpy as np
import pandas as pd

EPS = 1e-4


def logloss_vec(p, y):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS); y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def brier_vec(p, y):
    return (np.asarray(p, float) - np.asarray(y, float)) ** 2


def calibration_table(p, y, bins=10) -> pd.DataFrame:
    p = np.asarray(p, float); y = np.asarray(y, float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        if m.sum():
            rows.append({"bin": f"{edges[b]:.1f}-{edges[b+1]:.1f}", "n": int(m.sum()),
                         "mean_pred": float(p[m].mean()), "actual": float(y[m].mean())})
    return pd.DataFrame(rows)


def ece(p, y, bins=10) -> float:
    t = calibration_table(p, y, bins)
    return float((t.n * (t.mean_pred - t.actual).abs()).sum() / max(t.n.sum(), 1)) if len(t) else float("nan")


def calibration_slope(p, y):
    """Logistic recalibration y ~ a + b*logit(p). b<1 => over-confident, b>1 => under-confident.
    Returns (a, b, se_a, se_b) from the Fisher information."""
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS); y = np.asarray(y, float)
    X = np.column_stack([np.ones_like(p), np.log(p / (1 - p))])
    beta = np.array([0.0, 1.0])
    for _ in range(50):
        mu = 1 / (1 + np.exp(-X @ beta))
        W = mu * (1 - mu)
        H = X.T @ (X * W[:, None]) + 1e-9 * np.eye(2)
        step = np.linalg.solve(H, X.T @ (y - mu))
        beta += step
        if np.abs(step).max() < 1e-8:
            break
    cov = np.linalg.inv(H)
    return float(beta[0]), float(beta[1]), float(np.sqrt(cov[0, 0])), float(np.sqrt(cov[1, 1]))


def summarize(p, y, margin_pred=None, margin=None, total_pred=None, total=None) -> dict:
    p = np.asarray(p, float); y = np.asarray(y, float)
    out = {"n": int(len(y))}
    if len(y) == 0:
        return out
    out.update(accuracy=float(((p >= 0.5) == (y == 1)).mean()), brier=float(brier_vec(p, y).mean()),
               log_loss=float(logloss_vec(p, y).mean()), ece=ece(p, y))
    if len(y) >= 30:
        a, b, sa, sb = calibration_slope(p, y)
        out.update(calib_intercept=a, calib_slope=b, calib_slope_se=sb)
    if margin_pred is not None and margin is not None:
        e = np.asarray(margin_pred, float) - np.asarray(margin, float)
        out.update(margin_mae=float(np.abs(e).mean()), margin_rmse=float(np.sqrt((e ** 2).mean())),
                   margin_bias=float(e.mean()))
    if total_pred is not None and total is not None:
        e = np.asarray(total_pred, float) - np.asarray(total, float)
        out.update(total_mae=float(np.abs(e).mean()), total_bias=float(e.mean()))
    return out


def bh_adjust(pvals) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(pvals, float); n = len(p)
    if n == 0:
        return p
    order = np.argsort(p); ranked = p[order] * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n); out[order] = np.minimum(adj, 1.0)
    return out
