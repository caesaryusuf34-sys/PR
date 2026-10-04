import sqlite3
import numpy as np
import pandas as pd
import pytest

from sportsai.core.db import Store
from sportsai.core.leakage import TemporalGuard, LeakageError, snapshot_hash
from sportsai.core.metrics import summarize, bh_adjust, calibration_slope, logloss_vec
from sportsai.core.engine import Engine, confidence_score


def _pred(**kw):
    d = dict(pred_id="p1", sport="t", game_id="g1", home_id="h", away_id="a", home_name="H", away_name="A", neutral=0,
             kickoff_utc="2026-10-03T23:00:00Z", created_utc="2026-10-03T20:00:00Z", data_cutoff_utc="2026-10-03T20:00:00Z",
             origin="live", is_blind=1, model_version="v1.0", p_home=0.6, snapshot_sha256="x")
    d.update(kw); return d


def test_predictions_are_immutable(tmp_path):
    s = Store(tmp_path / "t.db")
    s.insert_prediction(_pred())
    with pytest.raises(sqlite3.DatabaseError, match="immutable"):
        s.con.execute("UPDATE predictions SET p_home=0.9")
    with pytest.raises(sqlite3.DatabaseError, match="immutable"):
        s.con.execute("DELETE FROM predictions")


def test_cutoff_after_kickoff_rejected(tmp_path):
    s = Store(tmp_path / "t.db")
    with pytest.raises(sqlite3.IntegrityError):
        s.insert_prediction(_pred(pred_id="p2", data_cutoff_utc="2026-10-04T01:00:00Z"))


def test_scores_never_enter_games_table(tmp_path):
    s = Store(tmp_path / "t.db")
    with pytest.raises(AssertionError):
        s.upsert_games([{"sport": "t", "game_id": "g", "kickoff_utc": "2026-01-01T00:00:00Z", "home_id": "h",
                         "away_id": "a", "home_score": 1}])


def test_temporal_guard():
    g = TemporalGuard(4.5)
    df = pd.DataFrame({"kickoff_utc": pd.to_datetime(["2026-10-03T16:00Z", "2026-10-03T23:00Z"], utc=True)})
    cut = pd.Timestamp("2026-10-03T22:00Z")
    assert len(g.visible(df, cut)) == 1          # 16:00 + 4.5h = 20:30 <= 22:00; the 23:00 game is invisible
    with pytest.raises(LeakageError):
        g.check(df, cut, "test")


def test_snapshot_hash_stable():
    a = {"features": {"x": 1.0000000001, "y": None}, "v": "1"}
    b = {"v": "1", "features": {"y": None, "x": 1.0}}
    assert snapshot_hash(a) == snapshot_hash(b)


def test_metrics():
    p = np.array([0.9, 0.2, 0.6, 0.4]); y = np.array([1, 0, 0, 1])
    s = summarize(p, y)
    assert s["accuracy"] == 0.5
    assert abs(s["brier"] - np.mean((p - y) ** 2)) < 1e-12
    assert abs(s["log_loss"] - logloss_vec(p, y).mean()) < 1e-12
    q = bh_adjust([0.01, 0.04, 0.03, 0.5])
    assert np.all(q >= np.array([0.01, 0.04, 0.03, 0.5]) - 1e-12) and q.max() <= 1
    rng = np.random.default_rng(0); pp = rng.uniform(0.05, 0.95, 20000); yy = (rng.uniform(size=20000) < pp).astype(float)
    a, b, sa, sb = calibration_slope(pp, yy)
    assert abs(b - 1) < 4 * sb and abs(a) < 4 * sa   # a perfectly calibrated forecaster has slope 1, intercept 0


def test_query_parsing(tmp_path, monkeypatch):
    E = Engine.__new__(Engine)
    assert Engine.parse_query(E, "Predict Miami vs Clemson") == ("Miami", "Clemson", False)
    assert Engine.parse_query(E, "Texas Tech @ Colorado?") == ("Texas Tech", "Colorado", True)
    assert Engine.parse_query(E, "predict Ohio State at Michigan") == ("Ohio State", "Michigan", True)


def test_confidence_bounds():
    assert confidence_score(0.5, 0.0, 0, 0, 0) == 1.0
    assert confidence_score(0.99, 0.0, 0, 0, 0) > 9
    assert confidence_score(0.7, 0.2, 1, 1, 1) == 1.0
