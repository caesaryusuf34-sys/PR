"""Integration leakage tests against the real database (skipped when the store has not been built)."""
import numpy as np
import pandas as pd
import pytest

from sportsai.config import Settings
from sportsai.core.db import Store
from sportsai.core.sport import get_adapter
from sportsai.sports.ncaaf.features import NCAAFFeatureBuilder

S = Settings()
pytestmark = pytest.mark.skipif(not S.db_path.exists(), reason="store not built")


@pytest.fixture(scope="module")
def env():
    st = Store(S.db_path); ad = get_adapter("ncaaf", S)
    return st, ad, ad.feature_builder(st)


def test_truncation_invariance(env):
    """Features at a cutoff must not change when every post-cutoff result/box score is deleted."""
    st, ad, fb = env
    done = st.completed_games("ncaaf")
    rng = np.random.default_rng(1)
    sample = done[done.season >= 2023].sample(6, random_state=3)
    for _, g in sample.groupby("season"):
        t = g.copy(); t["cutoff_utc"] = fb.training_cutoff(t.kickoff_utc)
        cut = t.cutoff_utc.min(); t["cutoff_utc"] = cut
        dur = pd.Timedelta(hours=ad.game_duration_hours)
        stats = st.team_game_stats("ncaaf")
        fb2 = NCAAFFeatureBuilder(st.games("ncaaf"), done[done.kickoff_utc + dur <= cut], stats[stats.kickoff_utc + dur <= cut], ad.guard)
        A = fb.build(t).set_index("game_id"); B = fb2.build(t).set_index("game_id")
        num = A.select_dtypes("number").columns
        assert np.nanmax(np.abs((A[num] - B[num]).values)) < 1e-9


def test_own_result_invisible(env):
    """A game's own result can never be used for its own features (cutoff <= kickoff < kickoff + duration)."""
    st, ad, fb = env
    done = st.completed_games("ncaaf")
    g = done[done.season == 2025].head(30).copy()
    g["cutoff_utc"] = g.kickoff_utc
    F = fb.build(g)
    vis = F.n_games_visible.values
    for (_, r), n in zip(g.iterrows(), vis):
        assert (done.kickoff_utc + pd.Timedelta(hours=ad.game_duration_hours) <= r.kickoff_utc).sum() == n


def test_cutoff_after_kickoff_raises(env):
    st, ad, fb = env
    g = st.completed_games("ncaaf").tail(1).copy()
    g["cutoff_utc"] = g.kickoff_utc + pd.Timedelta(minutes=1)
    with pytest.raises(ValueError):
        fb.build(g)


def test_batch_prediction_alignment():
    """Batch predictions must equal one-at-a-time predictions (guards against row-order mix-ups)."""
    from sportsai.core.engine import Engine
    E = Engine("ncaaf")
    if E.registry.champion_row() is None:
        pytest.skip("no champion")
    done = E.store.completed_games("ncaaf")
    g = done[done.season == 2025].groupby("kickoff_utc").filter(lambda x: len(x) >= 4).head(4)
    cut = g.kickoff_utc.min()
    batch = E.predict_games(g, origin="backtest", save=False, cutoff=cut, context=False)
    for (_, row), b in zip(g.iterrows(), batch):
        single = E.predict_games(row.to_frame().T, origin="backtest", save=False, cutoff=cut, context=False)[0]
        assert single["home_id"] == b["home_id"] == row.home_id
        assert abs(single["p_home"] - b["p_home"]) < 1e-9


def test_nfl_truncation_invariance():
    """Same leakage guarantee for the NFL adapter (shared football feature engine, NFL settings)."""
    st = Store(S.db_path); ad = get_adapter("nfl", S)
    done = st.completed_games("nfl")
    if done.empty:
        pytest.skip("nfl not synced")
    fb = ad.feature_builder(st)
    g = done[done.season == 2025].sample(6, random_state=5)
    t = g.copy(); t["cutoff_utc"] = fb.training_cutoff(t.kickoff_utc).min()
    cut = t.cutoff_utc.iloc[0]; dur = pd.Timedelta(hours=ad.game_duration_hours)
    stats = st.team_game_stats("nfl")
    fb2 = type(fb)(st.games("nfl"), done[done.kickoff_utc + dur <= cut], stats[stats.kickoff_utc + dur <= cut], ad.guard)
    A = fb.build(t).set_index("game_id"); B = fb2.build(t).set_index("game_id")
    num = A.select_dtypes("number").columns
    assert np.nanmax(np.abs((A[num] - B[num]).values)) < 1e-9
