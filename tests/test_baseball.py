"""Baseball adapters (MLB, KBO, NPB, CPBL): unit tests that need no network, plus leakage integration
tests against the real database (skipped for a league that has not been synced)."""
import numpy as np
import pandas as pd
import pytest

from sportsai.config import Settings
from sportsai.core.db import Store
from sportsai.core.leakage import TemporalGuard
from sportsai.core.sport import get_adapter
from sportsai.sports.baseball.features import BaseballFeatureBuilder, ridge_runs
from sportsai.sports.baseball.sources import NPBSource

S = Settings()


# ------------------------------------------------------------------------------------------- units
def test_ridge_runs_recovers_effects():
    rng = np.random.default_rng(0)
    n_t, n_p, n = 6, 12, 6000
    o, d, p = rng.normal(0, .5, n_t), rng.normal(0, .5, n_t), rng.normal(0, .8, n_p)
    off = rng.integers(0, n_t, n); dfn = (off + rng.integers(1, n_t, n)) % n_t
    spi = rng.integers(-1, n_p, n); loc = rng.choice([1.0, -1.0], n)
    y = 4.5 + 0.1 * loc + o[off] + d[dfn] + 0.6 * np.where(spi >= 0, p[np.maximum(spi, 0)], 0) + rng.normal(0, .3, n)
    mu, h, ro, rd, rp = ridge_runs(off, dfn, spi, loc, y, n_t, n_p, np.zeros(n_t), np.zeros(n_t), np.zeros(n_p), 1.0, 1.0, 0.6)
    assert abs(h - 0.1) < 0.03
    assert np.corrcoef(rp, p)[0, 1] > 0.95 and np.corrcoef(ro - ro.mean(), o)[0, 1] > 0.95


def test_ridge_runs_without_games_returns_priors():
    mu, h, ro, rd, rp = ridge_runs(np.array([], int), np.array([], int), np.array([], int), np.array([]), np.array([]),
                                   2, 2, np.array([.3, -.3]), np.array([.1, 0.]), np.array([.5, -.5]), 10., 10., .6)
    assert np.allclose(ro, [.3, -.3]) and np.allclose(rp, [.5, -.5])


def _builder(starters):
    games = pd.DataFrame({"game_id": ["g1"], "season": [2026], "kickoff_utc": pd.to_datetime(["2026-06-01T10:00Z"], utc=True),
                          "home_id": ["H"], "away_id": ["A"], "neutral": [0], "conf_game": [1]})
    done = games.iloc[:0].assign(home_score=[], away_score=[])
    st = pd.DataFrame(starters, columns=["game_id", "side", "team_id", "pitcher_id", "pitcher_name", "source", "captured_utc"])
    st["captured_utc"] = pd.to_datetime(st.captured_utc, utc=True)
    return BaseballFeatureBuilder(games, done, pd.DataFrame(), st, TemporalGuard(4.5))


def test_starter_resolution_is_point_in_time():
    fb = _builder([("g1", "home", "H", "p1", "Ace", "probable", "2026-05-31T09:00Z"),
                   ("g1", "home", "H", "p2", "Swap", "probable", "2026-06-01T08:00Z"),
                   ("g1", "home", "H", "p3", "Actual", "backfill", "2026-06-02T00:00Z")])
    c = lambda t: pd.Timestamp(t).value
    assert fb.resolve_starter("g1", "home", c("2026-06-01T07:00Z"))[0] == "p1"     # announcement known then
    assert fb.resolve_starter("g1", "home", c("2026-06-01T09:00Z"))[0] == "p2"     # later change of starter
    assert fb.resolve_starter("g1", "home", c("2026-05-30T00:00Z"))[0] == "p3"     # nothing captured yet -> backfill
    assert fb.resolve_starter("g1", "away", c("2026-06-01T09:00Z"))[0] is None


def test_tbd_capture_blocks_backfill():
    """A live prediction made while the starter was TBD must rebuild as TBD after the actual starter is known."""
    fb = _builder([("g1", "away", "A", None, None, "probable", "2026-06-01T06:00Z"),
                   ("g1", "away", "A", "p9", "Actual", "backfill", "2026-06-02T00:00Z")])
    assert fb.resolve_starter("g1", "away", pd.Timestamp("2026-06-01T06:30Z").value)[0] is None


def test_store_starters_dedupe(tmp_path):
    s = Store(tmp_path / "t.db")
    row = dict(game_id="g", side="home", team_id="H", pitcher_id="1", pitcher_name="A", source="probable")
    assert s.add_starters("mlb", [row], "2026-06-01T00:00:00Z") == 1
    assert s.add_starters("mlb", [row], "2026-06-01T01:00:00Z") == 0                  # unchanged announcement
    assert s.add_starters("mlb", [dict(row, pitcher_id="2")], "2026-06-01T02:00:00Z") == 1
    assert s.add_starters("mlb", [dict(row, source="backfill")], "2026-06-03T00:00:00Z") == 1
    assert s.add_starters("mlb", [dict(row, source="backfill", pitcher_id="3")], "2026-06-04T00:00:00Z") == 0
    assert len(s.starters("mlb")) == 3


NPB_BOX = """<div id="table_top_p" class="table_score"><table id="tablefix_t_p"><thead><tr><th>投手</th></tr></thead><tbody>
<tr><td></td><td class="player"><a href="/bis/players/111.html">下川</a></td><td>54</td><td>15</td>
<td><table class="table_inning"><tbody><tr><th>4</th><td></td></tr></tbody></table></td>
<td>3</td><td>1</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>2</td><td>2</td></tr>
<tr><td>●</td><td class="player"><a href="/bis/players/222.html">星</a></td><td>14</td><td>4</td>
<td><table class="table_inning"><tbody><tr><th>0</th><td>.2</td></tr></tbody></table></td>
<td>1</td><td>0</td><td>1</td><td>0</td><td>0</td><td>0</td><td>0</td><td>1</td><td>1</td></tr>
</tbody></table>
</div>"""


def test_npb_box_parser():
    ps = NPBSource._pitchers(NPB_BOX, "tablefix_t_p")
    assert [p["id"] for p in ps] == ["111", "222"]
    assert ps[0]["outs"] == 12 and ps[0]["bf"] == 15 and ps[0]["hr"] == 1 and ps[0]["er"] == 2
    assert ps[1]["outs"] == 2 and ps[1]["bb"] == 1 and ps[1]["r"] == 1


def test_ties_void_picks(tmp_path, monkeypatch):
    from sportsai.core.engine import Engine
    monkeypatch.setenv("SPORTSAI_ROOT", str(tmp_path))
    E = Engine.__new__(Engine)
    E.store = Store(tmp_path / "t.db"); E.sport = "npb"; E.adapter = get_adapter("npb", Settings(root=tmp_path, store_dir=tmp_path, cache_dir=tmp_path / "c"))
    E.S = Settings(root=tmp_path, store_dir=tmp_path, cache_dir=tmp_path / "c")
    for gid, hs, as_ in (("g1", 2, 2), ("g2", 3, 1)):
        E.store.insert_prediction(dict(pred_id=gid, sport="npb", game_id=gid, home_id="h", away_id="a", home_name="H", away_name="A",
                                       neutral=0, kickoff_utc="2026-06-01T09:00:00Z", created_utc="2026-06-01T08:00:00Z",
                                       data_cutoff_utc="2026-06-01T08:00:00Z", origin="live", is_blind=1, model_version="v1.0",
                                       p_home=0.6, pred_margin=0.5, pred_total=7.0, snapshot_sha256="x"))
        E.store.upsert_results([dict(sport="npb", game_id=gid, home_score=hs, away_score=as_, status="Final",
                                     collected_utc="2026-06-01T13:00:00Z", source="t")])
    assert E.evaluate() == 1                                      # the 2-2 tie is void, not a loss


# ------------------------------------------------------------------------------------------- integration
@pytest.mark.parametrize("league", ["mlb", "kbo", "npb", "cpbl"])
def test_truncation_invariance(league):
    """Features at a cutoff must not change when every result / box score after the cutoff is deleted."""
    if not (S.store_dir / f"{league}.db").exists():
        pytest.skip("store not built")
    st = Store(S.store_dir / f"{league}.db"); ad = get_adapter(league, S)
    done = st.completed_games(league)
    if len(done) < 500:
        pytest.skip(f"{league} not synced")
    fb = ad.feature_builder(st)
    season = int(done.season.max())
    g = done[done.season == season].iloc[len(done[done.season == season]) // 2:].head(8).copy()
    t = g.copy(); t["cutoff_utc"] = fb.training_cutoff(t.kickoff_utc).min()
    cut = t.cutoff_utc.iloc[0]; dur = pd.Timedelta(hours=ad.game_duration_hours)
    stats = st.team_game_stats(league)
    fb2 = type(fb)(st.games(league), done[done.kickoff_utc + dur <= cut], stats[stats.kickoff_utc + dur <= cut], st.starters(league), ad.guard)
    A = fb.build(t).set_index("game_id"); B = fb2.build(t).set_index("game_id")
    num = A.select_dtypes("number").columns
    assert np.nanmax(np.abs((A[num] - B[num]).values)) < 1e-9


@pytest.mark.parametrize("league", ["mlb", "kbo", "npb", "cpbl"])
def test_own_result_invisible(league):
    if not (S.store_dir / f"{league}.db").exists():
        pytest.skip("store not built")
    st = Store(S.store_dir / f"{league}.db"); ad = get_adapter(league, S)
    done = st.completed_games(league)
    if len(done) < 500:
        pytest.skip(f"{league} not synced")
    fb = ad.feature_builder(st)
    g = done.iloc[-40:-20].copy(); g["cutoff_utc"] = g.kickoff_utc
    F = fb.build(g).set_index("game_id").loc[g.game_id]
    for (_, r), n in zip(g.iterrows(), F.n_games_visible.values):
        assert (done.kickoff_utc + pd.Timedelta(hours=ad.game_duration_hours) <= r.kickoff_utc).sum() == n
