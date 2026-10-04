"""Cached point-in-time features for completed games (the learner's training/evaluation data).

Rows are rebuilt automatically when the amount of information visible at their cutoff changes
(e.g. a box score backfilled later), so the cache can never drift from a fresh reconstruction."""
from __future__ import annotations
import numpy as np
import pandas as pd

from .timeutil import now


class FeatureStore:
    def __init__(self, settings, store, adapter):
        self.S, self.store, self.adapter = settings, store, adapter

    def _path(self, version):
        p = self.S.cache_dir / self.adapter.sport
        p.mkdir(parents=True, exist_ok=True)
        return p / f"features_{version}.pkl"

    def training_frame(self, as_of=None, min_season: int = 2022, builder=None, verbose=True) -> pd.DataFrame:
        as_of = as_of or now()
        fb = builder or self.adapter.feature_builder(self.store)
        done = self.store.completed_games(self.adapter.sport)
        done = done[(done.season >= min_season) & (done.home_score != done.away_score)]
        done = done[done.kickoff_utc + pd.Timedelta(hours=self.adapter.game_duration_hours) <= as_of]
        path = self._path(fb.feature_version)
        cache = pd.read_pickle(path) if path.exists() else pd.DataFrame()
        if len(cache):
            cache = cache[cache.game_id.isin(done.game_id)]
            ng, nst = fb.visible_counts(cache.cutoff_utc)
            stale = (cache.n_games_visible.values != ng) | (cache.n_stats_visible.values != nst)
            if hasattr(fb, "row_signature"):     # pre-game inputs that are not time-indexed (e.g. announced starters)
                stale |= cache.row_signature.astype(str).values != fb.row_signature(cache)
            cache = cache[~stale]
        todo = done[~done.game_id.isin(cache.game_id)] if len(cache) else done
        if len(todo):
            if verbose:
                print(f"[features] building {len(todo)} game(s) ({fb.feature_version})", flush=True)
            t = todo[["game_id", "season", "kickoff_utc", "home_id", "away_id", "neutral", "conf_game"]].copy()
            t["cutoff_utc"] = fb.training_cutoff(t.kickoff_utc)
            new = fb.build(t)
            cache = pd.concat([cache, new], ignore_index=True) if len(cache) else new
            cache.to_pickle(path)
        tgt = done[["game_id", "home_score", "away_score", "seasontype", "home_name", "away_name"]]
        out = cache.merge(tgt, on="game_id", how="inner")
        out["margin"] = out.home_score - out.away_score
        out["total"] = out.home_score + out.away_score
        out["home_win"] = (out.margin > 0).astype(int)
        return out.sort_values(["kickoff_utc", "game_id"]).reset_index(drop=True)
