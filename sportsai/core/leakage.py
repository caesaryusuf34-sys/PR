"""Pre-game / post-game separation.

A post-game fact about game X (its score, box score, play-by-play) becomes *available* at
    available_at(X) = kickoff(X) + game_duration
and may be used to predict game Y only if available_at(X) <= cutoff(Y) <= kickoff(Y).

Every feature builder receives an explicit ``cutoff`` and must obtain its inputs through
``TemporalGuard.visible`` (which filters) and finish with ``TemporalGuard.check`` (which asserts).
"""
from __future__ import annotations
import hashlib, json
import numpy as np
import pandas as pd


class LeakageError(RuntimeError):
    pass


class TemporalGuard:
    def __init__(self, game_duration_hours: float):
        self.duration = pd.Timedelta(hours=game_duration_hours)

    def available_at(self, kickoff: pd.Series) -> pd.Series:
        return kickoff + self.duration

    def visible(self, frame: pd.DataFrame, cutoff: pd.Timestamp, col: str = "kickoff_utc") -> pd.DataFrame:
        """Rows whose post-game information is known at `cutoff`."""
        return frame[self.available_at(frame[col]) <= cutoff]

    def check(self, used: pd.DataFrame, cutoff: pd.Timestamp, what: str, col: str = "kickoff_utc"):
        if len(used) and (self.available_at(used[col]) > cutoff).any():
            bad = used[self.available_at(used[col]) > cutoff]
            raise LeakageError(f"{what}: {len(bad)} rows not yet available at cutoff {cutoff} "
                               f"(latest kickoff {bad[col].max()})")


def snapshot_hash(payload: dict) -> str:
    def norm(v):
        if isinstance(v, (float, np.floating)):
            return None if not np.isfinite(v) else round(float(v), 8)
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, dict):
            return {k: norm(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [norm(x) for x in v]
        return v
    return hashlib.sha256(json.dumps(norm(payload), sort_keys=True, default=str).encode()).hexdigest()
