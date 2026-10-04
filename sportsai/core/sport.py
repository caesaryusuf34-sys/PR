"""Sport adapter interface. The core (database, engine, learner, registry, metrics) is sport-agnostic;
everything sport-specific - data sources, parsing, feature construction, explanations - lives
behind this interface. Add a sport by implementing SportAdapter and registering it in ADAPTERS."""
from __future__ import annotations
from abc import ABC, abstractmethod
import pandas as pd


class FeatureBuilder(ABC):
    feature_version: str
    model_features: list[str]          # default model inputs
    candidate_features: list[str]      # superset the learner may choose from

    @abstractmethod
    def build(self, targets: pd.DataFrame) -> pd.DataFrame:
        """targets: one row per game with game_id, season, kickoff_utc, home_id, away_id, neutral,
        conf_game and cutoff_utc. Must use only information available at each row's cutoff."""

    @abstractmethod
    def training_cutoff(self, kickoff: pd.Series) -> pd.Series:
        """Information cutoff used when reconstructing features for historical games."""


class SportAdapter(ABC):
    sport: str
    game_duration_hours: float

    @abstractmethod
    def sync(self, store, *, full: bool = False, seasons: list[int] | None = None, stats: bool = True) -> dict: ...

    @abstractmethod
    def refresh_results(self, store, game_ids: list[str]) -> int: ...

    @abstractmethod
    def resolve_team(self, store, text: str) -> tuple[str, str]: ...

    @abstractmethod
    def find_game(self, store, team_a: str, team_b: str, after: pd.Timestamp, days: int = 21): ...

    @abstractmethod
    def feature_builder(self, store) -> FeatureBuilder: ...

    @abstractmethod
    def default_config(self) -> dict: ...

    @abstractmethod
    def pregame_context(self, game_id: str) -> dict: ...

    @abstractmethod
    def explain(self, row: pd.Series, pred: pd.Series, model) -> dict: ...

    def subgroups(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        return {}


def get_adapter(sport: str, settings) -> SportAdapter:
    sport = sport.lower()
    if sport in ("ncaaf", "cfb", "college-football"):
        from ..sports.ncaaf.adapter import NCAAFAdapter
        return NCAAFAdapter(settings)
    if sport in ("nfl",):
        from ..sports.nfl.adapter import NFLAdapter
        return NFLAdapter(settings)
    from ..sports.baseball.leagues import LEAGUES
    if sport in LEAGUES:
        return LEAGUES[sport](settings)
    raise KeyError(f"unknown sport {sport!r}; available: ncaaf, nfl, " + ", ".join(LEAGUES))
