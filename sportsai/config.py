"""Central configuration: storage locations and the learning policy.

Everything under ``store/`` is the system's persistent memory (database + model registry +
learning reports) and is meant to be version-controlled. ``cache/`` is rebuildable.
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(os.environ.get("SPORTSAI_ROOT", Path(__file__).resolve().parent.parent))


@dataclass
class LearningPolicy:
    # when is a learning cycle due?
    min_new_games: int = 100            # completed games since the champion's training cutoff
    # walk-forward validation
    eval_window_games: int = 3000       # most recent completed games used for out-of-sample evaluation
    n_folds: int = 6                    # contiguous time blocks inside the window
    confirm_folds: int = 2              # newest folds reserved for the champion-vs-challenger test
    # promotion rules for a structural change (new features / members / calibration / hyper-params)
    min_logloss_gain: float = 0.002     # required mean log-loss improvement on the confirmation window
    alpha: float = 0.05                 # one-sided significance level (block bootstrap by week)
    max_ece_worsening: float = 0.01
    min_confirm_games: int = 300
    bootstrap_reps: int = 2000
    max_candidates: int = 8
    # diagnostics
    high_conf: float = 0.75             # a "high-confidence" pick
    fdr: float = 0.10                   # Benjamini-Hochberg false-discovery rate for bias tests
    min_subgroup_n: int = 60


@dataclass
class Settings:
    root: Path = ROOT
    store_dir: Path = field(default_factory=lambda: ROOT / "store")
    cache_dir: Path = field(default_factory=lambda: ROOT / "cache")
    policy: LearningPolicy = field(default_factory=LearningPolicy)
    user_agent: str = "sportsai-research/1.0 (personal, non-commercial)"

    @property
    def db_path(self) -> Path:
        return self.store_dir / "sportsai.db"

    @property
    def models_dir(self) -> Path:
        return self.store_dir / "models"

    @property
    def reports_dir(self) -> Path:
        return self.store_dir / "reports"

    def ensure(self) -> "Settings":
        for p in (self.store_dir, self.cache_dir, self.models_dir, self.reports_dir):
            p.mkdir(parents=True, exist_ok=True)
        return self
