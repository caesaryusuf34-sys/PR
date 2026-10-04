"""sportsai — a persistent, self-improving sports prediction system.

Pipeline:  query -> pre-game data -> point-in-time features -> champion model -> logged prediction
           -> (game ends) -> result collection -> evaluation -> error analysis -> candidate models
           -> walk-forward validation -> statistically-gated promotion -> new model version.
"""
__version__ = "1.0.0"
