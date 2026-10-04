"""Import the 2026-10-03 slate predictions produced by the original one-off pipeline (src/), frozen in
git commit 82c7a54 at 2026-10-03T20:35:40Z, so the self-learning system's record starts with them."""
from __future__ import annotations
import json
import pandas as pd

from .leakage import snapshot_hash
from .timeutil import iso


def import_legacy(E, predictions_csv, features_csv, version="v0-legacy") -> int:
    P = pd.read_csv(predictions_csv, dtype={"event_id": str})
    F = pd.read_csv(features_csv, dtype={"event_id": str}).set_index("event_id")
    have = set(E.store.df("SELECT game_id FROM predictions WHERE sport=? AND model_version=?", (E.sport, version)).game_id)
    g = E.store.games(E.sport).set_index("game_id")
    E.registry.register_external(version, reason="one-off pipeline (src/) used for the 2026-10-03 slate; frozen in commit 82c7a54")
    n = 0
    for r in P.itertuples():
        if r.event_id in have or r.event_id not in g.index:
            continue
        gi = g.loc[r.event_id]
        feats = {k: (None if pd.isna(v) else float(v)) for k, v in F.loc[r.event_id].items()
                 if k not in ("away", "home") and isinstance(v, (int, float))}
        created = pd.Timestamp(r.prediction_freeze_utc).floor("s")
        snap = {"features": feats, "cutoff_utc": r.data_cutoff_utc, "model_version": version, "game_id": r.event_id}
        E.store.insert_prediction(dict(
            sport=E.sport, game_id=r.event_id, query="2026-10-03 full slate", home_id=gi.home_id, away_id=gi.away_id,
            home_name=gi.home_name, away_name=gi.away_name, neutral=int(gi.neutral), kickoff_utc=iso(gi.kickoff_utc),
            created_utc=iso(created), data_cutoff_utc=iso(pd.Timestamp(r.data_cutoff_utc)), origin="legacy_import",
            is_blind=int(created < gi.kickoff_utc), model_version=version, feature_version="legacy-v0",
            p_home=float(r.home_win_prob), pred_margin=float(r.m_final), pred_total=float(r.proj_total),
            proj_home=float(r.proj_home), proj_away=float(r.proj_away), confidence=float(r.confidence),
            upset_prob=float(r.upset_prob),
            member_preds_json=json.dumps({c: float(getattr(r, c)) for c in ("p_elo", "p_pts", "p_lr", "p_gbm", "p_reg", "m_elo", "m_pts", "m_reg", "m_gbm")}),
            features_json=json.dumps(feats), feature_names_json=json.dumps(list(feats)), explanation_json="{}",
            context_json=json.dumps({"source": "git commit 82c7a54"}), snapshot_sha256=snapshot_hash(snap)))
        n += 1
    return n
