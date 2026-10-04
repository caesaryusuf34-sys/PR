"""Leakage audit for logged predictions.

For every stored prediction the features are rebuilt *now* - when the database also contains the
game's own result and everything that happened afterwards - using the stored data cutoff. If the
pipeline is leak-free the rebuilt features equal the snapshot that was frozen before kickoff.
Also re-verifies the snapshot hash and the blind-timestamp invariants."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd

from .leakage import snapshot_hash


def audit(E, limit: int = 50, tol: float = 1e-6) -> dict:
    P = E.store.predictions(E.sport)
    fv = E.adapter.feature_builder_version
    P = P[(P.feature_version == fv) & P.game_id.notna()].tail(limit)
    out = {"checked": 0, "feature_mismatch": [], "hash_mismatch": [], "timestamp_violations": []}
    if P.empty:
        return out
    fb = E.adapter.feature_builder(E.store)
    g = E.store.games(E.sport).set_index("game_id")
    t = P[["game_id", "data_cutoff_utc"]].copy()
    t = t.join(g[["season", "kickoff_utc", "home_id", "away_id", "neutral", "conf_game"]], on="game_id")
    t["cutoff_utc"] = pd.to_datetime(t.data_cutoff_utc, utc=True)
    F = fb.build(t.drop(columns="data_cutoff_utc").drop_duplicates(["game_id", "cutoff_utc"]))
    F = F.set_index(["game_id", "cutoff_utc"])
    for r in P.itertuples():
        out["checked"] += 1
        snap = json.loads(r.features_json)
        key = (r.game_id, pd.Timestamp(r.data_cutoff_utc))
        if key not in F.index:
            out["feature_mismatch"].append({"pred_id": r.pred_id, "error": "not rebuildable"}); continue
        row = F.loc[key]
        diffs = {f: (v, float(row[f])) for f, v in snap.items() if f in row and v is not None and np.isfinite(row[f])
                 and abs(float(row[f]) - v) > tol * max(1, abs(v))}
        if diffs:
            out["feature_mismatch"].append({"pred_id": r.pred_id, "game": f"{r.away_name} @ {r.home_name}", "diffs": dict(list(diffs.items())[:5])})
        h = snapshot_hash({"features": snap, "cutoff_utc": r.data_cutoff_utc, "model_version": r.model_version,
                           "feature_version": r.feature_version, "game_id": r.game_id, "home_id": r.home_id, "away_id": r.away_id})
        if h != r.snapshot_sha256:
            out["hash_mismatch"].append(r.pred_id)
        if pd.Timestamp(r.data_cutoff_utc) > pd.Timestamp(r.kickoff_utc) or (r.is_blind and pd.Timestamp(r.created_utc) >= pd.Timestamp(r.kickoff_utc)):
            out["timestamp_violations"].append(r.pred_id)
    out["leak_free"] = not (out["feature_mismatch"] or out["hash_mismatch"] or out["timestamp_violations"])
    return out
