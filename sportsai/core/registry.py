"""Model registry: versioned artifacts on disk + metadata rows in the database.

Version scheme  v<major>.<minor>
  major bump = structural change accepted by the learner (features, members, calibration, ...)
  minor bump = same validated configuration re-estimated on more data
"""
from __future__ import annotations
import json, subprocess
from pathlib import Path
import joblib

from .db import Store
from .timeutil import iso, now


def _code_rev(root: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(root), "rev-parse", "--short", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


class ModelRegistry:
    def __init__(self, store: Store, models_dir: Path, sport: str, root: Path):
        self.store, self.dir, self.sport, self.root = store, models_dir / sport, sport, root
        self.dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- queries
    def versions(self):
        return self.store.df("SELECT * FROM model_versions WHERE sport=? ORDER BY created_utc", (self.sport,))

    def champion_row(self):
        v = self.store.df("SELECT * FROM model_versions WHERE sport=? AND status='champion'", (self.sport,))
        return None if v.empty else v.iloc[0]

    def load(self, version: str):
        row = self.store.df("SELECT artifact_path FROM model_versions WHERE sport=? AND version=?", (self.sport, version))
        if row.empty or not row.artifact_path.iloc[0]:
            raise KeyError(f"no artifact for {self.sport} {version}")
        return joblib.load(self.root / row.artifact_path.iloc[0])

    def champion(self):
        r = self.champion_row()
        if r is None:
            raise RuntimeError(f"no champion model for {self.sport}; run `python -m sportsai bootstrap`")
        return r.version, self.load(r.version), r

    def next_version(self, kind: str) -> str:
        vs = [v for v in self.versions().version if v.startswith("v") and "." in v and v[1:].replace(".", "").isdigit()]
        if not vs:
            return "v1.0"
        major, minor = max(tuple(int(x) for x in v[1:].split(".")) for v in vs)
        return f"v{major + 1}.0" if kind in ("structural",) else f"v{major}.{minor + 1}"

    # ---------------------------------------------------------------- writes
    def register(self, model, *, kind: str, reason: str, train_cutoff, n_train: int, validation: dict,
                 parent: str | None, promote: bool = True, version: str | None = None) -> str:
        version = version or self.next_version(kind)
        vdir = self.dir / version
        vdir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, vdir / "model.joblib", compress=3)
        meta = {"sport": self.sport, "version": version, "created_utc": iso(now()), "parent": parent, "kind": kind,
                "config": model.config, "features": model.features, "feature_version": model.feature_version,
                "train_cutoff_utc": iso(train_cutoff), "n_train": int(n_train), "validation": validation,
                "reason": reason, "code_rev": _code_rev(self.root)}
        (vdir / "metadata.json").write_text(json.dumps(meta, indent=1, default=str))
        with self.store.tx() as c:
            if promote:
                c.execute("UPDATE model_versions SET status='archived' WHERE sport=? AND status='champion'", (self.sport,))
            c.execute("""INSERT INTO model_versions (sport, version, created_utc, parent, status, kind, config_json,
                         feature_names_json, feature_version, train_cutoff_utc, n_train, validation_json, reason,
                         artifact_path, code_rev) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                      (self.sport, version, meta["created_utc"], parent, "champion" if promote else "archived", kind,
                       json.dumps(model.config, default=str), json.dumps(model.features), model.feature_version,
                       meta["train_cutoff_utc"], int(n_train), json.dumps(validation, default=str), reason,
                       str((vdir / "model.joblib").relative_to(self.root)), meta["code_rev"]))
        return version

    def register_external(self, version: str, *, reason: str, status: str = "legacy", kind: str = "legacy"):
        with self.store.tx() as c:
            c.execute("""INSERT OR IGNORE INTO model_versions (sport, version, created_utc, status, kind, reason)
                         VALUES (?,?,?,?,?,?)""", (self.sport, version, iso(now()), status, kind, reason))

    def rollback(self, to_version: str, reason: str):
        with self.store.tx() as c:
            c.execute("UPDATE model_versions SET status='archived' WHERE sport=? AND status='champion'", (self.sport,))
            c.execute("UPDATE model_versions SET status='champion', reason=reason || ' | rollback: ' || ? "
                      "WHERE sport=? AND version=?", (reason, self.sport, to_version))
