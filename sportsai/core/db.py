"""Persistent historical database (SQLite).

Information is split by *when it becomes knowable*:

  PRE-GAME  : games (schedule, venue, site), teams, pregame_context (forecast, records captured
              before kickoff), predictions (immutable, written before kickoff for live picks)
  POST-GAME : results, team_game_stats (box score / play-by-play aggregates) - usable as inputs
              only for games that kick off after this game is over (enforced by core.leakage)
  DERIVED   : evaluations (prediction vs result), model_versions, learning_runs

Predictions are append-only: SQLite triggers abort any UPDATE or DELETE on that table.
"""
from __future__ import annotations
import json, sqlite3, uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable
import pandas as pd

from .timeutil import iso, now

SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS teams (
  sport TEXT NOT NULL, team_id TEXT NOT NULL, location TEXT, name TEXT, abbr TEXT,
  display_name TEXT, short_name TEXT, updated_utc TEXT,
  PRIMARY KEY (sport, team_id));

CREATE TABLE IF NOT EXISTS games (               -- PRE-GAME: schedule facts only, never scores
  sport TEXT NOT NULL, game_id TEXT NOT NULL,
  season INTEGER, seasontype INTEGER, week INTEGER,
  kickoff_utc TEXT NOT NULL,
  home_id TEXT NOT NULL, away_id TEXT NOT NULL, home_name TEXT, away_name TEXT,
  neutral INTEGER DEFAULT 0, conf_game INTEGER DEFAULT 0,
  venue TEXT, venue_city TEXT, venue_state TEXT, indoor INTEGER,
  feed_group INTEGER, status TEXT,
  first_seen_utc TEXT, updated_utc TEXT,
  PRIMARY KEY (sport, game_id));
CREATE INDEX IF NOT EXISTS ix_games_kick ON games(sport, kickoff_utc);

CREATE TABLE IF NOT EXISTS results (             -- POST-GAME
  sport TEXT NOT NULL, game_id TEXT NOT NULL,
  home_score REAL NOT NULL, away_score REAL NOT NULL, status TEXT,
  collected_utc TEXT NOT NULL, source TEXT,
  PRIMARY KEY (sport, game_id));

CREATE TABLE IF NOT EXISTS team_game_stats (     -- POST-GAME
  sport TEXT NOT NULL, game_id TEXT NOT NULL, team_id TEXT NOT NULL,
  stats_json TEXT NOT NULL, stats_version TEXT, collected_utc TEXT NOT NULL,
  PRIMARY KEY (sport, game_id, team_id));

CREATE TABLE IF NOT EXISTS pregame_context (     -- PRE-GAME, timestamped capture (forecast, records, ref. line)
  sport TEXT NOT NULL, game_id TEXT NOT NULL, captured_utc TEXT NOT NULL, context_json TEXT,
  PRIMARY KEY (sport, game_id, captured_utc));

CREATE TABLE IF NOT EXISTS predictions (         -- PRE-GAME, append-only
  pred_id TEXT PRIMARY KEY, sport TEXT NOT NULL, game_id TEXT, query TEXT,
  home_id TEXT, away_id TEXT, home_name TEXT, away_name TEXT, neutral INTEGER,
  kickoff_utc TEXT, created_utc TEXT NOT NULL, data_cutoff_utc TEXT NOT NULL,
  origin TEXT NOT NULL CHECK (origin IN ('live','backtest','hypothetical','legacy_import')),
  is_blind INTEGER NOT NULL,
  model_version TEXT NOT NULL, feature_version TEXT,
  p_home REAL NOT NULL, pred_margin REAL, pred_total REAL, proj_home REAL, proj_away REAL,
  confidence REAL, upset_prob REAL,
  member_preds_json TEXT, features_json TEXT, feature_names_json TEXT,
  explanation_json TEXT, context_json TEXT,
  snapshot_sha256 TEXT NOT NULL,
  CHECK (julianday(data_cutoff_utc) <= julianday(COALESCE(kickoff_utc, data_cutoff_utc)) + 1e-9));
CREATE TRIGGER IF NOT EXISTS predictions_no_update BEFORE UPDATE ON predictions
  BEGIN SELECT RAISE(ABORT, 'predictions are immutable'); END;
CREATE TRIGGER IF NOT EXISTS predictions_no_delete BEFORE DELETE ON predictions
  BEGIN SELECT RAISE(ABORT, 'predictions are immutable'); END;

CREATE TABLE IF NOT EXISTS prediction_annotations (   -- append-only notes on immutable predictions
  id INTEGER PRIMARY KEY AUTOINCREMENT, pred_id TEXT NOT NULL, at_utc TEXT NOT NULL,
  kind TEXT NOT NULL CHECK (kind IN ('invalid','note')), reason TEXT NOT NULL);
CREATE TRIGGER IF NOT EXISTS annotations_no_update BEFORE UPDATE ON prediction_annotations
  BEGIN SELECT RAISE(ABORT, 'annotations are append-only'); END;
CREATE TRIGGER IF NOT EXISTS annotations_no_delete BEFORE DELETE ON prediction_annotations
  BEGIN SELECT RAISE(ABORT, 'annotations are append-only'); END;

CREATE TABLE IF NOT EXISTS evaluations (         -- DERIVED (recomputable)
  pred_id TEXT PRIMARY KEY, sport TEXT, game_id TEXT, evaluated_utc TEXT,
  home_win INTEGER, actual_margin REAL, actual_total REAL,
  correct INTEGER, brier REAL, logloss REAL,
  margin_error REAL, abs_margin_error REAL, total_error REAL, high_conf_miss INTEGER);

CREATE TABLE IF NOT EXISTS model_versions (
  sport TEXT NOT NULL, version TEXT NOT NULL, created_utc TEXT, parent TEXT,
  status TEXT CHECK (status IN ('champion','archived','rejected','legacy')),
  kind TEXT, config_json TEXT, feature_names_json TEXT, feature_version TEXT,
  train_cutoff_utc TEXT, n_train INTEGER, validation_json TEXT, reason TEXT,
  artifact_path TEXT, code_rev TEXT,
  PRIMARY KEY (sport, version));

CREATE TABLE IF NOT EXISTS learning_runs (
  run_id TEXT PRIMARY KEY, sport TEXT, started_utc TEXT, finished_utc TEXT, trigger TEXT,
  n_new_games INTEGER, champion_before TEXT, champion_after TEXT, decision TEXT,
  summary_json TEXT, report_path TEXT);

CREATE TABLE IF NOT EXISTS sync_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT, sport TEXT, at_utc TEXT, kind TEXT, detail TEXT);

CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""


def _clean(v):
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(v, "item"):
        return v.item()
    return v


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.con = sqlite3.connect(str(path), timeout=60)
        self.con.execute("PRAGMA foreign_keys=ON")
        self.con.executescript(SCHEMA)

    # ------------------------------------------------------------------ generic
    @contextmanager
    def tx(self):
        try:
            yield self.con
            self.con.commit()
        except Exception:
            self.con.rollback()
            raise

    def df(self, sql: str, params: Iterable = ()) -> pd.DataFrame:
        return pd.read_sql_query(sql, self.con, params=list(params))

    def _upsert(self, table: str, rows: list[dict], keys: list[str], keep_first: tuple[str, ...] = ()):
        if not rows:
            return 0
        cols = list(rows[0].keys())
        upd = [c for c in cols if c not in keys and c not in keep_first]
        sql = (f"INSERT INTO {table} ({','.join(cols)}) VALUES ({','.join('?' * len(cols))}) "
               f"ON CONFLICT ({','.join(keys)}) DO " + (f"UPDATE SET {','.join(f'{c}=excluded.{c}' for c in upd)}" if upd else "NOTHING"))
        with self.tx() as c:
            c.executemany(sql, [[_clean(r.get(k)) for k in cols] for r in rows])
        return len(rows)

    def log(self, sport: str, kind: str, detail: str):
        with self.tx() as c:
            c.execute("INSERT INTO sync_log (sport, at_utc, kind, detail) VALUES (?,?,?,?)", (sport, iso(now()), kind, detail))

    def get_meta(self, key: str, default=None):
        r = self.con.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(r[0]) if r else default

    def set_meta(self, key: str, value):
        with self.tx() as c:
            c.execute("INSERT INTO meta (key, value) VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                      (key, json.dumps(value, default=str)))

    # ------------------------------------------------------------------ pre-game
    def upsert_teams(self, rows):
        return self._upsert("teams", rows, ["sport", "team_id"])

    def upsert_games(self, rows, only_new: bool = False):
        for r in rows:
            assert "home_score" not in r and "away_score" not in r, "scores belong in `results` (post-game)"
        keep = tuple(rows[0].keys()) if (rows and only_new) else ("first_seen_utc",)
        return self._upsert("games", rows, ["sport", "game_id"], keep_first=keep)

    def add_pregame_context(self, sport, game_id, context: dict):
        with self.tx() as c:
            c.execute("INSERT OR IGNORE INTO pregame_context VALUES (?,?,?,?)",
                      (sport, game_id, iso(now()), json.dumps(context, default=str)))

    # ------------------------------------------------------------------ post-game
    def upsert_results(self, rows):
        return self._upsert("results", rows, ["sport", "game_id"])

    def upsert_team_game_stats(self, rows):
        return self._upsert("team_game_stats", rows, ["sport", "game_id", "team_id"])

    # ------------------------------------------------------------------ predictions
    def insert_prediction(self, row: dict) -> str:
        row = dict(row)
        row.setdefault("pred_id", uuid.uuid4().hex)
        cols = list(row.keys())
        with self.tx() as c:
            c.execute(f"INSERT INTO predictions ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                      [_clean(row[k]) for k in cols])
        return row["pred_id"]

    def annotate(self, pred_ids, kind: str, reason: str):
        """Predictions are never edited; a bad one is marked invalid (with a reason) and excluded downstream."""
        with self.tx() as c:
            c.executemany("INSERT INTO prediction_annotations (pred_id, at_utc, kind, reason) VALUES (?,?,?,?)",
                          [(p, iso(now()), kind, reason) for p in pred_ids])
            if kind == "invalid":
                c.executemany("DELETE FROM evaluations WHERE pred_id=?", [(p,) for p in pred_ids])

    def replace_evaluations(self, rows):
        return self._upsert("evaluations", rows, ["pred_id"])

    # ------------------------------------------------------------------ views
    def games(self, sport: str) -> pd.DataFrame:
        g = self.df("SELECT * FROM games WHERE sport=?", (sport,))
        g["kickoff_utc"] = pd.to_datetime(g.kickoff_utc, utc=True)
        return g

    def completed_games(self, sport: str) -> pd.DataFrame:
        g = self.df("""SELECT g.*, r.home_score, r.away_score, r.collected_utc AS result_collected_utc
                       FROM games g JOIN results r USING (sport, game_id) WHERE g.sport=?""", (sport,))
        g["kickoff_utc"] = pd.to_datetime(g.kickoff_utc, utc=True)
        return g.sort_values("kickoff_utc").reset_index(drop=True)

    def team_game_stats(self, sport: str) -> pd.DataFrame:
        s = self.df("""SELECT t.game_id, t.team_id, t.stats_json, g.kickoff_utc, g.season
                       FROM team_game_stats t JOIN games g USING (sport, game_id) WHERE t.sport=?""", (sport,))
        if s.empty:
            return s
        body = pd.DataFrame([json.loads(x) for x in s.stats_json])
        out = pd.concat([s.drop(columns="stats_json").reset_index(drop=True), body], axis=1)
        out["kickoff_utc"] = pd.to_datetime(out.kickoff_utc, utc=True)
        return out

    def predictions(self, sport: str) -> pd.DataFrame:
        return self.df("SELECT * FROM predictions WHERE sport=? ORDER BY created_utc", (sport,))

    def evaluated(self, sport: str) -> pd.DataFrame:
        return self.df("""SELECT p.*, e.home_win, e.actual_margin, e.actual_total, e.correct, e.brier, e.logloss,
                                 e.margin_error, e.abs_margin_error, e.total_error, e.high_conf_miss, e.evaluated_utc
                          FROM predictions p JOIN evaluations e USING (pred_id) WHERE p.sport=?""", (sport,))
