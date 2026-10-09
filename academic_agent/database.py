"""SQLite storage: papers, downloads (with hashes), every attempted source, and research runs."""
from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from models import Paper
from paper_match import normalize_doi, normalize_title

SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doi TEXT, norm_title TEXT NOT NULL, title TEXT NOT NULL, authors_json TEXT, year INTEGER, venue TEXT,
    abstract TEXT, url TEXT, arxiv_id TEXT, sources_json TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_papers_doi ON papers(doi) WHERE doi IS NOT NULL AND doi <> '';
CREATE INDEX IF NOT EXISTS ix_papers_title ON papers(norm_title);
CREATE TABLE IF NOT EXISTS downloads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id INTEGER REFERENCES papers(id) ON DELETE CASCADE,
    file_path TEXT, sha256 TEXT, size_bytes INTEGER, version_type TEXT, source_url TEXT, landing_url TEXT,
    provider TEXT, status TEXT NOT NULL, verification_json TEXT, access_date TEXT NOT NULL, query TEXT
);
CREATE INDEX IF NOT EXISTS ix_downloads_hash ON downloads(sha256);
CREATE INDEX IF NOT EXISTS ix_downloads_paper ON downloads(paper_id);
CREATE TABLE IF NOT EXISTS attempts (
    id INTEGER PRIMARY KEY AUTOINCREMENT, run_id INTEGER, paper_id INTEGER, url TEXT, stage TEXT, status TEXT,
    detail TEXT, provider TEXT, ts TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT, query TEXT NOT NULL, query_type TEXT, outcome TEXT, access_state TEXT,
    paper_id INTEGER, summary TEXT, result_json TEXT, started_at TEXT NOT NULL, finished_at TEXT
);
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Database:
    def __init__(self, path: Path | str = ":memory:"):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys=ON")
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    @contextmanager
    def _tx(self):
        with self._lock:
            try:
                yield self._conn
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ---- papers -------------------------------------------------------------------------
    def upsert_paper(self, p: Paper) -> int:
        doi = normalize_doi(p.doi) or None
        norm = normalize_title(p.title)
        ts = now_iso()
        with self._tx() as c:
            row = None
            if doi:
                row = c.execute("SELECT id FROM papers WHERE doi=?", (doi,)).fetchone()
                if row is None:       # same title/year recorded earlier without a DOI -> same paper
                    row = c.execute("SELECT id FROM papers WHERE norm_title=? AND COALESCE(year,0)=COALESCE(?,0) "
                                    "AND (doi IS NULL OR doi='')", (norm, p.year)).fetchone()
            else:
                row = c.execute("SELECT id FROM papers WHERE norm_title=? AND COALESCE(year,0)=COALESCE(?,0)",
                                (norm, p.year)).fetchone()
            vals = (doi, norm, p.title, json.dumps(p.authors), p.year, p.venue, p.abstract, p.url, p.arxiv_id,
                    json.dumps(p.sources))
            if row:
                c.execute("UPDATE papers SET doi=COALESCE(?,doi), norm_title=?, title=?, "
                          "authors_json=CASE WHEN ?='[]' THEN authors_json ELSE ? END, year=COALESCE(?,year), "
                          "venue=COALESCE(NULLIF(?,''),venue), abstract=COALESCE(NULLIF(?,''),abstract), "
                          "url=COALESCE(NULLIF(?,''),url), arxiv_id=COALESCE(NULLIF(?,''),arxiv_id), sources_json=?, "
                          "updated_at=? WHERE id=?",
                          (doi, norm, p.title, vals[3], vals[3], p.year, p.venue, p.abstract, p.url, p.arxiv_id,
                           vals[9], ts, row["id"]))
                return int(row["id"])
            cur = c.execute("INSERT INTO papers(doi,norm_title,title,authors_json,year,venue,abstract,url,arxiv_id,"
                            "sources_json,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", (*vals, ts, ts))
            return int(cur.lastrowid)

    def get_paper(self, paper_id: int) -> Optional[dict]:
        with self._lock:
            r = self._conn.execute("SELECT * FROM papers WHERE id=?", (paper_id,)).fetchone()
        return dict(r) if r else None

    # ---- downloads ----------------------------------------------------------------------
    def record_download(self, paper_id: int, *, file_path: str, sha256: str, size_bytes: int, version_type: str,
                        source_url: str, landing_url: str, provider: str, verification: dict, query: str,
                        status: str = "downloaded") -> int:
        with self._tx() as c:
            cur = c.execute(
                "INSERT INTO downloads(paper_id,file_path,sha256,size_bytes,version_type,source_url,landing_url,provider,"
                "status,verification_json,access_date,query) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (paper_id, file_path, sha256, size_bytes, version_type, source_url, landing_url, provider, status,
                 json.dumps(verification), now_iso(), query))
            return int(cur.lastrowid)

    def find_download_by_hash(self, sha256: str) -> Optional[dict]:
        return self._existing("d.sha256=?", (sha256,))

    def find_download_for_paper(self, doi: str = "", title: str = "", year: Optional[int] = None) -> Optional[dict]:
        """Previously downloaded file (that still exists on disk) for the same DOI or normalised title."""
        doi = normalize_doi(doi)
        if doi:
            hit = self._existing("p.doi=?", (doi,))
            if hit:
                return hit
        norm = normalize_title(title)
        if norm:
            return self._existing("p.norm_title=?", (norm,))
        return None

    def _existing(self, where: str, args: tuple) -> Optional[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT d.*, p.title, p.doi, p.year FROM downloads d JOIN papers p ON p.id=d.paper_id "
                f"WHERE d.status='downloaded' AND {where} ORDER BY d.id DESC", args).fetchall()
        for r in rows:
            if r["file_path"] and Path(r["file_path"]).is_file():
                return dict(r)
        return None

    # ---- attempts / runs ----------------------------------------------------------------
    def record_attempts(self, run_id: Optional[int], paper_id: Optional[int], attempts) -> None:
        ts = now_iso()
        with self._tx() as c:
            c.executemany(
                "INSERT INTO attempts(run_id,paper_id,url,stage,status,detail,provider,ts) VALUES (?,?,?,?,?,?,?,?)",
                [(run_id, paper_id, a.url, a.stage, a.status, a.detail, a.provider, ts) for a in attempts])

    def start_run(self, query: str, query_type: str) -> int:
        with self._tx() as c:
            return int(c.execute("INSERT INTO runs(query,query_type,started_at) VALUES (?,?,?)",
                                 (query, query_type, now_iso())).lastrowid)

    def finish_run(self, run_id: int, *, outcome: str, access_state: str, paper_id: Optional[int], summary: str,
                   result: dict) -> None:
        with self._tx() as c:
            c.execute("UPDATE runs SET outcome=?, access_state=?, paper_id=?, summary=?, result_json=?, finished_at=? "
                      "WHERE id=?", (outcome, access_state, paper_id, summary, json.dumps(result), now_iso(), run_id))

    def get_run(self, run_id: int) -> Optional[dict]:
        with self._lock:
            r = self._conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        if not r:
            return None
        d = dict(r)
        d["result"] = json.loads(d.pop("result_json") or "null")
        return d

    def get_attempts(self, run_id: int) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self._conn.execute("SELECT * FROM attempts WHERE run_id=? ORDER BY id", (run_id,))]

    # ---- history ------------------------------------------------------------------------
    def history(self, search: str = "", limit: int = 200) -> list[dict]:
        """Runs joined with the paper and the downloaded file, newest first; ``search`` is a substring filter."""
        sql = ("SELECT r.id AS run_id, r.query, r.query_type, r.outcome, r.access_state, r.started_at, "
               "p.title, p.authors_json, p.year, p.doi, p.venue, d.file_path, d.version_type, d.source_url, d.sha256, "
               "d.status AS download_status, d.access_date "
               "FROM runs r LEFT JOIN papers p ON p.id=r.paper_id "
               "LEFT JOIN downloads d ON d.id=(SELECT MAX(id) FROM downloads WHERE paper_id=p.id AND status='downloaded') ")
        args: list = []
        if search.strip():
            like = f"%{search.strip().lower()}%"
            sql += ("WHERE lower(r.query) LIKE ? OR lower(COALESCE(p.title,'')) LIKE ? OR lower(COALESCE(p.authors_json,'')) LIKE ? "
                    "OR lower(COALESCE(p.doi,'')) LIKE ? OR lower(COALESCE(p.venue,'')) LIKE ? ")
            args = [like] * 5
        sql += "ORDER BY r.id DESC LIMIT ?"
        args.append(limit)
        with self._lock:
            rows = self._conn.execute(sql, args).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["authors"] = json.loads(d.pop("authors_json") or "[]")
            out.append(d)
        return out
