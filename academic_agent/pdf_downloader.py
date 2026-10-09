"""Download a PDF safely and store it under ``Downloaded_Papers``.

Streaming download -> temp file inside the download directory -> structural validation. The caller
(the research agent) then runs the content verifier and either :meth:`PdfDownloader.finalize` the
file (moves it to its descriptive name) or :meth:`PdfDownloader.discard` it. Nothing is ever kept
unless it was validated.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from access_checker import AccessChecker, looks_like_pdf
from config import Settings
from http_client import (BlockedDomain, BudgetExceeded, HttpClient, HttpError, RateLimited, RobotsDisallowed)
from models import Paper, ProbeStatus, VersionType
from net_safety import URLValidationError
from paper_match import make_filename
from pdf_verifier import check_structure


@dataclass
class DownloadResult:
    status: ProbeStatus
    detail: str = ""
    path: Optional[Path] = None
    sha256: str = ""
    size: int = 0
    final_url: str = ""
    content_type: str = ""
    http_status: Optional[int] = None

    @property
    def ok(self) -> bool:
        return self.status == ProbeStatus.PDF and self.path is not None


class PdfDownloader:
    def __init__(self, http: HttpClient, checker: AccessChecker, settings: Settings, *, attempts: int = 2):
        self.http = http
        self.checker = checker
        self.settings = settings
        self.attempts = max(1, attempts)
        self._cleanup_stale_partials()

    def _cleanup_stale_partials(self, max_age_s: float = 86400.0) -> None:
        """Remove temp files left behind by an interrupted earlier run."""
        d = self.settings.download_dir / ".partial"
        if not d.is_dir():
            return
        cutoff = time.time() - max_age_s
        for f in d.glob("*.part"):
            try:
                if f.stat().st_mtime < cutoff:
                    f.unlink()
            except OSError:
                pass

    @property
    def partial_dir(self) -> Path:
        d = self.settings.download_dir / ".partial"
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ------------------------------------------------------------------------------------
    def download(self, url: str, *, referer: Optional[str] = None) -> DownloadResult:
        """Fetch ``url`` into a temporary file; retries transient failures (bounded)."""
        last = DownloadResult(ProbeStatus.NETWORK_ERROR, "download not attempted")
        for attempt in range(self.attempts):
            last = self._download_once(url, referer)
            if last.ok or last.status not in (ProbeStatus.NETWORK_ERROR, ProbeStatus.SERVER_ERROR, ProbeStatus.CORRUPT):
                return last
            if self.http.budget.exhausted:
                break
            time.sleep(min(1.5 * (attempt + 1), 4.0) if attempt + 1 < self.attempts else 0)
        return last

    def _download_once(self, url: str, referer: Optional[str]) -> DownloadResult:
        headers = {"Accept": "application/pdf,*/*;q=0.5"}
        if referer:
            headers["Referer"] = referer
        if self.http.is_no_crawl(url):
            return DownloadResult(ProbeStatus.ROBOTS_BLOCKED, "site forbids automated access (terms of use)")
        try:
            resp = self.http.request("GET", url, headers=headers, stream=True, check_robots=True)
        except RobotsDisallowed as exc:
            return DownloadResult(ProbeStatus.ROBOTS_BLOCKED, str(exc))
        except BlockedDomain as exc:
            return DownloadResult(ProbeStatus.BLOCKED_DOMAIN, str(exc))
        except URLValidationError as exc:
            return DownloadResult(ProbeStatus.UNSAFE_URL, str(exc))
        except BudgetExceeded as exc:
            return DownloadResult(ProbeStatus.BUDGET, str(exc))
        except RateLimited as exc:
            return DownloadResult(ProbeStatus.RATE_LIMITED, str(exc), http_status=429)
        except HttpError as exc:
            return DownloadResult(ProbeStatus.NETWORK_ERROR, str(exc))

        ctype = resp.headers.get("Content-Type", "").split(";")[0].strip().lower()
        final = resp.url or url
        if resp.status_code != 200:
            probe = self.checker.classify_response(url, resp)       # explains 401/402/403/404/5xx pages
            return DownloadResult(probe.status, probe.detail, final_url=final, content_type=ctype,
                                  http_status=resp.status_code)
        declared = resp.headers.get("Content-Length")
        declared_i = int(declared) if declared and declared.isdigit() else None
        if declared_i and declared_i > self.settings.max_pdf_bytes:
            resp.close()
            return DownloadResult(ProbeStatus.TOO_LARGE, f"declared size {declared_i / 1048576:.0f} MB exceeds limit of "
                                  f"{self.settings.max_pdf_mb} MB", final_url=final, content_type=ctype, http_status=200)

        fd, tmp_name = tempfile.mkstemp(prefix="dl_", suffix=".part", dir=self.partial_dir)
        tmp = Path(tmp_name)
        sha, size, first = hashlib.sha256(), 0, True
        try:
            with os.fdopen(fd, "wb") as out:
                chunks = iter(resp.iter_content(chunk_size=65536))
                for chunk in chunks:
                    if not chunk:
                        continue
                    if first:
                        first = False
                        if not looks_like_pdf(chunk[:1024]):
                            probe = self.checker.classify_response(url, _Replay(resp, chunk, chunks))
                            status = probe.status if probe.status != ProbeStatus.PDF else ProbeStatus.NOT_PDF
                            if status == ProbeStatus.HTML_PAGE:
                                status = ProbeStatus.NOT_PDF
                            detail = probe.detail if status != ProbeStatus.NOT_PDF else (
                                f"server returned {'HTML' if ctype == 'text/html' else ctype or 'non-PDF content'} instead of a PDF")
                            raise _Reject(DownloadResult(status, detail, final_url=final, content_type=ctype, http_status=200))
                    size += len(chunk)
                    if size > self.settings.max_pdf_bytes:
                        raise _Reject(DownloadResult(ProbeStatus.TOO_LARGE, f"download exceeded {self.settings.max_pdf_mb} MB",
                                                     final_url=final, content_type=ctype, http_status=200))
                    sha.update(chunk)
                    out.write(chunk)
        except _Reject as rej:
            self._rm(tmp)
            return rej.result
        except Exception as exc:  # noqa: BLE001  (connection reset, timeouts mid-stream)
            self._rm(tmp)
            return DownloadResult(ProbeStatus.NETWORK_ERROR, f"download interrupted: {type(exc).__name__}: {exc}"[:200],
                                  final_url=final, content_type=ctype)
        finally:
            resp.close()

        if size == 0:
            self._rm(tmp)
            return DownloadResult(ProbeStatus.EMPTY, "empty response body", final_url=final, content_type=ctype)
        if declared_i and size < declared_i and not resp.headers.get("Content-Encoding"):
            self._rm(tmp)
            return DownloadResult(ProbeStatus.CORRUPT, f"truncated: received {size} of {declared_i} bytes", final_url=final)
        ok, why = check_structure(tmp, self.settings.min_pdf_bytes)
        if not ok:
            self._rm(tmp)
            return DownloadResult(ProbeStatus.CORRUPT, why, final_url=final, content_type=ctype, size=size)
        return DownloadResult(ProbeStatus.PDF, "downloaded", path=tmp, sha256=sha.hexdigest(), size=size,
                              final_url=final, content_type=ctype, http_status=200)

    # ------------------------------------------------------------------------------------
    @staticmethod
    def _rm(path: Path) -> None:
        try:
            path.unlink()
        except OSError:
            pass

    def discard(self, result: DownloadResult) -> None:
        if result.path:
            self._rm(result.path)

    def finalize(self, result: DownloadResult, paper: Paper, version: VersionType, filename: Optional[str] = None) -> Path:
        """Move the validated temp file to ``Downloaded_Papers/<year>_<Author>_<Title>.pdf``."""
        assert result.path is not None
        self.settings.download_dir.mkdir(parents=True, exist_ok=True)
        name = filename or make_filename(paper, version)
        target = self.settings.download_dir / name
        n = 2
        while target.exists():
            if _sha256_file(target) == result.sha256:         # identical file already stored
                self._rm(result.path)
                return target
            target = self.settings.download_dir / f"{Path(name).stem}_{n}.pdf"
            n += 1
        shutil.move(str(result.path), str(target))
        return target


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class _Reject(Exception):
    def __init__(self, result: DownloadResult):
        self.result = result


class _Replay:
    """Wraps a response whose first chunk was already consumed so the access checker can classify it."""

    def __init__(self, resp, first_chunk: bytes, chunks):
        self._resp, self._first, self._chunks = resp, first_chunk, chunks
        self.headers, self.url, self.status_code = resp.headers, resp.url, resp.status_code

    def iter_content(self, chunk_size=8192):
        yield self._first
        yield from self._chunks

    def close(self):
        self._resp.close()
