"""Open a downloaded PDF, check that it is sound, and decide whether it is the requested paper.

The document is untrusted input: it is only *parsed* (pypdf, no rendering, no JavaScript, no
embedded-file extraction) and its text is used purely as evidence for comparison. Instructions that
might appear in the text are ignored.
"""
from __future__ import annotations

import enum
import re
import threading
from dataclasses import dataclass, field, asdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

from models import Paper, VersionType
from paper_match import (content_tokens, normalize_doi, normalize_title, squash, surname, title_similarity)

import logging

logging.getLogger("pypdf").setLevel(logging.ERROR)      # font-encoding warnings are irrelevant for text matching

try:
    from pypdf import PdfReader
    from pypdf.errors import PyPdfError
except ImportError:                     # pragma: no cover
    PdfReader = None
    PyPdfError = Exception


class Verdict(str, enum.Enum):
    VERIFIED = "verified"
    PARTIAL = "partially_verified"
    MISMATCH = "mismatch"
    UNREADABLE = "no_extractable_text"
    ENCRYPTED = "encrypted"
    CORRUPT = "corrupt"
    UNSAFE = "unsafe"


@dataclass
class PdfReport:
    verdict: Verdict = Verdict.CORRUPT
    page_count: int = 0
    text_chars: int = 0
    title_score: float = 0.0
    authors_score: Optional[float] = None
    abstract_score: Optional[float] = None
    doi_found: bool = False
    metadata_title: str = ""
    detected_version: VersionType = VersionType.UNKNOWN
    version_evidence: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    reasons: list = field(default_factory=list)
    excerpt: str = ""

    @property
    def accepted(self) -> bool:
        return self.verdict in (Verdict.VERIFIED, Verdict.PARTIAL)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["verdict"] = self.verdict.value
        d["detected_version"] = self.detected_version.value
        return d


_ACTIVE = (rb"/Launch", rb"/JavaScript", rb"/JS ", rb"/JS(", rb"/OpenAction", rb"/EmbeddedFile", rb"/RichMedia")
_VERSION_PATTERNS = [
    (VersionType.PREPRINT, re.compile(r"(preprint submitted|this is a preprint|\bpreprint\b|working paper|discussion paper|"
                                       r"not (?:yet )?peer[- ]reviewed|arxiv:\s?\d|ssrn electronic journal|submitted to)", re.I)),
    (VersionType.ACCEPTED, re.compile(r"(accepted manuscript|author accepted manuscript|accepted version|post-?print|"
                                       r"author'?s? (?:accepted )?(?:version|manuscript)|this is the accepted)", re.I)),
    (VersionType.PUBLISHED, re.compile(r"(version of record|published online|©\s?\d{4}.{0,60}(?:elsevier|springer|wiley|oxford|cambridge|"
                                        r"taylor|sage|plos|mdpi|frontiers|nature)|creative commons attribution)", re.I)),
]


def run_with_timeout(fn, timeout: float):
    """Run ``fn`` in a daemon thread so a pathological PDF cannot hang the application."""
    box: dict = {}

    def target():
        try:
            box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001
            box["error"] = exc

    th = threading.Thread(target=target, daemon=True)
    th.start()
    th.join(timeout)
    if th.is_alive():
        raise TimeoutError(f"PDF processing exceeded {timeout:.0f}s")
    if "error" in box:
        raise box["error"]
    return box.get("value")


def check_structure(path: Path, min_bytes: int = 4096) -> tuple[bool, str]:
    """Cheap byte-level checks: size, signature, end marker."""
    try:
        size = path.stat().st_size
        with open(path, "rb") as fh:
            head = fh.read(1024)
            fh.seek(max(0, size - 2048))
            tail = fh.read()
    except OSError as exc:
        return False, f"cannot read file: {exc}"
    if size == 0:
        return False, "file is empty"
    if b"%PDF-" not in head:
        return False, "missing %PDF- signature"
    if size < min_bytes:
        return False, f"file too small ({size} bytes) to be a full paper"
    if b"%%EOF" not in tail:
        return False, "no %%EOF marker - file looks truncated"
    return True, "ok"


def _dehyphenate(text: str) -> str:
    return re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)


def best_title_window(title: str, text: str) -> float:
    """Best fuzzy match of ``title`` anywhere in ``text`` (0..1)."""
    sq_title = squash(title)
    if not sq_title:
        return 0.0
    if sq_title in squash(text):
        return 1.0
    main = re.split(r"\s*[:–—]\s+", title, maxsplit=1)[0]
    best = 0.0
    if main != title and len(main.split()) >= 4 and squash(main) in squash(text):
        best = 0.92
    tt = normalize_title(title).split()
    words = normalize_title(text[:15000]).split()
    m = len(tt)
    if m == 0 or len(words) < m - 1:
        return best
    target = " ".join(tt)
    tset = set(tt)
    for size in {m - 1, m, m + 1, m + 2}:
        if size < 1:
            continue
        for i in range(0, max(1, len(words) - size + 1)):
            window = words[i:i + size]
            if len(tset & set(window)) < 0.6 * len(tset):
                continue
            r = SequenceMatcher(None, target, " ".join(window), autojunk=False).ratio()
            if r > best:
                best = r
    return round(best, 4)


class PdfVerifier:
    def __init__(self, text_pages: int = 4, timeout: float = 40.0, min_bytes: int = 4096):
        self.text_pages = text_pages
        self.timeout = timeout
        self.min_bytes = min_bytes

    def verify(self, path: Path | str, paper: Paper) -> PdfReport:
        path = Path(path)
        report = PdfReport()
        ok, why = check_structure(path, self.min_bytes)
        if not ok:
            report.verdict, report.reasons = Verdict.CORRUPT, [why]
            return report
        try:
            raw = path.read_bytes()
        except OSError as exc:
            report.reasons = [f"cannot read file: {exc}"]
            return report
        if b"/Launch" in raw:
            report.verdict = Verdict.UNSAFE
            report.reasons = ["PDF contains a /Launch action (can start programs) - rejected"]
            return report
        found = [a.decode().strip("(/ ") for a in _ACTIVE if a in raw]
        if found:
            report.warnings.append("contains active-content markers (never executed): " + ", ".join(sorted(set(found))))
        del raw
        if PdfReader is None:
            report.reasons = ["pypdf is not installed; cannot inspect the document"]
            report.verdict = Verdict.UNREADABLE
            return report
        try:
            return run_with_timeout(lambda: self._inspect(path, paper, report), self.timeout)
        except TimeoutError as exc:
            report.verdict, report.reasons = Verdict.CORRUPT, [str(exc)]
        except Exception as exc:  # noqa: BLE001  (malformed PDFs raise many exception types)
            report.verdict, report.reasons = Verdict.CORRUPT, [f"cannot parse PDF: {type(exc).__name__}: {str(exc)[:120]}"]
        return report

    # ------------------------------------------------------------------------------------
    def _inspect(self, path: Path, paper: Paper, report: PdfReport) -> PdfReport:
        reader = PdfReader(str(path), strict=False)
        if reader.is_encrypted:
            try:
                ok = reader.decrypt("")
            except Exception:  # noqa: BLE001
                ok = 0
            if not ok:
                report.verdict = Verdict.ENCRYPTED
                report.reasons = ["PDF is password-protected; no access without credentials"]
                return report
            report.warnings.append("PDF was encrypted with an empty user password (opened normally)")
        report.page_count = len(reader.pages)
        if report.page_count < 1:
            report.verdict, report.reasons = Verdict.CORRUPT, ["PDF has no pages"]
            return report
        try:
            meta = reader.metadata
            report.metadata_title = str(meta.title or "") if meta else ""
        except Exception:  # noqa: BLE001
            report.metadata_title = ""
        chunks = []
        for i in range(min(self.text_pages, report.page_count)):
            try:
                chunks.append(reader.pages[i].extract_text() or "")
            except Exception:  # noqa: BLE001
                chunks.append("")
        text = _dehyphenate("\n".join(chunks))[:40000]
        report.text_chars = len(text.strip())
        report.excerpt = re.sub(r"\s+", " ", text)[:400]
        if report.page_count == 1:
            report.warnings.append("single-page PDF - may be an abstract, cover sheet or flyer")
        self._score(paper, text, report)
        self._detect_version(text, report)
        return report

    def _score(self, paper: Paper, text: str, report: PdfReport) -> None:
        if report.text_chars < 40:
            meta_sim = title_similarity(paper.title, report.metadata_title) if report.metadata_title else 0.0
            if meta_sim >= 0.9:
                report.verdict = Verdict.PARTIAL
                report.title_score = meta_sim
                report.reasons.append("no extractable text (scanned?), but the embedded PDF title matches")
            else:
                report.verdict = Verdict.UNREADABLE
                report.reasons.append("no extractable text (scanned images?) - the document cannot be verified")
            return
        head = text[:12000]
        report.title_score = best_title_window(paper.title, head) if paper.title else 0.0
        if report.metadata_title:
            report.title_score = max(report.title_score, title_similarity(paper.title, report.metadata_title) * 0.98)
        if paper.authors:
            names = [surname(a) for a in paper.authors[:6]]
            names = [n for n in names if len(n) >= 3]
            sq_head = squash(text[:8000])
            if names:
                report.authors_score = round(sum(1 for n in names if squash(n) in sq_head) / len(names), 3)
        if paper.abstract:
            toks = list(dict.fromkeys(content_tokens(paper.abstract[:900])))[:70]
            doc = set(content_tokens(text[:15000]))
            if len(toks) >= 8:
                report.abstract_score = round(sum(1 for t in toks if t in doc) / len(toks), 3)
        dois = {normalize_doi(paper.doi), *(normalize_doi(d) for d in paper.related_dois)} - {""}
        flat = re.sub(r"\s+", "", text[:30000]).lower()
        report.doi_found = any(d in flat for d in dois)

        t, a, ab = report.title_score, report.authors_score, report.abstract_score
        a_txt = "n/a" if a is None else f"{a:.0%}"
        ab_txt = "n/a" if ab is None else f"{ab:.0%}"
        if report.doi_found and (t >= 0.6 or (a or 0) >= 0.5):
            report.verdict = Verdict.VERIFIED
            report.reasons.append("expected DOI printed in the document and title/authors agree")
        elif t >= 0.999:
            if a is None or a >= 0.34:
                report.verdict = Verdict.VERIFIED
                report.reasons.append("title found verbatim in the document" + (f", {a_txt} of authors present" if a is not None else ""))
            else:
                report.verdict = Verdict.PARTIAL
                report.reasons.append(f"title found verbatim but expected authors are mostly absent ({a_txt})")
        elif t >= 0.9 and a is not None and a >= 0.5:
            report.verdict = Verdict.VERIFIED if t >= 0.95 else Verdict.PARTIAL
            report.reasons.append(f"title nearly identical (score {t:.2f}) and {a_txt} of authors present")
        elif t >= 0.9 and a is None:
            report.verdict = Verdict.PARTIAL
            report.reasons.append(f"title nearly identical (score {t:.2f}); no author list to compare")
        elif (t >= 0.75 and (a or 0) >= 0.5 and (ab or 0) >= 0.5) or ((ab or 0) >= 0.7 and (a or 0) >= 0.5):
            report.verdict = Verdict.PARTIAL
            report.reasons.append(f"title similarity {t:.2f}, authors {a_txt}, abstract overlap {ab_txt} - "
                                  "probable match (the title may differ between versions)")
        else:
            report.verdict = Verdict.MISMATCH
            report.reasons.append(f"document does not match the requested paper: title score {t:.2f}, authors {a_txt}, abstract overlap {ab_txt}")

    @staticmethod
    def _detect_version(text: str, report: PdfReport) -> None:
        head = text[:6000]
        hits = {}
        for vt, rx in _VERSION_PATTERNS:
            m = rx.findall(head)
            if m:
                hits[vt] = [x if isinstance(x, str) else x[0] for x in m][:3]
        if VersionType.ACCEPTED in hits:
            report.detected_version = VersionType.ACCEPTED
        elif VersionType.PREPRINT in hits:
            report.detected_version = VersionType.PREPRINT
        elif VersionType.PUBLISHED in hits:
            report.detected_version = VersionType.PUBLISHED
        if report.detected_version != VersionType.UNKNOWN:
            report.version_evidence = [str(x)[:60] for x in hits[report.detected_version]]
