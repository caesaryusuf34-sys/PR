"""Plain data structures shared by every module (no behaviour that touches the network)."""
from __future__ import annotations

import enum
from dataclasses import dataclass, field, asdict
from typing import Any, Optional


class QueryType(str, enum.Enum):
    DOI = "doi"
    ARXIV = "arxiv"
    TITLE = "exact title"
    AUTHOR_YEAR = "author and year"
    TOPIC = "research topic / question"
    REGULATION = "regulation"


class VersionType(str, enum.Enum):
    PUBLISHED = "published"
    ACCEPTED = "accepted"
    PREPRINT = "preprint"
    UNKNOWN = "unknown"

    @property
    def label(self) -> str:
        return {
            "published": "Final published article",
            "accepted": "Accepted manuscript",
            "preprint": "Preprint",
            "unknown": "Full text (version not stated)",
        }[self.value]


class Outcome(str, enum.Enum):
    DOWNLOADED = "downloaded"
    LINK_FOUND = "link_found"            # verified PDF link returned, file not saved (auto-download off)
    NEEDS_CHOICE = "needs_choice"
    NO_FULLTEXT = "no_fulltext"
    NOT_FOUND = "not_found"


class AccessState(str, enum.Enum):
    """How the *publication* can be accessed (what the UI shows as the access-status badge)."""
    PUBLISHED = "published"
    ACCEPTED = "accepted"
    PREPRINT = "preprint"
    FULLTEXT_UNKNOWN_VERSION = "fulltext_unknown_version"
    OFFICIAL_REGULATION = "official_regulation"
    ABSTRACT_ONLY = "abstract_only"
    INACCESSIBLE = "inaccessible"
    UNKNOWN = "unknown"

    @property
    def label(self) -> str:
        return {
            "published": "Final published article (full text retrieved)",
            "accepted": "Accepted manuscript (full text retrieved)",
            "preprint": "Preprint (full text retrieved; not the final published article)",
            "fulltext_unknown_version": "Full text retrieved (version not stated)",
            "official_regulation": "Official regulation (PDF from the issuing authority / official portal)",
            "abstract_only": "Abstract-only record (no legal full text found)",
            "inaccessible": "Inaccessible (paywall / login / no open copy found)",
            "unknown": "Not identified",
        }[self.value]


class ProbeStatus(str, enum.Enum):
    PDF = "pdf"
    HTML_PAGE = "html_page"
    LOGIN_REQUIRED = "login_required"
    PAYWALL = "paywall"
    CAPTCHA = "captcha"
    ROBOTS_BLOCKED = "robots_blocked"
    FORBIDDEN = "forbidden"
    NOT_FOUND = "not_found"
    RATE_LIMITED = "rate_limited"
    SERVER_ERROR = "server_error"
    NETWORK_ERROR = "network_error"
    NOT_PDF = "not_pdf"
    TOO_LARGE = "too_large"
    UNSAFE_URL = "unsafe_url"
    BLOCKED_DOMAIN = "blocked_domain"
    EMPTY = "empty"
    BUDGET = "budget_exhausted"
    CORRUPT = "corrupt_or_truncated"
    REJECTED = "rejected_after_verification"


@dataclass
class FullTextLocation:
    url: str
    kind: str = "unknown"                 # "pdf" | "landing" | "unknown"
    version: VersionType = VersionType.UNKNOWN
    host_type: str = "web"                # publisher | repository | preprint_server | aggregator | doi | web
    provider: str = ""
    license: str = ""
    is_oa: Optional[bool] = None
    repository: str = ""
    origin_url: str = ""                  # page on which the link was discovered
    note: str = ""
    depth: int = 0
    trusted: bool = True                  # False when it came from an open-web hit that is not yet matched
    probed_pdf: bool = False              # already confirmed to return a PDF (skip the extra probe)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["version"] = self.version.value
        return d


@dataclass
class Paper:
    title: str = ""
    authors: list = field(default_factory=list)
    year: Optional[int] = None
    venue: str = ""
    doi: str = ""
    abstract: str = ""
    url: str = ""                         # canonical landing page (usually https://doi.org/<doi>)
    arxiv_id: str = ""
    pmid: str = ""
    pmcid: str = ""
    work_type: str = ""                   # journal-article, posted-content, ...
    is_oa: Optional[bool] = None
    citation_count: Optional[int] = None
    related_dois: list = field(default_factory=list)   # preprint / published counterparts
    locations: list = field(default_factory=list)      # list[FullTextLocation]
    sources: list = field(default_factory=list)        # provider names that returned this record
    record_urls: dict = field(default_factory=dict)    # provider -> URL of its record (for citations)
    provenance: list = field(default_factory=list)     # per-provider bibliographic snapshots
    score: float = 0.0                                 # relevance / match score for the current query

    @property
    def first_author_surname(self) -> str:
        from paper_match import surname
        return surname(self.authors[0]) if self.authors else ""

    @property
    def doi_url(self) -> str:
        return f"https://doi.org/{self.doi}" if self.doi else ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["locations"] = [l.to_dict() if isinstance(l, FullTextLocation) else l for l in self.locations]
        return d


@dataclass
class ProbeResult:
    url: str
    status: ProbeStatus
    final_url: str = ""
    http_status: Optional[int] = None
    content_type: str = ""
    detail: str = ""
    html: str = ""                        # decoded body when the response was an HTML page
    content_length: Optional[int] = None

    @property
    def ok_pdf(self) -> bool:
        return self.status == ProbeStatus.PDF

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        d.pop("html", None)
        return d


@dataclass
class Attempt:
    """One examined full-text location and what happened (the evidence trail)."""
    url: str
    status: str
    detail: str = ""
    provider: str = ""
    stage: str = ""                       # probe | crawl | download | verify
    version: str = ""
    http_status: Optional[int] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ProgressEvent:
    stage: str
    message: str
    url: str = ""
    level: str = "info"                   # info | success | warning | error

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Citation:
    label: int
    title: str
    url: str
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DownloadInfo:
    path: str
    sha256: str
    size_bytes: int
    source_url: str
    landing_url: str
    provider: str
    version: VersionType
    verification: dict = field(default_factory=dict)
    access_date: str = ""
    already_had: bool = False
    saved: bool = True                    # False in link-only mode: path is empty, source_url is the PDF link

    def to_dict(self) -> dict:
        d = asdict(self)
        d["version"] = self.version.value
        return d


@dataclass
class ResearchResult:
    query: str
    query_type: QueryType
    outcome: Outcome
    access_state: AccessState = AccessState.UNKNOWN
    paper: Optional[Paper] = None
    alternatives: list = field(default_factory=list)      # other candidate Papers (ambiguity / topic search)
    other_versions: list = field(default_factory=list)    # other FullTextLocations seen for the chosen paper
    download: Optional[DownloadInfo] = None
    official_link: str = ""
    attempts: list = field(default_factory=list)
    events: list = field(default_factory=list)
    citations: list = field(default_factory=list)
    providers_used: list = field(default_factory=list)
    providers_skipped: dict = field(default_factory=dict)
    pages_visited: list = field(default_factory=list)
    summary: str = ""
    match_note: str = ""
    requests_used: int = 0
    elapsed_seconds: float = 0.0
    run_id: Optional[int] = None
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "query_type": self.query_type.value,
            "outcome": self.outcome.value,
            "access_state": self.access_state.value,
            "paper": self.paper.to_dict() if self.paper else None,
            "alternatives": [p.to_dict() for p in self.alternatives],
            "other_versions": [l.to_dict() for l in self.other_versions],
            "download": self.download.to_dict() if self.download else None,
            "official_link": self.official_link,
            "attempts": [a.to_dict() for a in self.attempts],
            "events": [e.to_dict() for e in self.events],
            "citations": [c.to_dict() for c in self.citations],
            "providers_used": self.providers_used,
            "providers_skipped": self.providers_skipped,
            "pages_visited": self.pages_visited,
            "summary": self.summary,
            "match_note": self.match_note,
            "requests_used": self.requests_used,
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "error": self.error,
        }
