"""Settings and API configuration.

Every secret is read from environment variables (optionally loaded from a local ``.env`` file that
is never committed). Nothing in this module performs network access.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Mapping, Optional

BASE_DIR = Path(__file__).resolve().parent
APP_NAME = "AcademicResearchAgent"
APP_VERSION = "1.0"

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

# Shadow libraries and similar services distribute material without the rights holders' consent.
# The agent never contacts them, even if a search engine returns them.
DEFAULT_BLOCKED_DOMAINS = (
    "sci-hub.se", "sci-hub.st", "sci-hub.ru", "sci-hub.ee", "sci-hub.wf", "sci-hub.ren", "scihub.org",
    "sci-hub.tw", "sci-hub.do", "sci-hub.cc", "libgen.is", "libgen.rs", "libgen.st", "libgen.li",
    "libgen.lc", "library.lol", "z-lib.org", "z-library.sk", "zlibrary.to", "1lib.sk", "b-ok.org",
    "annas-archive.org", "annas-archive.se", "pdfdrive.com", "sci-net.xyz", "scimag.org",
)
# Sites whose terms forbid automated access and that sit behind login/anti-bot walls. They may be
# *listed* as links for a human to open, but the agent never requests them.
DEFAULT_NO_CRAWL_DOMAINS = (
    "researchgate.net", "academia.edu", "scribd.com", "sciencedirect.com", "linkedin.com",
    "facebook.com", "x.com", "twitter.com",
)


def load_dotenv(path: Optional[Path] = None, environ: Optional[dict] = None) -> int:
    """Load ``KEY=VALUE`` lines from a .env file into ``environ`` without overriding real variables."""
    path = path or BASE_DIR / ".env"
    environ = os.environ if environ is None else environ
    if not path.is_file():
        return 0
    loaded = 0
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        if key and key not in environ:
            environ[key] = value
            loaded += 1
    return loaded


def _env_int(env: Mapping[str, str], name: str, default: int) -> int:
    try:
        return int(env.get(name, "").strip() or default)
    except ValueError:
        return default


def _env_float(env: Mapping[str, str], name: str, default: float) -> float:
    try:
        return float(env.get(name, "").strip() or default)
    except ValueError:
        return default


def _env_bool(env: Mapping[str, str], name: str, default: bool) -> bool:
    val = env.get(name, "").strip().lower()
    if not val:
        return default
    return val in {"1", "true", "yes", "on"}


def _env_list(env: Mapping[str, str], name: str, default: tuple) -> tuple:
    val = env.get(name, "").strip()
    if not val:
        return default
    return tuple(x.strip().lower() for x in val.split(",") if x.strip())


@dataclass(frozen=True)
class Settings:
    # identity / credentials (all optional; providers that need a missing credential are skipped)
    contact_email: str = ""
    core_api_key: str = ""
    semantic_scholar_api_key: str = ""
    openalex_api_key: str = ""
    brave_api_key: str = ""
    google_cse_api_key: str = ""
    google_cse_cx: str = ""
    # storage
    download_dir: Path = BASE_DIR / "Downloaded_Papers"
    db_path: Path = BASE_DIR / "research_history.sqlite3"
    # networking
    connect_timeout: float = 8.0
    read_timeout: float = 25.0
    max_retries: int = 2
    max_redirects: int = 6
    host_min_interval: float = 0.5
    respect_robots: bool = True
    # search budget
    max_requests: int = 300          # every HTTP request (APIs, crawling, downloads) counts
    max_pages: int = 12              # HTML pages the crawler may visit
    max_depth: int = 2               # link depth from a seed page
    max_candidates: int = 16         # full-text locations the agent will examine
    max_download_attempts: int = 8   # PDFs actually downloaded
    max_search_queries: int = 8      # alternative-version query strings issued
    time_limit_seconds: float = 300.0
    results_per_provider: int = 8
    max_workers: int = 6
    # files
    max_pdf_mb: int = 60
    min_pdf_bytes: int = 4096
    max_html_kb: int = 2048
    max_json_mb: int = 6
    pdf_text_pages: int = 4
    pdf_timeout_seconds: float = 40.0
    # decisions
    exact_match_threshold: float = 0.93
    possible_match_threshold: float = 0.75
    accept_partial_verification: bool = True
    allow_preprints: bool = True
    auto_pick_topic: bool = False
    blocked_domains: tuple = DEFAULT_BLOCKED_DOMAINS
    no_crawl_domains: tuple = DEFAULT_NO_CRAWL_DOMAINS
    # per-host minimum seconds between requests (politeness / documented API limits)
    host_intervals: dict = field(default_factory=lambda: {
        "api.crossref.org": 0.15, "api.openalex.org": 0.25, "api.semanticscholar.org": 1.1,
        "api.unpaywall.org": 0.2, "api.core.ac.uk": 1.2, "www.ebi.ac.uk": 0.25, "doaj.org": 0.6,
        "export.arxiv.org": 3.1, "api.search.brave.com": 1.1, "www.googleapis.com": 0.5,
        "api.openaire.eu": 0.5, "doi.org": 0.2,
    })

    # ---- derived -------------------------------------------------------------------------
    @property
    def user_agent(self) -> str:
        contact = f"mailto:{self.contact_email}" if self.contact_email else "academic research tool"
        return f"{APP_NAME}/{APP_VERSION} ({contact})"

    @property
    def max_pdf_bytes(self) -> int:
        return self.max_pdf_mb * 1024 * 1024

    @property
    def has_valid_email(self) -> bool:
        return bool(_EMAIL_RE.match(self.contact_email or "")) and not self.contact_email.lower().endswith(
            ("@example.com", "@example.org", "@test.com"))

    @property
    def has_web_search(self) -> bool:
        return bool(self.brave_api_key or (self.google_cse_api_key and self.google_cse_cx))

    def replace(self, **changes) -> "Settings":
        return replace(self, **changes)

    def public_summary(self) -> dict:
        """Configuration overview that never exposes secret values."""
        return {
            "contact_email": self.contact_email or None,
            "core_api_key": bool(self.core_api_key),
            "semantic_scholar_api_key": bool(self.semantic_scholar_api_key),
            "openalex_api_key": bool(self.openalex_api_key),
            "brave_api_key": bool(self.brave_api_key),
            "google_cse": bool(self.google_cse_api_key and self.google_cse_cx),
            "download_dir": str(self.download_dir),
            "db_path": str(self.db_path),
        }

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None, *, load_env_file: bool = True) -> "Settings":
        if env is None:
            if load_env_file:
                load_dotenv()
            env = os.environ
        d = cls()
        email = (env.get("CONTACT_EMAIL") or env.get("UNPAYWALL_EMAIL") or "").strip()
        download_dir = Path(env["DOWNLOAD_DIR"]).expanduser() if env.get("DOWNLOAD_DIR") else d.download_dir
        db_path = Path(env["DATABASE_PATH"]).expanduser() if env.get("DATABASE_PATH") else d.db_path
        return cls(
            contact_email=email,
            core_api_key=env.get("CORE_API_KEY", "").strip(),
            semantic_scholar_api_key=env.get("SEMANTIC_SCHOLAR_API_KEY", "").strip(),
            openalex_api_key=env.get("OPENALEX_API_KEY", "").strip(),
            brave_api_key=env.get("BRAVE_API_KEY", "").strip(),
            google_cse_api_key=env.get("GOOGLE_CSE_API_KEY", "").strip(),
            google_cse_cx=env.get("GOOGLE_CSE_CX", "").strip(),
            download_dir=download_dir,
            db_path=db_path,
            connect_timeout=_env_float(env, "HTTP_CONNECT_TIMEOUT", d.connect_timeout),
            read_timeout=_env_float(env, "HTTP_READ_TIMEOUT", d.read_timeout),
            max_retries=_env_int(env, "HTTP_MAX_RETRIES", d.max_retries),
            host_min_interval=_env_float(env, "HOST_MIN_INTERVAL", d.host_min_interval),
            respect_robots=_env_bool(env, "RESPECT_ROBOTS", d.respect_robots),
            max_requests=_env_int(env, "AGENT_MAX_REQUESTS", d.max_requests),
            max_pages=_env_int(env, "AGENT_MAX_PAGES", d.max_pages),
            max_depth=_env_int(env, "AGENT_MAX_DEPTH", d.max_depth),
            max_candidates=_env_int(env, "AGENT_MAX_CANDIDATES", d.max_candidates),
            max_download_attempts=_env_int(env, "AGENT_MAX_DOWNLOADS", d.max_download_attempts),
            max_search_queries=_env_int(env, "AGENT_MAX_QUERIES", d.max_search_queries),
            time_limit_seconds=_env_float(env, "AGENT_TIME_LIMIT", d.time_limit_seconds),
            max_pdf_mb=_env_int(env, "MAX_PDF_MB", d.max_pdf_mb),
            accept_partial_verification=_env_bool(env, "ACCEPT_PARTIAL_VERIFICATION", d.accept_partial_verification),
            allow_preprints=_env_bool(env, "ALLOW_PREPRINTS", d.allow_preprints),
            auto_pick_topic=_env_bool(env, "AUTO_PICK_TOPIC", d.auto_pick_topic),
            blocked_domains=tuple(sorted(set(d.blocked_domains) | set(_env_list(env, "EXTRA_BLOCKED_DOMAINS", ())))),
            no_crawl_domains=tuple(sorted(set(d.no_crawl_domains) | set(_env_list(env, "EXTRA_NO_CRAWL_DOMAINS", ())))),
        )
