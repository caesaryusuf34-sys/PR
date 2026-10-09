"""Bounded, polite crawler for pages that may lead to a legal full-text copy.

Starting from seed pages (publisher landing pages, repository records, search hits) it extracts
bibliographic metadata and *candidate links* to the full text, and follows only links that look
relevant - never beyond ``max_pages`` / ``max_depth``. Everything it reads is untrusted data: HTML is parsed
(never executed) and text inside a page is never treated as an instruction.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urldefrag, urljoin, urlsplit

from bs4 import BeautifulSoup

from access_checker import AccessChecker
from config import Settings
from http_client import HttpClient
from models import FullTextLocation, Paper, ProbeResult, ProbeStatus, VersionType
from net_safety import ALLOWED_SCHEMES
from paper_match import classify_host, normalize_doi, normalize_title, squash, title_similarity

try:                                    # lxml is faster and more tolerant, but optional
    import lxml  # noqa: F401
    _PARSER = "lxml"
except ImportError:                     # pragma: no cover
    _PARSER = "html.parser"

_PDF_TEXT = re.compile(r"\b(pdf|full[- ]?text|download|view article|read article|open (?:full )?text)\b", re.I)
_VERSION_TEXT = re.compile(r"(accepted manuscript|author(?:'s|s)? (?:accepted )?(?:manuscript|version)|post-?print|"
                           r"pre-?print|green open access|repository version|self-archived|final version|version of record)", re.I)
_BAD_LINK_TEXT = re.compile(r"(supplement|appendix|reviewer|peer review|cover|table of contents|toc\b|slides?|poster|"
                            r"permission|reprint|citation|endnote|bibtex|ris\b|figure|presentation|erratum|correction|"
                            r"press release|flyer|brochure|sample|preview|first page|abstract only)", re.I)
_BAD_LINK_URL = re.compile(r"(supplement|suppl[_-]|appendix|mmc\d|media-\d|\.(?:zip|docx?|xlsx?|pptx?|csv|png|jpe?g|gif|svg|bib|ris)(?:$|\?)|"
                           r"/toc[/.]|/cover|/figure|/reviewer)", re.I)
_FILE_URL = re.compile(r"(\.pdf(?:$|[?#])|/pdf/|/pdfs?/|/download(?:/|\?|$)|/bitstreams?/|/viewcontent\.cgi|/fulltext|"
                       r"/files?/[^/]+\.pdf|/eprint/\d+/|/id/eprint/\d+/\d+/|ndownloader\.figshare\.com|/content/pdf/|/doi/pdf/|"
                       r"/articles?/[^/]+/pdf|/document/\d+|/record/\d+/files/|/records/[^/]+/files/)", re.I)
_FOLLOW_TEXT = re.compile(r"(full[- ]?text|open access|repository|accepted manuscript|author version|post-?print|"
                          r"preprint|green oa|view (?:the )?(?:article|record|item)|available (?:at|from|in)|"
                          r"institutional|download|pdf|eprint|dspace|handle|item record)", re.I)
_PAYWALL_RE = re.compile(r"(purchase|buy this|subscribe|rent this|get access|paywall|pay[- ]per[- ]view)", re.I)
MAX_PDF_LINKS_PER_PAGE = 6
MAX_FOLLOW_LINKS_PER_PAGE = 6


@dataclass
class LinkCandidate:
    url: str
    score: int
    text: str = ""
    source: str = ""      # meta | link-rel | anchor | jsonld


@dataclass
class PageInfo:
    url: str
    final_url: str = ""
    status: str = ""
    http_status: Optional[int] = None
    title: str = ""
    meta_title: str = ""
    meta_authors: list = field(default_factory=list)
    meta_doi: str = ""
    meta_year: Optional[int] = None
    meta_journal: str = ""
    meta_abstract: str = ""
    pdf_links: list = field(default_factory=list)       # list[LinkCandidate]
    follow_links: list = field(default_factory=list)    # list[LinkCandidate]
    open_access_signal: bool = False
    paywall_signal: bool = False
    version_hint: VersionType = VersionType.UNKNOWN
    match_score: float = 0.0
    matches_paper: bool = False
    note: str = ""

    def summary(self) -> dict:
        return {"url": self.final_url or self.url, "status": self.status, "title": self.meta_title or self.title,
                "match_score": round(self.match_score, 2), "matches": self.matches_paper,
                "pdf_links": [l.url for l in self.pdf_links], "note": self.note}


@dataclass
class CrawlReport:
    locations: list = field(default_factory=list)       # list[FullTextLocation] (PDF candidates)
    pages: list = field(default_factory=list)           # list[PageInfo]
    attempts: list = field(default_factory=list)        # (url, status, detail)
    metadata: dict = field(default_factory=dict)        # extra bibliographic facts learnt (doi, abstract, ...)
    fetched: int = 0                                    # pages requested (counts against max_pages)


# ----------------------------------------------------------------------------- parsing
def _meta_values(soup: BeautifulSoup, *names: str) -> list[str]:
    out: list[str] = []
    wanted = {n.lower() for n in names}
    for tag in soup.find_all("meta"):
        key = (tag.get("name") or tag.get("property") or "").strip().lower()
        if key in wanted and tag.get("content"):
            out.append(tag["content"].strip())
    return out


def _clean(url: str, base: str) -> Optional[str]:
    if not url:
        return None
    url = url.strip()
    if url.lower().startswith(("javascript:", "mailto:", "tel:", "data:", "#")):
        return None
    full = urldefrag(urljoin(base, url))[0]
    if urlsplit(full).scheme.lower() not in ALLOWED_SCHEMES:
        return None
    return full


def _jsonld_pdfs(soup: BeautifulSoup, base: str) -> list[str]:
    urls: list[str] = []

    def walk(node):
        if isinstance(node, dict):
            fmt = str(node.get("encodingFormat") or node.get("fileFormat") or "").lower()
            cu = node.get("contentUrl") or node.get("url")
            if "pdf" in fmt and isinstance(cu, str):
                urls.append(cu)
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    for s in soup.find_all("script", attrs={"type": "application/ld+json"}):
        try:
            walk(json.loads(s.string or s.get_text() or "null"))
        except (ValueError, TypeError):
            continue
    return [u for u in (_clean(x, base) for x in urls) if u]


def parse_page(html: str, base_url: str, paper: Optional[Paper] = None) -> PageInfo:
    soup = BeautifulSoup(html, _PARSER)
    info = PageInfo(url=base_url, final_url=base_url)
    t = soup.find("title")
    info.title = re.sub(r"\s+", " ", t.get_text()).strip() if t else ""
    h1 = soup.find("h1")
    h1_text = re.sub(r"\s+", " ", h1.get_text()).strip() if h1 else ""
    mt = _meta_values(soup, "citation_title", "dc.title", "dcterms.title", "eprints.title", "prism.title", "og:title")
    info.meta_title = mt[0] if mt else h1_text
    info.meta_authors = _meta_values(soup, "citation_author", "dc.creator", "dc.contributor", "eprints.creators_name")
    dois = _meta_values(soup, "citation_doi", "dc.identifier", "prism.doi", "bepress_citation_doi", "dc.identifier.doi")
    info.meta_doi = next((d for d in (normalize_doi(x) for x in dois) if d), "")
    date = (_meta_values(soup, "citation_publication_date", "citation_date", "citation_online_date", "dc.date",
                         "prism.publicationdate", "article:published_time") or [""])[0]
    m = re.search(r"(19|20)\d{2}", date)
    info.meta_year = int(m.group(0)) if m else None
    info.meta_journal = (_meta_values(soup, "citation_journal_title", "prism.publicationname", "dc.source") or [""])[0]
    info.meta_abstract = (_meta_values(soup, "citation_abstract", "dc.description", "description", "og:description") or [""])[0]

    jsonld_pdfs = _jsonld_pdfs(soup, base_url)          # read before scripts are dropped; parsed as data only

    # page-text signals (visible text only; scripts/styles dropped)
    for bad in soup(["script", "style", "noscript", "template"]):
        bad.decompose()
    text = re.sub(r"\s+", " ", soup.get_text(" ")).strip()
    head_text = text[:6000]
    info.open_access_signal = bool(re.search(r"open access|creative commons|cc[- ]by|free to read", head_text, re.I))
    info.paywall_signal = bool(_PAYWALL_RE.search(head_text)) and not info.open_access_signal
    vm = _VERSION_TEXT.search(head_text)
    if vm:
        v = vm.group(0).lower()
        info.version_hint = (VersionType.PREPRINT if "pre" in v and "print" in v and "post" not in v else
                             VersionType.ACCEPTED if any(k in v for k in ("accepted", "author", "post", "green", "self-archived", "repository")) else
                             VersionType.PUBLISHED)

    # page <-> paper matching
    if paper is not None and paper.title:
        scores = [title_similarity(paper.title, x) for x in [info.meta_title, h1_text, info.title] if x]
        info.match_score = max(scores, default=0.0)
        doi_hit = bool(paper.doi and (info.meta_doi == normalize_doi(paper.doi) or normalize_doi(paper.doi) in text.lower()))
        sq_title = squash(paper.title)
        in_text = len(sq_title) > 25 and sq_title in squash(text[:20000])
        info.matches_paper = info.match_score >= 0.8 or doi_hit or in_text
        if doi_hit or in_text:
            info.match_score = max(info.match_score, 0.9)

    # ---- links ---------------------------------------------------------------------------
    pdfs: dict[str, LinkCandidate] = {}
    follows: dict[str, LinkCandidate] = {}

    def add_pdf(url, score, text="", source=""):
        if url and (url not in pdfs or pdfs[url].score < score):
            pdfs[url] = LinkCandidate(url, score, text, source)

    for v in _meta_values(soup, "citation_pdf_url", "eprints.document_url", "bepress_citation_pdf_url", "wkhealth_pdf_url"):
        add_pdf(_clean(v, base_url), 100, "citation_pdf_url", "meta")
    for link in soup.find_all("link"):
        rel = " ".join(link.get("rel") or []).lower()
        typ = (link.get("type") or "").lower()
        if "alternate" in rel and "pdf" in typ:
            add_pdf(_clean(link.get("href"), base_url), 90, "link rel=alternate", "link-rel")
    for u in jsonld_pdfs:
        add_pdf(u, 85, "JSON-LD", "jsonld")
    abs_md = _meta_values(soup, "citation_abstract_html_url", "citation_fulltext_html_url")
    for v in abs_md:
        u = _clean(v, base_url)
        if u and u != base_url:
            follows.setdefault(u, LinkCandidate(u, 50, "citation_*_html_url", "meta"))

    for a in soup.find_all("a", href=True):
        href = _clean(a["href"], base_url)
        if not href or href == base_url:
            continue
        label = re.sub(r"\s+", " ", " ".join(filter(None, [a.get_text(" "), a.get("title") or "", a.get("aria-label") or ""]))).strip()[:160]
        cls = " ".join(a.get("class") or [])
        path_q = urlsplit(href).path + ("?" + urlsplit(href).query if urlsplit(href).query else "")
        if _BAD_LINK_TEXT.search(label) or _BAD_LINK_URL.search(path_q):
            continue
        is_file = bool(_FILE_URL.search(path_q))
        text_pdf = bool(_PDF_TEXT.search(label)) or bool(re.search(r"pdf|download", cls, re.I))
        if is_file and (text_pdf or path_q.lower().split("?")[0].endswith(".pdf")):
            add_pdf(href, 80 if path_q.lower().split("?")[0].endswith(".pdf") else 65, label, "anchor")
        elif is_file or (text_pdf and re.search(r"pdf", label, re.I)):
            add_pdf(href, 55, label, "anchor")
        elif _FOLLOW_TEXT.search(label) or re.search(r"doi\.org/|handle\.net/|/handle/|/eprint/|/record/", href):
            if not _PAYWALL_RE.search(label):
                follows.setdefault(href, LinkCandidate(href, 30 + (10 if _VERSION_TEXT.search(label) else 0), label, "anchor"))
    info.pdf_links = sorted(pdfs.values(), key=lambda l: -l.score)[:MAX_PDF_LINKS_PER_PAGE]
    info.follow_links = sorted(follows.values(), key=lambda l: -l.score)[:MAX_FOLLOW_LINKS_PER_PAGE]
    return info


# ----------------------------------------------------------------------------- crawler
class WebCrawler:
    def __init__(self, http: HttpClient, checker: AccessChecker, settings: Settings):
        self.http = http
        self.checker = checker
        self.settings = settings

    def crawl(self, seeds: list, paper: Paper, *, max_pages: Optional[int] = None,
              max_depth: Optional[int] = None, visited: Optional[set] = None, progress=None,
              prefetched: Optional[dict] = None) -> CrawlReport:
        """Visit ``seeds`` (list[FullTextLocation]) breadth-first and collect PDF candidates."""
        max_pages = self.settings.max_pages if max_pages is None else max_pages
        max_depth = self.settings.max_depth if max_depth is None else max_depth
        visited = visited if visited is not None else set()
        report = CrawlReport()
        queue: list = list(seeds)
        emit = progress or (lambda *a, **k: None)
        pages = 0
        while queue and pages < max_pages and not self.http.budget.exhausted:
            loc = queue.pop(0)
            key = loc.url.split("#")[0].rstrip("/")
            if key in visited:
                continue
            visited.add(key)
            emit("crawl", f"Visiting page (depth {loc.depth}): {loc.url}", loc.url)
            probe = (prefetched or {}).pop(loc.url, None)
            if probe is None:
                probe = self.checker.probe(loc.url)
                pages += 1
                report.fetched = pages
            if probe.status == ProbeStatus.PDF:
                report.locations.append(FullTextLocation(
                    url=probe.final_url or loc.url, kind="pdf", version=loc.version, host_type=loc.host_type,
                    provider=loc.provider or "crawler", license=loc.license, is_oa=loc.is_oa,
                    origin_url=loc.origin_url or loc.url, note="URL returned a PDF directly", depth=loc.depth,
                    trusted=loc.trusted, probed_pdf=True))
                report.attempts.append((loc.url, "pdf", "direct PDF"))
                continue
            if probe.status not in (ProbeStatus.HTML_PAGE, ProbeStatus.PAYWALL):
                report.attempts.append((loc.url, probe.status.value, probe.detail))
                continue
            info = parse_page(probe.html, probe.final_url or loc.url, paper)
            info.status, info.http_status = probe.status.value, probe.http_status
            report.pages.append(info)
            if probe.status == ProbeStatus.PAYWALL or info.paywall_signal:
                info.note = "paywall indicators on page"
            trusted = loc.trusted or info.matches_paper
            if not loc.trusted and not info.matches_paper:
                info.note = (info.note + "; " if info.note else "") + "page does not match the requested paper - links ignored"
                report.attempts.append((loc.url, "page_mismatch", f"page title '{(info.meta_title or info.title)[:80]}' does not match"))
                continue
            report.attempts.append((loc.url, probe.status.value, f"crawled; {len(info.pdf_links)} PDF link(s) found"))
            if info.meta_doi and paper.doi and info.meta_doi != normalize_doi(paper.doi):
                info.note += f" page DOI {info.meta_doi} differs from {paper.doi}"
                if info.match_score < 0.9:
                    continue
            if info.meta_abstract and not report.metadata.get("abstract") and len(info.meta_abstract) > 150:
                report.metadata["abstract"] = info.meta_abstract
            version = loc.version if loc.version != VersionType.UNKNOWN else info.version_hint
            host_type = classify_host(info.final_url or loc.url)
            for link in info.pdf_links:
                report.locations.append(FullTextLocation(
                    url=link.url, kind="pdf", version=version, host_type=loc.host_type if loc.host_type != "doi" else host_type,
                    provider=loc.provider or "crawler", license=loc.license,
                    is_oa=True if info.open_access_signal else loc.is_oa,
                    origin_url=info.final_url or loc.url, depth=loc.depth + 1,
                    note=f"found on page via {link.source}: {link.text[:60]}", trusted=trusted))
            if loc.depth < max_depth and not info.pdf_links:
                for link in info.follow_links:
                    queue.append(FullTextLocation(
                        url=link.url, kind="landing", version=version, host_type=classify_host(link.url),
                        provider=loc.provider or "crawler", origin_url=info.final_url or loc.url, depth=loc.depth + 1,
                        note=f"followed link: {link.text[:60]}", trusted=trusted))
        return report
