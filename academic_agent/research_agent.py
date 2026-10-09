"""The autonomous research workflow.

understand query -> identify the publication (cross-checking several sources) -> discover every known
full-text location -> examine them best-first (probe, crawl, download, verify) -> if nothing works,
discover alternative versions (query variants, relaxed providers, web search) -> decide.

Nothing is reported as downloaded unless a PDF was fetched, structurally validated and its content
matched the requested paper. If that is impossible the result says what was checked and why.
"""
from __future__ import annotations

import heapq
import json
import logging
import math
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import urlsplit

from access_checker import AccessChecker
from config import Settings
from database import Database, now_iso
from http_client import (BudgetExceeded, HttpClient, HttpError, RateLimited)
from models import (AccessState, Attempt, Citation, DownloadInfo, FullTextLocation, Outcome, Paper, ProbeStatus,
                    ProgressEvent, QueryType, ResearchResult, VersionType)
from net_safety import URLValidationError, host_matches
from paper_match import (MatchResult, QueryInfo, author_overlap, classify_host, classify_query, cluster_papers,
                         confirmations, content_tokens, md_escape,
                         is_preprint_doi, match_query, norm_url, normalize_doi, same_work, title_similarity,
                         topic_relevance)
from pdf_downloader import PdfDownloader
from pdf_verifier import PdfVerifier, Verdict
from regulation_finder import RegSearch, RegulationFinder
from regulation_match import RegRef
from batch import split_queries
from search_providers import (ProviderError, ProviderRegistry, SearchProvider, WebHit, WebSearchProvider, guess_kind)
from web_crawler import WebCrawler

log = logging.getLogger("research_agent")
RESTRICTED = {ProbeStatus.PAYWALL, ProbeStatus.LOGIN_REQUIRED, ProbeStatus.FORBIDDEN, ProbeStatus.CAPTCHA}


@dataclass
class _Ctx:
    qi: QueryInfo
    progress: Optional[Callable[[ProgressEvent], None]]
    refresh: bool = False
    events: list = field(default_factory=list)
    attempts: list = field(default_factory=list)
    notes: dict = field(default_factory=dict)           # provider -> outcome text
    used: list = field(default_factory=list)
    skipped: dict = field(default_factory=dict)
    paper: Optional[Paper] = None
    alternatives: list = field(default_factory=list)
    match_note: str = ""
    tried: set = field(default_factory=set)
    visited_pages: set = field(default_factory=set)
    page_summaries: list = field(default_factory=list)
    pages_fetched: int = 0
    downloads_tried: int = 0
    examined: int = 0
    restricted_seen: bool = False
    seq: int = 0
    locations_seen: dict = field(default_factory=dict)  # norm url -> FullTextLocation


# ----------------------------------------------------------------------------- ranking helpers
def effective_version(loc: FullTextLocation) -> VersionType:
    if loc.version == VersionType.UNKNOWN and loc.host_type == "preprint_server":
        return VersionType.PREPRINT
    return loc.version


def rank_location(loc: FullTextLocation) -> float:
    """Lower = tried first. Implements the authority order: official OA publisher version, institutional
    repository, trusted disciplinary / general repository, accepted manuscript, preprint, anything else."""
    v, ht = effective_version(loc), loc.host_type
    if v == VersionType.PREPRINT:
        base = 50
    elif v == VersionType.ACCEPTED:
        base = 40
    elif ht == "publisher":
        base = 10 if loc.is_oa else 45
    elif ht == "doi":
        base = 38
    elif ht == "institutional":
        base = 20
    elif ht in ("repository", "preprint_server"):
        base = 30
    else:
        base = 60
    if not loc.trusted:
        base += 5
    if loc.kind == "pdf":
        base -= 1
    return base + 0.2 * loc.depth


def resolve_version(loc: FullTextLocation, detected: VersionType) -> VersionType:
    """Provider tag unless the document itself clearly says it is an accepted manuscript / preprint."""
    tag = effective_version(loc)
    if detected == VersionType.UNKNOWN or detected == tag:
        return tag
    if tag == VersionType.UNKNOWN:
        return detected
    if detected in (VersionType.ACCEPTED, VersionType.PREPRINT) and tag == VersionType.PUBLISHED:
        return detected
    return tag


def _short_authors(authors: list, n: int = 3) -> str:
    if not authors:
        return "authors not listed"
    return ", ".join(authors[:n]) + (" et al." if len(authors) > n else "")


class ResearchAgent:
    def __init__(self, settings: Optional[Settings] = None, *, http: Optional[HttpClient] = None,
                 registry: Optional[ProviderRegistry] = None, db: Optional[Database] = None,
                 checker: Optional[AccessChecker] = None, crawler: Optional[WebCrawler] = None,
                 downloader: Optional[PdfDownloader] = None, verifier: Optional[PdfVerifier] = None):
        self.settings = settings or Settings.from_env()
        self.http = http or HttpClient(self.settings)
        self.registry = registry or ProviderRegistry.default(self.http, self.settings)
        self.db = db or Database(self.settings.db_path)
        self.checker = checker or AccessChecker(self.http)
        self.crawler = crawler or WebCrawler(self.http, self.checker, self.settings)
        self.downloader = downloader or PdfDownloader(self.http, self.checker, self.settings)
        self.verifier = verifier or PdfVerifier(self.settings.pdf_text_pages, self.settings.pdf_timeout_seconds,
                                                self.settings.min_pdf_bytes)
        self.regulations = RegulationFinder(self.http, self.checker, self.downloader, self.verifier, self.settings, self.db,
                                            web_providers=self.registry.available_web)

    # ======================================================================================
    # public API
    # ======================================================================================
    def run(self, query: str, progress: Optional[Callable[[ProgressEvent], None]] = None, *,
            refresh: bool = False) -> ResearchResult:
        self.http.reset_budget()
        qi = classify_query(query)
        ctx = _Ctx(qi=qi, progress=progress, refresh=refresh)
        run_id = self.db.start_run(query, qi.type.value)
        try:
            result = self._run(ctx)
        except BudgetExceeded as exc:
            result = self._finish(ctx, Outcome.NO_FULLTEXT if ctx.paper else Outcome.NOT_FOUND, None,
                                  error=f"search budget exhausted: {exc}")
        except Exception as exc:  # noqa: BLE001 - never crash the UI
            log.error("research run failed: %s", traceback.format_exc())
            self._emit(ctx, "error", f"Unexpected internal error: {type(exc).__name__}: {exc}", level="error")
            result = self._finish(ctx, Outcome.NO_FULLTEXT if ctx.paper else Outcome.NOT_FOUND, None,
                                  error=f"{type(exc).__name__}: {exc}")
        result.run_id = run_id
        paper_id = None
        try:
            if result.paper and result.outcome in (Outcome.DOWNLOADED, Outcome.LINK_FOUND, Outcome.NO_FULLTEXT):
                paper_id = self.db.upsert_paper(result.paper)
            self.db.record_attempts(run_id, paper_id, result.attempts)
            self.db.finish_run(run_id, outcome=result.outcome.value, access_state=result.access_state.value,
                               paper_id=paper_id, summary=result.summary, result=result.to_dict())
        except Exception:  # noqa: BLE001
            log.error("could not persist run: %s", traceback.format_exc())
        return result

    def run_batch(self, queries, progress: Optional[Callable[[ProgressEvent], None]] = None, *, refresh: bool = False) -> list[ResearchResult]:
        """Research several papers and/or regulations in one call (sequentially, each with its own budget).

        ``queries`` is a list of strings or one multi-line string (one item per line).
        """
        items = split_queries(queries, self.settings.max_batch) if isinstance(queries, str) else list(queries)[: self.settings.max_batch]
        results: list[ResearchResult] = []
        for i, q in enumerate(items, 1):
            def wrapped(ev: ProgressEvent, i=i) -> None:
                if progress:
                    progress(ProgressEvent(ev.stage, f"[{i}/{len(items)}] {ev.message}", ev.url, ev.level))
            if progress:
                progress(ProgressEvent("batch", f"[{i}/{len(items)}] Starting: {q[:120]}"))
            results.append(self.run(q, progress=wrapped, refresh=refresh))
        return results

    # ======================================================================================
    # workflow
    # ======================================================================================
    def _run(self, ctx: _Ctx) -> ResearchResult:
        qi = ctx.qi
        self._emit(ctx, "understand", f"Query understood as: {qi.type.value}"
                   + (f" - DOI {qi.doi}" if qi.doi else f" - arXiv:{qi.arxiv_id}" if qi.arxiv_id else
                      f" - \"{qi.title}\"" + (f" ({qi.author.title()}, {qi.year})" if qi.author else "")))
        if qi.type == QueryType.REGULATION:
            return self._run_regulation(ctx)
        # ---- step 2: identify ----------------------------------------------------------
        outcome = self._identify(ctx)
        if outcome is not None:
            return self._finish(ctx, outcome, None)
        paper = ctx.paper
        assert paper is not None
        self._emit(ctx, "identify", f"Identified: {paper.title} ({paper.year or 'n.d.'}) - {_short_authors(paper.authors)}",
                   paper.url, "success")
        self._enrich(ctx)

        # ---- already in the local library? --------------------------------------------
        if not ctx.refresh:
            have = self.db.find_download_for_paper(paper.doi, paper.title, paper.year)
            if have:
                self._emit(ctx, "decide", f"A verified copy is already stored locally: {have['file_path']}", level="success")
                info = DownloadInfo(path=have["file_path"], sha256=have["sha256"] or "", size_bytes=have["size_bytes"] or 0,
                                    source_url=have["source_url"] or "", landing_url=have["landing_url"] or "",
                                    provider=have["provider"] or "", version=VersionType(have["version_type"] or "unknown"),
                                    verification=json.loads(have["verification_json"] or "{}"),
                                    access_date=have["access_date"], already_had=True)
                return self._finish(ctx, Outcome.DOWNLOADED, info)

        # ---- step 3/4: discover and examine full-text locations ---------------------------
        self._discover_locations(ctx)
        heap: list = []
        self._push(ctx, heap, list(paper.locations))
        self._emit(ctx, "discover", f"{len(heap)} candidate full-text location(s) found; examining them best source first")
        info = self._attempt_loop(ctx, heap)
        if info is None and not self.http.budget.exhausted:
            self._discover_alternatives(ctx, heap)
            info = self._attempt_loop(ctx, heap)
        return self._finish(ctx, self._outcome_for(info), info)

    @staticmethod
    def _outcome_for(info: Optional[DownloadInfo]) -> Outcome:
        if info is None:
            return Outcome.NO_FULLTEXT
        return Outcome.DOWNLOADED if info.saved else Outcome.LINK_FOUND

    # ---------------------------------------------------------------------------------------
    # regulations (SEOJK, POJK, UU, PP ...)
    # ---------------------------------------------------------------------------------------
    def _run_regulation(self, ctx: _Ctx) -> ResearchResult:
        ref = ctx.qi.regulation
        assert ref is not None
        res = self.regulations.find(ref, lambda stage, msg, url="", level="info": self._emit(ctx, stage, msg, url, level), ctx.qi.raw)
        ctx.paper, ctx.used, ctx.notes = res.paper, res.used, res.notes
        ctx.attempts.extend(res.attempts)
        ctx.restricted_seen = res.restricted_seen
        ctx.examined = len(res.attempts)
        return self._finish_regulation(ctx, res, ref)

    def _finish_regulation(self, ctx: _Ctx, res: RegSearch, ref: RegRef) -> ResearchResult:
        info = res.info
        if info:
            state = AccessState.OFFICIAL_REGULATION
        elif res.candidates:
            state = AccessState.INACCESSIBLE
        else:
            state = AccessState.UNKNOWN
        official = (info.landing_url if info else "") or (res.candidates[0].page_url if res.candidates else "") or \
            (res.manual_links[0][1] if res.manual_links else "")
        r = ResearchResult(query=ctx.qi.raw, query_type=ctx.qi.type, outcome=res.outcome, access_state=state, paper=res.paper,
                           download=info, official_link=official, attempts=ctx.attempts, events=ctx.events, providers_used=ctx.used,
                           other_versions=res.related, match_note=f"Regulation reference parsed as {ref.canonical}.",
                           requests_used=self.http.budget.used, elapsed_seconds=self.http.budget.elapsed)
        cites: list[Citation] = []

        def cite(title: str, url: str, note: str = "") -> int:
            for c in cites:
                if c.url == url:
                    return c.label
            cites.append(Citation(len(cites) + 1, title, url, note))
            return len(cites)

        parts = [f"**Regulation.** {md_escape(ref.canonical)} — issuer: {md_escape(ref.issuer)}."
                 + (f" Title: *{md_escape(res.paper.title)}*." if res.paper else "")]
        sources = ", ".join(res.used) or "none"
        failed = [f"{n} ({why})" for n, why in res.notes.items() if not why.startswith("ok")]
        parts.append(f"**Sources checked.** {sources}." + (f" Not used / failed: {'; '.join(failed)}." if failed else "")
                     + f" {len(res.attempts)} PDF candidate(s) examined; {r.requests_used} HTTP requests in {r.elapsed_seconds:.0f}s.")
        if info:
            n = cite("Official PDF", info.source_url, "verified PDF of the regulation")
            if info.landing_url:
                cite("Official regulation page", info.landing_url, "issuing authority")
            v = info.verification or {}
            parts.append(f"**Result.** Found the official PDF [{n}] on {urlsplit(info.source_url).hostname}. It was opened and checked: "
                         f"{'; '.join(v.get('reasons', [])) or 'ok'}."
                         + (f" Saved to `{info.path}`." if info.saved else " Nothing was saved to disk (auto-download is off)."))
        elif res.outcome == Outcome.NO_FULLTEXT:
            link = cite("Official regulation page", res.candidates[0].page_url) if res.candidates else 0
            parts.append("**Result.** The regulation's official page was found" + (f" [{link}]" if link else "")
                         + ", but no PDF could be verified as this regulation. See the attempts below.")
        else:
            parts.append("**Result.** The regulation was not found in the sources that could be searched automatically. "
                         "Try these official search pages manually (they are search pages, not verified PDFs):")
            for label, url in res.manual_links:
                cite(label, url, "manual search page - not a verified PDF")
        if not info and res.outcome == Outcome.NO_FULLTEXT:
            for label, url in res.manual_links[:2]:
                cite(label, url, "manual search page - not a verified PDF")
        r.summary, r.citations = "\n\n".join(parts), cites
        return r

    # ---------------------------------------------------------------------------------------
    # identification
    # ---------------------------------------------------------------------------------------
    def _identify(self, ctx: _Ctx) -> Optional[Outcome]:
        qi, s = ctx.qi, self.settings
        selected, skipped = self.registry.select(qi)
        doi_keyed = {p.name for p in self.registry.providers if p.supports_doi_lookup or p.supports_fulltext_lookup}
        ctx.skipped = {k: v for k, v in skipped.items() if not (k in doi_keyed and "not relevant" in v)}
        if qi.type == QueryType.DOI:
            jobs = [(p.name, lambda p=p: p.lookup_doi(qi.doi)) for p in self.registry.providers
                    if p.supports_doi_lookup and p.available()[0]]
            self._emit(ctx, "providers", "Resolving the DOI with: " + ", ".join(n for n, _ in jobs))
            records = [r for _, r in self._collect(ctx, jobs, single=True)]
        elif qi.type == QueryType.ARXIV:
            jobs = [(p.name, lambda p=p: p.lookup_arxiv(qi.arxiv_id)) for p in self.registry.providers
                    if p.name in ("arxiv", "semantic_scholar") and p.available()[0]]
            self._emit(ctx, "providers", "Looking up the arXiv identifier with: " + ", ".join(n for n, _ in jobs))
            records = [r for _, r in self._collect(ctx, jobs, single=True)]
        else:
            text = qi.title
            jobs = [(p.name, lambda p=p: p.search(text, s.results_per_provider)) for p in selected]
            self._emit(ctx, "providers", f"Searching {len(jobs)} sources in parallel: " + ", ".join(n for n, _ in jobs))
            records = [r for _, r in self._collect(ctx, jobs)]
        for name, why in ctx.skipped.items():
            self._emit(ctx, "providers", f"Skipped {name}: {why}", level="warning" if "set " in why else "info")
        clusters = cluster_papers(records)
        if not clusters:
            self._emit(ctx, "identify", "No source returned a matching bibliographic record", level="warning")
            ctx.match_note = "No bibliographic record found in any queried source."
            return Outcome.NOT_FOUND

        matches: dict[int, MatchResult] = {}
        for c in clusters:
            matches[id(c)] = match_query(qi, c, s.exact_match_threshold, s.possible_match_threshold)
            c.score = matches[id(c)].score
        if qi.type in (QueryType.DOI, QueryType.ARXIV):
            best = max(clusters, key=lambda c: (c.score, len(c.sources)))
            if best.score < 1.0:
                ctx.match_note = "Identifier not confirmed by any record."
                ctx.alternatives = clusters[:5]
                return Outcome.NOT_FOUND
            ctx.paper = best
            ctx.match_note = f"Identifier matches exactly; confirmed by {len(best.sources)} source(s)."
            return None
        ranked = sorted(clusters, key=lambda c: (-c.score, -len(c.sources)))
        exact = [c for c in ranked if matches[id(c)].verdict == "exact"]
        if len(exact) == 1 or (len(exact) > 1 and exact[0].score - exact[1].score >= 0.04
                               and not self._distinct_namesakes(exact[0], exact[1])):
            ctx.paper = exact[0]
            n = len(exact[0].sources)
            ctx.match_note = (f"Title match {exact[0].score:.2f} ({'; '.join(matches[id(exact[0])].reasons)}); "
                              f"record confirmed by {n} source{'s' if n != 1 else ''}.")
            return None
        if len(exact) > 1:
            ctx.alternatives = exact[:6]
            ctx.match_note = "Several different publications share this title."
            self._emit(ctx, "identify", "Several different publications have this exact title - please choose", level="warning")
            return Outcome.NEEDS_CHOICE
        if qi.type in (QueryType.TOPIC, QueryType.AUTHOR_YEAR):
            return self._topic_outcome(ctx, clusters, "Research-topic query: ranked candidate papers are listed for you to choose.")
        close = [c for c in ranked if c.score >= s.possible_match_threshold]
        if close:
            ctx.alternatives = close[:6]
            ctx.match_note = (f"No exact title match; closest is {close[0].score:.2f} similar. "
                              "Similar titles are never downloaded automatically.")
            self._emit(ctx, "identify", "No record has exactly this title; similar titles found - please choose", level="warning")
            return Outcome.NEEDS_CHOICE
        if any(topic_relevance(qi, c) >= 0.6 for c in clusters):         # keyword phrase rather than a title
            return self._topic_outcome(ctx, clusters, "No record has this exact title; the most relevant papers are ranked below.")
        ctx.alternatives = ranked[:5]
        ctx.match_note = f"No record is similar enough to the requested title (best score {ranked[0].score:.2f})."
        self._emit(ctx, "identify", ctx.match_note, level="warning")
        return Outcome.NOT_FOUND

    def _topic_outcome(self, ctx: _Ctx, clusters: list, note: str) -> Outcome:
        qi = ctx.qi
        ordered = sorted(clusters, key=lambda c: -self._topic_score(c, qi))
        for c in ordered:
            c.score = round(self._topic_score(c, qi), 4)
        if self.settings.auto_pick_topic and ordered and ordered[0].score >= 0.3:
            ctx.paper = ordered[0]
            ctx.match_note = "Chosen automatically as the most relevant result for a topic query."
            return None                                                   # type: ignore[return-value]
        ctx.alternatives = ordered[:8]
        ctx.match_note = note
        self._emit(ctx, "identify", f"{len(ctx.alternatives)} relevant papers ranked - choose one to retrieve")
        return Outcome.NEEDS_CHOICE

    @staticmethod
    def _distinct_namesakes(a: Paper, b: Paper) -> bool:
        ov = author_overlap(a.authors, b.authors)
        return ov is not None and ov < 0.3

    @staticmethod
    def _topic_score(c: Paper, qi: QueryInfo) -> float:
        rel = topic_relevance(qi, c)
        rrf = sum(1.0 / (60 + s.get("rank", 0)) for s in c.provenance) / (len(c.provenance) / 60.0 or 1.0) if c.provenance else 0.0
        rrf = min(rrf * (min(len(c.sources), 5) / 5.0), 1.0)
        cit = min(math.log10(1 + (c.citation_count or 0)) / 3.0, 1.0)
        bonus = 0.0
        if qi.author and (author_overlap([qi.author], c.authors) or 0) > 0:
            bonus += 0.3
        if qi.year and c.year and abs(qi.year - c.year) <= 1:
            bonus += 0.1
        return 0.55 * rel + 0.2 * rrf + 0.2 * cit + (0.05 if c.is_oa else 0.0) + bonus

    # ---------------------------------------------------------------------------------------
    # enrichment / location discovery
    # ---------------------------------------------------------------------------------------
    def _enrich(self, ctx: _Ctx) -> None:
        """Cross-check the identified paper against the DOI-keyed sources it has not come from yet."""
        paper = ctx.paper
        assert paper is not None
        if not paper.doi:
            return
        jobs = [(p.name, lambda p=p: p.lookup_doi(paper.doi)) for p in self.registry.providers
                if p.supports_doi_lookup and p.available()[0] and p.name not in paper.sources]
        if not jobs:
            return
        self._emit(ctx, "verify-metadata", "Cross-checking the record against: " + ", ".join(n for n, _ in jobs))
        extra = [r for _, r in self._collect(ctx, jobs, single=True) if same_work(paper, r)]
        if extra:
            merged = cluster_papers([paper, *extra])[0]
            merged.score = paper.score
            ctx.paper = merged
        c = confirmations(ctx.paper)
        self._emit(ctx, "verify-metadata",
                   f"Metadata confirmed by {len(c['providers'])} source(s): {', '.join(c['providers'])}", level="success")

    def _discover_locations(self, ctx: _Ctx) -> None:
        paper = ctx.paper
        assert paper is not None
        jobs = [(p.name, lambda p=p: p.find_fulltext(paper)) for p in self.registry.providers
                if p.supports_fulltext_lookup and p.available()[0]]
        if jobs:
            self._emit(ctx, "discover", "Looking for open-access copies via: " + ", ".join(n for n, _ in jobs))
            for name, locs in self._collect(ctx, jobs, single=True, many=True):
                known = {norm_url(l.url) for l in paper.locations}
                for l in locs:
                    if norm_url(l.url) not in known:
                        known.add(norm_url(l.url))
                        paper.locations.append(l)
        if paper.arxiv_id:
            paper.locations.append(FullTextLocation(
                url=f"https://arxiv.org/pdf/{paper.arxiv_id}", kind="pdf", version=VersionType.PREPRINT,
                host_type="preprint_server", provider="arxiv", is_oa=True, repository="arXiv",
                note="arXiv identifier listed in the record"))
        for rel in paper.related_dois[:3]:
            if rel.startswith("10.48550/arxiv."):
                paper.locations.append(FullTextLocation(
                    url=f"https://arxiv.org/pdf/{rel.split('arxiv.', 1)[1]}", kind="pdf", version=VersionType.PREPRINT,
                    host_type="preprint_server", provider="crossref-relation", is_oa=True,
                    note=f"preprint counterpart {rel}"))
            else:
                paper.locations.append(FullTextLocation(
                    url=f"https://doi.org/{rel}", kind="landing",
                    version=VersionType.PREPRINT if is_preprint_doi(rel) else VersionType.UNKNOWN,
                    host_type="doi", provider="crossref-relation", note=f"related version {rel}"))
        if paper.doi:
            paper.locations.append(FullTextLocation(url=f"https://doi.org/{paper.doi}", kind="landing",
                                                    version=VersionType.PUBLISHED, host_type="doi", provider="doi",
                                                    note="publisher page reached through the DOI"))

    # ---------------------------------------------------------------------------------------
    # candidate examination
    # ---------------------------------------------------------------------------------------
    def _push(self, ctx: _Ctx, heap: list, locs) -> int:
        n = 0
        for loc in locs:
            key = norm_url(loc.url)
            if key in ctx.tried or key in ctx.locations_seen:
                continue
            if self.http.is_no_crawl(loc.url) or host_matches(urlsplit(loc.url).hostname or "", self.settings.blocked_domains):
                ctx.locations_seen[key] = loc
                self._attempt(ctx, loc.url, ProbeStatus.ROBOTS_BLOCKED.value,
                              "site does not permit automated access (terms of use / unauthorised source); link kept for you to open manually",
                              loc, "probe")
                continue
            ctx.locations_seen[key] = loc
            ctx.seq += 1
            heapq.heappush(heap, (rank_location(loc), ctx.seq, loc))
            n += 1
        return n

    def _attempt(self, ctx: _Ctx, url: str, status: str, detail: str, loc: Optional[FullTextLocation] = None,
                 stage: str = "", http_status: Optional[int] = None) -> None:
        ctx.attempts.append(Attempt(url=url, status=status, detail=detail[:300], provider=loc.provider if loc else "",
                                    stage=stage, version=effective_version(loc).value if loc else "", http_status=http_status))

    def _out_of_budget(self, ctx: _Ctx) -> bool:
        return (self.http.budget.exhausted or ctx.examined >= self.settings.max_candidates
                or ctx.downloads_tried >= self.settings.max_download_attempts)

    def _attempt_loop(self, ctx: _Ctx, heap: list) -> Optional[DownloadInfo]:
        s = self.settings
        while heap and not self._out_of_budget(ctx):
            _, _, loc = heapq.heappop(heap)
            key = norm_url(loc.url)
            if key in ctx.tried:
                continue
            ctx.tried.add(key)
            ctx.examined += 1
            ver = effective_version(loc)
            if not s.allow_preprints and ver == VersionType.PREPRINT:
                self._attempt(ctx, loc.url, "skipped", "preprints are disabled in settings", loc, "probe")
                continue
            if loc.kind == "landing":
                self._emit(ctx, "crawl", f"Opening {loc.host_type} page to look for the full text: {loc.url}", loc.url)
                self._crawl(ctx, heap, [loc])
                continue
            probe = None
            if not loc.probed_pdf:
                self._emit(ctx, "probe", f"Checking access ({ver.label}, {loc.provider or 'web'}): {loc.url}", loc.url)
                probe = self.checker.probe(loc.url, referer=loc.origin_url or None)
                if probe.status in (ProbeStatus.HTML_PAGE, ProbeStatus.PAYWALL):
                    self._attempt(ctx, loc.url, probe.status.value,
                                  "link did not return a PDF but an HTML page" + (f" - {probe.detail}" if probe.status == ProbeStatus.PAYWALL else "; inspecting it for the real file link"),
                                  loc, "probe", probe.http_status)
                    if probe.status == ProbeStatus.PAYWALL:
                        ctx.restricted_seen = True
                    self._crawl(ctx, heap, [loc], prefetched={loc.url: probe})
                    continue
                if probe.status != ProbeStatus.PDF:
                    if probe.status in RESTRICTED:
                        ctx.restricted_seen = True
                    self._attempt(ctx, loc.url, probe.status.value, probe.detail, loc, "probe", probe.http_status)
                    self._emit(ctx, "probe", f"Not usable ({probe.status.value}): {probe.detail}", loc.url, "warning")
                    continue
            ctx.downloads_tried += 1
            info = self._download_and_verify(ctx, loc)
            if info:
                return info
        return None

    def _crawl(self, ctx: _Ctx, heap: list, seeds: list, prefetched: Optional[dict] = None) -> None:
        remaining = self.settings.max_pages - ctx.pages_fetched
        if remaining <= 0:
            for loc in seeds:
                self._attempt(ctx, loc.url, "skipped", "page budget exhausted", loc, "crawl")
            return
        rep = self.crawler.crawl(seeds, ctx.paper, max_pages=remaining, visited=ctx.visited_pages, prefetched=prefetched,
                                 progress=lambda stage, msg, url="": self._emit(ctx, stage, msg, url))
        ctx.pages_fetched += rep.fetched
        for url, status, detail in rep.attempts:
            if status in {x.value for x in RESTRICTED}:
                ctx.restricted_seen = True
            self._attempt(ctx, url, status, detail, seeds[0], "crawl")
        ctx.page_summaries.extend(p.summary() for p in rep.pages)
        if rep.metadata.get("abstract") and ctx.paper and not ctx.paper.abstract:
            ctx.paper.abstract = rep.metadata["abstract"]
        added = self._push(ctx, heap, rep.locations)
        if added:
            self._emit(ctx, "crawl", f"{added} PDF link(s) discovered on the visited pages", level="success")

    def _download_and_verify(self, ctx: _Ctx, loc: FullTextLocation) -> Optional[DownloadInfo]:
        paper = ctx.paper
        assert paper is not None
        self._emit(ctx, "download", f"Downloading: {loc.url}", loc.url)
        dl = self.downloader.download(loc.url, referer=loc.origin_url or None)
        if not dl.ok:
            if dl.status in RESTRICTED:
                ctx.restricted_seen = True
            self._attempt(ctx, loc.url, dl.status.value, dl.detail, loc, "download", dl.http_status)
            self._emit(ctx, "download", f"Download failed ({dl.status.value}): {dl.detail}", loc.url, "warning")
            return None
        self._emit(ctx, "verify-pdf", f"Downloaded {dl.size / 1024:.0f} KB - verifying that it is the requested paper")
        report = self.verifier.verify(dl.path, paper)
        accepted = report.verdict == Verdict.VERIFIED or (report.verdict == Verdict.PARTIAL and self.settings.accept_partial_verification)
        if not accepted:
            self.downloader.discard(dl)
            self._attempt(ctx, loc.url, ProbeStatus.REJECTED.value,
                          f"{report.verdict.value}: " + "; ".join(report.reasons), loc, "verify")
            self._emit(ctx, "verify-pdf", f"Rejected ({report.verdict.value}): {'; '.join(report.reasons)}", loc.url, "warning")
            return None
        version = resolve_version(loc, report.detected_version)
        if not self.settings.allow_preprints and version == VersionType.PREPRINT:
            self.downloader.discard(dl)
            self._attempt(ctx, loc.url, "skipped", "document is a preprint and preprints are disabled", loc, "verify")
            return None
        if not self.settings.auto_download:        # link-only mode: the file was only needed to verify the content
            self.downloader.discard(dl)
            paper_id = self.db.upsert_paper(paper)
            verification = report.to_dict()
            self.db.record_download(paper_id, file_path="", sha256=dl.sha256, size_bytes=dl.size, version_type=version.value,
                                    source_url=dl.final_url or loc.url, landing_url=loc.origin_url or paper.doi_url or paper.url,
                                    provider=loc.provider, verification=verification, query=ctx.qi.raw, status="link_only")
            self._attempt(ctx, loc.url, "verified_link", "verified: " + "; ".join(report.reasons), loc, "verify")
            self._emit(ctx, "verify-pdf", f"Verified PDF link ({report.verdict.value}): {'; '.join(report.reasons)}", loc.url, "success")
            return DownloadInfo(path="", sha256=dl.sha256, size_bytes=dl.size, source_url=dl.final_url or loc.url,
                                landing_url=loc.origin_url or paper.doi_url or paper.url, provider=loc.provider, version=version,
                                verification=verification, access_date=now_iso(), saved=False)
        dup = self.db.find_download_by_hash(dl.sha256)
        if dup:
            self.downloader.discard(dl)
            path, already = Path(dup["file_path"]), True
        else:
            path, already = self.downloader.finalize(dl, paper, version), False
        paper_id = self.db.upsert_paper(paper)
        verification = report.to_dict()
        self.db.record_download(paper_id, file_path=str(path), sha256=dl.sha256, size_bytes=dl.size, version_type=version.value,
                                source_url=dl.final_url or loc.url, landing_url=loc.origin_url or paper.doi_url or paper.url,
                                provider=loc.provider, verification=verification, query=ctx.qi.raw)
        self._attempt(ctx, loc.url, "downloaded", "verified: " + "; ".join(report.reasons), loc, "verify")
        self._emit(ctx, "verify-pdf", f"Verified ({report.verdict.value}): {'; '.join(report.reasons)}", loc.url, "success")
        return DownloadInfo(path=str(path), sha256=dl.sha256, size_bytes=dl.size, source_url=dl.final_url or loc.url,
                            landing_url=loc.origin_url or paper.doi_url or paper.url, provider=loc.provider, version=version,
                            verification=verification, access_date=now_iso(), already_had=already)

    # ---------------------------------------------------------------------------------------
    # step 3: alternative-version discovery
    # ---------------------------------------------------------------------------------------
    def query_variants(self, paper: Paper) -> list[str]:
        t, first = paper.title.strip(), paper.first_author_surname.title()
        variants = [f'"{t}"']
        if first:
            variants.append(f'"{t}" {first}')
        if paper.doi:
            variants.append(paper.doi)
        variants += [f'"{t}" PDF', f'"{t}" open access', f'"{t}" repository', f'"{t}" accepted manuscript', f'"{t}" full text']
        return list(dict.fromkeys(variants))[: self.settings.max_search_queries]

    def _discover_alternatives(self, ctx: _Ctx, heap: list) -> None:
        paper = ctx.paper
        assert paper is not None
        self._emit(ctx, "discover", "No usable full text yet - searching for alternative versions "
                   "(repositories, preprints, author manuscripts)")
        first = paper.first_author_surname.title()
        text = f"{paper.title} {first}".strip()
        providers, _ = self.registry.select(ctx.qi, include_low_relevance=True)
        providers = [p for p in providers if p.available()[0]]
        jobs = [(p.name, lambda p=p: p.search(text, 6)) for p in providers]
        added = 0
        if jobs:
            self._emit(ctx, "discover", f"Re-querying {len(jobs)} sources with the title + first author: " + ", ".join(n for n, _ in jobs))
            for name, recs in self._collect(ctx, jobs, many=True):
                for r in recs:
                    if same_work(paper, r) or title_similarity(paper.title, r.title) >= 0.95:
                        added += self._push(ctx, heap, r.locations)
                        if r.arxiv_id and not paper.arxiv_id:
                            paper.arxiv_id = r.arxiv_id
        web = self.registry.available_web()
        if not web:
            self._emit(ctx, "discover", "No web-search API is configured (BRAVE_API_KEY or GOOGLE_CSE_*): open-web search for "
                       "repository copies was skipped", level="warning")
        queries = self.query_variants(paper)
        done = 0
        for q in queries:
            if not web or self.http.budget.exhausted or done >= self.settings.max_search_queries:
                break
            for wp in web:
                done += 1
                self._emit(ctx, "discover", f"Web search ({wp.label}): {q}")
                try:
                    hits = wp.search_web(q, 8)
                except (HttpError, ProviderError, URLValidationError) as exc:
                    self._emit(ctx, "discover", f"{wp.label} unavailable: {exc}", level="warning")
                    ctx.notes[wp.name] = str(exc)
                    continue
                ctx.notes[wp.name] = "ok"
                if wp.name not in ctx.used:
                    ctx.used.append(wp.name)
                locs = [self._hit_to_location(h, paper) for h in hits]
                added += self._push(ctx, heap, [l for l in locs if l])
        self._emit(ctx, "discover", f"{added} additional candidate location(s) found", level="success" if added else "info")

    def _hit_to_location(self, hit: WebHit, paper: Paper) -> Optional[FullTextLocation]:
        url = hit.url
        if not url or self.http.is_no_crawl(url):
            return None
        blob = f"{hit.title} {hit.snippet}"
        toks = set(content_tokens(paper.title))
        cov = len(toks & set(content_tokens(blob))) / len(toks) if toks else 0
        doi_hit = bool(paper.doi and paper.doi in (url + blob).lower())
        if not (title_similarity(paper.title, hit.title) >= 0.6 or cov >= 0.7 or doi_hit):
            return None
        return FullTextLocation(url=url, kind=guess_kind(url), version=VersionType.UNKNOWN, host_type=classify_host(url),
                                provider=hit.provider, trusted=False, note=f"web search hit: {hit.title[:80]}")

    # ---------------------------------------------------------------------------------------
    # helpers
    # ---------------------------------------------------------------------------------------
    def _emit(self, ctx: _Ctx, stage: str, message: str, url: str = "", level: str = "info") -> None:
        ev = ProgressEvent(stage=stage, message=message, url=url, level=level)
        ctx.events.append(ev)
        if ctx.progress:
            try:
                ctx.progress(ev)
            except Exception:  # noqa: BLE001 - UI problems must not stop the research
                log.debug("progress callback failed", exc_info=True)

    @staticmethod
    def _call(fn: Callable) -> tuple[Any, Optional[str]]:
        try:
            return fn(), None
        except RateLimited as exc:
            return None, f"rate limited / quota exceeded ({exc})"
        except BudgetExceeded as exc:
            return None, f"search budget exhausted ({exc})"
        except (HttpError, ProviderError, URLValidationError) as exc:
            return None, str(exc)
        except Exception as exc:  # noqa: BLE001 - one broken provider must not stop the others
            log.debug("provider error: %s", traceback.format_exc())
            return None, f"unexpected {type(exc).__name__}: {exc}"

    def _collect(self, ctx: _Ctx, jobs: list, *, single: bool = False, many: bool = False) -> list:
        """Run provider calls in parallel. Returns flattened [(provider, record)] (or (provider, list) if many)."""
        if not jobs:
            return []
        with ThreadPoolExecutor(max_workers=max(1, min(len(jobs), self.settings.max_workers))) as ex:
            futures = [(name, ex.submit(self._call, fn)) for name, fn in jobs]
            results = [(name, *f.result()) for name, f in futures]
        out: list = []
        for name, value, err in results:
            if err:
                ctx.notes[name] = err
                self._emit(ctx, "providers", f"{name}: {err} - continuing with the other sources", level="warning")
                continue
            if value is None or value == []:
                ctx.notes.setdefault(name, "no matching record")
                self._emit(ctx, "providers", f"{name}: no matching record")
                continue
            if name not in ctx.used:
                ctx.used.append(name)
            if many:
                out.append((name, value))
                ctx.notes[name] = f"ok ({len(value)})"
            elif single and not isinstance(value, list):
                out.append((name, value))
                ctx.notes[name] = "ok"
                self._emit(ctx, "providers", f"{name}: record found")
            else:
                recs = value if isinstance(value, list) else [value]
                out.extend((name, r) for r in recs)
                ctx.notes[name] = f"ok ({len(recs)} record{'s' if len(recs) != 1 else ''})"
                self._emit(ctx, "providers", f"{name}: {len(recs)} record(s)")
        return out

    # ---------------------------------------------------------------------------------------
    # result assembly
    # ---------------------------------------------------------------------------------------
    def _finish(self, ctx: _Ctx, outcome: Outcome, info: Optional[DownloadInfo], error: str = "") -> ResearchResult:
        paper = ctx.paper
        state = AccessState.UNKNOWN
        if outcome in (Outcome.DOWNLOADED, Outcome.LINK_FOUND) and info:
            state = {VersionType.PUBLISHED: AccessState.PUBLISHED, VersionType.ACCEPTED: AccessState.ACCEPTED,
                     VersionType.PREPRINT: AccessState.PREPRINT}.get(info.version, AccessState.FULLTEXT_UNKNOWN_VERSION)
        elif outcome == Outcome.NO_FULLTEXT and paper:
            state = AccessState.INACCESSIBLE if (ctx.restricted_seen or not paper.abstract) else AccessState.ABSTRACT_ONLY
        official = ""
        if paper:
            official = paper.doi_url or paper.url or next(iter(paper.record_urls.values()), "")
        result = ResearchResult(
            query=ctx.qi.raw, query_type=ctx.qi.type, outcome=outcome, access_state=state, paper=paper,
            alternatives=ctx.alternatives, download=info, official_link=official, attempts=ctx.attempts,
            events=ctx.events, providers_used=ctx.used, providers_skipped=ctx.skipped, pages_visited=ctx.page_summaries,
            match_note=ctx.match_note, requests_used=self.http.budget.used, elapsed_seconds=self.http.budget.elapsed, error=error)
        used_urls = {info.source_url} if info else set()
        result.other_versions = [l for k, l in ctx.locations_seen.items() if l.url not in used_urls]
        self._build_summary(ctx, result)
        return result

    def _build_summary(self, ctx: _Ctx, r: ResearchResult) -> None:
        cites: list[Citation] = []

        def cite(title: str, url: str, note: str = "") -> int:
            for c in cites:
                if c.url == url:
                    return c.label
            cites.append(Citation(len(cites) + 1, title, url, note))
            return len(cites)

        p, info, parts = r.paper, r.download, []
        if p:
            refs = []
            if p.doi_url:
                refs.append(cite("Publisher / DOI landing page", p.doi_url, "official record"))
            names = {"crossref": "Crossref", "openalex": "OpenAlex", "semantic_scholar": "Semantic Scholar",
                     "europepmc": "Europe PMC", "doaj": "DOAJ", "arxiv": "arXiv", "core": "CORE", "openaire": "OpenAIRE",
                     "unpaywall": "Unpaywall", "datacite/doi.org": "doi.org"}
            for prov, url in p.record_urls.items():
                refs.append(cite(f"{names.get(prov, prov)} record", url, "bibliographic metadata"))
            conf = confirmations(p)
            parts.append(f"**Publication.** *{md_escape(p.title)}* — {md_escape(_short_authors(p.authors))}"
                         f"{f' ({p.year})' if p.year else ''}{f', {md_escape(p.venue)}' if p.venue else ''}"
                         f"{f'. DOI {md_escape(p.doi)}' if p.doi else ''} " + "".join(f"[{n}]" for n in list(dict.fromkeys(refs))[:4]) + ". "
                         f"{r.match_note} Metadata agreed across {len(conf['providers'])} source(s): {', '.join(conf['providers']) or 'n/a'}.")
        elif r.match_note:
            parts.append(f"**Publication.** {r.match_note}")
        used = ", ".join(r.providers_used) or "none"
        failed = [f"{n} ({why})" for n, why in ctx.notes.items() if not (why.startswith("ok") or why == "no matching record")]
        parts.append(f"**Sources checked.** Queried: {used}." + (f" Unavailable or failed: {'; '.join(failed)}." if failed else "")
                     + (f" Not used: {', '.join(f'{k} ({v})' for k, v in r.providers_skipped.items())}." if r.providers_skipped else "")
                     + f" Examined {ctx.examined} candidate location(s), {ctx.pages_fetched} page(s), {ctx.downloads_tried} download(s); "
                       f"{r.requests_used} HTTP requests in {r.elapsed_seconds:.0f}s.")
        if r.outcome == Outcome.LINK_FOUND and info:
            n = cite("Verified PDF link", info.source_url, f"{info.version.label} via {info.provider}")
            if info.landing_url and info.landing_url != info.source_url:
                cite("Page where the PDF was found", info.landing_url, "landing / record page")
            v = info.verification or {}
            parts.append(f"**Result.** Found a working, legal PDF link for the **{info.version.label.lower()}** on {urlsplit(info.source_url).hostname} [{n}] "
                         f"(via {info.provider or 'web'}). It was opened and checked, and its content matches the paper: "
                         f"{'; '.join(v.get('reasons', [])) or 'ok'}. Nothing was saved to disk (auto-download is off)."
                         + (" This is **not** the final published article." if info.version in (VersionType.PREPRINT, VersionType.ACCEPTED) else ""))
        elif r.outcome == Outcome.DOWNLOADED and info:
            n = cite("Downloaded file source", info.source_url, f"{info.version.label} via {info.provider}")
            if info.landing_url and info.landing_url != info.source_url:
                cite("Page where the file was found", info.landing_url, "landing / record page")
            v = info.verification or {}
            verdict = "already stored in your library and re-verified earlier" if info.already_had else "downloaded and verified"
            parts.append(f"**Result.** The **{info.version.label.lower()}** was {verdict} from {urlsplit(info.source_url).hostname} [{n}] "
                         f"(found via {info.provider or 'web'}). Verification: {'; '.join(v.get('reasons', [])) or 'ok'}. "
                         f"Saved to `{info.path}`."
                         + (" This is **not** the final published article; cite the published version when available."
                            if info.version in (VersionType.PREPRINT, VersionType.ACCEPTED) else ""))
        elif r.outcome == Outcome.NO_FULLTEXT:
            counts: dict[str, int] = {}
            for a in r.attempts:
                counts[a.status] = counts.get(a.status, 0) + 1
            tally = ", ".join(f"{v}× {k.replace('_', ' ')}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])) or "no candidate locations"
            link = cite("Official publisher page (access may require subscription)", r.official_link) if r.official_link else 0
            parts.append("**Result.** No legally accessible full text could be retrieved and verified. "
                         f"What happened to the {len(r.attempts)} attempt(s): {tally}. "
                         + (f"The strongest verified record is available at the official page [{link}]; you may be able to access it through "
                            "your institution or by requesting the author's accepted manuscript. " if link else "")
                         + (f"Error: {r.error}" if r.error else ""))
        elif r.outcome == Outcome.NEEDS_CHOICE:
            parts.append("**Result.** More than one publication is plausible, so nothing was downloaded. Pick the intended paper below.")
        elif r.outcome == Outcome.NOT_FOUND:
            parts.append("**Result.** No matching publication could be identified; nothing was downloaded." + (f" {r.error}" if r.error else ""))
        r.summary = "\n\n".join(parts)
        r.citations = cites
