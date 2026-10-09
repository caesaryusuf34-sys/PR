"""Find the official PDF of an Indonesian regulation (SEOJK, POJK, UU, PP, PMK, PBI ...).

Sources, in order:
1. the OJK regulation database (its public search form - used like a person would, one request at a time);
2. official ``.go.id`` pages / PDFs found through a configured web-search API (optional).

A PDF is only returned if the regulation's identification line (e.g. "Nomor 19/SEOJK.06/2025") appears in its
text, or - for scanned PDFs without a text layer - if the official page for exactly that regulation links to it
(reported as such). Portals that block automated access (Cloudflare / WAF challenges) are reported, never bypassed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Optional
from urllib.parse import quote, unquote, urljoin, urlsplit

from bs4 import BeautifulSoup

from access_checker import AccessChecker
from config import Settings
from database import Database, now_iso
from http_client import HttpClient, HttpError
from models import Attempt, DownloadInfo, FullTextLocation, Outcome, Paper, ProbeStatus, VersionType
from net_safety import URLValidationError
from pdf_downloader import PdfDownloader
from pdf_verifier import PdfVerifier
from regulation_match import RegRef, RegVerification, reg_filename, slug_matches, verify_regulation_text
from web_crawler import parse_page

Emit = Callable[..., None]
_RELATED = re.compile(r"^(abstrak|faq|lampiran|ringkasan|naskah|penjelasan|siaran|infografis|sosialisasi|kajian)", re.I)
RESTRICTED = {ProbeStatus.PAYWALL, ProbeStatus.LOGIN_REQUIRED, ProbeStatus.FORBIDDEN, ProbeStatus.CAPTCHA}


@dataclass
class RegCandidate:
    page_url: str
    title: str
    source: str
    pdfs: list = field(default_factory=list)        # [(url, label)] main document first
    related: list = field(default_factory=list)     # [(url, label)] abstract / FAQ / annexes
    page_confirmed: bool = False


@dataclass
class RegSearch:
    outcome: Outcome = Outcome.NOT_FOUND
    paper: Optional[Paper] = None
    info: Optional[DownloadInfo] = None
    attempts: list = field(default_factory=list)
    notes: dict = field(default_factory=dict)
    candidates: list = field(default_factory=list)
    manual_links: list = field(default_factory=list)   # [(label, url)] search pages for a human (not verified PDFs)
    used: list = field(default_factory=list)
    related: list = field(default_factory=list)        # FullTextLocation (unverified companion documents)
    restricted_seen: bool = False


def manual_search_links(ref: RegRef) -> list[tuple[str, str]]:
    q = quote(ref.short)
    links = [("OJK - database regulasi (pencarian)", "https://www.ojk.go.id/id/regulasi/default.aspx"),
             ("JDIH BPK - peraturan.bpk.go.id (pencarian)", f"https://peraturan.bpk.go.id/Search?keywords={q}"),
             ("JDIH Bank Indonesia", "https://jdih.bi.go.id/"),
             ("JDIH Nasional", f"https://jdihn.go.id/search?keyword={q}")]
    if ref.issuer == "Otoritas Jasa Keuangan":
        return links
    return links[1:] + links[:1]


class OjkRegulationSource:
    name = "ojk.go.id"
    BASE = "https://www.ojk.go.id/id/regulasi/default.aspx"
    P = "ctl00$PlaceHolderMain$ctl01$"

    def __init__(self, http: HttpClient, checker: AccessChecker):
        self.http, self.checker = http, checker

    # -- public search form (ASP.NET postback), the same request a visitor's browser sends ----------
    def _results_for(self, ref: RegRef, with_type: bool, with_year: bool) -> list[tuple[str, str]]:
        resp, html = self.http.get_text(self.BASE, max_bytes=4_000_000)
        if resp.status_code != 200:
            raise HttpError(f"OJK regulation page returned HTTP {resp.status_code}", resp.status_code)
        form = BeautifulSoup(html, "lxml").find("form")
        if form is None:
            raise HttpError("OJK search form not found (site layout changed?)")
        data = {i["name"]: i.get("value", "") for i in form.find_all("input") if i.get("name") and i.get("type") in ("hidden", None, "text")}
        data.update({self.P + "TextBoxNomor": ref.first_number, self.P + "TextBoxJudul": ref.title if not ref.number else "",
                     self.P + "DropDownListSektor": "0", self.P + "DropDownListSubSektor": "0",
                     self.P + "DropDownListJenisRegulasi": (ref.ojk_jenis or "0") if with_type else "0",
                     self.P + "DropDownListTahun": str(ref.year) if with_year else "0", self.P + "ButtonSearch": "Cari"})
        resp = self.http.request("POST", self.BASE, data=data, stream=True, check_robots=True, headers={"Referer": self.BASE})
        if resp.status_code != 200:
            resp.close()
            raise HttpError(f"OJK search returned HTTP {resp.status_code}", resp.status_code)
        body = self.http.read_limited(resp, 6_000_000).decode("utf-8", "replace")
        out, seen = [], set()
        for a in BeautifulSoup(body, "lxml").find_all("a", href=True):
            href = a["href"]
            if "/regulasi/Pages/" in href and not href.lower().endswith(("default.aspx", "search.aspx")):
                url = urljoin(self.BASE, href)
                if url not in seen:
                    seen.add(url)
                    out.append((url, re.sub(r"\s+", " ", a.get_text(" ", strip=True))))
        return out

    def search(self, ref: RegRef, emit: Emit, max_pages: int = 3) -> list[RegCandidate]:
        hits: list[tuple[str, str]] = []
        for with_type, with_year in ((True, True), (False, True), (False, False)):
            try:
                results = self._results_for(ref, with_type, with_year)
            except (HttpError, URLValidationError) as exc:
                emit("providers", f"{self.name}: {exc}", level="warning")
                raise
            hits = [(u, t) for u, t in results if slug_matches(ref, unquote(u), t)]
            emit("providers", f"{self.name}: {len(results)} result(s), {len(hits)} match(es) for {ref.short}")
            if hits:
                break
        cands: list[RegCandidate] = []
        for url, title in hits[:max_pages]:
            probe = self.checker.probe(url)
            if probe.status not in (ProbeStatus.HTML_PAGE, ProbeStatus.PAYWALL):
                emit("crawl", f"Official page not usable ({probe.status.value}): {url}", url, "warning")
                continue
            cand = RegCandidate(page_url=probe.final_url or url, title=title, source=self.name, page_confirmed=True)
            for a in BeautifulSoup(probe.html, "lxml").find_all("a", href=True):
                if ".pdf" in a["href"].lower().split("?")[0]:
                    pdf = urljoin(cand.page_url, a["href"])
                    label = re.sub(r"\s+", " ", a.get_text(" ", strip=True)) or unquote(pdf.rsplit("/", 1)[-1])
                    fname = unquote(pdf.rsplit("/", 1)[-1])
                    target = cand.related if (_RELATED.match(fname) or _RELATED.match(label)) else cand.pdfs
                    if all(pdf != u for u, _ in cand.pdfs + cand.related):
                        target.append((pdf, label))
            emit("crawl", f"Official page: {cand.page_url} ({len(cand.pdfs)} PDF, {len(cand.related)} companion document(s))", cand.page_url)
            cands.append(cand)
        return cands


class WebRegulationSource:
    """Official-looking (.go.id) pages and PDFs found through a configured web-search API."""
    name = "web"

    def __init__(self, http: HttpClient, checker: AccessChecker, providers: list, settings: Settings):
        self.http, self.checker, self.providers, self.settings = http, checker, providers, settings

    def queries(self, ref: RegRef) -> list[str]:
        base = ref.canonical
        qs = [f'"{ref.short}"', f'"{base}" filetype:pdf', f'{base} site:ojk.go.id', f'{base} site:go.id', f'{ref.kind_name} {ref.first_number} {ref.year} pdf']
        if ref.title:
            qs.insert(1, f'"{ref.kind} {ref.first_number}" {ref.title} {ref.year}')
        return list(dict.fromkeys(qs))[: self.settings.max_search_queries]

    def search(self, ref: RegRef, emit: Emit, seen_urls: set) -> list[RegCandidate]:
        cands: list[RegCandidate] = []
        for q in self.queries(ref):
            if self.http.budget.exhausted:
                break
            for wp in self.providers:
                emit("discover", f"Web search ({wp.label}): {q}")
                try:
                    hits = wp.search_web(q, 8)
                except (HttpError, URLValidationError) as exc:
                    emit("discover", f"{wp.label} unavailable: {exc}", level="warning")
                    continue
                for h in hits:
                    host = (urlsplit(h.url).hostname or "").lower()
                    official = host.endswith(".go.id") or host == "go.id"
                    if h.url in seen_urls or not (official or h.url.lower().split("?")[0].endswith(".pdf")):
                        continue
                    if not slug_matches(ref, unquote(h.url), h.title, h.snippet):
                        continue
                    seen_urls.add(h.url)
                    if h.url.lower().split("?")[0].endswith(".pdf"):
                        cands.append(RegCandidate(h.url, h.title, wp.name, pdfs=[(h.url, h.title)], page_confirmed=official))
                        continue
                    probe = self.checker.probe(h.url)
                    if probe.status != ProbeStatus.HTML_PAGE:
                        continue
                    info = parse_page(probe.html, probe.final_url or h.url, None)
                    if info.pdf_links:
                        cands.append(RegCandidate(probe.final_url or h.url, h.title, wp.name,
                                                  pdfs=[(l.url, l.text) for l in info.pdf_links], page_confirmed=official))
            if len(cands) >= 3:
                break
        return cands


class RegulationFinder:
    def __init__(self, http: HttpClient, checker: AccessChecker, downloader: PdfDownloader, verifier: PdfVerifier,
                 settings: Settings, db: Database, web_providers=lambda: []):
        self.http, self.checker, self.downloader, self.verifier = http, checker, downloader, verifier
        self.settings, self.db, self._web = settings, db, web_providers
        self.ojk = OjkRegulationSource(http, checker)

    def find(self, ref: RegRef, emit: Emit, query: str) -> RegSearch:
        res = RegSearch(manual_links=manual_search_links(ref))
        emit("understand", f"Regulation recognised: {ref.canonical} (issuer: {ref.issuer})")
        # 1. official OJK database
        if ref.ojk_jenis or ref.issuer == "Otoritas Jasa Keuangan":
            try:
                emit("providers", f"Searching the OJK regulation database for {ref.short}")
                res.candidates += self.ojk.search(ref, emit)
                res.used.append(self.ojk.name)
                res.notes[self.ojk.name] = f"ok ({len(res.candidates)} page(s))"
            except (HttpError, URLValidationError) as exc:
                res.notes[self.ojk.name] = str(exc)
        else:
            res.notes[self.ojk.name] = "not applicable for this kind of regulation"
        # 2. verify what we have; 3. widen with web search
        if self._try_candidates(ref, res, emit, query):
            return res
        web = self._web()
        if web and not self.http.budget.exhausted:
            emit("discover", "No verified PDF yet - searching official websites through the web-search API")
            seen = {c.page_url for c in res.candidates}
            more = WebRegulationSource(self.http, self.checker, web, self.settings).search(ref, emit, seen)
            res.used += [w.name for w in web]
            res.candidates += more
            if more and self._try_candidates(ref, res, emit, query, only=more):
                return res
        elif not web:
            emit("discover", "No web-search API configured (BRAVE_API_KEY or GOOGLE_CSE_*): only the OJK database was searched for this "
                 "regulation. For other issuers (BI, ministries) use the manual search links below.", level="warning")
        res.outcome = Outcome.NO_FULLTEXT if res.candidates else Outcome.NOT_FOUND
        if res.candidates and not res.paper:
            res.paper = self._paper(ref, res.candidates[0])
        return res

    # ------------------------------------------------------------------------------------------
    def _paper(self, ref: RegRef, cand: RegCandidate) -> Paper:
        t = (cand.title or ref.title).strip()
        title = f"{ref.canonical}" + (f" tentang {t}" if t and t.lower() not in ref.canonical.lower() else "")
        return Paper(title=title, authors=[ref.issuer], year=ref.year, venue=ref.kind_name, url=cand.page_url,
                     work_type="regulation", sources=[cand.source], record_urls={cand.source: cand.page_url})

    def _try_candidates(self, ref: RegRef, res: RegSearch, emit: Emit, query: str, only: Optional[list] = None) -> bool:
        s = self.settings
        tried = {a.url for a in res.attempts}
        downloads = 0
        for cand in (only if only is not None else res.candidates):
            for pdf, label in cand.pdfs:
                if pdf in tried or downloads >= s.max_download_attempts or self.http.budget.exhausted:
                    continue
                tried.add(pdf)
                emit("probe", f"Checking access: {pdf}", pdf)
                probe = self.checker.probe(pdf, referer=cand.page_url)
                if probe.status != ProbeStatus.PDF:
                    if probe.status in RESTRICTED:
                        res.restricted_seen = True
                    res.attempts.append(Attempt(pdf, probe.status.value, probe.detail, cand.source, "probe"))
                    emit("probe", f"Not usable ({probe.status.value}): {probe.detail}", pdf, "warning")
                    continue
                downloads += 1
                emit("download", f"Downloading to verify: {pdf}", pdf)
                dl = self.downloader.download(pdf, referer=cand.page_url)
                if not dl.ok:
                    res.attempts.append(Attempt(pdf, dl.status.value, dl.detail, cand.source, "download"))
                    emit("download", f"Download failed ({dl.status.value}): {dl.detail}", pdf, "warning")
                    continue
                ext = self.verifier.extract_text(dl.path)
                v = (verify_regulation_text(ref, ext.text, ext.pages, cand.page_confirmed, ext.warnings) if ext.ok
                     else RegVerification("corrupt", [ext.reason]))
                if not v.accepted:
                    self.downloader.discard(dl)
                    res.attempts.append(Attempt(pdf, "rejected_after_verification", f"{v.verdict}: " + "; ".join(v.reasons), cand.source, "verify"))
                    emit("verify-pdf", f"Rejected ({v.verdict}): {'; '.join(v.reasons)}", pdf, "warning")
                    continue
                paper = self._paper(ref, cand)
                path = ""
                if s.auto_download:
                    path = str(self.downloader.finalize(dl, paper, VersionType.PUBLISHED, reg_filename(ref, cand.title)))
                else:
                    self.downloader.discard(dl)
                pid = self.db.upsert_paper(paper)
                self.db.record_download(pid, file_path=path, sha256=dl.sha256, size_bytes=dl.size, version_type="published",
                                        source_url=dl.final_url or pdf, landing_url=cand.page_url, provider=cand.source,
                                        verification=v.to_dict(), query=query, status="downloaded" if path else "link_only")
                res.attempts.append(Attempt(pdf, "verified_link" if not path else "downloaded", "verified: " + "; ".join(v.reasons),
                                            cand.source, "verify"))
                emit("verify-pdf", f"Verified ({v.verdict}): {'; '.join(v.reasons)}", pdf, "success")
                res.paper = paper
                res.info = DownloadInfo(path=path, sha256=dl.sha256, size_bytes=dl.size, source_url=dl.final_url or pdf,
                                        landing_url=cand.page_url, provider=cand.source, version=VersionType.PUBLISHED,
                                        verification=v.to_dict(), access_date=now_iso(), saved=bool(path))
                res.related = [FullTextLocation(url=u, kind="pdf", version=VersionType.PUBLISHED, host_type="publisher", provider=cand.source,
                                                is_oa=True, origin_url=cand.page_url, note=f"companion document: {lbl[:70]}")
                               for u, lbl in cand.related + [(u, l) for u, l in cand.pdfs if u != pdf]]
                res.outcome = Outcome.DOWNLOADED if path else Outcome.LINK_FOUND
                return True
        return False
