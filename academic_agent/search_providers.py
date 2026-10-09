"""Provider abstraction layer and the concrete scholarly-search integrations.

Adding a new source = subclass :class:`SearchProvider` (or :class:`WebSearchProvider`), implement the
methods that apply and add the class to ``DEFAULT_PROVIDER_CLASSES``. Providers only talk to
*official, documented APIs*; each one declares when it is relevant (:meth:`relevance`) so the registry
can pick sources according to the query and the field of the paper.

Methods
  search(query, limit)       -> list[Paper]            bibliographic search (records carry known full-text locations)
  lookup_doi(doi)            -> Paper | None           authoritative record for a DOI
  find_fulltext(paper)       -> list[FullTextLocation] extra open-access copies for an *identified* paper
  search_web(query, limit)   -> list[WebHit]           general web search (WebSearchProvider only)
"""
from __future__ import annotations

import abc
import html
import re
from dataclasses import dataclass
from typing import Iterable, Optional
from urllib.parse import quote, urlsplit

from defusedxml import ElementTree as SafeET

from config import Settings
from http_client import HttpClient
from models import FullTextLocation, Paper, VersionType
from paper_match import (QueryInfo, classify_host, content_tokens, normalize_doi, normalize_title, title_similarity)

_TAG = re.compile(r"<[^>]+>")


def clean_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        value = value[0] if value else ""
    return re.sub(r"\s+", " ", html.unescape(_TAG.sub(" ", str(value)))).strip()


def guess_kind(url: str) -> str:
    path = urlsplit(url).path.lower()
    return "pdf" if path.endswith(".pdf") or "/pdf/" in path or path.endswith("/pdf") else "landing"


def host_for(url: str, hint: str = "") -> str:
    cls = classify_host(url)
    if hint == "publisher" and cls in ("web", "doi"):
        return "publisher"
    if hint == "repository":
        return cls if cls in ("institutional", "repository", "preprint_server") else "repository"
    return cls


def _year(value) -> Optional[int]:
    m = re.search(r"(1[5-9]|20)\d{2}", str(value or ""))
    return int(m.group(0)) if m else None


def _snapshot(provider: str, p: Paper, rank: int) -> dict:
    return {"provider": provider, "rank": rank, "title": p.title, "year": p.year, "doi": p.doi,
            "authors": list(p.authors[:6]), "venue": p.venue}


class ProviderError(Exception):
    pass


@dataclass
class WebHit:
    title: str
    url: str
    snippet: str = ""
    provider: str = ""


# =============================================================================== base classes
class SearchProvider(abc.ABC):
    name = "provider"
    label = "Provider"
    description = ""
    needs = ""                        # human-readable credential requirement, if any
    supports_doi_lookup = False
    supports_fulltext_lookup = False

    def __init__(self, http: HttpClient, settings: Settings):
        self.http = http
        self.settings = settings

    def available(self) -> tuple[bool, str]:
        return True, ""

    def relevance(self, qi: QueryInfo) -> float:
        return 1.0

    def search(self, query: str, limit: int = 8) -> list[Paper]:
        return []

    def lookup_doi(self, doi: str) -> Optional[Paper]:
        return None

    def lookup_arxiv(self, arxiv_id: str) -> Optional[Paper]:
        return None

    def find_fulltext(self, paper: Paper) -> list[FullTextLocation]:
        return []

    # shared helper ------------------------------------------------------------------------
    def _tag(self, papers: Iterable[Paper]) -> list[Paper]:
        out = []
        for rank, p in enumerate(papers):
            p.sources = [self.name]
            p.provenance = [_snapshot(self.name, p, rank)]
            out.append(p)
        return out


class WebSearchProvider(SearchProvider):
    """General web search through an API whose terms permit this use."""

    def search_web(self, query: str, limit: int = 8) -> list[WebHit]:
        return []


# =============================================================================== Crossref
class CrossrefProvider(SearchProvider):
    name = "crossref"
    label = "Crossref"
    description = "Bibliographic metadata, DOI resolution, publisher full-text links, preprint relations"
    supports_doi_lookup = True
    BASE = "https://api.crossref.org/works"
    SELECT = ("DOI,title,author,issued,published,container-title,abstract,link,license,type,URL,relation,"
              "is-referenced-by-count")

    def _params(self, **extra) -> dict:
        p = dict(extra)
        if self.settings.has_valid_email:
            p["mailto"] = self.settings.contact_email
        return p

    def search(self, query, limit=8):
        data = self.http.get_json(self.BASE, params=self._params(**{"query.bibliographic": query, "rows": limit,
                                                                   "select": self.SELECT}))
        items = ((data or {}).get("message") or {}).get("items") or []
        return self._tag(self.parse_item(i) for i in items if i.get("title"))

    def lookup_doi(self, doi):
        data = self.http.get_json(f"{self.BASE}/{quote(normalize_doi(doi), safe='/')}", params=self._params())
        item = (data or {}).get("message")
        if not item or not item.get("title"):
            return None
        return self._tag([self.parse_item(item)])[0]

    @staticmethod
    def parse_item(item: dict) -> Paper:
        doi = normalize_doi(item.get("DOI", ""))
        authors = []
        for a in item.get("author") or []:
            name = " ".join(x for x in (a.get("given"), a.get("family")) if x) or a.get("name", "")
            if name:
                authors.append(clean_text(name))
        date = (item.get("issued") or item.get("published") or {}).get("date-parts") or [[None]]
        year = _year(date[0][0]) if date and date[0] else None
        license_urls = [l.get("URL", "") for l in item.get("license") or []]
        cc = next((u for u in license_urls if "creativecommons.org" in u), "")
        p = Paper(title=clean_text(item.get("title")), authors=authors, year=year,
                  venue=clean_text(item.get("container-title")), doi=doi,
                  abstract=clean_text(item.get("abstract")), url=f"https://doi.org/{doi}" if doi else item.get("URL", ""),
                  work_type=item.get("type", ""), citation_count=item.get("is-referenced-by-count"),
                  is_oa=True if cc else None)
        for rel_kind in ("is-preprint-of", "has-preprint", "is-version-of", "has-version"):
            for r in (item.get("relation") or {}).get(rel_kind, []) or []:
                if r.get("id-type") == "doi" and normalize_doi(r.get("id", "")) not in (doi, ""):
                    p.related_dois.append(normalize_doi(r["id"]))
        for link in item.get("link") or []:
            url = link.get("URL")
            if not url:
                continue
            ctype = link.get("content-type", "")
            if "html" in ctype and "pdf" not in ctype:
                continue
            ver = {"vor": VersionType.PUBLISHED, "am": VersionType.ACCEPTED}.get(link.get("content-version"), VersionType.UNKNOWN)
            p.locations.append(FullTextLocation(
                url=url, kind="pdf" if "pdf" in ctype else guess_kind(url), version=ver, host_type="publisher",
                provider="crossref", license=cc, is_oa=True if cc else None, trusted=True,
                note="publisher full-text link registered with Crossref" + ("" if cc else " (may require a subscription)")))
        if doi:
            p.record_urls["crossref"] = f"https://api.crossref.org/works/{doi}"
        return p


# =============================================================================== DOI content negotiation
class DoiOrgProvider(SearchProvider):
    name = "datacite/doi.org"
    label = "doi.org (CSL JSON)"
    description = "Metadata for DOIs from any registration agency (Crossref, DataCite, mEDRA...)"
    supports_doi_lookup = True

    def relevance(self, qi):
        return 0.0 if not qi.doi else 0.5

    def lookup_doi(self, doi):
        doi = normalize_doi(doi)
        data = self.http.get_json(f"https://doi.org/{quote(doi, safe='/')}",
                                  headers={"Accept": "application/vnd.citationstyles.csl+json"})
        if not data or not data.get("title"):
            return None
        authors = [clean_text(" ".join(x for x in (a.get("given"), a.get("family")) if x) or a.get("literal", ""))
                   for a in data.get("author") or []]
        parts = (data.get("issued") or {}).get("date-parts") or [[None]]
        p = Paper(title=clean_text(data["title"]), authors=[a for a in authors if a], year=_year(parts[0][0]) if parts[0] else None,
                  venue=clean_text(data.get("container-title")), doi=doi, abstract=clean_text(data.get("abstract")),
                  url=f"https://doi.org/{doi}", work_type=str(data.get("type", "")))
        p.record_urls["datacite/doi.org"] = f"https://doi.org/{doi}"
        return self._tag([p])[0]


# =============================================================================== OpenAlex
class OpenAlexProvider(SearchProvider):
    name = "openalex"
    label = "OpenAlex"
    description = "Scholarly graph: metadata, open-access status and every known copy of a work"
    supports_doi_lookup = True
    supports_fulltext_lookup = True
    BASE = "https://api.openalex.org/works"
    SELECT = ("id,doi,title,display_name,publication_year,authorships,primary_location,best_oa_location,locations,"
              "open_access,abstract_inverted_index,type,cited_by_count,ids")

    def _params(self, **extra) -> dict:
        p = dict(extra)
        if self.settings.openalex_api_key:
            p["api_key"] = self.settings.openalex_api_key
        if self.settings.has_valid_email:
            p["mailto"] = self.settings.contact_email
        return p

    def search(self, query, limit=8):
        data = self.http.get_json(self.BASE, params=self._params(search=query, **{"per-page": limit, "select": self.SELECT}))
        return self._tag(self.parse_work(w) for w in (data or {}).get("results") or [] if w.get("title") or w.get("display_name"))

    def lookup_doi(self, doi):
        data = self.http.get_json(self.BASE, params=self._params(filter=f"doi:{normalize_doi(doi)}", **{"per-page": 1, "select": self.SELECT}))
        res = (data or {}).get("results") or []
        return self._tag([self.parse_work(res[0])])[0] if res else None

    def find_fulltext(self, paper):
        if not paper.doi:
            return []
        p = self.lookup_doi(paper.doi)
        return list(p.locations) if p else []

    @staticmethod
    def _abstract(inv: Optional[dict]) -> str:
        if not inv:
            return ""
        slots = {}
        for word, positions in inv.items():
            for pos in positions:
                slots[pos] = word
        return " ".join(slots[i] for i in sorted(slots))

    @classmethod
    def parse_work(cls, w: dict) -> Paper:
        doi = normalize_doi(w.get("doi") or "")
        authors = [clean_text((a.get("author") or {}).get("display_name")) for a in w.get("authorships") or []]
        prim = w.get("primary_location") or {}
        ids = w.get("ids") or {}
        oa = w.get("open_access") or {}
        p = Paper(title=clean_text(w.get("title") or w.get("display_name")), authors=[a for a in authors if a],
                  year=w.get("publication_year"), venue=clean_text((prim.get("source") or {}).get("display_name")),
                  doi=doi, abstract=clean_text(cls._abstract(w.get("abstract_inverted_index"))),
                  url=f"https://doi.org/{doi}" if doi else (w.get("id") or ""), work_type=w.get("type") or "",
                  is_oa=oa.get("is_oa"), citation_count=w.get("cited_by_count"),
                  pmid=str(ids.get("pmid", "")).rsplit("/", 1)[-1], pmcid=str(ids.get("pmcid", "")).rsplit("/", 1)[-1])
        seen = set()
        locs = list(w.get("locations") or [])
        if w.get("best_oa_location"):
            locs.insert(0, w["best_oa_location"])
        for loc in locs:
            src = loc.get("source") or {}
            ver = {"publishedVersion": VersionType.PUBLISHED, "acceptedVersion": VersionType.ACCEPTED,
                   "submittedVersion": VersionType.PREPRINT}.get(loc.get("version"), VersionType.UNKNOWN)
            hint = "publisher" if src.get("type") in ("journal", "book series", "conference", "ebook platform") else (
                "repository" if src.get("type") == "repository" else "")
            for key, kind in (("pdf_url", "pdf"), ("landing_page_url", "landing")):
                url = loc.get(key)
                if not url or url in seen:
                    continue
                if kind == "landing" and not loc.get("is_oa"):
                    continue
                seen.add(url)
                p.locations.append(FullTextLocation(
                    url=url, kind=kind, version=ver, host_type=host_for(url, hint), provider="openalex",
                    license=loc.get("license") or "", is_oa=loc.get("is_oa"), repository=clean_text(src.get("display_name")),
                    note=f"OpenAlex location ({src.get('display_name', 'unknown source')})"))
        if w.get("id"):
            p.record_urls["openalex"] = w["id"]
        return p


# =============================================================================== Semantic Scholar
class SemanticScholarProvider(SearchProvider):
    name = "semantic_scholar"
    label = "Semantic Scholar"
    description = "Paper discovery, citations and open-access PDF links"
    supports_doi_lookup = True
    supports_fulltext_lookup = True
    BASE = "https://api.semanticscholar.org/graph/v1/paper"
    FIELDS = "title,authors,year,venue,externalIds,abstract,openAccessPdf,isOpenAccess,citationCount,publicationTypes,journal"

    def _headers(self) -> dict:
        return {"x-api-key": self.settings.semantic_scholar_api_key} if self.settings.semantic_scholar_api_key else {}

    def relevance(self, qi):
        return 0.9

    def search(self, query, limit=8):
        data = self.http.get_json(f"{self.BASE}/search", params={"query": query, "limit": limit, "fields": self.FIELDS},
                                  headers=self._headers())
        return self._tag(self.parse_paper(x) for x in (data or {}).get("data") or [] if x.get("title"))

    def lookup_doi(self, doi):
        data = self.http.get_json(f"{self.BASE}/DOI:{quote(normalize_doi(doi), safe='/')}", params={"fields": self.FIELDS},
                                  headers=self._headers())
        return self._tag([self.parse_paper(data)])[0] if data and data.get("title") else None

    def lookup_arxiv(self, arxiv_id):
        data = self.http.get_json(f"{self.BASE}/ARXIV:{quote(arxiv_id)}", params={"fields": self.FIELDS}, headers=self._headers())
        return self._tag([self.parse_paper(data)])[0] if data and data.get("title") else None

    def find_fulltext(self, paper):
        if not paper.doi:
            return []
        p = self.lookup_doi(paper.doi)
        return list(p.locations) if p else []

    @staticmethod
    def parse_paper(x: dict) -> Paper:
        ext = x.get("externalIds") or {}
        doi = normalize_doi(ext.get("DOI", ""))
        venue = clean_text(x.get("venue")) or clean_text((x.get("journal") or {}).get("name"))
        p = Paper(title=clean_text(x.get("title")), authors=[clean_text(a.get("name")) for a in x.get("authors") or [] if a.get("name")],
                  year=x.get("year"), venue=venue, doi=doi, abstract=clean_text(x.get("abstract")),
                  url=f"https://doi.org/{doi}" if doi else f"https://www.semanticscholar.org/paper/{x.get('paperId', '')}",
                  arxiv_id=str(ext.get("ArXiv", "") or ""), pmid=str(ext.get("PubMed", "") or ""),
                  pmcid=("PMC" + str(ext["PubMedCentral"]).removeprefix("PMC")) if ext.get("PubMedCentral") else "",
                  is_oa=x.get("isOpenAccess"), citation_count=x.get("citationCount"),
                  work_type=",".join(x.get("publicationTypes") or []))
        oa = x.get("openAccessPdf") or {}
        if oa.get("url"):
            status = (oa.get("status") or "").upper()
            ver = VersionType.PUBLISHED if status in ("GOLD", "HYBRID", "BRONZE") else VersionType.UNKNOWN
            p.locations.append(FullTextLocation(
                url=oa["url"], kind="pdf" if guess_kind(oa["url"]) == "pdf" else "unknown", version=ver,
                host_type=host_for(oa["url"], "repository" if status == "GREEN" else ""), provider="semantic_scholar",
                license=oa.get("license") or "", is_oa=True, note=f"Semantic Scholar openAccessPdf ({status or 'status n/a'})"))
        if x.get("paperId"):
            p.record_urls["semantic_scholar"] = f"https://www.semanticscholar.org/paper/{x['paperId']}"
        return p


# =============================================================================== Unpaywall
class UnpaywallProvider(SearchProvider):
    name = "unpaywall"
    label = "Unpaywall"
    description = "Legally available open-access copies (publisher, repositories) for a DOI"
    needs = "CONTACT_EMAIL (a real address; Unpaywall requires it)"
    supports_doi_lookup = True
    supports_fulltext_lookup = True
    BASE = "https://api.unpaywall.org/v2"

    def available(self):
        if not self.settings.has_valid_email:
            return False, "set CONTACT_EMAIL to a real e-mail address (required by Unpaywall)"
        return True, ""

    def relevance(self, qi):
        return 0.0          # DOI-keyed service: used through lookup_doi / find_fulltext only

    def lookup_doi(self, doi):
        data = self.http.get_json(f"{self.BASE}/{quote(normalize_doi(doi), safe='/')}", params={"email": self.settings.contact_email})
        return self._tag([self.parse_record(data)])[0] if data and data.get("title") else None

    def find_fulltext(self, paper):
        if not paper.doi:
            return []
        p = self.lookup_doi(paper.doi)
        return list(p.locations) if p else []

    @staticmethod
    def parse_record(d: dict) -> Paper:
        doi = normalize_doi(d.get("doi", ""))
        authors = [clean_text(" ".join(x for x in (a.get("given"), a.get("family")) if x)) for a in d.get("z_authors") or []]
        p = Paper(title=clean_text(d.get("title")), authors=[a for a in authors if a], year=d.get("year"),
                  venue=clean_text(d.get("journal_name")), doi=doi, url=f"https://doi.org/{doi}" if doi else "",
                  is_oa=d.get("is_oa"), work_type=d.get("genre") or "")
        seen = set()
        locs = list(d.get("oa_locations") or [])
        if d.get("best_oa_location"):
            locs.insert(0, d["best_oa_location"])
        for loc in locs:
            ver = {"publishedVersion": VersionType.PUBLISHED, "acceptedVersion": VersionType.ACCEPTED,
                   "submittedVersion": VersionType.PREPRINT}.get(loc.get("version"), VersionType.UNKNOWN)
            inst = loc.get("repository_institution") or ""
            hint = loc.get("host_type") or ""
            for key, kind in (("url_for_pdf", "pdf"), ("url_for_landing_page", "landing"), ("url", "unknown")):
                url = loc.get(key)
                if not url or url in seen or (key == "url" and (loc.get("url_for_pdf") or loc.get("url_for_landing_page"))):
                    continue
                seen.add(url)
                ht = "institutional" if inst else host_for(url, hint)
                p.locations.append(FullTextLocation(
                    url=url, kind=kind, version=ver, host_type=ht, provider="unpaywall", license=loc.get("license") or "",
                    is_oa=True, repository=clean_text(inst or loc.get("endpoint_id") or ""),
                    note=f"Unpaywall {hint or 'location'}" + (f" ({inst})" if inst else "") + f", evidence: {loc.get('evidence', 'n/a')}"))
        if doi:
            p.record_urls["unpaywall"] = f"https://unpaywall.org/{doi}"
        return p


# =============================================================================== CORE
class CoreProvider(SearchProvider):
    name = "core"
    label = "CORE"
    description = "Full texts harvested from thousands of institutional and subject repositories"
    needs = "CORE_API_KEY (free)"
    supports_fulltext_lookup = True
    BASE = "https://api.core.ac.uk/v3/search/works"

    def available(self):
        return (True, "") if self.settings.core_api_key else (False, "set CORE_API_KEY (free key from core.ac.uk/services/api)")

    def relevance(self, qi):
        return 0.8

    def _get(self, q: str, limit: int):
        return self.http.get_json(self.BASE, params={"q": q, "limit": limit},
                                  headers={"Authorization": f"Bearer {self.settings.core_api_key}"})

    def search(self, query, limit=8):
        data = self._get(query, limit)
        return self._tag(self.parse_work(w) for w in (data or {}).get("results") or [] if w.get("title"))

    def find_fulltext(self, paper):
        out: list[FullTextLocation] = []
        queries = []
        if paper.doi:
            queries.append(f'doi:"{paper.doi}"')
        if paper.title:
            queries.append('title:"' + paper.title.replace('"', " ") + '"')
        for q in queries:
            data = self._get(q, 5)
            for w in (data or {}).get("results") or []:
                cand = self.parse_work(w)
                same = (paper.doi and normalize_doi(cand.doi) == normalize_doi(paper.doi)) or \
                       title_similarity(cand.title, paper.title) >= 0.93
                if same:
                    out.extend(cand.locations)
            if out:
                break
        return out

    @staticmethod
    def parse_work(w: dict) -> Paper:
        doi = normalize_doi(w.get("doi") or "")
        p = Paper(title=clean_text(w.get("title")), authors=[clean_text(a.get("name")) for a in w.get("authors") or [] if a.get("name")],
                  year=w.get("yearPublished"), venue=clean_text((w.get("journals") or [{}])[0].get("title") if w.get("journals") else ""),
                  doi=doi, abstract=clean_text(w.get("abstract")), url=f"https://doi.org/{doi}" if doi else f"https://core.ac.uk/works/{w.get('id')}",
                  is_oa=True)
        for url, kind, note in ((w.get("downloadUrl"), "pdf", "CORE download URL"),
                                (w.get("sourceFulltextUrls") if isinstance(w.get("sourceFulltextUrls"), str) else None, "unknown", "repository source URL")):
            if url:
                p.locations.append(FullTextLocation(url=url, kind=kind if kind == "pdf" and guess_kind(url) == "pdf" or "core.ac.uk/download" in url else "unknown",
                                                    version=VersionType.UNKNOWN, host_type="repository", provider="core",
                                                    is_oa=True, note=note))
        if w.get("id"):
            p.record_urls["core"] = f"https://core.ac.uk/works/{w['id']}"
            p.locations.append(FullTextLocation(url=f"https://core.ac.uk/works/{w['id']}", kind="landing", host_type="repository",
                                                provider="core", is_oa=True, note="CORE record page"))
        return p


# =============================================================================== Europe PMC
class EuropePmcProvider(SearchProvider):
    name = "europepmc"
    label = "Europe PMC"
    description = "Biomedical and life-science literature, PubMed Central full texts, preprints"
    supports_doi_lookup = True
    supports_fulltext_lookup = True
    BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

    def relevance(self, qi):
        if qi.type.value == "doi":
            return 0.7
        return 1.0 if qi.domain == "biomedical" else 0.45

    def _query(self, q: str, limit: int):
        return self.http.get_json(self.BASE, params={"query": q, "format": "json", "resultType": "core", "pageSize": limit})

    def search(self, query, limit=8):
        q = f'TITLE:"{query}"' if len(query.split()) >= 5 and '"' not in query else query
        data = self._query(q, limit)
        res = ((data or {}).get("resultList") or {}).get("result") or []
        if not res and q != query:
            res = ((self._query(query, limit) or {}).get("resultList") or {}).get("result") or []
        return self._tag(self.parse_result(r) for r in res if r.get("title"))

    def lookup_doi(self, doi):
        data = self._query(f'DOI:"{normalize_doi(doi)}"', 1)
        res = ((data or {}).get("resultList") or {}).get("result") or []
        return self._tag([self.parse_result(res[0])])[0] if res and res[0].get("title") else None

    def find_fulltext(self, paper):
        p = self.lookup_doi(paper.doi) if paper.doi else None
        return list(p.locations) if p else []

    @staticmethod
    def _authors(r: dict) -> list[str]:
        lst = ((r.get("authorList") or {}).get("author")) or []
        names = []
        for a in lst:
            n = " ".join(x for x in (a.get("firstName"), a.get("lastName")) if x) or a.get("fullName", "")
            if n:
                names.append(clean_text(n))
        if names:
            return names
        for chunk in (r.get("authorString") or "").rstrip(".").split(", "):
            m = re.match(r"^(.*?)\s+([A-Z]{1,4})$", chunk.strip())
            names.append(f"{m.group(2)} {m.group(1)}" if m else chunk.strip())
        return [n for n in names if n]

    @classmethod
    def parse_result(cls, r: dict) -> Paper:
        doi = normalize_doi(r.get("doi") or "")
        pmcid = r.get("pmcid") or ""
        is_pre = r.get("source") == "PPR"
        p = Paper(title=clean_text(r.get("title")).rstrip("."), authors=cls._authors(r), year=_year(r.get("pubYear")),
                  venue=clean_text((r.get("journalInfo") or {}).get("journal", {}).get("title") or r.get("journalTitle") or ""),
                  doi=doi, abstract=clean_text(r.get("abstractText")), pmid=str(r.get("pmid") or ""), pmcid=pmcid,
                  url=f"https://doi.org/{doi}" if doi else "", is_oa=(r.get("isOpenAccess") == "Y"),
                  work_type="posted-content" if is_pre else (r.get("pubType") or ""))
        ver = VersionType.PREPRINT if is_pre else VersionType.UNKNOWN
        seen = set()
        for u in ((r.get("fullTextUrlList") or {}).get("fullTextUrl")) or []:
            url = u.get("url")
            if not url or url in seen or u.get("availabilityCode") not in ("OA", "F"):
                continue
            if u.get("documentStyle") in ("abs",):
                continue
            seen.add(url)
            site = u.get("site", "")
            p.locations.append(FullTextLocation(
                url=url, kind="pdf" if u.get("documentStyle") == "pdf" else "landing", version=ver,
                host_type="publisher" if site == "DOI" else host_for(url, "repository"), provider="europepmc", is_oa=True,
                note=f"Europe PMC full-text link ({site or 'n/a'}, {u.get('availability', '')})"))
        if pmcid and (r.get("isOpenAccess") == "Y" or r.get("inEPMC") == "Y"):
            url = f"https://europepmc.org/articles/{pmcid}?pdf=render"
            if url not in seen:
                p.locations.append(FullTextLocation(url=url, kind="pdf", version=ver, host_type="repository", provider="europepmc",
                                                    is_oa=True, repository="Europe PMC", note=f"Europe PMC rendered PDF of {pmcid}"))
        if r.get("id") and r.get("source"):
            p.record_urls["europepmc"] = f"https://europepmc.org/article/{r['source']}/{r['id']}"
        return p


# =============================================================================== DOAJ
class DoajProvider(SearchProvider):
    name = "doaj"
    label = "DOAJ"
    description = "Directory of Open Access Journals: articles that are open access at the publisher"
    BASE = "https://doaj.org/api/search/articles/"

    def relevance(self, qi):
        return 0.6

    def search(self, query, limit=8):
        query = re.sub(r'[+\-&|!(){}\[\]^"~*?:\\/]', " ", query)          # Lucene special characters break the query (HTTP 400)
        query = re.sub(r"\s+", " ", query).strip()
        q = f'bibjson.title:"{query}"' if len(query.split()) >= 5 else query
        data = self.http.get_json(self.BASE + quote(q, safe=""), params={"pageSize": limit})
        res = (data or {}).get("results") or []
        if not res and q != query:
            res = (self.http.get_json(self.BASE + quote(query, safe=""), params={"pageSize": limit}) or {}).get("results") or []
        return self._tag(self.parse_article(r) for r in res if (r.get("bibjson") or {}).get("title"))

    @staticmethod
    def parse_article(r: dict) -> Paper:
        b = r.get("bibjson") or {}
        doi = normalize_doi(next((i.get("id") for i in b.get("identifier") or [] if i.get("type", "").lower() == "doi"), "") or "")
        p = Paper(title=clean_text(b.get("title")), authors=[clean_text(a.get("name")) for a in b.get("author") or [] if a.get("name")],
                  year=_year(b.get("year")), venue=clean_text((b.get("journal") or {}).get("title")), doi=doi,
                  abstract=clean_text(b.get("abstract")), url=f"https://doi.org/{doi}" if doi else "", is_oa=True,
                  work_type="journal-article")
        for l in b.get("link") or []:
            url = l.get("url")
            if url and l.get("type") == "fulltext":
                kind = "pdf" if "pdf" in (l.get("content_type") or "").lower() or guess_kind(url) == "pdf" else "landing"
                p.locations.append(FullTextLocation(url=url, kind=kind, version=VersionType.PUBLISHED, host_type="publisher",
                                                    provider="doaj", is_oa=True, note="DOAJ: open-access full text at the publisher"))
        if r.get("id"):
            p.record_urls["doaj"] = f"https://doaj.org/article/{r['id']}"
        return p


# =============================================================================== arXiv
class ArxivProvider(SearchProvider):
    name = "arxiv"
    label = "arXiv"
    description = "Preprints in physics, mathematics, computer science, quantitative finance, economics, statistics"
    supports_doi_lookup = False
    BASE = "https://export.arxiv.org/api/query"
    NS = {"a": "http://www.w3.org/2005/Atom", "x": "http://arxiv.org/schemas/atom"}

    def relevance(self, qi):
        if qi.type.value == "arxiv":
            return 1.0
        if qi.domain == "quantitative":
            return 0.9
        return 0.25 if qi.domain == "biomedical" else 0.5

    def _fetch(self, params: dict) -> list[Paper]:
        resp = self.http.request("GET", self.BASE, params=params, stream=True, headers={"Accept": "application/atom+xml"})
        if resp.status_code == 429:
            resp.close()
            raise ProviderError("arXiv API rate limit (HTTP 429)")
        if resp.status_code >= 400:
            code = resp.status_code
            resp.close()
            raise ProviderError(f"arXiv API HTTP {code}")
        body = self.http.read_limited(resp, self.settings.max_json_mb * 1024 * 1024)
        return self.parse_feed(body)

    def search(self, query, limit=8):
        if len(query.split()) >= 5 and '"' not in query:
            papers = self._fetch({"search_query": f'ti:"{query}"', "max_results": limit})
            if papers:
                return self._tag(papers)
        terms = " AND ".join(f"all:{t}" for t in dict.fromkeys(normalize_title(query).split()) if len(t) > 2)[:400]
        return self._tag(self._fetch({"search_query": terms or f"all:{query}", "max_results": limit,
                                      "sortBy": "relevance"}))

    def lookup_arxiv(self, arxiv_id):
        res = self._fetch({"id_list": arxiv_id, "max_results": 1})
        return self._tag(res)[0] if res else None

    @classmethod
    def parse_feed(cls, body: bytes) -> list[Paper]:
        try:
            root = SafeET.fromstring(body)
        except Exception as exc:  # noqa: BLE001 - malformed or hostile XML
            raise ProviderError(f"unparseable arXiv response: {type(exc).__name__}") from exc
        out = []
        for e in root.findall("a:entry", cls.NS):
            raw_id = (e.findtext("a:id", "", cls.NS) or "").strip()
            m = re.search(r"arxiv\.org/abs/(.+?)(v\d+)?$", raw_id)
            aid = m.group(1) if m else ""
            title = clean_text(e.findtext("a:title", "", cls.NS))
            if not aid or not title or title.lower() == "error":
                continue
            doi = normalize_doi(e.findtext("x:doi", "", cls.NS) or "")
            p = Paper(title=title, authors=[clean_text(a.findtext("a:name", "", cls.NS)) for a in e.findall("a:author", cls.NS)],
                      year=_year(e.findtext("a:published", "", cls.NS)), venue=clean_text(e.findtext("x:journal_ref", "", cls.NS)),
                      doi=doi, abstract=clean_text(e.findtext("a:summary", "", cls.NS)), arxiv_id=aid,
                      url=f"https://arxiv.org/abs/{aid}", is_oa=True, work_type="posted-content")
            p.locations.append(FullTextLocation(url=f"https://arxiv.org/pdf/{aid}", kind="pdf", version=VersionType.PREPRINT,
                                                host_type="preprint_server", provider="arxiv", is_oa=True,
                                                repository="arXiv", note="arXiv preprint PDF (not necessarily the published version)"))
            p.record_urls["arxiv"] = f"https://arxiv.org/abs/{aid}"
            out.append(p)
        return out


# =============================================================================== OpenAIRE
class OpenAireProvider(SearchProvider):
    name = "openaire"
    label = "OpenAIRE"
    description = "European aggregator of institutional repositories and open-access publications"
    BASE = "https://api.openaire.eu/graph/v1/researchProducts"

    def relevance(self, qi):
        return 0.6

    def search(self, query, limit=8):
        data = self.http.get_json(self.BASE, params={"mainTitle": query, "type": "publication", "pageSize": limit})
        return self._tag(self.parse_product(r) for r in (data or {}).get("results") or [] if r.get("mainTitle"))

    @staticmethod
    def parse_product(r: dict) -> Paper:
        pids = {x.get("scheme"): x.get("value") for x in ((r.get("pids")) or []) if isinstance(x, dict)}
        doi = normalize_doi(pids.get("doi", "") or "")
        authors = [clean_text(a.get("fullName")) for a in r.get("authors") or [] if a.get("fullName")]
        p = Paper(title=clean_text(r.get("mainTitle")), authors=authors, year=_year(r.get("publicationDate")),
                  venue=clean_text((r.get("container") or {}).get("name")), doi=doi,
                  abstract=clean_text((r.get("descriptions") or [""])[0]),
                  url=f"https://doi.org/{doi}" if doi else "", is_oa=(r.get("bestAccessRight") or {}).get("code") == "c_abf2")
        for inst in r.get("instances") or []:
            right = (inst.get("accessRight") or {}).get("code")
            if right not in ("c_abf2", None):          # keep only OPEN (or unspecified) instances
                continue
            typ = (inst.get("type") or "").lower()
            ver = VersionType.PREPRINT if "preprint" in typ else VersionType.UNKNOWN
            hosted = clean_text((inst.get("hostedBy") or {}).get("name") if isinstance(inst.get("hostedBy"), dict) else inst.get("hostedBy"))
            for url in inst.get("urls") or []:
                if "doi.org" in url:
                    continue
                p.locations.append(FullTextLocation(
                    url=url, kind=guess_kind(url), version=ver, host_type=host_for(url, "repository"), provider="openaire",
                    license=inst.get("license") or "", is_oa=True, repository=hosted,
                    note=f"OpenAIRE instance hosted by {hosted or 'unknown repository'}"))
        if r.get("id"):
            p.record_urls["openaire"] = f"https://explore.openaire.eu/search/publication?pid={r['id']}"
        return p


# =============================================================================== web search providers
class BraveSearchProvider(WebSearchProvider):
    name = "web:brave"
    label = "Brave Search API"
    description = "General web search for repository copies and PDFs"
    needs = "BRAVE_API_KEY"
    BASE = "https://api.search.brave.com/res/v1/web/search"

    def available(self):
        return (True, "") if self.settings.brave_api_key else (False, "set BRAVE_API_KEY (api.search.brave.com)")

    def search_web(self, query, limit=8):
        data = self.http.get_json(self.BASE, params={"q": query, "count": min(limit, 20), "safesearch": "moderate"},
                                  headers={"X-Subscription-Token": self.settings.brave_api_key})
        return [WebHit(clean_text(r.get("title")), r["url"], clean_text(r.get("description")), self.name)
                for r in ((data or {}).get("web") or {}).get("results") or [] if r.get("url")]


class GoogleCseProvider(WebSearchProvider):
    name = "web:google_cse"
    label = "Google Programmable Search"
    description = "Web search through a Programmable Search Engine you configure (Custom Search JSON API)"
    needs = "GOOGLE_CSE_API_KEY and GOOGLE_CSE_CX"
    BASE = "https://www.googleapis.com/customsearch/v1"

    def available(self):
        if self.settings.google_cse_api_key and self.settings.google_cse_cx:
            return True, ""
        return False, "set GOOGLE_CSE_API_KEY and GOOGLE_CSE_CX"

    def search_web(self, query, limit=8):
        data = self.http.get_json(self.BASE, params={"key": self.settings.google_cse_api_key, "cx": self.settings.google_cse_cx,
                                                     "q": query, "num": min(limit, 10)})
        return [WebHit(clean_text(r.get("title")), r["link"], clean_text(r.get("snippet")), self.name)
                for r in (data or {}).get("items") or [] if r.get("link")]


# =============================================================================== registry
DEFAULT_PROVIDER_CLASSES = (CrossrefProvider, DoiOrgProvider, OpenAlexProvider, SemanticScholarProvider, UnpaywallProvider,
                            CoreProvider, EuropePmcProvider, DoajProvider, ArxivProvider, OpenAireProvider)
DEFAULT_WEB_CLASSES = (BraveSearchProvider, GoogleCseProvider)
MIN_RELEVANCE = 0.3


class ProviderRegistry:
    def __init__(self, providers: Iterable[SearchProvider], web_providers: Iterable[WebSearchProvider] = ()):
        self.providers = list(providers)
        self.web_providers = list(web_providers)

    @classmethod
    def default(cls, http: HttpClient, settings: Settings) -> "ProviderRegistry":
        return cls([c(http, settings) for c in DEFAULT_PROVIDER_CLASSES], [c(http, settings) for c in DEFAULT_WEB_CLASSES])

    def select(self, qi: QueryInfo, include_low_relevance: bool = False) -> tuple[list[SearchProvider], dict]:
        """Choose and order providers for this query; returns (selected, {name: reason skipped})."""
        chosen, skipped = [], {}
        for p in self.providers:
            ok, why = p.available()
            if not ok:
                skipped[p.name] = why
                continue
            rel = p.relevance(qi)
            if rel < (0.0 if include_low_relevance else MIN_RELEVANCE) or rel <= 0:
                skipped[p.name] = "not relevant for this kind of query"
                continue
            chosen.append((rel, p))
        chosen.sort(key=lambda t: -t[0])
        return [p for _, p in chosen], skipped

    def available_web(self) -> list[WebSearchProvider]:
        return [w for w in self.web_providers if w.available()[0]]

    def status(self) -> list[dict]:
        rows = []
        for p in [*self.providers, *self.web_providers]:
            ok, why = p.available()
            rows.append({"name": p.name, "label": p.label, "description": p.description, "available": ok,
                         "note": why or "ready", "needs": p.needs})
        return rows
