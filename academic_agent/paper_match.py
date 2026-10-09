"""Query understanding, title normalisation/similarity, author matching, de-duplication and merging.

The central rule: *similar words are not the same paper*. A candidate is only called an exact match
when its normalised title is (almost) identical; near misses are reported as "possible" so that the
agent asks the user instead of silently downloading the wrong publication.
"""
from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Iterable, Optional

from models import FullTextLocation, Paper, QueryType, VersionType

# ----------------------------------------------------------------------------- normalisation
_TAG_RE = re.compile(r"<[^>]+>")
_NONWORD_RE = re.compile(r"[^\w\s]|_", re.UNICODE)
_WS_RE = re.compile(r"\s+")
STOPWORDS = frozenset(
    "a an and are as at be by for from in into is it its of on or that the their this to using via vs with within "
    "do does how what why which when can between among toward towards new study analysis evidence".split()
)
_ALIASES = ((re.compile(r"\bp2p\b"), "peer to peer"), (re.compile(r"\bpeer\s*-?\s*to\s*-?\s*peer\b"), "peer to peer"),
            (re.compile(r"\bfin\s*-?\s*tech\b"), "fintech"))


def normalize_title(title: str) -> str:
    """Lower-case, strip markup/accents/punctuation, expand a few common abbreviations."""
    if not title:
        return ""
    t = html.unescape(str(title))
    t = _TAG_RE.sub(" ", t)
    t = unicodedata.normalize("NFKD", t)
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = unicodedata.normalize("NFKC", t).lower().replace("&", " and ")
    t = t.replace("-", " ").replace("‐", " ").replace("‑", " ").replace("–", " ").replace("—", " ")
    for pattern, repl in _ALIASES:
        t = pattern.sub(repl, t)
    t = _NONWORD_RE.sub(" ", t)
    return _WS_RE.sub(" ", t).strip()


def _stem(tok: str) -> str:
    if len(tok) > 4 and tok.endswith("ies"):
        return tok[:-3] + "y"
    if len(tok) > 4 and tok.endswith("es") and tok[-3] in "sxz":
        return tok[:-2]
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok


def content_tokens(text: str) -> list[str]:
    return [_stem(t) for t in normalize_title(text).split() if t not in STOPWORDS]


def squash(text: str) -> str:
    """Letters and digits only - robust to the odd spacing/hyphenation of PDF text extraction."""
    return re.sub(r"[^0-9a-z]", "", normalize_title(text).replace(" ", ""))


_SUBTITLE_SPLIT = re.compile(r"\s*(?::|\s[-\u2013\u2014]\s|\?\s)")


def is_subtitle_variant(a: str, b: str) -> bool:
    """True when one title is exactly the main title of the other, which adds a ': subtitle' (>= 4 words)."""
    na, nb = normalize_title(a), normalize_title(b)
    if not na or not nb or na == nb:
        return False
    for short, long_ in ((a, b), (b, a)):
        ns = normalize_title(short)
        if len(ns.split()) < 4:
            continue
        head = _SUBTITLE_SPLIT.split(html.unescape(long_), maxsplit=1)[0]
        if head != html.unescape(long_) and normalize_title(head) == ns:
            return True
    return False


def title_similarity(a: str, b: str) -> float:
    """0..1 similarity of two titles. 1.0 means identical after normalisation."""
    na, nb = normalize_title(a), normalize_title(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    seq = SequenceMatcher(None, na, nb, autojunk=False).ratio()
    ta, tb = set(content_tokens(a)), set(content_tokens(b))
    jac = len(ta & tb) / len(ta | tb) if (ta | tb) else 0.0
    wa, wb = na.split(), nb.split()
    tok_seq = SequenceMatcher(None, wa, wb, autojunk=False).ratio()      # sensitive to word order
    score = 0.4 * seq + 0.3 * jac + 0.3 * tok_seq
    if is_subtitle_variant(a, b):                                        # "Title" vs "Title: a subtitle"
        score = max(score, 0.9)
    return round(min(score, 1.0), 4)


# ----------------------------------------------------------------------------- DOI / ids
_DOI_RE = re.compile(r"(10\.\d{4,9}/[^\s\"']+)", re.I)
_ARXIV_NEW = re.compile(r"(?<![\d.])(\d{4}\.\d{4,5})(?:v\d+)?(?![\d.]*\d)")
_ARXIV_OLD = re.compile(r"\b([a-z\-]+(?:\.[A-Za-z]{2})?/\d{7})(?:v\d+)?\b")


def normalize_doi(value: str) -> str:
    if not value:
        return ""
    v = html.unescape(str(value)).strip()
    v = re.sub(r"^(?:https?://)?(?:dx\.)?doi\.org/", "", v, flags=re.I)
    v = re.sub(r"^doi:\s*", "", v, flags=re.I)
    m = _DOI_RE.search(v)
    if not m:
        return ""
    doi = m.group(1)
    if doi.endswith(">") and "<" not in doi:
        doi = doi[:-1]
    return doi.rstrip(".,;:)]}\"'").lower()


def extract_doi(text: str) -> str:
    return normalize_doi(text or "")


def extract_arxiv_id(text: str) -> str:
    t = (text or "").strip()
    m = re.search(r"arxiv\.org/(?:abs|pdf)/([^\s?#]+?)(?:\.pdf)?(?:$|[?#\s])", t + " ", re.I)
    if m:
        return re.sub(r"v\d+$", "", m.group(1))
    m = re.match(r"^(?:arxiv\s*[:\s]\s*)?(\d{4}\.\d{4,5})(?:v\d+)?$", t, re.I)
    if m:
        return m.group(1)
    m = re.match(r"^arxiv\s*[:\s]\s*(.+)$", t, re.I)
    if m:
        for rx in (_ARXIV_NEW, _ARXIV_OLD):
            mm = rx.search(m.group(1))
            if mm:
                return mm.group(1)
    return ""


# ----------------------------------------------------------------------------- authors
_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "phd", "md"}


def _ascii(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


def surname(name: str) -> str:
    """Family name of 'Given Family' or 'Family, Given' (lower-case, accent-free)."""
    if not name:
        return ""
    name = _ascii(html.unescape(name)).strip()
    if "," in name:
        last = name.split(",", 1)[0]
    else:
        toks = [t for t in re.split(r"\s+", name) if t.strip(".")]
        while len(toks) > 1 and toks[-1].lower().strip(".") in _SUFFIXES:
            toks.pop()
        last = toks[-1] if toks else ""
    last = re.sub(r"[^A-Za-zÀ-ɏ' ]", "", last).strip().lower()
    return last.split()[-1] if last else ""


def author_overlap(a: Iterable[str], b: Iterable[str], limit: int = 10) -> Optional[float]:
    """Share of the shorter author list found in the other (None when either list is empty)."""
    sa = [s for s in (surname(x) for x in list(a)[:limit]) if s]
    sb = [s for s in (surname(x) for x in list(b)[:limit]) if s]
    if not sa or not sb:
        return None
    small, big = (set(sa), set(sb)) if len(set(sa)) <= len(set(sb)) else (set(sb), set(sa))
    return len(small & big) / len(small)


# ----------------------------------------------------------------------------- query understanding
_QUESTION_START = re.compile(
    r"^(what|how|why|which|who|when|where|does|do|did|is|are|can|could|should|find|show|search|list|summari[sz]e|"
    r"papers?|research|studies|study|literature|review|recent|latest|evidence|tell me)\b", re.I)
_AUTHOR_YEAR = re.compile(
    r"^\(?\s*([A-Z][\w'’\-]+)(?:\s+et\s+al\.?|\s+(?:and|&)\s+[A-Z][\w'’\-]+)?\s*[,(]?\s*((?:19|20)\d{2})\)?[\s,:.\-]*(.*)$")


_APA = re.compile(r"^(.{3,200}?)\s*\(\s*((?:19|20)\d{2})[a-z]?\s*\)\s*[.,:]?\s*(.+)$")


@dataclass
class QueryInfo:
    raw: str
    type: QueryType
    title: str = ""            # text to match titles against
    doi: str = ""
    arxiv_id: str = ""
    author: str = ""
    year: Optional[int] = None
    domain: str = "general"    # general | biomedical | quantitative
    keywords: list = field(default_factory=list)


_BIOMED = re.compile(
    r"\b(patient|patients|clinical|trial|disease|cancer|tumou?r|gene|genes|genomic|protein|therapy|treatment|"
    r"covid|sars|virus|viral|vaccine|mice|mouse|cell|cells|drug|medical|health|hospital|diagnos\w*|surgery|"
    r"cardio\w*|neuro\w*|immun\w*|bacteria\w*|infection|epidemi\w*|mortality|pharmac\w*|biomarker\w*)\b", re.I)
_QUANT = re.compile(
    r"\b(neural|learning|quantum|algorithm|theorem|lattice|stochastic|bayesian|optimi[sz]ation|transformer|network|"
    r"graph|cosmolog\w*|particle|physics|mathematic\w*|asymptotic|regression|econometric\w*|volatility|asset|"
    r"portfolio|lending|credit|default|interest rates?|market|pricing|finance|financial)\b", re.I)


def detect_domain(text: str) -> str:
    bio, quant = len(_BIOMED.findall(text or "")), len(_QUANT.findall(text or ""))
    if bio and bio >= quant:
        return "biomedical"
    if quant:
        return "quantitative"
    return "general"


def classify_query(raw: str) -> QueryInfo:
    q = _WS_RE.sub(" ", (raw or "").strip().strip("\"'“”"))
    doi = extract_doi(q)
    if doi and (len(q) <= len(doi) + 24 or re.search(r"doi", q, re.I)):
        return QueryInfo(raw=raw, type=QueryType.DOI, doi=doi, title=q)
    arx = extract_arxiv_id(q)
    if arx:
        return QueryInfo(raw=raw, type=QueryType.ARXIV, arxiv_id=arx, title=q)
    domain = detect_domain(q)
    apa = _APA.match(q)                      # "Surname, A., & Surname, B. (2022). Title. Journal, 1(2), 3-4."
    if apa and len(apa.group(3).split()) >= 3:
        first = re.split(r"[,&]|\band\b", apa.group(1).strip())[0].strip().split()
        title = apa.group(3).strip()
        if re.search(r"\.\s+[A-Z][^.]*,\s*\d+", title):          # drop a trailing ". Journal, 12(3), 45-67"
            title = re.split(r"\.\s+(?=[A-Z][^.]*,\s*\d+)", title, maxsplit=1)[0]
        q_split = re.split(r"(?<=\?)\s+(?=[A-Z])", title, maxsplit=1)        # "Title? Journal Name" -> keep the question
        if len(q_split) == 2 and len(q_split[0].split()) >= 3:
            title = q_split[0]
        title = title.rstrip(" .")
        return QueryInfo(raw=raw, type=QueryType.AUTHOR_YEAR, author=(first[-1] if first else "").lower(), year=int(apa.group(2)),
                         title=title, domain=detect_domain(title), keywords=content_tokens(title))
    m = _AUTHOR_YEAR.match(q)
    if m and len(m.group(3).split()) >= 2:
        return QueryInfo(raw=raw, type=QueryType.AUTHOR_YEAR, author=m.group(1).lower(), year=int(m.group(2)),
                         title=m.group(3).strip(), domain=domain, keywords=content_tokens(m.group(3)))
    words = q.split()
    if q.endswith("?") or _QUESTION_START.match(q) or len(words) <= 3:
        return QueryInfo(raw=raw, type=QueryType.TOPIC, title=q.rstrip("?"), domain=domain,
                         keywords=content_tokens(q))
    return QueryInfo(raw=raw, type=QueryType.TITLE, title=q, domain=domain, keywords=content_tokens(q))


# ----------------------------------------------------------------------------- matching
_CORRECTION = re.compile(r"^\s*(correction|corrigendum|erratum|errata|retraction|retracted|expression of concern|"
                         r"reply to|comment on|response to|addendum|publisher correction)\b", re.I)


@dataclass
class MatchResult:
    score: float
    verdict: str                # exact | probable | possible | mismatch
    reasons: list = field(default_factory=list)


def match_query(qi: QueryInfo, paper: Paper, exact: float = 0.93, possible: float = 0.75) -> MatchResult:
    """How well does ``paper`` match what the user asked for?"""
    reasons: list[str] = []
    if qi.doi:
        if normalize_doi(paper.doi) == qi.doi or qi.doi in {normalize_doi(d) for d in paper.related_dois}:
            return MatchResult(1.0, "exact", ["DOI matches the query exactly"])
        return MatchResult(0.0, "mismatch", ["DOI differs from the query"])
    if qi.arxiv_id:
        if paper.arxiv_id and paper.arxiv_id.lower() == qi.arxiv_id.lower():
            return MatchResult(1.0, "exact", ["arXiv identifier matches the query"])
        return MatchResult(0.0, "mismatch", ["arXiv identifier differs"])
    score = title_similarity(qi.title, paper.title)
    reasons.append(f"title similarity {score:.2f}")
    if qi.author:
        ov = author_overlap([qi.author], paper.authors)
        if ov is not None and ov == 0:
            score *= 0.8
            reasons.append(f"author {qi.author!r} not among the authors")
        elif ov:
            reasons.append(f"author {qi.author!r} matches")
    if qi.year and paper.year:
        gap = abs(qi.year - paper.year)
        if gap > 1:
            score *= 0.9 if gap <= 3 else 0.75
            reasons.append(f"year differs by {gap}")
    if _CORRECTION.match(paper.title or "") and not _CORRECTION.match(qi.title):
        score *= 0.6
        reasons.append("record is a correction / comment, not the article itself")
    score = round(score, 4)
    if is_subtitle_variant(qi.title, paper.title) and score >= 0.9:
        reasons.append("query equals the main title of a title with subtitle")
        return MatchResult(max(score, exact), "exact", reasons)
    verdict = "exact" if score >= exact else "probable" if score >= (exact + possible) / 2 else (
        "possible" if score >= possible else "mismatch")
    return MatchResult(score, verdict, reasons)




def topic_relevance(qi: QueryInfo, paper: Paper) -> float:
    """Keyword coverage of the query in the title (weighted) and abstract."""
    kws = set(qi.keywords or content_tokens(qi.title))
    if not kws:
        return 0.0
    in_title = len(kws & set(content_tokens(paper.title))) / len(kws)
    in_abs = len(kws & set(content_tokens(paper.abstract))) / len(kws) if paper.abstract else 0.0
    return round(0.75 * in_title + 0.25 * in_abs, 4)


# ----------------------------------------------------------------------------- merge / dedupe
PROVIDER_AUTHORITY = {"crossref": 0, "datacite/doi.org": 1, "openalex": 2, "semantic_scholar": 3, "europepmc": 4,
                      "doaj": 5, "arxiv": 6, "core": 7, "unpaywall": 8, "openaire": 9}
_PREPRINT_DOI_PREFIXES = ("10.48550/", "10.1101/", "10.2139/", "10.31219/", "10.21203/", "10.20944/", "10.31234/",
                          "10.31235/", "10.5281/", "10.33774/")


def is_preprint_doi(doi: str) -> bool:
    return normalize_doi(doi).startswith(_PREPRINT_DOI_PREFIXES)


def same_work(a: Paper, b: Paper) -> bool:
    da, db = normalize_doi(a.doi), normalize_doi(b.doi)
    if da and db and da == db:
        return True
    if a.arxiv_id and b.arxiv_id and a.arxiv_id.lower() == b.arxiv_id.lower():
        return True
    if title_similarity(a.title, b.title) < 0.965:
        return False
    ov = author_overlap(a.authors, b.authors)
    if ov is not None and ov < 0.5:
        return False
    strong_authors = ov is not None and ov >= 0.8 and min(len(a.authors), len(b.authors)) >= 2
    if a.year and b.year and abs(a.year - b.year) > 3 and not strong_authors:
        return False
    return True


def norm_url(url: str) -> str:
    u = (url or "").strip()
    u = re.sub(r"^https?://(www\.)?", "", u, flags=re.I)
    return u.rstrip("/").lower()


def _merge_into(dst: Paper, src: Paper) -> None:
    rank = lambda p: min((PROVIDER_AUTHORITY.get(s, 20) for s in p.sources), default=20)
    if rank(src) < rank(dst) and src.title:
        dst.title = src.title
    dst.title = dst.title or src.title
    if len(src.authors) > len(dst.authors) or (rank(src) < rank(dst) and src.authors):
        dst.authors = list(src.authors)
    dd, sd = normalize_doi(dst.doi), normalize_doi(src.doi)
    if sd and not dd:
        dst.doi = sd
    elif sd and dd and sd != dd:
        # preprint / published counterpart: keep the journal DOI as primary
        if is_preprint_doi(dd) and not is_preprint_doi(sd):
            dst.related_dois.append(dd)
            dst.doi = sd
        elif sd not in dst.related_dois:
            dst.related_dois.append(sd)
    dst.year = dst.year or src.year
    if src.year and dst.year and not is_preprint_doi(src.doi or "") and src.work_type != "posted-content":
        dst.year = src.year if rank(src) <= rank(dst) else dst.year
    dst.venue = dst.venue or src.venue
    if len(src.abstract) > len(dst.abstract):
        dst.abstract = src.abstract
    dst.arxiv_id = dst.arxiv_id or src.arxiv_id
    dst.pmid = dst.pmid or src.pmid
    dst.pmcid = dst.pmcid or src.pmcid
    dst.work_type = dst.work_type if dst.work_type and dst.work_type != "posted-content" else (src.work_type or dst.work_type)
    if src.is_oa:
        dst.is_oa = True
    elif dst.is_oa is None:
        dst.is_oa = src.is_oa
    if src.citation_count is not None:
        dst.citation_count = max(dst.citation_count or 0, src.citation_count)
    for d in src.related_dois:
        nd = normalize_doi(d)
        if nd and nd != normalize_doi(dst.doi) and nd not in dst.related_dois:
            dst.related_dois.append(nd)
    seen = {norm_url(l.url) for l in dst.locations}
    for loc in src.locations:
        if norm_url(loc.url) not in seen:
            seen.add(norm_url(loc.url))
            dst.locations.append(loc)
    for s in src.sources:
        if s not in dst.sources:
            dst.sources.append(s)
    for k, v in src.record_urls.items():
        dst.record_urls.setdefault(k, v)
    dst.provenance.extend(src.provenance)
    dst.url = dst.url or src.url


def cluster_papers(papers: Iterable[Paper]) -> list[Paper]:
    """Merge records describing the same work (different providers, preprint vs. journal copy)."""
    clusters: list[Paper] = []
    for p in papers:
        if not p.title:
            continue
        if not p.provenance:
            p.provenance = [{"provider": (p.sources or ["?"])[0], "title": p.title, "year": p.year,
                             "doi": p.doi, "authors": list(p.authors[:6]), "venue": p.venue}]
        target = next((c for c in clusters if same_work(c, p)), None)
        if target is None:
            clusters.append(p)
        else:
            _merge_into(target, p)
    for c in clusters:
        c.doi = normalize_doi(c.doi)
        if c.doi and not c.url:
            c.url = f"https://doi.org/{c.doi}"
    return clusters


def confirmations(paper: Paper) -> dict:
    """Cross-check summary: how many independent providers agree on the key bibliographic fields."""
    snaps = paper.provenance or []
    def agree(field_name, key=lambda x: x):
        vals = [key(s.get(field_name)) for s in snaps if s.get(field_name)]
        if not vals:
            return 0, 0
        top = max(set(vals), key=vals.count)
        return vals.count(top), len(vals)
    return {
        "providers": sorted(set(paper.sources)),
        "doi": agree("doi", normalize_doi),
        "year": agree("year"),
        "title": (sum(1 for s in snaps if s.get("title") and title_similarity(s["title"], paper.title) >= 0.95), len(snaps)),
    }


# ----------------------------------------------------------------------------- host classification
_PRESERVATION_REPOS = ("arxiv.org", "europepmc.org", "ncbi.nlm.nih.gov", "biorxiv.org", "medrxiv.org", "ssrn.com",
                       "zenodo.org", "osf.io", "hal.science", "hal.archives-ouvertes.fr", "core.ac.uk", "repec.org",
                       "econstor.eu", "philarchive.org", "figshare.com", "chemrxiv.org", "researchsquare.com",
                       "preprints.org", "semanticscholar.org", "openaire.eu", "pure.mpg.de", "doaj.org")
_PREPRINT_SERVERS = ("arxiv.org", "biorxiv.org", "medrxiv.org", "ssrn.com", "chemrxiv.org", "researchsquare.com",
                     "preprints.org", "osf.io", "philarchive.org")
_INSTITUTIONAL_HINT = re.compile(
    r"(^|\.)(eprints?|dspace|repository|repositorio|repo|ir|scholarworks|digitalcommons|opus|pure|research|"
    r"publications|archive|openaccess|dash|etheses|theses|discovery|escholarship|cris)\.", re.I)
_EDU_HOST = re.compile(r"\.(edu|ac\.[a-z]{2}|edu\.[a-z]{2,3})$", re.I)


def classify_host(url: str) -> str:
    """Rough class of a host: preprint_server | repository (trusted/disciplinary) | institutional | doi | web."""
    from urllib.parse import urlsplit
    host = (urlsplit(url or "").hostname or "").lower()
    if host in ("doi.org", "dx.doi.org"):
        return "doi"
    for d in _PREPRINT_SERVERS:
        if host == d or host.endswith("." + d):
            return "preprint_server"
    for d in _PRESERVATION_REPOS:
        if host == d or host.endswith("." + d):
            return "repository"
    if _EDU_HOST.search(host) or _INSTITUTIONAL_HINT.search(host):
        return "institutional"
    return "web"


# ----------------------------------------------------------------------------- display safety
_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]()<>#|~!$])")


def md_escape(text: str) -> str:
    """Neutralise Markdown so untrusted titles/abstracts cannot inject links, images or formatting."""
    return _MD_SPECIAL.sub(r"\\\1", str(text or "")).replace("\n", " ")


# ----------------------------------------------------------------------------- filenames
_RESERVED = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10))}


def safe_component(text: str, max_len: int = 80) -> str:
    t = _ascii(html.unescape(_TAG_RE.sub(" ", text or "")))
    t = re.sub(r"[^A-Za-z0-9]+", "_", t).strip("_")
    return t[:max_len].rstrip("_")


def make_filename(paper: Paper, version: VersionType = VersionType.UNKNOWN, max_len: int = 120) -> str:
    """e.g. ``2024_Smith_Interest_Rates_and_P2P_Lending.pdf`` (suffix for non-final versions)."""
    year = str(paper.year) if paper.year else "n.d"
    sur = safe_component(surname(paper.authors[0]).title() if paper.authors else "Unknown", 30) or "Unknown"
    words = re.sub(r"[:;/\\|?*\"<>]", " ", _ascii(html.unescape(_TAG_RE.sub(" ", paper.title or "")))).split()
    title_part = safe_component("_".join(words[:8]), 70) or "Untitled"
    suffix = {VersionType.PREPRINT: "_preprint", VersionType.ACCEPTED: "_accepted_manuscript"}.get(version, "")
    stem = f"{year}_{sur}_{title_part}"[: max_len - len(suffix)].rstrip("_") + suffix
    if stem.lower() in _RESERVED:
        stem = "_" + stem
    return stem + ".pdf"
