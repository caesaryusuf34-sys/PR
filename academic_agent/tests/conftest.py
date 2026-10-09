"""Shared test helpers: a scripted fake web (no network), a tiny PDF generator and fake providers."""
from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Callable, Optional, Union

import pytest
import requests
from requests.adapters import BaseAdapter
from requests.structures import CaseInsensitiveDict
from urllib3.response import HTTPResponse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Settings                      # noqa: E402
from database import Database                    # noqa: E402
from http_client import HttpClient               # noqa: E402
from models import FullTextLocation, Paper, VersionType   # noqa: E402
from search_providers import ProviderRegistry, SearchProvider, WebHit, WebSearchProvider   # noqa: E402

PUBLIC_IP = "93.184.216.34"


# ----------------------------------------------------------------------------- PDF generator
def make_pdf(pages: list[list[str]] | list[str] = (), pad: int = 6000, extra_objects: bytes = b"",
             catalog_extra: bytes = b"") -> bytes:
    """Build a small, valid PDF whose pages contain the given lines of text (Helvetica)."""
    if pages and isinstance(pages[0], str):
        pages = [list(pages)]                          # type: ignore[list-item]
    pages = list(pages) or [[]]                        # type: ignore[assignment]
    objs: list[bytes] = []
    n_pages = len(pages)
    page_ids = [4 + 2 * i for i in range(n_pages)]
    objs.append(b"<< /Type /Catalog /Pages 2 0 R " + catalog_extra + b">>")
    objs.append(("<< /Type /Pages /Kids [" + " ".join(f"{i} 0 R" for i in page_ids) + f"] /Count {n_pages} >>").encode())
    objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")
    for i, lines in enumerate(pages):
        content = ["BT /F1 11 Tf 14 TL 50 780 Td"]
        for ln in lines:
            esc = ln.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            content.append(f"({esc}) Tj T*")
        content.append("ET")
        stream = ("\n".join(content) + "\n" + " " * pad).encode("latin-1", "replace")
        pid, cid = page_ids[i], page_ids[i] + 1
        objs.append((f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >> "
                     f"/Contents {cid} 0 R >>").encode())
        objs.append(f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream")
    if extra_objects:
        objs.append(extra_objects)
    out = io.BytesIO()
    out.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for n, body in enumerate(objs, start=1):
        offsets.append(out.tell())
        out.write(f"{n} 0 obj\n".encode() + body + b"\nendobj\n")
    xref = out.tell()
    out.write(f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode())
    for off in offsets:
        out.write(f"{off:010d} 00000 n \n".encode())
    out.write(f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
    return out.getvalue()


def paper_pdf(title: str, authors: list[str], extra: list[str] = (), doi: str = "", pages: int = 2) -> bytes:
    first = [title, ", ".join(authors), "Abstract", *extra]
    if doi:
        first.append(f"https://doi.org/{doi}")
    body = [["Introduction", "Lorem ipsum dolor sit amet consectetur adipiscing elit " * 2] for _ in range(pages - 1)]
    return make_pdf([first, *body])


# ----------------------------------------------------------------------------- fake web
Body = Union[bytes, str]


class FakeWeb(BaseAdapter):
    """Scripted responses keyed by URL (query strings ignored unless the route contains '?')."""

    def __init__(self):
        super().__init__()
        self.routes: dict[str, list] = {}
        self.calls: list[str] = []

    def add(self, url: str, body: Body = b"", status: int = 200, content_type: str = "text/html",
            headers: Optional[dict] = None, repeat: bool = True) -> "FakeWeb":
        h = {"Content-Type": content_type, **(headers or {})}
        b = body.encode() if isinstance(body, str) else body
        self.routes.setdefault(url, []).append([status, h, b, repeat])
        return self

    def add_pdf(self, url: str, pdf: bytes, **kw) -> "FakeWeb":
        return self.add(url, pdf, content_type="application/pdf", **kw)

    def redirect(self, url: str, to: str, status: int = 302) -> "FakeWeb":
        return self.add(url, b"", status=status, headers={"Location": to})

    def hits(self, fragment: str) -> int:
        return sum(1 for c in self.calls if fragment in c)

    def _lookup(self, url: str):
        for key in (url, url.split("?")[0]):
            if key in self.routes:
                return self.routes[key]
        return None

    def send(self, request, stream=False, timeout=None, verify=True, cert=None, proxies=None):
        url = request.url
        self.calls.append(url)
        queue = self._lookup(url)
        if queue is None:
            spec = [404, {"Content-Type": "text/html"}, b"<html><title>Not found</title><body>not found</body></html>", True]
        else:
            spec = queue[0] if (queue[0][3] or len(queue) == 1) else queue.pop(0)
        status, headers, body, _ = spec
        raw = HTTPResponse(body=io.BytesIO(body), headers={**headers, "Content-Length": headers.get("Content-Length", str(len(body)))},
                           status=status, preload_content=False, decode_content=True)
        resp = requests.Response()
        resp.status_code = status
        resp.headers = CaseInsensitiveDict(raw.headers)
        resp.raw = raw
        resp.url = url
        resp.request = request
        resp.encoding = "utf-8"
        return resp

    def close(self):
        pass


@pytest.fixture
def web() -> FakeWeb:
    return FakeWeb()


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings.from_env({"CONTACT_EMAIL": "tester@university.edu"}).replace(
        download_dir=tmp_path / "Downloaded_Papers", db_path=tmp_path / "test.sqlite3", host_min_interval=0.0,
        host_intervals={}, max_retries=2, time_limit_seconds=60.0, auto_download=True)


def make_client(settings: Settings, web: FakeWeb) -> HttpClient:
    session = requests.Session()
    session.trust_env = False
    session.mount("http://", web)
    session.mount("https://", web)
    return HttpClient(settings, session=session, resolver=lambda host, port: [PUBLIC_IP], sleeper=lambda s: None)


@pytest.fixture
def client(settings, web) -> HttpClient:
    return make_client(settings, web)


@pytest.fixture
def db(tmp_path) -> Database:
    d = Database(tmp_path / "t.sqlite3")
    yield d
    d.close()


# ----------------------------------------------------------------------------- fake providers
class FakeProvider(SearchProvider):
    """Returns canned Papers; every method can be told to fail."""

    def __init__(self, http, settings, name="fake", papers=(), by_doi=None, fulltext=None, fail: Optional[Exception] = None,
                 relevance_value=1.0, doi_lookup=True):
        super().__init__(http, settings)
        self.name, self.label = name, name
        self._papers, self._by_doi, self._fulltext, self._fail = list(papers), by_doi or {}, fulltext or {}, fail
        self._rel = relevance_value
        self.supports_doi_lookup = doi_lookup
        self.supports_fulltext_lookup = bool(fulltext)
        self.search_calls: list[str] = []

    def relevance(self, qi):
        return self._rel

    def search(self, query, limit=8):
        self.search_calls.append(query)
        if self._fail:
            raise self._fail
        return self._tag(_clone(p) for p in self._papers)

    def lookup_doi(self, doi):
        if self._fail:
            raise self._fail
        p = self._by_doi.get(doi) or next((x for x in self._papers if x.doi == doi), None)
        return self._tag([_clone(p)])[0] if p else None

    def find_fulltext(self, paper):
        if self._fail:
            raise self._fail
        return list(self._fulltext.get(paper.doi, []))


class FakeWebSearch(WebSearchProvider):
    def __init__(self, http, settings, hits_by_query=None, default_hits=()):
        super().__init__(http, settings)
        self.name, self.label = "web:fake", "Fake web search"
        self.queries: list[str] = []
        self._by_query, self._default = hits_by_query or {}, list(default_hits)

    def search_web(self, query, limit=8):
        self.queries.append(query)
        return list(self._by_query.get(query, self._default))


def _clone(p: Paper) -> Paper:
    import copy
    return copy.deepcopy(p)


def registry(*providers, web=()):
    return ProviderRegistry(providers, web)


def loc(url, kind="pdf", version=VersionType.UNKNOWN, host_type="repository", provider="fake", is_oa=True, **kw):
    return FullTextLocation(url=url, kind=kind, version=version, host_type=host_type, provider=provider, is_oa=is_oa, **kw)


TITLE = "The Impact of Interest Rates on Peer-to-Peer Lending Default Risk"
AUTHORS = ["Jane Smith", "Carlos Ortega", "Wei Zhang"]
DOI = "10.1234/p2p.2024.001"
