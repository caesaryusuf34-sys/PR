"""Provider parsing (canned API payloads), selection by query type, HTTP resilience."""
import json

import pytest

from http_client import RateLimited, RequestFailed
from models import VersionType
from paper_match import classify_query
from search_providers import (ArxivProvider, BraveSearchProvider, CoreProvider, CrossrefProvider, DoajProvider,
                              EuropePmcProvider, GoogleCseProvider, OpenAireProvider, OpenAlexProvider, ProviderError,
                              ProviderRegistry, SemanticScholarProvider, UnpaywallProvider)
from tests.conftest import FakeWeb, make_client

CROSSREF_ITEM = {
    "DOI": "10.1234/ABC.1", "title": ["Rates &amp; <i>Default</i> in P2P Lending"], "type": "journal-article",
    "author": [{"given": "Jane", "family": "Smith"}, {"name": "Consortium X"}], "issued": {"date-parts": [[2023, 5]]},
    "container-title": ["J. Fintech"], "abstract": "<jats:p>We study things.</jats:p>", "is-referenced-by-count": 7,
    "license": [{"URL": "https://creativecommons.org/licenses/by/4.0/"}],
    "relation": {"has-preprint": [{"id-type": "doi", "id": "10.48550/arXiv.2201.00001"}]},
    "link": [{"URL": "https://pub.example/a.pdf", "content-type": "application/pdf", "content-version": "vor"},
             {"URL": "https://pub.example/a.html", "content-type": "text/html"}]}


def test_crossref_parsing():
    p = CrossrefProvider.parse_item(CROSSREF_ITEM)
    assert p.title == "Rates & Default in P2P Lending" and p.doi == "10.1234/abc.1" and p.year == 2023
    assert p.authors == ["Jane Smith", "Consortium X"] and p.abstract == "We study things." and p.is_oa
    assert p.related_dois == ["10.48550/arxiv.2201.00001"]
    assert [l.url for l in p.locations] == ["https://pub.example/a.pdf"] and p.locations[0].version == VersionType.PUBLISHED


def test_openalex_parsing():
    w = {"id": "https://openalex.org/W1", "doi": "https://doi.org/10.1234/abc.1", "title": "T", "publication_year": 2022,
         "authorships": [{"author": {"display_name": "Jane Smith"}}], "open_access": {"is_oa": True},
         "abstract_inverted_index": {"We": [0], "study": [1], "rates": [2]}, "ids": {"pmid": "https://pubmed.ncbi.nlm.nih.gov/123"},
         "primary_location": {"source": {"display_name": "J", "type": "journal"}},
         "best_oa_location": {"pdf_url": "https://repo.univ.edu/x.pdf", "landing_page_url": "https://repo.univ.edu/x", "is_oa": True,
                              "version": "acceptedVersion", "license": "cc-by", "source": {"display_name": "Univ Repo", "type": "repository"}},
         "locations": [{"landing_page_url": "https://closed.example/x", "is_oa": False, "version": "publishedVersion"}]}
    p = OpenAlexProvider.parse_work(w)
    assert p.doi == "10.1234/abc.1" and p.abstract == "We study rates" and p.pmid == "123"
    pdf = [l for l in p.locations if l.kind == "pdf"][0]
    assert pdf.version == VersionType.ACCEPTED and pdf.host_type in ("institutional", "repository") and pdf.repository == "Univ Repo"
    assert all("closed.example" not in l.url for l in p.locations)           # closed landing pages are not offered as OA


def test_semantic_scholar_parsing():
    p = SemanticScholarProvider.parse_paper({"paperId": "abc", "title": "T", "year": 2020, "authors": [{"name": "A B"}],
                                             "externalIds": {"DOI": "10.1/X", "ArXiv": "1706.03762", "PubMedCentral": "55"},
                                             "openAccessPdf": {"url": "https://arxiv.org/pdf/1706.03762", "status": "GREEN"}, "isOpenAccess": True})
    assert p.arxiv_id == "1706.03762" and p.pmcid == "PMC55" and p.locations[0].host_type == "preprint_server"


def test_unpaywall_parsing_distinguishes_versions_and_hosts():
    d = {"doi": "10.1234/abc.1", "title": "T", "year": 2021, "is_oa": True, "z_authors": [{"given": "J", "family": "Smith"}],
         "oa_locations": [
             {"url_for_pdf": "https://pub.example/oa.pdf", "url_for_landing_page": "https://pub.example/oa", "host_type": "publisher", "version": "publishedVersion", "license": "cc-by"},
             {"url_for_pdf": None, "url_for_landing_page": "https://repo.univ.edu/1", "url": "https://repo.univ.edu/1", "host_type": "repository",
              "version": "acceptedVersion", "repository_institution": "Univ"}]}
    p = UnpaywallProvider.parse_record(d)
    kinds = {(l.url, l.kind, l.version, l.host_type) for l in p.locations}
    assert ("https://pub.example/oa.pdf", "pdf", VersionType.PUBLISHED, "publisher") in kinds
    assert ("https://repo.univ.edu/1", "landing", VersionType.ACCEPTED, "institutional") in kinds
    assert len([l for l in p.locations if l.url == "https://repo.univ.edu/1"]) == 1


def test_core_europepmc_doaj_openaire_parsing():
    c = CoreProvider.parse_work({"id": 9, "title": "T", "authors": [{"name": "A B"}], "yearPublished": 2019, "doi": "10.1/y1234", "downloadUrl": "https://core.ac.uk/download/pdf/9.pdf"})
    assert c.locations[0].url.endswith("9.pdf") and c.locations[0].kind == "pdf"
    e = EuropePmcProvider.parse_result({"id": "1", "source": "MED", "title": "Title.", "authorString": "Smith J, Doe AB.", "pubYear": "2018", "pmcid": "PMC1",
                                         "isOpenAccess": "Y", "fullTextUrlList": {"fullTextUrl": [
                                             {"availabilityCode": "OA", "documentStyle": "pdf", "url": "https://europepmc.org/a.pdf", "site": "Europe_PMC"},
                                             {"availabilityCode": "S", "documentStyle": "html", "url": "https://paywalled.example/x", "site": "DOI"}]}})
    assert e.title == "Title" and e.authors == ["J Smith", "AB Doe"]
    assert [l.url for l in e.locations if l.kind == "pdf"][0] == "https://europepmc.org/a.pdf" and all("paywalled" not in l.url for l in e.locations)
    pre = EuropePmcProvider.parse_result({"id": "2", "source": "PPR", "title": "P", "isOpenAccess": "Y", "pmcid": "PMC2"})
    assert pre.locations[0].version == VersionType.PREPRINT
    d = DoajProvider.parse_article({"id": "z", "bibjson": {"title": "T", "year": "2020", "author": [{"name": "A B"}], "identifier": [{"type": "doi", "id": "10.1/zzzz"}],
                                                          "link": [{"type": "fulltext", "url": "https://j.example/a", "content_type": "HTML"}], "journal": {"title": "J"}}})
    assert d.locations[0].version == VersionType.PUBLISHED and d.venue == "J"
    o = OpenAireProvider.parse_product({"id": "x", "mainTitle": "T", "publicationDate": "2020-01-01", "authors": [{"fullName": "A B"}],
                                        "pids": [{"scheme": "doi", "value": "10.1234/q1234"}],
                                        "instances": [{"accessRight": {"code": "c_abf2"}, "type": "Preprint", "urls": ["https://repo.example/x.pdf", "https://doi.org/10.1234/q1234"], "hostedBy": {"name": "Repo"}},
                                                      {"accessRight": {"code": "c_16ec"}, "urls": ["https://closed.example/x"]}]})
    assert [l.url for l in o.locations] == ["https://repo.example/x.pdf"] and o.locations[0].version == VersionType.PREPRINT


ARXIV_FEED = b"""<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
<entry><id>http://arxiv.org/abs/1706.03762v7</id><published>2017-06-12T17:57:34Z</published><title>Attention Is
 All You Need</title><summary>The dominant sequence.</summary><author><name>Ashish Vaswani</name></author><author><name>Noam Shazeer</name></author>
<arxiv:journal_ref>NeurIPS 2017</arxiv:journal_ref></entry></feed>"""


def test_arxiv_parsing_and_hostile_xml():
    p = ArxivProvider.parse_feed(ARXIV_FEED)[0]
    assert p.title == "Attention Is All You Need" and p.arxiv_id == "1706.03762" and p.year == 2017 and p.venue == "NeurIPS 2017"
    assert p.locations[0].url == "https://arxiv.org/pdf/1706.03762" and p.locations[0].version == VersionType.PREPRINT
    bomb = b'<?xml version="1.0"?><!DOCTYPE lolz [<!ENTITY lol "lol"><!ENTITY lol2 "&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;">]><feed xmlns="http://www.w3.org/2005/Atom"><entry>&lol2;</entry></feed>'
    with pytest.raises(ProviderError):
        ArxivProvider.parse_feed(bomb)


def test_registry_chooses_providers_by_query_and_credentials(settings, web):
    c = make_client(settings, web)
    reg = ProviderRegistry.default(c, settings)
    names = lambda q, **k: [p.name for p in reg.select(classify_query(q), **k)[0]]
    bio = names("Efficacy of vaccine therapy in cancer patients: a randomized clinical trial")
    quant = names("The Impact of Interest Rates on Peer-to-Peer Lending Default Risk")
    assert bio.index("europepmc") < bio.index("crossref") or "europepmc" in bio[:3]
    assert "arxiv" not in bio and "arxiv" in quant                              # arXiv only where it is likely to help
    assert "core" not in quant                                                  # no API key -> skipped, with a reason
    assert "set CORE_API_KEY" in reg.select(classify_query("x y z w v u"))[1]["core"]
    assert "arxiv" in names("Efficacy of vaccine therapy in cancer patients: a randomized clinical trial", include_low_relevance=True)
    keyed = ProviderRegistry.default(c, settings.replace(core_api_key="k"))
    assert "core" in [p.name for p in keyed.select(classify_query("a b c d e f"))[0]]
    assert not reg.available_web() and ProviderRegistry.default(c, settings.replace(brave_api_key="k")).available_web()


def test_web_search_providers_parse_results(settings, web):
    s = settings.replace(brave_api_key="b", google_cse_api_key="g", google_cse_cx="cx")
    c = make_client(s, web)
    web.add("https://api.search.brave.com/res/v1/web/search", json.dumps({"web": {"results": [{"title": "T", "url": "https://r.example/x.pdf", "description": "d"}]}}), content_type="application/json")
    web.add("https://www.googleapis.com/customsearch/v1", json.dumps({"items": [{"title": "T2", "link": "https://r2.example/y.pdf", "snippet": "s"}]}), content_type="application/json")
    assert BraveSearchProvider(c, s).search_web("q")[0].url == "https://r.example/x.pdf"
    assert GoogleCseProvider(c, s).search_web("q")[0].url == "https://r2.example/y.pdf"
    assert not BraveSearchProvider(make_client(settings, web), settings).available()[0]


def test_provider_end_to_end_through_the_http_layer(settings, web):
    web.add("https://api.crossref.org/works", json.dumps({"message": {"items": [CROSSREF_ITEM]}}), content_type="application/json")
    c = make_client(settings, web)
    res = CrossrefProvider(c, settings).search("Rates and Default in P2P Lending")
    assert res[0].sources == ["crossref"] and res[0].provenance[0]["provider"] == "crossref"
    assert "mailto=tester%40university.edu" in web.calls[0]                      # polite-pool identification


def test_rate_limit_quota_and_errors(settings, web):
    web.add("https://api.crossref.org/works/10.1234/q1234", "slow down", status=429)
    web.add("https://api.openalex.org/works", "oops", status=500)
    web.add("https://api.unpaywall.org/v2/10.1234/q1234", "{not json", content_type="application/json")
    c = make_client(settings, web)
    with pytest.raises(RateLimited):
        CrossrefProvider(c, settings).lookup_doi("10.1234/q1234")
    with pytest.raises(RateLimited):                                              # cool-down: no further requests to that host
        CrossrefProvider(c, settings).lookup_doi("10.1234/q1234")
    assert web.hits("api.crossref.org") == settings.max_retries + 1
    with pytest.raises(RequestFailed):
        OpenAlexProvider(c, settings).search("x")
    assert web.hits("api.openalex.org") == settings.max_retries + 1               # bounded retries
    with pytest.raises(RequestFailed):
        UnpaywallProvider(c, settings).lookup_doi("10.1234/q1234")


def test_unknown_doi_404_is_none_not_error(settings, web):
    assert CrossrefProvider(make_client(settings, web), settings).lookup_doi("10.1/nope") is None


def test_oversized_json_is_refused(settings, web):
    s = settings.replace(max_json_mb=1)
    web.add("https://api.crossref.org/works", b"{" + b" " * (2 * 1024 * 1024) + b"}", content_type="application/json", headers={"Content-Length": ""})
    with pytest.raises(Exception):
        CrossrefProvider(make_client(s, web), s).search("x")


def test_doaj_query_strips_lucene_special_characters(settings, web):
    web.add("https://doaj.org/api/search/articles/", json.dumps({"results": []}), content_type="application/json")
    c = make_client(settings, web)
    DoajProvider(c, settings).search("Does fintech threaten Islamic banking performance in Indonesia? (a study: 2023)")
    path = web.calls[0].split("?")[0].split("/articles/")[1]
    assert not any(tok in path for tok in ("%3F", "%28", "%29")) and path.count("%3A") == 1      # only the field separator remains
