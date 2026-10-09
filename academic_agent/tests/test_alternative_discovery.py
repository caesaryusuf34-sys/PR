"""The agent keeps investigating when the first source fails, and finds alternative versions."""
from models import AccessState, Outcome, VersionType
from tests.agent_helpers import PAYWALL_HTML, base_paper, build_agent, good_pdf, one_provider
from tests.conftest import AUTHORS, DOI, TITLE, FakeProvider, FakeWeb, FakeWebSearch, loc, paper_pdf

PUBLISHER_PDF = "https://publisher.example/article/1.pdf"
REPO_PDF = "https://repo.univ.edu/bitstream/handle/1/paper.pdf"


def statuses(result):
    return [(a.url, a.status) for a in result.attempts]


def test_paywalled_first_source_then_repository_copy_is_downloaded(settings, web):
    paper = base_paper(locations=[loc(PUBLISHER_PDF, "pdf", VersionType.PUBLISHED, "publisher", "primary")])
    finder = lambda c, s: [FakeProvider(c, s, name="primary", papers=[paper]),
                           FakeProvider(c, s, name="oa_finder", doi_lookup=False,
                                        fulltext={DOI: [loc(REPO_PDF, "pdf", VersionType.ACCEPTED, "institutional", "oa_finder")]})]
    web.add(PUBLISHER_PDF, PAYWALL_HTML)                                       # "OA" flag was stale: really paywalled
    web.add("https://doi.org/" + DOI, PAYWALL_HTML)
    web.add_pdf(REPO_PDF, good_pdf("Accepted manuscript - author version"))
    result = build_agent(settings, web, finder).run(TITLE)

    assert result.outcome == Outcome.DOWNLOADED and result.access_state == AccessState.ACCEPTED
    assert result.download.version == VersionType.ACCEPTED and result.download.source_url == REPO_PDF
    assert result.download.verification["verdict"] == "verified"
    saved = list(settings.download_dir.glob("*.pdf"))
    assert len(saved) == 1 and saved[0].name.endswith("_accepted_manuscript.pdf") and saved[0].read_bytes()[:5] == b"%PDF-"
    order = [u for u, _ in statuses(result)]
    assert order.index(PUBLISHER_PDF) < order.index(REPO_PDF)                  # authority order: publisher first
    assert dict(statuses(result))[PUBLISHER_PDF] == "paywall"
    assert "accepted manuscript" in result.summary.lower() and "not** the final published" in result.summary


def test_wrong_document_at_first_location_is_rejected_and_search_continues(settings, web):
    other = "https://repo.a.edu/other.pdf"
    paper = base_paper(locations=[loc(other, "pdf", VersionType.PUBLISHED, "institutional", "primary"),
                                  loc(REPO_PDF, "pdf", VersionType.UNKNOWN, "repository", "primary")])
    web.add_pdf(other, paper_pdf("Credit Scoring Models for Marketplace Lending", ["Anna Brown", "Li Wei"], pages=3))
    web.add_pdf(REPO_PDF, good_pdf())
    web.add("https://doi.org/" + DOI, PAYWALL_HTML)
    result = build_agent(settings, web, one_provider(paper)).run(DOI)
    assert result.outcome == Outcome.DOWNLOADED and result.download.source_url == REPO_PDF
    rejected = [a for a in result.attempts if a.status == "rejected_after_verification"]
    assert len(rejected) == 1 and rejected[0].url == other and "does not match" in rejected[0].detail
    assert len(list(settings.download_dir.glob("*.pdf"))) == 1 and not list(settings.download_dir.rglob("*.part"))


def test_published_open_access_version_is_preferred_over_preprint(settings, web):
    pre = "https://arxiv.org/pdf/2201.00001"
    paper = base_paper(locations=[loc(pre, "pdf", VersionType.PREPRINT, "preprint_server", "arxiv"),
                                  loc(PUBLISHER_PDF, "pdf", VersionType.PUBLISHED, "publisher", "primary")])
    web.add_pdf(PUBLISHER_PDF, good_pdf("Version of Record published online")).add_pdf(pre, good_pdf("Preprint submitted"))
    result = build_agent(settings, web, one_provider(paper)).run(TITLE)
    assert result.download.version == VersionType.PUBLISHED and result.access_state == AccessState.PUBLISHED
    assert web.hits("arxiv.org/pdf") == 0                                       # never even requested


def test_preprint_is_labelled_and_can_be_disabled(settings, web):
    pre = "https://arxiv.org/pdf/2201.00001"
    paper = base_paper(locations=[loc(pre, "pdf", VersionType.PREPRINT, "preprint_server", "arxiv")])
    web.add_pdf(pre, good_pdf("Preprint submitted to a journal"))
    web.add("https://doi.org/" + DOI, PAYWALL_HTML)
    r = build_agent(settings, web, one_provider(paper)).run(TITLE)
    assert r.access_state == AccessState.PREPRINT and r.download.path.endswith("_preprint.pdf")
    assert "not** the final published" in r.summary

    off = build_agent(settings.replace(allow_preprints=False, download_dir=settings.download_dir / "off",
                                       db_path=settings.db_path.with_name("off.sqlite3")), web, one_provider(paper)).run(TITLE)
    assert off.outcome == Outcome.NO_FULLTEXT and off.download is None


def test_version_stated_by_provider_is_corrected_by_the_document_itself(settings, web):
    paper = base_paper(locations=[loc(REPO_PDF, "pdf", VersionType.PUBLISHED, "institutional", "primary")])
    web.add_pdf(REPO_PDF, good_pdf("Author accepted manuscript - post-print"))
    r = build_agent(settings, web, one_provider(paper)).run(TITLE)
    assert r.download.version == VersionType.ACCEPTED


# --------------------------------------------------------------------------- following links with a bounded depth
def record_pages(web):
    web.add("https://repo.example.edu/record/1",
            f"<html><head><meta name='citation_title' content='{TITLE}'></head><body><h1>{TITLE}</h1>"
            "<p>Institutional repository record. " + "text " * 50 + "</p><a href='/record/1/details'>View full record</a></body></html>")
    web.add("https://repo.example.edu/record/1/details",
            f"<html><head><meta name='citation_title' content='{TITLE}'><meta name='citation_pdf_url' "
            "content='https://repo.example.edu/files/1/paper.pdf'></head><body><h1>" + TITLE + "</h1>" + "text " * 50 + "</body></html>")
    web.add_pdf("https://repo.example.edu/files/1/paper.pdf", good_pdf("Accepted manuscript"))


def test_crawler_follows_a_relevant_link_to_the_pdf(settings, web):
    record_pages(web)
    paper = base_paper(locations=[loc("https://repo.example.edu/record/1", "landing", VersionType.UNKNOWN, "institutional", "primary")])
    r = build_agent(settings.replace(max_depth=2), web, one_provider(paper)).run(DOI)
    assert r.outcome == Outcome.DOWNLOADED and r.download.source_url.endswith("/files/1/paper.pdf")
    assert [p["url"] for p in r.pages_visited][:2] == ["https://repo.example.edu/record/1", "https://repo.example.edu/record/1/details"]


def test_depth_and_page_budgets_stop_the_search(settings, web):
    record_pages(web)
    paper = base_paper(locations=[loc("https://repo.example.edu/record/1", "landing", VersionType.UNKNOWN, "institutional", "primary")])
    for kw in ({"max_depth": 0}, {"max_pages": 1}):
        s = settings.replace(download_dir=settings.download_dir / str(kw), db_path=settings.db_path.with_name(f"{len(str(kw))}.sqlite3"), **kw)
        r = build_agent(s, web, one_provider(paper)).run(DOI)
        assert r.outcome == Outcome.NO_FULLTEXT and r.download is None


def test_pages_that_do_not_match_the_paper_are_not_followed(settings, web):
    web.add("https://some.site/page", "<html><head><meta name='citation_title' content='Totally different topic'></head><body>"
            "<h1>Totally different topic</h1>" + "x " * 300 + "<a href='/p.pdf'>PDF</a></body></html>")
    web.add_pdf("https://some.site/p.pdf", good_pdf())
    paper = base_paper(locations=[loc("https://some.site/page", "landing", VersionType.UNKNOWN, "web", "web:fake", trusted=False)])
    r = build_agent(settings, web, one_provider(paper)).run(DOI)
    assert r.outcome == Outcome.NO_FULLTEXT and web.hits("some.site/p.pdf") == 0
    assert any(a.status == "page_mismatch" for a in r.attempts)


# --------------------------------------------------------------------------- step 3: alternative versions
def test_query_variants_follow_the_specification(settings, web):
    agent = build_agent(settings, web, one_provider(base_paper()))
    v = agent.query_variants(base_paper())
    assert f'"{TITLE}"' in v and f'"{TITLE}" Smith' in v and DOI in v
    for term in ("PDF", "open access", "repository", "accepted manuscript", "full text"):
        assert f'"{TITLE}" {term}' in v


def test_web_search_finds_a_repository_copy_when_publisher_is_closed(settings, web):
    paper = base_paper(locations=[])
    web.add("https://doi.org/" + DOI, PAYWALL_HTML)
    eprints = "https://eprints.example.ac.uk/123/1/paper.pdf"
    web.add_pdf(eprints, good_pdf("Accepted manuscript"))
    web.add("https://sci-hub.se/" + DOI, "pirated")
    hits = [
        _hit("Unrelated chemistry paper", "https://chem.example/paper.pdf", "about soil"),
        _hit(TITLE, "https://sci-hub.se/" + DOI, "pirated copy"),
        _hit(TITLE + " - ePrints", eprints, "accepted manuscript pdf"),
    ]
    holder = {}

    def provs(c, s):
        holder["p"] = FakeProvider(c, s, name="primary", papers=[paper])
        return [holder["p"]]

    def webs(c, s):
        holder["w"] = FakeWebSearch(c, s, default_hits=hits)
        return [holder["w"]]

    r = build_agent(settings, web, provs, webs).run(TITLE)
    assert r.outcome == Outcome.DOWNLOADED and r.download.source_url == eprints
    assert r.access_state == AccessState.ACCEPTED
    q = holder["w"].queries
    assert f'"{TITLE}"' in q and f'"{TITLE}" Smith' in q and f'"{TITLE}" accepted manuscript' in q and len(q) <= settings.max_search_queries
    assert any(TITLE in s and "Smith" in s for s in holder["p"].search_calls[1:])      # relaxed re-query with first author
    assert web.hits("sci-hub") == 0 and web.hits("chem.example") == 0                    # blocked / irrelevant never contacted


def test_missing_web_search_is_reported_not_hidden(settings, web):
    web.add("https://doi.org/" + DOI, PAYWALL_HTML)
    r = build_agent(settings, web, one_provider(base_paper(locations=[]))).run(TITLE)
    assert r.outcome == Outcome.NO_FULLTEXT
    assert any("No web-search API is configured" in e.message for e in r.events)


def test_alternative_provider_records_contribute_locations(settings, web):
    """A repository-indexed record (found only by the relaxed re-query) supplies the PDF."""
    primary = base_paper(locations=[])
    mirror = base_paper(locations=[loc(REPO_PDF, "pdf", VersionType.UNKNOWN, "repository", "mirror")])
    web.add("https://doi.org/" + DOI, PAYWALL_HTML)
    web.add_pdf(REPO_PDF, good_pdf())

    class Late(FakeProvider):
        def search(self, query, limit=8):
            self.search_calls.append(query)
            return self._tag([__import__("copy").deepcopy(mirror)]) if len(self.search_calls) > 1 else []

    r = build_agent(settings, web, lambda c, s: [FakeProvider(c, s, name="primary", papers=[primary]), Late(c, s, name="mirror")]).run(TITLE)
    assert r.outcome == Outcome.DOWNLOADED and r.download.source_url == REPO_PDF


# --------------------------------------------------------------------------- bounded effort
def test_request_budget_is_enforced(settings, web):
    locs = [loc(f"https://r{i}.example/p.pdf", "pdf", VersionType.UNKNOWN, "repository", "primary") for i in range(30)]
    for l in locs:
        web.add(l.url, PAYWALL_HTML)
    s = settings.replace(max_requests=12)
    r = build_agent(s, web, one_provider(base_paper(locations=locs))).run(TITLE)
    assert r.outcome == Outcome.NO_FULLTEXT and len(web.calls) <= 12


def test_candidate_cap_is_enforced(settings, web):
    locs = [loc(f"https://r{i}.example/p.pdf", "pdf", VersionType.UNKNOWN, "repository", "primary") for i in range(30)]
    for l in locs:
        web.add(l.url, "gone", status=404)
    r = build_agent(settings.replace(max_candidates=5), web, one_provider(base_paper(locations=locs))).run(TITLE)
    assert r.outcome == Outcome.NO_FULLTEXT and sum(1 for a in r.attempts if a.stage == "probe") <= 5


def _hit(title, url, snippet):
    from search_providers import WebHit
    return WebHit(title, url, snippet, "web:fake")


# --------------------------------------------------------------------------- link-only mode (auto_download off)
def test_link_only_mode_returns_a_verified_link_and_saves_nothing(settings, web):
    s = settings.replace(auto_download=False)
    paper = base_paper(locations=[loc(REPO_PDF, "pdf", VersionType.ACCEPTED, "institutional", "primary")])
    web.add_pdf(REPO_PDF, good_pdf("Accepted manuscript"))
    agent = build_agent(s, web, one_provider(paper))
    r = agent.run(TITLE)
    assert r.outcome == Outcome.LINK_FOUND and r.access_state == AccessState.ACCEPTED
    assert r.download.saved is False and r.download.source_url == REPO_PDF and r.download.path == ""
    assert r.download.verification["verdict"] == "verified"
    assert not list(s.download_dir.rglob("*.pdf")) and not list(s.download_dir.rglob("*.part"))
    assert any(c.url == REPO_PDF for c in r.citations) and "Nothing was saved" in r.summary
    assert agent.db.history(TITLE[:15])[0]["source_url"] == REPO_PDF


def test_link_only_mode_still_rejects_the_wrong_document(settings, web):
    s = settings.replace(auto_download=False)
    wrong = "https://repo.a.edu/other.pdf"
    paper = base_paper(locations=[loc(wrong, "pdf", VersionType.PUBLISHED, "institutional", "primary")])
    web.add_pdf(wrong, paper_pdf("Credit Scoring Models for Marketplace Lending", ["Anna Brown", "Li Wei"], pages=3))
    web.add("https://doi.org/" + DOI, PAYWALL_HTML)
    r = build_agent(s, web, one_provider(paper)).run(DOI)
    assert r.outcome == Outcome.NO_FULLTEXT and r.download is None      # an unverified link is never offered
