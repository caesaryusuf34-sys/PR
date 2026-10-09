"""Honest failure: inaccessible papers, unknown papers, ambiguity, provider outages - never an invented PDF."""
import re

from http_client import RateLimited
from models import AccessState, Outcome, Paper, QueryType, VersionType
from tests.agent_helpers import ABSTRACT, PAYWALL_HTML, base_paper, build_agent, good_pdf, one_provider
from tests.conftest import AUTHORS, DOI, TITLE, FakeProvider, FakeWeb, loc


LOGIN_HTML = "<html><title>Sign in</title><body><form><input type='password'></form>Please sign in to access this content</body></html>"
CAPTCHA_HTML = "<html><title>Just a moment...</title><body>Checking your browser before accessing</body></html>"


def inaccessible_paper():
    return base_paper(locations=[
        loc("https://pub-a.example/a.pdf", "pdf", VersionType.PUBLISHED, "publisher", "primary", is_oa=True),
        loc("https://pub-b.example/b.pdf", "pdf", VersionType.PUBLISHED, "publisher", "primary", is_oa=True),
        loc("https://pub-c.example/c.pdf", "pdf", VersionType.UNKNOWN, "repository", "primary"),
        loc("https://pub-d.example/d.pdf", "pdf", VersionType.UNKNOWN, "repository", "primary"),
        loc("https://pub-e.example/e.pdf", "pdf", VersionType.UNKNOWN, "repository", "primary"),
    ])


def closed_web():
    return (FakeWeb().add("https://pub-a.example/a.pdf", PAYWALL_HTML)
            .redirect("https://pub-b.example/b.pdf", "https://login.pub-b.example/auth/login").add("https://login.pub-b.example/auth/login", LOGIN_HTML)
            .add("https://pub-c.example/c.pdf", CAPTCHA_HTML, status=403)
            .add("https://pub-d.example/d.pdf", "gone", status=404)
            .add("https://pub-e.example/e.pdf", "denied", status=403)
            .add("https://doi.org/" + DOI, PAYWALL_HTML))


def test_inaccessible_paper_returns_verified_record_and_official_link_but_no_file(settings):
    web = closed_web()
    r = build_agent(settings, web, one_provider(inaccessible_paper())).run(TITLE)

    assert r.outcome == Outcome.NO_FULLTEXT and r.download is None
    assert r.access_state == AccessState.INACCESSIBLE
    assert r.paper.title == TITLE and r.paper.doi == DOI                         # strongest verified record is returned
    assert r.official_link == f"https://doi.org/{DOI}"
    assert not list(settings.download_dir.rglob("*.pdf")) and not list(settings.download_dir.rglob("*.part"))

    by_url = {a.url: a.status for a in r.attempts}
    assert by_url["https://pub-a.example/a.pdf"] == "paywall"
    assert by_url["https://pub-b.example/b.pdf"] == "login_required"
    assert by_url["https://pub-c.example/c.pdf"] == "captcha"
    assert by_url["https://pub-d.example/d.pdf"] == "not_found"
    assert by_url["https://pub-e.example/e.pdf"] == "forbidden"

    assert "No legally accessible full text" in r.summary and "paywall" in r.summary
    # nothing is fabricated: every URL the result mentions was either supplied by a source or is the DOI link
    known = {l.url for l in inaccessible_paper().locations} | {f"https://doi.org/{DOI}"}
    mentioned = set(re.findall(r"https?://[^\s)\]`]+", r.summary)) | {c.url for c in r.citations} | {a.url for a in r.attempts}
    assert all(u.rstrip(".,") in known or "doi.org" in u or "login.pub-b" in u for u in mentioned), mentioned - known


def test_no_login_or_captcha_bypass_attempted(settings):
    web = closed_web()
    build_agent(settings, web, one_provider(inaccessible_paper())).run(TITLE)
    # each blocked URL is requested a bounded number of times and nothing is re-requested with tricks
    for u in ("pub-a.example/a.pdf", "pub-b.example/b.pdf", "pub-c.example/c.pdf"):
        assert web.hits(u) <= 4


def test_run_is_recorded_in_history_with_attempts(settings):
    agent = build_agent(settings, closed_web(), one_provider(inaccessible_paper()))
    r = agent.run(TITLE)
    rows = agent.db.history(TITLE[:20])
    assert rows and rows[0]["outcome"] == "no_fulltext" and rows[0]["file_path"] is None
    assert len(agent.db.get_attempts(r.run_id)) == len(r.attempts)


def test_abstract_only_when_no_restriction_was_observed(settings):
    paper = base_paper(locations=[])                           # record + abstract, but no known copy anywhere
    web = FakeWeb()                                            # DOI page 404 -> nothing paywalled either
    r = build_agent(settings, web, one_provider(paper)).run(TITLE)
    assert r.outcome == Outcome.NO_FULLTEXT and r.access_state == AccessState.ABSTRACT_ONLY


def test_not_found(settings):
    r = build_agent(settings, FakeWeb(), lambda c, s: [FakeProvider(c, s, name="empty")]).run("A Paper That Does Not Exist Anywhere At All")
    assert r.outcome == Outcome.NOT_FOUND and r.paper is None and r.download is None
    assert "nothing was downloaded" in r.summary


def test_unknown_doi_is_not_found(settings):
    r = build_agent(settings, FakeWeb(), one_provider(base_paper())).run("10.9999/does.not.exist")
    assert r.query_type == QueryType.DOI and r.outcome == Outcome.NOT_FOUND


def test_similar_but_different_title_is_never_downloaded_automatically(settings):
    similar = base_paper(title="The Impact of Interest Rates on P2P Lending Default Rates", doi="10.5555/other",
                         locations=[loc("https://r.example/other.pdf", "pdf", VersionType.PUBLISHED, "publisher", "primary")])
    web = FakeWeb().add_pdf("https://r.example/other.pdf", good_pdf())
    r = build_agent(settings, web, one_provider(similar)).run(TITLE)
    assert r.outcome == Outcome.NEEDS_CHOICE and r.download is None and r.alternatives[0].doi == "10.5555/other"
    assert web.calls == [] and not list(settings.download_dir.rglob("*.pdf"))


def test_distinct_papers_with_the_same_title_ask_the_user(settings):
    a = base_paper(doi="10.1111/a", authors=["Ashish Vaswani", "Noam Shazeer"], year=2017)
    b = base_paper(doi="10.1111/b", authors=["J. Mark Bishop"], year=2025)
    r = build_agent(settings, FakeWeb(), lambda c, s: [FakeProvider(c, s, name="p", papers=[a, b])]).run(TITLE)
    assert r.outcome == Outcome.NEEDS_CHOICE and {p.doi for p in r.alternatives} == {"10.1111/a", "10.1111/b"} and r.download is None


def test_topic_query_returns_ranked_candidates_without_downloading(settings):
    papers = [base_paper(title="Interest rates and default in peer-to-peer lending markets", doi="10.1111/a", citation_count=50),
              base_paper(title="Machine learning for credit scoring", doi="10.1111/b", abstract="credit risk models")]
    web = FakeWeb()
    r = build_agent(settings, web, lambda c, s: [FakeProvider(c, s, name="p", papers=papers)]).run("What affects default rates in peer-to-peer lending?")
    assert r.query_type == QueryType.TOPIC and r.outcome == Outcome.NEEDS_CHOICE
    assert r.alternatives[0].doi == "10.1111/a" and web.calls == []


def test_exact_title_typed_in_lowercase_is_recognised_even_if_it_looks_like_a_topic(settings, web):
    paper = base_paper(title="Peer to peer lending", locations=[loc("https://r.example/p.pdf", "pdf", VersionType.UNKNOWN, "repository", "primary")])
    web.add_pdf("https://r.example/p.pdf", __import__("tests.conftest", fromlist=["paper_pdf"]).paper_pdf("Peer to peer lending", AUTHORS, pages=3))
    r = build_agent(settings, web, one_provider(paper)).run("peer to peer lending")
    assert r.outcome in (Outcome.NEEDS_CHOICE, Outcome.DOWNLOADED)


def test_provider_outage_and_quota_errors_do_not_stop_the_other_sources(settings, web):
    paper = base_paper(locations=[loc("https://r.example/p.pdf", "pdf", VersionType.UNKNOWN, "repository", "primary")])
    web.add_pdf("https://r.example/p.pdf", good_pdf())
    def provs(c, s):
        return [FakeProvider(c, s, name="quota", fail=RateLimited("HTTP 429", 429)),
                FakeProvider(c, s, name="broken", fail=RuntimeError("boom")),
                FakeProvider(c, s, name="good", papers=[paper])]
    r = build_agent(settings, web, provs).run(TITLE)
    assert r.outcome == Outcome.DOWNLOADED
    assert "quota" in r.summary and "broken" in r.summary and "good" in r.providers_used


def test_second_request_for_the_same_paper_uses_the_local_library(settings, web):
    paper = base_paper(locations=[loc("https://r.example/p.pdf", "pdf", VersionType.UNKNOWN, "repository", "primary")])
    web.add_pdf("https://r.example/p.pdf", good_pdf())
    agent = build_agent(settings, web, one_provider(paper))
    first = agent.run(TITLE)
    before = web.hits("r.example/p.pdf")
    second = agent.run(DOI)
    assert first.outcome == second.outcome == Outcome.DOWNLOADED
    assert second.download.already_had and second.download.path == first.download.path
    assert web.hits("r.example/p.pdf") == before and len(list(settings.download_dir.glob("*.pdf"))) == 1     # no network use
    forced = agent.run(DOI, refresh=True)
    assert forced.download and web.hits("r.example/p.pdf") > before and len(list(settings.download_dir.glob("*.pdf"))) == 1   # same hash: no duplicate file


def test_hostile_titles_cannot_inject_markdown_into_the_summary(settings, web):
    evil = base_paper(title="Interest Rates [click here](http://evil.example) ![x](http://evil.example/t.png) and Peer-to-Peer Lending Default Risk",
                      locations=[])
    r = build_agent(settings, web, one_provider(evil)).run(evil.title)
    assert "](http://evil.example" not in r.summary.replace("\\](", "")


def test_author_year_query_ranks_candidates_by_author_and_year(settings):
    papers = [base_paper(title="Interest rates and peer-to-peer lending default", doi="10.1111/a", authors=["Jane Smith"], year=2020),
              base_paper(title="Interest rates and peer-to-peer lending default", doi="10.1111/b", authors=["Pedro Lopez"], year=2020, abstract="")]
    r = build_agent(settings, FakeWeb(), lambda c, s: [FakeProvider(c, s, name="p", papers=papers)]).run("Smith 2020 lending default rates")
    assert r.query_type == QueryType.AUTHOR_YEAR and r.outcome == Outcome.NEEDS_CHOICE and r.alternatives[0].doi == "10.1111/a"


def test_keyword_phrase_without_exact_title_falls_back_to_ranked_candidates(settings):
    papers = [base_paper(title="Default risk in peer-to-peer lending: interest rate effects", doi="10.1111/a")]
    r = build_agent(settings, FakeWeb(), lambda c, s: [FakeProvider(c, s, name="p", papers=papers)]).run("peer-to-peer lending default risk interest rate effects analysis")
    assert r.outcome == Outcome.NEEDS_CHOICE and r.alternatives and r.download is None
