from database import Database
from models import Paper, VersionType
from research_agent import ResearchAgent
from tests.conftest import AUTHORS, DOI, TITLE, FakeProvider, FakeWebSearch, make_client, paper_pdf, registry

ABSTRACT = ("We study how changes in interest rates affect default risk of borrowers on peer-to-peer lending platforms "
            "using loan level data and a difference in differences design across several markets.")

PAYWALL_HTML = ("<html><head><title>Journal of Fintech</title></head><body><h1>" + TITLE + "</h1>" + "Abstract paragraph. " * 60 +
                "<a href='/buy'>Purchase this article</a> Subscribe to read the full text.</body></html>")


def base_paper(**kw) -> Paper:
    d = dict(title=TITLE, authors=list(AUTHORS), year=2024, doi=DOI, abstract=ABSTRACT, venue="Journal of Fintech",
             url=f"https://doi.org/{DOI}", work_type="journal-article")
    d.update(kw)
    return Paper(**d)


def good_pdf(*extra: str) -> bytes:
    return paper_pdf(TITLE, AUTHORS, list(extra), pages=3)


def build_agent(settings, web, providers_factory, web_factory=None) -> ResearchAgent:
    client = make_client(settings, web)
    provs = providers_factory(client, settings)
    webs = web_factory(client, settings) if web_factory else []
    return ResearchAgent(settings, http=client, registry=registry(*provs, web=webs), db=Database(settings.db_path))


def one_provider(paper, name="primary", **kw):
    return lambda c, s: [FakeProvider(c, s, name=name, papers=[paper], **kw)]
