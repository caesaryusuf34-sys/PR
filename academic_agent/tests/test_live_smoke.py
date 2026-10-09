"""Optional end-to-end checks against the real internet:  RUN_LIVE_TESTS=1 python -m pytest tests/test_live_smoke.py"""
import os

import pytest

from config import Settings
from models import Outcome, VersionType
from research_agent import ResearchAgent

pytestmark = [pytest.mark.live, pytest.mark.skipif(not os.getenv("RUN_LIVE_TESTS"), reason="set RUN_LIVE_TESTS=1 to run live tests")]


@pytest.fixture
def live_agent(tmp_path):
    s = Settings.from_env().replace(download_dir=tmp_path / "Downloaded_Papers", db_path=tmp_path / "live.sqlite3", auto_download=True)
    return ResearchAgent(s)


def test_arxiv_identifier_is_downloaded_and_verified(live_agent):
    r = live_agent.run("arXiv:1706.03762")
    assert r.outcome == Outcome.DOWNLOADED and r.download.version == VersionType.PREPRINT
    assert r.download.verification["verdict"] == "verified"


def test_open_access_doi_is_downloaded_from_the_publisher(live_agent):
    r = live_agent.run("10.1371/journal.pone.0000308")
    assert r.outcome == Outcome.DOWNLOADED and r.download.version == VersionType.PUBLISHED


def test_nonexistent_paper_is_not_invented(live_agent):
    r = live_agent.run("Zebra Quantum Mortgage Hamsters: A Fictional Study of Nonexistent Things Qzxv")
    assert r.outcome in (Outcome.NOT_FOUND, Outcome.NEEDS_CHOICE) and r.download is None
