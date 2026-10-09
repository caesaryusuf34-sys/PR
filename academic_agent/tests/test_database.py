"""SQLite storage: dedupe by DOI / title / hash, history search, file-existence checks."""
from models import Attempt, Paper
from tests.agent_helpers import base_paper


def test_papers_are_deduplicated_by_doi_and_title(db):
    a = db.upsert_paper(base_paper())
    b = db.upsert_paper(base_paper(venue="Updated venue"))
    c = db.upsert_paper(base_paper(doi="", venue="", title=base_paper().title.upper()))
    assert a == b == c and db.get_paper(a)["venue"] == "Updated venue"          # empty values never erase stored ones
    assert db.get_paper(a)["doi"] == base_paper().doi
    assert db.upsert_paper(base_paper(doi="10.9999/different", title="Another")) != a


def test_download_lookup_by_doi_title_and_hash_requires_the_file_to_exist(db, tmp_path):
    pid = db.upsert_paper(base_paper())
    f = tmp_path / "x.pdf"
    f.write_bytes(b"%PDF-1.4")
    db.record_download(pid, file_path=str(f), sha256="abc", size_bytes=8, version_type="published", source_url="https://s/x.pdf",
                       landing_url="", provider="p", verification={"verdict": "verified"}, query="q")
    assert db.find_download_by_hash("abc")["file_path"] == str(f)
    assert db.find_download_for_paper(doi=base_paper().doi)["sha256"] == "abc"
    assert db.find_download_for_paper(title=base_paper().title.lower())["sha256"] == "abc"
    f.unlink()
    assert db.find_download_by_hash("abc") is None and db.find_download_for_paper(doi=base_paper().doi) is None


def test_history_search_and_attempts(db):
    pid = db.upsert_paper(base_paper())
    rid = db.start_run("my query about p2p", "exact title")
    db.record_attempts(rid, pid, [Attempt(url="https://x", status="paywall", stage="probe")])
    db.finish_run(rid, outcome="no_fulltext", access_state="inaccessible", paper_id=pid, summary="s", result={"a": 1})
    assert db.history("p2p")[0]["title"] == base_paper().title
    assert db.history("smith")[0]["run_id"] == rid and db.history("nomatch") == []
    assert db.get_run(rid)["result"] == {"a": 1} and db.get_attempts(rid)[0]["status"] == "paywall"
