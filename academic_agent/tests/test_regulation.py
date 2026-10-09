"""Indonesian regulations (SEOJK / POJK / UU ...): parsing, official-source search, verification, honesty; batch input."""
import pytest

from batch import batch_markdown, batch_rows, rows_to_csv, split_queries
from models import AccessState, Outcome, QueryType
from paper_match import classify_query
from regulation_match import parse_regulation_ref, reg_filename, slug_matches, verify_regulation_text
from tests.agent_helpers import base_paper, build_agent, good_pdf, one_provider
from tests.conftest import AUTHORS, TITLE, FakeProvider, FakeWeb, loc, make_pdf

OJK = "https://www.ojk.go.id/id/regulasi/default.aspx"
PAGE = "https://www.ojk.go.id/id/regulasi/Pages/SEOJK-19-SEOJK06-2025-Penyelenggaraan-LPBBTI.aspx"
DOCS = "https://www.ojk.go.id/id/regulasi/Documents/Pages/SEOJK-19-SEOJK06-2025-Penyelenggaraan-LPBBTI/"
MAIN_PDF = DOCS + "SEOJK%2019-SEOJK06-2025%20Penyelenggaraan%20LPBBTI.pdf"
FORM = ("<html><body><form method='post' action='./default.aspx'><input type='hidden' name='__VIEWSTATE' value='abc'>"
        "<input type='hidden' name='__REQUESTDIGEST' value='dig'><input type='text' name='ctl00$PlaceHolderMain$ctl01$TextBoxNomor'>"
        "<input type='submit' name='ctl00$PlaceHolderMain$ctl01$ButtonSearch' value='Cari'></form></body></html>")


def results_html(*links):
    return "<html><body>" + "".join(f"<a href='{h}'>{t}</a>" for h, t in links) + "</body></html>"


def detail_html(extra=""):
    return ("<html><body><h1>SEOJK 19</h1>" + "x " * 400 +
            f"<a href='{MAIN_PDF}'>SEOJK 19-SEOJK06-2025 Penyelenggaraan LPBBTI</a>"
            f"<a href='{DOCS}Abstrak%20SEOJK%2019.pdf'>Abstrak SEOJK 19</a><a href='{DOCS}FAQ%20SEOJK%2019.pdf'>FAQ SEOJK 19</a>{extra}</body></html>")


def reg_pdf(number_line="SURAT EDARAN OTORITAS JASA KEUANGAN NOMOR 19/SEOJK.06/2025"):
    return make_pdf([[number_line, "TENTANG", "PENYELENGGARAAN LAYANAN PENDANAAN BERSAMA BERBASIS TEKNOLOGI INFORMASI", "Sehubungan dengan ..."],
                     ["Halaman dua " * 5]])


def ojk_web(pdf=None, results=None):
    w = FakeWeb().add(OJK, FORM, method="GET")
    w.add(OJK, results_html(*(results if results is not None else [(PAGE.replace("https://www.ojk.go.id", ""), "Penyelenggaraan LPBBTI")])), method="POST")
    w.add(PAGE, detail_html())
    w.add_pdf(MAIN_PDF, pdf if pdf is not None else reg_pdf())
    return w


# ============================================================================ parsing
@pytest.mark.parametrize("q,kind,number,year,sector", [
    ("SEOJK No. 19/SEOJK.06/2025", "SEOJK", "19", 2025, "06"),
    ("Surat Edaran OJK Nomor 19/SEOJK.06/2025 tentang LPBBTI", "SEOJK", "19", 2025, "06"),
    ("POJK Nomor 40 Tahun 2024", "POJK", "40", 2024, ""),
    ("POJK 40/POJK.05/2024", "POJK", "40", 2024, "05"),
    ("Peraturan Otoritas Jasa Keuangan Nomor 10 Tahun 2022", "POJK", "10", 2022, ""),
    ("UU No. 4 Tahun 2023", "UU", "4", 2023, ""),
    ("Undang-Undang Nomor 27 Tahun 2022 tentang Pelindungan Data Pribadi", "UU", "27", 2022, ""),
    ("PP 71 Tahun 2019", "PP", "71", 2019, ""),
    ("23/6/PBI/2021", "PBI", "23/6", 2021, ""),
    ("Peraturan Menteri Keuangan Nomor 70/PMK.010/2016", "PMK", "70", 2016, "010"),
    ("link pdf SEOJK 19 Tahun 2025", "SEOJK", "19", 2025, ""),
    ("UU 4/2023", "UU", "4", 2023, ""),
    ("PP No. 71/2019", "PP", "71", 2019, ""),
    ("POJK 40/2024", "POJK", "40", 2024, ""),
])
def test_regulation_references_are_parsed(q, kind, number, year, sector):
    r = parse_regulation_ref(q)
    assert r and (r.kind, r.number, r.year, r.sector) == (kind, number, year, sector)
    assert classify_query(q).type == QueryType.REGULATION


def test_papers_that_mention_a_regulation_are_not_hijacked():
    for q in ("Impact of UU No. 8 Tahun 1999 on consumer protection: a study", "Does fintech threaten Islamic banking performance in Indonesia?",
              "Smith, J. (2020). Rates and default"):
        assert classify_query(q).type != QueryType.REGULATION


def test_slug_matching_distinguishes_neighbouring_regulations():
    r = parse_regulation_ref("SEOJK No. 19/SEOJK.06/2025")
    assert slug_matches(r, "SEOJK-19-SEOJK06-2025-Penyelenggaraan-LPBBTI")
    assert not slug_matches(r, "SEOJK-19-SEOJK05-2025-Lain") and not slug_matches(r, "SEOJK-20-SEOJK06-2025-Lain")
    assert slug_matches(parse_regulation_ref("SEOJK 19 Tahun 2025"), "SEOJK-19-SEOJK06-2025-X")


def test_text_verification():
    r = parse_regulation_ref("SEOJK No. 19/SEOJK.06/2025")
    ok = verify_regulation_text(r, "SURAT EDARAN OTORITAS JASA KEUANGAN NOMOR 19/SEOJK.06/2025 TENTANG " + "lorem " * 30, 3, False)
    assert ok.verdict == "verified"
    other = verify_regulation_text(r, "SURAT EDARAN OTORITAS JASA KEUANGAN NOMOR 20/SEOJK.08/2025 TENTANG " + "lorem " * 30, 3, True)
    assert other.verdict == "mismatch" and not other.accepted
    assert verify_regulation_text(r, "", 5, True).verdict == "official_page" and verify_regulation_text(r, "", 5, False).verdict == "unreadable"
    assert verify_regulation_text(parse_regulation_ref("UU 4 Tahun 2023"), "UNDANG-UNDANG REPUBLIK INDONESIA NOMOR 4 TAHUN 2023 TENTANG " + "x " * 60, 2, False).verdict == "verified"


def test_filename():
    r = parse_regulation_ref("SEOJK No. 19/SEOJK.06/2025")
    assert reg_filename(r, "Penyelenggaraan LPBBTI") == "SEOJK_19_06_2025_Penyelenggaraan_LPBBTI.pdf"


# ============================================================================ finding the official PDF
def agent(settings, web, **kw):
    return build_agent(settings.replace(**kw) if kw else settings, web, lambda c, s: [])


def test_official_ojk_pdf_link_is_found_and_verified(settings):
    web = ojk_web()
    r = agent(settings.replace(auto_download=False), web).run("SEOJK No. 19/SEOJK.06/2025")
    assert r.query_type == QueryType.REGULATION and r.outcome == Outcome.LINK_FOUND and r.access_state == AccessState.OFFICIAL_REGULATION
    assert r.download.source_url == MAIN_PDF and r.download.verification["verdict"] == "verified" and r.download.saved is False
    assert r.official_link == PAGE and any(c.url == MAIN_PDF for c in r.citations)
    assert {l.url.rsplit("/", 1)[-1] for l in r.other_versions} == {"Abstrak%20SEOJK%2019.pdf", "FAQ%20SEOJK%2019.pdf"}      # companions listed, not mistaken for the law
    post = [x for x in web.requests if x[0] == "POST"][0]
    assert b"TextBoxNomor=19" in post[2].encode() or "TextBoxNomor=19" in str(post[2])
    assert "__VIEWSTATE=abc" in str(post[2]) and not list(settings.download_dir.rglob("*.pdf"))


def test_regulation_is_saved_when_auto_download_is_on(settings):
    r = agent(settings, ojk_web()).run("SEOJK 19/SEOJK.06/2025")
    assert r.outcome == Outcome.DOWNLOADED and r.download.path.endswith("SEOJK_19_06_2025_Surat_Edaran_OJK_Nomor_19_SEOJK_06_2025_tentang_Penyelenggaraan.pdf") or r.download.path.endswith(".pdf")
    assert len(list(settings.download_dir.glob("*.pdf"))) == 1


def test_a_different_regulation_behind_the_link_is_rejected(settings):
    r = agent(settings, ojk_web(pdf=reg_pdf("SURAT EDARAN OTORITAS JASA KEUANGAN NOMOR 20/SEOJK.08/2025"))).run("SEOJK No. 19/SEOJK.06/2025")
    assert r.outcome == Outcome.NO_FULLTEXT and r.download is None
    assert any(a.status == "rejected_after_verification" for a in r.attempts) and "no PDF could be verified" in r.summary


def test_scanned_regulation_is_accepted_only_because_the_official_page_links_it(settings):
    r = agent(settings.replace(auto_download=False), ojk_web(pdf=make_pdf([[]]))).run("SEOJK No. 19/SEOJK.06/2025")
    assert r.outcome == Outcome.LINK_FOUND and r.download.verification["verdict"] == "official_page"
    assert "scanned" in r.download.verification["reasons"][0]


def test_unknown_regulation_gives_manual_search_pages_not_invented_links(settings):
    web = ojk_web(results=[])
    r = agent(settings, web).run("SEOJK No. 99/SEOJK.06/2025")
    assert r.outcome == Outcome.NOT_FOUND and r.download is None
    manual = [c for c in r.citations if "manual search page" in c.note]
    assert manual and all(c.url.startswith("https://") for c in manual)
    assert not any(u.lower().endswith(".pdf") for u in (c.url for c in r.citations))


def test_neighbouring_regulation_in_results_is_not_mistaken_for_the_requested_one(settings):
    other = "/id/regulasi/Pages/SEOJK-19-SEOJK05-2025-Lain.aspx"
    r = agent(settings, ojk_web(results=[(other, "Lain")])).run("SEOJK No. 19/SEOJK.06/2025")
    assert r.outcome == Outcome.NOT_FOUND


def test_web_search_finds_an_official_go_id_pdf_when_ojk_search_fails(settings):
    from tests.conftest import FakeWebSearch
    from search_providers import WebHit
    pdf = "https://jdih.example.go.id/files/SEOJK-19-SEOJK06-2025.pdf"
    web = FakeWeb().add(OJK, FORM, method="GET").add(OJK, results_html(), method="POST").add_pdf(pdf, reg_pdf())
    hits = [WebHit("SEOJK 19/SEOJK.06/2025 LPBBTI", pdf, "salinan resmi", "web:fake"),
            WebHit("Blog SEOJK-19-SEOJK06-2025", "https://random-blog.example/seojk-19-seojk06-2025", "x", "web:fake")]
    holder = {}
    a = build_agent(settings.replace(auto_download=False), web, lambda c, s: [], lambda c, s: [FakeWebSearch(c, s, default_hits=hits)])
    r = a.run("SEOJK No. 19/SEOJK.06/2025")
    assert r.outcome == Outcome.LINK_FOUND and r.download.source_url == pdf and web.hits("random-blog") == 0


# ============================================================================ batch
def test_split_queries_handles_numbering_bullets_blank_lines_and_duplicates():
    text = "1. The Impact of Interest Rates\n\n- SEOJK No. 19/SEOJK.06/2025\n  \n• 10.1371/journal.pone.0000308\n2) the impact of interest rates\nab\n"
    assert split_queries(text) == ["The Impact of Interest Rates", "SEOJK No. 19/SEOJK.06/2025", "10.1371/journal.pone.0000308"]
    assert len(split_queries("\n".join(f"Title number {i} about lending" for i in range(40)), limit=25)) == 25


def test_mixed_batch_returns_one_result_per_line_in_order(settings):
    paper = base_paper(locations=[loc("https://r.example/p.pdf", "pdf", host_type="repository", provider="primary")])
    web = ojk_web().add_pdf("https://r.example/p.pdf", good_pdf())
    from database import Database
    from research_agent import ResearchAgent
    from tests.agent_helpers import ABSTRACT
    from tests.conftest import make_client, registry
    client = make_client(settings.replace(auto_download=False), web)
    ag = ResearchAgent(settings.replace(auto_download=False), http=client, registry=registry(FakeProvider(client, settings, name="primary", papers=[paper])), db=Database(settings.db_path))
    events = []
    text = f"1. {TITLE}\n2. SEOJK No. 19/SEOJK.06/2025\n3. A Paper That Does Not Exist Anywhere Qzxv Wvut"
    res = ag.run_batch(text, progress=events.append)
    assert [r.query_type for r in res] == [QueryType.TITLE, QueryType.REGULATION, QueryType.TITLE]
    assert [r.outcome for r in res] == [Outcome.LINK_FOUND, Outcome.LINK_FOUND, Outcome.NOT_FOUND]
    assert any(e.message.startswith("[2/3]") for e in events) and any(e.stage == "batch" for e in events)
    rows = batch_rows(res)
    assert rows[0]["pdf_link"] == "https://r.example/p.pdf" and rows[1]["pdf_link"] == MAIN_PDF and rows[2]["pdf_link"] == ""
    md = batch_markdown(res)
    assert "1. " + TITLE in md and MAIN_PDF in md and "Not found - no verified PDF link" in md
    csv_text = rows_to_csv(rows)
    assert csv_text.splitlines()[0].startswith("#,query,status,pdf_link") and MAIN_PDF in csv_text


def test_batch_size_is_capped(settings, web):
    ag = agent(settings.replace(max_batch=3), web)
    res = ag.run_batch("\n".join(f"A Totally Unknown Paper Title Number {i} qzx" for i in range(10)))
    assert len(res) == 3


# ============================================================================ strict identity (regression: UU 4/2023)
def test_faq_and_amending_pages_that_merely_mention_the_regulation_are_not_the_regulation(settings):
    faq = "/id/regulasi/Pages/FAQ-Pengembangan-dan-Penguatan-Sektor-Keuangan-(UU-PPSK).aspx"
    r = agent(settings, ojk_web(results=[(faq, "FAQ Ketentuan Terkait Dana Pensiun Pasca Berlakunya Undang-Undang Nomor 4 Tahun 2023")])).run("UU 4/2023")
    assert r.outcome == Outcome.NOT_FOUND and not r.attempts


def test_amending_regulation_text_mentioning_the_number_is_rejected():
    ref = parse_regulation_ref("SEOJK No. 19/SEOJK.04/2018")
    amending = "PERUBAHAN ATAS SURAT EDARAN OTORITAS JASA KEUANGAN NOMOR 19/SEOJK.04/2018 TENTANG LAPORAN " + "isi " * 60
    assert verify_regulation_text(ref, amending, 3, True).verdict == "mismatch"
    original = "SURAT EDARAN OTORITAS JASA KEUANGAN NOMOR 19/SEOJK.04/2018 TENTANG LAPORAN " + "isi " * 60
    assert verify_regulation_text(ref, original, 3, False).verdict == "verified"
    buried = "isi " * 1200 + " NOMOR 19/SEOJK.04/2018 " + "isi " * 60
    assert verify_regulation_text(ref, buried, 3, True).verdict == "mismatch"


JDIH_UU = {"data": [{"slug": "uu-4-tahun-2023", "bentuk": "Undang-Undang", "no": 4, "tahun": 2023, "nomor": "UU 4 TAHUN 2023", "status": "Berlaku",
                     "judul": "Pengembangan dan Penguatan Sektor Keuangan", "full_text_pdf": "/api/download/ABC/UU4TAHUN2023.pdf"},
                    {"slug": "uu-18-tahun-2023", "bentuk": "Undang-Undang", "no": 18, "tahun": 2023, "nomor": "UU 18 TAHUN 2023", "full_text_pdf": "/api/download/X/UU18.pdf"}]}


def test_uu_is_found_through_jdih_kemenkeu_and_verified(settings):
    import json
    web = ojk_web(results=[]).add("https://jdih.kemenkeu.go.id/api/search", json.dumps(JDIH_UU), content_type="application/json")
    uu_pdf = make_pdf([["UNDANG-UNDANG REPUBLIK INDONESIA", "NOMOR 4 TAHUN 2023", "TENTANG PENGEMBANGAN DAN PENGUATAN SEKTOR KEUANGAN", "x " * 60]])
    web.add_pdf("https://jdih.kemenkeu.go.id/api/download/ABC/UU4TAHUN2023.pdf", uu_pdf)
    r = agent(settings.replace(auto_download=False), web).run("UU 4/2023")
    assert r.outcome == Outcome.LINK_FOUND and r.download.source_url.endswith("/UU4TAHUN2023.pdf") and "Berlaku" in r.summary
    assert web.hits("UU18.pdf") == 0                                    # neighbouring number never fetched
