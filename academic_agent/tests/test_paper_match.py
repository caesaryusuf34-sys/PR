"""Title matching, query understanding, author matching, de-duplication, filenames."""
import pytest

from models import Paper, QueryType, VersionType
from paper_match import (author_overlap, classify_host, classify_query, cluster_papers, confirmations, extract_arxiv_id,
                         extract_doi, is_subtitle_variant, make_filename, match_query, md_escape, normalize_doi,
                         normalize_title, same_work, surname, title_similarity)

TITLE = "The Impact of Interest Rates on Peer-to-Peer Lending Default Risk"


class TestNormalisation:
    def test_case_punctuation_markup_and_accents(self):
        assert normalize_title("  Café <i>Economics</i>: A  Study! ") == "cafe economics a study"

    def test_p2p_aliases_are_equivalent(self):
        assert normalize_title("Peer-to-Peer Lending") == normalize_title("P2P lending") == "peer to peer lending"

    def test_empty(self):
        assert normalize_title("") == "" and title_similarity("", "x") == 0.0


class TestTitleSimilarity:
    def test_identical_after_normalisation(self):
        assert title_similarity(TITLE, TITLE.lower().replace("-", " ")) == 1.0
        assert title_similarity(TITLE, "The impact of interest rates on P2P lending default risk.") == 1.0

    @pytest.mark.parametrize("other", [
        "The Impact of Interest Rates on Peer-to-Peer Lending Default Rates",      # one word differs
        "Interest Rate Impact on P2P Lending Default Risk",                         # reworded
        "Credit Scoring in Peer-to-Peer Lending",                                   # related topic
        "The Impact of Interest Rates on Bank Lending Default Risk",
    ])
    def test_similar_but_different_papers_are_not_exact(self, other):
        assert title_similarity(TITLE, other) < 0.93

    def test_unrelated_is_low(self):
        assert title_similarity(TITLE, "Quantum Error Correction with Surface Codes") < 0.3

    def test_word_order_matters(self):
        assert title_similarity("Attention Is All You Need", "Is Attention All You Need?") < 0.93

    def test_longer_title_with_different_suffix_is_not_exact(self):
        assert title_similarity("Attention Is All You Need", "Attention is All You Need in Speech Separation") < 0.8

    def test_subtitle_is_recognised_only_after_a_colon(self):
        assert is_subtitle_variant(TITLE, TITLE + ": Evidence from Prosper")
        assert not is_subtitle_variant(TITLE, TITLE + " in Emerging Markets")
        assert title_similarity(TITLE, TITLE + ": Evidence from Prosper") >= 0.9


class TestQueryClassification:
    @pytest.mark.parametrize("q,doi", [
        ("10.1038/nature12373", "10.1038/nature12373"),
        ("https://doi.org/10.1016/j.jfineco.2020.01.001.", "10.1016/j.jfineco.2020.01.001"),
        ("doi: 10.1002/(SICI)1097-0258(19980430)17:8<873::AID-SIM779>3.0.CO;2-I", "10.1002/(sici)1097-0258(19980430)17:8<873::aid-sim779>3.0.co;2-i"),
    ])
    def test_doi(self, q, doi):
        qi = classify_query(q)
        assert qi.type == QueryType.DOI and qi.doi == doi

    def test_arxiv(self):
        assert classify_query("arXiv:1706.03762v5").arxiv_id == "1706.03762"
        assert classify_query("https://arxiv.org/abs/1706.03762").type == QueryType.ARXIV
        assert extract_arxiv_id("1706.03762") == "1706.03762"

    def test_title_vs_topic(self):
        assert classify_query(TITLE).type == QueryType.TITLE
        assert classify_query("What drives default in peer-to-peer lending?").type == QueryType.TOPIC
        assert classify_query("P2P lending").type == QueryType.TOPIC

    def test_author_and_year(self):
        qi = classify_query("Smith 2020 interest rates and peer-to-peer lending default")
        assert qi.type == QueryType.AUTHOR_YEAR and qi.author == "smith" and qi.year == 2020

    def test_domain_detection(self):
        assert classify_query("Efficacy of vaccine therapy in cancer patients: a clinical trial").domain == "biomedical"
        assert classify_query(TITLE).domain == "quantitative"

    def test_normalize_doi_variants(self):
        assert normalize_doi("HTTPS://DX.DOI.ORG/10.1000/ABC.") == "10.1000/abc"
        assert normalize_doi("not a doi") == "" and extract_doi("see doi 10.5555/xyz123, thanks") == "10.5555/xyz123"


class TestAuthors:
    def test_surnames(self):
        assert surname("Smith, John") == surname("John Smith") == surname("John A. Smith Jr.") == "smith"
        assert surname("Gabriel García Márquez") == "marquez"
        assert surname("van der Berg, J.") == "berg"

    def test_overlap(self):
        assert author_overlap(["Jane Smith", "Wei Zhang"], ["Zhang, Wei", "Jane Smith", "X Y"]) == 1.0
        assert author_overlap(["A Brown"], ["C Green"]) == 0.0
        assert author_overlap([], ["C Green"]) is None


class TestMatchQuery:
    def paper(self, title, authors=("Jane Smith",), year=2020, doi=""):
        return Paper(title=title, authors=list(authors), year=year, doi=doi)

    def test_exact_probable_possible_mismatch(self):
        qi = classify_query(TITLE)
        assert match_query(qi, self.paper(TITLE.upper())).verdict == "exact"
        assert match_query(qi, self.paper("The Impact of Interest Rates on P2P Lending Default Rates")).verdict in ("probable", "possible")
        assert match_query(qi, self.paper("Credit Scoring in Peer-to-Peer Lending")).verdict == "mismatch"

    def test_subtitle_counts_as_exact(self):
        assert match_query(classify_query(TITLE), self.paper(TITLE + ": Evidence from Prosper")).verdict == "exact"

    def test_corrections_never_beat_the_article(self):
        qi = classify_query(TITLE)
        m = match_query(qi, self.paper("Correction to: " + TITLE))
        assert m.verdict != "exact"

    def test_doi_must_match_exactly(self):
        qi = classify_query("10.1234/abc")
        assert match_query(qi, self.paper("Anything", doi="10.1234/ABC")).verdict == "exact"
        assert match_query(qi, self.paper(TITLE, doi="10.1234/abd")).verdict == "mismatch"

    def test_author_year_penalty(self):
        qi = classify_query("Smith 2020 " + TITLE)
        good = match_query(qi, self.paper(TITLE, ["Jane Smith"], 2020)).score
        wrong_author = match_query(qi, self.paper(TITLE, ["Pedro Lopez"], 2020)).score
        assert wrong_author < good


class TestClustering:
    def test_same_work_from_two_providers_is_merged(self):
        a = Paper(title=TITLE, authors=["Jane Smith"], year=2024, doi="10.1/x", sources=["crossref"], abstract="short")
        b = Paper(title=TITLE.lower(), authors=["Smith, Jane"], year=2024, doi="10.1/X", sources=["openalex"],
                  abstract="a much longer abstract text")
        out = cluster_papers([a, b])
        assert len(out) == 1 and out[0].abstract.startswith("a much longer")
        assert set(out[0].sources) == {"crossref", "openalex"}
        assert confirmations(out[0])["doi"] == (2, 2)

    def test_preprint_and_journal_version_merge_and_keep_journal_doi(self):
        pre = Paper(title=TITLE, authors=["Jane Smith"], year=2022, doi="10.48550/arxiv.2201.00001", sources=["arxiv"])
        pub = Paper(title=TITLE, authors=["Jane Smith"], year=2024, doi="10.1234/journal.1", sources=["crossref"])
        out = cluster_papers([pre, pub])
        assert len(out) == 1 and out[0].doi == "10.1234/journal.1" and "10.48550/arxiv.2201.00001" in out[0].related_dois

    def test_namesakes_with_different_authors_stay_apart(self):
        a = Paper(title="Attention Is All You Need", authors=["Ashish Vaswani", "Noam Shazeer"], year=2017, sources=["arxiv"])
        b = Paper(title="Attention Is All You Need", authors=["J. Mark Bishop"], year=2025, sources=["openaire"])
        assert len(cluster_papers([a, b])) == 2 and not same_work(a, b)

    def test_repository_copy_with_late_year_merges_when_authors_match(self):
        a = Paper(title="Attention Is All You Need", authors=["Ashish Vaswani", "Noam Shazeer"], year=2017, sources=["arxiv"])
        b = Paper(title="Attention Is All You Need", authors=["Ashish Vaswani", "Noam Shazeer"], year=2025, sources=["openaire"])
        assert len(cluster_papers([a, b])) == 1


class TestFilenameAndMisc:
    def test_descriptive_filename(self):
        p = Paper(title="Interest Rates and P2P Lending", authors=["John Smith"], year=2024)
        assert make_filename(p) == "2024_Smith_Interest_Rates_and_P2P_Lending.pdf"
        assert make_filename(p, VersionType.PREPRINT).endswith("_preprint.pdf")

    def test_filename_is_windows_safe(self):
        p = Paper(title='CON: <bad> "name" / with | chars?*', authors=["A. B"], year=None)
        name = make_filename(p)
        assert not any(c in name for c in '<>:"/\\|?*') and len(name) <= 124 and name.endswith(".pdf")

    def test_host_classes(self):
        assert classify_host("https://arxiv.org/pdf/1") == "preprint_server"
        assert classify_host("https://europepmc.org/x") == "repository"
        assert classify_host("https://eprints.lse.ac.uk/1/") == "institutional"
        assert classify_host("https://www.example-journal.com/a") == "web"

    def test_markdown_escape_defuses_links_and_images(self):
        out = md_escape("![x](http://evil) [a](http://b) *b*")
        assert "[" not in out.replace("\\[", "") and "![" not in out


def test_apa_citation_is_split_into_author_year_and_title():
    qi = classify_query("Takidah, E., & Kassim, S. (2022). The Shariah compliance of Islamic peer-to-peer (P2P) lending practices in Indonesia: Identification of issues and the way forward")
    assert qi.type == QueryType.AUTHOR_YEAR and qi.author == "takidah" and qi.year == 2022
    assert qi.title.startswith("The Shariah compliance") and qi.title.endswith("the way forward")
    full = classify_query("Smith, J. (2020). Rates and default. Journal of Finance, 12(3), 45-67.")
    assert full.title == "Rates and default" and full.author == "smith"


def test_apa_title_ending_in_question_mark_followed_by_journal_name():
    qi = classify_query("Thahirah, F. A., & Kasri, R. A. (2023). Does fintech threaten Islamic banking performance in Indonesia? Journal of Islamic Accounting and Finance Research")
    assert qi.author == "thahirah" and qi.year == 2023 and qi.title == "Does fintech threaten Islamic banking performance in Indonesia?"
