"""Streamlit front end for the Academic Research Agent.

The UI contains no research logic: it calls ``ResearchAgent.run`` and renders the returned
``ResearchResult`` (also available as ``result.to_dict()`` for a future API / JS front end).
Run with:  streamlit run app.py
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from config import Settings
from database import Database
from http_client import HttpClient
from batch import batch_markdown, batch_rows, rows_to_csv, split_queries
from models import AccessState, FullTextLocation, Outcome, Paper, ProgressEvent, QueryType, VersionType
from paper_match import confirmations, md_escape
from research_agent import ResearchAgent
from search_providers import ProviderRegistry

st.set_page_config(page_title="Academic Research Agent", page_icon="🎓", layout="wide")

STAGE_PROGRESS = {"understand": 0.05, "providers": 0.15, "identify": 0.3, "verify-metadata": 0.4, "discover": 0.5,
                  "crawl": 0.6, "probe": 0.65, "download": 0.8, "verify-pdf": 0.92, "decide": 0.97, "error": 1.0}
LEVEL_ICON = {"info": "▫️", "success": "✅", "warning": "⚠️", "error": "❌"}
STATE_STYLE = {
    AccessState.PUBLISHED: ("green", "Final published article"),
    AccessState.OFFICIAL_REGULATION: ("green", "Official regulation (issuing authority)"),
    AccessState.ACCEPTED: ("blue", "Accepted manuscript"),
    AccessState.PREPRINT: ("orange", "Preprint (not the final article)"),
    AccessState.FULLTEXT_UNKNOWN_VERSION: ("violet", "Full text - version not stated"),
    AccessState.ABSTRACT_ONLY: ("gray", "Abstract-only record"),
    AccessState.INACCESSIBLE: ("red", "Inaccessible publication"),
    AccessState.UNKNOWN: ("gray", "Not identified"),
}


# ----------------------------------------------------------------------------- services
@st.cache_resource
def get_db() -> Database:
    return Database(Settings.from_env().db_path)


def base_settings() -> Settings:
    return Settings.from_env()


def session_settings() -> Settings:
    """Environment settings overlaid with the sidebar choices."""
    s = base_settings()
    ss = st.session_state
    return s.replace(
        contact_email=ss.get("contact_email", s.contact_email).strip(),
        max_pages=int(ss.get("max_pages", s.max_pages)), max_depth=int(ss.get("max_depth", s.max_depth)),
        max_candidates=int(ss.get("max_candidates", s.max_candidates)),
        time_limit_seconds=float(ss.get("time_limit", s.time_limit_seconds)),
        allow_preprints=bool(ss.get("allow_preprints", s.allow_preprints)),
        auto_download=bool(ss.get("auto_download", s.auto_download)),
        accept_partial_verification=bool(ss.get("accept_partial", s.accept_partial_verification)))


def set_pick(idx: int, q: str) -> None:
    """Re-run just one item of the list with a more specific query (the other results stay)."""
    st.session_state["pending_pick"] = (idx, q)


def paper_query(p: Paper) -> str:
    """Most specific query that re-identifies exactly this paper (DOI > arXiv id > surname + year + title)."""
    if p.doi:
        return p.doi
    if p.arxiv_id:
        return f"arXiv:{p.arxiv_id}"
    if p.authors and p.year:
        return f"{p.first_author_surname.title()} {p.year} {p.title}"
    return p.title


# ----------------------------------------------------------------------------- sidebar
def sidebar() -> None:
    s = base_settings()
    with st.sidebar:
        st.header("Settings")
        st.text_input("Contact e-mail (needed by Unpaywall; used for polite API access)", key="contact_email",
                      value=s.contact_email, placeholder="you@university.edu")
        st.subheader("Search budget")
        st.slider("Max pages to visit", 1, 30, s.max_pages, key="max_pages")
        st.slider("Max link depth", 0, 3, s.max_depth, key="max_depth")
        st.slider("Max candidate locations", 4, 40, s.max_candidates, key="max_candidates")
        st.slider("Time limit (seconds)", 30, 600, int(s.time_limit_seconds), step=10, key="time_limit")
        st.subheader("Decisions")
        st.checkbox("Auto-download: also save the PDF to Downloaded_Papers (off = link only)", s.auto_download, key="auto_download")
        st.checkbox("Allow preprints when nothing better exists", s.allow_preprints, key="allow_preprints")
        st.checkbox("Accept 'partially verified' matches (title differs between versions)", s.accept_partial_verification,
                    key="accept_partial")
        st.checkbox("Re-download even if the paper is already in my library", False, key="refresh")
        st.caption("robots.txt, paywalls, logins and CAPTCHAs are always respected - never bypassed.")
        st.subheader("Sources")
        reg = ProviderRegistry.default(HttpClient(session_settings()), session_settings())
        for row in reg.status():
            st.markdown(f"{'🟢' if row['available'] else '⚪'} **{row['label']}**" + ("" if row["available"] else f" - {row['note']}"))
        st.caption(f"Downloads: `{s.download_dir}`")


# ----------------------------------------------------------------------------- research run
OUTCOME_LABEL = {Outcome.DOWNLOADED: "PDF saved and verified", Outcome.LINK_FOUND: "Verified PDF link found",
                 Outcome.NEEDS_CHOICE: "Please choose the intended paper", Outcome.NO_FULLTEXT: "Identified - no legal PDF found",
                 Outcome.NOT_FOUND: "Not found"}


def run_queries(queries: list[str]) -> list:
    """Run one or many queries sequentially with live progress; returns the results in input order."""
    agent = ResearchAgent(session_settings(), db=get_db())
    n = len(queries)
    bar = st.progress(0.0, text="Starting…")
    status = st.status(f"Researching {n} item(s)…", expanded=True)
    results = []
    for i, q in enumerate(queries):
        with status:
            st.markdown(f"**[{i + 1}/{n}] {md_escape(q[:140])}**")

        def on_event(ev: ProgressEvent, i=i) -> None:
            frac = (i + STAGE_PROGRESS.get(ev.stage, 0.5)) / n
            bar.progress(min(frac, 0.99), text=f"[{i + 1}/{n}] {ev.stage}: {ev.message[:80]}")
            with status:
                st.markdown(f"{LEVEL_ICON.get(ev.level, '▫️')} `{ev.stage}` {md_escape(ev.message)}")

        r = agent.run(q, progress=on_event, refresh=bool(st.session_state.get("refresh", False)))
        results.append(r)
        with status:
            st.markdown(f"➡️ **{OUTCOME_LABEL[r.outcome]}**")
    bar.progress(1.0, text="Finished")
    ok = sum(1 for r in results if r.outcome in (Outcome.DOWNLOADED, Outcome.LINK_FOUND))
    status.update(label=f"Finished: {ok}/{n} PDF link(s) verified", state="complete", expanded=False)
    return results


# ----------------------------------------------------------------------------- rendering
def render_metadata(p: Paper) -> None:
    st.subheader("Publication metadata")
    st.markdown(f"**{md_escape(p.title)}**")
    st.write(md_escape(", ".join(p.authors[:12]) + (" et al." if len(p.authors) > 12 else "")) or "Authors not listed")
    cols = st.columns(3)
    cols[0].metric("Year", p.year or "n.d.")
    cols[1].metric("Citations", p.citation_count if p.citation_count is not None else "n/a")
    cols[2].metric("Type", (p.work_type or "n/a")[:22])
    if p.venue:
        st.write(f"**Venue:** {md_escape(p.venue)}")
    if p.doi:
        st.markdown(f"**DOI:** [{md_escape(p.doi)}]({p.doi_url})")
    if p.arxiv_id:
        st.markdown(f"**arXiv:** [{md_escape(p.arxiv_id)}](https://arxiv.org/abs/{p.arxiv_id})")
    if p.related_dois:
        st.write("**Related versions (DOIs):** " + ", ".join(md_escape(d) for d in p.related_dois))
    c = confirmations(p)
    st.caption(f"Cross-checked across {len(c['providers'])} source(s): {', '.join(c['providers'])}")
    if p.abstract:
        with st.expander("Abstract"):
            st.write(md_escape(p.abstract))
    if len(p.provenance) > 1:
        with st.expander("Source-by-source bibliographic comparison"):
            st.dataframe([{"source": s["provider"], "title": s["title"], "year": s["year"], "doi": s["doi"],
                           "first authors": ", ".join(s["authors"][:3])} for s in p.provenance], width="stretch")


def render_regulation(result) -> None:
    p = result.paper
    left, right = st.columns(2)
    with left:
        st.subheader("Regulation")
        st.markdown(f"**{md_escape(p.title)}**")
        st.write(f"**Issuer:** {md_escape(p.authors[0] if p.authors else 'n/a')}  •  **Year:** {p.year or 'n/a'}  •  **Type:** {md_escape(p.venue)}")
        if p.url:
            st.markdown(f"**Official page:** [{md_escape(p.url[:90])}]({p.url})")
    with right:
        if result.download and not result.download.saved:
            render_link(result)
        elif result.download:
            render_download(result)
        else:
            st.subheader("Access")
            st.info("No verified PDF. Nothing here is a guessed address; use the manual search pages listed under Sources.")
    if result.other_versions:
        with st.expander(f"Companion documents on the official page ({len(result.other_versions)})"):
            st.dataframe([{"document": l.note.replace("companion document: ", ""), "link": l.url} for l in result.other_versions],
                          width="stretch", column_config={"link": st.column_config.LinkColumn("link")})


def render_link(result) -> None:
    d = result.download
    st.subheader("PDF link")
    st.link_button("Open PDF", d.source_url, type="primary")
    st.code(d.source_url, language=None)
    st.write(f"**Version:** {d.version.label}  •  **Checked:** {d.access_date or 'n/a'}")
    if d.landing_url and d.landing_url != d.source_url:
        st.markdown(f"**Article page:** [{md_escape(d.landing_url[:90])}]({d.landing_url})")
    st.caption("The link was opened and its PDF content matched this paper. Turn on auto-download in the sidebar to also save a copy.")
    if d.version in (VersionType.PREPRINT, VersionType.ACCEPTED):
        st.warning("This is not the final published article. Check the published version before citing page numbers or results.")


def render_download(result) -> None:
    d = result.download
    st.subheader("Download")
    path = Path(d.path)
    st.success(("Already in your library: " if d.already_had else "Saved: ") + path.name)
    st.code(str(path), language=None)
    st.write(f"**Version:** {d.version.label}  •  **Size:** {d.size_bytes / 1024:.0f} KB  •  **Retrieved:** {d.access_date or 'n/a'}")
    st.markdown(f"**Source:** [{md_escape(d.source_url[:90])}]({d.source_url})")
    st.caption(f"SHA-256 `{d.sha256}`")
    if path.is_file():
        st.download_button("Save a copy of the PDF", data=path.read_bytes(), file_name=path.name, mime="application/pdf")
    if d.version in (VersionType.PREPRINT, VersionType.ACCEPTED):
        st.warning("This is not the final published article. Check the published version before citing page numbers or results.")


def render_evidence(result) -> None:
    with st.expander("Evidence behind the decision", expanded=False):
        if result.download:
            v = result.download.verification
            st.markdown("**PDF verification**")
            st.write({"verdict": v.get("verdict"), "title match": v.get("title_score"), "authors found": v.get("authors_score"),
                      "abstract overlap": v.get("abstract_score"), "DOI printed in document": v.get("doi_found"),
                      "pages": v.get("page_count"), "reasons": v.get("reasons"), "warnings": v.get("warnings"),
                      "version hints in text": v.get("version_evidence")})
            if v.get("excerpt"):
                st.caption("First text of the document (untrusted content, shown as plain text):")
                st.text(v["excerpt"])
        if result.attempts:
            st.markdown("**Every location examined**")
            st.dataframe([{"result": a.status, "stage": a.stage, "provider": a.provider, "version": a.version, "url": a.url,
                           "detail": a.detail} for a in result.attempts], width="stretch",
                         column_config={"url": st.column_config.LinkColumn("url")})
        if result.pages_visited:
            st.markdown("**Pages inspected by the crawler**")
            st.dataframe(result.pages_visited, width="stretch")
        st.markdown("**Search log**")
        for ev in result.events:
            st.markdown(f"{LEVEL_ICON.get(ev.level, '▫️')} `{ev.stage}` {md_escape(ev.message)}")


def render_other_versions(result) -> None:
    locs: list[FullTextLocation] = result.other_versions
    if not locs:
        return
    status_by_url = {a.url: a for a in result.attempts}
    rows = []
    for l in locs:
        a = status_by_url.get(l.url)
        rows.append({"version": l.version.label, "type": l.host_type, "found via": l.provider,
                     "checked": a.status if a else "not needed / not reached", "link": l.url})
    with st.expander(f"Other versions and locations ({len(rows)})"):
        st.caption("Only the links marked 'downloaded' were retrieved; the rest are listed for transparency so you can open them yourself.")
        st.dataframe(rows, width="stretch", column_config={"link": st.column_config.LinkColumn("link")})


def render_choices(result, idx: int = 0) -> None:
    st.subheader("Which paper do you mean?")
    for i, p in enumerate(result.alternatives):
        with st.container(border=True):
            c1, c2 = st.columns([5, 1])
            known_oa = "open copy listed" if (p.is_oa or p.locations) else "no open copy listed"
            c1.markdown(f"**{md_escape(p.title)}**  \n{md_escape(', '.join(p.authors[:4]))} • {p.year or 'n.d.'} • "
                        f"{md_escape(p.venue or p.work_type or '')} • {known_oa} • sources: {', '.join(p.sources)} • score {p.score:.2f}")
            if p.doi:
                c1.markdown(f"[{md_escape(p.doi)}]({p.doi_url})")
            c2.button("Retrieve this", key=f"pick_{idx}_{i}", on_click=set_pick, args=(idx, paper_query(p)))


def render_result(result, idx: int = 0) -> None:
    color, label = STATE_STYLE[result.access_state]
    if result.outcome == Outcome.LINK_FOUND:
        st.success("Verified PDF link found (nothing was saved to disk).")
    elif result.outcome == Outcome.DOWNLOADED:
        st.success("Paper retrieved and verified.")
    elif result.outcome == Outcome.NEEDS_CHOICE:
        st.warning("More than one publication is plausible - nothing was downloaded.")
    elif result.outcome == Outcome.NO_FULLTEXT:
        st.warning("Publication identified, but no legally accessible full text could be verified. Nothing was downloaded.")
    else:
        st.error("No matching publication could be identified. Nothing was downloaded.")
    st.markdown(f"**Access status:** :{color}[{label}]")
    st.markdown("### Research summary")
    st.markdown(result.summary)
    if result.citations:
        st.markdown("**Sources**")
        for c in result.citations:
            st.markdown(f"[{c.label}] [{md_escape(c.title)}]({c.url})" + (f" - {md_escape(c.note)}" if c.note else ""))
    if result.outcome == Outcome.NEEDS_CHOICE or (result.outcome == Outcome.NOT_FOUND and result.alternatives):
        render_choices(result, idx)
    if result.paper and result.query_type == QueryType.REGULATION:
        render_regulation(result)
    elif result.paper:
        left, right = st.columns(2)
        with left:
            render_metadata(result.paper)
        with right:
            if result.download and not result.download.saved:
                render_link(result)
            elif result.download:
                render_download(result)
            else:
                st.subheader("Access")
                if result.official_link:
                    st.markdown(f"Official page (may need a subscription): [{result.official_link}]({result.official_link})")
                st.info("No file was saved. The links above are the verified bibliographic record; nothing here is a guessed PDF address.")
        render_other_versions(result)
    render_evidence(result)
    st.caption(f"{result.requests_used} HTTP requests • {result.elapsed_seconds:.0f}s")


def render_results(results: list) -> None:
    if len(results) == 1:
        render_result(results[0], 0)
        return
    st.markdown("### Results")
    rows = batch_rows(results)
    ok = sum(1 for r in results if r.outcome in (Outcome.DOWNLOADED, Outcome.LINK_FOUND))
    st.write(f"**{ok} of {len(results)}** items have a verified PDF link.")
    st.dataframe(rows, width="stretch", hide_index=True,
                 column_config={"pdf_link": st.column_config.LinkColumn("PDF link"), "article_page": st.column_config.LinkColumn("Article / official page")})
    with st.expander("Copy-friendly list of all links"):
        st.code(batch_markdown(results), language=None)
    st.download_button("Download table (CSV)", data=rows_to_csv(rows).encode("utf-8-sig"), file_name="research_results.csv", mime="text/csv")
    st.markdown("### Details")
    pick = st.selectbox("Show details for", range(len(results)), key="detail_pick",
                        format_func=lambda i: f"{i + 1}. {OUTCOME_LABEL[results[i].outcome]} - {(results[i].paper.title if results[i].paper else results[i].query)[:90]}")
    render_result(results[pick], pick)


def render_history() -> None:
    st.subheader("Previously retrieved papers")
    q = st.text_input("Search history (title, author, DOI, venue, query)", key="history_search")
    rows = get_db().history(q, 300)
    if not rows:
        st.info("No research runs recorded yet.")
        return
    table = [{"date": r["started_at"], "query": r["query"], "title": r["title"] or "", "year": r["year"],
              "result": r["outcome"], "version": r["version_type"] or "", "pdf link": r["source_url"] or "", "file": Path(r["file_path"]).name if r["file_path"] else ""}
             for r in rows]
    st.dataframe(table, width="stretch")
    with_files = [r for r in rows if r["file_path"] and Path(r["file_path"]).is_file()]
    if with_files:
        pick = st.selectbox("Open a stored paper", range(len(with_files)),
                            format_func=lambda i: f"{with_files[i]['title'] or with_files[i]['query']} ({with_files[i]['year'] or 'n.d.'})")
        r = with_files[pick]
        st.code(r["file_path"], language=None)
        if r["doi"]:
            st.markdown(f"DOI: [{r['doi']}](https://doi.org/{r['doi']})")
        if r["source_url"]:
            st.markdown(f"Source: [{md_escape(r['source_url'][:100])}]({r['source_url']})")
        st.download_button("Save a copy", data=Path(r["file_path"]).read_bytes(), file_name=Path(r["file_path"]).name,
                           mime="application/pdf", key="hist_dl")


def render_help() -> None:
    s = base_settings()
    st.markdown("""
**How it works.** The agent identifies the publication in several scholarly databases, cross-checks the
metadata, looks for every legal open copy (publisher, repository, preprint), checks each link for real
access, downloads the PDF, and verifies that its content is the requested paper before keeping it.

**What it will never do:** bypass paywalls, logins or CAPTCHAs, ignore robots.txt, or use shadow libraries.

**Optional API keys** (set as environment variables or in a `.env` file - see README):
`CONTACT_EMAIL` (Unpaywall + polite pools), `CORE_API_KEY`, `SEMANTIC_SCHOLAR_API_KEY`, `OPENALEX_API_KEY`,
`BRAVE_API_KEY` or `GOOGLE_CSE_API_KEY` + `GOOGLE_CSE_CX` (open-web search for repository copies).
""")
    st.json(s.public_summary())


# ----------------------------------------------------------------------------- page
def main() -> None:
    sidebar()
    st.title("🎓 Academic Research Agent")
    st.caption("Finds and verifies PDF links for academic papers and Indonesian regulations - only from legal, open sources.")
    tab_research, tab_history, tab_help = st.tabs(["Research", "History", "Setup & help"])
    with tab_research:
        st.text_area("One item per line: paper title, APA citation, DOI, arXiv id, research question, or regulation (e.g. SEOJK No. 19/SEOJK.06/2025)",
                     key="query_text", height=140,
                     placeholder="The Impact of Interest Rates on Peer-to-Peer Lending Default Risk\nSEOJK No. 19/SEOJK.06/2025\n10.1371/journal.pone.0000308")
        go = st.button("Research", type="primary")
        queries = split_queries(st.session_state.get("query_text", ""), base_settings().max_batch)
        pick = st.session_state.pop("pending_pick", None)
        if pick and st.session_state.get("results"):
            idx, q = pick
            new = run_queries([q])[0]
            st.session_state["results"][idx] = new
        elif go and queries:
            st.session_state["results"] = run_queries(queries)
        elif go:
            st.warning("Enter at least one title, DOI or regulation first.")
        if st.session_state.get("results"):
            render_results(st.session_state["results"])
    with tab_history:
        render_history()
    with tab_help:
        render_help()


main()
