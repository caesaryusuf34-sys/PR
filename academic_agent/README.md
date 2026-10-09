# Academic Research Agent

An autonomous research agent that **finds, verifies and downloads academic papers - only from legal, open sources.**
Give it a paper title, an APA citation, a DOI, an arXiv id, a research question, **or an Indonesian regulation (e.g. `SEOJK No. 19/SEOJK.06/2025`)** - one item, or **many at once (one per line)**. It will:

1. work out what kind of query it is and pick the right scholarly sources,
2. identify the publication and **cross-check** its metadata across several databases,
3. collect every known open copy (publisher, institutional repository, subject repository, preprint, author manuscript),
4. open each candidate, **check what the server really returns** (PDF signature, login page, paywall, CAPTCHA, robots.txt),
5. open the PDF, validate it, and **verify from the PDF text that it is the requested paper** - by default it then returns the verified **link only**; set `AUTO_DOWNLOAD=true` (or tick it in the sidebar) to also save the file to `Downloaded_Papers`,
6. if nothing works, keep looking for alternative versions (query variants, other databases, web search),
7. report what it found, which sources it used (with clickable citations), and *exactly* what it checked if it found nothing.

It never invents a PDF address, never claims success without a validated file, and **never bypasses a paywall, login, CAPTCHA or robots.txt.**

```
                     ┌────────────────────────────── app.py (Streamlit UI) ──────────────────────────────┐
                     │  search box · live progress · summary · citations · metadata · access badge · history │
                     └──────────────────────────────────────┬─────────────────────────────────────────────┘
                                                            │ ResearchResult (also .to_dict() for any future frontend)
                                              research_agent.py  (the orchestrator)
         ┌──────────────┬───────────────┬─────────────────┼──────────────────┬──────────────────┬───────────────┐
 search_providers.py  paper_match.py  web_crawler.py  access_checker.py  pdf_downloader.py  pdf_verifier.py  database.py
  Crossref OpenAlex   title/author     meta tags,       real GET, PDF     streaming, size   text extraction,   SQLite:
  Semantic Scholar    similarity,      PDF + "full      signature, login/ limits, retries,  title/author/DOI   papers, files,
  Unpaywall CORE      de-duplication,  text" links,     paywall/CAPTCHA   hashes, atomic    matching, version  attempts,
  Europe PMC DOAJ     query parsing    depth/page       classification    save              hints, safety      runs
  arXiv OpenAIRE      filenames        budgets
  Brave / Google CSE        └──────────────── http_client.py + net_safety.py: SSRF guard, rate limits, retries, robots.txt, budgets ───┘
```

### Regulations (SEOJK, POJK, PADK, UU, PP, PMK, PBI, SEBI …)
Type the reference as you normally would: `SEOJK No. 19/SEOJK.06/2025`, `POJK Nomor 40 Tahun 2024`, `UU No. 27 Tahun 2022`, `23/6/PBI/2021`.
The agent searches the **OJK regulation database** (its public search form), **JDIH Kemenkeu** (public JSON API: UU, PP, Perpres, PMK) and, if you configured a web-search key, official `.go.id` pages and PDFs.
A PDF link is returned only if the regulation's identification line (e.g. "Nomor 19/SEOJK.06/2025") is printed in the PDF text; a scanned PDF
without text is accepted only when the official page for exactly that regulation links to it, and that is stated in the result. Companion documents
on the same page (Abstrak, FAQ, Lampiran) are listed separately, never mistaken for the regulation. If nothing is found you get official *search pages*
(OJK, JDIH BPK, JDIH BI) clearly marked as search pages, not PDFs. Portals that block automated clients (e.g. peraturan.bpk.go.id, behind Cloudflare)
are reported and skipped, not bypassed. Bank Indonesia's JDIH is a JavaScript application and is not searched automatically.

### Many items at once
Paste several lines (titles, APA citations, DOIs, regulations - mixed is fine; numbering/bullets are stripped, duplicates dropped, max 25 by default,
`max_batch`). Items run one after another with live progress. You get a table with the PDF link for each, a copy-friendly list, a CSV download,
and a details view per item. Ambiguous titles stay in the list with a "Retrieve this" choice that re-runs only that line.
```python
results = ResearchAgent().run_batch("Title one\nSEOJK No. 19/SEOJK.06/2025\n10.1371/journal.pone.0000308")
```

---

## 1. Installation on Windows 10 / Windows 11

You need **Python 3.10 or newer** (3.11 or 3.12 recommended).

1. **Install Python** from <https://www.python.org/downloads/windows/>. On the first installer screen tick
   **"Add python.exe to PATH"**, then *Install Now*. Check it worked - open *PowerShell* (Start menu → "PowerShell"):
   ```powershell
   py --version
   ```
2. **Get the project** and open a terminal in the `academic_agent` folder
   (in File Explorer, open the folder, click the address bar, type `powershell`, press Enter).
3. **Create a virtual environment and install the dependencies:**
   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
   *If PowerShell says "running scripts is disabled"*, run this once and try again:
   `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`
   *(or use Command Prompt: `.venv\Scripts\activate.bat`)*.
4. **Configure your e-mail (and optional API keys)** - see section 2. Quickest way:
   ```powershell
   copy .env.example .env
   notepad .env
   ```
   Put your real address after `CONTACT_EMAIL=` and save.
5. **Start the app:**
   ```powershell
   streamlit run app.py
   ```
   Your browser opens at <http://localhost:8501>. (First start: Streamlit may ask for an e-mail - just press Enter.)
6. **Next time**, only steps `.\.venv\Scripts\Activate.ps1` and `streamlit run app.py` are needed.

Downloaded papers appear in `Downloaded_Papers\` next to `app.py`; the history database is `research_history.sqlite3`.
Windows tips: paths longer than 260 characters can fail - keep the project in a short folder such as `C:\research\academic_agent`;
file names are sanitised automatically (no reserved names like `CON`, no `:` or `?`).

macOS / Linux: identical, except `python3 -m venv .venv && source .venv/bin/activate`.

### Run the tests
```powershell
python -m pytest                       # 190+ offline tests, no internet needed, ~2 seconds
$env:RUN_LIVE_TESTS = "1"; python -m pytest tests/test_live_smoke.py     # optional real-internet checks
```

---

## 2. API setup

The agent works **without any key** (Crossref, OpenAlex, Semantic Scholar, Europe PMC, DOAJ, arXiv and OpenAIRE need none),
but each key below unlocks more sources or higher limits. Keys are read **only from environment variables** (or the local `.env`
file, which is git-ignored). They are never logged, shown in the UI, or sent anywhere except the provider they belong to.

| Variable | Needed for | How to get it |
|---|---|---|
| `CONTACT_EMAIL` | **Unpaywall** (required by them), faster "polite pool" at Crossref/OpenAlex, identifies the bot politely | Your real e-mail address. Fake/`example.com` addresses are rejected by Unpaywall. |
| `CORE_API_KEY` | CORE - full texts from thousands of institutional repositories | Free: <https://core.ac.uk/services/api> → *Register for an API key* |
| `SEMANTIC_SCHOLAR_API_KEY` | Higher Semantic Scholar rate limit (without a key you share a pool and often see HTTP 429) | Free: <https://www.semanticscholar.org/product/api#api-key> |
| `OPENALEX_API_KEY` | Higher OpenAlex limits if the free pool is saturated | <https://openalex.org> (account settings) |
| `BRAVE_API_KEY` | **Open-web search** for repository copies and PDFs ("title" PDF, "open access", "repository", …) | <https://api.search.brave.com> (has a free tier) |
| `GOOGLE_CSE_API_KEY` + `GOOGLE_CSE_CX` | Alternative open-web search through a Google Programmable Search Engine | Create an engine at <https://programmablesearchengine.google.com> (copy its *Search engine ID* = `GOOGLE_CSE_CX`), enable the *Custom Search API* in Google Cloud and create an API key |

Setting a variable for the current PowerShell window only:
```powershell
$env:CONTACT_EMAIL = "you@university.edu"
$env:CORE_API_KEY  = "xxxxxxxx"
streamlit run app.py
```
Permanently (new windows): `setx CONTACT_EMAIL "you@university.edu"` (then open a new terminal) - or just use the `.env` file.

The sidebar shows which sources are ready (green) and what each missing one needs. If a provider is down or over quota the agent
says so and carries on with the others.

> **Web search is optional but valuable.** Without Brave/Google credentials the agent still searches ten scholarly
> databases and crawls publisher/repository pages, but it cannot search the open web for a copy on a university site. It tells you
> when that step was skipped. There is deliberately **no scraping of consumer search engines**.

---

## 3. How a request is handled

| Stage | What happens |
|---|---|
| **Understand** | Detects DOI / arXiv id / author+year / exact title / research question, and the field (biomedical vs quantitative) to choose sources. |
| **Identify** | Queries the selected providers *in parallel*, merges records describing the same work (also preprint ↔ journal versions), and scores title similarity (normalised, order-sensitive). Only a (near-)verbatim title is treated as the requested paper. Similar titles (`…Default Rates` vs `…Default Risk`), several different papers with the same title, or a topic query → **ranked list, you choose** (nothing is downloaded). |
| **Cross-check** | Re-queries the DOI in the other sources and reports how many agree on title/year/DOI. |
| **Discover** | Collects locations from OpenAlex, Unpaywall, Semantic Scholar, Europe PMC, CORE, Crossref (links and preprint relations), DOAJ, arXiv, OpenAIRE, plus the DOI landing page. |
| **Rank** | 1 official open-access publisher version → 2 institutional repository → 3 trusted subject/general repository → 4 author-accepted manuscript → 5 preprint → 6 anything found on the open web. |
| **Examine** | For each candidate, best first: *probe* (real request) → if it is an HTML page, *crawl* it (citation meta tags, `rel=alternate` PDFs, "Full text/Download PDF" links, repository bitstreams, JSON-LD; follow relevant links within the depth/page budget) → *download* → *verify* the content. A wrong document, login page, paywall, CAPTCHA, 404, truncated or corrupt file is recorded and the next candidate is tried. |
| **Alternatives** | If nothing worked: the exact title in quotes, title + first author, the DOI, title + `PDF` / `open access` / `repository` / `accepted manuscript` / `full text` are sent to the web-search provider, and the scholarly databases are re-queried with title + first author. |
| **Decide** | Downloads the most authoritative **verified** version; labels it *final published article / accepted manuscript / preprint*. Otherwise returns the verified record with its official link and the full list of what was checked. |

### What "verified" means
After download the PDF text (first pages) is compared with the expected record: the title (exact or fuzzy window match), the author
surnames, the abstract overlap and the DOI. *Verified* = title found + authors consistent (or DOI + title). *Partially verified* = the title
differs between versions (typical for preprints) but authors and abstract agree - shown clearly, can be switched off in the sidebar.
A near-identical title with different authors, a scanned file without text, an encrypted file or a file containing a `/Launch` action are rejected.

### Access status badges
**Final published article** · **Accepted manuscript** · **Preprint (not the final article)** · **Full text - version not stated** ·
**Abstract-only record** (record and abstract known; no legal full text found, no restriction seen) · **Inaccessible** (paywall/login/CAPTCHA/forbidden observed or no open copy).
The version is taken from the source's metadata and corrected when the document itself says "accepted manuscript" / "preprint".

---

## 4. Security, reliability and compliance

* **SSRF** - every URL and every redirect hop is validated: http(s) only, no credentials, ports 80/443/8080/8443, all resolved addresses must be
  public (loopback, private, link-local, CGNAT, multicast, cloud-metadata, IPv4-mapped/6to4/Teredo/NAT64 forms, numeric `2130706433`/`0x7f000001` hosts are refused);
  on direct connections the *connected peer address* is checked as well (DNS-rebinding defence). If you run behind a corporate proxy, DNS validation still happens locally.
* **Untrusted content** - HTML is parsed, never executed; PDFs are only parsed (pypdf), never rendered; JSON/XML are size-limited and XML uses `defusedxml`
  (entity bombs rejected); text inside pages/PDFs is evidence only - embedded "instructions" are ignored; titles/abstracts are Markdown-escaped before display.
* **Limits** - per-host rate limits (e.g. arXiv 1 request/3 s), request timeouts, bounded retries with back-off and `Retry-After`, a 45 s cool-down for hosts returning 429,
  a global request budget, wall-clock limit, page/depth/candidate/download caps, 60 MB PDF cap (checked from headers *and* while streaming), 6 MB JSON cap.
* **Compliance** - `robots.txt` is honoured for crawling and downloads (RFC 9309 semantics, cached); paywalls, logins and CAPTCHAs are classified and left alone;
  shadow libraries (Sci-Hub, LibGen, Z-Library, Anna's Archive, …) are on a block-list and never contacted; sites whose terms forbid automation
  (ResearchGate, Academia.edu, ScienceDirect, …) are listed as links for you to open but never requested. Both lists are configurable
  (`EXTRA_BLOCKED_DOMAINS`, `EXTRA_NO_CRAWL_DOMAINS`). Only legally open copies are fetched; check your own institution's licence terms for reuse.
* **No fabricated results** - every URL shown came from a source or from the page it was found on; a "downloaded" result requires a validated, content-verified file.

---

## 5. Project layout

| File | Role |
|---|---|
| `app.py` | Streamlit UI (research, history, setup help). Only renders `ResearchResult`. |
| `research_agent.py` | Orchestration: identify → enrich → discover → rank → examine → alternatives → decide; progress events; summary + citations. |
| `search_providers.py` | Provider abstraction (`SearchProvider`, `WebSearchProvider`, `ProviderRegistry`) and the 10 scholarly + 2 web integrations. |
| `paper_match.py` | Query classification, title normalisation/similarity, author matching, de-duplication/merging, host classification, filenames. |
| `web_crawler.py` | Visits permitted pages; extracts metadata and full-text links; bounded depth/pages. |
| `access_checker.py` | Real-response classification: PDF / HTML / login / paywall / CAPTCHA / robots / errors. |
| `pdf_downloader.py` | Streaming, size-limited, retrying download into a temp file; atomic move to `Downloaded_Papers`. |
| `pdf_verifier.py` | Structure checks, text extraction, title/author/DOI/abstract verification, version hints, active-content flags. |
| `regulation_match.py`, `regulation_finder.py` | Regulation reference parsing, OJK database search, official-PDF verification. |
| `batch.py` | Multi-line input parsing, result table / list / CSV. |
| `database.py` | SQLite: papers, downloads (SHA-256), attempts, runs, searchable history. |
| `http_client.py`, `net_safety.py` | Hardened HTTP layer and URL validation. |
| `config.py`, `models.py` | Settings (env/.env) and shared data classes. |
| `tests/` | 190+ offline tests (fake web, generated PDFs) + optional live smoke tests. |

### Adding a source
Subclass `SearchProvider` (or `WebSearchProvider`) in `search_providers.py`, implement whichever of `search`, `lookup_doi`, `find_fulltext`
apply, declare `relevance(query_info)` and `available()` (credentials), and add the class to `DEFAULT_PROVIDER_CLASSES`. Nothing else changes.

### Using it without the UI
```python
from research_agent import ResearchAgent
result = ResearchAgent().run("10.1371/journal.pone.0000308", progress=lambda e: print(e.stage, e.message))
print(result.outcome, result.access_state, result.download and result.download.path)
print(result.to_dict())          # JSON-serialisable: ready for a FastAPI / React frontend
```

---

## 6. Honest limitations

* **Paywalled papers with no open copy cannot be retrieved** - by design. You get the verified record, the official link and the evidence trail.
* Some repositories (e.g. Europe PMC PDF rendering, Harvard DASH) sit behind anti-bot protection; those are reported as *CAPTCHA/forbidden* and skipped, not bypassed.
* Free API pools (OpenAlex, Semantic Scholar) are shared and may answer HTTP 429; add keys for reliability. The agent degrades gracefully and says which source was unavailable.
* Title matching is heuristic: titles shared by many papers (e.g. *"Attention Is All You Need"*) produce a "choose one" list rather than a guess.
* Scanned PDFs without a text layer cannot be content-verified and are therefore not kept.
* Version labels depend on source metadata and phrases in the document; when neither is conclusive the label is "version not stated".
