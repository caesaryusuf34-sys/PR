"""Batch input parsing and result tables (several papers and/or regulations in one go)."""
from __future__ import annotations

import csv
import io
import re
from typing import Iterable

from models import Outcome, ResearchResult

_BULLET = re.compile(r"^\s*(?:\(?\d{1,3}[.)\]]|[-*•‣▪●–])\s+")


def split_queries(text: str, limit: int = 25) -> list[str]:
    """One query per non-empty line. Leading numbering / bullets are removed; exact duplicates are dropped.

    Titles, DOIs, APA citations and regulation numbers can all be mixed. (Keep each reference on a single line.)
    """
    out: list[str] = []
    seen: set[str] = set()
    for line in (text or "").splitlines():
        q = _BULLET.sub("", line).strip()
        if len(q) < 3:
            continue
        key = re.sub(r"\s+", " ", q).lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(q)
    return out[:limit]


STATUS_TEXT = {
    Outcome.LINK_FOUND: "PDF link verified", Outcome.DOWNLOADED: "PDF saved",
    Outcome.NEEDS_CHOICE: "Choose the intended paper", Outcome.NO_FULLTEXT: "Identified, no legal PDF found",
    Outcome.NOT_FOUND: "Not found",
}


def result_title(r: ResearchResult) -> str:
    return r.paper.title if r.paper else ""


def pdf_link(r: ResearchResult) -> str:
    return r.download.source_url if r.download else ""


def batch_rows(results: Iterable[ResearchResult]) -> list[dict]:
    rows = []
    for i, r in enumerate(results, 1):
        rows.append({"#": i, "query": r.query, "status": STATUS_TEXT[r.outcome], "pdf_link": pdf_link(r),
                     "title": result_title(r), "year": r.paper.year if r.paper else None,
                     "version": r.download.version.label if r.download else "", "type": r.query_type.value,
                     "article_page": (r.download.landing_url if r.download else "") or r.official_link,
                     "saved_file": r.download.path if r.download and r.download.saved else ""})
    return rows


def batch_markdown(results: Iterable[ResearchResult]) -> str:
    """Copy-friendly numbered list: title, status and the PDF link (or the reason there is none)."""
    lines = []
    for row in batch_rows(results):
        head = f"{row['#']}. {row['title'] or row['query']}" + (f" ({row['year']})" if row["year"] else "")
        if row["pdf_link"]:
            lines.append(f"{head}\n   PDF: {row['pdf_link']}\n   [{row['version']}]")
        else:
            extra = f"\n   Page: {row['article_page']}" if row["article_page"] else ""
            lines.append(f"{head}\n   {row['status']} - no verified PDF link{extra}")
    return "\n".join(lines)


def rows_to_csv(rows: list[dict]) -> str:
    if not rows:
        return ""
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()
