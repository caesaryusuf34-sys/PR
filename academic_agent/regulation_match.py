"""Indonesian legal-instrument references: parsing, canonical forms, page/slug matching and text verification.

Examples understood::

    SEOJK No. 19/SEOJK.06/2025          POJK Nomor 40 Tahun 2024          UU No. 4 Tahun 2023
    Surat Edaran OJK Nomor 19/SEOJK.06/2025 tentang ...      PP 71 Tahun 2019      23/6/PBI/2021

The reference must *lead* the query (so a paper title that merely mentions a regulation is not hijacked).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

# canonical kind -> (issuer, OJK-site "Jenis Regulasi" dropdown value or None, human name)
KINDS: dict[str, tuple[str, Optional[str], str]] = {
    "SEOJK": ("Otoritas Jasa Keuangan", "Surat Edaran OJK", "Surat Edaran OJK"),
    "POJK": ("Otoritas Jasa Keuangan", "Peraturan OJK", "Peraturan OJK"),
    "PADK": ("Otoritas Jasa Keuangan", "Peraturan ADK", "Peraturan Anggota Dewan Komisioner OJK"),
    "UU": ("Pemerintah / DPR", "Undang-Undang", "Undang-Undang"),
    "PERPPU": ("Presiden", None, "Peraturan Pemerintah Pengganti Undang-Undang"),
    "PP": ("Pemerintah", "Peraturan Pemerintah", "Peraturan Pemerintah"),
    "PERPRES": ("Presiden", None, "Peraturan Presiden"),
    "PMK": ("Menteri Keuangan", "Peraturan/Keputusan Mentri", "Peraturan Menteri Keuangan"),
    "PBI": ("Bank Indonesia", "PPBI", "Peraturan Bank Indonesia"),
    "PADG": ("Bank Indonesia", None, "Peraturan Anggota Dewan Gubernur BI"),
    "SEBI": ("Bank Indonesia", "SEBI", "Surat Edaran Bank Indonesia"),
}

# words/abbreviations -> canonical kind (longest first matters, handled by ordering)
_KIND_WORDS = [
    (r"surat\s+edaran\s+(?:otoritas\s+jasa\s+keuangan|ojk)", "SEOJK"), (r"se\s*ojk", "SEOJK"),
    (r"peraturan\s+(?:otoritas\s+jasa\s+keuangan|ojk)", "POJK"), (r"pojk", "POJK"),
    (r"peraturan\s+anggota\s+dewan\s+komisioner(?:\s+ojk)?", "PADK"), (r"padk", "PADK"),
    (r"peraturan\s+pemerintah\s+pengganti\s+undang[\s-]*undang", "PERPPU"), (r"perpu{1,2}", "PERPPU"),
    (r"peraturan\s+pemerintah", "PP"), (r"pp", "PP"),
    (r"peraturan\s+presiden", "PERPRES"), (r"perpres", "PERPRES"),
    (r"peraturan\s+menteri\s+keuangan", "PMK"), (r"pmk", "PMK"),
    (r"peraturan\s+bank\s+indonesia", "PBI"), (r"ppbi", "PBI"), (r"pbi", "PBI"),
    (r"peraturan\s+anggota\s+dewan\s+gubernur(?:\s+bi)?", "PADG"), (r"padg", "PADG"),
    (r"surat\s+edaran\s+bank\s+indonesia", "SEBI"), (r"sebi", "SEBI"),
    (r"undang[\s-]*undang", "UU"), (r"uu", "UU"),
]
_KIND_ALT = "|".join(f"(?P<k{i}>{rx})" for i, (rx, _) in enumerate(_KIND_WORDS))
_LEAD = r"^(?:regulasi|peraturan\s+tentang|cari(?:kan)?|tolong|link|tautan|pdf|dokumen|file)?\s*(?:link|pdf|tautan)?\s*(?:dari|untuk|:)?\s*"
_NUM_NOMOR = r"(?:nomor|no\.?|nr\.?|number)?\s*"

# 19/SEOJK.06/2025 | 23/6/PBI/2021 | 12/POJK.05/2022
_SLASH = re.compile(r"(?P<n>\d{1,4}(?:\s*/\s*\d{1,3})?)\s*/\s*(?P<code>SEOJK|POJK|PADK|PBI|PPBI|PADG|SEBI|PMK)(?:\s*[.\-]?\s*(?P<sec>\d{1,3}))?\s*/\s*(?P<y>(?:19|20)\d{2})", re.I)
# word form: "<kind> [Nomor] 40 [(/code)] Tahun 2024"
_WORD = re.compile(_LEAD + r"(?:" + _KIND_ALT + r")\s*(?:ri|republik\s+indonesia)?\s*" + _NUM_NOMOR +
                   r"(?P<n>\d{1,4})\s*(?:tahun|thn\.?|th\.?)\s*(?P<y>(?:19|20)\d{2})(?P<rest>.*)$", re.I | re.S)
_TENTANG = re.compile(r"\btentang\b\s*(.+)$", re.I | re.S)


@dataclass
class RegRef:
    kind: str                     # canonical kind key of KINDS
    number: str                   # "19" or "23/6"
    year: int
    code: str = ""                # e.g. "SEOJK.06" (as printed), "" for 'Tahun' forms
    sector: str = ""              # "06"
    title: str = ""               # text after "tentang"
    raw: str = ""
    slash_form: bool = False

    @property
    def issuer(self) -> str:
        return KINDS[self.kind][0]

    @property
    def kind_name(self) -> str:
        return KINDS[self.kind][2]

    @property
    def ojk_jenis(self) -> Optional[str]:
        return KINDS[self.kind][1]

    @property
    def canonical(self) -> str:
        if self.slash_form:
            code = f"{self.kind}.{self.sector}" if self.sector else self.kind
            return f"{self.kind_name} Nomor {self.number}/{code}/{self.year}"
        return f"{self.kind_name} Nomor {self.number} Tahun {self.year}"

    @property
    def short(self) -> str:
        if self.slash_form:
            code = f"{self.kind}.{self.sector}" if self.sector else self.kind
            return f"{self.kind} {self.number}/{code}/{self.year}"
        return f"{self.kind} {self.number}/{self.year}"

    @property
    def first_number(self) -> str:
        return self.number.split("/")[0].strip()


def _squash(s: str) -> str:
    return re.sub(r"[^0-9a-z]", "", (s or "").lower())


def parse_regulation_ref(text: str) -> Optional[RegRef]:
    """Return a RegRef if ``text`` starts with (or is essentially) an Indonesian regulation reference."""
    q = re.sub(r"\s+", " ", (text or "").strip().strip("\"'\u201c\u201d"))
    if not q or len(q.split()) > 40:
        return None
    m = _SLASH.search(q)
    if m:
        before, after = q[: m.start()], q[m.end():]
        # the rest of the text must be a short qualifier ("SEOJK No.", "Surat Edaran OJK Nomor", "tentang ...") only
        if len(re.sub(r"\b(?:no\.?|nomor|nr\.?|ri|tahun|ojk|surat|edaran|peraturan|otoritas|jasa|keuangan|bank|indonesia|"
                      r"pemerintah|undang[\s-]*undang|link|pdf|dari|untuk|cari|regulasi|dokumen|file|tolong|"
                      r"seojk|pojk|padk|pbi|ppbi|padg|sebi|pmk|menteri|keuangan)\b[:.]?", " ", before, flags=re.I).split()) <= 1:
            code = m.group("code").upper()
            kind = {"PPBI": "PBI"}.get(code, code)
            sec = (m.group("sec") or "").zfill(2) if m.group("sec") else ""
            t = _TENTANG.search(after)
            return RegRef(kind=kind, number=re.sub(r"\s+", "", m.group("n")), year=int(m.group("y")),
                          code=f"{kind}.{sec}" if sec else kind, sector=sec, title=(t.group(1).strip(" .") if t else ""),
                          raw=q, slash_form=True)
    m = _WORD.match(q)
    if m:
        kind = next((_KIND_WORDS[i][1] for i in range(len(_KIND_WORDS)) if m.group(f"k{i}")), None)
        if kind:
            t = _TENTANG.search(m.group("rest") or "")
            return RegRef(kind=kind, number=m.group("n"), year=int(m.group("y")), title=(t.group(1).strip(" .") if t else ""), raw=q)
    return None


def slug_keys(ref: RegRef) -> list[str]:
    """Squashed strings that identify the regulation inside a page slug / title (e.g. ``seojk19seojk062025``)."""
    n, y, k = ref.first_number, str(ref.year), ref.kind.lower()
    keys = []
    if ref.slash_form:
        code = _squash(ref.code or ref.kind)
        keys += [f"{k}{n}{code}{y}", f"{k}nomor{n}{code}{y}", f"{n}{code}{y}"]
        if ref.number != n:
            keys.append(f"{k}{_squash(ref.number)}{code}{y}")
    else:
        keys += [f"{k}{n}tahun{y}", f"{k}nomor{n}tahun{y}", f"{k}no{n}tahun{y}", f"{k}{n}{y}"]
        for word, kk in _KIND_WORDS:
            if kk == ref.kind and " " in word.replace(r"\s+", " "):
                full = _squash(re.sub(r"\\s[+*]|\(\?:|\)|\||\?|\\", "", word.split("|")[0]))
                keys.append(f"{full}nomor{n}tahun{y}")
        if ref.kind == "UU":
            keys.append(f"undangundangnomor{n}tahun{y}")
    return list(dict.fromkeys(k for k in keys if k))


def text_keys(ref: RegRef) -> list[str]:
    """Squashed strings expected in the *document text* (identification line of the regulation)."""
    n, y = ref.first_number, str(ref.year)
    keys = [f"nomor{n}tahun{y}", f"no{n}tahun{y}"]
    if ref.slash_form:
        code = _squash(ref.code or ref.kind)
        keys += [f"{_squash(ref.number)}{code}{y}", f"nomor{_squash(ref.number)}{code}{y}"]
    return list(dict.fromkeys(keys))


def slug_matches(ref: RegRef, *texts: str) -> bool:
    raw = " ".join(texts).lower()
    blob = _squash(raw)
    if any(k in blob for k in slug_keys(ref)):
        return True
    if not ref.slash_form and ref.kind in ("SEOJK", "POJK", "PADK"):   # "SEOJK 19 Tahun 2025" vs page "SEOJK-19-SEOJK06-2025"
        k, n, y = ref.kind.lower(), re.escape(ref.first_number), ref.year
        return bool(re.search(rf"{k}[\s\-_.]*(?:nomor[\s\-_.]*)?{n}[\s\-_./]*(?:{k}[\s\-_.]*\d{{1,3}}[\s\-_./]*)?(?:tahun[\s\-_.]*)?{y}", raw))
    return False


@dataclass
class RegVerification:
    verdict: str                  # verified | official_page | mismatch | unreadable | corrupt
    reasons: list = field(default_factory=list)
    pages: int = 0
    excerpt: str = ""
    warnings: list = field(default_factory=list)

    @property
    def accepted(self) -> bool:
        return self.verdict in ("verified", "official_page")

    def to_dict(self) -> dict:
        return {"verdict": self.verdict, "reasons": self.reasons, "page_count": self.pages, "excerpt": self.excerpt,
                "warnings": self.warnings, "title_score": None, "authors_score": None, "abstract_score": None,
                "doi_found": False, "version_evidence": []}


def verify_regulation_text(ref: RegRef, text: str, pages: int, page_confirmed: bool, warnings: Optional[list] = None) -> RegVerification:
    """Is this the requested regulation? Its identification line must appear in the first pages."""
    squashed = _squash(text[:20000])
    excerpt = re.sub(r"\s+", " ", text)[:400]
    base = dict(pages=pages, excerpt=excerpt, warnings=list(warnings or []))
    if len(squashed) < 80:
        if page_confirmed:
            return RegVerification("official_page", [f"the PDF has no text layer (scanned); accepted because the official page for "
                                                      f"{ref.canonical} links to it"], **base)
        return RegVerification("unreadable", ["no extractable text (scanned) and no page evidence - cannot verify"], **base)
    hits = [k for k in text_keys(ref) if k in squashed]
    if hits:
        return RegVerification("verified", [f"identification '{ref.canonical}' found in the document text"], **base)
    return RegVerification("mismatch", [f"document text does not contain '{ref.canonical}' (it may be another or an amended regulation)"], **base)


def reg_filename(ref: RegRef, title: str = "", max_len: int = 110) -> str:
    base = re.sub(r"[^A-Za-z0-9]+", "_", f"{ref.kind}_{ref.number}_{ref.sector + '_' if ref.sector else ''}{ref.year}").strip("_")
    t = re.sub(r"[^A-Za-z0-9]+", "_", " ".join((title or ref.title).split()[:8])).strip("_")
    return (f"{base}_{t}" if t else base)[:max_len].rstrip("_") + ".pdf"
