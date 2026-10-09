"""Decide whether a URL really gives access to a document.

A link that merely *mentions* "PDF" proves nothing. :class:`AccessChecker` performs a real request,
looks at the status code, the bytes (``%PDF-`` signature) and the HTML (login / paywall / CAPTCHA
pages) and classifies the outcome. It never tries to get past an access control: a login page,
a paywall or a CAPTCHA is reported as such and the agent moves on to another source.
"""
from __future__ import annotations

import re
from typing import Optional
from urllib.parse import urlsplit

from http_client import (BlockedDomain, BudgetExceeded, HttpClient, HttpError, RateLimited, ResponseTooLarge,
                         RobotsDisallowed, decode_body)
from models import ProbeResult, ProbeStatus
from net_safety import UnresolvableHost, URLValidationError

PDF_MAGIC = b"%PDF-"
_LOGIN_URL = re.compile(r"(/login|/log-in|/signin|/sign-in|/sso\b|/auth(?:orize|enticate)?\b|shibboleth|/idp/|wayf|"
                        r"openathens|ezproxy|/cas/login|/saml|institutional[-_]?login|/account/login|/user/login)", re.I)
_LOGIN_TEXT = re.compile(r"(sign in to (?:access|continue|view|read)|log in to (?:access|continue|view|read)|"
                         r"institutional (?:login|access|sign.?in)|access through your institution|log in via your institution|"
                         r"please (?:log|sign) in|login required|authentication required|select your institution|"
                         r"you need to (?:log|sign) in)", re.I)
_PAYWALL_TEXT = re.compile(r"(purchase (?:this )?(?:article|chapter|pdf|access)|buy (?:this )?(?:article|chapter|pdf|now)|"
                           r"rent (?:this )?article|get (?:full )?access to this (?:article|content)|subscribe to (?:read|view|access|continue)|"
                           r"subscription required|pay[- ]per[- ]view|full text (?:is )?not available|"
                           r"this content is (?:only )?available (?:to|for) subscribers|access this article|"
                           r"to read the full[- ]text|unlock this article|check access|view access options|"
                           r"you do not have access|your institution does not have access|add to cart)", re.I)
_CAPTCHA_TEXT = re.compile(r"(captcha|are you (?:a )?(?:robot|human)|verify you are (?:a )?human|just a moment\.\.\.|"
                           r"checking your browser|cf-chl|attention required|unusual traffic|security check|"
                           r"access denied|request blocked|bot detection|enable javascript and cookies)", re.I)
_OA_TEXT = re.compile(r"(open access|creative commons|cc[- ]by|free to read|free access|freely available|"
                      r"this article is (?:freely )?available|this is an open access)", re.I)
_HTML_HINT = re.compile(rb"<\s*(!doctype\s+html|html|head|body|meta|title)\b", re.I)


def looks_like_pdf(head: bytes) -> bool:
    return PDF_MAGIC in head[:1024]


def looks_like_html(head: bytes) -> bool:
    return bool(_HTML_HINT.search(head[:2048]))


def classify_html(url: str, html: str, http_status: int = 200) -> tuple[ProbeStatus, str]:
    """Classify an HTML response as a usable page or as a wall (CAPTCHA / login / paywall)."""
    low = html[:60000]
    visible = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", low, flags=re.S | re.I)
    visible = re.sub(r"<[^>]+>", " ", visible)
    visible = re.sub(r"\s+", " ", visible)
    title_m = re.search(r"<title[^>]*>(.*?)</title>", low, re.S | re.I)
    title = re.sub(r"\s+", " ", title_m.group(1)).strip() if title_m else ""
    has_pdf_meta = bool(re.search(r"citation_pdf_url", low, re.I))
    short_page = len(visible) < 1500
    if _CAPTCHA_TEXT.search(title) or (short_page and _CAPTCHA_TEXT.search(visible)):
        return ProbeStatus.CAPTCHA, "anti-bot / CAPTCHA page (not bypassed)"
    has_password_field = bool(re.search(r"<input[^>]+type=[\"']?password", low, re.I))
    if _LOGIN_URL.search(urlsplit(url).path + "?" + urlsplit(url).query) and (has_password_field or _LOGIN_TEXT.search(visible)):
        return ProbeStatus.LOGIN_REQUIRED, "redirected to a login / institutional-authentication page"
    if has_password_field and short_page:
        return ProbeStatus.LOGIN_REQUIRED, "page asks for a password"
    if _LOGIN_TEXT.search(title) and not has_pdf_meta:
        return ProbeStatus.LOGIN_REQUIRED, f"login page ({title[:80]})"
    pay = _PAYWALL_TEXT.search(visible)
    if pay and not _OA_TEXT.search(visible):
        return ProbeStatus.PAYWALL, f"paywall indicator: '{pay.group(0)[:60]}'"
    return ProbeStatus.HTML_PAGE, "HTML page"


class AccessChecker:
    def __init__(self, http: HttpClient):
        self.http = http
        self.settings = http.settings

    def probe(self, url: str, *, check_robots: bool = True, referer: Optional[str] = None) -> ProbeResult:
        """Issue a real GET and classify what comes back. Never raises for network problems."""
        headers = {"Accept": "application/pdf,text/html;q=0.9,application/xhtml+xml;q=0.8,*/*;q=0.5"}
        if referer:
            headers["Referer"] = referer
        if self.http.is_no_crawl(url):
            return ProbeResult(url, ProbeStatus.ROBOTS_BLOCKED,
                               detail="site forbids automated access (terms of use); open it manually")
        try:
            resp = self.http.request("GET", url, headers=headers, stream=True, check_robots=check_robots)
        except RobotsDisallowed as exc:
            return ProbeResult(url, ProbeStatus.ROBOTS_BLOCKED, detail=str(exc))
        except BlockedDomain as exc:
            return ProbeResult(url, ProbeStatus.BLOCKED_DOMAIN, detail=str(exc))
        except UnresolvableHost as exc:
            return ProbeResult(url, ProbeStatus.NETWORK_ERROR, detail=str(exc))
        except URLValidationError as exc:
            return ProbeResult(url, ProbeStatus.UNSAFE_URL, detail=str(exc))
        except BudgetExceeded as exc:
            return ProbeResult(url, ProbeStatus.BUDGET, detail=str(exc))
        except RateLimited as exc:
            return ProbeResult(url, ProbeStatus.RATE_LIMITED, detail=str(exc), http_status=429)
        except HttpError as exc:
            return ProbeResult(url, ProbeStatus.NETWORK_ERROR, detail=str(exc))
        return self.classify_response(url, resp)

    def classify_response(self, url: str, resp) -> ProbeResult:
        """Classify an already-open streaming response (closes it)."""
        ctype = resp.headers.get("Content-Type", "").split(";")[0].strip().lower()
        final = resp.url or url
        status = resp.status_code
        clen = resp.headers.get("Content-Length")
        clen_i = int(clen) if clen and clen.isdigit() else None
        base = dict(url=url, final_url=final, http_status=status, content_type=ctype, content_length=clen_i)
        try:
            if status in (401, 407):
                return ProbeResult(status=ProbeStatus.LOGIN_REQUIRED, detail=f"HTTP {status}: authentication required", **base)
            if status == 402:
                return ProbeResult(status=ProbeStatus.PAYWALL, detail="HTTP 402: payment required", **base)
            if status == 429:
                return ProbeResult(status=ProbeStatus.RATE_LIMITED, detail="HTTP 429: rate limited", **base)
            if status in (404, 410):
                return ProbeResult(status=ProbeStatus.NOT_FOUND, detail=f"HTTP {status}: not found", **base)
            if status >= 500:
                return ProbeResult(status=ProbeStatus.SERVER_ERROR, detail=f"HTTP {status}: server error", **base)
            chunks = iter(resp.iter_content(chunk_size=8192))       # ONE iterator for the whole body (a second one would be empty)
            try:
                head = b""
                for chunk in chunks:
                    head += chunk
                    if len(head) >= 8192:
                        break
            except Exception as exc:  # network failure mid-read
                return ProbeResult(status=ProbeStatus.NETWORK_ERROR, detail=f"read error: {exc}", **base)
            if status == 403 or status >= 400:
                html = decode_body(head, ctype)
                cls, why = classify_html(final, html, status) if looks_like_html(head) else (None, "")
                if cls in (ProbeStatus.CAPTCHA, ProbeStatus.PAYWALL, ProbeStatus.LOGIN_REQUIRED):
                    return ProbeResult(status=cls, detail=f"HTTP {status}: {why}", **base)
                return ProbeResult(status=ProbeStatus.FORBIDDEN if status == 403 else ProbeStatus.NOT_FOUND,
                                   detail=f"HTTP {status}: access refused", **base)
            if not head:
                return ProbeResult(status=ProbeStatus.EMPTY, detail="empty response body", **base)
            if looks_like_pdf(head):
                if clen_i and clen_i > self.settings.max_pdf_bytes:
                    return ProbeResult(status=ProbeStatus.TOO_LARGE,
                                       detail=f"PDF is {clen_i / 1048576:.0f} MB (limit {self.settings.max_pdf_mb} MB)", **base)
                note = "" if "pdf" in ctype else f" (served as {ctype or 'unknown type'})"
                return ProbeResult(status=ProbeStatus.PDF, detail="valid %PDF signature" + note, **base)
            if looks_like_html(head) or ctype in ("text/html", "application/xhtml+xml"):
                rest = b""
                try:
                    limit = self.settings.max_html_kb * 1024 - len(head)
                    for chunk in chunks:
                        rest += chunk
                        if len(rest) >= limit:
                            break
                except Exception:
                    pass
                html = decode_body(head + rest, ctype)
                cls, why = classify_html(final, html, status)
                return ProbeResult(status=cls, detail=why, html=html, **base)
            return ProbeResult(status=ProbeStatus.NOT_PDF, detail=f"not a PDF (content-type {ctype or 'unknown'})", **base)
        finally:
            resp.close()
