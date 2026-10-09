"""Hardened HTTP layer used by every provider, the crawler, the access checker and the downloader.

* every request and every redirect hop is validated against SSRF rules (``net_safety``);
* the address actually connected to is re-checked (DNS-rebinding defence, direct connections);
* per-host rate limiting, bounded retries with back-off, ``Retry-After`` handling and cool-down;
* a shared request budget and wall-clock deadline for one research run;
* robots.txt support (RFC 9309 semantics) for crawling and file downloads;
* size-limited body readers - retrieved content is always treated as untrusted data.
"""
from __future__ import annotations

import ipaddress
import json
import random
import threading
import time
from typing import Callable, Optional
from urllib import robotparser
from urllib.parse import urljoin, urlsplit

import requests
from requests.adapters import HTTPAdapter
from urllib3.connection import HTTPConnection, HTTPSConnection
from urllib3.connectionpool import HTTPConnectionPool, HTTPSConnectionPool

from config import APP_NAME, Settings
from net_safety import UnsafeURL, URLValidationError, host_matches, ip_is_public, validate_url

REDIRECT_CODES = (301, 302, 303, 307, 308)
RETRY_STATUS = (429, 500, 502, 503, 504)


class HttpError(Exception):
    def __init__(self, message: str, status: Optional[int] = None):
        super().__init__(message)
        self.status = status


class BudgetExceeded(HttpError):
    pass


class RateLimited(HttpError):
    pass


class RequestFailed(HttpError):
    pass


class RobotsDisallowed(HttpError):
    pass


class BlockedDomain(UnsafeURL):
    pass


class ResponseTooLarge(HttpError):
    pass


# ----------------------------------------------------------------------------- budget
class Budget:
    """Request counter + wall-clock deadline shared by everything in one research run."""

    def __init__(self, max_requests: int = 300, time_limit: float = 300.0):
        self.max_requests = max_requests
        self.time_limit = time_limit
        self.used = 0
        self.started = time.monotonic()
        self._lock = threading.Lock()

    def consume(self) -> None:
        with self._lock:
            if self.used >= self.max_requests:
                raise BudgetExceeded(f"request budget of {self.max_requests} exhausted")
            if self.expired():
                raise BudgetExceeded(f"time limit of {self.time_limit:.0f}s reached")
            self.used += 1

    def expired(self) -> bool:
        return (time.monotonic() - self.started) > self.time_limit

    @property
    def exhausted(self) -> bool:
        return self.used >= self.max_requests or self.expired()

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started


# ----------------------------------------------------------------------------- guarded connections
def _check_peer(sock) -> None:
    try:
        peer = sock.getpeername()[0]
    except OSError:
        return
    try:
        ip = ipaddress.ip_address(str(peer).split("%", 1)[0])
    except ValueError:
        return
    if not ip_is_public(ip):
        try:
            sock.close()
        finally:
            raise UnsafeURL(f"connected to non-public address {ip}")


class _GuardedHTTPConnection(HTTPConnection):
    def _new_conn(self):
        sock = super()._new_conn()
        _check_peer(sock)
        return sock


class _GuardedHTTPSConnection(HTTPSConnection):
    def _new_conn(self):
        sock = super()._new_conn()
        _check_peer(sock)
        return sock


class _GuardedHTTPPool(HTTPConnectionPool):
    ConnectionCls = _GuardedHTTPConnection


class _GuardedHTTPSPool(HTTPSConnectionPool):
    ConnectionCls = _GuardedHTTPSConnection


class GuardedAdapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        super().init_poolmanager(*args, **kwargs)
        self.poolmanager.pool_classes_by_scheme = {"http": _GuardedHTTPPool, "https": _GuardedHTTPSPool}


# ----------------------------------------------------------------------------- client
class HttpClient:
    def __init__(self, settings: Settings, *, session: Optional[requests.Session] = None,
                 resolver: Optional[Callable] = None, budget: Optional[Budget] = None,
                 sleeper: Callable[[float], None] = time.sleep):
        self.settings = settings
        self.resolver = resolver
        self.budget = budget or Budget(settings.max_requests, settings.time_limit_seconds)
        self._sleep = sleeper
        if session is None:
            session = requests.Session()
            adapter = GuardedAdapter(pool_connections=16, pool_maxsize=16)
            session.mount("http://", adapter)
            session.mount("https://", adapter)
        self.session = session
        self.session.headers.update({"User-Agent": settings.user_agent, "Accept-Language": "en;q=0.9"})
        self._last_hit: dict[str, float] = {}
        self._cooldown: dict[str, float] = {}
        self._rate_lock = threading.Lock()
        self._robots: dict[str, tuple[float, Optional[robotparser.RobotFileParser], str]] = {}
        self._robots_lock = threading.Lock()

    # ---- helpers ------------------------------------------------------------------------
    def reset_budget(self, budget: Optional[Budget] = None) -> Budget:
        self.budget = budget or Budget(self.settings.max_requests, self.settings.time_limit_seconds)
        with self._rate_lock:
            self._cooldown.clear()
        return self.budget

    def _validate(self, url: str) -> str:
        host = (urlsplit(url).hostname or "").lower()
        if host_matches(host, self.settings.blocked_domains):
            raise BlockedDomain(f"{host} is on the blocked-domain list (unauthorised distribution)")
        validate_url(url, self.resolver)
        return url

    def is_no_crawl(self, url: str) -> bool:
        return host_matches(urlsplit(url).hostname or "", self.settings.no_crawl_domains)

    def _throttle(self, host: str) -> None:
        interval = self.settings.host_intervals.get(host, self.settings.host_min_interval)
        with self._rate_lock:
            now = time.monotonic()
            until = self._cooldown.get(host, 0.0)
            if until > now:
                raise RateLimited(f"{host} is rate limiting this client; cooling down for {until - now:.0f}s", 429)
            wait = self._last_hit.get(host, 0.0) + interval - now
            self._last_hit[host] = now + max(wait, 0.0)
        if wait > 0:
            self._sleep(wait)

    def _backoff(self, attempt: int, retry_after: Optional[str] = None) -> float:
        if retry_after:
            try:
                return min(float(retry_after), 8.0)
            except ValueError:
                pass
        return min(0.8 * (2 ** attempt) + random.uniform(0, 0.4), 8.0)

    def _send(self, method, url, *, params, headers, timeout, stream, retries, data=None):
        host = urlsplit(url).hostname or ""
        last_exc: Optional[Exception] = None
        for attempt in range(retries + 1):
            self._throttle(host)
            self.budget.consume()
            try:
                resp = self.session.request(method, url, params=params, headers=headers, timeout=timeout,
                                            stream=stream, allow_redirects=False, data=data)
            except UnsafeURL:
                raise
            except (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError) as exc:
                last_exc = exc
                if isinstance(exc, requests.exceptions.SSLError) or attempt >= retries:
                    break
                self._sleep(self._backoff(attempt))
                continue
            except requests.RequestException as exc:
                raise RequestFailed(f"{type(exc).__name__}: {exc}") from exc
            if resp.status_code in RETRY_STATUS and attempt < retries:
                delay = self._backoff(attempt, resp.headers.get("Retry-After"))
                resp.close()
                self._sleep(delay)
                continue
            if resp.status_code == 429:
                with self._rate_lock:
                    self._cooldown[host] = time.monotonic() + 45.0
            return resp
        raise RequestFailed(f"network error for {host}: {type(last_exc).__name__}: {last_exc}")

    # ---- public API ---------------------------------------------------------------------
    def request(self, method: str, url: str, *, params: Optional[dict] = None, headers: Optional[dict] = None,
                timeout: Optional[tuple] = None, stream: bool = False, check_robots: bool = False,
                retries: Optional[int] = None, max_redirects: Optional[int] = None,
                data: Optional[dict] = None) -> requests.Response:
        """Perform a request, following redirects manually so each hop is validated."""
        timeout = timeout or (self.settings.connect_timeout, self.settings.read_timeout)
        retries = self.settings.max_retries if retries is None else retries
        max_redirects = self.settings.max_redirects if max_redirects is None else max_redirects
        current, hop_params, hop_method, hop_data = url, params, method, data
        for _ in range(max_redirects + 1):
            self._validate(current)
            if check_robots and self.settings.respect_robots and not self.robots_allowed(current):
                raise RobotsDisallowed(f"robots.txt disallows fetching {current}")
            resp = self._send(hop_method, current, params=hop_params, headers=headers, timeout=timeout,
                              stream=stream, retries=retries, data=hop_data)
            loc = resp.headers.get("Location")
            if resp.status_code in REDIRECT_CODES and loc:
                resp.close()
                current = urljoin(current, loc)
                hop_params = None
                if resp.status_code == 303 or (resp.status_code in (301, 302) and hop_method == "POST"):
                    hop_method, hop_data = "GET", None
                continue
            resp.url = current          # final URL after our own redirect handling
            return resp
        raise RequestFailed(f"too many redirects starting at {url}")

    def read_limited(self, resp: requests.Response, max_bytes: int) -> bytes:
        """Read a (possibly compressed) body, aborting once ``max_bytes`` decoded bytes are exceeded."""
        declared = resp.headers.get("Content-Length")
        if declared and declared.isdigit() and int(declared) > max_bytes and not resp.headers.get("Content-Encoding"):
            resp.close()
            raise ResponseTooLarge(f"declared size {declared} exceeds {max_bytes} bytes")
        chunks, total = [], 0
        try:
            for chunk in resp.iter_content(chunk_size=16384):
                total += len(chunk)
                if total > max_bytes:
                    raise ResponseTooLarge(f"body exceeds {max_bytes} bytes")
                chunks.append(chunk)
        except requests.RequestException as exc:
            raise RequestFailed(f"error while reading body: {exc}") from exc
        finally:
            resp.close()
        return b"".join(chunks)

    def get_json(self, url: str, *, params: Optional[dict] = None, headers: Optional[dict] = None,
                 timeout: Optional[tuple] = None, retries: Optional[int] = None):
        """GET and decode JSON. Returns ``None`` for 404; raises RateLimited / RequestFailed otherwise."""
        h = {"Accept": "application/json"}
        h.update(headers or {})
        resp = self.request("GET", url, params=params, headers=h, timeout=timeout, stream=True, retries=retries)
        if resp.status_code == 404:
            resp.close()
            return None
        if resp.status_code == 429:
            resp.close()
            raise RateLimited(f"{urlsplit(url).hostname} returned HTTP 429 (quota or rate limit)", 429)
        if resp.status_code >= 400:
            status = resp.status_code
            resp.close()
            raise RequestFailed(f"HTTP {status} from {urlsplit(url).hostname}", status)
        body = self.read_limited(resp, self.settings.max_json_mb * 1024 * 1024)
        try:
            return json.loads(body.decode("utf-8", errors="replace"))
        except ValueError as exc:
            raise RequestFailed(f"invalid JSON from {urlsplit(url).hostname}") from exc

    def get_text(self, url: str, *, params: Optional[dict] = None, headers: Optional[dict] = None,
                 max_bytes: Optional[int] = None) -> tuple[requests.Response, str]:
        resp = self.request("GET", url, params=params, headers=headers, stream=True)
        raw = self.read_limited(resp, max_bytes or self.settings.max_html_kb * 1024)
        return resp, decode_body(raw, resp.headers.get("Content-Type", ""))

    # ---- robots.txt ---------------------------------------------------------------------
    def robots_allowed(self, url: str) -> bool:
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        now = time.monotonic()
        with self._robots_lock:
            cached = self._robots.get(origin)
        if cached and now - cached[0] < 3600:
            rp, mode = cached[1], cached[2]
        else:
            rp, mode = self._fetch_robots(origin)
            with self._robots_lock:
                self._robots[origin] = (now, rp, mode)
        if mode == "allow_all":
            return True
        if mode == "disallow_all" or rp is None:
            return False
        return rp.can_fetch(APP_NAME, url)

    def _fetch_robots(self, origin: str):
        try:
            resp = self.request("GET", origin + "/robots.txt", stream=True, retries=1, max_redirects=3,
                                timeout=(self.settings.connect_timeout, 10.0))
        except (HttpError, URLValidationError):
            return None, "disallow_all"            # robots.txt unreachable (5xx / network): be conservative
        if 400 <= resp.status_code < 500:
            resp.close()
            return None, "allow_all"               # RFC 9309: "unavailable" => no restrictions
        if resp.status_code >= 500:
            resp.close()
            return None, "disallow_all"
        try:
            raw = self.read_limited(resp, 512 * 1024)
        except HttpError:
            return None, "disallow_all"
        rp = robotparser.RobotFileParser()
        rp.parse(decode_body(raw, resp.headers.get("Content-Type", "")).splitlines())
        return rp, "rules"


def decode_body(raw: bytes, content_type: str = "") -> str:
    charset = None
    low = content_type.lower()
    if "charset=" in low:
        charset = low.split("charset=", 1)[1].split(";")[0].strip(" \"'")
    if not charset:
        head = raw[:4096].lower()
        idx = head.find(b"charset=")
        if idx != -1:
            tail = head[idx + 8: idx + 40]
            for sep in (b'"', b"'", b";", b">", b" "):
                tail = tail.split(sep)[0]
            charset = tail.decode("ascii", "ignore")
    for enc in (charset, "utf-8"):
        if enc:
            try:
                return raw.decode(enc, errors="replace")
            except LookupError:
                continue
    return raw.decode("utf-8", errors="replace")
