"""Polite HTTP client shared by the baseball sources: one session per thread, retries with exponential
backoff on network errors, 429 and 5xx (several official league sites reset connections under load).
A 404 returns None. Only public, unauthenticated endpoints are used."""
from __future__ import annotations
import threading, time
import requests


class Http:
    def __init__(self, user_agent: str, headers: dict | None = None, retries: int = 6, timeout: float = 40):
        self.ua, self.headers, self.retries, self.timeout = user_agent, dict(headers or {}), retries, timeout
        self._local = threading.local()

    @property
    def session(self) -> requests.Session:
        s = getattr(self._local, "s", None)
        if s is None:
            s = self._local.s = requests.Session()
            s.headers["User-Agent"] = self.ua
            s.headers.update(self.headers)
        return s

    def request(self, method: str, url: str, **kw):
        last = None
        for a in range(self.retries):
            try:
                r = self.session.request(method, url, timeout=self.timeout, **kw)
                if r.status_code == 404:
                    return None
                if r.status_code == 429 or r.status_code >= 500:
                    raise RuntimeError(f"HTTP {r.status_code}")
                r.raise_for_status()
                return r
            except Exception as e:  # network hiccup / reset -> back off and retry
                last = e
                time.sleep(min(2 ** a, 20))
        raise RuntimeError(f"request failed: {method} {url} {kw.get('params') or kw.get('data')}: {last}")

    def get_json(self, url: str, **params):
        r = self.request("GET", url, params=params)
        return None if r is None else r.json()

    def get_text(self, url: str, encoding: str = "utf-8", **params):
        r = self.request("GET", url, params=params)
        if r is None:
            return None
        r.encoding = encoding
        return r.text
