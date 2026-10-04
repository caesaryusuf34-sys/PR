"""Thin client for ESPN's public JSON endpoints (no authentication, no scraping of protected pages)."""
from __future__ import annotations
import time
import requests

BASE = "https://site.api.espn.com/apis/site/v2/sports/"


class ESPN:
    def __init__(self, league_path: str = "football/college-football", user_agent: str = "sportsai/1.0"):
        self.base = BASE + league_path
        self.s = requests.Session()
        self.s.headers["User-Agent"] = user_agent

    def get(self, endpoint: str, retries: int = 4, **params):
        last = None
        for a in range(retries):
            try:
                r = self.s.get(f"{self.base}/{endpoint}", params=params, timeout=40)
                if r.status_code == 404:
                    return {}
                r.raise_for_status()
                return r.json()
            except Exception as e:  # network hiccup -> exponential backoff
                last = e
                time.sleep(2 ** a)
        raise RuntimeError(f"ESPN request failed: {endpoint} {params}: {last}")

    def scoreboard(self, **params):
        params.setdefault("limit", 400)
        return self.get("scoreboard", **params)

    def summary(self, event_id: str):
        return self.get("summary", event=event_id)

    def team_schedule(self, team_id: str, season: int):
        return self.get(f"teams/{team_id}/schedule", season=season)
