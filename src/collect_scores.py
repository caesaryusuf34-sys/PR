"""Collect historical + current-season game results from ESPN's public scoreboard API.

Only completed games (status 'post') are kept as training/rating data; today's slate is
collected separately (collect_slate.py) and only games in 'pre' state are predicted.
"""
import json, os, sys, time, concurrent.futures as cf
import requests
import pandas as pd

BASE = "https://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard"
S = requests.Session()
S.headers["User-Agent"] = "cfb-research-model/1.0 (personal, non-commercial)"
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "scoreboards")
os.makedirs(OUT, exist_ok=True)

def fetch(season, seasontype, week, group):
    fn = os.path.join(OUT, f"{season}_{seasontype}_{week}_{group}.json")
    # Current season files are always re-fetched (status changes); past seasons cached.
    if os.path.exists(fn) and season < 2026:
        return json.load(open(fn))
    for attempt in range(4):
        try:
            r = S.get(BASE, params={"dates": season, "seasontype": seasontype, "week": week,
                                    "groups": group, "limit": 400}, timeout=30)
            r.raise_for_status()
            d = r.json()
            json.dump(d, open(fn, "w"))
            return d
        except Exception as e:
            time.sleep(2 ** attempt)
    return {"events": []}

def rows(d, group):
    for e in d.get("events", []):
        if not e.get("competitions"):
            continue
        c = e["competitions"][0]
        st = c["status"]["type"]
        comps = {x["homeAway"]: x for x in c["competitors"]}
        if "home" not in comps or "away" not in comps:
            continue
        h, a = comps["home"], comps["away"]
        yield dict(
            event_id=e["id"], date=e["date"], season=e["season"]["year"],
            seasontype=e["season"]["type"], week=e.get("week", {}).get("number"),
            state=st["state"], completed=st.get("completed", False),
            neutral=c.get("neutralSite", False), conf_game=c.get("conferenceCompetition", False),
            venue_id=c.get("venue", {}).get("id"), venue=c.get("venue", {}).get("fullName"),
            venue_city=c.get("venue", {}).get("address", {}).get("city"),
            venue_state=c.get("venue", {}).get("address", {}).get("state"),
            indoor=c.get("venue", {}).get("indoor"),
            home_id=h["team"]["id"], home=h["team"].get("location") or h["team"].get("displayName"),
            home_abbr=h["team"].get("abbreviation"), home_conf_id=h["team"].get("conferenceId"),
            away_id=a["team"]["id"], away=a["team"].get("location") or a["team"].get("displayName"),
            away_abbr=a["team"].get("abbreviation"), away_conf_id=a["team"].get("conferenceId"),
            home_score=pd.to_numeric(h.get("score"), errors="coerce"),
            away_score=pd.to_numeric(a.get("score"), errors="coerce"),
            group_query=group,
        )

if __name__ == "__main__":
    seasons = [int(x) for x in sys.argv[1:]] or list(range(2015, 2027))
    jobs = []
    for s in seasons:
        for w in range(1, 17):
            for g in (80, 81):
                jobs.append((s, 2, w, g))
        for g in (80, 81):
            jobs.append((s, 3, 1, g))
    allrows = []
    with cf.ThreadPoolExecutor(8) as ex:
        for (job, d) in zip(jobs, ex.map(lambda j: fetch(*j), jobs)):
            allrows.extend(rows(d, job[3]))
    df = pd.DataFrame(allrows).drop_duplicates("event_id")
    df.to_csv(os.path.join(os.path.dirname(__file__), "..", "data", "all_games_raw.csv"), index=False)
    print(df.groupby(["season", "state"]).size().unstack(fill_value=0))
