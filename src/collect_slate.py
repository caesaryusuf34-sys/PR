"""Collect today's slate (FBS + FCS scoreboards), keep games not yet started, and pull
pre-game context (venue, weather forecast, conference) from the ESPN event summary.
Market odds are recorded ONLY as an external reference column and are NOT model inputs."""
import os, json, datetime as dt
from zoneinfo import ZoneInfo
import requests, pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
SB = "https://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard"
SUM = "https://site.api.espn.com/apis/site/v2/sports/football/college-football/summary"
GROUPS = "https://site.api.espn.com/apis/site/v2/sports/football/college-football/groups"
ET = ZoneInfo("America/New_York")
DATE = os.environ.get("SLATE_DATE", "20261003")

def conferences():
    m = {}
    for div in (80, 81):
        try:
            d = requests.get(GROUPS, params={"groups": div}, timeout=30).json()
        except Exception:
            continue
        def walk(node):
            for g in node.get("groups", []) or node.get("children", []):
                for t in g.get("teams", []):
                    m[t["id"]] = g.get("abbreviation") or g.get("name")
                walk(g)
        walk(d)
    return m

def main():
    pulled = dt.datetime.now(dt.timezone.utc)
    events = {}
    for g in (80, 81):
        d = requests.get(SB, params={"dates": DATE, "groups": g, "limit": 400}, timeout=30).json()
        json.dump(d, open(os.path.join(ROOT, "data", "raw", f"scoreboard_{DATE}_g{g}.json"), "w"))
        for e in d.get("events", []):
            events.setdefault(e["id"], (e, g))
    rows = []
    for eid, (e, g) in events.items():
        c = e["competitions"][0]; st = c["status"]["type"]
        comps = {x["homeAway"]: x for x in c["competitors"]}
        h, a = comps["home"], comps["away"]
        kick = dt.datetime.fromisoformat(e["date"].replace("Z", "+00:00"))
        w = e.get("weather") or {}
        odds = (c.get("odds") or [{}])[0]
        def rec(x, typ):
            for r in x.get("records", []):
                if r.get("type") == typ or r.get("name") == typ: return r.get("summary")
            return None
        rows.append(dict(
            event_id=eid, date_utc=e["date"], kickoff_et=kick.astimezone(ET).strftime("%Y-%m-%d %I:%M %p ET"),
            status=st["name"], state=st["state"], scoreboard_group=g,
            away=a["team"]["location"], away_id=a["team"]["id"], away_abbr=a["team"]["abbreviation"],
            home=h["team"]["location"], home_id=h["team"]["id"], home_abbr=h["team"]["abbreviation"],
            away_rank=(a.get("curatedRank") or {}).get("current"), home_rank=(h.get("curatedRank") or {}).get("current"),
            away_record=rec(a, "total"), home_record=rec(h, "total"),
            away_conf_record=rec(a, "vsconf"), home_conf_record=rec(h, "vsconf"),
            venue=c.get("venue", {}).get("fullName"),
            venue_city=c.get("venue", {}).get("address", {}).get("city"),
            venue_state=c.get("venue", {}).get("address", {}).get("state"),
            indoor=c.get("venue", {}).get("indoor"), neutral=c.get("neutralSite"),
            conf_game=c.get("conferenceCompetition"),
            broadcast=", ".join(sum([b.get("names", []) for b in c.get("broadcasts", [])], [])),
            wx_condition=w.get("displayValue"), wx_temp_f=w.get("temperature"),
            market_spread_REFERENCE_ONLY=odds.get("details"), market_total_REFERENCE_ONLY=odds.get("overUnder"),
        ))
    df = pd.DataFrame(rows)
    # extra pre-game weather detail (gust / precip chance) from summary.gameInfo
    gust, precip, conf = [], [], {}
    for eid, state in zip(df.event_id, df.state):
        gu = pr = None
        if state == "pre":
            try:
                s = requests.get(SUM, params={"event": eid}, timeout=30).json()
                wi = (s.get("gameInfo") or {}).get("weather") or {}
                gu, pr = wi.get("gust"), wi.get("precipitation")
                for grp in (s.get("standings") or {}).get("groups", []):
                    name = (grp.get("header") or "").replace("2026 ", "").replace(" Standings", "")
                    name = name.replace(" Conference", "")
                    for ent in grp.get("standings", {}).get("entries", []):
                        conf[str(ent.get("id"))] = name
            except Exception:
                pass
        gust.append(gu); precip.append(pr)
    df["wx_gust_mph"] = gust; df["wx_precip_pct"] = precip
    df["away_conf"] = df.away_id.astype(str).map(conf); df["home_conf"] = df.home_id.astype(str).map(conf)
    df["data_pulled_utc"] = pulled.isoformat(timespec="seconds")
    df = df.sort_values(["date_utc", "event_id"])
    df.to_csv(os.path.join(ROOT, "data", "slate_all_today.csv"), index=False)
    print(df.groupby("state").size()); print(df[df.state == "pre"][["kickoff_et", "away", "home", "venue", "away_conf", "home_conf", "wx_condition", "wx_gust_mph"]].to_string())

if __name__ == "__main__":
    main()
