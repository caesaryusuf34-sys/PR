"""ESPN JSON -> rows. Scoreboard events split into PRE-GAME rows (games, teams) and POST-GAME rows
(results). Game summaries are compacted to the box score / QB / play-by-play fields we use."""
from __future__ import annotations
import pandas as pd

SPORT = "ncaaf"


def num(x):
    if isinstance(x, dict):
        x = x.get("value", x.get("displayValue"))
    try:
        return float(x)
    except Exception:
        return None


def scoreboard_rows(d: dict, feed_group: int, collected_utc: str):
    games, results, teams = [], [], []
    for e in d.get("events", []):
        if not e.get("competitions"):
            continue
        c = e["competitions"][0]
        comps = {x.get("homeAway"): x for x in c.get("competitors", [])}
        if "home" not in comps or "away" not in comps:
            continue
        h, a = comps["home"], comps["away"]
        st = c.get("status", e.get("status", {})).get("type", {})
        v = c.get("venue", {}) or {}
        season = (e.get("season") or {}).get("year")
        games.append(dict(
            sport=SPORT, game_id=str(e["id"]), season=season, seasontype=(e.get("season") or {}).get("type"),
            week=(e.get("week") or {}).get("number"),
            kickoff_utc=pd.Timestamp(e["date"]).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
            home_id=str(h["team"]["id"]), away_id=str(a["team"]["id"]),
            home_name=h["team"].get("location") or h["team"].get("displayName"),
            away_name=a["team"].get("location") or a["team"].get("displayName"),
            neutral=int(bool(c.get("neutralSite"))), conf_game=int(bool(c.get("conferenceCompetition"))),
            venue=v.get("fullName"), venue_city=(v.get("address") or {}).get("city"),
            venue_state=(v.get("address") or {}).get("state"), indoor=int(bool(v.get("indoor"))) if "indoor" in v else None,
            feed_group=feed_group, status=st.get("name"), first_seen_utc=collected_utc, updated_utc=collected_utc))
        for x in (h, a):
            t = x["team"]
            teams.append(dict(sport=SPORT, team_id=str(t["id"]), location=t.get("location"), name=t.get("name"),
                              abbr=t.get("abbreviation"), display_name=t.get("displayName"),
                              short_name=t.get("shortDisplayName"), updated_utc=collected_utc))
        if st.get("state") == "post" and st.get("completed") and num(h.get("score")) is not None and num(a.get("score")) is not None:
            results.append(dict(sport=SPORT, game_id=str(e["id"]), home_score=num(h["score"]), away_score=num(a["score"]),
                                status=st.get("name"), collected_utc=collected_utc, source="espn_scoreboard"))
    return games, results, teams


def compact_summary(eid: str, s: dict) -> dict:
    """Keep the fields needed for team-game statistics (box score, defense totals, starting QB, plays)."""
    out = {"event_id": str(eid), "teams": {}, "plays": [], "drives": []}
    bx = s.get("boxscore", {}) or {}
    for tm in bx.get("teams", []):
        tid = str(tm["team"]["id"])
        out["teams"][tid] = {"box": {st["name"]: st.get("displayValue") for st in tm.get("statistics", [])},
                             "def": {}, "qb": None}
    for p in bx.get("players", []):
        tid = str(p["team"]["id"])
        if tid not in out["teams"]:
            continue
        for cat in p.get("statistics", []):
            labels = cat.get("labels", [])
            if cat.get("name") == "defensive":
                tot = {}
                for a in cat.get("athletes", []):
                    for lab, v in zip(labels, a.get("stats", [])):
                        if lab in ("SACKS", "TFL", "QB HUR", "PD"):
                            tot[lab] = tot.get(lab, 0) + (num(v) or 0)
                out["teams"][tid]["def"] = tot
            if cat.get("name") == "passing" and cat.get("athletes"):
                best = None
                for a in cat["athletes"]:
                    st = dict(zip(labels, a.get("stats", [])))
                    ca = st.get("C/ATT", "0/0").split("/")
                    att = num(ca[1]) if len(ca) == 2 else 0
                    if best is None or (att or 0) > best["att"]:
                        best = {"id": a["athlete"].get("id"), "name": a["athlete"].get("displayName"),
                                "cmp": num(ca[0]), "att": att or 0, "yds": num(st.get("YDS")),
                                "td": num(st.get("TD")), "int": num(st.get("INT"))}
                out["teams"][tid]["qb"] = best
    for dr in (s.get("drives") or {}).get("previous", []):
        off = (dr.get("team") or {}).get("id")
        for p in dr.get("plays", []):
            ps, pe = p.get("start", {}) or {}, p.get("end", {}) or {}
            out["plays"].append({
                "off": (ps.get("team") or {}).get("id") or off, "drive_off": off, "type": (p.get("type") or {}).get("text"),
                "period": (p.get("period") or {}).get("number"),
                "down": ps.get("down"), "dist": ps.get("distance"), "ytez": ps.get("yardsToEndzone"),
                "end_down": pe.get("down"), "end_dist": pe.get("distance"), "end_ytez": pe.get("yardsToEndzone"),
                "end_off": (pe.get("team") or {}).get("id"),
                "yds": p.get("statYardage"), "pen": p.get("isPenalty"), "to": p.get("isTurnover"),
                "score": p.get("scoringPlay"), "hs": p.get("homeScore"), "as": p.get("awayScore")})
    return out
