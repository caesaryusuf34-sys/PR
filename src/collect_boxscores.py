"""Collect per-game team box scores, defensive totals, QB lines and play-by-play from ESPN
game summaries for completed games. Raw summaries are parsed in memory; compact tables are saved.

Leakage guard: only games whose kickoff is before the slate-day cutoff are collected.
"""
import os, sys, json, time, gzip, concurrent.futures as cf
import requests, pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
SUM = "https://site.api.espn.com/apis/site/v2/sports/football/college-football/summary"
S = requests.Session(); S.headers["User-Agent"] = "cfb-research-model/1.0 (personal, non-commercial)"
CUTOFF = pd.Timestamp("2026-10-03T12:00Z")  # nothing from the slate day itself
CACHE = os.path.join(ROOT, "data", "raw", "parsed"); os.makedirs(CACHE, exist_ok=True)

RUSH = {"Rush", "Rushing Touchdown"}
PASS = {"Pass Reception", "Pass Incompletion", "Passing Touchdown", "Sack",
        "Pass Interception Return", "Interception Return Touchdown", "Pass Interception",
        "Pass", "Passing Touchdown"}
FUMBLE = {"Fumble Recovery (Opponent)", "Fumble Recovery (Own)", "Fumble Return Touchdown"}

def num(x):
    try: return float(x)
    except Exception: return None

def parse(eid):
    fn = os.path.join(CACHE, f"{eid}.json.gz")
    if os.path.exists(fn):
        return json.load(gzip.open(fn, "rt"))
    s = None
    for a in range(4):
        try:
            r = S.get(SUM, params={"event": eid}, timeout=40); r.raise_for_status(); s = r.json(); break
        except Exception:
            time.sleep(2 ** a)
    if s is None:
        return None
    out = {"event_id": str(eid), "teams": {}, "plays": [], "drives": []}
    bx = s.get("boxscore", {})
    for tm in bx.get("teams", []):
        tid = tm["team"]["id"]; d = {}
        for st in tm.get("statistics", []):
            d[st["name"]] = st.get("displayValue")
        out["teams"][tid] = {"box": d, "def": {}, "qb": None}
    for p in bx.get("players", []):
        tid = p["team"]["id"]
        if tid not in out["teams"]: continue
        for cat in p.get("statistics", []):
            labels = cat.get("labels", [])
            if cat["name"] == "defensive":
                tot = {}
                for a in cat.get("athletes", []):
                    for lab, v in zip(labels, a.get("stats", [])):
                        if lab in ("SACKS", "TFL", "QB HUR", "PD"):
                            tot[lab] = tot.get(lab, 0) + (num(v) or 0)
                out["teams"][tid]["def"] = tot
            if cat["name"] == "passing" and cat.get("athletes"):
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
        st = dr.get("start", {}); en = dr.get("end", {})
        out["drives"].append({"off": off, "result": dr.get("result"), "plays": dr.get("offensivePlays"),
                              "yards": dr.get("yards"), "start_yl": st.get("yardLine"),
                              "is_score": dr.get("isScore"), "period": (st.get("period") or {}).get("number")})
        for p in dr.get("plays", []):
            t = (p.get("type") or {}).get("text")
            ps = p.get("start", {}); pe = p.get("end", {})
            out["plays"].append({
                "off": (ps.get("team") or {}).get("id") or off, "drive_off": off, "type": t,
                "period": (p.get("period") or {}).get("number"),
                "down": ps.get("down"), "dist": ps.get("distance"), "ytez": ps.get("yardsToEndzone"),
                "end_down": pe.get("down"), "end_dist": pe.get("distance"), "end_ytez": pe.get("yardsToEndzone"),
                "end_off": (pe.get("team") or {}).get("id"),
                "yds": p.get("statYardage"), "pen": p.get("isPenalty"), "to": p.get("isTurnover"),
                "score": p.get("scoringPlay"), "hs": p.get("homeScore"), "as": p.get("awayScore")})
    json.dump(out, gzip.open(fn, "wt"))
    return out

if __name__ == "__main__":
    seasons = [int(x) for x in sys.argv[1:]]
    g = pd.read_csv(os.path.join(ROOT, "data", "all_games_raw.csv"), dtype={"event_id": str})
    g["date"] = pd.to_datetime(g["date"], utc=True)
    g = g[(g.state == "post") & (g.date < CUTOFF) & g.season.isin(seasons)]
    ids = g.event_id.tolist()
    print("to fetch", len(ids), flush=True)
    t0 = time.time(); done = 0
    with cf.ThreadPoolExecutor(16) as ex:
        for r in ex.map(parse, ids):
            done += 1
            if done % 500 == 0: print(done, round(time.time() - t0), flush=True)
    print("done", done, round(time.time() - t0))
