"""League data sources -> canonical rows.

Every source returns plain dicts in one canonical shape so the adapter, database and feature builder
are shared by all leagues:

  games     PRE-GAME   game_id, season, seasontype (2 regular / 3 postseason), kickoff_utc, home/away id+name,
                       conf_game, venue, status, source_ref (the source's own key, e.g. a box-score path)
  results   POST-GAME  game_id, home_score, away_score, status            (final games only)
  teams                team_id, location, name, abbr, display_name, short_name
  starters  PRE-GAME   game_id, side, team_id, pitcher_id, pitcher_name, source
                       'probable' = announcement for a game that has not started (pitcher_id None = TBD)
                       'backfill' = announced/actual starter of a past game (training + backtests only)
  stats     POST-GAME  game_id, team_id, stats{...}   (starter line, team pitching totals)

Sources (public, unauthenticated):
  MLB   statsapi.mlb.com           schedule (+ probable pitchers), box scores
  KBO   koreabaseball.com          official game list (starters with ids, scores), schedule
  NPB   npb.jp                     monthly schedule (scores, announced starters), box scores
  CPBL  Yahoo Sports API           scoreboard + box scores (cpbl.com.tw refuses non-Taiwan clients)
"""
from __future__ import annotations
import concurrent.futures as cf
import json, re
from datetime import date, timedelta
from email.utils import parsedate_to_datetime
import pandas as pd

ISO = "%Y-%m-%dT%H:%M:%SZ"
LINE_KEYS = ("outs", "bf", "h", "hr", "bb", "hbp", "k", "r", "er")


def _utc(local: str, tz: str) -> str:
    return pd.Timestamp(local).tz_localize(tz).tz_convert("UTC").strftime(ISO)


def _int(x):
    try:
        return int(str(x).strip())
    except (TypeError, ValueError):
        return None


def daterange(a: date, b: date):
    d = a
    while d <= b:
        yield d
        d += timedelta(days=1)


def side_stats(sp_id, sp_name, sp: dict | None, team: dict | None, extra: dict | None = None) -> dict:
    """Canonical post-game team stats: starter id/name/line (sp_*) and team pitching totals (tm_*)."""
    out = {"sp_id": None if sp_id is None else str(sp_id), "sp_name": sp_name}
    for k in LINE_KEYS:
        out[f"sp_{k}"] = None if not sp or sp.get(k) is None else float(sp[k])
        out[f"tm_{k}"] = None if not team or team.get(k) is None else float(team[k])
    out.update(extra or {})
    return out


def empty_rows():
    return {"games": [], "results": [], "teams": [], "starters": [], "stats": []}


def merge_rows(parts):
    out = empty_rows()
    for p in parts:
        for k in out:
            out[k] += p.get(k, [])
    return out


# =========================================================================================== MLB
class MLBSource:
    league, tz, needs_box = "mlb", "America/New_York", True
    BASE = "https://statsapi.mlb.com/api/v1"
    GAME_TYPES = "R,F,D,L,W"          # regular season + wild card / division / LCS / World Series
    BOX_FIELDS = ("teams,home,away,team,id,pitchers,teamStats,pitching,outs,battersFaced,hits,homeRuns,baseOnBalls,"
                  "hitByPitch,strikeOuts,earnedRuns,runs,players,person,fullName,stats")

    def __init__(self, http):
        self.http = http

    def season_window(self, season):
        return date(season, 2, 15), date(season, 11, 30)

    def fetch_season(self, season, stamp, now) -> dict:
        return self.fetch_range(*self.season_window(season), stamp, now)

    def fetch_range(self, start: date, end: date, stamp: str, now: pd.Timestamp) -> dict:
        d = self.http.get_json(f"{self.BASE}/schedule", sportId=1, startDate=str(start), endDate=str(end),
                               gameType=self.GAME_TYPES, hydrate="probablePitcher,venue,team") or {}
        entries = {}
        for day in d.get("dates", []):
            for g in day.get("games", []):
                pk = str(g["gamePk"])
                bad = g["status"].get("detailedState") in ("Postponed", "Cancelled")
                # a postponed game re-appears (same gamePk) on its new date: keep the playable entry
                if pk not in entries or not bad:
                    entries[pk] = g
        out = empty_rows()
        for pk, g in entries.items():
            st = g["status"]; det = st.get("detailedState", "")
            h, a = g["teams"]["home"], g["teams"]["away"]
            ht, at = h["team"], a["team"]
            kick = pd.Timestamp(g["gameDate"]).tz_convert("UTC")
            out["games"].append(dict(
                game_id=pk, season=int(g["season"]), seasontype=2 if g["gameType"] == "R" else 3, week=None,
                kickoff_utc=kick.strftime(ISO), home_id=str(ht["id"]), away_id=str(at["id"]),
                home_name=ht.get("name"), away_name=at.get("name"), neutral=0,
                conf_game=int((ht.get("league") or {}).get("id") == (at.get("league") or {}).get("id")),
                venue=(g.get("venue") or {}).get("name"), status=det, source_ref=pk))
            for t in (ht, at):
                out["teams"].append(dict(team_id=str(t["id"]), location=t.get("locationName"), name=t.get("teamName"),
                                         abbr=t.get("abbreviation"), display_name=t.get("name"),
                                         short_name=t.get("shortName") or t.get("teamName")))
            final = st.get("abstractGameState") == "Final" and det not in ("Postponed", "Cancelled")
            if final and h.get("score") is not None and a.get("score") is not None:
                out["results"].append(dict(game_id=pk, home_score=float(h["score"]), away_score=float(a["score"]), status=det))
            upcoming = kick > now and st.get("abstractGameState") == "Preview"
            for side, tm in (("home", h), ("away", a)):
                pp = tm.get("probablePitcher") or {}
                if upcoming or pp.get("id"):
                    out["starters"].append(dict(game_id=pk, side=side, team_id=str(tm["team"]["id"]),
                                                pitcher_id=None if not pp.get("id") else str(pp["id"]),
                                                pitcher_name=pp.get("fullName"),
                                                source="probable" if upcoming else "backfill"))
        return out

    def fetch_box(self, game: dict) -> dict | None:
        d = self.http.get_json(f"{self.BASE}/game/{game['source_ref'] or game['game_id']}/boxscore", fields=self.BOX_FIELDS)
        if not d or "teams" not in d:
            return None
        def line(s):
            return {"outs": s.get("outs"), "bf": s.get("battersFaced"), "h": s.get("hits"), "hr": s.get("homeRuns"),
                    "bb": s.get("baseOnBalls"), "hbp": s.get("hitByPitch"), "k": s.get("strikeOuts"),
                    "r": s.get("runs"), "er": s.get("earnedRuns")}
        out = {}
        for side in ("home", "away"):
            t = d["teams"][side]
            ps = t.get("pitchers") or []
            sp = ps[0] if ps else None
            pl = (t.get("players") or {}).get(f"ID{sp}", {}) if sp else {}
            out[side] = side_stats(sp, (pl.get("person") or {}).get("fullName"),
                                   line((pl.get("stats") or {}).get("pitching") or {}) if sp else None,
                                   line((t.get("teamStats") or {}).get("pitching") or {}))
        return out


# =========================================================================================== KBO
KBO_TEAMS = {  # code: (English name, short, Korean short name)
    "OB": ("Doosan Bears", "Doosan", "두산"), "LG": ("LG Twins", "LG", "LG"), "WO": ("Kiwoom Heroes", "Kiwoom", "키움"),
    "SS": ("Samsung Lions", "Samsung", "삼성"), "HT": ("KIA Tigers", "KIA", "KIA"), "KT": ("KT Wiz", "KT", "KT"),
    "SK": ("SSG Landers", "SSG", "SSG"), "NC": ("NC Dinos", "NC", "NC"), "LT": ("Lotte Giants", "Lotte", "롯데"),
    "HH": ("Hanwha Eagles", "Hanwha", "한화")}
KBO_PARKS = {"잠실": "Jamsil Baseball Stadium", "고척": "Gocheok Sky Dome", "문학": "Incheon SSG Landers Field",
             "대구": "Daegu Samsung Lions Park", "광주": "Gwangju-Kia Champions Field", "수원": "Suwon KT Wiz Park",
             "창원": "Changwon NC Park", "사직": "Sajik Baseball Stadium", "대전": "Hanwha Life Eagles Park",
             "대전(신)": "Daejeon Hanwha Life Ballpark", "울산": "Ulsan Munsu Baseball Stadium", "포항": "Pohang Baseball Stadium",
             "청주": "Cheongju Baseball Stadium"}


class KBOSource:
    league, tz, needs_box = "kbo", "Asia/Seoul", False
    HOME = "https://www.koreabaseball.com/Schedule/Schedule.aspx"
    LIST = "https://www.koreabaseball.com/ws/Main.asmx/GetKboGameList"
    SCHED = "https://www.koreabaseball.com/ws/Schedule.asmx/GetScheduleList"
    SERIES = "0,3,4,5,7"     # regular season, semi-playoff, wild card, playoff, Korean Series

    def __init__(self, http):
        self.http = http

    def season_window(self, season):
        return date(season, 3, 1), date(season, 11, 30)

    def _post(self, url, data):
        s = self.http.session
        if not s.cookies.get("ASP.NET_SessionId"):      # the web service needs a session cookie
            self.http.request("GET", self.HOME)
        r = self.http.request("POST", url, data=data, headers={"X-Requested-With": "XMLHttpRequest", "Referer": self.HOME})
        if r is None:
            return None
        txt = r.content.decode("utf-8-sig")
        if not txt.lstrip().startswith("{"):            # HTML error page: drop the session and retry once
            s.cookies.clear()
            self.http.request("GET", self.HOME)
            r = self.http.request("POST", url, data=data, headers={"X-Requested-With": "XMLHttpRequest", "Referer": self.HOME})
            txt = r.content.decode("utf-8-sig") if r is not None else ""
            if not txt.lstrip().startswith("{"):
                raise RuntimeError(f"KBO web service returned a non-JSON page for {data}")
        return json.loads(txt)

    def game_dates(self, season: int) -> list[date]:
        days = set()
        for m in range(3, 12):
            d = self._post(self.SCHED, {"leId": "1", "srIdList": self.SERIES + ",9", "seasonId": str(season),
                                        "gameMonth": f"{m:02d}", "teamId": ""}) or {}
            for row in d.get("rows", []):
                for cell in row.get("row", []):
                    if cell.get("Class") == "day":
                        mm = re.match(r"(\d\d)\.(\d\d)", cell.get("Text", ""))
                        if mm:
                            days.add(date(season, int(mm.group(1)), int(mm.group(2))))
        return sorted(days)

    def day(self, d: date, stamp: str, now: pd.Timestamp) -> dict:
        js = self._post(self.LIST, {"leId": "1", "srId": self.SERIES, "date": d.strftime("%Y%m%d")}) or {}
        out = empty_rows()
        for g in js.get("game", []):
            hid, aid = g.get("HOME_ID"), g.get("AWAY_ID")
            if hid not in KBO_TEAMS or aid not in KBO_TEAMS:
                continue
            gid = g["G_ID"]
            kick = _utc(f"{g['G_DT']} {g.get('G_TM') or '18:30'}", self.tz)
            canceled = str(g.get("CANCEL_SC_ID", "0")) != "0"
            state = str(g.get("GAME_STATE_SC"))
            status = g.get("CANCEL_SC_NM") if canceled else {"1": "Scheduled", "2": "In Progress", "3": "Final"}.get(state, state)
            out["games"].append(dict(
                game_id=gid, season=int(g["SEASON_ID"]), seasontype=2 if int(g.get("SR_ID", 0)) == 0 else 3, week=None,
                kickoff_utc=kick, home_id=hid, away_id=aid, home_name=KBO_TEAMS[hid][0], away_name=KBO_TEAMS[aid][0],
                neutral=0, conf_game=1, venue=KBO_PARKS.get(g.get("S_NM"), g.get("S_NM")),
                status="Canceled" if canceled else status, source_ref=gid))
            final = state == "3" and not canceled
            if final:
                out["results"].append(dict(game_id=gid, home_score=float(g["B_SCORE_CN"]), away_score=float(g["T_SCORE_CN"]), status="Final"))
            upcoming = pd.Timestamp(kick) > now and state == "1" and not canceled
            for side, tid, pre in (("home", hid, "B"), ("away", aid, "T")):
                pid, nm = g.get(f"{pre}_PIT_P_ID"), (g.get(f"{pre}_PIT_P_NM") or "").strip() or None
                if upcoming or pid:
                    out["starters"].append(dict(game_id=gid, side=side, team_id=tid, pitcher_id=None if not pid else str(pid),
                                                pitcher_name=nm, source="probable" if upcoming else "backfill"))
                if final:                       # the starting pitcher of a completed game (no box-score lines)
                    out["stats"].append(dict(game_id=gid, team_id=tid, stats=side_stats(pid, nm, None, None)))
        for code, (full, short, ko) in KBO_TEAMS.items():
            out["teams"].append(dict(team_id=code, location=short, name=full.split(" ", 1)[-1], abbr=code,
                                     display_name=full, short_name=short))
        return out

    def fetch_range(self, start: date, end: date, stamp: str, now: pd.Timestamp, days=None, threads: int = 4) -> dict:
        days = days if days is not None else list(daterange(start, end))
        with cf.ThreadPoolExecutor(threads) as ex:
            return merge_rows(ex.map(lambda d: self.day(d, stamp, now), days))

    def fetch_season(self, season, stamp, now) -> dict:
        return self.fetch_range(None, None, stamp, now, days=self.game_dates(season))

    def fetch_box(self, game):
        return None


# =========================================================================================== NPB
NPB_TEAMS = {  # code: (English name, short, Japanese short name, league)
    "g": ("Yomiuri Giants", "Giants", "巨人", "CL"), "t": ("Hanshin Tigers", "Tigers", "阪神", "CL"),
    "db": ("Yokohama DeNA BayStars", "BayStars", "DeNA", "CL"), "c": ("Hiroshima Toyo Carp", "Carp", "広島", "CL"),
    "s": ("Tokyo Yakult Swallows", "Swallows", "ヤクルト", "CL"), "d": ("Chunichi Dragons", "Dragons", "中日", "CL"),
    "h": ("Fukuoka SoftBank Hawks", "Hawks", "ソフトバンク", "PL"), "f": ("Hokkaido Nippon-Ham Fighters", "Fighters", "日本ハム", "PL"),
    "m": ("Chiba Lotte Marines", "Marines", "ロッテ", "PL"), "e": ("Tohoku Rakuten Golden Eagles", "Eagles", "楽天", "PL"),
    "b": ("ORIX Buffaloes", "Buffaloes", "オリックス", "PL"), "l": ("Saitama Seibu Lions", "Lions", "西武", "PL")}
NPB_BY_JA = {v[2]: k for k, v in NPB_TEAMS.items()}


def _txt(html: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html.replace("&nbsp;", " "))).strip()


class NPBSource:
    league, tz, needs_box = "npb", "Asia/Tokyo", True
    SCHED = "https://npb.jp/games/{season}/schedule_{month:02d}_detail.html"
    BOX = "https://npb.jp/scores/{ref}box.html"
    STARTERS = "https://npb.jp/announcement/starter/"

    def __init__(self, http):
        self.http = http

    def season_window(self, season):
        return date(season, 3, 1), date(season, 11, 30)

    def month(self, season: int, month: int, stamp: str, now: pd.Timestamp) -> dict:
        html = self.http.get_text(self.SCHED.format(season=season, month=month))
        out = empty_rows()
        if not html:
            return out
        seen = {}
        for tr in re.findall(r'<tr id="date(\d{4})"[^>]*>(.*?)</tr>', html, re.S):
            mmdd, body = tr
            t1, t2 = re.search(r'class="team1">([^<]*)<', body), re.search(r'class="team2">([^<]*)<', body)
            if not t1 or not t2:
                continue
            h, a = NPB_BY_JA.get(t1.group(1).strip()), NPB_BY_JA.get(t2.group(1).strip())
            if not h or not a:
                continue
            day = date(season, int(mmdd[:2]), int(mmdd[2:]))
            key = f"{day:%Y%m%d}{h}{a}"
            seen[key] = seen.get(key, 0) + 1
            gid = key if seen[key] == 1 else f"{key}-{seen[key]}"
            tm = re.search(r'class="time">\s*(\d{1,2}:\d{2})', body)
            place = re.search(r'class="place">([^<]*)<', body)
            link = re.search(r'href="/scores/(\d{4}/\d{4}/[^"]+/)"', body)
            s1, s2 = re.search(r'class="score1">([^<]*)<', body), re.search(r'class="score2">([^<]*)<', body)
            comment = _txt(re.search(r'class="comment">(.*?)</div>', body, re.S).group(1)) if 'class="comment"' in body else ""
            text = _txt(body)
            kick = _utc(f"{day} {tm.group(1) if tm else '18:00'}", self.tz)
            canceled = any(w in comment + text for w in ("中止", "ノーゲーム"))
            reserve = "予備日" in text
            sc1, sc2 = (_int(s1.group(1)) if s1 else None), (_int(s2.group(1)) if s2 else None)
            final = (sc1 is not None and sc2 is not None and not canceled
                     and now > pd.Timestamp(kick) + pd.Timedelta(hours=3))
            status = "Canceled" if canceled else ("Reserve" if reserve else ("Final" if final else "Scheduled"))
            out["games"].append(dict(
                game_id=gid, season=season, seasontype=2, week=None, kickoff_utc=kick, home_id=h, away_id=a,
                home_name=NPB_TEAMS[h][0], away_name=NPB_TEAMS[a][0], neutral=0,
                conf_game=int(NPB_TEAMS[h][3] == NPB_TEAMS[a][3]),
                venue=re.sub(r"\s+", "", place.group(1)) if place else None, status=status,
                source_ref=link.group(1) if link else None))
            if final:
                out["results"].append(dict(game_id=gid, home_score=float(sc1), away_score=float(sc2), status="Final"))
            pits = [_txt(x) for x in re.findall(r'class="pit">(.*?)</div>', body, re.S)]
            sps = [p.split("：", 1)[1].strip() for p in pits if p.startswith("先発")]
            if pd.Timestamp(kick) > now and not canceled and not final:
                for side, tid, nm in (("home", h, sps[0] if len(sps) > 0 else None), ("away", a, sps[1] if len(sps) > 1 else None)):
                    # announced by surname only; the adapter maps names to player ids from past box scores
                    out["starters"].append(dict(game_id=gid, side=side, team_id=tid, pitcher_id=None, pitcher_name=nm, source="probable"))
        for code, (full, short, ja, lg) in NPB_TEAMS.items():
            out["teams"].append(dict(team_id=code, location=full.rsplit(" ", 1)[0] if code != "b" else "ORIX",
                                     name=short, abbr=code.upper(), display_name=full, short_name=short))
        return out

    def announced(self, now: pd.Timestamp) -> list[dict]:
        """Official probable starters for the next game day (予告先発, with player ids): left = home team."""
        html = self.http.get_text(self.STARTERS)
        if not html:
            return []
        hd = re.search(r"<h4>(\d{1,2})月(\d{1,2})日の予告先発投手</h4>", html)
        if not hd:
            return []
        today = now.tz_convert(self.tz).date()
        mo, dy = int(hd.group(1)), int(hd.group(2))
        day = date(today.year + (1 if mo < today.month - 6 else 0), mo, dy)
        out = []
        for unit in re.findall(r'<div class="unit [^"]*">(.*?)<div class="info">', html, re.S):
            side = {}
            for pos in ("left", "right"):
                m = re.search(rf'<div class="team_{pos}">\s*<img src="[^"]*logo_([a-z]+)_m\.gif".*?'
                              rf'href="/bis/players/(\d+)\.html">\s*<span>([^<]+)</span>', unit, re.S)
                if m:
                    side[pos] = (m.group(1), m.group(2), re.sub(r"\s+", " ", m.group(3).replace("\u3000", " ")).strip())
            if "left" in side and "right" in side and side["left"][0] in NPB_TEAMS and side["right"][0] in NPB_TEAMS:
                h, a = side["left"][0], side["right"][0]
                gid = f"{day:%Y%m%d}{h}{a}"
                out += [dict(game_id=gid, side="home", team_id=h, pitcher_id=side["left"][1], pitcher_name=side["left"][2], source="probable"),
                        dict(game_id=gid, side="away", team_id=a, pitcher_id=side["right"][1], pitcher_name=side["right"][2], source="probable")]
        return out

    def fetch_range(self, start: date, end: date, stamp: str, now: pd.Timestamp) -> dict:
        months = sorted({(d.year, d.month) for d in daterange(start, end) if 3 <= d.month <= 11})
        rows = merge_rows(self.month(y, m, stamp, now) for y, m in months)
        keep = {g["game_id"] for g in rows["games"]
                if start <= pd.Timestamp(g["kickoff_utc"]).tz_convert(self.tz).date() <= end}
        rows = {k: [r for r in v if k == "teams" or r["game_id"] in keep] for k, v in rows.items()}
        if end >= now.tz_convert(self.tz).date():
            upcoming = {g["game_id"] for g in rows["games"] if pd.Timestamp(g["kickoff_utc"]) > now and g["status"] == "Scheduled"}
            ann = {(r["game_id"], r["side"]): r for r in self.announced(now) if r["game_id"] in upcoming}
            rows["starters"] = [ann.pop((r["game_id"], r["side"]), r) for r in rows["starters"]] + list(ann.values())
        return rows

    def fetch_season(self, season, stamp, now) -> dict:
        with cf.ThreadPoolExecutor(4) as ex:
            return merge_rows(ex.map(lambda m: self.month(season, m, stamp, now), range(3, 12)))

    @staticmethod
    def _pitchers(html: str, table_id: str):
        m = re.search(rf'id="{table_id}"(.*?)</table>\s*</div>', html, re.S)
        if not m:
            return []
        out = []
        for seg in m.group(1).split('<td class="player">')[1:]:      # one segment per pitcher row
            pl = re.match(r'\s*<a href="/bis/players/(\d+)\.html">([^<]+)</a>', seg)
            ip = re.search(r'<table class="table_inning">.*?<th>([^<]*)</th>\s*<td>([^<]*)</td>', seg, re.S)
            if not pl or not ip:
                continue
            rest = re.sub(r'<table class="table_inning">.*?</table>', "", seg, flags=re.S).split("</tr>")[0]
            cells = [_txt(c) for c in re.findall(r"<td[^>]*>(.*?)</td>", rest, re.S)]
            # cells: pitches, BF, (innings cell), H, HR, BB, HBP, K, WP, BK, R, ER
            if len(cells) < 12:
                continue
            n = [_int(c) for c in cells]
            whole, frac = _txt(ip.group(1)), _txt(ip.group(2))
            outs = (_int(whole) or 0) * 3 + {".1": 1, ".2": 2}.get(frac, 0)
            out.append({"id": pl.group(1), "name": pl.group(2).strip(), "outs": outs, "bf": n[1], "h": n[3],
                        "hr": n[4], "bb": n[5], "hbp": n[6], "k": n[7], "r": n[10], "er": n[11]})
        return out

    def fetch_box(self, game: dict) -> dict | None:
        if not game.get("source_ref"):
            return None
        html = self.http.get_text(self.BOX.format(ref=game["source_ref"]))
        if not html:
            return None
        out = {}
        for side, tid in (("away", "tablefix_t_p"), ("home", "tablefix_b_p")):
            ps = self._pitchers(html, tid)
            if not ps:
                return None
            tot = {k: sum((p[k] or 0) for p in ps) for k in LINE_KEYS}
            out[side] = side_stats(ps[0]["id"], ps[0]["name"], ps[0], tot, {"staff": [[p["id"], p["name"]] for p in ps]})
        return out


# =========================================================================================== CPBL
CPBL_TEAMS = {"cpbl.t.1": ("CTBC Brothers", "Brothers", "中信兄弟"), "cpbl.t.2": ("Uni-President Lions", "Lions", "統一獅"),
              "cpbl.t.5": ("Fubon Guardians", "Guardians", "富邦悍將"), "cpbl.t.6": ("Rakuten Monkeys", "Monkeys", "樂天桃猿"),
              "cpbl.t.7": ("Wei Chuan Dragons", "Dragons", "味全龍"), "cpbl.t.8": ("TSG Hawks", "Hawks", "台鋼雄鷹")}
CPBL_STAT = {"139": "ip", "111": "h", "113": "r", "114": "er", "118": "bb", "121": "k", "115": "hr"}


class CPBLSource:
    league, tz, needs_box = "cpbl", "Asia/Taipei", True
    SB = "https://api-secure.sports.yahoo.com/v1/editorial/s/scoreboard"
    BOX = "https://api-secure.sports.yahoo.com/v1/editorial/s/boxscore/{gid}"
    Q = {"lang": "zh-Hant-TW", "region": "TW", "tz": "Asia/Taipei"}

    def __init__(self, http):
        self.http = http

    def season_window(self, season):
        return date(season, 3, 1), date(season, 11, 30)

    def day(self, d: date, stamp: str, now: pd.Timestamp) -> dict:
        js = self.http.get_json(self.SB, leagues="cpbl", date=str(d), **self.Q) or {}
        sb = (js.get("service") or {}).get("scoreboard") or {}
        out = empty_rows()
        for g in (sb.get("games") or {}).values():
            h, a = g.get("home_team_id"), g.get("away_team_id")
            phase = g.get("season_phase_id") or ""
            if h not in CPBL_TEAMS or a not in CPBL_TEAMS or phase not in ("season.phase.season", "season.phase.postseason"):
                continue
            kick = pd.Timestamp(parsedate_to_datetime(g["start_time"])).tz_convert("UTC")
            if kick.tz_convert(self.tz).date() != d:     # listed on a neighbouring date (time-zone edge)
                continue
            st = (g.get("status_type") or "").replace("status.type.", "")
            gid = g["gameid"]
            out["games"].append(dict(
                game_id=gid, season=int(g.get("season") or d.year), seasontype=2 if phase == "season.phase.season" else 3,
                week=None, kickoff_utc=kick.strftime(ISO), home_id=h, away_id=a, home_name=CPBL_TEAMS[h][0],
                away_name=CPBL_TEAMS[a][0], neutral=0, conf_game=1, venue=None,
                status={"final": "Final", "pregame": "Scheduled", "postponed": "Postponed", "cancelled": "Canceled"}.get(st, st),
                source_ref=gid))
            hs, as_ = _int(g.get("total_home_points")), _int(g.get("total_away_points"))
            if st == "final" and hs is not None and as_ is not None:
                out["results"].append(dict(game_id=gid, home_score=float(hs), away_score=float(as_), status="Final"))
        for tid, (full, short, zh) in CPBL_TEAMS.items():
            out["teams"].append(dict(team_id=tid, location=full.rsplit(" ", 1)[0], name=short, abbr=zh,
                                     display_name=full, short_name=short))
        return out

    def fetch_range(self, start: date, end: date, stamp: str, now: pd.Timestamp, threads: int = 8) -> dict:
        with cf.ThreadPoolExecutor(threads) as ex:
            return merge_rows(ex.map(lambda d: self.day(d, stamp, now), list(daterange(start, end))))

    def fetch_season(self, season, stamp, now) -> dict:
        a, b = self.season_window(season)
        return self.fetch_range(a, min(b, now.tz_convert(self.tz).date() + timedelta(days=10)), stamp, now)

    def fetch_box(self, game: dict) -> dict | None:
        gid = game["game_id"]
        js = self.http.get_json(self.BOX.format(gid=gid), **self.Q) or {}
        b = (js.get("service") or {}).get("boxscore") or {}
        lu = (b.get("gamelineups") or {}).get(gid) or {}
        ps = b.get("player_stats") or {}
        out = {}
        for side in ("home", "away"):
            P = ((lu.get(f"{side}_lineup") or {}).get("P")) or {}
            order = sorted(P.values(), key=lambda x: (_int(x.get("order")) or 99, _int(x.get("suborder")) or 99))
            lines = []
            for x in order:
                st = (ps.get(x["player_id"]) or {}).get("cpbl.stat_variation.2") or {}
                v = {name: st.get(f"cpbl.stat_type.{k}") for k, name in CPBL_STAT.items()}
                ip = str(v.pop("ip") or "0")
                whole, _, frac = ip.partition(".")
                v = {k: _int(x_) for k, x_ in v.items()}
                v["outs"] = (_int(whole) or 0) * 3 + (_int(frac[:1]) or 0)
                v["id"] = x["player_id"]
                lines.append(v)
            if not lines:
                return None
            tot = {k: sum((l.get(k) or 0) for l in lines) for k in ("outs", "h", "hr", "bb", "k", "r", "er")}
            out[side] = side_stats(lines[0]["id"], None, lines[0], tot)
        return out
