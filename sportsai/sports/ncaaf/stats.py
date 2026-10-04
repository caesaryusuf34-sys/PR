"""Play-by-play -> expected points model -> per team-game efficiency statistics (vectorized).

The EP model is a next-score model fitted once on historical plays and stored as a versioned
JSON table, so EPA for newly ingested games is computed with a fixed, reproducible model.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

STATS_VERSION = "ncaaf-s1"
RUSH = {"Rush", "Rushing Touchdown", "Fumble Recovery (Own)", "Fumble Recovery (Opponent)",
        "Fumble Return Touchdown", "Fumble"}
PASS = {"Pass Reception", "Pass Incompletion", "Passing Touchdown", "Sack", "Pass Interception Return",
        "Interception", "Interception Return Touchdown", "Pass Completion", "Pass"}
DIST_EDGES = np.array([3, 6, 10, 15])           # dist buckets: <=3, <=6, <=10, <=15, >15
NUMCOLS = ("down", "dist", "ytez", "end_down", "end_dist", "end_ytez", "yds", "hs", "as", "period")


def plays_frame(parsed: list[dict], home_of: dict[str, str]) -> pd.DataFrame:
    rows = []
    for d in parsed:
        eid = d["event_id"]
        if eid not in home_of:
            continue
        for j, p in enumerate(d.get("plays", [])):
            rows.append((eid, j, p.get("off"), p.get("drive_off"), p.get("type"), p.get("period"), p.get("down"),
                         p.get("dist"), p.get("ytez"), p.get("end_down"), p.get("end_dist"), p.get("end_ytez"),
                         p.get("end_off"), p.get("yds"), p.get("to"), p.get("hs"), p.get("as")))
    pl = pd.DataFrame(rows, columns=["event_id", "seq", "off", "drive_off", "type", "period", "down", "dist", "ytez",
                                     "end_down", "end_dist", "end_ytez", "end_off", "yds", "to", "hs", "as"])
    for c in NUMCOLS:
        pl[c] = pd.to_numeric(pl[c], errors="coerce")
    for c in ("off", "drive_off", "end_off"):
        pl[c] = pl[c].astype("string")
    pl["home_id"] = pl.event_id.map(home_of)
    pl = pl.sort_values(["event_id", "seq"]).reset_index(drop=True)
    g = pl.groupby("event_id", sort=False)
    prev_h = g["hs"].shift(1).fillna(0); prev_a = g["as"].shift(1).fillna(0)
    pl["hs"] = pl["hs"].fillna(prev_h); pl["as"] = pl["as"].fillna(prev_a)
    sc = (pl.hs - prev_h) - (pl["as"] - prev_a)
    val = np.sign(sc) * np.select([sc.abs() >= 6, sc.abs() >= 3, sc.abs() >= 2, sc.abs() >= 1], [7, 3, 2, 1], 0)
    pat = (sc.abs() <= 2) & pl.type.fillna("").str.contains("Extra|Two|2pt|Conversion", regex=True)
    pl["score_val_home"] = np.where(pat, 0, val)
    pl["off_is_home"] = (pl.off == pl.home_id).fillna(False).astype(bool)
    pl["pre_margin_off"] = np.where(pl.off_is_home, prev_h - prev_a, prev_a - prev_h)
    pl["half"] = np.where(pl.period <= 2, 1, np.where(pl.period <= 4, 2, 3))
    nxt = pl.score_val_home.replace(0, np.nan).groupby([pl.event_id, pl.half]).bfill().fillna(0)
    pl["next_score_off"] = np.where(pl.off_is_home, nxt, -nxt)
    pl["scrim"] = (pl.type.isin(RUSH | PASS) & pl.down.between(1, 4) & pl.ytez.between(1, 99) & pl.dist.between(1, 99))
    return pl


class EPModel:
    """EP(down, yards-to-endzone bucket, distance bucket), smoothed toward the field-position mean."""

    def __init__(self, table=None, by_yb=None, meta=None):
        self.table, self.by_yb, self.meta = table, by_yb, meta or {}

    @staticmethod
    def _idx(down, ytez, dist):
        d = np.clip(np.nan_to_num(down, nan=1), 1, 4).astype(int) - 1
        yb = (np.clip(np.nan_to_num(ytez, nan=75), 1, 99) // 5).astype(int)
        db = np.searchsorted(DIST_EDGES, np.nan_to_num(dist, nan=10), side="left")
        return d, yb, db

    def fit(self, pl: pd.DataFrame, k: float = 30.0):
        s = pl[pl.scrim]
        d, yb, db = self._idx(s.down.values, s.ytez.values, s.dist.values)
        y = s.next_score_off.values
        S = np.zeros((4, 20, 5)); N = np.zeros((4, 20, 5))
        np.add.at(S, (d, yb, db), y); np.add.at(N, (d, yb, db), 1)
        byS = np.bincount(yb, weights=y, minlength=20); byN = np.bincount(yb, minlength=20)
        by = byS / np.maximum(byN, 1)
        self.table = ((S + k * by[None, :, None]) / (N + k)).tolist()
        self.by_yb = by.tolist()
        self.meta = {"n_plays": int(len(s)), "smoothing_k": k}
        return self

    def ep(self, down, ytez, dist):
        T = np.asarray(self.table)
        d, yb, db = self._idx(np.asarray(down, float), np.asarray(ytez, float), np.asarray(dist, float))
        return T[d, yb, db]

    def save(self, path: Path):
        path.write_text(json.dumps({"version": "ep1", "table": self.table, "by_yb": self.by_yb, "meta": self.meta}))

    @classmethod
    def load(cls, path: Path):
        d = json.loads(path.read_text())
        return cls(d["table"], d["by_yb"], d["meta"])


def add_epa(pl: pd.DataFrame, ep: EPModel) -> pd.DataFrame:
    s = pl[pl.scrim].copy()
    s["ep0"] = ep.ep(s.down, s.ytez, s.dist.fillna(10))
    same = (s.end_off == s.off).fillna(False) & s.end_down.between(1, 4) & s.end_ytez.between(1, 99)
    flip = (s.end_off != s.off).fillna(False) & s.end_ytez.between(1, 99)
    ep1 = np.zeros(len(s))
    ep1 = np.where(same, ep.ep(s.end_down, s.end_ytez, s.end_dist.fillna(10)), ep1)
    ep1 = np.where(flip & ~same, -ep.ep(np.ones(len(s)), s.end_ytez, np.full(len(s), 10.0)), ep1)
    sv = np.where(s.off_is_home, s.score_val_home, -s.score_val_home)
    s["epa"] = np.clip(np.where(sv != 0, sv - s.ep0, ep1 - s.ep0), -10, 10)
    yds = s.yds.fillna(0)
    s["success"] = np.select([s.down == 1, s.down == 2], [yds >= 0.5 * s.dist, yds >= 0.7 * s.dist], yds >= s.dist).astype(float)
    s["is_pass"] = s.type.isin(PASS)
    s["explosive"] = np.where(s.is_pass, yds >= 16, yds >= 12).astype(float)
    s["sack"] = (s.type == "Sack").astype(float)
    s["turnover"] = s.to.fillna(False).astype(float)
    am = s.pre_margin_off.abs()
    garbage = ((s.period == 2) & (am > 38)) | ((s.period == 3) & (am > 28)) | ((s.period == 4) & (am > 22))
    return s[~garbage]


def _frac(x, i):
    try:
        a, b = str(x).replace("/", "-").split("-")
        return float([a, b][i])
    except Exception:
        return np.nan


def team_game_stats(parsed: list[dict], home_of: dict[str, str], ep: EPModel) -> list[dict]:
    """One row per (game, team) with offensive efficiency, drive, special-teams, box-score and QB fields."""
    if not parsed:
        return []
    pl = plays_frame(parsed, home_of)
    rows = {}
    if len(pl):
        s = add_epa(pl, ep)
        key = [s.event_id, s.off]
        agg = pd.DataFrame({
            "plays": s.groupby(key).size(), "epa_pp": s.groupby(key).epa.mean(), "sr": s.groupby(key).success.mean(),
            "ypp": s.groupby(key).yds.mean(), "expl": s.groupby(key).explosive.mean(),
            "to_rate": s.groupby(key).turnover.mean()})
        for nm, m in (("rush", ~s.is_pass), ("pass", s.is_pass)):
            ss = s[m]; k2 = [ss.event_id, ss.off]
            agg[f"{nm}_epa"] = ss.groupby(k2).epa.mean(); agg[f"{nm}_sr"] = ss.groupby(k2).success.mean()
        ps = s[s.is_pass]
        agg["sack_rate"] = ps.groupby([ps.event_id, ps.off]).sack.mean()
        # drives
        newd = (pl.drive_off != pl.drive_off.shift()) | (pl.event_id != pl.event_id.shift())
        pl["drive_key"] = newd.cumsum()
        sc = pl[pl.type.isin(RUSH | PASS) & pl.ytez.between(1, 99)]
        dr = sc.groupby("drive_key").agg(event_id=("event_id", "first"), tid=("drive_off", "first"),
                                         start=("ytez", "first"), best=("ytez", "min"))
        td = pl[pl.type.isin({"Rushing Touchdown", "Passing Touchdown"})].groupby("drive_key").size()
        fg = pl[pl.type == "Field Goal Good"].groupby("drive_key").size()
        dr["td"] = td.reindex(dr.index).fillna(0) > 0
        dr["fg"] = fg.reindex(dr.index).fillna(0) > 0
        dr["pts"] = np.where(dr.td, 7, np.where(dr.fg, 3, 0))
        dr["opp"] = dr.best <= 40; dr["rz"] = dr.best <= 20
        k3 = [dr.event_id, dr.tid]
        dagg = pd.DataFrame({"drives": dr.groupby(k3).size(), "pts_per_drive": dr.groupby(k3).pts.mean(),
                             "start_fp": (100 - dr.start).groupby(k3).mean(), "rz_trips": dr.groupby(k3).rz.sum()})
        o = dr[dr.opp]; dagg["ppo"] = o.groupby([o.event_id, o.tid]).pts.mean()
        r = dr[dr.rz]; dagg["rz_td_rate"] = r.groupby([r.event_id, r.tid]).td.mean()
        fgk = pl[pl.type.isin({"Field Goal Good", "Field Goal Missed", "Blocked Field Goal"})]
        dagg["fg_att"] = fgk.groupby([fgk.event_id, fgk.drive_off]).size()
        fgm = fgk[fgk.type == "Field Goal Good"]
        dagg["fg_made"] = fgm.groupby([fgm.event_id, fgm.drive_off]).size()
        dagg.index = dagg.index.set_names(agg.index.names)
        allm = agg.join(dagg, how="outer")
        for (eid, tid), r_ in allm.iterrows():
            if tid is None or pd.isna(tid):
                continue
            rows[(eid, str(tid))] = {k: (None if pd.isna(v) else round(float(v), 5)) for k, v in r_.items()}
    out = []
    for d in parsed:
        eid = d["event_id"]
        if eid not in home_of:
            continue
        for tid, t in d.get("teams", {}).items():
            r = dict(rows.get((eid, tid), {}))
            for c in ("fg_att", "fg_made"):
                r[c] = r.get(c) or 0.0
            box = t.get("box") or {}
            r["third_conv"] = _frac(box.get("thirdDownEff"), 0); r["third_att"] = _frac(box.get("thirdDownEff"), 1)
            r["turnovers"] = pd.to_numeric(box.get("turnovers"), errors="coerce")
            r["total_yards"] = pd.to_numeric(box.get("totalYards"), errors="coerce")
            r["pen_yds"] = _frac(box.get("totalPenaltiesYards"), 1)
            df_ = t.get("def") or {}
            r["sacks"] = df_.get("SACKS"); r["tfl"] = df_.get("TFL"); r["qb_hur"] = df_.get("QB HUR")
            q = t.get("qb") or {}
            for k in ("id", "name", "att", "cmp", "yds", "td", "int"):
                r[f"qb_{k}"] = q.get(k)
            r = {k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in r.items()}
            out.append({"game_id": eid, "team_id": str(tid), "stats": r})
    return out
