"""Feature engineering: game table, play-level EPA, per team-game efficiency metrics,
walk-forward Elo and opponent-adjusted (ridge) ratings.

Every rating used as a feature for a game is computed from games that kicked off strictly
before that game's week (historical rows) or before the slate-day cutoff (today's slate).
"""
import os, glob, gzip, json
import numpy as np, pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
CUTOFF = pd.Timestamp("2026-10-03T12:00Z")

RUSH = {"Rush", "Rushing Touchdown", "Fumble Recovery (Own)", "Fumble Recovery (Opponent)",
        "Fumble Return Touchdown", "Fumble"}
PASS = {"Pass Reception", "Pass Incompletion", "Passing Touchdown", "Sack", "Pass Interception Return",
        "Interception", "Interception Return Touchdown", "Pass Completion", "Pass"}


# ----------------------------------------------------------------------------- games
def load_games():
    g = pd.read_csv(os.path.join(ROOT, "data", "all_games_raw.csv"), dtype={"event_id": str, "home_id": str, "away_id": str})
    g["date"] = pd.to_datetime(g["date"], utc=True)
    g = g[(g.state == "post") & g.home_score.notna() & g.away_score.notna() & (g.date < CUTOFF)].copy()
    g["wk"] = np.where(g.seasontype == 3, 20, g.week)
    g = g.sort_values("date").reset_index(drop=True)
    # division per team-season: share of the team's games that appear in the FBS (group 80) feed
    long = pd.concat([g[["season", "home_id", "group_query"]].rename(columns={"home_id": "tid"}),
                      g[["season", "away_id", "group_query"]].rename(columns={"away_id": "tid"})])
    agg = long.groupby(["season", "tid"]).agg(n=("group_query", "size"), fbs=("group_query", lambda s: (s == 80).mean()))
    agg["div"] = np.where(agg.fbs > 0.5, "FBS", np.where(agg.n >= 4, "FCS", "OTHER"))
    return g, agg["div"]


def division_for(divs, season, tid):
    for s in (season, season - 1, season - 2):
        if (s, tid) in divs.index:
            return divs.loc[(s, tid)]
    return "OTHER"


# ----------------------------------------------------------------------------- Elo
ELO_INIT = {"FBS": 1500.0, "FCS": 1200.0, "OTHER": 950.0}

def run_elo(g, divs, K=24.0, HFA=55.0, revert=0.30, extra=None):
    """Margin-of-victory Elo (538 style). Returns pre-game home/away Elo for each row of g and
    final ratings. `extra` = list of (season, home_id, away_id) for which to just read ratings."""
    r, last_season = {}, None
    pre_h, pre_a = np.empty(len(g)), np.empty(len(g))
    for i, row in enumerate(g.itertuples(index=False)):
        if row.season != last_season:
            if last_season is not None:  # regress toward division mean
                for tid in list(r):
                    d = division_for(divs, row.season - 1, tid)
                    r[tid] = ELO_INIT[d] + (1 - revert) * (r[tid] - ELO_INIT[d])
            last_season = row.season
        for tid in (row.home_id, row.away_id):
            if tid not in r:
                r[tid] = ELO_INIT[division_for(divs, row.season, tid)]
        rh, ra = r[row.home_id], r[row.away_id]
        pre_h[i], pre_a[i] = rh, ra
        hfa = 0 if row.neutral else HFA
        diff = rh + hfa - ra
        exp = 1 / (1 + 10 ** (-diff / 400))
        m = row.home_score - row.away_score
        res = 1.0 if m > 0 else (0.0 if m < 0 else 0.5)
        mult = np.log(abs(m) + 1) * 2.2 / ((diff if m > 0 else -diff) * 0.001 + 2.2)
        delta = K * mult * (res - exp)
        r[row.home_id] += delta; r[row.away_id] -= delta
    return pre_h, pre_a, r


# ----------------------------------------------------------------------------- plays / EPA
def load_parsed():
    games, plays, teams = [], [], []
    for f in glob.glob(os.path.join(ROOT, "data", "raw", "parsed", "*.json.gz")):
        d = json.load(gzip.open(f, "rt"))
        eid = d["event_id"]
        for tid, t in d["teams"].items():
            row = {"event_id": eid, "tid": tid}
            row.update({f"box_{k}": v for k, v in t["box"].items()})
            row.update({f"def_{k}": v for k, v in (t.get("def") or {}).items()})
            q = t.get("qb") or {}
            row.update({f"qb_{k}": v for k, v in q.items()})
            teams.append(row)
        for j, p in enumerate(d["plays"]):
            p = dict(p); p["event_id"] = eid; p["seq"] = j
            plays.append(p)
    return pd.DataFrame(teams), pd.DataFrame(plays)


def add_epa(pl, g):
    """Next-score expected points model fitted on the collected plays, then play-level EPA."""
    hm = g.set_index("event_id")[["home_id", "away_id"]]
    pl = pl.merge(hm, left_on="event_id", right_index=True, how="inner")
    pl = pl.sort_values(["event_id", "seq"]).reset_index(drop=True)
    for c in ("down", "dist", "ytez", "end_down", "end_dist", "end_ytez", "yds", "hs", "as", "period"):
        pl[c] = pd.to_numeric(pl[c], errors="coerce")
    # score before play = previous row's score (same game)
    prev_h = pl.groupby("event_id")["hs"].shift(1).fillna(0); prev_a = pl.groupby("event_id")["as"].shift(1).fillna(0)
    pl["hs"] = pl["hs"].fillna(prev_h); pl["as"] = pl["as"].fillna(prev_a)
    dh, da = pl.hs - prev_h, pl["as"] - prev_a
    pl["off_is_home"] = pl.off == pl.home_id
    pl["pre_margin_off"] = np.where(pl.off_is_home, prev_h - prev_a, prev_a - prev_h)
    # value of the score change on this play for the HOME team (+) / AWAY team (-)
    sc = dh - da
    sc_val = np.sign(sc) * np.select([np.abs(sc) >= 6, np.abs(sc) >= 3, np.abs(sc) >= 2, np.abs(sc) >= 1], [7, 3, 2, 1], 0)
    # ignore PAT-only changes (1 or 2 after a TD) when computing "next score"
    sc_val = np.where((np.abs(sc) <= 2) & pl.type.fillna("").str.contains("Extra|Two|2pt|Conversion", regex=True), 0, sc_val)
    pl["score_val_home"] = sc_val
    half = np.where(pl.period <= 2, 1, np.where(pl.period <= 4, 2, 3))
    pl["half"] = half
    # next score (home perspective) within the half, inclusive of current play
    nxt = np.zeros(len(pl)); cur = 0.0; key = None
    ev, hv, sv = pl.event_id.values, pl.half.values, pl.score_val_home.values
    for i in range(len(pl) - 1, -1, -1):
        k = (ev[i], hv[i])
        if k != key:
            key, cur = k, 0.0
        if sv[i] != 0:
            cur = sv[i]
        nxt[i] = cur
    pl["next_score_off"] = np.where(pl.off_is_home, nxt, -nxt)
    scrim = pl.type.isin(RUSH | PASS) & pl.down.between(1, 4) & pl.ytez.between(1, 99) & pl.dist.between(1, 99)
    pl["scrim"] = scrim
    # EP table: smoothed binned means by down x ytez(5-yd) x dist-bucket, fitted on scrimmage plays
    pl["yb"] = (pl.ytez.clip(1, 99) // 5).astype("Int64")
    pl["db"] = pd.cut(pl.dist, [0, 3, 6, 10, 15, 100], labels=False)
    fit = pl[scrim]
    tab = fit.groupby(["down", "yb", "db"])["next_score_off"].agg(["mean", "size"])
    by_yb = fit.groupby(["yb"])["next_score_off"].mean()
    def ep(down, ytez, dist):
        yb = (np.clip(ytez, 1, 99) // 5).astype(int)
        db = pd.cut(pd.Series(dist), [0, 3, 6, 10, 15, 100], labels=False).values
        out = np.empty(len(down))
        for i in range(len(down)):
            key = (down[i], yb[i], db[i])
            if key in tab.index and tab.loc[key, "size"] >= 30:
                m, n = tab.loc[key, "mean"], tab.loc[key, "size"]
                out[i] = (m * n + by_yb.get(yb[i], 0) * 30) / (n + 30)
            else:
                out[i] = by_yb.get(yb[i], 0)
        return out
    s = pl[scrim].copy()
    s["ep0"] = ep(s.down.values, s.ytez.values, s.dist.fillna(10).values)
    same = (s.end_off == s.off) & s.end_down.between(1, 4) & s.end_ytez.between(1, 99)
    flip = (s.end_off != s.off) & s.end_ytez.between(1, 99)
    s["ep1"] = 0.0
    s.loc[same, "ep1"] = ep(s.loc[same, "end_down"].values, s.loc[same, "end_ytez"].values, s.loc[same, "end_dist"].fillna(10).values)
    s.loc[flip, "ep1"] = -ep(np.ones(flip.sum()), s.loc[flip, "end_ytez"].values, np.full(flip.sum(), 10.0))
    sv_off = np.where(s.off_is_home, s.score_val_home, -s.score_val_home)
    s["epa"] = np.where(sv_off != 0, sv_off - s.ep0, s.ep1 - s.ep0)
    s["epa"] = s.epa.clip(-10, 10)
    yds = s.yds.fillna(0)
    s["success"] = np.select([s.down == 1, s.down == 2], [yds >= 0.5 * s.dist, yds >= 0.7 * s.dist], yds >= s.dist).astype(float)
    s["is_pass"] = s.type.isin(PASS)
    s["explosive"] = np.where(s.is_pass, yds >= 16, yds >= 12).astype(float)
    s["sack"] = (s.type == "Sack").astype(float)
    s["turnover"] = s.to.fillna(False).astype(float)
    # garbage-time filter (Connelly thresholds)
    am = s.pre_margin_off.abs()
    garbage = ((s.period == 2) & (am > 38)) | ((s.period == 3) & (am > 28)) | ((s.period == 4) & (am > 22))
    s = s[~garbage]
    return s


def team_game_metrics(s, pl_all, tb):
    """Aggregate play-level data to offense team-game metrics."""
    s = s.copy()
    s["def_tid"] = np.where(s.off == s.home_id, s.away_id, s.home_id)
    grp = s.groupby(["event_id", "off"])
    m = pd.DataFrame({
        "def_tid": grp.def_tid.first(), "plays": grp.size(),
        "epa_pp": grp.epa.mean(), "sr": grp.success.mean(), "ypp": grp.yds.mean(),
        "expl": grp.explosive.mean(),
        "rush_epa": s[~s.is_pass].groupby(["event_id", "off"]).epa.mean(),
        "pass_epa": s[s.is_pass].groupby(["event_id", "off"]).epa.mean(),
        "rush_sr": s[~s.is_pass].groupby(["event_id", "off"]).success.mean(),
        "pass_sr": s[s.is_pass].groupby(["event_id", "off"]).success.mean(),
        "sack_rate": s[s.is_pass].groupby(["event_id", "off"]).sack.mean(),
        "to_rate": grp.turnover.mean(),
    }).reset_index().rename(columns={"off": "tid"})
    # drive-level: red zone, scoring opportunities, starting field position
    d = pl_all.copy()
    d["ytez"] = pd.to_numeric(d.ytez, errors="coerce")
    d["drive_key"] = d.groupby("event_id")["drive_off"].transform(lambda x: (x != x.shift()).cumsum())
    scrim = d[d.type.isin(RUSH | PASS) & d.ytez.between(1, 99)]
    dr = scrim.groupby(["event_id", "drive_key"]).agg(tid=("drive_off", "first"), start=("ytez", "first"), best=("ytez", "min"))
    td_types = {"Rushing Touchdown", "Passing Touchdown"}
    td = d[d.type.isin(td_types)].groupby(["event_id", "drive_key"]).size().rename("td")
    fg = d[d.type == "Field Goal Good"].groupby(["event_id", "drive_key"]).size().rename("fg")
    dr = dr.join(td).join(fg).fillna({"td": 0, "fg": 0})
    dr["pts"] = np.minimum(dr.td, 1) * 7 + np.where(dr.td > 0, 0, np.minimum(dr.fg, 1) * 3)
    dr["opp"] = dr.best <= 40; dr["rz"] = dr.best <= 20
    da = dr.reset_index().groupby(["event_id", "tid"]).agg(
        drives=("pts", "size"), pts_per_drive=("pts", "mean"), start_fp=("start", lambda x: (100 - x).mean()),
        opps=("opp", "sum"), opp_pts=("pts", lambda x: x[dr.loc[x.index, "opp"].values if False else slice(None)].sum()),
        rz_trips=("rz", "sum"))
    dr2 = dr.reset_index()
    ppo = dr2[dr2.opp].groupby(["event_id", "tid"]).pts.mean().rename("ppo")
    rztd = dr2[dr2.rz].groupby(["event_id", "tid"]).td.apply(lambda x: (x > 0).mean()).rename("rz_td_rate")
    da = da.drop(columns=["opp_pts"]).join(ppo).join(rztd).reset_index()
    # special teams: FG make rate (kicking team = drive offense)
    fgk = d[d.type.isin({"Field Goal Good", "Field Goal Missed", "Blocked Field Goal"})]
    fgs = fgk.groupby(["event_id", "drive_off"]).type.agg(fg_made=lambda x: (x == "Field Goal Good").sum(), fg_att="size").reset_index().rename(columns={"drive_off": "tid"})
    m = m.merge(da, on=["event_id", "tid"], how="left").merge(fgs, on=["event_id", "tid"], how="left")
    m[["fg_made", "fg_att"]] = m[["fg_made", "fg_att"]].fillna(0)
    # box score extras
    tb = tb.copy()
    def frac(x, i):
        try:
            a, b = str(x).replace("/", "-").split("-"); return float([a, b][i])
        except Exception:
            return np.nan
    tb["third_conv"] = tb.box_thirdDownEff.map(lambda x: frac(x, 0)); tb["third_att"] = tb.box_thirdDownEff.map(lambda x: frac(x, 1))
    tb["turnovers"] = pd.to_numeric(tb.box_turnovers, errors="coerce")
    tb["total_yards"] = pd.to_numeric(tb.box_totalYards, errors="coerce")
    tb["pen_yds"] = tb.box_totalPenaltiesYards.map(lambda x: frac(x, 1))
    for c in ("def_SACKS", "def_TFL", "def_QB HUR"):
        if c not in tb: tb[c] = np.nan
    keep = ["event_id", "tid", "third_conv", "third_att", "turnovers", "total_yards", "pen_yds", "def_SACKS", "def_TFL",
            "def_QB HUR", "qb_id", "qb_name", "qb_att", "qb_cmp", "qb_yds", "qb_td", "qb_int"]
    m = m.merge(tb[[c for c in keep if c in tb]], on=["event_id", "tid"], how="outer")
    return m


# ----------------------------------------------------------------------------- adjusted ratings
def ridge_ratings(obs, teams, prior_o, prior_d, lam_o, lam_d):
    """obs: DataFrame with off, dfn, y, loc(+1 home off, -1 away off, 0 neutral).
    Model: y = mu + o[off] + d[dfn] + h*loc. Ridge toward priors. Returns (mu, h, o, d)."""
    idx = {t: i for i, t in enumerate(teams)}; n = len(teams)
    P = 2 + 2 * n
    A = np.zeros((P, P)); b = np.zeros(P)
    if len(obs):
        oi = obs.off.map(idx).values; di = obs.dfn.map(idx).values
        y = obs.y.values.astype(float); loc = obs["loc"].values.astype(float)
        rows = len(obs)
        X = np.zeros((rows, P)); X[:, 0] = 1; X[:, 1] = loc
        X[np.arange(rows), 2 + oi] = 1; X[np.arange(rows), 2 + n + di] = 1
        A += X.T @ X; b += X.T @ y
    A[0, 0] += 1e-3; A[1, 1] += 1e-3 + (50 if len(obs) < 50 else 0)
    po = np.array([prior_o.get(t, 0.0) for t in teams]); pd_ = np.array([prior_d.get(t, 0.0) for t in teams])
    A[2:2 + n, 2:2 + n] += np.eye(n) * lam_o; b[2:2 + n] += lam_o * po
    A[2 + n:, 2 + n:] += np.eye(n) * lam_d; b[2 + n:] += lam_d * pd_
    if len(obs) == 0:
        mu = 0.0
        return mu, 0.0, dict(zip(teams, po)), dict(zip(teams, pd_))
    th = np.linalg.solve(A, b)
    return th[0], th[1], dict(zip(teams, th[2:2 + n])), dict(zip(teams, th[2 + n:]))
