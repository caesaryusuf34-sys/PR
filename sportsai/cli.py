"""Command-line interface.

  python -m sportsai predict "Miami vs Clemson"      # auto-updates, learns if due, predicts, logs
  python -m sportsai slate --date 20261010           # predict every unstarted game on a date
  python -m sportsai update                          # sync -> results -> evaluate -> learn if due
  python -m sportsai learn --force                   # run a learning cycle now
  python -m sportsai status | history | audit | versions
  python -m sportsai bootstrap [--as-of ISO]         # build the database and the first model
  python -m sportsai daemon --every-hours 6          # keep the loop running unattended

Baseball (MLB, KBO, NPB, CPBL) - same engine, one adapter per league:
  python -m sportsai menu                            # interactive: choose a league, then today / tomorrow / ...
  python -m sportsai --sport kbo today               # every not-yet-started game today (league-local date)
  python -m sportsai --sport npb tomorrow
  python -m sportsai --sport mlb day --date 2026-10-05
  python -m sportsai --sport cpbl predict "Brothers vs Monkeys"
"""
from __future__ import annotations
import argparse, json, sys, time
import pandas as pd

from .core.engine import Engine
from .core.timeutil import now, ts


def _print(obj):
    print(json.dumps(obj, indent=1, default=str))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sportsai")
    ap.add_argument("--sport", default="ncaaf", help="ncaaf, nfl, mlb, kbo, npb, cpbl")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("menu", help="interactive baseball menu (MLB / KBO / NPB / CPBL)")
    for name in ("today", "tomorrow", "day"):
        p = sub.add_parser(name, help="predict every not-yet-started game on a league-local date")
        if name == "day":
            p.add_argument("--date", required=True, help="YYYY-MM-DD in the league's local time zone")
        p.add_argument("--no-update", action="store_true"); p.add_argument("--no-save", action="store_true")
        p.add_argument("--include-started", action="store_true", help="also predict started/final games as non-blind backtests")
        p.add_argument("--report", default=None, help="write a markdown report to this path")
    p = sub.add_parser("predict"); p.add_argument("query", nargs="+"); p.add_argument("--no-update", action="store_true")
    p.add_argument("--no-learn", action="store_true"); p.add_argument("--no-save", action="store_true"); p.add_argument("--json", action="store_true")
    p = sub.add_parser("slate"); p.add_argument("--date", required=True, help="YYYYMMDD (US/Eastern game day)")
    p.add_argument("--no-update", action="store_true"); p.add_argument("--no-save", action="store_true")
    p.add_argument("--backtest", action="store_true", help="predict already-started games with a pre-kickoff cutoff (logged as non-blind)")
    p = sub.add_parser("week"); p.add_argument("--no-update", action="store_true"); p.add_argument("--no-save", action="store_true")
    p.add_argument("--report", default=None, help="write a markdown report to this path")
    p = sub.add_parser("update"); p.add_argument("--no-learn", action="store_true"); p.add_argument("--force-learn", action="store_true")
    p = sub.add_parser("learn"); p.add_argument("--force", action="store_true")
    sub.add_parser("status"); sub.add_parser("versions")
    p = sub.add_parser("history"); p.add_argument("--limit", type=int, default=30)
    p = sub.add_parser("audit"); p.add_argument("--limit", type=int, default=50)
    p = sub.add_parser("bootstrap"); p.add_argument("--as-of", default=None); p.add_argument("--no-sync", action="store_true")
    p = sub.add_parser("invalidate"); p.add_argument("pred_ids", nargs="+"); p.add_argument("--reason", required=True)
    p = sub.add_parser("import-legacy"); p.add_argument("--predictions", default="predictions.csv"); p.add_argument("--features", default="features.csv")
    p = sub.add_parser("daemon"); p.add_argument("--every-hours", type=float, default=6.0)
    a = ap.parse_args(argv)
    if a.cmd in (None, "menu"):
        from .sports.baseball.menu import run_menu
        return run_menu()
    E = Engine(a.sport)

    if a.cmd == "predict":
        rec = E.predict(" ".join(a.query), update=not a.no_update, save=not a.no_save, learn=not a.no_learn)
        if a.json:
            _print({k: v for k, v in rec.items() if not k.endswith("_json")})
        else:
            print(E.format(rec))
    elif a.cmd == "slate":
        if not a.no_update:
            E.auto_update()
        day = pd.Timestamp(a.date).tz_localize("America/New_York")
        g = E.store.games(E.sport)
        g = g[(g.kickoff_utc >= day.tz_convert("UTC") + pd.Timedelta(hours=5)) & (g.kickoff_utc < day.tz_convert("UTC") + pd.Timedelta(hours=29))]
        started = g.kickoff_utc <= now()
        if a.backtest:
            for k, grp in g.groupby("kickoff_utc"):
                recs = E.predict_games(grp, origin="backtest" if k <= now() else "live", save=not a.no_save, cutoff=min(k, now()), context=False)
                for r in recs:
                    print(f"{r['away_name']} @ {r['home_name']}: P(home)={r['p_home']:.3f} margin {r['pred_margin']:+.1f} [{r['origin']}]")
        else:
            g = g[~started]
            if g.empty:
                print("no unstarted games on that date"); return
            for r in E.predict_games(g, origin="live", save=not a.no_save):
                print(E.format(r)); print("-" * 70)
    elif a.cmd in ("today", "tomorrow", "day"):
        if not a.no_update:
            E.auto_update()
        local_today = now().tz_convert(getattr(E.adapter, "local_tz", "America/New_York")).date()
        day = {"today": local_today, "tomorrow": local_today + pd.Timedelta(days=1)}.get(a.cmd) or pd.Timestamp(a.date).date()
        recs, started = E.predict_day(day, save=not a.no_save, include_started=a.include_started)
        print(f"{E.sport.upper()} {day}: {len(recs)} prediction(s)")
        for r in recs:
            print(E.format(r)); print("-" * 70)
        if len(started) and not a.include_started:
            print("not predicted (started/final):", ", ".join(f"{r.away_name} @ {r.home_name}" for r in started.itertuples()))
        if a.report and recs:
            ch = E.registry.champion_row()
            md = E.week_report(recs, started if not a.include_started else started.iloc[:0], f"{E.sport.upper()} — predictions for {day}",
                               json.loads(ch.validation_json) if ch is not None and ch.validation_json else None)
            open(a.report, "w").write(md); print(f"report written to {a.report}")
    elif a.cmd == "week":
        if not a.no_update:
            E.auto_update()
        recs, started = E.predict_week(save=not a.no_save)
        for r in recs:
            print(E.format(r)); print("-" * 70)
        if len(started):
            print("not predicted (started/final):", ", ".join(f"{r.away_name} @ {r.home_name}" for r in started.itertuples()))
        if a.report and recs:
            import json as _j
            ch = E.registry.champion_row()
            md = E.week_report(recs, started, f"{E.sport.upper()} — predictions for the current week",
                               _j.loads(ch.validation_json) if ch is not None and ch.validation_json else None)
            open(a.report, "w").write(md); print(f"report written to {a.report}")
    elif a.cmd == "update":
        out = E.auto_update(learn=not a.no_learn, force_learn=a.force_learn)
        if "learning" in out:
            l = out["learning"]; print(f"[learn] decision={l['decision']} {l.get('champion_before')} -> {l.get('champion_after')}: {l.get('reason')}")
    elif a.cmd == "learn":
        l = E.learner().run(trigger="manual", force=a.force)
        print(f"[learn] decision={l['decision']} {l.get('champion_before')} -> {l.get('champion_after')}: {l.get('reason')}")
    elif a.cmd == "status":
        _print(E.status())
    elif a.cmd == "versions":
        print(E.registry.versions()[["version", "status", "kind", "parent", "created_utc", "train_cutoff_utc", "n_train", "reason"]].to_string())
    elif a.cmd == "history":
        ev = E.store.evaluated(E.sport).sort_values("kickoff_utc").tail(a.limit)
        for r in ev.itertuples():
            pick = r.home_name if r.p_home >= 0.5 else r.away_name
            print(f"{r.kickoff_utc[:16]} {r.away_name} @ {r.home_name}: pick {pick} ({max(r.p_home, 1 - r.p_home):.0%}) "
                  f"margin {r.pred_margin:+.1f} vs actual {r.actual_margin:+.0f} {'✓' if r.correct else '✗'} [{r.origin} {r.model_version}]")
    elif a.cmd == "audit":
        from .core.audit import audit
        _print(audit(E, limit=a.limit))
    elif a.cmd == "bootstrap":
        if not a.no_sync:
            print("[bootstrap] syncing all seasons…", flush=True)
            print(E.sync(full=True))
        res = E.learner().bootstrap(as_of=ts(a.as_of) if a.as_of else None)
        _print(res)
    elif a.cmd == "invalidate":
        E.store.annotate(a.pred_ids, "invalid", a.reason); print(f"marked {len(a.pred_ids)} prediction(s) invalid")
    elif a.cmd == "import-legacy":
        from .core.legacy import import_legacy
        print(f"imported {import_legacy(E, a.predictions, a.features)} legacy predictions")
    elif a.cmd == "daemon":
        while True:
            try:
                E.auto_update()
            except Exception as e:  # keep the loop alive; the next cycle retries
                print(f"[daemon] cycle failed: {e}", file=sys.stderr)
            time.sleep(a.every_hours * 3600)


if __name__ == "__main__":
    main()
