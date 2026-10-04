"""Interactive menu (Bahasa Indonesia): pilih liga dulu, lalu minta prediksi hari ini, besok, tanggal
tertentu, atau satu pertandingan.

    python -m sportsai            # atau: python -m sportsai menu

Every prediction goes through the same engine as the CLI: auto-update (new results -> grading ->
learning when due), point-in-time features, champion model, immutable logging.
"""
from __future__ import annotations
import json, os
import numpy as np
import pandas as pd

from ...core.engine import Engine
from ...core.timeutil import now

LEAGUES = [("mlb", "MLB  — Major League Baseball (Amerika Serikat)"), ("kbo", "KBO  — Korea Baseball Organization (Korea Selatan)"),
           ("npb", "NPB  — Nippon Professional Baseball (Jepang)"), ("cpbl", "CPBL — Chinese Professional Baseball League (Taiwan)")]
DISPLAY_TZ = os.environ.get("SPORTSAI_DISPLAY_TZ", "Asia/Jakarta")
TZ_LABEL = {"Asia/Jakarta": "WIB", "Asia/Makassar": "WITA", "Asia/Jayapura": "WIT"}.get(DISPLAY_TZ, DISPLAY_TZ)


def _t(iso_utc: str, tz: str) -> str:
    return pd.Timestamp(iso_utc).tz_convert(tz).strftime("%d %b %H:%M")


def kartu(E: Engine, r: dict) -> str:
    """Prediction card with Indonesian labels (factor details stay in the logged English wording)."""
    H, A, p = r["home_name"], r["away_name"], r["p_home"]
    pick = H if p >= 0.5 else A
    e = r["explanation"]
    tag = {"live": "prediksi BUTA sebelum laga", "backtest": "backtest (laga sudah mulai — bukan prediksi buta)",
           "hypothetical": "laga hipotetis (tidak ada jadwal)"}.get(r["origin"], r["origin"])
    tz = getattr(E.adapter, "local_tz", "UTC")
    L = [f"{A} @ {H}",
         f"[{tag} · model {r['model_version']} · mulai {_t(r['kickoff_utc'], DISPLAY_TZ)} {TZ_LABEL} / {_t(r['kickoff_utc'], tz)} waktu lokal]",
         "", f"Prediksi: {pick}",
         f"Peluang Menang: {A} {100 * (1 - p):.0f}% / {H} {100 * p:.0f}%",
         f"Skor Proyeksi: {A} {r['proj_away']:.1f} – {r['proj_home']:.1f} {H}",
         f"Selisih Proyeksi: {abs(r['pred_margin']):.1f} run · Total run: {r['pred_total']:.1f}",
         f"Peluang Upset: {100 * r['upset_prob']:.0f}%",
         f"Keyakinan: {r['confidence']:.1f}/10", "", "Faktor Kunci:"]
    L += [f"  {i}. {f}" for i, f in enumerate(e.get("key_factors", []), 1)]
    L += ["", f"Risiko Utama: {e.get('main_risk', '-')}"]
    n = e.get("notes", {})
    if n.get("starters"):
        L.append(f"Starter: {n['starters']}")
    if n.get("ties"):
        L.append(f"Catatan seri: {n['ties']}")
    mp = {k[2:]: v for k, v in r["members"].items() if k.startswith("p_") and k not in ("p_home", "p_raw", "p_std")}
    L.append("Model anggota P(" + H + " menang): " + ", ".join(f"{k} {v:.2f}" for k, v in mp.items()) + f" · ensemble {p:.2f}")
    if r.get("pred_id"):
        L.append(f"{'sudah tercatat' if r.get('duplicate_of_existing') else 'tercatat'} sebagai prediksi {r['pred_id'][:12]}… "
                 f"(hash {r['snapshot_sha256'][:12]}…)")
    return "\n".join(L)


def ringkasan(recs: list[dict]) -> str:
    if not recs:
        return ""
    L = ["", f"{'Laga':<52} {'Mulai ' + TZ_LABEL:<14} {'Pick':<26} {'Menang':>6} {'Skor':>11} {'Yakin':>5}"]
    for r in recs:
        p = r["p_home"]; pick = r["home_name"] if p >= 0.5 else r["away_name"]
        L.append(f"{(r['away_name'] + ' @ ' + r['home_name'])[:51]:<52} {_t(r['kickoff_utc'], DISPLAY_TZ):<14} {pick[:25]:<26} "
                 f"{100 * max(p, 1 - p):>5.0f}% {r['proj_away']:>4.1f}-{r['proj_home']:<4.1f} {r['confidence']:>5.1f}")
    return "\n".join(L)


def jadwal(E: Engine, day) -> str:
    g = E.games_on(day)
    if g.empty:
        return f"Tidak ada laga {E.sport.upper()} pada {day}."
    st = E.store.starters(E.sport)
    res = E.store.df("SELECT game_id, home_score, away_score FROM results WHERE sport=?", (E.sport,)).set_index("game_id")
    fb = E.adapter.feature_builder(E.store)
    L = [f"Jadwal {E.sport.upper()} {day}:"]
    c = int(now().value)
    for r in g.itertuples():
        sp = []
        for side in ("away", "home"):
            pid, nm, src = fb.resolve_starter(r.game_id, side, c)
            sp.append((nm or (pid if pid and not str(pid).startswith("name:") else None) or "TBD"))
        score = f"  FINAL {int(res.loc[r.game_id].away_score)}-{int(res.loc[r.game_id].home_score)}" if r.game_id in res.index else ""
        L.append(f"  {_t(r.kickoff_utc.isoformat(), DISPLAY_TZ)} {TZ_LABEL}  {r.away_name} ({sp[0]}) @ {r.home_name} ({sp[1]}){score}")
    return "\n".join(L)


def _predict_day(E: Engine, day, ask):
    recs, started = E.predict_day(day)
    if not recs and started.empty:
        print(f"Tidak ada laga {E.sport.upper()} pada {day} (zona waktu liga).")
        return
    for r in recs:
        print(kartu(E, r)); print("-" * 78)
    if len(started):
        res = E.store.df("SELECT game_id, home_score, away_score FROM results WHERE sport=?", (E.sport,)).set_index("game_id")
        print("Sudah mulai / selesai (tidak diprediksi buta):")
        for r in started.itertuples():
            sc = f" — FINAL {int(res.loc[r.game_id].away_score)}-{int(res.loc[r.game_id].home_score)}" if r.game_id in res.index else " — sedang berlangsung"
            print(f"  {r.away_name} @ {r.home_name}{sc}")
    print(ringkasan(recs))


def _status(E: Engine):
    st = E.status()
    print(f"Liga {E.sport.upper()} · prediksi menunggu hasil: {st['pending_predictions']} · {st['learning_due']}")
    for v in st["versions"]:
        print(f"  {v['version']:<8} {v['status']:<9} {v['kind']:<10} dilatih s/d {str(v['train_cutoff_utc'])[:16]} · {v['n_train']} laga · {str(v['reason'])[:90]}")
    ch = E.registry.champion_row()
    if ch is not None and ch.validation_json:
        w = json.loads(ch.validation_json).get("walk_forward", {})
        print(f"  validasi walk-forward ({w.get('n')} laga out-of-sample): akurasi {w.get('accuracy', 0):.1%}, "
              f"log loss {w.get('log_loss', 0):.3f}, Brier {w.get('brier', 0):.3f}, MAE selisih run {w.get('margin_mae', 0):.2f}")
    for k, s in (st.get("record") or {}).items():
        print(f"  rekor {k}: {s['n']} laga, akurasi {s.get('accuracy', 0):.1%}, log loss {s.get('log_loss', 0):.3f}")


def _history(E: Engine, n=20):
    ev = E.store.evaluated(E.sport).sort_values("kickoff_utc").tail(n)
    if ev.empty:
        print("Belum ada prediksi yang sudah dinilai.")
        return
    for r in ev.itertuples():
        pick = r.home_name if r.p_home >= 0.5 else r.away_name
        print(f"  {r.kickoff_utc[:10]} {r.away_name} @ {r.home_name}: pick {pick} ({max(r.p_home, 1 - r.p_home):.0%}) "
              f"skor akhir {r.actual_total / 2 - r.actual_margin / 2:.0f}-{r.actual_total / 2 + r.actual_margin / 2:.0f} "
              f"{'✓' if r.correct else '✗'} [{r.origin} {r.model_version}]")


def _ensure_model(E: Engine, ask) -> bool:
    if E.registry.champion_row() is not None:
        return True
    a = ask(f"Model {E.sport.upper()} belum ada. Bangun sekarang? (unduh data 2021–sekarang lalu latih; bisa makan beberapa menit) [y/N] ")
    if a.strip().lower() not in ("y", "ya", "yes"):
        return False
    print(E.sync(full=True))
    res = E.learner().bootstrap()
    w = res["validation"]["walk_forward"]
    print(f"Model {res['version']} siap · validasi walk-forward: akurasi {w['accuracy']:.1%}, log loss {w['log_loss']:.3f} ({w['n']} laga)")
    return True


def run_menu(ask=input):
    print("=" * 78)
    print(" SportsAI Baseball — mesin prediksi (sistem yang sama dengan NCAAF & NFL)")
    print(f" Waktu ditampilkan dalam {TZ_LABEL}. Prediksi dicatat permanen dan dinilai otomatis setelah laga.")
    print("=" * 78)
    while True:
        print("\nPilih liga:")
        for i, (_, name) in enumerate(LEAGUES, 1):
            print(f"  {i}. {name}")
        print("  0. Keluar")
        c = ask("> ").strip().lower()
        if c in ("0", "q", "keluar", "exit"):
            return
        sport = dict(zip(["1", "2", "3", "4"], [s for s, _ in LEAGUES])).get(c) or (c if c in dict(LEAGUES) else None)
        if not sport:
            print("Pilihan tidak dikenal."); continue
        E = Engine(sport)
        if not _ensure_model(E, ask):
            continue
        while True:
            today = now().tz_convert(E.adapter.local_tz).date()
            print(f"\n[{sport.upper()}] hari ini = {today} (waktu lokal liga)")
            print("  1. Prediksi hari ini\n  2. Prediksi besok\n  3. Prediksi tanggal tertentu\n"
                  "  4. Prediksi satu pertandingan (contoh: \"LG vs KIA\" atau \"Yankees @ Red Sox\")\n"
                  "  5. Jadwal & starter (tanpa prediksi)\n  6. Status model & rekor\n  7. Riwayat prediksi yang sudah dinilai\n"
                  "  8. Update data & belajar sekarang\n  9. Ganti liga\n  0. Keluar")
            c = ask("> ").strip()
            try:
                if c == "1":
                    E.auto_update(); _predict_day(E, today, ask)
                elif c == "2":
                    E.auto_update(); _predict_day(E, today + pd.Timedelta(days=1), ask)
                elif c == "3":
                    d = ask("Tanggal (YYYY-MM-DD): ").strip()
                    E.auto_update(); _predict_day(E, pd.Timestamp(d).date(), ask)
                elif c == "4":
                    q = ask("Pertandingan: ").strip()
                    print(kartu(E, E.predict(q)))
                elif c == "5":
                    d = ask("Tanggal (YYYY-MM-DD, kosong = hari ini): ").strip()
                    E.sync(); print(jadwal(E, pd.Timestamp(d).date() if d else today))
                elif c == "6":
                    _status(E)
                elif c == "7":
                    _history(E)
                elif c == "8":
                    out = E.auto_update()
                    if "learning" in out:
                        l = out["learning"]; print(f"[belajar] keputusan={l['decision']} {l.get('champion_before')} -> {l.get('champion_after')}: {l.get('reason')}")
                elif c == "9":
                    break
                elif c in ("0", "q"):
                    return
                else:
                    print("Pilihan tidak dikenal.")
            except (LookupError, ValueError, RuntimeError) as e:
                print(f"Gagal: {e}")
