"""Erzeugt `sws_rules.json`: die eingefrorenen Messungen der Regeln S, M, G (eigene Festlegungen) und der festen k an fünf Fällen.

Je Fall R Läufe (SplitMix64, jeder Lauf ein eigener Zufalls-Seed): Spleiß-Fall Anteil 10 % und 3 % (n = 2000, p = 5·10⁻⁵, R = 400), Lognormal und Exponential (n = 2000,
p = 5·10⁻⁵, R = 400; im Plan nicht vorgesehen, hier mitgemessen), Vehikel B (n = 5000, p = 10⁻⁵ je Container, R = 300). Je Lauf: 18 ML-Anpassungen (Gitter), die Regeln S und M mit den Bändern c = 1,5 / 2 / 3,
G (Kalibrierung ξ ∈ {0; 0,3}) und G2 (Kalibrierung ξ ∈ {0; 0,5}) und fünf feste k. Voraussetzung: `sws_gof_crit.json` liegt vor (Werkzeug `sws_make_crit.py`).
Zähleinheit: Läufe. Fehlermaß ln(Pegel-Dach / wahrer Pegel).

Aufruf:  python tools/sws_make_rules.py [Fall ...]          erzeugt/ergänzt die Datei (ohne Fall: alle fünf)
         python tools/sws_make_rules.py --check Fall         rechnet den Fall mit den Parametern der Datei neu und vergleicht als Band (Standardfehler)"""

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

import sws_algorithm as A  # noqa: E402
import sws_cases as K  # noqa: E402
import sws_constants as C  # noqa: E402
import sws_evaluation as EV  # noqa: E402

FILE = ROOT / "sws_rules.json"
RUNS = {"splice10": 400, "splice03": 400, "lognormal": 400, "exponential": 400, "vehicle": 300}
CASE_INDEX = {"splice10": 1, "splice03": 2, "lognormal": 3, "exponential": 4, "vehicle": 5}
CHUNK = 25
SPREAD_RULES = ("S2", "M2", "G", "obere 2 %", "obere 5 %", "obere 10 %", "k = √n")


def run_chunk(args):
    """Läufe [start, start + count) eines Falls: je Lauf die Tabelle je Schwelle, die gewählten Stufen und die Pegel der festen k."""
    case_key, start, count, seed_base = args
    n = C.N_FROZEN[case_key]
    p = C.P_TARGET[case_key]
    ks = A.make_grid(n)
    crit, crit2 = EV.crit_arrays(ks)
    rows = []
    for r in range(start, start + count):
        x = K.draw(case_key, n, K.derive_seed(seed_base, CASE_INDEX[case_key], r))
        res = A.analyze(x, p, ks, crit, crit2)
        rows.append({"level": res["stats"]["level"], "u": res["stats"]["u"], "picks": res["picks"], "fixed": res["fixed"]})
    return start, rows


def simulate(case_key, runs, seed_base, procs):
    chunks = [(case_key, s, min(CHUNK, runs - s), seed_base) for s in range(0, runs, CHUNK)]
    with Pool(procs) as pool:
        res = pool.map(run_chunk, chunks, chunksize=1)
    res.sort(key=lambda t: t[0])
    return [row for _s, rows in res for row in rows]


def aggregate(case_key, rows, runs, seed_base, seconds):
    n = C.N_FROZEN[case_key]
    p = C.P_TARGET[case_key]
    ks = A.make_grid(n)
    truth = K.true_level(case_key, p)
    levels = np.array([r["level"] for r in rows])
    us = np.array([r["u"] for r in rows])
    u0 = C.U0 if case_key in C.SPLICE else None
    grid = [{"k": int(k), **EV.error_stats(levels[:, j], truth)} for j, k in enumerate(ks)]
    fixed = {}
    for name in rows[0]["fixed"]:
        fixed[name] = {"k": int(rows[0]["fixed"][name][0]), **EV.error_stats([r["fixed"][name][1] for r in rows], truth)}
    rules = {}
    chosen_levels = {}
    for key in A.RULE_KEYS_ALL:
        picks = np.array([r["picks"][key] for r in rows])
        rules[key] = EV.aggregate_rule(levels, picks, us, ks, truth, u0)
        chosen_levels[key] = levels[np.arange(len(rows)), picks]
    cols = [chosen_levels[k] for k in ("S2", "M2", "G")] + [np.array([r["fixed"][nm][1] for r in rows]) for nm in SPREAD_RULES[3:]]
    spread = EV.spread_stats(np.array(cols).T, truth)
    return {
        "n": n, "p": p, "wahrer_pegel": truth, "laeufe": runs, "seed_basis": seed_base, "gitter_k": [int(k) for k in ks],
        "gitter": grid, "feste_k": fixed, "regeln": rules, "streuung_regeln": {"regeln": list(SPREAD_RULES), **spread}, "rechenzeit_s": round(seconds),
    }


def build_case(case_key, runs=None, seed_base=7000, procs=None):
    runs = runs or RUNS[case_key]
    t0 = time.time()
    rows = simulate(case_key, runs, seed_base, procs or int(os.environ.get("SWS_PROCS", "6")))
    return aggregate(case_key, rows, runs, seed_base, time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cases", nargs="*", help="Fälle (splice10 splice03 lognormal exponential vehicle); ohne Angabe alle")
    ap.add_argument("--check", action="store_true", help="den Fall mit den Parametern der Datei neu rechnen und als Band vergleichen")
    ap.add_argument("--procs", type=int, default=None)
    args = ap.parse_args()
    cases = args.cases or list(RUNS)
    old = json.loads(FILE.read_text(encoding="utf-8")) if FILE.exists() else {"fälle": {}}
    if args.check:
        bad = 0
        for ck in cases:
            o = old["fälle"][ck]
            new = build_case(ck, o["laeufe"], o["seed_basis"], args.procs)
            worst = max(abs(a["bias"] - b["bias"]) for a, b in zip(o["gitter"], new["gitter"]))
            worst_r = max(abs(o["regeln"][r]["rmse"] - new["regeln"][r]["rmse"]) for r in o["regeln"])
            print(f"{ck}: größter Abstand der Verzerrung je Stufe {worst:.2e}, des RMSE je Regel {worst_r:.2e}")
            bad += worst > 1e-6 or worst_r > 1e-6
        sys.exit(1 if bad else 0)
    stand = time.strftime("%Y-%m-%d")
    for ck in cases:
        print(f"{ck}: {RUNS[ck]} Läufe ...", flush=True)
        old["fälle"][ck] = build_case(ck, procs=args.procs)
        print(f"  fertig in {old['fälle'][ck]['rechenzeit_s']} s", flush=True)
    old.update({
        "beschreibung": "Eingefrorene Messungen der Regeln S, M, G und der festen k (Zähleinheit Läufe; Fehler ln(Pegel-Dach/wahr)); eigene Festlegungen, keine Literaturverfahren",
        "quelle": "tools/sws_make_rules.py (SplitMix64 aus sws_rng.py; Gitter, ML und Regeln aus sws_algorithm.py; kritische Werte der Regel G aus sws_gof_crit.json)", "stand": stand,
        "felder": {"fälle": {"quelle": "je Fall: Gitter, feste k, Regeln (Schlüssel S1.5 S2 S3 M1.5 M2 M3 G G2), Streuung zwischen den Regeln auf derselben Reihe", "stand": stand}},
        "umgebung": f"Python {sys.version.split()[0]}, numpy {np.__version__}",
    })
    FILE.write_text(json.dumps(old, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"geschrieben: {FILE.name}")


if __name__ == "__main__":
    main()
