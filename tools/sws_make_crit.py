"""Erzeugt `sws_gof_crit.json`: die kritischen Werte der Regel G (Anpassungstest) für alle k der beiden Gitter (n = 2000 und n = 5000, 34 verschiedene k).

Kalibrierung: je k und je ξ werden `R_CAL` exakte GPD-Stichproben der Größe k gezogen (SplitMix64, Inversion), die GPD per ML angepasst und √k·D (D = Kolmogorov-Smirnov-
Abstand der Stichprobe zur ANGEPASSTEN GPD) berechnet; der kritische Wert ist der 90-%-Quantil über alle Stichproben beider ξ (Mischung). Variante A: ξ ∈ {0; 0,3} (enthält die
Wahrheit des Spleiß-Falls), Variante B: ξ ∈ {0; 0,5} (ohne den wahren Wert). Zähleinheit: Stichproben (je k und Variante 2·R_CAL).

Aufruf (aus dem Ordner der Demo, Test-venv, OMP_NUM_THREADS=1):  python tools/sws_make_crit.py            erzeugt die Datei
                                                                    python tools/sws_make_crit.py --check    rechnet neu und vergleicht als Band (1e-5)"""

import argparse
import json
import math
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
from sws_rng import SplitMix64  # noqa: E402

FILE = ROOT / "sws_gof_crit.json"
R_CAL = 1000
VARIANTS = {"A": (0.0, 0.3), "B": (0.0, 0.5)}
VARIANT_TAG = {"A": 1, "B": 2}


def grid_ks():
    return sorted({int(k) for n in (2000, 5000) for k in A.make_grid(n)})


def calibrate_task(args):
    """Ein (k, Variante): sortierte Liste der √k·D-Werte aus 2·R_CAL Stichproben."""
    k, var, r_cal = args
    vals = []
    for ix, xi0 in enumerate(VARIANTS[var]):
        for r in range(r_cal):
            rng = SplitMix64(K.derive_seed(901, VARIANT_TAG[var], k, ix, r))
            if xi0 == 0.0:
                y = np.array([-math.log(1.0 - rng.random()) for _ in range(k)])
            else:
                y = np.array([((1.0 - rng.random()) ** (-xi0) - 1.0) / xi0 for _ in range(k)])
            xi, sg, _e = A.gpd_ml(y)
            if np.isfinite(xi):
                vals.append(A.ks_distance(y, xi, sg) * math.sqrt(k))
    return k, var, sorted(vals)


def compute(r_cal=R_CAL, procs=None):
    ks = grid_ks()
    tasks = [(k, v, r_cal) for k in ks for v in VARIANTS]
    with Pool(procs or int(os.environ.get("SWS_PROCS", "6"))) as pool:
        res = pool.map(calibrate_task, tasks, chunksize=1)
    out = {"A": {}, "B": {}}
    count = {"A": {}, "B": {}}
    for k, var, vals in res:
        out[var][str(k)] = round(float(np.quantile(vals, 0.90)), 6)
        count[var][str(k)] = len(vals)
    return out, count


def build(r_cal=R_CAL, procs=None):
    t0 = time.time()
    crit, count = compute(r_cal, procs)
    a, b = list(crit["A"].values()), list(crit["B"].values())
    stand = time.strftime("%Y-%m-%d")
    return {
        "beschreibung": "Kritische Werte der Regel G: 90-%-Quantil von √k·D (KS-Abstand der Überschreitungen zur ML-angepassten GPD) unter H0 (exakte GPD), je k des Gitters",
        "quelle": "tools/sws_make_crit.py (SplitMix64 aus sws_rng.py, ML aus sws_algorithm.py)", "stand": stand,
        "felder": {
            "A": {"quelle": "ξ ∈ {0; 0,3} gemischt, je k und ξ R_CAL Stichproben der Größe k", "stand": stand},
            "B": {"quelle": "ξ ∈ {0; 0,5} gemischt (ohne die Wahrheit 0,3 des Spleiß-Falls), je k und ξ R_CAL Stichproben der Größe k", "stand": stand},
            "stichproben_je_k_und_variante": {"quelle": "Zahl der Stichproben mit endlicher ML-Anpassung", "stand": stand},
            "r_cal": {"quelle": "Stichproben je k und ξ", "stand": stand},
        },
        "r_cal": r_cal,
        "stichproben_je_k_und_variante": count,
        "A": crit["A"], "B": crit["B"],
        "spanne": {"A": [min(a), max(a)], "B": [min(b), max(b)]},
        "rechenzeit_s": round(time.time() - t0),
        "umgebung": f"Python {sys.version.split()[0]}, numpy {np.__version__}",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="neu rechnen und als Band (1e-5) mit der Datei vergleichen")
    ap.add_argument("--procs", type=int, default=None)
    args = ap.parse_args()
    if args.check:
        old = json.loads(FILE.read_text(encoding="utf-8"))
        crit, _count = compute(old["r_cal"], args.procs)
        worst = max(abs(crit[v][k] - old[v][k]) for v in ("A", "B") for k in old[v])
        print(f"größter Abstand zur Datei: {worst:.2e}")
        sys.exit(0 if worst <= 1e-5 else 1)
    data = build(procs=args.procs)
    FILE.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"geschrieben: {FILE.name}, Spanne A {data['spanne']['A']}, B {data['spanne']['B']}, {data['rechenzeit_s']} s")


if __name__ == "__main__":
    main()
