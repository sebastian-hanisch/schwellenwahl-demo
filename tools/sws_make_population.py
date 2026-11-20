"""Erzeugt `sws_population.json`: die Populationskurven (ohne Zufall, ohne Rauschen) für Vehikel B und Lognormal.

Je Überschreitungsanteil ζ: die GPD, die die EXAKTE Überschreitungsverteilung am besten beschreibt (Maximum Likelihood auf der Population, Quantil-Quadratur über
Gauß-Legendre mit 400 Knoten, siehe `sws_population.gpd_population_ml`), und der Fehler ln(Pegel/wahr) des daraus gerechneten Pegels. Die Umkehrung der Überlebensfunktion
des Vehikels B kommt aus einer Tabelle (ln x gegen ln S, 27 001 Punkte, linear interpoliert), die dieses Werkzeug aus den 1224 Komponenten baut (scipy.special.erfc, falls
vorhanden, sonst math.erfc; nur im Werkzeug). Die Stützstellen sind alle ζ = k/n der beiden Gitter (n = 2000 und n = 5000) plus ein dichtes logarithmisches Netz.

Aufruf:  python tools/sws_make_population.py            erzeugt die Datei
         python tools/sws_make_population.py --check    rechnet neu und vergleicht als Band (1e-5)"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

import sws_algorithm as A  # noqa: E402
import sws_cases as K  # noqa: E402
import sws_constants as C  # noqa: E402
import sws_population as PO  # noqa: E402
import sws_vehicle as V  # noqa: E402

FILE = ROOT / "sws_population.json"
LN_X_LO, LN_X_HI, N_TABLE = -1.0, 13.0, 27001


def zeta_nodes():
    z = {k / n for n in (2000, 5000) for k in A.make_grid(n)}
    z |= {float(v) for v in np.geomspace(0.004, 0.4, 60)}
    return sorted(round(v, 12) for v in z)


def vehicle_inverse_table():
    """(ln S aufsteigend, ln x) der Mischung auf einem gleichmäßigen Netz in ln x."""
    w, mu, s = (np.array(v) for v in V.components())
    try:
        from scipy.special import erfc as erfc_fn
    except ImportError:
        erfc_fn = np.vectorize(math.erfc)
    lnx = np.linspace(LN_X_LO, LN_X_HI, N_TABLE)
    lns = np.empty(N_TABLE)
    for a in range(0, N_TABLE, 1500):
        z = (lnx[a:a + 1500, None] - mu[None, :]) / (s[None, :] * math.sqrt(2.0))
        lns[a:a + 1500] = np.log(np.sum(w[None, :] * 0.5 * erfc_fn(z), axis=1))
    return lns[::-1].copy(), lnx[::-1].copy()


def compute():
    nodes = zeta_nodes()
    lns, lnx = vehicle_inverse_table()

    def isf_vehicle(q):
        return np.exp(np.interp(np.log(np.asarray(q, float)), lns, lnx))

    # Genauigkeit der Tabelle gegen die exakte Bisektion (stdlib): höchstens 1e-5 in ln x
    for q in (0.4, 0.1, 1e-3, 1e-5, 1e-9):
        assert abs(math.log(float(isf_vehicle([q])[0])) - math.log(V.level(q))) < 1e-5, q
    out = {}
    for key, isf, p, truth in (("vehicle", isf_vehicle, C.P_TARGET["vehicle"], V.level(C.P_TARGET["vehicle"])),
                               ("lognormal", PO.lognormal_isf, C.P_TARGET["lognormal"], K.true_level("lognormal"))):
        rows = []
        for zeta in nodes:
            r = PO.population_row(isf, zeta, p, truth)
            rows.append({k: (round(v, 9) if isinstance(v, float) else v) for k, v in r.items()})
        out[key] = {"p": p, "wahrer_pegel": truth, "zeilen": rows}
    return out


def build():
    t0 = time.time()
    d = compute()
    stand = time.strftime("%Y-%m-%d")
    return {
        "beschreibung": "Populationskurven ohne Zufall: ML-GPD an die exakte Überschreitungsverteilung bei Überschreitungsanteil ζ, und Fehler ln(Pegel/wahr) bei p",
        "quelle": "tools/sws_make_population.py (sws_population.gpd_population_ml, Quantil-Quadratur mit 400 Gauß-Legendre-Knoten; Vehikel B aus sws_mixture.json)", "stand": stand,
        "felder": {
            "vehicle": {"quelle": "Vehikel B, p = 1e-5 je Container, wahrer Pegel exakt aus der Tabelle (Bisektion), Zeilen je ζ", "stand": stand},
            "lognormal": {"quelle": "Lognormal(σ = 1), p = 5e-5 je Beobachtung, Zeilen je ζ", "stand": stand},
        },
        "vehicle": d["vehicle"], "lognormal": d["lognormal"],
        "rechenzeit_s": round(time.time() - t0),
        "umgebung": f"Python {sys.version.split()[0]}, numpy {np.__version__}",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="neu rechnen und als Band (1e-5) mit der Datei vergleichen")
    args = ap.parse_args()
    if args.check:
        old = json.loads(FILE.read_text(encoding="utf-8"))
        new = compute()
        worst = 0.0
        for key in ("vehicle", "lognormal"):
            for a, b in zip(old[key]["zeilen"], new[key]["zeilen"]):
                for f in ("u", "xi", "sigma_u", "sigma_star", "ln_err"):
                    worst = max(worst, abs(a[f] - b[f]))
        print(f"größter Abstand zur Datei: {worst:.2e}")
        sys.exit(0 if worst <= 1e-5 else 1)
    data = build()
    FILE.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    v = {r["zeta"]: r for r in data["vehicle"]["zeilen"]}
    print(f"geschrieben: {FILE.name}, {len(data['vehicle']['zeilen'])} Zeilen je Fall, {data['rechenzeit_s']} s; "
          f"Vehikel B k = 25 von 5000: {v[0.005]['ln_err']:+.4f}, k = 648: {v[648 / 5000]['ln_err']:+.4f}")


if __name__ == "__main__":
    main()
