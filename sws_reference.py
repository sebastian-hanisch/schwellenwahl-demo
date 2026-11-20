"""Eingefrorene Messungen und Rechnungen: Laden der drei JSON-Dateien (`sws_gof_crit.json`, `sws_population.json`, `sws_rules.json`).

Jede Datei enthält Quelle und Stand je Feld (`quelle`, `stand`) und wird von einem Werkzeug unter `tools/` erzeugt; `--check` rechnet neu und vergleicht als Band."""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _load(name):
    return json.loads((HERE / name).read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def crit_data():
    """Kalibrierung der Regel G: {"A": {k: Wert}, "B": {k: Wert}} (Schlüssel ganze Zahlen)."""
    d = _load("sws_gof_crit.json")
    return {v: {int(k): float(x) for k, x in d[v].items()} for v in ("A", "B")}


@lru_cache(maxsize=1)
def population_data():
    """Populationskurven ohne Zufall: {"vehicle": [Zeilen], "lognormal": [Zeilen]} je Zeile ζ, u, ξ, σ_u, σ*, ln_err (aufsteigend in ζ)."""
    d = _load("sws_population.json")
    return {key: d[key]["zeilen"] for key in ("vehicle", "lognormal")}


@lru_cache(maxsize=1)
def rules_data():
    return _load("sws_rules.json")


def population_interp(case_key, zeta, field="ln_err"):
    """Populationskurve eines gefrorenen Falls bei ζ, linear in ln ζ zwischen den Stützstellen (die Stützstellen enthalten alle Gitter-k von n = 2000 und n = 5000)."""
    rows = population_data()[case_key]
    z = np.log([r["zeta"] for r in rows])
    v = [r[field] for r in rows]
    return float(np.interp(np.log(zeta), z, v))


def population_errors_for_grid(case_key, ks, n):
    """Populationsfehler ln(Pegel/wahr) an den k des Gitters (Feld, Index 0 = niedrigste Schwelle wie `ks`)."""
    return np.array([population_interp(case_key, k / n) for k in ks])
