"""Vehikel B: die ausgedachte Verweilzeit-Mischung der Linie (gemeinsame Gewichtstabelle, 1224 Komponenten).

Quelle und Stand je Feld: `sws_mixture_quelle.json`; `sws_mixture.json` ist eine unveränderte Kopie der gemeinsamen Tabelle der Linie (Kopier-Test in `tests/test_copies.py`).
Je Komponente ist ln(Verweilzeit) normalverteilt mit Mittel μ und Streuung s; das Gewicht ist ihr Anteil an den Containern. Die Tabelle ist die EXAKTE Aufzählung
(Art × Kunde × Zoll × Wochenende × Vorlauf, 4 704 Schritte, 1 224 verschiedene Komponenten), keine Stichprobe; der wahre Pegel kommt aus ihr, nicht aus gezogenen Werten.
Ziehen: Komponente über die Summenverteilung der Gewichte, dann exp(μ + s·normal()) aus dem SplitMix64 der Vorlage."""

import bisect
import json
import math
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
P_CONTAINER = 1e-5                                        # Zielgröße (a): Überschreitungswahrscheinlichkeit je Container
P_TAGESMAX = 1.0 - (1.0 - 1e-3) ** (1.0 / 100.0)          # Zielgröße (b): je Container, wenn das Tagesmaximum aus 100 Containern einmal in 1000 Tagen überschritten wird


@lru_cache(maxsize=1)
def components():
    """(Gewichte, μ, s) der 1224 Komponenten als Tupel."""
    d = json.loads((HERE / "sws_mixture.json").read_text(encoding="utf-8"))
    rows = d["komponenten"]
    return tuple(r[0] for r in rows), tuple(r[1] for r in rows), tuple(r[2] for r in rows)


@lru_cache(maxsize=1)
def _cumulative():
    w, _mu, _s = components()
    acc, out = 0.0, []
    for wi in w:
        acc += wi
        out.append(acc)
    return tuple(c / acc for c in out)


@lru_cache(maxsize=1)
def _scaled():
    w, mu, s = components()
    return w, mu, tuple(si * math.sqrt(2.0) for si in s)


def survival(x):
    """Exakte Überlebensfunktion P(D > x) der Mischung (x > 0): Summe der Gewichte mal ½·erfc((ln x − μ)/(s√2))."""
    w, mu, s2 = _scaled()
    lx = math.log(x)
    return math.fsum(wi * 0.5 * math.erfc((lx - m) / s) for wi, m, s in zip(w, mu, s2))


def level(p):
    """Pegel x mit P(D > x) = p, exakt per Bisektion auf ln x (0 < p < 1)."""
    lo, hi = -3.0, 14.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if survival(math.exp(mid)) > p:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-14:
            break
    return math.exp(0.5 * (lo + hi))


def daily_max_level(q=0.999, per_day=100):
    """Quantil des Tagesmaximums aus `per_day` Containern: F^per_day(x) = q, das heißt P(D > x) = 1 − q^(1/per_day) (Zielgröße (b) bei q = 0,999)."""
    return level(-math.expm1(math.log(q) / per_day))


def mean_and_median():
    """Mittel der Verweilzeit (exakt aus der Tabelle) und Median (Bisektion)."""
    w, mu, s = components()
    return math.fsum(wi * math.exp(m + si * si / 2.0) for wi, m, si in zip(w, mu, s)), level(0.5)


def draw(rng, n):
    """n Verweilzeiten aus der Mischung: Komponente über die Summenverteilung der Gewichte, dann exp(μ + s·normal()). Kosten rund 180 µs je Wert (exakte Normalzahl)."""
    _w, mu, s = components()
    cum = _cumulative()
    last = len(cum) - 1
    out = []
    for _ in range(n):
        c = min(bisect.bisect_right(cum, rng.random()), last)
        out.append(math.exp(mu[c] + s[c] * rng.normal()))
    return out
