"""Fälle: Ziehen einer Reihe, wahrer Pegel und die exakten Größen (alle Fälle haben einen exakten wahren Pegel).

* Spleiß-Fall (Konstruktion mit Dichtesprung): Körper Gleich(0, 1) mit Masse 1 − ζ₀, ab u₀ = 1 exakt GPD(ξ = 3/10, σ = 1/2) mit Masse ζ₀ über u₀.
* Lognormal (σ = 1) und Exponential: glatte Abweichung bzw. GPD bei jeder Schwelle exakt.
* Vehikel B: die ausgedachte Verweilzeit-Mischung (`sws_vehicle.py`).
Zufall nur aus `sws_rng.py` (SplitMix64, unveränderte Vorlage)."""

import math
from statistics import NormalDist

import numpy as np

import sws_constants as C
import sws_vehicle as V
from sws_rng import SplitMix64

_ND = NormalDist()


def spliced_level(p, zeta0, u0=C.U0, xi=C.XI0, sigma=C.SIG0):
    """Wahrer Pegel des Spleiß-Falls: x_p = u₀ + (σ/ξ)((p/ζ₀)^(−ξ) − 1), gültig für p < ζ₀."""
    return u0 + sigma / xi * ((p / zeta0) ** (-xi) - 1.0)


def true_level(case_key, p=None):
    """Wahrer Pegel zur Überschreitungswahrscheinlichkeit p je Beobachtung (Standard: das p des Falls)."""
    if case_key not in C.P_TARGET:
        raise ValueError(case_key)
    p = C.P_TARGET[case_key] if p is None else p
    if case_key in C.SPLICE:
        return spliced_level(p, C.SPLICE[case_key])
    if case_key == "lognormal":
        return math.exp(-C.LOGNORMAL_SIGMA * _ND.inv_cdf(p))
    if case_key == "exponential":
        return -math.log(p)
    if case_key == "vehicle":
        return V.level(p)
    raise ValueError(case_key)


def draw(case_key, n, seed_or_rng):
    """n unabhängige Beobachtungen des Falls; `seed_or_rng` ist ein Zufalls-Seed (ganze Zahl) oder ein SplitMix64."""
    rng = seed_or_rng if isinstance(seed_or_rng, SplitMix64) else SplitMix64(int(seed_or_rng))
    if case_key in C.SPLICE:
        zeta = C.SPLICE[case_key]
        out = []
        for _ in range(n):
            if rng.random() < zeta:
                v = 1.0 - rng.random()
                out.append(C.U0 + C.SIG0 / C.XI0 * (v ** (-C.XI0) - 1.0))
            else:
                out.append(rng.random() * C.U0)
        return np.array(out)
    if case_key == "lognormal":
        return np.array([math.exp(C.LOGNORMAL_SIGMA * rng.normal()) for _ in range(n)])
    if case_key == "exponential":
        return np.array([rng.expovariate(1.0) for _ in range(n)])
    if case_key == "vehicle":
        return np.array(V.draw(rng, n))
    raise ValueError(case_key)


def spliced_mrl(u, zeta0):
    """Exaktes e(u) = E[X − u | X > u] des Spleiß-Falls (geschlossen): u ≥ u₀: (σ + ξ(u − u₀))/(1 − ξ); darunter mit a = u₀ − u:
    ((1−ζ₀)/u₀·a²/2 + ζ₀(a + σ/(1−ξ))) / ((1−ζ₀)·a/u₀ + ζ₀)."""
    if u >= C.U0:
        return (C.SIG0 + C.XI0 * (u - C.U0)) / (1.0 - C.XI0)
    a = C.U0 - u
    num = (1.0 - zeta0) / C.U0 * a * a / 2.0 + zeta0 * (a + C.SIG0 / (1.0 - C.XI0))
    den = (1.0 - zeta0) * a / C.U0 + zeta0
    return num / den


def derive_seed(*parts):
    """Zufalls-Seed einer Messreihe aus ganzen Zahlen (Fall, Lauf, ...): jeder Teil wird durch einen SplitMix64-Schritt eingemischt, benachbarte Läufe bekommen
    unabhängig wirkende Ströme."""
    s = 0x243F6A8885A308D3
    for part in parts:
        s = SplitMix64((s ^ (int(part) & 0xFFFFFFFFFFFFFFFF)) & 0xFFFFFFFFFFFFFFFF).next_u64()
    return s


def _phibar(z):
    """Überlebensfunktion der Standardnormalverteilung."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def exact_mrl(case_key, u):
    """Exaktes Mean Residual Life e(u) = E[X − u | X > u] jedes Falls (geschlossene Form; Vehikel B als Summe über die 1224 Komponenten)."""
    if case_key in C.SPLICE:
        return spliced_mrl(u, C.SPLICE[case_key])
    if case_key == "exponential":
        return 1.0
    if case_key == "lognormal":
        s = C.LOGNORMAL_SIGMA
        return math.exp(s * s / 2.0) * _phibar(math.log(u) / s - s) / _phibar(math.log(u) / s) - u
    if case_key == "vehicle":
        w, mu, sg = V.components()
        lu = math.log(u)
        num = math.fsum(wi * (math.exp(m + s * s / 2.0) * _phibar((lu - m - s * s) / s) - u * _phibar((lu - m) / s)) for wi, m, s in zip(w, mu, sg))
        return num / V.survival(u)
    raise ValueError(case_key)
