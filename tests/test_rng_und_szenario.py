"""Zufall und Fälle: SplitMix64 ist reproduzierbar (bekannte Folge), die gezogenen Reihen haben die behauptete Verteilung (als Band mit Standardfehler) und sind für gleiche
Zufalls-Seeds gleich; abgeleitete Seeds sind verschieden."""

import math

import numpy as np
import pytest

import sws_cases as K
import sws_constants as C
from sws_rng import SplitMix64


def test_splitmix64_bekannte_folge_fuer_seed_0():
    """Referenzwerte des SplitMix64 mit Seed 0 (Vigna): 16294208416658607535, 7960286522194355700, 487617019471545679."""
    r = SplitMix64(0)
    assert [r.next_u64() for _ in range(3)] == [16294208416658607535, 7960286522194355700, 487617019471545679]


def test_random_liegt_in_0_1_und_ist_reproduzierbar():
    a, b = SplitMix64(42), SplitMix64(42)
    xs = [a.random() for _ in range(1000)]
    assert xs == [b.random() for _ in range(1000)] and all(0.0 <= v < 1.0 for v in xs)
    assert abs(float(np.mean(xs)) - 0.5) < 4 * math.sqrt(1 / 12 / 1000)


@pytest.mark.parametrize("case_key", ["splice10", "splice03", "lognormal", "exponential", "vehicle"])
def test_gleicher_seed_gleiche_reihe_anderer_seed_andere_reihe(case_key):
    a, b, c = K.draw(case_key, 200, 9), K.draw(case_key, 200, 9), K.draw(case_key, 200, 10)
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    assert a.shape == (200,) and np.all(np.isfinite(a)) and np.all(a >= 0)


def test_draw_nimmt_auch_einen_generator():
    assert np.array_equal(K.draw("exponential", 50, 4), K.draw("exponential", 50, SplitMix64(4)))


@pytest.mark.parametrize("case_key", ["splice10", "splice03"])
def test_spleissfall_anteil_ueber_u0_und_gpd_im_tail(case_key):
    zeta0 = C.SPLICE[case_key]
    x = K.draw(case_key, 40000, 21)
    share = float(np.mean(x > 1.0))
    assert abs(share - zeta0) < 4 * math.sqrt(zeta0 * (1 - zeta0) / x.size)
    y = x[x > 1.0] - 1.0
    assert abs(y.mean() - 5 / 7) < 4 * y.std(ddof=1) / math.sqrt(y.size)                 # mittlere Überschreitung der GPD: σ/(1 − ξ) = 5/7
    body = x[x <= 1.0]
    assert abs(body.mean() - 0.5) < 4 * body.std(ddof=1) / math.sqrt(body.size)         # Körper gleichverteilt auf (0, 1)


def test_ueberlebensfunktion_des_spleissfalls_stimmt_mit_den_haeufigkeiten():
    x = K.draw("splice10", 60000, 5)
    for t in (0.2, 0.9, 1.5, 3.0):
        p = 0.9 * (1 - t) + 0.1 if t < 1 else 0.1 * (1 + 0.3 * (t - 1) / 0.5) ** (-1 / 0.3)
        assert abs(float(np.mean(x > t)) - p) < 4 * math.sqrt(p * (1 - p) / x.size), t


def test_lognormal_und_exponential_im_band():
    z = np.log(K.draw("lognormal", 20000, 8))
    assert abs(z.mean()) < 4 / math.sqrt(z.size) and abs(z.std(ddof=1) - 1.0) < 4 * math.sqrt(0.5 / z.size)
    e = K.draw("exponential", 20000, 8)
    assert abs(e.mean() - 1.0) < 4 / math.sqrt(e.size)


def test_abgeleitete_seeds_sind_verschieden_und_reproduzierbar():
    seeds = {K.derive_seed(7000, 1, r) for r in range(1000)}
    assert len(seeds) == 1000
    assert K.derive_seed(1, 2, 3) == K.derive_seed(1, 2, 3) and K.derive_seed(1, 2, 3) != K.derive_seed(1, 3, 2)


def test_wahre_pegel_der_faelle():
    assert K.true_level("splice10") == pytest.approx(15.6322, abs=5e-4)
    assert K.true_level("splice03") == pytest.approx(10.6911, abs=5e-4)
    assert K.true_level("lognormal") == pytest.approx(48.9398, abs=5e-4)
    assert K.true_level("exponential") == pytest.approx(-math.log(5e-5), rel=1e-15)
    assert K.true_level("vehicle") == pytest.approx(287.158, abs=1e-3)
    with pytest.raises(ValueError):
        K.true_level("unbekannt")
