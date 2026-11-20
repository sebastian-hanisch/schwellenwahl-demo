"""Orakel Vehikel B: die eingefrorene Gewichtstabelle gegen eine eigene Aufzählung aus der Definition (4 704 Schritte, 2 376 mit Gewicht, 1 224 Komponenten), der wahre Pegel gegen
scipy (Brent, logsumexp) und die Ziehung gegen die exakte Verteilung (Anteile je Art, Mittel und Quantile als Band mit Standardfehler)."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

import sws_cases as K
import sws_vehicle as V
from vehikel_b_def import aufzaehlung

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def tabelle():
    return json.loads((ROOT / "sws_mixture.json").read_text(encoding="utf-8"))["komponenten"]


def test_eigene_aufzaehlung_hat_4704_schritte_2376_mit_gewicht_und_1224_komponenten():
    schritte, mit_gewicht, summe = aufzaehlung()
    assert (schritte, mit_gewicht, len(summe)) == (4704, 2376, 1224)
    assert sum(summe.values()) == pytest.approx(1.0, abs=1e-12)


def test_tabelle_gleich_eigene_aufzaehlung_auf_1e_12(tabelle):
    _, _, summe = aufzaehlung()
    eigene = {(round(mu, 10), s): w for (mu, s), w in summe.items()}
    assert len(tabelle) == len(eigene) == 1224
    for w, mu, s in tabelle:
        key = (round(mu, 10), s)
        assert key in eigene, key
        assert w == pytest.approx(eigene[key], abs=1e-12)
    assert sum(w for w, _mu, _s in tabelle) == pytest.approx(1.0, abs=1e-12)


def test_komponenten_je_art_48_576_576_24(tabelle):
    por_art = {}
    for _w, _mu, s in tabelle:
        por_art[s] = por_art.get(s, 0) + 1
    assert por_art == {0.55: 48, 0.25: 576, 0.35: 576, 0.9: 24}


def test_wahrer_pegel_gegen_scipy_brent_und_logsumexp(tabelle):
    special = pytest.importorskip("scipy.special")
    optimize = pytest.importorskip("scipy.optimize")
    w, mu, s = (np.array(c) for c in zip(*tabelle))

    def lnsf(lx):
        return float(special.logsumexp(np.log(w) + special.log_ndtr(-(lx - mu) / s)))

    def pegel(p):
        return math.exp(optimize.brentq(lambda lx: lnsf(lx) - math.log(p), -2.0, 14.0, xtol=1e-13, rtol=1e-14))

    assert V.level(1e-5) == pytest.approx(pegel(1e-5), rel=1e-11) and V.level(1e-5) == pytest.approx(287.15797, abs=2e-5)
    assert V.daily_max_level(0.999) == pytest.approx(287.12294, abs=2e-5) and V.daily_max_level(0.99) == pytest.approx(155.12830, abs=2e-5)
    for p in (0.5, 0.1, 1e-3, 1e-7):
        assert V.level(p) == pytest.approx(pegel(p), rel=1e-10)
        assert V.survival(V.level(p)) == pytest.approx(p, rel=1e-9)


def test_tagesmaximum_gegenprobe_per_potenz():
    """F^100(x) = 0,999 bei x = 287,123 und der Pegel (a) liegt wegen der Gleichheit der beiden Zielgrößen nur 0,012 % darüber."""
    x = V.daily_max_level(0.999)
    assert (1.0 - V.survival(x)) ** 100 == pytest.approx(0.999, abs=1e-9)
    assert V.level(1e-5) / x - 1 == pytest.approx(1.2e-4, abs=2e-5)


def test_ziehung_anteile_mittel_und_ueberschreitungswahrscheinlichkeit_im_band():
    rng_seed = 12345
    x = np.array(K.draw("vehicle", 20000, rng_seed))
    mean_exact, median_exact = V.mean_and_median()
    se_mean = x.std(ddof=1) / math.sqrt(x.size)
    assert abs(x.mean() - mean_exact) < 4 * se_mean                                      # Mittel im Band von vier Standardfehlern
    assert abs(np.median(x) - median_exact) < 0.15                                       # Median: dichte Stelle, grobes Band
    share = np.mean(x > 30.0)
    p = V.survival(30.0)
    assert abs(share - p) < 4 * math.sqrt(p * (1 - p) / x.size)


def test_ziehung_ist_fuer_gleichen_zufalls_seed_gleich_und_fuer_anderen_verschieden():
    a, b, c = K.draw("vehicle", 300, 5), K.draw("vehicle", 300, 5), K.draw("vehicle", 300, 6)
    assert np.array_equal(a, b) and not np.array_equal(a, c)
