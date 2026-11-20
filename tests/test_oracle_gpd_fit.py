"""Orakel: GPD-Anpassung, Verteilungsfunktion, Pegel und KS-Abstand der App gegen scipy (nur im Test) und gegen die Definition.

Konvention: `scipy.stats.genpareto` c = ξ (anders als `genextreme`, dort c = −ξ). Die Referenz `genpareto.fit` ist ein Optimierer ohne Gradienten und nur auf etwa 1e-4 bis 1e-5
genau; darum wird zusätzlich die Log-Likelihood verglichen (die App darf nicht schlechter sein) und der Gradient am eigenen Ergebnis geprüft."""

import math

import numpy as np
import pytest

import sws_algorithm as A

stats = pytest.importorskip("scipy.stats")


def _loglik(y, xi, sg):
    return float(np.sum(stats.genpareto.logpdf(y, xi, scale=sg)))


@pytest.mark.parametrize("xi0", [-0.3, 0.0, 0.3, 0.8])
def test_ml_gleich_scipy_fit_auf_40_stichproben(xi0):
    for seed in range(40):
        y = stats.genpareto.rvs(xi0, scale=2.0, size=150, random_state=1000 * seed + 7)
        xi, sg, edge = A.gpd_ml(y)
        c, _loc, sc = stats.genpareto.fit(y, floc=0)
        assert not edge
        assert xi == pytest.approx(c, abs=3e-3) and sg == pytest.approx(sc, rel=3e-3), (xi0, seed)
        assert _loglik(y, xi, sg) >= _loglik(y, c, sc) - 1e-7            # die App findet mindestens so gute Likelihood


def test_gradient_am_eigenen_ergebnis_ist_null():
    """Referenz ohne Optimierer: Gradient der Log-Likelihood per zentraler Differenz am Ergebnis der App."""
    for seed in range(30):
        y = stats.genpareto.rvs(0.25, scale=1.5, size=400, random_state=seed)
        xi, sg, _ = A.gpd_ml(y)
        h = 1e-5
        d_xi = (_loglik(y, xi + h, sg) - _loglik(y, xi - h, sg)) / (2 * h)
        d_sg = (_loglik(y, xi, sg + h) - _loglik(y, xi, sg - h)) / (2 * h)
        assert abs(d_xi) < 0.05 and abs(d_sg) < 0.05, (seed, d_xi, d_sg)        # Skala der Likelihood: Summe über 400 Werte


def test_ml_am_rand_xi_minus_1_wird_gemeldet():
    for seed in (0, 2, 3):
        y = stats.uniform.rvs(size=12, random_state=seed)                       # kleine Gleichverteilungs-Stichprobe: die Likelihood wächst bis ξ = −1
        xi, sg, edge = A.gpd_ml(y)
        assert xi == pytest.approx(-1.0, abs=1e-6) and edge and isinstance(sg, float) and sg > 0


def test_verteilungsfunktion_gleich_scipy():
    y = np.linspace(0.0, 30.0, 61)
    for xi, sg in ((0.0, 1.0), (0.3, 0.5), (-0.4, 2.0), (0.8, 3.0), (1e-12, 1.0)):
        assert A.gpd_cdf(y, xi, sg) == pytest.approx(stats.genpareto.cdf(y, xi, scale=sg), abs=1e-12), (xi, sg)


@pytest.mark.parametrize("xi,sg,u,zeta,p", [(0.3, 0.5, 1.0, 0.1, 5e-5), (0.0, 1.0, 3.0, 0.05, 1e-3), (-0.3, 2.0, 5.0, 0.02, 1e-4), (0.5, 4.0, 10.0, 0.2, 1e-5)])
def test_pegel_gleich_quantil_der_gpd(xi, sg, u, zeta, p):
    expected = u + stats.genpareto.ppf(1.0 - p / zeta, xi, scale=sg)           # Quantil der Überschreitungen zum Anteil p/ζ
    assert A.pot_level(xi, sg, u, zeta, p) == pytest.approx(expected, rel=1e-10)


def test_pegel_bei_xi_nahe_null_ist_stetig():
    a = A.pot_level(1e-9, 1.0, 0.0, 0.1, 1e-3)
    b = A.pot_level(0.0, 1.0, 0.0, 0.1, 1e-3)
    assert a == pytest.approx(b, rel=1e-7) and b == pytest.approx(math.log(100.0), rel=1e-12)


def test_ks_abstand_gleich_scipy_kstest():
    rng = np.random.default_rng(5)
    for _ in range(30):
        y = rng.exponential(2.0, int(rng.integers(5, 80)))
        assert A.ks_distance(y, 0.0, 2.0) == pytest.approx(stats.kstest(y, stats.expon(scale=2.0).cdf).statistic, abs=1e-12)
        xi, sg = 0.4, 1.3
        assert A.ks_distance(y, xi, sg) == pytest.approx(stats.kstest(y, stats.genpareto(xi, scale=sg).cdf).statistic, abs=1e-12)


def test_kritische_werte_bei_bekannten_parametern():
    assert stats.kstwobign.ppf(0.95) == pytest.approx(1.3581, abs=1e-3) and stats.kstwobign.ppf(0.90) == pytest.approx(1.2238, abs=1e-3)
    d = A.ks_distance(np.array([1.0, 2.0]), 0.0, 1.0)
    assert d == pytest.approx(stats.kstest([1.0, 2.0], "expon").statistic, abs=1e-15)
