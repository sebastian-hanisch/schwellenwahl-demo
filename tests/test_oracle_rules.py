"""Orakel Regeln S, M, G: eine zweite, bewusst einfache Umsetzung aus der Definition (reine Python-Schleifen, gewichtete Gerade über die Normalgleichungen statt `numpy.polyfit`)
gegen die Funktionen der App auf je 300 Zufallsinstanzen mit festem Zufalls-Seed; dazu die Rückfälle und die Reihe bis zum Pegel (Tabelle je Schwelle gegen Einzelanpassung)."""

import math

import numpy as np
import pytest

import sws_algorithm as A
import sws_cases as K


def s_ref(xis, ks, c):
    """Regel S aus der Definition: niedrigste Schwelle j mit mindestens drei höheren Nachbarn, bei der alle höheren h im Band c·max(1+ξ̂_h, 0,3)/√k_h liegen."""
    j_n = len(xis)
    for j in range(j_n):
        if j_n - 1 - j < 3:
            break
        if all(abs(xis[h] - xis[j]) <= c * max(1.0 + xis[h], 0.3) / math.sqrt(ks[h]) for h in range(j + 1, j_n)):
            return j
    return j_n - 1


def wls_line(u, e, w):
    """Gewichtete Gerade e = a + b·u über die Normalgleichungen; w sind die Gewichte der kleinsten Quadrate (Summe w·Residuum²), `numpy.polyfit(w=…)` quadriert sein w."""
    sw = sum(w)
    mu = sum(wi * ui for wi, ui in zip(w, u)) / sw
    me = sum(wi * ei for wi, ei in zip(w, e)) / sw
    sxx = sum(wi * (ui - mu) ** 2 for wi, ui in zip(w, u))
    b = sum(wi * (ui - mu) * (ei - me) for wi, ui, ei in zip(w, u, e)) / sxx
    return me - b * mu, b


def m_ref(us, es, ses, c):
    j_n = len(us)
    for j in range(j_n - 4):
        a, b = wls_line(us[j:], es[j:], [1.0 / s ** 2 for s in ses[j:]])
        if all(abs(e - (a + b * u)) <= c * s for u, e, s in zip(us[j:], es[j:], ses[j:])):
            return j
    return max(j_n - 5, 0)


def g_ref(dists, crit):
    for j, (d, q) in enumerate(zip(dists, crit)):
        if math.isfinite(d) and d <= q:
            return j
    return len(dists) - 1


def _random_instance(rng, j_n=18):
    ks = np.unique(np.round(np.geomspace(800, 25, j_n)).astype(int))[::-1]
    xis = 0.3 + np.cumsum(rng.normal(0, 0.06, j_n)) * rng.choice([0.0, 0.5, 1.0])      # Plateaus und Drift
    us = np.sort(rng.uniform(0.5, 3.0, j_n))
    es = 0.5 + 0.4 * us + rng.normal(0, 0.05, j_n) * rng.choice([0.2, 1.0, 3.0])
    ses = rng.uniform(0.02, 0.08, j_n)
    return ks, xis, us, es, ses


@pytest.mark.parametrize("c", [1.5, 2.0, 3.0])
def test_regel_s_gleich_referenz_auf_300_instanzen(c):
    rng = np.random.default_rng(11)
    for _ in range(300):
        ks, xis, *_ = _random_instance(rng)
        assert A.choose_S(xis, ks, c) == s_ref(list(xis), list(ks), c)


@pytest.mark.parametrize("c", [1.5, 2.0, 3.0])
def test_regel_m_gleich_referenz_auf_300_instanzen(c):
    rng = np.random.default_rng(12)
    for _ in range(300):
        _ks, _xis, us, es, ses = _random_instance(rng)
        assert A.choose_M(us, es, ses, c) == m_ref(list(us), list(es), list(ses), c)


def test_regel_g_gleich_referenz_auf_300_instanzen():
    rng = np.random.default_rng(13)
    crit = np.full(18, 0.85)
    for _ in range(300):
        d = rng.uniform(0.5, 1.4, 18)
        d[rng.integers(0, 18, 3)] = np.nan
        assert A.choose_G(d, crit) == g_ref(list(d), list(crit))


def test_rueckfaelle_bei_wenigen_stufen():
    xis = np.array([0.3, 0.9, 0.1])
    assert A.choose_S(xis, np.array([100, 50, 25])) == 2                                  # weniger als vier Stufen: höchste Schwelle
    us, es, ses = np.array([1.0, 2.0, 3.0]), np.array([1.0, 1.5, 2.0]), np.array([0.1, 0.1, 0.1])
    assert A.choose_M(us, es, ses) == 0                                                   # weniger als fünf Stufen: Rückfall auf Stufe 0
    assert A.choose_G(np.array([2.0, 2.0]), np.array([0.8, 0.8])) == 1                    # kein Test besteht: höchste Schwelle


def test_regel_m_mit_stufe_ohne_standardfehler_wird_uebersprungen():
    us = np.arange(8.0)
    es = 1 + us / 2
    ses = np.full(8, 0.1)
    ses[0] = 0.0                                                                          # alle Werte gleich: Standardfehler 0, keine Gewichte
    assert A.choose_M(us, es, ses) == 1


def test_tabelle_je_schwelle_gleich_einzelanpassung():
    x = K.draw("splice10", 2000, 77)
    ks = A.make_grid(2000)
    t = A.threshold_stats(x, ks, 5e-5)
    xs = np.sort(x)
    for j in (0, 7, 17):
        k = int(ks[j])
        u = xs[2000 - k - 1]
        y = xs[2000 - k:] - u
        xi, sg, _ = A.gpd_ml(y)
        assert t["u"][j] == u and t["xi"][j] == pytest.approx(xi, abs=1e-12) and t["sigma"][j] == pytest.approx(sg, rel=1e-12)
        assert t["e"][j] == pytest.approx(float(np.mean(y)), rel=1e-12) and t["se"][j] == pytest.approx(float(np.std(y, ddof=1)) / math.sqrt(k), rel=1e-12)
        assert t["level"][j] == pytest.approx(u + sg / xi * ((k / 2000 / 5e-5) ** xi - 1), rel=1e-12)
        assert t["D"][j] == pytest.approx(A.ks_distance(y, xi, sg) * math.sqrt(k), rel=1e-12)


def test_analyze_liefert_alle_regeln_und_feste_k():
    x = K.draw("exponential", 2000, 3)
    ks = A.make_grid(2000)
    res = A.analyze(x, 5e-5, ks, np.full(18, 0.85), np.full(18, 0.84))
    assert set(res["picks"]) == set(A.RULE_KEYS_ALL) == {"S1.5", "S2", "S3", "M1.5", "M2", "M3", "G", "G2"}
    assert set(res["fixed"]) == {"obere 2 %", "obere 5 %", "obere 10 %", "k = √n", "k = n^(2/3)/ln ln n"}
    for name, (k, level) in res["fixed"].items():
        assert level == pytest.approx(A.fixed_level(np.sort(x), k, 5e-5)) and math.isfinite(level)


def test_crit_for_interpoliert_linear_in_ln_k_und_trifft_die_stuetzstellen():
    table = {25: 0.80, 100: 0.84, 400: 0.88}
    assert list(A.crit_for([25, 100, 400], table)) == [0.80, 0.84, 0.88]
    assert A.crit_for([50], table)[0] == pytest.approx(0.82, abs=1e-12)                  # ln 50 liegt in der Mitte von ln 25 und ln 100
