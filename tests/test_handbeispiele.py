"""Handbeispiele des Plans als exakte Tests (Brüche, Ganzzahlen, Tafelwerte): jede Zahl ist von Hand nachgerechnet, der Erwartungswert steht als Rechnung im Test.

Spleiß-Fall: Körper Gleich(0, 1) mit Masse 9/10, ab u₀ = 1 exakt GPD(ξ = 3/10, σ = 1/2) mit Masse ζ₀ = 1/10 über u₀."""

import math
from fractions import Fraction as F

import numpy as np
import pytest

import sws_algorithm as A
import sws_cases as K
import sws_constants as C
import sws_population as PO
import sws_vehicle as V

ZETA0, U0, XI0, SIG0 = F(1, 10), F(1), F(3, 10), F(1, 2)


def e_exact(u):
    """e(u) = E[X − u | X > u] als Bruch: u ≥ 1: (σ + ξ(u − 1))/(1 − ξ); darunter mit a = 1 − u: (9/20 a² + a/10 + 1/14)/(9a/10 + 1/10)."""
    u = F(u)
    if u >= U0:
        return (SIG0 + XI0 * (u - U0)) / (1 - XI0)
    a = U0 - u
    return (F(9, 20) * a * a + a / 10 + F(1, 14)) / (F(9, 10) * a + F(1, 10))


# ---------------------------------------------------------------------------------------------------------------------------------------
# H 1: Mean Residual Life von Hand
# ---------------------------------------------------------------------------------------------------------------------------------------
def test_h1_mittlere_ueberschreitung_der_gpd_und_integral_der_ueberlebensfunktion():
    assert SIG0 / (1 - XI0) == F(5, 7)
    assert ZETA0 * SIG0 / (1 - XI0) == F(1, 14)


def test_h1_e_bei_0_halb_und_1_als_brueche():
    assert e_exact(0) == F(87, 140) == F(9, 10) * F(1, 2) + F(1, 10) * (1 + F(5, 7))   # e(0) = E[X] = 0,9 · 1/2 + 0,1 · (1 + 5/7)
    assert e_exact(F(1, 2)) == F(131, 308)                                              # Zähler 9/80 + 1/20 + 1/14 = 131/560, Nenner 11/20
    assert e_exact(1) == F(5, 7)


@pytest.mark.parametrize("u", [0.0, 0.5, 1.0, 2.0, 5.0])
def test_h1_geschlossene_form_der_app_gleich_dem_bruch(u):
    assert K.spliced_mrl(u, 0.1) == pytest.approx(float(e_exact(F(u).limit_denominator(1000))), rel=1e-12)


def test_h1_steigung_oberhalb_u0_ist_3_7_und_links_38_7():
    assert e_exact(2) == F(5, 7) + F(3, 7)
    assert e_exact(3) - e_exact(2) == XI0 / (1 - XI0) == F(3, 7)
    h_links, h_rechts = F(9, 10) / F(1, 10), (ZETA0 / SIG0) / F(1, 10)                  # Hazard = Dichte / S(u₀), S(u₀) = 1/10
    assert (h_links, h_rechts) == (9, 2)
    assert -1 + e_exact(1) * h_links == F(38, 7)                                        # e' = −1 + e·h
    assert -1 + e_exact(1) * h_rechts == F(3, 7)


def test_h1_dichtesprung_der_beobachtung_und_hazard_der_ueberschreitungen():
    assert ((1 - ZETA0) / U0, ZETA0 / SIG0) == (F(9, 10), F(1, 5))


def test_h1_differenzenquotient_links_und_rechts_von_u0():
    d = 1e-6
    f = lambda u: K.spliced_mrl(u, 0.1)  # noqa: E731
    assert (f(1.0) - f(1.0 - d)) / d == pytest.approx(38 / 7, abs=1e-4)
    assert (f(1.0 + d) - f(1.0)) / d == pytest.approx(3 / 7, abs=1e-8)


# ---------------------------------------------------------------------------------------------------------------------------------------
# H 2: Stabilität und Pegel
# ---------------------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("u", [F(1), F(3, 2), F(2), F(5)])
def test_h2_sigma_stern_ist_1_5_fuer_jede_schwelle_ab_u0(u):
    assert SIG0 + XI0 * (u - U0) - XI0 * u == F(1, 5)
    xi, sg, zu = PO.pop_fit_spliced(float(u), 0.1)
    assert xi == 0.3 and sg - 0.3 * float(u) == pytest.approx(0.2, abs=1e-12)


def test_h2_wahrer_pegel_15_632_und_gegenprobe():
    xp = K.spliced_level(5e-5, 0.1)
    assert xp == pytest.approx(1 + 0.5 / 0.3 * ((5e-5 / 0.1) ** (-0.3) - 1), rel=1e-15)
    assert xp == pytest.approx(15.6322, abs=5e-4)
    survival = 0.1 * (1 + 0.3 * (xp - 1) / 0.5) ** (-1 / 0.3)
    assert survival == pytest.approx(5e-5, rel=1e-9)


def test_h2_u0_liegt_bei_k_gleich_zeta0_mal_n():
    assert F(1, 10) * 2000 == 200 and F(3, 100) * 2000 == 60


# ---------------------------------------------------------------------------------------------------------------------------------------
# H 3: Populationsfehler (Tabelle der App gegen die Handwerte des Plans)
# ---------------------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("k,xi_ref,err_ref", [(289, 0.7752, 2.117), (235, 0.5002, 0.7639), (800, 0.2777, -0.2986)])
def test_h3_populationsfehler_gegen_die_tafel_des_plans(k, xi_ref, err_ref):
    row = PO.spliced_row(k, 2000, 0.1, 5e-5)
    assert row["xi"] == pytest.approx(xi_ref, abs=3e-3)
    assert row["ln_err"] == pytest.approx(err_ref, abs=0.02)


def test_h3_oberhalb_u0_ist_der_fehler_null():
    row = PO.spliced_row(150, 2000, 0.1, 5e-5)
    assert row["xi"] == 0.3 and row["ln_err"] == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------------------------------------------------------------------
# H 4: die Regeln S und M an Mini-Instanzen von Hand (Gitter absteigend, Index 0 = niedrigste Schwelle)
# ---------------------------------------------------------------------------------------------------------------------------------------
KK = np.array([800, 400, 200, 100, 50, 25, 12])


def test_h4_regel_s_waehlt_k_200():
    xis = np.array([0.80, 0.55, 0.32, 0.29, 0.31, 0.27, 0.35])
    se = np.maximum(1 + xis, 0.3) / np.sqrt(KK)
    assert abs(xis[1] - xis[0]) > 2 * se[1]                                             # Stufe 0 fällt durch: 0,25 > 2 · 1,55/20 = 0,155
    assert 2 * 1.32 / math.sqrt(200) == pytest.approx(0.1867, abs=1e-4) and abs(xis[2] - xis[1]) > 2 * se[2]   # Stufe 1: 0,23 > 0,187
    assert np.all(np.abs(xis[3:] - xis[2]) <= 2 * se[3:])                               # Stufe 2 besteht (Abstände 0,03 / 0,01 / 0,05 / 0,03)
    assert A.choose_S(xis, KK) == 2 and KK[A.choose_S(xis, KK)] == 200


def test_h4_regel_s_rueckfall_auf_die_hoechste_schwelle():
    assert A.choose_S(np.array([1.0, 0.6, 0.2, -0.2, -0.6, -1.0, -1.4]), KK) == len(KK) - 1      # lauter Widersprüche
    xis_late = np.array([1.2, 0.2, 0.9, -0.5, 0.3, 0.3, 0.3])                           # erst ab Stufe 4 stabil, dort nicht mehr prüfbar (nur drei Nachbarn nötig, J − 3 = 4 Stufen werden geprüft)
    assert A.choose_S(xis_late, KK) == len(KK) - 1


def test_h4_regel_s_plateau_bei_stufe_1_und_enges_band():
    xis_flat = np.array([0.8, 0.6, 0.5, 0.4, 0.3, 0.3, 0.3])                            # Stufe 0 fällt durch (0,2 > 2 · 1,6/20 = 0,16), Stufe 1 besteht
    assert A.choose_S(xis_flat, KK) == 1
    xis = np.array([0.80, 0.55, 0.32, 0.29, 0.31, 0.27, 0.35])
    assert A.choose_S(xis, KK, c=0.1) == len(KK) - 1                                    # mit c = 0,1 gibt es keinen Treffer


def test_h4_standardfehler_der_hoeheren_schwelle_nicht_der_geprueften():
    xis = np.array([0.80, 0.65, 0.62, 0.60, 0.60, 0.60, 0.60])
    # Abstände 0,15 / 0,18 / 0,20 / 0,20 / 0,20 / 0,20 gegen Bänder 2 (1 + ξ̂)/√k der Nachbarn 0,165 / 0,229 / 0,32 / 0,45 / 0,64 / 0,92: bestanden
    assert A.choose_S(xis, KK) == 0
    assert 2 * 1.8 / math.sqrt(800) == pytest.approx(0.1273, abs=1e-4)                  # mit dem Standardfehler der geprüften Schwelle (0,127 < 0,15) fiele Stufe 0 durch


def test_h4_regel_m_waehlt_stufe_2_gerade_und_rueckfall():
    us = np.arange(8.0)
    es = np.array([4.0, 2.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5])                             # e(u) = 1 + u/2 für u ≥ 2, davor Ausreißer
    ses = np.full(8, 0.1)
    assert np.allclose(es[2:], 1 + us[2:] / 2)
    assert A.choose_M(us, es, ses) == 2
    assert A.choose_M(us, np.array([5.0, 1.0, 6.0, 2.0, 7.0, 3.0, 8.0, 4.0]), ses) == len(us) - 5   # nichts passt: Rückfall auf J − 5
    assert A.choose_M(us, 1 + us / 2, ses) == 0                                          # exakt linear: niedrigste Schwelle


def test_h4_regel_m_waehlt_mit_dem_exakten_e_des_spleiss_falls_genau_den_knick():
    us = np.array([float(U0 + F(i, 4)) for i in range(-3, 5)])                           # u = 0,25 … 2,0
    es = np.array([float(e_exact(F(u).limit_denominator(1000))) for u in us])
    assert es[3] == pytest.approx(5 / 7, abs=1e-12) and (es[7] - es[3]) / (us[7] - us[3]) == pytest.approx(3 / 7, abs=1e-12)
    assert A.choose_M(us, es, np.full(len(us), 0.01)) == 3                               # Index 3 = u₀ = 1


# ---------------------------------------------------------------------------------------------------------------------------------------
# H 5: Regel G und Faustregeln
# ---------------------------------------------------------------------------------------------------------------------------------------
def test_h5_ks_abstand_von_hand():
    d = A.ks_distance(np.array([1.0, 2.0]), 0.0, 1.0)                                    # F(1) = 1 − 1/e = 0,6321 gibt D = max(0,6321, 0,3647, 0,1353)
    assert d == pytest.approx(1 - math.exp(-1), abs=1e-15)


@pytest.mark.parametrize("n,k_sqrt,k_23,k10", [(2000, 45, 78, 200), (5000, 71, 137, 500)])
def test_h5_faustregeln(n, k_sqrt, k_23, k10):
    assert round(math.sqrt(n)) == k_sqrt and round(n ** (2 / 3) / math.log(math.log(n))) == k_23 and round(0.1 * n) == k10
    f = A.fixed_ks(n)
    assert (f["k = √n"], f["k = n^(2/3)/ln ln n"], f["obere 10 %"]) == (k_sqrt, k_23, k10)
    assert f["obere 2 %"] == round(0.02 * n) and f["obere 5 %"] == round(0.05 * n)


def test_h5_gitter_ist_logarithmisch_absteigend_mit_18_stufen():
    g2, g5 = A.make_grid(2000), A.make_grid(5000)
    assert len(g2) == len(g5) == 18 and (g2[0], g2[-1]) == (800, 25) and (g5[0], g5[-1]) == (1000, 25)
    assert list(g2) == sorted(g2, reverse=True)
    assert list(g2[7:12]) == [192, 157, 128, 104, 85]                                    # die Stufen um u₀ = k 200 des Spleiß-Falls (V14: k ≤ 192 liegt oberhalb)


# ---------------------------------------------------------------------------------------------------------------------------------------
# H 6: Vehikel B, wahrer Pegel und Zielgrößen (Tabelle, exakt)
# ---------------------------------------------------------------------------------------------------------------------------------------
def test_h6_pegel_je_container_und_tagesmaximum_sind_verschiedene_zahlen():
    assert V.level(1e-5) == pytest.approx(287.158, abs=1e-3)                             # Zielgröße (a)
    assert V.daily_max_level(0.999) == pytest.approx(287.123, abs=1e-3)                  # Zielgröße (b)
    assert V.daily_max_level(0.99) == pytest.approx(155.128, abs=1e-3)
    assert V.P_TAGESMAX == pytest.approx(1.0005e-5, rel=2e-4)


def test_h6_mittel_median_und_ueberschreitung_von_30_tagen():
    mean, median = V.mean_and_median()
    assert mean == pytest.approx(5.646, abs=5e-4) and median == pytest.approx(4.166, abs=5e-4)
    assert V.survival(30.0) == pytest.approx(0.009830, abs=1e-6)


def test_h6_populationsfehler_bei_k_25_und_522_von_5000():
    import sws_reference as REF
    assert REF.population_interp("vehicle", 25 / 5000) == pytest.approx(0.065, abs=0.012)
    assert REF.population_interp("vehicle", 522 / 5000) == pytest.approx(0.812, abs=0.012)


def test_konstanten_der_fallparameter():
    assert (C.U0, C.XI0, C.SIG0) == (1.0, 0.3, 0.5) and C.SPLICE == {"splice10": 0.10, "splice03": 0.03}
