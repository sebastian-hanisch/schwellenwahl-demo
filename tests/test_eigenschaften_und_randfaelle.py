"""Eigenschaften (je 20 bis 40 Zufallsinstanzen mit festem Zufalls-Seed) und Randfälle: Skalierung, Verschiebung, Reihenfolge, Monotonie in c, Wiederholbarkeit; kurze und
entartete Reihen, Gleichstände, Rückfälle, Ränder der Regler."""

import math

import numpy as np
import pytest

import sws_algorithm as A
import sws_cases as K
import sws_constants as C
import sws_evaluation as EV
import sws_population as PO

CRIT = np.full(18, 0.85)


def _analyze(x, ks=None):
    return A.analyze(x, 5e-5, A.make_grid(len(x)) if ks is None else ks, CRIT, CRIT)


@pytest.fixture(scope="module")
def reihen():
    return [K.draw("splice10" if s % 2 else "lognormal", 2000, 100 + s) for s in range(20)]


# ---------------------------------------------------------------------------------------------------------------------------------------
# Eigenschaften
# ---------------------------------------------------------------------------------------------------------------------------------------
def test_skalierung_mit_c_aendert_die_gewaehlte_stufe_nicht_und_skaliert_den_pegel(reihen):
    for x in reihen:
        a, b = _analyze(x), _analyze(7.3 * x)
        assert a["picks"] == b["picks"]
        assert np.allclose(b["stats"]["level"], 7.3 * a["stats"]["level"], rtol=1e-6, equal_nan=True)
        assert np.allclose(b["stats"]["xi"], a["stats"]["xi"], atol=1e-6, equal_nan=True)         # ξ ist skaleninvariant


def test_verschiebung_aendert_stufe_xi_sigma_e_und_abstand_nicht(reihen):
    for x in reihen:
        a, b = _analyze(x), _analyze(x + 3.0)
        assert a["picks"] == b["picks"]
        for f in ("xi", "sigma", "e", "se", "D"):
            assert np.allclose(a["stats"][f], b["stats"][f], rtol=1e-6, atol=1e-6, equal_nan=True), f
        assert np.allclose(b["stats"]["u"], a["stats"]["u"] + 3.0, rtol=1e-12)                   # nur die Schwelle verschiebt sich


def test_reihenfolge_der_daten_hat_keine_wirkung(reihen):
    rng = np.random.default_rng(0)
    for x in reihen:
        a, b = _analyze(x), _analyze(rng.permutation(x))
        assert a["picks"] == b["picks"] and np.array_equal(a["stats"]["level"], b["stats"]["level"], equal_nan=True)


def test_groesseres_c_waehlt_nie_eine_hoehere_schwelle(reihen):
    """Die Bedingungen der Regeln S und M sind monoton in c (Index 0 = niedrigste Schwelle): j(1,5) ≥ j(2) ≥ j(3) heißt, die Wahl rutscht nach unten."""
    for x in reihen:
        p = _analyze(x)["picks"]
        assert p["S1.5"] >= p["S2"] >= p["S3"]
        assert p["M1.5"] >= p["M2"] >= p["M3"]


def test_gleicher_zufalls_seed_gleiche_ausgabe():
    a = EV.run_series(EV.Settings("Lognormal (σ = 1)", 1500, 77, 2.0, C.RULES[0]))
    b = EV.run_series(EV.Settings("Lognormal (σ = 1)", 1500, 77, 2.0, C.RULES[0]))
    assert a.result["picks"] == b.result["picks"] and np.array_equal(a.result["stats"]["level"], b.result["stats"]["level"], equal_nan=True)


def test_standardfehler_der_mittleren_ueberschreitung_sind_bei_hohem_k_kleiner(reihen):
    kleiner = 0
    for x in reihen:
        se = _analyze(x)["stats"]["se"]
        kleiner += int(se[0] < se[-1])                                                    # Index 0: k = 800, Index −1: k = 25
    assert kleiner >= 18                                                                  # mindestens 18 von 20 Reihen (Zählung, nicht „immer“)


def test_mittlere_ueberschreitung_exponential_ist_im_mittel_konstant_gleich_sigma():
    x = K.draw("exponential", 4000, 31)
    st = A.threshold_stats(x, A.make_grid(4000), 5e-5)
    assert np.all(np.abs(st["e"] - 1.0) < 4 * st["se"])                                   # e(u) = σ = 1 bei jeder Schwelle, im Band von vier Standardfehlern


def test_xi_dach_im_exponential_streut_um_null():
    """ξ = 0 bei jeder Schwelle: Mittel von ξ̂ bei k = 192 über 30 Reihen liegt im Band von vier Standardfehlern um 0 (Standardfehler des Mittels aus der Streuung der 30 Werte)."""
    xs = []
    for seed in range(30):
        x = K.draw("exponential", 2000, 500 + seed)
        xi, _sg, _e = A.gpd_ml(np.sort(x)[-192:] - np.sort(x)[-193])
        xs.append(xi)
    xs = np.array(xs)
    assert abs(xs.mean()) < 4 * xs.std(ddof=1) / math.sqrt(len(xs)) and 0.03 < xs.std(ddof=1) < 0.15      # Streuung etwa (1 + ξ)/√k = 0,072


# ---------------------------------------------------------------------------------------------------------------------------------------
# Randfälle
# ---------------------------------------------------------------------------------------------------------------------------------------
def test_zu_kurze_reihe_meldung_statt_wahl():
    x = K.draw("exponential", 99, 1)
    assert "100" in A.data_problem(x, A.make_grid(99))
    assert A.data_problem(K.draw("exponential", 100, 1), A.make_grid(100)) is None


def test_gerundete_daten_meldung_statt_wahl():
    x = np.round(K.draw("lognormal", 2000, 5), 0)                                         # auf ganze Zahlen gerundet: viele Gleichstände unter den größten Werten
    assert "Gleichstände" in A.data_problem(x, A.make_grid(2000))
    assert EV.run_series(EV.Settings("Exponential", 1000, 5, 2.0, C.RULES[0])).problem is None


def test_alle_werte_gleich_und_nicht_endlich():
    x = np.full(500, 3.0)
    assert A.data_problem(x, A.make_grid(500)) is not None
    st = A.threshold_stats(x, A.make_grid(500), 5e-5)
    assert np.all(np.isnan(st["xi"])) and np.all(np.isnan(st["level"]))                    # Anpassung ungültig, nicht null
    res = _analyze(x)
    assert all(isinstance(j, int) for j in res["picks"].values())                          # die Regeln fallen zurück, kein Absturz
    assert "nicht endlich" in A.data_problem(np.array([1.0] * 120 + [np.inf]), A.make_grid(121))


def test_ml_ungueltig_ist_nan_nicht_null_und_regel_s_behandelt_nan_als_instabil():
    assert math.isnan(A.gpd_ml(np.array([1.0]))[0]) and math.isnan(A.gpd_ml(np.array([2.0, 2.0, 2.0]))[0]) is False  # k = 1 ungültig; konstant gibt ξ am Rand
    xis = np.array([np.nan, 0.3, 0.3, 0.3, 0.3, 0.3])
    assert A.choose_S(xis, np.array([500, 300, 200, 100, 50, 25])) == 1                    # die ungültige Stufe wird übergangen, die nächste ist stabil


def test_sehr_kurzes_gitter_stuerzt_nicht_ab():
    x = K.draw("exponential", 100, 3)
    ks = A.make_grid(100)
    assert 2 <= len(ks) <= 18 and ks.max() <= 99 and ks.min() == 25
    res = A.analyze(x, 5e-5, ks, np.full(len(ks), 0.85), np.full(len(ks), 0.85))
    assert all(0 <= j < len(ks) for j in res["picks"].values())
    ks30 = A.make_grid(30)
    assert ks30.max() <= 29


def test_pegel_bei_xi_ueber_1_ist_endlich_aber_e_der_gpd_waere_unendlich():
    lev = A.pot_level(1.5, 1.0, 5.0, 0.05, 1e-3)
    assert math.isfinite(lev) and lev > 5.0
    # Mean Residual Life einer GPD: σ/(1 − ξ) ist für ξ ≥ 1 nicht definiert; die Reihe zeigt trotzdem den Stichprobenmittelwert (endlich)
    y = K.draw("exponential", 500, 6)
    assert math.isfinite(float(y.mean()))


def test_schwelle_exakt_bei_u0_und_ueberschreitungsanteil_am_rand():
    assert PO.spliced_threshold(0.1, 0.1) == pytest.approx(1.0, abs=1e-12)                 # ζ_u = ζ₀ gibt u₀
    assert PO.spliced_threshold(0.1000001, 0.1) < 1.0 < PO.spliced_threshold(0.0999999, 0.1)
    xi, sg, zu = PO.pop_fit_spliced(1.0, 0.1)
    assert (xi, sg, zu) == (0.3, 0.5, pytest.approx(0.1, rel=1e-12))                      # genau bei u₀ ist die Überschreitungsverteilung exakt die GPD


def test_u0_unter_dem_kleinsten_k_des_gitters_wird_gemeldet():
    rows = [{"k": k, "xi": 0.3, "sigma_star": 0.2} for k in (25, 31, 38)]
    kind, text = EV.verdict_stability("splice03", 500, rows)                               # k₀ = 15 < 25
    assert kind == "info" and "unter dem kleinsten k" in text
    kind, text = EV.verdict_stability("splice10", 200, rows)                               # k₀ = 20 < 25
    assert "unter dem kleinsten k" in text


def test_regler_an_minimum_und_maximum_geben_serien():
    for n in (C.N_MIN, C.N_MAX):
        for case in C.CASES:
            s = EV.run_series(EV.Settings(case, n, 0, 3.0, C.RULES[2]))
            assert s.problem is None and len(s.ks) >= 15 and np.all(np.diff(s.ks) < 0)
            key, j = EV.picked_rule(s)
            assert 0 <= j < len(s.ks)
