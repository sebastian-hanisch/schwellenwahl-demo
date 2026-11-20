"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Zahl im Text und jede Schwelle der Art steht hier; Fehlermaße und Aggregation an Mini-Instanzen von Hand."""

import math

import numpy as np
import pytest

import sws_constants as C
import sws_evaluation as EV

# ---------------------------------------------------------------------------------------------------------------------------------------
# Formatierung
# ---------------------------------------------------------------------------------------------------------------------------------------
def test_zahlen_mit_komma_und_echtem_minus():
    assert EV.de(1234.5, 1) == "1234,5" and EV.de(-1.5, 1) == "−1,5" and EV.de(-1.5, 1, ascii_minus=True) == "-1,5"
    assert EV.pct(0.517) == "51,7" and EV.pct(0.5, 0) == "50" and EV._signed(2.2418) == "+2,24" and EV._signed(-0.2986) == "−0,30"
    assert EV.band_label("1.5") == "1,5" and EV.band_label("2") == "2" and EV.rule_label("S1.5") == "S (c = 1,5)" and EV.rule_label("M3") == "M (c = 3)"
    assert EV.rule_label("G") == "G" and "ξ ∈ {0; 0,5}" in EV.rule_label("G2")


# ---------------------------------------------------------------------------------------------------------------------------------------
# Fehlermaße und Aggregation von Hand
# ---------------------------------------------------------------------------------------------------------------------------------------
def test_error_stats_von_hand_inklusive_grenzen_bei_faktor_2_und_halb():
    s = EV.error_stats([2.0, 4.0, 1.0, 8.0], 2.0)                                       # Verhältnisse 1, 2, 0,5, 4
    e = np.log([1.0, 2.0, 0.5, 4.0])
    assert s["bias"] == pytest.approx(e.mean(), abs=1e-15) and s["bias"] == pytest.approx(math.log(2) / 2, abs=1e-15)
    assert s["rmse"] == pytest.approx(math.sqrt((0 + math.log(2) ** 2 + math.log(2) ** 2 + math.log(4) ** 2) / 4), abs=1e-15)
    assert s["median_ratio"] == 1.5 and s["share_gt2"] == 0.25 and s["share_lt05"] == 0.0     # genau 2 zählt nicht als über 2, genau 0,5 nicht als unter 0,5
    assert s["share_below"] == 0.25 and s["n_gueltig"] == 4


def test_error_stats_ignoriert_ungueltige_laeufe_und_meldet_leere_menge():
    s = EV.error_stats([2.0, float("nan"), -1.0, 0.0, 4.0], 2.0)
    assert s["n_gueltig"] == 2 and s["median_ratio"] == 1.5
    leer = EV.error_stats([float("nan")], 2.0)
    assert leer["n_gueltig"] == 0 and math.isnan(leer["rmse"])


def test_standardfehler_des_anteils_und_des_rmse():
    assert EV.proportion_se(0.5, 400) == pytest.approx(0.025, abs=1e-12)                # Standardfehler eines Anteils um 50 % bei R = 400
    assert EV.rmse_standard_error(0.5, 50) == pytest.approx(0.05, abs=1e-12)


def test_aggregate_rule_zaehlt_stufen_und_u0():
    levels = np.array([[1.0, 2.0, 4.0], [2.0, 2.0, 2.0], [8.0, 4.0, 1.0], [2.0, 8.0, 2.0]])
    us = np.array([[0.5, 1.0, 1.5], [0.9, 1.0, 1.1], [0.2, 0.9, 1.2], [1.0, 1.0, 1.0]])
    r = EV.aggregate_rule(levels, [0, 1, 2, 0], us, [300, 200, 100], 2.0, u0=1.0)
    assert r["counts"] == [2, 1, 1] and r["share_bottom"] == 0.5
    assert r["k_median"] == 250.0 and r["k_q10"] == pytest.approx(np.quantile([300, 200, 100, 300], 0.1))      # gewählte k: 300, 200, 100, 300
    # gewählte Pegel: 1, 2, 1, 2 gegen wahr 2: Verhältnisse 0,5, 1, 0,5, 1
    assert r["share_lt05"] == 0.0 and r["median_ratio"] == 0.75
    assert r["share_u_ge_u0"] == 0.75                                                     # gewählte Schwellen 0,5 / 1,0 / 1,2 / 1,0: genau u₀ zählt als „bei oder über“, also 3 von 4


def test_aggregate_rule_u0_inklusive():
    levels = np.full((4, 2), 2.0)
    us = np.array([[0.5, 1.0], [0.9, 1.0], [1.0, 1.0], [0.99, 1.2]])
    r = EV.aggregate_rule(levels, [0, 0, 0, 1], us, [200, 100], 2.0, u0=1.0)              # gewählt: 0,5 / 0,9 / 1,0 / 1,2
    assert r["share_u_ge_u0"] == 0.5


def test_usable_stages_ab_der_gewaehlten_stufe_aufwaerts():
    pe = [0.9, 0.04, 0.2, 0.03, 0.01]                                                     # Index 0 = niedrigste Schwelle
    assert list(EV.usable_stages(pe, 0.05)) == [False, False, False, True, True]          # Stufe 1 selbst ist gut, aber Stufe 2 darüber nicht
    assert list(EV.usable_stages([-0.05, 0.05], 0.05)) == [True, True]                    # Betrag, Grenze inklusive
    assert EV.usable_share([3, 1, 4, 2], [False, False, True, True]) == 0.6


def test_streuung_zwischen_den_regeln_von_hand():
    sp = EV.spread_stats(np.array([[1.0, 2.0, 4.0], [3.0, 3.0, 3.0], [1.0, 1.0, 2.5], [2.0, 9.0, 2.0]]), 2.5)
    assert sp["n_gueltig"] == 4 and sp["median"] == pytest.approx(np.median([4.0, 1.0, 2.5, 4.5]))
    assert sp["share_gt2"] == 0.75 and sp["share_gt4"] == 0.25 and sp["max"] == 4.5       # 4 zählt nicht als über 4
    assert sp["share_truth_inside"] == 0.75                                               # [1,4] ja, [3,3] nein, [1,2,5] ja, [2,9] ja
    sp2 = EV.spread_stats(np.array([[1.0, float("nan")], [1.0, 2.0]]), 1.5)
    assert sp2["n_gueltig"] == 1 and sp2["n_laeufe"] == 2


def test_vorzeichenwechsel_der_verzerrung():
    ks = [100, 200, 300, 400, 500]
    assert EV.sign_change([0.5, 0.3, 0.05, -0.3, -0.4], ks) == (200, 400)                  # die Stufe bei 300 liegt unter 0,1 und wird übersprungen
    assert EV.sign_change([0.5, 0.3, 0.2], ks[:3]) is None and EV.sign_change([-0.3, 0.3], [800, 652]) == (652, 800)


# ---------------------------------------------------------------------------------------------------------------------------------------
# Verdikte zu Experiment 1 und 2 (Populationszeilen von Hand)
# ---------------------------------------------------------------------------------------------------------------------------------------
ROWS10 = [{"k": 192, "xi": 0.3, "sigma_star": 0.2}, {"k": 235, "xi": 0.5002, "sigma_star": -0.1478}, {"k": 289, "xi": 0.7752, "sigma_star": -0.529},
          {"k": 25, "xi": 0.3, "sigma_star": 0.2}]


def test_verdikt_stabilitaet_spleiss_fall_nennt_k_und_werte():
    kind, text = EV.verdict_stability("splice10", 2000, ROWS10)
    assert kind == "info"
    assert "k ≤ 192 von 200" in text and "0,50 (k = 235)" in text and "0,78 (k = 289)" in text and "σ* von 0,2 auf −0,53" in text and "Steigung 3/7" in text


def test_verdikt_stabilitaet_grenze_u0_ist_inklusive():
    rows = [{"k": 200, "xi": 0.3, "sigma_star": 0.2}, {"k": 210, "xi": 0.6, "sigma_star": -0.3}, {"k": 250, "xi": 0.7, "sigma_star": -0.4}]
    assert "k ≤ 200 von 200" in EV.verdict_stability("splice10", 2000, rows)[1]           # k = 200 = ζ₀·n liegt noch bei u₀
    assert "das Gitter reicht nicht" in EV.verdict_stability("splice10", 2000, rows[:2])[1]


def test_verdikt_stabilitaet_exponential_meldet_keinen_effekt_und_glatte_faelle_nennen_zahlen():
    kind, text = EV.verdict_stability("exponential", 2000, ROWS10)
    assert "ξ = 0 und σ* = 1" in text and "nichts zu wählen" in text and "Sprung" not in text
    rows = [{"k": 25, "xi": 0.225, "sigma_star": 0.98}, {"k": 100, "xi": 0.26, "sigma_star": 1.2}, {"k": 800, "xi": 0.291, "sigma_star": 1.55}]
    text = EV.verdict_stability("lognormal", 2000, rows)[1]
    assert "0,225 (k = 25)" in text and "0,291 (k = 800)" in text and "zwischen 0,98 und 1,55" in text and "größte Sprung von ξ_Pop zwischen zwei Stufen ist 0,035" in text


POP = [{"k": 25, "ln_err": 0.0}, {"k": 192, "ln_err": 0.0}, {"k": 235, "ln_err": 0.7639}, {"k": 289, "ln_err": 2.117}, {"k": 354, "ln_err": 2.2418}, {"k": 652, "ln_err": 0.193}, {"k": 800, "ln_err": -0.2986}]
RUNS = [{"k": 25, "bias": -0.17, "rmse": 0.65}, {"k": 192, "bias": 0.01, "rmse": 0.40}, {"k": 235, "bias": 0.82, "rmse": 1.05}, {"k": 289, "bias": 2.07, "rmse": 2.18},
        {"k": 354, "bias": 2.19, "rmse": 2.25}, {"k": 652, "bias": 0.17, "rmse": 0.34}, {"k": 800, "bias": -0.32, "rmse": 0.39}]


def test_verdikt_fehler_spleiss_fall_vorzeichenwechsel_und_scheinbares_minimum():
    kind, text = EV.verdict_error("splice10", 2000, POP, RUNS)
    assert kind == "info"
    assert "k ≤ 192 ist der Populationsfehler null" in text and "k = 354: ln +2,24 (Faktor 9,4)" in text
    assert "zwischen k = 652 und k = 800" in text and "(0,34, k = 652)" in text and "scheinbares Minimum" in text


def test_verdikt_fehler_ohne_vorzeichenwechsel_nennt_nur_den_kleinsten_rmse():
    runs = [dict(r, bias=abs(r["bias"]) + 0.2) for r in RUNS]
    text = EV.verdict_error("splice10", 2000, POP, runs)[1]
    assert "scheinbares Minimum" not in text and "Kleinster RMSE der festen k: 0,34 bei k = 652" in text


def test_verdikt_fehler_glatte_faelle():
    pop = [{"k": 25, "ln_err": 0.065}, {"k": 176, "ln_err": 0.264}, {"k": 648, "ln_err": 0.877}, {"k": 1000, "ln_err": 0.762}]
    runs = [{"k": 25, "bias": -0.09, "rmse": 0.73}, {"k": 176, "bias": 0.25, "rmse": 0.50}, {"k": 648, "bias": 0.86, "rmse": 0.92}, {"k": 1000, "bias": 0.75, "rmse": 0.81}]
    text = EV.verdict_error("vehicle", 5000, pop, runs)[1]
    assert "kein Kippen" in text and "+0,065 bei k = 25" in text and "+0,88 bei k = 648" in text and "größter Sprung zwischen zwei Stufen 0,61" in text
    assert "Minimum 0,50 bei k = 176" in text
    text = EV.verdict_error("exponential", 2000, [{"k": 25, "ln_err": 0.0}, {"k": 800, "ln_err": 0.0}], [{"k": 25, "bias": 0, "rmse": 0.26}, {"k": 800, "bias": 0, "rmse": 0.11}])[1]
    assert "bei jedem k null" in text and "von 0,26 (k = 25) auf 0,11 (k = 800)" in text and "keine Kippschwelle" in text


# ---------------------------------------------------------------------------------------------------------------------------------------
# Verdikte zu Experiment 3 und 4 (Messung als Hand-Wörterbuch)
# ---------------------------------------------------------------------------------------------------------------------------------------
def _rule(rmse, bottom=0.0, u0=None, median=1.0, k=100):
    r = {"rmse": rmse, "share_bottom": bottom, "median_ratio": median, "k_median": k, "k_q10": k, "k_q90": k, "counts": [0] * 18}
    if u0 is not None:
        r["share_u_ge_u0"] = u0
    return r


def _fake(case, n, rules, fixed, spread=None, R=400):
    return {"n": n, "laeufe": R, "p": 5e-5, "gitter_k": [800] + [0] * 17, "regeln": rules, "feste_k": fixed,
            "streuung_regeln": spread or {"median": 2.66, "q10": 1.74, "q90": 5.51, "max": 12.8, "share_gt2": 0.8, "share_gt4": 0.24, "share_truth_inside": 0.637, "n_laeufe": R, "n_gueltig": R}}


def _fixed(**kw):
    base = {"obere 2 %": 0.52, "obere 5 %": 0.5, "obere 10 %": 0.84, "k = √n": 0.54, "k = n^(2/3)/ln ln n": 0.51}
    base.update(kw)
    return {name: {"k": k, "rmse": base[name], "median_ratio": 1.1} for name, k in (("obere 2 %", 100), ("obere 5 %", 250), ("obere 10 %", 500), ("k = √n", 71), ("k = n^(2/3)/ln ln n", 137))}


def _rules_splice():
    return {"S2": _rule(0.552, u0=0.517), "M2": _rule(1.428, 0.36, u0=0.008), "G": _rule(0.41, u0=0.742), "G2": _rule(0.411, u0=0.742)}


def test_verdikt_regeln_spleiss_fall_nennt_anteile_und_kalibrierung(monkeypatch):
    monkeypatch.setattr(EV, "frozen", lambda ck: _fake(ck, 2000, _rules_splice(), _fixed()))
    kind, text = EV.verdict_rules("splice10", "2")
    assert "S (c = 2) wählt in 51,7 % der Läufe eine Schwelle bei oder über u₀, M (c = 2) in 0,8 %, G in 74,2 %" in text and "RMSE 0,55 / 1,43 / 0,41" in text
    assert "ξ ∈ {0; 0,3}" in text and "mit ξ ∈ {0; 0,5} wählt sie in 74,2 %" in text and "R = 400 Läufe" in text


def test_verdikt_regeln_glatter_fall_nennt_niedrigste_stufe_und_brauchbar(monkeypatch):
    rules = {"S2": _rule(0.66, 0.40), "M2": _rule(0.807, 0.89), "G": _rule(0.831, 0.82)}
    fake = _fake("vehicle", 5000, rules, _fixed(**{"obere 2 %": 0.533}), R=300)
    fake["gitter_k"] = [1000] + [0] * 17
    monkeypatch.setattr(EV, "frozen", lambda ck: fake)
    monkeypatch.setattr(EV, "usable_shares", lambda ck, key, tolerances=C.BRAUCHBAR_TOLERANZEN: {0.05: 0.0, 0.10: {"S2": 0.107, "M2": 0.0, "G": 0.0}[key]})
    text = EV.verdict_rules("vehicle", "2")[1]
    assert "M (c = 2) und G wählen in 89 % bzw. 82 % der Läufe die niedrigste Stufe (k = 1000 von 5000), S (c = 2) in 40 %" in text
    assert "RMSE 0,66 (S), 0,81 (M) und 0,83 (G) gegen 0,53 bei festem k = 100" in text and "11 % (S), 0 % (M) und 0 % (G)" in text and "höchstens 0,10" in text


def test_hauptfalle_niedrigste_schwelle_art_und_zaehleinheit(monkeypatch):
    fake = _fake("vehicle", 5000, {"S2": _rule(0.66, 0.4), "M2": _rule(0.8, 0.89), "G": _rule(0.83, 0.82)}, _fixed(), R=300)
    fake["gitter_k"] = [1000] + [0] * 17
    monkeypatch.setattr(EV, "frozen", lambda ck: fake)
    kind, text = EV.verdict_bottom("vehicle", "2")
    assert kind == "warning" and "für diese Regeln" in text and "89 % bzw. 82 % der 300 Läufe" in text and "k = 1000 von n = 5000" in text
    fake["regeln"]["M2"], fake["regeln"]["G"] = _rule(0.8, 0.49), _rule(0.8, 0.1)
    assert EV.verdict_bottom("vehicle", "2")[0] == "info"                                # unter 50 % in beiden Regeln: keine Warnung


def test_vergleich_rmse_grenze_bei_zwei_standardfehlern():
    assert EV._compare_rmse(0.39, 0.5, 50) == "besser" and EV._compare_rmse(0.41, 0.5, 50) == "im Rauschen"          # Standardfehler 0,05, zwei davon 0,10
    assert EV._compare_rmse(0.65, 0.5, 50) == "schlechter" and EV._compare_rmse(0.60, 0.5, 50) == "im Rauschen"   # der größere RMSE bestimmt den Standardfehler


def test_verdikt_gegen_feste_k_keine_regel_schlaegt_und_k_500(monkeypatch):
    rules = {"S1.5": _rule(0.627), "S2": _rule(0.66), "M2": _rule(0.807), "G": _rule(0.831)}
    monkeypatch.setattr(EV, "frozen", lambda ck: _fake(ck, 5000, rules, _fixed(**{"obere 10 %": 0.836}), R=300))
    text = EV.verdict_vs_fixed("vehicle")[1]
    assert "keine der 4 Regel-Varianten hat einen um mehr als zwei Standardfehler kleineren RMSE als das beste der festen k = 71 bis 250 (RMSE 0,50 bis 0,54" in text
    assert "S (c = 1,5), S (c = 2) um mehr als zwei Standardfehler besser" in text and "im Rauschen gleich: M (c = 2), G." in text


def test_verdikt_gegen_feste_k_eine_regel_ist_besser(monkeypatch):
    rules = {"S2": _rule(0.20), "G": _rule(0.5)}
    monkeypatch.setattr(EV, "frozen", lambda ck: _fake(ck, 2000, rules, _fixed(), R=400))
    text = EV.verdict_vs_fixed("exponential")[1]
    assert text.count("S (c = 2) hat einen um mehr als zwei Standardfehler kleineren RMSE") == 1 and "(RMSE 0,50)" in text


def test_verdikt_gegen_feste_k_spleiss_fall_vergleicht_g_mit_dem_besten_festen_k(monkeypatch):
    monkeypatch.setattr(EV, "frozen", lambda ck: _fake(ck, 2000, _rules_splice(), _fixed(**{"obere 10 %": 0.436}), R=400))
    text = EV.verdict_vs_fixed("splice10")[1]
    assert "G (RMSE 0,41) gegen das beste der fünf festen k (k = 500, obere 10 %, RMSE 0,44): im Rauschen" in text and "per Konstruktion" in text


def test_verdikt_festes_k_ist_riskant_nennt_beide_anteile():
    f10 = {"feste_k": {"obere 5 %": {"k": 100, "rmse": 0.455, "median_ratio": 0.92}}}
    f03 = {"feste_k": {"obere 5 %": {"k": 100, "rmse": 6.04, "median_ratio": 364.0}}}
    kind, text = EV.verdict_fixed_risk(f10, f03)
    assert kind == "warning" and "bei Anteil 10 % den RMSE 0,46" in text and "bei Anteil 3 % den RMSE 6,04" in text and "Median des Verhältnisses von 364" in text


def test_verdikt_streuung_der_regeln(monkeypatch):
    monkeypatch.setattr(EV, "frozen", lambda ck: _fake(ck, 5000, {}, _fixed(), R=300))
    kind, text = EV.verdict_spread("vehicle", live_ratio=3.0)
    assert "im Median beim 2,66-Fachen" in text and "Q10/Q90 1,74/5,51; Maximum 12,8" in text and "80,0 % der 300 Läufe über Faktor 2" in text
    assert "24,0 % über Faktor 4" in text and "63,7 % der Läufe" in text and "Die Reihe oben: höchster zu niedrigstem Pegel 3,00." in text
    assert "Die Reihe oben" not in EV.verdict_spread("vehicle")[1]


# ---------------------------------------------------------------------------------------------------------------------------------------
# Verdikt der Reihe oben (echte Reihe, Erwartung aus den Feldern der Reihe gerechnet)
# ---------------------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("rule", C.RULES)
def test_verdikt_der_reihe_nennt_gewaehltes_k_und_faktor(rule):
    s = EV.run_series(EV.Settings(C.CASE_EXPONENTIAL, 1000, 3, 2.0, rule))
    key, j = EV.picked_rule(s)
    lev = s.result["stats"]["level"][j]
    kind, text = EV.verdict_live(s)
    assert kind == "info" and f"k = {int(s.ks[j])} (Stufe {j + 1} von {len(s.ks)})" in text and f"das {EV.de(lev / s.truth)}-Fache" in text and EV.rule_label(key) in text


def test_verdikt_der_reihe_meldet_probleme_statt_wahl():
    s = EV.run_series(EV.Settings(C.CASE_EXPONENTIAL, 1000, 3, 2.0, C.RULES[0]))
    import dataclasses
    kind, text = EV.verdict_live(dataclasses.replace(s, problem="Weniger als 100 Beobachtungen: keine Schwellenwahl."))
    assert kind == "warning" and text.startswith("Weniger als 100")


def test_ein_preset_ohne_effekt_meldet_keinen_effekt():
    """Preset „Exponential“: Populationsfehler null bei jedem k, das Verdikt nennt keine Kippschwelle."""
    import sws_algorithm as A
    n = 2000
    rows = EV.population_rows("exponential", n, A.make_grid(n))
    assert all(r["ln_err"] == 0.0 and r["xi"] == 0.0 and r["sigma_star"] == 1.0 for r in rows)
    text = EV.verdict_error("exponential", n, rows, EV.frozen("exponential")["gitter"])[1]
    assert "keine Kippschwelle" in text and "Faktor" not in text
    assert "nichts zu wählen" in EV.verdict_stability("exponential", n, rows)[1]


def test_ueberschreitungswahrscheinlichkeit_lesbar_mit_hochzahl():
    assert EV.p_text(5e-5) == "5·10⁻⁵" and EV.p_text(1e-5) == "10⁻⁵" and EV.p_text(2.5e-4) == "2,5·10⁻⁴"
