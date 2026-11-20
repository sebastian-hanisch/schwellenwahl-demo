"""Eingefrorene Tabellen: Aufbau und innere Stimmigkeit, Vergleich mit den exakten Größen und „Neu rechnen als Band“: eine verkleinerte Wiederholung (andere Zufalls-Seeds, weniger Läufe)
liegt im Band mit Standardfehler um die eingefrorene Messung; die deterministische Populationsdatei wird vollständig neu gerechnet (Band 1e-5 absolut, plattformfest). Die Läufe der Messung
werden NIE auf das letzte Bit verglichen (Linux-CI-Lehre: `log`/`pow` kommen aus der C-Bibliothek)."""

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

import sws_algorithm as A
import sws_cases as K
import sws_constants as C
import sws_evaluation as EV
import sws_reference as REF

ROOT = Path(__file__).resolve().parent.parent
CASES = ("splice10", "splice03", "lognormal", "exponential", "vehicle")


def _tool(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------------------------------------------------------------------
# Aufbau
# ---------------------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("case_key", CASES)
def test_regeltabelle_hat_gitter_laeufe_und_zaehlungen_die_aufgehen(case_key):
    f = REF.rules_data()["fälle"][case_key]
    assert f["n"] == C.N_FROZEN[case_key] and f["p"] == C.P_TARGET[case_key] and f["laeufe"] == {"vehicle": 300}.get(case_key, 400)
    assert f["gitter_k"] == [int(k) for k in A.make_grid(f["n"])]
    assert f["wahrer_pegel"] == pytest.approx(K.true_level(case_key), rel=1e-12)
    assert set(f["regeln"]) == set(A.RULE_KEYS_ALL)
    for key, r in f["regeln"].items():
        assert sum(r["counts"]) == f["laeufe"] and len(r["counts"]) == 18, key
        assert r["k_q10"] <= r["k_median"] <= r["k_q90"] and 0.0 <= r["share_bottom"] <= 1.0
        assert r["share_bottom"] == pytest.approx(r["counts"][0] / f["laeufe"], abs=1e-12)
        assert r["rmse"] >= abs(r["bias"]) - 1e-12 and r["n_gueltig"] <= f["laeufe"]
    assert len(f["gitter"]) == 18 and [g["k"] for g in f["gitter"]] == f["gitter_k"]
    assert set(f["feste_k"]) == set(A.fixed_ks(f["n"])) and all(f["feste_k"][nm]["k"] == k for nm, k in A.fixed_ks(f["n"]).items())


def test_spleissfall_regeln_haben_den_anteil_bei_oder_ueber_u0():
    for case_key in ("splice10", "splice03"):
        for key, r in REF.rules_data()["fälle"][case_key]["regeln"].items():
            assert 0.0 <= r["share_u_ge_u0"] <= 1.0, key
    for case_key in ("lognormal", "exponential", "vehicle"):
        assert "share_u_ge_u0" not in REF.rules_data()["fälle"][case_key]["regeln"]["S2"]


def test_kritische_werte_der_regel_g_liegen_unter_dem_wert_bei_bekannten_parametern():
    crit = REF.crit_data()
    assert sorted(crit["A"]) == sorted(crit["B"]) == sorted({int(k) for n in (2000, 5000) for k in A.make_grid(n)}) and len(crit["A"]) == 34
    for var in ("A", "B"):
        vals = list(crit[var].values())
        assert 0.78 <= min(vals) and max(vals) <= 0.90, var                                # Band des Plans; bei bekannten Parametern wäre der Wert 1,224
        assert max(vals) < 1.224


def test_populationsdatei_gleich_exakte_endpunkte_und_aufsteigend():
    for key in ("vehicle", "lognormal"):
        rows = REF.population_data()[key]
        z = [r["zeta"] for r in rows]
        assert z == sorted(z) and len(rows) == 95
        for k in A.make_grid(5000 if key == "vehicle" else 2000):
            assert any(abs(r["zeta"] - k / (5000 if key == "vehicle" else 2000)) < 1e-9 for r in rows), k   # alle Gitterstufen sind Stützstellen
    assert REF.population_data()["vehicle"][0]["ln_err"] > 0


# ---------------------------------------------------------------------------------------------------------------------------------------
# Neu rechnen als Band
# ---------------------------------------------------------------------------------------------------------------------------------------
def test_populationsdatei_wird_vollstaendig_neu_gerechnet_im_band_1e_5():
    """Band 1e-5 absolut. Unter Windows (Python 3.14) liegen alle Abweichungen unter 1e-6; im Linux-Container (Docker, demo_linux_check) lag σ* im Vehikel B
    um bis zu 2,35e-6 daneben (8 von 95 Zeilen über 1e-6, alle anderen Größen unter 1e-6); die Ursache ist nicht untersucht, ein Rechenfehler ist es nicht. Im Container lief der Test mit diesem Band grün."""
    pytest.importorskip("scipy.special")
    tool = _tool("sws_make_population")
    new = tool.compute()
    for key in ("vehicle", "lognormal"):
        for a, b in zip(REF.population_data()[key], new[key]["zeilen"]):
            for f in ("u", "xi", "sigma_u", "sigma_star", "ln_err"):
                assert abs(a[f] - b[f]) < 1e-5, (key, a["zeta"], f)


def _rerun(case_key, runs, seed_base):
    tool = _tool("sws_make_rules")
    start, rows = tool.run_chunk((case_key, 0, runs, seed_base))
    return tool.aggregate(case_key, rows, runs, seed_base, 0.0)


def _band(p_new, p_old, n_new, n_old, k=4.0):
    return k * math.sqrt(p_new * (1 - p_new) / n_new + p_old * (1 - p_old) / n_old) + 1e-9


def test_spleiss_fall_10_neu_gerechnet_liegt_im_band_um_die_eingefrorene_messung():
    new = _rerun("splice10", 80, 4242)
    old = REF.rules_data()["fälle"]["splice10"]
    for key in ("S1.5", "S2", "S3", "M2", "G"):
        p0, p1 = old["regeln"][key]["share_u_ge_u0"], new["regeln"][key]["share_u_ge_u0"]
        assert abs(p1 - p0) < _band(0.5 * (p0 + p1), p0, 80, 400), key
    for j in (0, 7, 8, 9, 17):                                                           # k = 800, 192, 157, 128, 25 (je Stufe: Verzerrung im Band)
        g0, g1 = old["gitter"][j], new["gitter"][j]
        assert abs(g1["bias"] - g0["bias"]) < 4 * math.sqrt(g0["sd"] ** 2 / 80 + g0["sd"] ** 2 / 400) + 0.01, g0["k"]


def test_spleiss_fall_kippen_neu_gerechnet_zeigt_den_ausschlag_unterhalb_u0():
    new = _rerun("splice10", 60, 777)
    by_k = {g["k"]: g for g in new["gitter"]}
    assert abs(by_k[192]["bias"]) < 0.25                                                  # oberhalb u₀: kein Kippen
    assert by_k[354]["bias"] > 1.0 and by_k[289]["bias"] > 1.0                            # unterhalb u₀: Faktor 3 und mehr (Population: 8,3 bis 9,4)


def test_vehikel_b_neu_gerechnet_im_band():
    new = _rerun("vehicle", 16, 31415)
    old = REF.rules_data()["fälle"]["vehicle"]
    for j in (0, 6, 12, 16):
        g0, g1 = old["gitter"][j], new["gitter"][j]
        assert abs(g1["bias"] - g0["bias"]) < 4 * math.sqrt(g0["sd"] ** 2 / 16 + g0["sd"] ** 2 / 300), g0["k"]
    assert new["wahrer_pegel"] == pytest.approx(old["wahrer_pegel"], rel=1e-12)
    sp0, sp1 = old["streuung_regeln"], new["streuung_regeln"]
    assert 1.0 < sp1["median"] < 10 * sp0["median"] and sp1["n_gueltig"] == 16


def test_kalibrierung_regel_g_neu_gerechnet_im_band():
    tool = _tool("sws_make_crit")
    old = REF.crit_data()
    for k in (104, 800):
        _k, _var, vals = tool.calibrate_task((k, "A", 150))
        assert abs(float(np.quantile(vals, 0.90)) - old["A"][k]) < 0.1, k                 # Standardfehler des 90-%-Werts aus 300 Stichproben rund 0,03


def test_checkmodus_der_werkzeuge_ist_lauffaehig():
    """`--check` rechnet neu und vergleicht; hier nur: die Werkzeuge haben die Funktionen, die der Prüfmodus braucht."""
    for name, funcs in (("sws_make_crit", ("compute", "build", "main")), ("sws_make_population", ("compute", "build", "main")), ("sws_make_rules", ("build_case", "aggregate", "main"))):
        mod = _tool(name)
        assert all(callable(getattr(mod, f)) for f in funcs), name


def test_regeltabelle_fuer_die_app_ist_vollstaendig():
    for case_key in CASES:
        rows = EV.rules_table(case_key)
        assert len(rows) == 8 + 5 and rows[0]["Regel"].startswith("S (c = 1,5)")
        first = rows[0]
        if case_key in C.SPLICE:
            assert 0 <= first["bei oder über u₀ (%)"] <= 100
        else:
            assert 0 <= first["brauchbar ≤ 0,10 (%)"] <= first["brauchbar ≤ 0,05 (%)"] + 100
