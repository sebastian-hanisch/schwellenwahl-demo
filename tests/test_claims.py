"""Claims: jede Zahl der README hat eine Herkunft. Die Zahlen der Messungen stehen als Bausteine in `tests/claims_fragmente.py` (aus den eingefrorenen Tabellen und den deterministischen
Rechnungen berechnet); hier wird geprüft, dass jeder Baustein wörtlich in der README steht, dass die Tabelle „Zahl → Test“ nur vorhandene Tests nennt und dass App-Text und README
dieselben Texte haben."""

import ast
import re
from pathlib import Path

import pytest
from claims_fragmente import build

import sws_constants as C
import sws_evaluation as EV
import sws_texts as T

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")


def test_jeder_messbaustein_steht_woertlich_in_der_readme():
    fehlt = {k: v for k, v in build().items() if v not in README}
    assert not fehlt, f"in der README fehlen: {sorted(fehlt)}"


def test_zaehlungen_der_readme_sind_aus_den_tabellen_gerechnet():
    f = build()
    assert f["veh_le10"] == "k ≤ 39" and f["veh_le05"] == "kein Gitterpunkt" and f["crit_n"] == "34"
    assert f["fixed_ks_2000"] == "45 und 78" and f["fixed_ks_5000"] == "71 und 137"
    assert f["sp10_factor289"] == "8,3" and f["sp10_factor354"] == "9,4" and f["sp03_factor104"] == "568"
    assert (f["veh_a"], f["veh_b"], f["veh_b99"]) == ("287,158", "287,123", "155,128")
    assert f["grid_2000"] == "25, 31, 38, 46, 57, 69, 85, 104, 128, 157, 192, 235, 289, 354, 434, 532, 652, 800"


def _test_funktionen():
    namen = {}
    for p in (ROOT / "tests").glob("test_*.py"):
        tree = ast.parse(p.read_text(encoding="utf-8"))
        namen[p.name] = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    return namen


def test_zahl_test_tabelle_nennt_nur_vorhandene_tests():
    namen = _test_funktionen()
    refs = re.findall(r"`tests/(test_[a-z_]+\.py)::(test_[a-z0-9_]+)`", README)
    assert len(refs) >= 25, "die Tabelle „Zahl → Test“ ist zu kurz"
    for datei, name in refs:
        assert datei in namen and name in namen[datei], f"{datei}::{name} gibt es nicht"


def test_annahmen_und_vehikel_text_stehen_in_app_und_readme_gleich():
    for text in (T.ASSUMPTIONS.strip(), T.VEHIKEL_TEXT):
        assert text in README, text[:60]


def test_readme_ohne_verbotene_woerter_und_gedankenstriche_im_fliesstext():
    body = README.split("\n---\n")[0]
    ohne = re.sub(r"<!--.*?-->", "", body, flags=re.S)
    for wort in ("Saatwert", "Schwanz", "trägt", "tragen", "TODO", "geplant", "Phase 0", "läuft lokal", "Stück 5 von", "Zufallssaat"):
        assert wort not in ohne, wort
    ohne_code = re.sub(r"```.*?```", "", ohne, flags=re.S)
    assert " — " not in ohne_code and "—" not in ohne_code
    assert re.search(r"\bPareto\b(?!-Verteilung)", ohne_code) is None
    assert re.search(r"\bWeibull\b(?!-Typ)", ohne_code) is None
    assert "Seed" not in re.sub(r"Zufalls-Seed|Seed-Basis|Seed-Basen", "", ohne_code)          # nur „Zufalls-Seed“ (und die Seed-Basis der Messläufe)


def test_pflichtabschnitte_und_pflichtwoerter_der_readme():
    for kopf in ("## Warum dieses Problem", "## Befunde und Korrekturen", "## Modell", "## Methodik", "## Befunde (gemessen, keine Behauptungen)", "### Zahl → Test (Band oder exakt)",
                 "## Ehrliche Grenzen", "## 🚧 Bewusst nicht enthalten / nicht gemessen", "## Quellen", "## Tests", "## Dateistruktur", "## Bewusst nicht umgesetzt", "## Lokal ausführen"):
        assert kopf in README, kopf
    for muss in ("Zufalls-Seed", "Pareto-Verteilung", "Pegel", "Läufe", "Extremwertindex", "Weibull-Typ", "eigene Festlegung", "Tail", "nicht gemessen"):
        assert muss in README, muss
    assert README.index("## Warum dieses Problem") < README.index("## Befunde und Korrekturen") < README.index("## Modell") < README.index("## Ehrliche Grenzen") < README.index("## Quellen")


def test_readme_nennt_jedes_preset_und_die_quellen_mit_stufe():
    for name in C.PRESETS:
        assert name in README, name
    for quelle in ("Coles", "Embrechts", "Scarrott", "Belzile", "Bader", "Northrop", "Wadsworth", "Murphy", "Brink", "Langousis"):
        assert quelle in README, quelle
    assert "nicht gelesen" in README and "nach Zusammenfassung, ungeprüft" in README


def test_readme_nennt_die_sha256_der_kopierten_dateien_und_die_normen():
    assert "669f821c78e0bf38c47fa846232f27cd2ff126f4fceadca5369a5c9af5551c9f" in README and "d69b2b2fd67c1234cc0d79249676a662a2189d02803eea7d2edfb04b7a505a3f" in README
    assert "scipy.stats.genpareto" in README and "c = ξ" in README and "c = −ξ" in README


def test_keine_aussage_ueber_nutzen_in_logistik_lager_oder_hafen():
    ohne = re.sub(r"<!--.*?-->", "", README, flags=re.S)
    for satz in re.split(r"(?<=[.!?])\s+", ohne):
        if re.search(r"Logistik|Lager|Hafen", satz):
            assert re.search(r"nicht|keine|kein|ausgedacht|Lehrbuch|Abstract", satz), satz


def test_app_texte_ohne_verbotene_woerter():
    alle = " ".join([T.INTRO, T.HOW, T.KERNFRAGEN, T.EXP1_HELP, T.EXP2_HELP, T.EXP3_HELP, T.EXP4_HELP, T.VEHIKEL_TEXT, T.ASSUMPTIONS, T.MATH_LESEHILFE, T.MATH_BODY])
    for wort in ("Saatwert", "Schwanz", "trägt", "tragen", "TODO", "Phase 0"):
        assert wort not in alle, wort
    assert "—" not in alle and re.search(r"\bPareto\b(?!-Verteilung)", alle) is None


def test_verdikte_nennen_regeln_stets_als_eigene_festlegung_oder_mit_dem_zusatz_fuer_diese_regeln():
    assert "eigene Festlegungen" in EV.verdict_bottom("vehicle", "2")[1] and "für diese Regeln" in EV.verdict_bottom("vehicle", "2")[1]
    assert "eigene Regeln" in T.INTRO and "keine Literaturverfahren" in T.INTRO


@pytest.mark.parametrize("zahl", ["287,158", "287,123", "155,128"])
def test_vehikel_b_zielgroessen_stehen_in_readme_und_sind_verschieden(zahl):
    assert zahl in README
