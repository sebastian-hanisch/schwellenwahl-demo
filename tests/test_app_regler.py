"""Oberfläche (AppTest): jede Option jedes Auswahlreglers, Ränder der Zahlenregler, Permalink, Presets mit Verdikten, Expander und Diagramme. Kein echter Mausklick auf Plotly
(AppTest löst keine Auswahl aus); die Hauptfunktionen der Diagramme sind in `test_visualisierung.py` geprüft."""

import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import sws_constants as C

APP = str(Path(__file__).resolve().parent.parent / "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))


def _app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _texts(at):
    return [e.value for e in list(at.info) + list(at.warning) + list(at.success)]


def test_erste_seite_hat_charts_tabellen_und_verdikte():
    at = _app()
    assert len(at.get("plotly_chart")) == 8                                               # ξ, σ*, e(u), Familien, Fehler, RMSE, Histogramm, sieben Regeln
    assert len(at.dataframe) == 2 and len(at.metric) == 4
    texts = _texts(at)
    assert any("oberhalb u₀ (k ≤ 192 von 200)" in t for t in texts) and any("Die Regel wählt die niedrigste Schwelle" in t or "R = 400 Läufe" in t for t in texts)
    assert any("Die Verweilzeiten und der Spleiß-Fall sind ausgedacht" in t for t in texts)


@pytest.mark.parametrize("case", C.CASES)
def test_jeder_fall_laeuft_ohne_ausnahme(case):
    at = _app()
    at.selectbox(key="case_select").set_value(case).run()
    assert not at.exception, (case, [str(e) for e in at.exception])
    assert len(at.get("plotly_chart")) == 8 and len(_texts(at)) >= 6


@pytest.mark.parametrize("rule", C.RULES)
def test_jede_regel_laeuft_ohne_ausnahme_und_die_marke_folgt(rule):
    at = _app()
    at.selectbox(key="rule_select").set_value(rule).run()
    assert not at.exception, [str(e) for e in at.exception]
    erwartet = {"S": "Regel S (c = 2)", "M": "Regel M (c = 2)", "G": "Regel G wählt"}[C.RULE_KEYS[rule]]
    assert any(erwartet in t for t in _texts(at)) and len(at.metric) == 4


@pytest.mark.parametrize("band", C.BANDS)
def test_jedes_band_laeuft(band):
    at = _app()
    at.select_slider(key="band_select").set_value(band).run()
    assert not at.exception and any(f"c = {str(band).replace('.', ',').rstrip('0').rstrip(',')})" in t for t in _texts(at))


@pytest.mark.parametrize("n", [C.N_MIN, C.N_MAX])
def test_n_am_rand_fuer_jeden_fall(n):
    for case in (C.CASE_SPLICE03, C.CASE_VEHICLE):
        at = _app()
        at.selectbox(key="case_select").set_value(case).run()
        at.slider(key="n_slider").set_value(n).run()
        assert not at.exception, (case, n, [str(e) for e in at.exception])


@pytest.mark.parametrize("seed", [0, C.SEED_MAX])
def test_zufalls_seed_am_rand(seed):
    at = _app()
    at.number_input(key="seed_input").set_value(seed).run()
    assert not at.exception


def test_neue_reihe_ziehen_aendert_den_zufalls_seed():
    at = _app()
    before = at.number_input(key="seed_input").value
    next(b for b in at.sidebar.button if b.label.startswith("🎲")).click().run()
    assert not at.exception
    assert at.number_input(key="seed_input").value != before                          # Zufallszahl im Bereich 0 bis 999 999: Gleichheit hätte Wahrscheinlichkeit 10⁻⁶


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_preset_setzt_die_regler_und_zeigt_verdikte(name):
    at = _app()
    at.sidebar.button(key=f"preset_{name}").click().run()
    assert not at.exception, (name, [str(e) for e in at.exception])
    p = C.PRESETS[name]
    assert at.selectbox(key="case_select").value == p["case"] and at.slider(key="n_slider").value == p["n"] and at.number_input(key="seed_input").value == p["seed"]
    assert at.select_slider(key="band_select").value == p["band"] and at.selectbox(key="rule_select").value == p["rule"]
    assert len(_texts(at)) >= 6


def test_permalink_aus_der_adresszeile_wird_uebernommen_und_begrenzt():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.query_params["case"] = C.CASE_LOGNORMAL
    at.query_params["n"] = "99999"
    at.query_params["band"] = "3.0"
    at.query_params["seed"] = "-4"
    at.query_params["rule"] = "unbekannt"
    at.run()
    assert not at.exception
    assert at.selectbox(key="case_select").value == C.CASE_LOGNORMAL and at.slider(key="n_slider").value == C.N_MAX
    assert at.select_slider(key="band_select").value == 3.0 and at.number_input(key="seed_input").value == 0 and at.selectbox(key="rule_select").value == C.DEFAULT_RULE


def test_expander_per_titel_nicht_per_index():
    at = _app()
    titel = {e.label for e in at.expander}
    assert "📐 Mathematische Formulierung" in titel and "So funktioniert die Schwellenwahl" in titel
    math = next(e for e in at.expander if e.label == "📐 Mathematische Formulierung")
    text = " ".join(m.value for m in math.markdown)
    assert "**Lesehilfe.**" in text and "| Symbol | Bedeutung |" in text and "$$" in text and "ForwardStop" not in text


def test_metrik_delta_ist_mein_wert_minus_referenz_mit_ascii_minus():
    at = _app()
    wahr = float(at.metric[0].value.replace(",", "."))
    dach = float(at.metric[1].value.replace(",", "."))
    delta = at.metric[1].delta
    assert delta.endswith("gegen wahr") and "−" not in delta                                # ASCII-Minus (Streamlit färbt ein Unicode-Minus falsch)
    assert float(delta.split()[0].replace(",", ".")) == pytest.approx(dach - wahr, abs=0.011)


def test_keine_roten_fehler_im_ganzen_lauf_alle_vier_experimente_stehen_da():
    at = _app()
    sub = [s.value for s in at.subheader]
    assert [s for s in sub if s.startswith("🔬 Experiment")] == [
        "🔬 Experiment 1: Zwei Diagramme, eine Reihe – woran erkennt man eine zu tiefe Schwelle?", "🔬 Experiment 2: Wo kippt der Pegel?",
        "🔬 Experiment 3: Regeln wählen – findet eine Regel die Schwelle?", "🔬 Experiment 4: Dieselbe Reihe, sieben Regeln"]
    assert "🚧 Wo die Annahmen enden" in sub
