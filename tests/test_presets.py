"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, Permalink (Begrenzen und Einrasten); jedes Preset hat eine eingefrorene Messung."""

import pytest

import sws_constants as C
import sws_evaluation as EV
import sws_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_regler_bounds():
    for name, preset in C.PRESETS.items():
        assert preset["case"] in C.CASES and preset["rule"] in C.RULES and preset["band"] in C.BANDS, name
        assert C.N_MIN <= preset["n"] <= C.N_MAX and (preset["n"] - C.N_MIN) % C.N_STEP == 0 and 0 <= preset["seed"] <= C.SEED_MAX, name


def test_default_preset_equals_the_default_settings():
    first = next(iter(C.PRESETS.values()))
    assert EV.Settings(**first) == EV.Settings()
    assert (first["case"], first["n"], first["seed"], first["band"], first["rule"]) == (C.DEFAULT_CASE, C.DEFAULT_N, C.DEFAULT_SEED, C.DEFAULT_BAND, C.DEFAULT_RULE)


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_presets_unterscheiden_sich_voneinander():
    tuples = [tuple(sorted(p.items())) for p in C.PRESETS.values()]
    assert len(set(tuples)) == len(tuples)


def test_jeder_fall_hat_eine_eingefrorene_messung_und_ein_preset_oder_einen_regler():
    assert set(C.CASE_KEYS.values()) == set(C.P_TARGET) == set(C.N_FROZEN)
    for case in C.CASES:
        assert EV.frozen(C.CASE_KEYS[case])["n"] == C.N_FROZEN[C.CASE_KEYS[case]]
    assert {p["case"] for p in C.PRESETS.values()} == set(C.CASES)                        # jeder Fall hat mindestens ein Preset


def test_preset_namen_und_ihre_werte_passen_zusammen():
    """Vertauschte Beschriftungen (Playbook §11 Punkt 5): „10 %“ und „3 %“, „Lognormal“, „Exponential“, „Vehikel B“ gegen den Fall im Preset."""
    p = C.PRESETS
    assert p["Spleiß-Fall 10 %: Kippen bei k = 200"]["case"] == C.CASE_SPLICE10 and p["Spleiß-Fall 3 %: schärferer Ausschlag"]["case"] == C.CASE_SPLICE03
    assert p["Lognormal: weich"]["case"] == C.CASE_LOGNORMAL and p["Exponential: nichts kippt"]["case"] == C.CASE_EXPONENTIAL
    assert p["Verweilzeit-Mischung (Vehikel B)"]["case"] == C.CASE_VEHICLE and p["Verweilzeit-Mischung (Vehikel B)"]["n"] == 5000
    assert p["Dieselbe Reihe, sieben Regeln"]["case"] == C.CASE_VEHICLE and C.SPLICE["splice10"] * 2000 == 200 and C.SPLICE["splice03"] * 2000 == 60


@pytest.mark.parametrize("key,raw,expected", [
    ("n_slider", 100, 1000), ("n_slider", 99999, 5000), ("n_slider", 1260, 1500), ("n_slider", 1240, 1000), ("seed_input", -5, 0), ("seed_input", 10 ** 9, C.SEED_MAX),
    ("case_select", "gibt es nicht", C.DEFAULT_CASE), ("case_select", C.CASE_LOGNORMAL, C.CASE_LOGNORMAL), ("band_select", 7.0, C.DEFAULT_BAND), ("band_select", 1.5, 1.5),
    ("rule_select", "x", C.DEFAULT_RULE),
])
def test_permalink_wert_wird_begrenzt_und_eingerastet(key, raw, expected):
    assert P.clamp_value(P.SETTING_SPECS[key], raw) == expected
