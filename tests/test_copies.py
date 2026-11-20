"""Kopier-Tests: die kopierten Dateien sind unverändert (SHA-256), und wo die Quelle vorhanden ist, stimmen sie mit ihr überein.

* `sws_rng.py` = `tools/templates/rng_splitmix64_exact.py` des Website-Repos (reine Vorlage, nicht ändern).
* `sws_mixture.json` = gemeinsame Gewichtstabelle der Linie (Vehikel B, 1 224 Komponenten), unverändert kopiert; Quelle und Stand je Feld in `sws_mixture_quelle.json`.
* Konstanten der Verweilzeit-Mischung und die Medianformel = `verweilzeit-demo` (nur lesend, nur im Test; ohne den Nachbarordner übersprungen)."""

import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
PROJECTS = ROOT.parent
SHA_RNG = "d69b2b2fd67c1234cc0d79249676a662a2189d02803eea7d2edfb04b7a505a3f"
SHA_MIXTURE = "669f821c78e0bf38c47fa846232f27cd2ff126f4fceadca5369a5c9af5551c9f"


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_rng_ist_die_unveraenderte_vorlage():
    assert _sha(ROOT / "sws_rng.py") == SHA_RNG
    vorlage = PROJECTS / "sebastianhanisch-website" / "tools" / "templates" / "rng_splitmix64_exact.py"
    if not vorlage.exists():
        pytest.skip("Vorlage nicht vorhanden (CI): nur der SHA-256-Vergleich oben gilt")
    assert (ROOT / "sws_rng.py").read_bytes() == vorlage.read_bytes()


def test_vehikel_b_tabelle_ist_unveraenderte_kopie_der_gemeinsamen_tabelle():
    assert _sha(ROOT / "sws_mixture.json") == SHA_MIXTURE
    d = json.loads((ROOT / "sws_mixture.json").read_text(encoding="utf-8"))
    assert d["n_komponenten"] == len(d["komponenten"]) == 1224
    assert d["summe_gewichte"] == pytest.approx(1.0, abs=1e-12)
    quelle = PROJECTS / "extremwert-planung" / "vormessung" / "ev_v13_vehikel_b_tabelle.json"
    if quelle.exists():
        assert quelle.read_bytes() == (ROOT / "sws_mixture.json").read_bytes()


def test_quelle_und_stand_der_tabelle_stehen_im_begleitblatt():
    q = json.loads((ROOT / "sws_mixture_quelle.json").read_text(encoding="utf-8"))
    assert q["sha256"] == SHA_MIXTURE and q["stand"] and q["quelle"] and q["felder"]
    for feld in ("beschreibung", "n_komponenten", "summe_gewichte", "komponenten"):
        assert feld in q["felder"] and q["felder"][feld]["quelle"] and q["felder"][feld]["stand"]


def test_eingefrorene_dateien_tragen_quelle_und_stand_je_feld():
    for name, felder in (("sws_gof_crit.json", ("A", "B", "r_cal")), ("sws_population.json", ("vehicle", "lognormal")), ("sws_rules.json", ("fälle",))):
        d = json.loads((ROOT / name).read_text(encoding="utf-8"))
        assert d["quelle"] and d["stand"] and d["beschreibung"], name
        for f in felder:
            assert d["felder"][f]["quelle"] and d["felder"][f]["stand"], (name, f)
            assert f in d, (name, f)


def _lade(name, ordner):
    spec = importlib.util.spec_from_file_location(name, ordner / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_konstanten_und_medianformel_gleich_verweilzeit_demo():
    ordner = PROJECTS / "verweilzeit-demo"
    if not (ordner / "vwz_generator.py").exists():
        pytest.skip("verweilzeit-demo nicht vorhanden (CI)")
    sys.path.insert(0, str(ordner))
    sys.dont_write_bytecode = True
    try:
        for name in ("vwz_stk_core", "vwz_constants", "vwz_generator"):
            sys.modules.pop(name, None)
        Cv = _lade("vwz_constants", ordner)
        try:
            Gv = _lade("vwz_generator", ordner)
        except ImportError:
            pytest.skip("verweilzeit-demo lässt sich hier nicht importieren")
    finally:
        sys.path.remove(str(ordner))
    from vehikel_b_def import ART_P, S_BASE, U_SCALE, UK, median
    assert (ART_P, S_BASE, U_SCALE) == (tuple(Cv.ART_P), tuple(Cv.S_BASE), tuple(Cv.U_SCALE))
    assert Cv.K_KUNDEN == 12 and Cv.U_SD == 0.30 and Cv.P_ZOLL == 0.25
    assert np.array_equal(np.array(UK), np.asarray(Gv.UK)), "Kundeneffekte weichen ab"
    rng = np.random.default_rng(7)
    art = rng.integers(0, 4, 400)
    kunde = rng.integers(0, 12, 400)
    zoll = ((rng.random(400) < 0.5) & (art == 0)).astype(int)
    we = rng.integers(0, 2, 400)
    vor = np.where(art == 1, rng.uniform(1, 10, 400), np.where(art == 2, rng.uniform(0.5, 6, 400), 0.0))
    mu, s = Gv.log_median_and_s(art, kunde, zoll, we, vor)
    for i in range(400):
        a = int(art[i])
        assert abs(math.log(median(a, int(zoll[i]), int(we[i]), float(vor[i]))) + UK[int(kunde[i])] * U_SCALE[a] - mu[i]) < 1e-12 and abs(S_BASE[a] - s[i]) < 1e-15
