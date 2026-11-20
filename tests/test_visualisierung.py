"""Diagramme: jede Abbildung hat gesperrte Achsen (fixedrange), logarithmische umgekehrte k-Achse und die erwarteten Spuren; Daten kommen unverändert an."""

import numpy as np
import plotly.graph_objects as go

import sws_evaluation as EV
import sws_visualization as VZ


def _locked(fig):
    assert isinstance(fig, go.Figure)
    axes = [fig.layout[k] for k in fig.layout if k.startswith("xaxis") or k.startswith("yaxis")]
    assert axes and all(a.fixedrange for a in axes)


def test_stabilitaetsdiagramm_mit_und_ohne_band():
    ks = np.array([800, 400, 200])
    fig = VZ.build_stability(ks, [0.8, 0.5, 0.3], [0.1, 0.1, 0.1], [200, 400, 800], [0.3, 0.5, 0.78], "ξ", chosen_k=200, u0_k=200, band=2.0)
    _locked(fig)
    assert fig.layout.xaxis.type == "log" and fig.layout.xaxis.autorange == "reversed"
    assert [t.name for t in fig.data] == ["± 2 Standardfehler", "Reihe: ML-Schätzung", "Population (ohne Rauschen)"]
    assert list(fig.data[1].y) == [0.8, 0.5, 0.3] and list(fig.data[2].x) == [200, 400, 800]
    band = fig.data[0]
    assert max(band.y) == 0.8 + 0.2 and min(band.y) == 0.3 - 0.2                          # ± c · Standardfehler um die Schätzung
    ohne = VZ.build_stability(ks, [0.8, 0.5, 0.3], None, None, None, "σ*")
    _locked(ohne)
    assert [t.name for t in ohne.data] == ["Reihe: ML-Schätzung"]


def test_mean_residual_life_diagramm_sortiert_nach_schwelle():
    fig = VZ.build_mrl([3.0, 1.0, 2.0], [1.5, 1.0, 1.2], [0.1, 0.1, 0.1], [1.0, 3.0], [1.0, 1.5], 2.0, chosen_u=2.0, u0=1.0)
    _locked(fig)
    assert list(fig.data[1].x) == [1.0, 2.0, 3.0] and list(fig.data[1].y) == [1.0, 1.2, 1.5] and fig.data[2].name == "exakt e(u)"


def test_exakte_familien_haben_je_fall_zwei_linien():
    fig = VZ.build_exact_families(EV.exact_families())
    _locked(fig)
    assert len(fig.data) == 6 and fig.layout.xaxis.type == "log"


def test_fehlerkurve_und_rmse_kurve():
    ks = [25, 100, 800]
    fig = VZ.build_error_curve(ks, [0.0, 0.5, -0.3], [0.0, 0.4, -0.2], [0.8, 1.0, 0.9], [1.3, 2.0, 1.1], None, u0_k=200)
    _locked(fig)
    assert fig.data[0].name == "Läufe: Q10 bis Q90" and fig.data[-1].name == "Population (ohne Rauschen)"
    assert np.allclose(fig.data[0].y[:3], np.log([1.3, 2.0, 1.1]))
    _locked(VZ.build_error_curve(ks, [0, 0, 0], None, None, None, None))
    rm = VZ.build_rmse_curve(ks, [0.4, 0.3, 0.5], sign_change_between=(100, 800))
    _locked(rm)
    assert len(rm.layout.shapes) == 1


def test_histogramm_und_sieben_regeln():
    h = VZ.build_choice_hist([800, 400, 200], [5, 3, 2], "S (c = 2)", "#4c78a8", u0_k=200)
    _locked(h)
    assert list(h.data[0].y) == [5, 3, 2] and list(h.data[0].x) == ["800", "400", "200"]
    rows = [("S (c = 2)", 100, 20.0), ("M (c = 2)", 800, 50.0), ("G", 200, 30.0), ("fest obere 2 %", 40, 25.0)]
    fig = VZ.build_seven_rules(rows, 28.0)
    _locked(fig)
    assert list(fig.data[0].y) == [20.0, 50.0, 30.0, 25.0] and fig.layout.yaxis.type == "log"
