"""Schwellenwahl: wo ist die Schwelle hoch genug? - interaktive Konzept-Demo (Demo der Linie „Extremwerte erklärt“)
Sebastian Hanisch - Operations Research und Machine Learning

Lauffähig mit: streamlit run app.py
Aufbau nach DEMO-PLAYBOOK §2: fester Kopf, Schnellstart und Regler in der Seitenleiste, Kernfragen vor den Experimenten, Verdikte als reine Funktionen in sws_evaluation.py."""

import math

import numpy as np
import pandas as pd
import streamlit as st

import sws_constants as C
import sws_evaluation as EV
import sws_texts as T
from sws_layout import dunkles_design, kopf_css
from sws_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from sws_visualization import (
    build_choice_hist,
    build_error_curve,
    build_exact_families,
    build_mrl,
    build_rmse_curve,
    build_seven_rules,
    build_stability,
)

st.set_page_config(page_title="Schwellenwahl – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _series(settings):
    return EV.run_series(settings)


@st.cache_data(show_spinner=False)
def _population(case_key, n, ks):
    return EV.population_rows(case_key, n, list(ks))


@st.cache_data(show_spinner=False)
def _families():
    return EV.exact_families()


@st.cache_data(show_spinner=False)
def _mrl(case_key, u_lo, u_hi):
    return EV.exact_mrl_curve(case_key, u_lo, u_hi)


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in sws_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title(T.TITLE)
st.markdown(T.INTRO)

with st.expander(T.HOW_TITLE, expanded=True):
    st.markdown(T.HOW)

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Reihe**")
    case = st.selectbox("Fall", C.CASES, key="case_select", help="Welche Verteilung die Beobachtungen haben. Alle Fälle haben einen exakt bekannten wahren Pegel.")
    n = st.slider("Beobachtungen n", *bounds("n_slider"), step=C.N_STEP, key="n_slider",
                  help="Länge der Reihe oben (Experiment 1 und 4). Die eingefrorenen Messungen der Experimente 2 und 3 gelten für n = 2000 bzw. 5000.")
    seed = st.number_input("Zufalls-Seed der Reihe", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Reihe ziehen", width="stretch", on_click=randomize_seed)
    st.markdown("**Regel**")
    rule = st.selectbox("Regel", C.RULES, key="rule_select", help="Welche der drei eigenen Regeln die Marke „gewählt“ und das Ergebnis oben bestimmt.")
    band = st.select_slider("Band c (Standardfehler)", options=list(C.BANDS), key="band_select", format_func=lambda v: EV.de(v, 1),
                            help="Bandbreite der Regeln S und M und der Bänder in den Diagrammen; die Regel G hat kein Band.")

sync_query_params({k: v for k, v in (("case_select", case), ("n_slider", int(n)), ("seed_input", int(seed)), ("band_select", float(band)), ("rule_select", rule))})

settings = EV.Settings(case, int(n), int(seed), float(band), rule)
case_key = settings.key
band_key = f"{settings.band:g}"
rule_letter = C.RULE_KEYS[rule]

with st.spinner("Reihe wird gezogen und an 18 Schwellen angepasst (Vehikel B: etwa 1 bis 3 Sekunden) …"):
    series = _series(settings)
ks = series.ks
res = series.result
st_ = res["stats"]

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------
st.markdown("## 🎯 Ergebnis: eine Reihe, eine Regel")
if series.problem:
    st.warning(series.problem)
else:
    pick_key, pick_j = EV.picked_rule(series)
    level = float(st_["level"][pick_j])
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Wahrer Pegel", EV.de(series.truth), help=f"Wert, den eine Beobachtung mit der Wahrscheinlichkeit p = {EV.p_text(series.p)} überschreitet (exakt aus der Verteilung).")
    if math.isfinite(level):
        # Delta immer als „mein Wert minus Referenz“; die Farbe regelt delta_color="inverse" (Vorzeichen-Falle, DEMO-PLAYBOOK §2); ASCII-Minus, weil Streamlit sonst falsch einfärbt.
        m2.metric(f"Pegel-Dach, Regel {EV.rule_label(pick_key)}", EV.de(level), delta=f"{level - series.truth:+.2f}".replace(".", ",") + " gegen wahr", delta_color="inverse")
        m4.metric("Verhältnis Dach / wahr", EV.de(level / series.truth), delta=f"ln {math.log(level / series.truth):+.2f}".replace(".", ","), delta_color="off")
    else:
        m2.metric(f"Pegel-Dach, Regel {EV.rule_label(pick_key)}", "–", help="Die Anpassung liefert dort keinen endlichen Pegel.")
        m4.metric("Verhältnis Dach / wahr", "–")
    m3.metric("Gewähltes k", f"{int(ks[pick_j])}", delta=f"Stufe {pick_j + 1} von {len(ks)} (1 = niedrigste Schwelle)", delta_color="off")
    kind, text = EV.verdict_live(series)
    getattr(st, kind)(text)

# --- Kernfragen ----------------------------------------------------------------------------------------------------------------------------
st.markdown("## ❓ Kernfragen")
st.markdown(T.KERNFRAGEN)
st.markdown("---")

# --- Experiment 1 --------------------------------------------------------------------------------------------------------------------------
st.subheader("🔬 Experiment 1: Zwei Diagramme, eine Reihe – woran erkennt man eine zu tiefe Schwelle?")
st.caption(T.EXP1_HELP)
pop_rows = _population(case_key, settings.n, tuple(int(k) for k in ks))
zeta0 = C.SPLICE.get(case_key)
u0_k = zeta0 * settings.n if zeta0 else None
if not series.problem:
    xi_hat = st_["xi"]
    sstar_hat = st_["sigma"] - st_["xi"] * st_["u"]
    se_xi = np.maximum(1.0 + xi_hat, 0.3) / np.sqrt(ks)
    chosen_k = int(ks[EV.picked_rule(series)[1]])
    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(build_stability(ks, xi_hat, se_xi, [r["k"] for r in pop_rows], [r["xi"] for r in pop_rows], "ξ", chosen_k, u0_k, band=settings.band),
                        width="stretch", key="stab_xi")
    with col_b:
        st.plotly_chart(build_stability(ks, sstar_hat, None, [r["k"] for r in pop_rows], [r["sigma_star"] for r in pop_rows], "σ* = σ_u − ξ·u", chosen_k, u0_k),
                        width="stretch", key="stab_sigma")
    ex_u, ex_e = _mrl(case_key, float(np.min(st_["u"])), float(np.max(st_["u"])))
    st.plotly_chart(build_mrl(st_["u"], st_["e"], st_["se"], ex_u, ex_e, settings.band, chosen_u=float(st_["u"][EV.picked_rule(series)[1]]), u0=C.U0 if zeta0 else None),
                    width="stretch", key="mrl_chart")
kind, text = EV.verdict_stability(case_key, settings.n, pop_rows)
getattr(st, kind)(text)
with st.expander("Was beim Anheben der Schwelle bleibt: die exakten Fälle (ξ und σ* gegen ζ_u)"):
    st.markdown("Bei **Exponential**, der **Pareto-Verteilung (α = 2)** und der **Gleichverteilung** ist die Überschreitungsverteilung bei jeder Schwelle exakt eine GPD: ξ (0, 1/2, −1) und σ* (1, 0, 1) bleiben, "
                "σ_u = σ* + ξ·u wächst nur bei ξ > 0, bleibt bei ξ = 0 und fällt bei ξ < 0. Der Spleiß-Fall zeigt dagegen oben, was eine **gestörte** Variante tut.")
    st.plotly_chart(build_exact_families(_families()), width="stretch", key="families_chart")
with st.expander("Tabelle der Reihe (eine Zeile je Stufe)"):
    table = pd.DataFrame(EV.series_table(series)).rename(columns={"k": "k", "u": "Schwelle u", "xi": "ξ̂", "sigma_star": "σ̂*", "e": "e(u)", "level": "Pegel-Dach"})
    st.dataframe(table, hide_index=True, width="stretch")

st.markdown("---")

# --- Experiment 2 --------------------------------------------------------------------------------------------------------------------------
st.subheader("🔬 Experiment 2: Wo kippt der Pegel?")
fz = EV.frozen(case_key)
st.caption(T.EXP2_HELP + f" Eingefrorene Läufe: n = {fz['n']}, R = {fz['laeufe']} Läufe, p = {EV.p_text(fz['p'])} je Beobachtung; der Regler n wirkt hier nicht.")
fks = fz["gitter_k"]
pop_frozen = _population(case_key, fz["n"], tuple(fks))
grid_rows = sorted(fz["gitter"], key=lambda r: r["k"])
pop_sorted = sorted(pop_frozen, key=lambda r: r["k"])
f_u0 = zeta0 * fz["n"] if zeta0 else None
st.plotly_chart(build_error_curve([r["k"] for r in pop_sorted], [r["ln_err"] for r in pop_sorted], [r["bias"] for r in grid_rows],
                                  [r["q10_ratio"] for r in grid_rows], [r["q90_ratio"] for r in grid_rows], None, f_u0), width="stretch", key="error_chart")
sc = EV.sign_change([r["bias"] for r in grid_rows if (not zeta0 or r["k"] > f_u0)], [r["k"] for r in grid_rows if (not zeta0 or r["k"] > f_u0)]) if zeta0 else None
st.plotly_chart(build_rmse_curve([r["k"] for r in grid_rows], [r["rmse"] for r in grid_rows], sc), width="stretch", key="rmse_chart")
kind, text = EV.verdict_error(case_key, fz["n"], pop_frozen, fz["gitter"])
getattr(st, kind)(text)
if case_key in ("lognormal", "exponential"):
    st.info("**Ehrlicher Negativbefund:** bei glatten Verteilungen gibt es **keine** Kippschwelle; „die Schwelle hoch genug“ ist dort eine Abwägung zwischen Verzerrung und Streuung, kein Punkt.")
elif case_key == "vehicle":
    st.info("**Nebenbefund:** im Vehikel B gibt es im Gitter k = 25 bis 1000 **keine** Schwelle mit Populationsfehler ≤ 0,05 (kleinster Wert +0,065 bei k = 25); ≤ 0,10 nur für k ≤ 39. "
            "Im Spleiß-Fall gibt es sie (u₀).")

st.markdown("---")

# --- Experiment 3 --------------------------------------------------------------------------------------------------------------------------
st.subheader("🔬 Experiment 3: Regeln wählen – findet eine Regel die Schwelle?")
st.caption(T.EXP3_HELP)
hist_key = f"{rule_letter}{band_key}" if rule_letter in ("S", "M") else rule_letter
counts = fz["regeln"][hist_key]["counts"]
st.plotly_chart(build_choice_hist(fks, counts, EV.rule_label(hist_key), C.RULE_COLORS[rule_letter], f_u0), width="stretch", key="choice_hist")
rows = EV.rules_table(case_key)
df = pd.DataFrame(rows)
pct_cols = [c for c in df.columns if c.endswith("(%)")]
st.dataframe(df, hide_index=True, width="stretch",
             column_config={**{c: st.column_config.NumberColumn(c, format="%.1f %%") for c in pct_cols},
                            "RMSE": st.column_config.NumberColumn("RMSE", format="%.3f"),
                            "Median des Verhältnisses": st.column_config.NumberColumn("Median des Verhältnisses", format="%.2f")})
kind, text = EV.verdict_rules(case_key, band_key)
getattr(st, kind)(text)
if case_key not in C.SPLICE:
    kind, text = EV.verdict_bottom(case_key, band_key)
    getattr(st, kind)(text)
kind, text = EV.verdict_vs_fixed(case_key)
getattr(st, kind)(text)
if case_key in C.SPLICE:
    kind, text = EV.verdict_fixed_risk(EV.frozen("splice10"), EV.frozen("splice03"))
    getattr(st, kind)(text)
st.caption("Regel G ist auf den Testfall kalibriert, wenn ξ = 0,3 in der Kalibrierung steht (Spleiß-Fall); die Kalibrierung ohne den wahren Wert (G2) ändert das Ergebnis dort kaum. "
           "Die Ursache, warum M scheitert, ist nicht untersucht; hier wird nur gezählt.")

st.markdown("---")

# --- Experiment 4 --------------------------------------------------------------------------------------------------------------------------
st.subheader("🔬 Experiment 4: Dieselbe Reihe, sieben Regeln")
st.caption(T.EXP4_HELP)
live_ratio = None
if not series.problem:
    seven = EV.seven_rules(series)
    finite = [r[2] for r in seven if math.isfinite(r[2]) and r[2] > 0]
    if len(finite) == len(seven):
        live_ratio = max(finite) / min(finite)
    st.plotly_chart(build_seven_rules([r for r in seven if math.isfinite(r[2]) and r[2] > 0], series.truth), width="stretch", key="seven_chart")
else:
    st.warning(series.problem)
kind, text = EV.verdict_spread(case_key, live_ratio)
getattr(st, kind)(text)

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------
st.subheader("🚧 Wo die Annahmen enden")
st.markdown(T.ASSUMPTIONS)
st.info(T.VEHIKEL_TEXT)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(T.MATH_LESEHILFE)
    st.markdown(T.MATH_BODY)

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Extremwerte: Blöcke, Schwelle, Pegel](https://sebastianhanisch.net/konzepte-extremwert.html)."
)
