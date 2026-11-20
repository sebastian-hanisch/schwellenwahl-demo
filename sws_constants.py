"""Konstanten: Regler-Grenzen, feste Annahmen, Fälle, Farben und Presets."""

# --- Regler ----------------------------------------------------------------------------------------------------------------------------
CASE_SPLICE10, CASE_SPLICE03, CASE_LOGNORMAL, CASE_EXPONENTIAL, CASE_VEHICLE = (
    "Spleiß-Fall, Anteil 10 %", "Spleiß-Fall, Anteil 3 %", "Lognormal (σ = 1)", "Exponential", "Vehikel B (Verweilzeit-Mischung)")
CASES = (CASE_SPLICE10, CASE_SPLICE03, CASE_LOGNORMAL, CASE_EXPONENTIAL, CASE_VEHICLE)
CASE_KEYS = {CASE_SPLICE10: "splice10", CASE_SPLICE03: "splice03", CASE_LOGNORMAL: "lognormal", CASE_EXPONENTIAL: "exponential", CASE_VEHICLE: "vehicle"}
DEFAULT_CASE = CASE_SPLICE10
N_MIN, N_MAX, N_STEP, DEFAULT_N = 1000, 5000, 500, 2000
VEHICLE_N_MAX = 5000                         # Obergrenze live: eine Reihe des Vehikels B kostet etwa 1 s je 5000 Beobachtungen (SplitMix64 mit exakter Normalzahl)
SEED_MAX, DEFAULT_SEED = 999999, 35
BANDS = (1.5, 2.0, 3.0)
DEFAULT_BAND = 2.0
RULES = ("S (Stabilität)", "M (Mean Residual Life)", "G (Anpassungstest)")
RULE_KEYS = {RULES[0]: "S", RULES[1]: "M", RULES[2]: "G"}
DEFAULT_RULE = RULES[0]

# --- Gitter der Schwellen ----------------------------------------------------------------------------------------------------------------
GRID_STAGES = 18
GRID_K_MIN = 25
GRID_K_FRAC = 0.4                            # k_max = 0,4 n, höchstens 1000: n = 2000 gibt 25 … 800, n = 5000 gibt 25 … 1000
GRID_K_CAP = 1000

# --- Fälle -------------------------------------------------------------------------------------------------------------------------------
P_TARGET = {"splice10": 5e-5, "splice03": 5e-5, "lognormal": 5e-5, "exponential": 5e-5, "vehicle": 1e-5}   # Überschreitungswahrscheinlichkeit je Beobachtung
N_FROZEN = {"splice10": 2000, "splice03": 2000, "lognormal": 2000, "exponential": 2000, "vehicle": 5000}   # n der eingefrorenen Messungen
SPLICE = {"splice10": 0.10, "splice03": 0.03}                       # Anteil ζ₀ über u₀
U0, XI0, SIG0 = 1.0, 0.3, 0.5                                         # Spleiß-Fall: Körper Gleich(0, 1), ab u₀ = 1 exakt GPD(ξ = 3/10, σ = 1/2)
LOGNORMAL_SIGMA = 1.0
BRAUCHBAR_TOLERANZEN = (0.05, 0.10)                                  # „brauchbar“: Populationsfehler des Pegels für die gewählte und alle höheren Schwellen höchstens …

# --- Rechenaufwand -----------------------------------------------------------------------------------------------------------------------
POP_NODES_BODY, POP_NODES_TAIL = 80, 300                             # Gauß-Legendre-Knoten der Populations-ML im Spleiß-Fall
POP_NODES_QUANTILE = 400                                              # Knoten der Quantil-Quadratur (Lognormal, Vehikel B)

# --- Farben ------------------------------------------------------------------------------------------------------------------------------
COLOR_POP = "#2F6B65"
COLOR_RUN = "#4c78a8"
COLOR_TRUTH = "#b04a3a"
COLOR_BAND = "rgba(76,120,168,0.18)"
COLOR_MUTED = "rgba(150,150,150,0.55)"
RULE_COLORS = {"S": "#4c78a8", "M": "#e49444", "G": "#2F6B65"}

# --- Presets (Name -> Regler) ----------------------------------------------------------------------------------------------------------
PRESETS = {
    "Spleiß-Fall 10 %: Kippen bei k = 200": {"case": CASE_SPLICE10, "n": 2000, "seed": DEFAULT_SEED, "band": 2.0, "rule": RULES[0]},
    "Spleiß-Fall 3 %: schärferer Ausschlag": {"case": CASE_SPLICE03, "n": 2000, "seed": DEFAULT_SEED, "band": 2.0, "rule": RULES[0]},
    "Lognormal: weich": {"case": CASE_LOGNORMAL, "n": 2000, "seed": DEFAULT_SEED, "band": 2.0, "rule": RULES[0]},
    "Exponential: nichts kippt": {"case": CASE_EXPONENTIAL, "n": 2000, "seed": DEFAULT_SEED, "band": 2.0, "rule": RULES[0]},
    "Verweilzeit-Mischung (Vehikel B)": {"case": CASE_VEHICLE, "n": 5000, "seed": DEFAULT_SEED, "band": 2.0, "rule": RULES[0]},
    "Dieselbe Reihe, sieben Regeln": {"case": CASE_VEHICLE, "n": 5000, "seed": 7, "band": 2.0, "rule": RULES[2]},
}
PRESET_HELP = {
    "Spleiß-Fall 10 %: Kippen bei k = 200": "Körper gleichverteilt, ab u₀ = 1 exakt GPD, Anteil 10 % über u₀ (bei k = 200 von n = 2000): unter u₀ kippt der Pegel, darüber nicht.",
    "Spleiß-Fall 3 %: schärferer Ausschlag": "Dieselbe Konstruktion mit Anteil 3 % über u₀ (bei k = 60): der Ausschlag unter u₀ ist viel schärfer.",
    "Lognormal: weich": "Glatte Abweichung von der GPD: es gibt keine Kippschwelle, der Fehler driftet mit k.",
    "Exponential: nichts kippt": "Bei jeder Schwelle exakt GPD (ξ = 0): ξ und σ* bleiben, es gibt nichts zu wählen.",
    "Verweilzeit-Mischung (Vehikel B)": "Ausgedachte Verweilzeit-Mischung (1224 Komponenten), n = 5000, p = 10⁻⁵ je Container: kein Kippen, der Populationsfehler wächst stetig bis k ≈ 650 und fällt dann leicht.",
    "Dieselbe Reihe, sieben Regeln": "Vehikel B, anderer Zufalls-Seed: was sieben Regeln auf derselben Reihe als Pegel liefern (Experiment 4).",
}
