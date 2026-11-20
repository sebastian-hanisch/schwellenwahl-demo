"""Auswertung: Fehlermaße, Aggregation der Läufe, eine Reihe live und die Verdikte (reine Funktionen `verdict_*(...) -> (Art, Text)`).

Fehlermaß: ln(Pegel-Dach / wahrer Pegel); „Faktor > 2“ heißt Verhältnis über 2, „< 0,5“ unter 0,5 (Anteil der Läufe). Zähleinheiten: Läufe (R), Stufen des Gitters,
Überschreitungen k, Beobachtungen n."""

import math
from dataclasses import dataclass

import numpy as np

import sws_algorithm as A
import sws_cases as K
import sws_constants as C
import sws_reference as REF


def de(value, digits=2, ascii_minus=False):
    """Zahl mit Dezimalkomma und echtem Minuszeichen (ascii_minus=True: ASCII-Minus, nötig für die Delta-Anzeige von st.metric, die ein Unicode-Minus falsch einfärbt)."""
    s = f"{value:.{digits}f}".replace(".", ",")
    return s if ascii_minus else s.replace("-", "−")


def p_text(p):
    """Überschreitungswahrscheinlichkeit lesbar: 5e-05 gibt „5·10⁻⁵“, 1e-05 gibt „10⁻⁵“."""
    exp = math.floor(math.log10(p) + 1e-12)
    mant = p / 10 ** exp
    sup = "".join("⁻" if ch == "-" else "⁰¹²³⁴⁵⁶⁷⁸⁹"[int(ch)] for ch in str(exp))
    return f"10{sup}" if abs(mant - 1) < 1e-9 else f"{de(mant, 0 if abs(mant - round(mant)) < 1e-9 else 1)}·10{sup}"


def pct(value, digits=1):
    """Anteil (0 bis 1) als Prozentzahl mit Komma, ohne Leerzeichen vor dem Zeichen (das setzt der Aufrufer)."""
    return de(100.0 * value, digits)


# --------------------------------------------------------------------------------------------------------------------------------------
# Fehlermaße und Aggregation
# --------------------------------------------------------------------------------------------------------------------------------------
def error_stats(levels, truth):
    """Kennzahlen von ln(Pegel-Dach/wahr) über Läufe; Läufe ohne endlichen positiven Pegel zählen nicht (n_gueltig)."""
    v = np.asarray(levels, float)
    ok = np.isfinite(v) & (v > 0.0)
    r = v[ok] / truth
    e = np.log(r)
    if e.size == 0:
        return {"n_gueltig": 0, "bias": math.nan, "sd": math.nan, "rmse": math.nan, "median_ratio": math.nan, "q10_ratio": math.nan, "q90_ratio": math.nan,
                "share_below": math.nan, "share_gt2": math.nan, "share_lt05": math.nan}
    return {
        "n_gueltig": int(e.size), "bias": float(e.mean()), "sd": float(e.std(ddof=1)) if e.size > 1 else 0.0, "rmse": float(math.sqrt(np.mean(e ** 2))),
        "median_ratio": float(np.median(r)), "q10_ratio": float(np.quantile(r, 0.1)), "q90_ratio": float(np.quantile(r, 0.9)),
        "share_below": float(np.mean(e < 0)), "share_gt2": float(np.mean(r > 2)), "share_lt05": float(np.mean(r < 0.5)),
    }


def rmse_standard_error(rmse, n_runs):
    """Näherung des Standardfehlers des RMSE aus n_runs Läufen bei annähernd normalverteiltem Fehler: rmse / √(2 n_runs)."""
    return rmse / math.sqrt(2.0 * n_runs)


def proportion_se(share, n_runs):
    """Standardfehler eines Anteils."""
    return math.sqrt(max(share * (1.0 - share), 0.0) / n_runs)


def aggregate_rule(levels, picks, us, ks, truth, u0=None):
    """Eine Regel über alle Läufe: Fehlerkennzahlen des gewählten Pegels, Zählung der gewählten Stufen (Index 0 = niedrigste Schwelle), gewähltes k (Median, Q10, Q90),
    Anteil der Läufe an der niedrigsten Stufe und – im Spleiß-Fall – Anteil der Läufe, in denen die gewählte Schwelle bei oder über u₀ liegt.
    levels, us: Felder (Läufe × Stufen); picks: Index je Lauf."""
    picks = np.asarray(picks, int)
    rows = np.arange(len(picks))
    out = error_stats(np.asarray(levels)[rows, picks], truth)
    kk = np.asarray(ks)[picks]
    out.update({
        "counts": [int(np.sum(picks == j)) for j in range(len(ks))],
        "k_median": float(np.median(kk)), "k_q10": float(np.quantile(kk, 0.1)), "k_q90": float(np.quantile(kk, 0.9)),
        "share_bottom": float(np.mean(picks == 0)),
    })
    if u0 is not None:
        out["share_u_ge_u0"] = float(np.mean(np.asarray(us)[rows, picks] >= u0 - 1e-12))
    return out


def usable_share(counts, usable):
    """Anteil der Läufe, die eine „brauchbare“ Stufe gewählt haben: counts je Stufe, usable je Stufe wahr/falsch."""
    counts = np.asarray(counts, float)
    return float(np.sum(counts[np.asarray(usable, bool)]) / np.sum(counts))


def usable_stages(pop_errors, tol):
    """„Brauchbar“ (eigene Festlegung): der Populationsfehler des Pegels ist für diese und alle HÖHEREN Schwellen höchstens tol (Index 0 = niedrigste Schwelle)."""
    pe = np.abs(np.asarray(pop_errors, float))
    return np.array([bool(np.all(pe[i:] <= tol)) for i in range(len(pe))])


def spread_stats(level_matrix, truth):
    """Streuung zwischen den Regeln auf derselben Reihe: Spalten = Regeln, Zeilen = Läufe. Verhältnis größter / kleinster Pegel je Lauf (nur Läufe, in denen alle Pegel endlich und positiv sind)."""
    m = np.asarray(level_matrix, float)
    ok = np.all(np.isfinite(m) & (m > 0), axis=1)
    mm = m[ok]
    ratio = mm.max(axis=1) / mm.min(axis=1)
    return {
        "n_gueltig": int(ok.sum()), "n_laeufe": int(len(ok)), "median": float(np.median(ratio)), "q10": float(np.quantile(ratio, 0.1)), "q90": float(np.quantile(ratio, 0.9)),
        "max": float(ratio.max()), "share_gt2": float(np.mean(ratio > 2)), "share_gt4": float(np.mean(ratio > 4)),
        "share_truth_inside": float(np.mean((mm.min(axis=1) <= truth) & (mm.max(axis=1) >= truth))),
    }


# --------------------------------------------------------------------------------------------------------------------------------------
# Eine Reihe live
# --------------------------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Settings:
    case: str = C.DEFAULT_CASE
    n: int = C.DEFAULT_N
    seed: int = C.DEFAULT_SEED
    band: float = C.DEFAULT_BAND
    rule: str = C.DEFAULT_RULE

    @property
    def key(self):
        return C.CASE_KEYS[self.case]


def crit_arrays(ks):
    """Kritische Werte der Regel G (Kalibrierung A und B) zu den k des Gitters, linear in ln k aus der eingefrorenen Tabelle."""
    d = REF.crit_data()
    return A.crit_for(ks, d["A"]), A.crit_for(ks, d["B"])


@dataclass(frozen=True)
class Series:
    settings: Settings
    x: np.ndarray
    ks: np.ndarray
    truth: float
    p: float
    result: dict            # Ausgabe von A.analyze
    problem: str | None     # Meldung statt Wahl (zu wenig Daten, Gleichstände, ...)


def run_series(settings: Settings) -> Series:
    """Zieht eine Reihe (Zufalls-Seed, Fall, n) und rechnet alles: Tabelle je Schwelle, alle Regeln, feste k."""
    key = settings.key
    x = K.draw(key, settings.n, settings.seed)
    p = C.P_TARGET[key]
    ks = A.make_grid(settings.n)
    problem = A.data_problem(x, ks)
    crit, crit2 = crit_arrays(ks)
    result = A.analyze(x, p, ks, crit, crit2)
    return Series(settings, x, ks, K.true_level(key, p), p, result, problem)


def picked_rule(series: Series):
    """Gewählte Stufe der im Regler gewählten Regel (mit dem Band des Reglers)."""
    letter = C.RULE_KEYS[series.settings.rule]
    key = f"{letter}{series.settings.band:g}" if letter in ("S", "M") else letter
    return key, series.result["picks"][key]


def seven_rules(series: Series):
    """Die sieben Regeln der Experimente 4: S und M mit Band 2, G und die festen k 2 %, 5 %, 10 %, √n. Rückgabe: Liste (Name, k, Pegel)."""
    st, ks, picks, fx = series.result["stats"], series.ks, series.result["picks"], series.result["fixed"]
    rows = []
    for name, key in (("S (c = 2)", "S2"), ("M (c = 2)", "M2"), ("G", "G")):
        j = picks[key]
        rows.append((name, int(ks[j]), float(st["level"][j])))
    for name in ("obere 2 %", "obere 5 %", "obere 10 %", "k = √n"):
        k, lev = fx[name]
        rows.append((f"fest {name}", int(k), float(lev)))
    return rows


# --------------------------------------------------------------------------------------------------------------------------------------
# Eingefrorene Messungen: Zugriff und abgeleitete Größen
# --------------------------------------------------------------------------------------------------------------------------------------
def band_label(band_key):
    """„2“ gibt „2“, „1.5“ gibt „1,5“ (Schlüssel der Bänder in den Regelnamen)."""
    return str(band_key).replace(".", ",")


def rule_label(key):
    """Anzeigename eines Regelschlüssels: S2 gibt „S (c = 2)“, S1.5 gibt „S (c = 1,5)“, G gibt „G“, G2 gibt „G2 (Kalibrierung ξ ∈ {0; 0,5})“."""
    if key == "G":
        return "G"
    if key == "G2":
        return "G2 (Kalibrierung ξ ∈ {0; 0,5})"
    return f"{key[0]} (c = {band_label(key[1:])})"


def frozen(case_key):
    return REF.rules_data()["fälle"][case_key]


def usable_shares(case_key, rule_key, tolerances=C.BRAUCHBAR_TOLERANZEN):
    """Anteil der Läufe, deren gewählte Stufe „brauchbar“ ist (Populationsfehler des Pegels für diese und alle höheren Stufen höchstens tol), je Toleranz.
    Spleiß-Fall: stattdessen der Anteil mit gewählter Schwelle bei oder über u₀ (Schlüssel „u0“). Exponential: jede Stufe ist brauchbar (Populationsfehler 0)."""
    f = frozen(case_key)
    r = f["regeln"][rule_key]
    if case_key in C.SPLICE:
        return {"u0": r["share_u_ge_u0"]}
    ks = f["gitter_k"]
    out = {}
    for tol in tolerances:
        if case_key == "exponential":
            usable = np.ones(len(ks), bool)
        else:
            usable = usable_stages(REF.population_errors_for_grid(case_key, ks, f["n"]), tol)
        out[tol] = usable_share(r["counts"], usable)
    return out


def sign_change(biases, ks, min_abs=0.1):
    """Zwischen welchen Stufen wechselt die Verzerrung das Vorzeichen? Stufen mit einem Betrag unter min_abs (Verzerrung null im Rauschen) werden übersprungen;
    gefunden wird das erste Paar aufeinanderfolgender deutlicher Werte mit verschiedenem Vorzeichen. Rückgabe: (k_klein, k_groß) oder None."""
    sig = [(int(k), b) for k, b in zip(ks, biases) if abs(b) > min_abs]
    for (ka, a), (kb, b) in zip(sig, sig[1:]):
        if (a > 0) != (b > 0):
            return min(ka, kb), max(ka, kb)
    return None


# --------------------------------------------------------------------------------------------------------------------------------------
# Verdikte: reine Funktionen, jede Zahl im Text kommt aus den Argumenten
# --------------------------------------------------------------------------------------------------------------------------------------
def _signed(v, digits=2):
    return ("+" if v >= 0 else "−") + de(abs(v), digits)


def case_name(case_key):
    return {"splice10": "Spleiß-Fall, Anteil 10 %", "splice03": "Spleiß-Fall, Anteil 3 %", "lognormal": "Lognormal (σ = 1)", "exponential": "Exponential",
            "vehicle": "Vehikel B"}[case_key]


def verdict_stability(case_key, n, rows):
    """Verdikt zu Experiment 1. rows: Populationszeilen am Gitter (Felder k, xi, sigma_star)."""
    rows = sorted(rows, key=lambda r: r["k"])
    if case_key in C.SPLICE:
        zeta0 = C.SPLICE[case_key]
        k0 = int(round(zeta0 * n))
        inside = [r for r in rows if r["k"] <= k0]
        outside = [r for r in rows if r["k"] > k0]
        if not inside:
            return "info", f"{case_name(case_key)}, n = {n}: u₀ liegt bei k = {k0}, unter dem kleinsten k des Gitters ({rows[0]['k']}); das Gitter liegt ganz unterhalb von u₀."
        if len(outside) < 2:
            return "info", f"{case_name(case_key)}, n = {n}: u₀ liegt bei k = {k0}, das Gitter reicht nicht weit genug darunter."
        a, b = outside[0], outside[1]
        return "info", (f"{case_name(case_key)}, n = {n}: oberhalb u₀ (k ≤ {inside[-1]['k']} von {k0}) ist e(u) eine Gerade mit Steigung 3/7 und ξ_Pop = 0,30; "
                        f"darunter springt ξ_Pop auf {de(a['xi'])} (k = {a['k']}) und {de(b['xi'])} (k = {b['k']}), σ* von 0,2 auf {de(b['sigma_star'])}. "
                        "Das Diagramm der Reihe zeigt den Knick nur, wenn die Streuung (das Band) klein genug ist.")
    if case_key == "exponential":
        return "info", ("Exponential: die Überschreitungen sind bei jeder Schwelle exakt GPD mit ξ = 0 und σ* = 1; ξ und σ* bleiben beim Anheben der Schwelle gleich, "
                        "es gibt nichts zu wählen. Was das Diagramm der Reihe zeigt, ist nur das Rauschen der Schätzung.")
    xi_min = min(rows, key=lambda r: r["xi"])
    xi_max = max(rows, key=lambda r: r["xi"])
    jump = max(abs(rows[i + 1]["xi"] - rows[i]["xi"]) for i in range(len(rows) - 1))
    return "info", (f"{case_name(case_key)}, n = {n}: ξ_Pop liegt zwischen {de(xi_min['xi'], 3)} (k = {xi_min['k']}) und {de(xi_max['xi'], 3)} (k = {xi_max['k']}), σ* zwischen "
                    f"{de(min(r['sigma_star'] for r in rows), 2)} und {de(max(r['sigma_star'] for r in rows), 2)} (Population, ohne Rauschen); der größte Sprung von ξ_Pop zwischen zwei "
                    f"Stufen ist {de(jump, 3)}. Weder ξ noch σ* bleiben beim Anheben der Schwelle gleich: die GPD gilt bei keiner Schwelle exakt.")


def verdict_error(case_key, n, pop_rows, grid_rows):
    """Verdikt zu Experiment 2. pop_rows: Populationszeilen (k, ln_err) am Gitter; grid_rows: Läufe je Stufe (k, bias, rmse)."""
    pop = sorted(pop_rows, key=lambda r: r["k"])
    runs = sorted(grid_rows, key=lambda r: r["k"])
    rmin = min(runs, key=lambda r: r["rmse"])
    if case_key in C.SPLICE:
        k0 = int(round(C.SPLICE[case_key] * n))
        zero = [r for r in pop if abs(r["ln_err"]) < 1e-9]
        peak = max(pop, key=lambda r: r["ln_err"])
        below = [r for r in runs if r["k"] > k0]
        sc = sign_change([r["bias"] for r in below], [r["k"] for r in below])
        text = (f"{case_name(case_key)}, n = {n}: für k ≤ {zero[-1]['k'] if zero else '–'} ist der Populationsfehler null; der größte liegt bei k = {peak['k']}: "
                f"ln {_signed(peak['ln_err'])} (Faktor {de(math.exp(peak['ln_err']), 1)}). ")
        best_below = min(below, key=lambda r: r["rmse"]) if below else None
        if sc and best_below and sc[0] <= best_below["k"] <= sc[1]:
            text += (f"Unterhalb von u₀ wechselt die Verzerrung der Läufe zwischen k = {sc[0]} und k = {sc[1]} das Vorzeichen; der kleinste RMSE dort ({de(best_below['rmse'], 2)}, k = {best_below['k']}) "
                     "liegt in diesem Bereich: ein scheinbares Minimum durch den Vorzeichenwechsel, kein brauchbarer Wert.")
        else:
            text += f"Kleinster RMSE der festen k: {de(rmin['rmse'], 2)} bei k = {rmin['k']}."
        return "info", text
    first = pop[0]
    if case_key == "exponential":
        return "info", (f"Exponential: der Populationsfehler ist bei jedem k null (die GPD gilt exakt); der RMSE der Läufe fällt von {de(runs[0]['rmse'], 2)} (k = {runs[0]['k']}) "
                        f"auf {de(runs[-1]['rmse'], 2)} (k = {runs[-1]['k']}): vor allem Rauschen der Schätzung, dazu eine kleine Verzerrung der Läufe von ln {_signed(runs[0]['bias'], 2)} "
                        f"bei k = {runs[0]['k']}; es gibt keine Kippschwelle.")
    peak = max(pop, key=lambda r: r["ln_err"])
    jump = max(abs(pop[i + 1]["ln_err"] - pop[i]["ln_err"]) for i in range(len(pop) - 1))
    base = (f"{case_name(case_key)}, n = {n}: kein Kippen, der Populationsfehler (Quadratur, kein Rauschen) ist {_signed(first['ln_err'], 3)} bei k = {first['k']} und erreicht "
            f"{_signed(peak['ln_err'], 2)} bei k = {peak['k']} (größter Sprung zwischen zwei Stufen {de(jump, 2)}); ")
    return "info", base + f"der RMSE der Läufe hat bei festem k sein Minimum {de(rmin['rmse'], 2)} bei k = {rmin['k']}, mit Wissen der Wahrheit."


def verdict_rules(case_key, band_key):
    """Verdikt zu Experiment 3: die drei Regeln (S und M mit dem Band des Reglers, G) im gewählten Fall (eingefrorene Messung)."""
    f = frozen(case_key)
    rules = f["regeln"]
    s_key, m_key = f"S{band_key}", f"M{band_key}"
    names = {s_key: rule_label(s_key), m_key: rule_label(m_key), "G": "G"}
    R = f["laeufe"]
    if case_key in C.SPLICE:
        share = {k: rules[k]["share_u_ge_u0"] for k in names}
        return "info", (f"{case_name(case_key)}, R = {R} Läufe: {names[s_key]} wählt in {pct(share[s_key])} % der Läufe eine Schwelle bei oder über u₀, {names[m_key]} in {pct(share[m_key])} %, "
                        f"G in {pct(share['G'])} %; RMSE {de(rules[s_key]['rmse'])} / {de(rules[m_key]['rmse'])} / {de(rules['G']['rmse'])}. "
                        f"G ist auf ξ ∈ {{0; 0,3}} kalibriert (das enthält die Wahrheit des Spleiß-Falls); mit ξ ∈ {{0; 0,5}} wählt sie in {pct(rules['G2']['share_u_ge_u0'])} %.")
    tol = C.BRAUCHBAR_TOLERANZEN[1]
    bottom = {k: rules[k]["share_bottom"] for k in names}
    kb = f["gitter_k"][0]
    ref = f["feste_k"]["obere 2 %"]
    use = {k: usable_shares(case_key, k)[tol] for k in names}
    return "info", (f"{case_name(case_key)}, R = {R} Läufe, n = {f['n']}: {names[m_key]} und G wählen in {pct(bottom[m_key], 0)} % bzw. {pct(bottom['G'], 0)} % der Läufe die niedrigste Stufe "
                    f"(k = {kb} von {f['n']}), {names[s_key]} in {pct(bottom[s_key], 0)} %; RMSE {de(rules[s_key]['rmse'])} (S), {de(rules[m_key]['rmse'])} (M) und "
                    f"{de(rules['G']['rmse'])} (G) gegen {de(ref['rmse'])} bei festem k = {ref['k']}. „Brauchbar“ (Populationsfehler höchstens {de(tol, 2)} ab der gewählten Stufe) "
                    f"ist die Wahl in {pct(use[s_key], 0)} % (S), {pct(use[m_key], 0)} % (M) und {pct(use['G'], 0)} % (G) der Läufe.")


def verdict_bottom(case_key, band_key):
    """Hauptfalle (Fallenliste Nr. 2): die Regel wählt die niedrigste Schwelle; mit Zähleinheit Läufe und dem Zusatz „für diese Regeln“."""
    f = frozen(case_key)
    r = f["regeln"]
    bm, bg = r[f"M{band_key}"]["share_bottom"], r["G"]["share_bottom"]
    kind = "warning" if max(bm, bg) >= 0.5 else "info"
    return kind, (f"Die Regel wählt die niedrigste Schwelle: für diese Regeln (M mit c = {band_label(band_key)}, G; eigene Festlegungen) in {pct(bm, 0)} % bzw. {pct(bg, 0)} % der {f['laeufe']} Läufe "
                  f"die niedrigste Stufe des Gitters (k = {f['gitter_k'][0]} von n = {f['n']}), im Fall „{case_name(case_key)}“. Ob das schadet, hängt am Fall: siehe RMSE und Populationsfehler.")


def _compare_rmse(rmse_a, rmse_b, runs):
    """Vergleich zweier RMSE aus je `runs` Läufen: „besser“, „schlechter“ oder „im Rauschen“ (Unterschied höchstens zwei Standardfehler, Standardfehler ≈ RMSE/√(2·Läufe))."""
    se = max(rmse_standard_error(rmse_a, runs), rmse_standard_error(rmse_b, runs))
    if rmse_a < rmse_b - 2.0 * se:
        return "besser"
    if rmse_a > rmse_b + 2.0 * se:
        return "schlechter"
    return "im Rauschen"


def verdict_vs_fixed(case_key):
    """Negativbefund: schlägt eine Regel ein festes k? Spleiß-Fall: G gegen das beste der fünf festen k. Sonst: alle Regel-Varianten gegen das beste der festen k außer „obere 10 %“
    und gesondert gegen „obere 10 %“. „Besser“ heißt: der RMSE ist um mehr als zwei Standardfehler kleiner (Standardfehler ≈ RMSE/√(2R))."""
    f = frozen(case_key)
    runs = f["laeufe"]
    se_txt = lambda r: de(rmse_standard_error(r, runs), 2)  # noqa: E731
    head = f"{case_name(case_key)} (n = {f['n']}, R = {runs} Läufe): "
    if case_key in C.SPLICE:
        name, best = min(f["feste_k"].items(), key=lambda kv: kv[1]["rmse"])
        g = f["regeln"]["G"]
        cmp = _compare_rmse(g["rmse"], best["rmse"], runs)
        return "info", head + (f"G (RMSE {de(g['rmse'], 2)}) gegen das beste der fünf festen k (k = {best['k']}, {name}, RMSE {de(best['rmse'], 2)}): {cmp} "
                               f"(Standardfehler rund {se_txt(best['rmse'])}). „Obere 10 %“ trifft u₀ bei Anteil 10 % per Konstruktion (ζ₀·n = 200), nicht als Eigenschaft der Regel.")
    fixed_main = {n: v for n, v in f["feste_k"].items() if n != "obere 10 %"}
    best = min(fixed_main.values(), key=lambda v: v["rmse"])
    lo, hi = min(v["rmse"] for v in fixed_main.values()), max(v["rmse"] for v in fixed_main.values())
    kmin, kmax = min(v["k"] for v in fixed_main.values()), max(v["k"] for v in fixed_main.values())
    rules = f["regeln"]
    better = [rule_label(k) for k, r in rules.items() if _compare_rmse(r["rmse"], best["rmse"], runs) == "besser"]
    ten = f["feste_k"]["obere 10 %"]
    better10 = [rule_label(k) for k, r in rules.items() if _compare_rmse(r["rmse"], ten["rmse"], runs) == "besser"]
    noise10 = [rule_label(k) for k, r in rules.items() if _compare_rmse(r["rmse"], ten["rmse"], runs) == "im Rauschen"]
    if better:
        verb = "hat" if len(better) == 1 else "haben"
        first = f"{', '.join(better)} {verb} einen um mehr als zwei Standardfehler kleineren RMSE als das beste der festen k = {kmin} bis {kmax} (RMSE {de(best['rmse'], 2)}). "
    else:
        first = (f"keine der {len(rules)} Regel-Varianten hat einen um mehr als zwei Standardfehler kleineren RMSE als das beste der festen k = {kmin} bis {kmax} "
                 f"(RMSE {de(lo, 2)} bis {de(hi, 2)}, Standardfehler rund {se_txt(best['rmse'])}). ")
    tail = (f"Gegen das fest gewählte k = {ten['k']} (obere 10 %, RMSE {de(ten['rmse'], 2)}) sind " + (", ".join(better10) if better10 else "keine der Regel-Varianten") + " um mehr als zwei Standardfehler besser"
            + (f"; im Rauschen gleich: {', '.join(noise10)}." if noise10 else "."))
    return "info", head + first + tail


def verdict_fixed_risk(f10, f03):
    """Ein fest gewähltes k ist im Spleiß-Fall riskant: dasselbe k = 100 bei Anteil 10 % und 3 % (u₀ liegt dort bei k = 200 bzw. 60)."""
    a, b = f10["feste_k"]["obere 5 %"], f03["feste_k"]["obere 5 %"]
    return "warning", (f"Dasselbe feste k = {a['k']} hat im Spleiß-Fall bei Anteil 10 % den RMSE {de(a['rmse'], 2)} (u₀ liegt bei k = 200), bei Anteil 3 % den RMSE {de(b['rmse'], 2)} "
                       f"und einen Median des Verhältnisses von {de(b['median_ratio'], 0)} (u₀ liegt bei k = 60). Das Riskante ist die Konstruktion: u₀ ist dort bei k = 60 gesetzt; "
                       "im Vehikel B sind unter den fünf festen k die von 71 bis 250 die besten (im Nachhinein gelesen) und k = 500 ist ein schlechtes.")


def verdict_spread(case_key, live_ratio=None):
    """Verdikt zu Experiment 4: Streuung zwischen den sieben Regeln auf derselben Reihe (eingefrorene Statistik) und, falls gegeben, die der Reihe oben."""
    f = frozen(case_key)
    sp = f["streuung_regeln"]
    text = (f"{case_name(case_key)}: auf derselben Reihe liegt der höchste der sieben Pegel im Median beim {de(sp['median'])}-Fachen des niedrigsten "
            f"(Q10/Q90 {de(sp['q10'])}/{de(sp['q90'])}; Maximum {de(sp['max'], 1)}); in {pct(sp['share_gt2'])} % der {sp['n_laeufe']} Läufe über Faktor 2, "
            f"in {pct(sp['share_gt4'])} % über Faktor 4. Der wahre Pegel liegt im Bereich der sieben in {pct(sp['share_truth_inside'])} % der Läufe (n = {f['n']}).")
    if live_ratio is not None:
        text += f" Die Reihe oben: höchster zu niedrigstem Pegel {de(live_ratio)}."
    return "info", text


def verdict_live(series: "Series"):
    """Satz zur Reihe oben: gewählte Stufe der gewählten Regel, ihr Pegel gegen den wahren."""
    if series.problem:
        return "warning", series.problem
    key, j = picked_rule(series)
    lev = float(series.result["stats"]["level"][j])
    if not math.isfinite(lev) or lev <= 0:
        return "warning", f"Regel {rule_label(key)} wählt k = {int(series.ks[j])}, aber die Anpassung dort liefert keinen endlichen Pegel."
    ratio = lev / series.truth
    return "info", (f"Regel {rule_label(key)} wählt auf dieser Reihe k = {int(series.ks[j])} (Stufe {j + 1} von {len(series.ks)}) und liefert den Pegel {de(lev)} "
                    f"gegen den wahren {de(series.truth)}: das {de(ratio)}-Fache (ln {_signed(math.log(ratio))}).")


# --------------------------------------------------------------------------------------------------------------------------------------
# Daten für die Diagramme und Tabellen der App
# --------------------------------------------------------------------------------------------------------------------------------------
def population_rows(case_key, n, ks, p=None):
    """Populationszeilen ohne Zufall am Gitter ks (k, zeta, u, xi, sigma_u, sigma_star, ln_err): Spleiß-Fall live per Quadratur, Lognormal und Vehikel B aus der eingefrorenen
    Kurve (linear in ln ζ), Exponential geschlossen (ξ = 0, σ_u = 1, Fehler 0)."""
    import sws_population as PO
    p = C.P_TARGET[case_key] if p is None else p
    if case_key in C.SPLICE:
        return PO.spliced_curve(n, C.SPLICE[case_key], p, ks)
    rows = []
    for k in ks:
        zeta = k / n
        if case_key == "exponential":
            rows.append({"k": int(k), "zeta": zeta, "u": -math.log(zeta), "xi": 0.0, "sigma_u": 1.0, "sigma_star": 1.0, "ln_err": 0.0})
        else:
            rows.append({"k": int(k), "zeta": zeta, **{f: REF.population_interp(case_key, zeta, f) for f in ("u", "xi", "sigma_u", "sigma_star", "ln_err")}})
    return rows


def exact_families(n_points=60):
    """σ_u und σ* der drei exakten Fälle (Exponential, Pareto-Verteilung α = 2, Gleichverteilung) gegen ζ_u, logarithmisch von 0,5 bis 0,001."""
    import sws_population as PO
    out = {}
    for name in PO.EXACT_FAMILIES:
        rows = []
        for zeta in np.geomspace(0.5, 0.001, n_points):
            u = {"Exponential": -math.log(zeta), "Pareto-Verteilung (α = 2)": zeta ** -0.5, "Gleichverteilung": 1.0 - zeta}[name]
            xi, sg, sstar = PO.exact_gpd_family(name, u)
            rows.append({"zeta": float(zeta), "u": u, "xi": xi, "sigma_u": sg, "sigma_star": sstar})
        out[name] = rows
    return out


def exact_mrl_curve(case_key, u_lo, u_hi, n_points=60):
    """Exakte Kurve e(u) zwischen zwei Schwellen (für das Mean-Residual-Life-Diagramm)."""
    us = np.linspace(u_lo, u_hi, n_points)
    return us, np.array([K.exact_mrl(case_key, float(u)) for u in us])


def series_table(series: Series):
    """Tabelle je Stufe der Reihe (für die Anzeige): k, Schwelle, ξ̂, σ̂*, e(u), Pegel-Dach."""
    st, ks = series.result["stats"], series.ks
    return [{"k": int(ks[j]), "u": float(st["u"][j]), "xi": float(st["xi"][j]), "sigma_star": float(st["sigma"][j] - st["xi"][j] * st["u"][j]), "e": float(st["e"][j]),
             "level": float(st["level"][j])} for j in range(len(ks))]


def rules_table(case_key):
    """Zeilen der Regel-Tabelle von Experiment 3 (eingefrorene Messung): je Regel und je festem k. Anteile in Prozent (Zahl × 100)."""
    f = frozen(case_key)
    splice = case_key in C.SPLICE
    rows = []

    def add(label, r, kk, key=None):
        row = {"Regel": label, "RMSE": r["rmse"], "Median des Verhältnisses": r["median_ratio"], "über Faktor 2 (%)": 100 * r["share_gt2"], "unter Faktor 0,5 (%)": 100 * r["share_lt05"],
               "gewähltes k: Median (Q10/Q90)": kk, "niedrigste Stufe (%)": 100 * r["share_bottom"] if "share_bottom" in r else None}
        if key is not None and splice:
            row["bei oder über u₀ (%)"] = 100 * r["share_u_ge_u0"]
        elif key is not None:
            sh = usable_shares(case_key, key)
            row["brauchbar ≤ 0,05 (%)"], row["brauchbar ≤ 0,10 (%)"] = 100 * sh[C.BRAUCHBAR_TOLERANZEN[0]], 100 * sh[C.BRAUCHBAR_TOLERANZEN[1]]
        rows.append(row)

    for key in A.RULE_KEYS_ALL:
        r = f["regeln"][key]
        add(rule_label(key), r, f"{r['k_median']:.0f} ({r['k_q10']:.0f}/{r['k_q90']:.0f})", key)
    for name, r in f["feste_k"].items():
        add(f"fest {name} (k = {r['k']})", r, "–")
    return rows
