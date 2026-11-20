"""Zahlen und Tabellen der README als Text-Bausteine, berechnet aus den eingefrorenen Messungen und den deterministischen Rechnungen (Quadratur, Mischung, Brüche).

`tests/test_claims.py` prüft, dass jeder Baustein wörtlich in der README steht (und damit jede README-Zahl eine Herkunft hat); der Satzbau der README ist Handarbeit.
Die Bausteine mit Messungen (Läufe) stehen als „Messung dieser README“; ihre Neu-Rechnung als Band prüft `tests/test_tabellen_und_baender.py`."""

from functools import lru_cache

import numpy as np

import sws_algorithm as A
import sws_cases as K
import sws_constants as C
import sws_evaluation as EV
import sws_population as PO
import sws_reference as REF
import sws_vehicle as V

de = EV.de


def sg(v, digits=2):
    return ("+" if v >= 0 else "−") + de(abs(v), digits)


def _pc(v, digits=1):
    return de(100 * v, digits)


def _rule_row(key, r, kind):
    name = EV.rule_label(key)
    if kind == "splice":
        return f"| {name} | {_pc(r['share_u_ge_u0'])} % | {de(r['rmse'], 3)} | {de(r['median_ratio'])} | {r['k_median']:.0f} ({r['k_q10']:.0f}/{r['k_q90']:.0f}) | {_pc(r['share_bottom'], 0)} % |"
    return (f"| {name} | {de(r['rmse'], 3)} | {de(r['median_ratio'])} | {_pc(r['share_gt2'])} % / {_pc(r['share_lt05'])} % | {r['k_median']:.0f} ({r['k_q10']:.0f}/{r['k_q90']:.0f}) | "
            f"{_pc(r['share_bottom'], 0)} % |")


def _fixed_row(name, r, kind):
    label = f"fest k = {r['k']} ({name})"
    if kind == "splice":
        return f"| {label} | – | {de(r['rmse'], 3)} | {de(r['median_ratio'])} | – | – |"
    return f"| {label} | {de(r['rmse'], 3)} | {de(r['median_ratio'])} | {_pc(r['share_gt2'])} % / {_pc(r['share_lt05'])} % | – | – |"


def _table(case_key):
    f = EV.frozen(case_key)
    kind = "splice" if case_key in C.SPLICE else "other"
    if kind == "splice":
        head = ["| Regel | bei oder über u₀ | RMSE | Median des Verhältnisses | gewähltes k: Median (Q10/Q90) | niedrigste Stufe |", "|---|---|---|---|---|---|"]
    else:
        head = ["| Regel | RMSE | Median des Verhältnisses | über 2 / unter 0,5 | gewähltes k: Median (Q10/Q90) | niedrigste Stufe |", "|---|---|---|---|---|---|"]
    rows = [_rule_row(k, f["regeln"][k], kind) for k in A.RULE_KEYS_ALL]
    rows += [_fixed_row(nm, v, kind) for nm, v in f["feste_k"].items()]
    return "\n".join(head + rows)


def _grid_table(case_key, ks_show):
    f = EV.frozen(case_key)
    by = {g["k"]: g for g in f["gitter"]}
    rows = ["| k | Verzerrung (Mittel von ln) | Streuung (sd) | RMSE | Anteil unter wahr |", "|---|---|---|---|---|"]
    for k in ks_show:
        g = by[k]
        rows.append(f"| {k} | {sg(g['bias'])} | {de(g['sd'])} | {de(g['rmse'], 3)} | {_pc(g['share_below'], 0)} % |")
    return "\n".join(rows)


def _pop_list(rows, ks, digits=2, with_k=True):
    by = {r["k"]: r for r in rows}
    return " / ".join(sg(by[k]["ln_err"], digits) for k in ks)


@lru_cache(maxsize=1)
def _build():
    f = {}
    n = 2000
    g2000, g5000 = A.make_grid(2000), A.make_grid(5000)
    r10 = {r["k"]: r for r in PO.spliced_curve(n, 0.10, 5e-5, g2000)}
    r03 = {r["k"]: r for r in PO.spliced_curve(n, 0.03, 5e-5, g2000)}
    f["sp10_pop"] = " / ".join(sg(r10[k]["ln_err"]) for k in (235, 289, 354, 434, 532, 652, 800))
    f["sp10_factor289"], f["sp10_factor354"] = de(np.exp(r10[289]["ln_err"]), 1), de(np.exp(r10[354]["ln_err"]), 1)
    f["sp10_err289"], f["sp10_err354"] = sg(r10[289]["ln_err"]), sg(r10[354]["ln_err"])
    f["sp10_xi"] = " / ".join(de(r10[k]["xi"]) for k in (235, 289, 354))
    f["sp10_sstar"] = " / ".join(de(r10[k]["sigma_star"]) for k in (235, 289, 354))
    f["sp03_pop"] = ", ".join(f"{sg(r03[k]['ln_err'])} (k = {k})" for k in (69, 85, 104, 128, 157, 192, 235, 289, 354, 434, 800))
    f["sp03_factor104"] = de(np.exp(r03[104]["ln_err"]), 0)
    f["sp03_xi104"] = de(r03[104]["xi"])
    f["e_values"] = "e(0) = 87/140 = 0,6214, e(1/2) = 131/308 = 0,4253, e(1) = 5/7 = 0,7143"
    f["slopes"] = "3/7 = 0,429 rechts von u₀ und 38/7 = 5,43 links davon"
    f["levels"] = (f"{de(K.true_level('splice10'), 4)} (Spleiß 10 %), {de(K.true_level('splice03'), 4)} (Spleiß 3 %), {de(K.true_level('lognormal'), 4)} (Lognormal), "
                   f"{de(K.true_level('exponential'), 4)} (Exponential), {de(V.level(1e-5), 3)} Tage (Vehikel B)")
    f["veh_a"], f["veh_b"], f["veh_b99"] = de(V.level(1e-5), 3), de(V.daily_max_level(0.999), 3), de(V.daily_max_level(0.99), 3)
    mean, median = V.mean_and_median()
    f["veh_mean"], f["veh_median"], f["veh_p30"] = de(mean, 3), de(median, 3), de(V.survival(30.0), 6)
    # Vehikel B: Populationskurve
    vrows = {int(round(r["zeta"] * 5000)): r for r in REF.population_data()["vehicle"] if abs(r["zeta"] * 5000 - round(r["zeta"] * 5000)) < 1e-6}
    ks_v = (25, 60, 176, 272, 420, 648, 1000)
    f["veh_pop"] = " / ".join(sg(vrows[k]["ln_err"], 3) for k in ks_v)
    f["veh_xi"] = " / ".join(de(vrows[k]["xi"], 3) for k in ks_v)
    f["veh_sstar"] = f"{de(vrows[25]['sigma_star'])} bei k = 25, {de(vrows[648]['sigma_star'])} bei k = 648"
    errs = {int(k): REF.population_interp("vehicle", k / 5000) for k in g5000}
    le05 = [k for k in sorted(errs) if errs[k] <= 0.05]
    le10 = [k for k in sorted(errs) if errs[k] <= 0.10]
    f["veh_le05"] = "kein Gitterpunkt" if not le05 else f"k ≤ {max(le05)}"
    f["veh_le10"] = f"k ≤ {max(le10)}"
    f["veh_min_err"] = f"{sg(errs[25], 3)} bei k = 25"
    f["veh_peak"] = f"{sg(max(errs.values()), 2)} bei k = {max(errs, key=errs.get)}"
    f["veh_ps"] = f"{de(vrows[39]['ln_err'], 3).replace('0,', '+0,')} bei k = 39 und {sg(errs[48], 3)} bei k = 48"
    # Lognormal: Populationsfehler
    lrows = PO.population_row  # noqa: F841 (nur zur Lesbarkeit: Zeilen kommen aus der Datei)
    lerr = {int(k): REF.population_interp("lognormal", k / 2000) for k in g2000}
    usable = EV.usable_stages([lerr[int(k)] for k in g2000], 0.05)
    f["logn_le05"] = f"k ≤ {int(max(k for k, u in zip(g2000, usable) if u))}"
    f["logn_err800"] = sg(lerr[800])
    f["logn_err25"] = sg(lerr[25], 3)
    # Kalibrierung
    crit = REF.crit_data()
    a, b = list(crit["A"].values()), list(crit["B"].values())
    f["crit_a"], f["crit_b"] = f"{de(min(a), 3)} bis {de(max(a), 3)}", f"{de(min(b), 3)} bis {de(max(b), 3)}"
    f["crit_n"] = str(len(crit["A"]))
    # Tabellen der Läufe
    for ck in ("splice10", "splice03", "vehicle", "lognormal", "exponential"):
        f[f"table_{ck}"] = _table(ck)
    f["grid_vehicle"] = _grid_table("vehicle", (25, 60, 92, 114, 142, 176, 272, 420, 522, 648, 1000))
    f["grid_splice10"] = _grid_table("splice10", (192, 235, 289, 354, 434, 532, 652, 800))
    # Kennzahlen im Text
    s10, s03, veh = EV.frozen("splice10"), EV.frozen("splice03"), EV.frozen("vehicle")
    for tag, fr in (("s10", s10), ("s03", s03)):
        for key in ("S1.5", "S2", "S3", "M1.5", "M2", "M3", "G", "G2"):
            f[f"{tag}_u0_{key}"] = _pc(fr["regeln"][key]["share_u_ge_u0"])
            if key in ("G", "G2", "M3"):
                f[f"{tag}_rmse_{key}"] = de(fr["regeln"][key]["rmse"], 2)
    g10 = {g["k"]: g for g in s10["gitter"]}
    f["s10_bottom_M3"] = _pc(s10["regeln"]["M3"]["share_bottom"], 0)
    f["s10_bias652"], f["s10_bias800"], f["s10_rmse652"] = sg(g10[652]["bias"]), sg(g10[800]["bias"]), de(g10[652]["rmse"], 2)
    f["s03_k45_rmse"] = de(s03["feste_k"]["k = √n"]["rmse"], 2)
    ex = {g["k"]: g for g in EV.frozen("exponential")["gitter"]}
    f["exp_rmse25"], f["exp_rmse800"] = de(ex[25]["rmse"], 2), de(ex[800]["rmse"], 2)
    f["veh_pop1000"] = sg(vrows[1000]["ln_err"], 3)
    f["s10_fixed_rmse"] = " bis ".join((de(min(v["rmse"] for v in s10["feste_k"].values()), 2), de(max(v["rmse"] for v in s10["feste_k"].values()), 2)))
    f["s10_k100_rmse"], f["s03_k100_rmse"] = de(s10["feste_k"]["obere 5 %"]["rmse"], 3), de(s03["feste_k"]["obere 5 %"]["rmse"], 3)
    f["s03_k100_median"] = de(s03["feste_k"]["obere 5 %"]["median_ratio"], 0)
    f["veh_bottom_M2"], f["veh_bottom_G"] = _pc(veh["regeln"]["M2"]["share_bottom"], 0), _pc(veh["regeln"]["G"]["share_bottom"], 0)
    f["veh_rmse_M2"], f["veh_rmse_G"] = de(veh["regeln"]["M2"]["rmse"], 2), de(veh["regeln"]["G"]["rmse"], 2)
    f["veh_rmse_fixed100"] = de(veh["feste_k"]["obere 2 %"]["rmse"], 3)
    f["veh_rmse_fixed500"] = de(veh["feste_k"]["obere 10 %"]["rmse"], 2)
    fm = {nm: v for nm, v in veh["feste_k"].items() if nm != "obere 10 %"}
    f["veh_fixed_range"] = f"{de(min(v['rmse'] for v in fm.values()), 2)} bis {de(max(v['rmse'] for v in fm.values()), 2)}"
    f["veh_rules_range"] = f"{de(min(r['rmse'] for r in veh['regeln'].values()), 2)} bis {de(max(r['rmse'] for r in veh['regeln'].values()), 2)}"
    best = min(veh["gitter"], key=lambda g: g["rmse"])
    f["veh_best_grid"] = f"{de(best['rmse'], 2)} bei k = {best['k']}"
    sp = veh["streuung_regeln"]
    f["spread_veh"] = (f"Median {de(sp['median'])}, Q10/Q90 {de(sp['q10'])}/{de(sp['q90'])}, Maximum {de(sp['max'], 1)}; über Faktor 2 in {_pc(sp['share_gt2'])} %, über Faktor 4 in "
                       f"{_pc(sp['share_gt4'])} % der {sp['n_laeufe']} Läufe; der wahre Pegel liegt zwischen niedrigstem und höchstem der sieben in {_pc(sp['share_truth_inside'])} %")
    lines = ["| Fall | Median des Verhältnisses höchster / niedrigster | Q10 / Q90 | Maximum | über 2 | über 4 | wahrer Pegel im Bereich |", "|---|---|---|---|---|---|---|"]
    for ck in ("splice10", "splice03", "lognormal", "exponential", "vehicle"):
        s = EV.frozen(ck)["streuung_regeln"]
        lines.append(f"| {EV.case_name(ck)} (n = {EV.frozen(ck)['n']}, R = {s['n_laeufe']}) | {de(s['median'])} | {de(s['q10'])} / {de(s['q90'])} | {de(s['max'], 1)} | {_pc(s['share_gt2'])} % | {_pc(s['share_gt4'])} % | {_pc(s['share_truth_inside'])} % |")
    f["table_spread"] = "\n".join(lines)
    f["usable_veh"] = " / ".join(f"{_pc(EV.usable_shares('vehicle', k)[0.10], 0)} %" for k in ("S2", "M2", "G"))
    f["veh_s2_bottom"] = _pc(veh["regeln"]["S2"]["share_bottom"], 0)
    f["veh_s2_rmse"] = de(veh["regeln"]["S2"]["rmse"], 3)
    f["veh_k_med"] = f"{veh['regeln']['S2']['k_median']:.0f} ({veh['regeln']['S2']['k_q10']:.0f}/{veh['regeln']['S2']['k_q90']:.0f})"
    f["fixed_ks_2000"] = " und ".join(f"{k}" for k in (A.fixed_ks(2000)["k = √n"], A.fixed_ks(2000)["k = n^(2/3)/ln ln n"]))
    f["fixed_ks_5000"] = " und ".join(f"{k}" for k in (A.fixed_ks(5000)["k = √n"], A.fixed_ks(5000)["k = n^(2/3)/ln ln n"]))
    f["grid_2000"] = ", ".join(str(int(k)) for k in g2000[::-1])
    f["R"] = f"R = {s10['laeufe']} Läufe (Spleiß, Lognormal, Exponential) bzw. R = {veh['laeufe']} Läufe (Vehikel B)"
    f["time_rules"] = " + ".join(f"{EV.frozen(ck)['rechenzeit_s']} s" for ck in ("splice10", "splice03", "lognormal", "exponential", "vehicle"))
    return f


def build():
    return dict(_build())
