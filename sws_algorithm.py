"""Kern: GPD-Anpassung (Maximum Likelihood, Profil in einer Variablen), Pegel, Kolmogorov-Smirnov-Abstand, Schwellengitter und die drei Regeln S, M und G
(eigene Festlegungen dieser Demo, keine Literaturverfahren). Nur numpy und Standardbibliothek (scipy steht nur in den Tests).

Konvention: ξ > 0 Fréchet-Typ, ξ < 0 Weibull-Typ (`scipy.stats.genpareto` c = ξ). Schwelle bei k Überschreitungen = (k+1)-größter Wert, Gitter absteigend in k
(Index 0 = niedrigste Schwelle). σ* = σ_u − ξ·u ist die modifizierte Skala."""

import math

import numpy as np

import sws_constants as C
import sws_numeric as N

# --------------------------------------------------------------------------------------------------------------------------------------
# GPD: Verteilungsfunktion, Pegel, KS-Abstand
# --------------------------------------------------------------------------------------------------------------------------------------


def gpd_cdf(y, xi, sigma):
    """Verteilungsfunktion der GPD(ξ, σ) an Überschreitungen y ≥ 0."""
    y = np.asarray(y, float)
    if abs(xi) < 1e-9:
        return 1.0 - np.exp(-y / sigma)
    return 1.0 - np.maximum(1.0 + xi * y / sigma, 1e-300) ** (-1.0 / xi)


def ks_distance(y, xi, sigma):
    """Kolmogorov-Smirnov-Abstand D der Überschreitungen y zur GPD(ξ, σ) (nicht skaliert)."""
    ys = np.sort(np.asarray(y, float))
    k = ys.size
    f = gpd_cdf(ys, xi, sigma)
    i = np.arange(1, k + 1)
    return float(max(np.max(i / k - f), np.max(f - (i - 1) / k)))


def pot_level(xi, sigma, u, zeta, p):
    """Pegel zur Überschreitungswahrscheinlichkeit p je Beobachtung: u + σ/ξ ((ζ/p)^ξ − 1); ζ = Anteil der Beobachtungen über u."""
    r = zeta / p
    if abs(xi) < 1e-8:
        return u + sigma * math.log(r)
    return u + sigma / xi * (r ** xi - 1.0)


# --------------------------------------------------------------------------------------------------------------------------------------
# GPD-ML: Profil-Likelihood in t = θ·ȳ mit θ = ξ/σ (Grimshaw-Reduktion auf eine Variable): ξ(t) = mean ln(1 + t y/ȳ), σ = ξ ȳ / t
# --------------------------------------------------------------------------------------------------------------------------------------
_GRID_T = np.concatenate([np.linspace(-1.0, -1e-3, 70), [0.0], np.geomspace(1e-3, 40.0, 90)])   # der untere Teil wird je Stichprobe auf (-ȳ/y_max, 0) skaliert


def _xi_of_t(t, yn):
    """ξ(t) = mean ln(1 + t yn) für ein Feld t (yn = y/ȳ)."""
    return np.mean(np.log1p(np.multiply.outer(t, yn)), axis=-1)


def gpd_ml(y, xi_min=-1.0):
    """ML der GPD (ξ, σ) an Überschreitungen y > 0. Gibt (ξ, σ, Randtreffer) zurück; ξ < ξ_min ist ausgeschlossen (für ξ < −1 ist die Likelihood unbeschränkt),
    Randtreffer = True, wenn die Lösung am Rand ξ_min liegt oder am oberen Ende des Gitters. Ohne Lösung (alle Werte gleich oder nicht endlich): (nan, nan, True)."""
    y = np.asarray(y, float)
    k = y.size
    if k < 2 or not np.all(np.isfinite(y)):
        return math.nan, math.nan, True
    yb = float(y.mean())
    ymax = float(y.max())
    if yb <= 0.0 or ymax <= 0.0:
        return math.nan, math.nan, True
    yn = y / yb
    tlo = -yb / ymax * (1.0 - 1e-9)
    grid = _GRID_T.copy()
    grid[:70] = np.linspace(tlo, -1e-3, 70)
    grid = np.unique(grid)                                    # aufsteigend (tlo kann in seltenen Fällen über −10⁻³ liegen)
    log_yb = math.log(yb)

    def ll_vec(t):
        xi = _xi_of_t(t, yn)
        with np.errstate(divide="ignore", invalid="ignore"):
            val = k * np.log(t / xi) - k * log_yb - k * (xi + 1.0)
        val = np.where(np.abs(t) < 1e-9, -k * log_yb - k, val)
        bad = (xi < xi_min) | (xi == 0.0) | (t / np.where(xi == 0.0, 1.0, xi) <= 0.0)
        return np.where(bad & (np.abs(t) >= 1e-9), -np.inf, val)

    vals = ll_vec(grid)
    i = int(np.argmax(vals))
    if not np.isfinite(vals[i]):
        return math.nan, math.nan, True

    def nll1(t):
        v = float(ll_vec(np.array([t]))[0])
        return -v if np.isfinite(v) else 1e300

    a, b = float(grid[max(i - 1, 0)]), float(grid[min(i + 1, grid.size - 1)])
    t = float(grid[i])
    if a < b:
        tx, fx = N.golden_min(nll1, a, b, xtol=1e-10)
        if -fx >= vals[i] - 1e-12:
            t = float(tx)
    if abs(t) < 1e-9:
        return 0.0, yb, False
    xi = float(_xi_of_t(np.array([t]), yn)[0])
    sigma = xi * yb / t
    edge = (xi <= xi_min + 1e-3) or (i >= grid.size - 2)
    return xi, sigma, bool(edge)


# --------------------------------------------------------------------------------------------------------------------------------------
# Schwellengitter und Tabelle je Schwelle
# --------------------------------------------------------------------------------------------------------------------------------------


def make_grid(n, j=C.GRID_STAGES, k_min=C.GRID_K_MIN, frac=C.GRID_K_FRAC, k_cap=C.GRID_K_CAP):
    """Gitter der Überschreitungszahlen k, log-gleichmäßig, absteigend (Index 0 = niedrigste Schwelle). n = 2000: 25 … 800, n = 5000: 25 … 1000 (je 18 Stufen)."""
    k_max = min(int(round(frac * n)), k_cap, n - 1)
    k_max = max(k_max, k_min + 1)
    ks = np.unique(np.round(np.geomspace(k_max, k_min, j)).astype(int))
    return ks[::-1]


def fixed_ks(n):
    """Feste k der Vergleichsregeln: obere 2 %, 5 %, 10 % und die Faustregeln √n und n^(2/3)/ln ln n (nach Scarrott/MacDonald laut Quellenbericht, ungeprüft)."""
    out = {
        "obere 2 %": max(int(round(0.02 * n)), 10),
        "obere 5 %": max(int(round(0.05 * n)), 10),
        "obere 10 %": max(int(round(0.10 * n)), 10),
        "k = √n": int(round(math.sqrt(n))),
        "k = n^(2/3)/ln ln n": int(round(n ** (2.0 / 3.0) / math.log(math.log(n)))),
    }
    return {name: min(k, n - 1) for name, k in out.items()}


def threshold_stats(x, ks, p, xs=None):
    """Je Schwelle des Gitters: Schwelle u, ξ̂, σ̂, mittlere Überschreitung e(u) mit Standardfehler, √k·D und Pegel-Dach zu p. Rückgabe: Wörterbuch von Feldern (Länge J)."""
    xs = np.sort(np.asarray(x, float)) if xs is None else xs
    n = xs.size
    j = len(ks)
    out = {name: np.full(j, np.nan) for name in ("u", "xi", "sigma", "e", "se", "D", "level")}
    for a, k in enumerate(ks):
        k = int(k)
        u = xs[n - k - 1]
        y = xs[n - k:] - u
        xi, sg, _edge = gpd_ml(y)
        out["u"][a] = u
        out["e"][a] = y.mean()
        out["se"][a] = y.std(ddof=1) / math.sqrt(k) if k > 1 else math.nan
        if np.isfinite(xi):
            out["xi"][a] = xi
            out["sigma"][a] = sg
            out["D"][a] = ks_distance(y, xi, sg) * math.sqrt(k)
            out["level"][a] = pot_level(xi, sg, u, k / n, p)
    return out


def fixed_level(xs, k, p):
    """Pegel-Dach einer festen Schwelle bei k Überschreitungen (Schwelle = (k+1)-größter Wert); nan, wenn die Anpassung scheitert."""
    n = xs.size
    u = xs[n - k - 1]
    xi, sg, _edge = gpd_ml(xs[n - k:] - u)
    return pot_level(xi, sg, u, k / n, p) if np.isfinite(xi) else math.nan


# --------------------------------------------------------------------------------------------------------------------------------------
# Die drei Regeln (EIGENE Festlegungen). Rückgabe: Index im Gitter (0 = niedrigste Schwelle).
# --------------------------------------------------------------------------------------------------------------------------------------


def choose_S(xis, ks, c=2.0):
    """Regel S (Stabilität): die niedrigste Schwelle j, ab der alle ξ̂ der höheren Schwellen im Band ± c·(1+ξ̂_h)/√k_h um ξ̂_j liegen
    (Standardfehler der jeweils HÖHEREN Schwelle h; 1+ξ̂ nach unten durch 0,3 begrenzt). Geprüft werden nur Schwellen mit mindestens drei höheren Nachbarn;
    findet sich keine, wählt S die höchste Schwelle."""
    xis = np.nan_to_num(np.asarray(xis, float), nan=9.0)
    ks = np.asarray(ks, float)
    se = np.maximum(1.0 + xis, 0.3) / np.sqrt(ks)
    j_n = len(ks)
    for j in range(j_n - 3):
        if np.all(np.abs(xis[j + 1:] - xis[j]) <= c * se[j + 1:]):
            return j
    return j_n - 1


def choose_M(us, es, ses, c=2.0):
    """Regel M (Mean Residual Life): die niedrigste Schwelle j, ab der e(u) in allen höheren Schwellen innerhalb ± c Standardfehler um die gewichtete Gerade liegt
    (kleinste Quadrate mit dem Gewicht 1/Standardfehler², das ist `numpy.polyfit(w = 1/Standardfehler)`; mindestens fünf Stufen). Sonst Rückfall auf die Stufe J − 5 (bei weniger als fünf Stufen auf die Stufe 0)."""
    us, es, ses = (np.asarray(v, float) for v in (us, es, ses))
    j_n = len(us)
    fallback = max(j_n - 5, 0)
    for j in range(max(j_n - 4, 0)):
        if not (np.all(np.isfinite(es[j:])) and np.all(np.isfinite(ses[j:])) and np.all(ses[j:] > 0)):
            continue
        coef = np.polyfit(us[j:], es[j:], 1, w=1.0 / ses[j:])
        if np.all(np.abs(es[j:] - np.polyval(coef, us[j:])) <= c * ses[j:]):
            return j
    return fallback


def choose_G(dists, crit):
    """Regel G (Anpassungstest): die niedrigste Schwelle, deren √k·D (KS-Abstand der Überschreitungen zur ML-angepassten GPD) höchstens den unter H0 simulierten
    90-%-Wert `crit` erreicht, je Schwelle einzeln, ohne Mehrfachtest-Korrektur. Ohne Treffer: die höchste Schwelle."""
    dists = np.asarray(dists, float)
    ok = np.where(np.isfinite(dists) & (dists <= np.asarray(crit, float)))[0]
    return int(ok[0]) if ok.size else len(dists) - 1


def crit_for(ks, table):
    """Kritische Werte der Regel G zu den k des Gitters: aus der eingefrorenen Tabelle {k: Wert}; Zwischenwerte linear in ln k (nur wenn das Gitter nicht zu den
    kalibrierten k gehört, das ist bei beliebigem n im Regler der Fall)."""
    kk = np.array(sorted(table), float)
    vv = np.array([table[int(k)] for k in kk], float)
    return np.interp(np.log(np.asarray(ks, float)), np.log(kk), vv)


def choose_all(stats, ks, crit, c, crit2=None):
    """Alle Regeln auf eine Reihe: {Name: Index}; S, M mit Band c, G (Kalibrierung A) und ggf. G2 (Kalibrierung B)."""
    picks = {
        "S": choose_S(stats["xi"], ks, c),
        "M": choose_M(stats["u"], stats["e"], stats["se"], c),
        "G": choose_G(stats["D"], crit),
    }
    if crit2 is not None:
        picks["G2"] = choose_G(stats["D"], crit2)
    return picks


def data_problem(x, ks):
    """Meldung statt Wahl: None, wenn die Daten brauchbar sind, sonst ein Satz. Prüft Länge, Gleichstände unter den größten Werten und Streuung."""
    x = np.asarray(x, float)
    if x.size < 100:
        return "Weniger als 100 Beobachtungen: keine Schwellenwahl."
    if not np.all(np.isfinite(x)):
        return "Die Daten enthalten nicht endliche Werte."
    top = np.sort(x)[-(int(ks[0]) + 1):]
    if np.unique(top).size < 0.9 * top.size:
        return "Mehr als 10 % Gleichstände unter den größten Werten (gerundete Daten): die Regeln sind dafür nicht gemessen."
    if np.ptp(top) <= 0.0:
        return "Alle größten Werte sind gleich."
    return None


# --------------------------------------------------------------------------------------------------------------------------------------
# Eine Reihe von Anfang bis Ende: Tabelle je Schwelle, alle Regeln, feste k
# --------------------------------------------------------------------------------------------------------------------------------------
RULE_KEYS_ALL = tuple([f"S{c:g}" for c in C.BANDS] + [f"M{c:g}" for c in C.BANDS] + ["G", "G2"])


def analyze(x, p, ks=None, crit=None, crit2=None, with_fixed=True):
    """Alles, was die App und die Messungen an einer Reihe brauchen: Gitter, Tabelle je Schwelle, die gewählten Stufen aller Regeln ({Schlüssel: Index}; S und M mit allen
    Bändern c, G mit der Kalibrierung `crit`, G2 mit `crit2`) und die Pegel der festen k ({Name: (k, Pegel)})."""
    x = np.asarray(x, float)
    n = x.size
    ks = make_grid(n) if ks is None else np.asarray(ks)
    xs = np.sort(x)
    stats = threshold_stats(x, ks, p, xs)
    picks = {}
    for c in C.BANDS:
        picks[f"S{c:g}"] = choose_S(stats["xi"], ks, c)
        picks[f"M{c:g}"] = choose_M(stats["u"], stats["e"], stats["se"], c)
    if crit is not None:
        picks["G"] = choose_G(stats["D"], crit)
    if crit2 is not None:
        picks["G2"] = choose_G(stats["D"], crit2)
    fixed = {}
    if with_fixed:
        for name, k in fixed_ks(n).items():
            fixed[name] = (k, fixed_level(xs, k, p))
    return {"n": n, "ks": ks, "stats": stats, "picks": picks, "fixed": fixed}
