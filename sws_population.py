"""Populationsgrößen ohne Zufall: die GPD, die die WAHRE Überschreitungsverteilung bei einer Schwelle am besten beschreibt (Maximum Likelihood auf der Population,
Gauß-Legendre-Quadratur), und die exakten Kurven ξ, σ_u, σ* gegen die Schwelle. Nur numpy und Standardbibliothek.

* Spleiß-Fall: unterhalb u₀ mischt die Überschreitungsverteilung ein gleichverteiltes Stück (Körper) und die GPD (Tail); oberhalb u₀ ist sie exakt GPD.
* Allgemeine Fälle (Lognormal, Vehikel B): Quantil-Quadratur `gpd_population_ml(isf, ζ)` über die Umkehrung der Überlebensfunktion.
* Geschlossene Fälle: Exponential, Pareto-Verteilung, Gleichverteilung (GPD bei jeder Schwelle exakt)."""

import math
from statistics import NormalDist

import numpy as np

import sws_algorithm as A
import sws_cases as K
import sws_constants as C
import sws_numeric as N

_ND = NormalDist()
_GL_CACHE = {}


def _gl(n):
    """Gauß-Legendre-Knoten und -Gewichte auf [0, 1]."""
    if n not in _GL_CACHE:
        x, w = np.polynomial.legendre.leggauss(n)
        _GL_CACHE[n] = ((x + 1.0) / 2.0, w / 2.0)
    return _GL_CACHE[n]


# --------------------------------------------------------------------------------------------------------------------------------------
# Spleiß-Fall
# --------------------------------------------------------------------------------------------------------------------------------------
def spliced_threshold(zeta_u, zeta0):
    """Schwelle u zum Anteil ζ_u der Beobachtungen über u: im Körper aus ζ_u = (1−ζ₀)(u₀−u)/u₀ + ζ₀, im GPD-Teil aus ζ_u = ζ₀(1 + ξ(u−u₀)/σ)^(−1/ξ)."""
    if zeta_u > zeta0:
        return C.U0 - (zeta_u - zeta0) * C.U0 / (1.0 - zeta0)
    return C.U0 + C.SIG0 / C.XI0 * ((zeta_u / zeta0) ** (-C.XI0) - 1.0)


def pop_fit_spliced(u, zeta0):
    """Populations-ML (ξ, σ_u) der GPD an die wahre Überschreitungsverteilung des Spleiß-Falls über u; dazu der Anteil ζ_u der Beobachtungen über u.
    Oberhalb u₀ exakt GPD(ξ = 0,3, σ_u = 0,5 + 0,3 (u − u₀)); darunter Quadratur (Körper-Anteil gleichverteilt über (0, a), a = u₀ − u, Tail-Anteil GPD ab a)."""
    if u >= C.U0:
        return C.XI0, C.SIG0 + C.XI0 * (u - C.U0), zeta0 * (1.0 + C.XI0 * (u - C.U0) / C.SIG0) ** (-1.0 / C.XI0)
    a = C.U0 - u
    pb = (1.0 - zeta0) * a / C.U0
    zu = pb + zeta0
    wb, wt = pb / zu, zeta0 / zu
    sb, wsb = _gl(C.POP_NODES_BODY)
    yb = sb * a                                                  # Körper: gleichverteilt über (0, a)
    st, wst = _gl(C.POP_NODES_TAIL)
    jac = 4.0 * st ** 3                                           # Tail: Überschreitungsanteil s^4 (feinere Auflösung am Ende des Tails)
    yt = a + C.SIG0 / C.XI0 * (st ** (-4.0 * C.XI0) - 1.0)

    def nll(par):
        xi, sg = par
        if sg <= 0.0 or xi < 1e-6 or xi > 3.0:
            return 1e9

        def lh(y):
            return -math.log(sg) - (1.0 / xi + 1.0) * np.log1p(xi * y / sg)
        return -(wb * float(np.sum(wsb * lh(yb))) + wt * float(np.sum(wst * jac * lh(yt))))
    x, _f, _it = N.nelder_mead(nll, [C.XI0, C.SIG0], step=0.1, xatol=1e-9, fatol=1e-13, maxiter=4000)
    x, _f, _it = N.nelder_mead(nll, x, step=0.02, xatol=1e-10, fatol=1e-14, maxiter=4000)
    return float(x[0]), float(x[1]), zu


def spliced_row(k, n, zeta0, p):
    """Eine Zeile der Populationskurve des Spleiß-Falls bei k Überschreitungen von n: Schwelle, ξ_Pop, σ_u, σ*, Populationsfehler ln(Pegel/wahr), Mean Residual Life."""
    u = spliced_threshold(k / n, zeta0)
    xi, sg, zu = pop_fit_spliced(u, zeta0)
    lev = A.pot_level(xi, sg, u, zu, p)
    return {"k": int(k), "zeta": zu, "u": u, "xi": xi, "sigma_u": sg, "sigma_star": sg - xi * u,
            "ln_err": math.log(lev / K.spliced_level(p, zeta0)), "mrl": K.spliced_mrl(u, zeta0)}


def spliced_curve(n, zeta0, p, ks):
    return [spliced_row(k, n, zeta0, p) for k in ks]


# --------------------------------------------------------------------------------------------------------------------------------------
# Allgemeine Fälle: Quantil-Quadratur
# --------------------------------------------------------------------------------------------------------------------------------------
def gpd_population_ml(isf, zeta, n_s=C.POP_NODES_QUANTILE):
    """ML der GPD (ξ, σ_u) an die EXAKTE Überschreitungsverteilung: maximiert E[ln Dichte] per Gauß-Legendre über s (Überschreitungsanteil ζ·s⁴), kein Zufall.
    isf(q) = Quantil zum Überschreitungsanteil q (Feld). Gibt (ξ, σ_u, u) zurück."""
    u = float(isf(np.array([zeta]))[0])
    s, ws = _gl(n_s)
    w = ws * 4.0 * s ** 3                                        # dt = 4 s³ ds mit t = 1 − s⁴
    y = np.maximum(np.asarray(isf(zeta * s ** 4), float) - u, 1e-300)

    def nll(par):
        xi, ls = par
        sg = math.exp(ls)
        if abs(xi) < 1e-8:
            return float(np.sum(w * (np.log(sg) + y / sg)))
        t = 1.0 + xi * y / sg
        if np.any(t <= 0.0):
            return 1e12
        return float(np.sum(w * (np.log(sg) + (1.0 / xi + 1.0) * np.log(t))))
    best = None
    ls0 = math.log(float(np.sum(w * y)))
    for x0 in (0.1, 0.4, -0.1):
        x, f, _it = N.nelder_mead(nll, [x0, ls0], step=0.1, xatol=1e-10, fatol=1e-14, maxiter=6000)
        if best is None or f < best[1]:
            best = (x, f)
    x, _f, _it = N.nelder_mead(nll, best[0], step=0.02, xatol=1e-11, fatol=1e-15, maxiter=6000)
    return float(x[0]), math.exp(float(x[1])), u


def lognormal_isf(q):
    """Umkehrung der Überlebensfunktion der Lognormalverteilung (σ = 1): x mit P(X > x) = q."""
    return np.array([math.exp(-C.LOGNORMAL_SIGMA * _ND.inv_cdf(float(v))) for v in np.atleast_1d(q)])


def population_row(isf, zeta, p, x_true):
    """Zeile der Populationskurve zu einem Überschreitungsanteil ζ: ξ_Pop, σ_u, σ*, Schwelle und Populationsfehler ln(Pegel/wahr) bei p."""
    xi, sg, u = gpd_population_ml(isf, zeta)
    return {"zeta": zeta, "u": u, "xi": xi, "sigma_u": sg, "sigma_star": sg - xi * u, "ln_err": math.log(A.pot_level(xi, sg, u, zeta, p) / x_true)}


# --------------------------------------------------------------------------------------------------------------------------------------
# Geschlossene Fälle: GPD bei jeder Schwelle exakt
# --------------------------------------------------------------------------------------------------------------------------------------
EXACT_FAMILIES = {"Exponential": (0.0, 3.0), "Pareto-Verteilung (α = 2)": (1.0, 6.0), "Gleichverteilung": (0.0, 0.95)}


def exact_gpd_family(name, u):
    """(ξ, σ_u, σ*) bei der Schwelle u für die Fälle, in denen die Überschreitungsverteilung bei jeder Schwelle eine exakte GPD ist:
    Exponential(1): ξ = 0, σ_u = 1. Pareto-Verteilung (S(x) = x^(−2), x ≥ 1): ξ = 1/2, σ_u = u/2. Gleichverteilung (0, 1): ξ = −1, σ_u = 1 − u.
    σ* = σ_u − ξ·u ist in allen drei Fällen konstant (1, 0, 1); σ_u wächst nur bei ξ > 0, bleibt bei ξ = 0 und fällt bei ξ < 0."""
    if name == "Exponential":
        xi, sg = 0.0, 1.0
    elif name == "Pareto-Verteilung (α = 2)":
        xi, sg = 0.5, u / 2.0
    elif name == "Gleichverteilung":
        xi, sg = -1.0, 1.0 - u
    else:
        raise ValueError(name)
    return xi, sg, sg - xi * u
