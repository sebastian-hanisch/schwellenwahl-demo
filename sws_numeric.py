"""Kleine Numerik-Bausteine nur mit numpy und Standardbibliothek (scipy steht nur in den Tests der Linie):
Nelder-Mead für zwei Parameter, goldener Schnitt für eine Variable, Bisektion."""

import math

import numpy as np

_GOLD = (math.sqrt(5.0) - 1.0) / 2.0


def golden_min(f, a, b, xtol=1e-10, maxiter=200):
    """Minimum einer auf [a, b] unimodalen Funktion (goldener Schnitt); gibt (x, f(x)) zurück."""
    c, d = b - _GOLD * (b - a), a + _GOLD * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(maxiter):
        if abs(b - a) <= xtol:
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - _GOLD * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + _GOLD * (b - a)
            fd = f(d)
    x = 0.5 * (a + b)
    return x, f(x)


def nelder_mead(f, x0, step=0.1, xatol=1e-9, fatol=1e-13, maxiter=4000):
    """Nelder-Mead für eine Funktion von zwei oder mehr Parametern (Standardkoeffizienten 1, 2, 1/2, 1/2); gibt (x, f(x), Zahl der Iterationen) zurück.
    Abbruch, wenn der Simplex in x UND in f kleiner als die Toleranzen ist."""
    x0 = np.asarray(x0, float)
    n = x0.size
    simplex = [x0.copy()]
    for i in range(n):
        p = x0.copy()
        p[i] += step if p[i] == 0 else step * max(1.0, abs(p[i])) * 0.5
        simplex.append(p)
    vals = [f(p) for p in simplex]
    it = 0
    for it in range(1, maxiter + 1):
        order = sorted(range(n + 1), key=lambda i: vals[i])
        simplex = [simplex[i] for i in order]
        vals = [vals[i] for i in order]
        if max(float(np.max(np.abs(simplex[i] - simplex[0]))) for i in range(1, n + 1)) <= xatol and abs(vals[-1] - vals[0]) <= fatol:
            break
        centre = np.mean(simplex[:-1], axis=0)
        xr = centre + (centre - simplex[-1])
        fr = f(xr)
        if fr < vals[0]:
            xe = centre + 2.0 * (centre - simplex[-1])
            fe = f(xe)
            if fe < fr:
                simplex[-1], vals[-1] = xe, fe
            else:
                simplex[-1], vals[-1] = xr, fr
        elif fr < vals[-2]:
            simplex[-1], vals[-1] = xr, fr
        else:
            if fr < vals[-1]:
                xc = centre + 0.5 * (xr - centre)          # Kontraktion außen
            else:
                xc = centre + 0.5 * (simplex[-1] - centre)  # Kontraktion innen
            fc = f(xc)
            if fc < min(fr, vals[-1]):
                simplex[-1], vals[-1] = xc, fc
            else:
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    vals[i] = f(simplex[i])
    i = int(np.argmin(vals))
    return simplex[i], vals[i], it


def bisect(f, lo, hi, tol=1e-14, maxiter=200):
    """Nullstelle einer monotonen Funktion auf [lo, hi] (Vorzeichenwechsel vorausgesetzt)."""
    flo = f(lo)
    for _ in range(maxiter):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if (fm > 0) == (flo > 0):
            lo, flo = mid, fm
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)
