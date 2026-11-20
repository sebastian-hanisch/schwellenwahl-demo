"""Orakel Population: wahre Pegel, exakte e(u), Populations-ML (Gauß-Legendre der App gegen verschachteltes `scipy.integrate.quad` und gegen Quadratur über s mit anderen Knoten),
Vehikel B mit anderem Rechenweg (Brent statt Tabelle) und die exakten Familien. scipy nur im Test."""

import math

import numpy as np
import pytest

import sws_cases as K
import sws_constants as C
import sws_population as PO
import sws_reference as REF
import sws_vehicle as V

integrate = pytest.importorskip("scipy.integrate")
optimize = pytest.importorskip("scipy.optimize")
stats = pytest.importorskip("scipy.stats")
special = pytest.importorskip("scipy.special")


# ----------------------------------------------------------------------------------------------------------------------------------
# wahrer Pegel: S(x_p) = p in jedem Fall, mit der Überlebensfunktion aus scipy
# ----------------------------------------------------------------------------------------------------------------------------------
def _surv_splice(x, zeta0):
    return (1 - zeta0) * (1 - x) + zeta0 if x < 1 else zeta0 * (1 + 0.3 * (x - 1) / 0.5) ** (-1 / 0.3)


@pytest.mark.parametrize("case_key", ["splice10", "splice03"])
def test_wahrer_pegel_spleiss_fall_gegenprobe_ueberlebensfunktion(case_key):
    zeta0 = C.SPLICE[case_key]
    xp = K.true_level(case_key)
    assert _surv_splice(xp, zeta0) == pytest.approx(C.P_TARGET[case_key], rel=1e-9)
    assert xp == pytest.approx({"splice10": 15.6322, "splice03": 10.6911}[case_key], abs=5e-4)


def test_wahrer_pegel_lognormal_exponential_vehikel_gegen_scipy():
    assert K.true_level("lognormal") == pytest.approx(stats.lognorm.isf(5e-5, 1.0), rel=1e-10)
    assert K.true_level("exponential") == pytest.approx(stats.expon.isf(5e-5), rel=1e-12)
    assert K.true_level("vehicle") == pytest.approx(287.158, abs=1e-3)


# ----------------------------------------------------------------------------------------------------------------------------------
# e(u) in geschlossener Form gegen Quadratur der Überlebensfunktion
# ----------------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("u", [0.0, 0.5, 0.9, 1.5, 3.0])
def test_e_des_spleiss_falls_gegen_quadratur(u):
    zeta0 = 0.1
    num = integrate.quad(lambda x: _surv_splice(x, zeta0), u, 1.0, limit=200)[0] if u < 1 else 0.0
    num += integrate.quad(lambda x: _surv_splice(x, zeta0), max(u, 1.0), np.inf, limit=400)[0]
    assert K.exact_mrl("splice10", u) == pytest.approx(num / _surv_splice(u, zeta0), rel=1e-8)


@pytest.mark.parametrize("u", [0.5, 2.0, 10.0, 60.0])
def test_e_lognormal_und_vehikel_gegen_quadratur(u):
    sf = lambda x: stats.lognorm.sf(x, 1.0)  # noqa: E731
    assert K.exact_mrl("lognormal", u) == pytest.approx(integrate.quad(sf, u, np.inf, limit=400)[0] / sf(u), rel=1e-7)
    assert K.exact_mrl("exponential", u) == 1.0
    w, mu, s = (np.array(c) for c in zip(*REF_TABLE()))
    sfb = lambda x: float(np.sum(w * stats.norm.sf((math.log(x) - mu) / s)))  # noqa: E731
    assert K.exact_mrl("vehicle", u) == pytest.approx(integrate.quad(sfb, u, 5000.0, limit=500, epsabs=1e-12, epsrel=1e-11)[0] / sfb(u), rel=1e-6)


def REF_TABLE():
    import json
    from pathlib import Path
    return json.loads((Path(__file__).resolve().parent.parent / "sws_mixture.json").read_text(encoding="utf-8"))["komponenten"]


# ----------------------------------------------------------------------------------------------------------------------------------
# Populations-ML des Spleiß-Falls: App (Gauß-Legendre 80 + 300) gegen verschachteltes quad (anderer Rechenweg)
# ----------------------------------------------------------------------------------------------------------------------------------
def _pop_fit_quad(k, n=2000, zeta0=0.1):
    u = PO.spliced_threshold(k / n, zeta0)
    a = 1.0 - u
    zu = (1 - zeta0) * a + zeta0
    wb, wt = (1 - zeta0) * a / zu, zeta0 / zu

    def lh(y, xi, sg):
        return -math.log(sg) - (1 / xi + 1) * math.log1p(xi * y / sg)

    def nll(par):
        xi, sg = par
        if sg <= 0 or xi < 1e-6 or xi > 3:
            return 1e9
        tb = integrate.quad(lambda y: lh(y, xi, sg), 0, a)[0] / a
        tt = integrate.quad(lambda y: (1 / 0.5) * (1 + 0.3 * (y - a) / 0.5) ** (-1 / 0.3 - 1) * lh(y, xi, sg), a, np.inf, limit=300)[0]
        return -(wb * tb + wt * tt)
    r = optimize.minimize(nll, [0.4, 0.3], method="Nelder-Mead", options={"xatol": 1e-7, "fatol": 1e-12, "maxiter": 2000})
    return r.x[0], r.x[1], u, zu


@pytest.mark.parametrize("k", [235, 289, 800])
def test_populations_ml_spleiss_gegen_verschachteltes_quad(k):
    xi, sg, u, zu = _pop_fit_quad(k)
    row = PO.spliced_row(k, 2000, 0.1, 5e-5)
    assert row["xi"] == pytest.approx(xi, abs=3e-3)
    err = math.log((u + sg / xi * ((zu / 5e-5) ** xi - 1)) / K.spliced_level(5e-5, 0.1))
    assert row["ln_err"] == pytest.approx(err, abs=0.02)
    # Gegenprobe: der Populationsfehler erklärt die Simulationstabelle der Vormessung (Abstand höchstens 0,1, endliches k und Rauschen)
    sim = {235: 0.805, 289: 2.048, 800: -0.313}[k]
    assert abs(row["ln_err"] - sim) < 0.1


def test_populations_ml_bleibt_bei_anderer_knotenzahl_stabil(monkeypatch):
    base = PO.spliced_row(289, 2000, 0.1, 5e-5)["ln_err"]
    monkeypatch.setattr(C, "POP_NODES_BODY", 160)
    monkeypatch.setattr(C, "POP_NODES_TAIL", 500)
    assert PO.spliced_row(289, 2000, 0.1, 5e-5)["ln_err"] == pytest.approx(base, abs=2e-4)


def test_populations_ml_oberhalb_u0_exakt_gpd():
    for k in (25, 100, 192):
        row = PO.spliced_row(k, 2000, 0.1, 5e-5)
        assert row["xi"] == 0.3 and row["sigma_star"] == pytest.approx(0.2, abs=1e-12) and row["ln_err"] == pytest.approx(0.0, abs=1e-12)


def test_populations_ml_ist_das_maximum_der_log_dichte():
    """Eigenschaft statt Nachrechnung: an der Lösung ist der Erwartungswert der Log-Dichte größer als an gestörten Parametern (Spleiß-Fall, k = 289)."""
    u = PO.spliced_threshold(289 / 2000, 0.1)
    a = 1.0 - u
    zu = 0.9 * a + 0.1
    wb, wt = 0.9 * a / zu, 0.1 / zu

    def val(xi, sg):
        lh = lambda y: -math.log(sg) - (1 / xi + 1) * math.log1p(xi * y / sg)  # noqa: E731
        tb = integrate.quad(lh, 0, a)[0] / a
        tt = integrate.quad(lambda y: 2.0 * (1 + 0.3 * (y - a) / 0.5) ** (-1 / 0.3 - 1) * lh(y), a, np.inf, limit=300)[0]
        return wb * tb + wt * tt
    xi, sg, _zu = PO.pop_fit_spliced(u, 0.1)
    best = val(xi, sg)
    for dx, ds in ((0.02, 0), (-0.02, 0), (0, 0.02), (0, -0.02), (0.02, 0.02)):
        assert val(xi + dx, sg + ds) < best


# ----------------------------------------------------------------------------------------------------------------------------------
# allgemeine Quadratur: Lognormal gegen scipy-Quadratur über s mit anderen Knoten; Vehikel B gegen Brent (H 6)
# ----------------------------------------------------------------------------------------------------------------------------------
def test_populations_ml_lognormal_gegen_quad_ueber_s():
    zeta = 0.1
    u = float(stats.lognorm.isf(zeta, 1.0))

    def nll(par):
        xi, ls = par
        sg = math.exp(ls)

        def integrand(s):
            y = float(stats.lognorm.isf(zeta * s ** 4, 1.0)) - u
            return 4 * s ** 3 * (math.log(sg) + (1 / xi + 1) * math.log1p(xi * y / sg))
        return integrate.quad(integrand, 0.0, 1.0, limit=200)[0]
    r = optimize.minimize(nll, [0.3, 0.0], method="Nelder-Mead", options={"xatol": 1e-8, "fatol": 1e-12})
    xi, sg, u_app = PO.gpd_population_ml(PO.lognormal_isf, zeta)
    assert u_app == pytest.approx(u, rel=1e-12)
    assert xi == pytest.approx(r.x[0], abs=3e-3) and sg == pytest.approx(math.exp(r.x[1]), rel=3e-3)


def _vehicle_pop_brent(zeta, n_s=80):
    w, mu, s = (np.array(c) for c in zip(*REF_TABLE()))

    def level(p):
        f = lambda lx: float(special.logsumexp(np.log(w) + special.log_ndtr(-(lx - mu) / s))) - math.log(p)  # noqa: E731
        return math.exp(optimize.brentq(f, -2.0, 14.0, xtol=1e-13))
    u = level(zeta)
    sg_, wg_ = np.polynomial.legendre.leggauss(n_s)
    s_ = (sg_ + 1) / 2
    w_ = wg_ / 2 * 4 * s_ ** 3
    y = np.array([level(zeta * si ** 4) for si in s_]) - u

    def nll(par):
        xi_, ls_ = par
        sg = math.exp(ls_)
        t_ = 1 + xi_ * y / sg
        if xi_ < 1e-6 or np.any(t_ <= 0):
            return 1e12
        return float(np.sum(w_ * (np.log(sg) + (1 / xi_ + 1) * np.log(t_))))
    r = optimize.minimize(nll, [0.3, math.log(float(np.sum(w_ * y)))], method="Nelder-Mead", options={"xatol": 1e-9, "fatol": 1e-13, "maxiter": 4000})
    return float(r.x[0]), math.exp(float(r.x[1])), u


@pytest.mark.parametrize("k", [25, 522])
def test_populationsfehler_vehikel_b_mit_brent_statt_tabelle(k):
    zeta = k / 5000
    xi, sg, u = _vehicle_pop_brent(zeta)
    err = math.log((u + sg / xi * ((zeta / 1e-5) ** xi - 1)) / V.level(1e-5))
    assert REF.population_interp("vehicle", zeta, "ln_err") == pytest.approx(err, abs=0.012)
    assert REF.population_interp("vehicle", zeta, "xi") == pytest.approx(xi, abs=0.01)
    assert err == pytest.approx({25: 0.065, 522: 0.812}[k], abs=0.012)


# ----------------------------------------------------------------------------------------------------------------------------------
# exakte Familien: bei jeder Schwelle GPD
# ----------------------------------------------------------------------------------------------------------------------------------
def test_exakte_familien_sind_bei_jeder_schwelle_eine_gpd():
    cases = {
        "Exponential": (lambda x: stats.expon.sf(x), [0.5, 1.0, 3.0]),
        "Pareto-Verteilung (α = 2)": (lambda x: x ** -2.0, [1.0, 1.5, 4.0]),
        "Gleichverteilung": (lambda x: 1.0 - x, [0.1, 0.5, 0.9]),
    }
    for name, (sf, us) in cases.items():
        for u in us:
            xi, sg, sstar = PO.exact_gpd_family(name, u)
            for y in (0.05, 0.2, 0.4):
                if name == "Gleichverteilung" and y >= 1 - u:
                    continue
                assert sf(u + y) / sf(u) == pytest.approx(1 - float(stats.genpareto.cdf(y, xi, scale=sg)), rel=1e-12), (name, u, y)
            assert sstar == pytest.approx(sg - xi * u, abs=1e-15)
    assert {n: PO.exact_gpd_family(n, 0.5)[2] for n in cases} == {"Exponential": 1.0, "Pareto-Verteilung (α = 2)": 0.0, "Gleichverteilung": 1.0}


def test_sigma_u_waechst_nur_bei_xi_positiv_bleibt_bei_null_faellt_bei_negativ():
    a = [PO.exact_gpd_family("Exponential", u)[1] for u in (0.5, 1.0, 2.0)]
    b = [PO.exact_gpd_family("Pareto-Verteilung (α = 2)", u)[1] for u in (1.0, 2.0, 4.0)]
    c = [PO.exact_gpd_family("Gleichverteilung", u)[1] for u in (0.1, 0.5, 0.9)]
    assert a == [1.0, 1.0, 1.0] and b == sorted(b) and b[0] < b[-1] and c == sorted(c, reverse=True) and c[0] > c[-1]
