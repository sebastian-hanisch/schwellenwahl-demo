"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben.

Konvention der Achsen: k (Zahl der Überschreitungen) logarithmisch und umgekehrt (links niedrige Schwellen = viele Überschreitungen, rechts hohe Schwellen)."""

import math

import numpy as np
import plotly.graph_objects as go

import sws_constants as C


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=340):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.28), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def _k_axis(fig, title="k (Überschreitungen): links niedrige, rechts hohe Schwelle"):
    fig.update_xaxes(type="log", autorange="reversed", title_text=title)
    return fig


def _vline(fig, x, text, color):
    fig.add_vline(x=x, line=dict(color=color, width=1.5, dash="dot"), annotation_text=text, annotation_position="top", annotation_font_size=11)


def build_stability(ks, est, se, pop_k, pop_vals, ylabel, chosen_k=None, u0_k=None, name_est="Reihe: ML-Schätzung", name_pop="Population (ohne Rauschen)", band=None):
    """Stabilitätsdiagramm einer Größe (ξ oder σ*) gegen k: die Schätzwerte der Reihe (mit Band ± c·Standardfehler, falls se gegeben), die Populationskurve und Marken."""
    fig = go.Figure()
    ks = np.asarray(ks, float)
    est = np.asarray(est, float)
    if se is not None and band:
        hi, lo = est + band * np.asarray(se, float), est - band * np.asarray(se, float)
        fig.add_trace(go.Scatter(x=list(ks) + list(ks)[::-1], y=list(hi) + list(lo)[::-1], fill="toself", fillcolor=C.COLOR_BAND, line=dict(width=0),
                                 hoverinfo="skip", name=f"± {band:g} Standardfehler"))
    fig.add_trace(go.Scatter(x=ks, y=est, mode="lines+markers", line=dict(color=C.COLOR_RUN, width=2), marker=dict(size=7), name=name_est,
                             hovertemplate="k = %{x:.0f}: %{y:.3f}<extra></extra>"))
    if pop_k is not None:
        fig.add_trace(go.Scatter(x=pop_k, y=pop_vals, mode="lines", line=dict(color=C.COLOR_POP, width=2.5), name=name_pop, hovertemplate="k = %{x:.0f}: %{y:.3f}<extra></extra>"))
    if chosen_k is not None:
        _vline(fig, chosen_k, "gewählt", C.COLOR_TRUTH)
    if u0_k is not None:
        _vline(fig, u0_k, "u₀", "gray")
    fig.update_yaxes(title_text=ylabel)
    return _k_axis(_base(fig))


def build_mrl(us, es, ses, exact_u, exact_e, band, chosen_u=None, u0=None):
    """Mean-Residual-Life-Diagramm: e(u) der Reihe mit Band ± c·Standardfehler und die exakte Kurve."""
    us, es, ses = (np.asarray(v, float) for v in (us, es, ses))
    order = np.argsort(us)
    us, es, ses = us[order], es[order], ses[order]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(us) + list(us)[::-1], y=list(es + band * ses) + list(es - band * ses)[::-1], fill="toself", fillcolor=C.COLOR_BAND, line=dict(width=0),
                             hoverinfo="skip", name=f"± {band:g} Standardfehler"))
    fig.add_trace(go.Scatter(x=us, y=es, mode="lines+markers", line=dict(color=C.COLOR_RUN, width=2), marker=dict(size=7), name="Reihe: Mittel der Überschreitungen",
                             hovertemplate="u = %{x:.3f}: %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=exact_u, y=exact_e, mode="lines", line=dict(color=C.COLOR_POP, width=2.5), name="exakt e(u)", hovertemplate="u = %{x:.3f}: %{y:.3f}<extra></extra>"))
    if chosen_u is not None:
        _vline(fig, chosen_u, "gewählt", C.COLOR_TRUTH)
    if u0 is not None:
        _vline(fig, u0, "u₀", "gray")
    fig.update_xaxes(title_text="Schwelle u")
    fig.update_yaxes(title_text="e(u) = mittlere Überschreitung")
    return _base(fig)


def build_exact_families(rows_by_family):
    """σ_u (durchgezogen) und σ* (gestrichelt) gegen den Anteil ζ_u der Beobachtungen über der Schwelle (rechts hohe Schwelle) für die drei Fälle mit exakter GPD."""
    colors = {"Exponential": "#4c78a8", "Pareto-Verteilung (α = 2)": "#e49444", "Gleichverteilung": "#2F6B65"}
    fig = go.Figure()
    for name, rows in rows_by_family.items():
        z = [r["zeta"] for r in rows]
        fig.add_trace(go.Scatter(x=z, y=[r["sigma_u"] for r in rows], mode="lines", line=dict(color=colors[name], width=2.5), name=f"{name}: σ_u"))
        fig.add_trace(go.Scatter(x=z, y=[r["sigma_star"] for r in rows], mode="lines", line=dict(color=colors[name], width=2.5, dash="dash"), name=f"{name}: σ*"))
    fig.update_xaxes(type="log", autorange="reversed", title_text="ζ_u = Anteil der Beobachtungen über der Schwelle (rechts: hohe Schwelle)")
    fig.update_yaxes(title_text="Skala der Überschreitungen")
    return _base(fig, 320)


def build_error_curve(ks, pop_err, run_mean, run_q10, run_q90, rmse, u0_k=None):
    """Pegelfehler ln(Pegel/wahr) gegen k: Populationskurve (ohne Rauschen) und Läufe (Mittel, Q10 bis Q90 des Verhältnisses), darunter der RMSE der Läufe."""
    ks = np.asarray(ks, float)
    fig = go.Figure()
    if run_q10 is not None:
        fig.add_trace(go.Scatter(x=list(ks) + list(ks)[::-1], y=list(np.log(run_q90)) + list(np.log(run_q10))[::-1], fill="toself", fillcolor=C.COLOR_BAND, line=dict(width=0),
                                 hoverinfo="skip", name="Läufe: Q10 bis Q90"))
        fig.add_trace(go.Scatter(x=ks, y=run_mean, mode="lines+markers", line=dict(color=C.COLOR_RUN, width=2), marker=dict(size=6), name="Läufe: Mittel (Verzerrung)",
                                 hovertemplate="k = %{x:.0f}: %{y:+.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=ks, y=pop_err, mode="lines+markers", line=dict(color=C.COLOR_POP, width=2.5), marker=dict(size=6), name="Population (ohne Rauschen)",
                             hovertemplate="k = %{x:.0f}: %{y:+.3f}<extra></extra>"))
    fig.add_hline(y=0.0, line=dict(color="gray", width=1))
    if u0_k is not None:
        _vline(fig, u0_k, "u₀", "gray")
    fig.update_yaxes(title_text="ln(Pegel / wahrer Pegel)")
    return _k_axis(_base(fig))


def build_rmse_curve(ks, rmse, sign_change_between=None):
    """RMSE der Läufe je festem k; ein Vorzeichenwechsel der Verzerrung zwischen zwei Gitterwerten ist als Marke eingetragen."""
    fig = go.Figure(go.Scatter(x=ks, y=rmse, mode="lines+markers", line=dict(color=C.COLOR_RUN, width=2), marker=dict(size=6), name="RMSE je festem k",
                               hovertemplate="k = %{x:.0f}: %{y:.3f}<extra></extra>"))
    if sign_change_between is not None:
        fig.add_vrect(x0=sign_change_between[0], x1=sign_change_between[1], fillcolor="rgba(176,74,58,0.15)", line_width=0,
                      annotation_text="Vorzeichenwechsel der Verzerrung", annotation_position="top left", annotation_font_size=11)
    fig.update_yaxes(title_text="RMSE von ln(Pegel / wahr)", rangemode="tozero")
    return _k_axis(_base(fig, 260))


def build_choice_hist(ks, counts, rule_label, color, u0_k=None):
    """Verteilung der gewählten k (Zahl der Läufe je Stufe des Gitters)."""
    fig = go.Figure(go.Bar(x=[str(int(k)) for k in ks], y=counts, marker_color=color, name=rule_label, hovertemplate="k = %{x}: %{y} Läufe<extra></extra>"))
    fig.update_xaxes(title_text="gewähltes k (Stufe des Gitters; links niedrige, rechts hohe Schwelle)", type="category", categoryorder="array", categoryarray=[str(int(k)) for k in ks])
    fig.update_yaxes(title_text="Läufe")
    if u0_k is not None:
        fig.add_annotation(text=f"u₀ liegt bei k = {u0_k:.0f}", xref="paper", yref="paper", x=0.99, y=0.97, showarrow=False, font=dict(size=11))
    return _base(fig, 280)


def build_seven_rules(rows, truth):
    """Pegel der sieben Regeln auf derselben Reihe, mit dem wahren Pegel als Linie (logarithmische Achse)."""
    names = [f"{r[0]} (k = {r[1]})" for r in rows]
    levels = [r[2] for r in rows]
    colors = [C.RULE_COLORS.get(r[0][0], C.COLOR_MUTED) if r[0][0] in "SMG" and not r[0].startswith("fest") else C.COLOR_MUTED for r in rows]
    fig = go.Figure(go.Bar(x=names, y=levels, marker_color=colors, hovertemplate="%{x}: %{y:.2f}<extra></extra>", name="Pegel-Dach"))
    fig.add_hline(y=truth, line=dict(color=C.COLOR_TRUTH, width=2), annotation_text=f"wahrer Pegel {truth:.2f}", annotation_position="top left", annotation_font_size=11)
    fig.update_yaxes(title_text="Pegel (log)", type="log")
    fig.update_xaxes(title_text=None)
    return _base(fig, 320)


def safe_float(v, default=math.nan):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default
