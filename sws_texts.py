"""Texte der App; `ASSUMPTIONS` und `VEHIKEL_TEXT` stehen wörtlich auch in der README (`tests/test_claims.py` vergleicht beide Fassungen)."""

TITLE = "🎚️ Schwellenwahl: wo ist die Schwelle hoch genug?"

INTRO = """
Demo der Linie „Extremwerte erklärt“. Eine **Schwelle** muss **hoch genug** liegen, damit die Überschreitungen einer **verallgemeinerten Pareto-Verteilung (GPD)** folgen, und **tief genug**,
damit genug Werte bleiben. Die Demo zeigt an Reihen mit **bekannter Wahrheit** (ein Spleiß-Fall als Konstruktion, Lognormal, Exponential und die ausgedachte Verweilzeit-Mischung
„Vehikel B“), **wo der Pegel kippt, wenn die Schwelle zu tief liegt**, und was **drei eigene Regeln** (S, M, G; **keine Literaturverfahren**) im Vergleich zu einem fest gewählten **k** leisten.

Weiter unten: die Kernfragen, vier **🔬 Experimente**, der Abschnitt **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""

HOW_TITLE = "So funktioniert die Schwellenwahl"
HOW = """
1. **Schwelle und k.** Die n Werte werden sortiert; die **k größten** sind die Überschreitungen, die Schwelle u ist der (k+1)-größte Wert. Je kleiner k, desto höher die Schwelle.
2. **GPD anpassen und Pegel lesen.** An die k Überschreitungen y = x − u wird eine GPD (Extremwertindex **ξ**, Skala **σ**) per Maximum Likelihood angepasst; daraus folgt der **Pegel**: der Wert, den eine Beobachtung nur mit der Wahrscheinlichkeit p überschreitet.
3. **Gitter und Regeln.** Über 18 Schwellen (k von 25 bis 0,4·n, höchstens 1000, logarithmisch verteilt) wird die Anpassung wiederholt. Drei **eigene Regeln** wählen daraus eine Stufe: **S** (ξ̂ ist ab dieser Schwelle stabil), **M** (die mittlere Überschreitung e(u) liegt auf einer Geraden), **G** (ein Anpassungstest besteht). Zum Vergleich stehen feste k.
4. **Warum es auf die Schwelle ankommt.** Gilt die GPD nur oberhalb einer Grenze u₀, verzerrt eine tiefere Schwelle den Pegel in der Regel (wo der Fehler sein Vorzeichen wechselt, ist er zufällig nahe null); bei glatten Verteilungen driftet der Fehler stetig. Hier ist die Wahrheit bekannt, deshalb sieht man **ln(Pegel-Dach / wahrer Pegel)** und kann zählen, wie oft eine Regel richtig liegt.
"""

KERNFRAGEN = """
1. **Woran erkennt man, dass die Schwelle zu tief liegt?** → Experiment 1: Stabilitätsdiagramm und Mean-Residual-Life-Diagramm gegen die exakten Kurven.
2. **Wie schlimm wird der Pegel, wenn sie zu tief liegt?** → Experiment 2: Fehler des Pegels gegen k, ohne Rauschen (Population) und mit Rauschen (Läufe).
3. **Findet eine Regel die Schwelle, und besser als ein fester Wert?** → Experiment 3: drei Regeln gegen feste k, in Läufen gezählt.
4. **Wie weit streuen verschiedene Regeln auf denselben Daten?** → Experiment 4: sieben Pegel auf einer Reihe.
"""

EXP1_HELP = ("Oben: ξ̂ und σ̂\\* der Reihe gegen k mit dem Band ± c Standardfehler, dazu die **exakte** Populationskurve (kein Rauschen). Darunter e(u) gegen u mit der exakten Kurve. "
             "Die Marke „gewählt“ zeigt die Stufe der Regel aus der Seitenleiste, „u₀“ im Spleiß-Fall die Stelle, ab der die GPD exakt gilt.")

EXP2_HELP = ("Fehler ln(Pegel-Dach / wahrer Pegel) gegen k: grün die **Population** (die GPD, die die wahre Überschreitungsverteilung am besten beschreibt, per Quadratur, ohne Rauschen), "
             "blau die **Läufe** (eingefrorene Messung: Mittel und Bereich Q10 bis Q90 des Verhältnisses).")

EXP3_HELP = ("Eingefrorene Messung (Zähleinheit **Läufe**), gilt für die in der Tabelle genannten n unabhängig vom Regler n. Regeln und Gitter sind **eigene Festlegungen**. "
             "„Brauchbar“ heißt: der Populationsfehler des Pegels ist für die gewählte und alle höheren Stufen höchstens 0,05 bzw. 0,10 (eigene Festlegung); im Spleiß-Fall zählt stattdessen, "
             "ob die gewählte Schwelle bei oder über u₀ liegt.")

EXP4_HELP = ("Eine Reihe live, sieben Pegel: die Regeln S und M (c = 2), G und die festen k obere 2 %, 5 %, 10 % und √n. Der wahre Pegel ist exakt bekannt. "
             "Die Statistik über viele Reihen ist eingefroren.")

VEHIKEL_TEXT = ("Die Verweilzeiten und der Spleiß-Fall sind ausgedacht; es gibt in den gelesenen Quellen keine reale Datenquelle aus Hafen, Lager oder Wartezeiten. "
                "Gezeigt wird, wie die Wahl der Schwelle in diesem Modell wirkt, und dass die drei geprüften Regeln (eigene Festlegungen, keine Literaturverfahren) im Vehikel B (alle Regeln) und im Spleiß-Fall (S und M) meist nicht besser sind "
                "als ein fest gewähltes k, dass G im Spleiß-Fall ohne erkennbaren Vorsprung vor dem besten festen k bleibt und dass M und G bei den glatten Verteilungen (Lognormal, Exponential) besser sind. Was das für einen echten Betrieb heißt, wird hier nicht behauptet.")

ASSUMPTIONS = """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Unabhängige Beobachtungen** | Serielle Abhängigkeit und gerundete Werte sind hier **nicht gemessen**, ebenso Hill und Schwellenwahl unter Abhängigkeit. Die Literatur (nach Zusammenfassung, ungeprüft) nennt einen positiven Bias der Quantile; die Demo „Spitzen-Cluster“ der Linie (`spitzen-cluster-demo`) misst am M/M/1-Modell bei fester Schwelle einen negativen (bei Last 80 %, n = 3 000, k ≈ 150 und p = 10⁻³ in 90 % (89,6 %) von R = 2 000 Läufen ein Pegel unter dem wahren): wo beides nicht übereinstimmt, gilt deren Messung. | diese Demo zählt nicht, die Demo „Spitzen-Cluster“ der Linie |
| **Ausgedachte Fälle** | Der Spleiß-Fall ist eine **Konstruktion mit Dichtesprung** (Hazard der Überschreitungen 9 gegen 2 bei u₀): sein Kippen ist ein Extremfall, bei glatten Verteilungen driftet der Fehler. Vehikel B ist ausgedacht. | – |
| **Drei eigene Regeln** | S, M und G sind eigene Festlegungen mit diesen Gittern (18 Stufen), keine Literaturverfahren (ForwardStop, Bader u. a., Northrop u. a., Wadsworth/Tawn, Murphy u. a. nicht nachgebaut): die Ergebnisse gelten für diese Regeln, nicht für die Methodenklasse. G ist auf ξ ∈ {0; 0,3} bzw. {0; 0,5} kalibriert und ohne Mehrfachtest-Korrektur. | – |
| **n = 2 000 und 5 000** | Die eingefrorenen Messungen gelten nur für diese Reihenlängen, je Fall ein p (5·10⁻⁵ bzw. 10⁻⁵ je Beobachtung). | – |
| **Simulieren gegen Schätzen** | Hier ist die Wahrheit bekannt, deshalb lässt sich der Fehler zählen; bei echten Daten fehlt sie. Aussagen über den Nutzen in Logistik, Lager oder Hafen werden nicht gemacht. | – |
"""

MATH_LESEHILFE = """
**Lesehilfe.** *Gegeben:* n unabhängige Beobachtungen X₁, …, Xₙ. *Gesucht:* ein Pegel x_p, den eine Beobachtung nur mit der Wahrscheinlichkeit p überschreitet, und dazu eine Schwelle u, ab der die Überschreitungen einer GPD folgen.

| Symbol | Bedeutung |
|---|---|
| n, k | Zahl der Beobachtungen; Zahl der Überschreitungen (die k größten Werte) |
| u | Schwelle = (k+1)-größter Wert; höhere Schwelle heißt kleineres k |
| ζ_u = k/n | Anteil der Beobachtungen über der Schwelle |
| ξ, σ_u | Extremwertindex (nie mit dem Extremalindex θ vermischt) und Skala der GPD bei Schwelle u |
| σ\\* | modifizierte Skala σ_u − ξ·u, bleibt beim Anheben der Schwelle gleich, wenn die GPD gilt |
| e(u) | mittlere Überschreitung E[X − u ∣ X > u] (Mean Residual Life) |
| p, x_p | Überschreitungswahrscheinlichkeit je Beobachtung und der zugehörige Pegel |
| c | Bandbreite der Regeln S und M in Standardfehlern (1,5 / 2 / 3) |
| D | Kolmogorov-Smirnov-Abstand der Überschreitungen zur angepassten GPD |
"""

MATH_BODY = r"""
**Modell.** Oberhalb einer Schwelle u₀ gilt für die Überschreitungen y = X − u (u ≥ u₀) die verallgemeinerte Pareto-Verteilung; ξ > 0 ist Fréchet-Typ, ξ < 0 Weibull-Typ (`scipy.stats.genpareto`, c = ξ):

$$ G_{\xi,\sigma_u}(y) = 1 - \left(1 + \frac{\xi\, y}{\sigma_u}\right)^{-1/\xi}, \qquad \sigma_u = \sigma^{*} + \xi\, u $$

Der **Pegel** zur Überschreitungswahrscheinlichkeit p je Beobachtung (nur für p < ζ_u):

$$ x_p = u + \frac{\sigma_u}{\xi}\left(\left(\frac{\zeta_u}{p}\right)^{\xi} - 1\right) $$

Die mittlere Überschreitung einer GPD ist linear in u (Steigung ξ/(1 − ξ), nur für ξ < 1):

$$ e(u) = \frac{\sigma_u}{1 - \xi} = \frac{\sigma^{*} + \xi\, u}{1 - \xi} $$

**Spleiß-Fall** (Konstruktion): Körper gleichverteilt auf (0, 1) mit Masse 1 − ζ₀, ab u₀ = 1 exakt GPD(ξ = 3/10, σ = 1/2) mit Masse ζ₀; der wahre Pegel ist

$$ x_p = u_0 + \frac{\sigma}{\xi}\left(\left(\frac{p}{\zeta_0}\right)^{-\xi} - 1\right) $$

Die Dichte der Beobachtung springt bei u₀ von (1 − ζ₀) auf ζ₀/σ; unterhalb u₀ ist die Überschreitungsverteilung keine GPD.

**Population ohne Rauschen.** Die GPD (ξ, σ_u), die die wahre Überschreitungsverteilung F_u am besten beschreibt, maximiert den Erwartungswert der Log-Dichte g:

$$ (\xi_{\mathrm{Pop}}, \sigma_{\mathrm{Pop}}) = \arg\max_{\xi,\sigma} \int_0^\infty \ln g_{\xi,\sigma}(y)\, dF_u(y) $$

Das Integral wird per Gauß-Legendre-Quadratur gerechnet (kein Zufall); der Populationsfehler ist ln(x_p,Pop / x_p).

**Anpassung der Reihe.** Maximum Likelihood in einer Variablen: mit t = ξ ȳ/σ ist ξ(t) = mittleres ln(1 + t yᵢ/ȳ), und die Profil-Log-Likelihood ist k ln(t/ξ(t)) − k ln ȳ − k (ξ(t) + 1); ξ ≥ −1 wird verlangt.

**Die drei Regeln** (eigene Festlegungen) auf dem Gitter k₁ > k₂ > … > k₁₈ (Index 0 = niedrigste Schwelle):

$$ \text{S: } j^{*} = \min\Big\{ j : |\hat\xi_h - \hat\xi_j| \le c\,\frac{\max(1 + \hat\xi_h,\, 0{,}3)}{\sqrt{k_h}} \ \text{ für alle } h > j \Big\}, \quad j \le 14 $$

(Standardfehler der jeweils höheren Schwelle h; geprüft werden nur Schwellen mit mindestens drei höheren Nachbarn; ohne Treffer die höchste Schwelle.)

$$ \text{M: } j^{*} = \min\Big\{ j : |\hat e(u_h) - (a + b\,u_h)| \le c\,\mathrm{se}_h \ \text{ für alle } h \ge j \Big\}, \quad \mathrm{se}_h = \frac{s_h}{\sqrt{k_h}} $$

(Gerade (a, b) nach gewichteten kleinsten Quadraten mit dem Gewicht 1/se_h² angepasst; mindestens fünf Stufen, geprüft werden j ≤ 13; ohne Treffer der Index 13 = J − 5.)

$$ \text{G: } j^{*} = \min\big\{ j : \sqrt{k_j}\, D_j \le q_{0{,}9}(k_j) \big\}, \quad D_j = \sup_y |\hat F_j(y) - G_{\hat\xi_j, \hat\sigma_j}(y)| $$

(q₀,₉ = 90-%-Wert von √k·D unter H₀ für exakte GPD-Stichproben mit ξ ∈ {0; 0,3} bzw. {0; 0,5}, je Schwelle einzeln, ohne Mehrfachtest-Korrektur; ohne Treffer die höchste Schwelle.)

**Feste k:** obere 2 %, 5 %, 10 % und die Faustregeln k = √n und k = n^(2/3)/ln ln n (nach Scarrott/MacDonald laut Quellenbericht, ungeprüft).

*Skizze, kein vollständiges Modell:* Die Formeln nennen die Größen und Regeln; nicht ausgeschrieben sind die numerische Maximierung (Gitter plus goldener Schnitt), die Quadraturknoten und die Rückfälle bei weniger als fünf Stufen (M: Stufe 0, S: höchste Stufe).

**Literatur** (nur Lehrbuch oder Übersicht, keine Neuheit): Coles (2001), Abschnitt 4.3 (Bias gegen Varianz, Mean-Residual-Life-Diagramm, Stabilitätsdiagramm; gelesen); Embrechts, Klüppelberg, Mikosch (1997), Satz 3.4.13 (e(u) der GPD; gelesen); Scarrott und MacDonald (2012, Überblick über Schwellenwahl; nach dem Quellenbericht gelesen, hier nicht erneut geprüft); Belzile und Davison (2026, arXiv 2606.28540; nur nach Zusammenfassung, ungeprüft).
"""
