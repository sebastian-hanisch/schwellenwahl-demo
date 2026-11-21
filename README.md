# Schwellenwahl: wo ist die Schwelle hoch genug? – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-schwellenwahl-demo.streamlit.app/)**

Demo der Linie **„Extremwerte erklärt“** in der „Konzepte“-Reihe der Website „Sebastian Hanisch – Operations Research und Machine Learning“ (Bereich Statistik, Feld „Seltenes und Extremes“). Die Kante zur Demo davor („Überschreitungen und die GPD“) heißt `fix`: Dort ist die **Schwelle gegeben** (der Anteil oder die Zahl k der Überschreitungen), und die verallgemeinerte Pareto-Verteilung gilt oberhalb von ihr. Diese Demo hebt die Annahme auf: **die Schwelle muss aus den Daten gewählt werden**, und eine zu tiefe Schwelle verzerrt den Pegel, ein fester Wert zeigt das nicht. Gezeigt wird das an Reihen mit **bekannter Wahrheit**: ein Spleiß-Fall (Konstruktion mit Dichtesprung), Lognormal, Exponential und die ausgedachte Verweilzeit-Mischung „Vehikel B“.

Die Demo misst **drei eigene Regeln** (S: Stabilität, M: Mean Residual Life, G: Anpassungstest; **eigene Festlegungen, keine Literaturverfahren**) gegen ein fest gewähltes k, jeweils in Läufen gezählt, und zeigt, wie weit sieben verschiedene Regeln auf derselben Reihe auseinanderliegen.

## Warum dieses Problem

Aus den k größten von n Werten wird eine GPD angepasst und daraus ein **Pegel** gelesen: der Wert, den eine Beobachtung nur mit der Wahrscheinlichkeit p überschreitet. Liegt die Schwelle zu hoch, bleiben wenige Werte und die Schätzung streut; liegt sie zu tief, gilt die GPD dort nicht mehr, und der Pegel ist verzerrt. Hier wird gezählt, **wie stark** und **wo**: im Spleiß-Fall kippt der Pegel genau dort, wo die Schwelle unter die Grenze u₀ fällt, in der glatten Welt driftet der Fehler stetig, und die Regeln, die „die richtige Schwelle“ wählen sollen, sind in diesen Fällen bei S und M meist nicht besser als ein fest gewähltes k und bei G im Spleiß-Fall nicht erkennbar besser als das beste feste k; bei den glatten Verteilungen sind M und G besser als die festen k (Abschnitt C).

**Kernfragen:** (1) Woran erkennt man, dass die Schwelle zu tief liegt? (2) Wie schlimm wird der Pegel, wenn sie zu tief liegt? (3) Findet eine Regel die Schwelle, und besser als ein fester Wert? (4) Wie weit streuen verschiedene Regeln auf denselben Daten?

## Befunde und Korrekturen

Gegenüber dem Plan und den Vormessungen (`numpy.default_rng`, andere Zufalls-Seeds, kleinere Kalibrierung): alle Zufallsanteile sind hier mit SplitMix64 **neu gemessen**; die exakten Rechnungen (Brüche, Quadratur, Aufzählung von Vehikel B) sind unverändert.

* **Regel G, Kalibrierung.** Die Vormessung kalibrierte mit 200 exakten GPD-Stichproben je k und ξ; das gab eine Spanne der kritischen Werte von 0,810 bis 0,869 (ξ ∈ {0; 0,3}). Ein Nachlauf mit 200 Stichproben je k und ξ hier gab 0,794 bis 0,853, und die Regel G wählte im Spleiß-Fall mit Anteil 3 % in 53,3 % der Läufe eine Schwelle bei oder über u₀ (mit 1 000 Stichproben bei gleichem Zufallsstrom 51,5 %): **die Regel G reagiert auf ihre Kalibrierung, bei gleichem Zufallsstrom aber nur um 1,8 Punkte** (der Standardfehler eines Anteils bei R = 400 liegt bei rund 2,5 Punkten). Der Abstand zum Plan (46,0 %) mischt Kalibrierung und Zufallsstrom und lässt sich nicht der Kalibrierung allein zuschreiben. Die App benutzt, um das Rauschen der Kalibrierung zu verkleinern, 1 000 Stichproben je k und ξ (34 verschiedene k): Spanne 0,814 bis 0,862 (ξ ∈ {0; 0,3}) und 0,805 bis 0,841 (ξ ∈ {0; 0,5}), dazu in den Läufen derselbe Zufallsstrom wie vorher, nur die Kalibrierung anders (Messung dieser README, nicht im Test): G wählt bei Anteil 10 % in 70,2 % (Plan 74,2 %) und bei Anteil 3 % in 51,5 % (Plan 46,0 %) eine Schwelle bei oder über u₀; mit der Kalibrierung ohne den wahren Wert (G2, ξ ∈ {0; 0,5}) 71,2 % und 51,5 %.
* **Spleiß-Fall, Anteil 10 %** (R = 400, n = 2000): Anteil der Läufe mit gewählter Schwelle bei oder über u₀: S (c = 1,5 / 2 / 3) 71,0 / 50,0 / 24,5 % (Plan 67,2 / 51,7 / 27,0 %), M (c = 1,5 / 2 / 3) 3,8 / 0,2 / 0,0 % (Plan 3,5 / 0,8 / 0,0 %); RMSE von G 0,42 (Plan 0,41). S (c = 2) liegt in der Vormessung V2 (49,5 %), im Plan (51,7 %) und in den drei Läufen hier (50,0 / 51,5 / 50,5 %) bei 49,5 bis 51,7 %: ein Band um 50 %.
* **Spleiß-Fall, Anteil 3 %:** S 45,2 / 30,0 / 16,2 % (Plan 43,8 / 31,0 / 15,0 %), M durchgehend 0,0 %; das feste k = 100 hat den RMSE 6,038 (Plan 6,015) und den Median des Verhältnisses 396 (Plan 364).
* **Vehikel B** (R = 300, n = 5000): M (c = 2) und G wählen in 88 % bzw. 82 % der Läufe die niedrigste Stufe (Plan 89 und 82 %); der RMSE des besten Gitterpunkts ist 0,54 bei k = 114 (Plan 0,501 bei k = 176: das Minimum liegt flach zwischen k = 92 und 176, ein anderer Zufallsstrom verschiebt es); die Regel-Varianten haben RMSE 0,65 bis 0,85 (Plan 0,627 bis 0,837), die festen k = 71 bis 250 0,54 bis 0,59 (Plan 0,517 bis 0,546), das feste k = 500 0,85 (Plan 0,836). **„Keine Regel schlägt ein festes k = 71 bis 250“ bleibt, aber als Aussage mit Rauschgrenze** (Unterschied höchstens zwei Standardfehler, Standardfehler des RMSE rund RMSE/√(2R) = 0,02): gegen k = 500 sind **nur die Regeln S** um mehr als zwei Standardfehler besser, M und G liegen im Rauschen (der Plan zählte G mit 0,831 gegen 0,836 noch als „nicht besser“, hier 0,843 gegen 0,845: gleich).
* **Streuung der sieben Regeln auf derselben Reihe** (Vehikel B, R = 300): Median 2,76, Q10/Q90 1,58/5,81, Maximum 10,9; über Faktor 2 in 72,3 %, über Faktor 4 in 23,3 % der 300 Läufe; der wahre Pegel liegt zwischen niedrigstem und höchstem der sieben in 59,7 % (Plan: Median 2,66, Q10/Q90 1,74/5,51, Maximum 12,8, über Faktor 2 in 80,0 %, über 4 in 24,0 %, 63,7 %). Der Anteil über Faktor 2 ist ein weicher Rand: **72,3 % statt 80,0 %**; zwei Wiederholungen mit anderen Zufalls-Seed-Basen (Messung dieser README, nicht im Test) geben 78,7 und 75,7 %, den Median 3,01 und 2,72, über Faktor 4 in 30,0 und 20,0 %.
* **Wiederholungen** (je R wie oben, Seed-Basis 7000 / 8000 / 9000; Messung dieser README, nicht im Test): Spleiß 10 %: S (c = 2) 50,0 / 51,5 / 50,5 %, M (c = 2) 0,2 / 0,8 / 0,0 %, G 70,2 / 68,8 / 66,5 %, RMSE von G 0,425 / 0,441 / 0,421 gegen 0,433 / 0,463 / 0,434 bei festem k = 200; Spleiß 3 %: S (c = 2) 30,0 / 32,2 / 34,2 %, G 51,5 / 45,8 / 50,7 %; Vehikel B: niedrigste Stufe bei M (c = 2) 88 / 84 / 92 %, bei G 82 / 81 / 76 %, RMSE von G 0,843 / 0,868 / 0,825 gegen 0,544 / 0,546 / 0,526 bei festem k = 100, das RMSE-Minimum der festen k bei k = 114 / 176 / 176. **„G ist besser als die festen k“ lässt sich daraus nicht lesen:** G und das feste k = 200 liegen im Spleiß-Fall mit Anteil 10 % im Rauschen (Standardfehler des RMSE bei R = 400 rund 0,015 bis 0,02).
* **Regel M, Gewichte.** Die Gerade der Regel M ist eine gewichtete Anpassung mit `numpy.polyfit(w = 1/Standardfehler)`; das sind kleinste Quadrate mit dem Gewicht 1/Standardfehler² (nicht 1/Standardfehler). Der Plan sagt „Gewicht 1/Standardfehler“ und meint den Aufruf der Vormessung; der Orakel-Test rechnet die Gerade über die Normalgleichungen mit dem Gewicht 1/Standardfehler² nach.
* **σ_u wächst nicht immer.** Der Plan nennt für Exponential, Pareto-Verteilung und Gleichverteilung „ξ und σ* bleiben, σ_u wächst“. Gerechnet: σ_u = σ* + ξ·u wächst nur bei ξ > 0 (Pareto-Verteilung), bleibt bei ξ = 0 (Exponential) und **fällt** bei ξ < 0 (Gleichverteilung). Experiment 1 zeigt es so.
* **Zusatz zum Plan:** Lognormal und Exponential sind mit allen Regeln und festen k mitgemessen (der Plan hatte dort nur die Vormessung V2 mit S und M bei c = 2); die Zahlen stehen in der Tabelle der Experimente, nicht in den Verdikt-Ankern des Plans. Der Regler „Anteil ζ₀“ ist in den Fall eingegangen (Spleiß-Fall mit 10 % und mit 3 % als zwei Fälle), das „Fehlermaß“ von Experiment 2 zeigt beide Kurven (Population und Läufe) zugleich.
* **Unverändert gerechnet und gleich:** wahre Pegel (15,6322 (Spleiß 10 %), 10,6911 (Spleiß 3 %), 48,9398 (Lognormal), 9,9035 (Exponential), 287,158 Tage (Vehikel B)), die Populationsfehler des Spleiß-Falls (Tabelle unten), e(u) in Brüchen, die Populationskurve von Vehikel B (+0,065 / +0,133 / +0,264 / +0,422 / +0,691 / +0,877 / +0,762 bei k = 25 / 60 / 176 / 272 / 420 / 648 / 1000), die Gitter, die Faustregeln, die 1 224 Komponenten und die Zielgrößen 287,158 (je Container) und 287,123 (Tagesmaximum).

## Modell

**Konventionen.** ξ > 0 ist der Fréchet-Typ, ξ < 0 der Weibull-Typ (`scipy.stats.genpareto`: c = ξ; `scipy.stats.genextreme` wäre c = −ξ, hier nur als Test-Referenz benutzt). „Pegel“ ist der Wert x_p zur Überschreitungswahrscheinlichkeit p **je Beobachtung** (nicht die Wiederkehrperiode). σ\* = σ_u − ξ·u ist die modifizierte Skala, die beim Anheben der Schwelle gleich bleibt, wenn die GPD gilt. Der **Extremwertindex** ξ wird nie mit dem Extremalindex θ verwechselt (θ kommt hier nicht vor). **Zähleinheiten:** n Beobachtungen, k Überschreitungen (die k größten Werte, Schwelle = (k+1)-größter Wert), Stufen des Gitters, R Läufe. **Fehlermaß:** ln(Pegel-Dach / wahrer Pegel); „Faktor über 2“ heißt Verhältnis über 2, „unter 0,5“ unter 0,5 (Anteil der Läufe).

**Fälle** (alle mit exaktem wahren Pegel zur Überschreitungswahrscheinlichkeit p je Beobachtung; wahre Pegel: 15,6322 (Spleiß 10 %), 10,6911 (Spleiß 3 %), 48,9398 (Lognormal), 9,9035 (Exponential), 287,158 Tage (Vehikel B)):

* **Spleiß-Fall:** Körper Gleich(0, 1) mit Masse 1 − ζ₀, ab u₀ = 1 exakt GPD(ξ = 3/10, σ = 1/2) mit Masse ζ₀ über u₀; ζ₀ = 1/10 (u₀ liegt bei k = 200 von n = 2000) und ζ₀ = 3/100 (bei k = 60); p = 5·10⁻⁵ je Beobachtung. **Oberhalb u₀ ist die Überschreitungsverteilung exakt eine GPD, darunter nicht:** die Dichte der Beobachtung springt bei u₀ von 9/10 auf 1/5, der Hazard der Überschreitungen von 9 auf 2. Das ist eine **Konstruktion**, kein Vehikel.
* **Lognormal (σ = 1)** und **Exponential**: glatte Abweichung bzw. GPD bei jeder Schwelle exakt; n = 2000, p = 5·10⁻⁵.
* **Vehikel B**, die ausgedachte Verweilzeit-Mischung der Linie (Containerart, Kunde, Zoll, Wochenende, Vorlauf): n = 5000 Container je Lauf (50 Tage zu 100 Container), p = 10⁻⁵ **je Container**, wahrer Pegel 287,158 Tage (Zielgröße (a), exakt aus der Gewichtstabelle). Das Tagesmaximum aus 100 Containern, das einmal in 1000 Tagen überschritten wird, liegt bei 287,123 Tagen (Zielgröße (b)), das 0,99-Quantil bei 155,128: **zwei verschiedene Größen**, die nicht gleichgesetzt werden. Mittel der Verweilzeit 5,646, Median 4,166 Tage, P(Verweilzeit über 30 Tage) = 0,009830.

**Gitter.** 18 Stufen, k logarithmisch gleichmäßig von 25 bis 0,4·n (höchstens 1000), **absteigend** geordnet (Index 0 = niedrigste Schwelle): bei n = 2000 sind das 25, 31, 38, 46, 57, 69, 85, 104, 128, 157, 192, 235, 289, 354, 434, 532, 652, 800, bei n = 5000 25 bis 1000.

**Die drei Regeln** (alle **eigene Festlegungen**; die Ergebnisse gelten für diese Regeln und diese Gitter, nicht für die Methodenklasse):

* **S (Stabilität):** die niedrigste Schwelle, ab der alle ξ̂ der höheren Schwellen im Band ± c·(1+ξ̂_h)/√k_h um das ξ̂ dieser Schwelle liegen (Standardfehler der jeweils **höheren** Schwelle h, 1+ξ̂ nach unten durch 0,3 begrenzt); geprüft werden nur Schwellen mit mindestens drei höheren Nachbarn; ohne Treffer wählt S die höchste Schwelle.
* **M (Mean Residual Life):** die niedrigste Schwelle, ab der das Stichprobenmittel der Überschreitungen e(u) in allen höheren Schwellen innerhalb ± c Standardfehler um die gewichtete Gerade liegt (mindestens fünf Stufen); sonst Rückfall auf den Index 13 (Stufe J − 5), bei weniger als fünf Stufen auf Index 0.
* **G (Anpassungstest):** die niedrigste Schwelle, deren Kolmogorov-Smirnov-Abstand der Überschreitungen zur ML-angepassten GPD (√k·D) unter dem **unter H₀ simulierten 90-%-Wert** liegt, je Schwelle einzeln, ohne Mehrfachtest-Korrektur. Der kritische Wert kommt aus exakten GPD-Stichproben mit ξ ∈ {0; 0,3} (Kalibrierung G, **enthält die Wahrheit des Spleiß-Falls**) oder ξ ∈ {0; 0,5} (G2, ohne den wahren Wert). Die Regel hat nur die Idee des Verfahrens von Bader u. a. (Abstract gelesen) mit der Wahl der **niedrigsten** bestandenen Schwelle, keine ForwardStop-Kontrolle: sie ist **nicht dasselbe Verfahren**.
* **Feste k** zum Vergleich: obere 2 %, 5 %, 10 % und die Faustregeln k = √n und k = n^(2/3)/ln ln n (nach Scarrott/MacDonald laut Quellenbericht, ungeprüft): n = 2000: 45 und 78, n = 5000: 71 und 137.
* **c** (Bandbreite in Standardfehlern) ist ein Regler: 1,5 / 2 / 3.
* **„Brauchbar“** (nur ohne Spleiß-Fall): der **Populationsfehler** des Pegels (ohne Stichprobenrauschen) ist für die gewählte und alle höheren Stufen höchstens 0,05 bzw. 0,10; eigene Festlegung. Im Spleiß-Fall heißt „gut gewählt“: die gewählte Schwelle liegt bei oder über u₀.

**Presets** (Schnellstart in der Seitenleiste, das aktive ist hervorgehoben): „Spleiß-Fall 10 %: Kippen bei k = 200“ (Körper gleichverteilt, ab u₀ = 1 exakt GPD, Anteil 10 % über u₀ (bei k = 200 von n = 2000): unter u₀ kippt der Pegel, darüber nicht); „Spleiß-Fall 3 %: schärferer Ausschlag“ (Dieselbe Konstruktion mit Anteil 3 % über u₀ (bei k = 60): der Ausschlag unter u₀ ist viel schärfer); „Lognormal: weich“ (Glatte Abweichung von der GPD: es gibt keine Kippschwelle, der Fehler driftet mit k); „Exponential: nichts kippt“ (Bei jeder Schwelle exakt GPD (ξ = 0): ξ und σ* bleiben, es gibt nichts zu wählen); „Verweilzeit-Mischung (Vehikel B)“ (Ausgedachte Verweilzeit-Mischung (1224 Komponenten), n = 5000, p = 10⁻⁵ je Container: kein Kippen, der Populationsfehler wächst stetig bis k ≈ 650 und fällt dann leicht); „Dieselbe Reihe, sieben Regeln“ (Vehikel B, anderer Zufalls-Seed: was sieben Regeln auf derselben Reihe als Pegel liefern (Experiment 4)).

Die Formeln stehen im Expander „📐 Mathematische Formulierung“ der App mit Lesehilfe; er ist eine **Skizze** (die numerische Maximierung und die Quadraturknoten sind dort nicht ausgeschrieben).

## Methodik

* **Zufall.** Aus der reinen Vorlage `sws_rng.py` (SplitMix64, Kopier-Test): gleiche Zufalls-Seeds geben auf jeder Plattform dieselbe Reihe. Der Spleiß-Fall zieht den Körper per `random()` und den Tail per Inversion; Vehikel B wählt die Komponente über die Summenverteilung der Gewichte und zieht exp(μ + s·`normal()`). Zufallsanteile sind Messungen mit Standardfehler; Neu-Rechnen-Tests prüfen als Band, nie auf das letzte Bit.
* **GPD-ML** in numpy (Entscheidung der Linie: scipy nur in den Tests): Profil-Likelihood in einer Variablen (Grimshaw-Reduktion), Gitter plus goldener Schnitt, ξ ≥ −1; stimmt mit `scipy.stats.genpareto.fit` auf 40 Stichproben je ξ überein (Orakel-Test).
* **Population ohne Rauschen.** Die GPD, die die **wahre** Überschreitungsverteilung bei einer Schwelle am besten beschreibt (Maximum Likelihood auf der Population, Gauß-Legendre-Quadratur mit 80 + 300 Knoten im Spleiß-Fall, 400 Knoten über die Quantile bei Lognormal und Vehikel B), und der Fehler des daraus gerechneten Pegels. Kein Zufall, kein Rauschen; die Kurve des Spleiß-Falls rechnet die App live (rund 0,05 s), die von Lognormal und Vehikel B liegt in `sws_population.json`.
* **Eingefroren** (Quelle und Stand je Feld in den Dateien, Erzeuger unter `tools/` mit `--check`): `sws_gof_crit.json` (kritische Werte der Regel G, 1 000 Stichproben je k und ξ), `sws_population.json` (Populationskurven), `sws_rules.json` (die Läufe: R = 400 Läufe (Spleiß, Lognormal, Exponential) bzw. R = 300 Läufe (Vehikel B); Rechenzeit der Läufe in der Reihenfolge Spleiß 10 %, Spleiß 3 %, Lognormal, Exponential, Vehikel B: 3 s + 3 s + 11 s + 7 s + 21 s, Messung dieser README, nicht im Test). Live rechnet die App nur **eine** Reihe mit 18 + 5 ML-Anpassungen (Messung dieser README, nicht im Test, dieser Rechner, Streamlit Cloud ist langsamer): Spleiß-Fall rund 0,04 s, Lognormal bei n = 5000 rund 0,4 s, Vehikel B bei n = 5000 rund 0,5 s (davon 0,45 s für das Ziehen der 5000 Werte).
* **Vehikel B** ist die **exakte Aufzählung** (4 704 Schritte, 2 376 mit Gewicht, 1 224 verschiedene Komponenten); die Tabelle `sws_mixture.json` ist eine unveränderte Kopie der gemeinsamen Tabelle der Linie (SHA-256 `669f821c78e0bf38c47fa846232f27cd2ff126f4fceadca5369a5c9af5551c9f`, Quelle und Stand je Feld in `sws_mixture_quelle.json`). Ein Test zählt sie aus der Definition selbst nach; ein weiterer prüft die Konstanten und die Medianformel gegen den Quelltext der `verweilzeit-demo` (nur lesend, nur im Test). `sws_rng.py` hat den SHA-256 `d69b2b2fd67c1234cc0d79249676a662a2189d02803eea7d2edfb04b7a505a3f`.
* **Rückfälle und Meldungen statt Wahl:** weniger als 100 Beobachtungen, mehr als 10 % Gleichstände unter den größten Werten (gerundete Daten) oder nicht endliche Werte geben eine Meldung statt einer Wahl; die Wirkung gerundeter Daten ist **nicht gemessen**.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen sind **eigene Nachrechnung in diesem Modell**; Zähleinheit steht je Befund. Voraussetzungen: unabhängige Beobachtungen, die genannten Fälle, genau diese Regeln und Gitter.

### A. Der Spleiß-Fall ohne Rauschen

* **Mean Residual Life exakt:** e(u) = (σ + ξ(u − u₀))/(1 − ξ) für u ≥ u₀, Steigung ξ/(1 − ξ) = 3/7; darunter (a = 1 − u) e(u) = (9a²/20 + a/10 + 1/14)/(9a/10 + 1/10). Von Hand in Brüchen: e(0) = 87/140 = 0,6214, e(1/2) = 131/308 = 0,4253, e(1) = 5/7 = 0,7143; die Knickstelle ist exakt: Steigung 3/7 = 0,429 rechts von u₀ und 38/7 = 5,43 links davon (Hazard 9 gegen 2).
* **Populationsfehler des Pegels** (kein Zufall; n = 2000, p = 5·10⁻⁵, Anteil 10 %): oberhalb u₀ (k ≤ 192 von 200) **null**, ξ_Pop = 0,30, σ\* = 0,2. Darunter ln(Pegel/wahr) = +0,76 / +2,12 / +2,24 / +1,59 / +0,84 / +0,19 / −0,30 bei k = 235 / 289 / 354 / 434 / 532 / 652 / 800, also **Faktor 8,3 bei k = 289 und 9,4 bei k = 354**; ξ_Pop springt auf 0,50 / 0,78 / 0,80 (k = 235 / 289 / 354) und σ\* von 0,2 auf −0,15 / −0,53 / −0,56. Der Fehler ist **nicht monoton** in k: er springt hinter u₀ auf ein Maximum und fällt dann, bei k ≈ 700 wechselt das Vorzeichen.
* **Anteil 3 %** (u₀ bei k = 60): +0,67 (k = 69), +5,25 (k = 85), +6,34 (k = 104), +5,04 (k = 128), +3,65 (k = 157), +2,45 (k = 192), +1,45 (k = 235), +0,65 (k = 289), +0,06 (k = 354), −0,38 (k = 434), −0,99 (k = 800); das Maximum +2,24 wird bei Anteil 10 % erreicht, bei Anteil 3 % ist es +6,34 = **Faktor 568** bei k = 104 (ξ_Pop = 1,78), oberhalb (k ≤ 57) null: der **Spleiß-Fall ist ein Extremfall**.
* **Die Läufe** (R = 400, n = 2000, Zufalls-Seeds aus der Seed-Basis 7000): Verzerrung und RMSE der festen Stufen folgen der Population; das scheinbare Minimum des RMSE unter u₀ (k = 652: 0,34) ist ein Vorzeichenwechsel der Verzerrung (+0,17 bei k = 652, −0,32 bei k = 800), **kein brauchbarer Wert**.

| k | Verzerrung (Mittel von ln) | Streuung (sd) | RMSE | Anteil unter wahr |
|---|---|---|---|---|
| 192 | +0,01 | 0,40 | 0,403 | 50 % |
| 235 | +0,82 | 0,65 | 1,048 | 9 % |
| 289 | +2,07 | 0,68 | 2,175 | 0 % |
| 354 | +2,19 | 0,53 | 2,250 | 0 % |
| 434 | +1,55 | 0,45 | 1,614 | 0 % |
| 532 | +0,81 | 0,37 | 0,896 | 2 % |
| 652 | +0,17 | 0,29 | 0,339 | 30 % |
| 800 | −0,32 | 0,24 | 0,394 | 92 % |

* **Glatte Abweichungen kippen nicht.** Lognormal (σ = 1): der Populationsfehler ist höchstens 0,05 erst für k ≤ 85 (für diese und alle höheren Stufen), +0,015 bei k = 25 und +0,20 bei k = 800; der RMSE der Läufe **fällt** trotzdem im Ganzen bis k = 800 (zwischen k = 532 und 800 ist er flach: 0,299 / 0,299 / 0,296), weil die Varianz die Verzerrung übertrifft. Exponential: Populationsfehler null bei jedem k, der RMSE fällt von 0,26 (k = 25) auf 0,11 (k = 800); die Läufe haben bei k = 25 eine kleine Verzerrung von ln −0,08 (in 70 % der Läufe liegt der Pegel unter dem wahren).
* **Exakte Familien** (Experiment 1): bei Exponential (ξ = 0), Pareto-Verteilung α = 2 (ξ = 1/2) und Gleichverteilung (ξ = −1) ist die Überschreitungsverteilung bei jeder Schwelle eine exakte GPD, σ\* = 1 / 0 / 1 bleibt, σ_u = 1 / u/2 / 1 − u wächst nur bei ξ > 0, bleibt bei ξ = 0 und fällt bei ξ < 0.

### B. Vehikel B ohne Rauschen und mit Läufen

* **Populationskurve** (GPD-ML an die **exakte** Überschreitungsverteilung der Mischung, Quantil-Quadratur über die 1 224 Komponenten, kein Zufall): ξ_Pop 0,265 / 0,294 / 0,333 / 0,371 / 0,427 / 0,461 / 0,442 und Populationsfehler **+0,065 / +0,133 / +0,264 / +0,422 / +0,691 / +0,877 / +0,762** bei k = 25 / 60 / 176 / 272 / 420 / 648 / 1000; σ\* ändert sich von 6,28 bei k = 25, −0,21 bei k = 648. Weder ξ noch σ\* bleiben beim Anheben der Schwelle gleich; **es gibt in diesem Gitter kein Gitterpunkt mit Populationsfehler höchstens 0,05**, höchstens 0,10 hat nur k ≤ 39 (+0,096 bei k = 39 und +0,113 bei k = 48). Der Fehler wächst stetig von +0,065 bei k = 25 bis +0,88 bei k = 648 und fällt danach bis k = 1000 auf +0,76: **Vehikel B kippt nicht.**
* **Feste k** (R = 300, n = 5000, p = 10⁻⁵ je Container; Verzerrung und Streuung von ln(Pegel-Dach/wahr)): das U aus Verzerrung gegen Varianz, kleinster RMSE 0,54 bei k = 114 (mit Wissen der Wahrheit):

| k | Verzerrung (Mittel von ln) | Streuung (sd) | RMSE | Anteil unter wahr |
|---|---|---|---|---|
| 25 | −0,09 | 0,73 | 0,730 | 62 % |
| 60 | +0,05 | 0,58 | 0,584 | 49 % |
| 92 | +0,13 | 0,55 | 0,565 | 43 % |
| 114 | +0,15 | 0,52 | 0,538 | 39 % |
| 142 | +0,20 | 0,52 | 0,554 | 36 % |
| 176 | +0,25 | 0,50 | 0,556 | 34 % |
| 272 | +0,40 | 0,47 | 0,612 | 21 % |
| 420 | +0,66 | 0,40 | 0,768 | 5 % |
| 522 | +0,79 | 0,37 | 0,869 | 2 % |
| 648 | +0,86 | 0,34 | 0,921 | 0 % |
| 1000 | +0,75 | 0,30 | 0,811 | 0 % |

### C. Regeln: Spleiß-Fall, Vehikel B, glatte Fälle

Zähleinheit Läufe; „niedrigste Stufe“ ist der Anteil der Läufe, in denen die Regel die niedrigste Schwelle des Gitters wählt; das Verhältnis ist Pegel-Dach/wahrer Pegel.

**Spleiß-Fall, Anteil 10 %** (n = 2000, R = 400). Anteil der Läufe mit gewählter Schwelle bei oder über u₀:

| Regel | bei oder über u₀ | RMSE | Median des Verhältnisses | gewähltes k: Median (Q10/Q90) | niedrigste Stufe |
|---|---|---|---|---|---|
| S (c = 1,5) | 71,0 % | 0,547 | 0,84 | 192 (25/235) | 0 % |
| S (c = 2) | 50,0 % | 0,581 | 0,99 | 192 (46/235) | 0 % |
| S (c = 3) | 24,5 % | 0,778 | 1,41 | 235 (128/532) | 0 % |
| M (c = 1,5) | 3,8 % | 1,421 | 2,56 | 289 (235/652) | 7 % |
| M (c = 2) | 0,2 % | 1,245 | 1,19 | 652 (235/800) | 39 % |
| M (c = 3) | 0,0 % | 0,596 | 0,75 | 800 (800/800) | 93 % |
| G | 70,2 % | 0,425 | 0,98 | 192 (157/192) | 0 % |
| G2 (Kalibrierung ξ ∈ {0; 0,5}) | 71,2 % | 0,421 | 0,98 | 192 (157/192) | 0 % |
| fest k = 40 (obere 2 %) | – | 0,606 | 0,82 | – | – |
| fest k = 100 (obere 5 %) | – | 0,476 | 0,93 | – | – |
| fest k = 200 (obere 10 %) | – | 0,433 | 1,06 | – | – |
| fest k = 45 (k = √n) | – | 0,576 | 0,83 | – | – |
| fest k = 78 (k = n^(2/3)/ln ln n) | – | 0,514 | 0,94 | – | – |

**Spleiß-Fall, Anteil 3 %** (n = 2000, R = 400, u₀ bei k = 60):

| Regel | bei oder über u₀ | RMSE | Median des Verhältnisses | gewähltes k: Median (Q10/Q90) | niedrigste Stufe |
|---|---|---|---|---|---|
| S (c = 1,5) | 45,2 % | 0,650 | 0,89 | 57 (25/69) | 0 % |
| S (c = 2) | 30,0 % | 0,738 | 1,02 | 69 (25/69) | 0 % |
| S (c = 3) | 16,2 % | 1,006 | 1,30 | 69 (46/157) | 0 % |
| M (c = 1,5) | 0,0 % | 5,157 | 151,10 | 104 (85/289) | 1 % |
| M (c = 2) | 0,0 % | 4,187 | 34,46 | 157 (104/532) | 2 % |
| M (c = 3) | 0,0 % | 1,524 | 0,47 | 532 (192/800) | 28 % |
| G | 51,5 % | 0,578 | 0,98 | 57 (46/69) | 0 % |
| G2 (Kalibrierung ξ ∈ {0; 0,5}) | 51,5 % | 0,577 | 0,98 | 57 (46/69) | 0 % |
| fest k = 40 (obere 2 %) | – | 0,611 | 0,84 | – | – |
| fest k = 100 (obere 5 %) | – | 6,038 | 395,72 | – | – |
| fest k = 200 (obere 10 %) | – | 2,388 | 9,26 | – | – |
| fest k = 45 (k = √n) | – | 0,588 | 0,86 | – | – |
| fest k = 78 (k = n^(2/3)/ln ln n) | – | 3,821 | 7,83 | – | – |

* **Regel S mit engerem Band wählt öfter im GPD-Bereich** (Anteil 10 %: 71,0 / 50,0 / 24,5 % für c = 1,5 / 2 / 3). **Regel M scheitert** bei c = 1,5 und 2 (3,8 und 0,2 %); bei c = 3 sieht M mit RMSE 0,60 gut aus, **aber aus dem falschen Grund**: sie wählt in 93 % der Läufe die niedrigste Schwelle, wo die Verzerrung das Vorzeichen wechselt. Die **Ursache**, warum M scheitert, ist **nicht untersucht**.
* **Regel G ist im Spleiß-Fall die beste der drei Regeln, aber gegenüber dem besten festen k (k = 200, das u₀ per Konstruktion trifft) liegt ihr Vorsprung im Rauschen:** RMSE 0,42 gegen 0,43 bis 0,61 der festen k bei Anteil 10 %, darunter 0,43 bei k = 200; gegen die festen k = 40, 45 und 78 ist G mit 0,42 gegen 0,51 bis 0,61 besser (um mehr als zwei Standardfehler), gegen k = 100 (0,48) ist der Abstand kleiner; bei Anteil 3 % ist G (0,58) nicht besser als k = 45 (0,59). **G ist auf den Testfall kalibriert** (ξ = 0,3 steckt in der Kalibrierung); ohne den wahren Wert (G2) ändert sich das Ergebnis kaum: 71,2 % und RMSE 0,42 bei Anteil 10 %, 51,5 % und 0,58 bei Anteil 3 %. Die kritischen Werte von √k·D liegen zwischen 0,814 bis 0,862 (G) und 0,805 bis 0,841 (G2), bei bekannten Parametern wäre der 90-%-Wert 1,224.
* **Ein fest gewähltes k ist im Spleiß-Fall riskant, wenn u₀ unbekannt ist:** dasselbe k = 100 hat bei Anteil 10 % den RMSE 0,476 und bei Anteil 3 % 6,038 (Median des Verhältnisses 396). **Das Riskante ist die Konstruktion** (u₀ liegt dort bei k = 60); „obere 10 %“ (k = 200) trifft u₀ bei Anteil 10 % per Konstruktion (ζ₀·n = 200), nicht als Eigenschaft der Regel.

**Vehikel B** (n = 5000, R = 300; p = 10⁻⁵ je Container):

| Regel | RMSE | Median des Verhältnisses | über 2 / unter 0,5 | gewähltes k: Median (Q10/Q90) | niedrigste Stufe |
|---|---|---|---|---|---|
| S (c = 1,5) | 0,652 | 1,12 | 20,3 % / 16,3 % | 272 (25/1000) | 25 % |
| S (c = 2) | 0,689 | 1,39 | 27,3 % / 14,0 % | 420 (25/1000) | 42 % |
| S (c = 3) | 0,766 | 1,89 | 43,7 % / 9,7 % | 1000 (74/1000) | 68 % |
| M (c = 1,5) | 0,794 | 2,10 | 56,3 % / 0,0 % | 1000 (420/1000) | 71 % |
| M (c = 2) | 0,805 | 2,11 | 59,3 % / 0,0 % | 1000 (789/1000) | 88 % |
| M (c = 3) | 0,812 | 2,11 | 59,7 % / 0,0 % | 1000 (1000/1000) | 98 % |
| G | 0,843 | 2,15 | 61,7 % / 0,0 % | 1000 (805/1000) | 82 % |
| G2 (Kalibrierung ξ ∈ {0; 0,5}) | 0,846 | 2,15 | 61,3 % / 0,0 % | 1000 (805/1000) | 83 % |
| fest k = 100 (obere 2 %) | 0,544 | 1,14 | 15,3 % / 6,3 % | – | – |
| fest k = 250 (obere 5 %) | 0,595 | 1,45 | 20,7 % / 1,0 % | – | – |
| fest k = 500 (obere 10 %) | 0,845 | 2,16 | 56,0 % / 0,0 % | – | – |
| fest k = 71 (k = √n) | 0,575 | 1,05 | 17,0 % / 8,7 % | – | – |
| fest k = 137 (k = n^(2/3)/ln ln n) | 0,549 | 1,17 | 17,7 % / 3,7 % | – | – |

* **Die Regel wählt die niedrigste Schwelle** (Hauptfalle, für **diese** Regeln, Zähleinheit Läufe): M (c = 2) und G in 88 % bzw. 82 % der Läufe die niedrigste Stufe (k = 1000); dort liegt der Populationsfehler bei +0,76 (das Maximum +0,88 liegt bei k = 648). Das ist das Muster der Literatur („viele automatische Regeln wählen die niedrigste Kandidatenschwelle“, Belzile/Davison, nach Zusammenfassung ungeprüft), hier an **eigenen** Regeln gemessen, nicht an den Literaturverfahren. **Der Anpassungstest G bemerkt den Pegelfehler kaum:** er besteht bei der niedrigsten Stufe in 76 bis 82 % der Läufe (drei Seed-Basen, siehe die Wiederholungen oben; bei exakter GPD wäre der Sollwert 90 %).
* **Keine der Regeln schlägt ein festes k = 71 bis 250** (RMSE der Regel-Varianten 0,65 bis 0,85, der festen k 0,54 bis 0,59; gleiche Zähleinheit Läufe). Gegen das fest gewählte k = 500 (RMSE 0,85, ebenfalls a priori gesetzt, aber schlecht) sind nur die Regeln S besser, M und G nicht. Die Vergleichswerte k = 100 und 137 sind unter den fünf festen k die besten (im Nachhinein gelesen), nicht die besten Gitterpunkte (k = 114: RMSE 0,538); beide sind vorab festgelegt (obere 2 %, Faustregel), nur ihre Auswahl als „beste“ geschah im Nachhinein: „fest“ ist nur so gut wie der gewählte Wert.
* **Brauchbar** (Populationsfehler höchstens 0,10 ab der gewählten Stufe): in 12 % / 0 % / 0 % der Läufe bei S (c = 2) / M (c = 2) / G; höchstens 0,05 erreicht keine Regel, weil kein Gitterpunkt es erfüllt.
* **Lognormal und Exponential** (n = 2000, R = 400; im Plan nicht vorgesehen, hier mitgemessen): bei glatten Verteilungen **fällt der RMSE mit k**, deshalb gewinnen die Regeln, die die niedrigste Stufe wählen (M, G), gegen die festen k = 40 bis 100; das ist **kein Verdienst der Regel**, sondern die Abwägung ohne Kippschwelle.

**Lognormal (σ = 1):**

| Regel | RMSE | Median des Verhältnisses | über 2 / unter 0,5 | gewähltes k: Median (Q10/Q90) | niedrigste Stufe |
|---|---|---|---|---|---|
| S (c = 1,5) | 0,445 | 0,94 | 0,8 % / 11,2 % | 354 (25/800) | 38 % |
| S (c = 2) | 0,401 | 1,02 | 0,8 % / 9,0 % | 800 (46/800) | 57 % |
| S (c = 3) | 0,364 | 1,13 | 1,2 % / 6,0 % | 800 (104/800) | 80 % |
| M (c = 1,5) | 0,304 | 1,21 | 1,2 % / 0,2 % | 800 (652/800) | 89 % |
| M (c = 2) | 0,297 | 1,21 | 1,2 % / 0,0 % | 800 (800/800) | 96 % |
| M (c = 3) | 0,296 | 1,21 | 1,2 % / 0,0 % | 800 (800/800) | 100 % |
| G | 0,300 | 1,21 | 1,2 % / 0,0 % | 800 (652/800) | 89 % |
| G2 (Kalibrierung ξ ∈ {0; 0,5}) | 0,301 | 1,21 | 1,2 % / 0,2 % | 800 (652/800) | 88 % |
| fest k = 40 (obere 2 %) | 0,499 | 0,88 | 7,2 % / 10,2 % | – | – |
| fest k = 100 (obere 5 %) | 0,409 | 0,96 | 5,2 % / 4,8 % | – | – |
| fest k = 200 (obere 10 %) | 0,362 | 1,01 | 3,8 % / 2,2 % | – | – |
| fest k = 45 (k = √n) | 0,484 | 0,88 | 5,2 % / 9,5 % | – | – |
| fest k = 78 (k = n^(2/3)/ln ln n) | 0,437 | 0,94 | 6,0 % / 5,0 % | – | – |

**Exponential:**

| Regel | RMSE | Median des Verhältnisses | über 2 / unter 0,5 | gewähltes k: Median (Q10/Q90) | niedrigste Stufe |
|---|---|---|---|---|---|
| S (c = 1,5) | 0,205 | 0,90 | 0,2 % / 0,0 % | 354 (25/800) | 38 % |
| S (c = 2) | 0,188 | 0,92 | 0,0 % / 0,0 % | 800 (25/800) | 58 % |
| S (c = 3) | 0,169 | 0,95 | 0,0 % / 0,0 % | 800 (46/800) | 75 % |
| M (c = 1,5) | 0,114 | 0,99 | 0,0 % / 0,0 % | 800 (800/800) | 91 % |
| M (c = 2) | 0,112 | 0,99 | 0,0 % / 0,0 % | 800 (800/800) | 98 % |
| M (c = 3) | 0,112 | 0,99 | 0,0 % / 0,0 % | 800 (800/800) | 100 % |
| G | 0,116 | 0,99 | 0,0 % / 0,0 % | 800 (652/800) | 88 % |
| G2 (Kalibrierung ξ ∈ {0; 0,5}) | 0,116 | 0,99 | 0,0 % / 0,0 % | 800 (652/800) | 87 % |
| fest k = 40 (obere 2 %) | 0,238 | 0,93 | 0,8 % / 0,0 % | – | – |
| fest k = 100 (obere 5 %) | 0,188 | 0,96 | 0,2 % / 0,0 % | – | – |
| fest k = 200 (obere 10 %) | 0,160 | 0,96 | 0,0 % / 0,0 % | – | – |
| fest k = 45 (k = √n) | 0,228 | 0,94 | 0,2 % / 0,0 % | – | – |
| fest k = 78 (k = n^(2/3)/ln ln n) | 0,206 | 0,95 | 0,2 % / 0,0 % | – | – |

### D. Dieselbe Reihe, sieben Regeln

Auf jeder Reihe werden S (c = 2), M (c = 2), G und die festen k obere 2 %, 5 %, 10 % und √n angewandt; das Verhältnis des höchsten zum niedrigsten Pegel der sieben je Läufe (n = Beobachtungen wie in den Tabellen oben):

| Fall | Median des Verhältnisses höchster / niedrigster | Q10 / Q90 | Maximum | über 2 | über 4 | wahrer Pegel im Bereich |
|---|---|---|---|---|---|---|
| Spleiß-Fall, Anteil 10 % (n = 2000, R = 400) | 3,39 | 1,68 / 14,77 | 37,0 | 77,2 % | 44,8 % | 86,0 % |
| Spleiß-Fall, Anteil 3 % (n = 2000, R = 400) | 716,23 | 101,17 / 4596,69 | 32646,9 | 100,0 % | 100,0 % | 85,2 % |
| Lognormal (σ = 1) (n = 2000, R = 400) | 1,73 | 1,31 / 2,58 | 7,1 | 35,0 % | 0,5 % | 61,3 % |
| Exponential (n = 2000, R = 400) | 1,25 | 1,11 / 1,51 | 4,1 | 0,8 % | 0,2 % | 52,5 % |
| Vehikel B (n = 5000, R = 300) | 2,76 | 1,58 / 5,81 | 10,9 | 72,3 % | 23,3 % | 59,7 % |

Im Vehikel B liegt der höchste der sieben Pegel im Median beim Median 2,76, Q10/Q90 1,58/5,81, Maximum 10,9; über Faktor 2 in 72,3 %, über Faktor 4 in 23,3 % der 300 Läufe; der wahre Pegel liegt zwischen niedrigstem und höchstem der sieben in 59,7 %. Die Literatur nennt auf denselben Regendaten Hundertjahreswerte von 48 bis 348 mm (Faktor 7,2; nach Zusammenfassung, ungeprüft): dieselbe Größenordnung des Streubereichs, aber ein anderes Modell und andere Regeln, **kein Nachweis**.

### Zahl → Test (Band oder exakt)

Die Zahlen der Läufe stehen als Bausteine in `tests/claims_fragmente.py` und werden von `tests/test_claims.py` wörtlich in dieser README gefunden; ihre Neu-Rechnung als Band prüft `tests/test_tabellen_und_baender.py`. Alles andere ist exakt oder ein Band mit Standardfehler:

| Zahl | Test |
|---|---|
| e(0) = 87/140, e(1/2) = 131/308, e(1) = 5/7 (Brüche) | `tests/test_handbeispiele.py::test_h1_e_bei_0_halb_und_1_als_brueche` |
| Steigung 3/7 oberhalb, 38/7 unterhalb u₀ (Hazard 9 gegen 2) | `tests/test_handbeispiele.py::test_h1_steigung_oberhalb_u0_ist_3_7_und_links_38_7` |
| e(u) in geschlossener Form gegen Quadratur der Überlebensfunktion | `tests/test_oracle_population.py::test_e_des_spleiss_falls_gegen_quadratur` |
| wahrer Pegel 15,6322 und S(x_p) = p | `tests/test_handbeispiele.py::test_h2_wahrer_pegel_15_632_und_gegenprobe` |
| wahre Pegel aller Fälle gegen scipy | `tests/test_oracle_population.py::test_wahrer_pegel_lognormal_exponential_vehikel_gegen_scipy` |
| Populationsfehler +0,76 / +2,12 / −0,30 (k = 235 / 289 / 800) | `tests/test_handbeispiele.py::test_h3_populationsfehler_gegen_die_tafel_des_plans` |
| Populations-ML gegen verschachteltes `quad` | `tests/test_oracle_population.py::test_populations_ml_spleiss_gegen_verschachteltes_quad` |
| Fehler null und σ\* = 0,2 oberhalb u₀ | `tests/test_oracle_population.py::test_populations_ml_oberhalb_u0_exakt_gpd` |
| Populations-ML Lognormal gegen Quadratur über s | `tests/test_oracle_population.py::test_populations_ml_lognormal_gegen_quad_ueber_s` |
| Vehikel B: Populationsfehler +0,065 und +0,812 mit Brent statt Tabelle | `tests/test_oracle_population.py::test_populationsfehler_vehikel_b_mit_brent_statt_tabelle` |
| exakte Familien (ξ, σ_u, σ\*) | `tests/test_oracle_population.py::test_exakte_familien_sind_bei_jeder_schwelle_eine_gpd` |
| σ_u wächst nur bei ξ > 0, bleibt bei 0, fällt bei ξ < 0 | `tests/test_oracle_population.py::test_sigma_u_waechst_nur_bei_xi_positiv_bleibt_bei_null_faellt_bei_negativ` |
| 4 704 Schritte, 2 376 mit Gewicht, 1 224 Komponenten | `tests/test_oracle_vehicle.py::test_eigene_aufzaehlung_hat_4704_schritte_2376_mit_gewicht_und_1224_komponenten` |
| Gewichte gleich eigene Aufzählung auf 1e-12 | `tests/test_oracle_vehicle.py::test_tabelle_gleich_eigene_aufzaehlung_auf_1e_12` |
| Pegel 287,158 / 287,123 / 155,128 gegen scipy-Brent | `tests/test_oracle_vehicle.py::test_wahrer_pegel_gegen_scipy_brent_und_logsumexp` |
| Mittel 5,646, Median 4,166, P(über 30 Tage) | `tests/test_handbeispiele.py::test_h6_mittel_median_und_ueberschreitung_von_30_tagen` |
| Zielgrößen (a) und (b) sind verschiedene Zahlen | `tests/test_handbeispiele.py::test_h6_pegel_je_container_und_tagesmaximum_sind_verschiedene_zahlen` |
| Regel S wählt k = 200, Rückfall auf die höchste Schwelle | `tests/test_handbeispiele.py::test_h4_regel_s_waehlt_k_200` |
| Standardfehler der höheren Schwelle | `tests/test_handbeispiele.py::test_h4_standardfehler_der_hoeheren_schwelle_nicht_der_geprueften` |
| Regel M wählt Stufe 2 und Rückfall J − 5 | `tests/test_handbeispiele.py::test_h4_regel_m_waehlt_stufe_2_gerade_und_rueckfall` |
| Regel M findet den Knick u₀ im exakten e(u) | `tests/test_handbeispiele.py::test_h4_regel_m_waehlt_mit_dem_exakten_e_des_spleiss_falls_genau_den_knick` |
| Regeln S, M, G gegen eine zweite Umsetzung auf je 300 Instanzen | `tests/test_oracle_rules.py::test_regel_s_gleich_referenz_auf_300_instanzen` |
| KS-Abstand von Hand 1 − 1/e und gegen `kstest` | `tests/test_oracle_gpd_fit.py::test_ks_abstand_gleich_scipy_kstest` |
| Faustregeln 45 / 78 und 71 / 137, obere 2 %, 5 %, 10 % | `tests/test_handbeispiele.py::test_h5_faustregeln` |
| Gitter mit 18 Stufen (800 bis 25 und 1000 bis 25) | `tests/test_handbeispiele.py::test_h5_gitter_ist_logarithmisch_absteigend_mit_18_stufen` |
| ML gegen `genpareto.fit` und Gradient null | `tests/test_oracle_gpd_fit.py::test_ml_gleich_scipy_fit_auf_40_stichproben` |
| Pegel gleich Quantil der GPD | `tests/test_oracle_gpd_fit.py::test_pegel_gleich_quantil_der_gpd` |
| Skalierung, Verschiebung, Reihenfolge ohne Wirkung auf die Stufe | `tests/test_eigenschaften_und_randfaelle.py::test_skalierung_mit_c_aendert_die_gewaehlte_stufe_nicht_und_skaliert_den_pegel` |
| größeres c wählt nie eine höhere Schwelle | `tests/test_eigenschaften_und_randfaelle.py::test_groesseres_c_waehlt_nie_eine_hoehere_schwelle` |
| Populationsdatei vollständig neu gerechnet (Band 1e-5 absolut) | `tests/test_tabellen_und_baender.py::test_populationsdatei_wird_vollstaendig_neu_gerechnet_im_band_1e_5` |
| Läufe des Spleiß-Falls neu gerechnet im Band | `tests/test_tabellen_und_baender.py::test_spleiss_fall_10_neu_gerechnet_liegt_im_band_um_die_eingefrorene_messung` |
| Läufe des Vehikels B neu gerechnet im Band | `tests/test_tabellen_und_baender.py::test_vehikel_b_neu_gerechnet_im_band` |
| Kalibrierung G neu gerechnet im Band; Spanne unter 1,224 | `tests/test_tabellen_und_baender.py::test_kritische_werte_der_regel_g_liegen_unter_dem_wert_bei_bekannten_parametern` |
| Verdikt Experiment 1 (k ≤ 192 von 200, 0,50 / 0,78, −0,53) | `tests/test_verdicts.py::test_verdikt_stabilitaet_spleiss_fall_nennt_k_und_werte` |
| Verdikt Experiment 2 (Faktor 9,4, Vorzeichenwechsel) | `tests/test_verdicts.py::test_verdikt_fehler_spleiss_fall_vorzeichenwechsel_und_scheinbares_minimum` |
| Verdikt Experiment 3 (Anteile, niedrigste Stufe, Rauschgrenze) | `tests/test_verdicts.py::test_verdikt_regeln_spleiss_fall_nennt_anteile_und_kalibrierung` |
| Hauptfalle „die Regel wählt die niedrigste Schwelle“ | `tests/test_verdicts.py::test_hauptfalle_niedrigste_schwelle_art_und_zaehleinheit` |
| Verdikt Experiment 4 (Streuung der sieben Regeln) | `tests/test_verdicts.py::test_verdikt_streuung_der_regeln` |
| Ein Preset ohne Effekt (Exponential) meldet keinen Effekt | `tests/test_verdicts.py::test_ein_preset_ohne_effekt_meldet_keinen_effekt` |
| Fehlermaße von Hand (Faktor 2, 0,5, Median, Quantile) | `tests/test_verdicts.py::test_error_stats_von_hand_inklusive_grenzen_bei_faktor_2_und_halb` |

Messung dieser README, **nicht im Test**: die Wiederholungen mit den Seed-Basen 8000 und 9000, der Nachlauf der Kalibrierung mit 200 Stichproben, die Laufzeiten (der Läufe, der Suite, der Live-Rechnung) und die Zahl der Tests.

## Ehrliche Grenzen

### Wo die Annahmen enden

| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Unabhängige Beobachtungen** | Serielle Abhängigkeit und gerundete Werte sind hier **nicht gemessen**, ebenso Hill und Schwellenwahl unter Abhängigkeit. Die Literatur (nach Zusammenfassung, ungeprüft) nennt einen positiven Bias der Quantile; die Demo „Spitzen-Cluster“ der Linie (`spitzen-cluster-demo`) misst am M/M/1-Modell bei fester Schwelle einen negativen (bei Last 80 %, n = 3 000, k ≈ 150 und p = 10⁻³ in 90 % (89,6 %) von R = 2 000 Läufen ein Pegel unter dem wahren): wo beides nicht übereinstimmt, gilt deren Messung. | diese Demo zählt nicht, die Demo „Spitzen-Cluster“ der Linie |
| **Ausgedachte Fälle** | Der Spleiß-Fall ist eine **Konstruktion mit Dichtesprung** (Hazard der Überschreitungen 9 gegen 2 bei u₀): sein Kippen ist ein Extremfall, bei glatten Verteilungen driftet der Fehler. Vehikel B ist ausgedacht. | – |
| **Drei eigene Regeln** | S, M und G sind eigene Festlegungen mit diesen Gittern (18 Stufen), keine Literaturverfahren (ForwardStop, Bader u. a., Northrop u. a., Wadsworth/Tawn, Murphy u. a. nicht nachgebaut): die Ergebnisse gelten für diese Regeln, nicht für die Methodenklasse. G ist auf ξ ∈ {0; 0,3} bzw. {0; 0,5} kalibriert und ohne Mehrfachtest-Korrektur. | – |
| **n = 2 000 und 5 000** | Die eingefrorenen Messungen gelten nur für diese Reihenlängen, je Fall ein p (5·10⁻⁵ bzw. 10⁻⁵ je Beobachtung). | – |
| **Simulieren gegen Schätzen** | Hier ist die Wahrheit bekannt, deshalb lässt sich der Fehler zählen; bei echten Daten fehlt sie. Aussagen über den Nutzen in Logistik, Lager oder Hafen werden nicht gemacht. | – |

Die Verweilzeiten und der Spleiß-Fall sind ausgedacht; es gibt in den gelesenen Quellen keine reale Datenquelle aus Hafen, Lager oder Wartezeiten. Gezeigt wird, wie die Wahl der Schwelle in diesem Modell wirkt, und dass die drei geprüften Regeln (eigene Festlegungen, keine Literaturverfahren) im Vehikel B (alle Regeln) und im Spleiß-Fall (S und M) meist nicht besser sind als ein fest gewähltes k, dass G im Spleiß-Fall ohne erkennbaren Vorsprung vor dem besten festen k bleibt und dass M und G bei den glatten Verteilungen (Lognormal, Exponential) besser sind. Was das für einen echten Betrieb heißt, wird hier nicht behauptet.

### Weitere Grenzen

* **Ergebnisse gelten für diese Regeln.** „Keine Regel schlägt ein festes k“ ist nur gegen k = 71 bis 250, im Vehikel B, bei n = 5000 und für die **acht Regel-Varianten** dieser Demo gemessen; gegen ein fest gewähltes k = 500 sind die Regeln S besser. „Regel M trifft in 0,2 %“ gilt nur für M mit c = 2, 18 Stufen, Spleiß-Fall mit Anteil 10 %, R = 400.
* **„Es gibt keine objektiv richtige Schwelle“** wird nicht behauptet: im Spleiß-Fall gibt es sie (u₀); gemessen ist nur, dass es im Vehikel B im Gitter k = 25 bis 1000 keine Schwelle mit Populationsfehler höchstens 0,05 gibt. Die Aussage der Literatur („kein Verfahren ist durchweg das beste“, Belzile/Davison) ist nach Zusammenfassung übernommen, ungeprüft.
* **Gerundete Daten** sind nicht gemessen; die App meldet bei mehr als 10 % Gleichständen unter den größten Werten statt zu wählen.
* **Stabilitätsdiagramme** verwenden überlappende Stichproben (die k größten Werte jeder Stufe enthalten die der höheren) und sagen nichts über die Güte der Anpassung (Literaturhinweis nach Zusammenfassung, ungeprüft).
* **Browser:** die App ist mit AppTest und im Browser-Fenster geprüft (siehe Abschnitt „Tests“); ein echter Mausklick auf Plotly-Punkte kommt nicht vor. Streamlit Cloud ist nicht geprüft; die Tests liefen unter Linux im Docker-Container (siehe Abschnitt „Tests“).

## 🚧 Bewusst nicht enthalten / nicht gemessen

* **Serielle Abhängigkeit und gerundete Daten:** nicht gemessen; Hill und Schwellenwahl unter Abhängigkeit ebenso.
* **Literaturverfahren** (ForwardStop, Bader u. a., Northrop u. a., Wadsworth/Tawn, Murphy u. a., Lomba/Fraga Alves): nicht nachgebaut; die Aussage „viele Regeln wählen die niedrigste Schwelle“ ist nur für S, M, G und diese Gitter gemessen.
* **Regel G:** die Abhängigkeit des kritischen Werts von ξ ist nicht getrennt gemessen (größter Abstand der beiden Kalibrierungen bei gleichem k: siehe Tabelle der Kalibrierung); ohne Mehrfachtest-Korrektur.
* **Vehikel B:** nur n = 5000 (50 Tage zu 100 Container) und p = 10⁻⁵ je Container; Lognormal, Exponential und Spleiß-Fall nur n = 2000; andere n und andere p sind nicht gemessen (der Regler n wirkt nur auf die Reihe oben).
* **Schwellenunsicherheit im Pegel-Intervall** (Fortpflanzung der Wahl auf die Unsicherheit) und **Tail-Fraktion** als eigener Befund: nicht gemessen; die Läufe sind Bänder, kein Intervall.
* **Mean-Residual-Life-Diagramme in geübter Hand:** nicht gemessen; es ist nur die eigene Regel M getestet, und die Ursache ihres Scheiterns ist nicht untersucht.
* **Simulieren gegen Schätzen:** die App kennt die Wahrheit; Aussagen über den Nutzen für Logistik, Lager oder Hafen werden nicht gemacht.
* **Erklärseite „verständlich erklärt“, Grundlagen-Eintrag und PDF-Export:** nicht gebaut; der Footer verweist auf die Linie „Extremwerte erklärt“.

## Quellen

Alles ist Lehrbuch oder Übersicht; **keine Neuheit** (neu ist die Vorführbarkeit mit Zähleinheit und die Gegenüberstellung von Populationsfehler und Läufen). Stufe: **W** = gelesen (Gleichungen und Befunde in eigenen Worten, die Kopien der Bücher haben unklare Weitergaberechte), sonst „nicht gelesen“.

* **Coles (2001)**, *An Introduction to Statistical Modeling of Extreme Values*, Abschnitt 4.3.1 und 4.3.4, S. 78 bis 83, Gleichungen (4.8) bis (4.16): Bias gegen Varianz, Mean-Residual-Life-Diagramm (Steigung ξ/(1 − ξ) oberhalb einer brauchbaren Schwelle, die Deutung sei oft nicht eindeutig), Stabilitätsdiagramm von ξ̂ und σ\*. **W**.
* **Embrechts, Klüppelberg, Mikosch (1997)**, *Modelling Extremal Events*, Satz 3.4.13 (e): e(u) = (σ + ξ·u)/(1 − ξ) der GPD. **W**.
* **Scarrott und MacDonald (2012)**, Überblick über Schwellenwahl: Fixschwellen subjektiv, Schwellenunsicherheit meist vergessen, Faustregeln theoretisch zweifelhaft; der Quellenbericht führt die Arbeit als gelesen, die Stufe ist widersprüchlich (K gelesen, T nur Kurzseite): hier **„nach dem Quellenbericht“, nicht erneut am Text geprüft**; die Faustregeln k = √n und n^(2/3)/ln ln n sind „nach Scarrott/MacDonald laut Quellenbericht, ungeprüft“.
* **Belzile und Davison (2026)**, arXiv 2606.28540: kein Verfahren ist durchweg das beste, viele automatische Regeln wählen die niedrigste Kandidatenschwelle, Regendaten Padova (n = 3 311) mit Hundertjahreswerten zwischen 108 und 124,6 mm bei parametrischen und 48,0, 69,8 und 347,8 mm bei einzelnen semiparametrischen Verfahren, positiver Bias der Quantile bei serieller Abhängigkeit, gerundete Werte kippen Regeln: **nach Zusammenfassung des Abrufwerkzeugs, ungeprüft, Tabellen nicht eingesehen**.
* **Brink u. a. (2005), Wadsworth und Tawn (2012), Northrop u. a. (2017), Bader u. a. (2018), Murphy u. a. (2024), Langousis u. a. (2016):** nur Abstract oder Suchtreffer, **nicht gelesen, nicht nachgebaut**.
* **Pickands, Balkema/de Haan, Hill, Davison/Smith:** nicht im Original gelesen und nicht verwendet.
* Befunde ohne Quelle heißen „eigene Nachrechnung in diesem Modell“. Die Verweilzeit-Mischung stammt aus der Demo „Verweilzeit lernen statt annehmen“ (`verweilzeit-demo`) und ist dort ausgedacht.

## Tests

Die Suite hat **257 Tests** (`pytest --collect-only`). Laufzeit der ganzen Suite seriell 312 s in einem Lauf (Windows, Python 3.14.6, numpy 2.5.3, scipy 1.18.1, streamlit 1.65.0, ein Thread, der Rechner war von anderen Aufgaben belegt; eine Messung dieser README, nicht im Test). Im Linux-Container (Docker, Python 3.12, numpy 2.5.5, scipy 1.18.1, streamlit 1.67.0, `demo_linux_check`) liefen die 257 Tests in 171 s, 2 davon übersprungen (die beiden Kopier-Tests, die die `verweilzeit-demo` brauchen), die übrigen grün; die Demo-CI auf GitHub ist nicht gemessen. Zusammensetzung:

* **Handbeispiele** `tests/test_handbeispiele.py` (34 Tests): H 1 bis H 6 des Plans als exakte Tests (Brüche, Ganzzahlen, Tafelwerte).
* **Orakel** mit anderem Rechenweg (scipy nur hier): `tests/test_oracle_gpd_fit.py` (14 Tests: ML gegen `genpareto.fit`, Gradient am eigenen Ergebnis, Verteilungsfunktion, Pegel, KS-Abstand gegen `kstest`), `tests/test_oracle_population.py` (23 Tests: wahre Pegel, e(u) gegen Quadratur, Populations-ML gegen verschachteltes `quad`, Lognormal gegen Quadratur über s, Vehikel B mit Brent statt Tabelle, exakte Familien), `tests/test_oracle_vehicle.py` (7 Tests: eigene Aufzählung von Vehikel B, Pegel gegen scipy, Ziehung als Band), `tests/test_oracle_rules.py` (12 Tests: S, M, G gegen eine zweite Umsetzung auf je 300 Zufallsinstanzen, Rückfälle, Tabelle je Schwelle gegen Einzelanpassung).
* **Eigenschaften und Randfälle** `tests/test_eigenschaften_und_randfaelle.py` (17 Tests): Skalierung, Verschiebung, Reihenfolge, Monotonie in c, Wiederholbarkeit; kurze Reihen, Gleichstände, alle Werte gleich, ML am Rand, Gitter mit wenigen Stufen, ξ̂ über 1, Schwelle genau bei u₀, Regler an den Rändern.
* **Zufall und Fälle** `tests/test_rng_und_szenario.py` (14 Tests), **eingefrorene Tabellen und Neu-Rechnen als Band** `tests/test_tabellen_und_baender.py` (15 Tests), **Kopien** `tests/test_copies.py` (5 Tests), **Claims** `tests/test_claims.py` (14 Tests), **Verdikte und Fehlermaße** `tests/test_verdicts.py` (30 Tests), **Presets** `tests/test_presets.py` (18 Tests), **Diagramme** `tests/test_visualisierung.py` (5 Tests).
* **Oberfläche:** `tests/test_app.py` (3 Tests, jeder Knopf und jeder Zahlenregler an den Grenzen), `tests/test_app_regler.py` (27 Tests, jede Option jedes Auswahlreglers, jedes Preset, Permalink, Expander), `tests/test_layout.py` (16 Tests), `tests/test_footer.py` (2 Tests), `tests/test_runtime_imports.py` (1 Test).

Zufallsergebnisse werden nie auf das letzte Bit verglichen (Linux-CI, `log` und `pow` aus der C-Bibliothek), sondern als Band mit Standardfehler; Brüche und Ganzzahlen sind exakt. Die Prüfung mit Docker (`demo_linux_check.py`) fand unter Linux zuerst nur eine Abweichung: `sigma_star` der Populationsdatei lag im Vehikel B um bis zu 2,35e-6 neben der eingefrorenen Datei (8 von 95 Zeilen über 1e-6, alle anderen Größen unter 1e-6; kein Rechenfehler). Das Band des Tests ist darum 1e-5 absolut, und dieselbe Suite lief im Container grün. Die App wurde zusätzlich real gestartet (Streamlit, Port 10087) und im Browser-Fenster durchgeklickt, siehe den Bericht des Baus.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.streamlit/config.toml`, `.github/workflows/tests.yml` | Streamlit-Einstellungen (keine Nutzungsstatistik), CI mit Canary |
| `app.py` | Streamlit-App: Kopf, Schnellstart, vier Experimente, Grenzen, Formeln |
| `sws_algorithm.py` | Kern: GPD-ML (numpy), Pegel, KS-Abstand, Gitter, Regeln S, M, G, feste k, Reihe von Anfang bis Ende |
| `sws_numeric.py` | goldener Schnitt, Nelder-Mead, Bisektion (nur numpy und Standardbibliothek) |
| `sws_cases.py` | Fälle: Ziehen, wahre Pegel, exaktes e(u), abgeleitete Zufalls-Seeds |
| `sws_vehicle.py` | Vehikel B: Tabelle, Überlebensfunktion, Pegel, Ziehen |
| `sws_population.py` | Populationsgrößen ohne Zufall (Quadratur), exakte Familien |
| `sws_evaluation.py` | Fehlermaße, Aggregation, Reihe live, Verdikte als reine Funktionen |
| `sws_reference.py` | lädt die eingefrorenen Messungen |
| `sws_rng.py` | Kopie der Vorlage `rng_splitmix64_exact.py` (unverändert, Kopier-Test) |
| `sws_mixture.json`, `sws_mixture_quelle.json` | Tabelle von Vehikel B (unveränderte Kopie, Quelle und Stand je Feld) |
| `sws_gof_crit.json`, `sws_population.json`, `sws_rules.json` | eingefrorene Kalibrierung, Populationskurven und Läufe mit Quelle und Stand |
| `sws_constants.py`, `sws_presets.py`, `sws_layout.py`, `sws_texts.py`, `sws_visualization.py` | Regler-Grenzen und Presets, Permalink und Schnellstart, fester Kopf, Texte, Plotly-Abbildungen |
| `tools/sws_make_crit.py`, `tools/sws_make_population.py`, `tools/sws_make_rules.py` | erzeugen und prüfen die eingefrorenen Dateien (`--check`) |
| `requirements.txt`, `requirements-dev.txt` | Laufzeit (streamlit, numpy, pandas, plotly) und Test (plus pytest, scipy nur als Gegenprobe) |
| `tests/` | Handbeispiele, Orakel (scipy nur hier), Eigenschaften, Randfälle, Bänder, Claims, Verdikte, Presets, Layout, AppTest, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

* **scipy, scikit-learn, statsmodels in der App.** Die App rechnet mit numpy und der Standardbibliothek; scipy steht nur in `requirements-dev.txt` und im Werkzeug `tools/sws_make_population.py` (schneller Fehlerfunktions-Aufruf, mit Rückfall auf `math.erfc`).
* **Rückverweise und Imports aus anderen Demos.** Die Vehikel-B-Tabelle ist eine Kopie; nur der Kopier-Test liest die `verweilzeit-demo`, falls vorhanden.
* **Hill-Schätzer, Block-Maxima und Abhängigkeit:** gehören zu den Demos „GEV an Blockmaxima“, „Pegel statt Index“ und „Spitzen-Cluster“ der Linie.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py --server.port 10087
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Neu rechnen der eingefrorenen Dateien (Werkzeuge, mehrere Minuten): `python tools/sws_make_crit.py`, `python tools/sws_make_population.py`, `python tools/sws_make_rules.py`; jeweils mit `--check` zum Vergleich als Band.

Gebaut mit Streamlit, Plotly, NumPy und pandas.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Extremwerte: Blöcke, Schwelle, Pegel](https://sebastianhanisch.net/konzepte-extremwert.html).
