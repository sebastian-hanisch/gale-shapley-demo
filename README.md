# Gale–Shapley – Vorlieben statt Kosten, Stabilität statt Optimum – Streamlit-Demo

Sechstes Stück der **Matching-Linie** der "Konzepte"-Reihe für die Website "Sebastian Hanisch – Operations Research und Machine Learning", ein **unabhängiger Ast neben der Kostenlinie** (Wurzel: [Greedy-Matching](https://github.com/sebastian-hanisch/greedy-matching-demo)):
anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo **ein** Verfahren – **Gale–Shapley** (aufgeschobene Annahme) – an einem wachsenden Beispiel.
Die Ungarische Methode findet die billigste Zuordnung. Hier haben beide Seiten **Vorlieben** (jedes Fahrzeug eine Rangliste der Aufträge, jeder Auftrag eine der Fahrzeuge), und gesucht ist eine **stabile** Paarung: kein **blockierendes Paar**, also kein Fahrzeug und kein Auftrag, die einander lieber hätten als ihre jetzigen Partner. Eine Seite schlägt vor, die andere hält immer den besten Vorschlag und lehnt den Rest ab.
Gemessen wird der **Preis der Stabilität** gegen das Optimum – und **wer vorschlägt, entscheidet**: der Vorschlagende bekommt seine beste, der Empfänger seine schlechteste stabile Paarung.

**Wichtigster Befund der Vorarbeit:** ordnen **beide Seiten allein nach der Fahrzeit**, hat jedes Paar eine gemeinsame Wertung, und es gibt **genau eine** stabile Paarung – die Greedy-Paarung „Billigste Kante zuerst“ der Wurzel-Demo (auf allen getesteten Karten). Dann wären Verband, Vorschlagenden-Umschalter und Manipulation leer. Deshalb hat die Demo drei **Vorliebenmodelle**: *Nur Entfernung* (Greedy ist die stabile Paarung), **Entfernung mit Streuung** (Standard: jede Seite schätzt die Fahrzeit selbst, ±k Minuten, weiter aus der Entfernung abgeleitet) und *Zufall* (unabhängig von den Kosten).

**Einordnung in die Reihe (die Kanten des Graphen):** dieses Stück ersetzt das Ziel „billig“ durch „stabil“. Seine eigenen Schwächen sind die Ansatzpunkte der nächsten: es gibt nur zwei getrennte Seiten (**Stabile Mitbewohner**: dort muss keine stabile Paarung existieren; **Krankenhaus-Zulassung**: many-to-one; **Top Trading Cycles**: jeder besitzt schon etwas statt zu präferieren; **Nierentausch**: Kompatibilität statt Präferenz), alles ist vorab bekannt (**Online-Matching**). Die gesamte Matching-Linie ist inzwischen vollständig gebaut (13 Stücke).
```
greedy-matching-demo (Wurzel: eine gewählte Zuordnung bleibt)                     [gebaut]
  ├─ augmenting-path-demo (Verbesserungswege: Paare optimal, Kosten blind)        [gebaut]
  │    ├─ hopcroft-karp-demo (viele kürzeste Wege je Phase)                       [gebaut]
  │    ├─ hungarian-demo (Ungarische Methode: Paare zuerst, dann Kosten)           [gebaut]
  │    │    └─ auction-algorithm-demo (Auktionsalgorithmus: dezentral)             [gebaut]
  │    └─ blossom-demo (allgemeine Graphen: ungerade Kreise, Kontraktion)          [gebaut]
  │        └─ weighted-blossom-demo (Ungarisch + Blossom, Konvergenz)              [gebaut]
  ├─ gale-shapley-demo (Vorlieben statt Kosten, stabil)                            [dieses Stück]
  │    ├─ stabile-mitbewohner-demo (eine Gruppe statt zwei Seiten)                 [gebaut]
  │    ├─ krankenhaus-zulassung-demo (many-to-one, Kapazitäten)                    [gebaut]
  │    └─ top-trading-cycles-demo (Tausch ohne Geld, Wohnungsmarkt)                [gebaut]
  │         └─ nierentausch-demo (Kompatibilität statt Präferenz, kurze Zyklen)    [gebaut]
  └─ online-matching-demo (Aufträge kommen nacheinander)                          [gebaut]
```

## Ergebnis (Zahlen aus den Tests)

Jede hier genannte Zahl ist in `tests/test_claims.py` über die 100 festen Karten (Seeds 100000–100099) belegt (20 Fahrzeuge, 20 Aufträge, Reichweite 40, wo nichts anderes steht; Vorschlagende sind die Fahrzeuge). Aufwand = **Vorschläge**, nie Sekunden. Kosten sind über verschiedene Paarzahlen nicht vergleichbar; der Preis der Stabilität ist deshalb die **Prämie gegenüber der billigsten Paarung mit derselben Paarzahl** (aus den Grenzkosten der Ungarischen Methode), die verlorenen Paare stehen getrennt. Mittel und Median stehen zusammen.

| Frage | Ergebnis |
|---|---|
| Reine Entfernung | ✅ Genau **eine** stabile Paarung, gleich dem Greedy „Billigste Kante zuerst“, auf allen 100 Karten (und auf 180 weiteren in sechs Einstellungen getestet): 16,7 von 19,5 Paaren, Prämie im Median 3,4 % (Mittel 4,0 %). ❌ Das **Optimum ist auf allen 100 Karten instabil** (im Mittel 13,2 blockierende Paare, Median 12); Greedy „Auftrag für Auftrag“ hat 11,1 (Median 10,5). |
| Streuung ±20 Minuten (Standard) | ⚠️ Auf **34 von 100 Karten mehr als eine** stabile Paarung (im Mittel 1,6, höchstens 8), genau dort macht die Wahl der Vorschlagenden etwas aus. Prämie im Median **24,3 %** (Mittel 25,8 %), 2,25 Paare weniger als das Optimum. Bei Reichweite 150: 47 von 100 Karten, höchstens 12 stabile Paarungen, Prämie 26,0 %. |
| Zufallsvorlieben | ❌ Reichweite 150: auf **allen** 100 Karten mehr als eine stabile Paarung (im Mittel 6,35, Median 6, höchstens 22), Prämie im Median **162 %** (Mittel 168 %). Reichweite 40: 40 von 100 Karten, Prämie 64,6 %. |
| Wer vorschlägt, gewinnt | ✅ Zufall, Reichweite 150: der Vorschlagende bekommt im Mittel Rang 2,2 (0 = erste Wahl), der Empfänger 4,9; schlagen die Aufträge vor, drehen sich die Ränge um (5,2 und 2,0). Die Gesamtkosten unterscheiden sich kaum (1 079 gegen 1 068 Minuten). |
| Kurze Reichweite | ✅ Bei Reichweite 10 (nur Entfernung) ist das Optimum auf **79 von 100 Karten selbst stabil**, die Prämie liegt im Median bei 0 % (Mittel 0,07 %): Stabilität kostet dort fast nichts. |
| Ungleich große Seiten | ⚠️ 10 Fahrzeuge, 20 Aufträge: schlagen die Fahrzeuge vor, sind es im Mittel 14,5 Vorschläge (Median 14), schlagen die Aufträge vor 120,8 (Median 120) – die größere Seite geht ihre Listen bis zum Ende durch. Die Paarzahl (10) ist in jeder stabilen Paarung gleich (Landklinikensatz). |
| Streuung sweepen | ✅ Von 0 bis 60 Minuten steigt die Prämie streng von 3,9 % auf 48,1 % (40 Karten je Wert), die verlorenen Paare sinken von 2,65 auf 1,85 (mit mehr Streuung sind die Listen weniger einheitlich). Bei 0 gibt es nie mehr als eine stabile Paarung. |
| Anzahl stabiler Paarungen | ⚠️ Zufallsvorlieben, alles erreichbar, n = m = 4 / 8 / 12 / 20 / 40: im Mittel 1,57 / 2,52 / 3,27 / 5,8 / 14,47 (höchstens 30 bei n = 40); der Anteil der Karten mit V ≠ A wächst von 42 % auf 100 %. Mit Streuung ±20 nur 1,02 bis 2,4. |
| Aufwand | ⚠️ Vollständige Zufallslisten: 18 / 52 / 159 / 359 / 1 025 / 2 031 Vorschläge bei n = 10 / 20 / 40 / 80 / 160 / 320, etwa n · H_n (n · ln n); konstanter mittlerer Grad fast linear (19 bis 940, Steigung ≈ 1,1). Entfernungslisten (vollständig) wachsen schneller (4 372 bei n = 320). Die **Straße** (alle einer Meinung) braucht genau **n (n + 1) / 2** Vorschläge (55 / 210 / 820 bei n = 10 / 20 / 40), das Maximum n² − n + 1 für vollständige Listen wird nicht erreicht (n = 3 erschöpfend: 7 gegen 6). |
| Lohnt sich Lügen? | ✅ Erschöpfend auf 5 × 5-Karten (20 Karten, alle geordneten Teillisten, Zufallsvorlieben): **kein Fahrzeug** (Vorschlagende) kann gewinnen (0 von 100); **32 von 100 Aufträgen** (Empfänger) können – genau die mit verschiedenen Extrempaarungen. Kürzen auf einen einzigen Eintrag (den auftragsoptimalen Partner) sichert diesen Partner in 32 von 32 Fällen. Bei Streuung 20 gibt es auf 5 × 5 nichts zu gewinnen. |

## Was nicht funktioniert hat / widerlegte Vorab-Hypothesen

- **„Vorlieben aus der Entfernung erzeugen viele stabile Paarungen.“** Nein: dieselbe Wertung auf beiden Seiten ⇒ eine globale Rangfolge aller Paare ⇒ genau eine stabile Paarung (= Greedy). Nach den Messungen der Vorarbeit gilt das auch mit gegenläufigen Tie-Breaks und mit anderen *separablen* Kriterien (Entfernung plus Zusatzattribut, Gewichte, Profit, Geschwindigkeit); diese Vorversuche sind nicht Teil der Tests. Erst unabhängige Schätzfehler oder Zufall erzeugen Kreise.
- **„Gleichstände erzeugen mehrere stabile Paarungen.“** Nein, mit Index-Tie-Break auf beiden Seiten bleibt die stabile Paarung eindeutig.
- **„Das Optimum ist nie stabil.“** Falsch bei kurzer Reichweite: bei Reichweite 10 ist es auf 79 von 100 Karten stabil. Die Aussage gilt pro Einstellung.
- **„Der Aufwand ist n ln n.“** Nur für vollständige Listen; bei konstantem mittleren Grad ist er fast linear, und Entfernungslisten wachsen schneller als Zufallslisten.
- **„Die Kosten der stabilen Paarung lassen sich mit dem Optimum vergleichen.“** Nicht direkt: die stabile Paarung verliert bei Reichweite 40 auf 99 von 100 Karten Paare und ist schon deshalb billiger; deshalb die Prämie gegen die billigste Paarung gleicher Paarzahl.
- **Abgrenzung:** die Verbandsaufzählung ist über Rotationen (bis 2000 stabile Paarungen); Brute Force nur als Testorakel für n, m ≤ 6 (120 kleine Karten, keine Abweichung).

## Was die Demo zeigt

- **Vorlieben und Ablauf:** Schritt-Slider und ▶️ über die Vorschläge: die Karte mit den vorläufigen Paaren und dem letzten Ereignis (angenommen, verdrängt, abgelehnt), daneben die Vorlieben von Vorschlagendem und Empfänger und der Verlauf; das letzte Bild ist die **stabile Paarung mit ihrem Beweis** (S1–S6). Umschalter für Vorlieben (Entfernung / Streuung / Zufall), Streuung und Vorschlagende (Fahrzeuge / Aufträge).
- **Stabil – und was kostet das?** Ergebnis, Vergleichstabelle (Ungarisch, beide Greedy, beide Vorschlagende) mit Prämie und blockierenden Paaren, ein Prüfer für **blockierende Paare** jeder Paarung (auch des Optimums), Verteilung über 100 feste Karten mit Mittel und Median.
- **Alle stabilen Paarungen:** Anzahl, Landklinikensatz, Streudiagramm der Rangsummen mit den Extrempaarungen, Optimum und Greedy.
- **Lohnt sich Lügen?** Erschöpfende Probe auf 5 × 5 und Kürzen auf einen Eintrag.
- **Wovon hängt es ab?** Streuungs-Sweep, Anzahl gegen Kartengröße, Aufwand gegen Kartengröße mit der Straße.
- **Feste Lehrbuchkarten:** die billigste Kante klaut, gegenläufige Vorlieben (2 stabile Paarungen), Knuths 4 × 4 (10 stabile Paarungen), die Straße; **Wo die Annahmen enden.**

## Modell und Verfahren

- **Vorlieben:** strikt, Index-Tie-Break; Fahrzeug i ordnet Aufträge nach c_ij (+ ε), Auftrag j Fahrzeuge nach c_ij (+ ε). Streuung ganzzahlig: u aus `SplitMix64` je (Seed, Seite, Fahrzeug, Auftrag), ε = ((2k+1)·u >> 20) − k, **über k gekoppelt** (monotone Sweeps) und unabhängig von Reichweite und n, m; Zufall analog mit 30 Bit.
- **Aufgeschobene Annahme:** ein Vorschlagender fragt seine Liste von oben nach unten; der Empfänger hält vorläufig den besten Vorschlag. Ergebnis: stabil, unabhängig von der Bearbeitungsreihenfolge, beste stabile Paarung für die Vorschlagenden, schlechteste für die Empfänger; Vorschlagszahl = Σ (Rang des Partners + 1) bzw. Listenlänge.
- **Verband:** Rotationen ab der fahrzeugoptimalen Paarung; bei unvollständigen Listen ist ein nächster Kandidat, den diese Paarung unversorgt lässt, eine Sackgasse (Landklinikensatz).
- **Preis der Stabilität:** mit den Grenzkosten W_k der Ungarischen Methode ist Σ_{i≤k} W_i die billigste Paarung mit k Paaren; Prämie = (Kosten − Σ)/Σ.
- **Prüfer:** unabhängige Doppelschleife über alle möglichen Paare (S2), getrennt vom Verfahren.

## Dateien

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche |
| `gs_constants.py` | Regler-Grenzen, Presets und Hilfetexte, feste Seed-Mengen |
| `gs_presets.py` | Permalink, Preset- und Zufalls-Seed-Logik (Standardmuster des Portfolios, kopiert und um Vorlieben, Streuung und Vorschlagende ergänzt) |
| `gs_scenario.py` | Karten, eigener Zufallsgenerator, feste Karten (aus den Vorgängerdemos kopiert, dazu gegenläufige Vorlieben, Knuths 4×4 und die Straße) |
| `gs_greedy.py`, `gs_augment.py`, `gs_hungarian.py` | Greedy-Regeln, Verbesserungswege und Ungarische Methode aus den Vorgängerdemos (kopiert, ohne Import; die Ungarische Methode liefert Optimum und Grenzkosten) |
| `gs_preferences.py` | **Neu:** Vorliebenmodelle (Entfernung, Streuung, Zufall) |
| `gs_algorithm.py` | **Neu:** aufgeschobene Annahme, Ereignisprotokoll, Prüfer für blockierende Paare, Beweis |
| `gs_lattice.py` | **Neu:** alle stabilen Paarungen über Rotationen; Brute Force als Orakel |
| `gs_strategy.py` | **Neu:** erschöpfende Manipulationsprobe, Kürzen auf einen Eintrag |
| `gs_evaluation.py` | Einordnung, Prämie, Vergleichstabelle, Verteilung, Sweeps, Aufwand |
| `gs_visualization.py` | Plotly-Abbildungen (Achsen gesperrt für Touch-Geräte; Bögen bei Punkten auf einer Geraden; keine Legende auf den Karten, da sie in schmalen Spalten die Zeichenfläche verdrängt) |
| `tests/` | Algorithmus (Handfälle, Invarianten je Vorschlag, unabhängiger Prüfer, Brute Force gegen Rotationen, Sätze, Manipulation, Negativkontrollen), Auswertung, Presets, belegte Zahlen, AppTest-Rauchtests (auch Abspielen auf mehrbildrigen Karten) |

Die Kopien der Vorgänger werden durch Tests bewacht (Zufallsgenerator-Vektor, Seed-2-Karte: 17 / 232 / 20 / 316). Alle Daten sind synthetisch; die Laufzeit braucht nur numpy, pandas, plotly und streamlit (scipy und networkx sind reine Testorakel).

## Lokal starten

```bash
python -m venv venv
venv\Scripts\pip install -r requirements.txt
venv\Scripts\streamlit run app.py
```

## Tests ausführen

```bash
venv\Scripts\pip install -r requirements-dev.txt
venv\Scripts\python -m pytest tests -v
```

Die Logik rechnet ausschließlich mit ganzen Zahlen; die im Text genannten Anteile und Mittelwerte sind deshalb auf jeder Plattform identisch.
Die CI (`.github/workflows/tests.yml`) läuft auf Ubuntu mit Python 3.12, bei jedem Push und wöchentlich mit den jeweils neuesten Bibliotheksversionen.
