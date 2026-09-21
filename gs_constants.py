"""Konstanten, Regler-Grenzen, Presets und feste Seed-Mengen der Demo "Gale–Shapley"."""

# --- Regler (wie in den Vorgängerdemos) ---------------------------------------------------------------------------------
N_MIN, N_MAX, DEFAULT_N = 3, 40, 20          # Fahrzeuge
M_MIN, M_MAX, DEFAULT_M = 3, 40, 20          # Aufträge
REACH_MIN, REACH_MAX, DEFAULT_REACH = 10, 150, 40   # Reichweite in Minuten; ab 142 ist auf der 100x100-Karte alles erreichbar
BALLUNG_MIN, BALLUNG_MAX, DEFAULT_BALLUNG = 0, 100, 0   # ganze Prozent, Schritt 25
DEFAULT_SEED = 97                               # eine Karte, auf der die Streuung mehrere stabile Paarungen ergibt (Seed 165 der Vorgänger hat nur eine)
SEED_MAX = 2_000_000_000

NETS = {"random": "Zufällige Karte", "steal": "Billigste Kante klaut (2×2)", "p4": "Pfad aus vier Punkten", "cross2": "Gegenläufige Vorlieben (2×2)", "latin4": "Knuths 4×4 (10 stabile Paarungen)", "road": "Worst Case: die Straße (8×8)"}
DEFAULT_NET = "random"

FIXED_NETS = ("steal", "p4", "cross2", "latin4", "road")
LIST_NETS = ("cross2", "latin4")              # Karten mit festen Vorlieben: der Vorlieben-Umschalter gilt dort nicht

PREF_LABELS = {"dist": "Nur Entfernung", "noise": "Entfernung mit Streuung", "random": "Zufall"}
DEFAULT_PREF = "noise"
NOISE_MIN, NOISE_MAX, NOISE_STEP, DEFAULT_NOISE = 5, 60, 5, 20
PROPOSER_LABELS = {"V": "Fahrzeuge", "O": "Aufträge"}
DEFAULT_PROPOSER = "V"

# --- feste Seed-Mengen (dieselben wie in den Vorgängerdemos; unabhängig vom Nutzer-Seed) --------------------------------------------
DIST_SEEDS = tuple(range(100000, 100100))
SWEEP_SEEDS = DIST_SEEDS[:40]
SIZE_SEEDS = DIST_SEEDS[:40]
REACH_SWEEP = (10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 120, 150)
NOISE_SWEEP = (0, 5, 10, 20, 30, 40, 60)
COUNT_NS = (4, 8, 12, 20, 40)
EFFORT_NS = (10, 20, 40, 80, 160, 320)
EFFORT_SEEDS = DIST_SEEDS[:5]
SCALE_REACH_AT_20 = 40
MANIP_SEEDS = DIST_SEEDS[:20]

COLORS = {"matched": "#1f77b4", "propose": "#2ca02c", "reject": "#d62728", "displace": "#ff7f0e", "vehicle": "#111111", "order": "#ff7f0e", "block": "#9467bd", "hung": "#d62728",
          "greedy": "#8c564b", "stable_v": "#1f77b4", "stable_o": "#2ca02c"}

# --- Presets ------------------------------------------------------------------------------------------------------------------
_BASE = dict(net="random", pref=DEFAULT_PREF, noise=DEFAULT_NOISE, prop=DEFAULT_PROPOSER, n=DEFAULT_N, m=DEFAULT_M, reach=DEFAULT_REACH, ballung=DEFAULT_BALLUNG, seed=DEFAULT_SEED)
PRESETS = {
    "⚖️ Die billigste Kante klaut": {**_BASE, "net": "steal", "pref": "dist"},
    "🔀 Vorschlagende zählen": {**_BASE, "net": "cross2"},
    "🛣️ Worst Case: die Straße": {**_BASE, "net": "road", "pref": "dist"},
    "📏 Nur Entfernung": {**_BASE, "pref": "dist", "seed": 21},
    "🗺️ Mittlere Reichweite": {**_BASE},
    "🎲 Zufallsvorlieben": {**_BASE, "pref": "random", "reach": 150, "seed": 119},
    "📐 Mehr Aufträge als Fahrzeuge": {**_BASE, "n": 10, "reach": 150, "seed": 168},
    "📡 Knappe Reichweite": {**_BASE, "pref": "dist", "reach": 10, "seed": 13},
}
# Jede Zahl in diesen Texten ist in tests/test_claims.py über die 100 festen Karten (DIST_SEEDS) belegt
PRESET_HELP = {
    "⚖️ Die billigste Kante klaut": "Beide Seiten ordnen nach der Fahrzeit. Das billigste Paar V1–A1 ist stabil (keiner der beiden will weg), und so entsteht die Greedy-Paarung für 18 Minuten. Das Optimum (10 Minuten) ist NICHT stabil: V1 und A1 hätten einander lieber als ihre Partner.",
    "🔀 Vorschlagende zählen": "Gegenläufige Vorlieben, zwei stabile Paarungen: schlagen die Fahrzeuge vor, bekommt jedes Fahrzeug seine erste Wahl (V1–A1, V2–A2); schlagen die Aufträge vor, bekommt jeder Auftrag seine erste Wahl (A1–V2, A2–V1). Wer vorschlägt, entscheidet.",
    "🛣️ Worst Case: die Straße": "Alle Fahrzeuge bevorzugen denselben Auftrag und alle Aufträge dasselbe Fahrzeug: bei n = 8 sind es genau n (n + 1) / 2 = 36 Vorschläge, für beide Vorschlagenden - der geometrische Worst Case (Knuths Schranke n² − n + 1 gilt für vollständige Listen und wird hier nicht erreicht).",
    "📏 Nur Entfernung": "Beide Seiten ordnen nach der Fahrzeit: dieselbe Wertung ⇒ es gibt genau eine stabile Paarung, und sie ist die Greedy-Paarung „Billigste Kante zuerst“ der Wurzel-Demo (auf allen 100 Karten, 16,7 von 19,5 Paaren). Die Prämie gegenüber der billigsten Paarung gleicher Paarzahl liegt im Median bei 3,4 %.",
    "🗺️ Mittlere Reichweite": "Jede Seite schätzt die Fahrzeit selbst, mit Streuung ±20 Minuten. Dann gibt es auf 34 von 100 Karten (20 Fahrzeuge, 20 Aufträge, Reichweite 40) mehr als eine stabile Paarung (im Mittel 1,6, höchstens 8), und die Prämie steigt auf im Median 24,3 %. Diese Karte hat 4 stabile Paarungen.",
    "🎲 Zufallsvorlieben": "Unabhängige Zufallsvorlieben bei Reichweite 150: auf allen 100 Karten mehr als eine stabile Paarung (im Mittel 6,3, höchstens 22). Wer vorschlägt, gewinnt: der Vorschlagende bekommt im Mittel seinen 3. Partner (Rang 2,2), der Empfänger seinen 6. (4,9). Die Prämie liegt im Median bei 162 %.",
    "📐 Mehr Aufträge als Fahrzeuge": "10 Fahrzeuge, 20 Aufträge: schlagen die Fahrzeuge vor, sind es im Mittel 14,5 Vorschläge, schlagen die Aufträge vor 121 - die größere Seite geht ihre Listen bis zum Ende durch. Die Paarzahl ist in jeder stabilen Paarung gleich (10, Landklinikensatz).",
    "📡 Knappe Reichweite": "Wenige mögliche Paare: bei Reichweite 10 ist das Optimum auf 79 von 100 Karten selbst stabil - Stabilität kostet dort fast nichts (Prämie im Median 0 %).",
}
