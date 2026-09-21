"""Vorlieben: jede Seite ordnet ihre möglichen Partner (die Reichweite macht die Listen unvollständig, aber beidseitig gleich).

Drei Modelle, alle ganzzahlig und über einen eigenen Zufallsgenerator (SplitMix64) statt numpy:
- `dist`: beide Seiten ordnen nach der Fahrzeit c_ij. Beide Seiten haben dieselbe Wertung ⇒ die stabile Paarung ist eindeutig und gleich dem Greedy
  "Billigste Kante zuerst" der Wurzel-Demo.
- `noise`: jede Seite schätzt die Fahrzeit selbst (Stau, Ortskenntnis): c_ij + Streuung in [-k, k] Minuten, je Seite und Paar unabhängig gezogen. k = 0 ist `dist`,
  große k nähern sich `random`. Die Streuung ist über k gekoppelt (gleiche Grundziehung), damit Sweeps monoton sind, und hängt nicht von Reichweite oder n, m ab.
- `random`: unabhängige Zufallsvorlieben, ohne Bezug zu den Kosten.
Gleichstände werden nach dem Index gebrochen (Fahrzeuge nach Auftragsindex, Aufträge nach Fahrzeugindex); die Listen sind also strikt.
"""

from gs_scenario import SplitMix64

MASK = (1 << 64) - 1
PREF_LABELS = {"dist": "Nur Entfernung", "noise": "Entfernung mit Streuung", "random": "Zufall"}
DEFAULT_PREF = "noise"
NOISE_MIN, NOISE_MAX, NOISE_STEP, DEFAULT_NOISE = 5, 60, 5, 20
SIDE_V, SIDE_O = 1, 2


def _draw(seed, side, i, j, bits):
    """Reproduzierbare Ziehung 0 .. 2^bits - 1 für (Seed, Seite, Fahrzeug, Auftrag): unabhängig von n, m und Reichweite."""
    state = seed & MASK
    for part in (side, i, j):
        state = (state * 1000003 + part + 1) & MASK
    return SplitMix64(state).below(1 << bits)


def _noise(seed, side, i, j, k):
    return ((2 * k + 1) * _draw(seed, side, i, j, 20) >> 20) - k


def preferences(sc, model, noise=DEFAULT_NOISE, seed=0):
    """Strikte Vorlieben (pv, po): pv[i] = Aufträge von i, beste zuerst; po[j] = Fahrzeuge von j, beste zuerst. Karten mit festen Listen liefern diese unverändert."""
    if getattr(sc, "lists", None) is not None:
        pv, po = sc.lists
        return [list(x) for x in pv], [list(x) for x in po]
    n, m, c = sc.n, sc.m, sc.cost
    if model == "dist":
        kv = lambda i, j: int(c[i, j])
        ko = lambda j, i: int(c[i, j])
    elif model == "noise":
        kv = lambda i, j: int(c[i, j]) + _noise(seed, SIDE_V, i, j, noise)
        ko = lambda j, i: int(c[i, j]) + _noise(seed, SIDE_O, i, j, noise)
    elif model == "random":
        kv = lambda i, j: _draw(seed, SIDE_V, i, j, 30)
        ko = lambda j, i: _draw(seed, SIDE_O, i, j, 30)
    else:
        raise ValueError(model)
    pv = [sorted((j for j in range(m) if sc.feasible[i, j]), key=lambda j: (kv(i, j), j)) for i in range(n)]
    po = [sorted((i for i in range(n) if sc.feasible[i, j]), key=lambda i: (ko(j, i), i)) for j in range(m)]
    return pv, po


def ranks(lists):
    """Rangtabellen: ranks[a][b] = Platz von b in der Liste von a (0 = beste)."""
    return [{x: k for k, x in enumerate(lst)} for lst in lists]
