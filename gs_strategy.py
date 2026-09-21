"""Manipulation: lohnt es sich, falsche Vorlieben anzugeben? Erschöpfend über alle geordneten Teillisten (nur kleine Karten).

Bei Vorschlägen der Fahrzeuge gilt (Dubins/Freedman, Roth): kein Fahrzeug kann sich durch falsche Angaben verbessern; ein AUFTRAG (Empfänger) kann es, genau dann, wenn
die beiden Extrempaarungen für ihn verschieden sind. Kürzt ein Auftrag seine Liste auf einen einzigen Eintrag - den Partner, den er in der auftragsoptimalen stabilen
Paarung hat -, bekommt er ihn. Gemessen wird das hier, nicht angenommen.
"""

import itertools

from gs_algorithm import _rank, gale_shapley


def _partner_v(pv, po, i):
    mt, *_ = gale_shapley(pv, po, record=False)
    return mt.get(i)


def _partner_of_order(pv, po, j):
    mt, *_ = gale_shapley(pv, po, record=False)
    return {b: a for a, b in mt.items()}.get(j)


def best_response(pv, po, side, idx):
    """Beste erreichbare Wahl für einen Agenten bei Vorschlägen der Fahrzeuge: (Rang ehrlich, bester Rang mit Falschangabe, Falschangabe). None = unversorgt (schlechter als jeder Rang).
    side "V": Fahrzeug idx gibt beliebige geordnete Teillisten seiner Liste an; side "O": Auftrag idx entsprechend."""
    true_list = pv[idx] if side == "V" else po[idx]
    rk = {x: k for k, x in enumerate(true_list)}
    worst = len(true_list)

    def outcome(pv2, po2):
        p = _partner_v(pv2, po2, idx) if side == "V" else _partner_of_order(pv2, po2, idx)
        return worst if p is None else rk[p]

    honest = outcome(pv, po)
    best, best_list = honest, list(true_list)
    for k in range(1, len(true_list) + 1):
        for sub in itertools.permutations(true_list, k):
            if side == "V":
                pv2 = [list(x) for x in pv]
                pv2[idx] = list(sub)
                po2 = [[i for i in lst if i != idx or j in sub] for j, lst in enumerate(po)]
                res = outcome(pv2, po2)
            else:
                po2 = [list(x) for x in po]
                po2[idx] = list(sub)
                pv2 = [[j for j in lst if j != idx or i in sub] for i, lst in enumerate(pv)]
                res = outcome(pv2, po2)
            if res < best:
                best, best_list = res, list(sub)
    return honest, best, best_list


def truncate_to_one(pv, po, j, target):
    """Auftrag j gibt nur `target` (ein Fahrzeug) an. Rückgabe: bekommt er es (Vorschläge der Fahrzeuge)?"""
    po2 = [list(x) for x in po]
    po2[j] = [target]
    pv2 = [[o for o in lst if o != j or i == target] for i, lst in enumerate(pv)]
    mt, *_ = gale_shapley(pv2, po2, record=False)
    return {b: a for a, b in mt.items()}.get(j) == target
