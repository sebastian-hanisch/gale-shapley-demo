"""Unabhängiges Orakel: Stabilität nach Definition (eigene Schleife über alle Paare, nicht gs_algorithm.blocking_pairs), alle stabilen Paarungen durch
Aufzählen ALLER Paarungen, Optimalität der Vorschlagenden, Pessimalität der Empfänger, Landklinikensatz und Vorschlagszahl gegen ein eigenes,
einzeln und in zufälliger Reihenfolge arbeitendes Gale-Shapley - auf zufälligen unvollständigen Listen mit n, m <= 5."""

import random

from gs_algorithm import blocking_pairs, stable_matching
from gs_lattice import all_stable


def _blocking(pv, po, matching):
    pm, om = dict(matching), {j: i for i, j in matching}
    out = set()
    for i in range(len(pv)):
        for j in range(len(po)):
            if j in pv[i] and i in po[j] and pm.get(i) != j:
                i_wants = i not in pm or pv[i].index(j) < pv[i].index(pm[i])
                j_wants = j not in om or po[j].index(i) < po[j].index(om[j])
                if i_wants and j_wants:
                    out.add((i, j))
    return out


def _all_matchings(pv, po, i=0, used=frozenset()):
    if i == len(pv):
        yield []
        return
    yield from _all_matchings(pv, po, i + 1, used)
    for j in pv[i]:
        if j not in used and i in po[j]:
            for rest in _all_matchings(pv, po, i + 1, used | {j}):
                yield [(i, j)] + rest


def _lists(rng, n, m, p):
    edges = [(i, j) for i in range(n) for j in range(m) if rng.random() < p]
    pv = [[j for a, j in edges if a == i] for i in range(n)]
    po = [[i for i, b in edges if b == j] for j in range(m)]
    for lst in pv + po:
        rng.shuffle(lst)
    return pv, po


def _own_gs(pv, po, rng):
    nxt, hold, free, proposals = [0] * len(pv), {}, list(range(len(pv))), 0
    while free:
        p = free.pop(rng.randrange(len(free)))
        while nxt[p] < len(pv[p]):
            r = pv[p][nxt[p]]
            nxt[p] += 1
            proposals += 1
            if r not in hold:
                hold[r] = p
                break
            if po[r].index(p) < po[r].index(hold[r]):
                free.append(hold[r])
                hold[r] = p
                break
    return frozenset((p, r) for r, p in hold.items()), proposals


def test_stable_matchings_against_brute_force_by_definition():
    rng = random.Random(20261004)
    multi = 0
    for t in range(120):
        n, m = rng.randint(1, 5), rng.randint(1, 5)
        if t % 3 == 0:
            m = n                                                                  # jede dritte Karte: vollständige quadratische Listen
        pv, po = _lists(rng, n, m, 1.0 if t % 3 == 0 else rng.choice([0.4, 0.7, 1.0]))
        stable = [frozenset(mm) for mm in _all_matchings(pv, po) if not _blocking(pv, po, mm)]
        multi += len(stable) > 1
        # Landklinikensatz: dieselben Agenten in jeder stabilen Paarung
        assert len({frozenset(i for i, _ in mm) for mm in stable}) == 1 and len({frozenset(j for _, j in mm) for mm in stable}) == 1
        assert set(all_stable(pv, po)["matchings"]) == set(stable)
        for who in ("V", "O"):
            res = stable_matching(pv, po, who)
            assert frozenset(res.pairs) in stable
        v = stable_matching(pv, po, "V")
        pv_best = dict(v.pairs)
        for i in range(n):
            got = [pv[i].index(dict(mm)[i]) for mm in stable if i in dict(mm)]
            assert (not got) or pv[i].index(pv_best[i]) == min(got)               # Vorschlagende: bester stabiler Partner
        po_of = {j: i for i, j in v.pairs}
        for j in range(m):
            got = [po[j].index({b: a for a, b in mm}[j]) for mm in stable if j in {b for _, b in mm}]
            assert (not got) or po[j].index(po_of[j]) == max(got)                 # Empfänger: schlechtester stabiler Partner
        pairs, proposals = _own_gs(pv, po, rng)
        assert pairs == frozenset(v.pairs) and proposals == v.proposals
        allm = list(_all_matchings(pv, po))
        for mm in rng.sample(allm, min(3, len(allm))):
            assert set(blocking_pairs(pv, po, mm)) == _blocking(pv, po, mm)
    assert multi > 10                                                              # das Orakel ist nicht leer: oft mehrere stabile Paarungen
