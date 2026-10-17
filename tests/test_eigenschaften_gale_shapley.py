"""Eigenschaftstests (metamorph) für Gale-Shapley: Aussagen, die für die aufgeschobene Annahme gelten MÜSSEN, ohne dass man das Ergebnis vorher kennt.

Man verändert die Eingabe auf eine Weise, deren Wirkung auf die Ausgabe bekannt ist, und prüft diese Beziehung auf festen Zufallsinstanzen
(Seeds fest, je Eigenschaft 25 bis 50 Instanzen) plus Randfällen. Die Listen werden hier selbst erzeugt (beliebige, auch unvollständige
beidseitige Listen), nicht über `gs_preferences`; das vorhandene Orakel (test_oracle_stability.py) prüft dagegen Brute Force und Definition.

Geprüft wird:
 E1 Umnummerieren der Vorschlagenden und Empfänger permutiert Paarung und Vorschlagszahl nur.
 E2 Spiegelung: Rollen vertauschen (Empfänger schlagen vor) ist dasselbe wie "O" vorschlagen lassen; auch die Menge aller stabilen Paarungen spiegelt sich.
 E3 Verband bei größeren Karten (n bis 25 nur mit den beiden Extrempaarungen, n <= 7 mit allen): jede aufgezählte Paarung ist nach eigener Definition stabil,
    die vorschlagende Seite steht in ihrer Extrempaarung komponentenweise am besten, die andere am schlechtesten.
 E4 Zusätzlicher Empfänger: jeder Vorschlagende ist danach weakly besser, jeder bisherige Empfänger weakly schlechter dran (Gale/Sotomayor, vorschlagenoptimale Paarung).
 E5 Aufwertung: hebt ein Empfänger einen Vorschlagenden in seiner Liste, ist dieser weakly besser dran (Monotonie der aufgeschobenen Annahme).
 E6 Anreizverträglichkeit der Vorschlagenden: keine zufällige Falschangabe (Teilliste in beliebiger Reihenfolge) verbessert den eigenen Partner (Dubins/Freedman, Roth).
 E7 Schwache Pareto-Optimalität: keine Paarung, ob stabil oder nicht, stellt alle Vorschlagenden zugleich strikt besser als die Vorschlagenden-optimale stabile Paarung (Brute Force, n, m <= 5).
 E8 Vollständige Listen: Paarzahl = min(n, m) auf beiden Seiten, Vorschlagszahl zwischen Mindestmaß und der Knuth-Schranke n^2 - n + 1 (n = m).
 E9 Reine Entfernung: positive affine Transformation der Kosten (c * Kosten + d, c > 0) ändert die Vorlieben und damit die stabile Paarung nicht.
 E10 Randfälle: leere Eingaben, leere Listen, ein Vorschlagender, ein Empfänger, n != m.
"""

import itertools
import random
from dataclasses import replace

import pytest

from gs_algorithm import blocking_pairs, gale_shapley, stable_matching
from gs_lattice import all_stable
from gs_preferences import preferences
from gs_scenario import generate


# --- Hilfen ----------------------------------------------------------------------------------------------------------------------------------

def _market(rng, n, m, p=1.0):
    """Zufällige beidseitige Listen: jedes Paar ist mit Wahrscheinlichkeit p möglich, alle Listen zufällig geordnet."""
    edges = [(i, j) for i in range(n) for j in range(m) if p >= 1.0 or rng.random() < p]
    pv = [[j for a, j in edges if a == i] for i in range(n)]
    po = [[i for i, b in edges if b == j] for j in range(m)]
    for lst in pv + po:
        rng.shuffle(lst)
    return pv, po


def _pos(lst, x):
    """Platz von x in der Liste (0 = beste Wahl); None (unversorgt) ist schlechter als jeder Platz."""
    return len(lst) if x is None else lst.index(x)


def _by_prop(pp, pr):
    """gale_shapley mit Listen der Vorschlagenden pp und Empfänger pr: (Vorschlagender -> Empfänger, Empfänger -> Vorschlagender, Vorschläge)."""
    mt, proposals, _, _ = gale_shapley(pp, pr, record=False)
    return mt, {r: p for p, r in mt.items()}, proposals


def _blocking_own(pv, po, pairs):
    """Eigene Definition eines blockierenden Paars (nicht gs_algorithm.blocking_pairs)."""
    mv, mo = dict(pairs), {j: i for i, j in pairs}
    out = set()
    for i in range(len(pv)):
        for j in pv[i]:
            if mv.get(i) == j or i not in po[j]:
                continue
            if _pos(pv[i], mv.get(i)) > pv[i].index(j) and _pos(po[j], mo.get(j)) > po[j].index(i):
                out.add((i, j))
    return out


def _random_cases(seed, count, nmax=8, p_choices=(0.5, 0.8, 1.0)):
    rng = random.Random(seed)
    for _ in range(count):
        n, m = rng.randint(1, nmax), rng.randint(1, nmax)
        yield rng, n, m, rng.choice(p_choices)


# --- E1 Umnummerieren ------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("who", ["V", "O"])
def test_e1_relabelling_agents_only_permutes_the_result(who):
    for rng, n, m, p in _random_cases(11, 40, nmax=12):
        pv, po = _market(rng, n, m, p)
        qv, qo = rng.sample(range(n), n), rng.sample(range(m), m)        # neue Namen: Fahrzeug i heißt qv[i], Auftrag j heißt qo[j]
        pv2, po2 = [None] * n, [None] * m
        for i in range(n):
            pv2[qv[i]] = [qo[j] for j in pv[i]]
        for j in range(m):
            po2[qo[j]] = [qv[i] for i in po[j]]
        a, b = stable_matching(pv, po, who), stable_matching(pv2, po2, who)
        assert set(b.pairs) == {(qv[i], qo[j]) for i, j in a.pairs}
        assert a.proposals == b.proposals and a.count == b.count


# --- E2 Spiegelung ---------------------------------------------------------------------------------------------------------------------------

def test_e2_swapping_the_roles_mirrors_the_result_and_the_set_of_all_stable_matchings():
    for rng, n, m, p in _random_cases(12, 40, nmax=7):
        pv, po = _market(rng, n, m, p)
        o = stable_matching(pv, po, "O")                                  # Aufträge schlagen vor
        mirrored = stable_matching(po, pv, "V")                           # dieselbe Rechnung mit vertauschten Namen der Seiten
        assert set(mirrored.pairs) == {(j, i) for i, j in o.pairs} and mirrored.proposals == o.proposals
        fwd = {frozenset(mm) for mm in all_stable(pv, po)["matchings"]}
        bwd = {frozenset((i, j) for j, i in mm) for mm in all_stable(po, pv)["matchings"]}
        assert fwd == bwd                                                  # die Aufzählung über Rotationen hängt nicht von der Startseite ab


# --- E3 Verband ------------------------------------------------------------------------------------------------------------------------------

def test_e3_extreme_matchings_are_stable_and_ordered_for_larger_markets():
    for rng, n, m, p in _random_cases(13, 40, nmax=25, p_choices=(0.2, 0.5, 1.0)):
        pv, po = _market(rng, n, m, p)
        v, o = stable_matching(pv, po, "V"), stable_matching(pv, po, "O")
        assert not _blocking_own(pv, po, v.pairs) and not _blocking_own(pv, po, o.pairs)
        mv, mo = dict(v.pairs), dict(o.pairs)
        assert set(mv) == set(mo) and set(mv.values()) == set(mo.values())          # Landklinikensatz: dieselben Agenten sind versorgt
        for i in mv:
            assert pv[i].index(mv[i]) <= pv[i].index(mo[i])                          # Fahrzeuge: V-optimal ist weakly besser als O-optimal
        ov, oo = {j: i for i, j in v.pairs}, {j: i for i, j in o.pairs}
        for j in oo:
            assert po[j].index(oo[j]) <= po[j].index(ov[j])                          # Aufträge: O-optimal ist weakly besser als V-optimal


def test_e3_every_enumerated_stable_matching_lies_between_the_extremes():
    for rng, n, m, p in _random_cases(14, 50, nmax=7):
        pv, po = _market(rng, n, m, p)
        lat = all_stable(pv, po)
        v, o = dict(stable_matching(pv, po, "V").pairs), {i: j for i, j in stable_matching(pv, po, "O").pairs}
        for mm in lat["matchings"]:
            assert not _blocking_own(pv, po, mm)
            d = dict(mm)
            assert set(d) == set(v)
            for i in d:
                assert pv[i].index(v[i]) <= pv[i].index(d[i]) <= pv[i].index(o[i])      # Fahrzeug i: V-optimal <= beliebig stabil <= O-optimal
        assert frozenset(stable_matching(pv, po, "V").pairs) in lat["matchings"] and frozenset(stable_matching(pv, po, "O").pairs) in lat["matchings"]


# --- E4 zusätzlicher Empfänger ---------------------------------------------------------------------------------------------------------------

def test_e4_adding_a_receiver_makes_every_proposer_weakly_better_and_every_old_receiver_weakly_worse():
    for rng, n, m, p in _random_cases(15, 50, nmax=9):
        pv, po = _market(rng, n, m, p)
        mt, mo, _ = _by_prop(pv, po)
        takers = [i for i in range(n) if rng.random() < 0.6]                    # wer die neue Empfängerin überhaupt möglich findet
        pv2 = [list(x) for x in pv]
        for i in takers:
            pv2[i].insert(rng.randint(0, len(pv2[i])), m)                         # sie bekommt den Namen m
        new_list = list(takers)
        rng.shuffle(new_list)
        po2 = [list(x) for x in po] + [new_list]
        mt2, mo2, _ = _by_prop(pv2, po2)
        for i in range(n):
            assert _pos(pv2[i], mt2.get(i)) <= _pos(pv2[i], mt.get(i))             # Vorschlagende: weakly besser (Platz in der neuen Liste)
        for j in range(m):
            assert _pos(po[j], mo2.get(j)) >= _pos(po[j], mo.get(j))               # bisherige Empfänger: weakly schlechter


# --- E5 Aufwertung ---------------------------------------------------------------------------------------------------------------------------

def test_e5_being_ranked_higher_by_a_receiver_never_hurts_the_proposer():
    tried = 0
    for rng, n, m, p in _random_cases(16, 50, nmax=8, p_choices=(0.7, 1.0)):
        pv, po = _market(rng, n, m, p)
        if not any(len(x) > 1 for x in po):
            continue
        j = rng.choice([k for k in range(m) if len(po[k]) > 1])
        lst = po[j]
        a = rng.randrange(1, len(lst))
        i = lst[a]
        new = [x for x in lst if x != i]
        new.insert(rng.randrange(0, a), i)                                           # i rückt in der Liste von j nach oben, alle anderen bleiben in ihrer Reihenfolge
        po2 = [list(x) for x in po]
        po2[j] = new
        mt, _, _ = _by_prop(pv, po)
        mt2, _, _ = _by_prop(pv, po2)
        assert _pos(pv[i], mt2.get(i)) <= _pos(pv[i], mt.get(i))
        tried += 1
    assert tried >= 30


# --- E6 Anreizverträglichkeit der Vorschlagenden ---------------------------------------------------------------------------------------------

def test_e6_no_random_misreport_helps_a_proposer():
    checked = 0
    for rng, n, m, p in _random_cases(17, 40, nmax=6, p_choices=(0.7, 1.0)):
        pv, po = _market(rng, n, m, p)
        mt, _, _ = _by_prop(pv, po)
        for i in range(n):
            if not pv[i]:
                continue
            honest = _pos(pv[i], mt.get(i))
            for _ in range(8):
                k = rng.randint(1, len(pv[i]))
                lie = rng.sample(pv[i], k)                                              # beliebige geordnete Teilliste
                pv2 = [list(x) for x in pv]
                pv2[i] = lie
                mt2, _, _ = _by_prop(pv2, po)
                assert _pos(pv[i], mt2.get(i)) >= honest                                 # nach den WAHREN Vorlieben nie besser
                checked += 1
    assert checked > 500


# --- E7 schwache Pareto-Optimalität ----------------------------------------------------------------------------------------------------------

def _all_matchings(pv, po, i=0, used=frozenset()):
    if i == len(pv):
        yield {}
        return
    yield from _all_matchings(pv, po, i + 1, used)
    for j in pv[i]:
        if j not in used and i in po[j]:
            for rest in _all_matchings(pv, po, i + 1, used | {j}):
                yield {i: j, **rest}


def test_e7_no_matching_makes_all_proposers_strictly_better_than_the_proposer_optimal_stable_matching():
    for rng, n, m, p in _random_cases(18, 40, nmax=5):
        pv, po = _market(rng, n, m, p)
        mt, _, _ = _by_prop(pv, po)
        for mu in _all_matchings(pv, po):
            assert not all(_pos(pv[i], mu.get(i)) < _pos(pv[i], mt.get(i)) for i in range(n))


# --- E8 vollständige Listen ------------------------------------------------------------------------------------------------------------------

def test_e8_complete_lists_match_min_n_m_and_respect_the_proposal_bounds():
    rng = random.Random(19)
    for _ in range(40):
        n, m = rng.randint(1, 12), rng.randint(1, 12)
        pv, po = _market(rng, n, m, 1.0)
        for who in ("V", "O"):
            res = stable_matching(pv, po, who)
            assert res.count == min(n, m)
            assert not _blocking_own(pv, po, res.pairs)
            assert res.proposals >= min(n, m)
        if n == m:
            for who in ("V", "O"):
                assert stable_matching(pv, po, who).proposals <= n * n - n + 1      # Schranke für vollständige Listen (README: Maximum n^2 - n + 1)


# --- E9 affine Transformation der Kosten -----------------------------------------------------------------------------------------------------

def test_e9_pure_distance_preferences_ignore_positive_affine_cost_transformations():
    rng = random.Random(20)
    for seed in range(100, 130):
        sc = generate(rng.randint(3, 12), rng.randint(3, 12), rng.choice([30, 60, 150]), rng.choice([0, 50]), seed)
        c, d = rng.randint(1, 9), rng.randint(0, 50)
        sc2 = replace(sc, cost=sc.cost * c + d)                                       # gleiche Karte, Kosten c * Kosten + d
        pv, po = preferences(sc, "dist", 0, seed)
        pv2, po2 = preferences(sc2, "dist", 0, seed)
        assert (pv, po) == (pv2, po2)
        for who in ("V", "O"):
            assert stable_matching(pv, po, who).pairs == stable_matching(pv2, po2, who).pairs
        assert not blocking_pairs(pv2, po2, stable_matching(pv2, po2, "V").pairs)


# --- E10 Randfälle ---------------------------------------------------------------------------------------------------------------------------

def test_e10_edge_cases():
    assert gale_shapley([], [], record=False)[:2] == ({}, 0)                          # nichts da
    assert stable_matching([[], []], [], "V").pairs == () and stable_matching([[], []], [], "V").proposals == 0   # Vorschlagende ohne mögliche Partner
    assert stable_matching([], [[], []], "V").pairs == ()                             # nur Empfänger
    one = stable_matching([[0]], [[0]], "V")
    assert one.pairs == ((0, 0),) and one.proposals == 1
    # ein Empfänger, n Vorschlagende: er behält seinen Liebsten, jeder schlägt genau einmal vor
    for n in range(1, 7):
        for perm in itertools.islice(itertools.permutations(range(n)), 12):
            res = stable_matching([[0]] * n, [list(perm)], "V")
            assert res.pairs == ((perm[0], 0),) and res.proposals == n
    # ein Vorschlagender, m Empfänger, die ihn alle wollen: er bekommt seine erste Wahl nach einem Vorschlag
    res = stable_matching([[2, 0, 1]], [[0], [0], [0]], "V")
    assert res.pairs == ((0, 2),) and res.proposals == 1
    # zwei Vorschlagende mit derselben Liste und beide Empfänger bevorzugen Vorschlagenden 1: beide werden versorgt, Vorschlagszahl 3 (der Verdrängte schlägt neu vor)
    res = stable_matching([[0, 1], [0, 1]], [[1, 0], [1, 0]], "V")
    assert res.pairs == ((0, 1), (1, 0)) and res.proposals == 3
