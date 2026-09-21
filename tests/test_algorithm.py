"""Gale–Shapley: Wächter für die kopierten Vorgänger, Handfälle, Vorlieben, Invarianten je Vorschlag, Orakel (unabhängiger Prüfer, Brute Force, Verband), Sätze (Vorschlagenden-Optimalität,
Empfänger-Pessimalität, Landklinikensatz, Reihenfolgeunabhängigkeit), reine Entfernung = Greedy, Manipulation und Negativkontrollen."""

import itertools

import pytest

from gs_algorithm import blocking_pairs, certificate, gale_shapley, is_valid, stable_matching, state_at
from gs_greedy import RULES, optimum, run_rule
from gs_lattice import all_stable, brute_force_stable
from gs_preferences import _draw, _noise, preferences, ranks
from gs_scenario import SplitMix64, cross2, generate, latin4, p4_chain, road, steal_2x2, travel_cost
from gs_strategy import best_response, truncate_to_one


# --- Wächter für die kopierten Dateien der Vorgänger ---------------------------------------------------------------------------------

def test_splitmix64_reference_vector():
    rng = SplitMix64(0)
    assert [rng.next() for _ in range(2)] == [0xE220A8397B1DCDAF, 0x6E789E6AA1B965F4]


def test_the_root_numbers_are_reproduced_on_the_shared_maps():
    """Seed-2-Karte der Vorgänger: Greedy 17 Paare für 232 Minuten, Optimum 20 für 316; billigste Kante klaut auf der 2x2-Karte (18 gegen 10)."""
    sc = generate(20, 20, 40, 0, 2)
    g, o = run_rule(sc, "edge"), optimum(sc)
    assert (g.count, g.cost, o.count, o.cost) == (17, 232, 20, 316)
    sc = steal_2x2()
    assert sc.cost.tolist() == [[4, 5], [5, 14]] and all((run_rule(sc, r).count, run_rule(sc, r).cost) == (2, 18) for r in RULES) and (optimum(sc).count, optimum(sc).cost) == (2, 10)
    assert travel_cost(3, 4) == (5, 25)


# --- Vorlieben -------------------------------------------------------------------------------------------------------------------------------

def test_noise_draws_are_reproducible_bounded_and_coupled_over_k():
    assert _draw(97, 1, 3, 5, 20) == _draw(97, 1, 3, 5, 20) and _draw(97, 1, 3, 5, 20) != _draw(97, 2, 3, 5, 20)
    for i, j in itertools.product(range(6), repeat=2):
        assert all(-k <= _noise(7, 1, i, j, k) <= k for k in (5, 20, 60))
    u = _draw(7, 1, 2, 3, 20)
    assert all(_noise(7, 1, 2, 3, k) == ((2 * k + 1) * u >> 20) - k for k in range(0, 61, 5))        # gekoppelt: dieselbe Grundziehung für jedes k


def test_preferences_are_strict_mutual_and_independent_of_reach():
    for seed in range(20):
        for model in ("dist", "noise", "random"):
            sc = generate(8, 6, 50, 0, seed)
            pv, po = preferences(sc, model, 20, seed)
            assert all(len(set(x)) == len(x) for x in pv + po)
            assert {(i, j) for i, x in enumerate(pv) for j in x} == {(i, j) for j, x in enumerate(po) for i in x} == {(i, j) for i in range(8) for j in range(6) if sc.feasible[i, j]}
    a, b = generate(8, 6, 150, 0, 3), generate(8, 6, 40, 0, 3)                                     # dieselbe Karte, kleinere Reichweite: dieselbe Rangfolge, nur gekürzt
    pa, _ = preferences(a, "noise", 20, 3)
    pb, _ = preferences(b, "noise", 20, 3)
    assert all([j for j in pa[i] if b.feasible[i, j]] == pb[i] for i in range(8))


def test_distance_preferences_order_by_cost_with_index_ties():
    sc = generate(10, 10, 60, 0, 5)
    pv, po = preferences(sc, "dist", 0, 5)
    assert all(sorted(x, key=lambda j: (int(sc.cost[i, j]), j)) == x for i, x in enumerate(pv))
    assert all(sorted(x, key=lambda i: (int(sc.cost[i, j]), i)) == x for j, x in enumerate(po))


# --- Handfälle -------------------------------------------------------------------------------------------------------------------------------

def test_steal_is_stable_at_18_and_the_optimum_is_blocked_by_the_cheapest_pair():
    sc = steal_2x2()
    pv, po = preferences(sc, "dist", 0, 0)
    res = stable_matching(pv, po, "V")
    assert res.pairs == ((0, 0), (1, 1)) and sum(int(sc.cost[i, j]) for i, j in res.pairs) == 18 and not blocking_pairs(pv, po, res.pairs)
    assert blocking_pairs(pv, po, ((0, 1), (1, 0))) == [(0, 0)]                                    # das Optimum (10 Minuten) wird von V1-A1 blockiert


def test_p4_loses_a_pair_in_every_stable_matching():
    sc = p4_chain(1)
    pv, po = preferences(sc, "dist", 0, 0)
    assert stable_matching(pv, po, "V").count == stable_matching(pv, po, "O").count == 1 and optimum(sc).count == 2


def test_cross2_has_two_stable_matchings_and_each_side_gets_its_first_choice_when_proposing():
    sc = cross2()
    pv, po = preferences(sc, "noise", 20, 0)                                                        # feste Listen: das Modell spielt keine Rolle
    v, o = stable_matching(pv, po, "V"), stable_matching(pv, po, "O")
    assert v.pairs == ((0, 0), (1, 1)) and o.pairs == ((0, 1), (1, 0))
    assert all_stable(pv, po)["count"] == 2 == len(brute_force_stable(pv, po))


def test_latin4_has_ten_stable_matchings_by_rotations_and_by_brute_force():
    pv, po = preferences(latin4(), "dist", 0, 0)
    a, b = all_stable(pv, po), brute_force_stable(pv, po)
    assert a["count"] == 10 == len(b) and set(a["matchings"]) == set(b) and not a["capped"]


@pytest.mark.parametrize("n", [1, 2, 3, 4, 8, 20, 40])
def test_the_road_needs_exactly_n_times_n_plus_one_over_two_proposals_for_both_sides(n):
    sc = road(n)
    pv, po = preferences(sc, "dist", 0, 0)
    assert stable_matching(pv, po, "V").proposals == stable_matching(pv, po, "O").proposals == n * (n + 1) // 2 and stable_matching(pv, po, "V").count == n


def test_exhaustive_worst_case_for_complete_lists_of_size_three_is_seven_proposals():
    """Knuths Schranke n^2 - n + 1 = 7 für n = 3 wird von einer Vorliebenbelegung erreicht (die Straße erreicht sie nicht: 6)."""
    best = 0
    perms = list(itertools.permutations(range(3)))
    for pv_ in itertools.product(perms, repeat=3):
        for po_ in itertools.product(perms, repeat=3):
            best = max(best, gale_shapley([list(x) for x in pv_], [list(x) for x in po_], record=False)[1])
    assert best == 7


# --- Karten für die Invarianten -----------------------------------------------------------------------------------------------------------

def _cases(n_cases=90):
    for s in range(n_cases):
        n, m = 3 + s % 6, 3 + (s * 5) % 6
        for model in ("dist", "noise", "random"):
            yield generate(n, m, 25 + (s * 13) % 130, (s * 25) % 101, s), model, s


def test_result_is_valid_and_stable_for_both_proposers_and_every_model():
    for sc, model, s in _cases():
        pv, po = preferences(sc, model, 20, s)
        for who in ("V", "O"):
            res = stable_matching(pv, po, who)
            assert is_valid(pv, po, res.pairs) and not blocking_pairs(pv, po, res.pairs)


def test_event_log_invariants_proposals_go_down_the_list_and_holders_only_improve():
    for sc, model, s in list(_cases(40)):
        pv, po = preferences(sc, model, 20, s)
        for who, (pp, pr) in (("V", (pv, po)), ("O", (po, pv))):
            res = stable_matching(pv, po, who)
            rr = ranks(pr)
            last_idx, holder = {}, {}
            for p, r, outcome, cur in res.events:
                idx = pp[p].index(r)
                assert idx > last_idx.get(p, -1)                                                    # jeder Vorschlagende geht seine Liste von oben nach unten durch, ohne Wiederholung
                last_idx[p] = idx
                if r in holder:
                    assert outcome != "accept" and (outcome == "displace") == (rr[r][p] < rr[r][holder[r]])      # der Empfänger nimmt genau dann an, wenn der Neue besser ist
                    if outcome == "displace":
                        assert cur == holder[r]
                else:
                    assert outcome == "accept"
                if outcome != "reject":
                    holder[r] = p
            assert res.proposals == len(res.events) == sum(res.ptr)


def test_proposals_identity_certificate_and_replay():
    for sc, model, s in list(_cases(40)):
        pv, po = preferences(sc, model, 20, s)
        res = stable_matching(pv, po, "V")
        cert = certificate(pv, po, res)
        assert cert["s1"] and cert["s2"] and cert["s3"] and cert["s4"] and cert["all_ok"]
        st = state_at(res, sc.n, len(res.events))
        assert sorted((p, r) for r, p in st["holder"].items()) == sorted(res.pairs)
        assert state_at(res, sc.n, 0) == {"holder": {}, "ptr": [0] * sc.n, "matched": set()}


def test_result_does_not_depend_on_the_processing_order():
    for sc, model, s in list(_cases(40)):
        pv, po = preferences(sc, model, 20, s)
        base = gale_shapley(pv, po, record=False)
        for order in (list(reversed(range(sc.n))), sorted(range(sc.n), key=lambda i: (i * 7) % max(sc.n, 1))):
            other = gale_shapley(pv, po, order=order, record=False)
            assert other[0] == base[0] and other[1] == base[1]                                        # gleiche Paarung und sogar gleiche Vorschlagszahl


# --- Orakel: unabhängiger Prüfer, Brute Force, Sätze -------------------------------------------------------------------------------------------

def _small_cases(count=120):
    for s in range(count):
        n, m = 2 + s % 4, 2 + (s * 3) % 4
        yield generate(n, m, 40 + (s * 17) % 110, 0, s), ("random", "noise", "dist")[s % 3], s


def test_brute_force_stable_set_equals_the_rotation_enumeration():
    for sc, model, s in _small_cases():
        pv, po = preferences(sc, model, 40, s)
        assert set(all_stable(pv, po)["matchings"]) == set(brute_force_stable(pv, po))


def test_proposers_get_their_best_and_receivers_their_worst_stable_partner():
    for sc, model, s in _small_cases():
        pv, po = preferences(sc, model, 40, s)
        stable = brute_force_stable(pv, po)
        rv, ro = ranks(pv), ranks(po)
        v = dict(stable_matching(pv, po, "V").pairs)
        o = {j: i for i, j in v.items()}
        for i in range(sc.n):
            got = [rv[i][dict(mm)[i]] for mm in stable if i in dict(mm)]
            if got:
                assert rv[i][v[i]] == min(got)                                                       # Vorschlagende: der beste Partner in irgendeiner stabilen Paarung
        for j in range(sc.m):
            got = [ro[j][{b: a for a, b in mm}[j]] for mm in stable if j in {b for _, b in mm}]
            if got:
                assert ro[j][o[j]] == max(got)                                                       # Empfänger: der schlechteste


def test_rural_hospitals_and_stable_implies_maximal_and_at_least_half():
    for sc, model, s in _small_cases():
        pv, po = preferences(sc, model, 40, s)
        stable = brute_force_stable(pv, po)
        assert len({frozenset(i for i, _ in mm) for mm in stable}) == 1 and len({frozenset(j for _, j in mm) for mm in stable}) == 1
        assert len({len(mm) for mm in stable}) == 1
        nu = optimum(sc).count
        assert all(2 * len(mm) >= nu for mm in stable) and certificate(pv, po, stable_matching(pv, po, "V"))["s3"]


def test_the_stable_matching_has_at_least_half_the_pairs_of_the_optimum_and_p4_is_tight():
    for seed in range(100000, 100100):
        sc = generate(20, 20, 40, 0, seed)
        pv, po = preferences(sc, "random", 20, seed)
        assert 2 * stable_matching(pv, po, "V").count >= optimum(sc).count
    sc = p4_chain(3)
    pv, po = preferences(sc, "dist", 0, 0)
    assert stable_matching(pv, po, "V").count == 3 and optimum(sc).count == 6


def test_pure_distance_gives_a_unique_stable_matching_equal_to_greedy_cheapest_edge():
    """Beide Seiten ordnen nach derselben Kostenmatrix: eine globale Rangfolge der Paare, also genau eine stabile Paarung - das Greedy 'Billigste Kante zuerst'."""
    for cfg in ((20, 20, 40), (20, 20, 10), (20, 20, 150), (10, 20, 150), (20, 10, 40), (40, 40, 60)):
        for seed in range(100000, 100030):
            sc = generate(*cfg, 0, seed)
            pv, po = preferences(sc, "dist", 0, seed)
            v, o = stable_matching(pv, po, "V"), stable_matching(pv, po, "O")
            assert v.pairs == o.pairs == tuple(sorted(run_rule(sc, "edge").pairs)) and all_stable(pv, po)["count"] == 1


# --- Manipulation ----------------------------------------------------------------------------------------------------------------------------------

def test_proposers_never_gain_by_lying_and_receivers_gain_exactly_when_the_extremes_differ():
    receivers_with_choice = 0
    for seed in range(100000, 100025):
        sc = generate(4, 4, 300, 0, seed)
        pv, po = preferences(sc, "random", 0, seed)
        lat = all_stable(pv, po)
        for i in range(sc.n):
            honest, best, _ = best_response(pv, po, "V", i)
            assert best >= honest                                                                 # Vorschlagende (Fahrzeuge) gewinnen nie
        for j in range(sc.m):
            honest, best, _ = best_response(pv, po, "O", j)
            partners = {dict((b, a) for a, b in mm).get(j) for mm in lat["matchings"]}
            assert (best < honest) == (len(partners) > 1)                                        # Empfänger gewinnen genau dann, wenn die Extreme verschieden sind
            if len(partners) > 1:
                receivers_with_choice += 1
                target = min(partners, key=lambda x: ranks(po)[j][x])
                assert truncate_to_one(pv, po, j, target)                                        # Kürzen auf einen Eintrag sichert den auftragsoptimalen Partner
    assert receivers_with_choice > 0


# --- Negativkontrollen -----------------------------------------------------------------------------------------------------------------------------

def _unstable_count(**kw):
    bad = total = 0
    for sc, model, s in list(_cases(100)):
        pv, po = preferences(sc, model, 20, s)
        mt, *_ = gale_shapley(pv, po, record=False, **kw)
        pairs = tuple(sorted(mt.items()))
        total += 1
        bad += bool(blocking_pairs(pv, po, pairs)) or not is_valid(pv, po, pairs)
    return bad, total


def test_negative_controls_catch_wrong_receivers_and_wrong_proposal_order():
    assert _unstable_count()[0] == 0                                                                  # richtig: nie ein blockierendes Paar
    assert _unstable_count(receiver="first")[0] > 20                                                 # der Empfänger behält den ersten Vorschlag
    assert _unstable_count(receiver="worst")[0] > 20                                                 # der Empfänger behält den schlechtesten
    assert _unstable_count(propose="worst")[0] > 20                                                  # die Vorschlagenden beginnen mit ihrer schlechtesten Wahl


def test_ignoring_incomplete_lists_produces_pairs_that_are_not_possible():
    bad = 0
    for sc, model, s in list(_cases(60)):
        full_v = [list(range(sc.m)) for _ in range(sc.n)]
        full_o = [list(range(sc.n)) for _ in range(sc.m)]
        mt, *_ = gale_shapley(full_v, full_o, record=False)
        bad += any(not sc.feasible[i, j] for i, j in mt.items())
    assert bad > 10


def test_the_checker_flags_a_hand_built_unstable_matching_and_a_matching_with_an_impossible_pair():
    sc = steal_2x2()
    pv, po = preferences(sc, "dist", 0, 0)
    assert blocking_pairs(pv, po, ((0, 1), (1, 0))) == [(0, 0)] and blocking_pairs(pv, po, ()) and not is_valid(pv, po, ((0, 0), (0, 0)))
    hard = generate(6, 6, 20, 0, 1)
    pvh, poh = preferences(hard, "dist", 0, 1)
    impossible = next((i, j) for i in range(6) for j in range(6) if not hard.feasible[i, j])
    assert not is_valid(pvh, poh, (impossible,))
