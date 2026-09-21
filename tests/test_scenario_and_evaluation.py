"""Auswertung: Einordnung, Verdict, Vergleichstabelle, blockierende Paare, Verteilung über viele Karten, Sweeps, Manipulation."""

import dataclasses

import pytest

import gs_constants as C
import gs_evaluation as ev
from gs_scenario import build, generate


def test_build_fixed_nets_ignore_random_parameters():
    a, b = build("road", 5, 5, 99, 100, 123), build("road", 30, 30, 10, 0, 1)
    assert a.vehicles == b.vehicles and a.n == 8
    assert build("steal", 1, 1, 1, 1, 1).n == 2 and build("p4", 1, 1, 1, 1, 1).n == 2 and build("cross2", 1, 1, 1, 1, 1).lists is not None and build("latin4", 1, 1, 1, 1, 1).n == 4


@pytest.mark.parametrize("net,count,cost,n_stable,same_vo", [("steal", 2, 18, 1, True), ("p4", 1, 2, 1, True), ("cross2", 2, 80, 2, False), ("latin4", 4, None, 10, False), ("road", 8, 320, 1, True)])
def test_verdict_of_the_fixed_nets(net, count, cost, n_stable, same_vo):
    level, code, d = ev.verdict(ev.analyse(build(net, 20, 20, 40, 0, 2), "dist", 0, 2, "V"))
    assert (level, code) == ("success", ev.STABLE) and d["count"] == count and d["n_stable"] == n_stable and d["same_vo"] == same_vo and d["blocking"] == 0
    assert cost is None or d["cost"] == cost
    assert d["cert"]["all_ok"] and d["cert"]["s2"] and not d["capped"]


def test_verdict_none_without_any_feasible_edge():
    sc = next(s for s in (generate(3, 3, 10, 0, k) for k in range(200)) if not s.feasible.any())
    level, code, d = ev.verdict(ev.analyse(sc, "dist", 0, 0, "V"))
    assert (level, code) == ("info", ev.NONE) and d["count"] == 0


def test_verdict_flags_an_unstable_result():
    a = ev.analyse(generate(20, 20, 40, 0, 97), "noise", 20, 97, "V")
    broken = dataclasses.replace(a.result, pairs=a.result.pairs[1:])
    level, code, d = ev.verdict(dataclasses.replace(a, result=broken))
    assert (level, code) == ("error", ev.UNSTABLE) and d["blocking"] > 0


def test_premium_is_measured_against_the_cheapest_matching_with_the_same_number_of_pairs():
    a = ev.analyse(generate(20, 20, 40, 0, 97), "noise", 20, 97, "V")
    k = a.result.count
    assert a.kcost[a.hung.count] == a.hung.cost and a.kcost[k] <= ev.cost_of(a.scenario, a.result.pairs)
    assert ev.premium_pct(a.scenario, a.result.pairs, a.kcost) > 0
    assert ev.premium_pct(a.scenario, a.hung.pairs, a.kcost) == 0.0


def test_compare_table_rows_and_named_matchings():
    a = ev.analyse(generate(20, 20, 40, 0, 97), "noise", 20, 97, "V")
    rows = ev.compare_table(a)
    assert [r["label"] for r in rows] == ["Ungarische Methode (Optimum)", "Greedy: billigste Kante zuerst", "Greedy: Auftrag für Auftrag", "Stabil: Fahrzeuge schlagen vor", "Stabil: Aufträge schlagen vor"]
    assert rows[0]["premium_pct"] == 0.0 and rows[0]["blocking"] > 0 and rows[3]["blocking"] == 0 == rows[4]["blocking"] and rows[3]["proposals"] > 0
    named = ev.named_matchings(a)
    assert list(named)[:5] == ["Ergebnis (gewählte Vorschlagende)", "Andere Vorschlagende", "Optimum (Ungarische Methode)", "Greedy: billigste Kante zuerst", "Greedy: Auftrag für Auftrag"]
    assert sum(1 for k in named if k.startswith("Stabile Paarung Nr.")) == 4 == a.lattice["count"]


def test_distribution_fields_and_shares():
    d = ev.distribution(20, 20, 40, 0, "noise", 20)
    assert d["n_seeds"] == 100 == d["n_valid"] and len(d["premium_v"]) <= 100 and d["multi_share"] + d["unique_share"] == pytest.approx(1.0)
    assert d["capped_share"] == 0 and d["n_stable_max"] >= 2 and d["opt_unstable_share"] == 1.0


def test_distribution_is_repeatable_and_ignores_the_noise_value_for_other_models():
    assert ev.distribution(12, 12, 40, 25, "noise", 20) == ev.distribution(12, 12, 40, 25, "noise", 20)
    assert ev.distribution(12, 12, 40, 0, "dist", 20)["premium_median"] == ev.distribution(12, 12, 40, 0, "dist", 55)["premium_median"]


def test_distribution_without_feasible_pairs():
    bad = tuple(s for s in range(60) if not generate(3, 3, 10, 0, s).feasible.any())
    assert len(bad) >= 3
    assert ev.distribution(3, 3, 10, 0, "noise", 20, seeds=bad)["n_valid"] == 0


def test_cell_rows_are_integers():
    rows = ev.cell_rows(10, 10, 40, 0, "noise", 20, tuple(C.SWEEP_SEEDS[:5]))
    for row in rows:
        for key, val in row.items():
            if key in ("capped", "same_vo", "same_greedy"):
                assert isinstance(val, bool)
                continue
            flat = [x for v in (val if isinstance(val, tuple) else (val,)) for x in (v if isinstance(v, tuple) else (v,))]
            assert all(isinstance(v, int) for v in flat), key


def test_noise_sweep_count_vs_size_effort_and_manipulation_shapes():
    rows = ev.noise_sweep(10, 10, 40, 0, seed_set=C.SWEEP_SEEDS[:8], ks=(0, 20))
    assert [r["k"] for r in rows] == [0, 20] and rows[0]["multi"] == 0.0
    cs = ev.count_vs_size("random", 20, ns=(4, 6), seeds=C.SIZE_SEEDS[:6])
    assert [r["n"] for r in cs] == [4, 6] and all(r["n_stable"] >= 1 for r in cs)
    ef = ev.effort_scaling("random", 20, ns=(10, 20), seeds=C.EFFORT_SEEDS[:2])
    assert [r["road"] for r in ef] == [55, 210] and all(r["full"] > 0 and r["sparse"] > 0 for r in ef)
    big = ev.effort_scaling("random", 20, ns=(80,), seeds=C.EFFORT_SEEDS[:1])
    assert big[0]["road"] is None                                                              # die Straße gilt nur bis n = 40
    mp = ev.manipulation("random", 0, size=3, seeds=C.MANIP_SEEDS[:5])
    assert mp["veh_gain"] == 0 and mp["veh_n"] == 15 and mp["ord_n"] == 15
