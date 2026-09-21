"""Jede Zahl in den Hilfetexten, Presets, Tabellen und Grenzen der App und der README ist hier über die 100 festen Karten (DIST_SEEDS) belegt.
Alles rechnet mit ganzen Zahlen und einem eigenen Zufallsgenerator - die Werte sind auf jeder Plattform dieselben; die Toleranzen decken nur die Rundung auf die im Text genannten Stellen.
Positive UND negative Aussagen: wo Stabilität fast nichts kostet (kurze Reichweite, reine Entfernung) und wo viel (Zufallsvorlieben). Mittel und Median stehen zusammen."""

import pytest

import gs_evaluation as ev


def near(value, expected, tol):
    assert abs(value - expected) <= tol, f"{value:.3f} statt {expected}"


@pytest.fixture(scope="module")
def dist40():
    return ev.distribution(20, 20, 40, 0, "dist", 0)


@pytest.fixture(scope="module")
def noise40():
    return ev.distribution(20, 20, 40, 0, "noise", 20)


@pytest.fixture(scope="module")
def random150():
    return ev.distribution(20, 20, 150, 0, "random", 20)


# --- reine Entfernung ------------------------------------------------------------------------------------------------------------------------

def test_pure_distance_is_unique_and_equals_greedy_cheapest_edge(dist40):
    assert dist40["multi_share"] == 0.0 and dist40["n_stable_max"] == 1 and dist40["vo_differ_share"] == 0.0 and dist40["same_greedy_share"] == 1.0
    near(dist40["stable_pairs_mean"], 16.7, 0.05)                                                   # "16,7 von 19,5 Paaren"
    near(dist40["opt_pairs_mean"], 19.5, 0.05)
    near(dist40["lost_mean"], 2.72, 0.005)
    assert dist40["edge_blocking_mean"] == 0.0                                                       # Greedy selbst ist stabil
    near(dist40["premium_median"], 3.4, 0.05)                                                        # "Prämie im Median 3,4 %"
    near(dist40["premium_mean"], 4.0, 0.05)


def test_the_optimum_is_unstable_and_the_order_greedy_too(dist40):
    assert dist40["opt_unstable_share"] == 1.0                                                       # das Optimum ist auf allen 100 Karten instabil
    near(dist40["opt_blocking_mean"], 13.15, 0.005)
    assert dist40["opt_blocking_median"] == 12.0
    near(dist40["order_blocking_mean"], 11.1, 0.005)
    assert dist40["order_blocking_median"] == 10.5


def test_short_reach_stability_costs_almost_nothing():
    d = ev.distribution(20, 20, 10, 0, "dist", 0)
    near(d["opt_unstable_share"], 0.21, 0.0001)                                                     # "Optimum auf 79 von 100 Karten selbst stabil"
    assert d["premium_median"] == 0.0                                                                # "Prämie im Median 0 %"
    near(d["premium_mean"], 0.07, 0.005)
    near(d["stable_pairs_mean"], 7.08, 0.005)
    near(d["opt_pairs_mean"], 7.28, 0.005)


def test_all_reachable_distance_preferences_still_lose_money_but_no_pairs():
    d = ev.distribution(20, 20, 150, 0, "dist", 0)
    assert d["opt_unstable_share"] == 1.0 and d["lost_mean"] == 0.0 and d["multi_share"] == 0.0        # bei Reichweite 150 ist das Optimum auf keiner Karte stabil
    near(d["premium_median"], 14.1, 0.05)


# --- Streuung --------------------------------------------------------------------------------------------------------------------------------

def test_noise_20_makes_several_stable_matchings_and_the_price_grows(noise40):
    near(noise40["multi_share"], 0.34, 0.0001)                                                       # "auf 34 von 100 Karten mehr als eine"
    near(noise40["n_stable_mean"], 1.6, 0.05)                                                        # "im Mittel 1,6"
    assert noise40["n_stable_max"] == 8 and noise40["n_stable_median"] == 1.0                        # "höchstens 8"
    near(noise40["vo_differ_share"], 0.34, 0.0001)                                                   # V ≠ A genau auf den Karten mit mehreren stabilen Paarungen
    near(noise40["premium_median"], 24.3, 0.05)                                                      # "Prämie im Median 24,3 %"
    near(noise40["premium_mean"], 25.8, 0.05)
    assert noise40["premium_median"] > 5 * ev.distribution(20, 20, 40, 0, "dist", 0)["premium_median"]       # viel teurer als bei reiner Entfernung
    near(noise40["stable_pairs_mean"], 17.2, 0.05)
    assert noise40["opt_unstable_share"] == 1.0 and noise40["edge_blocking_mean"] > 10          # Greedy ist unter Streuung nicht mehr stabil


def test_noise_20_all_reachable():
    d = ev.distribution(20, 20, 150, 0, "noise", 20)
    near(d["multi_share"], 0.47, 0.0001)
    assert d["n_stable_max"] == 12
    near(d["premium_median"], 26.0, 0.05)


def test_the_noise_sweep_is_monotone_in_price_and_starts_unique():
    rows = ev.noise_sweep(20, 20, 40, 0)
    ks = [r["k"] for r in rows]
    assert ks == [0, 5, 10, 20, 30, 40, 60] and rows[0]["multi"] == 0.0 and rows[0]["n_stable"] == 1.0
    prem = [r["premium_median"] for r in rows]
    assert all(a < b for a, b in zip(prem, prem[1:])) and prem[0] < 4.5 and prem[-1] > 45          # 3,9 % bis 48,1 %
    lost = [r["lost"] for r in rows]
    assert all(a >= b for a, b in zip(lost, lost[1:])) and lost[0] > 2.6 and lost[-1] < 1.9          # mit der Streuung gehen weniger Paare verloren (2,65 bis 1,85)
    assert rows[-1]["multi"] > rows[1]["multi"] > 0.0


# --- Zufallsvorlieben und wer vorschlägt -------------------------------------------------------------------------------------------------------

def test_random_preferences_at_all_reachable(random150):
    assert random150["multi_share"] == 1.0                                                           # "auf allen 100 Karten mehr als eine stabile Paarung"
    near(random150["n_stable_mean"], 6.35, 0.005)                                                    # "im Mittel 6,3" (6,35)
    assert random150["n_stable_median"] == 6.0 and random150["n_stable_max"] == 22                   # "höchstens 22"
    near(random150["premium_median"], 162.4, 0.05)                                                   # "im Median 162 %"
    near(random150["premium_mean"], 167.6, 0.05)


def test_who_proposes_wins_and_the_total_cost_hardly_changes(random150):
    near(random150["rank_v_in_v"], 2.19, 0.005)                                                      # Vorschlagende: Rang 2,2
    near(random150["rank_o_in_v"], 4.88, 0.005)                                                      # Empfänger: 4,9
    near(random150["rank_v_in_o"], 5.20, 0.005)
    near(random150["rank_o_in_o"], 2.02, 0.005)
    assert random150["rank_v_in_v"] < random150["rank_v_in_o"] and random150["rank_o_in_o"] < random150["rank_o_in_v"]     # jede Seite ist besser dran, wenn sie vorschlägt
    assert abs(random150["cost_v_mean"] - random150["cost_o_mean"]) < 0.02 * random150["cost_v_mean"]                  # Gesamtkosten fast gleich (1079 gegen 1068)
    near(random150["cost_v_mean"], 1079, 0.6)
    near(random150["cost_o_mean"], 1068, 0.6)


def test_random_preferences_at_mid_reach():
    d = ev.distribution(20, 20, 40, 0, "random", 20)
    near(d["multi_share"], 0.40, 0.0001)
    near(d["premium_median"], 64.6, 0.05)
    assert d["lost_mean"] < ev.distribution(20, 20, 40, 0, "dist", 0)["lost_mean"]                 # negativ: bei Zufall gehen weniger Paare verloren als bei reiner Entfernung (1,62 gegen 2,72)


# --- Ungleich große Seiten ----------------------------------------------------------------------------------------------------------------------

def test_more_orders_than_vehicles_the_larger_side_proposes_much_more():
    d = ev.distribution(10, 20, 150, 0, "noise", 20)
    near(d["proposals_v_mean"], 14.5, 0.05)
    assert d["proposals_v_median"] == 14.0
    near(d["proposals_o_mean"], 120.8, 0.05)                                                         # "121"
    assert d["proposals_o_median"] == 120.0
    assert d["stable_pairs_mean"] == 10.0 == d["opt_pairs_mean"]                                     # die Paarzahl ist in jeder stabilen Paarung gleich: 10


# --- Verband und Aufwand ---------------------------------------------------------------------------------------------------------------------------

def test_the_number_of_stable_matchings_grows_with_the_map():
    rows = {r["n"]: r for r in ev.count_vs_size("random", 20)}
    assert [round(rows[n]["n_stable"], 2) for n in (4, 8, 12, 20, 40)] == [1.57, 2.52, 3.27, 5.8, 14.47]
    assert rows[40]["n_stable_max"] == 30 and [round(rows[n]["vo"], 2) for n in (4, 8, 12, 20, 40)] == [0.42, 0.75, 0.8, 1.0, 1.0]
    noise = {r["n"]: r for r in ev.count_vs_size("noise", 20)}
    assert round(noise[40]["n_stable"], 1) == 2.4 and round(noise[4]["n_stable"], 2) == 1.02 and noise[40]["n_stable_max"] == 12                         # "Mit Streuung ±20 nur 1,02 bis 2,4"
    assert all(rows[n]["capped"] == 0 for n in rows)


def test_effort_random_lists_grow_like_n_h_n_and_distance_lists_faster():
    rows = ev.effort_scaling("random", 20)
    full = [round(r["full"]) for r in rows]
    assert full == [18, 52, 159, 359, 1025, 2031] and [round(r["sparse"]) for r in rows] == [19, 44, 108, 226, 466, 940]
    assert all(0.6 <= r["full"] / r["nHn"] <= 1.3 for r in rows[2:])                                # vollständige Zufallslisten: etwa n · H_n
    import numpy as np
    slope = np.polyfit(np.log([r["n"] for r in rows]), np.log([r["sparse"] for r in rows]), 1)[0]
    assert 0.9 < slope < 1.3                                                                          # konstanter mittlerer Grad: fast linear
    dist = ev.effort_scaling("dist", 20)
    assert [round(r["full"]) for r in dist] == [29, 70, 206, 554, 2032, 4372] and dist[-1]["full"] > 2 * dist[-1]["nHn"]        # Entfernungslisten wachsen schneller


def test_the_road_needs_exactly_n_n_plus_one_over_two_proposals_up_to_n_40():
    rows = {r["n"]: r for r in ev.effort_scaling("random", 20)}
    assert [rows[n]["road"] for n in (10, 20, 40)] == [55, 210, 820] and all(rows[n]["road"] is None for n in (80, 160, 320))


# --- Manipulation ----------------------------------------------------------------------------------------------------------------------------------

def test_manipulation_experiment():
    mp = ev.manipulation("random", 0)
    assert (mp["veh_n"], mp["veh_gain"]) == (100, 0)                                                # Vorschlagende gewinnen nie
    assert (mp["ord_n"], mp["ord_gain"], mp["ord_diff"], mp["ord_diff_gain"]) == (100, 32, 32, 32)  # Empfänger gewinnen genau bei verschiedenen Extremen
    assert (mp["trunc_ok"], mp["trunc_n"]) == (32, 32)                                              # Kürzen auf einen Eintrag sichert den Partner immer
    calm = ev.manipulation("noise", 20)
    assert calm["ord_diff"] == 0 and calm["ord_gain"] == 0                                           # bei Streuung 20 auf 5 x 5-Karten gibt es nichts zu gewinnen
    pure = ev.manipulation("dist", 0)
    assert pure["ord_gain"] == 0 == pure["veh_gain"]
