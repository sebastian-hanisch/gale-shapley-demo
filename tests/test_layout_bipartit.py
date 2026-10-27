"""Darstellung "Zwei Seiten": Fahrzeuge links, Aufträge rechts. Nur Koordinaten ändern sich; Kosten, Vorlieben und Paare bleiben die der Karte."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

import gs_constants as C
import gs_evaluation as ev
from gs_algorithm import blocking_pairs, state_at
from gs_scenario import build
from gs_visualization import _two_sided_order, arrange, build_gs_map

APP = Path(__file__).resolve().parent.parent / "app.py"
CASES = [("random", 20, 20, 40, 0, 97), ("random", 8, 8, 25, 50, 7), ("random", 5, 12, 40, 0, 3), ("steal", 20, 20, 40, 0, 97), ("p4", 20, 20, 40, 0, 97),
         ("cross2", 20, 20, 40, 0, 97), ("road", 20, 20, 40, 0, 97)]


@pytest.mark.parametrize("params", CASES)
def test_arrange_puts_each_side_in_its_own_column_with_distinct_places(params):
    sc = build(*params)
    arr = arrange(sc, "bipartit")
    assert {x for x, _ in arr.vehicles} == {0} and {x for x, _ in arr.orders} == {100}
    assert len(set(arr.vehicles)) == sc.n and len(set(arr.orders)) == sc.m          # jede Ecke hat ihren eigenen Platz
    assert all(0 <= y <= 100 for _, y in arr.vehicles + arr.orders)


@pytest.mark.parametrize("params", CASES)
def test_arrange_changes_only_the_coordinates(params):
    sc = build(*params)
    arr = arrange(sc, "bipartit")
    assert np.array_equal(arr.cost, sc.cost) and np.array_equal(arr.feasible, sc.feasible) and arr.reach == sc.reach and arr.lists == sc.lists
    assert arr.n == sc.n and arr.m == sc.m


def test_arrange_map_is_the_identity_and_is_repeatable():
    sc = build("random", 20, 20, 40, 0, 97)
    assert arrange(sc, "map") is sc
    first, second = arrange(sc, "bipartit"), arrange(sc, "bipartit")
    assert first.vehicles == second.vehicles and first.orders == second.orders


def test_every_edge_of_the_two_sided_drawing_joins_the_two_columns():
    """Das ist die Aussage "bipartit" im Bild: jede gezeichnete mögliche Kante beginnt links (Fahrzeug) und endet rechts (Auftrag)."""
    sc = build("random", 12, 12, 40, 0, 97)
    arr = arrange(sc, "bipartit")
    for i in range(sc.n):
        for j in range(sc.m):
            if sc.feasible[i, j]:
                assert arr.vehicles[i][0] < arr.orders[j][0]


def test_barycenter_order_does_not_cross_more_than_the_map_order():
    """Die Anordnung nach den Plätzen der Partner kreuzt im Mittel nicht mehr als die reine Sortierung nach Kartenlage."""
    def crossings(sc, arr):
        es = [(arr.vehicles[i][1], arr.orders[j][1]) for i in range(sc.n) for j in range(sc.m) if sc.feasible[i, j]]
        return sum(1 for a in range(len(es)) for b in range(a + 1, len(es)) if (es[a][0] - es[b][0]) * (es[a][1] - es[b][1]) < 0)

    better = 0
    for seed in range(100000, 100008):
        sc = build("random", 12, 12, 30, 0, seed)
        vo, oo = _two_sided_order(sc, sweeps=0)
        v0 = {i: 100 - 100 * r / (sc.n - 1) for r, i in enumerate(vo)}
        o0 = {j: 100 - 100 * r / (sc.m - 1) for r, j in enumerate(oo)}
        plain = replace(sc, vehicles=tuple((0, v0[i]) for i in range(sc.n)), orders=tuple((100, o0[j]) for j in range(sc.m)))
        better += crossings(sc, arrange(sc, "bipartit")) <= crossings(sc, plain)
    assert better == 8


def _map_args(a, k):
    """Argumente für build_gs_map nach k Vorschlägen, wie die App sie bildet."""
    sc, res, proposer = a.scenario, a.result, a.proposer
    n_prop = sc.n if proposer == "V" else sc.m
    holder = state_at(res, n_prop, k)["holder"]
    pairs = [(p, r) if proposer == "V" else (r, p) for r, p in holder.items()]
    return sc, pairs, {i for i, _ in pairs}, {j for _, j in pairs}, (res.events[k - 1] if k > 0 else None), proposer


@pytest.mark.parametrize("layout", list(C.LAYOUT_LABELS))
@pytest.mark.parametrize("proposer", list(C.PROPOSER_LABELS))
@pytest.mark.parametrize("params", CASES)
def test_map_builds_in_both_layouts_at_every_kind_of_step(params, proposer, layout):
    a = ev.analyse(build(*params), "noise", 20, params[5], proposer)
    n_events = len(a.result.events)
    for k in sorted({0, 1, n_events // 2, n_events}):
        sc, pairs, mv, mo, event, prop = _map_args(a, k)
        fig = build_gs_map(sc, pairs, mv, mo, event, prop, layout=layout)
        assert len(fig.data) >= 3
        if layout == "bipartit":
            assert {x.text for x in fig.layout.annotations if x.text and "<b>" in x.text} == {"<b>Fahrzeuge</b>", "<b>Aufträge</b>"}
    blocking = blocking_pairs(a.pv, a.po, list(a.hung.pairs))          # das Optimum ist meist instabil: dann gibt es blockierende Paare zu zeichnen
    sc, pairs, mv, mo, event, prop = _map_args(a, n_events)
    assert len(build_gs_map(sc, pairs, mv, mo, event, prop, blocking=blocking, layout=layout).data) >= 3          # auch mit blockierenden Paaren (violett gestrichelt)


def test_two_sided_drawing_has_the_same_markers_as_the_map():
    a = ev.analyse(build("random", 8, 8, 40, 0, 97), "noise", 20, 97, "V")
    sc, pairs, mv, mo, event, prop = _map_args(a, len(a.result.events) // 2)
    rows = {}
    for layout in C.LAYOUT_LABELS:
        fig = build_gs_map(sc, pairs, mv, mo, event, prop, layout=layout)
        rows[layout] = sorted(h for t in fig.data if t.hovertext is not None for h in t.hovertext)
    assert rows["map"] == rows["bipartit"] and rows["map"]


@pytest.mark.parametrize("proposer", list(C.PROPOSER_LABELS))
def test_app_runs_in_the_two_sided_layout(proposer):
    at = AppTest.from_file(str(APP), default_timeout=600)
    at.query_params["layout"] = "bipartit"
    at.query_params["prop"] = proposer
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.radio(key="layout_radio").value == "bipartit"
    at.radio(key="layout_radio").set_value("map").run()
    assert not at.exception and at.radio(key="layout_radio").value == "map"


def test_app_plays_through_in_the_two_sided_layout():
    at = AppTest.from_file(str(APP), default_timeout=600)
    at.query_params["layout"] = "bipartit"
    at.run()
    [b for b in at.button if b.label == "▶️ Abspielen"][0].click()
    at.run()
    assert not at.exception, [e.value for e in at.exception]


def test_unknown_layout_in_the_permalink_falls_back_to_the_map():
    at = AppTest.from_file(str(APP), default_timeout=600)
    at.query_params["layout"] = "kreis"
    at.run()
    assert not at.exception and at.radio(key="layout_radio").value == C.DEFAULT_LAYOUT
