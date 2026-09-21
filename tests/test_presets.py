"""Presets: vollständig, in den Grenzen, und jede Beispielkarte zeigt, was ihr Hilfetext behauptet (typische Ziehung, Median der 100 festen Karten)."""

import pytest

import gs_constants as C
import gs_evaluation as ev
import gs_presets as P
from gs_scenario import build

KEYS = set(P.PRESET_KEYS)


def _a(p):
    sc = build(p["net"], p["n"], p["m"], p["reach"], p["ballung"], p["seed"])
    return ev.analyse(sc, p["pref"], p["noise"] if p["pref"] == "noise" else 0, p["seed"], p["prop"])


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP) and len(C.PRESETS) == 8
    assert all(C.PRESET_HELP[name].strip() for name in C.PRESETS)
    for name, p in C.PRESETS.items():
        assert set(p) == KEYS, name


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_preset_values_are_inside_the_bounds_and_on_the_step_grid(name):
    p = C.PRESETS[name]
    assert p["net"] in C.NETS and p["pref"] in C.PREF_LABELS and p["prop"] in C.PROPOSER_LABELS
    for key, state_key in P.PRESET_KEYS.items():
        spec = P.SETTING_SPECS[state_key]
        if spec.lo is not None:
            assert spec.lo <= p[key] <= spec.hi, (name, key)
    assert (p["reach"] - C.REACH_MIN) % 5 == 0 and p["ballung"] % 25 == 0 and (p["noise"] - C.NOISE_MIN) % C.NOISE_STEP == 0


def test_setting_specs_have_room_to_move():
    """Ein Regler mit lo == hi würde Streamlit abstürzen lassen."""
    assert all(spec.lo < spec.hi for spec in P.SETTING_SPECS.values() if spec.lo is not None)


def test_presets_use_seeds_outside_the_distribution_set():
    for name, p in C.PRESETS.items():
        assert p["seed"] not in C.DIST_SEEDS, name


def test_the_default_map_is_the_mid_reach_preset_and_shows_several_stable_matchings():
    mid = C.PRESETS["🗺️ Mittlere Reichweite"]
    assert (mid["n"], mid["m"], mid["reach"], mid["ballung"], mid["seed"], mid["pref"], mid["noise"], mid["prop"]) == (20, 20, 40, 0, C.DEFAULT_SEED, C.DEFAULT_PREF, C.DEFAULT_NOISE, C.DEFAULT_PROPOSER)
    _, _, d = ev.verdict(_a(mid))
    assert d["n_stable"] == 4 and not d["same_vo"]


def test_every_preset_is_stable_and_certified():
    for name, p in C.PRESETS.items():
        level, code, d = ev.verdict(_a(p))
        assert (level, code) == ("success", ev.STABLE) and d["cert"]["all_ok"], name


def test_the_presets_show_what_their_help_says():
    v = {name: ev.verdict(_a(p))[2] for name, p in C.PRESETS.items()}
    steal = v["⚖️ Die billigste Kante klaut"]
    assert (steal["cost"], steal["opt_cost"], steal["opt_blocking"]) == (18, 10, 1)
    cross = v["🔀 Vorschlagende zählen"]
    assert cross["n_stable"] == 2 and not cross["same_vo"]
    road = v["🛣️ Worst Case: die Straße"]
    assert road["proposals"] == 36 == road["other_proposals"]
    dist = v["📏 Nur Entfernung"]
    assert dist["same_as_greedy"] and dist["n_stable"] == 1 and abs(dist["premium_pct"] - 3.4) < 0.1
    rand = v["🎲 Zufallsvorlieben"]
    assert rand["n_stable"] == 6 and rand["count"] == 20
    wide = v["📐 Mehr Aufträge als Fahrzeuge"]
    assert (wide["count"], wide["opt_count"]) == (10, 10) and wide["proposals"] < 20 and wide["other_proposals"] > 100
    short = v["📡 Knappe Reichweite"]
    assert short["opt_stable"] and short["premium_pct"] in (None, 0.0)


def test_random_presets_are_typical_draws():
    """Die gezeigte Karte liegt nahe dem Median der 100 festen Karten derselben Einstellung (Prämie, Vorschläge)."""
    for name in ("🗺️ Mittlere Reichweite", "📏 Nur Entfernung", "🎲 Zufallsvorlieben", "📐 Mehr Aufträge als Fahrzeuge"):
        p = C.PRESETS[name]
        d = ev.verdict(_a(p))[2]
        dist = ev.distribution(p["n"], p["m"], p["reach"], p["ballung"], p["pref"], p["noise"] if p["pref"] == "noise" else 0)
        assert abs(d["premium_pct"] - dist["premium_median"]) <= 0.35 * dist["premium_median"], name
        assert abs(d["proposals"] - dist["proposals_v_median"]) <= 0.35 * dist["proposals_v_median"] + 2, name
    d = ev.verdict(_a(C.PRESETS["📐 Mehr Aufträge als Fahrzeuge"]))[2]
    assert abs(d["other_proposals"] - ev.distribution(10, 20, 150, 0, "noise", 20)["proposals_o_median"]) <= 4


def test_fixed_presets_hide_the_random_controls():
    assert {n for n, p in C.PRESETS.items() if p["net"] in C.FIXED_NETS} == {"⚖️ Die billigste Kante klaut", "🔀 Vorschlagende zählen", "🛣️ Worst Case: die Straße"}
