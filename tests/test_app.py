"""Rauchtests der Streamlit-Oberfläche per AppTest: Standard, jedes Preset, alle Kombinationen aus Vorlieben und Vorschlagenden (auch beim Abspielen auf mehrbildrigen Karten),
Randgrößen, Schritt-Zustand, ausgeblendete Regler, Permalink, Experimente auf Abruf, Schlüssel und Achsensperre."""

import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import gs_constants as C
from gs_presets import PRESET_KEYS

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app.py"

# Anfang der Meldung zur gezeigten Karte und (bei Zufallskarten) der Meldung zur Verteilung (Streamlit legt das führende Emoji in `icon`, nicht in `value`)
EXPECTED = {
    "⚖️ Die billigste Kante klaut": ("Stabile Paarung: 2 Paare für 18 Minuten", None),
    "🔀 Vorschlagende zählen": ("Stabile Paarung: 2 Paare für 80 Minuten", None),
    "🛣️ Worst Case: die Straße": ("Stabile Paarung: 8 Paare für 320 Minuten", None),
    "📏 Nur Entfernung": ("Stabile Paarung: 17 Paare für 271 Minuten", "Auf 100 % der 100 Karten"),
    "🗺️ Mittlere Reichweite": ("Stabile Paarung: 17 Paare für 300 Minuten", "Auf 100 % der 100 Karten"),
    "🎲 Zufallsvorlieben": ("Stabile Paarung: 20 Paare für 1014 Minuten", "Auf 100 % der 100 Karten"),
    "📐 Mehr Aufträge als Fahrzeuge": ("Stabile Paarung: 10 Paare für 182 Minuten", "Auf 100 % der 100 Karten"),
    "📡 Knappe Reichweite": ("Stabile Paarung: 7 Paare für 40 Minuten", "Auf 100 % der 100 Karten"),
}
COMBOS = [(p, q) for p in C.PREF_LABELS for q in C.PROPOSER_LABELS]


def _run(setup=None, timeout=600):
    at = AppTest.from_file(str(APP), default_timeout=timeout)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    if setup is not None:
        setup(at)
        at.run()
        assert not at.exception, [e.value for e in at.exception]
    return at


def _apply(at, p):
    for key, state_key in PRESET_KEYS.items():
        at.session_state[state_key] = p[key]


def _combo(pref, prop):
    def setup(at):
        at.session_state["pref_radio"], at.session_state["proposer_radio"] = pref, prop
    return setup


def _labels(at):
    return {w.label for w in list(at.sidebar.slider) + list(at.sidebar.selectbox) + list(at.sidebar.number_input) + list(at.sidebar.radio)}


def _texts(at):
    return [e.value for e in list(at.success) + list(at.warning) + list(at.info)]


def _has(at, prefix):
    return any(t.startswith(prefix) for t in _texts(at))


def _step(at):
    found = [s for s in at.slider if s.key == "gs_step"]
    return found[0] if found else None


def _play(at):
    [b for b in at.button if b.label == "▶️ Abspielen"][0].click()
    at.run()
    assert not at.exception, [e.value for e in at.exception]


def test_default_renders_without_exception():
    at = _run()
    assert any("Vorlieben und Ablauf" in m.value for m in at.markdown)
    assert _has(at, EXPECTED["🗺️ Mittlere Reichweite"][0]) and not at.error


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_renders_with_its_verdicts(name):
    at = _run(lambda a: _apply(a, C.PRESETS[name]))
    top, dist = EXPECTED[name]
    assert _has(at, top), _texts(at)
    if dist is not None:
        assert _has(at, dist), _texts(at)
    else:
        assert any(t.startswith("Feste Karte") for t in _texts(at))


@pytest.mark.parametrize("pref,prop", COMBOS)
def test_every_combination_renders_every_kind_of_step(pref, prop):
    at = _run(_combo(pref, prop))
    assert not at.error and _has(at, "Stabile Paarung:")
    last = int(_step(at).max)
    assert last > 20 and _step(at).value == last
    for k in (0, 1, 2, last // 3, last // 2, last - 1, last):
        _step(at).set_value(k)
        at.run()
        assert not at.exception, (pref, prop, k, [e.value for e in at.exception])


@pytest.mark.parametrize("pref,prop", COMBOS)
def test_play_runs_through_all_frames_without_duplicate_chart_keys(pref, prop):
    """Beim Abspielen entstehen in einem Lauf mehrere Diagramme mit demselben Namen - die Schlüssel tragen deshalb den Schritt (Regression: StreamlitDuplicateElementKey bei mehr als einem Bild)."""
    at = _run(_combo(pref, prop))
    assert _step(at).max > 20
    _play(at)


@pytest.mark.parametrize("net", ["steal", "cross2", "latin4", "road"])
def test_play_on_fixed_maps(net):
    at = _run(lambda a: a.session_state.__setitem__("net_select", net))
    assert _step(at).max >= 2                                                                    # mindestens drei Bilder mit Diagrammen gleichen Namens
    _play(at)


def test_no_feasible_pair_renders_and_play_is_disabled():
    from gs_scenario import generate
    seed = next(k for k in range(500) if not generate(C.N_MIN, C.M_MIN, C.REACH_MIN, 0, k).feasible.any())

    def setup(at):
        at.session_state["n_slider"], at.session_state["m_slider"], at.session_state["reach_slider"], at.session_state["seed_input"] = C.N_MIN, C.M_MIN, C.REACH_MIN, seed
    at = _run(setup)
    assert _step(at) is None and any(b.label == "▶️ Abspielen" and b.disabled for b in at.button)
    assert any(t.startswith("Kein einziges Paar ist möglich") for t in _texts(at))


def test_extreme_sizes_render():
    for n, m, reach in ((C.N_MIN, C.M_MIN, C.REACH_MIN), (C.N_MAX, C.M_MAX, C.REACH_MAX), (C.N_MIN, C.M_MAX, C.REACH_MIN), (C.N_MAX, C.M_MIN, C.REACH_MAX)):
        def setup(at, n=n, m=m, reach=reach):
            at.session_state["n_slider"], at.session_state["m_slider"], at.session_state["reach_slider"] = n, m, reach
        at = _run(setup)
        step = _step(at)
        assert step is None or step.value == step.max


def test_hidden_controls_follow_the_net():
    def labels_for(net):
        return _labels(_run(lambda a: a.session_state.__setitem__("net_select", net)))
    random_labels, fixed = labels_for("random"), labels_for("road")
    assert {"Karte", "Fahrzeuge", "Aufträge", "Reichweite [min]", "Ballung [%]", "Zufalls-Seed"} <= random_labels
    assert fixed == {"Karte"}                                             # keine toten Regler bei festen Karten


def test_the_preference_radio_is_hidden_on_maps_with_their_own_lists():
    def radios(at):
        return [r.key for r in at.radio]
    assert "pref_radio" in radios(_run()) and "pref_radio" not in radios(_run(lambda a: a.session_state.__setitem__("net_select", "cross2")))
    at = _run(lambda a: a.session_state.__setitem__("net_select", "latin4"))
    assert "pref_radio" not in radios(at) and "noise_slider" not in [s.key for s in at.slider] and any("eigene, feste Vorlieben" in c.value for c in at.caption)
    assert _has(at, "Stabile Paarung:") and any(m.label == "Stabile Paarungen" and m.value == "10" for m in at.metric)


def test_the_noise_slider_only_exists_for_noise_and_keeps_its_value():
    def sliders(at):
        return [s.key for s in at.slider]
    assert "noise_slider" in sliders(_run()) and "noise_slider" not in sliders(_run(lambda a: a.session_state.__setitem__("pref_radio", "dist")))
    at = _run()
    at.slider(key="noise_slider").set_value(35)
    at.run()
    at.session_state["pref_radio"] = "random"
    at.run()
    assert "noise_slider" not in sliders(at)
    at.session_state["pref_radio"] = "noise"
    at.run()
    assert not at.exception and at.session_state["noise_slider"] == 35                            # der zuletzt gewählte Wert kommt zurück


def test_hidden_slider_values_come_back_when_the_random_map_is_shown_again():
    at = _run(lambda a: a.session_state.__setitem__("reach_slider", 90))
    at.session_state["net_select"] = "steal"
    at.run()
    at.session_state["net_select"] = "random"
    at.run()
    assert not at.exception and at.slider(key="reach_slider").value == 90


def test_step_slider_returns_to_the_last_step_when_the_map_or_a_mode_changes():
    at = _run()
    last = int(_step(at).max)
    assert _step(at).value == last
    _step(at).set_value(3)
    at.run()
    assert _step(at).value == 3
    at.session_state["net_select"] = "steal"
    at.run()
    assert not at.exception and _step(at).value == _step(at).max and _step(at).max < last
    _step(at).set_value(2)
    at.run()
    at.session_state["proposer_radio"] = "O"
    at.run()
    assert not at.exception and _step(at).value == _step(at).max


def test_first_step_shows_the_start_a_middle_step_the_lists_and_the_last_the_certificate():
    at = _run()
    _step(at).set_value(0)
    at.run()
    assert not at.exception and any("Anfang: niemand hat einen Partner" in m.value for m in at.markdown)
    assert not any("S2 Kein blockierendes Paar" in t.value.to_dict("list").get("Bedingung", []) for t in at.table)            # der Beweis erst im letzten Schritt
    _step(at).set_value(5)
    at.run()
    assert not at.exception and any("schlägt" in m.value and "vor (Wahl Nr." in m.value for m in at.markdown)
    assert any("Wahl" in t.value.columns for t in at.table)
    _step(at).set_value(int(_step(at).max))
    at.run()
    cert = next(t.value.to_dict("list") for t in at.table if "S2 Kein blockierendes Paar" in t.value.to_dict("list").get("Bedingung", []))
    assert all(row.startswith("✅") for row in cert["Prüfung"]) and len(cert["Bedingung"]) == 6


def test_blocking_pair_check_flags_the_optimum_and_clears_the_result():
    at = _run()
    assert any(t.startswith("Kein blockierendes Paar") for t in _texts(at))
    at.selectbox(key="block_select").set_value("Optimum (Ungarische Methode)")
    at.run()
    assert not at.exception and any("blockierende Paare in dieser Paarung" in w.value for w in at.warning)
    at.selectbox(key="block_select").set_value("Stabile Paarung Nr. 2")
    at.run()
    assert not at.exception and any(t.startswith("Kein blockierendes Paar") for t in _texts(at))


def test_permalink_parameters_are_clamped_and_snapped():
    at = AppTest.from_file(str(APP), default_timeout=600)
    at.query_params["reach"] = "9999"
    at.query_params["ballung"] = "abc"
    at.query_params["n"] = "-5"
    at.query_params["noise"] = "22"
    at.run()
    assert not at.exception
    assert at.slider(key="reach_slider").value == C.REACH_MAX and at.slider(key="ballung_slider").value == C.DEFAULT_BALLUNG and at.slider(key="n_slider").value == C.N_MIN
    assert at.slider(key="noise_slider").value == 20                                                # auf die Regler-Schritte gerundet
    at = AppTest.from_file(str(APP), default_timeout=600)
    at.query_params["reach"] = "42"
    at.query_params["ballung"] = "60"
    at.run()
    assert at.slider(key="reach_slider").value == 40 and at.slider(key="ballung_slider").value == 50


def test_permalink_keeps_valid_modes_and_falls_back_for_unknown_ones():
    at = AppTest.from_file(str(APP), default_timeout=600)
    at.query_params["pref"] = "random"
    at.query_params["prop"] = "O"
    at.run()
    assert not at.exception and at.radio(key="pref_radio").value == "random" and at.radio(key="proposer_radio").value == "O"
    at = AppTest.from_file(str(APP), default_timeout=600)
    at.query_params["net"] = "ring"
    at.query_params["pref"] = "chaos"
    at.query_params["prop"] = "X"
    at.run()
    assert not at.exception and at.selectbox(key="net_select").value == C.DEFAULT_NET
    assert at.radio(key="pref_radio").value == C.DEFAULT_PREF and at.radio(key="proposer_radio").value == C.DEFAULT_PROPOSER


def test_randomize_moves_the_seed_but_not_the_distribution():
    at = _run()
    before = {m.label: m.value for m in at.metric}
    seed_before = at.number_input(key="seed_input").value
    [b for b in at.sidebar.button if "Neue Karte" in b.label][0].click()
    at.run()
    assert not at.exception and at.number_input(key="seed_input").value != seed_before
    after = {m.label: m.value for m in at.metric}
    for label in ("Mehrere stabile Paarungen", "Prämie (Median | Mittel)", "Optimum ist instabil", "Vorschläge (Mittel | Median)"):
        assert before[label] == after[label], label


def test_experiments_run_on_demand():
    at = _run()
    markers = ("Die Streuung ist über k gekoppelt", "Mit der Größe wächst die Zahl", "Die Straße (alle einer Meinung)", "5 × 5 Fahrzeuge und Aufträge")
    assert not any(any(m in c.value for m in markers) for c in at.caption)
    for key in ("noise_start", "count_start", "effort_start", "manip_start"):
        at.button(key=key).click()
        at.run()
        assert not at.exception, [e.value for e in at.exception]
    text = " ".join(c.value for c in at.caption)
    assert all(m in text for m in markers)


def test_experiments_with_pure_distance_and_on_a_fixed_map():
    at = _run(lambda a: (a.session_state.__setitem__("net_select", "steal"), a.session_state.__setitem__("pref_radio", "dist")))
    for key in ("noise_start", "count_start", "effort_start", "manip_start"):
        at.button(key=key).click()
        at.run()
        assert not at.exception, [e.value for e in at.exception]


def _calls(src, name):
    """Der Text jedes Aufrufs `name(...)` einschließlich verschachtelter Klammern."""
    out = []
    for m in re.finditer(re.escape(name) + r"\(", src):
        depth, i = 1, m.end()
        while depth:
            depth += {"(": 1, ")": -1}.get(src[i], 0)
            i += 1
        out.append(src[m.start():i])
    return out


def test_every_plotly_chart_has_an_explicit_unique_key_and_axes_are_locked():
    calls = _calls(APP.read_text(encoding="utf-8"), "plotly_chart")
    assert len(calls) == 7 and all(re.search(r'key=f?"[a-z_]+(_\{\w+\})?"', c) for c in calls), calls
    keys = [re.search(r'key=f?"([a-z_]+?)(?:_\{\w+\})?"', c).group(1) for c in calls]
    assert len(set(keys)) == 7, keys                                                              # jeder Schlüssel nur einmal
    assert sum(1 for c in calls if 'key=f"' in c) == 2                                            # nur die beiden Diagramme der Abspiel-Schleife tragen den Schritt
    viz = (ROOT / "gs_visualization.py").read_text(encoding="utf-8")
    bodies = [b for b in viz.split(chr(10) + "def ") if b.startswith("build_")]
    assert "fixedrange=True" in viz and len(bodies) == 7 and all("_base(" in b or "_map_layout(" in b for b in bodies)


def test_app_text_has_no_links_to_repository_files():
    assert not re.search(r"\]\(\w+\.py\)", APP.read_text(encoding="utf-8"))


def test_footer_is_verbatim():
    src = APP.read_text(encoding="utf-8")
    assert "https://sebastianhanisch.net/kontakt.html" in src and "Interesse an einer maßgeschneiderten Lösung für" in src and "Operations Research und Machine Learning" in src


def test_runtime_needs_only_numpy_pandas_plotly_streamlit():
    """Konvention der Konzepte-Wurzeln und -Stücke: Referenzbibliotheken (scipy, networkx) nur als Testorakel."""
    req = (ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
    assert "scipy" not in req and "networkx" not in req
    for path in ROOT.glob("*.py"):
        assert not re.search(r"^\s*(import|from)\s+(scipy|networkx)\b", path.read_text(encoding="utf-8"), re.M), path.name
