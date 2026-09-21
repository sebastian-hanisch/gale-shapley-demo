"""Gale–Shapley - Vorlieben statt Kosten, Stabilität statt Optimum - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo EIN Verfahren - Gale–Shapley - und lässt stattdessen das Beispiel wachsen.
Sechstes Stück der Matching-Linie der "Konzepte"-Reihe, unabhängiger Ast neben der Kostenlinie: der Kontrast zur Ungarischen Methode. Siehe README.

Lauffähig mit: streamlit run app.py
"""

import time

import numpy as np
import streamlit as st

import gs_constants as C
import gs_evaluation as ev
from gs_algorithm import blocking_pairs, state_at
from gs_lattice import rank_sums
from gs_preferences import ranks
from gs_presets import (
    KEPT,
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
)
from gs_scenario import build
from gs_visualization import build_count_vs_size, build_effort, build_gs_map, build_noise_sweep, build_premium_hist, build_progress, build_scatter

st.set_page_config(page_title="Gale–Shapley – Sebastian Hanisch", layout="wide")


def _pct(x, digits=0):
    return "–" if x is None else f"{x:.{digits}f} %".replace(".", ",")


def _share(x):
    """Anteil (0..1) als 'nn %'."""
    return f"{100 * x:.0f} %"


def _f(x, digits=1):
    return "–" if x is None else f"{x:.{digits}f}".replace(".", ",")


def _int(x):
    """Ganzzahl mit Leerzeichen als Tausendertrenner."""
    return f"{x:,.0f}".replace(",", " ")


@st.cache_resource(show_spinner=False, max_entries=16)
def _analysis(params):
    net, n, m, reach, ballung, seed, pref, noise, proposer = params
    return ev.analyse(build(net, n, m, reach, ballung, seed), pref, noise, seed, proposer)


@st.cache_data(show_spinner=False)
def _distribution(n, m, reach, ballung, pref, noise):
    return ev.distribution(n, m, reach, ballung, pref, noise)


@st.cache_data(show_spinner=False)
def _noise_sweep(n, m, reach, ballung):
    return ev.noise_sweep(n, m, reach, ballung)


@st.cache_data(show_spinner=False)
def _count_vs_size(pref, noise):
    return ev.count_vs_size(pref, noise)


@st.cache_data(show_spinner=False)
def _effort(pref, noise):
    return ev.effort_scaling(pref, noise)


@st.cache_data(show_spinner=False)
def _manipulation(pref, noise):
    return ev.manipulation(pref, noise)


@st.cache_data(show_spinner=False, max_entries=8)
def _series(params):
    a = _analysis(params)
    matched, rejected, bad = [], [], 0
    holder = {}
    matched.append(0)
    rejected.append(0)
    for p, r, outcome, cur in a.result.events:
        if outcome == "accept":
            holder[r] = p
        elif outcome == "displace":
            holder[r] = p
            bad += 1
        else:
            bad += 1
        matched.append(len(holder))
        rejected.append(bad)
    return matched, rejected


st.title("💍 Gale–Shapley – Vorlieben statt Kosten, Stabilität statt Optimum")
st.markdown(
    """
Die Ungarische Methode findet die **billigste** Zuordnung. Aber was, wenn beide Seiten **Vorlieben** haben - jedes Fahrzeug eine Rangliste der Aufträge, jeder Auftrag eine der Fahrzeuge? Dann zählt nicht das Optimum, sondern **Stabilität**: es darf kein **blockierendes Paar** geben,
also kein Fahrzeug und kein Auftrag, die einander lieber hätten als ihre jetzigen Partner. **Gale–Shapley** (aufgeschobene Annahme) findet immer eine stabile Paarung: eine Seite schlägt vor, die andere hält immer den besten Vorschlag und lehnt den Rest ab.
Der Preis der Stabilität ist Geld: die stabile Paarung ist meist teurer als das Optimum und hat oft weniger Paare - die Demo misst, wie viel. Und **wer vorschlägt, entscheidet**: der Vorschlagende bekommt seine beste, der Empfänger seine schlechteste stabile Paarung.
"""
)
st.caption(
    "Anders als die Fall-Demos im Portfolio, die an einem Anwendungsfall mehrere Verfahren vergleichen, zeigt diese Demo - sechstes Stück der Matching-Linie der \"Konzepte\"-Reihe, unabhängiger Ast neben der Kostenlinie - **ein** Verfahren an einem wachsenden Beispiel. "
    "Eine Besonderheit der Karte: ordnen **beide** Seiten allein nach der Fahrzeit, gibt es genau **eine** stabile Paarung - die Greedy-Paarung der Wurzel-Demo. Damit Vorlieben und Vorschlagende etwas ausmachen, schätzt in der Standardeinstellung jede Seite die Fahrzeit selbst, mit Streuung; ein Umschalter kennt auch Zufallsvorlieben. "
    "Die Schwächen dieses Stücks sind die Ansatzpunkte der nächsten: **Stabile Mitbewohner** (keine zwei Seiten, eine stabile Paarung existiert nicht immer) und **Online-Matching** - noch nicht gebaut. "
    "Die Referenz \"Optimum\" ist die Ungarische Methode aus der Demo dazu."
)

with st.expander("So funktioniert Gale–Shapley", expanded=True):
    st.markdown(
        """
1. **Vorlieben:** jede Seite ordnet die Partner, die in Frage kommen (Reichweite): Fahrzeuge ordnen Aufträge, Aufträge ordnen Fahrzeuge. Die Listen sind strikt und können unterschiedlich lang sein.
2. **Vorschlagen:** ein Vorschlagender ohne Partner fragt den nächsten Auftrag (bzw. das nächste Fahrzeug) auf seiner Liste, von oben nach unten.
3. **Halten:** der Empfänger hält den Vorschlag vorläufig, wenn er noch frei ist oder den Vorschlagenden seinem jetzigen Halter **vorzieht** (der wird verdrängt und fragt weiter unten). Sonst lehnt er ab.
4. **Ende:** wenn niemand mehr vorschlagen kann. Das Ergebnis ist **stabil** (kein blockierendes Paar) - und unabhängig von der Reihenfolge, in der die Vorschlagenden dran sind.
5. **Wer vorschlägt, gewinnt:** jeder Vorschlagende bekommt seinen besten Partner, den er in *irgendeiner* stabilen Paarung haben kann; jeder Empfänger seinen schlechtesten. Die Paarzahl ist in allen stabilen Paarungen gleich (Landklinikensatz).
        """
    )

st.caption("🎯 Schnellstart – eine Beispielkarte laden:")
names = list(C.PRESETS.keys())
for row in range(0, len(names), 4):
    preset_cols = st.columns(4)
    for col, name in zip(preset_cols, names[row:row + 4]):
        with col:
            st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP[name])

st.caption(
    "🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, "
    "um ein Szenario zu teilen."
)

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    net_key = st.selectbox(
        "Karte", list(C.NETS), key="net_select", format_func=lambda k: C.NETS[k],
        help="Eine zufällige Karte mit Fahrzeugen und Aufträgen oder eine feste Lehrbuchkarte. Zwei davon haben eigene Vorlieben: gegenläufige (zwei stabile Paarungen) und Knuths 4×4 (zehn stabile Paarungen).",
    )
    if net_key == "random":
        n = st.slider("Fahrzeuge", *bounds("n_slider"), key="n_slider", help="Anzahl der Fahrzeuge.")
        st.session_state[KEPT["n_slider"]] = n
        m = st.slider("Aufträge", *bounds("m_slider"), key="m_slider", help="Anzahl der Aufträge. Bei ungleich vielen Seiten bleiben Agenten der größeren Seite ohne Partner; die Paarzahl ist trotzdem in jeder stabilen Paarung gleich.")
        st.session_state[KEPT["m_slider"]] = m
        reach = st.slider(
            "Reichweite [min]", *bounds("reach_slider"), key="reach_slider", step=5,
            help="Wie weit ein Paar höchstens auseinander liegen darf (macht die Vorliebenlisten kürzer). Bei nur Entfernung als Vorlieben ist das Optimum bei Reichweite 10 auf 79 von 100 Karten selbst stabil, bei 40 und bei 150 auf keiner.",
        )
        st.session_state[KEPT["reach_slider"]] = reach
        ballung = st.slider("Ballung [%]", *bounds("ballung_slider"), key="ballung_slider", step=25, help="0 = Fahrzeuge und Aufträge gleichmäßig verteilt, 100 = alle um drei Stadtteile gruppiert.")
        st.session_state[KEPT["ballung_slider"]] = ballung
        seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
        st.session_state[KEPT["seed_input"]] = seed
        st.button("🎲 Neue Karte generieren", width="stretch", on_click=randomize_seed, help="Würfelt einen neuen Zufalls-Seed. Die Verteilung über 100 feste Karten weiter unten ändert sich dabei nicht.")
    else:
        n = int(st.session_state.get(KEPT["n_slider"], C.DEFAULT_N))
        m = int(st.session_state.get(KEPT["m_slider"], C.DEFAULT_M))
        reach = int(st.session_state.get(KEPT["reach_slider"], C.DEFAULT_REACH))
        ballung = int(st.session_state.get(KEPT["ballung_slider"], C.DEFAULT_BALLUNG))
        seed = int(st.session_state.get(KEPT["seed_input"], C.DEFAULT_SEED))
        st.caption("Diese Karte ist fest - es gibt nichts zu erzeugen. Zahl der Fahrzeuge und Aufträge, Reichweite, Ballung und Seed gehören zur zufälligen Karte.")

# --- Vorlieben und Ablauf ------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Vorlieben und Ablauf")
step_col, pref_col, prop_col, play_col = st.columns([4, 4, 3, 2])
with pref_col:
    if net_key in C.LIST_NETS:
        pref, noise = "dist", 0
        st.caption("Diese Karte hat eigene, feste Vorlieben - der Umschalter gilt hier nicht.")
    else:
        pref = st.radio("Vorlieben", list(C.PREF_LABELS), key="pref_radio", format_func=lambda k: C.PREF_LABELS[k],
                        help="Nur Entfernung: beide Seiten ordnen nach der Fahrzeit - dieselbe Wertung, genau eine stabile Paarung (= Greedy). Mit Streuung: jede Seite schätzt die Fahrzeit selbst (±k Minuten). Zufall: unabhängig von den Kosten.")
        if pref == "noise":
            if not st.session_state.get("_noise_shown") and KEPT["noise_slider"] in st.session_state:
                st.session_state["noise_slider"] = st.session_state[KEPT["noise_slider"]]          # der zuletzt gewählte Wert kommt zurück, wenn der Regler wieder erscheint
            noise = st.slider("Streuung ± [min]", *bounds("noise_slider"), key="noise_slider", step=C.NOISE_STEP,
                              help="Wie stark die Schätzungen der beiden Seiten auseinanderliegen können. 0 wäre reine Entfernung (eindeutig, = Greedy); je größer, desto mehr stabile Paarungen und desto teurer die Stabilität.")
            st.session_state[KEPT["noise_slider"]] = noise
            st.session_state["_noise_shown"] = True
        else:
            st.session_state["_noise_shown"] = False
            noise = 0
with prop_col:
    proposer = st.radio("Vorschlagende", list(C.PROPOSER_LABELS), key="proposer_radio", format_func=lambda k: C.PROPOSER_LABELS[k],
                        help="Wer vorschlägt, bekommt seine beste stabile Paarung; die andere Seite die schlechteste. Bei nur einer stabilen Paarung ist das gleichgültig.")

params = (net_key, int(n), int(m), int(reach), int(ballung), int(seed))
if net_key in C.FIXED_NETS:
    params = (net_key, C.DEFAULT_N, C.DEFAULT_M, C.DEFAULT_REACH, C.DEFAULT_BALLUNG, C.DEFAULT_SEED)
seed_eff = params[5]
params = params + (pref, int(noise) if pref == "noise" else 0, proposer)
with st.spinner("Rechne..."):
    a = _analysis(params)
sc, res = a.scenario, a.result
level, code, d = ev.verdict(a)
n_events = len(res.events)
n_prop = sc.n if proposer == "V" else sc.m
if st.session_state.get("gs_step_owner") != params:
    st.session_state["gs_step"] = n_events
    st.session_state["gs_step_owner"] = params
with step_col:
    if n_events > 0:
        step = st.slider("Vorschlag", 0, n_events, key="gs_step", help="Wie viele Vorschläge schon gemacht sind. Ganz rechts die fertige stabile Paarung mit ihrem Beweis.")
    else:
        step = 0
        st.caption("Hier gibt es keinen Vorschlag: kein Paar ist möglich.")
with play_col:
    auto_play = st.button("▶️ Abspielen", width="stretch", disabled=n_events == 0)
sync_query_params({"net_select": net_key, "pref_radio": pref if net_key not in C.LIST_NETS else st.session_state.get("pref_radio", C.DEFAULT_PREF), "noise_slider": int(noise) if pref == "noise" else int(st.session_state.get(KEPT["noise_slider"], C.DEFAULT_NOISE)),
                   "proposer_radio": proposer, "n_slider": int(n), "m_slider": int(m), "reach_slider": int(reach), "ballung_slider": int(ballung), "seed_input": int(seed)})
view_slot = st.empty()
matched_series, rejected_series = _series(params)
L_P = "F" if proposer == "V" else "A"
L_R = "A" if proposer == "V" else "F"
pv, po = a.pv, a.po
prop_lists = pv if proposer == "V" else po
recv_lists = po if proposer == "V" else pv
rk_p, rk_r = ranks(prop_lists), ranks(recv_lists)


def _cost(p, r):
    i, j = (p, r) if proposer == "V" else (r, p)
    return int(sc.cost[i, j])


def _event_text(k):
    if k == 0:
        return "Anfang: niemand hat einen Partner, alle Vorschlagenden beginnen mit ihrer ersten Wahl."
    p, r, outcome, cur = res.events[k - 1]
    head = f"{L_P}{p + 1} schlägt {L_R}{r + 1} vor (Wahl Nr. {rk_p[p][r] + 1}, Fahrzeit {_cost(p, r)} min): "
    if outcome == "accept":
        return head + f"{L_R}{r + 1} ist frei und hält {L_P}{p + 1} vorläufig."
    if outcome == "displace":
        return head + f"{L_R}{r + 1} hält {L_P}{cur + 1}, zieht aber {L_P}{p + 1} vor (Rang {rk_r[r][p] + 1} gegen {rk_r[r][cur] + 1}): {L_P}{cur + 1} wird verdrängt und schlägt als Nächstes vor."
    return head + f"{L_R}{r + 1} zieht ihren Halter vor: {L_P}{p + 1} wird abgelehnt und geht in seiner Liste weiter."


def _cert_table():
    lat = a.lattice
    cert = d["cert"]
    ok = lambda b: "✅" if b else "❌"
    rows = [("S1 Gültige Paarung", f"{ok(cert['s1'])} {res.count} Paare, jede Seite höchstens einmal, nur mögliche Paare"),
            ("S2 Kein blockierendes Paar", f"{ok(cert['s2'])} {len(cert['blocking'])} blockierende Paare geprüft über {d['edges']} mögliche Paare"),
            ("S3 Maximal", f"{ok(cert['s3'])} kein mögliches Paar mit zwei freien Agenten (folgt aus S2)"),
            ("S4 Vorschlagszahl", f"{ok(cert['s4'])} {res.proposals} Vorschläge = Summe über die Vorschlagenden von (Rang des Partners + 1) bzw. Listenlänge")]
    if "s5" in cert:
        rows += [("S5 Bester stabiler Partner", f"{ok(cert['s5'])} jeder Vorschlagende hat den besten Partner, den er in einer der {lat['count']} stabilen Paarungen bekommt"),
                 ("S6 Landklinikensatz", f"{ok(cert['s6'])} in allen {lat['count']} stabilen Paarungen sind dieselben Fahrzeuge und dieselben Aufträge versorgt")]
    else:
        rows.append(("S5/S6 Verband", f"⚠️ mehr als {a.lattice['count'] - 1} stabile Paarungen: der Verband wurde abgebrochen, S5 und S6 nicht geprüft"))
    return {"Bedingung": [r[0] for r in rows], "Prüfung": [r[1] for r in rows]}


def _render(k):
    """Zustand nach den ersten k Vorschlägen: links die Karte, rechts die Vorlieben der beiden Beteiligten und der Verlauf; am Ende der Beweis."""
    with view_slot.container():
        stt = state_at(res, n_prop, k)
        holder = stt["holder"]
        pairs_now = [(p, r) if proposer == "V" else (r, p) for r, p in holder.items()]
        mv = {i for i, _ in pairs_now}
        mo = {j for _, j in pairs_now}
        ev_k = res.events[k - 1] if k > 0 else None
        c1, c2 = st.columns([3, 2])
        c1.markdown(f"**Nach {k} von {n_events} Vorschlägen** - " + _event_text(k))
        c1.plotly_chart(build_gs_map(sc, pairs_now, mv, mo, ev_k, proposer), width="stretch", key=f"gs_map_{k}")
        c2.markdown("**Verlauf**")
        c2.plotly_chart(build_progress(matched_series, rejected_series, k), width="stretch", key=f"gs_progress_{k}")
        if ev_k is not None:
            p, r, outcome, cur = ev_k
            seen = {x for x in (e[0] for e in res.events[:k] if e[1] == r)}
            c2.markdown(f"**Liste von {L_P}{p + 1}** (Vorschlagender) und **von {L_R}{r + 1}** (Empfänger)")
            plist = prop_lists[p][:8]
            c2.table({"Wahl": [rk_p[p][x] + 1 for x in plist], f"{L_R}": [f"{L_R}{x + 1}" for x in plist], "Fahrzeit": [_cost(p, x) for x in plist],
                      "Stand": ["gefragt" if rk_p[p][x] <= rk_p[p][r] else "offen" for x in plist]})
            rlist = recv_lists[r][:8]
            c2.table({"Wahl": [rk_r[r][x] + 1 for x in rlist], f"{L_P}": [f"{L_P}{x + 1}" for x in rlist], "Fahrzeit": [_cost(x, r) for x in rlist],
                      "Stand": ["hält jetzt" if holder.get(r) == x else ("hat gefragt" if x in seen else "noch nicht gefragt") for x in rlist]})
        if k == n_events:
            st.markdown("**Beweis:** die Paarung ist stabil")
            st.table(_cert_table())
            st.caption("S2 ist die Definition: für jedes mögliche Paar, das nicht gewählt ist, zieht mindestens eine der beiden Seiten ihren jetzigen Partner vor. Unabhängig vom Verfahren geprüft (Doppelschleife über alle möglichen Paare).")


if auto_play:
    frames = sorted({int(round(x)) for x in np.linspace(0, n_events, min(n_events, 40) + 1)})
    for k in frames:
        _render(k)
        time.sleep(min(0.6, 6.0 / max(len(frames), 1)))
    step = n_events
else:
    _render(step)

st.caption("Quadrate sind Fahrzeuge, Kreise Aufträge (ausgefüllt: haben einen vorläufigen Partner, hohl: frei); blaue Linien halten Paare vorläufig. Das letzte Ereignis ist hervorgehoben: grüne Linie = Vorschlag angenommen, orange = ein Halter wird verdrängt, rot gestrichelt = abgelehnt; "
           "grüner Ring = der Vorschlagende, roter Ring = der Verdrängte. Im Verlauf: grün = wie viele Vorschlagende gerade versorgt sind, rot = bisherige Ablehnungen und Verdrängungen.")

st.markdown("---")

# --- Stabil - und was kostet das? -----------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Stabil – und was kostet das?")
st.caption("Kosten sind über verschiedene Paarzahlen nicht vergleichbar: eine stabile Paarung hat oft weniger Paare als das Optimum und ist damit schon deshalb billiger. Der **Preis der Stabilität** wird deshalb als **Prämie gegenüber der billigsten Paarung mit derselben Paarzahl** ausgewiesen, die verlorenen Paare getrennt. "
           "**Blockierende Paare** zählen zwei Seiten, die einander lieber hätten als ihre jetzigen Partner - bei einer stabilen Paarung gibt es keine.")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Paare (stabil)", f"{d['count']} von {d['opt_count']}", delta=f"{d['lost']} weniger als das Optimum" if d["lost"] else "so viele wie das Optimum", delta_color="off",
          help="Jede stabile Paarung ist maximal, hat also mindestens halb so viele Paare wie das Optimum; alle stabilen Paarungen haben gleich viele.")
m2.metric("Kosten", f"{d['cost']} min", delta=(f"{_pct(d['premium_pct'], 1)} über der billigsten Paarung mit {d['count']} Paaren" if d["premium_pct"] is not None else "so billig wie möglich"), delta_color="off",
          help=f"Optimum (Ungarische Methode): {d['opt_cost']} min mit {d['opt_count']} Paaren, dieses Optimum hat {d['opt_blocking']} blockierende Paare.")
m3.metric("Vorschläge", _int(d["proposals"]), delta=f"andere Seite: {_int(d['other_proposals'])}", delta_color="off",
          help="Aufwand in Vorschlägen (Kopfzahl, nie Sekunden). Bei ungleich vielen Seiten braucht die größere Seite mehr.")
m4.metric("Stabile Paarungen", (f"mindestens {d['n_stable'] - 1}" if d["capped"] else str(d["n_stable"])), delta=("Vorschlagende zählen: die Extreme sind verschieden" if not d["same_vo"] else "nur eine (oder gleiche Extreme)"), delta_color="off",
          help="Aufgezählt über die Rotationen ab der fahrzeugoptimalen Paarung (bis 2000). Bei einer einzigen stabilen Paarung ist es gleichgültig, wer vorschlägt.")

if code == "none":
    st.info("ℹ️ Kein einziges Paar ist möglich – die Reichweite ist zu klein. Es gibt nichts zuzuordnen.")
elif code == "unstable":
    st.error(f"Die Paarung ist nicht stabil: {d['blocking']} blockierende Paare. Das dürfte nicht vorkommen.")
else:
    same = " Sie ist genau die Greedy-Paarung „Billigste Kante zuerst“ der Wurzel-Demo." if d["same_as_greedy"] and pref == "dist" and net_key not in C.LIST_NETS else ""
    opt = (f"Das Optimum ({d['opt_count']} Paare, {d['opt_cost']} min) hat {d['opt_blocking']} blockierende Paare" if d["opt_blocking"] else f"Das Optimum ({d['opt_count']} Paare, {d['opt_cost']} min) ist selbst stabil")
    prem = f", {_pct(d['premium_pct'], 1)} teurer als die billigste Paarung mit {d['count']} Paaren" if d["premium_pct"] else ""
    st.success(f"✅ Stabile Paarung: {d['count']} Paare für {d['cost']} Minuten - kein blockierendes Paar{prem}.{same} {opt}.")

st.markdown("**Die Verfahren im Vergleich auf dieser Karte**")
cmp_rows = ev.compare_table(a)
st.table({"Verfahren": [r["label"] for r in cmp_rows], "Paare": [r["count"] for r in cmp_rows], "Kosten [min]": [r["cost"] for r in cmp_rows],
          "Prämie": [_pct(r["premium_pct"], 1) for r in cmp_rows], "blockierende Paare": [r["blocking"] for r in cmp_rows], "Vorschläge": [str(r["proposals"]) if r["proposals"] is not None else "–" for r in cmp_rows]})

st.markdown("**Blockierende Paare prüfen**")
named = ev.named_matchings(a)
if st.session_state.get("block_select") not in named:
    st.session_state["block_select"] = "Ergebnis (gewählte Vorschlagende)"
choice = st.selectbox("Paarung", list(named), key="block_select", help="Jede der Paarungen wird mit einer unabhängigen Doppelschleife über alle möglichen Paare auf blockierende Paare geprüft (unter den oben gewählten Vorlieben).")
bl = blocking_pairs(pv, po, named[choice])
if bl:
    rv_, ro_ = ranks(pv), ranks(po)
    mv_, mo_ = dict(named[choice]), {j: i for i, j in named[choice]}
    st.warning(f"⚠️ {len(bl)} blockierende Paare in dieser Paarung (die ersten {min(len(bl), 12)}):")
    st.table({"Fahrzeug": [f"F{i + 1}" for i, _ in bl[:12]], "Auftrag": [f"A{j + 1}" for _, j in bl[:12]],
              "Fahrzeug: neue Wahl gegen jetzt": [f"Nr. {rv_[i][j] + 1} statt " + (f"Nr. {rv_[i][mv_[i]] + 1}" if i in mv_ else "allein") for i, j in bl[:12]],
              "Auftrag: neue Wahl gegen jetzt": [f"Nr. {ro_[j][i] + 1} statt " + (f"Nr. {ro_[j][mo_[j]] + 1}" if j in mo_ else "allein") for i, j in bl[:12]]})
else:
    st.success("✅ Kein blockierendes Paar: diese Paarung ist stabil.")

if code != "none":
    if net_key in C.FIXED_NETS:
        st.info("Feste Karte: es gibt nur diese eine Ziehung. Für die Verteilung über viele Karten eine zufällige Karte wählen.")
    else:
        st.markdown(f"**Nicht nur diese eine Karte:** {len(C.DIST_SEEDS)} feste Karten mit denselben Einstellungen (Fahrzeuge {n}, Aufträge {m}, Reichweite {reach}, Ballung {ballung} %, Vorlieben: {C.PREF_LABELS[pref]}"
                    + (f", Streuung ±{noise} min" if pref == "noise" else "") + "), getrennt vom Seed oben.")
        dist = _distribution(int(n), int(m), int(reach), int(ballung), pref, int(noise) if pref == "noise" else 0)
        if dist["n_valid"] == 0:
            st.info("ℹ️ Bei dieser Reichweite gibt es auf keiner der Karten ein mögliches Paar.")
        else:
            p1, p2, p3, p4 = st.columns(4)
            p1.metric("Mehrere stabile Paarungen", _share(dist["multi_share"]), delta=f"im Mittel {_f(dist['n_stable_mean'])}, höchstens {dist['n_stable_max']}", delta_color="off",
                      help="Anteil der Karten, auf denen es mehr als eine stabile Paarung gibt - nur dort macht die Wahl der Vorschlagenden etwas aus.")
            p2.metric("Prämie (Median | Mittel)", f"{_pct(dist['premium_median'], 1)} | {_pct(dist['premium_mean'], 1)}", delta=f"{_f(dist['lost_mean'], 2)} Paare weniger als das Optimum", delta_color="off",
                      help="Mehrkosten der stabilen Paarung (Fahrzeuge schlagen vor) gegenüber der billigsten Paarung mit derselben Paarzahl.")
            p3.metric("Optimum ist instabil", _share(dist["opt_unstable_share"]), delta=f"blockierende Paare: {_f(dist['opt_blocking_mean'])} | {_f(dist['opt_blocking_median'], 0)} (Mittel | Median)", delta_color="off",
                      help="Auf diesem Anteil der Karten hat das Optimum der Ungarischen Methode mindestens ein blockierendes Paar.")
            p4.metric("Vorschläge (Mittel | Median)", f"{_f(dist['proposals_v_mean'])} | {_f(dist['proposals_v_median'], 0)}", delta=f"Aufträge schlagen vor: {_f(dist['proposals_o_mean'])} | {_f(dist['proposals_o_median'], 0)}", delta_color="off",
                           help="Vorschläge, wenn die Fahrzeuge bzw. die Aufträge vorschlagen.")
            st.success(f"✅ Auf {_share(1.0)} der {dist['n_seeds']} Karten dieser Einstellung ist das Ergebnis stabil. Preis der Stabilität: Prämie im Median {_pct(dist['premium_median'], 1)} (Mittel {_pct(dist['premium_mean'], 1)}), "
                       f"{_f(dist['lost_mean'], 2)} Paare weniger als das Optimum im Mittel; das Optimum selbst ist auf {_share(dist['opt_unstable_share'])} der Karten instabil.")
            h1, h2 = st.columns([3, 2])
            h1.plotly_chart(build_premium_hist(dist["premium_v"], current=d["premium_pct"], median=dist["premium_median"]), width="stretch", key="premium_hist")
            h1.caption("Prämie über die Karten (Fahrzeuge schlagen vor); die Verteilung ist rechtsschief, deshalb Median und Mittel zusammen.")
            h2.markdown("**Wer vorschlägt** (Mittel über die Karten)")
            h2.table({"Vorschlagende": ["Fahrzeuge", "Aufträge"], "Prämie (Median)": [_pct(dist["premium_median"], 1), _pct(dist["premium_o_median"], 1)], "Ø Rang der Fahrzeuge": [_f(dist["rank_v_in_v"], 2), _f(dist["rank_v_in_o"], 2)],
                      "Ø Rang der Aufträge": [_f(dist["rank_o_in_v"], 2), _f(dist["rank_o_in_o"], 2)], "Kosten": [_f(dist["cost_v_mean"], 0), _f(dist["cost_o_mean"], 0)]})
            h2.caption("Rang 0 = erste Wahl. Der Vorschlagende hat den besseren Rang; die Gesamtkosten ändern sich dabei kaum.")

st.markdown("---")

# --- Alle stabilen Paarungen ------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Alle stabilen Paarungen")
lat = a.lattice
if lat["count"] == 1:
    st.info("Auf dieser Karte gibt es **genau eine** stabile Paarung: die Fahrzeuge und die Aufträge bekommen dasselbe, egal wer vorschlägt. " + ("Bei reiner Entfernung ist das immer so: beide Seiten haben dieselbe Wertung." if pref == "dist" else ""))
else:
    mats = lat["matchings"]
    veh_sets = {frozenset(i for i, _ in mm) for mm in mats}
    ord_sets = {frozenset(j for _, j in mm) for mm in mats}
    st.markdown(f"**{'Mindestens ' + str(lat['count'] - 1) if lat['capped'] else lat['count']} stabile Paarungen.** Landklinikensatz: in allen sind dieselben {len(next(iter(veh_sets)))} Fahrzeuge und dieselben {len(next(iter(ord_sets)))} Aufträge versorgt "
                f"({'stimmt' if len(veh_sets) == 1 and len(ord_sets) == 1 else 'stimmt hier nicht - das dürfte nicht vorkommen'}). Die Kosten der stabilen Paarungen liegen zwischen {d['min_stable_cost']} und {d['max_stable_cost']} Minuten.")
    sums = rank_sums(pv, po, mats)
    pts = [(f"Nr. {k}", x, y, "stable") for k, (x, y) in enumerate(sums, start=1)]
    rv_all = ev.stable_matching(pv, po, "V").pairs
    ro_all = ev.stable_matching(pv, po, "O").pairs
    rs = lambda pairs: (sum(ranks(pv)[i][j] for i, j in pairs), sum(ranks(po)[j][i] for i, j in pairs))
    pts += [("Fahrzeuge schlagen vor",) + rs(rv_all) + ("v",), ("Aufträge schlagen vor",) + rs(ro_all) + ("o",), ("Optimum",) + rs(tuple(sorted(a.hung.pairs))) + ("opt",),
            ("Greedy: billigste Kante",) + rs(tuple(sorted(a.edge.pairs))) + ("greedy",)]
    st.plotly_chart(build_scatter(pts), width="stretch", key="scatter_chart")
    st.caption("Jeder Punkt ist eine stabile Paarung: die Summe der Ränge, die die Fahrzeuge (waagerecht) und die Aufträge (senkrecht) ihren Partnern geben. Die beiden Sterne sind die Extreme: wer vorschlägt, ist auf seiner Achse ganz unten. Optimum und Greedy sind zum Vergleich eingezeichnet (die Summen zählen nur versorgte Agenten).")

st.markdown("**Lohnt sich Lügen?**")
st.caption("Bei Vorschlägen der Fahrzeuge kann sich **kein Fahrzeug** durch falsche Vorlieben verbessern - aber ein **Auftrag** kann es, genau dann, wenn die beiden Extrempaarungen für ihn verschieden sind. Kürzt er seine Liste auf einen einzigen Eintrag (seinen besten stabilen Partner), bekommt er ihn.")
if st.button("5×5-Karten erschöpfend durchprobieren (20 Karten, alle geordneten Teillisten, dauert einige Sekunden)", key="manip_start"):
    st.session_state["manip_on"] = (pref, int(noise) if pref == "noise" else 0)
if st.session_state.get("manip_on") == (pref, int(noise) if pref == "noise" else 0):
    with st.spinner("Probiere alle Teillisten..."):
        mp = _manipulation(pref, int(noise) if pref == "noise" else 0)
    st.table({"Wer lügt": ["Fahrzeuge (schlagen vor)", "Aufträge (Empfänger)"], "Agenten": [mp["veh_n"], mp["ord_n"]], "können gewinnen": [mp["veh_gain"], mp["ord_gain"]],
              "davon: Extreme verschieden": ["–", f"{mp['ord_diff']} (davon gewinnen {mp['ord_diff_gain']})"], "Kürzen auf einen Eintrag sichert den Partner": ["–", f"{mp['trunc_ok']} von {mp['trunc_n']}"]})
    st.caption("5 × 5 Fahrzeuge und Aufträge, alles erreichbar, 20 feste Karten, Vorlieben wie oben; die Vorschlagenden sind die Fahrzeuge. Bei reiner Entfernung oder kleiner Streuung gibt es hier kaum mehrere stabile Paarungen - dann gibt es auch nichts zu gewinnen. Bei Zufallsvorlieben gewinnen genau die Aufträge, deren Extrempartner verschieden sind.")

st.markdown("---")

# --- Experimente auf Abruf -------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Wovon hängt es ab?")
if st.button("Streuung von 0 bis 60 Minuten durchfahren (40 Karten je Wert, dauert wenige Sekunden)", key="noise_start"):
    st.session_state["noise_on"] = (int(n), int(m), int(reach), int(ballung))
if st.session_state.get("noise_on") == (int(n), int(m), int(reach), int(ballung)):
    with st.spinner(f"Rechne {len(C.NOISE_SWEEP)} Streuungen × {len(C.SWEEP_SEEDS)} Karten..."):
        nrows = _noise_sweep(int(n), int(m), int(reach), int(ballung))
    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(build_noise_sweep(nrows), width="stretch", key="noise_chart")
    c2.table({"Streuung [min]": [r["k"] for r in nrows], "mehrere stabile": [_share(r["multi"]) for r in nrows], "Prämie (Median)": [_pct(r["premium_median"], 1) for r in nrows], "Paare verloren": [_f(r["lost"], 2) for r in nrows]})
    st.caption("Mittel über 40 feste Karten je Wert; Fahrzeuge, Aufträge, Reichweite und Ballung wie oben. Bei 0 (nur Entfernung) gibt es nie mehr als eine stabile Paarung; je größer die Streuung, desto öfter macht die Wahl der Vorschlagenden etwas aus - und desto teurer wird die Stabilität. "
               "Die Streuung ist über k gekoppelt (gleiche Grundziehung), deshalb sind die Kurven glatt.")

if st.button("Anzahl stabiler Paarungen gegen die Kartengröße (n = 4 bis 40, alles erreichbar)", key="count_start"):
    st.session_state["count_on"] = (pref, int(noise) if pref == "noise" else 0)
if st.session_state.get("count_on") == (pref, int(noise) if pref == "noise" else 0):
    cpref = pref if pref != "dist" else "noise"
    with st.spinner("Rechne 5 Kartengrößen × 40 Karten..."):
        crows = _count_vs_size(cpref, int(noise) if cpref == "noise" else 0)
    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(build_count_vs_size(crows), width="stretch", key="count_chart")
    c2.table({"n = m": [r["n"] for r in crows], "stabile Paarungen (Mittel)": [_f(r["n_stable"], 2) for r in crows], "Maximum": [r["n_stable_max"] for r in crows], "Karten mit V ≠ A": [_share(r["vo"]) for r in crows]})
    st.caption(f"Vorlieben: {C.PREF_LABELS[cpref]}" + (f" (±{noise} min)" if cpref == "noise" else "") + ", n = m, alles erreichbar, 40 Karten je Größe (bei reiner Entfernung wäre es immer 1, deshalb hier die Streuung). "
               "Mit der Größe wächst die Zahl der stabilen Paarungen - und damit die Fälle, in denen es einen Unterschied macht, wer vorschlägt.")

if st.button("Aufwand gegen die Kartengröße (n = 10 bis 320)", key="effort_start"):
    st.session_state["effort_on"] = (pref, int(noise) if pref == "noise" else 0)
if st.session_state.get("effort_on") == (pref, int(noise) if pref == "noise" else 0):
    epref = pref if pref != "dist" else "random"
    with st.spinner("Rechne 6 Kartengrößen × 5 Karten..."):
        erows = _effort(epref, int(noise) if epref == "noise" else 0)
    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(build_effort(erows), width="stretch", key="effort_chart")
    c2.table({"n = m": [r["n"] for r in erows], "vollständige Listen": [_int(r["full"]) for r in erows], "konst. Grad": [_int(r["sparse"]) for r in erows], "Straße": [_int(r["road"]) if r["road"] is not None else "–" for r in erows], "n · H_n": [_int(r["nHn"]) for r in erows]})
    st.caption(f"Vorschläge der Fahrzeuge, Vorlieben: {C.PREF_LABELS[epref]}" + (f" (±{noise} min)" if epref == "noise" else "") + ", Mittel über 5 Karten. Bei vollständigen Zufallslisten wächst der Aufwand etwa wie n · H_n (n · ln n); bei konstantem mittleren Grad fast linear. "
               "Die Straße (alle einer Meinung) braucht genau n (n + 1) / 2 Vorschläge, aber nur bis n = 40 (danach überlappen sich Fahrzeuge und Aufträge auf der Geraden).")

st.markdown("---")

# --- Grenzen -----------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Es gibt zwei getrennte Seiten** | Fahrzeuge und Aufträge sind zwei Gruppen. Sollen sich Fahrer untereinander paaren (Mitbewohner), kann es **gar keine** stabile Paarung geben. | **Stabile Mitbewohner** |
| **Die Vorlieben sind strikt** | Bei Gleichstand in den Listen gibt es schwache und starke Stabilität; die Eindeutigkeit bei reiner Entfernung hängt an unserem Index-Tie-Break. | (nicht gebaut) |
| **Jeder Agent nimmt genau einen Partner** | Krankenhäuser mit mehreren Stellen (Kapazitäten) brauchen die Variante mit Kontingenten. | (nicht gebaut) |
| **Nur Stabilität zählt** | Die stabile Paarung ist oft teurer und hat weniger Paare als das Optimum (Prämie im Median 24,3 % bei Streuung 20; das Optimum ist instabil). Wer Kosten will, nimmt die Ungarische Methode. | **Ungarische Methode**, **Auktionsalgorithmus** (gebaut) |
| **Alle sagen die Wahrheit** | Vorschlagende können nichts gewinnen, Empfänger schon (Kürzen der Liste): das Verfahren ist nur für eine Seite anreizverträglich. | Mechanismusdesign |
| **Alles ist vorab bekannt** | Kommen Agenten nacheinander und sind Zusagen bindend, ist nur Online-Matching möglich. | **Online-Matching** |
"""
)
st.caption("Die Nachbarn der Matching-Linie (noch nicht gebaut): Blossom, Gewichteter Blossom, Stabile Mitbewohner und Online-Matching. Bereits gebaut: die Wurzel (Greedy-Matching), die Verbesserungswege, Hopcroft–Karp, die Ungarische Methode, der Auktionsalgorithmus und diese Demo.")

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Modell.** Bipartiter Graph mit Fahrzeugen $V$, Aufträgen $O$ und möglichen Paaren $E$ (beidseitig akzeptabel). Jedes $v\in V$ hat eine strikte Rangliste $\succ_v$ über seine möglichen Aufträge, jedes $o\in O$ eine über seine möglichen Fahrzeuge (unvollständige Listen).

**Stabilität.** Eine Paarung $M$ heißt stabil, wenn sie gültig ist und es **kein blockierendes Paar** gibt: kein $(v,o)\in E\setminus M$ mit $o\succ_v M(v)$ und $v\succ_o M(o)$ (wer allein ist, zieht jeden möglichen Partner vor).

**Aufgeschobene Annahme.** Solange ein Vorschlagender $p$ frei ist und Einträge übrig hat, fragt er den besten noch nicht gefragten Empfänger $r$. Ist $r$ frei oder zieht $p$ seinem Halter vor, hält $r$ vorläufig $p$ und lässt den bisherigen Halter frei; sonst lehnt er ab. Jeder Vorschlagende fragt jeden Empfänger höchstens einmal: höchstens $|E|$ Vorschläge, bei vollständigen Listen der Größe $n$ höchstens $n^2-n+1$. Auf Zufallslisten sind es im Mittel etwa $n H_n$.

**Sätze (Gale–Shapley 1962, McVitie–Wilson, Roth).** (1) Das Ergebnis ist stabil. (2) Es hängt nicht von der Reihenfolge der Vorschläge ab und ist für die Vorschlagenden die **beste** stabile Paarung (jeder hat den besten Partner, den er in irgendeiner stabilen Paarung bekommt), für die Empfänger die **schlechteste**. (3) **Landklinikensatz:** in allen stabilen Paarungen sind dieselben Agenten versorgt, die Paarzahl ist gleich. (4) Jede stabile Paarung ist maximal, hat also mindestens $\nu/2$ Paare ($\nu$ = Größe eines größtmöglichen Matchings).

**Verband.** Die stabilen Paarungen bilden einen distributiven Verband. Ausgehend von der fahrzeugoptimalen Paarung erhält man alle anderen durch **Rotationen**: das versorgte Fahrzeug $v$ zeigt auf den ersten Auftrag $s(v)$ unterhalb seines Partners, der $v$ seinem Halter vorzieht; ein Kreis solcher Zeiger schiebt alle seine Fahrzeuge auf ihr $s(v)$ (schlechter für Fahrzeuge, besser für Aufträge).

**Preis der Stabilität.** Mit den Grenzkosten $W_k$ der Ungarischen Methode ist $\sum_{i\le k}W_i$ die Kosten der billigsten Paarung mit $k$ Paaren; die Prämie einer stabilen Paarung mit $k$ Paaren ist $(c(M)-\sum_{i\le k}W_i)/\sum_{i\le k}W_i$.

**Reine Entfernung.** Ordnen beide Seiten nach derselben Kostenmatrix $c_{ij}$ (mit Index-Tie-Break), gibt es eine globale Rangfolge aller Paare; das billigste Paar ist in jeder stabilen Paarung enthalten, das zweitbilligste unter den übrigen usw. Die eindeutige stabile Paarung ist deshalb genau das Greedy „Billigste Kante zuerst“.

**Manipulation (Dubins–Freedman, Roth).** Bei Vorschlägen der Fahrzeuge kann sich kein Fahrzeug durch falsche Angaben verbessern; ein Auftrag kann es, wenn die Extrempaarungen für ihn verschieden sind.

**Grenzen.** (1) Zwei Seiten. (2) Strikte Listen. (3) Ein Partner je Agent. (4) Nur Stabilität. (5) Ehrlichkeit. (6) Alles vorab bekannt.

Implementiert in `gs_scenario.py` (Karten, eigener Zufallsgenerator, feste Karten), `gs_preferences.py` (Vorliebenmodelle), `gs_algorithm.py` (aufgeschobene Annahme, Prüfer, Beweis), `gs_lattice.py` (Verband), `gs_strategy.py` (Manipulation), `gs_greedy.py` und `gs_hungarian.py` (Greedy und Ungarische Methode aus den Vorgängerdemos), `gs_evaluation.py` (Kennzahlen, Verteilung, Sweeps).
        """
    )

st.markdown("---")

st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
