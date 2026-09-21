"""Auswertung: eine Karte (`analyse`, `verdict`, `compare_table`), viele Karten (`distribution`), Streuungs-Sweep, Anzahl stabiler Paarungen gegen Kartengröße, Aufwand,
Manipulation. Alles ganzzahlig und deterministisch; nur die Anzeige-Statistiken (Anteile, Mittel, Mediane) sind Gleitkomma.

Kosten sind über verschiedene Paarzahlen nicht vergleichbar (eine stabile Paarung hat oft weniger Paare als das Optimum). Der Preis der Stabilität wird deshalb als
"Prämie gegen die billigste Paarung mit derselben Paarzahl" ausgewiesen (aus den Grenzkosten der Ungarischen Methode: Summe der ersten k Wegkosten = billigste Paarung mit k Paaren)
und die verlorenen Paare getrennt. Aufwand = Vorschläge. Mittel und Median stehen zusammen.
"""

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

import gs_constants as C
from gs_algorithm import blocking_pairs, certificate, stable_matching
from gs_greedy import run_rule
from gs_hungarian import hungarian
from gs_lattice import all_stable, brute_force_stable, rank_sums
from gs_preferences import DEFAULT_NOISE, ranks, preferences
from gs_scenario import build, generate, road
from gs_strategy import best_response, truncate_to_one

STABLE, UNSTABLE, NONE = "stable", "unstable", "none"


@dataclass
class Analysis:
    scenario: object
    pref: str
    noise: int
    seed: int
    proposer: str
    pv: list
    po: list
    result: object        # stabile Paarung der gewählten Vorschlagenden (mit Protokoll)
    other: object         # stabile Paarung der anderen Seite
    lattice: dict         # alle stabilen Paarungen (begrenzt)
    hung: object          # Ungarische Methode (Optimum)
    edge: object          # Greedy: billigste Kante zuerst
    order: object         # Greedy: Auftrag für Auftrag
    kcost: tuple          # kcost[k] = Kosten der billigsten Paarung mit k Paaren


def _kcost(hung):
    out = [0]
    for w in hung.marginal:
        out.append(out[-1] + w)
    return tuple(out)


def cost_of(sc, pairs):
    return int(sum(sc.cost[i, j] for i, j in pairs))


def premium_pct(sc, pairs, kcost):
    """Mehrkosten in Prozent gegenüber der billigsten Paarung mit derselben Paarzahl (None, wenn diese kostenlos ist)."""
    k = len(pairs)
    base = kcost[k] if k < len(kcost) else None
    return None if not base else 100.0 * (cost_of(sc, pairs) - base) / base


def analyse(sc, pref=C.DEFAULT_PREF, noise=DEFAULT_NOISE, seed=0, proposer=C.DEFAULT_PROPOSER):
    pv, po = preferences(sc, pref, noise, seed)
    other_side = "O" if proposer == "V" else "V"
    hung = hungarian(sc, record=False)
    return Analysis(sc, pref, noise, seed, proposer, pv, po, stable_matching(pv, po, proposer), stable_matching(pv, po, other_side), all_stable(pv, po), hung,
                    run_rule(sc, "edge"), run_rule(sc, "order"), _kcost(hung))


def verdict(a):
    """(Stufe, Code, Zahlen) für die Anzeige; `Zahlen` enthält jede Zahl, die der Text nennt."""
    sc, r, o, h = a.scenario, a.result, a.other, a.hung
    blocking = blocking_pairs(a.pv, a.po, r.pairs)
    opt_block = blocking_pairs(a.pv, a.po, h.pairs)
    lat = a.lattice
    cert = certificate(a.pv, a.po, r, lat if not lat["capped"] else None)
    costs = [cost_of(sc, m) for m in lat["matchings"]]
    code = NONE if not any(a.pv) else (STABLE if not blocking else UNSTABLE)
    data = {"count": r.count, "cost": cost_of(sc, r.pairs), "opt_count": h.count, "opt_cost": h.cost, "lost": h.count - r.count, "premium_pct": premium_pct(sc, r.pairs, a.kcost),
            "blocking": len(blocking), "opt_blocking": len(opt_block), "opt_stable": not opt_block, "proposals": r.proposals, "other_proposals": o.proposals, "other_count": o.count,
            "other_cost": cost_of(sc, o.pairs), "other_premium_pct": premium_pct(sc, o.pairs, a.kcost), "n_stable": lat["count"], "capped": lat["capped"], "same_vo": r.pairs == o.pairs,
            "edge_count": a.edge.count, "edge_cost": a.edge.cost, "edge_blocking": len(blocking_pairs(a.pv, a.po, a.edge.pairs)), "order_count": a.order.count,
            "order_cost": a.order.cost, "order_blocking": len(blocking_pairs(a.pv, a.po, a.order.pairs)), "min_stable_cost": min(costs), "max_stable_cost": max(costs), "cert": cert,
            "same_as_greedy": r.pairs == tuple(sorted(a.edge.pairs)), "edges": int(sc.feasible.sum())}
    level = {NONE: "info", STABLE: "success", UNSTABLE: "error"}[code]
    return level, code, data


def named_matchings(a):
    """Paarungen zum Prüfen auf blockierende Paare: Ergebnis, andere Vorschlagende, Optimum, beide Greedy und (bis zu 30) weitere stabile Paarungen."""
    out = {"Ergebnis (gewählte Vorschlagende)": a.result.pairs, "Andere Vorschlagende": a.other.pairs, "Optimum (Ungarische Methode)": tuple(sorted(a.hung.pairs)),
           "Greedy: billigste Kante zuerst": tuple(sorted(a.edge.pairs)), "Greedy: Auftrag für Auftrag": tuple(sorted(a.order.pairs))}
    for k, m in enumerate(a.lattice["matchings"][:30], start=1):
        out[f"Stabile Paarung Nr. {k}"] = tuple(sorted(m))
    return out


def compare_table(a):
    sc, pv, po = a.scenario, a.pv, a.po
    rows = []
    for label, pairs, props in (("Ungarische Methode (Optimum)", tuple(sorted(a.hung.pairs)), None), ("Greedy: billigste Kante zuerst", tuple(sorted(a.edge.pairs)), None),
                                ("Greedy: Auftrag für Auftrag", tuple(sorted(a.order.pairs)), None), ("Stabil: Fahrzeuge schlagen vor", stable_matching(pv, po, "V").pairs, stable_matching(pv, po, "V").proposals),
                                ("Stabil: Aufträge schlagen vor", stable_matching(pv, po, "O").pairs, stable_matching(pv, po, "O").proposals)):
        rows.append({"label": label, "count": len(pairs), "cost": cost_of(sc, pairs), "premium_pct": premium_pct(sc, pairs, a.kcost), "blocking": len(blocking_pairs(pv, po, pairs)), "proposals": props})
    return rows


# --- viele Karten --------------------------------------------------------------------------------------------------------------

def _mean_rank(pairs, lists, side):
    rk = ranks(lists)
    vals = [rk[i][j] for i, j in pairs] if side == "V" else [rk[j][i] for i, j in pairs]
    return sum(vals), len(vals)


@lru_cache(maxsize=256)
def cell_rows(n, m, reach, ballung, pref, noise, seeds):
    """Je Seed als Ganzzahlen (plattformunabhängig): Optimum, Greedy, beide stabilen Paarungen, Anzahl stabiler Paarungen, Ränge. Rückgabe: Tupel von Dicts."""
    rows = []
    for sd in seeds:
        sc = generate(n, m, reach, ballung, sd)
        pv, po = preferences(sc, pref, noise, sd)
        h = hungarian(sc, record=False)
        kc = _kcost(h)
        sv, so = stable_matching(pv, po, "V"), stable_matching(pv, po, "O")
        lat = all_stable(pv, po)
        eg, eo = run_rule(sc, "edge"), run_rule(sc, "order")
        hp = tuple(sorted(h.pairs))
        rows.append({"hung": (h.count, h.cost), "opt_blocking": len(blocking_pairs(pv, po, hp)), "edge": (eg.count, eg.cost, len(blocking_pairs(pv, po, eg.pairs))),
                     "order": (eo.count, eo.cost, len(blocking_pairs(pv, po, eo.pairs))), "kc": kc,
                     "sv": (sv.count, cost_of(sc, sv.pairs), sv.proposals), "so": (so.count, cost_of(sc, so.pairs), so.proposals),
                     "n_stable": lat["count"], "capped": lat["capped"], "same_vo": sv.pairs == so.pairs, "same_greedy": sv.pairs == tuple(sorted(eg.pairs)),
                     "rank_v_in_v": _mean_rank(sv.pairs, pv, "V"), "rank_o_in_v": _mean_rank(sv.pairs, po, "O"), "rank_v_in_o": _mean_rank(so.pairs, pv, "V"), "rank_o_in_o": _mean_rank(so.pairs, po, "O")})
    return tuple(rows)


def _med(values):
    return float(np.median(values)) if len(values) else None


def _prem(cost, count, kc):
    base = kc[count] if count < len(kc) else 0
    return None if not base else 100.0 * (cost - base) / base


def distribution(n, m, reach, ballung, pref=C.DEFAULT_PREF, noise=DEFAULT_NOISE, seeds=C.DIST_SEEDS):
    """Verteilung über viele Karten: Preis der Stabilität, Anzahl stabiler Paarungen, Vorschlagszahl, Blockierende Paare (Mittel und Median)."""
    rows = [r for r in cell_rows(n, m, reach, ballung, pref, noise, tuple(seeds)) if r["hung"][0] > 0]
    nv = len(rows)
    if nv == 0:
        return {"n_seeds": len(seeds), "n_valid": 0}
    pv_ = [p for p in (_prem(r["sv"][1], r["sv"][0], r["kc"]) for r in rows) if p is not None]
    po_ = [p for p in (_prem(r["so"][1], r["so"][0], r["kc"]) for r in rows) if p is not None]
    ns = [r["n_stable"] for r in rows]
    mean_rank = lambda key: sum(r[key][0] for r in rows) / max(sum(r[key][1] for r in rows), 1)
    return {"n_seeds": len(seeds), "n_valid": nv, "opt_pairs_mean": float(np.mean([r["hung"][0] for r in rows])), "stable_pairs_mean": float(np.mean([r["sv"][0] for r in rows])),
            "lost_mean": float(np.mean([r["hung"][0] - r["sv"][0] for r in rows])), "edge_pairs_mean": float(np.mean([r["edge"][0] for r in rows])),
            "opt_unstable_share": float(np.mean([r["opt_blocking"] > 0 for r in rows])), "opt_blocking_mean": float(np.mean([r["opt_blocking"] for r in rows])), "opt_blocking_median": _med([r["opt_blocking"] for r in rows]),
            "edge_blocking_mean": float(np.mean([r["edge"][2] for r in rows])), "order_blocking_mean": float(np.mean([r["order"][2] for r in rows])), "order_blocking_median": _med([r["order"][2] for r in rows]),
            "premium_mean": float(np.mean(pv_)) if pv_ else None, "premium_median": _med(pv_), "premium_o_mean": float(np.mean(po_)) if po_ else None, "premium_o_median": _med(po_),
            "n_stable_mean": float(np.mean(ns)), "n_stable_median": _med(ns), "n_stable_max": max(ns), "multi_share": float(np.mean([x > 1 for x in ns])), "unique_share": float(np.mean([x == 1 for x in ns])),
            "capped_share": float(np.mean([r["capped"] for r in rows])), "vo_differ_share": float(np.mean([not r["same_vo"] for r in rows])), "same_greedy_share": float(np.mean([r["same_greedy"] for r in rows])),
            "proposals_v_mean": float(np.mean([r["sv"][2] for r in rows])), "proposals_v_median": _med([r["sv"][2] for r in rows]),
            "proposals_o_mean": float(np.mean([r["so"][2] for r in rows])), "proposals_o_median": _med([r["so"][2] for r in rows]),
            "rank_v_in_v": mean_rank("rank_v_in_v"), "rank_o_in_v": mean_rank("rank_o_in_v"), "rank_v_in_o": mean_rank("rank_v_in_o"), "rank_o_in_o": mean_rank("rank_o_in_o"),
            "cost_v_mean": float(np.mean([r["sv"][1] for r in rows])), "cost_o_mean": float(np.mean([r["so"][1] for r in rows])), "opt_cost_mean": float(np.mean([r["hung"][1] for r in rows])),
            "premium_v": pv_}


def noise_sweep(n, m, reach, ballung, seed_set=C.SWEEP_SEEDS, ks=C.NOISE_SWEEP):
    """Je Streuung k (Minuten): Anteil der Karten mit mehreren stabilen Paarungen, Anteil V ≠ O, Prämie, verlorene Paare (k = 0 ist die reine Entfernung)."""
    out = []
    for k in ks:
        d = distribution(n, m, reach, ballung, "dist" if k == 0 else "noise", max(k, 1), seed_set)
        if d["n_valid"] == 0:
            continue
        out.append({"k": k, "multi": d["multi_share"], "vo": d["vo_differ_share"], "premium_median": d["premium_median"], "premium_mean": d["premium_mean"], "lost": d["lost_mean"], "n_stable": d["n_stable_mean"]})
    return out


def count_vs_size(pref, noise, ns=C.COUNT_NS, seeds=C.SIZE_SEEDS, reach=150):
    """Anzahl stabiler Paarungen und Anteil V ≠ O gegen die Kartengröße (n = m, alles erreichbar)."""
    out = []
    for n in ns:
        d = distribution(n, n, reach, 0, pref, noise, seeds)
        out.append({"n": n, "n_stable": d["n_stable_mean"], "n_stable_max": d["n_stable_max"], "vo": d["vo_differ_share"], "capped": d["capped_share"]})
    return out


@lru_cache(maxsize=8)
def effort_scaling(pref, noise, ns=C.EFFORT_NS, seeds=C.EFFORT_SEEDS):
    """Vorschläge gegen die Kartengröße: vollständige Listen (alles erreichbar), konstanter mittlerer Grad und die Worst-Case-Straße; dazu n · H_n."""
    out = []
    for n in ns:
        full, sparse = [], []
        reach = max(5, round(C.SCALE_REACH_AT_20 * math.sqrt(20 / n)))
        for sd in seeds:
            for lst, r in ((full, 300), (sparse, reach)):
                sc = generate(n, n, r, 0, sd)
                pv, po = preferences(sc, pref, noise, sd)
                lst.append(stable_matching(pv, po, "V", record=False).proposals)
        out.append({"n": n, "full": float(np.mean(full)), "sparse": float(np.mean(sparse)), "road": (stable_matching(*preferences(road(n), "dist"), "V", record=False).proposals if n <= 40 else None),
                    "nHn": n * sum(1 / k for k in range(1, n + 1))})
    return tuple(out)


@lru_cache(maxsize=8)
def manipulation(pref, noise, size=5, seeds=C.MANIP_SEEDS):
    """Erschöpfende Manipulationsprobe auf size x size-Karten (alles erreichbar; Vorschläge der Fahrzeuge): wer kann sich durch falsche Vorlieben verbessern?
    Rückgabe: Anteile Fahrzeuge / Aufträge, die gewinnen können, und ob Kürzen auf einen Eintrag den auftragsoptimalen Partner sichert."""
    veh_gain = veh_n = ord_gain = ord_n = ord_diff = ord_diff_gain = trunc_ok = trunc_n = 0
    for sd in seeds:
        sc = generate(size, size, 300, 0, sd)
        pv, po = preferences(sc, pref, noise, sd)
        lat = all_stable(pv, po)
        rk_o = ranks(po)
        for i in range(sc.n):
            honest, best, _ = best_response(pv, po, "V", i)
            veh_n += 1
            veh_gain += best < honest
        best_o = {}
        for m_ in lat["matchings"]:
            for i, j in m_:
                best_o[j] = min(best_o.get(j, 10 ** 9), rk_o[j][i])
        for j in range(sc.m):
            honest, best, _ = best_response(pv, po, "O", j)
            ord_n += 1
            ord_gain += best < honest
            worst_partner = None
            partners = {dict((b, a) for a, b in m_).get(j) for m_ in lat["matchings"]}
            if len(partners) > 1:
                ord_diff += 1
                ord_diff_gain += best < honest
                target = min(partners, key=lambda x: rk_o[j][x])
                trunc_n += 1
                trunc_ok += truncate_to_one(pv, po, j, target)
    return {"veh_gain": veh_gain, "veh_n": veh_n, "ord_gain": ord_gain, "ord_n": ord_n, "ord_diff": ord_diff, "ord_diff_gain": ord_diff_gain, "trunc_ok": trunc_ok, "trunc_n": trunc_n}


def scenario_from_settings(net, n, m, reach, ballung, seed):
    return build(net, n, m, reach, ballung, seed)
