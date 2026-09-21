"""Der Verband der stabilen Paarungen: alle stabilen Paarungen aus der fahrzeugoptimalen durch Rotationen.

Ausgehend von der fahrzeugoptimalen Paarung M0 (Fahrzeuge schlagen vor) zeigt jedes versorgte Fahrzeug i auf den ersten Auftrag s(i) unterhalb seines jetzigen Partners,
der i seinem jetzigen Halter vorzieht; das Fahrzeug, das s(i) hält, ist der Nachfolger. Ein Kreis solcher Zeiger ist eine Rotation: schiebt man alle Fahrzeuge des Kreises auf
ihren s(i), entsteht wieder eine stabile Paarung (schlechter für die Fahrzeuge, besser für die Aufträge). Unvollständige Listen: ist der nächste Kandidat ein Auftrag, den
die fahrzeugoptimale Paarung unversorgt lässt, ist die Kette eine Sackgasse (Landklinikensatz: unversorgte Agenten bleiben in jeder stabilen Paarung unversorgt).
Die Aufzählung ist auf `limit` stabile Paarungen begrenzt ("mindestens"); Brute Force dient nur den Tests.
"""

from gs_algorithm import _rank, gale_shapley

LIMIT = 2000


def all_stable(pv, po, limit=LIMIT):
    """Alle stabilen Paarungen als frozenset von (Fahrzeug, Auftrag). Rückgabe: {"matchings": [...], "capped": bool, "count": int}."""
    rv, ro = _rank(pv), _rank(po)
    m0 = frozenset(gale_shapley(pv, po, record=False)[0].items())
    seen, stack, capped = {m0}, [m0], False
    while stack:
        cur = stack.pop()
        mv = dict(cur)
        mo = {j: i for i, j in mv.items()}
        nxt, snext = {}, {}
        for i, j in mv.items():
            for j2 in pv[i][rv[i][j] + 1:]:
                if j2 not in mo:
                    break                                     # Sackgasse: in jeder stabilen Paarung unversorgt
                if ro[j2][i] < ro[j2][mo[j2]]:
                    nxt[i], snext[i] = mo[j2], j2
                    break
        color = {}
        for i0 in nxt:
            if i0 in color:
                continue
            path, i = [], i0
            while i in nxt and i not in color:
                color[i] = i0
                path.append(i)
                i = nxt[i]
            if i in color and color[i] == i0 and i in path:
                cyc = path[path.index(i):]
                new = dict(mv)
                for a in cyc:
                    new[a] = snext[a]
                key = frozenset(new.items())
                if key not in seen:
                    seen.add(key)
                    if len(seen) > limit:
                        return {"matchings": sorted(seen, key=sorted), "capped": True, "count": len(seen)}
                    stack.append(key)
    return {"matchings": sorted(seen, key=sorted), "capped": capped, "count": len(seen)}


def rank_sums(pv, po, matchings):
    """Je Paarung (Summe der Ränge der Fahrzeuge, Summe der Ränge der Aufträge); 0 = beste Wahl, unversorgte zählen nicht."""
    rv, ro = _rank(pv), _rank(po)
    return [(sum(rv[i][j] for i, j in m), sum(ro[j][i] for i, j in m)) for m in matchings]


def brute_force_stable(pv, po):
    """Alle stabilen Paarungen durch Aufzählen aller Paarungen (nur winzige Karten): Orakel für die Rotationen."""
    from gs_algorithm import blocking_pairs
    n = len(pv)
    out = []

    def rec(i, used, cur):
        if i == n:
            if not blocking_pairs(pv, po, cur):
                out.append(frozenset(cur))
            return
        rec(i + 1, used, cur)
        for j in pv[i]:
            if j not in used:
                rec(i + 1, used | {j}, cur + [(i, j)])

    rec(0, frozenset(), [])
    return out
