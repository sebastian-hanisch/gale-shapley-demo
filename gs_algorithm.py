"""Gale-Shapley (aufgeschobene Annahme): eine Seite schlägt vor, die andere hält immer den besten bisherigen Vorschlag und lehnt den Rest ab.

Jede Seite hat strikte Vorlieben über ihre möglichen Partner (unvollständige Listen: die Reichweite bestimmt, wer in Frage kommt). Ein Vorschlagender geht seine Liste
von oben nach unten durch; der Empfänger nimmt vorläufig an, wenn er noch frei ist oder den Vorschlagenden dem bisherigen Halter vorzieht (der wird verdrängt und
schlägt weiter unten in seiner Liste erneut vor). Das Verfahren endet, wenn jeder Vorschlagende versorgt ist oder seine Liste erschöpft hat. Das Ergebnis ist eine
STABILE Paarung (kein blockierendes Paar: zwei Seiten, die sich beide gegenüber ihren Partnern vorziehen) und - unabhängig von der Bearbeitungsreihenfolge - die beste
stabile Paarung für die Vorschlagenden und die schlechteste für die Empfänger.

Aufwand = Vorschläge (Kopfzahl), nie Sekunden. Alles ganzzahlig. Ereignisprotokoll für die Wiedergabe: ein Ereignis je Vorschlag.
Schalter für Negativkontrollen (Standard = richtiges Verfahren): `receiver` ("best" | "first" | "worst": welchen Vorschlag der Empfänger hält), `propose` ("best" | "worst":
in welcher Reihenfolge der Vorschlagende seine Liste abarbeitet).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Result:
    proposer: str          # "V" (Fahrzeuge schlagen vor) oder "O" (Aufträge schlagen vor)
    pairs: tuple           # ((Fahrzeug, Auftrag), ...) nach Fahrzeug sortiert
    proposals: int
    events: tuple          # je Vorschlag (Vorschlagender, Empfänger, Ausgang, Verdrängter); Ausgang "accept" | "displace" | "reject"
    ptr: tuple             # je Vorschlagender: wie viele Einträge seiner Liste er geprüft hat (Rang des Endpartners + 1, bzw. Listenlänge)

    @property
    def count(self):
        return len(self.pairs)


def _rank(lists):
    return [{x: k for k, x in enumerate(lst)} for lst in lists]


def gale_shapley(pp, pr, *, order=None, receiver="best", propose="best", record=True):
    """Aufgeschobene Annahme; `pp` = Listen der Vorschlagenden, `pr` = Listen der Empfänger. Rückgabe: (Paare {Vorschlagender: Empfänger}, Vorschläge, Ereignisse, ptr)."""
    rr = _rank(pr)
    n = len(pp)
    todo = list(order) if order is not None else list(range(n))
    lists = [list(x) if propose == "best" else list(reversed(x)) for x in pp]
    ptr = [0] * n
    holder, events, proposals = {}, [], 0
    free = todo[::-1]
    while free:
        p = free.pop()
        while ptr[p] < len(lists[p]):
            r = lists[p][ptr[p]]
            ptr[p] += 1
            proposals += 1
            cur = holder.get(r)
            if cur is None:
                holder[r] = p
                if record:
                    events.append((p, r, "accept", -1))
                break
            better = rr[r][p] < rr[r][cur] if receiver == "best" else (rr[r][p] > rr[r][cur] if receiver == "worst" else False)
            if better:
                holder[r] = p
                if record:
                    events.append((p, r, "displace", cur))
                free.append(cur)
                break
            if record:
                events.append((p, r, "reject", -1))
    return {p: r for r, p in holder.items()}, proposals, events, tuple(ptr)


def stable_matching(pv, po, proposer="V", **kw):
    """Stabile Paarung als Result mit Paaren (Fahrzeug, Auftrag)."""
    if proposer == "V":
        mt, prop, ev, ptr = gale_shapley(pv, po, **kw)
        pairs = tuple(sorted(mt.items()))
    else:
        mt, prop, ev, ptr = gale_shapley(po, pv, **kw)
        pairs = tuple(sorted((i, j) for j, i in mt.items()))
    return Result(proposer, pairs, prop, tuple(ev), ptr)


def state_at(res, n_proposers, k):
    """Zustand nach den ersten k Vorschlägen: Halter je Empfänger, Zeiger je Vorschlagender, freie (noch nicht versorgte) Vorschlagende."""
    holder, ptr = {}, [0] * n_proposers
    for p, r, outcome, cur in res.events[:k]:
        ptr[p] += 1
        if outcome in ("accept", "displace"):
            holder[r] = p
    matched = set(holder.values())
    return {"holder": holder, "ptr": ptr, "matched": matched}


def blocking_pairs(pv, po, pairs):
    """Blockierende Paare (Fahrzeug, Auftrag): beide möglich und beide ziehen einander ihrem jetzigen Partner (oder dem Alleinsein) vor. Unabhängige Doppelschleife."""
    rv, ro = _rank(pv), _rank(po)
    mv, mo = dict(pairs), {j: i for i, j in pairs}
    out = []
    for i, lst in enumerate(pv):
        for j in lst:
            if mv.get(i) == j:
                continue
            vi = rv[i][mv[i]] if i in mv else len(lst)
            oj = ro[j][mo[j]] if j in mo else len(po[j])
            if rv[i][j] < vi and ro[j][i] < oj:
                out.append((i, j))
    return out


def is_valid(pv, po, pairs):
    """Gültige Paarung: jede Seite höchstens einmal, nur mögliche Paare."""
    vs, os_ = [i for i, _ in pairs], [j for _, j in pairs]
    return len(set(vs)) == len(vs) and len(set(os_)) == len(os_) and all(j in pv[i] for i, j in pairs)


def certificate(pv, po, res, lattice=None):
    """Prüfzeilen der stabilen Paarung: S1 gültig, S2 kein blockierendes Paar, S3 maximal, S4 Vorschlagszahl = Summe (Rang + 1) bzw. Listenlänge,
    S5 (mit Verband) der Vorschlagende hat seinen besten stabilen Partner, S6 (mit Verband) dieselben Agenten sind in allen stabilen Paarungen versorgt (Landklinikensatz)."""
    prop_lists = pv if res.proposer == "V" else po
    pairs = res.pairs
    mv, mo = dict(pairs), {j: i for i, j in pairs}
    blocking = blocking_pairs(pv, po, pairs)
    maximal = not any(i not in mv and j not in mo for i, lst in enumerate(pv) for j in lst)
    prop_map = dict(pairs) if res.proposer == "V" else {j: i for i, j in pairs}
    rk = _rank(prop_lists)
    identity = sum(res.ptr) == res.proposals == sum(rk[p][prop_map[p]] + 1 if p in prop_map else len(prop_lists[p]) for p in range(len(prop_lists)))
    cert = {"s1": is_valid(pv, po, pairs), "s2": not blocking, "blocking": blocking, "s3": maximal, "s4": identity}
    if lattice is not None:
        stable = lattice["matchings"]
        sets = {frozenset(i for i, _ in m) for m in stable}, {frozenset(j for _, j in m) for m in stable}
        cert["s6"] = len(sets[0]) == 1 and len(sets[1]) == 1
        best = True
        for p in range(len(prop_lists)):
            partners = []
            for m in stable:
                d = dict(m) if res.proposer == "V" else {j: i for i, j in m}
                if p in d:
                    partners.append(rk[p][d[p]])
            if partners and (p not in prop_map or rk[p][prop_map[p]] != min(partners)):
                best = False
        cert["s5"] = best
    cert["all_ok"] = all(cert[k] for k in ("s1", "s2", "s3", "s4") + tuple(k for k in ("s5", "s6") if k in cert))
    return cert
