"""Literatur-Referenzwerte für Gale-Shapley: Beispiele, deren Ergebnis in einer eingesehenen Quelle steht (Einsicht jeweils am 2026-10-17).

Verglichen wird das Ergebnis der Demo mit der Quelle - nicht mit einem Wert, den der Code der Demo selbst erzeugt hat.

 L1  Knuth, Motwani, Pittel, "Stable Husbands", arXiv:math/9201303v1 (Stanford). Einleitung (Seite 1): Alice, Brigitte, Cindy, Debra und Wilfred, Xavier, Yuri, Zeke
     mit den dort in einer Tabelle stehenden Rangfolgen. Aussagen der Quelle: (AW, BX, CY, DZ) ist instabil ("A prefers Z to W and at the same time Z prefers A to D"),
     (AZ, BW, CX, DY) ist stabil, (AY, BW, CX, DZ) ist die zweite stabile Paarung, "only two matchings are stable". Abschnitt 2 (Seite 3): die Ablaufprotokoll-Tabelle des
     Verfahrens mit den Jungen als Vorschlagenden findet als ERSTE stabile Paarung (AZ, BW, CX, DY); bis dahin stehen dort sieben Vorschläge (W-A, X-C, Y-B, Z-B abgelehnt,
     Z-A, W-B, Y-D). Dass die Mädchen als Vorschlagende die andere Paarung bekommen, folgt aus der Quelle (nur zwei stabile Paarungen) und der Optimalitätsaussage; es steht
     dort nicht wörtlich.
 L2  Vorlesungsnotizen MS&E 319 Matching Theory, Stanford, Lecture 8 (Scribe Emil Martayan, 20.05.2019), Abschnitt 1 "Stable Matching", Beispiel: M = {A, B, C}, W = {1, 2, 3},
     A: 123, B: 213, C: 321, 1: BAC, 2: ABC, 3: CBA. Aussagen der Quelle: P_M(A) = 1, P_M(B) = 2, P_M(C) = 3 hat keine "rogue pairs" (stabil); P_M(C) = 1, P_M(B) = 2, P_M(A) = 3 ist
     instabil, "(A, 1) is a rogue pair". (Die Klammerbemerkung der Quelle "1, 3, A, C get their first choices" stimmt nicht: Frau 1 bekommt A, ihre zweite Wahl; sie wird hier
     nicht verwendet.) Dass das Verfahren mit den Männern als Vorschlagenden die stabile Paarung A1, B2, C3 liefert, ist von Hand gerechnet: alle drei Männer bekommen ihre erste Wahl.
 L3  Wikipedia-Beispiel (gegengerechnet), "Stable marriage problem", Abschnitt zum Beispiel: A: YXZ, B: ZYX, C: XZY; X: BAC, Y: CBA, Z: ACB. Genannt werden drei stabile Paarungen
     (AY, BZ, CX) [Männer erste Wahl], (AX, BY, CZ) [alle zweite Wahl], (AZ, BX, CY) [Frauen erste Wahl]. Wikipedia ist keine Primärquelle; die drei Paarungen wurden hier mit
     der Definition (Brute Force über alle Paarungen) nachgerechnet, der Test prüft zusätzlich, dass es keine weiteren gibt.

NICHT als Literaturwert verwendet (Quelle nicht einsehbar): Gale/Shapley 1962 (nur als Bildscan ohne Text erreichbar), Knuth 1976/1997 (Buch), die 10 stabilen Paarungen der
Karte `latin4` der Demo (kein einsehbarer Beleg für genau diese Listen).
"""

import itertools

from gs_algorithm import blocking_pairs, stable_matching
from gs_lattice import all_stable


def _brute_stable(pv, po):
    """Alle stabilen Paarungen nach Definition (eigene Aufzählung, vollständige Listen)."""
    n, m = len(pv), len(po)
    out = set()
    for k in range(min(n, m) + 1):
        for vs in itertools.combinations(range(n), k):
            for os_ in itertools.permutations(range(m), k):
                mm = tuple(zip(vs, os_))
                if not blocking_pairs(pv, po, mm):
                    out.add(frozenset(mm))
    return out


# --- L1 Knuth/Motwani/Pittel: Jungen W X Y Z = Fahrzeuge 0..3, Mädchen A B C D = Aufträge 0..3 ----------------------------------------------

KMP_PV = [[0, 1, 3, 2],        # Wilfred: A > B > D > C
          [2, 0, 3, 1],        # Xavier:  C > A > D > B
          [1, 3, 0, 2],        # Yuri:    B > D > A > C
          [1, 0, 2, 3]]        # Zeke:    B > A > C > D
KMP_PO = [[2, 1, 3, 0],        # Alice:    Y > X > Z > W
          [1, 0, 2, 3],        # Brigitte: X > W > Y > Z
          [0, 2, 1, 3],        # Cindy:    W > Y > X > Z
          [1, 0, 3, 2]]        # Debra:    X > W > Z > Y
# (Junge, Mädchen): AZ = (3, 0), BW = (0, 1), CX = (1, 2), DY = (2, 3); AY = (2, 0), DZ = (3, 3)
KMP_BOYS_OPTIMAL = ((0, 1), (1, 2), (2, 3), (3, 0))              # (AZ, BW, CX, DY)
KMP_GIRLS_OPTIMAL = ((0, 1), (1, 2), (2, 0), (3, 3))             # (AY, BW, CX, DZ)


def test_l1_knuth_motwani_pittel_four_by_four_has_exactly_two_stable_matchings():
    assert _brute_stable(KMP_PV, KMP_PO) == {frozenset(KMP_BOYS_OPTIMAL), frozenset(KMP_GIRLS_OPTIMAL)}      # "only two matchings are stable"
    lat = all_stable(KMP_PV, KMP_PO)
    assert lat["count"] == 2 and set(lat["matchings"]) == {frozenset(KMP_BOYS_OPTIMAL), frozenset(KMP_GIRLS_OPTIMAL)} and not lat["capped"]


def test_l1_the_source_unstable_matching_is_blocked_by_alice_and_zeke_and_the_stable_ones_are_not():
    unstable = ((0, 0), (1, 1), (2, 2), (3, 3))                      # (AW, BX, CY, DZ) als (Junge, Mädchen): W-A, X-B, Y-C, Z-D
    assert (3, 0) in blocking_pairs(KMP_PV, KMP_PO, unstable)        # Zeke (3) und Alice (0): "A prefers Z to W and at the same time Z prefers A to D"
    assert not blocking_pairs(KMP_PV, KMP_PO, KMP_BOYS_OPTIMAL) and not blocking_pairs(KMP_PV, KMP_PO, KMP_GIRLS_OPTIMAL)


def test_l1_deferred_acceptance_with_the_boys_proposing_finds_the_first_matching_of_the_source_trace_after_seven_proposals():
    v = stable_matching(KMP_PV, KMP_PO, "V")
    assert v.pairs == KMP_BOYS_OPTIMAL and v.proposals == 7           # Ablaufprotokoll Abschnitt 2: sieben Vorschläge bis zur ersten Ausgabe (AZ, BW, CX, DY)
    o = stable_matching(KMP_PV, KMP_PO, "O")
    assert o.pairs == KMP_GIRLS_OPTIMAL                               # Mädchen als Vorschlagende: die andere der beiden stabilen Paarungen


def test_l1_alices_stable_husbands_are_yuri_and_zeke_and_brigitte_has_only_wilfred():
    husbands = {g: {dict((b, a) for a, b in mm)[g] for mm in all_stable(KMP_PV, KMP_PO)["matchings"]} for g in range(4)}
    assert husbands[0] == {2, 3} and husbands[1] == {0}               # Quelle: "Alice's stable husbands ... are Yuri and Zeke. Brigitte has only one ... Wilfred"


# --- L2 Stanford MS&E 319, Lecture 8: Männer A B C = 0..2, Frauen 1 2 3 = 0..2 ---------------------------------------------------------------

ST_PV = [[0, 1, 2], [1, 0, 2], [2, 1, 0]]       # A: 123, B: 213, C: 321
ST_PO = [[1, 0, 2], [0, 1, 2], [2, 1, 0]]       # 1: BAC, 2: ABC, 3: CBA


def test_l2_stanford_lecture_example_stable_and_unstable_matching():
    stable = ((0, 0), (1, 1), (2, 2))                                 # A1, B2, C3
    unstable = ((0, 2), (1, 1), (2, 0))                               # P_M(C) = 1, P_M(B) = 2, P_M(A) = 3
    assert not blocking_pairs(ST_PV, ST_PO, stable)
    assert (0, 0) in blocking_pairs(ST_PV, ST_PO, unstable)           # "(A, 1) is a rogue pair"
    assert stable_matching(ST_PV, ST_PO, "V").pairs == stable         # von Hand: jeder Mann bekommt seine erste Wahl


# --- L3 Wikipedia-Beispiel (gegengerechnet): Männer A B C = 0..2, Frauen X Y Z = 0..2 ---------------------------------------------------------

WP_PV = [[1, 0, 2], [2, 1, 0], [0, 2, 1]]       # A: YXZ, B: ZYX, C: XZY
WP_PO = [[1, 0, 2], [2, 1, 0], [0, 2, 1]]       # X: BAC, Y: CBA, Z: ACB
WP_MEN_FIRST = ((0, 1), (1, 2), (2, 0))         # (AY, BZ, CX)
WP_ALL_SECOND = ((0, 0), (1, 1), (2, 2))        # (AX, BY, CZ)
WP_WOMEN_FIRST = ((0, 2), (1, 0), (2, 1))       # (AZ, BX, CY)


def test_l3_wikipedia_three_by_three_example_has_the_three_named_stable_matchings_and_no_more():
    expected = {frozenset(WP_MEN_FIRST), frozenset(WP_ALL_SECOND), frozenset(WP_WOMEN_FIRST)}
    assert _brute_stable(WP_PV, WP_PO) == expected
    lat = all_stable(WP_PV, WP_PO)
    assert lat["count"] == 3 and set(lat["matchings"]) == expected
    assert stable_matching(WP_PV, WP_PO, "V").pairs == WP_MEN_FIRST       # Männer schlagen vor: erste Wahl für jeden Mann, dritte für jede Frau
    assert stable_matching(WP_PV, WP_PO, "O").pairs == WP_WOMEN_FIRST     # Frauen schlagen vor: erste Wahl für jede Frau
