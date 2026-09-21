"""Plotly-Abbildungen von Gale–Shapley: Karte mit vorläufigen Paaren und dem letzten Ereignis, Verlauf, Verband der stabilen Paarungen (Streudiagramm), Prämie, Streuungs-Sweep, Anzahl und Aufwand.
Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht zoomen. Punkte auf einer Geraden werden mit Bögen gezeichnet."""

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import gs_constants as C


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.08), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def _collinear(sc):
    """Liegen alle Punkte auf einer Geraden? Dann würden sich die Paar-Linien überdecken - sie werden gebogen gezeichnet."""
    pts = list(sc.vehicles + sc.orders)
    (x0, y0), (x1, y1) = pts[0], next((p for p in pts if p != pts[0]), pts[0])
    return all((x1 - x0) * (y - y0) - (y1 - y0) * (x - x0) == 0 for x, y in pts)


def _edge_path(sc, i, j, curved, steps=14):
    """Punkte einer Paar-Linie: gerade, oder (bei Punkten auf einer Geraden) als Bogen, dessen Seite je Paar wechselt."""
    (vx, vy), (ox, oy) = sc.vehicles[i], sc.orders[j]
    if not curved:
        return [vx, ox], [vy, oy]
    dx, dy = ox - vx, oy - vy
    side = 1 if (i + j) % 2 == 0 else -1
    cx, cy = (vx + ox) / 2 - side * 0.35 * dy, (vy + oy) / 2 + side * 0.35 * dx      # Kontrollpunkt senkrecht zur Verbindung
    ts = [k / steps for k in range(steps + 1)]
    return ([(1 - t) ** 2 * vx + 2 * (1 - t) * t * cx + t * t * ox for t in ts], [(1 - t) ** 2 * vy + 2 * (1 - t) * t * cy + t * t * oy for t in ts])


def _segments(sc, pairs, curved=False):
    """Linienspur für eine Menge von Paaren (None trennt die Segmente)."""
    x, y = [], []
    for i, j in pairs:
        px, py = _edge_path(sc, i, j, curved)
        x += px + [None]
        y += py + [None]
    return x, y


def _feasible_pairs(sc):
    return [(i, j) for i in range(sc.n) for j in range(sc.m) if sc.feasible[i, j]]


def _map_layout(fig, sc, height):
    xs = [p[0] for p in sc.vehicles + sc.orders]
    ys = [p[1] for p in sc.vehicles + sc.orders]
    pad = 8
    if _collinear(sc):
        # Punkte auf einer Geraden: nur die Bögen brauchen Höhe. Das Seitenverhältnis wird freigegeben, sonst wird eine lange Kette zu einem dünnen Streifen.
        span = max(max(xs) - min(xs), max(ys) - min(ys), 1)
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        if max(xs) - min(xs) >= max(ys) - min(ys):
            fig.update_xaxes(visible=False, range=[min(xs) - pad, max(xs) + pad])
            fig.update_yaxes(visible=False, range=[cy - span * 0.22, cy + span * 0.22])
        else:
            fig.update_xaxes(visible=False, range=[cx - span * 0.22, cx + span * 0.22])
            fig.update_yaxes(visible=False, range=[min(ys) - pad, max(ys) + pad])
        return _base(fig, min(height, 320))
    # constrain="domain": bei gleichem Maßstab schrumpft die Zeichenfläche statt dass sich der Wertebereich aufbläht (sonst kann eine Karte, die beim ersten Zeichnen in einer noch schmalen Spalte steht, dauerhaft auf einen Punkt kollabieren)
    fig.update_xaxes(visible=False, range=[min(xs) - pad, max(xs) + pad], scaleanchor="y", scaleratio=1, constrain="domain")
    fig.update_yaxes(visible=False, range=[min(ys) - pad, max(ys) + pad], constrain="domain")
    return _base(fig, height)


def _scaled(values, vmax, lo, hi):
    return [lo + (hi - lo) * min(v, vmax) / vmax if vmax else lo for v in values]


def build_gs_map(sc, pairs_now, matched_v, matched_o, event=None, proposer="V", blocking=(), height=430):
    """Karte nach k Vorschlägen: vorläufige Paare blau, das letzte Ereignis hervorgehoben (Vorschlag angenommen grün, verdrängt orange, abgelehnt rot gestrichelt); der Vorschlagende ist
    umringt, ein verdrängter Partner rot. `blocking` (Paare (Fahrzeug, Auftrag)) werden violett gestrichelt gezeichnet."""
    curved = _collinear(sc)
    fig = go.Figure()
    ex, ey = _segments(sc, _feasible_pairs(sc), curved)
    fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", line=dict(color="rgba(150,150,150,0.3)", width=1), hoverinfo="skip", name="mögliche Paare"))
    px, py = _segments(sc, list(pairs_now), curved)
    fig.add_trace(go.Scatter(x=px, y=py, mode="lines", line=dict(color=C.COLORS["matched"], width=3.5), hoverinfo="skip", name="vorläufig gehalten"))
    if blocking:
        bx, by = _segments(sc, list(blocking), curved)
        fig.add_trace(go.Scatter(x=bx, y=by, mode="lines", line=dict(color=C.COLORS["block"], width=4, dash="dash"), hoverinfo="skip", name="blockierend"))
    rings = []
    if event is not None:
        p, r, outcome, cur = event
        i, j = (p, r) if proposer == "V" else (r, p)
        color = {"accept": C.COLORS["propose"], "displace": C.COLORS["displace"], "reject": C.COLORS["reject"]}[outcome]
        lx, ly = _segments(sc, [(i, j)], curved)
        fig.add_trace(go.Scatter(x=lx, y=ly, mode="lines", line=dict(color=color, width=5, dash="dash" if outcome == "reject" else None), hoverinfo="skip", name=outcome))
        rings.append(("V" if proposer == "V" else "O", p, C.COLORS["propose"]))
        if cur >= 0:
            rings.append(("V" if proposer == "V" else "O", cur, C.COLORS["reject"]))
    small = sc.n + sc.m <= 24
    for kind, pts, matched, symbol, prefix, tpos, color in (("Fahrzeug", sc.vehicles, matched_v, "square", "F", "top center", "#2e7d32"), ("Auftrag", sc.orders, matched_o, "circle", "A", "bottom center", C.COLORS["order"])):
        for on in (True, False):
            idx = [k for k in range(len(pts)) if (k in matched) == on]
            if idx:
                fig.add_trace(go.Scatter(x=[pts[k][0] for k in idx], y=[pts[k][1] for k in idx], mode="markers+text" if small else "markers", name=f"{kind} {'gehalten' if on else 'frei'}",
                                         text=[f"{prefix}{k + 1}" for k in idx] if small else None, textposition=tpos, hovertext=[f"{kind} {k + 1}" for k in idx], hoverinfo="text",
                                         marker=dict(symbol=symbol if on else symbol + "-open", size=12, color=color, line=dict(width=2, color=color))))
    for side, k, color in rings:
        pt = (sc.vehicles if side == "V" else sc.orders)[k]
        fig.add_trace(go.Scatter(x=[pt[0]], y=[pt[1]], mode="markers", hoverinfo="skip", name="Ereignis", marker=dict(symbol="circle-open", size=28, color=color, line=dict(width=3, color=color))))
    fig = _map_layout(fig, sc, height)
    fig.update_layout(showlegend=False)          # die Legende würde in schmalen Spalten die Zeichenfläche auf fast null drücken; die Farben stehen in der Erklärung unter der Karte
    return fig


def build_progress(matched_series, rejected_series, k, height=240):
    """Über die Vorschläge: vorläufig versorgte Vorschlagende (Linie) und bisherige Ablehnungen (Treppe); senkrecht der aktuelle Schritt."""
    fig = go.Figure()
    xs = list(range(len(matched_series)))
    fig.add_trace(go.Scatter(x=xs, y=matched_series, mode="lines", name="versorgt", line=dict(color=C.COLORS["propose"])))
    fig.add_trace(go.Scatter(x=xs, y=rejected_series, mode="lines", name="Ablehnungen und Verdrängungen", line=dict(color=C.COLORS["reject"], shape="hv")))
    fig.add_vline(x=k, line=dict(color="#333", width=2))
    fig.update_xaxes(title="Vorschlag")
    fig.update_yaxes(title="Anzahl", rangemode="tozero")
    fig = _base(fig, height)
    fig.update_layout(legend=dict(orientation="h", y=-0.45))
    return fig


def build_scatter(points, height=340):
    """Alle stabilen Paarungen als Punkte (Summe der Ränge der Fahrzeuge gegen die der Aufträge; 0 = immer die erste Wahl): unten links ist gut für beide Seiten.
    `points`: Liste (Name, x, y, Stil) mit Stil 'stable', 'v', 'o', 'opt', 'greedy'."""
    styles = {"stable": ("Stabile Paarungen", "circle", C.COLORS["stable_v"], 9), "v": ("Fahrzeuge schlagen vor", "star", C.COLORS["stable_v"], 16), "o": ("Aufträge schlagen vor", "star", C.COLORS["stable_o"], 16),
              "opt": ("Optimum (Ungarisch)", "diamond", C.COLORS["hung"], 13), "greedy": ("Greedy", "x", C.COLORS["greedy"], 11)}
    fig = go.Figure()
    for key, (label, symbol, color, size) in styles.items():
        pts = [p for p in points if p[3] == key]
        if pts:
            fig.add_trace(go.Scatter(x=[p[1] for p in pts], y=[p[2] for p in pts], mode="markers", name=label, text=[p[0] for p in pts], hoverinfo="text+x+y",
                                     marker=dict(symbol=symbol, size=size, color=color, line=dict(width=1, color="#222"), opacity=0.85)))
    fig.update_xaxes(title="Summe der Ränge der Fahrzeuge (kleiner = besser)")
    fig.update_yaxes(title="Summe der Ränge der Aufträge")
    fig = _base(fig, height)
    fig.update_layout(legend=dict(orientation="h", y=-0.3), height=height + 40)
    return fig


def build_premium_hist(premiums, current=None, median=None, height=280):
    """Prämie der stabilen Paarung gegenüber der billigsten Paarung mit gleicher Paarzahl [%] über die Karten; Median als Linie, Ihre Ziehung als Marke."""
    fig = go.Figure(go.Histogram(x=premiums, xbins=dict(size=5), marker_color=C.COLORS["stable_v"], opacity=0.75, name="Karten"))
    if median is not None:
        fig.add_vline(x=median, line=dict(color="#333", width=2), annotation_text=f"Median {median:.0f} %".replace(".", ","), annotation_position="top right")
    if current is not None:
        fig.add_vline(x=current, line=dict(color="#d62728", dash="dash"), annotation_text="Ihre Ziehung", annotation_position="top left")
    fig.update_xaxes(title="Mehrkosten gegenüber der billigsten Paarung gleicher Paarzahl [%]")
    fig.update_yaxes(title="Karten")
    return _base(fig, height)


def build_noise_sweep(rows, height=340):
    """Gegen die Streuung k [min]: Anteil der Karten mit mehreren stabilen Paarungen (Linie), Median der Prämie [%] (rechte Achse)."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    ks = [r["k"] for r in rows]
    fig.add_trace(go.Scatter(x=ks, y=[100 * r["multi"] for r in rows], mode="lines+markers", name="mehrere stabile Paarungen [% der Karten]", line=dict(color=C.COLORS["stable_v"])), secondary_y=False)
    fig.add_trace(go.Scatter(x=ks, y=[r["premium_median"] for r in rows], mode="lines+markers", name="Prämie (Median) [%]", line=dict(color=C.COLORS["hung"])), secondary_y=True)
    fig.update_xaxes(title="Streuung k [min] (0 = nur Entfernung)")
    fig.update_yaxes(title="Karten [%]", secondary_y=False, rangemode="tozero")
    fig.update_yaxes(title="Prämie [%]", secondary_y=True, rangemode="tozero", showgrid=False)
    fig = _base(fig, height)
    fig.update_layout(legend=dict(orientation="h", y=-0.3), height=height + 40)
    return fig


def build_count_vs_size(rows, height=320):
    """Anzahl stabiler Paarungen (Mittel und Maximum) gegen die Kartengröße n = m (alles erreichbar); der Anteil der Karten, auf denen die Wahl der Vorschlagenden etwas ausmacht, steht daneben."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    ns = [r["n"] for r in rows]
    fig.add_trace(go.Scatter(x=ns, y=[r["n_stable"] for r in rows], mode="lines+markers", name="stabile Paarungen (Mittel)", line=dict(color=C.COLORS["stable_v"])), secondary_y=False)
    fig.add_trace(go.Scatter(x=ns, y=[r["n_stable_max"] for r in rows], mode="lines+markers", name="Maximum", line=dict(color=C.COLORS["stable_v"], dash="dot")), secondary_y=False)
    fig.add_trace(go.Scatter(x=ns, y=[100 * r["vo"] for r in rows], mode="lines+markers", name="Karten mit V ≠ A [%]", line=dict(color=C.COLORS["stable_o"])), secondary_y=True)
    fig.update_xaxes(title="Fahrzeuge = Aufträge")
    fig.update_yaxes(title="stabile Paarungen", secondary_y=False, rangemode="tozero")
    fig.update_yaxes(title="Karten [%]", secondary_y=True, rangemode="tozero", showgrid=False, range=[0, 105])
    fig = _base(fig, height)
    fig.update_layout(legend=dict(orientation="h", y=-0.3), height=height + 40)
    return fig


def build_effort(rows, height=340):
    """Vorschläge gegen die Kartengröße (doppelt logarithmisch): vollständige Listen, konstanter mittlerer Grad, die Straße (Worst Case, nur bis n = 40) und n · H_n als Bezug."""
    fig = go.Figure()
    ns = [r["n"] for r in rows]
    fig.add_trace(go.Scatter(x=ns, y=[r["full"] for r in rows], mode="lines+markers", name="vollständige Listen", line=dict(color=C.COLORS["stable_v"])))
    fig.add_trace(go.Scatter(x=ns, y=[r["sparse"] for r in rows], mode="lines+markers", name="konstanter mittlerer Grad", line=dict(color=C.COLORS["stable_o"])))
    road = [r for r in rows if r["road"] is not None]
    fig.add_trace(go.Scatter(x=[r["n"] for r in road], y=[r["road"] for r in road], mode="lines+markers", name="Straße (Worst Case)", line=dict(color=C.COLORS["hung"])))
    fig.add_trace(go.Scatter(x=ns, y=[r["nHn"] for r in rows], mode="lines", name="n · H_n", line=dict(color="#555", dash="dot")))
    fig.update_xaxes(title="Fahrzeuge = Aufträge", type="log")
    fig.update_yaxes(title="Vorschläge", type="log")
    fig = _base(fig, height)
    fig.update_layout(legend=dict(orientation="h", y=-0.3), height=height + 40)
    return fig
