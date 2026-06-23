"""Pulse-timeline visualisation (dependency-free SVG) and JSON export.

One horizontal lane per exchange edge J_{i,i+1}; time runs left to right; each
pulse is a rectangle whose height encodes amplitude and whose colour encodes its
role (intra / inter / route).  Pure standard library so it always renders, in CI
or anywhere, with no numpy/matplotlib needed.
"""

from __future__ import annotations

import json
from typing import List, Tuple

from .hardware import edge_name
from .schedule import Schedule

_ROLE_COLOR = {
    "intra": "#4C78A8",
    "inter": "#E45756",
    "route": "#F58518",
}


def _xml_escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def timeline_records(schedule: Schedule) -> List[dict]:
    out = []
    for p in sorted(schedule.pulses, key=lambda x: (x.start or 0.0, x.edge)):
        out.append({
            "edge": list(p.edge),
            "output": edge_name(p.edge),
            "gate": p.gate,
            "role": p.role,
            "logical_qubits": list(p.logical_qubits),
            "start": round(p.start or 0.0, 6),
            "duration": round(p.duration or 0.0, 6),
            "area": round(p.area, 6),
            "j": round(p.j or 0.0, 6),
        })
    return out


def write_timeline_json(schedule: Schedule, path: str) -> None:
    payload = {
        "num_dots": schedule.num_dots,
        "makespan": round(schedule.makespan, 6),
        "j_max": schedule.j_max,
        "pulse_count": len(schedule.pulses),
        "pulses": timeline_records(schedule),
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def render_svg(schedule: Schedule, title: str = "EO pulse timeline") -> str:
    edges = schedule.edges()
    if not edges:
        return "<svg xmlns='http://www.w3.org/2000/svg' width='400' height='60'></svg>"

    lane_h = 34
    pad_left = 90
    pad_top = 46
    pad_bottom = 40
    width_px = 900
    plot_w = width_px - pad_left - 30
    height_px = pad_top + lane_h * len(edges) + pad_bottom

    makespan = schedule.makespan or 1.0
    sx = plot_w / makespan
    j_max = schedule.j_max or 1.0

    edge_y = {e: pad_top + i * lane_h for i, e in enumerate(edges)}
    parts: List[str] = []
    parts.append(
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width_px}' height='{height_px}' "
        f"font-family='monospace' font-size='11'>"
    )
    parts.append(f"<rect width='{width_px}' height='{height_px}' fill='white'/>")
    parts.append(f"<text x='12' y='22' font-size='15' font-weight='bold'>{_xml_escape(title)}</text>")
    parts.append(
        f"<text x='12' y='38' fill='#555'>pulses={len(schedule.pulses)}  "
        f"makespan={makespan:.3g}  edges={len(edges)}</text>"
    )

    # lanes + labels
    for e in edges:
        y = edge_y[e]
        parts.append(
            f"<line x1='{pad_left}' y1='{y + lane_h - 6}' x2='{pad_left + plot_w}' "
            f"y2='{y + lane_h - 6}' stroke='#ddd' stroke-width='1'/>"
        )
        parts.append(
            f"<text x='{pad_left - 8}' y='{y + lane_h - 9}' text-anchor='end' "
            f"fill='#333'>{edge_name(e)}</text>"
        )

    # time axis ticks
    n_ticks = 6
    for k in range(n_ticks + 1):
        t = makespan * k / n_ticks
        x = pad_left + t * sx
        parts.append(
            f"<line x1='{x:.1f}' y1='{pad_top - 6}' x2='{x:.1f}' "
            f"y2='{height_px - pad_bottom + 4}' stroke='#f0f0f0' stroke-width='1'/>"
        )
        parts.append(
            f"<text x='{x:.1f}' y='{height_px - pad_bottom + 18}' text-anchor='middle' "
            f"fill='#777'>{t:.2g}</text>"
        )
    parts.append(
        f"<text x='{pad_left + plot_w / 2:.0f}' y='{height_px - 6}' text-anchor='middle' "
        f"fill='#555'>time (ns)</text>"
    )

    # pulses
    for p in schedule.pulses:
        y0 = edge_y[p.edge]
        x = pad_left + (p.start or 0.0) * sx
        w = max(1.0, (p.duration or 0.0) * sx)
        amp = (p.j or j_max) / j_max
        h = max(4.0, (lane_h - 12) * amp)
        ry = y0 + (lane_h - 6) - h
        color = _ROLE_COLOR.get(p.role, "#999999")
        parts.append(
            f"<rect x='{x:.1f}' y='{ry:.1f}' width='{w:.1f}' height='{h:.1f}' "
            f"fill='{color}' fill-opacity='0.85' stroke='{color}' stroke-width='0.5'>"
            f"<title>{_xml_escape(p.gate)} {p.role} area={p.area:.3g} "
            f"t=[{(p.start or 0.0):.3g},{p.end:.3g}]</title></rect>"
        )

    # legend
    lx = pad_left
    ly = height_px - pad_bottom + 30
    for i, (role, color) in enumerate(_ROLE_COLOR.items()):
        x = lx + i * 150
        parts.append(f"<rect x='{x}' y='{ly - 9}' width='11' height='11' fill='{color}'/>")
        parts.append(f"<text x='{x + 15}' y='{ly}' fill='#333'>{role}</text>")

    parts.append("</svg>")
    return "\n".join(parts)


def write_svg(schedule: Schedule, path: str, title: str = "EO pulse timeline") -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(render_svg(schedule, title))
