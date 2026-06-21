"""Pulse-parameter -> performance landscape: sweeps, robustness, SVG heatmaps.

This is the research-facing layer: it maps a 2-D slice of pulse parameters
(e.g. a boundary exchange area vs an intra-qubit area) to fidelity / leakage and
to a *robustness* map (the local gradient magnitude), so one can read off

- fidelity cliffs (high-gradient ridges),
- leakage basins (where exchange drives population out of S=1/2 (x) S=1/2),
- noise-robust flat regions (low-gradient plateaus = real noise insensitivity).

Heatmaps are emitted as dependency-free SVG so the whole pipeline stays light.
"""

from __future__ import annotations

import json
from typing import Callable, Dict, List, Sequence

import numpy as np


def sweep2d(func: Callable[[float, float], Dict[str, float]],
            xs: Sequence[float], ys: Sequence[float]) -> Dict[str, np.ndarray]:
    """Evaluate ``func(x, y) -> {metric: value}`` over the grid (ys rows, xs cols)."""
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    grids: Dict[str, np.ndarray] = {}
    for r, y in enumerate(ys):
        for c, x in enumerate(xs):
            res = func(float(x), float(y))
            for k, v in res.items():
                if k not in grids:
                    grids[k] = np.full((len(ys), len(xs)), np.nan)
                grids[k][r, c] = v
    return grids


def gradient_magnitude(grid: np.ndarray, xs: Sequence[float], ys: Sequence[float]
                       ) -> np.ndarray:
    """|grad| of a metric over the grid: a robustness / noise-sensitivity map.

    Small values = flat region = insensitive to pulse-parameter (charge-noise)
    fluctuations; large values = a cliff.
    """
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    gy, gx = np.gradient(grid, ys, xs)
    return np.sqrt(gx ** 2 + gy ** 2)


# ---- minimal perceptual colormap (viridis-like) -------------------------------
_VIRIDIS = [
    (0.267, 0.005, 0.329), (0.283, 0.141, 0.458), (0.254, 0.265, 0.530),
    (0.207, 0.372, 0.553), (0.164, 0.471, 0.558), (0.128, 0.567, 0.551),
    (0.135, 0.659, 0.518), (0.267, 0.749, 0.441), (0.478, 0.821, 0.318),
    (0.741, 0.873, 0.150), (0.993, 0.906, 0.144),
]


def _color(t: float) -> str:
    t = 0.0 if np.isnan(t) else min(1.0, max(0.0, t))
    pos = t * (len(_VIRIDIS) - 1)
    i = int(pos)
    frac = pos - i
    if i >= len(_VIRIDIS) - 1:
        r, g, b = _VIRIDIS[-1]
    else:
        c0, c1 = _VIRIDIS[i], _VIRIDIS[i + 1]
        r, g, b = (c0[k] + frac * (c1[k] - c0[k]) for k in range(3))
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"


def heatmap_svg(grid: np.ndarray, xs: Sequence[float], ys: Sequence[float],
                title: str = "", xlabel: str = "x", ylabel: str = "y",
                value_label: str = "value", vmin: float = None, vmax: float = None
                ) -> str:
    """Render a 2-D grid as an SVG heatmap (rows = ys ascending upward)."""
    grid = np.asarray(grid, float)
    nrows, ncols = grid.shape
    vmin = float(np.nanmin(grid)) if vmin is None else vmin
    vmax = float(np.nanmax(grid)) if vmax is None else vmax
    span = (vmax - vmin) or 1.0

    cell = 26
    pad_l, pad_t, pad_b, pad_r = 70, 46, 56, 96
    w = pad_l + ncols * cell + pad_r
    h = pad_t + nrows * cell + pad_b

    p: List[str] = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{w}' height='{h}' "
        f"font-family='monospace' font-size='11'>",
        f"<rect width='{w}' height='{h}' fill='white'/>",
        f"<text x='12' y='24' font-size='15' font-weight='bold'>{_esc(title)}</text>",
    ]
    # cells: row 0 of grid drawn at the BOTTOM so y increases upward
    for r in range(nrows):
        for c in range(ncols):
            val = grid[r, c]
            t = (val - vmin) / span
            x = pad_l + c * cell
            y = pad_t + (nrows - 1 - r) * cell
            p.append(f"<rect x='{x}' y='{y}' width='{cell}' height='{cell}' "
                     f"fill='{_color(t)}'/>")
    # axes labels (sparse ticks)
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    for c in range(0, ncols, max(1, ncols // 6)):
        x = pad_l + c * cell + cell / 2
        p.append(f"<text x='{x:.1f}' y='{pad_t + nrows*cell + 16}' "
                 f"text-anchor='middle' fill='#555'>{xs[c]:.2g}</text>")
    for r in range(0, nrows, max(1, nrows // 6)):
        y = pad_t + (nrows - 1 - r) * cell + cell / 2 + 4
        p.append(f"<text x='{pad_l - 8}' y='{y:.1f}' text-anchor='end' "
                 f"fill='#555'>{ys[r]:.2g}</text>")
    p.append(f"<text x='{pad_l + ncols*cell/2:.0f}' y='{h-8}' text-anchor='middle' "
             f"fill='#333'>{_esc(xlabel)}</text>")
    p.append(f"<text x='16' y='{pad_t + nrows*cell/2:.0f}' fill='#333' "
             f"transform='rotate(-90 16 {pad_t + nrows*cell/2:.0f})' "
             f"text-anchor='middle'>{_esc(ylabel)}</text>")

    # colorbar
    cb_x = pad_l + ncols * cell + 24
    cb_h = nrows * cell
    steps = 64
    for k in range(steps):
        t = k / (steps - 1)
        y = pad_t + (1 - t) * cb_h - cb_h / steps
        p.append(f"<rect x='{cb_x}' y='{y:.1f}' width='14' height='{cb_h/steps+1:.1f}' "
                 f"fill='{_color(t)}'/>")
    p.append(f"<text x='{cb_x+18}' y='{pad_t+6}' fill='#555'>{vmax:.3g}</text>")
    p.append(f"<text x='{cb_x+18}' y='{pad_t+cb_h}' fill='#555'>{vmin:.3g}</text>")
    p.append(f"<text x='{cb_x}' y='{pad_t-8}' fill='#333'>{_esc(value_label)}</text>")
    p.append("</svg>")
    return "\n".join(p)


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def save_grids_json(grids: Dict[str, np.ndarray], xs: Sequence[float],
                    ys: Sequence[float], path: str) -> None:
    payload = {
        "xs": list(map(float, xs)),
        "ys": list(map(float, ys)),
        "grids": {k: np.asarray(v).tolist() for k, v in grids.items()},
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def write_svg(svg: str, path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(svg)


_SERIES_COLORS = ["#4C78A8", "#E45756", "#54A24B", "#F58518", "#B279A2"]


def line_plot_svg(series, title="", xlabel="x", ylabel="y", logx=False, logy=False,
                  width=720, height=440):
    """Render labelled line series ``[(name, xs, ys), ...]`` as an SVG plot."""
    pad_l, pad_t, pad_r, pad_b = 70, 44, 150, 56
    pw, ph = width - pad_l - pad_r, height - pad_t - pad_b

    def tx(v):
        return np.log10(v) if logx else v

    def ty(v):
        return np.log10(max(v, 1e-18)) if logy else v

    allx = [tx(x) for _, xs, _ in series for x in xs]
    ally = [ty(y) for _, _, ys in series for y in ys]
    xmin, xmax = min(allx), max(allx)
    ymin, ymax = min(ally), max(ally)
    xs_span = (xmax - xmin) or 1.0
    ys_span = (ymax - ymin) or 1.0

    def px(v):
        return pad_l + (tx(v) - xmin) / xs_span * pw

    def py(v):
        return pad_t + ph - (ty(v) - ymin) / ys_span * ph

    p = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' "
         f"font-family='monospace' font-size='11'>",
         f"<rect width='{width}' height='{height}' fill='white'/>",
         f"<text x='12' y='24' font-size='15' font-weight='bold'>{_esc(title)}</text>",
         f"<rect x='{pad_l}' y='{pad_t}' width='{pw}' height='{ph}' fill='none' "
         f"stroke='#ccc'/>"]
    # gridline ticks (5 each)
    for k in range(6):
        gx = xmin + xs_span * k / 5
        x = pad_l + (gx - xmin) / xs_span * pw
        lab = (10 ** gx) if logx else gx
        p.append(f"<line x1='{x:.1f}' y1='{pad_t}' x2='{x:.1f}' y2='{pad_t+ph}' "
                 f"stroke='#f0f0f0'/>")
        p.append(f"<text x='{x:.1f}' y='{pad_t+ph+16}' text-anchor='middle' "
                 f"fill='#666'>{lab:.3g}</text>")
        gy = ymin + ys_span * k / 5
        y = pad_t + ph - (gy - ymin) / ys_span * ph
        lab = (10 ** gy) if logy else gy
        p.append(f"<line x1='{pad_l}' y1='{y:.1f}' x2='{pad_l+pw}' y2='{y:.1f}' "
                 f"stroke='#f0f0f0'/>")
        p.append(f"<text x='{pad_l-6}' y='{y+4:.1f}' text-anchor='end' "
                 f"fill='#666'>{lab:.2g}</text>")
    for i, (name, xs, ys) in enumerate(series):
        c = _SERIES_COLORS[i % len(_SERIES_COLORS)]
        pts = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in zip(xs, ys))
        p.append(f"<polyline points='{pts}' fill='none' stroke='{c}' stroke-width='2'/>")
        for x, y in zip(xs, ys):
            p.append(f"<circle cx='{px(x):.1f}' cy='{py(y):.1f}' r='2.5' fill='{c}'/>")
        ly = pad_t + 6 + i * 18
        p.append(f"<rect x='{pad_l+pw+14}' y='{ly-9}' width='12' height='12' fill='{c}'/>")
        p.append(f"<text x='{pad_l+pw+30}' y='{ly+1}' fill='#333'>{_esc(name)}</text>")
    p.append(f"<text x='{pad_l+pw/2:.0f}' y='{height-12}' text-anchor='middle' "
             f"fill='#333'>{_esc(xlabel)}</text>")
    p.append(f"<text x='16' y='{pad_t+ph/2:.0f}' fill='#333' "
             f"transform='rotate(-90 16 {pad_t+ph/2:.0f})' text-anchor='middle'>"
             f"{_esc(ylabel)}</text>")
    p.append("</svg>")
    return "\n".join(p)


def histogram_svg(samples, bins=40, title="", xlabel="fidelity", width=720, height=400):
    """Render a histogram of ``samples`` as SVG."""
    samples = np.asarray(samples, float)
    pad_l, pad_t, pad_r, pad_b = 60, 44, 24, 52
    pw, ph = width - pad_l - pad_r, height - pad_t - pad_b
    counts, edges = np.histogram(samples, bins=bins)
    cmax = counts.max() or 1
    xmin, xmax = edges[0], edges[-1]
    xspan = (xmax - xmin) or 1.0
    p = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' "
         f"font-family='monospace' font-size='11'>",
         f"<rect width='{width}' height='{height}' fill='white'/>",
         f"<text x='12' y='24' font-size='15' font-weight='bold'>{_esc(title)}</text>"]
    bw = pw / len(counts)
    for i, c in enumerate(counts):
        h = ph * c / cmax
        x = pad_l + i * bw
        y = pad_t + ph - h
        p.append(f"<rect x='{x:.1f}' y='{y:.1f}' width='{bw+0.5:.1f}' height='{h:.1f}' "
                 f"fill='#4C78A8' fill-opacity='0.85'/>")
    mean = float(samples.mean())
    mx = pad_l + (mean - xmin) / xspan * pw
    p.append(f"<line x1='{mx:.1f}' y1='{pad_t}' x2='{mx:.1f}' y2='{pad_t+ph}' "
             f"stroke='#E45756' stroke-width='1.5' stroke-dasharray='4 3'/>")
    p.append(f"<text x='{mx:.1f}' y='{pad_t-4}' text-anchor='middle' fill='#E45756'>"
             f"mean {mean:.5f}</text>")
    for k in range(6):
        gx = xmin + xspan * k / 5
        x = pad_l + (gx - xmin) / xspan * pw
        p.append(f"<text x='{x:.1f}' y='{pad_t+ph+16}' text-anchor='middle' "
                 f"fill='#666'>{gx:.4g}</text>")
    p.append(f"<text x='{pad_l+pw/2:.0f}' y='{height-10}' text-anchor='middle' "
             f"fill='#333'>{_esc(xlabel)}</text>")
    p.append("</svg>")
    return "\n".join(p)
