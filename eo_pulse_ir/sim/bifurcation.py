"""Bifurcation analysis of the exchange-only control landscape.

The pulse-control problem is a nonlinear map from pulse parameters to a
performance scalar (gate fidelity).  As a physical parameter is varied (a field
gradient, an exchange/valley suppression, a drift), the *critical points* of that
landscape — the optima the optimiser can converge to — are created, destroyed and
merged.  Those are bifurcations (mostly saddle-node) of the control landscape,
and counting them measures how rugged / complex the control problem becomes.

This module provides critical-point detection on a (periodic) 2-D landscape grid
and a sweep that tracks the optima vs a parameter, yielding a bifurcation diagram.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np


def _neighbors(i: int, j: int, n: int, m: int, periodic: bool):
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            ii, jj = i + di, j + dj
            if periodic:
                yield ii % n, jj % m
            elif 0 <= ii < n and 0 <= jj < m:
                yield ii, jj


def local_maxima(G: np.ndarray, periodic: bool = True) -> List[Tuple[int, int, float]]:
    """Return [(i, j, value)] for strict local maxima of grid ``G``."""
    G = np.asarray(G, float)
    n, m = G.shape
    out = []
    for i in range(n):
        for j in range(m):
            v = G[i, j]
            if all(G[a, b] <= v for a, b in _neighbors(i, j, n, m, periodic)) and \
               any(G[a, b] < v for a, b in _neighbors(i, j, n, m, periodic)):
                out.append((i, j, float(v)))
    return out


def classify_critical(G: np.ndarray, i: int, j: int, periodic: bool = True) -> str:
    """Classify a grid point via the discrete Hessian: max / min / saddle / flat."""
    n, m = G.shape

    def at(a, b):
        if periodic:
            return G[a % n, b % m]
        a = min(max(a, 0), n - 1); b = min(max(b, 0), m - 1)
        return G[a, b]

    fxx = at(i + 1, j) - 2 * G[i, j] + at(i - 1, j)
    fyy = at(i, j + 1) - 2 * G[i, j] + at(i, j - 1)
    fxy = (at(i + 1, j + 1) - at(i + 1, j - 1) - at(i - 1, j + 1) + at(i - 1, j - 1)) / 4
    det = fxx * fyy - fxy * fxy
    if abs(det) < 1e-14:
        return "flat"
    if det < 0:
        return "saddle"
    return "max" if fxx < 0 else "min"


@dataclass
class BifurcationPoint:
    param: float
    num_maxima: int
    global_max: float
    maxima: List[Tuple[float, float, float]]   # (x, y, value)


def optima_sweep(grid_fn: Callable[[float], np.ndarray], params: Sequence[float],
                 xs: Sequence[float], ys: Sequence[float], periodic: bool = True,
                 threshold: float = None) -> List[BifurcationPoint]:
    """Track local maxima of ``grid_fn(param)`` over a sweep of ``param``.

    ``grid_fn(param)`` returns a 2-D landscape on (ys, xs).  Returns a
    bifurcation point per param (count + locations of optima); a change in the
    count across the sweep marks a saddle-node bifurcation.
    """
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    out: List[BifurcationPoint] = []
    for p in params:
        G = grid_fn(p)
        mx = local_maxima(G, periodic)
        if threshold is not None:
            mx = [t for t in mx if t[2] >= threshold]
        locs = [(float(xs[j]), float(ys[i]), v) for (i, j, v) in mx]
        out.append(BifurcationPoint(float(p), len(mx),
                                    float(G.max()) if G.size else 0.0, locs))
    return out
