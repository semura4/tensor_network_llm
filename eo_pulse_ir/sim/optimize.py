"""Find exchange-pulse areas that realise a target gate (Nelder-Mead, numpy-only).

This is what turns the IR's *placeholder* template areas into *validated* ones:
fix the edge pattern of a sequence, then optimise the per-pulse areas to maximise
the simulated gate fidelity.  No scipy dependency.
"""

from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

import numpy as np

from .fidelity import average_gate_fidelity
from .simulator import logical_block


def sequence_fidelity(areas: Sequence[float], edges: Sequence[Tuple[int, int]],
                      num_qubits: int, target: np.ndarray) -> float:
    pulses = list(zip(edges, areas))
    M = logical_block(pulses, num_qubits)
    return average_gate_fidelity(M, target)


def nelder_mead(f: Callable[[np.ndarray], float], x0: np.ndarray,
                step: float = 0.6, max_iter: int = 2000, tol: float = 1e-12
                ) -> Tuple[np.ndarray, float]:
    """Minimise scalar ``f`` over a vector argument. Returns (x_best, f_best)."""
    n = len(x0)
    simplex = [np.array(x0, dtype=float)]
    for i in range(n):
        p = np.array(x0, dtype=float)
        p[i] += step
        simplex.append(p)
    fvals = [f(p) for p in simplex]

    a, g, r, s = 1.0, 2.0, 0.5, 0.5  # reflection, expansion, contraction, shrink
    for _ in range(max_iter):
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        if abs(fvals[-1] - fvals[0]) < tol:
            break
        centroid = np.mean(simplex[:-1], axis=0)
        xr = centroid + a * (centroid - simplex[-1])
        fr = f(xr)
        if fvals[0] <= fr < fvals[-2]:
            simplex[-1], fvals[-1] = xr, fr
            continue
        if fr < fvals[0]:
            xe = centroid + g * (xr - centroid)
            fe = f(xe)
            if fe < fr:
                simplex[-1], fvals[-1] = xe, fe
            else:
                simplex[-1], fvals[-1] = xr, fr
            continue
        xc = centroid + r * (simplex[-1] - centroid)
        fc = f(xc)
        if fc < fvals[-1]:
            simplex[-1], fvals[-1] = xc, fc
            continue
        for i in range(1, len(simplex)):
            simplex[i] = simplex[0] + s * (simplex[i] - simplex[0])
            fvals[i] = f(simplex[i])
    order = np.argsort(fvals)
    return simplex[order[0]], fvals[order[0]]


def optimize_areas(edges: Sequence[Tuple[int, int]], num_qubits: int,
                   target: np.ndarray, restarts: int = 12, seed: int = 0
                   ) -> Tuple[List[float], float]:
    """Optimise pulse areas for a fixed edge pattern; return (areas, fidelity).

    Uses several random restarts to dodge local optima; the infidelity 1-F is
    minimised.
    """
    rng = np.random.default_rng(seed)
    n = len(edges)
    best_x, best_inf = None, np.inf
    for _ in range(restarts):
        x0 = rng.uniform(0, 2 * np.pi, size=n)
        x, inf = nelder_mead(
            lambda a: 1.0 - sequence_fidelity(a, edges, num_qubits, target), x0)
        if inf < best_inf:
            best_x, best_inf = x, inf
    areas = [float(v % (2 * np.pi)) for v in best_x]
    return areas, 1.0 - best_inf
