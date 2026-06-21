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


def _fd_gradient(f: Callable[[np.ndarray], float], x: np.ndarray, eps: float = 1e-6
                 ) -> np.ndarray:
    """Central finite-difference gradient (cheap: tiny Hilbert space)."""
    g = np.zeros_like(x)
    for i in range(len(x)):
        xp = x.copy(); xp[i] += eps
        xm = x.copy(); xm[i] -= eps
        g[i] = (f(xp) - f(xm)) / (2 * eps)
    return g


def adam_maximize(fidelity: Callable[[np.ndarray], float], x0: np.ndarray,
                  steps: int = 600, lr: float = 0.1, tol: float = 1e-10
                  ) -> Tuple[np.ndarray, float]:
    """Maximise ``fidelity`` over a vector via Adam on a finite-difference gradient."""
    x = np.array(x0, dtype=float)
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    best_x, best_f = x.copy(), fidelity(x)
    for t in range(1, steps + 1):
        g = _fd_gradient(fidelity, x)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        mh = m / (1 - b1 ** t)
        vh = v / (1 - b2 ** t)
        x = x + lr * mh / (np.sqrt(vh) + eps)
        f = fidelity(x)
        if f > best_f:
            best_f, best_x = f, x.copy()
        if 1.0 - best_f < tol:
            break
    return best_x, best_f


def optimize_areas_adam(edges: Sequence[Tuple[int, int]], num_qubits: int,
                        target: np.ndarray, steps: int = 600, restarts: int = 24,
                        seed: int = 0, lr: float = 0.1
                        ) -> Tuple[List[float], float]:
    """Gradient-based area optimisation for a fixed edge pattern (robust in high-D).

    Returns (areas mod 2*pi, fidelity).  Use this instead of :func:`optimize_areas`
    when the sequence has many pulses (e.g. a 19-pulse CNOT).
    """
    rng = np.random.default_rng(seed)
    n = len(edges)

    def fid(a):
        return sequence_fidelity(a, edges, num_qubits, target)

    best_x, best_f = None, -np.inf
    for _ in range(restarts):
        x0 = rng.uniform(0, 2 * np.pi, size=n)
        x, f = adam_maximize(fid, x0, steps=steps, lr=lr)
        if f > best_f:
            best_x, best_f = x, f
        if 1.0 - best_f < 1e-9:
            break
    areas = [float(v % (2 * np.pi)) for v in best_x]
    return areas, float(best_f)


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
