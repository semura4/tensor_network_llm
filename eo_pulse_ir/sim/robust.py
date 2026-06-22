"""Robust optimal control: design pulses that are insensitive to a parameter spread.

Standard GRAPE maximises fidelity at one nominal operating point. *Robust* design
maximises the **ensemble-averaged** fidelity over a distribution of perturbations
— here the valley-phase mismatch Δφ (which scales the boundary-exchange areas by
cos²(Δφ/2)) and/or charge-noise area fluctuations. The result is a gate that
trades a little peak fidelity for a much flatter response — exactly what the
valley-uniformity and charge-noise problems call for.

Each ensemble sample is an affine area transform ``a_eff = scale * a + offset``;
the ensemble gradient chains through it: grad_a = mean_k(grad_eff(a_eff_k) * scale_k).
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from .optimize import fidelity_and_grad
from .simulator import logical_block
from .fidelity import average_gate_fidelity

Edge = Tuple[int, int]


def is_inter_edge(edge: Edge) -> bool:
    """Boundary (inter-triple) edge: bridges dot 3k+2 and 3k+3."""
    return edge[0] % 3 == 2


def valley_phase_samples(edges: Sequence[Edge], dphis: Sequence[float]
                         ) -> List[np.ndarray]:
    """Per-sample area scale vectors: boundary areas * cos^2(Δφ/2)."""
    inter = np.array([is_inter_edge(tuple(e)) for e in edges])
    samples = []
    for d in dphis:
        s = np.ones(len(edges))
        s[inter] = np.cos(d / 2) ** 2
        samples.append(s)
    return samples


def charge_noise_samples(n_edges: int, sigma: float, n_samples: int,
                         seed: int = 0) -> List[np.ndarray]:
    """Per-sample multiplicative area noise scale vectors (1 + eps), eps~N(0,sigma)."""
    rng = np.random.default_rng(seed)
    return [1.0 + rng.normal(0.0, sigma, size=n_edges) for _ in range(n_samples)]


def joint_valley_noise_samples(edges: Sequence[Edge], delta: float, sigma: float,
                               n_samples: int, seed: int = 0) -> List[np.ndarray]:
    """Joint ensemble: per sample, a valley-phase mismatch Δφ ~ U[-delta, delta]
    (boundary areas * cos^2(Δφ/2)) AND per-edge charge noise (1 + N(0, sigma))."""
    rng = np.random.default_rng(seed)
    inter = np.array([is_inter_edge(tuple(e)) for e in edges])
    out = []
    for _ in range(n_samples):
        s = np.ones(len(edges))
        d = rng.uniform(-delta, delta)
        s[inter] = np.cos(d / 2) ** 2
        s = s * (1.0 + rng.normal(0.0, sigma, size=len(edges)))
        out.append(s)
    return out


def ensemble_fidelity(areas, edges, num_qubits, target,
                      scales: Sequence[np.ndarray]) -> float:
    areas = np.asarray(areas, float)
    fs = []
    for s in scales:
        M = logical_block(list(zip(edges, areas * s)), num_qubits)
        fs.append(average_gate_fidelity(M, target))
    return float(np.mean(fs))


def ensemble_fidelity_and_grad(areas, edges, num_qubits, target,
                               scales: Sequence[np.ndarray]):
    """Mean fidelity and its gradient w.r.t. base areas over the ensemble."""
    areas = np.asarray(areas, float)
    F = 0.0
    g = np.zeros(len(edges))
    for s in scales:
        f, ge = fidelity_and_grad(areas * s, edges, num_qubits, target)
        F += f
        g += ge * s                       # chain rule through a_eff = s * a
    n = len(scales)
    return F / n, g / n


def robust_design(edges, num_qubits, target, scales: Sequence[np.ndarray],
                  x0=None, steps: int = 300, restarts: int = 1, lr: float = 0.05,
                  seed: int = 0) -> Tuple[List[float], float]:
    """Adam ascent on the ensemble-averaged fidelity. Returns (areas, ensemble F)."""
    rng = np.random.default_rng(seed)
    n = len(edges)
    best_x, best_f = None, -1.0
    for r in range(restarts):
        x = (np.asarray(x0, float).copy() if (x0 is not None and r == 0)
             else rng.uniform(0, 2 * np.pi, size=n))
        m = np.zeros(n); v = np.zeros(n); b1, b2, eps = 0.9, 0.999, 1e-8
        bx, bf = x.copy(), ensemble_fidelity(x, edges, num_qubits, target, scales)
        for t in range(1, steps + 1):
            _, gr = ensemble_fidelity_and_grad(x, edges, num_qubits, target, scales)
            m = b1 * m + (1 - b1) * gr
            v = b2 * v + (1 - b2) * gr * gr
            x = x + lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)
            fe = ensemble_fidelity(x, edges, num_qubits, target, scales)
            if fe > bf:
                bf, bx = fe, x.copy()
        if bf > best_f:
            best_f, best_x = bf, bx
    return [float(a % (2 * np.pi)) for a in best_x], float(best_f)
