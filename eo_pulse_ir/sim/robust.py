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


# Canonical 2-logical-qubit role -> dot-edge map (control = positions 0, target
# = position 1), matching eo_pulse_ir.compile._resolve_role for ctrl_pos=0,
# tgt_pos=1.  A robust gate optimised on this isolated 6-dot block transfers to
# any adjacent pair, because each 2-qubit gate is a local block in the IR.
_ROLE_EDGE = {
    "ctrl_low": (0, 1), "ctrl_high": (1, 2), "inter": (2, 3),
    "tgt_low": (3, 4), "tgt_high": (4, 5),
}


def roles_to_edges(roles: Sequence[str]) -> List[Edge]:
    """Map template role names to canonical 2-qubit dot edges."""
    return [_ROLE_EDGE[r] for r in roles]


def robust_gate_library(sigma: float, spread: float,
                        gate_names: Sequence[str] = ("cx", "swap"),
                        steps: int = 200, n_train: int = 12,
                        seed: int = 1) -> dict:
    """Build a valley+charge-robust gate library for a device profile.

    For each gate, takes the built-in template's *role sequence* (and uses its
    areas as the warm-start x0), then re-optimises the areas for the joint
    ensemble (valley-phase spread ``±spread·π`` and charge noise ``sigma``).
    Returns ``{gate_name: [(role, area), ...]}`` — the role sequence is
    unchanged, so it is a drop-in ``gate_library`` for
    :func:`eo_pulse_ir.compile.synthesize` / ``compile_circuit``.

    This is the seam that lets the (1-D or 2-D) place-and-route run on
    *device-calibrated, valley-robust* pulse costs instead of nominal templates.
    """
    from ..native import two_qubit_template
    from . import gates as _gate_targets

    targets = {"cx": _gate_targets.CNOT, "cnot": _gate_targets.CNOT,
               "swap": _gate_targets.SWAP, "cxswap": _gate_targets.CXSWAP}
    delta = spread * np.pi
    lib: dict = {}
    for name in gate_names:
        template = two_qubit_template(name)
        roles = [r for r, _ in template]
        x0 = np.array([a for _, a in template], float)
        edges = roles_to_edges(roles)
        train = joint_valley_noise_samples(edges, delta, sigma, n_train, seed=seed)
        areas, _F = robust_design(edges, 2, targets[name], train,
                                  x0=x0, steps=steps, seed=seed)
        lib[name] = list(zip(roles, [float(a) for a in areas]))
    return lib


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
