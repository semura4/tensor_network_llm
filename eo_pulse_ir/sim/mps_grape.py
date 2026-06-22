"""MPS-GRAPE: optimal control on a tensor network.

Optimal pulse design (GRAPE) for *state preparation*, run on a matrix-product
state instead of a dense 2**(3n) vector.  This is where the control layer and the
tensor-network layer meet: gate fidelity needs all 2**nq logical columns
(exponential), but the *state-transfer* objective

    J(theta) = |<target | U(theta) | init>|^2

needs only one evolved state, so it scales to many logical qubits when the target
is not too entangled (e.g. GHZ, with bond dimension 2).

The gradient is computed by the adjoint (back-propagation) method on MPSs — one
forward sweep storing the evolved states, one backward sweep storing the
co-states, then a local two-site operator overlap per pulse:

    dJ/dtheta_k = 2 Re( conj(c) · <phi_{k+1}| (-i S.S_k) |psi_{k+1}> ) / norm,

verified against finite differences to ~1e-10.  O(N) per gradient (vs O(N^2) for
finite differences), so it is practical for deep sequences.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from .mps import MPS, _two_site_exchange, two_site_sdots

Edge = Tuple[int, int]
_HSS = two_site_sdots()


def ghz_target(num_qubits: int, chi_max: int = 8) -> MPS:
    """Encoded GHZ state (|0...0>_L + |1...1>_L)/sqrt(2) as a compressed MPS."""
    a = MPS.logical_register([0] * num_qubits)
    b = MPS.logical_register([1] * num_qubits)
    return a.add(b).compress(chi_max=chi_max).normalized()


def evolve(init: MPS, areas: Sequence[float], edges: Sequence[Edge],
           chi_max: int, tol: float = 1e-12) -> MPS:
    psi = init.copy()
    for (e, a) in zip(edges, areas):
        psi.apply_two_site(_two_site_exchange(a), e[0], chi_max, tol)
    return psi


def state_prep_fidelity(init: MPS, target: MPS, areas: Sequence[float],
                        edges: Sequence[Edge], chi_max: int) -> float:
    psi = evolve(init, areas, edges, chi_max)
    c = psi.overlap(target)
    return float(abs(c) ** 2 / (psi.overlap(psi).real * target.overlap(target).real))


def fidelity_and_grad(init: MPS, target: MPS, areas: Sequence[float],
                      edges: Sequence[Edge], chi_max: int, tol: float = 1e-12):
    """State-prep fidelity and its adjoint gradient w.r.t. each pulse area."""
    N = len(edges)
    # forward states psi[k] = (G_{k-1}...G_0) init ; psi[0] = init
    psi = [init.copy()]
    for (e, a) in zip(edges, areas):
        p = psi[-1].copy()
        p.apply_two_site(_two_site_exchange(a), e[0], chi_max, tol)
        psi.append(p)
    c = psi[-1].overlap(target)
    tnorm = target.overlap(target).real
    pnorm = psi[-1].overlap(psi[-1]).real
    fid = float(abs(c) ** 2 / (pnorm * tnorm))

    # backward co-states phi[k] = (G_k^dag ... G_{N-1}^dag) target ; phi[N] = target
    phi = [None] * (N + 1)
    phi[N] = target.copy()
    for k in range(N - 1, -1, -1):
        p = phi[k + 1].copy()
        e, a = edges[k], areas[k]
        p.apply_two_site(_two_site_exchange(-a), e[0], chi_max, tol)
        phi[k] = p

    grad = np.zeros(N)
    denom = pnorm * tnorm
    for k in range(N):
        dc = -1j * psi[k + 1].local_operator_overlap(phi[k + 1], _HSS, edges[k][0])
        grad[k] = 2.0 * np.real(np.conj(c) * dc) / denom
    return fid, grad


def grape_state_prep(init: MPS, target: MPS, edges: Sequence[Edge],
                     chi_max: int = 32, steps: int = 400, restarts: int = 6,
                     lr: float = 0.15, seed: int = 0, tol: float = 1e-9
                     ) -> Tuple[List[float], float]:
    """Adam GRAPE for state preparation on an MPS; returns (areas, fidelity)."""
    rng = np.random.default_rng(seed)
    n = len(edges)
    best_x, best_f = None, -1.0
    for _ in range(restarts):
        x = rng.uniform(0, 2 * np.pi, size=n)
        m = np.zeros(n); v = np.zeros(n); b1, b2, eps = 0.9, 0.999, 1e-8
        bx, bf = x.copy(), -1.0
        for t in range(1, steps + 1):
            f, g = fidelity_and_grad(init, target, x, edges, chi_max)
            m = b1 * m + (1 - b1) * g
            v = b2 * v + (1 - b2) * g * g
            x = x + lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)
            if f > bf:
                bf, bx = f, x.copy()
            if 1 - bf < tol:
                break
        if bf > best_f:
            best_f, best_x = bf, bx
        if 1 - best_f < tol:
            break
    return [float(a % (2 * np.pi)) for a in best_x], float(best_f)
