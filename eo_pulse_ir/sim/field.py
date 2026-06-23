"""Static Zeeman field / magnetic-field gradient on the dot array.

Adds a site-dependent Zeeman term  H_Z = sum_i b_i S_z^i  to the exchange
dynamics.  This is the physics that breaks the exchange-only protection:

- exchange  H_ex = sum J_ij S_i.S_j  conserves total S^2 and total S_z, so a
  single encoded qubit (one triple) never leaks out of its S=1/2 doublet;
- a *uniform* field (all b_i equal) is just b.S_z^total: it commutes with S^2 and
  exchange, contributing only a global phase -- no leakage, no gate error;
- a *field gradient* (unequal b_i) does NOT commute with S^2.  It couples the
  logical S=1/2 subspace to the S=3/2 states, so even pure intra-triple exchange
  now leaks, and two-qubit gates pick up extra error.

Because exchange and Zeeman act simultaneously during a pulse, the propagator is
no longer the closed-form SWAP expression; we diagonalise H = J_max S_i.S_j + H_Z
once per edge (numpy ``eigh``) and reuse it for every pulse on that edge.
"""

from __future__ import annotations

from typing import Dict, Iterable, Sequence, Tuple

import numpy as np

from .encoding import DOTS_PER_QUBIT, logical_basis
from .operators import s_dot_s

Edge = Tuple[int, int]


def zeeman_energies(n: int, gradient: float, b0: float = 0.0) -> np.ndarray:
    """Per-site Zeeman energies b_i = b0 + gradient*(i - center), i = 0..n-1.

    ``gradient`` is the Zeeman difference per dot (same units as exchange, hbar=1).
    Only the *gradient* (differences) affects fidelity/leakage; ``b0`` is a global
    offset (pure global phase).
    """
    center = (n - 1) / 2.0
    return np.array([b0 + gradient * (i - center) for i in range(n)], dtype=float)


def zeeman_diagonal(b: Sequence[float]) -> np.ndarray:
    """Diagonal of H_Z = sum_i b_i S_z^i over the 2**n computational basis."""
    n = len(b)
    dim = 1 << n
    b = np.asarray(b, float)
    diag = np.zeros(dim)
    for idx in range(dim):
        tot = 0.0
        for i in range(n):
            bit = (idx >> (n - 1 - i)) & 1   # 0 = up (+1/2), 1 = down (-1/2)
            tot += b[i] * (0.5 if bit == 0 else -0.5)
        diag[idx] = tot
    return diag


def _edge_area(p) -> Tuple[Edge, float]:
    if hasattr(p, "edge") and hasattr(p, "area"):
        return tuple(p.edge), float(p.area)
    (edge, area) = p
    return (int(edge[0]), int(edge[1])), float(area)


def logical_block_field(pulses: Iterable, num_qubits: int, b: Sequence[float],
                        j_max: float = 1.0) -> np.ndarray:
    """Logical block M = L^dagger U L with a static Zeeman field on throughout.

    Each pulse of area A on edge (i,j) evolves under H = j_max*S_i.S_j + H_Z for
    time tau = A / j_max (constant-amplitude square pulse); the field acts during
    every pulse.  ``b`` is the per-site Zeeman energy array (length 3*num_qubits).
    """
    n = num_qubits * DOTS_PER_QUBIT
    pulses = list(pulses)
    Hz = np.diag(zeeman_diagonal(b)).astype(complex)

    # diagonalise H = j_max*S.S + H_Z once per distinct edge
    eig: Dict[Edge, Tuple[np.ndarray, np.ndarray]] = {}
    for p in pulses:
        e, _ = _edge_area(p)
        if e not in eig:
            H = j_max * s_dot_s(n, e[0], e[1]).astype(complex) + Hz
            w, V = np.linalg.eigh(H)
            eig[e] = (w, V)

    psi = logical_basis(num_qubits).astype(complex)   # (2**n, d)
    for p in pulses:
        e, area = _edge_area(p)
        tau = area / j_max if j_max else 0.0
        w, V = eig[e]
        phase = np.exp(-1j * w * tau)
        psi = V @ (phase[:, None] * (V.conj().T @ psi))

    L = logical_basis(num_qubits)
    return L.conj().T @ psi


def simulate_field(pulses, num_qubits: int, gradient: float, target=None,
                   j_max: float = 1.0, b0: float = 0.0) -> dict:
    """Convenience: build a linear gradient, evolve, and score leakage/fidelity."""
    from .fidelity import average_gate_fidelity, leakage
    n = num_qubits * DOTS_PER_QUBIT
    b = zeeman_energies(n, gradient, b0)
    M = logical_block_field(pulses, num_qubits, b, j_max)
    out = {"gradient": gradient, "leakage": leakage(M), "M": M}
    if target is not None:
        out["fidelity"] = average_gate_fidelity(M, target)
        out["infidelity"] = 1.0 - out["fidelity"]
    return out
