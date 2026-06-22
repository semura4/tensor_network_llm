"""Valley degree of freedom: spin (x) valley 4-level dots.

Extends the pure-spin exchange model with a per-dot **valley pseudo-spin**, so the
simulator can study valley leakage and the E_VS >> J requirement that the
literature identifies as the central silicon-specific constraint
(see docs/valley_splitting_research.md).

Model (effective, Heitler-London style; documented assumptions).
Each dot is spin(2) (x) valley(2) = 4 levels (index = 2*spin + valley, valley 0 =
ground, 1 = excited, in the dot's *local* frame). An exchange pulse on edge (i,j)
is the physical electron swap — a partial SWAP of the two dots' **full** spin (x)
valley states — but the two dots' valley frames differ by a **valley phase**
Δφ = φ_j − φ_i, so the swap is twisted (dot j's valley rotated by Δφ). A per-dot
**valley splitting** E_VS,i penalises the excited valley and acts *during* the
pulse. Consequences, all reproduced by the model:

- valley phases aligned (Δφ = 0): the swap is valley-trivial → exchange is pure
  spin Heisenberg, **zero leakage** for any E_VS (recovers the spin-only model);
- valley-phase mismatch with small E_VS: population leaks to excited valley;
- large **E_VS >> J suppresses valley leakage** (off-resonant intervalley terms).

The logical subspace is the encoded spin doublet with **all dots in the ground
valley**; leakage is any excited-valley population. Cost is 4**(3*nq) (use nq<=2).

For cheaply re-scoring a *full* validated gate against valley-phase mismatch
without the 4**n cost, see :func:`effective_valley_areas` (the intravalley
area-suppression model).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .encoding import DOTS_PER_QUBIT, logical_basis

Edge = Tuple[int, int]

_I2 = np.eye(2)
_NV = np.diag([0.0, 1.0])           # excited-valley projector (valley basis)


def _ry(t: float) -> np.ndarray:
    c, s = np.cos(t / 2), np.sin(t / 2)
    return np.array([[c, -s], [s, c]])


def _swap_two_dots() -> np.ndarray:
    """Permutation swapping two 4-level (spin(x)valley) dots (16x16)."""
    P = np.zeros((16, 16))
    for a in range(4):
        for b in range(4):
            P[b * 4 + a, a * 4 + b] = 1.0
    return P


_SWAP16 = _swap_two_dots()
_G16 = (_SWAP16 - 0.5 * np.eye(16)) / 2.0          # exchange generator, full SWAP at area=pi
# valley-splitting on each of the two active dots (excited valley costs E_VS)
_HVS_A = np.kron(np.kron(_I2, _NV), np.eye(4))
_HVS_B = np.kron(np.eye(4), np.kron(_I2, _NV))


def _twist(dphi: float) -> np.ndarray:
    rb = np.kron(_I2, _ry(dphi))                   # rotate dot B's valley frame
    return np.kron(np.eye(4), rb)


def two_dot_pulse(area: float, dphi: float, evs_i: float, evs_j: float,
                  j_max: float = 1.0) -> np.ndarray:
    """16x16 propagator for an exchange pulse on two spin(x)valley dots.

    ``dphi`` = valley-phase mismatch, ``evs_*`` = the dots' valley splittings.
    """
    U = _twist(dphi)
    G = U @ _G16 @ U.conj().T
    H = j_max * G + evs_i * _HVS_A + evs_j * _HVS_B
    tau = area / j_max if j_max else 0.0
    w, V = np.linalg.eigh(H)
    return (V * np.exp(-1j * w * tau)) @ V.conj().T


def _embed_spin_to_spinvalley(n_dots: int) -> np.ndarray:
    """Isometry (4**n x 2**n): spin basis -> spin(x)valley with all valleys ground."""
    dim_s = 1 << n_dots
    E = np.zeros((4 ** n_dots, dim_s))
    for s in range(dim_s):
        f = 0
        for k in range(n_dots):
            spin = (s >> (n_dots - 1 - k)) & 1
            f += (2 * spin + 0) * (4 ** (n_dots - 1 - k))   # valley = ground (0)
        E[f, s] = 1.0
    return E


def _apply_two_dot(state: np.ndarray, n_dots: int, i: int, op16: np.ndarray) -> np.ndarray:
    """Apply a 16x16 op to adjacent dots (i, i+1) of a 4**n_dots state vector/stack."""
    left = 4 ** i
    right = 4 ** (n_dots - i - 2)
    cols = state.shape[1] if state.ndim == 2 else 1
    sh = state.reshape(left, 16, right, cols) if state.ndim == 2 else \
        state.reshape(left, 16, right)
    if state.ndim == 2:
        out = np.einsum("xy,lyrc->lxrc", op16, sh)
        return out.reshape(4 ** n_dots, cols)
    out = np.einsum("xy,lyr->lxr", op16, sh)
    return out.reshape(4 ** n_dots)


def logical_block_valley(pulses: Sequence, num_qubits: int,
                         valley_phase: Sequence[float], e_vs,
                         j_max: float = 1.0) -> np.ndarray:
    """Logical block M (d x d) over the spin-doublet (x) all-ground-valley subspace.

    ``valley_phase[k]`` and ``e_vs`` (scalar or per-dot) define the valley frame and
    splitting of dot k. Leakage = any excited-valley population is captured by M
    being sub-unitary.
    """
    n = num_qubits * DOTS_PER_QUBIT
    e_vs = np.full(n, e_vs, float) if np.isscalar(e_vs) else np.asarray(e_vs, float)
    phase = np.asarray(valley_phase, float)
    E = _embed_spin_to_spinvalley(n)
    Lspin = logical_basis(num_qubits)
    psi = (E @ Lspin).astype(complex)               # (4**n, d), all valleys ground
    for p in pulses:
        if hasattr(p, "edge"):
            (i, j), area = tuple(p.edge), float(p.area)
        else:
            (i, j), area = (int(p[0][0]), int(p[0][1])), float(p[1])
        op = two_dot_pulse(area, phase[j] - phase[i], e_vs[i], e_vs[j], j_max)
        psi = _apply_two_dot(psi, n, i, op)
    Lfull = (E @ Lspin)
    return Lfull.conj().T @ psi


def simulate_valley(pulses: Sequence, num_qubits: int, valley_phase, e_vs,
                    target: Optional[np.ndarray] = None, j_max: float = 1.0) -> dict:
    """Valley-aware simulation: logical block, valley leakage, and (optional) fidelity."""
    from .fidelity import average_gate_fidelity, leakage
    n = num_qubits * DOTS_PER_QUBIT
    if np.isscalar(valley_phase):
        valley_phase = np.full(n, valley_phase, float)
    M = logical_block_valley(pulses, num_qubits, valley_phase, e_vs, j_max)
    out = {"M": M, "valley_leakage": leakage(M)}
    if target is not None:
        out["fidelity"] = average_gate_fidelity(M, target)
        out["infidelity"] = 1.0 - out["fidelity"]
    return out


# ---- cheap effective model: intravalley exchange suppression -------------------
def effective_valley_areas(pulses: Sequence, valley_phase: Sequence[float]
                           ) -> List[Tuple[Edge, float]]:
    """Scale each pulse's area by cos^2(Δφ/2) of its edge (intravalley suppression).

    A cheap spin-only proxy (no leakage) for re-scoring a full validated gate vs
    valley-phase mismatch: J_eff = J cos^2((φ_i−φ_j)/2), nulling at Δφ = π.
    """
    phase = np.asarray(valley_phase, float)
    out = []
    for p in pulses:
        if hasattr(p, "edge"):
            (i, j), area = tuple(p.edge), float(p.area)
        else:
            (i, j), area = (int(p[0][0]), int(p[0][1])), float(p[1])
        out.append(((i, j), area * np.cos((phase[i] - phase[j]) / 2) ** 2))
    return out
