"""Matrix-product-state (TEBD) evolver for exchange-only pulse dynamics.

The piecewise-constant EO dynamics is a stream of *nearest-neighbour* two-site
gates (each exchange pulse acts on adjacent dots) plus optional one-site Zeeman
gates — exactly the setting where time-evolving block decimation (TEBD) on a
matrix product state scales beyond exact 2**n diagonalisation.  This connects the
control IR back to this repository's original tensor-network theme: the same
MPS contraction that powered the language model now propagates spin dynamics.

An MPS is a list of rank-3 tensors ``A[i]`` of shape ``(chi_left, 2, chi_right)``.
Two-site gates are applied by merging neighbours, acting with the 4x4 gate, and
re-splitting via a truncated SVD (bond dimension capped at ``chi_max``).  With a
large enough ``chi_max`` the evolution is exact and matches the dense simulator;
for shallow/structured circuits the bond dimension stays small and large dot
arrays become tractable.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from .encoding import DOTS_PER_QUBIT, triple_logical_states
from .operators import _swap_index  # noqa: F401  (kept for parity / debugging)


def _two_site_exchange(area: float) -> np.ndarray:
    """4x4 exchange propagator exp(-i*area*S.S) on two spins (basis order uu,ud,du,dd)."""
    SWAP = np.array([[1, 0, 0, 0],
                     [0, 0, 1, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1]], dtype=complex)
    I4 = np.eye(4, dtype=complex)
    return np.exp(1j * area / 4) * (np.cos(area / 2) * I4 - 1j * np.sin(area / 2) * SWAP)


class MPS:
    def __init__(self, tensors: List[np.ndarray]):
        self.A = tensors

    @property
    def n(self) -> int:
        return len(self.A)

    def copy(self) -> "MPS":
        return MPS([t.copy() for t in self.A])

    # ---- construction ------------------------------------------------------
    @staticmethod
    def from_site_tensors_of_state(vec: np.ndarray, n_sites: int) -> "MPS":
        """Exact MPS of a small dense state vector (length 2**n_sites)."""
        tensors = []
        psi = vec.reshape(1, -1)            # (chi_left=1, rest)
        chi_l = 1
        for s in range(n_sites - 1):
            psi = psi.reshape(chi_l * 2, -1)
            U, S, Vh = np.linalg.svd(psi, full_matrices=False)
            k = len(S)
            tensors.append(U.reshape(chi_l, 2, k))
            psi = (np.diag(S) @ Vh)
            chi_l = k
        tensors.append(psi.reshape(chi_l, 2, 1))
        return MPS(tensors)

    @staticmethod
    def logical_register(bits: Sequence[int]) -> "MPS":
        """Product (over triples) MPS of an encoded computational-basis state.

        Each logical qubit is its 3-dot |0_L>/|1_L> doublet state (bond dim <=2
        within a triple, 1 between triples), so the whole register is built with
        no 2**n vector.
        """
        L = triple_logical_states()         # (8, 2)
        tensors: List[np.ndarray] = []
        for b in bits:
            triple = MPS.from_site_tensors_of_state(L[:, b], DOTS_PER_QUBIT)
            tensors.extend(triple.A)
        return MPS(tensors)

    # ---- gates -------------------------------------------------------------
    def apply_one_site(self, gate2: np.ndarray, i: int) -> None:
        self.A[i] = np.einsum("ab,lbr->lar", gate2, self.A[i])

    def apply_two_site(self, gate4: np.ndarray, i: int, chi_max: int = 64,
                       tol: float = 1e-12) -> float:
        """Apply a 4x4 gate to sites (i, i+1); return the discarded weight."""
        Al, Ar = self.A[i], self.A[i + 1]
        cl, _, cm = Al.shape
        _, _, cr = Ar.shape
        theta = np.tensordot(Al, Ar, axes=(2, 0))        # (cl,2,2,cr)
        theta = theta.reshape(cl, 4, cr)
        theta = np.einsum("xy,lyr->lxr", gate4, theta)   # apply gate on phys pair
        theta = theta.reshape(cl, 2, 2, cr).transpose(0, 1, 2, 3).reshape(cl * 2, 2 * cr)
        U, S, Vh = np.linalg.svd(theta, full_matrices=False)
        keep = S > tol
        k = int(min(chi_max, np.count_nonzero(keep)))
        k = max(k, 1)
        discarded = float(np.sum(S[k:] ** 2))
        U, S, Vh = U[:, :k], S[:k], Vh[:k, :]
        self.A[i] = U.reshape(cl, 2, k)
        self.A[i + 1] = (np.diag(S) @ Vh).reshape(k, 2, cr)
        return discarded

    # ---- contractions ------------------------------------------------------
    def overlap(self, other: "MPS") -> complex:
        """<other | self>."""
        E = np.ones((1, 1), dtype=complex)
        for Aa, Bb in zip(self.A, other.A):
            # E_{a',b'} = sum E_{a,b} A[a,s,a'] conj(B[b,s,b'])
            E = np.einsum("ab,asx,bsy->xy", E, Aa, Bb.conj())
        return complex(E[0, 0])

    def norm(self) -> float:
        return float(np.sqrt(np.real(self.overlap(self))))

    def max_bond(self) -> int:
        return max(t.shape[2] for t in self.A)

    def to_dense(self) -> np.ndarray:
        """Full state vector (small systems only)."""
        psi = self.A[0]
        for t in self.A[1:]:
            psi = np.tensordot(psi, t, axes=(psi.ndim - 1, 0))
        return psi.reshape(-1)


def evolve_pulses(mps: MPS, pulses, chi_max: int = 64, tol: float = 1e-12
                  ) -> Tuple[MPS, int, float]:
    """Apply a pulse list (edge, area) to an MPS via TEBD.

    Returns (evolved MPS, max bond dimension reached, total discarded weight).
    Pulses must be on nearest-neighbour edges (true for EO synthesis + routing).
    """
    out = mps.copy()
    max_chi = out.max_bond()
    discarded = 0.0
    for p in pulses:
        if hasattr(p, "edge"):
            (i, j), area = tuple(p.edge), float(p.area)
        else:
            (i, j), area = (int(p[0][0]), int(p[0][1])), float(p[1])
        if j != i + 1:
            raise ValueError(f"MPS evolver needs nearest-neighbour edges, got {(i, j)}")
        discarded += out.apply_two_site(_two_site_exchange(area), i, chi_max, tol)
        max_chi = max(max_chi, out.max_bond())
    return out, max_chi, discarded
