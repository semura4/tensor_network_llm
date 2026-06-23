"""Spin-1/2 operators and exact exchange propagators on an n-dot register.

Conventions
-----------
- ``n`` spin-1/2 particles (one per quantum dot), Hilbert space dimension 2**n.
- Computational basis ordered with dot 0 as the *most significant* bit, i.e. the
  state |b_0 b_1 ... b_{n-1}> has index sum(b_k * 2**(n-1-k)).  Bit value 0 = |up>
  (S_z = +1/2), bit value 1 = |down> (S_z = -1/2).
- Exchange on edge (i, j):  H = S_i . S_j  with  S = (1/2) sigma.  A square pulse
  of area A = integral J dt gives the propagator  U = exp(-i A S_i.S_j).

The exchange propagator has a closed form (no matrix exponential needed):

    S_i.S_j = (SWAP_ij - I/2) / 2,      SWAP_ij^2 = I
    exp(-i A S_i.S_j) = e^{iA/4} ( cos(A/2) I  -  i sin(A/2) SWAP_ij )

so a full SWAP (A = pi) is recovered up to a global phase.
"""

from __future__ import annotations

import numpy as np


def _swap_index(idx: int, n: int, i: int, j: int) -> int:
    """Return the basis index with the bits for dots i and j swapped."""
    bi = (idx >> (n - 1 - i)) & 1
    bj = (idx >> (n - 1 - j)) & 1
    if bi == bj:
        return idx
    # flip both bits
    return idx ^ (1 << (n - 1 - i)) ^ (1 << (n - 1 - j))


def swap_matrix(n: int, i: int, j: int) -> np.ndarray:
    """Permutation matrix that swaps spins on dots i and j (full 2**n space)."""
    dim = 1 << n
    P = np.zeros((dim, dim), dtype=complex)
    for col in range(dim):
        P[_swap_index(col, n, i, j), col] = 1.0
    return P


def exchange_propagator(n: int, i: int, j: int, area: float) -> np.ndarray:
    """Exact propagator exp(-i*area*S_i.S_j) on the full 2**n Hilbert space."""
    dim = 1 << n
    I = np.eye(dim, dtype=complex)
    S = swap_matrix(n, i, j)
    return np.exp(1j * area / 4.0) * (np.cos(area / 2.0) * I - 1j * np.sin(area / 2.0) * S)


def sz_total(n: int) -> np.ndarray:
    """Diagonal total-S_z operator (units of hbar), for sanity checks/sectors."""
    dim = 1 << n
    diag = np.empty(dim)
    for idx in range(dim):
        n_down = bin(idx).count("1")
        diag[idx] = 0.5 * (n - 2 * n_down)
    return np.diag(diag).astype(complex)


def s_dot_s(n: int, i: int, j: int) -> np.ndarray:
    """The exchange coupling operator S_i . S_j on the full space (Hermitian)."""
    return (swap_matrix(n, i, j) - 0.5 * np.eye(1 << n)) / 2.0


def apply_pulse(state: np.ndarray, n: int, i: int, j: int, area: float) -> np.ndarray:
    """Apply one exchange pulse to a state vector or to a stack of columns.

    ``state`` has shape (2**n,) or (2**n, k); avoids forming the full propagator
    by using the closed-form  U = e^{iA/4}(cos(A/2) I - i sin(A/2) SWAP).
    """
    perm = np.array([_swap_index(idx, n, i, j) for idx in range(1 << n)])
    swapped = state[perm]
    phase = np.exp(1j * area / 4.0)
    return phase * (np.cos(area / 2.0) * state - 1j * np.sin(area / 2.0) * swapped)
