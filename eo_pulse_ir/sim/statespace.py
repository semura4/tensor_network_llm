"""State-space (dynamical-systems) view of exchange-only pulse control.

The operator picture is ``dU/dt = -i H(t) U`` on the group SU(2**n).  The
*state-space* picture is the Schrödinger flow of the state itself,

    d|psi>/dt = -i H(t) |psi>,

which, writing ``psi = u + i v`` and stacking ``x = [u; v] in R^{2d}``, is a real
*linear* dynamical system

    dx/dt = A_m  x ,      A_m = [[ Hi, Hr ], [ -Hr, Hi ]]   (H = Hr + i Hi),

with ``A_m`` skew-symmetric (the flow is a rotation on the state sphere, norm
preserved).  Exchange-only control is therefore a **switched linear system**: a
finite set of constant vector fields ``A_m`` (one per active edge / mode) and a
piecewise-constant switching signal.

For a single logical qubit this reduces to the Bloch sphere S^2 with
``db/dt = omega_m x b`` — two rotation vector fields about axes 120 deg apart.
The logical subspace is an *invariant manifold* of the exchange-only flow
(leakage = the manifold defect); a field gradient destroys its invariance.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from .encoding import DOTS_PER_QUBIT, logical_basis
from .operators import s_dot_s


def to_real(psi: np.ndarray) -> np.ndarray:
    return np.concatenate([psi.real, psi.imag])


def from_real(x: np.ndarray) -> np.ndarray:
    d = len(x) // 2
    return x[:d] + 1j * x[d:]


def real_generator(H: np.ndarray) -> np.ndarray:
    """Real 2d x 2d skew-symmetric vector field A with dx/dt = A x  <=>  psi'=-iH psi."""
    Hr, Hi = H.real, H.imag
    return np.block([[Hi, Hr], [-Hr, Hi]])


def flow(H: np.ndarray, psi: np.ndarray, t: float) -> np.ndarray:
    """Exact state-space flow e^{-iHt}|psi> (via Hermitian eigendecomposition)."""
    w, V = np.linalg.eigh(H)
    return V @ (np.exp(-1j * w * t) * (V.conj().T @ psi))


def bloch_vector(state2: np.ndarray) -> np.ndarray:
    a, b = state2[0], state2[1]
    return np.array([2 * np.real(np.conj(a) * b),
                     2 * np.imag(np.conj(a) * b),
                     np.abs(a) ** 2 - np.abs(b) ** 2])


def cross_field(omega: np.ndarray):
    """Return the Bloch vector field f(b) = omega x b (a linear ODE on R^3)."""
    omega = np.asarray(omega, float)
    return lambda b: np.cross(omega, b)


class StateSpaceSystem:
    """EO control as a switched linear dynamical system on the state sphere."""

    def __init__(self, num_qubits: int, j_max: float = 1.0):
        self.num_qubits = num_qubits
        self.n = num_qubits * DOTS_PER_QUBIT
        self.j_max = j_max
        self.edges: List[Tuple[int, int]] = [(i, i + 1) for i in range(self.n - 1)]

    def hamiltonian(self, edge: Tuple[int, int]) -> np.ndarray:
        return self.j_max * s_dot_s(self.n, edge[0], edge[1]).astype(complex)

    def vector_field(self, edge: Tuple[int, int]) -> np.ndarray:
        """The constant real linear vector field A_m for the active mode ``edge``."""
        return real_generator(self.hamiltonian(edge))

    def integrate(self, x0: np.ndarray, word: Sequence[Tuple[Tuple[int, int], float]]
                  ) -> np.ndarray:
        """Flow the real state x0 through a control word [(edge, dwell), ...]."""
        psi = from_real(np.asarray(x0, float))
        for edge, tau in word:
            psi = flow(self.hamiltonian(tuple(edge)), psi, tau)
        return to_real(psi)

    def logical_manifold_defect(self, psi: np.ndarray) -> float:
        """Distance^2 of a state from the logical (DFS) manifold = leakage."""
        L = logical_basis(self.num_qubits)
        proj = L @ (L.conj().T @ psi)
        return float(1.0 - np.real(np.vdot(proj, proj)) / np.real(np.vdot(psi, psi)))
