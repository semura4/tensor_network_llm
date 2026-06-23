"""Exchange-only control as a piecewise-constant (switched bilinear) system.

This is the explicit dynamical-systems view of EO pulse control: a right-invariant
switched bilinear system on SU(2**n),

    dU/dt = -i (H_drift + J · G_{m(t)}) U ,

with a finite mode set (the nearest-neighbour exchange generators G_e = S_i·S_j),
an always-on drift H_drift (the Zeeman field), and a control *word* — a sequence
of (mode, dwell-time) segments.  A pulse list is exactly such a word; the gate
optimiser is control synthesis on this system.

`EOControlSystem` makes that structure first-class and exposes a controllability
analysis (the generated Lie algebra) via :mod:`eo_pulse_ir.sim.lie`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .encoding import DOTS_PER_QUBIT, logical_basis
from .field import zeeman_diagonal, zeeman_energies
from .lie import (as_generator, lie_closure, restrict, sector_indices,
                  subspace_coupling, traceless)
from .operators import s_dot_s

Edge = Tuple[int, int]


@dataclass
class EOControlSystem:
    """Switched bilinear EO control system on ``num_qubits`` encoded qubits."""

    num_qubits: int
    j_max: float = 1.0
    gradient: float = 0.0          # Zeeman gradient per dot (drift)
    b0: float = 0.0                # uniform Zeeman offset (drift)

    def __post_init__(self) -> None:
        self.n = self.num_qubits * DOTS_PER_QUBIT
        self.edges: List[Edge] = [(i, i + 1) for i in range(self.n - 1)]
        self._b = zeeman_energies(self.n, self.gradient, self.b0)
        self._Hz = np.diag(zeeman_diagonal(self._b)).astype(complex)
        self._eig: Dict[Edge, Tuple[np.ndarray, np.ndarray]] = {}

    # ---- dynamics ----------------------------------------------------------
    def segment_hamiltonian(self, edge: Edge) -> np.ndarray:
        """H = j_max·S_i·S_j + H_drift for the active mode ``edge``."""
        return self.j_max * s_dot_s(self.n, edge[0], edge[1]).astype(complex) + self._Hz

    def _edge_eig(self, edge: Edge):
        if edge not in self._eig:
            self._eig[edge] = np.linalg.eigh(self.segment_hamiltonian(edge))
        return self._eig[edge]

    def propagator(self, word: Sequence[Tuple[Edge, float]]) -> np.ndarray:
        """Total unitary for a control word ``[(edge, dwell_time), ...]``."""
        dim = 1 << self.n
        U = np.eye(dim, dtype=complex)
        for edge, tau in word:
            w, V = self._edge_eig(tuple(edge))
            U = (V * np.exp(-1j * w * tau)) @ (V.conj().T) @ U
        return U

    def word_from_pulses(self, pulses) -> List[Tuple[Edge, float]]:
        """Convert a pulse list (edge, area) into a control word (edge, dwell)."""
        word = []
        for p in pulses:
            if hasattr(p, "edge"):
                edge, area = tuple(p.edge), float(p.area)
            else:
                edge, area = (int(p[0][0]), int(p[0][1])), float(p[1])
            word.append((edge, area / self.j_max))
        return word

    def logical_block(self, word: Sequence[Tuple[Edge, float]]) -> np.ndarray:
        L = logical_basis(self.num_qubits)
        return L.conj().T @ self.propagator(word) @ L

    # ---- controllability ---------------------------------------------------
    def control_generators(self, sector_down: Optional[int] = None,
                           include_drift: bool = True) -> List[np.ndarray]:
        """Anti-Hermitian mode generators (optionally restricted to an S_z sector).

        ``sector_down`` selects the total-S_z sector (number of down spins) to keep
        matrices small; ``None`` uses the full 2**n space.
        """
        idx = sector_indices(self.n, sector_down) if sector_down is not None else None

        def prep(H):
            M = restrict(H, idx) if idx is not None else H
            return as_generator(traceless(M))

        gens = [prep(self.j_max * s_dot_s(self.n, e[0], e[1])) for e in self.edges]
        if include_drift and (self.gradient != 0.0 or self.b0 != 0.0):
            gens.append(prep(self._Hz))
        return gens

    def lie_dimension(self, sector_down: Optional[int] = None) -> int:
        """Dimension of the reachable Lie algebra (controllability measure)."""
        dim, _ = lie_closure(self.control_generators(sector_down))
        return dim

    def leakage_coupling(self, sector_down: int) -> float:
        """Max algebra coupling between the logical subspace and its complement.

        Built in the given S_z sector; 0 means the dynamics is confined to the
        logical (DFS) block — leakage is uncontrollable, i.e. protected.
        """
        idx = np.asarray(sector_indices(self.n, sector_down))
        _, basis = lie_closure(self.control_generators(sector_down))
        # logical isometry restricted to the sector
        L = logical_basis(self.num_qubits)
        Lr = L[idx, :]                      # (sector_dim, d_logical)
        Lr, _ = np.linalg.qr(Lr)            # orthonormalise within the sector
        # leakage subspace = orthonormal complement of Lr in the sector
        full = np.eye(len(idx), dtype=complex)
        proj = full - Lr @ Lr.conj().T
        u, s, _ = np.linalg.svd(proj)
        leak = u[:, s > 1e-9]
        return subspace_coupling(basis, Lr, leak)
