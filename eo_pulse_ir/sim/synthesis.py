"""Optimal pulse design for n-qubit exchange-only gates (control synthesis).

This is the pulse-control-engineering view: an n-qubit logical target is a
boundary condition for the switched bilinear control system, and we *design* the
control word (the exchange-pulse areas) that steers the propagator onto it.  We
use a layered ansatz over all nearest-neighbour edges and the exact
analytic-gradient optimiser (a GRAPE-style scheme on the EO system).

Tractable for a few logical qubits (the dense simulator is 2**(3n)); the result
is a concrete pulse list on the canonical dot layout (control qubits 0..n-1 on
dots 0..3n-1), directly simulatable and scorable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .fidelity import leakage
from .optimize import optimize_areas_grad
from .simulator import logical_block

Edge = Tuple[int, int]


def _qubit_intra(q: int, local: int) -> List[Edge]:
    """Intra-triple edges for logical qubit q (alternating, ``local`` pulses)."""
    lo, hi = (3 * q, 3 * q + 1), (3 * q + 1, 3 * q + 2)
    pattern = [hi, lo, hi, lo]
    return pattern[:local]


def inter_edges(num_qubits: int) -> List[Edge]:
    """Boundary edges between adjacent triples."""
    return [(3 * q + 2, 3 * q + 3) for q in range(num_qubits - 1)]


def layered_ansatz(num_qubits: int, n_layers: int, local: int = 3,
                   final_local: bool = True) -> List[Edge]:
    """Edge pattern: n_layers x [local block on each qubit, then all boundary edges]."""
    inter = inter_edges(num_qubits)
    seq: List[Edge] = []
    for _ in range(n_layers):
        for q in range(num_qubits):
            seq += _qubit_intra(q, local)
        seq += inter
    if final_local:
        for q in range(num_qubits):
            seq += _qubit_intra(q, local)
    return seq


@dataclass
class GateDesign:
    pulses: List[Tuple[Edge, float]]      # (edge, area), program order
    fidelity: float
    leakage: float
    num_qubits: int

    @property
    def num_pulses(self) -> int:
        return len(self.pulses)


def design_gate(target: np.ndarray, num_qubits: int, n_layers: int = 4,
                local: int = 3, final_local: bool = True, restarts: int = 24,
                steps: int = 1500, seed: int = 0,
                ansatz: Optional[List[Edge]] = None) -> GateDesign:
    """Design an exchange-pulse sequence realising ``target`` (an n-qubit unitary).

    Returns the best :class:`GateDesign` found (areas optimised against the
    leakage-aware average gate fidelity).
    """
    edges = ansatz if ansatz is not None else layered_ansatz(
        num_qubits, n_layers, local, final_local)
    areas, F = optimize_areas_grad(edges, num_qubits, target, steps=steps,
                                   restarts=restarts, seed=seed)
    M = logical_block(list(zip(edges, areas)), num_qubits)
    return GateDesign(pulses=list(zip(edges, areas)), fidelity=F,
                      leakage=leakage(M), num_qubits=num_qubits)
