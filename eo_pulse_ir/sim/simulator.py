"""Simulate an exchange-pulse list and score it against a target gate.

Consumes exactly the pulse representation produced by the IR (each pulse has an
``edge`` (i, j) and an ``area``), so the same object can be synthesised by
``eo_pulse_ir.compile`` or ingested from an optimiser / eoqrid and then handed
here for a *real* (not heuristic) fidelity and leakage number.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple, Union

import numpy as np

from .encoding import DOTS_PER_QUBIT, logical_basis
from .fidelity import average_gate_fidelity, gate_overlap, leakage
from .operators import apply_pulse

# a pulse is either an object with .edge/.area, or a ((i, j), area) tuple
PulseLike = Union[Tuple[Tuple[int, int], float], object]


def _edge_area(p: PulseLike) -> Tuple[Tuple[int, int], float]:
    if hasattr(p, "edge") and hasattr(p, "area"):
        return tuple(p.edge), float(p.area)
    (edge, area) = p
    return (int(edge[0]), int(edge[1])), float(area)


def logical_block(pulses: Iterable[PulseLike], num_qubits: int) -> np.ndarray:
    """Return M = L^dagger U L, the logical block of the pulse-list evolution."""
    n = num_qubits * DOTS_PER_QUBIT
    psi = logical_basis(num_qubits).astype(complex)  # (2**n, 2**nq)
    for p in pulses:
        (i, j), area = _edge_area(p)
        psi = apply_pulse(psi, n, i, j, area)
    L = logical_basis(num_qubits)
    return L.conj().T @ psi


def simulate(pulses: Sequence[PulseLike], num_qubits: int,
             target: Optional[np.ndarray] = None) -> dict:
    """Compute the logical block and its leakage; add fidelity if a target given."""
    M = logical_block(pulses, num_qubits)
    out = {"M": M, "leakage": leakage(M), "num_qubits": num_qubits,
           "num_pulses": len(pulses)}
    if target is not None:
        out["fidelity"] = average_gate_fidelity(M, target)
        out["overlap"] = gate_overlap(M, target)
    return out
