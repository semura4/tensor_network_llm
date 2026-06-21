"""Exchange-only physics simulator (optional, requires numpy).

A small dense Heisenberg-exchange simulator that turns the IR's pulse list into a
real subspace-restricted gate fidelity and leakage, and maps the
pulse-parameter -> (fidelity, leakage, robustness) landscape.

Importing this subpackage requires numpy; the core ``eo_pulse_ir`` IR does not.
"""

from . import gates
from .control import EOControlSystem
from .encoding import DOTS_PER_QUBIT, logical_basis, triple_logical_states
from .fidelity import average_gate_fidelity, gate_overlap, leakage
from .field import logical_block_field, simulate_field, zeeman_energies
from .lie import lie_closure, sector_indices, subspace_coupling
from .operators import apply_pulse, exchange_propagator, s_dot_s, swap_matrix
from .simulator import logical_block, simulate

__all__ = [
    "gates",
    "logical_basis", "triple_logical_states", "DOTS_PER_QUBIT",
    "average_gate_fidelity", "gate_overlap", "leakage",
    "exchange_propagator", "swap_matrix", "s_dot_s", "apply_pulse",
    "logical_block", "simulate",
    "logical_block_field", "simulate_field", "zeeman_energies",
    "EOControlSystem", "lie_closure", "sector_indices", "subspace_coupling",
]
