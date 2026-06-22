"""Exchange-only physics simulator (optional, requires numpy).

A small dense Heisenberg-exchange simulator that turns the IR's pulse list into a
real subspace-restricted gate fidelity and leakage, and maps the
pulse-parameter -> (fidelity, leakage, robustness) landscape.

Importing this subpackage requires numpy; the core ``eo_pulse_ir`` IR does not.
"""

from . import gates
from .bifurcation import classify_critical, local_maxima, optima_sweep
from .control import EOControlSystem
from .encoding import DOTS_PER_QUBIT, logical_basis, triple_logical_states
from .fidelity import average_gate_fidelity, gate_overlap, leakage
from .field import logical_block_field, simulate_field, zeeman_energies
from .lie import lie_closure, sector_indices, subspace_coupling
from .mps import MPS, evolve_pulses
from .mps_grape import ghz_target, grape_state_prep, state_prep_fidelity
from .operators import apply_pulse, exchange_propagator, s_dot_s, swap_matrix
from .simulator import logical_block, simulate
from .statespace import StateSpaceSystem, real_generator
from .synthesis import GateDesign, design_gate, layered_ansatz

__all__ = [
    "gates",
    "logical_basis", "triple_logical_states", "DOTS_PER_QUBIT",
    "average_gate_fidelity", "gate_overlap", "leakage",
    "exchange_propagator", "swap_matrix", "s_dot_s", "apply_pulse",
    "logical_block", "simulate",
    "logical_block_field", "simulate_field", "zeeman_energies",
    "EOControlSystem", "lie_closure", "sector_indices", "subspace_coupling",
    "MPS", "evolve_pulses",
    "StateSpaceSystem", "real_generator",
    "design_gate", "layered_ansatz", "GateDesign",
    "ghz_target", "grape_state_prep", "state_prep_fidelity",
    "local_maxima", "classify_critical", "optima_sweep",
]
