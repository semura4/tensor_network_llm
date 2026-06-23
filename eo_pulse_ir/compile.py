"""Synthesiser: logical :class:`Circuit` -> list of :class:`Pulse`.

This is the heart of the IR.  For each logical gate we:

1. (two-qubit only) route the operands to adjacency by inserting logical SWAPs,
   emitting their exchange pulses and updating the topology;
2. look up the gate's exchange-pulse template;
3. resolve each template role to a concrete dot edge using the current layout.

The result is a flat, program-ordered pulse list ready for scheduling.

Supports both :class:`LinearTopology` (1-D chain) and :class:`GridTopology`
(2-D square-lattice patch).
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

from .circuit import Circuit, Gate
from .native import one_qubit_template, two_qubit_template
from .schedule import Pulse
from .topology import DOTS_PER_QUBIT, GridTopology, LinearTopology

Topology = Union[LinearTopology, GridTopology]


def _physical_role(role: str) -> str:
    """Map a template edge-role to a physical category: intra / inter / route."""
    if role == "inter":
        return "inter"
    if role == "route":
        return "route"
    return "intra"


def _slot_index(topo: Topology, pos) -> int:
    """Convert a topology position to a flat slot index for dot math."""
    if isinstance(topo, GridTopology):
        return pos  # GridTopology positions are already flat slot ints
    return pos  # LinearTopology positions are ints too


def _resolve_role(role: str, topo: Topology,
                  ctrl_pos, tgt_pos, q_for_1q: int) -> Tuple[int, int]:
    """Resolve a template edge-role to a concrete (low_dot, high_dot) pair."""
    if role == "intra_low":
        return topo.intra_edges(q_for_1q)[0]
    if role == "intra_high":
        return topo.intra_edges(q_for_1q)[1]

    ctrl_idx = _slot_index(topo, ctrl_pos)
    tgt_idx = _slot_index(topo, tgt_pos)

    if role == "ctrl_high":
        base = ctrl_idx * DOTS_PER_QUBIT
        return (base + 1, base + 2)
    if role == "ctrl_low":
        base = ctrl_idx * DOTS_PER_QUBIT
        return (base, base + 1)
    if role == "tgt_low":
        base = tgt_idx * DOTS_PER_QUBIT
        return (base, base + 1)
    if role == "tgt_high":
        base = tgt_idx * DOTS_PER_QUBIT
        return (base + 1, base + 2)
    if role == "inter":
        if isinstance(topo, GridTopology):
            return topo.inter_edge(ctrl_pos, tgt_pos)
        return topo.inter_edge(ctrl_pos)
    raise ValueError(f"cannot resolve role {role!r}")


def _position_of(topo: Topology, q: int):
    """Return the current position of logical qubit ``q``."""
    if isinstance(topo, GridTopology):
        return topo.slot_of[q]
    return topo.position_of[q]


def _are_adjacent(topo: Topology, pos_a, pos_b) -> bool:
    if isinstance(topo, GridTopology):
        return topo.grid_distance(pos_a, pos_b) == 1
    return abs(pos_a - pos_b) == 1


def _emit_logical_swap_grid(topo: GridTopology, slot_a: int, slot_b: int,
                            qubits: Tuple[int, ...]) -> List[Pulse]:
    """Emit SWAP pulses between two adjacent slots on a grid."""
    pulses: List[Pulse] = []
    for role, area in two_qubit_template("swap"):
        edge = _resolve_role(role, topo, slot_a, slot_b,
                             qubits[0] if qubits else 0)
        pulses.append(Pulse(edge=edge, area=area, gate="swap(route)",
                            logical_qubits=qubits, role="route"))
    return pulses


def _emit_logical_swap_linear(topo: LinearTopology, pos: int,
                              qubits: Tuple[int, ...]) -> List[Pulse]:
    """Emit SWAP pulses between positions ``pos`` and ``pos+1``."""
    pulses: List[Pulse] = []
    ctrl_pos, tgt_pos = pos, pos + 1
    for role, area in two_qubit_template("swap"):
        edge = _resolve_role(role, topo, ctrl_pos, tgt_pos,
                             qubits[0] if qubits else 0)
        pulses.append(Pulse(edge=edge, area=area, gate="swap(route)",
                            logical_qubits=qubits, role="route"))
    return pulses


def synthesize(circuit: Circuit,
               topology: Optional[Topology] = None
               ) -> Tuple[List[Pulse], Topology]:
    """Compile a logical circuit to an exchange-pulse list.

    Parameters
    ----------
    circuit : Circuit
        The logical circuit to compile.
    topology : LinearTopology | GridTopology | None
        Physical layout.  If None, defaults to ``LinearTopology(num_qubits)``.
        For :class:`GridTopology`, call :meth:`assign_initial_layout` before
        passing it here.

    Returns
    -------
    (pulses, topology)
        The flat pulse list (program order) and the final topology state.
    """
    if topology is None:
        topology = LinearTopology(circuit.num_qubits)

    # For GridTopology, ensure layout is assigned
    if isinstance(topology, GridTopology) and not topology.slot_of:
        topology.assign_initial_layout(list(range(circuit.num_qubits)),
                                       circuit_gates=circuit.gates)

    pulses: List[Pulse] = []

    for g in circuit.gates:
        if not g.is_two_qubit:
            q = g.qubits[0]
            param = g.params if g.params else None
            pos_q = _position_of(topology, q)
            for role, area in one_qubit_template(g.name, param):
                edge = _resolve_role(role, topology, pos_q, pos_q, q)
                pulses.append(Pulse(edge=edge, area=area, gate=g.name,
                                    logical_qubits=g.qubits,
                                    role=_physical_role(role)))
            continue

        # two-qubit gate: route operands to adjacency
        q_ctrl, q_tgt = g.qubits[0], g.qubits[1]

        if isinstance(topology, GridTopology):
            swap_steps = topology.route_adjacent(q_ctrl, q_tgt)
            for (slot_a, slot_b) in swap_steps:
                qa = topology.occupant[slot_a]
                qb = topology.occupant[slot_b]
                moved = (qa if qa >= 0 else 0, qb if qb >= 0 else 0)
                pulses.extend(_emit_logical_swap_grid(topology, slot_a, slot_b, moved))
                topology.swap_slots(slot_a, slot_b)
        else:
            for (pa, pb) in topology.route_adjacent(q_ctrl, q_tgt):
                moved = (topology.order[pa], topology.order[pb])
                pulses.extend(_emit_logical_swap_linear(topology, pa, moved))
                topology.swap_positions(pa)

        ctrl_pos = _position_of(topology, q_ctrl)
        tgt_pos = _position_of(topology, q_tgt)
        assert _are_adjacent(topology, ctrl_pos, tgt_pos), \
            f"routing failed: {ctrl_pos} and {tgt_pos} not adjacent"

        for role, area in two_qubit_template(g.name):
            edge = _resolve_role(role, topology, ctrl_pos, tgt_pos, q_ctrl)
            pulses.append(Pulse(edge=edge, area=area, gate=g.name,
                                logical_qubits=g.qubits,
                                role=_physical_role(role)))

    return pulses, topology
