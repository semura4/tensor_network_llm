"""Synthesiser: logical :class:`Circuit` -> list of :class:`Pulse`.

This is the heart of the IR.  For each logical gate we:

1. (two-qubit only) route the operands to adjacency by inserting logical SWAPs,
   emitting their exchange pulses and updating the topology;
2. look up the gate's exchange-pulse template;
3. resolve each template role to a concrete dot edge using the current layout.

The result is a flat, program-ordered pulse list ready for scheduling.
"""

from __future__ import annotations

from typing import List, Tuple

from .circuit import Circuit, Gate
from .native import one_qubit_template, two_qubit_template
from .schedule import Pulse
from .topology import LinearTopology


def _physical_role(role: str) -> str:
    """Map a template edge-role to a physical category: intra / inter / route."""
    if role == "inter":
        return "inter"
    if role == "route":
        return "route"
    return "intra"


def _resolve_role(role: str, topo: LinearTopology, ctrl_pos: int, tgt_pos: int,
                  q_for_1q: int) -> Tuple[int, int]:
    if role == "intra_low":
        return topo.intra_edges(q_for_1q)[0]
    if role == "intra_high":
        return topo.intra_edges(q_for_1q)[1]
    if role == "ctrl_high":
        return (ctrl_pos * 3 + 1, ctrl_pos * 3 + 2)
    if role == "ctrl_low":
        return (ctrl_pos * 3, ctrl_pos * 3 + 1)
    if role == "tgt_low":
        return (tgt_pos * 3, tgt_pos * 3 + 1)
    if role == "tgt_high":
        return (tgt_pos * 3 + 1, tgt_pos * 3 + 2)
    if role == "inter":
        return topo.inter_edge(ctrl_pos)
    raise ValueError(f"cannot resolve role {role!r}")


def _emit_logical_swap(topo: LinearTopology, pos: int, qubits: Tuple[int, ...]) -> List[Pulse]:
    """Emit the exchange pulses for a routing SWAP between positions pos, pos+1."""
    pulses: List[Pulse] = []
    ctrl_pos, tgt_pos = pos, pos + 1
    for role, area in two_qubit_template("swap"):
        edge = _resolve_role(role, topo, ctrl_pos, tgt_pos, qubits[0] if qubits else 0)
        pulses.append(Pulse(edge=edge, area=area, gate="swap(route)",
                            logical_qubits=qubits, role="route"))
    return pulses


def synthesize(circuit: Circuit) -> Tuple[List[Pulse], LinearTopology]:
    """Compile a logical circuit to an exchange-pulse list.

    Returns the pulse list (program order) and the final topology (positions may
    have changed due to routing SWAPs).
    """
    topo = LinearTopology(circuit.num_qubits)
    pulses: List[Pulse] = []

    for g in circuit.gates:
        if not g.is_two_qubit:
            q = g.qubits[0]
            param = g.params if g.params else None
            for role, area in one_qubit_template(g.name, param):
                edge = _resolve_role(role, topo, topo.position_of[q], topo.position_of[q], q)
                pulses.append(Pulse(edge=edge, area=area, gate=g.name,
                                    logical_qubits=g.qubits, role=_physical_role(role)))
            continue

        # two-qubit gate: route operands to adjacency (control left of target)
        q_ctrl, q_tgt = g.qubits[0], g.qubits[1]
        for (pa, pb) in topo.route_adjacent(q_ctrl, q_tgt):
            moved = (topo.order[pa], topo.order[pb])
            pulses.extend(_emit_logical_swap(topo, pa, moved))
            topo.swap_positions(pa)

        ctrl_pos = topo.position_of[q_ctrl]
        tgt_pos = topo.position_of[q_tgt]
        assert tgt_pos == ctrl_pos + 1, (ctrl_pos, tgt_pos)

        for role, area in two_qubit_template(g.name):
            edge = _resolve_role(role, topo, ctrl_pos, tgt_pos, q_ctrl)
            pulses.append(Pulse(edge=edge, area=area, gate=g.name,
                                logical_qubits=g.qubits, role=_physical_role(role)))

    return pulses, topo
