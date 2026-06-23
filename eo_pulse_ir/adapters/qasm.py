"""OpenQASM-2 subset adapter (round-trips with the front-end parser)."""

from __future__ import annotations

from ..circuit import Circuit, parse_circuit

# our gate name -> OpenQASM name (qelib1)
_QASM_NAME = {"cx": "cx", "cnot": "cx", "cxswap": "cxswap"}
_PARAM = {"rz", "rx", "ry", "u"}


def from_openqasm(text: str) -> Circuit:
    """Parse an OpenQASM-2 subset string into a Circuit (alias of parse_circuit)."""
    return parse_circuit(text)


def to_openqasm(circuit: Circuit, register: str = "q") -> str:
    """Emit an OpenQASM-2 string for a Circuit.

    Uses qelib1 names where possible; ``cxswap`` is emitted as an opaque gate
    declaration (it has no qelib1 equivalent) so the output stays self-describing.
    """
    lines = ["OPENQASM 2.0;", 'include "qelib1.inc";']
    if any(g.name == "cxswap" for g in circuit.gates):
        lines.append("opaque cxswap c,t;")
    lines.append(f"qreg {register}[{circuit.num_qubits}];")
    for g in circuit.gates:
        name = _QASM_NAME.get(g.name, g.name)
        params = ""
        if g.name in _PARAM and g.params:
            params = "(" + ",".join(f"{p:.10g}" for p in g.params) + ")"
        qubits = ",".join(f"{register}[{q}]" for q in g.qubits)
        lines.append(f"{name}{params} {qubits};")
    return "\n".join(lines) + "\n"
