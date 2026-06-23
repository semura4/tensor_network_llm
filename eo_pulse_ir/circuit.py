"""Logical circuit representation and a small QASM-lite / OpenQASM-2 subset parser.

The circuit layer is deliberately tiny: it is the *front end* of the EO Pulse
Control IR.  A circuit is just an ordered list of logical gates over a register
of logical qubits.  Everything physical (encoding into quantum dots, exchange
pulse synthesis, scheduling, hardware mapping) happens downstream.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple


# Single-qubit gates whose logical rotation axis is aligned with one exchange
# generator in the standard 3-dot encoding (cheap: one exchange pulse).
ALIGNED_1Q = {"z", "s", "sdg", "t", "tdg", "rz"}
# Single-qubit gates that need an alternating-generator sequence.
GENERIC_1Q = {"x", "y", "h", "rx", "ry", "u"}
ONE_Q_GATES = ALIGNED_1Q | GENERIC_1Q
# Two-qubit gates we know how to synthesise into exchange pulses.
TWO_Q_GATES = {"cx", "cnot", "swap", "cxswap"}
PARAM_GATES = {"rz", "rx", "ry"}   # one angle each
# arbitrary single-qubit gate u(theta, phi, lam): three angles
U_GATE = "u"


@dataclass
class Gate:
    """A single logical gate.

    name: lowercase gate name (e.g. "cx", "h", "rz").
    qubits: logical qubit indices the gate acts on, in role order
            (for two-qubit gates: (control, target)).
    params: continuous parameters in radians (e.g. rotation angle).
    """

    name: str
    qubits: Tuple[int, ...]
    params: Tuple[float, ...] = ()

    def __post_init__(self) -> None:
        self.name = self.name.lower()
        if self.name == "cnot":
            self.name = "cx"

    @property
    def is_two_qubit(self) -> bool:
        return self.name in TWO_Q_GATES

    def __str__(self) -> str:
        p = f"({', '.join(f'{x:.4g}' for x in self.params)})" if self.params else ""
        q = " ".join(str(i) for i in self.qubits)
        return f"{self.name}{p} {q}"


@dataclass
class Circuit:
    num_qubits: int
    gates: List[Gate] = field(default_factory=list)

    def add(self, name: str, qubits: Sequence[int], params: Sequence[float] = ()) -> "Circuit":
        g = Gate(name, tuple(qubits), tuple(float(x) for x in params))
        _validate_gate(g, self.num_qubits)
        self.gates.append(g)
        return self

    def __len__(self) -> int:
        return len(self.gates)

    def __iter__(self):
        return iter(self.gates)

    def __str__(self) -> str:
        head = f"// circuit: {self.num_qubits} logical qubits, {len(self.gates)} gates"
        return "\n".join([head] + [str(g) for g in self.gates])


def _validate_gate(g: Gate, num_qubits: int) -> None:
    if g.name not in ONE_Q_GATES and g.name not in TWO_Q_GATES:
        raise ValueError(f"unsupported gate: {g.name!r}")
    arity = 2 if g.is_two_qubit else 1
    if len(g.qubits) != arity:
        raise ValueError(f"gate {g.name!r} expects {arity} qubit(s), got {g.qubits}")
    for q in g.qubits:
        if not (0 <= q < num_qubits):
            raise ValueError(f"qubit index {q} out of range [0,{num_qubits})")
    if g.is_two_qubit and g.qubits[0] == g.qubits[1]:
        raise ValueError(f"two-qubit gate {g.name!r} needs distinct qubits: {g.qubits}")
    if g.name in PARAM_GATES and len(g.params) != 1:
        raise ValueError(f"gate {g.name!r} expects one angle parameter")
    if g.name == U_GATE and len(g.params) != 3:
        raise ValueError("gate 'u' expects three angle parameters (theta, phi, lam)")


def _tokenize_line(line: str) -> str:
    # strip comments
    for marker in ("//", "#"):
        idx = line.find(marker)
        if idx != -1:
            line = line[:idx]
    return line.strip().rstrip(";").strip()


def _parse_qubit_token(tok: str) -> int:
    tok = tok.strip()
    if "[" in tok and tok.endswith("]"):
        # OpenQASM form  q[3]
        return int(tok[tok.index("[") + 1 : -1])
    return int(tok)


def parse_circuit(text: str) -> Circuit:
    """Parse a QASM-lite / OpenQASM-2 subset into a :class:`Circuit`.

    Accepted forms (mix freely)::

        OPENQASM 2.0;          // ignored
        include "qelib1.inc";  // ignored
        qreg q[3];             // or  'qubits 3'
        h q[0];                // or  'h 0'
        cx q[0],q[1];          // or  'cx 0 1'
        rz(0.7853) q[0];       // or  'rz 0 0.7853'
        cxswap q[0],q[2];      // EO-native gate

    The parser is intentionally permissive; it raises ValueError on anything it
    does not understand so problems surface early.
    """

    num_qubits = None
    gates: List[Gate] = []

    for raw in text.splitlines():
        line = _tokenize_line(raw)
        if not line:
            continue
        low = line.lower()
        # ignore headers, register/classical declarations, and non-gate statements
        first = low.split()[0].split("(")[0]
        if first in ("openqasm", "include", "creg", "opaque", "gate", "barrier",
                     "measure", "reset"):
            continue
        if low.startswith("qreg"):
            # qreg q[3]
            num_qubits = _parse_qubit_token(line.split()[1])
            continue
        if low.startswith("qubits"):
            num_qubits = int(line.split()[1])
            continue

        # gate line: split into "head args"
        head, _, rest = line.partition(" ")
        name = head
        params: List[float] = []
        if "(" in head:
            # OpenQASM form: rz(0.7853) q[0]
            name = head[: head.index("(")]
            arg = head[head.index("(") + 1 : head.rindex(")")]
            params = [float(x) for x in arg.split(",") if x.strip()]
        name = name.lower()
        toks = [t for t in rest.replace(",", " ").split() if t.strip()]

        if name in PARAM_GATES and not params:
            # QASM-lite parametrised form: 'rz 0 0.7853' -> last token is the angle
            params = [float(toks.pop())]
        elif name == U_GATE and not params:
            # QASM-lite form: 'u 0 theta phi lam' -> first token qubit, then 3 angles
            params = [float(t) for t in toks[1:4]]
            toks = toks[:1]

        qubits = [_parse_qubit_token(t) for t in toks]
        gates.append(Gate(name, tuple(qubits), tuple(params)))

    if num_qubits is None:
        # infer from gates
        max_q = max((q for g in gates for q in g.qubits), default=-1)
        num_qubits = max_q + 1

    circ = Circuit(num_qubits)
    for g in gates:
        _validate_gate(g, num_qubits)
        circ.gates.append(g)
    return circ


def parse_circuit_file(path: str) -> Circuit:
    with open(path, "r", encoding="utf-8") as fh:
        return parse_circuit(fh.read())
