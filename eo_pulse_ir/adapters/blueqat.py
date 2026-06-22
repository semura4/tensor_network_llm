"""Blueqat-style front-end adapter.

Converts a gate list (or, optionally, a Blueqat ``Circuit``) into our
:class:`~eo_pulse_ir.circuit.Circuit`. The gate-list form is the stable contract;
the direct Blueqat import is optional and guarded so this module needs no
third-party packages.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

from ..circuit import Circuit

# Blueqat / common aliases -> our gate names
_ALIAS = {"cnot": "cx", "ccx": None, "toffoli": None}


def from_gate_list(gate_list: Iterable[Tuple], num_qubits: Optional[int] = None
                   ) -> Circuit:
    """Build a Circuit from ``[(name, (qubits...), (params...)), ...]``.

    ``name`` is a gate name (Blueqat/QASM aliases accepted); ``qubits`` a tuple of
    indices; ``params`` an optional tuple of angles.
    """
    items = []
    max_q = -1
    for entry in gate_list:
        name = entry[0].lower()
        qubits = tuple(entry[1]) if len(entry) > 1 else ()
        params = tuple(entry[2]) if len(entry) > 2 else ()
        name = _ALIAS.get(name, name)
        if name is None:
            raise ValueError(f"gate {entry[0]!r} not supported by the EO front end")
        items.append((name, qubits, params))
        max_q = max([max_q, *qubits]) if qubits else max_q
    n = num_qubits if num_qubits is not None else max_q + 1
    circ = Circuit(n)
    for name, qubits, params in items:
        circ.add(name, qubits, params)
    return circ


def from_blueqat(circuit) -> Circuit:
    """Convert a Blueqat ``Circuit`` to ours (best-effort; requires blueqat).

    Blueqat stores operations on ``circuit.ops``; each op exposes a name, target
    qubits and parameters. We map the common gate set; unsupported gates raise.
    """
    try:
        ops = circuit.ops
    except AttributeError as e:                      # pragma: no cover
        raise TypeError("expected a Blueqat Circuit with an `.ops` attribute") from e
    n = getattr(circuit, "n_qubits", None)
    gate_list = []
    for op in ops:                                   # pragma: no cover (needs blueqat)
        name = getattr(op, "name", None) or op[0]
        targets = getattr(op, "targets", None)
        params = getattr(op, "params", ()) or ()
        if targets is None:
            targets = op[1]
        gate_list.append((name, tuple(targets), tuple(params)))
    return from_gate_list(gate_list, n)
