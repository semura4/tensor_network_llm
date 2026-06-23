"""eoqrid adapter — convert eoqrid-transpiled circuits to our pulse IR.

eoqrid (https://github.com/samn33/eoqrid) transpiles logical quantum circuits
into exchange interactions (``Ex`` gates) for EO silicon qubits.  Its output is
a Qiskit ``QuantumCircuit`` whose gates are ``Ex(param0, param1)`` acting on
physical qubit indices.

This adapter extracts those ``Ex`` gates and converts them to the pulse-record
format consumed by :func:`~eo_pulse_ir.adapters.external.pulse_records_to_result`
and the rest of the IR pipeline.

Requires Qiskit only at call time (guarded import), so this module loads
without third-party packages.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence

from ..pipeline import CompileResult, compile_pulse_records


def _default_param_to_area(params: Sequence) -> float:
    """Default parameter → exchange-area mapping.

    eoqrid's ``Ex`` gate carries two parameters.  The first is treated as the
    exchange area A = ∫J dt in radians (consistent with the standard EO
    convention where a full SWAP is A = π).  Override via ``param_to_area``
    if your eoqrid version uses a different convention.
    """
    return float(params[0])


def from_eoqrid(qc_native,
                param_to_area: Optional[Callable] = None,
                gate_name: str = "ex") -> List[dict]:
    """Convert an eoqrid-transpiled ``QuantumCircuit`` to pulse records.

    Parameters
    ----------
    qc_native
        Output of ``EoqSimulator().transpile(qc)`` — a Qiskit
        ``QuantumCircuit`` containing ``Ex`` gates.
    param_to_area : callable, optional
        ``f(params) -> float`` mapping gate parameters to exchange area in
        radians.  Defaults to ``params[0]``.
    gate_name : str
        Name of the exchange gate in the circuit (default ``"ex"``).

    Returns
    -------
    list[dict]
        Pulse records (``edge``, ``area``, ``gate``, ``role``) compatible with
        :func:`~eo_pulse_ir.adapters.external.pulse_records_to_result`.
    """
    if param_to_area is None:
        param_to_area = _default_param_to_area

    records: List[dict] = []

    for instruction in qc_native.data:
        # Qiskit 1.x: CircuitInstruction with .operation / .qubits
        # Older Qiskit: tuple (gate, qargs, cargs)
        if hasattr(instruction, "operation"):
            gate = instruction.operation
            qargs = instruction.qubits
        else:
            gate, qargs, _ = instruction

        name = getattr(gate, "name", "").lower()
        if name != gate_name.lower():
            continue

        area = param_to_area(gate.params)

        indices: List[int] = []
        for q in qargs:
            if hasattr(q, "_index"):
                indices.append(q._index)
            elif hasattr(qc_native, "find_bit"):
                indices.append(qc_native.find_bit(q).index)
            else:
                indices.append(int(q))

        if len(indices) < 2:
            continue

        q0, q1 = indices[0], indices[1]
        records.append({
            "edge": [min(q0, q1), max(q0, q1)],
            "area": area,
            "gate": "ex",
            "role": "intra",
        })

    return records


def compile_eoqrid(qc_native, num_dots: Optional[int] = None,
                   hw=None, param_to_area: Optional[Callable] = None,
                   gate_name: str = "ex") -> CompileResult:
    """Convert an eoqrid-transpiled circuit and compile it in one step.

    Parameters
    ----------
    qc_native
        Output of ``EoqSimulator().transpile(qc)``.
    num_dots : int, optional
        Total physical dots.  Inferred from the circuit if omitted.
    hw : HardwareConfig, optional
        Hardware model for cost evaluation.
    param_to_area, gate_name
        Forwarded to :func:`from_eoqrid`.

    Returns
    -------
    CompileResult
        Full compilation result (schedule, metrics, memory).
    """
    records = from_eoqrid(qc_native, param_to_area=param_to_area,
                          gate_name=gate_name)
    if num_dots is None:
        max_q = max((max(r["edge"]) for r in records), default=0)
        num_dots = max_q + 1
    return compile_pulse_records(records, num_dots=num_dots, hw=hw)
