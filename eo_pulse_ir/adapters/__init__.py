"""Adapters connecting the EO Pulse Control IR to external formats and SDKs.

These are the seams that let other tools produce/consume the IR (see
docs/IR_SPEC.md):

- :mod:`qasm`     — OpenQASM-2 subset in/out (round-trip with the front end);
- :mod:`external` — external optimiser / eoqrid pulse records <-> IR;
- :mod:`blueqat`  — Blueqat-style gate lists / Circuit -> our Circuit.

All adapters are standard-library only (the optional Blueqat import is guarded).
"""

from .blueqat import from_blueqat, from_gate_list
from .eoqrid import compile_eoqrid, from_eoqrid
from .external import ir_to_pulse_records, pulse_records_to_result
from .qasm import from_openqasm, to_openqasm

__all__ = [
    "to_openqasm", "from_openqasm",
    "pulse_records_to_result", "ir_to_pulse_records",
    "from_gate_list", "from_blueqat",
    "from_eoqrid", "compile_eoqrid",
]
