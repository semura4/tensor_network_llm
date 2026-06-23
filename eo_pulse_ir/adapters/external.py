"""External pulse-record adapter (optimiser / eoqrid seam).

See docs/IR_SPEC.md section 4. A pulse record needs at least ``edge`` and
``area``; ``gate``/``role``/``logical_qubits`` are optional.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from ..pipeline import CompileResult, compile_pulse_records
from ..visualize import timeline_records


def pulse_records_to_result(records: Sequence[dict], num_dots: int,
                            hw=None) -> CompileResult:
    """Compile external pulse records (e.g. eoqrid / optimiser output) into the IR.

    ``records`` is a list of dicts with ``edge`` [low,high] and ``area`` (radians).
    Returns a full :class:`CompileResult` (schedule, metrics, memory) so the whole
    pipeline runs on externally-supplied pulses.
    """
    return compile_pulse_records(list(records), num_dots=num_dots, hw=hw)


def ir_to_pulse_records(result: CompileResult) -> List[dict]:
    """Export a compiled result back to the canonical pulse-record list."""
    return timeline_records(result.schedule)
