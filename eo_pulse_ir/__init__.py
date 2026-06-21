"""EO Pulse Control IR.

A dependency-free intermediate representation and control-cost evaluator that
sits between exchange-only (EO) gate synthesis and an HRL-style cryo-CMOS
controller.  It takes a logical circuit (QASM-lite / OpenQASM-2 subset),
synthesises exchange pulses, schedules them on a quantum-dot line, scores the
schedule (timing, parallelism, idle, heuristic leakage/noise proxies), and emits
instruction/pattern memory images, a pulse-timeline visualisation and a cost
report.

Quick start::

    from eo_pulse_ir import parse_circuit, compile_circuit, emit_artifacts
    circ = parse_circuit("qubits 2\\nh 0\\ncx 0 1\\n")
    result = compile_circuit(circ)
    emit_artifacts(result, "out/", title="bell")
    print(result.metrics.as_dict())
"""

from .circuit import Circuit, Gate, parse_circuit, parse_circuit_file
from .compile import synthesize
from .hardware import HardwareConfig, MemoryReport, build_memory
from .metrics import CostMetrics, compute_metrics
from .pipeline import (CompileResult, compile_circuit, compile_pulse_records,
                       emit_artifacts)
from .report import render_report
from .schedule import Pulse, Schedule, pulses_from_records, schedule_pulses
from .topology import LinearTopology
from .visualize import render_svg, timeline_records, write_svg, write_timeline_json

__version__ = "0.1.0"

__all__ = [
    "Circuit", "Gate", "parse_circuit", "parse_circuit_file",
    "synthesize", "LinearTopology",
    "Pulse", "Schedule", "schedule_pulses", "pulses_from_records",
    "HardwareConfig", "MemoryReport", "build_memory",
    "CostMetrics", "compute_metrics",
    "CompileResult", "compile_circuit", "compile_pulse_records", "emit_artifacts",
    "render_report", "render_svg", "write_svg", "write_timeline_json",
    "timeline_records",
    "__version__",
]
