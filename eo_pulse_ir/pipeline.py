"""End-to-end pipeline: circuit -> pulses -> schedule -> metrics -> artefacts."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Optional

from .circuit import Circuit
from .compile import synthesize
from .hardware import HardwareConfig, MemoryReport, build_memory
from .metrics import CostMetrics, compute_metrics
from .report import (render_report, write_instruction_memory_csv,
                     write_pattern_memory_csv, write_report)
from .schedule import Pulse, Schedule, pulses_from_records, schedule_pulses
from .visualize import render_svg, timeline_records, write_svg, write_timeline_json


@dataclass
class CompileResult:
    schedule: Schedule
    metrics: CostMetrics
    memory: MemoryReport
    hardware: HardwareConfig


def compile_circuit(circuit: Circuit, hw: Optional[HardwareConfig] = None) -> CompileResult:
    """Compile a logical circuit all the way to scheduled pulses + metrics."""
    hw = hw or HardwareConfig()
    pulses, _topo = synthesize(circuit)
    schedule = schedule_pulses(pulses, num_dots=circuit.num_qubits * 3, j_max=hw.j_max)
    mem = build_memory(schedule, hw)
    metrics = compute_metrics(schedule, hw)
    return CompileResult(schedule=schedule, metrics=metrics, memory=mem, hardware=hw)


def compile_pulse_records(records: List[dict], num_dots: int,
                          hw: Optional[HardwareConfig] = None) -> CompileResult:
    """Compile externally supplied pulses (e.g. optimiser / eoqrid output)."""
    hw = hw or HardwareConfig()
    pulses = pulses_from_records(records)
    schedule = schedule_pulses(pulses, num_dots=num_dots, j_max=hw.j_max)
    mem = build_memory(schedule, hw)
    metrics = compute_metrics(schedule, hw)
    return CompileResult(schedule=schedule, metrics=metrics, memory=mem, hardware=hw)


def emit_artifacts(result: CompileResult, out_dir: str, title: str = "circuit") -> List[str]:
    """Write the full artefact set to ``out_dir``; return the paths written."""
    os.makedirs(out_dir, exist_ok=True)
    paths = {
        "instruction_memory.csv": lambda p: write_instruction_memory_csv(result.memory, p),
        "pattern_memory.csv": lambda p: write_pattern_memory_csv(result.memory, p),
        "pulse_timeline.json": lambda p: write_timeline_json(result.schedule, p),
        "pulse_timeline.svg": lambda p: write_svg(result.schedule, p, f"EO pulse timeline — {title}"),
        "control_cost_report.md": lambda p: write_report(
            result.metrics, result.memory, result.hardware, p,
            f"EO Pulse Control IR — {title}"),
        "metrics.json": lambda p: _dump_json(result.metrics.as_dict(), p),
    }
    written = []
    for name, fn in paths.items():
        full = os.path.join(out_dir, name)
        fn(full)
        written.append(full)
    return written


def _dump_json(obj, path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2)
