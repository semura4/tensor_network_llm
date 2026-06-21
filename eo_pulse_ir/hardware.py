"""Cryo-CMOS controller model and the hardware-instruction back end.

The defaults follow the publicly described HRL-style cryo-CMOS controller: a
small number of instruction *sequencers*, a shared *instruction memory*, and a
per-output *pattern memory* whose words specify DAC voltages and pulse widths.
This module turns a scheduled pulse list into those memory images and reports
how much of each resource a circuit consumes.

Numbers here (sequencer count, memory depths, DAC width, the exponential J(V)
exchange-vs-voltage relation) are modelling choices; all are configurable on
:class:`HardwareConfig`.  They are representative, not a datasheet.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

from .schedule import Schedule


@dataclass
class HardwareConfig:
    # exchange / pulse model
    j_max: float = 1.0            # max exchange amplitude (rad/ns); sets pulse width
    clock_ghz: float = 1.0        # controller sample clock -> width quantum = 1/clock ns

    # cryo-CMOS controller resources (HRL-style defaults)
    num_sequencers: int = 6
    instruction_memory_words: int = 6144
    pattern_memory_words_per_output: int = 512
    words_per_pulse: int = 1      # instruction words emitted per pulse
    sync_words_per_layer: int = 1 # sync/barrier words per distinct time slot

    # DAC + exchange-vs-voltage model:  J(V) = j0 * exp((V - v0)/vc)
    dac_bits: int = 14
    v_min: float = -0.5
    v_max: float = 0.5
    j0: float = 1.0
    v0: float = 0.0
    vc: float = 0.10              # voltage lever arm of exchange (V); also noise scale

    # heuristic error model
    base_leak_rate: float = 1e-3  # per unit boundary-pulse weight
    symmetric: bool = True        # symmetric (sweet-spot) operation lowers noise sensitivity

    @property
    def dac_max_code(self) -> int:
        return (1 << self.dac_bits) - 1


def voltage_for_j(j: float, hw: HardwareConfig) -> float:
    """Barrier/detuning voltage producing exchange ``j`` under J(V)=j0 exp((V-v0)/vc)."""
    j = max(j, 1e-12)
    return hw.v0 + hw.vc * math.log(j / hw.j0)


def dac_code_for_j(j: float, hw: HardwareConfig) -> int:
    v = voltage_for_j(j, hw)
    frac = (v - hw.v_min) / (hw.v_max - hw.v_min) if hw.v_max > hw.v_min else 0.0
    code = round(frac * hw.dac_max_code)
    return max(0, min(hw.dac_max_code, code))


def width_code(duration: float, hw: HardwareConfig) -> int:
    return max(1, round(duration * hw.clock_ghz))


def edge_name(edge: Tuple[int, int]) -> str:
    return f"J{edge[0]}_{edge[1]}"


def assign_sequencer(edge: Tuple[int, int], num_dots: int, hw: HardwareConfig) -> int:
    """Map an exchange-output channel to one of the controller's sequencers.

    Contiguous regions of the dot line are assigned to successive sequencers,
    mirroring how a tiled controller fans out to a 1-D/2-D dot array.
    """
    span = max(num_dots - 1, 1)
    seq = edge[0] * hw.num_sequencers // span
    return max(0, min(hw.num_sequencers - 1, seq))


@dataclass
class MemoryReport:
    instruction_words_used: int
    instruction_words_capacity: int
    pattern_words_per_output: Dict[str, int]
    pattern_words_capacity: int
    sequencer_conflicts: int
    pattern_memory: List[dict]
    instruction_memory: List[dict]

    @property
    def instruction_overflow(self) -> bool:
        return self.instruction_words_used > self.instruction_words_capacity

    @property
    def pattern_overflow(self) -> bool:
        return any(v > self.pattern_words_capacity for v in self.pattern_words_per_output.values())

    @property
    def max_pattern_words(self) -> int:
        return max(self.pattern_words_per_output.values(), default=0)


def build_memory(schedule: Schedule, hw: HardwareConfig) -> MemoryReport:
    """Produce instruction-memory and pattern-memory images plus usage stats."""
    num_dots = schedule.num_dots

    # ---- pattern memory: distinct (output, j, width) settings, per output ----
    pattern_index: Dict[Tuple[str, int, int], int] = {}
    pattern_memory: List[dict] = []
    per_output_patterns: Dict[str, set] = {}

    for p in schedule.pulses:
        name = edge_name(p.edge)
        dcode = dac_code_for_j(p.j or hw.j_max, hw)
        wcode = width_code(p.duration or 0.0, hw)
        key = (name, dcode, wcode)
        per_output_patterns.setdefault(name, set()).add((dcode, wcode))
        if key not in pattern_index:
            pattern_index[key] = len(pattern_memory)
            pattern_memory.append({
                "pattern_id": len(pattern_memory),
                "output": name,
                "sequencer": assign_sequencer(p.edge, num_dots, hw),
                "dac_code": dcode,
                "voltage": round(voltage_for_j(p.j or hw.j_max, hw), 6),
                "width_samples": wcode,
                "exchange_area": round(p.area, 6),
            })

    pattern_words_per_output = {k: len(v) for k, v in per_output_patterns.items()}

    # ---- instruction memory: one instruction per pulse + per-slot sync ----
    distinct_starts = sorted({round(p.start or 0.0, 9) for p in schedule.pulses})
    instr_used = (len(schedule.pulses) * hw.words_per_pulse
                  + len(distinct_starts) * hw.sync_words_per_layer)

    instruction_memory: List[dict] = []
    for p in sorted(schedule.pulses, key=lambda x: (x.start or 0.0, x.edge)):
        name = edge_name(p.edge)
        dcode = dac_code_for_j(p.j or hw.j_max, hw)
        wcode = width_code(p.duration or 0.0, hw)
        instruction_memory.append({
            "addr": len(instruction_memory),
            "sequencer": assign_sequencer(p.edge, num_dots, hw),
            "output": name,
            "op": "PULSE",
            "pattern_id": pattern_index[(name, dcode, wcode)],
            "t_start_samples": width_code_start(p.start or 0.0, hw),
            "width_samples": wcode,
            "gate": p.gate,
        })

    conflicts = _sequencer_conflicts(schedule, num_dots, hw)

    return MemoryReport(
        instruction_words_used=instr_used,
        instruction_words_capacity=hw.instruction_memory_words,
        pattern_words_per_output=pattern_words_per_output,
        pattern_words_capacity=hw.pattern_memory_words_per_output,
        sequencer_conflicts=conflicts,
        pattern_memory=pattern_memory,
        instruction_memory=instruction_memory,
    )


def width_code_start(t: float, hw: HardwareConfig) -> int:
    return max(0, round(t * hw.clock_ghz))


def _sequencer_conflicts(schedule: Schedule, num_dots: int, hw: HardwareConfig) -> int:
    """Count time-overlapping pulse pairs whose outputs share a sequencer.

    A sequencer issues a single instruction stream, so two distinct outputs that
    must be driven with different patterns at the same instant cannot both be
    served by it — that is a control-resource conflict the scheduler/layout must
    avoid.
    """
    by_seq: Dict[int, List] = {}
    for p in schedule.pulses:
        seq = assign_sequencer(p.edge, num_dots, hw)
        by_seq.setdefault(seq, []).append(p)

    conflicts = 0
    for plist in by_seq.values():
        plist.sort(key=lambda x: x.start or 0.0)
        for i in range(len(plist)):
            pi = plist[i]
            for j in range(i + 1, len(plist)):
                pj = plist[j]
                if (pj.start or 0.0) >= pi.end:
                    break  # sorted by start; no further overlap with pi
                if pi.edge == pj.edge:
                    continue  # same output, serialised already -> not a conflict
                # overlapping in time, different outputs, same sequencer
                if (pj.start or 0.0) < pi.end and (pi.start or 0.0) < pj.end:
                    conflicts += 1
    return conflicts
