"""Control-cost metrics computed from a scheduled pulse list.

These are the quantities the IR exists to expose.  Two of them
(``estimated_leakage_risk`` and ``noise_sensitivity``) are explicitly *heuristic
proxies*, not simulated physical fidelities — they rank schedules, they do not
predict experiment.  The rest (counts, timing, parallelism, idle, memory usage)
are exact functions of the schedule and hardware model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Dict

from .hardware import HardwareConfig, build_memory
from .schedule import Schedule

# per-role weighting for the leakage heuristic
_LEAK_WEIGHT = {"inter": 1.0, "route": 0.7, "intra": 0.2}


@dataclass
class CostMetrics:
    pulse_count: int
    boundary_pulse_count: int
    total_time: float
    critical_path_pulses: int
    peak_parallelism: float
    avg_parallelism: float
    idle_time: float
    idle_fraction: float
    estimated_leakage_risk: float    # heuristic probability proxy in [0,1)
    noise_sensitivity: float         # heuristic RMS phase sensitivity (rad)
    instruction_memory_usage: int
    instruction_memory_fraction: float
    pattern_memory_usage_max: int
    pattern_memory_fraction: float
    sequencer_conflict: int

    def as_dict(self) -> Dict:
        return asdict(self)


def _critical_path_pulses(schedule: Schedule) -> int:
    """Longest chain of pulses sharing dots (depth in 'pulses')."""
    last_depth: Dict[int, int] = {}
    best = 0
    for p in sorted(schedule.pulses, key=lambda x: (x.start or 0.0, x.index)):
        a, b = p.edge
        d = 1 + max(last_depth.get(a, 0), last_depth.get(b, 0))
        last_depth[a] = last_depth[b] = d
        best = max(best, d)
    return best


def _parallelism(schedule: Schedule):
    """Peak and time-averaged number of simultaneously active pulses."""
    events = []
    busy_area = 0.0
    for p in schedule.pulses:
        events.append((p.start or 0.0, +1))
        events.append((p.end, -1))
        busy_area += (p.duration or 0.0)
    events.sort(key=lambda e: (e[0], -e[1]))
    cur = peak = 0
    for _, delta in events:
        cur += delta
        peak = max(peak, cur)
    avg = busy_area / schedule.makespan if schedule.makespan > 0 else 0.0
    return float(peak), float(avg)


def _idle(schedule: Schedule):
    used = schedule.used_dots()
    busy: Dict[int, float] = {d: 0.0 for d in used}
    for p in schedule.pulses:
        for d in p.edge:
            busy[d] += (p.duration or 0.0)
    total_idle = sum(schedule.makespan - busy[d] for d in used)
    denom = schedule.makespan * len(used) if used and schedule.makespan > 0 else 1.0
    return total_idle, (total_idle / denom if denom else 0.0)


def _leakage(schedule: Schedule, hw: HardwareConfig) -> float:
    """Heuristic leakage-risk proxy.

    Each pulse contributes an independent per-pulse leakage probability scaled by
    its role weight (boundary/inter pulses leak more) and by how far its area is
    from a 'closed' value (a partial exchange leaves population mid-rotation).
    Combined as 1 - prod(1 - p_i).
    """
    surv = 1.0
    for p in schedule.pulses:
        w = _LEAK_WEIGHT.get(p.role, 0.5)
        # openness: 0 at full/zero SWAP (area multiple of pi), 1 at half SWAP
        openness = abs(math.sin(p.area)) ** 2
        pi = hw.base_leak_rate * w * (0.5 + 0.5 * openness)
        surv *= (1.0 - min(pi, 0.999))
    return 1.0 - surv


def _noise_sensitivity(schedule: Schedule, hw: HardwareConfig) -> float:
    """Heuristic charge-noise sensitivity.

    With J(V)=j0 exp((V-v0)/vc), a relative exchange fluctuation dJ/J = dV/vc maps
    to a pulse-angle error of (area)*(dV/vc).  Summed in quadrature over pulses
    gives an RMS phase-sensitivity per unit voltage noise; symmetric operation is
    modelled as a constant suppression factor.
    """
    s2 = sum((p.area / hw.vc) ** 2 for p in schedule.pulses)
    rms = math.sqrt(s2)
    if hw.symmetric:
        rms *= 0.25
    return rms


def compute_metrics(schedule: Schedule, hw: HardwareConfig | None = None) -> CostMetrics:
    hw = hw or HardwareConfig(j_max=schedule.j_max)
    peak, avg = _parallelism(schedule)
    idle_total, idle_frac = _idle(schedule)
    mem = build_memory(schedule, hw)
    boundary = sum(1 for p in schedule.pulses if p.is_boundary)

    return CostMetrics(
        pulse_count=len(schedule.pulses),
        boundary_pulse_count=boundary,
        total_time=round(schedule.makespan, 6),
        critical_path_pulses=_critical_path_pulses(schedule),
        peak_parallelism=peak,
        avg_parallelism=round(avg, 4),
        idle_time=round(idle_total, 6),
        idle_fraction=round(idle_frac, 4),
        estimated_leakage_risk=round(_leakage(schedule, hw), 6),
        noise_sensitivity=round(_noise_sensitivity(schedule, hw), 4),
        instruction_memory_usage=mem.instruction_words_used,
        instruction_memory_fraction=round(
            mem.instruction_words_used / mem.instruction_words_capacity, 4),
        pattern_memory_usage_max=mem.max_pattern_words,
        pattern_memory_fraction=round(
            mem.max_pattern_words / mem.pattern_words_capacity, 4),
        sequencer_conflict=mem.sequencer_conflicts,
    )
