"""The pulse-level intermediate representation and the scheduler.

A :class:`Pulse` is one square exchange pulse on one nearest-neighbour edge.
A :class:`Schedule` is a time-resolved list of pulses plus the makespan.

Scheduling model
----------------
We use an as-soon-as-possible (ASAP) list scheduler driven by *dot
availability*: a pulse on edge (a, b) cannot start until both dots a and b are
free, and it occupies both for its duration.  This is exactly the right
constraint for exchange control (a dot can only participate in one exchange at a
time), and it lets pulses on well-separated edges run concurrently for free,
which is where parallelism in EO control comes from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class Pulse:
    edge: Tuple[int, int]            # (low_dot, high_dot), high = low + 1
    area: float                      # exchange action; full SWAP = pi
    gate: str                        # originating logical gate name
    logical_qubits: Tuple[int, ...]  # logical operands of that gate
    role: str                        # intra_low / intra_high / inter / route
    # filled in by the scheduler:
    start: Optional[float] = None
    duration: Optional[float] = None
    j: Optional[float] = None        # exchange amplitude used
    index: int = -1                  # program order

    @property
    def end(self) -> float:
        return (self.start or 0.0) + (self.duration or 0.0)

    @property
    def is_boundary(self) -> bool:
        return self.role in ("inter", "route")


@dataclass
class Schedule:
    pulses: List[Pulse]
    num_dots: int
    makespan: float = 0.0
    j_max: float = 1.0

    def used_dots(self) -> List[int]:
        s = set()
        for p in self.pulses:
            s.update(p.edge)
        return sorted(s)

    def edges(self) -> List[Tuple[int, int]]:
        seen = []
        s = set()
        for p in self.pulses:
            if p.edge not in s:
                s.add(p.edge)
                seen.append(p.edge)
        return sorted(seen)


def schedule_pulses(pulses: List[Pulse], num_dots: int, j_max: float = 1.0) -> Schedule:
    """ASAP-schedule a program-ordered list of pulses.

    Durations are set from the (constant-amplitude) hardware model: with a square
    pulse of amplitude ``j_max``, a pulse of exchange area ``A`` lasts ``A/j_max``.
    """
    avail: Dict[int, float] = {d: 0.0 for d in range(num_dots)}
    makespan = 0.0
    for i, p in enumerate(pulses):
        a, b = p.edge
        start = max(avail[a], avail[b])
        p.start = start
        p.j = j_max
        p.duration = p.area / j_max if j_max else 0.0
        p.index = i
        end = p.end
        avail[a] = end
        avail[b] = end
        makespan = max(makespan, end)
    return Schedule(pulses=pulses, num_dots=num_dots, makespan=makespan, j_max=j_max)


def _normalize_role(role) -> str:
    """Coerce an external/template role string into intra / inter / route."""
    if not role:
        return "inter"  # conservative default (counts as a boundary pulse)
    role = str(role).lower()
    if role == "inter":
        return "inter"
    if role == "route":
        return "route"
    return "intra"


def pulses_from_records(records: List[dict]) -> List[Pulse]:
    """Build pulses from external (e.g. optimiser / eoqrid) output.

    Each record is a dict with keys ``edge`` ([low, high]), ``area`` (radians),
    and optionally ``gate``, ``logical_qubits``, ``role``.  This is the
    integration seam for replacing the built-in templates with calibrated data.
    """
    out: List[Pulse] = []
    for r in records:
        edge = tuple(r["edge"])
        out.append(
            Pulse(
                edge=(int(edge[0]), int(edge[1])),
                area=float(r["area"]),
                gate=str(r.get("gate", "ext")),
                logical_qubits=tuple(r.get("logical_qubits", ())),
                role=_normalize_role(r.get("role")),
            )
        )
    return out
