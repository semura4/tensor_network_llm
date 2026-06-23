"""Output artefacts: instruction/pattern memory CSVs and the cost report."""

from __future__ import annotations

import csv
from typing import List

from .hardware import HardwareConfig, MemoryReport
from .metrics import CostMetrics


def write_instruction_memory_csv(mem: MemoryReport, path: str) -> None:
    fields = ["addr", "sequencer", "output", "op", "pattern_id",
              "t_start_samples", "width_samples", "gate"]
    _write_csv(path, fields, mem.instruction_memory)


def write_pattern_memory_csv(mem: MemoryReport, path: str) -> None:
    fields = ["pattern_id", "output", "sequencer", "dac_code", "voltage",
              "width_samples", "exchange_area"]
    _write_csv(path, fields, mem.pattern_memory)


def _write_csv(path: str, fields: List[str], rows: List[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def _bar(frac: float, width: int = 20) -> str:
    frac = max(0.0, min(1.0, frac))
    filled = int(round(frac * width))
    return "#" * filled + "-" * (width - filled)


def render_report(metrics: CostMetrics, mem: MemoryReport, hw: HardwareConfig,
                  title: str = "EO Pulse Control IR — Control Cost Report") -> str:
    m = metrics
    flag_i = "  ⚠ OVERFLOW" if mem.instruction_overflow else ""
    flag_p = "  ⚠ OVERFLOW" if mem.pattern_overflow else ""
    flag_c = "  ⚠" if m.sequencer_conflict else ""

    lines = [
        f"# {title}",
        "",
        "## Schedule",
        "",
        "| metric | value |",
        "|---|---|",
        f"| pulse_count | {m.pulse_count} |",
        f"| boundary_pulse_count (inter/route) | {m.boundary_pulse_count} |",
        f"| total_time (ns) | {m.total_time:g} |",
        f"| critical_path_pulses | {m.critical_path_pulses} |",
        f"| peak_parallelism | {m.peak_parallelism:g} |",
        f"| avg_parallelism | {m.avg_parallelism:g} |",
        f"| idle_time (dot·ns) | {m.idle_time:g} |",
        f"| idle_fraction | {m.idle_fraction:.1%} |",
        "",
        "## Heuristic physics proxies",
        "",
        "_Ranking proxies, not simulated fidelities — see `metrics.py`._",
        "",
        "| metric | value |",
        "|---|---|",
        f"| estimated_leakage_risk | {m.estimated_leakage_risk:.4g} |",
        f"| noise_sensitivity (rad / unit V) | {m.noise_sensitivity:g} |",
        "",
        "## Cryo-CMOS controller budget",
        "",
        f"- instruction memory: **{m.instruction_memory_usage} / "
        f"{hw.instruction_memory_words}** words "
        f"`[{_bar(m.instruction_memory_fraction)}]` "
        f"{m.instruction_memory_fraction:.1%}{flag_i}",
        f"- pattern memory (busiest output): **{m.pattern_memory_usage_max} / "
        f"{hw.pattern_memory_words_per_output}** words "
        f"`[{_bar(m.pattern_memory_fraction)}]` "
        f"{m.pattern_memory_fraction:.1%}{flag_p}",
        f"- sequencer_conflict: **{m.sequencer_conflict}**{flag_c} "
        f"(controller has {hw.num_sequencers} sequencers)",
        "",
        "## Pattern memory per output",
        "",
        "| output | pattern words |",
        "|---|---|",
    ]
    for out, n in sorted(mem.pattern_words_per_output.items()):
        lines.append(f"| {out} | {n} |")
    lines.append("")
    return "\n".join(lines)


def write_report(metrics: CostMetrics, mem: MemoryReport, hw: HardwareConfig,
                 path: str, title: str = "EO Pulse Control IR — Control Cost Report") -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(render_report(metrics, mem, hw, title))
