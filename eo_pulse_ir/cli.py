"""Command-line entry point for the EO Pulse Control IR.

Examples::

    python -m eo_pulse_ir.cli examples/bell.qasm -o out/bell
    python -m eo_pulse_ir.cli examples/ghz.qasm -o out/ghz --j-max 1.0
    python -m eo_pulse_ir.cli --pulses optimiser_out.json --num-dots 9 -o out/ext
"""

from __future__ import annotations

import argparse
import json
import sys

from .circuit import parse_circuit_file
from .hardware import HardwareConfig
from .pipeline import compile_circuit, compile_pulse_records, emit_artifacts


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="eo_pulse_ir",
        description="Compile an EO logical circuit to exchange pulses, schedule, "
                    "score and emit cryo-CMOS controller memory images.")
    p.add_argument("circuit", nargs="?", help="QASM-lite / OpenQASM-2 circuit file")
    p.add_argument("-o", "--out-dir", default="eo_out", help="output directory")
    p.add_argument("--title", default=None, help="title for artefacts")
    p.add_argument("--pulses", help="JSON file of external pulse records (bypasses synthesis)")
    p.add_argument("--num-dots", type=int, help="dot count (required with --pulses)")
    # hardware knobs
    p.add_argument("--j-max", type=float, default=1.0, help="max exchange amplitude (rad/ns)")
    p.add_argument("--num-sequencers", type=int, default=6)
    p.add_argument("--instruction-words", type=int, default=6144)
    p.add_argument("--pattern-words", type=int, default=512)
    p.add_argument("--no-symmetric", action="store_true",
                   help="disable sweet-spot noise suppression in the heuristic")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    hw = HardwareConfig(
        j_max=args.j_max,
        num_sequencers=args.num_sequencers,
        instruction_memory_words=args.instruction_words,
        pattern_memory_words_per_output=args.pattern_words,
        symmetric=not args.no_symmetric,
    )

    if args.pulses:
        if args.num_dots is None:
            print("error: --num-dots is required with --pulses", file=sys.stderr)
            return 2
        with open(args.pulses, "r", encoding="utf-8") as fh:
            records = json.load(fh)
        if isinstance(records, dict):
            records = records.get("pulses", [])
        result = compile_pulse_records(records, num_dots=args.num_dots, hw=hw)
        title = args.title or "external-pulses"
    else:
        if not args.circuit:
            print("error: provide a circuit file or --pulses", file=sys.stderr)
            return 2
        circ = parse_circuit_file(args.circuit)
        result = compile_circuit(circ, hw=hw)
        title = args.title or args.circuit.rsplit("/", 1)[-1].rsplit(".", 1)[0]

    written = emit_artifacts(result, args.out_dir, title=title)

    m = result.metrics
    print(f"[eo_pulse_ir] {title}: {m.pulse_count} pulses, "
          f"makespan={m.total_time:g} ns, peak_parallelism={m.peak_parallelism:g}, "
          f"leakage~{m.estimated_leakage_risk:.3g}, "
          f"instr={m.instruction_memory_usage}/{hw.instruction_memory_words}, "
          f"seq_conflicts={m.sequencer_conflict}")
    print("[eo_pulse_ir] wrote:")
    for w in written:
        print(f"  - {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
