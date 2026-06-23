"""Unified command-line app for the EO Pulse Control IR.

Install with ``pip install -e .`` (or ``pip install -e .[sim]`` for the physics
layers) and run::

    eo compile examples/ghz.qasm -o out/ghz       # full IR pipeline + artefacts
    eo dashboard examples/ghz.qasm -o out/ghz      # self-contained HTML dashboard
    eo demo                                         # build dashboards for the examples

Without installing: ``python -m eo_pulse_ir.app <subcommand> ...``.
"""

from __future__ import annotations

import argparse
import os
import sys

from . import __version__
from .circuit import parse_circuit_file
from .dashboard import build_dashboard
from .hardware import HardwareConfig
from .pipeline import compile_circuit, emit_artifacts

_EXAMPLES = {
    "bell": "examples/bell.qasm",
    "ghz": "examples/ghz.qasm",
}


def _hw(args) -> HardwareConfig:
    return HardwareConfig(
        j_max=getattr(args, "j_max", 1.0),
        num_sequencers=getattr(args, "num_sequencers", 6),
    )


def _compile(args) -> int:
    circ = parse_circuit_file(args.circuit)
    title = args.title or os.path.splitext(os.path.basename(args.circuit))[0]
    result = compile_circuit(circ, hw=_hw(args))
    written = emit_artifacts(result, args.out_dir, title=title)
    m = result.metrics
    print(f"[eo] {title}: {m.pulse_count} pulses, makespan={m.total_time:g} ns, "
          f"peak_parallelism={m.peak_parallelism:g}, "
          f"instr={m.instruction_memory_usage}/{result.hardware.instruction_memory_words}, "
          f"seq_conflicts={m.sequencer_conflict}")
    for w in written:
        print(f"  - {w}")
    return 0


def _dashboard(args) -> int:
    circ = parse_circuit_file(args.circuit)
    title = args.title or os.path.splitext(os.path.basename(args.circuit))[0]
    result = compile_circuit(circ, hw=_hw(args))
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, "index.html")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(build_dashboard(result, title=title))
    print(f"[eo] wrote dashboard {path}  (open in a browser)")
    return 0


def _demo(args) -> int:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for name, rel in _EXAMPLES.items():
        circ = parse_circuit_file(os.path.join(root, rel))
        result = compile_circuit(circ)
        out = os.path.join(args.out_dir, name)
        os.makedirs(out, exist_ok=True)
        emit_artifacts(result, out, title=name)
        with open(os.path.join(out, "index.html"), "w", encoding="utf-8") as fh:
            fh.write(build_dashboard(result, title=name))
        print(f"[eo] {name}: artefacts + dashboard in {out}/")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="eo", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=f"eo-pulse-ir {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp):
        sp.add_argument("circuit", help="QASM-lite / OpenQASM-2 circuit file")
        sp.add_argument("-o", "--out-dir", default="eo_out")
        sp.add_argument("--title", default=None)
        sp.add_argument("--j-max", type=float, default=1.0)
        sp.add_argument("--num-sequencers", type=int, default=6)

    add_common(sub.add_parser("compile", help="compile a circuit and emit all artefacts"))
    add_common(sub.add_parser("dashboard", help="build a self-contained HTML dashboard"))
    d = sub.add_parser("demo", help="build artefacts + dashboards for the bundled examples")
    d.add_argument("-o", "--out-dir", default="eo_out")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return {"compile": _compile, "dashboard": _dashboard, "demo": _demo}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
