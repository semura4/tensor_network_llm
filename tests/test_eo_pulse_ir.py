"""Tests for the EO Pulse Control IR (standard library only)."""

import json
import math
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eo_pulse_ir import (HardwareConfig, compile_circuit, compile_pulse_records,
                         emit_artifacts, parse_circuit, render_svg)
from eo_pulse_ir.native import one_qubit_template, two_qubit_template
from eo_pulse_ir.schedule import Pulse, schedule_pulses
from eo_pulse_ir.topology import LinearTopology


class TestCircuitParsing(unittest.TestCase):
    def test_qasm_lite(self):
        c = parse_circuit("qubits 2\nh 0\ncx 0 1\n")
        self.assertEqual(c.num_qubits, 2)
        self.assertEqual([g.name for g in c.gates], ["h", "cx"])
        self.assertEqual(c.gates[1].qubits, (0, 1))

    def test_openqasm_subset(self):
        c = parse_circuit(
            'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[3];\n'
            "rz(0.5) q[0];\ncx q[0],q[1];\n")
        self.assertEqual(c.num_qubits, 3)
        self.assertEqual(c.gates[0].name, "rz")
        self.assertAlmostEqual(c.gates[0].params[0], 0.5)
        self.assertEqual(c.gates[1].qubits, (0, 1))

    def test_qasm_lite_param(self):
        c = parse_circuit("qubits 1\nrz 0 0.7853981633974483\n")
        self.assertAlmostEqual(c.gates[0].params[0], math.pi / 4)

    def test_cnot_alias_and_validation(self):
        c = parse_circuit("qubits 2\ncnot 0 1\n")
        self.assertEqual(c.gates[0].name, "cx")
        with self.assertRaises(ValueError):
            parse_circuit("qubits 1\ncx 0 0\n")
        with self.assertRaises(ValueError):
            parse_circuit("qubits 1\nfoo 0\n")
        with self.assertRaises(ValueError):
            parse_circuit("qubits 1\nh 5\n")


class TestTemplates(unittest.TestCase):
    def test_one_qubit_counts(self):
        self.assertEqual(len(one_qubit_template("z")), 1)
        self.assertEqual(len(one_qubit_template("rz", math.pi / 2)), 1)
        self.assertEqual(len(one_qubit_template("h")), 3)
        self.assertEqual(len(one_qubit_template("rx", 0.3)), 3)
        # parametrised area flows through
        self.assertAlmostEqual(one_qubit_template("rz", 1.234)[0][1], 1.234)
        self.assertAlmostEqual(one_qubit_template("rx", 0.3)[1][1], 0.3)

    def test_two_qubit_counts(self):
        self.assertEqual(len(two_qubit_template("cx")), 34)
        self.assertEqual(len(two_qubit_template("swap")), 7)
        self.assertEqual(len(two_qubit_template("cxswap")), 13)


class TestTopologyRouting(unittest.TestCase):
    def test_dot_mapping(self):
        t = LinearTopology(3)
        self.assertEqual(t.num_dots, 9)
        self.assertEqual(t.dots_of_qubit(0), (0, 1, 2))
        self.assertEqual(t.intra_edges(1), ((3, 4), (4, 5)))
        self.assertEqual(t.inter_edge(0), (2, 3))

    def test_route_makes_adjacent(self):
        t = LinearTopology(4)
        swaps = t.route_adjacent(0, 3)
        for (a, b) in swaps:
            t.swap_positions(a)
        self.assertEqual(t.position_of[3] - t.position_of[0], 1)

    def test_route_reverse_order(self):
        t = LinearTopology(4)
        swaps = t.route_adjacent(3, 0)  # control to the right of target
        for (a, b) in swaps:
            t.swap_positions(a)
        # control must end up immediately left of target
        self.assertEqual(t.position_of[0] - t.position_of[3], 1)


class TestScheduling(unittest.TestCase):
    def test_serialisation_on_shared_dot(self):
        pulses = [
            Pulse(edge=(0, 1), area=math.pi, gate="a", logical_qubits=(0,), role="intra_low"),
            Pulse(edge=(1, 2), area=math.pi, gate="b", logical_qubits=(0,), role="intra_high"),
        ]
        s = schedule_pulses(pulses, num_dots=3, j_max=1.0)
        # share dot 1 -> must serialise
        self.assertGreaterEqual(s.pulses[1].start, s.pulses[0].end - 1e-12)
        self.assertAlmostEqual(s.makespan, 2 * math.pi)

    def test_parallel_on_disjoint_edges(self):
        pulses = [
            Pulse(edge=(0, 1), area=math.pi, gate="a", logical_qubits=(0,), role="intra_low"),
            Pulse(edge=(3, 4), area=math.pi, gate="b", logical_qubits=(1,), role="intra_low"),
        ]
        s = schedule_pulses(pulses, num_dots=6, j_max=1.0)
        self.assertAlmostEqual(s.pulses[0].start, 0.0)
        self.assertAlmostEqual(s.pulses[1].start, 0.0)
        self.assertAlmostEqual(s.makespan, math.pi)

    def test_duration_scales_with_jmax(self):
        pulses = [Pulse(edge=(0, 1), area=math.pi, gate="a", logical_qubits=(0,), role="intra_low")]
        s = schedule_pulses(pulses, num_dots=2, j_max=2.0)
        self.assertAlmostEqual(s.pulses[0].duration, math.pi / 2)


class TestPipeline(unittest.TestCase):
    def test_bell_end_to_end(self):
        c = parse_circuit("qubits 2\nh 0\ncx 0 1\n")
        r = compile_circuit(c)
        # h = 3 pulses, cx = 34 pulses (validated template), no routing (adjacent)
        self.assertEqual(r.metrics.pulse_count, 3 + 34)
        self.assertGreater(r.metrics.total_time, 0)
        self.assertGreaterEqual(r.metrics.critical_path_pulses, 1)
        self.assertEqual(r.schedule.num_dots, 6)
        self.assertGreaterEqual(r.metrics.boundary_pulse_count, 1)

    def test_routing_adds_pulses(self):
        adj = compile_circuit(parse_circuit("qubits 4\ncx 0 1\n"))
        far = compile_circuit(parse_circuit("qubits 4\ncx 0 3\n"))
        # the far CX requires routing SWAPs -> strictly more pulses
        self.assertGreater(far.metrics.pulse_count, adj.metrics.pulse_count)

    def test_leakage_and_noise_in_range(self):
        r = compile_circuit(parse_circuit("qubits 2\nh 0\ncx 0 1\n"))
        self.assertTrue(0.0 <= r.metrics.estimated_leakage_risk < 1.0)
        self.assertGreater(r.metrics.noise_sensitivity, 0.0)

    def test_memory_overflow_flag(self):
        c = parse_circuit("qubits 2\nh 0\ncx 0 1\n")
        hw = HardwareConfig(instruction_memory_words=4)  # tiny -> overflow
        r = compile_circuit(c, hw=hw)
        self.assertTrue(r.memory.instruction_overflow)
        self.assertGreater(r.metrics.instruction_memory_fraction, 1.0)

    def test_external_pulses(self):
        with open(os.path.join(os.path.dirname(__file__), "..", "examples",
                               "external_pulses.json")) as fh:
            data = json.load(fh)
        r = compile_pulse_records(data["pulses"], num_dots=data["num_dots"])
        self.assertEqual(r.metrics.pulse_count, 5)
        self.assertGreater(r.metrics.total_time, 0)

    def test_emit_artifacts(self):
        c = parse_circuit("qubits 2\nh 0\ncx 0 1\n")
        r = compile_circuit(c)
        with tempfile.TemporaryDirectory() as d:
            written = emit_artifacts(r, d, title="bell")
            for w in written:
                self.assertTrue(os.path.exists(w) and os.path.getsize(w) > 0)
            with open(os.path.join(d, "pulse_timeline.json")) as fh:
                tl = json.load(fh)
            self.assertEqual(tl["pulse_count"], r.metrics.pulse_count)

    def test_svg_renders(self):
        r = compile_circuit(parse_circuit("qubits 2\nh 0\ncx 0 1\n"))
        svg = render_svg(r.schedule, "test")
        self.assertTrue(svg.startswith("<svg"))
        self.assertIn("</svg>", svg)


if __name__ == "__main__":
    unittest.main()
