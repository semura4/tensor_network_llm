"""Tests for the EO Pulse Control IR (standard library only)."""

import json
import math
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eo_pulse_ir import (GridTopology, HardwareConfig, compile_circuit,
                         compile_pulse_records, emit_artifacts, parse_circuit,
                         render_svg)
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
        self.assertEqual(len(one_qubit_template("x")), 3)
        self.assertEqual(len(one_qubit_template("y")), 4)   # Y needs 4 pulses
        self.assertEqual(len(one_qubit_template("rx", 0.3)), 7)  # H Rz H
        self.assertEqual(len(one_qubit_template("ry", 0.3)), 9)  # S H Rz H Sdg
        # rz area implements Rz(theta) via A = (-theta) mod 2*pi
        self.assertAlmostEqual(one_qubit_template("rz", 1.0)[0][1],
                               (-1.0) % (2 * math.pi))

    def test_two_qubit_counts(self):
        self.assertEqual(len(two_qubit_template("cx")), 34)
        self.assertEqual(len(two_qubit_template("swap")), 27)
        self.assertEqual(len(two_qubit_template("cxswap")), 48)


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


class TestGridTopology(unittest.TestCase):
    def test_grid_dot_mapping(self):
        g = GridTopology(2, 2)
        g.assign_initial_layout([0, 1, 2, 3])
        self.assertEqual(g.num_dots, 12)
        self.assertEqual(g.dots_of_slot(0), (0, 1, 2))
        self.assertEqual(g.dots_of_slot(3), (9, 10, 11))

    def test_grid_neighbours(self):
        g = GridTopology(3, 3)
        # corner: 2 neighbours
        self.assertEqual(len(g.slot_neighbours(0)), 2)
        # center: 4 neighbours
        self.assertEqual(len(g.slot_neighbours(4)), 4)
        # edge: 3 neighbours
        self.assertEqual(len(g.slot_neighbours(1)), 3)

    def test_grid_inter_edge(self):
        g = GridTopology(2, 2)
        # slot 0 to slot 1 (horizontal): dot 2 of slot 0 to dot 0 of slot 1
        edge = g.inter_edge(0, 1)
        self.assertEqual(edge, (2, 3))
        # slot 0 to slot 2 (vertical): dot 2 of slot 0 to dot 0 of slot 2
        edge = g.inter_edge(0, 2)
        self.assertEqual(edge, (2, 6))

    def test_grid_route_already_adjacent(self):
        g = GridTopology(2, 2)
        g.assign_initial_layout([0, 1, 2, 3])
        swaps = g.route_adjacent(0, 1)
        self.assertEqual(swaps, [])

    def test_grid_route_diagonal(self):
        g = GridTopology(2, 2)
        g.assign_initial_layout([0, 1, 2, 3])
        # q0 at slot 0 (0,0), q3 at slot 3 (1,1) — diagonal, distance 2
        swaps = g.route_adjacent(0, 3)
        self.assertGreater(len(swaps), 0)
        for (a, b) in swaps:
            g.swap_slots(a, b)
        self.assertEqual(g.grid_distance(g.slot_of[0], g.slot_of[3]), 1)

    def test_initial_placement_center_hub(self):
        """Hub qubit in a star circuit should be placed at the center."""
        circ = parse_circuit("qubits 9\n" +
                             "\n".join(f"cx 0 {i}" for i in range(1, 9)))
        g = GridTopology(3, 3)
        g.assign_initial_layout(list(range(9)), circuit_gates=circ.gates)
        # center slot is 4 on a 3x3 grid; q0 should be there
        self.assertEqual(g.slot_of[0], 4)

    def test_grid_compile_bell(self):
        """2-qubit Bell on a 1x2 grid should match linear compilation."""
        circ = parse_circuit("qubits 2\nh 0\ncx 0 1\n")
        grid = GridTopology(1, 2)
        grid.assign_initial_layout([0, 1], circuit_gates=circ.gates)
        r_grid = compile_circuit(circ, topology=grid)
        r_lin = compile_circuit(circ)
        self.assertEqual(r_grid.metrics.pulse_count, r_lin.metrics.pulse_count)

    def test_grid_routing_reduces_pulses_vs_linear(self):
        """Ring circuit on a 2x2 grid should need fewer routing SWAPs."""
        circ = parse_circuit("qubits 4\ncx 0 1\ncx 1 2\ncx 2 3\ncx 3 0\n")
        grid = GridTopology(2, 2)
        grid.assign_initial_layout(list(range(4)), circuit_gates=circ.gates)
        r_grid = compile_circuit(circ, topology=grid)
        r_lin = compile_circuit(circ)
        self.assertLessEqual(r_grid.metrics.pulse_count, r_lin.metrics.pulse_count)


class TestGateLibrary(unittest.TestCase):
    def test_gate_library_substitutes_areas(self):
        """A custom gate_library overrides the 2-qubit areas (same role count)."""
        from eo_pulse_ir.native import two_qubit_template
        circ = parse_circuit("qubits 2\ncx 0 1\n")
        # build a fake "robust" library: same roles, areas all set to 1.0
        roles = [r for r, _ in two_qubit_template("cx")]
        fake = {"cx": [(r, 1.0) for r in roles]}
        r_default = compile_circuit(circ)
        r_lib = compile_circuit(circ, gate_library=fake)
        # same pulse count (role sequence unchanged)
        self.assertEqual(r_lib.metrics.pulse_count, r_default.metrics.pulse_count)
        # but the cx pulse areas are now all 1.0
        cx_areas = [p.area for p in r_lib.schedule.pulses if p.gate == "cx"]
        self.assertTrue(all(abs(a - 1.0) < 1e-12 for a in cx_areas))
        self.assertEqual(len(cx_areas), len(roles))

    def test_gate_library_affects_routing_swaps(self):
        """The library's 'swap' entry is used for routing SWAPs too."""
        from eo_pulse_ir.native import two_qubit_template
        circ = parse_circuit("qubits 4\ncx 0 3\n")  # needs routing
        roles = [r for r, _ in two_qubit_template("swap")]
        fake = {"swap": [(r, 0.5) for r in roles]}
        r_lib = compile_circuit(circ, gate_library=fake)
        route_areas = [p.area for p in r_lib.schedule.pulses if p.role == "route"]
        self.assertGreater(len(route_areas), 0)
        self.assertTrue(all(abs(a - 0.5) < 1e-12 for a in route_areas))


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


class TestAdapters(unittest.TestCase):
    def test_qasm_roundtrip(self):
        from eo_pulse_ir.adapters import from_openqasm, to_openqasm
        circ = parse_circuit("qubits 3\nh 0\ncx 0 1\nrz 2 0.5\ncxswap 0 2\n")
        qasm = to_openqasm(circ)
        back = from_openqasm(qasm)
        self.assertEqual(back.num_qubits, circ.num_qubits)
        self.assertEqual([g.name for g in back.gates], [g.name for g in circ.gates])
        self.assertEqual(back.gates[1].qubits, (0, 1))
        self.assertAlmostEqual(back.gates[2].params[0], 0.5)

    def test_from_gate_list(self):
        from eo_pulse_ir.adapters import from_gate_list
        circ = from_gate_list([("h", (0,)), ("cnot", (0, 1)), ("rz", (1,), (0.3,))])
        self.assertEqual(circ.num_qubits, 2)
        self.assertEqual([g.name for g in circ.gates], ["h", "cx", "rz"])

    def test_external_records_roundtrip(self):
        from eo_pulse_ir.adapters import ir_to_pulse_records, pulse_records_to_result
        recs = [{"edge": [2, 3], "area": 3.14159, "role": "inter"},
                {"edge": [1, 2], "area": 1.57, "role": "intra"}]
        result = pulse_records_to_result(recs, num_dots=6)
        out = ir_to_pulse_records(result)
        self.assertEqual(len(out), 2)
        self.assertEqual(result.metrics.pulse_count, 2)

    def test_eoqrid_adapter_mock(self):
        from eo_pulse_ir.adapters.eoqrid import from_eoqrid, compile_eoqrid

        # Mock eoqrid output: a Qiskit-like QuantumCircuit with Ex gates
        class MockQubit:
            def __init__(self, idx):
                self._index = idx

        class MockGate:
            def __init__(self, name, params):
                self.name = name
                self.params = params

        class MockInstruction:
            def __init__(self, gate, qubits):
                self.operation = gate
                self.qubits = qubits

        class MockCircuit:
            def __init__(self, instructions):
                self.data = instructions

        qc = MockCircuit([
            MockInstruction(MockGate("ex", [1.5708, 0.5]),
                            [MockQubit(0), MockQubit(1)]),
            MockInstruction(MockGate("ex", [2.3562, 0.3]),
                            [MockQubit(1), MockQubit(2)]),
            MockInstruction(MockGate("barrier", []),
                            [MockQubit(0)]),
        ])
        records = from_eoqrid(qc)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["edge"], [0, 1])
        self.assertAlmostEqual(records[0]["area"], 1.5708)
        self.assertEqual(records[1]["edge"], [1, 2])

        # compile end-to-end
        result = compile_eoqrid(qc)
        self.assertEqual(result.metrics.pulse_count, 2)
        self.assertGreater(result.metrics.total_time, 0)

    def test_eoqrid_custom_param_to_area(self):
        from eo_pulse_ir.adapters.eoqrid import from_eoqrid

        class MockQubit:
            def __init__(self, idx):
                self._index = idx

        class MockGate:
            def __init__(self, name, params):
                self.name = name
                self.params = params

        class MockInstruction:
            def __init__(self, gate, qubits):
                self.operation = gate
                self.qubits = qubits

        class MockCircuit:
            def __init__(self, instructions):
                self.data = instructions

        qc = MockCircuit([
            MockInstruction(MockGate("ex", [2.0, 0.5]),
                            [MockQubit(0), MockQubit(1)]),
        ])
        # custom: area = param[0] * param[1]
        records = from_eoqrid(qc, param_to_area=lambda p: p[0] * p[1])
        self.assertAlmostEqual(records[0]["area"], 1.0)


class TestApp(unittest.TestCase):
    def test_dashboard_html(self):
        from eo_pulse_ir.dashboard import build_dashboard
        r = compile_circuit(parse_circuit("qubits 2\nh 0\ncx 0 1\n"))
        doc = build_dashboard(r, title="bell")
        self.assertIn("<!doctype html>", doc)
        self.assertIn("<svg", doc)
        self.assertIn("Control-cost metrics", doc)

    def test_app_compile_and_dashboard(self):
        import os
        from eo_pulse_ir.app import main
        example = os.path.join(os.path.dirname(__file__), "..", "examples", "bell.qasm")
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(main(["compile", example, "-o", d]), 0)
            self.assertEqual(main(["dashboard", example, "-o", d]), 0)
            self.assertTrue(os.path.getsize(os.path.join(d, "index.html")) > 0)


if __name__ == "__main__":
    unittest.main()
