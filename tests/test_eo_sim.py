"""Tests for the optional exchange-only physics simulator (requires numpy)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import numpy as np
    _HAVE_NUMPY = True
except Exception:  # pragma: no cover
    _HAVE_NUMPY = False


@unittest.skipUnless(_HAVE_NUMPY, "numpy required for the physics simulator")
class TestOperators(unittest.TestCase):
    def test_full_swap_at_pi(self):
        from eo_pulse_ir.sim.operators import exchange_propagator, swap_matrix
        U = exchange_propagator(2, 0, 1, np.pi)
        S = swap_matrix(2, 0, 1)
        self.assertLess(np.max(np.abs(U - np.exp(-1j * np.pi / 4) * S)), 1e-12)

    def test_propagator_unitary(self):
        from eo_pulse_ir.sim.operators import exchange_propagator
        U = exchange_propagator(3, 1, 2, 0.937)
        self.assertLess(np.max(np.abs(U.conj().T @ U - np.eye(8))), 1e-12)

    def test_apply_pulse_matches_propagator(self):
        from eo_pulse_ir.sim.operators import apply_pulse, exchange_propagator
        n = 3
        st = np.eye(1 << n, dtype=complex)
        out = apply_pulse(st, n, 0, 1, 1.3)
        U = exchange_propagator(n, 0, 1, 1.3)
        self.assertLess(np.max(np.abs(out - U)), 1e-12)


@unittest.skipUnless(_HAVE_NUMPY, "numpy required for the physics simulator")
class TestEncodingFidelity(unittest.TestCase):
    def test_logical_basis_orthonormal(self):
        from eo_pulse_ir.sim.encoding import logical_basis
        for nq in (1, 2):
            L = logical_basis(nq)
            self.assertEqual(L.shape, (1 << (3 * nq), 1 << nq))
            self.assertLess(np.max(np.abs(L.conj().T @ L - np.eye(1 << nq))), 1e-12)

    def test_identity_fidelity(self):
        from eo_pulse_ir.sim import gates, simulate
        r = simulate([], 1, target=gates.I1)
        self.assertAlmostEqual(r["fidelity"], 1.0, places=10)
        self.assertLess(abs(r["leakage"]), 1e-10)


@unittest.skipUnless(_HAVE_NUMPY, "numpy required for the physics simulator")
class TestPhysics(unittest.TestCase):
    def test_intra_low_is_rz(self):
        from eo_pulse_ir.sim import gates
        from eo_pulse_ir.sim.fidelity import average_gate_fidelity, leakage
        from eo_pulse_ir.sim.simulator import logical_block
        theta = 0.8
        M = logical_block([((0, 1), theta)], 1)
        self.assertAlmostEqual(average_gate_fidelity(M, gates.rz(-theta)), 1.0, places=9)
        self.assertLess(abs(leakage(M)), 1e-10)

    def test_single_qubit_is_leakage_free(self):
        from eo_pulse_ir.sim.fidelity import leakage
        from eo_pulse_ir.sim.simulator import logical_block
        M = logical_block([((0, 1), 0.7), ((1, 2), 1.3), ((0, 1), 2.1)], 1)
        self.assertLess(abs(leakage(M)), 1e-10)

    def test_two_qubit_boundary_leaks(self):
        from eo_pulse_ir.sim.fidelity import leakage
        from eo_pulse_ir.sim.simulator import logical_block
        M = logical_block([((2, 3), np.pi / 2)], 2)
        self.assertGreater(leakage(M), 0.1)


@unittest.skipUnless(_HAVE_NUMPY, "numpy required for the physics simulator")
class TestOptimizeAndIRBridge(unittest.TestCase):
    def test_x_and_h_synthesize_with_3_pulses(self):
        from eo_pulse_ir.sim import gates
        from eo_pulse_ir.sim.optimize import optimize_areas
        edges = [(1, 2), (0, 1), (1, 2)]
        for V in (gates.X, gates.H):
            _, F = optimize_areas(edges, 1, V, restarts=8, seed=2)
            self.assertGreater(F, 0.999)

    def test_y_needs_four_pulses(self):
        from eo_pulse_ir.sim import gates
        from eo_pulse_ir.sim.optimize import optimize_areas
        _, f3 = optimize_areas([(1, 2), (0, 1), (1, 2)], 1, gates.Y, restarts=8, seed=2)
        _, f4 = optimize_areas([(0, 1), (1, 2), (0, 1), (1, 2)], 1, gates.Y, restarts=8, seed=2)
        self.assertLess(f3, 0.9)       # 3 alternating pulses cannot reach Y
        self.assertGreater(f4, 0.999)  # 4 can

    def test_ir_pulses_feed_simulator(self):
        # Pulse objects from the IR are consumed directly by the simulator.
        from eo_pulse_ir import parse_circuit, synthesize
        from eo_pulse_ir.sim.simulator import logical_block
        pulses, _ = synthesize(parse_circuit("qubits 1\nrz 0 0.5\n"))
        M = logical_block(pulses, 1)
        self.assertEqual(M.shape, (2, 2))

    def test_validated_cnot_template(self):
        # The native CNOT template, synthesised through the IR and simulated,
        # must reproduce its validated leakage-free fidelity.
        from eo_pulse_ir import parse_circuit, synthesize
        from eo_pulse_ir.sim import gates, simulate
        pulses, _ = synthesize(parse_circuit("qubits 2\ncx 0 1\n"))
        r = simulate(pulses, 2, target=gates.CNOT)
        self.assertGreater(r["fidelity"], 0.999)
        self.assertLess(r["leakage"], 1e-3)


@unittest.skipUnless(_HAVE_NUMPY, "numpy required for the physics simulator")
class TestLandscape(unittest.TestCase):
    def test_sweep_and_heatmap(self):
        from eo_pulse_ir.sim.landscape import (gradient_magnitude, heatmap_svg,
                                               sweep2d)
        xs = np.linspace(0, 1, 5)
        ys = np.linspace(0, 1, 4)
        grids = sweep2d(lambda x, y: {"v": x * y}, xs, ys)
        self.assertEqual(grids["v"].shape, (4, 5))
        g = gradient_magnitude(grids["v"], xs, ys)
        self.assertEqual(g.shape, (4, 5))
        svg = heatmap_svg(grids["v"], xs, ys, "t", "x", "y", "v")
        self.assertTrue(svg.startswith("<svg") and svg.endswith("</svg>"))


if __name__ == "__main__":
    unittest.main()
