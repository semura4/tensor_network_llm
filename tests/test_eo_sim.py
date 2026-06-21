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
        self.assertGreater(r["fidelity"], 0.9999)
        self.assertLess(r["leakage"], 1e-6)

    def test_validated_single_qubit_templates(self):
        # Every single-qubit native template must implement its gate exactly.
        import numpy as np
        from eo_pulse_ir import parse_circuit, synthesize
        from eo_pulse_ir.sim import gates, simulate
        cases = [("h", gates.H), ("x", gates.X), ("y", gates.Y), ("z", gates.Z),
                 ("s", gates.S), ("t", gates.T)]
        for g, V in cases:
            pulses, _ = synthesize(parse_circuit(f"qubits 1\n{g} 0\n"))
            r = simulate(pulses, 1, target=V)
            self.assertGreater(r["fidelity"], 0.99999, f"{g} fidelity")
        # parametrised rotations composed from validated H + exact Rz
        for g, V in [("rz", gates.rz(0.7)), ("rx", gates.rx(0.7)), ("ry", gates.ry(0.7))]:
            pulses, _ = synthesize(parse_circuit(f"qubits 1\n{g} 0 0.7\n"))
            r = simulate(pulses, 1, target=V)
            self.assertGreater(r["fidelity"], 0.99999, f"{g} fidelity")

    def test_validated_swap_template(self):
        from eo_pulse_ir import parse_circuit, synthesize
        from eo_pulse_ir.sim import gates, simulate
        pulses, _ = synthesize(parse_circuit("qubits 2\nswap 0 1\n"))
        r = simulate(pulses, 2, target=gates.SWAP)
        self.assertGreater(r["fidelity"], 0.9999)
        self.assertLess(r["leakage"], 1e-6)

    def test_validated_cxswap_template(self):
        from eo_pulse_ir import parse_circuit, synthesize
        from eo_pulse_ir.sim import gates, simulate
        pulses, _ = synthesize(parse_circuit("qubits 2\ncxswap 0 1\n"))
        r = simulate(pulses, 2, target=gates.CXSWAP)
        self.assertGreater(r["fidelity"], 0.999)
        self.assertLess(r["leakage"], 1e-3)


@unittest.skipUnless(_HAVE_NUMPY, "numpy required for the physics simulator")
class TestMPS(unittest.TestCase):
    def test_init_matches_dense(self):
        from eo_pulse_ir.sim.encoding import logical_basis
        from eo_pulse_ir.sim.mps import MPS
        m = MPS.logical_register([0, 1])
        dense = logical_basis(2)[:, 1]    # |01_L>
        self.assertTrue(np.allclose(m.to_dense(), dense, atol=1e-10))

    def test_mps_matches_dense_cnot(self):
        from eo_pulse_ir import parse_circuit, synthesize
        from eo_pulse_ir.sim.encoding import logical_basis
        from eo_pulse_ir.sim.mps import MPS, evolve_pulses
        from eo_pulse_ir.sim.operators import apply_pulse
        pulses, _ = synthesize(parse_circuit("qubits 2\ncx 0 1\n"))
        psi = logical_basis(2)[:, 0].astype(complex)
        for x in pulses:
            psi = apply_pulse(psi, 6, x.edge[0], x.edge[1], x.area)
        mE, chi, disc = evolve_pulses(MPS.logical_register([0, 0]), pulses, chi_max=64)
        fid = abs(np.vdot(psi, mE.to_dense())) ** 2 / (
            np.vdot(psi, psi).real * mE.overlap(mE).real)
        self.assertGreater(fid, 1 - 1e-9)

    def test_scaling_bounded_bond(self):
        from eo_pulse_ir import parse_circuit, synthesize
        from eo_pulse_ir.sim.mps import MPS, evolve_pulses
        circ = "qubits 6\nh 0\n" + "".join(f"cx {i} {i+1}\n" for i in range(5))
        pulses, _ = synthesize(parse_circuit(circ))
        _, chi, disc = evolve_pulses(MPS.logical_register([0] * 6), pulses,
                                     chi_max=32, tol=1e-10)
        self.assertLessEqual(chi, 32)
        self.assertLess(disc, 1e-6)


@unittest.skipUnless(_HAVE_NUMPY, "numpy required for the physics simulator")
class TestQuaternion(unittest.TestCase):
    def test_axis_geometry(self):
        from eo_pulse_ir.sim import quaternion as q
        self.assertAlmostEqual(float(q.N1 @ q.N2), -0.5, places=6)  # 120 degrees

    def test_axis_angle_to_su2(self):
        from eo_pulse_ir.sim import gates
        from eo_pulse_ir.sim import quaternion as q
        from eo_pulse_ir.sim.fidelity import average_gate_fidelity
        U = q.to_su2(q.from_axis_angle([0, 0, 1], 0.7))
        self.assertGreater(average_gate_fidelity(U, gates.rz(0.7)), 0.99999)

    def test_rotate_bloch(self):
        from eo_pulse_ir.sim import quaternion as q
        v = q.rotate_bloch(q.from_axis_angle([0, 1, 0], np.pi / 2), np.array([0, 0, 1.0]))
        self.assertTrue(np.allclose(v, [1, 0, 0], atol=1e-9))

    def test_compile_unitary_exact(self):
        from eo_pulse_ir.sim import quaternion as q
        from eo_pulse_ir.sim.fidelity import average_gate_fidelity
        from eo_pulse_ir.sim.simulator import logical_block
        role = {"intra_low": (0, 1), "intra_high": (1, 2)}
        rng = np.random.default_rng(1)
        worst = 1.0
        for _ in range(20):
            A = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
            U, _ = np.linalg.qr(A)
            seq = q.compile_unitary(U)
            M = logical_block([(role[r], a) for r, a in seq], 1)
            worst = min(worst, average_gate_fidelity(M, U))
        self.assertGreater(worst, 0.99999)

    def test_u_gate_front_end(self):
        from eo_pulse_ir import parse_circuit, synthesize
        from eo_pulse_ir.sim import simulate
        from eo_pulse_ir.sim import quaternion as q
        c = parse_circuit("qubits 1\nu 0 0.5 0.6 0.7\n")
        Rz = lambda a: np.array([[np.exp(-1j*a/2), 0], [0, np.exp(1j*a/2)]])
        Ry = lambda a: np.array([[np.cos(a/2), -np.sin(a/2)], [np.sin(a/2), np.cos(a/2)]])
        U = Rz(0.6) @ Ry(0.5) @ Rz(0.7)
        pulses, _ = synthesize(c)
        self.assertGreater(simulate(pulses, 1, target=U)["fidelity"], 0.99999)


@unittest.skipUnless(_HAVE_NUMPY, "numpy required for the physics simulator")
class TestControlLie(unittest.TestCase):
    def test_lie_closure_su2(self):
        from eo_pulse_ir.sim.lie import lie_closure
        X = np.array([[0, 1], [1, 0]], complex)
        Y = np.array([[0, -1j], [1j, 0]], complex)
        dim, _ = lie_closure([1j * X, 1j * Y])
        self.assertEqual(dim, 3)   # iX, iY generate su(2)

    def test_single_qubit_exchange_is_dfs(self):
        from eo_pulse_ir.sim.control import EOControlSystem
        s = EOControlSystem(1)
        self.assertEqual(s.lie_dimension(sector_down=1), 4)   # su(2) (+) u(1) block
        self.assertLess(s.leakage_coupling(1), 1e-9)          # no logical<->leakage

    def test_gradient_enlarges_algebra_and_couples_leakage(self):
        from eo_pulse_ir.sim.control import EOControlSystem
        s = EOControlSystem(1, gradient=1.0)
        self.assertEqual(s.lie_dimension(sector_down=1), 8)   # full su(3)
        self.assertGreater(s.leakage_coupling(1), 0.1)

    def test_two_qubit_exchange_couples_leakage(self):
        from eo_pulse_ir.sim.control import EOControlSystem
        s = EOControlSystem(2)
        self.assertGreater(s.leakage_coupling(2), 0.1)        # boundary leaks (Q2)

    def test_control_system_reproduces_gate(self):
        from eo_pulse_ir import parse_circuit, synthesize
        from eo_pulse_ir.sim import gates
        from eo_pulse_ir.sim.control import EOControlSystem
        from eo_pulse_ir.sim.fidelity import average_gate_fidelity
        s = EOControlSystem(2)
        pulses, _ = synthesize(parse_circuit("qubits 2\ncx 0 1\n"))
        M = s.logical_block(s.word_from_pulses(pulses))
        self.assertGreater(average_gate_fidelity(M, gates.CNOT), 0.9999)


@unittest.skipUnless(_HAVE_NUMPY, "numpy required for the physics simulator")
class TestField(unittest.TestCase):
    def _cnot(self):
        from eo_pulse_ir import parse_circuit, synthesize
        p, _ = synthesize(parse_circuit("qubits 2\ncx 0 1\n"))
        return [(tuple(x.edge), x.area) for x in p]

    def test_zero_field_matches_exchange_only(self):
        from eo_pulse_ir.sim import gates, simulate
        from eo_pulse_ir.sim.field import logical_block_field
        pulses = self._cnot()
        M = logical_block_field(pulses, 2, np.zeros(6), j_max=1.0)
        ref = simulate(pulses, 2, target=gates.CNOT)["fidelity"]
        from eo_pulse_ir.sim.fidelity import average_gate_fidelity
        self.assertAlmostEqual(average_gate_fidelity(M, gates.CNOT), ref, places=6)

    def test_uniform_field_is_harmless(self):
        from eo_pulse_ir import parse_circuit, synthesize
        from eo_pulse_ir.sim import gates
        from eo_pulse_ir.sim.fidelity import average_gate_fidelity, leakage
        from eo_pulse_ir.sim.field import logical_block_field, zeeman_energies
        p, _ = synthesize(parse_circuit("qubits 1\nx 0\n"))
        M = logical_block_field(p, 1, zeeman_energies(3, 0.0, b0=0.7))
        self.assertGreater(average_gate_fidelity(M, gates.X), 0.99999)
        self.assertLess(leakage(M), 1e-9)

    def test_gradient_breaks_dfs_single_qubit(self):
        from eo_pulse_ir import parse_circuit, synthesize
        from eo_pulse_ir.sim import gates
        from eo_pulse_ir.sim.field import simulate_field
        p, _ = synthesize(parse_circuit("qubits 1\nx 0\n"))
        no = simulate_field([(tuple(x.edge), x.area) for x in p], 1, 0.0, target=gates.X)
        yes = simulate_field([(tuple(x.edge), x.area) for x in p], 1, 0.1, target=gates.X)
        self.assertLess(no["leakage"], 1e-9)        # intra-only is leakage-free
        self.assertGreater(yes["leakage"], 1e-3)    # gradient induces leakage

    def test_gradient_degrades_two_qubit(self):
        from eo_pulse_ir.sim import gates
        from eo_pulse_ir.sim.field import simulate_field
        pulses = self._cnot()
        clean = simulate_field(pulses, 2, 0.0, target=gates.CNOT)
        noisy = simulate_field(pulses, 2, 0.02, target=gates.CNOT)
        self.assertGreater(clean["fidelity"], 0.9999)
        self.assertGreater(noisy["infidelity"], clean["infidelity"])


@unittest.skipUnless(_HAVE_NUMPY, "numpy required for the physics simulator")
class TestNoise(unittest.TestCase):
    def _cnot_pulses(self):
        from eo_pulse_ir import parse_circuit, synthesize
        pulses, _ = synthesize(parse_circuit("qubits 2\ncx 0 1\n"))
        return [tuple(p.edge) for p in pulses], [p.area for p in pulses]

    def test_zero_noise_matches_baseline(self):
        from eo_pulse_ir.sim import gates
        from eo_pulse_ir.sim.noise import montecarlo_fidelity
        edges, areas = self._cnot_pulses()
        r = montecarlo_fidelity(edges, areas, 2, gates.CNOT, sigma=0.0, n_samples=8)
        self.assertGreater(r["mean_fidelity"], 0.9999)
        self.assertAlmostEqual(r["std_fidelity"], 0.0, places=9)

    def test_noise_degrades_and_susceptibility_positive(self):
        from eo_pulse_ir.sim import gates
        from eo_pulse_ir.sim.noise import robustness_sweep, susceptibility
        edges, areas = self._cnot_pulses()
        sweep = robustness_sweep(edges, areas, 2, gates.CNOT,
                                 [0.0, 0.01, 0.03], n_samples=60, seed=0)
        self.assertGreater(sweep[0]["mean_fidelity"], sweep[-1]["mean_fidelity"])
        self.assertGreater(susceptibility(sweep), 0.0)

    def test_correlation_models_run(self):
        from eo_pulse_ir.sim import gates
        from eo_pulse_ir.sim.noise import montecarlo_fidelity
        edges, areas = self._cnot_pulses()
        for corr in ("per_edge", "global", "independent"):
            r = montecarlo_fidelity(edges, areas, 2, gates.CNOT, 0.02,
                                    n_samples=30, correlation=corr, seed=1)
            self.assertTrue(0.0 < r["mean_fidelity"] <= 1.0)


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
