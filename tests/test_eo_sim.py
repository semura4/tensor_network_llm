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
class TestCalibration(unittest.TestCase):
    def test_calibrate_and_predict_hierarchy(self):
        from eo_pulse_ir import parse_circuit, synthesize
        from eo_pulse_ir.sim import gates
        from eo_pulse_ir.sim.calibration import calibrate_sigma, mean_fidelity
        cx = [(tuple(p.edge), p.area) for p in
              synthesize(parse_circuit("qubits 2\ncx 0 1\n"))[0]]
        h = [(tuple(p.edge), p.area) for p in
             synthesize(parse_circuit("qubits 1\nh 0\n"))[0]]
        sigma = calibrate_sigma(cx, 2, gates.CNOT, 0.99, n_samples=200)
        self.assertTrue(0.0 < sigma < 0.05)
        f_cx = mean_fidelity(cx, 2, gates.CNOT, sigma, 400)
        f_h = mean_fidelity(h, 1, gates.H, sigma, 400)
        self.assertAlmostEqual(f_cx, 0.99, delta=0.01)
        self.assertGreater(f_h, f_cx)        # 1Q better than 2Q (fewer pulses)


@unittest.skipUnless(_HAVE_NUMPY, "numpy required for the physics simulator")
class TestValley(unittest.TestCase):
    def test_aligned_valleys_match_spin_only(self):
        from eo_pulse_ir import parse_circuit, synthesize
        from eo_pulse_ir.sim import gates
        from eo_pulse_ir.sim.valley import simulate_valley
        p, _ = synthesize(parse_circuit("qubits 2\ncx 0 1\n"))
        r = simulate_valley(p, 2, valley_phase=0.0, e_vs=5.0, target=gates.CNOT)
        self.assertGreater(r["fidelity"], 0.9999)        # matches spin-only CNOT
        self.assertLess(r["valley_leakage"], 1e-6)

    def test_phase_mismatch_leaks(self):
        import numpy as np
        from eo_pulse_ir.sim.valley import two_dot_pulse
        from eo_pulse_ir.sim.fidelity import leakage
        L = np.zeros((16, 4))
        for k, (sa, sb) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
            L[(2 * sa) * 4 + (2 * sb), k] = 1.0
        M = L.conj().T @ two_dot_pulse(np.pi / 2, np.pi / 2, 0.0, 0.0) @ L
        self.assertGreater(leakage(M), 0.1)

    def test_large_evs_suppresses_leakage(self):
        import numpy as np
        from eo_pulse_ir.sim.valley import two_dot_pulse
        from eo_pulse_ir.sim.fidelity import leakage
        L = np.zeros((16, 4))
        for k, (sa, sb) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
            L[(2 * sa) * 4 + (2 * sb), k] = 1.0
        def lk(evs):
            M = L.conj().T @ two_dot_pulse(np.pi / 2, np.pi / 2, evs, evs) @ L
            return leakage(M)
        self.assertLess(lk(10.0), lk(0.0))               # E_VS >> J protects

    def test_effective_areas_null_at_pi(self):
        import numpy as np
        from eo_pulse_ir.sim.valley import effective_valley_areas
        pulses = [((2, 3), 1.0)]                          # an inter edge
        out0 = effective_valley_areas(pulses, [0, 0, 0, 0, 0, 0])
        outp = effective_valley_areas(pulses, [0, 0, 0, np.pi, np.pi, np.pi])
        self.assertAlmostEqual(out0[0][1], 1.0, places=6)        # aligned: unchanged
        self.assertLess(outp[0][1], 1e-9)                        # Δφ=π: nulled


@unittest.skipUnless(_HAVE_NUMPY, "numpy required for the physics simulator")
class TestBifurcation(unittest.TestCase):
    def test_local_maxima_count(self):
        from eo_pulse_ir.sim.bifurcation import local_maxima
        # two clear bumps on a flat-ish periodic grid
        x = np.linspace(0, 2 * np.pi, 40, endpoint=False)
        X, Y = np.meshgrid(x, x)
        G = np.cos(X) + np.cos(Y)            # maxima where both cos=1 -> one peak on torus
        self.assertEqual(len(local_maxima(G, periodic=True)), 1)

    def test_classify_critical(self):
        from eo_pulse_ir.sim.bifurcation import classify_critical
        x = np.linspace(0, 2 * np.pi, 40, endpoint=False)
        X, Y = np.meshgrid(x, x)
        G = np.cos(X) + np.cos(Y)
        # peak at index (0,0) is a maximum
        self.assertEqual(classify_critical(G, 0, 0), "max")

    def test_optima_sweep_detects_more_optima(self):
        from eo_pulse_ir.sim.bifurcation import optima_sweep
        x = np.linspace(0, 2 * np.pi, 32, endpoint=False)
        # param p adds a second cosine harmonic -> more optima as p grows
        def grid(p):
            X, Y = np.meshgrid(x, x)
            return np.cos(X) + np.cos(Y) + p * (np.cos(2 * X) + np.cos(2 * Y))
        sweep = optima_sweep(grid, [0.0, 1.0], x, x, periodic=True)
        self.assertLessEqual(sweep[0].num_maxima, sweep[1].num_maxima)


@unittest.skipUnless(_HAVE_NUMPY, "numpy required for the physics simulator")
class TestMPSGrape(unittest.TestCase):
    def test_ghz_target_matches_dense(self):
        from eo_pulse_ir.sim.encoding import logical_basis
        from eo_pulse_ir.sim.mps_grape import ghz_target
        g = ghz_target(3)
        L = logical_basis(3)
        ghz = L[:, 0] + L[:, -1]
        ghz /= np.linalg.norm(ghz)
        self.assertGreater(abs(np.vdot(ghz, g.to_dense())) ** 2, 1 - 1e-9)

    def test_adjoint_gradient_matches_fd(self):
        from eo_pulse_ir.sim.mps import MPS
        from eo_pulse_ir.sim.mps_grape import (fidelity_and_grad, ghz_target,
                                               state_prep_fidelity)
        init, tgt = MPS.logical_register([0, 0]), ghz_target(2)
        edges = [(1, 2), (0, 1), (2, 3), (3, 4), (2, 3)]
        rng = np.random.default_rng(0)
        a = rng.uniform(0, 2 * np.pi, size=len(edges))
        _, ga = fidelity_and_grad(init, tgt, a, edges, chi_max=32)
        gfd = np.zeros(len(edges))
        eps = 1e-6
        for k in range(len(edges)):
            ap = a.copy(); ap[k] += eps
            am = a.copy(); am[k] -= eps
            gfd[k] = (state_prep_fidelity(init, tgt, ap, edges, 32)
                      - state_prep_fidelity(init, tgt, am, edges, 32)) / (2 * eps)
        self.assertLess(np.max(np.abs(ga - gfd)), 1e-6)

    def test_grape_prepares_bell_encoded(self):
        from eo_pulse_ir.sim.mps import MPS
        from eo_pulse_ir.sim.mps_grape import ghz_target, grape_state_prep
        from eo_pulse_ir.sim.synthesis import layered_ansatz
        init, tgt = MPS.logical_register([0, 0]), ghz_target(2)
        edges = layered_ansatz(2, 3)
        _, F = grape_state_prep(init, tgt, edges, chi_max=16, steps=200,
                                restarts=3, seed=1)
        self.assertGreater(F, 0.9)


@unittest.skipUnless(_HAVE_NUMPY, "numpy required for the physics simulator")
class TestSynthesis(unittest.TestCase):
    def test_layered_ansatz_structure(self):
        from eo_pulse_ir.sim.synthesis import inter_edges, layered_ansatz
        self.assertEqual(inter_edges(3), [(2, 3), (5, 6)])
        self.assertGreater(len(layered_ansatz(3, 4)), len(layered_ansatz(3, 2)))

    def test_design_single_qubit_gate(self):
        from eo_pulse_ir.sim.synthesis import design_gate
        rng = np.random.default_rng(2)
        A = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
        U, _ = np.linalg.qr(A)
        d = design_gate(U, 1, n_layers=2, restarts=8, steps=800, seed=0)
        self.assertGreater(d.fidelity, 0.999)
        self.assertEqual(d.num_qubits, 1)

    def test_design_returns_simulatable_pulses(self):
        from eo_pulse_ir.sim import gates
        from eo_pulse_ir.sim.fidelity import average_gate_fidelity
        from eo_pulse_ir.sim.simulator import logical_block
        from eo_pulse_ir.sim.synthesis import design_gate
        d = design_gate(gates.Z, 1, n_layers=1, restarts=4, steps=400, seed=0)
        M = logical_block(d.pulses, 1)
        self.assertAlmostEqual(average_gate_fidelity(M, gates.Z), d.fidelity, places=6)


@unittest.skipUnless(_HAVE_NUMPY, "numpy required for the physics simulator")
class TestStateSpace(unittest.TestCase):
    def test_real_vector_field_exact(self):
        from eo_pulse_ir.sim.operators import s_dot_s
        from eo_pulse_ir.sim.statespace import real_generator, to_real
        H = s_dot_s(3, 0, 1).astype(complex)
        A = real_generator(H)
        rng = np.random.default_rng(0)
        psi = rng.normal(size=8) + 1j * rng.normal(size=8)
        self.assertLess(np.max(np.abs(A @ to_real(psi) - to_real(-1j * H @ psi))), 1e-12)
        self.assertLess(np.max(np.abs(A + A.T)), 1e-12)   # skew-symmetric

    def test_flow_matches_operator(self):
        from eo_pulse_ir import parse_circuit, synthesize
        from eo_pulse_ir.sim.control import EOControlSystem
        from eo_pulse_ir.sim.encoding import logical_basis
        from eo_pulse_ir.sim.statespace import StateSpaceSystem, to_real, from_real
        p, _ = synthesize(parse_circuit("qubits 2\ncx 0 1\n"))
        ss, cs = StateSpaceSystem(2), EOControlSystem(2)
        x0 = to_real(logical_basis(2)[:, 0].astype(complex))
        xT = ss.integrate(x0, cs.word_from_pulses(p))
        op = cs.propagator(cs.word_from_pulses(p)) @ logical_basis(2)[:, 0]
        self.assertLess(np.max(np.abs(from_real(xT) - op)), 1e-9)

    def test_logical_manifold_invariant_under_exchange(self):
        from eo_pulse_ir.sim.encoding import logical_basis
        from eo_pulse_ir.sim.statespace import StateSpaceSystem, flow
        ss = StateSpaceSystem(1)
        psi = logical_basis(1)[:, 0].astype(complex)
        for edge, tau in [((0, 1), 0.7), ((1, 2), 1.1), ((0, 1), 0.5)]:
            psi = flow(ss.hamiltonian(edge), psi, tau)
        self.assertLess(ss.logical_manifold_defect(psi), 1e-10)


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
