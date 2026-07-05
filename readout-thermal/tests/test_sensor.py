"""M1 tests: SPEC §6 T1 (linear conductance), T2 (FWHM), T6 (small-signal SNR)."""

import numpy as np
from scipy.optimize import brentq

from readout_thermal.constants import E, KB
from readout_thermal.sensor import (
    conductance,
    current,
    default_bias,
    delta_x,
    lockin_x,
)

TE = 0.1  # K
A = 1e-9  # A


def test_t1_linear_conductance() -> None:
    """T1: numerical dI/dV at V->0 matches SPEC §2.2 to rel. tol 1e-6."""
    kt = KB * TE
    eps = np.array([0.0, 0.5, -0.5, 1.0, -2.0, 3.0]) * kt
    dv = 1e-4 * kt / E  # e*dv = 1e-4 kB*Te, deep in linear response
    g_num = (current(eps, dv, TE, A) - current(eps, -dv, TE, A)) / (2 * dv)
    g_ana = conductance(eps, TE, A)
    assert np.all(np.abs(g_num / g_ana - 1) < 1e-6)


def test_t2_fwhm() -> None:
    """T2: conductance peak FWHM in eps = 3.525*kB*Te within 0.5%.

    Measured on the NUMERICAL finite-difference dI/dV of the §2.1 current
    (not the analytic §2.2 formula) so the test pins the implementation.
    """
    kt = KB * TE
    dv = 1e-4 * kt / E

    def g_num(eps: float) -> float:
        return float((current(eps, dv, TE, A) - current(eps, -dv, TE, A)) / (2 * dv))

    g_half = g_num(0.0) / 2

    def diff(eps: float) -> float:
        return g_num(eps) - g_half

    right = brentq(diff, 0.0, 10 * kt, xtol=1e-30)
    left = brentq(diff, -10 * kt, 0.0, xtol=1e-30)
    fwhm = right - left
    assert abs(fwhm / (3.525 * kt) - 1) < 0.005


def test_t6_small_signal_snr_linear() -> None:
    """T6: SNR proportional to Vpp within 2% for e*Vpp < 0.1*kB*Te0.

    SNR = DeltaX / sigma_I (SPEC §4) with sigma_I a Vpp-independent constant,
    so SNR proportional to Vpp is equivalent to DeltaX/Vpp being constant.
    """
    te0 = TE
    kt = KB * te0
    delta_eps = 2.0 * kt
    eps_b = default_bias(te0, A, delta_eps)
    sigma_i = 1e-13  # arbitrary constant noise floor [A]
    vpp = np.array([0.01, 0.03, 0.06, 0.099]) * kt / E
    snr = delta_x(vpp, te0, A, delta_eps, eps_b) / sigma_i
    ratio = snr / vpp
    assert np.all(np.abs(ratio / ratio[0] - 1) < 0.02)


def test_lockin_small_signal_matches_conductance() -> None:
    """Cross-check of §2.3 quadrature: X -> G(eps)*Vpp/2 as Vpp -> 0."""
    kt = KB * TE
    eps = np.array([0.0, 1.0, -1.5]) * kt
    vpp = 1e-3 * kt / E
    x = lockin_x(eps, TE, vpp, A)
    assert np.allclose(x, conductance(eps, TE, A) * vpp / 2, rtol=1e-5)


def test_lockin_quadrature_converged_at_large_vpp() -> None:
    """Quadrature adequacy at e*Vpp = 200*kB*Te: adaptive n_t agrees with a
    4x-oversampled reference to 1e-8 relative (auditor finding M1-3)."""
    kt = KB * TE
    vpp = 200 * kt / E
    eps = np.array([0.0, 5.0, -20.0]) * kt
    x = lockin_x(eps, TE, vpp, A)
    from readout_thermal.sensor import quadrature_points

    n_ref = 4 * quadrature_points(vpp, TE)
    x_ref = lockin_x(eps, TE, vpp, A, n_t=n_ref)
    assert np.allclose(x, x_ref, rtol=1e-8)


def test_current_odd_in_bias() -> None:
    """§2.1 symmetry: symmetric bias drop makes I odd in V at fixed eps."""
    kt = KB * TE
    v = 0.5 * kt / E
    eps = np.array([0.3, -0.7]) * kt
    assert np.allclose(current(eps, v, TE, A), -current(eps, -v, TE, A))
