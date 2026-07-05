"""M2 tests: SPEC §6 T3 (baseline), T4 (monotonicity), T5 (fixed point),
T7 (heating shifts the optimum), T8 (P_diss >= 0)."""

import numpy as np
import pytest

from readout_thermal.constants import E, KB
from readout_thermal.selfconsistent import (
    count_sign_changes,
    residual,
    solve_te,
    solve_te_grid,
)
from readout_thermal.sensor import default_bias, delta_x
from readout_thermal.thermal import ModelParams, background_power, dissipated_power

TE0 = 0.1  # K
TPH = 0.05  # K
A = 1e-9  # A
DELTA_EPS = 2.0 * KB * TE0  # J
SIGMA_EFF = 5e-11  # W/K^5
KT0_V = KB * TE0 / E  # kB*Te0 expressed as a voltage [V]


def make_params(**overrides) -> ModelParams:
    defaults = dict(
        a=A,
        delta_eps=DELTA_EPS,
        eps_b=default_bias(TE0, A, DELTA_EPS),
        te0=TE0,
        tph=TPH,
        sigma_eff=SIGMA_EFF,
    )
    defaults.update(overrides)
    return ModelParams(**defaults)


def test_t3_baseline() -> None:
    """T3: Te(Vpp -> 0) = Te0 within solver tolerance.

    At e*Vpp = 1e-4*kB*Te0 the physical heating shift is ~1e-10 K, so the
    1e-9 K bound leaves no room for a spurious offset above brentq's xtol.
    """
    params = make_params()
    te = solve_te(1e-4 * KT0_V, params)
    assert abs(te - TE0) < 1e-9


def test_t4_monotonic() -> None:
    """T4: Te strictly increasing in Vpp."""
    params = make_params()
    vpps = np.logspace(-1, 2, 15) * KT0_V
    tes = solve_te_grid(vpps, params)
    assert np.all(np.diff(tes) > 0)


def test_t5_unique_fixed_point() -> None:
    """T5: exactly one sign change of R(Te) in the bracket [Tph, 10 K].

    Checked both through the library helper and through an INDEPENDENT
    fine-grid sign count computed directly from residual() in this test, so a
    broken helper cannot certify itself.
    """
    params = make_params()
    for vpp_kt in (1e-3, 1.0, 30.0, 100.0):
        vpp = vpp_kt * KT0_V
        assert count_sign_changes(vpp, params) == 1
        tes = np.linspace(TPH, 10.0, 1500)
        r = np.array([residual(float(t), vpp, params) for t in tes])
        assert int(np.count_nonzero(np.diff(np.sign(r[r != 0])))) == 1


def test_t7_heating_shifts_optimum() -> None:
    """T7: argmax_Vpp SNR with self-heating ON is strictly smaller than with
    heating OFF (Sigma_eff -> infinity limit, i.e. Te pinned at Te0).

    SNR = DeltaX/sigma_I (SPEC §4) with sigma_I a Vpp-independent constant,
    so argmax SNR == argmax DeltaX.
    """
    params = make_params()
    vpps = np.logspace(-0.5, 2.5, 60) * KT0_V
    tes_on = solve_te_grid(vpps, params)
    dx_on = np.array(
        [
            float(delta_x(v, te, A, DELTA_EPS, params.eps_b))
            for v, te in zip(vpps, tes_on)
        ]
    )
    dx_off = np.array(
        [float(delta_x(v, TE0, A, DELTA_EPS, params.eps_b)) for v in vpps]
    )
    i_on, i_off = int(np.argmax(dx_on)), int(np.argmax(dx_off))
    assert 0 < i_on < len(vpps) - 1, "heating-ON optimum must be interior"
    assert 0 < i_off < len(vpps) - 1, "heating-OFF optimum must be interior"
    assert vpps[i_on] < vpps[i_off]


def test_t8_p_diss_nonnegative() -> None:
    """T8: P_diss >= 0 everywhere."""
    vpps = np.concatenate(([0.0], np.logspace(-2, 3, 40) * KT0_V))
    for te in (TPH, TE0, 0.3, 1.0):
        for eps_b in (-5 * KB * te, 0.0, 0.7 * KB * te, 20 * KB * te):
            p = dissipated_power(te, vpps, A, eps_b, DELTA_EPS)
            assert np.all(p >= 0)


def test_p_diss_small_signal_magnitude() -> None:
    """§3.1 amplitude convention pinned against the §2.2 analytic limit:
    per state P(Vpp -> 0) -> G(eps)*<V^2> = G(eps)*Vpp^2/8, and §3.1 takes
    the MEAN of the two charge states. Catches factor-of-2 / amplitude
    errors that the shape tests (T3-T8) cannot see."""
    from readout_thermal.sensor import conductance

    eps_b = 0.7 * KB * TE0
    vpp = 1e-3 * KT0_V
    p_num = float(dissipated_power(TE0, vpp, A, eps_b, DELTA_EPS))
    g_mean = 0.5 * (
        float(conductance(eps_b, TE0, A)) + float(conductance(eps_b + DELTA_EPS, TE0, A))
    )
    assert np.isclose(p_num, g_mean * vpp**2 / 8, rtol=1e-5)


def test_sensitivity_exponents_and_wf_baseline() -> None:
    """§3.2 sensitivity paths: for p in {4, 5, 6} and include_wf in
    {False, True}, P_bg matches the closed form and Te(Vpp -> 0) = Te0."""
    kappa = 1e-9  # W/K^2
    for p in (4, 5, 6):
        for wf in (False, True):
            params = make_params(p=p, include_wf=wf, kappa_wf=kappa if wf else 0.0)
            expected = SIGMA_EFF * (TE0**p - TPH**p)
            if wf:
                expected += kappa * (TE0**2 - TPH**2)
            assert background_power(params) == pytest.approx(expected)
            assert abs(solve_te(1e-4 * KT0_V, params) - TE0) < 1e-9


def test_background_power_baseline_consistency() -> None:
    """§3.2: P_bg equals the cooling power at Te0, and te0 < tph raises."""
    params = make_params()
    assert background_power(params) == pytest.approx(
        SIGMA_EFF * (TE0**5 - TPH**5)
    )
    with pytest.raises(ValueError):
        background_power(make_params(te0=0.5 * TPH))


def test_solve_te_raises_out_of_bracket() -> None:
    """§3.3: hard failure (no silent clipping) when Te would exceed bracket."""
    params = make_params(sigma_eff=1e-25)  # absurdly weak cooling
    with pytest.raises(ValueError, match="no sign change"):
        solve_te(100 * KT0_V, params)
