"""M3 tests: noise model (SPEC §4) and sweep composition."""

import numpy as np

from readout_thermal.constants import E, KB
from readout_thermal.noise import B_DEFAULT, shot_sigma, snr
from readout_thermal.selfconsistent import solve_te
from readout_thermal.sensor import default_bias, delta_x
from readout_thermal.sweep import sweep, uvpp_to_v, v_to_uvpp
from readout_thermal.thermal import ModelParams

TE0 = 0.1
TPH = 0.05
A = 1e-9
DELTA_EPS = 2.0 * KB * TE0
SIGMA_EFF = 5e-11
SIGMA_I = 2e-13
KT0_V = KB * TE0 / E


def make_params() -> ModelParams:
    return ModelParams(
        a=A,
        delta_eps=DELTA_EPS,
        eps_b=default_bias(TE0, A, DELTA_EPS),
        te0=TE0,
        tph=TPH,
        sigma_eff=SIGMA_EFF,
    )


def test_snr_definition() -> None:
    """§4: SNR = DeltaX / sigma_I, elementwise."""
    dx = np.array([1e-12, 2e-12])
    assert np.allclose(snr(dx, SIGMA_I), dx / SIGMA_I)


def test_shot_noise_definition() -> None:
    """§4: sigma_shot = sqrt(2*e*Ibar*B), literal expression only; negative
    Ibar rejected pending QUESTIONS.md Q2 (no convention baked in)."""
    import pytest

    i_mean = 1e-10
    assert np.isclose(shot_sigma(i_mean), np.sqrt(2 * E * i_mean * B_DEFAULT))
    with pytest.raises(ValueError, match="Q2"):
        shot_sigma(-1e-10)


def test_unit_helpers_roundtrip() -> None:
    x = np.array([2.0, 85.0, 1000.0])
    assert np.allclose(v_to_uvpp(uvpp_to_v(x)), x)
    assert np.isclose(uvpp_to_v(85.0), 85e-6)


def test_sweep_composition() -> None:
    """Sweep returns §2.4 DeltaX evaluated at the §3.3 Te, divided per §4 —
    pinned by INDEPENDENT recomputation of every point, so a sweep that used
    the wrong Te (or the wrong composition) cannot pass."""
    params = make_params()
    vpps = np.logspace(-0.5, 1.5, 7) * KT0_V
    res = sweep(vpps, params, SIGMA_I)
    assert res.snr.shape == vpps.shape
    assert np.all(np.diff(res.te) > 0)  # T4 through the sweep path
    for i, vpp in enumerate(vpps):
        te_i = solve_te(float(vpp), params)
        dx_i = float(delta_x(float(vpp), te_i, A, DELTA_EPS, params.eps_b))
        assert np.isclose(res.te[i], te_i, rtol=0, atol=1e-12)
        assert np.isclose(res.dx[i], dx_i, rtol=1e-12)
        assert np.isclose(res.snr[i], dx_i / SIGMA_I, rtol=1e-12)
    # heating OFF pins Te at Te0 (Sigma_eff -> infinity limit of §6 T7)
    res_off = sweep(vpps, params, SIGMA_I, self_heating=False)
    assert np.allclose(res_off.te, TE0)
    for i, vpp in enumerate(vpps):
        dx_i = float(delta_x(float(vpp), TE0, A, DELTA_EPS, params.eps_b))
        assert np.isclose(res_off.dx[i], dx_i, rtol=1e-12)


def test_sweep_reoptimized_bias_not_worse() -> None:
    """§2.4: re-optimizing eps_b at each point cannot reduce DeltaX below the
    fixed-bias value (up to optimizer tolerance)."""
    params = make_params()
    vpps = np.array([0.5, 5.0, 50.0]) * KT0_V
    fixed = sweep(vpps, params, SIGMA_I)
    reopt = sweep(vpps, params, SIGMA_I, reoptimize_bias=True)
    assert np.all(reopt.dx >= fixed.dx * (1 - 1e-6))
