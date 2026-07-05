"""Thermal model: self-heating and heat balance.

Implements SPEC §3.1-§3.2. All quantities SI: power [W], temperature [K],
sigma_eff [W/K^p], kappa_wf [W/K^2].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .sensor import current, quadrature_points

FloatOrArray = float | np.ndarray


@dataclass(frozen=True)
class ModelParams:
    """Model constants (SPEC §2-§5). The five calibration constants of §5 are
    a, delta_eps, te0, sigma_eff (sigma_i lives in the noise model, §4);
    tph, p, include_wf, kappa_wf are fixed inputs / sensitivity flags (§1, §3.2).
    """

    a: float  # A [A], §2.1 calibration constant
    delta_eps: float  # [J], Coulomb-peak shift, §2.4
    eps_b: float  # [J], fixed bias point (DEFAULT mode), §2.4
    te0: float  # [K], base electron temperature, §3.2
    tph: float  # [K], phonon/bath temperature, §1
    sigma_eff: float  # [W/K^p], electron-phonon coupling, §3.2
    p: int = 5  # e-ph exponent, §3.2 (default 5; sensitivity: 4, 6)
    include_wf: bool = False  # §3.2 optional Wiedemann-Franz lead term
    kappa_wf: float = 0.0  # [W/K^2], §3.2


def dissipated_power(
    te: float,
    vpp: FloatOrArray,
    a: float,
    eps_b: float,
    delta_eps: float,
    n_t: int | None = None,
) -> np.ndarray:
    """Time-averaged dissipated power. Implements SPEC §3.1.

    P_diss(Te, Vpp) = (1/T) * int_0^T I(eps, V(t); Te) * V(t) dt, evaluated
    for each charge state (eps = eps_b and eps = eps_b + delta_eps); returns
    the MEAN of the two states (documented approximation of §3.1 — occupancy
    duty cycle unknown). Same uniform-grid trapezoidal quadrature over one
    period as §2.3 (sample count from quadrature_points(), tracking the
    sech^2 scale); the quasi-static current makes the result independent of
    the excitation frequency. Returns watts; vectorized over vpp.
    """
    vpp = np.asarray(vpp, dtype=float)
    n_t = quadrature_points(vpp, te, n_t)
    t = np.linspace(0.0, 1.0, n_t)  # one period in units of T
    v_t = (vpp[..., np.newaxis] / 2) * np.sin(2 * np.pi * t)
    p_states = [
        np.trapezoid(current(eps, v_t, te, a) * v_t, t, axis=-1)
        for eps in (eps_b, eps_b + delta_eps)
    ]
    return 0.5 * (p_states[0] + p_states[1])


def cooling_power(
    te: FloatOrArray,
    tph: float,
    sigma_eff: float,
    p: int = 5,
    include_wf: bool = False,
    kappa_wf: float = 0.0,
) -> np.ndarray:
    """Heat flow out of the electron system. Implements SPEC §3.2.

    Electron-phonon term: Sigma_eff * (Te^p - Tph^p), p = 5 default
    (sensitivity: p in {4, 6}). Optional Wiedemann-Franz lead term
    (include_wf): P_WF = kappa_wf * (Te^2 - Tph^2), treated as an additional
    cooling channel in the heat balance (it vanishes at Te = Tph and removes
    heat for Te > Tph; see QUESTIONS.md). Returns watts.
    """
    te = np.asarray(te, dtype=float)
    out = sigma_eff * (te**p - tph**p)
    if include_wf:
        out = out + kappa_wf * (te**2 - tph**2)
    return out


def background_power(params: ModelParams) -> float:
    """Background heat load P_bg. Implements SPEC §3.2.

    Fixed by requiring Te(Vpp -> 0) = Te0: since P_diss(Vpp -> 0) = 0, the
    heat balance P_diss + P_bg = cooling gives
    P_bg = Sigma_eff*(Te0^p - Tph^p) [+ kappa_wf*(Te0^2 - Tph^2) if WF].
    Returns watts. Requires Te0 >= Tph (P_bg >= 0).
    """
    if params.te0 < params.tph:
        raise ValueError("te0 must be >= tph (P_bg would be negative)")
    return float(
        cooling_power(
            params.te0,
            params.tph,
            params.sigma_eff,
            params.p,
            params.include_wf,
            params.kappa_wf,
        )
    )
