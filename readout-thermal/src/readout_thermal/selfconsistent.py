"""Self-consistent electro-thermal fixed point.

Implements SPEC §3.3: for each Vpp solve
R(Te) = P_diss(Te, Vpp) + P_bg - Sigma_eff*(Te^p - Tph^p) = 0
for Te in [Tph, TE_MAX] with scipy.optimize.brentq; hard failure on
non-convergence or non-unique root, never silent clipping.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from .sensor import quadrature_points
from .thermal import ModelParams, background_power, cooling_power, dissipated_power

TE_MAX: float = 10.0
"""Upper end of the Te bracket [K] (SPEC §3.3)."""


def residual(te: float, vpp: float, params: ModelParams, n_t: int | None = None) -> float:
    """Heat-balance residual R(Te). Implements SPEC §3.3.

    R(Te) = P_diss(Te, Vpp) + P_bg - Sigma_eff*(Te^p - Tph^p); when the
    optional WF channel of §3.2 is enabled it enters the cooling term (and
    P_bg) per the reading logged in QUESTIONS.md Q1. Returns watts.
    """
    p_diss = float(
        dissipated_power(te, vpp, params.a, params.eps_b, params.delta_eps, n_t=n_t)
    )
    p_cool = float(
        cooling_power(
            te, params.tph, params.sigma_eff, params.p, params.include_wf, params.kappa_wf
        )
    )
    return p_diss + background_power(params) - p_cool


def count_sign_changes(
    vpp: float,
    params: ModelParams,
    te_max: float = TE_MAX,
    n_check: int = 200,
    n_t: int | None = None,
) -> int:
    """Number of sign changes of R(Te) on a grid over [Tph, te_max].

    Diagnostic helper for SPEC §3.3 / §6 T5. Zero samples inherit the
    preceding sign (the first sample defaults to + if zero, which occurs
    only in the te0 == tph, vpp -> 0 corner where the root sits at Tph).
    """
    n_t = quadrature_points(vpp, params.tph, n_t)
    tes = np.linspace(params.tph, te_max, n_check)
    r = np.array([residual(float(t), vpp, params, n_t=n_t) for t in tes])
    signs = np.sign(r)
    if signs[0] == 0:
        signs[0] = 1
    for i in range(1, len(signs)):
        if signs[i] == 0:
            signs[i] = signs[i - 1]
    return int(np.count_nonzero(np.diff(signs)))


def _assert_monotone_beyond(
    root: float,
    vpp: float,
    params: ModelParams,
    te_max: float,
    n_t: int,
    n_check: int = 100,
) -> None:
    """Uniqueness assertion of SPEC §3.3: R strictly monotone decreasing (and
    hence strictly negative) for Te beyond the root, so the root is unique
    and no tangent (double) root or close root pair exists above it.
    Raises ValueError on violation.
    """
    tes = np.linspace(root, te_max, n_check + 1)[1:]
    r = np.array([residual(float(t), vpp, params, n_t=n_t) for t in tes])
    if not (np.all(r < 0) and np.all(np.diff(r) < 0)):
        raise ValueError(
            "R(Te) is not monotone decreasing beyond the root: "
            "fixed point not certified unique (SPEC §3.3)"
        )


def solve_te(
    vpp: float,
    params: ModelParams,
    te_max: float = TE_MAX,
    n_t: int | None = None,
    check_unique: bool = True,
) -> float:
    """Self-consistent electron temperature for one Vpp. Implements SPEC §3.3.

    brentq on Te in [Tph, te_max]. The quadrature sample count is frozen once
    per solve at the worst case Te = Tph (sharpest integrand), so R(Te) is a
    continuous function of Te inside brentq. Raises ValueError if there is no
    sign change in the bracket or (with check_unique) if R is not monotone
    decreasing beyond the root; raises RuntimeError from brentq on
    non-convergence. Never clips.
    """
    n_t = quadrature_points(vpp, params.tph, n_t)
    r_lo = residual(params.tph, vpp, params, n_t=n_t)
    r_hi = residual(te_max, vpp, params, n_t=n_t)
    if r_lo < 0.0:
        raise ValueError(
            f"R(Tph) = {r_lo:.3e} W < 0: no root in bracket (check te0 >= tph)"
        )
    if r_hi >= 0.0:
        raise ValueError(
            f"R({te_max} K) = {r_hi:.3e} W >= 0: no sign change in bracket; "
            "electron system would exceed the bracket, refuse to clip"
        )
    if r_lo == 0.0:
        te = params.tph  # te0 == tph and vpp -> 0 corner: root at the boundary
    else:
        te = brentq(
            lambda t: residual(t, vpp, params, n_t=n_t),
            params.tph,
            te_max,
            xtol=1e-12,
            rtol=1e-12,
            maxiter=200,
            full_output=False,
            disp=True,  # raise on non-convergence
        )
    if check_unique:
        _assert_monotone_beyond(float(te), vpp, params, te_max, n_t)
    return float(te)


def solve_te_grid(
    vpps: np.ndarray,
    params: ModelParams,
    te_max: float = TE_MAX,
    n_t: int | None = None,
    check_unique: bool = True,
) -> np.ndarray:
    """Vectorized wrapper: self-consistent Te for each Vpp in the grid."""
    return np.array(
        [solve_te(float(v), params, te_max, n_t, check_unique) for v in np.asarray(vpps)]
    )
