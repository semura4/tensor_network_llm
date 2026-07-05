"""V_exc sweeps: SNR(Vpp) and Te(Vpp) from the self-consistent model.

Composes SPEC §2 (signal), §3 (self-consistent Te) and §4 (SNR). SI units
internally; µVpp/mK conversion helpers live here for the I/O boundary
(CLAUDE.md conventions).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .noise import snr as snr_of
from .selfconsistent import solve_te
from .sensor import delta_x, optimal_bias
from .thermal import ModelParams


def uvpp_to_v(vexc_uvpp: float | np.ndarray) -> np.ndarray:
    """I/O boundary: excitation amplitude µVpp -> V (SI)."""
    return np.asarray(vexc_uvpp, dtype=float) * 1e-6


def v_to_uvpp(vpp: float | np.ndarray) -> np.ndarray:
    """I/O boundary: excitation amplitude V (SI) -> µVpp."""
    return np.asarray(vpp, dtype=float) * 1e6


def k_to_mk(te: float | np.ndarray) -> np.ndarray:
    """I/O boundary: temperature K (SI) -> mK."""
    return np.asarray(te, dtype=float) * 1e3


@dataclass(frozen=True)
class SweepResult:
    """One V_exc sweep. All SI: vpp [V], te [K], dx [A]; snr dimensionless."""

    vpp: np.ndarray
    te: np.ndarray
    dx: np.ndarray
    snr: np.ndarray


def sweep(
    vpps: np.ndarray,
    params: ModelParams,
    sigma_i: float,
    self_heating: bool = True,
    reoptimize_bias: bool = False,
) -> SweepResult:
    """SNR(Vpp) and Te(Vpp) over a V_exc grid.

    For each Vpp: Te from the §3.3 fixed point (or pinned at Te0 when
    self_heating=False — the Sigma_eff -> infinity limit of §6 T7); charge
    signal DeltaX per §2.4 at the fixed bias params.eps_b, or with eps_b
    re-optimized at every (Vpp, Te) point when reoptimize_bias=True (§2.4);
    SNR = DeltaX/sigma_i per §4.

    Documented reading (QUESTIONS.md Q4): with reoptimize_bias=True the §3.3
    heat balance is still evaluated at the fixed params.eps_b, and eps_b is
    re-optimized afterwards for the signal at the resulting Te — SPEC does
    not specify a joint (Te, eps_b) fixed point.
    """
    vpps = np.asarray(vpps, dtype=float)
    tes = np.empty_like(vpps)
    dxs = np.empty_like(vpps)
    for i, vpp in enumerate(vpps):
        te = solve_te(float(vpp), params) if self_heating else params.te0
        if reoptimize_bias:
            eps_b = optimal_bias(te, float(vpp), params.a, params.delta_eps)
        else:
            eps_b = params.eps_b
        tes[i] = te
        dxs[i] = float(delta_x(float(vpp), te, params.a, params.delta_eps, eps_b))
    snrs = snr_of(dxs, sigma_i)
    return SweepResult(vpp=vpps, te=tes, dx=dxs, snr=snrs)
