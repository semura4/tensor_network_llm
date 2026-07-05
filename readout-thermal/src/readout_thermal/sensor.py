"""Sensor transport: single-level sequential tunneling, kB*Te >> h*Gamma.

Implements SPEC §2. All quantities in SI units: eps and delta_eps in J,
voltages in V, temperatures in K, currents in A.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit

from .constants import E, KB

FloatOrArray = float | np.ndarray


def fermi(x: FloatOrArray) -> np.ndarray:
    """Fermi function f(x) = 1/(1 + exp(x)) (SPEC §1), dimensionless argument.

    Evaluated as expit(-x), which is the same function in an
    overflow-safe form for large |x|.
    """
    return expit(-np.asarray(x, dtype=float))


def quadrature_points(vpp: FloatOrArray, te: float, n_t: int | None = None) -> int:
    """Number of uniform time samples per period for the §2.3/§3.1 quadratures.

    If n_t is given, it is returned unchanged. Otherwise the count scales with
    the sharpness of the integrand: the current of §2.1 varies on the scale
    kB*Te in eps, i.e. the resonance is crossed in a phase interval
    ~ kB*Te/(e*Vpp/2) of the drive, so the trapezoidal error on the periodic
    integrand decays like exp(-c*n_t*kB*Te/(e*Vpp)). n_t = 16*e*Vpp/(kB*Te)
    (floor 257) keeps that error negligible at any drive amplitude.
    """
    if n_t is not None:
        return n_t
    vmax = float(np.max(np.abs(np.asarray(vpp, dtype=float))))
    scale = E * vmax / (KB * te)
    return int(max(257, 16 * np.ceil(scale) + 1))


def current(eps: FloatOrArray, v: FloatOrArray, te: float, a: float) -> np.ndarray:
    """Finite-bias sequential-tunneling current. Implements SPEC §2.1.

    I(eps, V; Te) = A * [ f((eps - eV/2)/(kB*Te)) - f((eps + eV/2)/(kB*Te)) ]

    Parameters: eps [J] detuning, v [V] instantaneous bias (symmetric drop,
    eta = 1/2 per SPEC §1), te [K] electron temperature, a [A] the single
    calibration constant A = e*GL*GR/(GL+GR). Returns current [A]; eps and v
    broadcast together.
    """
    kt = KB * te
    eps = np.asarray(eps, dtype=float)
    v = np.asarray(v, dtype=float)
    return a * (fermi((eps - E * v / 2) / kt) - fermi((eps + E * v / 2) / kt))


def conductance(eps: FloatOrArray, te: float, a: float) -> np.ndarray:
    """Analytic linear-response conductance. Implements SPEC §2.2.

    G(eps) = dI/dV|_{V->0} = (A*e)/(4*kB*Te) * sech^2( eps/(2*kB*Te) )

    Returns conductance [S] (A [A] * e [C] / (kB*Te) [J] = A/V).
    """
    kt = KB * te
    eps = np.asarray(eps, dtype=float)
    return (a * E) / (4 * kt) / np.cosh(eps / (2 * kt)) ** 2


def lockin_x(
    eps: FloatOrArray,
    te: float,
    vpp: FloatOrArray,
    a: float,
    freq: float = 1.0,
    n_t: int | None = None,
) -> np.ndarray:
    """In-phase first-harmonic lock-in signal. Implements SPEC §2.3.

    X(eps; Te, Vpp) = (2/T) * int_0^T I(eps, V(t); Te) * sin(2*pi*f*t) dt
    with V(t) = (Vpp/2)*sin(2*pi*f*t) (SPEC §1) and T = 1/f, evaluated by
    trapezoidal quadrature on a uniform grid over one period; the sample
    count follows quadrature_points() so accuracy tracks the sech^2 scale
    kB*Te/(e*Vpp) at large drive. The quasi-static current I(V) of §2.1
    makes the result exactly independent of `freq`; `freq` is kept only to
    mirror §2.3 — if any frequency-dependent physics is ever added to SPEC,
    this placeholder default (1.0) must be replaced by the paper's fixed
    demodulation frequency (SPEC §1). Returns amperes. eps and vpp broadcast
    together.
    """
    eps = np.asarray(eps, dtype=float)
    vpp = np.asarray(vpp, dtype=float)
    n_t = quadrature_points(vpp, te, n_t)
    period = 1.0 / freq
    t = np.linspace(0.0, period, n_t)
    s = np.sin(2 * np.pi * freq * t)
    v_t = (vpp[..., np.newaxis] / 2) * s
    i_t = current(eps[..., np.newaxis], v_t, te, a)
    return (2 / period) * np.trapezoid(i_t * s, t, axis=-1)


def delta_x(
    vpp: FloatOrArray,
    te: float,
    a: float,
    delta_eps: float,
    eps_b: FloatOrArray,
    n_t: int | None = None,
) -> np.ndarray:
    """Charge signal between the two sensor states. Implements SPEC §2.4.

    State 0: eps = eps_b. State 1: eps = eps_b + delta_eps.
    DeltaX(Vpp) = | X(eps_b) - X(eps_b + delta_eps) |   [A]
    """
    x0 = lockin_x(eps_b, te, vpp, a, n_t=n_t)
    x1 = lockin_x(np.asarray(eps_b, dtype=float) + delta_eps, te, vpp, a, n_t=n_t)
    return np.abs(x0 - x1)


def optimal_bias(
    te: float,
    vpp: float,
    a: float,
    delta_eps: float,
    n_scan: int = 401,
    n_t: int | None = None,
) -> float:
    """Bias point eps_b maximizing DeltaX at given (Vpp, Te). Implements the
    bias-point optimization of SPEC §2.4 (used both for the DEFAULT fixed bias
    at the base condition and for `reoptimize_bias=True`).

    Numerics: coarse scan over eps_b in +/-(|delta_eps| + e*Vpp/2 + 8*kB*Te)
    — the e*Vpp/2 term is required because at finite drive the lock-in peak
    edges (and the DeltaX maximum) sit near |eps| ~ e*Vpp/2 — then bounded
    scalar refinement between the neighbours of the best grid point with
    xatol tied to the bracket width (SciPy's default absolute xatol of 1e-5
    would be larger than the whole bracket in SI energy units and would skip
    refinement entirely). Returns eps_b [J].
    """
    kt = KB * te
    half_width = abs(delta_eps) + E * vpp / 2 + 8 * kt
    grid = np.linspace(-half_width, half_width, n_scan)
    vals = delta_x(vpp, te, a, delta_eps, grid, n_t=n_t)
    i = int(np.argmax(vals))
    lo = grid[max(i - 1, 0)]
    hi = grid[min(i + 1, n_scan - 1)]
    res = minimize_scalar(
        lambda x: -float(delta_x(vpp, te, a, delta_eps, x, n_t=n_t)),
        bounds=(lo, hi),
        method="bounded",
        options={"xatol": 1e-6 * (hi - lo)},
    )
    return float(res.x)


def default_bias(
    te0: float,
    a: float,
    delta_eps: float,
    small_vpp_factor: float = 0.01,
) -> float:
    """DEFAULT fixed bias point of SPEC §2.4: eps_b maximizing DeltaX at the
    base condition (Vpp small, Te = Te0).

    Numerics: "Vpp small" is realized as e*Vpp = small_vpp_factor * kB * Te0
    (default 0.01). In this linear-response limit DeltaX is proportional to
    Vpp with an eps_b-dependence that no longer depends on Vpp, so the
    returned eps_b is insensitive to the exact factor; this is a documented
    quadrature-scale choice, not a physics assumption. Returns eps_b [J].
    """
    vpp = small_vpp_factor * KB * te0 / E
    return optimal_bias(te0, vpp, a, delta_eps)
