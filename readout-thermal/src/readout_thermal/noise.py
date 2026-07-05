"""Noise model and SNR.

Implements SPEC §4. All quantities SI: currents [A], bandwidth [Hz].
"""

from __future__ import annotations

import numpy as np

from .constants import E

FloatOrArray = float | np.ndarray

B_DEFAULT: float = 500e3
"""Effective noise bandwidth [Hz] (SPEC §1: default 500 kHz from 1 MS/s
sampling; only ever appears in the product S_I*B / the shot-noise variance)."""


def shot_sigma(i_mean: FloatOrArray, b: float = B_DEFAULT) -> np.ndarray:
    """Optional shot-noise term. Implements SPEC §4.

    sigma_shot^2 = 2*e*Ibar*B, so sigma_shot = sqrt(2*e*Ibar*B) [A].
    Ibar (the mean current) is supplied by the caller: SPEC §4 does not
    prescribe how it is averaged, how this term combines with sigma_I, or
    whether it enters the SNR denominator (see QUESTIONS.md Q2) — so this
    function only evaluates the literal §4 expression and no combination
    rule is implemented. Raises for Ibar < 0 pending Q2.
    """
    i_mean = np.asarray(i_mean, dtype=float)
    if np.any(i_mean < 0):
        raise ValueError(
            "shot_sigma: negative Ibar; SPEC §4 gives no sign/averaging "
            "convention for the mean current (QUESTIONS.md Q2)"
        )
    return np.sqrt(2 * E * i_mean * b)


def snr(delta_x: FloatOrArray, sigma_i: FloatOrArray) -> np.ndarray:
    """SNR(Vpp) = DeltaX(Vpp) / sigma_I. Implements SPEC §4.

    sigma_i = sqrt(S_I*B) is the single fitted noise-floor constant of §4/§5
    (white, Vpp-independent). Dimensionless.
    """
    return np.asarray(delta_x, dtype=float) / np.asarray(sigma_i, dtype=float)
