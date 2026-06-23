"""Subspace-restricted gate fidelity and leakage from a logical block M.

Given the logical block ``M = L^dagger U L`` (d x d, generally sub-unitary
because population can leak out of the encoded subspace), we use the standard
average gate fidelity for a (possibly non-unitary) process compared to a target
unitary ``V`` (Pedersen, Moller & Molmer 2007):

    F_avg = ( |Tr(V^dagger M)|^2 + Tr(M^dagger M) ) / ( d (d + 1) )

The first term penalises coherent error (it is phase-insensitive, so global
phase does not matter); the second penalises leakage, since Tr(M^dagger M) = d
only if no population left the subspace.  Leakage is reported separately as

    L = 1 - Tr(M^dagger M) / d
"""

from __future__ import annotations

import numpy as np


def leakage(M: np.ndarray) -> float:
    """Average population that leaves the logical subspace (0 = none)."""
    d = M.shape[0]
    return float(1.0 - np.real(np.trace(M.conj().T @ M)) / d)


def average_gate_fidelity(M: np.ndarray, target: np.ndarray) -> float:
    """Leakage-aware average gate fidelity of M against the target unitary."""
    d = M.shape[0]
    coherent = np.abs(np.trace(target.conj().T @ M)) ** 2
    survival = np.real(np.trace(M.conj().T @ M))
    return float((coherent + survival) / (d * (d + 1)))


def gate_overlap(M: np.ndarray, target: np.ndarray) -> float:
    """Phase-insensitive process overlap |Tr(V^dagger M)|^2 / d^2 in [0,1].

    Ignores leakage normalisation; useful as a 'how close in shape' diagnostic
    independent of the survival term.
    """
    d = M.shape[0]
    return float(np.abs(np.trace(target.conj().T @ M)) ** 2 / d ** 2)
