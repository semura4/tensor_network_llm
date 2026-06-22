"""Calibrate the simulator's error model to reported device fidelities.

The closed-system simulator predicts ideal gate fidelities (F ~ 1). Real silicon
exchange-only gates are limited mainly by charge noise (dJ/J), with valley a
secondary channel (see docs/valley_splitting_research.md). This module fits a
single phenomenological **quasi-static per-edge area-noise** parameter sigma
(= dJ/J) so the simulator reproduces a reported reference fidelity, turning it
into a *calibrated predictor* rather than an ideal one.

Honesty: this is a phenomenological fit (one knob to one number), not a
first-principles noise derivation. Its value is that, once sigma is fixed to one
gate, the **relative** fidelities of other gates (which scale with pulse count)
are predicted — and they match the reported 1Q/2Q hierarchy.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from .noise import montecarlo_fidelity

Edge = Tuple[int, int]


def mean_fidelity(pulses: Sequence[Tuple[Edge, float]], num_qubits: int, target,
                  sigma: float, n_samples: int = 400, correlation: str = "per_edge",
                  relative: bool = True, seed: int = 0) -> float:
    edges = [tuple(e) for e, _ in pulses]
    areas = [a for _, a in pulses]
    return montecarlo_fidelity(edges, areas, num_qubits, target, sigma,
                               n_samples=n_samples, relative=relative,
                               correlation=correlation, seed=seed)["mean_fidelity"]


def calibrate_sigma(pulses: Sequence[Tuple[Edge, float]], num_qubits: int, target,
                    target_fidelity: float, n_samples: int = 400,
                    sigma_hi: float = 0.05, iters: int = 24) -> float:
    """Bisection: find the area-noise sigma giving ``target_fidelity`` for a gate.

    Mean fidelity decreases monotonically with sigma, so bisection is robust.
    """
    lo, hi = 0.0, sigma_hi
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        f = mean_fidelity(pulses, num_qubits, target, mid, n_samples)
        if f > target_fidelity:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def predict_fidelities(sigma: float,
                       gates_spec: List[Tuple[str, Sequence, int, object]],
                       n_samples: int = 600) -> List[dict]:
    """Predict mean fidelity at a calibrated sigma for several gates.

    ``gates_spec`` = [(label, pulses, num_qubits, target), ...].
    """
    out = []
    for label, pulses, nq, target in gates_spec:
        f = mean_fidelity(pulses, nq, target, sigma, n_samples)
        out.append({"gate": label, "num_pulses": len(pulses),
                    "predicted_fidelity": f, "predicted_infidelity": 1.0 - f})
    return out
