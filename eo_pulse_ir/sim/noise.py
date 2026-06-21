"""Monte-Carlo robustness of exchange-pulse sequences under area (charge) noise.

In an exchange-only device the dominant error channel is charge noise, which
fluctuates the exchange coupling J and hence the *pulse area* A = J·tau.  This
module injects area fluctuations and Monte-Carlo samples the resulting gate
fidelity, giving a real (simulated) robustness distribution rather than the
core IR's heuristic ``noise_sensitivity`` proxy.

Noise models
------------
- ``relative=True``  : multiplicative area error A -> A·(1+eps) (models dJ/J,
  the natural charge-noise quantity); ``relative=False`` is additive (timing /
  absolute control error).
- correlation:
    * ``"per_edge"``    one quasi-static eps per edge, shared by all pulses on
      that edge within a shot — the realistic 1/f charge-noise picture (each dot
      pair has its own slow fluctuation that is constant over a gate).
    * ``"global"``      one eps for the whole shot (global J miscalibration).
    * ``"independent"`` an independent eps per pulse (fast / per-pulse control error).
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from .fidelity import average_gate_fidelity
from .simulator import logical_block

Edge = Tuple[int, int]


def perturb_areas(areas: Sequence[float], edges: Sequence[Edge], sigma: float,
                  rng: np.random.Generator, relative: bool = True,
                  correlation: str = "per_edge") -> np.ndarray:
    areas = np.asarray(areas, dtype=float)
    n = len(areas)
    if sigma <= 0:
        return areas.copy()
    if correlation == "independent":
        eps = rng.normal(0.0, sigma, size=n)
    elif correlation == "global":
        eps = np.full(n, rng.normal(0.0, sigma))
    elif correlation == "per_edge":
        per: Dict[Edge, float] = {}
        eps = np.empty(n)
        for i, e in enumerate(edges):
            e = tuple(e)
            if e not in per:
                per[e] = rng.normal(0.0, sigma)
            eps[i] = per[e]
    else:
        raise ValueError(f"unknown correlation {correlation!r}")
    return areas * (1.0 + eps) if relative else areas + eps


def montecarlo_fidelity(edges: Sequence[Edge], areas: Sequence[float],
                        num_qubits: int, target: np.ndarray, sigma: float,
                        n_samples: int = 500, relative: bool = True,
                        correlation: str = "per_edge", seed: int = 0) -> dict:
    """Sample gate fidelity under area noise of strength ``sigma``."""
    rng = np.random.default_rng(seed)
    Fs = np.empty(n_samples)
    for k in range(n_samples):
        a = perturb_areas(areas, edges, sigma, rng, relative, correlation)
        M = logical_block(list(zip(edges, a)), num_qubits)
        Fs[k] = average_gate_fidelity(M, target)
    return {
        "sigma": float(sigma), "n": int(n_samples),
        "mean_fidelity": float(Fs.mean()),
        "std_fidelity": float(Fs.std()),
        "mean_infidelity": float(1.0 - Fs.mean()),
        "p50": float(np.percentile(Fs, 50)),
        "p10": float(np.percentile(Fs, 10)),
        "p90": float(np.percentile(Fs, 90)),
        "worst_fidelity": float(Fs.min()),
        "samples": Fs.tolist(),
    }


def robustness_sweep(edges: Sequence[Edge], areas: Sequence[float],
                     num_qubits: int, target: np.ndarray, sigmas: Sequence[float],
                     n_samples: int = 400, relative: bool = True,
                     correlation: str = "per_edge", seed: int = 0) -> List[dict]:
    """Monte-Carlo fidelity across a range of noise strengths."""
    return [montecarlo_fidelity(edges, areas, num_qubits, target, s, n_samples,
                                relative, correlation, seed + i)
            for i, s in enumerate(sigmas)]


def susceptibility(sweep: List[dict]) -> float:
    """Quadratic noise-susceptibility coefficient c in  infidelity ≈ c · sigma².

    Fit through the small-noise points (least squares of mean_infidelity vs
    sigma²).  A single comparable robustness number per gate (smaller = better).
    """
    s2 = np.array([r["sigma"] ** 2 for r in sweep])
    inf = np.array([r["mean_infidelity"] for r in sweep])
    # subtract the sigma=0 floor so we fit the noise-induced part
    floor = inf[s2 == 0.0].min() if (s2 == 0.0).any() else 0.0
    mask = s2 > 0
    if not mask.any():
        return 0.0
    c = float(np.sum(s2[mask] * (inf[mask] - floor)) / np.sum(s2[mask] ** 2))
    return c
