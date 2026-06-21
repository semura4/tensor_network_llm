"""Exchange-pulse *templates* for logical gates in the EO encoding.

Each logical gate is expanded into an ordered list of exchange pulses, where a
pulse is described by a *role* (which physical edge it uses, relative to the
gate's operands) and an *area* (the exchange action ``A = integral J dt``; a full
SWAP between two dots is ``A = pi``).

Provenance of each template
---------------------------
- **CNOT (cx)** is *numerically validated*: its (role, area) pulses were optimised
  against the physics simulator (``eo_pulse_ir.sim``) to average gate fidelity
  0.9996 with leakage 3.8e-4 in the 3-dot S=1/2 encoding (see ``_CX_VALIDATED``
  and ``scripts/eo_optimize_cnot.py``).
- **Single-qubit gates** use the minimal exchange-generator counts the simulator
  confirms (1 pulse for Z-axis gates; 3 alternating pulses span the rest).
- **SWAP / CXSWAP** remain *representative templates*: their pulse counts and edge
  structure follow the published exchange-only constructions, but the individual
  areas are placeholders, not yet optimised to a target unitary.

This module is also the seam where external data enters: feed an optimiser /
`eoqrid` pulse list through :func:`eo_pulse_ir.schedule.pulses_from_records` to
replace any template with calibrated values.  The cost/scheduling/hardware IR
downstream is exact for whatever pulse list it is given.
"""

from __future__ import annotations

import math
from typing import List, Tuple

PI = math.pi
FULL_SWAP = PI

# A pulse spec is (role, area).  Roles are resolved to concrete dot edges by the
# synthesiser, using the operand topology.
PulseSpec = Tuple[str, float]

ALIGNED_1Q = {"z", "s", "sdg", "t", "tdg", "rz"}
_ALIGNED_AREA = {"z": PI, "s": PI / 2, "sdg": PI / 2, "t": PI / 4, "tdg": PI / 4}


def one_qubit_template(name: str, param: float | None = None) -> List[PulseSpec]:
    """Expand a single-qubit logical gate into intra-triple exchange pulses.

    Aligned gates (z/s/t/rz) need a single pulse on one exchange generator;
    everything else uses a 3-pulse alternating-generator sequence (the two EO
    generators act about axes ~120 degrees apart, so 3-4 pulses span SU(2)).
    """
    name = name.lower()
    if name in ALIGNED_1Q:
        area = abs(param) if name == "rz" else _ALIGNED_AREA[name]
        return [("intra_low", area)]

    if name in ("x", "y"):
        mid = PI
    elif name == "h":
        mid = PI / 2
    elif name in ("rx", "ry"):
        mid = abs(param) if param is not None else PI / 2
    else:
        raise ValueError(f"no 1-qubit template for {name!r}")
    return [("intra_high", PI / 2), ("intra_low", mid), ("intra_high", PI / 2)]


# Validated leakage-free CNOT, optimised against the physics simulator
# (eo_pulse_ir.sim) by analytic-gradient search over a KAK-style ansatz: four
# boundary exchanges, each dressed by full single-qubit-capable blocks on the
# control and target triples.  Average gate fidelity F = 0.999614, leakage
# 3.8e-4 vs CNOT in the 3-dot S=1/2 encoding.  Reproduce / re-optimise with
# scripts/eo_optimize_cnot.py.  A naive 19-pulse round-robin tops out at
# F ~ 0.78 in this model, so the validated sequence is longer than the textbook
# Fong-Wandzura pulse count but is numerically verified end-to-end.  A few areas
# sit at ~0 or ~2*pi (effective no-ops the optimiser left in place).
_CX_VALIDATED: List[PulseSpec] = [
    ("ctrl_high", 6.2831852918),
    ("ctrl_low", 4.8235891651),
    ("ctrl_high", 0.0000000705),
    ("tgt_low", 4.4608844497),
    ("tgt_high", 2.2451100490),
    ("tgt_low", 3.9982626136),
    ("inter", 4.6808804741),
    ("ctrl_high", 4.6970563318),
    ("ctrl_low", 2.2019744467),
    ("ctrl_high", 4.6970563097),
    ("tgt_low", 4.8068596310),
    ("tgt_high", 2.2056854772),
    ("tgt_low", 4.6741103089),
    ("inter", 4.6278413964),
    ("ctrl_high", 2.1926664128),
    ("ctrl_low", 4.6855682128),
    ("ctrl_high", 2.1926664617),
    ("tgt_low", 2.2721151744),
    ("tgt_high", 4.4989131380),
    ("tgt_low", 2.0773274373),
    ("inter", 4.4070726994),
    ("ctrl_high", 5.8483359443),
    ("ctrl_low", 2.9215677337),
    ("ctrl_high", 5.8483359552),
    ("tgt_low", 3.4776839627),
    ("tgt_high", 5.4933159611),
    ("tgt_low", 2.2296147043),
    ("inter", 0.5013412537),
]


def _cx_template() -> List[PulseSpec]:
    """Validated nearest-neighbour exchange CNOT (28 pulses, F = 0.9996).

    Returns the simulator-validated (role, area) sequence in ``_CX_VALIDATED``.
    Roles resolve to the bridging edges of the control/target triples
    (ctrl_low/ctrl_high, inter, tgt_low/tgt_high).
    """
    return list(_CX_VALIDATED)


def _swap_template() -> List[PulseSpec]:
    """Representative encoded-SWAP between two adjacent logical qubits.

    Encoded SWAP is comparatively cheap and high-fidelity (it permutes the dot
    contents without leaving the logical subspace), so we model it with a short
    sequence dominated by full SWAPs on the bridging edges.
    """
    return [
        ("ctrl_high", FULL_SWAP),
        ("inter", FULL_SWAP),
        ("tgt_low", FULL_SWAP),
        ("inter", FULL_SWAP),
        ("ctrl_high", FULL_SWAP),
        ("tgt_low", FULL_SWAP),
        ("inter", FULL_SWAP),
    ]


def _cxswap_template() -> List[PulseSpec]:
    """Representative combined CXSWAP (cheaper than CX then SWAP separately)."""
    roles = ["ctrl_high", "inter", "tgt_low"]
    pulses: List[PulseSpec] = []
    for k in range(13):
        role = roles[k % 3]
        area = FULL_SWAP if k % 3 != 1 else FULL_SWAP / 2
        pulses.append((role, area))
    return pulses


def two_qubit_template(name: str) -> List[PulseSpec]:
    name = name.lower()
    if name in ("cx", "cnot"):
        return _cx_template()
    if name == "swap":
        return _swap_template()
    if name == "cxswap":
        return _cxswap_template()
    raise ValueError(f"no 2-qubit template for {name!r}")


# Metadata used by the report layer so users can see provenance at a glance.
TEMPLATE_INFO = {
    "1q-aligned": {"pulses": 1, "source": "template (axis-aligned exchange generator)"},
    "1q-generic": {"pulses": 3, "source": "template (alternating EO generators)"},
    "cx": {"pulses": 28, "source": "optimized (analytic-gradient, F=0.99961, leak=3.8e-4)"},
    "swap": {"pulses": 7, "source": "template (encoded-SWAP, representative)"},
    "cxswap": {"pulses": 13, "source": "template (combined CXSWAP, representative)"},
}
