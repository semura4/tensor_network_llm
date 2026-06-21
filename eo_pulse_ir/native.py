"""Exchange-pulse *templates* for logical gates in the EO encoding.

Each logical gate is expanded into an ordered list of exchange pulses, where a
pulse is described by a *role* (which physical edge it uses, relative to the
gate's operands) and an *area* (the exchange action ``A = integral J dt``; a full
SWAP between two dots is ``A = pi``).

Provenance of each template
---------------------------
- **CNOT (cx)** is *numerically validated*: its (role, area) pulses were optimised
  against the physics simulator (``eo_pulse_ir.sim``) to average gate fidelity
  0.99999999 with leakage 7.7e-9 in the 3-dot S=1/2 encoding (see
  ``_CX_VALIDATED`` and ``scripts/eo_optimize_2q.py``).
- **SWAP** is *numerically validated*: F = 0.99999999, leakage 3.2e-10, 27 pulses
  (encoded SWAP is cheaper than CNOT); see ``_SWAP_VALIDATED``.
- **CXSWAP** is *numerically validated*: F = 0.99992, leakage 6.3e-5, 48 pulses
  (a higher-entangling permutation, needs more boundary exchanges); see
  ``_CXSWAP_VALIDATED``.
- **Single-qubit gates** use the minimal exchange-generator counts the simulator
  confirms (1 pulse for Z-axis gates; 3 alternating pulses span the rest).

All three two-qubit gates (CX, SWAP, CXSWAP) are now simulator-validated rather
than placeholder templates.

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
# control and target triples.  Average gate fidelity F = 0.99999999, leakage
# 7.7e-9 vs CNOT in the 3-dot S=1/2 encoding.  Reproduce / re-optimise with
# scripts/eo_optimize_2q.py.  A naive 19-pulse round-robin tops out at
# F ~ 0.78 in this model, so the validated sequence is longer than the textbook
# Fong-Wandzura pulse count but is numerically verified end-to-end.
_CX_VALIDATED: List[PulseSpec] = [
    ("ctrl_high", 4.4137500413),
    ("ctrl_low", 3.1826568501),
    ("ctrl_high", 1.9518464551),
    ("tgt_low", 1.9250246148),
    ("tgt_high", 5.5600420536),
    ("tgt_low", 3.8885308054),
    ("inter", 2.6174392878),
    ("ctrl_high", 1.5561061501),
    ("ctrl_low", 4.0579438889),
    ("ctrl_high", 1.5571544926),
    ("tgt_low", 1.5565546286),
    ("tgt_high", 4.0577518066),
    ("tgt_low", 1.5567943509),
    ("inter", 5.4245739697),
    ("ctrl_high", 3.3619487055),
    ("ctrl_low", 0.4439060211),
    ("ctrl_high", 3.3706870514),
    ("tgt_low", 0.4438759958),
    ("tgt_high", 3.3674557677),
    ("tgt_low", 0.4439004865),
    ("inter", 5.4249313322),
    ("ctrl_high", 4.0573548777),
    ("ctrl_low", 1.5572133273),
    ("ctrl_high", 4.0573382919),
    ("tgt_low", 1.5567305949),
    ("tgt_high", 4.0576666192),
    ("tgt_low", 1.5564435410),
    ("inter", 2.6174798498),
    ("ctrl_high", 4.7114459672),
    ("ctrl_low", 4.9788738963),
    ("ctrl_high", 3.9833706400),
    ("tgt_low", 3.9480593164),
    ("tgt_high", 5.5808892469),
    ("tgt_low", 1.8867090363),
]


def _cx_template() -> List[PulseSpec]:
    """Validated nearest-neighbour exchange CNOT (34 pulses, F = 0.99999999).

    Returns the simulator-validated (role, area) sequence in ``_CX_VALIDATED``.
    Roles resolve to the bridging edges of the control/target triples
    (ctrl_low/ctrl_high, inter, tgt_low/tgt_high).
    """
    return list(_CX_VALIDATED)


# Validated leakage-free logical SWAP (eo_pulse_ir.sim, analytic-gradient search,
# KAK-style ansatz: 3 boundary exchanges + full single-qubit dressing).  Average
# gate fidelity F = 0.99999999, leakage 3.2e-10 vs SWAP.  Encoded SWAP is cheaper
# than CNOT here (N=27 suffices).  Reproduce with scripts/eo_optimize_2q.py.
_SWAP_VALIDATED: List[PulseSpec] = [
    ("ctrl_high", 3.8570295936),
    ("ctrl_low", 2.6890492883),
    ("ctrl_high", 5.8889853995),
    ("tgt_low", 1.7720617556),
    ("tgt_high", 4.4910341855),
    ("tgt_low", 5.7880600812),
    ("inter", 3.1474925139),
    ("ctrl_high", 3.6800539696),
    ("ctrl_low", 6.2831740320),
    ("ctrl_high", 5.7506141528),
    ("tgt_low", 6.2831846681),
    ("tgt_high", 2.6962206619),
    ("tgt_low", 3.1474903177),
    ("inter", 3.1474890567),
    ("ctrl_high", 0.0605870244),
    ("ctrl_low", 3.1475056239),
    ("ctrl_high", 3.1474902565),
    ("tgt_low", 3.3260600289),
    ("tgt_high", 3.1474866587),
    ("tgt_low", 3.1475016875),
    ("inter", 3.1474873212),
    ("ctrl_high", 4.9070636403),
    ("ctrl_low", 4.0696020731),
    ("ctrl_high", 5.6751692226),
    ("tgt_low", 1.4707855251),
    ("tgt_high", 2.5480725973),
    ("tgt_low", 2.4149979136),
]


def _swap_template() -> List[PulseSpec]:
    """Validated logical SWAP (27 pulses, F = 0.99999999).

    Returns the simulator-validated (role, area) sequence in ``_SWAP_VALIDATED``.
    """
    return list(_SWAP_VALIDATED)


# Validated leakage-free CXSWAP = SWAP . CNOT (eo_pulse_ir.sim, analytic-gradient
# search, KAK-style ansatz: 6 boundary exchanges + full single-qubit dressing).
# Average gate fidelity F = 0.99992, leakage 6.3e-5 vs CXSWAP.  CXSWAP is a
# higher-entangling permutation, so it needs more boundary exchanges than CNOT
# (N=34) or SWAP (N=27).  Reproduce with scripts/eo_optimize_2q.py --target cxswap.
_CXSWAP_VALIDATED: List[PulseSpec] = [
    ("ctrl_high", 2.3663143677),
    ("ctrl_low", 3.1423617496),
    ("ctrl_high", 4.5575040862),
    ("tgt_low", 5.3247997640),
    ("tgt_high", 4.9335601522),
    ("tgt_low", 3.0191896402),
    ("inter", 4.2535130453),
    ("ctrl_high", 3.5283039654),
    ("ctrl_low", 5.7312930310),
    ("ctrl_high", 1.4275142320),
    ("tgt_low", 2.0488343953),
    ("tgt_high", 4.4999005690),
    ("tgt_low", 2.0268219172),
    ("inter", 2.2116749599),
    ("ctrl_high", 4.0006594906),
    ("ctrl_low", 1.5917832289),
    ("ctrl_high", 4.0896001615),
    ("tgt_low", 0.0056238177),
    ("tgt_high", 5.0229804586),
    ("tgt_low", 0.0368629950),
    ("inter", 5.0153702070),
    ("ctrl_high", 6.2024474817),
    ("ctrl_low", 4.4820011506),
    ("ctrl_high", 3.2661785658),
    ("tgt_low", 1.7985568999),
    ("tgt_high", 4.2919690605),
    ("tgt_low", 5.0353096197),
    ("inter", 3.2871109794),
    ("ctrl_high", 0.0261899489),
    ("ctrl_low", 4.7535761513),
    ("ctrl_high", 3.0657475423),
    ("tgt_low", 2.9459432081),
    ("tgt_high", 3.2694940872),
    ("tgt_low", 3.1804374442),
    ("inter", 3.0485668746),
    ("ctrl_high", 5.7178870474),
    ("ctrl_low", 2.7828989155),
    ("ctrl_high", 2.7818290185),
    ("tgt_low", 5.8232282838),
    ("tgt_high", 4.3409123873),
    ("tgt_low", 1.0545868859),
    ("inter", 5.4401330592),
    ("ctrl_high", 1.8850752186),
    ("ctrl_low", 5.3424842576),
    ("ctrl_high", 0.1752082379),
    ("tgt_low", 1.5228879246),
    ("tgt_high", 3.1897734151),
    ("tgt_low", 2.4871341149),
]


def _cxswap_template() -> List[PulseSpec]:
    """Validated CXSWAP = SWAP . CNOT (48 pulses, F = 0.99992).

    Returns the simulator-validated (role, area) sequence in ``_CXSWAP_VALIDATED``.
    """
    return list(_CXSWAP_VALIDATED)


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
    "cx": {"pulses": 34, "source": "optimized (analytic-gradient, F=0.99999999, leak=7.7e-9)"},
    "swap": {"pulses": 27, "source": "optimized (analytic-gradient, F=0.99999999, leak=3.2e-10)"},
    "cxswap": {"pulses": 48, "source": "optimized (analytic-gradient, F=0.99992, leak=6.3e-5)"},
}
