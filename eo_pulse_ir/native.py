"""Exchange-pulse *templates* for logical gates in the EO encoding.

Each logical gate is expanded into an ordered list of exchange pulses, where a
pulse is described by a *role* (which physical edge it uses, relative to the
gate's operands) and an *area* (the exchange action ``A = integral J dt``; a full
SWAP between two dots is ``A = pi``).

IMPORTANT — honesty about these templates
-----------------------------------------
The *pulse counts* and *edge structure* reflect the published exchange-only
constructions (e.g. the 19-pulse nearest-neighbour Fong--Wandzura CNOT, and the
comparatively cheap encoded-SWAP).  The *individual pulse areas are
representative placeholders*, NOT numerically optimised angles that reproduce a
target unitary at high fidelity.  This module is the seam where real data should
enter: feed an optimiser / `eoqrid` pulse list through
:func:`eo_pulse_ir.schedule.pulses_from_records` to replace the templates with
calibrated values.  The cost/scheduling/hardware IR downstream is exact for
whatever pulse list it is given.
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


def _cx_template() -> List[PulseSpec]:
    """Representative 19-pulse nearest-neighbour exchange CNOT.

    Edge structure follows the three edges that bridge the control and target
    triples: ctrl_high = (3c+1,3c+2), inter = (3c+2,3t), tgt_low = (3t,3t+1).
    Areas alternate between full and half SWAP as a stand-in for the optimised
    sequence.
    """
    roles = ["ctrl_high", "inter", "tgt_low"]
    pulses: List[PulseSpec] = []
    for k in range(19):
        role = roles[k % 3]
        area = FULL_SWAP if k % 2 == 0 else FULL_SWAP / 2
        pulses.append((role, area))
    return pulses


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
    "cx": {"pulses": 19, "source": "template (NN exchange CNOT, Fong-Wandzura count)"},
    "swap": {"pulses": 7, "source": "template (encoded-SWAP, representative)"},
    "cxswap": {"pulses": 13, "source": "template (combined CXSWAP, representative)"},
}
