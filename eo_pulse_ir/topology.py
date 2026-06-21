"""Physical layout: a linear array of quantum dots hosting exchange-only qubits.

In the exchange-only (EO) encoding each *logical* qubit is stored in a triple of
quantum dots (the DiVincenzo / decoherence-free-subsystem encoding).  The only
control knob is the nearest-neighbour exchange interaction J_{i,i+1} between
adjacent dots, which we drive with square pulses.

This module owns the mapping (logical qubit -> dot triple) and the edge
bookkeeping, plus a tiny nearest-neighbour router that inserts logical SWAPs so
that a two-qubit gate's operands become adjacent on the line.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

DOTS_PER_QUBIT = 3


@dataclass
class LinearTopology:
    """A 1-D chain of dots, three dots per logical qubit.

    Logical qubit ``q`` (in *position* order, see :attr:`order`) occupies dots
    ``[3p, 3p+1, 3p+2]`` where ``p`` is its current position on the line.
    Routing changes the position mapping, never the dot count.
    """

    num_qubits: int

    def __post_init__(self) -> None:
        # order[position] = logical qubit id currently at that position
        self.order: List[int] = list(range(self.num_qubits))
        # position_of[logical qubit] = its current line position
        self.position_of: Dict[int, int] = {q: q for q in range(self.num_qubits)}

    @property
    def num_dots(self) -> int:
        return self.num_qubits * DOTS_PER_QUBIT

    def dots_of_position(self, pos: int) -> Tuple[int, int, int]:
        base = pos * DOTS_PER_QUBIT
        return (base, base + 1, base + 2)

    def dots_of_qubit(self, q: int) -> Tuple[int, int, int]:
        return self.dots_of_position(self.position_of[q])

    def intra_edges(self, q: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """The two within-triple exchange edges of logical qubit ``q``.

        Returns ((low, mid), (mid, high)) = (intra_low, intra_high).
        """
        d0, d1, d2 = self.dots_of_qubit(q)
        return ((d0, d1), (d1, d2))

    def inter_edge(self, pos: int) -> Tuple[int, int]:
        """The exchange edge bridging the triple at ``pos`` and ``pos+1``."""
        return (pos * DOTS_PER_QUBIT + 2, (pos + 1) * DOTS_PER_QUBIT)

    def swap_positions(self, pos: int) -> None:
        """Record that the logical qubits at ``pos`` and ``pos+1`` were swapped."""
        a, b = self.order[pos], self.order[pos + 1]
        self.order[pos], self.order[pos + 1] = b, a
        self.position_of[a], self.position_of[b] = pos + 1, pos

    def route_adjacent(self, q_ctrl: int, q_tgt: int) -> List[Tuple[int, int]]:
        """Plan a sequence of nearest-neighbour position SWAPs so that the two
        logical qubits end up adjacent with the control immediately left of the
        target.  Returns the list of (position, position+1) swaps to apply (the
        caller is responsible for emitting the SWAP pulses and then calling
        :meth:`swap_positions`).  Does not mutate state.
        """
        pc, pt = self.position_of[q_ctrl], self.position_of[q_tgt]
        swaps: List[Tuple[int, int]] = []
        # Move the control next to and just left of the target by bubbling it.
        # Work on a scratch copy of positions.
        cur = pc
        if pc < pt:
            # bubble control rightwards until it sits just left of target
            while cur < pt - 1:
                swaps.append((cur, cur + 1))
                cur += 1
        else:
            # control is to the right of target: bubble it left to pt+1, then it
            # will be just right of target; flip so control is left of target.
            while cur > pt + 1:
                swaps.append((cur - 1, cur))
                cur -= 1
            # now control at pt+1, target at pt -> one more swap puts control left
            swaps.append((pt, pt + 1))
        return swaps
