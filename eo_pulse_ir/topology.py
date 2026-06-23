"""Physical layout: quantum-dot arrays hosting exchange-only qubits.

In the exchange-only (EO) encoding each *logical* qubit is stored in a triple of
quantum dots (the DiVincenzo / decoherence-free-subsystem encoding).  The only
control knob is the nearest-neighbour exchange interaction J_{i,i+1} between
adjacent dots, which we drive with square pulses.

This module owns the mapping (logical qubit -> dot triple) and the edge
bookkeeping, plus routing that inserts logical SWAPs so that a two-qubit gate's
operands become adjacent.

Two topologies are provided:

- :class:`LinearTopology` — a 1-D chain (the original).
- :class:`GridTopology`  — a 2-D square-lattice patch of encoded qubits
  (e.g. 2x2, 3x3, 4x4).  Supports interaction-weighted initial layout,
  BFS-based encoded-SWAP routing, and layer-parallel scheduling.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

DOTS_PER_QUBIT = 3


# ---------------------------------------------------------------------------
# Base protocol — both topologies expose these
# ---------------------------------------------------------------------------

class _TopoBase:
    """Shared interface for LinearTopology and GridTopology."""

    num_qubits: int

    def dots_of_qubit(self, q: int) -> Tuple[int, int, int]:
        raise NotImplementedError

    def intra_edges(self, q: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        raise NotImplementedError

    def inter_edge(self, ctrl_pos: int, tgt_pos: int) -> Tuple[int, int]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 1-D linear chain (original)
# ---------------------------------------------------------------------------

@dataclass
class LinearTopology(_TopoBase):
    """A 1-D chain of dots, three dots per logical qubit.

    Logical qubit ``q`` (in *position* order, see :attr:`order`) occupies dots
    ``[3p, 3p+1, 3p+2]`` where ``p`` is its current position on the line.
    Routing changes the position mapping, never the dot count.
    """

    num_qubits: int

    def __post_init__(self) -> None:
        self.order: List[int] = list(range(self.num_qubits))
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
        """The two within-triple exchange edges of logical qubit ``q``."""
        d0, d1, d2 = self.dots_of_qubit(q)
        return ((d0, d1), (d1, d2))

    def inter_edge(self, pos_or_ctrl: int, pos2: int = -1) -> Tuple[int, int]:
        """The exchange edge bridging two adjacent positions.

        For backwards compat: ``inter_edge(pos)`` bridges ``pos`` and ``pos+1``.
        ``inter_edge(ctrl_pos, tgt_pos)`` bridges two explicit positions.
        """
        if pos2 == -1:
            pos2 = pos_or_ctrl + 1
        lo, hi = min(pos_or_ctrl, pos2), max(pos_or_ctrl, pos2)
        return (lo * DOTS_PER_QUBIT + 2, hi * DOTS_PER_QUBIT)

    def swap_positions(self, pos: int) -> None:
        """Record that the logical qubits at ``pos`` and ``pos+1`` were swapped."""
        a, b = self.order[pos], self.order[pos + 1]
        self.order[pos], self.order[pos + 1] = b, a
        self.position_of[a], self.position_of[b] = pos + 1, pos

    def route_adjacent(self, q_ctrl: int, q_tgt: int) -> List[Tuple[int, int]]:
        """Plan nearest-neighbour position SWAPs so the two qubits end up adjacent
        with the control immediately left of the target.  Returns the list of
        (position, position+1) swaps to apply."""
        pc, pt = self.position_of[q_ctrl], self.position_of[q_tgt]
        swaps: List[Tuple[int, int]] = []
        cur = pc
        if pc < pt:
            while cur < pt - 1:
                swaps.append((cur, cur + 1))
                cur += 1
        else:
            while cur > pt + 1:
                swaps.append((cur - 1, cur))
                cur -= 1
            swaps.append((pt, pt + 1))
        return swaps


# ---------------------------------------------------------------------------
# 2-D square-lattice patch
# ---------------------------------------------------------------------------

@dataclass
class GridTopology(_TopoBase):
    """A 2-D square-lattice patch of encoded qubits.

    ``rows`` x ``cols`` encoded-qubit *slots* on a grid.  Each slot holds
    3 physical dots (an encoded qubit).  Neighbours on the grid share an
    endpoint-to-endpoint interface (the last dot of one group connects to the
    first dot of the next, by convention).

    Dot numbering
    ~~~~~~~~~~~~~
    Slot at grid position ``(r, c)`` has flat index ``s = r * cols + c``.
    Its 3 dots are ``[3s, 3s+1, 3s+2]``.

    Edge convention
    ~~~~~~~~~~~~~~~
    For two *adjacent* slots ``s_a`` and ``s_b`` (s_a < s_b), the inter-group
    edge connects dot ``3*s_a + 2`` to dot ``3*s_b`` (i.e. the "high" end of
    the lower-index group to the "low" end of the higher-index group).

    Initial layout + routing
    ~~~~~~~~~~~~~~~~~~~~~~~~
    :meth:`assign_initial_layout` uses an interaction-weighted greedy heuristic
    with optional pair-swap local search (same idea as kaluza1/exchange-pulse-
    optimizer, but stdlib-only — no CP-SAT, so no optimality proof).

    :meth:`route_adjacent` produces a BFS-shortest-path SWAP chain on the
    encoded-qubit grid.
    """

    rows: int
    cols: int

    def __post_init__(self) -> None:
        self.num_qubits: int = self.rows * self.cols
        # slot_of[logical_qubit] = flat grid index currently holding it
        self.slot_of: Dict[int, int] = {}
        # occupant[slot] = logical qubit id, or -1 if empty
        self.occupant: Dict[int, int] = {s: -1 for s in range(self.num_qubits)}

    # ---- geometry helpers ----

    @property
    def num_dots(self) -> int:
        return self.num_qubits * DOTS_PER_QUBIT

    @property
    def num_slots(self) -> int:
        return self.rows * self.cols

    def _rc(self, slot: int) -> Tuple[int, int]:
        return divmod(slot, self.cols)

    def _slot(self, r: int, c: int) -> int:
        return r * self.cols + c

    def slot_neighbours(self, slot: int) -> List[int]:
        """Return adjacent slots (up/down/left/right) on the grid."""
        r, c = self._rc(slot)
        out: List[int] = []
        if r > 0:
            out.append(self._slot(r - 1, c))
        if r < self.rows - 1:
            out.append(self._slot(r + 1, c))
        if c > 0:
            out.append(self._slot(r, c - 1))
        if c < self.cols - 1:
            out.append(self._slot(r, c + 1))
        return out

    def grid_distance(self, s1: int, s2: int) -> int:
        r1, c1 = self._rc(s1)
        r2, c2 = self._rc(s2)
        return abs(r1 - r2) + abs(c1 - c2)

    def dots_of_slot(self, slot: int) -> Tuple[int, int, int]:
        base = slot * DOTS_PER_QUBIT
        return (base, base + 1, base + 2)

    def dots_of_qubit(self, q: int) -> Tuple[int, int, int]:
        return self.dots_of_slot(self.slot_of[q])

    def intra_edges(self, q: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        d0, d1, d2 = self.dots_of_qubit(q)
        return ((d0, d1), (d1, d2))

    def inter_edge(self, slot_a: int, slot_b: int = -1) -> Tuple[int, int]:
        """Inter-group edge between two adjacent slots.

        Convention: low-index slot's dot 2 connects to high-index slot's dot 0.
        """
        if slot_b == -1:
            slot_b = slot_a + 1
        lo, hi = min(slot_a, slot_b), max(slot_a, slot_b)
        return (lo * DOTS_PER_QUBIT + 2, hi * DOTS_PER_QUBIT)

    # ---- initial layout ----

    def assign_initial_layout(self, logical_qubits: Sequence[int],
                              circuit_gates=None,
                              layout_decay: float = 0.98,
                              local_search_rounds: int = 2) -> None:
        """Assign logical qubits to grid slots using an interaction-weighted
        greedy heuristic, then refine with pair-swap local search.

        ``circuit_gates`` is an iterable of Gate objects (or anything with
        ``.is_two_qubit`` and ``.qubits``).  If None, qubits are placed in order.
        """
        n = len(logical_qubits)
        if n > self.num_slots:
            raise ValueError(f"need {n} slots but grid has {self.num_slots}")

        if circuit_gates is None:
            for i, q in enumerate(logical_qubits):
                self.slot_of[q] = i
                self.occupant[i] = q
            return

        # build interaction weights
        weights: Dict[FrozenSet[int], float] = {}
        k = 0
        for g in circuit_gates:
            if g.is_two_qubit:
                pair = frozenset(g.qubits)
                weights[pair] = weights.get(pair, 0.0) + layout_decay ** k
                k += 1

        # greedy placement: place qubits one at a time, each time picking the
        # slot that minimises weighted-distance to already-placed neighbours
        placed: Dict[int, int] = {}  # logical -> slot
        free_slots: Set[int] = set(range(self.num_slots))

        # order qubits by total interaction weight (most connected first)
        q_weight: Dict[int, float] = {}
        for pair, w in weights.items():
            for q in pair:
                q_weight[q] = q_weight.get(q, 0.0) + w
        ordered = sorted(logical_qubits, key=lambda q: -q_weight.get(q, 0.0))

        for q in ordered:
            best_slot, best_cost = -1, float("inf")
            for s in free_slots:
                cost = 0.0
                for pair, w in weights.items():
                    if q not in pair:
                        continue
                    other = next(iter(pair - {q}))
                    if other in placed:
                        d = self.grid_distance(s, placed[other])
                        cost += w * max(0, d - 1)
                cost_with_tie = (cost, s)
                if cost_with_tie < (best_cost, best_slot):
                    best_cost, best_slot = cost, s
            placed[q] = best_slot
            free_slots.discard(best_slot)

        # local search: pair-swap refinement
        def _objective(assignment: Dict[int, int]) -> float:
            total = 0.0
            for pair, w in weights.items():
                qs = list(pair)
                if qs[0] in assignment and qs[1] in assignment:
                    d = self.grid_distance(assignment[qs[0]], assignment[qs[1]])
                    total += w * max(0, d - 1)
            return total

        qs_list = list(placed.keys())
        for _ in range(local_search_rounds):
            improved = True
            while improved:
                improved = False
                for i in range(len(qs_list)):
                    for j in range(i + 1, len(qs_list)):
                        qa, qb = qs_list[i], qs_list[j]
                        cur = _objective(placed)
                        placed[qa], placed[qb] = placed[qb], placed[qa]
                        nxt = _objective(placed)
                        if nxt < cur - 1e-12:
                            improved = True
                        else:
                            placed[qa], placed[qb] = placed[qb], placed[qa]

        for q, s in placed.items():
            self.slot_of[q] = s
            self.occupant[s] = q

    # ---- routing ----

    def _bfs_path(self, src: int, dst: int) -> List[int]:
        """BFS shortest path on the slot grid from ``src`` to ``dst``."""
        if src == dst:
            return [src]
        visited: Dict[int, int] = {src: -1}
        queue: deque[int] = deque([src])
        while queue:
            cur = queue.popleft()
            for nb in self.slot_neighbours(cur):
                if nb not in visited:
                    visited[nb] = cur
                    if nb == dst:
                        path = []
                        s = dst
                        while s != -1:
                            path.append(s)
                            s = visited[s]
                        return path[::-1]
                    queue.append(nb)
        raise ValueError(f"no path from slot {src} to slot {dst}")

    def route_adjacent(self, q_ctrl: int, q_tgt: int) -> List[Tuple[int, int]]:
        """Return a list of (slot_a, slot_b) SWAP steps that bring ``q_ctrl``
        next to ``q_tgt`` on the grid.

        After applying all SWAPs (call :meth:`swap_slots` for each), the two
        qubits will be in adjacent slots.  The caller emits the SWAP pulses
        and applies the state update.
        """
        s_ctrl = self.slot_of[q_ctrl]
        s_tgt = self.slot_of[q_tgt]
        if self.grid_distance(s_ctrl, s_tgt) <= 1:
            return []

        # move ctrl toward tgt along shortest path
        path = self._bfs_path(s_ctrl, s_tgt)
        swaps: List[Tuple[int, int]] = []
        for step in range(len(path) - 2):
            swaps.append((path[step], path[step + 1]))
        return swaps

    def swap_slots(self, slot_a: int, slot_b: int) -> None:
        """Record that the qubits in ``slot_a`` and ``slot_b`` were swapped."""
        qa, qb = self.occupant[slot_a], self.occupant[slot_b]
        self.occupant[slot_a], self.occupant[slot_b] = qb, qa
        if qa >= 0:
            self.slot_of[qa] = slot_b
        if qb >= 0:
            self.slot_of[qb] = slot_a

    # ---- layer-parallel scheduling helpers ----

    def slot_dots_used(self, slot_a: int, slot_b: int) -> Set[int]:
        """All dots involved in an operation between two adjacent slots."""
        d_a = set(self.dots_of_slot(slot_a))
        d_b = set(self.dots_of_slot(slot_b))
        d_a.add(self.inter_edge(slot_a, slot_b)[0])
        d_a.add(self.inter_edge(slot_a, slot_b)[1])
        return d_a | d_b

    def are_independent(self, ops: List[Tuple[int, int]], new_op: Tuple[int, int]) -> bool:
        """Check whether ``new_op`` (pair of slots) can run in parallel with ``ops``."""
        new_dots = self.slot_dots_used(new_op[0], new_op[1])
        for (a, b) in ops:
            if self.slot_dots_used(a, b) & new_dots:
                return False
        return True
