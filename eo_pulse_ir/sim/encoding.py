"""Exchange-only logical encoding: 3 dots -> one logical qubit (DFS doublet).

Each logical qubit lives in the total-spin S=1/2, S_z=+1/2 subspace of three
spins (the decoherence-free-subsystem / DiVincenzo encoding).  We use the
canonical basis in which the within-triple exchange on dots (0,1) is the logical
Z generator:

    |0_L> =  (|UDU> - |DUU>) / sqrt(2)                      # singlet(0,1) (x) up_2
    |1_L> =  sqrt(2/3)|UUD> - sqrt(1/3) (|UDU> + |DUU>)/sqrt(2)

The third S_z=+1/2 state of the triple is the S=3/2 component: population there
is *leakage*.  This module assembles the logical basis of an n-qubit register as
columns of a (2**(3*nq), 2**nq) isometry L; the simulator restricts evolution to
that subspace via M = L^dagger U L.
"""

from __future__ import annotations

import numpy as np

DOTS_PER_QUBIT = 3


def triple_logical_states() -> np.ndarray:
    """Return a (8, 2) isometry whose columns are |0_L>, |1_L> of one triple."""
    v0 = np.zeros(8, dtype=complex)
    v1 = np.zeros(8, dtype=complex)
    # bit order within triple: idx = 4*b0 + 2*b1 + b2, b=0 -> up, b=1 -> down
    UDU, DUU, UUD = 0b010, 0b100, 0b001
    v0[UDU] = 1 / np.sqrt(2)
    v0[DUU] = -1 / np.sqrt(2)
    v1[UUD] = np.sqrt(2 / 3)
    v1[UDU] = -np.sqrt(1 / 3) / np.sqrt(2)
    v1[DUU] = -np.sqrt(1 / 3) / np.sqrt(2)
    return np.column_stack([v0, v1])


def logical_basis(num_qubits: int) -> np.ndarray:
    """Isometry L of shape (2**(3*nq), 2**nq).

    Column c (c in binary = b_0 b_1 ... with qubit 0 most significant) is the
    encoded register state |b_0>_L (x) |b_1>_L (x) ...   in dot order.
    """
    single = triple_logical_states()  # (8, 2)
    dim_log = 1 << num_qubits
    cols = []
    for c in range(dim_log):
        vec = np.array([1.0 + 0j])
        for q in range(num_qubits):
            b = (c >> (num_qubits - 1 - q)) & 1
            vec = np.kron(vec, single[:, b])
        cols.append(vec)
    return np.column_stack(cols)


def triple_s2_projector_half() -> np.ndarray:
    """Projector (8x8) onto the S=1/2 subspace of a single triple.

    Useful for diagnosing *where* leaked population goes; the simulator's leakage
    metric does not need it, but it is handy for tests and reports.
    """
    L = triple_logical_states()       # columns span S=1/2, S_z=+1/2 (2 states)
    # The full S=1/2 subspace also includes S_z=-1/2; for a leakage check at fixed
    # S_z=+1/2 it suffices to project onto these two states plus their S_z=-1/2
    # partners.  Here we return the rank-2 projector at S_z=+1/2, which is what a
    # logical computation at S_z=+1/2 stays within.
    return L @ L.conj().T
