"""Target logical unitaries, in the basis used by :func:`encoding.logical_basis`.

For a 2-qubit gate the basis order is |00>, |01>, |10>, |11> with qubit 0 the
most significant (control), matching the column order of the logical isometry.
"""

from __future__ import annotations

import numpy as np

_ISQ2 = 1 / np.sqrt(2)

I1 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[_ISQ2, _ISQ2], [_ISQ2, -_ISQ2]], dtype=complex)
S = np.array([[1, 0], [0, 1j]], dtype=complex)
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)


def rz(theta: float) -> np.ndarray:
    return np.array([[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]], dtype=complex)


def rx(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def ry(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


CNOT = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0],
], dtype=complex)

CZ = np.diag([1, 1, 1, -1]).astype(complex)

SWAP = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
], dtype=complex)


def by_name(name: str, param: float | None = None) -> np.ndarray:
    name = name.lower()
    table = {"i": I1, "x": X, "y": Y, "z": Z, "h": H, "s": S, "t": T,
             "cx": CNOT, "cnot": CNOT, "cz": CZ, "swap": SWAP}
    if name in table:
        return table[name]
    if name == "rz":
        return rz(param)
    if name == "rx":
        return rx(param)
    if name == "ry":
        return ry(param)
    raise ValueError(f"unknown target gate {name!r}")
