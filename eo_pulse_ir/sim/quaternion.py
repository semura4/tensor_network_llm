"""Quaternion view of single-qubit exchange-only control.

On the decoherence-free subsystem the logical group is SU(2) ≅ the unit
quaternions, and each intra-triple exchange pulse is a *fixed-axis rotation*
(angle = pulse area).  Measured from the simulator, the two native axes are

    intra_low  : n1 = (0, 0, -1)            (the logical -Z axis)
    intra_high : n2 = (√3/2, 0, 1/2)        (120° from n1)

so single-qubit EO control is a piecewise-constant trajectory of rotations about
two axes 120° apart — a clean dynamical-systems picture, and quaternions are the
natural, fast, numerically stable representation (4 reals, no 8×8 matrices).

This module provides quaternion algebra, SU(2) <-> quaternion conversion, slerp,
Bloch-vector rotation, the native axes, and an exact analytic single-qubit
compiler (any 2×2 unitary -> exchange pulses via a ZXZ decomposition built from
the native Rz and the validated H).
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from ..native import _H_1Q, _rz_area  # numpy-free building blocks

PulseSpec = Tuple[str, float]

_X = np.array([[0, 1], [1, 0]], complex)
_Y = np.array([[0, -1j], [1j, 0]], complex)
_Z = np.array([[1, 0], [0, -1]], complex)
_I = np.eye(2, dtype=complex)

# native single-qubit rotation axes (measured from the simulator)
N1 = np.array([0.0, 0.0, -1.0])              # intra_low  (logical -Z)
N2 = np.array([np.sqrt(3) / 2, 0.0, 0.5])    # intra_high (120° from N1)
AXIS_ANGLE_DEG = 120.0


# ---- quaternion algebra (q = [w, x, y, z]) ------------------------------------
def qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def qconj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def qnormalize(q: np.ndarray) -> np.ndarray:
    return np.asarray(q, float) / np.linalg.norm(q)


def from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    return np.array([np.cos(angle / 2), *(np.sin(angle / 2) * axis)])


def to_su2(q: np.ndarray) -> np.ndarray:
    """Unit quaternion -> SU(2) matrix  U = wI - i(xX + yY + zZ)."""
    w, x, y, z = q
    return w * _I - 1j * (x * _X + y * _Y + z * _Z)


def from_su2(U: np.ndarray) -> np.ndarray:
    """SU(2) (or any 2×2 unitary, phase dropped) -> unit quaternion."""
    U = U / np.sqrt(np.linalg.det(U))
    w = np.real(U[0, 0] + U[1, 1]) / 2
    z = np.real(1j * (U[0, 0] - U[1, 1]) / 2)
    x = np.real(1j * (U[0, 1] + U[1, 0]) / 2)
    y = np.real((U[0, 1] - U[1, 0]) / 2)
    return qnormalize(np.array([w, x, y, z]))


def slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    q0 = qnormalize(q0); q1 = qnormalize(q1)
    d = float(np.dot(q0, q1))
    if d < 0:
        q1 = -q1; d = -d
    if d > 0.9995:
        return qnormalize(q0 + t * (q1 - q0))
    th0 = np.arccos(d)
    th = th0 * t
    q2 = qnormalize(q1 - q0 * d)
    return q0 * np.cos(th) + q2 * np.sin(th)


def rotate_bloch(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate a 3-vector by the rotation represented by quaternion q."""
    p = np.array([0.0, *v])
    r = qmul(qmul(q, p), qconj(q))
    return r[1:]


# ---- analytic single-qubit compiler ------------------------------------------
def zxz_angles(U: np.ndarray) -> Tuple[float, float, float]:
    """Euler angles (a, b, c) with U = Rz(a) Rx(b) Rz(c) up to global phase."""
    U = U / np.sqrt(np.linalg.det(U))
    b = 2 * np.arccos(min(1.0, abs(U[0, 0])))
    if abs(np.sin(b / 2)) < 1e-12:                 # b ~ 0: only a+c matters
        a = -2 * np.angle(U[0, 0])
        return float(a), float(b), 0.0
    # arg(U00) = -(a+c)/2 ; arg(U01) = -pi/2 - (a-c)/2
    apc = -2 * np.angle(U[0, 0])
    amc = -2 * (np.angle(U[0, 1]) + np.pi / 2)
    a = (apc + amc) / 2
    c = (apc - amc) / 2
    return float(a), float(b), float(c)


def compile_unitary(U: np.ndarray) -> List[PulseSpec]:
    """Exact exchange-pulse sequence for any single-qubit unitary.

    Uses U = Rz(a) Rx(b) Rz(c) with Rz native (intra_low) and Rx = H Rz H from the
    validated H.  Pulses are returned in application order (first applied first):
    Rz(c), H, Rz(b), H, Rz(a).  Near-trivial Rz factors are dropped.
    """
    a, b, c = zxz_angles(np.asarray(U, complex))
    seq: List[PulseSpec] = []

    def add_rz(theta):
        if abs((theta + np.pi) % (2 * np.pi) - np.pi) > 1e-9:
            seq.append(("intra_low", _rz_area(theta)))

    add_rz(c)
    if abs((b + np.pi) % (2 * np.pi) - np.pi) > 1e-9:
        seq.extend(_H_1Q)
        add_rz(b)
        seq.extend(_H_1Q)
    add_rz(a)
    return seq or [("intra_low", 0.0)]


def compile_quaternion(q: np.ndarray) -> List[PulseSpec]:
    """Exact pulse sequence for a target given as a unit quaternion."""
    return compile_unitary(to_su2(qnormalize(q)))
