"""SE(3) helpers. Convention: T_a_b = pose of frame b in frame a (maps b-coordinates to a-coordinates)."""
from __future__ import annotations

import math

import numpy as np


def inv_T(T):
    R, t = T[:3, :3], T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def T_from_xy_yaw(x, y, yaw, z=0.0):
    c, s = math.cos(yaw), math.sin(yaw)
    T = np.eye(4)
    T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    T[:3, 3] = [x, y, z]
    return T


def yaw_of(T) -> float:
    return math.atan2(T[1, 0], T[0, 0])


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def apply(T, p):
    p = np.asarray(p, float)
    return (T[:3, :3] @ p.T).T + T[:3, 3] if p.ndim == 2 else T[:3, :3] @ p + T[:3, 3]


def rotate(T, v):
    v = np.asarray(v, float)
    return (T[:3, :3] @ v.T).T if v.ndim == 2 else T[:3, :3] @ v


def angle_between(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(math.atan2(np.linalg.norm(np.cross(a, b)), a @ b))
