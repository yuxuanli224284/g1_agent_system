"""Camera geometry: back-projection, projection, ray-plane intersection, two-view triangulation.
Camera frames are ROS optical (x right, y down, z forward); K is 3x3; T_a_b maps b-coordinates to a."""
from __future__ import annotations

import numpy as np


def backproject(K, u, v, z):
    """Pixel(s) + depth along the optical axis -> camera-frame points (N x 3)."""
    u, v, z = (np.asarray(a, float) for a in (u, v, z))
    x = (u - K[0, 2]) / K[0, 0] * z
    y = (v - K[1, 2]) / K[1, 1] * z
    return np.stack([x, y, z], axis=-1)


def pixel_ray(K, u, v):
    """Unit ray direction (camera frame) through pixel (u, v)."""
    d = np.array([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], 1.0])
    return d / np.linalg.norm(d)


def project(K, T_cam_x, pts):
    """Points in frame x -> (uv N x 2, depth N). T_cam_x maps x-coordinates into the camera frame."""
    P = np.asarray(pts, float).reshape(-1, 3)
    pc = (T_cam_x[:3, :3] @ P.T).T + T_cam_x[:3, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = pc[:, :2] / pc[:, 2:3] * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
    return uv, pc[:, 2]


def in_image(uv, z, w, h, margin_frac=0.0):
    mx, my = margin_frac * w, margin_frac * h
    return (z > 0) & (uv[:, 0] >= mx) & (uv[:, 0] <= w - mx) & (uv[:, 1] >= my) & (uv[:, 1] <= h - my)


def ray_plane(origin, direction, plane_point, plane_normal, min_cos=1e-3):
    """Intersection of origin + t * direction (t > 0) with the plane; None if (nearly) parallel or behind."""
    d, n = np.asarray(direction, float), np.asarray(plane_normal, float)
    den = float(d @ n)
    if abs(den) < min_cos:
        return None
    t = float((np.asarray(plane_point, float) - np.asarray(origin, float)) @ n) / den
    return None if t <= 0 else np.asarray(origin, float) + t * d


def triangulate(o1, d1, o2, d2):
    """Midpoint of the shortest segment between two rays; returns (point, gap between the rays)."""
    o1, d1, o2, d2 = (np.asarray(a, float) for a in (o1, d1, o2, d2))
    w0 = o1 - o2
    a, b, c, d, e = d1 @ d1, d1 @ d2, d2 @ d2, d1 @ w0, d2 @ w0
    den = a * c - b * b
    if abs(den) < 1e-12:
        return None, float("inf")
    s, t = (b * e - c * d) / den, (a * e - b * d) / den
    p1, p2 = o1 + s * d1, o2 + t * d2
    return (p1 + p2) / 2, float(np.linalg.norm(p1 - p2))


def horizontal_distance_to_plane(p, plane_point, plane_normal):
    """distance from p to the plane measured horizontally along the plane's horizontal normal."""
    n = np.asarray(plane_normal, float).copy(); n[2] = 0.0
    nn = np.linalg.norm(n)
    if nn < 1e-9:
        raise ValueError("plane is horizontal")
    n /= nn
    return float((np.asarray(p, float) - np.asarray(plane_point, float))[:2] @ n[:2])
