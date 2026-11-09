"""Front-face plane of a detected box.

Depth pixels inside the detection box are back-projected to the root frame; RANSAC only accepts plane hypotheses whose
normal is within max_normal_tilt_deg of horizontal and faces the camera (the top face, the table and the floor are
rejected by construction); the face corners are the minimum-area rectangle of the inliers in the plane, with in-plane
axes (horizontal, up).
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from ..common import config
from .geometry import backproject


def fit_front_face(depth, K, T_root_cam, bbox, cfg: dict | None = None) -> dict:
    c = cfg or config.load("perception")["plane_fit"]
    h, w = depth.shape
    x0, y0, x1, y1 = (int(round(v)) for v in bbox)
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    vs, us = np.mgrid[y0:y1:c["stride"], x0:x1:c["stride"]]
    z = depth[vs, us]
    lo, hi = c["depth_valid"]
    m = np.isfinite(z) & (z > lo) & (z < hi)
    if m.sum() < c["min_inliers"]:
        return {"status": "fail", "reason": f"only {int(m.sum())} valid depth pixels in the box (range {lo}-{hi} m)", "n_valid": int(m.sum())}
    pc = backproject(K, us[m], vs[m], z[m])
    P = (T_root_cam[:3, :3] @ pc.T).T + T_root_cam[:3, 3]
    # per-point surface normals from the sampled grid (neighbours one stride right / down): a point of another surface
    # that happens to lie within inlier_dist of the plane (2026-11-06: the floor strip along the base of a floor
    # stack stretched the fitted face from 0.40 m to 1.3 m wide) has a different normal and is not a face inlier
    G = np.full(vs.shape + (3,), np.nan)
    G[m] = P
    if min(vs.shape) < 2:
        return {"status": "fail", "reason": "detection box too small for local normals", "n_valid": int(len(P))}
    gy, gx = np.gradient(G, axis=(0, 1))             # central differences, one-sided at the grid border
    gn = np.cross(gx, gy)
    gn /= np.linalg.norm(gn, axis=2, keepdims=True)
    Pn = gn[m]                                       # NaN next to invalid depth
    cam = T_root_cam[:3, 3]
    rng = np.random.default_rng(c["seed"])
    max_nz = math.sin(math.radians(c["max_normal_tilt_deg"]))
    best_n, best_d, best_cnt = None, None, 0
    for _ in range(c["ransac_iters"]):
        a, b, e = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, e - a)
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n /= nn
        if n @ (cam - a) < 0:
            n = -n                                   # face the camera
        if abs(n[2]) > max_nz:
            continue                                 # not a (near-)vertical face
        cnt = int((np.abs((P - a) @ n) < c["inlier_dist"]).sum())
        if cnt > best_cnt:
            best_n, best_d, best_cnt = n, float(a @ n), cnt
    if best_n is None or best_cnt < c["min_inliers"]:
        return {"status": "fail", "reason": f"no near-vertical plane facing the camera with >= {c['min_inliers']} inliers (best {best_cnt})",
                "n_valid": int(len(P))}
    # inliers: on the plane and with a local normal oriented like it (same max_normal_tilt_deg bound as the plane
    # itself); points without a local normal (next to invalid depth) are not counted
    cos_t = math.cos(math.radians(c["max_normal_tilt_deg"]))
    agree = np.isfinite(Pn[:, 0]) & (np.abs(np.nan_to_num(Pn) @ best_n) >= cos_t)
    inl = P[(np.abs(P @ best_n - best_d) < c["inlier_dist"]) & agree]
    if len(inl) < c["min_inliers"]:
        return {"status": "fail", "reason": f"only {len(inl)} plane inliers with a consistent surface normal", "n_valid": int(len(P))}
    # least-squares refinement on the inliers, keep the orientation
    cen = inl.mean(axis=0)
    _, _, Vt = np.linalg.svd(inl - cen)
    n = Vt[2] if Vt[2] @ best_n > 0 else -Vt[2]
    up = np.array([0.0, 0.0, 1.0]) - n[2] * n; up /= np.linalg.norm(up)
    right = np.cross(up, n)                          # right as seen from the camera side
    uv = np.stack([(inl - cen) @ right, (inl - cen) @ up], axis=1).astype(np.float32)
    (cx, cy), (rw, rh), ang = cv2.minAreaRect(uv)
    box2 = cv2.boxPoints(((cx, cy), (rw, rh), ang))
    corners = np.array([cen + p[0] * right + p[1] * up for p in box2])
    center = cen + cx * right + cy * up
    return {"status": "ok", "center": center, "normal": n, "right": right, "up": up, "corners": corners,
            "size": [float(rw), float(rh)], "n_inliers": int(len(inl)), "n_valid": int(len(P)),
            "inlier_frac": float(len(inl) / len(P)), "rms": float(np.sqrt(np.mean(((inl - cen) @ n) ** 2)))}
