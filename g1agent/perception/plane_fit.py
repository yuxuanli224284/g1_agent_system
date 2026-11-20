"""Front-face plane of a detected box.

Depth pixels inside the detection box are back-projected to the root frame; RANSAC only accepts plane hypotheses whose
normal is within max_normal_tilt_deg of horizontal and faces the camera (the top face, the table and the floor are
rejected by construction); the face corners are the minimum-area rectangle of the inliers in the plane, with in-plane
axes (horizontal, up).

Depth noise (sigma_coeff): stereo depth has a per-pixel error sigma_z = sigma_coeff * z^2 (sigma_coeff = disparity noise /
(focal x baseline), perception.yaml head_depth). Point-to-plane tolerances are max(inlier_dist, inlier_sigmas x sigma_z).
With noisy depth the per-point surface normals of the sampled grid are meaningless (1-2 cm noise over 2-5 mm spacing),
so the floor strip along the base of a floor stack (which lies on the face plane) is removed as the support plane
first: the dominant near-horizontal plane, if it has min_inliers points. With noise-free depth (sigma_coeff 0) the
per-point normal test is kept as before.
Extents with noisy depth: stereo resolves surfaces only at the scale of its matching window (min_structure_px, the
block size), and the surviving noise outliers of the floor along the face's base line also lie on the face plane; one
such point stretched the fitted face from 0.80 to 1.17 m (2026-11-13). The inlier mask on the sampling grid is opened
at the matching-window scale and the face is the largest connected region of it.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from ..common import config
from .geometry import backproject


def fit_support_plane(P, tol, cfg: dict, rng) -> dict:
    """Dominant near-horizontal plane of the points P (N x 3, a frame with z up): RANSAC over cfg ransac_iters draws of
    3 points (from rng), hypotheses whose normal is within max_normal_tilt_deg of vertical, inliers = points within their
    tolerance tol (per point) of the plane. Returns {"mask" (inliers of the best hypothesis, None if none), "count",
    "point", "normal" (unit, z >= 0)}; the caller decides whether count is enough (min_inliers). The same draws as the
    support-plane removal of fit_front_face, which calls it."""
    min_nz = math.cos(math.radians(cfg["max_normal_tilt_deg"]))
    sup_cnt, sup, best = 0, None, None
    for _ in range(cfg["ransac_iters"]):
        a, b, e = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, e - a)
        nn = np.linalg.norm(n)
        if nn < 1e-9 or abs(n[2]) / nn < min_nz:
            continue
        on = np.abs((P - a) @ (n / nn)) < tol
        if on.sum() > sup_cnt:
            sup_cnt, sup, best = int(on.sum()), on, (a, n / nn)
    if best is None:
        return {"mask": None, "count": 0, "point": None, "normal": None}
    a, n = best
    return {"mask": sup, "count": sup_cnt, "point": np.asarray(a, float), "normal": n if n[2] >= 0 else -n}


def fit_front_face(depth, K, T_root_cam, bbox, cfg: dict | None = None, sigma_coeff: float = 0.0,
                   min_structure_px: int = 0) -> dict:
    c = cfg or config.load("perception")["plane_fit"]
    h, w = depth.shape
    x0, y0, x1, y1 = (int(round(v)) for v in bbox)
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        # the box lies outside the image (e.g. a face projected from its known corners while the head looks elsewhere;
        # 2026-11-16 wall6x5: refine_with_wrist at a close stance with the 69 deg D435i view raised a ValueError here)
        return {"status": "fail", "reason": "the box is outside the depth image", "n_valid": 0}
    vs, us = np.mgrid[y0:y1:c["stride"], x0:x1:c["stride"]]
    z = depth[vs, us]
    lo, hi = c["depth_valid"]
    m = np.isfinite(z) & (z > lo) & (z < hi)
    if m.sum() < c["min_inliers"]:
        return {"status": "fail", "reason": f"only {int(m.sum())} valid depth pixels in the box (range {lo}-{hi} m)", "n_valid": int(m.sum())}
    pc = backproject(K, us[m], vs[m], z[m])
    P = (T_root_cam[:3, :3] @ pc.T).T + T_root_cam[:3, 3]
    tol = np.maximum(c["inlier_dist"], c["inlier_sigmas"] * sigma_coeff * z[m] ** 2)   # per-point plane tolerance (m)
    gidx = np.flatnonzero(m.ravel())                 # grid cell of every point
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
    if sigma_coeff > 0:
        # noisy depth: remove the support plane (floor / table top under the box) before the face search
        sp = fit_support_plane(P, tol, c, rng)
        sup, sup_cnt = sp["mask"], sp["count"]
        if sup is not None and sup_cnt >= c["min_inliers"]:
            keep = ~sup
            P, Pn, tol, gidx = P[keep], Pn[keep], tol[keep], gidx[keep]
            if len(P) < c["min_inliers"]:
                return {"status": "fail", "reason": f"only {len(P)} depth points off the support plane", "n_valid": int(len(P))}
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
        cnt = int((np.abs((P - a) @ n) < tol).sum())
        if cnt > best_cnt:
            best_n, best_d, best_cnt = n, float(a @ n), cnt
    if best_n is None or best_cnt < c["min_inliers"]:
        return {"status": "fail", "reason": f"no near-vertical plane facing the camera with >= {c['min_inliers']} inliers (best {best_cnt})",
                "n_valid": int(len(P))}
    # inliers: on the plane and with a local normal oriented like it (same max_normal_tilt_deg bound as the plane
    # itself); points without a local normal (next to invalid depth) are not counted
    cos_t = math.cos(math.radians(c["max_normal_tilt_deg"]))
    if sigma_coeff > 0:
        agree = np.ones(len(P), bool)                # noisy depth: no per-point normals (support plane removed above)
    else:
        agree = np.isfinite(Pn[:, 0]) & (np.abs(np.nan_to_num(Pn) @ best_n) >= cos_t)
    sel = (np.abs(P @ best_n - best_d) < tol) & agree
    if sigma_coeff > 0 and min_structure_px > 0:
        grid = np.zeros(vs.size, np.uint8)
        grid[gidx[sel]] = 1
        grid = grid.reshape(vs.shape)
        k = max(1, int(round(min_structure_px / c["stride"])))
        grid = cv2.morphologyEx(grid, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
        n_lab, lab = cv2.connectedComponents(grid, connectivity=8)
        if n_lab > 1:
            big = 1 + int(np.argmax(np.bincount(lab.ravel())[1:]))
            region = (lab.ravel() == big)
            sel = sel & region[gidx]
    inl = P[sel]
    if len(inl) < c["min_inliers"]:
        return {"status": "fail", "reason": f"only {len(inl)} plane inliers with a consistent surface normal", "n_valid": int(len(P))}
    # least-squares refinement on the inliers, keep the orientation
    cen = inl.mean(axis=0)
    # economy SVD: only Vt is used; the full one also builds the N x N left factor (a whole-image refit, the wall-layout
    # stance re-registration, has ~1e5 inliers: 160 s and tens of GB). Vt is the same (bit-identical in the tests)
    _, _, Vt = np.linalg.svd(inl - cen, full_matrices=False)
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


ON, BEHIND, FRONT = 1, 2, 3          # wall_extent pixel labels


def wall_extent(depth, K, T_root_cam, T_odom_root, plane: dict, cfg: dict | None = None, sigma_coeff: float = 0.0,
                min_structure_px: int = 0, fov_margin_frac: float = 0.0) -> dict:
    """Lateral extent and side edges of a box wall on a known front plane, from one head depth frame (wall layout task,
    Toolbox._wall_extent). plane: {"normal" (odom, unit, horizontal, pointing to the camera), "right" (odom, the robot's
    right along the plane), "point" (odom, on the plane)}; s = (P - point) . right.
    1. Every pixel (full resolution) is back-projected to odom and labelled by its signed distance d to the plane: "on"
       (|d| < tol, the plane-fit tolerance max(inlier_dist, inlier_sigmas x sigma_coeff x z^2)), "behind" (d <= -tol, or
       no valid depth: an occlusion band or a surface out of range) or "front" (d >= tol).
    2. The "on" mask is opened at the matching-window scale (min_structure_px) and its largest 8-connected component is
       the wall (the rules of fit_front_face).
    3. Per image row of the component, its leftmost / rightmost pixel is an EDGE OBSERVATION when (a) it lies strictly
       inside the image margin (fov_margin_frac of the width; otherwise the image cut the run) and (b) walking outward
       from it, the first pixel that is not "on" is "behind" (the depth step behind the wall's side, or the stereo
       occlusion band there), not "front" (something in front of the wall hides its end). No tolerance: containment and
       sign tests only.
    4. Its position: the ray through the pixel boundary (x - 0.5 / x + 0.5, row centre) intersected with the plane: the
       pixel position of the step (~2 mm per px at 1.3 m), not the noisy depth values.
    5. Per side over the edge rows: median s (the floor crease along the base widens a few rows: the median drops them),
       MAD, number of rows and the z range.
    Returns {"left" | "right": {"s", "mad", "n_rows", "z_range"} or None, "seen": [s_min, s_max] over every component
    row (None without a component), "rejected": {"left" | "right": {"cut", "front"}} (rows whose end was not an edge),
    "rows": [[row, s_left, s_right, left is edge, right is edge]] (log only)}. Left / right as seen from the camera side
    (the plane's right axis)."""
    c = cfg or config.load("perception")["plane_fit"]
    H, W = depth.shape
    T_odom_cam = np.asarray(T_odom_root, float) @ np.asarray(T_root_cam, float)
    R, t = T_odom_cam[:3, :3], T_odom_cam[:3, 3]
    n = np.asarray(plane["normal"], float)
    p0 = np.asarray(plane["point"], float)
    right = np.asarray(plane["right"], float)
    z = np.asarray(depth, float)
    lo, hi = c["depth_valid"]
    valid = np.isfinite(z) & (z > lo) & (z < hi)
    zz = np.where(valid, z, 0.0)
    vs, us = np.mgrid[0:H, 0:W]
    P = backproject(K, us, vs, zz) @ R.T + t
    d = (P - p0) @ n
    tol = np.maximum(c["inlier_dist"], c["inlier_sigmas"] * sigma_coeff * zz ** 2)
    lab = np.full((H, W), FRONT, np.uint8)
    lab[d <= -tol] = BEHIND
    lab[np.abs(d) < tol] = ON
    lab[~valid] = BEHIND
    on = (lab == ON).astype(np.uint8)
    k = max(1, int(min_structure_px))
    if k > 1:
        on = cv2.morphologyEx(on, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    out = {"left": None, "right": None, "seen": None, "rejected": {"left": {"cut": 0, "front": 0},
                                                                    "right": {"cut": 0, "front": 0}}, "rows": []}
    n_lab, comp = cv2.connectedComponents(on, connectivity=8)
    if n_lab < 2:
        return out
    big = 1 + int(np.argmax(np.bincount(comp.ravel())[1:]))
    comp = comp == big
    # which image direction is the plane's right: the projection of a step along it
    Kf = np.asarray(K, float)
    T_cam_odom = np.linalg.inv(T_odom_cam)
    cen = P[comp].mean(axis=0)
    qc = T_cam_odom[:3, :3] @ cen + T_cam_odom[:3, 3]
    rc = T_cam_odom[:3, :3] @ right
    x_is_right = rc[0] * qc[2] - qc[0] * rc[2] > 0        # sign of du/ds at the component's centre
    mx = fov_margin_frac * W
    cam = t

    def s_at(px, py):
        dr = R @ np.array([(px - Kf[0, 2]) / Kf[0, 0], (py - Kf[1, 2]) / Kf[1, 1], 1.0])
        den = float(dr @ n)
        if abs(den) < 1e-9:
            return None, None
        tt = float((p0 - cam) @ n) / den
        if tt <= 0:
            return None, None
        Q = cam + tt * dr
        return float((Q - p0) @ right), float(Q[2])

    samples = {"left": [], "right": []}
    s_all = []
    for y in np.flatnonzero(comp.any(axis=1)):
        xs = np.flatnonzero(comp[y])
        ends = {}
        for end, x, step in (("lo", int(xs[0]), -1), ("hi", int(xs[-1]), 1)):
            side = ("left" if end == "lo" else "right") if x_is_right else ("right" if end == "lo" else "left")
            s, zq = s_at(x + 0.5 * step, float(y))
            inside = x > mx if end == "lo" else x < W - 1 - mx
            beyond = lab[y, x + step::step] if end == "hi" else lab[y, x - 1::-1] if x > 0 else lab[y, :0]
            first = beyond[beyond != ON]
            if not inside or first.size == 0:
                kind = "cut"
            elif first[0] == FRONT:
                kind = "front"
            else:
                kind = "edge"
            if s is not None:
                s_all.append(s)
                if kind == "edge":
                    samples[side].append((s, zq))
            if kind != "edge":
                out["rejected"][side][kind] += 1
            ends[side] = (s, kind == "edge")
        out["rows"].append([int(y), ends.get("left", (None,))[0], ends.get("right", (None,))[0],
                            ends.get("left", (None, False))[1], ends.get("right", (None, False))[1]])
    if s_all:
        out["seen"] = [float(min(s_all)), float(max(s_all))]
    for side in ("left", "right"):
        if samples[side]:
            S = np.array([v[0] for v in samples[side]])
            Z = np.array([v[1] for v in samples[side]])
            med = float(np.median(S))
            out[side] = {"s": med, "mad": float(np.median(np.abs(S - med))), "n_rows": int(len(S)),
                         "z_range": [float(Z.min()), float(Z.max())]}
    return out

