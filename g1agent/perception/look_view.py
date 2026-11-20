"""Hand-camera look pose for a box face of a given wall layout (Toolbox.compute_wrist_target of an entry of kind "cell",
Toolbox.search_cell): where to put the scanner tip so that the hand camera (g1_v2: the D405 on the bracket) sees the
WHOLE box face, and the exact test whether a face is in its free view.

Free view of the hand camera: its image shrunk by the in-view margin of every in-view test (perception.yaml
fov_margin_frac) AND the pixels whose ray does not hit an occluder (the robot's own parts in front of the camera: the
scanner head and handle on g1_v2) at positive depth (exact ray / box slab test per pixel, once: the camera is rigid on
the scanner bracket). A face is covered when its 4 corners are in front of the camera inside the margin rectangle and
no occluded pixel centre inside that rectangle lies inside its projected (convex) quadrilateral. Only occluded pixels
next to a non-occluded pixel of the rectangle need to be tested: a convex quadrilateral whose corners are free and that
holds an occluded pixel crosses the border of the occluded region (exact up to the pixel).

Look pose (search_view): the face lies in the plane frame (x = right, y = up, z = the plane normal toward the robot,
the face centre at the origin). The scanner tip looks along -z; its image x axis (the IK roll reference, ArmIK
img_x_in_tip) is the face's right axis rotated by a quarter roll k about the normal: L_k = cos k right + sin k up (k =
0, 90, 180, 270 deg: counter-clockwise as seen from the robot). For each stand-off s from s_min upward in steps and
every tip offset (h, v) in the plane on the step grid inside the camera's footprint, the face is tested as above; the
first s at which some roll covers it wins, every roll that covers it there is a candidate, and per roll the chosen
(h, v) is the feasible grid point nearest the centroid of its feasible set (the largest margin, no number)."""
from __future__ import annotations

import math

import numpy as np


def roll_rotation(k_deg: float, img_x_in_tip) -> np.ndarray:
    """Rotation (plane frame) of the scanner-tip frame looking along -normal with its image x axis along L_k (the
    ArmIK.roll_target construction with the sign kept)."""
    k = math.radians(k_deg)
    d = np.array([0.0, 0.0, -1.0])
    L = np.array([math.cos(k), math.sin(k), 0.0])
    c = np.asarray(img_x_in_tip, float)
    c = c / np.linalg.norm(c)
    ex = np.array([1.0, 0.0, 0.0])
    if abs(float(c @ ex)) > 1e-9:
        raise ValueError("the scanner image x axis must be orthogonal to the scan axis")
    return np.column_stack([d, L, np.cross(d, L)]) @ np.column_stack([ex, c, np.cross(ex, c)]).T


def roll_axis(k_deg: float, right, up) -> np.ndarray:
    """L_k in the frame of the given right / up axes (odom)."""
    k = math.radians(k_deg)
    return math.cos(k) * np.asarray(right, float) + math.sin(k) * np.asarray(up, float)


def _slab_hits(o, D, lo, hi) -> np.ndarray:
    """Rays o + t D (D: N x 3) that hit the box [lo, hi] at some t > 0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        Ds = np.where(np.abs(D) < 1e-12, 1e-12, D)
        t1 = (lo - o) / Ds
        t2 = (hi - o) / Ds
    tn = np.minimum(t1, t2).max(axis=1)
    tf = np.maximum(t1, t2).min(axis=1)
    return (tf >= tn) & (tf > 0)


class HandView:
    """Free view of a hand camera. K: 3x3; size: [height, width]; T_tip_cam: 4x4 camera optical frame in the scanner-
    tip frame; img_x_in_tip: the scanner image x axis in the tip frame; occluders: [{"name", "center", "size",
    "T_cam_box"}] boxes (axis-aligned in their link frame; T_cam_box maps link coordinates into the camera frame);
    margin_frac: the in-view margin. Boxes that contain the camera's optical centre (the bracket, the camera's own
    housing) are not occluders: the camera looks out of them."""

    def __init__(self, K, size, T_tip_cam, img_x_in_tip, occluders, margin_frac: float):
        self.K = np.asarray(K, float)
        self.H, self.W = int(size[0]), int(size[1])
        self.T_tip_cam = np.asarray(T_tip_cam, float)
        self.img_x_in_tip = np.asarray(img_x_in_tip, float)
        self.margin_frac = float(margin_frac)
        vs, us = np.mgrid[0:self.H, 0:self.W]
        D = np.stack([(us - self.K[0, 2]) / self.K[0, 0], (vs - self.K[1, 2]) / self.K[1, 1], np.ones(us.shape)], -1)
        D = D.reshape(-1, 3)
        occ = np.zeros(len(D), bool)
        self.used, self.dropped = [], []
        for b in occluders:
            T_cb = np.asarray(b["T_cam_box"], float)
            T_bc = np.linalg.inv(T_cb)                      # camera -> box (link) frame
            o = T_bc[:3, 3]
            lo = np.asarray(b["center"], float) - np.asarray(b["size"], float) / 2
            hi = np.asarray(b["center"], float) + np.asarray(b["size"], float) / 2
            if np.all(o >= lo) and np.all(o <= hi):
                self.dropped.append(b["name"])
                continue
            self.used.append(b["name"])
            occ |= _slab_hits(o, D @ T_bc[:3, :3].T, lo, hi)
        self.occluded = occ.reshape(self.H, self.W)
        mx, my = self.margin_frac * self.W, self.margin_frac * self.H
        rect = (us >= mx) & (us <= self.W - mx) & (vs >= my) & (vs <= self.H - my)
        self.free = rect & ~self.occluded
        occ_in = rect & self.occluded
        import cv2
        near_free = cv2.dilate(self.free.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        yy, xx = np.nonzero(occ_in & near_free)
        self.border_px = np.c_[xx, yy].astype(float)       # occluded pixel centres next to a free pixel

    def project(self, P_cam):
        """Camera-frame points (... x 3) -> (uv ... x 2, z ...)."""
        P = np.asarray(P_cam, float)
        z = P[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = P[..., :2] / z[..., None] * [self.K[0, 0], self.K[1, 1]] + [self.K[0, 2], self.K[1, 2]]
        return uv, z

    def covered(self, uv, z) -> np.ndarray:
        """uv: N x 4 x 2 projected quadrilateral corners (in order around the quad), z: N x 4 depths -> N booleans: the
        quadrilateral lies in the free view (corners in front, inside the margin rectangle; no occluded pixel inside)."""
        uv = np.asarray(uv, float).reshape(-1, 4, 2)
        z = np.asarray(z, float).reshape(-1, 4)
        mx, my = self.margin_frac * self.W, self.margin_frac * self.H
        with np.errstate(invalid="ignore"):
            ok = np.all(z > 0, axis=1) & np.all((uv[..., 0] >= mx) & (uv[..., 0] <= self.W - mx)
                                                & (uv[..., 1] >= my) & (uv[..., 1] <= self.H - my), axis=1)
        idx = np.flatnonzero(ok)
        B = self.border_px
        if len(B) == 0 or len(idx) == 0:
            return ok
        for i0 in range(0, len(idx), 64):                  # chunks: N x 4 x M cross products
            ii = idx[i0:i0 + 64]
            Q = uv[ii]                                      # n x 4 x 2
            E = np.roll(Q, -1, axis=1) - Q
            W_ = B[None, None, :, :] - Q[:, :, None, :]     # n x 4 x M x 2
            cr = E[:, :, None, 0] * W_[..., 1] - E[:, :, None, 1] * W_[..., 0]
            inside = np.all(cr >= 0, axis=1) | np.all(cr <= 0, axis=1)   # n x M
            ok[ii] = ~inside.any(axis=1)
        return ok

    def whole(self, uv, z, px: float = 1.0) -> bool:
        """A detected quadrilateral (4 x 2 corners around it, depths z) is seen whole: the quad grown by px pixels on
        every side (the pixel quantisation of a detection box: the 1 px of the crop-border cut test) is covered and its
        centre pixel is free (covered tests the occluded border pixels only, so a quad lying wholly inside the occluded
        region would pass it). A detection whose box ends within px of the image border, the in-view margin or an
        occluded pixel shows its visible part only."""
        g = grow_quad(uv, px)
        if not self.covered(g[None], np.asarray(z, float).reshape(1, 4))[0]:
            return False
        u, v = np.round(g.mean(axis=0)).astype(int)
        return bool(self.free[v, u])


def grow_quad(uv, px: float) -> np.ndarray:
    """A convex quadrilateral (4 x 2, corners in order around it, either orientation) with every edge moved outward by
    px (its new corners: the intersections of adjacent moved edges)."""
    Q = np.asarray(uv, float).reshape(4, 2)
    c = Q.mean(axis=0)
    lines = []
    for i in range(4):
        p, d = Q[i], Q[(i + 1) % 4] - Q[i]
        n = np.array([d[1], -d[0]]) / np.linalg.norm(d)
        if n @ (c - p) > 0:                                # outward: away from the centroid
            n = -n
        lines.append((n, float(n @ p) + px))               # n . x = n . p + px
    return np.asarray([np.linalg.solve(np.array([lines[i - 1][0], lines[i][0]]), [lines[i - 1][1], lines[i][1]])
                       for i in range(4)])


def face_corners(half_w: float, half_h: float) -> np.ndarray:
    """Corners (plane frame, around the rectangle) of a face centred at the origin."""
    return np.array([[-half_w, half_h, 0.0], [half_w, half_h, 0.0], [half_w, -half_h, 0.0], [-half_w, -half_h, 0.0]])


def search_view(hv: HandView, face_w: float, face_h: float, delta: float, e: float, step: float, s_min: float,
                s_max: float, rolls=(0.0, 90.0, 180.0, 270.0)) -> dict:
    """Minimum tip stand-off s (from s_min in steps of `step`) at which the face (face_w x face_h, grown by delta
    laterally and by e on every side) is covered by the hand camera's free view for some quarter roll; see the module
    docstring. Returns {"s", "candidates": [{"roll_deg", "h", "v", "n_feasible"}], "grown": [w, h]}. Raises when no
    stand-off up to s_max covers the face (a geometry error, not a task outcome)."""
    if step <= 0:
        raise ValueError(f"search step must be > 0, got {step}")
    a, b = face_w / 2 + delta + e, face_h / 2 + e
    X = face_corners(a, b)
    Rts = {k: roll_rotation(k, hv.img_x_in_tip) for k in rolls}
    n_s = int(math.floor((s_max - s_min) / step + 1e-9))
    for i in range(n_s + 1):
        s = round(s_min + i * step, 9)
        cands = []
        for k in rolls:
            T_pt = np.eye(4)
            T_pt[:3, :3], T_pt[:3, 3] = Rts[k], [0.0, 0.0, s]
            T_pc = T_pt @ hv.T_tip_cam                         # camera in the plane frame, tip at (0, 0, s)
            Rc, oc = T_pc[:3, :3], T_pc[:3, 3]
            # footprint of the image on the plane: the corner rays from the camera (tip offset 0)
            F = []
            for u, v in ((0, 0), (hv.W, 0), (hv.W, hv.H), (0, hv.H)):
                d = Rc @ np.array([(u - hv.K[0, 2]) / hv.K[0, 0], (v - hv.K[1, 2]) / hv.K[1, 1], 1.0])
                if d[2] >= 0:
                    raise ValueError(f"hand-camera image corner ray ({u}, {v}) does not reach the face plane (roll {k})")
                F.append(oc[:2] + (-oc[2] / d[2]) * d[:2])
            F = np.asarray(F)
            # the face centre (plane origin) must lie in the footprint shifted by the tip offset (h, v)
            hs = step * np.arange(math.ceil(-F[:, 0].max() / step), math.floor(-F[:, 0].min() / step) + 1)
            vs = step * np.arange(math.ceil(-F[:, 1].max() / step), math.floor(-F[:, 1].min() / step) + 1)
            if len(hs) == 0 or len(vs) == 0:
                continue
            HH, VV = np.meshgrid(hs, vs, indexing="ij")
            sh = np.c_[HH.ravel(), VV.ravel(), np.zeros(HH.size)]
            # corners in the camera frame for every offset: Rc^T (X - oc - shift)
            Pc = ((X[None, :, :] - oc - sh[:, None, :]) @ Rc)
            uv, z = hv.project(Pc)
            ok = hv.covered(uv, z)
            if not ok.any():
                continue
            fe = sh[ok, :2]
            m = fe.mean(axis=0)
            j = int(np.lexsort((np.abs(fe).sum(axis=1), np.linalg.norm(fe - m, axis=1)))[0])
            cands.append({"roll_deg": float(k), "h": float(fe[j, 0]), "v": float(fe[j, 1]), "n_feasible": int(ok.sum())})
        if cands:
            return {"s": s, "candidates": cands, "grown": [2 * a, 2 * b]}
    raise RuntimeError(f"no stand-off in [{s_min}, {s_max}] m puts the {2 * a:.3f} x {2 * b:.3f} m face in the hand "
                       "camera's free view")
