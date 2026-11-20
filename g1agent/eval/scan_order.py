"""Task-3 scan order and scanner stand-off on the evaluation side (pure functions, no ground truth access here; the
evaluator feeds them GT labels). Implements the rules of the design once, independently of the toolbox, so the
evaluator never imports algorithm code:

- Order "left to right, then top to bottom" (rule "columns", the default): two labels share a column iff their lateral
  extents (y in the reference root frame, left = +y) overlap; columns are the connected groups of overlapping extents.
  Columns go left to right, inside a column labels go top to bottom (descending z).
- Order rule "rows" (a property of the task's target spec, scenario scan_targets.order): rows are the connected groups
  of overlapping vertical extents (z); rows go top to bottom, inside a row labels go left to right.
- Adaptive stand-off: d = label_width / (2 tan(hfov / 2)) / fill, clamped to [d_min, d_max].
"""
from __future__ import annotations

import math

import numpy as np


def adaptive_standoff(label_width_m: float, hfov_rad: float, fill: float, d_min: float, d_max: float) -> float:
    """Distance at which a label of width label_width_m spans `fill` of the scanner image width, clamped."""
    d = label_width_m / (2.0 * math.tan(hfov_rad / 2.0)) / fill
    return float(min(max(d, d_min), d_max))


def hfov_from_intrinsics(fx: float, width: float) -> float:
    return 2.0 * math.atan(width / 2.0 / fx)


ORDER_RULES = ("columns", "rows")


def _overlap_groups(items: list[dict], lo: str, hi: str) -> list[list[dict]]:
    """Connected components of the 'extents [lo, hi] overlap' relation (strict overlap; touching extents are separate
    groups), from the highest extent down. Sweeping the items by descending hi, an item joins the current group iff its
    hi lies above the group's lowest lo; no later item can bridge back to an earlier group because later items end
    lower."""
    srt = sorted(items, key=lambda it: (-it[hi], -it[lo], str(it["key"])))
    groups, low = [], None
    for it in srt:
        if groups and it[hi] > low:
            groups[-1].append(it)
            low = min(low, it[lo])
        else:
            groups.append([it])
            low = it[lo]
    return groups


def column_order(items: list[dict]) -> list:
    """items: [{"key", "y_lo", "y_hi", "z"}] in one reference frame (left = +y). Returns the keys in scan order.

    Columns = connected groups of overlapping lateral extents, left to right. Ties inside a column (equal z) are broken
    by descending lateral centre, then by key, so the order is deterministic."""
    out = []
    for col in _overlap_groups(items, "y_lo", "y_hi"):         # already left to right
        out += [it["key"] for it in sorted(col, key=lambda it: (-it["z"], -(it["y_lo"] + it["y_hi"]) / 2, str(it["key"])))]
    return out


def row_order(items: list[dict]) -> list:
    """items: [{"key", "y_lo", "y_hi", "z_lo", "z_hi", "z"}]. Rows = connected groups of overlapping vertical extents,
    top to bottom; inside a row left to right (descending lateral centre), ties by descending z, then key."""
    out = []
    for row in _overlap_groups(items, "z_lo", "z_hi"):         # already top to bottom
        out += [it["key"] for it in sorted(row, key=lambda it: (-(it["y_lo"] + it["y_hi"]) / 2, -it["z"], str(it["key"])))]
    return out


def order_items(items: list[dict], rule: str = "columns") -> list:
    if rule == "columns":
        return column_order(items)
    if rule == "rows":
        return row_order(items)
    raise ValueError(f"unknown order rule {rule!r} (known: {ORDER_RULES})")


def label_corners(center, normal, up, size) -> np.ndarray:
    """4x3 corners of a planar label (center, outward normal, up, size [width, height]) in the same frame."""
    c, n, u = (np.asarray(v, float) for v in (center, normal, up))
    r = np.cross(u, n)
    r /= np.linalg.norm(r)
    w, h = float(size[0]) / 2.0, float(size[1]) / 2.0
    return np.array([c + sx * w * r + sy * h * u for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])


def order_labels_in_frame(labels: list[dict], T_world_frame, rule: str = "columns") -> list:
    """labels: [{"key", "center", "normal", "up", "size"}] in world; T_world_frame: 4x4 reference (root) frame.
    Returns the keys in scan order (order rule `rule`)."""
    T_fw = np.linalg.inv(np.asarray(T_world_frame, float))
    items = []
    for lb in labels:
        cw = label_corners(lb["center"], lb["normal"], lb["up"], lb["size"])
        cf = (T_fw[:3, :3] @ cw.T).T + T_fw[:3, 3]
        zc = float((T_fw[:3, :3] @ np.asarray(lb["center"], float) + T_fw[:3, 3])[2])
        items.append({"key": lb["key"], "y_lo": float(cf[:, 1].min()), "y_hi": float(cf[:, 1].max()),
                      "z_lo": float(cf[:, 2].min()), "z_hi": float(cf[:, 2].max()), "z": zc})
    return order_items(items, rule)


# ---------------------------------------------------------------------------------------------- multi-face stacks
# Four-face tour of a free-standing stack (task 3 target "all", shared interfaces I2 / I3): front (-x, facing the robot
# start) -> right (-y) -> back (+x) -> left (+y), counter-clockwise seen from above, continuing each face's own
# left-to-right column order. Index i = the I3 "face" index of face_start / face_end and the I4 scan_plan face_id.
FACE_TOUR = ("-x", "-y", "+x", "+y")
FACE_NAMES = ("front", "right", "back", "left")


def face_frame(normal, origin) -> np.ndarray:
    """4x4 ideal frame of a face: forward = -n (horizontal), left = n x z, up = z, origin given (the face's GT obs
    pose). The column rule of order_labels_in_frame in this frame is the face's own left-to-right, top-to-bottom."""
    n = np.array([normal[0], normal[1], 0.0], float)
    n /= np.linalg.norm(n)
    z = np.array([0.0, 0.0, 1.0])
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = -n, np.cross(n, z), z, np.asarray(origin, float)
    return T


def assign_faces(label_normals, face_normals) -> list[int]:
    """Index of the face whose outward normal best matches each label normal (argmax of the dot product)."""
    F = np.asarray(face_normals, float).reshape(-1, 3)
    return [int(np.argmax(F @ np.asarray(n, float))) for n in label_normals]


def labelled_faces(labels: list[dict], faces: list[dict], rule: str = "columns"):
    """labels: [{"key", "center", "normal", "up", "size"}] (world); faces: GT "faces" entries (I2) {"face", "normal",
    "gt_obs_pose"}. Returns [(face label, [keys in that face's scan order])] for the faces carrying at least one label,
    in tour order (FACE_TOUR). Each face is ordered in its own ideal frame (face_frame at its gt_obs_pose)."""
    for f in faces:
        if f["face"] not in FACE_TOUR:
            raise ValueError(f"unknown face {f['face']!r} (tour {FACE_TOUR})")
    faces = sorted(faces, key=lambda f: FACE_TOUR.index(f["face"]))
    idx = assign_faces([lb["normal"] for lb in labels], [f["normal"] for f in faces]) if labels else []
    out = []
    for i, f in enumerate(faces):
        mine = [lb for lb, j in zip(labels, idx) if j == i]
        if mine:
            out.append((f["face"], order_labels_in_frame(mine, face_frame(f["normal"], f["gt_obs_pose"]["pos"]), rule)))
    return out
