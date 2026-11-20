"""Task-3 scan order and scanner stand-off on the evaluation side (pure functions, no ground truth access here; the
evaluator feeds them GT labels). Implements the rules of the design once, independently of the toolbox, so the
evaluator never imports algorithm code:

- Order "left to right, then top to bottom" (rule "columns", the default): two labels share a column iff their lateral
  extents (y in the reference root frame, left = +y) overlap; columns are the connected groups of overlapping extents.
  Columns go left to right, inside a column labels go top to bottom (descending z).
- Order rule "rows" (a property of the task's target spec, scenario scan_targets.order): rows are the connected groups
  of overlapping vertical extents (z); rows go top to bottom, inside a row labels go left to right.
- Order rule "serpentine" (scenario scan_targets.order, user decision 2026-11-19): the rows of rule "rows", top to
  bottom; the top row (of the rows that hold expected labels) left to right, the next right to left, alternating.
- Order unit "box" (scenarios with a wall layout, order_boxes_in_frame): the rule is applied to the GT boxes of the
  labels (their labelled faces), grouped into rows / columns by centre containment; labels of one box by the label
  rule (serpentine: in the direction of the box's row).
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


ORDER_RULES = ("columns", "rows", "serpentine")
ORDER_UNITS = ("label", "box")


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


def _in_row(row: list[dict], left_to_right: bool) -> list:
    """Keys of one row: by lateral centre in the given direction (left = +y), ties by descending z, then key."""
    s = -1.0 if left_to_right else 1.0
    return [it["key"] for it in sorted(row, key=lambda it: (s * (it["y_lo"] + it["y_hi"]) / 2, -it["z"], str(it["key"])))]


def row_order(items: list[dict], left_to_right: bool = True) -> list:
    """items: [{"key", "y_lo", "y_hi", "z_lo", "z_hi", "z"}]. Rows = connected groups of overlapping vertical extents,
    top to bottom; inside a row left to right (descending lateral centre; right to left with left_to_right=False), ties
    by descending z, then key."""
    out = []
    for row in _overlap_groups(items, "z_lo", "z_hi"):         # already top to bottom
        out += _in_row(row, left_to_right)
    return out


def serpentine_order(items: list[dict]) -> list:
    """items as row_order. The rows of row_order, top to bottom; the k-th row (k = 0 at the top, counted over the rows
    that hold items) left to right when k is even, right to left when k is odd."""
    out = []
    for k, row in enumerate(_overlap_groups(items, "z_lo", "z_hi")):
        out += _in_row(row, k % 2 == 0)
    return out


def order_items(items: list[dict], rule: str = "columns") -> list:
    if rule == "columns":
        return column_order(items)
    if rule == "rows":
        return row_order(items)
    if rule == "serpentine":
        return serpentine_order(items)
    raise ValueError(f"unknown order rule {rule!r} (known: {ORDER_RULES})")


def label_corners(center, normal, up, size) -> np.ndarray:
    """4x3 corners of a planar label (center, outward normal, up, size [width, height]) in the same frame."""
    c, n, u = (np.asarray(v, float) for v in (center, normal, up))
    r = np.cross(u, n)
    r /= np.linalg.norm(r)
    w, h = float(size[0]) / 2.0, float(size[1]) / 2.0
    return np.array([c + sx * w * r + sy * h * u for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])


def _frame_items(labels: list[dict], T_world_frame) -> list[dict]:
    """Order items ({"key", "y_lo", "y_hi", "z_lo", "z_hi", "z"}) of GT labels in a reference frame."""
    T_fw = np.linalg.inv(np.asarray(T_world_frame, float))
    items = []
    for lb in labels:
        cw = label_corners(lb["center"], lb["normal"], lb["up"], lb["size"])
        cf = (T_fw[:3, :3] @ cw.T).T + T_fw[:3, 3]
        zc = float((T_fw[:3, :3] @ np.asarray(lb["center"], float) + T_fw[:3, 3])[2])
        items.append({"key": lb["key"], "y_lo": float(cf[:, 1].min()), "y_hi": float(cf[:, 1].max()),
                      "z_lo": float(cf[:, 2].min()), "z_hi": float(cf[:, 2].max()), "z": zc})
    return items


def order_labels_in_frame(labels: list[dict], T_world_frame, rule: str = "columns") -> list:
    """labels: [{"key", "center", "normal", "up", "size"}] in world; T_world_frame: 4x4 reference (root) frame.
    Returns the keys in scan order (order rule `rule`)."""
    return order_items(_frame_items(labels, T_world_frame), rule)


# ---------------------------------------------------------------------------------------------- order unit "box"
def yaw_only(T) -> np.ndarray:
    """The 4x4 frame with T's origin and heading (yaw of its x axis) and a vertical z axis."""
    T = np.asarray(T, float)
    yaw = math.atan2(T[1, 0], T[0, 0])
    c, s = math.cos(yaw), math.sin(yaw)
    out = np.eye(4)
    out[:3, :3] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    out[:3, 3] = T[:3, 3]
    return out


def box_face(box: dict, normal) -> dict:
    """The side face of a GT box (I2 "boxes" entry {"pos", "size": [d (x), w (y), h (z)], "quat_wxyz" yaw-only})
    whose outward normal best matches `normal` (argmax of the dot product over the four side faces): {"center",
    "normal", "corners" (4x3)}, world frame."""
    w, x, y, z = box["quat_wxyz"]
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    h = np.asarray(box["size"], float) / 2
    sides = [(a, sg) for a in (0, 1) for sg in (-1.0, 1.0)]
    a, sg = sides[int(np.argmax([sg * (R[:, a] @ np.asarray(normal, float)) for a, sg in sides]))]
    b = 1 - a                                                # the other horizontal axis
    pos = np.asarray(box["pos"], float)
    centre = pos + R[:, a] * sg * h[a]
    corners = np.array([centre + R[:, b] * sb * h[b] + R[:, 2] * sz * h[2] for sb, sz in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
    return {"center": centre, "normal": R[:, a] * sg, "corners": corners}


def _containment_groups(items: list[dict], lo: str, hi: str, c: str) -> list[list[dict]]:
    """Connected groups of the relation 'either item's centre c lies strictly inside the other's extent (lo, hi)',
    from the highest mean centre down. For boxes of one size this equals strict extent overlap, but it does not depend
    on whether two touching extents overlap by float noise after a frame transform."""
    parent = list(range(len(items)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i, p in enumerate(items):
        for j in range(i + 1, len(items)):
            q = items[j]
            if p[lo] < q[c] < p[hi] or q[lo] < p[c] < q[hi]:
                parent[find(i)] = find(j)
    groups = {}
    for i, it in enumerate(items):
        groups.setdefault(find(i), []).append(it)
    return sorted(groups.values(), key=lambda g: (-float(np.mean([it[c] for it in g])), min(str(it["key"]) for it in g)))


def order_boxes_in_frame(labels: list[dict], boxes: dict, T_world_frame, rule: str = "columns") -> list:
    """Order unit "box": labels [{"key", "center", "normal", "up", "size", "box_id"}] (world), boxes {id: GT box},
    T_world_frame: 4x4 reference frame (the evaluator passes a yaw-only frame). Each label's order item is the face of
    its box that carries it (box_face with the label normal), in the frame. Boxes are grouped into rows (centre z
    strictly inside the other's z extent, connected groups, top to bottom) and columns (the same on y, left = +y, left
    to right). Rule "rows": rows from the top, columns from the left inside a row; "columns": columns from the left,
    rows from the top inside a column; "serpentine": rows from the top, the k-th row (k = 0 at the top, over the rows
    holding labels) left to right when k is even, right to left when k is odd. Labels of one box: the label rule in the
    same frame (serpentine: row_order in the direction of the box's row). Returns the keys in scan order."""
    if rule not in ORDER_RULES:
        raise ValueError(f"unknown order rule {rule!r} (known: {ORDER_RULES})")
    T_fw = np.linalg.inv(np.asarray(T_world_frame, float))
    faces = {}
    for lb in labels:
        f = box_face(boxes[lb["box_id"]], lb["normal"])
        if lb["box_id"] in faces and not np.allclose(faces[lb["box_id"]]["normal"], f["normal"]):
            raise ValueError(f"box {lb['box_id']} carries labels on two faces: order unit 'box' is defined on one face")
        faces[lb["box_id"]] = f
    items = []
    for bid, f in faces.items():
        cf = (T_fw[:3, :3] @ f["corners"].T).T + T_fw[:3, 3]
        cc = T_fw[:3, :3] @ f["center"] + T_fw[:3, 3]
        items.append({"key": bid, "y_lo": float(cf[:, 1].min()), "y_hi": float(cf[:, 1].max()), "y": float(cc[1]),
                      "z_lo": float(cf[:, 2].min()), "z_hi": float(cf[:, 2].max()), "z": float(cc[2])})
    row = {it["key"]: k for k, g in enumerate(_containment_groups(items, "z_lo", "z_hi", "z")) for it in g}
    col = {it["key"]: k for k, g in enumerate(_containment_groups(items, "y_lo", "y_hi", "y")) for it in g}
    rows = sorted(set(row.values()))
    l2r = {b: rule != "serpentine" or rows.index(row[b]) % 2 == 0 for b in row}
    yc = {it["key"]: it["y"] for it in items}

    def key(b):
        c = col[b] if l2r[b] else -col[b]
        return ((col[b], row[b]) if rule == "columns" else (row[b], c)), (-yc[b] if l2r[b] else yc[b]), str(b)
    litems = {it["key"]: it for it in _frame_items(labels, T_world_frame)}
    out = []
    for b in sorted(faces, key=key):
        mine = [litems[lb["key"]] for lb in labels if lb["box_id"] == b]
        out += row_order(mine, l2r[b]) if rule == "serpentine" else order_items(mine, rule)
    return out


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
