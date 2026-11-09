"""Task-3 scan order and scanner stand-off on the algorithm side (pure functions, no ground truth). The evaluator has its
own copy of the same rules (g1agent/eval/scan_order.py; algorithm code may not import the evaluation package,
tests/test_no_gt_leak.py); tests/test_task3_logic.py checks that both give the same order.

- Order "left to right, then top to bottom" (user requirement 4; column interpretation = design decision): two labels
  share a column iff their lateral extents (y in the planning root frame, left = +y) overlap; columns are the connected
  groups of overlapping extents. Columns go left to right, inside a column labels go top to bottom (descending z).
- Adaptive stand-off (user requirement 5): d = label_width / (2 tan(hfov / 2)) / fill, clamped to [d_min, d_max].
"""
from __future__ import annotations

import math

import numpy as np


def hfov_from_intrinsics(fx: float, width: float) -> float:
    """Horizontal field of view (rad) of a pinhole camera with focal length fx (px) and image width (px)."""
    return 2.0 * math.atan(width / 2.0 / fx)


def adaptive_standoff(label_width_m: float, hfov_rad: float, fill: float, d_min: float, d_max: float) -> float:
    """Distance at which a label of width label_width_m spans `fill` of the scanner image width, clamped."""
    d = label_width_m / (2.0 * math.tan(hfov_rad / 2.0)) / fill
    return float(min(max(d, d_min), d_max))


def column_order(items: list[dict]) -> list:
    """items: [{"key", "y_lo", "y_hi", "z"}] in one reference frame (left = +y). Returns the keys in scan order.

    Columns = connected components of the 'lateral extents overlap' relation (strict overlap: touching extents are
    separate columns). Sweeping the items by descending y_hi, an item joins the current column iff its y_hi lies above
    the column's lowest y_lo; no later item can bridge back to an earlier column because later items end lower.
    Ties inside a column (equal z) are broken by descending lateral centre, then by key, so the order is deterministic."""
    srt = sorted(items, key=lambda it: (-it["y_hi"], -it["y_lo"], str(it["key"])))
    columns, lo = [], None
    for it in srt:
        if columns and it["y_hi"] > lo:
            columns[-1].append(it)
            lo = min(lo, it["y_lo"])
        else:
            columns.append([it])
            lo = it["y_lo"]
    out = []
    for col in columns:                                   # already left to right
        out += [it["key"] for it in sorted(col, key=lambda it: (-it["z"], -(it["y_lo"] + it["y_hi"]) / 2, str(it["key"])))]
    return out


def order_items_in_frame(labels: list[dict], T_odom_frame) -> list[dict]:
    """labels: [{"key", "corners" (Nx3 odom), "center" (3, odom)}]; T_odom_frame: 4x4 reference (planning root) frame.
    Returns the column_order items ({"key", "y_lo", "y_hi", "z"}) in that frame."""
    T_fo = np.linalg.inv(np.asarray(T_odom_frame, float))
    items = []
    for lb in labels:
        c = np.asarray(lb["corners"], float)
        cf = (T_fo[:3, :3] @ c.T).T + T_fo[:3, 3]
        zc = float((T_fo[:3, :3] @ np.asarray(lb["center"], float) + T_fo[:3, 3])[2])
        items.append({"key": lb["key"], "y_lo": float(cf[:, 1].min()), "y_hi": float(cf[:, 1].max()), "z": zc})
    return items


def quad_width(corners) -> float:
    """Width of a planar label quad (4x3, consecutive corners): the longer of the two mean opposite-side lengths. The
    scanner's roll about its axis is free, so the label's longer extent is what has to fit the scanner image width."""
    c = np.asarray(corners, float)
    s = [np.linalg.norm(c[(i + 1) % 4] - c[i]) for i in range(4)]
    return float(max((s[0] + s[2]) / 2.0, (s[1] + s[3]) / 2.0))


def point_in_quad(p, quad) -> bool:
    """p (2,) inside the convex pixel quad (4x2, consecutive corners), boundary included."""
    q = np.asarray(quad, float)
    sign = 0.0
    for i in range(len(q)):
        a, b = q[i], q[(i + 1) % len(q)]
        cr = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
        if cr != 0:
            if sign == 0:
                sign = np.sign(cr)
            elif np.sign(cr) != sign:
                return False
    return True
