"""Task-3 scan order and scanner stand-off on the algorithm side (pure functions, no ground truth). The evaluator has its
own copy of the same rules (g1agent/eval/scan_order.py; algorithm code may not import the evaluation package,
tests/test_no_gt_leak.py); tests/test_task3_logic.py checks that both give the same order.

- Order "left to right, then top to bottom" (user requirement 4; column interpretation = design decision): two labels
  share a column iff their lateral extents (y in the planning root frame, left = +y) overlap; columns are the connected
  groups of overlapping extents. Columns go left to right, inside a column labels go top to bottom (descending z).
  This is the default order rule "columns".
- Order rule "rows" (a property of the task's target spec, user decision 2026-11-19 for the 4-box scanroom test): rows
  are the connected groups of overlapping vertical extents (z), the same rule on the other axis; rows go top to bottom,
  inside a row labels go left to right (descending lateral centre).
- Target grid (task instruction in robot terms, e.g. "columns 1-2 from the left, rows 2-3 from the top of the box
  wall"): the columns and rows above, counted from the stated edge, over the localised labels (parse_targets,
  resolve_grid, grid_cell).
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


ORDER_RULES = ("columns", "rows")


def _overlap_groups(items: list[dict], lo: str, hi: str) -> list[list[dict]]:
    """Connected components of the 'extents [lo, hi] overlap' relation (strict overlap: touching extents are separate
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

    Columns = connected groups of overlapping lateral extents (_overlap_groups), left to right. Ties inside a column
    (equal z) are broken by descending lateral centre, then by key, so the order is deterministic."""
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
    """Scan order of the items by an order rule (ORDER_RULES)."""
    if rule == "columns":
        return column_order(items)
    if rule == "rows":
        return row_order(items)
    raise ValueError(f"unknown order rule {rule!r} (known: {ORDER_RULES})")


def order_items_in_frame(labels: list[dict], T_odom_frame) -> list[dict]:
    """labels: [{"key", "corners" (Nx3 odom), "center" (3, odom)}]; T_odom_frame: 4x4 reference (planning root) frame.
    Returns the order items ({"key", "y_lo", "y_hi", "z_lo", "z_hi", "z"}) in that frame."""
    T_fo = np.linalg.inv(np.asarray(T_odom_frame, float))
    items = []
    for lb in labels:
        c = np.asarray(lb["corners"], float)
        cf = (T_fo[:3, :3] @ c.T).T + T_fo[:3, 3]
        zc = float((T_fo[:3, :3] @ np.asarray(lb["center"], float) + T_fo[:3, 3])[2])
        items.append({"key": lb["key"], "y_lo": float(cf[:, 1].min()), "y_hi": float(cf[:, 1].max()),
                      "z_lo": float(cf[:, 2].min()), "z_hi": float(cf[:, 2].max()), "z": zc})
    return items


# ------------------------------------------------------------------------------------------------- target grid
# Each counted dimension: (spec key, axis of the items, the edge it is counted from). Columns are counted on the lateral
# axis (y, left = +y), rows on the vertical axis (z); _overlap_groups lists them from the high end (left / top).
_DIMS = {"columns": (("columns_from_left", "left"), ("columns_from_right", "right")),
         "rows": (("rows_from_top", "top"), ("rows_from_bottom", "bottom"))}
_AXIS = {"columns": ("y_lo", "y_hi"), "rows": ("z_lo", "z_hi")}


def parse_targets(spec) -> dict | None:
    """Task target spec in robot terms -> {"columns": None | {"from", "index"}, "rows": None | {"from", "index"},
    "order": rule, "spec": the spec}. spec: None (every label, order "columns") or a dict with at most one of
    columns_from_left / columns_from_right, at most one of rows_from_top / rows_from_bottom (lists of 1-based indices
    counted from that edge of the box wall) and "order" (ORDER_RULES, default "columns"). Anything else is an error."""
    if spec is None:
        return None
    if not isinstance(spec, dict):
        raise ValueError(f"scan target spec must be a dict, got {type(spec).__name__}")
    known = {k for opts in _DIMS.values() for k, _ in opts} | {"order"}
    bad = set(spec) - known
    if bad:
        raise ValueError(f"unknown scan target keys {sorted(bad)} (known: {sorted(known)})")
    out = {"order": spec.get("order", "columns"), "spec": dict(spec)}
    if out["order"] not in ORDER_RULES:
        raise ValueError(f"scan target order must be one of {ORDER_RULES}, got {out['order']!r}")
    for dim, opts in _DIMS.items():
        given = [(k, edge) for k, edge in opts if k in spec]
        if len(given) > 1:
            raise ValueError(f"scan target spec counts {dim} from two edges: {[k for k, _ in given]}")
        out[dim] = None
        if given:
            k, edge = given[0]
            idx = spec[k]
            if not isinstance(idx, list) or not idx or any(not isinstance(i, int) or isinstance(i, bool) or i < 1 for i in idx) \
                    or len(set(idx)) != len(idx):
                raise ValueError(f"scan target {k} must be a non-empty list of distinct indices >= 1, got {idx!r}")
            out[dim] = {"from": edge, "index": sorted(idx)}
    return out


def describe_targets(t: dict | None) -> str:
    """The target spec in words (robot terms, for prompts and tool results)."""
    if t is None:
        return "every barcode label"
    parts = []
    for dim in ("columns", "rows"):
        if t[dim] is not None:
            parts.append(f"{dim} {', '.join(str(i) for i in t[dim]['index'])} from the {t[dim]['from']}")
    return "the labels of the boxes in " + " and ".join(parts) + " of the box wall" if parts else "every barcode label"


def order_rule_text(rule: str) -> str:
    return {"columns": "columns of overlapping lateral label extents, left to right as seen facing this side; "
                       "top to bottom inside a column",
            "rows": "rows of overlapping vertical label extents, top to bottom; left to right inside a row as seen "
                    "facing this side"}[rule]


_OPP = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}
_HIGH_EDGE = {"columns": "left", "rows": "top"}     # the edge at the high end of each axis (left = +y, top = +z)


def _n_missing(space: float, unit: float) -> int:
    """Number of box rows / columns a free space can hold: the nearest whole number of boxes of size `unit`. The
    decision between k and k + 1 missing boxes lies halfway between the two hypotheses (space = k or k + 1 boxes)."""
    return max(0, int(math.floor(space / unit + 0.5)))


def _dim_slots(items: list[dict], dim: str, spec: dict | None, edges: list, centres: list) -> dict:
    """Slots (box rows / columns) of one dimension, counted from the spec's edge. See resolve_grid."""
    lo_k, hi_k = _AXIS[dim]
    edge = spec["from"] if spec is not None else _HIGH_EDGE[dim]
    high = edge == _HIGH_EDGE[dim]
    # u: the axis coordinate oriented so that larger u = closer to the counted edge
    to_u = (lambda lo, hi: (lo, hi)) if high else (lambda lo, hi: (-hi, -lo))
    from_u = to_u                                           # the same map is its own inverse
    groups = _overlap_groups(items, lo_k, hi_k)             # high end first
    if not high:
        groups = groups[::-1]
    gi = {it["key"]: i for i, g in enumerate(groups) for it in g}
    # box faces carrying the labels (the fitted face each label was found on)
    faces = {}
    for it in items:
        f = it.get("face")
        if f is None:
            continue
        e = faces.setdefault(f["key"], {"u": to_u(f[lo_k], f[hi_k]), "full": bool(f["seen"][edge] and f["seen"][_OPP[edge]]),
                                        "groups": set()})
        e["groups"].add(gi[it["key"]])
    # a face whose both edges along the axis were seen inside an image and that carries labels of one group only
    full = [f for f in faces.values() if f["full"] and len(f["groups"]) == 1]
    unit = min((f["u"][1] - f["u"][0] for f in full), default=None)        # the smallest box seen whole
    singles = {}
    for f in full:
        if f["u"][1] - f["u"][0] < 1.5 * unit:        # under 1.5 boxes: one box (halfway between one and two boxes)
            singles.setdefault(next(iter(f["groups"])), []).append(f)
    box = {i: (min(f["u"][0] for f in fs), max(f["u"][1] for f in fs)) for i, fs in singles.items()}
    warnings, notes = [], []
    slots = []                                              # in counting order, each {"u", "box", "keys", "inserted"}

    def insert(k, lo, hi, where):
        for j in range(k):                                  # equal shares of the free space, from the counted side
            slots.append({"u": (hi - (j + 1) * (hi - lo) / k, hi - j * (hi - lo) / k), "box": True, "keys": [],
                          "inserted": True})
        if k:
            notes.append(f"{k} {dim[:-1]}(s) of boxes without a localised label {where} (from the box faces)")

    anchored = None
    gaps_unknown = []                                       # slot counts before which a gap could not be excluded
    if spec is not None and groups:
        # the wall edge: a face edge seen inside a head image with no detected box beyond it in that view, and no fitted
        # face centre / localised label centre beyond it in any view
        beyond = [to_u(c, c)[1] for c in centres] + [sum(to_u(it[lo_k], it[hi_k])) / 2 for it in items]
        cand = [to_u(ob["coord"], ob["coord"])[1] for ob in edges if ob["edge"] == edge and ob["clear"]]
        cand = [e for e in cand if not any(b > e for b in beyond)]
        if not cand:
            anchored = False
            warnings.append(f"the {edge} edge of the box wall was not seen with nothing beyond it: {dim} are counted "
                            f"from the {edge}most {dim[:-1]} of labels seen")
        elif 0 not in box:
            anchored = False
            warnings.append(f"the {edge} edge of the box wall was seen, but the boxes of the {edge}most {dim[:-1]} of "
                            f"labels were not seen whole: {dim} of boxes without labels between it and the edge cannot "
                            "be excluded")
        else:
            anchored = True
            e = max(cand)
            insert(_n_missing(e - box[0][1], unit), box[0][1], e, f"at the {edge} edge")
    for i, g in enumerate(groups):
        if spec is not None and i > 0:
            if i - 1 in box and i in box:
                insert(_n_missing(box[i - 1][0] - box[i][1], unit), box[i][1], box[i - 1][0],
                       f"between the {dim} of labels {len(slots)} and {len(slots) + 1}")
            else:
                gaps_unknown.append(len(slots))
        lo_u, hi_u = (box[i] if i in box else
                      (min(to_u(it[lo_k], it[hi_k])[0] for it in g), max(to_u(it[lo_k], it[hi_k])[1] for it in g)))
        slots.append({"u": (lo_u, hi_u), "box": i in box, "keys": [it["key"] for it in g], "inserted": False})
    if spec is not None:
        last = max(spec["index"])
        for n in gaps_unknown:
            if n < last:                                    # the count up to a requested index crosses this gap
                warnings.append(f"between {dim[:-1]}s {n} and {n + 1} (counted from the {edge}) a {dim[:-1]} of boxes "
                                "without localised labels cannot be excluded (their box faces were not seen whole)")
    out = []
    for n, sl in enumerate(slots):
        lo, hi = from_u(*sl["u"])
        out.append({"index": n + 1, "lo": float(lo), "hi": float(hi), "box": sl["box"], "keys": sl["keys"],
                    "inserted": sl["inserted"]})
    evidence = None
    if spec is not None:
        evidence = "box faces" if not warnings else "labels (box faces not resolved where the warnings say)"
    return {"slots": out, "anchored": anchored, "warnings": warnings, "notes": notes, "evidence": evidence,
            "edge": edge, "unit": unit}


def resolve_grid(items: list[dict], targets: dict, edges: list | None = None, face_centres: list | None = None) -> dict:
    """Box columns and rows of a side (items: order_items_in_frame of every label of the side, planning frame, each with
    "face": the fitted box face it was found on {"key", "y_lo", "y_hi", "z_lo", "z_hi", "seen": {edge: bool}} or None),
    indexed from the edges the target spec counts from, and the labels in the requested cells.

    - Rows / columns of LABELS: the overlap groups of the label extents (as the order rule).
    - Box evidence, independent of label recall where it exists: a box face seen whole along the axis (both its edges
      inside a head image) and carrying labels of one group only is one box when it is under 1.5 times the smallest
      such face (halfway between one and two boxes). The smallest one is the box size. Between two consecutive label
      groups, and between the counted edge of the wall and the outermost label group, the free space between their
      boxes holds round(space / box size) rows / columns of boxes WITHOUT a localised label: they are inserted as empty
      slots (indices are never shifted by a missed label row), so a label found there later fills its cell.
    - The wall edge (edges: [{"edge", "coord" (planning frame z / y), "clear": no detected box beyond it in the view
      that saw it}]) counts only when no fitted face centre (face_centres: [coord on the axis] of every face fitted on
      the side, per dimension {"columns": [...], "rows": [...]}) and no label centre lies beyond it.
    - Where that evidence is missing (edge not seen clear, boxes not seen whole), the count rests on the labels alone:
      it is reported in "warnings" (the plan must not claim success with it).

    Returns {"columns" / "rows": [slots {"index", "lo", "hi", "box", "keys", "inserted"}], "cell_of": {key: {"column",
    "row"}}, "selected", "requested", "resolved", "missing" (cells), "anchored": {edge: bool}, "warnings", "notes",
    "evidence": {dim: str}}. A cell is {"column": i | None, "row": j | None} (None = any, for an unconstrained
    dimension)."""
    dims = {}
    for dim in ("columns", "rows"):
        dims[dim] = _dim_slots(items, dim, targets[dim], [e for e in (edges or []) if e["edge"] in _DIM_EDGES[dim]],
                               (face_centres or {}).get(dim, []))
    index_of = {dim: {k: sl["index"] for sl in dims[dim]["slots"] for k in sl["keys"]} for dim in dims}
    want = {dim: (None if targets[dim] is None else targets[dim]["index"]) for dim in ("columns", "rows")}
    cell_of = {it["key"]: {"column": index_of["columns"][it["key"]], "row": index_of["rows"][it["key"]]} for it in items}
    selected = [it["key"] for it in items
                if all(want[d] is None or cell_of[it["key"]][d[:-1]] in want[d] for d in ("columns", "rows"))]
    requested = [{"column": c, "row": r} for c in (want["columns"] or [None]) for r in (want["rows"] or [None])]
    have = [cell_of[k] for k in selected]
    resolved = [c for c in requested if any(all(c[a] is None or c[a] == h[a] for a in ("column", "row")) for h in have)]
    missing = [c for c in requested if c not in resolved]
    return {"columns": dims["columns"]["slots"], "rows": dims["rows"]["slots"], "cell_of": cell_of, "selected": selected,
            "requested": requested, "resolved": resolved, "missing": missing,
            "anchored": {dims[d]["edge"]: dims[d]["anchored"] for d in dims if targets[d] is not None},
            "warnings": dims["columns"]["warnings"] + dims["rows"]["warnings"],
            "notes": dims["columns"]["notes"] + dims["rows"]["notes"],
            "evidence": {d: dims[d]["evidence"] for d in dims if targets[d] is not None}}


_DIM_EDGES = {"columns": ("left", "right"), "rows": ("top", "bottom")}


def grid_cell(item: dict, grid: dict) -> dict:
    """Cell of a label found after the grid was resolved (plan update): per dimension, the slot that holds it: a box slot
    (box extent known) when it contains the label centre, a label-only slot when it overlaps the label (strict overlap,
    the rule that formed the label groups); None when no slot or more than one does."""
    out = {}
    for dim, a in (("columns", "column"), ("rows", "row")):
        lo, hi = _AXIS[dim]
        c = (item[lo] + item[hi]) / 2
        hit = [g["index"] for g in grid[dim]
               if (g["lo"] <= c <= g["hi"] if g["box"] else (item[hi] > g["lo"] and g["hi"] > item[lo]))]
        out[a] = hit[0] if len(hit) == 1 else None
    return out


def grid_conflict(item: dict, cell: dict, grid: dict, targets: dict) -> str | None:
    """A label found after the grid was resolved that no slot holds, in a constrained dimension, on the counted side of
    the farthest requested slot: the count up to the requested cells did not see it (the indices may be off)."""
    for dim, a in (("columns", "column"), ("rows", "row")):
        spec = targets[dim]
        if spec is None or cell[a] is not None:
            continue
        lo, hi = _AXIS[dim]
        c = (item[lo] + item[hi]) / 2
        far = [g for g in grid[dim] if g["index"] == max(spec["index"])]
        if not far:                                        # fewer slots than the farthest requested index
            return f"a label outside the resolved {dim} (the grid has fewer {dim} than requested)"
        g = far[0]
        if (c > g["lo"]) if spec["from"] == _HIGH_EDGE[dim] else (c < g["hi"]):
            return f"a label on the counted side of {dim[:-1]} {g['index']} that no resolved {dim[:-1]} holds"
    return None


def cell_targeted(cell: dict, targets: dict) -> bool:
    """A label in this cell is to be scanned (a None index in a constrained dimension is not)."""
    return all(targets[d] is None or cell[d[:-1]] in targets[d]["index"] for d in ("columns", "rows"))


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
