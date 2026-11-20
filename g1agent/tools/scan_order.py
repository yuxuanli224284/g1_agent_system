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
- Order rule "serpentine" (user decision 2026-11-19, the wall-layout suite): the rows of rule "rows", top to bottom;
  the top row (of the rows that hold labels to scan) left to right, the next right to left, alternating, so the robot
  starts each row where it ended the previous one. Ties inside a row as in rule "rows" (by descending z, then key).
- Target grid (task instruction in robot terms, e.g. "columns 1-2 from the left, rows 2-3 from the top of the box
  wall", or "all columns, rows 2-4 from the bottom"): box rows / columns from the box faces seen whole (labels go to the
  nearest box slot holding their centre; label overlap groups only where no box was seen whole), counted from the
  stated edge ("all": anchored at both edges); the floor bounds the bottom edge (parse_targets, resolve_grid,
  grid_cell).
- Scan order inside a resolved grid (order_by_cells): the order rule applied to the box cells of the labels, so a
  duplicate or a fragment inside a cell never forms a row / column of its own.
- Scan order of a given wall layout (order_lattice): the order rule applied to the layout cells themselves (no label
  extents); several labels inside one cell: order_in_cell.
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


ORDER_RULES = ("columns", "rows", "serpentine")


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
    """Scan order of the items by an order rule (ORDER_RULES)."""
    if rule == "columns":
        return column_order(items)
    if rule == "rows":
        return row_order(items)
    if rule == "serpentine":
        return serpentine_order(items)
    raise ValueError(f"unknown order rule {rule!r} (known: {ORDER_RULES})")


def order_in_cell(items: list[dict], rule: str, left_to_right: bool = True) -> list:
    """Scan order of several labels inside ONE box cell (items as order_items): the label rule of the task; for rule
    "serpentine" the rows of labels inside the cell top to bottom, each in the lateral direction of the cell's row
    (left_to_right: that row's direction, see order_lattice), so the robot keeps walking the same way."""
    if rule == "serpentine":
        return row_order(items, left_to_right)
    return order_items(items, rule)


def order_lattice(cells: list[dict], rule: str = "columns") -> list[dict]:
    """Scan order of the cells of a given wall layout (cells: [{"key", "column": 1 = leftmost, "row": 1 = top}], the
    lattice itself, no label extents). Rule "rows": rows from the top, columns from the left inside a row; "columns":
    columns from the left, rows from the top inside a column; "serpentine": rows from the top, the k-th row (k = 0 at
    the top, counted over the rows that hold cells) left to right when k is even, right to left when k is odd.
    Returns [{"key", "left_to_right": the lateral direction of the cell's row (order_in_cell)}] in scan order."""
    if rule not in ORDER_RULES:
        raise ValueError(f"unknown order rule {rule!r} (known: {ORDER_RULES})")
    rank = {r: k for k, r in enumerate(sorted({c["row"] for c in cells}))}
    l2r = {c["key"]: rule != "serpentine" or rank[c["row"]] % 2 == 0 for c in cells}

    def key(c):
        col = c["column"] if l2r[c["key"]] else -c["column"]
        return ((c["column"], c["row"]) if rule == "columns" else (c["row"], col)), str(c["key"])
    return [{"key": c["key"], "left_to_right": l2r[c["key"]]} for c in sorted(cells, key=key)]


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
    """Task target spec in robot terms -> {"columns": None | {"from", "index"} | {"from", "index": None, "all": True},
    "rows": (same), "order": rule, "spec": the spec}. spec: None (every label, order "columns") or a dict with at most
    one of columns_from_left / columns_from_right / columns, at most one of rows_from_top / rows_from_bottom / rows
    (*_from_*: lists of 1-based indices counted from that edge of the box wall; "columns" / "rows": the value "all",
    every column / row of the wall, which needs both of its edges) and "order" (ORDER_RULES, default "columns").
    Anything else is an error."""
    if spec is None:
        return None
    if not isinstance(spec, dict):
        raise ValueError(f"scan target spec must be a dict, got {type(spec).__name__}")
    known = {k for opts in _DIMS.values() for k, _ in opts} | set(_DIMS) | {"order"}
    bad = set(spec) - known
    if bad:
        raise ValueError(f"unknown scan target keys {sorted(bad)} (known: {sorted(known)})")
    out = {"order": spec.get("order", "columns"), "spec": dict(spec)}
    if out["order"] not in ORDER_RULES:
        raise ValueError(f"scan target order must be one of {ORDER_RULES}, got {out['order']!r}")
    for dim, opts in _DIMS.items():
        given = [(k, edge) for k, edge in opts if k in spec]
        if dim in spec:
            if given:
                raise ValueError(f"scan target spec gives {dim!r} together with {[k for k, _ in given]}")
            if spec[dim] != "all":
                raise ValueError(f"scan target {dim!r} must be \"all\", got {spec[dim]!r}")
            # the count runs from the high edge (left / top); "all" is anchored at both edges, so the start side
            # does not change the result
            out[dim] = {"from": _HIGH_EDGE[dim], "index": None, "all": True}
            continue
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
        if t[dim] is not None and t[dim].get("all"):
            parts.append(f"all {dim}")
        elif t[dim] is not None:
            parts.append(f"{dim} {', '.join(str(i) for i in t[dim]['index'])} from the {t[dim]['from']}")
    return "the labels of the boxes in " + " and ".join(parts) + " of the box wall" if parts else "every barcode label"


def order_rule_text(rule: str) -> str:
    return {"columns": "columns of overlapping lateral label extents, left to right as seen facing this side; "
                       "top to bottom inside a column",
            "rows": "rows of overlapping vertical label extents, top to bottom; left to right inside a row as seen "
                    "facing this side",
            "serpentine": "rows of overlapping vertical label extents, top to bottom; the top row left to right as seen "
                          "facing this side, the next right to left, alternating"}[rule]


_OPP = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}
_HIGH_EDGE = {"columns": "left", "rows": "top"}     # the edge at the high end of each axis (left = +y, top = +z)


def _n_missing(space: float, unit: float) -> int:
    """Number of box rows / columns a free space can hold: the nearest whole number of boxes of size `unit`. The
    decision between k and k + 1 missing boxes lies halfway between the two hypotheses (space = k or k + 1 boxes)."""
    return max(0, int(math.floor(space / unit + 0.5)))


def _mid(extents) -> float:
    """Mean centre of a list of extents (lo, hi)."""
    return float(np.mean([(lo + hi) / 2 for lo, hi in extents]))


def _nearest_slot(c: float, extents: list, centres: list) -> int | None:
    """Index of the box slot for a label centre c: among the slots whose extent [lo, hi] holds c, the one whose centre is
    nearest (the axis is split halfway between two slot centres where their extents overlap); None when no slot holds
    it or two hold it at the same distance."""
    hit = sorted((abs(c - centres[j]), j) for j, (lo, hi) in enumerate(extents) if lo <= c <= hi)
    if not hit or (len(hit) > 1 and hit[0][0] == hit[1][0]):
        return None
    return hit[0][1]


def _dim_slots(items: list[dict], dim: str, spec: dict | None, edges: list, centres: list, faces: list = (),
               bounds: dict | None = None) -> dict:
    """Slots (box rows / columns) of one dimension, counted from the spec's edge. See resolve_grid."""
    lo_k, hi_k = _AXIS[dim]
    edge = spec["from"] if spec is not None else _HIGH_EDGE[dim]
    every = spec is not None and bool(spec.get("all"))
    high = edge == _HIGH_EDGE[dim]
    # u: the axis coordinate oriented so that larger u = closer to the counted edge
    to_u = (lambda lo, hi: (lo, hi)) if high else (lambda lo, hi: (-hi, -lo))
    from_u = to_u                                           # the same map is its own inverse
    # 1. box slots: the fitted box faces of the side (one entry per fitted face, key) seen whole along the axis (both
    # edges inside a head image).
    # - Box size (unit): the smallest whole face that CARRIES a label (a plan entry was found on it), the evidence the
    #   rule trusted before D4. A face without a label may be a part of a box or a floor patch that happens to lie
    #   inside the image: it never sets the box size.
    # - A whole face is one box when round(size / unit) == 1 (_n_missing: halfway between no box and one, and between
    #   one and two); with or without a label it is then box evidence. A sliver (under half a box) or a face of several
    #   boxes is not.
    # - One box row / column = the boxes whose centres lie within half a unit of the row's mean centre
    #   (_n_missing(distance, unit) == 0: halfway between the same row, distance 0, and the next one, distance 1 unit;
    #   extents of adjacent boxes touch, so strict extent overlap would chain neighbours on fitting noise). Faces are
    #   taken from the counted edge; each joins the current row (the nearest one) or starts the next.
    # - A row counts only when its faces agree on one box: round(union / unit) == 1. Otherwise its faces are no box
    #   evidence, and a note says so (the lost evidence is visible).
    # - Slot extent = the union of its faces (outer bounds); slot centre = the mean of its face centres (a sloppy fit
    #   moves it by its error / number of faces, and it decides between two slots that both hold a label, step 2).
    fs = {}
    for f in list(faces) + [it["face"] for it in items if it.get("face") is not None]:
        fs.setdefault(f["key"], f)
    labelled = {it["face"]["key"] for it in items if it.get("face") is not None}
    whole = {k: to_u(f[lo_k], f[hi_k]) for k, f in fs.items() if f["seen"][edge] and f["seen"][_OPP[edge]]}
    unit = min((hi - lo for k, (lo, hi) in whole.items() if k in labelled), default=None)
    singles = [] if unit is None else sorted((u for u in whole.values() if _n_missing(u[1] - u[0], unit) == 1),
                                             key=lambda u: -(u[0] + u[1]))
    clusters = []
    for u in singles:                                       # from the counted edge
        if clusters and _n_missing(_mid(clusters[-1]) - (u[0] + u[1]) / 2, unit) == 0:
            clusters[-1].append(u)
        else:
            clusters.append([u])
    warnings, notes = [], []
    box_u, box_c = [], []
    for cl in clusters:
        lo_u, hi_u = min(u[0] for u in cl), max(u[1] for u in cl)
        if _n_missing(hi_u - lo_u, unit) == 1:
            box_u.append((lo_u, hi_u))
            box_c.append(_mid(cl))
        else:
            lo, hi = from_u(lo_u, hi_u)
            notes.append(f"{len(cl)} box face(s) seen whole at {dim[:-1]} extent {lo:.3f}..{hi:.3f} m do not agree on "
                         f"one box ({(hi_u - lo_u) / unit:.1f} box sizes together): not used as box evidence")
    # 2. each label goes to the box slot whose centre is nearest to the label centre, among the slots whose extent holds
    # it (grid_cell applies the same rule to late labels): a fit that reaches into the next row does not take that
    # row's labels. Labels in no box slot (or exactly halfway between two) form label-only slots: their overlap groups,
    # as the order rule
    box_keys = [[] for _ in box_u]
    free = []
    for it in items:
        j = _nearest_slot(sum(to_u(it[lo_k], it[hi_k])) / 2, box_u, box_c)
        if j is not None:
            box_keys[j].append(it["key"])
        else:
            free.append({"key": it["key"], "u_lo": to_u(it[lo_k], it[hi_k])[0], "u_hi": to_u(it[lo_k], it[hi_k])[1]})
    core = [{"u": b, "c": c, "box": True, "keys": ks} for b, c, ks in zip(box_u, box_c, box_keys)]
    core += [{"u": (lo, hi), "c": (lo + hi) / 2, "box": False, "keys": [it["key"] for it in g]}
             for g in _overlap_groups(free, "u_lo", "u_hi")
             for lo, hi in [(min(it["u_lo"] for it in g), max(it["u_hi"] for it in g))]]
    core.sort(key=lambda sl: (-sl["c"], -sl["u"][1]))      # from the counted edge
    anchored = {}

    def anchor(which):
        """The wall edge at the counted end ("near") or the opposite end ("far"): an edge observation that is clear
        (no detected box beyond it in the view that saw it) or, for an edge with a physical bound (bounds: the floor
        under the bottom edge), that leaves no room for a box between it and the bound (_n_missing == 0); and no fitted
        face centre / localised label centre beyond it in any view; and the outermost slot at that end must be a box
        slot (else rows / columns of boxes without labels between it and the edge cannot be excluded). Returns the
        edge (u) or None, with a warning."""
        name = edge if which == "near" else _OPP[edge]
        s = 1 if which == "near" else -1                    # outward direction of this edge in u
        bound = None if bounds is None or name not in bounds else to_u(bounds[name], bounds[name])[0]
        beyond = ([s * to_u(c, c)[0] for c in centres] + [s * sum(to_u(f[lo_k], f[hi_k])) / 2 for f in fs.values()]
                  + [s * sum(to_u(it[lo_k], it[hi_k])) / 2 for it in items])
        cand = []
        for ob in edges:
            if ob["edge"] != name:
                continue
            e = to_u(ob["coord"], ob["coord"])[0]
            on_bound = bound is not None and unit is not None and _n_missing(s * (bound - e), unit) == 0
            if ob["clear"] or on_bound:
                cand.append(s * e)
        cand = [e for e in cand if not any(b > e for b in beyond)]
        outer = core[0] if which == "near" else core[-1]
        if not cand:
            anchored[name] = False
            warnings.append(f"the {name} edge of the box wall was not seen with nothing beyond it: {dim} are counted "
                            f"from the {name}most {dim[:-1]} of labels seen")
            return None
        if not outer["box"]:
            anchored[name] = False
            warnings.append(f"the {name} edge of the box wall was seen, but the boxes of the {name}most {dim[:-1]} of "
                            f"labels were not seen whole: {dim} of boxes without labels between it and the edge cannot "
                            "be excluded")
            return None
        anchored[name] = True
        return s * max(cand)

    slots = []                                  # in counting order, each {"u", "c", "box", "keys", "inserted"}

    def insert(k, lo, hi, where):
        for j in range(k):                                  # equal shares of the free space, from the counted side
            u = (hi - (j + 1) * (hi - lo) / k, hi - j * (hi - lo) / k)
            slots.append({"u": u, "c": (u[0] + u[1]) / 2, "box": True, "keys": [], "inserted": True})
        if k:
            notes.append(f"{k} {dim[:-1]}(s) of boxes without a localised label {where} (from the box faces)")

    gaps_unknown = []                                       # slot counts before which a gap could not be excluded
    if spec is not None and not core:
        for name in ((edge, _OPP[edge]) if every else (edge,)):
            anchored[name] = False
        warnings.append(f"no {dim[:-1]} of boxes or labels was localised: {dim} cannot be counted")
    near = far = None
    if spec is not None and core:
        near = anchor("near")
        if every:
            far = anchor("far")
    if near is not None:
        insert(_n_missing(near - core[0]["u"][1], unit), core[0]["u"][1], near, f"at the {edge} edge")
    for i, sl in enumerate(core):
        if spec is not None and i > 0:
            if core[i - 1]["box"] and sl["box"]:
                insert(_n_missing(core[i - 1]["u"][0] - sl["u"][1], unit), sl["u"][1], core[i - 1]["u"][0],
                       f"between the {dim} of labels {len(slots)} and {len(slots) + 1}")
            else:
                gaps_unknown.append(len(slots))
        if sl["box"] and not sl["keys"]:
            notes.append(f"{dim[:-1]} {len(slots) + 1} (from the {edge}): boxes seen whole without a localised label")
        slots.append({**sl, "inserted": False})
    if far is not None:
        insert(_n_missing(core[-1]["u"][0] - far, unit), far, core[-1]["u"][0], f"at the {_OPP[edge]} edge")
    if spec is not None:
        last = len(slots) if every else max(spec["index"])  # "all": every gap matters
        for n in gaps_unknown:
            if n < last:                                    # the count up to a requested index crosses this gap
                warnings.append(f"between {dim[:-1]}s {n} and {n + 1} (counted from the {edge}) a {dim[:-1]} of boxes "
                                "without localised labels cannot be excluded (their box faces were not seen whole)")
    out = []
    for n, sl in enumerate(slots):
        lo, hi = from_u(*sl["u"])
        out.append({"index": n + 1, "lo": float(lo), "hi": float(hi), "center": float(from_u(sl["c"], sl["c"])[0]),
                    "box": sl["box"], "keys": sl["keys"], "inserted": sl["inserted"]})
    evidence = None
    if spec is not None:
        evidence = "box faces" if not warnings else "labels (box faces not resolved where the warnings say)"
    return {"slots": out, "anchored": anchored, "warnings": warnings, "notes": notes, "evidence": evidence,
            "edge": edge, "unit": unit}


def resolve_grid(items: list[dict], targets: dict, edges: list | None = None, face_centres: list | None = None,
                 faces: list | None = None, bounds: dict | None = None) -> dict:
    """Box columns and rows of a side (items: order_items_in_frame of every label of the side, planning frame, each with
    "face": the fitted box face it was found on {"key", "y_lo", "y_hi", "z_lo", "z_hi", "seen": {edge: bool}} or None),
    indexed from the edges the target spec counts from, and the labels in the requested cells.

    - Box slots, independent of label recall where they exist: the fitted box faces of the side (faces: [face dicts
      as above, planning frame], with or without labels, plus the faces of the items; one entry per key) seen whole
      along the axis (both its edges inside a head image). The box size is the smallest such face that carries a label
      (a face without a label may be part of a box or a floor patch: it never sets the box size); a face of
      round(size / box size) == 1 is one box (halfway rules), with or without a label; boxes whose centres lie less
      than half a box size from the mean centre of a row (halfway between the same and the next row) form that box row
      / column when together they still round to one box; its extent = their union, its centre = the mean of their
      centres. A row whose faces disagree is no box evidence (a note says so).
    - Each label goes to the box slot nearest to its centre among those whose extent holds it (as grid_cell for late
      labels), so two label groups (a duplicate, a fragment) inside one box row are one row, and a fit reaching into
      the next row does not take that row's labels. Labels in no box slot form label-only slots: the overlap groups of
      their extents (as the order rule).
    - Between two consecutive box slots, and between the counted edge of the wall and the outermost box slot, the free
      space holds round(space / box size) rows / columns of boxes WITHOUT a localised label: they are inserted as empty
      slots (indices are never shifted by a missed label row), so a label found there later fills its cell. A gap next
      to a label-only slot cannot be excluded: a warning when the count up to a requested index crosses it.
    - The wall edge (edges: [{"edge", "coord" (planning frame z / y), "clear": no detected box beyond it in the view
      that saw it}]) counts when it is clear, or when it lies on a physical bound (bounds: {edge: coord}, e.g.
      {"bottom": floor z}; nothing stands below the floor) with no room for a box between them; and only when no
      fitted face centre (face_centres: per dimension {"columns": [...], "rows": [...]}, plus the faces' centres) and
      no label centre lies beyond it; and only when the outermost slot at that end is a box slot. The bound never
      stands in for an edge that was not seen.
    - A dimension given as "all" (parse_targets) is anchored at BOTH edges (rows / columns of boxes without labels
      inserted at both ends) and requests every slot.
    - Where that evidence is missing (an edge not seen, boxes not seen whole), the count rests on the labels alone:
      it is reported in "warnings" (the plan must not claim success with it).

    Returns {"columns" / "rows": [slots {"index", "lo", "hi", "center", "box", "keys", "inserted"}], "cell_of": {key:
    {"column", "row"}}, "selected", "requested", "resolved", "missing" (cells), "anchored": {edge: bool}, "warnings",
    "notes", "evidence": {dim: str}}. A cell is {"column": i | None, "row": j | None} (None = any, for an unconstrained
    dimension)."""
    dims = {}
    for dim in ("columns", "rows"):
        dims[dim] = _dim_slots(items, dim, targets[dim], [e for e in (edges or []) if e["edge"] in _DIM_EDGES[dim]],
                               (face_centres or {}).get(dim, []), faces or [], bounds)
    index_of = {dim: {k: sl["index"] for sl in dims[dim]["slots"] for k in sl["keys"]} for dim in dims}
    want = {}
    for dim in ("columns", "rows"):
        spec = targets[dim]
        want[dim] = (None if spec is None else
                     [sl["index"] for sl in dims[dim]["slots"]] if spec.get("all") else spec["index"])
    cell_of = {it["key"]: {"column": index_of["columns"][it["key"]], "row": index_of["rows"][it["key"]]} for it in items}
    selected = [it["key"] for it in items
                if all(want[d] is None or cell_of[it["key"]][d[:-1]] in want[d] for d in ("columns", "rows"))]
    requested = [{"column": c, "row": r} for c in (want["columns"] or [None]) for r in (want["rows"] or [None])]
    have = [cell_of[k] for k in selected]
    resolved = [c for c in requested if any(all(c[a] is None or c[a] == h[a] for a in ("column", "row")) for h in have)]
    missing = [c for c in requested if c not in resolved]
    anchored = {}
    for d in dims:
        if targets[d] is not None:
            anchored.update(dims[d]["anchored"])
    return {"columns": dims["columns"]["slots"], "rows": dims["rows"]["slots"], "cell_of": cell_of, "selected": selected,
            "requested": requested, "resolved": resolved, "missing": missing, "anchored": anchored,
            "warnings": dims["columns"]["warnings"] + dims["rows"]["warnings"],
            "notes": dims["columns"]["notes"] + dims["rows"]["notes"],
            "evidence": {d: dims[d]["evidence"] for d in dims if targets[d] is not None}}


_DIM_EDGES = {"columns": ("left", "right"), "rows": ("top", "bottom")}


def grid_cell(item: dict, grid: dict) -> dict:
    """Cell of a label found after the grid was resolved (plan update): per dimension, the slot that holds it, by the
    rule resolve_grid used: the box slot (box extent known) nearest to the label centre among those whose extent
    contains it; else a label-only slot when it overlaps the label (strict overlap, the rule that formed the label
    groups); None when no slot holds it, or two box slots at the same distance or two label-only slots do."""
    out = {}
    for dim, a in (("columns", "column"), ("rows", "row")):
        lo, hi = _AXIS[dim]
        c = (item[lo] + item[hi]) / 2
        box = [g for g in grid[dim] if g["box"]]
        j = _nearest_slot(c, [(g["lo"], g["hi"]) for g in box], [g["center"] for g in box])
        if j is not None or any(g["lo"] <= c <= g["hi"] for g in box):
            out[a] = None if j is None else box[j]["index"]
            continue
        hit = [g["index"] for g in grid[dim] if not g["box"] and item[hi] > g["lo"] and g["hi"] > item[lo]]
        out[a] = hit[0] if len(hit) == 1 else None
    return out


def grid_conflict(item: dict, cell: dict, grid: dict, targets: dict) -> str | None:
    """A label found after the grid was resolved that no slot holds, in a constrained dimension: for a dimension given
    as "all", always (the wall extends beyond the rows / columns counted); otherwise on the counted side of the farthest
    requested slot (the count up to the requested cells did not see it: the indices may be off)."""
    for dim, a in (("columns", "column"), ("rows", "row")):
        spec = targets[dim]
        if spec is None or cell[a] is not None:
            continue
        if spec.get("all"):
            return f"a label that no resolved {dim[:-1]} holds: the box wall extends beyond the {dim} counted"
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
    """A label in this cell is to be scanned (a None index in a constrained dimension is not; "all": any slot)."""
    return all(targets[d] is None or (cell[d[:-1]] is not None if targets[d].get("all") else
                                      cell[d[:-1]] in targets[d]["index"]) for d in ("columns", "rows"))


def order_by_cells(items: list[dict], cells: dict, grid: dict | None, rule: str = "columns") -> list:
    """Scan order of labels by their box cells in a resolved grid (items: order items, cells: {key: {"column", "row"}},
    grid: {"columns": slots, "rows": slots} of resolve_grid). Rule "rows": row slots from the top, inside a row column
    slots from the left; rule "columns": column slots from the left, inside a column row slots from the top (slot
    position = its centre, so the result does not depend on the edge the indices are counted from); rule "serpentine":
    row slots from the top, the k-th of the rows holding labels (k = 0 at the top) left to right when k is even, right
    to left when k is odd. Labels of one cell keep the order of order_items (serpentine: of row_order in their row's
    direction). Without a grid, or when a label has no cell in either dimension, order_items."""
    base = order_items(items, rule)
    keys = [it["key"] for it in items]
    if grid is None or any(cells[k]["column"] is None or cells[k]["row"] is None for k in keys):
        return base
    rank = {k: i for i, k in enumerate(base)}
    pos = {a: {g["index"]: (g["lo"] + g["hi"]) / 2 for g in grid[dim]} for dim, a in (("columns", "column"), ("rows", "row"))}
    if rule == "serpentine":
        rows = sorted({cells[k]["row"] for k in keys}, key=lambda i: -pos["row"][i])          # top first
        l2r = {i: k % 2 == 0 for k, i in enumerate(rows)}
        rank_dir = {d: {k: i for i, k in enumerate(row_order(items, d))} for d in (True, False)}

        def skey(k):
            d = l2r[cells[k]["row"]]
            c = pos["column"][cells[k]["column"]]
            return (-pos["row"][cells[k]["row"]], -c if d else c, rank_dir[d][k])
        return sorted(keys, key=skey)

    def key(k):
        r, c = -pos["row"][cells[k]["row"]], -pos["column"][cells[k]["column"]]      # top first, left first
        return ((r, c) if rule == "rows" else (c, r), rank[k])
    return sorted(base, key=key)


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
