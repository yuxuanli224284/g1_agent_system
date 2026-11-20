"""Task-3 target spec in robot terms (columns / rows of the box wall counted from an edge, scan order rule): parsing,
the label grid, plan_scans with targets on a synthetic head frame (targets planned in the spec's order, the rest
not_targeted, a missing cell and an unseen edge reported), the plan update, the extra looks before the grid is resolved,
and the Harness / rule success bookkeeping."""
import numpy as np
import pytest

from g1agent.common import config
from g1agent.perception import geometry as geo
from g1agent.tools import scan_order as so
from g1agent.tools import toolbox as tbmod
from test_task3_logic import FakeRobot, _harness, _head_frame, bare_toolbox

SPEC = {"columns_from_left": [1, 2], "rows_from_top": [2, 3], "order": "rows"}


# ------------------------------------------------------------------ pure rules
def test_row_order_matches_evaluator_copy_and_rule():
    from g1agent.eval import scan_order as ev
    rng = np.random.default_rng(1)
    for _ in range(300):
        items = []
        for k in range(int(rng.integers(1, 8))):
            y, z = float(rng.uniform(-0.6, 0.6)), float(rng.uniform(0.2, 1.3))
            w, h = float(rng.uniform(0.04, 0.2)), float(rng.uniform(0.02, 0.08))
            items.append({"key": f"k{k}", "y_lo": y - w / 2, "y_hi": y + w / 2, "z_lo": z - h / 2, "z_hi": z + h / 2, "z": z})
        assert so.row_order(items) == ev.row_order(items) and so.column_order(items) == ev.column_order(items)
    it = lambda k, y, z: {"key": k, "y_lo": y - 0.05, "y_hi": y + 0.05, "z_lo": z - 0.02, "z_hi": z + 0.02, "z": z}
    grid = [it("a", 0.4, 1.0), it("b", 0.2, 1.01), it("c", 0.4, 0.7), it("d", 0.2, 0.69)]
    assert so.order_items(grid, "rows") == ["a", "b", "c", "d"] and so.order_items(grid, "columns") == ["a", "c", "b", "d"]
    with pytest.raises(ValueError):
        so.order_items(grid, "spiral")


@pytest.mark.parametrize("bad", [{"columns_from_left": [1], "columns_from_right": [1]}, {"rows_from_top": []},
                                 {"rows_from_top": [0]}, {"rows_from_top": [1, 1]}, {"rows_from_top": "2"},
                                 {"order": "diagonal"}, {"boxes": ["C1R4"]}, ["C1R4"]])
def test_parse_targets_rejects_bad_specs(bad):
    with pytest.raises(ValueError):
        so.parse_targets(bad)


def test_parse_and_describe():
    t = so.parse_targets(SPEC)
    assert t["columns"] == {"from": "left", "index": [1, 2]} and t["rows"] == {"from": "top", "index": [2, 3]}
    assert t["order"] == "rows" and so.parse_targets(None) is None
    assert so.describe_targets(t) == "the labels of the boxes in columns 1, 2 from the left and rows 2, 3 from the top of the box wall"
    assert so.parse_targets({"rows_from_bottom": [1]})["order"] == "columns"


# a 3-column wall seen from the robot's left part: columns y 0.5 / 0.3 / 0.1 (box 0.2 m wide), box rows R5..R2 with tops
# 1.25 / 1.0 / 0.75 / 0.5 (0.25 m tall); one label at the top of each box (as on the scanroom wall)
COLS3 = (0.5, 0.3, 0.1)
TOPS = {5: 1.25, 4: 1.0, 3: 0.75, 2: 0.5}


def _box_face(y, top, seen=True):
    return {"key": (round(y, 3), top), "y_lo": y - 0.1, "y_hi": y + 0.1, "z_lo": top - 0.25, "z_hi": top,
            "seen": {e: seen for e in ("top", "bottom", "left", "right")}}


def _grid_items(rows=(5, 4, 3), skip=(), faces=True):
    out = []
    for ci, y in enumerate(COLS3):
        for r in rows:
            if (ci + 1, r) in skip:
                continue
            z = TOPS[r] - 0.025
            out.append({"key": f"C{ci + 1}R{r}", "y_lo": y - 0.05, "y_hi": y + 0.05, "z_lo": z - 0.02, "z_hi": z + 0.02, "z": z,
                        "face": _box_face(y, TOPS[r]) if faces else None})
    return out


def _evidence(rows, top_clear=None):
    """Edge observations (top edges of the top-row faces, left edges of the C1 faces) and the fitted face centres."""
    top_clear = {r: True for r in rows} if top_clear is None else top_clear
    edges = [{"edge": "top", "coord": TOPS[r], "clear": top_clear[r]} for r in rows]
    edges += [{"edge": "left", "coord": COLS3[0] + 0.1, "clear": True}]
    centres = {"columns": [y for y in COLS3 for _ in rows], "rows": [TOPS[r] - 0.125 for _ in COLS3 for r in rows]}
    return edges, centres


def test_resolve_grid_on_box_faces():
    t = so.parse_targets(SPEC)
    edges, centres = _evidence((5, 4, 3), top_clear={5: True, 4: False, 3: False})
    g = so.resolve_grid(_grid_items(), t, edges, centres)
    assert sorted(g["selected"]) == ["C1R3", "C1R4", "C2R3", "C2R4"] and g["missing"] == [] and g["warnings"] == []
    assert g["anchored"] == {"left": True, "top": True} and g["evidence"] == {"columns": "box faces", "rows": "box faces"}
    assert len(g["columns"]) == 3 and len(g["rows"]) == 3 and g["notes"] == []
    # counted from the right / bottom: the far edges were never seen clear -> the count rests on the labels: warned
    g2 = so.resolve_grid(_grid_items(), so.parse_targets({"columns_from_right": [1], "rows_from_bottom": [1]}), edges, centres)
    assert g2["selected"] == ["C3R3"] and g2["anchored"] == {"right": False, "bottom": False} and len(g2["warnings"]) == 2


def test_outer_row_without_labels_is_inserted_from_the_box_faces():
    """The top row's labels were all missed, but its boxes were fitted (centres above the next row) and its top edge was
    seen clear: the row is kept as an empty slot, so the targets do not shift (rows 2-3 stay R4 / R3)."""
    t = so.parse_targets(SPEC)
    edges, centres = _evidence((5, 4, 3), top_clear={5: True, 4: False, 3: False})
    g = so.resolve_grid(_grid_items(skip={(c, 5) for c in (1, 2, 3)}), t, edges, centres)
    assert sorted(g["selected"]) == ["C1R3", "C1R4", "C2R3", "C2R4"] and g["warnings"] == [] and g["anchored"]["top"]
    assert g["rows"][0]["inserted"] and g["rows"][0]["keys"] == [] and "at the top edge" in g["notes"][0]


def test_outer_row_unseen_is_never_counted_silently():
    """Neither the top row's labels nor its boxes were localised, and the next row's top edge had a detected box beyond it
    (the unresolved top row): no wall edge, the count from the top rests on the labels -> warning (no success claim)."""
    t = so.parse_targets(SPEC)
    edges, centres = _evidence((4, 3), top_clear={4: False, 3: False})
    g = so.resolve_grid(_grid_items(rows=(4, 3)), t, edges, centres)
    assert g["anchored"]["top"] is False and any("top edge" in w for w in g["warnings"])
    assert g["evidence"]["rows"].startswith("labels")


def test_interior_row_missed_becomes_an_empty_slot_and_a_late_label_fills_it():
    t = so.parse_targets(SPEC)
    edges, centres = _evidence((5, 4, 3, 2), top_clear={5: True, 4: False, 3: False, 2: False})
    items = _grid_items(rows=(5, 4, 3, 2), skip={(c, 3) for c in (1, 2, 3)})
    g = so.resolve_grid(items, t, edges, centres)
    # R5 = 1, R4 = 2, the missed R3 = 3 (inserted between the R4 and R2 boxes), R2 = 4: never renumbered
    assert [len(r["keys"]) for r in g["rows"]] == [3, 3, 0, 3] and g["rows"][2]["inserted"]
    assert sorted(g["selected"]) == ["C1R4", "C2R4"] and g["missing"] == [{"column": 1, "row": 3}, {"column": 2, "row": 3}]
    late = next(it for it in _grid_items(rows=(3,)) if it["key"] == "C1R3")
    cell = so.grid_cell(late, g)
    assert cell == {"column": 1, "row": 3} and so.cell_targeted(cell, t) and so.grid_conflict(late, cell, g, t) is None


def test_labels_only_count_is_warned_and_conflicts_are_found():
    """Without box faces seen whole (DINO grouping boxes), a fully missed interior row cannot be excluded: warned."""
    t = so.parse_targets(SPEC)
    edges, centres = _evidence((5, 4, 3))
    g = so.resolve_grid(_grid_items(faces=False), t, edges, centres)
    assert g["evidence"] == {"columns": "labels (box faces not resolved where the warnings say)",
                             "rows": "labels (box faces not resolved where the warnings say)"}
    assert any("cannot be excluded" in w for w in g["warnings"]) and sorted(g["selected"]) == ["C1R3", "C1R4", "C2R3", "C2R4"]
    # a label found later above the top row of a label-only grid: on the counted side, held by no row -> a conflict
    late = {"key": "x", "y_lo": 0.45, "y_hi": 0.55, "z_lo": 1.4, "z_hi": 1.44, "z": 1.42}
    cell = so.grid_cell(late, g)
    assert cell["row"] is None and so.grid_conflict(late, cell, g, t) is not None
    low = {"key": "y", "y_lo": 0.45, "y_hi": 0.55, "z_lo": 0.2, "z_hi": 0.24, "z": 0.22}     # beyond the requested rows
    assert so.grid_conflict(low, so.grid_cell(low, g), g, t) is None


# ------------------------------------------------------------------ plan_scans with a target spec
def _setup(monkeypatch, rows=(5, 4, 3), skip=(), tall_top=False):
    """Synthetic head frame (camera at odom (0.5, -0.2, 1.2) looking along +x, f 200 px) on a wall at x = 1.5: one box
    detection and fitted face per box, one label at the top of each box. tall_top: the top row's boxes reach above the
    image (their top edge is never seen)."""
    tb = bare_toolbox(FakeRobot(), variant="g1_v2")
    tb.set_scan_targets(SPEC)
    h = _head_frame()
    T_odom_cam = h["T_odom_root"] @ h["T_root_cam"]
    x = 1.5
    labels = {"live": [(f"C{ci + 1}R{r}", y, TOPS[r] - 0.025) for ci, y in enumerate(COLS3) for r in rows
                       if (ci + 1, r) not in skip]}

    def px(p):
        uv, _ = geo.project(h["K"], np.linalg.inv(T_odom_cam), np.atleast_2d(p))
        return uv

    faces = {}
    for ci, y in enumerate(COLS3):
        for r in rows:
            top = 2.6 if (tall_top and r == rows[0]) else TOPS[r]
            bot = TOPS[r] - 0.25
            C = np.array([[x, y + 0.1, top], [x, y - 0.1, top], [x, y - 0.1, bot], [x, y + 0.1, bot]])
            faces[f"b{ci + 1}{r}"] = {"center": C.mean(axis=0), "normal": np.array([-1.0, 0, 0]), "right": np.array([0, -1.0, 0]),
                                      "up": np.array([0, 0, 1.0]), "corners": C, "size": [0.2, top - bot], "t": 3.0}

    def bbox(f):
        uv = px(f["corners"])
        return [max(0.0, uv[:, 0].min()), max(0.0, uv[:, 1].min()), min(640.0, uv[:, 0].max()), min(480.0, uv[:, 1].max())]

    def fake_detect_boxes(prompt="cardboard box", view="auto"):
        tb.head = h
        tb.ws.boxes = {k: tbmod.BoxEstimate(k, bbox(f), 0.9, 3.0) for k, f in faces.items()}
        return {"status": "ok", "data": {}, "reason": ""}

    def fake_fit(box_id):
        tb.ws.boxes[box_id].face = dict(faces[box_id])
        return {"status": "ok", "data": {}, "reason": ""}

    def fake_rectified(frame, f):
        out = []
        C = np.asarray(f["corners"])
        for name, y, z in labels["live"]:
            if not (C[:, 1].min() < y < C[:, 1].max() and C[:, 2].min() < z < C[:, 2].max()):
                continue
            q3 = np.array([[x, y + sy * 0.05, z + sz * 0.02] for sy, sz in ((1, 1), (-1, 1), (-1, -1), (1, -1))])
            q = px(q3)
            rec = {"name": name, "kind": "label", "corners": q, "center": q.mean(axis=0), "score": 0.6}
            out.append((rec, {"center": q3.mean(axis=0), "bar_dir": np.array([0, -1.0, 0]), "corners": q3, "width": 0.10}))
        return out

    monkeypatch.setattr(tb, "detect_boxes", fake_detect_boxes)
    monkeypatch.setattr(tb, "_fit_one", fake_fit)
    monkeypatch.setattr(tb, "_labels_on_rectified_face", fake_rectified)
    tb.det = None
    return tb, labels


def _names(tb, ids):
    return [f"C{round((0.6 - tb.ws.barcodes[b].center[1]) / 0.2 + 0.5)}R{[r for r, t in TOPS.items() if abs(t - 0.025 - tb.ws.barcodes[b].center[2]) < 1e-6][0]}"
            for b in ids]


def test_plan_scans_plans_only_the_requested_cells_in_row_order(monkeypatch):
    tb, _ = _setup(monkeypatch)
    r = tb.plan_scans()
    assert r["status"] == "ok" and r["data"]["n"] == 4, r
    pend = [b for b in tb.ws.scan_plan if tb.ws.barcodes[b].status == "pending"]
    assert _names(tb, pend) == ["C1R4", "C2R4", "C1R3", "C2R3"]                 # rows from the top, left to right
    assert sum(tb.ws.barcodes[b].status == "not_targeted" for b in tb.ws.scan_plan) == 5
    t = r["data"]["targets"]
    assert t["missing_cells"] == [] and t["edges_seen"] == {"left": True, "top": True} and "WARNING" not in r["reason"]
    assert (t["n_columns"], t["n_rows"]) == (3, 3) and t["evidence"] == {"columns": "box faces", "rows": "box faces"}
    # the budget extends by the 4 targets only; next_barcode walks the targets only
    h = _harness(tb)
    base = h.base_budget["max_tool_calls"]
    h._after("plan_scans", {}, r)
    assert h.budget["max_tool_calls"] == base + 4 * h.base_budget["per_barcode"]["max_tool_calls"]
    tb._plan_update_from_here = lambda: None
    assert _names(tb, [tb.next_barcode()["data"]["barcode_id"]]) == ["C1R4"]
    # finish: the not_targeted labels do not count
    for b in pend:
        tb.ws.barcodes[b].status = "scanned"
    assert h._task3_work_left() is None
    h.call("finish", {"task": 3, "success": True, "summary": "x"})
    assert h.finished["success"] is True and h.finished["missing_cells"] == [] and h.finished["grid_warnings"] == []


def test_unseen_top_edge_is_warned_and_refuses_success(monkeypatch):
    tb, _ = _setup(monkeypatch, tall_top=True)          # the top row reaches above the image: its top edge never seen
    r = tb.plan_scans()
    assert r["status"] == "ok" and "WARNING" in r["reason"] and r["data"]["targets"]["edges_seen"]["top"] is False
    h = _harness(tb)
    for b in tb.ws.scan_plan:
        if tb.ws.barcodes[b].status == "pending":
            tb.ws.barcodes[b].status = "scanned"
    assert "could not be verified" in h._task3_work_left()
    assert h.call("finish", {"task": 3, "success": True, "summary": "x"})["status"] == "fail"   # refused claim


def test_interior_row_missed_in_plan_scans_and_a_later_label_fills_the_cell(monkeypatch):
    tb, labels = _setup(monkeypatch, rows=(5, 4, 3, 2), skip={(c, 3) for c in (1, 2, 3)})
    r = tb.plan_scans()
    assert r["data"]["n"] == 2 and r["data"]["targets"]["missing_cells"] == [{"column": 1, "row": 3}, {"column": 2, "row": 3}]
    assert "NOT FOUND: column 1, row 3; column 2, row 3" in r["reason"] and "without a localised label" in r["reason"]
    h = _harness(tb)
    saved = {b: tb.ws.barcodes[b].status for b in tb.ws.scan_plan}
    for b in tb.ws.scan_plan:
        if saved[b] == "pending":
            tb.ws.barcodes[b].status = "scanned"
    assert "requested target cell" in h._task3_work_left()
    for b, st in saved.items():
        tb.ws.barcodes[b].status = st
    # a plan update from a closer view finds C1R3 / C2R3: targets in their row-3 cells, ordered after row 2
    labels["live"] += [(f"C{ci + 1}R3", y, TOPS[3] - 0.025) for ci, y in enumerate(COLS3)]
    u = tb.plan_scans()
    added = {a["cell"]["column"]: a["status"] for a in u["data"]["added"]}
    assert added == {1: "pending", 2: "pending", 3: "not_targeted"} and tb.ws.missing_cells() == []
    pend = [b for b in tb.ws.scan_plan if tb.ws.barcodes[b].status == "pending"]
    assert _names(tb, pend) == ["C1R4", "C2R4", "C1R3", "C2R3"] and tb.ws.grid_warnings() == []


def test_extra_looks_cover_every_standing_view_before_the_grid_is_resolved(monkeypatch):
    """Before the grid of a side is resolved, plan_scans looks from every calibrated standing head view (the top row may
    lie above the plan view); afterwards only the finer views (the default rule)."""
    tb = bare_toolbox(FakeRobot(), variant="g1_v2")
    tb.set_scan_targets(SPEC)
    tb.body_cmd = {"height_default": 0.74, "height_range": [0.3, 0.8], "rpy_default": [0, 0, 0]}
    tb.info = {"head_K": [[200.0, 0, 320], [0, 200.0, 240], [0, 0, 1]], "head_size": [480, 640]}
    levels = [{"pitch_cmd": 0.0}, {"pitch_cmd": -0.1}, {"pitch_cmd": -0.2}]
    monkeypatch.setattr(tb, "_lean_levels", lambda: levels)
    monkeypatch.setattr(tb, "_body", lambda: type("B", (), {"levels": [{"height_cmd": 0.74}]})())
    monkeypatch.setattr(tb, "_cam_T_at", lambda x, y, yaw, lv: np.eye(4))          # sees nothing: never "finer"
    monkeypatch.setattr(tb, "_q", lambda: None)
    looked = []
    monkeypatch.setattr(tb, "_set_view", lambda lv: (looked.append(lv["pitch_cmd"]), setattr(tb, "view_pitch", lv["pitch_cmd"]),
                                                     {"status": "ok"})[2])
    monkeypatch.setattr(tb, "plan_scans", lambda _update=None: {"status": "ok", "data": {"added": []}, "reason": ""})
    monkeypatch.setattr(tb, "_upright", lambda: None)
    f = {"corners": np.array([[1.5, 0.6, 1.25], [1.5, -0.1, 1.25], [1.5, -0.1, 0.5], [1.5, 0.6, 0.5]]), "normal": np.array([-1.0, 0, 0])}
    tb.ws.boxes = {"box0": tbmod.BoxEstimate("box0", [0, 0, 1, 1], 0.9, 3.0, face=f)}
    tb.ws.scan_plan = ["bc0"]
    tb.head = _head_frame()
    tb.ws.target_grid = {"targets": tb.scan_targets, "edges_seen": [], "faces": {}}
    low = tb._low_view_update()
    assert looked == [-0.1, -0.2] and [d["lean"] for d in low["looks"]] == [-0.1, -0.2]
    looked.clear(); tb.view_pitch = 0.0
    tb.ws.target_grid["faces"][0] = {"missing": []}                              # resolved: the default rule again
    tb._low_view_update()
    assert looked == []


def test_rule_task3_with_targets_stays_near_the_start(monkeypatch):
    """Rule orchestrator: with a target spec phase 1 is detect + select all + plan_scans (no observation pose walk)."""
    from g1agent.orchestrators import rule
    tb = bare_toolbox(FakeRobot(), variant="g1_v2")
    tb.set_scan_targets(SPEC)
    calls = []
    h = _harness(tb)
    real = h.call

    def call(name, args=None):
        calls.append(name)
        if name in ("detect_boxes", "select_target_box"):
            return {"status": "ok", "data": {}, "reason": ""}
        if name == "plan_scans":
            return {"status": "ok", "data": {"n": 0}, "reason": ""}
        if name == "next_barcode":
            return {"status": "ok", "data": {"done": True}, "reason": ""}
        return real(name, args)

    monkeypatch.setattr(h, "call", call)
    tb.remember_home = lambda: None
    rule.Task3(h, config.load_all()["task3"], "all").run()
    assert calls[:3] == ["detect_boxes", "select_target_box", "plan_scans"]
    assert "compute_obs_pose" not in calls and "approach" not in calls
