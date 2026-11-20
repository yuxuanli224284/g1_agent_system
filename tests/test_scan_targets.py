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


def test_plan_diagnostics_are_logged_but_never_shown_to_the_llm(monkeypatch):
    """D11 diagnostics (label width, detector kind / score / cv2 symbol quad per entry, dropped duplicates, the slot
    table) go to the tool log (the "diagnostics" key, kept by the Harness log record) and never into the LLM's view of
    the result (tool_result_text): the plan entries and the targets report the LLM sees have the fields of the 4-box
    tag g1v2-sub4-run1 (review 2026-11-19: +~50 % plan_scans text, kept for the whole agent_full episode)."""
    from g1agent.orchestrators.common import tool_result_text
    tb, labels = _setup(monkeypatch)
    name, y, z = labels["live"][0]
    labels["live"].append((name + "_dup", y + 0.01, z))           # a second detection of the C1R5 label: dropped
    h = _harness(tb)
    r = h.call("plan_scans", {})
    assert r["status"] == "ok"
    tag_entry = {"id", "box_id", "order_index", "corners_px", "center_odom", "width_m", "standoff_m", "status", "face_id",
                 "cell"}
    tag_targets = {"description", "spec", "order", "n_columns", "n_rows", "requested_cells", "missing_cells", "edges_seen",
                   "evidence", "warnings", "notes", "note"}
    assert all(set(e) == tag_entry for e in r["data"]["barcodes"]) and set(r["data"]["targets"]) == tag_targets
    text = tool_result_text(r)
    for k in ("diagnostics", "dropped_labels", "label_width_m", "symbol_corners_px", "refined_by_cv2", "slots"):
        assert f'"{k}"' not in text
    d = r["diagnostics"]
    assert len(d["plan_entries"]) == len(r["data"]["barcodes"]) and d["plan_entries"][0]["kind"] == "label"
    assert [x["rule"] for x in d["dropped_labels"]] == ["same label on this face (higher score kept)"]
    assert set(d["target_grid"]["slots"]) == {"columns", "rows"}
    assert h.log.tools[-1]["result"]["diagnostics"] == d              # the Harness log record keeps them


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


# ------------------------------------------------------------------ scan-room rows 2-4 from the bottom x all columns
# (design D2 "all", D3 floor bound, D4 box-based slots and cell order). A 6 x 5 wall: columns y 0.5 .. -0.5 (box 0.2 m
# wide, C1 = left = +y), box rows R1..R5 with bottoms 0 / 0.25 / ... (0.25 m tall, R1 on the floor); one label per box
# near its top. `lift` raises the whole wall (boxes on a table).
SPEC_R234 = {"columns": "all", "rows_from_bottom": [2, 3, 4], "order": "rows"}
COLS6 = (0.5, 0.3, 0.1, -0.1, -0.3, -0.5)


def _wall_face(c, r, lift=0.0, seen=True):
    y, bot = COLS6[c - 1], lift + 0.25 * (r - 1)
    return {"key": ("wall", c, r), "y_lo": y - 0.1, "y_hi": y + 0.1, "z_lo": bot, "z_hi": bot + 0.25,
            "seen": {e: seen for e in ("top", "bottom", "left", "right")}}


def _wall_label(key, c, r, z, h=0.04, w=0.1, lift=0.0, face_seen=True):
    y = COLS6[c - 1]
    return {"key": key, "y_lo": y - w / 2, "y_hi": y + w / 2, "z_lo": z - h / 2, "z_hi": z + h / 2, "z": z,
            "face": _wall_face(c, r, lift, face_seen)}


def _wall_items(rows=(1, 2, 3, 4), cols=range(1, 7), skip=(), lift=0.0, drop=0.025, unseen_rows=()):
    """One label per box, its centre `drop` below the box top; the faces of `unseen_rows` were not seen whole."""
    return [_wall_label(f"C{c}R{r}", c, r, lift + 0.25 * r - drop, lift=lift, face_seen=r not in unseen_rows)
            for r in rows for c in cols if (c, r) not in skip]


def _wall_evidence(rows=(1, 2, 3, 4), cols=range(1, 7), lift=0.0, bottom_clear=False, right_clear=True):
    """Edge observations: the bottom edges of the lowest row's faces (not clear by default: a floor patch detected as a
    box below the wall), the left edges of the C1 faces and the right edges of the rightmost column's faces; and the
    fitted face centres."""
    lo_r, cols = min(rows), list(cols)
    edges = [{"edge": "bottom", "coord": lift + 0.25 * (lo_r - 1), "clear": bottom_clear} for _ in cols]
    edges += [{"edge": "left", "coord": COLS6[0] + 0.1, "clear": True} for _ in rows]
    edges += [{"edge": "right", "coord": COLS6[cols[-1] - 1] - 0.1, "clear": right_clear} for _ in rows]
    centres = {"columns": [COLS6[c - 1] for c in cols for _ in rows],
               "rows": [lift + 0.25 * (r - 1) + 0.125 for _ in cols for r in rows]}
    return edges, centres


FLOOR = {"bottom": 0.0}


def test_r234_parse_all_columns_and_describe():
    t = so.parse_targets(SPEC_R234)
    assert t["columns"] == {"from": "left", "index": None, "all": True} and t["rows"] == {"from": "bottom", "index": [2, 3, 4]}
    assert t["order"] == "rows"
    assert so.describe_targets(t) == "the labels of the boxes in all columns and rows 2, 3, 4 from the bottom of the box wall"
    assert so.parse_targets({"rows": "all"})["rows"] == {"from": "top", "index": None, "all": True}
    for bad in ({"columns": "all", "columns_from_left": [1]}, {"columns": "every"}, {"rows": "all", "rows_from_top": [1]},
                {"columns": ["all"]}, {"columns": True}):
        with pytest.raises(ValueError):
            so.parse_targets(bad)


def test_r234_bottom_edge_anchored_by_the_floor_bound():
    """The bottom-edge observation is not clear (a floor detection below it), but it lies on the floor (odom z 0): no box
    row fits between them, so it is the wall's bottom edge; R2-R4 are rows 2-4 from the bottom."""
    t = so.parse_targets(SPEC_R234)
    edges, centres = _wall_evidence()
    g = so.resolve_grid(_wall_items(), t, edges, centres, bounds=FLOOR)
    assert g["anchored"] == {"left": True, "right": True, "bottom": True} and g["warnings"] == []
    assert [len(r["keys"]) for r in g["rows"]] == [6, 6, 6, 6] and all(r["box"] for r in g["rows"])
    assert all(g["cell_of"][f"C{c}R{r}"] == {"column": c, "row": r} for c in range(1, 7) for r in (1, 2, 3, 4))
    assert len(g["requested"]) == 18 and g["missing"] == [] and len(g["selected"]) == 18
    assert g["evidence"] == {"columns": "box faces", "rows": "box faces"}
    # without the bound the same unclear edge does not count: warned (no success claim)
    g0 = so.resolve_grid(_wall_items(), t, edges, centres)
    assert g0["anchored"]["bottom"] is False and any("bottom edge" in w for w in g0["warnings"])


def test_r234_floor_bound_never_anchors_boxes_on_a_table():
    """The same wall 0.75 m higher (on a table), bottom edge not clear: three box rows would fit between the edge and the
    floor, so the floor does not anchor it; the count is warned."""
    t = so.parse_targets(SPEC_R234)
    edges, centres = _wall_evidence(lift=0.75)
    g = so.resolve_grid(_wall_items(lift=0.75), t, edges, centres, bounds=FLOOR)
    assert g["anchored"]["bottom"] is False and any("bottom edge" in w for w in g["warnings"])
    assert g["evidence"]["rows"].startswith("labels")


def test_r234_all_columns_requests_every_cell_and_reports_missing_ones():
    t = so.parse_targets(SPEC_R234)
    edges, centres = _wall_evidence()
    g = so.resolve_grid(_wall_items(skip={(6, 3)}), t, edges, centres, bounds=FLOOR)
    assert len(g["requested"]) == 18 and g["missing"] == [{"column": 6, "row": 3}] and g["warnings"] == []
    assert so.cell_targeted({"column": 6, "row": 2}, t) and not so.cell_targeted({"column": None, "row": 2}, t)
    assert not so.cell_targeted({"column": 3, "row": 1}, t)
    # a late label in the missing cell fills it; one beyond the right edge (no column holds it) is a conflict
    late = _wall_label("late", 6, 3, 0.725)
    cell = so.grid_cell(late, g)
    assert cell == {"column": 6, "row": 3} and so.grid_conflict(late, cell, g, t) is None
    out = {"key": "out", "y_lo": -0.75, "y_hi": -0.65, "z_lo": 0.705, "z_hi": 0.745, "z": 0.725}
    cell = so.grid_cell(out, g)
    assert cell["column"] is None and "beyond the columns counted" in so.grid_conflict(out, cell, g, t)


def test_r234_unseen_right_edge_is_warned_where_the_old_spec_passed():
    """Only columns 1-5 were localised and the right edge of C5 had a detected box beyond it (C6). With "all columns"
    the right edge is unanchored: warned, success refused. Regression for the hole: the same input with no column key
    (the old way to say "every column") returned no missing cell and no warning."""
    edges, centres = _wall_evidence(cols=range(1, 6), right_clear=False)
    items = _wall_items(cols=range(1, 6))
    g = so.resolve_grid(items, so.parse_targets(SPEC_R234), edges, centres, bounds=FLOOR)
    assert g["anchored"]["right"] is False and any("right edge" in w for w in g["warnings"])
    old = so.resolve_grid(items, so.parse_targets({"rows_from_bottom": [2, 3, 4], "order": "rows"}), edges, centres,
                          bounds=FLOOR)
    assert old["missing"] == [] and old["warnings"] == []


def test_r234_all_columns_inserts_label_less_columns_at_both_edges():
    """C1 and C6 carry no localised label but both side edges were seen clear: one empty column is inserted at each end
    from the box faces, so the requested cells of C1 / C6 are reported missing (never silently fewer columns)."""
    t = so.parse_targets(SPEC_R234)
    edges, centres = _wall_evidence()
    g = so.resolve_grid(_wall_items(cols=range(2, 6)), t, edges, centres, bounds=FLOOR)
    assert [c["inserted"] for c in g["columns"]] == [True, False, False, False, False, True] and g["warnings"] == []
    assert len(g["requested"]) == 18 and sorted({c["column"] for c in g["missing"]}) == [1, 6]


def test_r234_label_less_box_faces_are_box_evidence():
    """Row R3 has no localised label, but its faces were fitted and seen whole (passed as faces): R3 is an observed box
    row (not inserted), and the R2 / R4 indices do not depend on it."""
    t = so.parse_targets(SPEC_R234)
    edges, centres = _wall_evidence()
    faces = [_wall_face(c, 3) for c in range(1, 7)]
    g = so.resolve_grid(_wall_items(rows=(1, 2, 4)), t, edges, centres, faces=faces, bounds=FLOOR)
    assert [r["keys"] == [] for r in g["rows"]] == [False, False, True, False] and not g["rows"][2]["inserted"]
    assert g["cell_of"]["C1R4"]["row"] == 4 and g["warnings"] == [] and len(g["missing"]) == 6
    assert any("without a localised label" in n for n in g["notes"])


def _sliver(c, r, lift=0.0):
    """A 2.5 mm tall label fragment 1 cm below the top of box (c, r) (bc14 of the 4-box run), on that box's face."""
    return _wall_label("sliver", c, r, lift + 0.25 * r - 0.01, h=0.0025, w=0.06)


def test_r234_sliver_on_a_box_face_does_not_make_a_row():
    """A fragment whose extent does not overlap the real label of its box: before D4 it formed a row of its own and its
    face (carrying two label groups) was dropped as box evidence. Now both labels go to the box slot holding their
    centres: one row, the R3 labels keep their index (counted from the bottom and from the top), no warning."""
    items = _wall_items(drop=0.035) + [_sliver(3, 4)]                # real labels [top-5.5 cm, top-1.5 cm]
    edges, centres = _wall_evidence()
    g = so.resolve_grid(items, so.parse_targets(SPEC_R234), edges, centres, bounds=FLOOR)
    assert len(g["rows"]) == 4 and g["warnings"] == [] and "sliver" in g["rows"][3]["keys"]
    assert g["cell_of"]["sliver"] == {"column": 3, "row": 4}
    assert all(g["cell_of"][f"C{c}R3"]["row"] == 3 for c in range(1, 7))
    # counted from the top (the 4-box spec form): R5 R4 R3 seen, top edge clear -> R3 is row 3, not shifted to 4
    items = _wall_items(rows=(5, 4, 3), cols=(1, 2, 3), drop=0.035) + [_sliver(3, 4)]
    edges = [{"edge": "top", "coord": 1.25, "clear": True}, {"edge": "left", "coord": 0.6, "clear": True}]
    centres = {"columns": [COLS6[c - 1] for c in (1, 2, 3)], "rows": [0.25 * r - 0.125 for r in (5, 4, 3)]}
    g = so.resolve_grid(items, so.parse_targets(SPEC), edges, centres)
    assert len(g["rows"]) == 3 and g["warnings"] == [] and g["cell_of"]["sliver"]["row"] == 2
    assert all(g["cell_of"][f"C{c}R3"]["row"] == 3 for c in (1, 2, 3))
    assert sorted(g["selected"]) == ["C1R3", "C1R4", "C2R3", "C2R4"]


def test_r234_label_outside_every_whole_box_is_a_label_only_slot_and_warned():
    """The R3 faces were never seen whole: the R3 labels lie in no box slot, they form a label-only row, and the gaps
    next to it cannot be excluded (warning, no success claim)."""
    t = so.parse_targets(SPEC_R234)
    edges, centres = _wall_evidence()
    g = so.resolve_grid(_wall_items(unseen_rows=(3,)), t, edges, centres, bounds=FLOOR)
    assert [r["box"] for r in g["rows"]] == [True, True, False, True]
    assert any("cannot be excluded" in w for w in g["warnings"])


def _extra_face(key, y_lo, y_hi, z_lo, z_hi):
    """A label-less fitted face seen whole (an extra view's fit, a part of a box, a floor patch)."""
    return {"key": ("extra", key), "y_lo": y_lo, "y_hi": y_hi, "z_lo": z_lo, "z_hi": z_hi,
            "seen": {e: True for e in ("top", "bottom", "left", "right")}}


def _r234_baseline(g):
    assert g["warnings"] == [] and len(g["selected"]) == 18 and g["missing"] == [] and len(g["rows"]) == 4
    assert all(g["cell_of"][f"C{c}R{r}"] == {"column": c, "row": r} for c in range(1, 7) for r in (1, 2, 3, 4))


def test_r234_fit_reaching_into_the_next_row_does_not_take_its_labels():
    """Review 2026-11-19: the labels sit 2.1-2.5 cm below their face top. One fit of C3R4 whose bottom reaches 3 cm into
    R3 (e.g. from a squat look) widens the R4 slot's extent over the R3 labels; they go to the slot with the nearest
    centre (R3), not to a label-only row (which shifted R3 / R4 to rows 4 / 5 and left 6 cells not targeted). The same
    rule holds for a late R3 label (grid_cell)."""
    t = so.parse_targets(SPEC_R234)
    edges, centres = _wall_evidence()
    sloppy = _extra_face("C3R4 low", 0.0, 0.2, 0.72, 1.0)
    g = so.resolve_grid(_wall_items(skip={(3, 3)}), t, edges, centres, faces=[sloppy], bounds=FLOOR)
    assert g["warnings"] == [] and len(g["rows"]) == 4 and g["rows"][3]["lo"] == pytest.approx(0.72)
    assert all(g["cell_of"][f"C{c}R3"]["row"] == 3 for c in (1, 2, 4, 5, 6))
    assert g["missing"] == [{"column": 3, "row": 3}]
    late = _wall_label("late", 3, 3, 0.725)
    cell = so.grid_cell(late, g)
    assert cell == {"column": 3, "row": 3} and so.grid_conflict(late, cell, g, t) is None
    g = so.resolve_grid(_wall_items(), t, edges, centres, faces=[sloppy], bounds=FLOOR)
    _r234_baseline(g)


@pytest.mark.parametrize("extra", [_extra_face("sliver", 0.25, 0.35, 0.30, 0.38),      # 0.08 m: under half a box
                                   _extra_face("half R2", 0.25, 0.35, 0.25, 0.375)])     # the lower half of an R2 box
def test_r234_undersized_label_less_face_never_sets_the_box_size(extra):
    """Review 2026-11-19: a label-less face seen whole that is smaller than a box (a part of a box or a floor patch
    inside the image) never sets the box size (only faces carrying a label do): the real faces stay single boxes, the
    rows keep their indices. A face under half a box is no box evidence; a half box rounds to one box and joins the
    row whose centre is within half a box."""
    t = so.parse_targets(SPEC_R234)
    edges, centres = _wall_evidence()
    g = so.resolve_grid(_wall_items(), t, edges, centres, faces=[extra], bounds=FLOOR)
    _r234_baseline(g)
    assert all(r["box"] and not r["inserted"] for r in g["rows"]) and g["notes"] == []


def test_r234_faces_that_disagree_on_one_box_are_reported():
    """A label-less fit 1.4 boxes tall whose centre lies within half a box of the R3 row joins that row, but together
    the faces span 1.6 boxes: the row is no box evidence (the R3 labels form a label-only row, warned), and a note says
    which faces were not used, so the lost evidence is visible."""
    t = so.parse_targets(SPEC_R234)
    edges, centres = _wall_evidence()
    tall = _extra_face("tall", -0.1, 0.1, 0.55, 0.90)
    g = so.resolve_grid(_wall_items(), t, edges, centres, faces=[tall], bounds=FLOOR)
    assert [r["box"] for r in g["rows"]] == [True, True, False, True]
    assert any("do not agree on one box" in n and "0.500..0.900" in n for n in g["notes"])
    assert any("cannot be excluded" in w for w in g["warnings"])


def test_r234_cell_order_ignores_duplicates_inside_a_cell():
    """Order by box cells: rows from the top, columns from the left inside a row (rule "rows"); a fragment inside C3R4
    stays in its cell (the label-extent rule alone would put it in a row of its own, first)."""
    t = so.parse_targets(SPEC_R234)
    items = _wall_items(drop=0.035) + [_sliver(3, 4)]
    edges, centres = _wall_evidence()
    g = so.resolve_grid(items, t, edges, centres, bounds=FLOOR)
    sel = [it for it in items if it["key"] in g["selected"]]
    expect = [f"C{c}R{r}" for r in (4, 3, 2) for c in range(1, 7)]
    assert so.order_items(sel, "rows")[0] == "sliver"                                  # the old effect
    order = so.order_by_cells(sel, g["cell_of"], g, "rows")
    assert [k for k in order if k != "sliver"] == expect and order.index("sliver") in (2, 3)
    cols = so.order_by_cells(sel, g["cell_of"], g, "columns")
    assert [k for k in cols if k != "sliver"] == [f"C{c}R{r}" for c in range(1, 7) for r in (4, 3, 2)]
    # rows counted from the top give the same physical order
    g_top = {"columns": g["columns"], "rows": [{**s, "index": 5 - s["index"]} for s in g["rows"]]}
    cells_top = {k: {"column": c["column"], "row": 5 - c["row"]} for k, c in g["cell_of"].items()}
    assert so.order_by_cells(sel, cells_top, g_top, "rows") == order
    # a label without a cell (found late, held by no slot): the label rule
    cells = dict(g["cell_of"], sliver={"column": 3, "row": None})
    assert so.order_by_cells(sel, cells, g, "rows") == so.order_items(sel, "rows")
    assert so.order_by_cells(sel, g["cell_of"], None, "rows") == so.order_items(sel, "rows")


# ------------------------------------------------------------------ scan-room rows 2-4: head views, phase 1, stances, prompts
def test_upright_head_views_take_no_lean_look_before_the_grid_is_resolved(monkeypatch):
    """head_views "upright" with the REAL _lean_levels (g1_v2 calibration, d435i lean views on): the extra looks before the
    grid is resolved take no lean-back look (the default rule above takes every calibrated lean level)."""
    tb = bare_toolbox(FakeRobot(), variant="g1_v2")
    tb.set_scan_targets(SPEC)
    tb.set_head_views("upright")
    tb.body_cmd = {"height_default": 0.74, "height_range": [0.3, 0.8], "rpy_default": [0, 0, 0]}
    tb.info = {"head_K": [[200.0, 0, 320], [0, 200.0, 240], [0, 0, 1]], "head_size": [480, 640]}
    assert tb.cfg["robot"]["lean_views"] and len(tb._lean_levels()) == 1
    monkeypatch.setattr(tb, "_body", lambda: type("B", (), {"levels": [{"height_cmd": 0.74}]})())
    monkeypatch.setattr(tb, "_q", lambda: None)
    looked = []
    monkeypatch.setattr(tb, "_set_view", lambda lv: (looked.append(lv["pitch_cmd"]), {"status": "ok"})[1])
    monkeypatch.setattr(tb, "plan_scans", lambda _update=None: {"status": "ok", "data": {"added": []}, "reason": ""})
    f = {"corners": np.array([[1.5, 0.6, 1.25], [1.5, -0.1, 1.25], [1.5, -0.1, 0.5], [1.5, 0.6, 0.5]]), "normal": np.array([-1.0, 0, 0])}
    tb.ws.boxes = {"box0": tbmod.BoxEstimate("box0", [0, 0, 1, 1], 0.9, 3.0, face=f)}
    tb.ws.scan_plan = ["bc0"]
    tb.head = _head_frame()
    tb.ws.target_grid = {"targets": tb.scan_targets, "edges_seen": [], "faces": {}}
    low = tb._low_view_update()
    assert looked == [] and [d for d in (low or {"looks": []})["looks"] if "lean" in d] == []


def _rule_calls(monkeypatch, tb):
    from g1agent.orchestrators import rule
    calls = []
    h = _harness(tb)
    real = h.call

    def call(name, args=None):
        calls.append(name)
        if name in ("detect_boxes", "select_target_box", "fit_front_face", "compute_obs_pose"):
            return {"status": "ok", "data": {}, "reason": ""}
        if name == "approach":
            return {"status": "partial", "data": {"walk_done": True}, "reason": "walk finished"}
        if name == "plan_scans":
            return {"status": "ok", "data": {"n": 0}, "reason": ""}
        if name == "next_barcode":
            return {"status": "ok", "data": {"done": True}, "reason": ""}
        return real(name, args)

    monkeypatch.setattr(h, "call", call)
    tb.remember_home = lambda: None
    rule.Task3(h, config.load_all()["task3"], "all").run()
    return calls


@pytest.mark.parametrize("head_views", [None, "upright"])
def test_rule_task3_phase1_view(monkeypatch, head_views):
    """Rule orchestrator, target spec: phase1_view "obs_pose" runs the task-1 stage (detect -> select all -> fit ->
    compute_obs_pose -> approach) before plan_scans; the 4-box default (start_pose) never calls compute_obs_pose, also
    with upright head views."""
    tb = bare_toolbox(FakeRobot(), variant="g1_v2")
    tb.set_scan_targets(SPEC)
    tb.set_head_views(head_views)
    calls = _rule_calls(monkeypatch, tb)
    assert calls[:3] == ["detect_boxes", "select_target_box", "plan_scans"] and "compute_obs_pose" not in calls
    tb = bare_toolbox(FakeRobot(), variant="g1_v2")
    tb.set_scan_targets({"columns": "all", "rows_from_bottom": [2, 3, 4], "order": "rows"})
    tb.set_head_views(head_views)
    tb.set_phase1_view("obs_pose")
    calls = _rule_calls(monkeypatch, tb)
    assert calls[:6] == ["detect_boxes", "select_target_box", "fit_front_face", "compute_obs_pose", "approach", "plan_scans"]


def test_failed_stance_is_not_planned_again_for_the_same_target(monkeypatch):
    """D8: after check_wrist_placement fails with "re-plan the stance" at a reached stance, plan_stance does not return
    that stance again for the same barcode and the unchanged wrist target; a changed target may use it again; with every
    feasible stance failed, plan_stance fails (an attempt, not "unreachable")."""
    from test_task3_logic import plan_of
    tb = bare_toolbox(FakeRobot(), variant="g1_v2")
    plan_of(tb, 1)
    bc = tb.ws.barcodes["bc0"]
    bc.face["corners"] = np.array([[1.0, 0.4, 1.2], [1.0, -0.4, 1.2], [1.0, -0.4, 0.8], [1.0, 0.4, 0.8]])
    tb.next_barcode()
    tb.compute_wrist_target()
    tb.ik = type("IK", (), {"cfg": {"restarts": 6, "plan_stall_window": 100, "plan_stall_eps_m": 0.0005, "seed": 0},
                            "roll_error_deg": lambda *a: 0.0})()
    monkeypatch.setattr(tb, "_q", lambda: None)
    S = tb._c2()["stance"]
    # feasible: the nearest distance row, the two outermost lateral offsets on the robot's right (everything else: IK)
    feasible = lambda p: abs(p[0] - (bc.center[0] - S["min_face_dist_m"])) < 1e-9 and \
        p[1] < bc.center[1] + S["lateral_range"][0] + 1.5 * S["step"]
    monkeypatch.setattr(tb, "_eval_stance_jobs", lambda jobs, body: [
        {"status": "ok", "sols": [], "gap": 0.1, "margin": 0.3, "man": 0.01} if feasible(j["Tc"][:2, 3]) else {"status": "ik"}
        for j in jobs])
    monkeypatch.setattr(tb, "_stance_robust", lambda *a: True)
    monkeypatch.setattr(tb, "_label_view", lambda t, off, w=None: {"in_view": True})
    ident = lambda st: (st["face_dist"], st["lateral_offset"], st["level_index"])

    def reach_and_fail_placement():
        """The robot reached the planned stance; move_arm('final') left the scanner 3 cm behind the stand-off."""
        st = tb.ws.stance_target
        tb._stance_reached = st
        tb._last_arm = {"pose": "final", "barcode_id": "bc0", "status": "partial"}
        t = tb.ws.wrist_target
        T = np.eye(4)
        T[:3, 0] = -np.asarray(bc.normal)
        T[:3, 3] = bc.center + (t["offset"] + 0.03) * np.asarray(bc.normal)
        tb.kin = type("K", (), {"pose": lambda self, q, frame: T})()
        r = tb.check_wrist_placement()
        assert r["status"] == "fail" and "re-plan the stance" in r["reason"] and r["data"]["failed_stance"]

    assert tb.plan_stance()["status"] == "ok"
    a = ident(tb.ws.stance_target)
    reach_and_fail_placement()
    assert tb.plan_stance()["status"] == "ok"
    b = ident(tb.ws.stance_target)
    assert b != a
    pos0 = np.array(tb.ws.wrist_target["pos"], float)
    tb.ws.wrist_target["pos"] = pos0 + np.array([0.0, 0.0, 0.001])     # a changed estimate: a new problem
    assert tb.plan_stance()["status"] == "ok" and ident(tb.ws.stance_target) == a
    tb.ws.wrist_target["pos"] = pos0
    assert tb.plan_stance()["status"] == "ok" and ident(tb.ws.stance_target) == b
    reach_and_fail_placement()
    r = tb.plan_stance()
    assert r["status"] == "fail" and "every stance tried for this target failed placement" in r["reason"]
    assert r["data"]["rejected"]["failed_placement"] == 2 and bc.status == "pending" and "bc0" in tb._stance_attempted


SPEC_R234_TASK = {"columns": "all", "rows_from_bottom": [2, 3, 4], "order": "rows"}


def _scanroom_obs_toolbox():
    """g1_v2 kinematics, the D435i colour intrinsics (848 x 480, 69 deg), upright head views, obs_pose phase 1, the
    rows 2-4 x all columns spec; the robot root at the odom origin facing +x, 0.60 m from the 6 x 5 wall (boxes 0.20 m
    wide x 0.25 m tall, columns y +0.5 .. -0.5, the floor at odom z 0)."""
    from g1agent.kinematics.fk import Kinematics
    tb = bare_toolbox(FakeRobot(), variant="g1_v2")
    tb.kin = Kinematics(tb.cfg["robot"]["urdf"])
    fx = 424.0 / np.tan(np.radians(69.0) / 2)
    tb.info = {"head_K": [[fx, 0, 424.0], [0, fx, 240.0], [0, 0, 1]], "head_size": [480, 848]}
    tb.body_cmd = {"height_default": 0.74, "height_range": [0.3, 0.8], "rpy_default": [0, 0, 0]}
    q0 = tb.kin.q_from_state([], [])
    tb._q = lambda: q0.copy()
    tb.set_scan_targets(SPEC_R234_TASK)
    tb.set_head_views("upright")
    tb.set_phase1_view("obs_pose")
    return tb


def _fit_visible_faces(tb, x_root, wall_x=0.60):
    """What detect_boxes + fit_front_face give from root (x_root, 0, yaw 0) in the upright view: per box face, the
    rectangle of its points inside the head image (a face cut by the image border ends there)."""
    up = tb._lean_levels()[0]
    Tc = tb._cam_T_at(x_root, 0.0, 0.0, up)
    tb.head = {"rgb": np.zeros((480, 848, 3), np.uint8), "K": np.asarray(tb.info["head_K"], float), "T_root_cam": Tc,
               "T_odom_root": np.eye(4), "t": 1.0}
    tb.ws.boxes = {}
    yy, zz = np.meshgrid(np.linspace(0, 1, 41), np.linspace(0, 1, 41))
    for c, y in enumerate((0.5, 0.3, 0.1, -0.1, -0.3, -0.5)):
        for r in range(1, 6):
            P = np.c_[np.full(yy.size, wall_x), y + 0.1 - 0.2 * yy.ravel(), 0.25 * (r - 1) + 0.25 * zz.ravel()]
            uv, zc = geo.project(tb.head["K"], np.linalg.inv(Tc), P)
            inside = geo.in_image(uv, zc, 848, 480, 0.0)          # the fit reaches the image border (no margin)
            if not inside.any():
                continue
            V = P[inside]
            y0, y1, z0, z1 = V[:, 1].min(), V[:, 1].max(), V[:, 2].min(), V[:, 2].max()
            C = np.array([[wall_x, y1, z1], [wall_x, y0, z1], [wall_x, y0, z0], [wall_x, y1, z0]])
            face = {"center": C.mean(axis=0), "normal": np.array([-1.0, 0, 0]), "right": np.array([0, -1.0, 0]),
                    "up": np.array([0, 0, 1.0]), "corners": C, "size": [y1 - y0, z1 - z0], "t": 1.0}
            tb.ws.boxes[f"C{c + 1}R{r}"] = tbmod.BoxEstimate(f"C{c + 1}R{r}", [0, 0, 1, 1], 0.9, 1.0, face=face)
    tb.ws.target_selector = tb.ws.target_box_id = "all"


def test_obs_pose_from_the_spec_sees_the_floor_line_rows_and_side_edges():
    """D9 with the real g1_v2 kinematics and upright calibration: from the 0.60 m start the region is the fitted wall
    part from the floor (odom z 0) up to row 4 (unit = the smallest face seen whole, 0.25 m); compute_obs_pose picks the
    nearest upright pose that holds it (design estimate 1.20-1.35 m from the wall), and re-planning from there (the
    region grows with the faces seen) converges inside that window with every region corner in view."""
    tb = _scanroom_obs_toolbox()
    _fit_visible_faces(tb, 0.0)
    r = tb.compute_obs_pose("all")
    assert r["status"] == "ok", r
    sv = r["data"]["spec_view"]
    assert sv["unit"] == pytest.approx(0.25, abs=1e-6) and sv["z_range"] == pytest.approx([0.0, 1.0], abs=1e-6)
    ds = [r["data"]["root_target_odom"]["d_obs"]]
    for _ in range(8):                                   # approach: walk there, re-perceive, re-plan (max_segments 8)
        rt = tb.ws.root_target
        _fit_visible_faces(tb, rt["pos"][0])
        r = tb.compute_obs_pose("all")
        assert r["status"] == "ok"
        ds.append(r["data"]["root_target_odom"]["d_obs"])
        if abs(ds[-1] - ds[-2]) < 1e-9:
            break
    assert abs(ds[-1] - ds[-2]) < 1e-9 and 1.20 - 1e-9 <= ds[-1] <= 1.35 + 1e-9, ds
    lat = r["data"]["spec_view"]["lateral"]
    assert lat[0] <= -0.6 + 1e-6 and lat[1] >= 0.6 - 1e-6                     # both side edges of the wall
    rt = tb.ws.root_target
    inside, _ = tb._in_view(tb._cam_T_at(rt["pos"][0], rt["pos"][1], rt["yaw"], tb._lean_levels()[0]),
                            r["data"]["spec_view"]["corners_odom"])
    assert inside.all() and rt["view_pitch"] == 0.0
    # the in-view tests use the region (one pseudo-face), and the region never shrinks
    assert list(tb._obs_faces) == ["spec_region"]
    tb.ws.boxes = {k: b for k, b in list(tb.ws.boxes.items())[:2]}
    assert tb._spec_view_points([b.face for b in tb.ws.boxes.values()])[1]["lateral"] == pytest.approx(lat)


def test_obs_pose_is_centred_on_the_region_not_on_the_faces_of_this_view():
    """Review 2026-11-19: the region keeps every face fitted so far; a later view that fitted only a part of the wall
    (here the left column) must not move the pose onto that part: the pose stays on the region's lateral middle."""
    tb = _scanroom_obs_toolbox()
    _fit_visible_faces(tb, 0.0)
    tb.compute_obs_pose("all")
    _fit_visible_faces(tb, tb.ws.root_target["pos"][0])
    r = tb.compute_obs_pose("all")
    lat, y_mid = r["data"]["spec_view"]["lateral"], r["data"]["root_target_odom"]["pos"][1]
    assert y_mid == pytest.approx(0.0, abs=1e-6)                    # the wall is centred on y 0
    tb.ws.boxes = {k: b for k, b in tb.ws.boxes.items() if k.startswith("C1")}
    r = tb.compute_obs_pose("all")
    assert r["data"]["spec_view"]["lateral"] == pytest.approx(lat)
    assert r["data"]["root_target_odom"]["pos"][1] == pytest.approx(y_mid, abs=1e-6)


# 4-box scenario texts as they were before the head_views / phase1_view change (g1v2-sub4-run1): byte for byte
SUB4_START = ("Start task 3: scan only the labels of the boxes in columns 1, 2 from the left and rows 2, 3 from the top of "
              "the box wall, row by row from the top, left to right within a row (the tools fix this order). Work from "
              "near your start pose: do not walk back to observe the whole wall. plan_scans looks from every standing "
              "head view from here (upright and leaned back) to see the top row and the edges of the wall, and counts "
              "the rows and columns from there. This task covers the side of the boxes facing you only.")
SUB4_RULES = (
    "- Phase 1 sequence: detect_boxes -> select_target_box(\"all\") -> plan_scans. Work from near the start pose: do NOT\n"
    "  call compute_obs_pose / approach and do not walk back to observe the whole wall. plan_scans looks from every\n"
    "  standing head view from here (upright and leaned back) to see the top row and the edges of the wall, counts the\n"
    "  columns and rows of the labels from the edges the task names, plans ONLY the labels in the requested cells (the others\n"
    "  are listed as not_targeted and are never scanned) and fixes the scan order (rows top to bottom, left to right inside a row).\n"
    "  It reports a requested cell without a label, an edge of the wall it could not see, or a count it could not verify\n"
    "  from the box faces, as a WARNING in its reason (finish(success=true) is then refused).\n"
    "  If no box is visible, call look_head and choose an explore move (left / right / back / forward <= 0.5 m,\n"
    "  turn_left / turn_right <= 15 deg); the head camera looks about 30 deg down, so far or tall boxes may appear at the top edge or above it.\n"
    "  detect_boxes and the perception tools lean the torso back by themselves to look higher (the robot stands upright again\n"
    "  before it walks or moves the arm); you do not control the head view.\n"
    "  Only if plan_scans finds no label at all, explore a little sideways and call detect_boxes and plan_scans\n"
    "  again. Explore moves that would end closer than the minimum distance to a box face are refused by the tool.")
SUB4_SYSTEM_SHA256 = "d2b42b6dc4af9e0f3de99dcf875595f11fd4cc46ec9356fd7d83649b9e6dba7a"
ALL_SYSTEM_SHA256 = "c3c00b7216550b06b7c25112d1d2d82ba23ae766884307ab5c8e33d003ce0bc9"
# json.dumps(openai_tools(TASK_TOOLS[3], g1_v2 tool_texts)) at the tag g1v2-sub4-run1 (computed in a worktree of the tag)
TASK3_TOOLS_SHA256 = "172e3dcef8228fd08f0a47b590808a06acc6513fd86bad97c295707639504a97"


def _agent_full_texts(monkeypatch, spec, head_views=None, phase1_view=None):
    """System prompt, start message, every message (JSON), prompt version and the tools list (JSON) of run_agent_full on
    a g1_v2 / d435i toolbox with the scenario's task fields."""
    import json
    from g1agent.orchestrators.agent_full import run_agent_full
    from test_task3_logic import FakeLLM, ScriptedTB
    tb = ScriptedTB(FakeRobot(), plan_n=1)
    tb.cfg = config.apply_head_camera(config.apply_robot_variant(config.load_all(), "g1_v2"), "d435i")
    tb.set_scan_targets(spec)
    tb.set_head_views(head_views)
    tb.set_phase1_view(phase1_view)
    monkeypatch.setattr(tb, "look_head", lambda: (tbmod.ok({}), np.zeros((4, 4, 3), np.uint8)))
    h = _harness(tb)
    tools_seen = []

    class ToolsLLM(FakeLLM):
        def chat(self, messages, tools=None, tool_choice=None):
            tools_seen.append(tools)
            return super().chat(messages, tools, tool_choice)

    llm = ToolsLLM([("finish", {"task": 3, "success": False, "summary": "x"})])
    run_agent_full(h, config.load_all()["task3"], "all", llm, h.log)
    msgs = llm.seen[0]
    start = " ".join(p["text"] for p in msgs[1]["content"] if p.get("type") == "text")
    return msgs[0]["content"], start, json.dumps(msgs), h.log.llms[0]["prompt_version"], json.dumps(tools_seen[0])


def test_four_box_prompt_texts_are_unchanged_and_the_upright_task_never_mentions_leaning(monkeypatch):
    """Golden texts: the 4-box scenario (lean views on, start_pose) and the every-label task give exactly the texts of
    the tagged 4-box state; the rows 2-4 scenario (head_views upright, obs_pose) never mentions leaning back and asks
    for the observation pose walk."""
    import hashlib
    import json
    from g1agent.orchestrators import agent_full as af
    system, start, _, _, tools = _agent_full_texts(monkeypatch, SPEC)
    assert start.startswith(SUB4_START) and hashlib.sha256(system.encode()).hexdigest() == SUB4_SYSTEM_SHA256
    assert hashlib.sha256(tools.encode()).hexdigest() == TASK3_TOOLS_SHA256          # the tools list too (review)
    tb = bare_toolbox(FakeRobot(), variant="g1_v2")
    tb.set_scan_targets(SPEC)
    assert af.task3_prompt_kw(tb, True)["phase1_rules"] == SUB4_RULES
    assert af._start_text(3, "all", False, tb.scan_targets) == SUB4_START
    system, _, _, _, tools = _agent_full_texts(monkeypatch, None)
    assert hashlib.sha256(system.encode()).hexdigest() == ALL_SYSTEM_SHA256
    assert hashlib.sha256(tools.encode()).hexdigest() == TASK3_TOOLS_SHA256
    system, start, everything, version, tools = _agent_full_texts(monkeypatch, SPEC_R234_TASK, "upright", "obs_pose")
    # obs_pose: plan_scans is called at the observation pose only; every other tool text is the tag's
    plan = {t["function"]["name"]: t["function"]["description"] for t in json.loads(tools)}
    assert plan["plan_scans"].startswith("Task 3, after reaching the observation pose: localise")
    _, _, _, _, tools4 = _agent_full_texts(monkeypatch, SPEC)
    assert {k: v for k, v in plan.items() if k != "plan_scans"} == \
        {t["function"]["name"]: t["function"]["description"] for t in json.loads(tools4) if t["function"]["name"] != "plan_scans"}
    assert "(head views: upright only, scenario)" in version
    for text in (system, start):
        assert "lean" not in text.replace("clean", "") and af.LEAN_VIEW_NOTE.strip() not in text
    assert "leaned back" not in everything
    assert "compute_obs_pose -> approach" in system and "upright head views only" in start
    assert "all columns" in start and "rows 2, 3, 4 from the bottom" in start


# ------------------------------------------------------------------ order rule "serpentine" (wall-layout suite)
def test_serpentine_order_matches_evaluator_copy_and_rule():
    """Rule "serpentine" (user decision 2026-11-19): the rows of rule "rows", the top row left to right, the next right
    to left, alternating over the rows that hold items. The evaluator's copy gives the same order on random input;
    rows / columns are unchanged (row_order's default direction is the old one)."""
    from g1agent.eval import scan_order as ev
    rng = np.random.default_rng(7)
    for _ in range(300):
        items = []
        for k in range(int(rng.integers(1, 9))):
            y, z = float(rng.uniform(-0.6, 0.6)), float(rng.uniform(0.2, 1.3))
            w, h = float(rng.uniform(0.04, 0.2)), float(rng.uniform(0.02, 0.08))
            items.append({"key": f"k{k}", "y_lo": y - w / 2, "y_hi": y + w / 2, "z_lo": z - h / 2, "z_hi": z + h / 2, "z": z})
        assert so.order_items(items, "serpentine") == ev.order_items(items, "serpentine")
        for d in (True, False):
            assert so.row_order(items, d) == ev.row_order(items, d)
        assert so.row_order(items) == so.row_order(items, True)
        # serpentine = the rows of rule "rows" (same sets, same row sequence), every second row reversed (the random
        # lateral centres are distinct, so reversing a row equals sorting it right to left)
        rows = so._overlap_groups(items, "z_lo", "z_hi")
        assert so.order_items(items, "serpentine") == \
            [k for i, r in enumerate(rows) for k in (so.row_order(r) if i % 2 == 0 else so.row_order(r)[::-1])]
    it = lambda k, y, z: {"key": k, "y_lo": y - 0.05, "y_hi": y + 0.05, "z_lo": z - 0.02, "z_hi": z + 0.02, "z": z}
    grid = [it("a", 0.4, 1.0), it("b", 0.2, 1.01), it("c", 0.4, 0.7), it("d", 0.2, 0.69), it("e", 0.4, 0.4), it("f", 0.2, 0.41)]
    assert so.order_items(grid, "serpentine") == ["a", "b", "d", "c", "e", "f"]
    assert so.order_items(grid, "rows") == ["a", "b", "c", "d", "e", "f"]
    assert so.order_items(grid, "columns") == ["a", "c", "e", "b", "d", "f"]
    t = so.parse_targets({"columns": "all", "rows_from_bottom": [2, 3, 4], "order": "serpentine"})
    assert t["order"] == "serpentine" and so.ORDER_RULES == ("columns", "rows", "serpentine")
    assert "the next right to left, alternating" in so.order_rule_text("serpentine")
    assert so.order_rule_text("rows") == ("rows of overlapping vertical label extents, top to bottom; left to right inside "
                                          "a row as seen facing this side")


def test_serpentine_order_by_cells_r234_wall():
    """Rows 2-4 x all columns by box cells: R4 left to right, R3 right to left, R2 left to right; a fragment inside a
    cell stays in its cell; the result does not depend on the edge the rows are counted from; without a grid (or a
    label without a cell) the label rule."""
    t = so.parse_targets(dict(SPEC_R234, order="serpentine"))
    items = _wall_items(drop=0.035) + [_sliver(3, 4)]
    edges, centres = _wall_evidence()
    g = so.resolve_grid(items, t, edges, centres, bounds=FLOOR)
    sel = [it for it in items if it["key"] in g["selected"]]
    expect = ([f"C{c}R4" for c in range(1, 7)] + [f"C{c}R3" for c in range(6, 0, -1)]
              + [f"C{c}R2" for c in range(1, 7)])
    order = so.order_by_cells(sel, g["cell_of"], g, "serpentine")
    assert [k for k in order if k != "sliver"] == expect and order.index("sliver") in (2, 3)
    g_top = {"columns": g["columns"], "rows": [{**s, "index": 5 - s["index"]} for s in g["rows"]]}
    cells_top = {k: {"column": c["column"], "row": 5 - c["row"]} for k, c in g["cell_of"].items()}
    assert so.order_by_cells(sel, cells_top, g_top, "serpentine") == order
    cells = dict(g["cell_of"], sliver={"column": 3, "row": None})
    assert so.order_by_cells(sel, cells, g, "serpentine") == so.order_items(sel, "serpentine")
    assert so.order_by_cells(sel, g["cell_of"], None, "serpentine") == so.order_items(sel, "serpentine")
    # the alternation counts the rows that hold labels: without the R3 labels, R2 runs right to left
    sel2 = [it for it in sel if not it["key"].endswith("R3")]
    assert so.order_by_cells(sel2, g["cell_of"], g, "serpentine")[-6:] == [f"C{c}R2" for c in range(6, 0, -1)]
    # rows / columns by cells are unchanged by the new rule (pinned by test_r234_cell_order_ignores_duplicates...)
    assert [k for k in so.order_by_cells(sel, g["cell_of"], g, "rows") if k != "sliver"] == \
        [f"C{c}R{r}" for r in (4, 3, 2) for c in range(1, 7)]


def test_serpentine_order_by_cells_reverses_labels_inside_a_cell_with_its_row():
    """Two labels side by side in one box (C2R3, a right-to-left row): they are read right to left, as the row."""
    t = so.parse_targets(dict(SPEC_R234, order="serpentine"))
    items = _wall_items(drop=0.035)
    items = [x for x in items if x["key"] != "C2R3"]
    y = COLS6[1]
    items += [{"key": "C2R3a", "y_lo": y + 0.01, "y_hi": y + 0.07, "z_lo": 0.69, "z_hi": 0.73, "z": 0.71},
              {"key": "C2R3b", "y_lo": y - 0.07, "y_hi": y - 0.01, "z_lo": 0.69, "z_hi": 0.73, "z": 0.71}]
    edges, centres = _wall_evidence()
    g = so.resolve_grid(items, t, edges, centres, bounds=FLOOR)
    sel = [it for it in items if it["key"] in g["selected"]]
    order = so.order_by_cells(sel, g["cell_of"], g, "serpentine")
    r3 = [k for k in order if "R3" in k]
    assert r3 == ["C6R3", "C5R3", "C4R3", "C3R3", "C2R3b", "C2R3a", "C1R3"]          # a is on the left of b
    assert so.order_in_cell([x for x in sel if x["key"].startswith("C2R3")], "serpentine", False) == ["C2R3b", "C2R3a"]
    assert so.order_in_cell([x for x in sel if x["key"].startswith("C2R3")], "serpentine", True) == ["C2R3a", "C2R3b"]
    assert so.order_in_cell([x for x in sel if x["key"].startswith("C2R3")], "rows", False) == ["C2R3a", "C2R3b"]


def test_order_lattice_rules():
    """The layout cells themselves (column 1 = leftmost, row 1 = top): rows, columns, serpentine; the serpentine
    alternation counts the rows that hold cells; each cell reports its row's direction (for order_in_cell)."""
    cells = [{"key": f"C{c}R{r}", "column": c, "row": r} for r in (2, 3, 4) for c in range(1, 4)][::-1]   # shuffled
    keys = lambda rule: [x["key"] for x in so.order_lattice(cells, rule)]
    assert keys("rows") == [f"C{c}R{r}" for r in (2, 3, 4) for c in (1, 2, 3)]
    assert keys("columns") == [f"C{c}R{r}" for c in (1, 2, 3) for r in (2, 3, 4)]
    assert keys("serpentine") == ["C1R2", "C2R2", "C3R2", "C3R3", "C2R3", "C1R3", "C1R4", "C2R4", "C3R4"]
    snake = so.order_lattice(cells, "serpentine")
    assert [x["left_to_right"] for x in snake] == [True] * 3 + [False] * 3 + [True] * 3
    assert all(x["left_to_right"] for x in so.order_lattice(cells, "rows"))
    gap = [c for c in cells if c["row"] != 3]                       # rows 2 and 4 only: row 4 is the second row
    assert [x["key"] for x in so.order_lattice(gap, "serpentine")] == ["C1R2", "C2R2", "C3R2", "C3R4", "C2R4", "C1R4"]
    with pytest.raises(ValueError):
        so.order_lattice(cells, "spiral")
