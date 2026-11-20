"""CPU tests of the wall-layout STRIP mode (scan_targets order "strips", design scanroom_strip_design.md section 9 with
the user answers of 2026-11-20: the floor is a check only, the left end is the only anchor, every strip restarts its
top row left to right; no simulator, no GPU). Synthetic head frames use the model head (dz 0): the settled head of the
simulator equals it (design Appendix A, C0 P1). Plan-stance tests stub the stance evaluation (IK) with a geometric rule:
they test which stance the planner chooses (in place, shared, single), not the arm's reach."""
import json
import math
import os
import sys

import numpy as np
import pytest

from g1agent.common import config
from g1agent.common.transforms import T_from_xy_yaw, apply, inv_T
from g1agent.eval import evaluator
from g1agent.eval import scan_order as ev
from g1agent.harness.guards import Harness
from g1agent.harness.registry import LAYOUT_ONLY, STRIP_ONLY, TASK_TOOLS, task_tools
from g1agent.orchestrators import rule
from g1agent.tools import scan_order as so
from g1agent.tools import toolbox as tbmod
from test_task3_eval import _lattice_gt, _subset_gt_boxes
from test_task3_logic import FakeLLM, FakeLog, FakeRobot, bare_toolbox, read
from test_wall_layout import (LAYOUT, WALL_X, Cam, _body, _stored_frame, box_faces, cell_x, head_frame,
                              kin_toolbox)

SPEC_Q = {"columns_from_left": [1, 2], "rows_from_bottom": [2, 3, 4], "order": "strips", "strip_columns": 2}
SPEC_F = {"columns": "all", "rows_from_bottom": [2, 3, 4], "order": "strips", "strip_columns": 2}
QUICK = ["C1R4", "C2R4", "C2R3", "C1R3", "C1R2", "C2R2"]
FULL = (["C1R4", "C2R4", "C2R3", "C1R3", "C1R2", "C2R2"] + ["C3R4", "C4R4", "C4R3", "C3R3", "C3R2", "C4R2"]
        + ["C5R4", "C6R4", "C6R3", "C5R3", "C5R2", "C6R2"])
SUITES = "scanroom6x5_c12r234_strip_L0", "scanroom6x5_r234_strip_L0"


def cells_of(tb):
    return [f"C{b.wall['cell'][0]}R{b.wall['cell'][1]}" for b in (tb.ws.barcodes[i] for i in tb.ws.scan_plan)]


def planned(spec=SPEC_Q, x=WALL_X - 0.6, y=0.4, layout=LAYOUT):
    """A strip-mode toolbox registered from the left end (root at (x, y) facing the wall, the left edge in view)."""
    tb = kin_toolbox(spec=spec, layout=layout)
    h = head_frame(tb, x, y)
    tb.head = h
    tb.ws.boxes = {k: tbmod.BoxEstimate(k, [0, 0, 1, 1], 0.9, 1.0, face=f) for k, f in box_faces().items()}
    tb._stack_init(list(box_faces().values()))
    tb._stack_add_faces(0, list(box_faces().values()))
    T = np.asarray(h["T_odom_root"], float)
    tb._odom = lambda: T
    r = tb._plan_layout(h, box_faces(), {})
    return tb, r


# ------------------------------------------------------------------ 1. task spec
def test_parse_targets_strip_columns_only_with_the_order_strips():
    t = so.parse_targets(SPEC_Q)
    assert t["order"] == "strips" and t["strip_columns"] == 2
    for bad in (0, -1, True, 2.0, "2", None):
        with pytest.raises(ValueError, match="strip_columns"):
            so.parse_targets(dict(SPEC_Q, strip_columns=bad))
    with pytest.raises(ValueError, match="strip_columns"):
        so.parse_targets({k: v for k, v in SPEC_Q.items() if k != "strip_columns"})
    with pytest.raises(ValueError, match="parameter of the order 'strips'"):
        so.parse_targets(dict(SPEC_Q, order="serpentine"))
    assert "strip_columns" not in so.parse_targets({"columns": "all", "rows_from_bottom": [2, 3, 4], "order": "rows"})
    with pytest.raises(ValueError, match="box-lattice"):
        so.order_items([{"key": "a", "y_lo": 0, "y_hi": 1, "z_lo": 0, "z_hi": 1, "z": 0.5}], "strips")


def test_set_wall_layout_strip_mode_needs_whole_strips():
    tb = kin_toolbox(spec=SPEC_F)
    assert tb.strip_mode and tb.strip_columns == 2 and tb.n_strips == 3
    with pytest.raises(ValueError, match="does not split into strips of 4"):
        kin_toolbox(spec=dict(SPEC_F, strip_columns=4))
    a = kin_toolbox()                                                       # approach A: no strip mode
    assert not a.strip_mode and a.strip_columns is None and a.n_strips is None
    assert not bare_toolbox(FakeRobot()).strip_mode and bare_toolbox(FakeRobot())._next_strip() is None


def test_run_task_eval_args_for_strips():
    sys.path.insert(0, os.path.join(config.ROOT, "scripts"))
    import run_task
    sc = {"id": "x", "target_boxes": ["C1R4"], "scan_targets": SPEC_Q, "wall_layout": LAYOUT}
    assert run_task.subset_eval_args(sc, 3) == {"target_boxes": ["C1R4"], "order_rule": "strips", "order_unit": "box",
                                               "strip_columns": 2, "layout_columns": 6}
    with pytest.raises(ValueError, match="needs a wall_layout"):
        run_task.subset_eval_args({k: v for k, v in sc.items() if k != "wall_layout"}, 3)
    other = dict(sc, scan_targets={"columns": "all", "rows_from_bottom": [2], "order": "serpentine"})
    assert run_task.subset_eval_args(other, 3) == {"target_boxes": ["C1R4"], "order_rule": "serpentine", "order_unit": "box"}


# ------------------------------------------------------------------ 2. order
def _lattice(spec_cells):
    return [{"key": k, "column": int(k[1]), "row": 6 - int(k[3])} for k in spec_cells]      # row 1 = top of 5


def test_order_lattice_strips():
    keys = lambda cells: [x["key"] for x in so.order_lattice(_lattice(cells), "strips", 2)]
    assert keys(sorted(QUICK)) == QUICK
    full = [f"C{c}R{r}" for c in range(1, 7) for r in (2, 3, 4)]
    assert keys(full[::-1]) == FULL
    out = so.order_lattice(_lattice(full), "strips", 2)
    assert [x["strip"] for x in out] == [1] * 6 + [2] * 6 + [3] * 6
    assert [x["left_to_right"] for x in out[:6]] == [True, True, False, False, True, True]
    # columns 3-4 only: strip 2, its own top row first, left to right
    assert keys([f"C{c}R{r}" for c in (3, 4) for r in (2, 3, 4)]) == ["C3R4", "C4R4", "C4R3", "C3R3", "C3R2", "C4R2"]
    # rows counted per strip: strip 1 holds rows 4 and 2, strip 2 only row 3 -> both start left to right
    assert keys(["C1R4", "C2R4", "C1R2", "C2R2", "C3R3", "C4R3"]) == ["C1R4", "C2R4", "C2R2", "C1R2", "C3R3", "C4R3"]
    assert [so.strip_of(c, 2) for c in range(1, 7)] == [1, 1, 2, 2, 3, 3]
    with pytest.raises(ValueError, match="strip_columns"):
        so.order_lattice(_lattice(QUICK), "strips")
    with pytest.raises(ValueError, match="parameter of the order 'strips'"):
        so.order_lattice(_lattice(QUICK), "rows", 2)
    assert so.order_in_cell([{"key": "a", "y_lo": 0.1, "y_hi": 0.2, "z_lo": 0, "z_hi": 1, "z": 0.5},
                             {"key": "b", "y_lo": -0.2, "y_hi": -0.1, "z_lo": 0, "z_hi": 1, "z": 0.5}], "strips", False) == ["b", "a"]
    assert "strips of 2 columns from the left" in so.order_rule_text("strips", 2)


# ------------------------------------------------------------------ 3. evaluator
@pytest.mark.parametrize("sid", SUITES)
def test_evaluator_strip_order_equals_the_agent_order_on_the_suites(sid):
    sc, gt, stages = _subset_gt_boxes(sid + "_0000")
    st = sc["scan_targets"]
    box = {b["content"]: b["box_id"] for b in gt["barcodes"]}
    kw = {"target_boxes": sc["target_boxes"], "order_rule": "strips", "order_unit": "box", "strip_columns": 2,
          "layout_columns": sc["wall_layout"]["columns"]}
    ws = {"scan_plan": [], "scanned": []}
    m = evaluator.evaluate(3, gt, {}, ws, stages=stages, scan_log=[], **kw)
    agent = [x["key"] for x in so.order_lattice(_lattice(sc["target_boxes"]), "strips", st["strip_columns"])]
    assert [box[c] for c in m["gt_order"]] == agent == (QUICK if "c12" in sid else FULL)
    assert m["strip_columns"] == 2 and m["layout_columns"] == 6 and m["order_unit"] == "box"
    with pytest.raises(ValueError, match="columns"):
        evaluator.evaluate(3, gt, {}, ws, stages=stages, scan_log=[], **dict(kw, layout_columns=5))
    with pytest.raises(ValueError, match="order unit 'box'"):
        evaluator.evaluate(3, gt, {}, ws, stages=stages, scan_log=[], **dict(kw, order_unit="label"))
    r = evaluator.evaluate(3, gt, {}, ws, stages=stages, scan_log=[], target_boxes=sc["target_boxes"],
                           order_rule="serpentine", order_unit="box")
    assert "strip_columns" not in r and "layout_columns" not in r


@pytest.mark.parametrize("yaw", [0.0, 0.7, -2.3])
def test_evaluator_strips_use_absolute_columns_in_a_yawed_frame(yaw):
    """Only columns 3-4 expected on a 4-column wall: they are strip 2 (absolute columns, every GT box), whose first
    row runs left to right; touching boxes under float noise of a yawed frame group by centre containment."""
    labels = [(f"C{c}R{r}", c, r, 0.0, 0.25 * r - 0.03) for c in (1, 2, 3, 4) for r in (1, 2, 3)]
    gt, stages, _ = _lattice_gt(labels, yaw, cols=4, rows=3)
    target = [f"C{c}R{r}" for c in (3, 4) for r in (1, 2, 3)]
    o = evaluator.evaluate(3, gt, {}, {"scan_plan": [], "scanned": []}, stages=stages, scan_log=[], target_boxes=target,
                           order_rule="strips", order_unit="box", strip_columns=2, layout_columns=4)["gt_order"]
    assert o == ["C3R3", "C4R3", "C4R2", "C3R2", "C3R1", "C4R1"]
    with pytest.raises(ValueError, match="order unit 'box'"):
        ev.order_items([], "strips")


def test_evaluator_strips_one_side_test_without_a_tolerance():
    """The labels' side is checked by box_face's own face choice (no rounding tolerance): a wall whose boxes carry a
    tiny yaw jitter is one side; a label on another side of its box is refused."""
    def wall(jitter):
        boxes, labels = {}, []
        for k, y in enumerate((0.3, 0.1, -0.1, -0.3)):
            a = 0.5 * jitter * k
            boxes[f"b{k}"] = {"pos": [1.0, y, 0.125], "size": [0.3, 0.2, 0.25],
                              "quat_wxyz": [math.cos(a), 0.0, 0.0, math.sin(a)]}
            labels.append({"key": f"l{k}", "center": [0.85, y, 0.125], "normal": [-1.0, 0.0, 0.0], "up": [0, 0, 1],
                           "size": [0.05, 0.05], "box_id": f"b{k}"})
        return boxes, labels
    boxes, labels = wall(1e-7)
    assert ev.order_boxes_in_frame(labels, boxes, np.eye(4), "strips", 2, 4) == ["l0", "l1", "l2", "l3"]
    labels[3] = dict(labels[3], center=[1.0, -0.4, 0.125], normal=[0.0, -1.0, 0.0])
    with pytest.raises(ValueError, match="one labelled wall side"):
        ev.order_boxes_in_frame(labels, boxes, np.eye(4), "strips", 2, 4)


# ------------------------------------------------------------------ 5. settle before the first head view
class SettleRobot:
    def __init__(self):
        self.calls = []

    def stand(self, duration):
        self.calls.append(("stand", duration))
        return {"status": "ok", "reason": f"stood {duration:.2f} s"}

    def get_head_obs(self, depth=False):
        self.calls.append(("head",))
        return {"rgb": np.zeros((4, 4, 3), np.uint8), "depth": np.zeros((4, 4)), "K": np.eye(3).tolist(), "timestamp": 1.0}

    def get_odom(self):
        return {"T_odom_root": np.eye(4).tolist()}


@pytest.mark.parametrize("layout", [LAYOUT, None])
def test_layout_episodes_stand_still_once_before_the_first_head_capture(layout):
    tb = kin_toolbox(spec=SPEC_Q if layout else {"columns": "all", "rows_from_bottom": [2], "order": "rows"},
                     layout=layout)
    tb.robot = SettleRobot()
    tb.cfg["perception"]["head_depth"]["source"] = "render"
    tb._capture_head()
    tb._capture_head()
    settle = tb.cfg["robot"]["body"]["settle_s"]
    exp = [("stand", settle), ("head",), ("head",)] if layout else [("head",), ("head",)]
    assert tb.robot.calls == exp


# ------------------------------------------------------------------ 6. phase-1 region
def test_strip_view_region_from_the_start_then_the_left_end():
    tb = kin_toolbox(spec=SPEC_Q)
    faces = list(box_faces().values())
    tb.head = head_frame(tb, WALL_X - 0.6, 0.0)                         # start: no side edge in view
    _, info, ref = tb._strip_view_points(faces)
    S = info["seen"]
    assert info["edges_seen"] == {"left": False, "right": False}
    assert info["lateral"] == pytest.approx([S[1] - 1.2, S[0] + 0.4], abs=1e-9)
    assert info["z_range"] == [0.75, 1.0]                                # row 4: the top requested row of strip 1
    tb.head = head_frame(tb, WALL_X - 0.75, -float(np.mean(info["lateral"])))   # the pose centred on that region
    _, info, ref = tb._strip_view_points(faces)
    assert info["edges_seen"]["left"] and info["lateral"] == pytest.approx([-0.6, -0.2], abs=0.003)
    assert ref[1] == pytest.approx(0.4, abs=0.003)


def _stub_extent(tb, frames, faces):
    out = None
    for t, e, seen in frames:
        tb.head = {"t": t}
        tb._wall_extent = lambda h, pl, e=e, seen=seen: {
            "left": None if e.get("left") is None else {"s": e["left"], "depth": 0.6},
            "right": None if e.get("right") is None else {"s": e["right"], "depth": 0.6}, "seen": seen}
        out = tb._strip_view_points(faces)
    return out


def test_strip_view_edges_checked_against_the_wall_seen_right_only_and_mismatch():
    faces = list(box_faces().values())
    tb = kin_toolbox(spec=SPEC_Q)                    # a "left edge" with wall seen 0.42 m left of it: dropped, logged
    _, info, _ = _stub_extent(tb, [(1.0, {"left": 0.12}, [-0.3, 0.3])], faces)
    assert info["edge_rejected"][0]["side"] == "left" and info["edges_seen"] == {"left": False, "right": False}
    tb = kin_toolbox(spec=SPEC_Q)                    # right edge only: walk to the left end W left of it
    _, info, _ = _stub_extent(tb, [(1.0, {"right": 0.6}, [0.1, 0.6])], faces)
    assert info["lateral"] == pytest.approx([-0.6, -0.2], abs=1e-9) and "layout_mismatch" not in info
    tb = kin_toolbox(spec=SPEC_Q)                    # one frame: a right edge the same frame contradicts is dropped
    _, info, _ = _stub_extent(tb, [(1.0, {"right": 0.6}, [-0.75, 0.6])], faces)
    assert info["edge_rejected"][0]["side"] == "right" and "layout_mismatch" not in info
    tb = kin_toolbox(spec=SPEC_Q)                    # the remembered right edge, then the wall seen 0.15 m beyond where
    _, info, _ = _stub_extent(tb, [(1.0, {"right": 0.6}, [0.1, 0.6]),       # the layout puts the left end
                                   (2.0, {}, [-0.75, -0.2])], faces)
    assert info["layout_mismatch"]["s_right"] == 0.6
    tb.ws.boxes = {k: tbmod.BoxEstimate(k, [0, 0, 1, 1], 0.9, 1.0, face=f) for k, f in box_faces().items()}
    tb.ws.target_box_id = "all"
    tb.head = head_frame(tb, WALL_X - 0.6)
    r = tb.compute_obs_pose("all")
    assert r["status"] == "fail" and "does not match the layout" in r["reason"] and tb.ws.root_target is None
    tb = kin_toolbox(spec=SPEC_Q)                    # a remembered left edge with the wall seen beyond it later: dropped
    _, info, _ = _stub_extent(tb, [(1.0, {"left": -0.2}, [-0.2, 0.3]), (2.0, {}, [-0.6, -0.1])], faces)
    rj = info["edge_rejected"]
    assert len(rj) == 1 and rj[0]["side"] == "left" and not rj[0]["new"] and rj[0]["s"] == pytest.approx(-0.2)
    assert rj[0]["seen"] == pytest.approx([-0.6, 0.3])
    assert info["edges_seen"]["left"] is False and "left" not in tb._layout_edges


# ------------------------------------------------------------------ 7. registration
def test_strip_registration_left_edge_anchor_margin_and_plan():
    tb, r = planned()
    assert r["status"] == "ok", r["reason"]
    w = tb.ws.wall
    assert tb._wall_axes()["origin"] == pytest.approx([WALL_X, 0.6, 0.0], abs=0.003)
    assert w["anchor"]["side"] == "left" and w["closure"] == [] and w["z0"]["source"] == "odom"
    fx = tb.head["K"][0, 0]
    d_edge = 10.3 * w["anchor"]["depth_m"] / fx
    assert w["anchor"]["delta_edge"] == pytest.approx(d_edge) and w["delta_lat"] == math.ceil(round(d_edge * 1000, 6)) / 1000
    assert cells_of(tb) == QUICK and r["data"]["strips"]["strip"] == 1
    assert tb.ws.strips["current"] == 1 and tb.ws.strips["strips"][1]["ref_source"] == "plan_scans"
    assert "strip 1 = columns 1-2 at the left end" in r["reason"]
    assert r["data"]["order_rule"].startswith("strips of 2 columns")


def test_strip_registration_failures_and_closure():
    tb, r = planned(x=WALL_X - 0.6, y=-0.4)                           # right end: no left edge
    assert r["status"] == "fail" and "the left side edge of the wall" in r["reason"] and tb.ws.wall is None
    tb, r = planned(x=WALL_X - 1.2, y=0.0)                            # both edges: the right one is the closure
    assert r["status"] == "ok" and len(tb.ws.wall["closure"]) == 1
    assert abs(tb.ws.wall["closure"][0]["ds_m"]) < 0.005 and not tb.ws.wall["anchor"]["closure_open"]
    tb, r = planned(x=WALL_X - 1.2, y=0.0, layout=dict(LAYOUT, columns=4))   # a 0.8 m layout, a 1.2 m wall seen:
    assert r["status"] == "fail" and "contradicts the wall seen" in r["reason"]  # the left edge cannot be its end
    tb, r = planned(x=WALL_X - 1.2, y=0.0, layout=dict(LAYOUT, columns=8))   # a 1.6 m layout: the closure fails
    assert r["status"] == "fail" and "does not match the layout" in r["reason"] and tb.ws.wall is None


def test_strip_registration_floor_is_a_check_only():
    """User answer Q1 (2026-11-20): the floor seen is logged against the odometry floor and never applied; half a box
    off is logged as rejected, not a failure."""
    tb = kin_toolbox(spec=SPEC_Q)
    h = head_frame(tb, WALL_X - 1.2, 0.0, floor_z=0.15)
    tb._layout_edges = {}
    failed, diag = tb._register_wall_strip(h, list(box_faces(floor_z=0.15).values()))
    w = tb.ws.wall
    assert failed is None and w["z0"] == {"value": 0.0, "source": "odom", "t": h["t"]}
    fc = w["floor_checks"][0]
    assert fc["z_floor"] == pytest.approx(0.15, abs=0.01) and "rejected" in fc["note"]
    assert tb._wall_axes()["origin"][2] == 0.0


# ------------------------------------------------------------------ 8. lateral margin
def test_strip_margin_exact_float_safe_and_growing_with_the_walk():
    tb, _ = planned()
    tb.ws.wall["anchor"]["delta_edge"] = 0.07
    assert tb._strip_delta(0.0) == 0.07                                   # not 0.071 (float-safe ceil)
    assert tb._strip_delta(0.5) == 0.08 and tb._strip_delta(-0.5) == 0.08   # eps 0.02 x 0.5 m
    tb.ws.wall["anchor"]["delta_edge"] = 0.0701
    assert tb._strip_delta(0.0) == 0.071
    tb.ws.wall["anchor"]["delta_edge"] = 0.01
    T = T_from_xy_yaw(WALL_X - 0.6, -0.1, 0.0, 0.75)                     # 0.5 m right of the anchor pose (y 0.4)
    tb._odom = lambda: T
    assert tb._strip_D_now() == pytest.approx(0.5) and tb._delta_now() == 0.02
    far = next(tb.ws.barcodes[b] for b in tb.ws.scan_plan if tb.ws.barcodes[b].wall["cell"][0] == 1)
    # planned from here for column 1 (0.6 m back to the left): D = 0.5 - 0.6 -> 0.1 m
    assert tb._delta_plan(far) == pytest.approx(0.012, abs=0.001)          # (+1 mm: the frame's origin, ceil)
    far.wall["delta_plan"] = 0.012
    assert tb._search_grow(far, plan=True)[0] == 0.012 and tb._search_grow(far)[0] == 0.02


def test_strip_reregistration_reanchors_logs_closure_and_checks_the_floor():
    tb, _ = planned()
    w = tb.ws.wall
    w["anchor"]["closure_open"] = True
    T = T_from_xy_yaw(WALL_X - 0.6, -0.4, 0.0, 0.75)
    tb._odom = lambda: T
    h = head_frame(tb, WALL_X - 0.6, -0.4)                             # the right end: only the right edge
    tb._capture_head = lambda: (setattr(tb, "head", h), h)[1]
    rec = tb._reregister_at_stance()
    assert rec["anchored"] == ["right"] and w["anchor"]["side"] == "right" and abs(rec["ds_m"]) < 0.004
    assert len(w["closure"]) == 1 and w["closure"][0]["side"] == "right" and w["closure"][0]["D_m"] == pytest.approx(0.8)
    assert rec["delta_lat"] == w["delta_lat"] and rec["D_m"] == pytest.approx(0.0)
    fc = rec["floor_check"]
    assert w["floor_checks"][-1] is fc and fc["odom_floor"] == 0.0
    assert fc["z_floor"] is None or abs(fc["z_floor"]) < 0.01
    T0 = tb._wall_T().copy()                                           # the floor is never applied
    assert np.allclose(T0[2, 3], 0.0)


# ------------------------------------------------------------------ 9. ledger and tools
def test_next_barcode_stays_in_the_strip_quick_spec_ends_after_strip_1():
    tb, _ = planned()
    r = tb.next_barcode()
    d = r["data"]
    assert d["kind"] == "cell" and d["row_from_bottom"] == 4 and d["new_row"] and d["strip"]["strip"] == 1
    assert d["strip"]["rows"] == [{"row_from_bottom": r_, "boxes": 2, "scanned": 0, "handled": 0} for r_ in (4, 3, 2)]
    assert "a new row" in r["reason"] and r["diagnostics"]["row_mates"]
    mate = r["diagnostics"]["row_mates"][0]
    assert mate not in r["reason"]
    for b in tb.ws.scan_plan:
        tb.ws.barcodes[b].status = "scanned"
    r = tb.next_barcode()
    assert r["data"]["done"] and not r["data"].get("strip_done") and r["data"]["strip"]["done"]


def _full(spec=SPEC_F):
    tb, r = planned(spec=spec)
    assert r["status"] == "ok", r["reason"]
    walks, rereg = [], []
    tb._walk_to = lambda pos, yaw: (walks.append((np.asarray(pos, float).copy(), yaw)), [{"status": "ok"}])[1]

    def rr():
        rereg.append(1)
        return {"refit": True, "anchored": [], "ds_m": None, "delta_lat": 0.02}
    tb._reregister_at_stance = rr
    return tb, walks, rereg


def test_strip_done_then_go_to_next_strip_one_step_from_the_reference():
    tb, walks, rereg = _full()
    tb.next_barcode()
    assert tb.go_to_next_strip()["reason"].startswith("strip 1 still has unfinished boxes")
    for b in tb.ws.scan_plan:
        if tb._entry_strip(tb.ws.barcodes[b]) == 1:
            tb.ws.barcodes[b].status = "no_label"
    tb._stance_attempted.add(tb.ws.current_barcode_id)
    tb.ws.barcodes[tb.ws.current_barcode_id].status = "scanned"
    assert "has not been reported finished" in tb.go_to_next_strip()["reason"]
    r = tb.next_barcode()
    d = r["data"]
    assert d["strip_done"] and d["next_strip"] == 2 and not d["done"] and "barcode_id" not in d
    assert "call go_to_next_strip" in r["reason"] and "6 of 6 boxes handled" in r["reason"]
    ref = tb.ws.strips["strips"][1]["ref_w"]
    g = tb.go_to_next_strip()
    assert g["status"] == "ok", g["reason"]
    assert len(walks) == 1 and len(rereg) == 1 and tb.ws.strips["current"] == 2 and tb.ws.strips["done_reported"] is None
    target_w = apply(inv_T(tb._wall_T()), np.r_[walks[0][0], 0.0][None])[0]
    assert target_w[:2] == pytest.approx([ref[0] + 0.4, ref[1]], abs=1e-9)
    assert tb.ws.strips["strips"][2]["ref_w"] == pytest.approx([ref[0] + 0.4, ref[1]])
    assert "at strip 2 of 3 (columns 3-4)" in g["reason"]
    assert tb.next_barcode()["data"]["barcode_id"] and cells_of(tb)[6] == "C3R4"


def test_go_to_next_strip_crosses_empty_strips_in_steps_with_a_yawed_frame():
    spec = {"columns_from_left": [1, 2, 5, 6], "rows_from_bottom": [4], "order": "strips", "strip_columns": 2}
    tb, walks, rereg = _full(spec)
    a = math.radians(3.0)                                                  # the registered frame yawed in odom
    T = tb._wall_T()
    R = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1.0]])
    T[:3, :3] = R @ T[:3, :3]
    tb.ws.wall["T_odom_wall"] = T
    for b in tb.ws.scan_plan:
        if tb._entry_strip(tb.ws.barcodes[b]) == 1:
            tb.ws.barcodes[b].status = "scanned"
    assert tb.next_barcode()["data"]["next_strip"] == 3
    ref = tb.ws.strips["strips"][1]["ref_w"]
    g = tb.go_to_next_strip()
    assert g["status"] == "ok" and len(walks) == 2 and len(rereg) == 2 and tb.ws.strips["current"] == 3
    for k, (pos, yaw) in enumerate(walks):
        pw = apply(inv_T(tb._wall_T()), np.r_[pos, 0.0][None])[0]
        assert pw[:2] == pytest.approx([ref[0] + 0.4 * (k + 1), ref[1]], abs=1e-9)
        assert yaw == pytest.approx(math.atan2(T[1, 1], T[0, 1]))           # facing the (yawed) wall


def test_finish_refused_while_a_later_strip_has_requested_boxes():
    tb, _, _ = _full()
    c = config.load_all()
    h = Harness(tb, 3, c["task3"]["budget"], FakeLog(), c["robot"]["limits"])
    for b in tb.ws.scan_plan:
        tb.ws.barcodes[b].status = "scanned"
    tb.ws.barcodes[tb.ws.scan_plan[-1]].status = "pending"
    tb.ws.strips["current"] = 1
    assert tb._next_strip() == 3
    r = h.call("finish", {"task": 3, "success": True, "summary": "x"})
    assert r["status"] == "fail" and "go_to_next_strip" in r["reason"]
    h.call("finish", {"task": 3, "success": False, "summary": "x"})
    assert h.finished["success"] is False


# ------------------------------------------------------------------ 10. harness bookkeeping
def test_harness_strip_done_reply_and_member_bookkeeping_before_the_status_gate():
    tb, _, _ = _full()
    c = config.load_all()
    h = Harness(tb, 3, c["task3"]["budget"], FakeLog(), c["robot"]["limits"])
    assert h.tools == [n for n in TASK_TOOLS[3] if n != "go_to_next_face"] + LAYOUT_ONLY + STRIP_ONLY
    for b in tb.ws.scan_plan:
        if tb._entry_strip(tb.ws.barcodes[b]) == 1:
            tb.ws.barcodes[b].status = "scanned"
    h.fail_streak = {"move_arm": 2}
    r = h.call("next_barcode", {})
    assert r["data"]["strip_done"] and h.fail_streak.get("move_arm") == 2
    names = [s[0] for s in h.log.stages]
    assert "strip_end" in names and "barcode_start" not in names
    r = h.call("go_to_next_strip", {})
    assert r["status"] == "ok" and h.log.stages[-1][0] == "strip_start"
    per = c["task3"]["budget"]["per_barcode"]
    b0 = dict(h.budget)
    mid = tb.ws.scan_plan[7]

    def sc(barcode_id=None):
        tb.ws.barcodes[mid].status = "no_label"
        return tbmod.with_diagnostics(tbmod.partial("not covered", {"barcode_id": tb.ws.scan_plan[6]}),
                                      {"cells": [{"barcode_id": mid, "outcome": "none", "barcode_status": "no_label"},
                                                 {"barcode_id": "bcX", "outcome": "found", "added": ["bc99"]}]})
    tb.search_cell = sc
    h.call("search_cell", {})
    assert ("barcode_end", {"id": mid, "status": "no_label", "content": None}) in h.log.stages
    assert h.budget["max_tool_calls"] == b0["max_tool_calls"] + per["max_tool_calls"]


# ------------------------------------------------------------------ 11 / 13. stances: in place, shared, single (IK stub)
def stub_stances(tb, ok_cells=None, robust=True, reach=0.30):
    """Stance evaluation stub: a job is IK-feasible when its target lies within `reach` laterally of the root and 0.25-0.75
    m in front of it (and its barcode is in ok_cells, when given); manipulability falls with the lateral offset."""
    seen = []

    def evalj(jobs, body):
        out = []
        for j in jobs:
            Tc, t = j["Tc"], j["t"]
            v = inv_T(Tc) @ np.r_[np.asarray(t["pos"], float), 1.0]
            bid = t.get("barcode_id")
            good = 0.0 <= v[0] <= 0.5 and abs(v[1]) <= reach and (ok_cells is None or bid in ok_cells)
            seen.append(j)
            out.append({"status": "ok", "sols": [{"q_full": None}], "gap": 0.0, "margin": 0.1, "man": 1.0 - abs(v[1])}
                       if good else {"status": "ik"})
        return out
    tb._eval_stance_jobs = evalj
    tb._body_clearance = lambda body, sols, Tc, obs, poly: (1.0, 1.0, None)
    tb._stance_robust = lambda *a, **k: robust
    tb._body_model = _body(tb)
    tb.base_height_cmd = 0.74
    tb.arm_at_home = True
    return seen


def at(tb, x, y):
    T = T_from_xy_yaw(x, y, 0.0, 0.75)
    tb._odom = lambda: T


def first_cell(tb):
    tb.next_barcode()
    assert tb.compute_wrist_target()["status"] == "ok"
    return tb._current()


def test_plan_stance_in_place_for_the_row_then_approach_without_walking():
    tb, _ = planned()
    stub_stances(tb)
    at(tb, WALL_X - 0.6, 0.4)
    bc = first_cell(tb)
    mate = tb.ws.scan_plan[1]
    r = tb.plan_stance()
    st = tb.ws.stance_target
    assert r["status"] == "ok" and st["in_place"] == tb._walk_seq and st["serves"] == [bc.id, mate]
    assert set(st["look_alts"]) == {bc.id, mate} and st["level_index"] == 0 and "where the robot stands" in r["reason"]
    tb._walk_guard = lambda: (_ for _ in ()).throw(AssertionError("no walk"))
    a = tb.approach("stance_target")
    assert a["status"] == "ok" and "no walk needed" in a["reason"] and "call search_cell" in a["reason"]
    assert tb._stance_reached is st and bc.id in tb._stance_attempted


def test_plan_stance_shared_grid_stance_when_not_in_place_and_single_when_a_member_is_out_of_reach():
    tb, _ = planned()
    stub_stances(tb)
    at(tb, WALL_X - 0.6, -0.5)                                            # far right: nothing in place
    bc = first_cell(tb)
    mate = tb.ws.scan_plan[1]
    tb.plan_stance()
    st = tb.ws.stance_target
    assert "in_place" not in st and st["serves"] == [bc.id, mate]
    mid = 0.5 * (tb.ws.barcodes[mate].center[1] + bc.center[1])
    assert abs(st["pelvis_pos"][1] - mid) <= 0.30 + 1e-9                  # serves both: between them
    tb2, _ = planned()
    stub_stances(tb2, ok_cells={tb2.ws.scan_plan[0]})
    at(tb2, WALL_X - 0.6, -0.5)
    b2 = first_cell(tb2)
    tb2.plan_stance()
    st2 = tb2.ws.stance_target
    assert st2["serves"] == [b2.id] and "in_place" not in st2


def test_in_place_skips_robustness_and_goes_stale_after_a_walk():
    tb, _ = planned()
    stub_stances(tb, robust=False)
    at(tb, WALL_X - 0.6, 0.4)
    first_cell(tb)
    assert tb.plan_stance()["status"] == "ok" and tb.ws.stance_target["in_place"] == tb._walk_seq
    tb._walk_seq += 1                                                     # a walking tool ran
    r = tb.approach("stance_target")
    assert r["status"] == "fail" and "moved since this stance was planned" in r["reason"]
    tb3, _ = planned()
    stub_stances(tb3, robust=False)
    at(tb3, WALL_X - 0.6, -0.5)
    first_cell(tb3)
    r = tb3.plan_stance()                                                 # not in place, grid never robust
    assert r["status"] == "fail" and r["data"].get("unreachable")


def test_in_place_failed_placement_is_skipped_until_a_walk():
    tb, _ = planned()
    stub_stances(tb)
    at(tb, WALL_X - 0.6, 0.4)
    bc = first_cell(tb)
    tb.plan_stance()
    st = tb.ws.stance_target
    tb._record_failed_stance(bc, st)
    assert ("in_place", tb._walk_seq, 0) in {k[2:] for k in tb._failed_stances}
    tb.plan_stance()
    st2 = tb.ws.stance_target
    assert st2.get("in_place") is None or st2["level_index"] != 0


# ------------------------------------------------------------------ 12. search_cell over the served cells
def search_row(labels, scan_here=True):
    tb, _ = planned()
    stub_stances(tb)
    at(tb, WALL_X - 0.6, 0.4)
    bc = first_cell(tb)
    tb.plan_stance()
    st = tb.ws.stance_target
    tb._stance_reached = st
    cam = Cam(tb, labels, None, scan_here)
    return tb, bc, tb.ws.barcodes[st["serves"][1]], cam


def test_search_cell_searches_the_row_member_and_converts_both():
    x1, x2 = cell_x(1), cell_x(2)
    a = [x1[0] + 0.05, x1[0] + 0.15, 0.85, 0.9]
    b = [x2[0] + 0.05, x2[0] + 0.15, 0.85, 0.9]
    tb, bc, m, cam = search_row([a, b])
    r = tb.search_cell()
    assert r["status"] == "ok" and bc.kind == "label" and m.kind == "label" and m.status == "pending"
    assert r["diagnostics"]["cells"] == [{"barcode_id": m.id, "outcome": "found", "added": []}]
    assert m.id not in r["reason"] and "the next box of this row was also searched: label found" in r["reason"]
    assert m.id not in tb._stance_attempted and tb.ws.wrist_target["barcode_id"] == bc.id
    assert cam.homes == 1                                                 # arm home before the member's view


def test_search_cell_member_no_label_and_member_partial():
    x1 = cell_x(1)
    a = [x1[0] + 0.05, x1[0] + 0.15, 0.85, 0.9]
    tb, bc, m, cam = search_row([a])
    r = tb.search_cell()
    assert m.status == "no_label" and r["diagnostics"]["cells"][0]["barcode_status"] == "no_label"
    assert "no label in the box (recorded)" in r["reason"] and r["data"]["row_members"] == [
        {"outcome": "none", "barcode_status": "no_label"}]
    x2 = cell_x(2)
    big = [x2[0] - 0.5, x2[1] + 0.5, 0.80, 0.99]                         # wider than any view: the member stays a cell
    tb, bc, m, cam = search_row([a, big])
    r = tb.search_cell()
    assert m.kind == "cell" and m.status == "pending" and m.id not in tb._stance_attempted
    assert "it gets its own stance when it is current" in r["reason"]


# ------------------------------------------------------------------ 11b. failed-stance identity across re-registrations
def _refit(tb, shift, dyaw):
    """A re-registration of the frame (rotation dyaw about the odom origin, shift along the wall), entries re-synced."""
    T = T_from_xy_yaw(0.0, 0.0, dyaw, 0.0) @ np.array(tb.ws.wall["T_odom_wall"], float)
    T[:3, 3] += shift * T[:3, 0]
    tb.ws.wall["T_odom_wall"] = T
    tb._sync_layout_all()


def _replan(tb):
    assert tb.compute_wrist_target()["status"] == "ok"
    r = tb.plan_stance()
    assert r["status"] == "ok", r["reason"]
    return dict(tb.ws.stance_target)


def _same(a, b):
    return np.hypot(*np.subtract(a["pelvis_pos"], b["pelvis_pos"])) < 0.01 and a["level_index"] == b["level_index"]


@pytest.mark.parametrize("shift,dyaw", [(0.0, 0.0), (0.0003, 0.0), (0.0003, 0.0004), (0.001, -0.0007)])
def test_failed_grid_stance_stays_skipped_after_a_reregistration(shift, dyaw):
    """A grid look stance recorded failed (search_cell partial) is not planned again after the frame moved a little
    (rel(p) floats differ after a re-sync; the identity is the grid's (d, l), members and level)."""
    tb, _ = planned()
    stub_stances(tb)
    _refit(tb, 0.0, 0.003)
    at(tb, WALL_X - 0.6, -0.5)                                            # far right: a grid stance
    bc = first_cell(tb)
    st = _replan(tb)
    assert "in_place" not in st
    tb._record_failed_stance(bc, tb.ws.stance_target)
    _refit(tb, shift, dyaw)
    st2 = _replan(tb)
    assert not _same(st, st2) and st2["stance_key"] != st["stance_key"]


def test_failed_walked_stance_is_not_planned_again_in_place():
    """A walked grid stance the robot stands on fails: its in-place twin (same place, same level) is skipped too."""
    tb, _ = planned()
    stub_stances(tb)
    at(tb, WALL_X - 0.6, -0.5)
    bc = first_cell(tb)
    st = _replan(tb)
    assert "in_place" not in st and st["level_index"] == 0
    tb._walk_seq += 1                                                     # the walk to it (approach -> _walk_guard)
    T = T_from_xy_yaw(st["pos"][0], st["pos"][1], st["yaw"], 0.75)
    tb._odom = lambda: T
    tb._stance_reached = tb.ws.stance_target
    tb._record_failed_stance(bc, tb.ws.stance_target)
    assert ("in_place", tb._walk_seq, 0) in {k[2:] for k in tb._failed_stances}
    st2 = _replan(tb)
    assert not (st2.get("in_place") is not None and st2["level_index"] == st["level_index"])
    assert not _same(st, st2)


def test_member_left_partial_gets_another_stance_when_current():
    """search_cell leaves the row member partial from the shared look stance: when it is current, plan_stance does not
    give it the same place and level again (in place, no walk since)."""
    x1, x2 = cell_x(1), cell_x(2)
    a = [x1[0] + 0.05, x1[0] + 0.15, 0.85, 0.9]
    big = [x2[0] - 0.5, x2[1] + 0.5, 0.80, 0.99]
    tb, bc, m, cam = search_row([a, big])
    st = tb.ws.stance_target
    tb.search_cell()
    assert m.kind == "cell" and m.status == "pending" and m.id not in tb._stance_attempted
    assert (m.id, ("look",), "in_place", tb._walk_seq, st["level_index"]) in tb._failed_stances
    for b in tb.ws.scan_plan:
        if tb.ws.barcodes[b].box_id == bc.box_id:
            tb.ws.barcodes[b].status = "scanned"
    assert tb.next_barcode()["data"]["barcode_id"] == m.id
    st2 = _replan(tb)
    assert not (st2.get("in_place") is not None and st2["level_index"] == st["level_index"])


def test_label_target_identity_is_the_wall_frame_position_and_targets_follow_a_reregistration():
    """Strip mode: a label's failed-stance target identity is its wall-frame position (unchanged by a re-registration,
    which moves its odometry centre); the stored label wrist target is re-derived on the re-synced entry with its own
    stand-off; a look target is left alone."""
    x1, x2 = cell_x(1), cell_x(2)
    tb, bc, m, cam = search_row([[x1[0] + 0.05, x1[0] + 0.15, 0.85, 0.9], [x2[0] + 0.05, x2[0] + 0.15, 0.85, 0.9]])
    assert tb.search_cell()["status"] == "ok" and bc.kind == "label"
    t = dict(tb.ws.wrist_target)
    ident = tb._target_identity(bc, t)
    odom_key = tbmod._stance_target_key(t["pos"])
    _refit(tb, 0.004, 0.002)
    tb._resync_label_targets()
    t2 = tb.ws.wrist_target
    assert tb._target_identity(bc, t2) == ident and tbmod._stance_target_key(t2["pos"]) != odom_key
    assert t2["offset"] == t["offset"] and np.allclose(t2["pos"], bc.center + t["offset"] * bc.normal, atol=0, rtol=0)
    assert np.array_equal(t2["forward"], -bc.normal)
    look = tb._look_target(m)
    tb.ws.wrist_target = look
    tb._resync_label_targets()
    assert tb.ws.wrist_target is look


def test_plan_stance_reply_names_no_member_and_the_summary_serialises():
    """The members a strip stance serves are only in the diagnostics (design C1 P6); ws.summary() (look_head,
    result.json) serialises a strip stance and an approach-A look stance."""
    tb, _ = planned()
    stub_stances(tb)
    at(tb, WALL_X - 0.6, 0.4)
    bc = first_cell(tb)
    mate = tb.ws.scan_plan[1]
    r = tb.plan_stance()
    st = tb.ws.stance_target
    assert r["status"] == "ok" and st["serves"] == [bc.id, mate]
    assert mate not in json.dumps(r["data"]) and not {"serves", "look_alts", "target_pos_by", "stance_key"} & set(
        r["data"]["stance_odom"])
    assert r["diagnostics"]["stance_members"]["serves"] == [bc.id, mate]
    json.dumps(tb.ws.summary())
    a = kin_toolbox()
    a.ws.stance_target = {"barcode_id": "bc0", "purpose": "look", "look_alt": 0, "target_pos": ["look"],
                          "face_dist": 0.5, "lateral_offset": 0.0, "level_index": 0, "pos": [0.5, 0.0], "yaw": 0.0}
    assert json.loads(json.dumps(a.ws.summary()))["stance_target"]["target_pos"] == ["look"]


# ------------------------------------------------------------------ 14. re-registration triggers
def test_reregistration_triggers_in_strip_mode_only():
    tb, _ = planned()
    n = []
    tb._reregister_at_stance = lambda: (n.append(1), {"refit": True})[1]
    tb.robot.set_body_pose = lambda h, rpy, ramp, settle: {"status": "ok", "reason": "", "pelvis_z": h, "tilt_deg": 0.0}
    tb._body_model = _body(tb)
    tb.base_height_cmd = 0.74
    tb.arm_at_home = True
    r = tb.set_base_height_tool(0.45)                                     # the tool: re-registers
    assert r["status"] == "ok" and n == [1] and r["data"]["reregistration"] == {"refit": True}
    assert tb._walk_guard() is None and n == [1]                          # the guard's stand-up: no head capture
    a = kin_toolbox()                                                     # approach A: never
    a._reregister_at_stance = lambda: (_ for _ in ()).throw(AssertionError("no re-registration"))
    a.robot.set_body_pose = tb.robot.set_body_pose
    a._body_model = _body(a)
    a.base_height_cmd, a.arm_at_home = 0.74, True
    assert a.set_base_height_tool(0.45)["status"] == "ok"
    # a walked stance arrival re-registers (strip mode)
    tb._walk_guard = lambda: None
    tb.ws.stance_target = {"pos": [0.7, 0.4], "yaw": 0.0, "barcode_id": None, "base_height": 0.74}
    tb._approach = lambda target: tbmod.ok({"remaining": [0, 0, 0]})
    tb.approach("stance_target")
    assert n == [1, 1] and tb.ws.strips["strips"][1]["ref_source"] == "stance"


def test_not_current_text_in_strip_mode():
    tb, _ = planned()
    tb.next_barcode()
    other = tb.ws.scan_plan[3]
    r = tb.compute_wrist_target(other)
    assert r["status"] == "fail" and "finish the current box" in r["reason"] and "strip order" in r["reason"]


# ------------------------------------------------------------------ 15. prompts and tools
def _texts(monkeypatch, spec):
    from g1agent.orchestrators.agent_full import run_agent_full
    from test_task3_logic import ScriptedTB
    tb = ScriptedTB(FakeRobot(), plan_n=1)
    tb.cfg = config.apply_head_camera(config.apply_robot_variant(config.load_all(), "g1_v2"), "d435i")
    tb.set_scan_targets(spec)
    tb.set_head_views("upright")
    tb.set_phase1_view("obs_pose")
    tb.set_wall_layout(LAYOUT)
    monkeypatch.setattr(tb, "look_head", lambda: (tbmod.ok({}), np.zeros((4, 4, 3), np.uint8)))
    c = config.load_all()
    h = Harness(tb, 3, c["task3"]["budget"], FakeLog(), c["robot"]["limits"])
    seen = []

    class L(FakeLLM):
        def chat(self, messages, tools=None, tool_choice=None):
            seen.append(tools)
            return super().chat(messages, tools, tool_choice)
    llm = L([("finish", {"task": 3, "success": False, "summary": "x"})])
    run_agent_full(h, c["task3"], "all", llm, h.log)
    msgs = llm.seen[0]
    start = " ".join(p["text"] for p in msgs[1]["content"] if p.get("type") == "text")
    return msgs[0]["content"], start, h.log.llms[0]["prompt_version"], seen[0]


def test_strip_prompt_tools_and_version(monkeypatch):
    from g1agent.orchestrators import agent_full as af
    system, start, version, tools = _texts(monkeypatch, SPEC_Q)
    names = [t["function"]["name"] for t in tools]
    assert "go_to_next_strip" in names and "go_to_next_face" not in names and "search_cell" in names
    assert version.endswith("(wall layout, strips)") and "{" not in system
    assert af.PHASE2_LOOP_STRIP.format(w=2) in system and "If go_to_next_strip fails" in system
    assert "strips of 2 columns from the LEFT end" in start and "left side edge" in start
    desc = {t["function"]["name"]: t["function"]["description"] for t in tools}
    for n, note in af.STRIP_TOOL_NOTES.items():
        assert desc[n].endswith(note)
    for n, note in af.LAYOUT_TOOL_NOTES.items():
        assert note in desc[n]
    assert desc["plan_scans"].startswith("Task 3, after reaching the observation pose: register the wall seen from "
                                         "here to the given layout from its LEFT side edge")
    tb = bare_toolbox(FakeRobot())
    tb.wall_layout, tb.strip_columns = dict(LAYOUT), 2
    assert task_tools(3, tb) == [n for n in TASK_TOOLS[3] if n != "go_to_next_face"] + ["search_cell", "go_to_next_strip"]


# ------------------------------------------------------------------ 16. rule path
def test_rule_strip_episode_moves_to_the_next_strip_and_ends(monkeypatch):
    from test_wall_layout import LayoutRuleTB
    robot = FakeRobot({"bc0": [read("A")], "bc1": [read("B")]})
    tb = LayoutRuleTB(robot, {"bc0": "label", "bc1": "label"})
    tb.scan_targets = so.parse_targets(SPEC_F)
    tb.strip_columns, tb.n_strips = 2, 3
    plan0 = tb.plan_scans

    def plan_scans():
        r = plan0()
        for b, cell in zip(tb.ws.scan_plan, ([1, 4], [3, 4])):
            tb.ws.barcodes[b].wall = {"cell": cell, "left_to_right": True}
        tb.ws.strips = {"width": 2, "n": 3, "current": 1, "done_reported": None,
                        "strips": {k: {"columns": [2 * k - 1, 2 * k], "ref_w": [0.0, -0.6], "ref_source": "plan_scans"}
                                   for k in (1, 2, 3)}, "moves": []}
        return r
    tb.plan_scans = plan_scans
    moves = []

    def go():
        moves.append(tb.ws.strips["current"])
        tb.ws.strips["current"], tb.ws.strips["done_reported"] = 2, None
        return tbmod.ok({"strip": 2, "columns": [3, 4]}, "at strip 2 of 3 (columns 3-4), wall re-registered: call next_barcode")
    tb.go_to_next_strip = go
    c = config.load_all()
    h = Harness(tb, 3, c["task3"]["budget"], FakeLog(), c["robot"]["limits"])
    rule.Task3(h, c["task3"], "all").run()
    tools = [r["tool"] for r in h.log.tools]
    assert moves == [1] and tools.count("go_to_next_strip") == 1 and tools.count("scan") == 2
    assert h.finished["success"] is True
