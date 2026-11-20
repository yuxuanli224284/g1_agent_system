"""CPU tests of the task-3 logic (no simulator, no detector model, no GPU): scan order, adaptive stand-off, config
inheritance, plan_scans on a synthetic head frame, Harness bookkeeping (budget extension, stages, ledger success), the
rule Task3 flow on a scripted toolbox and the recovery-loop budget stop."""
import math

import numpy as np
import pytest

from g1agent.common import config
from g1agent.common.transforms import T_from_xy_yaw
from g1agent.harness.guards import Harness
from g1agent.harness.registry import TASK_TOOLS
from g1agent.harness.world_state import BarcodeEstimate, WorldState
from g1agent.orchestrators import rule
from g1agent.perception import geometry as geo
from g1agent.tools import scan_order as so
from g1agent.tools import toolbox as tbmod


# ------------------------------------------------------------------ pure rules
def test_column_order_matches_evaluator_copy():
    from g1agent.eval import scan_order as ev
    rng = np.random.default_rng(0)
    for _ in range(300):
        n = int(rng.integers(1, 7))
        items = []
        for k in range(n):
            y = float(rng.uniform(-0.6, 0.6)); w = float(rng.uniform(0.04, 0.2))
            items.append({"key": f"k{k}", "y_lo": y - w / 2, "y_hi": y + w / 2, "z": float(rng.uniform(0.7, 1.3))})
        assert so.column_order(items) == ev.column_order(items)


def test_column_order_columns_then_top_to_bottom():
    items = [{"key": "right", "y_lo": -0.35, "y_hi": -0.25, "z": 1.2},
             {"key": "left_low", "y_lo": 0.22, "y_hi": 0.32, "z": 0.9},
             {"key": "left_high", "y_lo": 0.25, "y_hi": 0.35, "z": 1.1}]
    assert so.column_order(items) == ["left_high", "left_low", "right"]
    # touching extents are separate columns (strict overlap)
    items = [{"key": "a", "y_lo": 0.0, "y_hi": 0.1, "z": 0.5}, {"key": "b", "y_lo": -0.1, "y_hi": 0.0, "z": 1.5}]
    assert so.column_order(items) == ["a", "b"]


def test_adaptive_standoff_nominal_and_clamp():
    hfov = so.hfov_from_intrinsics(1087.0, 640.0)
    assert math.degrees(hfov) == pytest.approx(32.8, abs=0.05)
    S = config.load("task3")["standoff"]
    d = so.adaptive_standoff(0.08, hfov, S["fill_frac"], S["min_m"], S["max_m"])
    assert d == pytest.approx(0.20, abs=0.002)                 # the nominal 0.08 m symbol -> the validated 0.20 m
    assert so.adaptive_standoff(0.01, hfov, S["fill_frac"], S["min_m"], S["max_m"]) == S["min_m"]
    assert so.adaptive_standoff(1.0, hfov, S["fill_frac"], S["min_m"], S["max_m"]) == S["max_m"]


def test_quad_width_and_point_in_quad():
    q = np.array([[0, 0, 0], [0.12, 0, 0], [0.12, 0.05, 0], [0, 0.05, 0]], float)
    assert so.quad_width(q) == pytest.approx(0.12)
    sq = [[0, 0], [10, 0], [10, 10], [0, 10]]
    assert so.point_in_quad([5, 5], sq) and so.point_in_quad([0, 5], sq) and not so.point_in_quad([11, 5], sq)


# ------------------------------------------------------------------ config
def test_task3_config_inherits_task1_and_task2():
    c = config.load_all()
    t1, t2, t3 = c["task1"], c["task2"], c["task3"]
    assert t3["approach"] == t1["approach"] and t3["arrive"] == t1["arrive"]
    assert t3["stance"] == t2["stance"] and t3["offset_m"] == t2["offset_m"] and t3["scan"] == t2["scan"]
    assert t3["standoff"]["mode"] == "adaptive" and t2["standoff"]["mode"] == "fixed"
    assert t3["budget"]["per_barcode"]["max_tool_calls"] > 0 and "inherits" not in t3


def test_config_inheritance_conflict_is_an_error():
    with pytest.raises(ValueError):
        config.merge_inherited({}, {"a": {"budget": 1}, "b": {"budget": 2}})
    assert config.merge_inherited({"budget": 3}, {"a": {"budget": 1}, "b": {"budget": 2}})["budget"] == 3


# ------------------------------------------------------------------ fakes
class FakeLog:
    def __init__(self):
        self.tools, self.stages, self.llms = [], [], []

    def tool(self, rec):
        self.tools.append(rec)

    def llm(self, rec):
        self.llms.append(rec)

    def stage(self, name, info=None):
        self.stages.append((name, info))

    def image(self, rgb, tag):
        return f"{tag}.jpg"


class FakeRobot:
    def __init__(self, reads=None):
        self.reads = reads or {}          # barcode id -> list of scan replies (consumed in order)
        self.tb = None

    def draw_markers(self, markers, frame="odom"):
        return {"status": "ok"}

    def get_odom(self):
        return {"T_odom_root": np.eye(4)}

    def scan(self):
        bid = self.tb.ws.current_barcode_id
        q = self.reads.get(bid, [])
        if q:
            return q.pop(0)
        return {"status": "no_read", "content": None, "decoded": [], "timestamp": 0.0}


def bare_toolbox(robot, task=3, variant="g1_v1"):
    """A Toolbox without models / kinematics: only the state the bookkeeping tools need (config of robot variant
    `variant`; the scanner field of view is the DS2208-like imager's on every variant)."""
    tb = tbmod.Toolbox.__new__(tbmod.Toolbox)
    tb.robot = robot
    tb.cfg = config.apply_head_camera(config.apply_robot_variant(config.load_all(), variant), "d435i")
    tb.ws = WorldState()
    tb.markers = {}
    tb.task = task
    tb.arm_at_home = True
    tb._last_label = None
    tb._wrist_detection = None
    tb.scanner_hfov = so.hfov_from_intrinsics(1087.0, 640.0)
    tb.head = None
    tb.body_cmd = None                  # walker without a body (squat) command unless a test sets one
    tb.base_height_cmd = None
    tb._body_model = None
    tb.multi_face = False
    tb.scan_targets = None               # every label, column order (Toolbox.set_scan_targets)
    tb.head_views = None                 # the camera model's lean views (Toolbox.set_head_views)
    tb.phase1_view = "start_pose"        # Toolbox.set_phase1_view
    tb._obs_region = None
    tb.view_pitch = 0.0                  # head view upright (torso lean, robot.yaml view)
    tb._leans = None
    tb._plan_update_from_here = lambda: None   # next_barcode's automatic look (needs the head camera); tested separately
    robot.tb = tb
    return tb


def plan_of(tb, n, width=0.08):
    tb.ws.barcodes, tb.ws.scan_plan = {}, []
    for k in range(n):
        bid = f"bc{k}"
        tb.ws.barcodes[bid] = BarcodeEstimate(bid, f"box{k}", np.array([1.0, 0.3 - 0.3 * k, 1.0]), np.array([-1.0, 0, 0]), None,
                                              "head", 0.0, face={"center": np.array([1.0, 0, 1.0]), "normal": np.array([-1.0, 0, 0]),
                                                                "right": np.array([0, -1.0, 0]), "up": np.array([0, 0, 1.0])},
                                              width_m=width, status="pending", order_index=k)
        tb.ws.scan_plan.append(bid)


def read(content):
    return {"status": "ok", "content": content, "decoded": [{"content": content, "polygon": [[1, 1], [5, 1], [5, 5], [1, 5]]}],
            "timestamp": 1.0}


# ------------------------------------------------------------------ scan ledger / next_barcode
def test_next_barcode_scan_dedupe_and_failed():
    robot = FakeRobot({"bc0": [read("A")], "bc1": [read("A")], "bc2": [read("C")]})
    tb = bare_toolbox(robot)
    plan_of(tb, 4)
    r = tb.next_barcode()
    assert r["data"]["barcode_id"] == "bc0" and r["data"]["index"] == 0 and r["data"]["standoff_m"] == pytest.approx(0.2, abs=0.002)
    tb.ws.wrist_target = {"pos": [0, 0, 0]}
    s = tb.scan()
    assert s["status"] == "ok" and s["data"]["new"] and s["data"]["scanned"] == ["A"] and s["data"]["polygons"][0]["content"] == "A"
    assert tb.next_barcode()["data"]["barcode_id"] == "bc1" and tb.ws.wrist_target is None      # per-barcode slots cleared
    s = tb.scan()
    assert s["data"]["duplicate_of"] == "A" and not s["data"]["new"] and tb.ws.barcodes["bc1"].status == "duplicate"
    tb.next_barcode()
    tb.scan()
    r = tb.next_barcode()                                           # bc2 scanned -> bc3
    assert r["data"]["barcode_id"] == "bc3"
    assert tb.scan()["status"] == "fail"                            # no read
    r = tb.next_barcode()                                           # bc3 unresolved -> failed, plan exhausted
    assert r["data"]["done"] and r["data"]["closed"] == {"id": "bc3", "status": "failed"}
    assert [e["content"] for e in tb.ws.scanned] == ["A", "C"]
    assert [p["status"] for p in tb.ws.summary()["scan_plan"]] == ["scanned", "duplicate", "scanned", "failed"]


def test_compute_wrist_target_adaptive_and_clears_refined():
    tb = bare_toolbox(FakeRobot())
    plan_of(tb, 1, width=0.15)
    tb.next_barcode()
    tb.ws.wrist_target_refined = {"stale": True}
    r = tb.compute_wrist_target()
    S = config.load("task3")["standoff"]
    d = so.adaptive_standoff(0.15, tb.scanner_hfov, S["fill_frac"], S["min_m"], S["max_m"])
    assert r["data"]["offset"] == pytest.approx(d, abs=1e-3) and tb.ws.wrist_target_refined is None
    assert r["data"]["pre_offset"] == pytest.approx(d + 0.10, abs=1e-3)
    tb.compute_wrist_target(refined=True)                          # refined keeps the head target's stand-off
    assert tb.ws.wrist_target_refined["offset"] == pytest.approx(tb.ws.wrist_target["offset"])
    tb2 = bare_toolbox(FakeRobot(), task=2)                        # task 2 keeps the fixed 0.20 m
    plan_of(tb2, 1, width=0.15)
    tb2.ws.current_barcode_id = "bc0"
    assert tb2.compute_wrist_target()["data"]["offset"] == pytest.approx(0.20)


# ------------------------------------------------------------------ plan_scans on a synthetic head frame
def _head_frame():
    K = np.array([[200.0, 0, 320], [0, 200.0, 240], [0, 0, 1]])
    T_root_cam = np.eye(4)
    T_root_cam[:3, :3] = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)   # optical z = +x, x = -y, y = -z
    T_root_cam[:3, 3] = [0, 0, 1.2]
    return {"rgb": np.zeros((480, 640, 3), np.uint8), "depth": None, "K": K, "T_root_cam": T_root_cam,
            "T_odom_root": T_from_xy_yaw(0.5, -0.2, 0.0), "t": 3.0}


def _face(y0, y1, x=1.0):
    c = np.array([x, (y0 + y1) / 2, 1.0])
    corners = np.array([[x, y1, 1.2], [x, y0, 1.2], [x, y0, 0.8], [x, y1, 0.8]])
    return {"center": c, "normal": np.array([-1.0, 0, 0]), "right": np.array([0, -1.0, 0]), "up": np.array([0, 0, 1.0]),
            "corners": corners, "size": [y1 - y0, 0.4], "t": 3.0}


def test_plan_scans_orders_columns_and_dedupes(monkeypatch):
    tb = bare_toolbox(FakeRobot())
    h = _head_frame()
    T_odom_cam = h["T_odom_root"] @ h["T_root_cam"]
    # odom-frame scene: two boxes; left box (+y) has two stacked labels, right box one label
    faces = {"box0": _face(0.0, 0.4, x=1.5), "box1": _face(-0.6, -0.2, x=1.4)}
    labels = [("L_top", 0.2, 1.1), ("L_low", 0.21, 0.9), ("R", -0.4, 1.0)]

    def px(p):
        uv, z = geo.project(h["K"], np.linalg.inv(T_odom_cam), np.atleast_2d(p))
        return uv[0]

    recs = []
    for name, y, z in labels:
        q3 = [np.array([1.5 if y > -0.1 else 1.4, y + sy * 0.05, z + sz * 0.025]) for sy, sz in ((1, 1), (-1, 1), (-1, -1), (1, -1))]
        q = np.array([px(p) for p in q3])
        recs.append({"name": name, "kind": "label", "corners": q, "center": q.mean(axis=0), "score": 0.6, "bar_dir": np.array([1.0, 0.0])})
    dup = dict(recs[0], corners=recs[0]["corners"] + 1.0, center=recs[0]["center"] + 1.0, score=0.5)   # a second detection

    def fake_detect_boxes(prompt="cardboard box", view="auto"):
        tb.head = h
        tb.ws.boxes = {k: tbmod.BoxEstimate(k, [0, 0, 1, 1], 0.9, 3.0) for k in faces}
        return {"status": "ok", "data": {}, "reason": ""}

    def fake_fit(box_id):
        tb.ws.boxes[box_id].face = faces[box_id]
        return {"status": "ok", "data": {}, "reason": ""}

    def fake_rectified(frame, face):
        # labels of this face (the duplicate detection included), est on the plane, rec in head px
        out = []
        for name, y, z in labels + [("dup", 0.2, 1.1)]:
            xf = 1.5 if y > -0.1 else 1.4
            if abs(xf - face["center"][0]) > 1e-9:
                continue
            q3 = np.array([[xf, y + sy * 0.05, z + sz * 0.025] for sy, sz in ((1, 1), (-1, 1), (-1, -1), (1, -1))])
            rec = next(r for r in recs + [dup] if r["name"] == name) if name != "dup" else dup
            out.append((rec, {"center": q3.mean(axis=0), "bar_dir": np.array([0, -1.0, 0]), "corners": q3, "width": 0.10}))
        return out

    monkeypatch.setattr(tb, "detect_boxes", fake_detect_boxes)
    monkeypatch.setattr(tb, "_fit_one", fake_fit)
    monkeypatch.setattr(tb, "_labels_on_rectified_face", fake_rectified)
    tb.det = None
    r = tb.plan_scans()
    assert r["status"] == "ok", r
    bcs = r["data"]["barcodes"]
    assert len(bcs) == 3                                            # the duplicate detection is merged
    assert [b["box_id"] for b in bcs] == ["box0", "box0", "box1"]
    zs = [tb.ws.barcodes[b["id"]].center[2] for b in bcs]
    assert zs[0] > zs[1]                                            # top to bottom inside the left column
    for b in bcs:
        assert b["width_m"] == pytest.approx(0.10, abs=1e-3)
        # head width = the label detection box (0.10 m here); the stand-off rule is defined on the symbol, so the head
        # plan is conservative (farther) until the wrist measures the symbol
        assert b["standoff_m"] == pytest.approx(0.10 / (2 * math.tan(tb.scanner_hfov / 2)) / config.load("task3")["standoff"]["fill_frac"], abs=0.003)
    assert tb.ws.scan_plan == ["bc0", "bc1", "bc2"] and tb.ws.current_barcode_id is None
    tb.ws.barcodes["bc0"].status = "scanned"                       # once the plan runs, a call only updates it
    r2 = tb.plan_scans()
    assert r2["status"] == "ok" and r2["data"]["updated"] is True and r2["data"]["added"] == []
    assert tb.ws.scan_plan == ["bc0", "bc1", "bc2"] and tb.ws.barcodes["bc0"].status == "scanned"
    # a label seen for the first time (left column, below the others) is added among the pending ones, in order
    labels.append(("L_new", 0.19, 0.85))
    recs.append({"name": "L_new", "kind": "label", "corners": recs[1]["corners"] + 30.0, "center": recs[1]["center"] + 30.0,
                 "score": 0.6, "bar_dir": np.array([1.0, 0.0])})
    r3 = tb.plan_scans()
    assert [a["id"] for a in r3["data"]["added"]] == ["bc3"]
    assert tb.ws.scan_plan == ["bc0", "bc1", "bc3", "bc2"]         # bc3 lies below bc1 in the left column
    # a label whose estimate was later moved far away (bad refinement) is still recognised by its planned position:
    # it is not added again, so repeated plan updates cannot loop on one label
    tb.ws.barcodes["bc1"].center = tb.ws.barcodes["bc1"].center + np.array([0.0, 0.0, 0.6])
    r4 = tb.plan_scans()
    assert r4["data"]["added"] == [] and len(tb.ws.scan_plan) == 4


@pytest.mark.parametrize("multi_face", [True, False])
def test_stack_side_without_labels(monkeypatch, multi_face):
    """Multi-face tour: a side whose faces are fitted but carry no label is planned empty (ok, 0 barcodes) and the tour
    moves on (next_barcode -> face_done; finish(success) still needs the other sides). Single-face task: the same view
    is a failure (nothing to scan at all)."""
    tb = bare_toolbox(FakeRobot())
    tb.multi_face = multi_face
    h = _head_frame()
    faces = {"box0": _face(0.0, 0.4, x=1.5), "box1": _face(-0.4, 0.0, x=1.5)}

    def fake_detect_boxes(prompt="cardboard box", view="auto"):
        tb.head = h
        tb.ws.boxes = {k: tbmod.BoxEstimate(k, [0, 0, 1, 1], 0.9, 3.0) for k in faces}
        return {"status": "ok", "data": {}, "reason": ""}

    def fake_fit(box_id):
        tb.ws.boxes[box_id].face = faces[box_id]
        return {"status": "ok", "data": {}, "reason": ""}

    monkeypatch.setattr(tb, "detect_boxes", fake_detect_boxes)
    monkeypatch.setattr(tb, "_fit_one", fake_fit)
    monkeypatch.setattr(tb, "_labels_on_rectified_face", lambda frame, face: [])
    tb.det = None
    r = tb.plan_scans()
    if not multi_face:
        assert r["status"] == "fail" and "no barcode label" in r["reason"]
        assert tb.next_barcode()["status"] == "fail"
        return
    assert r["status"] == "ok" and r["data"]["n"] == 0 and r["data"]["first_for_face"] is True, r
    assert "nothing to scan" in r["reason"] and tb.ws.stack["planned"] == [0] and tb.ws.scan_plan == []
    nb = tb.next_barcode()
    assert nb["status"] == "ok" and nb["data"]["face_done"] is True and nb["data"]["next_face"] == "right"
    hh = _harness(tb)
    assert hh._task3_work_left() == "the right side of the stack has not been visited"
    tb.ws.stack["visited"] += [1, 2, 3]
    assert hh._task3_work_left() is None


@pytest.mark.parametrize("task,axis_off,expect", [(3, 0.01, "ok"), (3, 0.023, "partial"), (2, 0.023, "fail"),
                                                  (3, "pre", "fail")])
def test_placement_check_measures_in_task3(task, axis_off, expect):
    """Task 3: a placement a little outside the check tolerances is reported as partial with "call scan now" (the scan
    decides; 2026-11-07 the LLM re-walked 4 times on a 2.3 cm axis offset and never scanned). Task 2 keeps the fail:
    there the check is the success criterion."""
    tb = bare_toolbox(FakeRobot(), task=task)
    plan_of(tb, 1)
    bc = tb.ws.barcodes["bc0"]
    tb.ws.wrist_target = {"barcode_id": "bc0", "offset": 0.20, "pos": (bc.center - np.array([0.2, 0, 0])).tolist(), "forward": [1.0, 0, 0]}
    Tw = np.eye(4)
    Tw[:3, :3] = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])   # wrist forward (x) = +x = -normal
    Tw[:3, 3] = bc.center + (np.array([-0.35, 0.0, 0.0]) if axis_off == "pre" else np.array([-0.20, axis_off, 0.0]))
    tb._odom = lambda: np.eye(4)
    tb._q = lambda: None
    tb.kin = type("K", (), {"pose": lambda self, q, frame: Tw})()
    r = tb.check_wrist_placement()
    assert r["status"] == expect, r
    if expect == "partial":
        assert "call scan now" in r["reason"]


@pytest.mark.parametrize("miss,expect", [(0.03, "partial"), (0.21, "fail")])
def test_move_arm_partial_only_while_the_label_stays_in_the_scanner_view(miss, expect):
    """2026-11-07 stack4f_L0_0002: the raised arm stopped 0.21 m off, "partial: go on" was reported and the scan read
    nothing. A residual is partial only while the label centre stays inside the wrist camera's view cone (the scan /
    refine can recover); otherwise it is an IK failure (re-plan the stance)."""
    tb = bare_toolbox(FakeRobot(), task=2)
    tb.side = "left"
    tb.ws.wrist_target = {"pos": [0.8, 0.0, 1.0], "forward": [1.0, 0.0, 0.0], "offset": 0.2, "barcode_id": "bc0"}
    tb._odom = lambda: np.eye(4)
    tb._q = lambda: None
    tb.robot.set_arm_joints = lambda side, q, dur: {"status": "ok"}
    tb.robot.stand = lambda s: {"status": "ok"}
    tb.ik = type("IK", (), {"solve": lambda self, q, p, d, roll_axis=None: {"ok": True, "q_arm": np.zeros(7)}})()
    T = np.eye(4)
    T[:3, 3] = [0.8, miss, 1.0]                         # the arm always stops `miss` m to the side of the target
    tb.kin = type("K", (), {"pose": lambda self, q, frame: T})()
    tb._roll_in_root = lambda t: None
    r = tb.move_arm("final")
    assert r["status"] == expect, r
    half = 0.2 * math.tan(tb.scanner_hfov / 2)
    assert r["data"]["view"]["view_half_width_m"] == pytest.approx(half, abs=1e-3)
    if expect == "fail":
        assert r["reason"].startswith("IK")             # the callers' re-plan branch (prompt, rule orchestrator)


@pytest.mark.parametrize("miss,decoded,read_before,expect", [(0.01, [], None, "unreadable"),
                                                         (0.03, [], None, "pending"),      # 3 + 4 cm > 5.9 cm half-view
                                                         (0.10, [], None, "pending"),
                                                         (0.01, ["OTHER"], None, "pending"),
                                                         (0.01, [], "F4ATLFRVG", "pending"),
                                                         ("pre", [], None, "pending")])     # final pose not reached
def test_scan_records_a_label_without_readable_code(miss, decoded, read_before, expect):
    """2026-11-07 stack4c_L0_0001: 23 printed shipping stickers on the cardbox texture were planned as labels. A label
    that was inside the scanner view and gave no decodable code at all (after the retries) is "unreadable", a result,
    not a failure; out of view (placement problem) or with some other code decoded it stays a plain no-read."""
    tb = bare_toolbox(FakeRobot(), task=3)
    plan_of(tb, 1)
    tb.next_barcode()
    bc = tb.ws.barcodes["bc0"]
    bc.wrist_read = read_before
    tb.ws.wrist_target = {"barcode_id": "bc0", "pos": (bc.center - np.array([0.2, 0, 0])).tolist(), "forward": [1.0, 0, 0], "offset": 0.2}
    tb._odom = lambda: np.eye(4)
    tb._q = lambda: None
    T = np.eye(4)
    T[:3, 3] = bc.center - (np.array([0.35, 0.01, 0.0]) if miss == "pre" else np.array([0.2, miss, 0.0]))
    tb.kin = type("K", (), {"pose": lambda self, q, frame: T})()
    tb.robot.scan = lambda: {"status": "no_read", "decoded": [{"content": c} for c in decoded], "content": None}
    tb.refine_with_wrist = lambda barcode_id=None: tbmod.ok({})
    poses = []
    tb.move_arm = lambda pose="pre", side=None: (poses.append(pose), tbmod.ok({"history": [{"pos_err_m": 0.0}]}))[1]
    r = tb.scan()
    assert r["status"] == "fail" and bc.status == expect, (r, bc.status)
    # a label the wrist camera read during refine is scanned once more from the pre-pose, and is never "unreadable"
    assert ("pre" in poses) == (read_before is not None)
    if expect == "unreadable":
        assert "no readable code" in bc.fail_reason
        h = _harness(tb)
        bc.status = "pending"                                    # once more through the harness: its barcode_end stage
        tb.scan()
        h._after("scan", {}, {"status": "fail", "data": {}, "reason": "no code read"})
        assert ("barcode_end", "unreadable") in [(st[0], (st[1] or {}).get("status")) for st in h.log.stages]
        tb.ws.barcodes["bc1"] = BarcodeEstimate("bc1", "b", np.zeros(3), np.array([-1.0, 0, 0]), None, "head", 0.0,
                                                status="scanned", order_index=1, face_id=0)
        tb.ws.scan_plan.append("bc1")
        assert h._task3_work_left() is None                     # nothing pending: finish(success) allowed
        fin = h._finish({"task": 3, "success": True, "summary": "x"})
        assert fin["data"]["success"] is True                   # scanned + unreadable = every label handled


def test_next_barcode_looks_again_after_each_barcode_and_check_guides_to_final():
    """2026-11-07 stack4c_L0_0001 (v5): a label missed from the observation pose was never added (no plan update by
    the LLM), and a label read at the pre-pose was never scanned: the check told the LLM to re-plan the stance although
    the arm had only not been sent to 'final' yet."""
    tb = bare_toolbox(FakeRobot(), task=3)
    plan_of(tb, 2)
    looks = []
    tb._plan_update_from_here = lambda: looks.append(1) or {"status": "ok", "reason": "", "added": [], "diagnostics": None}
    tb.next_barcode()
    assert looks == []                                           # first barcode: the plan was just made
    bc = tb.ws.barcodes["bc0"]
    tb._stance_attempted.add("bc0")
    tb.ws.wrist_target = {"barcode_id": "bc0", "offset": 0.20, "pos": (bc.center - np.array([0.2, 0, 0])).tolist(), "forward": [1.0, 0, 0]}
    Tw = np.eye(4)
    Tw[:3, 3] = bc.center - np.array([0.35, 0.0, 0.0])          # arm at the pre-pose
    tb._odom = lambda: np.eye(4)
    tb._q = lambda: None
    tb.kin = type("K", (), {"pose": lambda self, q, frame: Tw})()
    tb._last_arm = {"pose": "pre", "barcode_id": "bc0", "status": "ok"}
    r = tb.check_wrist_placement()
    assert r["status"] == "fail" and "call move_arm('final')" in r["reason"]
    tb._last_arm = {"pose": "final", "barcode_id": "bc0", "status": "fail"}
    assert "re-plan the stance" in tb.check_wrist_placement()["reason"]
    r = tb.next_barcode()
    assert looks == [1] and r["data"]["plan_update"]["status"] == "ok"   # looked again before choosing bc1


def test_barcode_tools_work_on_the_current_barcode_only():
    """2026-11-07 stack4c_L0_0001 (v6): the LLM went on to bc8 without next_barcode and looped on bc7's stance."""
    tb = bare_toolbox(FakeRobot(), task=3)
    plan_of(tb, 2)
    r = tb.compute_wrist_target("bc0")
    assert r["status"] == "fail" and "call next_barcode" in r["reason"]           # nothing current yet
    tb.next_barcode()
    r = tb.compute_wrist_target("bc1")
    assert r["status"] == "fail" and "bc1 is not the current barcode (bc0)" in r["reason"]
    assert tb.refine_with_wrist("bc1")["status"] == "fail"


# ------------------------------------------------------------------ Harness bookkeeping
def _harness(tb, task=3):
    c = config.load_all()
    return Harness(tb, task, c[f"task{task}"]["budget"], FakeLog(), c["robot"]["limits"])


def test_harness_unknown_task_raises_and_tool_sets():
    c = config.load_all()
    with pytest.raises(ValueError):
        Harness(bare_toolbox(FakeRobot()), 4, c["task3"]["budget"], FakeLog(), c["robot"]["limits"])
    assert {"plan_scans", "next_barcode", "scan"} <= set(TASK_TOOLS[3])
    assert "scan" in TASK_TOOLS[2] and "plan_scans" not in TASK_TOOLS[2] and "scan" not in TASK_TOOLS[1]


def test_harness_budget_extension_stages_and_ledger(monkeypatch):
    robot = FakeRobot({"bc0": [read("A")], "bc1": [read("B")]})
    tb = bare_toolbox(robot)
    cfg_budget = config.load_all()["task3"]["budget"]
    h = _harness(tb)

    def fake_plan():
        plan_of(tb, 2)
        return {"status": "ok", "data": {"n": 2}, "reason": ""}

    monkeypatch.setattr(tb, "plan_scans", fake_plan)
    base = h.budget["max_tool_calls"]
    h.fail_streak["move_arm"] = 2
    h.call("plan_scans")
    per = cfg_budget["per_barcode"]
    assert h.budget["max_tool_calls"] == base + 2 * per["max_tool_calls"]
    assert h.budget["max_wall_s"] == cfg_budget["max_wall_s"] + 2 * per["max_wall_s"]
    assert config.load_all()["task3"]["budget"]["max_tool_calls"] == base          # config dict untouched
    h.call("next_barcode")
    assert h.fail_streak.get("move_arm", 0) == 0                                    # retries are per barcode
    h.call("scan")
    h.call("next_barcode")
    h.call("scan")
    h.call("next_barcode")
    h.n_calls = h.budget["max_tool_calls"]                                          # over budget: finish still works
    r = h.call("finish", {"task": 3, "success": False, "summary": "x"})
    assert r["status"] == "ok" and h.finished["success"] is True and h.finished["claimed_success"] is False
    names = [s[0] for s in h.log.stages]
    assert names == ["task1_end", "barcode_start", "barcode_end", "barcode_start", "barcode_end", "scan_done"]
    assert h.log.stages[2][1]["status"] == "scanned"


def test_finish_without_plan_emits_task1_end_and_fails():
    h = _harness(bare_toolbox(FakeRobot()))
    r = h.call("finish", {"task": 3, "success": True, "summary": "x"})       # a success claim without a plan: refused
    assert r["status"] == "fail" and "no scan plan yet" in r["reason"] and h.finished is None
    h.call("finish", {"task": 3, "success": False, "summary": "x"})
    assert h.finished["success"] is False and [s[0] for s in h.log.stages] == ["task1_end", "scan_done"]


# ------------------------------------------------------------------ rule Task3 on a scripted toolbox
class ScriptedTB(tbmod.Toolbox):
    """Real plan / ledger tools, scripted perception and motion."""

    def __init__(self, robot, plan_n=2):
        t = bare_toolbox(robot)
        self.__dict__.update(t.__dict__)
        robot.tb = self
        self.plan_n = plan_n
        self.calls = []

    def remember_home(self):
        self.arm_at_home = True

    def detect_boxes(self, prompt="cardboard box"):
        return tbmod.ok({"n": 2})

    def select_target_box(self, selector):
        self.ws.target_box_id = selector
        return tbmod.ok({"target_box_id": selector})

    def fit_front_face(self, box_id=None):
        return tbmod.ok()

    def compute_obs_pose(self, box_id=None):
        return tbmod.ok()

    def approach(self, target="root_target"):
        self.calls.append(("approach", target))
        if target == "root_target":                    # arrival check fails: the fast end goes on to plan_scans
            return tbmod.fail("not within tolerance after the fine moves", {"walk_done": True})
        return tbmod.ok()

    def plan_scans(self):
        if self.ws.scan_plan:                          # an update from a later viewpoint: nothing new here
            return tbmod.ok({"updated": True, "added": []})
        plan_of(self, self.plan_n)
        return tbmod.ok({"n": self.plan_n})

    def plan_stance(self, which="wrist_target"):
        return tbmod.ok()

    def move_arm(self, pose="pre", side=None):
        self.calls.append(("move_arm", pose))
        self.arm_at_home = pose == "home"
        return tbmod.ok({"pose": pose})

    def refine_with_wrist(self, barcode_id=None):
        return tbmod.ok()

    def stereo_check_wrist(self, barcode_id=None):
        return tbmod.ok()

    def check_wrist_placement(self):
        return tbmod.ok()


def test_rule_task3_flow():
    robot = FakeRobot({"bc0": [read("A")],
                       "bc1": [{"status": "no_read", "content": None, "decoded": [], "timestamp": 2.0}, read("B")]})
    tb = ScriptedTB(robot)
    h = _harness(tb)
    rule.Task3(h, config.load_all()["task3"], "all").run()
    tools = [r["tool"] for r in h.log.tools]
    assert tools.count("approach") == 3                             # phase-1 approach (no retry) + 2 stances
    assert tools.index("plan_scans") == tools.index("approach") + 1
    assert tools.count("scan") == 2                                 # one scan tool call per barcode ...
    sc1 = [r for r in h.log.tools if r["tool"] == "scan"][1]["result"]
    assert sc1["status"] == "ok" and len(sc1["data"]["retries"]) == 1   # ... bc1's no_read retried inside the tool
    assert h.finished["task"] == 3 and h.finished["success"] is True
    assert [e["content"] for e in tb.ws.scanned] == ["A", "B"]
    # the arm is home before each stance walk
    seq = [c for c in tb.calls]
    for i, c in enumerate(seq):
        if c == ("approach", "stance_target"):
            last_arm = [x for x in seq[:i] if x[0] == "move_arm"]
            assert not last_arm or last_arm[-1] == ("move_arm", "home")


def test_task_class_dispatch():
    assert rule.task_class(3) is rule.Task3
    with pytest.raises(ValueError):
        rule.task_class(5)


def test_recover_stops_when_budget_is_exhausted():
    class H:
        def __init__(self):
            self.n = 0

        def over_budget(self):
            return "budget" if self.n >= 10 else None

        def call(self, name, args=None):
            self.n += 1
            if self.n >= 10:
                return {"status": "fail", "data": {}, "reason": "budget", "budget_exhausted": True}
            return {"status": "fail", "data": {}, "reason": "x"}

    h = H()
    always_explore = lambda h_, tool, r, attempt: {"explore": {"direction": "left", "distance": 0.1}}
    r = rule.recover(h, always_explore, "approach", {}, h.call("approach"))
    assert r["status"] == "fail" and h.n <= 11


def test_arrival_error_along_normal_and_lateral():
    tb = bare_toolbox(FakeRobot())
    tgt = {"pos": [1.0, 0.0], "yaw": 0.0}
    tb._odom = lambda: T_from_xy_yaw(1.02, 0.04, math.radians(1.0))
    along, lateral, dyaw = tb._arrival_error(tgt)
    assert along == pytest.approx(0.02) and lateral == pytest.approx(0.04) and math.degrees(dyaw) == pytest.approx(1.0)


def test_move_summary_drops_ground_truth_fields():
    r = {"status": "ok", "reason": "x", "elapsed_s": 1.0, "requested": [0, 0, 0], "odom_displacement": [0, 0, 0],
         "gt_displacement": [1, 1, 1], "gt_error": {"pos_m": 0.1}, "odom_error_vs_gt": {"pos_m": 0.0}}
    assert set(tbmod.move_summary(r)) == {"status", "reason", "elapsed_s", "requested", "odom_displacement"}


# ------------------------------------------------------------------ agent_full task 3: history compaction
class FakeLLM:
    def __init__(self, script):
        self.script, self.seen = list(script), []

    def image_part(self, rgb):
        return {"type": "image_url", "image_url": {"url": "data:"}}

    def chat(self, messages, tools=None, tool_choice=None):
        self.seen.append(messages)
        name, args = self.script.pop(0)
        tc = {"id": f"c{len(self.seen)}", "name": name, "arguments": args, "arguments_raw": "", "parse_error": None}
        am = {"role": "assistant", "content": f"I call {name}.",
              "tool_calls": [{"id": tc["id"], "type": "function", "function": {"name": name, "arguments": "{}"}}]}
        return {"content": am["content"], "reasoning_content": None, "tool_calls": [tc], "finish_reason": "tool_calls",
                "usage": None, "assistant_message": am, "t_start": 0.0, "t_end": 0.1}


def test_agent_full_task3_compacts_history_at_plan_scans(monkeypatch):
    from g1agent.orchestrators.agent_full import run_agent_full
    robot = FakeRobot({"bc0": [read("A")]})
    tb = ScriptedTB(robot, plan_n=1)
    monkeypatch.setattr(tb, "look_head", lambda: (tbmod.ok({}), np.zeros((4, 4, 3), np.uint8)))
    h = _harness(tb)
    llm = FakeLLM([("detect_boxes", {}), ("plan_scans", {}), ("next_barcode", {}), ("scan", {}), ("next_barcode", {}),
                   ("finish", {"task": 3, "success": True, "summary": "done"})])
    run_agent_full(h, config.load_all()["task3"], "all", llm, h.log)
    after_plan = llm.seen[2]                       # the request right after plan_scans
    assert len(after_plan) == 2 and after_plan[0]["role"] == "system" and "Phase 1 is finished" in after_plan[1]["content"]
    assert any(r.get("kind") == "history_compaction" for r in h.log.llms)
    assert h.finished["success"] is True and [e["content"] for e in tb.ws.scanned] == ["A"]


class SquatTB(ScriptedTB):
    """ScriptedTB whose stance for bc1 needs a squat (plan_stance base_height below standing)."""

    def __init__(self, robot):
        super().__init__(robot)
        self.body_cmd = {"height_default": 0.74, "height_range": [0.2, 0.74], "rpy_limit_rad": [0.4, 0.6, 0.8]}
        self.base_height_cmd = 0.74

    def plan_stance(self, which="wrist_target"):
        h = 0.45 if self.ws.current_barcode_id == "bc1" else 0.74
        self.ws.stance_target = {"pos": [0.5, 0.0], "yaw": 0.0, "barcode_id": self.ws.current_barcode_id, "base_height": h}
        return tbmod.ok({"stance_odom": self.ws.stance_target, "squat": h < 0.74})

    def set_base_height(self, height=None):
        height = 0.74 if height is None else float(height)
        self.calls.append(("set_base_height", height))
        self.base_height_cmd = height
        return tbmod.ok({"height_cmd": height})

    def approach(self, target="root_target"):
        g = self._walk_guard()
        if g is not None:
            self.calls.append(("approach_refused", target))
            return g
        return super().approach(target)


def test_rule_squats_after_walking_and_stands_up_with_the_arm_home():
    robot = FakeRobot({"bc0": [read("A")], "bc1": [read("B")]})
    tb = SquatTB(robot)
    h = _harness(tb)
    rule.Task3(h, config.load_all()["task3"], "all").run()
    assert h.finished["success"] is True
    seq = tb.calls
    assert ("approach_refused", "stance_target") not in seq
    i_sq = seq.index(("set_base_height", 0.45))
    assert seq[i_sq - 1] == ("approach", "stance_target")          # squat right after walking to the stance
    i_up = seq.index(("set_base_height", 0.74))
    assert i_up > i_sq and seq[i_up - 1] == ("move_arm", "home")    # stand up only with the arm at home
    assert tb.base_height_cmd == 0.74                               # the episode ends standing


def test_walking_while_squatting_stands_up_first_or_refuses_with_the_arm_raised():
    tb = SquatTB(FakeRobot())
    tb.base_height_cmd = 0.45
    tb.arm_at_home = False
    r = tb.explore("forward", 0.2)                                   # arm raised: refused, nothing moves
    assert r["status"] == "fail" and "move_arm('home')" in r["reason"] and tb.calls == []
    tb.arm_at_home = True
    assert tb._walk_guard() is None                                  # arm home: stands up, then may walk
    assert tb.calls == [("set_base_height", 0.74)] and tb.base_height_cmd == 0.74


def test_rectified_face_maps_detections_to_the_plane(monkeypatch):
    """A detection in the rectified face image maps back to the right point / width on the face plane."""
    tb = bare_toolbox(FakeRobot())
    h = _head_frame()
    face = _face(0.0, 0.4, x=1.5)                       # y 0..0.4, z 0.8..1.2; camera depth to the face 1.0 m
    seen = {}

    def fake_dino(rgb, roi, det):
        Hr, Wr = rgb.shape[:2]
        s_ = Wr / 0.4                                   # px per metre of the rectified face
        seen["s"], seen["size"] = s_, (Wr, Hr)
        # label centred at y = 0.3, z = 1.1, 0.10 x 0.05 m: rect x = (0.2 - (y - 0.2)) ... u axis = -y
        cx, cy = (0.2 - (0.3 - 0.2)) * s_, (0.2 - (1.1 - 1.0)) * s_
        q = np.array([[cx - 0.05 * s_, cy - 0.025 * s_], [cx + 0.05 * s_, cy - 0.025 * s_], [cx + 0.05 * s_, cy + 0.025 * s_],
                      [cx - 0.05 * s_, cy + 0.025 * s_]])
        sym = None if not seen.get("with_symbol") else (q - q.mean(axis=0)) * [0.8, 0.6] + q.mean(axis=0)   # 8 cm code
        return [{"kind": "label", "corners": q if sym is None else sym, "center": q.mean(axis=0), "score": 0.6,
                 "bar_dir": np.array([1.0, 0.0]), "label_corners": q, "label_center": q.mean(axis=0), "symbol_corners": sym,
                 "refined_by_cv2": sym is not None}]

    monkeypatch.setattr(tbmod, "detect_label_dino", fake_dino)
    tb.det = None
    (rec, est), = tb._labels_on_rectified_face(h, face)
    assert seen["s"] == pytest.approx(200.0, rel=0.02)              # focal 200 px / nearest depth 1.0 m
    assert est["center"] == pytest.approx([1.5, 0.3, 1.1], abs=0.01)
    assert est["width"] == pytest.approx(0.10, abs=0.01) and tbmod._label_width(est) == pytest.approx(0.10, abs=0.01)
    assert abs(est["bar_dir"][1]) == pytest.approx(1.0, abs=1e-6)  # rect x axis = the face's lateral axis
    assert rec["symbol_corners"] is None
    # D5: with a cv2 symbol inside the label, the label extent (corners, centre, label width) stays the label box and
    # only the stand-off width is the symbol's
    seen["with_symbol"] = True
    (rec, est), = tb._labels_on_rectified_face(h, face)
    assert est["center"] == pytest.approx([1.5, 0.3, 1.1], abs=0.01)
    assert tbmod._label_width(est) == pytest.approx(0.10, abs=0.01) and est["width"] == pytest.approx(0.08, abs=0.01)
    assert rec["symbol_corners"] is not None and np.ptp(rec["corners"][:, 0]) > np.ptp(rec["symbol_corners"][:, 0])


class UnreachableTB(SquatTB):
    """bc1 is too low for every body height: plan_stance reports it unreachable."""

    def plan_stance(self, which="wrist_target"):
        if self.ws.current_barcode_id == "bc1":
            bc = self.ws.barcodes["bc1"]
            reason = "barcode bc1 unreachable, too low: label centre at z=0.19 m; ..."
            bc.status, bc.fail_reason = "unreachable", reason
            return tbmod.fail(reason, {"unreachable": True, "barcode_id": "bc1", "label_z_m": 0.19})
        return super().plan_stance(which)


def test_unreachable_barcode_is_reported_logged_and_skipped():
    robot = FakeRobot({"bc0": [read("A")], "bc1": [read("B")], "bc2": [read("C")]})
    tb = UnreachableTB(robot)
    tb.plan_n = 3
    h = _harness(tb)
    rule.Task3(h, config.load_all()["task3"], "all").run()
    ends = [s for s in h.log.stages if s[0] == "barcode_end"]
    st = {i["id"]: i for _, i in ends}
    assert st["bc1"]["status"] == "unreachable" and "too low" in st["bc1"]["reason"]
    assert st["bc0"]["status"] == "scanned" and st["bc2"]["status"] == "scanned"       # the scan went on
    assert [e["content"] for e in tb.ws.scanned] == ["A", "C"]
    exp = {p["id"]: p for p in tb.ws.scan_plan_export()}
    assert exp["bc1"]["status"] == "unreachable" and exp["bc1"]["fail_reason"].startswith("barcode bc1 unreachable")
    assert h.finished["success"] is False


def test_same_planned_label_by_rectangle_overlap():
    """Two head estimates of one label 5 cm apart are the same label (the 3x3 stack duplicate, 2026-11-07); a label
    beside it (rectangles not overlapping) and a label on a surface one label width deeper are not."""
    def rect(y, z, x=1.3, w=0.10, h=0.05):
        return np.array([[x, y + w / 2, z + h / 2], [x, y - w / 2, z + h / 2], [x, y - w / 2, z - h / 2], [x, y + w / 2, z - h / 2]])
    b = BarcodeEstimate("bc1", "box", np.array([1.3, 0.02, 0.57]), np.array([-1.0, 0, 0]), None, "head", 0.0,
                        corners_odom=rect(0.02, 0.57), width_m=0.10, plan_center=np.array([1.3, 0.02, 0.57]))
    same = {"center": np.array([1.3, -0.03, 0.57]), "corners": rect(-0.03, 0.57), "width": 0.08}
    beside = {"center": np.array([1.3, -0.12, 0.57]), "corners": rect(-0.12, 0.57), "width": 0.10}
    deeper = {"center": np.array([1.45, -0.03, 0.57]), "corners": rect(-0.03, 0.57, x=1.45), "width": 0.10}
    assert tbmod._same_planned_label(same, b)
    assert not tbmod._same_planned_label(beside, b)
    assert not tbmod._same_planned_label(deeper, b)


# ------------------------------------------------------------------ multi-face (every side of a stack)
def test_face_ids_footprint_and_signed_distance():
    tb = bare_toolbox(FakeRobot())
    tb.ws.stack = {"n0": np.array([-1.0, 0, 0]), "faces": {}, "current": 0, "visited": [0], "multi_face": True}
    # front -x, right -y (robot's right while facing +x), back +x, left +y
    assert [tb._face_id(n) for n in ([-1, 0, 0], [0, -1, 0], [1, 0, 0], [0, 1, 0], [-0.9, -0.3, 0.1])] == [0, 1, 2, 3, 0]
    front = {"normal": [-1.0, 0, 0], "corners": np.array([[1.3, y, z] for y in (-0.4, 0.4) for z in (0.0, 1.14)])}
    right = {"normal": [0, -1.0, 0], "corners": np.array([[x, -0.4, z] for x in (1.3, 2.1) for z in (0.0, 1.14)])}
    tb._stack_add_faces(0, [front])
    assert tb._footprint() is None                                 # one side: no closed footprint yet
    assert tb._root_clearance([0.7, 0.0]) == pytest.approx(0.6)    # in front of the front plane
    tb._stack_add_faces(1, [right])
    poly = tb._footprint()
    assert poly is not None
    d = tbmod._poly_signed_distance(np.array([[0.7, 0.0], [2.7, 0.0], [1.7, 0.0], [1.7, -1.0]]), poly)
    # hull of the front + right corners: the triangle (1.3,-0.4) (1.3,0.4) (2.1,-0.4) until the back-left corner is seen;
    # in front / beside it outside (+), inside (-); behind it farther than 0.6 (the unknown corner is cut off)
    assert d[0] == pytest.approx(0.6) and d[1] > 0.6 and d[2] < 0 and d[3] == pytest.approx(0.6)


class FourFaceTB(ScriptedTB):
    """Real next_barcode / plan bookkeeping; scripted per-side plans and walking."""

    def __init__(self, robot, per_face=(2, 1, 2, 1)):
        super().__init__(robot)
        self.multi_face = True
        self.per_face = per_face
        self.ws.stack = {"n0": np.array([-1.0, 0, 0]), "faces": {0: {"normal": np.array([-1.0, 0, 0]), "corners": np.zeros((4, 3))}},
                         "current": 0, "visited": [0], "multi_face": True}

    def plan_scans(self):
        cur = self.ws.stack["current"]
        if any(self.ws.barcodes[b].face_id == cur for b in self.ws.scan_plan):
            return tbmod.ok({"updated": True, "added": []})
        for k in range(self.per_face[cur]):
            bid = f"f{cur}b{k}"
            self.ws.barcodes[bid] = BarcodeEstimate(bid, "box", np.array([1.0, 0.3 - 0.3 * k, 1.0]), np.array([-1.0, 0, 0]), None,
                                                    "head", 0.0, width_m=0.08, status="pending",
                                                    order_index=len(self.ws.scan_plan), face_id=cur)
            self.ws.scan_plan.append(bid)
        self.calls.append(("plan_scans", cur))
        return tbmod.ok({"n": self.per_face[cur], "face_id": cur, "first_for_face": True})

    def go_to_next_face(self):
        nf = self._next_face()
        self.calls.append(("go_to_next_face", nf))
        self.ws.stack["current"] = nf
        self.ws.stack["visited"].append(nf)
        return tbmod.ok({"face_id": nf})


def test_rule_visits_every_side_in_tour_order():
    reads = {f"f{f}b{k}": [read(f"C{f}{k}")] for f, n in enumerate((2, 1, 2, 1)) for k in range(n)}
    tb = FourFaceTB(FakeRobot(reads))
    h = _harness(tb)
    rule.Task3(h, config.load_all()["task3"], "all").run()
    walk = [c for c in tb.calls if c[0] in ("plan_scans", "go_to_next_face")]
    assert walk == [("plan_scans", 0), ("go_to_next_face", 1), ("plan_scans", 1), ("go_to_next_face", 2), ("plan_scans", 2),
                    ("go_to_next_face", 3), ("plan_scans", 3)]
    assert [e["content"] for e in tb.ws.scanned] == ["C00", "C01", "C10", "C20", "C21", "C30"]
    stages = [s[0] for s in h.log.stages]
    assert stages.count("face_start") == 4 and stages.count("face_end") == 4 and stages[-1] == "scan_done"
    assert h.finished["success"] is True


def test_single_face_task_never_walks_around():
    tb = FourFaceTB(FakeRobot({"f0b0": [read("A")], "f0b1": [read("B")]}))
    tb.multi_face = False
    tb.ws.stack["multi_face"] = False
    h = _harness(tb)
    rule.Task3(h, config.load_all()["task3"], "all").run()
    assert not any(c[0] == "go_to_next_face" for c in tb.calls)
    assert [e["content"] for e in tb.ws.scanned] == ["A", "B"]


def test_infrastructure_errors_end_the_episode_instead_of_a_tool_failure():
    tb = ScriptedTB(FakeRobot())
    h = _harness(tb)

    def oom():
        import torch
        raise torch.cuda.OutOfMemoryError("CUDA out of memory")
    tb.detect_boxes = lambda prompt="cardboard box", view="auto": oom()
    with pytest.raises(Exception, match="out of memory"):
        h.call("detect_boxes", {})
    tb.detect_boxes = lambda prompt="cardboard box", view="auto": (_ for _ in ()).throw(ValueError("bad box"))
    assert h.call("detect_boxes", {})["status"] == "fail"           # an ordinary tool error stays a structured failure


def test_next_barcode_refuses_to_skip_an_unattempted_barcode_and_offsets_are_ignored():
    tb = bare_toolbox(FakeRobot())
    plan_of(tb, 2)
    tb.next_barcode()
    r = tb.next_barcode()                                           # bc0 never attempted: refused, still current
    assert r["status"] == "fail" and "not been attempted" in r["reason"] and tb.ws.current_barcode_id == "bc0"
    w = tb.compute_wrist_target("bc0", 0.137)                       # an offset from the caller is ignored (adaptive)
    assert w["status"] == "ok" and w["data"]["offset_ignored"] == pytest.approx(0.137) and w["data"]["offset"] != pytest.approx(0.137)
    tb._stance_attempted.add("bc0")                                 # after an attempt, skipping is allowed
    r = tb.next_barcode()
    assert r["status"] == "ok" and r["data"]["closed"] == {"id": "bc0", "status": "failed"}


def test_label_dino_drops_enclosing_box(monkeypatch):
    """A detection enclosing another one is the carrying surface (box face), not a label; the looser of two nested
    boxes on one label is dropped too; disjoint labels are all kept."""
    from g1agent.perception import barcode_detector as bd

    class Det:
        cfg = {"box_threshold": 0.3}

        def detect(self, img, prompt):
            return [{"bbox": [100, 100, 400, 400], "score": 0.2},      # box face around label A
                    {"bbox": [150, 150, 210, 180], "score": 0.3},      # label A
                    {"bbox": [600, 100, 660, 130], "score": 0.4},      # label B
                    {"bbox": [595, 95, 665, 135], "score": 0.26}]      # looser duplicate of B
    monkeypatch.setattr(bd, "detect_barcodes", lambda *a, **k: [])
    img = np.zeros((300, 300, 3), np.uint8)
    cfg = {"upscale": 3, "label_threshold": 0.25, "label_prompt": "x", "label_max_area_frac": 0.5}
    out = bd.detect_label_dino(img, [0, 0, 300, 300], Det(), cfg)
    assert sorted(round(r["score"], 2) for r in out) == [0.3, 0.4]
    assert Det.cfg["box_threshold"] == 0.3


def test_agent_full_task3_compacts_history_at_every_next_barcode(monkeypatch):
    """The context holds one barcode's exchange at a time: after each successful next_barcode the request is the system
    prompt plus one user message (phase-1 summary + that next_barcode result with the scan plan)."""
    from g1agent.orchestrators.agent_full import run_agent_full
    robot = FakeRobot({"bc0": [read("A")], "bc1": [read("B")]})
    tb = ScriptedTB(robot, plan_n=2)
    monkeypatch.setattr(tb, "look_head", lambda: (tbmod.ok({}), np.zeros((4, 4, 3), np.uint8)))
    h = _harness(tb)
    llm = FakeLLM([("detect_boxes", {}), ("plan_scans", {}), ("next_barcode", {}), ("scan", {}), ("next_barcode", {}),
                   ("scan", {}), ("next_barcode", {}), ("finish", {"task": 3, "success": True, "summary": "done"})])
    run_agent_full(h, config.load_all()["task3"], "all", llm, h.log)
    for k in (3, 5, 7):                                          # the requests right after each next_barcode
        req = llm.seen[k]
        assert len(req) == 2 and req[0]["role"] == "system"
        assert "Phase 1 is finished" in req[1]["content"] and "Context compacted at next_barcode" in req[1]["content"]
    assert '"barcode_id": "bc1"' in llm.seen[5][1]["content"]     # the current barcode is in the kept result
    assert len(llm.seen[4]) == 4                                 # within a barcode the exchange accumulates
    kinds = [r.get("at") for r in h.log.llms if r.get("kind") == "history_compaction"]
    assert kinds == ["phase", "barcode", "barcode", "barcode"]
    assert h.finished["success"] is True and [e["content"] for e in tb.ws.scanned] == ["A", "B"]


def test_finish_success_refused_while_sides_remain_and_agent_full_keeps_the_task_statement(monkeypatch):
    """2026-11-07: after 2 of 4 sides the LLM called finish(success=true); the claim must agree with the plan and the
    tour. The refused claim does not end the episode; the compacted context still says every side is to be scanned."""
    from g1agent.orchestrators.agent_full import run_agent_full
    tb = FourFaceTB(FakeRobot({"f0b0": [read("A")]}), per_face=(1, 0, 0, 0))
    monkeypatch.setattr(tb, "look_head", lambda: (tbmod.ok({}), np.zeros((4, 4, 3), np.uint8)))
    h = _harness(tb)
    llm = FakeLLM([("plan_scans", {}), ("finish", {"task": 3, "success": True, "summary": "x"}),           # pending f0b0
                   ("next_barcode", {}), ("scan", {}),
                   ("finish", {"task": 3, "success": True, "summary": "x"}),                                # 3 sides left
                   ("next_barcode", {}), ("go_to_next_face", {}), ("plan_scans", {}), ("next_barcode", {}),
                   ("go_to_next_face", {}), ("plan_scans", {}), ("next_barcode", {}),
                   ("go_to_next_face", {}), ("plan_scans", {}), ("next_barcode", {}),
                   ("finish", {"task": 3, "success": True, "summary": "all sides"})])
    run_agent_full(h, config.load_all()["task3"], "all", llm, h.log)
    refused = [r for r in h.log.tools if r["tool"] == "finish" and r["result"]["status"] == "fail"]
    assert len(refused) == 2
    assert "f0b0 still pending" in refused[0]["result"]["reason"]
    assert "right side of the stack has not been visited" in refused[1]["result"]["reason"]
    assert "EVERY side of the box stack" in llm.seen[1][1]["content"]          # task statement kept after compaction
    assert h.finished["success"] is True and h.finished["summary"] == "all sides"


def test_harness_success_needs_the_whole_tour():
    tb = FourFaceTB(FakeRobot({"f0b0": [read("A")]}), per_face=(1, 0, 0, 0))
    h = _harness(tb)
    for name in ("plan_scans", "next_barcode", "scan"):
        h.call(name, {})
    h.call("finish", {"task": 3, "success": False, "summary": "give up"})       # giving up is always allowed
    assert h.finished is not None and h.finished["success"] is False           # every plan entry scanned, 3 sides unseen


def _side(stack_xy0, size, fid, M=None):
    """Fitted face of side fid of an axis-aligned square stack [x0, x0 + size] x [y0, y0 + size] (front = -x side),
    optionally seen through a drifted odometry M (2D rotation + translation applied to world points)."""
    x0, y0 = stack_xy0
    ends = {0: [(x0, y0), (x0, y0 + size)], 1: [(x0, y0), (x0 + size, y0)],
            2: [(x0 + size, y0), (x0 + size, y0 + size)], 3: [(x0, y0 + size), (x0 + size, y0 + size)]}[fid]
    normal = {0: (-1, 0), 1: (0, -1), 2: (1, 0), 3: (0, 1)}[fid]
    C = np.array([[*ends[0], 0.0], [*ends[1], 0.0], [*ends[1], 1.0], [*ends[0], 1.0]])
    n = np.array([*normal, 0.0])
    if M is not None:
        R, t = M
        C[:, :2] = C[:, :2] @ R.T + t
        n[:2] = R @ n[:2]
    return {"corners": C, "normal": n, "center": C.mean(axis=0)}


def test_stack_model_is_reregistered_to_each_fresh_side_under_odometry_drift():
    """The odometry rotates by 15 deg and shifts while the robot walks to the right side: the face directions, the
    footprint and the older sides follow the fresh fit, so the next side is predicted where it really is (in the
    drifted odometry), not where the front's old measurement puts it."""
    tb = bare_toolbox(FakeRobot())
    tb.multi_face = True
    tb.ws.stack = {"n0": np.array([-1.0, 0, 0]), "faces": {}, "current": 0, "visited": [0], "multi_face": True}
    tb._stack_add_faces(0, [_side((1.3, -0.4), 0.8, 0)])
    a = math.radians(15.0)
    M = (np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]]), np.array([0.07, -0.05]))
    tb._stack_add_faces(1, [_side((1.3, -0.4), 0.8, 1, M)])
    dirs = tb._face_dirs()
    for fid in range(4):
        assert dirs[fid] == pytest.approx(_side((1.3, -0.4), 0.8, fid, M)["normal"], abs=1e-9)
    front = tb._stack_face(0)["corners"]
    assert front[:, :2] == pytest.approx(_side((1.3, -0.4), 0.8, 0, M)["corners"][:, :2], abs=1e-9)
    hull = tb._footprint()                                     # the true footprint in the drifted odometry
    true = _side((1.3, -0.4), 0.8, 0, M)["corners"][:2, :2].tolist() + _side((1.3, -0.4), 0.8, 1, M)["corners"][1:2, :2].tolist()
    for p in true:
        assert min(np.linalg.norm(hull - np.array(p), axis=1)) < 1e-9
    reg = tb.ws.stack["registrations"][-1]
    assert reg["face"] == 1 and reg["dyaw_deg"] == pytest.approx(15.0, abs=1e-6)
    # a later refit of the same side under the same odometry changes nothing
    tb._stack_add_faces(1, [_side((1.3, -0.4), 0.8, 1, M)])
    assert tb._face_dirs()[2] == pytest.approx(dirs[2], abs=1e-9)
    assert tb.ws.stack["registrations"][-1]["dyaw_deg"] == pytest.approx(0.0, abs=1e-6)
    assert tb.ws.stack["registrations"][-1]["dxy_m"] == pytest.approx([0.0, 0.0], abs=1e-9)


def test_stance_obstacles_on_a_multi_side_stack_are_the_current_side_only():
    tb = bare_toolbox(FakeRobot())
    tb.ws.stack = {"n0": np.array([-1.0, 0, 0]), "faces": {}, "current": 1, "visited": [0, 1], "multi_face": True}
    for k, fid in enumerate((0, 1)):
        tb.ws.barcodes[f"b{k}"] = BarcodeEstimate(f"b{k}", "box", np.zeros(3), np.array([-1.0, 0, 0]), None, "head", 0.0,
                                                  face={"corners": np.zeros((4, 3)), "normal": np.array([1.0, 0, 0])},
                                                  face_id=fid)
    assert len(tb._known_faces()) == 1
    tb.ws.stack["multi_face"] = False
    assert len(tb._known_faces()) == 2


def test_arm_poses_need_the_planned_stance_reached_and_a_planned_stance_alone_is_no_attempt(monkeypatch):
    """2026-11-07: the LLM raised the arm right after plan_stance without walking there, then gave the barcode up.
    move_arm pre / final is refused until approach('stance_target') arrived at the current stance; a successful
    plan_stance alone does not let next_barcode close the barcode; any walking leaves the stance."""
    tb = bare_toolbox(FakeRobot())
    plan_of(tb, 2)
    tb.next_barcode()
    tb.side = "left"
    stance = {"pos": [0.5, 0.0], "yaw": 0.0, "barcode_id": "bc0", "base_height": 0.74}
    monkeypatch.setattr(tb, "_plan_stance", lambda which="wrist_target": (setattr(tb.ws, "stance_target", stance), tbmod.ok({}))[1])
    monkeypatch.setattr(tb, "_approach", lambda target: tbmod.ok({}))
    monkeypatch.setattr(tb, "_standing", lambda: True)
    r = tb.move_arm("pre")
    assert r["status"] == "fail" and "no stance planned for bc0" in r["reason"]
    assert tb.plan_stance()["status"] == "ok"
    r = tb.move_arm("pre")
    assert r["status"] == "fail" and "approach('stance_target')" in r["reason"]
    r = tb.next_barcode()
    assert r["status"] == "fail" and "not been attempted" in r["reason"]           # plan only: not an attempt
    assert tb.approach("stance_target")["status"] == "ok"
    r = tb.move_arm("pre")
    assert "stance" not in (r.get("reason") or "")                                 # past the guard (no wrist target)
    assert tb._walk_guard() is None                     # every walking tool (explore, approach, go_to_next_face) starts so
    assert "approach('stance_target')" in tb.move_arm("pre")["reason"]
    assert tb.next_barcode()["data"]["closed"] == {"id": "bc0", "status": "failed"}  # the walk there was an attempt


def test_the_same_failing_call_is_stopped_even_with_successes_in_between():
    """2026-11-07: move_arm('pre') failed 14x, each after a successful move_arm('home') that reset the per-tool streak."""
    tb = ScriptedTB(FakeRobot())
    h = _harness(tb)
    n = h.budget["max_retries_per_tool"]
    tb.move_arm = lambda pose="pre", side=None: tbmod.ok({}) if pose == "home" else tbmod.fail("no stance planned for bc7")
    reasons = []
    for _ in range(n + 2):
        h.call("move_arm", {"pose": "home"})
        reasons.append(h.call("move_arm", {"pose": "pre"})["reason"])
    assert "this exact move_arm call failed" in reasons[-1]


def test_stack_side_views_reach_the_known_stack_top():
    """A side fitted without its top row is looked at up to the stack's known top (other sides): views only."""
    tb = bare_toolbox(FakeRobot())
    C = np.array([[2.0, -0.4, 0.0], [2.0, 0.4, 0.0], [2.0, 0.4, 0.76], [2.0, -0.4, 0.76]])
    assert np.array_equal(tb._to_stack_top(C), C)                    # no stack model: unchanged
    tb.ws.stack = {"faces": {0: {"corners": np.array([[0, 0, 0.0], [0, 1, 0.0], [0, 1, 1.14], [0, 0, 1.14]])}}}
    out = tb._to_stack_top(C)
    assert out.shape == (6, 3) and np.allclose(sorted(out[:, 2])[-2:], [1.14, 1.14])
    assert np.allclose(out[4:, :2], C[2:, :2])                       # the top corners raised, same x / y
    tb.ws.stack["faces"][0]["corners"][:, 2] = [0, 0, 0.5, 0.5]      # stack lower than this side: unchanged
    assert np.array_equal(tb._to_stack_top(C), C)


def _v(variant):
    return config.apply_robot_variant(config.load_all(), variant)


def test_head_camera_profiles_follow_the_reported_model():
    """robot.yaml head_camera: the D435i gets stereo depth, its colour frame and lean views; the upstream single camera
    gets render depth, its own frame (d435_link origin, 15 mm right of the D435i colour camera) and no lean views
    (g1_v1, the only variant that can carry it)."""
    from g1agent.kinematics.fk import Kinematics
    d = config.apply_head_camera(_v("g1_v1"), "d435i")
    u = config.apply_head_camera(_v("g1_v1"), "upstream")
    assert (d["perception"]["head_depth"]["source"], d["robot"]["lean_views"]) == ("stereo", True)
    assert (u["perception"]["head_depth"]["source"], u["robot"]["lean_views"]) == ("render", False)
    kin = Kinematics(d["robot"]["urdf"])
    q = kin.q_from_state([], [])
    Td, Tu = kin.pose(q, d["robot"]["head_cam_frame"]), kin.pose(q, u["robot"]["head_cam_frame"])
    assert np.allclose(Td[:3, :3], Tu[:3, :3]) and abs(np.linalg.norm(Td[:3, 3] - Tu[:3, 3]) - 0.015) < 1e-6
    for bad in (None, "t800"):
        with pytest.raises(ValueError):
            config.apply_head_camera(_v("g1_v1"), bad)
    with pytest.raises(ValueError):                   # the upstream camera does not exist on g1_v2's mount
        config.apply_head_camera(_v("g1_v2"), "upstream")


def test_g1_v2_head_camera_is_the_variant_mount():
    """g1_v2: head_cam_optical (torso_link) = the variant's camera mount (URDF d435_joint, 26.15 deg pitch) x
    module_origin x the D435i colour position, ROS optical axes; at the neutral pose the axis is 26.15 deg below
    horizontal (the standing torso lean of 3.85 deg makes it 30 deg, preview.json)."""
    import math
    from g1agent.kinematics.fk import Kinematics
    cfg = config.apply_head_camera(_v("g1_v2"), "d435i")
    kin = Kinematics(cfg["robot"]["urdf"])
    q = kin.q_from_state([], [])
    T = np.linalg.inv(kin.pose(q, "torso_link")) @ kin.pose(q, cfg["robot"]["head_cam_frame"])
    from scipy.spatial.transform import Rotation
    p = 0.4564035994
    Rm = Rotation.from_euler("y", p).as_matrix()
    want = np.array([0.0586235, 0.01753, 0.46987]) + Rm @ (np.array([0.0258, -0.0025, 0.02]) + [0.0, 0.015, 0.0])
    assert np.allclose(T[:3, 3], want, atol=1e-6)
    assert np.allclose(T[:3, :3], Rm @ np.array([[0, 0, 1.0], [-1.0, 0, 0], [0, -1.0, 0]]), atol=1e-6)
    z = kin.pose(q, cfg["robot"]["head_cam_frame"])[:3, 2]
    assert math.degrees(math.asin(-z[2])) == pytest.approx(26.15, abs=1e-3)


def test_no_lean_views_without_them():
    """upstream camera: only the upright view is planned; the task-3 prompt has no lean-view sentence (v7 text)."""
    from g1agent.orchestrators import agent_full as af
    tb = bare_toolbox(FakeRobot())
    tb.cfg = config.apply_head_camera(_v("g1_v1"), "upstream")
    tb.body_cmd = {"height_default": 0.74, "height_range": [0.3, 0.8], "rpy_default": [0, 0, 0]}
    lv = tb._lean_levels()
    assert len(lv) == 1 and abs(lv[0]["pitch_cmd"]) < 1e-9
    tb.cfg = config.apply_head_camera(_v("g1_v1"), "d435i"); tb._leans = None
    assert len(tb._lean_levels()) > 1
    from g1agent.orchestrators.common import load_prompt
    kw = {"selector": "all", "max_tool_calls": 1, "max_explore_steps": 1, "per_barcode_calls": 1}
    with_note, _ = load_prompt("task3_agent_full", **kw, **af.task3_prompt_kw(tb, True))
    without, _ = load_prompt("task3_agent_full", **kw, **af.task3_prompt_kw(tb, False))
    assert "lean the torso back" in with_note and "lean the torso back" not in without


# ------------------------------------------------------------------ scan-room rows 2-4 design (D1, D4-D7)
def test_upright_head_views_keep_only_the_upright_level():
    """Scenario head_views "upright": _lean_levels is the pitch-0 level only although the camera model (d435i) has lean
    views; the effective flag is off; any other value is an error."""
    tb = bare_toolbox(FakeRobot(), variant="g1_v2")
    tb.body_cmd = {"height_default": 0.74, "height_range": [0.3, 0.8], "rpy_default": [0, 0, 0]}
    assert tb.cfg["robot"]["lean_views"] is True and tb.lean_views_on() and len(tb._lean_levels()) > 1
    tb.set_head_views("upright")
    lv = tb._lean_levels()
    assert len(lv) == 1 and abs(lv[0]["pitch_cmd"]) < 1e-9 and not tb.lean_views_on()
    tb.set_head_views(None)
    assert len(tb._lean_levels()) > 1
    with pytest.raises(ValueError):
        tb.set_head_views("leaned")
    with pytest.raises(ValueError):
        tb.set_phase1_view("walk")
    tb.set_phase1_view(None)
    assert tb.phase1_view == "start_pose"


class _LabelDet:
    cfg = {"box_threshold": 0.3}

    def detect(self, img, prompt):
        return [{"bbox": [300, 300, 600, 420], "score": 0.5}]          # upscale 3 -> label box x 100..200, y 100..140


@pytest.mark.parametrize("quad, accepted", [
    ([[95, 110], [150, 110], [150, 130], [95, 130]], False),          # partly left of the unpadded label box (padding only)
    ([[110, 105], [190, 105], [190, 135], [110, 135]], True)])        # inside: the symbol
def test_label_dino_keeps_the_label_box_and_takes_cv2_only_inside_it(monkeypatch, quad, accepted):
    from g1agent.perception import barcode_detector as bd
    q = np.asarray(quad, float)
    monkeypatch.setattr(bd, "detect_barcodes", lambda *a, **k: [{"kind": "barcode", "corners": q, "center": q.mean(axis=0),
                                                                 "bar_dir": np.array([0.0, 1.0]), "coherence": 0.9}])
    img = np.zeros((300, 300, 3), np.uint8)
    cfg = {"upscale": 3, "label_threshold": 0.25, "label_prompt": "x", "label_max_area_frac": 0.5, "roi_pad_frac": 0.05}
    (r,) = bd.detect_label_dino(img, [0, 0, 300, 300], _LabelDet(), cfg)
    box = np.array([[100, 100], [200, 100], [200, 140], [100, 140]], float)
    assert np.allclose(r["label_corners"], box) and np.allclose(r["label_center"], [150, 120])
    assert r["refined_by_cv2"] is accepted and r["cv2_quads_outside_label"] == (0 if accepted else 1)
    if accepted:
        assert np.allclose(r["symbol_corners"], q) and np.allclose(r["corners"], q)
    else:
        assert r["symbol_corners"] is None and np.allclose(r["corners"], box)


def _paper(y, z, x=1.3, w=0.106, h=0.038):
    """A scan-room paper label rectangle (0.106 x 0.038 m) on a -x face, consecutive corners."""
    return np.array([[x, y + w / 2, z + h / 2], [x, y - w / 2, z + h / 2], [x, y - w / 2, z - h / 2], [x, y + w / 2, z - h / 2]])


def test_paper_sized_label_boxes_are_one_label_and_neighbours_are_not():
    """D5/D6: with the label extent = the paper box, two estimates of one paper 1.6 cm apart overlap (the bc16 / bc3
    case); the neighbouring label of the next column (0.20 m) or row (0.25 m) does not."""
    n = np.array([-1.0, 0, 0])
    b = BarcodeEstimate("bc3", "box", np.array([1.3, 0.3, 0.98]), n, None, "head", 0.0, corners_odom=_paper(0.3, 0.98),
                        width_m=0.03, label_width_m=0.106, plan_center=np.array([1.3, 0.3, 0.98]))
    same = {"center": np.array([1.3, 0.3, 0.964]), "corners": _paper(0.3, 0.964)}
    assert tbmod._same_planned_label(same, b) and tbmod._same_est(same, {"corners": _paper(0.3, 0.98)}, n)
    for dy, dz in ((0.20, 0.0), (0.0, 0.25)):
        other = {"center": np.array([1.3, 0.3 - dy, 0.98 - dz]), "corners": _paper(0.3 - dy, 0.98 - dz)}
        assert not tbmod._same_planned_label(other, b) and not tbmod._same_est(other, same, n)
    # the 2.5 mm sliver at the seam that cv2 returned (bc14) would not overlap; its paper box does
    sliver = {"corners": _paper(0.3, 1.009, w=0.03, h=0.0025)}
    assert not tbmod._same_est(sliver, {"corners": _paper(0.3, 0.98, w=0.03, h=0.01)}, n)
    assert tbmod._same_est({"corners": _paper(0.3, 1.0)}, {"corners": _paper(0.3, 0.98)}, n)


def _one_face_scene(monkeypatch, tb, labels):
    """plan_scans on one fitted face (x 1.5, y -0.4..0.4, z 0.8..1.2) with the given head label estimates
    [(name, y, z, score)] (paper boxes), the detections projected into the synthetic head frame."""
    h = _head_frame()
    T_odom_cam = h["T_odom_root"] @ h["T_root_cam"]
    face = _face(-0.4, 0.4, x=1.5)

    def px(P):
        uv, _ = geo.project(h["K"], np.linalg.inv(T_odom_cam), np.atleast_2d(P))
        return uv

    def fake_detect_boxes(prompt="cardboard box", view="auto"):
        tb.head = h
        tb.ws.boxes = {"box0": tbmod.BoxEstimate("box0", [0, 0, 1, 1], 0.9, 3.0)}
        return {"status": "ok", "data": {}, "reason": ""}

    def fake_fit(box_id):
        tb.ws.boxes[box_id].face = face
        return {"status": "ok", "data": {}, "reason": ""}

    def fake_rectified(frame, f):
        out = []
        for name, y, z, score in sorted(labels, key=lambda l: -l[3]):
            q3 = _paper(y, z, x=1.5)
            q = px(q3)
            out.append(({"name": name, "kind": "label", "corners": q, "center": q.mean(axis=0), "score": score,
                         "refined_by_cv2": False, "symbol_corners": None},
                        {"center": q3.mean(axis=0), "bar_dir": np.array([0, -1.0, 0]), "corners": q3, "width": 0.106}))
        return out

    monkeypatch.setattr(tb, "detect_boxes", fake_detect_boxes)
    monkeypatch.setattr(tb, "_fit_one", fake_fit)
    monkeypatch.setattr(tb, "_labels_on_rectified_face", fake_rectified)
    tb.det = None
    return px


def test_plan_scans_drops_overlapping_detections_whose_centres_lie_outside_each_other(monkeypatch):
    """D6 (the bc4 / bc5 case): two detections of one paper label 7 cm apart laterally overlap on the face, but neither
    centre lies in the other's quad (the former pixel test kept both). One plan entry; the same for a plan update."""
    tb = bare_toolbox(FakeRobot())
    labels = [("A", 0.20, 1.10, 0.6), ("A2", 0.13, 1.10, 0.5)]
    px = _one_face_scene(monkeypatch, tb, labels)
    qa, qb = px(_paper(0.20, 1.10, x=1.5)), px(_paper(0.13, 1.10, x=1.5))
    assert not so.point_in_quad(qa.mean(axis=0), qb) and not so.point_in_quad(qb.mean(axis=0), qa)
    r = tb.plan_scans()
    assert r["status"] == "ok" and len(r["data"]["barcodes"]) == 1, r
    b0, d0 = r["data"]["barcodes"][0], r["diagnostics"]["plan_entries"][0]          # diagnostics: logged only (D11)
    assert d0["id"] == b0["id"] and d0["score"] == 0.6 and d0["label_width_m"] == pytest.approx(0.106, abs=1e-3)
    assert [d["matched"] for d in r["diagnostics"]["dropped_labels"]] == [b0["id"]]       # the higher score kept
    tb.ws.barcodes[b0["id"]].status = "scanned"                    # the plan runs: a call updates it
    labels += [("B", -0.20, 0.95, 0.6), ("B2", -0.27, 0.95, 0.5)]
    u = tb.plan_scans()
    assert u["data"]["updated"] and len(u["data"]["added"]) == 1 and len(tb.ws.scan_plan) == 2
    assert tb.ws.barcodes[u["data"]["added"][0]["id"]].label_width_m == pytest.approx(0.106, abs=1e-3)
    rules = [d["rule"] for d in u["diagnostics"]["dropped_labels"]]       # B2 (same face), A / A2 (the plan entry)
    assert sorted(rules) == ["same label as a plan entry"] + ["same label on this face (higher score kept)"] * 2


def _gate_setup(monkeypatch, jump):
    """Current barcode bc0 with a fragment stand-off width (0.03 m) and a paper label width (0.11 m); the hand camera
    detects a label `jump` m (lateral) from the estimate."""
    tb = bare_toolbox(FakeRobot())
    plan_of(tb, 2, width=0.03)
    for b in tb.ws.barcodes.values():
        b.label_width_m = 0.11
        b.face["corners"] = np.array([[1.0, 0.4, 1.2], [1.0, 0.2, 1.2], [1.0, 0.2, 0.8], [1.0, 0.4, 0.8]])
    tb.next_barcode()
    tb.compute_wrist_target()
    bc = tb.ws.barcodes["bc0"]
    h = _head_frame()
    h["T_odom_root"] = T_from_xy_yaw(0.0, 0.0, math.pi)          # looking away from the face: no plane refit
    monkeypatch.setattr(tb, "_capture_head", lambda: h)
    monkeypatch.setattr(tb, "_detect_wrist", lambda expect=None: ({"frame": 1}, {"center": [10.0, 10.0], "kind": "label"}))
    target = bc.center + np.array([0.0, -jump, 0.0])
    monkeypatch.setattr(tb, "_pixel_on_face", lambda frame, px_, plane: target.copy())
    monkeypatch.setattr(tb, "_wrist_symbol_width", lambda *a, **k: {"width_m": None})
    return tb, bc, target


def test_wrist_gate_uses_the_label_width_not_the_fragment_width(monkeypatch):
    """D7 (2026-11-19 calls 9/18/27/36): the correct label 6 cm away was rejected against a 3 cm fragment width. With the
    label's extent (0.11 m) it is accepted; a detection 15 cm away is still rejected, with the nearest other plan entry
    reported."""
    tb, bc, target = _gate_setup(monkeypatch, 0.06)
    r = tb.refine_with_wrist()
    assert r["status"] == "ok" and np.allclose(bc.center, target) and bc.source == "wrist"
    tb, bc, target = _gate_setup(monkeypatch, 0.15)
    r = tb.refine_with_wrist()
    assert r["status"] == "partial" and "not the planned label" in r["reason"]
    assert r["data"]["nearest_other_plan_entry"]["id"] == "bc1"


def test_wrist_target_reports_the_label_width_and_logs_the_standoff_width(monkeypatch):
    """Review 2026-11-19: compute_wrist_target's label_width_m is the label's extent (the same number plan_scans logs
    and the D7 gate uses), never the stand-off / cv2 fragment width, which is logged only (not shown to the LLM)."""
    from g1agent.orchestrators.common import tool_result_text
    tb, bc, _ = _gate_setup(monkeypatch, 0.06)
    r = tb.compute_wrist_target()
    assert r["data"]["label_width_m"] == pytest.approx(0.11) and r["diagnostics"]["standoff_width_m"] == pytest.approx(0.03)
    assert "standoff_width_m" not in tool_result_text(r)


def test_pending_targets_are_ordered_by_their_grid_cell():
    """D4: once the side's grid is resolved and every pending entry has a cell, the order follows the cells (rows from
    the top, left to right; a spec counted from the bottom is turned around over the row slots), not the label extents:
    a label fragment high in its box row is not ordered as a row of its own."""
    tb = bare_toolbox(FakeRobot())
    tb.set_scan_targets({"columns_from_left": [1, 2], "rows_from_bottom": [1, 2], "order": "rows"})
    tb.ws.scan_ref_T = np.eye(4)
    # bottom row (cell row 1) and the row above it (row 2); "frag" belongs to row 1 but its extent lies above r2's labels
    spec = {"r2c1": (0.3, 0.95, 2, 1), "r2c2": (0.1, 0.95, 2, 2), "r1c1": (0.3, 0.70, 1, 1), "frag": (0.1, 0.99, 1, 2)}
    for k, (y, z, row, col) in spec.items():
        tb.ws.barcodes[k] = BarcodeEstimate(k, "box", np.array([1.0, y, z]), np.array([-1.0, 0, 0]), None, "head", 0.0,
                                            corners_odom=_paper(y, z, x=1.0, h=0.004 if k == "frag" else 0.038),
                                            status="pending", grid_cell={"column": col, "row": row})
        tb.ws.scan_plan.append(k)
    tb._reorder_pending(0)
    assert tb.ws.scan_plan[0] == "frag"                              # no grid yet: the label rule (highest first)
    slot = lambda i, lo, hi: {"index": i, "lo": lo, "hi": hi}
    tb.ws.target_grid = {"faces": {0: {"columns": [slot(1, 0.2, 0.4), slot(2, 0.0, 0.2)],     # from the left
                                       "rows": [slot(1, 0.5, 0.75), slot(2, 0.75, 1.0)]}}}       # from the bottom
    tb._reorder_pending(0)
    assert tb.ws.scan_plan == ["r2c1", "r2c2", "r1c1", "frag"]
    tb.ws.barcodes["frag"].grid_cell = {"column": None, "row": 1}   # a late label no slot holds: back to the label rule
    tb._reorder_pending(0)
    assert tb.ws.scan_plan[0] == "frag"


def test_grid_evidence_bounds_the_floor_in_the_planning_frame():
    """D3: the floor (odom z 0) under the side's faces, in the planning frame (root at pelvis height 0.75 m)."""
    tb = bare_toolbox(FakeRobot())
    C = np.array([[1.3, 0.3, 1.0], [1.3, 0.1, 1.0], [1.3, 0.1, 0.75], [1.3, 0.3, 0.75]])
    tb.ws.target_grid = {"edges_seen": [], "face_centres": [{"face_id": 0, "center": np.array([1.3, 0.2, 0.9])},
                                                           {"face_id": 0, "center": np.array([1.3, -0.2, 0.6])}],
                         "fitted_faces": [{"face_id": 0, "face": {"t": 1.0, "corners": C, "edges_seen": {"top": True}}}],
                         "faces": {}}
    edges, centres, faces, bounds = tb._grid_evidence(0, T_from_xy_yaw(0.5, 0.0, 0.0, 0.75))
    assert bounds == {"bottom": pytest.approx(-0.75)} and centres["rows"] == pytest.approx([0.15, -0.15])
    # a fitted face (with or without labels) is passed as box evidence, in the planning frame
    assert len(faces) == 1 and (faces[0]["z_lo"], faces[0]["z_hi"]) == pytest.approx((0.0, 0.25))
    assert faces[0]["seen"] == {"top": True}
    assert tb._grid_evidence(1, np.eye(4))[3] is None               # no face of that side: no bound


# ------------------------------------------------------------------ wall-layout branch (g1v2-scanroom-layout): shared code
def test_rectified_detection_reports_its_bar_direction_in_source_pixels(monkeypatch):
    """_labels_on_rectified_face: rec is a detection in the SOURCE image, so is its bar direction (the detector's is in
    rectified pixels; refine_with_wrist maps rec["bar_dir"] through the source camera). Here the bars run along the
    rectified x axis = the face's lateral axis, which the head camera (looking along +x) images along its own x."""
    tb = bare_toolbox(FakeRobot())
    h = _head_frame()
    face = _face(0.0, 0.4, x=1.5)

    def fake_dino(rgb, roi, det):
        Wr = rgb.shape[1]
        s_ = Wr / 0.4
        q = np.array([[10, 10], [10 + 0.1 * s_, 10], [10 + 0.1 * s_, 10 + 0.05 * s_], [10, 10 + 0.05 * s_]], float)
        return [{"kind": "label", "corners": q, "center": q.mean(axis=0), "score": 0.6, "bar_dir": np.array([0.0, 1.0]),
                 "label_corners": q, "label_center": q.mean(axis=0), "symbol_corners": None, "refined_by_cv2": False}]

    monkeypatch.setattr(tbmod, "detect_label_dino", fake_dino)
    tb.det = None
    (rec, est), = tb._labels_on_rectified_face(h, face)
    assert abs(est["bar_dir"][2]) == pytest.approx(1.0, abs=1e-6)          # vertical bars on the face
    T = np.linalg.inv(h["T_odom_root"] @ h["T_root_cam"])
    uv, _ = geo.project(h["K"], T, np.array([est["center"], est["center"] + 0.01 * est["bar_dir"]]))
    d = (uv[1] - uv[0]) / np.linalg.norm(uv[1] - uv[0])
    assert np.asarray(rec["bar_dir"]) == pytest.approx(d, abs=1e-3)       # the projected direction, not (0, 1) blindly


def test_move_arm_through_reach_and_signed_roll_only_for_look_targets():
    """move_arm's IK + correction loop is _reach (search_cell uses it for the look pose): the same calls as before for a
    label target (no roll_signed keyword), roll_signed=True for a look target; a cell entry refuses pre / final."""
    tb = bare_toolbox(FakeRobot(), task=2)
    tb.side = "left"
    tb._odom = lambda: np.eye(4)
    tb._q = lambda: None
    tb.robot.set_arm_joints = lambda side, q, dur: {"status": "ok"}
    tb.robot.stand = lambda s: {"status": "ok"}
    calls = []

    class IK:
        def solve(self, q, p, d, roll_axis=None, **kw):
            calls.append(kw)
            return {"ok": True, "q_arm": np.zeros(7)}
    tb.ik = IK()
    T = np.eye(4)
    T[:3, 3] = [0.6, 0.0, 1.0]
    tb.kin = type("K", (), {"pose": lambda self, q, frame: T})()
    tb._roll_in_root = lambda t: None
    tb.ws.wrist_target = {"pos": [0.6, 0.0, 1.0], "forward": [1.0, 0.0, 0.0], "offset": 0.2, "barcode_id": "bc0"}
    assert tb.move_arm("final")["status"] == "ok" and calls == [{}]
    r = tb._reach(dict(tb.ws.wrist_target, purpose="look", roll_signed=True), 0.2, "left")
    assert not r["ik_fail"] and calls[-1] == {"roll_signed": True}
    tb.wall_layout = {"columns": 6, "rows": 5, "box_w": 0.2, "box_h": 0.25, "base": "floor"}
    plan_of(tb, 1)
    tb.ws.current_barcode_id = "bc0"
    tb.ws.barcodes["bc0"].kind = "cell"
    for pose in ("pre", "final"):
        assert "call search_cell" in tb.move_arm(pose)["reason"]


def test_stance_eval_without_pre_pose_and_with_a_signed_roll():
    """stance_eval.eval_candidate: pre_dists [] (a look pose, approached from home) gives gap 0 with no extra solve; a
    target with roll_signed solves its roll with the sign; other targets call the IK as before."""
    from g1agent.tools import stance_eval as SE
    seen = []

    class IK:
        frame, vi = "tip", np.arange(7)

        def solve(self, q, p, d, stall, roll, rng=None, **kw):
            seen.append(kw)
            return {"ok": True, "limit_margin_min": 0.1, "q_full": None}

    class Kin:
        def jacobian(self, q, frame):
            return np.eye(6, 7)
    t = {"pos": [0.5, 0, 1.0], "forward": [1.0, 0, 0], "offset": 0.3, "long_axis": [0, 1.0, 0], "roll_signed": True}
    job = {"Tc": np.eye(4), "q_lv": None, "t": t, "finals": [0.3], "pre_dists": [], "stall": None, "seed": [0, 0, 0]}
    r = SE.eval_candidate(Kin(), IK(), None, None, job)
    assert r["status"] == "ok" and r["gap"] == 0.0 and len(r["sols"]) == 1 and seen == [{"roll_signed": True}]
    seen.clear()
    t2 = {k: v for k, v in t.items() if k != "roll_signed"}
    r = SE.eval_candidate(Kin(), IK(), None, None, dict(job, t=t2, pre_dists=[0.4]))
    assert r["gap"] == pytest.approx(0.1) and seen == [{}, {}]


def test_plan_stance_width_upper_bound_includes_the_hand_label_width():
    """plan_stance's required final distances: a head or a hand-label width (an upper bound of the symbol) plans the
    clamp min_m and the pre-pose range; only a wrist-measured width plans the stand-off itself."""
    import inspect
    src = inspect.getsource(tbmod.Toolbox._stance_member)          # plan_stance's required poses (2026-11-20 split)
    assert 'bc.width_source != "wrist"' in src and 'bc.width_source == "head"' not in src
    assert "self._stance_member(bc, t)" in inspect.getsource(tbmod.Toolbox._plan_stance)


def test_harness_registry_layout_tools_leave_the_task_lists_unchanged():
    from g1agent.harness.registry import LAYOUT_ONLY, TASK2_TOOLS, TOOLS
    assert LAYOUT_ONLY == ["search_cell"] and "search_cell" in TOOLS
    from g1agent.harness.registry import STRIP_ONLY
    assert STRIP_ONLY == ["go_to_next_strip"]                         # strip episodes only (2026-11-20)
    assert TASK_TOOLS[3] == [n for n in TOOLS if n not in LAYOUT_ONLY + STRIP_ONLY] and "search_cell" not in TASK2_TOOLS
    assert "no_label" in __import__("g1agent.harness.world_state", fromlist=["x"]).BARCODE_STATUSES
