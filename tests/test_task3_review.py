"""CPU tests of the task-3 review fixes: stance clearance to neighbouring faces, no unplanned barcode once a scan plan
exists, stage events logged after the tool record that triggers them, the rule Task3 stop when plan_scans is
rejected, and agent_full recording the home arm pose for tasks with arm tools."""
import numpy as np
import pytest

from g1agent.common import config
from g1agent.orchestrators import rule
from g1agent.tools import toolbox as tbmod
from test_task3_logic import FakeLLM, FakeLog, FakeRobot, ScriptedTB, _harness, bare_toolbox, plan_of, read


def _face(y0, y1, x):
    return {"center": np.array([x, (y0 + y1) / 2, 1.0]), "normal": np.array([-1.0, 0, 0]), "right": np.array([0, -1.0, 0]),
            "up": np.array([0, 0, 1.0]), "corners": np.array([[x, y1, 1.2], [x, y0, 1.2], [x, y0, 0.8], [x, y1, 0.8]])}


def test_horizontal_distance_to_finite_face():
    f = _face(-0.2, 0.2, x=1.0)
    assert tbmod.horizontal_distance_to_face([0.5, 0.0, 0.0], f) == pytest.approx(0.5)      # in front: to the plane
    assert tbmod.horizontal_distance_to_face([1.0, 0.5, 0.0], f) == pytest.approx(0.3)      # beside: to the edge
    assert tbmod.horizontal_distance_to_face([0.7, 0.6, 0.0], f) == pytest.approx(0.5)      # diagonal to the corner


def test_known_faces_include_barcode_face_copies():
    tb = bare_toolbox(FakeRobot())
    plan_of(tb, 2)
    for b in tb.ws.barcodes.values():
        b.face = dict(b.face, corners=_face(-0.2, 0.2, 1.0)["corners"])
    tb.ws.boxes = {"box0": tbmod.BoxEstimate("box0", [0, 0, 1, 1], 0.9, 0.0)}
    tb.ws.boxes["box0"].face = _face(0.3, 0.7, 0.8)
    assert len(tb._known_faces()) == 3


def test_detect_barcode_head_refused_once_a_plan_exists():
    tb = bare_toolbox(FakeRobot())
    plan_of(tb, 2)
    tb.next_barcode()
    r = tb.detect_barcode_head()
    assert r["status"] == "fail" and tb.ws.current_barcode_id == "bc0"


class OrderLog(FakeLog):
    def __init__(self):
        super().__init__()
        self.events = []

    def tool(self, rec):
        super().tool(rec)
        self.events.append(("tool", rec["tool"]))

    def stage(self, name, info=None):
        super().stage(name, info)
        self.events.append(("stage", name))


def test_stage_events_follow_their_tool_record(monkeypatch):
    tb = bare_toolbox(FakeRobot({"bc0": [read("A")]}))
    h = _harness(tb)
    h.log = OrderLog()

    def fake_plan():
        plan_of(tb, 1)
        return {"status": "ok", "data": {"n": 1}, "reason": ""}

    monkeypatch.setattr(tb, "plan_scans", fake_plan)
    for name in ("plan_scans", "next_barcode", "scan", "next_barcode"):
        h.call(name)
    assert h.log.events == [("tool", "plan_scans"), ("stage", "task1_end"), ("tool", "next_barcode"), ("stage", "barcode_start"),
                            ("tool", "scan"), ("stage", "barcode_end"), ("tool", "next_barcode"), ("stage", "scan_done")]


def test_rule_task3_stops_when_plan_scans_is_rejected():
    class NoLabelTB(ScriptedTB):
        def plan_scans(self):
            return tbmod.fail("no barcode label localised on any fitted face")

        def explore(self, direction, distance):
            return tbmod.ok({"move_status": "ok"})

        def approach(self, target="root_target"):          # arrival passes: only plan_scans fails
            return tbmod.ok({"walk_done": True})

    tb = NoLabelTB(FakeRobot())
    h = _harness(tb)
    c = config.load_all()["task3"]
    c["explore_sequence"] = [["forward", 0.1]] * 6               # every explore move valid: only the rejection stops it
    rule.Task3(h, c, "all").run()
    tools = [r["tool"] for r in h.log.tools]
    # max_retries_per_tool + 1 executed failures, then one rejected call ends the episode (no further phase-1 redo)
    assert tools.count("plan_scans") == c["budget"]["max_retries_per_tool"] + 2
    assert h.log.tools[-2]["result"].get("rejected") and tools[-1] == "finish"
    assert h.finished["success"] is False


def test_agent_full_records_home_pose_when_the_task_has_arm_tools(monkeypatch):
    from g1agent.orchestrators.agent_full import run_agent_full
    tb = ScriptedTB(FakeRobot(), plan_n=1)
    called = []
    monkeypatch.setattr(tb, "remember_home", lambda: called.append(True))
    monkeypatch.setattr(tb, "look_head", lambda: (tbmod.ok({}), np.zeros((4, 4, 3), np.uint8)))
    h = _harness(tb)
    run_agent_full(h, config.load_all()["task3"], "all", FakeLLM([("finish", {"task": 3, "success": False, "summary": "x"})]), h.log)
    assert called == [True]
