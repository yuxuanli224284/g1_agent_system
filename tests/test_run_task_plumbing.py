"""run_task.run() plumbing on CPU with fake robot / eval port / detector / toolbox / rule orchestrator: logger events go
to the eval port (stage -> 'stage', everything else -> 'annotate'), the harness root target is snapshotted when the
task1_end stage fires (the evaluator pairs it with the GT state stored at that stage), and result.json + the timeline
are written."""
import json
import os
import sys

import pytest

SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")


class FakeWS:
    def __init__(self):
        self.root_target = None
        self.wrist_target = self.wrist_target_refined = None
        self.explore_steps = 0

    def summary(self):
        return {"root_target": self.root_target, "scan_plan": [], "scanned": [], "current_barcode_id": None}


class FakeTB:
    def __init__(self, robot, box_detector=None):
        self.ws = FakeWS()

    def set_task(self, task):
        self.task = task


class FakeRPC:
    def __init__(self, calls):
        self.calls = calls

    def call(self, method, **kw):
        self.calls.append((method, kw))
        if method == "stage":
            return {"status": "ok", "actual": {}}
        if method == "get_stage_states":
            return {"task1_end": {"t_sim": 1.0, "actual": {}}}
        if method == "get_scan_log":
            return []
        if method == "get_stage_log":
            return [{"name": "task1_end", "info": {"arrival": "ok"}, "t_sim": 1.0, "actual": {}}]
        return {"status": "ok"}


class FakeEval:
    calls = []

    def __init__(self):
        self.rpc = FakeRPC(FakeEval.calls)

    def reset(self, sc): return {"status": "ok"}
    def set_gt_markers(self, on): return {"status": "ok"}
    def record_start(self, path, fps=None): return {"status": "ok"}
    def record_stop(self): return {"status": "ok"}
    def snapshot(self, path=None): return {"status": "ok"}
    def get_gt(self): return {}
    def get_actual_state(self): return {}


class FakeTask3:
    def __init__(self, h, cfg, selector):
        self.h, self.selector = h, selector

    def run(self):
        tb = self.h.tb
        tb.ws.root_target = {"pos": [0.75, 0.0], "yaw": 0.0, "d_obs": 0.6}
        self.h.log.stage("task1_end", {"arrival": "ok"})
        tb.ws.root_target = {"pos": [1.1, 0.3], "yaw": 0.1, "d_obs": 0.3}     # recomputed later in the scan phase
        self.h.log.tool({"i": 0, "tool": "next_barcode", "args": {}, "result": {"status": "ok", "reason": "",
                         "data": {"done": True}}, "seconds": 0.0})


def test_run_task3_plumbing(tmp_path, monkeypatch):
    sys.path.insert(0, SCRIPTS)
    import run_task
    import g1agent.eval.eval_client as EC
    import g1agent.eval.evaluator as EV
    import g1agent.orchestrators.rule as RULE
    import g1agent.perception.box_detector as BD
    import g1agent.robot.sim_client as SC
    import g1agent.tools.toolbox as TB
    seen = {}

    def fake_eval(task, gt, actual, ws, stages=None, scan_log=None):
        seen.update(task=task, ws=ws, stages=stages, scan_log=scan_log)
        return {"success": False, "failure": "fake"}

    FakeEval.calls = []
    monkeypatch.setattr(SC, "SimRobot", lambda: object())
    monkeypatch.setattr(EC, "EvalClient", FakeEval)
    monkeypatch.setattr(BD, "BoxDetector", lambda device=None: None)
    monkeypatch.setattr(TB, "Toolbox", FakeTB)
    monkeypatch.setattr(RULE, "Task3", FakeTask3)
    monkeypatch.setattr(EV, "evaluate", fake_eval)
    run = run_task.make_episode_runner()
    sc = {"id": "t1_fake", "task": 1, "target": "left", "level": 0, "factor": None}
    ep = str(tmp_path / "ep")
    res = run(sc, "rule", ep, video=False, task=3)
    assert res["task"] == 3 and res["scenario_task"] == 1 and res["orchestrator_error"] is None
    assert seen["task"] == 3
    assert seen["ws"]["task1_end_root_target"]["d_obs"] == 0.6 and seen["ws"]["root_target"]["d_obs"] == 0.3
    methods = [m for m, _ in FakeEval.calls]
    assert methods[:2] == ["stage", "annotate"]
    assert FakeEval.calls[0][1] == {"name": "task1_end", "info": {"arrival": "ok"}}
    assert FakeEval.calls[1][1]["event"]["kind"] == "tool"
    assert {"get_stage_states", "get_scan_log", "get_stage_log"} <= set(methods)
    for f in ("result.json", "timeline.md", "timeline.jsonl", "stages.jsonl", "tool_calls.jsonl"):
        assert os.path.exists(os.path.join(ep, f)), f
    res_json = json.load(open(os.path.join(ep, "result.json")))
    assert res_json["stages"]["task1_end"]["t_sim"] == 1.0 and [e["name"] for e in res_json["stage_log"]] == ["task1_end"]


def test_eval_rpc_raises_on_error_reply():
    sys.path.insert(0, SCRIPTS)
    import run_task

    class E:
        class rpc:
            @staticmethod
            def call(method, **kw):
                return {"status": "error", "reason": "unknown method"}
    with pytest.raises(RuntimeError):
        run_task._eval_rpc(E, "annotate", event={})
