"""Task-3 evaluation pieces on CPU: scan order rule, adaptive stand-off, the evaluator's task-3 branch on a synthetic
episode, logger events, the LLM client's reasoning / timing fields and the timeline builder."""
import json
import math
import os
import sys

import numpy as np
import pytest

from g1agent.eval import evaluator
from g1agent.eval.scan_order import adaptive_standoff, column_order, hfov_from_intrinsics, order_labels_in_frame
from g1agent.harness.logger import EpisodeLogger, tool_overlay

HFOV = hfov_from_intrinsics(1087.0, 640.0)


# ------------------------------------------------------------------ scan order / stand-off
def test_column_order_left_to_right_then_top_to_bottom():
    items = [  # left = +y; A over B share a column (overlapping extents), C is a column to the right, D right of C
        {"key": "B", "y_lo": 0.20, "y_hi": 0.30, "z": 0.8},
        {"key": "A", "y_lo": 0.22, "y_hi": 0.32, "z": 1.0},
        {"key": "C", "y_lo": -0.05, "y_hi": 0.05, "z": 1.2},
        {"key": "D", "y_lo": -0.40, "y_hi": -0.30, "z": 0.7}]
    assert column_order(items) == ["A", "B", "C", "D"]


def test_column_order_chained_overlap_is_one_column_and_touching_is_not():
    # X overlaps Y, Y overlaps Z, X and Z do not: still one column (connected), ordered by height
    items = [{"key": "X", "y_lo": 0.0, "y_hi": 0.10, "z": 0.5}, {"key": "Y", "y_lo": 0.08, "y_hi": 0.18, "z": 0.9},
             {"key": "Z", "y_lo": 0.16, "y_hi": 0.26, "z": 0.7}]
    assert column_order(items) == ["Y", "Z", "X"]
    touching = [{"key": "L", "y_lo": 0.10, "y_hi": 0.20, "z": 0.5}, {"key": "R", "y_lo": 0.0, "y_hi": 0.10, "z": 0.9}]
    assert column_order(touching) == ["L", "R"]


def test_adaptive_standoff():
    assert math.degrees(HFOV) == pytest.approx(32.8, abs=0.1)
    S = evaluator.config.load("task3")["standoff"]
    assert adaptive_standoff(0.08, HFOV, S["fill_frac"], 0.12, 0.30) == pytest.approx(0.20, abs=0.002)   # nominal symbol
    assert adaptive_standoff(0.02, HFOV, S["fill_frac"], 0.12, 0.30) == 0.12
    assert adaptive_standoff(0.50, HFOV, S["fill_frac"], 0.12, 0.30) == 0.30


def _bc(content, y, z, w=0.10, distractor=False):
    return {"content": content, "center": [1.3485, y, z], "normal": [-1.0, 0.0, 0.0], "up": [0.0, 0.0, 1.0],
            "size": [w, 0.05], "pattern_size": [w - 0.02, 0.03], "box_id": "B", "distractor": distractor}   # 1 cm backing


def test_order_labels_in_root_frame():
    T = np.eye(4)
    labels = [dict(_bc(c, y, z), key=c) for c, y, z in (("low_left", 0.25, 0.8), ("right", -0.25, 0.9), ("top_left", 0.27, 1.0))]
    assert order_labels_in_frame(labels, T) == ["top_left", "low_left", "right"]
    # robot turned around (yaw 180 deg): its left is world -y
    Tb = np.diag([-1.0, -1.0, 1.0, 1.0])
    assert order_labels_in_frame(labels, Tb) == ["right", "top_left", "low_left"]


# ------------------------------------------------------------------ evaluator task 3
def _actual(x, y=0.0, z=0.0):
    Twr = np.eye(4); Twr[:3, 3] = [x, y, 0.8]
    Twh = np.eye(4)
    Twh[:3, :3] = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)   # optical frame looking along +x
    Twh[:3, 3] = [x, y, 1.2]
    return {"T_world_root": Twr, "T_world_odom_now": np.eye(4), "T_world_headcam": Twh,
            "head_K": [[300, 0, 320], [0, 300, 240], [0, 0, 1]], "head_size": [480, 640],
            "T_world_wrist_point": np.eye(4)}


def _tip(target, err=(0.0, 0.0, 0.0)):
    T = np.eye(4); T[:3, 3] = np.asarray(target, float) + np.asarray(err)   # +x = scan direction = -normal
    return T


@pytest.fixture
def episode():
    face = {"center": [1.35, 0.0, 0.9], "normal": [-1.0, 0.0, 0.0],
            "corners": [[1.35, 0.2, 0.75], [1.35, -0.2, 0.75], [1.35, -0.2, 1.05], [1.35, 0.2, 1.05]]}
    gt = {"barcodes": [_bc("R", -0.25, 0.9), _bc("L", 0.25, 0.9), _bc("DX", 0.0, 0.9, distractor=True)],
          "target_face": face, "task1_gt_obs_pose": {"pos": [0.75, 0.0, 0.0], "yaw": 0.0},
          "robot_init": {"pos": [-0.7, 0.0, 0.0], "yaw": 0.0}}
    # root target at task1_end (d_obs 0.60, the robot stands 0.59 m from the face) vs the final one, recomputed in the
    # scan phase: the task-1 metrics must use the former
    ws = {"root_target": {"pos": [1.05, 0.0], "yaw": 0.0, "d_obs": 0.30},
          "task1_end_root_target": {"pos": [0.75, 0.0], "yaw": 0.0, "d_obs": 0.60},
          "scan_plan": [{"id": "bc0", "status": "scanned"}, {"id": "bc1", "status": "scanned"}],
          "scanned": [{"content": "L"}, {"content": "R"}]}
    stages = {"task1_end": {"t_sim": 30.0, "actual": _actual(0.76)}}
    S = evaluator.config.load("task3")["standoff"]                       # configs/task3.yaml (real config)
    d = adaptive_standoff(0.08, evaluator.scanner_hfov_gt(), S["fill_frac"], S["min_m"], S["max_m"])
    tgt = {c: np.array([1.3485 - d, y, 0.9]) for c, y in (("L", 0.25), ("R", -0.25))}
    return gt, ws, stages, tgt


def test_task3_success(episode):
    gt, ws, stages, tgt = episode
    hitL = {"hit": {"content": "L", "distance_m": 0.2}}
    log = [{"t_sim": 40.0, "content": None, "decoded": [], "gt": hitL, "T_world_tip": _tip(tgt["L"], (0, 0.01, 0))},
           {"t_sim": 41.0, "content": "L", "decoded": [{"content": "L"}], "gt": hitL, "T_world_tip": _tip(tgt["L"], (0.0, 0.003, 0.004))},
           {"t_sim": 50.0, "content": "R", "decoded": [{"content": "R"}], "gt": {"hit": None}, "T_world_tip": _tip(tgt["R"])},
           {"t_sim": 51.0, "content": "R", "decoded": [{"content": "R"}], "gt": {"hit": None}, "T_world_tip": _tip(tgt["R"])}]
    m = evaluator.evaluate(3, gt, _actual(1.0), ws, stages=stages, scan_log=log)
    assert m["t1_success"] and m["order_frame"] == "task1_end"
    assert m["task1"]["d_obs_harness"] == 0.60 and m["t1_d_err_m"] == pytest.approx(0.01, abs=1e-6)
    assert m["gt_order"] == ["L", "R"] and m["n_gt_barcodes"] == 2        # distractor excluded
    assert m["scan_recall"] == 1.0 and m["n_duplicate_reads"] == 1 and m["n_no_read"] == 1 and m["n_wrong_contents"] == 0
    assert m["order_ok"] and m["ledger_matches_log"] and m["success"], m.get("failure")
    assert m["tip_err_max_m"] == pytest.approx(0.005, abs=1e-6)
    assert m["reads"][0]["ref_source"] == "scan_ray" and m["reads"][1]["ref_source"] == "decoded"
    assert m["gt_standoff_m"]["L"] == pytest.approx(0.20, abs=0.002)


def test_task3_wrong_order_and_missing_stage(episode):
    gt, ws, stages, tgt = episode
    log = [{"t_sim": 1.0, "content": "R", "decoded": ["R"], "gt": {"hit": None}, "T_world_tip": _tip(tgt["R"])},
           {"t_sim": 2.0, "content": "L", "decoded": ["L"], "gt": {"hit": None}, "T_world_tip": _tip(tgt["L"])}]
    m = evaluator.evaluate(3, gt, _actual(1.0), ws, stages=stages, scan_log=log)
    assert m["scan_recall"] == 1.0 and not m["order_ok"] and not m["order_ok_scanned"] and not m["success"]
    m = evaluator.evaluate(3, gt, _actual(1.0), ws, stages={}, scan_log=log[1:])
    assert m["order_frame"] == "robot_init" and not m["t1_success"] and m["order_ok_scanned"] and not m["success"]
    assert "1/2" in m["failure"] and "task1_end" not in m["failure"]   # the task-1 phase is reported (t1_success), not gating


def test_task3_foreign_content_fails(episode):
    gt, ws, stages, tgt = episode
    log = [{"t_sim": 1.0, "content": c, "decoded": [c], "gt": {"hit": None}, "T_world_tip": _tip(tgt.get(c, tgt["L"]))}
           for c in ("L", "DX", "R")]
    m = evaluator.evaluate(3, gt, _actual(1.0), ws, stages=stages, scan_log=log)
    assert m["order_ok"] and m["wrong_contents"] == ["DX"] and not m["success"]


def test_evaluate_rejects_unknown_task():
    with pytest.raises(ValueError):
        evaluator.evaluate(4, {}, _actual(0.0), {})


# ------------------------------------------------------------------ logger events
def test_logger_events_and_stage(tmp_path):
    ev = []
    log = EpisodeLogger(str(tmp_path), on_event=ev.append)
    log.tool({"i": 0, "tool": "detect_boxes", "args": {}, "result": {"status": "ok", "reason": "", "data": {
        "n": 2, "boxes": [{"id": "box0", "bbox": [1, 2, 3, 4], "score": 0.5}, {"id": "box1", "bbox": [5, 6, 7, 8], "score": 0.4}]}}, "seconds": 0.1})
    log.llm({"kind": "agent_full", "prompt_version": "v", "messages": [],
             "reply": {"content": "Detect the boxes first.", "reasoning_content": None,
                       "tool_calls": [{"id": "c0", "name": "detect_boxes", "arguments": {}}], "finish_reason": "tool_calls", "usage": None},
             "t_start": 100.0, "t_end": 101.5})
    log.llm({"kind": "explore_invalid", "args": {}, "error": "x"})          # no reply -> no event
    log.stage("task1_end", {"ok": True})
    assert [e["kind"] for e in ev] == ["tool", "llm", "stage"]
    assert ev[0]["overlay"] == {"head_boxes": [{"id": "box0", "bbox": [1, 2, 3, 4]}, {"id": "box1", "bbox": [5, 6, 7, 8]}]}
    assert ev[1]["thought"] == "Detect the boxes first." and ev[1]["latency_s"] == 1.5
    assert ev[1]["tool_calls"] == [{"name": "detect_boxes", "arguments": {}}]
    assert ev[2] == {"kind": "stage", "t_unix": ev[2]["t_unix"], "name": "task1_end", "info": {"ok": True}}
    recs = [json.loads(line) for line in open(tmp_path / "tool_calls.jsonl")]
    assert "t_unix" in recs[0] and "t_wall" in recs[0]
    assert json.loads(open(tmp_path / "stages.jsonl").read())["stage"] == "task1_end"


def test_tool_overlay_fields():
    scan = {"status": "ok", "data": {"content": "A", "decoded": ["A"], "polygons": [{"content": "A", "polygon": [[0, 0], [1, 0], [1, 1]]}],
                                     "new": True, "duplicate_of": None, "scanned": ["A"]}}
    assert tool_overlay("scan", scan) == {"wrist_polygons": [{"content": "A", "polygon": [[0, 0], [1, 0], [1, 1]]}], "scanned": ["A"]}
    plan = {"status": "ok", "data": {"barcodes": [{"id": "bc0", "box_id": "box0", "corners_px": [[0, 0]] * 4, "status": "pending",
                                                   "order_index": 0}]}}
    ov = tool_overlay("plan_scans", plan)
    assert ov["head_labels"] == [{"id": "bc0", "corners_px": [[0, 0]] * 4}]
    assert ov["scan_plan"] == [{"id": "bc0", "status": "pending", "content": None}]
    assert tool_overlay("select_target_box", {"status": "ok", "data": {"target_box_id": "box1", "bbox": [0, 0, 1, 1]}}) == {"target_box": "box1"}
    assert tool_overlay("next_barcode", {"status": "ok", "data": {"barcode_id": "bc1", "index": 1}}) == {"current_barcode": "bc1"}
    assert tool_overlay("refine_with_wrist", {"status": "fail", "data": {}}) == {}


# ------------------------------------------------------------------ LLM client fields + timeline
def test_llm_client_reasoning_and_times():
    from test_llm_client import H, REPLIES, client, tc
    import threading
    from http.server import HTTPServer
    s = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    try:
        REPLIES[:] = [{"message": {"role": "assistant", "content": "partial"}, "finish": "length"},
                      {"message": {"role": "assistant", "content": "Look first.", "reasoning_content": "I should look.",
                                   "tool_calls": [tc("look_head", "{}")]}, "finish": "tool_calls"}]
        r = client(f"http://127.0.0.1:{s.server_port}/v1").chat([{"role": "user", "content": "hi"}], tools=[])
    finally:
        s.shutdown()
    assert r["content"] == "Look first." and r["reasoning_content"] == "I should look."
    assert r["t_start"] <= r["attempts"][0]["t_start"] <= r["attempts"][1]["t_end"] == r["t_end"]
    assert [a["finish_reason"] for a in r["attempts"]] == ["length", "tool_calls"]


def test_timeline(tmp_path):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    from make_timeline import build_timeline
    from g1agent.orchestrators.common import log_llm
    log = EpisodeLogger(str(tmp_path))
    msgs = [{"role": "system", "content": "SYS"},
            {"role": "user", "content": [{"type": "text", "text": "Start."}, {"type": "image_url", "_path": "images/001_llm_start.jpg"}]}]
    t = log.t0
    log_llm(log, "agent_full", "v1", msgs, {"content": "Find boxes.", "reasoning_content": None, "finish_reason": "tool_calls",
                                             "tool_calls": [{"id": "c0", "name": "detect_boxes", "arguments": {}, "parse_error": None}],
                                             "usage": {"prompt_tokens": 5, "completion_tokens": 3}, "t_start": t, "t_end": t + 0.001,
                                             "attempts": [{"finish_reason": "tool_calls"}]})
    import time
    time.sleep(0.05)
    log.tool({"i": 0, "tool": "detect_boxes", "args": {}, "result": {"status": "ok", "reason": "", "data": {"n": 2}}, "seconds": 0.01})
    log.stage("task1_end", {"arrived": True})
    md, jl = build_timeline(str(tmp_path))
    ev = [json.loads(line) for line in open(jl)]
    assert [e["type"] for e in ev] == ["llm", "tool", "stage"]
    assert ev[0]["t_start_s"] == pytest.approx(0.0, abs=1e-3) and ev[0]["system_prompt"] == "SYS"
    assert ev[0]["given"] == [{"role": "user", "text": "Start.", "images": ["images/001_llm_start.jpg"]}]
    text = open(md).read()
    assert "Find boxes." in text and "STAGE `task1_end`" in text and "detect_boxes" in text


# ------------------------------------------------------------------ evaluator task 3, four-face stack (I2 / I3)
def _ep4(labelled=("-x", "-y", "+x", "+y")):
    from test_compositor import _gt4
    gt = _gt4(labelled)
    ws = {"root_target": {"pos": [0.6, 0.0], "yaw": 0.0, "d_obs": 0.6},
          "task1_end_root_target": {"pos": [0.6, 0.0], "yaw": 0.0, "d_obs": 0.6}, "scan_plan": [], "scanned": []}
    stages = {"task1_end": {"t_sim": 30.0, "actual": _actual(0.6)}}
    return gt, ws, stages


def _log(contents):
    return [{"t_sim": float(i), "content": c, "decoded": [c] if c else [], "gt": {"hit": None}, "T_world_tip": np.eye(4)}
            for i, c in enumerate(contents)]


TOUR4 = ["F0", "F1", "F2", "R0", "R1", "K0", "K1", "L0", "L1"]


def test_labelled_faces_order_each_face_in_its_own_frame():
    from g1agent.eval.scan_order import FACE_TOUR, face_frame, labelled_faces
    from test_compositor import GT4_ORDER, _gt4
    gt = _gt4()
    labels = [dict(b, key=b["content"]) for b in gt["barcodes"] if not b["distractor"]]
    groups = labelled_faces(labels, list(reversed(gt["faces"])))         # input order does not matter: tour order out
    assert [f for f, _ in groups] == list(FACE_TOUR) and dict(groups) == GT4_ORDER
    T = face_frame([0.0, -1.0, 0.0], [1.5, -1.0, 0.0])                  # right face: forward +y, left -x
    assert np.allclose(T[:3, 0], [0, 1, 0]) and np.allclose(T[:3, 1], [-1, 0, 0]) and np.allclose(T[:3, 2], [0, 0, 1])
    # a label whose normal is tilted 30 deg towards -y still belongs to the front face (argmax of the normal dot)
    tilted = dict(labels[0], key="T", normal=[-math.cos(math.radians(30)), -math.sin(math.radians(30)), 0.0])
    assert "T" in dict(labelled_faces(labels + [tilted], gt["faces"]))["-x"]


def test_task3_four_faces_success():
    gt, ws, stages = _ep4()
    m = evaluator.evaluate(3, gt, _actual(1.0), ws, stages=stages, scan_log=_log(["F0", None, "F1", "F2", "F2", "R0", "R1",
                                                                                     "K0", "K1", "L0", "L1"]))
    assert m["order_frame"] == "faces" and m["gt_order"] == TOUR4 and m["n_gt_barcodes"] == 9      # distractor excluded
    assert [pf["face"] for pf in m["per_face"]] == ["-x", "-y", "+x", "+y"] and [pf["face_idx"] for pf in m["per_face"]] == [0, 1, 2, 3]
    assert all(pf["order_ok"] and pf["recall"] == 1.0 for pf in m["per_face"])
    assert m["face_order"] == ["-x", "-y", "+x", "+y"] and m["front_first"] and m["faces_contiguous"] and m["face_order_ok"]
    assert m["order_ok"] and m["scan_recall"] == 1.0 and m["n_duplicate_reads"] == 1 and m["success"], m.get("failure")


def test_task3_four_faces_other_face_order_reported_not_gating():
    gt, ws, stages = _ep4()
    reads = ["F0", "F1", "F2", "L0", "L1", "K0", "K1", "R0", "R1"]       # clockwise tour: each face whole and in order
    m = evaluator.evaluate(3, gt, _actual(1.0), ws, stages=stages, scan_log=_log(reads))
    assert m["face_order"] == ["-x", "+y", "+x", "-y"] and not m["face_order_ok"] and not m["order_ok"]
    assert m["faces_contiguous"] and m["front_first"] and m["success"], m.get("failure")


@pytest.mark.parametrize("reads,why", [
    (["F0", "F1", "R0", "F2", "R1", "K0", "K1", "L0", "L1"], "not scanned contiguously"),     # came back to the front
    (["R0", "R1", "F0", "F1", "F2", "K0", "K1", "L0", "L1"], "is not the first labelled tour face"),
    (["F0", "F1", "F2", "R1", "R0", "K0", "K1", "L0", "L1"], "face -y order"),                 # right face reversed
    (["F0", "F1", "F2", "R0", "R1", "K0", "K1", "L0"], "scanned 8/9"),
])
def test_task3_four_faces_failures(reads, why):
    gt, ws, stages = _ep4()
    m = evaluator.evaluate(3, gt, _actual(1.0), ws, stages=stages, scan_log=_log(reads))
    assert not m["success"] and why in m["failure"], m["failure"]


def test_task3_four_faces_foreign_content_fails():
    gt, ws, stages = _ep4()
    m = evaluator.evaluate(3, gt, _actual(1.0), ws, stages=stages, scan_log=_log(TOUR4[:3] + ["DX"] + TOUR4[3:]))
    assert m["order_ok"] and m["wrong_contents"] == ["DX"] and not m["success"]


def test_task3_single_labelled_face_metrics_unchanged(episode):
    """A GT with "faces" but labels on one side only (stack3 / stack9 / t1 / t2) scores exactly as without "faces"."""
    gt, ws, stages, tgt = episode
    log = [{"t_sim": 1.0, "content": "L", "decoded": ["L"], "gt": {"hit": None}, "T_world_tip": _tip(tgt["L"])},
           {"t_sim": 2.0, "content": "R", "decoded": ["R"], "gt": {"hit": None}, "T_world_tip": _tip(tgt["R"])}]
    from test_compositor import _gt4
    g4 = dict(gt, faces=_gt4(labelled=())["faces"])                      # every side present, only -x labelled
    base = evaluator.evaluate(3, gt, _actual(1.0), ws, stages=stages, scan_log=log)
    assert evaluator.evaluate(3, g4, _actual(1.0), ws, stages=stages, scan_log=log) == base
    assert base["order_frame"] == "task1_end" and "per_face" not in base and base["success"]
    gf, wsf, stf = _ep4(labelled=("-x",))
    nofaces = {k: v for k, v in gf.items() if k != "faces"}
    lf = _log(["F0", "F1", "F2"])
    assert evaluator.evaluate(3, gf, _actual(1.0), wsf, stages=stf, scan_log=lf) == evaluator.evaluate(
        3, nofaces, _actual(1.0), wsf, stages=stf, scan_log=lf)
