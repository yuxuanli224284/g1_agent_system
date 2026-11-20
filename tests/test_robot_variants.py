"""Robot variants on the agent side (CPU): the model follows the variant the robot reports (config.apply_robot_variant,
fail-loud), the g1_v2 agent URDF is the description URDF + the simulator's frames (provenance), the scanner / hand-camera
split (IK roll on the scanner imager, images from the hand camera), and the variant texts."""
import hashlib
import math
import os
import re
import sys

import numpy as np
import pytest

from g1agent.common import config
from g1agent.kinematics.fk import Kinematics
from g1agent.kinematics.ik import ArmIK
from g1agent.tools import toolbox as tbmod
from test_task3_logic import FakeRobot, bare_toolbox, plan_of

SIM = config.load("system")["sim_server"]["sim_root"]


def _v(variant, head="d435i"):
    return config.apply_head_camera(config.apply_robot_variant(config.load_all(), variant), head)


def test_apply_robot_variant_is_fail_loud_and_merges_the_profile():
    for bad in (None, "g1_v3"):
        with pytest.raises(ValueError):
            config.apply_robot_variant(config.load_all(), bad)
    v1, v2 = _v("g1_v1"), _v("g1_v2")
    assert v1["robot"]["hand_camera"]["source"] == "scanner" and v1["robot"]["hand_camera"]["is_scanner"]
    assert v2["robot"]["hand_camera"] == {"source": "hand_camera", "name": "left_d405", "frame": "hand_cam_left_d405_optical",
                                          "is_scanner": False}
    assert v1["robot"]["urdf"].endswith("assets/g1_dex1_scanner.urdf") and v2["robot"]["urdf"].endswith("assets/g1_v2_scanner.urdf")
    assert v1["perception"]["barcode"]["wrist_upscale"] == 1 and v2["perception"]["barcode"]["wrist_upscale"] == 2
    assert config.load("perception")["barcode"]["wrist_upscale"] == 1                  # the file's value is untouched
    assert v2["robot"]["view"]["calibration"] == "assets/g1_v2/gear_lean_calibration.json"
    assert v2["robot"]["body"]["calibration"] == v1["robot"]["body"]["calibration"]    # stated approximation (robot.yaml)
    for k in ("scanner_cam_frame", "wrist_point_frame"):
        assert v1["robot"][k] == v2["robot"][k]


def test_system_variant_env_override(monkeypatch):
    monkeypatch.setenv("G1_ROBOT_VARIANT", "g1_v1")
    assert config.load("system")["sim_server"]["robot_variant"] == "g1_v1"
    monkeypatch.setenv("G1_ROBOT_VARIANT", "g1_v2")
    assert config.load("system")["sim_server"]["robot_variant"] == "g1_v2"


def test_kinematics_and_ik_need_the_resolved_variant():
    with pytest.raises(ValueError):
        Kinematics(None)
    kin = Kinematics(_v("g1_v1")["robot"]["urdf"])
    with pytest.raises(ValueError):
        ArmIK(kin, config.load("robot"))                           # raw robot.yaml: no variant frames


@pytest.mark.parametrize("variant", ["g1_v1", "g1_v2"])
def test_roll_reference_is_the_scanner_imager(variant):
    """The IK roll reference is the scanner imager's image x axis in the tip frame ([0, -1, 0] on both URDFs); on g1_v2
    the D405's image x happens to be the same axis but is not what the IK reads."""
    rc = _v(variant)["robot"]
    kin = Kinematics(rc["urdf"])
    ik = ArmIK(kin, rc)
    assert np.allclose(ik.img_x_in_tip, [0.0, -1.0, 0.0], atol=1e-6)
    import pinocchio as pin
    q = pin.neutral(kin.model)
    if variant == "g1_v2":
        T_t, T_d = kin.pose(q, "scanner_tip"), kin.pose(q, rc["hand_camera"]["frame"])
        z_d = (np.linalg.inv(T_t) @ T_d)[:3, 2]                    # D405 axis: +x of the wrist tilted ~10.3 deg down
        assert math.degrees(math.atan2(-z_d[2], z_d[0])) == pytest.approx(10.27, abs=0.05)


@pytest.mark.parametrize("variant", ["g1_v1", "g1_v2"])
def test_reach_chain_follows_the_model(variant):
    from g1agent.tools.body import BodyModel
    rc = _v(variant)["robot"]
    kin = Kinematics(rc["urdf"])
    chain = BodyModel._link_chain(kin, rc["wrist_point_frame"], "left_shoulder_pitch_link")
    base = ["left_shoulder_pitch_link", "left_shoulder_roll_link", "left_shoulder_yaw_link", "left_elbow_link",
            "left_wrist_roll_link", "left_wrist_pitch_link", "left_wrist_yaw_link"]
    assert chain == base + (["left_hand_base_link", "scanner_tip"] if variant == "g1_v1" else ["scanner_tip"])


# ------------------------------------------------------------------ g1_v2 agent URDF provenance
def test_g1_v2_urdf_frames_are_the_simulator_configs():
    sys.path.insert(0, SIM)
    from g1_barcode_sim.camera import make_T
    from g1_barcode_sim.config import load_all
    sim = load_all("scenario", {"robot": {"variant": "g1_v2"}, "camera": {"model": "d435i"}})
    rc = _v("g1_v2")["robot"]
    kin = Kinematics(rc["urdf"])
    q = kin.q_from_state([], [])
    rel = lambda parent, child: np.linalg.inv(kin.pose(q, parent)) @ kin.pose(q, child)
    h = sim["robot"]["hand_cameras"]["left_d405"]
    assert np.allclose(rel(h["parent_link"], "hand_cam_left_d405_optical"), make_T(h["pos"], h["rot_wxyz"]), atol=1e-6)
    m = sim["robot"]["scanner_mount"]
    T_tip = rel(m["parent_link"], "scanner_tip")
    assert np.allclose(T_tip, make_T(m["pos"], m["quat_wxyz"]), atol=1e-6)            # tip = window centre on g1_v2
    T_cam = np.linalg.inv(T_tip) @ rel(m["parent_link"], "wrist_cam_optical")
    assert np.allclose(T_cam[:3, 3], sim["scanner"]["camera"]["pos_in_tip"], atol=1e-6)
    cc = sim["camera"]
    assert np.allclose(rel("torso_link", "head_cam_optical"), make_T(cc["offset_pos"], cc["offset_rot_wxyz"]), atol=1e-6)
    assert np.allclose(rel("torso_link", "head_ir_left_optical"), make_T(cc["stereo"]["left_pos"], cc["stereo"]["rot_wxyz"]), atol=1e-6)
    assert cc["mount"]["rpy"][1] == pytest.approx(0.4564035994)
    # the URDF's own D405 optical frame is a placeholder ~10 cm off: the agent frame is the measured one
    T_ph = rel("left_wrist_yaw_link", "left_d405_depth_optical_frame")
    assert np.linalg.norm(T_ph[:3, 3] - np.asarray(h["pos"])) > 0.05


def test_g1_v2_urdf_header_and_reexport(tmp_path):
    """The committed asset records its source (path + sha256) and is exactly what the offline exporter writes now."""
    src = open(_v("g1_v2")["robot"]["urdf"]).read()
    path = re.search(r"source: (\S+)", src).group(1)
    sha = re.search(r"source sha256: ([0-9a-f]{64})", src).group(1)
    if not os.path.exists(path):
        pytest.skip(f"description URDF {path} not on this machine")
    assert hashlib.sha256(open(path, "rb").read()).hexdigest() == sha, "description URDF changed: re-export the agent URDF"
    sys.path.insert(0, os.path.join(config.ROOT, "scripts"))
    import export_urdf
    out = tmp_path / "v2.urdf"
    export_urdf.export_offline("g1_v2", str(out))
    assert out.read_text() == src


# ------------------------------------------------------------------ Toolbox: model from the robot info
class InfoRobot(FakeRobot):
    def __init__(self, info):
        super().__init__()
        self.info = info
        self.hand_calls = []

    def get_robot_info(self):
        return self.info

    def get_joint_state(self):
        return {"names": [], "q": []}

    def get_wrist_obs(self, side):
        return {"status": "ok", "rgb": np.zeros((480, 640, 3), np.uint8), "K": np.eye(3), "timestamp": 1.0}

    def get_hand_camera_obs(self, name, depth=False):
        self.hand_calls.append(name)
        return {"status": "ok", "rgb": np.zeros((480, 848, 3), np.uint8), "K": np.eye(3), "T_root_cam": np.eye(4),
                "timestamp": 2.0}


def _info(variant, **kw):
    K = [[1087.0, 0, 320], [0, 1087.0, 240], [0, 0, 1]]
    d = {"robot_variant": variant, "head_camera_model": "d435i", "wrist_side": "left", "wrist_K": K, "wrist_size": [480, 640],
         "hand_cameras": {"left_d405": {"side": "left"}} if variant == "g1_v2" else {}, "body_command": None}
    d.update(kw)
    return d


def test_toolbox_builds_the_reported_variant_and_fails_loud():
    tb = tbmod.Toolbox(InfoRobot(_info("g1_v2")), box_detector=object())
    assert tb.kin.urdf.endswith("g1_v2_scanner.urdf") and tb.cfg["robot"]["robot_variant"] == "g1_v2"
    assert math.degrees(tb.scanner_hfov) == pytest.approx(32.8, abs=0.05)              # the scanner, not the D405
    w = tb._capture_wrist()
    assert tb.robot.hand_calls == ["left_d405"] and w["camera"] == "left_d405" and w["rgb"].shape == (480, 848, 3)
    assert np.allclose(w["T_root_cam"], tb.kin.pose(tb._q(), "hand_cam_left_d405_optical"))
    v1 = tbmod.Toolbox(InfoRobot(_info("g1_v1")), box_detector=object())
    w1 = v1._capture_wrist()
    assert v1.robot.hand_calls == [] and w1["camera"] == "scanner"
    assert np.allclose(w1["T_root_cam"], v1.kin.pose(v1._q(), "wrist_cam_optical"))
    for info in (_info(None), {k: v for k, v in _info("g1_v1").items() if k != "robot_variant"},     # an old server
                 _info("g1_v2", hand_cameras={}), _info("g1_v2", wrist_side="right"),
                 _info("g1_v2", hand_cameras={"left_d405": {"side": "right"}}), _info("g1_v2", head_camera_model="upstream")):
        with pytest.raises(ValueError):
            tbmod.Toolbox(InfoRobot(info), box_detector=object())


def test_variant_texts_and_tool_descriptions():
    from g1agent.harness.registry import openai_tools
    v1, v2 = _v("g1_v1")["robot"], _v("g1_v2")["robot"]
    d = lambda texts: {t["function"]["name"]: t["function"]["description"] for t in openai_tools(None, texts)}
    a, b = d(v1["tool_texts"]), d(v2["tool_texts"])
    assert "D405" in b["look_wrist"] and "D405" not in a["look_wrist"] and a["plan_scans"] == b["plan_scans"]
    assert "scanner's own view" in b["scan"]
    with pytest.raises(ValueError):
        openai_tools(None, {"no_such_tool": "x"})
    tb = bare_toolbox(InfoRobot(_info("g1_v2")), variant="g1_v2")
    tb._capture_wrist = lambda: {"t": 1.0, "rgb": np.zeros((2, 2, 3), np.uint8)}
    assert "D405" in tb.look_wrist()[0]["reason"]


def test_scan_from_the_pre_pose_only_when_the_hand_camera_is_the_scanner():
    """A decode in the hand-camera image during refine proves the code readable; only where that camera IS the scanner
    imager (g1_v1) does it also mean "the scanner reads from the pre-pose" (scan's move-back fallback)."""
    for variant, expect in (("g1_v1", True), ("g1_v2", False)):
        tb = bare_toolbox(FakeRobot(), variant=variant)
        plan_of(tb, 1)
        tb.next_barcode()
        tb.ws.barcodes["bc0"].wrist_read = "A"
        tb.ws.wrist_target = {"pos": [0, 0, 0], "offset": 0.2, "forward": [1.0, 0, 0], "barcode_id": "bc0"}
        tb.refine_with_wrist = lambda: {"status": "partial", "reason": "x", "data": {}}
        tb.move_arm = lambda pose: {"status": "fail" if pose == "final" else "ok", "reason": "", "data": {}}
        r = tb.scan()
        assert any("move_arm_pre_where_read" in t for t in r["data"]["retries"]) is expect


def test_detect_wrist_keeps_the_detection_at_the_projected_estimate(monkeypatch):
    """Several labels in the wide hand-camera view: the one where the current estimate projects is kept, not the one
    nearest the image centre."""
    tb = bare_toolbox(FakeRobot(), variant="g1_v2")
    K = np.array([[446.8, 0, 424.0], [0, 446.8, 240.0], [0, 0, 1]])
    T_root_cam = np.eye(4); T_root_cam[:3, :3] = np.array([[0, 0, 1.0], [-1.0, 0, 0], [0, -1.0, 0]])
    frame = {"rgb": np.zeros((480, 848, 3), np.uint8), "K": K, "T_root_cam": T_root_cam, "T_odom_root": np.eye(4)}
    tb._capture_wrist = lambda: frame
    det = lambda c: {"center": np.array(c, float), "corners": np.zeros((4, 2)), "kind": "barcode"}
    monkeypatch.setattr(tbmod, "detect_barcodes", lambda rgb, roi, cfg: [det([430, 240]), det([170, 240])])
    _, rec = tb._detect_wrist()
    assert rec["center"][0] == 430                                       # no estimate: nearest the image centre
    _, rec = tb._detect_wrist(np.array([0.35, 0.2, 0.0]))                 # 0.2 m to the left at 0.35 m -> u ~ 169
    assert rec["center"][0] == 170


# ------------------------------------------------------------------ review fixes (2026-11-19)
def _qr_frame(u_sym, side=0.05, dist=0.30, fx=446.8, W=848, H=480):
    """Hand-camera frame looking straight at the plane x = dist with one QR symbol centred at pixel column u_sym."""
    import cv2
    qr = cv2.QRCodeEncoder.create().encode("NEIGHBOUR-01")
    ys, xs = np.where(qr < 128)
    sym = qr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    s = fx * side / dist / sym.shape[0]
    M = np.array([[s, 0, u_sym - s * sym.shape[1] / 2], [0, s, H / 2 - s * sym.shape[0] / 2]])
    img = cv2.warpAffine(sym, M, (W, H), flags=cv2.INTER_NEAREST, borderValue=255)
    K = np.array([[fx, 0, W / 2], [0, fx, H / 2], [0, 0, 1.0]])
    T = np.eye(4); T[:3, :3] = np.array([[0, 0, 1.0], [-1.0, 0, 0], [0, -1.0, 0]])
    return ({"rgb": cv2.cvtColor(img, cv2.COLOR_GRAY2RGB), "K": K, "T_root_cam": T, "T_odom_root": np.eye(4)},
            {"center": np.array([dist, 0.0, 0.0]), "normal": np.array([-1.0, 0.0, 0.0])})


def test_symbol_width_only_from_the_detected_label():
    """Only a neighbour decodes (the planned label's bars are not resolved): no width and no read are taken from it."""
    tb = bare_toolbox(FakeRobot(), variant="g1_v2")
    frame, plane = _qr_frame(150.0)                                   # the neighbour, ~0.18 m left of the detection
    anchor = np.array([0.30, 0.0, 0.0])                               # the detection's point on the plane (image centre)
    w = tb._wrist_symbol_width(frame, {"center": [424.0, 240.0]}, plane, anchor, 0.10)
    assert w["width_m"] is None and "content" not in w and w["decoded_other"] == ["NEIGHBOUR-01"]
    w = tb._wrist_symbol_width(frame, {"center": [150.0, 240.0]}, plane)          # the detection is that symbol
    assert w["width_m"] == pytest.approx(0.05, rel=0.03) and w["content"] == "NEIGHBOUR-01"
    # plane-distance acceptance: a detection centre just off the outline, within one label width on the plane
    w = tb._wrist_symbol_width(frame, {"center": [215.0, 240.0]}, plane, np.array([0.30, 0.0 + (424 - 215) / 446.8 * 0.30, 0.0]), 0.10)
    assert w["content"] == "NEIGHBOUR-01"


def test_g1_v1_explore_prompt_is_the_pre_port_text():
    """agent_assist explore prompt on g1_v1, tasks 1-3: byte-identical to the v2 prompt before the port (pinned)."""
    from g1agent.orchestrators.agent_assist import explore_prompt_kw
    from g1agent.orchestrators.common import load_prompt
    from g1agent.tools.scan_order import parse_targets
    old = ("You help a Unitree G1 humanoid robot find {goal} on a table. The robot's fixed procedure could not\n"
           "proceed: S\nThe attached image is the robot's head camera now (it looks down steeply; far objects appear at "
           "the top edge or above it).\nExploration moves made so far: none\nChoose ONE exploration move by calling "
           "explore (left / right / back / forward <= 0.5 m, turn_left / turn_right <= 15 deg),\nor call give_up if "
           "exploring cannot help. Do not output coordinates.\nBefore the tool call, write ONE short sentence in your "
           "message text explaining your choice (what you see in the image\nand why this move helps; it is logged as "
           "your reasoning); then call the tool.")
    tb = bare_toolbox(FakeRobot(), variant="g1_v1")
    for task, sel, goal in ((1, "left", "the left cardboard box"), (2, "right", "the right cardboard box"),
                            (3, "all", "every cardboard box")):
        text, _ = load_prompt("task1_agent_assist_explore", situation="S", history="none", **explore_prompt_kw(task, sel, tb))
        assert text == old.format(goal=goal)
    tb2 = bare_toolbox(FakeRobot(), variant="g1_v2")
    tb2.scan_targets = parse_targets({"rows_from_top": [2]})
    kw = explore_prompt_kw(3, "all", tb2)
    assert "30 deg" in kw["head_view"] and "on a table" not in kw["goal"] and "rows 2 from the top" in kw["goal"]


def test_variant_head_view_and_final_refine_texts_in_the_prompts():
    from g1agent.orchestrators import agent_full as af
    from g1agent.orchestrators.common import load_prompt
    for variant in ("g1_v1", "g1_v2"):
        tb = bare_toolbox(FakeRobot(), variant=variant)
        tx = tb.cfg["robot"]["texts"]
        t1, v = load_prompt("task1_agent_full", selector="left", max_tool_calls=1, max_explore_steps=1, head_view=tx["head_view"])
        assert v == "prompt: task1_agent_full v3" and tx["head_view"] in t1 and "{" not in t1
        t3, _ = load_prompt("task3_agent_full", selector="all", max_tool_calls=1, max_explore_steps=1, per_barcode_calls=1,
                            **af.task3_prompt_kw(tb, True))
        assert ("refine_with_wrist again" in t3) is (variant == "g1_v2")
    assert "steeply" in _v("g1_v1")["robot"]["texts"]["head_view"] and "30 deg" in _v("g1_v2")["robot"]["texts"]["head_view"]
    assert "move_arm('final')" in _v("g1_v2")["robot"]["tool_texts"]["refine_with_wrist"]


def test_run_suite_row_reports_subset_outcomes():
    sys.path.insert(0, os.path.join(config.ROOT, "scripts"))
    import run_suite
    sc = {"id": "s", "suite": "x", "level": "L0", "factor": None}
    res = {"task": 3, "mode": "rule", "robot_variant": "g1_v2", "target_boxes": ["C1R4"], "tool_calls": 1, "explore_steps": 0,
           "wall_s": 1.0, "llm": None, "harness_finish": {"success": False, "missing_cells": [{"column": 1, "row": 3}],
                                                           "grid_warnings": ["w"]},
           "metrics": {"success": True, "n_non_target_reads": 2}}
    r = run_suite.row(sc, res)
    assert (r["n_non_target_reads"], r["n_missing_cells"], r["n_grid_warnings"], r["robot_variant"]) == (2, 1, 1, "g1_v2")
    assert {"n_non_target_reads", "n_missing_cells", "n_grid_warnings"} <= set(run_suite.COLS)
