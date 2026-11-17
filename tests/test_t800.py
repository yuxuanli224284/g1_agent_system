"""CPU tests of the robot selection (G1 / EngineAI T800 Pro) on the harness side: config selection and task overrides,
the derived T800 observation-distance bound, the T800 URDF against the vendor URDF and the simulator's mount configs,
the right-arm IK, the T800 clearance model, set_base_height without a body command, and the robot prompts (G1 rendering
unchanged). No simulator / GPU.  run: python -m pytest tests/test_t800.py"""
import hashlib
import math
import os
import sys

import numpy as np
import pytest
import yaml

from g1agent.common import config

SIM = config.load("system")["sim_server"]["sim_root"]


@pytest.fixture
def t800(monkeypatch):
    monkeypatch.setenv(config.ROBOT_ENV, "t800")
    yield


# ------------------------------------------------------------------ config selection
def test_default_is_g1(monkeypatch):
    monkeypatch.delenv(config.ROBOT_ENV, raising=False)
    rc = config.load("robot")
    assert config.robot_model() == "g1" and rc["model"] == "g1" and rc["arm_side"] == "left"
    assert rc["urdf"] == "assets/g1_dex1_scanner.urdf" and rc["root_frame"] == "pelvis"
    assert config.load("task1")["d_obs_range"] == [0.60, 1.00] == config.load("task3")["d_obs_range"]


def test_g1_arm_joints_are_the_g1_pattern(monkeypatch):
    monkeypatch.delenv(config.ROBOT_ENV, raising=False)
    parts = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"]
    rc = config.load("robot")
    assert all(rc["arm_joints"][s] == [f"{s}_{p}_joint" for p in parts] for s in ("left", "right"))


def test_t800_selection(t800):
    rc = config.load("robot")
    assert rc["model"] == "t800" and rc["arm_side"] == "right" and rc["root_frame"] == "LINK_BASE"
    assert rc["urdf"] == "assets/t800pro_scanner.urdf"
    assert rc["ik"] == config._load("robot")["ik"]                      # inherited G1 blocks
    assert rc["body"]["calibration"] is None and rc["body"]["standing_clearance"]
    # the observation range override reaches task1 and, through inherits, task3; task2 is untouched
    assert config.load("task1")["d_obs_range"] == [0.60, 1.50] == config.load("task3")["d_obs_range"]
    assert config.load("task2") == config._load("task2")


def test_unknown_robot(monkeypatch):
    monkeypatch.setenv(config.ROBOT_ENV, "h1")
    with pytest.raises(ValueError):
        config.load("robot")


def test_t800_arm_joints_match_the_simulator(t800):
    sim = yaml.safe_load(open(os.path.join(SIM, "configs", "robot_t800.yaml")))
    rc = config.load("robot")
    assert rc["arm_joints"] == sim["arm_ik"]["joints"]
    assert rc["arm_side"] == sim["scanner_mount"]["side"]


# ------------------------------------------------------------------ observation distance (user decision: back up, head at 0)
def _floor_bound(nominal, margin, d0, step):
    """Smallest d on the grid d0 + k step at which the floor line of a face is inside the image with the margin."""
    T = np.asarray(nominal["T_root_cam"], float)
    K = np.asarray(nominal["K"], float)
    h_cam = nominal["root_height"] + T[2, 3]
    pitch = math.asin(-T[2, 2])                                         # optical axis below the horizontal
    v_low = nominal["height"] * (1 - margin)
    a = pitch + math.atan((v_low - K[1, 2]) / K[1, 1])
    d = h_cam / math.tan(a) + T[0, 3]                                   # camera ahead of the root by T[0, 3]
    return d0 + math.ceil((d - d0) / step - 1e-9) * step, d


def test_t800_obs_range_bound_derived_from_the_camera(t800):
    nominal = yaml.safe_load(open(os.path.join(SIM, "configs", "testsuite", "head_camera_nominal_t800.yaml")))
    t1 = config.load("task1")
    bound, d = _floor_bound(nominal, config.load("perception")["fov_margin_frac"], t1["d_obs_range"][0], t1["d_obs_step"])
    assert d == pytest.approx(1.452, abs=0.002)
    assert t1["d_obs_range"][1] == pytest.approx(bound)
    # the simulator's ground truth uses the same range (configs/testsuite/factors_t800.yaml)
    sys.path.insert(0, SIM)
    from g1_barcode_sim.testsuite import load_factors
    S = load_factors("t800")["task2_start"]
    assert S["d_obs_range"] == t1["d_obs_range"] and S["d_obs_step"] == t1["d_obs_step"]
    assert S["fov_margin_frac"] == config.load("perception")["fov_margin_frac"]


# ------------------------------------------------------------------ URDF / FK
def _vendor():
    sys.path.insert(0, SIM)
    from g1_barcode_sim.config import load_all
    from g1_barcode_sim.t800_robot import _sdk, urdf_fk
    cfg = load_all("s1_table", {"robot": {"model": "t800"}})
    path = _sdk(cfg["robot"], cfg["robot"]["asset"]["urdf"])
    if not os.path.exists(path):
        pytest.skip(f"vendor URDF {path} not available")
    return cfg, path, urdf_fk


def test_t800_urdf_matches_vendor_urdf_and_sim_frames(t800):
    from g1_barcode_sim.camera import T_from_xyz_rpy
    from g1_barcode_sim.scanner_frames import transforms
    from g1agent.kinematics.fk import Kinematics
    cfg, path, urdf_fk = _vendor()
    rc = cfg["robot"]
    kin = Kinematics()
    st = transforms(rc["scanner_mount"], cfg["scanner"])
    T_chain = np.eye(4)
    for j in rc["head_camera"]["chain"]:
        T_chain = T_chain @ T_from_xyz_rpy(j["xyz"], j["rpy"])
    rng = np.random.default_rng(0)
    names = list(kin.joint_q_index)
    assert len(names) == 43
    for _ in range(20):
        qd = {n: rng.uniform(kin.lower[kin.joint_q_index[n]], kin.upper[kin.joint_q_index[n]]) for n in names}
        q = kin.q_from_state(list(qd), list(qd.values()))
        for link, frame, T_off in (("LINK_HEAD_YAW", "head_cam_optical", T_chain), ("LINK_WRIST_ROLL_R", "scanner_tip", st["T_hand_tip"]),
                                   ("LINK_WRIST_ROLL_R", "wrist_cam_optical", st["T_hand_cam"]),
                                   ("LINK_WRIST_ROLL_R", "scanner_mount", st["T_hand_G"]), ("LINK_ANKLE_ROLL_L", "LINK_ANKLE_ROLL_L", np.eye(4))):
            root, T = urdf_fk(path, link, qd)
            assert root == "LINK_BASE"
            A, B = kin.pose(q, frame), T @ T_off
            assert np.linalg.norm(A[:3, 3] - B[:3, 3]) < 1e-5                     # USD float32 frames
            assert np.abs(A[:3, :3] - B[:3, :3]).max() < 1e-4


def test_t800_head_camera_matches_measured_nominal(t800):
    """FK at the joint state measured in the simulator = the simulator's own URDF-FK mount check (< 0.01 mm)."""
    from g1agent.kinematics.fk import Kinematics
    nominal = yaml.safe_load(open(os.path.join(SIM, "configs", "testsuite", "head_camera_nominal_t800.yaml")))
    kin = Kinematics()
    jq = {k: math.radians(v) for k, v in nominal["mount_check"]["joint_q_deg"].items()}
    T = kin.pose(kin.q_from_state(list(jq), list(jq.values())), config.load("robot")["head_cam_frame"])
    assert np.abs(T - np.asarray(nominal["mount_check"]["T_root_cam_fk"], float)).max() < 1e-5


# ------------------------------------------------------------------ IK
def test_t800_ik_right_arm_recovers_fk_targets(t800):
    from g1agent.kinematics.fk import Kinematics
    from g1agent.kinematics.ik import ArmIK
    kin = Kinematics()
    ik = ArmIK(kin)
    assert ik.joints[0] == "J27_SHOULDER_PITCH_R" and ik.frame == "scanner_tip" and len(ik.qi) == 7
    rng = np.random.default_rng(3)
    n_ok = 0
    for _ in range(10):
        q = np.zeros(kin.model.nq)
        q[ik.qi] = rng.uniform(ik.lo + 0.2 * (ik.hi - ik.lo), ik.hi - 0.2 * (ik.hi - ik.lo))
        T = kin.pose(q, ik.frame)
        r = ik.solve(np.zeros(kin.model.nq), T[:3, 3], T[:3, 0], roll_axis=T[:3, :3] @ ik.img_x_in_tip)
        n_ok += r["ok"]
        if r["ok"]:
            assert r["pos_err"] < 0.002 and r["angle_err_deg"] < 1.0 and r["roll_err_deg"] < 1.0
    assert n_ok >= 8


# ------------------------------------------------------------------ clearance model without squat levels
def _t800_geometry(kin):
    """Synthetic collision points on the T800 collider links: spheres (base, knees, elbow / wrist), hull points."""
    rng = np.random.default_rng(0)
    names = ["LINK_BASE", "LINK_HIP_PITCH_L", "LINK_KNEE_PITCH_L", "LINK_WAIST_YAW", "LINK_HEAD_YAW",
             "LINK_ELBOW_PITCH_R", "LINK_ELBOW_YAW_R", "LINK_WRIST_ROLL_R"]
    pts, rad, own = [], [], []
    for bi, n in enumerate(names):
        if n in ("LINK_BASE", "LINK_HEAD_YAW", "LINK_ELBOW_PITCH_R"):
            P, r = np.zeros((1, 3)), np.array([0.06])
        elif n == "LINK_KNEE_PITCH_L":                                    # sphere + cylinder rim points (mixed)
            P = np.vstack([np.zeros((1, 3)), rng.uniform(-0.04, 0.04, (8, 3))]); r = np.r_[0.05, np.zeros(8)]
        else:
            P, r = rng.uniform(-0.05, 0.05, (8, 3)), np.zeros(8)
        pts.append(P); rad.append(r); own.append(np.full(len(P), bi))
    return {"body_names": names, "points": np.vstack(pts), "radius": np.concatenate(rad), "owner": np.concatenate(own)}


def test_t800_body_model_standing_only(t800):
    from g1agent.kinematics.fk import Kinematics
    from g1agent.tools.body import BodyModel
    kin = Kinematics()
    B = config.load("robot")["body"]
    body = BodyModel(kin, _t800_geometry(kin), None, "right", B["links"])
    assert body.levels == []
    assert "LINK_HEAD_YAW" in body.legs and "LINK_WRIST_ROLL_R" in body.arm and "LINK_ELBOW_PITCH_R" in body.arm
    assert body.leg_shapes["LINK_KNEE_PITCH_L"][0] == "mixed" and body.leg_shapes["LINK_BASE"][0] == "spheres"
    assert body.shoulder_frame == "LINK_SHOULDER_PITCH_R" and 0.5 < body.arm_reach < 1.0
    d, name = body.self_clearance(np.zeros(kin.model.nq))
    assert np.isfinite(d) and name in body.legs
    with pytest.raises(KeyError):
        BodyModel(kin, _t800_geometry(kin), None, "right", dict(B["links"], arm=["LINK_NOPE"]))


class _NoBodyRobot:
    def get_robot_info(self):
        return {"robot_model": "t800", "head_K": np.eye(3), "head_size": [480, 640], "wrist_K": None, "wrist_size": None,
                "body_command": None}

    def get_odom(self):
        return {"T_odom_root": np.eye(4)}

    def set_body_pose(self, *a, **k):
        raise AssertionError("set_body_pose must not be called without a body command")


def test_t800_set_base_height_fails_gracefully(t800):
    from g1agent.tools import toolbox as tbmod
    tb = tbmod.Toolbox(_NoBodyRobot(), box_detector=object())
    r = tb.set_base_height(0.6)
    assert r["status"] == "fail" and "no body" in r["reason"]
    assert tb._standing()


def test_robot_mismatch_is_refused(monkeypatch):
    monkeypatch.delenv(config.ROBOT_ENV, raising=False)                # harness configured for G1, server says T800
    from g1agent.tools import toolbox as tbmod
    with pytest.raises(RuntimeError, match="t800"):
        tbmod.Toolbox(_NoBodyRobot(), box_detector=object())


# ------------------------------------------------------------------ prompts
KW = {"task1_agent_full": dict(selector="left", max_tool_calls=40, max_explore_steps=6),
      "task2_agent_full": dict(selector="left", max_tool_calls=40, max_explore_steps=6),
      "task3_agent_full": dict(selector="all", max_tool_calls=40, max_explore_steps=6, per_barcode_calls=28),
      "task1_agent_assist_explore": dict(goal="G", situation="S", history="H"),
      "recovery": dict(task=3, tool="T", result="R", attempt=1)}
# sha256 of the G1 prompts rendered with KW by the templates before the robot placeholders (git 3738045)
G1_SHA = {"task1_agent_full": "da4d18eb1dd49814a4af45c374bc63585624ae6c2153b39fc65a153bde5e2b67",
          "task2_agent_full": "3d4c3f96e0dbee098816968e53424e2b8453f28af5a2d930f3b32f9c0bfdbb21",
          "task3_agent_full": "d8f92b2efb9841e5dd3d0151d99b56a440ff4637b52ef2a7a5c71b37466f7cc4",
          "task1_agent_assist_explore": "765ae0bc55f19aebe887c0f7fe40ce7b811d033bc2bdb3141f9133509b8ba1f0",
          "recovery": "8433b078d04a25ed267fde907aa5d9f2a617f5034a1bc9520fd1a358cf106fe3"}


def test_g1_prompts_unchanged(monkeypatch):
    monkeypatch.delenv(config.ROBOT_ENV, raising=False)
    from g1agent.orchestrators.common import load_prompt
    for name, kw in KW.items():
        assert hashlib.sha256(load_prompt(name, **kw)[0].encode()).hexdigest() == G1_SHA[name], name


def test_t800_prompts(t800):
    from g1agent.orchestrators.common import load_prompt
    for name, kw in KW.items():
        txt = load_prompt(name, **kw)[0]
        assert "Unitree" not in txt and "G1" not in txt and "EngineAI T800" in txt and "steeply" not in txt
    assert "in its right hand" in load_prompt("task3_agent_full", **KW["task3_agent_full"])[0]
    assert "cannot squat" in load_prompt("task3_agent_full", **KW["task3_agent_full"])[0]


# ------------------------------------------------------------------ observation pose from a face cut by the image border
class _ObsRobot(_NoBodyRobot):
    def __init__(self, model="t800", root_z=0.0):
        self.model = model
        self.T = np.eye(4)
        self.T[2, 3] = root_z                                   # odometry carries the root height (as the sim's)

    def get_robot_info(self):
        return dict(super().get_robot_info(), robot_model=self.model, head_K=[[243.2, 0, 320], [0, 243.2, 240], [0, 0, 1]])

    def draw_markers(self, ms, frame="odom"):
        return {"status": "ok"}

    def get_odom(self):
        return {"T_odom_root": self.T}

    def move_root_relative(self, dx, dy, dyaw, timeout=10.0, pos_tol=None, yaw_tol=None, speed_scale=1.0):
        return {"status": "ok"}


def _obs_tb(model):
    from g1agent.harness.world_state import BoxEstimate
    from g1agent.tools import toolbox as tbmod
    nominal = yaml.safe_load(open(os.path.join(SIM, "configs", "testsuite", f"head_camera_nominal{'' if model == 'g1' else '_' + model}.yaml")))
    tb = tbmod.Toolbox(_ObsRobot(model, nominal["root_height"]), box_detector=object())
    T_or = tb.robot.T
    tb.head = {"rgb": np.zeros((480, 640, 3), np.uint8), "K": np.asarray(nominal["K"], float),
               "T_root_cam": np.asarray(nominal["T_root_cam"], float), "T_odom_root": T_or, "t": 7.0}
    tb.ws.boxes = {"box0": BoxEstimate("box0", [0, 0, 640, 480], 0.9, 7.0)}
    tb.ws.target_box_id = "box0"
    return tb


def _wall_face(x, z0, z1, t=7.0):
    C = np.array([[x, 0.6, z0], [x, -0.6, z0], [x, -0.6, z1], [x, 0.6, z1]], float)
    return {"center": C.mean(axis=0), "normal": np.array([-1.0, 0, 0]), "corners": C, "t": t}


def test_t800_backs_off_from_a_cut_face(t800):
    tb = _obs_tb("t800")
    # wall 0.6 m ahead seen only down to 0.93 m (the T800 view from 0.6 m): bottom corners at the image's lower edge
    tb.ws.boxes["box0"].face = _wall_face(0.6, 0.93, 1.25)
    r = tb.compute_obs_pose()
    assert r["status"] == "partial" and "bottom" in r["data"]["root_target_odom"]["cut_by_image_border"]
    assert r["data"]["root_target_odom"]["d_obs"] == pytest.approx(1.50)
    # the whole wall (floor to 1.25 m) from 1.5 m: not cut, the smallest fitting distance is chosen (= 1.50)
    tb.robot.T[0, 3] = 0.6 - 1.5                              # root 1.5 m in front of the face at x = 0.6
    tb.ws.boxes["box0"].face = _wall_face(0.6, 0.0, 1.25)
    r = tb.compute_obs_pose()
    assert r["status"] == "ok" and "cut_by_image_border" not in r["data"]["root_target_odom"]
    assert r["data"]["root_target_odom"]["d_obs"] == pytest.approx(1.50)


def test_g1_does_not_back_off(monkeypatch):
    monkeypatch.delenv(config.ROBOT_ENV, raising=False)
    assert "obs_back_off_cut_faces" not in config.load("robot")
    tb = _obs_tb("g1")
    tb.ws.boxes["box0"].face = _wall_face(0.6, 0.93, 1.25)
    r = tb.compute_obs_pose()
    assert "cut_by_image_border" not in r["data"]["root_target_odom"]


def test_t800_stance_arrival_without_body_command(t800):
    tb = _obs_tb("t800")
    tb.ws.stance_target = {"pos": [0.0, 0.0], "yaw": 0.0, "base_height": None}
    r = tb._approach("stance_target")
    assert r["status"] == "ok" and r["reason"] == "at the stance"
