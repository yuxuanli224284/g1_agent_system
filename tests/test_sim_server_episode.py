"""sim_server.Episode glue on CPU with a stand-in env (no Isaac): record_start / step hook / annotate / scan / stage /
record_stop, the I1 return shapes, hold-frame counts and a playable H.264 file. Skipped without an ffmpeg binary."""
import json
import math
import shutil
import subprocess
import sys
import types

import numpy as np
import pytest

from g1agent.robot import sim_server as S
from test_compositor import BODY, _gt, _gt4


def _ffmpeg():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return shutil.which("ffmpeg")


class _Markers:
    def __init__(self):
        self.specs = {}


class _FakeEnv:
    policy_dt = 0.02
    scanner_on = True
    scanner_side = "left"
    hand_cams = {}                       # g1_v1: no hand camera

    def __init__(self):
        self.scenario = {"id": "t1_fake", "task": 1}
        self.rec_cams = {"main": object(), "side": object()}
        self.cfg = {"camera": {"recording": {"cameras": {"main": {"width": 1280, "height": 720},
                                                         "side": {"width": 640, "height": 360}}}},
                    "robot": {"hand_cameras": None}}
        self.read_cams = []
        self.t = 0.0
        self.x = -0.7
        self.decoded = []
        self.collision_model = types.SimpleNamespace(world_points=lambda: BODY + [self.x, 0.0, 0.78])
        self.posed = {}

    def get_ground_truth(self):
        g = _gt()
        g["barcodes"] = [dict(b, type="code128") for b in g["barcodes"]]
        return g

    def frame_state(self):
        Tr = np.eye(4); Tr[:3, 3] = [self.x, 0.0, 0.78]
        Tt = Tr.copy(); Tt[:3, 3] += [0.3, 0.2, 0.2]
        K = np.array([[1087.0, 0, 320], [0, 1087.0, 240], [0, 0, 1]])
        return {"t_sim": self.t, "T_world_root": Tr, "T_world_root_odom": Tr, "T_world_headcam": Tr, "head_K": K,
                "head_size": [480, 640], "min_gap": {"gap": 0.5, "body": "pelvis", "obstacle": "B0"},
                "T_world_tip": Tt, "T_world_wristcam": Tt, "wrist_K": K, "wrist_size": [480, 640]}

    def camera_rgb(self, name):
        self.read_cams.append(name)
        h, w = {"main": (720, 1280), "side": (360, 640), "left_d405": (480, 848)}.get(name, (480, 640))
        return np.full((h, w, 3), 80, np.uint8)

    def pose_rec_camera(self, name, T, hfov):
        self.posed[name] = (T, hfov)
        W, H = (1280, 720) if name == "main" else (640, 360)
        f = W / 2 / math.tan(math.radians(hfov) / 2)
        return np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])

    def settle_renders(self):
        pass

    def scan(self):
        ok = bool(self.decoded)
        return {"status": "ok" if ok else "no_read", "content": self.decoded[0]["content"] if ok else None,
                "decoded": self.decoded, "timestamp": self.t, "gt": {"hit": None}}

    def step(self, ep, n):
        for _ in range(n):
            self.t += self.policy_dt
            self.x += 0.01
            ep.on_step()


@pytest.mark.skipif(_ffmpeg() is None, reason="no ffmpeg binary")
def test_episode_recording_and_i1_methods(tmp_path, monkeypatch):
    exe = _ffmpeg()
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", types.SimpleNamespace(get_ffmpeg_exe=lambda: exe))
    from g1agent.viz.compositor import Compositor
    env, mk = _FakeEnv(), _Markers()
    actual = lambda: {"T_world_root": env.frame_state()["T_world_root"], "t_sim": env.t}
    ep = S.Episode(env, mk, Compositor(S.REC, S.VIZ["colors"], S.VIZ["sizes"]), actual)
    path = str(tmp_path / "ep" / "episode.mp4")
    r = ep.record_start(path, task=3)
    r_cams = ep.rec["cams"]
    assert r["task"] == 3 and [t["d"] for t in ep.rec["targets"]] == pytest.approx([0.20, 0.20], abs=0.002)
    every = round(1 / (S.REC["fps"] * env.policy_dt))
    assert r["every_steps"] == every and set(r["cameras"]) == {"main", "side"} and set(env.posed) == {"main", "side"}
    env.step(ep, 3 * every)                                        # 3 step frames (+1 at start)
    hold = int(round(S.REC["llm_hold_s"] * r["fps"]))
    assert ep.annotate({"kind": "llm", "purpose": "agent_full", "thought": "看一下", "tool_calls": [], "latency_s": 1.0}) == {"status": "ok"}
    ep.annotate({"kind": "tool", "name": "detect_boxes", "args": {}, "status": "ok", "reason": "",
                 "overlay": {"head_boxes": [{"id": "box_0", "bbox": [1, 2, 30, 40]}]}})
    st = ep.stage("task1_end", {"note": "x"})
    assert st["status"] == "ok" and st["actual"]["t_sim"] == env.t
    env.decoded = [{"content": "C1", "polygon": [[1, 1], [20, 1], [20, 9], [1, 9]], "center_px": [10, 5]}]
    rep = ep.scan()
    assert rep["content"] == "C1" and "polygon" in rep["decoded"][0]
    ep.scan()                                                      # duplicate read
    log = ep.ov.scans
    assert [e["content"] for e in log] == ["C1", "C1"] and log[0]["gt"] == {"hit": None} and len(log[0]["T_world_tip"]) == 4
    assert set(ep.stages["task1_end"]) >= {"t_sim", "actual"}
    posed = dict(env.posed)
    ep.stage("face_start", {"face": 0, "normal_odom": [-1.0, 0, 0], "n_planned": 2})     # single-face scene: cameras stay
    assert env.posed == posed and ep.rec["cams"] is r_cams
    out = ep.record_stop()
    det = int(round(S.REC["detection_hold_s"] * r["fps"]))
    assert out["frames"] == 1 + 3 + hold + det + 2 * det
    frames = [json.loads(l) for l in open(str(tmp_path / "ep" / "episode.frames.jsonl"))]
    assert sum(f["n"] for f in frames) == out["frames"] and frames[-1]["metrics"]["scanned"]["k"] == 1
    p = subprocess.run([exe, "-v", "error", "-i", path, "-f", "null", "-"], capture_output=True, text=True)
    assert p.returncode == 0 and not p.stderr.strip()
    ep.reset()
    assert ep.ov.scans == [] and ep.stages == {} and ep.stage_log == [] and ep.rec is None


def test_gt_standoff_follows_the_evaluated_task():
    from g1agent.common import config
    from g1agent.eval import evaluator
    qr = {"content": "Q", "size": [0.06, 0.06], "pattern_size": [0.04, 0.04]}   # a small code: adaptive d clamped to d_min
    assert S.gt_standoff_fn(2)(qr) == config.load("task2")["offset_m"]
    assert S.gt_standoff_fn(3)(qr) == evaluator.gt_standoff([0.04, 0.04], evaluator.scanner_hfov_gt())
    assert S.gt_standoff_fn(3)(qr) != S.gt_standoff_fn(2)(qr)
    with pytest.raises(ValueError):                                # no symbol size recorded: not scorable, no fallback
        S.gt_standoff_fn(3)({"content": "Q", "size": [0.06, 0.06]})


class _FakeEnv4(_FakeEnv):
    def get_ground_truth(self):
        g = _gt4()
        g["barcodes"] = [dict(b, type="code128") for b in g["barcodes"]]
        return g


@pytest.mark.skipif(_ffmpeg() is None, reason="no ffmpeg binary")
def test_episode_stage_log_and_face_cameras(tmp_path, monkeypatch):
    exe = _ffmpeg()
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", types.SimpleNamespace(get_ffmpeg_exe=lambda: exe))
    from g1agent.viz import compositor as C
    env, mk = _FakeEnv4(), _Markers()
    actual = lambda: {"T_world_root": env.frame_state()["T_world_root"], "t_sim": env.t}
    ep = S.Episode(env, mk, C.Compositor(S.REC, S.VIZ["colors"], S.VIZ["sizes"]), actual)
    ep.record_start(str(tmp_path / "episode.mp4"), task=3)
    start = {k: v[0].copy() for k, v in env.posed.items()}
    ep.stage("task1_end", {"n": 1})
    env.step(ep, 2)
    ep.stage("task1_end", {"n": 2})                                    # a repeated stage: the dict keeps the first
    seen = []
    for face in range(4):
        ep.stage("face_start", {"face": face, "normal_odom": [0, 0, 0], "n_planned": 2})
        gt, fr = ep.rec["gt"], S.REC["framing"]
        want = C.face_framing(gt, ep.rec["targets"], ep.rec["body"], ep.rec["root_z"], fr, face)
        for name, (fwd, pts) in want.items():
            W, H = (1280, 720) if name == "main" else (640, 360)
            T, _ = C.fit_fixed_camera(pts, fwd, fr["hfov_deg"], W, H, fr["margin_frac"])
            assert np.allclose(env.posed[name][0], T) and np.allclose(ep.rec["cams"][name][1], T)
        seen.append(env.posed["main"][0].copy())
        env.step(ep, 2)                                                # frames composed with the face text + face obs pose
        ep.stage("face_end", {"face": face, "statuses": {f"bc{face}": "scanned"}})
    assert not np.allclose(seen[0], start["main"]) and all(not np.allclose(seen[i], seen[i + 1]) for i in range(3))
    assert ep.ov.face_text().startswith("面 face 4/4: left (+y)")
    ep.stage("scan_done", {})
    log = ep.stage_log
    assert [e["name"] for e in log] == ["task1_end", "task1_end"] + ["face_start", "face_end"] * 4 + ["scan_done"]
    assert ep.stages["task1_end"]["info"] == {"n": 1} and ep.stages["face_start"]["info"]["face"] == 0
    assert log[1]["info"] == {"n": 2} and log[1]["t_sim"] > log[0]["t_sim"] and set(log[0]) == {"name", "info", "t_sim", "actual"}
    ep.record_stop()


class _FakeEnvV2(_FakeEnv):
    """g1_v2: the left D405 on the scanner bracket; a subset scenario (target_boxes, evaluation only)."""
    hand_cams = {"left_d405": object()}

    def __init__(self):
        super().__init__()
        self.cfg["robot"] = {"hand_cameras": {"left_d405": {"parent_link": "left_wrist_yaw_link"}}}
        self.scenario = {"id": "sub_fake", "task": 3, "target_boxes": ["B0"]}


@pytest.mark.skipif(_ffmpeg() is None, reason="no ffmpeg binary")
def test_record_start_subset_targets_and_hand_camera_panel(tmp_path, monkeypatch):
    exe = _ffmpeg()
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", types.SimpleNamespace(get_ffmpeg_exe=lambda: exe))
    from g1agent.viz.compositor import Compositor
    env, mk = _FakeEnvV2(), _Markers()
    actual = lambda: {"T_world_root": env.frame_state()["T_world_root"], "t_sim": env.t}
    ep = S.Episode(env, mk, Compositor(S.REC, S.VIZ["colors"], S.VIZ["sizes"]), actual)
    composed = []
    real = ep.comp.compose
    ep.comp.compose = lambda c: (composed.append(c), real(c))[1]
    ep.record_start(str(tmp_path / "episode.mp4"), task=3)
    assert composed and not any(m["color"] == "green" and m["kind"] == "root" for m in composed[0]["markers"])
    assert any(m["color"] == "green" and m["kind"] == "pose" for m in composed[0]["markers"])     # the target points stay
    gt = env.get_ground_truth()
    assert len(gt["barcodes"]) == 2 and [t["box_id"] for t in ep.rec["targets"]] == ["B0"]     # METRICS n = 1, not 2
    assert "left_d405" in env.read_cams and "wrist" in env.read_cams                         # D405 panel + scanner inset
    env.step(ep, 2)
    assert ep.record_stop()["frames"] >= 1
    frames = [json.loads(l) for l in open(str(tmp_path / "episode.frames.jsonl"))]
    assert all(f["metrics"]["root"] is None for f in frames)            # no whole-wall GT obs pose on a subset task
    env.scenario["target_boxes"] = ["NOPE"]
    with pytest.raises(ValueError):
        ep.record_start(str(tmp_path / "bad.mp4"), task=3)
    ep.rec = None
    assert S.scanner_hand_camera(_FakeEnv()) is None and S.scanner_hand_camera(env) == "left_d405"


def test_robot_info_reports_variant_hand_cameras_and_scanner():
    """robot_info / get_hand_camera_obs on a stand-in g1_v2 env: what the agent chooses its model from."""
    class Cam:
        def __init__(self, f):
            import torch
            self.data = types.SimpleNamespace(intrinsic_matrices=torch.tensor([[[f, 0, 424.0], [0, f, 240.0], [0, 0, 1.0]]]))

    class Env:
        scanner_on, scanner_side, policy_dt = True, "left", 0.02
        hand_cams = {"left_d405": Cam(446.8)}
        robot = types.SimpleNamespace(data=types.SimpleNamespace(joint_names=["a"]))
        walker = object()
        arm = types.SimpleNamespace(limits={"left": [], "right": []})
        cfg = {"robot": {"variant": "g1_v2", "velocity_limits": {},
                         "hand_cameras": {"left_d405": {"parent_link": "left_wrist_yaw_link", "width": 848, "height": 480,
                                                        "hfov_deg": 87.0, "data_types": ["rgb", "distance_to_image_plane"]}},
                         "gripper": {"joints": {"left": [], "right": ["right_dex1_finger_joint_1"]}},
                         "scanner_mount": {"parent_link": "left_wrist_yaw_link"}},
               "scanner": {"held_by": "rigid"}, "camera": {"model": "d435i"}}
        ir_cams = None

        def get_obs(self):
            return {"K": np.eye(3), "rgb": np.zeros((480, 848, 3)), "wrist_K": np.eye(3), "wrist_rgb": np.zeros((480, 640, 3))}

        def hand_camera_obs(self, name):
            return {"rgb": np.zeros((480, 848, 3), np.uint8), "K": np.eye(3), "T_root_cam": np.eye(4), "depth": np.ones((480, 848))}

        def frame_state(self):
            return {"t_sim": 3.5}

    pytest.importorskip("torch")
    sys.path.insert(0, S.SYS["sim_root"])
    env = Env()
    info = S.robot_info(env)
    assert info["robot_variant"] == "g1_v2" and info["grippers"] == {"left": False, "right": True}
    hc = info["hand_cameras"]["left_d405"]
    assert hc["side"] == "left" and hc["size"] == [480, 848] and hc["depth"] and hc["K"][0][0] == pytest.approx(446.8)
    assert info["scanner"] == {"parent_link": "left_wrist_yaw_link", "held_by": "rigid"} and "bolted" in info["wrist_point"]
    assert info["wrist_size"] == [480, 640]                                  # the scanner imager, not the D405
    o = S.hand_cam_obs(env, "left_d405")
    assert o["status"] == "ok" and o["timestamp"] == 3.5 and "depth" not in o and o["name"] == "left_d405"
    assert "depth" in S.hand_cam_obs(env, "left_d405", depth=True)
    assert S.hand_cam_obs(env, "right_d405")["status"] == "error"
