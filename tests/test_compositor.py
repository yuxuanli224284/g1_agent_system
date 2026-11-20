"""Composite-video logic (g1agent/viz/compositor.py) on CPU: fixed-camera framing, overlay event state, per-frame
metrics, frame composition and the H.264 writer (skipped when no ffmpeg binary is available)."""
import math
import os
import shutil
import subprocess
import time

import cv2
import numpy as np
import pytest

from g1agent.common import config
from g1agent.eval import evaluator
from g1agent.viz import compositor as C

VIZ = config.load("viz")
REC = VIZ["recording"]
HFOV = evaluator.scanner_hfov_gt()
STANDOFF = lambda bc: evaluator.gt_standoff(evaluator.gt_symbol_size(bc), HFOV)   # the GT rule the sim server passes in
# stand-in for the robot's collision points in its root frame (root 0.78 m above the ground, body 1.3 m tall)
BODY = np.array([[x, y, z] for x in (-0.15, 0.25) for y in (-0.25, 0.25) for z in (-0.78, 0.52)])


def _gt(robot=(-0.7, 0.0, 0.0)):
    """t1-like scenario: table front x 1.30, two 0.4 x 0.3 x 0.3 boxes, one 0.10 x 0.05 label on each -x face."""
    boxes, bcs = [], []
    for i, y in enumerate((-0.25, 0.25)):
        boxes.append({"id": f"B{i}", "size": [0.3, 0.4, 0.3], "pos": [1.45, y, 0.9], "quat_wxyz": [1, 0, 0, 0], "distractor": False})
        bcs.append({"content": f"C{i}", "box_id": f"B{i}", "center": [1.2985, y, 0.9], "normal": [-1.0, 0, 0], "up": [0, 0, 1.0],
                    "size": [0.10, 0.05], "pattern_size": [0.08, 0.03], "distractor": False})
    return {"robot_init": {"pos": [robot[0], robot[1], 0.0], "yaw": robot[2]}, "boxes": boxes, "barcodes": bcs,
            "target_face": {"center": [1.30, 0.25, 0.9], "normal": [-1.0, 0, 0]},
            "task1_gt_obs_pose": {"pos": [0.70, 0.25, 0.0], "yaw": 0.0}}


FACE_N = {"-x": (-1.0, 0.0, 0.0), "-y": (0.0, -1.0, 0.0), "+x": (1.0, 0.0, 0.0), "+y": (0.0, 1.0, 0.0)}
# expected per-face scan order of _gt4 (each face in its own frame: forward -n, left n x z)
GT4_ORDER = {"-x": ["F0", "F1", "F2"], "-y": ["R0", "R1"], "+x": ["K0", "K1"], "+y": ["L0", "L1"]}


def _gt4(labelled=("-x", "-y", "+x", "+y"), robot=(-0.7, 0.0, 0.0)):
    """Free-standing stack of two 0.6 (x) x 0.4 (y) x 0.6 (z) boxes side by side along y (x 1.2..1.8, y -0.4..0.4),
    labels 0.10 x 0.05 on the faces in `labelled` (front: a column F0 over F1 at its left, F2 at its right; every other
    face two labels in one row) and a distractor on the front; I2 "faces" in tour order with a GT obs pose 0.6 m out."""
    boxes = [{"id": f"B{i}", "size": [0.6, 0.4, 0.6], "pos": [1.5, y, 0.9], "quat_wxyz": [1, 0, 0, 0], "distractor": False}
             for i, y in enumerate((-0.2, 0.2))]
    lab = {"-x": [("F2", (1.1985, -0.2, 0.9)), ("F1", (1.1985, 0.2, 0.8)), ("F0", (1.1985, 0.2, 1.0))],
           "-y": [("R1", (1.65, -0.4015, 0.9)), ("R0", (1.35, -0.4015, 0.9))],
           "+x": [("K1", (1.8015, 0.2, 0.9)), ("K0", (1.8015, -0.2, 0.9))],
           "+y": [("L1", (1.35, 0.4015, 0.9)), ("L0", (1.65, 0.4015, 0.9))]}
    bcs = [{"content": c, "box_id": "B0" if p[1] < 0 else "B1", "center": list(p), "normal": list(FACE_N[f]),
            "up": [0, 0, 1.0], "size": [0.10, 0.05], "pattern_size": [0.08, 0.03], "distractor": False, "face": f}
           for f in labelled for c, p in lab[f]]
    bcs.append({"content": "DX", "box_id": "B1", "center": [1.1985, 0.0, 0.7], "normal": [-1.0, 0, 0], "up": [0, 0, 1.0],
                "size": [0.10, 0.05], "pattern_size": [0.08, 0.03], "distractor": True, "face": "-x"})
    allc = np.vstack([C.box_corners(b["pos"], b["size"], b["quat_wxyz"]) for b in boxes])
    faces = []
    for f in ("-x", "-y", "+x", "+y"):
        n = np.array(FACE_N[f])
        cs = allc[np.isclose(allc @ n, (allc @ n).max())]
        c = cs.mean(axis=0)
        p = c + 0.6 * n
        faces.append({"face": f, "normal": n.tolist(), "center": c.tolist(), "corners": cs.tolist(),
                      "gt_obs_pose": {"pos": [float(p[0]), float(p[1]), 0.0], "yaw": math.atan2(-n[1], -n[0]), "d_obs": 0.6},
                      "contents": [b["content"] for b in bcs if b["face"] == f]})
    return {"robot_init": {"pos": [robot[0], robot[1], 0.0], "yaw": robot[2]}, "boxes": boxes, "barcodes": bcs,
            "target_face": {k: faces[0][k] for k in ("center", "normal", "corners")},
            "task1_gt_obs_pose": faces[0]["gt_obs_pose"], "faces": faces}


def _in_frame(K, T, pts, W, H, margin):
    uv, z = C.project(K, T, pts)
    tol = 1e-6
    return (np.all(z > 0) and np.all(uv[:, 0] >= margin * W - tol) and np.all(uv[:, 0] <= (1 - margin) * W + tol)
            and np.all(uv[:, 1] >= margin * H - tol) and np.all(uv[:, 1] <= (1 - margin) * H + tol)), uv


def test_fit_fixed_camera_contains_points_tightly():
    rng = np.random.default_rng(0)
    for _ in range(20):
        pts = rng.uniform(-2, 2, (30, 3))
        fwd = rng.normal(size=3); fwd[2] = -abs(fwd[2]) * 0.3
        T, K = C.fit_fixed_camera(pts, fwd, 60.0, 1280, 720, 0.05)
        ok, uv = _in_frame(K, T, pts, 1280, 720, 0.05)
        assert ok
        # tight: the content touches the margin on both sides of at least one image axis
        lo_u, hi_u = uv[:, 0].min() - 0.05 * 1280, 0.95 * 1280 - uv[:, 0].max()
        lo_v, hi_v = uv[:, 1].min() - 0.05 * 720, 0.95 * 720 - uv[:, 1].max()
        assert min(max(lo_u, hi_u), max(lo_v, hi_v)) < 1e-3
        assert np.allclose(T[:3, 2], fwd / np.linalg.norm(fwd))


def test_gt_targets_use_the_evaluator_standoff():
    # the nominal 0.10 m label lands at the validated 0.20 m, with the evaluator's own rule
    tg = C.gt_targets(_gt(), STANDOFF)
    assert [t["content"] for t in tg] == ["C0", "C1"]
    for t in tg:
        assert t["d"] == pytest.approx(evaluator.gt_standoff([0.08, 0.03], HFOV)) and abs(t["d"] - 0.20) < 0.002
        assert np.allclose(t["pos"], t["center"] + t["d"] * t["normal"])


@pytest.mark.parametrize("robot", [(-0.7, 0.0, 0.0), (-3.7, 0.4, 0.3), (0.75, 0.1, 0.0)])
def test_framing_places_main_behind_left_and_side_left_of_boxes(robot):
    gt = _gt(robot)
    tg = C.gt_targets(gt, STANDOFF)
    fr = REC["framing"]
    for name, (fwd, pts) in C.framing(gt, tg, BODY, 0.78, fr).items():
        W, H = (1280, 720) if name == "main" else (640, 360)
        T, K = C.fit_fixed_camera(pts, fwd, fr["hfov_deg"], W, H, fr["margin_frac"])
        assert _in_frame(K, T, pts, W, H, fr["margin_frac"])[0]
        c = T[:3, 3]
        if name == "main":                     # behind-left of the robot start, in its frame
            x, y, yaw = robot
            d = c[:2] - [x, y]
            fx, lx = d @ [math.cos(yaw), math.sin(yaw)], d @ [-math.sin(yaw), math.cos(yaw)]
            assert fx < 0 and lx > 0 and c[2] > 0
        else:                                  # beyond the boxes on the +y side, in front of the face plane
            assert c[1] > 0.25 + 0.2 and c[0] < 1.30 and c[2] > 0


def test_overlay_state_events_and_detections_pinned_to_their_frame():
    ov = C.OverlayState()
    assert ov.on_event({"kind": "llm", "thought": "x", "tool_calls": []}, 1.0) == "llm"
    boxes = [{"id": "box_0", "bbox": [10, 10, 50, 50]}, {"id": "box_1", "bbox": [300, 10, 350, 50]}]
    assert ov.on_event({"kind": "tool", "name": "detect_boxes", "overlay": {"head_boxes": boxes}}, 2.0) == "detection"
    assert ov.on_event({"kind": "tool", "name": "approach", "overlay": {"target_box": "box_1"}}, 2.0) is None
    assert ov.head_detections_at(2.0)["boxes"] == boxes and ov.head_detections_at(2.02) is None
    assert ov.n_boxes == 2 and ov.target_text() == "box_1 / right"
    assert ov.on_event({"kind": "tool", "name": "detect_boxes", "overlay": {"head_boxes": []}}, 3.0) is None
    assert ov.n_boxes == 0
    with pytest.raises(ValueError):
        ov.on_event({"kind": "bogus"}, 3.0)


def test_current_target_by_content_then_by_fresh_harness_target():
    gt = _gt()
    tg = C.gt_targets(gt, STANDOFF)
    ov = C.OverlayState()
    near_c1 = {"pos": tg[1]["pos"] + [0.0, 0.02, 0.01], "t_set": 1.0}
    assert ov.current_target(tg, [near_c1])["content"] == "C1"           # no current_barcode: nearest label
    ov.on_event({"kind": "tool", "name": "next_barcode", "overlay": {"current_barcode": "bc_0",
                 "scan_plan": [{"id": "bc_0", "status": "pending", "content": None}]}}, 5.0)
    assert ov.current_target(tg, [near_c1]) is None                       # stale target from before bc_0 was chosen
    near_c0 = {"pos": tg[0]["pos"] + [0.05, 0.0, 0.0], "t_set": 6.0}
    assert ov.current_target(tg, [near_c0, near_c1])["content"] == "C0"
    ov.on_event({"kind": "tool", "name": "scan", "overlay": {"scan_plan": [{"id": "bc_0", "status": "scanned", "content": "C1"}]}}, 7.0)
    assert ov.current_target(tg, [near_c0])["content"] == "C1"            # decoded content wins


def test_frame_metrics_face_frame_and_dedupe():
    gt = _gt()
    tg = C.gt_targets(gt, STANDOFF)
    Tr = np.eye(4); Tr[:3, 3] = [0.70 - 0.03, 0.25 + 0.02, 0.78]       # 3 cm farther from the face, 2 cm to the left
    Tt = np.eye(4); Tt[:3, 3] = tg[1]["pos"] + [0.01, -0.02, 0.005]
    scans = [{"status": "ok", "content": "C1"}, {"status": "ok", "content": "C1"}, {"status": "no_read", "content": None},
             {"status": "ok", "content": "XX"}]
    m = C.frame_metrics(gt, {"t_sim": 3.0, "T_world_root": Tr, "T_world_tip": Tt}, tg, tg[1], scans)
    assert m["root"]["dn"] == pytest.approx(0.03) and m["root"]["dl"] == pytest.approx(0.02)
    assert m["tip"]["dn"] == pytest.approx(-0.01) and m["tip"]["dl"] == pytest.approx(-0.02) and m["tip"]["dz"] == pytest.approx(0.005)
    assert m["scanned"]["k"] == 1 and m["scanned"]["n"] == 2 and m["scanned"]["not_gt"] == ["XX"]
    assert m["root"]["frozen_t"] is None
    Tw = np.eye(4); Tw[:3, 3] = [0.0, -0.5, 0.78]                       # walked on to a scan stance after the task-1 end
    m2 = C.frame_metrics(gt, {"t_sim": 9.0, "T_world_root": Tw, "T_world_tip": Tt}, tg, tg[1], scans,
                         {"t_sim": 3.0, "actual": {"T_world_root": Tr}})
    assert m2["root"]["frozen_t"] == 3.0 and m2["root"]["dn"] == pytest.approx(0.03) and m2["root"]["dl"] == pytest.approx(0.02)


def test_compose_full_frame():
    comp = C.Compositor(REC, VIZ["colors"], VIZ["sizes"])
    gt = _gt()
    tg = C.gt_targets(gt, STANDOFF)
    cams = {}
    for name, (fwd, pts) in C.framing(gt, tg, BODY, 0.78, REC["framing"]).items():
        W, H = (1280, 720) if name == "main" else (640, 360)
        T, K = C.fit_fixed_camera(pts, fwd, 60.0, W, H, 0.05)
        cams[name] = (K, T)
    rng = np.random.default_rng(1)
    img = lambda h, w: rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    Tr = np.eye(4); Tr[:3, 3] = [-0.7, 0, 0.78]
    m = C.frame_metrics(gt, {"t_sim": 1.0, "T_world_root": Tr, "T_world_tip": Tr}, tg, tg[0], [])
    frame = comp.compose({
        "t_sim": 1.0, "hold": "llm", "images": {"main": img(720, 1280), "side": img(360, 640), "ego": img(480, 640), "wrist": img(480, 640)},
        "cams": cams, "markers": [{"color": "green", "kind": "root", "pos": [0.7, 0.25, 0], "yaw": 0.0},
                                  {"color": "yellow", "kind": "root", "pos": [-0.7, 0, 0.78], "yaw": 0.0, "body_z": 0.78,
                                   "label": "root (pelvis)"},
                                  {"color": "yellow", "kind": "pose", "pos": tg[0]["pos"], "dir": [1.0, 0, 0]},
                                  {"color": "red", "kind": "point", "pos": [0.5, 0, 0.5]}],
        "head_det": {"boxes": [{"id": "box_0", "bbox": [10, 20, 100, 200]}], "labels": [{"id": "l", "corners_px": [[1, 1], [9, 1], [9, 5], [1, 5]]}]},
        "target_box": "box_0", "ego_text": "boxes seen: 1; localised target: box_0", "wrist_polys": [[[10, 10], [60, 10], [60, 40], [10, 40]]],
        "wrist_text": "scanned: no", "agent": {"stage": {"name": "task1_end", "t_sim": 0.5}, "scan_text": "扫码",
                                              "llm": {"purpose": "agent_full", "thought": "我先看看箱子。" * 40, "latency_s": 1.2,
                                                      "tool_calls": [{"name": "approach", "arguments": {"target": "root_target"}}]},
                                              "tool": {"name": "detect_boxes", "args": {}, "status": "ok", "reason": ""}},
        "metrics": m})
    assert frame.shape == (1080, 1920, 3) and frame.dtype == np.uint8
    import os
    if os.environ.get("COMPOSE_PNG"):
        cv2.imwrite(os.environ["COMPOSE_PNG"], frame[:, :, ::-1])


def _ffmpeg():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return shutil.which("ffmpeg")


def _decoded_frames(exe, path):
    r = subprocess.run([exe, "-v", "error", "-i", path, "-f", "null", "-"], capture_output=True, text=True)
    p = subprocess.run([exe, "-i", path, "-map", "0:v", "-c", "copy", "-f", "null", "-"], capture_output=True, text=True)
    frames = [ln for ln in p.stderr.splitlines() if ln.startswith("frame=")]
    n = int(frames[-1].split("frame=")[1].split()[0]) if frames else 0
    return r.returncode, r.stderr, n, p.stderr


@pytest.mark.skipif(_ffmpeg() is None, reason="no ffmpeg binary")
def test_h264_writer_plays_and_survives_kill(tmp_path):
    exe = _ffmpeg()
    path = str(tmp_path / "ok.mp4")
    w = C.H264Writer(path, 320, 180, 5.0, exe)
    for i in range(30):
        w.write(np.full((180, 320, 3), i * 8, np.uint8), repeat=1 + (i == 3) * 4)
    assert w.close() == 34
    code, err, n, info = _decoded_frames(exe, path)
    assert code == 0 and not err.strip() and n == 34
    assert "h264" in info and "yuv420p" in info
    # killed writer: the fragmented file must still decode up to its last complete fragment
    path2 = str(tmp_path / "killed.mp4")
    w = C.H264Writer(path2, 320, 180, 5.0, exe)
    for i in range(200):
        w.write(np.full((180, 320, 3), (i * 5) % 255, np.uint8))
    w.proc.stdin.flush()
    time.sleep(2.0)
    w.proc.kill(); w.proc.wait()
    code, err, n, _ = _decoded_frames(exe, path2)
    assert n > 0 and "moov atom not found" not in err
    bad = C.H264Writer(str(tmp_path / "bad.mp4"), 320, 180, 5.0, exe)
    with pytest.raises(ValueError):
        bad.write(np.zeros((10, 10, 3), np.uint8))
    bad.proc.kill(); bad.proc.wait()
    with pytest.raises(RuntimeError):            # a dead encoder is reported, not written into silently
        bad.write(np.zeros((180, 320, 3), np.uint8))


def test_wrap_keeps_text_and_width():
    comp = C.Compositor(REC, VIZ["colors"], VIZ["sizes"])
    text = "两个箱子都在视野中, the left one is the target_box_1234567890; 我先计算观察位姿。\n\nsecond paragraph"
    lines = list(comp._wrap(text, comp.small, 120))
    assert all(comp.small.getlength(ln) <= 120 for ln in lines if len(ln) > 1)
    assert "".join(lines) == text.replace("\n", "") and lines.count("") == 1


# ------------------------------------------------------------------ multi-face stack (I2 / I3)
def test_face_framing_single_face_is_none():
    tg = C.gt_targets(_gt(), STANDOFF)
    assert C.face_framing(_gt(), tg, BODY, 0.78, REC["framing"], 0) is None            # legacy GT: no "faces"
    g = _gt4(labelled=("-x",))
    assert C.face_framing(g, C.gt_targets(g, STANDOFF), BODY, 0.78, REC["framing"], 0) is None   # one labelled face


@pytest.mark.parametrize("face_idx,quadrant", [(0, (-1, -1)), (1, (1, -1)), (2, (1, 1)), (3, (-1, 1))])
def test_face_framing_main_diagonal_side_on_face(face_idx, quadrant):
    gt = _gt4()
    tg = C.gt_targets(gt, STANDOFF)
    fr = REC["framing"]
    cams = C.face_framing(gt, tg, BODY, 0.78, fr, face_idx)
    centre = np.array([1.5, 0.0])
    face = gt["faces"][face_idx]
    n, l, _ = C.face_axes(face["normal"])
    for name, (fwd, pts) in cams.items():
        W, H = (1280, 720) if name == "main" else (640, 360)
        T, K = C.fit_fixed_camera(pts, fwd, fr["hfov_deg"], W, H, fr["margin_frac"])
        assert _in_frame(K, T, pts, W, H, fr["margin_frac"])[0]
        c = T[:3, 3]
        if name == "main":            # on the diagonal between this face and the next one of the tour (ccw)
            d = c[:2] - centre
            assert np.sign(d[0]) == quadrant[0] and np.sign(d[1]) == quadrant[1]
            assert abs(abs(d[0]) - abs(d[1])) < 0.5 * np.linalg.norm(d)
            # content: the current face's and the next face's targets, the robot at both GT obs poses
            nxt = gt["faces"][(face_idx + 1) % 4]
            for f in (face, nxt):
                ob = f["gt_obs_pose"]
                assert any(np.allclose(p, C.place_points(BODY, ob["pos"][0], ob["pos"][1], ob["yaw"], 0.78)[0]) for p in pts)
        else:                         # beyond the face's l edge, in front of its plane
            assert (c - np.asarray(face["center"])) @ l > 0.4 and (c - np.asarray(face["center"])) @ n > 0
    side_contents = {t["content"] for t in tg if any(np.allclose(t["pos"], p) for p in cams["side"][1])}
    assert side_contents == set(GT4_ORDER[face["face"]])


def test_face_framing_skips_unlabelled_face_with_ccw_bisector():
    gt = _gt4(labelled=("-x", "+x"))            # right / left sides unlabelled: front -> back, walking along -y
    tg = C.gt_targets(gt, STANDOFF)
    fwd, _ = C.face_framing(gt, tg, BODY, 0.78, REC["framing"], 0)["main"]
    h = -fwd[:2] / np.linalg.norm(fwd[:2])        # horizontal direction from the content to the camera
    assert h == pytest.approx([0.0, -1.0], abs=1e-9)
    with pytest.raises(ValueError):
        g = _gt4(); g["faces"] = g["faces"][:3]
        C.face_framing(g, C.gt_targets(g, STANDOFF), BODY, 0.78, REC["framing"], 3)   # no +y side in the GT


def test_overlay_face_text_and_compose():
    ov = C.OverlayState()
    assert ov.face_text() is None
    ov.on_stage("face_start", {"face": 1, "normal_odom": [0, -1, 0], "n_planned": 2}, 5.0)
    assert ov.face_text().startswith("面 face 2/4: right (-y)") and "planned 2" in ov.face_text()
    ov.on_stage("barcode_start", {"id": "bc3"}, 6.0)
    ov.on_stage("face_end", {"face": 1, "statuses": {"bc3": "scanned", "bc4": "unreachable"}}, 9.0)
    assert ov.face_text().endswith("done: 1/2 scanned") and ov.stage["name"] == "face_end"
    comp = C.Compositor(REC, VIZ["colors"], VIZ["sizes"])
    gt = _gt4()
    tg = C.gt_targets(gt, STANDOFF)
    cams = {k: C.fit_fixed_camera(pts, fwd, 60.0, *((1280, 720) if k == "main" else (640, 360)), 0.05)[::-1]
            for k, (fwd, pts) in C.face_framing(gt, tg, BODY, 0.78, REC["framing"], 1).items()}
    Tr = np.eye(4); Tr[:3, 3] = [1.5, -1.0, 0.78]
    m = C.frame_metrics(gt, {"t_sim": 9.0, "T_world_root": Tr, "T_world_tip": Tr}, tg, tg[3], [])
    z = lambda h, w: np.zeros((h, w, 3), np.uint8)
    frame = comp.compose({"t_sim": 9.0, "hold": None, "images": {"main": z(720, 1280), "side": z(360, 640), "ego": z(480, 640),
                                                                 "wrist": z(480, 640)},
                          "cams": cams, "markers": [], "ego_text": "", "wrist_text": "",
                          "agent": {"stage": ov.stage, "scan_text": "", "face": ov.face_text()}, "metrics": m})
    assert frame.shape == (1080, 1920, 3)


@pytest.mark.parametrize("with_hand", [False, True])
def test_wrist_panel_with_and_without_hand_camera(with_hand):
    """g1_v1: the WRIST panel is the scanner image. g1_v2: the D405 fills the panel and the scanner image (with its
    polygons, scanner pixels) is an inset in the panel's bottom-right corner."""
    comp = C.Compositor(REC, VIZ["colors"], VIZ["sizes"])
    gt = _gt()
    tg = C.gt_targets(gt, STANDOFF)
    cams = {k: C.fit_fixed_camera(pts, fwd, 60.0, *((1280, 720) if k == "main" else (640, 360)), 0.05)[::-1]
            for k, (fwd, pts) in C.framing(gt, tg, BODY, 0.78, REC["framing"]).items()}
    Tr = np.eye(4); Tr[:3, 3] = [-0.7, 0, 0.78]
    m = C.frame_metrics(gt, {"t_sim": 1.0, "T_world_root": Tr, "T_world_tip": Tr}, tg, tg[0], [])
    z = lambda h, w, c: np.full((h, w, 3), c, np.uint8)
    images = {"main": z(720, 1280, 0), "side": z(360, 640, 0), "ego": z(480, 640, 0), "wrist": z(480, 640, (200, 0, 0))}
    if with_hand:
        images["hand"] = z(480, 848, (0, 0, 200))
    frame = comp.compose({"t_sim": 1.0, "hold": None, "images": images, "cams": cams, "markers": [], "ego_text": "",
                          "wrist_text": "", "wrist_polys": [[[100, 100], [500, 100], [500, 300], [100, 300]]],
                          "agent": {"stage": None, "scan_text": ""}, "metrics": m})
    x, y, w, h = REC["panels"]["wrist"]
    centre = frame[y + h // 2, x + w // 2]
    corner = frame[y + h - REC["overlay"]["pad_px"] - 5, x + w - REC["overlay"]["pad_px"] - 5]
    if with_hand:
        assert tuple(centre) == (0, 0, 200) and tuple(corner) == (200, 0, 0)
    else:
        assert tuple(centre) == (200, 0, 0)


def test_subset_task_has_no_whole_wall_obs_pose_metrics():
    """A subset task (scenario target_boxes): the robot works from near its start, so the METRICS root line is n/a (the
    evaluator's rule), and the frame still composes."""
    comp = C.Compositor(REC, VIZ["colors"], VIZ["sizes"])
    gt = _gt()
    tg = C.gt_targets(gt, STANDOFF)[:1]
    Tr = np.eye(4); Tr[:3, 3] = [-0.7, 0, 0.78]
    full = C.frame_metrics(gt, {"t_sim": 1.0, "T_world_root": Tr, "T_world_tip": Tr}, tg, tg[0], [])
    sub = C.frame_metrics(gt, {"t_sim": 1.0, "T_world_root": Tr, "T_world_tip": Tr}, tg, tg[0], [], subset=True)
    assert full["root"] is not None and sub["root"] is None and sub["tip"] == full["tip"] and sub["scanned"]["n"] == 1
    cams = {k: C.fit_fixed_camera(pts, fwd, 60.0, *((1280, 720) if k == "main" else (640, 360)), 0.05)[::-1]
            for k, (fwd, pts) in C.framing(gt, tg, BODY, 0.78, REC["framing"]).items()}
    z = lambda h, w: np.zeros((h, w, 3), np.uint8)
    frame = comp.compose({"t_sim": 1.0, "hold": None, "images": {"main": z(720, 1280), "side": z(360, 640), "ego": z(480, 640),
                                                                 "wrist": z(480, 640)},
                          "cams": cams, "markers": [], "ego_text": "", "wrist_text": "", "agent": {"stage": None, "scan_text": ""},
                          "metrics": sub})
    assert frame.shape == (1080, 1920, 3)
