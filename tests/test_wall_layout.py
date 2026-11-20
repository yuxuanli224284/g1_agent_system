"""CPU tests of the wall-layout task (approach A, branch g1v2-scanroom-layout; no simulator, no detector model, no GPU):
set_wall_layout validation, the wall side edges on synthetic head depth (plane_fit.wall_extent), the registration and
its failures, the layout plan (one kind-cell entry per requested box in the lattice order), the hand-camera look pose
(look_view.search_view on the real g1_v2 URDF and D405), search_cell with a fake hand camera and detector (label
found / no_label / partial / cut labels / the neighbour sliver), the stance re-registration under yaw drift, the
refusals of the label path for a cell entry, the Harness bookkeeping, the rule path and the prompt texts.

Synthetic scene (render_depth): the 6 x 5 box wall of the scan room, front face at odom x = WALL_X, y in [-0.6, 0.6],
z in [0, 1.25] (boxes 0.20 x 0.25 m, 0.40 m deep), the room wall 0.5 m behind it, the floor at odom z 0; head poses from
the real g1_v2 kinematics and upright view calibration; noise-free depth unless a test adds noise."""
import hashlib
import json
import math

import numpy as np
import pytest

from g1agent.common import config
from g1agent.common.transforms import T_from_xy_yaw, apply, inv_T
from g1agent.harness.guards import Harness
from g1agent.harness.registry import LAYOUT_ONLY, TASK_TOOLS, openai_tools, task_tools
from g1agent.harness.world_state import BarcodeEstimate
from g1agent.orchestrators import rule
from g1agent.perception import geometry as geo
from g1agent.perception import plane_fit
from g1agent.perception.look_view import HandView, search_view
from g1agent.tools import toolbox as tbmod
from test_task3_logic import FakeLLM, FakeLog, FakeRobot, ScriptedTB, bare_toolbox, read

SPEC_R234 = {"columns": "all", "rows_from_bottom": [2, 3, 4], "order": "rows"}
LAYOUT = {"columns": 6, "rows": 5, "box_w": 0.20, "box_h": 0.25, "base": "floor"}
WALL_X = 1.3
FX_HEAD = 424.0 / np.tan(np.radians(69.0) / 2)
K_HEAD = np.array([[FX_HEAD, 0, 424.0], [0, FX_HEAD, 240.0], [0, 0, 1]])
FX_D405 = 446.8
K_D405 = np.array([[FX_D405, 0, 424.0], [0, FX_D405, 240.0], [0, 0, 1]])
# g1_barcode_sim configs/robot_variants.yaml g1_v2 clearance_boxes on the D405's link (left_wrist_yaw_link)
BOXES = [{"name": "bracket", "link": "left_wrist_yaw_link", "center": [0.0815, 0.005, 0.0753], "size": [0.0841, 0.047, 0.1908]},
         {"name": "scanner_head", "link": "left_wrist_yaw_link", "center": [0.1111, 0.0049, 0.0931], "size": [0.092, 0.0752, 0.0461]},
         {"name": "scanner_handle", "link": "left_wrist_yaw_link", "center": [0.0999, 0.005, 0.0175], "size": [0.0676, 0.0521, 0.105]},
         {"name": "left_d405", "link": "left_wrist_yaw_link", "center": [0.0789, 0.005, 0.1418], "size": [0.0296, 0.0421, 0.0449]}]


# ------------------------------------------------------------------ fixtures
def kin_toolbox(spec=SPEC_R234, layout=LAYOUT, order=None):
    """g1_v2 kinematics + IK, the D435i colour intrinsics, the D405 with its occluder boxes, upright head views, obs_pose
    phase 1, the target spec and the wall layout; the root at the odom origin facing +x."""
    from g1agent.kinematics.fk import Kinematics
    from g1agent.kinematics.ik import ArmIK
    tb = bare_toolbox(FakeRobot(), variant="g1_v2")
    tb.kin = Kinematics(tb.cfg["robot"]["urdf"])
    tb.ik = ArmIK(tb.kin, tb.cfg["robot"])
    tb.side = tb.cfg["robot"]["arm_side"]
    tb.info = {"head_K": K_HEAD.tolist(), "head_size": [480, 848],
               "hand_cameras": {"left_d405": {"side": "left", "K": K_D405.tolist(), "size": [480, 848], "occluders": BOXES}}}
    tb.body_cmd = {"height_default": 0.74, "height_range": [0.3, 0.8], "rpy_default": [0, 0, 0]}
    q0 = tb.kin.q_from_state([], [])
    tb._q = lambda: q0.copy()
    tb.set_scan_targets(dict(spec, **({"order": order} if order else {})))
    tb.set_head_views("upright")
    tb.set_phase1_view("obs_pose")
    if layout is not None:
        tb.set_wall_layout(layout)
    return tb


def render_depth(Tc, K=K_HEAD, size=(480, 848), wall_x=WALL_X, half_w=0.6, top=1.25, depth_box=0.4, gap=0.5, floor_z=0.0,
                 poles=(), side_faces=False):
    """Depth (along the optical axis) of the synthetic scene seen from camera pose Tc (odom); NaN where nothing is hit.
    poles: axis-aligned boxes [lo, hi] (odom) in front of the wall; side_faces: also the stack's two side faces
    (y = +-half_w, depth_box deep: visible from beyond a wall end)."""
    h, w = size
    vs, us = np.mgrid[0:h, 0:w]
    Dc = np.stack([(us - K[0][2]) / K[0][0], (vs - K[1][2]) / K[1][1], np.ones(us.shape)], -1)
    D = Dc @ np.asarray(Tc)[:3, :3].T                # odom ray per pixel, unit optical-axis component: t = depth
    o = np.asarray(Tc)[:3, 3]
    best = np.full((h, w), np.inf)

    def cand(t, ok):
        nonlocal best
        best = np.minimum(best, np.where(ok & (t > 1e-6), t, np.inf))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (wall_x - o[0]) / D[..., 0]
        P = o + t[..., None] * D
        cand(t, (np.abs(P[..., 1]) <= half_w) & (P[..., 2] >= floor_z) & (P[..., 2] <= top))
        t = (top - o[2]) / D[..., 2]
        P = o + t[..., None] * D
        cand(t, (np.abs(P[..., 1]) <= half_w) & (P[..., 0] >= wall_x) & (P[..., 0] <= wall_x + depth_box))
        t = (wall_x + gap - o[0]) / D[..., 0]
        cand(t, np.ones(t.shape, bool))
        t = (floor_z - o[2]) / D[..., 2]
        cand(t, np.ones(t.shape, bool))
        if side_faces:
            for ys in (-half_w, half_w):
                t = (ys - o[1]) / D[..., 1]
                P = o + t[..., None] * D
                cand(t, (P[..., 0] >= wall_x) & (P[..., 0] <= wall_x + depth_box) & (P[..., 2] >= floor_z)
                     & (P[..., 2] <= top))
        for lo, hi in poles:
            t1, t2 = (np.asarray(lo, float) - o) / D, (np.asarray(hi, float) - o) / D
            tn, tf = np.minimum(t1, t2).max(-1), np.maximum(t1, t2).min(-1)
            cand(tn, (tf >= tn) & (tn > 0))
    best[~np.isfinite(best)] = np.nan
    return best


# the simulator's g1_v2 head camera stands ~8 cm higher than this fixture's (2026-11-20: the start frame's face fit
# reaches z 1.252 m in the sim, 1.185 m here), so that the start view reaches the wall top there
SIM_HEAD_DZ = 0.08
STEREO_SIGMA = 0.20 / (447.0 * 0.05)   # perception.yaml head_depth disparity_sigma_px / (f x B) of the D435i pair


def head_frame(tb, x, y=0.0, yaw=0.0, noise=0.0, seed=0, dz=0.0, roll_deg=0.0, **scene):
    """Head frame of the synthetic scene from root (x, y, yaw), upright; T_odom_root = the root pose, T_root_cam = FK
    (with the camera raised by dz and rolled about its optical axis by roll_deg: the robot's head as measured, not as
    modelled)."""
    from g1agent.common.transforms import T_from_xy_yaw
    up = tb._lean_levels()[0]
    Tc = tb._cam_T_at(x, y, yaw, up).copy()
    Tc[2, 3] += dz
    a = np.radians(roll_deg)
    Tc[:3, :3] = Tc[:3, :3] @ np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1.0]])
    depth = render_depth(Tc, **scene)
    if noise > 0:
        # stereo depth noise sigma_z = noise x z^2, correlated over the matching window (7 px block: SGBM averages the
        # disparity over it), unit variance per pixel
        import cv2
        g = cv2.blur(np.random.default_rng(seed).normal(0, 1, depth.shape), (7, 7), borderType=cv2.BORDER_REFLECT)
        depth = depth + g / g.std() * noise * depth ** 2
    To = T_from_xy_yaw(x, y, yaw, float(up["pelvis_z"]))
    return {"rgb": np.zeros((480, 848, 3), np.uint8), "depth": depth, "K": K_HEAD.copy(), "T_root_cam": inv_T(To) @ Tc,
            "T_odom_root": To, "t": 1.0, "depth_sigma_coeff": noise, "view": {"pitch": 0.0}}


def squat_frame(tb, x, y=0.0, yaw=0.0, height=0.45, pelvis_pitch_deg=None, pelvis_z=None, noise=0.0, seed=0, **scene):
    """Head frame squatted to a calibrated body level (robot.yaml body calibration: the level's leg / waist joints,
    pelvis height and pitch, unless given as measured), root (pelvis) at (x, y, yaw). T_odom_root carries the pelvis
    tilt (the simulator's odometry does: roll / pitch are observable)."""
    import os
    from g1agent.tools.body import Ry
    cal = json.load(open(os.path.join(config.ROOT, tb.cfg["robot"]["body"]["calibration"])))["levels"]
    lv = min(cal, key=lambda l: abs(l["height_cmd"] - height))
    q = tb._q()
    for n, v in lv["q"].items():
        if n in tb.kin.joint_q_index and not any(k in n for k in ("shoulder", "elbow", "wrist", "hand")):
            q[tb.kin.joint_q_index[n]] = float(v)
    To = T_from_xy_yaw(x, y, yaw, lv["pelvis_z"] if pelvis_z is None else pelvis_z)
    To[:3, :3] = To[:3, :3] @ Ry(math.radians(lv["pelvis_pitch_deg"] if pelvis_pitch_deg is None else pelvis_pitch_deg))
    Trc = tb.kin.pose(q, tb.cfg["robot"]["head_cam_frame"])
    depth = render_depth(To @ Trc, **scene)
    if noise > 0:
        import cv2
        g = cv2.blur(np.random.default_rng(seed).normal(0, 1, depth.shape), (7, 7), borderType=cv2.BORDER_REFLECT)
        depth = depth + g / g.std() * noise * depth ** 2
    return {"rgb": np.zeros((480, 848, 3), np.uint8), "depth": depth, "K": K_HEAD.copy(), "T_root_cam": Trc,
            "T_odom_root": To, "t": 1.0, "depth_sigma_coeff": noise, "view": {"pitch": 0.0}}


def box_faces(wall_x=WALL_X, half_w=0.6, cols=6, rows=5, bw=0.2, bh=0.25, t=1.0, floor_z=0.0):
    """Exact front faces of the wall's boxes (what fit_front_face gives on noise-free depth)."""
    out = {}
    for c in range(1, cols + 1):
        y1 = half_w - (c - 1) * bw
        y0 = y1 - bw
        for r in range(1, rows + 1):
            z0, z1 = floor_z + (r - 1) * bh, floor_z + r * bh
            C = np.array([[wall_x, y1, z1], [wall_x, y0, z1], [wall_x, y0, z0], [wall_x, y1, z0]])
            out[f"C{c}R{r}"] = {"center": C.mean(axis=0), "normal": np.array([-1.0, 0, 0]), "right": np.array([0, -1.0, 0]),
                                "up": np.array([0, 0, 1.0]), "corners": C, "size": [bw, bh], "t": t}
    return out


PLANE = {"normal": np.array([-1.0, 0, 0]), "right": np.array([0, -1.0, 0]), "point": np.array([WALL_X, 0, 0])}


def extent(h, plane=PLANE, cfg=None, floor_z=0.0):
    return plane_fit.wall_extent(h["depth"], h["K"], h["T_root_cam"], h["T_odom_root"], plane,
                                 cfg or config.load("perception")["plane_fit"], h["depth_sigma_coeff"], 7, 0.05, floor_z)


# ------------------------------------------------------------------ 1. set_wall_layout
def test_set_wall_layout_validation():
    tb = kin_toolbox(layout=None)
    for bad in ({**LAYOUT, "extra": 1}, {k: v for k, v in LAYOUT.items() if k != "base"}, {**LAYOUT, "columns": 0},
                {**LAYOUT, "rows": 2.0}, {**LAYOUT, "columns": True}, {**LAYOUT, "box_w": 0}, {**LAYOUT, "box_h": -0.25},
                {**LAYOUT, "box_w": float("nan")}, {**LAYOUT, "base": "table"}, {**LAYOUT, "rows": 3}, "6x5"):
        with pytest.raises(ValueError):
            tb.set_wall_layout(bad)                       # rows 3 < requested row 4 from the bottom
    assert tb.wall_layout is None
    tb.set_wall_layout(LAYOUT)
    assert tb.wall_layout == LAYOUT and tb.ws.wall is None
    t = bare_toolbox(FakeRobot(), variant="g1_v2")
    with pytest.raises(ValueError, match="target spec"):
        t.set_wall_layout(LAYOUT)
    t.set_scan_targets(SPEC_R234)
    with pytest.raises(ValueError, match="obs_pose"):
        t.set_wall_layout(LAYOUT)
    t.set_phase1_view("obs_pose")
    t.multi_face = True
    with pytest.raises(ValueError, match="one wall"):
        t.set_wall_layout(LAYOUT)
    t.multi_face = False
    t.set_scan_targets({"columns_from_right": [7], "order": "rows"})
    with pytest.raises(ValueError, match="columns"):
        t.set_wall_layout(LAYOUT)


# ------------------------------------------------------------------ 2. side edges on synthetic depth
def test_wall_extent_noise_free_edges_within_one_pixel_and_the_floor_strip_removed():
    tb = kin_toolbox()
    h = head_frame(tb, WALL_X - 1.2)
    ext = extent(h)
    px = 1.3 / FX_HEAD                                          # one pixel footprint at the wall (~2 mm)
    assert ext["left"]["s"] == pytest.approx(-0.6, abs=px) and ext["right"]["s"] == pytest.approx(0.6, abs=px)
    rows = np.array([[r[1], r[2]] for r in ext["rows"] if r[3] and r[4]], float)
    assert np.abs(rows[:, 0] + 0.6).max() < px and np.abs(rows[:, 1] - 0.6).max() < px
    assert ext["left"]["n_rows"] > 100 and ext["right"]["n_rows"] > 100
    # the floor strip along the base lies on the plane and runs on beyond the wall's ends: removed at the floor height,
    # the seen extent is the wall's; without the removal it is wider
    assert ext["floor_px_removed"] > 0 and ext["seen"] == pytest.approx([-0.6, 0.6], abs=px)
    raw = extent(h, floor_z=None)
    assert raw["floor_px_removed"] is None and raw["seen"][1] - raw["seen"][0] > 1.2 + 10 * px


def test_wall_extent_with_depth_noise_and_an_invalid_band():
    tb = kin_toolbox()
    h = head_frame(tb, WALL_X - 1.2, noise=0.004, seed=1)
    # a stereo occlusion band beside the left edge: invalid depth (a "behind" observation) on the rays that pass the
    # wall plane within 3 cm beyond the edge
    Tc = h["T_odom_root"] @ h["T_root_cam"]
    vs, us = np.mgrid[0:480, 0:848]
    D = np.stack([(us - K_HEAD[0, 2]) / FX_HEAD, (vs - K_HEAD[1, 2]) / FX_HEAD, np.ones(us.shape)], -1) @ Tc[:3, :3].T
    y = Tc[1, 3] + (WALL_X - Tc[0, 3]) / D[..., 0] * D[..., 1]
    h["depth"][(y > 0.6) & (y < 0.63)] = np.nan
    ext = extent(h)
    # the medians stay within the stereo matching window (7 px, ~1.5 cm at 1.3 m) of the true edges
    win = 7 * 1.3 / FX_HEAD
    assert ext["left"]["s"] == pytest.approx(-0.6, abs=win) and ext["right"]["s"] == pytest.approx(0.6, abs=win)


def test_wall_extent_cut_side_and_occluded_side_are_not_edges():
    tb = kin_toolbox()
    ext = extent(head_frame(tb, WALL_X - 0.7, y=-0.3))       # close and shifted left: the left end runs out of the image
    assert ext["left"] is None and ext["rejected"]["left"]["cut"] > 0 and ext["right"] is not None
    pole = ([WALL_X - 0.3, 0.45, 0.0], [WALL_X - 0.2, 0.75, 2.0])  # a post in front of the left edge, floor to above view
    ext = extent(head_frame(tb, WALL_X - 1.2, poles=[pole]))
    assert ext["left"] is None and ext["rejected"]["left"]["front"] > 0
    assert ext["right"]["s"] == pytest.approx(0.6, abs=0.003)


@pytest.mark.parametrize("roll_deg", [0.0, 0.3, -0.3, 1.0])
def test_wall_extent_no_side_edge_on_the_wall_top_from_the_start_pose(roll_deg):
    """2026-11-20 phase 1 (af2): from the start pose 0.6 m from the wall, with the simulator's head height the top rows
    of the view reach the wall top; rows that end where the (slightly slanted) top edge crosses them, with the box
    tops / room wall behind, were taken for side edges (s -0.33 / -0.42 m, z 1.25 m). The side test (c): a side edge
    runs vertically on the plane. Both sides run out of the image here: no edge; the top's row ends are "boundary"."""
    tb = kin_toolbox()
    for seed in (1, 2):
        ext = extent(head_frame(tb, 0.7, yaw=0.015, dz=SIM_HEAD_DZ, roll_deg=roll_deg, noise=STEREO_SIGMA, seed=seed))
        assert ext["left"] is None and ext["right"] is None, (ext["left"], ext["right"])
        assert ext["rejected"]["left"]["boundary"] + ext["rejected"]["right"]["boundary"] > 0


def test_wall_extent_real_edges_with_the_simulator_head_height():
    """The side test keeps the real side edges: centred at 1.2 m both within a pixel (noise-free) and within the
    stereo matching window (stereo noise); from the af2 obs pose (y -0.656) and the C3R4 retry stance only the right
    edge, from the first C3R4 stance (wall top in view, no end) none (the code before the side test found a right
    edge at z 1.25 m there)."""
    tb = kin_toolbox()
    px, win = 1.3 / FX_HEAD, 7 * 1.3 / FX_HEAD
    ext = extent(head_frame(tb, WALL_X - 1.2, dz=SIM_HEAD_DZ))
    assert ext["left"]["s"] == pytest.approx(-0.6, abs=px) and ext["right"]["s"] == pytest.approx(0.6, abs=px)
    assert ext["left"]["n_rows"] > 100 and ext["right"]["n_rows"] > 100
    ext = extent(head_frame(tb, WALL_X - 1.2, dz=SIM_HEAD_DZ, noise=STEREO_SIGMA, seed=2))
    assert ext["left"]["s"] == pytest.approx(-0.6, abs=win) and ext["right"]["s"] == pytest.approx(0.6, abs=win)
    assert ext["seen"] == pytest.approx([-0.6, 0.6], abs=win)                   # the floor strip removed
    ext = extent(head_frame(tb, WALL_X - 1.2, y=-0.656, dz=SIM_HEAD_DZ, noise=STEREO_SIGMA, seed=3))
    assert ext["left"] is None and ext["right"]["s"] == pytest.approx(0.6, abs=win)
    ext = extent(head_frame(tb, 0.62, y=-0.29, dz=SIM_HEAD_DZ, noise=STEREO_SIGMA, seed=4))
    assert ext["left"] is None and ext["right"]["s"] == pytest.approx(0.6, abs=win)
    ext = extent(head_frame(tb, 0.606, y=-0.075, dz=SIM_HEAD_DZ, noise=STEREO_SIGMA, seed=5))
    assert ext["left"] is None and ext["right"] is None


# ------------------------------------------------------------------ 3. registration
def _register(tb, h, faces=None):
    tb.ws.boxes = {k: tbmod.BoxEstimate(k, [0, 0, 1, 1], 0.9, 1.0, face=f) for k, f in (faces or box_faces()).items()}
    tb.ws.stack = None
    return tb._register_wall(h, list((faces or box_faces()).values()))


def test_registration_both_edges_anchor_residual_and_frame():
    tb = kin_toolbox()
    failed, diag = _register(tb, head_frame(tb, WALL_X - 1.2))
    assert failed is None
    w = tb.ws.wall
    ax = tb._wall_axes()
    assert ax["origin"] == pytest.approx([WALL_X, 0.6, 0.0], abs=0.003)           # bottom-left corner seen from the robot
    assert ax["right"] == pytest.approx([0, -1, 0], abs=1e-9) and ax["normal"] == pytest.approx([-1, 0, 0], abs=1e-9)
    reg = w["registration"]
    assert abs(reg["r"]) < 0.005 and w["delta_lat"] == pytest.approx(abs(reg["r"]) / 2)
    assert reg["z_floor_depth"] == pytest.approx(0.0, abs=0.01)
    # cells come from the current frame: C1 R2 is y 0.4-0.6, z 0.25-0.5
    f = tb._cell_face(1, 2)
    assert f["corners"][:, 1].min() == pytest.approx(0.4, abs=0.003) and f["corners"][:, 2].max() == pytest.approx(0.5, abs=1e-9)


def test_registration_failures_name_the_reason():
    tb = kin_toolbox(layout={**LAYOUT, "columns": 5})                    # 1.0 m layout, 1.2 m wall: |r| = box_w
    failed, _ = _register(tb, head_frame(tb, WALL_X - 1.2))
    assert failed["status"] == "fail" and "does not match the layout" in failed["reason"] and tb.ws.wall is None
    tb = kin_toolbox()
    failed, _ = _register(tb, head_frame(tb, WALL_X - 0.7, y=-0.3))
    assert "the left side edge (cut by the image border) of the wall is not in the head view" in failed["reason"]
    assert "compute_obs_pose and approach again" in failed["reason"]
    pole = ([WALL_X - 0.3, -0.75, 0.0], [WALL_X - 0.2, -0.45, 2.0])
    failed, _ = _register(tb, head_frame(tb, WALL_X - 1.2, poles=[pole]))
    assert "the right side edge (occluded)" in failed["reason"]
    # the floor (and the wall standing on it) seen 0.15 m above the odometry floor (>= box_h / 2): the row indices would
    # be off by one
    faces = box_faces(floor_z=0.15)
    failed, _ = _register(tb, head_frame(tb, WALL_X - 1.2, floor_z=0.15), faces)
    assert "the floor seen is 0.15 m from the odometry floor" in failed["reason"] and tb.ws.wall is None
    # 0.10 m (< box_h / 2) is reported, not decided
    failed, diag = _register(tb, head_frame(tb, WALL_X - 1.2, floor_z=0.10), box_faces(floor_z=0.10))
    assert failed is None and tb.ws.wall["registration"]["z_floor_depth"] == pytest.approx(0.10, abs=0.01)


def test_fit_front_face_unchanged_by_the_support_plane_refactor():
    """fit_front_face on a seeded noisy fixture (box face + floor, the support-plane removal path) gives exactly the
    result of the code before fit_support_plane was factored out (values recorded at 21f856f)."""
    H, W, fx = 120, 160, 120.0
    K = np.array([[fx, 0, W / 2], [0, fx, H / 2], [0, 0, 1.0]])
    p = np.radians(30.0)
    fwd, right = np.array([np.cos(p), 0, -np.sin(p)]), np.array([0, -1.0, 0])
    Rc = np.column_stack([right, np.cross(fwd, right), fwd])
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = Rc, [0, 0, 1.2]
    vs, us = np.mgrid[0:H, 0:W]
    D = np.stack([(us - K[0, 2]) / fx, (vs - K[1, 2]) / fx, np.ones(us.shape)], -1) @ Rc.T
    o = T[:3, 3]
    t_face = (1.3 - o[0]) / D[..., 0]
    P = o + t_face[..., None] * D
    on_face = (t_face > 0) & (np.abs(P[..., 1]) <= 0.3) & (P[..., 2] >= 0) & (P[..., 2] <= 0.5)
    with np.errstate(divide="ignore"):
        t_floor = np.where(D[..., 2] < 0, -o[2] / D[..., 2], np.inf)
    depth = np.where(on_face, t_face, t_floor) * (D @ Rc[:, 2])
    depth = depth + np.random.default_rng(3).normal(0, 1, depth.shape) * 0.004 * depth ** 2
    depth[~np.isfinite(depth)] = np.nan
    r = plane_fit.fit_front_face(depth, K, T, [10, 10, W - 10, H - 10], sigma_coeff=0.004, min_structure_px=7)
    assert r["n_inliers"] == 274 and r["n_valid"] == 329
    np.testing.assert_array_equal(r["center"], [1.2993346794807512, -0.012785683426271887, 0.24188857762121452])
    np.testing.assert_array_equal(r["normal"], [-0.9999848423789512, -0.0011154357634232581, 0.005391735843071923])
    assert r["rms"] == 0.00787320290950743


# ------------------------------------------------------------------ 4. plan
def _planned(tb, order=None):
    h = head_frame(tb, WALL_X - 1.2)
    tb.head = h
    tb.ws.boxes = {k: tbmod.BoxEstimate(k, [0, 0, 1, 1], 0.9, 1.0, face=f) for k, f in box_faces().items()}
    tb._stack_init(list(box_faces().values()))
    tb._stack_add_faces(0, list(box_faces().values()))
    return tb._plan_layout(h, box_faces(), {})


@pytest.mark.parametrize("order", ["rows", "serpentine", "columns"])
def test_plan_one_cell_entry_per_requested_box_in_the_lattice_order(order):
    tb = kin_toolbox(order=order)
    r = _planned(tb)
    assert r["status"] == "ok", r["reason"]
    ws = tb.ws
    cells = [tuple(ws.barcodes[b].wall["cell"]) for b in ws.scan_plan]
    if order == "rows":
        exp = [(c, r_) for r_ in (4, 3, 2) for c in range(1, 7)]
    elif order == "serpentine":
        exp = [(c, 4) for c in range(1, 7)] + [(c, 3) for c in range(6, 0, -1)] + [(c, 2) for c in range(1, 7)]
    else:
        exp = [(c, r_) for c in range(1, 7) for r_ in (4, 3, 2)]
    assert cells == exp and len(ws.scan_plan) == 18
    b0 = ws.barcodes[ws.scan_plan[0]]
    assert b0.kind == "cell" and b0.box_id == f"layout_c{exp[0][0]}_r{exp[0][1]}" and b0.width_m is None
    assert b0.grid_cell == {"column": exp[0][0], "row": exp[0][1]}                # all columns from the left, rows from the bottom
    assert ws.missing_cells() == [] and ws.grid_warnings() == []
    assert all(e["kind"] == "cell" for e in ws.scan_plan_export())
    assert r["data"]["n"] == 18 and "planned from the wall layout" in r["reason"]
    assert abs(r["data"]["registration"]["r"]) < 0.005
    assert "diagnostics" in r and "registration" in r["diagnostics"]
    # a later plan_scans changes nothing and captures nothing
    tb.detect_boxes = lambda *a, **k: (_ for _ in ()).throw(AssertionError("no capture"))
    r2 = tb.plan_scans()
    assert r2["status"] == "ok" and r2["data"] == {"updated": True, "added": []} and len(ws.scan_plan) == 18


def test_export_kind_only_for_layout_entries():
    tb = bare_toolbox(FakeRobot())
    from test_task3_logic import plan_of
    plan_of(tb, 2)
    assert all("kind" not in e for e in tb.ws.scan_plan_export())


def test_obs_pose_region_from_the_side_edges():
    tb = kin_toolbox()
    tb.head = head_frame(tb, WALL_X - 1.2)
    corners, info, ref = tb._spec_view_points(list(box_faces().values()))
    assert info["layout"] and info["lateral"] == pytest.approx([-0.6, 0.6], abs=0.003)
    assert info["z_range"] == [0.0, 0.25] and info["edges_seen"] == {"left": True, "right": True}
    assert ref[1] == pytest.approx(0.0, abs=0.003)
    # from a view that cuts the left end, the remembered left edge still fixes the region
    tb.head = head_frame(tb, WALL_X - 0.7, y=-0.3)
    _, info2, _ = tb._spec_view_points(list(box_faces().values()))
    assert info2["lateral"] == pytest.approx(info["lateral"], abs=0.003)



@pytest.mark.parametrize("seen_w, expect", [(1.0, [-0.7, 0.7]), (1.3, [-0.65, 0.65])])
def test_obs_pose_region_without_an_edge_from_the_seen_extent(seen_w, expect):
    """No side edge seen: every wall placement holding the seen extent, [s_max - W, s_min + W]; a seen extent wider than
    W (no placement holds it) keeps the extent itself, never a region narrower than what was seen."""
    tb = kin_toolbox()
    faces = list(box_faces().values())
    plane = tb._wall_plane(faces)
    tb._wall_extent = lambda h, pl: {"left": None, "right": None, "seen": None}
    tb._layout_edges = {"seen": np.array([plane["point"] + v * plane["right"] for v in (-seen_w / 2, seen_w / 2)])}
    _, info, _ = tb._layout_view_points(faces)
    assert info["lateral"] == pytest.approx(expect, abs=1e-9) and info["edges_seen"] == {"left": False, "right": False}

def _edges_seen_by(tb, faces, frames):
    """_layout_view_points over a sequence of (t, {"left": s or None, "right": s or None}) edge observations."""
    plane = tb._wall_plane(faces)
    out = None
    for t, e in frames:
        tb.head = {"t": t}
        tb._wall_extent = lambda h, pl, e=e: {"left": None if e["left"] is None else {"s": e["left"]},
                                               "right": None if e["right"] is None else {"s": e["right"]},
                                               "seen": [-0.3, 0.3]}
        out = tb._layout_view_points(faces)
    return plane, out


def test_obs_pose_region_edge_memory_newest_consistent_pair_and_contradictions():
    """Edges remembered over phase 1 (2026-11-20 af2: a stale false left edge, -0.70 m from the real right edge, won
    over that right edge three times, the remembered right one was never used): both sides consistent with the layout
    width (halfway rule) -> their middle; contradicting -> the side measured in the newer frame; contradicting within
    one frame -> neither (the seen extent); edges_seen = the edges used."""
    faces = list(box_faces().values())
    W = 1.2
    tb = kin_toolbox()                                   # stale false left edge, then the real right edge
    _, (_, info, _) = _edges_seen_by(tb, faces, [(1.0, {"left": 0.12, "right": None}), (2.0, {"left": None, "right": 0.63})])
    assert info["lateral"] == pytest.approx([0.63 - W, 0.63], abs=1e-9)
    assert info["edges_seen"] == {"left": False, "right": True} and info["edge_conflict"]["dropped"] == ["left"]
    assert "left" not in tb._layout_edges
    tb = kin_toolbox()                                   # the reverse order: the newer left edge wins
    _, (_, info, _) = _edges_seen_by(tb, faces, [(1.0, {"left": None, "right": -0.2}), (2.0, {"left": -0.61, "right": None})])
    assert info["lateral"] == pytest.approx([-0.61, -0.61 + W], abs=1e-9) and info["edges_seen"] == {"left": True, "right": False}
    tb = kin_toolbox()                                   # consistent pair from two frames: their middle
    _, (_, info, _) = _edges_seen_by(tb, faces, [(1.0, {"left": -0.61, "right": None}), (2.0, {"left": None, "right": 0.62})])
    assert info["lateral"] == pytest.approx([0.005 - W / 2, 0.005 + W / 2], abs=1e-9)
    assert info["edges_seen"] == {"left": True, "right": True} and "edge_conflict" not in info
    tb = kin_toolbox()                                   # contradiction within one frame: neither, the seen extent
    _, (_, info, _) = _edges_seen_by(tb, faces, [(1.0, {"left": 0.12, "right": 0.63})])
    assert info["edges_seen"] == {"left": False, "right": False} and info["edge_conflict"]["dropped"] == ["left", "right"]
    assert info["lateral"] == pytest.approx([0.3 - W, -0.3 + W], abs=1e-9)


def test_obs_pose_from_the_start_pose_with_the_wall_top_in_view_then_both_edges_at_the_planned_pose():
    """CPU end to end of the af2 phase-1 failure: from the start pose (simulator head height, stereo noise, a small
    roll) no side edge, the region from the seen extent, the pose centred on the wall (the code before the side test
    found a false edge and planned the pose 0.66 m off to one side); from the planned pose the registration sees both
    edges."""
    tb = kin_toolbox()
    tb.ws.boxes = {k: tbmod.BoxEstimate(k, [0, 0, 1, 1], 0.9, 1.0, face=f) for k, f in box_faces().items()}
    tb.ws.target_box_id = "all"
    tb.ws.stack = None
    tb.head = head_frame(tb, 0.7, yaw=0.015, dz=SIM_HEAD_DZ, roll_deg=-0.3, noise=STEREO_SIGMA, seed=1)
    tb._odom = lambda: tb.head["T_odom_root"]
    r = tb.compute_obs_pose("all")
    assert r["status"] == "ok", r["reason"]
    sv, p = r["data"]["spec_view"], r["data"]["root_target_odom"]
    assert sv["edges_seen"] == {"left": False, "right": False}
    assert abs(p["pos"][1]) < 0.1 and p["d_obs"] == pytest.approx(1.2)
    h = head_frame(tb, p["pos"][0], y=p["pos"][1], yaw=p["yaw"], dz=SIM_HEAD_DZ, noise=STEREO_SIGMA, seed=7)
    tb2 = kin_toolbox()
    failed, _ = _register(tb2, h)
    assert failed is None and abs(tb2.ws.wall["registration"]["r"]) < 0.02


# ------------------------------------------------------------------ 5. hand-camera look pose
def _hand_view(margin):
    import pinocchio as pin
    tb = kin_toolbox()
    q = pin.neutral(tb.kin.model)
    cam = tb.kin.pose(q, tb.cfg["robot"]["hand_camera"]["frame"])
    tip = tb.kin.pose(q, tb.cfg["robot"]["wrist_point_frame"])
    occ = [dict(b, T_cam_box=inv_T(cam) @ tb.kin.pose(q, b["link"])) for b in BOXES]
    return HandView(K_D405, [480, 848], inv_T(tip) @ cam, tb.ik.img_x_in_tip, occ, margin)


def test_search_view_on_the_g1_v2_d405():
    """Offline FK of the design (facesearch/roll.py, no image margin): rolls 90 / 270 tie at s 0.245, roll 0 needs 0.320;
    the bracket box (it contains the camera) is not an occluder."""
    hv = _hand_view(0.0)
    assert sorted(hv.used) == ["scanner_handle", "scanner_head"] and sorted(hv.dropped) == ["bracket", "left_d405"]
    r = search_view(hv, 0.20, 0.25, 0.0, 0.0, 0.005, 0.12, 2.0)
    assert r["s"] == pytest.approx(0.245, abs=0.005) and sorted(c["roll_deg"] for c in r["candidates"]) == [90.0, 270.0]
    assert search_view(hv, 0.20, 0.25, 0.0, 0.0, 0.005, 0.12, 2.0, rolls=(0.0,))["s"] == pytest.approx(0.320, abs=0.005)
    # the chosen pose covers the face at s, and no offset does at s - step
    from g1agent.perception.look_view import face_corners, roll_rotation
    c = r["candidates"][0]

    def cov(s, h, v):
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = roll_rotation(c["roll_deg"], hv.img_x_in_tip), [h, v, s]
        Tpc = T @ hv.T_tip_cam
        Pc = (face_corners(0.10, 0.125) - Tpc[:3, 3]) @ Tpc[:3, :3]
        uv, z = hv.project(Pc)
        return bool(hv.covered(uv[None], z[None])[0])
    assert cov(r["s"], c["h"], c["v"])
    assert not any(cov(r["s"] - 0.005, h, v) for h in np.arange(-0.2, 0.2, 0.005) for v in np.arange(-0.2, 0.2, 0.005))


def test_look_target_with_the_in_view_margin():
    """With the in-view margin of every in-view test (fov_margin_frac 0.05) and e = correction_tol the look stand-off is
    0.30 m (the design's 0.275 m had no margin); both inboard / outboard rolls are alternatives."""
    tb = kin_toolbox()
    _planned(tb)
    tb.ws.wall["delta_lat"] = 0.0
    tb.next_barcode()
    r = tb.compute_wrist_target(offset=0.15)
    t = tb.ws.wrist_target
    assert r["status"] == "ok" and r["data"]["kind"] == "cell" and r["data"]["offset_ignored"] == 0.15
    assert t["purpose"] == "look" and t["roll_signed"] and sorted(a["roll_deg"] for a in t["alternatives"]) == [90.0, 270.0]
    assert t["offset"] == pytest.approx(0.30, abs=1e-9)
    bc = tb._current()
    f = bc.face
    for a in t["alternatives"]:                              # the tip at s in front of the face, offset in its plane
        assert float((a["pos"] - f["center"]) @ f["normal"]) == pytest.approx(t["offset"])
        assert abs(float(a["long_axis"] @ f["normal"])) < 1e-9



def test_hand_view_whole_quad_one_pixel_from_the_scanner_body_is_cut():
    """A detected label box ending within one pixel of an occluded pixel (the scanner head) shows its visible part only:
    not whole; 1.5 px away it is whole; a box lying wholly inside the occluded region is not whole."""
    from g1agent.perception.look_view import grow_quad
    hv = _hand_view(0.05)
    occ_rows = np.nonzero(hv.occluded[:, 380:470].any(axis=1))[0]
    r0 = int(occ_rows.min())                                 # first occluded row of these columns
    assert hv.free[r0 - 3, 380:470].all()
    quad = lambda bottom: np.array([[380.0, r0 - 60.0], [470.0, r0 - 60.0], [470.0, bottom], [380.0, bottom]])
    z = np.full(4, 0.3)
    assert hv.whole(quad(r0 - 1.5), z) and not hv.whole(quad(r0 - 0.5), z)
    assert hv.covered(quad(r0 - 0.5)[None], z[None])[0]      # the plain coverage test would take the cut box as whole
    yy, xx = np.nonzero(hv.occluded & (cv2_erode(hv.occluded, 7)))
    u, v = float(xx[len(xx) // 2]), float(yy[len(yy) // 2])
    assert not hv.whole(np.array([[u - 2, v - 2], [u + 2, v - 2], [u + 2, v + 2], [u - 2, v + 2]]), z)
    g = grow_quad(np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 5.0], [0.0, 5.0]]), 1.0)
    assert g == pytest.approx(np.array([[-1.0, -1.0], [11.0, -1.0], [11.0, 6.0], [-1.0, 6.0]]))


def cv2_erode(mask, k):
    import cv2
    return cv2.erode(mask.astype(np.uint8), np.ones((k, k), np.uint8)) > 0

# ------------------------------------------------------------------ 7. search_cell with a fake hand camera
class Cam:
    """Fake hand camera + detector for search_cell: the hand camera sits exactly where the tip target puts it (FK of the
    look pose, plus an optional error), the detector returns every label of `labels` (wall-frame rectangles [x0, x1,
    z0, z1]) clipped to the crop it is given (a label crossing the crop border is seen only in part, as a real crop).
    scan_here: the result of the scan-pose test at the look stance (_scan_pose_from_here, tested on its own)."""

    def __init__(self, tb, labels, err=None, scan_here=True):
        self.tb, self.labels, self.err = tb, [list(map(float, l)) for l in labels], err
        self.reached, self.crops, self.homes, self.aims = [], [], 0, []
        tb._reach = self.reach
        tb._capture_wrist = self.capture
        tb._labels_on_rectified_face = self.detect
        tb._reregister_at_stance = lambda: {"refit": False, "note": "test"}
        tb._scan_pose_from_here = lambda t: {"ok": scan_here, "status": "ok" if scan_here else "ik"}
        tb.move_arm = self.home

    def home(self, pose="pre", side=None):
        assert pose == "home"
        self.homes += 1
        self.tb.arm_at_home = True
        return tbmod.ok({"pose": "home"})

    def reach(self, t, off, side, aim=None):
        self.reached.append(dict(t))
        self.aims.append(aim)
        self.tb.arm_at_home = False
        return {"ik_fail": False, "history": [{"pos_err_m": 0.0, "angle_err_deg": 0.0}]}

    def capture(self):
        t = self.reached[-1]
        d = np.asarray(t["forward"], float)
        L = np.asarray(t["long_axis"], float)
        c, ex = self.tb.ik.img_x_in_tip, np.array([1.0, 0, 0])
        T = np.eye(4)
        T[:3, :3] = np.column_stack([d, L, np.cross(d, L)]) @ np.column_stack([ex, c, np.cross(ex, c)]).T
        T[:3, 3] = t["pos"]
        if self.err is not None:
            T[:3, 3] = T[:3, 3] + self.err
        return {"rgb": np.zeros((480, 848, 3), np.uint8), "K": K_D405.copy(), "T_root_cam": T @ self.tb._hand_view().T_tip_cam,
                "T_odom_root": np.eye(4), "t": 2.0, "camera": "left_d405"}

    def detect(self, w, face):
        x0, x1, z0, z1 = face["rect_w"]
        self.crops.append(list(face["rect_w"]))
        T = self.tb._wall_T()
        out = []
        for a0, a1, b0, b1 in self.labels:
            c0, c1, d0, d1 = max(a0, x0), min(a1, x1), max(b0, z0), min(b1, z1)
            if c0 >= c1 or d0 >= d1:
                continue
            C = apply(T, np.array([[c0, 0, d1], [c1, 0, d1], [c1, 0, d0], [c0, 0, d0]]))
            rec = {"center": [0.0, 0.0], "corners": [[0.0, 0.0]] * 4, "kind": "dino", "score": 0.5, "refined_by_cv2": False,
                   "symbol_corners": None, "rectified": {"px_per_m": 1500.0, "size": [1, 1]}}
            out.append((rec, {"center": C.mean(axis=0), "corners": C, "bar_dir": np.asarray(T[:3, 2]),
                              "width": float(c1 - c0)}))
        return out


def search_setup(labels, order="rows", delta=0.0, err=None, scan_here=True):
    """A planned layout toolbox at the first box (C1 R4 for rows), its look stance reached, arm home, a fake camera."""
    tb = kin_toolbox(order=order)
    _planned(tb)
    tb.ws.wall["delta_lat"] = delta
    tb.next_barcode()
    assert tb.compute_wrist_target()["status"] == "ok"
    bc = tb._current()
    st = {"barcode_id": bc.id, "purpose": "look", "look_alt": 0, "target_pos": ["look"], "face_dist": 0.5,
          "lateral_offset": 0.0, "level_index": 0, "pos": [0.5, 0.0], "yaw": 0.0}
    tb.ws.stance_target = st
    tb._stance_reached = st
    tb.arm_at_home = True
    return tb, bc, Cam(tb, labels, err, scan_here)


def cell_x(c):          # wall-frame x extent of column c (box_w 0.20)
    return 0.2 * (c - 1), 0.2 * c


def test_search_cell_finds_the_own_label_and_converts_the_entry():
    x0, _ = cell_x(1)
    lab = [x0 + 0.05, x0 + 0.156, 0.957, 1.0]                 # a 0.106 m paper label reaching the box top (R4)
    tb, bc, cam = search_setup([lab])
    r = tb.search_cell()
    assert r["status"] == "ok" and r["data"]["converted"] and r["data"]["added"] == [] and "move_arm('final')" in r["reason"]
    assert bc.kind == "label" and bc.width_source == "hand_label" and bc.source == "hand"
    assert bc.label_width_m == pytest.approx(0.106) and bc.status == "pending"
    wt = tb.ws.wrist_target
    assert wt.get("purpose") is None and wt["barcode_id"] == bc.id
    assert wt["offset"] >= tb.cfg["task3"]["offset_m"] - 1e-9            # hand label width: an upper bound, >= offset_m
    assert bc.wall["center"] == pytest.approx([x0 + 0.103, 0.0, 0.9785], abs=1e-9)   # stored in the wall frame
    assert len(cam.reached) == 1 and cam.homes == 0
    # the label path now runs: the refusal is gone, the cell entry's look stance is kept
    assert tb._layout_refusal(bc) is None


def test_search_cell_ignores_a_whole_neighbour_label_and_records_no_label_after_a_second_search():
    x0, x1 = cell_x(1)
    lab = [x1 + 0.01, x1 + 0.05, 0.96, 0.99]                  # whole inside the grown crop, centre in the next box
    tb, bc, cam = search_setup([lab], delta=0.05)
    r = tb.search_cell()
    assert r["status"] == "ok" and r["data"]["barcode_status"] == "no_label" and bc.status == "no_label"
    assert cam.homes == 1 and len(cam.reached) == 2                       # re-register + search again (arm home first)
    assert "no label in this box" in r["reason"] and "grown by" in bc.fail_reason
    assert tb.scan()["status"] == "fail" and "no_label" in tb.scan()["reason"]


def test_search_cell_two_labels_in_one_box_adds_the_second_as_the_next_entry():
    x0, _ = cell_x(1)
    a, b = [x0 + 0.02, x0 + 0.08, 0.95, 0.99], [x0 + 0.12, x0 + 0.18, 0.80, 0.84]
    tb, bc, cam = search_setup([b, a])
    plan0 = list(tb.ws.scan_plan)
    r = tb.search_cell()
    added = r["data"]["added"]
    assert r["status"] == "ok" and len(added) == 1
    nb = tb.ws.barcodes[added[0]]
    assert tb.ws.scan_plan == [plan0[0], added[0]] + plan0[1:]
    assert [tb.ws.barcodes[x].order_index for x in tb.ws.scan_plan] == list(range(len(tb.ws.scan_plan)))
    assert bc.center[2] > nb.center[2]                                    # rule rows: the top label first
    assert nb.kind == "label" and nb.grid_cell == bc.grid_cell and nb.status == "pending" and nb.width_source == "hand_label"


def test_search_cell_partial_when_the_face_was_not_covered():
    x0, _ = cell_x(1)
    tb, bc, cam = search_setup([[x0 + 0.05, x0 + 0.15, 0.96, 0.99]], err=np.array([0.0, 0.10, 0.0]))
    r = tb.search_cell()
    assert r["status"] == "partial" and "not fully in the hand-camera view" in r["reason"] and bc.status == "pending"
    assert bc.kind == "cell" and len(tb._failed_stances) == 1



@pytest.mark.parametrize("err", [[0.0, 0.009, 0.0], [0.0, -0.009, 0.0], [0.009, 0.0, 0.0], [0.0, 0.0, -0.009],
                                 [0.005, 0.005, 0.005], [-0.005, -0.005, -0.005]])
def test_search_cell_look_pose_keeps_the_face_covered_under_a_placement_error_below_e(err):
    """The look pose is planned for the face grown by delta + e, the face grown by delta is verified: a tip placement
    error below e = correction_tol (what _reach accepts) keeps the face covered (verifying the planned rectangle itself
    left mm of slack: lateral errors of 5-9 mm failed it)."""
    x0, _ = cell_x(1)
    tb, bc, cam = search_setup([[x0 + 0.05, x0 + 0.156, 0.957, 1.0]], err=np.array(err))
    assert np.linalg.norm(err) < tb.cfg["robot"]["arm_motion"]["correction_tol"]
    r = tb.search_cell()
    assert r["status"] == "ok" and r["data"]["converted"] and bc.kind == "label"


def test_search_cell_refuses_a_box_already_recorded_no_label():
    x0, x1 = cell_x(1)
    tb, bc, cam = search_setup([[x1 + 0.01, x1 + 0.05, 0.96, 0.99]], delta=0.05)
    assert tb.search_cell()["data"]["barcode_status"] == "no_label"
    tb.arm_at_home = True
    cam.labels = [[x0 + 0.05, x0 + 0.15, 0.96, 0.99]]                   # a detection now would convert a no_label box
    r = tb.search_cell()
    assert r["status"] == "fail" and "already recorded no_label" in r["reason"] and "next_barcode" in r["reason"]
    assert bc.kind == "cell" and bc.status == "no_label"

def test_search_cell_cut_label_gets_one_centred_view_and_stays_unresolved_when_larger_than_the_view():
    x0, x1 = cell_x(1)
    big = [x0 - 0.5, x1 + 0.5, 0.80, 0.99]                     # wider than any view
    tb, bc, cam = search_setup([big])
    r = tb.search_cell()
    assert r["status"] == "partial" and "not seen whole" in r["reason"]
    assert len(cam.reached) == 2 and bc.kind == "cell" and bc.status == "pending"   # exactly one extra view


def test_search_cell_never_assigns_a_neighbour_sliver():
    """Design 6.5: a neighbour label 0.10 m wide centred 3 cm outside the right seam; the crop reaches e = 1 cm past the
    seam, the visible part's centre lies inside the cell. It is cut, its centred view sees it whole, centre outside."""
    x0, x1 = cell_x(1)
    tb, bc, cam = search_setup([[x1 + 0.03 - 0.05, x1 + 0.03 + 0.05, 0.96, 0.99]])
    r = tb.search_cell()
    assert r["data"].get("barcode_status") == "no_label" and bc.kind == "cell"
    assert len(cam.reached) == 4                                          # 2 searches x (look + one centred view)


def test_search_cell_preconditions():
    tb, bc, cam = search_setup([])
    tb.arm_at_home = False
    assert "arm must be at home" in tb.search_cell()["reason"]
    tb.arm_at_home = True
    tb._stance_reached = None
    assert "not at the planned stance" in tb.search_cell()["reason"]
    tb.ws.stance_target = dict(tb.ws.stance_target, purpose=None)
    assert "no hand-camera view stance" in tb.search_cell()["reason"]
    other = bare_toolbox(FakeRobot())
    assert "wall layout" in other.search_cell()["reason"]


# ------------------------------------------------------------------ 8. stance re-registration
@pytest.mark.parametrize("pose", [(0.65, -0.89, 0.0, 0.45), (0.85, -1.0, 0.3, 0.45), (0.75, 1.0, -0.3, 0.70),
                                  (0.95, 1.0, -0.3, 0.30)])
def test_wall_extent_edge_seen_beyond_the_wall_end_with_and_without_the_side_face(pose):
    """From beyond a wall end the stack's side face is in view: the run of "on" pixels goes on over it down to d = -tol,
    and the boundary ray met d = 0 outward of the corner by tol x (s_cam - s_edge) / d_cam (5.5-28 mm on these views,
    2026-11-20 review). The ray is now intersected with d = -tol when the run ends on the side face, with d = 0 when no
    side face is there (the step at the corner): within 1 mm of the true end both ways, and with stereo noise."""
    x, y, yaw, lvl = pose
    tb = kin_toolbox()
    sd, true = ("left", -0.6) if y > 0 else ("right", 0.6)
    for side_faces in (False, True):
        e = extent(squat_frame(tb, x, y=y, yaw=yaw, height=lvl, side_faces=side_faces))[sd]
        assert e is not None and e["s"] == pytest.approx(true, abs=0.001), (side_faces, e)
        for seed in (0, 1):
            e = extent(squat_frame(tb, x, y=y, yaw=yaw, height=lvl, side_faces=side_faces, noise=STEREO_SIGMA,
                                   seed=seed))[sd]
            assert e is not None and e["s"] == pytest.approx(true, abs=0.002), (side_faces, seed, e)


def test_wall_extent_the_wall_base_is_no_side_edge_under_stereo_noise():
    """The floor band at the wall's base lies within tol of the plane: labelling it by the sign of d labelled it by the
    noise, and a squatted view with the right end out of view (s_cam -0.98) took the base for a 3-row right edge at
    s -0.11 (seed 0, before the "floor" label). The run ending at the floor is the base: no edge."""
    tb = kin_toolbox()
    for seed in range(4):
        ext = extent(squat_frame(tb, 0.65, y=1.0, yaw=-0.3, height=0.45, noise=STEREO_SIGMA, seed=seed))
        assert ext["right"] is None and ext["left"]["s"] == pytest.approx(-0.6, abs=0.002), (seed, ext["right"])
        assert ext["rejected"]["right"]["floor"] + ext["rejected"]["left"]["floor"] > 0


def _stored_frame(tb, dyaw_deg=0.0, dlat=0.0, root=(0.75, 0.0)):
    """The true wall frame, then perturbed as odometry would: rotated about the root by dyaw, shifted along the wall."""
    tb.ws.wall = {"T_odom_wall": tb._wall_frame(np.array([WALL_X, 0.6, 0.0]), np.array([0, -1.0, 0]), np.array([-1.0, 0, 0])),
                  "W": 1.2, "H": 1.25, "columns": 6, "rows": 5, "box_w": 0.2, "box_h": 0.25, "delta_lat": 0.0, "t": 1.0,
                  "anchor_root_xy": np.array(root, float), "registration": {}, "history": []}
    a = math.radians(dyaw_deg)
    Rz = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1.0]])
    T = tb._wall_T()
    o = T[:3, 3].copy()
    o[:2] = np.asarray(root) + Rz[:2, :2] @ (o[:2] - np.asarray(root))
    T2 = tb._wall_frame(o + dlat * (Rz @ T[:3, 0]), Rz @ T[:3, 0], Rz @ -T[:3, 1])
    tb.ws.wall["T_odom_wall"] = T2


def test_reregistration_corrects_yaw_about_the_root_and_keeps_the_lateral_coordinate():
    tb = kin_toolbox()
    _stored_frame(tb, dyaw_deg=3.0, root=(0.75, 0.0))
    h = head_frame(tb, 0.75, half_w=3.0)                               # a wide wall: no side edge in view
    tb._capture_head = lambda: h
    root = h["T_odom_root"][:3, 3]
    lat0 = float((root - tb._wall_axes()["origin"]) @ tb._wall_axes()["right"])
    rec = tb._reregister_at_stance()
    ax = tb._wall_axes()
    assert rec["refit"] and rec["ds_m"] is None and rec["dpsi_deg"] == pytest.approx(-3.0, abs=0.2)
    assert ax["normal"] == pytest.approx([-1, 0, 0], abs=0.004)
    assert float(ax["origin"] @ -ax["normal"]) == pytest.approx(WALL_X, abs=0.01)    # the plane on the refit
    assert float((root - ax["origin"]) @ ax["right"]) == pytest.approx(lat0, abs=1e-9)
    assert tb.ws.wall["history"] == [rec]


def test_reregistration_reanchors_on_a_side_edge_in_view_and_a_failed_refit_changes_nothing():
    tb = kin_toolbox()
    _stored_frame(tb, dlat=0.03, root=(0.75, 0.45))
    h = head_frame(tb, 0.75, y=0.45)                                    # at the C1 end: the left edge in view
    tb._capture_head = lambda: h
    rec = tb._reregister_at_stance()
    assert "left" in rec["anchored"] and rec["ds_m"] == pytest.approx(-0.03, abs=0.004)
    assert tb._wall_axes()["origin"] == pytest.approx([WALL_X, 0.6, 0.0], abs=0.004)
    T = tb._wall_T().copy()
    h2 = dict(h, depth=np.full_like(h["depth"], np.nan))
    tb._capture_head = lambda: h2
    rec2 = tb._reregister_at_stance()
    assert not rec2["refit"] and "refit failed" in rec2["note"] and np.array_equal(tb._wall_T(), T)


def test_layout_entries_follow_the_reregistered_frame():
    tb = kin_toolbox()
    _planned(tb)
    bc = tb.ws.barcodes[tb.ws.scan_plan[0]]
    w0 = np.array(bc.wall["center"])
    T = tb._wall_T()
    T[:3, 3] += [0.0, 0.02, 0.0]
    tb.ws.wall["T_odom_wall"] = T
    tb._sync_layout_all()
    assert np.allclose(apply(inv_T(T), bc.center), w0) and np.allclose(bc.face["center"], bc.center)


def test_reregistration_squatted_beyond_the_wall_end_fits_the_front_not_the_side_face():
    """2026-11-20 rule run i=151 (bc17 = C6R2): pelvis 0.65 m in front of the face, 0.29 m beyond its right end,
    squatted to base_height 0.45 (pelvis z 0.447, pitch 21.45 deg measured). Before: dpsi 90.8 deg, dd 0.34 m, ds 2.1 m
    (reproduced here: 90.0 / 0.36 / 2.12): the bbox of the projected wall rectangle held the stack's right side face
    and the fit ran in the pitched root frame (the front's normal 21 deg off horizontal, beyond the 20 deg test). Now:
    the front plane from the wall's predicted pixels, in odom; the right edge in view re-anchors; every record logs
    the per-side edges."""
    for noise in (0.0, STEREO_SIGMA):
        tb = kin_toolbox()
        _stored_frame(tb, root=(0.65, -0.89))
        h = squat_frame(tb, 0.65, y=-0.89, height=0.45, pelvis_pitch_deg=21.45, pelvis_z=0.447, noise=noise, seed=3,
                        side_faces=True)
        tb._capture_head = lambda: h
        rec = tb._reregister_at_stance()
        assert rec["refit"] and abs(rec["dpsi_deg"]) < 0.5 and abs(rec["dd_m"]) < 0.01, rec
        assert rec["anchored"] == ["right"] and abs(rec["ds_m"]) < 0.02
        assert tb._wall_axes()["origin"] == pytest.approx([WALL_X, 0.6, 0.0], abs=0.02)
        e = rec["edges"]["right"]
        assert set(e) == {"s", "mad", "n_rows", "z_range"} and e["n_rows"] > 100 and rec["edges"]["left"] is None
        assert rec["edges_on"] == "refit plane"
    # in front of the wall at the deepest squat levels (pelvis pitched 25-26 deg): a 1 deg heading drift corrected
    for height in (0.3, 0.25):
        tb = kin_toolbox()
        _stored_frame(tb, dyaw_deg=1.0, root=(0.65, 0.0))
        h = squat_frame(tb, 0.65, height=height, noise=STEREO_SIGMA, seed=3, side_faces=True)
        tb._capture_head = lambda: h
        rec = tb._reregister_at_stance()
        assert rec["refit"] and rec["dpsi_deg"] == pytest.approx(-1.0, abs=0.2), rec


@pytest.mark.parametrize("dlat, dyaw", [(0.01, 0.0), (0.03, 0.0), (0.06, 0.0), (0.04, -2.0)])
def test_reregistration_beyond_the_wall_end_keeps_the_side_face_out_of_the_predicted_region(dlat, dyaw):
    """The i=151 view with the stored right end beyond the true one (dlat; odometry error below box_w / 2): rays
    through the strip in between reach the side face, and with stereo noise those pixels tilted the refit 0.5-0.8 deg
    and moved the far end 1-1.6 cm (2026-11-20 review). The end the camera is beyond is pulled in by box_w / 2: no
    side-face pixel in the region, the refit within 0.4 deg and 6 mm, the lateral re-anchor within 2 mm."""
    tb = kin_toolbox()
    _stored_frame(tb, dlat=dlat, dyaw_deg=dyaw, root=(0.65, -0.89))
    h = squat_frame(tb, 0.65, y=-0.89, height=0.45, pelvis_pitch_deg=21.45, pelvis_z=0.447, noise=STEREO_SIGMA, seed=3,
                    side_faces=True)
    region = tb._wall_region(h)
    Tc = h["T_odom_root"] @ h["T_root_cam"]
    clean = squat_frame(tb, 0.65, y=-0.89, height=0.45, pelvis_pitch_deg=21.45, pelvis_z=0.447, side_faces=True)["depth"]
    vs, us = np.mgrid[0:480, 0:848]
    P = (np.stack([(us - K_HEAD[0, 2]) / FX_HEAD, (vs - K_HEAD[1, 2]) / FX_HEAD, np.ones(us.shape)], -1)
         * clean[..., None]) @ Tc[:3, :3].T + Tc[:3, 3]
    side = np.isfinite(clean) & (np.abs(np.abs(P[..., 1]) - 0.6) < 1e-3) & (P[..., 0] > WALL_X + 1e-3)
    assert region.sum() > 1000 and not (region & side).any()
    tb._capture_head = lambda: h
    rec = tb._reregister_at_stance()
    ax = tb._wall_axes()
    assert rec["refit"] and rec["anchored"] == ["right"], rec
    assert abs(math.degrees(math.atan2(ax["normal"][1], -ax["normal"][0]))) < 0.4
    far = ax["origin"] + 1.2 * ax["right"]                   # the far (left) end, 1.2 m along the wall
    assert abs(float(ax["origin"][0]) - WALL_X) < 0.006 and abs(float(far[0]) - WALL_X) < 0.006
    assert float(ax["origin"][1]) == pytest.approx(0.6, abs=0.002)


def test_head_face_fit_in_odom_at_a_pitched_squat():
    """Every head face fit (detect_boxes' view count, fit_front_face, refine_with_wrist's head refit, the stance
    re-registration) runs in odom (_fit_face_odom): squatted to base_height 0.45 the pelvis is pitched 21 deg, beyond
    max_normal_tilt_deg 20, and in the root frame the wall's own normal failed the near-vertical test."""
    tb = kin_toolbox()
    for noise in (0.0, STEREO_SIGMA):
        h = squat_frame(tb, 0.65, height=0.45, pelvis_pitch_deg=21.45, pelvis_z=0.447, noise=noise, seed=3,
                        side_faces=True)
        bbox = [300, 100, 548, 380]
        root = plane_fit.fit_front_face(h["depth"], h["K"], h["T_root_cam"], bbox, sigma_coeff=h["depth_sigma_coeff"],
                                        min_structure_px=tb._min_structure_px())
        if noise == 0.0:      # with noise a 3-point hypothesis the noise tilts below 20 deg can pass (it did, 4 of 4 seeds)
            assert root["status"] != "ok"
        r = tb._fit_face_odom(h, bbox)
        assert r["status"] == "ok" and r["normal"] == pytest.approx([-1.0, 0.0, 0.0], abs=0.01)
        assert float(r["center"][0]) == pytest.approx(WALL_X, abs=0.005)


def test_reregistration_rejects_a_correction_of_half_a_box_and_skips_a_wall_out_of_view():
    """Same-wall check, the registration's halfway rule: a plane correction moving the wall by >= box_w / 2 keeps the
    previous frame; a re-anchor moving it by >= box_w / 2 is not applied (the plane correction is); the registered
    wall out of the head view: no refit."""
    tb = kin_toolbox()
    _stored_frame(tb, root=(0.75, 0.0))
    T = tb._wall_T()
    T[:3, 3] += [0.15, 0.0, 0.0]                         # stored 0.15 m deeper than the wall seen (>= box_w / 2)
    tb.ws.wall["T_odom_wall"] = T
    h = head_frame(tb, 0.75, half_w=3.0)
    tb._capture_head = lambda: h
    rec = tb._reregister_at_stance()
    assert not rec["refit"] and "refit rejected" in rec["note"] and rec["shift_m"] == pytest.approx(0.15, abs=0.01)
    assert rec["edges_on"] == "stored plane" and set(rec["edges"]) == {"left", "right", "rejected"}
    assert np.array_equal(tb._wall_T(), T) and tb.ws.wall["history"] == [rec]
    T[:3, 3] += [-0.08, 0.0, 0.0]                        # 0.07 m (< box_w / 2): a correction, applied
    tb.ws.wall["T_odom_wall"] = T.copy()
    rec = tb._reregister_at_stance()
    assert rec["refit"] and rec["dd_m"] == pytest.approx(0.07, abs=0.01)        # toward the robot (the normal)
    tb = kin_toolbox()                                   # lateral: stored 0.12 m off, the left edge in view
    _stored_frame(tb, dlat=0.12, root=(0.75, 0.45))
    h = head_frame(tb, 0.75, y=0.45)
    tb._capture_head = lambda: h
    rec = tb._reregister_at_stance()
    assert rec["refit"] and rec["anchored"] == [] and rec["ds_m"] is None and "re-anchor rejected" in rec["note"]
    assert rec["ds_rejected_m"] == pytest.approx(-0.12, abs=0.004) and rec["edges"]["left"] is not None
    assert tb._wall_axes()["origin"] == pytest.approx([WALL_X, 0.6 - 0.12, 0.0], abs=0.004)
    tb = kin_toolbox()                                   # facing away from the wall
    _stored_frame(tb, root=(0.75, 0.0))
    T = tb._wall_T().copy()
    h = head_frame(tb, 0.75, yaw=math.pi)
    tb._capture_head = lambda: h
    rec = tb._reregister_at_stance()
    assert not rec["refit"] and rec["region_px"] == 0 and "not in the head view" in rec["note"]
    assert np.array_equal(tb._wall_T(), T)


# ------------------------------------------------------------------ 9. refusals and next_barcode
def test_label_path_tools_refuse_a_cell_entry_and_next_barcode_has_no_plan_update():
    tb = kin_toolbox()
    _planned(tb)
    tb._plan_update_from_here = lambda: (_ for _ in ()).throw(AssertionError("no plan update in layout mode"))
    r = tb.next_barcode()
    assert r["data"]["kind"] == "cell" and r["data"]["standoff_m"] is None and r["data"]["width_m"] is None
    for res in (tb.move_arm("pre"), tb.move_arm("final"), tb.refine_with_wrist(), tb.stereo_check_wrist(),
                tb.check_wrist_placement(), tb.scan()):
        assert res["status"] == "fail" and "call search_cell" in res["reason"]
    assert tb._current().id not in tb._stance_attempted                  # a refused scan is no attempt
    tb._stance_attempted.add(tb._current().id)
    assert tb.next_barcode()["data"]["kind"] == "cell"


# ------------------------------------------------------------------ 10. Harness
def _layout_harness(tb):
    c = config.load_all()
    return Harness(tb, 3, c["task3"]["budget"], FakeLog(), c["robot"]["limits"])


def test_harness_search_cell_only_in_layout_episodes_budget_and_no_label_stage():
    tb = bare_toolbox(FakeRobot())
    h = _layout_harness(tb)
    assert "search_cell" not in h.tools and h.tools == TASK_TOOLS[3]
    r = h.call("search_cell", {})
    assert r["status"] == "fail" and r["reason"] == "tool 'search_cell' is not part of task 3" and r.get("rejected")
    tb = kin_toolbox()
    _planned(tb)
    h = _layout_harness(tb)
    assert h.tools == TASK_TOOLS[3] + LAYOUT_ONLY
    per = config.load_all()["task3"]["budget"]["per_barcode"]
    b0 = dict(h.budget)
    tb.search_cell = lambda barcode_id=None: tbmod.ok({"barcode_id": "bc0", "converted": True, "added": ["bc18"]})
    h.call("search_cell", {})
    assert h.budget["max_tool_calls"] == b0["max_tool_calls"] + per["max_tool_calls"]
    assert h.budget["max_wall_s"] == b0["max_wall_s"] + per["max_wall_s"]

    def no_label(barcode_id=None):
        tb.ws.barcodes["bc1"].status = "no_label"
        return tbmod.ok({"barcode_id": "bc1", "barcode_status": "no_label"})
    tb.search_cell = no_label
    h.call("search_cell", {})
    assert ("barcode_end", {"id": "bc1", "status": "no_label", "content": None}) in h.log.stages


def test_harness_success_counts_no_label_as_handled():
    tb = kin_toolbox()
    _planned(tb)
    h = _layout_harness(tb)
    sts = ["scanned", "no_label", "unreadable"] + ["no_label"] * 15
    for b, s in zip(tb.ws.scan_plan, sts):
        tb.ws.barcodes[b].status = s
    r = h.call("finish", {"task": 3, "success": True, "summary": "x"})
    assert r["status"] == "ok" and h.finished["success"] is True
    tb.ws.barcodes[tb.ws.scan_plan[0]].status = "no_label"           # nothing read at all: no success
    h2 = _layout_harness(tb)
    h2.call("finish", {"task": 3, "success": False, "summary": "x"})
    assert h2.finished["success"] is False


# ------------------------------------------------------------------ 11. rule path
class LayoutRuleTB(ScriptedTB):
    """ScriptedTB with a wall layout: plan_scans plans cell entries; search_cell per barcode from `outcomes`
    ("label" | "label_far" (found, its scan pose not reachable from the look stance) | "no_label" | "partial" | "ik"
    (the look pose not reachable from the stance), or a list of them consumed one per search); move_arm('final') fails
    as IK the first `final_ik_fails` times."""

    def __init__(self, robot, outcomes, final_ik_fails=0):
        super().__init__(robot, plan_n=len(outcomes))
        self.wall_layout = dict(LAYOUT)
        self.outcomes = dict(outcomes)
        self.final_ik_fails = final_ik_fails

    def plan_scans(self):
        if self.ws.scan_plan:
            return tbmod.ok({"updated": True, "added": []})
        from test_task3_logic import plan_of
        plan_of(self, self.plan_n)
        for b in self.ws.scan_plan:
            self.ws.barcodes[b].kind = "cell"
        return tbmod.ok({"n": self.plan_n})

    def compute_wrist_target(self, barcode_id=None, offset=None, refined=False):
        self.calls.append(("compute_wrist_target", barcode_id))
        return tbmod.ok()

    def plan_stance(self, which="wrist_target"):
        bc = self._current()
        self.calls.append(("plan_stance", bc.kind))
        self.ws.stance_target = {"barcode_id": bc.id, **({"purpose": "look"} if bc.kind == "cell" else {})}
        return tbmod.ok()

    def search_cell(self, barcode_id=None):
        bc = self._current()
        self.calls.append(("search_cell", bc.id))
        o = self.outcomes[bc.id]
        if isinstance(o, list):
            o = o.pop(0)
        if o in ("label", "label_far"):
            bc.kind = "label"
            self.arm_at_home = False                      # the search leaves the arm at the look pose
            if o == "label_far":
                self.ws.stance_target = None
            return tbmod.ok({"barcode_id": bc.id, "converted": True, "added": [], "scan_from_this_stance": o == "label"})
        if o == "no_label":
            bc.status = "no_label"
            return tbmod.ok({"barcode_id": bc.id, "barcode_status": "no_label"})
        if o == "ik":
            return tbmod.fail("hand-camera view not reachable from this stance: move_arm('home'), compute_wrist_target, "
                              "plan_stance, approach", {"barcode_id": bc.id, "ik": "IK: target not reachable"})
        return tbmod.partial("the box face was not fully in the hand-camera view", {"barcode_id": bc.id})

    def move_arm(self, pose="pre", side=None):
        if pose == "final" and self.final_ik_fails > 0:
            self.final_ik_fails -= 1
            self.calls.append(("move_arm", "final_ik_fail"))
            return tbmod.fail("IK: target not reachable (residual 0.1 m / 3 deg)")
        return super().move_arm(pose, side)


def test_rule_layout_flow_search_then_final_and_no_label_without_scan():
    robot = FakeRobot({"bc0": [read("A")]})
    tb = LayoutRuleTB(robot, {"bc0": "label", "bc1": "no_label"})
    h = _layout_harness(tb)
    rule.Task3(h, config.load_all()["task3"], "all").run()
    tools = [r["tool"] for r in h.log.tools]
    assert tools.count("search_cell") == 2 and tools.count("scan") == 1 and tools.count("plan_scans") == 1
    i = tools.index("search_cell")
    assert tools[i + 1:i + 3] == ["move_arm", "refine_with_wrist"]          # final straight after the search, no pre
    assert ("move_arm", "pre") not in tb.calls and "stereo_check_wrist" not in tools
    assert h.finished["success"] is True and tb.ws.barcodes["bc1"].status == "no_label"


def test_rule_layout_replans_once_after_a_final_ik_failure_at_the_look_stance_and_after_a_partial_search():
    robot = FakeRobot({"bc0": [read("A")]})
    tb = LayoutRuleTB(robot, {"bc0": "label"}, final_ik_fails=1)
    h = _layout_harness(tb)
    rule.Task3(h, config.load_all()["task3"], "all").run()
    assert [c for c in tb.calls if c[0] == "plan_stance"] == [("plan_stance", "cell"), ("plan_stance", "label")]
    assert ("move_arm", "pre") in tb.calls and h.finished["success"] is True
    tb = LayoutRuleTB(FakeRobot(), {"bc0": "partial"})
    h = _layout_harness(tb)
    rule.Task3(h, config.load_all()["task3"], "all").run()
    assert [c[0] for c in tb.calls].count("search_cell") == 2 and h.finished["success"] is False



def test_rule_layout_goes_to_the_refined_final_pose_after_the_search():
    """Design 6.4: final -> refine_with_wrist -> final -> check -> scan. After a search no refined target exists yet, so
    the first refine is compared with the label target the final pose went to; it moved by more than place.tol_m, so the
    arm goes to the final pose again."""
    class TB(LayoutRuleTB):
        def search_cell(self, barcode_id=None):
            r = super().search_cell(barcode_id)
            bc = self._current()
            self.ws.wrist_target = {"barcode_id": bc.id, "pos": np.zeros(3), "offset": 0.265}
            self.ws.wrist_target_refined = None
            return r

        def refine_with_wrist(self, barcode_id=None):
            bc = self._current()
            self.ws.wrist_target_refined = {"barcode_id": bc.id, "pos": np.array([0.05, 0.0, 0.0]), "offset": 0.215}
            return tbmod.ok()
    tb = TB(FakeRobot({"bc0": [read("A")]}), {"bc0": "label"})
    h = _layout_harness(tb)
    rule.Task3(h, config.load_all()["task3"], "all").run()
    tools = [r["tool"] for r in h.log.tools]
    i = tools.index("search_cell")
    assert tools[i + 1:i + 7] == ["move_arm", "refine_with_wrist", "move_arm", "refine_with_wrist",
                                  "check_wrist_placement", "scan"]
    assert [c for c in tb.calls if c[0] == "move_arm" and c[1] == "final"] == [("move_arm", "final")] * 2

# ------------------------------------------------------------------ 13. prompts
# r234 scenario (no wall layout) texts as they were before the wall-layout change, computed at 21f856f with the
# goldens of the scratchpad goldens_pre_layout (system.txt sha256 identical)
R234_SYSTEM_SHA256 = "99ac25ead188a38b159421c23adf727df6f724eb5fe0d5e03b523043bf5371f8"
R234_TOOLS_SHA256 = "dd371bafaf265ee0c81cdfbd1e58cdfed825efb1d1afd4d315c2d4b1c0753660"
R234_START = ("Start task 3: scan only the labels of the boxes in all columns and rows 2, 3, 4 from the bottom of the box "
              "wall, row by row from the top, left to right within a row (the tools fix this order). Walk to the "
              "observation pose the tools compute (upright head views only; compute_obs_pose, then approach, which may "
              "walk back): from there the head sees the wall edges the task counts from and the requested rows. Then "
              "plan_scans counts the rows and columns from there. This task covers the side of the boxes facing you only.")


def _texts(monkeypatch, layout, order="rows"):
    from g1agent.orchestrators.agent_full import run_agent_full
    tb = ScriptedTB(FakeRobot(), plan_n=1)
    tb.cfg = config.apply_head_camera(config.apply_robot_variant(config.load_all(), "g1_v2"), "d435i")
    tb.set_scan_targets(dict(SPEC_R234, order=order))
    tb.set_head_views("upright")
    tb.set_phase1_view("obs_pose")
    if layout is not None:
        tb.set_wall_layout(layout)
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
    return msgs[0]["content"], start, h.log.llms[0]["prompt_version"], json.dumps(seen[0])


def test_r234_prompt_and_tools_are_byte_identical_without_a_layout(monkeypatch):
    system, start, version, tools = _texts(monkeypatch, None)
    assert hashlib.sha256(system.encode()).hexdigest() == R234_SYSTEM_SHA256
    assert hashlib.sha256(tools.encode()).hexdigest() == R234_TOOLS_SHA256
    assert start.startswith(R234_START) and "(wall layout)" not in version and "search_cell" not in tools + system


@pytest.mark.parametrize("order", ["rows", "serpentine"])
def test_layout_prompt_tools_and_version(monkeypatch, order):
    from g1agent.orchestrators import agent_full as af
    system, start, version, tools = _texts(monkeypatch, LAYOUT, order)
    names = [t["function"]["name"] for t in json.loads(tools)]
    assert names == [n for n in openai_tools_names() if n in TASK_TOOLS[3] + LAYOUT_ONLY] and "search_cell" in names
    assert version.endswith("(wall layout)")
    assert af.PHASE2_LOOP_LAYOUT in system and "plan_scans (adds labels" not in system and "{" not in system
    assert "search_cell" in system and "no_label" in system and "every requested box handled" in system
    assert "The wall layout is given: 6 columns x 5 rows of boxes, each 0.2 m wide x 0.25 m tall" in start
    assert "both side edges of the wall" in start and "lean" not in system.replace("clean", "")
    desc = {t["function"]["name"]: t["function"]["description"] for t in json.loads(tools)}
    assert desc["plan_scans"].startswith("Task 3, after reaching the observation pose: register the wall")
    for n, note in af.LAYOUT_TOOL_NOTES.items():
        assert desc[n].endswith(note)
    if order == "serpentine":
        assert "the top row left to right, the next right to left" in start and "alternating" in system


def openai_tools_names():
    return [t["function"]["name"] for t in openai_tools(None)]


def test_task_tools_helper():
    tb = bare_toolbox(FakeRobot())
    assert task_tools(3, tb) == TASK_TOOLS[3] and "search_cell" not in TASK_TOOLS[2] and "search_cell" not in TASK_TOOLS[3]
    tb.wall_layout = dict(LAYOUT)
    assert task_tools(3, tb) == TASK_TOOLS[3] + ["search_cell"] and task_tools(2, tb) == TASK_TOOLS[2]


# ------------------------------------------------------------------ 2026-11-19 layout test run (agent_full) fixes
def test_summary_holds_a_look_target_with_alternative_poses():
    """look_head crashed (TypeError) while ws.wrist_target was a look target: its 'alternatives' is a list of dicts."""
    tb = kin_toolbox()
    tb.ws.wrist_target = {"pos": np.array([1.0, 0.123456, 0.8]), "offset": 0.3, "purpose": "look",
                          "alternatives": [{"pos": [1.0, 0.1, 0.8], "roll_deg": 90.0, "long_axis": np.array([0.0, 0.0, 1.0])},
                                           {"pos": [1.0, 0.2, 0.8], "roll_deg": 270.0, "long_axis": [0.0, 0.0, -1.0]}],
                          "rect": [[0.0, 0.1], [0.2, 0.3]], "empty": []}
    s = tb.ws.summary()["wrist_target"]
    assert s["pos"] == [1.0, 0.123, 0.8] and s["rect"] == [[0.0, 0.1], [0.2, 0.3]] and s["empty"] == []
    assert s["alternatives"][0] == {"pos": [1.0, 0.1, 0.8], "roll_deg": 90.0, "long_axis": [0.0, 0.0, 1.0]}
    assert s["alternatives"][1]["long_axis"] == [0.0, 0.0, -1.0]
    json.dumps(s)


def test_walking_tools_refuse_while_the_arm_is_raised():
    tb = kin_toolbox()
    tb.arm_at_home = False
    reached = object()
    tb._stance_reached = reached
    g = tb._walk_guard()
    assert g is not None and g["status"] == "fail" and "move_arm('home')" in g["reason"]
    assert tb._stance_reached is reached                         # a refused walk keeps the reached stance


# ------------------------------------------------------------------ 2026-11-19 layout rule run: look-pose aim, scan stance
def _sagging_arm(tb, sag_deg, joint="wrist_pitch"):
    """The arm of a fake robot reaches every command with a steady joint offset (a PD / gravity sag of one joint: an
    orientation error that does not change between nearby commands)."""
    from g1agent.kinematics.ik import ARM
    q0 = tb.kin.q_from_state([], [])
    sag = np.zeros(len(ARM))
    sag[ARM.index(joint)] = math.radians(sag_deg)
    cur = {"q": q0.copy()}

    def set_arm(side, q_arm, dur):
        q = q0.copy()
        q[tb.ik.qi] = np.asarray(q_arm, float) + sag
        cur["q"] = q
        return {"status": "ok"}
    tb.robot.set_arm_joints = set_arm
    tb.robot.stand = lambda s: {"status": "ok"}
    tb._q = lambda: cur["q"].copy()
    return cur, q0


def _look_setup():
    """The look stance plan_stance chooses for C1 R4 from (0.5, 0) (stance grid: face_dist 0.70, lateral -0.15, roll 270),
    its look target and the aim of its face view (_look_view): the rectangle planned (grown by delta_lat and e) and the
    one that has to be seen (grown by delta_lat)."""
    tb = kin_toolbox()
    _planned(tb)
    z = float(tb._lean_levels()[0]["pelvis_z"])
    tb.next_barcode()
    bc = tb._current()
    Tr = T_from_xy_yaw(0.6, 0.35, 0.0, z)
    tb._odom = lambda: Tr
    t = tb._look_target(bc)
    alt = next(a for a in t["alternatives"] if a["roll_deg"] == 270.0)
    ta = dict(t, pos=alt["pos"], long_axis=alt["long_axis"])
    dl, e = tb._search_grow()
    c, r = bc.wall["cell"]
    aim = {"corners": tb._rect_face(*tb._cell_rect(c, r, dl + e, e))["corners"],
           "see": tb._rect_face(*tb._cell_rect(c, r, dl, 0.0))["corners"]}
    return tb, ta, aim, e


@pytest.mark.parametrize("joint, sag_deg", [("wrist_pitch", 6.0), ("wrist_roll", 5.0), ("shoulder_pitch", -5.0)])
def test_look_pose_closed_loop_corrects_the_full_tip_pose_under_a_steady_sag(joint, sag_deg):
    """2026-11-19 scan room (rule run, cells C3R4 / C4R4): the look pose reached with the tip 8 mm off (below
    correction_tol, the tip loop stopped) and the scan axis 4.8 deg off left the face uncovered. A fake arm with a
    steady joint sag: the look loop corrects the full tip pose (orientation pre-compensated as the position is) and
    covers the face within the correction budget, without leaving IK reach, also where a camera translation alone needed
    a step out of reach (shoulder_pitch -5 deg: 6 cm) or could not undo the roll (wrist_roll)."""
    tb, ta, aim, e = _look_setup()
    off = float(ta["offset"])
    cur, q0 = _sagging_arm(tb, sag_deg, joint)
    r = tb._reach(ta, off, tb.side, aim=aim)
    h = r["history"]
    A = tb.cfg["robot"]["arm_motion"]
    assert not r["ik_fail"] and not any(x.get("reverted_to_best") for x in h) and len(h) <= 1 + A["correction_iters"]
    assert h[0]["view_err_m"] > e and not h[0]["planned_covered"]
    assert h[-1]["planned_covered"] or h[-1]["view_err_m"] < e
    assert tb._view_error(ta, off, aim["see"])["covered"]
    if joint == "wrist_roll":
        assert h[0]["roll_err_deg"] > 4.0 and h[-1]["roll_err_deg"] < 1.0
    # no sag: the planned pose itself, one command
    _sagging_arm(tb, 0.0)
    r0 = tb._reach(ta, off, tb.side, aim=aim)
    assert len(r0["history"]) == 1 and r0["history"][0]["planned_covered"] and r0["history"][0]["view_err_m"] < 0.003


def test_look_pose_first_command_is_held_to_its_view_not_to_the_ik_tolerance_of_a_scan_pose():
    """2026-11-19 rule run: every squat look stance failed the first IK of its look pose by 4-7 mm / 0.7-1.4 deg (scan
    pose tolerance 2 mm / 1 deg) and the box was lost. A look pose's IK attempt that did not converge is sent when its
    view (FK of the solution) shows the rectangle that has to be seen; one whose view misses it is an IK failure, nothing
    sent. A scan pose's attempt is held to the placement check instead (test_scan_pose_unconverged_first_command_...)."""
    tb, ta, aim, e = _look_setup()
    off = float(ta["offset"])
    cur, q0 = _sagging_arm(tb, 0.0)
    real = tb.ik.solve
    sent = []
    set_arm = tb.robot.set_arm_joints
    tb.robot.set_arm_joints = lambda side, q, dur: (sent.append(q), set_arm(side, q, dur))[1]
    tb.ik.solve = lambda *a, **k: dict(real(*a, **k), ok=False)          # converged to ~2 mm, reported not ok
    r = tb._reach(ta, off, tb.side, aim=aim)
    assert not r["ik_fail"] and len(sent) == 1 and r["history"][0]["ik_unconverged"]["pos_err_m"] < 0.003
    assert tb._view_error(ta, off, aim["see"])["covered"]
    sent.clear()
    cur["q"] = q0.copy()
    tb.ik.solve = lambda *a, **k: dict(real(*a, **k), ok=False, q_full=q0.copy(), q_arm=q0[tb.ik.qi].copy())
    r = tb._reach(ta, off, tb.side, aim=aim)                                   # an attempt that looks elsewhere
    assert r["ik_fail"] and r["reason"].startswith("IK: target not reachable") and not sent


def _label_target(pos, forward, offset):
    """A label wrist target as compute_wrist_target makes it (vertical bars: the long axis horizontal on the face)."""
    f = np.asarray(forward, float) / np.linalg.norm(forward)
    L = np.cross(-f, [0.0, 0.0, 1.0])
    return {"pos": np.asarray(pos, float), "forward": f, "offset": offset, "barcode_id": "bc0", "long_axis": L / np.linalg.norm(L)}


def test_scan_pose_from_the_look_stance_is_the_stance_test_of_plan_stance():
    """The logged look stances and label targets of the 2026-11-19 rule run (odometry, pelvis z of the upright level):
    C1 R4 was scanned from its look stance, from the C3 R4 stance move_arm('final') ended 0.117 m off the scanner axis
    (the label 0.47 m to the robot's left). The stance test of plan_stance (planning IK of the final pose) tells them
    apart."""
    tb = kin_toolbox()
    tb.body_cmd = None                                     # no body model (collision geometry) in the CPU tests
    z = 0.746
    cases = [((0.576, 0.311, 0.032), ([1.006, 0.585, 0.982], [0.999, 0.049, 0.0], 0.266), True),
             ((0.619, -0.275, 0.061), ([0.991, 0.17, 0.983], [0.998, 0.068, 0.0], 0.3), False)]
    for (x, y, yaw), (pos, fwd, off), reachable in cases:
        tb._odom = lambda x=x, y=y, yaw=yaw: T_from_xy_yaw(x, y, yaw, z)
        r = tb._scan_pose_from_here(_label_target(pos, fwd, off))
        assert r["ok"] is reachable and r["status"] == ("ok" if reachable else "ik")
    assert tb.ik.cfg["restarts"] == config.apply_robot_variant(config.load_all(), "g1_v2")["robot"]["ik"]["restarts"]


def test_search_cell_label_out_of_reach_of_the_look_stance_gets_its_own_scan_stance():
    """The label found is not reachable from the look stance: the reply names the label path from plan_stance on, the
    look stance is dropped (move_arm('final') refuses), and plan_stance plans a scan stance (pre-pose, scan stand-off)
    for it although the look stance was recorded failed before (its key is the box's look, not the label's target)."""
    x0, _ = cell_x(1)
    tb, bc, cam = search_setup([[x0 + 0.01, x0 + 0.11, 0.957, 1.0]], scan_here=False)
    tb._failed_stances.add((bc.id, ("look",), 0.7, -0.15, 0))      # an earlier partial search at a look stance
    r = tb.search_cell()
    assert r["status"] == "ok" and r["data"]["converted"] and r["data"]["scan_from_this_stance"] is False
    assert "plan its scan stance" in r["reason"] and "move_arm('home'), compute_wrist_target -> plan_stance" in r["reason"]
    assert "move_arm('pre')" in r["reason"] and tb.ws.stance_target is None and bc.kind == "label"
    # the label path of the prompt and of rule mode: final -> refine -> final before the placement check
    assert ("move_arm('final') -> refine_with_wrist -> move_arm('final') -> check_wrist_placement -> scan"
            in r["reason"])
    m = tbmod.Toolbox.move_arm(tb, "final")
    assert m["status"] == "fail" and "no stance planned" in m["reason"]
    assert tb.ws.wrist_target.get("purpose") is None
    # the scan stance (planned from the look stance; standing level only, no body model in the CPU tests)
    z = float(tb._lean_levels()[0]["pelvis_z"])
    tb.body_cmd = None
    tb.ik.cfg["plan_workers"] = 0
    tb._odom = lambda: T_from_xy_yaw(0.6, 0.35, 0.0, z)
    p = tb.plan_stance()
    st = tb.ws.stance_target
    assert p["status"] == "ok" and st.get("purpose") is None and st["target_pos"] != ["look"]
    assert p["data"]["rejected"]["failed_placement"] == 0
    # reachable from the look stance: the label is scanned from there
    tb, bc, cam = search_setup([[x0 + 0.01, x0 + 0.11, 0.957, 1.0]], scan_here=True)
    r = tb.search_cell()
    assert r["data"]["scan_from_this_stance"] is True and r["reason"].startswith("label found in the box face; call move_arm")
    assert tb.ws.stance_target is not None


def test_search_cell_look_views_are_commanded_with_their_aim():
    """Every look view (the face view and the view centred on a cut label) is reached with the aim of its own
    rectangles: the one planned (the cell grown by delta_lat and e, the loop's stop) and the one to be seen (grown by
    delta_lat, the first command's test)."""
    x0, x1 = cell_x(1)
    tb, bc, cam = search_setup([[x1 - 0.03, x1 + 0.07, 0.96, 0.99]])
    tb.search_cell()
    T = tb._wall_T()
    _, e = tb._search_grow()
    assert len(cam.aims) >= 2 and all(a is not None for a in cam.aims)
    planned, see = apply(inv_T(T), cam.aims[0]["corners"]), apply(inv_T(T), cam.aims[0]["see"])
    assert planned[:, 0].min() == pytest.approx(x0 - e) and planned[:, 0].max() == pytest.approx(x1 + e)
    assert planned[:, 2].min() == pytest.approx(0.75 - e) and planned[:, 2].max() == pytest.approx(1.0 + e)
    assert see[:, 0].min() == pytest.approx(x0) and see[:, 0].max() == pytest.approx(x1)
    assert see[:, 2].min() == pytest.approx(0.75) and see[:, 2].max() == pytest.approx(1.0)
    second = apply(inv_T(T), cam.aims[1]["see"])
    assert second[:, 0].max() - second[:, 0].min() == pytest.approx(x1 - x0) and second[:, 0].min() > x0


def test_rule_layout_search_retry_and_scan_stance_replan_are_separate():
    """The 2026-11-19 rule run, C3 R4: a partial search, the search again from a new stance found the label, then
    move_arm('final') failed (IK) at the look stance and the box was given up (one shared re-plan flag). Now the search
    retry and the stance re-plan are separate; a label out of reach of the look stance goes to the label path."""
    cfg = config.load_all()["task3"]
    tb = LayoutRuleTB(FakeRobot({"bc0": [read("A")]}), {"bc0": ["partial", "label"]}, final_ik_fails=1)
    h = _layout_harness(tb)
    rule.Task3(h, cfg, "all").run()
    assert [c[0] for c in tb.calls].count("search_cell") == 2
    assert [c for c in tb.calls if c[0] == "plan_stance"] == [("plan_stance", "cell")] * 2 + [("plan_stance", "label")]
    assert ("move_arm", "pre") in tb.calls and h.finished["success"] is True
    for outcomes in (["label_far"], ["partial", "label_far"]):
        tb = LayoutRuleTB(FakeRobot({"bc0": [read("A")]}), {"bc0": list(outcomes)})
        h = _layout_harness(tb)
        rule.Task3(h, cfg, "all").run()
        tools = [r["tool"] for r in h.log.tools]
        i = len(tools) - 1 - tools[::-1].index("search_cell")
        # straight to the label path: arm home, then the label's own stance, pre-pose, refine, final
        assert tools[i + 1:i + 7] == ["move_arm", "compute_wrist_target", "plan_stance", "approach", "move_arm",
                                      "refine_with_wrist"]
        k = tb.calls.index(("plan_stance", "label"))
        assert tb.calls[k - 1] == ("compute_wrist_target", "bc0") and tb.calls[k + 1:k + 3] == [
            ("approach", "stance_target"), ("move_arm", "pre")]
        assert [c for c in tb.calls if c[0] == "plan_stance"] == [("plan_stance", "cell")] * len(outcomes) + [("plan_stance", "label")]
        assert ("move_arm", "pre") in tb.calls and h.finished["success"] is True


def test_rule_layout_searches_again_from_a_new_stance_when_the_look_pose_is_not_reachable():
    """2026-11-19 rule run: search_cell failed with the look pose out of IK reach at every squat stance; the tool
    recorded the stance as failed and said to re-plan, rule mode gave the box up. Now it re-plans and searches once
    more, as after a partial search (one shared retry: a second failure ends the box)."""
    cfg = config.load_all()["task3"]
    tb = LayoutRuleTB(FakeRobot({"bc0": [read("A")]}), {"bc0": ["ik", "label"]})
    h = _layout_harness(tb)
    rule.Task3(h, cfg, "all").run()
    assert [c for c in tb.calls if c[0] in ("search_cell", "plan_stance")] == [
        ("plan_stance", "cell"), ("search_cell", "bc0"), ("plan_stance", "cell"), ("search_cell", "bc0")]
    assert h.finished["success"] is True
    for outcomes in (["ik", "ik"], ["partial", "ik"], ["ik", "partial"]):
        tb = LayoutRuleTB(FakeRobot(), {"bc0": list(outcomes)})
        h = _layout_harness(tb)
        rule.Task3(h, cfg, "all").run()
        assert [c[0] for c in tb.calls].count("search_cell") == 2 and h.finished["success"] is False


# ------------------------------------------------------------------ 2026-11-20 rule run: scan-pose sag, squat body model
SCAN_STANCE = ((0.576, 0.311, 0.032), ([1.006, 0.585, 0.982], [0.999, 0.049, 0.0], 0.266))   # C1 R4, scanned from there


def _scan_setup():
    """The logged look stance of C1 R4 (standing, odometry) and its label target, reachable from there."""
    tb = kin_toolbox()
    tb.body_cmd = None
    (x, y, yaw), (pos, fwd, off) = SCAN_STANCE
    tb._odom = lambda: T_from_xy_yaw(x, y, yaw, 0.746)
    return tb, _label_target(pos, fwd, off), off


def _placement(tb, t, off):
    fwd = np.asarray(t["forward"], float)
    T = tb._odom() @ tb.kin.pose(tb._q(), tb.cfg["robot"]["wrist_point_frame"])
    return tb._placement_measures(T, np.asarray(t["pos"]) + t["offset"] * fwd, -fwd, off, np.asarray(t["long_axis"]))


@pytest.mark.parametrize("joint, sag_deg", [("wrist_pitch", 6.0), ("wrist_yaw", -6.0), ("wrist_roll", 5.0)])
def test_scan_pose_closed_loop_corrects_the_full_tip_pose_under_a_steady_sag(joint, sag_deg):
    """2026-11-19/20 logs: every label pose (9 of 9, rule layout run and r234 agent_full run) ended 2.7-9.8 deg off the
    label normal after the tip-position loop, and 3 of 6 placements failed the check's axis offset (the label centre
    1.7-3.4 cm off the scanner axis). The scan-pose loop now pre-compensates the commanded rotation as the look loop
    does (one loop): under a steady joint sag the first command misses the check, the corrected pose passes it with the
    tip, the axis and the roll corrected, within the correction budget and without a revert; no sag: one command."""
    tb, t, off = _scan_setup()
    cur, q0 = _sagging_arm(tb, sag_deg, joint)
    A = tb.cfg["robot"]["arm_motion"]
    r = tb._reach(t, off, tb.side)
    h = r["history"]
    assert not r["ik_fail"] and not any(x.get("reverted_to_best") for x in h) and 1 < len(h) <= 1 + A["correction_iters"]
    assert set(h[0]) == {"pos_err_m", "angle_err_deg", "axis_offset_m"}
    T = tb.kin.pose(tb._q(), tb.cfg["robot"]["wrist_point_frame"])
    _, d = tb._target_in_root(t, offset=off)
    roll = tb.ik.roll_error_deg(T[:3, :3], d, tb._roll_in_root(t))
    pm = _placement(tb, t, off)
    assert pm["good"] and max(h[-1]["pos_err_m"], h[-1]["axis_offset_m"]) < A["correction_tol"]
    assert h[-1]["angle_err_deg"] < 1.0 and roll < 1.0
    if joint != "wrist_roll":                               # a tilt misses the label by about range x tilt
        assert h[0]["angle_err_deg"] > 4.0 and h[0]["axis_offset_m"] > 0.015
    _sagging_arm(tb, 0.0)
    r0 = tb._reach(t, off, tb.side)
    assert len(r0["history"]) == 1 and r0["history"][0]["axis_offset_m"] < 0.003


def test_scan_pose_unconverged_first_command_is_held_to_the_placement_check():
    """2026-11-20 rule run i=145: a re-planned label pre-pose failed IK by 7 mm / 3.1 deg (IK tolerance 2 mm / 1 deg) and
    the barcode was lost. A scan pose's IK attempt that did not converge is sent when its FK pose passes the placement
    check's tolerances at that distance (task2.yaml check, what check_wrist_placement accepts); one the check would
    reject (the attempt 3 cm to the side: axis offset > axis_offset_max_m 0.02) is an IK failure, nothing sent."""
    tb, t, off = _scan_setup()
    cur, q0 = _sagging_arm(tb, 0.0)
    real = tb.ik.solve
    sent = []
    set_arm = tb.robot.set_arm_joints
    tb.robot.set_arm_joints = lambda side, q, dur: (sent.append(q), set_arm(side, q, dur))[1]
    tb.ik.solve = lambda *a, **k: dict(real(*a, **k), ok=False)          # converged to ~2 mm, reported not ok
    r = tb._reach(t, off, tb.side)
    assert not r["ik_fail"] and len(sent) == 1 and r["history"][0]["ik_unconverged"]["pos_err_m"] < 0.003
    assert _placement(tb, t, off)["good"]
    sent.clear()
    cur["q"] = q0.copy()
    side_root = tb._roll_in_root(t)                                        # horizontal, along the face
    tb.ik.solve = lambda q, p, *a, **k: dict(real(q, np.asarray(p) + 0.03 * side_root, *a, **k), ok=False)
    r = tb._reach(t, off, tb.side)
    assert r["ik_fail"] and r["reason"].startswith("IK: target not reachable") and not sent


def test_scan_pose_correction_out_of_ik_reach_keeps_its_position_part():
    """A steady wrist_pitch sag of 4 deg with wrist_pitch 0.5 deg from its IK limit at the first solution (the logged
    C1 R4 stance): the pre-compensated rotation leaves the IK's reach. The revert to the first pose left the label 3.2
    cm off the scanner axis (the check fails; 2026-11-20 review), the position step with the rotation commanded before
    reaches it (10b05eb's position-only correction: 1.95 cm, passes). Now that step is sent: logged rotation_kept, no
    revert, the check passes."""
    from g1agent.kinematics.ik import ARM
    tb, t, off = _scan_setup()
    _sagging_arm(tb, 0.0)
    p, d = tb._target_in_root(t, offset=off)
    j = ARM.index("wrist_pitch")
    tb.ik.lo[j] = tb.ik.solve(tb._q(), p, d, roll_axis=tb._roll_in_root(t))["q_arm"][j] - math.radians(0.5)
    _sagging_arm(tb, 4.0, "wrist_pitch")
    r = tb._reach(t, off, tb.side)
    h = r["history"]
    assert not r["ik_fail"] and not any(x.get("reverted_to_best") for x in h)
    assert any(x.get("rotation_kept") for x in h) and _placement(tb, t, off)["good"]


def test_scan_pose_unconverged_first_command_is_an_ik_failure_when_the_pose_reached_fails_the_check():
    """The unconverged first command is accepted on the FK of the IK solution, before the sag. At the edge of reach the
    corrections are out of IK reach, the arm reverts to the first pose, and its IK residual plus the sag fails the
    check: 2026-11-20 review, 4 of 4 such cases (IK residual -4..4 deg, sag 5 deg) ended "partial" 2.1-5.8 cm off the
    axis, and rule mode's single re-plan (on "IK" failures only) never ran. The pose reached is held to the same test:
    an IK failure, as at 10b05eb."""
    from scipy.spatial.transform import Rotation
    for tilt in (-4.0, 4.0):
        tb, t, off = _scan_setup()
        _sagging_arm(tb, 5.0, "wrist_pitch")
        real = tb.ik.solve
        n = {"k": 0}

        def solve(q, p, d, *a, **k):
            n["k"] += 1
            if n["k"] == 1:                               # the best attempt `tilt` deg off, not converged
                ax = np.cross(d, [0, 0, 1.0])
                d2 = Rotation.from_rotvec(math.radians(tilt) * ax / np.linalg.norm(ax)).apply(d)
                return dict(real(q, p, d2, *a, **k), ok=False)
            return dict(real(q, p, d, *a, **k), ok=False, q_full=q, q_arm=q[tb.ik.qi])   # every correction out of reach
        tb.ik.solve = solve
        r = tb._reach(t, off, tb.side)
        assert r["ik_fail"] and r["reason"].startswith("IK: target not reachable"), r.get("reason")
        assert r["history"][0]["ik_unconverged"] and not _placement(tb, t, off)["good"]


def _body(tb):
    """BodyModel on the g1_v2 kinematics with the configured squat calibration and a minimal collision geometry (one
    leg and one arm link: the geometry plays no part in the level model)."""
    import os
    from g1agent.tools.body import BodyModel
    cal = json.load(open(os.path.join(config.ROOT, tb.cfg["robot"]["body"]["calibration"])))["levels"]
    geom = {"body_names": ["left_knee_link", "left_elbow_link"], "points": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            "radius": [0.05, 0.05], "owner": [0, 1]}
    return BodyModel(tb.kin, geom, cal, "left", tb.cfg["robot"]["wrist_point_frame"])


def _tilt(pitch_deg, roll_deg=0.0):
    """Ry(pitch) Rx(roll): the odometry rotation with the yaw removed (R = Rz Ry Rx, g1_barcode_sim env._update_odom)."""
    from scipy.spatial.transform import Rotation
    return Rotation.from_euler("YX", [pitch_deg, roll_deg], degrees=True).as_matrix()


def test_body_level_measured_at_the_stance_replaces_the_calibration():
    """A squat level is planned with the state measured at it (pelvis height and full tilt from the odometry pose, leg /
    waist joints from the joint state) once the robot has been there; other levels keep the calibration; a command
    that is no calibrated level is not recorded."""
    tb = kin_toolbox()
    body = _body(tb)
    cal45 = next(c for c in body.levels if c["height_cmd"] == 0.45)
    T = T_from_xy_yaw(0.3, -0.2, 0.5, 0.449)
    T[:3, :3] = T[:3, :3] @ _tilt(21.2, 0.6)
    q = tb.kin.q_from_state([], [])
    for n, v in cal45["q"].items():
        if n in tb.kin.joint_q_index:
            q[tb.kin.joint_q_index[n]] = v
    q[tb.kin.joint_q_index["waist_pitch_joint"]] += 0.05
    m = body.observe(0.45, T, q)
    lv = {c["height_cmd"]: c for c in body.levels}
    assert lv[0.45] is m and m["measured"] and m["pelvis_z"] == pytest.approx(0.449)
    assert m["pelvis_pitch_deg"] == pytest.approx(21.2, abs=1e-6)
    assert all(not c.get("measured") for h, c in lv.items() if h != 0.45) and len(lv) == len(body.calibration)
    assert np.allclose(body.T_root(0.3, -0.2, 0.5, m), T, atol=1e-9)               # the measured pose, roll included
    q_lv = body.q_at(tb.kin.q_from_state([], []), m)
    assert q_lv[tb.kin.joint_q_index["waist_pitch_joint"]] == pytest.approx(cal45["q"]["waist_pitch_joint"] + 0.05)
    assert body.observe(0.47, T, q) is None and len(body.measured) == 1


def test_set_base_height_measures_the_level_only_with_the_arm_home():
    """set_base_height records the settled state of the level (the calibration's conditions: upright, arms home) and
    logs it in its diagnostics (not shown to the LLM); with the arm raised the waist balances the arm, nothing is
    recorded."""
    tb = kin_toolbox()
    tb._body_model = _body(tb)
    T = T_from_xy_yaw(0.3, -0.2, 0.5, 0.449)
    T[:3, :3] = T[:3, :3] @ _tilt(21.4)
    tb._odom = lambda: T
    tb.robot.set_body_pose = lambda h, rpy, ramp, settle: {"status": "ok", "reason": "", "pelvis_z": 0.449, "tilt_deg": 21.4}
    tb.arm_at_home = False
    r = tb.set_base_height(0.45)
    assert r["status"] == "ok" and "diagnostics" not in r and not tb._body_model.measured
    tb.arm_at_home = True
    r = tb.set_base_height(0.45)
    assert r["diagnostics"]["body_level_measured"]["pelvis_z"] == pytest.approx(0.449)
    assert set(r["diagnostics"]["body_level_measured"]["waist"]) == {"waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"}
    assert tb._body_model.levels[[c["height_cmd"] for c in tb._body_model.levels].index(0.45)]["measured"]


# the logged squat stances of the 2026-11-20 rule run whose poses failed IK after the squat: plan_stance's stance (pelvis
# position, yaw, level), the measured pelvis z / tilt after the squat (approach data base_height), the target
LOGGED_SQUAT = {
    # i=76 -> 78: look pose of C? R3 (alternative 0, roll 90 deg) at 0.70; search_cell IK 6 mm / 1.1 deg
    "look_070": {"stance": (0.823, -0.848, 0.108), "level": 0.70, "measured": (0.702, 1.424),
                 "target": {"pos": [1.007, -0.506, 0.625], "forward": [0.99326, 0.11588, 0.0], "offset": 0.33,
                            "long_axis": [0.0, 0.0, 1.0], "roll_signed": True, "purpose": "look", "barcode_id": "bc6"},
                 "finals": [0.33], "pre": None},
    # i=143 -> 145: label C5 R2 at 0.40 (pre gap -0.05: pre-pose 0.25 m, final the clamp 0.12 m); move_arm('pre') IK
    # 7 mm / 3.1 deg
    "label_040": {"stance": (0.956, -0.52, 0.172), "level": 0.40, "measured": (0.401, 25.052),
                  "target": {"pos": [1.004, -0.207, 0.472], "forward": [0.985, 0.171, 0.0], "offset": 0.3,
                             "long_axis": [0.171, -0.985, 0.0], "barcode_id": "bc16"},
                  "finals": [0.12], "pre": 0.25},
}


@pytest.mark.parametrize("case", sorted(LOGGED_SQUAT))
def test_logged_squat_stances_are_not_robust_on_the_measured_body_pose(case):
    """Root cause of the squat IK failures (plan accepted, execution failed by 4-14 mm): the stances were planned on the
    squat calibration (g1_v1's, robot.yaml: an approximation for g1_v2), whose pelvis is 11-22 mm lower than the
    pelvis g1_v2 settles at (all 14 logged squat arrivals). plan_stance's robustness test (the stance displaced by the
    walk tolerance) passes with the calibrated level and fails with the measured pelvis height and tilt (the measured
    tilt taken as pitch; the leg / waist joints of the calibration: the run logged no joint state): planned on the body
    pose the robot is measured in, these stances are not chosen. All 12 logged squat look stances behave so (2 of the
    7 displaced poses unreachable each, offline check)."""
    c = LOGGED_SQUAT[case]
    tb = kin_toolbox()
    body = _body(tb)
    lv = next(l for l in body.levels if l["height_cmd"] == c["level"])
    t = {k: (np.asarray(v, float) if isinstance(v, list) else v) for k, v in c["target"].items()}
    x, y, yaw = c["stance"]
    q_lv = body.q_at(tb.kin.q_from_state([], []), lv)
    tb.ik.cfg["restarts"] = 1                                                   # as plan_stance plans
    assert tb._stance_robust(t, body.T_root(x, y, yaw, lv), q_lv, c["finals"], c["pre"])
    z, tilt = c["measured"]
    T = T_from_xy_yaw(x, y, yaw, z)
    T[:3, :3] = T[:3, :3] @ _tilt(tilt)
    m = body.observe(c["level"], T, q_lv)
    assert not tb._stance_robust(t, body.T_root(x, y, yaw, m), q_lv, c["finals"], c["pre"])
