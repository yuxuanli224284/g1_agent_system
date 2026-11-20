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
from g1agent.common.transforms import apply, inv_T
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
                 poles=()):
    """Depth (along the optical axis) of the synthetic scene seen from camera pose Tc (odom); NaN where nothing is hit.
    poles: axis-aligned boxes [lo, hi] (odom) in front of the wall."""
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
        for lo, hi in poles:
            t1, t2 = (np.asarray(lo, float) - o) / D, (np.asarray(hi, float) - o) / D
            tn, tf = np.minimum(t1, t2).max(-1), np.maximum(t1, t2).min(-1)
            cand(tn, (tf >= tn) & (tn > 0))
    best[~np.isfinite(best)] = np.nan
    return best


def head_frame(tb, x, y=0.0, yaw=0.0, noise=0.0, seed=0, **scene):
    """Head frame of the synthetic scene from root (x, y, yaw), upright; T_odom_root = the root pose, T_root_cam = FK."""
    from g1agent.common.transforms import T_from_xy_yaw
    up = tb._lean_levels()[0]
    Tc = tb._cam_T_at(x, y, yaw, up)
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


def extent(h, plane=PLANE, cfg=None):
    return plane_fit.wall_extent(h["depth"], h["K"], h["T_root_cam"], h["T_odom_root"], plane,
                                 cfg or config.load("perception")["plane_fit"], h["depth_sigma_coeff"], 7, 0.05)


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
def test_wall_extent_noise_free_edges_within_one_pixel_and_crease_rows_dropped():
    tb = kin_toolbox()
    h = head_frame(tb, WALL_X - 1.2)
    ext = extent(h)
    px = 1.3 / FX_HEAD                                          # one pixel footprint at the wall (~2 mm)
    assert ext["left"]["s"] == pytest.approx(-0.6, abs=px) and ext["right"]["s"] == pytest.approx(0.6, abs=px)
    # the floor crease along the base (floor points within the plane tolerance) widens some rows: the median drops them
    rows = np.array([[r[1], r[2]] for r in ext["rows"] if r[3] and r[4]], float)
    assert np.abs(rows[:, 0] + 0.6).max() > 5 * px or ext["seen"][0] < -0.6 - 5 * px
    assert ext["left"]["n_rows"] > 100 and ext["right"]["n_rows"] > 100


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
    z0, z1]) clipped to the crop it is given (a label crossing the crop border is seen only in part, as a real crop)."""

    def __init__(self, tb, labels, err=None):
        self.tb, self.labels, self.err = tb, [list(map(float, l)) for l in labels], err
        self.reached, self.crops, self.homes = [], [], 0
        tb._reach = self.reach
        tb._capture_wrist = self.capture
        tb._labels_on_rectified_face = self.detect
        tb._reregister_at_stance = lambda: {"refit": False, "note": "test"}
        tb.move_arm = self.home

    def home(self, pose="pre", side=None):
        assert pose == "home"
        self.homes += 1
        self.tb.arm_at_home = True
        return tbmod.ok({"pose": "home"})

    def reach(self, t, off, side):
        self.reached.append(dict(t))
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


def search_setup(labels, order="rows", delta=0.0, err=None):
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
    return tb, bc, Cam(tb, labels, err)


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
    ("label" | "no_label" | "partial"); move_arm('final') fails as IK the first `final_ik_fails` times."""

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
        if o == "label":
            bc.kind = "label"
            return tbmod.ok({"barcode_id": bc.id, "converted": True, "added": []})
        if o == "no_label":
            bc.status = "no_label"
            return tbmod.ok({"barcode_id": bc.id, "barcode_status": "no_label"})
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
