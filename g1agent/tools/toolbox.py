"""Deterministic tools. Every tool returns {"status": ok|fail|partial, "data": {...},
"reason": str}; numbers are rounded to 3 decimals and no large arrays are returned (images only via look_*).

Frames: perception happens in the root frame at capture time and is stored in the odometry frame (WorldState);
camera poses come from the joint state through pinocchio FK (head <-> wrist via kinematics, not odometry).
No ground truth is used anywhere in this module (tests/test_no_gt_leak.py).

Robot variant (robot info robot_variant -> config.apply_robot_variant): the kinematic model, the scanner frames and the
hand camera. Two roles that coincide on g1_v1 and are separate devices on g1_v2:
- the SCANNER (decoding in scan(), its imager's field of view = scanner_hfov: stand-off, view cone; IK target scanner_tip,
  roll reference = its image axis);
- the HAND CAMERA (robot.yaml variants.<v>.hand_camera): every image the tools take on the scanner hand (look_wrist,
  refine_with_wrist detection / rays / symbol width, stereo_check_wrist). g1_v1: the scanner imager itself; g1_v2: the
  Intel D405 on the left-wrist bracket ("the original left-hand (wrist) camera now corresponds to the D405").

Task 3 (scan every barcode): plan_scans localises every label on every fitted face and fixes the order (scan_order.py:
columns left to right, top to bottom inside a column; or rows top to bottom, left to right inside a row when the task's
target spec says so); next_barcode walks the plan and clears the per-barcode slots; every barcode tool defaults to
ws.current_barcode_id and to the face copy stored with the barcode; scan dedupes by decoded content into the ws.scanned
ledger. A target spec in robot terms (set_scan_targets: columns / rows of the box wall counted from an edge) is resolved
on the label grid of the first plan of a side; the labels outside the requested cells stay in the plan as not_targeted.
A task with a given wall layout (set_wall_layout, approach A) plans boxes instead of labels: plan_scans registers the
wall (plane, side edges, floor) and plans one entry of kind "cell" per requested box in the lattice order; each box is
visited with the hand-camera search (compute_wrist_target -> plan_stance -> approach -> search_cell), which turns the
entry into its label or records no_label (see "task 3: wall layout" below). Every layout branch is gated on wall_layout.
"""
from __future__ import annotations

import math
import time

import numpy as np

from ..common import config
from ..common.transforms import T_from_xy_yaw, angle_between, apply, inv_T, rotate, wrap, yaw_of
from ..harness.world_state import BarcodeEstimate, BoxEstimate, WorldState
from ..kinematics.fk import Kinematics
from ..kinematics.ik import ArmIK
from ..perception import geometry as geo
from ..perception.barcode_detector import decode_symbol_polygons, detect_barcodes, detect_label_dino
from ..perception.plane_fit import fit_front_face, fit_support_plane, wall_extent
from .scan_order import (adaptive_standoff, cell_targeted, describe_targets, grid_cell, grid_conflict, hfov_from_intrinsics,
                         order_by_cells, order_in_cell, order_items,
                         order_items_in_frame, order_lattice, order_rule_text as scan_order_text, parse_targets,
                         quad_width, resolve_grid)


def R3(x):
    if x is None:
        return None
    if isinstance(x, dict):
        return {k: R3(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [R3(v) for v in (x.tolist() if isinstance(x, np.ndarray) else x)]
    if isinstance(x, (float, np.floating)):
        return round(float(x), 3)
    if isinstance(x, np.integer):
        return int(x)
    return x


def ok(data=None, reason=""):
    return {"status": "ok", "data": R3(data or {}), "reason": reason}


def _check_reason(reason):
    # ok(data, reason) and fail/partial(reason, data) take their arguments in opposite orders; a swapped call
    # silently put the data dict into "reason" (2026-11-06), so a non-string reason is an error, not a value
    if not isinstance(reason, str):
        raise TypeError(f"tool result reason must be a str, got {type(reason).__name__}: {reason!r:.120}")
    return reason


def fail(reason, data=None):
    return {"status": "fail", "data": R3(data or {}), "reason": _check_reason(reason)}


def partial(reason, data=None):
    return {"status": "partial", "data": R3(data or {}), "reason": _check_reason(reason)}


def with_diagnostics(res, diag):
    """Attach log-only diagnostics to a tool result (design D11: no behaviour change). The harness logs every result key
    but "image" (guards.Harness.call); the LLM's view of a result (orchestrators.common.tool_result_text) leaves
    "diagnostics" out, so they never change the LLM input."""
    if diag:
        res["diagnostics"] = R3(diag)
    return res


# fields of a move_root_relative reply that may be logged / shown to the agent: the simulator reply also carries
# ground-truth displacement / error fields (evaluation only), which must never reach the harness logs or the LLM
MOVE_LOG_KEYS = ("status", "reason", "elapsed_s", "requested", "odom_displacement")


def move_summary(r: dict) -> dict:
    return {k: r[k] for k in MOVE_LOG_KEYS if k in r}


class Toolbox:
    # task parameter: the given wall layout (set_wall_layout; None = no layout, every branch below is gated on it).
    # Class-level defaults so that every Toolbox, also one built without __init__ (tests), has them
    wall_layout = None
    _layout_edges = None              # wall side edges / extent seen in phase 1 (odom points), _layout_view_points
    _look_cache = None                # look-pose search results per (face size, delta, e), _search_view

    def __init__(self, robot, box_detector=None, device: str | None = None):
        from ..perception.box_detector import BoxDetector
        self.robot = robot
        # the model follows what the robot reports (never system.yaml): robot variant first (URDF, scanner / hand-camera
        # frames, calibrations, allowed head models), then the head camera
        self.info = robot.get_robot_info()
        self.cfg = config.load_all()
        config.apply_robot_variant(self.cfg, self.info.get("robot_variant"))
        config.apply_head_camera(self.cfg, self.info.get("head_camera_model"))   # frames, depth source, lean views
        self._check_robot_info(self.info)
        self.kin = Kinematics(self.cfg["robot"]["urdf"])
        self.ik = ArmIK(self.kin, self.cfg["robot"])
        self.det = box_detector or BoxDetector(device)
        self.side = self.cfg["robot"]["arm_side"]
        self.ws = WorldState()
        self.head = None
        self.wrist = None
        self.markers = {}
        self.episode_start_odom = self._odom()
        self.task = None                  # set by the Harness; selects the task config (task3.yaml for task 3)
        self._obs_faces = None            # faces compute_obs_pose planned with (frozen during the approach, see there)
        self.arm_at_home = True           # the episode starts with the arms in the default pose
        self._last_label = None           # head label detection of detect_barcode_head (task-2 path)
        self._wrist_detection = None      # last wrist detection of refine_with_wrist
        K, size = self.info.get("wrist_K"), self.info.get("wrist_size")
        # SCANNER horizontal field of view from the scanner imager's intrinsics (robot info wrist_K / wrist_size: 32.8 deg
        # for the DS2208-like imager, on every variant); None when the robot reports no scanner imager (the adaptive
        # stand-off then fails loudly). Never the hand camera's (the D405's 87 deg would put every label at min_m)
        self.scanner_hfov = None if K is None or size is None else hfov_from_intrinsics(float(np.asarray(K, float)[0, 0]), float(size[1]))
        # body (squat) command: None = the walker has none (stance planning stays at the standing height)
        bc = self.info.get("body_command")
        self.body_cmd = bc
        self.base_height_cmd = None if bc is None else float(bc["height_default"])
        self._body_model = None
        self.view_pitch = 0.0             # torso pitch offset of the current head view (0 = upright; robot.yaml view)
        self._leans = None
        # task parameter: scan every vertical face of the stack (front, right, back, left) instead of the front only
        # (set by the runner from the task instruction; the default keeps the single-face behaviour)
        self.multi_face = False
        # task parameter: which labels to scan, in robot terms (set_scan_targets; None = every label, column order)
        self.scan_targets = None
        # task parameters of the scenario (run_task.py): head views (set_head_views; None = the camera model's
        # robot.yaml lean_views) and where phase 1 of a target-spec task looks from (set_phase1_view)
        self.head_views = None
        self.phase1_view = "start_pose"
        self._obs_region = None           # phase-1 spec view region of the obs_pose mode (_spec_view_points)

    def _check_robot_info(self, info):
        """The robot must have its scanner on the configured arm and the variant's hand camera."""
        rc = self.cfg["robot"]
        if info.get("wrist_side") != rc["arm_side"]:
            raise ValueError(f"robot reports the scanner on side {info.get('wrist_side')!r}, robot.yaml arm_side is {rc['arm_side']!r}")
        hc = rc["hand_camera"]
        if hc["source"] == "hand_camera":
            cams = info.get("hand_cameras") or {}
            if hc["name"] not in cams:
                raise ValueError(f"robot variant {rc['robot_variant']}: hand camera {hc['name']!r} not reported by the robot "
                                 f"(hand_cameras: {sorted(cams)})")
            if cams[hc["name"]]["side"] != rc["arm_side"]:
                raise ValueError(f"hand camera {hc['name']!r} is on the {cams[hc['name']]['side']} hand, the scanner arm is {rc['arm_side']}")

    def set_scan_targets(self, spec):
        """Task instruction: which labels to scan, in robot terms (scan_order.parse_targets: columns / rows counted from
        an edge of the box wall, and the scan order rule). None = every label, columns left to right."""
        self.scan_targets = parse_targets(spec)

    HEAD_VIEWS = (None, "upright")
    PHASE1_VIEWS = ("start_pose", "obs_pose")

    def set_head_views(self, mode):
        """Scenario field head_views: None = the camera model's lean views (robot.yaml head_camera.<model>.lean_views);
        "upright" = standing head views upright only, no torso lean-back anywhere (user decision 2026-11-19 for the
        scan-room rows 2-4 task: no pitch-up, the top row is not scanned). Every lean goes through _lean_levels."""
        if mode not in self.HEAD_VIEWS:
            raise ValueError(f"head_views must be one of {self.HEAD_VIEWS}, got {mode!r}")
        self.head_views = mode
        self._leans = None

    def lean_views_on(self) -> bool:
        """The effective lean-view flag: the camera model has lean views and the scenario does not restrict the head
        to upright views (the flag _lean_levels and the task-3 prompt use)."""
        return bool(self.cfg["robot"]["lean_views"]) and self.head_views != "upright"

    def set_phase1_view(self, mode):
        """Scenario field phase1_view, for a task with a target spec: "start_pose" (default, None) = phase 1 stays near
        the start pose and plan_scans looks from there (user decision 2026-11-19, 4-box task); "obs_pose" = walk to the
        observation pose compute_obs_pose chooses from the spec (the wall edges the task counts from and the requested
        rows; may be behind the start, user decision 2026-11-19 Q1 for the rows 2-4 task)."""
        mode = "start_pose" if mode is None else mode
        if mode not in self.PHASE1_VIEWS:
            raise ValueError(f"phase1_view must be one of {self.PHASE1_VIEWS}, got {mode!r}")
        self.phase1_view = mode

    def _spec_view_mode(self) -> bool:
        """compute_obs_pose plans the view from the target spec (obs_pose mode of a target-spec task)."""
        return self.scan_targets is not None and self.phase1_view == "obs_pose"

    # ================================================================== body (squat / stand)
    def _body(self):
        """BodyModel (tools/body.py), built on first use: squat calibration + the robot's collision geometry."""
        if self.body_cmd is None:
            return None
        if self._body_model is None:
            import json
            import os
            from .body import BodyModel
            B = self.cfg["robot"]["body"]
            cal = json.load(open(os.path.join(config.ROOT, B["calibration"])))["levels"]
            self._body_model = BodyModel(self.kin, self.robot.get_robot_geometry(), cal, self.side,
                                         self.cfg["robot"]["wrist_point_frame"])
        return self._body_model

    def _standing(self):
        return self.base_height_cmd is None or self.base_height_cmd >= float(self.body_cmd["height_default"]) - 1e-6

    def set_base_height(self, height: float | None = None):
        """Squat to / stand up at a commanded pelvis height (GR00T body command), standing still. The planned squat
        heights come from plan_stance (data base_height); None (or the default height) stands up."""
        if self.body_cmd is None:
            return fail("this robot has no body (height) command")
        if height is None:
            height = float(self.body_cmd["height_default"])
        lo, hi = self.body_cmd["height_range"]
        if not lo <= float(height) <= hi:
            return fail(f"height must be in [{lo}, {hi}] m")
        B = self.cfg["robot"]["body"]
        # squat levels are calibrated with the torso upright: a leaned head view is undone with the height change
        r = self.robot.set_body_pose(float(height), self._torso_rpy(0.0), B["ramp_s"], B["settle_s"])
        if r["status"] != "ok":
            return fail(f"body command: {r['status']}: {r['reason']}")
        self.base_height_cmd = float(height)
        self.view_pitch = 0.0
        return ok({"height_cmd": float(height), "standing": self._standing(), "pelvis_z": r.get("pelvis_z"),
                   "tilt_deg": r.get("tilt_deg")}, r["reason"])

    # ================================================================== head views (torso lean)
    # The D435i head camera looks ~52 deg down on g1_v1 (~30 deg on g1_v2's raised mount) with a 42 deg vertical field of
    # view (colour): standing upright on g1_v1 it sees the floor 0.5-2 m ahead and the lower ~0.8 m of a stack at 0.6 m,
    # nothing above (g1_v2 at 0.6 m from a wall: z ~0.7-1.2 m, estimated). Leaning the torso back
    # (GR00T torso pitch command, waist joint) raises the view. A head view = (body height, torso pitch); the tools
    # choose it from what they have to see: the calibrated lean levels (robot.yaml view.calibration, measured in Isaac)
    # and FK give the camera pose of every view at any root pose. Walking, squatting and arm motions are upright.
    def _torso_rpy(self, pitch):
        r = np.asarray(self.body_cmd["rpy_default"], float).copy()
        r[1] += float(pitch)
        return r.tolist()

    def _view_now(self):
        return {"height_cmd": self.base_height_cmd, "pitch": self.view_pitch}

    def _lean_levels(self):
        """Calibrated standing head views, upright first, then increasingly leaned back (the levels whose head camera
        points higher than upright; measured: status ok). [] for a robot without a body command."""
        if self.body_cmd is None:
            return []
        if self._leans is None:
            import json
            import os
            V = self.cfg["robot"]["view"]
            cal = json.load(open(os.path.join(config.ROOT, V["calibration"])))["levels"]
            ok_ = [l for l in cal if l["status"] == "ok"]
            up = min(ok_, key=lambda l: abs(l["pitch_cmd"]))
            if abs(up["pitch_cmd"]) > 1e-9:
                raise ValueError("view calibration has no upright level (pitch_cmd 0)")
            higher = [l for l in ok_ if l["cam_pitch_below_horizontal_deg"] < up["cam_pitch_below_horizontal_deg"] - 1e-6]
            # a head camera without lean views (robot.yaml head_camera.<model>.lean_views), or a scenario with upright
            # head views only (set_head_views), stays upright
            self._leans = [up] + (sorted(higher, key=lambda l: abs(l["pitch_cmd"])) if self.lean_views_on() else [])
        return self._leans

    def _cam_T_at(self, x, y, yaw, lv):
        """Head camera pose (odom) standing at root ground position (x, y), yaw, in lean level lv (measured joints,
        pelvis height and tilt; the arm as it is now)."""
        from .body import Ry
        q = self._q()
        for n, v in lv["q"].items():
            if n in self.kin.joint_q_index and not any(k in n for k in ("shoulder", "elbow", "wrist", "hand")):
                q[self.kin.joint_q_index[n]] = float(v)
        T = T_from_xy_yaw(x, y, yaw, float(lv["pelvis_z"]))
        T[:3, :3] = T[:3, :3] @ Ry(math.radians(lv["pelvis_pitch_deg"]))
        return T @ self.kin.pose(q, self.cfg["robot"]["head_cam_frame"])

    def _in_view(self, Tc, pts):
        K = np.asarray(self.info["head_K"], float)
        H, W = self.info["head_size"]
        uv, z = geo.project(K, inv_T(Tc), np.asarray(pts, float))
        return geo.in_image(uv, z, W, H, self.cfg["perception"]["fov_margin_frac"]), z

    def _view_for(self, pts, x=None, y=None, yaw=None):
        """Lean level that puts the points (odom) in the head image from root pose (x, y, yaw) (default: here): the
        least lean that sees all of them, else the one that sees most (ties: less lean). (level, n_in)."""
        T0 = self._odom()
        x = T0[0, 3] if x is None else x
        y = T0[1, 3] if y is None else y
        yaw = yaw_of(T0) if yaw is None else yaw
        best = None
        for lv in self._lean_levels():
            inside, _ = self._in_view(self._cam_T_at(x, y, yaw, lv), pts)
            n = int(np.sum(inside))
            if n == len(inside):
                return lv, n
            if best is None or n > best[1]:
                best = (lv, n)
        return best

    def _set_view(self, lv):
        """Stand upright or lean back to a calibrated level, standing still (arm at home)."""
        p = float(lv["pitch_cmd"])
        if abs(p - self.view_pitch) < 1e-9 and self._standing():
            return ok({"pitch": p})
        if not self.arm_at_home:
            return fail("changing the head view needs the arm at home (call move_arm('home'))")
        V = self.cfg["robot"]["view"]
        r = self.robot.set_body_pose(float(self.body_cmd["height_default"]), self._torso_rpy(p), V["ramp_s"], V["settle_s"])
        if r["status"] != "ok":
            return fail(f"head view (torso pitch {p}): {r['status']}: {r['reason']}")
        self.view_pitch = p
        self.base_height_cmd = float(self.body_cmd["height_default"])
        return ok({"pitch": p, "tilt_deg": r.get("tilt_deg")})

    def _upright(self):
        """Standing upright (walking, squatting and arm motions start from here)."""
        if self.body_cmd is None or (abs(self.view_pitch) < 1e-9 and self._standing()):
            return None
        r = self._set_view(self._lean_levels()[0])
        return None if r["status"] == "ok" else r

    def _target_points(self):
        """Corners (odom) of what the robot is working toward: the faces compute_obs_pose planned with, else the fitted
        faces of the selected target; None when nothing is known yet."""
        if getattr(self, "_obs_faces", None):
            return np.vstack([f["corners"] for f in self._obs_faces.values()])
        if self.ws.target_box_id is not None:
            fs = self._faces(self.ws.target_box_id)
            if fs:
                return np.vstack([np.asarray(f["corners"], float) for f in fs])
        return None

    def _to_stack_top(self, C):
        """Corners (odom) to look at for a stack side: the given ones plus, when the stack model knows a higher top on
        any side, the top corners raised to it. The sides of a free-standing stack share its height; a side first seen
        from its near edge may be fitted without its top row (2026-11-13 stack4c_L0_0001: the back side was fitted
        without its top boxes, the observation pose was planned for a 0.9 m side, and the top label at 1.1 m was never
        in view). Views only: label positions still come from the faces actually fitted."""
        st = self.ws.stack
        C = np.asarray(C, float)
        if st is None or not st.get("faces"):
            return C
        H = max(float(np.asarray(f["corners"], float)[:, 2].max()) for f in st["faces"].values())
        zt = float(C[:, 2].max())
        if H <= zt + 1e-6:
            return C
        top = C[C[:, 2] > zt - 1e-6].copy()
        top[:, 2] = H
        return np.vstack([C, top])

    def _c1(self):
        """Task-1 parameters (approach, arrival, observation distance) of the running task."""
        return self.cfg["task3"] if self.task == 3 else self.cfg["task1"]

    def _c2(self):
        """Task-2 parameters (stand-off, stance, refine, place, check) of the running task."""
        return self.cfg["task3"] if self.task == 3 else self.cfg["task2"]

    # ================================================================== state helpers
    def _odom(self):
        return np.asarray(self.robot.get_odom()["T_odom_root"], float)

    def _q(self):
        js = self.robot.get_joint_state()
        return self.kin.q_from_state(js["names"], js["q"])

    def _capture_head(self):
        """Head frame: colour image, depth registered to it, camera pose (FK) and odometry. Depth source: perception.yaml
        head_depth.source (stereo = own matching on the infrared pair; render = the camera's delivered depth)."""
        src = self.cfg["perception"]["head_depth"]["source"]
        o = self.robot.get_head_obs(depth=src == "render")
        q = self._q()
        stereo_stats = None
        if src == "stereo":
            if "ir_left" not in o:
                raise RuntimeError("head_depth.source is stereo but the head camera gives no infrared pair")
            depth, stereo_stats = self._stereo(o).compute(o["ir_left"], o["ir_right"])
        elif src == "render":
            depth = o["depth"]
        else:
            raise ValueError(f"perception.yaml head_depth.source must be stereo | render, got {src!r}")
        self.head = {"rgb": o["rgb"], "depth": depth, "K": np.asarray(o["K"], float),
                     "T_root_cam": self.kin.pose(q, self.cfg["robot"]["head_cam_frame"]),
                     "T_odom_root": self._odom(), "t": o["timestamp"], "q": q, "stereo": stereo_stats,
                     "depth_sigma_coeff": self._stereo_depth.sigma_coeff if src == "stereo" else 0.0,
                     "view": self._view_now()}
        if src == "stereo":
            self.head["ir_left"] = o["ir_left"]
        return self.head

    def _min_structure_px(self):
        """Smallest surface the head depth resolves (px): the stereo matching window; 0 for delivered depth."""
        hd = self.cfg["perception"]["head_depth"]
        return int(hd["sgbm"]["block_size"]) if hd["source"] == "stereo" else 0

    def _stereo(self, o):
        """StereoDepth for the head's infrared pair, built on the first stereo frame (its calibration)."""
        if getattr(self, "_stereo_depth", None) is None:
            from ..perception.stereo_depth import StereoDepth
            si = self.info.get("head_stereo")
            if si is None:
                raise RuntimeError("robot info has no head_stereo (the head camera is not a stereo camera)")
            q = self._q()
            fr = self.cfg["robot"]
            T_color_left = inv_T(self.kin.pose(q, fr["head_cam_frame"])) @ self.kin.pose(q, fr["head_ir_left_frame"])
            self._stereo_depth = StereoDepth(self.cfg["perception"]["head_depth"], o["K_ir"], o["K_ir_right"],
                                             o["T_irleft_irright"], si["size"], self.info["head_K"], self.info["head_size"],
                                             T_color_left)
        return self._stereo_depth

    def _capture_wrist(self):
        """Hand-camera frame (robot.yaml variants.<v>.hand_camera): its image and K from the robot, its pose from FK of
        the SAME camera's frame (pixels of one camera posed with another camera's frame would corrupt every ray).
        g1_v1: the scanner imager (get_wrist_obs); g1_v2: the D405 (get_hand_camera_obs)."""
        hc = self.cfg["robot"]["hand_camera"]
        if hc["source"] == "scanner":
            o = self.robot.get_wrist_obs(self.side)
        elif hc["source"] == "hand_camera":
            o = self.robot.get_hand_camera_obs(hc["name"])
        else:
            raise ValueError(f"hand_camera.source {hc['source']!r}")
        if o.get("status") == "error":
            raise RuntimeError(o["reason"])
        q = self._q()
        self.wrist = {"rgb": o["rgb"], "K": np.asarray(o["K"], float), "T_root_cam": self.kin.pose(q, hc["frame"]),
                      "T_odom_root": self._odom(), "t": o["timestamp"], "q": q, "camera": hc["name"] or "scanner"}
        return self.wrist

    def _publish_markers(self):
        self.robot.draw_markers(list(self.markers.values()), frame="odom")

    def _marker(self, mid, color, kind, pos, **kw):
        self.markers[mid] = dict({"id": mid, "color": color, "kind": kind, "pos": [float(v) for v in pos]}, **{k: (np.asarray(v, float).tolist() if isinstance(v, (list, np.ndarray)) else v) for k, v in kw.items()})
        self._publish_markers()

    def _face(self, box_id):
        b = self.ws.boxes.get(box_id)
        return None if b is None else b.face

    def _project_odom(self, frame, pts_odom):
        """Project odom points into a captured camera frame -> (uv, z)."""
        T_cam_odom = inv_T(frame["T_odom_root"] @ frame["T_root_cam"])
        return geo.project(frame["K"], T_cam_odom, pts_odom)

    # ================================================================== observation tools
    def look_head(self):
        h = self._capture_head()
        return ok({"summary": self.ws.summary(), "timestamp": h["t"]}, "head image attached"), h["rgb"]

    def look_wrist(self):
        w = self._capture_wrist()
        return ok({"timestamp": w["t"]}, self.cfg["robot"]["texts"]["look_wrist_result"]), w["rgb"]

    def detect_boxes(self, prompt: str = "cardboard box", view: str = "auto"):
        """Detect boxes in the head image. view "auto" (standing, arm at home): with a target known (faces planned by
        compute_obs_pose / fitted), the head view that sees its faces from here (_view_for); with nothing known, the
        view with the most detections, searched from upright to the most leaned level (D435i: from the start pose the
        upright view sees only the floor). view "keep": the current view (plan updates while squatting / leaning)."""
        prompt = f"a {prompt}." if not prompt.startswith("a ") else prompt
        prev_target_face = self._face(self.ws.target_box_id) if self.ws.target_box_id else None
        searched = None
        can_move = view == "auto" and self.body_cmd is not None and self.arm_at_home and self._standing()
        pts = self._target_points() if can_move else None
        if can_move and pts is not None:
            lv, _ = self._view_for(pts)
            r = self._set_view(lv)
            if r["status"] != "ok":
                return r
        h = self._capture_head()
        dets = self.det.detect(h["rgb"], prompt=prompt)
        if can_move and pts is None:
            # nothing known: look from every standing view and keep the one with the most detections that are box
            # faces, i.e. whose depth region fits a vertical face plane (perception.yaml plane_fit, the test
            # fit_front_face applies next), then the most detections. 2026-11-13, D435i from the start pose (2.5 m):
            # upright the head sees only the floor, and the detector returned floor regions as "boxes" (one covering the
            # whole image, or two floor patches); a plain count, or the detection boxes' sides inside the image, kept
            # the floor view.
            def faces_in(hh, ds):
                return sum(fit_front_face(hh["depth"], hh["K"], hh["T_root_cam"], d["bbox"], sigma_coeff=hh["depth_sigma_coeff"], min_structure_px=self._min_structure_px())["status"] == "ok" for d in ds)
            searched = [{"pitch": self.view_pitch, "n": len(dets), "faces": faces_in(h, dets)}]
            best = ((searched[0]["faces"], len(dets)), self.view_pitch, h, dets)
            for lv in self._lean_levels():
                if abs(lv["pitch_cmd"] - searched[0]["pitch"]) < 1e-9:
                    continue
                r = self._set_view(lv)
                if r["status"] != "ok":
                    return r
                hh = self._capture_head()
                dd = self.det.detect(hh["rgb"], prompt=prompt)
                sc = (faces_in(hh, dd), len(dd))
                searched.append({"pitch": lv["pitch_cmd"], "n": sc[1], "faces": sc[0]})
                if sc > best[0]:
                    best = (sc, lv["pitch_cmd"], hh, dd)
            if abs(best[1] - self.view_pitch) > 1e-9:
                r = self._set_view(next(l for l in self._lean_levels() if abs(l["pitch_cmd"] - best[1]) < 1e-9))
                if r["status"] != "ok":
                    return r
                h = self._capture_head()                  # the frame at the chosen view (its pose for fitting)
                dets = self.det.detect(h["rgb"], prompt=prompt)
            else:
                h, dets = best[2], best[3]
                self.head = h
        self.ws.boxes = {f"box{i}": BoxEstimate(f"box{i}", d["bbox"], d["score"], h["t"]) for i, d in enumerate(dets)}
        self.ws.target_box_id = None
        if self.ws.target_selector is not None:
            self._reselect(prev_target_face)
        view_info = {"pitch": self.view_pitch, "height_cmd": self.base_height_cmd, "searched": searched,
                     "stereo": h.get("stereo")}
        if not dets:
            return fail("no box detected in the head image", {"n": 0, "view": view_info})
        return ok({"n": len(dets), "boxes": [{"id": f"box{i}", "bbox": d["bbox"], "score": d["score"]} for i, d in enumerate(dets)],
                   "order": "left to right in the image (robot's left first)", "target_box_id": self.ws.target_box_id,
                   "view": view_info})

    def _reselect(self, prev_face):
        ids = list(self.ws.boxes)                     # left -> right
        if self.ws.target_selector == "all":
            self.ws.target_box_id = "all" if ids else None
        elif len(ids) >= 2:
            self.ws.target_box_id = ids[0] if self.ws.target_selector == "left" else ids[-1] if self.ws.target_selector == "right" else None
        elif len(ids) == 1 and prev_face is not None:
            uv, z = self._project_odom(self.head, [prev_face["center"]])
            x0, y0, x1, y1 = self.ws.boxes[ids[0]].bbox
            pad = 0.25 * (x1 - x0)
            if z[0] > 0 and x0 - pad <= uv[0, 0] <= x1 + pad:
                self.ws.target_box_id = ids[0]

    def select_target_box(self, selector: str):
        if selector == "all":
            if not self.ws.boxes:
                return fail("no boxes detected yet (call detect_boxes)")
            self.ws.target_selector, self.ws.target_box_id = "all", "all"
            return ok({"target_box_id": "all", "box_ids": list(self.ws.boxes)})
        if selector in ("left", "right"):
            self.ws.target_selector = selector
            ids = list(self.ws.boxes)
            if len(ids) >= 2:
                self.ws.target_box_id = ids[0] if selector == "left" else ids[-1]
            elif len(ids) == 1:
                return fail("only one box visible: cannot tell whether it is the left or the right one", {"visible": ids})
            else:
                return fail("no boxes detected yet (call detect_boxes)")
        elif selector in self.ws.boxes:
            self.ws.target_box_id = selector
        else:
            return fail(f"unknown selector {selector!r}", {"known": list(self.ws.boxes)})
        b = self.ws.boxes[self.ws.target_box_id]
        return ok({"target_box_id": b.id, "bbox": b.bbox})

    def _box_ids(self, box_id):
        return list(self.ws.boxes) if box_id == "all" else [box_id]

    def _faces(self, box_id):
        """Fitted faces of a box id, or of every box for "all" (with a stack model: only faces of the current stack
        face, so faces of another side seen obliquely never mix into an observation pose)."""
        fs = [f for f in (self._face(b) for b in self._box_ids(box_id)) if f is not None]
        if box_id == "all" and self.ws.stack is not None:
            fs = [f for f in fs if self._face_id(f["normal"]) == self.ws.stack["current"]]
        return fs

    # ================================================================== stack model (multi-face)
    TOUR = ("front", "right", "back", "left")

    # The stack model lives in its own planar frame S (the odometry frame at the first plan) and is mapped to odometry
    # by a rigid transform (R_OS, t_OS) that is re-registered whenever a side is fitted afresh (_stack_add_faces).
    # Odometry drifts (the simulator models 1 deg yaw and 2 % distance per metre walked): 2026-11-07, walking around
    # the four-side stack the face directions predicted from the front drifted by 9 / 16 / 22 deg on the right / back
    # / left sides, the left side's walk, fit and stance obstacles went wrong and its 3 codes were judged unreachable.
    # Stored geometry is therefore only trusted relative to the latest measurement.
    def _stack_T(self):
        st = self.ws.stack
        return np.asarray(st.get("R_OS", np.eye(2)), float), np.asarray(st.get("t_OS", np.zeros(2)), float)

    def _stack_to_odom(self, C):
        R, t = self._stack_T()
        C = np.array(C, float)
        C[:, :2] = C[:, :2] @ R.T + t
        return C

    def _odom_to_stack(self, C):
        R, t = self._stack_T()
        C = np.array(C, float)
        C[:, :2] = (C[:, :2] - t) @ R
        return C

    def _stack_face(self, fid):
        """Stack side fid in odometry: outward normal, corners, centre (+ the observation distance used there)."""
        f = self.ws.stack["faces"][fid]
        C = self._stack_to_odom(f["corners"])
        return {"normal": self._face_dirs()[fid], "corners": C, "center": C.mean(axis=0), "d_obs": f.get("d_obs")}

    def _face_dirs(self):
        """Outward horizontal normals of the 4 stack faces in tour order (odometry): front n0, right Rz(+90) n0 (the
        robot's right while facing the front), back -n0, left Rz(-90) n0."""
        R, _ = self._stack_T()
        n0s = np.asarray(self.ws.stack["n0"], float)
        n0 = np.array([*(R @ n0s[:2]), 0.0])
        rz = lambda v, s: np.array([-s * v[1], s * v[0], 0.0])
        return [n0, rz(n0, 1.0), -n0, rz(n0, -1.0)]

    def _face_id(self, normal) -> int:
        """Stack face of a fitted normal: the nearest of the 4 face directions (they are 90 deg apart: no threshold)."""
        if self.ws.stack is None:
            return 0
        n = np.asarray(normal, float)
        nh = np.array([n[0], n[1], 0.0]); nh /= np.linalg.norm(nh)
        return int(np.argmax([float(nh @ d) for d in self._face_dirs()]))

    def _stack_init(self, faces):
        """The first plan defines the front: of the fitted faces, the one facing the robot most."""
        fwd = self._odom()[:3, 0]
        best = min(faces, key=lambda f: float(np.asarray(f["normal"], float)[:2] @ fwd[:2]))
        n = np.asarray(best["normal"], float)
        n0 = np.array([n[0], n[1], 0.0]) / np.linalg.norm(n[:2])
        self.ws.stack = {"n0": n0, "faces": {}, "current": 0, "visited": [0], "multi_face": bool(self.multi_face)}

    def _stack_add_faces(self, fid, faces):
        """Record the corners of the freshly fitted faces of stack side fid (the footprint is the hull of all sides).
        When the side before it in the tour is known, the stack frame is first re-registered to this fresh fit: the
        rotation from the side's fitted normal, the translation from the corner the two sides share, seen once as the
        end of the earlier side (its plane, its lateral extent toward this side) and once fresh as the end of this side
        (its plane, its extent toward the earlier side)."""
        if not faces:
            return
        st = self.ws.stack
        C = np.vstack([np.asarray(f["corners"], float) for f in faces])
        nb = (fid - 1) % len(self.TOUR)
        if fid != 0 and nb in st["faces"]:
            nh = np.sum([np.asarray(f["normal"], float)[:2] / np.linalg.norm(np.asarray(f["normal"], float)[:2]) for f in faces], axis=0)
            nh /= np.linalg.norm(nh)
            d_pred = self._face_dirs()[fid][:2]
            delta = math.atan2(d_pred[0] * nh[1] - d_pred[1] * nh[0], float(d_pred @ nh))
            Rd = np.array([[math.cos(delta), -math.sin(delta)], [math.sin(delta), math.cos(delta)]])
            R, t = self._stack_T()
            st["R_OS"], st["t_OS"] = Rd @ R, Rd @ t
            dirs = self._face_dirs()
            dk, dn = dirs[fid][:2], dirs[nb][:2]
            Fn = self._stack_face(nb)["corners"][:, :2]
            corner_model = float(np.mean(Fn @ dn)) * dn + float(np.max(Fn @ dk)) * dk
            Cf = C[:, :2]
            corner_fresh = float(np.mean(Cf @ dk)) * dk + float(np.max(Cf @ dn)) * dn
            st["t_OS"] = st["t_OS"] + (corner_fresh - corner_model)
            st.setdefault("registrations", []).append({"face": fid, "dyaw_deg": round(math.degrees(delta), 2),
                                                       "dxy_m": np.round(corner_fresh - corner_model, 3).tolist()})
        Cs = self._odom_to_stack(C)
        prev = st["faces"].get(fid)
        if prev is not None:
            Cs = np.vstack([np.asarray(prev["corners"], float), Cs])
        st["faces"][fid] = dict(prev or {}, corners=Cs)

    def _footprint(self):
        """Convex hull (xy, counter-clockwise) of every stack-face corner seen so far, once two non-parallel faces are
        known (before that the single front face is handled as a plane, as for a single-face task); None otherwise."""
        st = self.ws.stack
        if st is None or len(st["faces"]) < 2:
            return None
        fids = sorted(st["faces"])
        if all(abs(float(self._face_dirs()[a] @ self._face_dirs()[fids[0]])) > 0.5 for a in fids):
            return None                                   # only parallel faces (front + back): no closed footprint
        from scipy.spatial import ConvexHull
        P = np.vstack([self._stack_face(f)["corners"][:, :2] for f in fids])
        return P[ConvexHull(P).vertices]

    def fit_front_face(self, box_id: str | None = None):
        box_id = box_id or self.ws.target_box_id
        if box_id != "all":
            return self._fit_one(box_id)
        if not self.ws.boxes:
            return fail("no boxes detected (call detect_boxes)")
        res = {b: self._fit_one(b) for b in self.ws.boxes}
        good = [b for b, r in res.items() if r["status"] == "ok"]
        data = {"box_id": "all", "faces": [res[b]["data"] for b in good],
                "unfitted": {b: r["reason"] for b, r in res.items() if r["status"] != "ok"}}
        if len(good) == len(res):
            return ok(data)
        if good:
            return partial(f"front face fitted for {len(good)} of {len(res)} boxes", data)
        return fail("no front face fitted", data)

    def _fit_one(self, box_id):
        if box_id not in self.ws.boxes:
            return fail(f"unknown box {box_id!r}; call detect_boxes", {"known": list(self.ws.boxes)})
        h = self.head
        r = fit_front_face(h["depth"], h["K"], h["T_root_cam"], self.ws.boxes[box_id].bbox, sigma_coeff=h["depth_sigma_coeff"], min_structure_px=self._min_structure_px())
        if r["status"] != "ok":
            return fail(r["reason"], {"n_valid": r.get("n_valid")})
        T = h["T_odom_root"]
        face = {"center": apply(T, r["center"]), "normal": rotate(T, r["normal"]), "right": rotate(T, r["right"]),
                "up": rotate(T, r["up"]), "corners": apply(T, r["corners"]), "size": r["size"], "t": h["t"]}
        self.ws.boxes[box_id].face = face
        d = float(geo.horizontal_distance_to_plane(T[:3, 3], face["center"], face["normal"]))
        return ok({"box_id": box_id, "center_odom": face["center"], "normal_odom": face["normal"], "size_wh": r["size"],
                   "n_inliers": r["n_inliers"], "rms": r["rms"], "distance_m": d})

    @staticmethod
    def _faces_reference(faces):
        """Reference point and horizontal normal of one face, or of the union of several faces ("all"): the mean
        horizontal normal; the lateral middle of all corners; along the normal the face nearest to the robot (so
        d_obs is the clearance to the nearest face); the mean face-centre height."""
        nhs = []
        for f in faces:
            n = np.asarray(f["normal"], float)
            nhs.append(np.array([n[0], n[1], 0.0]) / np.linalg.norm(n[:2]))
        nh = np.sum(nhs, axis=0); nh /= np.linalg.norm(nh)
        if len(faces) == 1:
            return np.asarray(faces[0]["center"], float), nh
        lat = np.array([-nh[1], nh[0], 0.0])
        C = np.vstack([np.asarray(f["corners"], float) for f in faces])
        s = C @ lat
        a = max(float(np.asarray(f["center"], float) @ nh) for f in faces)
        z = float(np.mean([f["center"][2] for f in faces]))
        return 0.5 * (s.min() + s.max()) * lat + a * nh + np.array([0.0, 0.0, z]), nh

    def _whole_vertically(self, h, face) -> bool:
        """Both the top and the bottom edge of the face (its two highest / lowest corners) project inside head frame h
        with the perception margin (the test _note_face_edges applies): its height is the box's, not cut by the image."""
        C = np.asarray(face["corners"], float)
        uv, z = self._project_odom(h, C)
        H, W = h["rgb"].shape[:2]
        inside = geo.in_image(uv, z, W, H, self.cfg["perception"]["fov_margin_frac"])
        return bool(np.all(inside[np.argsort(-C[:, 2])[:2]]) and np.all(inside[np.argsort(C[:, 2])[:2]]))

    def _spec_view_points(self, faces):
        """Points (odom) the observation pose of a target-spec task must hold in the head view (obs_pose mode,
        set_phase1_view): the 4 corners of a rectangle on the front face plane (_faces_reference) holding what the spec
        counts from.
        - Lateral: the union of the lateral extents of every face fitted so far in phase 1 (self._obs_region only
          grows). A face cut by the image border ends at that border, inside the wall, so the union never exceeds it.
        - Vertical, rows counted from the bottom: from the floor (odom z 0: the simulator's odom z is the world pelvis z
          and stance planning assumes ground z 0) up to the highest requested row, max(index) x unit, unit = the
          smallest face height seen whole (the target grid's box size). No face seen whole yet: the floor line only (the
          nearest pose that sees the floor; whether the requested rows fit is then the grid's decision, with warnings).
        - Vertical otherwise (rows from the top, or no row key): today's corners raised to the known stack top; "all"
          rows also need the bottom edge: from the floor up to that top.
        Also returns the pose reference point: the region's lateral middle (not the middle of the faces fitted in this
        view, which after a walk may be a part of the region), at the plane and height of _faces_reference.
        With a wall layout: _layout_view_points."""
        if self.wall_layout is not None:
            return self._layout_view_points(faces)
        center, nh = self._faces_reference(faces)
        lat = np.array([-nh[1], nh[0], 0.0])
        C = np.vstack([np.asarray(f["corners"], float) for f in faces])
        reg = self._obs_region or {"pts": np.zeros((0, 3)), "unit": None}
        unit = reg["unit"]
        for f in faces:
            # judged in the head frame the face was fitted from (same capture time) only
            if self.head is not None and f["t"] == self.head["t"] and self._whole_vertically(self.head, f):
                hgt = float(np.ptp(np.asarray(f["corners"], float)[:, 2]))
                unit = hgt if unit is None else min(unit, hgt)
        pts = np.vstack([reg["pts"], C])
        self._obs_region = {"pts": pts, "unit": unit}
        s = pts @ lat
        rows = self.scan_targets["rows"]
        z_floor = 0.0
        if rows is not None and rows["from"] == "bottom" and not rows.get("all"):
            z_lo, z_hi = z_floor, z_floor + (max(rows["index"]) * unit if unit is not None else 0.0)
        else:
            top = self._to_stack_top(C)
            z_lo, z_hi = float(top[:, 2].min()), float(top[:, 2].max())
            if rows is not None and rows.get("all"):
                z_lo = z_floor
        a = float(np.asarray(center, float) @ nh)
        P = lambda sv, zv: sv * lat + a * nh + np.array([0.0, 0.0, zv])
        corners = np.array([P(s.max(), z_hi), P(s.min(), z_hi), P(s.min(), z_lo), P(s.max(), z_lo)])
        ref = P(0.5 * (s.min() + s.max()), float(np.asarray(center, float)[2]))
        return corners, {"unit": unit, "z_range": [z_lo, z_hi], "lateral": [float(s.min()), float(s.max())],
                         "n_face_fits": int(len(pts) // 4)}, ref

    def compute_obs_pose(self, box_id: str | None = None):
        box_id = box_id or self.ws.target_box_id
        faces = self._faces(box_id)
        if not faces:
            return fail("no front face for this box (call fit_front_face)")
        c1 = self._c1()
        center, nh = self._faces_reference(faces)
        spec_view = None
        if self._spec_view_mode():
            corners, spec_view, center = self._spec_view_points(faces)     # the pose centred on the region
        else:
            corners = self._to_stack_top(np.vstack([np.asarray(f["corners"], float) for f in faces]))
        yaw = math.atan2(-nh[1], -nh[0])
        T_root_cam = self.head["T_root_cam"] if self.head is not None else self.kin.pose(self._q(), self.cfg["robot"]["head_cam_frame"])
        z_root = self._odom()[2, 3]
        W, H = self.info["head_size"][1], self.info["head_size"][0]
        K = np.asarray(self.info["head_K"], float)
        n_steps = int(round((c1["d_obs_range"][1] - c1["d_obs_range"][0]) / c1["d_obs_step"]))
        leans = self._lean_levels()
        chosen, fits = None, False
        # the nearest distance at which some standing head view (upright first, then leaned back) sees every corner
        for k in range(n_steps + 1):
            d = c1["d_obs_range"][0] + k * c1["d_obs_step"]
            p = center + d * nh
            for lv in (leans or [None]):
                T_odom_cam = (T_from_xy_yaw(p[0], p[1], yaw, z_root) @ T_root_cam if lv is None
                              else self._cam_T_at(p[0], p[1], yaw, lv))
                uv, z = geo.project(K, inv_T(T_odom_cam), corners)
                if np.all(geo.in_image(uv, z, W, H, self.cfg["perception"]["fov_margin_frac"])):
                    chosen, fits = (p, d, lv), True
                    break
            if chosen is not None:
                break
        if chosen is None:
            d = c1["d_obs_range"][1]
            p = center + d * nh
            chosen = (p, d, None if not leans else self._view_for(corners, p[0], p[1], yaw)[0])
        p, d, lv = chosen
        self.ws.root_target = {"pos": [float(p[0]), float(p[1])], "yaw": yaw, "d_obs": d, "box_id": box_id, "fits": fits,
                               "n_faces": len(faces), "view_pitch": None if lv is None else float(lv["pitch_cmd"])}
        if spec_view is not None:
            # obs_pose mode: the in-view tests (check_face_in_view, arrive_check, detect_boxes' view choice) test the
            # spec region, held as one pseudo-face; it is recomputed at every re-perception (_reperceive_target)
            self._obs_faces = {"spec_region": {"center": corners.mean(axis=0), "normal": np.asarray(nh, float).copy(),
                                               "corners": np.asarray(corners, float).copy()}}
        else:
            # The faces this pose was planned with (full extents, normally seen from far) are frozen for the approach:
            # close-range refits see the faces cut by the image edge, so their extents (and hence the union reference
            # point, d_obs and the in-view test) are no longer the boxes' (2026-11-06: the goal moved by ~0.15 m between
            # fine moves and the in-view test failed at the planned pose). Refits then only correct the plane position.
            self._obs_faces = {b: {k: np.asarray(v, float).copy() for k, v in self._face(b).items() if k in ("center", "normal", "corners")}
                               for b in self._box_ids(box_id) if self._face(b) is not None}
        self._marker("root_target", "red", "root", [p[0], p[1], 0.0], yaw=yaw)
        data = {"root_target_odom": self.ws.root_target}
        if spec_view is not None:
            data["spec_view"] = dict(spec_view, corners_odom=corners)
        if fits:
            return ok(data)
        return partial("face does not fit the image with the margin at any d in range; using the upper bound", data)

    def check_face_in_view(self, box_id: str | None = None):
        box_id = box_id or self.ws.target_box_id
        rt = self.ws.root_target
        if getattr(self, "_obs_faces", None) and rt is not None and rt["box_id"] == box_id:
            faces = list(self._obs_faces.values())       # full extents (see compute_obs_pose), re-anchored by refits
        else:
            faces = self._faces(box_id)
        if not faces:
            return fail("no front face estimate")
        h = self._capture_head()
        uv, z = self._project_odom(h, np.vstack([np.asarray(f["corners"], float) for f in faces]))
        W, H = h["rgb"].shape[1], h["rgb"].shape[0]
        inside = geo.in_image(uv, z, W, H, self.cfg["perception"]["fov_margin_frac"])
        data = {"corners_px": uv, "inside": inside.tolist()}
        return ok(data) if inside.all() else fail("face corners outside the image (with margin)", data)

    # ================================================================== motion tools
    def _rel(self, pos, yaw):
        T = self._odom()
        rel = inv_T(T_from_xy_yaw(T[0, 3], T[1, 3], yaw_of(T))) @ T_from_xy_yaw(pos[0], pos[1], yaw)
        return rel[0, 3], rel[1, 3], wrap(yaw_of(rel))

    def move_to(self, pos, yaw, timeout=None):
        dx, dy, dyaw = self._rel(pos, yaw)
        r = self.robot.move_root_relative(dx, dy, dyaw, timeout or self._c1()["approach"]["move_timeout_s"])
        return r

    def _walk_guard(self):
        """Walking while squatted degrades badly (GR00T in MuJoCo: 0.38 m in 7 s at height 0.60 vs 1.09 m standing): a
        walking tool stands the robot up first, which needs the arm at home (it must not sweep through the legs while
        they extend). 2026-11-07: the agent_full LLM squatted before walking and never squatted again at the stance, so
        standing up / squatting belong to the tools, not to the call order."""
        self._stance_reached = None                     # every walking tool calls this guard first: the robot leaves
        if self._standing() and abs(self.view_pitch) < 1e-9:   # the stance it may have reached
            return None
        if self._standing():                            # leaned back for a head view: upright before walking
            if not self.arm_at_home:
                return fail("the robot is leaning back with the arm raised: call move_arm('home') first")
            r = self._upright()
            return None if r is None else fail(f"upright before walking: {r['reason']}")
        if not self.arm_at_home:
            return fail(f"the robot is squatting (height command {self.base_height_cmd:.2f} m) with the arm raised: "
                        "call move_arm('home') first (walking tools stand the robot up themselves)")
        r = self.set_base_height(None)
        if r["status"] != "ok":
            return fail(f"standing up before walking: {r['reason']}")
        return None

    def explore(self, direction: str, distance: float):
        g = self._walk_guard()
        if g is not None:
            return g
        lim = self.cfg["robot"]["limits"]
        lin = {"left": (0, 1), "right": (0, -1), "back": (-1, 0), "forward": (1, 0)}
        if direction in lin:
            if not 0 < distance <= lim["max_move_m"]:
                return fail(f"distance must be in (0, {lim['max_move_m']}] m")
            ax = lin[direction]
            # refuse a move that ends closer to a fitted box face than the stance search allows (2026-11-06: the LLM
            # explored 0.5 m forward from 0.67 m and walked the torso into the box)
            T = self._odom()
            p_new = T[:2, 3] + T[:2, :2] @ np.array([ax[0] * distance, ax[1] * distance])
            dmin = self._c2()["stance"]["min_face_dist_m"]
            for bid, b in self.ws.boxes.items():
                if b.face is None:
                    continue
                n = np.asarray(b.face["normal"], float)[:2]; n /= np.linalg.norm(n)
                d_now = float((T[:2, 3] - np.asarray(b.face["center"], float)[:2]) @ n)
                d_new = float((p_new - np.asarray(b.face["center"], float)[:2]) @ n)
                if d_now > 0 and d_new < dmin and d_new < d_now:
                    return fail(f"move refused: it would end {d_new:.2f} m from the front face of {bid} (minimum {dmin} m)")
            r = self.robot.move_root_relative(ax[0] * distance, ax[1] * distance, 0.0, self._c1()["approach"]["move_timeout_s"])
        elif direction in ("turn_left", "turn_right"):
            if not 0 < distance <= lim["max_turn_deg"]:
                return fail(f"turn must be in (0, {lim['max_turn_deg']}] deg")
            r = self.robot.move_root_relative(0.0, 0.0, math.radians(distance) * (1 if direction == "turn_left" else -1),
                                              self._c1()["approach"]["move_timeout_s"])
        else:
            return fail(f"unknown direction {direction!r}")
        if np.linalg.norm(self._odom()[:2, 3] - self.episode_start_odom[:2, 3]) > lim["workspace_radius_m"]:
            return fail("left the workspace")
        self.ws.explore_steps += 1
        data = {"move_status": r["status"], "explore_steps": self.ws.explore_steps}
        return ok(data, r.get("reason", "")) if r["status"] in ("ok", "timeout") else fail(r.get("reason") or f"move {r['status']}", data)

    def approach(self, target: str = "root_target"):
        """Segmented approach: walk min(remaining * 0.5, 1 m), re-perceive, correct the target;
        remaining < 0.3 m: one last move; then the arrival check with up to 2 fine moves. Every move reply is logged
        (move_summary). data["walk_done"] marks a result whose walk ended and only the arrival check decided.
        target="stance_target": a successful arrival marks the planned stance as reached (move_arm requires it); any
        walk to it counts as an attempt of the current barcode (next_barcode may then close it)."""
        g = self._walk_guard()
        if g is not None:
            return g
        if target == "stance_target" and self.ws.stance_target is not None:
            r = self._approach(target)
            if self.ws.current_barcode_id is not None:
                self._stance_attempted.add(self.ws.current_barcode_id)
            if r["status"] == "ok":
                self._stance_reached = self.ws.stance_target
            return r
        return self._approach(target)

    def _approach(self, target):
        A = self._c1()["approach"]
        tgt = self.ws.root_target if target == "root_target" else self.ws.stance_target
        if tgt is None:
            return fail(f"no {target} (compute it first)")
        segments, log = 0, []

        def biased(t):
            """Aim final_bias_m short of the target along its facing direction (finish with forward corrections)."""
            b = A["final_bias_m"] if target == "root_target" else 0.0
            return [t["pos"][0] - b * math.cos(t["yaw"]), t["pos"][1] - b * math.sin(t["yaw"])]

        while segments < A["max_segments"]:
            dx, dy, dyaw = self._rel(biased(tgt), tgt["yaw"])
            dist = math.hypot(dx, dy)
            if dist < A["final_below_m"]:
                r = self.robot.move_root_relative(dx, dy, dyaw, A["move_timeout_s"], speed_scale=A["precise_speed_scale"])
                log.append({"final": [dx, dy, dyaw], "status": r["status"], "move": move_summary(r)})
                break
            step = min(dist * A["segment_frac"], A["max_segment_m"])
            r = self.robot.move_root_relative(step * dx / dist, step * dy / dist, dyaw * step / dist, A["move_timeout_s"])
            segments += 1
            log.append({"segment": [step * dx / dist, step * dy / dist, dyaw * step / dist], "status": r["status"], "move": move_summary(r)})
            if r["status"] in ("fell", "collision"):
                return fail(f"move: {r['status']} ({r.get('reason')})", {"log": log})
            if target == "root_target":                                # re-perceive and correct the target
                rr = self._reperceive_target()
                if rr["status"] != "ok":
                    return fail(f"target lost during the approach: {rr['reason']}", {"log": log})
                tgt = self.ws.root_target
        if target != "root_target":
            dx, dy, dyaw = self._rel(tgt["pos"], tgt["yaw"])
            data = {"remaining": [dx, dy, math.degrees(dyaw)], "log": log}
            bh = tgt.get("base_height")
            if bh is not None and abs(bh - self.base_height_cmd) > 1e-6:
                # the stance includes its body height (plan_stance): arriving at the stance = walking there standing,
                # then squatting to it
                r = self.set_base_height(bh)
                if r["status"] != "ok":
                    return fail(f"squat to the stance height {bh}: {r['reason']}", data)
                data["base_height"] = r["data"]
            return ok(data, f"at the stance, body height command {self.base_height_cmd:.2f} m")
        return self.arrive_check(log)

    def _reperceive_target(self):
        for attempt in range(2):
            self.detect_boxes()
            if self.ws.target_box_id is None:
                if attempt == 0:
                    A = self._c1()["approach"]
                    self.robot.move_root_relative(-A["lost_backoff_m"], 0, 0, A["move_timeout_s"])
                    continue
                return fail("target box not re-identified after backing off")
            r = self.fit_front_face(self.ws.target_box_id)
            if r["status"] == "fail":                 # "all": partial = some faces fitted, the union uses those
                return r
            if self._spec_view_mode():
                # obs_pose mode: the pose is planned afresh from the spec region, which grows with every face seen
                # (box ids are renumbered by every detect_boxes, so frozen faces cannot be matched to refits by name).
                # The region only grows and is bounded by the wall, and d is searched nearest-first: the target
                # converges (bounded by approach.max_segments and arrive.max_fine_tunes). A pose that does not fit at
                # any d (partial) keeps the upper bound as the target, as a first plan does.
                r = self.compute_obs_pose(self.ws.target_box_id)
                return r if r["status"] == "fail" else dict(r, status="ok")
            return self._retarget_from_refit()
        return fail("target lost")

    def _retarget_from_refit(self):
        """Keep the observation target planned by compute_obs_pose (its d_obs and lateral position, decided on the full
        faces) and move it with the refitted face planes: the mean along-normal shift of the frozen faces that were
        refitted, and the refitted mean normal. Odometry drift is corrected; truncated close-range extents are ignored."""
        rt, frozen = self.ws.root_target, getattr(self, "_obs_faces", None)
        refit = {b: self._face(b) for b in self._box_ids(rt["box_id"])} if rt is not None else {}
        common = [b for b in (frozen or {}) if refit.get(b) is not None]
        if not common:                                 # nothing to anchor on: plan afresh from what is visible
            return self.compute_obs_pose(self.ws.target_box_id) | {"status": "ok"}
        nhs = [np.array([refit[b]["normal"][0], refit[b]["normal"][1], 0.0]) for b in common]
        nh = np.sum([v / np.linalg.norm(v) for v in nhs], axis=0); nh /= np.linalg.norm(nh)
        shift = float(np.mean([(np.asarray(refit[b]["center"], float) - frozen[b]["center"]) @ nh for b in common]))
        for f in frozen.values():
            f["center"] = f["center"] + shift * nh
            f["corners"] = f["corners"] + shift * nh
        for b in common:
            frozen[b]["normal"] = np.asarray(refit[b]["normal"], float)
        lat = np.array([-nh[1], nh[0], 0.0])
        a = max(float(f["center"] @ nh) for f in frozen.values())
        s_old = float(np.array([rt["pos"][0], rt["pos"][1], 0.0]) @ lat)
        p = s_old * lat + (a + rt["d_obs"]) * nh
        yaw = math.atan2(-nh[1], -nh[0])
        self.ws.root_target = dict(rt, pos=[float(p[0]), float(p[1])], yaw=yaw)
        self._marker("root_target", "red", "root", [p[0], p[1], 0.0], yaw=yaw)
        return ok({"root_target_odom": self.ws.root_target, "plane_shift_m": shift, "anchored_on": common})

    def _arrival_error(self, tgt):
        """Robot pose vs the root target, in the target's frame: along = along the facing direction (toward the face,
        i.e. the distance error along the face normal), lateral = along the target's left axis, dyaw."""
        T = self._odom()
        e = T[:2, 3] - np.asarray(tgt["pos"], float)
        c, s_ = math.cos(tgt["yaw"]), math.sin(tgt["yaw"])
        return float(e @ [c, s_]), float(e @ [-s_, c]), wrap(yaw_of(T) - tgt["yaw"])

    def arrive_check(self, log=None):
        """Arrival = |distance error along the face normal| < dist_tol and |dyaw| < yaw_tol and the face(s) in view.
        The lateral offset is not tested: the task (and evaluator.py) only needs the distance to the face, the yaw and
        the face in view, and the lateral offset matters only through the in-view test. Fine moves therefore correct
        the along-normal distance and the yaw, and the lateral offset only while the faces are not in view."""
        A = self._c1()["arrive"]
        AP = self._c1()["approach"]
        log = [] if log is None else log
        for k in range(A["max_fine_tunes"] + 1):
            rr = self._reperceive_target()
            if rr["status"] != "ok":
                return fail(f"arrival check: {rr['reason']}", {"log": log, "walk_done": True})
            tgt = self.ws.root_target
            along, lateral, dyaw = self._arrival_error(tgt)
            view = self.check_face_in_view()
            in_view = view["status"] == "ok"
            err = {"along_normal_m": along, "lateral_m": lateral, "dyaw_deg": math.degrees(dyaw)}
            if abs(along) < A["dist_tol_m"] and abs(math.degrees(dyaw)) < A["yaw_tol_deg"] and in_view:
                return ok({"error": err, "fine_tunes": k, "d_obs": tgt["d_obs"], "face_in_view": True, "log": log, "walk_done": True})
            if k < A["max_fine_tunes"]:
                c, s_ = math.cos(tgt["yaw"]), math.sin(tgt["yaw"])
                keep = lateral if in_view else 0.0           # keep the current lateral position while the faces are in view
                goal = [tgt["pos"][0] - keep * s_, tgt["pos"][1] + keep * c]
                dx, dy, dyaw_cmd = self._rel(goal, tgt["yaw"])
                # the executing controller must stop tighter than this check, else it settles at the check's boundary
                r = self.robot.move_root_relative(dx, dy, dyaw_cmd, AP["move_timeout_s"],
                                                  pos_tol=A["dist_tol_m"] * A["fine_tol_frac"], yaw_tol=math.radians(A["yaw_tol_deg"]) * A["fine_tol_frac"],
                                                  speed_scale=AP["precise_speed_scale"])
                log.append({"fine": [dx, dy, dyaw_cmd], "error_before": err, "lateral_corrected": not in_view, "move": move_summary(r)})
        data = {"error": err, "face_in_view": view["status"], "log": log, "walk_done": True}
        if self.task == 3:     # the scan phase plans its own stances; phase 1 is over once the walk has finished
            return partial("walk finished; arrival check not within tolerance. Phase 1 is over: call plan_scans next", data)
        return fail("not within tolerance after the fine moves", data)

    # ================================================================== barcode / wrist tools
    def _current(self, barcode_id=None):
        """The barcode a tool works on: the explicit id, else the current barcode (next_barcode / detect_barcode_head)."""
        return self.ws.barcodes.get(barcode_id or self.ws.current_barcode_id)

    def _not_current(self, barcode_id) -> str | None:
        """Task 3: the barcode tools work on the current barcode only (the plan's order is next_barcode's). 2026-11-07
        stack4c_L0_0001: the LLM went on to bc8 without next_barcode, planned its stance while bc7 was still current,
        and looped 14 times on "no stance planned for bc7" until the context was full."""
        cur = self.ws.current_barcode_id
        if self.task != 3 or barcode_id is None or barcode_id == cur:
            return None
        if cur is None:
            return f"no current barcode: call next_barcode (it gives the next barcode of the plan)"
        return f"{barcode_id} is not the current barcode ({cur}): call next_barcode to move on to the next barcode of the plan"

    def _label_on_face(self, frame, rec, face):
        """Label detection (head px, detect_label_dino) -> centre, bar direction, corners (odom) and width on the face
        plane (the accepted cv2 symbol quad, else the label box: rec "corners"), and the label box's width
        (label_width: rec "label_corners", the label's extent for the wrist consistency gate)."""
        c = self._pixel_on_face(frame, rec["center"], face)
        if c is None:
            return None
        p2 = self._pixel_on_face(frame, rec["center"] + 10 * np.asarray(rec["bar_dir"]), face)
        bar = None if p2 is None else (p2 - c) / (np.linalg.norm(p2 - c) + 1e-12)
        on_face = lambda Q: (lambda P: None if any(q is None for q in P) else np.asarray(P))(
            [self._pixel_on_face(frame, q, face) for q in np.asarray(Q, float)])
        corners, label = on_face(rec["corners"]), on_face(rec["label_corners"])
        return {"center": c, "bar_dir": bar, "corners": corners, "width": None if corners is None else quad_width(corners),
                "label_width": None if label is None else quad_width(label)}

    def detect_barcode_head(self, box_id: str | None = None):
        if self.ws.scan_plan:
            # it would make an unplanned barcode the current one and take it out of the plan's order / ledger
            return fail("a scan plan exists: barcodes come from plan_scans and next_barcode, not from detect_barcode_head",
                        {"current_barcode_id": self.ws.current_barcode_id})
        box_id = box_id or self.ws.target_box_id
        face = self._face(box_id)
        if face is None:
            return fail("no front face for the target (call fit_front_face)")
        h = self._capture_head()
        uv, z = self._project_odom(h, face["corners"])
        if np.any(z <= 0):
            return fail("face behind the camera")
        x0, y0 = uv.min(axis=0); x1, y1 = uv.max(axis=0)
        labels = detect_label_dino(h["rgb"], [x0, y0, x1, y1], self.det)
        import cv2
        poly = uv.astype(np.float32)
        hull = cv2.convexHull(poly)
        labels = [l for l in labels if cv2.pointPolygonTest(hull, (float(l["center"][0]), float(l["center"][1])), False) >= 0]
        if not labels:
            return fail("no barcode found on the target face in the head image")
        l = labels[0]
        bid = f"bc_{box_id}"
        self.ws.barcodes[bid] = BarcodeEstimate(bid, box_id, None, None, None, "head_px", h["t"], pixel=l["center"].tolist(),
                                                face=dict(face), corners_px=np.asarray(l["corners"], float).tolist())
        self.ws.current_barcode_id = bid
        self._last_label = {"id": bid, "rec": l, "frame": h}
        return ok({"barcode_id": bid, "center_px": l["center"], "corners_px": l["corners"], "kind": l["kind"],
                   "score": l.get("score"), "refined_by_cv2": bool(l.get("refined_by_cv2"))})

    def _pixel_on_face(self, frame, px, face):
        """Ray through pixel px of a captured frame intersected with the face plane (odom)."""
        T_odom_cam = frame["T_odom_root"] @ frame["T_root_cam"]
        d = rotate(T_odom_cam, geo.pixel_ray(frame["K"], px[0], px[1]))
        return geo.ray_plane(T_odom_cam[:3, 3], d, face["center"], face["normal"])

    def estimate_barcode_pose(self, barcode_id: str | None = None):
        barcode_id = barcode_id or (self._last_label or {}).get("id")
        if barcode_id not in self.ws.barcodes or self._last_label is None or self._last_label["id"] != barcode_id:
            return fail("no barcode detection (call detect_barcode_head)")
        bc = self.ws.barcodes[barcode_id]
        face = bc.face
        est = self._label_on_face(self._last_label["frame"], self._last_label["rec"], face)
        if est is None:
            return fail("barcode ray (nearly) parallel to the face plane")
        bc.center, bc.normal, bc.bar_dir, bc.source = est["center"], np.asarray(face["normal"], float), est["bar_dir"], "head"
        bc.corners_odom, bc.width_m, bc.label_width_m = est["corners"], est["width"], est["label_width"]
        return ok({"barcode_id": barcode_id, "center_odom": est["center"], "normal_odom": bc.normal, "bar_dir_odom": est["bar_dir"],
                   "width_m": est["width"]})

    # ------------------------------------------------------------------ task 3: scan plan
    def plan_scans(self, _update: bool | None = None):
        """Localise every label on every fitted face in one head image and fix the scan order (scan_order.py). Once the
        plan is being executed, a call from another viewpoint UPDATES it instead: labels already in the plan (same
        place on the face) are kept as they are, labels seen for the first time are added as pending and ordered with
        the other pending ones by the same rule in the planning frame; scanned / failed entries never move.
        With a wall layout (set_wall_layout): the wall is registered and one entry per requested box planned
        (_plan_layout); once that plan exists a call changes nothing (no capture)."""
        ws = self.ws
        if self.wall_layout is not None and ws.wall is not None:
            return ok({"updated": True, "added": []}, "the cells are fixed by the wall layout")
        # an internal update looks from the view it was called in (a squat / lean of the extra looks, the stance)
        r = self.detect_boxes(view="keep" if _update else "auto")
        if r["status"] != "ok":
            return fail("no box detected in the head image")
        h = self.head
        boxes = [{"id": k, "bbox": b.bbox} for k, b in ws.boxes.items()]
        faces, skipped = {}, {}
        for bid in ws.boxes:
            rf = self._fit_one(bid)
            if rf["status"] == "ok":
                faces[bid] = ws.boxes[bid].face
            else:
                skipped[bid] = f"face not fitted: {rf['reason']}"
        # stack faces: the first plan fixes the front; every plan works on the current stack face only (faces of
        # another side seen obliquely are not planned from here: their labels are planned when that side is visited)
        if faces and ws.stack is None:
            self._stack_init(list(faces.values()))
        cur = 0 if ws.stack is None else ws.stack["current"]
        for bid in list(faces):
            fid = self._face_id(faces[bid]["normal"])
            if fid != cur:
                skipped[bid] = f"face of stack side {self.TOUR[fid]} (current side {self.TOUR[cur]})"
                del faces[bid]
        if ws.stack is not None:
            self._stack_add_faces(cur, list(faces.values()))
        if self.wall_layout is not None:
            return self._plan_layout(h, faces, skipped)
        face_entries = [b for b in ws.scan_plan if ws.barcodes[b].face_id == cur]
        updating = _update if _update is not None else bool(face_entries)
        if self.scan_targets is not None:
            if ws.target_grid is None:
                ws.target_grid = {"targets": self.scan_targets, "edges_seen": [], "face_centres": [], "fitted_faces": [],
                                  "faces": {}}
            self._note_face_edges(h, list(faces.values()))
        per_face, dropped = [], []
        cam = (h["T_odom_root"] @ h["T_root_cam"])[:3, 3]
        for bid, face in faces.items():
            uv, z = self._project_odom(h, face["corners"])
            if np.any(z <= 0):
                skipped[bid] = "face behind the camera"
                continue
            kept = []                                   # detector output is sorted by score: the first kept is the best
            for l, est in self._labels_on_rectified_face(h, face):
                c = {"box_id": bid, "face": face, "rec": l, "est": est, "range": float(np.linalg.norm(est["center"] - cam))}
                dup = next((k for k in kept if _same_est(est, k["est"], face["normal"])), None)
                if dup is not None:                     # a second detection of a label already kept on this face
                    dropped.append((c, dup, "same label on this face (higher score kept)"))
                    continue
                kept.append(c)
            per_face += kept
        # the same label found from two overlapping face regions: it lies on the surface nearest to the camera along
        # its pixel ray (the visible one). One duplicate rule everywhere (_same_face_label: the label rectangles overlap
        # on the face and lie on one surface), also between the candidates of one update before they are merged
        cands = []
        for c in sorted(per_face, key=lambda c: c["range"]):
            dup = next((k for k in cands if _same_est(c["est"], k["est"], k["face"]["normal"])), None)
            if dup is None:
                cands.append(c)
            else:
                dropped.append((c, dup, "same label from another face region (nearest kept)"))
        if updating:
            return self._merge_into_plan(cands, h, boxes, skipped, dropped)
        multi = ws.stack is not None and ws.stack.get("multi_face")
        if not cands and not (multi and faces):
            return with_diagnostics(fail("no barcode label localised on any fitted face",
                                         {"boxes": boxes, "n_faces": len(faces), "skipped": skipped}),
                                    {"dropped_labels": _dropped_report(dropped)})
        # multi-face tour with this side's faces fitted but no label on them: a valid outcome (not every side of a stack
        # carries labels); the side is planned empty (after the low look below, which may still find low labels) and
        # the tour goes on with next_barcode
        items = order_items_in_frame([{"key": i, "corners": c["est"]["corners"], "center": c["est"]["center"]}
                                      for i, c in enumerate(cands)], h["T_odom_root"])
        order = order_items(items, self._order_rule())
        for i in order:                                    # appended after the plans of earlier faces (face-major)
            c, est = cands[i], cands[i]["est"]
            k = len(ws.barcodes)
            while f"bc{k}" in ws.barcodes:
                k += 1
            bid = f"bc{k}"
            ws.barcodes[bid] = BarcodeEstimate(bid, c["box_id"], est["center"], np.asarray(c["face"]["normal"], float), est["bar_dir"],
                                               "head", h["t"], pixel=np.asarray(c["rec"]["center"], float).tolist(),
                                               face=dict(c["face"]), corners_odom=est["corners"],
                                               corners_px=np.asarray(c["rec"]["corners"], float).tolist(), width_m=est["width"],
                                               label_width_m=_label_width(est), detection=_detection_info(c["rec"]),
                                               status="pending", order_index=len(ws.scan_plan),
                                               plan_center=np.asarray(est["center"], float).copy(), face_id=cur)
            c["bid"] = bid
            ws.scan_plan.append(bid)
        ws.scan_ref_T = np.asarray(h["T_odom_root"], float).copy()
        ws.scan_ref_Ts[cur] = ws.scan_ref_T
        if ws.stack is not None:
            ws.stack.setdefault("planned", []).append(cur)
            if ws.root_target is not None:
                ws.stack["faces"][cur]["d_obs"] = float(ws.root_target["d_obs"])
        ws.current_barcode_id = None
        self._clear_barcode_slots()
        low = self._low_view_update()
        low_diag = None if low is None else low.pop("diagnostics")
        targets = self._resolve_targets(cur) if self.scan_targets is not None else None
        side = [b for b in ws.scan_plan if ws.barcodes[b].face_id == cur]
        out = [{"id": b, "box_id": ws.barcodes[b].box_id, "order_index": ws.barcodes[b].order_index,
                "corners_px": ws.barcodes[b].corners_px, "center_odom": ws.barcodes[b].center,
                "width_m": ws.barcodes[b].width_m, "standoff_m": self._standoff_or_none(ws.barcodes[b]), "status": ws.barcodes[b].status,
                "face_id": cur, **({"cell": ws.barcodes[b].grid_cell} if targets is not None else {})}
               for b in side]
        n = sum(o["status"] == "pending" for o in out)
        data = {"n": n, "barcodes": out, "boxes": boxes, "skipped": skipped,
                "low_view": low, "face_id": cur, "face": self.TOUR[cur], "first_for_face": True,
                "order_rule": scan_order_text(self._order_rule())}
        diag = {"plan_entries": [_entry_diagnostics(ws.barcodes[b]) for b in side],
                "dropped_labels": _dropped_report(dropped), "low_view_looks": low_diag,
                **({"target_grid": self._targets_diagnostics(cur)} if targets is not None else {})}
        if targets is not None:
            data["targets"] = targets
            return with_diagnostics(ok(data, f"{n} barcode(s) planned on the {self.TOUR[cur]} side for "
                                             f"{targets['description']}{targets['note']}; call next_barcode"), diag)
        return with_diagnostics(ok(data,
                  f"{n} barcode(s) planned on the {self.TOUR[cur]} side; call next_barcode" if out else
                  f"no barcode label on the {self.TOUR[cur]} side ({len(faces)} face(s) fitted"
                  f"{', also looked from a squat' if low and low.get('height_cmd') is not None else ''}): this side "
                  f"has nothing to scan; call next_barcode to move on"), diag)

    # ------------------------------------------------------------------ task 3: wall layout (approach A)
    # The task gives the wall layout (columns x rows of equal boxes of box_w x box_h standing on the floor; user decision
    # 2026-11-19). The robot registers the wall once from the observation pose (front plane from the box face fits, both
    # side edges from the head depth, floor = odom z 0 cross-checked against the depth floor) and plans one entry of kind
    # "cell" per requested box in the lattice order of the task's rule. At each box: a stance for the hand-camera view of
    # the whole box face (compute_wrist_target / plan_stance), search_cell (re-register the wall from the head, aim the
    # D405 at the face, find the label inside it), then the g1_v2 final-pose path. Head label detection is not used.
    # Stored layout geometry lives in the WALL frame (WorldState.wall T_odom_wall: x = right along the wall, y = into the
    # wall, z = up, origin = the bottom-left corner of the front face as seen by the robot); odometry is trusted only
    # between two registrations (user decision 2026-11-19: real-robot odometry is unreliable).
    LAYOUT_KEYS = ("columns", "rows", "box_w", "box_h", "base")
    LAYOUT_BASES = ("floor",)        # row 1 stands on the floor, at odom z 0 (the D3 rule of _grid_evidence)

    def set_wall_layout(self, layout):
        """Scenario field wall_layout (task instruction): {"columns", "rows" (integers >= 1), "box_w", "box_h" (m, > 0),
        "base": "floor"}. Needs the target spec (set_scan_targets) and the obs_pose phase 1 (set_phase1_view) first; a
        multi-face task is refused, and so is a requested column / row beyond the layout."""
        if not isinstance(layout, dict):
            raise ValueError(f"wall_layout must be a dict, got {type(layout).__name__}")
        missing = [k for k in self.LAYOUT_KEYS if k not in layout]
        unknown = sorted(set(layout) - set(self.LAYOUT_KEYS))
        if missing or unknown:
            raise ValueError(f"wall_layout keys: missing {missing}, unknown {unknown} (known: {list(self.LAYOUT_KEYS)})")
        for k in ("columns", "rows"):
            v = layout[k]
            if not isinstance(v, int) or isinstance(v, bool) or v < 1:
                raise ValueError(f"wall_layout {k} must be an integer >= 1, got {v!r}")
        for k in ("box_w", "box_h"):
            v = layout[k]
            if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) or v <= 0:
                raise ValueError(f"wall_layout {k} must be a positive number (m), got {v!r}")
        if layout["base"] not in self.LAYOUT_BASES:
            raise ValueError(f"wall_layout base must be one of {self.LAYOUT_BASES}, got {layout['base']!r}")
        if self.scan_targets is None:
            raise ValueError("a wall layout needs the task's target spec (set_scan_targets first)")
        if self.phase1_view != "obs_pose":
            raise ValueError("a wall layout is registered from the observation pose: phase1_view must be 'obs_pose' "
                             "(set_phase1_view first)")
        if self.multi_face:
            raise ValueError("a wall layout describes one wall: a task over every side of a stack cannot use it")
        for dim in ("columns", "rows"):
            d = self.scan_targets[dim]
            if d is not None and d.get("index") and max(d["index"]) > layout[dim]:
                raise ValueError(f"the target spec asks for {dim} {d['index']} from the {d['from']}, the wall layout has "
                                 f"{layout[dim]} {dim}")
        self.wall_layout = {"columns": int(layout["columns"]), "rows": int(layout["rows"]),
                            "box_w": float(layout["box_w"]), "box_h": float(layout["box_h"]), "base": layout["base"]}
        self.ws.wall = None
        self._layout_edges = {}
        self._look_cache = {}

    # ---- wall frame
    @staticmethod
    def _wall_frame(origin, right, normal) -> np.ndarray:
        """T_odom_wall from the front face's bottom-left corner, its right axis and its normal (toward the robot)."""
        T = np.eye(4)
        T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = right, -np.asarray(normal, float), [0.0, 0.0, 1.0], origin
        return T

    def _wall_T(self) -> np.ndarray:
        return np.asarray(self.ws.wall["T_odom_wall"], float)

    def _wall_axes(self) -> dict:
        T = self._wall_T()
        return {"origin": T[:3, 3].copy(), "right": T[:3, 0].copy(), "up": T[:3, 2].copy(), "normal": -T[:3, 1]}

    def _rect_face(self, x0, x1, z0, z1) -> dict:
        """A rectangle of the wall's front plane (wall-frame x in [x0, x1], z in [z0, z1]) as a face dict (odom, current
        registration)."""
        T = self._wall_T()
        C = apply(T, np.array([[x0, 0.0, z1], [x1, 0.0, z1], [x1, 0.0, z0], [x0, 0.0, z0]]))
        ax = self._wall_axes()
        return {"center": C.mean(axis=0), "normal": ax["normal"], "right": ax["right"], "up": ax["up"], "corners": C,
                "size": [float(x1 - x0), float(z1 - z0)], "t": self.ws.wall["t"], "rect_w": [x0, x1, z0, z1]}

    def _cell_rect(self, c, r, grow_lat=0.0, grow_v=0.0) -> list:
        """Wall-frame rectangle [x0, x1, z0, z1] of layout cell (column c from the left, row r from the bottom, 1-based),
        grown by grow_lat laterally and grow_v vertically."""
        L = self.wall_layout
        return [(c - 1) * L["box_w"] - grow_lat, c * L["box_w"] + grow_lat,
                (r - 1) * L["box_h"] - grow_v, r * L["box_h"] + grow_v]

    def _cell_face(self, c, r) -> dict:
        return dict(self._rect_face(*self._cell_rect(c, r)), layout_cell=[int(c), int(r)])

    def _search_grow(self):
        """Growth of a box face for the hand-camera search: delta_lat (the registration's lateral uncertainty) laterally
        and e = robot.yaml arm_motion.correction_tol (the placement error the arm loop accepts) on every side."""
        return float(self.ws.wall["delta_lat"]), float(self.cfg["robot"]["arm_motion"]["correction_tol"])

    def _sync_layout(self, bc):
        """Odometry fields of a layout entry from its wall-frame geometry (current registration)."""
        T, w = self._wall_T(), bc.wall
        bc.face = self._cell_face(*w["cell"])
        bc.center = apply(T, w["center"])
        bc.normal = rotate(T, w["normal"])
        bc.corners_odom = None if w["corners"] is None else apply(T, w["corners"])
        bc.bar_dir = None if w["bar_dir"] is None else rotate(T, w["bar_dir"])

    def _store_layout(self, bc):
        """Wall-frame geometry of a layout entry from its odometry fields (a measurement made now)."""
        Ti = inv_T(self._wall_T())
        bc.wall.update(center=apply(Ti, bc.center), normal=rotate(Ti, bc.normal),
                       corners=None if bc.corners_odom is None else apply(Ti, bc.corners_odom),
                       bar_dir=None if bc.bar_dir is None else rotate(Ti, bc.bar_dir))

    def _sync_layout_all(self):
        for bc in self.ws.barcodes.values():
            if bc.wall is not None:
                self._sync_layout(bc)

    # ---- registration
    @staticmethod
    def _wall_plane(faces) -> dict:
        """Front plane of the wall from the box face fits: the mean horizontal normal (the layout boxes stand on the
        floor, their fronts are vertical) and the median of the face centres along it (a few bad fits do not move it).
        right = up x normal (the robot's right, as plane_fit)."""
        nhs = []
        for f in faces:
            n = np.asarray(f["normal"], float)
            nhs.append(np.array([n[0], n[1], 0.0]) / np.linalg.norm(n[:2]))
        nh = np.sum(nhs, axis=0)
        nh /= np.linalg.norm(nh)
        off = float(np.median([float(np.asarray(f["center"], float) @ nh) for f in faces]))
        return {"normal": nh, "right": np.cross([0.0, 0.0, 1.0], nh), "point": off * nh, "offset": off,
                "n_faces": len(faces)}

    def _wall_extent(self, h, plane) -> dict:
        """Side edges and lateral extent of the wall on a plane in head frame h (plane_fit.wall_extent)."""
        return wall_extent(h["depth"], h["K"], h["T_root_cam"], h["T_odom_root"], plane, self.cfg["perception"]["plane_fit"],
                           sigma_coeff=h["depth_sigma_coeff"], min_structure_px=self._min_structure_px(),
                           fov_margin_frac=self.cfg["perception"]["fov_margin_frac"])

    def _floor_height(self, h, base_pt, plane) -> dict | None:
        """Cross-check of the odometry floor: the support plane (plane_fit.fit_support_plane, the plane-fit parameters)
        of the head depth points IN FRONT of the wall plane (signed distance >= the point's plane tolerance: the floor
        between the robot and the wall; with the wall in the sample, ~70 % of the points at the observation pose, the
        RANSAC draws almost never hit three floor points), its height at base_pt (odom x, y). None when no support
        plane has min_inliers."""
        c = self.cfg["perception"]["plane_fit"]
        z = np.asarray(h["depth"], float)
        vs, us = np.mgrid[0:z.shape[0]:c["stride"], 0:z.shape[1]:c["stride"]]
        zz = z[vs, us]
        lo, hi = c["depth_valid"]
        m = np.isfinite(zz) & (zz > lo) & (zz < hi)
        if m.sum() < c["min_inliers"]:
            return None
        P = apply(h["T_odom_root"] @ h["T_root_cam"], geo.backproject(h["K"], us[m], vs[m], zz[m]))
        tol = np.maximum(c["inlier_dist"], c["inlier_sigmas"] * h["depth_sigma_coeff"] * zz[m] ** 2)
        front = (P - np.asarray(plane["point"], float)) @ np.asarray(plane["normal"], float) >= tol
        P, tol = P[front], tol[front]
        if len(P) < c["min_inliers"]:
            return None
        sp = fit_support_plane(P, tol, c, np.random.default_rng(c["seed"]))
        if sp["mask"] is None or sp["count"] < c["min_inliers"]:
            return None
        a, n = sp["point"], sp["normal"]
        x, y = float(base_pt[0]), float(base_pt[1])
        return {"z": float(a[2] - (n[0] * (x - a[0]) + n[1] * (y - a[1])) / n[2]), "n_inliers": int(sp["count"])}

    def _layout_view_points(self, faces):
        """Observation-pose region of a wall-layout task (the _spec_view_points contract): on the front plane
        (_wall_plane), laterally the wall's place as far as the edges seen in phase 1 tell it (self._layout_edges, odom
        points: phase 1 walks ~1 m, ~2 cm drift): [s_L, s_L + W] with the left edge seen, [s_R - W, s_R] with the right
        one, else [s_max - W, s_min + W] (every placement consistent with the extent seen; the face-fit extents only
        without any wall component in view; a seen extent wider than W fits no placement: [s_min, s_max]); vertically the bottom row, [0, box_h] above the odom floor (in an upright
        head view the wall's lateral coverage is widest at its lowest points, and the floor line is in view for the
        floor cross-check)."""
        L = self.wall_layout
        W = L["columns"] * L["box_w"]
        plane = self._wall_plane(faces)
        center, _ = self._faces_reference(faces)
        p0, right = plane["point"], plane["right"]
        mem = self._layout_edges if self._layout_edges is not None else {}
        self._layout_edges = mem
        ext = self._wall_extent(self.head, plane)
        for side in ("left", "right"):
            if ext[side] is not None:
                mem[side] = p0 + ext[side]["s"] * right
        if ext["seen"] is not None:
            mem["seen"] = np.vstack([mem.get("seen", np.zeros((0, 3))), [p0 + v * right for v in ext["seen"]]])
        s_of = lambda P: np.atleast_2d(np.asarray(P, float) - p0) @ right
        if "left" in mem:
            lo = float(s_of(mem["left"])[0])
            hi = lo + W
        elif "right" in mem:
            hi = float(s_of(mem["right"])[0])
            lo = hi - W
        else:
            S = s_of(mem["seen"]) if "seen" in mem else s_of(np.vstack([np.asarray(f["corners"], float) for f in faces]))
            if float(S.max() - S.min()) > W:               # more than the layout's width seen (no placement of the wall
                lo, hi = float(S.min()), float(S.max())    # holds it: the seen extent is too wide): hold all of it
            else:
                lo, hi = float(S.max()) - W, float(S.min()) + W
        z_lo, z_hi = 0.0, L["box_h"]
        up = np.array([0.0, 0.0, 1.0])
        P = lambda sv, zv: p0 + sv * right + zv * up
        corners = np.array([P(hi, z_hi), P(lo, z_hi), P(lo, z_lo), P(hi, z_lo)])
        ref = P(0.5 * (lo + hi), float(np.asarray(center, float)[2]))
        return corners, {"layout": True, "lateral": [lo, hi], "z_range": [z_lo, z_hi],
                         "edges_seen": {sd: sd in mem for sd in ("left", "right")}, "seen": ext["seen"]}, ref

    def _register_wall(self, h, faces):
        """Initial registration of the wall frame from head frame h (wall-layout plan_scans): plane (_wall_plane), both
        side edges (_wall_extent; both are required: no unverified one-edge anchoring), the lateral anchor a = (s_L +
        s_R) / 2 (a symmetric stereo fattening of the edges cancels in a and shows in the width residual r = s_R - s_L -
        W), the floor cross-check. Halfway rules: |r| >= box_w / 2 (a wrong column count or a coplanar neighbour gives
        |r| >= box_w) and a depth floor >= box_h / 2 from the odom floor (row indices off by one) are failures. Returns
        (None (registered: ws.wall) or the tool failure, the log-only diagnostics)."""
        L = self.wall_layout
        W, H = L["columns"] * L["box_w"], L["rows"] * L["box_h"]
        plane = self._wall_plane(faces)
        ext = self._wall_extent(h, plane)
        diag = {"plane": plane, "extent": ext}
        missing = [sd for sd in ("left", "right") if ext[sd] is None]
        if missing:
            def why(sd):
                rj = ext["rejected"][sd]
                if ext["seen"] is None:
                    return "no wall surface found on the plane"
                return "cut by the image border" if rj["cut"] >= rj["front"] else "occluded"
            what = " and the ".join(f"{sd} side edge ({why(sd)})" for sd in missing)
            return with_diagnostics(fail(f"the {what} of the wall {'is' if len(missing) == 1 else 'are'} not in the head view: "
                                         "call compute_obs_pose and approach again (they plan the view from the edges "
                                         "seen so far), then plan_scans",
                                         {"edges": {sd: ext[sd] for sd in ("left", "right")}, "seen": ext["seen"]}), diag), diag
        sL, sR = ext["left"]["s"], ext["right"]["s"]
        W_meas = sR - sL
        r = W_meas - W
        if abs(r) >= L["box_w"] / 2:
            return with_diagnostics(fail(f"the wall seen is {W_meas:.2f} m wide, the layout says {W:.2f} m: the wall does "
                                         "not match the layout", {"W_meas": W_meas, "W": W}), diag), diag
        right, nh = plane["right"], plane["normal"]
        origin = plane["point"] + (0.5 * (sL + sR) - W / 2) * right
        zf = self._floor_height(h, origin + (W / 2) * right, plane)
        diag["floor"] = zf
        if zf is not None and abs(zf["z"]) >= L["box_h"] / 2:
            return with_diagnostics(fail(f"the floor seen is {zf['z']:.2f} m from the odometry floor: row indices would be "
                                         "off by one", {"z_floor_depth": zf["z"]}), diag), diag
        self.ws.wall = {"T_odom_wall": self._wall_frame(origin, right, nh), "W": W, "H": H, "columns": L["columns"],
                        "rows": L["rows"], "box_w": L["box_w"], "box_h": L["box_h"], "delta_lat": abs(r) / 2,
                        "t": h["t"], "anchor_root_xy": np.asarray(h["T_odom_root"], float)[:2, 3].copy(),
                        "registration": {"s_L": sL, "s_R": sR, "W_meas": W_meas, "r": r, "mad_L": ext["left"]["mad"],
                                         "mad_R": ext["right"]["mad"], "n_rows_L": ext["left"]["n_rows"],
                                         "n_rows_R": ext["right"]["n_rows"],
                                         "z_floor_depth": None if zf is None else zf["z"],
                                         "z_floor_note": None if zf is not None else
                                         "no support plane in the head depth: the odometry floor is used unchecked",
                                         "plane_offset": plane["offset"], "n_faces": plane["n_faces"], "t": h["t"]},
                        "history": []}
        return None, diag

    def _requested_cells(self) -> list:
        """Requested layout cells [(c, r)] (column from the left, row from the bottom, 1-based): "all" / an absent
        dimension -> every index; *_from_left / *_from_bottom i -> i; *_from_right / *_from_top i -> n + 1 - i."""
        t, L = self.scan_targets, self.wall_layout

        def idx(dim, n):
            d = t[dim]
            if d is None or d.get("all"):
                return list(range(1, n + 1))
            return sorted(i if d["from"] in ("left", "bottom") else n + 1 - i for i in d["index"])
        return [(c, r) for c in idx("columns", L["columns"]) for r in idx("rows", L["rows"])]

    def _spec_cell(self, c, r) -> dict:
        """Layout cell -> the target-grid cell in the spec's counting ({"column", "row"}; None for a dimension the spec
        does not name; "all" counts from the left / top, as parse_targets)."""
        t, L = self.scan_targets, self.wall_layout

        def one(dim, k, n):
            d = t[dim]
            if d is None:
                return None
            return k if d["from"] in ("left", "bottom") else n + 1 - k
        return {"column": one("columns", c, L["columns"]), "row": one("rows", r, L["rows"])}

    def _plan_layout(self, h, faces, skipped):
        """plan_scans of a wall-layout task: register the wall (_register_wall), then one entry of kind "cell" per
        requested box, ordered by the lattice itself (scan_order.order_lattice: the task's rule on the cells, no label
        extents). No head label detection, no extra looks, no grid resolution: the grid is the layout."""
        ws = self.ws
        boxes = [{"id": k, "bbox": b.bbox} for k, b in ws.boxes.items()]
        if not faces:
            return fail("no box face fitted from here", {"boxes": boxes, "skipped": skipped})
        failed, reg_diag = self._register_wall(h, list(faces.values()))
        if failed is not None:
            return failed
        L, wall = self.wall_layout, ws.wall
        rule = self._order_rule()
        cells = self._requested_cells()
        order = order_lattice([{"key": f"c{c}_r{r}", "column": c, "row": L["rows"] + 1 - r} for c, r in cells], rule)
        by_key = {f"c{c}_r{r}": (c, r) for c, r in cells}
        T_ref = np.asarray(h["T_odom_root"], float)
        for o in order:
            c, r = by_key[o["key"]]
            k = len(ws.barcodes)
            while f"bc{k}" in ws.barcodes:
                k += 1
            bid = f"bc{k}"
            face = self._cell_face(c, r)
            bc = BarcodeEstimate(bid, f"layout_c{c}_r{r}", face["center"].copy(), face["normal"].copy(), None, "layout",
                                 h["t"], face=face, corners_odom=face["corners"].copy(), status="pending",
                                 order_index=len(ws.scan_plan), plan_center=face["center"].copy(), face_id=0,
                                 grid_cell=self._spec_cell(c, r), kind="cell")
            bc.wall = {"cell": [c, r], "left_to_right": bool(o["left_to_right"]), "corners": None, "bar_dir": None}
            self._store_layout(bc)
            ws.barcodes[bid] = bc
            ws.scan_plan.append(bid)
        ws.scan_ref_T = T_ref.copy()
        ws.scan_ref_Ts[0] = ws.scan_ref_T
        if ws.stack is not None:
            ws.stack.setdefault("planned", []).append(0)
            if ws.root_target is not None and 0 in ws.stack["faces"]:
                ws.stack["faces"][0]["d_obs"] = float(ws.root_target["d_obs"])
        ws.current_barcode_id = None
        self._clear_barcode_slots()
        # target grid of the layout (WorldState.missing_cells / grid_warnings and the Harness work unchanged): slots in
        # the planning frame, every requested cell present by construction
        T_fo = inv_T(T_ref)
        slots = {}
        for dim, n in (("columns", L["columns"]), ("rows", L["rows"])):
            out = []
            for k in range(1, n + 1):
                rect = self._cell_rect(k, 1) if dim == "columns" else self._cell_rect(1, k)
                if dim == "columns":
                    rect = [rect[0], rect[1], 0.0, wall["H"]]
                else:
                    rect = [0.0, wall["W"], rect[2], rect[3]]
                C = apply(T_fo, self._rect_face(*rect)["corners"])
                a = C[:, 1] if dim == "columns" else C[:, 2]
                sc = self._spec_cell(k, 1)["column"] if dim == "columns" else self._spec_cell(1, k)["row"]
                out.append({"index": k if sc is None else sc, "lo": float(a.min()), "hi": float(a.max()),
                            "center": float(a.mean()), "box": True, "inserted": False})
            slots[dim] = sorted(out, key=lambda sl: sl["index"])
        ws.target_grid = {"targets": self.scan_targets, "edges_seen": [], "face_centres": [], "fitted_faces": [],
                          "faces": {0: {"columns": slots["columns"], "rows": slots["rows"],
                                        "requested": [self._spec_cell(c, r) for c, r in cells], "missing": [],
                                        "anchored": {"left": True, "right": True, "bottom": True}, "warnings": [],
                                        "notes": [f"the cells come from the given wall layout ({L['columns']} x {L['rows']} "
                                                  f"boxes of {L['box_w']} x {L['box_h']} m) registered to the wall seen "
                                                  "from here"],
                                        "evidence": "layout", "bounds": {"bottom": 0.0}}}}
        targets = self._targets_report(0)
        reg = wall["registration"]
        out = [{"id": b, "kind": ws.barcodes[b].kind, "cell": ws.barcodes[b].grid_cell,
                "center_odom": ws.barcodes[b].center, "status": ws.barcodes[b].status, "face_id": 0} for b in ws.scan_plan
               if ws.barcodes[b].wall is not None]
        data = {"n": len(out), "barcodes": out, "boxes": boxes, "skipped": skipped, "face_id": 0, "face": self.TOUR[0],
                "first_for_face": True, "order_rule": scan_order_text(rule), "targets": targets,
                "registration": {"W_meas": reg["W_meas"], "r": reg["r"],
                                 "edges": [{"side": "left", "s": reg["s_L"], "n_rows": reg["n_rows_L"]},
                                           {"side": "right", "s": reg["s_R"], "n_rows": reg["n_rows_R"]}],
                                 "z_floor_depth": reg["z_floor_depth"]}}
        diag = {"registration": dict(reg, delta_lat=wall["delta_lat"], T_odom_wall=wall["T_odom_wall"]), **reg_diag}
        return with_diagnostics(ok(data, f"{len(out)} box(es) planned from the wall layout for {targets['description']}; "
                                         "call next_barcode"), diag)

    # ---- hand-camera look pose
    def _hand_view(self):
        """HandView (perception/look_view.py) of the variant's hand camera: its K and size from the robot info, its pose
        on the scanner tip and the occluder boxes (robot info hand_cameras[name]["occluders"], link frames) through the
        URDF. Built once: the camera is rigid on the bracket."""
        if getattr(self, "_hand_view_model", None) is None:
            import pinocchio as pin
            from ..perception.look_view import HandView
            hc = self.cfg["robot"]["hand_camera"]
            if hc["source"] != "hand_camera":
                raise ValueError("the wall-layout look pose needs a separate hand camera (robot variant hand_camera "
                                 "source 'hand_camera')")
            cam = (self.info.get("hand_cameras") or {}).get(hc["name"])
            if cam is None or "occluders" not in cam:
                raise ValueError(f"robot info hand_cameras[{hc['name']!r}] has no 'occluders': the wall-layout look pose "
                                 "needs the robot's parts in front of the hand camera")
            q = pin.neutral(self.kin.model)
            T_cam = self.kin.pose(q, hc["frame"])
            T_tip = self.kin.pose(q, self.cfg["robot"]["wrist_point_frame"])
            occ = [dict(o, T_cam_box=inv_T(T_cam) @ self.kin.pose(q, o["link"])) for o in cam["occluders"]]
            self._hand_view_model = HandView(cam["K"], cam["size"], inv_T(T_tip) @ T_cam, self.ik.img_x_in_tip, occ,
                                             self.cfg["perception"]["fov_margin_frac"])
        return self._hand_view_model

    def _search_view(self, face_w, face_h, delta, e) -> dict:
        """look_view.search_view for this episode (cached per face size, delta and e): stand-off from the scanner's
        standoff.min_m upward in steps of e (the placement tolerance), up to the farthest viewing distance of the task
        (d_obs_range upper end; beyond it is a geometry error)."""
        if self._look_cache is None:
            self._look_cache = {}
        key = (float(face_w), float(face_h), float(delta), float(e))
        if key not in self._look_cache:
            from ..perception.look_view import search_view
            self._look_cache[key] = search_view(self._hand_view(), face_w, face_h, delta, e, e,
                                                float(self._c2()["standoff"]["min_m"]), float(self._c1()["d_obs_range"][1]))
        return self._look_cache[key]

    def _look_target(self, bc) -> dict:
        """Wrist target of the hand-camera view of the whole box face of a cell entry (current registration): the
        minimum tip stand-off s with the face grown by delta_lat laterally and e all round in the free view (e: the
        placement slack; search_cell verifies the face grown by delta_lat only from the pose reached), one
        alternative per quarter roll that reaches it (their tie is decided by plan_stance: reachability, manipulability).
        long_axis = L_k (look_view.roll_axis), signed: the side of the scan axis the camera sits on matters."""
        from ..perception.look_view import roll_axis
        L = self.wall_layout
        dl, e = self._search_grow()
        sv = self._search_view(L["box_w"], L["box_h"], dl, e)
        f = self._cell_face(*bc.wall["cell"])
        ctr, n, right, up = (np.asarray(f[k], float) for k in ("center", "normal", "right", "up"))
        alts = [{"pos": ctr + cd["h"] * right + cd["v"] * up + sv["s"] * n, "long_axis": roll_axis(cd["roll_deg"], right, up),
                 "roll_signed": True, "roll_deg": cd["roll_deg"], "h": cd["h"], "v": cd["v"]} for cd in sv["candidates"]]
        return {"purpose": "look", "barcode_id": bc.id, "offset": float(sv["s"]), "forward": -n, "bar_dir": None,
                "alternatives": alts, "pos": alts[0]["pos"], "long_axis": alts[0]["long_axis"], "roll_signed": True,
                "face_size": [L["box_w"], L["box_h"]], "delta_lat": dl, "grow_e": e, "cell": list(bc.wall["cell"])}

    @staticmethod
    def _roll_kw(t) -> dict:
        """IK keyword for a target whose roll axis is directed (a look pose); none for every other target, so their
        IK calls are the calls they were."""
        return {"roll_signed": True} if t.get("roll_signed") else {}

    # ---- stance re-registration and the search
    def _reregister_at_stance(self) -> dict:
        """Re-register the wall frame from the head at a stance (arm home): fit_front_face on the bbox of the projected
        wall rectangle; the frame is rotated about the vertical axis through the CURRENT ROOT by the yaw between the
        stored and the refitted normal and shifted along the new normal onto the refit (the plane gives two of the three
        planar degrees of freedom; the lateral one stays as odometry predicts it, and the sim's heading drift acts on the
        root). A side edge in view (_wall_extent) re-anchors the lateral position (both edges: their middle). Every
        record is appended to ws.wall["history"]; a failed refit changes nothing."""
        wall = self.ws.wall
        h = self._capture_head()
        rec = {"t": h["t"], "refit": False}
        C = self._rect_face(0.0, wall["W"], 0.0, wall["H"])["corners"]
        uv, z = self._project_odom(h, C)
        if np.any(z <= 0):
            rec["note"] = "a wall corner lies behind the head camera: no refit"
            wall["history"].append(rec)
            return rec
        Hh, Ww = h["depth"].shape
        x0, y0 = np.clip(uv.min(axis=0), 0, [Ww, Hh])
        x1, y1 = np.clip(uv.max(axis=0), 0, [Ww, Hh])
        r = fit_front_face(h["depth"], h["K"], h["T_root_cam"], [x0, y0, x1, y1], sigma_coeff=h["depth_sigma_coeff"],
                           min_structure_px=self._min_structure_px())
        if r["status"] != "ok":
            rec["note"] = f"refit failed: {r['reason']}"
            wall["history"].append(rec)
            return rec
        To = np.asarray(h["T_odom_root"], float)
        c_new = apply(To, r["center"])
        n_new = rotate(To, r["normal"])
        n_new = np.array([n_new[0], n_new[1], 0.0]) / np.linalg.norm(n_new[:2])
        ax = self._wall_axes()
        n_old = ax["normal"]
        dpsi = math.atan2(n_old[0] * n_new[1] - n_old[1] * n_new[0], float(n_old[:2] @ n_new[:2]))
        cz, sz = math.cos(dpsi), math.sin(dpsi)
        Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
        root = To[:3, 3]
        origin = ax["origin"].copy()
        origin[:2] = root[:2] + Rz[:2, :2] @ (origin[:2] - root[:2])
        right, normal = Rz @ ax["right"], Rz @ n_old
        dd = float((c_new - origin) @ normal)
        origin = origin + dd * normal
        ext = self._wall_extent(h, {"normal": normal, "right": right, "point": origin})
        el, er = ext["left"], ext["right"]
        ds = None
        if el is not None and er is not None:
            ds = 0.5 * (el["s"] + er["s"]) - wall["W"] / 2
        elif el is not None:
            ds = el["s"]
        elif er is not None:
            ds = er["s"] - wall["W"]
        net = float((root[:2] - np.asarray(wall["anchor_root_xy"], float)) @ right[:2])
        if ds is not None:
            origin = origin + ds * right
            wall["anchor_root_xy"] = root[:2].copy()
        wall["T_odom_wall"] = self._wall_frame(origin, right, normal)
        rec.update(refit=True, dpsi_deg=math.degrees(dpsi), dd_m=dd, ds_m=ds,
                   anchored=[sd for sd, e_ in (("left", el), ("right", er)) if e_ is not None],
                   net_lateral_since_anchor_m=net)
        wall["history"].append(rec)
        self._sync_layout_all()
        return rec

    def _view_coverage(self, w, corners_odom) -> dict:
        """The face (4 corners, odom) in the hand camera's free view from the pose of frame w (FK of the joints
        reached): look_view.HandView.covered; plus the fraction of a sample grid of the face inside the free view (log)."""
        hv = self._hand_view()
        C = np.asarray(corners_odom, float)
        uv, z = self._project_odom(w, C)
        cov = bool(hv.covered(uv[None], z[None])[0])
        a = np.linspace(0.0, 1.0, 21)
        A, B = np.meshgrid(a, a)
        G = C[0] + A.ravel()[:, None] * (C[1] - C[0]) + B.ravel()[:, None] * (C[3] - C[0])
        guv, gz = self._project_odom(w, G)
        u, v = np.round(guv[:, 0]).astype(int), np.round(guv[:, 1]).astype(int)
        inside = (gz > 0) & (u >= 0) & (u < hv.W) & (v >= 0) & (v < hv.H)
        frac = float(np.mean(inside & hv.free[np.clip(v, 0, hv.H - 1), np.clip(u, 0, hv.W - 1)]))
        return {"covered": cov, "free_frac": frac}

    def _cell_detections(self, w, rect_w) -> list:
        """Labels on a wall rectangle (wall-frame [x0, x1, z0, z1]) in hand-camera frame w: the rectified detector
        (_labels_on_rectified_face) on that crop; each with its label rectangle and centre in the wall frame and "cut":
        its centre is the centre of the visible part only, because its rectified label box lies within one rectified
        pixel of the crop border (pixel quantisation of the crop; "cut_crop") or its label box, grown by one hand-camera
        pixel, leaves the hand camera's free view (image border, in-view margin, the scanner body: HandView.whole;
        "hidden")."""
        Ti = inv_T(self._wall_T())
        x0, x1, z0, z1 = rect_w
        hv = self._hand_view()
        out = []
        for rec, est in self._labels_on_rectified_face(w, self._rect_face(*rect_w)):
            px = float(rec["rectified"]["px_per_m"])
            Cw = apply(Ti, est["corners"])
            tol = 1.0 / px
            cut_crop = bool(Cw[:, 0].min() <= x0 + tol or Cw[:, 0].max() >= x1 - tol
                            or Cw[:, 2].min() <= z0 + tol or Cw[:, 2].max() >= z1 - tol)
            uv, zc = self._project_odom(w, est["corners"])
            hidden = not hv.whole(uv, zc)
            out.append({"rec": rec, "est": est, "center_w": apply(Ti, est["center"]), "px_per_m": px,
                        "rect_w": [float(Cw[:, 0].min()), float(Cw[:, 0].max()), float(Cw[:, 2].min()), float(Cw[:, 2].max())],
                        "cut": cut_crop or hidden, "cut_crop": cut_crop, "hidden": hidden})
        return out

    def _look_view(self, t, see_w, crop_w) -> dict:
        """Move the scanner tip to look target t (IK + closed-loop correction, _reach), capture the hand camera, test the
        rectangle see_w (wall frame: the region the view has to show) for coverage from the pose reached and, when
        covered, detect labels on the crop crop_w (see_w grown by the placement tolerance e the look pose was planned
        with: a label of the box reaching the face edge stays clear of the crop border; what of it lies outside the
        free view is caught by the cut test)."""
        rr = self._reach(t, float(t["offset"]), self.side)
        if rr["ik_fail"]:
            return {"ik_fail": True, "reason": rr["reason"]}
        w = self._capture_wrist()
        cov = self._view_coverage(w, self._rect_face(*see_w)["corners"])
        dets = self._cell_detections(w, crop_w) if cov["covered"] else []
        return {"ik_fail": False, "covered": cov["covered"], "coverage": cov, "dets": dets, "frame": w,
                "residual": rr["history"][-1] if rr["history"] else None, "see_w": list(see_w), "rect_w": list(crop_w)}

    def _search_round(self, bc, st) -> dict:
        """One search of the cell from the current registration: the look pose of the stance's alternative (its roll and
        in-plane offset, on the re-registered cell), coverage of the cell grown by delta_lat (the face that has to be
        seen: the look pose was planned for it grown by e more on every side, so an arm placement error up to e =
        correction_tol, what _reach accepts, keeps it covered) and detection on the cell grown by delta_lat + e, one
        extra view (same roll and stand-off) centred on the visible part of each cut detection that overlaps the cell,
        then the assignment: uncut detections whose centre lies in the (ungrown) cell rectangle, duplicates across views
        merged (_same_face_label). outcome: found | none (covered, nothing assigned, nothing unresolved) | uncovered | cut
        (a cut detection not seen whole in its own centred view) | ik."""
        t = self._look_target(bc)
        alt = t["alternatives"][int(st.get("look_alt") or 0)]
        ta = dict(t, pos=alt["pos"], long_axis=alt["long_axis"])
        self.ws.wrist_target = ta
        dl, e = self._search_grow()
        c, r = bc.wall["cell"]
        cell = self._cell_rect(c, r)
        see = self._cell_rect(c, r, dl, 0.0)
        grown = self._cell_rect(c, r, dl + e, e)
        v = self._look_view(ta, see, grown)
        views = [v]
        if v["ik_fail"]:
            return {"outcome": "ik", "reason": v["reason"], "views": views}
        if not v["covered"]:
            return {"outcome": "uncovered", "views": views}
        ax = self._wall_axes()
        cx, cz = 0.5 * (cell[0] + cell[1]), 0.5 * (cell[2] + cell[3])
        overlaps = lambda a: a[0] < cell[1] and cell[0] < a[1] and a[2] < cell[3] and cell[2] < a[3]
        unresolved = []
        for d in [d for d in v["dets"] if d["cut"] and overlaps(d["rect_w"])]:
            dx, dz = float(d["center_w"][0] - cx), float(d["center_w"][2] - cz)
            t2 = dict(ta, pos=np.asarray(ta["pos"], float) + dx * ax["right"] + dz * ax["up"])
            shift = lambda a: [a[0] + dx, a[1] + dx, a[2] + dz, a[3] + dz]
            v2 = self._look_view(t2, shift(see), shift(grown))
            v2["extra_for"] = d["center_w"]
            views.append(v2)
            whole = [] if v2["ik_fail"] or not v2["covered"] else \
                [d2 for d2 in v2["dets"] if not d2["cut"] and _same_face_label(d2["est"]["corners"], d["est"]["corners"],
                                                                                ax["normal"], _label_width(d2["est"]))]
            if not whole:
                unresolved.append(d)
        assigned = []
        for vv in views:
            for d in ([] if vv["ik_fail"] or not vv["covered"] else vv["dets"]):
                x, z = float(d["center_w"][0]), float(d["center_w"][2])
                if d["cut"] or not (cell[0] <= x <= cell[1] and cell[2] <= z <= cell[3]):
                    continue
                if any(_same_face_label(d["est"]["corners"], a["est"]["corners"], ax["normal"],
                                        max(_label_width(d["est"]), _label_width(a["est"]))) for a in assigned):
                    continue
                d["frame"] = vv["frame"]
                assigned.append(d)
        outcome = "found" if assigned else "cut" if unresolved else "none"
        return {"outcome": outcome, "views": views, "assigned": assigned, "unresolved": unresolved}

    def _set_label(self, b, d):
        """A layout entry becomes the label of detection d (search_cell): kind label, centre / corners = the label box,
        width = the stand-off width (cv2 symbol quad, else the label box): an upper bound of the symbol ("hand_label",
        the _standoff clamp keeps the scanner at least at offset_m), bar direction on the plane."""
        est, rec, w = d["est"], d["rec"], d["frame"]
        face = self._cell_face(*b.wall["cell"])
        b.kind, b.source, b.t = "label", "hand", w["t"]
        b.center = np.asarray(est["center"], float).copy()
        b.plan_center = b.center.copy()
        b.corners_odom = np.asarray(est["corners"], float).copy()
        b.normal = np.asarray(face["normal"], float).copy()
        b.bar_dir = None if est["bar_dir"] is None else np.asarray(est["bar_dir"], float).copy()
        b.label_width_m = quad_width(est["corners"])
        b.width_m, b.width_source, b.width_support = float(est["width"]), "hand_label", 0
        b.face = face
        b.pixel = np.asarray(rec["center"], float).tolist()
        b.corners_px = np.asarray(rec["corners"], float).tolist()
        b.detection = _detection_info(rec)
        self._store_layout(b)

    def _record_failed_stance(self, bc, st):
        self._failed_stances.add((bc.id, tuple(st["target_pos"]), st["face_dist"], st["lateral_offset"], st["level_index"]))

    def search_cell(self, barcode_id: str | None = None):
        """Wall-layout task: find the label of the current box (an entry of kind "cell") with the hand camera, at the
        stance plan_stance planned for its look pose (reached), arm home.
        1. Re-register the wall from the head (_reregister_at_stance).
        2. Look pose of the stance's roll alternative on the re-registered cell; IK + closed loop (_reach).
        3. Coverage from the pose reached (FK): the cell grown by delta_lat in the hand camera's free view (the look pose
           was planned for it grown by e = correction_tol more: the placement slack of the arm loop).
        4-6. Detection on the cell grown by delta_lat + e, cut detections, assignment (_search_round).
        Nothing found with the face covered: the arm goes home, the wall is re-registered once more and the cell
        searched again (user decision 2026-11-19); still nothing -> "no_label" (a result: the whole face was seen and
        holds no label centre). A label found: the entry becomes that label (kind label, width_source hand_label), further
        labels of the box become new entries right after it, and its wrist target is computed. Not covered / a cut label
        not resolved: partial (never no_label), the stance recorded as failed so that a re-plan picks another one."""
        if self.wall_layout is None or self.ws.wall is None:
            return fail("search_cell is for a task with a wall layout, after plan_scans registered the wall")
        if (nc := self._not_current(barcode_id)) is not None:
            return fail(nc)
        bc = self._current(barcode_id)
        if bc is None:
            return fail("no current box: call next_barcode")
        if bc.kind != "cell":
            return fail(f"{bc.id} is a label entry (its label is found): compute_wrist_target -> plan_stance -> "
                        "approach -> move_arm('pre') -> refine_with_wrist ..., no search_cell")
        if bc.status != "pending":
            return fail(f"{bc.id} is already recorded {bc.status}: call next_barcode")
        st = self.ws.stance_target
        if st is None or st.get("barcode_id") != bc.id or st.get("purpose") != "look":
            return fail(f"no hand-camera view stance planned for {bc.id}: call compute_wrist_target, plan_stance, then "
                        "approach('stance_target')")
        if getattr(self, "_stance_reached", None) is not st:
            return fail("the robot is not at the planned stance: call approach('stance_target') first")
        if not self.arm_at_home:
            return fail("the arm must be at home (the wall is re-registered from the head): call move_arm('home')")
        self._stance_attempted.add(bc.id)
        rounds = []
        for k in range(2):
            if k == 1:                                    # the second search re-registers with the arm out of the view
                hm = self.move_arm("home")
                if hm["status"] != "ok":
                    return fail(f"arm home before the second search: {hm['reason']}")
            rr = self._reregister_at_stance()
            res = self._search_round(bc, st)
            res["reregistration"] = rr
            rounds.append(res)
            if res["outcome"] != "none":
                break
        last = rounds[-1]
        n_views = sum(len(rd["views"]) for rd in rounds)
        rereg = [rd["reregistration"] for rd in rounds]

        def view_log(v):
            if v["ik_fail"]:
                return {"ik_fail": v["reason"]}
            return {"covered": v["covered"], "coverage": v["coverage"], "residual": v["residual"], "see_w": v["see_w"],
                    "rect_w": v["rect_w"],
                    "detections": [{"center_w": d["center_w"], "rect_w": d["rect_w"], "cut": d["cut"],
                                    "cut_crop": d["cut_crop"], "hidden": d["hidden"],
                                    "score": d["rec"].get("score"), "kind": d["rec"].get("kind"), "px_per_m": d["px_per_m"]}
                                   for d in v["dets"]]}
        diag = {"rounds": [{"outcome": rd["outcome"], "reregistration": rd["reregistration"],
                            "views": [view_log(v) for v in rd["views"]]} for rd in rounds]}
        base = {"barcode_id": bc.id, "views": n_views, "reregistration": rereg}
        if last["outcome"] == "ik":
            self._record_failed_stance(bc, st)
            return with_diagnostics(fail("hand-camera view not reachable from this stance: move_arm('home'), "
                                         "compute_wrist_target, plan_stance, approach", dict(base, ik=last["reason"])), diag)
        if last["outcome"] == "found":
            added = self._convert_cell(bc, last["assigned"])
            tr = self.compute_wrist_target(bc.id)
            if tr["status"] != "ok":
                return with_diagnostics(fail(f"label found, wrist target: {tr['reason']}", base), diag)
            return with_diagnostics(ok(dict(base, converted=True, added=added,
                                            label={"center_odom": bc.center, "label_width_m": bc.label_width_m,
                                                   "cell": bc.grid_cell},
                                            wrist_target_odom=tr["data"]["wrist_target_odom"], offset=tr["data"]["offset"]),
                                       "label found in the box face; call move_arm('final')"
                                       + (f" ({len(added)} more label(s) in this box added as the next entries)" if added else "")),
                                    diag)
        if last["outcome"] == "none":
            dl, _ = self._search_grow()
            bc.status = "no_label"
            bc.fail_reason = (f"the whole box face (grown by {dl:.3f} m laterally) was in the hand-camera view and no "
                              f"label centre lies in it ({len(rounds)} searches)")
            return with_diagnostics(ok(dict(base, barcode_status="no_label"),
                                       "no label in this box: move_arm('home'), next_barcode"), diag)
        self._record_failed_stance(bc, st)
        why = ("the box face was not fully in the hand-camera view from the pose reached" if last["outcome"] == "uncovered"
               else "a label at the border of the box face was not seen whole, also not from a view centred on it")
        return with_diagnostics(partial(f"{why}: move_arm('home'), compute_wrist_target -> plan_stance -> approach -> "
                                        "search_cell again (once), then next_barcode", dict(base, outcome=last["outcome"])),
                                diag)

    def _convert_cell(self, bc, assigned) -> list:
        """The cell entry becomes the first of its labels in the within-cell order of the task's rule
        (scan_order.order_in_cell, in the row direction of the cell for "serpentine"); the others become new label
        entries right after it (same cell). Returns the added ids."""
        ws = self.ws
        items = [{"key": i, "y_lo": -d["rect_w"][1], "y_hi": -d["rect_w"][0], "z_lo": d["rect_w"][2],
                  "z_hi": d["rect_w"][3], "z": float(d["center_w"][2])} for i, d in enumerate(assigned)]
        order = order_in_cell(items, self._order_rule(), bc.wall["left_to_right"])
        self._set_label(bc, assigned[order[0]])
        added = []
        at = ws.scan_plan.index(bc.id)
        for j, i in enumerate(order[1:]):
            k = len(ws.barcodes)
            while f"bc{k}" in ws.barcodes:
                k += 1
            nb = BarcodeEstimate(f"bc{k}", bc.box_id, None, None, None, "hand", 0.0, status="pending", face_id=0,
                                 grid_cell=dict(bc.grid_cell), kind="label")
            nb.wall = {"cell": list(bc.wall["cell"]), "left_to_right": bc.wall["left_to_right"]}
            self._set_label(nb, assigned[i])
            ws.barcodes[nb.id] = nb
            ws.scan_plan.insert(at + 1 + j, nb.id)
            added.append(nb.id)
        for i, b in enumerate(ws.scan_plan):
            ws.barcodes[b].order_index = i
        return added

    def _detect_wrist_cell(self, bc):
        """refine_with_wrist detector of a layout entry: the search_cell detector on the cell crop grown by delta_lat + e
        (rectified; cut detections, i.e. cut by the crop border or not whole in the hand camera's free view (the image
        border, the in-view margin, the scanner body: _cell_detections), and centres outside the cell dropped); the one
        nearest to the current estimate. Returns (frame, rec, est, n_hidden) or (frame, None, None, n_hidden); n_hidden:
        the labels centred in the cell dropped only because part of them lies outside the free view."""
        w = self._capture_wrist()
        dl, e = self._search_grow()
        c, r = bc.wall["cell"]
        cell = self._cell_rect(c, r)
        keep, n_hidden = [], 0
        for d in self._cell_detections(w, self._cell_rect(c, r, dl + e, e)):
            x, z = float(d["center_w"][0]), float(d["center_w"][2])
            if not (cell[0] <= x <= cell[1] and cell[2] <= z <= cell[3]):
                continue
            if d["cut"]:
                n_hidden += int(d["hidden"] and not d["cut_crop"])
                continue
            keep.append(d)
        if not keep:
            return w, None, None, n_hidden
        d = min(keep, key=lambda d_: float(np.linalg.norm(np.asarray(d_["est"]["center"]) - np.asarray(bc.center))))
        return w, d["rec"], d["est"], n_hidden

    def _layout_refusal(self, bc, scanning: bool = False):
        """Tools of the label path refused for a box whose label is not found yet (the scanner would read a neighbour
        into the cell), and scan for a box recorded no_label."""
        if bc is None or self.wall_layout is None:
            return None
        if scanning and bc.status == "no_label":
            return fail(f"{bc.id} holds no label (search_cell recorded no_label): move_arm('home'), then next_barcode")
        if bc.kind == "cell":
            return fail(f"the label of {bc.id} is not found yet: call search_cell")
        return None

    # ------------------------------------------------------------------ task 3: target grid (robot-terms target spec)
    def _order_rule(self) -> str:
        return "columns" if self.scan_targets is None else self.scan_targets["order"]

    def _note_face_edges(self, h, faces):
        """Box-face evidence of this head image for the target grid (scan_order.resolve_grid): for each fitted face of
        the current side, which of its edges were seen (the two corners with the largest z = top, smallest z = bottom,
        leftmost / rightmost along the face, both projecting inside the image with the perception margin,
        perception.yaml fov_margin_frac, the margin every in-view test uses: a face cut by the image border was fitted up
        to that border, not up to its edge), stored on the face (face["edges_seen"], copied with every label found on it),
        whether a seen edge is clear (no detected box of this view has its centre beyond the edge's image line), the
        face centre (a box beyond a candidate wall edge, seen in any view) and the face itself (box evidence for the grid
        whether or not a label was localised on it)."""
        g = self.ws.target_grid
        if g is None:
            return
        H, W = h["rgb"].shape[:2]
        cur = 0 if self.ws.stack is None else self.ws.stack["current"]
        dets = [np.array([(b.bbox[0] + b.bbox[2]) / 2, (b.bbox[1] + b.bbox[3]) / 2]) for b in self.ws.boxes.values()]
        for f in faces:
            C = np.asarray(f["corners"], float)
            uv, z = self._project_odom(h, C)
            inside = geo.in_image(uv, z, W, H, self.cfg["perception"]["fov_margin_frac"])
            pc = self._project_odom(h, np.asarray(f["center"], float)[None])[0][0]
            lat = C @ np.asarray(f["right"], float)        # face right as seen from the camera side
            seen = {}
            for edge, idx in (("top", np.argsort(-C[:, 2])[:2]), ("bottom", np.argsort(C[:, 2])[:2]),
                              ("left", np.argsort(lat)[:2]), ("right", np.argsort(-lat)[:2])):
                seen[edge] = bool(np.all(inside[idx]))
                if not seen[edge]:
                    continue
                p1, p2 = uv[idx[0]], uv[idx[1]]
                side = lambda q: float((p2[0] - p1[0]) * (q[1] - p1[1]) - (p2[1] - p1[1]) * (q[0] - p1[0]))
                s_in = side(pc)
                clear = not any(side(q) * s_in < 0 for q in dets)       # a detection centre on the far side of the edge
                g["edges_seen"].append({"face_id": cur, "edge": edge, "pts": C[idx].copy(), "clear": clear})
            f["edges_seen"] = seen
            g["face_centres"].append({"face_id": cur, "center": np.asarray(f["center"], float).copy()})
            g["fitted_faces"].append({"face_id": cur, "face": {k: f[k] for k in ("t", "corners", "edges_seen")}})

    def _grid_evidence(self, fid, T_odom_frame):
        """Edge observations, fitted-face centres and the physical bounds of side fid in the planning frame (resolve_grid
        inputs). An edge's coordinate is the conservative end of its two corners (top: the lower one, left: the one
        further right, ...). bounds["bottom"]: the floor below the side, the planning-frame z of the odom point (mean
        face-centre x, y, 0): floor = odom z 0 (the simulator's odom z is the world pelvis z, g1_barcode_sim env.py,
        and stance planning already assumes ground z 0, tools/body.py; on the real robot this depends on the state
        estimator's height reference and must be confirmed). None when no face of the side was fitted.
        Also every face fitted on the side in a plan view (_face_item), with or without labels: box evidence."""
        T_fo = inv_T(np.asarray(T_odom_frame, float))
        tf = lambda P: (T_fo[:3, :3] @ np.atleast_2d(np.asarray(P, float)).T).T + T_fo[:3, 3]
        edges = []
        for e in self.ws.target_grid["edges_seen"]:
            if e["face_id"] != fid:
                continue
            P = tf(e["pts"])
            v = {"top": P[:, 2].min(), "bottom": P[:, 2].max(), "left": P[:, 1].min(), "right": P[:, 1].max()}[e["edge"]]
            edges.append({"edge": e["edge"], "coord": float(v), "clear": e["clear"]})
        odom_cs = [np.asarray(c["center"], float) for c in self.ws.target_grid["face_centres"] if c["face_id"] == fid]
        cs = [tf(c)[0] for c in odom_cs]
        bounds = None
        if odom_cs:
            m = np.mean(odom_cs, axis=0)
            bounds = {"bottom": float(tf([m[0], m[1], 0.0])[0][2])}
        faces = [_face_item(e["face"], T_fo) for e in self.ws.target_grid["fitted_faces"] if e["face_id"] == fid]
        return edges, {"columns": [float(c[1]) for c in cs], "rows": [float(c[2]) for c in cs]}, faces, bounds

    def _plan_items(self, bids, fid, with_faces: bool = False):
        """Order / grid items of plan entries in their side's planning frame (the planned label rectangles, never moved,
        and the current centres). with_faces: also the box face each label was found on (its extent in the planning
        frame and which of its edges were seen, _note_face_edges), for the target grid."""
        ws = self.ws
        T = ws.scan_ref_Ts.get(fid, ws.scan_ref_T)
        items = order_items_in_frame([{"key": b, "corners": ws.barcodes[b].corners_odom, "center": ws.barcodes[b].center}
                                      for b in bids], T)
        if with_faces:
            T_fo = inv_T(np.asarray(T, float))
            for it, b in zip(items, bids):
                f = ws.barcodes[b].face
                it["face"] = None if f is None or "edges_seen" not in f else _face_item(f, T_fo)
        return items

    def _reorder_pending(self, fid):
        """Pending entries of side fid (the current barcode excluded) re-ordered by the plan's rule in that side's
        planning frame; every other entry keeps its place. When the side's target grid is resolved and every pending
        entry has a cell, they are ordered by cell (scan_order.order_by_cells: the box rows / columns, not the label
        extents, so a label fragment or a duplicate inside a box row is never ordered as a row of its own), ties inside
        a cell by the label rule; otherwise by the label rule (scan_order.order_items)."""
        ws = self.ws
        pend = [b for b in ws.scan_plan if ws.barcodes[b].status == "pending" and b != ws.current_barcode_id
                and ws.barcodes[b].face_id == fid]
        done = [b for b in ws.scan_plan if b not in pend]
        grid = None if ws.target_grid is None else ws.target_grid["faces"].get(fid)
        cells = {b: ws.barcodes[b].grid_cell for b in pend}
        if any(c is None for c in cells.values()):     # an entry planned before the grid of its side: no cell order
            grid = None
        ws.scan_plan = done + list(order_by_cells(self._plan_items(pend, fid), cells, grid, self._order_rule()))
        for i, b in enumerate(ws.scan_plan):
            ws.barcodes[b].order_index = i

    def _resolve_targets(self, fid) -> dict:
        """First plan of a side with a target spec, after every extra look: the box grid of the side (scan_order.
        resolve_grid: label groups, rows / columns of boxes without a localised label inserted from the box faces, the
        wall edge only with nothing beyond it), the labels outside the requested cells marked not_targeted, the targeted
        ones ordered by the spec's rule. Every requested cell not found and every count that rests on the labels alone
        is reported (warnings; the Harness refuses a success claim with them): never silently fewer or shifted."""
        ws = self.ws
        ents = [b for b in ws.scan_plan if ws.barcodes[b].face_id == fid and ws.barcodes[b].status == "pending"]
        edges, centres, faces, bounds = self._grid_evidence(fid, ws.scan_ref_Ts.get(fid, ws.scan_ref_T))
        grid = resolve_grid(self._plan_items(ents, fid, with_faces=True), self.scan_targets, edges, centres, faces=faces,
                            bounds=bounds)
        for b in ents:
            bc = ws.barcodes[b]
            bc.grid_cell = grid["cell_of"][b]
            if b not in grid["selected"]:
                bc.status = "not_targeted"
                bc.fail_reason = f"not a target (column {bc.grid_cell['column']}, row {bc.grid_cell['row']})"
        ws.target_grid["faces"][fid] = {k: grid[k] for k in ("columns", "rows", "requested", "missing", "anchored",
                                                             "warnings", "notes", "evidence")}
        ws.target_grid["faces"][fid]["bounds"] = bounds
        self._reorder_pending(fid)                       # by cell: the grid of the side is stored first
        return self._targets_report(fid)

    def _targets_report(self, fid) -> dict:
        g = self.ws.target_grid["faces"][fid]
        t = self.scan_targets
        cell = lambda c: ", ".join(f"{a} {c[a]}" for a in ("column", "row") if c[a] is not None)
        note = ""
        if g["missing"]:
            note += (f". WARNING: only {len(g['requested']) - len(g['missing'])} of {len(g['requested'])} requested cells "
                     f"were found; NOT FOUND: {'; '.join(cell(c) for c in g['missing'])} (no label localised there)")
        for w in g["warnings"]:
            note += f". WARNING: {w}"
        for n in g["notes"]:
            note += f". Note: {n}"
        return {"description": describe_targets(t), "spec": t["spec"], "order": t["order"],
                "n_columns": len(g["columns"]), "n_rows": len(g["rows"]),
                "requested_cells": g["requested"], "missing_cells": g["missing"], "edges_seen": g["anchored"],
                "evidence": g["evidence"], "warnings": g["warnings"], "notes": g["notes"], "note": note}

    def _targets_diagnostics(self, fid) -> dict:
        """Slot table of side fid's target grid per dimension (index, extent, centre, box / label-only, inserted) and the
        physical bounds it used: logged with the plan (with_diagnostics), not shown to the LLM."""
        g = self.ws.target_grid["faces"][fid]
        return {"slots": {dim: [{"index": sl["index"], "lo": sl["lo"], "hi": sl["hi"], "center": sl["center"],
                                 "kind": "box" if sl["box"] else "label-only", "inserted": sl["inserted"]}
                                for sl in g[dim]] for dim in ("columns", "rows")},
                "bounds": g.get("bounds")}

    def _low_view_update(self):
        """Extra looks right after the first plan of a side (task 3, robot standing, arm home): the plan view does not
        resolve the whole side equally. Low part: the head camera looks ~50 deg down from ~1.2 m, so low labels on a
        wide / tall face are far and oblique (2026-11-07 3x3 stack: 1-2 bottom-row labels missed standing, found after
        squatting) -> every calibrated squat level (upright) whose view holds the faces' bottom corners more finely than
        the plan view (focal length / depth at the worst bottom corner). High part (D435i, 42 deg vertical view): every
        calibrated lean level whose view holds the faces' top corners more finely than the plan view. New labels are
        merged into the plan after each look (same rule as a plan update); the robot ends standing upright."""
        body = self._body()
        if body is None or not self._standing() or not self.arm_at_home \
                or not (self.ws.scan_plan or (self.ws.stack or {}).get("planned")):
            return None
        faces = [b.face for b in self.ws.boxes.values() if b.face is not None
                 and (self.ws.stack is None or self._face_id(b.face["normal"]) == self.ws.stack["current"])]
        if not faces:
            return None

        def extreme(top):
            out = []
            for f in faces:
                C = np.asarray(f["corners"], float)
                zz = C[:, 2].max() if top else C[:, 2].min()
                out += [c for c in C if abs(c[2] - zz) < 1e-6] or [C[np.argmax(C[:, 2]) if top else np.argmin(C[:, 2])]]
            return np.asarray(out)
        bottoms, tops = extreme(False), self._to_stack_top(extreme(True))
        tops = tops[tops[:, 2] >= tops[:, 2].max() - 1e-6]
        T0 = self._odom()
        q0 = self._q()
        yaw = yaw_of(T0)
        R_yaw = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
        K = np.asarray(self.info["head_K"], float)

        def finer(Tc, pts):
            inside, z = self._in_view(Tc, pts)
            return float(K[0, 0] / z.max()) if np.all(inside) else None   # px per metre at the farthest point

        def squat_cam(lv):
            q_lv = body.q_at(q0, lv)
            sh = R_yaw @ body.foot_shift(lv, body.levels[0], q0)
            return body.T_root(T0[0, 3] + sh[0], T0[1, 3] + sh[1], yaw, lv) @ self.kin.pose(q_lv, self.cfg["robot"]["head_cam_frame"])

        h_plan = self.head                                   # the frame the plan was made from
        T_plan = h_plan["T_odom_root"] @ h_plan["T_root_cam"]
        s_bot, s_top = finer(T_plan, bottoms), finer(T_plan, tops)
        # every level that sees the part more finely than the plan view. One "best" level is not enough: the
        # open-vocabulary label detector's recall is view-dependent and not monotonic in the height (2026-11-07 cardbox
        # stack, 11 levels x 2 sides: the deepest squat, finest on the bottom, found none of the 4 lower labels, the
        # 0.40-0.60 levels found all 4, 0.35 only 1); the union over the levels found every label of both sides.
        plan_pitch = self.view_pitch
        cur = 0 if self.ws.stack is None else self.ws.stack["current"]
        if self.scan_targets is not None and cur not in (self.ws.target_grid or {"faces": {}})["faces"]:
            # a target spec counts columns / rows from an edge of the wall: before the grid of this side is resolved,
            # look from EVERY standing head view (coverage, not resolution: the top row may lie above the plan view,
            # 2026-11-19 g1_v2 at 0.6 m from the scanroom wall sees z ~0.68-1.19 m upright, the top row is at 1.2 m)
            leans = [(finer(self._cam_T_at(T0[0, 3], T0[1, 3], yaw, lv), tops), lv) for lv in self._lean_levels()
                     if abs(lv["pitch_cmd"] - plan_pitch) > 1e-9]
        else:
            leans = [(sc, lv) for lv in self._lean_levels() if abs(lv["pitch_cmd"] - plan_pitch) > 1e-9
                     and (sc := finer(self._cam_T_at(T0[0, 3], T0[1, 3], yaw, lv), tops)) is not None
                     and (s_top is None or sc > s_top)]
        squats = [(sc, lv) for lv in body.levels[1:] if (sc := finer(squat_cam(lv), bottoms)) is not None
                  and (s_bot is None or sc > s_bot)]
        done, added, diags = [], [], []                 # diags: per look, logged by plan_scans (with_diagnostics)
        for kind, looks in (("lean", leans), ("squat", squats)):
            if kind == "squat" and looks and self._upright() is not None:      # squats start from upright standing
                done.append({"squat": None, "status": "failed: upright before squatting"})
                break
            for sc, lv in looks:
                r = self._set_view(lv) if kind == "lean" else self.set_base_height(lv["height_cmd"])
                if r["status"] != "ok":
                    done.append({kind: lv.get("pitch_cmd", lv.get("height_cmd")), "status": f"failed: {r['reason']}"})
                    break
                u = self.plan_scans(_update=True)
                new = [a["id"] for a in (u["data"].get("added") or [])] if u["status"] == "ok" else []
                added += new
                done.append({kind: lv["pitch_cmd"] if kind == "lean" else lv["height_cmd"],
                             "px_per_m": None if sc is None else round(sc, 1), "added": new})
                diags.append({kind: done[-1][kind], **u.get("diagnostics", {})})
        if done:
            st = self._upright() if self._standing() else None
            if not self._standing():
                r = self.set_base_height(None)
                st = None if r["status"] == "ok" else r
            if st is not None:
                raise RuntimeError(f"standing upright after the extra looks failed: {st['reason']}")
        return {"looks": done, "px_per_m_plan": {"bottom": None if s_bot is None else round(s_bot, 1),
                                                  "top": None if s_top is None else round(s_top, 1)},
                "height_cmd": [d["squat"] for d in done if "squat" in d] or None, "added": added,
                "diagnostics": diags}

    def _merge_into_plan(self, cands, h, boxes, skipped, dropped=None):
        """plan_scans update (see there). A detection is a label already in the plan when its label rectangle overlaps
        a planned label's rectangle on the face and lies on the same surface (_same_planned_label)."""
        ws = self.ws
        cur = 0 if ws.stack is None else ws.stack["current"]
        grid = None if ws.target_grid is None else ws.target_grid["faces"].get(cur)
        added = []
        dropped = [] if dropped is None else dropped
        for c in cands:
            est = c["est"]
            same = next((b for b in ws.barcodes.values() if b.face_id == cur and _same_planned_label(est, b)), None)
            if same is not None:
                c["bid"] = same.id
                dropped.append((c, same, "same label as a plan entry"))
                continue
            k = len(ws.barcodes)
            while f"bc{k}" in ws.barcodes:
                k += 1
            bid = f"bc{k}"
            ws.barcodes[bid] = BarcodeEstimate(bid, c["box_id"], est["center"], np.asarray(c["face"]["normal"], float), est["bar_dir"],
                                               "head", h["t"], pixel=np.asarray(c["rec"]["center"], float).tolist(),
                                               face=dict(c["face"]), corners_odom=est["corners"],
                                               corners_px=np.asarray(c["rec"]["corners"], float).tolist(), width_m=est["width"],
                                               label_width_m=_label_width(est), detection=_detection_info(c["rec"]),
                                               status="pending", plan_center=np.asarray(est["center"], float).copy(), face_id=cur)
            c["bid"] = bid
            ws.scan_plan.append(bid)
            added.append(bid)
            if grid is not None:
                # the side's grid is resolved: the new label belongs to the slot (box column / row, or the label group
                # where the boxes were not resolved) that holds it; one that no slot holds on the counted side of the
                # requested cells means the count missed something: reported, and no success claim (Harness)
                bc = ws.barcodes[bid]
                item = self._plan_items([bid], cur)[0]
                bc.grid_cell = grid_cell(item, grid)
                conflict = grid_conflict(item, bc.grid_cell, grid, self.scan_targets)
                if conflict is not None:
                    grid["warnings"].append(f"{bid}: {conflict} (found after the grid was resolved; the counts may be off)")
                if cell_targeted(bc.grid_cell, self.scan_targets):
                    grid["missing"] = [m for m in grid["missing"]
                                       if not all(m[a] is None or m[a] == bc.grid_cell[a] for a in ("column", "row"))]
                else:
                    bc.status = "not_targeted"
                    bc.fail_reason = (f"not a target (column {bc.grid_cell['column']}, row {bc.grid_cell['row']}; None = "
                                      "outside the grid resolved at the first plan)")
        if added:
            # pending entries of the current side (old and new) re-ordered by the plan's rule in that side's planning
            # frame; the rest keep their places
            self._reorder_pending(cur)
        out = [{"id": b, "box_id": ws.barcodes[b].box_id, "order_index": ws.barcodes[b].order_index,
                "corners_px": ws.barcodes[b].corners_px, "center_odom": ws.barcodes[b].center,
                "width_m": ws.barcodes[b].width_m, "status": ws.barcodes[b].status,
                **({"cell": ws.barcodes[b].grid_cell} if grid is not None else {})} for b in added]
        n_new = sum(o["status"] == "pending" for o in out)
        return with_diagnostics(
            ok({"updated": True, "added": out, "n_seen": len(cands), "n_planned": len(ws.scan_plan), "boxes": boxes,
                "skipped": skipped, "scan_plan": ws.scan_plan_export()},
               (f"plan update: {len(added)} new label(s) added" + (f" ({n_new} of them targets)" if grid is not None else ""))
               if added else "plan update: no new label"),
            {"added": [_entry_diagnostics(ws.barcodes[b]) for b in added], "dropped_labels": _dropped_report(dropped)})

    def _labels_on_rectified_face(self, h, face):
        """Labels on a fitted face, detected on the face rectified to a fronto-parallel view: the plane maps the head
        image to face coordinates (homography), sampled at the finest source sampling over the face (focal length /
        nearest face depth: no source detail dropped). Without it the steep head view (camera ~47 deg down) squashes
        labels on the lower part of a tall face into ~20 x 5 px slivers the detector misses (2026-11-06 floor stack:
        1 of 3 labels found from the observation pose, 3 of 3 rectified). Returns
        [(rec, est)]: rec = detection in head-image px (label box corners / centre projected back, the cv2 symbol quad
        as symbol_corners or None; overlays and diagnostics), est on the plane: centre / corners = the LABEL box (the
        detection box: the label's extent, barcode_detector.detect_label_dino), bar direction, and width = the stand-off
        width: the cv2 symbol quad's when one was accepted inside the label, else the label box's. The head label box is
        ~1.23x the code (paper 0.106 vs code 0.086 m, 2026-11-19 scanroom): with it the adaptive stand-off would sit at
        the max clamp; a symbol fragment cannot bring the scanner closer than offset_m (_standoff)."""
        import cv2
        c = np.asarray(face["center"], float)
        n = np.asarray(face["normal"], float); n /= np.linalg.norm(n)
        v = np.array([0.0, 0.0, 1.0]) - n[2] * n; v /= np.linalg.norm(v)
        u = np.cross(v, n)                              # image right when seen from the camera side
        rel = np.asarray(face["corners"], float) - c
        umin, umax, vmin, vmax = (rel @ u).min(), (rel @ u).max(), (rel @ v).min(), (rel @ v).max()
        T_cam_odom = inv_T(h["T_odom_root"] @ h["T_root_cam"])
        zc = (T_cam_odom[:3, :3] @ np.asarray(face["corners"], float).T).T[:, 2] + T_cam_odom[2, 3]
        K = np.asarray(h["K"], float)
        s = float(K[0, 0] / max(zc.min(), 1e-3))        # px per metre on the rectified face
        Wr, Hr = int(np.ceil((umax - umin) * s)), int(np.ceil((vmax - vmin) * s))
        P3 = np.array([c + umin * u + vmax * v, c + umax * u + vmax * v, c + umax * u + vmin * v, c + umin * u + vmin * v])
        src, _ = self._project_odom(h, P3)
        H = cv2.getPerspectiveTransform(np.array([[0, 0], [Wr, 0], [Wr, Hr], [0, Hr]], np.float32), src.astype(np.float32))
        rect = cv2.warpPerspective(h["rgb"], H, (Wr, Hr), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP)
        to3d = lambda xy: c + (umin + xy[0] / s) * u + (vmax - xy[1] / s) * v
        out = []
        for l in detect_label_dino(rect, [0, 0, Wr, Hr], self.det):
            corners = np.array([to3d(q) for q in np.asarray(l["label_corners"], float)])
            center = to3d(np.asarray(l["label_center"], float))
            sym = None if l["symbol_corners"] is None else np.array([to3d(q) for q in np.asarray(l["symbol_corners"], float)])
            bd = l.get("bar_dir")
            bar = None
            if bd is not None:
                bar = float(bd[0]) * u - float(bd[1]) * v
                bar /= np.linalg.norm(bar)
            cpx, _ = self._project_odom(h, corners)
            ctr, _ = self._project_odom(h, center[None])
            spx = None if sym is None else self._project_odom(h, sym)[0]
            # rec is a detection in the SOURCE image: its bar direction too (the detector's is in rectified pixels; a
            # caller that maps rec["bar_dir"] through the source camera, refine_with_wrist, would turn it otherwise)
            rec = dict(l, corners=cpx, center=ctr[0], label_corners=cpx, label_center=ctr[0], symbol_corners=spx,
                       bar_dir=None if bar is None else _image_direction(h, center, bar),
                       rectified={"px_per_m": s, "size": [Wr, Hr]})
            out.append((rec, {"center": center, "bar_dir": bar, "corners": corners,
                              "width": quad_width(corners if sym is None else sym)}))
        return out

    def _clear_barcode_slots(self):
        ws = self.ws
        ws.wrist_target = ws.wrist_target_refined = ws.stance_target = None
        self._wrist_detection = None
        self._last_label = None

    def next_barcode(self):
        ws = self.ws
        if not ws.scan_plan and not (ws.stack or {}).get("planned"):    # a side planned empty is a plan
            return fail("no scan plan (call plan_scans first)")
        closed = None
        prev = ws.current_barcode_id
        if prev is not None and prev in ws.barcodes and ws.barcodes[prev].status == "pending" \
                and prev not in self._stance_attempted:
            # a barcode is never given up before its stance was even planned (2026-11-07: the LLM skipped three
            # barcodes after a refused tool call); after a plan_stance or scan attempt (failed or not) skipping is allowed
            return fail(f"barcode {prev} has not been attempted yet: compute_wrist_target -> plan_stance -> approach -> "
                        "move_arm / refine_with_wrist -> scan; call next_barcode only when it was scanned or failed")
        if prev is not None and prev in ws.barcodes and ws.barcodes[prev].status == "pending":
            ws.barcodes[prev].status = "failed"
            closed = {"id": prev, "status": "failed"}
        self._clear_barcode_slots()
        cur = 0 if ws.stack is None else ws.stack["current"]
        # wall layout: the cells are fixed by the layout, no plan update from here
        update = self._plan_update_from_here() if self.task == 3 and prev is not None and self.wall_layout is None else None
        diag = None if update is None else {"plan_update": update.pop("diagnostics")}     # logged, not shown (D11)
        nxt = next((b for b in ws.scan_plan if ws.barcodes[b].status == "pending" and ws.barcodes[b].face_id == cur), None)
        ws.current_barcode_id = nxt
        base = {"closed": closed, "n_planned": len(ws.scan_plan), "scanned": [e["content"] for e in ws.scanned],
                "scan_plan": ws.scan_plan_export(), "face_id": cur, "face": self.TOUR[cur], "plan_update": update}
        if nxt is None:
            nf = self._next_face()
            if nf is not None:
                return with_diagnostics(ok(dict(base, done=False, face_done=True, next_face=self.TOUR[nf], next_face_id=nf),
                                           f"every barcode of the {self.TOUR[cur]} side has been handled: call "
                                           f"go_to_next_face (next: {self.TOUR[nf]}), then plan_scans"), diag)
            return with_diagnostics(ok(dict(base, done=True), "every planned barcode has been handled"), diag)
        bc = ws.barcodes[nxt]
        remaining = sum(ws.barcodes[b].status == "pending" for b in ws.scan_plan) - 1
        data = dict(base, done=False, barcode_id=nxt, index=bc.order_index, box_id=bc.box_id, remaining=remaining,
                    width_m=bc.width_m, standoff_m=self._standoff_or_none(bc))
        if self.wall_layout is not None:
            data["kind"] = bc.kind
        return with_diagnostics(ok(data, f"current barcode: {nxt}"), diag)

    def _plan_update_from_here(self) -> dict:
        """Look again from where the robot is now (closer to the side than the observation pose) before choosing the
        next barcode: label detection on busy box textures is view-dependent (2026-11-07 stack4c_L0_0001: a label
        missed from the observation pose was never added because the LLM skipped the prompt's plan_scans)."""
        u = self.plan_scans(_update=True)
        return {"status": u["status"], "reason": u["reason"],
                "added": [a["id"] for a in (u["data"].get("added") or [])] if u["status"] == "ok" else [],
                "diagnostics": u.get("diagnostics")}

    def _next_face(self):
        """Next stack side in the tour (front -> right -> back -> left) not visited yet; None for a single-face task or
        when every side has been visited."""
        st = self.ws.stack
        if st is None or not st.get("multi_face"):
            return None
        for k in range(st["current"] + 1, len(self.TOUR)):
            if k not in st["visited"]:
                return k
        return None

    def _move_axis(self, dx=0.0, dy=0.0, dyaw=0.0, log=None):
        """One pure-axis move in the robot frame (forward OR sideways OR turn in place): the movement controller only
        follows a straight line for such moves (it saturates x / y / yaw separately). Refused when the straight path
        would bring the robot root closer than stance.min_face_dist_m to the stack footprint (or, before the footprint
        is closed, to a known face plane in front of it)."""
        assert (dx != 0) + (dy != 0) + (dyaw != 0) <= 1, "pure-axis moves only"
        T = self._odom()
        if dx or dy:
            dmin = self._c2()["stance"]["min_face_dist_m"]
            for a in np.linspace(0.0, 1.0, 11):
                p = T[:2, 3] + T[:2, :2] @ np.array([a * dx, a * dy])
                c = self._root_clearance(p)
                if c is not None and c < dmin - 1e-6:
                    return fail(f"move refused: the path would pass {c:.2f} m from the stack (minimum {dmin} m)")
        r = self.robot.move_root_relative(dx, dy, dyaw, self._c1()["approach"]["move_timeout_s"])
        if log is not None:
            log.append({"move": [dx, dy, dyaw], "status": r["status"], "summary": move_summary(r)})
        if r["status"] not in ("ok", "timeout"):
            return fail(f"move {r['status']}: {r.get('reason')}")
        return ok()

    def _root_clearance(self, pxy):
        """Horizontal clearance of a root position to the stack: signed distance to the footprint polygon (negative
        inside) once it is closed, else the smallest signed distance in front of the known faces' planes within their
        lateral extent (None when nothing is known)."""
        poly = self._footprint()
        if poly is not None:
            return float(_poly_signed_distance(np.asarray(pxy, float)[None, :], poly)[0])
        st = self.ws.stack
        if st is None or not st["faces"]:
            return None
        best = None
        for fid in st["faces"]:
            f = self._stack_face(fid)
            n = np.asarray(f["normal"], float)[:2]
            C = np.asarray(f["corners"], float)[:, :2]
            r_ = np.array([-n[1], n[0]])
            lat = (np.asarray(pxy, float) - C.mean(axis=0)) @ r_
            ext = (C - C.mean(axis=0)) @ r_
            if ext.min() <= lat <= ext.max():
                d = float((np.asarray(pxy, float) - C.mean(axis=0)) @ n)
                best = d if best is None else min(best, d)
        return best

    def go_to_next_face(self):
        """Walk around the stack to the next side of the tour and stand at its observation pose (task 3, every side).
        Arm home, stand up; back off to the ring distance in front of the current side; sidestep past the corner by
        the same distance; turn 90 deg in place to face the next side; detect + fit it (its extent gives the stack
        depth); then sidestep / step to its observation pose (compute_obs_pose on that side) and refine once from
        there. Pure-axis moves only, each checked against the stack footprint. The next call is plan_scans."""
        ws, st = self.ws, self.ws.stack
        if st is None:
            return fail("no stack model (call plan_scans first)")
        nf = self._next_face()
        if nf is None:
            return fail("no side left to visit" if st.get("multi_face") else "this task scans the front side only")
        cur = st["current"]
        if any(ws.barcodes[b].status == "pending" and ws.barcodes[b].face_id == cur for b in ws.scan_plan):
            return fail(f"the {self.TOUR[cur]} side still has pending barcodes: call next_barcode")
        if cur not in st["faces"]:
            return fail(f"the {self.TOUR[cur]} side was never fitted")
        log = []
        if not self.arm_at_home:
            r = self.move_arm("home")
            if r["status"] != "ok":
                return fail(f"arm home: {r['reason']}")
        g = self._walk_guard()
        if g is not None:
            return g
        dirs = self._face_dirs()
        n_c, n_n = dirs[cur], dirs[nf]
        Fc = self._stack_face(cur)["corners"]
        d_ring = float(st["faces"][cur].get("d_obs") or self._c1()["d_obs_range"][0])
        corner = Fc[np.argmax(Fc[:, :2] @ n_n[:2]), :2]
        face_pt = Fc.mean(axis=0)[:2]

        def face_yaw(n):
            return math.atan2(-n[1], -n[0])

        # 1. face the current side
        T = self._odom()
        r = self._move_axis(dyaw=wrap(face_yaw(n_c) - yaw_of(T)), log=log)
        if r["status"] != "ok":
            return fail(f"turn to the {self.TOUR[cur]} side: {r['reason']}", {"log": log})
        # 2. back off / step to the ring distance in front of it
        T = self._odom()
        s_ = float((T[:2, 3] - face_pt) @ n_c[:2])
        if abs(s_ - d_ring) > 1e-3:
            r = self._move_axis(dx=s_ - d_ring, log=log)
            if r["status"] != "ok":
                return fail(f"ring distance: {r['reason']}", {"log": log})
        # 3. sidestep past the corner by the ring distance (lateral axis of the robot = its left)
        T = self._odom()
        left = T[:2, :2] @ np.array([0.0, 1.0])
        goal = corner + d_ring * n_n[:2]
        dy = float((goal - T[:2, 3]) @ left)
        r = self._move_axis(dy=dy, log=log)
        if r["status"] != "ok":
            return fail(f"sidestep to the corner: {r['reason']}", {"log": log})
        # 4. turn in place to face the next side
        T = self._odom()
        r = self._move_axis(dyaw=wrap(face_yaw(n_n) - yaw_of(T)), log=log)
        if r["status"] != "ok":
            return fail(f"turn to the {self.TOUR[nf]} side: {r['reason']}", {"log": log})
        # 5. sidestep along the next side, into the stack's depth direction, by the ring distance: the robot then
        #    stands square to the side's near edge and the side lies in the head camera's view (from the corner point
        #    itself only its first ~0.2 m are visible: 2026-11-07, the right side could not be fitted from there)
        T = self._odom()
        left = T[:2, :2] @ np.array([0.0, 1.0])
        r = self._move_axis(dy=float(np.dot(-n_c[:2], left)) * d_ring, log=log)
        if r["status"] != "ok":
            return fail(f"sidestep along the {self.TOUR[nf]} side: {r['reason']}", {"log": log})
        # 6. detect + fit the next side; the current side switches only when this tool succeeds
        st["current"] = nf
        res = None
        try:
            res = self._fit_and_reach_side(nf, log)
        finally:
            if res is None or res["status"] != "ok":
                st["current"] = cur
        return res

    def _fit_and_reach_side(self, nf, log):
        """go_to_next_face steps 6-7: fit the side from here, walk to its observation pose, refit once from there."""
        ws, st = self.ws, self.ws.stack
        n_c = self._face_dirs()[(nf - 1) % len(self.TOUR)]
        refine = []
        for k in range(2):                                 # from the side's near edge, then once more from the observation pose
            if k == 0:                                     # the new side is unknown: the head view is searched
                self._obs_faces, self.ws.target_box_id = None, None
            r = self.detect_boxes()
            if r["status"] != "ok":
                return fail(f"no box detected from the {self.TOUR[nf]} side", {"log": log})
            self.ws.target_box_id = "all"
            fitted, seen = [], []
            for bid in list(ws.boxes):
                rf = self._fit_one(bid)
                if rf["status"] == "ok":
                    fid = self._face_id(ws.boxes[bid].face["normal"])
                    seen.append({"box": bid, "side": self.TOUR[fid], "normal": ws.boxes[bid].face["normal"]})
                    if fid == nf:
                        fitted.append(ws.boxes[bid].face)
                else:
                    seen.append({"box": bid, "fit": rf["reason"]})
            if not fitted:
                return fail(f"the {self.TOUR[nf]} side was not fitted from here", {"log": log, "seen": seen})
            if k == 0:
                st["faces"].pop(nf, None)
            self._stack_add_faces(nf, fitted)
            po = self.compute_obs_pose("all")
            if po["status"] == "fail":
                return fail(f"observation pose of the {self.TOUR[nf]} side: {po['reason']}", {"log": log})
            tgt = ws.root_target
            # 6. sidestep along the side, then step to the observation distance (pure-axis moves)
            T = self._odom()
            dxy = np.asarray(tgt["pos"], float) - T[:2, 3]
            loc = T[:2, :2].T @ dxy
            refine.append({"pass": k, "offset_robot": loc.tolist(), "d_obs": tgt["d_obs"]})
            if abs(loc[1]) > 1e-3:
                r = self._move_axis(dy=float(loc[1]), log=log)
                if r["status"] != "ok":
                    return fail(f"sidestep along the {self.TOUR[nf]} side: {r['reason']}", {"log": log})
            T = self._odom()
            loc = T[:2, :2].T @ (np.asarray(tgt["pos"], float) - T[:2, 3])
            if abs(loc[0]) > 1e-3:
                r = self._move_axis(dx=float(loc[0]), log=log)
                if r["status"] != "ok":
                    return fail(f"step to the observation distance: {r['reason']}", {"log": log})
            T = self._odom()
            dyaw = wrap(tgt["yaw"] - yaw_of(T))
            if abs(dyaw) > 1e-3:
                self._move_axis(dyaw=dyaw, log=log)
        st["visited"].append(nf)
        st["faces"][nf]["d_obs"] = float(ws.root_target["d_obs"])
        n_c = self._face_dirs()[(nf - 1) % len(self.TOUR)]  # re-registered by the fits above
        C = self._stack_face(nf)["corners"]
        depth = float(np.ptp(C[:, :2] @ n_c[:2]))           # extent of the new side along the previous side's normal
        return ok({"face_id": nf, "face": self.TOUR[nf], "obs_pose": ws.root_target, "side_extent_m": depth,
                   "refine": refine, "log": log},
                  f"at the observation pose of the {self.TOUR[nf]} side: call plan_scans")

    # ------------------------------------------------------------------ wrist target / stance / arm
    def _standoff(self, bc):
        """Scanner stand-off for a barcode: fixed (offset_m) or adaptive from the estimated label width."""
        c2 = self._c2()
        S = c2["standoff"]
        if S["mode"] == "fixed":
            return float(c2["offset_m"])
        if S["mode"] != "adaptive":
            raise ValueError(f"unknown standoff mode {S['mode']!r}")
        if bc.width_m is None:
            raise RuntimeError(f"label width of {bc.id} unknown: no adaptive stand-off")
        if self.scanner_hfov is None:
            raise RuntimeError("scanner imager intrinsics unknown (robot info wrist_K / wrist_size): no adaptive stand-off")
        d = adaptive_standoff(bc.width_m, self.scanner_hfov, S["fill_frac"], S["min_m"], S["max_m"])
        if bc.width_source != "wrist":
            # a head width may push the scanner farther than the validated stand-off (task2.yaml offset_m), never
            # closer: head label boxes come out both too large (DINO box, +20 %) and too small (a partial box on the
            # rectified face: 3.5 / 4.9 cm for 8 cm codes, 2026-11-07 agent_full run -> clamped to 0.12 m, no read).
            # Only the wrist measurement (the decoded symbol) may bring the scanner closer than that.
            d = max(d, float(c2["offset_m"]))
        return d

    def _standoff_or_none(self, bc):
        """The stand-off of a label entry; None for a wall-layout cell entry (no label width before search_cell)."""
        return None if bc.kind == "cell" else self._standoff(bc)

    def _pre_gap(self):
        """Pre-pose gap behind the stand-off: the one plan_stance found reachable for the current barcode (stance
        "pre_gap"), else the configured pre - final gap (task2.yaml 0.30 - 0.20)."""
        st = self.ws.stance_target
        wt = self.ws.wrist_target
        if st is not None and st.get("pre_gap") is not None and wt is not None and st.get("barcode_id") == wt.get("barcode_id"):
            return float(st["pre_gap"])
        c2 = self._c2()
        return float(c2["pre_offset_m"] - c2["offset_m"])

    def _pre_offset(self, off):
        """Pre-pose distance for a stand-off: the pre gap (_pre_gap) behind it."""
        return float(off) + self._pre_gap()

    def compute_wrist_target(self, barcode_id: str | None = None, offset: float | None = None, refined: bool = False):
        if (nc := self._not_current(barcode_id)) is not None:
            return fail(nc)
        bc = self._current(barcode_id)
        if bc is None or bc.center is None:
            return fail("no barcode pose (call estimate_barcode_pose / plan_scans + next_barcode)")
        ignored = None
        if bc.kind == "cell":
            # wall layout, a box whose label is not found yet: the hand-camera view of the whole box face (search_cell);
            # a caller's offset is ignored as for labels
            t = self._look_target(bc)
            self.ws.wrist_target = t
            self.ws.wrist_target_refined = None
            self._marker("wrist_target", "blue", "pose", t["pos"], dir=t["forward"])
            return with_diagnostics(
                ok({"barcode_id": bc.id, "kind": "cell", "offset_ignored": None if offset is None else float(offset),
                    "look": {"offset": t["offset"], "rolls": [a["roll_deg"] for a in t["alternatives"]],
                             "face_size": t["face_size"], "delta_lat": t["delta_lat"]}},
                   "hand-camera view of the whole box face planned; call plan_stance"),
                {"alternatives": [{k: a[k] for k in ("pos", "roll_deg", "h", "v")} for a in t["alternatives"]]})
        if offset is not None and self._c2()["standoff"]["mode"] == "adaptive":
            # task 3: the stand-off is computed from the label width; an offset from the caller is ignored, not refused
            # (2026-11-07: the LLM invented offsets and scanned from 0.14 m; refusing them instead made it skip to
            # plan_stance without a wrist target and then skip the barcode)
            ignored, offset = float(offset), None
        if offset is not None:
            off = float(offset)
        elif refined and bc.width_source != "wrist" and self.ws.wrist_target is not None \
                and self.ws.wrist_target.get("barcode_id") == bc.id:
            off = float(self.ws.wrist_target["offset"])       # no wrist width: keep the stand-off of the head target
        else:
            off = self._standoff(bc)
        t = {"pos": bc.center + off * bc.normal, "forward": -bc.normal, "bar_dir": bc.bar_dir, "offset": off, "barcode_id": bc.id,
             "long_axis": self._long_axis(bc)}
        if refined:
            self.ws.wrist_target_refined = t
            self._marker("wrist_refined", "purple", "pose", t["pos"], dir=t["forward"])
        else:
            self.ws.wrist_target = t
            self.ws.wrist_target_refined = None             # a refined target of an earlier estimate is stale now
            self._marker("wrist_target", "blue", "pose", t["pos"], dir=t["forward"])
        # label_width_m: the label's extent (the head label box), the width every same-label test and the wrist gate
        # use; the stand-off width (cv2 symbol / wrist symbol, bc.width_m) is logged only (D11, never shown to the LLM)
        return with_diagnostics(
            ok({"barcode_id": bc.id, "wrist_target_odom": {"pos": t["pos"], "forward": t["forward"]}, "offset": off,
                "offset_ignored": ignored,
                "pre_offset": self._pre_offset(off), "label_width_m": bc.label_width_m,
                "note": "roll about the forward axis is not constrained (the scanner reads at any orientation)"}),
            {"standoff_width_m": bc.width_m, "width_source": bc.width_source})

    @staticmethod
    def _long_axis(bc):
        """The barcode's long axis on its face (odom): perpendicular to the bars in the face plane (code128: the axis the
        scan line must follow), else the longer side of the label quad; None if neither is known. The scanner image's
        horizontal axis is aligned with it (2026-11-07 user review: a real scanner reads a code whose bars are tilted
        against the image frame poorly; roll was left free before)."""
        n = np.asarray(bc.normal, float)
        if bc.bar_dir is not None:
            L = np.cross(n, np.asarray(bc.bar_dir, float))
        elif bc.corners_odom is not None:
            C = np.asarray(bc.corners_odom, float)
            e1, e2 = C[1] - C[0], C[2] - C[1]
            L = e1 if np.linalg.norm(e1) >= np.linalg.norm(e2) else e2
        else:
            return None
        L = L - (L @ n) * n
        return None if np.linalg.norm(L) < 1e-9 else L / np.linalg.norm(L)

    def _roll_in_root(self, t, T_odom_root=None):
        L = t.get("long_axis")
        if L is None:
            return None
        return rotate(inv_T(self._odom() if T_odom_root is None else T_odom_root), L)

    def _target_in_root(self, t, T_odom_root=None, offset=None):
        T = inv_T(self._odom() if T_odom_root is None else T_odom_root)
        pos = np.asarray(t["pos"], float)
        if offset is not None:                       # same target line, different distance from the barcode
            pos = pos + (offset - t["offset"]) * (-np.asarray(t["forward"], float))
        return apply(T, pos), rotate(T, t["forward"])

    @property
    def _stance_attempted(self):
        if not hasattr(self, "_stance_attempted_ids"):
            self._stance_attempted_ids = set()
        return self._stance_attempted_ids

    @property
    def _failed_stances(self):
        """Stances that failed placement, {(barcode id, wrist target position, face_dist, lateral_offset, level index)}
        (check_wrist_placement records, _plan_stance skips them for the same barcode and unchanged target)."""
        if not hasattr(self, "_failed_stance_ids"):
            self._failed_stance_ids = set()
        return self._failed_stance_ids

    def plan_stance(self, which: str = "wrist_target"):
        """A planned stance is an attempt of the barcode only when planning fails (no stance / unreachable); otherwise
        the attempt starts with the walk to it (approach)."""
        r = self._plan_stance(which)
        if r["status"] != "ok" and self.ws.current_barcode_id is not None:
            self._stance_attempted.add(self.ws.current_barcode_id)
        return r

    def _plan_stance(self, which: str = "wrist_target"):
        """Root stance for the wrist target: a grid of positions relative to the barcode (distance from its face in
        [min_face_dist, max_face_dist], lateral offset along the robot's left axis), yaw facing the face; every point
        needs IK solutions for the pre-pose and the final pose. Cost: largest joint-limit margin, then the shortest
        walk from the current root."""
        t = self.ws.wrist_target_refined if which == "refined" else self.ws.wrist_target
        if t is None and self.ws.current_barcode_id is not None and which != "refined":
            # the wrist target of the current barcode follows from its estimate alone: computed here when the caller
            # skipped compute_wrist_target (2026-11-07: the LLM did, then skipped the barcode)
            r = self.compute_wrist_target(self.ws.current_barcode_id)
            if r["status"] != "ok":
                return fail(f"no wrist target: {r['reason']}")
            t = self.ws.wrist_target
        if t is None:
            return fail("no wrist target")
        bc = self.ws.barcodes.get(t.get("barcode_id"))
        if bc is None or bc.center is None:
            return fail("the wrist target has no barcode estimate")
        S = self._c2()["stance"]
        T0 = self._odom()
        n = -np.asarray(t["forward"], float)
        nh = np.array([n[0], n[1], 0.0]); nh /= np.linalg.norm(nh)
        yaw = math.atan2(-nh[1], -nh[0])
        left = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
        base = np.array([bc.center[0], bc.center[1], 0.0])
        # every known face is an obstacle: with boxes of different setbacks, a stance lateral to the barcode can stand
        # in front of a neighbouring face that sticks out toward the robot (min_face_dist applies to it too). The
        # barcode's own face never rejects more than the plane test above (distance to a segment >= to its line).
        obstacles = self._known_faces()
        poly = self._footprint()
        q0 = self._q()
        dists = np.arange(S["min_face_dist_m"], S["max_face_dist_m"] + 1e-9, S["step"])
        lat = np.arange(S["lateral_range"][0], S["lateral_range"][1] + 1e-9, S["step"])
        # required final distances: the stand-off, and with a head width (an upper bound of the symbol) also the clamp
        # min_m, the closest the wrist measurement can bring the final pose. Pre-pose: the configured gap behind the
        # stand-off is preferred; the pre-pose only has to keep the wrist camera at least as far as the largest
        # possible final distance (the final move then goes forward), so smaller gaps down to 0 are accepted in
        # stance-grid steps when the preferred one is out of reach (deep squats, 2026-11-06 stack scene)
        # With a head width (an upper bound of the symbol: the true stand-off lies in [min_m, offset]) the final
        # distance is not known yet: the stance must reach the clamp min_m and a pre-pose as far as possible up to
        # offset + the pre gap, from which the wrist camera measures the code; the final pose then lies between the
        # two reachable ends (forward, or back if the pre-pose had to be closer than the measured stand-off). A head
        # stand-off that is out of reach does not veto the stance (2026-11-06 stack scene: the 0.19 m code at the
        # head's 0.30 m is unreachable in any squat, at the measured 0.20 m it is reachable).
        c2 = self._c2()
        S3 = c2["standoff"]
        gap0 = float(c2["pre_offset_m"] - c2["offset_m"])
        look = t.get("purpose") == "look"
        if look:
            # wall layout, the hand-camera view of a box face (search_cell): the look pose only, approached from home
            # (no pre-pose); its quarter-roll alternatives tie geometrically, reachability decides (the largest
            # manipulability, the planner's posture criterion)
            finals, pre_dists = [float(t["offset"])], []
        elif S3["mode"] == "adaptive" and bc.width_source != "wrist":
            # head / hand_label width: an upper bound of the symbol
            finals = [float(S3["min_m"])]
            pre_dists = list(np.arange(t["offset"] + gap0, float(S3["min_m"]) - 1e-9, -S["step"]))
        else:
            finals = [t["offset"]]
            pre_dists = list(np.arange(t["offset"] + gap0, t["offset"] - 1e-9, -S["step"]))
        body = self._body()
        # body heights to try, standing first: the walker's measured squat levels (robot.yaml body.calibration), or
        # only the current height for a walker without a body command. The first (highest) level with a feasible
        # stance wins: squatting costs time and balance margin.
        levels = body.levels if body is not None else [None]
        margin = self.cfg["robot"]["body"]["clearance_margin_m"]
        R_yaw = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
        rejected = {"reach": 0, "ik": 0, "face_clearance": 0, "self_clearance": 0, "not_robust": 0, "failed_placement": 0}
        best, chosen, chosen_li, fallback, cands = None, None, None, None, []
        # stances that already failed placement for this barcode and this exact wrist target (check_wrist_placement):
        # the planner is deterministic (per-candidate IK seeds), so identical inputs would give the identical stance and
        # the identical failure (2026-11-19 scanroom: the same stance 4 times, ~600 s). A changed target is a new problem.
        # a look target is recomputed on every re-registration of the wall: its identity is the box (the barcode id)
        tpos = ("look",) if look else _stance_target_key(t["pos"])
        alts = [dict(t, pos=a["pos"], long_axis=a["long_axis"]) for a in t["alternatives"]] if look else [t]
        failed = {(d_, l_, li_) for (b_, tp_, d_, l_, li_) in self._failed_stances if b_ == bc.id and tp_ == tpos}
        restarts = self.ik.cfg["restarts"]
        self.ik.cfg["restarts"] = 1
        stall = (int(self.ik.cfg["plan_stall_window"]), float(self.ik.cfg["plan_stall_eps_m"]))
        try:
            for li, lv in enumerate(levels):
                q_lv = q0 if lv is None else body.q_at(q0, lv)
                # with the feet fixed the pelvis moves while squatting: walk (standing) to the pelvis position minus
                # that shift, so the squatted pelvis lands on the planned point
                shift = np.zeros(2) if lv is None else R_yaw @ body.foot_shift(lv, levels[0], q0)
                jobs, meta = [], []
                for d in dists:
                    for l in lat:
                        if (float(d), float(l), li) in failed:
                            rejected["failed_placement"] += 1
                            continue
                        p = base + d * nh + l * left
                        pw = p[:2] - shift
                        if poly is not None:
                            # closed footprint (multi-face): signed distance to it, so a stance behind / beside the stack
                            # is judged against the whole stack, not against the infinite front planes
                            if _poly_signed_distance(np.asarray(pw)[None, :2], poly)[0] < S["min_face_dist_m"] - 1e-9:
                                continue
                        else:
                            if geo.horizontal_distance_to_plane(np.r_[pw, 0.0], bc.center, n) < S["min_face_dist_m"] - 1e-9:
                                continue
                            if any(horizontal_distance_to_face(np.r_[pw, 0.0], f) < S["min_face_dist_m"] - 1e-9 for f in obstacles):
                                continue
                        Tc = T_from_xy_yaw(p[0], p[1], yaw, T0[2, 3]) if lv is None else body.T_root(p[0], p[1], yaw, lv)
                        ji = []
                        for ta in alts:                       # one job per grid point (per look alternative)
                            ji.append(len(jobs))
                            jobs.append({"Tc": Tc, "q_lv": q_lv, "t": ta, "finals": finals, "pre_dists": pre_dists,
                                         "stall": stall, "seed": [int(self.ik.cfg["seed"]), li, len(jobs)]})
                        meta.append((p, pw, d, l, Tc, ji))
                # reach prefilter + IK of the required poses + posture, per grid point (tools/stance_eval.py), in the
                # planning process pool when configured (identical results: per-candidate IK seeds)
                evs = self._eval_stance_jobs(jobs, body)
                for (p, pw, d, l, Tc, ji) in meta:
                    oks = [j for j in ji if evs[j]["status"] == "ok"]
                    if not oks:
                        rejected["reach" if all(evs[j]["status"] == "reach" for j in ji) else "ik"] += 1
                        continue
                    j = max(oks, key=lambda j_: evs[j_]["man"])  # the alternative with the best posture (one: itself)
                    ev, ai, ta = evs[j], ji.index(j), alts[ji.index(j)]
                    sols, gap, m, man = ev["sols"], ev["gap"], ev["margin"], ev["man"]
                    travel = float(np.hypot(pw[0] - T0[0, 3], pw[1] - T0[1, 3]))
                    # arm posture: the position manipulability sqrt(det(Jp Jp^T)) of the scanner-tip Jacobian over the
                    # 7 arm joints, smallest over the required poses (Yoshikawa). It falls towards both singular
                    # ends: the nearly straight arm (2026-11-06: a 0.70 m stance chosen by joint margin stayed 9 cm
                    # off) and the folded arm (then chosen by "least extension": elbow up beside the head, joints at
                    # their limits, unnatural and the arm correction diverged) -> the most manipulable pose wins
                    cost = (-gap, -man, travel)              # the preferred pre gap first, then posture, walk
                    cands.append((cost, p, pw, d, l, m, travel, None, gap, Tc, q_lv, sols, man, ta, ai))
                # The cost does not depend on the clearances or on robustness, so the candidates are taken in cost order
                # and the (expensive) checks run only until the first one passes: the same stance as checking every
                # candidate and picking the cheapest passing one (2026-11-07: the arm-vs-legs check of every
                # IK-feasible candidate was 72 % of a 66 s plan_stance).
                # Robustness: the walk ends within the robot's move tolerance of the stance (robot.yaml body
                # stance_robust_*), so the stance is re-checked displaced by it (2026-11-06: a stance at the joint-limit
                # bound became unreachable by 3 mm after the real walk + squat).
                for cand in sorted(cands, key=lambda c_: c_[0]):
                    cost, p, pw, d, l, m, travel, _, gap, Tc, q_lv, sols, man, ta, ai = cand
                    clear = None
                    if body is not None:
                        fc = min(body.face_clearance(s_["q_full"], Tc, obstacles, poly) for s_ in sols)
                        if fc < margin:
                            rejected["face_clearance"] += 1
                            continue
                        sc = min(body.self_clearance(s_["q_full"])[0] for s_ in sols)
                        if sc < margin:
                            rejected["self_clearance"] += 1
                            continue
                        sh = (Tc @ self.kin.pose(q_lv, body.shoulder_frame))[:3, 3]
                        tip = np.asarray(ta["pos"], float) + (finals[0] - ta["offset"]) * (-np.asarray(ta["forward"], float))
                        clear = {"face_m": fc, "arm_leg_m": sc, "arm_extension": round(float(np.linalg.norm(tip - sh) / body.arm_reach), 2),
                                 "manipulability": round(man, 4)}
                    cand = (cost, p, pw, d, l, m, travel, clear, gap, Tc, q_lv, ai)
                    if fallback is None:
                        fallback = (cand, lv)
                    rb = self._stance_robust(ta, Tc, q_lv, finals, None if look else t["offset"] + gap)
                    rejected["not_robust"] += not rb
                    if rb:
                        best, chosen, chosen_li = cand, lv, li
                        break
                cands = []
                if best is not None:
                    break
        finally:
            self.ik.cfg["restarts"] = restarts
        robust = best is not None
        if best is None and rejected["failed_placement"]:
            # not unreachable: stances exist, but every remaining one failed placement for this unchanged target (an
            # attempt of the barcode: plan_stance records it, so next_barcode may close it)
            return fail(f"every stance tried for this target failed placement ({rejected['failed_placement']} stance(s) "
                        "already failed for this unchanged wrist target, no other stance feasible); call next_barcode",
                        {"barcode_id": bc.id, "rejected": rejected, "levels_tried": len(levels)})
        if best is None:
            # unreachable: no stance at any measured body height keeps the scanner poses reachable within the walk
            # tolerance (a stance reachable only exactly was tried twice in the 2026-11-06 stack run and failed both
            # times after squatting). The barcode is reported, not attempted; the scan goes on with the next one.
            z = float(bc.center[2])
            lo, hi = levels[-1], levels[0]
            if lo is None:
                where = "at the current body height"
                side = ""
            elif z < float(hi["pelvis_z"]):
                where = (f"even at the deepest measured squat (height command {lo['height_cmd']}, pelvis z {lo['pelvis_z']:.2f} m)")
                side = "too low: "
            else:
                where = f"even standing (pelvis z {hi['pelvis_z']:.2f} m)"
                side = "too high: "
            reason = (f"barcode {bc.id} unreachable, {side}label centre at z={z:.2f} m; no stance {where} keeps the pre-pose "
                      f"and the final scan pose reachable within the walk tolerance (+-{self.cfg['robot']['body']['stance_robust_pos_m']} m, "
                      f"+-{self.cfg['robot']['body']['stance_robust_yaw_deg']} deg)"
                      + (" (reachable only with an exact stop)" if fallback is not None else ""))
            bc.status, bc.fail_reason = "unreachable", reason
            return fail(reason, {"unreachable": True, "barcode_id": bc.id, "label_z_m": z, "rejected": rejected,
                                 "levels_tried": len(levels), "exact_stop_only": fallback is not None})
        _, p, pw, d, l, m, travel, clear, gap, _, _, ai = best
        self.ws.stance_target = {"pos": [float(pw[0]), float(pw[1])], "yaw": yaw, "face_dist": float(d), "lateral_offset": float(l),
                                 "barcode_id": bc.id, "travel_m": travel,
                                 "base_height": None if chosen is None else float(chosen["height_cmd"]),
                                 "pelvis_pos": [float(p[0]), float(p[1])],
                                 "pelvis_z": None if chosen is None else float(chosen["pelvis_z"]), "pre_gap": gap,
                                 "level_index": chosen_li, "target_pos": list(tpos)}
        if look:                                     # only look stances carry these keys (other logs unchanged)
            self.ws.stance_target.update(purpose="look", look_alt=int(ai),
                                         look_roll_deg=float(t["alternatives"][ai]["roll_deg"]))
        self._marker("stance", "red", "root", [pw[0], pw[1], 0.0], yaw=yaw)
        squat = chosen is not None and chosen is not levels[0]
        return ok({"stance_odom": self.ws.stance_target, "joint_limit_margin_rad": m, "clearance": clear,
                   "squat": squat, "robust": robust, "rejected": rejected},
                  "stance planned (the robot has not moved): call approach('stance_target') to walk there"
                  + (f"; it then squats to base_height {chosen['height_cmd']} (pelvis z {chosen['pelvis_z']:.2f} m) by itself"
                     if squat else ""))

    def _eval_stance_jobs(self, jobs, body):
        """Evaluate stance candidates serially or in the planning pool (robot.yaml ik.plan_workers; 0 = serial)."""
        from . import stance_eval as SE
        shoulder = None if body is None else body.shoulder_frame
        reach = None if body is None else body.arm_reach
        nw = int(self.ik.cfg["plan_workers"])
        if nw <= 0 or len(jobs) < 2:
            return [SE.eval_candidate(self.kin, self.ik, shoulder, reach, j) for j in jobs]
        if getattr(self, "_plan_pool", None) is None:
            import multiprocessing as mp
            import os
            from concurrent.futures import ProcessPoolExecutor
            # one BLAS / OpenMP thread per worker: the parallelism is the process pool. With the library defaults every
            # worker started dozens of threads (2026-11-07: 16 workers x default threads on a loaded 256-core host
            # slowed the whole run, simulators included, by several times). The env is read when the spawned workers
            # import numpy, so it is set around the pool start only.
            keys = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
            saved = {k: os.environ.get(k) for k in keys}
            os.environ.update({k: "1" for k in keys})
            try:
                # the workers build the same model as this process: the variant's URDF and resolved robot config
                self._plan_pool = ProcessPoolExecutor(max_workers=nw, mp_context=mp.get_context("spawn"),
                                                      initializer=SE.init_worker,
                                                      initargs=(self.kin.urdf, self.cfg["robot"], self.ik.side, shoulder, reach))
                list(self._plan_pool.map(int, range(nw)))   # start every worker now, while the env is set
            finally:
                for k, v in saved.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v
        k = -(-len(jobs) // nw)                       # one contiguous chunk per worker
        chunks = [jobs[i:i + k] for i in range(0, len(jobs), k)]
        out = []
        for res in self._plan_pool.map(SE.eval_batch, chunks):
            out += res
        return out

    def _manipulability(self, q) -> float:
        """Position manipulability of the scanner tip over the arm joints: sqrt(det(Jp Jp^T)) (m^3-ish units)."""
        J = self.kin.jacobian(q, self.cfg["robot"]["wrist_point_frame"])[:3, self.ik.vi]
        return float(math.sqrt(max(np.linalg.det(J @ J.T), 0.0)))

    def _stance_robust(self, t, Tc, q_lv, finals, pre_dist) -> bool:
        """Every required pose (finals + the chosen pre-pose distance; None = no pre-pose, a look pose) is still
        reachable with the stance displaced by the robot's move tolerance: +-pos along the stance's x and y, +-yaw about
        the pelvis."""
        B = self.cfg["robot"]["body"]
        dp, dy = B["stance_robust_pos_m"], math.radians(B["stance_robust_yaw_deg"])
        stall = (int(self.ik.cfg["plan_stall_window"]), float(self.ik.cfg["plan_stall_eps_m"]))
        for ex, ey, eyaw in ((dp, 0, 0), (-dp, 0, 0), (0, dp, 0), (0, -dp, 0), (0, 0, dy), (0, 0, -dy)):
            Tp = Tc @ T_from_xy_yaw(ex, ey, eyaw, 0.0)
            for off in list(finals) + ([] if pre_dist is None else [pre_dist]):
                pr, dr = self._target_in_root(t, Tp, off)
                if not self.ik.solve(q_lv, pr, dr, stall, self._roll_in_root(t, Tp), **self._roll_kw(t))["ok"]:
                    return False
        return True

    def _known_faces(self):
        """Every fitted face the harness knows: the boxes of the last detection and the face copies of all barcodes
        (box ids are re-assigned by detect_boxes, the barcode copies survive it). With a registered wall layout: the wall
        rectangle (the authoritative geometry; the per-box fits are stale odometry copies)."""
        if self.wall_layout is not None and self.ws.wall is not None:
            return [self._rect_face(0.0, self.ws.wall["W"], 0.0, self.ws.wall["H"])]
        faces = [b.face for b in self.ws.boxes.values() if b.face is not None]
        st = self.ws.stack
        cur = None if st is None or not st.get("multi_face") else st["current"]
        # multi-side stack: barcode copies of other sides were fitted elsewhere and are in odometry as it drifted since;
        # those sides are covered by the re-registered stack footprint instead (plan_stance checks both)
        faces += [b.face for b in self.ws.barcodes.values() if b.face is not None and (cur is None or b.face_id == cur)]
        return faces

    def move_arm(self, pose: str = "pre", side: str | None = None):
        """See _move_arm; records the last arm command (pose, barcode, result) for the placement check's guidance."""
        if pose != "home" and (rf := self._layout_refusal(self._current())) is not None:
            return rf
        r = self._move_arm(pose, side)
        t = self.ws.wrist_target
        self._last_arm = {"pose": pose, "barcode_id": None if t is None else t.get("barcode_id"), "status": r["status"]}
        return r

    def _move_arm(self, pose: str = "pre", side: str | None = None):
        """pose: 'pre' (stand-off + the pre gap) | 'final' (the stand-off, refined target if available) | 'home'. IK
        on the current joint state, set_arm_joints, then closed-loop corrections of the commanded target (arm PD sag)
        using FK."""
        side = side or self.side
        A = self.cfg["robot"]["arm_motion"]
        if pose == "home":
            js = self.robot.get_joint_state()
            r = self.robot.set_arm_joints(side, self._home_q(js), A["duration_s"])
            self.arm_at_home = True
            return ok(r | {"pose": "home"})
        if abs(self.view_pitch) > 1e-9:               # stances and IK are planned with the torso upright
            r = self._upright()
            if r is not None:
                return fail(f"upright before moving the arm: {r['reason']}")
        if self.task == 3 and self.ws.current_barcode_id is not None:
            # the arm poses are only reachable from the stance planned for them (2026-11-07: the agent_full LLM raised
            # the arm right after plan_stance without walking there: 10 cm short, the barcode was given up)
            st = self.ws.stance_target
            if st is None or st.get("barcode_id") != self.ws.current_barcode_id:
                return fail(f"no stance planned for {self.ws.current_barcode_id}: call plan_stance, then approach('stance_target')")
            if getattr(self, "_stance_reached", None) is not st:
                return fail("the robot is not at the planned stance: call approach('stance_target') first")
        t = self.ws.wrist_target_refined if (pose == "final" and self.ws.wrist_target_refined is not None) else self.ws.wrist_target
        if t is None:
            return fail("no wrist target")
        off = t["offset"] if pose == "final" else self._pre_offset(t["offset"])
        rr = self._reach(t, off, side)
        if rr["ik_fail"]:
            return fail(rr["reason"], {"iters": rr["iters"]})
        hist = rr["history"]
        data = {"pose": pose, "offset": off, "barcode_id": t.get("barcode_id"), "history": hist}
        if hist[-1]["pos_err_m"] >= A["correction_tol"]:
            # "partial" only while the scan / refine can still recover: the label centre must lie inside the wrist
            # camera's view cone from the pose reached (2026-11-07 stack4f_L0_0002: a raised arm stopped 0.21 m / 23 deg
            # off, "partial: go on" was reported and the scan could not read). Outside it the pose is not reachable
            # from this stance, which the caller handles as an IK failure (re-plan the stance).
            view = self._label_view(t, off)
            data["view"] = view
            lat, half = view["label_lateral_m"], view["view_half_width_m"]
            if not view["in_view"]:
                return fail(f"IK: target not reachable from this stance (residual {hist[-1]['pos_err_m']:.3f} m / "
                            f"{hist[-1]['angle_err_deg']:.1f} deg after {len(hist)} correction(s): the label centre is "
                            f"{lat:.3f} m off the scanner axis, outside its view half-width {half:.3f} m)", data)
            # the closed-loop corrections did not converge (2026-11-06 stack scene: the raised, nearly straight arm
            # stayed 9 cm off); reported, not hidden behind "ok"
            return partial(f"arm reached the target with a residual {hist[-1]['pos_err_m']:.3f} m / {hist[-1]['angle_err_deg']:.1f} deg "
                           f"after {len(hist)} correction(s); this is not a failure: go on (pre: refine_with_wrist; final: "
                           "check_wrist_placement, then scan, which retries by itself)", data)
        return ok(data)

    def _reach(self, t, off, side) -> dict:
        """IK on the current joint state to wrist target t at distance off, set_arm_joints, then closed-loop corrections
        of the commanded target (arm PD sag) using FK (move_arm and search_cell's look poses). Returns {"ik_fail",
        "history"} (+ "reason", "iters" when IK found no solution for the first command)."""
        A = self.cfg["robot"]["arm_motion"]
        p_goal, d_goal = self._target_in_root(t, offset=off)
        p_cmd = p_goal.copy()
        hist = []
        best = None                                   # (error, q_arm) of the best command sent so far
        for it in range(1 + A["correction_iters"]):
            q = self._q()
            sol = self.ik.solve(q, p_cmd, d_goal, roll_axis=self._roll_in_root(t), **self._roll_kw(t))
            if not sol["ok"]:
                if best is not None:                  # a correction left the reachable set: keep the best pose reached
                    self.robot.set_arm_joints(side, best[1], A["duration_s"])
                    self.robot.stand(A["settle_s"])
                    hist.append({"reverted_to_best": True, "pos_err_m": best[0], "angle_err_deg": best[2]})
                    break
                return {"ik_fail": True, "iters": it, "history": hist,
                        "reason": f"IK: target not reachable (residual {sol['pos_err']:.3f} m / {sol['angle_err_deg']:.1f} deg)"}
            self.robot.set_arm_joints(side, sol["q_arm"], A["duration_s"])
            self.arm_at_home = False
            self.robot.stand(A["settle_s"])
            p_goal, d_goal = self._target_in_root(t, offset=off)   # the base may have drifted: re-express the target
            T = self.kin.pose(self._q(), self.cfg["robot"]["wrist_point_frame"])
            err = p_goal - T[:3, 3]
            e, a = float(np.linalg.norm(err)), math.degrees(angle_between(T[:3, 0], d_goal))
            hist.append({"pos_err_m": e, "angle_err_deg": a})
            if best is not None and e > best[0]:
                # the correction made it worse: the base / torso moves with the arm (GR00T balances the raised arm with
                # the waist), so a full-gain correction can diverge (2026-11-06 stack scene: 3.5 -> 18 -> 36 cm).
                # Go back to the best command and stop correcting
                self.robot.set_arm_joints(side, best[1], A["duration_s"])
                self.robot.stand(A["settle_s"])
                T = self.kin.pose(self._q(), self.cfg["robot"]["wrist_point_frame"])
                p_goal, d_goal = self._target_in_root(t, offset=off)
                hist.append({"reverted_to_best": True, "pos_err_m": float(np.linalg.norm(p_goal - T[:3, 3])),
                             "angle_err_deg": math.degrees(angle_between(T[:3, 0], d_goal))})
                break
            if best is None or e < best[0]:
                best = (e, sol["q_arm"], a)
            if e < A["correction_tol"]:
                break
            p_cmd = p_cmd + err
        return {"ik_fail": False, "history": hist}

    def _label_view(self, t, off, label_width: float | None = None) -> dict:
        """Where the wrist target's label centre lies relative to the SCANNER's view cone from the current arm pose
        (wrist point frame, x = scanner axis): lateral distance from the axis vs the view half-width at that range.
        With label_width, "label_in_view" also requires the whole label (centre offset + half width) inside the view."""
        if self.scanner_hfov is None:
            raise RuntimeError("scanner imager intrinsics unknown (robot info wrist_K / wrist_size): no view check")
        T = self.kin.pose(self._q(), self.cfg["robot"]["wrist_point_frame"])
        p_goal, d_goal = self._target_in_root(t, offset=off)
        v = (p_goal + d_goal * off) - T[:3, 3]                  # wrist point -> label centre
        fwd = T[:3, 0] / np.linalg.norm(T[:3, 0])
        along = float(v @ fwd)
        lat = float(np.linalg.norm(v - along * fwd))
        half = along * math.tan(self.scanner_hfov / 2)
        out = {"label_lateral_m": round(lat, 3), "label_along_m": round(along, 3), "view_half_width_m": round(half, 3),
               "in_view": bool(along > 0 and lat < half)}
        if label_width is not None:
            out["label_in_view"] = bool(along > 0 and lat + label_width / 2 < half)
        return out

    def _home_q(self, js):
        if not hasattr(self, "_q_home"):
            raise RuntimeError("home pose unknown")
        return self._q_home

    def remember_home(self):
        js = self.robot.get_joint_state()
        q = self.kin.q_from_state(js["names"], js["q"])
        self._q_home = q[self.ik.qi].copy()
        self.arm_at_home = True

    def _detect_wrist(self, expect_odom=None):
        """Label detection in a hand-camera frame (cv2 at the hand camera's upscale, perception.yaml barcode
        wrist_upscale per robot variant; GroundingDINO when cv2 finds nothing). The detection kept is the one nearest
        to where the label is expected: expect_odom (the current label estimate) projected into this frame, else the
        image centre. The scanner axis projects near the image centre on both variants, but the D405's 87 deg view
        holds 2-3 neighbouring labels at the pre-pose, and the label estimate, not the image centre, says which one is
        the planned label (the consistency gate in refine_with_wrist only rejects, it cannot recover)."""
        w = self._capture_wrist()
        recs = detect_barcodes(w["rgb"], None, cfg=dict(self.cfg["perception"]["barcode"], upscale=self.cfg["perception"]["barcode"]["wrist_upscale"]))
        H, W = w["rgb"].shape[:2]
        if not recs:
            recs = detect_label_dino(w["rgb"], [0, 0, W, H], self.det)
        if not recs:
            return w, None
        ref = np.array([W / 2, H / 2])
        if expect_odom is not None:
            uv, z = self._project_odom(w, np.asarray(expect_odom, float)[None])
            if z[0] > 0:
                ref = uv[0]
        recs.sort(key=lambda r: np.hypot(r["center"][0] - ref[0], r["center"][1] - ref[1]))
        return w, recs[0]

    def refine_with_wrist(self, barcode_id: str | None = None):
        """re-fit the face plane from the head, transform it to the hand camera through FK, detect the
        barcode in the hand-camera image, intersect its centre ray with the plane, update the wrist target."""
        if (nc := self._not_current(barcode_id)) is not None:
            return fail(nc)
        bc = self._current(barcode_id)
        if (rf := self._layout_refusal(bc)) is not None:
            return rf
        if bc is None or bc.center is None:
            return fail("no barcode estimate")
        if self.ws.wrist_target is None or self.ws.wrist_target.get("barcode_id") != bc.id:
            return fail(f"no wrist target for {bc.id} (call compute_wrist_target)")
        face = bc.face
        layout = bc.wall is not None
        est = None
        n_hidden = 0

        def detect():
            # wall layout: the label only inside the box's cell (search_cell's detector), never the whole-image pick
            # (2026-11-19 run call 41: it returned the R3 label 0.25 m below the R4 target)
            nonlocal est, n_hidden
            if layout:
                w_, rec_, est, n_hidden = self._detect_wrist_cell(bc)
                return w_, rec_
            return self._detect_wrist(bc.center)
        h = self._capture_head()
        uv, z = self._project_odom(h, face["corners"])
        refit = None
        if np.all(z > 0):
            x0, y0 = np.clip(uv.min(axis=0), 0, None); x1, y1 = uv.max(axis=0)
            r = fit_front_face(h["depth"], h["K"], h["T_root_cam"], [x0, y0, x1, y1], sigma_coeff=h["depth_sigma_coeff"], min_structure_px=self._min_structure_px())
            if r["status"] == "ok":
                T = h["T_odom_root"]
                refit = {"center": apply(T, r["center"]), "normal": rotate(T, r["normal"])}
        plane = refit or {"center": face["center"], "normal": face["normal"]}
        w, rec = detect()
        tried = []
        S = self._c2()["refine"]
        if rec is None:
            for du, dv in S["search_offsets_m"]:
                t0 = self.ws.wrist_target
                shifted = dict(t0, pos=np.asarray(t0["pos"]) + du * np.asarray(face["right"]) + dv * np.asarray(face["up"]))
                saved, self.ws.wrist_target = self.ws.wrist_target, shifted
                self.move_arm("pre")
                self.ws.wrist_target = saved
                w, rec = detect()
                tried.append([du, dv])
                if rec is not None:
                    break
        if rec is None:
            reach = max(abs(v) for o in S["search_offsets_m"] for v in o)
            if n_hidden:
                # wall layout: a label of the box is in the image but partly outside the hand camera's free view (behind
                # the scanner body at a close stand-off, or at the image border): its centre would be that of the
                # visible part; say so instead of "not found"
                return partial(f"the label of the box is only partly in the hand-camera view (hidden by the scanner body "
                               f"or the image border; searched +-{reach * 100:.0f} cm); keeping the current estimate",
                               {"searched": tried, "label_partly_hidden": True})
            return partial(f"barcode not found in the hand-camera image (searched +-{reach * 100:.0f} cm); keeping the head "
                           "estimate", {"searched": tried})
        c = self._pixel_on_face(w, rec["center"], plane)
        if c is None:
            return fail("hand-camera ray parallel to the face plane")
        # consistency gate: the wrist detection must be the planned label, i.e. lie within one label width of the
        # current estimate (head-vs-wrist differences so far: a few cm). 2026-11-06 stack scene: with the arm 36 cm
        # off, the wrist camera detected something else and moved the label from 1.10 m to 1.70 m. The width is the
        # LABEL's extent (the head label box, label_width_m), never the stand-off / symbol width: 2026-11-19 scanroom, a
        # 3 cm cv2 fragment width rejected the correct label 6 cm away four times.
        w_lab = bc.label_width_m or 0.0
        jump = float(np.linalg.norm(np.asarray(c) - np.asarray(bc.center)))
        if w_lab > 0 and jump > w_lab:
            others = [b for b in self.ws.barcodes.values() if b.id != bc.id and b.face_id == bc.face_id and b.center is not None]
            near = min(others, key=lambda b: float(np.linalg.norm(np.asarray(b.center) - np.asarray(c))), default=None)
            return partial(f"wrist detection {jump:.2f} m from the label estimate (> label width {w_lab:.2f} m): not the planned "
                           "label, estimate kept", {"barcode_id": bc.id, "rejected_center_odom": c, "kind": rec["kind"],
                                                    "nearest_other_plan_entry": None if near is None else {
                                                        "id": near.id, "status": near.status, "center_odom": near.center,
                                                        "dist_m": float(np.linalg.norm(np.asarray(near.center) - np.asarray(c)))}})
        bc.center, bc.normal, bc.source = c, np.asarray(plane["normal"], float), "wrist"
        if layout:
            if est["bar_dir"] is not None:                 # the rectified detector's bar direction on the plane
                bc.bar_dir = np.asarray(est["bar_dir"], float)
        elif rec.get("bar_dir") is not None:
            # bar direction from the wrist image (close, sharp): a second pixel along the bars mapped onto the plane
            c2 = self._pixel_on_face(w, np.asarray(rec["center"], float) + 10.0 * np.asarray(rec["bar_dir"], float), plane)
            if c2 is not None and np.linalg.norm(c2 - c) > 1e-6:
                bc.bar_dir = (c2 - c) / np.linalg.norm(c2 - c)
        self._wrist_detection = {"rec": rec, "frame": w, "barcode_id": bc.id}
        width = self._wrist_symbol_width(w, rec, plane, c, w_lab if w_lab > 0 else None)
        if width.get("content"):
            # the hand camera decoded this label: the code itself is decodable (evidence against "unreadable" in scan).
            # Only where the hand camera IS the scanner imager (g1_v1) does it also mean the scanner reads from this pose
            bc.wrist_read = width["content"]
        if width["width_m"] is not None:
            # a wrist width replaces the estimate only if it is a real extent and has at least the support (decoded
            # outlines) of the wrist width it would replace: 2026-11-07 3x3 stack, a single degenerate outline (0.0 m)
            # overwrote two 3-outline measurements of 8.5 cm, the stand-off fell to 0.12 m and the code was not read
            if width["width_m"] <= 0:
                width["rejected"] = "zero extent (degenerate outline)"
            elif bc.width_source == "wrist" and width["n_outlines"] < bc.width_support:
                width["rejected"] = f"{width['n_outlines']} outline(s) < {bc.width_support} of the kept wrist width"
            else:
                bc.width_m, bc.width_source, bc.width_support = width["width_m"], "wrist", int(width["n_outlines"])
        if layout:
            self._store_layout(bc)                         # the measurement, kept in the wall frame
        tr = self.compute_wrist_target(bc.id, refined=True)
        return ok({"barcode_id": bc.id, "center_odom": c, "plane_refit": refit is not None, "kind": rec["kind"], "searched": tried,
                   "symbol_width": width, "width_m": bc.width_m, "width_source": bc.width_source, "offset": tr["data"]["offset"]})

    def _wrist_symbol_width(self, w, rec, plane, anchor=None, max_dist: float | None = None) -> dict:
        """Width of the symbol the hand-camera detection points at, on the face plane: zbar outlines (one per decoding
        rotation, cut ones excluded; the image upsampled by the hand camera's wrist_upscale, as for detection) of the
        decoded symbol nearest the detection centre, each pixel ray intersected with the plane, the longer side of the
        minimum-area rectangle in the plane; median over the outlines. Offline check on the simulator's label textures
        with the scanner imager (code128 0.05-0.10 m, 0.20-0.40 m, 0-80 deg, backed / unbacked): median error +0.5 %,
        96 of 97 within 3 % (offline measurement). width_m None = nothing decoded in full view (the head estimate is
        kept: the stand-off stays at least task2.yaml offset_m, see _standoff).
        Only a symbol of the kept detection counts (a wide hand camera holds neighbouring labels, which zbar may decode
        while the planned one is too small): one of its outlines contains the detection centre, or its centre on the
        plane lies within max_dist (the label width, as refine_with_wrist's consistency gate) of anchor (the detection's
        point on the plane). Otherwise width_m None and no content: neither a neighbour's width nor its read is taken."""
        import cv2
        B = self.cfg["perception"]["barcode"]
        polys = decode_symbol_polygons(w["rgb"], B["wrist_decode_rotation_step_deg"], upscale=B["wrist_upscale"])
        if not polys:
            return {"width_m": None, "reason": "no symbol decoded in full view of the hand camera"}
        ctr = np.asarray(rec["center"], float)

        def belongs(poly):
            hull = cv2.convexHull(np.asarray(poly, np.float32))
            if cv2.pointPolygonTest(hull, (float(ctr[0]), float(ctr[1])), False) >= 0:
                return True
            if anchor is None or max_dist is None:
                return False
            p = self._pixel_on_face(w, np.asarray(poly, float).mean(axis=0), plane)
            return p is not None and float(np.linalg.norm(np.asarray(p) - np.asarray(anchor, float))) <= max_dist

        own = [k for k in polys if any(belongs(p) for p in polys[k])]
        if not own:
            return {"width_m": None, "reason": "no decoded symbol belongs to the detected label (only neighbours decoded)",
                    "decoded_other": sorted(polys)}
        content = min(own, key=lambda k: min(np.linalg.norm(p.mean(axis=0) - ctr) for p in polys[k]))
        n = np.asarray(plane["normal"], float)
        u = np.cross([0.0, 0.0, 1.0], n)
        u /= np.linalg.norm(u)
        v = np.cross(n, u)
        widths = []
        for poly in polys[content]:
            pts = [self._pixel_on_face(w, q, plane) for q in poly]
            if any(p is None for p in pts):
                continue
            P = np.asarray(pts, float) - np.asarray(plane["center"], float)
            (_, _), (a, b), _ = cv2.minAreaRect(np.c_[P @ u, P @ v].astype(np.float32))
            widths.append(float(max(a, b)))
        if not widths:
            return {"width_m": None, "reason": "decoded outline rays parallel to the face plane"}
        return {"width_m": float(np.median(widths)), "content": content, "n_outlines": len(widths),
                "spread_m": float(max(widths) - min(widths))}

    def stereo_check_wrist(self, barcode_id: str | None = None):
        """shift the wrist ~4 cm along the face, triangulate the barcode centre from the two hand-camera
        views (FK poses), compare with refine_with_wrist."""
        bc = self._current(barcode_id)
        if (rf := self._layout_refusal(bc)) is not None:
            return rf
        if bc is None or bc.source != "wrist" or self._wrist_detection is None or self._wrist_detection["barcode_id"] != bc.id:
            return fail("call refine_with_wrist first")
        face = bc.face
        S = self._c2()["stereo"]
        v1 = self._wrist_detection
        t0 = self.ws.wrist_target
        shifted = dict(t0, pos=np.asarray(t0["pos"]) + S["baseline_m"] * np.asarray(face["right"]))
        saved, self.ws.wrist_target = self.ws.wrist_target, shifted
        r = self.move_arm("pre")
        self.ws.wrist_target = saved
        w2, rec2 = self._detect_wrist(bc.center)
        if rec2 is None:
            return partial("barcode not visible in the second hand-camera view")
        rays = []
        for fr, rc in ((v1["frame"], v1["rec"]), (w2, rec2)):
            T = fr["T_odom_root"] @ fr["T_root_cam"]
            rays.append((T[:3, 3], rotate(T, geo.pixel_ray(fr["K"], rc["center"][0], rc["center"][1]))))
        p, gap = geo.triangulate(rays[0][0], rays[0][1], rays[1][0], rays[1][1])
        diff = float(np.linalg.norm(p - bc.center))
        data = {"triangulated_odom": p, "ray_gap_m": gap, "diff_to_refined_m": diff, "baseline_m": S["baseline_m"]}
        if diff > S["max_disagreement_m"]:
            return partial(f"stereo and plane-intersection estimates differ by {diff:.3f} m (> {S['max_disagreement_m']})", data)
        return ok(data)

    def adopt_stereo(self, barcode_id: str | None, point):
        """Trust the motion-stereo centre over the plane intersection, unless it is farther than one
        label width from the current estimate: then the triangulation itself failed (2026-11-07 stack run: with the
        raised arm, stereo-vs-plane differences of 0.10 / 0.20 / 1.6 m were adopted and wrecked the target) and the
        estimate is kept, the same consistency gate as refine_with_wrist."""
        bc = self._current(barcode_id)
        jump = float(np.linalg.norm(np.asarray(point, float) - np.asarray(bc.center, float)))
        if bc.width_m and jump > bc.width_m:
            return partial(f"stereo centre {jump:.2f} m from the estimate (> label width {bc.width_m:.2f} m): triangulation "
                           "rejected, estimate kept", {"barcode_id": bc.id, "rejected_center_odom": point})
        bc.center, bc.source = np.asarray(point, float), "stereo"
        if bc.wall is not None:
            self._store_layout(bc)
        return self.compute_wrist_target(bc.id, refined=True)

    def check_wrist_placement(self):
        if (rf := self._layout_refusal(self._current())) is not None:
            return rf
        C = self._c2()["check"]
        t = self.ws.wrist_target_refined or self.ws.wrist_target
        bc = None if t is None else self.ws.barcodes.get(t.get("barcode_id"))
        if t is None or bc is None:
            return fail("no target")
        T = self._odom() @ self.kin.pose(self._q(), self.cfg["robot"]["wrist_point_frame"])
        p, fwd = T[:3, 3], T[:3, 0]
        n = np.asarray(bc.normal, float)
        dist = float((p - bc.center) @ n)
        v = bc.center - p
        axis_off = float(np.linalg.norm(v - (v @ fwd) * fwd))
        ang = math.degrees(angle_between(fwd, -n))
        L = t.get("long_axis")
        roll = None if L is None else self.ik.roll_error_deg(T[:3, :3], fwd, np.asarray(L, float))
        # roll: image horizontal vs the barcode's long axis, same tolerance as the scan-axis angle
        good = abs(dist - t["offset"]) <= C["dist_tol_m"] and axis_off < C["axis_offset_max_m"] and ang < C["angle_max_deg"] \
            and (roll is None or roll < C["angle_max_deg"])
        data = {"barcode_id": bc.id, "distance_to_plane_m": dist, "target_offset_m": t["offset"], "axis_offset_m": axis_off,
                "angle_deg": ang, "roll_deg": roll}
        if good:
            return ok(data)
        if self.task == 3:
            # task 3: the check measures the placement, the scan decides. Out of tolerance by a few cm the scanner can
            # still read, and scan itself refines and retries; the rule orchestrator always scans after the check.
            # 2026-11-07 (stack4f_L0_0001): the agent_full LLM read "fail" as a stop, re-walked the stance 4 times
            # (axis offset 2.0-2.4 cm vs 2 cm) without ever scanning, and lost the barcode. That holds only while the
            # scanner is at the stand-off with the label in view: 2026-11-07 stack4c_L0_0001, move_arm('final') was not
            # reachable, the arm stayed at the pre-pose (0.35 m vs 0.20 m) and "call scan now" made it scan from there.
            at_standoff = abs(dist - t["offset"]) <= C["dist_tol_m"]
            in_view = self._label_view(t, t["offset"])["in_view"]
            if at_standoff and in_view:
                return partial("placement outside the check tolerances (it measures, it does not decide): call scan now; "
                               "scan reads from a few cm off and refines and retries by itself", data)
            where = (f"the scanner is not at the scanning pose ({dist:.2f} m from the label plane vs the stand-off "
                     f"{t['offset']:.2f} m{'' if in_view else ', label outside the scanner view'})")
            last = getattr(self, "_last_arm", None) or {}
            if last.get("pose") != "final" or last.get("barcode_id") != bc.id:
                # 2026-11-07 stack4c_L0_0001: the LLM called the check right after refine_with_wrist (arm at the
                # pre-pose), was told to re-plan the stance, and re-planned 4 times without ever moving to 'final'
                return fail(f"{where}: the arm was not sent to the final pose yet; call move_arm('final'), then scan", data)
            st = getattr(self, "_stance_reached", None)
            if st is not None and st.get("barcode_id") == bc.id:
                # this stance failed placement for this wrist target: plan_stance will not plan it again for the same
                # target (identity: barcode, target position, face distance, lateral offset, body level)
                self._failed_stances.add((bc.id, tuple(st["target_pos"]), st["face_dist"], st["lateral_offset"],
                                          st["level_index"]))
                data["failed_stance"] = {k: st[k] for k in ("face_dist", "lateral_offset", "base_height", "level_index")}
            return fail(f"{where}: move_arm('final') did not reach it ({last.get('status')}); re-plan the stance "
                        "(move_arm('home'), compute_wrist_target, plan_stance, approach)", data)
        return fail("placement outside the tolerances", data)

    def scan(self):
        """Trigger the scanner (see _scan_once). No read with a wrist target: up to scan.retries_on_no_read retries of
        refine_with_wrist -> move_arm final -> scan inside the tool (2026-11-07: the agent_full LLM skipped the
        prompt's retry after a 2 cm miss; rule mode did it), every step recorded in data["retries"]."""
        if (rf := self._layout_refusal(self._current(), scanning=True)) is not None:
            return rf
        if self.ws.current_barcode_id is not None:
            self._stance_attempted.add(self.ws.current_barcode_id)      # a scan attempt also counts as attempted
        r = self._scan_once()
        tries = []
        for _ in range(self._c2()["scan"]["retries_on_no_read"]):
            if r["status"] == "ok" or self.ws.wrist_target is None or self._current() is None:
                break
            rf = self.refine_with_wrist()
            mv = self.move_arm("final")
            tries.append({"refine_with_wrist": {"status": rf["status"], "reason": rf["reason"]},
                          "move_arm_final": {"status": mv["status"], "reason": mv["reason"],
                                             "residual_m": (mv["data"].get("history") or [{}])[-1].get("pos_err_m")}})
            if mv["status"] == "fail":
                break
            r = self._scan_once()
        if tries:
            r["data"]["retries"] = tries
            r["reason"] = f"{r['reason']} (after {len(tries)} retry: refine_with_wrist + move_arm final)"
        bc = self._current()
        if r["status"] != "ok" and bc is not None and bc.wrist_read and self.ws.wrist_target is not None \
                and self.cfg["robot"]["hand_camera"]["is_scanner"]:
            # the hand camera (here the scanner's own imager) decoded this label during refine_with_wrist from the
            # pre-pose distance: the code is readable from there, so scan from that pose (2026-11-07 stack4c_L0_0001:
            # read at 0.40 m during refine, not at 0.20 m where the 8 cm label was 2.8 cm off the axis, half outside the
            # 6 cm half-view). A separate hand camera (g1_v2 D405: 5 cm behind / 4.5 cm above the window, 87 deg) that
            # decodes from the pre-pose says nothing about what the scanner reads there
            mv = self.move_arm("pre")
            tries.append({"move_arm_pre_where_read": {"status": mv["status"], "reason": mv["reason"], "read_before": bc.wrist_read}})
            if mv["status"] != "fail":
                r = self._scan_once()
            r["data"]["retries"] = tries
        t = self.ws.wrist_target_refined or self.ws.wrist_target
        if r["status"] != "ok" and bc is not None and bc.status == "pending" and t is not None and not r["data"].get("decoded") \
                and not bc.wrist_read and self.task == 3:
            # the label centre was inside the scanner's view at the stand-off and nothing at all decoded in the whole
            # image, after the retries: the label carries no readable code (a damaged / printed-on code, e.g. the
            # shipping stickers printed on the cardbox texture, 2026-11-07 stack4c_L0_0001: 23 such labels). Recorded
            # as "unreadable" with its reason, distinct from "failed" (the robot could not reach / place the scanner).
            # the whole label inside the scanner view (label width: wrist measurement or head estimate), and the scanner
            # at the stand-off (placement-check distance tolerance): a no-read from the pre-pose distance or farther
            # (final pose not reached) says nothing about the label (2026-11-07 stack4c_L0_0001: a real code was
            # recorded unreadable after scanning from 0.35 m when move_arm('final') was not reachable)
            view = self._label_view(t, t["offset"], bc.width_m)
            r["data"]["view"] = view
            at_standoff = abs(view["label_along_m"] - t["offset"]) <= self._c2()["check"]["dist_tol_m"]
            if view["label_in_view"] and at_standoff:
                bc.status = "unreadable"
                bc.fail_reason = (f"no readable code: the whole label was in the scanner view ({view['label_lateral_m']:.3f} m off "
                                  f"the axis at {view['label_along_m']:.2f} m) and nothing decoded after {len(tries)} retry, "
                                  "nor in any earlier hand-camera view")
                r["data"]["barcode_status"] = "unreadable"
                r["reason"] = f"{r['reason']}; label recorded as unreadable (no readable code at close range): move_arm('home'), then call next_barcode"
        return r

    def _scan_once(self):
        """Trigger the scanner. The content nearest the image centre counts for the current barcode; every content is
        counted once (ws.scanned ledger, first read only): a content read before is a duplicate."""
        r = self.robot.scan()
        if r.get("status") not in ("ok", "no_read"):
            return fail(f"scanner: {r.get('reason') or r.get('status')}")
        dec = r.get("decoded", [])
        ws = self.ws
        bc = self._current()
        content = r.get("content") if r.get("status") == "ok" else None
        new, dup_of = False, None
        if content is not None:
            prev = next((e for e in ws.scanned if e["content"] == content), None)
            if prev is None:
                new = True
                ws.scanned.append({"content": content, "barcode_id": None if bc is None else bc.id,
                                   "order_index": None if bc is None else bc.order_index, "t": r.get("timestamp")})
                if bc is not None and bc.status != "scanned":
                    bc.status, bc.content = "scanned", content
            else:
                dup_of = content
                if bc is not None and bc.status == "pending" and prev["barcode_id"] != bc.id:
                    bc.status, bc.content = "duplicate", content
        data = {"barcode_id": None if bc is None else bc.id, "content": content, "decoded": [d["content"] for d in dec],
                "polygons": [{"content": d["content"], "polygon": d["polygon"]} for d in dec if d.get("polygon") is not None],
                "new": new, "duplicate_of": dup_of, "scanned": [e["content"] for e in ws.scanned],
                "barcode_status": None if bc is None else bc.status}
        if ws.scan_plan:
            data["scan_plan"] = ws.scan_plan_export()
        if content is None:
            return fail("no code read", data)
        if dup_of is not None:
            return ok(data, f"{content!r} was read before: duplicate, not counted again")
        return ok(data, f"new code {content!r}")


def horizontal_distance_to_face(p, face) -> float:
    """Horizontal distance from point p to a FINITE box face: the face seen from above is the segment between its
    extreme corners along its horizontal right axis (a point beside the face is measured to the nearest edge)."""
    c = np.asarray(face["center"], float)[:2]
    r = np.asarray(face["right"], float)[:2]
    r = r / np.linalg.norm(r)
    s = (np.asarray(face["corners"], float)[:, :2] - c) @ r
    q = np.asarray(p, float)[:2] - c
    t = float(np.clip(q @ r, s.min(), s.max()))
    return float(np.linalg.norm(q - t * r))


def _poly_signed_distance(P, poly):
    """Signed distance (m) of points P (N x 2) to a convex counter-clockwise polygon (M x 2): negative inside."""
    P = np.asarray(P, float)
    V = np.asarray(poly, float)
    E = np.roll(V, -1, axis=0) - V
    d = np.full(len(P), np.inf)
    inside = np.ones(len(P), bool)
    for a, e in zip(V, E):
        w = P - a
        t = np.clip((w @ e) / (e @ e), 0.0, 1.0)
        d = np.minimum(d, np.linalg.norm(w - t[:, None] * e, axis=1))
        inside &= (e[0] * w[:, 1] - e[1] * w[:, 0]) >= 0
    return np.where(inside, -d, d)


def _face_item(f, T_fo) -> dict:
    """A fitted box face as a target-grid face in the planning frame (T_fo: odom -> planning frame): its extents, which
    of its edges a plan view saw (_note_face_edges) and a key that identifies the fit (capture time + corners), the same
    for the face and every barcode's copy of it."""
    C = (T_fo[:3, :3] @ np.asarray(f["corners"], float).T).T + T_fo[:3, 3]
    return {"key": (float(f["t"]), tuple(np.round(np.asarray(f["corners"], float), 6).ravel())),
            "y_lo": float(C[:, 1].min()), "y_hi": float(C[:, 1].max()),
            "z_lo": float(C[:, 2].min()), "z_hi": float(C[:, 2].max()), "seen": f["edges_seen"]}


def _stance_target_key(pos) -> tuple:
    """Identity of a wrist target position for the failed-stance record: the exact floats (an unchanged estimate gives
    bit-identical numbers through compute_wrist_target, so no tolerance)."""
    return tuple(float(v) for v in np.asarray(pos, float))


def _image_direction(frame, X, d) -> list:
    """Image direction (unit, px) of a step along d (odom) at point X (odom) in a captured frame: the derivative of the
    pinhole projection, no finite step."""
    T = inv_T(frame["T_odom_root"] @ frame["T_root_cam"])
    q = apply(T, X)
    dq = rotate(T, d)
    K = np.asarray(frame["K"], float)
    g = np.array([K[0, 0] * (dq[0] * q[2] - q[0] * dq[2]), K[1, 1] * (dq[1] * q[2] - q[1] * dq[2])]) / q[2] ** 2
    return (g / np.linalg.norm(g)).tolist()


def _label_width(est) -> float:
    """Width of a head label estimate: the longer side of its label rectangle on the face (est corners = the label
    detection box, _labels_on_rectified_face)."""
    return quad_width(est["corners"])


def _same_face_label(E, Q, n, w) -> bool:
    """Two label rectangles (4x3 odom corners) are one physical label when they overlap on the face (strict overlap of
    their extents along the face's horizontal and vertical axes: two physical labels cannot overlap) and lie on the
    same surface (distance of their centres along the face normal n at most w, one label width)."""
    n = np.asarray(n, float); n = n / np.linalg.norm(n)
    u = np.cross([0.0, 0.0, 1.0], n); u /= np.linalg.norm(u)
    v = np.cross(n, u)
    E, Q = np.asarray(E, float), np.asarray(Q, float)
    if abs(float((E.mean(axis=0) - Q.mean(axis=0)) @ n)) > w:
        return False
    eu, ev, qu, qv = E @ u, E @ v, Q @ u, Q @ v
    return bool(eu.min() < qu.max() and qu.min() < eu.max() and ev.min() < qv.max() and qv.min() < ev.max())


def _same_est(a, b, n) -> bool:
    """Two head label estimates of one plan_scans call (same face, or two overlapping face regions) are one label
    (_same_face_label with the larger label width)."""
    return _same_face_label(a["corners"], b["corners"], n, max(_label_width(a), _label_width(b)))


def _same_planned_label(est, b) -> bool:
    """A new detection (est: centre, corners odom) is the planned label b when their label rectangles overlap on the
    face and lie on the same surface (_same_face_label; tolerance along the normal = the larger LABEL width, b's
    label_width_m, never the stand-off / wrist symbol width). Compared with b's planned rectangle (corners_odom, never
    moved) and with that rectangle moved to b's current centre, so a later bad refinement cannot make the same label
    look new (plan updates cannot loop). 2026-11-07: the former "centres within half a label width" test re-added the
    middle label of the 3x3 stack, whose two head estimates differed by 5 cm."""
    if b.corners_odom is None or est.get("corners") is None or b.normal is None:
        return False
    w = max(_label_width(est), b.label_width_m or 0.0)
    rects = [np.asarray(b.corners_odom, float)]
    if b.center is not None and b.plan_center is not None:
        rects.append(rects[0] + (np.asarray(b.center, float) - np.asarray(b.plan_center, float)))
    return any(_same_face_label(est["corners"], Q, b.normal, w) for Q in rects)


def _detection_info(rec) -> dict:
    """Head detection diagnostics of a plan entry (logged with the plan, never shown to the LLM; no decision uses them):
    detector kind, whether cv2 found the symbol inside the label, the detector score and the symbol quad in head px
    (the label box is the entry's corners_px)."""
    sym = rec.get("symbol_corners")
    return {"kind": rec.get("kind"), "refined_by_cv2": rec.get("refined_by_cv2"), "score": rec.get("score"),
            "symbol_corners_px": None if sym is None else np.asarray(sym, float).tolist()}


def _entry_diagnostics(b) -> dict:
    """Log-only record of a plan entry (with_diagnostics): its id, the label width (the same-label tests and the wrist
    gate use it) and the head detection diagnostics."""
    return {"id": b.id, "label_width_m": b.label_width_m, **(b.detection or {})}


def _dropped_report(dropped) -> list:
    """Duplicate candidates of a plan_scans call: [(candidate, what it matched: a kept candidate or a plan entry
    (BarcodeEstimate), rule)] -> log records with the matched plan entry id (a kept candidate's once it is planned)."""
    out = []
    for c, k, why in dropped:
        if isinstance(k, BarcodeEstimate):
            matched, mc = k.id, k.center
        else:
            matched, mc = k.get("bid"), k["est"]["center"]
        out.append({"box_id": c["box_id"], "center_odom": c["est"]["center"], "label_corners_px": c["rec"]["corners"],
                    "score": c["rec"].get("score"), "matched": matched, "matched_center_odom": mc, "rule": why})
    return out
