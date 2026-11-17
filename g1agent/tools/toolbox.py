"""Deterministic tools. Every tool returns {"status": ok|fail|partial, "data": {...},
"reason": str}; numbers are rounded to 3 decimals and no large arrays are returned (images only via look_*).

Frames: perception happens in the root frame at capture time and is stored in the odometry frame (WorldState);
camera poses come from the joint state through pinocchio FK (head <-> wrist via kinematics, not odometry).
No ground truth is used anywhere in this module (tests/test_no_gt_leak.py).

Task 3 (scan every barcode): plan_scans localises every label on every fitted face and fixes the order (scan_order.py:
columns left to right, top to bottom inside a column); next_barcode walks the plan and clears the per-barcode slots;
every barcode tool defaults to ws.current_barcode_id and to the face copy stored with the barcode; scan dedupes by
decoded content into the ws.scanned ledger.
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
from ..perception.plane_fit import fit_front_face
from .scan_order import adaptive_standoff, column_order, hfov_from_intrinsics, order_items_in_frame, point_in_quad, quad_width


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


# fields of a move_root_relative reply that may be logged / shown to the agent: the simulator reply also carries
# ground-truth displacement / error fields (evaluation only), which must never reach the harness logs or the LLM
MOVE_LOG_KEYS = ("status", "reason", "elapsed_s", "requested", "odom_displacement")


def move_summary(r: dict) -> dict:
    return {k: r[k] for k in MOVE_LOG_KEYS if k in r}


class Toolbox:
    def __init__(self, robot, box_detector=None, device: str | None = None):
        from ..perception.box_detector import BoxDetector
        self.robot = robot
        self.cfg = config.load_all()
        self.kin = Kinematics()
        self.ik = ArmIK(self.kin)
        self.det = box_detector or BoxDetector(device)
        self.info = robot.get_robot_info()
        # the robot behind the interface must be the one the configs describe ($G1_ROBOT, common/config.py); a server
        # without the key is a G1 server from before the robot selection existed
        served = self.info.get("robot_model", "g1")
        if served != self.cfg["robot"]["model"]:
            raise RuntimeError(f"the robot server runs {served!r}, this harness is configured for "
                               f"{self.cfg['robot']['model']!r} (set {config.ROBOT_ENV} / --robot on both sides)")
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
        # scanner horizontal field of view from the wrist intrinsics (32.8 deg for the DS2208-like imager); None
        # when the robot has no scanner camera (the adaptive stand-off then fails loudly)
        self.wrist_hfov = None if K is None or size is None else hfov_from_intrinsics(float(np.asarray(K, float)[0, 0]), float(size[1]))
        # body (squat) command: None = the walker has none (stance planning stays at the standing height)
        bc = self.info.get("body_command")
        self.body_cmd = bc
        self.base_height_cmd = None if bc is None else float(bc["height_default"])
        self._body_model = None
        # task parameter: scan every vertical face of the stack (front, right, back, left) instead of the front only
        # (set by the runner from the task instruction; the default keeps the single-face behaviour)
        self.multi_face = False

    # ================================================================== body (squat / stand)
    def _body(self):
        """BodyModel (tools/body.py), built on first use: squat calibration + the robot's collision geometry. A robot
        without a body command has none, unless its robot file asks for the standing clearance model
        (body.standing_clearance, T800: no squat levels, the clearance checks of standing stances)."""
        B = self.cfg["robot"]["body"]
        standing_only = self.body_cmd is None and B.get("standing_clearance", False)
        if self.body_cmd is None and not standing_only:
            return None
        if self._body_model is None:
            import json
            import os
            from .body import BodyModel
            if standing_only:
                cal = None
            else:
                if B["calibration"] is None:
                    raise ValueError("the robot has a body command but the robot file has no body.calibration")
                cal = json.load(open(os.path.join(config.ROOT, B["calibration"])))["levels"]
            self._body_model = BodyModel(self.kin, self.robot.get_robot_geometry(), cal, self.side, B.get("links"))
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
        r = self.robot.set_body_pose(float(height), None, B["ramp_s"], B["settle_s"])
        if r["status"] != "ok":
            return fail(f"body command: {r['status']}: {r['reason']}")
        self.base_height_cmd = float(height)
        return ok({"height_cmd": float(height), "standing": self._standing(), "pelvis_z": r.get("pelvis_z"),
                   "tilt_deg": r.get("tilt_deg")}, r["reason"])

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
        o = self.robot.get_head_obs()
        q = self._q()
        self.head = {"rgb": o["rgb"], "depth": o["depth"], "K": np.asarray(o["K"], float),
                     "T_root_cam": self.kin.pose(q, self.cfg["robot"]["head_cam_frame"]),
                     "T_odom_root": self._odom(), "t": o["timestamp"]}
        return self.head

    def _capture_wrist(self):
        o = self.robot.get_wrist_obs(self.side)
        if o.get("status") == "error":
            raise RuntimeError(o["reason"])
        q = self._q()
        self.wrist = {"rgb": o["rgb"], "K": np.asarray(o["K"], float), "T_root_cam": self.kin.pose(q, self.cfg["robot"]["wrist_cam_frame"]),
                      "T_odom_root": self._odom(), "t": o["timestamp"], "q": q}
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
        return ok({"timestamp": w["t"]}, "wrist (scanner) camera image attached"), w["rgb"]

    def detect_boxes(self, prompt: str = "cardboard box"):
        h = self._capture_head()
        prev_target_face = self._face(self.ws.target_box_id) if self.ws.target_box_id else None
        dets = self.det.detect(h["rgb"], prompt=f"a {prompt}." if not prompt.startswith("a ") else prompt)
        self.ws.boxes = {f"box{i}": BoxEstimate(f"box{i}", d["bbox"], d["score"], h["t"]) for i, d in enumerate(dets)}
        self.ws.target_box_id = None
        if self.ws.target_selector is not None:
            self._reselect(prev_target_face)
        if not dets:
            return fail("no box detected in the head image", {"n": 0})
        return ok({"n": len(dets), "boxes": [{"id": f"box{i}", "bbox": d["bbox"], "score": d["score"]} for i, d in enumerate(dets)],
                   "order": "left to right in the image (robot's left first)", "target_box_id": self.ws.target_box_id})

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
        r = fit_front_face(h["depth"], h["K"], h["T_root_cam"], self.ws.boxes[box_id].bbox)
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

    def compute_obs_pose(self, box_id: str | None = None):
        box_id = box_id or self.ws.target_box_id
        faces = self._faces(box_id)
        if not faces:
            return fail("no front face for this box (call fit_front_face)")
        c1 = self._c1()
        center, nh = self._faces_reference(faces)
        corners = np.vstack([np.asarray(f["corners"], float) for f in faces])
        yaw = math.atan2(-nh[1], -nh[0])
        T_root_cam = self.head["T_root_cam"] if self.head is not None else self.kin.pose(self._q(), self.cfg["robot"]["head_cam_frame"])
        z_root = self._odom()[2, 3]
        W, H = self.info["head_size"][1], self.info["head_size"][0]
        K = np.asarray(self.info["head_K"], float)
        n_steps = int(round((c1["d_obs_range"][1] - c1["d_obs_range"][0]) / c1["d_obs_step"]))
        chosen, fits = None, False
        cut = self._faces_cut_by_border(faces) if self.cfg["robot"].get("obs_back_off_cut_faces", False) else None
        if cut:
            # A face fitted from a view that cuts it at the image border is only a part of the face (T800 at 0.6 m
            # from a wall: the level head camera sees the wall down to ~0.9 m only), so its extent cannot place the
            # whole face in the image. Robot file obs_back_off_cut_faces (T800, user decision 2026-11-10: back up,
            # head at 0): plan at the far end of d_obs_range, the distance at which a face standing on the floor fits
            # the image (robot_t800.yaml task_overrides); the caller refits there and plans again on the whole face.
            d = c1["d_obs_range"][1]
            chosen = (center + d * nh, d)
        for k in range(0 if cut else n_steps + 1):
            d = c1["d_obs_range"][0] + k * c1["d_obs_step"]
            p = center + d * nh
            T_odom_cam = T_from_xy_yaw(p[0], p[1], yaw, z_root) @ T_root_cam
            uv, z = geo.project(K, inv_T(T_odom_cam), corners)
            if np.all(geo.in_image(uv, z, W, H, self.cfg["perception"]["fov_margin_frac"])):
                chosen, fits = (p, d), True
                break
        if chosen is None:
            d = c1["d_obs_range"][1]
            chosen = (center + d * nh, d)
        p, d = chosen
        self.ws.root_target = {"pos": [float(p[0]), float(p[1])], "yaw": yaw, "d_obs": d, "box_id": box_id, "fits": fits,
                               "n_faces": len(faces)}
        if cut:
            self.ws.root_target["cut_by_image_border"] = cut
        # The faces this pose was planned with (full extents, normally seen from far) are frozen for the approach:
        # close-range refits see the faces cut by the image edge, so their extents (and hence the union reference
        # point, d_obs and the in-view test) are no longer the boxes' (2026-11-06: the goal moved by ~0.15 m between
        # fine moves and the in-view test failed at the planned pose). Refits then only correct the plane position.
        self._obs_faces = {b: {k: np.asarray(v, float).copy() for k, v in self._face(b).items() if k in ("center", "normal", "corners")}
                           for b in self._box_ids(box_id) if self._face(b) is not None}
        self._marker("root_target", "red", "root", [p[0], p[1], 0.0], yaw=yaw)
        data = {"root_target_odom": self.ws.root_target}
        if fits:
            return ok(data)
        if cut:
            return partial(f"the fitted face is cut by the head image border ({', '.join(cut)}): only part of it is "
                           f"known, so the robot backs up to the farthest observation distance {d:.2f} m; after "
                           "approach, call detect_boxes, fit_front_face and compute_obs_pose again (whole face in view) "
                           "and approach the new pose", data)
        return partial("face does not fit the image with the margin at any d in range; using the upper bound", data)

    def _faces_cut_by_border(self, faces) -> list:
        """Image borders ("bottom", "top", "left", "right") beyond which a face fitted from the current head capture
        may continue: one of its corners lies outside the image region inside the fov margin (perception.yaml
        fov_margin_frac, the same margin that defines "whole face in view"). Faces fitted from another capture
        are not judged (their image is not this one)."""
        h = self.head
        if h is None:
            return []
        cur = [f for f in faces if f.get("t") == h["t"]]
        if not cur:
            return []
        H, W = h["rgb"].shape[:2]
        m = self.cfg["perception"]["fov_margin_frac"]
        uv, z = self._project_odom(h, np.vstack([np.asarray(f["corners"], float) for f in cur]))
        out = []
        for name, bad in (("bottom", uv[:, 1] > H * (1 - m)), ("top", uv[:, 1] < H * m), ("left", uv[:, 0] < W * m),
                          ("right", uv[:, 0] > W * (1 - m)), ("behind", z <= 0)):
            if np.any(bad):
                out.append(name)
        return out

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
        if self._standing():                            # the stance it may have reached
            return None
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
            # no body command (T800): base_height_cmd is None and the reply has no height
            return ok(data, "at the stance" + ("" if self.base_height_cmd is None
                                               else f", body height command {self.base_height_cmd:.2f} m"))
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
        """Label detection (head px) -> centre, bar direction, corners (odom) and width on the face plane."""
        c = self._pixel_on_face(frame, rec["center"], face)
        if c is None:
            return None
        p2 = self._pixel_on_face(frame, rec["center"] + 10 * np.asarray(rec["bar_dir"]), face)
        bar = None if p2 is None else (p2 - c) / (np.linalg.norm(p2 - c) + 1e-12)
        corners = [self._pixel_on_face(frame, q, face) for q in np.asarray(rec["corners"], float)]
        corners = None if any(q is None for q in corners) else np.asarray(corners)
        return {"center": c, "bar_dir": bar, "corners": corners, "width": None if corners is None else quad_width(corners)}

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
        bc.corners_odom, bc.width_m = est["corners"], est["width"]
        return ok({"barcode_id": barcode_id, "center_odom": est["center"], "normal_odom": bc.normal, "bar_dir_odom": est["bar_dir"],
                   "width_m": est["width"]})

    # ------------------------------------------------------------------ task 3: scan plan
    def plan_scans(self, _update: bool | None = None):
        """Localise every label on every fitted face in one head image and fix the scan order (scan_order.py). Once the
        plan is being executed, a call from another viewpoint UPDATES it instead: labels already in the plan (same
        place on the face) are kept as they are, labels seen for the first time are added as pending and ordered with
        the other pending ones by the same rule in the planning frame; scanned / failed entries never move."""
        ws = self.ws
        r = self.detect_boxes()
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
        face_entries = [b for b in ws.scan_plan if ws.barcodes[b].face_id == cur]
        updating = _update if _update is not None else bool(face_entries)
        per_face, dropped = [], []
        cam = (h["T_odom_root"] @ h["T_root_cam"])[:3, 3]
        for bid, face in faces.items():
            uv, z = self._project_odom(h, face["corners"])
            if np.any(z <= 0):
                skipped[bid] = "face behind the camera"
                continue
            kept = []                                   # detector output is sorted by score
            for l, est in self._labels_on_rectified_face(h, face):
                if any(_same_label(l, k["rec"]) for k in kept):
                    continue                            # a second detection of a label already kept on this face
                kept.append({"box_id": bid, "face": face, "rec": l, "est": est, "range": float(np.linalg.norm(est["center"] - cam))})
            per_face += kept
        # the same label found from two overlapping face regions: it lies on the surface nearest to the camera along
        # its pixel ray (the visible one)
        cands = []
        for c in sorted(per_face, key=lambda c: c["range"]):
            if not any(_same_label(c["rec"], k["rec"]) for k in cands):
                cands.append(c)
        if updating:
            return self._merge_into_plan(cands, h, boxes, skipped)
        multi = ws.stack is not None and ws.stack.get("multi_face")
        if not cands and not (multi and faces):
            return fail("no barcode label localised on any fitted face", {"boxes": boxes, "n_faces": len(faces), "skipped": skipped,
                                                                         "dropped_labels": dropped})
        # multi-face tour with this side's faces fitted but no label on them: a valid outcome (not every side of a stack
        # carries labels); the side is planned empty (after the low look below, which may still find low labels) and
        # the tour goes on with next_barcode
        items = order_items_in_frame([{"key": i, "corners": c["est"]["corners"], "center": c["est"]["center"]}
                                      for i, c in enumerate(cands)], h["T_odom_root"])
        order = column_order(items)
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
                                               status="pending", order_index=len(ws.scan_plan),
                                               plan_center=np.asarray(est["center"], float).copy(), face_id=cur)
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
        out = [{"id": b, "box_id": ws.barcodes[b].box_id, "order_index": ws.barcodes[b].order_index,
                "corners_px": ws.barcodes[b].corners_px, "center_odom": ws.barcodes[b].center,
                "width_m": ws.barcodes[b].width_m, "standoff_m": self._standoff(ws.barcodes[b]), "status": "pending",
                "face_id": cur}
               for b in ws.scan_plan if ws.barcodes[b].face_id == cur]
        return ok({"n": len(out), "barcodes": out, "boxes": boxes, "skipped": skipped, "dropped_labels": dropped,
                   "low_view": low, "face_id": cur, "face": self.TOUR[cur], "first_for_face": True,
                   "order_rule": "columns of overlapping lateral label extents, left to right as seen facing this side; "
                                 "top to bottom inside a column"},
                  f"{len(out)} barcode(s) planned on the {self.TOUR[cur]} side; call next_barcode" if out else
                  f"no barcode label on the {self.TOUR[cur]} side ({len(faces)} face(s) fitted"
                  f"{', also looked from a squat' if low and low.get('height_cmd') is not None else ''}): this side "
                  f"has nothing to scan; call next_barcode to move on")

    def _low_view_update(self):
        """Second look at the lower part of the faces from a squat, right after the first plan (task 3, robot standing
        with the arm home). The head camera looks ~47 deg down from ~1.2 m: low labels on a wide / tall face are far
        and oblique (2026-11-07 3x3 stack: 1-2 of the bottom-row labels missed standing, found after squatting). The
        robot looks from every calibrated squat level whose head camera sees the faces' bottom corners (inside the image
        with the fov margin) more finely than standing (focal length / depth, worst bottom corner); none -> no low look.
        New labels are merged into the plan after each look (same rule as a plan update), then the robot stands up."""
        body = self._body()
        if body is None or not body.levels or not self._standing() or not self.arm_at_home \
                or not (self.ws.scan_plan or (self.ws.stack or {}).get("planned")):
            return None
        faces = [b.face for b in self.ws.boxes.values() if b.face is not None
                 and (self.ws.stack is None or self._face_id(b.face["normal"]) == self.ws.stack["current"])]
        if not faces:
            return None
        bottoms = []
        for f in faces:
            C = np.asarray(f["corners"], float)
            zmin = C[:, 2].min()
            bottoms += [c for c in C if c[2] < zmin + 1e-6] or [C[np.argmin(C[:, 2])]]
        bottoms = np.asarray(bottoms)
        T0 = self._odom()
        q0 = self._q()
        yaw = yaw_of(T0)
        R_yaw = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
        K = np.asarray(self.info["head_K"], float)
        H, W = self.info["head_size"]
        mfrac = self.cfg["perception"]["fov_margin_frac"]

        def score(lv):
            q_lv = q0 if lv is body.levels[0] else body.q_at(q0, lv)
            sh = np.zeros(2) if lv is body.levels[0] else R_yaw @ body.foot_shift(lv, body.levels[0], q0)
            Tr = body.T_root(T0[0, 3] + sh[0], T0[1, 3] + sh[1], yaw, lv)
            Tc = Tr @ self.kin.pose(q_lv, self.cfg["robot"]["head_cam_frame"])
            uv, z = geo.project(K, inv_T(Tc), bottoms)
            if not np.all(geo.in_image(uv, z, W, H, mfrac)):
                return None
            return float(K[0, 0] / z.max())             # px per metre at the worst (farthest) bottom corner

        # every calibrated squat level that sees the faces' bottom corners more finely than standing, highest first.
        # One "best" level is not enough: the open-vocabulary label detector's recall is view-dependent and not monotonic
        # in the height (2026-11-07 cardbox stack, 11 levels x 2 sides: the deepest squat, finest on the bottom, found
        # none of the 4 lower labels, the 0.40-0.60 levels found all 4, 0.35 only 1); the union over the levels found
        # every label of both sides (offline measurement).
        s_stand = score(body.levels[0])
        looks = [(sc, lv) for lv in body.levels[1:] if (sc := score(lv)) is not None and (s_stand is None or sc > s_stand)]
        if not looks:
            return {"height_cmd": None, "reason": "no squat sees the faces' bottom better than standing", "added": []}
        done, added = [], []
        for sc, lv in looks:
            r = self.set_base_height(lv["height_cmd"])
            if r["status"] != "ok":
                done.append({"height_cmd": lv["height_cmd"], "status": f"squat failed: {r['reason']}"})
                break
            u = self.plan_scans(_update=True)
            new = [a["id"] for a in (u["data"].get("added") or [])] if u["status"] == "ok" else []
            added += new
            done.append({"height_cmd": lv["height_cmd"], "px_per_m_bottom": round(sc, 1), "added": new})
        st = self.set_base_height(None)
        if st["status"] != "ok":
            raise RuntimeError(f"standing up after the low look failed: {st['reason']}")
        return {"height_cmd": [d["height_cmd"] for d in done], "looks": done,
                "px_per_m_bottom_standing": None if s_stand is None else round(s_stand, 1), "added": added}

    def _merge_into_plan(self, cands, h, boxes, skipped):
        """plan_scans update (see there). A detection is a label already in the plan when its centre lies within half
        a label width (the larger of the two estimates) of a planned label's centre on the same face plane."""
        ws = self.ws
        cur = 0 if ws.stack is None else ws.stack["current"]
        added = []
        for c in cands:
            est = c["est"]
            same = any(_same_planned_label(est, b) for b in ws.barcodes.values() if b.face_id == cur)
            if same:
                continue
            k = len(ws.barcodes)
            while f"bc{k}" in ws.barcodes:
                k += 1
            bid = f"bc{k}"
            ws.barcodes[bid] = BarcodeEstimate(bid, c["box_id"], est["center"], np.asarray(c["face"]["normal"], float), est["bar_dir"],
                                               "head", h["t"], pixel=np.asarray(c["rec"]["center"], float).tolist(),
                                               face=dict(c["face"]), corners_odom=est["corners"],
                                               corners_px=np.asarray(c["rec"]["corners"], float).tolist(), width_m=est["width"],
                                               status="pending", plan_center=np.asarray(est["center"], float).copy(), face_id=cur)
            ws.scan_plan.append(bid)
            added.append(bid)
        if added:
            # pending entries of the current side (old and new) re-ordered by the plan's rule in that side's planning
            # frame; the rest keep their places
            pend = [b for b in ws.scan_plan if ws.barcodes[b].status == "pending" and b != ws.current_barcode_id
                    and ws.barcodes[b].face_id == cur]
            done = [b for b in ws.scan_plan if b not in pend]
            items = order_items_in_frame([{"key": b, "corners": ws.barcodes[b].corners_odom, "center": ws.barcodes[b].center}
                                          for b in pend], ws.scan_ref_Ts.get(cur, ws.scan_ref_T))
            ws.scan_plan = done + list(column_order(items))
            for i, b in enumerate(ws.scan_plan):
                ws.barcodes[b].order_index = i
        out = [{"id": b, "box_id": ws.barcodes[b].box_id, "order_index": ws.barcodes[b].order_index,
                "corners_px": ws.barcodes[b].corners_px, "center_odom": ws.barcodes[b].center,
                "width_m": ws.barcodes[b].width_m, "status": "pending"} for b in added]
        return ok({"updated": True, "added": out, "n_seen": len(cands), "n_planned": len(ws.scan_plan), "boxes": boxes,
                   "skipped": skipped, "scan_plan": ws.scan_plan_export()},
                  f"plan update: {len(added)} new label(s) added" if added else "plan update: no new label")

    def _labels_on_rectified_face(self, h, face):
        """Labels on a fitted face, detected on the face rectified to a fronto-parallel view: the plane maps the head
        image to face coordinates (homography), sampled at the finest source sampling over the face (focal length /
        nearest face depth: no source detail dropped). Without it the steep head view (camera ~47 deg down) squashes
        labels on the lower part of a tall face into ~20 x 5 px slivers the detector misses (2026-11-06 floor stack:
        1 of 3 labels found from the observation pose, 3 of 3 rectified). Returns
        [(rec, est)]: rec = detection in head-image px (centre / corners projected back, for overlays and duplicate
        tests), est = centre / bar direction / corners (odom) / width on the plane."""
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
            corners = np.array([to3d(q) for q in np.asarray(l["corners"], float)])
            center = to3d(np.asarray(l["center"], float))
            bd = l.get("bar_dir")
            bar = None
            if bd is not None:
                bar = float(bd[0]) * u - float(bd[1]) * v
                bar /= np.linalg.norm(bar)
            cpx, _ = self._project_odom(h, corners)
            ctr, _ = self._project_odom(h, center[None])
            rec = dict(l, corners=cpx, center=ctr[0], rectified={"px_per_m": s, "size": [Wr, Hr]})
            out.append((rec, {"center": center, "bar_dir": bar, "corners": corners, "width": quad_width(corners)}))
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
        update = self._plan_update_from_here() if self.task == 3 and prev is not None else None
        nxt = next((b for b in ws.scan_plan if ws.barcodes[b].status == "pending" and ws.barcodes[b].face_id == cur), None)
        ws.current_barcode_id = nxt
        base = {"closed": closed, "n_planned": len(ws.scan_plan), "scanned": [e["content"] for e in ws.scanned],
                "scan_plan": ws.scan_plan_export(), "face_id": cur, "face": self.TOUR[cur], "plan_update": update}
        if nxt is None:
            nf = self._next_face()
            if nf is not None:
                return ok(dict(base, done=False, face_done=True, next_face=self.TOUR[nf], next_face_id=nf),
                          f"every barcode of the {self.TOUR[cur]} side has been handled: call go_to_next_face "
                          f"(next: {self.TOUR[nf]}), then plan_scans")
            return ok(dict(base, done=True), "every planned barcode has been handled")
        bc = ws.barcodes[nxt]
        remaining = sum(ws.barcodes[b].status == "pending" for b in ws.scan_plan) - 1
        return ok(dict(base, done=False, barcode_id=nxt, index=bc.order_index, box_id=bc.box_id, remaining=remaining,
                       width_m=bc.width_m, standoff_m=self._standoff(bc)), f"current barcode: {nxt}")

    def _plan_update_from_here(self) -> dict:
        """Look again from where the robot is now (closer to the side than the observation pose) before choosing the
        next barcode: label detection on busy box textures is view-dependent (2026-11-07 stack4c_L0_0001: a label
        missed from the observation pose was never added because the LLM skipped the prompt's plan_scans)."""
        u = self.plan_scans(_update=True)
        return {"status": u["status"], "reason": u["reason"],
                "added": [a["id"] for a in (u["data"].get("added") or [])] if u["status"] == "ok" else []}

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
        if self.wrist_hfov is None:
            raise RuntimeError("wrist camera intrinsics unknown (robot info wrist_K / wrist_size): no adaptive stand-off")
        d = adaptive_standoff(bc.width_m, self.wrist_hfov, S["fill_frac"], S["min_m"], S["max_m"])
        if bc.width_source != "wrist":
            # a head width may push the scanner farther than the validated stand-off (task2.yaml offset_m), never
            # closer: head label boxes come out both too large (DINO box, +20 %) and too small (a partial box on the
            # rectified face: 3.5 / 4.9 cm for 8 cm codes, 2026-11-07 agent_full run -> clamped to 0.12 m, no read).
            # Only the wrist measurement (the decoded symbol) may bring the scanner closer than that.
            d = max(d, float(c2["offset_m"]))
        return d

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
        return ok({"barcode_id": bc.id, "wrist_target_odom": {"pos": t["pos"], "forward": t["forward"]}, "offset": off,
                   "offset_ignored": ignored,
                   "pre_offset": self._pre_offset(off), "label_width_m": bc.width_m,
                   "note": "roll about the forward axis is not constrained (the scanner reads at any orientation)"})

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
        if S3["mode"] == "adaptive" and bc.width_source == "head":
            finals = [float(S3["min_m"])]
            pre_dists = list(np.arange(t["offset"] + gap0, float(S3["min_m"]) - 1e-9, -S["step"]))
        else:
            finals = [t["offset"]]
            pre_dists = list(np.arange(t["offset"] + gap0, t["offset"] - 1e-9, -S["step"]))
        body = self._body()
        # body heights to try, standing first: the walker's measured squat levels (robot.yaml body.calibration), or
        # only the current height for a walker without a body command (T800: its standing clearance model still
        # checks the stances, robot_t800.yaml body.standing_clearance). The first (highest) level with a feasible
        # stance wins: squatting costs time and balance margin.
        levels = body.levels if body is not None and body.levels else [None]
        margin = self.cfg["robot"]["body"]["clearance_margin_m"]
        R_yaw = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
        rejected = {"reach": 0, "ik": 0, "face_clearance": 0, "self_clearance": 0, "not_robust": 0}
        best, chosen, fallback, cands = None, None, None, []
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
                        jobs.append({"Tc": Tc, "q_lv": q_lv, "t": t, "finals": finals, "pre_dists": pre_dists, "stall": stall,
                                     "seed": [int(self.ik.cfg["seed"]), li, len(jobs)]})
                        meta.append((p, pw, d, l, Tc))
                # reach prefilter + IK of the required poses + posture, per grid point (tools/stance_eval.py), in the
                # planning process pool when configured (identical results: per-candidate IK seeds)
                for (p, pw, d, l, Tc), ev in zip(meta, self._eval_stance_jobs(jobs, body)):
                    if ev["status"] != "ok":
                        rejected[ev["status"]] += 1
                        continue
                    sols, gap, m, man = ev["sols"], ev["gap"], ev["margin"], ev["man"]
                    travel = float(np.hypot(pw[0] - T0[0, 3], pw[1] - T0[1, 3]))
                    # arm posture: the position manipulability sqrt(det(Jp Jp^T)) of the scanner-tip Jacobian over the
                    # 7 arm joints, smallest over the required poses (Yoshikawa). It falls towards both singular
                    # ends: the nearly straight arm (2026-11-06: a 0.70 m stance chosen by joint margin stayed 9 cm
                    # off) and the folded arm (then chosen by "least extension": elbow up beside the head, joints at
                    # their limits, unnatural and the arm correction diverged) -> the most manipulable pose wins
                    cost = (-gap, -man, travel)              # the preferred pre gap first, then posture, walk
                    cands.append((cost, p, pw, d, l, m, travel, None, gap, Tc, q_lv, sols, man))
                # The cost does not depend on the clearances or on robustness, so the candidates are taken in cost order
                # and the (expensive) checks run only until the first one passes: the same stance as checking every
                # candidate and picking the cheapest passing one (2026-11-07: the arm-vs-legs check of every
                # IK-feasible candidate was 72 % of a 66 s plan_stance).
                # Robustness: the walk ends within the robot's move tolerance of the stance (robot.yaml body
                # stance_robust_*), so the stance is re-checked displaced by it (2026-11-06: a stance at the joint-limit
                # bound became unreachable by 3 mm after the real walk + squat).
                for cand in sorted(cands, key=lambda c_: c_[0]):
                    cost, p, pw, d, l, m, travel, _, gap, Tc, q_lv, sols, man = cand
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
                        tip = np.asarray(t["pos"], float) + (finals[0] - t["offset"]) * (-np.asarray(t["forward"], float))
                        clear = {"face_m": fc, "arm_leg_m": sc, "arm_extension": round(float(np.linalg.norm(tip - sh) / body.arm_reach), 2),
                                 "manipulability": round(man, 4)}
                    cand = (cost, p, pw, d, l, m, travel, clear, gap, Tc, q_lv)
                    if fallback is None:
                        fallback = (cand, lv)
                    rb = self._stance_robust(t, Tc, q_lv, finals, t["offset"] + gap)
                    rejected["not_robust"] += not rb
                    if rb:
                        best, chosen = cand, lv
                        break
                cands = []
                if best is not None:
                    break
        finally:
            self.ik.cfg["restarts"] = restarts
        robust = best is not None
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
        _, p, pw, d, l, m, travel, clear, gap, _, _ = best
        self.ws.stance_target = {"pos": [float(pw[0]), float(pw[1])], "yaw": yaw, "face_dist": float(d), "lateral_offset": float(l),
                                 "barcode_id": bc.id, "travel_m": travel,
                                 "base_height": None if chosen is None else float(chosen["height_cmd"]),
                                 "pelvis_pos": [float(p[0]), float(p[1])],
                                 "pelvis_z": None if chosen is None else float(chosen["pelvis_z"]), "pre_gap": gap}
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
                self._plan_pool = ProcessPoolExecutor(max_workers=nw, mp_context=mp.get_context("spawn"),
                                                      initializer=SE.init_worker, initargs=(shoulder, reach))
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
        """Every required pose (finals + the chosen pre-pose distance) is still reachable with the stance displaced by
        the robot's move tolerance: +-pos along the stance's x and y, +-yaw about the pelvis."""
        B = self.cfg["robot"]["body"]
        dp, dy = B["stance_robust_pos_m"], math.radians(B["stance_robust_yaw_deg"])
        stall = (int(self.ik.cfg["plan_stall_window"]), float(self.ik.cfg["plan_stall_eps_m"]))
        for ex, ey, eyaw in ((dp, 0, 0), (-dp, 0, 0), (0, dp, 0), (0, -dp, 0), (0, 0, dy), (0, 0, -dy)):
            Tp = Tc @ T_from_xy_yaw(ex, ey, eyaw, 0.0)
            for off in list(finals) + [pre_dist]:
                pr, dr = self._target_in_root(t, Tp, off)
                if not self.ik.solve(q_lv, pr, dr, stall, self._roll_in_root(t, Tp))["ok"]:
                    return False
        return True

    def _known_faces(self):
        """Every fitted face the harness knows: the boxes of the last detection and the face copies of all barcodes
        (box ids are re-assigned by detect_boxes, the barcode copies survive it)."""
        faces = [b.face for b in self.ws.boxes.values() if b.face is not None]
        st = self.ws.stack
        cur = None if st is None or not st.get("multi_face") else st["current"]
        # multi-side stack: barcode copies of other sides were fitted elsewhere and are in odometry as it drifted since;
        # those sides are covered by the re-registered stack footprint instead (plan_stance checks both)
        faces += [b.face for b in self.ws.barcodes.values() if b.face is not None and (cur is None or b.face_id == cur)]
        return faces

    def move_arm(self, pose: str = "pre", side: str | None = None):
        """See _move_arm; records the last arm command (pose, barcode, result) for the placement check's guidance."""
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
        p_goal, d_goal = self._target_in_root(t, offset=off)
        p_cmd = p_goal.copy()
        hist = []
        best = None                                   # (error, q_arm) of the best command sent so far
        for it in range(1 + A["correction_iters"]):
            q = self._q()
            sol = self.ik.solve(q, p_cmd, d_goal, roll_axis=self._roll_in_root(t))
            if not sol["ok"]:
                if best is not None:                  # a correction left the reachable set: keep the best pose reached
                    self.robot.set_arm_joints(side, best[1], A["duration_s"])
                    self.robot.stand(A["settle_s"])
                    hist.append({"reverted_to_best": True, "pos_err_m": best[0], "angle_err_deg": best[2]})
                    break
                return fail(f"IK: target not reachable (residual {sol['pos_err']:.3f} m / {sol['angle_err_deg']:.1f} deg)", {"iters": it})
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

    def _label_view(self, t, off, label_width: float | None = None) -> dict:
        """Where the wrist target's label centre lies relative to the wrist camera's view cone from the current arm pose
        (wrist point frame, x = scanner axis): lateral distance from the axis vs the view half-width at that range.
        With label_width, "label_in_view" also requires the whole label (centre offset + half width) inside the view."""
        if self.wrist_hfov is None:
            raise RuntimeError("wrist camera intrinsics unknown (robot info wrist_K / wrist_size): no view check")
        T = self.kin.pose(self._q(), self.cfg["robot"]["wrist_point_frame"])
        p_goal, d_goal = self._target_in_root(t, offset=off)
        v = (p_goal + d_goal * off) - T[:3, 3]                  # wrist point -> label centre
        fwd = T[:3, 0] / np.linalg.norm(T[:3, 0])
        along = float(v @ fwd)
        lat = float(np.linalg.norm(v - along * fwd))
        half = along * math.tan(self.wrist_hfov / 2)
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

    def _detect_wrist(self):
        w = self._capture_wrist()
        recs = detect_barcodes(w["rgb"], None, cfg=dict(self.cfg["perception"]["barcode"], upscale=self.cfg["perception"]["barcode"]["wrist_upscale"]))
        if not recs:
            H, W = w["rgb"].shape[:2]
            recs = detect_label_dino(w["rgb"], [0, 0, W, H], self.det)
        if not recs:
            return w, None
        H, W = w["rgb"].shape[:2]
        recs.sort(key=lambda r: np.hypot(r["center"][0] - W / 2, r["center"][1] - H / 2))
        return w, recs[0]

    def refine_with_wrist(self, barcode_id: str | None = None):
        """re-fit the face plane from the head, transform it to the wrist camera through FK, detect the
        barcode in the wrist image, intersect its centre ray with the plane, update the wrist target."""
        if (nc := self._not_current(barcode_id)) is not None:
            return fail(nc)
        bc = self._current(barcode_id)
        if bc is None or bc.center is None:
            return fail("no barcode estimate")
        if self.ws.wrist_target is None or self.ws.wrist_target.get("barcode_id") != bc.id:
            return fail(f"no wrist target for {bc.id} (call compute_wrist_target)")
        face = bc.face
        h = self._capture_head()
        uv, z = self._project_odom(h, face["corners"])
        refit = None
        if np.all(z > 0):
            x0, y0 = np.clip(uv.min(axis=0), 0, None); x1, y1 = uv.max(axis=0)
            r = fit_front_face(h["depth"], h["K"], h["T_root_cam"], [x0, y0, x1, y1])
            if r["status"] == "ok":
                T = h["T_odom_root"]
                refit = {"center": apply(T, r["center"]), "normal": rotate(T, r["normal"])}
        plane = refit or {"center": face["center"], "normal": face["normal"]}
        w, rec = self._detect_wrist()
        tried = []
        if rec is None:
            S = self._c2()["refine"]
            for du, dv in S["search_offsets_m"]:
                t0 = self.ws.wrist_target
                shifted = dict(t0, pos=np.asarray(t0["pos"]) + du * np.asarray(face["right"]) + dv * np.asarray(face["up"]))
                saved, self.ws.wrist_target = self.ws.wrist_target, shifted
                self.move_arm("pre")
                self.ws.wrist_target = saved
                w, rec = self._detect_wrist()
                tried.append([du, dv])
                if rec is not None:
                    break
        if rec is None:
            return partial("barcode not found in the wrist image (searched +-5 cm); keeping the head estimate", {"searched": tried})
        c = self._pixel_on_face(w, rec["center"], plane)
        if c is None:
            return fail("wrist ray parallel to the face plane")
        # consistency gate: the wrist detection must be the planned label, i.e. lie within one label width of the
        # current estimate (head-vs-wrist differences so far: a few cm). 2026-11-06 stack scene: with the arm 36 cm
        # off, the wrist camera detected something else and moved the label from 1.10 m to 1.70 m.
        w_lab = max(bc.width_m or 0.0, (self.ws.barcodes[bc.id].width_m if bc.id in self.ws.barcodes else 0.0) or 0.0)
        jump = float(np.linalg.norm(np.asarray(c) - np.asarray(bc.center)))
        if w_lab > 0 and jump > w_lab:
            return partial(f"wrist detection {jump:.2f} m from the label estimate (> label width {w_lab:.2f} m): not the planned "
                           "label, estimate kept", {"barcode_id": bc.id, "rejected_center_odom": c, "kind": rec["kind"]})
        bc.center, bc.normal, bc.source = c, np.asarray(plane["normal"], float), "wrist"
        if rec.get("bar_dir") is not None:
            # bar direction from the wrist image (close, sharp): a second pixel along the bars mapped onto the plane
            c2 = self._pixel_on_face(w, np.asarray(rec["center"], float) + 10.0 * np.asarray(rec["bar_dir"], float), plane)
            if c2 is not None and np.linalg.norm(c2 - c) > 1e-6:
                bc.bar_dir = (c2 - c) / np.linalg.norm(c2 - c)
        self._wrist_detection = {"rec": rec, "frame": w, "barcode_id": bc.id}
        width = self._wrist_symbol_width(w, rec, plane)
        if width.get("content"):
            # this label is readable from the current arm pose (the wrist camera is the scanner's camera)
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
        tr = self.compute_wrist_target(bc.id, refined=True)
        return ok({"barcode_id": bc.id, "center_odom": c, "plane_refit": refit is not None, "kind": rec["kind"], "searched": tried,
                   "symbol_width": width, "width_m": bc.width_m, "width_source": bc.width_source, "offset": tr["data"]["offset"]})

    def _wrist_symbol_width(self, w, rec, plane) -> dict:
        """Width of the symbol the wrist detection points at, on the face plane: zbar outlines (one per decoding
        rotation, cut ones excluded) of the decoded symbol nearest the detection centre, each pixel ray intersected
        with the plane, the longer side of the minimum-area rectangle in the plane; median over the outlines. Offline
        check on the simulator's label textures (code128 0.05-0.10 m, 0.20-0.40 m, 0-80 deg, backed / unbacked):
        median error +0.5 %, 96 of 97 within 3 % (offline measurement). width_m None = nothing decoded
        in full view (the head estimate is kept)."""
        import cv2
        polys = decode_symbol_polygons(w["rgb"], self.cfg["perception"]["barcode"]["wrist_decode_rotation_step_deg"])
        if not polys:
            return {"width_m": None, "reason": "no symbol decoded in full view of the wrist camera"}
        ctr = np.asarray(rec["center"], float)
        content = min(polys, key=lambda k: min(np.linalg.norm(p.mean(axis=0) - ctr) for p in polys[k]))
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
        """shift the wrist ~4 cm along the face, triangulate the barcode centre from the two wrist
        views (FK poses), compare with refine_with_wrist."""
        bc = self._current(barcode_id)
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
        w2, rec2 = self._detect_wrist()
        if rec2 is None:
            return partial("barcode not visible in the second wrist view")
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
        return self.compute_wrist_target(bc.id, refined=True)

    def check_wrist_placement(self):
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
            return fail(f"{where}: move_arm('final') did not reach it ({last.get('status')}); re-plan the stance "
                        "(move_arm('home'), compute_wrist_target, plan_stance, approach)", data)
        return fail("placement outside the tolerances", data)

    def scan(self):
        """Trigger the scanner (see _scan_once). No read with a wrist target: up to scan.retries_on_no_read retries of
        refine_with_wrist -> move_arm final -> scan inside the tool (2026-11-07: the agent_full LLM skipped the
        prompt's retry after a 2 cm miss; rule mode did it), every step recorded in data["retries"]."""
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
        if r["status"] != "ok" and bc is not None and bc.wrist_read and self.ws.wrist_target is not None:
            # the wrist camera decoded this label during refine_with_wrist from the pre-pose distance: the code is
            # readable from there, so scan from that pose (2026-11-07 stack4c_L0_0001: read at 0.40 m during refine,
            # not at 0.20 m where the 8 cm label was 2.8 cm off the axis, half outside the 6 cm half-view)
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
                                  "nor in any earlier wrist view")
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


def _same_planned_label(est, b) -> bool:
    """A new detection (est: centre, corners odom) is the planned label b when their label rectangles overlap on the
    face (two physical labels cannot overlap) and they lie on the same surface (distance along the face normal below
    one label width). Compared with b's planned rectangle (corners_odom, never moved) and with that rectangle moved to
    b's current centre, so a later bad refinement cannot make the same label look new (plan updates cannot loop).
    2026-11-07: the former "centres within half a label width" test re-added the middle label of the 3x3 stack, whose
    two head estimates differed by 5 cm."""
    if b.corners_odom is None or est.get("corners") is None or b.normal is None:
        return False
    n = np.asarray(b.normal, float); n = n / np.linalg.norm(n)
    u = np.cross([0.0, 0.0, 1.0], n); u /= np.linalg.norm(u)
    v = np.cross(n, u)
    w = max(est.get("width") or 0.0, b.width_m or 0.0)
    E = np.asarray(est["corners"], float)
    rects = [np.asarray(b.corners_odom, float)]
    if b.center is not None and b.plan_center is not None:
        rects.append(rects[0] + (np.asarray(b.center, float) - np.asarray(b.plan_center, float)))
    for Q in rects:
        if abs(float((E.mean(axis=0) - Q.mean(axis=0)) @ n)) > w:
            continue
        eu, ev, qu, qv = E @ u, E @ v, Q @ u, Q @ v
        if eu.min() < qu.max() and qu.min() < eu.max() and ev.min() < qv.max() and qv.min() < ev.max():
            return True
    return False


def _same_label(a, b) -> bool:
    """Two label detections (head px) are the same physical label when either centre lies inside the other's quad."""
    return point_in_quad(a["center"], b["corners"]) or point_in_quad(b["center"], a["corners"])
