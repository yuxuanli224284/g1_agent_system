"""Composite episode video: pure numpy / cv2 / PIL, no simulator import, so every piece here
is unit-testable on CPU. The sim server (g1agent/robot/sim_server.py) gathers the camera buffers and the state and calls
these functions; nothing here touches the robot's cameras or the USD markers.

Pieces
- Fixed-camera framing: fit_fixed_camera() places a camera of given orientation and FOV so that a set of world points
  (robot start, boxes, GT targets, from the scenario layout) fills the image with a margin (closed-form frustum fit).
- OverlayState: the latest agent events (annotate / stage / robot-port scans) and which pixel overlays belong to which
  sim time. A pixel-space detection (head boxes / labels, pyzbar polygons) is drawn only on frames of the sim time it was
  computed at (frozen-state "hold" frames), never on later frames where the camera has moved.
- frame_metrics(): root vs GT task-1 obs pose and scanner tip vs the GT target point of the current barcode, in the
  target-face frame (n = outward face normal, towards the robot; l = n x z = the robot's left; z = up), plus the
  scanned count (distinct decoded contents that are GT barcodes).
- Compositor: draws the 1920x1080 layout (viz.yaml recording.panels) with 2-D projected markers and PIL (CJK) text.
- H264Writer: raw RGB piped to an ffmpeg binary (libx264, yuv420p, fragmented MP4 so a killed run stays playable).
"""
from __future__ import annotations

import math
import os
import subprocess
from fractions import Fraction

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..eval.scan_order import FACE_NAMES, FACE_TOUR, assign_faces

Z_UP = np.array([0.0, 0.0, 1.0])


# ---------------------------------------------------------------------------------------------------------- geometry
def project(K, T_world_cam, pts):
    """(uv Nx2, z N) of world points through a pinhole camera (ROS optical frame: x right, y down, z forward)."""
    P = np.asarray(pts, float).reshape(-1, 3)
    R, t = np.asarray(T_world_cam, float)[:3, :3], np.asarray(T_world_cam, float)[:3, 3]
    pc = (P - t) @ R
    z = pc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = (np.asarray(K, float) @ pc.T).T
        uv = uv[:, :2] / uv[:, 2:3]
    return uv, z


def look_rotation(forward, up=Z_UP) -> np.ndarray:
    """R_world_cam (columns: x right, y down, z forward) for a camera looking along `forward`, image up ~ world `up`."""
    z = np.asarray(forward, float) / np.linalg.norm(forward)
    x = np.cross(z, np.asarray(up, float))
    if np.linalg.norm(x) < 1e-9:
        raise ValueError("camera forward is parallel to the up vector")
    x /= np.linalg.norm(x)
    return np.stack([x, np.cross(z, x), z], axis=1)


def intrinsics(hfov_deg: float, width: int, height: int) -> np.ndarray:
    f = width / 2.0 / math.tan(math.radians(hfov_deg) / 2.0)
    return np.array([[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]])


def fit_fixed_camera(points, forward, hfov_deg: float, width: int, height: int, margin_frac: float):
    """Camera with orientation look_rotation(forward) and the given hFOV, placed as close as possible to the points
    while every point projects inside the image with a border of margin_frac of the width / height.

    Closed form: in the rotated frame q = R^T p, a camera at c sees q inside the frustum iff |q_x - c_x| <= a (q_z - c_z)
    and |q_y - c_y| <= b (q_z - c_z), a / b = the half-FOV tangents shrunk by the margin. With L = max(q_x - a q_z),
    U = min(q_x + a q_z) (same for y with b: M, N) the deepest feasible c_z is min((U - L) / 2a, (N - M) / 2b), and the
    centred c_x = (L + U) / 2, c_y = (M + N) / 2. Returns (T_world_cam 4x4, K 3x3)."""
    if not 0.0 <= margin_frac < 0.5:
        raise ValueError(f"margin_frac must be in [0, 0.5), got {margin_frac}")
    P = np.asarray(points, float).reshape(-1, 3)
    R = look_rotation(forward)
    K = intrinsics(hfov_deg, width, height)
    a = (width / 2.0) / K[0, 0] * (1.0 - 2.0 * margin_frac)
    b = (height / 2.0) / K[1, 1] * (1.0 - 2.0 * margin_frac)
    q = P @ R
    L, U = np.max(q[:, 0] - a * q[:, 2]), np.min(q[:, 0] + a * q[:, 2])
    M, N = np.max(q[:, 1] - b * q[:, 2]), np.min(q[:, 1] + b * q[:, 2])
    cz = min((U - L) / (2 * a), (N - M) / (2 * b))
    c_local = np.array([(L + U) / 2.0, (M + N) / 2.0, cz])
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, R @ c_local
    return T, K


def view_direction(azimuth_deg: float, elevation_deg: float, axis0, axis90) -> np.ndarray:
    """Unit vector from the content towards the camera: azimuth measured in the horizontal plane from axis0 towards
    axis90 (both horizontal, orthogonal), elevation above the horizontal."""
    az, el = math.radians(azimuth_deg), math.radians(elevation_deg)
    h = math.cos(az) * np.asarray(axis0, float) + math.sin(az) * np.asarray(axis90, float)
    return math.cos(el) * h + math.sin(el) * Z_UP


def face_axes(normal):
    """Target-face frame: n = horizontal outward normal (towards the robot), l = n x z (the robot's left when it faces
    the face), z = up."""
    n = np.array([normal[0], normal[1], 0.0], float)
    n /= np.linalg.norm(n)
    return n, np.cross(n, Z_UP), Z_UP.copy()


def box_corners(pos, size, quat_wxyz) -> np.ndarray:
    """8 world corners of a box (layout size [d (x), w (y), h (z)], yaw-only quaternion)."""
    w, x, y, z = quat_wxyz
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    c, s = math.cos(yaw), math.sin(yaw)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    h = np.asarray(size, float) / 2
    loc = np.array([[sx * h[0], sy * h[1], sz * h[2]] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    return loc @ Rz.T + np.asarray(pos, float)


def label_corners(center, normal, up, size) -> np.ndarray:
    """4 world corners of a planar label; right = up x normal (the face's right axis seen from in front)."""
    c, n, u = (np.asarray(v, float) for v in (center, normal, up))
    r = np.cross(u, n)
    r /= np.linalg.norm(r)
    w, h = float(size[0]) / 2, float(size[1]) / 2
    return np.array([c + sx * w * r + sy * h * u for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])


def gt_targets(gt: dict, standoff_m) -> list[dict]:
    """GT scanner target point of every non-distractor barcode: centre + d_gt * normal. standoff_m(gt_barcode) -> d_gt
    is the evaluator's GT stand-off rule (sim_server.gt_standoff_fn), passed in so the METRICS panel and the episode
    evaluation use one definition (symbol width, task3.yaml standoff constants, scanner hFOV)."""
    out = []
    for bc in gt["barcodes"]:
        if bc["distractor"]:
            continue
        d = float(standoff_m(bc))
        c, n = np.asarray(bc["center"], float), np.asarray(bc["normal"], float)
        out.append({"content": bc["content"], "box_id": bc["box_id"], "center": c, "normal": n, "up": np.asarray(bc["up"], float),
                    "size": bc["size"], "d": d, "pos": c + d * n})
    return out


def place_points(pts_root, x: float, y: float, yaw: float, z: float) -> np.ndarray:
    """Root-frame points of the robot body placed at a ground pose (x, y, yaw) with root height z."""
    c, s = math.cos(yaw), math.sin(yaw)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    return np.asarray(pts_root, float) @ Rz.T + [x, y, z]


def framing(gt: dict, targets: list[dict], robot_pts_root, root_z: float, cfg: dict):
    """Fixed recording-camera directions and content points from the scenario (viz.yaml recording.framing).
    robot_pts_root: the robot's body points in its root frame (the sim's collision-model points at the start pose),
    placed at the robot start and at the GT obs pose with root height root_z, so the whole body stays in frame.
    MAIN: azimuth in the robot-start frame (0 = robot forward, 90 = robot left); content = the robot at its start and at
    the GT obs pose, every box and the GT target points.
    SIDE: azimuth in the target-face frame (0 = l, the robot's left side of the boxes; 90 = n, towards the robot);
    content = the boxes' front faces (corners facing the robot), the GT target points and the robot at the GT obs pose,
    i.e. the robot-to-face gap. Returns {name: (forward, points)}."""
    ri = gt["robot_init"]
    fwd0 = np.array([math.cos(ri["yaw"]), math.sin(ri["yaw"]), 0.0])
    left0 = np.cross(Z_UP, fwd0)
    ob = gt["task1_gt_obs_pose"]
    at_start = place_points(robot_pts_root, ri["pos"][0], ri["pos"][1], ri["yaw"], root_z)
    at_obs = place_points(robot_pts_root, ob["pos"][0], ob["pos"][1], ob["yaw"], root_z)
    tg = [t["pos"] for t in targets]
    n, l, _ = face_axes(gt["target_face"]["normal"])
    boxes = [box_corners(b["pos"], b["size"], b["quat_wxyz"]) for b in gt["boxes"]]
    fronts = [c[np.isclose(c @ n, (c @ n).max())] for c in boxes]          # corners of the face towards the robot
    pts = {"main": np.vstack([at_start, at_obs, *boxes, *tg]),
           "side": np.vstack([at_obs, *fronts, *tg])}
    dirs = {"main": view_direction(cfg["main"]["azimuth_deg"], cfg["main"]["elevation_deg"], fwd0, left0),
            "side": _side_direction(n, cfg)}
    return {k: (-dirs[k], pts[k]) for k in ("main", "side")}


def _side_direction(normal, cfg: dict) -> np.ndarray:
    """SIDE view direction for a face: viz.yaml side azimuth / elevation in that face's frame (0 = l, 90 = n)."""
    n, l, _ = face_axes(normal)
    return view_direction(cfg["side"]["azimuth_deg"], cfg["side"]["elevation_deg"], l, n)


def labelled_tour(gt: dict, targets: list[dict]):
    """Faces of a multi-face stack (GT "faces", I2) carrying GT targets, in tour order, with their targets:
    [(face entry, [targets])]. A target belongs to the face whose normal best matches its own (as in the evaluator).
    Empty for a GT without "faces"."""
    faces = sorted(gt.get("faces") or [], key=lambda f: FACE_TOUR.index(f["face"]))
    if not faces:
        return []
    idx = assign_faces([t["normal"] for t in targets], [f["normal"] for f in faces]) if targets else []
    out = []
    for i, f in enumerate(faces):
        mine = [t for t, j in zip(targets, idx) if j == i]
        if mine:
            out.append((f, mine))
    return out


def face_framing(gt: dict, targets: list[dict], robot_pts_root, root_z: float, cfg: dict, face_idx: int):
    """Recording cameras while the robot works on tour face face_idx (I3 face_start) of a multi-face stack; None when
    fewer than two faces carry GT targets (single-face scene: the record_start framing stays).
    MAIN: diagonal overview between the current face and the next labelled face of the tour (cyclic): view direction
    = the counter-clockwise bisector from n_cur to n_next (the tour turns counter-clockwise, so for adjacent faces this
    is the n_cur + n_next diagonal, for opposite faces the side the robot walks along), at the viz.yaml main
    elevation; content = every box, the GT targets of both faces and the robot at both faces' GT obs poses.
    SIDE: the record_start side rule applied to the current face: its corners (the exposed box faces on that side),
    its GT targets and the robot at its GT obs pose, seen from its l side. Returns {name: (forward, points)}."""
    tour = labelled_tour(gt, targets)
    if len(tour) < 2:
        return None
    name = FACE_TOUR[face_idx]
    cur = next((f for f in gt["faces"] if f["face"] == name), None)
    if cur is None:
        raise ValueError(f"face_start for face {face_idx} ({name}): the stack has no exposed {name} side in the GT")
    cur_tg = next((tg for f, tg in tour if f["face"] == name), [])
    after = [p for p in tour if FACE_TOUR.index(p[0]["face"]) > face_idx] + [p for p in tour if FACE_TOUR.index(p[0]["face"]) < face_idx]
    nxt, nxt_tg = after[0] if after else (cur, [])

    def robot_at(f):
        ob = f["gt_obs_pose"]
        return place_points(robot_pts_root, ob["pos"][0], ob["pos"][1], ob["yaw"], root_z)

    n_cur, _, _ = face_axes(cur["normal"])
    n_nxt, _, _ = face_axes(nxt["normal"])
    ccw = math.atan2(float(np.cross(n_cur, n_nxt)[2]), float(n_cur @ n_nxt)) % (2 * math.pi)    # in [0, 2 pi)
    main_dir = view_direction(math.degrees(ccw / 2), cfg["main"]["elevation_deg"], n_cur, np.cross(Z_UP, n_cur))
    boxes = [box_corners(b["pos"], b["size"], b["quat_wxyz"]) for b in gt["boxes"]]
    pair_tg = [t["pos"] for t in cur_tg + (nxt_tg if nxt is not cur else [])]
    main_pts = np.vstack([robot_at(cur)] + ([robot_at(nxt)] if nxt is not cur else []) + boxes + pair_tg)
    side_pts = np.vstack([robot_at(cur), np.asarray(cur["corners"], float).reshape(-1, 3)] + [t["pos"] for t in cur_tg])
    return {"main": (-main_dir, main_pts), "side": (-_side_direction(cur["normal"], cfg), side_pts)}


# ---------------------------------------------------------------------------------------------------------- metrics
def nearest_label_to_target(targets: list[dict], p) -> dict | None:
    """GT label whose outward normal axis (centre + t n, t >= 0) passes closest to a harness target point p."""
    best, bd = None, float("inf")
    for t in targets:
        v = np.asarray(p, float) - t["center"]
        along = max(float(v @ t["normal"]), 0.0)
        d = float(np.linalg.norm(v - along * t["normal"]))
        if d < bd:
            best, bd = t, d
    return best


def frame_metrics(gt: dict, fs: dict, targets: list[dict], current: dict | None, scan_log: list[dict],
                  task1_end: dict | None = None) -> dict:
    """Numbers for the METRICS panel. Root and tip differences in the target-face frame (see module doc), actual minus
    GT. The root is the articulation root body (G1: pelvis); the GT obs pose is a ground pose (x, y, yaw), so the root
    comparison uses the pelvis' ground projection and has no z component.
    task1_end: the stage state {"t_sim", "actual": {"T_world_root"}} once the task-1 phase has ended. The robot then
    walks on to its scan stances, so a live root-vs-obs-pose difference no longer measures anything: the root numbers
    are frozen at the task-1 end (out["root"]["frozen_t"] = its sim time)."""
    n, l, z = face_axes(gt["target_face"]["normal"])
    ob = gt["task1_gt_obs_pose"]
    Tr = np.asarray((task1_end["actual"] if task1_end is not None else fs)["T_world_root"], float)
    d = Tr[:3, 3] - np.asarray(ob["pos"], float)
    yaw = math.atan2(Tr[1, 0], Tr[0, 0])
    out = {"t_sim": float(fs["t_sim"]),
           "root": {"dn": float(d @ n), "dl": float(d @ l), "dxy": float(math.hypot(d[0], d[1])),
                    "dyaw_deg": math.degrees((yaw - ob["yaw"] + math.pi) % (2 * math.pi) - math.pi),
                    "frozen_t": None if task1_end is None else float(task1_end["t_sim"])},
           "tip": None, "min_gap": fs.get("min_gap"), "pelvis_z": float(np.asarray(fs["T_world_root"], float)[2, 3]),
           "height_cmd": fs.get("height_cmd")}
    if current is not None and "T_world_tip" in fs:
        e = np.asarray(fs["T_world_tip"], float)[:3, 3] - current["pos"]
        cn, cl, cz = face_axes(current["normal"])
        out["tip"] = {"content": current["content"], "d_gt": current["d"], "dn": float(e @ cn), "dl": float(e @ cl),
                      "dz": float(e @ cz), "norm": float(np.linalg.norm(e))}
    gt_contents = [t["content"] for t in targets]
    read = []
    for s in scan_log:
        c = s.get("content")
        if s.get("status") == "ok" and c is not None and c not in read:
            read.append(c)
    out["scanned"] = {"k": sum(c in gt_contents for c in read), "n": len(gt_contents), "contents": read,
                      "not_gt": [c for c in read if c not in gt_contents]}
    return out


def _dashed(img, a, b, col, width: int, dash: int):
    """Dashed 2-D line from pixel a to pixel b (dash and gap both `dash` px)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    L = float(np.linalg.norm(b - a))
    if L < 1:
        return
    u = (b - a) / L
    for t0 in np.arange(0.0, L, 2 * dash):
        p0, p1 = a + u * t0, a + u * min(t0 + dash, L)
        cv2.line(img, tuple(int(round(v)) for v in p0), tuple(int(round(v)) for v in p1), col, width, cv2.LINE_AA)


# ---------------------------------------------------------------------------------------------------------- events
class OverlayState:
    """Latest agent / stage / scan events of one episode (sim server side). """

    def __init__(self):
        self.llm = None              # latest llm event
        self.tool = None             # latest tool event
        self.stage = None            # {"name", "info", "t_sim"}
        self.head = None             # {"boxes", "labels", "t_sim"} latest head detections (pixel space)
        self.n_boxes = None          # boxes seen by the latest detect (persistent text)
        self.box_order = []          # latest head boxes [{"id", "bbox"}]
        self.target_box = None
        self.scan_plan = None
        self.current_barcode = None
        self.current_since = None    # sim time at which current_barcode last changed
        self.scans = []              # robot-port scan() log entries (sim server), get_scan_log
        self.face = None             # multi-face tour (I3): {"idx", "n_planned", "t_sim", "statuses" (None until face_end)}

    def on_event(self, ev: dict, t_sim: float) -> str | None:
        """Store an annotate() event. Returns the hold kind the server should write frozen frames for: "llm" for an LLM
        reply, "detection" for a tool result carrying head-camera pixel detections (only valid on this frame)."""
        if ev.get("kind") == "llm":
            self.llm = dict(ev, t_sim=t_sim)
            return "llm"
        if ev.get("kind") != "tool":
            raise ValueError(f"annotate: unknown event kind {ev.get('kind')!r}")
        self.tool = dict(ev, t_sim=t_sim)
        ov = ev.get("overlay") or {}
        hold = None
        if ov.get("head_boxes") is not None or ov.get("head_labels") is not None:
            self.head = {"boxes": ov.get("head_boxes") or [], "labels": ov.get("head_labels") or [], "t_sim": t_sim}
            if ov.get("head_boxes") is not None:
                self.n_boxes = len(ov["head_boxes"])
                self.box_order = list(ov["head_boxes"])
            hold = "detection" if (self.head["boxes"] or self.head["labels"]) else None
        if "target_box" in ov:
            self.target_box = ov["target_box"]
        if "scan_plan" in ov:
            self.scan_plan = ov["scan_plan"]
        if "current_barcode" in ov and ov["current_barcode"] != self.current_barcode:
            self.current_barcode, self.current_since = ov["current_barcode"], t_sim
        return hold

    def on_stage(self, name: str, info, t_sim: float):
        self.stage = {"name": name, "info": info, "t_sim": t_sim}
        if name == "face_start":
            self.face = {"idx": int(info["face"]), "n_planned": info.get("n_planned"), "t_sim": t_sim, "statuses": None}
        elif name == "face_end" and self.face is not None and self.face["idx"] == int(info["face"]):
            self.face["statuses"] = dict(info.get("statuses") or {})

    def on_scan(self, entry: dict):
        self.scans.append(entry)

    # ---------------------------------------------------------------- derived
    def head_detections_at(self, t_sim: float):
        return self.head if self.head is not None and self.head["t_sim"] == t_sim else None

    def last_scan(self):
        return self.scans[-1] if self.scans else None

    def face_text(self) -> str | None:
        """'face 2/4: right (-y) ...' of the tour face being worked on (None before any face_start)."""
        f = self.face
        if f is None:
            return None
        txt = f"面 face {f['idx'] + 1}/{len(FACE_TOUR)}: {FACE_NAMES[f['idx']]} ({FACE_TOUR[f['idx']]})"
        if f["n_planned"] is not None:
            txt += f"   计划 planned {f['n_planned']}"
        if f["statuses"] is not None:
            st = list(f["statuses"].values())
            txt += f"   完成 done: {sum(s == 'scanned' for s in st)}/{len(st)} scanned"
        return txt

    def target_text(self) -> str:
        """'<box id> / left|right' from the latest head boxes (left = smallest bbox centre u in the head image)."""
        if self.target_box is None:
            return "-"
        us = {b["id"]: (b["bbox"][0] + b["bbox"][2]) / 2 for b in self.box_order}
        if self.target_box not in us or len(us) < 2:
            return str(self.target_box)
        rank = sorted(us, key=us.get).index(self.target_box)
        side = "left" if rank == 0 else "right" if rank == len(us) - 1 else f"#{rank + 1} from left"
        return f"{self.target_box} / {side}"

    def current_target(self, targets: list[dict], harness_targets: list[dict]):
        """GT target of the barcode the agent works on: the scan_plan entry of current_barcode mapped by its decoded
        content, else the GT label nearest to the harness wrist target set after current_barcode was selected.
        harness_targets: [{"pos" (world), "t_set" (sim time)}], most refined first. None when unknown."""
        if self.current_barcode is not None:
            for e in self.scan_plan or []:
                if e.get("id") == self.current_barcode and e.get("content"):
                    for t in targets:
                        if t["content"] == e["content"]:
                            return t
        since = self.current_since if self.current_barcode is not None else None
        for h in harness_targets:
            if since is None or h["t_set"] >= since:
                return nearest_label_to_target(targets, h["pos"])
        return None


# ---------------------------------------------------------------------------------------------------------- drawing
def _rgb255(c):
    return tuple(int(round(255 * v)) for v in c)


def fit_into(img: np.ndarray, w: int, h: int):
    """Resize to fit (w, h) keeping the aspect ratio, centred; returns (panel, scale, ox, oy)."""
    ih, iw = img.shape[:2]
    s = min(w / iw, h / ih)
    nw, nh = int(round(iw * s)), int(round(ih * s))
    out = np.zeros((h, w, 3), np.uint8)
    ox, oy = (w - nw) // 2, (h - nh) // 2
    out[oy:oy + nh, ox:ox + nw] = img if (nw, nh) == (iw, ih) else cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    return out, s, ox, oy


def _font(path: str, face: str, size: int):
    """Face `face` (family name) of a font file or collection (.ttc)."""
    i = 0
    while True:
        try:
            f = ImageFont.truetype(path, size, index=i)
        except OSError:          # index past the last face of the collection (index 0 failing = unreadable file: raise)
            if i == 0:
                raise
            break
        i += 1
        if " ".join(f.getname()[:1]) == face:
            return f
    raise ValueError(f"font face {face!r} not found in {path}")


class Compositor:
    """Draws one composite frame. cfg = viz.yaml recording; colors / sizes = viz.yaml colors / sizes (markers)."""

    def __init__(self, cfg: dict, colors: dict, sizes: dict):
        self.cfg, self.sizes = cfg, sizes
        self.W, self.H = cfg["canvas"]
        self.panels = {k: tuple(v) for k, v in cfg["panels"].items()}
        for k, (x, y, w, h) in self.panels.items():
            if x < 0 or y < 0 or x + w > self.W or y + h > self.H:
                raise ValueError(f"panel {k} {self.panels[k]} outside the {self.W}x{self.H} canvas")
        fc = cfg["font"]
        self.font = _font(fc["path"], fc["face"], fc["size_px"])
        self.small = _font(fc["path"], fc["face"], fc["small_px"])
        self.col = {k: _rgb255(v) for k, v in colors.items()}
        self.ov = cfg["overlay"]

    # ---------------------------------------------------------------- 3-D markers on a fixed camera panel
    def _draw_markers(self, img, K, T, s, ox, oy, markers):
        lw, rmin = self.ov["line_px"], self.ov["min_dot_px"]

        def px(p):
            uv, z = project(K, T, [p])
            if z[0] <= 0:
                return None, None
            return (int(round(uv[0, 0] * s + ox)), int(round(uv[0, 1] * s + oy))), float(z[0])

        for m in markers:
            col = self.col[m["color"]]
            p = np.asarray(m["pos"], float)
            if m["kind"] == "root":
                # ground pose: disc + heading arrow on the floor (the plane the GT obs pose lives in). A marker with
                # "body_z" (the actual root body, G1 pelvis) also gets a dot at that height joined to the floor disc
                # by a dashed vertical line, so the floor disc reads as the pelvis' projection, not as a foot.
                if m.get("body_z") is not None:
                    top = np.array([p[0], p[1], float(m["body_z"])])
                    ct, zt = px(top)
                    cb, _ = px(np.array([p[0], p[1], self.sizes["root_height"]]))
                    if ct is not None and cb is not None:
                        _dashed(img, cb, ct, col, lw, self.ov["dash_px"])
                        cv2.circle(img, ct, max(rmin, int(round(K[0, 0] * s * self.sizes["pelvis_radius"] / zt))), col, -1,
                                   cv2.LINE_AA)
                        cv2.circle(img, ct, max(rmin, int(round(K[0, 0] * s * self.sizes["pelvis_radius"] / zt))), (0, 0, 0),
                                   1, cv2.LINE_AA)
                        if m.get("label"):
                            cv2.putText(img, m["label"], (ct[0] + 8, ct[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3,
                                        cv2.LINE_AA)
                            cv2.putText(img, m["label"], (ct[0] + 8, ct[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1,
                                        cv2.LINE_AA)
                p = np.array([p[0], p[1], self.sizes["root_height"]])
                tip = p + self.sizes["root_arrow"] * np.array([math.cos(m["yaw"]), math.sin(m["yaw"]), 0.0])
                r_w = self.sizes["root_radius"]
            elif m["kind"] == "pose":
                d = np.asarray(m["dir"], float)
                tip = p + self.sizes["pose_arrow"] * d / np.linalg.norm(d)
                r_w = self.sizes["point_radius"]
            else:
                tip, r_w = None, self.sizes["point_radius"]
            c, z = px(p)
            if c is None:
                continue
            r = max(rmin, int(round(K[0, 0] * s * r_w / z)))
            cv2.circle(img, c, r, col, -1, cv2.LINE_AA)
            if tip is not None:
                ct, _ = px(tip)
                if ct is not None:
                    cv2.arrowedLine(img, c, ct, col, lw, cv2.LINE_AA, tipLength=0.3)

    # ---------------------------------------------------------------- text helpers
    @staticmethod
    def _wrap(text: str, font, width: int):
        """Lines of at most `width` px, broken at any character (CJK and long tokens wrap too). A generator (the caller
        stops once the panel is full); each line's break is found by bisection on the rendered prefix length."""
        for para in str(text).split("\n"):
            if not para:
                yield ""
            while para:
                if font.getlength(para) <= width:
                    yield para
                    break
                lo, hi = 1, len(para)            # longest prefix that fits (at least one character)
                while lo < hi:
                    mid = (lo + hi + 1) // 2
                    if font.getlength(para[:mid]) <= width:
                        lo = mid
                    else:
                        hi = mid - 1
                yield para[:lo]
                para = para[lo:]

    def _text_block(self, draw, x, y, w, h, items, bg=None):
        """items: [(text, font, color)]; wraps to width w and stops when the height h is used up (the last drawn line
        then ends with an ellipsis). bg: fill colour behind each drawn line. Returns the y below the last line."""
        cy, last = y, None
        for text, font, col in items:
            lh = int(font.size * 1.3)
            for line in self._wrap(text, font, w):
                if cy + lh > y + h:
                    if last is not None:
                        lx, ly, lline, lfont, lcol = last
                        draw.text((lx + min(lfont.getlength(lline), w - lfont.getlength("…")), ly), "…", font=lfont, fill=lcol)
                    return cy
                if bg is not None:
                    draw.rectangle([x - self.ov["pad_px"] // 2, cy, x + font.getlength(line) + self.ov["pad_px"] // 2, cy + lh], fill=bg)
                draw.text((x, cy), line, font=font, fill=col)
                last = (x, cy, line, font, col)
                cy += lh
        return cy

    # ---------------------------------------------------------------- main entry
    def compose(self, c: dict) -> np.ndarray:
        """c: {"t_sim", "hold" (None | "llm" | "detection" | "scan"), "images": {main, side, ego, wrist},
        "cams": {main: (K, T), side: (K, T)}, "markers": [...], "head_det": None | {"boxes", "labels"},
        "target_box": id | None, "ego_text", "wrist_polys": None | [[[u, v], ...]], "wrist_text",
        "agent": {"stage", "llm", "tool", "scan_text", "face" (optional OverlayState.face_text())},
        "metrics": frame_metrics()}"""
        canvas = np.zeros((self.H, self.W, 3), np.uint8)
        P = self.panels
        lw = self.ov["line_px"]
        # MAIN / SIDE: fixed cameras + projected 3-D markers
        for name in ("main", "side"):
            x, y, w, h = P[name]
            panel, s, ox, oy = fit_into(c["images"][name], w, h)
            K, T = c["cams"][name]
            self._draw_markers(panel, K, T, s, ox, oy, c["markers"])
            canvas[y:y + h, x:x + w] = panel
        # EGO: head camera + the harness's own head detections when they belong to this frame
        x, y, w, h = P["ego"]
        panel, s, ox, oy = fit_into(c["images"]["ego"], w, h)
        det = c.get("head_det")
        box_labels = []
        if det:
            for b in det["boxes"]:
                x0, y0, x1, y1 = (v * s for v in b["bbox"])
                col = self.col["red"] if b["id"] == c.get("target_box") else self.col["blue"]
                cv2.rectangle(panel, (int(x0 + ox), int(y0 + oy)), (int(x1 + ox), int(y1 + oy)), col, lw, cv2.LINE_AA)
                box_labels.append((int(x0 + ox) + x, max(int(y0 + oy) - self.small.size - 4, 0) + y, str(b["id"]), col))
            for lb in det["labels"]:
                q = np.asarray(lb["corners_px"], float) * s + [ox, oy]
                cv2.polylines(panel, [q.round().astype(np.int32)], True, self.col["purple"], lw, cv2.LINE_AA)
        canvas[y:y + h, x:x + w] = panel
        # WRIST: scanner camera + pyzbar polygons of a decode made on this frame
        x, y, w, h = P["wrist"]
        panel, s, ox, oy = fit_into(c["images"]["wrist"], w, h)
        for poly in c.get("wrist_polys") or []:
            q = np.asarray(poly, float) * s + [ox, oy]
            cv2.polylines(panel, [q.round().astype(np.int32)], True, self.col["green"], lw + 1, cv2.LINE_AA)
        canvas[y:y + h, x:x + w] = panel

        img = Image.fromarray(canvas)
        dr = ImageDraw.Draw(img)
        pad = self.ov["pad_px"]
        white, grey = (255, 255, 255), (190, 190, 190)

        def caption(name, text):        # top-left of a camera panel, wrapped inside it, on a black strip
            px_, py_, pw, ph = P[name]
            self._text_block(dr, px_ + pad, py_, pw - 2 * pad, ph, [(text, self.small, white)], bg=(0, 0, 0))

        for bx, by, txt, col in box_labels:
            dr.text((bx, by), txt, font=self.small, fill=col)
        hold = {"llm": "  [冻结: LLM 思考]", "detection": "  [冻结: 检测结果]", "scan": "  [冻结: 扫码]"}.get(c.get("hold"), "")
        caption("main", f"MAIN 固定观察相机   t_sim = {c['t_sim']:.2f} s{hold}   "
                        "绿 GT  黄 实际  红 根目标  蓝 腕目标(头)  紫 腕目标(腕)")
        caption("side", "SIDE 侧面相机 (机器人-箱面间隙)")
        caption("ego", "EGO 头部相机   " + c["ego_text"])
        caption("wrist", "WRIST 扫码相机   " + c["wrist_text"])

        # AGENT panel
        x, y, w, h = P["agent"]
        a = c["agent"]
        items = [("AGENT", self.font, white)]
        st = a.get("stage")
        items.append((f"阶段 stage: {st['name']} (t_sim {st['t_sim']:.1f} s)" if st else "阶段 stage: -", self.small, white))
        if a.get("face"):
            items.append((a["face"], self.small, white))
        items.append((a.get("scan_text", ""), self.small, self.col["green"]))
        tool = a.get("tool")
        if tool:
            items.append((f"工具 {tool['name']}({_short(tool.get('args'))}) → {tool.get('status')}: {tool.get('reason') or ''}",
                          self.small, white))
        llm = a.get("llm")
        if llm:                       # last, so a long thought is what gets cut at the panel bottom
            calls = ", ".join(f"{tc['name']}({_short(tc.get('arguments'))})" for tc in llm.get("tool_calls") or [])
            lat = llm.get("latency_s")
            items.append((f"LLM [{llm.get('purpose') or ''}]" + (f" {lat:.1f} s" if lat is not None else "")
                          + " → " + (calls or "(no tool call)"), self.small, self.col["yellow"]))
            items.append(("思考: " + (llm.get("thought") or "(无文字 / no text)"), self.small, white))
            if llm.get("reasoning"):
                items.append(("推理 reasoning: " + llm["reasoning"], self.small, grey))
        self._text_block(dr, x + pad, y + pad, w - 2 * pad, h - 2 * pad, items)

        # METRICS panel
        x, y, w, h = P["metrics"]
        m = c["metrics"]
        r = m["root"]
        items = [("METRICS (目标面坐标系: n 法向朝机器人, l 机器人左, z 上; 实际 - GT)", self.small, white),
                 (f"骨盆高度 pelvis z {m['pelvis_z']:.3f} m" + (f"  (高度指令 {m['height_cmd']:.2f})" if m.get("height_cmd") is not None else ""),
                  self.small, white),
                 ((f"根(pelvis 地面投影) vs GT 观察位姿 — 任务1结束时 (t_sim {r['frozen_t']:.1f} s, 已冻结):"
                   if r.get("frozen_t") is not None else "根(pelvis 地面投影) vs GT 观察位姿 — 实时:"), self.small, white),
                 (f"    dn {r['dn']:+.3f}  dl {r['dl']:+.3f} m   |dxy| {r['dxy']:.3f} m   dyaw {r['dyaw_deg']:+.1f}°",
                  self.font, white)]
        tp = m["tip"]
        if tp:
            items += [(f"扫码头 vs GT 目标点 [{tp['content']}] (d={tp['d_gt']:.3f} m):", self.small, white),
                      (f"    dn {tp['dn']:+.3f}  dl {tp['dl']:+.3f}  dz {tp['dz']:+.3f} m", self.font, white),
                      (f"    |d| {tp['norm']:.3f} m", self.font, white)]
        else:
            items += [("扫码头 vs GT 目标点: - (当前条码未知)", self.small, white)]
        sc = m["scanned"]
        items.append((f"已扫描 scanned: {sc['k']}/{sc['n']}" + (f"   非GT: {', '.join(sc['not_gt'])}" if sc["not_gt"] else ""),
                      self.font, self.col["green"]))
        g = m.get("min_gap")
        items.append((f"最小间隙 min gap: {g['gap']:.3f} m ({g['body']} ↔ {g['obstacle']})" if g else "最小间隙 min gap: -",
                      self.small, grey))
        self._text_block(dr, x + pad, y + pad, w - 2 * pad, h - 2 * pad, items)
        for name, (px_, py_, pw, ph) in P.items():
            dr.rectangle([px_, py_, px_ + pw - 1, py_ + ph - 1], outline=(70, 70, 70))
        return np.asarray(img)


def _short(v) -> str:
    """Compact one-line rendering of tool arguments for the AGENT panel."""
    if not v:
        return ""
    if isinstance(v, dict):
        return ", ".join(f"{k}={_short(x) if isinstance(x, (dict, list)) else x}" for k, x in v.items())
    if isinstance(v, list):
        return "[" + ", ".join(f"{x:.3g}" if isinstance(x, float) else _short(x) if isinstance(x, (dict, list)) else str(x)
                               for x in v) + "]"
    return str(v)


# ---------------------------------------------------------------------------------------------------------- encoder
class H264Writer:
    """Raw RGB frames piped to an ffmpeg binary: libx264, yuv420p, fragmented MP4 (-movflags frag_keyframe+empty_moov,
    plus one fragment per frame period via -frag_duration, each flushed to disk) so a file whose writer was killed still
    opens and plays up to its last complete fragment. -tune zerolatency: x264 otherwise holds back ~rc-lookahead + one
    frame per frame thread (measured: nothing but the 36-byte header on disk after 200 frames on this 256-core host),
    which a killed run would lose. Encoding runs in the ffmpeg process."""

    def __init__(self, path: str, width: int, height: int, fps: float, ffmpeg_exe: str):
        if width % 2 or height % 2:
            raise ValueError(f"yuv420p needs even frame dimensions, got {width}x{height}")
        self.path, self.size, self.n = path, (width, height), 0
        rate = Fraction(fps).limit_denominator(1000)
        self.log_path = path + ".ffmpeg.log"
        self._log = open(self.log_path, "wb")
        cmd = [ffmpeg_exe, "-hide_banner", "-loglevel", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", f"{rate.numerator}/{rate.denominator}",
               "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-tune", "zerolatency",
               "-movflags", "frag_keyframe+empty_moov", "-frag_duration", str(int(round(1e6 / fps))),
               "-flush_packets", "1", path]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=self._log)
        self._check("start")

    def _check(self, when: str):
        if self.proc.poll() is not None:
            raise RuntimeError(f"ffmpeg exited (code {self.proc.returncode}) at {when}: {self._log_tail()}")

    def _log_tail(self) -> str:
        self._log.flush()
        with open(self.log_path, "rb") as f:
            return f.read()[-2000:].decode(errors="replace")

    def write(self, rgb: np.ndarray, repeat: int = 1):
        if rgb.shape != (self.size[1], self.size[0], 3) or rgb.dtype != np.uint8:
            raise ValueError(f"frame must be {self.size[1]}x{self.size[0]}x3 uint8, got {rgb.shape} {rgb.dtype}")
        self._check(f"frame {self.n}")
        buf = np.ascontiguousarray(rgb).tobytes()
        for _ in range(repeat):
            self.proc.stdin.write(buf)
            self.n += 1

    def close(self) -> int:
        """Finish the file; raises if ffmpeg failed. Returns the number of frames written."""
        self.proc.stdin.close()
        code = self.proc.wait()
        tail = self._log_tail()
        self._log.close()
        if code != 0:
            raise RuntimeError(f"ffmpeg failed (code {code}) writing {self.path}: {tail}")
        if not tail.strip():
            os.remove(self.log_path)
        return self.n
