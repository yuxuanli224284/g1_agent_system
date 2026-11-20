"""Whole-body model for stance planning with squats (three-box floor stack, 2026-11-06).

- Squat levels: measured GR00T body-command calibration (configs/robot.yaml body.calibration): for each commanded base
  height, the settled pelvis height and pitch and the full joint state, measured in Isaac standing still with the arms
  at home (g1_barcode_sim scripts/controller_compare/body_motion_eval.py --parts squat). Only measured levels are planned.
- Robot geometry: the robot's own collision points per link (hull vertices / sphere centres + radii, link frames) from
  the robot port (get_robot_geometry): CAD knowledge, not ground truth. Used for
    * clearance to the known box faces of every robot point (knees move ~0.3 m forward of the pelvis in a deep squat)
    * arm-vs-leg clearance: the simulated articulation has self-collisions disabled, so an arm passing through the
      thighs would go unnoticed in simulation; the planner rejects such poses itself.
Pure numpy + pinocchio (through Kinematics)."""
from __future__ import annotations

import math

import numpy as np

LEG_KEYS = ("hip", "knee", "ankle")
ARM_KEYS = ("elbow", "wrist", "hand")
TRUNK = ("pelvis", "pelvis_contour_link", "torso_link")


def Ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


class BodyModel:
    def __init__(self, kin, geometry: dict, calibration: list, arm_side: str, wrist_point_frame: str):
        from scipy.spatial import ConvexHull, QhullError
        self.kin = kin
        names = list(geometry["body_names"])
        P = np.asarray(geometry["points"], float)
        rad = np.asarray(geometry["radius"], float)
        own = np.asarray(geometry["owner"], int)
        self.bodies = {}
        for bi, n in enumerate(names):
            m = own == bi
            if not m.any():
                continue
            try:
                self.kin.frame_id(n)
            except KeyError:
                raise KeyError(f"robot geometry link {n!r} has no frame in the kinematics URDF")
            self.bodies[n] = {"pts": P[m], "rad": rad[m]}
        # links the arm must stay clear of: legs, pelvis and torso (the arm's own shoulder links are not checked)
        self.legs = [n for n in self.bodies if (n.startswith(("left_", "right_")) and any(k in n for k in LEG_KEYS))
                     or n in TRUNK]
        self.arm = [n for n in self.bodies if n.startswith(arm_side + "_") and any(k in n for k in ARM_KEYS)]
        if not self.legs or not self.arm:
            raise ValueError(f"leg links {self.legs} / arm links {self.arm}: geometry incomplete")
        # leg links as convex hulls (half-spaces, unit normals) or, for sphere-only links (the feet), spheres
        self.leg_shapes = {}
        for n in self.legs:
            b = self.bodies[n]
            if np.all(b["rad"] == 0) and len(b["pts"]) >= 4:
                try:
                    self.leg_shapes[n] = ("hull", ConvexHull(b["pts"]).equations)
                    continue
                except QhullError:
                    pass                                    # flat point set: fall back to its points as spheres
            self.leg_shapes[n] = ("spheres", b["pts"], b["rad"])
        # reach bound of the scanner arm: the chain's consecutive link origins are rigidly spaced, so the sum of their
        # distances bounds |scanner tip - shoulder| for every joint configuration (a pure prefilter, never a decision).
        # The chain is the model's own link chain from the scanner tip up to the shoulder (g1_v1: ... wrist_yaw ->
        # hand_base -> scanner_tip; g1_v2: ... wrist_yaw -> scanner_tip on the bracket)
        import pinocchio as pin
        chain = self._link_chain(kin, wrist_point_frame, f"{arm_side}_shoulder_pitch_link")
        q_n = pin.neutral(kin.model)
        org = [kin.pose(q_n, f)[:3, 3] for f in chain]
        self.shoulder_frame = chain[0]
        self.arm_reach = float(sum(np.linalg.norm(b - a) for a, b in zip(org[:-1], org[1:])))
        self.levels = sorted(calibration, key=lambda c: -c["height_cmd"])
        if not self.levels:
            raise ValueError("empty body calibration")

    @staticmethod
    def _link_chain(kin, tip: str, root_link: str) -> list:
        """Link (body) frames from root_link down to the tip frame, following the model's frame tree upwards."""
        import pinocchio as pin
        m = kin.model
        f, out = kin.frame_id(tip), []
        while True:
            fr = m.frames[f]
            if fr.type == pin.FrameType.BODY:
                out.append(fr.name)
            if fr.name == root_link:
                return out[::-1]
            if f == 0:
                raise ValueError(f"{root_link} is not an ancestor of {tip} in the kinematics URDF")
            f = fr.parentFrame

    # ------------------------------------------------------------------ poses
    def q_at(self, q_base, level: dict):
        """Configuration with the level's measured leg + waist joints, everything else (the arm) from q_base."""
        q = np.array(q_base, float)
        for n, v in level["q"].items():
            if n in self.kin.joint_q_index and not any(k in n for k in ("shoulder", "elbow", "wrist", "hand")):
                q[self.kin.joint_q_index[n]] = float(v)
        return q

    def T_root(self, x, y, yaw, level: dict, z_ground: float = 0.0):
        """Pelvis pose (odom) for a pelvis ground position (x, y), yaw, at a level: measured height and pitch."""
        from ..common.transforms import T_from_xy_yaw
        T = T_from_xy_yaw(x, y, yaw, z_ground + level["pelvis_z"])
        T[:3, :3] = T[:3, :3] @ Ry(math.radians(level["pelvis_pitch_deg"]))
        return T

    def foot_shift(self, level: dict, standing: dict, q_base):
        """Horizontal pelvis displacement (yaw-aligned root frame, x forward) when squatting from `standing` to
        `level` with the feet fixed: the walk target is the squat pelvis position minus this."""
        def feet(lv):
            q = self.q_at(q_base, lv)
            T = self.T_root(0.0, 0.0, 0.0, lv)
            return np.mean([(T @ self.kin.pose(q, n))[:2, 3] for n in ("left_ankle_roll_link", "right_ankle_roll_link")], axis=0)
        return feet(standing) - feet(level)

    def points(self, q, T_root, names):
        out_p, out_r = [], []
        for n in names:
            T = T_root @ self.kin.pose(q, n)
            b = self.bodies[n]
            out_p.append(b["pts"] @ T[:3, :3].T + T[:3, 3])
            out_r.append(b["rad"])
        return np.vstack(out_p), np.concatenate(out_r)

    # ------------------------------------------------------------------ clearances
    def self_clearance(self, q):
        """Lower bound of the arm-to-leg distance (m, < 0 = the arm is inside a leg link), root frame: arm points
        against leg half-spaces (max over facet planes is <= the true distance outside a convex hull) or spheres."""
        I = np.eye(4)
        ap, ar = self.points(q, I, self.arm)
        best = (math.inf, None, None)
        for n in self.legs:
            T = self.kin.pose(q, n)
            sh = self.leg_shapes[n]
            pl = (ap - T[:3, 3]) @ T[:3, :3]                 # arm points in the leg link frame
            if sh[0] == "hull":
                d = (pl @ sh[1][:, :3].T + sh[1][:, 3]).max(axis=1) - ar
            else:
                d = (np.linalg.norm(pl[:, None, :] - sh[1][None], axis=2) - sh[2][None]).min(axis=1) - ar
            i = int(np.argmin(d))
            if d[i] < best[0]:
                best = (float(d[i]), n, i)
        return best[0], best[1]

    def face_clearance(self, q, T_root, faces, footprint=None):
        """Smallest horizontal clearance (m) of any robot point to the known box faces: signed along the face normal
        inside the face's lateral extent (< 0 = behind the face plane, inside the box), distance to the face edge
        beside it. Faces: {"center", "normal", "corners"} (odom)."""
        P, R = self.points(q, T_root, list(self.bodies))
        if footprint is not None:
            # closed stack footprint (multi-face): signed horizontal distance of every robot point to it (negative
            # inside); the front-plane test below would call every point behind the stack "inside the box"
            from .toolbox import _poly_signed_distance
            return float((_poly_signed_distance(P[:, :2], footprint) - R).min())
        best = (math.inf, None)
        for f in faces:
            c = np.asarray(f["center"], float)
            n = np.asarray(f["normal"], float)[:2]; n /= np.linalg.norm(n)
            r = np.array([-n[1], n[0]])
            s = (np.asarray(f["corners"], float)[:, :2] - c[:2]) @ r
            lat = (P[:, :2] - c[:2]) @ r
            t = np.clip(lat, s.min(), s.max())
            nd = (P[:, :2] - c[:2]) @ n
            inside = t == lat
            d = np.where(inside, nd, np.hypot(lat - t, nd)) - R
            i = int(np.argmin(d))
            if d[i] < best[0]:
                best = (float(d[i]), i)
        return best[0]
