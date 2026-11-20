"""Left-arm IK for the scanner tip: position (3) + scan direction (2) and, with a roll axis, the roll about the scan axis
(the scanner image's horizontal axis parallel to the barcode's long axis, sign free: 6 constraints); 7 arm joints,
damped least squares on the pinocchio model (the same task the simulator's aim controller uses, solved offline).

xr_teleoperate's IK was the obvious candidate; that solver targets the Dex3/Inspire hand frames of its own
URDF, while this robot carries a scanner (g1_v1: held by a Dex1 gripper; g1_v2: bolted to a left-wrist bracket), so the
task frame (scanner tip) only exists in our agent URDF. A small DLS solver on that model is used instead.

Frames come from the resolved robot config (config.apply_robot_variant): wrist_point_frame (the scanner tip) and
scanner_cam_frame (the scanner's own imager, whose image x axis is the roll reference). The roll reference is the
decoder's image axis, never a separate hand camera (g1_v2's D405), even where both happen to give the same axis.
"""
from __future__ import annotations

import numpy as np

from .fk import Kinematics

ARM = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"]


def rotation_vector(a, b):
    c = np.cross(a, b); s, co = np.linalg.norm(c), float(a @ b)
    ang = np.arctan2(s, co)
    if s < 1e-9:
        if co > 0:
            return np.zeros(3)
        ax = np.cross(a, [1.0, 0, 0])
        ax = ax if np.linalg.norm(ax) > 1e-6 else np.cross(a, [0, 1.0, 0])
        return ax / np.linalg.norm(ax) * ang
    return c / s * ang


class ArmIK:
    def __init__(self, kin: Kinematics, robot_cfg: dict, side: str | None = None):
        """robot_cfg: cfg["robot"] after config.apply_robot_variant (frames of the variant)."""
        rc = robot_cfg
        if "scanner_cam_frame" not in rc:
            raise ValueError("ArmIK needs the robot config resolved for the robot variant (config.apply_robot_variant)")
        self.cfg = dict(rc["ik"])
        self.kin = kin
        self.side = side or rc["arm_side"]
        self.frame = rc["wrist_point_frame"]
        self.joints = [f"{self.side}_{j}_joint" for j in ARM]
        self.qi = np.array([self.kin.joint_q_index[j] for j in self.joints])
        self.vi = np.array([self.kin.joint_v_index[j] for j in self.joints])
        m = self.cfg["joint_limit_margin"]
        self.lo, self.hi = self.kin.lower[self.qi] + m, self.kin.upper[self.qi] - m
        self.rng = np.random.default_rng(self.cfg["seed"])
        # the scanner image's horizontal axis (scanner imager optical x) in the scanner-tip frame: a fixed transform of the
        # model ([0, -1, 0] for both agent URDFs); the roll constraint aligns it with the barcode's long axis
        import pinocchio as pin
        q_n = pin.neutral(self.kin.model)
        R_tc = np.linalg.inv(self.kin.pose(q_n, self.frame))[:3, :3] @ self.kin.pose(q_n, rc["scanner_cam_frame"])[:3, :3]
        self.img_x_in_tip = R_tc[:, 0] / np.linalg.norm(R_tc[:, 0])

    def roll_target(self, R, d_t, roll_axis):
        """Target rotation of the scanner-tip frame: x along d_t, image horizontal along the roll axis made orthogonal to
        d_t, sign chosen nearest to the current orientation R (a code reads the same both ways round)."""
        L = np.asarray(roll_axis, float) - (np.asarray(roll_axis, float) @ d_t) * d_t
        L /= np.linalg.norm(L)
        if (R @ self.img_x_in_tip) @ L < 0:
            L = -L
        ex, c = np.array([1.0, 0.0, 0.0]), self.img_x_in_tip
        return np.column_stack([d_t, L, np.cross(d_t, L)]) @ np.column_stack([ex, c, np.cross(ex, c)]).T

    def roll_error_deg(self, R, d_t, roll_axis) -> float:
        """Angle (deg, 0-90) between the scanner image's horizontal axis and the barcode's long axis about the scan axis."""
        L = np.asarray(roll_axis, float) - (np.asarray(roll_axis, float) @ d_t) * d_t
        L /= np.linalg.norm(L)
        x = R @ self.img_x_in_tip
        x = x - (x @ d_t) * d_t
        x /= np.linalg.norm(x)
        return float(np.degrees(np.arccos(min(1.0, abs(float(x @ L))))))

    def error(self, q, p_target, d_target):
        T = self.kin.pose(q, self.frame)
        a = T[:3, 0]
        return T, p_target - T[:3, 3], rotation_vector(a, d_target), a

    def _solve_from(self, q, p_t, d_t, stall=None, roll_axis=None):
        """stall = (window, eps): give up when the weighted error (position + aim_weight * angle) improved by less than
        eps over the last `window` iterations (planning only, see robot.yaml ik.plan_stall_*). roll_axis: also align the
        scanner image's horizontal axis with it (full orientation target, see roll_target)."""
        from scipy.spatial.transform import Rotation
        c, w = self.cfg, self.cfg["aim_weight"]
        hist = []
        for it in range(c["max_iters"]):
            T, ep, er, a = self.error(q, p_t, d_t)
            ang = np.degrees(np.linalg.norm(er))
            if roll_axis is not None:
                er = Rotation.from_matrix(self.roll_target(T[:3, :3], d_t, roll_axis) @ T[:3, :3].T).as_rotvec()
                ang = np.degrees(np.linalg.norm(er))      # full orientation error (aim and roll)
            if np.linalg.norm(ep) < c["tol_pos"] and ang < c["tol_angle_deg"]:
                return q, True, it
            if stall is not None:
                hist.append(float(np.linalg.norm(ep) + w * np.linalg.norm(er)))
                if len(hist) > stall[0] and hist[-1 - stall[0]] - hist[-1] < stall[1]:
                    return q, False, it
            J = self.kin.jacobian(q, self.frame)[:, self.vi]
            if roll_axis is None:
                P = np.eye(3) - np.outer(a, a)          # rotation about the scan axis is free
            else:
                P = np.eye(3)
            A = np.vstack([J[:3], w * (P @ J[3:])])
            e = np.concatenate([ep, w * (P @ er)])
            dq = c["step_gain"] * A.T @ np.linalg.solve(A @ A.T + c["dls_lambda"] ** 2 * np.eye(6), e)
            q = q.copy()
            q[self.qi] = np.clip(q[self.qi] + dq, self.lo, self.hi)
        return q, False, c["max_iters"]

    def solve(self, q_now, p_target, d_target, stall=None, roll_axis=None, rng=None) -> dict:
        """q_now: full pinocchio configuration (other joints held). Returns {"ok", "q_arm", "pos_err", "angle_err_deg",
        "iters", "start"}; ok=False -> best attempt (unreachable or not converged)."""
        d_target = np.asarray(d_target, float) / np.linalg.norm(d_target)
        p_target = np.asarray(p_target, float)
        best = None
        starts = [np.asarray(q_now, float)]
        for _ in range(self.cfg["restarts"]):
            s = np.asarray(q_now, float).copy()
            s[self.qi] = (self.rng if rng is None else rng).uniform(self.lo, self.hi)
            starts.append(s)
        for k, s in enumerate(starts):
            q, ok, it = self._solve_from(s, p_target, d_target, stall, roll_axis)
            T, ep, er, _ = self.error(q, p_target, d_target)
            r = {"ok": ok, "q_arm": q[self.qi].copy(), "q_full": q, "pos_err": float(np.linalg.norm(ep)),
                 "angle_err_deg": float(np.degrees(np.linalg.norm(er))), "iters": it, "start": k,
                 "roll_err_deg": None if roll_axis is None else self.roll_error_deg(T[:3, :3], d_target, roll_axis),
                 "limit_margin_min": float(np.min(np.minimum(q[self.qi] - self.lo, self.hi - q[self.qi])))}
            score = r["pos_err"] + self.cfg["aim_weight"] * np.radians(r["angle_err_deg"] + (r["roll_err_deg"] or 0.0))
            if best is None or (r["ok"] and not best["ok"]) or (r["ok"] == best["ok"] and score < best["_score"]):
                best = dict(r, _score=score)
            if ok and k == 0:
                break
        best.pop("_score")
        return best
