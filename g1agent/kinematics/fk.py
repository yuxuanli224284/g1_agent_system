"""Forward kinematics with pinocchio on the agent URDF of the robot variant (fixed base = pelvis = root). The URDF is
always given explicitly: config.apply_robot_variant resolves it from the variant the robot reports (robot.yaml
variants.<variant>.urdf); there is no default model, so a v1 model can never silently drive a v2 robot."""
from __future__ import annotations

import os

import numpy as np
import pinocchio as pin

from ..common import config


class Kinematics:
    def __init__(self, urdf: str):
        if not urdf:
            raise ValueError("Kinematics needs the robot variant's URDF (config.apply_robot_variant -> cfg['robot']['urdf'])")
        path = urdf if os.path.isabs(urdf) else os.path.join(config.ROOT, urdf)
        self.urdf = path
        self.model = pin.buildModelFromUrdf(path)
        self.data = self.model.createData()
        self.joint_q_index = {self.model.names[j]: self.model.joints[j].idx_q for j in range(1, self.model.njoints)}
        self.joint_v_index = {self.model.names[j]: self.model.joints[j].idx_v for j in range(1, self.model.njoints)}
        self.lower = self.model.lowerPositionLimit.copy()
        self.upper = self.model.upperPositionLimit.copy()

    def q_from_state(self, names, q) -> np.ndarray:
        """Joint state (names, values) -> pinocchio configuration (unknown names ignored, missing joints = 0)."""
        out = pin.neutral(self.model)
        for n, v in zip(names, q):
            if n in self.joint_q_index:
                out[self.joint_q_index[n]] = float(v)
        return out

    def frame_id(self, name: str) -> int:
        fid = self.model.getFrameId(name)
        if fid >= self.model.nframes:
            raise KeyError(f"frame {name!r} not in the URDF")
        return fid

    def pose(self, q, frame: str) -> np.ndarray:
        """T_root_frame (4x4)."""
        pin.framesForwardKinematics(self.model, self.data, q)
        return self.data.oMf[self.frame_id(frame)].homogeneous.copy()

    def jacobian(self, q, frame: str) -> np.ndarray:
        """6 x nv frame jacobian in the root-aligned frame at the frame origin (rows: linear, angular)."""
        pin.computeJointJacobians(self.model, self.data, q)
        pin.framesForwardKinematics(self.model, self.data, q)
        return pin.getFrameJacobian(self.model, self.data, self.frame_id(frame), pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
