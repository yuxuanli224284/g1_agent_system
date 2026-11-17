"""RobotInterface over RPC to the simulation server's robot port. Deliberately has no ground-truth methods; evaluation lives in g1agent/eval/eval_client.py."""
from __future__ import annotations

import numpy as np

from ..common import config
from .interface import RobotInterface
from .rpc import RpcClient


class SimRobot(RobotInterface):
    def __init__(self, instance: int | None = None):
        import os
        s = config.load("system")["sim_server"]
        i = int(os.environ.get("SIM_INSTANCE", "0")) if instance is None else instance
        self.rpc = RpcClient(s["host"], s["robot_port"] + i * s["instance_port_stride"], s["rpc_timeout_s"])

    def get_robot_info(self): return self.rpc.call("get_robot_info")
    def get_head_obs(self, depth=False): return self.rpc.call("get_head_obs", depth=bool(depth))
    def get_wrist_obs(self, side="left"): return self.rpc.call("get_wrist_obs", side=side)
    def get_joint_state(self): return self.rpc.call("get_joint_state")
    def get_odom(self): return self.rpc.call("get_odom")
    def walk(self, vx, vy, wz, duration): return self.rpc.call("walk", vx=float(vx), vy=float(vy), wz=float(wz), duration=float(duration))
    def move_root_relative(self, dx, dy, dyaw, timeout=10.0, pos_tol=None, yaw_tol=None, speed_scale=1.0):
        return self.rpc.call("move_root_relative", dx=float(dx), dy=float(dy), dyaw=float(dyaw), timeout=float(timeout),
                             pos_tol=pos_tol, yaw_tol=yaw_tol, speed_scale=float(speed_scale))
    def stand(self, duration): return self.rpc.call("stand", duration=float(duration))
    def set_body_pose(self, height=None, torso_rpy=None, duration=1.0, settle=1.0):
        return self.rpc.call("set_body_pose", height=None if height is None else float(height),
                             torso_rpy=None if torso_rpy is None else [float(v) for v in torso_rpy],
                             duration=float(duration), settle=float(settle))
    def get_robot_geometry(self): return self.rpc.call("get_robot_geometry")
    def set_arm_joints(self, side, q, duration):
        return self.rpc.call("set_arm_joints", side=side, q=np.asarray(q, float), duration=float(duration))
    def set_gripper(self, side, opening): return self.rpc.call("set_gripper", side=side, opening=float(opening))
    def scan(self): return self.rpc.call("scan")
    def draw_markers(self, markers, frame="odom"): return self.rpc.call("draw_markers", markers_list=markers, frame=frame)
    def clear_markers(self): return self.rpc.call("clear_markers")
    def ping(self): return self.rpc.call("ping")
