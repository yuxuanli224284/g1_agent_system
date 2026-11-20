"""RobotInterface: everything the harness may do with the robot. The simulator client and a
future real-robot (DDS) implementation provide the same methods. No ground truth here."""
from __future__ import annotations

from abc import ABC, abstractmethod


class RobotInterface(ABC):
    @abstractmethod
    def get_robot_info(self) -> dict: ...                       # robot variant, joint names, camera intrinsics, limits
    @abstractmethod
    def get_head_obs(self, depth: bool = False) -> dict: ...    # rgb, K, T_root_headcam, timestamp; stereo head (D435i):
                                                                # ir_left, ir_right, K_ir, T_root_ir_left, T_irleft_irright;
                                                                # depth (m, colour frame): the camera's own depth, if asked
    @abstractmethod
    def get_wrist_obs(self, side: str = "left") -> dict: ...    # the scanner imager's frame: rgb, depth, K, T_root_wristcam,
                                                                # timestamp (simulator only: a real DS2208 gives no image)
    @abstractmethod
    def get_hand_camera_obs(self, name: str, depth: bool = False) -> dict: ...   # a camera on a hand (robot info
                                                                # hand_cameras, e.g. left_d405): rgb, K, T_root_cam,
                                                                # timestamp; depth (m), if asked
    @abstractmethod
    def get_joint_state(self) -> dict: ...                      # names, q
    @abstractmethod
    def get_odom(self) -> dict: ...                             # T_odom_root
    @abstractmethod
    def walk(self, vx: float, vy: float, wz: float, duration: float) -> dict: ...
    @abstractmethod
    def move_root_relative(self, dx: float, dy: float, dyaw: float, timeout: float = 10.0,
                           pos_tol: float | None = None, yaw_tol: float | None = None, speed_scale: float = 1.0) -> dict: ...
    @abstractmethod
    def stand(self, duration: float) -> dict: ...
    @abstractmethod
    def set_body_pose(self, height: float | None = None, torso_rpy=None, duration: float = 1.0, settle: float = 1.0) -> dict: ...
    # squat / stand up (pelvis height command, m) and torso roll/pitch/yaw (rad), standing still
    @abstractmethod
    def get_robot_geometry(self) -> dict: ...                   # the robot's own collision points per link (body frames)
    @abstractmethod
    def set_arm_joints(self, side: str, q, duration: float) -> dict: ...
    @abstractmethod
    def set_gripper(self, side: str, opening: float) -> dict: ...   # only on a hand with a gripper (robot info grippers)
    @abstractmethod
    def scan(self) -> dict: ...                                 # trigger the scanner on the left hand (held by the gripper
                                                                # or bolted to a wrist bracket, robot info scanner)
    @abstractmethod
    def draw_markers(self, markers: list, frame: str = "odom") -> dict: ...   # simulator only (no-op on hardware)
    @abstractmethod
    def clear_markers(self) -> dict: ...
