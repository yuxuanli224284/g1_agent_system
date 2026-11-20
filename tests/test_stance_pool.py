"""plan_stance gives the same stance serially and with the planning process pool (per-candidate IK seeds)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from g1agent.common import config  # noqa: E402
from g1agent.common.transforms import T_from_xy_yaw  # noqa: E402
from g1agent.harness.world_state import BarcodeEstimate, WorldState  # noqa: E402
from g1agent.kinematics.fk import Kinematics  # noqa: E402
from g1agent.kinematics.ik import ArmIK  # noqa: E402
from g1agent.tools import scan_order as so  # noqa: E402
from g1agent.tools import toolbox as tbmod  # noqa: E402


class R:
    def __init__(self, kin):
        import pinocchio as pin
        self.q = pin.neutral(kin.model)
        self.names = list(kin.joint_q_index)

    def get_odom(self):
        return {"T_odom_root": T_from_xy_yaw(0.6, 0.0, 0.0, 0.74)}

    def get_joint_state(self):
        return {"names": self.names, "q": [0.0] * len(self.names)}

    def draw_markers(self, *a, **k):
        return {"status": "ok"}


def _tb(workers, variant="g1_v1"):
    tb = tbmod.Toolbox.__new__(tbmod.Toolbox)
    tb.cfg = config.apply_head_camera(config.apply_robot_variant(config.load_all(), variant), "d435i")
    tb.ws = WorldState(); tb.markers = {}; tb.task = 3; tb.scan_targets = None
    tb.kin = Kinematics(tb.cfg["robot"]["urdf"]); tb.ik = ArmIK(tb.kin, tb.cfg["robot"]); tb.ik.cfg = dict(tb.ik.cfg, plan_workers=workers)
    tb.robot = R(tb.kin); tb.side = "left"; tb.arm_at_home = True; tb.head = None
    tb._last_label = tb._wrist_detection = None
    tb.scanner_hfov = so.hfov_from_intrinsics(1087.0, 640.0)
    tb.body_cmd = None; tb.base_height_cmd = None; tb._body_model = None; tb.multi_face = False; tb.view_pitch = 0.0; tb._leans = None
    face = {"center": np.array([1.3, 0.0, 0.9]), "normal": np.array([-1.0, 0, 0]), "right": np.array([0, -1.0, 0]),
            "up": np.array([0, 0, 1.0]), "corners": np.array([[1.3, y, z] for y in (-0.2, 0.2) for z in (0.75, 1.05)])}
    bc = BarcodeEstimate("bc0", "box0", np.array([1.2985, 0.05, 0.92]), np.array([-1.0, 0, 0]), np.array([0, 0, 1.0]), "head",
                         0.0, face=face, width_m=0.08, status="pending", order_index=0)
    tb.ws.barcodes = {"bc0": bc}; tb.ws.scan_plan = ["bc0"]; tb.ws.current_barcode_id = "bc0"
    tb.compute_wrist_target("bc0")
    return tb


@pytest.mark.parametrize("variant", ["g1_v1", "g1_v2"])
def test_pool_matches_serial(variant):
    """Also checks that the pool workers build the parent's variant (their own config read would be the v1 default)."""
    a = _tb(0, variant).plan_stance()
    tb = _tb(2, variant)
    try:
        b = tb.plan_stance()
    finally:
        tb._plan_pool.shutdown()
    assert a["status"] == b["status"] == "ok"
    assert a["data"]["stance_odom"] == b["data"]["stance_odom"] and a["data"]["rejected"] == b["data"]["rejected"]
