"""Wrist-camera symbol width (refine_with_wrist -> Toolbox._wrist_symbol_width) and the stand-off it sets."""
import math

import cv2
import numpy as np
import pytest

from g1agent.common import config
from g1agent.tools import scan_order as so
from test_task3_logic import FakeRobot, bare_toolbox, plan_of


def _frame(side_m, dist, rot_deg, fx=1087.0, W=640, H=480):
    """Wrist frame looking straight at the plane x = dist (camera optical axis = world +x), with a QR symbol of side
    side_m rotated by rot_deg in the plane, centred on the axis. Returns (frame, plane)."""
    qr = cv2.QRCodeEncoder.create().encode("WIDTH-TEST-01")                 # modules, no quiet zone? -> crop to symbol
    ys, xs = np.where(qr < 128)
    sym = qr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    n = sym.shape[0]
    px = fx * side_m / dist                                                  # symbol side in pixels
    s = px / n
    M = cv2.getRotationMatrix2D((n / 2, n / 2), rot_deg, s)
    M[:, 2] += [W / 2 - n / 2, H / 2 - n / 2]
    img = cv2.warpAffine(sym, M, (W, H), flags=cv2.INTER_NEAREST, borderValue=255)
    rgb = cv2.cvtColor(cv2.GaussianBlur(img, (3, 3), 0.7), cv2.COLOR_GRAY2RGB)
    K = np.array([[fx, 0, W / 2], [0, fx, H / 2], [0, 0, 1.0]])
    T_root_cam = np.eye(4)
    T_root_cam[:3, :3] = np.array([[0, 0, 1.0], [-1.0, 0, 0], [0, -1.0, 0]])   # optical (x right, y down, z fwd) -> root
    frame = {"rgb": rgb, "K": K, "T_root_cam": T_root_cam, "T_odom_root": np.eye(4)}
    plane = {"center": np.array([dist, 0.0, 0.0]), "normal": np.array([-1.0, 0.0, 0.0])}
    return frame, plane


@pytest.mark.parametrize("side,dist,rot", [(0.05, 0.30, 0.0), (0.05, 0.30, 20.0), (0.08, 0.40, 35.0)])
def test_wrist_symbol_width_on_the_plane(side, dist, rot):
    tb = bare_toolbox(FakeRobot())
    frame, plane = _frame(side, dist, rot)
    w = tb._wrist_symbol_width(frame, {"center": [320.0, 240.0]}, plane)
    assert w["width_m"] == pytest.approx(side, rel=0.03), w


def test_wrist_symbol_width_none_when_nothing_decodes():
    tb = bare_toolbox(FakeRobot())
    frame, plane = _frame(0.05, 0.30, 0.0)
    frame["rgb"] = np.full_like(frame["rgb"], 200)
    assert tb._wrist_symbol_width(frame, {"center": [320.0, 240.0]}, plane)["width_m"] is None


def test_refined_target_uses_the_wrist_width():
    tb = bare_toolbox(FakeRobot())
    plan_of(tb, 1, width=0.12)                                     # head: the loose label box
    tb.next_barcode()
    head = tb.compute_wrist_target()["data"]["offset"]
    bc = tb.ws.barcodes["bc0"]
    bc.width_m, bc.width_source = 0.08, "wrist"                   # what refine_with_wrist stores
    tb.compute_wrist_target(refined=True)
    S = config.load("task3")["standoff"]
    want = so.adaptive_standoff(0.08, tb.wrist_hfov, S["fill_frac"], S["min_m"], S["max_m"])
    assert tb.ws.wrist_target_refined["offset"] == pytest.approx(want) and want < head
    assert want == pytest.approx(0.20, abs=0.002)
