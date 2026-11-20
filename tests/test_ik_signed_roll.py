"""ArmIK signed roll (design scanroom_layout §6.2): roll_signed=True keeps the direction of the roll axis (a hand-camera
look pose, where the side of the scan axis the D405 sits on matters); the default stays sign free and bit-identical."""
import numpy as np
import pinocchio as pin
import pytest

from g1agent.common import config
from g1agent.kinematics.fk import Kinematics
from g1agent.kinematics.ik import ArmIK

# ArmIK.solve outputs of the code BEFORE the signed-roll change (agent commit 21f856f, this conda env), recorded as
# float.hex: the default (unsigned) path must reproduce them bit for bit. Inputs: _cases below; the calls ran in the
# order of the test (every solve draws its restart starts from the IK's rng, so "far" depends on the calls before it).
REF = {
    'g1_v1': {
        'free': {"ok": True, "iters": 10, "start": 0, "q_arm": [
            '-0x1.34479142d98b7p-1', '0x1.ae5b927f1b367p-4', '0x1.6c3ecf3950aa8p-3', '0x1.4769f2e6932c2p-2',
            '-0x1.ade61ce462043p-4', '-0x1.ee32be6fd4950p-3', '-0x1.8a0c1e6c37028p-5']},
        'roll': {"ok": True, "iters": 11, "start": 0, "q_arm": [
            '-0x1.2a11507ec7411p-1', '0x1.29355f35cb897p-2', '0x1.b52c3dea5042fp-4', '0x1.5c7dfe80b4d14p-2',
            '0x1.a4ae381348bf8p-3', '-0x1.fc1ea5d67ca1bp-3', '0x1.621125b9aa303p-4']},
        'far': {"ok": False, "iters": 300, "start": 5, "q_arm": [
            '-0x1.992e94a434de6p+0', '-0x1.f0ac240278e00p-3', '0x1.bd33dacf090cdp+0', '0x1.175f7fb1b5930p-1',
            '0x1.e5ba0b45cda87p+0', '-0x1.7c1702ed08081p-3', '-0x1.126a4d2cdd5f4p-2']},
    },
    'g1_v2': {
        'free': {"ok": True, "iters": 10, "start": 0, "q_arm": [
            '-0x1.3beef0d2e63e1p-1', '0x1.6bc0f0ca12286p-6', '0x1.35359ec96db30p-3', '0x1.5869913751714p-2',
            '-0x1.f821836d1519ep-4', '-0x1.0db32057c90ddp-2', '-0x1.4fa30b401f252p-6']},
        'roll': {"ok": True, "iters": 10, "start": 0, "q_arm": [
            '-0x1.29c63e9aaa598p-1', '0x1.291f4f98ae2c9p-2', '0x1.b85258f47ba75p-4', '0x1.6453b34291ce7p-2',
            '0x1.a5a7c69f4f533p-3', '-0x1.0773ff12595edp-2', '0x1.6749cf64a734cp-4']},
        'far': {"ok": False, "iters": 300, "start": 6, "q_arm": [
            '-0x1.df73fb411db0ap+0', '-0x1.b7a177e4cfa88p-4', '0x1.2c64905335487p+1', '0x1.4226671cf5fe4p-1',
            '-0x1.4d63941aa9072p-2', '0x1.a8a1610843320p-1', '-0x1.58f3f966088f0p-5']},
    },
}


def _ik(variant):
    rc = config.apply_head_camera(config.apply_robot_variant(config.load_all(), variant), "d435i")["robot"]
    kin = Kinematics(rc["urdf"])
    return rc, kin, ArmIK(kin, rc)


def _cases(kin, ik):
    """The recorded inputs: the tip pose of a fixed arm configuration (reachable), and that point 0.5 m farther in x
    (unreachable: every restart runs, drawing from the IK's own rng)."""
    q0 = pin.neutral(kin.model)
    q1 = q0.copy()
    q1[ik.qi] = [-0.6, 0.3, 0.1, 0.4, 0.2, -0.3, 0.1]
    T = kin.pose(q1, ik.frame)
    return q0, T[:3, 3], T[:3, 0], T[:3, :3] @ ik.img_x_in_tip


def _hex(q):
    return [float(x).hex() for x in q]


@pytest.mark.parametrize("variant", ["g1_v1", "g1_v2"])
def test_default_roll_is_bit_identical_to_the_pre_change_solver(variant):
    _, kin, ik = _ik(variant)
    q0, p, d, L = _cases(kin, ik)
    ref = REF[variant]
    runs = {"free": ik.solve(q0, p, d),
            "roll": ik.solve(q0, p, d, roll_axis=L),
            "roll_neg": ik.solve(q0, p, d, roll_axis=-L),                    # sign free: the same solution
            "stall_roll": ik.solve(q0, p, d, (100, 0.0005), L, rng=np.random.default_rng(3)),
            "far": ik.solve(q0, p + np.array([0.5, 0.0, 0.0]), d, roll_axis=L)}
    runs["roll_false"] = ik.solve(q0, p, d, roll_axis=L, roll_signed=False)     # explicit default
    for name, r in runs.items():
        exp = ref["far" if name == "far" else "free" if name == "free" else "roll"]
        assert _hex(r["q_arm"]) == exp["q_arm"], name
        assert (r["ok"], r["iters"], r["start"]) == (exp["ok"], exp["iters"], exp["start"]), name


def test_roll_target_keeps_the_given_sign_only_when_signed():
    _, kin, ik = _ik("g1_v2")
    d = np.array([1.0, 0.0, 0.0])
    L = np.array([0.0, 0.3, 1.0])                                   # not orthogonal to d after a tilt: projected first
    Lp = np.array([0.0, 0.3, 1.0]) / np.linalg.norm([0.0, 0.3, 1.0])
    R_tgt_pos = ik.roll_target(np.eye(3), d, L, signed=True)
    for R in (np.eye(3), R_tgt_pos, ik.roll_target(np.eye(3), d, -L, signed=True)):
        Rs = ik.roll_target(R, d, L, signed=True)
        assert np.allclose(Rs @ ik.img_x_in_tip, Lp) and np.allclose(Rs[:3, 0], d)
        Ru = ik.roll_target(R, d, L)                                # nearest sign to R
        x_now = R @ ik.img_x_in_tip
        assert np.allclose(Ru @ ik.img_x_in_tip, Lp if x_now @ Lp >= 0 else -Lp)
    # the current orientation has the image axis along -L: unsigned keeps -L, signed turns to +L
    R_neg = ik.roll_target(np.eye(3), d, -L, signed=True)
    assert np.allclose(ik.roll_target(R_neg, d, L) @ ik.img_x_in_tip, -Lp)
    assert np.allclose(ik.roll_target(R_neg, d, L, signed=True) @ ik.img_x_in_tip, Lp)
    assert ik.roll_error_deg(R_neg, d, L) == pytest.approx(0.0, abs=1e-6)
    assert ik.roll_error_deg(R_neg, d, L, signed=True) == pytest.approx(180.0, abs=1e-6)
    assert ik.roll_error_deg(R_tgt_pos, d, L, signed=True) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("p", [[0.35, 0.20, 0.10], [0.30, 0.25, 0.20]])
def test_signed_look_pose_puts_the_d405_on_the_requested_side(p):
    """g1_v2, scan axis +x of the root (facing a wall), image horizontal vertical (the 90/270 rolls of a look pose). The
    D405 sits 4.5 cm from the scan axis along -(x_tip x img_x) (URDF), so the requested sign of the roll axis L decides
    its side: its offset along d x L must be about -4.5 cm for both signs (FK on the solution)."""
    rc, kin, ik = _ik("g1_v2")
    q0 = pin.neutral(kin.model)
    d = np.array([1.0, 0.0, 0.0])
    T_n = kin.pose(q0, ik.frame)
    c_tip = (np.linalg.inv(T_n) @ kin.pose(q0, rc["hand_camera"]["frame"]))[:3, 3]
    side = float(c_tip @ np.cross([1.0, 0.0, 0.0], ik.img_x_in_tip))           # D405 offset along x_tip x img_x
    assert side == pytest.approx(-0.045, abs=0.002)
    sides = []
    for L in (np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, -1.0])):
        r = ik.solve(q0, p, d, roll_axis=L, roll_signed=True)
        assert r["ok"], (p, L)
        T = kin.pose(r["q_full"], ik.frame)
        assert (T[:3, :3] @ ik.img_x_in_tip) @ L > np.cos(np.radians(ik.cfg["tol_angle_deg"]))
        assert r["roll_err_deg"] < ik.cfg["tol_angle_deg"]
        off = kin.pose(r["q_full"], rc["hand_camera"]["frame"])[:3, 3] - T[:3, 3]
        assert float(off @ np.cross(d, L)) == pytest.approx(side, abs=0.005)
        sides.append(np.sign(off[1]))
    assert sides[0] == -sides[1]                                    # the two signs put the D405 on opposite sides (y)
