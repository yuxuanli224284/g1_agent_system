"""Live check of the harness kinematics (robot file urdf) against the running simulator: the head camera, scanner tip and
scanner camera poses the simulator reports (root frame) vs pinocchio FK of the reported joint state, at the standing
pose and at random scanner-arm poses (set_arm_joints within the robot's soft joint limits, shrunk towards the home
pose by --spread). Needs a sim server of the same robot and $G1_EVAL_TOKEN.

G1_ROBOT=t800 SIM_INSTANCE=100 python scripts/check_fk_live.py --n 6 --out runs/fk_live_t800.json
"""
import argparse, json, math, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from g1agent.common import config  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--robot", choices=sorted(config.ROBOT_FILES), default=None)
ap.add_argument("--n", type=int, default=6, help="random arm poses")
ap.add_argument("--spread", type=float, default=0.5, help="fraction of the soft range around the home pose sampled")
ap.add_argument("--duration", type=float, default=2.0, help="set_arm_joints ramp time (s)")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default=None)
args = ap.parse_args()
if args.robot is not None:
    os.environ[config.ROBOT_ENV] = args.robot
import numpy as np  # noqa: E402
from g1agent.eval.eval_client import EvalClient  # noqa: E402
from g1agent.kinematics.fk import Kinematics  # noqa: E402
from g1agent.robot.sim_client import SimRobot  # noqa: E402

RC = config.load("robot")
robot, ev, kin = SimRobot(), EvalClient(), Kinematics()
info = robot.get_robot_info()
if info.get("robot_model", "g1") != RC["model"]:
    raise SystemExit(f"server robot {info.get('robot_model', 'g1')!r} != configured {RC['model']!r}")
side = RC["arm_side"]


def err(A, B):
    A, B = np.asarray(A, float), np.asarray(B, float)
    dR = A[:3, :3].T @ B[:3, :3]
    return {"trans_mm": round(float(np.linalg.norm(A[:3, 3] - B[:3, 3]) * 1e3), 4),
            "rot_deg": round(float(np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1.0, 1.0)))), 5)}


def measure(tag):
    js = robot.get_joint_state()
    q = kin.q_from_state(js["names"], js["q"])
    head = robot.get_head_obs()
    wrist = robot.get_wrist_obs(side)
    if wrist.get("status") != "ok":
        raise SystemExit(f"get_wrist_obs({side}): {wrist}")
    a = ev.rpc.call("get_actual_state")
    T_root_tip = np.linalg.inv(np.asarray(a["T_world_root"], float)) @ np.asarray(a["T_world_wrist_point"], float)
    return {"pose": tag,
            "head_cam": err(kin.pose(q, RC["head_cam_frame"]), head["T_root_headcam"]),
            "scanner_cam": err(kin.pose(q, RC["wrist_cam_frame"]), wrist["T_root_wristcam"]),
            "scanner_tip": err(kin.pose(q, RC["wrist_point_frame"]), T_root_tip),
            "arm_q": np.round(np.asarray(js["q"], float)[[js["names"].index(j) for j in RC["arm_joints"][side]]], 4).tolist()}


rows = [measure("standing")]
js = robot.get_joint_state()
home = np.asarray(js["q"], float)[[js["names"].index(j) for j in RC["arm_joints"][side]]]
lim = np.asarray(info["arm_soft_limits"][side], float)
rng = np.random.default_rng(args.seed)
for k in range(args.n):
    lo = home + args.spread * (lim[:, 0] - home)
    hi = home + args.spread * (lim[:, 1] - home)
    q = rng.uniform(lo, hi)
    r = robot.set_arm_joints(side, q, args.duration)
    if r["status"] != "ok":
        raise SystemExit(f"set_arm_joints: {r}")
    robot.stand(1.0)
    rows.append(measure(f"random_{k}"))
robot.set_arm_joints(side, home, args.duration)
robot.stand(1.0)
summary = {k: {m: max(r[k][m] for r in rows) for m in ("trans_mm", "rot_deg")} for k in ("head_cam", "scanner_cam", "scanner_tip")}
out = {"robot": RC["model"], "urdf": RC["urdf"], "n_poses": len(rows), "max_error": summary, "poses": rows}
print(json.dumps(out, indent=1))
if args.out:
    json.dump(out, open(args.out, "w"), indent=1)
