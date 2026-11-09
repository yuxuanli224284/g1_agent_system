"""M1 (processes talk) + M2 (FK of the client == simulator) check. Needs the sim server running.

python scripts/check_m1_m2.py [--vlm]      (g1_agent env; G1_EVAL_TOKEN from runs/.eval_token)
"""
import argparse, json, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
from g1agent.common.transforms import inv_T, angle_between
from g1agent.eval.eval_client import EvalClient
from g1agent.kinematics.fk import Kinematics
from g1agent.kinematics.ik import ArmIK
from g1agent.robot.sim_client import SimRobot

ap = argparse.ArgumentParser(); ap.add_argument("--out", default=os.path.join(ROOT, "runs", "check_m1_m2.json"))
args = ap.parse_args()
robot, ev, kin = SimRobot(), EvalClient(), Kinematics()
ik = ArmIK(kin)
res = {}
t = time.time(); res["ping"] = robot.ping(); res["ping_s"] = round(time.time() - t, 3)
res["reset"] = ev.reset("t2_L0_0000")
info = robot.get_robot_info()
h = robot.get_head_obs(); w = robot.get_wrist_obs("left")
res["head_obs"] = {"rgb": list(h["rgb"].shape), "depth": list(h["depth"].shape), "K00": float(h["K"][0, 0])}
res["wrist_obs"] = {"rgb": list(w["rgb"].shape), "K00": float(w["K"][0, 0])}


def fk_check(tag):
    js = robot.get_joint_state(); h = robot.get_head_obs(); w = robot.get_wrist_obs("left")
    q = kin.q_from_state(js["names"], js["q"])
    out = {}
    for name, T_sim, frame in (("head", h["T_root_headcam"], "head_cam_optical"), ("wrist", w["T_root_wristcam"], "wrist_cam_optical")):
        T_fk = kin.pose(q, frame)
        dp = float(np.linalg.norm(T_fk[:3, 3] - T_sim[:3, 3]))
        R = T_fk[:3, :3].T @ T_sim[:3, :3]
        da = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
        out[name] = {"pos_err_mm": round(dp * 1000, 3), "rot_err_deg": round(da, 4)}
    out["pass"] = all(v["pos_err_mm"] < 5 and v["rot_err_deg"] < 1 for v in out.values() if isinstance(v, dict))
    res[f"fk_{tag}"] = out


fk_check("start")
# walk
od0 = robot.get_odom()["T_odom_root"]
res["walk"] = robot.walk(0.3, 0.0, 0.0, 1.5)
od1 = robot.get_odom()["T_odom_root"]
res["walk_odom_dx"] = float((inv_T(od0) @ od1)[0, 3])
# arm: IK to a scanner pose in front of the chest, set_arm_joints
js = robot.get_joint_state(); q = kin.q_from_state(js["names"], js["q"])
sol = ik.solve(q, [0.40, 0.18, 0.15], [1.0, 0.0, 0.0])
res["ik"] = {k: sol[k] for k in ("ok", "pos_err", "angle_err_deg", "iters")}
r = robot.set_arm_joints("left", sol["q_arm"], 2.0); robot.stand(1.0)
js = robot.get_joint_state(); q = kin.q_from_state(js["names"], js["q"])
T = kin.pose(q, "scanner_tip")
res["arm_reached"] = {"tip_err_m": float(np.linalg.norm(T[:3, 3] - [0.40, 0.18, 0.15])), "dir_err_deg": float(np.degrees(angle_between(T[:3, 0], [1, 0, 0]))),
                      "q_track_err_rad": float(np.max(np.abs(np.asarray(r["q_cmd"]) - np.asarray(r["q_now"]))))}
fk_check("arm_moved")
res["scan"] = {k: v for k, v in robot.scan().items() if k in ("status", "content")}
res["M1_pass"] = res["head_obs"]["rgb"] == [480, 640, 3] and res["wrist_obs"]["rgb"] == [480, 640, 3] and res["walk_odom_dx"] > 0.05 and res["arm_reached"]["tip_err_m"] < 0.05   # open loop (PD sag); move_arm adds closed-loop correction
res["M2_pass"] = res["fk_start"]["pass"] and res["fk_arm_moved"]["pass"]
json.dump(res, open(args.out, "w"), indent=1, default=str)
print(json.dumps({k: v for k, v in res.items() if k not in ("ping",)}, indent=1, default=str))
