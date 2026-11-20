"""M1 (processes talk) + M2 (FK of the client == simulator) check. Needs the sim server running.

The agent model is the one of the robot variant the server reports (robot info robot_variant, config.apply_robot_variant):
this is the acceptance test of an agent URDF (scripts/export_urdf.py). M2 compares, at the start pose and with the arm
raised, the client's FK against the simulator's own poses of every frame the agent uses: head colour camera, head
left infrared imager (D435i), scanner imager, scanner tip and every hand camera (g1_v2: left_d405). Tolerance: < 5 mm /
< 1 deg (unchanged from the first M2 check).

python scripts/check_m1_m2.py [--scenario t2_L0_0000]   (g1_agent env; G1_EVAL_TOKEN from runs/.eval_token)
"""
import argparse, json, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
from g1agent.common import config
from g1agent.common.transforms import inv_T, angle_between
from g1agent.eval.eval_client import EvalClient
from g1agent.kinematics.fk import Kinematics
from g1agent.kinematics.ik import ArmIK
from g1agent.robot.sim_client import SimRobot

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=os.path.join(ROOT, "runs", "check_m1_m2.json"))
ap.add_argument("--scenario", default="t2_L0_0000")
args = ap.parse_args()
robot, ev = SimRobot(), EvalClient()
res = {}
t = time.time(); res["ping"] = robot.ping(); res["ping_s"] = round(time.time() - t, 3)
res["reset"] = ev.reset(args.scenario)
info = robot.get_robot_info()
# the model of the robot the simulator built (system.yaml robot_variant / head_camera, or $G1_ROBOT_VARIANT / $G1_HEAD_CAMERA)
cfg = config.apply_head_camera(config.apply_robot_variant(config.load_all(), info.get("robot_variant")), info.get("head_camera_model"))
RC = cfg["robot"]
kin = Kinematics(RC["urdf"])
ik = ArmIK(kin, RC)
res.update(robot_variant=info.get("robot_variant"), head_camera_model=info.get("head_camera_model"), urdf=RC["urdf"],
           hand_cameras={n: {k: v for k, v in h.items() if k != "K"} for n, h in (info.get("hand_cameras") or {}).items()})
h = robot.get_head_obs(depth=True); w = robot.get_wrist_obs(RC["arm_side"])
res["head_obs"] = {"rgb": list(h["rgb"].shape), "depth": list(h["depth"].shape), "K00": float(h["K"][0, 0])}
res["wrist_obs"] = {"rgb": list(w["rgb"].shape), "K00": float(w["K"][0, 0])}
res["hand_obs"] = {}
for n in info.get("hand_cameras") or {}:
    o = robot.get_hand_camera_obs(n)
    res["hand_obs"][n] = {"rgb": list(o["rgb"].shape), "K00": float(o["K"][0, 0])}


def pose_err(T_fk, T_sim):
    T_sim = np.asarray(T_sim, float)
    dp = float(np.linalg.norm(T_fk[:3, 3] - T_sim[:3, 3]))
    R = T_fk[:3, :3].T @ T_sim[:3, :3]
    da = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
    return {"pos_err_mm": round(dp * 1000, 3), "rot_err_deg": round(da, 4)}


def fk_check(tag):
    js = robot.get_joint_state(); h = robot.get_head_obs(); w = robot.get_wrist_obs(RC["arm_side"])
    sc = robot.scan()                              # its reply carries the scanner tip pose the simulator used
    q = kin.q_from_state(js["names"], js["q"])
    pairs = [("head", h["T_root_headcam"], RC["head_cam_frame"]),
             ("scanner_imager", w["T_root_wristcam"], RC["scanner_cam_frame"])]
    if RC["head_ir_left_frame"] and "T_root_ir_left" in h:
        pairs.append(("head_ir_left", h["T_root_ir_left"], RC["head_ir_left_frame"]))
    if "T_root_scanner_tip" in sc:
        pairs.append(("scanner_tip", sc["T_root_scanner_tip"], RC["wrist_point_frame"]))
    else:
        res[f"fk_{tag}_scanner_tip"] = f"scan() reply has no T_root_scanner_tip (keys {sorted(sc)})"
    for n in info.get("hand_cameras") or {}:
        frame = f"hand_cam_{n}_optical"
        pairs.append((n, robot.get_hand_camera_obs(n)["T_root_cam"], frame))
    out = {name: pose_err(kin.pose(q, frame), T_sim) for name, T_sim, frame in pairs}
    out["pass"] = all(v["pos_err_mm"] < 5 and v["rot_err_deg"] < 1 for v in out.values() if isinstance(v, dict))
    res[f"fk_{tag}"] = out


fk_check("start")
# arm: IK to a scanner pose in front of the chest, set_arm_joints. Done at the start pose, BEFORE the walk: the
# target is 0.40 m ahead of the root, and after walking 0.3 m towards the t2 table it lies inside box B1 (the arm
# is then blocked by the box, measured on g1_v2: 0.38 m tip error; from the start pose it is in free space)
js = robot.get_joint_state(); q = kin.q_from_state(js["names"], js["q"])
sol = ik.solve(q, [0.40, 0.18, 0.15], [1.0, 0.0, 0.0])
res["ik"] = {k: sol[k] for k in ("ok", "pos_err", "angle_err_deg", "iters")}
r = robot.set_arm_joints(RC["arm_side"], sol["q_arm"], 2.0); robot.stand(1.0)
js = robot.get_joint_state(); q = kin.q_from_state(js["names"], js["q"])
T = kin.pose(q, RC["wrist_point_frame"])
res["arm_reached"] = {"tip_err_m": float(np.linalg.norm(T[:3, 3] - [0.40, 0.18, 0.15])), "dir_err_deg": float(np.degrees(angle_between(T[:3, 0], [1, 0, 0]))),
                      "q_track_err_rad": float(np.max(np.abs(np.asarray(r["q_cmd"]) - np.asarray(r["q_now"]))))}
fk_check("arm_moved")
# walk
od0 = robot.get_odom()["T_odom_root"]
res["walk"] = robot.walk(0.3, 0.0, 0.0, 1.5)
od1 = robot.get_odom()["T_odom_root"]
res["walk_odom_dx"] = float((inv_T(od0) @ od1)[0, 3])
res["scan"] = {k: v for k, v in robot.scan().items() if k in ("status", "content")}
# image sizes as the robot reports them (robot info), not literals of one camera model
sizes_ok = (res["head_obs"]["rgb"][:2] == list(info["head_size"]) and res["wrist_obs"]["rgb"][:2] == list(info["wrist_size"])
            and all(res["hand_obs"][n]["rgb"][:2] == list(hc["size"]) for n, hc in (info.get("hand_cameras") or {}).items()))
res["M1_pass"] = sizes_ok and res["walk_odom_dx"] > 0.05 and res["arm_reached"]["tip_err_m"] < 0.05   # open loop (PD sag); move_arm adds closed-loop correction
res["M2_pass"] = res["fk_start"]["pass"] and res["fk_arm_moved"]["pass"]
json.dump(res, open(args.out, "w"), indent=1, default=str)
print(json.dumps({k: v for k, v in res.items() if k not in ("ping",)}, indent=1, default=str))
