"""Offline scanner-arm reachability of a scenario's labels for the selected robot, standing (no squat), with the stance
grid and IK the harness uses (task2.yaml / task3.yaml stance, tools/stance_eval.eval_candidate, robot.yaml ik).

For every label (ground-truth layout of the scenario, world frame = odometry at the episode start) and every scanner
stand-off d in --standoffs: is there a stance on the plan_stance grid (root distance from the label's face in
[min_face_dist, max_face_dist], lateral offset in lateral_range, step; yaw facing the face) where the IK reaches the
scan pose at d (tip position + scan direction + roll: image horizontal along the label's long axis) and a pre-pose
(d + the task2 pre gap, shortened in grid steps down to d, as plan_stance)? The robot stands at its standing joint
state (T800: the walk policy's default pose, sim robot_t800.yaml; G1: the standing body calibration level) and root
height (the head-camera nominal file of its test suites). Not checked here: the clearance model (needs the simulator's
collision geometry) and the walk-tolerance robustness re-check; plan_stance adds both.

G1_ROBOT=t800 python scripts/reach_report.py --scenario wall6x5_L0_0000          (or --robot t800)
"""
import argparse, json, math, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from g1agent.common import config  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--robot", choices=sorted(config.ROBOT_FILES), default=None)
ap.add_argument("--scenario", default="wall6x5_L0_0000")
ap.add_argument("--standoffs", type=float, nargs="+", default=None,
                help="scanner stand-offs (m); default: task3.yaml standoff min_m, task2.yaml offset_m, standoff max_m")
ap.add_argument("--out", default=None, help="write the per-label table as JSON")
args = ap.parse_args()
if args.robot is not None:
    os.environ[config.ROBOT_ENV] = args.robot
MODEL = config.robot_model()
SIM = config.load("system")["sim_server"]["sim_root"]
sys.path.insert(0, SIM)
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from g1_barcode_sim.testsuite import load_factors, load_scenario  # noqa: E402
from g1_barcode_sim.testsuite.scenario import to_layout  # noqa: E402
from g1agent.common.transforms import T_from_xy_yaw  # noqa: E402
from g1agent.kinematics.fk import Kinematics  # noqa: E402
from g1agent.kinematics.ik import ArmIK  # noqa: E402
from g1agent.tools import stance_eval as SE  # noqa: E402

RC, T2, T3 = config.load("robot"), config.load("task2"), config.load("task3")
F = load_factors(MODEL)
nominal = yaml.safe_load(open(os.path.join(SIM, F["head_camera_nominal"])))
root_z = float(nominal["root_height"])
kin = Kinematics()
ik = ArmIK(kin)
ik.cfg = dict(ik.cfg, restarts=1)                       # as plan_stance
stall = (int(ik.cfg["plan_stall_window"]), float(ik.cfg["plan_stall_eps_m"]))
if MODEL == "t800":
    from g1_barcode_sim.config import load_robot_cfg
    from g1_barcode_sim.t800_robot import joint_tables
    q_stand = joint_tables(load_robot_cfg({"model": "t800"}))["default_q"]
else:
    q_stand = json.load(open(os.path.join(ROOT, RC["body"]["calibration"])))["levels"][0]["q"]
q0 = kin.q_from_state(list(q_stand), list(q_stand.values()))
sh_frame = RC["body"]["links"]["arm_chain"][0] if "links" in RC["body"] else f"{RC['arm_side']}_shoulder_pitch_link"

S = T2["stance"]
dists = np.arange(S["min_face_dist_m"], S["max_face_dist_m"] + 1e-9, S["step"])
lats = np.arange(S["lateral_range"][0], S["lateral_range"][1] + 1e-9, S["step"])
gap0 = float(T2["pre_offset_m"] - T2["offset_m"])
standoffs = args.standoffs or [float(T3["standoff"]["min_m"]), float(T2["offset_m"]), float(T3["standoff"]["max_m"])]
lay = to_layout(load_scenario(args.scenario), F)


def label_reach(bc):
    c = np.asarray(bc["center"], float)
    n = np.asarray(bc["normal"], float)
    nh = np.array([n[0], n[1], 0.0]) / np.linalg.norm(n[:2])
    yaw = math.atan2(-nh[1], -nh[0])
    left = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
    long_axis = np.asarray(bc["right"], float)          # the label's horizontal edge = the code's long axis (bars
                                                        # across it), sign free as in the harness roll constraint
    res = {"box_id": bc["box_id"], "z": round(float(c[2]), 3)}
    for d in standoffs:
        t = {"pos": c + d * n, "forward": -n, "offset": d, "long_axis": long_axis}
        pre = list(np.arange(d + gap0, d - 1e-9, -S["step"]))
        best = None
        for fd in dists:
            for l in lats:
                p = np.array([c[0], c[1], 0.0]) + fd * nh + l * left
                Tc = T_from_xy_yaw(p[0], p[1], yaw, root_z)
                job = {"Tc": Tc, "q_lv": q0, "t": t, "finals": [d], "pre_dists": pre, "stall": stall, "seed": [0, 0, 0]}
                ev = SE.eval_candidate(kin, ik, None, None, job)
                if ev["status"] == "ok" and (best is None or (-ev["gap"], -ev["man"]) < (-best["gap"], -best["man"])):
                    q = ev["sols"][0]["q_arm"]
                    best = {"face_dist": round(float(fd), 2), "lateral": round(float(l), 2), "gap": round(ev["gap"], 2),
                            "man": round(ev["man"], 5), "margin_rad": round(ev["margin"], 3), "q_final": np.round(q, 3).tolist()}
        res[f"d{d:.2f}"] = best
    return res


if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    with ProcessPoolExecutor(max_workers=int(os.environ.get("REACH_WORKERS", "8")), mp_context=mp.get_context("fork")) as pool:
        rows = list(pool.map(label_reach, lay["barcodes"]))
    for res in rows:
        print(json.dumps({k: (v if not isinstance(v, dict) else {kk: v[kk] for kk in ("face_dist", "lateral", "gap")})
                          for k, v in res.items()}), flush=True)
    # summary per label height (row)
    print(f"\n{MODEL} standing, root z {root_z} m, scenario {args.scenario}: labels reachable per height "
          f"(stance grid face {S['min_face_dist_m']}-{S['max_face_dist_m']} m, lateral {S['lateral_range']} m)")
    for z in sorted({r["z"] for r in rows}):
        rr = [r for r in rows if r["z"] == z]
        print(f"  z {z:.3f} m: " + ", ".join(f"d={d:.2f}: {sum(r[f'd{d:.2f}'] is not None for r in rr)}/{len(rr)}" for d in standoffs))
    if args.out:
        json.dump({"robot": MODEL, "scenario": args.scenario, "root_z": root_z, "standoffs": standoffs, "labels": rows},
                  open(args.out, "w"), indent=1)
