"""Export the kinematic tree of the simulated robot from its USD to URDF, so the client's pinocchio model is exactly the
simulated robot. G1: 29 DoF + Dex1 + scanner; T800: EngineAI T800 Pro (43 joints) + scanner in the right hand.

Every UsdPhysics joint becomes a URDF joint: origin = (localPos0, localRot0) * inv(localRot1) (localPos1 must be 0),
axis = localRot1 * physics:axis (so the motion is the same). Extra fixed frames (from the simulator configs):
head_cam_optical (camera parent link * camera offset: G1 camera_head.yaml on d435_link, T800 the robot_t800.yaml
head_camera chain on LINK_HEAD_YAW), scanner_mount, scanner_tip, wrist_cam_optical (scanner_mount.parent_link *
scanner transforms). Optical frames use the ROS convention (x right, y down, z forward).

--source env (default): the live stage of G1BarcodeEnv (Isaac Sim, GPU), robot prim /World/envs/env_0/Robot, extras
  from the env. Runs in unitree_sim_env.
--source usd: the robot USD file itself, read with pxr on the CPU (no Isaac Sim app; the stage the env spawns
  references this file unchanged), extras from the same simulator configs and pure functions the env uses
  (g1_barcode_sim config.load_all, scanner_frames.transforms). pxr lives in the Isaac Sim install: run through
  scripts/usd_python.sh. --usd defaults to the robot file's asset.usd (T800); G1 needs --usd.

python scripts/export_urdf.py                                           -> assets/g1_dex1_scanner.urdf (G1, live env)
scripts/usd_python.sh scripts/export_urdf.py --robot t800 --source usd  -> assets/t800pro_scanner.urdf
The output path is the selected robot file's `urdf` (configs/robot.yaml, configs/robot_t800.yaml).
"""
import argparse, math, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from g1agent.common import config  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--robot", choices=sorted(config.ROBOT_FILES), default=None,
                help=f"robot (sets ${config.ROBOT_ENV}; default: ${config.ROBOT_ENV} or g1)")
ap.add_argument("--source", choices=("env", "usd"), default="env")
ap.add_argument("--usd", default=None, help="--source usd: robot USD file (default: the sim robot file's asset.usd)")
ap.add_argument("--out", default=None, help="output path (default: the robot file's urdf)")
args = ap.parse_args()
if args.robot is not None:
    os.environ[config.ROBOT_ENV] = args.robot
MODEL = config.robot_model()
RC = config.load("robot")
SIM = config.load("system")["sim_server"]["sim_root"]
sys.path.insert(0, SIM)
import numpy as np  # noqa: E402
from g1_barcode_sim.camera import make_T, quat_wxyz_to_R  # noqa: E402

URDF_NAME = {"g1": "g1_29dof_dex1_scanner", "t800": "t800pro_scanner"}   # <robot name=...> (G1: the original name)
OVERRIDES = None if MODEL == "g1" else {"robot": {"model": MODEL}}       # G1: exactly the original env call


def rpy(R):
    # URDF rpy: R = Rz(y) Ry(p) Rx(r)
    p = math.asin(max(-1.0, min(1.0, -R[2, 0])))
    if abs(math.cos(p)) > 1e-9:
        r, y = math.atan2(R[2, 1], R[2, 2]), math.atan2(R[1, 0], R[0, 0])
    else:
        r, y = math.atan2(-R[1, 2], R[1, 1]), 0.0
    return r, p, y


def origin(T):
    r, p, y = rpy(T[:3, :3])
    return f'<origin xyz="{T[0, 3]:.7f} {T[1, 3]:.7f} {T[2, 3]:.7f}" rpy="{r:.9f} {p:.9f} {y:.9f}"/>'


def joints_to_urdf(stage, robot):
    """(links, joint lines) of every UsdPhysics joint below the prim path `robot`."""
    from pxr import UsdPhysics
    links, joints = set(), []
    for prim in stage.Traverse():
        if not str(prim.GetPath()).startswith(robot) or not prim.IsA(UsdPhysics.Joint):
            continue
        j = UsdPhysics.Joint(prim)
        b0, b1 = j.GetBody0Rel().GetTargets(), j.GetBody1Rel().GetTargets()
        if not b0 or not b1:
            continue
        parent, child = str(b0[0]).split("/")[-1], str(b1[0]).split("/")[-1]
        lp0, lr0, lp1, lr1 = (j.GetLocalPos0Attr().Get(), j.GetLocalRot0Attr().Get(), j.GetLocalPos1Attr().Get(), j.GetLocalRot1Attr().Get())
        if np.linalg.norm(np.array(lp1)) > 1e-6:
            raise RuntimeError(f"{prim.GetPath()}: localPos1 {lp1} != 0 not supported")
        q0 = [lr0.GetReal(), *lr0.GetImaginary()]; q1 = [lr1.GetReal(), *lr1.GetImaginary()]
        R1 = quat_wxyz_to_R(q1)
        T = make_T(list(lp0), q0); T[:3, :3] = T[:3, :3] @ R1.T
        kind = prim.GetTypeName()
        if kind == "PhysicsFixedJoint":
            jt, axis, lim = "fixed", None, None
        else:
            tok = prim.GetAttribute("physics:axis").Get()
            # The T800 USD (Isaac URDF importer, g1_barcode_sim t800/convert_urdf.py) authors an EMPTY axis token on
            # every joint whose URDF axis is not a coordinate axis and puts that axis into localRot0 / localRot1, so
            # the joint turns about its frame's X (the UsdPhysics schema default). Checked: localRot1 * X equals the
            # vendor URDF <axis> of every such joint (tests/test_t800.py compares the FK with the vendor URDF).
            # Every G1 joint authors X / Y / Z, so the G1 export is unchanged.
            if tok == "":
                tok = "X"
            ax = {"X": [1, 0, 0], "Y": [0, 1, 0], "Z": [0, 0, 1]}[tok]
            axis = R1 @ np.array(ax, float)
            lo, hi = prim.GetAttribute("physics:lowerLimit").Get(), prim.GetAttribute("physics:upperLimit").Get()
            if kind == "PhysicsRevoluteJoint":
                jt, lim = "revolute", (math.radians(lo), math.radians(hi))
            elif kind == "PhysicsPrismaticJoint":
                jt, lim = "prismatic", (lo, hi)
            else:
                raise RuntimeError(f"unsupported joint type {kind}")
        links.update([parent, child])
        s = f'  <joint name="{prim.GetName()}" type="{jt}"><parent link="{parent}"/><child link="{child}"/>{origin(T)}'
        if axis is not None:
            s += f'<axis xyz="{axis[0]:.7f} {axis[1]:.7f} {axis[2]:.7f}"/><limit lower="{lim[0]:.7f}" upper="{lim[1]:.7f}" effort="100" velocity="10"/>'
        joints.append(s + "</joint>")
    return links, joints


def write(links, joints, cc, mount, T_body_mount, T_body_tip, T_body_wcam):
    """Add the extra frames (cc = camera cfg, mount = robot scanner_mount cfg) and write the URDF."""
    if cc["offset_convention"] != "ros":
        raise ValueError("camera offset_convention must be 'ros' (head_cam_optical is an optical frame)")
    extra = [("head_cam_optical", cc["prim_parent_link"], make_T(cc["offset_pos"], cc["offset_rot_wxyz"])),
             ("scanner_mount", mount["parent_link"], T_body_mount),
             ("scanner_tip", mount["parent_link"], T_body_tip),
             ("wrist_cam_optical", mount["parent_link"], T_body_wcam)]
    for name, parent, T in extra:
        if parent not in links:
            raise RuntimeError(f"extra frame {name}: parent link {parent} not in the exported tree")
        links.add(name)
        joints.append(f'  <joint name="{name}_joint" type="fixed"><parent link="{parent}"/><child link="{name}"/>{origin(T)}</joint>')
    xml = ['<?xml version="1.0"?>', '<!-- generated by scripts/export_urdf.py from the simulated USD; do not edit -->',
           f'<robot name="{URDF_NAME[MODEL]}">']
    xml += [f'  <link name="{l}"/>' for l in sorted(links)] + joints + ["</robot>"]
    out = args.out or os.path.join(ROOT, RC["urdf"])
    open(out, "w").write("\n".join(xml) + "\n")
    print("WROTE", out, len(links), "links", len(joints), "joints", flush=True)


if args.source == "usd":
    from pxr import Usd
    from g1_barcode_sim.config import load_all
    from g1_barcode_sim.scanner_frames import transforms
    cfg = load_all("s1_table", OVERRIDES)
    usd = args.usd or (cfg["robot"].get("asset") or {}).get("usd")
    if usd is None:
        raise SystemExit("--source usd: give --usd (the G1 robot file has no asset.usd)")
    stage = Usd.Stage.Open(usd)
    if not stage:
        raise SystemExit(f"cannot open {usd}")
    mc = cfg["robot"]["scanner_mount"]
    st = transforms(mc, cfg["scanner"])
    links, joints = joints_to_urdf(stage, str(stage.GetDefaultPrim().GetPath()))
    write(links, joints, cfg["camera"], mc, st["T_hand_G"], st["T_hand_tip"], st["T_hand_cam"])
    sys.exit(0)

from g1_barcode_sim.env import G1BarcodeEnv  # noqa: E402

env = G1BarcodeEnv("s1_table", headless=True, seed=0, config_overrides=OVERRIDES)
try:
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    links, joints = joints_to_urdf(stage, "/World/envs/env_0/Robot")
    write(links, joints, env.cfg["camera"], env.cfg["robot"]["scanner_mount"], env._T_body_mount, env._T_body_tip, env._T_body_wcam)
except BaseException:
    import traceback; traceback.print_exc(); sys.stdout.flush(); os._exit(1)
sys.stdout.flush(); os._exit(0)
