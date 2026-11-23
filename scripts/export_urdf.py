"""Agent-side URDF of a simulated robot variant (the client's pinocchio model must be exactly the simulated robot).

g1_v1 (Isaac, unitree_sim_env): the kinematic tree of the simulated G1 (29 DoF + Dex1 + scanner) exported from its USD.
  Every UsdPhysics joint becomes a URDF joint: origin = (localPos0, localRot0) * inv(localRot1) (localPos1 must be 0),
  axis = localRot1 * physics:axis (so the motion is the same). Extra fixed frames (from the simulator configs):
  head_cam_optical (d435_link * camera_head.yaml offset; D435i: the colour camera), head_ir_left/right_optical (D435i
  infrared pair), head_cam_optical_upstream, scanner_mount, scanner_tip, wrist_cam_optical (left_hand_base_link *
  scanner transforms).
      python scripts/export_urdf.py --variant g1_v1   -> assets/g1_dex1_scanner.urdf

g1_v2 (offline, no Isaac; the g1_agent env is enough): the simulator imports this robot from a description URDF
  (g1_barcode_sim robot_variants.yaml g1_v2 robot.asset, <description_dir>/g1_v2_sim.urdf), so its kinematic tree IS that
  file. It is copied without its <visual> / <collision> elements (pinocchio's model needs links, joints and inertias;
  the mesh paths would dangle here), and the frames the simulator adds from its configs are appended, all read from the
  simulator's own (pure yaml / numpy) config loader, variant g1_v2 merged:
  - head_cam_optical, head_ir_left_optical, head_ir_right_optical on torso_link: the D435i colour / infrared cameras
    (camera_head.yaml d435i block resolved with the variant's mount and module_origin, camera.resolve_head_model). The
    mount must equal the URDF d435_joint (checked; the simulator merges d435_link into torso_link and uses the mount).
  - scanner_mount, scanner_tip, wrist_cam_optical (the scanner imager) on the scanner_mount parent link: the same chain
    as the simulator's scanner.transforms (scanner.py imports Isaac Lab at module level, so its 4 lines are re-done here
    with its R_G_OPT constant read from its source).
  - hand_cam_<name>_optical for every robot.yaml hand camera (left_d405): the measured camera pose of the variant
    (NOT the URDF's left_d405_depth_optical_frame, a placeholder ~10 cm from the D405 visual: variant comment).
  Optical frames use the ROS convention (x right, y down, z forward). The header records the source path and sha256.
      python scripts/export_urdf.py --variant g1_v2   -> assets/g1_v2_scanner.urdf
"""
import argparse
import ast
import hashlib
import math
import os
import sys
import xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from g1agent.common import config  # noqa: E402

SIM = config.load("system")["sim_server"]["sim_root"]
sys.path.insert(0, SIM)
import numpy as np  # noqa: E402

OUT = {"g1_v1": os.path.join(ROOT, "assets", "g1_dex1_scanner.urdf"),
       "g1_v2": os.path.join(ROOT, "assets", "g1_v2_scanner.urdf")}


def rpy(R):
    # URDF rpy: R = Rz(y) Ry(p) Rx(r)
    p = math.asin(max(-1.0, min(1.0, -R[2, 0])))
    if abs(math.cos(p)) > 1e-9:
        r, y = math.atan2(R[2, 1], R[2, 2]), math.atan2(R[1, 0], R[0, 0])
    else:
        r, y = math.atan2(-R[1, 2], R[1, 1]), 0.0
    return r, p, y


def origin_attrs(T):
    r, p, y = rpy(T[:3, :3])
    return {"xyz": f"{T[0, 3]:.7f} {T[1, 3]:.7f} {T[2, 3]:.7f}", "rpy": f"{r:.9f} {p:.9f} {y:.9f}"}


def origin(T):
    a = origin_attrs(T)
    return f'<origin xyz="{a["xyz"]}" rpy="{a["rpy"]}"/>'


# ---------------------------------------------------------------------------------------------- g1_v2 (offline)
def scanner_R_G_OPT() -> np.ndarray:
    """R_G_OPT of the simulator's scanner.py, read from its source (the module imports Isaac Lab at import time)."""
    src = os.path.join(SIM, "g1_barcode_sim", "scanner.py")
    for node in ast.parse(open(src).read(), src).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "R_G_OPT" for t in node.targets):
            call = node.value                     # np.array([[...], ...])
            return np.array(ast.literal_eval(call.args[0]), float)
    raise RuntimeError(f"R_G_OPT not found in {src}")


def description_urdf(rcfg: dict) -> str:
    """The variant's description URDF, resolved as the simulator does (g1_barcode_sim tasks/barcode_task.robot_urdf_path:
    asset.description_dir, else $G1_V2_DESCRIPTION, else assets/g1_description_v2 in the simulator repository)."""
    a = rcfg["asset"]
    d = a["description_dir"] or os.environ.get("G1_V2_DESCRIPTION") or os.path.join(SIM, "assets", "g1_description_v2")
    p = os.path.join(d, a["urdf"])
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    return p


def variant_frames(variant: str) -> dict:
    """Everything the agent URDF of an URDF-imported variant needs from the simulator's configs: {"source_urdf",
    "frames": [(name, parent link, T_parent_frame 4x4)], "mount": the camera mount}."""
    from g1_barcode_sim.camera import make_T
    from g1_barcode_sim.config import load_all
    cfg = load_all("scenario", {"robot": {"variant": variant}, "camera": {"model": "d435i"}})
    rc, sc, cc = cfg["robot"], cfg["scanner"], cfg["camera"]
    if (rc.get("asset") or {}).get("kind") != "urdf":
        raise ValueError(f"robot variant {variant} is not imported from a URDF (use the Isaac export)")
    if cc["offset_convention"] != "ros":
        raise ValueError("camera_head.yaml offset_convention must be 'ros'")
    pl = cc["prim_parent_link"]
    frames = [("head_cam_optical", pl, make_T(cc["offset_pos"], cc["offset_rot_wxyz"]))]
    frames += [(f"head_ir_{s}_optical", pl, make_T(cc["stereo"][f"{s}_pos"], cc["stereo"]["rot_wxyz"])) for s in ("left", "right")]
    mc = rc["scanner_mount"]
    T_hand_G = make_T(mc["pos"], mc["quat_wxyz"])
    T_G_tip = np.eye(4); T_G_tip[:3, 3] = sc["tip"]["pos"]
    T_tip_cam = np.eye(4); T_tip_cam[:3, :3] = scanner_R_G_OPT(); T_tip_cam[:3, 3] = sc["camera"]["pos_in_tip"]
    frames += [("scanner_mount", mc["parent_link"], T_hand_G), ("scanner_tip", mc["parent_link"], T_hand_G @ T_G_tip),
               ("wrist_cam_optical", mc["parent_link"], T_hand_G @ T_G_tip @ T_tip_cam)]
    for n, h in (rc.get("hand_cameras") or {}).items():
        frames.append((f"hand_cam_{n}_optical", h["parent_link"], make_T(h["pos"], h["rot_wxyz"])))
    return {"source_urdf": description_urdf(rc), "frames": frames, "mount": cc["mount"], "prim_parent_link": pl}


def export_offline(variant: str, out: str) -> dict:
    vf = variant_frames(variant)
    src = vf["source_urdf"]
    raw = open(src, "rb").read()
    tree = ET.ElementTree(ET.fromstring(raw))
    robot = tree.getroot()
    # the mount the camera frames were built with must be the description's d435_joint (the simulator turns that joint
    # into this mount when it merges d435_link into its parent)
    dj = next((j for j in robot.iter("joint") if j.get("name") == "d435_joint"), None)
    if dj is None:
        raise ValueError(f"{src}: no d435_joint")
    o = dj.find("origin")
    jpos = [float(v) for v in o.get("xyz").split()]
    jrpy = [float(v) for v in o.get("rpy").split()]
    m = vf["mount"]
    if dj.find("parent").get("link") != vf["prim_parent_link"] or not np.allclose(jpos, m["pos"], atol=1e-9) \
            or not np.allclose(jrpy, m["rpy"], atol=1e-9):
        raise ValueError(f"camera mount {m} / parent {vf['prim_parent_link']} != {src} d435_joint "
                         f"{dj.find('parent').get('link')} xyz {jpos} rpy {jrpy}")
    links = {l.get("name") for l in robot.iter("link")}
    for link in robot.findall("link"):
        for tag in ("visual", "collision"):
            for e in link.findall(tag):
                link.remove(e)
    for e in robot.findall("material"):
        robot.remove(e)
    for name, parent, T in vf["frames"]:
        if name in links:
            raise ValueError(f"frame {name} already exists in {src}")
        if parent not in links:
            raise ValueError(f"parent link {parent} of {name} not in {src}")
        ET.SubElement(robot, "link", name=name)
        j = ET.SubElement(robot, "joint", name=f"{name}_joint", type="fixed")
        ET.SubElement(j, "parent", link=parent)
        ET.SubElement(j, "child", link=name)
        ET.SubElement(j, "origin", **origin_attrs(T))
    ET.indent(tree, space="  ")
    sha = hashlib.sha256(raw).hexdigest()
    head = ["<?xml version='1.0' encoding='utf-8'?>",
            f"<!-- generated by scripts/export_urdf.py (variant {variant}); do not edit.",
            f"     source (relative to the simulator repository): {os.path.relpath(src, SIM)}",   # machine-independent
            f"     source sha256: {sha}",
            f"     visual / collision / material elements removed; frames added from the simulator configs "
            f"(g1_barcode_sim robot_variants.yaml {variant}, scanner.yaml, camera_head.yaml): "
            + ", ".join(n for n, _, _ in vf["frames"]),
            f"     head camera mount (d435_joint): xyz {m['pos']} rpy {m['rpy']} (pitch {math.degrees(m['rpy'][1]):.2f} deg) -->"]
    xml = "\n".join(head) + "\n" + ET.tostring(robot, encoding="unicode") + "\n"
    open(out, "w").write(xml)
    return {"out": out, "source": src, "sha256": sha, "frames": [n for n, _, _ in vf["frames"]]}


# ---------------------------------------------------------------------------------------------- g1_v1 (Isaac)
def export_isaac(variant: str, out: str):
    from g1_barcode_sim.env import G1BarcodeEnv
    env = G1BarcodeEnv("s1_table", headless=True, seed=0, config_overrides={"robot": {"variant": variant}})
    try:
        from pxr import UsdPhysics
        import omni.usd
        from g1_barcode_sim.camera import make_T, quat_wxyz_to_R
        stage = omni.usd.get_context().get_stage()
        robot = "/World/envs/env_0/Robot"
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
                ax = {"X": [1, 0, 0], "Y": [0, 1, 0], "Z": [0, 0, 1]}[prim.GetAttribute("physics:axis").Get()]
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
        # extra frames
        # both head-camera models (camera_head.yaml model), whichever the env was built with: head_cam_optical = the D435i
        # colour camera, head_cam_optical_upstream = unitree's single camera, head_ir_left/right_optical = the D435i pair
        from g1_barcode_sim.config import load_camera_cfg
        cc, cu = load_camera_cfg({"model": "d435i"}), load_camera_cfg({"model": "upstream"})
        extra = [("head_cam_optical", cc["prim_parent_link"], make_T(cc["offset_pos"], cc["offset_rot_wxyz"])),
                 ("head_cam_optical_upstream", cu["prim_parent_link"], make_T(cu["offset_pos"], cu["offset_rot_wxyz"])),
                 ("scanner_mount", env.cfg["robot"]["scanner_mount"]["parent_link"], env._T_body_mount),
                 ("scanner_tip", env.cfg["robot"]["scanner_mount"]["parent_link"], env._T_body_tip),
                 ("wrist_cam_optical", env.cfg["robot"]["scanner_mount"]["parent_link"], env._T_body_wcam)]
        extra += [(f"head_ir_{s}_optical", cc["prim_parent_link"], make_T(cc["stereo"][f"{s}_pos"], cc["offset_rot_wxyz"]))
                  for s in ("left", "right")]
        for name, parent, T in extra:
            links.add(name)
            joints.append(f'  <joint name="{name}_joint" type="fixed"><parent link="{parent}"/><child link="{name}"/>{origin(T)}</joint>')
        xml = ['<?xml version="1.0"?>', '<!-- generated by scripts/export_urdf.py from the simulated USD; do not edit -->', '<robot name="g1_29dof_dex1_scanner">']
        xml += [f'  <link name="{l}"/>' for l in sorted(links)] + joints + ["</robot>"]
        open(out, "w").write("\n".join(xml) + "\n")
        print("WROTE", out, len(links), "links", len(joints), "joints", flush=True)
    except BaseException:
        import traceback; traceback.print_exc(); sys.stdout.flush(); os._exit(1)
    sys.stdout.flush(); os._exit(0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=sorted(OUT))
    a = ap.parse_args()
    if a.variant == "g1_v1":
        export_isaac(a.variant, OUT[a.variant])          # the upstream USD: needs Isaac (unitree_sim_env)
    else:
        print("WROTE", export_offline(a.variant, OUT[a.variant]), flush=True)
