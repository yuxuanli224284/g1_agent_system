"""Per-candidate stance evaluation for Toolbox.plan_stance (reach prefilter, IK of the required poses, posture cost),
as a pure function so it runs the same serially or in a process pool (2026-11-07: plan_stance took 35-112 s per
squatting barcode with the roll constraint, all of it independent IK per grid point).

Workers are started with the spawn method (no inherited CUDA context from the detector in the parent) and die with
their parent (PR_SET_PDEATHSIG; a forked child that outlived its parent once kept
holding GPU memory). They import only numpy / pinocchio / the kinematics modules.

Determinism: the IK random restart of a candidate draws from its own generator seeded by (level, grid index), so the
serial and the pooled evaluation give identical results."""
from __future__ import annotations

import math

import numpy as np

_W = None          # worker state: {"kin", "ik", "shoulder", "reach"}


def _die_with_parent():
    import ctypes
    import signal
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    PR_SET_PDEATHSIG = 1
    if libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL) != 0:
        raise OSError(ctypes.get_errno(), "prctl(PR_SET_PDEATHSIG) failed")


def init_worker(urdf: str, robot_cfg: dict, side: str, shoulder_frame: str, arm_reach: float | None):
    """Process-pool initializer: own kinematics + IK of the parent's robot variant (planning: 1 restart, as plan_stance
    sets it). The URDF and the resolved robot config come from the parent: a spawned worker re-reading the configs would
    not know the variant the robot reported."""
    global _W
    _die_with_parent()
    from ..kinematics.fk import Kinematics
    from ..kinematics.ik import ArmIK
    kin = Kinematics(urdf)
    ik = ArmIK(kin, robot_cfg, side)
    ik.cfg = dict(ik.cfg, restarts=1)
    _W = {"kin": kin, "ik": ik, "shoulder": shoulder_frame, "reach": arm_reach}


def _rot_inv(T):
    R = T[:3, :3].T
    return R, -R @ T[:3, 3]


def target_in_root(t, Tc, offset):
    R, p0 = _rot_inv(Tc)
    pos = np.asarray(t["pos"], float) + (offset - t["offset"]) * (-np.asarray(t["forward"], float))
    return R @ pos + p0, R @ np.asarray(t["forward"], float)


def roll_in_root(t, Tc):
    L = t.get("long_axis")
    if L is None:
        return None
    return Tc[:3, :3].T @ np.asarray(L, float)


def manipulability(kin, ik, q) -> float:
    J = kin.jacobian(q, ik.frame)[:3, ik.vi]
    return float(math.sqrt(max(np.linalg.det(J @ J.T), 0.0)))


def eval_candidate(kin, ik, shoulder, reach, job) -> dict:
    """job: {"Tc", "q_lv", "t", "finals", "pre_dists", "stall", "seed"}. Returns {"status": "reach" | "ik" | "ok"}; ok
    adds "sols" (required finals then the chosen pre-pose), "gap", "margin", "man"."""
    t, Tc, q_lv = job["t"], job["Tc"], job["q_lv"]
    rng = np.random.default_rng(job["seed"])
    if reach is not None:
        # the closest required final target farther from the shoulder than the arm chain is long is never reachable
        sh = (Tc @ kin.pose(q_lv, shoulder))[:3, 3]
        pts = [np.asarray(t["pos"], float) + (off - t["offset"]) * (-np.asarray(t["forward"], float)) for off in job["finals"]]
        if min(np.linalg.norm(q_ - sh) for q_ in pts) > reach:
            return {"status": "reach"}
    roll = roll_in_root(t, Tc)
    sols = []
    for off in job["finals"]:                         # required: stop at the first unreachable one
        pr, dr = target_in_root(t, Tc, off)
        sols.append(ik.solve(q_lv, pr, dr, job["stall"], roll, rng=rng))
        if not sols[-1]["ok"]:
            return {"status": "ik"}
    gap = None
    for dp in job["pre_dists"]:                       # farthest reachable pre-pose
        pr, dr = target_in_root(t, Tc, dp)
        sp = ik.solve(q_lv, pr, dr, job["stall"], roll, rng=rng)
        if sp["ok"]:
            gap = float(dp - t["offset"])
            sols.append(sp)
            break
    if gap is None:
        return {"status": "ik"}
    return {"status": "ok", "sols": sols, "gap": gap, "margin": min(s["limit_margin_min"] for s in sols),
            "man": min(manipulability(kin, ik, s["q_full"]) for s in sols)}


def eval_batch(jobs):
    """Pool task: evaluate a list of jobs in the worker."""
    return [eval_candidate(_W["kin"], _W["ik"], _W["shoulder"], _W["reach"], j) for j in jobs]
