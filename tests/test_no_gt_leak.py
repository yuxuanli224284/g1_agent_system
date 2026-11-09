"""Static isolation check: orchestrators/, tools/, perception/, harness/, kinematics/, llm/ and
the robot client must not import the evaluation package, must not call ground-truth / eval-port methods and must not
import the simulator package (which exposes ground truth)."""
import ast
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALGO = ["g1agent/orchestrators", "g1agent/tools", "g1agent/perception", "g1agent/harness", "g1agent/kinematics", "g1agent/llm",
        "g1agent/robot/sim_client.py", "g1agent/robot/interface.py", "g1agent/robot/rpc.py"]
FORBIDDEN_MODULES = ("g1agent.eval", ".eval", "eval_client", "evaluator", "g1_barcode_sim")
FORBIDDEN_NAMES = {"get_gt", "get_ground_truth", "get_actual_state", "EvalClient", "evaluate", "T_world_root_gt",
                   "set_gt_markers", "task1_gt_obs_pose", "task2_gt_wrist_target"}


def _files():
    for p in ALGO:
        full = os.path.join(ROOT, p)
        if os.path.isfile(full):
            yield full
        else:
            for d, _, fs in os.walk(full):
                yield from (os.path.join(d, f) for f in fs if f.endswith(".py"))


def test_no_ground_truth_access():
    bad = []
    for f in _files():
        tree = ast.parse(open(f).read(), f)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                mod = ("." * node.level) + (node.module or "")
                if any(mod.endswith(x.lstrip(".")) and x.lstrip(".") in mod for x in FORBIDDEN_MODULES) or mod.startswith("..eval"):
                    bad.append(f"{f}: from {mod} import ...")
            elif isinstance(node, ast.Import):
                bad += [f"{f}: import {a.name}" for a in node.names if any(x.strip(".") in a.name for x in FORBIDDEN_MODULES)]
            elif isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_NAMES:
                bad.append(f"{f}:{node.lineno} .{node.attr}")
            elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
                bad.append(f"{f}:{node.lineno} {node.id}")
            elif isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in FORBIDDEN_NAMES:
                bad.append(f"{f}:{node.lineno} '{node.value}'")
    assert not bad, "ground-truth access in algorithm code:\n" + "\n".join(bad)


def test_robot_client_has_no_gt_methods():
    from g1agent.robot.sim_client import SimRobot
    assert not any(n in FORBIDDEN_NAMES for n in dir(SimRobot))
