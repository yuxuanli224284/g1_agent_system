"""M3: on the L0 task-2 scenes (robot at the true observation pose), head-camera perception of the
target face normal and the barcode centre vs ground truth. Criteria: normal error < 3 deg, barcode centre error < 2 cm.

python scripts/check_m3.py [--suite t2_L0] [--device cpu]
"""
import argparse, json, math, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
from g1agent.common import config
sys.path.insert(0, config.load("system")["sim_server"]["sim_root"])
from g1_barcode_sim.testsuite import load_suite
from g1agent.common.transforms import angle_between, apply, rotate
from g1agent.eval.eval_client import EvalClient
from g1agent.perception.box_detector import BoxDetector
from g1agent.robot.sim_client import SimRobot
from g1agent.tools.toolbox import Toolbox

ap = argparse.ArgumentParser(); ap.add_argument("--suite", default="t2_L0"); ap.add_argument("--device", default="cpu")
a = ap.parse_args()
robot, ev, det = SimRobot(), EvalClient(), BoxDetector(a.device)
rows = []
for sc in load_suite(a.suite):
    ev.reset(sc)
    tb = Toolbox(robot, box_detector=det)
    steps = [tb.detect_boxes(), tb.select_target_box(sc["target"]), tb.fit_front_face(), tb.detect_barcode_head(), tb.estimate_barcode_pose()]
    st = [s["status"] for s in steps]
    gt, act = ev.get_gt(), ev.get_actual_state()
    Two = np.asarray(act["T_world_odom_now"])
    r = {"scenario": sc["id"], "steps": st}
    if all(s == "ok" for s in st):
        face = tb.ws.boxes[tb.ws.target_box_id].face
        n_w = rotate(Two, face["normal"])
        r["normal_err_deg"] = math.degrees(angle_between(n_w, gt["target_face"]["normal"]))
        bc = next(iter(tb.ws.barcodes.values()))
        gbc = next(b for b in gt["barcodes"] if b["content"] == gt["target_barcode"])
        r["barcode_center_err_m"] = float(np.linalg.norm(apply(Two, bc.center) - np.asarray(gbc["center"])))
        r["face_center_err_m"] = float(np.linalg.norm(apply(Two, face["center"]) - np.asarray(gt["target_face"]["center"])))
        r["pass"] = r["normal_err_deg"] < 3.0 and r["barcode_center_err_m"] < 0.02
    else:
        r["pass"] = False
    rows.append(r)
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
ok = sum(r["pass"] for r in rows)
summary = {"suite": a.suite, "n": len(rows), "pass": ok,
           "normal_err_deg_max": max((r.get("normal_err_deg", float("nan")) for r in rows), default=None),
           "barcode_center_err_m_max": max((r.get("barcode_center_err_m", float("nan")) for r in rows), default=None)}
json.dump({"rows": rows, "summary": summary}, open(os.path.join(ROOT, "runs", f"check_m3_{a.suite}.json"), "w"), indent=1)
print("M3", json.dumps(summary))
