"""Run a list of scenarios with the composite episode video (+ result.json, timeline.md).
python scripts/record_episodes.py --mode rule --out runs/videos_rule t1_L0_0003 t2_L0_0000 ...
python scripts/record_episodes.py --mode rule --task 3 --out runs/videos_task3 t1_L0_0000   # merged find + scan task"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_task import make_episode_runner
from run_suite import check_variant

ap = argparse.ArgumentParser()
ap.add_argument("scenarios", nargs="+")
ap.add_argument("--mode", default="rule", choices=["rule", "agent_assist", "agent_full"])
ap.add_argument("--out", required=True)
ap.add_argument("--device", default=None)
ap.add_argument("--until_success", action="store_true", help="stop after the first successful episode of the list")
ap.add_argument("--task", type=int, default=None, choices=[1, 2, 3], help="override the scenarios' task (e.g. 3)")
a = ap.parse_args()
run = make_episode_runner(a.device)
for sc in a.scenarios:
    ep = os.path.join(a.out, sc)
    if os.path.exists(os.path.join(ep, "result.json")) and os.path.exists(os.path.join(ep, "episode.mp4")):
        print("SKIP", sc, flush=True)
        r = json.load(open(os.path.join(ep, "result.json")))
        # without --task the scenario's own task is wanted (result.json files written before the override existed
        # have no scenario_task: they always ran the scenario's task)
        want = a.task if a.task is not None else r.get("scenario_task", r["task"])
        if r["task"] != want:
            raise SystemExit(f"{ep} holds a task-{r['task']} episode, task {want} asked: use another --out")
        check_variant(ep, r)
    else:
        r = run(sc, a.mode, ep, video=True, task=a.task)
        print("EPISODE", sc, a.mode, "success", r["metrics"].get("success"), f"{r['wall_s']}s", flush=True)
    if a.until_success and r["metrics"].get("success"):
        break
print("RECORD_DONE", flush=True)
