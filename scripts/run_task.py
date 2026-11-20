"""Run one scenario (one task) end to end and evaluate it.

python scripts/run_task.py --scenario t2_L0_0000 --mode rule [--task 3] [--run_dir runs/xxx] [--no_video]
--scenario: a scenario id of a generated suite (g1_barcode_sim suites/) or a scenario .json path.
--task overrides the scenario's task on the agent / evaluation side (e.g. --task 3 runs a t1 scenario as the merged
"find + scan all barcodes" task); the simulator is reset with the scenario unchanged.
Task-3 subset scenarios: "scan_targets" (the target in robot terms, e.g. columns 1-2 from the left, rows 2-3 from the top,
order rows) is the task instruction given to the agent; "target_boxes" (GT box ids) goes to the evaluator only.
Needs the sim server (scripts/start_sim_server.sh) and, for agent modes, the VLM server (scripts/start_vllm.sh).
Per episode: tool_calls.jsonl, llm.jsonl, stages.jsonl, images/, result.json (incl. stage states, full stage log + scan log), final.png,
episode.mp4 (composite video, unless --no_video), timeline.md / timeline.jsonl.
"""
import argparse, copy, json, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from g1agent.common import config  # noqa: E402

SIM = config.load("system")["sim_server"]["sim_root"]
sys.path.insert(0, SIM)
TASKS = (1, 2, 3)


def _eval_rpc(ev, method, **kw):
    """Eval-port call that surfaces an error reply (e.g. a server without the method) instead of returning it as data."""
    rep = ev.rpc.call(method, **kw)
    if isinstance(rep, dict) and rep.get("status") == "error":
        raise RuntimeError(f"eval port {method}: {rep.get('reason')}")
    return rep


def load_scenario_arg(scenario: str) -> dict:
    """A scenario id of a generated suite, or a path to a scenario .json file."""
    if scenario.endswith(".json"):
        with open(scenario) as f:
            return json.load(f)
    from g1_barcode_sim.testsuite import load_scenario
    return load_scenario(scenario)


def subset_eval_args(sc: dict, task: int) -> dict:
    """Evaluator arguments of a task-3 subset scenario: the GT target box ids and the order rule of its target spec."""
    if task != 3:
        return {}
    return {"target_boxes": sc.get("target_boxes"), "order_rule": (sc.get("scan_targets") or {}).get("order", "columns")}


def make_episode_runner(device=None):
    """Load the heavy parts once (detector, kinematics, clients); returns run(scenario, mode, ep_dir) -> result."""
    from g1agent.eval.eval_client import EvalClient
    from g1agent.eval.evaluator import evaluate
    from g1agent.harness.guards import Harness
    from g1agent.harness.logger import EpisodeLogger
    from g1agent.perception.box_detector import BoxDetector
    from g1agent.robot.sim_client import SimRobot
    from g1agent.tools.toolbox import Toolbox, R3
    robot, ev = SimRobot(), EvalClient()
    det = BoxDetector(device)
    cfg = config.load_all()
    llm = None

    def run(scenario, mode, ep_dir, video=True, task=None):
        """task: None = the scenario's task; 1/2/3 overrides it (agent + evaluation side only)."""
        nonlocal llm
        from make_timeline import build_timeline
        sc = load_scenario_arg(scenario) if isinstance(scenario, str) else scenario
        task = sc["task"] if task is None else int(task)
        if task not in TASKS:
            raise ValueError(f"unknown task {task!r} (known: {TASKS})")

        stage_ws = {}              # harness estimates at a stage, paired with the GT state the eval port stores there

        def on_event(e):           # harness log events -> composite-video overlays / stage states on the eval port
            if e["kind"] == "stage":
                if e["name"] == "task1_end":
                    # task-1 metrics of task 3 compare the GT state at this stage with the harness's root target AT
                    # this stage (a later compute_obs_pose in the scan phase would overwrite ws.root_target)
                    stage_ws["task1_end_root_target"] = copy.deepcopy(tb.ws.root_target)
                _eval_rpc(ev, "stage", name=e["name"], info=e["info"])
            else:
                _eval_rpc(ev, "annotate", event=e)

        log = EpisodeLogger(ep_dir, on_event=on_event)
        t0 = time.time()
        ev.reset(sc)
        ev.set_gt_markers(True)
        if video:
            ev.record_start(os.path.join(ep_dir, "episode.mp4"), task=task)   # task 3: adaptive GT stand-off in the METRICS panel
        tb = Toolbox(robot, box_detector=det)
        # task instructions (not ground truth): scan every side of the stack when the scenario asks for it; scan only
        # the labels of the boxes at the given places of the wall, in robot terms (scan_targets; the GT box ids,
        # target_boxes, never reach the agent)
        tb.multi_face = task == 3 and sc.get("scan_faces") == "all"
        if task == 3:
            tb.set_scan_targets(sc.get("scan_targets"))
        tcfg = cfg[f"task{task}"]
        h = Harness(tb, task, tcfg["budget"], log, cfg["robot"]["limits"])
        # tasks 1/2 work on the scenario's left/right target box; task 3 covers every box (task3.yaml selector_default)
        selector = tcfg["selector_default"] if task == 3 else sc["target"]
        if mode in ("agent_assist", "agent_full") and llm is None:
            from g1agent.llm.client import LLMClient
            llm = LLMClient()
        err = None
        try:
            if mode == "rule":
                from g1agent.orchestrators.rule import Task1, Task2, Task3
                RULE = {1: Task1, 2: Task2, 3: Task3}
                RULE[task](h, tcfg, selector).run()
            elif mode == "agent_assist":
                from g1agent.orchestrators.agent_assist import run_agent_assist
                run_agent_assist(h, tcfg, selector, llm, log)
            elif mode == "agent_full":
                from g1agent.orchestrators.agent_full import run_agent_full
                run_agent_full(h, tcfg, selector, llm, log)
            else:
                raise ValueError(mode)
        except Exception as e:            # an orchestrator crash is an episode failure, recorded with its traceback
            import traceback
            from g1agent.llm.client import LLMTruncated
            err = f"{type(e).__name__}: {e}"
            rec = {"error": err, "traceback": traceback.format_exc()}
            if isinstance(e, LLMTruncated):
                rec["llm_attempts"] = e.attempts
            log.write("orchestrator_error.txt", rec)
        if video:
            ev.record_stop()
        ev.snapshot(os.path.join(ep_dir, "final.png"))
        gt, actual = ev.get_gt(), ev.get_actual_state()
        stages, scan_log = _eval_rpc(ev, "get_stage_states"), _eval_rpc(ev, "get_scan_log")
        stage_log = _eval_rpc(ev, "get_stage_log")          # every stage occurrence (face_start / face_end per face, ...)
        summ = tb.ws.summary()
        ws = {"root_target": tb.ws.root_target, "wrist_target": tb.ws.wrist_target, "wrist_target_refined": tb.ws.wrist_target_refined}
        if task == 3:
            ws.update({k: summ[k] for k in ("scan_plan", "scanned", "current_barcode_id")}, **stage_ws)
        metrics = evaluate(task, gt, actual, ws, stages=stages, scan_log=scan_log, **subset_eval_args(sc, task))
        res = {"scenario": sc["id"], "task": task, "scenario_task": sc["task"], "mode": mode,
               "robot_variant": tb.info["robot_variant"], "head_camera_model": tb.info["head_camera_model"],
               "target_boxes": sc.get("target_boxes"), "scan_targets": sc.get("scan_targets"),
               "robot_init": sc.get("robot_init"), "harness_finish": h.finished,
               "orchestrator_error": err, "tool_calls": h.n_calls, "explore_steps": tb.ws.explore_steps,
               "wall_s": round(time.time() - t0, 1), "llm": dict(llm.stats) if llm else None, "world_state": summ,
               "metrics": metrics, "stages": stages, "stage_log": stage_log, "scan_log": scan_log, "factor": sc.get("factor"), "level": sc["level"]}
        log.write("result.json", R3(res))
        build_timeline(ep_dir)
        if llm:
            for k in llm.stats:
                llm.stats[k] = 0 if isinstance(llm.stats[k], int) else 0.0
        return R3(res)

    return run


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--mode", default="rule", choices=["rule", "agent_assist", "agent_full"])
    ap.add_argument("--run_dir", default=None)
    ap.add_argument("--no_video", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--task", type=int, default=None, choices=TASKS, help="override the scenario's task (e.g. 3)")
    a = ap.parse_args()
    tag = f"_task{a.task}" if a.task is not None else ""
    sc = load_scenario_arg(a.scenario)
    rd = a.run_dir or os.path.join(ROOT, "runs", time.strftime("%Y%m%d_%H%M%S") + f"_{sc['id']}_{a.mode}{tag}")
    run = make_episode_runner(a.device)
    r = run(sc, a.mode, os.path.join(rd, sc["id"]), video=not a.no_video, task=a.task)
    print(json.dumps({k: r[k] for k in ("scenario", "mode", "robot_variant", "harness_finish", "orchestrator_error", "tool_calls",
                                        "wall_s", "metrics")}, indent=1))
