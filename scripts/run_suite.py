"""Run a test suite with one orchestration mode; resumable.

python scripts/run_suite.py --suite t1_L1 --mode rule                 # all t1_L1_<factor> suites
python scripts/run_suite.py --suite t2_L0 t2_L1 --mode agent_full --run_name nightly_agent_full
python scripts/run_suite.py --suite t1_L0 --mode rule --task 3        # t1 scenarios as the merged find + scan task
Writes runs/<run_name>/results.csv (one row per scenario) and runs/<run_name>/<scenario>/ (logs, video, result.json).
"""
import argparse, csv, json, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from g1agent.common import config  # noqa: E402
sys.path.insert(0, config.load("system")["sim_server"]["sim_root"])
from g1_barcode_sim.testsuite import list_suites, load_suite  # noqa: E402

COLS = ["scenario", "suite", "level", "task", "mode", "factor_name", "factor_value", "harness_success", "eval_success",
        "d_err_m", "yaw_err_deg", "perception_err_pos_m", "perception_err_yaw_deg", "execution_err_pos_m", "execution_err_yaw_deg",
        "wrist_dist_to_plane_m", "wrist_axis_offset_m", "wrist_angle_err_deg", "head_perception_err_m", "wrist_perception_err_m",
        "execution_err_m",
        # task 3 (merged find + scan all): task-1 phase at the task1_end stage, then the scan phase
        "t1_success", "t1_d_err_m", "t1_yaw_err_deg", "t1_face_in_view_gt",
        "n_gt_barcodes", "n_planned", "n_scan_calls", "n_no_read", "n_scanned_distinct", "scan_recall", "n_wrong_contents",
        "n_duplicate_reads", "n_read_not_aimed", "order_ok", "order_ok_scanned", "tip_err_mean_m", "tip_err_max_m",
        "n_placement_ok", "ledger_matches_log",
        "tool_calls", "explore_steps", "llm_calls", "llm_prompt_tokens", "llm_completion_tokens",
        "llm_length_truncations", "wall_s", "failure"]


def row(sc, res):
    m, f = res["metrics"], sc.get("factor") or {}
    fv = f.get("value") if isinstance(f, dict) else None
    hf = res.get("harness_finish") or {}
    llm = res.get("llm") or {}
    r = {"scenario": sc["id"], "suite": sc["suite"], "level": sc["level"], "task": res["task"], "mode": res["mode"],
         "factor_name": f.get("name") if isinstance(f, dict) else None, "factor_value": json.dumps(fv) if isinstance(fv, dict) else fv,
         "harness_success": hf.get("success"), "eval_success": m.get("success"), "tool_calls": res["tool_calls"],
         "explore_steps": res["explore_steps"], "llm_calls": llm.get("calls"), "llm_prompt_tokens": llm.get("prompt_tokens"),
         "llm_completion_tokens": llm.get("completion_tokens"), "llm_length_truncations": llm.get("length_truncations"),
         "wall_s": res["wall_s"], "failure": None if m.get("success") else (res.get("orchestrator_error") or hf.get("summary") or m.get("failure"))}
    for k in COLS:
        if k in m and k not in r:     # metric columns only: e.g. metrics["failure"] must not replace the episode-level
            r[k] = m[k]               # failure above (orchestrator error / harness summary come first)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", nargs="+", required=True, help="suite names; t1_L1 / t2_L1 expand to all their factors")
    ap.add_argument("--mode", required=True, choices=["rule", "agent_assist", "agent_full"])
    ap.add_argument("--run_name", default=None)
    ap.add_argument("--video", action="store_true", help="record an mp4 per episode (screenshots are always saved)")
    ap.add_argument("--device", default="cpu", help="detector device")
    ap.add_argument("--scenarios", nargs="+", default=None, help="run only these scenario ids of the given suites")
    ap.add_argument("--task", type=int, default=None, choices=[1, 2, 3],
                    help="override every scenario's task (e.g. --task 3 runs t1 scenarios as the merged find + scan task)")
    a = ap.parse_args()
    known = [m["suite"] for m in list_suites()]
    names = [n for s in a.suite for n in (known if s == "all" else [k for k in known if k == s or k.startswith(s + "_")])]
    if not names:
        raise SystemExit(f"no generated suites match {a.suite}")
    scs = [sc for n in names for sc in load_suite(n)]
    if a.scenarios is not None:
        unknown = set(a.scenarios) - {sc["id"] for sc in scs}
        if unknown:
            raise SystemExit(f"--scenarios not in the given suites: {sorted(unknown)}")
        scs = [sc for sc in scs if sc["id"] in set(a.scenarios)]
    run_name = a.run_name or time.strftime("%Y%m%d_%H%M") + f"_{'+'.join(a.suite)}_{a.mode}" + (f"_task{a.task}" if a.task else "")
    out = os.path.join(ROOT, "runs", run_name)
    os.makedirs(out, exist_ok=True)
    json.dump({"suites": names, "mode": a.mode, "task_override": a.task, "n": len(scs), "argv": sys.argv[1:],
               "sim_instance": os.environ.get("SIM_INSTANCE", "0")}, open(os.path.join(out, "run.json"), "w"), indent=1)
    SY = os.environ.get("PROGRESS_CLIENT_DIR")      # optional progress reporter (a dir with switchyard_progress.py)
    prog = None
    if SY:
        sys.path.insert(0, SY)
        from switchyard_progress import Run
        prog = Run(f"hbs-{run_name}", label=f"HumanScan suite {'+'.join(a.suite)} · {a.mode}", gpus=[])
    from run_task import make_episode_runner
    import torch
    torch.set_num_threads(int(os.environ.get("DETECTOR_THREADS", "16")))
    run = make_episode_runner(a.device)
    csv_path = os.path.join(out, "results.csv")
    rows = []
    for i, sc in enumerate(scs):
        ep = os.path.join(out, sc["id"])
        rp = os.path.join(ep, "result.json")
        if os.path.exists(rp):
            res = json.load(open(rp))
            want = sc["task"] if a.task is None else a.task
            if res["task"] != want:
                raise SystemExit(f"{rp} was run as task {res['task']}, this run asks for task {want}: use another --run_name")
        else:
            if prog:
                prog.progress(i, len(scs), unit="episodes", status=f"{sc['id']}")
            res = run(sc, a.mode, ep, video=a.video, task=a.task)
        rows.append(row(sc, res))
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLS)
            w.writeheader(); w.writerows(rows)
        print(f"[{i + 1}/{len(scs)}] {sc['id']} {a.mode} eval_success={rows[-1]['eval_success']} "
              f"harness={rows[-1]['harness_success']} {rows[-1]['wall_s']}s fail={rows[-1]['failure']}", flush=True)
    if prog:
        prog.done()
    print("RUN_DONE", out, flush=True)


if __name__ == "__main__":
    main()
