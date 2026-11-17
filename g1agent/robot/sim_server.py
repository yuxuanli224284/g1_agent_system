"""Simulation server: G1BarcodeEnv behind ZeroMQ. Runs in unitree_sim_env.

Two REP sockets on one thread (Isaac Sim must be driven from its main thread):
- robot port : RobotInterface methods only (the harness side). No ground truth: scan() is returned without its
               evaluation-only `gt` field.
- eval port  : episode control and evaluation, answers only requests carrying the token from $G1_EVAL_TOKEN:
               reset (scenario), ground truth, actual state, GT / actual markers, third-person snapshots, the composite
               episode video (record_start / record_stop / annotate), stage states and the robot-port scan log, and the full stage log (every occurrence, get_stage_log).

Composite video: two fixed recording cameras (MAIN behind-left of
the robot start, SIDE at the robot's left of the boxes) posed once per episode from the scenario layout (and, on a
multi-face stack, re-posed at every face_start stage on that face of the tour: compositor.face_framing), plus the head
and scanner cameras, read straight from their render buffers in a step hook every round(1 / (fps * policy_dt)) policy
steps (no extra renders, no USD marker toggling, no get_obs); markers and detections are drawn as 2-D overlays.
The video runs in simulated time; LLM replies, scans and head detections add frozen-state "hold" frames.

python -m g1agent.robot.sim_server            (see scripts/start_sim_server.sh)
"""
from __future__ import annotations

import math
import os
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from g1agent.common import config  # noqa: E402
from g1agent.common.serialization import pack, unpack  # noqa: E402

SYS = config.load("system")["sim_server"]
VIZ = config.load("viz")
REC = VIZ["recording"]
sys.path.insert(0, SYS["sim_root"])


def main():
    import numpy as np
    import zmq
    from g1_barcode_sim.env import G1BarcodeEnv
    token = os.environ.get(SYS["eval_token_env"])
    if not token:
        raise SystemExit(f"set ${SYS['eval_token_env']} (evaluation token) before starting the server")
    overrides = {"robot": {"mode_policies": SYS["mode_policies"]}}
    if REC["cameras_enabled"]:
        overrides["camera"] = {"recording": {"enabled": True}}
    env = G1BarcodeEnv("scenario", headless=True, seed=0, config_overrides=overrides)
    import cv2
    import omni.usd
    from g1_barcode_sim.camera import inv_T, look_at_ros
    from g1agent.viz.compositor import Compositor
    from g1agent.viz.markers_usd import MarkerLayer
    stage = omni.usd.get_context().get_stage()
    markers = MarkerLayer(stage, VIZ["colors"], VIZ["sizes"])
    state = {"gt_markers": False}

    def T_world_odom():
        fs = env.frame_state()
        return fs["T_world_root"] @ inv_T(fs["T_world_root_odom"])

    def third_person_view(hfov=None):
        tp = VIZ["third_person"]
        T = env._T_world_root()
        yaw = math.atan2(T[1, 0], T[0, 0]); c, s = math.cos(yaw), math.sin(yaw)
        w = lambda p: np.array([T[0, 3] + c * p[0] - s * p[1], T[1, 3] + s * p[0] + c * p[1], p[2]])
        _refresh_gt_markers()
        markers.show(True)
        img = env.render_aux(look_at_ros(w(tp["eye_offset"]), w(tp["look_at"])), hfov_deg=hfov or tp["hfov_deg"])["rgb"]
        markers.show(False)
        env.env.sim.render()            # re-render without markers so no robot-camera buffer keeps them
        return img

    def actual_state():
        """GT state from the cheap accessor (same values as via get_obs, without touching the depth-noise stream)."""
        fs = env.frame_state()
        Twr = fs["T_world_root"]
        return {"T_world_root": Twr, "T_world_wrist_point": fs["T_world_tip"], "T_world_odom_now": Twr @ inv_T(fs["T_world_root_odom"]),
                "T_world_headcam": fs["T_world_headcam"], "head_K": fs["head_K"], "head_size": fs["head_size"],
                "scenario_id": env.scenario["id"] if env.scenario else None, "t_sim": fs["t_sim"]}

    def _refresh_gt_markers():
        if not state["gt_markers"] or env.scenario is None:
            markers.clear("eval")
            return
        gt = env.get_ground_truth()
        ob, wt = gt["task1_gt_obs_pose"], gt["task2_gt_wrist_target"]
        a = actual_state()
        Tr, Tt = a["T_world_root"], a["T_world_wrist_point"]
        # task 3 over every box ("all") has no single GT wrist target (sim get_ground_truth)
        markers.set([
            {"id": "gt_root", "color": "green", "kind": "root", "pos": ob["pos"], "yaw": ob["yaw"]}]
            + ([{"id": "gt_wrist", "color": "green", "kind": "pose", "pos": wt["pos"], "dir": wt["forward"]}] if wt is not None else [])
            + [{"id": "act_root", "color": "yellow", "kind": "root", "pos": Tr[:3, 3].tolist(), "yaw": math.atan2(Tr[1, 0], Tr[0, 0])},
               {"id": "act_wrist", "color": "yellow", "kind": "pose", "pos": Tt[:3, 3].tolist(), "dir": Tt[:3, 0].tolist()}],
            group="eval")

    ep = Episode(env, markers, Compositor(REC, VIZ["colors"], VIZ["sizes"]), actual_state)
    env.step_hooks.append(ep.on_step)

    # ------------------------------------------------------------------ robot port
    def harness_markers(ms, frame):
        """Harness markers come in the odometry frame (its only fixed frame); drawn at their world position now."""
        T = T_world_odom() if frame == "odom" else env._T_world_root()
        out = []
        for m in ms:
            m = dict(m)
            m["pos"] = (T[:3, :3] @ np.asarray(m["pos"], float) + T[:3, 3]).tolist()
            if "dir" in m:
                m["dir"] = (T[:3, :3] @ np.asarray(m["dir"], float)).tolist()
            if "yaw" in m:
                m["yaw"] = float(m["yaw"]) + math.atan2(T[1, 0], T[0, 0])
            out.append(m)
        return out

    R = {
        "ping": lambda: {"status": "ok", "t": env.get_obs()["timestamp"]},
        "get_robot_info": lambda: robot_info(env),
        "get_head_obs": lambda depth=False: head_obs(env, depth),
        "get_wrist_obs": lambda side="left": wrist_obs(env, side),
        "get_joint_state": lambda: {"names": list(env.robot.data.joint_names), "q": env.robot.data.joint_pos[0].cpu().numpy(),
                                    "timestamp": env.get_obs()["timestamp"]},
        "get_odom": lambda: {"T_odom_root": env.get_obs()["T_world_root_odom"], "timestamp": env.get_obs()["timestamp"]},
        "walk": lambda vx, vy, wz, duration: strip_obs(env.step(vx, vy, wz, duration)),
        "move_root_relative": lambda dx, dy, dyaw, timeout=10.0, pos_tol=None, yaw_tol=None, speed_scale=1.0: env.move_relative(dx, dy, dyaw, timeout, pos_tol, yaw_tol, speed_scale),
        "stand": lambda duration: strip_obs(env.stand_still(duration)),
        "set_body_pose": lambda height=None, torso_rpy=None, duration=1.0, settle=1.0:
            strip_obs(env.set_body_pose(height, torso_rpy, duration, settle)),
        # the robot's own geometry (link collision hull vertices / spheres in the link frames): CAD knowledge, not GT
        "get_robot_geometry": lambda: {"body_names": list(env.collision_model.body_names),
                                       "points": env.collision_model.local_pts, "radius": env.collision_model.local_rad,
                                       "owner": env.collision_model.owner},
        "set_arm_joints": lambda side, q, duration: set_arm_joints(env, side, q, duration),
        "set_gripper": lambda side, opening: env.set_gripper(side, opening),
        "scan": lambda: {k: v for k, v in ep.scan().items() if k != "gt"},
        "draw_markers": lambda markers_list, frame="odom": (ep.note_harness_markers(markers_list),
                                                            markers.set(harness_markers(markers_list, frame), group="harness"),
                                                            {"status": "ok"})[2],
        "clear_markers": lambda: (ep.note_harness_markers([]), markers.clear("harness"), {"status": "ok"})[2],
    }

    # ------------------------------------------------------------------ eval port
    def reset(scenario):
        ep.reset()                      # finalises a recording left open by the previous episode
        markers.clear()
        o = env.reset(scenario=scenario)
        return {"status": "ok", "scenario_id": env.scenario["id"], "task": env.scenario["task"], "timestamp": o["timestamp"]}

    def snapshot(path=None):
        img = third_person_view()
        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        return {"status": "ok", "path": path, "shape": list(img.shape)}

    def invisibility_check():
        """Robot-camera frames rendered at one physical state (no physics step in between):
        A markers in the stage but hidden (the normal state), B markers removed, B2 = B rendered again (renderer noise
        baseline), C markers shown (control: proves the probes are in view of both cameras). Pass: A == B up to the
        baseline, C != B. Also the frames handed out right after a third-person snapshot equal those before it."""
        def cams():
            for _ in range(VIZ["invisibility_render_passes"]):
                env.env.sim.render()
            env.head_cam.update(0.0, force_recompute=True)
            env.wrist_cam.update(0.0, force_recompute=True)
            return (env.head_cam.data.output["rgb"][0].cpu().numpy()[..., :3].copy(),
                    env.wrist_cam.data.output["rgb"][0].cpu().numpy()[..., :3].copy())
        def diff(a, b):
            d = np.abs(a.astype(np.int16) - b.astype(np.int16)).max(axis=-1)
            return {"max": int(d.max()), "p999": float(np.percentile(d, 99.9)),
                    **{f"n_gt{t}": int((d > t).sum()) for t in (10, 30, 60, 100)}}
        o0 = env.get_obs()
        probes = [{"id": "probe_head", "color": "red", "kind": "point", "pos": (o0["T_world_root_gt"] @ o0["T_root_cam"] @ np.array([0, 0, 0.6, 1.0]))[:3].tolist()},
                  {"id": "probe_wrist", "color": "purple", "kind": "point", "pos": (o0["T_world_root_gt"] @ o0["T_root_wrist_cam"] @ np.array([0, 0, 0.15, 1.0]))[:3].tolist()}]
        # consecutive renders at one state, toggling only the markers, so every compared pair has the same render history
        cams()                                                   # warm-up
        seq = []
        for present in (True, False, True, False):
            markers.set(probes, group="probe") if present else markers.clear("probe")
            seq.append(cams())
        same = cams()                                            # no change -> renderer noise baseline
        markers.set(probes, group="probe"); markers.show(True)
        C = cams()                                               # control: markers shown
        markers.show(False); markers.clear("probe")
        B = cams()
        out = {}
        for k, i in (("head", 0), ("wrist", 1)):
            out[f"{k}_hidden_vs_removed"] = [diff(seq[j][i], seq[j + 1][i]) for j in range(3)]
            out[f"{k}_baseline"] = diff(seq[3][i], same[i])
            out[f"{k}_shown_vs_removed"] = diff(C[i], B[i])
        before = env.get_obs(); snapshot(); after = env.get_obs()
        out["head_after_snapshot"] = diff(before["rgb"], after["rgb"])
        out["wrist_after_snapshot"] = diff(before["wrist_rgb"], after["wrist_rgb"])
        return out

    E = {
        "reset": reset,
        "get_gt": lambda: env.get_ground_truth(),
        "get_actual_state": actual_state,
        "set_gt_markers": lambda enabled: (state.__setitem__("gt_markers", bool(enabled)), {"status": "ok"})[1],
        "snapshot": snapshot,
        "record_start": ep.record_start,
        "record_stop": ep.record_stop,
        "annotate": ep.annotate,
        "stage": ep.stage,
        "get_stage_states": lambda: {k: dict(v) for k, v in ep.stages.items()},
        "get_stage_log": lambda: [dict(e) for e in ep.stage_log],
        "get_scan_log": lambda: list(ep.ov.scans),
        "invisibility_check": invisibility_check,
        "shutdown": lambda: {"status": "ok", "shutdown": True},
    }

    ctx = zmq.Context()
    rs, es = ctx.socket(zmq.REP), ctx.socket(zmq.REP)
    inst = int(os.environ.get("SIM_INSTANCE", "0"))
    rport, eport = SYS["robot_port"] + inst * SYS["instance_port_stride"], SYS["eval_port"] + inst * SYS["instance_port_stride"]
    rs.bind(f"tcp://{SYS['host']}:{rport}")
    es.bind(f"tcp://{SYS['host']}:{eport}")
    poller = zmq.Poller()
    poller.register(rs, zmq.POLLIN); poller.register(es, zmq.POLLIN)
    print(f"SIM_SERVER_READY instance={inst} robot={rport} eval={eport}", flush=True)
    running = True
    while running:
        for sock, _ in poller.poll(1000):
            req = unpack(sock.recv())
            table = R if sock is rs else E
            t0 = time.time()
            try:
                if sock is es and req.get("token") != token:
                    rep = {"status": "error", "reason": "eval port: bad or missing token"}
                elif req["method"] not in table:
                    rep = {"status": "error", "reason": f"unknown method {req['method']!r}"}
                else:
                    rep = table[req["method"]](**req.get("args", {}))
                    if sock is es and req["method"] == "shutdown":
                        running = False
            except Exception as e:      # report to the caller; the server keeps serving
                rep = {"status": "error", "reason": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}
                print(rep["traceback"], flush=True)
            if isinstance(rep, dict) and "status" in rep:   # status replies only: data replies (get_gt, stage states) keep their schema
                rep.setdefault("_wall_s", round(time.time() - t0, 3))
            sock.send(pack(rep))
    # env.close() can block here while sockets are open; reply sent -> close the sockets and end the process at once
    # (the same immediate exit G1BarcodeEnv.close() uses, SimulationApp.close(skip_cleanup=True) semantics)
    rs.close(linger=0); es.close(linger=0); ctx.term()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)


def gt_standoff_fn(task: int):
    """GT scanner stand-off d(gt barcode) of the green target points / METRICS, the one the evaluator scores the task
    with: task 2 = the fixed task2.yaml offset_m (evaluator _task2), otherwise the adaptive GT rule (evaluator
    gt_standoff: task3.yaml standoff, GT symbol size, scanner hFOV from the sim camera definition)."""
    from g1agent.eval.evaluator import gt_standoff, scanner_hfov_gt
    from g1agent.eval.evaluator import gt_symbol_size
    if task == 2:
        d = float(config.load("task2")["offset_m"])
        return lambda bc: d
    hfov = scanner_hfov_gt()
    return lambda bc: gt_standoff(gt_symbol_size(bc), hfov)


class Episode:
    """Evaluation-side state of one episode on the sim server: agent overlay events (annotate), stage states, the
    robot-port scan log and the composite recording. Reset with the scenario."""

    def __init__(self, env, markers, comp, actual_state):
        from g1agent.viz import compositor as C
        self.C, self.env, self.markers, self.comp, self.actual_state = C, env, markers, comp, actual_state
        self.rec = None
        self.reset()

    def reset(self):
        if self.rec is not None:
            self.record_stop()
        self.ov = self.C.OverlayState()
        self.stages = {}             # stage name -> its FIRST occurrence (the evaluator reads stages["task1_end"])
        self.stage_log = []          # every stage call in order: {"name", "info", "t_sim", "actual"} (get_stage_log)
        self.harness_t = {}          # harness marker id -> (raw spec as drawn, sim time it last changed)

    # ---------------------------------------------------------------- inputs
    def note_harness_markers(self, ms):
        """Remember when each harness marker last changed (raw odometry-frame spec; the harness republishes all of its
        markers whenever one changes), so a wrist target set before the current barcode was chosen is not used."""
        t = self.env.frame_state()["t_sim"]
        old, self.harness_t = self.harness_t, {}
        for m in ms:
            prev = old.get(m["id"])
            self.harness_t[m["id"]] = prev if prev is not None and prev[0] == m else (dict(m), t)

    def harness_wrist_targets(self):
        """Harness wrist targets in world (viz.yaml colors: purple = refined with the wrist camera, blue = from the head
        camera), most refined first: [{"pos", "t_set"}]."""
        out = []
        for color in ("purple", "blue"):
            for m in self.markers.specs.values():
                if m["group"] == "harness" and m["color"] == color and m["id"] in self.harness_t:
                    out.append({"pos": m["pos"], "t_set": self.harness_t[m["id"]][1]})
        return out

    def scan(self):
        """robot-port scan(): env.scan() logged with its evaluation-only GT and the world tip pose (get_scan_log)."""
        r = self.env.scan()
        fs = self.env.frame_state()
        self.ov.on_scan({"t_sim": fs["t_sim"], "status": r["status"], "content": r.get("content"),
                         "decoded": r.get("decoded", []), "gt": r.get("gt"),
                         "T_world_tip": fs["T_world_tip"].tolist() if "T_world_tip" in fs else None,
                         "pelvis_z": float(fs["T_world_root"][2, 3]),
                         "height_cmd": fs.get("height_cmd")})
        if self.rec is not None:
            self._write("scan", REC["detection_hold_s"])
        return r

    def annotate(self, event):
        hold = self.ov.on_event(event, self.env.frame_state()["t_sim"])
        if hold is not None and self.rec is not None:
            self._write(hold, REC["llm_hold_s"] if hold == "llm" else REC["detection_hold_s"])
        return {"status": "ok"}

    def stage(self, name, info=None):
        """A harness stage (I3). Every call is kept in stage_log; the stages dict keeps the first occurrence of each
        name. face_start (multi-face tour) re-poses the recording cameras on that face (_pose_face_cameras)."""
        a = self.actual_state()
        self.stage_log.append({"name": name, "info": info, "t_sim": a["t_sim"], "actual": a})
        self.stages.setdefault(name, {"t_sim": a["t_sim"], "actual": a, "info": info})
        self.ov.on_stage(name, info, a["t_sim"])
        if name == "face_start" and self.rec is not None:
            self._pose_face_cameras(int(info["face"]))
        return {"status": "ok", "actual": a}

    # ---------------------------------------------------------------- recording
    def record_start(self, path, fps=None, task=None):
        """task: the task the episode is evaluated as (run_task --task override); default the scenario's task. It
        selects the GT scanner stand-off of the green targets / METRICS, as in the evaluator."""
        import imageio_ffmpeg
        C, env = self.C, self.env
        if self.rec is not None:
            self.record_stop()
        if not env.rec_cams:
            raise RuntimeError("recording cameras are off (viz.yaml recording.cameras_enabled)")
        if env.scenario is None:
            raise RuntimeError("record_start needs a scenario episode: reset(scenario) first")
        if not env.scanner_on:
            raise RuntimeError("the composite video needs the scanner camera (configs/scanner.yaml enabled)")
        fps = float(fps or REC["fps"])
        every = max(1, int(round(1.0 / (fps * env.policy_dt))))
        fps_eff = 1.0 / (every * env.policy_dt)            # 1 video second = 1 simulated second
        gt = env.get_ground_truth()                         # static during the episode: cached
        fs = env.frame_state()
        task = int(env.scenario["task"] if task is None else task)
        targets = C.gt_targets(gt, gt_standoff_fn(task))
        Twr = fs["T_world_root"]
        body = (env.collision_model.world_points() - Twr[:3, 3]) @ Twr[:3, :3]      # robot body points, root frame
        cams = self._pose_cams(C.framing(gt, targets, body, float(Twr[2, 3]), REC["framing"]))
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        writer = C.H264Writer(path, *REC["canvas"], fps_eff, imageio_ffmpeg.get_ffmpeg_exe())
        self.rec = {"path": path, "writer": writer, "every": every, "fps": fps_eff, "count": 0, "gt": gt, "task": task,
                    "targets": targets, "cams": cams, "body": body, "root_z": float(Twr[2, 3]),
                    "multi_face": len(C.labelled_tour(gt, targets)) > 1,
                    "frames": open(os.path.splitext(path)[0] + ".frames.jsonl", "w")}
        self._write(None, None)
        print(f"[record_start] {path}: {fps_eff:g} fps (every {every} policy steps), scenario {env.scenario['id']} as task {task}, "
              + ", ".join(f"{k} at {[round(float(v), 2) for v in T[:3, 3]]}" for k, (_, T) in cams.items()), flush=True)
        return {"status": "ok", "path": path, "fps": fps_eff, "every_steps": every, "task": task,
                "cameras": {k: {"K": K, "T_world_cam": T} for k, (K, T) in cams.items()}}

    def _pose_cams(self, framing):
        """Fit and place the fixed recording cameras for {name: (forward, points)}; returns {name: (K, T_world_cam)}."""
        fr, env = REC["framing"], self.env
        cams = {}
        for name, (fwd, pts) in framing.items():
            c = env.cfg["camera"]["recording"]["cameras"][name]
            T, _ = self.C.fit_fixed_camera(pts, fwd, fr["hfov_deg"], c["width"], c["height"], fr["margin_frac"])
            cams[name] = (env.pose_rec_camera(name, T, fr["hfov_deg"]), T)
        env.settle_renders()
        return cams

    def _pose_face_cameras(self, face_idx: int):
        """Multi-face tour: MAIN on the diagonal between this face and the next labelled one, SIDE on this face
        (compositor.face_framing). A single-face scene keeps its record_start cameras."""
        rec = self.rec
        fr = self.C.face_framing(rec["gt"], rec["targets"], rec["body"], rec["root_z"], REC["framing"], face_idx)
        if fr is None:
            return
        rec["cams"] = self._pose_cams(fr)
        print(f"[face_start] face {face_idx}: " + ", ".join(f"{k} at {[round(float(v), 2) for v in T[:3, 3]]}"
                                                           for k, (_, T) in rec["cams"].items()), flush=True)

    def record_stop(self):
        if self.rec is None:
            return {"status": "ok", "path": None}
        rec, self.rec = self.rec, None
        rec["frames"].close()
        n = rec["writer"].close()
        print(f"[record_stop] {rec['path']}: {n} frames", flush=True)
        return {"status": "ok", "path": rec["path"], "frames": n}

    def on_step(self):
        if self.rec is None:
            return
        self.rec["count"] += 1
        if self.rec["count"] % self.rec["every"] == 0:
            self._write(None, None)

    def _write(self, hold, hold_s):
        """One frame of the current state (hold None), or round(hold_s * fps) identical frozen-state frames."""
        n = 1 if hold is None else int(round(hold_s * self.rec["fps"]))
        if n <= 0:
            return
        import json
        C, env, ov, rec = self.C, self.env, self.ov, self.rec
        fs = env.frame_state()
        t = fs["t_sim"]
        gt, targets = rec["gt"], rec["targets"]
        current = ov.current_target(targets, self.harness_wrist_targets())
        met = C.frame_metrics(gt, fs, targets, current, ov.scans, self.stages.get("task1_end"))
        ob = gt["task1_gt_obs_pose"]
        Tr, Tt = fs["T_world_root"], fs["T_world_tip"]
        # multi-face tour: also the GT obs pose of the face being worked on (face 0 = the front = task1_gt_obs_pose)
        fob = [f["gt_obs_pose"] for f in gt.get("faces") or []
               if rec["multi_face"] and ov.face is not None and ov.face["idx"] > 0 and f["face"] == C.FACE_TOUR[ov.face["idx"]]]
        mk = ([{"color": "green", "kind": "root", "pos": o["pos"], "yaw": o["yaw"]} for o in [ob] + fob]
              + [{"color": "green", "kind": "pose", "pos": g["pos"], "dir": -g["normal"]} for g in targets]
              + [m for m in self.markers.specs.values() if m["group"] == "harness"]
              + [{"color": "yellow", "kind": "root", "pos": Tr[:3, 3], "yaw": math.atan2(Tr[1, 0], Tr[0, 0]),
                  "body_z": float(Tr[2, 3]), "label": "root (pelvis)"},
                 {"color": "yellow", "kind": "pose", "pos": Tt[:3, 3], "dir": Tt[:3, 0]}])
        last = ov.last_scan()
        sc = met["scanned"]
        if last is None:
            scan_txt = "scanned: - (no scan yet)"
        elif last["status"] == "ok":
            prev = [s["content"] for s in ov.scans[:-1] if s["status"] == "ok"]
            scan_txt = f"scanned: yes - {last['content']} ({'duplicate' if last['content'] in prev else 'new'})"
        else:
            scan_txt = f"scanned: no ({last['status']})"
        scan_txt += f"   {sc['k']}/{sc['n']}"
        frame = self.comp.compose({
            "t_sim": t, "hold": hold,
            "images": {"main": env.camera_rgb("main"), "side": env.camera_rgb("side"), "ego": env.camera_rgb("head"),
                       "wrist": env.camera_rgb("wrist")},
            "cams": rec["cams"], "markers": mk,
            "head_det": ov.head_detections_at(t), "target_box": ov.target_box,
            "ego_text": f"boxes seen: {'-' if ov.n_boxes is None else ov.n_boxes}; localised target: {ov.target_text()}",
            "wrist_polys": ([d["polygon"] for d in last["decoded"]] if last is not None and last["t_sim"] == t else None),
            "wrist_text": scan_txt,
            "agent": {"stage": ov.stage, "llm": ov.llm, "tool": ov.tool, "scan_text": "扫码 " + scan_txt,
                      "face": ov.face_text()},
            "metrics": met})
        rec["writer"].write(frame, n)
        rec["frames"].write(json.dumps({"frame": rec["writer"].n - n, "n": n, "hold": hold, "t_sim": t,
                                        "t_unix": time.time(), "metrics": met}) + "\n")
        rec["frames"].flush()


def strip_obs(r):
    return {k: v for k, v in r.items() if k != "obs"}


def wrist_obs(env, side):
    if not env.scanner_on or side != env.scanner_side:
        return {"status": "error", "reason": f"no wrist camera on the {side} hand"}
    o = env.get_obs()
    return {"status": "ok", "rgb": o["wrist_rgb"], "depth": o.get("wrist_depth"), "K": o["wrist_K"],
            "T_root_wristcam": o["T_root_wrist_cam"], "timestamp": o["timestamp"]}


def set_arm_joints(env, side, q, duration):
    n = max(1, int(round(float(duration) / env.policy_dt)))
    qc = env.arm.set_joint_targets(side, q, n)
    r = env.stand_still(float(duration))
    jp = env.robot.data.joint_pos[0].cpu().numpy()
    return {"status": r["status"], "reason": r["reason"], "q_cmd": qc, "q_now": jp[env.arm.joint_idx[side]]}


def head_obs(env, depth: bool = False) -> dict:
    """Head camera frame as the robot gives it. D435i (camera_head.yaml model d435i): the colour image and the
    infrared pair (images + stereo calibration); the depth image only when asked for (depth=True): the render depth
    stands in for the camera's own on-chip depth, and is used only by the render-depth baseline (perception.yaml
    head_depth.source: render). Single upstream camera: rgb + its render depth."""
    o = env.get_obs()
    out = {"rgb": o["rgb"], "K": o["K"], "T_root_headcam": o["T_root_cam"], "timestamp": o["timestamp"]}
    if env.ir_cams is None or depth:
        out["depth"] = o["depth"]
    if env.ir_cams is not None:
        st = env.head_stereo_obs()
        out.update({k: st[k] for k in ("ir_left", "ir_right", "K_ir", "K_ir_right", "T_root_ir_left", "T_irleft_irright")})
    return out


def robot_info(env):
    from g1_barcode_sim.arm import ARM_JOINTS
    o = env.get_obs()
    return {"joint_names": list(env.robot.data.joint_names), "arm_joints": ARM_JOINTS,
            "head_K": o["K"], "head_size": list(o["rgb"].shape[:2]),
            # D435i infrared pair (None for the single upstream camera): intrinsics, size, baseline
            "head_stereo": (None if env.ir_cams is None else
                            {"K_ir": env.ir_cams["left"].data.intrinsic_matrices[0].cpu().numpy().astype(float),
                             "size": [env.cfg["camera"]["stereo"]["height"], env.cfg["camera"]["stereo"]["width"]],
                             "baseline_m": env.cfg["camera"]["stereo"]["baseline_m"]}),
            "wrist_side": env.scanner_side if env.scanner_on else None,
            "wrist_K": o.get("wrist_K"), "wrist_size": list(o["wrist_rgb"].shape[:2]) if "wrist_rgb" in o else None,
            "policy_dt": env.policy_dt, "velocity_limits": env.cfg["robot"]["velocity_limits"],
            "arm_soft_limits": {s: env.arm.limits[s] for s in ARM_JOINTS},
            "wrist_point": "scanner tip (exit-window centre) of the scanner held by the left gripper",
            # body (squat / torso) command of the GR00T walker; None for a walker without one
            "body_command": ({"height_default": env.walker.height_default, "height_range": list(env.walker.height_range),
                              "rpy_limit_rad": [float(v) for v in env.walker.rpy_limit],
                              "rpy_default": [float(v) for v in env.walker.rpy_default]}
                             if hasattr(env.walker, "set_body_command") else None)}


if __name__ == "__main__":
    main()
