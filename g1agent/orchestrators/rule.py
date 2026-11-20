"""Rule orchestrators (fixed state machines) for task 1, task 2 and task 3 (task 1
over all boxes, then every barcode in the fixed scan order). Every action goes through Harness.call, exactly like the LLM
orchestrators. `explore_policy` decides exploration moves (rule: the fixed sequence; agent_assist plugs in the LLM
there) and `recovery_policy` decides what to do after a tool failure (rule: retry once, then give up).

Stages return results; only run() calls finish, always with the episode's task id (h.task), so a stage can be reused
inside task 3: Task1.stage() (find + approach), Task2.place_and_scan(barcode_id) (wrist placement + scan)."""
from __future__ import annotations

import math

import numpy as np


class RuleExplore:
    def __init__(self, seq):
        self.seq, self.k = list(seq), 0

    def __call__(self, h, situation: str):
        if self.k >= len(self.seq):
            return None
        d, dist = self.seq[self.k]
        self.k += 1
        return {"direction": d, "distance": dist}


def rule_recovery(h, tool, result, attempt):
    return "retry" if attempt == 0 else "give_up"


def recover(h, policy, tool, args, r):
    """Recovery loop after a failed call: ask the policy (retry / explore / give up) until the tool succeeds. Stops as
    soon as the budget is exhausted or the harness rejects the call (a runaway explore/retry loop made 1195 approach
    calls after the budget was gone in m5_assist_gear_4/t1_L1_box_shape_0007)."""
    attempt = 0
    while r["status"] != "ok":
        if r.get("budget_exhausted") or r.get("rejected") or h.over_budget():
            return r
        act = policy(h, tool, r, attempt)
        if isinstance(act, dict) and act.get("explore"):
            e = h.call("explore", act["explore"])
            if e.get("budget_exhausted") or h.over_budget():
                return r
        elif act != "retry":
            return r
        attempt += 1
        r = h.call(tool, args)
    return r


class Task1:
    """fast_end (task 3): the stage ends right after the walk; an arrival check that fails after the fine moves is not
    retried (the scan phase plans its own stances), any other approach failure goes through the recovery policy."""

    def __init__(self, h, cfg, selector, explore_policy=None, recovery_policy=None, fast_end=False):
        self.h, self.cfg, self.selector, self.fast_end = h, cfg, selector, fast_end
        self.explore = explore_policy or RuleExplore(cfg["explore_sequence"])
        self.recover = recovery_policy or rule_recovery

    def _explore(self, why):
        mv = self.explore(self.h, why)
        if mv is None:
            return False
        r = self.h.call("explore", mv)
        return r["status"] != "fail" or not r.get("budget_exhausted")

    def stage(self) -> dict:
        """Find the target (or all boxes) and walk to the observation pose. Returns {"success", "summary", "walked",
        "result"}; walked = the approach was run to its end (its arrival check decided)."""
        h = self.h
        while h.finished is None:
            stop = h.over_budget()
            if stop:
                return {"success": False, "walked": False, "summary": stop}
            r = h.call("detect_boxes", {})
            if r["status"] != "ok":
                if not self._explore("no box detected"):
                    return {"success": False, "walked": False, "summary": "no box found after exploring"}
                continue
            r = h.call("select_target_box", {"selector": self.selector})
            if r["status"] != "ok":
                if not self._explore(f"target not identifiable: {r['reason']}"):
                    return {"success": False, "walked": False, "summary": "target box not identified"}
                continue
            r = h.call("fit_front_face", {})
            if r["status"] == "fail":                 # "all": partial = some faces fitted, the pose uses those
                # far away: depth invalid / too few inliers -> advance 1 m toward the detection
                if not self._advance_toward_target():
                    return {"success": False, "walked": False, "summary": f"front face not fitted: {r['reason']}"}
                continue
            h.call("compute_obs_pose", {})
            args = {"target": "root_target"}
            r = h.call("approach", args)
            if r["status"] != "ok" and not (self.fast_end and r["data"].get("walk_done")):
                r = recover(h, self.recover, "approach", args, r)
            ok = r["status"] == "ok"
            return {"success": ok, "walked": ok or bool(r["data"].get("walk_done")), "result": r,
                    "summary": "arrived at the observation pose" if ok else r["reason"]}
        return {"success": False, "walked": False, "summary": "episode already finished"}

    def run(self):
        s = self.stage()
        if self.h.finished is not None:
            return {"status": "ok"}
        return self.h.call("finish", {"task": self.h.task, "success": s["success"], "summary": s["summary"]})

    def _advance_toward_target(self):
        tb = self.h.tb
        ids = list(tb.ws.boxes) if tb.ws.target_box_id == "all" else [tb.ws.target_box_id]
        bbs = [tb.ws.boxes[i].bbox for i in ids if i in tb.ws.boxes]
        if not bbs or tb.head is None:
            return False
        from ..perception.geometry import pixel_ray
        bb = np.array(bbs, float)
        u = (bb[:, 0].min() + bb[:, 2].max()) / 2                   # centre of the (union of the) detection box(es)
        v = (bb[:, 1].min() + bb[:, 3].max()) / 2
        d = tb.head["T_root_cam"][:3, :3] @ pixel_ray(tb.head["K"], u, v)
        bearing = math.atan2(d[1], d[0])
        adv = self.cfg["too_few_inliers_advance_m"]
        r = self.h.call("explore", {"direction": "turn_left" if bearing > 0 else "turn_right", "distance": min(15.0, abs(math.degrees(bearing)))}) \
            if abs(math.degrees(bearing)) > 3 else {"status": "ok"}
        moved = 0.0
        while moved < adv - 1e-6:
            step = min(0.5, adv - moved)
            r = self.h.call("explore", {"direction": "forward", "distance": step})
            if r["status"] == "fail":
                return False
            moved += step
        return True

    def _with_recovery(self, tool, args):
        return recover(self.h, self.recover, tool, args, self.h.call(tool, args))


class Task2:
    def __init__(self, h, cfg, selector, explore_policy=None, recovery_policy=None):
        self.h, self.cfg, self.selector = h, cfg, selector
        self.recover = recovery_policy or rule_recovery
        self.explore = explore_policy

    def _perceive_target(self):
        h = self.h
        for tool, args in (("detect_boxes", {}), ("select_target_box", {"selector": self.selector}), ("fit_front_face", {})):
            r = h.call(tool, args)
            if r["status"] != "ok":
                return r
        return {"status": "ok"}

    def _locate(self) -> dict:
        """Steps 1-3: target face, barcode in the head image, barcode pose. {"ok", "summary"}"""
        h, c = self.h, self.cfg
        r = self._perceive_target()
        if r["status"] != "ok" and self.explore is not None:
            mv = self.explore(h, f"target face not perceived: {r['reason']}")
            if mv:
                h.call("explore", mv)
                r = self._perceive_target()
        if r["status"] != "ok":
            return {"ok": False, "summary": f"target face: {r['reason']}"}
        r = h.call("detect_barcode_head", {})
        if r["status"] != "ok":                    # advance 0.1 m and retry once
            h.call("explore", {"direction": "forward", "distance": c["head_retry_advance_m"]})
            r = self._perceive_target()
            r = h.call("detect_barcode_head", {}) if r["status"] == "ok" else r
            if r["status"] != "ok":
                return {"ok": False, "summary": "no barcode on the target face"}
        r = h.call("estimate_barcode_pose", {})
        if r["status"] != "ok":
            return {"ok": False, "summary": f"estimate_barcode_pose: {r['reason']}"}
        return {"ok": True, "summary": "barcode located"}

    def _stand_up(self):
        """Arm home, then stand up (the arm must not sweep through the legs while they extend)."""
        h = self.h
        if not h.tb.arm_at_home:
            h.call("move_arm", {"pose": "home"})
        if not h.tb._standing():
            h.call("set_base_height", {})

    def place_and_scan(self, barcode_id: str | None = None, home_after: bool = False, _replanned: bool = False,
                       before_stand_up=None, _searched_again: bool = False) -> dict:
        """Steps 4-9 for one barcode (default: the current one), then scan (one retry via refine_with_wrist +
        move_arm final on no read) and, with home_after, move the arm home and stand up. The arm goes home and the
        robot stands before any base move; after walking to a squat stance it squats (plan_stance base_height).
        Wall layout: the search of a box (search_cell) is repeated once from a new stance after a partial search or
        a look pose not reachable from the look stance (_searched_again), and a label found whose scan pose is not
        reachable from the look stance gets its scan stance planned as any label entry; the stance re-plan of the label (_replanned: an IK failure of the pre / final
        pose) is separate from both, once.
        Returns {"placed", "scanned", "summary"}."""
        h, c = self.h, self.cfg
        bid = {"barcode_id": barcode_id} if barcode_id else {}

        def end(placed, scanned, summary):
            if home_after:
                if not h.tb.arm_at_home:
                    h.call("move_arm", {"pose": "home"})
                if before_stand_up is not None:
                    before_stand_up()                     # e.g. task 3: look for new labels from here (low view)
                self._stand_up()
            return {"placed": placed, "scanned": scanned, "summary": summary}

        for tool, args in (("compute_wrist_target", dict(bid)), ("plan_stance", {})):
            r = h.call(tool, args)
            if r["status"] != "ok":
                return end(False, False, f"{tool}: {r['reason']}")
        if not h.tb.arm_at_home:                   # walking with the arm raised drifts the heading and risks collisions
            r = h.call("move_arm", {"pose": "home"})
            if r["status"] != "ok":
                return end(False, False, f"arm home before walking: {r['reason']}")
        if not h.tb._standing() and (h.tb.ws.stance_target or {}).get("in_place") is None:
            # walking while squatted degrades (toolbox _walk_guard); a strip-mode stance planned where the robot stands
            # has no walk (approach changes the height itself)
            r = h.call("set_base_height", {})
            if r["status"] != "ok":
                return end(False, False, f"stand up before walking: {r['reason']}")
        r = h.call("approach", {"target": "stance_target"})
        if r["status"] == "fail":
            return end(False, False, f"stance: {r['reason']}")
        bh = (h.tb.ws.stance_target or {}).get("base_height")
        if bh is not None and abs(bh - h.tb.base_height_cmd) > 1e-6:
            r = h.call("set_base_height", {"height": bh})
            if r["status"] != "ok":
                return end(False, False, f"squat to {bh}: {r['reason']}")
        cur = h.tb.ws.barcodes.get(h.tb.ws.current_barcode_id)
        searched = cur is not None and cur.kind == "cell"
        if searched:
            # wall layout, a box whose label is not found yet: the hand-camera search at the look stance; a found label
            # goes straight to the final pose from this stance (the search replaces the pre-pose and the first refine)
            r = h.call("search_cell", dict(bid))
            if r["status"] == "fail" and "ik" not in (r.get("data") or {}):
                return end(False, False, f"search_cell: {r['reason']}")
            if r["status"] != "ok":
                # partial, or the look pose not reachable from this stance (fail with data "ik"; the tool recorded the
                # stance as failed): as the tool says, a new stance and the search once more
                if not _searched_again:
                    self._stand_up()
                    return self.place_and_scan(barcode_id, home_after, _replanned, before_stand_up, _searched_again=True)
                return end(False, False, f"search_cell: {r['reason']}")
            if r["data"].get("barcode_status") == "no_label":
                return end(False, False, "no label in the box")
            if not r["data"]["scan_from_this_stance"]:
                # the look stance serves the face search; the label found is out of reach from it: the entry is a label
                # now, so the next pass is the label path (its scan stance with the pre-pose, refine, final)
                self._stand_up()
                return self.place_and_scan(barcode_id, home_after, _replanned, before_stand_up, _searched_again)
        if not searched:
            r = h.call("move_arm", {"pose": "pre"})
            if r["status"] == "fail":                 # partial: reached with a residual, the wrist camera still looks
                if not _replanned and r["reason"].startswith("IK"):
                    # the pose became unreachable after the real walk / squat: stand up, plan again from here once
                    self._stand_up()
                    return self.place_and_scan(barcode_id, home_after, True, before_stand_up, _searched_again)
                return end(False, False, f"pre-pose: {r['reason']}")
            r = h.call("refine_with_wrist", dict(bid))
            if r["status"] == "ok":
                s = h.call("stereo_check_wrist", dict(bid))
                if _stereo_disagrees(s):
                    h.call("move_arm", {"pose": "pre"})
                    r2 = h.call("refine_with_wrist", dict(bid))
                    s2 = h.call("stereo_check_wrist", dict(bid)) if r2["status"] == "ok" else s
                    if _stereo_disagrees(s2):
                        h.tb.adopt_stereo(barcode_id, s2["data"]["triangulated_odom"])   # trust the motion stereo
        for k in range(c["place"]["max_iters"] + 1):
            r = h.call("move_arm", {"pose": "final"})
            if r["status"] == "fail":                 # partial = reached with a residual: refine / check / scan decide
                bc = h.tb.ws.barcodes.get(h.tb.ws.current_barcode_id)
                looked = (h.tb.ws.stance_target or {}).get("purpose") == "look"
                if not _replanned and r["reason"].startswith("IK") and bc is not None \
                        and (bc.width_source == "wrist" or looked):
                    # the stance was planned on the head width (or, wall layout, for the hand-camera view, without the
                    # label); the measured stand-off is out of reach from it:
                    # stand up and plan the stance once more on the measured width
                    self._stand_up()
                    return self.place_and_scan(barcode_id, home_after, True, before_stand_up, _searched_again)
                return end(False, False, f"final pose: {r['reason']}")
            if k == c["place"]["max_iters"]:
                break
            before = h.tb.ws.wrist_target_refined
            if before is None and searched:
                # wall layout: the search replaced the pre-pose refine, so the final pose just reached is the unrefined
                # label target; a refine that moves it goes to the final pose again (final -> refine -> final)
                before = h.tb.ws.wrist_target
            rr = h.call("refine_with_wrist", dict(bid))
            if rr["status"] != "ok" or before is None:
                break
            moved = float(np.linalg.norm(np.asarray(h.tb.ws.wrist_target_refined["pos"]) - np.asarray(before["pos"])))
            if moved <= c["place"]["tol_m"]:
                break
        placement = h.call("check_wrist_placement", {})
        placed = placement["status"] == "ok"
        sc = h.call("scan", {})                       # the no-read retry happens inside the scan tool
        scanned = sc["status"] == "ok"
        summary = ("wrist placed at the stand-off in front of the barcode" if placed else placement["reason"]) + \
                  (f"; scanned {sc['data'].get('content')!r}" if scanned else f"; scan: {sc['reason']}")
        return end(placed, scanned, summary)

    def run(self):
        h = self.h
        h.tb.remember_home()
        loc = self._locate()
        if not loc["ok"]:
            return h.call("finish", {"task": h.task, "success": False, "summary": loc["summary"]})
        p = self.place_and_scan(None, home_after=False)     # task 2 is evaluated with the arm at the barcode
        return h.call("finish", {"task": h.task, "success": p["placed"], "summary": p["summary"]})


def _stereo_disagrees(s) -> bool:
    """stereo_check_wrist measured a disagreement (partial with the comparison data), as opposed to 'not visible'."""
    return s["status"] == "partial" and s["data"].get("diff_to_refined_m") is not None


class Task3:
    """Task 1 over all boxes (fast end), plan_scans, then every planned barcode in the plan's order (next_barcode decides,
    never this code) with Task2.place_and_scan; finish(task=3) with the success computed by the Harness from the ledger.
    The scenario's left/right selector is not used: task 3 observes and scans every box. With a target spec in robot
    terms (Toolbox.scan_targets) only the targeted labels count, and phase 1 stays near the start pose (_stay_stage;
    phase1_view "start_pose", the default) or walks to the observation pose compute_obs_pose plans from the spec
    (phase1_view "obs_pose": the task-1 stage, Toolbox.set_phase1_view). With a wall layout (Toolbox.set_wall_layout)
    every planned box is searched with search_cell at its look stance (place_and_scan), no plan update in phase 2, and
    a box recorded no_label counts as handled; worked in strips (order "strips"), next_barcode's strip_done is followed
    by go_to_next_strip."""

    def __init__(self, h, cfg, selector, explore_policy=None, recovery_policy=None):
        self.h, self.cfg = h, cfg
        self.t1 = Task1(h, cfg, "all", explore_policy, recovery_policy, fast_end=True)
        self.t2 = Task2(h, cfg, "all", explore_policy, recovery_policy)

    def _stay_stage(self) -> dict:
        """Phase 1 of a task with a target spec (labels of some boxes of the wall, given in robot terms): the robot works
        from near its start pose (user decision 2026-11-19: do not walk back to observe the whole wall); plan_scans
        looks with every standing head view from here. Detect + select every box only."""
        h = self.h
        r = h.call("detect_boxes", {})
        if r["status"] != "ok":
            return {"walked": False, "summary": r["reason"]}
        r = h.call("select_target_box", {"selector": "all"})
        return {"walked": r["status"] == "ok", "summary": r["reason"]}

    def run(self):
        h = self.h
        h.tb.remember_home()
        stay = h.tb.scan_targets is not None and h.tb.phase1_view == "start_pose"
        while True:                                                   # phase 1 (task 1) until a scan plan exists
            s = self._stay_stage() if stay else self.t1.stage()
            if h.finished is not None:
                return {"status": "ok"}
            if not s["walked"]:
                if stay and not h.over_budget() and self.t1._explore(f"boxes not found from the start pose: {s['summary']}"):
                    continue
                return h.call("finish", {"task": h.task, "success": False, "summary": f"task-1 phase: {s['summary']}"})
            r = h.call("plan_scans", {})
            if r["status"] == "ok":
                break
            # no label localised (or no face): explore per the budget and run phase 1 again
            # rejected = plan_scans hit its retry limit (fail streaks reset only on next_barcode): redoing phase 1
            # cannot make it run again
            if r.get("budget_exhausted") or r.get("rejected") or not self.t1._explore(f"no barcode label localised: {r['reason']}"):
                return h.call("finish", {"task": h.task, "success": False, "summary": f"plan_scans: {r['reason']}"})
        results = []
        while h.finished is None and not h.over_budget():            # phase 2: the plan's order, one barcode at a time
            nb = h.call("next_barcode", {})
            if nb["status"] != "ok" or nb["data"].get("done"):
                break
            if nb["data"].get("face_done"):
                # multi-face task: walk around to the next side and plan it; a side that cannot be reached ends the
                # tour (its labels stay unplanned); a side without labels gives face_done again -> the next side
                g = h.call("go_to_next_face", {})
                if g["status"] != "ok":
                    break
                h.call("plan_scans", {})
                continue
            if nb["data"].get("strip_done"):
                # wall layout worked in strips: the current strip is finished, step to the next one holding requested
                # boxes (the tool refuses otherwise); a strip that cannot be reached ends the task
                g = h.call("go_to_next_strip", {})
                if g["status"] != "ok":
                    break
                continue
            # a wall layout fixes the cells: no plan update before standing up
            results.append(self.t2.place_and_scan(nb["data"]["barcode_id"], home_after=True,
                                                  before_stand_up=None if h.tb.wall_layout is not None
                                                  else lambda: h.call("plan_scans", {})))
        if h.finished is not None:
            return {"status": "ok"}
        ws = h.tb.ws
        plan = [b for b in ws.scan_plan if ws.targeted(b)]           # not_targeted labels are not part of the task
        n_ok = sum(ws.barcodes[b].status == "scanned" for b in plan)
        n_unr = sum(ws.barcodes[b].status == "unreadable" for b in plan)
        n_nol = sum(ws.barcodes[b].status == "no_label" for b in plan)     # wall layout: a box seen whole without label
        missing, warn = ws.missing_cells(), ws.grid_warnings()
        return h.call("finish", {"task": h.task,
                                 "success": n_ok > 0 and n_ok + n_unr + n_nol == len(plan) and not missing and not warn,
                                 "summary": f"scanned {n_ok} of {len(plan)} planned barcodes"
                                            + (f", {n_unr} label(s) without a readable code" if n_unr else "")
                                            + (f", {n_nol} box(es) without a label" if n_nol else "")
                                            + (f", requested cells not found: {missing}" if missing else "")
                                            + (f", target count not verified: {warn}" if warn else "")
                                            + ("" if not h.over_budget() else f" ({h.over_budget()})")})


TASKS = {1: Task1, 2: Task2, 3: Task3}


def task_class(task: int):
    """The orchestrator class of a task id; an unknown id is an error (no silent fallback to another task)."""
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}; known: {sorted(TASKS)}")
    return TASKS[task]
