"""Harness: validation, limits, budgets, logging around every tool call.

Task 3 bookkeeping lives here so that the rule and the LLM orchestrators behave the same:
- plan_scans ok -> the budget becomes base + per_barcode x n_planned (configs/task3.yaml budget.per_barcode; labels
  outside a target spec's cells, status not_targeted, are not counted), and the end of the task-1 phase is signalled
  once (log.stage("task1_end"));
- next_barcode ok -> the per-tool failure streaks are reset (retries are per barcode), barcode_end of an unfinished
  barcode (failed), barcode_start of the next one, scan_done when the plan is exhausted;
- scan resolving the current barcode -> barcode_end;
- search_cell (wall layout only) -> labels it added extend the budget like planned ones, no_label -> barcode_end;
- finish -> the scan ledger and plan statuses are attached and success is computed from the ledger (and the
  stack tour, every requested target cell found and the target count verified, ws.grid_warnings; a box of a wall
  layout recorded no_label is handled, like an unreadable label); finish(success=true)
  is refused while the plan or the tour still has work.
"""
from __future__ import annotations

import json
import time
import traceback

from pydantic import ValidationError

from .registry import TASK_TOOLS, TOOLS, task_tools


class Harness:
    def __init__(self, toolbox, task: int, budget: dict, logger, limits: dict):
        if task not in TASK_TOOLS:
            raise ValueError(f"unknown task {task!r}; known: {sorted(TASK_TOOLS)}")
        self.tb, self.task, self.log, self.limits = toolbox, task, logger, limits
        self.base_budget = dict(budget)             # the config dict is shared between episodes: never mutate it
        self.budget = dict(budget)
        self.n_calls, self.t0 = 0, time.time()
        self.fail_streak = {}
        self.finished = None                       # {"task", "success", "summary"} once finish() is called
        self.last_image = None
        self.stages_done = set()                   # task-3 stage markers emitted once (task1_end, scan_done)
        self.barcodes_ended = set()
        self.last_approach = None                  # result of the last approach(root_target): phase-1 arrival info
        toolbox.task = task                        # the toolbox reads the running task's config (Toolbox._c1/_c2)
        # the episode's tools: a wall-layout episode (Toolbox.set_wall_layout, called before the Harness is built) also
        # gets the layout-only tools
        self.tools = task_tools(task, toolbox)

    def over_budget(self) -> str | None:
        if self.n_calls >= self.budget["max_tool_calls"]:
            return f"tool-call budget ({self.budget['max_tool_calls']}) exhausted"
        if self.tb.ws.explore_steps > self.budget["max_explore_steps"]:
            return f"exploration budget ({self.budget['max_explore_steps']}) exhausted"
        if time.time() - self.t0 > self.budget["max_wall_s"]:
            return f"time budget ({self.budget['max_wall_s']} s) exhausted"
        return None

    def call(self, name: str, args: dict | None = None) -> dict:
        args = dict(args or {})
        t = time.time()
        rec = {"i": self.n_calls, "tool": name, "args": args}
        parsed = None
        stop = None if name == "finish" else self.over_budget()   # ending the episode is always allowed
        executed = False
        if stop:
            res = {"status": "fail", "data": {}, "reason": stop, "budget_exhausted": True}
        elif name not in TOOLS:
            res = {"status": "fail", "data": {}, "reason": f"unknown tool {name!r}", "rejected": True}
        elif name not in self.tools:
            res = {"status": "fail", "data": {}, "reason": f"tool {name!r} is not part of task {self.task}", "rejected": True}
        else:
            try:
                parsed = TOOLS[name][0](**args).model_dump()
            except ValidationError as e:
                parsed, res = None, {"status": "fail", "data": {}, "reason": f"invalid arguments: {e.errors()}", "rejected": True}
            if parsed is not None:
                res = self._guard(name, parsed)
                if res is None:
                    # the same call (tool + arguments) failing again and again counts even when other calls of the tool
                    # succeed in between (2026-11-07: move_arm('pre') failed 14x, each time after a successful
                    # move_arm('home'), which reset the per-tool streak)
                    ck = name + json.dumps(parsed, sort_keys=True, default=str)
                    if self.fail_streak.get(name, 0) > self.budget["max_retries_per_tool"]:
                        res = {"status": "fail", "data": {}, "reason": f"{name} failed {self.fail_streak[name]} times in a row; retry budget exhausted", "rejected": True}
                    elif self.fail_streak.get(ck, 0) > self.budget["max_retries_per_tool"]:
                        res = {"status": "fail", "data": {}, "rejected": True,
                               "reason": f"this exact {name} call failed {self.fail_streak[ck]} times; retry budget exhausted. Do something "
                                         "else (e.g. next_barcode to move on)"}
                    else:
                        res = self._execute(name, parsed)
                        executed = True
        self.n_calls += 1
        ck = name + json.dumps(parsed if parsed is not None else args, sort_keys=True, default=str)
        if res["status"] == "fail" and not res.get("rejected"):
            self.fail_streak[name] = self.fail_streak.get(name, 0) + 1
            self.fail_streak[ck] = self.fail_streak.get(ck, 0) + 1
        elif res["status"] != "fail":
            self.fail_streak[name] = 0
            self.fail_streak[ck] = 0
        rec.update(result={k: v for k, v in res.items() if k != "image"}, seconds=round(time.time() - t, 2))
        if "image" in res:
            rec["image"] = self.log.image(res["image"], name)
        self.log.tool(rec)
        if executed:                       # after the tool record: the stage events it triggers follow it in the timeline
            self._after(name, parsed, res)
        return res

    def _guard(self, name, a):
        L = self.limits
        if name == "explore":
            if a["direction"].startswith("turn") and not 0 < a["distance"] <= L["max_turn_deg"]:
                return {"status": "fail", "data": {}, "reason": f"turn must be in (0, {L['max_turn_deg']}] deg", "rejected": True}
            if not a["direction"].startswith("turn") and not 0 < a["distance"] <= L["max_move_m"]:
                return {"status": "fail", "data": {}, "reason": f"move must be in (0, {L['max_move_m']}] m", "rejected": True}
        if name == "finish" and a["task"] != self.task:
            return {"status": "fail", "data": {}, "reason": f"this episode is task {self.task}", "rejected": True}
        if name == "finish" and self.task == 3 and a["success"]:
            # a success claim must agree with the scan plan: 2026-11-07 the LLM finished the four-side stack with
            # success=true after 2 of 4 sides (its plan held only the sides seen so far). Giving up (success=false)
            # stays allowed; a refused claim does not end the episode, the tool-call budget still bounds it.
            left = self._task3_work_left()
            if left:
                return {"status": "fail", "data": {}, "rejected": True,
                        "reason": f"finish(success=true) refused: {left}. Call next_barcode (it moves to the next barcode "
                                  "or reports face_done for the next side); finish with success=false only to give up"}
        return None

    def _task3_work_left(self) -> str | None:
        """What is still to do in task 3 according to the scan plan and the stack tour (None when nothing)."""
        ws = self.tb.ws
        if not ws.scan_plan and not (ws.stack or {}).get("planned"):   # a stack side planned empty is a plan
            return "no scan plan yet"
        pending = [b for b in ws.scan_plan if ws.barcodes[b].status == "pending"]
        if pending:
            return f"barcode(s) {', '.join(pending)} still pending"
        nf = self.tb._next_face()
        if nf is not None:
            return f"the {self.tb.TOUR[nf]} side of the stack has not been visited"
        missing = ws.missing_cells()
        if missing:
            return f"requested target cell(s) without a localised label: {missing}"
        warn = ws.grid_warnings()
        if warn:
            return f"the target cells could not be verified: {'; '.join(warn)}"
        return None

    def _execute(self, name, a):
        tb = self.tb
        if name == "finish":               # bookkeeping, not a tool: its errors must surface, not become a tool failure
            return self._finish(a)
        try:
            if name == "look_head":
                r, img = tb.look_head(); return dict(r, image=img)
            if name == "look_wrist":
                r, img = tb.look_wrist(); return dict(r, image=img)
            if name == "approach":
                return tb.approach(a["target"])
            if name == "compute_wrist_target":
                return tb.compute_wrist_target(a["barcode_id"], a["offset"])
            fn = getattr(tb, name)
            return fn(**{k: v for k, v in a.items()})
        except Exception as e:             # tool errors go back to the caller as a structured failure, and are logged
            if _infrastructure_error(e):
                # not something the agent can act on (2026-11-07: a CUDA OOM in the detector came back as a tool
                # failure and the LLM retried useless tools for 40 calls): the episode ends with the error
                raise
            return {"status": "fail", "data": {}, "reason": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}

    # ------------------------------------------------------------------ task-3 bookkeeping
    def _stage(self, name, info=None):
        self.log.stage(name, info)

    def _end_barcode(self, bid, status):
        if bid is None or bid in self.barcodes_ended:
            return
        self.barcodes_ended.add(bid)
        bc = self.tb.ws.barcodes.get(bid)
        info = {"id": bid, "status": status, "content": None if bc is None else bc.content}
        if bc is not None and bc.fail_reason:
            info["reason"] = bc.fail_reason
        self._stage("barcode_end", info)

    def _task1_end(self, info):
        if "task1_end" not in self.stages_done:
            self.stages_done.add("task1_end")
            self._stage("task1_end", info)

    def _scan_done(self):
        if "scan_done" not in self.stages_done:
            self.stages_done.add("scan_done")
            ws = self.tb.ws
            self._stage("scan_done", {"scanned": [e["content"] for e in ws.scanned],
                                      "statuses": {b: ws.barcodes[b].status for b in ws.scan_plan}})

    def _after(self, name, a, res):
        if name == "approach" and a["target"] == "root_target":
            d = res.get("data") or {}
            self.last_approach = {"status": res["status"], "reason": res.get("reason"), "error": d.get("error")}
        if self.task == 3 and name == "plan_stance" and res["status"] == "fail" and (res.get("data") or {}).get("unreachable"):
            self._end_barcode(res["data"]["barcode_id"], "unreachable")     # reported with its reason (stage + log)
        if self.task == 3 and name == "scan" and res["status"] != "ok":
            # a label recorded unreadable ends with a failed scan call (no code read): its barcode_end is due here
            cur = self.tb.ws.current_barcode_id
            bc = self.tb.ws.barcodes.get(cur) if cur else None
            if bc is not None and bc.status == "unreadable":
                self._end_barcode(bc.id, "unreadable")
        if self.task != 3 or res["status"] != "ok":
            return
        ws = self.tb.ws
        d = res["data"]
        if name == "plan_scans":
            n = sum(ws.targeted(b) for b in ws.scan_plan)
            per = self.base_budget.get("per_barcode")
            if per is not None:
                for k, v in per.items():
                    self.budget[k] = self.base_budget[k] + n * v
            if not d.get("updated"):                  # a plan update (new labels seen later) is not the task-1 end
                self._task1_end({"arrival": self.last_approach, "n_planned": n,
                                 "budget": {k: self.budget[k] for k in ("max_tool_calls", "max_wall_s")}})
            if d.get("first_for_face") and (ws.stack or {}).get("multi_face"):   # first plan of a stack side (I3)
                st = ws.stack or {}
                fid = int(d.get("face_id", 0))
                nrm = self.tb._face_dirs()[fid] if fid in (st.get("faces") or {}) else None   # odometry, re-registered
                self._stage("face_start", {"face": fid, "normal_odom": None if nrm is None else [round(float(v), 3) for v in nrm],
                                           "n_planned": int(d.get("n", 0))})
        elif name == "next_barcode":
            if not (d.get("face_done") or d.get("done")):
                # retries are counted per barcode; a side change (face_done) does not reset them, so a failing
                # go_to_next_face / plan_scans cycle hits the retry limit (2026-11-07: it looped until the budget)
                self.fail_streak = {}
            if d.get("closed"):
                self._end_barcode(d["closed"]["id"], d["closed"]["status"])
            if (d.get("face_done") or d.get("done")) and (ws.stack or {}).get("multi_face"):
                fid = int(d.get("face_id", 0))
                self._stage("face_end", {"face": fid, "statuses": {b: ws.barcodes[b].status for b in ws.scan_plan
                                                                   if ws.barcodes[b].face_id == fid}})
            if d.get("done"):
                self._scan_done()
            elif not d.get("face_done"):
                self._stage("barcode_start", {"id": d["barcode_id"], "index": d["index"]})
        elif name == "scan":
            bc = ws.barcodes.get(ws.current_barcode_id) if ws.current_barcode_id else None
            if bc is not None and bc.status in ("scanned", "duplicate", "unreadable"):
                self._end_barcode(bc.id, bc.status)
        elif name == "search_cell":
            # labels added to the plan (a second label in the box) get their per-barcode budget like planned ones
            per = self.base_budget.get("per_barcode")
            if d.get("added") and per is not None:
                for k, v in per.items():
                    self.budget[k] += len(d["added"]) * v
            if d.get("barcode_status") == "no_label":
                self._end_barcode(d["barcode_id"], "no_label")

    def _finish(self, a):
        fin = dict(a)
        if self.task == 3:
            ws = self.tb.ws
            self._task1_end({"arrival": self.last_approach, "n_planned": len(ws.scan_plan), "note": "finish before plan_scans"})
            cur = ws.current_barcode_id
            if cur is not None and ws.barcodes[cur].status == "pending":
                ws.barcodes[cur].status = "failed"
                self._end_barcode(cur, "failed")
            self._scan_done()
            statuses = {b: ws.barcodes[b].status for b in ws.scan_plan}
            task = [s for s in statuses.values() if s != "not_targeted"]
            # success comes from the ledger, not from the caller: every planned (targeted) barcode read once, as itself,
            # on a multi-side task every side of the tour visited (a plan only holds the sides seen so far), and with a
            # target spec every requested cell found (a cell without a label is never silently dropped)
            fin.update(claimed_success=a["success"],
                       # a label verified unreadable at close range (no code decodable in full view) is a result, not a
                       # failure of the robot; so is a box of a wall layout whose whole face was seen without a label
                       # (no_label: user decision 2026-11-19 Q2, the evaluator catches a detector miss);
                       # failed / unreachable / pending are not
                       success=any(s == "scanned" for s in task)
                       and all(s in ("scanned", "unreadable", "no_label") for s in task) and self.tb._next_face() is None
                       and not ws.missing_cells() and not ws.grid_warnings(),
                       scanned=[dict(e) for e in ws.scanned], plan_status=statuses)
            if ws.target_grid is not None:
                fin["missing_cells"] = ws.missing_cells()
                fin["grid_warnings"] = ws.grid_warnings()
        self.finished = fin
        return {"status": "ok", "data": fin, "reason": "episode finished"}


def _infrastructure_error(e: BaseException) -> bool:
    """Errors of the machine / simulator connection, not of the task: out of (GPU) memory, a lost or timed-out RPC
    connection to the robot / simulator server."""
    if isinstance(e, (MemoryError, ConnectionError, TimeoutError)):
        return True
    import torch
    return isinstance(e, torch.cuda.OutOfMemoryError)
