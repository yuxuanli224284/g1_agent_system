"""agent_full: the LLM decides every step through tool calling over the shared registry.

Task 3: the scan order is never the LLM's choice (plan_scans fixes it, next_barcode follows it). The message history
is compacted at two kinds of boundary, so the context is bounded by one barcode's exchange whatever the number of
barcodes and sides (vLLM max_model_len 32768; 2026-11-07: the four-side stack, 12 codes, overflowed at 33288 tokens
after 6 codes, ~4 k tokens per barcode):
- the phase boundary (the first successful plan_scans): system prompt + a summary of phase 1 and the plan;
- every later successful next_barcode: system prompt + that phase-1 summary + the next_barcode result. The barcodes
  handled before are closed; their state (status, content, fail reason, side) is in the scan plan that next_barcode
  returns, which is the authoritative record, so their messages are dropped."""
from __future__ import annotations

import json

from ..harness.registry import TASK_TOOLS, openai_tools
from ..tools.scan_order import describe_targets
from .common import clean, image_message, load_prompt, log_llm, tool_result_text

KEEP_IMAGES = 2          # images kept in the context (older ones -> text placeholder); vLLM limit-mm-per-prompt is 8
MAX_NUDGES = 3           # replies without a tool call tolerated before the episode is ended


def _prune_images(msgs):
    seen = 0
    for m in reversed(msgs):
        if isinstance(m.get("content"), list):
            if any(p.get("type") == "image_url" for p in m["content"]):
                seen += 1
                if seen > KEEP_IMAGES:
                    m["content"] = [p for p in m["content"] if p.get("type") != "image_url"] + [{"type": "text", "text": "[older image removed]"}]


def _start_text(task, selector, multi_face=False, targets=None):
    """targets: the parsed target spec (scan_order.parse_targets) or None (every label)."""
    if task == 3:
        sides = (" This task covers EVERY side of the box stack (front, right, back, left): after a side is finished, "
                 "next_barcode returns face_done and you call go_to_next_face, then plan_scans." if multi_face else
                 " This task covers the side of the boxes facing you only.")
        if targets is not None:
            return (f"Start task 3: scan only {describe_targets(targets)}, "
                    f"{'row by row from the top, left to right within a row' if targets['order'] == 'rows' else 'column by column from the left, top to bottom within a column'}"
                    " (the tools fix this order). Work from near your start pose: do not walk back to observe the whole "
                    "wall. plan_scans looks from every standing head view from here (upright and leaned back) to see the "
                    "top row and the edges of the wall, and counts the rows and columns from there." + sides)
        return "Start task 3 (phase 1: find all boxes and approach them)." + sides
    return f"Start task {task}. Target: the {selector} box."


def phase1_summary(h, plan_result: dict, later: list, task_text: str) -> str:
    """The user message that replaces the phase-1 history: the task statement of the start message (it would otherwise
    be dropped with it: 2026-11-07 the LLM, no longer told that every side is to be scanned, finished after 2 of 4),
    what phase 1 did, the plan, the extended budget, and the calls the same reply made after plan_scans
    (later = [(name, result)])."""
    arr = h.last_approach
    lines = [f"Task (as given at the start): {task_text}",
             f"Phase 1 is finished after {h.n_calls} tool calls (explore moves: {h.tb.ws.explore_steps}).",
             "Last approach: " + ("none" if arr is None else json.dumps(arr, ensure_ascii=False)) + ".",
             "plan_scans result (the fixed scan order): " + tool_result_text(plan_result),
             *[f"Then {name}: {tool_result_text(res)}" for name, res in later],
             f"Budget now: at most {h.budget['max_tool_calls']} tool calls in total ({h.n_calls} used).",
             "Phase 2: call next_barcode and handle the barcodes one by one as the system prompt describes."]
    return "\n".join(lines)


def barcode_boundary_text(h, nb_result: dict, later: list) -> str:
    """The part of the compacted context that follows the phase-1 summary at a next_barcode boundary."""
    lines = [f"Context compacted at next_barcode (after {h.n_calls} tool calls): the messages about the barcodes handled "
             "before were removed. The scan_plan below is the authoritative state of every barcode.",
             "next_barcode result: " + tool_result_text(nb_result),
             *[f"Then {name}: {tool_result_text(res)}" for name, res in later],
             f"Budget now: at most {h.budget['max_tool_calls']} tool calls in total ({h.n_calls} used)."]
    return "\n".join(lines)


# task-3 prompt sentence for a head camera with lean views (robot.yaml head_camera.<model>.lean_views); without them
# the prompt is the one before lean views (v7 text)
LEAN_VIEW_NOTE = ("\n  detect_boxes and the perception tools lean the torso back by themselves to look higher (the robot "
                  "stands upright again\n  before it walks or moves the arm); you do not control the head view.")

# task-3 phase-1 texts (prompt placeholders): every label (the v8 text) | a target spec (columns / rows of the wall given
# in robot terms; the robot works from near its start pose, user decision 2026-11-19)
PHASE1_ALL_GOAL = ("Phase 1 (find): find the cardboard boxes (on a table, on the floor or stacked) and walk to the observation "
                   "pose in front of ALL of them, where\nevery front face is visible in the head camera.")
PHASE1_ALL_RULES = """- Phase 1 sequence: detect_boxes -> select_target_box("all") -> fit_front_face -> compute_obs_pose -> approach.
  If no box is visible, call look_head and choose an explore move (left / right / back / forward <= 0.5 m,
  turn_left / turn_right <= 15 deg); {head_view}.{head_view_note}
  If fit_front_face fails because the boxes are too far (no valid depth), move forward and try again.
- Phase 1 ends when approach returns ok, or partial ("walk finished; arrival check not within tolerance"): in both
  cases call plan_scans next; do NOT explore to fix a small arrival error (the scan phase plans its own stances).
  plan_scans localises every label and fixes the scan order (columns left to right, top to bottom). Only if it finds
  no label, explore and redo phase 1. Explore moves that would end closer than the minimum distance to a box face are
  refused by the tool."""
PHASE1_TARGET_GOAL = ("Phase 1 (find): from near the start pose, find the box wall in front of you and localise its barcode "
                      "labels.\nThe task names the labels to scan by their place on the wall ({targets}).")
PHASE1_TARGET_RULES = """- Phase 1 sequence: detect_boxes -> select_target_box("all") -> plan_scans. Work from near the start pose: do NOT
  call compute_obs_pose / approach and do not walk back to observe the whole wall. plan_scans looks from every
  standing head view from here (upright and leaned back) to see the top row and the edges of the wall, counts the
  columns and rows of the labels from the edges the task names, plans ONLY the labels in the requested cells (the others
  are listed as not_targeted and are never scanned) and fixes the scan order ({order}).
  It reports a requested cell without a label, an edge of the wall it could not see, or a count it could not verify
  from the box faces, as a WARNING in its reason (finish(success=true) is then refused).
  If no box is visible, call look_head and choose an explore move (left / right / back / forward <= 0.5 m,
  turn_left / turn_right <= 15 deg); {head_view}.{head_view_note}
  Only if plan_scans finds no label at all, explore a little sideways and call detect_boxes and plan_scans
  again. Explore moves that would end closer than the minimum distance to a box face are refused by the tool."""


def task3_prompt_kw(tb, lean: bool) -> dict:
    """Variant and target texts of the task-3 prompt (robot.yaml variants.<v>.texts, Toolbox.scan_targets)."""
    tx = tb.cfg["robot"]["texts"]
    t = getattr(tb, "scan_targets", None)
    note = LEAN_VIEW_NOTE if lean else ""
    if t is None:
        return {"robot_text": tx["robot"], "phase1_goal": PHASE1_ALL_GOAL, "scan_what": "every barcode label on the boxes",
                "phase1_rules": PHASE1_ALL_RULES.format(head_view=tx["head_view"], head_view_note=note), "success_note": "",
                "final_refine_note": tx["final_refine_note"]}
    order = ("rows top to bottom, left to right inside a row" if t["order"] == "rows"
             else "columns left to right, top to bottom inside a column")
    return {"robot_text": tx["robot"], "phase1_goal": PHASE1_TARGET_GOAL.format(targets=describe_targets(t)),
            "scan_what": describe_targets(t),
            "phase1_rules": PHASE1_TARGET_RULES.format(order=order, head_view=tx["head_view"], head_view_note=note),
            "success_note": " and every requested cell found and verified (labels outside the requested cells do not count)",
            "final_refine_note": tx["final_refine_note"]}


def run_agent_full(h, tcfg, selector, llm, log):
    b = tcfg["budget"]
    lean = h.task == 3 and h.tb.cfg["robot"]["lean_views"]      # only the task-3 prompt has the lean-view note
    kw = {"selector": selector, "max_tool_calls": b["max_tool_calls"], "max_explore_steps": b["max_explore_steps"],
          "robot_text": h.tb.cfg["robot"]["texts"]["robot"], "head_view": h.tb.cfg["robot"]["texts"]["head_view"]}
    if h.task == 3:
        kw["per_barcode_calls"] = b["per_barcode"]["max_tool_calls"]
        kw.update(task3_prompt_kw(h.tb, lean))
    system, version = load_prompt(f"task{h.task}_agent_full", **kw)
    version += f" ({h.tb.cfg['robot']['robot_variant']})"
    if h.task == 3 and not lean:
        version += " (head camera without lean views: no lean-view note)"
    if getattr(h.tb, "scan_targets", None) is not None:
        version += " (target spec)"
    tools = openai_tools(TASK_TOOLS[h.task], h.tb.cfg["robot"]["tool_texts"])
    if "move_arm" in TASK_TOOLS[h.task]:
        # move_arm("home") needs the episode's initial arm pose (the rule orchestrators record it the same way); without
        # it every home move fails with "home pose unknown", and the task-3 prompt asks for home before each walk
        h.tb.remember_home()
    _, img = h.tb.look_head()
    task_text = _start_text(h.task, selector, getattr(h.tb, "multi_face", False), getattr(h.tb, "scan_targets", None))
    msgs = [{"role": "system", "content": system}, image_message(llm, log, img, task_text + " Current head camera image attached.", "start")]
    nudges = 0
    compacted = False
    head = None                                         # the phase-1 summary, kept at every later compaction
    while h.finished is None:
        stop = h.over_budget()
        if stop:
            h.call("finish", {"task": h.task, "success": False, "summary": stop})
            break
        _prune_images(msgs)
        r = llm.chat(clean(msgs), tools=tools, tool_choice="auto")
        log_llm(log, "agent_full", version, msgs, r)
        msgs.append(r["assistant_message"])
        if not r["tool_calls"]:
            nudges += 1
            if nudges > MAX_NUDGES:
                h.call("finish", {"task": h.task, "success": False, "summary": "the LLM stopped calling tools"})
                break
            msgs.append({"role": "user", "content": "Write one sentence of reasoning, then call a tool (or finish)."})
            continue
        images, boundary, later = [], None, []          # boundary = (kind, result) of the first compaction point
        for tc in r["tool_calls"]:
            if tc["parse_error"]:
                res = {"status": "fail", "data": {}, "reason": tc["parse_error"]}
            else:
                res = h.call(tc["name"], tc["arguments"])
            if "image" in res:
                images.append((tc["name"], res["image"]))
            msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": tool_result_text(res)})
            if boundary is not None:
                later.append((tc["name"], res))
            elif res["status"] == "ok" and tc["name"] == "plan_scans" and not compacted:
                boundary = ("phase", res)
            elif res["status"] == "ok" and tc["name"] == "next_barcode" and compacted:
                boundary = ("barcode", res)
            if h.finished is not None:
                break
        if boundary is not None and h.finished is None:
            kind, res = boundary
            if kind == "phase":
                head = summary = phase1_summary(h, res, later, task_text)
                compacted = True
            else:
                summary = head + "\n\n" + barcode_boundary_text(h, res, later)
            log.llm({"kind": "history_compaction", "at": kind, "dropped_messages": len(msgs) - 1, "summary": summary})
            msgs = [msgs[0], {"role": "user", "content": summary}]
            continue                                   # images of the dropped turn belong to the closed segment
        for name, im in images:
            msgs.append(image_message(llm, log, im, f"Image from {name}.", name))
