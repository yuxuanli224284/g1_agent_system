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


def _start_text(task, selector, multi_face=False):
    if task == 3:
        sides = (" This task covers EVERY side of the box stack (front, right, back, left): after a side is finished, "
                 "next_barcode returns face_done and you call go_to_next_face, then plan_scans." if multi_face else
                 " This task covers the side of the boxes facing you only.")
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


def run_agent_full(h, tcfg, selector, llm, log):
    b = tcfg["budget"]
    lean = h.task == 3 and h.tb.cfg["robot"]["lean_views"]      # only the task-3 prompt has the lean-view note
    kw = {"selector": selector, "max_tool_calls": b["max_tool_calls"], "max_explore_steps": b["max_explore_steps"],
          "head_view_note": LEAN_VIEW_NOTE if lean else ""}
    if h.task == 3:
        kw["per_barcode_calls"] = b["per_barcode"]["max_tool_calls"]
    system, version = load_prompt(f"task{h.task}_agent_full", **kw)
    if h.task == 3 and not lean:
        version += " (head camera without lean views: no lean-view note)"
    tools = openai_tools(TASK_TOOLS[h.task])
    if "move_arm" in TASK_TOOLS[h.task]:
        # move_arm("home") needs the episode's initial arm pose (the rule orchestrators record it the same way); without
        # it every home move fails with "home pose unknown", and the task-3 prompt asks for home before each walk
        h.tb.remember_home()
    _, img = h.tb.look_head()
    task_text = _start_text(h.task, selector, getattr(h.tb, "multi_face", False))
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
