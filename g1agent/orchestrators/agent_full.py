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

from ..harness.registry import (PLAN_SCANS_BODY, PLAN_SCANS_FROM_OBS, PLAN_SCANS_LAYOUT_BODY, PLAN_SCANS_STRIP_BODY, TOOLS,
                                openai_tools, task_tools)
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


# the scan order of a target spec in the start message, per order rule (scan_order.ORDER_RULES)
ORDER_WORDS = {"rows": "row by row from the top, left to right within a row",
               "columns": "column by column from the left, top to bottom within a column",
               "serpentine": "row by row from the top, the top row left to right, the next right to left, alternating",
               # a template (strip width w): formatted only for the order "strips"
               "strips": "strip by strip from the left end of the wall, {w} columns per strip; inside a strip row by row "
                         "from the top, the first row left to right, the next right to left, alternating"}
# ... and in the phase-1 rules
ORDER_RULE_WORDS = {"rows": "rows top to bottom, left to right inside a row",
                    "columns": "columns left to right, top to bottom inside a column",
                    "serpentine": "rows top to bottom, the top row left to right, the next right to left, alternating",
                    "strips": "strip by strip from the left end of the wall, {w} columns per strip; inside a strip row "
                              "by row from the top, the first row left to right, the next right to left, alternating"}


def order_words(words: dict, targets: dict) -> str:
    """The order of a target spec in words (ORDER_WORDS / ORDER_RULE_WORDS); the order "strips" names its width."""
    return words["strips"].format(w=targets["strip_columns"]) if targets["order"] == "strips" else words[targets["order"]]


def _start_text(task, selector, multi_face=False, targets=None, phase1_view="start_pose", lean=True, layout=None):
    """targets: the parsed target spec (scan_order.parse_targets) or None (every label). phase1_view (Toolbox.
    set_phase1_view) and lean (the effective lean-view flag, Toolbox.lean_views_on) choose the phase-1 sentences of a
    target-spec task; "leaned back" is said only when lean views are on. layout: the given wall layout (Toolbox.
    wall_layout) or None."""
    if task == 3:
        sides = (" This task covers EVERY side of the box stack (front, right, back, left): after a side is finished, "
                 "next_barcode returns face_done and you call go_to_next_face, then plan_scans." if multi_face else
                 " This task covers the side of the boxes facing you only.")
        if targets is not None:
            head = (f"Start task 3: scan only {describe_targets(targets)}, {order_words(ORDER_WORDS, targets)}"
                    " (the tools fix this order).")
            if phase1_view == "obs_pose" and layout is not None and targets["order"] == "strips":
                views = "upright and leaned-back head views" if lean else "upright head views only"
                return (head + f" The wall layout is given: {layout['columns']} columns x {layout['rows']} rows of boxes, "
                        f"each {layout['box_w']:g} m wide x {layout['box_h']:g} m tall, standing on the floor. You work in "
                        f"strips of {targets['strip_columns']} columns from the LEFT end of the wall: walk to the pose "
                        f"the tools compute at the left end (compute_obs_pose, then approach; {views}): from there the "
                        "head sees the left side edge. Then plan_scans registers the wall and plans one entry per "
                        "requested box." + sides)
            if phase1_view == "obs_pose" and layout is not None:
                views = "upright and leaned-back head views" if lean else "upright head views only"
                return (head + f" The wall layout is given: {layout['columns']} columns x {layout['rows']} rows of boxes, "
                        f"each {layout['box_w']:g} m wide x {layout['box_h']:g} m tall, standing on the floor. Walk to "
                        f"the observation pose the tools compute ({views}; compute_obs_pose, then approach, which may walk "
                        "back): from there the head sees both side edges of the wall. Then plan_scans registers the wall "
                        "to the layout and plans one entry per requested box." + sides)
            if phase1_view == "obs_pose":
                views = "upright and leaned-back head views" if lean else "upright head views only"
                return (head + f" Walk to the observation pose the tools compute ({views}; compute_obs_pose, then "
                        "approach, which may walk back): from there the head sees the wall edges the task counts from and "
                        "the requested rows. Then plan_scans counts the rows and columns from there." + sides)
            look = ("from every standing head view from here (upright and leaned back) to see the top row and the edges "
                    "of the wall" if lean else "from the upright head view and its squat views from here to see the "
                    "edges of the wall")
            return (head + " Work from near your start pose: do not walk back to observe the whole "
                    f"wall. plan_scans looks {look}, and counts the rows and columns from there." + sides)
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
# in robot terms): phase1_view "start_pose" = the robot works from near its start pose (user decision 2026-11-19, 4-box
# task), "obs_pose" = it walks (also back) to the observation pose planned from the spec (user decision 2026-11-19 Q1)
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
  call compute_obs_pose / approach and do not walk back to observe the whole wall. plan_scans looks {look_text}, counts the
  columns and rows of the labels from the edges the task names, plans ONLY the labels in the requested cells (the others
  are listed as not_targeted and are never scanned) and fixes the scan order ({order}).
  It reports a requested cell without a label, an edge of the wall it could not see, or a count it could not verify
  from the box faces, as a WARNING in its reason (finish(success=true) is then refused).
  If no box is visible, call look_head and choose an explore move (left / right / back / forward <= 0.5 m,
  turn_left / turn_right <= 15 deg); {head_view}.{head_view_note}
  Only if plan_scans finds no label at all, explore a little sideways and call detect_boxes and plan_scans
  again. Explore moves that would end closer than the minimum distance to a box face are refused by the tool."""
# where plan_scans looks from near the start pose: every standing view with lean views (the 4-box text), else upright
LOOK_LEAN = ("from every\n  standing head view from here (upright and leaned back) to see the top row and the edges of the "
             "wall")
LOOK_UPRIGHT = "from the upright head view and its squat views\n  from here to see the edges of the wall"
PHASE1_TARGET_OBS_GOAL = ("Phase 1 (find): find the box wall in front of you, walk to the observation pose the tools compute "
                          "for the task and localise its\nbarcode labels there. The task names the labels to scan by their "
                          "place on the wall ({targets}).")
PHASE1_TARGET_OBS_RULES = """- Phase 1 sequence: detect_boxes -> select_target_box("all") -> fit_front_face -> compute_obs_pose -> approach ->
  plan_scans. compute_obs_pose picks the nearest pose from which the {view_text} sees the wall edges the task counts
  from and the requested rows (rows from the bottom: the floor line under the wall; all columns: both side edges);
  approach walks there, also backwards, and widens the view as it sees more of the wall.
- Phase 1 ends when approach returns ok, or partial ("walk finished; arrival check not within tolerance"): in both
  cases call plan_scans next; do NOT explore to fix a small arrival error (the scan phase plans its own stances).
  plan_scans looks from there ({look_text}), counts the columns and rows of the labels from the edges the task names,
  plans ONLY the labels in the requested cells (the others are listed as not_targeted and are never scanned) and fixes
  the scan order ({order}).
  It reports a requested cell without a label, an edge of the wall it could not see, or a count it could not verify
  from the box faces, as a WARNING in its reason (finish(success=true) is then refused).
  If no box is visible, call look_head and choose an explore move (left / right / back / forward <= 0.5 m,
  turn_left / turn_right <= 15 deg); {head_view}.{head_view_note}
  If fit_front_face fails because the boxes are too far (no valid depth), move forward and try again.
  Only if plan_scans finds no label at all, explore a little sideways and redo phase 1. Explore moves that would end
  closer than the minimum distance to a box face are refused by the tool."""


# wall layout (Toolbox.set_wall_layout, user decision 2026-11-19 approach A): phase 1 walks to the observation pose and
# registers the wall to the given layout; phase 2 visits every requested box with the hand-camera search (search_cell)
PHASE1_TARGET_LAYOUT_GOAL = ("Phase 1 (find): find the box wall in front of you, walk to the observation pose the tools "
                             "compute for the given wall layout\nand register the wall there. The task names the boxes to "
                             "scan by their place on the wall ({targets}).")
PHASE1_TARGET_LAYOUT_RULES = """- Phase 1 sequence: detect_boxes -> select_target_box("all") -> fit_front_face -> compute_obs_pose -> approach ->
  plan_scans. compute_obs_pose picks the nearest pose from which the {view_text} sees both side edges of the wall and
  its bottom row (the layout gives the wall's size); approach walks there, also backwards.
- Phase 1 ends when approach returns ok, or partial ("walk finished; arrival check not within tolerance"): in both
  cases call plan_scans next; do NOT explore to fix a small arrival error (the scan phase plans its own stances).
  plan_scans registers the wall seen from there to the given layout (front plane, both side edges, floor) and plans
  one entry of kind cell per requested box, in the fixed scan order ({order}).
  If plan_scans reports a side edge not in view, call compute_obs_pose and approach again, then plan_scans. If it
  reports that the wall does not match the layout, explore a little sideways and redo phase 1 once; if it still does
  not match, finish(success=false).
  If no box is visible, call look_head and choose an explore move (left / right / back / forward <= 0.5 m,
  turn_left / turn_right <= 15 deg); {head_view}.{head_view_note}
  If fit_front_face fails because the boxes are too far (no valid depth), move forward and try again. Explore moves
  that would end closer than the minimum distance to a box face are refused by the tool."""
# phase-2 loop of the task-3 prompt: every scenario without a wall layout keeps the v9 text (PHASE2_LOOP + the robot
# variant's final_refine_note, the prompt hash is unchanged); a wall layout has its own loop
PHASE2_LOOP = """- Phase 2: you never choose the order. Repeat for each barcode:
  next_barcode -> compute_wrist_target (never pass an offset: the stand-off is adaptive) -> plan_stance ->
  move_arm("home") -> approach(target="stance_target") -> move_arm("pre") -> refine_with_wrist -> move_arm("final") ->
  check_wrist_placement -> scan -> move_arm("home") -> plan_scans (adds labels seen for the first time from this
  closer / lower view)."""
PHASE2_LOOP_LAYOUT = """- Phase 2: you never choose the order. Each plan entry of kind cell is one requested box whose label is not found yet:
  next_barcode -> compute_wrist_target (the hand-camera view of the whole box face) -> plan_stance -> move_arm("home") ->
  approach(target="stance_target") -> search_cell (finds the label inside the box face; the entry becomes that label) ->
  move_arm("final") -> refine_with_wrist -> move_arm("final") -> check_wrist_placement -> scan -> move_arm("home").
  If search_cell finds the label but says its scan pose is not reachable from this stance (the stance was planned to
  view the whole face): move_arm("home"), compute_wrist_target -> plan_stance -> approach -> move_arm("pre") ->
  refine_with_wrist -> move_arm("final") -> refine_with_wrist -> move_arm("final") -> check_wrist_placement -> scan.
  search_cell recording "no_label" (the whole face was seen and holds no label) is a result: move_arm("home"),
  next_barcode. search_cell partial (the face not fully seen, or a label at its border not seen whole) or failing with
  "hand-camera view not reachable from this stance": move_arm("home"), compute_wrist_target -> plan_stance -> approach
  -> search_cell again (once), then next_barcode. An entry of kind label
  (a second label search_cell added): compute_wrist_target -> plan_stance -> move_arm("home") -> approach -> move_arm("pre")
  -> refine_with_wrist -> move_arm("final") -> refine_with_wrist -> move_arm("final") -> check_wrist_placement -> scan."""
# wall layout worked in strips (scan_targets order "strips", Toolbox.strip_mode; user decisions 2026-11-20): phase 1 walks
# to the LEFT end and registers the wall from its left side edge; phase 2 works the strips one after the other
PHASE1_TARGET_STRIP_GOAL = ("Phase 1 (find): find the box wall in front of you, walk to its LEFT end (the pose the tools "
                            "compute for the first strip of the given wall layout)\nand register the wall there. The task "
                            "names the boxes to scan by their place on the wall ({targets}).")
PHASE1_TARGET_STRIP_RULES = """- Phase 1 sequence: detect_boxes -> select_target_box("all") -> fit_front_face -> compute_obs_pose -> approach ->
  plan_scans. compute_obs_pose picks the nearest pose from which the {view_text} sees the wall's left side edge and the
  first strip (from what it has seen so far: first in front of the part of the wall where the left end can be);
  approach walks there and re-plans as it sees more of the wall.
- Phase 1 ends when approach returns ok, or partial ("walk finished; arrival check not within tolerance"): in both
  cases call plan_scans next; do NOT explore to fix a small arrival error (the scan phase plans its own stances).
  plan_scans registers the wall from its left side edge (front plane, left edge; the floor where in view) and plans
  one entry of kind cell per requested box, in the fixed order ({order}).
  If plan_scans reports the left side edge not in view, call compute_obs_pose and approach again, then plan_scans. If
  compute_obs_pose or plan_scans reports that the wall does not match the layout, explore a little sideways and redo
  phase 1 once; if it still does not match, finish(success=false).
  If no box is visible, call look_head and choose an explore move (left / right / back / forward <= 0.5 m,
  turn_left / turn_right <= 15 deg); {head_view}.{head_view_note}
  If fit_front_face fails because the boxes are too far (no valid depth), move forward and try again. Explore moves
  that would end closer than the minimum distance to a box face are refused by the tool."""
PHASE2_LOOP_STRIP = """- Phase 2: you never choose the order. The wall is worked in strips of {w} columns from the left end; next_barcode
  reports the strip, its rows and how many boxes of each row are handled. Inside a strip it goes row by row from the
  top (plan_stance squats lower for lower rows by itself and keeps the robot where it stands when it can); the next
  strip comes only after the current one is finished. Work only on the current box: next_barcode gives the others.
  An entry of kind cell is a requested box whose label is not found yet:
  next_barcode -> compute_wrist_target -> plan_stance -> move_arm("home") -> approach(target="stance_target") (walks
  or stays, squats, re-registers the wall) -> search_cell (searches the current box and the other boxes this stance was
  planned for; each label found makes its entry a label) -> move_arm("home") -> then the label path of the current entry.
  Label path (an entry of kind label): compute_wrist_target -> plan_stance -> move_arm("home") -> approach ->
  move_arm("pre") -> refine_with_wrist -> move_arm("final") -> refine_with_wrist -> move_arm("final") ->
  check_wrist_placement -> scan -> move_arm("home").
  If search_cell says the current label can be scanned from this stance: move_arm("final") -> refine_with_wrist ->
  move_arm("final") -> check_wrist_placement -> scan instead.
  search_cell recording "no_label" for a box is a result. search_cell partial, or failing with "hand-camera view not
  reachable from this stance": move_arm("home"), compute_wrist_target -> plan_stance -> approach -> search_cell again
  (once), then next_barcode.
  When next_barcode returns strip_done=true: call go_to_next_strip (never walk to the next strip with explore /
  approach yourself), then next_barcode. If go_to_next_strip fails: move_arm("home") and call it once more; if it fails
  again, finish(success=false)."""
# sentences appended to the tool texts in a strip episode, after LAYOUT_TOOL_NOTES (which stay as they are: they are in
# the approach-A tool texts)
STRIP_TOOL_NOTES = {
    "search_cell": " In strips it also searches the other boxes of the row that this stance was planned for.",
    "next_barcode": " In strips it returns strip_done=true when the current strip is finished (call go_to_next_strip).",
    "plan_stance": " In strips it first tries the place where the robot stands (only squatting), then plans one stance "
                   "for the row's boxes of the strip."}
# sentences appended to the effective tool texts in a wall-layout episode (run_agent_full)
LAYOUT_TOOL_NOTES = {
    "compute_wrist_target": " For an entry of kind cell it plans the hand-camera view of the whole box face (call plan_stance, "
                            "approach, then search_cell).",
    "refine_with_wrist": " With a wall layout it looks for the label only inside the current box face.",
    "scan": " With a wall layout it is refused for an entry of kind cell (call search_cell first) and for a box recorded "
            "no_label.",
    "next_barcode": " With a wall layout each entry is one requested box (kind cell) until search_cell finds its label "
                    "(kind label); the data says the kind."}


def task3_prompt_kw(tb, lean: bool) -> dict:
    """Variant and target texts of the task-3 prompt (robot.yaml variants.<v>.texts, Toolbox.scan_targets, Toolbox.
    phase1_view, Toolbox.wall_layout). lean: the effective lean-view flag (Toolbox.lean_views_on): without it no text
    mentions leaning back."""
    tx = tb.cfg["robot"]["texts"]
    t = getattr(tb, "scan_targets", None)
    note = LEAN_VIEW_NOTE if lean else ""
    loop = PHASE2_LOOP + tx["final_refine_note"]
    if t is None:
        return {"robot_text": tx["robot"], "phase1_goal": PHASE1_ALL_GOAL, "scan_what": "every barcode label on the boxes",
                "phase1_rules": PHASE1_ALL_RULES.format(head_view=tx["head_view"], head_view_note=note), "success_note": "",
                "final_refine_note": tx["final_refine_note"], "phase2_loop": loop}
    order = order_words(ORDER_RULE_WORDS, t)
    if getattr(tb, "wall_layout", None) is not None and getattr(tb, "strip_columns", None):
        return {"robot_text": tx["robot"], "phase1_goal": PHASE1_TARGET_STRIP_GOAL.format(targets=describe_targets(t)),
                "scan_what": describe_targets(t),
                "phase1_rules": PHASE1_TARGET_STRIP_RULES.format(
                    order=order, head_view=tx["head_view"], head_view_note=note,
                    view_text="head (upright or leaned back)" if lean else "upright head"),
                "success_note": " and every requested box handled: its label read, or recorded unreadable or no_label",
                "final_refine_note": tx["final_refine_note"],
                "phase2_loop": PHASE2_LOOP_STRIP.format(w=t["strip_columns"])}
    if getattr(tb, "wall_layout", None) is not None:
        return {"robot_text": tx["robot"], "phase1_goal": PHASE1_TARGET_LAYOUT_GOAL.format(targets=describe_targets(t)),
                "scan_what": describe_targets(t),
                "phase1_rules": PHASE1_TARGET_LAYOUT_RULES.format(
                    order=order, head_view=tx["head_view"], head_view_note=note,
                    view_text="head (upright or leaned back)" if lean else "upright head"),
                "success_note": " and every requested box handled: its label read, or recorded unreadable or no_label",
                "final_refine_note": tx["final_refine_note"], "phase2_loop": PHASE2_LOOP_LAYOUT}
    if tb.phase1_view == "obs_pose":
        goal = PHASE1_TARGET_OBS_GOAL.format(targets=describe_targets(t))
        rules = PHASE1_TARGET_OBS_RULES.format(
            order=order, head_view=tx["head_view"], head_view_note=note,
            view_text="head (upright or leaned back)" if lean else "upright head",
            look_text=("every standing head view, upright and leaned back, and its squat views" if lean
                       else "the upright head view and its squat views"))
    else:
        goal = PHASE1_TARGET_GOAL.format(targets=describe_targets(t))
        rules = PHASE1_TARGET_RULES.format(order=order, head_view=tx["head_view"], head_view_note=note,
                                           look_text=LOOK_LEAN if lean else LOOK_UPRIGHT)
    return {"robot_text": tx["robot"], "phase1_goal": goal,
            "scan_what": describe_targets(t),
            "phase1_rules": rules,
            "success_note": " and every requested cell found and verified (labels outside the requested cells do not count)",
            "final_refine_note": tx["final_refine_note"], "phase2_loop": loop}


def run_agent_full(h, tcfg, selector, llm, log):
    b = tcfg["budget"]
    # only the task-3 prompt has the lean-view note; the effective flag: camera model and scenario head_views
    lean = h.task == 3 and h.tb.lean_views_on()
    kw = {"selector": selector, "max_tool_calls": b["max_tool_calls"], "max_explore_steps": b["max_explore_steps"],
          "robot_text": h.tb.cfg["robot"]["texts"]["robot"], "head_view": h.tb.cfg["robot"]["texts"]["head_view"]}
    if h.task == 3:
        kw["per_barcode_calls"] = b["per_barcode"]["max_tool_calls"]
        kw.update(task3_prompt_kw(h.tb, lean))
    system, version = load_prompt(f"task{h.task}_agent_full", **kw)
    version += f" ({h.tb.cfg['robot']['robot_variant']})"
    if h.task == 3 and not lean:
        version += (" (head views: upright only, scenario)" if h.tb.head_views == "upright"
                    else " (head camera without lean views: no lean-view note)")
    if getattr(h.tb, "scan_targets", None) is not None:
        version += " (target spec)"
    layout = getattr(h.tb, "wall_layout", None) if h.task == 3 else None
    strips = layout is not None and getattr(h.tb, "strip_columns", None) is not None
    if strips:
        version += " (wall layout, strips)"
    elif layout is not None:
        version += " (wall layout)"
    tool_texts = dict(h.tb.cfg["robot"]["tool_texts"])
    if h.task == 3 and h.tb.phase1_view == "obs_pose":
        # obs_pose phase 1: plan_scans is called at the observation pose only (the registry text also allows the start
        # pose, for the 4-box scenario, whose tools text stays that of the g1v2-sub4-run1 tag)
        if "plan_scans" in tool_texts:
            raise ValueError("the robot variant replaces the plan_scans text: the obs_pose text cannot be applied")
        tool_texts["plan_scans"] = PLAN_SCANS_FROM_OBS + (PLAN_SCANS_STRIP_BODY if strips else PLAN_SCANS_LAYOUT_BODY
                                                          if layout is not None else PLAN_SCANS_BODY)
    if layout is not None:
        for name, note_ in LAYOUT_TOOL_NOTES.items():
            tool_texts[name] = tool_texts.get(name, TOOLS[name][1]) + note_
    if strips:
        for name, note_ in STRIP_TOOL_NOTES.items():
            tool_texts[name] = tool_texts.get(name, TOOLS[name][1]) + note_
    names = task_tools(h.task, h.tb)              # the Harness's list: layout-only tools in a layout episode only
    tools = openai_tools(names, tool_texts)
    if "move_arm" in names:
        # move_arm("home") needs the episode's initial arm pose (the rule orchestrators record it the same way); without
        # it every home move fails with "home pose unknown", and the task-3 prompt asks for home before each walk
        h.tb.remember_home()
    _, img = h.tb.look_head()
    task_text = _start_text(h.task, selector, getattr(h.tb, "multi_face", False), getattr(h.tb, "scan_targets", None),
                            h.tb.phase1_view if h.task == 3 else None, lean, layout)
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
