"""Tool registry: pydantic argument models -> OpenAI function-calling schemas. The rule
orchestrators and the LLM orchestrators call tools through the same Harness.call().

The descriptions below are those of the robot variant g1_v1 (the scanner imager is the wrist camera); a variant whose
hand camera is a separate device (g1_v2: the D405 on the left-wrist bracket) replaces some of them with its own texts
(robot.yaml variants.<v>.tool_texts, passed to openai_tools by the LLM orchestrator)."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class NoArgs(BaseModel):
    pass


class DetectBoxesArgs(BaseModel):
    prompt: str = Field("cardboard box", description="what to detect (open-vocabulary)")


class SelectArgs(BaseModel):
    selector: str = Field(..., description='"left" or "right" (robot\'s view), "all" (every detected box; task 3) or a box id from detect_boxes')


class BoxArgs(BaseModel):
    box_id: Optional[str] = Field(None, description='box id or "all"; default: the selected target')


class ApproachArgs(BaseModel):
    target: Literal["root_target", "stance_target"] = Field("root_target", description="root_target (task 1) or stance_target (task 2)")


class ExploreArgs(BaseModel):
    direction: Literal["left", "right", "back", "forward", "turn_left", "turn_right"]
    distance: float = Field(..., description="metres (<= 0.5) for left/right/back/forward, degrees (<= 15) for turns")


class BarcodeArgs(BaseModel):
    barcode_id: Optional[str] = Field(None, description="barcode id; default: the current barcode (task 3: the one chosen by next_barcode)")


class WristTargetArgs(BaseModel):
    barcode_id: Optional[str] = None
    offset: Optional[float] = Field(None, description="distance of the wrist point in front of the barcode (m); default: the configured stand-off (task 2: 0.20 m; task 3: adaptive from the estimated label width)")


class MoveArmArgs(BaseModel):
    pose: Literal["pre", "final", "home"] = Field(..., description="pre: the wrist target's stand-off + 0.10 m in front of the barcode; final: the stand-off (refined target if available); home: default arm pose (before every base move)")


class BaseHeightArgs(BaseModel):
    height: Optional[float] = Field(None, description="commanded pelvis height (m): plan_stance's base_height to squat; omit to stand up")


class FinishArgs(BaseModel):
    task: int = Field(..., description="the episode's task id: 1, 2 or 3")
    success: bool
    summary: str


# plan_scans text in two parts: where it is called from, and what it does. The obs_pose phase 1 of a target-spec task
# (Toolbox.set_phase1_view) replaces the first part with PLAN_SCANS_FROM_OBS (orchestrators.agent_full); every
# other scenario keeps the text of the 4-box tag g1v2-sub4-run1 byte for byte.
PLAN_SCANS_FROM_START = ("Task 3, after reaching the observation pose (or, when the task names the labels by their "
                         "place on the wall, from near the start pose): ")
PLAN_SCANS_FROM_OBS = "Task 3, after reaching the observation pose: "
PLAN_SCANS_BODY = ("localise every barcode label on every visible box face in the head image and order them (columns left to right, top to bottom inside a column; or rows top to bottom, left to right inside a row when the task says so). With a target spec only the labels in the requested cells are planned (the others are listed as not_targeted). The order is fixed; next_barcode follows it. Called again while the plan runs (e.g. after a barcode, arm at home, before standing up), it only ADDS labels seen for the first time from the current view.")

# plan_scans body of a task with a wall layout (Toolbox.set_wall_layout; orchestrators.agent_full applies it)
PLAN_SCANS_LAYOUT_BODY = ("register the wall seen from here to the given layout (front plane, both side edges, floor) and "
                          "plan one entry of kind cell per requested box, in the order of the task's rule; fails, naming "
                          "the reason, when a side edge is not in view or the wall does not match the layout; a later "
                          "call changes nothing.")
SEARCH_CELL_TEXT = ("Task 3 with a wall layout: for the current box (an entry of kind cell), stand at the planned stance "
                    "with the arm home, then: re-register the wall from the head, move the hand camera (D405) to the view "
                    "of the whole box face, find the label inside that face and make the entry that label (its wrist "
                    "target is computed); a second label in the same box is added as the next entry. When the "
                    "label's scan pose is not reachable from this stance the reply says so: the label then gets its own "
                    "scan stance (compute_wrist_target -> plan_stance -> approach -> move_arm('pre') ...). Records "
                    "no_label when the whole face was seen and holds none.")

TOOLS = {
    "look_head": (NoArgs, "Capture the head camera image (attached to the reply) and a short state summary."),
    "detect_boxes": (DetectBoxesArgs, "Detect boxes in the head image; ids are ordered left to right (robot's left first)."),
    "select_target_box": (SelectArgs, "Choose the target box by 'left'/'right'/'all' or by id ('all' = every detected box, task 3)."),
    "fit_front_face": (BoxArgs, "Fit the front face plane of a box from the depth image (centre, normal, corners in the odometry frame)."),
    "compute_obs_pose": (BoxArgs, "Compute the observation distance d_obs (the nearest distance of the task's search range with the whole face(s) in view with a margin) and the root target."),
    "approach": (ApproachArgs, "Walk to the root target in segments, re-perceiving after each segment; ends with an arrival check (distance along the face normal, yaw, faces in view). With stance_target: walk to the stance (standing up first if squatting; the arm must be at home), then squat to the stance's planned body height; no arrival check."),
    "check_face_in_view": (BoxArgs, "Check that the estimated face corners project inside the head image with a margin."),
    "explore": (ExploreArgs, "Constrained exploration move (<= 0.5 m or <= 15 deg)."),
    "detect_barcode_head": (BoxArgs, "Detect the barcode on the target face in the head image."),
    "estimate_barcode_pose": (BarcodeArgs, "Barcode centre = pixel ray x face plane; normal = face normal."),
    "compute_wrist_target": (WristTargetArgs, "Wrist target = barcode centre + offset * normal, forward axis = -normal (offset default: the configured stand-off)."),
    "plan_stance": (NoArgs, "Search a reachable root stance around the current barcode (distance from the face and lateral offset relative to the barcode, and the body height: standing first, then the measured squat levels) for the wrist target (pre-pose and final pose; knees / body clear of the boxes, arm clear of the legs); stores it as stance_target. data.stance_odom.base_height < standing = squat after walking."),
    "set_base_height": (BaseHeightArgs, "Squat to / stand up at a pelvis height, standing still (about 3 s). Normally not needed: approach(stance_target) squats to the planned stance height on arrival and every walking tool stands the robot up first (the arm must be at home)."),
    "move_arm": (MoveArmArgs, "Move the left arm (scanner) with IK; closed-loop correction of the arm sag."),
    "look_wrist": (NoArgs, "Capture the wrist (scanner) camera image (attached to the reply)."),
    "refine_with_wrist": (BarcodeArgs, "Re-detect the barcode in the wrist image and intersect with the re-fitted face plane; updates the refined wrist target."),
    "stereo_check_wrist": (BarcodeArgs, "Shift the wrist 4 cm, triangulate the barcode centre from two wrist views and compare with the refined estimate."),
    "check_wrist_placement": (NoArgs, "Check the wrist point: at the wrist target's stand-off +- 0.02 m from the barcode plane, barcode centre within 2 cm of the axis, angle < 10 deg."),
    "plan_scans": (NoArgs, PLAN_SCANS_FROM_START + PLAN_SCANS_BODY),
    "go_to_next_face": (NoArgs, "Task 3 with every side of the stack: walk around the stack to the next side of the tour (front -> right -> back -> left) and stand at its observation pose (arm home, stand up, back off, sidestep past the corner, turn, fit the side, step to its observation pose). Call it when next_barcode returns face_done=true, then plan_scans."),
    "next_barcode": (NoArgs, "Task 3: switch to the next barcode of the scan plan (an unfinished current barcode is marked failed); clears the per-barcode wrist / stance targets. Works on the current side of the stack: when that side is finished and the task covers every side, it returns face_done=true (call go_to_next_face, then plan_scans); done=true when everything has been handled."),
    "search_cell": (BarcodeArgs, SEARCH_CELL_TEXT),
    "scan": (NoArgs, "Trigger the scanner and decode what it sees. The code nearest the image centre counts for the current barcode; a content read before is a duplicate and is not counted again. On no read the tool itself retries once (refine_with_wrist + move_arm final + scan); data.retries records it."),
    "finish": (FinishArgs, "End the task and report success or failure with a one-sentence summary."),
}


# Tools each task may use. 2026-11-06: with every tool exposed, the
# agent_full LLM continued a task-1 episode into the barcode / arm tools and failed it. Task 3 = task 1 + scanning every
# barcode: all tools, including the scan-plan tools.
TASK1_TOOLS = ["look_head", "detect_boxes", "select_target_box", "fit_front_face", "compute_obs_pose", "approach",
               "check_face_in_view", "explore", "finish"]
TASK3_ONLY = ["plan_scans", "next_barcode", "go_to_next_face"]
# tools of a task-3 episode with a wall layout only (Toolbox.wall_layout): every other episode's tool list and texts stay
# as they were (task_tools)
LAYOUT_ONLY = ["search_cell"]
TASK2_TOOLS = [n for n in TOOLS if n not in TASK3_ONLY + LAYOUT_ONLY]
TASK_TOOLS = {1: TASK1_TOOLS, 2: TASK2_TOOLS, 3: [n for n in TOOLS if n not in LAYOUT_ONLY]}


def task_tools(task: int, toolbox) -> list:
    """The tools of an episode: the task's tools, plus the layout-only tools in a task-3 episode with a wall layout."""
    return TASK_TOOLS[task] + (LAYOUT_ONLY if task == 3 and getattr(toolbox, "wall_layout", None) is not None else [])


def openai_tools(names=None, texts: dict | None = None) -> list[dict]:
    """texts: {tool name: description} replacing the registry text (robot.yaml variants.<v>.tool_texts)."""
    unknown = set(texts or {}) - set(TOOLS)
    if unknown:
        raise ValueError(f"tool texts for unknown tools {sorted(unknown)}")
    out = []
    for name, (model, desc) in TOOLS.items():
        if names is not None and name not in names:
            continue
        desc = (texts or {}).get(name, desc)
        schema = model.model_json_schema()
        schema.pop("title", None)
        for p in schema.get("properties", {}).values():
            p.pop("title", None)
        out.append({"type": "function", "function": {"name": name, "description": desc, "parameters": schema}})
    return out
