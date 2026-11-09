"""Tool registry: pydantic argument models -> OpenAI function-calling schemas. The rule
orchestrators and the LLM orchestrators call tools through the same Harness.call()."""
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


TOOLS = {
    "look_head": (NoArgs, "Capture the head camera image (attached to the reply) and a short state summary."),
    "detect_boxes": (DetectBoxesArgs, "Detect boxes in the head image; ids are ordered left to right (robot's left first)."),
    "select_target_box": (SelectArgs, "Choose the target box by 'left'/'right'/'all' or by id ('all' = every detected box, task 3)."),
    "fit_front_face": (BoxArgs, "Fit the front face plane of a box from the depth image (centre, normal, corners in the odometry frame)."),
    "compute_obs_pose": (BoxArgs, "Compute the observation distance d_obs (0.6-1.0 m, whole face in view with a margin) and the root target."),
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
    "plan_scans": (NoArgs, "Task 3, after reaching the observation pose: localise every barcode label on every visible box face in the head image and order them (columns left to right, top to bottom inside a column). The order is fixed; next_barcode follows it. Called again while the plan runs (e.g. after a barcode, arm at home, before standing up), it only ADDS labels seen for the first time from the current view."),
    "go_to_next_face": (NoArgs, "Task 3 with every side of the stack: walk around the stack to the next side of the tour (front -> right -> back -> left) and stand at its observation pose (arm home, stand up, back off, sidestep past the corner, turn, fit the side, step to its observation pose). Call it when next_barcode returns face_done=true, then plan_scans."),
    "next_barcode": (NoArgs, "Task 3: switch to the next barcode of the scan plan (an unfinished current barcode is marked failed); clears the per-barcode wrist / stance targets. Works on the current side of the stack: when that side is finished and the task covers every side, it returns face_done=true (call go_to_next_face, then plan_scans); done=true when everything has been handled."),
    "scan": (NoArgs, "Trigger the scanner and decode what it sees. The code nearest the image centre counts for the current barcode; a content read before is a duplicate and is not counted again. On no read the tool itself retries once (refine_with_wrist + move_arm final + scan); data.retries records it."),
    "finish": (FinishArgs, "End the task and report success or failure with a one-sentence summary."),
}


# Tools each task may use. 2026-11-06: with every tool exposed, the
# agent_full LLM continued a task-1 episode into the barcode / arm tools and failed it. Task 3 = task 1 + scanning every
# barcode: all tools, including the scan-plan tools.
TASK1_TOOLS = ["look_head", "detect_boxes", "select_target_box", "fit_front_face", "compute_obs_pose", "approach",
               "check_face_in_view", "explore", "finish"]
TASK3_ONLY = ["plan_scans", "next_barcode", "go_to_next_face"]
TASK2_TOOLS = [n for n in TOOLS if n not in TASK3_ONLY]
TASK_TOOLS = {1: TASK1_TOOLS, 2: TASK2_TOOLS, 3: list(TOOLS)}


def openai_tools(names=None) -> list[dict]:
    out = []
    for name, (model, desc) in TOOLS.items():
        if names is not None and name not in names:
            continue
        schema = model.model_json_schema()
        schema.pop("title", None)
        for p in schema.get("properties", {}).values():
            p.pop("title", None)
        out.append({"type": "function", "function": {"name": name, "description": desc, "parameters": schema}})
    return out
