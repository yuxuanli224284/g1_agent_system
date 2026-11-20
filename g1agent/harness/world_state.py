"""World state kept by the harness. All estimates live in the odometry frame with the time
they were made.

Task 3 (scan every barcode) adds a scan plan: the ordered barcode ids made by Toolbox.plan_scans, the barcode being
worked on (current_barcode_id) and the ledger of first reads (`scanned`, one entry per distinct content). The wrist /
stance slots below always belong to the current barcode; Toolbox.next_barcode clears them when it switches.
A task with a target spec (Toolbox.set_scan_targets: e.g. columns 1-2 from the left, rows 2-3 from the top) keeps every
label it localised in the plan, the ones outside the requested cells with status not_targeted (never scanned), and the
resolved label grid in target_grid."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

BARCODE_STATUSES = ("pending", "scanned", "duplicate", "failed", "unreachable", "unreadable", "not_targeted", "no_label")


@dataclass
class BoxEstimate:
    id: str
    bbox: list                      # last detection box (head image px)
    score: float
    t: float
    face: dict | None = None        # {"center", "normal", "right", "up", "corners" (4x3), "size"} odom frame


@dataclass
class BarcodeEstimate:
    id: str
    box_id: str
    center: np.ndarray              # odom
    normal: np.ndarray
    bar_dir: np.ndarray | None      # in-plane bar direction (odom), None if unknown
    source: str                     # head | wrist | stereo
    t: float
    pixel: list | None = None
    face: dict | None = None        # copy of the box face the label lies on (box ids are re-assigned by detect_boxes)
    corners_odom: np.ndarray | None = None   # 4x3 label corners on the face plane (head estimate: the label box)
    corners_px: list | None = None  # 4x2 label corners in the head image the label was found in
    width_m: float | None = None    # stand-off width on the plane (longer side, see scan_order.quad_width): head = the
                                    # cv2 symbol quad inside the label box when one was accepted, else the label box (an
                                    # upper bound of the symbol); wrist = the decoded symbol
    label_width_m: float | None = None   # width of the LABEL (the paper the detector found: head detection box on the
                                    # face plane), never replaced by a wrist symbol width; the extent every same-label
                                    # test and the wrist consistency gate use (corners_odom is that box)
    detection: dict | None = None   # head detection diagnostics (kind, refined_by_cv2, score, label / symbol corners px)
    width_source: str = "head"      # head | wrist | hand_label (wall layout: the hand-camera label box, an upper bound
                                    # of the symbol like the head width)
    width_support: int = 0          # wrist width: number of decoded outlines (image rotations) it is the median of
    status: str = "pending"         # pending | scanned | duplicate | failed | unreachable
    fail_reason: str | None = None  # why it was not scanned (e.g. unreachable: label height vs the reachable range)
    wrist_read: str | None = None   # content the wrist camera decoded on this label during refine_with_wrist (readable)
    plan_center: np.ndarray | None = None   # centre when first planned (head); never moved by later refinements
    content: str | None = None      # decoded content once scanned (or the duplicate content)
    order_index: int | None = None  # position in the scan plan
    face_id: int = 0                # stack face it lies on: 0 front, 1 right, 2 back, 3 left (StackModel tour order)
    grid_cell: dict | None = None   # target grid cell {"column", "row"} (1-based from the spec's edges; None = outside)
    kind: str = "label"             # label | cell (wall layout: one requested box whose label is not found yet; its centre
                                    # / face / corners are the box face, search_cell turns it into a label entry)
    wall: dict | None = None        # wall layout: the entry's geometry in the WALL frame (WorldState.wall): {"cell": [c, r]
                                    # (layout indices: column from the left, row from the bottom), "center", "normal",
                                    # "corners" (4x3 or None), "bar_dir" (or None), "left_to_right"}; the odom fields
                                    # above are re-expressed from it after every wall (re-)registration


@dataclass
class WorldState:
    boxes: dict = field(default_factory=dict)
    target_selector: str | None = None
    target_box_id: str | None = None
    barcodes: dict = field(default_factory=dict)
    root_target: dict | None = None          # {"pos" [x, y] odom, "yaw" odom, "d_obs", "box_id"}
    wrist_target: dict | None = None         # {"pos", "forward", "offset", "barcode_id"} odom (head estimate)
    wrist_target_refined: dict | None = None # {"pos", "forward", "offset", "barcode_id"} odom (wrist camera)
    stance_target: dict | None = None
    explore_steps: int = 0
    history: list = field(default_factory=list)
    scan_plan: list = field(default_factory=list)   # ordered barcode ids (plan_scans)
    scan_ref_T: np.ndarray | None = None     # T_odom_root of the head frame the (current face's) plan was ordered in
    scan_ref_Ts: dict = field(default_factory=dict)  # face id -> T_odom_root that face's plan was ordered in
    stack: dict | None = None                # multi-face StackModel: {"n0": front normal (odom, horizontal unit),
                                             #  "faces": {fid: {"normal", "corners", "center"}}, "current": fid,
                                             #  "visited": [fid], "multi_face": bool}; None before the first plan
    current_barcode_id: str | None = None
    scanned: list = field(default_factory=list)     # [{"content", "barcode_id", "order_index", "t"}], first read only
    wall: dict | None = None                 # wall layout (Toolbox.set_wall_layout): the registered wall frame
                                             #  {"T_odom_wall" (x = right along the wall, y = into the wall, z = up; origin
                                             #  = bottom-left corner of the front face as seen by the robot), "W", "H",
                                             #  "columns", "rows", "box_w", "box_h", "delta_lat", "registration", "history"}
    strips: dict | None = None               # wall layout worked in strips (Toolbox.strip_mode): the strip ledger
                                             #  {"width", "n", "current" (the strip the robot works on), "done_reported"
                                             #  (the strip next_barcode reported finished), "strips": {k: {"columns",
                                             #  "ref_w" (wall-frame x, y of the strip's reference stance), "ref_source"}},
                                             #  "moves": [go_to_next_strip records]}; None otherwise
    target_grid: dict | None = None          # with a target spec: {"targets" (scan_order.parse_targets), "edges_seen":
                                             #  [{"face_id", "edge", "pts" (odom), "clear"}], "face_centres":
                                             #  [{"face_id", "center" (odom)}], "fitted_faces": [{"face_id", "face":
                                             #  {"t", "corners" (odom), "edges_seen"}}], "faces": {fid: resolved grid}}

    def targeted(self, bid) -> bool:
        """A plan entry that is part of the task (everything but not_targeted)."""
        return self.barcodes[bid].status != "not_targeted"

    def missing_cells(self) -> list:
        """Requested target cells not found on any planned side ([] without a target spec)."""
        if self.target_grid is None:
            return []
        return [dict(c, face_id=fid) for fid, g in self.target_grid["faces"].items() for c in g["missing"]]

    def grid_warnings(self) -> list:
        """Target-grid counts that rest on the labels alone (wall edge not seen clear, a row / column of boxes without
        labels not excluded) or a label found later that the grid does not hold ([] without a target spec)."""
        if self.target_grid is None:
            return []
        return [f"side {fid}: {w}" for fid, g in self.target_grid["faces"].items() for w in g["warnings"]]

    def scan_plan_export(self) -> list:
        r = lambda v: None if v is None else np.round(np.asarray(v, float), 3).tolist()
        out = []
        for bid in self.scan_plan:
            b = self.barcodes[bid]
            out.append({"id": bid, "box_id": b.box_id, "order_index": b.order_index, "center_odom": r(b.center),
                        "width_m": None if b.width_m is None else round(float(b.width_m), 4),
                        "width_source": b.width_source, "status": b.status, "fail_reason": b.fail_reason,
                        "content": b.content, "face_id": int(b.face_id), "cell": b.grid_cell})
            if b.kind != "label":                    # only wall-layout entries carry it: other exports are unchanged
                out[-1]["kind"] = b.kind
        return out

    def summary(self) -> dict:
        r = lambda v: None if v is None else np.round(np.asarray(v, float), 3).tolist()
        return {"boxes": {k: {"bbox": [round(x) for x in b.bbox], "has_face": b.face is not None,
                              "face_center_odom": r(b.face["center"]) if b.face else None} for k, b in self.boxes.items()},
                "target_selector": self.target_selector, "target_box_id": self.target_box_id,
                "barcodes": {k: {"box": b.box_id, "center_odom": r(b.center), "source": b.source, "status": b.status,
                                 "content": b.content, "width_m": None if b.width_m is None else round(float(b.width_m), 4)}
                             for k, b in self.barcodes.items()},
                "root_target": None if self.root_target is None else {k: (r(v) if isinstance(v, (list, np.ndarray)) else v) for k, v in self.root_target.items()},
                "wrist_target": _target(self.wrist_target, r),
                "wrist_target_refined": _target(self.wrist_target_refined, r),
                "stance_target": None if self.stance_target is None else {k: (r(v) if isinstance(v, (list, np.ndarray)) else v) for k, v in self.stance_target.items()},
                "explore_steps": self.explore_steps,
                "scan_plan": self.scan_plan_export(),
                "scanned": [dict(s) for s in self.scanned],
                "current_barcode_id": self.current_barcode_id}


def _target(t, r):
    """Numeric sequences rounded by r; a sequence holding non-numbers (e.g. the look target's list of alternative poses,
    which are dicts) is walked element by element, so any target structure can be summarised."""
    if t is None:
        return None
    return {k: (_seq(v, r) if isinstance(v, (list, tuple, np.ndarray)) else v) for k, v in t.items()}


def _numeric(v) -> bool:
    if isinstance(v, np.ndarray):
        return v.dtype.kind in "biuf"
    if isinstance(v, (list, tuple)):
        return all(_numeric(x) for x in v)
    return isinstance(v, (int, float, np.number, np.bool_))


def _seq(v, r):
    if _numeric(v):
        return r(v)
    return [_seq(x, r) if isinstance(x, (list, tuple, np.ndarray)) else _target(x, r) if isinstance(x, dict) else x
            for x in v]
