"""World state kept by the harness. All estimates live in the odometry frame with the time
they were made.

Task 3 (scan every barcode) adds a scan plan: the ordered barcode ids made by Toolbox.plan_scans, the barcode being
worked on (current_barcode_id) and the ledger of first reads (`scanned`, one entry per distinct content). The wrist /
stance slots below always belong to the current barcode; Toolbox.next_barcode clears them when it switches."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

BARCODE_STATUSES = ("pending", "scanned", "duplicate", "failed")


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
    corners_odom: np.ndarray | None = None   # 4x3 label corners on the face plane (head estimate)
    corners_px: list | None = None  # 4x2 label corners in the head image the label was found in
    width_m: float | None = None    # estimated width on the plane (longer side, see scan_order.quad_width): head = the
                                    # label detection box (an upper bound of the symbol), wrist = the decoded symbol
    width_source: str = "head"      # head | wrist
    width_support: int = 0          # wrist width: number of decoded outlines (image rotations) it is the median of
    status: str = "pending"         # pending | scanned | duplicate | failed | unreachable
    fail_reason: str | None = None  # why it was not scanned (e.g. unreachable: label height vs the reachable range)
    wrist_read: str | None = None   # content the wrist camera decoded on this label during refine_with_wrist (readable)
    plan_center: np.ndarray | None = None   # centre when first planned (head); never moved by later refinements
    content: str | None = None      # decoded content once scanned (or the duplicate content)
    order_index: int | None = None  # position in the scan plan
    face_id: int = 0                # stack face it lies on: 0 front, 1 right, 2 back, 3 left (StackModel tour order)


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

    def scan_plan_export(self) -> list:
        r = lambda v: None if v is None else np.round(np.asarray(v, float), 3).tolist()
        out = []
        for bid in self.scan_plan:
            b = self.barcodes[bid]
            out.append({"id": bid, "box_id": b.box_id, "order_index": b.order_index, "center_odom": r(b.center),
                        "width_m": None if b.width_m is None else round(float(b.width_m), 4),
                        "width_source": b.width_source, "status": b.status, "fail_reason": b.fail_reason,
                        "content": b.content, "face_id": int(b.face_id)})
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
    if t is None:
        return None
    return {k: (r(v) if isinstance(v, (list, tuple, np.ndarray)) else v) for k, v in t.items()}
