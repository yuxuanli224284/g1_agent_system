"""Evaluator - the only module that reads ground truth. Runs after an episode on the
harness's logged estimates (odometry frame) + the simulator's evaluation channel (GT, actual state, stage states,
scan log)."""
from __future__ import annotations

import math
import os

import numpy as np
import yaml

from ..common import config
from ..common.transforms import angle_between, apply, wrap
from ..perception import geometry as geo
from .scan_order import (FACE_TOUR, ORDER_RULES, ORDER_UNITS, adaptive_standoff, hfov_from_intrinsics, labelled_faces,
                         order_boxes_in_frame, order_labels_in_frame, yaw_only)


def _world_pose2(T_world_odom, pos, yaw):
    p = apply(T_world_odom, [pos[0], pos[1], 0.0])
    return p[:2], wrap(yaw + math.atan2(T_world_odom[1, 0], T_world_odom[0, 0]))


def _task1(gt: dict, actual: dict, ws: dict, E: dict) -> dict:
    """Task-1 formula: root distance to the GT target face plane vs the harness's d_obs, yaw facing the face, all GT face
    corners in the head image. E: eval tolerances (task1.yaml / task3.yaml eval)."""
    Two = np.asarray(actual["T_world_odom_now"], float)
    Twr = np.asarray(actual["T_world_root"], float)
    f = gt["target_face"]
    n = np.asarray(f["normal"], float)
    d_act = geo.horizontal_distance_to_plane(Twr[:3, 3], f["center"], n)
    nh = np.array([n[0], n[1], 0.0]); nh /= np.linalg.norm(nh)
    yaw_face = math.atan2(-nh[1], -nh[0])
    yaw_err = abs(math.degrees(wrap(math.atan2(Twr[1, 0], Twr[0, 0]) - yaw_face)))
    K = np.asarray(actual["head_K"], float); H, W = actual["head_size"]
    T_cam_world = np.linalg.inv(np.asarray(actual["T_world_headcam"], float))
    uv, z = geo.project(K, T_cam_world, f["corners"])
    in_view = bool(np.all(geo.in_image(uv, z, W, H, 0.0)))
    rt = ws.get("root_target")
    out = {"d_actual_m": d_act, "yaw_err_deg": yaw_err, "face_in_view_gt": in_view,
           "gt_obs_pose": gt["task1_gt_obs_pose"]}
    if rt is None:
        out.update(success=False, failure="no root target estimated")
        return out
    out["d_obs_harness"] = rt["d_obs"]
    out["d_err_m"] = abs(d_act - rt["d_obs"])
    red_p, red_yaw = _world_pose2(Two, rt["pos"], rt["yaw"])
    g = gt["task1_gt_obs_pose"]
    out["perception_err_pos_m"] = float(np.linalg.norm(red_p - np.asarray(g["pos"][:2])))
    out["perception_err_yaw_deg"] = abs(math.degrees(wrap(red_yaw - g["yaw"])))
    out["execution_err_pos_m"] = float(np.linalg.norm(Twr[:2, 3] - red_p))
    out["execution_err_yaw_deg"] = abs(math.degrees(wrap(math.atan2(Twr[1, 0], Twr[0, 0]) - red_yaw)))
    out["success"] = bool(out["d_err_m"] < E["dist_tol_m"] and yaw_err < E["yaw_tol_deg"] and in_view)
    if not out["success"]:
        out["failure"] = (f"task-1 pose not reached (d_err {out['d_err_m']:.3f} m, yaw_err {yaw_err:.1f} deg, "
                          f"face in view {in_view})")
    return out


def wrist_metrics(T_world_tip, center, normal, offset_m: float) -> dict:
    """Scanner tip (4x4, +x = scan direction) vs a label: distance along the normal, offset from the label axis, angle
    between the scan direction and -normal, and the 3-D error to the target point center + offset * normal."""
    T = np.asarray(T_world_tip, float)
    c, n = np.asarray(center, float), np.asarray(normal, float)
    p, fwd = T[:3, 3], T[:3, 0]
    v = p - c
    dist = float(v @ n)
    return {"dist_to_plane_m": dist, "axis_offset_m": float(np.linalg.norm(v - dist * n)),
            "angle_err_deg": math.degrees(angle_between(fwd, -n)),
            "tip_err_m": float(np.linalg.norm(p - (c + offset_m * n))), "target_offset_m": float(offset_m)}


def _task2(gt: dict, actual: dict, ws: dict) -> dict:
    E = config.load("task2")["eval"]
    Two = np.asarray(actual["T_world_odom_now"], float)
    bc = next(b for b in gt["barcodes"] if b["content"] == gt["target_barcode"])
    wm = wrist_metrics(actual["T_world_wrist_point"], bc["center"], bc["normal"], config.load("task2")["offset_m"])
    p = np.asarray(actual["T_world_wrist_point"], float)[:3, 3]
    out = {"wrist_dist_to_plane_m": wm["dist_to_plane_m"], "wrist_axis_offset_m": wm["axis_offset_m"],
           "wrist_angle_err_deg": wm["angle_err_deg"], "gt_wrist_target": gt["task2_gt_wrist_target"]["pos"]}
    green = np.asarray(gt["task2_gt_wrist_target"]["pos"], float)
    for key, name in (("wrist_target", "head_perception_err_m"), ("wrist_target_refined", "wrist_perception_err_m")):
        t = ws.get(key)
        out[name] = None if t is None else float(np.linalg.norm(apply(Two, t["pos"]) - green))
    final = ws.get("wrist_target_refined") or ws.get("wrist_target")
    out["execution_err_m"] = None if final is None else float(np.linalg.norm(p - apply(Two, final["pos"])))
    out["success"] = bool(abs(wm["dist_to_plane_m"] - config.load("task2")["offset_m"]) <= E["dist_tol_m"]
                          and out["wrist_axis_offset_m"] < E["axis_offset_max_m"] and out["wrist_angle_err_deg"] < E["angle_max_deg"])
    return out


def scanner_hfov_gt() -> float:
    """Horizontal field of view of the simulated scanner imager, from the simulator's camera definition (GT side):
    hfov = 2 atan(horizontal_aperture / 2 / focal_length) (32.8 deg for the DS2208-like imager)."""
    p = os.path.join(config.load("system")["sim_server"]["sim_root"], "configs", "scanner.yaml")
    with open(p) as f:
        cam = yaml.safe_load(f)["camera"]
    fx = cam["focal_length"] / cam["horizontal_aperture"] * cam["width"]
    return hfov_from_intrinsics(fx, cam["width"])


def gt_symbol_size(bc: dict):
    """GT size the stand-off rule is defined on: the drawn symbol (pattern_size, without the white backing; see
    configs/task3.yaml standoff). A layout that does not record it cannot be scored with the adaptive rule."""
    if bc.get("pattern_size") is None:
        raise ValueError(f"GT barcode {bc.get('content')!r} has no pattern_size: the adaptive stand-off is defined on the symbol")
    return bc["pattern_size"]


def gt_standoff(symbol_size, hfov_rad: float) -> float:
    """GT adaptive stand-off for a barcode: the design formula (configs/task3.yaml standoff) with the GT symbol size
    (gt_symbol_size). The longer side is used as the width, as on the algorithm side: the scanner's roll is free, so
    the longer extent is what has to fit the image width."""
    S = config.load("task3")["standoff"]
    return adaptive_standoff(float(max(symbol_size[0], symbol_size[1])), hfov_rad, S["fill_frac"], S["min_m"], S["max_m"])


def _decoded_contents(decoded) -> list:
    return [d["content"] if isinstance(d, dict) else d for d in (decoded or [])]


def _task3(gt: dict, actual: dict, ws: dict, stages: dict, scan_log: list, target_boxes=None, order_rule="columns",
           order_unit="label") -> dict:
    """Merged task (find the boxes, then scan every GT barcode once, left to right then top to bottom).
    Subset task (target_boxes = the scenario's GT box ids, evaluation only; order_rule = the scenario's scan_targets
    order, the rule the agent was told): the expected set is the barcodes of those boxes, ordered by order_rule;
    reading another GT label is reported (non_target_reads) and is never a failure (user decision 2026-11-19); the
    task-1 metrics are not applicable (the robot works from near its start, no observation pose of the whole wall).
    - task-1 metrics: the task-1 formula on the actual state stored at the "task1_end" stage;
    - expected set: every non-distractor GT barcode, ordered by the column rule in the root frame at task1_end
      (the robot start pose if the run never reached task1_end; `order_frame` says which);
    - reads come from the simulator's scan log (every scan() call, with the GT scan-ray hit and the tip pose);
      a content counts once, at its first read;
    - per read: scanner-tip error vs the GT adaptive target center + d_gt * normal (d_gt from the GT label width);
    - several labelled sides (gt "faces", I2): the expected order is the four-face tour (scan_order.FACE_TOUR), each
      face ordered in its own ideal frame at its GT obs pose (`order_frame` = "faces"), and success instead needs
      every expected barcode read, each face in its own order, the front (first labelled tour face) first, the faces
      contiguous and no foreign content (_tour_metrics; face_order_ok is reported, not gating);
    - success = every expected barcode read, in the expected order, and no foreign content read. The task-1 phase
      (arrival accuracy) is reported separately in "task1" / "t1_success" and does not gate success: the task-1 phase
      ends early by design (fast end) and the scan phase plans its own stances (user decision 2026-11-06).
    - order_unit "box" (scenarios with a wall layout: the agent plans one entry per layout cell): the expected order
      is the rule applied to the GT boxes of the expected labels (scan_order.order_boxes_in_frame, in the yaw-only
      frame of the reference root pose; boxes grouped by centre containment), labels of one box by the label rule;
      output key "order_unit" only then. A plan entry recorded "no_label" (search_cell saw the whole box face and
      found no label) is listed in "reported_no_label"; it never excuses an expected label: an unread GT label still
      fails the scan count."""
    out = {}
    if order_rule not in ORDER_RULES:
        raise ValueError(f"unknown order rule {order_rule!r} (known: {ORDER_RULES})")
    if order_unit not in ORDER_UNITS:
        raise ValueError(f"unknown order unit {order_unit!r} (known: {ORDER_UNITS})")
    t1 = stages.get("task1_end")
    if t1 is None:
        out["task1"] = {"success": False, "failure": "the run never reached the task1_end stage"}
        T_ref = None
    elif target_boxes is not None:
        out["task1"] = {"success": None, "note": "n/a (subset task)",
                        "t_sim": t1.get("t_sim")}
        T_ref = np.asarray(t1["actual"]["T_world_root"], float)
    else:
        # the harness root target as it was at task1_end (the runner snapshots it when the stage fires); the final
        # ws root_target may have been recomputed during the scan phase
        out["task1"] = _task1(gt, t1["actual"], {"root_target": ws["task1_end_root_target"]}, config.load("task3")["eval"])
        out["task1"]["t_sim"] = t1.get("t_sim")
        T_ref = np.asarray(t1["actual"]["T_world_root"], float)
    for k in ("success", "d_err_m", "yaw_err_deg", "face_in_view_gt", "perception_err_pos_m", "execution_err_pos_m"):
        out[f"t1_{k}"] = out["task1"].get(k)
    if T_ref is None:
        ri = gt["robot_init"]
        c, s = math.cos(ri["yaw"]), math.sin(ri["yaw"])
        T_ref = np.array([[c, -s, 0, ri["pos"][0]], [s, c, 0, ri["pos"][1]], [0, 0, 1, 0], [0, 0, 0, 1]], float)
        out["order_frame"] = "robot_init"
    else:
        out["order_frame"] = "task1_end"

    all_gt = [b for b in gt["barcodes"] if not b["distractor"]]
    if len({b["content"] for b in all_gt}) != len(all_gt):
        raise ValueError("GT barcode contents are not unique; content-based scan evaluation is undefined")
    exp = all_gt
    if target_boxes is not None:
        tb = set(target_boxes)
        exp = [b for b in all_gt if b["box_id"] in tb]
        missing = tb - {b["box_id"] for b in exp}
        if missing:
            raise ValueError(f"target boxes without a GT barcode: {sorted(missing)}")
        out["target_boxes"] = list(target_boxes)
    out["order_rule"] = order_rule
    by_content = {b["content"]: b for b in exp}
    other_gt = {b["content"] for b in all_gt} - set(by_content)
    labels = [{"key": b["content"], "center": b["center"], "normal": b["normal"], "up": b["up"], "size": b["size"]}
              for b in exp]
    # several labelled sides of a free-standing stack (I2 "faces"): the four-face tour, each face in its own ideal frame;
    # otherwise (one labelled face, or a legacy GT without "faces") the single-face rule in the task1_end root frame
    groups = labelled_faces(labels, gt["faces"], order_rule) if gt.get("faces") else []
    multi = len(groups) > 1
    if multi and order_unit == "box":
        raise ValueError("order unit 'box' is defined for one labelled wall face, the GT has several labelled faces")
    if multi:
        order = [k for _, keys in groups for k in keys]
        out["order_frame"] = "faces"
    elif order_unit == "box":
        boxes = {b["id"]: b for b in gt["boxes"]}
        order = order_boxes_in_frame([dict(lb, box_id=b["box_id"]) for lb, b in zip(labels, exp)], boxes,
                                     yaw_only(T_ref), order_rule)
        out["order_unit"] = "box"
    else:
        order = order_labels_in_frame(labels, T_ref, order_rule)
    hfov = scanner_hfov_gt()
    E2 = config.load("task2")["eval"]
    out["gt_order"] = order
    out["gt_standoff_m"] = {c: gt_standoff(gt_symbol_size(by_content[c]), hfov) for c in order}
    out["scanner_hfov_deg"] = math.degrees(hfov)

    reads, first, seen = [], [], set()
    for i, e in enumerate(scan_log):
        content = e.get("content")
        hit = (e.get("gt") or {}).get("hit")
        aimed = hit["content"] if hit else None
        r = {"i": i, "t_sim": e.get("t_sim"), "content": content, "decoded": _decoded_contents(e.get("decoded")),
             "aimed_content": aimed, "new": content is not None and content not in seen, "in_gt": content in by_content}
        ref = content if content in by_content else (aimed if aimed in by_content else None)
        if ref is not None:
            b = by_content[ref]
            wm = wrist_metrics(e["T_world_tip"], b["center"], b["normal"], out["gt_standoff_m"][ref])
            wm["placement_ok"] = bool(abs(wm["dist_to_plane_m"] - wm["target_offset_m"]) <= E2["dist_tol_m"]
                                      and wm["axis_offset_m"] < E2["axis_offset_max_m"] and wm["angle_err_deg"] < E2["angle_max_deg"])
            r.update(ref_content=ref, ref_source="decoded" if ref == content else "scan_ray", **wm)
        reads.append(r)
        if r["new"]:
            seen.add(content)
            first.append(r)

    distinct = [r["content"] for r in first]
    scanned_gt = [c for c in distinct if c in by_content]
    non_target = [c for c in distinct if c in other_gt]           # other GT labels of a subset task: reported only
    wrong = [c for c in distinct if c not in by_content and c not in other_gt]
    n_ok_reads = sum(r["content"] is not None for r in reads)
    tip = [r["tip_err_m"] for r in first if r["in_gt"]]
    out.update({
        "n_gt_barcodes": len(order), "n_scan_calls": len(reads), "n_no_read": len(reads) - n_ok_reads,
        "n_scanned_distinct": len(scanned_gt), "scan_recall": len(scanned_gt) / len(order) if order else None,
        "wrong_contents": wrong, "n_wrong_contents": len(wrong), "n_duplicate_reads": n_ok_reads - len(distinct),
        "non_target_reads": non_target, "n_non_target_reads": len(non_target),
        "n_read_not_aimed": sum(r["content"] is not None and r["aimed_content"] is not None and r["content"] != r["aimed_content"]
                                for r in reads),
        "scan_order": scanned_gt,
        "order_ok": scanned_gt == order,
        "order_ok_scanned": scanned_gt == [c for c in order if c in set(scanned_gt)],
        "tip_err_mean_m": float(np.mean(tip)) if tip else None, "tip_err_max_m": float(np.max(tip)) if tip else None,
        "n_placement_ok": sum(bool(r.get("placement_ok")) for r in first if r["in_gt"]),
        "reads": reads,
    })
    plan = ws.get("scan_plan") or []
    ledger = [s["content"] for s in (ws.get("scanned") or [])]
    out.update({"n_planned": len(plan),
                "plan_status": {st: sum(p["status"] == st for p in plan) for st in sorted({p["status"] for p in plan})},
                "reported_unreachable": [{"id": p["id"], "center_odom": p.get("center_odom"), "reason": p.get("fail_reason")}
                                         for p in plan if p["status"] == "unreachable"],
                "ledger_contents": ledger, "ledger_matches_log": ledger == distinct})
    if any(p["status"] == "no_label" for p in plan):
        out["reported_no_label"] = [{"id": p["id"], "box_id": p.get("box_id"), "cell": p.get("cell"),
                                     "center_odom": p.get("center_odom"), "reason": p.get("fail_reason")}
                                    for p in plan if p["status"] == "no_label"]
    fails = []
    if len(scanned_gt) < len(order):
        fails.append(f"scanned {len(scanned_gt)}/{len(order)} GT barcodes")
    elif not multi and not out["order_ok"]:
        fails.append(f"scan order {scanned_gt} != GT order {order}")
    if multi:
        out.update(_tour_metrics(groups, scanned_gt))
        if len(scanned_gt) == len(order):
            fails += [f"face {pf['face']} order {pf['read_order']} != GT {pf['gt_order']}" for pf in out["per_face"]
                      if not pf["order_ok"]]
        if not out["front_first"]:
            fails.append(f"first face read {out['face_order'][:1]} is not the first labelled tour face {groups[0][0]}")
        if not out["faces_contiguous"]:
            fails.append(f"faces not scanned contiguously: {out['face_order']}")
    if wrong:
        fails.append(f"read foreign contents {wrong}")
    out["success"] = not fails
    if fails:
        out["failure"] = "; ".join(fails)
    return out


def _tour_metrics(groups: list, scanned_gt: list) -> dict:
    """Multi-face task-3 scoring. groups: [(face, [GT keys in that face's order])] of the labelled faces in tour order
    (scan_order.labelled_faces); scanned_gt: the GT contents in the order of their first reads.
    - per_face: GT vs read order of each labelled face (order_ok = all of it read, in its own column order);
    - face_order: the faces of the successive first reads with repeats collapsed (the actual tour);
    - faces_contiguous: no face appears twice in face_order (a face is finished before the next one starts);
    - front_first: the first face read is the first labelled face of the tour (-x, the front, when it carries labels);
    - face_order_ok: face_order equals the tour over every labelled face (reported, not gating success)."""
    face_of = {k: f for f, keys in groups for k in keys}
    per_face = []
    for f, keys in groups:
        ro = [c for c in scanned_gt if face_of[c] == f]
        per_face.append({"face": f, "face_idx": FACE_TOUR.index(f), "n_gt": len(keys), "n_read": len(ro),
                         "recall": len(ro) / len(keys), "gt_order": keys, "read_order": ro, "order_ok": ro == keys})
    seq = [face_of[c] for c in scanned_gt]
    face_order = [f for i, f in enumerate(seq) if i == 0 or f != seq[i - 1]]
    return {"per_face": per_face, "face_order": face_order,
            "faces_contiguous": len(face_order) == len(set(face_order)),
            "front_first": bool(face_order) and face_order[0] == groups[0][0],
            "face_order_ok": face_order == [f for f, _ in groups]}


def evaluate(task: int, gt: dict, actual: dict, ws: dict, stages: dict | None = None, scan_log: list | None = None,
             target_boxes: list | None = None, order_rule: str = "columns", order_unit: str = "label") -> dict:
    """ws: harness world-state export {"root_target", "wrist_target", "wrist_target_refined"} (odom frame); task 3 also
    needs "scan_plan" and "scanned" (I4), "task1_end_root_target" (the harness root target snapshotted when the
    task1_end stage fired; required when that stage exists), the eval-port stage states (get_stage_states) and scan
    log (get_scan_log). Task 3 subset (scenario target_boxes / scan_targets.order; order_unit "box" for scenarios
    with a wall layout): see _task3."""
    if task == 1:
        return _task1(gt, actual, ws, config.load("task1")["eval"])
    if task == 2:
        return _task2(gt, actual, ws)
    if task == 3:
        if stages is None or scan_log is None:
            raise ValueError("task 3 evaluation needs the stage states and the scan log")
        return _task3(gt, actual, ws, stages, scan_log, target_boxes, order_rule, order_unit)
    raise ValueError(f"unknown task {task!r}")
