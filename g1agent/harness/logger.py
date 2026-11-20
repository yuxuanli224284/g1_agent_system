"""Episode logging: one directory per episode with tool_calls.jsonl, llm.jsonl, stages.jsonl and images/.

Every record carries t_wall (seconds since the logger was created), t_unix (absolute, at write time) and log_seq (write
order across the files). An optional
on_event callback receives a compact event per tool call / LLM exchange / stage (schema I1 of the task-3 design):
  {"kind": "llm", "t_unix", "purpose", "thought", "reasoning", "tool_calls": [{"name", "arguments"}], "latency_s"}
  {"kind": "tool", "t_unix", "name", "args", "status", "reason", "overlay": {...}}
  {"kind": "stage", "t_unix", "name", "info"}
The callback is how the runner feeds the episode video overlays; this module never talks to the simulator itself.
"""
from __future__ import annotations

import json
import os
import time

import cv2
import numpy as np


def _json(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(type(o).__name__)


def tool_overlay(name: str, result: dict) -> dict:
    """The small pixel-space overlay of a tool result: only detections that exist in the result data (I1/I3).
    Keys absent from the data are absent from the overlay (nothing is invented)."""
    d = result.get("data") or {}
    ov = {}
    if isinstance(d.get("boxes"), list) and all("bbox" in b for b in d["boxes"]):
        ov["head_boxes"] = [{"id": b.get("id"), "bbox": b["bbox"]} for b in d["boxes"]]
    if name == "select_target_box" and d.get("target_box_id") is not None:
        ov["target_box"] = d["target_box_id"]
    if isinstance(d.get("barcodes"), list):
        bcs = d["barcodes"]
        labels = [{"id": b.get("id"), "corners_px": b["corners_px"]} for b in bcs if b.get("corners_px") is not None]
        if labels:
            ov["head_labels"] = labels
        if any("status" in b for b in bcs):
            ov["scan_plan"] = [{"id": b.get("id"), "status": b.get("status"), "content": b.get("content")} for b in bcs]
    if isinstance(d.get("scan_plan"), list):
        ov["scan_plan"] = [{"id": b.get("id"), "status": b.get("status"), "content": b.get("content")} for b in d["scan_plan"]]
    if isinstance(d.get("polygons"), list):
        ov["scanner_polygons"] = [{"content": p.get("content"), "polygon": p["polygon"]} for p in d["polygons"] if p.get("polygon") is not None]
    if isinstance(d.get("scanned"), list):
        ov["scanned"] = [s["content"] if isinstance(s, dict) else s for s in d["scanned"]]
    if name == "detect_barcode_head" and d.get("corners_px") is not None:     # task 2: one label, corners at the top level
        ov["head_labels"] = [{"id": d.get("barcode_id"), "corners_px": d["corners_px"]}]
    if name == "next_barcode":
        bid = d.get("barcode_id", d.get("id"))
        if bid is not None or d.get("done"):
            ov["current_barcode"] = bid
        if d.get("box_id") is not None:                                          # task 3 has no select_target_box
            ov["target_box"] = d["box_id"]
    return ov


class EpisodeLogger:
    def __init__(self, ep_dir: str, on_event=None):
        self.dir = ep_dir
        os.makedirs(os.path.join(ep_dir, "images"), exist_ok=True)
        self.t0 = time.time()
        self.n_img = 0
        self.on_event = on_event
        self.n_rec = 0                           # write order across all jsonl files (ties in time are ordered by it)

    def _append(self, name, rec):
        now = time.time()
        rec = dict(rec, t_wall=round(now - self.t0, 3), t_unix=round(now, 3), log_seq=self.n_rec)
        self.n_rec += 1
        with open(os.path.join(self.dir, name), "a") as f:
            f.write(json.dumps(rec, default=_json, ensure_ascii=False) + "\n")
        return rec

    def _emit(self, event):
        if self.on_event is not None:
            self.on_event(event)

    def tool(self, rec):
        rec = self._append("tool_calls.jsonl", rec)
        res = rec.get("result") or {}
        self._emit({"kind": "tool", "t_unix": rec["t_unix"], "name": rec.get("tool"), "args": rec.get("args"),
                    "status": res.get("status"), "reason": res.get("reason"), "overlay": tool_overlay(rec.get("tool"), res)})

    def llm(self, rec):
        rec = self._append("llm.jsonl", rec)
        rep = rec.get("reply")
        if rep is None:                          # bookkeeping records (e.g. invalid decision arguments) carry no reply
            return
        lat = None if rec.get("t_start") is None or rec.get("t_end") is None else round(rec["t_end"] - rec["t_start"], 3)
        self._emit({"kind": "llm", "t_unix": rec["t_unix"], "purpose": rec.get("kind"), "thought": rep.get("content") or "",
                    "reasoning": rep.get("reasoning_content"),
                    "tool_calls": [{"name": tc.get("name"), "arguments": tc.get("arguments")} for tc in rep.get("tool_calls") or []],
                    "latency_s": lat})

    def stage(self, name: str, info: dict | None = None):
        """Mark an orchestration stage (e.g. task1_end, barcode_start, barcode_end, scan_done)."""
        rec = self._append("stages.jsonl", {"stage": name, "info": info})
        self._emit({"kind": "stage", "t_unix": rec["t_unix"], "name": name, "info": info})

    def image(self, rgb, tag) -> str:
        self.n_img += 1
        p = os.path.join(self.dir, "images", f"{self.n_img:03d}_{tag}.jpg")
        cv2.imwrite(p, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        return os.path.relpath(p, self.dir)

    def write(self, name, obj):
        with open(os.path.join(self.dir, name), "w") as f:
            json.dump(obj, f, indent=1, default=_json, ensure_ascii=False)
