"""Open-vocabulary box detection with GroundingDINO (transformers zero-shot object detection)."""
from __future__ import annotations

import numpy as np
import torch

from ..common import config


def _iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def _frac_inside(child, parent):
    x0, y0, x1, y1 = max(child[0], parent[0]), max(child[1], parent[1]), min(child[2], parent[2]), min(child[3], parent[3])
    return max(0.0, x1 - x0) * max(0.0, y1 - y0) / ((child[2] - child[0]) * (child[3] - child[1]) + 1e-9)


class BoxDetector:
    def __init__(self, device: str | None = None):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        self.cfg = config.load("perception")["box_detector"]
        self.device = device or config.load("system")["harness_client"]["detector_device"]
        self.proc = AutoProcessor.from_pretrained(self.cfg["model"])
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(self.cfg["model"]).to(self.device).eval()

    @torch.no_grad()
    def detect(self, rgb: np.ndarray, prompt: str | None = None) -> list[dict]:
        """Detections sorted left-to-right in the image: [{"bbox": [x0, y0, x1, y1], "score", "label"}]."""
        prompt = prompt or self.cfg["prompt"]
        inp = self.proc(images=rgb, text=prompt, return_tensors="pt").to(self.device)
        out = self.model(**inp)
        r = self.proc.post_process_grounded_object_detection(out, inp.input_ids, threshold=self.cfg["box_threshold"],
                                                             text_threshold=self.cfg["text_threshold"],
                                                             target_sizes=[rgb.shape[:2]])[0]
        dets = [{"bbox": [float(v) for v in b], "score": float(s), "label": str(l)}
                for b, s, l in zip(r["boxes"].cpu().numpy(), r["scores"].cpu().numpy(), r.get("text_labels", r.get("labels")))]
        dets.sort(key=lambda d: -d["score"])
        keep = []
        for d in dets:                                   # NMS
            if all(_iou(d["bbox"], k["bbox"]) < self.cfg["nms_iou"] for k in keep):
                keep.append(d)
        # a detection enclosing several others is the group, not a box
        keep = [d for d in keep if sum(_frac_inside(o["bbox"], d["bbox"]) >= self.cfg["container_overlap"]
                                       for o in keep if o is not d) < self.cfg["container_min_children"]]
        keep.sort(key=lambda d: (d["bbox"][0] + d["bbox"][2]) / 2)
        return keep
