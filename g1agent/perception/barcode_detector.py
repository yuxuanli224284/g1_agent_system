"""Barcode / QR localisation in an image region (detection only, no decoding needed): OpenCV's cv2.barcode and
cv2.QRCodeDetector on an upsampled ROI. Bar direction = the direction perpendicular to the dominant image gradient
inside the detected quadrilateral (structure tensor)."""
from __future__ import annotations

import cv2
import numpy as np

from ..common import config


def _bar_direction(gray, quad):
    mask = np.zeros(gray.shape, np.uint8)
    cv2.fillConvexPoly(mask, quad.astype(np.int32), 1)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    m = mask > 0
    J = np.array([[np.sum(gx[m] ** 2), np.sum(gx[m] * gy[m])], [np.sum(gx[m] * gy[m]), np.sum(gy[m] ** 2)]])
    w, V = np.linalg.eigh(J)
    g = V[:, 1]                                   # dominant gradient direction (across the bars)
    coherence = float((w[1] - w[0]) / (w[1] + w[0] + 1e-9))
    return np.array([-g[1], g[0]]), coherence     # along the bars


def detect_barcodes(rgb, roi=None, cfg: dict | None = None) -> list[dict]:
    """roi = [x0, y0, x1, y1] (pixels) or None. Returns [{"kind", "corners" (4x2, full-image px), "center",
    "bar_dir" (unit, image), "coherence", "width_px"}], largest first."""
    c = cfg or config.load("perception")["barcode"]
    h, w = rgb.shape[:2]
    if roi is None:
        roi = [0, 0, w, h]
    x0, y0, x1, y1 = roi
    px, py = c["roi_pad_frac"] * (x1 - x0), c["roi_pad_frac"] * (y1 - y0)
    x0, y0, x1, y1 = int(max(0, x0 - px)), int(max(0, y0 - py)), int(min(w, x1 + px)), int(min(h, y1 + py))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return []
    s = c["upscale"]
    crop = cv2.cvtColor(rgb[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY)
    big = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    found = []
    ok, pts = cv2.barcode.BarcodeDetector().detectMulti(big)
    if ok and pts is not None:
        found += [("barcode", p) for p in pts]
    ok, pts = cv2.QRCodeDetector().detectMulti(big)
    if ok and pts is not None:
        found += [("qr", p) for p in pts]
    out = []
    for kind, p in found:
        q = np.asarray(p, float).reshape(4, 2)
        d, coh = _bar_direction(big, q)
        full = q / s + [x0, y0]
        out.append({"kind": kind, "corners": full, "center": full.mean(axis=0), "bar_dir": d, "coherence": coh,
                    "width_px": float(max(np.linalg.norm(full[1] - full[0]), np.linalg.norm(full[2] - full[1])))})
    out.sort(key=lambda r: -cv2.contourArea(r["corners"].astype(np.float32)))
    return out


def _encloses(a, b) -> bool:
    """Box b = [x0, y0, x1, y1] lies inside box a and is strictly smaller (equal boxes enclose nothing)."""
    return a[0] <= b[0] and a[1] <= b[1] and b[2] <= a[2] and b[3] <= a[3] and \
        (b[2] - b[0]) * (b[3] - b[1]) < (a[2] - a[0]) * (a[3] - a[1])


def detect_label_dino(rgb, roi, box_detector, cfg: dict | None = None) -> list[dict]:
    """Label localisation with the open-vocabulary detector on the upsampled ROI (head camera). Returns the same
    record format as detect_barcodes (corners = detection box), with cv2 refinement inside the label when possible."""
    c = cfg or config.load("perception")["barcode"]
    h, w = rgb.shape[:2]
    x0, y0, x1, y1 = (int(round(v)) for v in roi)
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return []
    s = c["upscale"]
    crop = cv2.resize(rgb[y0:y1, x0:x1], None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    th = box_detector.cfg["box_threshold"]
    box_detector.cfg["box_threshold"] = c["label_threshold"]
    try:
        dets = box_detector.detect(crop, prompt=c["label_prompt"])
    finally:
        box_detector.cfg["box_threshold"] = th
    area = crop.shape[0] * crop.shape[1]
    dets = [d for d in dets if (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]) <= c["label_max_area_frac"] * area]
    # A detection that encloses another one is the surface carrying it (a box face), not a label: labels do not
    # contain labels. With the threshold low enough for labels on printed boxes (scores 0.25-0.42 there vs 0.41-0.72
    # on plain kraft, 2026-11-07, 14 captured views), whole box faces (scores up to 0.20) are the false positives this
    # removes; it also drops the looser of two nested boxes on the same label.
    dets = [d for d in dets if not any(e is not d and _encloses(d["bbox"], e["bbox"]) for e in dets)]
    out = []
    for d in dets:
        bx0, by0, bx1, by1 = d["bbox"]
        q = np.array([[bx0, by0], [bx1, by0], [bx1, by1], [bx0, by1]]) / s + [x0, y0]
        rec = {"kind": "label", "corners": q, "center": q.mean(axis=0), "score": d["score"], "width_px": float(q[1, 0] - q[0, 0])}
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        rec["bar_dir"], rec["coherence"] = _bar_direction(gray, q)
        refined = detect_barcodes(rgb, [q[0, 0], q[0, 1], q[2, 0], q[2, 1]], cfg=c)      # exact corners if cv2 can
        if refined:
            r = refined[0]
            rec.update({"kind": r["kind"], "corners": r["corners"], "center": r["center"], "bar_dir": r["bar_dir"],
                        "coherence": r["coherence"], "refined_by_cv2": True})
        out.append(rec)
    out.sort(key=lambda r: -r["score"])
    return out


def decode_symbol_polygons(rgb, rotation_step_deg: float) -> dict:
    """Decode every barcode / QR symbol in an image with zbar and return its outline polygons: {content: [polygon
    (Nx2 px, original image), ...]}, one polygon per image rotation that decoded it. zbar reads 1-D codes only within
    ~+-17 deg of an image axis, so the image is decoded at rotations 0, step, ... < 90 deg (canvas enlarged, nothing
    cropped) and the polygons are mapped back (same scheme as the simulator's scanner, g1_barcode_sim env.scan).
    A polygon that touches the image border belongs to a symbol cut by the image edge: it is not a full outline and
    is left out."""
    from pyzbar import pyzbar
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    H, W = gray.shape
    out: dict = {}
    for ang in np.arange(0.0, 90.0, rotation_step_deg):
        M = cv2.getRotationMatrix2D((W / 2, H / 2), float(ang), 1.0)
        cs, sn = abs(M[0, 0]), abs(M[0, 1])
        nw, nh = int(np.ceil(H * sn + W * cs)), int(np.ceil(H * cs + W * sn))
        M[0, 2] += nw / 2 - W / 2
        M[1, 2] += nh / 2 - H / 2
        Minv = cv2.invertAffineTransform(M)
        for r in pyzbar.decode(cv2.warpAffine(gray, M, (nw, nh), borderValue=255)):
            pts = np.array([(p.x, p.y) for p in r.polygon], float)
            poly = (Minv[:, :2] @ pts.T).T + Minv[:, 2]
            if poly[:, 0].min() < 1 or poly[:, 1].min() < 1 or poly[:, 0].max() > W - 2 or poly[:, 1].max() > H - 2:
                continue
            out.setdefault(r.data.decode(), []).append(poly)
    return out
