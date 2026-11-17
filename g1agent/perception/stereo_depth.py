"""Head depth from the D435i infrared pair: rectification, semi-global block matching (OpenCV StereoSGBM), disparity ->
depth in the left imager frame, then registered to the colour camera (the frame every perception step works in), as
the RealSense SDK's depth-to-colour alignment does.

Inputs are what the camera gives on the robot as well: the two infrared images, their intrinsics and the right imager
pose in the left imager frame (factory calibration). The colour <-> left imager transform comes from the robot model
(URDF frames head_cam_optical / head_ir_left_optical, the realsense2_description layout).

Parameters: perception.yaml head_depth.
"""
from __future__ import annotations

import time

import numpy as np


class StereoDepth:
    def __init__(self, cfg: dict, K_left, K_right, T_left_right, size_ir, K_color, size_color, T_color_left):
        import cv2
        self.cv2 = cv2
        self.cfg = cfg
        H, W = int(size_ir[0]), int(size_ir[1])
        self.size_ir = (H, W)
        self.K_color = np.asarray(K_color, float)
        self.size_color = (int(size_color[0]), int(size_color[1]))
        self.T_color_left = np.asarray(T_color_left, float)
        T_lr = np.asarray(T_left_right, float)
        T_rl = np.linalg.inv(T_lr)                                  # left-frame points -> right frame (cv2's R, T)
        Kl, Kr = np.asarray(K_left, float), np.asarray(K_right, float)
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(Kl, np.zeros(5), Kr, np.zeros(5), (W, H), T_rl[:3, :3],
                                                    T_rl[:3, 3].reshape(3, 1), flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
        self.map_l = cv2.initUndistortRectifyMap(Kl, np.zeros(5), R1, P1, (W, H), cv2.CV_32FC1)
        self.map_r = cv2.initUndistortRectifyMap(Kr, np.zeros(5), R2, P2, (W, H), cv2.CV_32FC1)
        self.R1 = R1                                                # left frame -> rectified left frame
        self.f = float(P1[0, 0])
        self.cx, self.cy = float(P1[0, 2]), float(P1[1, 2])
        self.baseline = float(-P2[0, 3] / P2[0, 0])                 # P2[0, 3] = -f * B
        S = cfg["sgbm"]
        bs = int(S["block_size"])
        modes = {"sgbm": cv2.STEREO_SGBM_MODE_SGBM, "3way": cv2.STEREO_SGBM_MODE_SGBM_3WAY,
                 "hh": cv2.STEREO_SGBM_MODE_HH}
        self.sgbm = cv2.StereoSGBM_create(minDisparity=0, numDisparities=int(S["num_disparities"]), blockSize=bs,
                                          P1=int(S["p1_per_px"]) * bs * bs, P2=int(S["p2_per_px"]) * bs * bs,
                                          disp12MaxDiff=int(S["disp12_max_diff"]), uniquenessRatio=int(S["uniqueness_ratio"]),
                                          speckleWindowSize=int(S["speckle_window"]), speckleRange=int(S["speckle_range"]),
                                          preFilterCap=int(S["prefilter_cap"]), mode=modes[S["mode"]])
        lo, hi = cfg["valid_range_m"]
        if self.f * self.baseline / lo > int(S["num_disparities"]):
            raise ValueError(f"head_depth: num_disparities {S['num_disparities']} cannot reach the near limit {lo} m "
                             f"(needs {self.f * self.baseline / lo:.1f} px)")
        self.range = (float(lo), float(hi))
        self.sigma_coeff = float(cfg["disparity_sigma_px"]) / (self.f * self.baseline)   # sigma_z = coeff x z^2 (m)

    def disparity(self, ir_left, ir_right) -> np.ndarray:
        cv2 = self.cv2
        l = cv2.remap(np.asarray(ir_left, np.uint8), *self.map_l, cv2.INTER_LINEAR)
        r = cv2.remap(np.asarray(ir_right, np.uint8), *self.map_r, cv2.INTER_LINEAR)
        return self.sgbm.compute(l, r).astype(np.float32) / 16.0   # rectified left image, px; invalid < 0

    def compute(self, ir_left, ir_right) -> tuple[np.ndarray, dict]:
        """Depth image (m, 0 = unknown) in the colour camera frame and stats."""
        t0 = time.time()
        disp = self.disparity(ir_left, ir_right)
        H, W = self.size_ir
        v, u = np.nonzero(disp > 0)
        d = disp[v, u]
        Z = self.f * self.baseline / d
        keep = (Z >= self.range[0]) & (Z <= self.range[1])
        u, v, Z = u[keep], v[keep], Z[keep]
        X = np.stack([(u - self.cx) * Z / self.f, (v - self.cy) * Z / self.f, Z], 1)   # rectified left (geometry.py pixel convention)
        P = X @ self.R1                                             # R1^T X: back to the left imager frame
        Pc = P @ self.T_color_left[:3, :3].T + self.T_color_left[:3, 3]
        depth = self._splat(Pc)
        stats = {"valid_frac_ir": float(keep.sum()) / float(H * W), "valid_frac_color": float((depth > 0).mean()),
                 "ms": round(1000 * (time.time() - t0), 1)}
        return depth, stats

    def _splat(self, Pc) -> np.ndarray:
        """Nearest depth per colour pixel. Each point covers the colour pixels of its footprint (the colour camera is
        finer than the infrared one: fx 617 vs 447 px), so the registered image has no grid of holes."""
        H, W = self.size_color
        K = self.K_color
        z = Pc[:, 2]
        ok = z > 1e-3
        Pc, z = Pc[ok], z[ok]
        uc = K[0, 0] * Pc[:, 0] / z + K[0, 2]                    # geometry.project convention
        vc = K[1, 1] * Pc[:, 1] / z + K[1, 2]
        n = max(1, int(np.ceil(K[0, 0] / self.f)))                  # footprint in colour px (2 for 617 / 447 px)
        out = np.full((H, W), np.inf, np.float32)
        u0 = np.floor(uc - (n - 1) / 2.0).astype(int)
        v0 = np.floor(vc - (n - 1) / 2.0).astype(int)
        for du in range(n):
            for dv in range(n):
                uu, vv = u0 + du, v0 + dv
                m = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
                np.minimum.at(out, (vv[m], uu[m]), z[m].astype(np.float32))
        out[~np.isfinite(out)] = 0.0
        return out
