"""Stereo depth (perception/stereo_depth.py) on a synthetic rectified pair: a textured fronto-parallel plane and a
textured slanted plane, rendered into both images by their exact disparity; the depth registered to a colour camera
15 mm to the side must match the planes."""
import numpy as np

from g1agent.common import config
from g1agent.perception.stereo_depth import StereoDepth


def _K(w, h, f):
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])


def _pair(depth_fn, W=848, H=480, f=447.0, B=0.05, seed=0):
    """Left / right images of a random texture lying on the surface depth_fn(u, v) (left image coordinates)."""
    rng = np.random.default_rng(seed)
    tex = rng.random((H, W + 200)).astype(np.float32)
    import cv2
    tex = cv2.GaussianBlur(tex, (3, 3), 0.8)
    u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    Z = depth_fn(u, v)
    left = tex[:, 100:100 + W]
    # right image pixel (u, v) sees the left-image point u + d, with d = f B / Z at that point (solved by iteration)
    ur = u.copy()
    for _ in range(5):
        Zr = depth_fn(ur, v)
        ur = u + f * B / Zr
    right = cv2.remap(tex, ur + 100, v, cv2.INTER_LINEAR)
    to8 = lambda a: np.clip(a * 255, 0, 255).astype(np.uint8)
    return to8(left), to8(right), Z


def _sd(f=447.0, B=0.05, fc=617.0, dy=0.015):
    T_lr = np.eye(4); T_lr[0, 3] = B            # right imager 50 mm to the right (+x of the optical frame)
    T_cl = np.eye(4); T_cl[0, 3] = dy            # left imager seen from the colour camera: 15 mm to its right
    cfg = config.load("perception")["head_depth"]
    return StereoDepth(cfg, _K(848, 480, f), _K(848, 480, f), T_lr, [480, 848], _K(848, 480, fc), [480, 848], T_cl)


def test_fronto_parallel_plane_depth_and_registration():
    sd = _sd()
    for z0 in (0.6, 1.4, 2.5):
        l, r, _ = _pair(lambda u, v: np.full_like(u, z0))
        d, st = sd.compute(l, r)
        m = d > 0
        assert m.mean() > 0.7, (z0, st)
        assert abs(float(np.median(d[m])) - z0) < 0.01 * z0, (z0, float(np.median(d[m])))


def test_slanted_plane_follows_the_slope():
    sd = _sd()
    f = 447.0
    # plane Z = 1 + 0.5 * y  (y = (v - cy) Z / f): Z = 1 / (1 - 0.5 (v - cy) / f)
    zf = lambda u, v: 1.0 / (1.0 - 0.5 * (v - 240.0) / f)
    l, r, Z = _pair(zf)
    d, _ = sd.compute(l, r)
    # in the colour frame the plane is the same (colour camera translated along x only, same orientation)
    vv = np.arange(480, dtype=np.float32)[:, None] * np.ones((1, 848), np.float32)
    zc = 1.0 / (1.0 - 0.5 * (vv - 240.0) / 617.0)
    m = d > 0
    rel = np.abs(d[m] - zc[m]) / zc[m]
    assert np.median(rel) < 0.01 and np.percentile(rel, 90) < 0.03
