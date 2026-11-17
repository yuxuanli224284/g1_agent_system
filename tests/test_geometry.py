"""Ray-plane intersection, triangulation, projection round trip, plane fitting on synthetic depth."""
import math

import numpy as np

from g1agent.perception import geometry as geo
from g1agent.perception.plane_fit import fit_front_face

K = np.array([[300.0, 0, 320], [0, 300.0, 240], [0, 0, 1]])


def test_ray_plane():
    p = geo.ray_plane([0, 0, 0], [1, 0, 0], [2, 5, 5], [-1, 0, 0])
    assert np.allclose(p, [2, 0, 0])
    assert geo.ray_plane([0, 0, 0], [0, 1, 0], [2, 0, 0], [-1, 0, 0]) is None        # parallel
    assert geo.ray_plane([0, 0, 0], [-1, 0, 0], [2, 0, 0], [-1, 0, 0]) is None       # behind


def test_triangulate():
    target = np.array([1.0, 0.2, 0.5])
    o1, o2 = np.zeros(3), np.array([0, 0.04, 0])
    p, gap = geo.triangulate(o1, target - o1, o2, target - o2)
    assert np.allclose(p, target, atol=1e-9) and gap < 1e-9


def test_project_backproject_roundtrip():
    rng = np.random.default_rng(0)
    pts = np.c_[rng.uniform(-1, 1, 50), rng.uniform(-1, 1, 50), rng.uniform(0.5, 3, 50)]
    uv, z = geo.project(K, np.eye(4), pts)
    back = geo.backproject(K, uv[:, 0], uv[:, 1], z)
    assert np.allclose(back, pts)


def test_horizontal_distance():
    assert math.isclose(geo.horizontal_distance_to_plane([0, 0, 1], [1, 0, 0], [-1, 0, 0]), 1.0)


def test_plane_fit_synthetic_face():
    """Camera 1 m in front of a vertical face (normal -x), looking along +x; a table plane below must be ignored."""
    H, W = 480, 640
    # camera optical frame in root: z_cam = +x_root, x_cam = -y_root, y_cam = -z_root
    T = np.eye(4); T[:3, :3] = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]]); T[:3, 3] = [0, 0, 1.0]
    vs, us = np.mgrid[0:H, 0:W]
    d = np.full((H, W), 1.0, np.float32)                     # face at x = 1 -> depth 1 everywhere above the table line
    ray_z = (vs - 240) / 300.0                                # downward component per unit depth
    table = ray_z > 0.3                                       # rays hitting the "table" (z = 0.7) before the face
    d[table] = (0.3 / ray_z[table]).astype(np.float32)
    r = fit_front_face(d, K, T, [100, 50, 540, 450], cfg={"depth_valid": [0.2, 3.0], "stride": 4, "ransac_iters": 300,
                                                         "inlier_dist": 0.01, "inlier_sigmas": 2.0, "max_normal_tilt_deg": 20.0, "min_inliers": 100, "seed": 0})
    assert r["status"] == "ok"
    assert np.allclose(r["normal"], [-1, 0, 0], atol=1e-3)
    assert abs(r["center"][0] - 1.0) < 1e-3


def test_plane_fit_noisy_depth_keeps_the_face_extent():
    """Stereo-like depth (noise sigma = c z^2) of a 0.8 m wide vertical face standing on a floor: with the noise model
    the support plane is removed and the extent comes from the largest region; the base line's floor outliers must not
    stretch the face (2026-11-13: 0.80 m fitted 1.17 m wide)."""
    import numpy as np
    from g1agent.common import config
    from g1agent.perception.plane_fit import fit_front_face
    W, H, f = 848, 480, 617.0
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
    # camera 1.2 m above the floor looking down 40 deg along +x; face plane x = 1.2, |y| <= 0.4, 0 <= z <= 0.8
    a = np.radians(40.0)
    R = np.array([[0, -np.sin(a), np.cos(a)], [-1, 0, 0], [0, -np.cos(a), -np.sin(a)]])   # optical -> root
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = [0, 0, 1.2]
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    d = np.stack([(u - W / 2) / f, (v - H / 2) / f, np.ones_like(u, float)], -1) @ R.T
    t_face = (1.2 - 0.0) / d[..., 0]
    hit = t_face[..., None] * d
    on_face = (d[..., 0] > 0) & (np.abs(hit[..., 1]) <= 0.4) & (hit[..., 2] + 1.2 >= 0) & (hit[..., 2] + 1.2 <= 0.8)
    t_floor = np.where(d[..., 2] < 0, -1.2 / np.minimum(d[..., 2], -1e-9), np.inf)
    t = np.where(on_face & (t_face < t_floor), t_face, t_floor)
    z = t                                              # the ray has optical z = 1, so t is the optical depth
    z = np.where(np.isfinite(z) & (z < 3.0), z, 0.0)
    c = 0.0089
    rng = np.random.default_rng(0)
    zn = np.where(z > 0, z + rng.normal(0, 1, z.shape) * c * z ** 2, 0.0)
    cfg = config.load("perception")["plane_fit"]
    r = fit_front_face(zn, K, T, [0, 0, W, H], cfg, sigma_coeff=c, min_structure_px=7)
    assert r["status"] == "ok"
    width = float(np.ptp(r["corners"][:, 1]))
    assert abs(width - 0.8) < 0.1, width


def test_plane_fit_box_outside_the_image_is_a_failure_not_an_error():
    """A box entirely outside the image (after clipping: empty) returns status fail (refine_with_wrist then keeps the
    stored face plane) instead of raising."""
    d = np.full((480, 640), 1.0, np.float32)
    for bbox in ([700, 100, 900, 300], [100, 500, 300, 700], [-300, -200, -10, -5]):
        r = fit_front_face(d, K, np.eye(4), bbox)
        assert r["status"] == "fail" and "outside" in r["reason"]
