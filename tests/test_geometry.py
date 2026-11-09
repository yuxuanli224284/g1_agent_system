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
                                                         "inlier_dist": 0.01, "max_normal_tilt_deg": 20.0, "min_inliers": 100, "seed": 0})
    assert r["status"] == "ok"
    assert np.allclose(r["normal"], [-1, 0, 0], atol=1e-3)
    assert abs(r["center"][0] - 1.0) < 1e-3
