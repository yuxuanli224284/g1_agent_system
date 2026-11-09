"""Diagnostic markers as USD prims (runs inside the simulator process).

Markers must be invisible to the robot's cameras. Isaac's debug-draw overlay only exists in a
viewport (we run headless and capture third-person views with a render product), so markers are USD geometry under
/World/Markers that is kept invisible and shown only for the duration of a third-person render; afterwards they are
hidden again and the scene re-rendered, so every robot-camera frame is rendered without them
(tests/test_markers_invisible.py checks head and wrist images pixel by pixel).

Marker spec (dict): {"id": str, "color": red|blue|purple|green|yellow, "kind": point|pose|root,
                     "pos": [x, y, z] (world), "dir": [x, y, z] (pose: forward axis), "yaw": rad (root)}
"""
from __future__ import annotations

import math

import numpy as np
from pxr import Gf, UsdGeom, UsdShade, Sdf

MROOT = "/World/Markers"


def _mat(stage, path, rgb):
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, f"{path}/pbr")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*[0.6 * c for c in rgb]))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def _orient_z_to(d):
    """Quaternion (w, x, y, z) rotating +z onto unit vector d."""
    d = np.asarray(d, float); d = d / np.linalg.norm(d)
    z = np.array([0.0, 0.0, 1.0])
    c = float(z @ d)
    if c > 1 - 1e-9:
        return Gf.Quatd(1, 0, 0, 0)
    if c < -1 + 1e-9:
        return Gf.Quatd(0, 1, 0, 0)
    ax = np.cross(z, d); ax /= np.linalg.norm(ax)
    a = math.acos(c)
    s = math.sin(a / 2)
    return Gf.Quatd(math.cos(a / 2), *(ax * s))


class MarkerLayer:
    def __init__(self, stage, colors: dict, sizes: dict):
        self.stage, self.colors, self.sizes = stage, colors, sizes
        UsdGeom.Xform.Define(stage, MROOT)
        UsdGeom.Imageable(stage.GetPrimAtPath(MROOT)).MakeInvisible()
        self.mats = {k: _mat(stage, f"{MROOT}/_mat_{k}", v) for k, v in colors.items()}
        self.specs: dict[str, dict] = {}

    # ---------------------------------------------------------------- authoring
    def set(self, markers: list[dict], group: str):
        """Replace all markers of a group (harness: red/blue/purple; evaluator: green/yellow)."""
        for k in [k for k, v in self.specs.items() if v["group"] == group]:
            self.stage.RemovePrim(f"{MROOT}/{k}")
            del self.specs[k]
        for m in markers:
            key = f"{group}_{m['id']}".replace("-", "_").replace(".", "_")
            self.specs[key] = dict(m, group=group)
            self._author(key, m)
        UsdGeom.Imageable(self.stage.GetPrimAtPath(MROOT)).MakeInvisible()

    def clear(self, group: str | None = None):
        for k in [k for k, v in self.specs.items() if group is None or v["group"] == group]:
            self.stage.RemovePrim(f"{MROOT}/{k}")
            del self.specs[k]

    def _author(self, key, m):
        st, sz = self.stage, self.sizes
        base = f"{MROOT}/{key}"
        UsdGeom.Xform.Define(st, base)
        mat = self.mats[m["color"]]
        p = np.asarray(m["pos"], float)

        def sphere(path, c, r):
            s = UsdGeom.Sphere.Define(st, path); s.CreateRadiusAttr(r)
            UsdGeom.Xformable(s).AddTranslateOp().Set(Gf.Vec3d(*c))
            UsdShade.MaterialBindingAPI.Apply(s.GetPrim()).Bind(mat)

        def arrow(path, c, d, length, r):
            d = np.asarray(d, float); d /= np.linalg.norm(d)
            cyl = UsdGeom.Cylinder.Define(st, f"{path}_shaft"); cyl.CreateRadiusAttr(r); cyl.CreateHeightAttr(length); cyl.CreateAxisAttr("Z")
            x = UsdGeom.Xformable(cyl)
            x.AddTranslateOp().Set(Gf.Vec3d(*(c + d * length / 2))); x.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(_orient_z_to(d))
            UsdShade.MaterialBindingAPI.Apply(cyl.GetPrim()).Bind(mat)
            cone = UsdGeom.Cone.Define(st, f"{path}_head"); cone.CreateRadiusAttr(2.5 * r); cone.CreateHeightAttr(5 * r); cone.CreateAxisAttr("Z")
            x = UsdGeom.Xformable(cone)
            x.AddTranslateOp().Set(Gf.Vec3d(*(c + d * (length + 2.5 * r)))); x.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(_orient_z_to(d))
            UsdShade.MaterialBindingAPI.Apply(cone.GetPrim()).Bind(mat)

        if m["kind"] == "root":          # ground dot + heading arrow
            q = np.array([p[0], p[1], sz["root_height"]])
            sphere(f"{base}/dot", q, sz["root_radius"])
            arrow(f"{base}/dir", q, [math.cos(m["yaw"]), math.sin(m["yaw"]), 0.0], sz["root_arrow"], sz["shaft_radius"])
        elif m["kind"] == "pose":        # point + forward arrow
            sphere(f"{base}/dot", p, sz["point_radius"])
            arrow(f"{base}/dir", p, m["dir"], sz["pose_arrow"], sz["shaft_radius"])
        else:
            sphere(f"{base}/dot", p, sz["point_radius"])

    # ---------------------------------------------------------------- rendering
    def show(self, on: bool):
        im = UsdGeom.Imageable(self.stage.GetPrimAtPath(MROOT))
        im.MakeVisible() if on else im.MakeInvisible()
