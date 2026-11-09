"""YAML configs of g1_agent_system (configs/*.yaml).

A config may start with `inherits: [a, b, ...]`: the top-level keys of the listed configs are taken first (in order),
then the file's own top-level keys replace them. Two parents defining the same key differently is an error unless the
file itself sets that key (no silent "last parent wins")."""
from __future__ import annotations

import os

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CFG_DIR = os.path.join(ROOT, "configs")


def _read(name: str) -> dict:
    with open(os.path.join(CFG_DIR, f"{name}.yaml")) as f:
        return yaml.safe_load(f)


def merge_inherited(own: dict, parents: dict) -> dict:
    """own: the file's dict (with its "inherits" key removed); parents: {name: resolved dict} in inheritance order."""
    merged, origin = {}, {}
    for pname, p in parents.items():
        for k, v in p.items():
            if k in merged and merged[k] != v and k not in own:
                raise ValueError(f"config key {k!r} differs between {origin[k]} and {pname}; set it explicitly")
            merged[k], origin[k] = v, pname
    merged.update(own)
    return merged


def _resolve_paths(name: str, cfg: dict) -> dict:
    """system.yaml: sim_root comes from $G1_SIM_ROOT when set, else the file's value (relative to this repo)."""
    if name == "system":
        s = cfg["sim_server"]
        s["sim_root"] = os.path.abspath(os.environ.get("G1_SIM_ROOT") or os.path.join(ROOT, s["sim_root"]))
    return cfg


def load(name: str, _seen: tuple = ()) -> dict:
    if name in _seen:
        raise ValueError(f"config inheritance cycle: {' -> '.join(_seen + (name,))}")
    own = _read(name)
    parents = own.pop("inherits", None) if isinstance(own, dict) else None
    if not parents:
        return _resolve_paths(name, own)
    return merge_inherited(own, {p: load(p, _seen + (name,)) for p in parents})


def load_all() -> dict:
    return {n: load(n) for n in ("system", "llm", "robot", "perception", "task1", "task2", "task3", "viz")}
