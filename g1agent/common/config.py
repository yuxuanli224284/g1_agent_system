"""YAML configs of g1_agent_system (configs/*.yaml).

A config may start with `inherits: [a, b, ...]`: the top-level keys of the listed configs are taken first (in order),
then the file's own top-level keys replace them. Two parents defining the same key differently is an error unless the
file itself sets that key (no silent "last parent wins").

Robot selection: the robot is chosen per process by the environment variable G1_ROBOT (default g1; the --robot flag
of run_task / run_suite / sim_server / export_urdf sets it, so process-pool workers and child processes inherit it).
load("robot") returns the selected robot file: g1 -> configs/robot.yaml (unchanged G1 behaviour), t800 ->
configs/robot_t800.yaml (EngineAI T800 Pro, inherits robot.yaml). A robot file may carry `task_overrides: {<config
name>: {key: value}}`, applied to that config wherever it is loaded (also through `inherits`, e.g. task1 -> task3)."""
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


ROBOT_ENV = "G1_ROBOT"
ROBOT_FILES = {"g1": "robot", "t800": "robot_t800"}     # robot model -> config name


def robot_model() -> str:
    """The robot this process works with ($G1_ROBOT, default g1)."""
    m = os.environ.get(ROBOT_ENV, "g1")
    if m not in ROBOT_FILES:
        raise ValueError(f"${ROBOT_ENV}={m!r}: unknown robot; known: {sorted(ROBOT_FILES)}")
    return m


def _robot_file() -> dict:
    """The selected robot file (inheritance resolved), without task overrides of its own."""
    name = ROBOT_FILES[robot_model()]
    cfg = _load(name)
    if cfg.get("model") != robot_model():
        raise ValueError(f"configs/{name}.yaml must declare model: {robot_model()} (got {cfg.get('model')!r})")
    return cfg


def _load(name: str, _seen: tuple = (), _task_overrides: dict | None = None) -> dict:
    if name in _seen:
        raise ValueError(f"config inheritance cycle: {' -> '.join(_seen + (name,))}")
    own = _read(name)
    parents = own.pop("inherits", None) if isinstance(own, dict) else None
    if not parents:
        out = _resolve_paths(name, own)
    else:
        out = merge_inherited(own, {p: _load(p, _seen + (name,), _task_overrides) for p in parents})
    over = (_task_overrides or {}).get(name)
    if over:
        unknown = [k for k in over if k not in out]
        if unknown:
            raise ValueError(f"robot task_overrides for {name}: keys {unknown} are not in configs/{name}.yaml")
        out = dict(out, **over)
    return out


def load(name: str) -> dict:
    """Config `name`; "robot" = the selected robot file (robot_model()). Other configs get the selected robot file's
    task_overrides (G1: none, so they are exactly the files)."""
    if name == "robot":
        return _robot_file()
    return _load(name, _task_overrides=_robot_file().get("task_overrides"))


def load_all() -> dict:
    return {n: load(n) for n in ("system", "llm", "robot", "perception", "task1", "task2", "task3", "viz")}
