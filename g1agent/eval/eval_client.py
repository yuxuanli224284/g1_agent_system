"""Evaluation / episode-control client (token-protected eval port). Only the evaluator and the suite runner use it."""
from __future__ import annotations

import os

from ..common import config
from ..robot.rpc import RpcClient


class EvalClient:
    def __init__(self, instance: int | None = None):
        s = config.load("system")["sim_server"]
        i = int(os.environ.get("SIM_INSTANCE", "0")) if instance is None else instance
        token = os.environ.get(s["eval_token_env"])
        if not token:
            raise RuntimeError(f"${s['eval_token_env']} not set")
        self.rpc = RpcClient(s["host"], s["eval_port"] + i * s["instance_port_stride"], s["rpc_timeout_s"], token=token)

    def reset(self, scenario): return self.rpc.call("reset", scenario=scenario)
    def get_gt(self): return self.rpc.call("get_gt")
    def get_actual_state(self): return self.rpc.call("get_actual_state")
    def set_gt_markers(self, enabled: bool): return self.rpc.call("set_gt_markers", enabled=bool(enabled))
    def snapshot(self, path=None): return self.rpc.call("snapshot", path=path)
    def record_start(self, path, fps=None, task=None): return self.rpc.call("record_start", path=path, fps=fps, task=task)
    def record_stop(self): return self.rpc.call("record_stop")
    # agent events for the video overlays, stage states, robot-port scan log
    def annotate(self, event: dict): return self.rpc.call("annotate", event=event)
    def stage(self, name: str, info: dict | None = None): return self.rpc.call("stage", name=name, info=info)
    def get_stage_states(self): return self.rpc.call("get_stage_states")
    def get_scan_log(self): return self.rpc.call("get_scan_log")
    def invisibility_check(self): return self.rpc.call("invisibility_check")
    def shutdown(self): return self.rpc.call("shutdown")
