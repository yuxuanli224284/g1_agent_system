"""ZeroMQ REQ client with msgpack+numpy payloads."""
from __future__ import annotations

import zmq

from ..common.serialization import pack, unpack


class RpcClient:
    def __init__(self, host: str, port: int, timeout_s: float, token: str | None = None):
        self.addr, self.timeout_ms, self.token = f"tcp://{host}:{port}", int(timeout_s * 1000), token
        self.ctx = zmq.Context.instance()
        self._connect()

    def _connect(self):
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.connect(self.addr)

    def call(self, method: str, **args):
        req = {"method": method, "args": args}
        if self.token is not None:
            req["token"] = self.token
        self.sock.send(pack(req))
        try:
            rep = unpack(self.sock.recv())
        except zmq.error.Again:
            self.sock.close(); self._connect()          # a REQ socket is unusable after a missed reply
            raise TimeoutError(f"{self.addr} {method}: no reply within {self.timeout_ms / 1000:.0f} s")
        if isinstance(rep, dict) and rep.get("status") == "error" and "traceback" in rep:
            raise RuntimeError(f"server error in {method}: {rep['reason']}\n{rep['traceback']}")
        return rep
