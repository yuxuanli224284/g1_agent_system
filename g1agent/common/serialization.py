"""msgpack with numpy arrays (no extra dependency): an ndarray becomes {"__nd__": 1, "dtype", "shape", "data"}."""
from __future__ import annotations

import msgpack
import numpy as np


def _default(o):
    if isinstance(o, np.ndarray):
        a = np.ascontiguousarray(o)
        return {"__nd__": 1, "dtype": a.dtype.str, "shape": list(a.shape), "data": a.tobytes()}
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(f"cannot serialize {type(o).__name__}")


def _hook(d):
    if d.get("__nd__") == 1:
        return np.frombuffer(d["data"], dtype=np.dtype(d["dtype"])).reshape(d["shape"]).copy()
    return d


def pack(obj) -> bytes:
    return msgpack.packb(obj, default=_default, use_bin_type=True)


def unpack(b: bytes):
    return msgpack.unpackb(b, object_hook=_hook, raw=False, strict_map_key=False)
