"""M4: markers never reach the robot's cameras. Integration test against a running sim server
(skipped when none is reachable): head / wrist images with markers present vs removed and right after a third-person
snapshot must be pixel-identical, while the third-person render must show them."""
import os

import pytest


def _client():
    from g1agent.eval.eval_client import EvalClient
    tok = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", ".eval_token")
    if "G1_EVAL_TOKEN" not in os.environ and os.path.exists(tok):
        os.environ["G1_EVAL_TOKEN"] = open(tok).read().strip()
    c = EvalClient()
    c.rpc.timeout_ms = 300000
    return c


def test_markers_invisible_to_robot_cameras():
    try:
        ev = _client()
        ev.reset("t2_L0_0000")
    except (TimeoutError, RuntimeError, OSError) as e:
        pytest.skip(f"sim server not reachable: {e}")
    r = ev.invisibility_check()
    # RTX real-time rendering is not bit-exact between consecutive frames (temporal AA): measured same-state noise reaches
    # 55 grey levels on a few pixels, while a marker changes pixels by > 100 levels. A pixel counts as changed when it
    # differs by more than STRONG levels.
    STRONG = 60
    for cam in ("head", "wrist"):
        assert r[f"{cam}_shown_vs_removed"][f"n_gt{STRONG}"] > 0, f"control failed: probe not visible to the {cam} camera when shown: {r}"
        for pair in r[f"{cam}_hidden_vs_removed"]:
            assert pair[f"n_gt{STRONG}"] == 0, f"markers leak into the {cam} camera: {r}"
    assert r["head_after_snapshot"]["max"] == 0 and r["wrist_after_snapshot"]["max"] == 0, r
