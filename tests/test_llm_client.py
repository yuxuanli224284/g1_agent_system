"""LLM client against a local mock OpenAI-compatible server: tool-call parsing, truncation redraw, malformed JSON; and
the agent_full orchestrator driven by a scripted mock LLM over a fake toolbox (M7)."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

from g1agent.llm.client import LLMClient, LLMTruncated

REPLIES = []
REQUESTS = []


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        n = int(self.headers["Content-Length"])
        req = json.loads(self.rfile.read(n))
        assert "max_tokens" not in req and "max_completion_tokens" not in req       # global rule: no output caps
        REQUESTS.append(req)
        rep = REPLIES.pop(0)
        if rep.get("sleep"):
            import time
            time.sleep(rep["sleep"])                     # a runaway reply still generating when the client times out
        body = json.dumps({"id": "x", "object": "chat.completion", "created": 0, "model": req["model"],
                           "choices": [{"index": 0, "message": rep["message"], "finish_reason": rep["finish"]}],
                           "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(body)))
        self.end_headers(); self.wfile.write(body)


@pytest.fixture(scope="module")
def server():
    s = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{s.server_port}/v1"
    s.shutdown()


def tc(name, args, i=0):
    return {"id": f"c{i}", "type": "function", "function": {"name": name, "arguments": args}}


def client(url):
    return LLMClient({"base_url": url, "model": "mock", "api_key_env": "NONE", "temperature": 0.0, "max_tokens": None,
                      "timeout_s": 2, "image_max_side": 64, "max_retries_on_length": 1,
                      "redraw_sampling": {"temperature": 0.7, "top_p": 0.8, "top_k": 20}})


def test_tool_call_parsing(server):
    REPLIES[:] = [{"message": {"role": "assistant", "content": None, "tool_calls": [tc("explore", '{"direction": "left", "distance": 0.3}')]}, "finish": "tool_calls"}]
    r = client(server).chat([{"role": "user", "content": "hi"}], tools=[])
    assert r["tool_calls"][0]["name"] == "explore" and r["tool_calls"][0]["arguments"] == {"direction": "left", "distance": 0.3}


def test_truncated_reply_is_redrawn_not_parsed(server):
    REPLIES[:] = [{"message": {"role": "assistant", "content": "partial", "tool_calls": [tc("explore", '{"direc')]}, "finish": "length"},
                  {"message": {"role": "assistant", "content": None, "tool_calls": [tc("look_head", "{}")]}, "finish": "tool_calls"}]
    c = client(server)
    r = c.chat([{"role": "user", "content": "hi"}], tools=[])
    assert r["tool_calls"][0]["name"] == "look_head" and c.stats["length_truncations"] == 1
    REPLIES[:] = [{"message": {"role": "assistant", "content": "x"}, "finish": "length"}] * 2
    with pytest.raises(LLMTruncated):
        c.chat([{"role": "user", "content": "hi"}])


def test_redraw_samples_and_timeout_is_redrawn(server):
    """2026-11-07: a runaway greedy reply hit the 120 s timeout 3 times (identical SDK resends) and ended the episode.
    A timed-out or length-truncated reply is redrawn with the model's sampling config, never resent greedily."""
    REPLIES[:] = [{"message": {"role": "assistant", "content": "x"}, "finish": "stop", "sleep": 3},
                  {"message": {"role": "assistant", "content": None, "tool_calls": [tc("look_head", "{}")]}, "finish": "tool_calls"}]
    REQUESTS.clear()
    c = client(server)
    r = c.chat([{"role": "user", "content": "hi"}], tools=[])
    assert r["tool_calls"][0]["name"] == "look_head" and c.stats["timeouts"] == 1
    assert r["attempts"][0]["finish_reason"] == "timeout"
    assert REQUESTS[0]["temperature"] == 0.0
    assert REQUESTS[1]["temperature"] == 0.7 and REQUESTS[1]["top_p"] == 0.8 and REQUESTS[1]["top_k"] == 20


def test_malformed_arguments(server):
    REPLIES[:] = [{"message": {"role": "assistant", "content": None, "tool_calls": [tc("explore", "{not json")]}, "finish": "tool_calls"}]
    r = client(server).chat([{"role": "user", "content": "hi"}], tools=[])
    assert r["tool_calls"][0]["arguments"] is None and "not valid JSON" in r["tool_calls"][0]["parse_error"]


class FakeTB:
    class WS:
        explore_steps = 0
    ws = WS()

    def look_head(self):
        return {"status": "ok", "data": {}, "reason": ""}, np.zeros((8, 8, 3), np.uint8)

    def detect_boxes(self, prompt):
        return {"status": "ok", "data": {"n": 2}, "reason": ""}


class FakeLog:
    def __init__(self): self.tools, self.llms = [], []
    def tool(self, r): self.tools.append(r)
    def llm(self, r): self.llms.append(r)
    def image(self, rgb, tag): return f"{tag}.jpg"


def test_agent_full_with_mock_llm(server):
    from g1agent.harness.guards import Harness
    from g1agent.orchestrators.agent_full import run_agent_full
    REPLIES[:] = [{"message": {"role": "assistant", "content": None, "tool_calls": [tc("detect_boxes", "{}")]}, "finish": "tool_calls"},
                  {"message": {"role": "assistant", "content": "thinking"}, "finish": "stop"},          # no tool call -> nudge
                  {"message": {"role": "assistant", "content": None, "tool_calls": [tc("finish", '{"task": 1, "success": true, "summary": "done"}')]}, "finish": "tool_calls"}]
    log = FakeLog()
    h = Harness(FakeTB(), 1, {"max_tool_calls": 10, "max_explore_steps": 3, "max_retries_per_tool": 2, "max_wall_s": 60}, log,
                {"max_move_m": 0.5, "max_turn_deg": 15})
    run_agent_full(h, {"budget": {"max_tool_calls": 10, "max_explore_steps": 3}}, "left", client(server), log)
    assert h.finished == {"task": 1, "success": True, "summary": "done"}
    assert [t["tool"] for t in log.tools] == ["detect_boxes", "finish"]
