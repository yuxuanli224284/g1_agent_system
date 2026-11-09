"""vLLM smoke test: one request with an image and one tool definition must come back with a
valid tool call. python scripts/smoke_vlm.py [image.png]"""
import os, sys, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import cv2
from g1agent.llm.client import LLMClient

from g1agent.common import config
img_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(config.load("system")["sim_server"]["sim_root"],
                                                               "suites/t2_L0/previews/t2_L0_0000_head.png")
rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
llm = LLMClient()
tools = [{"type": "function", "function": {"name": "explore", "description": "Move the robot to look for the target box.",
          "parameters": {"type": "object", "properties": {"direction": {"type": "string", "enum": ["left", "right", "back", "turn_left", "turn_right"]},
                                                          "distance": {"type": "number", "description": "metres (<= 0.5) or degrees for turns (<= 15)"}},
                         "required": ["direction", "distance"]}}}]
msgs = [{"role": "system", "content": "You control a humanoid robot. Always answer by calling exactly one tool."},
        {"role": "user", "content": [{"type": "text", "text": "This is the robot's head camera. Describe nothing; call the explore tool to move 0.3 m to the left."},
                                     llm.image_part(rgb)]}]
r = llm.chat(msgs, tools=tools)
print(json.dumps({"tool_calls": r["tool_calls"], "content": r["content"], "finish_reason": r["finish_reason"], "usage": r["usage"], "stats": llm.stats}, indent=1))
ok = len(r["tool_calls"]) == 1 and r["tool_calls"][0]["name"] == "explore" and r["tool_calls"][0]["parse_error"] is None
print("SMOKE_VLM", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
