"""Shared helpers for the LLM orchestrators: prompts, image messages, LLM logging."""
from __future__ import annotations

import json
import os

from ..common import config


def load_prompt(name: str, **kw) -> tuple[str, str]:
    """Returns (text, version tag). The robot-specific placeholders ({robot_name}, {scanner_hand}, {head_camera_*},
    {body_height_*}) come from the selected robot file's `prompt` block (configs/robot.yaml: the original G1 wording);
    a caller's keyword of the same name is an error."""
    p = os.path.join(config.ROOT, "prompts", f"{name}.md")
    txt = open(p).read()
    first = txt.splitlines()[0]
    version = first.replace("<!--", "").replace("-->", "").strip()
    body = "\n".join(txt.splitlines()[1:]).strip()
    robot_vars = config.load("robot")["prompt"]
    clash = sorted(set(robot_vars) & set(kw))
    if clash:
        raise ValueError(f"prompt keywords {clash} are robot prompt variables (configs/robot*.yaml prompt)")
    return body.format(**robot_vars, **kw), version


def log_llm(log, kind, version, messages, reply):
    """Log an LLM exchange; images are saved to files and replaced by their paths. Writes the reply text (the agent's
    stated rationale), reasoning_content when the server returns one, and the call's unix start / end times and
    per-attempt record (truncated attempts included)."""
    def strip(m):
        if isinstance(m.get("content"), list):
            parts = []
            for p in m["content"]:
                parts.append({"type": "image", "note": p.get("_path", "image")} if p.get("type") == "image_url" else p)
            return dict(m, content=parts)
        return m
    log.llm({"kind": kind, "prompt_version": version, "messages": [strip(m) for m in messages],
             "reply": {"content": reply["content"], "reasoning_content": reply.get("reasoning_content"),
                       "tool_calls": reply["tool_calls"], "finish_reason": reply["finish_reason"], "usage": reply["usage"]},
             "t_start": reply.get("t_start"), "t_end": reply.get("t_end"), "attempts": reply.get("attempts")})


def image_message(llm, log, rgb, text, tag):
    part = llm.image_part(rgb)
    part["_path"] = log.image(rgb, f"llm_{tag}")
    return {"role": "user", "content": [{"type": "text", "text": text}, part]}


def clean(messages):
    """Drop the private _path keys before sending."""
    out = []
    for m in messages:
        if isinstance(m.get("content"), list):
            m = dict(m, content=[{k: v for k, v in p.items() if k != "_path"} for p in m["content"]])
        out.append(m)
    return out


def tool_result_text(res: dict) -> str:
    return json.dumps({k: v for k, v in res.items() if k not in ("image", "traceback")}, ensure_ascii=False)
