"""OpenAI-compatible chat client. Only `chat.completions` + `tools` of the openai SDK are
used, so switching vLLM/Qwen3-VL -> GPT is a configs/llm.yaml change.

Global rule: no output-length cap is sent (max_tokens stays unset unless llm.yaml sets one explicitly, which it must
not). A reply that nevertheless ends with finish_reason == "length" is never parsed: it is logged, counted and
redrawn (llm.yaml max_retries_on_length); the truncation rate is reported in `stats`. A request that times out (a
runaway reply still generating; llm.yaml timeout_s is above the time to reach the context limit) is handled the same way.
Redraws sample with llm.yaml redraw_sampling (the model's generation config): a greedy redraw repeats the runaway.
"""
from __future__ import annotations

import base64
import json
import os
import time

import cv2
import numpy as np

from ..common import config


class LLMTruncated(RuntimeError):
    def __init__(self, msg, attempts=None):
        super().__init__(msg)
        self.attempts = attempts or []


class LLMClient:
    def __init__(self, cfg: dict | None = None):
        from openai import OpenAI
        self.cfg = cfg or config.load("llm")
        if self.cfg.get("max_tokens") is not None:
            raise ValueError("llm.yaml max_tokens must be null (global rule: no output-length caps)")
        # max_retries=0: the SDK would resend a timed-out request unchanged (same greedy runaway); chat() redraws instead
        self.client = OpenAI(base_url=self.cfg["base_url"], api_key=os.environ.get(self.cfg["api_key_env"]) or "EMPTY",
                             timeout=self.cfg["timeout_s"], max_retries=0)
        self.stats = {"calls": 0, "attempts": 0, "length_truncations": 0, "timeouts": 0, "prompt_tokens": 0,
                      "completion_tokens": 0, "seconds": 0.0}

    def image_part(self, rgb: np.ndarray) -> dict:
        """RGB uint8 image -> OpenAI image_url content part (JPEG, longest side <= image_max_side)."""
        h, w = rgb.shape[:2]
        s = self.cfg["image_max_side"] / max(h, w)
        if s < 1:
            rgb = cv2.resize(rgb, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()}}

    def chat(self, messages: list, tools: list | None = None, tool_choice: str = "auto") -> dict:
        """Returns {"content", "reasoning_content" (None unless the server sends it), "tool_calls": [{"id", "name",
        "arguments" (dict) | None, "arguments_raw", "parse_error"}], "finish_reason", "usage", "assistant_message",
        "t_start", "t_end" (unix, whole call incl. redraws), "attempts": [{"t_start", "t_end", "finish_reason", "usage"}]}.
        Raises LLMTruncated (with .attempts) when every attempt ends with finish_reason == "length"."""
        self.stats["calls"] += 1
        t_call = time.time()
        attempts = []
        for attempt in range(1 + self.cfg["max_retries_on_length"]):
            t0 = time.time()
            kw = dict(model=self.cfg["model"], messages=messages, temperature=self.cfg["temperature"])
            if attempt > 0:                                # redraw: the model's own sampling, not a greedy repeat
                rs = dict(self.cfg["redraw_sampling"])
                kw.update(temperature=rs.pop("temperature"), top_p=rs.pop("top_p"), extra_body=rs)
            if tools:
                kw.update(tools=tools, tool_choice=tool_choice)
            from openai import APITimeoutError
            try:
                r = self.client.chat.completions.create(**kw)
            except APITimeoutError:
                t1 = time.time()
                self.stats["attempts"] += 1
                self.stats["timeouts"] += 1
                self.stats["seconds"] += t1 - t0
                attempts.append({"t_start": t0, "t_end": t1, "finish_reason": "timeout", "usage": None})
                continue                                   # a runaway reply still generating: redraw
            t1 = time.time()
            self.stats["attempts"] += 1
            self.stats["seconds"] += t1 - t0
            if r.usage:
                self.stats["prompt_tokens"] += r.usage.prompt_tokens
                self.stats["completion_tokens"] += r.usage.completion_tokens
            ch = r.choices[0]
            attempts.append({"t_start": t0, "t_end": t1, "finish_reason": ch.finish_reason,
                             "usage": r.usage.model_dump() if r.usage else None})
            if ch.finish_reason == "length":
                self.stats["length_truncations"] += 1
                attempts[-1]["truncated_content"] = ch.message.content   # logged with the call, never parsed
                continue                                   # never parse a truncated reply; redraw
            calls = []
            for tc in ch.message.tool_calls or []:
                raw = tc.function.arguments
                try:
                    args, err = (json.loads(raw) if raw else {}), None
                except json.JSONDecodeError as e:
                    args, err = None, f"arguments are not valid JSON: {e}"
                calls.append({"id": tc.id, "name": tc.function.name, "arguments": args, "arguments_raw": raw, "parse_error": err})
            am = {"role": "assistant", "content": ch.message.content or ""}
            if ch.message.tool_calls:
                am["tool_calls"] = [{"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                                    for tc in ch.message.tool_calls]
            return {"content": ch.message.content, "reasoning_content": getattr(ch.message, "reasoning_content", None),
                    "tool_calls": calls, "finish_reason": ch.finish_reason, "usage": r.usage.model_dump() if r.usage else None,
                    "assistant_message": am, "t_start": t_call, "t_end": t1, "attempts": attempts}
        raise LLMTruncated(f"reply truncated (finish_reason=length) or timed out {1 + self.cfg['max_retries_on_length']} times", attempts)
