"""agent_assist: the rule state machine runs the task; the LLM is consulted only to choose an
exploration move when the target is missing / incomplete, and to choose retry / explore / give up after a tool failure."""
from __future__ import annotations

from pydantic import ValidationError

from ..harness.registry import ExploreArgs, openai_tools
from .common import clean, image_message, load_prompt, log_llm
from .rule import task_class

DECISION_TOOLS = openai_tools(["explore"]) + [
    {"type": "function", "function": {"name": "retry", "description": "Run the failed tool again.", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "give_up", "description": "Stop: the task cannot be completed.",
                                      "parameters": {"type": "object", "properties": {"reason": {"type": "string"}}, "required": ["reason"]}}}]


def _decide(h, llm, log, prompt_name, image_rgb, tag, **kw):
    text, version = load_prompt(prompt_name, **kw)
    msgs = [image_message(llm, log, image_rgb, text, tag)]
    r = llm.chat(clean(msgs), tools=DECISION_TOOLS, tool_choice="required")
    log_llm(log, tag, version, msgs, r)
    for tc in r["tool_calls"]:
        if tc["parse_error"] is None:
            return tc["name"], tc["arguments"]
    return "give_up", {"reason": "no valid decision from the LLM"}


def run_agent_assist(h, tcfg, selector, llm, log):
    history = []
    T = task_class(h.task)                         # raises on an unknown task id
    goal = "every cardboard box" if h.task == 3 else f"the {selector} cardboard box"

    def explore_policy(h, situation):
        _, img = h.tb.look_head()
        name, args = _decide(h, llm, log, "task1_agent_assist_explore", img, "explore", goal=goal,
                             situation=situation, history=history or "none")
        if name != "explore":
            return None
        try:
            mv = ExploreArgs(**args).model_dump()
        except (ValidationError, TypeError) as e:   # invalid arguments from the LLM: logged, treated as no move
            log.llm({"kind": "explore_invalid", "args": args, "error": str(e)})
            return None
        history.append(f"{mv['direction']} {mv['distance']}")
        return mv

    def recovery_policy(h, tool, result, attempt):
        _, img = h.tb.look_head()
        name, args = _decide(h, llm, log, "recovery", img, "recovery", task=h.task, tool=tool,
                             result={k: v for k, v in result.items() if k != "traceback"}, attempt=attempt)
        if name == "retry":
            return "retry"
        if name == "explore":
            try:
                return {"explore": ExploreArgs(**args).model_dump()}
            except (ValidationError, TypeError) as e:
                log.llm({"kind": "recovery_invalid", "args": args, "error": str(e)})
                return "give_up"
        return "give_up"

    return T(h, tcfg, selector, explore_policy=explore_policy, recovery_policy=recovery_policy).run()
