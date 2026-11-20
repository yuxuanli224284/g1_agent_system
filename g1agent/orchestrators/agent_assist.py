"""agent_assist: the rule state machine runs the task; the LLM is consulted only to choose an
exploration move when the target is missing / incomplete, and to choose retry / explore / give up after a tool failure."""
from __future__ import annotations

from pydantic import ValidationError

from ..harness.registry import ExploreArgs, openai_tools
from ..tools.scan_order import describe_targets
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


def explore_prompt_kw(task, selector, tb) -> dict:
    """Goal and head-view text of the explore prompt. Without a target spec the goals are those of the v2 prompt
    ("... on a table"), and g1_v1's head-view text is its sentence, so a g1_v1 run sends the same prompt as before the
    port; a target spec (Toolbox.set_scan_targets) names the wall cells in robot terms."""
    targets = getattr(tb, "scan_targets", None)
    if task != 3:
        goal = f"the {selector} cardboard box on a table"
    elif targets is None:
        goal = "every cardboard box on a table"
    else:
        goal = f"the wall of cardboard boxes holding {describe_targets(targets)}"
    return {"goal": goal, "head_view": tb.cfg["robot"]["texts"]["explore_head_view"]}


def run_agent_assist(h, tcfg, selector, llm, log):
    history = []
    T = task_class(h.task)                         # raises on an unknown task id
    ekw = explore_prompt_kw(h.task, selector, h.tb)

    def explore_policy(h, situation):
        _, img = h.tb.look_head()
        name, args = _decide(h, llm, log, "task1_agent_assist_explore", img, "explore",
                             situation=situation, history=history or "none", **ekw)
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
