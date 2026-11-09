"""Agent run timeline of one episode: <ep>/timeline.md (human-readable) + <ep>/timeline.jsonl (one event per line).

Merges llm.jsonl (what the LLM was given, what it said / reasoned, which tools it called), tool_calls.jsonl (every
harness tool call: args, status, reason, data, image) and stages.jsonl (orchestration stages) in chronological order.
Times are seconds since the episode logger started; LLM calls use their real start / end (t_start / t_end), tool calls
start at t_wall - seconds. Older logs without t_start are placed at their end time (marked "start unknown").

python scripts/make_timeline.py runs/<run>/<scenario> [more episode or run dirs ...]
(a run dir = every sub-directory that has a tool_calls.jsonl). run_task.py calls build_timeline() after every episode.
"""
from __future__ import annotations

import json
import os
import sys


def _jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _js(o):
    return json.dumps(o, ensure_ascii=False)


def _t0_unix(recs):
    """Logger start in unix time, from any record that has both t_unix and t_wall (new logs); None for old logs."""
    for r in recs:
        if "t_unix" in r and "t_wall" in r:
            return r["t_unix"] - r["t_wall"]
    return None


def _given(messages):
    """What was new for this LLM call: the messages after the last assistant message (system prompt excluded)."""
    last = max((i for i, m in enumerate(messages) if m.get("role") == "assistant"), default=-1)
    return [m for m in messages[last + 1:] if m.get("role") != "system"]


def _msg_summary(m):
    """One input message -> {"role", "text", "images", "tool_result"}; nothing is shortened."""
    out = {"role": m.get("role"), "text": None, "images": []}
    c = m.get("content")
    if isinstance(c, list):
        texts = []
        for p in c:
            if p.get("type") == "image":
                out["images"].append(p.get("note"))
            elif p.get("type") == "text":
                texts.append(p.get("text", ""))
        out["text"] = "\n".join(texts)
    else:
        out["text"] = c
    if m.get("role") == "tool":                     # tool messages are the JSON of orchestrators.common.tool_result_text
        res = json.loads(c)
        out["tool_result"] = {"status": res.get("status"), "reason": res.get("reason")}
        out["text"] = None
    return out


def collect(ep_dir: str) -> list[dict]:
    llm, tools, stages = (_jsonl(os.path.join(ep_dir, n)) for n in ("llm.jsonl", "tool_calls.jsonl", "stages.jsonl"))
    t0 = _t0_unix(llm + tools + stages)
    ev = []
    seen_versions = set()
    for k, r in enumerate(llm):
        end = r["t_wall"]
        start = None if (t0 is None or r.get("t_start") is None) else r["t_start"] - t0
        if t0 is not None and r.get("t_end") is not None:
            end = r["t_end"] - t0
        rep = r.get("reply")
        e = {"type": "llm", "seq": k, "log_seq": r.get("log_seq"), "t_start_s": start, "t_end_s": end, "purpose": r.get("kind"),
             "prompt_version": r.get("prompt_version")}
        if rep is None:                                   # bookkeeping record (invalid decision arguments)
            e.update(note={kk: v for kk, v in r.items() if kk not in ("kind", "t_wall", "t_unix")})
            ev.append(e)
            continue
        msgs = r.get("messages") or []
        sys_msgs = [m for m in msgs if m.get("role") == "system"]
        if sys_msgs and r.get("prompt_version") not in seen_versions:
            e["system_prompt"] = sys_msgs[0].get("content")
            seen_versions.add(r.get("prompt_version"))
        att = r.get("attempts")                           # absent in logs written before the attempts were recorded
        e.update(n_messages=len(msgs), given=[_msg_summary(m) for m in _given(msgs)],
                 thought=rep.get("content"), reasoning=rep.get("reasoning_content"),
                 tool_calls=[{"name": tc.get("name"), "arguments": tc.get("arguments"),
                              **({"arguments_raw": tc.get("arguments_raw"), "parse_error": tc["parse_error"]} if tc.get("parse_error") else {})}
                             for tc in rep.get("tool_calls") or []],
                 finish_reason=rep.get("finish_reason"), usage=rep.get("usage"),
                 n_attempts=None if att is None else len(att),
                 n_truncated=None if att is None else sum(a.get("finish_reason") == "length" for a in att))
        ev.append(e)
    for k, r in enumerate(tools):
        res = r.get("result") or {}
        ev.append({"type": "tool", "seq": k, "log_seq": r.get("log_seq"), "t_start_s": r["t_wall"] - r.get("seconds", 0.0), "t_end_s": r["t_wall"],
                   "i": r.get("i"), "name": r.get("tool"), "args": r.get("args"), "status": res.get("status"),
                   "reason": res.get("reason"), "data": res.get("data"),
                   "flags": [f for f in ("rejected", "budget_exhausted") if res.get(f)], "image": r.get("image"),
                   "seconds": r.get("seconds")})
    for k, r in enumerate(stages):
        ev.append({"type": "stage", "seq": k, "log_seq": r.get("log_seq"), "t_start_s": r["t_wall"], "t_end_s": r["t_wall"], "name": r.get("stage"),
                   "info": r.get("info")})
    # equal start times: write order (log_seq) when the logs have it; older logs: LLM call, then its tools, then stages
    rank = {"llm": 0, "tool": 1, "stage": 2}
    ev.sort(key=lambda e: (e["t_start_s"] if e["t_start_s"] is not None else e["t_end_s"],
                           e["log_seq"] if e["log_seq"] is not None else rank[e["type"]], e["seq"]))
    return ev


def _t(x):
    return "?" if x is None else f"{x:8.2f}"


def render_md(ep_dir: str, ev: list[dict]) -> str:
    L = [f"# Agent timeline: {os.path.basename(os.path.normpath(ep_dir))}", ""]
    rp = os.path.join(ep_dir, "result.json")
    if os.path.exists(rp):
        with open(rp) as f:
            res = json.load(f)
        m = res.get("metrics") or {}
        L += [f"- scenario `{res.get('scenario')}`, task {res.get('task')}, mode `{res.get('mode')}`, wall {res.get('wall_s')} s",
              f"- harness finish: `{_js(res.get('harness_finish'))}`",
              f"- eval success: **{m.get('success')}**" + (f" — {m['failure']}" if m.get("failure") else ""),
              f"- tool calls {res.get('tool_calls')}, LLM stats `{_js(res.get('llm'))}`"]
        if res.get("orchestrator_error"):
            L.append(f"- orchestrator error: `{res['orchestrator_error']}`")
    if os.path.exists(os.path.join(ep_dir, "episode.mp4")):
        L.append("- video: [episode.mp4](episode.mp4)")
    n_llm = [e for e in ev if e["type"] == "llm" and "thought" in e]
    L += [f"- {len(n_llm)} LLM calls ({sum(1 for e in n_llm if not (e['thought'] or e['reasoning']))} without any text), "
          f"{sum(e['type'] == 'tool' for e in ev)} tool calls, {sum(e['type'] == 'stage' for e in ev)} stages", "",
          "Times: seconds since the episode started (start → end).", ""]
    for e in ev:
        if e["type"] == "stage":
            L += [f"## [{_t(e['t_start_s'])} s] STAGE `{e['name']}`", ""]
            if e["info"] is not None:
                L += [f"`{_js(e['info'])}`", ""]
        elif e["type"] == "llm":
            head = f"### [{_t(e['t_start_s'])} → {_t(e['t_end_s'])} s] LLM `{e['purpose']}`"
            if e["t_start_s"] is not None and e["t_end_s"] is not None:
                head += f" ({e['t_end_s'] - e['t_start_s']:.2f} s)"
            L += [head, ""]
            if "note" in e:
                L += [f"`{_js(e['note'])}`", ""]
                continue
            u = e.get("usage") or {}
            L.append(f"- prompt `{e['prompt_version']}`, {e['n_messages']} messages, tokens {u.get('prompt_tokens')}/"
                     f"{u.get('completion_tokens')}, finish `{e['finish_reason']}`, attempts {e['n_attempts']} "
                     f"(truncated {e['n_truncated']})")
            if e.get("system_prompt"):
                L += ["", "<details><summary>system prompt</summary>", "", "```", e["system_prompt"], "```", "", "</details>", ""]
            L.append("- **given**:")
            for g in e["given"]:
                parts = []
                if g.get("tool_result"):
                    parts.append(f"tool result status={g['tool_result']['status']} reason={_js(g['tool_result']['reason'])}")
                if g.get("text"):
                    parts.append(_js(g["text"]))
                if g["images"]:
                    parts.append("images: " + ", ".join(f"[{p}]({p})" for p in g["images"]))
                L.append(f"  - {g['role']}: " + "; ".join(parts))
            L.append(f"- **thought**: {_js(e['thought']) if e['thought'] else '(no text)'}")
            if e["reasoning"]:
                L.append(f"- **reasoning**: {_js(e['reasoning'])}")
            L.append("- **tool calls**: " + (", ".join(f"`{tc['name']}({_js(tc['arguments'])})`" + (f" PARSE ERROR {tc['parse_error']}" if tc.get("parse_error") else "")
                                                       for tc in e["tool_calls"]) or "(none)"))
            L.append("")
        else:
            flags = f" [{', '.join(e['flags'])}]" if e["flags"] else ""
            L += [f"#### [{_t(e['t_start_s'])} → {_t(e['t_end_s'])} s] tool #{e['i']} `{e['name']}` → **{e['status']}**{flags}"
                  f" ({e['seconds']} s)", "",
                  f"- args `{_js(e['args'])}`", f"- reason: {e['reason']}"]
            if e["image"]:
                L.append(f"- image: [{e['image']}]({e['image']})")
            if e["data"]:
                L += ["", "<details><summary>data</summary>", "", "```json", json.dumps(e["data"], ensure_ascii=False),
                      "```", "", "</details>"]
            L.append("")
    ep = os.path.join(ep_dir, "orchestrator_error.txt")
    if os.path.exists(ep):
        with open(ep) as f:
            L += ["## Orchestrator error", "", "```", f.read(), "```", ""]
    return "\n".join(L)


def build_timeline(ep_dir: str) -> tuple[str, str]:
    ev = collect(ep_dir)
    pj, pm = os.path.join(ep_dir, "timeline.jsonl"), os.path.join(ep_dir, "timeline.md")
    with open(pj, "w") as f:
        for e in ev:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    with open(pm, "w") as f:
        f.write(render_md(ep_dir, ev))
    return pm, pj


def _episode_dirs(d):
    if os.path.exists(os.path.join(d, "tool_calls.jsonl")) or os.path.exists(os.path.join(d, "llm.jsonl")):
        return [d]
    return sorted(os.path.join(d, s) for s in os.listdir(d) if os.path.exists(os.path.join(d, s, "tool_calls.jsonl")))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for d in sys.argv[1:]:
        eps = _episode_dirs(d)
        if not eps:
            raise SystemExit(f"no episode logs under {d}")
        for ep in eps:
            print("TIMELINE", build_timeline(ep)[0], flush=True)
