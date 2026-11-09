"""Summarise runs: per run, per level and per L1 factor the success rate,
the error split (perception vs execution), time, tool / LLM calls; success-vs-factor curves for L1.

python scripts/summarize.py --runs runs/<run1> runs/<run2> ... [--out runs/summary_<name>]
"""
import argparse, json, os, sys, time
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+", required=True)
ap.add_argument("--out", default=None)
a = ap.parse_args()
df = pd.concat([pd.read_csv(os.path.join(r, "results.csv")).assign(run=os.path.basename(r.rstrip("/"))) for r in a.runs], ignore_index=True)
out = a.out or os.path.join(os.path.dirname(a.runs[0].rstrip("/")), "summary_" + time.strftime("%Y%m%d_%H%M"))
os.makedirs(out, exist_ok=True)
df["eval_success"] = df["eval_success"].astype(str).str.lower() == "true"
df["harness_success"] = df["harness_success"].astype(str).str.lower() == "true"
L = ["# 评测汇总", "", f"生成时间 {time.strftime('%F %T')}；runs: {', '.join(sorted(df['run'].unique()))}", ""]
err_cols = {1: ["d_err_m", "perception_err_pos_m", "execution_err_pos_m", "yaw_err_deg"],
            2: ["wrist_dist_to_plane_m", "wrist_axis_offset_m", "wrist_angle_err_deg", "head_perception_err_m", "wrist_perception_err_m", "execution_err_m"]}
for task in sorted(df["task"].unique()):
    d = df[df["task"] == task]
    L += [f"## 任务 {task}", ""]
    g = d.groupby(["level", "mode"])
    tab = g.agg(n=("scenario", "count"), success=("eval_success", "mean"), harness_says_success=("harness_success", "mean"),
                wall_s=("wall_s", "mean"), tool_calls=("tool_calls", "mean"), llm_calls=("llm_calls", "mean"),
                **{c: (c, "median") for c in err_cols[task] if c in d})
    L += ["按级别 × 模式（误差列为中位数）：", "", tab.round(3).to_markdown(), ""]
    l1 = d[d["level"] == "L1"]
    if len(l1):
        t2 = l1.groupby(["factor_name", "mode"]).agg(n=("scenario", "count"), success=("eval_success", "mean")).unstack("mode")
        L += ["L1 各因素成功率：", "", t2.round(2).to_markdown(), ""]
        for fac, fd in l1.groupby("factor_name"):
            fig, ax = plt.subplots(figsize=(5, 3))
            for mode, md in fd.groupby("mode"):
                md = md.copy()
                try:
                    md["x"] = pd.to_numeric(md["factor_value"])
                    s = md.groupby("x")["eval_success"].mean()
                    ax.plot(s.index, s.values, marker="o", label=mode)
                except (ValueError, TypeError):
                    s = md.groupby("factor_value")["eval_success"].mean()
                    ax.plot(range(len(s)), s.values, marker="o", label=mode)
                    ax.set_xticks(range(len(s))); ax.set_xticklabels([str(x)[:14] for x in s.index], rotation=30, fontsize=7)
            ax.set_title(f"task {task} · {fac}"); ax.set_ylabel("success"); ax.set_ylim(-0.05, 1.05); ax.legend(fontsize=7)
            fig.tight_layout(); p = os.path.join(out, f"t{task}_L1_{fac}.png"); fig.savefig(p, dpi=110); plt.close(fig)
            L.append(f"![t{task} {fac}]({os.path.basename(p)})")
        L.append("")
    fails = d[~d["eval_success"]]
    if len(fails):
        L += ["失败原因（harness 自述 / 编排异常）计数：", "", fails.groupby(["mode", "failure"]).size().rename("n").reset_index().to_markdown(index=False), ""]
trunc = df["llm_length_truncations"].fillna(0).sum()
calls = df["llm_calls"].fillna(0).sum()
L += ["## LLM", "", f"LLM 调用 {int(calls)} 次，被截断（finish_reason=length）{int(trunc)} 次，截断率 {trunc / max(calls, 1):.2%}。", ""]
open(os.path.join(out, "report.md"), "w").write("\n".join(L))
df.to_csv(os.path.join(out, "all_results.csv"), index=False)
print("SUMMARY", os.path.join(out, "report.md"))
