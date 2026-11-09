"""Pick K scenarios per suite for the agent modes from a finished rule-mode run (user decision 2026-11-06: 2 per suite).
Suites the rule run did not reach (or rows without a rule result) count as "no failure observed".
Policy: a suite with rule failures -> its first failure + the rule success closest to it in factor value (does the LLM
rescue what the state machine cannot, and does it keep what it can); a suite without failures -> the two ends of the
factor range (first and last scenario). Prints JSON {suite: [scenario ids]}.
python scripts/select_agent_subset.py --rule_runs runs/m5_rule_gear_* [--k 2]"""
import argparse, csv, glob, json

ap = argparse.ArgumentParser()
ap.add_argument("--rule_runs", nargs="+", required=True)
ap.add_argument("--k", type=int, default=2)
a = ap.parse_args()
if a.k != 2:
    raise SystemExit("the selection policy is defined for k = 2 (one failure + nearest success, or the two range ends)")
rows = [r for d in a.rule_runs for r in csv.DictReader(open(f"{d}/results.csv"))]
by = {}
for r in rows:
    by.setdefault(r["suite"], []).append(r)
# every generated L0 / L1 suite takes part, also those the (stopped) rule run never reached: their two range ends
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "g1_barcode_sim"))
from g1_barcode_sim.testsuite import list_suites, load_suite
ALL = [m["suite"] for m in list_suites() if m["suite"].startswith(("t1_L0", "t2_L0", "t1_L1_", "t2_L1_"))]
for su in ALL:
    if su not in by:
        ids = [sc["id"] for sc in load_suite(su)]
        by[su] = [{"scenario": ids[0], "eval_success": "", "factor_value": ""}, {"scenario": ids[-1], "eval_success": "", "factor_value": ""}]


def fval(r, i):
    """numeric factor position: the value when it is a number, else the scenario's index in the suite"""
    try:
        return float(r["factor_value"])
    except ValueError:
        return float(i)


out = {}
for suite, rs in sorted(by.items()):
    rs = sorted(rs, key=lambda r: r["scenario"])
    fails = [i for i, r in enumerate(rs) if r["eval_success"] == "False"]
    oks = [i for i, r in enumerate(rs) if r["eval_success"] == "True"]
    if fails and oks:
        f = fails[0]
        s = min(oks, key=lambda i: (abs(fval(rs[i], i) - fval(rs[f], f)), i))
        pick = [rs[s]["scenario"], rs[f]["scenario"]]
    elif fails:                       # every scenario failed: the two range ends
        pick = [rs[0]["scenario"], rs[-1]["scenario"]]
    else:
        pick = [rs[0]["scenario"], rs[-1]["scenario"]]
    out[suite] = list(dict.fromkeys(pick))      # a one-scenario suite yields a single pick
print(json.dumps(out, indent=1))
