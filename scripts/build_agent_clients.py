"""Split the agent-mode subset (runs/agent_subset.json from select_agent_subset.py) over N sim instances and print the
CLIENTS string for a batch launcher (one sim server + one client per instance): each instance runs its scenarios in agent_assist, then in agent_full.
Picked scenarios that have no rule-mode result yet also get one rule run, last (agent modes first: user priority).
python scripts/build_agent_clients.py runs/agent_subset.json N runs/m5_rule_gear_*"""
import csv, json, sys

sub = json.load(open(sys.argv[1]))
n = int(sys.argv[2])
rule_done = {r["scenario"] for d in sys.argv[3:] for r in csv.DictReader(open(f"{d}/results.csv"))}
bins = [[0, []] for _ in range(n)]
for suite, ids in sorted(sub.items(), key=lambda kv: (-len(kv[1]), kv[0])):
    b = min(bins, key=lambda b: b[0])
    b[0] += len(ids)
    b[1].append((suite, ids))
cmds = []
for i, (_, items) in enumerate(bins):
    if not items:
        continue
    su = " ".join(s for s, _ in items)
    ids = " ".join(x for _, xs in items for x in xs)
    # "$P" is expanded by the batch launcher (the g1_agent python)
    c = (f"scripts/run_suite.py --suite {su} --scenarios {ids} --mode agent_assist --run_name m5_assist_gear_{i} && "
         f"$P -u scripts/run_suite.py --suite {su} --scenarios {ids} --mode agent_full --run_name m5_full_gear_{i}")
    todo = [x for _, xs in items for x in xs if x not in rule_done]
    if todo:
        su_r = " ".join(s for s, xs in items if any(x in todo for x in xs))
        c += f" && $P -u scripts/run_suite.py --suite {su_r} --scenarios {' '.join(todo)} --mode rule --run_name m5_rule_gear_extra_{i}"
    cmds.append(c)
print(";".join(cmds))
