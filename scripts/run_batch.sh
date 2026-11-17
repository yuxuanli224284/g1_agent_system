#!/usr/bin/env bash
# N sim servers on one GPU (GPU=k: GPU 7 = user-approved shared card outside gpuq, or a card leased via gpuq by the caller)
# Guards (same as g1_barcode_sim/scripts/controller_compare/run_eval_gpu7.sh, NOTES §11 of g1_barcode_sim):
#  pre-flight free-memory check, watchdog that stops OUR jobs if GPU 7 runs low, clean server shutdown, leftover check.
# usage (detached): CLIENTS='cmd0;cmd1;...' TAG=name setsid nohup bash scripts/run_batch_gpu7.sh > runs/batch_<tag>.out 2>&1 &
#   one client command per sim instance; each runs with SIM_INSTANCE=i in the g1_agent env, cwd = g1_agent_system.
cd "$(dirname "$0")/.."
B=/data/yicheng/HumanoidBarcodeScan
export PROGRESS_CLIENT_DIR=$HOME/random/switchyard/clients
INST_BASE=${INST_BASE:-0}   # first SIM_INSTANCE (ports + 10*i); use a free range when another batch is running
TAG=${TAG:?set TAG}; IFS=';' read -ra CLIENTS <<< "${CLIENTS:?set CLIENTS}"; N=${#CLIENTS[@]}
GPU=${GPU:?set GPU (physical index)}
PER_JOB_MIB=5000      # measured: one G1BarcodeEnv instance holds 4767 MiB on this card (2026-11-06)
HEADROOM_MIB=4096     # kept free for the other tenants of GPU 7 at all times (our choice, reported to the user)
LOGD=runs/batch_$TAG; mkdir -p $LOGD
q() { nvidia-smi -i $GPU --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | tr -d ' '; }
UUID=$(nvidia-smi -i $GPU --query-gpu=uuid --format=csv,noheader)
holders() { nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader | grep "$UUID" | cut -d, -f1 | sort; }
IFS=, read used total <<< "$(q)"; free=$((total - used)); need=$((N * PER_JOB_MIB + HEADROOM_MIB))
holders > $LOGD/holders_before.txt
echo "PREFLIGHT GPU$GPU free=${free} need=${need} instances=$N holders_before=$(tr '\n' ' ' < $LOGD/holders_before.txt)"
[ "$free" -lt "$need" ] && { echo "ABORT not enough free memory on GPU$GPU"; exit 2; }

SERVERS=()
for i in $(seq 0 $((N - 1))); do
  SIM_GPU=$GPU SIM_INSTANCE=$((INST_BASE + i)) bash scripts/start_sim_server.sh > $LOGD/sim_server_$i.log 2>&1 &
  SERVERS+=($!)
done
for i in $(seq 0 $((N - 1))); do      # wait until every server is up (or died)
  until grep -q SIM_SERVER_READY $LOGD/sim_server_$i.log; do
    kill -0 ${SERVERS[$i]} 2>/dev/null || { echo "ABORT sim server $i died during start (see $LOGD/sim_server_$i.log)"; exit 3; }
    sleep 5
  done
done
echo "SERVERS_READY pids ${SERVERS[*]}"

export G1_EVAL_TOKEN=$(cat runs/.eval_token)
P=$B/conda_envs/g1_agent/bin/python
CL=()
for i in $(seq 0 $((N - 1))); do
  # the client's GroundingDINO runs on the approved card too (2026-11-07: without this it defaulted to physical GPU 0,
  # which is not leased to this project, and hit CUDA OOM when that card was full)
  (export SIM_INSTANCE=$((INST_BASE + i)) CUDA_VISIBLE_DEVICES=$GPU; eval "$P -u ${CLIENTS[$i]}") > $LOGD/client_$i.log 2>&1 &
  CL+=($!)
done
echo "CLIENTS_STARTED pids ${CL[*]}"
alive() { for c in "${CL[@]}"; do kill -0 $c 2>/dev/null && return 0; done; return 1; }
while alive; do
  IFS=, read used total <<< "$(q)"
  if [ $((total - used)) -lt $HEADROOM_MIB ]; then
    echo "WATCHDOG GPU$GPU free $((total - used)) MiB < $HEADROOM_MIB: stopping our clients and servers"
    for c in "${CL[@]}"; do kill -TERM $c 2>/dev/null; done
    for s in "${SERVERS[@]}"; do pkill -TERM -P $s 2>/dev/null; kill -TERM $s 2>/dev/null; done
    sleep 20
    for s in "${SERVERS[@]}"; do pkill -KILL -P $s 2>/dev/null; kill -KILL $s 2>/dev/null; done
    break
  fi
  sleep 15
done
wait "${CL[@]}" 2>/dev/null
for i in $(seq 0 $((N - 1))); do echo "CLIENT $i: $(grep -E 'RUN_DONE|RECORD_DONE|Traceback|"success"' $LOGD/client_$i.log | tail -1)"; done
# clean shutdown of every server through the eval port, then make sure nothing is left
for i in $(seq 0 $((N - 1))); do
  SIM_INSTANCE=$((INST_BASE + i)) $P -c "from g1agent.eval.eval_client import EvalClient; EvalClient().shutdown()" > /dev/null 2>&1 &
done
sleep 30
for s in "${SERVERS[@]}"; do kill -0 $s 2>/dev/null && { pkill -KILL -P $s 2>/dev/null; kill -KILL $s 2>/dev/null; echo "server wrapper $s needed SIGKILL"; }; done
sleep 5
holders > $LOGD/holders_after.txt
# only this launch's instances (another batch's servers may legitimately still be running); a leaked fork keeps the
# parent's command line and environment, so SIM_INSTANCE identifies it too
mine=$(for p in $(pgrep -u "$USER" -f "g1agent.robot.sim_server"); do
  inst=$(tr '\0' '\n' < /proc/$p/environ 2>/dev/null | sed -n 's/^SIM_INSTANCE=//p')
  [ -n "$inst" ] && [ "$inst" -ge "$INST_BASE" ] && [ "$inst" -lt $((INST_BASE + N)) ] && ls -l /proc/$p/fd 2>/dev/null | grep -q /dev/nvidia && echo $p
done)
echo "POSTCHECK holders_after=$(tr '\n' ' ' < $LOGD/holders_after.txt) new_vs_before=$(comm -13 $LOGD/holders_before.txt $LOGD/holders_after.txt | tr '\n' ' ') leftover_mine=[${mine}]"
echo BATCH_DONE
