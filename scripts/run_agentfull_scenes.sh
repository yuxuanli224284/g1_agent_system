#!/usr/bin/env bash
# agent_full on several scenarios at once, one shared vLLM, on GPU=k (copy of run_agentfull_scenes_gpu7.sh with the card as a parameter)
# other tenants must not be affected: run_batch_gpu7.sh pre-flight / watchdog / post-check). Switchyard progress =
# finished episodes / scenarios.
# usage: SCENS="a b" OUT=runs/x INST_BASE=4 setsid nohup bash scripts/run_agentfull_scenes_gpu7.sh > runs/x.out 2>&1 &
cd "$(dirname "$0")/.."
SCENS=${SCENS:?scenarios}; OUT=${OUT:?out dir}; TASK=${TASK:-3}
GPU=${GPU:?set GPU (physical index)}
UUID=$(nvidia-smi -i $GPU --query-gpu=uuid --format=csv,noheader)
holders() { nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader | grep "$UUID" | cut -d, -f1 | sort; }
SP="python3 $HOME/random/switchyard/clients/switchyard_progress.py"; RUN=humanscan-agentfull-$(basename $OUT)
N=$(echo $SCENS | wc -w)
$SP --id $RUN --label "HumanScan agent_full ($SCENS)" --gpus $GPU --pid $$ --status "Starting vLLM" --done 0 --total $N
holders > $OUT.holders_before.txt
VLLM_GPU=$GPU VLLM_KV_GIB=8 VLLM_GMU=0.32 setsid bash scripts/start_vllm.sh > $OUT.vllm.log 2>&1 &
VPGID=$!
for _ in $(seq 1 60); do
  grep -q "Application startup complete" $OUT.vllm.log && break
  kill -0 $VPGID 2>/dev/null || { echo "ABORT vLLM died during start ($OUT.vllm.log)"; $SP --id $RUN --finished --status "Failed: vLLM"; exit 3; }
  sleep 10
done
grep -q "Application startup complete" $OUT.vllm.log || { echo "ABORT vLLM not up after 10 min"; kill -TERM -- -$VPGID; exit 3; }
echo "VLLM_READY $(date +%T) gpu${GPU}_used=$(nvidia-smi -i $GPU --query-gpu=memory.used --format=csv,noheader)"
( while sleep 30; do $SP --id $RUN --status "Running episodes" --done $(ls $OUT/agent_full/*/result.json 2>/dev/null | wc -l) --total $N; done ) &
WATCH=$!
CL=""; for s in $SCENS; do CL="${CL:+$CL;}scripts/run_task.py --scenario $s --mode agent_full --task $TASK --run_dir $OUT/agent_full"; done
echo "CLIENTS $CL"
TAG=agentfull_$(basename $OUT) CLIENTS="$CL" GPU=$GPU bash scripts/run_batch.sh
kill $WATCH 2>/dev/null
kill -TERM -- -$VPGID 2>/dev/null; sleep 30; kill -KILL -- -$VPGID 2>/dev/null; sleep 5
holders > $OUT.holders_after.txt
d=$(ls $OUT/agent_full/*/result.json 2>/dev/null | wc -l)
$SP --id $RUN --done $d --total $N --finished --status "$([ $d -eq $N ] && echo Done || echo "Failed: $d/$N episodes")"
echo "VLLM_STOPPED POSTCHECK holders_after=$(tr '\n' ' ' < $OUT.holders_after.txt) new_vs_before=$(comm -13 $OUT.holders_before.txt $OUT.holders_after.txt | tr '\n' ' ')"
echo ALL_DONE
