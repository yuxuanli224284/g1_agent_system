#!/usr/bin/env bash
# VLM service: Qwen3-VL behind vLLM's OpenAI-compatible API (model in configs/llm.yaml, port in configs/system.yaml).
#   setsid nohup bash scripts/start_vllm.sh > runs/vllm.log 2>&1 &
# Environment (all optional):
#   VLLM_GPU=k                 run on GPU k; with VLLM_KV_GIB + VLLM_GMU for a card shared with other jobs (see below)
#   G1_CONDA_PREFIX            directory holding the conda envs, if they are not registered by name
#   HF_HOME, TMPDIR, VLLM_CACHE_ROOT   model cache / temp / compile cache locations
# A file `local.env` in the repo root (not tracked) is sourced first; it may set the variables above and define a shell
# function `gpu_launch <label> <cmd...>` that wraps the command (e.g. a GPU queue on a shared machine).
cd "$(dirname "$0")/.."
[ -f local.env ] && source local.env
eval "$(python3 - <<'PY'
import yaml
s = yaml.safe_load(open("configs/system.yaml"))["vlm_server"]; l = yaml.safe_load(open("configs/llm.yaml"))
print(f'PORT={s["port"]} GMU={s["gpu_memory_utilization"]} MAXLEN={s["max_model_len"]} MODEL={l["model"]} ENV_NAME={s["conda_env"]}')
PY
)"
source "$(conda info --base)/etc/profile.d/conda.sh"
unset CC CXX CFLAGS CXXFLAGS CPPFLAGS LDFLAGS
set +u; conda activate ${G1_CONDA_PREFIX:+$G1_CONDA_PREFIX/}$ENV_NAME
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
# hermes = the tool-call format of Qwen2.5/Qwen3 chat templates (<tool_call>{json}</tool_call>); checked by the smoke test
ARGS=(serve "$MODEL" --host 127.0.0.1 --port $PORT --max-model-len $MAXLEN --enable-auto-tool-choice --tool-call-parser hermes
      --limit-mm-per-prompt '{"image": 8, "video": 0}')   # no video inputs: video 0 also skips vLLM's max-size video profiling (10+ min on a busy card)
if [ -n "$VLLM_GPU" ]; then
  # Card shared with other jobs: cap our share of the card.
  # vLLM's default KV sizing profiles free memory before/after a dummy pass and asserts the other tenants did not
  # release memory meanwhile, which a shared card does not guarantee (another job freed 15 GiB mid-profile ->
  # AssertionError). A fixed KV size skips that profiling; --enforce-eager skips torch.compile + CUDA-graph capture
  # (startup took 13 min on the shared card).
  # --gpu-memory-utilization is still checked at startup (free memory must cover it), so it is our cap on the card
  : "${VLLM_KV_GIB:?set VLLM_KV_GIB (KV cache size in GiB) when sharing a card}"
  : "${VLLM_GMU:?set VLLM_GMU (fraction of the card vLLM may take) when sharing a card}"
  export CUDA_VISIBLE_DEVICES=$VLLM_GPU
  exec vllm "${ARGS[@]}" --gpu-memory-utilization $VLLM_GMU --kv-cache-memory-bytes $(python3 -c "print(int($VLLM_KV_GIB * 2**30))") --enforce-eager
fi
if declare -F gpu_launch > /dev/null; then
  gpu_launch "vLLM ${MODEL}" vllm "${ARGS[@]}" --gpu-memory-utilization $GMU; exit $?
fi
exec vllm "${ARGS[@]}" --gpu-memory-utilization $GMU
