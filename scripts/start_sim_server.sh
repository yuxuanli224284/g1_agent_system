#!/usr/bin/env bash
# Start the simulation server (G1BarcodeEnv behind ZeroMQ). Detached usage:
#   setsid nohup bash scripts/start_sim_server.sh > runs/sim_server.log 2>&1 &
#   export G1_EVAL_TOKEN=$(cat runs/.eval_token)    # created on first start; the harness needs it for evaluation
# Environment (all optional):
#   SIM_INSTANCE=i     parallel instances, ports + 10*i (configs/system.yaml)
#   SIM_GPU=k          run on GPU k (sets CUDA_VISIBLE_DEVICES); otherwise the inherited CUDA_VISIBLE_DEVICES is used
#   G1_CONDA_PREFIX    directory holding the conda envs, if they are not registered by name
#   G1_SIM_ROOT        path of g1_barcode_sim (default ../g1_barcode_sim)
#   G1_ROBOT           robot to simulate: g1 (default) | t800 (EngineAI T800 Pro); the harness must run with the same
# A file `local.env` in the repo root (not tracked) is sourced first; it may set the variables above and define a shell
# function `gpu_launch <label> <cmd...>` that wraps the command (e.g. a GPU queue on a shared machine).
cd "$(dirname "$0")/.."
[ -f local.env ] && source local.env
mkdir -p runs
[ -f runs/.eval_token ] || (umask 077; python3 -c "import secrets;print(secrets.token_hex(16))" > runs/.eval_token)
export G1_EVAL_TOKEN=$(cat runs/.eval_token)
ENV_NAME=$(python3 -c "import yaml;print(yaml.safe_load(open('configs/system.yaml'))['sim_server']['conda_env'])")
source "$(conda info --base)/etc/profile.d/conda.sh"
unset CC CXX CFLAGS CXXFLAGS CPPFLAGS LDFLAGS
set +u; conda activate ${G1_CONDA_PREFIX:+$G1_CONDA_PREFIX/}$ENV_NAME
export OMNI_KIT_ACCEPT_EULA=YES
export SIM_INSTANCE=${SIM_INSTANCE:-0}
[ -n "$SIM_GPU" ] && export CUDA_VISIBLE_DEVICES=$SIM_GPU
if [ -z "$SIM_GPU" ] && declare -F gpu_launch > /dev/null; then
  gpu_launch "sim server $SIM_INSTANCE" python -u -m g1agent.robot.sim_server; exit $?
fi
exec python -u -m g1agent.robot.sim_server
