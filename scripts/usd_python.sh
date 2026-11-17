#!/usr/bin/env bash
# Run python with Pixar USD (pxr) on the CPU, without starting the Isaac Sim app: the USD core libraries and the
# PhysX schema shipped inside the Isaac Sim install of the simulator conda env (configs/system.yaml sim_server.conda_env).
# Used by the offline URDF export:  scripts/usd_python.sh scripts/export_urdf.py --robot t800 --source usd
# Environment: G1_CONDA_PREFIX as in start_sim_server.sh (directory holding the conda envs, if not registered by name).
set -eo pipefail
cd "$(dirname "$0")/.."
[ -f local.env ] && source local.env
ENV_NAME=$(python3 -c "import yaml;print(yaml.safe_load(open('configs/system.yaml'))['sim_server']['conda_env'])")
source "$(conda info --base)/etc/profile.d/conda.sh"
unset CC CXX CFLAGS CXXFLAGS CPPFLAGS LDFLAGS
set +u; conda activate ${G1_CONDA_PREFIX:+$G1_CONDA_PREFIX/}$ENV_NAME
# locate the isaacsim package without importing it (its __init__ bootstraps Kit and asks for the EULA)
X=$(python -c "import importlib.util, os; print(os.path.join(importlib.util.find_spec('isaacsim').submodule_search_locations[0], 'extscache'))")
L=$(ls -d "$X"/omni.usd.libs-* | head -n 1)
PH=$(ls -d "$X"/omni.usd.schema.physx-* | head -n 1)
[ -d "$L/pxr" ] || { echo "no pxr in $L" >&2; exit 2; }
export PYTHONPATH="$L:$PH${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$L/bin:$L/bin/deps:$PH/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec python "$@"
