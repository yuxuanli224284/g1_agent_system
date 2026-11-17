# 安装与运行

本页说明如何在一台 Linux + NVIDIA GPU 的机器上装好三个进程所需的环境，并跑通第一个场景。仿真环境本身的安装见 [g1_barcode_sim 文档](https://yuxuanli224284.github.io/g1_docs/barcode_sim/installation.html)。

## 目录布局

agent 系统把 g1_barcode_sim 当作一个 Python 包导入（仿真服务器进程里）。默认两个仓库并排放置：

```
workspace/
├── g1_barcode_sim/          # 仿真环境
├── unitree_sim_isaaclab/    # g1_barcode_sim 依赖的上游机器人资产
└── g1_agent_system/         # 本仓库
```

如果 g1_barcode_sim 在别处，设置环境变量 `G1_SIM_ROOT=/path/to/g1_barcode_sim`（`configs/system.yaml` 的 `sim_server.sim_root` 是相对本仓库的默认值）。

## 三个 conda 环境

| 环境 | 进程 | 内容 |
|---|---|---|
| `unitree_sim_env` | 仿真服务器 | g1_barcode_sim 的环境（Isaac Sim 5.1 + Isaac Lab），另加 `pyzmq msgpack pin` |
| `vllm_env` | VLM 服务 | vLLM 0.11.2（含 torch 2.9），提供 OpenAI 兼容接口 |
| `g1_agent` | Harness 客户端 | torch、transformers（GroundingDINO）、pinocchio、openai SDK、pydantic、opencv、pyzbar |

环境名写在 `configs/system.yaml` 的 `conda_env` 字段里。环境不是按名字注册在 conda 里、而是装在某个目录下时，设置 `G1_CONDA_PREFIX=<放环境的目录>`，启动脚本会激活 `$G1_CONDA_PREFIX/<环境名>`。

```bash
# 1) 仿真服务器环境：先按 g1_barcode_sim 文档装好 unitree_sim_env，再补三个包
conda activate unitree_sim_env
pip install pyzmq msgpack pin

# 2) VLM 服务环境
conda create -y -n vllm_env python=3.11 && conda activate vllm_env
pip install "vllm==0.11.2"

# 3) Harness 客户端环境（torch 的 CUDA 版本按你的驱动选择）
conda create -y -n g1_agent python=3.11 && conda activate g1_agent
pip install torch==2.9.0 torchvision --index-url https://download.pytorch.org/whl/cu128
pip install openai pydantic pyyaml numpy scipy opencv-contrib-python transformers pyzmq msgpack \
            matplotlib pandas pin pytest pyzbar pillow
conda install -y -c conda-forge zbar        # pyzbar 需要 zbar 共享库

# 4) 模型权重（只需下载一次）
python -c "from huggingface_hub import snapshot_download as d; d('Qwen/Qwen3-VL-8B-Instruct'); d('IDEA-Research/grounding-dino-tiny')"
```

显存：仿真服务器约 5 GB；vLLM 上的 Qwen3-VL-8B 权重约 17 GB，再加 KV cache；GroundingDINO-tiny 在 GPU 上不到 1 GB，也可以在 CPU 上跑（`configs/system.yaml` 的 `harness_client.detector_device: cpu`，更慢）。三个进程可以放在同一张显存足够的卡上，也可以分开。

## 启动

```bash
cd g1_agent_system
mkdir -p runs

# 仿真服务器（第一次启动会生成评测令牌 runs/.eval_token）
setsid nohup bash scripts/start_sim_server.sh > runs/sim_server.log 2>&1 &

# VLM 服务（rule 模式不需要）
setsid nohup bash scripts/start_vllm.sh > runs/vllm.log 2>&1 &
```

两个启动脚本可用的环境变量：

| 变量 | 作用 |
|---|---|
| `SIM_GPU` / `VLLM_GPU` | 指定 GPU 编号（设置 `CUDA_VISIBLE_DEVICES`）；不设就沿用当前的 `CUDA_VISIBLE_DEVICES` |
| `VLLM_KV_GIB`、`VLLM_GMU` | 和其他任务共用一张卡时必须设置：固定 KV cache 大小（GiB）和 vLLM 可占的显存比例，跳过 vLLM 启动时的显存探测（共享卡上探测会因别的进程释放显存而断言失败） |
| `SIM_INSTANCE=i` | 并行跑多个仿真服务器，第 i 个实例的端口依次加 `10·i` |
| `G1_CONDA_PREFIX`、`G1_SIM_ROOT` | 见上文 |
| `HF_HOME`、`TMPDIR`、`VLLM_CACHE_ROOT` | 模型缓存、临时目录、vLLM 编译缓存 |

仓库根目录下的 `local.env`（不进 git）会被两个启动脚本先读入，可以把本机的设置写在里面；其中还可以定义一个 shell 函数 `gpu_launch <标签> <命令...>`，用来把命令交给本机的 GPU 排队工具。

端口在 `configs/system.yaml` 里：机器人端口 31555、评测端口 31556、vLLM 31800（都在 Linux 临时端口范围 32768–60999 之外，2026-11-16 起）。

## 自检

```bash
conda activate g1_agent
export G1_EVAL_TOKEN=$(cat runs/.eval_token)
python -m pytest tests                    # 单元测试（需要仿真服务器的几项会自动跳过）
python scripts/check_m1_m2.py             # 进程互通；正运动学与仿真一致
python scripts/smoke_vlm.py               # 带图片和工具定义请求 VLM，检查能否返回合法工具调用
python scripts/check_m3.py                # L0 场景上的感知精度
```

## 运行

```bash
export G1_EVAL_TOKEN=$(cat runs/.eval_token)

# 单个场景：任务 2（条码前手腕就位），规则编排
python scripts/run_task.py --scenario t2_L0_0000 --mode rule

# 地面箱堆场景（场景自带任务 3：找箱 + 按顺序扫所有码），LLM 全程决策
python scripts/run_task.py --scenario stack4c_L0_0001 --mode agent_full

# 批量跑一个 suite（可断点续跑），再汇总
python scripts/run_suite.py --suite t1_L0 t1_L1 --mode rule --run_name t1_rule
python scripts/summarize.py --runs runs/t1_rule
```

每个 episode 的输出在 `runs/<run_name>/<scenario>/`：`tool_calls.jsonl`、`llm.jsonl`、`stages.jsonl`、`images/`、`result.json`、`final.png`、合成视频 `episode.mp4` 和时间线 `timeline.md`。各文件的含义见 [评测](evaluation.md) 和 [可视化](visualization.md)。

## 换模型

只改 `configs/llm.yaml`，例如换成任意 OpenAI 兼容服务：

```yaml
base_url: https://api.openai.com/v1
model: gpt-4.1
api_key_env: OPENAI_API_KEY        # 然后 export OPENAI_API_KEY=...
```

`max_tokens` 必须保持 `null`：系统不限制输出长度，被截断（`finish_reason == "length"`）的回复不会被解析，而是重新生成，并计入截断率统计（见 [LLM 服务与客户端](llm.md)）。
