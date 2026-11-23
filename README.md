# g1_agent_system：G1 人形机器人找码扫码 Agent 系统

<!-- branch-overview:start -->
> **分支一览**：本仓库按机器人 / 场景分成几个分支，`main` 是通用版，场景专用的改动放在各自分支，不并入 `main`。两个仓库（[g1_barcode_sim](https://github.com/yuxuanli224284/g1_barcode_sim) 仿真、[g1_agent_system](https://github.com/yuxuanli224284/g1_agent_system) agent）要用配套的分支。

| 分支 | 内容 | 状态 | 配套的 g1_barcode_sim 分支 |
|---|---|---|---|
| [`main`](https://github.com/yuxuanli224284/g1_agent_system/tree/main) | 通用版 agent：规则状态机或 Qwen3-VL（vLLM）通过工具调用编排找箱、定位条码、走位、扫码；D435i 红外双目立体深度 | 稳定 | [`main`](https://github.com/yuxuanli224284/g1_barcode_sim/tree/main) |
| [`g1-camera-switch`](https://github.com/yuxuanli224284/g1_agent_system/tree/g1-camera-switch) | 头部相机切换：agent 按机器人报告的相机型号（D435i / 原版单目）自动选设置 | 已合并进 `main`，仅作历史保留 | [`main`](https://github.com/yuxuanli224284/g1_barcode_sim/tree/main) |
| [`g1v2-scanroom`](https://github.com/yuxuanli224284/g1_agent_system/tree/g1v2-scanroom) | 适配实验室改装版 G1（`g1_v2`）与 5×6 扫码间任务（第 2–4 层逐行扫） | 早期版本，已被下一行取代 | [`g1v2-scanroom`](https://github.com/yuxuanli224284/g1_barcode_sim/tree/g1v2-scanroom) |
| [`g1v2-scanroom-layout`](https://github.com/yuxuanli224284/g1_agent_system/tree/g1v2-scanroom-layout) | 在 `g1v2-scanroom` 之上：已知墙面布局、墙面坐标系逐段重新配准、两列一段（strip）扫描、手上 D405 搜箱面 | 最新，18 个码全部扫到（`scanroom6x5_r234_strip_L0`） | [`g1v2-scanroom-layout`](https://github.com/yuxuanli224284/g1_barcode_sim/tree/g1v2-scanroom-layout) |
| **`t800`（当前）** | EngineAI T800 Pro 适配（`--robot t800`） | 未完成 | [`t800`](https://github.com/yuxuanli224284/g1_barcode_sim/tree/t800) |

标签：`g1-v1-upstream-cam` = 换 D435i 之前的版本；`g1v2-sub4-run1` = 扫码间 4 箱测试时的状态。
<!-- branch-overview:end -->

让 Unitree G1 人形机器人在仓库里找到纸箱、定位条码标签、走过去并用左手的手持扫码枪逐个扫码。系统运行在 [g1_barcode_sim](https://github.com/yuxuanli224284/g1_barcode_sim)（Isaac Sim 5.1 / Isaac Lab）仿真环境上：几何计算都放在确定性的工具里，由规则状态机或视觉语言模型（Qwen3-VL-8B，经 vLLM 提供服务）通过工具调用编排。

![系统总图](docs/images/system_overview.png)

**详细文档**（每个组件的实现、从最初版本开始的每一次改进、实验记录）：<https://yuxuanli224284.github.io/g1_docs/agent_system/>（源文件在 [`docs/`](docs/)，Sphinx）。

## 主要内容

| 部分 | 说明 |
|---|---|
| 三个进程 | 仿真服务器（G1BarcodeEnv + ZeroMQ RPC）、VLM 服务（vLLM 部署 Qwen3-VL-8B-Instruct，bf16，OpenAI 兼容接口）、Harness 客户端（感知、工具、编排、评测） |
| 编排模式 | `rule` 固定状态机；`agent_assist` 状态机 + LLM 只处理异常；`agent_full` LLM 通过工具调用决定每一步 |
| 感知 | GroundingDINO 检测箱子和条码标签，深度平面拟合求箱面，下蹲补看低处标签，扫码枪相机近距离修正 |
| 运动 | pinocchio 正运动学 / 阻尼最小二乘 IK，站位评估，GR00T WBC 行走与下蹲（仿真侧） |
| 任务 | 任务 1 找箱并接近；任务 2 条码前手腕就位；任务 3 找到所有箱子并按固定顺序扫完所有码（含四面箱堆） |
| 评测 | 只有评测器能通过带令牌的端口读取真值；合成视频、时间线和汇总报告 |

## 快速开始

需要先按 g1_barcode_sim 的文档装好仿真环境，并把两个仓库并排放置。三个 conda 环境的完整安装步骤见 [安装与运行](docs/installation.md)。

```bash
git clone https://github.com/yuxuanli224284/g1_agent_system.git && cd g1_agent_system
mkdir -p runs
setsid nohup bash scripts/start_sim_server.sh > runs/sim_server.log 2>&1 &   # 仿真服务器
setsid nohup bash scripts/start_vllm.sh > runs/vllm.log 2>&1 &               # VLM 服务（rule 模式不需要）

conda activate g1_agent
export G1_EVAL_TOKEN=$(cat runs/.eval_token)
python -m pytest tests                                                   # 单元测试
python scripts/run_task.py --scenario t2_L0_0000 --mode rule             # 桌面场景，规则编排
python scripts/run_task.py --scenario stack4c_L0_0001 --mode agent_full  # 纸箱箱堆，LLM 全程决策
```

**机器人选择（G1 / EngineAI T800 Pro）**：按进程用环境变量 `G1_ROBOT` 选择（默认 `g1`，行为与以前完全相同）；
`G1_ROBOT=t800` 时 harness 读 `configs/robot_t800.yaml`（继承 `robot.yaml`：右手持扫码枪、`assets/t800pro_scanner.urdf`、
无下蹲、观察距离上限 1.50 m——头保持 0°，靠后退看全整面），仿真服务器以 `config_overrides={"robot": {"model": "t800"}}`
启动 T800。仿真端与 harness 端必须同一机器人（`get_robot_info` 的 `robot_model` 不一致时 Toolbox 直接报错）。

```bash
G1_ROBOT=t800 SIM_INSTANCE=100 setsid nohup bash scripts/start_sim_server.sh > runs/sim_t800.log 2>&1 &   # 端口 48555/48556
SIM_INSTANCE=100 python scripts/run_task.py --robot t800 --scenario wall6x5_L0_0000 --mode rule
scripts/usd_python.sh scripts/export_urdf.py --robot t800 --source usd     # 从 T800 USD 离线导出 URDF（CPU，无需 Isaac 应用）
G1_ROBOT=t800 python scripts/reach_report.py --scenario wall6x5_L0_0000     # 站立可达性（每行标签）
```

## 目录结构

```
g1agent/robot/          RobotInterface、ZeroMQ RPC、仿真服务器
g1agent/perception/     箱子 / 条码检测、平面拟合、几何
g1agent/kinematics/     正运动学与 IK（pinocchio）
g1agent/tools/          工具箱（每个工具返回 status / data / reason）、站位评估、扫码顺序
g1agent/harness/        工具注册表、保护规则、世界状态、日志
g1agent/llm/            OpenAI 兼容客户端（截断与超时后重新生成）
g1agent/orchestrators/  rule / agent_assist / agent_full
g1agent/eval/           评测器（唯一读取真值的模块）
g1agent/viz/            第三人称标记、合成视频
prompts/                LLM 提示词（第一行是版本号）
configs/                系统、机器人、感知、任务、LLM、可视化配置
scripts/                启动、运行、批量评测、汇总、视频剪辑
tests/                  单元测试
docs/                   文档源文件
```

## 许可证

本仓库代码使用 MIT License（见 [`LICENSE`](LICENSE)）。`assets/g1_dex1_scanner.urdf` 由 [unitree_sim_isaaclab](https://github.com/unitreerobotics/unitree_sim_isaaclab)（Apache-2.0）的机器人模型导出；Qwen3-VL 与 GroundingDINO 的权重遵循各自的许可证，不随本仓库分发。
