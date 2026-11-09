# g1_agent_system

**Unitree G1 · 手持扫码枪 · Qwen3-VL-8B · GroundingDINO · Isaac Sim**

g1_agent_system 是一个让人形机器人在仓库里**找箱子、找条码、走过去用手持扫码枪逐个扫码**的 agent 系统，运行在 [g1_barcode_sim](https://yuxuanli224284.github.io/g1_docs/barcode_sim/) 仿真环境上。系统把所有几何计算放在确定性的工具里，由规则状态机或视觉语言模型（VLM）通过工具调用来编排流程。

```{figure} images/system_overview.png
:alt: 系统总图：仿真服务器、VLM 服务、Harness 客户端三个进程，客户端内部的编排层、Harness、工具箱、感知与运动学，以及评测通道

系统总图。三个进程通过本机端口通信；评测器只能通过带令牌的评测端口读取真值，agent 一侧拿不到真值。
```

## 这份文档写什么

- **结构与实现**：每个组件做什么、怎么实现、关键参数从哪里来。
- **演进过程**：每个组件从最初版本开始的每一次修改——当时遇到的问题、数据、根因、改法和效果都记录在组件页的"演进记录"里，按时间排列。全系统的更新日志见 [开发时间线](changelog.md)，每轮实验的数据见 [实验记录](results.md)。文档中的日期和时间取自开发日志，均为 UTC。

## 三种编排模式

| 模式 | 流程由谁决定 | LLM 的作用 |
|---|---|---|
| `rule` | 固定状态机 | 不使用 |
| `agent_assist` | 同一个状态机 | 只在找不到目标或工具失败时，选择探索动作或 retry / explore / give_up |
| `agent_full` | LLM 通过工具调用决定每一步 | 全部流程 |

三种模式共用同一份工具注册表和同一个 Harness。LLM 不输出任何数值坐标，所有坐标都由工具计算。

```{toctree}
:maxdepth: 2
:caption: 入门

architecture
installation
tasks
```

```{toctree}
:maxdepth: 2
:caption: 组件实现

robot_interface
perception
motion
toolbox
harness
llm
orchestrators
evaluation
visualization
testing
```

```{toctree}
:maxdepth: 2
:caption: 开发过程

changelog
results
limitations
```
