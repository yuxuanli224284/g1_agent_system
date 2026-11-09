# Harness：注册表、保护规则与状态

## 作用与在系统中的位置

Harness 是编排层和工具层之间的一层。不论动作由规则状态机还是 LLM 决定，每一次工具调用都只走一个入口 `Harness.call(name, args)`。它按固定顺序做这些事：预算检查、工具是否存在、工具是否属于当前任务、参数校验、语义限制、重试上限，然后执行工具、写日志，任务 3 还要记账（扫描进度、阶段事件、成功判定）。

这样安排有两个目的：

- **三种编排模式行为一致。** rule、agent_assist、agent_full 走同一条执行路径，同一套限制和日志格式，结果可以直接对比。`guards.py` 的模块文档写明，任务 3 的记账放在 harness 里，就是为了让规则编排器和 LLM 编排器表现相同。
- **约束不只靠提示词。** 工具白名单、探索幅度、重试上限、"还有活没干完就不许宣布成功"这些规则都在代码里强制执行。LLM 越界时，调用会被拒绝，拒绝原因作为工具结果返回给它。

涉及的文件（路径相对于仓库根目录 `g1_agent_system/`）：

| 文件 | 内容 |
|---|---|
| `g1agent/harness/registry.py` | 工具注册表：23 个工具的参数模型和描述、每个任务的工具白名单、OpenAI function-calling schema 生成 |
| `g1agent/harness/guards.py` | `Harness` 类：调用入口、预算、守卫、重试上限、执行、任务 3 记账、成功判定 |
| `g1agent/harness/world_state.py` | `WorldState` 和各估计量的数据类，harness 持有的世界状态 |
| `g1agent/harness/logger.py` | `EpisodeLogger`：每个 episode 一个目录，写 jsonl 日志和图像，并向外推送事件 |

与其它组件的接口：

| 对方 | 方向 | 接口 |
|---|---|---|
| 编排层（`g1agent/orchestrators/`） | 调用 Harness | `h.call(name, args)` 返回 `{"status","data","reason",...}`；`h.over_budget()`；`h.finished`；`h.budget`、`h.n_calls`、`h.last_approach`（agent_full 压缩历史时读取） |
| agent_full 的 LLM | 间接 | `openai_tools(TASK_TOOLS[task])` 生成交给模型的工具列表；被拒绝的调用以 `rejected: True` 的失败结果回到对话里 |
| 工具层（`g1agent/tools/toolbox.py` 的 `Toolbox`） | Harness 调用 | `_execute` 按工具名分派到 `Toolbox` 的同名方法；任务 3 记账读取 `tb.ws`（WorldState）、`tb._next_face()`、`tb._face_dirs()`、`tb.TOUR` |
| 运行脚本（`scripts/run_task.py`） | 构造 Harness | `Harness(tb, task, tcfg["budget"], log, cfg["robot"]["limits"])`，预算来自 `configs/task{1,2,3}.yaml`，限值来自 `configs/robot.yaml` |
| 视频叠加层 / 评测脚本 | 读取日志 | `EpisodeLogger(on_event=...)` 推送的事件；`tool_calls.jsonl`、`llm.jsonl`、`stages.jsonl` |

一次调用的流程：

```
编排器（rule / agent_assist / agent_full）
        │ Harness.call(name, args)
        ▼
  1 预算检查 over_budget()（finish 除外）    ──超出──► fail, budget_exhausted
  2 工具名在 TOOLS 中？                      ──否────► fail, rejected
  3 工具在 TASK_TOOLS[task] 中？              ──否────► fail, rejected
  4 pydantic 参数校验                         ──失败──► fail, rejected
  5 语义守卫 _guard()                         ──不过──► fail, rejected
  6 重试上限（按工具 / 按"工具+参数"）         ──超出──► fail, rejected
  7 _execute() → Toolbox 方法（finish → _finish 记账）
        │
  n_calls += 1；更新失败计数；EpisodeLogger.tool() 写日志（图像另存）
        │
  只对真正执行过的调用：_after() 任务 3 记账 → EpisodeLogger.stage()
```

## 当前实现

### 工具注册表 `g1agent/harness/registry.py`

**参数模型。** 每个工具对应一个 pydantic `BaseModel`，它同时用来校验参数和生成 JSON Schema：

| 模型 | 字段与默认值 | 说明 |
|---|---|---|
| `NoArgs` | 无 | |
| `DetectBoxesArgs` | `prompt: str = "cardboard box"` | 开放词表检测的目标 |
| `SelectArgs` | `selector: str`（必填） | `"left"`/`"right"`（机器人视角）、`"all"`（任务 3，所有箱子）或箱子 id |
| `BoxArgs` | `box_id: Optional[str] = None` | 默认取已选目标 |
| `ApproachArgs` | `target: Literal["root_target","stance_target"] = "root_target"` | root_target 是观察位姿（任务 1，以及任务 3 的阶段 1），stance_target 是扫码站位（任务 2、3）；字段描述里只写了任务 1/任务 2 |
| `ExploreArgs` | `direction: Literal[left,right,back,forward,turn_left,turn_right]`，`distance: float` | 平移单位为米，转向单位为度 |
| `BarcodeArgs` | `barcode_id: Optional[str] = None` | 默认取当前条码 |
| `WristTargetArgs` | `barcode_id`、`offset`，都默认 `None` | offset 省略时用配置的 stand-off |
| `MoveArmArgs` | `pose: Literal["pre","final","home"]` | |
| `BaseHeightArgs` | `height: Optional[float] = None` | 省略即站起 |
| `FinishArgs` | `task: int`，`success: bool`，`summary: str` | |

模型本身不做数值范围校验，比如 `ExploreArgs.distance` 只在 description 里写了"≤ 0.5 m / ≤ 15°"。数值上限由下文的 `_guard` 执行。

**`TOOLS`。** 这是一个字典，把工具名映射到 `(参数模型, 描述字符串)`，一共 23 个工具：`look_head`、`detect_boxes`、`select_target_box`、`fit_front_face`、`compute_obs_pose`、`approach`、`check_face_in_view`、`explore`、`detect_barcode_head`、`estimate_barcode_pose`、`compute_wrist_target`、`plan_stance`、`set_base_height`、`move_arm`、`look_wrist`、`refine_with_wrist`、`stereo_check_wrist`、`check_wrist_placement`、`plan_scans`、`go_to_next_face`、`next_barcode`、`scan`、`finish`。描述字符串就是模型看到的工具说明，其中写了调用顺序方面的提示，例如 `next_barcode` 返回 `face_done=true` 后应调用 `go_to_next_face`，再调用 `plan_scans`。描述里出现的数值（如 0.6–1.0 m、4 cm、±0.02 m）只是对 Toolbox 行为的文字说明，实现见工具层的文档。

**每个任务的工具白名单：**

```python
TASK1_TOOLS = ["look_head", "detect_boxes", "select_target_box", "fit_front_face", "compute_obs_pose",
               "approach", "check_face_in_view", "explore", "finish"]           # 9 个
TASK3_ONLY  = ["plan_scans", "next_barcode", "go_to_next_face"]
TASK2_TOOLS = [n for n in TOOLS if n not in TASK3_ONLY]                       # 20 个
TASK_TOOLS  = {1: TASK1_TOOLS, 2: TASK2_TOOLS, 3: list(TOOLS)}                # 任务 3 用全部 23 个
```

划分依据是任务定义：任务 1 只需要找到箱子并走到观察位姿；任务 2 是在任务 1 的感知和接近之后定位条码、放置腕部；任务 3 等于任务 1 加上扫描所有条码，所以开放全部工具。为什么要划分，见演进记录 v1。

**`openai_tools(names=None)`。** 遍历 `TOOLS`，`names` 不为空时只输出名单里的工具。对每个工具调用 `model.model_json_schema()` 生成参数 schema，删掉顶层和每个属性上 pydantic 自动加的 `title`，输出 `{"type":"function","function":{"name","description","parameters"}}`。agent_full 把 `openai_tools(TASK_TOOLS[task])` 交给 LLM；agent_assist 只取 `openai_tools(["explore"])` 作为决策工具之一。

### 调用入口 `Harness`（`g1agent/harness/guards.py`）

**构造 `Harness(toolbox, task, budget, logger, limits)`。**

- `task` 不在 `TASK_TOOLS` 里时抛 `ValueError`，不会悄悄退回到别的任务。
- `base_budget = dict(budget)` 和 `budget = dict(budget)` 都是副本。配置 dict 在同一进程的多个 episode 之间共享，绝不能修改；任务 3 按条码数扩展的是 `budget`。
- `fail_streak = {}`：失败计数。键有两种，一种是工具名，一种是"工具名 + 参数 JSON"。
- `finished`：调用 `finish` 后存结束信息，此前为 `None`。编排器的主循环以它为结束条件。
- 任务 3 记账字段：`stages_done`（只发一次的阶段标记）、`barcodes_ended`（已发 `barcode_end` 的条码）、`last_approach`（最近一次 `approach(root_target)` 的结果摘要，作为阶段 1 的到达信息）。
- 设置 `toolbox.task = task`，Toolbox 据此读取当前任务的配置。

**预算 `over_budget()`。** 按顺序检查三项，命中第一项就返回原因字符串：

| 条件 | 配置键（`configs/task*.yaml` 的 `budget`） | 任务 1/2/3 的基础值 |
|---|---|---|
| `n_calls >= max_tool_calls` | `max_tool_calls` | 40 |
| `ws.explore_steps > max_explore_steps`（严格大于） | `max_explore_steps` | 6 |
| 墙钟时间 `> max_wall_s` | `max_wall_s` | 1800 s |

另有 `max_retries_per_tool: 2`，供重试上限使用。这几个基础值在配置里没有写依据。

**`call(name, args)` 的步骤：**

1. **预算。** 除 `finish` 外先调 `over_budget()`。超出时返回 `{"status":"fail","reason":<原因>,"budget_exhausted":True}`。`finish` 不受预算限制：结束 episode 永远允许。
2. **工具不存在**或**不在当前任务白名单里**：返回失败并带 `rejected: True`，原因分别是 `unknown tool 'x'` 和 `tool 'x' is not part of task N`。
3. **参数校验。** 执行 `TOOLS[name][0](**args).model_dump()`，只捕获 `pydantic.ValidationError`，返回 `invalid arguments: <e.errors()>`，带 `rejected`。
4. **语义守卫 `_guard`**（见下）。
5. **重试上限。** 阈值是 `budget["max_retries_per_tool"]`（当前为 2），判断用严格大于：
   - 按工具名：`fail_streak[name] > 2` 时拒绝，原因是"{name} 连续失败 N 次，重试预算用完"；
   - 按调用：`fail_streak[name + json.dumps(parsed, sort_keys=True)] > 2` 时拒绝，原因是"这个完全相同的调用失败了 N 次……请做别的事（比如 next_barcode 换下一个）"；
   - 两项都没超才执行。因为用的是严格大于，同一工具或同一调用最多能连续真正失败 3 次，之后再调用就被拒绝。
6. **计数。** 不论结果如何，`n_calls += 1`。被拒绝和超预算的调用也算一次，所以 LLM 反复发越界调用同样会耗尽预算。
7. **更新失败计数。**
   - 结果为 fail 且不带 `rejected` 时，工具键和调用键都加 1。超预算的结果不带 `rejected`，所以也会累加。
   - 结果不是 fail（ok 或 partial）时两个键都清零。按工具名的计数在该工具任何一次成功后清零；按调用的计数只有在完全相同的调用成功时才清零。
   - 被拒绝的调用不改变计数。
8. **日志。** 记录 `{"i","tool","args","result"（去掉 image）,"seconds"}`。结果里有图像时，先用 `log.image()` 存成 jpg，再把相对路径写进记录。
9. **任务 3 记账。** 只有真正执行过的调用才进入 `_after()`。这一步放在写完工具记录之后，这样它触发的阶段事件在时间线上排在工具记录后面。

**语义守卫 `_guard(name, a)`：**

- `explore`：转向类（`turn_left`/`turn_right`）要求 `0 < distance <= max_turn_deg`，平移类要求 `0 < distance <= max_move_m`。两个限值来自 `configs/robot.yaml` 的 `limits`，当前是 15.0° 和 0.5 m。
- `finish`：`task` 参数必须等于本 episode 的任务号。
- 任务 3 的 `finish(success=true)`：先调 `_task3_work_left()`，还有剩余工作就拒绝，并提示"调用 next_barcode（它会切到下一个条码，或者报告 face_done 进入下一侧）；只有放弃时才用 success=false 结束"。放弃（`success=false`）始终允许。被拒绝的成功声明不结束 episode，episode 长度仍由调用预算限制。

`_task3_work_left()` 依次检查三项，返回第一个命中的原因：

1. `ws.scan_plan` 为空，且 `ws.stack.planned` 也不为真时，返回"还没有扫描计划"。多侧面任务里某一侧规划出 0 个条码，也算已经有计划。
2. 计划中还有 `pending` 的条码时，列出这些条码。
3. `tb._next_face()` 不为 `None`，说明巡视路线（front → right → back → left）上还有某一侧没去过。

**执行 `_execute(name, a)`：**

- `finish` 直接调 `_finish(a)`，不放在 try 里。它是记账逻辑，不是工具，出错时必须直接暴露，不能变成一次"工具失败"。
- `look_head`/`look_wrist` 的工具返回 `(r, img)`，合并为 `dict(r, image=img)`；`approach` 和 `compute_wrist_target` 按位置传参；其余工具用 `getattr(tb, name)(**a)` 调用。
- 工具内部异常用 `except Exception` 捕获，这是 harness 里唯一一处宽泛捕获，目的是把工具错误变成带完整堆栈的结构化失败 `{"status":"fail","reason":"<类型>: <消息>","traceback":...}`，交给调用方并写进日志。agent_full 把结果交给 LLM 前会去掉 `traceback`。
- 例外是基础设施错误。模块级函数 `_infrastructure_error(e)` 对 `MemoryError`、`ConnectionError`、`TimeoutError` 和 `torch.cuda.OutOfMemoryError` 返回真，这类异常会重新抛出，episode 带着错误结束。它们是机器或仿真连接层面的问题，agent 没法靠换个工具解决。

### 任务 3 的记账

任务 3 是"任务 1 + 按固定顺序扫描所有条码"，可能还要绕箱堆四个侧面。记账在 `_after()` 和 `_finish()` 里完成，规则编排器和 LLM 编排器共用同一套。

**动态预算。** 每次 `plan_scans` 成功后，若 `base_budget` 里有 `per_barcode`，对其中每个键 k：

```
budget[k] = base_budget[k] + n_planned × per_barcode[k]
```

`configs/task3.yaml` 中 `per_barcode` 的取值和依据：

| 键 | 值 | 依据（配置注释） |
|---|---|---|
| `max_tool_calls` | 28 | rule 模式任务 2 实测最多 21 次调用（57 个 episode），加 `next_barcode` 1 次、`scan` 1 次、一次无读数重试 3 次（refine、final、scan）、前后两次回 home 2 次 |
| `max_wall_s` | 600 s | 同样 57 个 episode 中任务 2 最长 518 s（p90 为 324 s），加 scan 重试和两次回 home（各约 3 s），向上取整 |

因为每次 `plan_scans`（包括后来补充新标签的调用）都按累计的计划条码数重新计算，所以预算随看到的条码增加而增长。

**阶段事件（写入 `stages.jsonl`）：**

| 事件 | 触发时机 | 信息 |
|---|---|---|
| `task1_end` | 第一次成功的 `plan_scans`（带 `updated` 的补充规划不算）；`finish` 时若还没发过则补发，附 `note: "finish before plan_scans"` | 到达信息 `last_approach`、计划条码数、扩展后的预算 |
| `face_start` | 多侧面任务中某一侧第一次规划（`first_for_face`） | 侧面 id、该侧法向（`tb._face_dirs()`，里程计坐标系，3 位小数）、该侧计划条码数 |
| `barcode_start` | `next_barcode` 切到一个新条码 | 条码 id、序号 |
| `barcode_end` | 条码有了结局：`scan` 后状态变为 scanned / duplicate / unreadable；`plan_stance` 报告 unreachable；`next_barcode` 关闭了未完成的当前条码（failed）；`finish` 时当前条码仍是 pending（改为 failed） | id、状态、内容，有失败原因时附 `reason`；每个条码只发一次 |
| `face_end` | 多侧面任务中 `next_barcode` 返回 `face_done` 或 `done` | 侧面 id、该侧各条码状态 |
| `scan_done` | `next_barcode` 返回 `done`，或 `finish` | 已读内容列表、各条码状态；只发一次 |

**失败计数的清零时机。** `next_barcode` 真正切到下一个条码时清空整个 `fail_streak`，所以重试次数按条码计算。返回 `face_done`（换侧）或 `done` 时不清零，这样 `go_to_next_face` / `plan_scans` 反复失败的循环会碰到重试上限。

**结束 `_finish(a)`。** 任务 3 中，成功与否按账本判定，不采信调用方的声明：

```
success = 至少一个条码状态为 "scanned"
          且 所有计划条码的状态都属于 {"scanned", "unreadable"}
          且 tb._next_face() is None        （多侧面任务的巡视已走完）
```

- `unreadable` 表示扫码枪在近距离、标签完整在视野内的情况下仍然解不出任何码（例如贴纸假码）。这是一个结果，不算机器人失败。
- `failed`、`unreachable`、`pending` 都不算成功。
- 返回的结束信息里另外保存 `claimed_success`（调用方声称的值）、`success`（账本判定）、`scanned`（读到的条码列表）和 `plan_status`。

任务 1、2 的 `finish` 直接采用调用方给的 `success`，episode 是否真正成功由独立的评测器判断。

### 世界状态 `g1agent/harness/world_state.py`

所有估计量都放在里程计（odometry）坐标系下，并带估计时刻。这样底盘移动之后，之前的估计仍然可用。

- **`BoxEstimate`**：`id`、最近一次检测框 `bbox`（头部图像像素）、`score`、`t`，以及 `face`：`{"center","normal","right","up","corners"(4×3),"size"}`。
- **`BarcodeEstimate`**：`id`、`box_id`、`center`、`normal`、`bar_dir`、`source`（head / wrist / stereo）、`t`、`pixel`。任务 3 新增的字段有：
  - `face`：所在箱面的副本（`detect_boxes` 会重新分配箱子 id，所以要存副本）；
  - `corners_odom`、`corners_px`；
  - `width_m`、`width_source`、`width_support`：标签宽度，来自头部时是检测框宽度（上界），来自腕部时是解码出的符号宽度；
  - `status`（默认 `pending`）、`fail_reason`、`wrist_read`、`plan_center`、`content`、`order_index`；
  - `face_id`：0 front、1 right、2 back、3 left。
- **`WorldState`**：
  - 估计量：`boxes`、`barcodes`；
  - 目标：`target_selector`、`target_box_id`、`root_target`、`wrist_target`（头部估计）、`wrist_target_refined`（腕部估计）、`stance_target`；
  - 计数：`explore_steps`（预算检查要用）、`history`；
  - 任务 3：`scan_plan`（有序条码 id）、`scan_ref_T` / `scan_ref_Ts`（排序时用的头部位姿）、`stack`（多侧面箱堆模型，含 `faces`、`current`、`visited`、`multi_face`、`planned`）、`current_barcode_id`，以及首次读取账本 `scanned`（`[{"content","barcode_id","order_index","t"}]`，每种内容只记第一次）。
  - 腕部和站位相关的目标总是属于当前条码，`Toolbox.next_barcode` 切换条码时会清空它们。
- 方法：`scan_plan_export()` 按计划顺序导出每个条码的位置、宽度、状态、内容、侧面；`summary()` 输出整个世界状态的可序列化摘要（数值取整）。

### Episode 日志 `g1agent/harness/logger.py`

`EpisodeLogger(ep_dir, on_event=None)` 在每个 episode 的目录下写这些文件：

| 文件 | 写入方法 | 内容 |
|---|---|---|
| `tool_calls.jsonl` | `tool(rec)` | 每次工具调用一条，由 `Harness.call` 写入 |
| `llm.jsonl` | `llm(rec)` | 每次 LLM 往返一条（含提示词版本号、每次尝试的记录，被截断的尝试也记录） |
| `stages.jsonl` | `stage(name, info)` | 任务 3 的阶段事件 |
| `images/{序号:03d}_{tag}.jpg` | `image(rgb, tag)` | 工具返回的图像和发给 LLM 的图像 |
| 任意 `.json` | `write(name, obj)` | 结果汇总等 |

- 每条 jsonl 记录自动加上 `t_wall`（相对 logger 创建时刻的秒数）、`t_unix` 和 `log_seq`（跨所有 jsonl 文件的写入序号，时间相同时按它排序）。
- 序列化只认识 numpy 数组和 numpy 标量，其他不认识的类型直接抛 `TypeError`，不会悄悄转成字符串。
- `on_event` 回调用来给 episode 视频的叠加层喂数据，logger 本身从不与仿真器通信。工具事件附带 `tool_overlay(name, result)` 提取的像素空间信息（检测框、标签角点、腕部解码多边形、扫描计划状态等），而且只使用结果数据里实际存在的检测，数据里没有的键叠加层里也没有。
- 记账类的 LLM 记录（例如 agent_assist 中决策参数无效）没有 `reply`，不推送事件。

## 演进记录

### v0 · 2026-11-05 · 初版：WorldState、工具注册表和统一入口 Harness.call

**初版设计。** 系统搭建当天先写了 harness 的三块：

- `world_state.py`：`BoxEstimate` 包含 id、检测框、分数、时间和 `face`（里程计坐标系下的 center/normal/right/up/corners/size）；`BarcodeEstimate` 包含 id、box_id、center、normal、bar_dir、来源（head/wrist/stereo）、时间和像素位置；`WorldState` 保存箱子、目标选择、条码、`root_target`、`wrist_target`、`wrist_target_refined`、`stance_target`、`explore_steps`、`history`，并提供 `summary()`，把数值取整后交给 LLM。所有估计放在里程计坐标系下，底盘移动后仍然可用。
- `registry.py`：每个工具一个 pydantic 参数模型（`DetectBoxesArgs` 的 prompt 默认 `"cardboard box"`，`SelectArgs` 的 selector 当时只有 left/right 或箱子 id），并由模型自动生成 OpenAI function-calling schema。只有任务 1、2 的工具。
- `guards.py`：规则编排和 LLM 编排都通过同一个 `Harness.call()` 调用工具，每次调用依次经过参数校验、限制检查、预算检查和写日志（`tool_calls.jsonl`）。因此三种模式的日志格式完全一样，可以直接比较。开发日志记载初版设计中还有"前置条件检查"一项。在当前的 `guards.py` 中，前置条件检查不是独立步骤，工具自身的前置条件（例如任务 3 中 `move_arm` 要求已到达当前条码的站位）在 Toolbox 内检查。

初版里后来出问题的几处：

- 没有按任务区分工具，`openai_tools()` 输出全部工具；
- 构造函数直接保存传入的预算 dict（`self.budget = budget`），没有复制；
- `finish` 也要过预算检查；
- 失败计数只按工具名计算。

### v1 · 2026-11-06 · 按任务限定工具集，Harness 拒绝当前任务以外的工具

**现象。** agent_full 跑任务 1 的演示 episode（`demo_agent_gear/agent_full/t1_L0_0000`）时，编号 4 的调用（从 0 计）`approach` 失败后，LLM 转去调用 `detect_barcode_head`、`estimate_barcode_pose`、`move_arm` 等任务 2 的工具，把任务 1 做偏了。评测器只检查底盘是否到达观察位姿，所以这次被误判为成功。

**根因。** `run_agent_full` 调用 `openai_tools()` 时没有传名单，所有工具（包括条码和手臂工具）都进了 LLM 的工具列表。当时的 `task1_agent_full` 提示词（v1）写明了任务 1 的目标，但没有限制可用工具；Harness 也不检查工具是否属于当前任务。

**修改。**

- `registry.py` 新增 `TASK1_TOOLS`（9 个：`look_head`、`detect_boxes`、`select_target_box`、`fit_front_face`、`compute_obs_pose`、`approach`、`check_face_in_view`、`explore`、`finish`），以及 `TASK_TOOLS = {1: TASK1_TOOLS, 2: list(TOOLS)}`。依据是任务定义：任务 1 只是找箱子并走到观察位姿。
- `agent_full.py` 改为只把 `openai_tools(TASK_TOOLS[h.task])` 交给 LLM。
- `guards.py` 的 `call()` 增加一道检查：工具名不在 `TASK_TOOLS[self.task]` 中就拒绝执行。这样即使 LLM 越界调用，也会被代码挡住，不只靠提示词约束。

**效果。** 单元测试 11 个全部通过。之后的 agent_full 批量评测以新进程启动，用的是修复后的代码。

### v2 · 2026-11-06 · 支持任务 3：动态预算、阶段事件、finish 不受预算限制、预算改为副本

这一版是任务 1、2 合并为任务 3 时做的，同时修了一个预算耗尽后死循环的问题。

**现象。** agent_assist 批量评测中，episode `m5_assist_gear_4/t1_L1_box_shape_0007` 的 `tool_calls.jsonl` 有 2379 条记录。最后几条都是 `approach`/`explore` 失败，原因是 `tool-call budget (40) exhausted`，此时墙钟时间已到约 1892 s。规则编排器 `rule.py` 的文档记载，这个 episode 在预算用完后又调用了 1195 次 `approach`。

**根因（两层）。**

- 编排层：规则编排器的恢复循环收到失败结果后继续 explore / retry，不检查结果里的 `budget_exhausted`。这部分在编排层修复（`rule.recover` 在预算耗尽或调用被拒绝时立即返回，见编排层文档）。
- Harness：预算耗尽后，连 `finish` 也被预算检查拒绝，`h.finished` 一直是 `None`，而编排器的主循环正是以 `h.finished is None` 为条件，所以停不下来。

**修改（`guards.py`）。**

- `finish` 跳过预算检查：结束 episode 永远允许。`finish` 也不放进工具的 try/except，它的错误直接暴露。
- 未知任务号在构造时抛 `ValueError`。
- 预算改为复制：`base_budget = dict(budget)`、`budget = dict(budget)`。配置 dict 在多个 episode 之间共享，任务 3 要按 episode 改写预算，不复制就会互相污染。
- 任务 3 的记账：
  - `plan_scans` 成功后预算变为"基础值 + 每码增量 × 计划条码数"，并发一次 `task1_end`；
  - `next_barcode` 时重置失败计数，并发 `barcode_end` / `barcode_start` / `scan_done`；
  - `finish` 时补发缺失的阶段事件，把未完成的当前条码标为 failed，按账本计算 success。

**修改（`registry.py`、`world_state.py`）。**

- 注册 `plan_scans`、`next_barcode`、`scan`。
- `TASK_TOOLS` 改为：任务 2 是除 `plan_scans`、`next_barcode` 之外的全部工具，任务 3 是全部工具（当时 21 个）。
- `FinishArgs` 接受任务 1/2/3；`WristTargetArgs.offset` 改为可选（任务 3 的 stand-off 按标签宽度自适应）。
- `BarcodeEstimate` 增加 `face`、角点、`width_m`、`status`、`content`、`order_index`。
- `WorldState` 增加 `scan_plan`、`scan_ref_T`、`current_barcode_id`、`scanned` 账本和 `scan_plan_export()`。

**每码预算的取值依据。** 每码 28 次调用、600 s，依据是 rule 模式下任务 2 实测最多 21 次调用、最长 518 s，再加上扫码相关的几次调用（明细见"当前实现"中的表）。

**效果（后续运行中的 `task1_end` 记录）。** 扩展后的预算与公式一致：

| 运行 | 计划条码数 | `task1_end` 中的预算 |
|---|---|---|
| 2026-11-07 早间的任务 3 运行 | 2 | `max_tool_calls` 96、`max_wall_s` 3000 |
| 第 8 轮 stack4f（四侧箱堆） | 3 | `max_tool_calls` 124、`max_wall_s` 3600 |
| af4c_2side_v2（两侧纸箱堆） | 7 | `max_tool_calls` 236 |

第 8 轮时曾怀疑多侧面任务的预算不够。核对代码后确认，每次 `plan_scans` 都会按累计的计划条码数重算预算，四侧都规划完后墙钟上限会达到 9000 s，所以不是问题。

2026-11-07 有一条需求：扫码一直读不出时要有截止。核对后，现有设计已有三层上限：

1. `scan` 工具内部最多重试 1 次（`retries_on_no_read: 1`），之后返回失败；
2. 同一条码上同一工具连续失败超过 `max_retries_per_tool`（2）次后，调用被 Harness 拒绝，换码时清零；
3. 全局预算（基础 40 次 + 每码 28 次 × 条码数，外加墙钟上限），用完强制结束。

同时指出一个缺口：第 3 层预算由各条码共用，LLM 在一个码上浪费的调用会挤占后面条码的份额。是否改为每码单独计预算，尚待决定，当前代码没有改。

### v3 · 2026-11-07 · 侧面级阶段事件，只在多侧面任务中记录

**背景。** 任务 3 加入了绕箱堆四个侧面的多侧面巡视，新增工具 `go_to_next_face`。视频剪辑和评测需要知道每个侧面何时开始、何时结束。

**初版改动和遇到的问题。** `guards.py` 在一个侧面第一次规划时记 `face_start`，在 `next_barcode` 返回 `face_done`/`done` 时记 `face_end`。第一版无条件记录，结果单侧面任务的阶段序列测试失败：`scan_done` 前面多了一个 `face_end`，报 `AssertionError: 'face_end' != 'scan_done'`。

**修改。**

- 只在 `ws.stack.multi_face` 为真时记录这两个事件。
- `registry.py` 中 `next_barcode` 的描述改为：只处理箱堆当前这一侧，这一侧做完且任务覆盖所有侧面时返回 `face_done=true`（随后调用 `go_to_next_face`，再调用 `plan_scans`）。
- `go_to_next_face` 加入 `TASK3_ONLY`。加上此前为下蹲动作新增的 `set_base_height`，任务 3 的工具数变为 23 个（任务 2 为 20 个）。
- 提示词 `prompts/task3_agent_full.md` 同步加入相应规则。

**效果。** 修复后 73 个测试通过。

### v4 · 2026-11-07 · 基础设施错误直接结束 episode；换侧时不清零失败计数

这两处修改都来自 agent_full 第 7 轮运行（场景为 stack9p 和 stack4f）：问题 1 出现在这一轮的第一次启动，问题 2 出现在修复问题 1 之后的重跑（`runs/af_round7`）。

**问题 1：CUDA 显存不足被当成工具失败。**

- 现象：stack9p 的 episode 一开始，`detect_boxes` 就连续报 `OutOfMemoryError: CUDA out of memory`（第 0 和第 3 次调用）。这些错误被 `_execute` 的 `except Exception` 包装成普通工具失败交给 LLM。LLM 无从处理，只能换工具、换参数反复尝试；代码注释记载它又调用了 40 次无用的工具。
- 根因：harness 不区分"任务里的失败"（agent 可以换个做法）和"机器层面的失败"（显存、内存耗尽，到仿真服务器的连接断开或超时），把后者也交给了 agent。显存不足本身的起因是运行客户端的检测器没有固定在分配给本项目的 GPU 上，这一点在运行脚本中单独修复。
- 修改：新增 `_infrastructure_error(e)`，对 `MemoryError`、`ConnectionError`、`TimeoutError`、`torch.cuda.OutOfMemoryError` 返回真，`_execute` 遇到这类异常时重新抛出，episode 带着错误立即结束。普通工具错误（如 `ValueError`）仍以结构化失败返回，让 agent 自己处理。
- 效果：新增测试 `test_infrastructure_errors_end_the_episode_instead_of_a_tool_failure`，共 94 个测试通过。

**问题 2：换侧时清零失败计数，失败循环停不下来。**

- 现象：重跑中的 stack4f episode 在正面 3 个条码中扫到 2 个（另 1 个判为不可达）后，`go_to_next_face` 第一次绕到右侧面就失败了（没能拟合出右侧面）。之后"`next_barcode`（face_done）→ `go_to_next_face` 失败 → `plan_scans` 失败"反复循环，约 70 次调用后被手动停止。代码注释把它记为"一直循环到预算用完"。
- 根因：`_after` 在每次 `next_barcode` 成功时都清空 `fail_streak`，包括返回 `face_done` 的情况。于是"`next_barcode`（face_done）→ `go_to_next_face` 失败 → `plan_scans` 失败 → `next_barcode`"这样的循环每一圈都会把失败计数清零，重试上限永远不会触发。
- 修改：只有 `next_barcode` 真正切到下一个条码时才清零，返回 `face_done` 或 `done` 时保留计数。这样换侧失败的循环会碰到重试上限。规则编排器里也写明，`plan_scans` 被拒绝说明它已到重试上限，重做阶段 1 也无法让它再执行，所以直接结束。
- 效果：本轮全部修复后测试通过（开发日志记为 95 个）。

### v5 · 2026-11-07 · 拒绝与账本矛盾的 finish(success=true)，成功由账本判定且要求走完巡视

**现象。** 第 9 轮 stack4f（四侧箱堆），LLM 只做完 4 个侧面中的 2 个，就以 `finish(success=true)` 结束了 episode。

**根因（两条）。**

1. "要扫描全部四个侧面"写在开局的 user 消息里，agent_full 第一次压缩历史时把这条消息删掉了，LLM 不再知道任务要覆盖所有侧面。这部分在编排层修复：压缩摘要的首行原样写回任务陈述（见编排层文档）。
2. Harness 接受了一个与实际状态矛盾的成功声明。LLM 的扫描计划只包含已经看到的侧面，计划里的条码全部扫完，不代表任务完成。

**修改（`guards.py`）。**

- `_guard` 新增：任务 3 中 `finish` 带 `success=true` 时调用 `_task3_work_left()`。返回 "no scan plan yet"、"barcode(s) X still pending" 或 "the {side} side of the stack has not been visited" 中任一原因，就拒绝这次调用（标记 `rejected`），并提示 LLM 调用 `next_barcode`。被拒绝的声明不结束 episode；主动放弃（`success=false`）仍然允许。
- `_finish` 的成功判定改为：所有状态都是 scanned，并且 `tb._next_face() is None`（巡视已完成）。调用方的声明另存为 `claimed_success`。
- 旧测试 `test_finish_without_plan_emits_task1_end_and_fails` 改为先断言 `success=true` 被拒绝，再用 `success=false` 结束。
- 新增测试 `test_finish_success_refused_while_sides_remain_and_agent_full_keeps_the_task_statement`（用一个四侧面的假 Toolbox）。

**效果。** 逻辑测试 42 个通过；全量测试 99 个通过、1 个跳过。

### v6 · 2026-11-07 · face_start 的法向改为从 `_face_dirs()` 读取

**现象。** 箱堆模型改到箱堆自身坐标系（S 坐标系）之后，`face_start` 事件里的 `normal_odom` 变成了 `null`。

**根因。** 重构后 `ws.stack["faces"][fid]` 里不再存 `normal`，`guards.py` 仍然用 `(st.get("faces") or {}).get(fid, {}).get("normal")` 读取，得到 `None`。

**修改。** 改为 `self.tb._face_dirs()[fid] if fid in faces else None`，即重新配准后里程计坐标系下的侧面法向。

**效果。** `test_task3_logic` 中 35 个测试通过。

### v7 · 2026-11-07 · 空侧面算有计划；新增"无可读码"（unreadable）结果并计入成功

**空侧面。** 两侧纸箱堆场景中，有的侧面拟合到了箱面但没有任何标签。Toolbox 的 `plan_scans` 改为在多侧面模式下对这种侧面返回 ok、0 个条码，并置 `ws.stack["planned"]`。Harness 的 `_task3_work_left()` 相应地把 `ws.stack.planned` 为真也视为"已有计划"，避免因为 `scan_plan` 为空而判为"还没有扫描计划"。对应测试为 `test_stack_side_without_labels`：多侧面时，空侧面之后巡视继续，`finish(success)` 仍要求其他侧面完成；单侧面任务中，同样的视角判为失败（根本没有可扫的东西）。

```{figure} images/eval_stack4c_stickers.jpg
:width: 90%

两侧纸箱堆场景的正面。箱面上既有真条码，也有印在快递单上的条码图案（贴纸假码）；后者在扫码枪近距离完整视野下也解不出码。
```

**无可读码。**

- 现象：带快递单贴纸的纸箱堆上，有些"标签"是贴纸上的假码，扫码枪在近处看得到却解不出来。按原来的判定，这些标签最后是 `failed`，整个 episode 被算成机器人执行失败，但机器人其实已经做了能做的一切。
- 根因：成功判定把"标签本身不可读"和"机器人没能把扫码枪放好"混为一类。
- 修改：
  - Toolbox 的 `scan` 新增 `unreadable` 判定：扫码枪在 stand-off 容差内、整张标签在视野内、重试后仍解不出任何码，而且此前腕部相机也从未在这张标签上解出过内容。判定时写明 `fail_reason`。
  - Harness 的 `_finish` 成功判定改为"至少一个 scanned，且所有状态属于 {scanned, unreadable}，且巡视已走完"。
  - 规则编排器的成功判定同步改为 `n_ok > 0 and n_ok + n_unr == len(scan_plan)`；提示词 v7 加入"unreadable 是一个结果，不是失败"。
- 效果：之后的运行（af4c_2side_v2 起）中，贴纸假码被记为 `unreadable`，例如 af4c_2side_v6 正面 7 个标签的结果为 3 个 scanned、4 个 unreadable。

### v8 · 2026-11-07 · 失败计数同时按工具和按"工具+参数"计算

**现象。**

- 两侧纸箱堆的 v5 运行中，背面出现多个 failed/unreachable，运行被停止。
- v6 运行（`af4c_2side_v6`）在右侧面的 bc7 开始后上下文溢出，episode 中止。回看日志，`move_arm('pre')` 一共失败了 14 次，但重试上限 `max_retries_per_tool` 一次也没有触发。

**根因。**

- 重试上限的计数被清零了。`fail_streak` 只按工具名计数，任何一次该工具的成功调用都会清零，而每两次失败的 `move_arm('pre')` 之间都有一次成功的 `move_arm('home')`。
- 另外，LLM 跳过了 `next_barcode`，直接对下一个条码操作，陷入循环，直到上下文溢出。

**修改。**

- `guards.py`：
  - 新增 `import json`，在 `call()` 中初始化 `parsed = None`；
  - 以 `ck = name + json.dumps(parsed 或 args, sort_keys=True)` 作为"同一调用"的键，失败时工具键和调用键同时加 1，成功时各自清零；
  - 调用键的连续失败超过 `max_retries_per_tool` 时直接拒绝，返回 "this exact {name} call failed N times; retry budget exhausted. Do something else (e.g. next_barcode to move on)"。
  - 沿用已有的 `max_retries_per_tool` 配置，没有新增阈值。
- Toolbox（工具层）：任务 3 中 `compute_wrist_target`、`refine_with_wrist` 只接受当前条码，传入别的条码时拒绝，并提示先调用 `next_barcode`。扫码顺序由 `next_barcode` 决定，LLM 不能跳着做。

**效果。** 57 个相关测试通过；完整测试套件 120 个通过、1 个跳过。

### v9 · 2026-11-07 · 补发"无可读码"的 barcode_end 事件

**现象。** 剪辑 v7 运行的视频时，`scripts/make_barcode_clips.py` 一个"无可读码"片段都没有生成。查 `stages.jsonl`，状态为 unreadable 的 `barcode_end` 事件数为 0。

**根因。** 条码被判为 unreadable 时，`scan` 调用本身以失败结束（没有读到码）。而 `_after` 在任务 3 中遇到非 ok 结果会提前返回，原有 `scan` 分支里对 unreadable 调用 `_end_barcode` 的逻辑根本走不到。

**修改。** 在 `_after` 提前返回之前加一段：任务 3、工具为 `scan` 且结果不是 ok 时，若当前条码状态为 `unreadable`，调用 `_end_barcode(bc.id, "unreadable")`。`_end_barcode` 本身按条码去重，所以不会重复发事件。

**效果。** 52 个相关测试通过。v7 运行本身没有自动生成的片段，改为按 `next_barcode` 的时间戳手动剪出了 bc1（贴纸）的判定过程。

## 已知限制

- **预算由各条码共用。** 任务 3 的调用预算和墙钟预算是"基础值 + 每码增量 × 条码数"的总量，不是每码单独计算。LLM 在一个条码上浪费的调用会挤占后面条码的份额。是否改为每码单独计预算尚未决定。
- **基础预算没有写明依据。** `max_tool_calls` 40、`max_explore_steps` 6、`max_retries_per_tool` 2、`max_wall_s` 1800 在配置中没有注释依据；只有任务 3 的每码增量（28 次 / 600 s）来自实测。
- **状态枚举不一致。** `world_state.py` 的 `BARCODE_STATUSES = ("pending","scanned","duplicate","failed")` 里没有 `unreachable` 和 `unreadable`，`BarcodeEstimate.status` 的字段注释里也没有 `unreadable`，但 `guards.py` 和 Toolbox 实际都在使用这两种状态。`BARCODE_STATUSES` 在 harness 中没有被引用，所以目前不影响运行，但它不是可靠的状态清单。
- **参数模型没有数值范围。** 例如 `ExploreArgs.distance` 的上限只写在 description 里，数值限制由 `_guard` 根据 `configs/robot.yaml` 执行；其他工具的数值参数（如 `offset`、`height`）在 harness 层没有范围检查，由 Toolbox 处理。
- **任务 1、2 的 `finish` 采信调用方。** 只有任务 3 用账本判定成功；任务 1、2 的 `success` 来自调用方，真实结果要看独立评测器。v1 的现象说明评测器本身也可能误判（当时只检查底盘位置）。
- **重试上限是"连续失败次数"。** 按工具名的计数在任何一次成功后清零，按调用的计数在参数稍有不同时就是另一个键。LLM 如果每次都换一点参数，重复失败仍然不会触发上限，只能靠调用预算兜底（推测：v8 之后的运行中尚未见到这种情况的记录）。
- **`except Exception` 仍然存在。** `_execute` 用它把工具内部错误变成结构化失败，只有列出的四类基础设施错误会重新抛出。其他属于"机器层面"的异常如果不是这四类的子类，仍会被当作工具失败交给 agent（推测：例如通信库自定义、且不继承 `ConnectionError` 的异常类型）。
