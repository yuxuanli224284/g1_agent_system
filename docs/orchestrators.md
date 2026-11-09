# 编排模式与提示词

编排层（orchestrator）回答的问题只有一个：**下一步调用哪个工具**。系统里有三种编排模式，它们共用同一套工具和同一个执行入口 `Harness.call(name, args)`，区别只在"谁来做决定"：

- **rule**：固定状态机，不使用 LLM；
- **agent_assist**：仍由 rule 状态机跑任务，只在两个卡住的地方（找不到目标、工具失败）问 LLM；
- **agent_full**：LLM 通过 tool calling 决定每一步。

提示词（`prompts/*.md`）是 LLM 两种模式的系统提示和决策提示，和编排代码一起演进，所以放在同一页里讲。

本页涉及的文件（路径相对于仓库根目录 `g1_agent_system/`）：

| 文件 | 作用 |
|---|---|
| `g1agent/orchestrators/common.py` | 共享辅助函数：读提示词并解析版本号、组装图像消息、记录 LLM 交互 |
| `g1agent/orchestrators/rule.py` | rule 模式：`Task1` / `Task2` / `Task3` 状态机、`RuleExplore`、`rule_recovery`、`recover()` |
| `g1agent/orchestrators/agent_assist.py` | agent_assist 模式：把 LLM 决策函数插进 rule 状态机的两个插口 |
| `g1agent/orchestrators/agent_full.py` | agent_full 模式：LLM 主循环、图像裁剪、任务 3 的上下文压缩 |
| `prompts/recovery.md` | agent_assist：工具失败后的决策提示（v2） |
| `prompts/task1_agent_assist_explore.md` | agent_assist：探索决策提示（v2，任务 1/2/3 共用） |
| `prompts/task1_agent_full.md` | agent_full 任务 1 系统提示（v2） |
| `prompts/task2_agent_full.md` | agent_full 任务 2 系统提示（v2） |
| `prompts/task3_agent_full.md` | agent_full 任务 3 系统提示（v7） |
| `configs/task1.yaml`、`task2.yaml`、`task3.yaml` | 编排用到的探索序列、重试距离、放置迭代次数和预算 |

## 作用与在系统中的位置

```
                 ┌──────────── rule.py ────────────┐
   rule 模式 ──► │ Task1 / Task2 / Task3 状态机     │
                 │  插口 explore_policy  ◄──────────┼── RuleExplore（固定序列）
                 │  插口 recovery_policy ◄──────────┼── rule_recovery（重试一次）
                 └──────────────┬──────────────────┘
agent_assist ──► 同一状态机，两个插口换成 LLM 决策（_decide → LLMClient.chat, tool_choice="required"）
                                │
agent_full ──► LLM 主循环（LLMClient.chat, tool_choice="auto"，工具 = TASK_TOOLS[task]）
                                │
                                ▼   三种模式都只走这一个入口
                        Harness.call(name, args)
                 （白名单、参数校验、预算、重试上限、执行、日志、任务 3 记账）
                                │
                                ▼
                     Toolbox → 仿真器 / 机器人
```

和其它组件的接口：

| 对方 | 方向 | 内容 |
|---|---|---|
| `g1agent/harness/guards.py`（`Harness`） | 编排层调用 | `h.call(name, args)` 执行动作；`h.over_budget()` 查预算；`h.finished` 判断 episode 是否结束；`h.last_approach`、`h.n_calls`、`h.budget` 用于 agent_full 的压缩摘要 |
| `g1agent/harness/registry.py` | 编排层读取 | `TASK_TOOLS[task]` 每个任务允许的工具；`openai_tools(names)` 生成 OpenAI 格式的工具 schema；`ExploreArgs` 用于校验 LLM 给出的探索动作 |
| `g1agent/llm/client.py`（`LLMClient`） | LLM 模式调用 | `llm.chat(messages, tools, tool_choice)` 返回解析好的工具调用；`llm.image_part(rgb)` 编码图像 |
| `g1agent/harness/logger.py`（`EpisodeLogger`） | 编排层写入 | `log.llm(...)` 写 `llm.jsonl`（含提示词版本号）；`log.image(...)` 存图 |
| `g1agent/tools/toolbox.py`（`h.tb`） | 少量直接调用 | `look_head()` 取图给 LLM；`remember_home()` 记录手臂初始位姿；`adopt_stereo()` 采用运动立体结果 |
| `scripts/run_task.py` | 调用编排层 | 按 `--mode` 选 rule 类（脚本里自己的 `{1: Task1, 2: Task2, 3: Task3}[task]`）`.run()`、`run_agent_assist` 或 `run_agent_full`；设置 `tb.multi_face` |
| `configs/task*.yaml` | 编排层读取 | 探索序列、前进距离、放置迭代、预算（见下文常量汇总） |

设计上的分工是：**数值全部由工具计算，编排层只决定调用顺序**。LLM 不输出任何坐标，提示词和 agent_assist 的决策工具都体现了这一点。

## 当前实现

### 三种模式一览

| 模式 | 入口 | 谁做决策 | LLM 的作用 |
|---|---|---|---|
| rule | `rule.py`：按任务号选 `Task1` / `Task2` / `Task3`（`run_task.py` 用自己的字典，`agent_assist` 用 `task_class(task)`），再 `.run()` | 固定状态机 | 不使用 |
| agent_assist | `agent_assist.py::run_agent_assist` | rule 状态机 | 只在两处被询问：(1) 目标缺失或无法识别时选一个探索动作；(2) 工具失败后选 retry / explore / give_up |
| agent_full | `agent_full.py::run_agent_full` | LLM（tool calling） | 每一步都由它选；任务 3 的扫码顺序仍由工具（`plan_scans` / `next_barcode`）固定，LLM 不能选 |

三种模式在同一个任务 3 场景（`t1_L0_0000`，桌上两个箱子各一个条码）上的一次对比，取自 2026-11-06 任务 3 首轮（agent_full 一行是修复后的重跑，见演进记录 v7）：

| 模式 | 扫到 | 顺序 | 任务 1 阶段到位 | 枪尖误差 | 用时 | LLM 调用 |
|---|---|---|---|---|---|---|
| rule | 2/2 | 正确 | 差 6.9 cm 未达标 | 3.9 cm | 542 s | 0 |
| agent_assist | 2/2 | 正确 | 达标 | 3.6 cm | 553 s | 0 |
| agent_full（修复后重跑） | 2/2 | 正确 | 达标 | 3.9 cm | 565 s | 28，截断 0 次 |

agent_assist 这一轮没有遇到需要 LLM 的情况，所以 LLM 调用为 0。

### `common.py`：共享辅助函数

**`load_prompt(name, **kw) -> (text, version)`**

1. 读取 `prompts/{name}.md`（根目录取自 `config.ROOT`）。
2. 首行去掉 `<!--` 和 `-->` 并 strip，作为版本标签，例如 `prompt: task3_agent_full v7`。
3. 其余行 strip 后作为正文，用 `str.format(**kw)` 填入占位符。

因为用 `str.format`，正文中的 `{...}` 都被当作占位符，提示词里不能出现字面的花括号。多传的关键字参数不会报错（例如任务 1/2 也会收到 `max_explore_steps`，任务 2 的提示词里并没有这个占位符）。

**`log_llm(log, kind, version, messages, reply)`**

把一次 LLM 往返写入 `llm.jsonl`。消息里的图像部分替换成 `{"type": "image", "note": <图像文件路径>}`，日志里只记路径不记 base64。记录字段：`kind`、`prompt_version`、`messages`，以及回复中的 `content`（模型写的一句理由）、`reasoning_content`（服务端返回时才有）、`tool_calls`、`finish_reason`、`usage`、`t_start`、`t_end` 和 `attempts`。`attempts` 包括被截断后重抽的那些尝试，所以截断率可以从日志统计。

**`image_message(llm, log, rgb, text, tag)`**

用 `llm.image_part(rgb)` 编码图像，同时用 `log.image(rgb, f"llm_{tag}")` 把图存成文件，文件路径写进私有键 `_path`。返回一条 `role="user"`、内容为 `[文本, 图像]` 的消息。

**`clean(messages)`**

发送前删除私有键 `_path`，返回新列表，不改原消息。

**`tool_result_text(res)`**

工具结果转 JSON 文本（`ensure_ascii=False`），去掉 `image` 和 `traceback`。图像另外以图像消息给 LLM；堆栈只进日志，不给 LLM。

### `rule.py`：规则状态机

模块约定：每个动作都经过 `Harness.call`，和 LLM 编排器一样；各阶段（stage）只返回结果，只有 `run()` 调用 `finish`，而且 `task` 参数总是本 episode 的任务号 `h.task`。这样阶段可以在任务 3 中复用：`Task1.stage()` 负责寻找和接近，`Task2.place_and_scan(barcode_id)` 负责腕部放置和扫码。

#### 两个插口：`RuleExplore` 与 `rule_recovery`

`RuleExplore(seq)`：构造参数是探索序列 `[(direction, distance), ...]`，每调用一次返回下一项 `{"direction", "distance"}`，序列用完返回 `None`（状态机把 `None` 当作"探索不可行"）。rule 模式用 `configs/task1.yaml: explore_sequence`：

| 序号 | 动作 |
|---|---|
| 1 | `forward 0.5` |
| 2 | `left 0.5` |
| 3 | `right 1.0` |
| 4 | `back 0.5` |
| 5 | `turn_left 15` |
| 6 | `turn_right 30` |

配置注释给出的依据：头部相机向下约 48°，4–5 m 外的箱子在画面顶边或画面之外（`t1_L0` 场景预览图），所以第一步先向前走。其余 5 步照搬任务书给出的 rule 探索序列（左 0.5 → 右 1.0 → 后退 0.5 → 左转 15° → 右转 30°）。第 3、6 步超过了单次探索上限（`configs/robot.yaml`：`max_move_m` 0.5、`max_turn_deg` 15），会被 Harness 拒绝，见"已知限制"。

```{figure} images/orch_llm_start_view.jpg
:width: 480px

agent_full 开局时交给 LLM 的头部相机图像（纸箱箱堆场景 `stack4c_L0_0001`）。相机大幅下视，几米外的箱堆只出现在画面上缘，这是探索序列先向前、提示词强调"远处物体在画面顶边"的原因。
```

`rule_recovery(h, tool, result, attempt)`：`attempt == 0` 返回 `"retry"`，否则返回 `"give_up"`，即"重试一次后放弃"。"一次"没有数据依据。

#### `recover(h, policy, tool, args, r)`：失败后的恢复循环

在 `r["status"] != "ok"` 时循环，每轮：

1. `r.budget_exhausted`、`r.rejected` 或 `h.over_budget()` 任一为真就立即返回 `r`；
2. `act = policy(h, tool, r, attempt)`；
3. `act` 是 `{"explore": mv}` 时执行 `explore`，执行后预算耗尽就返回；`act` 既不是探索也不是 `"retry"` 时返回；
4. `attempt += 1`，重新调用该工具。

第 1 步和第 3 步的预算检查来自一次死循环事故（见演进记录 v4）。

#### `Task1`：找到目标箱子并走到观察位姿

构造参数 `(h, cfg, selector, explore_policy=None, recovery_policy=None, fast_end=False)`；未传策略时用 `RuleExplore(cfg["explore_sequence"])` 和 `rule_recovery`。

`stage()` 在 `h.finished is None` 时循环：

1. 超预算 → 返回失败。
2. `detect_boxes`；失败就探索，探索不成 → `"no box found after exploring"`。
3. `select_target_box(selector)`；失败就以 `"target not identifiable: <原因>"` 为情境去探索。
4. `fit_front_face`；只有 `status == "fail"` 才处理（`"all"` 模式下 partial 表示部分箱面已拟合，位姿用这些面算）。处理方式是 `_advance_toward_target()`：距离远时深度无效或内点太少，就朝检测框前进。
5. `compute_obs_pose` → `approach(target="root_target")`。
6. approach 不成功时进入 `recover(...)`；例外是 `fast_end=True` 且 `data.walk_done` 为真（行走已完成，只是到达检查没过）时不恢复。
7. 返回 `{"success": status == "ok", "walked": ok 或 walk_done, "summary", "result"}`。

`_explore(why)`：策略返回 `None` 时返回 `False`；否则执行 `explore`，只有"失败且预算耗尽"才返回 `False`。被 Harness 拒绝的探索（例如超出单步上限）返回 `True`，状态机会接着重新感知。

`_advance_toward_target()`：

1. 目标为 `"all"` 时取所有检测框，否则取目标框；没有 bbox 返回 `False`。
2. 取所有框并集的中心像素，经 `pixel_ray(K, u, v)` 和 `T_root_cam` 得到机器人根坐标系下的方向，用 `atan2` 求方位角。
3. |方位角| > 3° 时先转向，角度 `min(15, |方位角|)`。转向调用的返回值不检查。
4. 以每步 `min(0.5, 剩余)` 米前进，累计到 `too_few_inliers_advance_m` = 1.0 m（`task1.yaml` 注释："step 3 failure: advance toward the detection and retry"），任何一步失败返回 `False`。

3° 阈值没有依据；15° 和 0.5 m 与探索上限一致。

`fast_end`（任务 3 用）：行走结束后阶段立即结束，精调后到达检查仍不过也不重试，因为扫码阶段会自己为每个条码规划站位。

#### `Task2`：定位条码、放置腕部、扫码

构造参数 `(h, cfg, selector, explore_policy=None, recovery_policy=None)`。注意 `explore` 直接保存传入值：rule 模式下任务 2 **没有探索策略**（`None`），agent_assist 中为 LLM 策略。

`_locate()`（任务书步骤 1–3）：

1. `_perceive_target()` = `detect_boxes` → `select_target_box` → `fit_front_face`。失败且有探索策略时只探索一次再感知，仍失败返回 `"target face: ..."`。
2. `detect_barcode_head`；失败时 `explore forward 0.10 m`（`task2.yaml: head_retry_advance_m`，注释 "no barcode -> advance 0.1 m and retry once"），重新感知后再试一次。
3. `estimate_barcode_pose`。

`place_and_scan(barcode_id=None, home_after=False, _replanned=False, before_stand_up=None)`（步骤 4–9）：

1. `compute_wrist_target(bid)` → `plan_stance`，任一失败就结束。
2. 行走前：手臂不在 home 先 `move_arm("home")`（举臂行走会让朝向漂移，并有碰撞风险）；没站直先 `set_base_height`（下蹲时行走效果变差）。
3. `approach(target="stance_target")`；规划的 `base_height` 与当前指令高度差 > 1e-6 时 `set_base_height(height=bh)` 下蹲。
4. `move_arm("pre")`；只在 `fail` 时处理（partial 表示带残差到达，腕部相机仍能看到目标）。失败原因以 `"IK"` 开头且尚未重规划时，站起来递归调用自身（`_replanned=True`），只重规划一次。
5. `refine_with_wrist`，成功时再 `stereo_check_wrist`。`_stereo_disagrees(s)`（`status == "partial"` 且 `data.diff_to_refined_m` 不为 `None`）为真时，再做一次 `move_arm("pre")` → `refine_with_wrist` → `stereo_check_wrist`；仍不一致就 `h.tb.adopt_stereo(...)` 采用运动立体结果。
6. 最终位姿循环，`k = 0 .. place.max_iters`（2）：`move_arm("final")`；IK 失败、尚未重规划、且当前条码 `width_source == "wrist"` 时站起来重规划一次（站位是按头部估计的宽度规划的，换成腕部实测宽度后够不到）。非最后一轮时 `refine_with_wrist`，腕部目标移动 ≤ `place.tol_m`（0.01 m）就退出。这两个值在配置里没有依据注释。
7. `check_wrist_placement`，`placed = (status == "ok")`。
8. `scan`，`scanned = (status == "ok")`。读不到时的重试在 scan 工具内部完成。
9. 若 `home_after`：手臂回 home，执行 `before_stand_up()` 回调（任务 3 用它在较低视角补充规划新标签），再站起。站起前先回 home，避免手臂扫过正在伸展的腿。

`run()`：`remember_home()` → `_locate()` → `place_and_scan(None, home_after=False)` → `finish(success=placed)`。任务 2 的评估在手臂停在条码前时进行，所以不收臂。

#### `Task3`：任务 1 + 规划扫码 + 逐个扫码

构造 `Task1(..., "all", fast_end=True)` 和 `Task2(..., "all")`；场景给出的 left/right selector 不使用，任务 3 观察并扫描所有箱子。

`run()`：

1. `remember_home()`。
2. **阶段 1**：`t1.stage()`；`walked` 为假 → `finish(fail, "task-1 phase: ...")`。接着 `plan_scans`，ok 就进入阶段 2；否则若 `budget_exhausted`、`rejected` 或探索失败就以失败结束，不然探索后重做阶段 1。`rejected` 表示 `plan_scans` 已达重试上限，失败计数只在 `next_barcode` 时清零，重做阶段 1 也不能让它再执行。
3. **阶段 2**（未结束且未超预算）：
   - `next_barcode`；非 ok 或 `done` → 退出循环；
   - `face_done` → `go_to_next_face`（失败则结束巡视，该侧标签保持未规划），成功则 `plan_scans` 后继续；
   - 其他情况 → `t2.place_and_scan(barcode_id, home_after=True, before_stand_up=lambda: h.call("plan_scans", {}))`。
   - 扫码顺序完全由 `next_barcode` 决定。
4. 结束：`n_ok` = 扫描计划中 `"scanned"` 的数量，`n_unr` = `"unreadable"` 的数量，`success = n_ok > 0 and n_ok + n_unr == len(scan_plan)`。Harness 的 `_finish` 会用账本重新判定成功，调用方的值另存为 `claimed_success`。

`TASKS = {1: Task1, 2: Task2, 3: Task3}`；`task_class(task)` 遇到未知任务号抛 `ValueError`，不静默退回其他任务。

### `agent_assist.py`：rule 主导、LLM 辅助

**决策工具 `DECISION_TOOLS`**：`explore`（schema 来自 `openai_tools(["explore"])`，即 `ExploreArgs`）、`retry`（无参数）、`give_up`（必填 `reason`）。三者都不含坐标。

**`_decide(h, llm, log, prompt_name, image_rgb, tag, **kw)`**：

1. `load_prompt` 生成文本和版本号；
2. 构造一条消息：文本 + 头部相机图像；
3. `llm.chat(clean(msgs), tools=DECISION_TOOLS, tool_choice="required")`，强制模型调用工具；
4. `log_llm` 记录；
5. 返回第一个 `parse_error is None` 的工具调用 `(name, arguments)`；全部解析失败时返回 `("give_up", {"reason": "no valid decision from the LLM"})`。

每次决策都是**单轮**：不带对话历史，只有当前提示、当前图像和（探索时）已做动作列表。

**`run_agent_assist(h, tcfg, selector, llm, log)`**：

- `T = task_class(h.task)`；`goal` 在任务 3 为 `"every cardboard box"`，否则 `"the {selector} cardboard box"`。
- `explore_policy(h, situation)`：`h.tb.look_head()` 取图 → 用 `task1_agent_assist_explore` 提示（goal、situation、history，history 为空时传 `"none"`）调用 `_decide`。结果不是 `explore` 就返回 `None`；用 `ExploreArgs(**args)` 校验，只捕获 `ValidationError` 和 `TypeError`，捕获后写一条 `explore_invalid` 日志再返回 `None`；通过后把 `"direction distance"` 追加进 history 并返回动作。
- `recovery_policy(h, tool, result, attempt)`：取头部图像，用 `recovery` 提示（result 去掉 traceback）调用 `_decide`。`retry` → `"retry"`；`explore` → 校验后返回 `{"explore": ...}`，参数无效记 `recovery_invalid` 日志并返回 `"give_up"`；其他 → `"give_up"`。
- 最后 `T(h, tcfg, selector, explore_policy=..., recovery_policy=...).run()`。

LLM 的探索动作由 `Task1._explore` 或 `Task2._locate` 执行，恢复决定由 `rule.recover` 执行，同样受预算和拒绝条件约束。

一个实际例子（2026-11-06，任务 1 场景 `t1_L0_0000`）：第一次 approach 侧向差 3.7 cm，没有通过 3 cm 的到位检查，`error=[0.009, 0.037, 0.229]`；恢复策略问 LLM，LLM 选择前进 0.3 m，之后重新 approach 成功。整个 episode 312.5 s，误差 2.6 cm，LLM 调用 1 次。

### `agent_full.py`：LLM 全程决策

#### 常量

| 常量 | 值 | 依据 |
|---|---|---|
| `KEEP_IMAGES` | 2 | 上下文中保留的图像数，更早的图换成文字占位。注释只给出 vLLM 每个 prompt 最多 8 张图的上限，没有说明为何取 2 |
| `MAX_NUDGES` | 3 | 容忍的"不调用工具"回复次数，超过就结束 episode。无依据 |
| 上下文上限 | 32768 tokens | vLLM `max_model_len`，是压缩设计的约束条件 |

#### 辅助函数

- `_prune_images(msgs)`：从最新消息往前数含图像的消息，第 3 条及更早的删除图像部分，追加文本 `"[older image removed]"`。
- `_start_text(task, selector, multi_face=False)`：
  - 任务 3：`"Start task 3 (phase 1: find all boxes and approach them)."`，多侧面任务追加"本任务覆盖箱堆所有侧面……side 完成后 next_barcode 返回 face_done，调用 go_to_next_face，再 plan_scans"；单面任务追加"只覆盖朝向你的一面"。`multi_face` 来自 `scripts/run_task.py`：`tb.multi_face = task == 3 and sc.get("scan_faces") == "all"`。
  - 任务 1/2：`"Start task {task}. Target: the {selector} box."`
- `phase1_summary(h, plan_result, later, task_text)`：替换阶段 1 历史的 user 消息，依次包含：`"Task (as given at the start): {task_text}"`、阶段 1 用掉的工具调用数和探索步数、最后一次 approach（`h.last_approach`）、`plan_scans` 结果（固定的扫码顺序）、同一条回复中 plan_scans 之后的调用及结果、当前预算、一句阶段 2 指引。
- `barcode_boundary_text(h, nb_result, later)`：以 `"Context compacted at next_barcode ..."` 开头，说明之前条码的消息已删除、scan_plan 是每个条码状态的权威记录，然后是 `next_barcode` 结果、同一条回复中之后的调用、当前预算。

#### 主循环 `run_agent_full(h, tcfg, selector, llm, log)`

1. 提示词参数 `kw = {selector, max_tool_calls, max_explore_steps}`，任务 3 另加 `per_barcode_calls`（`budget.per_barcode.max_tool_calls` = 28）。系统提示 `load_prompt(f"task{task}_agent_full", **kw)`；工具 `openai_tools(TASK_TOOLS[task])`（任务 1 共 9 个、任务 2 共 20 个、任务 3 共 23 个，都含 `look_head` 和 `finish`）。
2. 任务工具中有 `move_arm`（任务 2、3）时先 `h.tb.remember_home()`，否则每次 `move_arm("home")` 都报 "home pose unknown"。
3. 初始消息：system + user（`task_text + " Current head camera image attached."` 和头部图像）。
4. 循环直到 `h.finished`：
   - 超预算 → `finish(success=False, summary=原因)`；
   - `_prune_images` → `llm.chat(clean(msgs), tools, tool_choice="auto")` → `log_llm(kind="agent_full")` → 把 `assistant_message` 追加进历史；
   - **回复没有工具调用**：`nudges += 1`，超过 `MAX_NUDGES` 就 `finish(fail, "the LLM stopped calling tools")`，否则追加 `"Write one sentence of reasoning, then call a tool (or finish)."`；
   - **逐个处理工具调用**：参数 JSON 解析失败的调用不执行，构造 `{"status":"fail","reason":parse_error}` 作为结果（这种调用不经过 Harness，不计入工具预算）；其余调用 `h.call(name, arguments)`。结果以 `role="tool"` 消息追加，图像暂存；
   - **识别压缩边界**（每条回复只取第一个）：尚未压缩时第一个 ok 的 `plan_scans` 是 phase 边界；压缩过之后每个 ok 的 `next_barcode` 是 barcode 边界；边界之后的调用记入 `later`；
   - **执行压缩**：phase 边界 → `head = phase1_summary(...)`；barcode 边界 → `head + "\n\n" + barcode_boundary_text(...)`。写一条 `history_compaction` 日志（`at`、`dropped_messages`、`summary`），历史重置为 `[system, user(summary)]`，本轮图像随旧段落丢弃；
   - 没有压缩时，本轮工具返回的图像逐张以 user 消息 `"Image from {name}."` 追加。

压缩的依据：任务 3 的扫码顺序不由 LLM 选，已处理条码的状态（status、内容、失败原因、所在侧面）都在 `next_barcode` 返回的 scan plan 里，所以旧对话可以安全丢弃。上下文长度上限约等于"一个条码的往返"，和条码数、侧面数无关。实测（2026-11-07 第 9 轮）：每个条码开始时提示长度回落到约 4.8K token，一个码扫完最多约 7.7K；第 12 轮整局最长一次提示 11069 token。

任务 1 和任务 2 没有 `plan_scans` / `next_barcode`，不会触发压缩，依赖 episode 本身短。

### 提示词

所有提示词首行都是 `<!-- prompt: <name> vN -->`，由 `load_prompt` 解析后写进每条 LLM 日志的 `prompt_version`，所以任何一次运行用的是哪个版本都可以从日志查到。仓库里只保留当前版本的文件。

| 文件 | 当前版本 | 使用者 | 占位符 |
|---|---|---|---|
| `prompts/recovery.md` | v2 | agent_assist `recovery_policy` | `{task}` `{tool}` `{result}` `{attempt}` |
| `prompts/task1_agent_assist_explore.md` | v2 | agent_assist `explore_policy`（任务 1/2/3 共用） | `{goal}` `{situation}` `{history}` |
| `prompts/task1_agent_full.md` | v2 | agent_full 任务 1 | `{selector}` `{max_tool_calls}` `{max_explore_steps}` |
| `prompts/task2_agent_full.md` | v2 | agent_full 任务 2 | `{selector}` `{max_tool_calls}` |
| `prompts/task3_agent_full.md` | v7 | agent_full 任务 3 | `{max_tool_calls}` `{max_explore_steps}` `{per_barcode_calls}` |

共同结构：

1. 角色：监督或控制一台 Unitree G1 人形机器人（任务 2、3 中左手持扫码枪）。
2. 任务目标。
3. 规则：所有数值（位置、距离、位姿、stand-off）由工具计算，模型只选工具、探索方向和失败处理（assist 提示写作 "Do not output coordinates"）；典型调用序列；各种失败的处理；预算。
4. 推理要求：每次调用工具前，在消息正文里写**一句**简短理由（作为推理记录），然后再调用工具。

各提示词要点：

- **recovery v2**：告诉模型工具 `{tool}` 失败了、失败结果、本 episode 中已尝试次数，附当前头部图像；要求恰好调用一个工具：retry、explore（一次受约束的移动，之后流程继续）或 give_up。
- **task1_agent_assist_explore v2**：说明固定流程无法继续的原因 `{situation}`；提醒头部相机大幅下视，远处物体在画面顶边或之外；列出已做的探索动作；要求选一个探索动作（平移 ≤0.5 m、转向 ≤15°），无济于事时 give_up。
- **task1_agent_full v2**：典型序列 `detect_boxes → select_target_box("{selector}") → fit_front_face → compute_obs_pose → approach → finish`；看不到或认不出时 `look_head` 后探索，桌子远时向前有用；`fit_front_face` 因深度无效失败时向前后重试；预算 `{max_tool_calls}` 次调用、`{max_explore_steps}` 次探索。
- **task2_agent_full v2**：目标是枪尖在条码前 0.20 m 并对准；典型序列 `detect_boxes → select_target_box → fit_front_face → detect_barcode_head → estimate_barcode_pose → compute_wrist_target(offset=0.2) → plan_stance → approach(stance_target) → move_arm("pre") → refine_with_wrist → stereo_check_wrist → move_arm("final") → check_wrist_placement → scan → finish`；scan 无读数时 `refine_with_wrist → move_arm("final")` 再扫；stereo 报分歧时 `move_arm("pre")` 再精修。
- **task3_agent_full v7**（规则最多）：
  - 两个阶段：阶段 1 找到所有箱子（桌上、地上或堆叠）并走到能看到全部前表面的观察位姿；阶段 2 按工具定好的顺序每个标签扫一次。
  - 阶段 1 结束条件：approach 返回 ok 或 partial 都接着调用 `plan_scans`，不要为小的到达误差去探索；只有 plan_scans 找不到任何标签时才探索并重做阶段 1；离箱面太近的探索会被工具拒绝。
  - 阶段 2 每个条码：`next_barcode → compute_wrist_target`（不要传 offset，stand-off 是自适应的）`→ plan_stance → move_arm("home") → approach(stance_target) → move_arm("pre") → refine_with_wrist → move_arm("final") → check_wrist_placement → scan → move_arm("home") → plan_scans`（补充从更近更低视角首次看到的标签）。
  - 下蹲由 `plan_stance` 和 `approach(stance_target)` 负责，模型不用自己调 `set_base_height`。
  - `move_arm` 或 `check_wrist_placement` 返回 partial 不是失败，继续执行到 scan；只有 pre / final 因 IK 不可达而失败时，回 home 重新规划站位并 approach 一次，仍失败就 `next_barcode`。
  - `plan_stance` 报告条码过低或过高、任何身高都够不到时不要重试，直接 `next_barcode`。
  - scan 自带一次重试；扫码枪看得到标签但完全解不出码时记为 `"unreadable"`，这是一个结果而不是失败。
  - 行走前（approach / explore）必须先 `move_arm("home")`。
  - 多侧面：收到 `face_done=true` 调用 `go_to_next_face`，不要自己绕行；巡视顺序 front → right → back → left；没有条码的侧面规划出 0 个条码。
  - 每个条码之后都调用 `next_barcode`；scan plan 只含已经看到的侧面，计划全部扫完不代表任务完成。
  - `done=true` 时 `finish(task=3, ...)`；成功由扫描记录计算，还有条码或侧面未完成时 `finish(success=true)` 会被拒绝。
  - 预算：阶段 1 为 `{max_tool_calls}` 次调用和 `{max_explore_steps}` 次探索，`plan_scans` 每规划一个条码增加 `{per_barcode_calls}` 次。

`task3_agent_full` 的版本变化一览（详细原因见演进记录）：

| 版本 | 日期 | 改动 | 触发问题 |
|---|---|---|---|
| v1 | 2026-11-06 | 初版：两阶段；"approach 之后阶段 1 结束（即使到达检查失败），然后调用 plan_scans" | — |
| v2 | 2026-11-06 | approach 返回 ok 或 partial 都直接 plan_scans，不要 explore 修小误差；靠近箱面的 explore 会被拒绝 | 首跑 LLM 用 explore 修到位误差，撞箱，0/2 |
| v3 | 2026-11-07 | 箱子可能在桌上、地上或堆叠；序列加入 `set_base_height` 下蹲 / 站起 | 引入低处条码和下蹲 |
| v3（原地修改，未升版本号） | 2026-11-07 | 下蹲改为工具自动完成；不可达的码不要重试；`move_arm` partial 不是失败；多侧面 face_done 流程；compute_wrist_target 不传 offset | LLM 把蹲和走的顺序弄反、把 partial 当失败、自编 offset |
| v4 | 2026-11-07 | 每个码之后都调 `next_barcode`；计划只含已见侧面，全部 scanned 不代表完成；未完成时 `finish(success=true)` 会被拒绝 | LLM 扫完 2 侧就宣布成功 |
| v5 | 2026-11-07 | 某一侧可能没有条码：plan_scans 规划 0 个，next_barcode 进入下一侧 | 空侧面 |
| v6 | 2026-11-07 | `check_wrist_placement` 返回 partial 也不是失败，继续 scan | LLM 把放置检查"fail"当停止信号，反复重走站位 |
| v7 | 2026-11-07 | 看得到却解不出码的标签记为 `unreadable`，是结果不是失败 | 纸箱贴纸上的假条码花纹 |

### 关键常量与配置汇总

| 名称 | 位置 | 数值 | 依据 |
|---|---|---|---|
| `explore_sequence` | `configs/task1.yaml` | forward 0.5, left 0.5, right 1.0, back 0.5, turn_left 15, turn_right 30 | 先向前：头部相机约 48° 下视。第 3、6 步超上限，见已知限制 |
| `too_few_inliers_advance_m` | `configs/task1.yaml` | 1.0 m | "step 3 failure: advance toward the detection and retry" |
| 方位转向阈值 | `rule.py::_advance_toward_target` | 3° | 无依据 |
| 单次转向 / 单步前进 | 同上 | 15° / 0.5 m | 与探索上限一致 |
| `rule_recovery` 重试次数 | `rule.py` | 1 | 无依据 |
| `head_retry_advance_m` | `configs/task2.yaml` | 0.10 m | "no barcode -> advance 0.1 m and retry once" |
| `place.max_iters` / `place.tol_m` | `configs/task2.yaml` | 2 / 0.01 m | 无注释 |
| IK 失败判断 | `rule.py::Task2.place_and_scan` | `reason.startswith("IK")` | 区分 IK 不可达和其他失败 |
| 重规划次数 | 同上 | 1（`_replanned`） | 无数值依据 |
| 下蹲高度比较容差 | `rule.py` | 1e-6 | 浮点比较 |
| `KEEP_IMAGES` | `agent_full.py` | 2 | 服务端上限 8，取 2 无依据 |
| `MAX_NUDGES` | `agent_full.py` | 3 | 无依据 |
| `tool_choice` | agent_assist / agent_full | `"required"` / `"auto"` | 无注释 |
| 预算（任务 1/2，任务 3 阶段 1） | `configs/task*.yaml` | `max_tool_calls` 40、`max_explore_steps` 6、`max_retries_per_tool` 2、`max_wall_s` 1800 | 无注释 |
| `per_barcode.max_tool_calls` | `configs/task3.yaml` | 28 | rule 模式任务 2 实测最大 21 次（57 个 episode）+ next_barcode 1 + scan 1 + 一次无读数重试 3 + 前后两次回 home 2 |
| `per_barcode.max_wall_s` | `configs/task3.yaml` | 600 s | 任务 2 实测最长 518 s（p90 324 s），加 scan 重试和两次回 home，向上取整 |
| 任务 2 stand-off | `prompts/task2_agent_full.md`、`configs/task2.yaml` | 0.20 m | 任务书规定的固定 stand-off |

## 演进记录

以下按时间顺序记录编排层和提示词的每一次改动。日期为开发时间；"轮"指同一场景的一次完整运行。

### v0 · 2026-11-05 · rule 状态机初版：任务 1/2，探索和恢复做成可替换插口

**当时的设计**。`g1agent/orchestrators/rule.py` 实现任务 1 和任务 2 的固定状态机，所有动作都通过 `Harness.call` 执行。从一开始就留了两个可替换的策略接口：

- `explore_policy` 决定探索动作。rule 模式用 `RuleExplore`，按配置里的固定序列依次执行，用完返回 `None`；
- `recovery_policy` 决定工具失败后怎么办。`rule_recovery` 第一次失败重试，第二次放弃。

`Task1.run` 每轮先检查 `over_budget`，再按 检测 → 选择 → 拟合 → 算观察位姿 → 接近 → 结束 的顺序执行。失败后的恢复由 `_with_recovery` 循环负责。

这样设计是为了让三种模式共用执行路径：rule 全用固定规则；agent_assist 只把这两个插口换成 LLM；agent_full 由 LLM 自己选工具。三者的成功率差异只来自"谁做决策"，不来自工具或执行路径的差异。

### v1 · 2026-11-05 · agent_full / agent_assist 初版与提示词 v1

**当时的设计**。

- `g1agent/orchestrators/common.py`：`load_prompt` 从提示词首行注释解析版本号，`log_llm` 记录每次 LLM 交互，图像另存成文件、日志里只记路径。
- `prompts/task1_agent_full.md` v1：角色是"通过工具控制 Unitree G1"；任务 1 是找到 `{selector}` 纸箱，走到能在头部相机里看到整个前表面的观察位姿；规则写明"所有数字（位置、距离、位姿）都由工具计算，不得编造坐标，只选择调用哪个工具、探索方向和失败后怎么做"；给出典型序列 `detect_boxes → select_target_box("{selector}") → fit_front_face → compute_obs_pose → approach → finish`，看不到箱子时调用 `look_head` 看图。`task2_agent_full.md` v1 同理。
- agent_assist 的 `recovery.md` 和 `task1_agent_assist_explore.md` 也在这一阶段建立（v1 原文没有保留）。
- agent_full 初版把**所有工具**都开放给 LLM。

### v2 · 2026-11-05 · 探索动作加入"前进"，LLM 不设输出长度上限

**问题**。任务书规定的探索动作只有左、右、后、左转、右转。头部相机向下约 48°，4–5 m 外的箱子在画面上边缘之外，不往前走就看不到箱子。

**改法**。

- 探索动作新增 `forward`（`registry.ExploreArgs` 的方向枚举、`configs/task1.yaml` 的 `explore_sequence` 第一步改为 `forward 0.5`，提示词中的探索说明同步加入 forward 和"相机下视、远处物体在画面顶边"的提醒）。
- 调用 LLM 不传 `max_tokens`。这是项目的设计原则（不设输出长度上限），截断的回复丢弃并重抽，由 `LLMClient` 负责（见 LLM 页）。

这两处都是对任务书的偏离，在当时的进度报告里单独列出。

### v3 · 2026-11-06 · L0 演示：agent_full 在任务 1 中越界调用任务 2 工具 → 按任务限制工具集

**当时的版本**。agent_full 对每个任务开放全部工具。

**现象**。在最简单的 L0 场景上录了 4 个演示 episode，评测器把 4 个都判为成功：

| episode | 用时 | 误差 | LLM 调用 | 备注 |
|---|---|---|---|---|
| agent_assist 任务 1（`t1_L0_0000`） | 312.5 s | 2.6 cm | 1 | approach 侧向差 3.7 cm 未过 3 cm 检查，LLM 选择前进 0.3 m 后重新 approach 成功 |
| agent_assist 任务 2（`t2_L0_0000`） | 199.4 s | 0.9 cm | 0 | 状态机一次完成 |
| agent_full 任务 2 | 357.6 s | 0.9 cm | 19 | 枪尖距条码面 0.200 m |
| agent_full 任务 1 | 381.7 s | 4.8 cm | — | 评测器判成功，但实际不算 |

agent_full 任务 1 的工具调用日志显示：approach 失败后，LLM 先 `look_head`，再越界调用了 `detect_barcode_head`、`estimate_barcode_pose`、抬臂等任务 2 的工具，最后自己报告失败，视频里手臂也是抬着的。之后发了更正说明这一个不算真正成功。

**根因**。LLM 能看到它不该用的工具，就会用。靠提示词说"这是任务 1"不足以约束。

**改法**。`g1agent/harness/registry.py` 增加按任务的白名单 `TASK_TOOLS`：任务 1 只有寻找和接近的 9 个工具；任务 2 当时开放全部工具（那时还没有任务 3）。后来加入任务 3 时改为：任务 2 不含 `plan_scans`、`next_barcode`、`go_to_next_face`，任务 3 开放全部。agent_full 用 `openai_tools(TASK_TOOLS[task])` 生成工具列表，`Harness.call` 拒绝白名单外的调用（`rejected: True`）。

**同期发现、未改动的问题**。rule 模式在 `t2_L1_box_lateral_0010`（箱子组横向偏移大，视野里只有一个箱子）上报 `only one box visible: cannot tell whether it is the left or the right one`，4.4 s 就失败。原因是 rule 模式下 `Task2` 没有探索策略，看不到第二个箱子时不会转身或移动。这正是 agent_assist 由 LLM 选探索动作的情况，当时的判断是 agent 模式的批量对比会显示出这一差别；rule 模式本身没有改。

### v4 · 2026-11-06 · 预算耗尽后恢复循环不停（单个 episode 2379 次调用）→ `recover()` 遇预算或拒绝即停

**当时的版本**。`rule.py` 的 `_with_recovery` 循环：工具失败 → 问恢复策略 → explore 或 retry → 再调用工具，直到成功或策略放弃。explore 之后的分支不检查预算。

**现象**。agent_assist 跑 `m5_assist_gear_4/t1_L1_box_shape_0007`（大箱子）时，`tool_calls.jsonl` 有 2379 行，`t_wall` 约 1891.5 s，而工具调用预算是 40。最后的记录全是 approach 和 explore 交替返回 `tool-call budget (40) exhausted`，预算耗尽后仅 approach 就调用了 1195 次。这个 episode 最后被手动终止。

**根因**。Harness 在预算耗尽后返回 `fail + budget_exhausted`，但恢复循环只看 `status`，把它当成普通失败继续恢复；LLM 恢复策略每次都给出 explore 或 retry，于是形成死循环。

**改法**。

- `_with_recovery` 改为 `recover(h, policy, tool, args, r)`：每轮先检查 `r.budget_exhausted`、`r.rejected`、`h.over_budget()`，任一为真立即返回；explore 之后也检查预算。
- Harness 中 `finish` 不受预算拦截，保证超预算后仍能正常结束 episode。
- 新增单元测试 `test_recover_stops_when_budget_is_exhausted`。

**效果**。预算或拒绝成为恢复循环的硬性出口，之后的运行中没有再出现预算耗尽后继续调用的情况。

### v5 · 2026-11-06 · 重构为可复用阶段并加入任务 3；agent_full 加阶段压缩；提示词统一升到 v2

**背景**。任务 3 = 任务 1（找到所有箱子并接近）+ 逐个扫描所有条码。原来的 `Task1.run` / `Task2.run` 直接调用 `finish`，无法在任务 3 中复用。

**改法**。

- `rule.py`：
  - `Task1.stage()` 返回 `{success, walked, summary}`，不再调用 `finish`；任务 3 下支持 `fast_end`。
  - `Task2` 拆出 `_locate()` 和 `place_and_scan(barcode_id, home_after)`：底盘移动前如果手臂抬着就先回 home。
  - 新增 `Task3.run()`：`remember_home` → 阶段 1 对所有箱子执行 → `plan_scans`（找不到标签就探索后重做阶段 1）→ `next_barcode` / `place_and_scan` 循环 → `finish(3)`。
  - 新增 `TASKS` 和 `task_class()`，未知任务号抛错。
  - stereo 结果原来用子串 `'differ'` 判断是否不一致，改为 `_stereo_disagrees`：检查对比数据 `diff_to_refined_m` 是否存在。依据是项目的设计原则：不用关键字匹配做判断。
- `agent_assist.py`：通过 `task_class` 分发；原来宽泛的 `except Exception` 收窄为 `(ValidationError, TypeError)`，并记日志（项目的设计原则：不用宽泛的异常捕获）。
- `agent_full.py`：第一次 `plan_scans` 成功时，把消息历史压缩成 `[system, 阶段 1 摘要 + 计划 + 新预算]`，写 `history_compaction` 日志，以控制在 `max_model_len` 32768 以内。
- 提示词：
  - 所有提示词升到 v2，新增"每次调用工具前在消息正文里写一句理由"。原因是 Qwen3-VL-8B-Instruct 没有单独的"思考"输出，回复的 `content` 往往为空，日志里看不到它为什么这样做；
  - 新增 `prompts/task3_agent_full.md` v1，写明扫码顺序由工具决定，不让 LLM 选。

### v6 · 2026-11-06 · 代码审查修复：agent_full 不记 home 位姿、阶段事件顺序、plan_scans 被拒后反复重做

审查任务 3 相关代码时发现并修复了 6 个问题，其中与编排层有关的 3 个：

1. **agent_full 不记录 home 位姿**。只有 rule 的 `run()` 调用 `remember_home`，agent_full 中每次 `move_arm("home")` 都报 `home pose unknown`，而任务 3 提示词要求每次行走前回 home。改为：任务工具里有 `move_arm` 时，`run_agent_full` 开头先调用 `h.tb.remember_home()`。
2. **阶段事件排在工具记录之前**。`Harness.call` 先发阶段事件再写工具记录，时间线和视频里 `task1_end` 出现在触发它的 `plan_scans` 之前。改为先写工具记录再执行 `_after`。
3. **plan_scans 被拒后 rule Task3 仍反复重做阶段 1**。改为 `plan_scans` 返回 `rejected` 就以失败结束（它已达重试上限，失败计数只在 `next_barcode` 时清零，重做阶段 1 也不能让它再执行）。

其余 3 个在工具层（站位与所有已知箱面保持至少 0.35 m、任务 3 有 scan_plan 后拒绝 `detect_barcode_head`、`plan_scans` 报告被丢弃的标签）。新增 `tests/test_task3_review.py` 共 6 个测试，测试从 55 个通过增加到 61 个通过。

**同时发现、未修的问题**。`configs/task1.yaml` 的 `explore_sequence` 中 `[right, 1.0]` 和 `[turn_right, 30]` 超过 `robot.yaml` 的 `max_move_m` 0.5 和 `max_turn_deg` 15，Harness 会拒绝（复现时返回 `explore right 1.0 -> fail move must be in (0, 0.5] m`），这两步白白占用调用次数、机器人不动。拆成合法小步共需 8 步，超过 `max_explore_steps` 6，修复需要决定缩短序列还是提高探索预算，这次没有改。另外，阶段 1 的 approach 失败计数在多次重做之间累积，最多重做 3 次后 approach 会被拒绝。

### v7 · 2026-11-06 · task3_agent_full v1 → v2：到位检查没过时 LLM 用 explore 修误差并撞箱

**当时的版本**。`task3_agent_full` v1 写的是："Phase 1 ends after approach, even if its arrival check failed: then call plan_scans ... If it finds no label, explore and redo phase 1."

**现象**。agent_full 任务 3 首跑（`t1_L0_0000`）失败，0/2，用时 254.7 s，LLM 12 次调用、0 次截断。过程：

1. `detect_boxes → select_target_box(all) → fit_front_face → compute_obs_pose → approach`，approach 约 160 s 后返回 fail：沿法向误差 0.071 m、侧向 −0.019 m、`face_in_view` 未通过；
2. LLM 选择 `explore forward 0.5`（超时未完成），再 `forward 0.3` 报 `torso_link within -0.002 m of B0`（碰到箱子），左转、右转同样因碰撞失败，后退被拒（explore 连续失败，重试预算用完）；
3. 调用 `plan_scans` 失败（"no barcode label localised on any fitted face"，检测到的那个标签因射线与箱面几乎平行被丢弃），最后 `finish(success=false)`，总结为 "label rays nearly parallel to the face planes; exploration budget exhausted"。

同场景下 rule 模式的 approach 正常。

**根因**。

- 近距离时头部相机只看到箱面一部分，重新拟合的箱面被截断，观察位姿目标漂移；"箱面全部在视野内"检查在边缘处总是失败；
- 提示词虽然写了"approach 之后调用 plan_scans"，但 approach 返回的是 `fail`，LLM 自然把它当作需要修正的失败，用 explore 去修小误差；
- 当时 `final_bias_m` = 0.05 让机器人停在目标前方 5 cm；
- explore 没有防撞约束。

**改法**。

- 工具层：观察距离和完整箱面范围由第一次远距离拟合固定；任务 3 中行走已完成但到达检查未过时返回 `partial`（"walk finished; arrival check not within tolerance"），并提示下一步调用 plan_scans；会停在离已知箱面 0.35 m 以内的 explore 直接拒绝（0.35 m 沿用已有的站位最小距离）；`final_bias_m` 改为 0。
- 提示词 v2：approach 返回 ok 或 partial 时都直接调用 `plan_scans`，不要为小误差 explore；说明靠近箱面的 explore 会被拒绝。
- rule 侧对应的是 `Task1` 的 `fast_end`：`walk_done` 时不进入恢复。

**效果**。修复后重跑同一场景：

| 模式 | 扫到 | 任务 1 阶段 | 枪尖误差 | 用时 | LLM 调用 |
|---|---|---|---|---|---|
| agent_full 首跑（v1） | 0/2 | 失败 | — | 254.7 s | 12 |
| agent_full 重跑（v2） | 2/2 | 达标 | 3.9 cm | 565 s | 28，截断 0 次 |

### v8 · 2026-11-07 · 下蹲：rule 流程加入下蹲 / 站起，task3_agent_full 升到 v3；预备位不可达时重规划

**背景**。任务 3 的场景扩展到箱子放在地上或堆叠，低处的条码需要下蹲才能够到。

**观察（04:39）**。在此之前对比 agent_full 的实际调用序列，发现它每个码只做一次腕部精修、从不调用 `stereo_check_wrist`；两次 `check_wrist_placement` 都没过（偏离光轴 2.8 cm，超过 `axis_offset_max_m` 0.02 m），仍直接扫码并读出（314 s）；rule 做完整精修，偏离 1.7–1.8 cm 通过。当时的判断是：条码更小或偏得更多时 agent_full 可能读不出，应把"检查不过先修正再扫"交给工具强制。

**改法（07:41）**。

- `rule.py::place_and_scan`：手臂回 home →（若在蹲着先站起）→ `approach(stance_target)` →（`plan_stance` 要求下蹲时）`set_base_height(base_height)` → `move_arm pre` → `refine_with_wrist` → `move_arm final` → `check_wrist_placement` → `scan` → `move_arm home` →（若蹲着）站起。
- `registry.py` 新增 `BaseHeightArgs` 和 `set_base_height`，`height` 可选，省略即站起。
- `task3_agent_full` v2 → v3：阶段 1 中"箱子在桌上"改为"在桌上、地上或堆叠"；阶段 2 序列中加入 `[set_base_height() if squatting]` 和 `[set_base_height(base_height) if plan_stance asked for a squat]` 两步。

**改法（08:14）**。原来只有 final 位姿 IK 不可达时才重规划，现在 pre 位姿也一样：站起来，从当前实际位置重新 `plan_stance` 一次再试（`_replanned` 标志保证只重规划一次）。依据是 pre 位姿在真实行走或下蹲之后可能变得不可达。

### v9 · 2026-11-07 · 把 LLM 容易做错的步骤搬进工具（task3_agent_full 仍为 v3，原地修改）

这一段时间对 `task3_agent_full.md` 做了几处修改，但没有升版本号，所以这期间的日志都记为 v3。这是一个流程上的疏漏：同一个版本号对应了不同的提示词文本。

**问题与改法**（按出现顺序）：

1. **蹲和走的顺序弄反**。agent_full 的 LLM 把下蹲和行走的顺序弄错（当时的报告记为"搞错蹲和走的顺序"；原来的 `_walk_guard` 在蹲着时直接拒绝行走）。改为站起 / 下蹲由工具负责：每个行走类工具先通过 `_walk_guard` 让机器人站起（要求手臂在 home），`approach(stance_target)` 到达后自动蹲到 `plan_stance` 规划的高度。提示词去掉 `set_base_height` 步骤，改为"plan_stance 选择身体高度，approach(stance_target) 到达后下蹲，你不需要自己调用 set_base_height"。
2. **不可达的码（09:09）**。这一条来自需求，不是观察到的 LLM 错误：条码低于可达下限时要检测出来、报错给出原因、记入日志，然后继续扫下一个。提示词加入：`plan_stance` 报告条码过低或过高、任何身高都够不到时，不要重试，原因已记录，调用 `next_barcode`。
3. **跳过读不出后的重试**。LLM 跳过提示词里"读不出就精修再扫一次"的步骤（2 cm 偏差后未重试）；rule 模式会做。改为 `scan` 工具内部自动重试一次（`refine_with_wrist` + `move_arm final` + `scan`），rule 的 `place_and_scan` 随之去掉自己的重试。
4. **把 `move_arm` 的 partial 当失败（11:23）**。`stack9` 第二轮漏了右下码：手臂到 final 还差约 3 cm，`move_arm` 返回 partial，原文是 "arm did not reach the target: residual ..."。LLM 当成失败，收臂重新规划，没有扫码；之后两次重新规划时"深蹲 + 画面对齐"刚好落在手臂可达范围边缘，差 5 mm / 2°，它就放弃了这个码。rule 遇到同样的 partial 会照常扫码，靠 scan 自带的重试读到。改法：工具返回文本改为 "arm reached the target with a residual ... this is not a failure: go on (pre: refine_with_wrist; final: check_wrist_placement, then scan, which retries by itself)"；提示词加入"move_arm 返回 partial 不是失败"。
5. **自编 offset**。LLM 调用 `compute_wrist_target` 时自带 offset，曾在 0.14 m 处扫码而读不出。最初的处理是任务 3 中拒绝带 offset 的调用（见 v11 的后续）。

**阶段性结论（09:47）**。两种模式调用同一套工具，差别只在 LLM 是否按流程走。这几轮 LLM 犯过的错：自编扫码距离、把蹲和走的顺序弄反、失败后不重试，前两个已改成由工具强制执行。在当前任务上 agent_full 不如 rule 稳定，因为保证成功的步骤都已在工具和规则里，LLM 主要带来出错的机会。

### v10 · 2026-11-07 · 多侧面任务：开场指令区分单面 / 全部侧面，提示词加入 face_done 流程

**背景**。任务 3 扩展到箱堆的四个侧面都要扫（`scan_faces: all`）。

**改法**。

- 提示词（12:07，仍为 v3）：原来"next_barcode 返回 done=true 时调用 finish"之前加入多侧面规则：next_barcode 在当前侧完成时返回 `face_done=true` → 调用 `go_to_next_face`（它绕到下一侧并站到观察位姿，不要自己用 explore / approach 绕行）→ 对这一侧 `plan_scans` → 继续 next_barcode 循环；巡视顺序 front → right → back → left。
- `agent_full.py::_start_text` 增加 `multi_face` 参数（12:11）。多侧面任务追加 "This task covers EVERY side of the box stack (front, right, back, left): after a side is finished, next_barcode returns face_done and you call go_to_next_face, then plan_scans."；单面任务追加 "This task covers the side of the boxes facing you only."。`scripts/run_task.py` 设置 `tb.multi_face = task == 3 and sc.get("scan_faces") == "all"`。
- rule 的 `Task3` 增加对应的循环：`face_done` → `go_to_next_face` → `plan_scans`。

测试共 93 个通过。

### v11 · 2026-11-07 · 拒绝 offset 反而导致跳码；上下文溢出 → 每次 next_barcode 都压缩历史

**问题一：拒绝 offset 后 LLM 跳过所有码**。`stack9p` 中 LLM 调用 `compute_wrist_target` 时自带 offset（0.27、0.267），工具按"任务 3 的 stand-off 是自适应的"直接拒绝；LLM 随后调用 `plan_stance` 得到 "no wrist target"，然后不断调用 `next_barcode` 跳过码，还在单面任务里调用 `go_to_next_face`（被拒：this task scans the front side only），第 19 步就 finish，0 个码（0/3）。`stack4f` 中同样出现：自带 offset 被拒 → `plan_stance` 无腕部目标 → `approach` 无 stance_target。"先 plan_stance 后 compute_wrist_target"这个错误在前几轮已经出现过（三箱堆叠场景的 agent_full 运行，当时 LLM 自己纠正了顺序）。

**改法**（工具层，记录在这里是因为它是对 LLM 行为的约束）：

- 任务 3 中传入的 offset 被**忽略**而不是拒绝，结果里注明；
- `plan_stance` 在 LLM 跳过 `compute_wrist_target` 时自己算出当前码的腕部目标；
- `next_barcode` 在当前码连站位都没规划过时拒绝跳过，提示正确的序列。
- 提示词中的序列写成 "compute_wrist_target (never pass an offset: the stand-off is adaptive)"。

**问题二：上下文溢出**。第 8 轮在四面箱堆（12 个码）上扫完前两面 6 个码、走向背面时上下文达到 33288 tokens，超过 vLLM 的 `max_model_len` 32768。每扫一个码对话约增长 4K tokens，12 个码必然溢出。v5 的压缩只在阶段边界做一次，不够。

**改法（13:15）**。`agent_full.py` 把压缩点扩展为两种：

1. phase：第一次成功的 `plan_scans`，压缩成 system + 阶段 1 摘要（`head`）；
2. barcode：压缩过之后，每次成功的 `next_barcode`，保留 system + `head` + 这次 `next_barcode` 的完整结果 + 之后的调用 + 预算。

新增 `barcode_boundary_text(h, nb_result, later)`；每次压缩写 `{"kind": "history_compaction", "at": kind}` 日志。新增单元测试 `test_agent_full_task3_compacts_history_at_every_next_barcode`（用假 LLM 和脚本化的 toolbox），全量测试 97 个通过、1 个跳过。

**效果**。第 9 轮：每个码开始时提示长度回落到约 4.8K token（如 `C:phase 4315`、`C:barcode 4750`、`C:barcode 4768`），一个码扫完最多约 7.7K（7702）。第 12 轮整局最长一次提示 11069 token。上下文上限只取决于单个码那一段，与码数和侧面数无关。

### v12 · 2026-11-07 · 压缩丢掉了任务陈述 → 摘要写回任务陈述；task3_agent_full 升到 v4；Harness 拒绝过早的成功声明

**当时的版本**。v11 的压缩把开局 user 消息一起删掉；"要扫全部四面"这句话只写在开局消息 `_start_text` 里。

**现象**。第 9 轮（`stack4f`）压缩生效，正面和右侧面 6/6 都扫到，但 LLM 没有去后两面，直接调用 `finish(success=true)`，episode 以 6/12 结束（`success=false`）。期间还有一次 `move_arm final` 残差 0.034 m / 8.7°，`check_wrist_placement` 判为超差，之后通过重新规划恢复。

**根因**。

1. 任务陈述在压缩时丢失。scan plan 只含已经看到的侧面，LLM 看到计划里的码全部 scanned，就以为任务完成。
2. Harness 接受了与实际状态矛盾的成功声明。

**改法**。

- `agent_full.py`（13:33）：`_start_text` 不再自带图片说明，开局消息改为 `task_text + " Current head camera image attached."`；`phase1_summary(h, plan_result, later, task_text)` 第一行写 `"Task (as given at the start): {task_text}"`。因为 barcode 边界的摘要以 `head` 开头，两种压缩后任务约束都原样保留。单元测试验证压缩后的上下文仍包含"扫所有侧面"的说明。
- `task3_agent_full` v3 → v4（13:34）：在"done=true 时调用 finish"之前加入：每个码结束（scanned、failed 或 unreachable）后都调用 `next_barcode`，只有它能说明还有下一个码、下一侧（face_done）还是全部完成（done）；scan plan 只含目前见过的侧面，计划里全部 scanned 不代表完成；任务覆盖所有侧面时，仍有码或侧面未完成时 `finish(success=true)` 会被拒绝。
- `guards.py`：`finish(success=true)` 在任务 3 还有剩余工作时被拒绝；`_finish` 的成功标志改由账本计算，调用方的值另存为 `claimed_success`（详见 Harness 页）。

这三处分别从"信息不丢""提示明确""状态不可伪造"三个层面处理同一个问题，而不是只在提示词里加一句。

### v13 · 2026-11-07 · 规划站位后不走过去就抬手，陷入循环 → move_arm 要求先到达站位

**现象**。第 11 轮正面 3/3；右侧面 bc4 失败：LLM 调用 `plan_stance` 成功后没有调用 `approach('stance_target')`，直接 `move_arm pre`，残差 0.105 m / 19.2°，之后放弃了这个码。背面 3/3，前三面共 8/9。右侧面法向偏航（里程计对真值）为 9.8°，这是里程计漂移本身。左侧面这次能规划出站位，但 LLM 在这里反复"规划站位 → 不走 → 抬手失败"，陷入循环，这一轮被手动停止。

**根因**。

- `plan_stance` 成功时返回 "squat to base_height 0.55 after walking"，容易被理解成已经到位；
- 工具层没有强制"先走到站位再抬臂"，只靠提示词约束。

**改法**（工具层）。任务 3 中 `move_arm` 的 pre / final 要求：当前码已规划站位，并且机器人确实到达了这个站位，否则返回 "the robot is not at the planned stance: call approach('stance_target') first"。同一次修改中，`plan_stance` 成功时的返回文本改为 "stance planned (the robot has not moved): call approach('stance_target') to walk there"；只规划了站位不算尝试过这个码，`next_barcode` 仍拒绝跳过，真正走过去之后才允许跳过。

### v14 · 2026-11-07 · task3_agent_full v5 / v6 / v7：空侧面、放置检查的 partial、不可读标签

这三个版本都在同一天 16:32–18:35 之间，针对的是纸箱箱堆场景（`stack4f`、`stack4c`）。

**v5（16:32）：没有条码的侧面**。巡视顺序的说明之后加一句："A side may carry no barcode: plan_scans then plans 0 barcodes for it (status ok), and next_barcode moves on to the next side." 对应 Harness 中"某一侧计划为空也算已有计划"的修改，避免 LLM 把 0 个标签当作失败去探索。

**v6（16:36）：放置检查超差不是停止信号**。

- 现象：`stack4f_L0_0001` 中 `check_wrist_placement` 返回 fail（偏离光轴 2.0–2.4 cm，容差 2 cm），LLM 把 "fail" 理解为停止，重新走了 4 次站位，一次也没有扫码，丢了这个码。
- 改法：任务 3 中放置略超容差、且扫码枪确实在 stand-off 处、标签在视野内时，`check_wrist_placement` 返回 partial 并提示 "call scan now"；任务 2 仍按失败处理，因为任务 2 的成功标准就是位置合格。提示词改为 "move_arm or check_wrist_placement returning partial ... is NOT a failure: continue the sequence and call scan"。新增参数化测试（任务 3 偏 0.01 → ok，偏 0.023 → partial；任务 2 偏 0.023 → fail），逻辑测试 40 个通过。
- 后续补充：之后在 `stack4c_L0_0001` 中还出现过 LLM 在手臂仍停在 pre 位（0.35 m）时就调用放置检查、被提示 "call scan now" 后从 0.35 m 扫码；以及在 `refine_with_wrist` 之后直接调用检查、被要求重新规划站位后反复规划 4 次。工具改为：手臂没有被送到 final 时返回 fail 并提示 "call move_arm('final'), then scan"；只有确实在 stand-off 时才返回 partial。

**v7（18:35）：不可读标签是一个结果**。

- 现象：纸箱模型自带的面单贴纸上印着假条码花纹，被标签检测识别成标签（`stack4c_L0_0001` 中这类标签共 23 个）。例如 bc1 处没有真码，腕部相机近看在完整视野内也解不出任何码。这些标签被记为 failed，任务永远不可能成功。
- 改法：扫码枪在近距离完整看到标签、重试后仍完全解不出码时，`scan` 把它记为 `unreadable`；Harness 和 rule 的成功判定都把 `unreadable` 计入"已处理"；提示词加入 "When the scanner had the label in view and nothing at all decoded, scan records the label as 'unreadable' ...: that is a result, not a failure; go on with next_barcode."
- 效果见 v15 后的运行：正面 3 个真码全部扫到，4 处贴纸全部判为 unreadable，没有一个记为失败。

### v15 · 2026-11-07 · LLM 越过当前码去处理下一个码，失败计数被别的成功调用清零 → 只允许操作当前码 + 按"工具+参数"计失败

**当时的版本**。工具允许显式传入 `barcode_id` 去操作任何条码；Harness 的连续失败计数只按工具名，任何一次成功都清零。

**现象**。运行 `af4c_2side_v6`（纸箱箱堆 `stack4c_L0_0001`）在正面完成后（1312 s 时正面 3/3 真码已扫）突然结束，中间没有右侧面的 `face_end` 事件，`success=false`。`tool_calls.jsonl` 显示：LLM 在 bc7 还是当前码时没有调用 `next_barcode`，自行转去处理 bc8（显式传 `barcode_id`），为 bc8 规划站位；而 `move_arm('pre')` 仍按当前码 bc7 查站位，反复失败于 `no stance planned for bc7`。LLM 每次又调用 `plan_stance` 再试，共循环 14 次。由于中间穿插着成功的 `move_arm('home')`，按工具名的失败计数每次被清零，重试上限从未触发，最终上下文被撑满（32608 tokens），日志中最后的 LLM 尝试是一条被截断的回复。

**根因**。两层：

1. 工具层允许对非当前码操作，打乱了 `next_barcode` 决定的扫码顺序；
2. Harness 的失败计数只按工具名，识别不了"同一个调用反复失败"。

**改法**。

- 工具层：新增 `_not_current(barcode_id)`，任务 3 中条码类工具只能操作当前码，传入其他 id 返回 "{id} is not the current barcode ({cur}): call next_barcode to move on to the next barcode of the plan"。
- `guards.py`：增加按"工具名 + 参数 JSON"计的失败键，只有完全相同的调用成功才清零；超限时提示改做别的事（例如 `next_barcode`）。
- 编排层代码不变：这两个约束都在 agent_full 之下的层实现，对 rule 模式没有影响（rule 从不越过当前码）。

**效果**。修复后的下一次运行（`af4c_2side_v7`，同一场景）：右侧面没有再循环，正常结束；6 个真码全部扫到，每一侧的扫码顺序都和真值一致；正面 4 处贴纸全部判为 unreadable。最后一个真码在 4284 s 扫到。

## 已知限制

- **`explore_sequence` 有两步超出上限**。`configs/task1.yaml` 中 `right 1.0` 和 `turn_right 30` 超过 `max_move_m` 0.5 / `max_turn_deg` 15，在 rule 模式（任务 1、任务 3 阶段 1）中会被 Harness 拒绝，白白占用调用次数。拆成合法小步需要 8 步，超过 `max_explore_steps` 6，需要决定缩短序列还是提高探索预算，目前未改。
- **rule 模式任务 2 不会探索**。`Task2` 在 rule 模式下没有探索策略，看不到目标时直接失败（例如 `t2_L1_box_lateral_0010`，4.4 s 失败）。
- **agent_assist 的探索提示词写的是"在桌上"**。`task1_agent_assist_explore.md` 被任务 1/2/3 共用，正文写 "find {goal} on a table"，而任务 3 的箱子可能在地上或堆叠。目前没有针对任务 3 的探索提示词。
- **提示词旧版本没有完整保留**。仓库只有当前版本的文件；`recovery`、`task1_agent_assist_explore`、`task2_agent_full` 的 v1 原文无法复原。`task3_agent_full` v3 期间有多次原地修改未升版本号（见 v9、v10），这段时间日志里的 `prompt_version` 不能唯一确定提示词文本。
- **agent_full 在任务 3 上不如 rule 稳定**。大部分保证成功的步骤（下蹲顺序、scan 重试、offset、只操作当前码、先到站位再抬臂）已经由工具强制，LLM 的作用主要剩下"按流程调用"；LLM 每次新出现的偏离流程的方式，都需要在工具或 Harness 中再加约束。
- **没有依据的常量**：`KEEP_IMAGES` 2、`MAX_NUDGES` 3、`rule_recovery` 重试 1 次、方位转向阈值 3°、`place.max_iters` 2、`place.tol_m` 0.01 m、`tool_choice` 的取值。
- **任务 1/2 的 agent_full 没有上下文压缩**。只靠 episode 本身短；长时间探索或反复失败时没有保护，只能靠工具调用预算结束。
- **`_advance_toward_target` 不检查转向结果**。转向失败时仍会继续前进。
- **提示词不能含字面花括号**。`load_prompt` 用 `str.format` 填充。
- **解析失败的工具调用不计入预算**。agent_full 中参数 JSON 无效的调用不经过 Harness，不增加 `n_calls`；连续解析失败只受 LLM 调用本身和墙钟预算限制。
