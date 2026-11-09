# 评测与测试场景

本页介绍系统怎样判断一次运行（episode）成功与否：评测通道怎样拿到真值，各任务的指标怎样计算，测试场景集怎样组成，以及怎样批量运行和汇总结果。演进记录部分按时间列出评测相关的每次修改，也列出仿真侧对评测有影响的变化。

视频合成和诊断标记（`g1agent/viz/`）有单独的实现说明。本页只写它和评测共用的部分，例如 GT stand-off 的定义、标记不可见自检。

## 作用与在系统中的位置

评测侧（`g1agent/eval/`）是整个系统中唯一读取真值（GT）的部分。算法侧的编排器、工具、感知、Harness 都拿不到真值。它们只把自己的估计写进世界状态（里程计 odom 坐标系）。episode 结束后，评测器把这些估计、仿真器记下的实际状态和扫码日志，与真值逐项比较，得出指标和成功判定。

一次 episode 的数据流：

```
测试场景 JSON（g1_barcode_sim/suites/<套件>/scenarios/*.json）
      │ reset(scenario)                                    ┌─ 算法侧拿不到这条通道 ─┐
      ▼                                                    │                        │
仿真服务器 ──机器人端口──► RobotInterface ◄── 工具 ◄── 编排器（rule / agent_assist / agent_full）
      │                                                       │ 日志事件（tool / llm / stage）
      │◄─────────── 评测端口（需令牌）: stage / annotate ─────┘
      │
      └─评测端口: get_gt / get_actual_state / get_stage_states / get_stage_log / get_scan_log
                     │
                     ▼
        evaluator.evaluate(task, gt, actual, ws, stages, scan_log) ──► result.json ──► results.csv ──► report.md
```

和其他组件的接口：

| 对方 | 方向 | 内容 | 位置 |
|---|---|---|---|
| 仿真服务器评测端口 | 评测侧 → 仿真 | `reset`、`get_gt`、`get_actual_state`、`get_stage_states`、`get_stage_log`、`get_scan_log`、`snapshot`、`record_start/stop`、`set_gt_markers`、`invisibility_check`、`shutdown` | `g1agent/eval/eval_client.py`（`get_stage_log` 由 `scripts/run_task.py` 直接经 `ev.rpc` 调用），服务端在 `g1agent/robot/sim_server.py` |
| Harness / 世界状态 | 算法 → 评测 | `root_target`、`wrist_target`、`wrist_target_refined`（odom 坐标系）；任务 3 另有 `scan_plan`、`scanned`（账本）、`current_barcode_id` | `g1agent/harness/world_state.py`，由 `scripts/run_task.py` 组装成 `ws` |
| Harness 日志 | 算法 → 仿真 | 阶段事件（`task1_end`、`face_start`、`face_end`、`barcode_start` 等）转发给评测端口的 `stage`，其他事件转发给 `annotate`（只用于视频叠加） | `g1agent/harness/logger.py`、`run_task.py` 的 `on_event` |
| 测试场景集 | 仿真包 → 运行脚本 | 场景 JSON、`manifest.json`、`list_suites()`/`load_suite()` | `g1_barcode_sim/suites/`、`g1_barcode_sim/testsuite/` |
| 可视化 | 共用定义 | GT 扫码目标点的 stand-off（`gt_standoff`），METRICS 面板与评测用同一个函数 | `sim_server.gt_standoff_fn`、`g1agent/viz/compositor.py` |
| 汇总报告 | 运行结果 → 报告 | `results.csv` → `report.md`、`all_results.csv` | `scripts/run_suite.py`、`scripts/summarize.py` |

```{figure} images/eval_t2_final_markers.jpg
:width: 85%

任务 2（t2_L0_0000，rule 模式）结束时的第三人称截图 `final.png`。扫码枪前端的黄色箭头是仿真器在评测模式下画出的"实际到达位姿"标记。这类标记只在第三人称渲染中出现，机器人自己的相机看不到。
```

## 当前实现

### 1. 评测通道：`EvalClient` 与评测端口

`g1agent/eval/eval_client.py` 中的 `EvalClient` 是评测端口的客户端，只有评测器和测试集运行脚本使用它。

- 构造参数 `instance`：缺省时取环境变量 `SIM_INSTANCE`，再缺省为 0。端口为 `eval_port + i · instance_port_stride`，即 `47556 + 10·i`。机器人端口同样按 `10·i` 偏移，所以一台机器上可以并行跑多个仿真实例。
- 令牌：从环境变量读取，变量名由 `configs/system.yaml` 的 `eval_token_env` 指定（值为 `G1_EVAL_TOKEN`）。没有令牌时抛出 `RuntimeError`。令牌由启动脚本 `scripts/start_sim_server.sh` 第一次运行时生成，保存在 `runs/.eval_token`；仿真服务器本身在没有令牌时拒绝启动。
- 超时：`rpc_timeout_s = 600 s`。配置注释的理由是，一次长距离行走或一次 episode reset 可能要几分钟墙钟时间。
- 方法都是对 `rpc.call` 的一行封装，直接返回服务器的回复：
  - 场景控制：`reset(scenario)`、`shutdown()`；
  - 读真值与实际状态：`get_gt()`、`get_actual_state()`；
  - 标记与截图：`set_gt_markers(enabled)`、`snapshot(path)`；
  - 录像：`record_start(path, fps, task)`、`record_stop()`；
  - 视频叠加事件：`annotate(event)`、`stage(name, info)`；
  - 阶段与扫码记录：`get_stage_states()`、`get_scan_log()`；
  - 自检：`invisibility_check()`。

  评测端口的 `get_stage_log` 在 `EvalClient` 中没有封装方法，由 `scripts/run_task.py` 经 `_eval_rpc` 直接调用 `ev.rpc`。

服务端（`sim_server.py`）有三处和评测直接相关的记录：

- **阶段记录**：`Episode.stage(name, info)` 每次被调用时，都会把 `{name, info, t_sim, actual}` 追加到 `stage_log`，其中 `actual` 是那一刻的实际状态。`stages` 字典用 `setdefault` 只保留每个阶段名的第一次出现。`get_stage_states()` 返回字典，`get_stage_log()` 返回完整列表。
- **扫码日志**：机器人端口的 `scan()` 先调用仿真的 `env.scan()`，再把下面这些字段记进扫码日志：`t_sim`、`status`、`content`（解码内容）、`decoded`（全部解码结果，带 pyzbar 多边形）、`gt`（仿真给出的诊断，其中 `gt.hit` 是从枪口沿扫描方向发出的射线命中的标签）、`T_world_tip`（枪口在世界系的位姿）、`pelvis_z`、`height_cmd`。算法侧拿到的 `scan()` 返回值不含 `gt` 和世界系位姿。
- **实际状态**：`actual_state()` 通过 `env.frame_state()` 读取并返回根位姿 `T_world_root`、枪口位姿 `T_world_wrist_point`、当前里程计系 `T_world_odom_now`、头部相机位姿 `T_world_headcam` 及其内参 `head_K` / `head_size`，以及 `t_sim`。`frame_state()` 不拷贝图像，也不做深度处理，所以读取实际状态不会扰动深度噪声的随机数流。

**真值隔离**。`tests/test_no_gt_leak.py` 用 AST 静态扫描 `orchestrators/`、`tools/`、`perception/`、`harness/`、`kinematics/`、`llm/` 以及机器人客户端的 `sim_client.py`、`interface.py`、`rpc.py`，以下写法都会让测试失败：

- import `g1agent.eval`、`eval_client`、`evaluator`、`g1_barcode_sim`；
- 以属性、名字或字符串常量的形式出现 `get_gt`、`get_ground_truth`、`get_actual_state`、`EvalClient`、`evaluate`、`T_world_root_gt`、`set_gt_markers`、`task1_gt_obs_pose`、`task2_gt_wrist_target`。

测试还检查 `SimRobot` 类本身没有这些方法。场景 JSON 里也不放真值目标，真值目标在 reset 后由仿真服务器计算。

### 2. 真值的内容（仿真侧提供）

`get_gt()` 返回仿真的 `env.get_ground_truth()`，所有量都在世界坐标系，单位米，四元数按 wxyz。评测用到的字段如下：

| 字段 | 含义 | 用于 |
|---|---|---|
| `robot_init` | 机器人起始位姿 | 任务 3 没有到达 task1_end 时的排序参考系 |
| `target_face` | 目标面（中心、法向、四个角点）。多箱堆垛时是正面（-x）所有未被完全遮挡的箱面的并集 | 任务 1 距离、朝向、"面是否在视野内" |
| `task1_gt_obs_pose` | GT 观察位姿：在 d_obs 范围 [0.60, 1.00] m 内，取能让目标面四角都留边距落进头部相机的最小距离 | 任务 1 感知误差 |
| `task2_gt_wrist_target` | 目标条码中心 + 0.20 × 法向，前向 = −法向 | 任务 2 感知误差 |
| `barcodes[]` | 每个条码的 `content`、`center`、`normal`、`up`、`size`（含白底）、`pattern_size`（只含印刷符号）、`distractor`、所在面 | 任务 2、3 |
| `target_barcode` | 任务 2 的目标条码内容 | 任务 2 |
| `faces` | 多面堆垛中每个有暴露面的侧面（`-x`、`-y`、`+x`、`+y`），带各自的中心、法向和 GT 观察位姿 | 任务 3 多面排序 |

### 3. 总入口 `evaluate`

`g1agent/eval/evaluator.py`：

```python
evaluate(task, gt, actual, ws, stages=None, scan_log=None)
```

- `task == 1` → `_task1(gt, actual, ws, task1.yaml 的 eval)`；
- `task == 2` → `_task2(gt, actual, ws)`；
- `task == 3` → `_task3(...)`，缺少 `stages` 或 `scan_log` 时抛出 `ValueError`；
- 其他任务号抛出 `ValueError`，不会默认按某个任务处理。

`ws` 由 `run_task.py` 组装：

- 任务 1、2：`root_target`、`wrist_target`、`wrist_target_refined`；
- 任务 3 另加 `scan_plan`、`scanned`、`current_barcode_id`，以及 `task1_end_root_target`。最后这一项是 `task1_end` 阶段触发那一刻，对 Harness 根目标做的深拷贝。

### 4. 任务 1 指标：`_task1`

任务 1 要求机器人走到目标箱面前的观察位姿。指标分三组：

**(1) 实际到位情况**

- `d_actual_m`：实际根位置到 GT 目标面平面的水平距离，由 `geo.horizontal_distance_to_plane` 计算。
- `yaw_err_deg`：先把面法向 n 取水平分量并归一化，得到正对该面的朝向 `yaw_face = atan2(-n_y, -n_x)`；再取实际根朝向与 `yaw_face` 之差的绝对值。
- `face_in_view_gt`：用实际头部相机的内参 `head_K` 和位姿 `T_world_headcam`，把 GT 目标面的 4 个角点投影到图像上，全部落在图内（边距 0）时为真。

**(2) 误差分解**

误差分解的目的是区分"看错了"和"走错了"。

- Harness 没有根目标时直接判失败，原因为 `"no root target estimated"`。
- `d_err_m = |d_actual − rt.d_obs|`：实际距离与 Harness 自己选定的观察距离之差。
- 感知误差 `perception_err_pos_m` / `perception_err_yaw_deg`：Harness 根目标用 `T_world_odom_now` 变到世界系后，与 `task1_gt_obs_pose` 的差。在视频标记里，这对应红色标记（Harness 根目标）与绿色标记（GT）之间的差。
- 执行误差 `execution_err_pos_m` / `execution_err_yaw_deg`：实际根位姿与 Harness 根目标（世界系）的差，对应黄色（实际）与红色之间的差。

**(3) 成功判定**

`d_err < dist_tol_m`、`yaw_err < yaw_tol_deg`、`face_in_view_gt` 三者同时成立才算成功。感知误差和执行误差只报告，不参与判定。

- 容差在 `configs/task1.yaml` 的 eval 段：`dist_tol_m = 0.05`，`yaw_tol_deg = 5.0`。数值来自任务书，配置注释没有另给依据。
- Harness 自己的到位检查更严：0.03 m / 3.0°。任务书给出的理由是 Harness 检查基于估计值，评测基于真值，所以评测容差放宽。

### 5. 任务 2 指标：`_task2` 与 `wrist_metrics`

任务 2 要求把扫码枪枪口放到目标条码前 0.20 m 处。

`wrist_metrics(T_world_tip, center, normal, offset_m)` 是任务 2 和任务 3 共用的几何函数。扫码枪位姿的 +x 轴是扫描方向。它返回：

- `dist_to_plane_m`：沿法向的距离 `(p − c)·n`；
- `axis_offset_m`：枪口偏离标签轴线的距离，即 `p − c` 去掉法向分量后的长度；
- `angle_err_deg`：扫描方向与 −n 的夹角；
- `tip_err_m`：枪口到目标点 `c + offset·n` 的三维距离；
- `target_offset_m`：本次使用的 offset。

`_task2` 的计算：

- 目标条码是 GT 中 `content == target_barcode` 的那一个；offset 取 `configs/task2.yaml` 的 `offset_m = 0.20`（固定 stand-off）。
- 输出 `wrist_dist_to_plane_m`、`wrist_axis_offset_m`、`wrist_angle_err_deg`。
- 感知误差：
  - `head_perception_err_m`：头部相机估计的 `wrist_target`（蓝）与 `task2_gt_wrist_target`（绿）的距离；
  - `wrist_perception_err_m`：腕部相机修正后的 `wrist_target_refined`（紫）与绿的距离。
- 执行误差 `execution_err_m`：实际枪口位置（黄）与最终 Harness 目标的距离。最终目标优先取 refined，没有时取头部估计。
- 成功判定：`|dist − 0.20| ≤ dist_tol_m`、`axis_offset < axis_offset_max_m`、`angle < angle_max_deg` 三者同时成立。`configs/task2.yaml` eval 中的值为 0.03 m / 0.03 m / 15°，来自任务书；Harness 侧的检查是 0.02 / 0.02 / 10°。

### 6. 任务 3 指标：`_task3`

任务 3 是合并任务：先完成任务 1 的找箱和接近，然后把场景中每个非干扰条码各扫一次。顺序规定为"列从左到右，列内从上到下"；多面堆垛还要按 前 → 右 → 后 → 左 绕行。

#### 6.1 GT stand-off：每个条码的目标距离

任务 3 的扫码距离不是固定的 0.20 m，而是按条码印刷符号的宽度自适应计算。评测侧用同一个公式代入真值尺寸：

- `scanner_hfov_gt()`：读取仿真包 `configs/scanner.yaml` 的 `camera` 段，计算 `fx = focal_length / horizontal_aperture · width`，再得 `hfov = 2·atan(width/2/fx)`。仿真的扫码相机按 DS2208 类成像器设定，结果为 32.8°。
- `gt_symbol_size(bc)`：返回 GT 中的 `pattern_size`，即印刷的符号本身，不含白色底板。缺这个字段时抛出 `ValueError`，不做回退，因为 stand-off 规则是按符号定义的。
- `gt_standoff(symbol_size, hfov)`：取符号的长边作为宽度（扫码枪的 roll 不受约束，必须装进图像宽度的是较长的那一维），代入 `g1agent/eval/scan_order.py` 的 `adaptive_standoff`：

  `d = width / (2·tan(hfov/2)) / fill_frac`，再截断到 `[min_m, max_m]`。

  参数取自 `configs/task3.yaml` 的 standoff 段：
  - `fill_frac = 0.68`：让标称 0.08 m 的 code128 符号落在已经验证过的 0.20 m（fx 1087、宽 640 px 时 hFOV 32.8°，0.08 / (2·tan 16.40°) / 0.20 = 0.679）；
  - `min_m = 0.12`：设计需求第 5 条的决定；
  - `max_m = 0.30`：配置注释没有给出依据。

仿真服务器的 `gt_standoff_fn` 和视频 METRICS 面板也调用这个函数：任务 2 返回固定的 `offset_m`，任务 3 返回自适应值。评分和可视化因此用的是同一个定义。

#### 6.2 任务 1 阶段

- `stages` 中没有 `task1_end` 时：`task1.success = False`，原因为 "the run never reached the task1_end stage"。
- 有 `task1_end` 时：用该阶段保存的 `actual` 和 `ws["task1_end_root_target"]` 套用任务 1 的公式，容差取 `configs/task3.yaml` eval（0.05 m / 5.0°，注释写明与 task1.yaml 相同）。
- 结果平铺为 `t1_success`、`t1_d_err_m`、`t1_yaw_err_deg`、`t1_face_in_view_gt`、`t1_perception_err_pos_m`、`t1_execution_err_pos_m`。
- **任务 1 阶段不参与任务 3 的成功判定**，只单独报告。原因是：任务 3 的任务 1 阶段按设计会提前结束（approach 走完即进入扫码），扫码阶段会为每个条码自己规划站位（2026-11-06 的决定）。

#### 6.3 期望集合与 GT 顺序

- 期望集合是所有 `distractor == False` 的 GT 条码。它们的 content 必须互不相同，否则抛出 `ValueError`，因为按 content 评分在这种情况下没有定义。
- **单面**（GT 没有 `faces` 字段，或只有一个面带标签）：
  - 用 `order_labels_in_frame` 在参考系中套用列规则；
  - 参考系取 task1_end 时的实际根位姿（`order_frame = "task1_end"`）；
  - 没有到达 task1_end 时，用 `robot_init` 构造参考系（`order_frame = "robot_init"`）。
- **多面**（不止一个面带标签）：
  - 用 `labelled_faces` 把标签分到各面，每个面在自己的理想坐标系中排序，再按环绕顺序 `-x → -y → +x → +y` 拼接；
  - `order_frame = "faces"`。
- 输出 `gt_order`、`gt_standoff_m`（每个条码的 GT stand-off）、`scanner_hfov_deg`。

#### 6.4 列规则（`g1agent/eval/scan_order.py`）

评测侧把顺序规则独立实现了一份，评测器因此不 import 算法代码（`g1agent/tools/scan_order.py`）。`tests/test_task3_logic.py::test_column_order_matches_evaluator_copy` 用 300 组随机输入检查两份 `column_order` 的输出完全相同。

- `label_corners(center, normal, up, size)`：求平面标签的 4 个角点，右方向取 `up × normal`。
- `order_labels_in_frame(labels, T_world_frame)`：把每个标签的 4 个角点变换到参考系，取 y 的最小、最大值作为 `y_lo`、`y_hi`（左为 +y），取中心的 z，然后调用 `column_order`。
- `column_order(items)`：
  1. 按 `(-y_hi, -y_lo, str(key))` 排序，从最左边的标签开始；
  2. 当前项的 `y_hi` 大于当前列已有的最小 `y_lo` 时并入当前列并更新最小值，否则开新列。重叠按严格不等判断，边缘恰好相接的两个标签分属两列。这样得到的就是"横向范围重叠"关系的连通分量，链式重叠（A 与 B、B 与 C 重叠）也算一列；
  3. 列内按 `(-z, -横向中心, str(key))` 排序，即从上到下，并列时按确定的规则打破。
- 多面部分：
  - `FACE_TOUR = ("-x", "-y", "+x", "+y")`，`FACE_NAMES = ("front", "right", "back", "left")`。从上方看是逆时针。索引 i 就是阶段事件 `face_start`/`face_end` 中的 face 索引，也是扫描计划中的 `face_id`；
  - `face_frame(normal, origin)`：一个面的理想坐标系，forward = −n（水平分量），left = n × z，up = z，原点是该面的 GT 观察位姿。在这个坐标系里套用列规则，得到的就是从该面正前方看过去的"左 → 右、上 → 下"；
  - `assign_faces`：每个标签归到法向点积最大的面。标签法向倾斜 30° 时仍归到正确的面（有单元测试）；
  - `labelled_faces(labels, faces)`：面名不在 `FACE_TOUR` 中时抛出 `ValueError`；按环绕顺序排序后分配标签，只返回带标签的面 `[(face, [按顺序排列的 key])]`。

#### 6.5 逐次读码

遍历 `scan_log`，每条记录对应一次 `scan()` 调用：

- 记录 `content`、`decoded`、`aimed_content`（`gt.hit` 中射线命中的 content）、`new`（该 content 第一次出现）、`in_gt`。
- 选参照条码 `ref`：解码内容在 GT 中就用它（`ref_source = "decoded"`）；否则射线命中的 content 在 GT 中就用它（`ref_source = "scan_ray"`）。所以"瞄准了但没读出"的那次扫码也能算出放置误差。
- 有 `ref` 时，用 `wrist_metrics` 计算枪口相对 GT 目标点（中心 + d_gt·n）的误差，再求 `placement_ok`：沿用 `task2.yaml` eval 的三个容差，但目标距离换成 d_gt。
- 同一个 content 只在第一次读到时计数。

#### 6.6 汇总字段

| 字段 | 含义 |
|---|---|
| `n_gt_barcodes` | 期望条码数 |
| `n_scan_calls` | scan 调用次数 |
| `n_no_read` | 没有读出内容的次数 |
| `n_scanned_distinct` | 读到的不重复 GT 条码数 |
| `scan_recall` | 上一项除以期望数 |
| `wrong_contents` / `n_wrong_contents` | 读到的非 GT 内容（外来内容） |
| `n_duplicate_reads` | 有内容的读码次数减去不重复内容数 |
| `n_read_not_aimed` | 解码内容与射线命中内容不一致的次数 |
| `scan_order` | 按首次读到的时间排列的 GT 内容 |
| `order_ok` | `scan_order` 与 `gt_order` 完全一致 |
| `order_ok_scanned` | 只看已读到的那些，相对顺序是否正确 |
| `tip_err_mean_m` / `tip_err_max_m` | GT 条码首次读码时的枪口误差 |
| `n_placement_ok` | 首次读码中放置合格的条数 |
| `reads` | 每次读码的完整记录 |
| `n_planned` / `plan_status` | 扫描计划条目数，以及各状态（scanned / failed / unreachable / unreadable 等）的计数 |
| `reported_unreachable` | 状态为 unreachable 的条目，含 `id`、`center_odom`、`reason` |
| `ledger_contents` / `ledger_matches_log` | Harness 账本中的内容序列，以及它是否等于日志中按首次读到排列的不重复内容序列 |

多面场景另有 `_tour_metrics(groups, scanned_gt)` 的输出：

- `per_face`：每个面的 GT 顺序、读码顺序、recall，以及 `order_ok`（该面全部读到且顺序正确）；
- `face_order`：依次取每次首读所在的面，把连续重复的面合并；
- `faces_contiguous`：`face_order` 中没有重复的面，即一个面扫完才去下一个面；
- `front_first`：第一个读到的面是第一个带标签的环绕面；
- `face_order_ok`：`face_order` 等于所有带标签面的环绕顺序（只报告）。

#### 6.7 成功判定

下面的失败项一条都没有出现才算成功，失败原因用 `"; "` 连接写进 `failure`：

| 条件 | 失败原因文本 | 适用 |
|---|---|---|
| 读到的 GT 条码少于期望数 | `scanned k/n GT barcodes` | 全部 |
| 全部读到但顺序不对 | `scan order ... != GT order ...` | 单面 |
| 全部读到，但某个面内部顺序不对 | `face X order ... != GT ...` | 多面 |
| 首个读到的面不是第一个带标签的环绕面 | `first face read ... is not the first labelled tour face ...` | 多面 |
| 某个面被分成多段扫描 | `faces not scanned contiguously: ...` | 多面 |
| 读到非 GT 内容 | `read foreign contents [...]` | 全部 |

- 多面场景的全局 `order_ok` 和 `face_order_ok` 只报告，不参与判定。例如顺时针绕行时只记录顺序，仍判成功（有单元测试）。
- 重复读码、no_read、放置不合格、任务 1 阶段失败都只报告，不参与判定。

### 7. 测试场景集

测试场景由仿真包生成和保存，路径为 `g1_barcode_sim/suites/<套件名>/`。每个套件有 `manifest.json`（不含时间戳，生成时间另记在 `generation_log.json`）、`scenarios/*.json` 和 `previews/`（俯视图和头部视图预览）。

#### 7.1 参数化场景 `table_two_boxes`（任务 1、2）

- 布局：一张与机器人平行的桌子，桌沿在 x = 1.30；桌上并排两个箱子，条码贴在朝 −x（朝向机器人）的箱面上。每个场景由一个 JSON 完整描述。
- 因素和范围在 `g1_barcode_sim/configs/testsuite/factors.yaml`，例如：
  - 机器人到桌距离标称 3.0 m，范围 2.0–5.0 m；横向偏移 −1.5–1.5 m；
  - 箱子 w 0.40 [0.20, 0.60]、h 0.30 [0.15, 0.50]、d 0.30 [0.20, 0.50]，外观 kraft / kraft_tape / white_box / printed；
  - 桌子 depth 0.60 [0.50, 0.80]、width 1.20 [1.00, 1.80]、height 0.75 [0.70, 0.80]。
- 确定性：每个场景由 `(套件名, seed, 序号, 第几次尝试)` 经 SHA-256 派生出独立的随机数流。每个套件在内存里生成两次，必须逐字节一致。项目目录不是 git 仓库，所以 manifest 用 `code_hash` 代替 commit。
- 有效性：
  - 机器人占地圆半径 0.35 m，不能与桌子、箱子重叠；
  - L0 网格和 L2 中单个场景重采样 50 次仍无效就报错；
  - L1 中几何上不可行的扫描点跳过，记入 `skipped_infeasible`；
  - L1 中箱子放不下时自动放大桌子，记入 `factor.table_adjusted`。
- 任务 2 的起始位姿：
  - 基准是 GT 观察位姿，用实测的头部相机标称参数 `configs/testsuite/head_camera_nominal.yaml`（离地 1.26 m，向下约 48°）计算；
  - 噪声：L0 为 0，其他级别 ±0.03 m / ±3°，`start_noise` 因素线性放大到 ±0.08 m / ±8°；
  - 横向偏移先在 [−0.4, 0.4] m 内以 1 cm 步长求出可见区间，再在区间内取值；
  - 最后检查箱面四角和条码四角都在画面内。
- 条码内容是按种子生成的大写字母数字随机串，去掉 I、O、0、1；可读编号另存为 `content_id`。
- 灯光按实测标定：none 背景从 3000/1000 改成 280/90 后，灰色地面在头部相机中约 150/255（原来接近白色，233/255）；white 背景取 1000/300，保证 1.3 倍光强下也不会截断到 255。

#### 7.2 套件一览

各 `manifest.json` 均为 version 1、seed 0：

| 套件 | 任务 | 级别 | 场景数 | 说明 |
|---|---|---|---|---|
| `t1_L0` | 1 | L0 | 8 | 网格：`dist_to_table` {2,3,4,5} × `target` {left,right} |
| `t1_L1_background` / `barcode_size` / `box_appearance` / `box_shape` / `lighting` / `placement` / `robot_dist` / `robot_lateral` | 1 | L1 | 12 / 10 / 12 / 12 / 10 / 12 / 12 / 13 | 每个套件只改一个因素 |
| `t1_L2` | 1 | L2 | 100 | 拒绝采样：`boxes_wider_than_table` 32，`box_depth_exceeds_table` 10 |
| `t2_L0` | 2 | L0 | 8 | 网格：`group_y` {−0.15, −0.05, 0.05, 0.15} × `target` |
| `t2_L1_background` / `barcode_pos` / `barcode_size` / `barcode_type` / `box_lateral` / `start_noise` | 2 | L1 | 12 / 12 / 10 / 12 / 12 / 12 | |
| `t2_L1_box_shape` | 2 | L1 | 10（定义 n = 12） | index 10、11（因素值 0.909、1.0）各尝试 50 次仍不可见，跳过 |
| `t2_L2` | 2 | L2 | 100 | 拒绝：`boxes_wider_than_table` 44，`box_depth_exceeds_table` 21，`task2_not_visible` 20；`t2_L2_0056`、`0057` 找不到留足边距的距离，d_obs 取上界 |
| `stack3_L0` | 3 | L0 | 2 | 手写，三箱落地竖直堆叠 |
| `stack9_L0` / `stack9p_L0` / `stack9c_L0` | 3 | L0 | 1 / 1 / 1 | 手写，3×3 单面箱堆，三种外观 |
| `stack4f_L0` | 3 | L0 | 3 | 手写，2×2×3 四面箱堆，三种起始位姿 |
| `stack4c_L0` | 3 | L0 | 3 | 手写，四面箱堆，快递纸箱外观 |

合计：任务 1 共 201 个场景，任务 2 共 188 个，任务 3 共 11 个。任务 1、2 的场景也可以用 `--task 3` 当任务 3 跑（例如 `t1_L0_0000`、`t2_L1_barcode_size_0001/0009`）。

#### 7.3 任务 3 的手写场景

任务 3 场景都是手写的。`stack3_L0` 的场景说明里写明，生成器还不支持地面纸箱堆。这些场景的 `code_hash` 为 `null`。

| 场景 | 布局 | 条码 | 测什么 |
|---|---|---|---|
| `stack3_L0_0000` | 无桌，三个 0.40×0.40×0.38 m 箱子竖直堆叠，机器人在 2.5 m 外 | 3 个，高 1.10 / 0.57 / 0.30 m | 举手、下蹲、深蹲扫码；一列内从上到下 |
| `stack3_L0_0001` | 同上 | 底层码保留在 0.19 m | 够不到的码必须报告为 unreachable，而不是硬扫 |
| `stack9_L0_0000` | 3 列 × 3 层共 9 箱，宽 1.2 m、高 1.14 m，素色牛皮纸 | 5 个：L2 1.10 m、R2 1.07 m、C1 0.57 m、L0 0.30 m、R0 0.32 m | 多列顺序，GT 顺序为左上 → 左下 → 中 → 右上 → 右下 |
| `stack9p_L0_0000` | 同 stack9，外观在 printed（印满 FRAGILE 等文字）和 kraft_tape 之间交替 | 位置与 stack9 相同 | 印刷图案和胶带会不会造成误检、漏检 |
| `stack9c_L0_0000` | 同 stack9，快递纸箱模型外观 | 5 个，-x 面 | 纸箱外观下的单面扫码 |
| `stack4f_L0_0000` | 2×2×3 共 12 箱，正面 x = 1.30，`scan_faces: "all"` | 12 个，每面 3 个（上 / 中 / 下） | 四面绕行、各面内顺序 |
| `stack4f_L0_0001` | 同上；机器人在中线右侧 0.5 m、左转 30° | 同上 | 起始朝向与箱堆不平行 |
| `stack4f_L0_0002` | 同上；机器人在中线左侧 0.7 m（已超出 +y 侧面）、右转 35° | 同上 | 同上 |
| `stack4c_L0_0000` | 四面箱堆，快递纸箱模型 | 12 个，4 面 | 纸箱外观的四面绕行 |
| `stack4c_L0_0001` | 同上 | 6 个，只贴在 -x、+x 两面 | 无码侧面、纸箱自带的面单贴纸（含假条码图案） |
| `stack4c_L0_0002` | 同上 | 60 个，每个箱面 5 个，只在两面 | 高密度标签 |

不平行起始位姿的偏移量和角度是我们手选的测试条件，选取原则是箱堆正面仍在头部相机视野和 3 m 深度范围内，不是算法规则。

```{figure} images/eval_stack4c_stickers.jpg
:width: 85%

`stack4c_L0_0001` 的快递纸箱外观。左上方白底条码是场景的真码，箱面上的 PARCEL POST 面单是纸箱模型自带的贴图，上面也印着条码图案，但读不出内容。机器人从头部相机无法区分两者，只能走近核验。
```

### 8. 跑一个 episode：`scripts/run_task.py`

`make_episode_runner(device)` 只加载一次 `SimRobot`、`EvalClient`、`BoxDetector(device)` 和全部配置；LLM 客户端在首次进入 agent 模式时才创建。返回的 `run(scenario, mode, ep_dir, video=True, task=None)` 依次执行：

1. 加载场景。`task` 可以覆盖场景自带的任务，只影响 agent 侧和评测侧；仿真器仍按原场景 reset。任务号不在 (1, 2, 3) 中时抛出 `ValueError`。
2. 注册日志回调 `on_event`：
   - stage 事件发给评测端口的 `stage`，其他事件发给 `annotate`；
   - `task1_end` 触发时深拷贝 `tb.ws.root_target`，存为 `task1_end_root_target`；
   - `stage`、`annotate` 以及第 7 步的 `get_stage_states`、`get_scan_log`、`get_stage_log` 都经过 `_eval_rpc`：回复为 `{"status": "error"}` 时抛出 `RuntimeError`，不把错误回复当作普通数据返回。`reset`、`get_gt` 等其余调用直接用 `EvalClient` 的方法。
3. `reset` → `set_gt_markers(True)` → 需要视频时 `record_start(episode.mp4, task)`。
4. 构造 Toolbox。只有任务 3 且场景 `scan_faces == "all"` 时才设 `tb.multi_face = True`。这是任务指令，不是真值。
5. 构造 Harness，参数为任务、预算和机器人运动限制。选择器：任务 3 用 `selector_default`（`all`），任务 1、2 用场景给定的 left / right。
6. 按模式分派到编排器 `rule`（Task1 / Task2 / Task3）、`agent_assist`、`agent_full`。
   - 编排器抛出的任何异常都记为本 episode 失败，写入 `orchestrator_error.txt`（含 traceback）；
   - 如果是 `LLMTruncated`，另外记录各次尝试；
   - 这里有意使用宽泛 `except`（系统中另外两处是仿真服务器的 RPC 分发和 Harness 的工具调用包装，同样把错误作为结构化失败返回）。它不吞掉错误：错误写进结果，episode 显式标为失败，批量运行可以继续跑下一个场景。
7. `record_stop` → `snapshot(final.png)` → 取 GT、实际状态、`get_stage_states`、`get_scan_log`、`get_stage_log`。
8. 组装 `ws`，调用 `evaluate`。
9. 写出 `result.json`，字段为：scenario、task、scenario_task、mode、harness_finish、orchestrator_error、tool_calls、explore_steps、wall_s、llm（统计）、world_state、metrics、stages、stage_log、scan_log、factor、level。
10. `build_timeline(ep_dir)` 生成时间线，然后把 LLM 统计清零，留给下一个 episode。

每个 episode 目录下有 `tool_calls.jsonl`、`llm.jsonl`、`stages.jsonl`、`images/`、`result.json`、`final.png`、`episode.mp4`、`timeline.md`、`timeline.jsonl`。

命令行参数：`--scenario`（必填）、`--mode`（默认 rule）、`--run_dir`、`--no_video`、`--device`、`--task`。`--run_dir` 是 run 目录，episode 写在它下面以场景名命名的子目录里；默认为 `runs/<时间戳>_<scenario>_<mode>[_taskN]`。

### 9. 批量运行

#### 9.1 `scripts/run_suite.py`：运行测试集，支持断点续跑

- 套件名展开：`all` 表示全部已生成的套件；其他名字匹配同名套件，或以 `名字_` 开头的套件，例如 `t1_L1` 展开为所有 `t1_L1_<因素>`。没有匹配时退出。
- `--scenarios`：只跑列出的场景 id；有 id 不在所选套件里时退出。
- `--task`：覆盖所有场景的任务号。
- 输出：
  - `runs/<run_name>/run.json`：套件、模式、任务覆盖、场景数、argv、SIM_INSTANCE；
  - `results.csv`：每跑完一个 episode 整体重写一次，所以中途停掉也不会丢已完成的行。
- 断点续跑：场景目录下已有 `result.json` 时直接读取，不重跑。读到的任务号与本次要求不一致时退出，提示换一个 `--run_name`。
- 检测器线程数：`torch.set_num_threads(DETECTOR_THREADS)`，环境变量默认 16，代码中没有给出依据。`--device` 默认 cpu。`--video` 默认关闭，截图总是保存。
- 可选的进度上报：设了外部进度客户端目录时，逐 episode 上报进度。
- `COLS`（CSV 列）：通用列、任务 1 和 2 的误差列、任务 3 的列（`t1_*`、`n_gt_barcodes` … `ledger_matches_log`）、调用和 token 统计、`llm_length_truncations`、`wall_s`、`failure`。
- `row(sc, res)` 中 `failure` 的取值优先级：编排异常 > Harness 的自述 summary > 指标中的 failure。指标字段只填入 `COLS` 中有、且行里还没有的列，`metrics["failure"]` 因此不能覆盖 episode 级的失败原因。

#### 9.2 多实例并行

仿真比实时慢：v15 实测 3 个实例并行、带录像时约 0.2 倍实时，不录像时测过约 0.32 倍实时（物理在 CPU 上）。所以批量评测靠并行多个仿真实例：

- 每个实例设不同的 `SIM_INSTANCE=i`，端口加 `10·i`；
- 各实例的套件或场景按负载分配，由批量启动脚本 `scripts/run_batch_*.sh` 起服务、跑客户端、收尾。

`scripts/select_agent_subset.py`：根据已完成的 rule 结果，为每个套件挑 K 个场景给 agent 模式用。

- K = 2 是 2026-11-06 的决定；传入其他 K 时退出，因为挑选策略只针对 K = 2 定义。
- 覆盖前缀为 `t1_L0`、`t2_L0`、`t1_L1_`、`t2_L1_` 的全部套件；rule 没有跑到的套件视为"未观察到失败"。
- 挑选策略：
  - 有失败也有成功：取第一个失败，加上因素值离它最近的一个成功（因素值不是数值时用序号代替）。用意是同时回答两个问题：LLM 能否救回状态机救不了的场景，以及能否保住状态机本来能完成的场景；
  - 全部失败或全部成功：取首尾两个场景，即因素范围的两端；
  - 去重；套件只有一个场景时只选一个。
- 输出 JSON `{suite: [ids]}`。

`scripts/build_agent_clients.py`：

- 贪心分配：套件按场景数降序、名称升序排列，每个依次放进当前负载最小的实例。
- 每个实例先跑 `agent_assist`（run 名 `m5_assist_gear_i`），再跑 `agent_full`（`m5_full_gear_i`）；还没有 rule 结果的场景最后补跑一次 rule（`m5_rule_gear_extra_i`），这样三种模式在相同场景上可比。agent 模式排在前面，是按需求的优先级。
- 命令串用 `;` 连接，交给批量启动脚本执行。

`scripts/record_episodes.py`：按列表逐个场景运行并录视频。已有 `result.json` 和 `episode.mp4` 时跳过，但会核对任务号：不带 `--task` 时期望值取 `scenario_task`；更早的 result.json 没有这个字段，此时取 `task`，因为当时总是按场景自身的任务运行。`--until_success` 在第一个成功的 episode 后停止。

### 10. 汇总与回看

#### 10.1 `scripts/summarize.py`

- 读取多个 run 的 `results.csv`，合并时给每行加上 run 名。
- 生成 `report.md`：
  - 每个任务一张"级别 × 模式"表：样本数、评测成功率、Harness 自称成功率、平均 wall_s、平均工具调用数、平均 LLM 调用数、误差列的中位数；
  - L1：各因素在各模式下的成功率表，以及每个因素一张"成功率随因素取值变化"的图。因素值能转成数值时按数值画；否则（只捕获 `ValueError` / `TypeError`）按类别等距排列，刻度标签截到 14 个字符；图尺寸 5×3，dpi 110；
  - 失败原因计数；
  - LLM 截断率：`length_truncations` 总和 / 调用总数。项目的设计原则是不给 LLM 设输出长度上限，截断的回复不解析、重新生成，并且每次运行都要报告截断率。
- 同时输出 `all_results.csv`。
- 现状：误差列字典 `err_cols` 只有任务 1、2 两个键，输入中有任务 3 的行时会抛出 `KeyError`。任务 3 的结果目前直接看 `results.csv` 和 `result.json`。

#### 10.2 时间线与剪辑

评测器给出"成功与否"，但只看指标会误判（见演进记录 v10）。所以每个 episode 还会生成可读的时间线：

- `scripts/make_timeline.py` 把 `llm.jsonl`、`tool_calls.jsonl`、`stages.jsonl` 按时间合并成 `timeline.md` / `timeline.jsonl`。
  - 时间零点 `t0 = t_unix − t_wall`。
  - 每次 LLM 调用记录：真实起止时间、这次新给的消息（最后一条 assistant 之后的消息，图片按文件名列出）、每个 prompt 版本第一次出现时的 system prompt、思考内容和 reasoning、工具调用和参数（解析出错时附原始参数）、finish_reason、token 用量、尝试次数和其中被截断的次数。
  - 工具事件带 `rejected`、`budget_exhausted` 标记。
  - 排序键为 (开始时间, `log_seq`, seq)。
  - 消息全文保留，不截短。
  - 兼容没有 `t_start`、`log_seq`、`attempts` 的旧日志。
- `scripts/make_barcode_clips.py`：从任务 3 的视频中按 `barcode_start` … `barcode_end`（可选按面 `face_start` … `face_end`）剪片段。用 `episode.frames.jsonl` 中的 `t_unix` 二分查找，把阶段时间映射到帧号，片段末尾多取 5 帧。文件名中带条码序号、面号、计划中心高度和最终状态。
- `scripts/make_highlight.py`：给每个 episode 加标题后拼成集锦。

### 11. 与评测相关的测试

| 文件 | 内容 |
|---|---|
| `tests/test_task3_eval.py` | 列规则（重叠成列、链式重叠、恰好相接不成列）；自适应 stand-off（hFOV 32.8°，名义 0.20 m，上下截断）；机器人掉头 180° 后左右翻转；任务 3 评测：成功、顺序错误、缺 task1_end 时退回 robot_init 且失败原因不提任务 1 阶段、外来内容判失败、未知任务号报错；四面场景：每面各自排序、法向倾斜 30° 仍归到正面、完整环绕判成功、顺时针只报告、四种失败（不连续、首个面不对、面内顺序反了、少读一个）、只有一个面带标签时结果与无 `faces` 完全相同 |
| `tests/test_run_task_plumbing.py` | 用假的 robot、评测端口、检测器、Toolbox、Task3 跑真实的 `run(..., task=3)`：任务号被覆盖；`task1_end_root_target` 是阶段触发时的快照（d_obs 0.6），而最终 `root_target` 是之后重算的值（0.3）；stage 发往 `stage`、tool 发往 `annotate`；调用了三个日志接口；各文件写出；`_eval_rpc` 遇 error 回复抛异常 |
| `tests/test_no_gt_leak.py` | 算法代码访问不到真值的静态隔离检查（见第 1 节） |
| `tests/test_markers_invisible.py` | 连接正在运行的仿真服务器（连不上时跳过），检查诊断标记对机器人相机不可见：对照组"显示 vs 移除"时探针可见；"隐藏 vs 移除"时没有像素差超过 `STRONG = 60` 个灰度级；第三人称 snapshot 前后，头部和腕部图像完全相同（最大差 0） |
| `tests/test_sim_server_episode.py` | `gt_standoff_fn`：任务 2 返回固定 `offset_m`，任务 3 返回自适应值，缺 `pattern_size` 时报错且不回退；`stage_log` 记录每一次出现，`stages` 只保留第一次 |
| `tests/test_task3_logic.py` | 其中 `test_column_order_matches_evaluator_copy` 用 300 组随机输入核对算法侧和评测侧的 `column_order` |

`STRONG = 60` 的依据：仿真器实时渲染在相邻帧之间不是逐位一致的（时间抗锯齿），实测同一状态下的噪声在少数像素上可达 55 个灰度级，而标记造成的变化超过 100 个灰度级。

### 12. 关键常量

| 位置 | 常量 | 值 | 依据 |
|---|---|---|---|
| `configs/task1.yaml` eval | dist_tol_m / yaw_tol_deg | 0.05 m / 5.0° | 任务书；注释未另给 |
| `configs/task3.yaml` eval | 同上 | 0.05 / 5.0 | 注释："same as task1.yaml eval" |
| `configs/task2.yaml` | offset_m | 0.20 m | 任务书；仿真实测 0.20 / 0.30 m 能解码 |
| `configs/task2.yaml` eval | dist_tol_m / axis_offset_max_m / angle_max_deg | 0.03 / 0.03 / 15° | 任务书 |
| `configs/task3.yaml` standoff | fill_frac | 0.68 | 名义 0.08 m 符号落在已验证的 0.20 m |
| 同上 | min_m / max_m | 0.12 / 0.30 m | min：设计需求第 5 条；max：未给出 |
| `evaluator.scanner_hfov_gt` | hFOV | 32.8° | 由仿真扫码相机定义推导 |
| `configs/system.yaml` | eval_port / instance_port_stride / rpc_timeout_s | 47556 / 10 / 600 s | 本机默认端口被占用；超时：长距离行走或 reset 可能要几分钟 |
| `run_suite.py` | DETECTOR_THREADS 默认值 | 16 | 未给出 |
| `select_agent_subset.py` | K | 2 | 2026-11-06 的决定 |
| `summarize.py` | 因素标签截断 / 图尺寸 / dpi | 14 字符 / 5×3 / 110 | 未给出（只影响显示） |
| `make_barcode_clips.py` | 尾部多取帧数 | 5 帧 | 未给出 |
| `tests/test_markers_invisible.py` | STRONG | 60 灰度级 | 实测同状态噪声 ≤ 55，标记造成的变化 > 100 |

## 演进记录

### v0 · 2026-11-05 · 测试场景集生成器与 table_two_boxes 场景

**初版设计**。按测试集规格实现了参数化场景 `table_two_boxes`：

- 一张与机器人平行的桌子，桌沿放在 x = 1.30，与仿真自带场景 s1_table 的开阔通道相同；桌上并排两个箱子，条码贴在朝向机器人的箱面上。
- 因素表在 `configs/testsuite/factors.yaml`。
- `testsuite/geometry.py`（纯 numpy）负责有效性检查、任务 2 起始位姿，以及评测要用的 `task1_gt_obs_pose`、`task2_gt_wrist_target`。
- `testsuite/scenario.py` 把场景 JSON 转成仿真已有的 `layout.expand_scene` 格式，仿真、GT 导出、间隙模型和可见性代码都不用改。
- 标签贴面偏移 1.5 mm；贴图分辨率 8000 px/m，保证 4 cm 宽、20 字符的 code128 每个模块至少 2 px。

**遇到的问题**

1. 任务 2 套件要从 GT 观察位姿出发计算起点，需要头部相机的真实安装参数。仿真实测：离地 1.26 m，光轴向下约 48°。写入 `configs/testsuite/head_camera_nominal.yaml`。
2. 第一次生成 `t2` 时报错：`t2_L0` 的 `group_y` 网格 [−0.30, −0.10, 0.10, 0.30] 在名义箱子尺寸下放不下。根因是两个箱子组的横向偏移受桌宽约束，|group_y| ≤ (1.20 − 0.90)/2。改为 [−0.15, −0.05, 0.05, 0.15]。
3. L1 扫描中几何上不可行的点（例如箱面太大，在 d_obs 最大值处也看不全）会让生成报错。改为跳过并记入 manifest 的 `skipped_infeasible`。`t2_L1_box_shape` 的两个最大档就是这样跳过的。
4. 规格中"内容长度 6–20 个字符"这个因素与可读格式（本身就有 12 个字符）冲突。条码内容改为随机串，可读编号另存为 `content_id`。
5. none 背景在原灯光 3000/1000 下，灰色地面在头部相机里接近白色（233/255）。改成 280/90 后约 150/255；white 背景取 1000/300。

**结果**：共 18 个套件，任务 1 有 201 个场景，任务 2 有 188 个；每个套件生成两次逐字节一致，预览图全部生成。与规格不一致的地方写进了测试集说明的第 10 节。

### v1 · 2026-11-05 · 仿真侧提供场景模式、评测专用真值和扫码日志所需的数据

这一版是仿真侧的变化，它们决定了评测能拿到什么数据。

- **GT 导出**（`g1_barcode_sim/ground_truth.py`）：导出每个箱子、每个条码的中心、法向和参考站位，世界坐标系。这份 GT 后来成为评测中"到位误差"和"扫码是否正确"的真值来源。
- **场景模式**：`G1BarcodeEnv` 的 `scene` 默认值改为 `"scenario"`。每次 `reset(scenario=...)` 删除旧场景，按场景 JSON 重建桌子、箱子、标签和背景（`scenario_builder.py`）。同时新增 `task1_gt_obs_pose`、`task2_gt_wrist_target` 两个只供评测使用的字段。
- **`scan()` 的诊断**：仿真的 `scan()` 对枪头相机画面用 pyzbar 解码，并附带 `gt_hit` 诊断（从枪口沿扫描方向的射线与标签求交，给出距离和入射角）。评测侧后来用 `gt.hit` 判断"瞄准了哪个码"。
  - 当时发现 zbar 只能读条纹方向在图像轴约 ±17° 内的一维码，而扫码枪的滚转是自由的。仿真因此对图像按 0、15、…、75° 旋转后再解码，模拟全向扫码器。这一点在 v18 被撤回。
- **`step_hooks`**：每个策略步更新完里程计后调用的回调列表，录像器靠它在长时间行走中逐步抓帧。

### v2 · 2026-11-05 · 评测器初版：分开计算感知误差和执行误差

**初版设计**。`g1agent/eval/evaluator.py` 在 episode 结束后运行，按任务书 4.5 节判定成功。

- 输入有两部分：Harness 记录的估计值（odom 坐标系下的 `root_target`、`wrist_target`、`wrist_target_refined`），以及评测通道给出的真值和实际状态。评测器先把 odom 坐标转到世界坐标，再与真值比较。
- 任务 1 输出 `d_actual`、`yaw_err`、`face_in_view_gt`、`d_err`、`perception_err`（估计目标与真值目标之差）、`execution_err`（实际位姿与估计目标之差）。
- 任务 2 输出 `wrist_dist_to_plane`、`wrist_axis_offset`、`wrist_angle_err`、`head_perception_err`、`wrist_perception_err`。
- `scripts/run_task.py` 跑单个场景，输出 `result.json` 和 `tool_calls.jsonl`。

**第一批结果**（rule 模式，当时仍是仿真原来的宇树行走策略）：

| 场景 | 模式 | 结果 | 关键数据 |
|---|---|---|---|
| t2_L0_0000 | rule | 成功 | 18 次工具调用，106.8 s；手腕到条码平面 0.192 m，偏轴 0.007 m，角度 4.089°，头部感知误差 0.003 m |
| t1_L0_0000 | rule | 成功 | 6 次调用，27.6 s；d_actual 0.618 m（d_obs 0.6），d_err 0.018 m，偏航 1.56°，感知 0.036 m，执行 0.019 m |
| t1_L0_0006 | rule | 失败 | approach 返回 "not within tolerance after the fine moves"；一段超时，最后一步走了 0.232 m 仍未进入容差 |
| t2_L0_0001 | agent_full | 成功 | 17 次调用，126.8 s；0.194 m，偏轴 0.007 m，角度 2.947°，头部感知误差 0.003 m |

### v3 · 2026-11-05 · M1–M3 自检：进程互通、运动学、感知精度

开发按里程碑 M1–M8 推进：M1 进程互通，M2 运动学，M3 感知精度，M4 标记不可见，M5–M8 为三种模式在测试集上的批量评测。在跑批量评测之前，先用自检脚本确认各环节可用：

- **`scripts/check_m1_m2.py`**：
  - 头部和腕部图像都是 480×640×3，腕部相机 K00 = 1087.36；
  - 起始位姿下 pinocchio 正运动学算出的头部、腕部相机位姿与仿真的误差为 0.0 mm / 0.0°；
  - 以 0.3 m/s 行走 1.5 s，里程计前进 0.163 m。原判据要求 `walk_odom_dx > 0.2`，实测达不到；这一项只用来证明"能走起来"，所以放宽到 `> 0.05`。手臂判据保持 `tip_err < 0.05 m`，并注明开环下 PD 会下垂，由 `move_arm` 工具闭环修正。
- **`scripts/check_m3.py`**：把机器人放在 L0 任务 2 场景的 GT 观察位姿上，用头部相机估计目标箱面法向和条码中心，与真值比较。判据：法向误差 < 3°，条码中心误差 < 2 cm。结果 8/8 通过，法向误差 ≤ 0.2°，条码中心误差 ≤ 1.6 cm。例如 t2_L0_0000 法向 0.18°、条码中心 0.0142 m。

### v4 · 2026-11-05 · 标记不可见自检（M4）：从逐像素相等改为噪声基线加阈值

**初版**。`invisibility_check` 在同一物理状态下渲染三组机器人相机画面：标记隐藏（A）、标记移除（B）、标记显示（C，对照组）。要求 A 与 B 的不同像素数为 0。

**问题**。第一次运行判失败：头部相机"隐藏 vs 移除"有 253317 个像素不同。但把 B 再渲染一次作为基线，也有 134212 个像素不同。

**根因**。仿真器的实时渲染在相邻帧之间不是逐位一致的（时间抗锯齿），"有没有像素不同"这个判据测到的是渲染噪声，而不是标记。

**修改**。

1. 差值改为统计每个像素的最大通道差，分别数超过 10 / 30 / 60 / 100 个灰度级的像素数。一组实测数据（头部相机）：
   - 基线：最大 55，超过 60 的像素 0 个；
   - "显示 vs 移除"：最大 109，超过 60 的像素 43 个。
   
   因此阈值取 `STRONG = 60`。
2. 比较顺序改为在同一状态下连续渲染，只切换标记，让每一对被比较的画面有相同的渲染历史。

**效果**。阶段汇报时这项测试共跑 12 次，11 次通过，第一次失败没能复现，原因未查明。

### v5 · 2026-11-05 · `run_suite.py`：按套件和模式批量运行，支持断点续跑

**初版设计**。

- `--suite t1_L1 --mode rule` 跑所有 `t1_L1_<factor>` 套件；也可以一次指定多个套件，并用 `--run_name` 命名。
- 每个场景一行写进 `runs/<run_name>/results.csv`；日志、视频和 `result.json` 放在 `runs/<run_name>/<scenario>/` 下；已完成的场景跳过。
- 依赖的 tabulate 包当时没装，补装后写进了安装脚本。
- 同时写了 `scripts/summarize.py` 的初版（级别 × 模式表、L1 因素曲线、失败原因计数、截断率）。

### v6 · 2026-11-05 · 误差分解指出瓶颈在行走执行，批量评测暂停

**现象**。开 3 个仿真实例做 L1 试跑（rule）：

- `t2_L1_barcode_type_0003` 成功：20 次调用，151.2 s，到平面 0.2 m，偏轴 0.002 m，角度 2.365°；
- `t1_L1_robot_lateral_0000` 失败：d_actual 0.529 m，d_err 0.071 m，**感知误差只有 0.02 m，执行误差 0.08 m**。

**分析**。评测器的误差分解把问题定位到行走执行，不在感知。仿真侧给 `move_relative` 加了 `speed_scale`，让最后几步走慢一点，然后复测：

| 场景 | Harness 判定 | 评测判定 | d_err | 执行误差 |
|---|---|---|---|---|
| t1_L1_robot_lateral_0000 | 失败 | 成功 | 0.032 | 0.034 |
| t1_L0_0003 | 失败 | 失败 | 0.065 | 0.077 |
| t1_L1_robot_lateral_0012 | 失败 | 成功 | 0.049 | 0.070 |

这些场景的感知误差只有 0.002–0.016 m。随后在仿真里直接测 `move_relative`（不经过 agent）：之前报告的 3–4 cm 误差是在命令结束那一刻测的，停稳后实际为 6–14 cm；从 0.4 m/s 把命令清零后底盘还要多走 10–15 cm；后退几乎不起作用。前后试了 5 种补偿办法，都不可靠。

**结论**。按"同一问题试两次不成就停下来问"的规定，M5–M8 批量评测没有开始，等待需求方在三条路中选择：按现状评测、放宽 Harness 到位容差、继续改行走。等待期间关闭了仿真服务器和 VLM 服务，不空占 GPU。

### v7 · 2026-11-06 · 仿真行走控制器换成 GR00T Decoupled WBC（仿真侧）

**背景**。三层对照实验（上游原版程序、上游 DDS 控制链、我们的进程内接入）结果一致，例如 policy.onnx 前进段的净位移分别为 0.129 m 和 0.124 m。这说明过冲、不能后退、低速死区是宇树仿真行走策略本身的特性，不是接入问题。

**修改**。仿真默认行走控制器改为 NVIDIA GR00T Decoupled WBC（官方预训练权重）。同一场景下与原 policy1 的对比：

| 项目 | GR00T | policy1 |
|---|---|---|
| 站立 10 s 漂移 | 1.9 cm / 0.12° | 9.4 cm / 0.96° |
| `move_relative` 8 个目标在容差内 | 7/8，平均 1.9 cm，共 33 s | 2/8，平均 9.6 cm，共 103 s |

agent 侧 `sim_server` 传给环境的 `mode_policies` 改为 null。

**对评测的影响**。新控制器下的 rule 冒烟测试：

- 任务 1（t1_L0_0003）：执行误差 0.003 m、0.032°，d_err 0.002 m，123.5 s。同一场景在旧控制器下一次失败（274.1 s，误差 7.7 cm），一次成功（250.1 s，误差 2.9 cm）。
- 任务 2（t2_L0_0000）：手腕放置误差 0.9 cm（旧控制器 1.2 cm）。

此后所有结果都基于 GR00T，与之前的结果不能直接比较。

### v8 · 2026-11-06 · M5 rule 模式全量批量评测（6 实例并行）

**设计**。规模为任务 1 的 101 个场景加任务 2 的 88 个，覆盖 L0 和全部 L1 共 17 个套件，每个模式 189 个 episode。按旧控制器的速度，单模式串行约需 11 小时。改为在一张共享的 GPU 上开 6 个仿真实例并行，按场景数贪心装箱，各实例分到 [33, 34, 34, 32, 32, 24] 个场景。

**结果**。需求方随后认为 rule 成功率已经足够、要尽快评测 agent，rule 全量提前停止。归档的 `m5_rule_gear_0..5/results.csv` 中：

- 共 138 个 episode，成功 131 个（约 95%）；按任务分，任务 1 为 77/82，任务 2 为 54/56。停止时日志里的统计是"完成 135 个"，与归档文件的 138 不一致，推测是停止过程中又有几个 episode 写进了结果。
- 7 个失败：
  - `t1_L1_box_shape_0007`–`0011`：因素值 ≥ 0.636 的 5 个大箱子全部失败，到位误差 0.36–0.40 m；
  - `t2_L1_box_lateral_0010`：偏移过大，只看到一个箱子；
  - `t2_L1_barcode_pos_0008`：条码在角落，手够不到，偏轴 6.3 cm。
- 没跑到的套件：t1_L0、t1_L1_lighting、t2_L1_barcode_size、t2_L1_box_shape。
- 6 个实例并发时每个 episode 约 3–5 分钟，比单实例的约 2 分钟慢，原因是 6 个实例共享 CPU（物理仿真跑在 CPU 上）。

### v9 · 2026-11-06 · agent 模式只跑每个套件的 2 个代表场景

**问题**。agent 模式如果也跑全量，两种模式共 378 个 episode，预计 5–6 小时。需求方认为太多。episode 多的原因不是重复次数（每个场景本来只跑 1 次），而是场景种类多：17 个套件，每个 L1 因素扫 10–13 个取值。

**修改**。

- rule 保持全量；agent_assist 和 agent_full 每个套件只选 2 个场景，共 17 × 2 = 34 个，两种模式合计 68 个，预计 1–1.5 小时。
- 挑选规则由我们自定并已报告：rule 在该套件有失败时，选 1 个失败场景加上因素值最近的成功场景（例如 box_shape 选 0007 和 0006）；全部成功时选因素范围两端。
- 实现：
  - 新增 `scripts/select_agent_subset.py`（单场景套件用 `dict.fromkeys` 去重）；
  - `run_suite.py` 增加 `--scenarios`，传入不存在的 id 直接报错；
  - 新增 `scripts/build_agent_clients.py`，按负载把子集分到 N 个实例。
- rule 全量停掉之后，`select_agent_subset.py` 改为覆盖全部 17 个套件，rule 没跑到的套件取因素范围两端；`build_agent_clients.py` 对选中但没有 rule 结果的场景追加一次 rule 运行（共 8 个），排在 agent 两种模式之后。

**结果**。agent_assist 批量在暂停前完成的 8 个 episode 全部成功（均为任务 1，187.2–458.2 s）。agent_full 批量没有留下结果，因为之后的重点转向了任务 3。

### v10 · 2026-11-06 · 评测器判成功，但 agent 实际越界：只看指标会误判

**现象**。L0 演示中，agent_full 任务 1（t1_L0_0000）被评测器判成功（d_err 2.4 cm、偏航 0.95°，执行误差 4.8 cm），382 s，LLM 19 次。逐步查看工具调用日志后发现：approach 失败后，LLM 越界执行了任务 2 的步骤（找条码、抬手臂、放手腕），最后自己报告失败。视频里手臂是抬着的。

**根因**。任务 1 的评测只看底盘最终是否在观察位姿附近，不检查过程。

**处理**。

- 更正了已经发出的结论。其余三个演示 episode 逐一核实为真成功。其中 agent_assist 任务 1 第一次 approach 差 3.7 cm，没过到位检查，LLM 选择前进 0.3 m 后重新 approach 并成功。
- 形成规则：评测结论不能只看评测器的单一指标，要结合工具调用日志核对。这直接推动了 v11 的完整时间线。
- 同时写了"跑完当前 episode 就暂停"的脚本：记下每个客户端已完成的 episode 数，日志里多出一条完成记录就停掉该客户端。下一个 episode 刚开始、还没有结果文件，恢复时会重跑。

### v11 · 2026-11-06 · 完整的 agent 时间线日志

**需求**。输出完整的 agent 运行日志，包括 agent 在什么时候思考了什么、输出了什么。

**修改**。

- `harness/logger.py` 的 `EpisodeLogger(ep_dir, on_event)` 给每条记录加上 `t_unix` 和所有文件统一的写入序号 `log_seq`。
  - `tool()`、`llm()` 把事件转发出去，工具事件附带 `tool_overlay`，只从工具结果里实际存在的字段构建；
  - 新增 `stage()`，写入 `stages.jsonl`。
- `run_task` 的 `on_event` 把阶段事件转给评测端口的 `stage`，其他事件转给 `annotate`；出错时 `_eval_rpc` 直接抛出，不静默忽略。
- 新增 `scripts/make_timeline.py`，内容见"当前实现"第 10.2 节。时间线表头单独统计没有返回任何文本的 LLM 调用次数。旧日志也能处理。
- 在 `demo_agent_gear/agent_full/t2_L0_0000` 的副本上做了测试。

### v12 · 2026-11-06 · 任务 3 评测器初版、`--task` 覆盖参数、新的结果列

**初版设计**。

- `evaluator.py` 拆成 `_task1`、`wrist_metrics`、`_task2`，公式不变。
- 新增 `scanner_hfov_gt()`（32.8°，从仿真 `scanner.yaml` 的焦距和光圈读出）和 `gt_standoff()`，与算法侧公式相同，但代入真值标签尺寸。这时用的是标签整体宽度（含白底），`fill_frac = 0.85`，让名义 0.10 m 标签落在 0.20 m。
- `_task3`：
  - 任务 1 指标用 task1_end 阶段保存的实际状态和 task3 的容差计算；
  - 期望集合是所有非干扰项的 GT 条码，按 task1_end 时刻根坐标系下的列规则排序；
  - 对每条扫码记录计算 content、是否瞄准、是否新码、枪口相对真值自适应目标的误差；
  - 汇总 `scan_recall`、`wrong_contents`、`n_duplicate_reads`、`order_ok`、`tip_err_mean/max`、`n_placement_ok`、`ledger_matches_log` 等；
  - **成功条件**：任务 1 阶段达标，且每个 GT 条码都读到、顺序完全正确、没有读到外来内容。
- `evaluate` 遇到未知任务时改为抛错。之前未知任务会默认按任务 2 处理。
- 新增 `g1agent/eval/scan_order.py`：评测侧的列规则副本（算法代码不能 import eval 包，反过来评测也不 import 算法），用随机测试保证两份结果一致。
- `run_task.py`、`run_suite.py`、`record_episodes.py` 增加 `--task` 覆盖参数（例如 `--task 3` 把 t1/t2 场景当任务 3 跑）。`result.json` 增加 `scenario_task`、`stages`、`scan_log`；`results.csv` 增加 `t1_success`、`scan_recall`、`order_ok` 等列。
- `tests/test_task3_eval.py` 初版 12 个测试。

**仿真侧配套变化**：

- 仿真环境新增两台只出 RGB 的固定录像相机（1280×720 和 640×360），以及 `frame_state()`、`camera_rgb(name)`；
- `env.scan()` 的每条解码结果增加 `polygon` 字段，即 pyzbar 多边形映射回原始腕部图像坐标；
- `sim_server` 的 `actual_state()` 和 `T_world_odom()` 改用 `frame_state()`，数值不变，但不再扰动深度噪声随机流，评测读取实际状态不会改变机器人后续看到的深度噪声。

### v13 · 2026-11-06 · 评测审查修复 3 个 bug

对 v12 做代码审查，发现并修复了三个问题：

1. **任务 1 阶段比较了不同时刻的量**。`_task3` 用 task1_end 时刻的真值状态，去和 episode 结束时的 `ws["root_target"]` 比较。但任务 3 里扫码阶段可能再次调用 `compute_obs_pose`，把它覆盖掉。
   - 修复：`run_task` 的 `on_event` 在 `task1_end` 事件触发时深拷贝 `tb.ws.root_target`，存为 `task1_end_root_target`；评测器改用这份快照。
2. **失败原因被覆盖**。`row()` 把 metrics 中所有属于 `COLS` 的键都拷进 `results.csv`。失败 episode 的 metrics 现在总带 `failure` 字段，覆盖了原来的优先级（编排异常优先，其次 Harness 摘要）。
   - 修复：只拷贝行里还没有的键。
3. **可能复用别的任务的结果**。`record_episodes.py` 在没给 `--task` 时不检查已有 episode 属于哪个任务，可能静默复用任务 3 的结果。
   - 修复：按 `scenario_task` 检查。

同时新增 `tests/test_run_task_plumbing.py`，用假对象跑真实的 `run_task.run()` 和 Harness 完整流程。

### v14 · 2026-11-06 · 任务 3 成功判据不再看任务 1 阶段

**现象**。任务 3 首轮在 `t1_L0_0000` 上跑 rule 模式。Harness 报告 2/2 按顺序扫完：bc0 的内容 `Y4F46A2PK` 在仿真时间 44.92 s 读到，bc1 的内容 `A396DJJEY` 在 72.82 s 读到。但评测器的 `success` 是 false。

**根因**。任务 1 阶段停在离目标差 6.9 cm 处，超过 5 cm 容差。这是因为旧参数 `final_bias_m = 0.05`（停在目标前 5 cm，原本是为旧行走策略设的）。而 v12 的判据把任务 1 阶段失败也算作任务 3 失败。可是任务 3 的任务 1 阶段按设计会"快速结束"，扫码阶段会为每个条码自己规划站位，任务 1 阶段的到位精度不影响扫码结果。

**修改**（2026-11-06 的决定）。删除 `if not out["task1"]["success"]: fails.append(...)`。成功只由"全部期望码读到、顺序正确、无外来内容"决定，任务 1 阶段结果单独报告为 `task1` / `t1_success`。`tests/test_task3_eval.py` 对应的旧断言（failure 里含 task1_end）改为断言 failure 含 "1/2" 且不含 "task1_end"。改后测试 62 项通过。

**首轮结果**（`t1_L0_0000`，2 个码）：

| 模式 | 扫码 | 任务 1 阶段 | 枪口误差 | 用时 | LLM |
|---|---|---|---|---|---|
| rule | 2/2，顺序对 | 差 6.9 cm（未过） | 3.9 cm | 542 s | 0 |
| agent_assist | 2/2，顺序对 | 通过 | 3.6 cm | 553 s | 0 |
| agent_full（修复前） | 0/2 | — | — | 255 s | 12 |
| agent_full（修复后） | 2/2，顺序对 | 通过 | 3.9 cm | 565 s | 28，截断 0 |

遗留问题：所有模式的枪口都偏远约 4 cm（见 v16）；只跑了一个场景，而现有测试场景每个箱子只有一个码，"从上到下"的顺序没有测到。

### v15 · 2026-11-07 · 重录与耗时分解：瓶颈在仿真速度

**重录**。去掉 `final_bias_m` 之后，在 `t1_L0_0000` 上重录任务 3（`task3_v3`）：

| 模式 | 结果 | 任务 1 阶段 | 枪口误差 | 用时 | LLM |
|---|---|---|---|---|---|
| rule | 成功 | 通过 | 4.0 cm | 351.7 s | 0 |
| agent_assist | 成功 | 通过 | 4.0 cm | 363.9 s | 0 |
| agent_full | 成功 | 通过 | 4.0 cm | 314.0 s | 28，截断 0 |

**发现的定义问题**。视频 METRICS 面板中冻结的根部横向差约 0.29 m。这不是到位误差，而是两个"观察位姿"的定义不同：系统的观察位姿按所有箱子的中间算（任务 3 观察全部箱子），GT 观察位姿按场景指定的单个目标箱面算。到位判据不看横向，所以不影响判定。是否修改 GT 定义留待需求方决定。

**耗时分解**（基于三个 run 的工具调用和帧时间戳）：

| | rule | agent_assist | agent_full |
|---|---|---|---|
| 墙钟 | 352 s | 364 s | 314 s |
| 仿真时间 | 72 s | 81 s | 57 s |
| 仿真速度（倍实时） | 0.21 | 0.22 | 0.18 |
| LLM 总耗时 | 0 | 0 | 38 s（28 次，平均 1.4 s） |
| 手臂运动 | 154 s | 147 s | 151 s |
| 走路 | 124 s | 151 s | 111 s |

结论：仿真本身约占墙钟的 85%（物理在 CPU 上，3 个实例并行，每步渲染 4 台相机）。所以墙钟时间主要反映仿真速度，不能直接当作真机耗时。按仿真时间看，38 s 的 LLM 对 57 s 的动作约占 40%。

### v16 · 2026-11-07 · GT stand-off 改按印刷符号尺寸（仿真导出 `pattern_size`）

**现象**。所有模式的枪口都比 GT 目标点偏远约 4 cm。

**根因**。头部相机把标签宽度估大了约 20%（估成 12 cm，实际 10 cm），自适应距离因此算成约 0.24 m。更根本的问题是 stand-off 的定义：扫码枪要装下并分辨的是印刷的符号，不是白色底板；而且有没有底板，符号都在。

**修改**。

- 仿真 `ground_truth.py` 的 `build_gt` 为每个条码额外导出 `pattern_size`，只含条码图案。例如 t1_L0_0000：`size` 为 [0.1, 0.05]，`pattern_size` 为 [0.08, 0.03]。
- 评测侧 `gt_standoff` 改用 `gt_symbol_size(bc)`。缺字段时抛出 `ValueError`，不回退到 `size`。
- `fill_frac` 从 0.85 改为 0.68，让名义 0.08 m 符号仍落在已验证的 0.20 m。
- `sim_server.gt_standoff_fn` 和视频中绿色目标点同步使用这个定义。
- 算法侧同时改为：头部阶段用检测框估计的宽度（上界，偏保守），到预备位后换成腕部解码轮廓测得的宽度。

**效果**（rule 模式任务 3）：

| 场景 | 符号宽（真值） | 腕部实测 | GT 距离 | 枪口误差 均值 / 最大 | 结果 |
|---|---|---|---|---|---|
| t1_L0_0000 | 8 / 8 cm | 8.0–8.3 cm | 0.20 / 0.20 m | 0.7 / 1.0 cm | 2/2，453 s |
| t2_L1_barcode_size_0001 | 8 / 5 cm | 8.1 / 5.1 cm | 0.20 / 0.12 m | 0.4 / 0.5 cm | 2/2，356 s |
| t2_L1_barcode_size_0009 | 8 / 12 cm | 8.1–8.3 / 12.1–12.2 cm | 0.20 / 0.30 m | 0.5 / 0.8 cm | 2/2，414 s |

修正前 t1_L0_0000 的枪口误差为 4.0 cm。

### v17 · 2026-11-07 · 三箱落地堆叠场景 stack3：评测和标记不再假设单一目标

**问题**。有三处代码默认场景只有一个目标箱：

1. 现有场景格式总是添加桌子，并把所有箱子放在桌面同一高度，没有"放在另一个箱子上"或落地的选项。
2. GT 的 `target_box_id`、`target_face`、`task1_gt_obs_pose`、`task2_gt_wrist_target` 都只针对单个目标箱计算。
3. agent 侧 `sim_server` 第 87 行无条件读取 `task2_gt_wrist_target` 来画标记。

**修改**（仿真侧和 agent 侧）。

- `testsuite/scenario.py` 的 `to_layout` 和 `geometry.py` 支持 `table` 为 None 的场景，箱子可以落地和竖直堆叠（`on` 字段）；
- 任务可以以所有箱子为目标；
- `sim_server` 的 GT 标记在缺少单目标字段时不再崩溃。

**新场景**。`stack3_L0_0000`：无桌，机器人起始距离 2.5 m，三个 0.40×0.40×0.38 m 箱子，码中心高 0.19 / 0.57 / 1.10 m。此前单独测过 GR00T 的下蹲能力：高度指令 0.25 m 时骨盆实测 0.25 m，不摔。

**各轮结果**：

| 轮次 | 场景 | 模式 | 结果 | 说明 |
|---|---|---|---|---|
| 1 | 0000 | rule | 1/3 | 三个箱子被检测成 1 个，plan_scans 只找到顶层码 |
| 2 | 0000 | rule | 1/3 | 找到 2/3，底层 0.19 m 漏检；中层首次在完整任务中下蹲，但预备位 IK 失败 |
| 3 | 0000 | rule | 2/3，702 s | 中层蹲到高度指令 0.60（骨盆 0.562 m）扫到；底层蹲到最低档 0.25 后 IK 差 5 mm / 3.3°，站起重规划后同样失败 |
| — | — | — | — | 场景调整：0000 的底层码贴到 0.30 m；新增 0001，保留 0.19 m，专门测"够不到就报告" |
| 试 1 | 0000 | agent_full | 0/3 | LLM 自己给 `compute_wrist_target` 传了 offset 0.142，并把下蹲和行走的顺序弄反 |
| 4 | 0000 / 0001 / 0000 | rule / rule / agent_full | 1/3 / 2/3 + 报告不可达 / 1/3 | 0001 的底层码判 too low（z = 0.19 m），报告链路按预期工作 |
| 5 | 0000 / 0001 / 0000 | rule / rule / agent_full | 3/3（642 s）/ 2/3 + 不可达（451 s）/ 2/3（551 s，LLM 42 次，截断 0） | agent_full 中层码第一次没读出后没有修正重扫，直接跳到下一个码 |
| 6 | 0000 / 0001 / 0000 | rule / rule / agent_full | 3/3（509 s）/ 2/3 + 不可达（496 s）/ 3/3（454 s，LLM 41 次，截断 0） | 8 次扫码全部第一次读到；枪口误差从上轮的 1.6–2.2 cm 降到 0.6–0.8 cm |

评测侧的意义有两点：

- `reported_unreachable` 字段让我们能区分"漏扫"和"正确地报告够不到"。stack3_L0_0001 在评测上仍是 2/3、判失败，因为成功判据要求读到全部 GT 条码，但结果中可以看到不可达报告是正确的。
- 场景的码位按机器人能力调整，并单独保留一个不可达场景，而不是放宽判据。

### v18 · 2026-11-07 · 仿真扫码器只解码原始画面（结果从此不能与之前直接比较）

**修改**（仿真侧，需求方 2026-11-07 的决定）。`g1_barcode_sim/configs/scanner.yaml` 中 `rotation_step_deg` 从 15 改为 90，解码循环只跑 0°，即只解码采集到的原始图像。

**理由**。真实扫码器读不出与画面明显倾斜的码，之前的 0–75° 旋转解码比真实情况宽松。改后 roll 没对准就读不出来。

**影响**。此后的 `n_no_read`、`scan_recall` 与之前的不能直接比较，相关报告都已注明。算法侧相应加了 roll 约束，让扫码枪的横滚与条码对齐。

### v19 · 2026-11-07 · 3×3 整面场景 stack9 / stack9p：列规则的真值要与直觉一致

**场景设计**。`stack9_L0_0000`：落地 3 列 × 3 层共 9 个箱子，正面贴 5 个码，每层至少一个，左右有偏移。

**问题**。第一次用 `order_labels_in_frame` 算 GT 顺序时，左下码排在左上码前面（B5L0RTY4D 在 T5L2QWX8A 之前）。

**根因**。左右偏移让这两个码的横向范围没有重叠，按列规则不在同一列分组里。列规则本身是对的（按横向范围重叠分列），问题在场景的码位。

**修改**。调整场景而不是规则：L0 的横向位置 u 改为 0.0，R0 改为 −0.04。GT 顺序变为左上 → 左下 → 中 → 右上 → 右下，即"左列从上到下、再中列、再右列"。

**各轮结果**：

| 轮次 | 场景 | 模式 | 结果 | 用时 | LLM（次 / 截断） | 说明 |
|---|---|---|---|---|---|---|
| 1 | stack9 | rule | 4/5，失败 | 1016 s | — | 左下码因码宽测量为 0 而失败 |
| 1 | stack9 | agent_full | 5 个内容都读到，`order_ok` False，dup 1 | 1108 s | 79 / 0 | 顺序为左上 → 中 → 左下 …；评测器准确指出是顺序错，不是漏扫 |
| 2（v18 之后） | stack9 | rule | 5/5，顺序对 | 1177 s | — | scan 内部重试首次真正生效 |
| 2 | stack9 | agent_full | 4/5 | 1246 s | 82 / 0 | 把 `move_arm` 的 partial 当成失败，放弃了右下码 |
| 3 | stack9 | agent_full | 5/5，顺序对 | 912 s | 69 / 0 | partial 提示改为写明"这不是失败，请继续" |
| 规划提速后 | stack9 | rule / agent_full | 5/5 / 5/5 | 1056 s / 869 s | — / 70 / 0 | 站位规划合计 rule 205 → 107 s、agent_full 201 → 100 s，整轮只快 5–10%，大部分时间在仿真动作上 |
| 8 | stack9p | agent_full | 5/5，顺序对 | 910 s | 63 / 0 | 印刷外观；扫码 8 次，读不出 3 次，都由 scan 工具内部重试读出 |

### v20 · 2026-11-07 · 多面评测：按面排序、环绕判定、`stage_log`

**需求**。四面箱堆要求按 前 → 右 → 后 → 左 绕行，每面内仍按列规则。单面的全局排序不再适用：从机器人起点的坐标系看，背面的"左右"与从背面正前方看是相反的。

**仿真侧**。

- `testsuite/geometry.py` 新增 `TOUR_FACES = ("-x", "-y", "+x", "+y")`、`box_face`、`blockers`、`_fully_covered`、`exposed_faces`、`stack_faces`。部分被遮挡的面也算暴露面。
- `get_ground_truth` 支持 target `"all"`，并输出 `faces` 列表。
- `target_face` 和 `task1_gt_obs_pose` 改用正面暴露面的并集。
- 新场景 `stack4f_L0_0000`：2×2×3 共 12 箱，12 个码，每面 3 个。仿真侧测试 20/20 通过。

**评测侧**。

- `scan_order.py` 新增 `face_frame`、`assign_faces`、`labelled_faces`；`evaluator.py` 新增 `_tour_metrics`。
- 多面的成功判据重新定义：
  - 全部读到；
  - 每面内部顺序正确；
  - 首个面是正面（`front_first`）；
  - 各面连续（`faces_contiguous`）；
  - 无外来内容。
  
  全局 `order_ok` 和 `face_order_ok` 只报告。结果保留 `reported_unreachable`。
- `sim_server` 的 `Episode.stage` 把每次调用追加到 `stage_log`，并新增评测端口的 `get_stage_log`。**行为变化**：`stages` 字典由"最后一次覆盖"改为 `setdefault` 保留第一次出现。原因是多面场景里 `face_start`/`face_end` 每个面都会触发，而 `task1_end` 的语义是第一次。
- `run_task.py` 保存 `stage_log`。测试共 93 项通过。

**各轮结果**（agent_full，stack4f_L0_0000）：

| 轮次 | 结果 | 说明 |
|---|---|---|
| 首次 | 正面 2/3 后卡住 | 正面顺序符合列规则；左下码报不可达；换到右侧面时 "the right side was not fitted from here"，之后进入死循环 |
| — | — | 场景调整：在扫码枪必须横滚对齐的约束下，0.30 m 的码实测够不到。按需求方"码位符合机器人最低限"的要求，4 个底层码从 0.30–0.32 m 统一抬到 0.33 m |
| 10 | 9/12，失败 | 前三面各 3/3，各面顺序都对；左侧面 3 个码全被误判为够不到（其他三面同高度的码都扫到了，说明是误判） |
| 12 | 12/12，成功 | 顺序与真值完全一致，`front_first`、`faces_contiguous` 都为真；12 次扫码 0 次读不出；12 次放置全部合格；枪口误差平均 1.1 cm、最大 3.5 cm；误报不可达 0；1656 s；LLM 143 次，截断 0，最长提示 11069 token |

第 12 轮的时间分配：第一阶段 95 s，每面约 290–310 s，每次换面约 120 s（共 3 次）。这只是一个场景跑一次的结果，成功率是否稳定还没有验证。

### v21 · 2026-11-07 · 默认只评测 agent_full，rule 仅用于诊断

**此前的做法**。每轮同时跑 rule 和 agent_full。两者用同一套工具，区别只在下一步由谁决定，所以 rule 是对照组：agent_full 失败时，看 rule 在同一场景能否成功，就能区分是工具问题（手臂、规划、检测）还是 LLM 决策问题。前面几轮修的很多问题都在工具层。

**修改**（需求方的决定）。工具已经比较稳，最终交付的是 agent_full，所以之后默认只跑 agent_full，只有需要区分工具问题和 LLM 问题时才单独跑一次 rule。

### v22 · 2026-11-07 · 起始朝向不平行：stack4f_L0_0001 / 0002

**场景**。以 `stack4f_L0_0000` 为基础，只改 `robot_init`（`factors_varied = ["robot_init"]`）：

- 0001：在中线右侧 0.5 m，左转 30°；
- 0002：在中线左侧 0.7 m（已超出 +y 侧面的 0.40 m），右转 35°。

离线核算了正面各角点的方位角：原场景为 −9.1°～9.1°，0001 为 −27.7°～−10.2°，正面仍在视野内。

**结果**（agent_full）：

| 场景 | 结果 | 用时 | LLM（次 / 截断） | 漏扫原因 |
|---|---|---|---|---|
| 0001 | 11/12，顺序正确，`front_first` / `faces_contiguous` 为真 | 2494 s | 162 / 0 | bc4：放置检查返回 fail |
| 0002 | 11/12，顺序正确 | 2368 s | 152 / 0 | bc10：手臂偏 21 cm 时仍报 partial |

两个场景都正确认出 −x 面为正面，第一阶段约 120 s 后开始扫码。0001 的第一阶段：`t1_success = True`，感知误差 0.048 m，执行误差 0.025 m；scan 调用 13 次，no_read 2 次，`scan_recall` 0.917，没有读错，也没有重复。

**注意**。两次 LLM 都在 finish 中声称全部成功（`claimed_success = true`），评测结果是 11/12。这说明 Harness 自述与评测之间的差异必须单独看，不能用自述代替评测。两个漏扫原因都在工具层修复，修复在下一轮生效。

### v23 · 2026-11-07 · 快递纸箱外观 cardbox 与 stack4c / stack9c 场景（仿真侧）

**修改**。

- `scenario_builder.py` 新增外观 `"cardbox"`：引用 Isaac 自带的快递纸箱模型（`factors.yaml` 的 `appearance.cardbox_usd`），关闭模型物理，按局部包围盒缩放并居中到箱子尺寸；包围盒为空时抛出 `RuntimeError`。
- `textures.py` 给 cardbox 加了牛皮纸背景色 (0.66, 0.50, 0.32)，只在标签没有白底时使用。
- `testsuite/scenario.py` 的 `to_layout` 允许同一箱面贴多张标签，但按旋转后的轴对齐范围检查不能重叠，重叠就抛出 `ValueError`。仿真侧测试从 15 项增加到 17 项，全部通过。
- 新场景：`stack4c_L0_0000`（12 码 / 4 面）、`0001`（6 码，只在 ±x 两面）、`0002`（60 码，两面）、`stack9c_L0_0000`（5 码，−x 面）。

**场景自检**。为了确认真码读不出是不是渲染本身的问题，写了近拍检查：对每个 GT 标签，把辅助相机放在沿法向 0.25 m 处拍照，再用 pyzbar 解码。

| 场景 | 可解码 |
|---|---|
| stack4f_L0_0000 | 12/12 |
| stack4c_L0_0001 | 6/6 |
| stack4c_L0_0002 | 60/60 |
| stack9c_L0_0000 | 5/5 |
| stack4c_L0_0000 | 6/12（失败的都在 ±y 侧面） |

对 stack4c_L0_0000 的 RB0 −y 标签单独近拍：0.25 m 和 0.6 m 能解出，1.2 m 解不出。这项排查没有形成结论。因此后续评测选用的是全部可解的 `stack4c_L0_0001`。

### v24 · 2026-11-07 · 纸箱两面场景 stack4c_L0_0001：从 4/6 到 6/6，评测数据如何指导修改

纸箱模型自带的面单贴纸印着条码图案，但读不出内容。从头部相机看，它们和真码很像，机器人只能走近核验。这一组 7 轮运行（agent_full）的改动大多在工具层（扫描计划、unreadable 判定、补看策略），本节只记录评测侧数据怎样暴露问题、怎样验证修改。

**第 1 轮（af4c_2side）**：真码 4/6，漏了正面底层和背面中层；正面先扫、各面连续；读错和重复都是 0；4055 s；LLM 306 次，截断 0。

- 按面统计：正面计划了 7 个目标，3 个真码只扫到 2 个；右侧面没有码，7 处贴纸被当成标签，全部失败，但没有误读出内容；
- 逐码统计：计划中 23 个没有扫到的目标（大多是贴纸，也包括计划中漏扫的真码）平均每个花 117.6 s 的工具时间（move_arm 46.2 s、plan_stance 23.0 s、approach 21.2 s、scan 16.4 s、refine_with_wrist 10.4 s）；
- 把扫到的码和真值对照，得到里程计漂移：正面约 (−0.035, −0.078) m，背面约 (0.06–0.10, −0.225) m。

**用历史数据检验假设**。"手腕近看时解不出码"能否作为"这里不是真码"的证据？我们汇总了归档中所有 run 和第 1 轮的工具调用记录，统计每次对一个码的尝试在 refine_with_wrist 时有没有解出、这个码最终有没有扫到（表中单位为"次尝试"）：

| | refine 时已解码 | refine 时未解码 | partial |
|---|---|---|---|
| 最终扫到的真码 | 125 | 33 | 2 |
| 始终没扫到的码 | 8 | 34 | 1 |

结论有两条：

- refine 时解不出，不能证明没有真码（对最终扫到的真码，有 33 次尝试在 refine 时也没解出）。所以"无可读码（unreadable）"的判定必须收紧；
- refine 时解出过的码仍可能最终漏扫（8 次）。所以要在曾读出的位置补扫。

**中间轮次**：

| 轮次 | 正面结果 | 暴露的问题 |
|---|---|---|
| v2 | 3 个真码都进了计划；底层和中层扫到 | 顶层真码 F4ATLFRVG 被误判为 unreadable |
| v3 | 顶层、底层扫到，1333 s 结束 | 中层真码 F4AMRF38K 的最终位姿 IK 不可达，枪口停在离面 0.35 m 处（目标距离 0.274 m，偏轴 3.9 cm）就扫码并被判 unreadable，还没到扫码距离就下了"无码"结论 |
| v5 | 3/3，1021 s（第 1 轮正面 1189 s 只有 2/3） | 右侧面 4 处大张面单（约 22 cm 宽，扫码距离上的视野只有约 12 cm）永远无法整张进入视野，只能记为 failed，这是符合设计的保守判定 |
| v6 | 3/3，其余 4 处全部正确判为 unreadable，没有 failed | 这一轮没有跑完全部四面（归档目录名标注为上下文溢出） |

**第 7 轮（af4c_2side_v7）**：

- 真码 6/6，`scan_recall` 1.0；`order_ok`、`faces_contiguous`、`front_first` 都为真；读错、重复、未对准读都是 0；
- 枪口误差平均 1.0 cm、最大 1.8 cm；
- 6778 s；LLM 459 次调用 / 460 次尝试，长度截断 1 次（0.2%，已自动重新采样），超时 0；
- Harness 自报失败，因为 9 处 failed 和 6 处 unreachable 的贴纸无法核验，而评测按 GT 判成功。

召回从 4/6 提升到 6/6，代价是用时增加，主要花在逐个走近核验 32 处贴纸上（38 个计划目标减去 6 个真码），每处约 2 分钟。

## 已知限制

- **样本量小**。任务 3 的每个场景通常只跑 1 次，所有任务 3 的成功或失败都是单次结果，没有成功率估计。M5 的批量评测只完成了 rule 的 138 个 episode 和 agent_assist 的 8 个，agent_full 没有批量结果。之后的重点转到了任务 3，任务 1、2 的 agent 批量评测没有补跑。
- **`summarize.py` 不支持任务 3**。`err_cols` 只有任务 1、2 的键，输入中有任务 3 的行时抛出 `KeyError`。`make_highlight.py` 对任务 3 会走"其他任务"分支，执行误差和腕部感知误差显示为 nan。
- **评测只看结果，不看过程**。任务 1 的评测只看最终位姿，曾把越界执行任务 2 步骤的 agent_full 运行判为成功（v10）。任务 3 的 `success` 不惩罚重复读码、no_read、放置不合格和多余的核验时间。判断一次运行"做得对不对"，仍需要看时间线。
- **Harness 自述与评测不一致**。自述可能偏乐观（v22 中 LLM 声称全部成功，评测为 11/12），也可能偏保守（v24 中 Harness 自报失败，评测为 6/6）。报告里两者都要列出。
- **任务 1 阶段的观察位姿定义不一致**。在 `table_two_boxes` 场景上跑任务 3 时，系统按所有箱子居中计算观察位姿，GT 按场景指定的单个目标面计算，冻结的根部横向差约 0.29 m。到位判据不看横向，所以不影响判定。是否改 GT 定义还没有决定。
- **部分容差没有记录依据**。任务 1、2 的评测容差来自任务书，配置注释没有另给依据；`max_m = 0.30` 没有依据。
- **任务 3 场景全是手写的**。场景生成器还不支持地面箱堆，任务 3 套件的 `code_hash` 为 `null`。码的高度按机器人能力调整过（stack3 底层 0.19 → 0.30 m，stack4f 底层 0.30–0.32 → 0.33 m），所以这些场景反映的是"在机器人可达范围内能否扫完"，不是任意贴码位置。
- **stack4c_L0_0000 的侧面标签**。近拍检查只有 6/12 能解码，失败的都在 ±y 侧面，原因没有查明，这个场景目前不适合用来衡量召回。
- **扫码判定依赖渲染画面**。仿真的 `scan()` 是对枪头相机渲染图像做 pyzbar 解码，不模拟真机的解码距离和角度限制。v18 之后只解码原始画面，比之前严格，但与真实扫码器的差异没有量化。
- **墙钟时间受仿真速度支配**。仿真约 0.2 倍实时，墙钟时间的约 85% 花在仿真上（v15 的分解）。并行实例共享 CPU，单个 episode 的用时还取决于同时运行的实例数，不能直接当作真机耗时比较。
- **自碰撞未仿真**。机器人模型关闭了自碰撞，下蹲时手臂可以穿过大腿和膝盖（仿真侧记录）。评测不会因此判失败；真机上同样的动作是否可行没有验证。
- **标记不可见测试需要运行中的仿真服务器**，连不上时跳过。阶段汇报时 12 次运行中有 1 次失败没能复现，原因未查明。
