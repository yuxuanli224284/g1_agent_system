# 单元测试

## 作用与在系统中的位置

`tests/` 目录下的测试检查系统各部分在不启动仿真器、不加载检测模型、不调用真实 LLM 的情况下是否按设计工作。测试的对象覆盖几何与感知、工具箱与 harness、编排器、LLM 客户端、评测器、视频合成、仿真服务器的 episode 胶水层，以及 `scripts/run_task.py` 的数据流。除 `test_markers_invisible.py` 外都是 CPU 测试，靠假环境、假 RPC 端口、脚本化的工具箱和本地 mock LLM 服务器代替真实组件。

测试还承担两项"守规矩"的职责：

- **真值隔离**：`test_no_gt_leak.py` 用静态扫描保证算法代码拿不到真值；`test_task3_logic.py` 保证评测侧自带的顺序规则与算法侧的实现给出相同结果。
- **项目的设计原则**：mock LLM 服务器对每个请求断言不含输出长度上限；截断或超时的回复必须重新抽样，不能被解析。

与其它组件的对应关系：

| 被测组件 | 主要测试文件 | 测试方式 |
|---|---|---|
| 感知几何 `g1agent/perception/geometry.py`、`plane_fit.py` | `test_geometry.py` | 合成射线、合成深度图 |
| 腕部符号宽度 `Toolbox._wrist_symbol_width` | `test_wrist_width.py` | 合成腕部相机帧（印有 QR 码） |
| 工具箱、harness、守卫、rule/agent 编排器（任务 3） | `test_task3_logic.py`、`test_task3_review.py`、`test_stance_pool.py` | 脚本化工具箱（`ScriptedTB` 等）、假机器人 |
| LLM 客户端 `g1agent/llm/client.py`、`agent_full` | `test_llm_client.py` | 本地 OpenAI 兼容 mock 服务器 |
| 评测器 `g1agent/eval/evaluator.py`、`scan_order.py` | `test_task3_eval.py` | 手写的 GT、扫码日志 |
| 视频合成 `g1agent/viz/compositor.py` | `test_compositor.py` | 合成场景，CPU 上真实编码 H.264 |
| 仿真服务器胶水层 `g1agent/robot/sim_server.py` 的 `Episode` | `test_sim_server_episode.py` | 不含仿真器的替身环境 |
| `scripts/run_task.py` 中 `make_episode_runner()` 返回的 `run()` | `test_run_task_plumbing.py` | 假机器人、假评测端口、假检测器 |
| 真值隔离 | `test_no_gt_leak.py` | AST 静态扫描 |
| USD 标记对机器人相机不可见 | `test_markers_invisible.py` | 集成测试，连接运行中的仿真服务器 |

## 当前实现

### 运行方式与规模

- `tests/conftest.py` 只做一件事：把仓库根目录加入 `sys.path`，测试因此可以直接 `import g1agent...`。
- 在 agent 侧的 Python 环境里运行 `python -m pytest -q tests`。当前共收集 121 个测试（按 pytest 计数，参数化的每组算一个）：

| 文件 | 测试数 |
|---|---|
| `test_task3_logic.py` | 52 |
| `test_task3_eval.py` | 21 |
| `test_compositor.py` | 18 |
| `test_task3_review.py` | 6 |
| `test_geometry.py` | 5 |
| `test_llm_client.py` | 5 |
| `test_wrist_width.py` | 5 |
| `test_sim_server_episode.py` | 3 |
| `test_no_gt_leak.py` | 2 |
| `test_run_task_plumbing.py` | 2 |
| `test_stance_pool.py` | 1 |
| `test_markers_invisible.py` | 1 |

- 最近一次全量运行（2026-11-07 23:42）的结果是 120 passed、1 skipped，用时 618.14 s。被跳过的是 `test_markers_invisible.py`（当时没有仿真服务器）。不含这个测试时，全套只需十几秒（2026-11-07 白天的多次运行为 7–18 s）；全量运行的 600 多秒几乎都花在这个测试等待超时上，原因见演进记录 v3。
- 日常开发时用 `--ignore=tests/test_markers_invisible.py` 排除这个集成测试。
- 需要 ffmpeg 的测试（`test_compositor.py` 中的 H.264 测试、`test_sim_server_episode.py` 的两个录像测试）用 `pytest.mark.skipif` 标注：找不到 ffmpeg 可执行文件时跳过。ffmpeg 优先取 `imageio_ffmpeg` 自带的版本。

### 写测试的几条做法

- **用替身代替重组件**：测试工具箱时用只带记账状态的 `Toolbox`（`test_task3_logic.py` 中的 `bare_toolbox`），或者感知和运动都写成脚本的 `ScriptedTB`；测试仿真服务器胶水层时用不含仿真器的替身环境；测试 LLM 时起一个本地 HTTP 服务器。这些替身只实现被测代码真正用到的属性，所以被测类新增属性时，替身也要同步补上（见 v5、v6）。
- **事故写进 docstring**：由运行事故引出的测试，docstring 里写明日期、场景和现象，例如 `test_move_arm_partial_only_while_the_label_stays_in_the_scanner_view` 写的是"2026-11-07 stack4f_L0_0002：手臂停在离目标 0.21 m 处，仍报告 partial"。读测试就能知道它防的是哪个问题。
- **阈值写明来源**：能从配置文件取的值直接读配置（如 stand-off 的 `fill_frac`、`min_m`、`max_m`）；测试自己设的阈值尽量在注释里写清依据（如 `STRONG = 60`）。也有没写依据的测试局部值，例如 `test_geometry.py` 的 RANSAC 参数只用于合成数据，见"已知限制"。

### 逐个文件

#### `test_geometry.py`（5 个）

- `test_ray_plane`：射线与平面求交；射线与平面平行、平面在射线背后两种情况返回 None。
- `test_triangulate`：两条射线三角化还原目标点，间隙小于 1e-9。
- `test_project_backproject_roundtrip`：50 个随机点投影后再反投影，坐标还原。内参取 fx = fy = 300、cx = 320、cy = 240。
- `test_horizontal_distance`：点到平面的水平距离。
- `test_plane_fit_synthetic_face`：相机在竖直面前方 1 m，下方有一张"桌面"。`fit_front_face` 必须拟合出法向 [-1, 0, 0]（误差 1e-3）、中心 x = 1.0（误差 1e-3），并忽略桌面。RANSAC 参数（300 次迭代、内点距离 0.01 m、最大倾角 20°、最少 100 个内点）是测试局部设定的值。
- 为什么加：初版就有，检查感知几何这一层最基础的数学。

#### `test_no_gt_leak.py`（2 个）

- `test_no_ground_truth_access`：用 AST 扫描算法代码，即 `g1agent/` 下的 `orchestrators/`、`tools/`、`perception/`、`harness/`、`kinematics/`、`llm/`，以及 `robot/sim_client.py`、`robot/interface.py`、`robot/rpc.py`。
  - 禁止 import 的模块：`g1agent.eval`（包括相对导入 `..eval`）、`eval_client`、`evaluator`、仿真包 `g1_barcode_sim`。
  - 禁止以属性、名字或字符串常量形式出现的名称：`get_gt`、`get_ground_truth`、`get_actual_state`、`EvalClient`、`evaluate`、`T_world_root_gt`、`set_gt_markers`、`task1_gt_obs_pose`、`task2_gt_wrist_target`。字符串常量也查，是为了防止通过 `getattr(obj, "get_gt")` 绕过。
- `test_robot_client_has_no_gt_methods`：`SimRobot` 类上不存在以上任何名称的方法。
- 为什么加：初版就有。评测必须只在 episode 结束后读取真值，算法在运行中一旦能读到真值，所有成功率都失去意义。

#### `test_markers_invisible.py`（1 个，集成测试，里程碑 M4）

- 先连接运行中的仿真服务器并 reset 到 `t2_L0_0000`；遇到 `TimeoutError`、`RuntimeError`、`OSError` 时跳过。
- 调用评测端口的 `invisibility_check`，由仿真服务器在同一物理状态下连续渲染、只切换标记，然后检查三件事：
  1. 对照组：标记显示与移除相比，头部和腕部相机中都有像素差超过 60 个灰度级，证明探针确实在两台相机的视野里；
  2. 标记隐藏与移除相比，没有任何像素差超过 60 个灰度级；
  3. 拍完第三人称 snapshot 之后，头部和腕部图像与之前完全相同（最大差为 0）。
- 阈值 `STRONG = 60` 的依据（代码注释）：仿真器的实时渲染在相邻帧之间不是逐位一致的（时间抗锯齿），实测同一状态下的噪声在少数像素上可达 55 个灰度级，而标记造成的变化超过 100 个灰度级。
- 为什么加：诊断标记（GT 目标点、harness 目标）只给录像相机看。如果机器人相机拍到了它们，就等于把真值泄露给了感知。这个测试的比较方式改过一次，见 v1。

#### `test_llm_client.py`（5 个，含里程碑 M7）

- mock 服务器 `H.do_POST` 对每个请求断言 `max_tokens` 和 `max_completion_tokens` 都不存在，对应"不设输出长度上限"的设计原则。测试客户端的配置把 `max_tokens` 设为 None。
- `test_tool_call_parsing`：工具调用参数被解析成字典。
- `test_truncated_reply_is_redrawn_not_parsed`：第一次回复以 `finish_reason == "length"` 结束（参数只有半截 JSON），客户端不解析它，重新抽样后得到 `look_head`，`length_truncations` 计 1；连续截断超过 `max_retries_on_length` 时抛出 `LLMTruncated`。
- `test_redraw_samples_and_timeout_is_redrawn`：mock 服务器让第一次回复睡 3 s，超过客户端 2 s 的超时。客户端把它记为一次 `timeout`，然后重新抽样。断言首次请求 temperature 为 0.0，重抽时为 0.7、top_p 0.8、top_k 20（测试配置中的 `redraw_sampling`）。
- `test_malformed_arguments`：参数不是合法 JSON 时，`arguments=None`，并附带 `parse_error`。
- `test_agent_full_with_mock_llm`：用脚本化的 LLM 回复驱动 `agent_full` 编排器跑一个假工具箱，包括"回复里没有工具调用"时编排器发出提醒，最后以 finish 结束。
- 为什么加：除超时重抽测试外的四个从初版起就有；超时重抽测试是 2026-11-07 的事故后加的，见 v8。

#### `test_task3_eval.py`（21 个）

- 列规则 `scan_order.column_order`：重叠的标签归为一列；链式重叠（A 与 B、B 与 C）也是一列；边缘恰好相接的两个标签分属两列。
- 自适应 stand-off：hFOV 32.8° 时名义符号对应 0.20 m，以及上下截断。
- `test_order_labels_in_root_frame`：机器人掉头（yaw 180°）后，左右顺序翻转。
- 评测器任务 3（共用 fixture `episode`）：
  - `test_task3_success`：使用 task1_end 快照；排除干扰项；重复读码和 no_read 只计数；`ref_source` 能区分 `scan_ray` 和 `decoded`。
  - `test_task3_wrong_order_and_missing_stage`：顺序错误判失败；缺少 task1_end 时参考系退回 `robot_init`，且失败原因中不提任务 1 阶段。
  - `test_task3_foreign_content_fails`：读到 GT 以外的内容判失败。
  - `test_evaluate_rejects_unknown_task`：任务号 4 抛出异常。
- 日志与时间线：`EpisodeLogger` 的事件和阶段、`tool_overlay` 的字段、LLM 客户端记录的 reasoning 和起止时间、`build_timeline` 生成时间线。
- 四面堆垛：
  - `test_labelled_faces_order_each_face_in_its_own_frame`：每个面在自己的坐标系里排序；标签法向倾斜 30° 时仍归到正面。
  - `test_task3_four_faces_success`：完整环绕判成功。
  - `test_task3_four_faces_other_face_order_reported_not_gating`：顺时针环绕只报告，不影响成功。
  - `test_task3_four_faces_failures`（参数化）：面不连续、首个面不对、面内顺序反了、少读一个，四种都判失败。
  - `test_task3_four_faces_foreign_content_fails`：读到外来内容判失败。
  - `test_task3_single_labelled_face_metrics_unchanged`：只有一个面带标签时，结果与没有 `faces` 字段时完全相同。
- 为什么加：任务 3 的评测规则（列规则、环绕顺序、哪些项参与判定）是需求里逐条定下的，测试把每一条固定下来。

#### `test_task3_logic.py`（52 个）

模块 docstring 写明这是任务 3 逻辑的 CPU 测试，不需要仿真器、检测模型或 GPU。按主题分组：

- **纯规则**：
  - `test_column_order_matches_evaluator_copy`：300 组随机输入下，算法侧 `g1agent/tools/scan_order.py` 和评测侧 `g1agent/eval/scan_order.py` 的 `column_order` 输出完全相同。评测器不 import 算法代码，两份实现靠这个测试保持一致。
  - 自适应 stand-off 的名义值与截断；`quad_width`、`point_in_quad`。
- **配置**：任务 3 继承任务 1 和 2 的配置；继承出现冲突时报错。
- **工具与 harness**：
  - `next_barcode`、`scan` 的去重和失败关闭；
  - `compute_wrist_target`：pre 位姿在 d + 0.10 m 处；refined 目标沿用头部目标的 stand-off；任务 2 固定 0.20 m；
  - `plan_scans`：在合成头部帧上分列、去重、增量更新；按计划位置的矩形重叠识别同一个标签；
  - 无标签的侧面规划为空；
  - 放置检查：任务 3 稍超容差时返回 partial，任务 2 返回 fail；
  - `move_arm` 只在标签仍在扫码视野锥内时返回 partial；
  - 没有可读码的标签记为 unreadable（6 组参数：只有"在视野内且什么码都没解出"的一组记为 unreadable，其余 5 组仍是普通的 no-read）；
  - 每个条码完成后重新观察，检查提示引导去 final 位姿；
  - 条码工具只作用于当前条码；不允许跳过未尝试过的条码；
  - harness 的工具集合、预算扩展、阶段事件、账本；没有扫描计划时 finish 被拒绝；还有面未扫时拒绝 success；成功需要完成整个环绕。
- **编排**：
  - `test_rule_task3_flow`：rule Task3 的完整流程，每次走向站位前手臂先回 home；
  - `task_class` 分派；recover 在预算耗尽时停止；
  - 到位误差分解为沿法向和横向两个分量；`move_summary` 去掉真值字段；
  - `agent_full` 在 `plan_scans` 和每次 `next_barcode` 之后压缩对话历史；
  - 下蹲：走到站位后再下蹲，手臂回 home 后再站起；下蹲时要走路必须先站起，手臂抬着时拒绝走路；
  - 矫正后的面图像中的检测能映射回平面；
  - 不可达的条码被报告、记录并跳过；
  - 面编号、占地多边形与有符号距离；按环绕顺序访问各面；单面任务不绕行；
  - 基础设施错误（如显存不足）直接结束 episode，普通工具错误仍返回结构化失败；
  - 嵌套检测框中丢弃外层框（外层是箱面，不是标签）；
  - 里程计漂移下（转 15° 并平移），每到一个新面就重新配准堆垛模型；
  - 多面场景下，站位规划的障碍物只包括当前面；
  - 必须先到达规划的站位才能抬手臂；
  - 同一个调用反复失败时被拦停，即使中间夹着成功的其他调用。
- 为什么加：最初的 18 个是在 2026-11-06 实现任务 3 时写的，其余是后来每次运行暴露一个问题时补的，见 v2、v5、v6、v7、v8。

#### `test_task3_review.py`（6 个）

任务 3 代码评审中发现问题的修复，每个问题一个测试：

- 到有限大小面的水平距离：机器人在面前方时算到平面，在侧面时算到边缘，斜向时算到角点。
- 已知面包括条码所在面的副本。
- 扫描计划存在后，拒绝 `detect_barcode_head`。
- stage 事件记录在触发它的工具记录之后。
- `plan_scans` 被拒绝后，rule Task3 停止：`plan_scans` 共被调用 `max_retries_per_tool` + 2 次，其中 `max_retries_per_tool` + 1 次执行并失败，最后 1 次被守卫拒绝，随后以 finish 结束。
- 有手臂工具的任务里，`agent_full` 会记录 home 位姿。

#### `test_stance_pool.py`（1 个）

- 同一个 Toolbox 状态下，`plan_workers = 0`（串行）和 `plan_workers = 2`（进程池）得到的 `stance_odom` 和被拒候选完全相同。
- 为什么加：2026-11-07 把站位规划改成进程池并行（`configs/robot.yaml` 中 `plan_workers: 16`），需要保证并行化不改变结果，见 v6。这个测试真的跑 IK，单独运行约 8 s。

#### `test_wrist_width.py`（5 个）

- `test_wrist_symbol_width_on_the_plane`（3 组参数）：合成正对平面的腕部相机帧（fx 1087、640×480），在平面内放一个 QR 码，三组（边长、距离、旋转）为 (0.05 m, 0.30 m, 0°)、(0.05 m, 0.30 m, 20°)、(0.08 m, 0.40 m, 35°)。`_wrist_symbol_width` 测得的宽度相对误差不超过 3%。
- `test_wrist_symbol_width_none_when_nothing_decodes`：图像中没有可解码的内容时返回 None。
- `test_refined_target_uses_the_wrist_width`：refined 目标使用腕部测得的宽度，0.08 m 的符号对应 stand-off 0.20 m，比头部估计的值小。
- 为什么加：stand-off 从"按标签宽度"改为"按印刷符号宽度"定义时新增，见 v4。

#### `test_compositor.py`（18 个）

- 测试场景：`_gt()` 是类任务 1 的场景，两个箱子各带一个 0.10×0.05 m 的标签，符号尺寸 0.08×0.03 m；`_gt4()` 是四面堆垛，含一个干扰项。
- `fit_fixed_camera`：20 组随机点全部落在画面内，至少一个轴两侧都贴到边距（误差 1e-3），光轴等于 forward。
- `gt_targets` 使用评测器的 stand-off：名义符号对应 0.20 m（误差 0.002）。
- `framing`（3 种机器人起点）：MAIN 相机都在左后方，SIDE 相机都在箱子的 +y 侧、目标面平面之前。
- `OverlayState`：检测结果只在它所在的那一帧可见；`target_text` 正确；未知事件抛出异常。
- `current_target`：按 content 匹配、陈旧目标、新目标三种情况。
- `frame_metrics`：面坐标系分量、读码去重、传入 task1_end 后根部数值冻结。
- 整帧合成输出 1080×1920×3 uint8。
- `H264Writer`：34 帧能完整解码，编码为 h264、yuv420p；写入进程被杀后文件仍可解码，不出现"moov atom not found"；帧尺寸错误抛出 `ValueError`；编码器已退出时抛出 `RuntimeError`。
- `_wrap`：中英混排时不丢字、不超宽。
- `face_framing`：单面返回 None；4 个面上 MAIN 相机分别落在对应的对角象限（4 组参数）；跳过无标签面时取逆时针角平分线；GT 中缺该面时抛出异常。
- `face_text` 能参与整帧合成。

#### `test_sim_server_episode.py`（3 个）

用不含仿真器的替身环境测 `g1agent/robot/sim_server.py` 中的 `Episode`：

- `test_episode_recording_and_i1_methods`：`record_start` 返回 task、every_steps，两台相机都已摆好；帧数 = 1 + 3 + LLM 冻结帧 + 检测冻结帧 + 2 × 扫码冻结帧；`episode.frames.jsonl` 的帧数与之吻合，最后一帧的已扫计数为 1；ffmpeg 解码无错误；reset 清空状态；单面场景中 `face_start` 不改变相机。
- `test_gt_standoff_follows_the_evaluated_task`：`gt_standoff_fn` 在任务 2 返回固定的 `offset_m`，在任务 3 返回评测器的自适应值；缺少 `pattern_size` 时抛出 `ValueError`，不回退。
- `test_episode_stage_log_and_face_cameras`：四面场景中，每个 `face_start` 都按 `face_framing` 重新摆放相机，相邻两个面的相机位置不同；`stage_log` 记录每一次出现，`stages` 字典只保留第一次。

#### `test_run_task_plumbing.py`（2 个）

- `test_run_task3_plumbing`：用假 robot、假评测端口、假检测器、假 Toolbox 和假 Task3，对 `make_episode_runner()` 返回的 `run` 调用 `run(..., task=3)`。检查任务号被正确覆盖；`task1_end_root_target` 是阶段触发时的快照（d_obs 0.6），而最终 `root_target` 是之后重算的值（0.3）；stage 事件发往 `stage`，tool 事件发往 `annotate`；调用了 `get_stage_states`、`get_scan_log`、`get_stage_log`；result.json、timeline、jsonl 文件都已写出。
- `test_eval_rpc_raises_on_error_reply`：`_eval_rpc` 遇到 `{"status": "error"}` 回复时抛出异常，而不是把错误当数据返回。

## 演进记录

测试数量的变化（来自各次 pytest 输出；"排除 marker"指排除 `test_markers_invisible.py` 后的数）：

| 时间 | 结果 | 说明 |
|---|---|---|
| 2026-11-05 20:00 | 11/12 通过 | 初版，标记测试失败 |
| 2026-11-05 20:04 | 12 passed | 标记测试改为连续渲染 + 灰度阈值 |
| 2026-11-06 14:12–14:24 | 53 → 61 → 62 passed（排除 marker） | 任务 3、视频合成、run_task 测试 |
| 2026-11-07 06:20 | 67 passed（排除 marker）+ 13 passed（合成与 episode） | 符号宽度定义 |
| 2026-11-07 07:52 | 69 passed（排除 marker） | 下蹲 |
| 2026-11-07 11:38–12:35 | 73 → 83 → 95 passed（排除 marker） | 并行站位规划、多面堆垛 |
| 2026-11-07 12:53–16:58 | 96 → 104 passed，1 skipped（全量） | 由运行事故引出的测试 |
| 2026-11-07 23:42 | 120 passed，1 skipped（全量） | v8 修复后的复跑 |

### v0 · 2026-11-05 · 初版：几何、真值隔离、mock LLM、标记不可见，共 12 个测试

**初版设计。** 系统的第一版代码写完后，加了 4 个测试文件和 `conftest.py`：

- `test_geometry.py`（5 个）：射线与平面求交（含平行、在背后两种返回 None）、三角化、投影往返、点到平面水平距离、合成深度上的平面拟合（K 取 fx = 300、cx = 320、cy = 240）。
- `test_no_gt_leak.py`（2 个）：算法代码不得访问真值，内容与当前版本相同。
- `test_llm_client.py`（4 个）：mock LLM 服务器上的工具调用解析、截断重抽、非法 JSON、`agent_full` 编排。
- `test_markers_invisible.py`（1 个）：需要仿真服务器在运行。

这一版的思路是：凡是能离开仿真器检查的数学和隔离规则，都写成 CPU 测试；只有"标记是否进入机器人相机"必须在真实渲染里看，单独做成集成测试，连不上服务器就跳过。

**遇到的问题。** 第一次运行全套，12 个中 11 个通过，`test_markers_invisible_to_robot_cameras` 失败（pytest 输出 `.........F..`）。修正过程见 v1。

### v1 · 2026-11-05 · 标记不可见测试：从逐像素相等改为连续渲染 + 灰度阈值 60

**当时的版本。** 仿真服务器的 `invisibility_check` 在同一物理状态下分别渲染"标记隐藏"和"标记移除"两种情况，要求机器人相机的两张图逐像素相等，即只要有一个像素不同就算不同。

**现象和数据。** 为了看清楚差异，我们先加了一张"同一状态再渲染一次"的基线图，按"任一通道不同"统计不同像素数：

- 头部相机：隐藏 vs 移除 253317 个像素不同，而基线（什么都不改，只是再渲染一次）也有 134212 个；
- 腕部相机：隐藏 vs 移除 273038 个，基线 135998 个。

也就是说，什么都不改，两次渲染也有十几万个像素不同，逐像素相等的判据本身不成立。

**根因。** 仿真器的实时渲染在相邻帧之间不是逐位一致的（时间抗锯齿）。改成统计灰度差以后：

- 头部相机的基线（同一状态）最大差为 55 个灰度级，没有像素超过 60；
- 但头部相机的"隐藏 vs 移除"仍有 15 个像素超过 60（最大 97；腕部相机这一项最大只有 9）。而这两张图是在不同的渲染历史下得到的（中间插着别的渲染），我们推测是时间抗锯齿把历史带进了当前帧；下面改成相同渲染历史后差异消失，与这一推测一致。

**怎么改的。**

1. `sim_server.py` 的 `invisibility_check` 改为一段连续渲染：先预热一次，然后依次"放入标记、移除、放入、移除"各渲染一次，再不做任何改动渲染一次作为噪声基线，然后显示标记渲染一次作为对照组，最后移除标记再渲染一次与对照组比较。这样每一对被比较的图都有相同的渲染历史，只差标记本身。每次读取相机图像前先渲染几次，次数由 `configs/viz.yaml` 的 `invisibility_render_passes`（2）控制。
2. 差异的统计改为逐像素取各通道的最大绝对差，再数超过阈值的像素。
3. 测试判据改为 `STRONG = 60`：隐藏 vs 移除的每一对都不能有像素差超过 60；对照组必须有；第三人称 snapshot 前后的机器人相机图像必须完全相同。60 取在同状态噪声上限 55 和标记造成的变化（超过 100）之间。

**改后效果。** 连续渲染方案下（场景 t2_L0_0000）：

| 比较 | 最大差 | 超过 60 的像素 |
|---|---|---|
| 头部：隐藏 vs 移除（3 对） | — | 0, 0, 0 |
| 头部：基线 | 32 | 0 |
| 头部：显示 vs 移除（对照） | 119 | 38 |
| 腕部：隐藏 vs 移除（3 对） | — | 0, 0, 0 |
| 腕部：基线 | 2 | 0 |
| 腕部：显示 vs 移除（对照） | 149 | 22425 |
| snapshot 前后（头部、腕部） | 0 | 0 |

之后该测试单独跑 1 次、全套连跑 3 次（每次 12 passed）、单独再跑 8 次，全部通过。当时的开发记录写的是"跑 12 次 11 次通过，第一次失败未能复现，原因待查"。按日志中的先后顺序，那一次失败发生在上述修改之前，用的还是逐像素相等的判据；修改之后的各次运行没有失败（这一判断来自日志顺序，推测那次失败就是旧判据造成的，没有再单独复现确认）。

### v2 · 2026-11-06 · 任务 3：测试从 53 个增加到 62 个，并随评测定义同步修改

**当时的版本。** 任务 3（找到箱子并按顺序扫描所有条码）的工具、harness、编排器、评测器、视频合成按三个分区并行实现，每个分区自带 CPU 测试：

- `test_task3_logic.py`（18 个）：扫码顺序（含与评测器副本一致）、自适应距离、配置继承、扫码账本、`next_barcode`、合成头部帧上的 `plan_scans`、Harness 的预算/阶段/账本、脚本化工具箱上的 rule Task3、恢复循环在预算耗尽时停止、到位误差、`move_summary` 去掉真值字段、用假 LLM 测 `agent_full` 的历史压缩。
- `test_task3_review.py`（6 个）：评审中发现的问题，内容与当前版本相同。
- `test_task3_eval.py`（12 个）：列规则（堆叠、链式、相接、yaw 180°）、0.10 m 标签对应 0.20 m 及两端截断、任务 3 的成功/顺序错/缺 task1_end/混入外来内容/未知任务。
- 另有 `test_run_task_plumbing.py`、`test_compositor.py`、`test_sim_server_episode.py`。

实现阶段结束时 53 个通过，任务分区评审后 61 个，视频分区评审后 62 个（排除 marker 测试，1 个 deselected）。`test_no_gt_leak` 仍然通过。

**遇到的问题。** 同一天按需求修改了任务 3 的成功定义：任务 1 阶段（到位精度）只报告为 `t1_success`，不再参与任务 3 的成功判定，因为任务 1 阶段按设计会提前结束，扫码阶段自己规划站位。改完 `evaluator._task3` 后，全套 1 failed、61 passed：`test_task3_wrong_order_and_missing_stage` 仍断言失败原因里包含 `task1_end`，实际失败原因是 `'scanned 1/2 GT barcodes'`。

**根因。** 测试固定的是旧定义（任务 1 阶段失败会写进失败原因），不是代码出错。

**怎么改的。** 把那一行断言改为 `"1/2" in m["failure"] and "task1_end" not in m["failure"]`，并加注释说明任务 1 阶段只报告、不参与判定。

**改后效果。** 62 passed，1 deselected。

**限制。** 这一批测试全部只在 CPU 上运行，当时没有在仿真器中验证。

### v3 · 2026-11-07 · 全量测试"卡住"：标记测试在没有仿真服务器时等满 600 s 超时

**现象。** 在没有仿真服务器的情况下跑 `python -m pytest -q tests -m "not marker"`，进程运行 3 分钟没有输出。py-spy 没有权限，从 `/proc` 看进程处于睡眠状态，内核等待点为 `do_poll`。我们杀掉了这个进程。

**根因。** 当时的判断是"`test_markers_invisible.py` 在没有服务器时有时阻塞在连接上而不是 skip"。对照代码和后来的运行用时，更准确的原因有两层：

1. `-m "not marker"` 按 pytest 的 mark 筛选，但这个测试没有注册任何名为 marker 的 mark，所以什么也没排除。此前用的 `-k "not marker"` 按测试名的关键字筛选，才真正排除了它。
2. 测试的 `_client()` 在构造 `EvalClient` 之后才把 `c.rpc.timeout_ms` 改成 300000。但 `g1agent/robot/rpc.py` 的 `RpcClient` 在 `_connect()` 时就已经把接收超时 `RCVTIMEO` 设到 socket 上，取值是 `configs/system.yaml` 的 `rpc_timeout_s`（600 s），之后再改 `timeout_ms` 只对下一次重连生效。ZeroMQ 的 REQ socket 在对端不存在时发送不会报错，所以第一次 `reset` 要等满 600 s 才抛出 `TimeoutError`，然后测试才被跳过。

数据支持第 2 点：此后包含该测试的全量运行用时都是 615.4–619.2 s，而用 `--ignore` 排除它的运行只需 7–18 s，差值约等于 600 s。所以这个测试不是永久挂起，而是等满 600 s 后跳过；进程被杀时只等了 3 分钟。

**怎么处理的。** 日常运行改用 `--ignore=tests/test_markers_invisible.py` 排除它；需要完整结果时让全量运行在后台跑完（约 10 分钟）。测试代码本身没有改，超时设置不生效的问题仍在，见"已知限制"。

### v4 · 2026-11-07 · stand-off 改按印刷符号宽度定义：新增 `test_wrist_width.py`，更新旧断言

**当时的版本。** 自适应 stand-off 按标签宽度计算，名义 0.10 m 的标签对应 0.20 m（`fill_frac = 0.85`）。测试中的默认宽度、断言值都按 0.10 m 的标签写。

**为什么要改。** stand-off 改为按印刷符号本身（不含白色衬底）定义：名义符号 0.08 m，`fill_frac` 改为 0.68，使名义符号仍落在已验证的 0.20 m。同时新增由腕部相机实测符号宽度、据此修正 refined 目标的功能。

**怎么改的。**

- 新增 `tests/test_wrist_width.py`（5 个）：合成正对平面的腕部帧（fx 1087、640×480），在平面内按不同旋转放置 QR 码，验证 `_wrist_symbol_width` 测得的宽度和由此设定的扫码距离。
- 更新旧测试：
  - `test_adaptive_standoff` 改为 0.08 m 符号 → 0.20 m；
  - `test_task3_logic.py` 的 `plan_of` 默认宽度从 0.10 改为 0.08；
  - `plan_scans` 测试中，头部测得的是 0.10 m 的标签框，对应距离改为按公式断言（头部规划偏保守）；
  - `test_compositor.py` 的 `gt_targets` 断言改用符号尺寸 [0.08, 0.03]。

**过程中的失败。** 首次运行：`test_compositor.py` 1 个失败（0.1999 vs 0.2499），`test_task3_logic.py` 2 个失败（0.25 vs 0.20）。都是断言仍按旧的标签宽度算期望值，按上面的方式改完即通过。

**改后效果。** `test_wrist_width.py` 5 passed；agent 侧 67 passed（排除 marker 测试，1 deselected）；合成与 episode 测试 13 passed。

### v5 · 2026-11-07 · 下蹲功能：替身补字段，新增下蹲流程测试

**当时的版本。** 为了扫描低处的条码，工具箱加入了下蹲（`base_height_cmd`、`body_cmd`、身体模型 `_body_model`），仿真服务器加入了对应的行走控制。

**遇到的问题。**

1. `test_sim_server_episode.py` 失败：替身环境没有 walker 对象。
2. agent 侧测试套件卡住，定位为 `test_markers_invisible.py`（原因同 v3），改用 `--ignore` 排除。
3. `test_rule_task3_flow` 报错：`ScriptedTB` 没有 `base_height_cmd` 属性。

**根因。** 替身只实现了被测代码当时用到的属性，被测代码新增的属性替身里没有。

**怎么改的。** 第 1 个问题改的是被测代码：`sim_server.py` 记录 `height_cmd` 时不再读 `env.walker`，改为从帧状态里取，替身环境因此不需要 walker。第 3 个问题改的是替身：`bare_toolbox` 默认把 `body_cmd`、`base_height_cmd`、`_body_model` 设为 None；新增 `SquatTB`（bc1 的站位需要下蹲）和测试 `test_rule_squats_after_walking_and_stands_up_with_the_arm_home`：先走到站位再下蹲，手臂回 home 之后再站起。随后又补了 `test_walking_while_squatting_stands_up_first_or_refuses_with_the_arm_raised`。

**改后效果。** 由 66 passed、1 failed 变为 69 passed（7.53 s）。

### v6 · 2026-11-07 · 并行站位规划与多面堆垛：73 → 95 个

**改动一：站位规划改为进程池。** 实测串行规划三个码分别用 10/30/104 s，8 进程 5/12/53 s，16 进程 5/10/47 s，32 进程 5/9/50 s；16 进程之后基本不再变快，于是 `configs/robot.yaml` 中设 `plan_workers: 16`。为保证并行不改变结果，新增 `test_stance_pool.py`：同一目标下串行和 2 个进程选出的站位、被拒候选必须完全一致。加入后 73 passed。

**改动二：多面堆垛（绕箱堆扫四个面）。** 工具箱新增 `multi_face` 属性后，`test_plan_scans_orders_columns_and_dedupes` 报 `AttributeError: 'Toolbox' object has no attribute 'multi_face'`（1 failed、72 passed）。原因与 v5 相同：替身缺新属性。在 `test_task3_logic.py` 的替身和 `test_stance_pool.py` 中补上 `tb.multi_face = False` 后恢复 73 passed。

随后加入多面相关的测试，例如面编号与占地多边形的有符号距离（`test_face_ids_footprint_and_signed_distance`）、按环绕顺序访问各面、单面任务不绕行、基础设施错误直接结束 episode、矩形重叠判定为同一标签；评测和视频侧有四面评分（`test_task3_eval.py` 中的 `four_faces` 系列）和换面时的相机摆放（`face_framing` 系列）。测试数依次为 83、93、94、95，全部通过。各测试具体在哪一次加入，日志中没有逐条对应，这里只列出这段时间的总数。

### v7 · 2026-11-07 · 运行事故引出的回归测试：96 → 104 个

**背景。** 这一阶段主要跑 `agent_full`（LLM 全权编排）的任务 3 场景。每次运行暴露的问题修复后，都补一个测试，测试的 docstring 写明出事的日期和场景。这一段（12:53 至 16:58）的全量运行结果依次为 96、97、99、101、102、102、104 passed，各 1 skipped（被跳过的是标记测试，每次用时约 615 s，原因见 v3）。

主要问题和对应的测试：

| 现象 | 根因与修改 | 对应测试 |
|---|---|---|
| 4 个面只扫完 2 个，LLM 就调用 finish(success=true) | harness 没有核对计划；还有面未扫时拒绝 success | `test_finish_success_refused_while_sides_remain_and_agent_full_keeps_the_task_statement`、`test_harness_success_needs_the_whole_tour` |
| `plan_stance` 之后没有走过去就抬手臂，然后放弃这个码 | 抬手臂前没有检查是否到达站位；改为必须先到达 | `test_arm_poses_need_the_planned_stance_reached_and_a_planned_stance_alone_is_no_attempt` |
| 一次工具调用被拒后，LLM 连续跳过了三个码（代码注释中的记录）；另外 LLM 曾自己编 stand-off 传给 `compute_wrist_target` | 当前码还没尝试过时，`next_barcode` 返回 fail（"not been attempted"），当前码不变（现在算作尝试的是：走向站位、扫码、或站位规划失败；只规划成功不算，见上面"没有走过去就抬手臂"那一行）；调用方传入的 offset 被忽略而不是拒绝。这个测试在 12:30 加入，已计入 v6 的 95 个 | `test_next_barcode_refuses_to_skip_an_unattempted_barcode_and_offsets_are_ignored` |
| 印花箱场景在观察位只检出 5 个标签中的 2 个：背景杂时检测器给标签的分数偏低（0.25–0.51），低于旧阈值 0.35 | 标签阈值从 0.35 降到 0.25（按 14 个采集视角的统计手定：真标签都 ≥ 0.25，整个箱面的误检都 ≤ 0.20，余量只有 0.05）；同时加一条结构规则：包住其他检测框的框是承载标签的箱面，不是标签，丢弃外层框。离线复核印花箱 2/5 → 5/5 | `test_label_dino_drops_enclosing_box` |
| 四面堆箱第 8 轮扫到 6/12 时 LLM 上下文溢出：对话历史每扫一个码约增长 4K token，12 个码一定超过 32K 的上下文上限 | 每次 `next_barcode` 成功后压缩历史，请求只含 system prompt 和一条用户消息（第一阶段摘要 + 该次 `next_barcode` 结果与扫描计划） | `test_agent_full_task3_compacts_history_at_every_next_barcode` |
| LLM 因 2.3 cm 的轴向偏差重新走了 4 次，一直没有扫码 | 任务 3 中放置检查稍超容差时返回 partial，并提示立即扫码 | `test_placement_check_measures_in_task3` |
| `stack4f_L0_0002`：手臂停在离目标 0.21 m 处仍报告 partial，结果扫码读不到 | 只有标签仍在扫码视野锥内时才算 partial | `test_move_arm_partial_only_while_the_label_stays_in_the_scanner_view` |
| 侧面拟合出来但没有标签 | 无标签的侧面规划为空（ok，0 个码），环绕继续 | `test_stack_side_without_labels` |

16:58 读到结果的那次全量运行在 16:35 启动，结果为 104 passed、1 skipped，用时 615.76 s。它启动时已包含无标签侧面的测试（`test_stack_side_without_labels`，2 组参数，102 → 104），但还不包含之后才写的两个测试：放置检查（16:36 加入，当时 3 组参数，后来又加了 "pre" 一组）和 `move_arm` 视野锥（16:54 加入，2 组参数）。这两个测试当时只在单独运行 `test_task3_logic.py` 时验证过（40、42 passed），第一次进入全量运行是 v8 的 120 个。当时的报告写的是"104 个通过，1 个跳过；新增的 4 个测试也全部通过"，与上面的时间顺序不完全对应。

### v8 · 2026-11-07 · unreadable、自动补看、超时重抽：120 个通过

**当时的问题**（都来自 2026-11-07 的运行）：

- `stack4c_L0_0001`：纸箱纹理上 23 张印刷的快递贴纸被规划成标签，机器人逐个去扫。
- 同一场景 v5：从观察位姿漏看的标签一直没有补进计划；在 pre 位姿已经读到的标签也没有被扫描。
- 同一场景 v6：LLM 没调用 `next_barcode` 就去处理下一个条码，一直在上一个码的站位上打转。
- `move_arm('pre')` 连续失败 14 次，每次之间都有一次成功的 `move_arm('home')`，把按工具计的连续失败计数清零，守卫一直拦不住。
- 一条贪心解码的 LLM 回复失控，连续 3 次撞上 120 s 超时（SDK 每次原样重发同一请求），episode 因此结束。

**根因与修改。**

- 没有可读码的"标签"无法区分：新增 unreadable 结果，并在 `barcode_end` 钩子里处理。测试 `test_scan_records_a_label_without_readable_code`（6 组参数）。
- 计划只在开始时生成一次：每个条码完成后重新观察，检查提示改为引导去 final 位姿。测试 `test_next_barcode_looks_again_after_each_barcode_and_check_guides_to_final`。
- 条码工具可以作用于任意码：改为只作用于当前条码。测试 `test_barcode_tools_work_on_the_current_barcode_only`。
- 连续失败按工具计数会被中间的成功清零：改为对同一个失败调用单独计数并拦停。测试 `test_the_same_failing_call_is_stopped_even_with_successes_in_between`。
- 温度 0 的重发会逐 token 重复同一个失控回复：超时或截断后改用模型自己的采样配置（`configs/llm.yaml` 的 `redraw_sampling`：temperature 0.7、top_p 0.8、top_k 20）重新抽样；`timeout_s` 从 120 改为 900 s，依据是按实测约 65 tok/s 的解码速度，生成到 32768 token 的上下文上限约需 500 s。测试 `test_redraw_samples_and_timeout_is_redrawn`。

**改后效果。** 在共享的 GPU 上复跑完整测试套件：120 passed、1 skipped，用时 618.14 s。当时的报告邮件误写成 122 个通过，随后发了更正。

## 已知限制

- **标记测试的超时设置不生效。** `test_markers_invisible.py` 的 `_client()` 在构造之后才修改 `rpc.timeout_ms`，而接收超时在连接时已按 `rpc_timeout_s`（600 s）设到 socket 上。没有仿真服务器时，这个测试要等满 600 s 才跳过，全量运行因此需要约 10 分钟。另外它没有注册 pytest mark，`-m "not marker"` 排除不了它，只能用 `-k "not marker"` 或 `--ignore`。
- **标记测试只覆盖一个场景。** 它只在 `t2_L0_0000` 上检查，判据依赖实测的渲染噪声（≤ 55 个灰度级）。换渲染设置或分辨率后噪声分布可能改变，需要重新测量。
- **任务 1、任务 2 的评分没有单独的单元测试。** `tests/` 中没有调用 `evaluate(1, …)` 或 `evaluate(2, …)`。`evaluator._task1` 和 `wrist_metrics` 只通过任务 3 测试间接用到（任务 1 阶段指标、扫码头误差）；`_task2` 只在真实 episode 中被用到。
- **脚本大多没有测试。** `run_suite.py`、`summarize.py`、`record_episodes.py`、`select_agent_subset.py`、`build_agent_clients.py`、`make_barcode_clips.py`、`make_highlight.py` 都没有测试。已知 `summarize.py` 遇到任务 3 的结果行会因 `err_cols[3]` 抛出 `KeyError`，`make_highlight.py` 对任务 3 会显示 nan，都没有被测试发现。
- **替身与真实类会走样。** `ScriptedTB`、`bare_toolbox`、替身环境只实现当时用到的属性。被测类新增属性时测试会以 `AttributeError` 失败（v5、v6 各出现一次），需要手动同步。反过来，替身行为与真实组件不一致的地方，测试发现不了。
- **CPU 测试不能代替仿真验证。** 感知、行走、IK 的真实表现只能在仿真 episode 中看到；单元测试通过只说明逻辑按设计执行。
- **测试局部参数。** `test_geometry.py` 的 RANSAC 参数、`test_wrist_width.py` 的 3% 误差上限、`test_compositor.py` 的 1e-3 和 0.002 容差，都是测试中设定的值，没有另外说明依据。
