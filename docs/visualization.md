# 可视化与视频

## 作用与在系统中的位置

可视化部分回答一个问题：一次 episode 里机器人到底做了什么，做得对不对。它产出三类东西：

- **诊断标记**：在仿真场景里画出 harness 的目标、真值目标和实际到达位姿，用于第三人称截图 `final.png`。
- **合成视频** `episode.mp4`：一个 1920×1080 的六格画面，把两台固定观察相机、机器人头部相机、腕部扫码相机、agent 的思考与工具调用、实时误差指标放在同一帧里。
- **后处理产物**：按条码或按面剪出的片段、带字幕的集锦，以及按时间列出所有 LLM 调用和工具调用的 `timeline.md`。

这部分只负责"看"，不参与决策和打分。它和评测侧一样能读到真值（GT），但只把真值画到给人看的画面上，机器人相机的画面里永远不出现任何标记。

| 对接方 | 接口 | 方向 | 内容 |
|---|---|---|---|
| harness / 工具箱 | 机器人端口 `draw_markers(markers, frame="odom")`、`clear_markers()` | harness → 仿真服务器 | 红（根目标）、蓝（头部相机估计的腕目标）、紫（腕部相机修正后的腕目标）三种标记，坐标在里程计系 |
| harness 日志（`g1agent/harness/logger.py`） | `on_event` 回调 → 评测端口 `annotate(event)`、`stage(name, info)` | harness → 仿真服务器 | LLM 回复、工具调用及其叠加信息（检测框、标签角点、扫描计划、当前条码）、阶段事件 |
| 运行脚本（`scripts/run_task.py`） | 评测端口 `set_gt_markers`、`record_start(path, fps, task)`、`record_stop`、`snapshot(path)` | 运行脚本 → 仿真服务器 | 开关真值标记、开始和结束录像、保存 `final.png` |
| 评测器（`g1agent/eval/evaluator.py`） | `gt_standoff`、`scanner_hfov_gt`、`gt_symbol_size`（经 `sim_server.gt_standoff_fn`） | 评测器 → 合成器 | 真值扫码目标点的 stand-off 定义，保证画面和打分用同一个口径 |
| 评测侧顺序规则（`g1agent/eval/scan_order.py`） | `FACE_TOUR`、`FACE_NAMES`、`assign_faces` | 评测侧 → 合成器、剪辑脚本 | 多面堆垛的面编号和标签归属 |
| 测试（`tests/test_markers_invisible.py` 等） | 评测端口 `invisibility_check()` | 测试 → 仿真服务器 | 检查标记不进入机器人相机 |

代码位置：

- `g1agent/viz/markers_usd.py`：在仿真器进程里把标记写成 USD 几何体。
- `g1agent/viz/compositor.py`：合成视频的全部 CPU 逻辑（取景、投影、指标、文字排版、H.264 编码），只依赖 numpy、cv2、PIL，可以脱离仿真器做单元测试。
- `g1agent/robot/sim_server.py`：`Episode` 类收集相机缓冲和事件，调用合成器写帧；`invisibility_check` 自检。
- `configs/viz.yaml`：颜色、尺寸、画布布局、取景、帧率、定格时长。
- `scripts/record_episodes.py`、`scripts/make_highlight.py`、`scripts/make_barcode_clips.py`、`scripts/make_timeline.py`：批量录制和后处理。

## 当前实现

### 1. 颜色约定

五种颜色写在 `configs/viz.yaml` 的 `colors` 段，诊断标记、合成视频、集锦字幕都用同一套：

| 颜色 | RGB（0–1） | 含义 | 由谁画 |
|---|---|---|---|
| 红 | [0.95, 0.10, 0.10] | harness 的根目标（观察位姿或扫码站位） | harness |
| 蓝 | [0.15, 0.35, 1.00] | harness 用头部相机估计的腕目标 | harness |
| 紫 | [0.70, 0.20, 0.90] | harness 用腕部相机修正后的腕目标 | harness |
| 绿 | [0.10, 0.85, 0.20] | 真值目标：任务 1 的观察位姿、每个条码的真值扫码点 | 仿真器（评测模式） |
| 黄 | [1.00, 0.85, 0.05] | 实际到达的根位姿和扫码头位姿 | 仿真器（评测模式） |

标记有三种形状：`root`（离地 `root_height` 处的圆点加水平朝向箭头）、`pose`（点加方向箭头）、`point`（只有点）。米制尺寸在 `sizes` 段：`root_radius 0.04`、`root_height 0.01`、`root_arrow 0.25`、`pelvis_radius 0.03`、`point_radius 0.012`、`pose_arrow 0.08`、`shaft_radius 0.006`（单位 m）。这些尺寸只影响画面，配置中没有给出取值依据。

### 2. 诊断标记：`g1agent/viz/markers_usd.py`

**为什么是 USD 几何体。** 仿真器的 debug-draw 叠加层只存在于交互视口中。我们以无头模式运行，用 render product 拍第三人称图，debug-draw 用不了。所以标记做成 `/World/Markers` 下的 USD prim。

**对机器人相机不可见的做法。** 标记根节点平时设为不可见。第三人称渲染（`sim_server.third_person_view`）时先 `show(True)`，渲染完 `show(False)`，再调用一次 `env.env.sim.render()`，保证之后机器人相机读到的缓冲不带标记。

**`MarkerLayer` 类：**

- 构造时创建根 Xform 并设为不可见；为每种颜色建一个 UsdPreviewSurface 材质（`_mat`：diffuse 取配置颜色，emissive 为 0.6 × 颜色，roughness 0.5，这几个数值代码中没有说明依据）。
- `set(markers, group)`：整组替换。约定 harness 用 `harness` 组（红/蓝/紫），评测用 `eval` 组（绿/黄），自检用 `probe` 组。prim 名把 `-` 和 `.` 换成 `_`。每次替换后都把根节点重新设为不可见。
- `clear(group)`：删除一组或全部标记。
- `_author`：root 类在 `root_height` 高度放一个球再加水平箭头；pose 类放点再加方向箭头；point 类只放球。箭头由圆柱杆和圆锥头组成，锥头半径 2.5r、高 5r（未给出依据）。
- `_orient_z_to(d)`：求把 +z 转到 d 的四元数，对同向（`c > 1−1e-9`）和反向（`c < −1+1e-9`）两种退化情况单独处理。
- `show(on)`：切换根节点可见性。
- `specs`：保存当前所有标记的规格。合成视频不再切换 USD 可见性，而是从这里读出 harness 标记，在 2-D 画面上自己投影（见第 4 节）。

**真值标记**（`sim_server._refresh_gt_markers`）：`set_gt_markers(True)` 之后，每次第三人称渲染前重新生成 `eval` 组：绿色的任务 1 观察位姿（root）和任务 2 腕目标（pose，任务 3 对全部箱子时没有单一腕目标，省略），黄色的实际根位姿（root）和实际扫码头位姿（pose）。

**harness 标记的坐标**（`sim_server.harness_markers`）：harness 只有里程计这一个固定坐标系，标记以 odom 坐标传入，服务器用当前的 `T_world_odom` 换到世界系再画。

**第三人称相机**（`viz.yaml third_person`）：在拍摄时刻的机器人根坐标系中定义，`eye_offset [0.6, 1.8, 1.6]`、`look_at [0.9, 0, 0.8]`、`hfov 60°`，即机器人左前方、跟随机器人。现在只用于 `snapshot()` 和每个 episode 结束时的 `final.png`。

### 3. 不可见性自检：`invisibility_check` 与 `tests/test_markers_invisible.py`

`sim_server.invisibility_check()` 在一个物理状态下（中间不推进物理）连续渲染，只切换标记：

1. 在头部相机前 0.6 m、腕部相机前 0.15 m 各放一个探针标记。
2. 预热渲染一次。
3. 依次"有、无、有、无"四次切换探针（根节点保持隐藏），每次渲染后读头部和腕部 RGB，得到 3 对"隐藏 vs 移除"。
4. 不做任何改变再渲染一次，与第 4 帧比较，得到渲染噪声基线。
5. 把探针设为显示再渲染，与移除后的帧比较，得到对照组"显示 vs 移除"。
6. 拍一次第三人称 snapshot，比较拍摄前后机器人相机的画面。

每次读相机前渲染 `invisibility_render_passes = 2` 次（与 `aux_render_passes` 相同）。差值按像素取三个通道的最大绝对差，报告 `max`、`p999` 以及超过 10/30/60/100 灰度级的像素数。

测试的判定（`STRONG = 60`）：

- 对照组：探针显示时，头部和腕部相机中都有像素差超过 60，证明探针确实在视野里。
- 3 对"隐藏 vs 移除"中都没有像素差超过 60。
- snapshot 前后头部和腕部画面完全相同（最大差为 0）。

阈值 60 的依据写在测试注释中：仿真器的实时渲染在相邻帧之间不是逐位一致的（时间抗锯齿），实测同一状态下的噪声在少数像素上可达 55 个灰度级，而标记造成的变化超过 100 个灰度级。该测试需要连接正在运行的仿真服务器，连不上（`TimeoutError`/`RuntimeError`/`OSError`）时跳过。

### 4. 合成视频

#### 4.1 画面布局

画布 1920×1080，六个面板（`viz.yaml recording.panels`，格式 [x, y, w, h]）：

| 面板 | 位置 | 内容 |
|---|---|---|
| MAIN | [0, 0, 1280, 720] | 固定观察相机，机器人起点的左后方，叠加投影的三维标记和颜色图例 |
| EGO | [1280, 0, 640, 360] | 头部相机原图；检测发生的那一帧画检测框（目标箱红色、其余蓝色、标签多边形紫色），并写"看到几个箱子、定位到哪个" |
| SIDE | [1280, 360, 640, 360] | 固定侧视相机，横向看过机器人与箱面之间的间隙，能看到真值扫码点和机器人到箱面的距离 |
| WRIST | [0, 720, 480, 360] | 腕部扫码相机原图；扫码那一帧画 pyzbar 解码多边形（绿色），并写是否读到、是否新码、已扫计数 |
| AGENT | [480, 720, 800, 360] | 阶段、面进度、扫码结果、最近一次工具调用及其结果，然后是 LLM 的用途、延迟、工具调用、思考内容和 reasoning |
| METRICS | [1280, 720, 640, 360] | pelvis 高度与高度指令、根部误差、扫码头误差、已扫计数和非 GT 内容、最小间隙 |

```{figure} images/viz_composite_pelvis.jpg
:width: 100%

当前版本的合成画面（多面堆垛：纸箱外观、两面贴码共 6 个真码的场景，agent_full 模式，LLM 思考时的定格帧）。MAIN 中黄色实心点画在 pelvis 高度并用虚线连到地面投影；METRICS 中根部误差在任务 1 结束时冻结。
```

#### 4.2 录制流程（`sim_server.Episode`）

`Episode` 保存一个 episode 在仿真服务器侧的全部可视化状态：叠加事件（`OverlayState`）、阶段记录、扫码日志、harness 标记的变化时间和录像句柄。随场景 `reset` 一起重置；上一个 episode 没有 `record_stop` 时，reset 会先把它收尾。

**`record_start(path, fps=None, task=None)`：**

1. 检查前提：录像相机已创建（`recording.cameras_enabled`）、已加载场景、扫码相机已开启，任一不满足就抛出 `RuntimeError`，不静默降级。
2. 帧间隔 `every = max(1, round(1 / (fps · policy_dt)))`。`fps = 5`、`policy_dt = 0.02 s` 时 every = 10，即每 10 个策略步一帧。实际帧率 `fps_eff = 1 / (every · policy_dt)`，视频的 1 秒对应仿真的 1 秒。
3. 读取一次真值并缓存（episode 内静态）。`task` 为本次评测用的任务号（`run_task --task` 可覆盖），默认取场景自带任务；用它从 `gt_standoff_fn(task)` 取得真值 stand-off，再调用 `compositor.gt_targets` 算出全部真值扫码点。
4. 用机器人碰撞模型的点（换到根坐标系）和 `compositor.framing` 求出 MAIN、SIDE 两台相机的取景，`fit_fixed_camera` 拟合位置后用 `env.pose_rec_camera` 摆好，再 `settle_renders`。
5. 打开 `H264Writer` 和 `<stem>.frames.jsonl`，写第一帧。

**`on_step`**：注册为环境的 step hook，每 `every` 个策略步调用一次 `_write(None)`。

**`_write(hold, hold_s)`**：

- 直接读取四台相机的 RGB 缓冲（`env.camera_rgb("main"/"side"/"head"/"wrist")`），不做额外渲染、不切换 USD 标记、不调用 `get_obs`。
- 组装标记列表：绿色观察位姿（多面场景下另加当前面的观察位姿）、每个真值扫码点（pose，方向为 −n）、`MarkerLayer.specs` 中的 harness 标记、黄色实际根位姿（带 `body_z` 和标签 "root (pelvis)"）、黄色扫码头位姿。
- 调用 `frame_metrics` 和 `Compositor.compose` 得到一帧。
- 普通帧写 1 次；定格帧写 `round(hold_s · fps)` 次同一帧。
- 每写一次，在 `episode.frames.jsonl` 追加一行：起始帧号、重复次数 `n`、定格类型、`t_sim`、`t_unix`、当时的指标。剪辑脚本靠这里的 `t_unix` 把阶段时间对到帧号。

**定格帧**（视频按仿真时间走，思考和检测本身不占仿真时间，所以需要定格才能看清）：

- `annotate` 收到 LLM 回复：定格 `llm_hold_s = 1.0` s，让文字可以读完。
- `annotate` 收到带头部像素检测的工具结果，或机器人端口 `scan()` 被调用：定格 `detection_hold_s = 1.0` s。
- 检测和扫码也定格，是因为检测叠加只画在检测发生的那个仿真时刻（见 4.4）。rule 模式没有 LLM 定格，检测时刻一般不落在每 10 步一帧的采样点上，不定格的话检测框永远不会出现在视频里。

**`stage(name, info)`**：每次调用都追加到 `stage_log`；`stages` 字典只保留每个名字第一次出现（评测器读 `stages["task1_end"]`）。多面场景中 `face_start` 触发 `_pose_face_cameras`，按 `face_framing` 为当前面重新摆放 MAIN 和 SIDE；单面场景 `face_framing` 返回 None，相机保持不变。

**`scan()`**：调用环境的扫码，把结果连同评测专用的 `gt` 字段、扫码头世界位姿、pelvis 高度、高度指令记入扫码日志；返回给机器人端口时去掉 `gt` 字段。

**`record_stop()`**：关闭 `frames.jsonl` 和编码器，返回帧数，并打印 `[record_stop]` 日志行。

#### 4.3 取景：`compositor.framing` / `fit_fixed_camera` / `face_framing`

**`fit_fixed_camera(points, forward, hfov, w, h, margin_frac)`** 是闭式求解：给定视线方向和视场角，把相机放在"所有内容点都落在画面内、四周留出 margin"的最近位置。

- 先求 `a = (w/2)/fx·(1−2m)`、`b = (h/2)/fy·(1−2m)`，把点转到相机朝向坐标系 `q = Rᵀp`。
- `L = max(q_x − a·q_z)`、`U = min(q_x + a·q_z)`，y 方向同理得 M、N。
- 最深可行位置 `c_z = min((U−L)/2a, (N−M)/2b)`，居中位置 `c_x = (L+U)/2`、`c_y = (M+N)/2`。
- `margin_frac` 必须在 [0, 0.5) 内，否则抛出 `ValueError`。

**`framing`**（record_start 时调用一次）：

| 相机 | 方位角定义的坐标系 | 方位角 / 仰角 | 要框进的内容 |
|---|---|---|---|
| MAIN | 机器人起点坐标系（0 = 前，90 = 左） | 135° / 24° | 起点处的机器人、GT 观察位姿处的机器人、所有箱子、所有真值扫码点 |
| SIDE | 目标面坐标系（0 = l，机器人正对该面时的左方；90 = n，面外法向） | 0° / 24° | 各箱朝向机器人那一面的角点、真值扫码点、GT 观察位姿处的机器人 |

参数依据（`viz.yaml framing` 注释）：135° 是左后方的角平分线；24° 沿用旧第三人称相机的仰角 atan(0.8/1.82)；`hfov 60°` 与 `third_person.hfov_deg` 相同；`margin_frac 0.05` 与仿真测试集参数 `task2_start.fov_margin_frac` 一致。

**`face_framing(..., face_idx)`**（多面堆垛，每个 `face_start` 时调用）：

- 用 `labelled_tour` 按环绕顺序（前 −x → 右 −y → 后 +x → 左 +y）列出带真值目标的面，归属规则与评测器相同（`assign_faces`）。带目标的面少于 2 个时返回 None；GT 中没有该面时抛出 `ValueError`。
- MAIN：相机所在方向（从内容指向相机的水平方向）取从当前面法向 n_cur 逆时针转到下一个带标签面法向 n_next 的角平分线，仰角沿用 MAIN 的 24°。相邻两面时是对角方向，相对两面时是机器人绕行经过的那一侧。框进所有箱子、两个面的目标、两个面观察位姿处的机器人。
- SIDE：对当前面套用 record_start 时的 side 规则。

**几何辅助函数**：`project`（针孔投影，ROS 光学坐标系）、`look_rotation`（forward 与 up 平行时抛出 `ValueError`，阈值 1e-9 属数值保护）、`intrinsics`、`view_direction`、`face_axes`（n 为水平外法向，`l = n × z`）、`box_corners`（只取 yaw）、`label_corners`（与评测侧实现相同）、`place_points`（把机器人身体点放到某个地面位姿）。

**`gt_targets(gt, standoff_m)`**：对每个非干扰条码算出真值扫码点 `pos = center + d·n`。`standoff_m` 由外部传入：仿真服务器的 `gt_standoff_fn(task)` 在任务 2 返回 `configs/task2.yaml` 的固定 `offset_m = 0.20`，其他任务调用评测器的 `gt_standoff(gt_symbol_size(bc), scanner_hfov_gt())`（印刷符号长边、`task3.yaml` 的 `fill_frac 0.68`、`min_m 0.12`、`max_m 0.30`、扫码器 hFOV 32.8°）。GT 缺 `pattern_size` 时直接报错，没有回退。这样画面上的绿色目标点和 METRICS 数字与评测打分只有一处定义。

#### 4.4 叠加事件：`compositor.OverlayState`

- `on_event(ev, t_sim)`：
  - `kind == "llm"`：保存这次 LLM 事件，返回 `"llm"`，服务器据此写定格帧。
  - `kind == "tool"`：若叠加信息带 `head_boxes` 或 `head_labels`，记录检测结果及当时的 `t_sim`，更新箱子数和箱子左右顺序，检测非空时返回 `"detection"`；同时更新 `target_box`、`scan_plan`，`current_barcode` 变化时记下 `current_since`。
  - 其他 kind 抛出 `ValueError`。
- `on_stage`：`face_start` 时记录面号和计划数，同面号的 `face_end` 到来时记录各条码状态。
- `head_detections_at(t)`：只有 t 恰好等于检测时的仿真时间才返回检测结果。相机移动之后的帧不会再画旧的像素框。
- `face_text()`：形如"面 face 2/4: right (-y) 计划 planned N 完成 done: k/n scanned"。
- `target_text()`：目标箱 id，加上它在最近一次头部检测中按 bbox 中心横坐标排出的位置（left、right 或 "#k from left"）。
- `current_target(targets, harness_targets)`：判断当前在处理哪个真值目标。优先用 scan_plan 中当前条码已解码的 content 匹配；否则在 `current_since` 之后设置的 harness 腕目标里按"紫色优先、蓝色其次"取第一个，找外法向射线离它最近的真值标签（`nearest_label_to_target`）；都没有时返回 None，面板显示"-"。

头部检测叠加由 harness 日志生成（`logger.tool_overlay`）：`plan_scans` 的结果从 `barcodes[*].corners_px` 读标签角点，任务 2 的 `detect_barcode_head` 从 data 顶层的 `corners_px` 读。

#### 4.5 指标：`compositor.frame_metrics`

所有分量都在目标面坐标系里（n 法向朝向机器人，l 机器人左方，z 向上），符号为"实际 − GT"。

- **根部**：机器人根刚体是 pelvis。GT 观察位姿只有 (x, y, yaw)，所以用 pelvis 的地面投影比较，给出 `dn`、`dl`、`|dxy|`、`dyaw`（归一化到 [−180°, 180°)），没有 dz。
- **根部冻结**：传入 `task1_end` 阶段状态后，根部数值固定为任务 1 结束时的值，`frozen_t` 记下当时的仿真时间，面板标"已冻结"。此后机器人会走向扫码站位，实时的根位姿差不再是误差。
- **扫码头**：已知当前目标时，给出扫码头相对其真值扫码点的 `dn/dl/dz/|d|`，并标出该点的 d。
- **已扫计数**：只统计 `status == "ok"` 且 content 不为空的不重复内容，输出 `k/n` 和读到的非 GT 内容。
- **其他**：`min_gap`（机器人与环境的最小间隙及对应的两个物体）、`pelvis_z`、`height_cmd`。

#### 4.6 绘制：`compositor.Compositor`

- 构造时检查每个面板都在画布内，否则 `ValueError`。字体用 Noto Sans CJK SC（字号 18，小字 15），`_font` 在 .ttc 字体集合中按 family 名逐个 index 查找：只在 index > 0 处捕获 `OSError`（越过集合末尾），index 0 就失败说明文件不可读，直接抛出；找不到指定 family 时抛出 `ValueError`。
- `_draw_markers`：把三维标记投影到 MAIN/SIDE。半径按 `fx·s·r_world/z` 换算成像素，不小于 `min_dot_px = 4`；相机后方的点跳过。`root` 类画地面圆盘和朝向箭头；带 `body_z` 时再在 pelvis 高度画实心点、用虚线（`dash_px = 6`）连到地面圆盘，并写标签。
- `_wrap`：可以在任意字符处断行，中英文混排和超长 token 都能折行，断点用二分法按渲染宽度查找。`_text_block`：高度用完时停止，在最后一行末尾补"…"，行高为字号 × 1.3。
- AGENT 面板把 LLM 内容放在最后，文字过长时被截掉的是思考内容的尾部；完整内容在 `timeline.md`。
- 各相机面板左上角有黑底说明文字；定格帧标"[冻结: LLM 思考 / 检测结果 / 扫码]"；MAIN 标颜色图例；每个面板画灰色边框。
- `_short(v)`：把工具参数压成一行，列表中的浮点数保留 3 位有效数字（字典里直接出现的数值按原样输出）。

#### 4.7 编码：`compositor.H264Writer`

原始 RGB 帧经管道送入 ffmpeg（仿真环境里 `imageio_ffmpeg` 自带的可执行文件），编码为 libx264、yuv420p，普通浏览器和播放器都能直接打开。

| 参数 | 作用 |
|---|---|
| `-tune zerolatency` | 不让 x264 按 rc-lookahead 和帧线程扣留帧。实测不加时写了 200 帧后磁盘上只有 36 字节的文件头 |
| `-movflags frag_keyframe+empty_moov` | 分片 MP4：写入进程被杀时，文件仍能播放到最后一个完整分片 |
| `-frag_duration round(1e6/fps)` µs | 每个帧周期一个分片 |
| `-flush_packets 1` | 逐包刷新到磁盘 |
| 帧率 `Fraction(fps).limit_denominator(1000)` | 精确表示非整数帧率 |

检查：宽高必须是偶数（yuv420p 要求）；每次写帧前检查 ffmpeg 是否还活着，已退出时抛出 `RuntimeError` 并附上日志末尾 2000 字节；帧尺寸或 dtype 不符时抛出 `ValueError`；`close()` 遇到非零退出码时抛出异常，日志为空时删除日志文件，返回写入的帧数。`write(rgb, repeat)` 支持同一帧写多次，定格帧就用这个实现。

### 5. 运行脚本中的可视化部分

**`scripts/run_task.py`**：`reset` 后调用 `set_gt_markers(True)`；需要视频时 `record_start(<ep>/episode.mp4, task=task)`；harness 日志的 `on_event` 回调把 stage 事件发给评测端口 `stage`，其他事件发给 `annotate`；编排结束后 `record_stop`、`snapshot(final.png)`；最后调用 `build_timeline(ep_dir)`。每个 episode 目录里与可视化有关的产物是 `episode.mp4`、`episode.frames.jsonl`、`final.png`、`timeline.md`、`timeline.jsonl`。批量运行脚本 `run_suite.py` 默认不录视频（`--video` 打开），截图总是保存。

**`scripts/record_episodes.py`**：对给定场景列表逐个以录像方式运行（复用 `make_episode_runner`）。已有 `result.json` 和 `episode.mp4` 时跳过，但会核对任务号（旧的 result.json 没有 `scenario_task` 时取 `task`），不一致时退出。`--until_success` 遇到第一个成功的 episode 就停止。

**`scripts/make_barcode_clips.py`**（任务 3）：

- 读取 `stages.jsonl` 中每个码的 `barcode_start`、`barcode_end`，用 `episode.frames.jsonl` 的 `t_unix` 二分查找映射到帧号，末尾多取 5 帧。
- 用 `select=between(n, f0, f1), setpts=N/5/TB` 按帧号截取，输出 1280×720、`-r 5`、libx264 `crf 26`、`preset slow`、yuv420p、faststart。
- 文件名 `<prefix><序号>[_f<face_id>]_<id>_z<计划中心高度>m_<状态>.mp4`，face_id 只在多面计划中出现。
- `--per_face`：另外按 `face_start` 到 `face_end` 每个面剪一段，文件名带该面扫到的码数，例如 `face1_right_2of3_scanned.mp4`。
- 最后输出整段 720p 的 `0_full_episode.mp4`。
- 没扫到的码也照样剪出，方便看在哪一步出问题。

**`scripts/make_highlight.py`**：给每个已录制的 episode 加标题再拼成集锦。第一行是场景、任务、模式、成败、因素值；第二行是误差（任务 1 为到位误差和观察距离，其他任务为手腕执行误差和感知误差），失败时附上 harness summary 的前 70 个字符；底部是颜色图例。标题框成功用 0x2e7d32、失败用 0xc62828，字号 26/22/20，顶部遮罩 92 px。每个 episode 先转成 `episode_h264.mp4`（crf 23，`-r 10`），再用 concat 拼接。这个脚本调用系统中的 `ffmpeg`。

**`scripts/make_timeline.py`**：把 `llm.jsonl`、`tool_calls.jsonl`、`stages.jsonl` 按时间合并成 `timeline.jsonl` 和 `timeline.md`。LLM 事件用真实的起止时间，记录新增的输入消息、每个 prompt 版本第一次出现时的 system prompt、thought、reasoning、工具调用（解析失败时附原始参数）、finish_reason、usage、尝试次数和被截断次数；工具事件带 rejected、budget_exhausted 标记和图像路径。消息全文保留，不截短。视频 AGENT 面板放不下的内容都在这里。

## 演进记录

### v0 · 2026-11-05 · USD 诊断标记和跟随机器人的第三人称录像

**初版设计。** 需求规定诊断标记不能出现在机器人相机里。仿真器的 debug-draw 叠加层只存在于交互视口，而我们无头运行、用 render product 拍第三人称图，debug-draw 用不了。于是在 `g1agent/viz/markers_usd.py` 里把标记写成 `/World/Markers` 下的 USD 几何体，平时不可见，只在第三人称渲染期间显示，渲染完立即隐藏并重新渲染一次。标记分 point / pose / root 三种，五色约定和米制尺寸（`root_radius 0.04`、`point_radius 0.012`、`pose_arrow 0.08` 等）写进 `configs/viz.yaml`，这两部分沿用至今。

同一时期的 episode 录像也走第三人称渲染（以下是 2026-11-06 代码调研时记录的状态）：

- `record_start` 用 `cv2.VideoWriter`，fourcc 为 `'mp4v'`（MPEG-4 Part 2），分辨率 1280×720，没有检查 `isOpened()`。
- 帧率 `third_person.video_fps = 2`，按仿真时间计，每 25 个策略步一帧。
- 每帧调用 `third_person_view()`：相机按当前机器人位姿重新计算（`eye_offset [0.6, 1.8, 1.6]`），是跟随机器人的左前方视角；每帧还要重新取真值、调用 `get_obs`、重写 USD 标记、显示标记渲染、隐藏后再渲染。

```{figure} images/viz_thirdperson_v0.jpg
:width: 80%

v0 的第三人称视角（`final.png`，任务 2 场景）：相机在机器人左前方并跟随机器人，黄色箭头是实际扫码头位姿。
```

### v1 · 2026-11-05 · 不可见性自检改为"同一渲染历史下只切换标记"

**当时的版本。** `invisibility_check` 在同一物理状态下比较三种情况：标记存在但隐藏（A）、标记删除（B）、标记显示（C），另把 B 再渲染一次作为噪声基线，按"任意像素不同"计数。

**问题。** 在 `t2_L0_0000` 上，头部相机 hidden_vs_removed 有 253317 个像素不同，而 B 与自身重渲的噪声基线也有 134212 个像素不同（腕部为 273038 对 135998）。渲染器本身就有噪声，这个判据没有意义。

**第二版。** 改成按差值阈值计数（`max`、`p999`、`n_gt10/30/60/100`）。头部 hidden_vs_removed 仍有 `n_gt10 = 1412`、`n_gt60 = 15`、max 97，而头部基线 max 55、`n_gt60 = 0`。

**根因。** 被比较的两帧渲染历史不同。实时渲染器的时间抗锯齿让每一帧都依赖前几帧，A、B 之间插着别的渲染，差异来自历史而不是标记。

**怎么改的。** 第三版在 `sim_server.invisibility_check` 中先预热渲染一次，再连续渲染并只切换探针的有无（有、无、有、无），比较相邻两帧；之后不改任何东西再渲染一次作为基线；对照组和 snapshot 前后比较保留。测试 `tests/test_markers_invisible.py` 用 `STRONG = 60` 判定，依据是第二版测得的同状态噪声最大 55、标记引起的变化超过 100。

**改后效果。**

| 量 | 头部相机 | 腕部相机 |
|---|---|---|
| hidden_vs_removed（3 对，n_gt60） | 0, 0, 0 | 0, 0, 0 |
| 噪声基线 max | 32 | 2 |
| 对照组 shown_vs_removed（max / n_gt60） | 119 / 38 | 149 / 22425 |
| snapshot 前后 max | 0 | 0 |

此后该测试单独跑 1 次通过，全量 12 个测试连跑 3 次都通过，单独再跑 8 次也都通过，修改后没有再出现失败。

### v2 · 2026-11-06 · 批量录制和带字幕的集锦

**需求。** 当时的批量评测为节省时间没开 `--video`，每个 episode 只存数值结果和一张 `final.png`。需求方要求至少保存一些成功和失败的视频。

**初版设计。**

- `scripts/record_episodes.py`：对场景列表逐个以录像方式运行，复用 `run_task.make_episode_runner`，已有结果的场景跳过。
- `scripts/make_highlight.py`：顶部字幕写场景、任务、模式、成败、关键误差、因素值、失败原因；底部写颜色图例；中文用 Noto Sans CJK；用 ffmpeg libx264 转成 H.264 的 `episode_h264.mp4`，再拼接成集锦。
- 当时选了 8 个代表性场景，与批量评测并行录制。测试拼出的 `highlight_partial.mp4` 含 2 个 episode，3.0 MB。

**决定。** 需求方随后表示不必每轮都录，先把实验跑完，再在每类场景下录 1–3 个，节省时间更重要。于是停掉了并行录制。完成的只有 `t1_L0_0003`（224.1 s）和 `t2_L0_0000`（257.4 s），都成功；`t2_L0_0000` 的扫码头停在条码前 0.20 m，真值与实际位置差 0.9 cm。

**问题。**

- `t1_L0_0003` 的视频只有 24.0 s，而 episode 跑了 224 s 墙钟时间。原因是录像按仿真时间记帧（1 秒视频 = 1 秒仿真时间），LLM 推理、规划等墙钟耗时不占仿真时间。这不是 bug，但视频里看不到这些耗时，也没有任何"在想什么"的信息。
- 中途停掉的录制进程留下了打不开的 mp4（见 v3 的根因）。

### v3 · 2026-11-06 · 六格合成视频与可中断的 H.264 编码

**需求。** 需求方反馈每个 episode 的 mp4 都打不开，并提出新画面：一个画面里包含左后方固定视角（最像正常观察，占图最大）、箱子侧面的固定视角（看目标点和机器人到箱子的距离）、ego 头部视角（标注是否看到箱子、定位到哪个箱子）、left wrist 视角（画识别框，文字显示是否扫码成功）、右下角显示根部和扫码头相对目标点的三维差距及已扫数量。

**问题与根因**（只读代码调研的结论）：

- 编码：`cv2.VideoWriter` 的 `'mp4v'` 是 MPEG-4 Part 2，浏览器、QuickTime、Windows 默认播放器、VS Code 预览只解码 H.264/yuv420p。仿真环境里的 cv2 只带 mpeg4、libvpx、libvpx-vp9 编码器，没有 libx264，所以不能只改 fourcc；又因为没检查 `isOpened()`，改了也会静默失败。PyAV 和 `imageio_ffmpeg` 自带的 ffmpeg 有 libx264。
- 截断：没执行到 `record_stop`（例如服务器被杀）时，mp4 缺 moov atom 无法打开，现有 20 个文件中有 2 个是这样坏的。
- 视角：第三人称相机每帧跟随机器人，是左前方视角，不是需求要的固定左后方视角。
- 开销：每帧重新取真值、调用 `get_obs` 并做多次额外渲染。

**怎么改的。**

- 新增 `g1agent/viz/compositor.py`（纯 numpy/cv2/PIL）：`fit_fixed_camera` 闭式视锥拟合（hfov 60°、边距 5%）；`framing` 给出 MAIN（起点坐标系方位 135°）和 SIDE（目标面坐标系方位 0°），仰角都是 24°；`frame_metrics` 在目标面坐标系下算根部和扫码头误差、按内容去重的已扫计数；`OverlayState` 管理事件；`Compositor.compose` 按 1920×1080 六格布局绘制；字体 Noto Sans CJK SC，18/15 px。
- 仿真侧新增两台固定的录像相机（MAIN 1280×720、SIDE 640×360，复用辅助相机的镜头参数）和只返回位姿、不复制图像的 `frame_state()`；扫码结果增加 pyzbar 多边形（映射回原始腕部图像坐标）。
- `sim_server` 新增 `Episode` 类：每 10 个策略步写一帧（5 fps × 0.02 s）；直接读四台相机的 RGB 缓冲，不再走 `render_aux` 和 `get_obs`，录像不再切换 USD 标记，标记改为 2-D 投影叠加。
- 定格：LLM 回复定格 1.0 s；检测和扫码也定格 1.0 s。后者偏离了最初设计，原因是检测框只画在检测发生的仿真时刻，rule 模式没有 LLM 定格，不定格的话检测框永远不会显示。
- 新增 `H264Writer`：经管道送 ffmpeg，libx264、yuv420p、`-tune zerolatency`、`frag_keyframe+empty_moov`、每帧一个分片、逐包刷新。
- EGO 只在检测时刻画 GroundingDINO 的框；WRIST 只在扫码时刻画 pyzbar 多边形。需求中"如果条码识别用的是 YOLO 就画 bounding box"一项：扫码头的识别是 pyzbar 解码，所以画解码得到的多边形。
- 日志侧：`EpisodeLogger.on_event` 把事件转发到评测端口的 `annotate`/`stage`，LLM 记录 `t_start`/`t_end`/`reasoning_content`，并新增 `timeline.md`/`timeline.jsonl`。

**改后效果。**

- 编码器被 `kill -9` 后，200 帧中有 199 帧仍可播放；不加 `zerolatency` 时只剩 36 字节的文件头。
- 任务 3 rule 集成测试（`t1_L0_0000`）运行中途复制出 `episode.mp4`，ffprobe 结果为 h264、1920×1080、yuv420p，说明录制过程中文件也能打开。
- 截图核对：MAIN 能看到机器人、桌子、两个箱子和各色标记；EGO 显示看到 2 个箱子；SIDE 显示每个标签的真值扫码点和机器人到箱面的距离；METRICS 显示根部和扫码头偏差、已扫 0/2、最小间隙。

```{figure} images/viz_composite_v2.jpg
:width: 100%

v3/v4 时期的合成画面（任务 3，rule 模式，扫码后的定格帧）。此时黄色实际根位姿只画在地面（MAIN 中机器人两脚之间），METRICS 的根部误差是实时值：机器人已走到扫码站位，显示 |dxy| 0.288 m。
```

### v4 · 2026-11-06 · 画面里的真值目标与评测器统一口径

**当时的版本。** `compositor.gt_targets` 用标签 `size[0]` 作为宽度，并在 `viz.yaml recording.standoff` 里自带一份自适应 stand-off 常数。

**问题。** 评测器在任务 3 用标签尺寸的 `max(size)` 和 `configs/task3.yaml` 的常数，在任务 2 用固定 0.20 m。于是：

- QR 标签（0.06×0.06）或其他尺寸的标签，视频中的绿色真值目标和 METRICS 中"扫码头相对真值目标"的数字与实际打分不一致；
- 任务 2 的视频用自适应距离（例如 0.12 m），评测器用 0.20 m。

**根因。** 同一个"真值扫码点"在合成器和评测器里各定义了一份。

**怎么改的。**

- `gt_targets(gt, standoff_m)` 改为接收一个函数参数，删除 compositor 和 `viz.yaml` 中的重复常数。
- `sim_server` 新增 `gt_standoff_fn(task)`：任务 2 返回 `task2.yaml` 的 `offset_m`，其他任务调用 `evaluator.gt_standoff`。只保留一处定义。
- `record_start` 增加 `task` 参数，`sim_server` 增加 `[record_start]`/`[record_stop]` 日志行。新增两个测试，全量 62 passed、1 deselected。

**后续（2026-11-07）。** 评测器的自适应 stand-off 改为按印刷符号尺寸计算（`gt_symbol_size` 读 GT 的 `pattern_size`，`task3.yaml` 的 `fill_frac` 从 0.85 改为 0.68）。因为合成器只接收 `gt_standoff_fn` 给出的函数，这次改动不需要改 `compositor.py`，画面自动跟随评测口径。

**集成时又补了两处：**

- `run_task` 原来调用 `ev.record_start(path)` 没传 task，用 `--task 3` 跑任务 2 场景时，视频里的目标和指标会按任务 2 的 0.20 m 算；改为 `record_start(..., task=task)`。
- `logger.tool_overlay` 原来只从 `plan_scans` 的 `barcodes[*].corners_px` 读头部标签角点，而任务 2 的 `detect_barcode_head` 把 `corners_px` 放在 data 顶层，所以任务 2 的标签框从来没画出来过；新增分支，读到时生成 `head_labels`。

### v5 · 2026-11-07 · 实际根位姿画在 pelvis 高度，根部指标在任务 1 结束时冻结

**问题。** 看视频时有人问"为什么 root 是左脚，一般不都是髋部么？"。另外，扫码阶段 METRICS 的"根 vs GT 观察位姿"显示差 0.29 m，容易被读成误差。

**排查与根因。**

- 位置取自 `robot.data.root_pos_w`，即关节链的根刚体 pelvis。各阶段 root 高度约 0.746 m（例如 task1_end 时 0.746，某次 barcode_end 时 0.747），与 GR00T 站立髋高 0.74 m 一致，数据没有错。
- 问题在画法：观察位姿目标只有 (x, y, yaw)，为了与真值在同一平面比较，黄色实际 root 被投影到地面、离地 1 cm（`root_height 0.01`）画圆点和箭头。从左后方斜视时，左脚正好盖住两脚之间的这个点。
- 扫码阶段的 0.29 m 是机器人按计划走到扫码站位后离开了观察位姿，不是误差。

**怎么改的**（改前备份为 `compositor_before_pelvis.py`）：

- `_draw_markers`：标记带 `body_z` 时，在真实 pelvis 高度（约 0.75 m）画实心点并标注 "root (pelvis)"，用竖直虚线连到地面投影，地面圆盘和朝向箭头保留。绿色 GT 观察位姿和红色根目标本身是地面位姿，仍画在地面。
- `frame_metrics` 新增参数接收 task1_end 阶段状态：任务 1 结束后根部一行冻结为结束时刻的值并返回 `frozen_t`；面板标题改为"根(pelvis 地面投影) vs GT 观察位姿"，冻结后标"已冻结"。
- `sim_server` 把阶段记录传给合成器。`viz.yaml` 新增 `pelvis_radius 0.03 m`、`dash_px 6`，只影响画面。
- `tests/test_compositor.py` 增加冻结断言（dn ≈ 0.03、dl ≈ 0.02）；agent 63 个测试和 compositor 13 个测试通过。

**改后效果。** 只改画图，不影响数据和评测。按需求重录了三种模式的视频：rule 扫到 2/2 个码（352 s）、agent_assist 2/2（364 s）、agent_full 2/2（314 s，28 次 LLM 调用，截断 0 次），扫码头误差均为 4.0 cm。

这次修改只针对合成视频。`markers_usd.py` 的 root 标记仍只画地面上的圆点和箭头，所以 `final.png` 没有变化。

### v6 · 2026-11-07 · 按条码自动剪片段

**需求。** 运行结束后要三个码各自的扫码视频。

**第一次是手工剪的**：按 `stages.jsonl` 中每个码的 `barcode_start` 到 `barcode_end`，经 `episode.frames.jsonl` 的 `t_unix` 映射到视频帧，用 ffmpeg 剪出每个码一段（从走向站位开始，经下蹲、举枪到扫码结束），再加完整 episode 的 720p 版本。没扫到的码也照样剪出，方便看在哪一步出问题。这一轮共 6 个附件 7.6 MB（完整 720p 3.6 MB，三个码的片段分别为 2.3 MB、596 KB、772 KB）。

**怎么改的。** 把这个流程固化为 `scripts/make_barcode_clips.py`：

- 用 `t_unix` 二分查找映射帧号，末尾多取 5 帧；
- 每个码输出一段 720p H.264（5 fps，crf 26），再输出整段 720p；
- 文件名带顺序、码 id、高度和最终状态（scanned/failed/unreachable），`--prefix` 区分不同运行。初版的高度优先取所读到的码的真值高度，否则取计划中心。

**效果。** 例如 `A_rule_z030_3_bc2_z0.30m_scanned.mp4`；在另一个底部码高 0.19 m 的场景中，底部码被报告为不可达，片段名为 `B_rule_z019_3_bc2_z0.19m_unreachable.mp4`，从文件名就能看出哪个码在哪个高度失败。

### v7 · 2026-11-07 · 多面堆垛：录像相机按面重新摆位，片段按面剪辑

**问题。** 多面堆垛场景中，机器人要绕到箱堆的另一面扫码。record_start 时定下的固定相机只针对第一面，绕到其他面后，SIDE 看不到当前面与机器人之间的间隙，MAIN 也可能被箱堆挡住（推测：日志没有记录具体的遮挡现象，只记录了改动）。

**怎么改的。**

- `compositor.face_framing`：`face_start` 时按 4.3 节的规则为当前面重新摆放 MAIN（当前面与下一个带标签面的逆时针角平分线）和 SIDE（当前面的侧视）；单面场景返回 None，相机不动。`sim_server.Episode.stage` 在 `face_start` 时调用它。
- `OverlayState.face_text()` 在 AGENT 面板叠加当前面、计划数和完成数。
- `make_barcode_clips.py`：文件名在扫码序号之后加入所在面 `face_id`（仅多面计划），高度改为只取计划中心高度（改动原因日志未记录）；新增 `--per_face`，按 `face_start` 到 `face_end` 每个面剪一段，文件名带该面扫到的码数，例如 `face1_right_2of3_scanned.mp4`。
- 测试覆盖：单面返回 None；4 个面上 MAIN 分别落在对应的对角象限；SIDE 只框进当前面的目标；跳过无标签面时取逆时针角平分线；GT 缺该面时抛出异常；四面场景中每个 `face_start` 都重新摆位且相邻两面的相机位置不同。

### v8 · 2026-11-07 · 剪辑使用中的两个问题和邮件附件的体积

**问题 1：运行环境缺依赖。** 在 agent 侧的 conda 环境里运行 `make_barcode_clips.py` 时报 `ModuleNotFoundError: No module named 'imageio_ffmpeg'`。该脚本用 `imageio_ffmpeg` 自带的 ffmpeg，而 agent 环境没装这个包。处理：改用装有它的 Python（系统 python3 或仿真环境）运行脚本，代码没有改。

**问题 2：附件太大。** 四面场景（12 个码）的单码片段加起来太大，不适合作为邮件附件。处理：用 `--per_face` 只发每个面的片段（例如 `af4f_face0_front_3of3_scanned.mp4`），完整 episode 另用 ffmpeg 手工缩到 960×540（libx264，crf 31），约 5.61 MB；5 个附件共 16.4 MB。540p 这一步没有写进脚本。

**问题 3：手工剪辑用错帧率。** 有一段"贴纸被判定为无可读码"的示例没有对应的自动片段（harness 当时没有为判定为 unreadable 的标签记 `barcode_end` 阶段，随后在 harness 中补上），需要按工具调用时间手工剪。第一次按 30 fps 计算时间轴，剪出帧 406–558 只有约 5.07 s，等于快放 6 倍。根因是 `episode.mp4` 的录制帧率是 5 fps。改为 `select=between(n,406,558), setpts=N/5/TB, -r 5` 重剪，得到 30.6 s、0.64 MB 的正常速度片段。另外生成了全程 8 倍速延时（540p，7.2 MB）和 6 个真码的单独片段，共 8 段约 14 MB。`make_barcode_clips.py` 本身按帧号截取并固定 `setpts=N/5/TB`，不受这个错误影响，但它把 5 fps 写死了（见已知限制）。

## 已知限制

- **剪辑脚本写死了 5 fps。** `make_barcode_clips.py` 的 `setpts=N/5/TB` 和 `-r 5` 与 `viz.yaml recording.fps = 5` 相同但没有从配置读取。修改录制帧率后，片段会快放或慢放，与 v8 问题 3 同类。
- **两个脚本依赖不同的 ffmpeg。** `make_barcode_clips.py` 要求 `imageio_ffmpeg`，agent 环境里没有；`make_highlight.py` 调用系统 `ffmpeg`。
- **`make_highlight.py` 不支持任务 3。** 任务 3 进入"其他任务"分支，而任务 3 的指标里没有 `execution_err_m`、`wrist_perception_err_m`，字幕中这两项显示为 nan。
- **计划中心缺失时文件名高度为 nan。** `make_barcode_clips.py` 在计划条目没有 `center_odom` 时用 nan 填充，不报错。
- **`final.png` 仍用旧画法。** 第三人称截图的相机跟随机器人（左前方），USD root 标记只画地面上的圆点和箭头，v5 的 pelvis 画法只用于合成视频。
- **视频按仿真时间走。** LLM 推理和规划的墙钟耗时在视频里只表现为 1 s 的定格帧，延迟只以文字写在 AGENT 面板。
- **AGENT 面板会截断长思考。** 超出面板高度的思考内容尾部被"…"替代，完整内容只能看 `timeline.md`。
- **检测叠加按浮点时间相等匹配。** `head_detections_at` 要求 `t_sim` 完全相等，依赖服务器在检测时刻立即写定格帧；检测叠加如果不触发定格，就不会出现在视频里。
- **无依据的画面参数。** 标记尺寸、`llm_hold_s`/`detection_hold_s` 的 1.0 s、行高 1.3 × 字号、USD 材质的 emissive 0.6 与 roughness 0.5、箭头锥头尺寸、集锦的 crf 和字号等，配置或代码中都没有给出依据，只影响观感。
- **不可见性阈值的余量小。** `STRONG = 60` 来自一个场景（`t2_L0_0000`）上测得的噪声最大 55，换场景或渲染设置后噪声可能更大。该测试需要正在运行的仿真服务器，纯 CPU 测试时会跳过。
- **单面场景不重新取景。** 相机在 record_start 时按起点、观察位姿、箱子和真值扫码点定一次；机器人如果走到这些内容点之外会出画。只有多面堆垛才在 `face_start` 时重新摆位。
