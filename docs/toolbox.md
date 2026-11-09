# 工具箱（Tools）

工具箱是编排器（规则编排或 LLM 编排）能直接调用的全部动作和测量的集合。代码在 `g1agent/tools/toolbox.py`（`Toolbox` 类和几个模块级几何函数）和 `g1agent/tools/scan_order.py`（扫描顺序与 stand-off 的纯函数），参数在 `configs/task1.yaml`、`configs/task2.yaml`、`configs/task3.yaml`。本页路径都相对于仓库根目录 `g1_agent_system/`。

## 作用与在系统中的位置

工具箱把感知、运动学、全身模型和机器人接口包装成一组"一次调用做一件事"的工具。编排器只看到工具名、参数和工具返回的 `{status, data, reason}`，不直接接触图像数组、关节角或仿真器。

| 相邻组件 | 方向 | 接口 |
|---|---|---|
| Harness 工具注册表 `g1agent/harness/registry.py` | 调用工具箱 | `TOOLS` 字典给每个工具定义参数模型和英文描述；`TASK_TOOLS` 决定每个任务暴露哪些工具（见下表） |
| Harness 守卫 `g1agent/harness/guards.py` | 读工具结果 | 按工具结果计预算、计失败次数、判断 `finish` 是否成立（例如 `next_barcode` 切到新条码时清零失败计数，返回 `face_done` / `done` 时不清零，`plan_scans` 返回计划后按条码数扩展预算） |
| 编排器 `g1agent/orchestrators/rule.py`、`agent_full.py` | 调用工具箱 | 规则编排按固定流程调用；LLM 编排由模型决定下一步。开始时调用 `remember_home()` 记下手臂 home 位 |
| 世界状态 `g1agent/harness/world_state.py` | 工具箱读写 | `WorldState`：箱子、拟合面、观察目标、条码估计 `BarcodeEstimate`、扫描计划、已读账本 `scanned`、箱堆模型 `stack` |
| 感知 `g1agent/perception/` | 工具箱调用 | `BoxDetector`（箱子与标签检测）、`plane_fit.fit_front_face`、`barcode_detector`（`detect_label_dino`、`detect_barcodes`、`decode_symbol_polygons`）、`geometry` |
| 运动学与全身模型 `g1agent/kinematics/`、`g1agent/tools/body.py`、`g1agent/tools/stance_eval.py` | 工具箱调用 | 正运动学求相机和枪口位姿，`ArmIK` 求手臂解，`BodyModel` 给出下蹲档位和间隙，`stance_eval` 并行评估候选站位 |
| 机器人接口 `g1agent/robot/interface.py` | 工具箱调用 | `get_joint_state`、`get_odom`、头/腕相机、`move_root_relative`、`set_body_pose`、`set_arm_joints`、`stand`、`scan`、`get_robot_info`、`get_robot_geometry`、`draw_markers` |
| 评测 `g1agent/eval/` | 不被调用 | 工具箱不导入评测包；评测端的排序规则是 `scan_order.py` 的单独副本 |

各任务能用的工具（`registry.py` 中的 `TASK_TOOLS`）：

| 任务 | 暴露的工具 |
|---|---|
| 任务 1：找到目标箱子，走到观察位姿 | `look_head`、`detect_boxes`、`select_target_box`、`fit_front_face`、`compute_obs_pose`、`approach`、`check_face_in_view`、`explore`、`finish` |
| 任务 2：扫描目标箱上的一个条码 | 除 `plan_scans`、`next_barcode`、`go_to_next_face` 以外的全部工具 |
| 任务 3：任务 1 + 按固定顺序把每个条码扫一次（可多面） | 全部工具 |

`finish` 由 Harness 实现，不在工具箱里。`adopt_stereo` 是工具箱方法，但没有注册给 LLM，只有规则编排使用。注册表的注释记录了按任务限制工具的原因：2026-11-06 所有工具都暴露时，LLM 编排在任务 1 的回合里继续调用了条码和手臂工具，导致任务失败。

任务 3 中一个条码的典型调用链如下（规则编排就是这条链；LLM 编排调用同样的工具，其中多处顺序现在由工具自己保证，见演进记录）：

```text
阶段 1（找箱子、走到观察位姿）
  detect_boxes → select_target_box('all') → fit_front_face → compute_obs_pose → approach('root_target')
阶段 2（规划）
  plan_scans                       # 矫正面检标签、排序、编号 bc0..；之后自动下蹲补看
每个条码
  next_barcode                     # 选出当前条码；上一个码结束后先从当前位置补看一次
  compute_wrist_target             # 可省略：plan_stance 会自动补算
  plan_stance                      # 网格 × 下蹲档位，IK + 间隙 + 鲁棒性
  move_arm('home') → approach('stance_target')   # 站着走过去，到了再自动蹲下
  move_arm('pre') → refine_with_wrist            # 腕相机修正中心、实测符号宽度
  move_arm('final') → check_wrist_placement → scan   # scan 内部会再重试
  move_arm('home')
当前面扫完（多面任务）
  next_barcode 返回 face_done → go_to_next_face → plan_scans → ……
```

```{figure} images/toolbox_stack4c_labels.jpg
:width: 90%

纸箱两面场景（stack4c）的一面：左上方白底小标签是真条码，两个箱子上的"PARCEL POST"运单贴纸是箱面纹理的一部分，上面的条纹读不出来。工具箱的 `unreadable` 状态就是为这类贴纸设计的。
```

## 当前实现

### 通用约定

**返回格式**（`toolbox.py` 文件头）。每个工具都返回 `{"status": "ok" | "fail" | "partial", "data": {...}, "reason": str}`：

- `R3(x)` 递归处理 dict、list、tuple 和 ndarray，浮点数四舍五入到 3 位小数，`np.integer` 转成 `int`。工具不返回大数组，图像只通过 `look_head` / `look_wrist` 以附件形式返回。
- 构造函数 `ok(data, reason)` 先 data 后 reason；`fail(reason, data)` 和 `partial(reason, data)` 先 reason 后 data。`_check_reason` 要求 reason 必须是 `str`，否则抛 `TypeError`（演进 v3）。
- 三种状态的含义：`ok` 是达到目的；`fail` 是没达到且需要调用方换做法；`partial` 是没完全达到但可以继续（具体语义在每个工具里写明，reason 会说明下一步）。

**不泄露真值**。`MOVE_LOG_KEYS = ("status", "reason", "elapsed_s", "requested", "odom_displacement")`，`move_summary(r)` 只保留这些字段。仿真器对行走命令的回复里带有真值位移和误差（只给评测用），它们不能进入日志，也不能交给 LLM。`tests/test_no_gt_leak.py` 检查整个算法代码不导入评测包、不读真值。

**坐标系**。感知在拍摄时刻的机器人根坐标系（root，骨盆）里完成，结果用拍摄时的里程计位姿 `T_odom_root` 存进里程计坐标系（odom）下的 `WorldState`。头相机和腕相机的位姿都由关节状态经 pinocchio 正运动学（`Kinematics.pose`）算出，两台相机之间的关系不用里程计推算。

**任务参数选择**。`_c1()` 返回任务 1 阶段的参数（任务 3 时返回 `cfg["task3"]`，否则 `cfg["task1"]`）；`_c2()` 返回条码阶段的参数（任务 3 时返回 `cfg["task3"]`，否则 `cfg["task2"]`）。任务 3 的配置通过 `inherits: [task1, task2]` 合并了两者，所以任务 3 下两个函数拿到的是同一个字典。

### `Toolbox` 的状态

构造函数 `Toolbox.__init__(robot, box_detector=None, device=None)` 初始化：

| 成员 | 内容 |
|---|---|
| `robot` | 机器人或仿真接口 |
| `cfg` | `config.load_all()` 读入的全部 yaml：system、llm、robot、perception、task1、task2、task3、viz |
| `kin`、`ik` | `Kinematics()`（正运动学、雅可比）和 `ArmIK(kin)`（手臂 IK） |
| `det` | `BoxDetector`，箱子和标签都用它检测；可由外部注入，避免重复加载模型 |
| `info` | `robot.get_robot_info()`：`head_K`、`head_size`、`wrist_K`、`wrist_size`、`body_command` 等 |
| `side` | 扫描臂，取自 `robot.yaml` 的 `arm_side`，当前为 `left` |
| `ws` | `WorldState()`，所有估计量都存在这里 |
| `episode_start_odom` | 回合开始时的里程计位姿，`explore` 用它限制工作区半径 |
| `task` | 由 Harness 设置（1/2/3），决定参数和若干工具的行为 |
| `_obs_faces` | `compute_obs_pose` 规划时冻结的面（接近过程中不变） |
| `arm_at_home` | 手臂是否在 home 位，初值 `True` |
| `_last_label` | 任务 2 路径下 `detect_barcode_head` 最近一次的标签检测 |
| `_wrist_detection` | `refine_with_wrist` 最近一次的腕部检测 |
| `wrist_hfov` | `hfov_from_intrinsics(fx, width)` 由腕相机内参算出的水平视场角（注释：DS2208 类成像器为 32.8°）。没有扫描相机时为 `None`，此时自适应 stand-off 会直接报错 |
| `body_cmd`、`base_height_cmd` | 身体高度（下蹲）命令的描述和当前命令值。行走控制器不支持高度命令时为 `None`，站位规划只在站立高度进行 |
| `multi_face` | 是否扫描箱堆的四个竖直面，默认 `False`（只扫正面），由 `scripts/run_task.py` 在任务 3 且场景 `scan_faces == "all"` 时设为 `True` |
| `_stance_attempted` | 已经"尝试过"的条码集合（见 `next_barcode`） |
| `_stance_reached` | 最近一次 `approach('stance_target')` 成功到达的站位；任何行走都会清空它 |
| `_last_arm` | 最近一次手臂命令 `{pose, barcode_id, status}` |

表中最后三项不在构造函数里赋值：`_stance_attempted` 是第一次访问时才建立的属性（底层集合 `_stance_attempted_ids`），`_stance_reached` 由 `_walk_guard` / `approach` 写入，`_last_arm` 由 `move_arm` 写入，读取处都用 `getattr` 兼容尚未赋值的情况。

内部辅助：

- `_odom()` 返回 `robot.get_odom()["T_odom_root"]`；`_q()` 读关节状态并转成 pinocchio 的 `q`。
- `_capture_head()` 保存头相机的 rgb、depth、K、`T_root_cam`（FK 到 `robot.head_cam_frame`）、`T_odom_root`、时间戳 `t`；`_capture_wrist()` 保存腕相机的 rgb、K、`T_root_cam`（FK 到 `robot.wrist_cam_frame`）、`T_odom_root`、`t`、`q`，机器人返回 `status == "error"` 时抛 `RuntimeError`。
- `_project_odom(frame, pts)` 把 odom 下的点投影到某一帧图像上，返回 `(uv, z)`。
- `_marker` / `_publish_markers` 把可视化标记以 odom 坐标发给 `robot.draw_markers`。

### 身体高度：`set_base_height` 与 `_walk_guard`

- `_body()` 第一次调用时构建 `BodyModel`（`g1agent/tools/body.py`），输入是下蹲标定 `assets/gear_body_calibration.json` 的 `levels`（11 个实测档位，高度命令 0.74 到 0.25）和 `robot.get_robot_geometry()` 返回的碰撞几何。没有身体高度命令时返回 `None`。
- `_standing()`：没有高度命令，或当前命令 ≥ `height_default − 1e-6`。
- **`set_base_height(height=None)`**：原地下蹲或站起，`None` 表示回到默认高度。高度超出机器人报告的 `height_range` 时返回 fail。调用 `robot.set_body_pose(height, None, ramp_s, settle_s)`，`robot.yaml` 中 `body.ramp_s = 1.0`、`settle_s = 2.0`，与标定时的渐变和保持时间相同。返回 `height_cmd`、`standing`、`pelvis_z`、`tilt_deg`。正常流程不需要直接调用它：`approach('stance_target')` 到达后自动下蹲，所有行走工具会先自动站起。
- **`_walk_guard()`**：所有行走工具（`explore`、`approach`、`go_to_next_face`）第一步都调用它。先把 `_stance_reached` 清为 `None`（一旦走动就离开了已到达的站位）；如果处于下蹲：手臂不在 home 位则返回 fail，提示先 `move_arm('home')`（站起时手臂不能扫过腿）；手臂在 home 位则自动 `set_base_height(None)` 站起。依据：蹲着走效果很差（GR00T 在 MuJoCo 中高度 0.60 时 7 s 只走 0.38 m，站立时 1.09 m），而且站起和下蹲应由工具负责，不能依赖调用顺序（演进 v13）。

### 观察与箱子工具

**`look_head()`**：拍一张头相机图像作为附件，同时返回 `ws.summary()` 和时间戳。**`look_wrist()`**：拍一张腕（扫描）相机图像作为附件。

**`detect_boxes(prompt="cardboard box")`**

1. 拍头相机图像，提示词规范化成 `"a {prompt}."`，调用 `BoxDetector.detect`（GroundingDINO + NMS + 容器框过滤，见感知页）。
2. 用检测结果重建 `ws.boxes`，编号 `box0`、`box1`……，顺序为图像中从左到右（机器人的左边在前）。
3. 清空 `target_box_id`；如果之前设过 `target_selector`，调用 `_reselect(prev_target_face)` 重新确定目标。
4. 没有检测到箱子时返回 fail，`{"n": 0}`。

`_reselect(prev_face)` 的规则：

| 情况 | 处理 |
|---|---|
| `selector == "all"` | 目标设为 `"all"` |
| 检测到 ≥ 2 个箱子 | `left` 选最左，`right` 选最右 |
| 只检测到 1 个箱子，且之前有目标面 | 把之前目标面的中心投影到新图像；投影点在相机前方（z > 0）且落在这个箱子 bbox 左右各放宽 `0.25 × bbox 宽` 的范围内时，判定为同一个目标 |

0.25 的放宽系数是硬编码常量，代码没有给出依据。

**`select_target_box(selector)`**：`"all"` 把所有箱子作为目标（任务 3）；`"left"` / `"right"` 需要至少 2 个箱子，只看到 1 个时返回 fail（无法判断它是哪一个）；也可以直接传箱子 id；其它值返回 fail。

**`fit_front_face(box_id=None)`**：单个箱子走 `_fit_one`；`"all"` 对每个箱子分别拟合，全部成功返回 ok，部分成功返回 partial，全部失败返回 fail，`data.unfitted` 列出各箱子的失败原因。

`_fit_one(box_id)`：调用 `perception.plane_fit.fit_front_face(depth, K, T_root_cam, bbox)` 在 root 系拟合竖直前表面，用 `T_odom_root` 把 center、normal、right、up、corners 转到 odom，连同 `size` 和 `t` 存入 `ws.boxes[id].face`；返回 `center_odom`、`normal_odom`、`size_wh`、`n_inliers`、`rms`，以及机器人到该平面的水平距离 `distance_m`（`geometry.horizontal_distance_to_plane`）。

### 观察位姿与接近（任务 1 阶段）

**`_faces(box_id)`** 返回一个箱子或全部箱子（`"all"`）的拟合面。对 `"all"` 且存在箱堆模型时，只保留 `_face_id == current` 的面，防止斜着看到的侧面混进观察位姿的计算。

**`_faces_reference(faces)`**（静态方法）求参考点和水平法向 `nh`：`nh` 是各面水平法向的平均；只有一个面时参考点是该面中心；多个面时，横向取所有角点在横轴上投影的中点，沿法向取最靠近机器人的那个面（这样 `d_obs` 就是到最近面的间距），高度取各面中心高度的平均。

**`compute_obs_pose(box_id=None)`**

1. 得到参考点 `center` 和法向 `nh`，朝向 `yaw = atan2(−nh_y, −nh_x)`，即正对面。
2. `d` 在 `d_obs_range = [0.60, 1.00]` 内按 `d_obs_step = 0.05` 递增，共 9 档。每档计算候选位置 `p = center + d·nh`，用当前的 `T_root_cam` 和 `head_K`、`head_size` 投影所有角点。
3. 取第一个让所有角点都落在图像内（留 `perception.fov_margin_frac = 0.05` 边距）的 `d`。没有任何一档满足时取 1.00，返回 partial。
4. 结果写入 `ws.root_target = {pos, yaw, d_obs, box_id, fits, n_faces}`，并把规划用的面（center、normal、corners）复制到 `self._obs_faces` 冻结（演进 v6）。

**`check_face_in_view(box_id=None)`**：如果存在冻结面且 `root_target` 对应同一个 `box_id`，用冻结面（全尺寸，位置经过重拟合重新锚定）；否则用当前拟合的面。拍头相机图像，检查所有角点是否在图像内（5% 边距），返回 `corners_px` 和每个角点的 `inside`。

**`move_to(pos, yaw, timeout=None)`**：用 `_rel(pos, yaw)` 把 odom 下的目标位姿换成相对当前位姿的 `(dx, dy, dyaw)`，直接调用 `move_root_relative`，超时默认 `approach.move_timeout_s = 20.0`。

**`explore(direction, distance)`**

- 先 `_walk_guard()`。
- 平移 `forward` / `left` / `right` / `back`：距离必须在 `(0, max_move_m]` 内（`robot.yaml` 中 0.5 m）。对每个已拟合的箱子面，如果机器人当前在面的前方（`d_now > 0`），而移动后到面的距离小于 `stance.min_face_dist_m = 0.35` 且比移动前更近，就拒绝这次移动（演进 v6）。
- 转向 `turn_left` / `turn_right`：角度必须在 `(0, max_turn_deg]` 内（15°）。
- 移动后离开回合起点超过 `workspace_radius_m = 8.0` m 时返回 fail（"left the workspace"）。
- 移动状态为 `ok` 或 `timeout` 都算成功，`ws.explore_steps += 1`。

**`approach(target="root_target")`** 是 `_approach` 的外层：先 `_walk_guard()`；目标是 `"stance_target"` 时，只要开始走，就把当前条码记入 `_stance_attempted`，成功到达后把 `_stance_reached` 设为当前站位（`move_arm` 要求这个标记）。

**`_approach(target)`**：

1. `biased(t)`：在目标朝向上提前 `final_bias_m` 停下。`root_target` 用 `final_bias_m`（当前 0.0），站位目标用 0。
2. 循环最多 `max_segments = 8` 段：剩余距离小于 `final_below_m = 0.3` 时，以 `precise_speed_scale = 0.5` 的速度走完最后一步并结束；否则走 `min(剩余 × segment_frac(0.5), max_segment_m(1.0))`，yaw 按同样比例分配。移动结果为 `fell` 或 `collision` 时立即 fail。目标是 `root_target` 时，每段之后调用 `_reperceive_target()`。
3. 目标是站位且站位带的 `base_height` 与当前高度命令不同时，到达后 `set_base_height(bh)`，即"站着走到站位，再蹲到规划高度"。
4. 目标是 `root_target` 时，最后进入 `arrive_check(log)`。

`precise_speed_scale = 0.5` 的依据（`task1.yaml` 注释）：控制器在距目标 0.15 m 之前一直饱和在 0.4 m/s，全速走短距离最多冲过头约 12 cm。

**`_reperceive_target()`**：最多 2 次（硬编码）`detect_boxes` → `fit_front_face`。第一次找不到目标时后退 `lost_backoff_m = 0.3` m 再检测，第二次仍找不到返回 fail。拟合成功后交给 `_retarget_from_refit()`；对 `"all"`，partial（部分面拟合成功）也可以继续，用这些面的并集。

**`_retarget_from_refit()`**：保留 `compute_obs_pose` 规划的 `d_obs` 和横向位置，只用重拟合的平面修正目标：

1. 对冻结面和重拟合面都存在的箱子，计算重拟合的平均法向 `nh` 和沿该法向的平均位移 `shift`；
2. 所有冻结面整体平移 `shift·nh`，法向换成重拟合结果；
3. 新目标 = 原横向坐标 + (最近面沿法向的位置 + `d_obs`)·`nh`。

没有可锚定的面时重新调用 `compute_obs_pose`，并把状态强制设为 ok。这样既能修正里程计漂移，又不受近距离下面被画面边缘截断的影响（演进 v6）。

**`arrive_check(log=None)`**

- 到达判据：`|沿法向误差| < dist_tol_m (0.03)`、`|dyaw| < yaw_tol_deg (3.0)`、面在视野内。误差由 `_arrival_error(tgt)` 在目标坐标系下计算（沿法向、横向、dyaw）。
- 横向偏移不参与判定：任务和评测（`evaluator.py`）只关心到面的距离、yaw 和面是否在视野内，横向偏移只通过视野检查间接起作用（演进 v5）。
- 最多 `max_fine_tunes = 2` 次微调：面在视野内时保持当前横向位置，只修正法向距离和 yaw；面不在视野内时横向一并修正。微调时控制器的停止容差为判定容差 × `fine_tol_frac (0.5)`（1.5 cm / 1.5°），速度 × 0.5。注释："执行控制器必须比检查停得更紧，否则会停在检查的边界上"。
- 仍不满足时：任务 3 返回 partial，提示"阶段 1 结束，下一步调用 plan_scans"；其它任务返回 fail。

### 单条码工具（任务 2 路径）

- `_current(barcode_id=None)` 返回指定的条码，没有指定时返回当前条码。
- `_not_current(barcode_id)`：仅任务 3。传入的不是当前条码时返回提示 "{id} is not the current barcode ({cur}): call next_barcode…"；当前没有条码时提示先调用 `next_barcode`。`compute_wrist_target` 和 `refine_with_wrist` 开头调用它（演进 v36）。
- `_pixel_on_face(frame, px, face)`：像素射线与面平面的交点（odom）；射线与平面接近平行时返回 `None`。
- `_label_on_face(frame, rec, face)`：把头相机图像中的标签检测映射到面平面上，得到中心、条方向 `bar_dir`（沿 `bar_dir` 偏移 10 px 取第二个像素，两点求交后相减；10 px 硬编码）、四个角点和宽度 `quad_width`。

**`detect_barcode_head(box_id=None)`**

1. 已经存在 `scan_plan` 时返回 fail（否则会产生计划外的当前条码）。
2. 把面角点投影到头相机图像，取外接框，调用 `detect_label_dino`。
3. 只保留中心落在面角点凸包内的标签（`cv2.pointPolygonTest`）。
4. 取第一个标签，生成 `bc_{box_id}` 写入 `ws.barcodes`，设为当前条码，记下 `_last_label`。

**`estimate_barcode_pose(barcode_id=None)`**：用 `_last_label` 和 `_label_on_face` 求条码中心、法向（即面法向）、条方向、角点和宽度，`source = "head"`。

### 扫描计划：`plan_scans` 及其辅助函数（任务 3）

**`plan_scans(_update=None)`**：首次调用建立计划，之后的调用是对计划的更新。

1. `detect_boxes`，对每个箱子 `_fit_one`，拟合失败的箱子记入 `skipped`。
2. 第一次调用且还没有箱堆模型时调用 `_stack_init`。只保留属于当前面（`_face_id == current`）的拟合面，其它面记入 `skipped` 并注明"属于 X 侧"。
3. `_stack_add_faces(current, faces)`（多面时会做重配准，见下一节）。
4. 当前面已经有计划条目，或显式传入 `_update=True` 时，进入更新模式。
5. 对每个面调用 `_labels_on_rectified_face`，在矫正后的面图像上检测标签：
   - 同一面内用 `_same_label` 去重（任一中心落在另一方的四边形内即视为同一标签）；检测结果已按得分排序，先出现的保留。
   - 跨面：同一标签可能被两个重叠的面区域都检测到。按"标签中心到相机的距离"从近到远排序后去重，保留最近的，也就是可见表面上的那一个。
6. 更新模式下交给 `_merge_into_plan` 合并，然后返回。
7. 首次规划：
   - 单面任务里一个标签都没有时返回 fail（"no barcode label localised on any fitted face"）。多面巡视中，本面已拟合但没有标签是合法结果，这一面按空计划处理（演进 v28）。
   - 用 `order_items_in_frame` + `column_order` 在当前头相机的 `T_odom_root` 坐标系下排序，依次编号 `bc{k}`（k 自增，跳过已占用的编号）。每个 `BarcodeEstimate` 的状态为 `pending`，并记录 `order_index`、`plan_center`（规划时中心的副本，之后不再改动）和 `face_id`。新条目追加在之前各面的条目之后（按面优先）。
   - 记录规划参考位姿 `ws.scan_ref_T` 和 `ws.scan_ref_Ts[cur]`，把本面加入 `stack.planned`，记下本面的 `d_obs`。
   - 清空当前条码和各条码相关的目标槽位（`_clear_barcode_slots`）。
   - 调用 `_low_view_update()`，从下蹲视角再看一次。
8. 返回每个条目的 `id`、`box_id`、`order_index`、`corners_px`、`center_odom`、`width_m`、`standoff_m`、`status`、`face_id`，以及 `low_view`。`order_rule` 字段写明排序规则："列按标签横向范围是否重叠划分，面对本面从左到右；列内从上到下"。

**`_labels_on_rectified_face(h, face)`**：把面矫正成正视图后再检测标签（演进 v8）。

1. 面内坐标轴：`v` 为竖直方向在面内的投影，`u = v × n`（从相机一侧看是图像右方），由角点得到 u、v 的范围。
2. 采样密度 `s = fx / 最近角点深度`（px/m）。最近角点处源图采样最细，按它采样不会丢掉源图细节；深度下限钳到 `1e-3`。
3. 用面在 u、v 方向的包围矩形的四个角（投影到头相机图像）求单应（`cv2.getPerspectiveTransform`），`cv2.warpPerspective`（`INTER_CUBIC | WARP_INVERSE_MAP`）生成 `Wr × Hr` 的矫正图。
4. 在矫正图上运行 `detect_label_dino`，结果映射回 3D 得到中心、角点、条方向（矫正图中的 `bar_dir` 换算为 `bd_x·u − bd_y·v`）和宽度；同时投影回头相机图像得到 `rec`，用于叠加显示和去重。

**`_low_view_update()`**：首次规划后从下蹲视角补看面的下部（演进 v20、v32）。

- 前提：有身体模型、机器人站立、手臂在 home 位、已经有计划。
- 取当前面上所有面的底边角点。对每个标定档位 `lv`，用 `body.q_at`、`body.foot_shift`、`body.T_root` 预测双脚不动时的头相机位姿，计算所有底角是否都在图像内（5% 边距），以及得分 `fx / 最远底角深度`（最差底角处每米对应的像素数）。
- 取所有在图像内、且得分高于站立得分的档位，按标定档位的高度从高到低（从浅蹲到深蹲）逐个执行，代码不按得分排序：蹲下 → `plan_scans(_update=True)` → 记录新加入的条码。最后站起；站起失败抛 `RuntimeError`。
- 没有合格档位时返回 "no squat sees the faces' bottom better than standing"。
- 不只取"最好的一档"的原因：开放词表检测器的召回率随视角变化，且不随高度单调变化（数据见演进 v32）。

**`_merge_into_plan(cands, h, boxes, skipped)`**：对每个候选标签，用 `_same_planned_label` 判断它是否已经在当前面的计划里，不在就新建 `pending` 条目。有新条目时，当前面上所有 `pending` 且不是当前条码的条目（旧的和新的）在该面的规划参考系 `scan_ref_Ts[cur]` 下按 `column_order` 重排；已扫、失败、当前条码和其它面的条目位置不变；最后重新编号 `order_index`。返回 `updated`、`added`、`n_seen`、`n_planned`、`scan_plan`。

**`_same_planned_label(est, b)`**（模块级）：新检测 `est` 是否就是已计划的标签 `b`，两个条件同时满足：

- 同一表面：两者中心沿面法向的距离 ≤ `w`（两者宽度中较大的那个）；
- 面内矩形（沿面内 u、v 轴的轴对齐包围）严格重叠——两个物理标签不可能互相重叠，所以这里不需要人为距离阈值。

比较对象有两个：`b` 规划时的矩形（`corners_odom`，不随后续精化移动），以及这个矩形平移到 `b` 当前中心后的位置。即使 `b` 后来被错误地精化，同一个标签也不会被当成新标签，计划更新不会陷入循环（演进 v17、v19）。

**`next_barcode()`**

1. 既没有计划也没有已规划的面时返回 fail。
2. 上一个条码仍是 `pending` 且没被尝试过（不在 `_stance_attempted` 中）时拒绝跳过，提示完整流程 `compute_wrist_target → plan_stance → approach → move_arm / refine_with_wrist → scan`。"尝试过"指：已开始走向站位（不论结果）、`plan_stance` 失败、或调用过 `scan`；只是规划成功不算（演进 v24、v27）。
3. 上一个条码已尝试但仍是 `pending` 时标为 `failed`。
4. 清空槽位。任务 3 下如果有上一个条码，调用 `_plan_update_from_here()`（即 `plan_scans(_update=True)`）从当前位置补看一次（演进 v35）。
5. 取当前面上第一个 `pending` 条码作为新的当前条码。当前面没有剩余时：还有未访问的面返回 `face_done=True` 和 `next_face`，提示调用 `go_to_next_face`；所有面都访问过返回 `done=True`。否则返回 `index`、`box_id`、`remaining`、`width_m`、`standoff_m`。
6. 每次都返回 `scanned`（已读内容列表）和 `scan_plan_export()`。

`_clear_barcode_slots()` 清空 `wrist_target`、`wrist_target_refined`、`stance_target`、`_wrist_detection`、`_last_label`。`_next_face()`：单面任务返回 `None`；多面任务返回巡视顺序中编号大于当前面、且还没访问过的第一个面。

### 多面箱堆模型与 `go_to_next_face`（任务 3，多面）

**数据结构 `ws.stack`**：

| 字段 | 含义 |
|---|---|
| `n0` | 正面外法向（S 系） |
| `faces` | `{fid: {"corners", "d_obs"}}`，角点存在 S 系 |
| `current` / `visited` / `planned` | 当前面、已访问的面、已规划的面 |
| `multi_face` | 是否扫描四个面 |
| `R_OS`、`t_OS` | S 系到 odom 的刚体变换 |
| `registrations` | 每次重配准的修正量（`dyaw_deg`、`dxy_m`） |

- `TOUR = ("front", "right", "back", "left")` 是巡视顺序。
- 箱堆模型放在自己的平面坐标系 S 里（取第一次规划时的 odom 系），通过 `(R_OS, t_OS)` 映射到 odom（演进 v26）。`_stack_T`、`_stack_to_odom`、`_odom_to_stack` 只变换 xy，z 不变。`_stack_face(fid)` 返回面 fid 在 odom 下的法向、角点、中心和 `d_obs`。
- `_face_dirs()`：四个面的水平外法向，依次为 `n0`、`Rz(+90°)·n0`（机器人面对正面时的右侧）、`−n0`、`Rz(−90°)·n0`，再用 `R_OS` 转到 odom。
- `_face_id(normal)`：取与四个方向内积最大的面。四个方向两两相差 90°，所以不需要阈值。没有箱堆模型时返回 0。
- `_stack_init(faces)`：第一次规划时，从已拟合的面里选法向与机器人前向内积最小（最正对机器人）的那个作为正面，初始化 `n0`。

**`_stack_add_faces(fid, faces)`**：记录面 fid 的角点。巡视中前一个面 `nb` 已知且 `fid ≠ 0` 时先重配准：

1. 旋转：本次拟合的平均水平法向 `nh` 与模型预测方向 `d_pred` 的有符号夹角为 `delta`，更新 `R_OS ← Rd·R`、`t_OS ← Rd·t`。
2. 平移：两个面共享一条棱。模型中的棱位置 `corner_model` = 前一个面在其法向上的平均位置 + 它沿本面方向的最大横向延伸；新测到的 `corner_fresh` = 本面在其法向上的平均位置 + 它沿前一个面方向的最大延伸。用两者之差修正 `t_OS`。
3. 修正量记入 `registrations`；新角点转到 S 系，与这个面之前的角点合并（`vstack`）。

**`_footprint()`**：已知面少于 2 个，或所有面互相平行（每个面方向与第一个面方向的 |内积| > 0.5）时返回 `None`；否则返回所有面角点 xy 的凸包（`scipy.spatial.ConvexHull`，逆时针）。0.5 硬编码，没有注释依据；由于方向只有平行和垂直两种关系，0 到 1 之间任何值效果相同。

**`_root_clearance(pxy)`**：轮廓闭合时返回到轮廓多边形的有符号距离（内部为负，`_poly_signed_distance`）；未闭合时，在点落在其横向范围内的已知面中取到面平面的最小有符号距离；什么都不知道时返回 `None`。

**`_move_axis(dx=0, dy=0, dyaw=0, log=None)`**：每次只沿一个轴运动（`assert`）。运动控制器对 x、y、yaw 分别饱和，只有单轴运动才能保证走直线。平移时在路径上取 11 个等距点（`np.linspace(0,1,11)`，硬编码），任一点 `_root_clearance < min_face_dist_m − 1e-6` 就拒绝。移动状态 `ok` 或 `timeout` 视为成功。

**`go_to_next_face()`**

前置检查：存在箱堆模型；有下一个面；当前面上没有 `pending` 条码；当前面已拟合。

1. 手臂不在 home 位先 `move_arm("home")`，再 `_walk_guard()` 站起。
2. 环绕距离 `d_ring` = 当前面记录的 `d_obs`（没有记录时用 `d_obs_range[0] = 0.60`）；棱角 `corner` = 当前面上沿下一面法向最远的角点。
3. 单轴运动序列：原地转向正对当前面 → 前进/后退到 `d_ring` → 横移到 `corner + d_ring·n_next` → 原地转向正对下一面 → 沿下一面再横移 `d_ring`（朝箱堆纵深方向，演进 v25）。
4. 临时把 `current` 设为下一面，调用 `_fit_and_reach_side`；不成功时在 `finally` 中恢复原来的面。

**`_fit_and_reach_side(nf, log)`**：共两遍（`range(2)`，先在面的近端，再在观察位姿）。每遍：`detect_boxes`、目标设为 `"all"`、逐箱 `_fit_one`，只收集属于第 nf 面的拟合面；第一遍先删除 nf 面原有角点，再 `_stack_add_faces`（这一步会重配准）；然后 `compute_obs_pose("all")`，按横移 → 前后 → 转向的顺序走到观察位姿。两遍结束后把 nf 加入 `visited`，记下 `d_obs`，用重配准后的上一面法向算出本面延伸长度 `side_extent_m`（即箱堆深度）。失败时在返回里写明看到了哪些面、各属于哪一侧（`seen`）。返回提示：下一步调用 `plan_scans`。

### 手腕目标与 stand-off

**`_standoff(bc)`**：扫描枪尖端到条码的距离。

- `standoff.mode == "fixed"`（任务 2）：返回 `offset_m = 0.20`。
- `mode == "adaptive"`（任务 3）：`adaptive_standoff(width_m, wrist_hfov, fill_frac, min_m, max_m)`。宽度来自头相机（`width_source != "wrist"`）时再取 `max(d, offset_m)`：头相机宽度只能让扫描枪更远，只有腕部实测宽度才能让距离小于 0.20 m（演进 v15）。
- 宽度未知或缺腕相机内参时抛 `RuntimeError`，不退回默认值；其它 mode 抛 `ValueError`。

**`_pre_gap()` / `_pre_offset(off)`**：预位姿比最终位姿多后退的距离。`plan_stance` 为当前条码找到过可达的 `pre_gap` 就用它，否则用 `pre_offset_m − offset_m = 0.30 − 0.20 = 0.10`。预位姿距离 = `off + pre_gap`。

**`compute_wrist_target(barcode_id=None, offset=None, refined=False)`**

- 先过 `_not_current` 检查。
- 自适应模式下调用方传入的 `offset` 被忽略（不报错），记在 `offset_ignored` 里（演进 v13、v24）。
- `refined=True`、宽度不是腕部实测、且已有同一条码的头相机目标时，沿用那个目标的 offset；否则调用 `_standoff`。
- 目标：`pos = 中心 + off·法向`，`forward = −法向`，以及 `bar_dir`、`offset`、`barcode_id`、`long_axis`。`refined=False` 时清空旧的 `wrist_target_refined`。

**`_long_axis(bc)`**：有 `bar_dir` 时取 `n × bar_dir`（面内垂直于条纹的方向，也就是 code128 扫描线需要走的方向）；否则取标签四边形较长的边；投影到面内并归一化。IK 用它约束绕扫描轴的 roll，使扫描图像的水平轴与条码长轴平行（演进 v21）。返回数据的 `note` 字段仍写着"绕前向轴的 roll 不受约束"，与当前行为不一致（见已知限制）。

`_roll_in_root(t, T_odom_root=None)`、`_target_in_root(t, T_odom_root=None, offset=None)` 把长轴、目标点和方向转到 root 系，可以沿同一条目标线换一个距离。

### 站位规划：`plan_stance`

本节只说明工具层的行为，IK、全身模型和并行评估的细节见"运动学、全身模型与站位规划"一页。

- `plan_stance(which="wrist_target")` 是外层：规划失败（没有可行站位或条码不可达）时把当前条码记为已尝试；规划成功不算尝试。
- `_plan_stance(which)`：
  1. 没有手腕目标但有当前条码时自动调用 `compute_wrist_target`（演进 v24）。
  2. 网格以条码为基准：沿水平法向 0.35 到 0.70 m、横向 −0.40 到 0.40 m，步长都是 0.05 m，两端都包含（代码中 `arange` 的上界加了 1e-9）；朝向正对面。
  3. 障碍：轮廓闭合时到轮廓的有符号距离 ≥ 0.35；未闭合时到条码所在平面、以及到每个已知有限面（`horizontal_distance_to_face`）的距离 ≥ 0.35。`_known_faces()` 给出最近一次检测到的箱子面加各条码保存的面副本，多面模式下只取当前面的副本。
  4. 需要可达的距离：自适应且宽度来自头相机时 `finals = [min_m]`（0.12），`pre_dists` 从 `offset + 0.10` 按 0.05 递减到 `min_m`；其它情况 `finals = [offset]`，`pre_dists` 从 `offset + 0.10` 递减到 `offset`（演进 v8、v15）。
  5. 身体高度先站立，再依次试标定下蹲档位，第一个有可行站位的档位胜出。行走目标 `pw` 减去 `foot_shift`，保证蹲下后骨盆落在规划点上。
  6. 候选评估交给 `_eval_stance_jobs`（`stance_eval.py`，进程池 `plan_workers = 16`，spawn 启动，每个 worker 的 BLAS/OpenMP 线程数设为 1）。评估期间 `ik.cfg["restarts"]` 临时改为 1，`finally` 中恢复；停滞判据 100 次迭代内误差下降不足 0.5 mm；每个候选的 IK 种子固定为 `[seed, 档位下标, 网格下标]`。
  7. 代价 `cost = (−gap, −man, travel)`：先比可达的预位姿间隙，再比 Yoshikawa 可操作度（所有所需位姿中的最小值），再比行走距离（演进 v11）。
  8. 按代价顺序逐个检查，第一个通过的就选定：面间隙 ≥ 0.03、手臂与腿间隙 ≥ 0.03、鲁棒性 `_stance_robust`（站位偏移 ±0.03 m（x、y）和 ±1.7° 共 6 种情况下所有 finals 和所选 pre 仍 IK 可达）。第一个通过两项间隙的候选记为 `fallback`，只用于在失败信息中标注"只有精确停下才可达"，不会被采用。
  9. 失败：条码标为 `unreachable`，写明标签中心高度、偏低还是偏高、站立或最深下蹲时的骨盆高度和鲁棒容差；返回 `rejected` 计数、`levels_tried`、`exact_stop_only`（演进 v12）。
  10. 成功：写入 `ws.stance_target = {pos, yaw, face_dist, lateral_offset, barcode_id, travel_m, base_height, pelvis_pos, pelvis_z, pre_gap}`，reason 写明"stance planned (the robot has not moved): call approach('stance_target') to walk there"，需要下蹲时再说明会自动蹲到哪个高度（演进 v27）。

`Toolbox._manipulability(q)` 仍定义在 `toolbox.py` 中，但当前没有地方调用它，实际使用的是 `stance_eval.manipulability`。

### 手臂运动：`move_arm`

**`move_arm(pose="pre", side=None)`** 包装 `_move_arm`，并记录 `_last_arm = {pose, barcode_id, status}`，供 `check_wrist_placement` 给提示用（演进 v35）。

**`_move_arm(pose, side)`**

1. `pose == "home"`：设为 `remember_home()` 记下的关节角，耗时 `duration_s = 2.0`，`arm_at_home = True`。
2. 任务 3 有当前条码时：必须已为它规划了站位，并且已经通过 `approach` 到达（`_stance_reached is ws.stance_target`），否则返回 fail，并说明下一步（演进 v27）。
3. `final` 用精化后的目标（如果有），距离为 offset；`pre` 用 `off + pre_gap`。
4. 闭环修正，最多 `1 + correction_iters(2)` 轮：在当前关节状态上解 IK（带 roll 约束，不做停滞检测）→ `set_arm_joints` → `stand(settle_s = 1.0)` → 把目标重新表达到当前 root 系（底座可能漂移）→ 用 FK 计算尖端误差 `e`。`e` 比之前最好的还大时退回最好的命令并停止（演进 v10）；`e < correction_tol (0.01 m)` 时停止；否则满增益修正命令 `p_cmd += err`。修正中途 IK 不可解但已有更好的姿态时退回该姿态；第一次就不可解直接 fail。
5. 残差 ≥ 0.01 m 时用 `_label_view` 判断标签中心是否还在腕相机视锥内：不在视锥内返回 fail（"从此站位不可达"），调用方应重新规划站位；在视锥内返回 partial，reason 写明"this is not a failure: go on"（演进 v9、v22、v30）。

**`_label_view(t, off, label_width=None)`**：在尖端（`wrist_point_frame`）坐标系下以 x 轴为扫描轴，算标签中心沿轴距离 `along` 和偏离轴的横向距离 `lat`，视野半宽 `half = along·tan(hfov/2)`；`in_view = along > 0 且 lat < half`。给了 `label_width` 时还计算 `label_in_view`：`lat + width/2 < half`，即整个标签都在视野内。没有内参时抛 `RuntimeError`。

`remember_home()` 记下当前手臂关节角作为 home；还没记录时 `_home_q` 抛 `RuntimeError`。

### 腕部精化：`refine_with_wrist`、`stereo_check_wrist`、`adopt_stereo`

**`_detect_wrist()`**：拍腕相机图像，先用 `detect_barcodes`（`perception.barcode.wrist_upscale = 1`），检测不到时退回 `detect_label_dino`（全图）；有多个结果时取最靠近图像中心的那个。

**`refine_with_wrist(barcode_id=None)`**：

1. 过 `_not_current` 检查，且必须已有同一条码的手腕目标。
2. 用头相机重新拟合面平面（ROI 为面角点投影的外接框），失败时沿用原来的面。
3. 腕部检测失败时，按 `refine.search_offsets_m = [[0.05,0],[−0.05,0],[0,0.05],[0,−0.05]]` 在面内 (right, up) 方向依次偏移，每次移到 pre 位姿重新检测；仍找不到返回 partial，保留头相机估计。
4. 腕部检测中心的射线与平面求交，得到新中心 `c`。
5. 一致性门限：`c` 与当前估计的距离超过标签宽度时，认为检测到的不是计划中的标签，返回 partial 并保留原估计（演进 v10）。
6. 采纳新中心，`source = "wrist"`；有 `bar_dir` 时用沿条纹方向偏移 10 px 的第二个像素重算条方向。
7. `_wrist_symbol_width` 测符号宽度。解码到内容时记入 `bc.wrist_read`（表示从这个手臂位姿能读出这个码，演进 v33）。宽度必须 > 0，且已有腕部宽度时新测量的轮廓数不能少于原来的支持数 `width_support`，才替换原估计。
8. 调用 `compute_wrist_target(refined=True)`。

**`_wrist_symbol_width(w, rec, plane)`**（演进 v7）：调用 `decode_symbol_polygons` 按 15° 步长旋转解码，排除被图像边缘截断的轮廓，取中心离检测点最近的那个内容的所有轮廓；每个轮廓的像素射线投到面平面，在面内用 `cv2.minAreaRect` 求最小外接矩形取长边；结果取所有轮廓的中位数。注释给出的离线验证：仿真器标签纹理上（code128，宽 0.05–0.10 m，距离 0.20–0.40 m，角度 0–80°，有无白底都包括），中位误差 +0.5%，97 个样本中 96 个误差在 3% 以内。返回 `width_m`、`content`、`n_outlines`、`spread_m`。

**`stereo_check_wrist(barcode_id=None)`**：要求已调用过 `refine_with_wrist`。手腕沿面的 right 方向平移 `stereo.baseline_m = 0.04`，重新检测，用两次视图的射线三角化（`geometry.triangulate`）；与平面求交的中心相差超过 `max_disagreement_m = 0.02` 时返回 partial。

**`adopt_stereo(barcode_id, point)`**：采纳三角化中心，但用与 `refine_with_wrist` 相同的一致性门限，偏离超过一个标签宽度就拒绝（演进 v14）。采纳后 `source = "stereo"` 并重算精化目标。只有规则编排使用。

### 放置检查：`check_wrist_placement`

用 odom·FK 得到枪口位姿，计算四个量：

| 检查项 | 条件（`task2.yaml → check`） |
|---|---|
| 尖端到条码平面的距离与 offset 之差 | ≤ `dist_tol_m`（0.02 m） |
| 条码中心偏离扫描轴的距离 | < `axis_offset_max_m`（0.02 m） |
| 扫描轴与 −法向的夹角 | < `angle_max_deg`（10°） |
| roll 误差（`ik.roll_error_deg`，图像水平轴与长轴的夹角），有长轴时才检查 | 与夹角共用 10° |

全部通过返回 ok。超出时：

- 任务 2：直接 fail（任务 2 的成功判据就是放置合格）。
- 任务 3：这个工具只测量，不决定（演进 v29、v34、v35）：
  - 在 stand-off 距离上（`|dist − offset| ≤ 0.02`）且标签中心在视野内：返回 partial，提示"call scan now"；
  - 不在扫描位姿上，且最近一次手臂命令不是本条码的 `final`：返回 fail，提示先 `move_arm('final')` 再 scan；
  - 已经发过 `final` 但没到：返回 fail，提示重新规划站位（`move_arm('home')`、`compute_wrist_target`、`plan_stance`、`approach`）。

### 扫码：`scan` 与 `_scan_once`

**`_scan_once()`**：调用 `robot.scan()`，状态既不是 `ok` 也不是 `no_read` 时 fail。读到的内容 `content`（最靠近图像中心的那个）按内容记入 `ws.scanned` 账本，只记第一次：新内容把当前条码改为 `scanned`；已经由别的条码读过的，当前条码改为 `duplicate`，返回 ok 并说明"之前读过，不重复计数"。返回 `decoded`（所有解码内容）、`polygons`、`new`、`duplicate_of`、`scanned`、`barcode_status`，有计划时附 `scan_plan`。

**`scan()`**

1. 把当前条码记为已尝试，执行 `_scan_once()`。
2. 没读到且有手腕目标时，最多重试 `scan.retries_on_no_read = 1` 次：`refine_with_wrist` → `move_arm("final")` → `_scan_once`；`move_arm` 返回 fail 时停止（演进 v16）。
3. 仍没读到，但 refine 时腕相机解出过这个标签（`bc.wrist_read`）：`move_arm("pre")` 回到那个距离再扫一次（演进 v33）。
4. 仍没读到时（仅任务 3），以下条件全部满足就把标签标为 `unreadable`，写入 `fail_reason`，与 `failed` 区分（演进 v31、v33、v34）：
   - 条码处于 `pending`；
   - 这次扫描整幅图像里什么都没解码出来；
   - 之前的腕部视图也从没读到过（`wrist_read` 为空）；
   - 整个标签都在扫描视野内（`label_in_view`，宽度用腕部测量值或头相机估计值）；
   - 扫描枪在 stand-off 距离上：沿轴距离与 offset 之差 ≤ `check.dist_tol_m`（0.02 m）。

   标为 `unreadable` 后提示 "move_arm('home'), then call next_barcode"。
5. 每一步重试记在 `data["retries"]`。

### `scan_order.py`：扫描顺序与 stand-off

文件头说明：评测端在 `g1agent/eval/scan_order.py` 有一份同样规则的副本，因为算法代码不能导入评测包；`tests/test_task3_logic.py` 检查两份代码给出的顺序一致。

- `hfov_from_intrinsics(fx, width) = 2·atan(width/2/fx)`（弧度）。腕相机 fx 1087、宽 640 px 时为 32.8°。
- `adaptive_standoff(label_width_m, hfov_rad, fill, d_min, d_max)`：`d = label_width / (2·tan(hfov/2)) / fill`，钳到 `[d_min, d_max]`。含义是在距离 d 处标签宽度恰好占扫描图像宽度的 `fill`。对应需求 5。
- `column_order(items)`：输入 `[{key, y_lo, y_hi, z}]`（y 为规划 root 系下的横向坐标，左为 +y）。
  1. 按 `(−y_hi, −y_lo, key)` 排序，从左往右扫；
  2. 某项的 `y_hi` 大于当前列的最小 `y_lo`（严格重叠，仅相接不算）就加入当前列并更新最小 `y_lo`，否则新开一列。后面的项右端更低，不会回头连到更早的列；
  3. 列内按 `(−z, −横向中心, key)` 排序，保证顺序确定。

  "从左到右、再从上到下"对应需求 4；把它解释成"横向范围重叠的标签属于同一列"是一个设计决定。
- `order_items_in_frame(labels, T_odom_frame)`：把标签角点和中心从 odom 转到参考系（规划时的 root 系），输出 `column_order` 需要的 items。
- `quad_width(corners)`：两组对边平均长度中较大的那个。扫描枪绕自身轴的 roll 是自由的（当前由 IK 对齐到长轴），要放进扫描图像宽度的是标签较长的那一边。
- `point_in_quad(p, quad)`：点是否在凸四边形内（含边界），检查所有叉积同号。

### 模块级几何函数

- `horizontal_distance_to_face(p, face)`：俯视下有限面是沿水平 right 轴、介于两端角点之间的线段；返回点到这条线段的水平距离，点在面的侧面时量到最近的端点。
- `_poly_signed_distance(P, poly)`：点到逆时针凸多边形的有符号距离，内部为负。
- `_same_label(a, b)`：两个像素检测的任一中心落在另一方的四边形内，就判为同一个标签。
- `_same_planned_label`：见上文。

### 任务配置

**`configs/task1.yaml`**（任务 1）

| 键 | 值 | 说明 / 依据 |
|---|---|---|
| `selector_default` | `left` | 只有 `scripts/run_task.py` 在任务 3 中读取这个键（取任务 3 的值 `all`）；任务 1 / 2 用场景给出的 left / right 目标，所以这里的 `left` 不起作用 |
| `d_obs_range` / `d_obs_step` | `[0.60, 1.00]` / `0.05` | 观察距离搜索范围和步长；没有给出依据 |
| `approach.segment_frac` | `0.5` | 每段走 `min(剩余 × 0.5, max_segment)` |
| `approach.max_segment_m` | `1.0` | 没有给出依据 |
| `approach.final_below_m` | `0.3` | 剩余不足 0.3 m 时一步走完 |
| `approach.lost_backoff_m` | `0.3` | 丢失目标时后退再检测 |
| `approach.precise_speed_scale` | `0.5` | 全速短距离最多冲过头约 12 cm |
| `approach.final_bias_m` | `0.0` | 原为 0.05（旧行走控制器），见演进 v1、v6 |
| `approach.move_timeout_s` / `max_segments` | `20.0` / `8` | 没有给出依据 |
| `arrive.dist_tol_m` / `yaw_tol_deg` | `0.03` / `3.0` | Harness 端基于估计值的到达检查 |
| `arrive.max_fine_tunes` | `2` | 没有给出依据 |
| `arrive.fine_tol_frac` | `0.5` | 微调停止容差为判定容差的一半 |
| `too_few_inliers_advance_m` | `1.0` | 规则编排在拟合失败时向目标前进再试 |
| `explore_sequence` | forward 0.5、left 0.5、right 1.0、back 0.5、turn_left 15、turn_right 30 | 规则编排的探索序列；`forward` 放第一步，因为头相机向下约 48°，4–5 m 外箱子在图像上边缘或更上方 |
| `budget.*` | 工具调用 40、探索 6、单工具重试 2、墙钟 1800 s | Harness 守卫使用；没有给出依据 |
| `eval.dist_tol_m` / `yaw_tol_deg` | `0.05` / `5.0` | 只给评测用 |

**`configs/task2.yaml`**（任务 2）

| 键 | 值 | 说明 / 依据 |
|---|---|---|
| `offset_m` | `0.20` | 枪口沿法向到条码的固定距离；任务 3 中称为"已验证的 stand-off" |
| `pre_offset_m` | `0.30` | 预位姿；对任意 stand-off d，预位姿在 `d + 0.10` |
| `standoff.mode` | `fixed` | 任务 3 改为 `adaptive` |
| `head_retry_advance_m` | `0.10` | 规则编排：头相机没看到条码时前进再试 |
| `stance.min_face_dist_m` | `0.35` | root 原点到面平面的最小水平距离，也是网格最近一行；`explore`、`_move_axis`、站位网格都用它；没有数值依据 |
| `stance.max_face_dist_m` | `0.70` | 网格最远一行；依据：m5 阶段任务 2 的 61 个站位（旧网格）选在 0.52–0.63 m |
| `stance.lateral_range` | `[−0.40, 0.40]` | 以条码为参照；同样 61 个站位横向落在 −0.37 到 +0.11 m（旧网格是机器人当前位置 ±0.30 m） |
| `stance.step` | `0.05` | 网格步长，也是 pre 距离递减步长 |
| `refine.search_offsets_m` | `[[0.05,0],[−0.05,0],[0,0.05],[0,−0.05]]` | 腕部检测失败时的面内搜索 |
| `stereo.baseline_m` / `max_disagreement_m` | `0.04` / `0.02` | 运动立体基线和允许差异 |
| `place.max_iters` / `tol_m` | `2` / `0.01` | 规则编排使用，工具箱不引用 |
| `scan.retries_on_no_read` | `1` | 设计决定：一次重试 = refine、move_arm final、scan |
| `check.dist_tol_m` / `axis_offset_max_m` / `angle_max_deg` | `0.02` / `0.02` / `10.0` | Harness 端放置检查，roll 也用 10° |
| `eval.*` | `0.03` / `0.03` / `15.0` | 只给评测用 |

**`configs/task3.yaml`**（任务 3 = 任务 1 + 扫描全部条码）

| 键 | 值 | 说明 / 依据 |
|---|---|---|
| `inherits` | `[task1, task2]` | 继承两者，本文件的键覆盖继承值 |
| `selector_default` | `all` | 阶段 1 观察所有检测到的箱子 |
| `standoff.mode` | `adaptive` | `hfov` 来自腕相机内参；`symbol_width` 指印刷的符号本身（条和模块），不含白色底衬。规划先用头相机估计（标签框，是符号尺寸的上界，更远、更保守），到预位姿后换成腕部实测值 |
| `standoff.fill_frac` | `0.68` | 让标称 0.08 m 的 code128 符号落在已验证的 0.20 m：`0.08 / (2·tan 16.40°) / 0.20 = 0.679`。原值 0.85 针对 0.10 m 标签 |
| `standoff.min_m` / `max_m` | `0.12` / `0.30` | `min_m` 是需求中的决定（设计需求 5）；`max_m` 没有单独依据 |
| `budget.*` | `40 / 6 / 2 / 1800` | 阶段 1 预算与任务 1 相同；`plan_scans` 返回计划后 Harness 按条码数扩展 |
| `budget.per_barcode.max_tool_calls` | `28` | m5 规则模式中任务 2 的最大调用数 21（57 个回合），加 next_barcode 1、scan 1、一次 no_read 重试 3、扫描前后 move_arm home 2 |
| `budget.per_barcode.max_wall_s` | `600` | 同样 57 个回合中任务 2 最大耗时 518 s（p90 324 s），加扫描重试和两次手臂回 home（各约 3 s），向上取整 |
| `eval.dist_tol_m` / `yaw_tol_deg` | `0.05` / `5.0` | 任务 1 阶段的评测容差；每次读码的放置检查用 task2 的 eval |

配置继承（`g1agent/common/config.py`）是浅合并：本文件的顶层键整体替换父配置的同名项。两个父配置对同一个键给出不同值、而本文件没有显式设置时抛 `ValueError`（task1 与 task2 的 `eval` 不同，所以 task3 必须自己设置 `eval`）；检测到继承环也抛 `ValueError`。

### 硬编码常量（没有配置项）

| 位置 | 常量 | 依据 |
|---|---|---|
| `_reselect` | 横向放宽 0.25 × bbox 宽 | 没有 |
| `_footprint` | 判断平行的 \|cos\| > 0.5 | 没有说明；只有平行和垂直两种关系，取值不影响结果 |
| `_reperceive_target` | 最多尝试 2 次 | 没有 |
| `_label_on_face` / `refine_with_wrist` | 求条方向时偏移 10 px | 没有 |
| `_move_axis` | 路径上 11 个采样点 | 没有 |
| `_fit_and_reach_side` | 拟合两遍 | 注释说明含义：先在面的近端，再在观察位姿 |
| `_labels_on_rectified_face` | 深度下限 1e-3 | 数值保护 |
| 多处 | 1e-6、1e-9、1e-3、1e-12 | 数值容差 |
| `_eval_stance_jobs` | 每个 worker 的 BLAS/OpenMP 线程数 1 | 有，见演进 v18 |
| `refine_with_wrist` 的 partial 提示 | 文字 "searched +-5 cm" | 与 `search_offsets_m` 数值一致，但文字写死 |

### 设计原则在工具箱中的体现

- **前置条件做进工具，不依赖 LLM 遵守调用顺序**：行走前自动站起、到站位后自动下蹲、缺手腕目标时自动补算、没到站位不许抬手、只接受当前条码、未尝试的条码不许跳过、读不出时工具内部重试、每扫完一个码自动补看。
- **数值由工具决定**：自适应模式下忽略调用方给的 offset；扫描顺序由 `scan_order.py` 决定，LLM 不能改。
- **不引入无依据的阈值**：配置里的阈值大多在注释中写明来源（实测分布、控制器容差、标定、需求决定）；`_face_id` 和 `_same_planned_label` 特意用几何事实代替阈值。没有依据的常量在上面的表中如实标出。
- **不用宽泛的 try/except**：工具箱里只有 `try/finally` 用于恢复状态（`_plan_stance` 恢复 `ik.cfg["restarts"]`、`go_to_next_face` 恢复 `current`、恢复线程数环境变量）。错误直接抛出：缺宽度或内参抛 `RuntimeError`，未知 mode 抛 `ValueError`，站不起来抛 `RuntimeError`，reason 类型错误抛 `TypeError`。
- **修 bug 找根因**：下面的演进记录中，大多数修改从运行中观察到的现象出发，改在工具语义层面（先决条件、一致性门限、冻结、重配准、状态区分），不针对单个场景调参。

## 演进记录

日期取自开发日志。文中 af4c_2side_vN 之类是运行目录名（纸箱两面场景的第 N 次运行），与本页的版本号 vN 无关。代码注释里个别事件的日期比日志早一天（例如手臂修正发散、矫正面检测），这里以日志为准。版本号按时间统一编号；下表按工具列出相关版本，方便按工具查阅。

| 工具 / 部件 | 相关版本 |
|---|---|
| 返回格式、`_check_reason` | v0、v3 |
| `approach`、`arrive_check`、`compute_obs_pose`、`explore` | v0、v1、v5、v6 |
| stand-off（`_standoff`、`fill_frac`、腕部测宽） | v4、v7、v15 |
| `plan_scans`、矫正面检测、合并与去重 | v5、v8、v9、v17、v19 |
| 下蹲补看 `_low_view_update` | v20、v32 |
| `next_barcode` | v5、v24、v35、v36 |
| `plan_stance`（工具层） | v5、v8、v11、v12、v18、v24、v27 |
| `move_arm` | v9、v10、v22、v27、v30、v35 |
| `refine_with_wrist`、`adopt_stereo` | v10、v14、v33 |
| `compute_wrist_target`、`_long_axis` | v13、v21、v24 |
| `_walk_guard` | v13、v27 |
| `check_wrist_placement` | v21、v29、v34、v35 |
| `scan`（重试、`unreadable`） | v2、v5、v16、v31、v33、v34 |
| 多面：箱堆模型、`go_to_next_face` | v23、v25、v26、v28 |
| 条码工具只作用于当前码 `_not_current` | v36 |

### v0 · 2026-11-05 · 工具箱初版：设计文档 §3.3 的全部确定性工具

**初版设计**。`g1agent/tools/toolbox.py` 实现了设计文档列出的工具：`detect_boxes`、`select_target_box`、`fit_front_face`、`compute_obs_pose`、`approach`、`explore`、`check_face_in_view`、`look_head`、`detect_barcode_head`、`estimate_barcode_pose`、`compute_wrist_target`、`refine_with_wrist`、`stereo_check_wrist`、`move_arm`、`check_wrist_placement` 等（`finish` 在 Harness）。约定从一开始就定下：

- 所有工具返回 `{status: ok|fail|partial, data, reason}`，数值保留 3 位小数，不返回大数组，图像只通过 `look_*` 返回；
- 感知在拍摄时刻的 root 系完成，结果存进 odom 系的 `WorldState`；头/腕相机位姿由关节状态经 pinocchio FK 得到（设计文档 §4.1 规定头腕之间用运动学换算，不用里程计）；
- 模块内不使用真值，由 `tests/test_no_gt_leak.py` 检查；
- `approach` 按 `task1.yaml` 分段行走，每段之后重新检测。

### v1 · 2026-11-05 · `approach` 加 `final_bias_m = 0.05`：先停在目标前方，再向前微调

**当时的版本**。`approach` 直接瞄准观察位姿。

**问题**。当时的行走策略（policy1）最多冲过头约 8 cm，而且往回走基本无效（vx = −0.3 m/s 走 2 s 只后退约 6 cm），小幅前进修正则有效。

**根因**。控制器不能可靠后退，冲过头之后无法纠正。

**修改**。`task1.yaml` 的 `approach` 新增 `final_bias_m: 0.05`，`_approach` 沿朝向把目标设在真实目标前 5 cm，最后用前进微调走完。仿真端 `move_relative` 同时增加 `pos_tol` / `yaw_tol` 参数，允许单次调用覆盖默认容差。

**效果**。t1_L0_0006 从失败变为成功：17 次调用、211 s，距离误差 0.003 m，偏航 0.415°，感知误差 0.019 m，执行误差 0.005 m。t1_L0_0003 被 Harness 判失败（容差 3 cm），评测器判成功（距离误差 0.03 m）。这个偏置在换用 GR00T 控制器后取消（v6）。

### v2 · 2026-11-05 · `scan` 初版：触发扫描器，取最靠近画面中心的码

**初版设计**。`scan` 调用 `robot.scan()`。仿真端扫描器对枪头相机画面做全向解码：图像旋转 0/15/…/75° 后分别用 pyzbar 解码（15° 步长的依据是实测 zbar 角度容差约 ±17°），返回离画面中心最近的码，另附只供评测的真值诊断。枪头相机按 Zebra DS2208 类成像器取值：640×480，视场 32.8°×24.8°。

**实测**。场景 1 中枪口离标签 0.16–0.40 m 都能扫出，自检 2/2 通过；场景 3 中层 0.67 m 能扫，顶层 1.1 m 够不到且容易摔倒，底层 0.22 m 因当时不能下蹲够不到。后来仿真端扫描器改得更接近真实设备，要求成像与条码对齐（见 v21）。

### v3 · 2026-11-06 · 修复 5 处 ok/fail 参数顺序颠倒，新增 `_check_reason`

**当时的版本**。`ok(data, reason)` 与 `fail/partial(reason, data)` 参数顺序相反，有 5 处写成了 `(ok if cond else fail)(data, reason)` 这种统一调用：`compute_obs_pose`（ok/partial）、`check_face_in_view`、`explore`、`check_wrist_placement`、`scan`。

**问题**。走到 fail 分支时，数据字典被放进了 `reason` 字段。LLM 是读失败原因做决策的，这会直接误导 LLM 编排。

**根因**。两个构造函数参数顺序不一致，类型上又没有检查，错误被静默写入。

**修改**。新增 `_check_reason()`：reason 不是 `str` 就立即抛 `TypeError`；5 处调用全部改成显式分支，例如 `scan` 改为 `ok(data) if ... else fail("no code read", data)`。

**效果**。grep 统一调用写法剩余 0 处；当时 11 个单元测试全部通过。

### v4 · 2026-11-06 · 扫码距离改为按标签宽度自适应（决定）

**当时的版本**。扫码距离固定为 `offset_m = 0.20`。

**分析**。腕相机水平视场 32.8°（fx 1087，宽 640），距离 d 处画面宽 0.59·d，0.20 m 时为 11.8 cm。标称标签（code128 码条 8 cm，含白边 10 cm）至少要 d ≥ 0.17 m 才能完整入画；实测 0.12 m 失败，0.16–0.40 m 成功。一个模块约 0.48 mm，0.20 m 时约 2.6 像素，0.40 m 时约 1.3 像素，接近解码极限。对标称标签 0.20 m 合理，但标签尺寸不同时固定距离不合适。

**决定**。需求方同意按标签宽度自适应：`d = 标签宽度 / (2·tan(hfov/2)) / 0.85`，钳到 `[0.12, 0.30]` m；0.85 让标称 10 cm 标签正好落在 0.20 m；预位姿为 d + 0.10 m（由原配置 0.30 − 0.20 推出）。0.85 和范围是当时新引入的固定参数。按此公式，5 cm 标签约 0.12 m，12 cm 标签约 0.24 m。评测器的真值目标点用同一公式、代入真实标签尺寸计算。任务 2 仍保持 0.20 m。实现落在 v5 的 `scan_order.adaptive_standoff` 中。

### v5 · 2026-11-06 · 任务 3 工具初版：`plan_scans`、`next_barcode`、扫码账本；到位判定去掉横向项；站位网格以条码为参照

**此前遇到的问题**。

- t1_L1_box_shape（目标箱从 0.40×0.30×0.30 m 均匀放大到 0.60×0.50×0.50 m）在规则批量评测中，因子 ≥ 0.636 的 5 个场景全部失败（"not within tolerance after the fine moves"），0–0.545 的 7 个成功，即 7/12。以 0007（因子 0.636）为例，感知误差 2 cm，最终到位误差 38.7 cm；0008（因子 0.727）最终到位误差 36.8 cm，感知误差也有 40.8 cm。日志显示每段移动都报 ok，但重新估计的目标在两段之间跳了约 0.4 m，机器人最后停在离箱面约 0.99 m 处（目标 0.6 m）。当时推测根因在到位判定和近距离重估计；GR00T 能后退之后这个问题才暴露出来。
- 最简单的 L0 场景里，第一次 `approach` 的误差为 [0.009, 0.037, 0.229]（沿法向、横向、yaw），横向 3.7 cm 超过 3 cm 容差，需要 LLM 介入。

**修改**。

- 新增 `g1agent/tools/scan_order.py`（纯函数）：`adaptive_standoff`、`column_order`（横向范围严格重叠归为同一列，列从左到右，列内从上到下）、`quad_width` 等。
- `plan_scans`：`detect_boxes`、拟合所有箱面，保留中心落在箱面内的全部 DINO 标签，在平面上估计中心、角点、宽度；两个检测中任一中心落在另一个的四边形内就合并；按列规则排序，编号 bc0..bcN；计划一旦开始执行就不允许重新规划。
- `next_barcode`：把未完成的当前条码标为 `failed`，清空该条码相关的槽位，取下一个待扫码。
- `scan`：只有 `content` 计入账本，之前读到过的内容记为 `duplicate`，不计数。
- `compute_wrist_target` 默认使用自适应距离；`select_target_box` 支持 `"all"`。
- 到位判定改为：法向误差 < 0.03 m、yaw < 3°、箱面都在视野内，去掉横向项（与评测器的任务 1 成功定义一致）；只有箱面出画时微调才修横向。
- 每段移动的日志经 `move_summary` 白名单过滤，去掉仿真回传的真值字段。
- `plan_stance` 改为以条码为参照的网格：离面 0.35–0.70 m，横向 −0.40 到 0.40 m，步长 0.05 m（依据：m5 任务 2 中 61 个实际站位落在 0.52–0.63 m、横向 −0.37 到 +0.11 m）；代价先看关节限位裕量，再看行走距离。

**效果**。开发日志里没有针对这一版到位判定的单独统计。随后在 t1_L0_0000 上的任务 3 首轮运行中，规则编排扫出 2/2、顺序正确（LLM 编排那一轮撞箱，修复见 v6）。

### v6 · 2026-11-06 · 冻结观察面、近距重拟合只修正平面位置；`explore` 防撞；任务 3 到位失败返回 partial；`final_bias_m` 归零

**当时的版本**。`_reperceive_target` 每段之后用重拟合结果重新算观察位姿；`explore` 没有距离约束；`arrive_check` 不满足时一律 fail。

**问题**。

- LLM 编排第一次跑任务 3 时，LLM 在离箱 0.67 m 处向前探索 0.5 m，躯干撞上箱子。
- 近距离重拟合时面被图像边缘截断，测出的尺寸不再是整个箱子的尺寸，目标点在精移动之间移动约 0.15 m，到了规划位姿后视野检查仍不通过（v5 中大箱子失败的同一机理）。

**根因**。观察位姿的计算依赖面的完整范围，而近距离看到的只是一部分；探索移动的安全完全依赖 LLM 的判断。

**修改**（`toolbox.py`）。

1. `__init__` 新增 `self._obs_faces`；`compute_obs_pose` 在远处规划时把各箱面的 center / normal / corners 冻结到 `_obs_faces`；`check_face_in_view` 对同一 `box_id` 使用冻结的全尺寸面。
2. `_reperceive_target` 改用 `_retarget_from_refit()`：近距重拟合只取匹配箱面沿法向的平均位移和新法向，保留 `d_obs` 和原横向位置，`p = s_old·lat + (a + d)·nh`。
3. `explore` 拒绝终点距任一已拟合箱面小于 `stance.min_face_dist_m = 0.35` m 的平移（沿用已有的站位最小距离，不是新值）。
4. 任务 3 中 `approach` 走完但到位检查不在容差内时返回 partial："walk finished; arrival check not within tolerance. Phase 1 is over: call plan_scans next"。
5. GR00T 能停在约 1 cm 以内并能后退，`final_bias_m` 从 0.05 改为 0.0。

**效果**。62 个单元测试全部通过；LLM 编排重跑 2/2 成功。

### v7 · 2026-11-07 · 预位姿用腕相机实测符号宽度；`fill_frac` 0.85 → 0.68

**当时的版本**。最终扫码距离完全由头相机估计的标签宽度算出。

**问题**。t1_L0_0000 中两个码的头部估计宽度都是 0.12 m（实际 0.10 m），距离算成 0.24 m 而不是 0.20 m，枪尖偏远约 4 cm。

**根因**。头相机看到的标签框包含白色底衬且偏松，宽度只会偏大约 20%；另外扫描枪需要容纳和分辨的是条码符号本身，而测试集里有没有底衬的码。

**修改**。

1. `refine_with_wrist` 在预位姿调用新的 `_wrist_symbol_width`：用 `decode_symbol_polygons` 解码最靠近图像中心的码，把轮廓投到箱面得到米制宽度，多角度取中位数；更新 `BarcodeEstimate` 的宽度和新字段 `width_source`（head / wrist），重算 d。解不出时保留头部估计并在日志中写明原因。
2. 测量对象从"白色标签宽度"改为"条码符号宽度"；`task3.yaml` 公式改为 `d = symbol_width/(2·tan(hfov/2))/fill_frac`，`fill_frac` 由 0.85 改为 0.68，使标称 8 cm 符号仍落在已验证的 0.20 m。
3. 头部估计只用于保守地规划站位和预位姿；`plan_stance` 在 `width_source == "head"` 时额外检查 0.12 m 处 IK 可达，保证腕部测量后拉近仍够得着。

**效果**。t1_L0_0000 枪尖误差约 4 cm → 平均 0.7 cm（最大 1.0 cm）。代价：同一场景工具调用 32 → 36 次，仿真时间 72.2 → 109.0 s（按工具累计的墙钟耗时：`move_arm` 153 → 229 s，`approach` 124 → 154 s，`plan_stance` 6 → 11 s），原因是预位姿从 0.34 m 变成 0.40 m，手臂多做一轮修正，到最终位后中心又移动 1 cm 以上。

### v8 · 2026-11-07 · 堆叠场景第一轮：矫正面检测标签；`plan_stance` 只要求最近距离可达

**当时的版本**。`plan_scans` 在头相机原图的面 ROI 上（放大后）跑标签检测；`plan_stance` 要求头部估计的 stand-off 和对应预位姿都可达。

**问题**（三箱落地堆叠，规则编排第一轮只扫到顶层 1/3）。

- 中层和底层标签没检测到。头相机从约 1.2 m 高向下约 47° 看，下方标签离得远，在原图中被压成约 20×5 px 的细条。
- 底层码按头部估计的 stand-off（0.30 m）在任何下蹲高度都不可达，而按腕部实测的 0.20 m 是可达的。
- 同轮还发现面拟合把地面算进来（宽 1.3 m 而不是 0.40 m）、手臂几乎伸直差 9 cm、手臂可能穿过腿；前者在感知页记录，后两者见 v9、v11。

**根因**。原图里斜看的远处区域采样太粗，检测器看不到；头部宽度是符号宽度的上界，用它算出的距离可能比真实需要的远很多，不能用来否决站位。

**修改**。

1. 新增 `_labels_on_rectified_face`：用拟合平面和相机几何把面矫正成正视图，采样密度取面上最细的源图采样（`fx / 最近角点深度`），不丢信息，再在整个矫正面上跑同样的检测。
2. `_plan_stance`：宽度来自头相机时 `finals = [min_m]`，`pre_dists` 从 `offset + gap0` 递减到 `min_m`；头部 stand-off 不可达时不否决站位。深蹲时首选 pre gap 常常不可达，允许按网格步长缩小 gap。

**效果**。离线在观察位姿视图上：标签检出 1/3 → 3/3，位置误差约 1 cm 以内，顺序从上到下正确。第三轮运行中三个标签都从观察位姿找到。

### v9 · 2026-11-07 · `move_arm` 不收敛时报 partial；`plan_scans` 支持执行中更新

**当时的版本**。`move_arm` 闭环修正结束后一律报 ok；扫描计划开始执行后再调 `plan_scans` 返回 fail。

**问题**。

- 第一轮堆叠运行中抬手几乎伸直，最终离目标 9 cm，`move_arm` 仍报 ok。
- 中层/底层标签在观察位姿漏检时，没有途径把它补进计划。

**根因**。`move_arm` 不报告残差，掩盖了未到位；计划被设计成一次性的。

**修改**。

1. `move_arm` 最后一次误差 ≥ `correction_tol` 时返回 partial，附残差（m / °）和修正次数；`rule.py` 中预位姿只在 fail 时中止，partial 继续（腕相机仍可观察）。
2. `plan_scans` 在计划执行中再次调用时进入更新模式：已在计划中的码（同位置）不变，新看到的码按同一排序规则插入待扫码中，已扫码不受影响，结果带 `updated` / `added`。规则编排在每扫完一个码、仍蹲着的时候调用一次（蹲着时头部离下方标签更近、俯角更小）。`guards.py` 中带 `updated` 的 `plan_scans` 只扩展预算，不再记一次阶段 1 结束。`set_base_height` 结果同时报告实际骨盆高度。

**效果**。70 个测试通过。第三轮运行中，顶层码第一次尝试失败后，扫完后的"再看一次"重新检测到它，第二次读出。

### v10 · 2026-11-07 · `move_arm` 修正变差就回退；`refine_with_wrist` 一致性门限

**当时的版本**。`move_arm` 对指令目标做满增益闭环修正 `p_cmd += err`，不检查误差是否在减小；`refine_with_wrist` 无条件采纳腕部检测的新中心。

**问题**。第三轮堆叠运行中，顶层码（1.10 m，举手）最终位修正越修越偏：3.5 → 18 → 36 cm（残差 0.358 m / 35.5°）。腕相机此时指向错误位置，把码的估计位置从 1.10 m 改到 1.70 m，污染了后续计划。

**根因**。GR00T 用腰部平衡抬起的手臂，底座和躯干随手臂一起动，满增益修正可能发散；腕相机在偏得很远时会检测到别的东西。

**修改**。

1. `_move_arm`：某次修正使误差变大时，退回误差最小的那次指令并停止修正。
2. `refine_with_wrist`：新中心离原估计超过一个标签宽度时不采用，返回 partial，保留原估计。

**效果**。70 个测试通过；这些修复在第 3 轮运行中的客户端未生效，从下一轮开始生效。

### v11 · 2026-11-07 · `plan_stance` 代价改为可操作度

**当时的版本**。v5 中代价先看关节限位裕量；第一轮堆叠后改为"手臂伸展最少优先"。

**问题**。

- 按关节裕量选出 0.70 m 处的站位，手臂几乎伸直，始终差 9 cm。
- 改成伸展最少后，顶层码站位只离箱 0.45 m，手肘抬到头侧、小臂折回脸前，几个关节贴着限位（余量 0），动作不自然；这也很可能是 v10 中修正发散到 36 cm 的原因（推测：关节贴限位，躯干再一晃修正就跟不上）。

**根因**。两种代价都只惩罚一端的奇异（伸直或折叠），另一端反而被偏好。

**修改**。代价改为 `(−gap, −man, travel)`，`man` 是枪口位置雅可比在 7 个手臂关节上的 Yoshikawa 可操作度 `sqrt(det(Jp·Jpᵀ))`，在所有所需位姿中取最小值。可操作度在伸直和折叠时都低，中间的自然姿态最高。

**效果**。离线重规划：1.10 m 标签选出站立、离面 0.65 m、横向 −0.3 m 的站位，手臂伸展比 0.86、可操作度 0.038；0.57 m 标签选出下蹲 0.65 档、离面 0.5 m，伸展比 0.82、可操作度 0.054。

### v12 · 2026-11-07 · 不可达条码：鲁棒性检查与 `unreachable` 状态

**当时的版本**。`plan_stance` 只要找到 IK 可行的站位就采用。

**问题**。

- 有一个站位贴着关节限位边界，实际走过去再下蹲后，差 3 mm 就不可达。
- 底层码（标签中心 z = 0.19 m）只有"精确停下"才可达的站位，试了两次，两次都在下蹲后失败。重规划结果完全相同（规划是确定性的、机器人几乎在原地）。

**根因**。行走有 ±0.03 m、±1.7° 的停止容差，规划没有考虑这部分误差。

**修改**。

1. 新增 `_stance_robust`：站位沿 ±x、±y 偏移 0.03 m、绕骨盆偏转 ±1.7°（等于仿真 `move_relative` 的容差），所有所需距离仍需 IK 可达。
2. `world_state` 新增状态 `unreachable` 和字段 `fail_reason`（随计划导出）。所有实测下蹲档位都没有鲁棒站位时，`plan_stance` 返回 unreachable 并给出原因，例如 "too low: label centre at z=0.19 m; no stance even at the deepest measured squat (height command 0.25, pelvis z 0.25 m) keeps the pre-pose and the final scan pose reachable within the walk tolerance (±0.03 m, ±1.7 deg) (reachable only with an exact stop)"，也区分 too high。原因写入 timeline 的 `barcode_end`、`result.json` 的 `reported_unreachable` 和计划的 `fail_reason`。LLM 编排的提示词增加：报告不可达时不要重试，调用 `next_barcode`。

**效果**。新增 UnreachableTB 测试，71 个通过。之后这类码不再反复硬试，扫描继续进行下一个。

### v13 · 2026-11-07 · 下蹲时行走自动先站起；自适应模式拒绝 LLM 传入的 offset

**当时的版本**。`_walk_guard` 在下蹲时直接拒绝行走，要求 LLM 先调用 `set_base_height`；`compute_wrist_target` 接受任意 `offset`。

**问题**。LLM 编排自己编造扫码距离（在 0.14 m 处扫码），并且搞错下蹲和行走的先后：行走前先蹲下，到了站位却没再蹲。

**根因**。数值和前置条件交给了 LLM。

**修改**。

1. `_walk_guard`：下蹲且手臂在 home 时自动站起；手臂举着时拒绝并提示先 `move_arm('home')`，不做任何移动。`approach('stance_target')` 到达后自动蹲到规划高度。
2. 自适应模式下 `compute_wrist_target` 收到 `offset` 时返回 fail："the stand-off is adaptive in this task: call compute_wrist_target without an offset"（v24 又改为忽略）。

**效果**。测试改为 `test_walking_while_squatting_stands_up_first_or_refuses_with_the_arm_raised`，71 个通过。

### v14 · 2026-11-07 · `adopt_stereo` 增加一致性门限

**当时的版本**。`adopt_stereo` 无条件采纳运动立体三角化的中心。

**问题**。规则编排第 4 轮的 0000 场景中，举手状态下三角化与平面交点相差 0.10、0.20、1.6 m 的结果都被采纳，手腕目标被完全破坏。

**根因**。举手时腕相机两次视图的基线和检测都可能不可靠，三角化会给出离谱的点。

**修改**。与 `refine_with_wrist` 相同：三角化点离当前估计超过一个标签宽度时判定三角化失败，保留原估计。这条路径只在规则编排中使用。

**效果**。71 个测试通过。

### v15 · 2026-11-07 · 头相机宽度只能把扫码距离拉远，不能拉近

**当时的版本**。`_standoff` 对头相机宽度和腕部宽度一视同仁。

**问题**。LLM 编排第二次运行中，中层等码失败：头相机在矫正面上检测出的标签框不完整，8 cm 的码估成 3.5 / 4.9 cm，自适应距离被钳到最小 0.12 m，太近读不出。

**根因**。头相机标签框既可能偏大（DINO 框约 +20%），也可能偏小（部分框），不能作为"可以更近"的依据。

**修改**。`_standoff` 中宽度来源不是 wrist 时取 `max(d, offset_m)`（`task2.yaml` 的 0.20 m）；只有腕部解码实测的宽度能让扫描枪更靠近。

**效果**。71 个测试通过。

### v16 · 2026-11-07 · `scan` 在工具内部重试

**当时的版本**。读不出后的重试（`refine_with_wrist` → `move_arm final` → `scan`）写在 `rule.py` 和提示词里。

**问题**。差 2 cm 没读到时，LLM 编排跳过了提示词里的重试，规则模式则做了重试。

**根因**。重试依赖调用方遵守流程。

**修改**。`scan` 在有手腕目标且读不出时，在工具内部最多重试 `scan.retries_on_no_read` 次（`task2.yaml` 中为 1，设计决定），每步记录到 `data["retries"]`；`rule.py` 中的重试循环删除，只调用一次 `scan`；`registry.py` 中 `scan` 的描述同步更新。为防止一直读不出时卡住，共有三重上限：工具内重试 1 次、Harness 的 `max_retries_per_tool = 2`、每码预算。

**效果**。测试改为每码一次 `scan` 调用、重试在工具内发生，71 个通过。

### v17 · 2026-11-07 · 补扫时用条码首次规划位置去重（`plan_center`）

**当时的版本**。计划更新时，新检测只和条码的当前估计比较。

**问题**。复查可能循环的路径时发现：如果某码的估计位置在尝试中被改坏（v10 中顶层码被改到 1.70 m 后作为 bc3 加回计划），同一个码会被当成新码再次加入，理论上可能一直循环到全局预算耗尽。

**根因**。去重的参照（当前估计）本身可能被错误的精化移动。

**修改**。`BarcodeEstimate` 新增 `plan_center`，记录首次进入计划时的头部位置，之后的精化不改动它；补扫时新检测只要与当前估计或这个原始位置之一吻合，就判为同一个码。

**效果**。每个实际的码在计划中最多一项，补扫一定会结束；有单测覆盖，71 个通过。

### v18 · 2026-11-07 · `plan_stance` 提速

**当时的版本**。站位候选串行评估；失败的 IK 求解跑满 300 次迭代；对每个 IK 可行候选都做手臂与腿的碰撞检查。

**问题**。加上 roll 约束后，`plan_stance` 对每个需要下蹲的条码要花 35–112 s。

**根因**（先测时间分布再改）。705 次失败的 IK 求解占 IK 工作量的 95%；手臂与腿检查占一次 66 s `plan_stance` 的 72%；各网格点之间的 IK 互相独立却串行执行。

**修改**（细节见运动学页）。

1. 单候选评估抽成纯函数 `stance_eval.eval_candidate`，用 spawn 进程池并行（`plan_workers = 16`），每个候选的 IK 随机数种子固定为 (seed, 档位, 网格下标)，串行和并行结果相同。
2. 只在站位规划中加停滞检测：100 次迭代内误差下降不足 0.5 mm 就放弃。
3. 按代价顺序做昂贵检查，第一个通过的就选定，结果与全部检查后选最便宜的相同。
4. 启动进程池时临时把 `OMP_NUM_THREADS`、`OPENBLAS_NUM_THREADS`、`MKL_NUM_THREADS` 设为 1：库默认值下每个 worker 起几十个线程，在负载很高的共享主机上让整个运行（包括仿真器）慢了好几倍。

**效果**。3 个堆叠条码（带 roll 约束，选出的站位与串行一致）：串行 10 / 30 / 104 s，8 进程 5 / 12 / 53 s，16 进程 5 / 10 / 47 s，32 进程 5 / 9 / 50 s，已饱和。停滞检测使失败求解的中位数在第 118 次迭代停下，205 次成功求解中误伤 1 次（0.5%）。

### v19 · 2026-11-07 · 同码判断改为"标签矩形在箱面上是否重叠"

**当时的版本**（v17 之后）。`_same_planned_label` 判定"两中心相距小于半个标签宽度"（约 3–4 cm）。

**问题**。3×3 箱堆（stack9）LLM 编排第一轮补扫时，把中间码重复加进了计划：新加入的 bc3 在 (−0.03, 0.57)，原来的 bc1 在 (0.02, 0.57)。机器人再去扫一次，读出的内容被记为重复（"'M5C1HPV7C' was read before: duplicate, not counted again"）。计数没有出错，但浪费了一轮走位和扫码。

**根因**。从两个视角估出的位置可以差 5 cm 左右，大于半个标签宽度的阈值。

**修改**。改成"同一表面上（沿法向距离 ≤ 较大的宽度）且面内矩形严格重叠"，同时与规划时的矩形和平移到当前中心后的矩形比较。依据是物理事实：两个真实标签不可能互相重叠，所以不需要人为距离阈值。

**效果**。新增 `test_same_planned_label_by_rectangle_overlap`：同一标签两次估计相距 5 cm 判为同一码，旁边矩形不重叠的标签、深一个标签宽的另一表面上的标签都不算同一码；72 个测试通过。stack9 第二轮补扫没有再产生重复条目（重复 0 次）。

### v20 · 2026-11-07 · 首次规划后自动下蹲补看（初版：只看一个最佳档位）

**当时的版本**。只在扫完码后、仍蹲着时补看（v9），观察位姿只有站立视角。

**问题**。stack9 规则运行中，左下角标签在观察位姿漏检，扫完中间码后补看才找到（bc4，高 0.30 m）。它在最左列，本该紧跟左上码，却在中间码之后才被扫，整体顺序因此偏离。

**根因**。站立时头相机看箱面下部距离远、角度斜。

**修改**。`plan_scans` 首次规划后调用 `_low_view_update()`：用下蹲标定和头相机几何，对每个档位预测双脚不动时的头相机位姿，选"能看到所有底角且底部每米像素数最高"的一档，蹲下补看一次，合并新标签后站起。高度由标定和几何算出，不是手选值。

**效果**。stack9 第二轮：规则编排站立看到 4 个码，下蹲补看找到漏掉的 bc4，5 个码从一开始就在计划里，顺序为左上 → 左下 → 中间 → 右上 → 右下；LLM 编排站立看到 3 个，下蹲补到 bc3、bc4，计划与规则编排相同。

### v21 · 2026-11-07 · 约束扫描枪 roll，放置检查增加 roll 项

**当时的版本**。IK 只约束位置和扫描方向，绕扫描轴的 roll 自由；`check_wrist_placement` 不检查 roll。

**问题**。需求评审指出：真实扫码枪读取条纹与图像轴倾斜的码效果很差。仿真扫描器同时改得更接近真实设备。

**修改**。

1. 新增 `_long_axis(bc)`，手腕目标携带 `long_axis`，IK 约束图像水平轴与条码长轴平行（正负号取离当前姿态近的一个）。
2. `check_wrist_placement` 新增 roll 检查：图像水平轴与长轴的夹角 < 10°，沿用已有的扫描轴角度容差，不另引入数值；返回 `roll_deg`。

**效果**。stack9 第二轮所有放置的 roll 偏差在 0.7–6.0° 之间；LLM 编排第一个码 roll 偏差 0.9°，第一次就读出。

### v22 · 2026-11-07 · `move_arm` 的 partial 明确写成"不是失败，继续"

**当时的版本**。partial 的 reason 是 "arm did not reach the target: residual X m / Y deg after N correction(s)"。

**问题**。LLM 编排把这种 partial 当作失败处理。

**修改**。reason 改为 "arm reached the target with a residual ...; this is not a failure: go on (pre: refine_with_wrist; final: check_wrist_placement, then scan, which retries by itself)"，提示词同步加上"partial 不是失败"。

**效果**。72 个测试通过。重跑中 LLM 编排在第 15 步遇到残差 0.052 m / 2.4° 的 partial 后继续执行，3×3 场景最终 5/5，顺序正确、无重复，用时 912 s。

### v23 · 2026-11-07 · 多面扫描初版：箱堆模型、按面计划、`face_done` 与 `go_to_next_face`

**初版设计**。

- `Toolbox` 新增 `multi_face`（默认 `False`，保持单面行为）。
- 箱堆模型 `ws.stack`：第一次规划时把最正对机器人的面定为正面，记下法向 `n0`，另三面方向由 `n0` 旋转 90° / 180° / 270° 得到，拟合出的面按最近方向归类（`TOUR`、`_face_dirs`、`_face_id`、`_stack_init`、`_stack_add_faces`）。
- `plan_scans` 只为当前面上的码排顺序，计划条目带 `face_id`，每面单独记录 `scan_ref_Ts`。
- `next_barcode` 只取当前面的待扫码，当前面扫完返回 `face_done=true` 并告知下一面，全部扫完返回 `done`；LLM 不能自定顺序或跳面。
- `go_to_next_face`：后退、横移过角、原地转 90°、拟合新面、走到新面的观察位，只用单轴直线移动（`_move_axis`）。
- 避碰改用箱堆占地凸包（`_footprint`、`_poly_signed_distance`、`_root_clearance`），站位和身体间隙按到整个箱堆的有符号距离判断，不再只看无限延伸的正面平面；`BodyModel.face_clearance` 增加 `footprint` 参数。
- 规则编排增加面循环（`face_done` → `go_to_next_face` → `plan_scans`）。

**效果**。新增 `test_face_ids_footprint_and_signed_distance` 等，共 83 个测试通过。

### v24 · 2026-11-07 · 忽略（而非拒绝）LLM 给的 offset；`plan_stance` 自动补算手腕目标；`next_barcode` 拒绝跳过未尝试的码

**当时的版本**。v13 中自适应模式拒绝传入的 offset；`plan_stance` 没有手腕目标时返回 "no wrist target"；`next_barcode` 可以随时跳过当前码。

**问题**。offset 被拒后，LLM 不建手腕目标就直接 `plan_stance`，失败后放弃了每个码。另一次运行中，一次工具调用被拒绝后，LLM 连续跳过了 3 个条码。

**根因**。拒绝会把 LLM 推向"跳过"；跳过本身没有前提条件。

**修改**。

1. `compute_wrist_target` 在自适应模式下忽略调用方的 offset，在 `offset_ignored` 中注明，不报错。
2. `_plan_stance` 发现没有手腕目标时自动调用 `compute_wrist_target`。
3. `next_barcode` 拒绝跳过还没尝试过的码：用 `_stance_attempted` 记录，调用过 `plan_stance`（无论成败）或 `scan` 才算尝试过（v27 收紧）。

**效果**。新增 `test_next_barcode_refuses_to_skip_an_unattempted_barcode_and_offsets_are_ignored`，95 个测试通过。

### v25 · 2026-11-07 · `go_to_next_face` 转身后沿新面再横移；失败时恢复当前面

**当时的版本**。拐角点设在离两面各 0.6 m 处，机器人在这里转身面朝右侧面后直接拟合；工具一开始就把 `st["current"]` 切换成下一个面。

**问题**。第一次绕到侧面就失败，随后在第 40–73 步之间循环：`plan_scans` 失败（no barcode label localised）→ `next_barcode` 认为右侧面已处理完、提示去背面 → `go_to_next_face` 报 "the right side was never fitted"。

**根因**。

- 设计问题：在拐角点转身后，右侧面整个在右前方，从 0.6 m 处延伸到 1.4 m；头相机水平视场半角约 52°，在 0.6 m 前方距离上只能看到右侧 0.78 m 以内，只斜着看到最近的十几厘米（代码注释记为约 0.2 m），拟合不出面。
- 状态 bug：切换 `current` 后失败没有改回。

**修改**。

1. 第 5 步改为转身后沿新侧面再横移 `d_ring`（0.6 m），让机器人正对该面的起始边，再检测和拟合（抽出 `_fit_and_reach_side`）；横移时始终与侧面保持 `d_ring`。
2. `current = nf` 之后用 `try/finally` 包住，结果不是 ok 就恢复原来的面。
3. 失败时返回看到的面及其所属侧（`seen`）。
4. `guards.py` 中 `next_barcode` 只在不是 `face_done` / `done` 时清零失败计数，换面不会重置失败计数。

**效果**。95 个测试通过；在后续四面运行中，右侧面和其余各面都完成了拟合（见 v26、v28）。

### v26 · 2026-11-07 · 箱堆模型改存在自己的坐标系 S，每到一面重新配准

**当时的版本**。箱堆模型的面方向（由 `n0` 旋转得到）、占地轮廓和已扫码所在面的位置都存在里程计坐标里。

**问题**。四面运行到左侧面时：观察位走到了左前角，没站到左侧面中央；正面的一个码被错归到左侧面；`plan_stance` 把旧面的过时位置当障碍物，所有站位都被判为碰撞，3 个条码报"够不到"。各面法向相对真值的偏航误差：正面 2.9°、右侧面 9.0°、背面 15.6°、左侧面 21.8°；背面 bc8 的 x 误差为 −0.138 m。

**根因**。仿真有意模拟里程计误差（每走 1 m 偏航 1°、距离误差 2%），每次换面约累积 6.5°；模型存在 odom 里，误差逐面累积。

**修改**。

1. 模型存在平面坐标系 S（第一次规划时的 odom 系），通过 `(R_OS, t_OS)` 映射到 odom；新增 `_stack_T`、`_stack_to_odom`、`_odom_to_stack`、`_stack_face(fid)`；`_face_dirs` 用 `R_OS` 旋转 `n0`。
2. `_stack_add_faces` 在新面被拟合、且巡视中的前一面已知时重配准：旋转取拟合法向与预测方向的夹角，平移用两面共享的棱（模型侧取前一面朝新面方向的端点，新面侧取新平面朝前一面方向的端点）；修正量记入 `registrations`。
3. `_footprint`、`_root_clearance`、`go_to_next_face`、`_fit_and_reach_side` 都改用 S 数据在 odom 下的视图；每扫完一个码重新规划时也会再配准。

**效果**。误差只来自最近一段路的漂移，不再逐面累积（漂移本身无法消除）。新增单元测试：模拟 15° 偏航漂移加平移，配准后的面方向、占地轮廓和旧面位置都与漂移后的真实几何一致；全量 101 个通过、1 个跳过。

### v27 · 2026-11-07 · `move_arm` 要求先到达站位；只规划不算尝试；`plan_stance` 返回文字写明"机器人还没动"

**当时的版本**。`move_arm` 不检查机器人是否已在站位；`plan_stance` 成功时的 reason 为 "squat to base_height {h} after walking (pelvis z ...)"。

**问题**。第 11 轮中 LLM 在 `plan_stance` 之后没有走过去就直接抬手，差了 10 cm，最后放弃了这个条码。LLM 把 `plan_stance` 的返回文字误读成"已经到位"。

**根因**。"先走到站位"只写在提示词里；返回文字没有明确下一步。

**修改**。

1. 任务 3 中 `move_arm` 的 pre / final 要求当前码有 `ws.stance_target`，且 `self._stance_reached is ws.stance_target`，否则拒绝并说明下一步："no stance planned for {id}: call plan_stance, then approach('stance_target')" 或 "the robot is not at the planned stance: call approach('stance_target') first"。
2. `approach()` 包装 `_approach(target)`：目标是站位时先把当前码加入 `_stance_attempted`，成功后设置 `_stance_reached`。
3. `_walk_guard` 开头重置 `_stance_reached = None`，任何行走都会清掉"已到站位"。
4. "尝试过"改为：已开始走向站位、`plan_stance` 失败、或尝试过扫码；只是规划成功不算。
5. `plan_stance` 成功时的 reason 改为 "stance planned (the robot has not moved): call approach('stance_target') to walk there"，需要下蹲时再加 "; it then squats to base_height X (pelvis z Y m) by itself"。

**效果**。新增 `test_arm_poses_need_the_planned_stance_reached_and_a_planned_stance_alone_is_no_attempt`；全量 102 个通过、1 个跳过。

### v28 · 2026-11-07 · 多面巡视中没有码的面计划为空，不算失败

**当时的版本**。`plan_scans` 只要没有定位到任何标签就返回 fail("no barcode label localised on any fitted face")。

**问题**。在 stack4c_L0_0001（只有前后两面有码）上，绕到无码的侧面就会失败。需求是"四面都看、没码就不扫"。

**修改**。多面任务中，本面已拟合成功但没有标签时，计划 0 个码，状态 ok（"nothing to scan; call next_barcode"），把当前面记入 `ws.stack["planned"]`，随后 `next_barcode` 直接进入下一面。`guards.py` 的 finish 判定和 `next_barcode` 都把"已规划的空面"算作有计划。单面任务遇到同样画面仍按失败处理。提示词升到 v5，说明某一面可能没有码。

**效果**。新增参数化测试 `test_stack_side_without_labels`（`multi_face` 为 True / False），37 个逻辑测试通过。后续纸箱场景第 7 次运行（af4c_2side_v7）中四面全部走到，面的顺序连续、正面在先。

### v29 · 2026-11-07 · 任务 3 的 `check_wrist_placement` 超容差时改为 partial 并提示扫码

**当时的版本**。超出容差一律返回 fail("placement outside the tolerances")。

**问题**。起始朝向与箱堆不平行的一轮（stack4f_L0_0001）中，右侧面 bc4（z 1.06 m）的轴向偏移 2.0–2.4 cm，只比 2 cm 容差多一点（到平面距离 0.202–0.204 m，正常）。LLM 把 fail 当成停止信号，重走站位 4 次，一次 `scan` 都没调用，最后重试次数用完丢掉了这个码。

**根因**。这个检查本来只是测量：偏几厘米扫描器通常仍能读出，`scan` 自己也会精化并重试；规则编排在检查之后总是去扫。

**修改**。容差内仍返回 ok；超出时任务 3 返回 partial："placement outside the check tolerances (it measures, it does not decide): call scan now; scan reads from a few cm off and refines and retries by itself"。任务 2 保留 fail。提示词升到 v6。

**效果**。新增 `test_placement_check_measures_in_task3`：任务 3 偏 0.01 m → ok，偏 0.023 m → partial；任务 2 偏 0.023 m → fail。40 个逻辑测试通过。这一改动后来被证明太宽（v34）。

### v30 · 2026-11-07 · `move_arm` 只有标签仍在视锥内才报 partial

**当时的版本**。残差超过 `correction_tol` 一律报 partial "继续"（v9、v22）。

**问题**。stack4f_L0_0002 左侧面 bc10（z ≈ 1.08 m，举手）：站位离面 0.65 m、横向 −0.35 m，预位姿偏移 0.362 m，标签宽 0.105 m。`move_arm("pre")` 停在离目标 0.21 m、偏 23° 的位置，仍报 partial，LLM 照做，扫码读不到。

**根因**。partial 只看残差是否超阈值，没有检查从实际到达的位姿上标签是否还在腕相机视野里。

**修改**。残差 ≥ `correction_tol` 时，用 FK 求实际到达的枪口位姿，计算标签中心的 `along` 和 `lat`，半视野宽 `along·tan(wrist_hfov/2)`（0.2 m 处约 5.9 cm，由腕相机内参推出，不是新拍的数字）。标签中心在视锥内才返回 partial；在视锥外返回 fail，按 IK 不可达处理，引导重新规划站位。内参未知时抛 `RuntimeError`，不静默跳过。

**效果**。新增 `test_move_arm_partial_only_while_the_label_stays_in_the_scanner_view`：偏 0.03 m → partial，偏 0.21 m → fail；42 个逻辑测试通过。

### v31 · 2026-11-07 · 新增扫码结果 `unreadable`（无可读码）

**当时的版本**。读不出的码最终都记为 `failed`。

**问题**。纸箱纹理上印着运单贴纸（stack4c_L0_0001 上有 23 个），检测器在检测阶段无法把它们和真码分开：在纸箱场景上试过 4 种提示词，误检都在 13–29 个之间。这些贴纸被记为执行失败，混淆了"机器人做不到"和"这里本来就没有码"。

**根因**。在检测阶段区分真假需要针对样本的关键词或阈值，违反项目的设计原则；真正能判断"有没有可读码"的是近距离扫码本身。

**决定与修改**。不在检测端过滤，让机器人照常近距离去扫。把 `move_arm` 中的视锥判定抽成共用函数 `_label_view`。`scan` 重试后仍然没有任何码解出、且标签中心在扫码视野内时，把该码记为 `unreadable` 并写 `fail_reason`，作为一个结果而不是失败；标签不在视野内（放置问题）或解出的是别的码时仍按普通 no-read 处理，状态保持 `pending`。Harness 的 `_end_barcode` 在 `barcode_end` 事件里带上 `fail_reason`；`guards._finish` 的成功条件改为有扫到的码、所有状态都在 (scanned, unreadable) 中且没有下一面；规则编排的 finish 也把 unreadable 计入。真码召回只看真值标签。

**效果**。新增 `test_scan_records_a_label_without_readable_code`：偏 0.01 m 且无解码 → unreadable，偏 0.10 m → pending，解出其他码 → pending；45 个逻辑测试通过。判定条件在 v33、v34 中两次收紧。

### v32 · 2026-11-07 · 下蹲补看从"一个最佳档位"改为"所有合格档位都看并合并"

**当时的版本**（v20）。只取底部每米像素数最高、且比站立更好的一档。

**问题与数据**。在纸箱两面场景的正面和背面观察位姿，逐档（11 个档位，0.74 到 0.25）采集头部图像离线测试：最深的下蹲（底部分辨率最高）一个下排标签都没找到，0.40–0.60 档找到了全部 4 个，0.35 档只找到 1 个。

**根因**。开放词表检测器的召回率依赖视角，不随高度单调变化，"分辨率最高"不等于"检出最多"。

**修改**。凡是头相机能看到所有底角（留 5% 边距）、且得分高于站立的标定档位都蹲下看一次，按档位高度从高到低执行，每档新检出的标签按计划更新规则合并，最后站起。返回的 `low_view` 记录每档的 `height_cmd`、`px_per_m_bottom` 和新增的码。提示词升到 v7。代价是第一阶段多花约 2 分钟。

**效果**。离线：各档结果合并后两面的标签全部找到。纸箱两面场景第 2 次运行（af4c_2side_v2）的正面共蹲了 0.70 到 0.25 的 10 个档位：0.7 档新增 bc2 / bc3 / bc4，0.6 档新增 bc5 / bc6，0.65、0.55、0.5 档无新增（之后各档的输出在日志中被截断）；0.7 到 0.5 档的底部每米像素数依次为 195.3、203.7、213.0、220.3、227.7。正面 3 个真码都在这一步进入了计划。

### v33 · 2026-11-07 · `unreadable` 收紧（一）：整张标签在视野内；手腕读出过的码回预位姿补扫

**当时的版本**（v31）。只要求标签中心在视野内。

**问题**。纸箱两面场景第 2 次运行（af4c_2side_v2）中一个真码被判为 `unreadable`：refine 时腕相机在 0.40 m 处读出过它，但在 0.20 m 处 8 cm 的标签偏离轴 2.8 cm，一半落在 6 cm 半视野之外。

**根因**。最终位姿没读出不代表这里没有码；判定没有利用"之前已经读出过"这一信息，也只看了中心而不是整张标签。

**修改**。

1. `BarcodeEstimate` 新增 `wrist_read`，`refine_with_wrist` 通过 `_wrist_symbol_width` 解出内容时写入。
2. `scan` 失败时如果 `bc.wrist_read` 不为空，先 `move_arm('pre')` 回到曾经读出的距离再扫一次，记为 `move_arm_pre_where_read`。
3. `unreadable` 要求整张标签在视野内（`label_in_view`：`along > 0` 且 `lat + 宽/2 < 半视野宽`），并且从未被腕相机解出过。不满足的（如比视野还宽的大面单）如实记为 `failed`，不武断判为无码。

**效果**。测试改为四参数（miss、decoded、read_before、expect），47 个逻辑测试通过。

### v34 · 2026-11-07 · `unreadable` 收紧（二）：必须在扫码距离上；放置检查不在 stand-off 时报 fail

**当时的版本**（v29、v33）。`check_wrist_placement` 在任务 3 中只要超容差就报 partial "call scan now"。

**问题**。第 3 次运行（af4c_2side_v3）中 `move_arm('final')` 不可达，手臂停在预位姿（0.35 m，目标 0.20 m），检查仍提示"现在扫码"，`scan` 在错误的距离上把真码判成了 `unreadable`；同一轮正面中层的真码 bc1 也是在 0.28 m 处扫后被判为 `unreadable`。

**根因**。v29 的"只测量"没有区分"略偏几厘米"和"根本没到扫码距离"；后者的 no-read 说明不了标签本身。

**修改**。

1. `check_wrist_placement`：只有在 stand-off 距离上（`|dist − offset| ≤ check.dist_tol_m`，0.02 m）且标签在视野内时才报 partial；否则 fail，引导重新规划站位。
2. `unreadable` 再加一条：扫描枪在 stand-off 距离上（沿轴距离与 offset 之差 ≤ 0.02 m）。阈值取自已有配置。

**效果**。测试中的手腕目标补上 `pos` 和 `forward` 以计算距离，49 个逻辑测试通过。

### v35 · 2026-11-07 · 放置检查按最近一次手臂命令给提示；`next_barcode` 每码之后自动补看

**当时的版本**。`check_wrist_placement` 不在扫码位姿时一律提示重新规划站位；扫完一个码后的计划更新依赖 LLM 主动调用 `plan_scans`。

**问题**（第 5 次运行 af4c_2side_v5，背面在 3401 s 结束时 3 个真码只扫到 1 个）。

- LLM 在 `refine_with_wrist` 之后（手臂在预位姿）就调用检查，被告知重新规划站位，重规划了 4 次，始终没移到 final；有的标签在预位姿已被读出，实际上手臂只是还没收到 `final` 命令。背面中层 F4CMLBTNL 因此漏掉。
- 计划与真值逐条对照（这一面的里程计偏移约 (0.022, −0.199) m）：bc13 离 F4CTRBPBS 0.14 m，最终 failed；中层真码附近没有计划条目，说明观察位姿就漏检了，而 LLM 没有按提示调用 `plan_scans`，所以一直没补进计划。

**根因**。检查的提示没有区分"没下过 final"和"下了 final 但没到"；补看依赖 LLM 自觉调用。纸箱纹理上的标签检测受视角影响很大，近处再看一次更可能检出。

**修改**。

1. `move_arm` 拆成外层 `move_arm` 和实际执行的 `_move_arm`，外层记录 `self._last_arm = {pose, barcode_id, status}`。`check_wrist_placement` 不在扫码位姿时：本码还没执行过 `final` → 提示 "the arm was not sent to the final pose yet; call move_arm('final'), then scan"；`final` 已执行但没到 → 才提示重新规划站位。
2. `next_barcode` 在任务 3 中只要上一个码已结束，就自动调用 `_plan_update_from_here()` 从当前位置补看一次再选下一个码，返回 `plan_update`；去掉了原来的前置条件。

**效果**。新增 `test_next_barcode_looks_again_after_each_barcode_and_check_guides_to_final`，50 个逻辑测试通过。第 7 次运行（af4c_2side_v7）中这两个码都扫到了（bc13 在 2930 s，bc18 在 4284 s）。

### v36 · 2026-11-07 · `_not_current`：条码工具只作用于当前码

**当时的版本**。条码工具接受任意 `barcode_id`。

**问题**。第 6 次运行（af4c_2side_v6，stack4c_L0_0001）中，正面 3 个真码全部扫到、4 处贴纸全部正确判为 `unreadable`，但随后运行提前中止：LLM 没有调用 `next_barcode` 就去处理 bc8，在 bc7 还是当前条码时为 bc8 规划站位，随后在 "no stance planned for bc7" 上循环了 14 次，直到上下文写满。

**根因**。扫码顺序由 `next_barcode` 按计划给出，LLM 自行跳码会让站位、手臂目标与当前码错位。

**修改**。新增 `_not_current(barcode_id)`：任务 3 下传入的不是 `ws.current_barcode_id` 时返回 fail，提示 "{id} is not the current barcode ({cur}): call next_barcode to move on to the next barcode of the plan"；没有当前码时提示先调用 `next_barcode`。加在 `compute_wrist_target` 和 `refine_with_wrist` 开头。`scan` 判为 `unreadable` 后的提示由 "go on with the next barcode" 改为 "move_arm('home'), then call next_barcode"。

**效果**。57 个逻辑测试通过。

## 已知限制

- **几个常量没有依据**：`_reselect` 的 0.25 放宽系数、`_reperceive_target` 的 2 次尝试、条方向的 10 px 偏移、`_move_axis` 的 11 个采样点，以及 `task1.yaml` 中 `d_obs_range`、`max_segment_m`、`move_timeout_s`、`max_segments`、`max_fine_tunes`、`budget.*`，`task2.yaml` 中 `stance.min_face_dist_m`、`stance.step`，`task3.yaml` 中 `standoff.max_m`。它们目前只是"能用"，没有经过测量。
- **代码与注释不一致**：`compute_wrist_target` 返回的 `note` 仍写着"roll 不受约束"，实际 IK 已用 `long_axis` 约束 roll；`refine_with_wrist` 的 partial 提示里 "searched +-5 cm" 是写死的文字；`Toolbox._manipulability` 已不再被调用；`task1.yaml` 的 `selector_default` 只在任务 3 中被读取（读到的是 `task3.yaml` 覆盖后的值），它自己的 `left` 不起作用。
- **`detect_barcode_head` 只取第一个标签**（任务 2 路径），箱面上有多个标签时选哪个取决于检测得分。
- **重配准只能抵消到上一次拟合为止的漂移**：两次拟合之间（例如在同一面内走向多个站位）的里程计误差仍然存在，靠到站后的重拟合和腕部精化补偿。多面模型假设四个面两两垂直；占地轮廓用凸包表示，凹形箱堆的凹进部分会被当成障碍（推测，代码中没有对应的测试）。
- **`unreadable` 依赖"在 stand-off 处、整张标签在视野内、从未解出过"三个条件**：比扫描视野还宽的大面单永远满足不了第二条，只能记为 `failed`；条件都满足但真码因光照或模糊没解出时会被误判为无码（推测，尚未观察到）。
- **下蹲补看耗时**：所有合格档位都看一遍，第一阶段约多花 2 分钟；同时 `plan_stance` 对需要深蹲的码仍要几秒到几十秒（16 进程下 3 个堆叠条码为 5 / 10 / 47 s）。
- **腕部测宽带来额外动作**：预位姿和最终位姿之间宽度和中心会再更新一次，同一场景仿真时间从 72.2 s 增加到 109.0 s（v7）。当时提出的"首次腕部测宽后固定距离、之后只修正中心"还没有实现。
- **`adopt_stereo` 没有注册给 LLM**，运动立体只在规则编排中使用。
- **`check_wrist_placement` 在任务 2 和任务 3 中语义不同**：任务 2 是判定，任务 3 只是测量并给出下一步提示，调用方需要按任务理解它的 fail / partial。
