# 运动学、站位与身体控制

## 作用与在系统中的位置

这一部分负责机器人"身体"层面的所有几何计算和动作执行：

- 由关节状态算出头部相机、腕部扫码相机和扫码枪口在机器人根坐标系中的位姿（正运动学）；
- 给定条码位置，求左臂 7 个关节使枪口对准条码（逆运动学）；
- 为每个条码选一个"站位"：机器人站在哪里、朝向哪里、是否需要下蹲、蹲到哪一档；
- 执行行走、下蹲 / 站起和抬臂动作，并对执行结果做闭环修正和如实汇报。

感知模块给出"条码在哪里"，这一部分回答"机器人怎样才能把扫码枪放到那里"。

```
scripts/export_urdf.py  ──(离线一次)──▶ assets/g1_dex1_scanner.urdf
                                               │
                                     g1agent/kinematics/fk.py  Kinematics（pinocchio，基座固定在骨盆）
                                               │
                     ┌─────────────────────────┼──────────────────────────┐
           g1agent/kinematics/ik.py     g1agent/tools/body.py       g1agent/tools/toolbox.py
           ArmIK（7 关节 DLS）           BodyModel（下蹲档位 +        相机位姿 / 站位规划 /
                     │                    碰撞点 + 间隙）              move_arm / 行走 / 下蹲
                     └──────────┬───────────────┘                          │
                        g1agent/tools/stance_eval.py  ◀──（串行或进程池）───┘
                        单个候选站位的评估（纯函数）
```

与其它组件的接口：

| 对方 | 方向 | 内容 |
|---|---|---|
| 机器人端口 `g1agent/robot/interface.py` | 调用 | `get_joint_state`、`get_odom`、`move_root_relative(dx, dy, dyaw, timeout, pos_tol, yaw_tol, speed_scale)`、`set_body_pose(height, torso_rpy, duration, settle)`、`get_robot_geometry()`、`set_arm_joints(side, q, duration)`、`stand(s)`、`get_robot_info()` |
| 感知（箱面拟合、标签检测、腕部宽度测量） | 输入 | 箱面平面（中心、法向、角点）、条码中心 / 长轴 / 宽度及其来源（头部或腕部） |
| 世界状态 `WorldState` | 读写 | `wrist_target`、`wrist_target_refined`、`stance_target`、`root_target`、条码状态（`unreachable` 等） |
| 编排器（rule 状态机、LLM 编排） | 被调用 | 以工具形式暴露：`approach`、`explore`、`set_base_height`、`compute_wrist_target`、`plan_stance`、`move_arm`、`check_wrist_placement`、`go_to_next_face` 等，统一返回 `{status, data, reason}` |
| 评测 | 隔离 | 仿真器移动回复里的真值字段被 `move_summary` 过滤；本部分不使用任何真值（`tests/test_no_gt_leak.py` 检查） |

**坐标系约定**：运动学模型的根是骨盆（`configs/robot.yaml` 的 `root_frame: pelvis`），`Kinematics.pose` 返回 `T_root_frame`。工具层再用里程计位姿 `T_odom_root` 转到里程计坐标系（odom）。相机之间的相对位姿只由关节状态经正运动学得到，不用里程计推算（`toolbox.py` 文件头："head <-> wrist via kinematics, not odometry"）。

```{figure} images/motion_stack3_start.jpg
:width: 480px

三箱落地堆叠场景开始时的头部相机画面（640×480）。箱堆正面上下贴了三个条码，高度分别为 1.10 m、0.57 m 和 0.19 m（后来调到 0.30 m）。站位规划的大部分改进都是在这个场景里做的：最高的码要抬臂过肩，中间和最低的码要下蹲。
```

## 当前实现

本节各路径都相对于仓库根目录 `g1_agent_system/`。

### 1. URDF 导出：`scripts/export_urdf.py`

**作用**：把仿真里的 G1（29 自由度 + Dex1 夹爪 + 扫码枪）运动学树导出为 URDF，让客户端的 pinocchio 模型和仿真机器人完全一致。脚本在仿真环境里运行一次，输出 `assets/g1_dex1_scanner.urdf`。

**步骤**：

1. 从 `configs/system.yaml` 的 `sim_server.sim_root` 找到仿真包，实例化 `G1BarcodeEnv("s1_table", headless=True, seed=0)`。
2. 遍历 stage 上 `/World/envs/env_0/Robot` 下所有 `UsdPhysics.Joint`：
   - 跳过缺 body0 或 body1 的关节；父 / 子 link 名取目标路径最后一段。
   - 要求 `localPos1 == 0`（范数大于 1e-6 时抛 `RuntimeError`，不支持这种情况）。
   - 关节原点 = `make_T(localPos0, localRot0) · R1ᵀ`（`R1 = localRot1`），即 origin = (localPos0, localRot0) · inv(localRot1)。
   - `PhysicsFixedJoint` 导出为 `fixed`；转动 / 移动关节的轴为 `axis = R1 @ physics:axis`，这样运动方向不变。转动关节上下限由度换成弧度；其他关节类型抛 `RuntimeError`。
   - `<limit>` 中的 `effort="100" velocity="10"` 是写死的，代码没有给出依据；pinocchio 只用位置限位。
3. 追加四个固定帧，变换取自仿真配置：`head_cam_optical`（挂在 `camera.prim_parent_link` 下）、`scanner_mount`、`scanner_tip`、`wrist_cam_optical`（挂在 `robot.scanner_mount.parent_link` 下，变换为 `env._T_body_mount` / `_T_body_tip` / `_T_body_wcam`）。光学帧用 ROS 约定（x 右、y 下、z 前）。
4. `rpy()` 按 R = Rz(y)·Ry(p)·Rx(r) 分解，`|cos p| ≤ 1e-9` 时走万向锁分支。
5. 顶层 `except BaseException` 打印完整 traceback 后以退出码 1 结束，失败不会被吞掉。

当前产物有 54 个 `<joint>`。例如 `scanner_tip_joint` 的父 link 是 `left_hand_base_link`，xyz = (0, 0.181, 0.060)，yaw = π/2。

### 2. 正运动学：`g1agent/kinematics/fk.py` 的 `Kinematics`

| 成员 | 说明 |
|---|---|
| `__init__(urdf=None)` | 读 `configs/robot.yaml` 的 `urdf`，`pin.buildModelFromUrdf` 建模；建立关节名 → q / v 下标表，复制上下限。 |
| `q_from_state(names, q)` | 机器人上报的 (names, values) → pinocchio 配置。从 `pin.neutral` 出发，未知名字忽略，缺失关节为 0。 |
| `frame_id(name)` | 帧名 → id，不存在抛 `KeyError`。 |
| `pose(q, frame)` | `framesForwardKinematics` 后返回 4×4 `T_root_frame`。 |
| `jacobian(q, frame)` | 6×nv 帧雅可比，参考系 `LOCAL_WORLD_ALIGNED`；前 3 行线速度，后 3 行角速度。 |

`configs/robot.yaml` 中的帧配置：`head_cam_frame: head_cam_optical`；`arm_side: left`；`wrist_point_frame: scanner_tip`（枪口，即扫码枪出光窗中心）；`wrist_cam_frame: wrist_cam_optical`（DS2208 类成像器，640×480，水平视场 32.8°）；`wrist_forward_axis: [1, 0, 0]`（扫描方向是 `scanner_tip` 的 x 轴）。

### 3. 手臂逆运动学：`g1agent/kinematics/ik.py` 的 `ArmIK`

**任务**：左臂 7 个关节（`shoulder_pitch/roll/yaw`、`elbow`、`wrist_roll/pitch/yaw`），目标帧是枪口 `scanner_tip`。约束为位置 3 个 + 扫描方向 2 个；给了 roll 轴时再加绕扫描轴的滚转 1 个（扫码图像的水平轴平行于条码长轴，正反都算），共 6 个。求解器是 pinocchio 模型上的阻尼最小二乘（DLS）。

**为什么不复用 xr_teleoperate 的 IK**：那个求解器的目标帧是它自己 URDF 里的 Dex3 / Inspire 手部帧，而本机器人装的是夹持扫码枪的 Dex1 夹爪，枪口帧只存在于我们导出的 URDF 里。

**主要函数**：

- `rotation_vector(a, b)`：把单位向量 a 转到 b 的旋转向量。平行时返回 0；反平行时取一条与 a 垂直的轴，转角 π。阈值 1e-9 / 1e-6 只用于防除零。
- `__init__`：有效限位 `lo / hi` = 软限位向内收 `joint_limit_margin`；`img_x_in_tip` 是中性位姿下腕部相机光学 x 轴在枪口帧中的方向（模型固定变换，当前 URDF 为 [0, −1, 0]）。
- `roll_target(R, d_t, roll_axis)`：构造枪口帧的完整目标旋转：x 轴沿 `d_t`；roll 轴投影到垂直于 `d_t` 的平面得到 L，图像水平轴对准 ±L 中离当前姿态更近的一个。
- `roll_error_deg(R, d_t, roll_axis)`：图像水平轴与条码长轴绕扫描轴的夹角，0–90°。
- `_solve_from(q, p_t, d_t, stall, roll_axis)`：一次 DLS 迭代求解：
  1. 位置误差 `ep`；方向误差 `er`：无 roll 轴时是 x 轴到 `d_t` 的旋转向量，有 roll 轴时是 `rotvec(roll_target · Rᵀ)`（完整姿态误差）。
  2. 收敛：`|ep| < tol_pos` 且角度 < `tol_angle_deg`。
  3. 停滞检测（只在站位规划中传入 `stall=(window, eps)`）：记录 `|ep| + aim_weight·|er|`，最近 `window` 次迭代总共下降不到 `eps` 就放弃。
  4. 雅可比取 7 个手臂列。无 roll 轴时角速度行左乘 `P = I − a aᵀ`，让绕扫描轴的转动自由；有 roll 轴时 `P = I`。
  5. `A = [J_lin; w·P·J_ang]`，`e = [ep; w·P·er]`，`dq = step_gain · Aᵀ (A Aᵀ + λ² I)⁻¹ e`，更新后截断到 `[lo, hi]`。
- `solve(q_now, p_target, d_target, stall=None, roll_axis=None, rng=None)`：多起点求解。起点是当前构型加 `restarts` 个在 `[lo, hi]` 内均匀采样的手臂构型；当前构型收敛就立即返回。评分 = `pos_err + aim_weight · rad(angle_err + roll_err)`，成功的优先，再比评分。传入 `rng` 时用调用方的随机数生成器（保证并行规划结果可复现）。失败返回 `ok=False` 和最好的一次尝试。

**配置 `configs/robot.yaml → ik`**：

| 键 | 值 | 依据 |
|---|---|---|
| `dls_lambda` | 0.05 | 注释中没有依据 |
| `step_gain` | 0.5 | 每次迭代纠正误差的比例（离线求解，无动力学） |
| `max_iters` | 300 | 注释中没有依据 |
| `tol_pos` / `tol_angle_deg` | 0.002 m / 1.0° | 注释中没有依据 |
| `aim_weight` | 0.20 m/rad | 与仿真器的瞄准 IK 相同：按 0.20 m 扫描距离把角度误差换算成长度 |
| `joint_limit_margin` | 0.05 rad | 与每个软限位保持的距离 |
| `restarts` | 6 | 第一次不收敛时追加的随机起点数；没有给出数值依据。站位规划期间临时改为 1 |
| `plan_workers` | 16 | 站位规划进程池大小，0 = 串行。依据见演进记录 v14 |
| `plan_stall_window` / `plan_stall_eps_m` | 100 / 0.0005 m | 只用于站位规划，`move_arm` 不用。依据见 v12 |
| `seed` | 0 | — |

### 4. 全身模型：`g1agent/tools/body.py` 的 `BodyModel`

两个数据来源：

- **下蹲档位**：在仿真中实测的身体高度指令标定（`assets/gear_body_calibration.json`）。规划只使用实测过的档位，不在档位之间插值。
- **机器人碰撞几何**：机器人端口 `get_robot_geometry` 给出的每个 link 的碰撞点（凸包顶点，或球心 + 半径，在 link 帧下）。这是机器人设计数据，不是真值。

它用于两类检查：机器人所有点到已知箱面的间隙（深蹲时膝盖前移到骨盆前方约 0.3 m）；手臂与腿 / 躯干的间隙（仿真关闭了自碰撞，手臂穿过大腿在仿真里不会被发现，所以由规划器自己拒绝这类姿态）。

**构造**：

1. 按 link 分组碰撞点；有点的 link 在 URDF 里没有对应帧就抛 `KeyError`。
2. 用名字关键字选 link：`legs` = 名字以 `left_` / `right_` 开头且含 `hip` / `knee` / `ankle` 的 link，再加 `pelvis`、`pelvis_contour_link`、`torso_link`；`arm` = 以 `left_` 开头且含 `elbow` / `wrist` / `hand` 的 link（肩部 link 不参与）。这几张关键字表对应 G1 的 link 命名，代码中没有其它依据。任一集合为空抛 `ValueError`。
3. 半径全为 0 且不少于 4 个点的腿部 link 用 `scipy ConvexHull` 求半空间方程；点集共面时只捕获 `QhullError`，退化为按点当球处理。
4. `arm_reach`：中性位姿下肩 → 枪口链上相邻帧原点距离之和，是任意构型下 |枪口 − 肩| 的上界，只用于预筛，不参与决策。
5. 标定档位按 `height_cmd` 降序排列（站立在前）；为空抛 `ValueError`。

**姿态函数**：

| 函数 | 说明 |
|---|---|
| `q_at(q_base, level)` | 用该档位实测的腿部和腰部关节覆盖 `q_base`，手臂关节保持不变。 |
| `T_root(x, y, yaw, level, z_ground=0)` | 骨盆在 odom 中的位姿：高度 = 实测 `pelvis_z`，再右乘实测俯仰 `Ry(pelvis_pitch_deg)`。 |
| `foot_shift(level, standing, q_base)` | 双脚不动时，从站立蹲到该档位骨盆的水平位移（两个 `ankle_roll_link` 平均位置之差）。行走目标 = 下蹲后的骨盆目标 − 这个位移。 |
| `points(q, T_root, names)` | 指定 link 的碰撞点变换到给定坐标系。 |

**间隙函数**：

- `self_clearance(q)`：手臂与腿 / 躯干之间距离的**下界**（m，负值表示手臂点在腿 link 内部）。手臂点变换到每个腿 link 的帧：凸包 link 取各面平面有符号距离的最大值（在凸包外时这个值不大于真实距离），球体 link 取到球面距离，再减手臂点半径。返回 `(最小值, link 名)`。
- `face_clearance(q, T_root, faces, footprint=None)`：机器人任意点到已知箱面的最小**水平**间隙。给了 `footprint`（多面任务中闭合的堆叠外轮廓）时用到多边形的有符号距离；否则逐个箱面计算：点在面横向范围内取沿法向的有符号距离，在范围外取到面边缘的距离，再减半径。

**下蹲标定** `assets/gear_body_calibration.json`：GR00T 步行控制器，静止站立、双臂 home 位，每档从上一档渐变 1.0 s 再保持 2.0 s，取保持结束时的值；所有档位状态正常，脚漂移不超过 0.2 mm。共 11 档，每档记录 33 个关节值：

| height_cmd | 骨盆高度 pelvis_z (m) | 骨盆俯仰 (°) |
|---|---|---|
| 0.74（站立） | 0.7458 | −0.18 |
| 0.70 | 0.6902 | 0.84 |
| 0.65 | 0.6213 | 4.09 |
| 0.60 | 0.5631 | 8.67 |
| 0.55 | 0.5156 | 12.70 |
| 0.50 | 0.4715 | 16.47 |
| 0.45 | 0.4272 | 20.73 |
| 0.40 | 0.3788 | 24.44 |
| 0.35 | 0.3277 | 25.52 |
| 0.30 | 0.2815 | 25.40 |
| 0.25 | 0.2494 | 26.05 |

### 5. 下蹲与站起：`Toolbox.set_base_height` 等

- `_body()`：第一次使用时用 `robot.get_robot_geometry()` 和标定档位构建 `BodyModel`；行走控制器不支持身体高度指令时返回 None，此时站位规划只在当前高度进行。
- `_standing()`：没有高度指令，或当前指令 ≥ `height_default − 1e-6`。
- `set_base_height(height=None)`（工具）：None 表示站起。高度必须在机器人报告的 `height_range` 内；调用 `robot.set_body_pose(height, None, ramp_s, settle_s)`，返回 `pelvis_z`、`tilt_deg`。躯干俯仰 / 横滚指令目前不使用（传 None）。

**配置 `configs/robot.yaml → body`**：

| 键 | 值 | 依据 |
|---|---|---|
| `calibration` | `assets/gear_body_calibration.json` | 实测；规划只使用这些档位 |
| `ramp_s` / `settle_s` | 1.0 / 2.0 s | 等于标定时的渐变和保持时长，每档在这个时长下都稳定 |
| `clearance_margin_m` | 0.03 m | 机器人到箱面、手臂到腿的最小间隙 = 运动测试中最大手臂瞄准误差 8 mm + 放置距离容差 0.02 m（`task2.yaml check.dist_tol_m`），向上取整 |
| `stance_robust_pos_m` / `stance_robust_yaw_deg` | 0.03 m / 1.7° | 等于仿真器 `move_relative` 的停止容差（pos_tol 0.03 m，yaw_tol 0.03 rad ≈ 1.7°） |

### 6. 行走

**`_walk_guard()`**：每个行走工具第一步都调用它。

1. 把 `_stance_reached` 清为 None（机器人一走动就离开了已到达的站位）。
2. 机器人处于下蹲时：手臂不在 home 位就返回 fail，提示先 `move_arm('home')`（站起时手臂不能扫过伸展的腿）；手臂在 home 位就自动 `set_base_height(None)` 站起。

站起和下蹲都由工具负责，不依赖编排器的调用顺序（原因见 v11）。

**`move_to` / `_rel`**：odom 目标位姿 → 相对当前位姿的 (dx, dy, dyaw) → `move_root_relative`。

**`explore(direction, distance)`**：平移 `forward/left/right/back`，距离在 (0, `max_move_m`] 内；转向 `turn_left/turn_right`，角度在 (0, `max_turn_deg`] 内。终点到任一已拟合箱面的距离小于 `stance.min_face_dist_m`（0.35 m）且这次移动在靠近该面时拒绝。移动后离 episode 起点超过 `workspace_radius_m` 返回 fail。

**`approach(target)` / `_approach`**：分段接近。

- 目标为 `root_target`（观察位姿）：每段走 `min(剩余 × segment_frac, max_segment_m)`，每段后重新感知并修正目标（`_reperceive_target` → `_retarget_from_refit`，只用重拟合的平面修正位置，保留规划时的观察距离和横向位置）；剩余小于 `final_below_m` 时以 `precise_speed_scale` 速度走最后一步，再做 `arrive_check`。最多 `max_segments` 段；回复状态为 `fell` / `collision` 立即 fail。
- 目标为 `stance_target`（扫码站位）：不重新感知。站着走过去，到达后站位带的 `base_height` 与当前不同时自动 `set_base_height(bh)` 下蹲。成功时 `_stance_reached = stance_target`；无论成功与否都把当前条码记为"已尝试"。
- `biased(t)`：在目标朝向上提前 `final_bias_m` 停下（只对 `root_target`），当前为 0。

**`arrive_check(log)`**：到达条件为 |沿法向距离误差| < `dist_tol_m`、|dyaw| < `yaw_tol_deg`、箱面在视野内。横向偏差不检查：任务和评测只需要到面的距离、朝向和面在视野内。最多 `max_fine_tunes` 次微调：面在视野内时保持当前横向位置，不在视野内时才纠正横向；微调的停止容差 = 检查容差 × `fine_tol_frac`（执行控制器要比检查停得更紧，否则会停在检查边界上）。任务 3 中仍未到达返回 partial，提示直接进入 `plan_scans`。

**配置**（`configs/robot.yaml → limits`；`configs/task1.yaml`，任务 3 继承）：

| 键 | 值 | 依据 |
|---|---|---|
| `limits.max_move_m` / `max_turn_deg` | 0.5 m / 15° | 设计文档 §3.4 的防护项；具体数值没有给出依据 |
| `limits.max_approach_segment_m` / `workspace_radius_m` | 1.0 m / 8.0 m | 同上 |
| `approach.segment_frac` / `max_segment_m` | 0.5 / 1.0 m | 只说明用法 |
| `approach.final_below_m` / `lost_backoff_m` | 0.3 m / 0.3 m | 只说明用法 |
| `approach.precise_speed_scale` | 0.5 | 全速短距离移动曾过冲多达约 12 cm：控制器在离目标 0.15 m 之前一直饱和在 0.4 m/s |
| `approach.final_bias_m` | 0.0 | 原为 0.05，见 v2 |
| `approach.move_timeout_s` / `max_segments` | 20 s / 8 | 没有给出依据 |
| `arrive.dist_tol_m` / `yaw_tol_deg` | 0.03 m / 3.0° | harness 端基于估计的到达检查 |
| `arrive.max_fine_tunes` / `fine_tol_frac` | 2 / 0.5 | 微调在到达容差的一半（1.5 cm / 1.5°）停下 |

**绕堆叠移动（多面任务）**：`_move_axis(dx, dy, dyaw)` 每次只沿一个轴移动（控制器在 x / y / yaw 上各自饱和，只有单轴运动才走直线）；平移时在路径上取 11 个等距采样点（数量没有依据），任一点到堆叠的间隙 `_root_clearance` 小于 `min_face_dist_m` 就拒绝。`go_to_next_face` 的路线（转身 → 前后到环绕距离 → 侧移越过角点 → 转向下一面 → 沿下一面再侧移一个环绕距离）和堆叠模型的重配准在"工具层"一页中详细说明。

### 7. 腕部目标与站距

- `compute_wrist_target(...)`：目标点 `pos = 条码中心 + off·法向`，`forward = −法向`，同时带上条码长轴 `long_axis`。自适应站距模式下调用方传入的 offset 被忽略（记入 `offset_ignored`），不报错。
- `_standoff(bc)`：`fixed` 模式用 `offset_m`（0.20 m）；`adaptive` 模式（任务 3）用 `clamp(width / (2·tan(hfov/2)) / fill_frac, min_m, max_m)`。宽度来自头部时再取 `max(d, offset_m)`：头部估出的宽度只能让扫码枪退得更远，只有腕部实测宽度才能让它更近。缺宽度或缺内参抛 `RuntimeError`。
- `_long_axis(bc)`：有条纹方向 `bar_dir` 时取 `法向 × bar_dir`（code128 扫描线应沿的方向），否则取标签四边形较长的边，投影到面内并归一化。
- `_pre_gap()`：优先用 `plan_stance` 为当前条码找到的 `pre_gap`，否则用 `pre_offset_m − offset_m` = 0.10 m。预备位距离 = 站距 + `pre_gap`。

相关配置：`task2.yaml` 中 `offset_m: 0.20`、`pre_offset_m: 0.30`；`task3.yaml` 中 `standoff.mode: adaptive`、`fill_frac: 0.68`（标称 0.08 m 的 code128 落在已验证的 0.20 m：fx 1087、640 px 宽对应 32.8°，0.08 / (2·tan 16.40°) / 0.20 = 0.679）、`min_m: 0.12`（设计需求 5）、`max_m: 0.30`。

### 8. 站位规划：`plan_stance` / `_plan_stance`

`plan_stance` 是外层包装：规划失败时把这次计为对当前条码的一次尝试；规划成功不算尝试，尝试从走向站位开始计。

`_plan_stance(which)` 的步骤：

1. **取目标**：`which="refined"` 用细化后的腕部目标，否则用 `wrist_target`；当前条码还没有腕部目标时自动调用 `compute_wrist_target`。
2. **站位网格**：以条码中心为原点，沿水平化的面法向取离面距离 0.35–0.70 m，沿机器人左轴取横向 −0.40 到 +0.40 m，步长都是 0.05 m 且包含两端（代码里上限加了 1e-9，共 8 × 17 = 136 个点），朝向一律正对箱面。
3. **障碍**：`_known_faces()` 返回当前检测到的箱面和各条码保存的面副本；多面任务中只保留当前面的副本，其它面由重配准后的堆叠外轮廓代表。`min_face_dist` 对邻面同样适用（箱子前后错落时，横向站位可能落在某个凸出邻面的正前方）。
4. **必须可达的距离**：
   - 自适应模式且宽度来自头部：头部宽度是符号宽度的上界，真实站距落在 [`min_m`, offset] 之间，最终距离要等腕部实测后才知道。所以 `finals = [min_m]`（必须够得着最近的可能位置），`pre_dists` 从 `offset + 0.10` 以 0.05 递减到 `min_m`。
   - 其它情况：`finals = [offset]`，`pre_dists` 从 `offset + 0.10` 递减到 `offset`。预备位只需要让腕部相机先看清条码，所以首选间隙够不着时接受更小的间隙。
5. **按下蹲档位循环**（站立在先；没有身体模型时只用当前高度）。第一个（最高的）有可行站位的档位胜出，因为下蹲耗时且减小平衡裕度。每档：
   - `q_lv = body.q_at(q0, lv)`；行走目标 `pw = p − R_yaw · foot_shift(lv)`，保证蹲下后骨盆正好落在规划点上。
   - 网格点预筛：外轮廓闭合时 `pw` 到多边形的有符号距离 ≥ `min_face_dist`；否则到条码面平面和每个障碍面（`horizontal_distance_to_face`，到有限线段的距离）都 ≥ `min_face_dist`。
   - 骨盆位姿 `Tc = body.T_root(p, yaw, lv)`，为每个点生成一个 job，IK 种子为 `[seed, 档位下标, 网格下标]`，交给 `_eval_stance_jobs`（见第 9 节）。
   - 成功候选记代价 `cost = (−gap, −man, travel)`：先要大的预备间隙，再要高的可操作度，最后要短的行走距离。
   - **按代价顺序做昂贵检查，第一个全部通过的即选定**。代价与这些检查无关，所以结果与"全部检查后选最便宜的合格者"相同：
     1. `face_clearance`（所有必需解的最小值）≥ `clearance_margin_m`；
     2. `self_clearance` ≥ `clearance_margin_m`；
     3. 记录 `clear = {face_m, arm_leg_m, arm_extension, manipulability}`；第一个通过两项间隙检查的候选记为 `fallback`，只用于失败信息中标注"只有精确停位才够得着"，不会被采用；
     4. `_stance_robust`：站位分别沿 ±x、±y 偏移 `stance_robust_pos_m`，绕骨盆偏转 ±`stance_robust_yaw_deg`，共 6 种扰动（逐个施加，不组合），所有必需的最终距离和所选预备距离都必须仍有 IK 解。这一步在主进程串行执行。
   - 规划期间 `ik.cfg["restarts"]` 临时改为 1，在 `finally` 中恢复。
6. **失败**：所有档位都没有鲁棒站位时，条码标为 `unreachable`，原因写明：标签中心高度低于站立骨盆高度时说"最深下蹲也够不着（太低）"，否则说"站立也够不着（太高）"，附上骨盆高度和容差；只能精确停位够到时注明 "reachable only with an exact stop"。`data` 中有 `rejected` 各类计数（`reach`、`ik`、`face_clearance`、`self_clearance`、`not_robust`）、`levels_tried`、`exact_stop_only`。
7. **成功**：写入 `ws.stance_target = {pos(行走目标), yaw, face_dist, lateral_offset, barcode_id, travel_m, base_height, pelvis_pos, pelvis_z, pre_gap}`，提示调用 `approach('stance_target')`，需要下蹲时说明会自动蹲到哪一档。

**`task2.yaml → stance`**：

| 键 | 值 | 依据 |
|---|---|---|
| `min_face_dist_m` | 0.35 m | 最近网格行，也是 `explore` 和 `_move_axis` 的安全下限；没有数值依据 |
| `max_face_dist_m` | 0.70 m | 任务 2 早期运行中实测的 61 个站位在 0.52–0.63 m，再留余量 |
| `lateral_range` | [−0.40, 0.40] m | 同样 61 个站位横向在 −0.37 到 +0.11 m |
| `step` | 0.05 m | 没有依据 |

### 9. 单候选评估与进程池：`g1agent/tools/stance_eval.py`、`_eval_stance_jobs`

`stance_eval.py` 把单个候选站位的评估写成纯函数，串行调用和进程池调用行为相同。worker 只导入 numpy、pinocchio 和运动学模块。

| 函数 | 说明 |
|---|---|
| `init_worker(shoulder_frame, arm_reach)` | 进程池初始化：`_die_with_parent()`（`prctl(PR_SET_PDEATHSIG, SIGKILL)`，失败抛 `OSError`），建立自己的 `Kinematics` / `ArmIK`，`restarts` 设为 1。 |
| `target_in_root(t, Tc, offset)` | 把腕部目标沿扫描线平移到距离 `offset`，再变换到候选骨盆位姿 `Tc` 的根系。 |
| `roll_in_root(t, Tc)` | 条码长轴转到根系；没有长轴返回 None。 |
| `manipulability(kin, ik, q)` | 枪口位置雅可比在 7 个手臂关节上的 `sqrt(det(Jp Jpᵀ))`（Yoshikawa 可操作度）。手臂完全伸直和完全折叠时都很低。 |
| `eval_candidate(kin, ik, shoulder, reach, job)` | ① 臂长预筛：离肩最近的必需目标仍远于 `arm_reach` → `"reach"`；② 对每个 `finals` 距离求 IK（带停滞检测和 roll），不可达 → `"ik"`；③ 在 `pre_dists` 中由远到近找第一个可达的预备位，记 `gap`，都不可达 → `"ik"`；④ 成功返回 `{sols, gap, margin, man}`，`man` 为各解可操作度的最小值。 |
| `eval_batch(jobs)` | worker 内依次评估一批 job。 |

`Toolbox._eval_stance_jobs(jobs, body)`：`plan_workers ≤ 0` 或 job 少于 2 个时串行；否则惰性创建 `ProcessPoolExecutor(max_workers=16, spawn, initializer=init_worker)`，并立即 `map(int, range(nw))` 把所有 worker 拉起来。创建期间临时把 `OMP_NUM_THREADS`、`OPENBLAS_NUM_THREADS`、`MKL_NUM_THREADS` 设为 1，之后恢复（这些环境变量在 spawn 出的 worker 导入 numpy 时读取）。任务切成每个 worker 一个连续块，按顺序拼回。因为每个候选的随机数由自己的种子决定，串行和并行结果逐项相同（`tests/test_stance_pool.py` 检查）。

### 10. 手臂运动：`move_arm` / `_move_arm`

外层 `move_arm` 记录 `_last_arm = {pose, barcode_id, status}`，供放置检查给提示。`_move_arm(pose)`：

1. `home`：`set_arm_joints(side, _q_home, duration_s)`，`arm_at_home = True`。home 位由编排器开始时调用 `remember_home()` 记录，未记录时抛 `RuntimeError`。
2. **前置条件**（任务 3）：必须已为当前条码规划了站位，且已通过 `approach` 到达（`_stance_reached is stance_target`）。
3. **目标**：`final` 优先用细化目标，距离为站距；`pre` 的距离为站距 + `pre_gap`。
4. **闭环修正**，共 `1 + correction_iters` 轮：
   - 在当前关节状态上求 IK（带 roll，不做停滞检测，`restarts=6`）。不可达时：已有更好的结果就退回它并结束；否则返回 fail 并附残差。
   - `set_arm_joints` → `stand(settle_s)` → 把目标重新表达到当前根系（底座可能漂移）→ 用正运动学测枪口误差。
   - 误差比之前最好的还大：退回最好的指令并停止修正。
   - 误差 < `correction_tol` 结束；否则 `p_cmd += err`，对指令目标做一次满增益修正，补偿手臂 PD 下垂。
5. **结果汇报**：最终残差 ≥ `correction_tol` 时，用 `_label_view` 判断标签中心是否仍在腕部相机视锥内：不在则返回 fail（当作从此站位不可达，需重新规划）；在则返回 partial，并明确说明"这不是失败，继续"。

`_label_view(t, off, label_width=None)`：在枪口帧中把枪口到标签中心的向量分解为沿扫描轴的 `along` 和横向的 `lat`，半视野 `half = along · tan(hfov/2)`，`in_view = along > 0 且 lat < half`；给了宽度时还要求整个标签在视野内。

**配置 `robot.yaml → arm_motion`**：`duration_s: 2.0`、`settle_s: 1.0`、`correction_iters: 2`、`correction_tol: 0.01 m`。后两项对应设计文档 §4.3 第 8 步"偏差 > 1 cm 时再迭代一次（最多 2 次）"；`duration_s` 和 `settle_s` 没有给出数值依据。

### 11. 其它运动学用途

- `check_wrist_placement`：用 odom · 正运动学得到枪口位姿，计算到条码平面的距离与站距之差（≤ 0.02 m）、条码中心偏离扫描轴的距离（< 0.02 m）、扫描轴与 −法向的夹角（< 10°），以及 `ik.roll_error_deg` 给出的 roll 误差（与夹角共用 10° 容差）。任务 3 中这个工具只测量不决定：在站距处且标签在视野内时返回 partial 并要求扫码；还没执行过本条码的 `final` 时提示先 `move_arm('final')`。
- `compute_obs_pose` / `check_face_in_view` / `_low_view_update`：用正运动学得到的头部相机位姿投影箱面角点。`_low_view_update` 还用 `q_at`、`foot_shift`、`T_root` 预测每个下蹲档位的头部相机位姿（见 v19）。
- 模块级函数：`horizontal_distance_to_face(p, face)`（俯视下点到有限面线段的距离）、`_poly_signed_distance(P, poly)`（点到逆时针凸多边形的有符号距离，内部为负）。
- `Toolbox._manipulability(q)` 有定义但当前没有调用，实际使用 `stance_eval.manipulability`。

## 演进记录

日期和时刻取自开发日志。代码注释里有几处同一批改动标为 2026-11-06（例如 `body.py` 文件头），日志里对应的时间是 2026-11-07 上午，推测是两处记录用的时区不同；下面一律以日志为准。

### v0 · 2026-11-05 · 从仿真 USD 导出 URDF，pinocchio 正运动学 + 5 维 DLS 手臂 IK

**初版设计**：

- `scripts/export_urdf.py` 直接读取仿真里 G1 29 自由度 + Dex1 + 扫码枪的 USD，把每个 `UsdPhysics` 关节转成 URDF 关节（转换方式见"当前实现"第 1 节），并从仿真配置补上 `head_cam_optical`、`scanner_mount`、`scanner_tip`、`wrist_cam_optical` 四个固定帧。这样客户端模型和仿真机器人完全一致，不需要另找 G1 的官方 URDF（当时本地既没有 G1 的 URDF，也没有 xr_teleoperate 仓库）。
- `configs/robot.yaml`：根帧 `pelvis`，扫码臂 `left`，腕部点 `scanner_tip`，腕部相机参数按 DS2208 类成像器（640×480，水平视场 32.8°）；IK 参数 `dls_lambda 0.05`、`step_gain 0.5`、`max_iters 300`、`tol_pos 0.002 m`、`tol_angle 1°`、`aim_weight 0.20 m/rad`（按 0.20 m 扫码距离把角度误差换算成长度）。
- `ArmIK` 是 5 维任务：枪口位置 3 维 + 扫描方向 2 维，不约束绕扫描轴的滚转；7 个关节足够。没有复用 xr_teleoperate 的 IK，原因见"当前实现"第 3 节。
- `move_arm` 按设计文档 §4.3 第 8 步做 `correction_iters = 2` 次满增益闭环修正，补偿手臂 PD 下垂；目标锚定在里程计坐标系，底盘漂移时自动补偿。

**验证数据**：模型 nq = 33，共 110 个帧；离线 IK 6 次迭代收敛，位置误差 1.3 mm，角度误差 0.21°。用 pinocchio 算出的头部和腕部相机位姿与仿真真值相比误差为 0.000 mm / 0.000°，运动学里程碑 M2 判定通过。仿真器端的枪口瞄准控制（同样是 5 维任务）实测 1–2 s 收敛，误差约 0 cm / 0°。

### v1 · 2026-11-05 · 行走停位不准：瞄准目标前 5 cm、末段减速

**当时的版本**：行走用仿真包自带的 policy1 控制器。`approach` 停在目标前 5 cm（`final_bias_m = 0.05`），之前汇报的 `move_relative` 到位误差 3–4 cm。

**问题**：3–4 cm 是在命令结束那一刻测的，机器人停稳后实际误差 6–14 cm。机器人冲过头又退不回来，任务 1 经常过不了工具层自己"误差 < 3 cm"的到位检查（评测的要求是 < 5 cm）。

**根因**：在上游行走策略本身：速度命令归零后还要多走 10–15 cm；完全不能后退（后退 8 cm、15 cm 位移都是 0）；小于约 10 cm 的移动经常起步不了。

**尝试**：停在目标前方、半速、按实测速度预测停止点、按命令速度预测停止点、加长减速距离，5 种办法都不可靠。按设计文档"同一问题尝试两次不成功就停下请需求方决定"的规则，停下来请需求方决定。之后的处理是更换行走控制器（见 v2）。末段半速 `precise_speed_scale = 0.5` 保留至今，代码注释给的依据是全速短距离移动曾过冲多达约 12 cm；它是否就是在这一阶段引入的，日志中没有明确记录（推测）。

### v2 · 2026-11-06 · 换用 GR00T 控制器后 `final_bias_m` 0.05 → 0

**当时的版本**：行走控制器已换成 GR00T 全身控制器，但 `task1.yaml` 的 `approach.final_bias_m` 仍是为 policy1 设置的 0.05（policy1 过冲后走不回来，所以故意瞄准目标前 5 cm）。

**问题**：L0 最简单场景中 `approach` 的到位检查第一次没通过（横向差 3.7 cm，容差 3 cm）；rule 模式在任务 1 阶段停在离目标差 6.9 cm 处。当时决定不针对单个场景调参，和批量评测的失败一起分析。

**根因**：GR00T 能在约 1 cm 内停住并且能后退，5 cm 的提前量已经不需要，反而让机器人系统性地停在目标前方。

**修改**：`final_bias_m` 改为 0.0（瞄准目标本身），在配置注释里写明历史原因。

**效果**：之后任务 3 的重录运行中，rule 模式任务 1 阶段通过（`t1_success=True`）。

**同日的一个判断**：需求方问"抬着手臂走路时航向会偏"这个问题影响大不大。我们查了 rule 和工具层的调用顺序：两种编排都是先在手臂处于 home 时走到位、停下，再抬臂瞄准，机器人从不在抬臂状态下行走；抬臂后站立时航向只漂 0.04–0.24°，瞄准目标固定在世界系，IK 会补偿底盘的小幅移动。唯一可能受影响的是 LLM 编排（理论上可能在 `move_arm` 之后再调 `approach`），决定先不处理。后来在 v11 和 v17 中，由工具自身的前置条件把这种情况彻底排除。

### v3 · 2026-11-06 · 初版站位规划：只在站立高度搜索，按关节余量选

**初版设计**：任务 2（单个条码）阶段的 `plan_stance` 在站立高度下搜索站位网格，要求预备位和最终位都有 IK 解、不碰箱子，然后按"关节限位余量最大"、再按"行走距离最短"选。网格最早以机器人当前位置为中心、横向 ±0.30 m；后来改成以当前条码为参照：离面 0.35–0.70 m、横向 −0.40 到 +0.40 m、步长 0.05 m，共 8×17 = 136 个候选点。新范围来自任务 2 早期运行中实测的 61 个站位的分布（离面 0.52–0.63 m、横向 −0.37 到 +0.11 m）再留余量；0.35 m 与 `explore` 的安全下限相同。改网格的具体日期日志中没有记录。

**问题**：

- 2026-11-06，任务 2 场景 `t2_L1_barcode_pos_0008` 中条码贴在箱面一角，失败：腕部到条码平面 0.297 m，执行误差 0.117 m，而头部和腕部感知误差只有 0.013 m 和 0.007 m，说明感知没问题，是手臂够不到。
- 2026-11-07 回顾任务 3 流水线时发现：两个条码的站位都选在网格最远的 0.70 m 边界上（`face_dist 0.7, lat −0.3`）。"关节余量最大"偏好站得远，搜索上限在起作用；站得远手臂伸得直，对行走漂移更敏感。
- 站位只在水平面上搜索，不会下蹲；当时的测试集规定条码中心离地不超过 1.3 m，只有"桌上并排两箱"一种场景。

这些问题在 v5–v10 中逐步解决。

### v4 · 2026-11-07 · 全身动作独立测试与下蹲标定（11 档）

**背景**：需求方要求先排除全身控制器本身的干扰，再让 agent 使用下蹲。

**做法**：新增一个不涉及 agent 的全身动作测试脚本（仿真仓库，`body_motion_eval.py`）：机器人传送到空地，分三段测试：下蹲（高度指令从 0.74 逐级降低再站起）、弯腰（在站立和最深稳定下蹲高度上给躯干俯仰指令）、抬臂（在骨盆坐标系的前向 / 高度网格上给枪口瞄准目标，站立和下蹲各测一遍）。每个姿态记录状态（ok / fell / collision）、骨盆高度和俯仰、脚漂移、枪口瞄准误差，并渲染侧视图。

**结果**（仿真实测）：

| 测试 | 结果 |
|---|---|
| 下蹲：指令 0.65 / 0.55 / 0.45 / 0.35 / 0.25 | 骨盆 0.6216 / 0.5156 / 0.4273 / 0.3277 / 0.2495 m，俯仰 4.15 / 12.7 / 20.74 / 25.52 / 26.09°，脚漂移 ≤ 0.1 mm，全部 ok，站起回到 0.7465 m |
| 弯腰（站立）：俯仰指令 0.2 / 0.4 / 0.6 rad | 躯干俯仰 11.56 / 26.91 / 43.07°，脚不动、不摔；下蹲 0.45 + 俯仰 0.4 也 ok |
| 站立抬臂，目标骨盆系 z = 0.62（枪口离地 1.398 m） | 误差 11.6 cm / 14.5°，够不到 |
| 站立抬臂，z = 0.45 / 0.2 / 0.0 / −0.1 | 误差 4 mm / 0.8 mm / 5.9 mm / 3.6 mm |

随后按同样方法做正式标定：站立不动、双臂 home 位，11 档高度指令各渐变 1.0 s、保持 2.0 s，记录骨盆高度、俯仰和全部关节状态，写入 `assets/gear_body_calibration.json`（表见"当前实现"第 4 节）。`robot.yaml` 新增 `body` 段：`calibration` 指向该文件，`ramp_s = 1.0`、`settle_s = 2.0` 与标定时相同。

**同时发现的问题**：下蹲 + 低位抬臂的截图里，手臂直接穿过了大腿和膝盖。根因是仿真机器人模型关闭了自碰撞（`enabled_self_collisions=False`），仿真不会报这种碰撞，真机上会撞。另外下蹲时膝盖会前移约 0.29 m，到箱子的距离要重新算。决定：不依赖仿真，由 agent 规划下蹲站位时自己检查手臂与腿、膝盖与箱面的间隙（v5）。

### v5 · 2026-11-07 · 全身模型 `BodyModel`，站位规划扩展到下蹲档位

**修改**：

- 新增 `g1agent/tools/body.py`：用标定档位 + `get_robot_geometry` 的碰撞点（凸包顶点用 `scipy ConvexHull`）计算机器人到箱面的间隙、手臂（elbow / wrist / hand）与腿（hip / knee / ankle）的间隙。初版在 `face_clearance` 里从 `toolbox` 导入 `horizontal_distance_to_face`，随即改为在 `body.py` 内联计算（日志没有写明原因）。
- `plan_stance` 由"只在站立高度搜索"改为依次尝试各实测档位（站立优先）；每个候选检查预备位和最终位 IK、身体到箱面 ≥ 3 cm、手臂离腿 ≥ 3 cm（`clearance_margin_m = 0.03`，配置注释给出的依据是"运动测试中最大手臂瞄准误差 8 mm + 放置距离容差 0.02 m，向上取整"）；按标定计算下蹲时骨盆的前移量，修正行走目标点（`foot_shift`）。
- 新增工具 `set_base_height`（蹲下 / 站起）。
- 新增 `_walk_guard`：下蹲状态**拒绝**行走（这时还不会自动站起）。依据：GR00T 在 MuJoCo 中高度 0.60 时 7 s 只走了 0.38 m，站立时能走 1.09 m。

### v6 · 2026-11-07 · 最低处的码规划失败：预备位退距、先解最终位、臂长预筛、头部宽度只要求 `min_m`

**问题**：在 CPU 上用实测几何、标定和离线机器人桩验证三个码：

| 码高 | 结果 |
|---|---|
| 1.10 m（顶层） | 站立可行，离面 0.65 m，箱面间隙 12.3 cm，手臂离腿 32.9 cm，耗时 38.7 s（IK 拒绝 74 个） |
| 0.57 m（中层） | 蹲到 0.60 档（骨盆 0.56 m），手臂离腿 5.7 cm |
| 0.19 m（底层） | 用头部估计的码宽时规划失败；用腕部实测码宽时蹲到最低 0.25 档可行，但关节余量很小 |

此外每个码规划要 40–150 s，因为每个下蹲档都要在 136 个站位上解 IK。

**根因**：头部估计的码宽偏大，算出站距 0.30 m；规划要求同时够到"站距 + 0.10 m"的预备位，即 0.40 m，深蹲时手臂够不到。而预备位只是为了让腕部相机先看清条码，真正的约束是"不比可能的最大站距更近"，+0.10 m 应当只是优先而非硬性要求。

**修改**：

1. 预备位多退 0.10 m 只作优先，够不到时依次退 0.05 m、0（`pre_gap` 记入结果）。
2. 先解最终位 IK，失败立即跳过该候选。
3. `BodyModel.arm_reach`：手臂链相邻帧原点距离之和作为肩到枪口距离的上界，预先排除肯定够不着的站位（只预筛不决策，`rejected` 新增 `reach` 计数）。改后顶层码 48.6 s。
4. 码宽来自头部时，头部宽度是上界，真实站距落在 [`min_m`, offset] 之间。站位只要求够到 `min_m`（0.12 m）和尽可能远的预备位；头部估计的站距本身够不着时不否决站位。改后顶层码 35.9 s，`reach` 预筛拒绝 11 个，IK 拒绝 33 个。

### v7 · 2026-11-07 · 第一轮堆叠运行：抬臂残差如实报告；代价改为"手臂伸展最少"

**问题**：第一轮三箱堆叠的 rule 运行中，顶层码站位按"关节余量最大"选在离面 0.70 m 处，手臂几乎伸直，执行后偏差 9 cm，而 `move_arm` 仍然返回 ok。

**根因**：一是代价函数偏好远站位（同 v3）；二是 `move_arm` 修正结束后不论残差多大都报 ok，掩盖了执行误差。

**修改**：

- `move_arm` 最终残差超过 `correction_tol` 时返回 partial（不阻断流程，但如实报告）。
- 站位代价改为 `(−gap, 伸展率, −关节余量, travel)`：以最终位的手臂伸展率（|枪口 − 肩| / `arm_reach`）最小为主，关节余量为次。
- 间隙检查从"手臂对腿"扩展到"手臂对腿、骨盆、躯干"（即现在 `legs` 集合里的 `TRUNK` 三个 link）。

**效果**：离线顶层码站位改为离面 0.45 m。第二轮运行中，顶层码预备位偏 2.9 cm / 3.2°（第一轮 9 cm），`move_arm` 正确地报为 partial。但"伸展最少"带来了新问题，见 v10。

### v8 · 2026-11-07 · 鲁棒站位：停偏 ±3 cm / ±1.7° 仍够得着才采用

**问题**：第二轮中，中层码规划出的姿态关节余量为 0。机器人第一次在完整任务中下蹲（0.65 档）后，IK 报告预备位不可达，只差 3 mm / 1.7°。

**根因**：真实行走允许 3 cm / 1.7° 的到位误差，下蹲后的骨盆也与标定略有不同；规划只要求精确站点可达，任何偏差都可能让贴边的站位失效。原流程预备位失败即放弃该码，不再重规划。

**修改**：`plan_stance` 先收集每档所有候选并按代价排序，再对排名靠前的候选在到位容差下（新配置项 `stance_robust_pos_m = 0.03`、`stance_robust_yaw_deg = 1.7`，即仿真器移动指令的停止容差）检查预备位和最终位是否仍可达，保留第一个通过的；`rejected` 新增 `not_robust`。所有候选都不鲁棒时，这一版**退回**只在精确站点可达的最优站位，结果中标 `robust=False`。

**效果**：离线顶层码 57.2 s，站立离面 0.5 m，鲁棒；中层码 61.4 s，蹲到 0.60 档、离面 0.55 m，鲁棒。第三轮中层码一次扫到，下蹲后骨盆 0.562 m，与标定的 0.56 m 一致，这是第一个在下蹲状态下扫到的码。底层码（0.19 m）只得到 `robust=False` 的站位，后续处理见 v10。

### v9 · 2026-11-07 · 抬臂闭环修正发散：退回最好的指令

**问题**：第三轮顶层码的最终抬臂：枪口偏离目标 36 cm / 35°（第二轮同一位置约 3 cm）。重试时又偏到 35 cm / 23°，下一次 `move_arm` 调用才回到 2.1 cm / 0.6°，随后扫到。代码注释记录的误差序列是 3.5 → 18 → 36 cm。

**根因**：GR00T 会用腰部去平衡抬起的手臂，底座和躯干随手臂一起动，肩部位置随之变化；满增益修正 `p_cmd += err` 在这种耦合下会过冲并发散。当时的站位又是"伸展最少"选出的折叠姿态、关节贴着限位（v10 中进一步分析）。

**修改**：`_move_arm` 记录目前为止最好的一次指令；某次修正让误差变大时，退回最好的指令并停止修正。同时在腕部细化中加入一致性门限（偏离超过一个标签宽度就拒绝，属于感知部分）：那次发散时腕部相机看错了地方，把标签位置从 1.10 m 改成了 1.70 m。

### v10 · 2026-11-07 · 可操作度代价；只能精确停位的站位判为不可达

**问题 1（姿态别扭）**：从顶层码的视频截帧看到，手肘抬到头侧、向外张开，小臂折回脸前再把枪口朝前（"鸡翅膀"姿态）。这不是为了避开箱子——规划只检查终点姿态是否可达、是否碰撞，不规划中间路径。根因是 v7 的"伸展最少"：顶层站位只离箱 0.45 m，手臂只能在头旁折起，多个关节贴着限位（余量 0）。我们推测这种折叠姿态也是 v9 中修正发散到 36 cm 的原因之一：关节贴限位时躯干一晃，修正就跟不上。

**修改 1**：用机器人学标准指标——枪口位置雅可比在 7 个手臂关节上的 Yoshikawa 可操作度 `sqrt(det(Jp Jpᵀ))`——选手臂姿态。它在完全伸直和完全折叠时都很低，中间的自然姿态最高，能同时避开两种极端。代价改为 `(−gap, −可操作度, travel)`。离线结果：顶层码站立离面 0.65 m，伸展率 0.86，可操作度 0.038，手臂离腿 16.8 cm；中层码蹲到 0.65 档，离面 0.5 m，可操作度 0.054。

**问题 2（不鲁棒站位的处理）**：第三轮底层码（0.19 m）按 v8 的退回策略拿到一个非鲁棒站位（最深 0.25 档，离面 0.50 m，横向 −0.35 m）。下蹲成功（骨盆 0.25 m），但预备位 IK 差 5 mm / 3.3° 不可达；站起后重规划得到同一个站位，再试又差 5 mm / 3.2°，最终失败。规划是确定性的，机器人位置几乎没变，重试不会有不同结果。需求方同时要求：条码低于机器人可达下限时，agent 要能检测出来、报错给出原因、写入日志，然后继续扫下一个。

**修改 2**：只能精确停位才够得着的站位不再尝试：条码标为 `unreachable`，原因区分"太低 / 太高"，附骨盆高度和鲁棒容差，并注明 "reachable only with an exact stop"；第一个通过间隙检查的候选只作为 `fallback` 用于这条说明。

**场景调整**：离线测试底层码（头部码宽 0.054 m）：0.27 m 需蹲到 0.30 档（骨盆 0.28 m），鲁棒，88 s；0.30 m 蹲到 0.35 档，鲁棒，67 s。鲁棒可达下限约 0.27 m，于是把场景 0000 的底层码贴到 0.30 m；场景 0001 保留 0.19 m，专门测"够不到就报错"。

**效果**：下一轮（第 5 轮）中，底层码贴到 0.30 m 的场景 rule 扫到 3/3，深蹲扫底层码时骨盆 0.33 m，枪口误差 1.4 cm；同一场景 LLM 编排扫到 2/3（中层码第一次没读出，LLM 没有按规定重试就跳到了下一个码）；底层码保留 0.19 m 的场景中，该码被正确报为不可达（`barcode bc2 unreachable, too low: label centre at z=0.19 m; no stance even at the deepest measured squat (height command 0.25, pelvis z 0.25 m) ...`），跳过后继续扫下一个。再下一轮验证（scan 工具读不出时自己重试一次，并加入 v12 的规划提速）中，rule 和 LLM 编排都扫到 3/3。

### v11 · 2026-11-07 · 行走前自动站起、到达站位自动下蹲

**当时的版本**：v5 的 `_walk_guard` 在下蹲状态下拒绝行走，提示调用方先 `set_base_height` 站起；`approach` 到达站位后也不会自己下蹲，下蹲要由编排器按顺序调用。

**问题**：LLM 编排的一次运行约 8 分钟就结束，扫到 0 个码，LLM 总结为"所有条码都不可达或扫码失败"。代码注释记录的现象是：LLM 在行走前就蹲下，到了站位后却没有再蹲。

**根因**：站起 / 下蹲是站位规划结果的一部分，却依赖编排器按正确顺序调用；LLM 不一定遵守这种隐含顺序。

**修改**：站起和下蹲改由工具负责：行走工具在下蹲状态下自动站起（手臂必须在 home 位，否则 fail，避免站起时手臂扫过伸展的腿）；`approach('stance_target')` 到达后按站位的 `base_height` 自动下蹲。

### v12 · 2026-11-07 · 规划提速一：按代价顺序惰性检查 + IK 停滞检测

**问题**：cProfile 显示中层码（0.57 m）一次 `plan_stance` 耗时 66 s：

- 约 72% 在 `self_clearance`：手臂几千个点要对 15 个腿和躯干部件的凸包逐面算距离，而且每个 IK 可解的候选都要算（另一次对 0.30 m 码的 cProfile：72 次调用共 17.7 s，每次约 0.25–0.28 s）；
- 约 26% 在 IK，其中 95% 的迭代花在最终失败的求解上（在 0.30 m 码上统计：失败 705 次共 211500 步，每次都跑满 300 步；成功 205 次共 11815 步）。

**修改 1（惰性检查）**：排序标准（预备间隙、可操作度、行走距离）都不依赖间隙和鲁棒性检查的结果，所以先对所有 IK 可解的候选排序，再从最好的开始依次做箱面间隙、手臂-腿间隙、鲁棒性检查，第一个全部通过的即为答案。检查次数从 168 次降到几十次。保留旧版本"全部检查完再挑"在 4 个码高上逐一对比，选出的站位完全一致：

| 码高 | 旧版 | 惰性检查 |
|---|---|---|
| 1.10 m | 55.9 s | 3.3 s |
| 0.57 m | 75.2 s | 27.2 s |
| 0.30 m | 89.9 s | 34.7 s |
| 0.19 m | 判为不可达 | 判为不可达 |

**修改 2（停滞检测）**：`ArmIK._solve_from` 增加 `stall=(window, eps)`：加权误差（位置 + `aim_weight` × 角度）在 `window` 步内下降不到 `eps` 即判失败。阈值依据 0.30 m 码上的实测轨迹：成功求解的迭代数 p50 41、p95 160、最大 224；比较了 window 25 / 50 / 100 与 eps 2 / 0.5 mm 的组合（例如 window 50、eps 0.5 mm 会误判 205 次成功中的 12 次），选 window 100、eps 0.5 mm：失败求解中位数在第 118 步停下，205 次成功中只有 1 次（0.5%）被误判。配置为 `ik.plan_stall_window` / `plan_stall_eps_m`，只用于 `plan_stance` 中的 IK 调用，执行时驱动手臂的 IK 不加。

**效果**：4 个码高选出的站位不变；规划时间 1.10 m 56 → 3 s，0.57 m 75 → 22 s，0.30 m 90 → 27 s，0.19 m（判不可达）99 → 49 s。71 个测试通过。剩余时间主要花在排序靠前、但手臂离腿过近被淘汰的候选上（每个约 0.28 s）。

### v13 · 2026-11-07 · IK 增加 roll 约束：扫码画面水平轴平行于条码长轴

**问题**：需求方指出扫码时扫码相机画面的边框和条码不平行，真实扫码枪在这种情况下读不出来。

```{figure} images/motion_wrist_roll_free.jpg
:width: 400px

加 roll 约束之前一次演示运行中的腕部扫码相机画面：条码与画面边框明显倾斜。仿真扫码器能读出这样的码，真实扫码枪读不出。
```

**根因**：这是设计漏洞。v0 的 IK 只约束枪口位置（3 维）和扫描方向（2 维），枪绕自身轴的转角是自由的，画面边框和条码之间可以是任意角度。仿真里照样读得出来，是因为仿真扫码器把图像旋转 0、15、…、75° 各解码一次，相当于全向读码；zbar 实测只能读与图像轴夹角约 ±17° 以内的条码。

**修改**：

- `ArmIK.solve` 增加 `roll_axis` 参数：要求扫码相机图像的 x 轴（枪口帧的 −y 方向，用正运动学核对 `img_x_in_tip = [0, −1, 0]`）平行于条码长轴，正反都算平行，共 6 个约束；新增 `roll_target`、`roll_error_deg`。
- 工具层新增 `_long_axis(bc)`：条码长轴在面内、与条纹垂直，即 code128 扫描线要走的方向；规划时取头部估计，到预备位后用腕部检测更新。腕部目标中带上 `long_axis`，由 `_roll_in_root` 转到根系。
- 站位规划、鲁棒性检查和 `move_arm` 执行都使用完整约束；roll 约束下够不到的码按"不可达"报告。
- `check_wrist_placement` 增加 roll 偏差检查，阈值 10°（与枪口角度阈值一致）。

**效果**：离线测试三个目标方向都收敛（误差 1.8–1.9 mm）；带 roll 的离线规划：1.10 m 码 8 s（站立），0.57 m 码 26 s（蹲到 0.55 档）。3×3 箱墙运行中所有摆位的 roll 偏差为 0.7–6.0°。

**代价**：

- 深蹲码的站位规划每个要 35–112 s（3×3 运行中 5 个码分别为 11 / 112 / 37 / 10 / 35 s），整轮约 20 分钟，引出了 v14 的并行化。
- 最低可扫码高从约 0.27 m 升到约 0.30 m。四面堆叠场景中正面左下码中心高 0.30 m，被判为"太低，最深下蹲也够不着"；它正好卡在边界上，站位一换就可能够不到。决定把场景底层码统一提到 0.33 m 以上留余量，代码不改。
- 抬臂到高处时 2–3 cm 的位置残差仍在，放置检查会判不合格，但码都能读到。

### v14 · 2026-11-07 · 规划提速二：单候选纯函数 + 16 进程池

**问题**：加 roll 约束后每个深蹲码规划 35–112 s，而这些时间全是各网格点之间互相独立的 IK。

**修改**：

1. 新文件 `g1agent/tools/stance_eval.py`：把单个候选的评估（可达性预筛、所需位姿 IK、可操作度评分）写成纯函数（`init_worker` / `eval_candidate` / `eval_batch` / `target_in_root` / `roll_in_root` / `manipulability`），串行和进程池调用同一个函数。
2. 确定性：`ArmIK.solve` 增加 `rng` 参数，每个候选的随机重启用自己的生成器，种子由（seed，下蹲档位，网格编号）决定。离线对比 3 个码高，并行和串行选出的站位和各类淘汰计数完全一致；`tests/test_stance_pool.py` 检查串行与 2 进程选出的站位相同（73 个测试通过）。
3. worker 用 spawn 启动（不继承父进程里检测器的 CUDA 上下文），并设 `PR_SET_PDEATHSIG` 随父进程退出。原因：以前曾有 fork 出来的子进程在父进程退出后继续存活，一直占着共享 GPU 的显存。
4. 进程数由实测决定：

| 进程数 | 三个码的规划时间 |
|---|---|
| 串行 | 10 / 30 / 104 s |
| 8 | 5 / 12 / 53 s |
| 16 | 5 / 10 / 47 s |
| 32 | 5 / 9 / 50 s |

16 之后不再变快，剩下的时间在主进程串行的间隙检查和鲁棒性检查上，所以 `plan_workers = 16`。

**随后的问题**：第一次仿真验证时，每个 worker 又各自按库默认值起了几十个 BLAS 线程。服务器本来负载就高，平均负载从 108.53 升到 191.15，整个运行（包括仿真器）都被拖慢：rule 的 `approach` 用了 135.1 s，`plan_scans` 用了 34.6 s，520 s 时还一个码都没扫。我们停掉这次运行。

**修改**：创建进程池期间临时把 `OMP_NUM_THREADS`、`OPENBLAS_NUM_THREADS`、`MKL_NUM_THREADS` 设为 1，并立即拉起所有 worker（环境变量在 worker 导入 numpy 时读取），之后恢复原值。

**效果**：3×3 箱墙上，LLM 编排 5/5 顺序正确，规划总时间 201 → 100 s，总用时 912 → 869 s；rule 5/5，规划 205 → 107 s，总用时 1177 → 1056 s。总用时只减少 5–10%，因为大部分时间花在仿真里的行走、抬臂和下蹲上，而且仿真比实时慢。

### v15 · 2026-11-07 · 把 LLM 的调用顺序问题收进工具：站距不被编造、腕部目标自动补算

这一组改动都发生在 2026-11-07 的 LLM 编排运行中，开发日志中没有逐条定位到具体时刻，依据来自代码注释和当天的运行记录。

- **`compute_wrist_target` 忽略编造的 offset**：LLM 自己编了一个 offset，从 0.14 m 处扫码。先改成"自适应模式下传 offset 就拒绝"，结果 LLM 跳过 `compute_wrist_target` 直接调 `plan_stance`，因为没有腕部目标而放弃了这个条码（日志中记录到 LLM 先调 `plan_stance` 后调 `compute_wrist_target` 的顺序错误）。现在自适应模式下静默忽略传入的 offset，并在返回中注明 `offset_ignored`。
- **`_plan_stance` 自动补算腕部目标**：当前条码还没有腕部目标时，`_plan_stance` 自己调用 `compute_wrist_target`，不再因调用顺序失败。
- **头部宽度只能让站距变远**：头部标签框可能偏大（检测框约 +20%），也可能偏小——矫正面上只框到一部分时，8 cm 的码被估成 3.5 / 4.9 cm，站距被截到 0.12 m，结果读不出。现在 `_standoff` 在宽度来自头部时取 `max(d, offset_m)`，只有腕部实测宽度能让扫码枪比 0.20 m 更近。

### v16 · 2026-11-07 · 多面任务：站位障碍只取当前面，其它面由外轮廓代表

**当时的版本**：`_known_faces` 返回最近一次检测到的箱面，加上所有条码各自保存的面副本。

**问题**：四面堆叠任务第 10 轮中，左侧面的条码全部被判为"够不着"。

**根因**：多面任务中，其它面的条码面副本是在旧里程计下拟合的，里程计漂移之后这些位置已经过时，`plan_stance` 却把它们当作障碍物，挡住了真正可行的站位。

**修改**：多面任务中条码的面副本只保留当前面的；其它面的避障交给重配准后的堆叠外轮廓（`_footprint` / `_root_clearance`），`plan_stance` 和 `_move_axis` 都检查它。`tests/test_task3_logic.py` 新增测试，验证多面任务里障碍只取当前面。

**效果**：第 11 轮左侧面能规划出站位。

### v17 · 2026-11-07 · `move_arm` 必须先到达规划的站位

**问题**：第 11 轮右侧面扫到 2/3，bc4 失败：LLM 在 `plan_stance` 之后没有走过去就直接抬手，差了约 10 cm，最后放弃了这个条码。

**根因**：`plan_stance` 只写入 `stance_target`，机器人并没有移动；工具层没有要求"抬臂前必须到达站位"。

**修改**：

- `_move_arm` 的前置条件：任务 3 中必须已为当前条码规划了站位，且已通过 `approach` 到达（`_stance_reached is stance_target`）。
- `_walk_guard` 在任何行走前清除 `_stance_reached`；`approach('stance_target')` 成功时设置它。
- 规划成功不算"已尝试"，尝试从走向站位开始计；`next_barcode` 据此拒绝跳过从未尝试过的条码。

### v18 · 2026-11-07 · `move_arm` 按视锥判定 partial / fail；放置检查只测量不决定

**问题**：在两个机器人初始朝向与箱面不平行的四面堆叠场景中，LLM 编排各扫到 11/12，漏掉的码暴露出两个问题：

- 抬起的手臂停在偏离目标 0.21 m / 23° 处，`move_arm` 仍报 "partial: go on"，结果扫不出；
- `check_wrist_placement` 返回 fail 时，LLM 把它当成停止信号，在轴偏 2.0–2.4 cm（容差 2 cm）的情况下重新走了 4 次站位，一次也没扫码。随后在纸箱两面场景（`stack4c_L0_0001`）中又出现一次：LLM 在预备位（0.35 m，目标 0.20 m）上就做检查，被告知重新规划站位，结果重规划了 4 次，始终没移到最终位（据代码注释）。

**根因**：v7 的 partial 只看残差大小，不看残差是否还允许扫码；放置检查的 fail 被 LLM 理解为"放弃并重来"，而不是"可以继续扫"。

**修改**：

- 新增 `_label_view(t, off, label_width)`，计算标签相对枪口的横向距离、沿轴距离、半视野宽度、`in_view` 和 `label_in_view`。`move_arm` 残差超限时，标签中心不在扫码视锥内就返回 fail（"IK: target not reachable from this stance…"，需重新规划站位）；在视锥内才返回 partial。
- 任务 3 中 `check_wrist_placement` 只测量：在站距处且标签在视野内时返回 partial 并要求扫码；还没执行过本条码的 `final` 时提示先 `move_arm('final')`；已执行过 `final` 仍未到位时提示重新规划站位。

### v19 · 2026-11-07 · 低视角补看：所有比站立更清楚的下蹲档位取并集

**当时的版本**：任务 3 首次 `plan_scans` 之后，`_low_view_update` 只挑一个"采样最细"的下蹲档位蹲下补看一次。引入它的原因：头部相机从约 1.2 m 高处向下约 47° 看，宽或高的面上靠下的标签距离远、角度斜；3×3 箱墙中站立时漏检 1–2 个底排标签，蹲下后找到了。

**问题**：纸箱两面场景（`stack4c_L0_0001`）第一轮，正面 3 个真码只扫到 2 个，底层真码 F4ABLFFEB 没有扫到（计划中的底层条码 bc3 没读出）。

**实验**：在正面和背面的观察位姿上逐档采集头部图像（11 档），跑真实的检测流水线，看每档检出哪些真码。22 个视角共 76 个检测，其中 30 个靠近真值：

| 档位（正面） | 检出的真码 |
|---|---|
| 0.70 / 0.74 | 只有顶层 F4ATLFRVG |
| 0.40–0.60 和 0.30 | 底层 F4ABLFFEB 和中层 F4AMRF38K |
| 0.35 / 0.65 | 只有 F4AMRF38K |
| 0.25 | 没有检出真码 |

背面最深的 0.25 档一个真码都没检出。代码注释中这次离线分析的汇总是：最深的下蹲（底部分辨率最高）一个下排标签都没找到，0.40–0.60 档找到全部 4 个，0.35 档只找到 1 个。

**根因**：开放词表标签检测器的召回率依赖视角，并且不随高度单调变化；"分辨率最高"的档位不一定检得出来。

**修改**：对每个标定档位，用 `q_at`、`foot_shift` 和 `T_root` 预测双脚不动时的头部相机位姿，把当前面所有底边角点投影到图像中；所有角点都在图像内（留 5% 边距）时评分 `fx / 最远角点深度`。选出评分高于站立评分的**所有**档位，从高到低逐个下蹲、执行 `plan_scans(_update=True)`、合并新标签，最后站起（站起失败抛 `RuntimeError`）。

**效果**：离线分析中，把各档位结果合并后，两个面的标签全部找到。修复后重跑同一场景，正面 3 个真码在第一次规划时全部进入计划，上一轮漏掉的底层码 F4ABLFFEB 扫到了；代价是第一阶段多花约 2 分钟逐档下蹲补看。

## 已知限制

- **最低和最高可扫高度**：加 roll 约束后，最低可扫码高约 0.30 m（深蹲 0.25–0.35 档），0.30 m 正好在边界上，场景约定底层码不低于 0.33 m。站立抬臂时枪口离地 1.398 m 的目标已够不到（v4 测试误差 11.6 cm），离线估计保持水平持枪时枪口最高约 1.36–1.40 m。
- **抬高手臂的残差**：抬臂过肩时 2–3 cm 的位置残差一直存在，放置检查常判不合格（但码通常能读出）。根因是 GR00T 用腰部平衡抬起的手臂，底座和躯干随手臂移动；当前的处理（满增益修正 + 退回最好指令）只是防止发散，没有消除残差。
- **只检查终点姿态**：站位规划和间隙检查只针对预备位和最终位的静态姿态，不规划手臂从 home 到目标的中间路径；仿真关闭了自碰撞，路径上的手臂-腿碰撞在仿真中看不到。
- **自碰撞检查范围**：`self_clearance` 只检查扫码臂（elbow / wrist / hand）对腿和躯干，肩部 link 和另一只手臂不参与；结果是距离下界，偏保守。link 的选取依赖名字关键字表。
- **标定条件**：下蹲档位是在双臂 home 位、静止站立时测的；完整任务中行走后下蹲的骨盆姿态会与标定略有差别（v8 中观察到），抬臂时 GR00T 还会用腰部平衡手臂。规划只使用 11 个实测档位，不在档位之间插值；躯干俯仰指令虽然实测可用（v4），但当前不参与规划。
- **鲁棒性检查**：6 种扰动逐个施加（±x、±y、±yaw），不检查组合偏差；它和两项间隙检查都在主进程串行执行，是并行化后剩余规划时间的主要部分（16 进程时三个码仍需 5 / 10 / 47 s）。
- **网格搜索**：站位朝向一律正对箱面，不搜索 yaw；网格步长 0.05 m 和最近行 0.35 m 没有数值依据；`max_face_dist_m` 和 `lateral_range` 依据的是任务 2 单箱场景的站位分布。
- **没有依据的参数**：IK 的 `dls_lambda`、`max_iters`、`tol_pos`、`tol_angle_deg`、`restarts`；`limits.*`（设计文档只列出了这些防护项，没有给出数值来源）；`arm_motion.duration_s`、`settle_s`；`approach.move_timeout_s`、`max_segments`；`_move_axis` 的 11 个路径采样点；URDF 中的 effort / velocity 字段。这些在配置注释中都没有给出数值来源。
- **死代码**：`Toolbox._manipulability` 有定义但没有调用。
- **只在仿真中验证**：以上数据全部来自 Isaac 仿真（个别行走数据来自 MuJoCo），真机上的行走停位、下蹲稳定性和抬臂残差尚未验证。
