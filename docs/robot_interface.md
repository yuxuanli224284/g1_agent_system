# 机器人接口与仿真服务器

本页介绍 agent 系统与机器人之间的那一层：harness 通过什么接口控制机器人，仿真器怎样作为独立进程提供这个接口，真值怎样与 agent 隔离开，以及这一层用到的公共工具（配置加载、序列化、坐标变换）和 M1/M2 自检脚本。路径都相对仓库根目录 `g1_agent_system/`。

## 作用与在系统中的位置

agent 系统分成几个独立进程：harness 进程（工具层、编排器、感知）、仿真服务器进程（Isaac 仿真）和 VLM 服务进程。本页的组件连接前两者。

```
 harness 进程（g1_agent 环境）                     仿真服务器进程（unitree_sim_env 环境）
 ┌───────────────────────────────┐                ┌───────────────────────────────────────────┐
 │ Toolbox / 编排器               │                │                                           │
 │   └ SimRobot (RobotInterface) │── robot port ─▶│ REP socket rs → 方法表 R（不含真值）        │
 │        └ RpcClient (REQ)      │  31555 + 10·i  │                                           │
 │ run_task.py 中的 EvalClient    │── eval port ──▶│ REP socket es → 方法表 E（需令牌，含真值）  │
 │   （回合控制、阶段、录像）       │  31556 + 10·i  │        │                                  │
 └───────────────────────────────┘                │        ▼                                  │
       消息格式：msgpack + numpy                   │ G1BarcodeEnv（Isaac 仿真） + Episode       │
       （g1agent/common/serialization.py）         │  （阶段日志、扫码日志、合成录像）            │
                                                  └───────────────────────────────────────────┘
```

| 对接方 | 用到本组件的什么 | 方向 |
|---|---|---|
| 工具层 `g1agent/tools/toolbox.py` | 只拿到一个 `SimRobot`（`RobotInterface` 实现），通过它取图、读里程计、行走、动手臂、扫码、画标记 | harness → robot port |
| 运行脚本 `scripts/run_task.py` | 持有 `EvalClient`：重置场景、打开真值标记、开始/结束录像，把 harness 日志里的阶段和事件转发到 eval port | 运行器 → eval port |
| 评测器 `g1agent/eval/evaluator.py` | 读真值、阶段状态（`stages["task1_end"]`）、带真值的扫码日志；`gt_standoff_fn` 反过来调用评测器的真值站距规则 | 双向 |
| 运动学 `g1agent/kinematics/*` | `check_m1_m2.py` 用它核对客户端 FK 与仿真器给出的相机位姿；站位规划用 `get_robot_geometry` 返回的碰撞点 | 读取 |
| 可视化 `g1agent/viz/*` | 仿真服务器内部用 `MarkerLayer`（USD 标记）和 `Compositor`（合成录像） | 服务器内部调用 |
| 仿真包 `g1_barcode_sim` | `G1BarcodeEnv` 提供实际的仿真、行走控制器、扫码枪相机和真值 | 服务器内部调用 |
| 配置 `configs/system.yaml`、`configs/robot.yaml` | 端口、超时、令牌变量名；harness 侧的机器人几何、IK、限位参数 | 所有进程读取 |

设计上有两条硬性要求，都来自需求文档：

1. **接口里不含真值。** harness 能调用的方法只描述"机器人自己能知道的东西"：相机图像、关节角、里程计、自身几何。场景中箱子和条码的真实位置只在 eval port 上提供。
2. **仿真和真机用同一个接口。** `RobotInterface` 是抽象类；仿真客户端 `SimRobot` 是它的一个实现，将来的真机（DDS）实现也要提供同一组方法。

## 当前实现

### 1. 公共工具 `g1agent/common/`

#### 1.1 配置加载 `g1agent/common/config.py`

所有进程都通过这个模块读 `configs/*.yaml`。

| 名称 | 作用 |
|---|---|
| `ROOT` | 仓库根目录，由本文件位置向上三级算出 |
| `CFG_DIR` | `ROOT/configs` |
| `_read(name)` | 用 `yaml.safe_load` 读 `configs/{name}.yaml` |
| `merge_inherited(own, parents)` | 合并父配置和本文件配置（规则见下） |
| `_resolve_paths(name, cfg)` | 只处理 `system`：`sim_server.sim_root` 优先取环境变量 `G1_SIM_ROOT`，否则把文件里的相对路径按仓库根目录解析成绝对路径 |
| `load(name, _seen=())` | 递归加载，处理 `inherits:` 继承并检测循环 |
| `load_all()` | 依次加载 `system, llm, robot, perception, task1, task2, task3, viz` 八个配置，返回 `{名字: dict}` |

**继承规则**（`merge_inherited` 的 docstring）：

1. 按 `inherits` 列表的顺序取每个父配置的顶层键。
2. 两个父配置给同一个键设了不同的值，而本文件没有显式设这个键时，抛出 `ValueError("config key ... differs between A and B; set it explicitly")`。docstring 写明目的是不允许"silent 'last parent wins'"，也就是不让后一个父配置悄悄覆盖前一个。
3. 最后用本文件自己的顶层键覆盖（`merged.update(own)`）。合并只在顶层进行，不做深层合并。

**循环检测**：`_seen` 记录当前继承链，遇到已在链上的名字时抛出 `ValueError("config inheritance cycle: a -> b -> a")`。

#### 1.2 序列化 `g1agent/common/serialization.py`

RPC 消息的编码和解码，只依赖 `msgpack` 和 `numpy`。

| 函数 | 作用 |
|---|---|
| `_default(o)` | `np.ndarray` 先转成连续内存，再编码成 `{"__nd__": 1, "dtype": a.dtype.str, "shape": [...], "data": bytes}`；numpy 标量用 `.item()` 转成 Python 标量；其他不认识的类型抛 `TypeError("cannot serialize ...")`，不做静默转换 |
| `_hook(d)` | 解码时遇到 `__nd__ == 1` 的字典，用 `np.frombuffer(...).reshape(shape).copy()` 还原。`.copy()` 让结果是可写的独立数组，而不是指向消息缓冲区的只读视图 |
| `pack(obj)` | `msgpack.packb(obj, default=_default, use_bin_type=True)` |
| `unpack(b)` | `msgpack.unpackb(b, object_hook=_hook, raw=False, strict_map_key=False)`，允许非字符串的 map 键 |

为什么不用现成的 `msgpack_numpy`，见演进记录 v0。

#### 1.3 坐标变换 `g1agent/common/transforms.py`

**命名约定**：`T_a_b` 表示坐标系 b 在坐标系 a 中的位姿，它把 b 中的坐标变到 a 中。全仓库的变量都按这个约定命名，例如 `T_odom_root`、`T_root_headcam`、`T_world_tip`。

| 函数 | 输入 → 输出 | 实现 |
|---|---|---|
| `inv_T(T)` | 4×4 → 4×4 | 旋转取 `R.T`，平移取 `-R.T @ t`（刚体变换的解析逆） |
| `T_from_xy_yaw(x, y, yaw, z=0.0)` | → 4×4 | 绕 z 轴转 yaw，再平移到 (x, y, z) |
| `yaw_of(T)` | → float | `atan2(T[1,0], T[0,0])` |
| `wrap(a)` | → float | 把角度归一化到 [-π, π) |
| `apply(T, p)` | 点，单个 (3,) 或批量 (N,3) | `R p + t` |
| `rotate(T, v)` | 向量，单个或批量 | 只做旋转 `R v` |
| `angle_between(a, b)` | 两个 3 维向量 → 弧度 | `atan2(‖a×b‖, a·b)`；在 0 和 π 附近比 `arccos` 数值上更稳 |

### 2. 机器人接口 `g1agent/robot/interface.py`

`RobotInterface` 是一个抽象基类，列出 harness 能对机器人做的全部操作。模块 docstring 写明：仿真客户端和将来的真机（DDS）实现提供同样的方法；这里没有真值。

| 方法 | 参数 | 返回（据行内注释） |
|---|---|---|
| `get_robot_info()` | 无 | 关节名、相机内参、各类限制 |
| `get_head_obs()` | 无 | `rgb`、`depth`（米）、`K`、`T_root_headcam`、`timestamp` |
| `get_wrist_obs(side="left")` | 手的侧别 | `rgb`、`depth`、`K`、`T_root_wristcam`、`timestamp` |
| `get_joint_state()` | 无 | `names`、`q` |
| `get_odom()` | 无 | `T_odom_root` |
| `walk(vx, vy, wz, duration)` | 速度指令和时长 | dict |
| `move_root_relative(dx, dy, dyaw, timeout=10.0, pos_tol=None, yaw_tol=None, speed_scale=1.0)` | 相对当前根坐标系的位移目标 | dict |
| `stand(duration)` | 时长 | dict |
| `set_body_pose(height=None, torso_rpy=None, duration=1.0, settle=1.0)` | 骨盆高度指令（m）、躯干 roll/pitch/yaw（rad） | dict；下蹲、起立、调整躯干都在原地站立时进行 |
| `get_robot_geometry()` | 无 | 机器人各连杆的碰撞点（连杆坐标系） |
| `set_arm_joints(side, q, duration)` | 手臂关节目标 | dict |
| `set_gripper(side, opening)` | 夹爪开度 | dict |
| `scan()` | 无 | 触发手持扫码枪 |
| `draw_markers(markers, frame="odom")` | 标记列表 | 只在仿真中生效，真机上什么也不做 |
| `clear_markers()` | 无 | 清除标记 |

`move_root_relative` 的 `timeout=10.0`、`speed_scale=1.0` 和 `set_body_pose` 的 `duration=1.0`、`settle=1.0` 是接口默认值，代码注释没有写依据；工具层调用时一般会显式传入配置里的值（如 `robot.yaml body.ramp_s/settle_s`）。

### 3. RPC 客户端 `g1agent/robot/rpc.py`

`RpcClient` 是基于 ZeroMQ REQ socket 的同步 RPC 客户端。

- `__init__(host, port, timeout_s, token=None)`：地址 `tcp://{host}:{port}`，接收超时 `timeout_s * 1000` 毫秒，使用进程级共享的 `zmq.Context.instance()`。
- `_connect()`：新建 REQ socket，设 `RCVTIMEO` 为超时毫秒数、`LINGER=0`，然后连接。
- `call(method, **args)`：
  1. 构造请求 `{"method": method, "args": args}`，有令牌时加上 `"token"`。
  2. `pack` 后发送，阻塞等待回复。
  3. 只捕获 `zmq.error.Again`（接收超时）。超时后关闭旧 socket 并重新连接，再抛出 `TimeoutError("... no reply within N s")`。源码注释给出原因："a REQ socket is unusable after a missed reply"：REQ socket 严格按"发一次、收一次"轮转，错过一次回复后它就无法再发送。
  4. 回复是 `status == "error"` 且带 `traceback` 的 dict（服务器执行方法时抛了异常）时，抛出 `RuntimeError`，消息里带服务器给出的原因和完整 traceback。
  5. 不带 `traceback` 的 error 回复原样返回给调用方，例如令牌错误、未知方法，或 `get_wrist_obs` 的"no wrist camera"。这类错误属于正常的业务结果，由调用方判断。

### 4. 仿真客户端 `g1agent/robot/sim_client.py`

`SimRobot(RobotInterface)` 通过 RPC 连接仿真服务器的 robot port。docstring 强调它**故意没有任何真值方法**，评测相关的调用放在 `g1agent/eval/eval_client.py`。

- `__init__(instance=None)`：读 `config.load("system")["sim_server"]`。实例号取参数 `instance`，没传时取环境变量 `SIM_INSTANCE`（默认 `"0"`）。端口 = `robot_port + i * instance_port_stride`，超时 = `rpc_timeout_s`。不传令牌，因为 robot port 不需要。
- 每个接口方法都直接转发成 `rpc.call(同名方法, ...)`。客户端在发送前把数值参数显式转成 `float`（`move_root_relative` 的 `pos_tol`、`yaw_tol` 例外，原样传递，可为 `None`）；`set_arm_joints` 的 `q` 转成 `np.asarray(q, float)`；`set_body_pose` 中为 `None` 的参数保持 `None`，`torso_rpy` 转成三个 float 的列表。
- `draw_markers(markers, frame)` 在线路上用的参数名是 `markers_list`，与服务器方法表里 lambda 的参数名对应。
- 接口之外多一个 `ping()`，用于连通性测试。

### 5. 仿真服务器 `g1agent/robot/sim_server.py`

模块加载时读取 `SYS = config.load("system")["sim_server"]`、`VIZ = config.load("viz")`、`REC = VIZ["recording"]`，并把仿真包目录 `SYS["sim_root"]` 加进 `sys.path`。启动命令是 `python -m g1agent.robot.sim_server`，一般通过 `scripts/start_sim_server.sh` 启动（见第 8 节）。

#### 5.1 启动

`main()` 依次执行：

1. 从 `SYS["eval_token_env"]` 指定的环境变量（`G1_EVAL_TOKEN`）读评测令牌。没有令牌就以 `SystemExit` 退出，并提示先设置这个变量。
2. 构造仿真覆盖参数 `{"robot": {"mode_policies": SYS["mode_policies"]}}`；`viz.yaml` 的 `recording.cameras_enabled` 为真时再加 `{"camera": {"recording": {"enabled": True}}}`，打开两台录像相机。
3. `env = G1BarcodeEnv("scenario", headless=True, seed=0, config_overrides=overrides)`：场景模式、无界面、`seed=0`。
4. 创建 USD 标记层 `MarkerLayer(stage, VIZ["colors"], VIZ["sizes"])`，用 `state = {"gt_markers": False}` 控制是否显示真值标记。
5. 创建 `Episode(env, markers, Compositor(REC, colors, sizes), actual_state)`，把 `ep.on_step` 加进 `env.step_hooks`，仿真每走一个策略步就调用一次（用于录像取帧）。
6. 建两个 REP socket，分别绑定 `tcp://host:rport` 和 `tcp://host:eport`（端口都加上 `SIM_INSTANCE * instance_port_stride`），就绪后打印 `SIM_SERVER_READY instance=… robot=… eval=…`。启动脚本和批量脚本靠这一行判断服务器已可用。

#### 5.2 `main()` 内的辅助闭包

| 函数 | 作用 |
|---|---|
| `T_world_odom()` | 用 `env.frame_state()` 中的 `T_world_root` 和里程计估计 `T_world_root_odom` 算出 `T_world_root @ inv_T(T_world_root_odom)`，即里程计原点在世界系中的位姿 |
| `third_person_view(hfov=None)` | 第三人称截图。相机眼点和注视点（`viz.third_person.eye_offset/look_at`）按机器人当前 yaw 转到世界系；先刷新真值标记，`markers.show(True)` 后用 `env.render_aux` 渲染，再 `markers.show(False)`，最后多调用一次 `env.env.sim.render()`。注释说明最后这次渲染是"re-render without markers so no robot-camera buffer keeps them"，确保机器人相机的缓冲区里不残留标记 |
| `actual_state()` | 返回真实状态：`T_world_root`、`T_world_wrist_point`（即 `T_world_tip`）、`T_world_odom_now`、`T_world_headcam`、`head_K`、`head_size`、`scenario_id`、`t_sim`。docstring 说明它用轻量访问器 `frame_state()`，数值与 `get_obs` 相同，但"without touching the depth-noise stream"，即不消耗深度噪声的随机数流 |
| `_refresh_gt_markers()` | 真值标记关闭或没有场景时清空 `eval` 组；否则在 `eval` 组画四个标记：绿色真值观察位姿 `gt_root`（`task1_gt_obs_pose`）、绿色真值手腕目标 `gt_wrist`（`task2_gt_wrist_target`，为 `None` 时不画，注释说明任务 3 的"all"模式没有单一手腕目标）、黄色实际根位姿 `act_root`、黄色实际手腕位姿 `act_wrist`（方向取 `T_world_tip` 的 x 轴） |
| `harness_markers(ms, frame)` | 把 harness 发来的标记从里程计系（`frame="odom"`）或根坐标系转换到世界系：`pos` 做完整变换，`dir` 只做旋转，`yaw` 加上该坐标系的 yaw。docstring 说明里程计系是 harness 唯一的固定坐标系 |

#### 5.3 robot port 方法表 `R`

harness 只能调用这些方法。

| 方法 | 实现 |
|---|---|
| `ping` | `{"status": "ok", "t": timestamp}` |
| `get_robot_info` | 模块函数 `robot_info(env)`（见 5.7） |
| `get_head_obs` | 从 `env.get_obs()` 取 `rgb, depth, K`，`T_root_cam` 改名为 `T_root_headcam`，加 `timestamp` |
| `get_wrist_obs` | 模块函数 `wrist_obs(env, side)` |
| `get_joint_state` | `env.robot.data.joint_names`、`joint_pos[0]`（转成 numpy）、`timestamp` |
| `get_odom` | `T_odom_root` 取自 `get_obs()["T_world_root_odom"]`，这是仿真里模拟出来的里程计估计，不是真实位姿 |
| `walk` | `env.step(vx, vy, wz, duration)`，经 `strip_obs` 去掉返回中的 `obs` 字段 |
| `move_root_relative` | `env.move_relative(dx, dy, dyaw, timeout, pos_tol, yaw_tol, speed_scale)` |
| `stand` | `env.stand_still(duration)`，经 `strip_obs` |
| `set_body_pose` | `env.set_body_pose(height, torso_rpy, duration, settle)`，经 `strip_obs` |
| `get_robot_geometry` | 返回 `collision_model` 的 `body_names`、`points`（各连杆坐标系下的凸包顶点和球心，取自 `local_pts`）、`radius`（取自 `local_rad`）、`owner`。注释说明这是 CAD 层面的知识，不算真值 |
| `set_arm_joints` | 模块函数 `set_arm_joints`（见 5.7） |
| `set_gripper` | `env.set_gripper` |
| `scan` | 调用 `ep.scan()`，**返回前删掉 `gt` 字段** |
| `draw_markers` | 先用 `ep.note_harness_markers` 记下各标记的变更时间，再转到世界系，写进 `harness` 组 |
| `clear_markers` | 记一次空标记列表，然后清空 `harness` 组 |

#### 5.4 eval port 方法表 `E`

每个请求都必须带正确的令牌。

| 方法 | 实现 |
|---|---|
| `reset(scenario)` | 先 `ep.reset()`（会结束上一回合没关的录像），再 `markers.clear()`、`env.reset(scenario=...)`；返回 `scenario_id, task, timestamp` |
| `get_gt` | `env.get_ground_truth()` |
| `get_actual_state` | `actual_state()` |
| `set_gt_markers(enabled)` | 开关真值标记 |
| `snapshot(path=None)` | 第三人称截图；给了 `path` 就用 cv2 存成图片 |
| `record_start / record_stop / annotate / stage` | 转给 `Episode` 的同名方法 |
| `get_stage_states` | 每个阶段名第一次出现时的记录 |
| `get_stage_log` | 按顺序列出所有阶段调用 |
| `get_scan_log` | robot port 上所有扫码记录（含真值），即 `ep.ov.scans` |
| `invisibility_check` | 验证标记对机器人相机不可见（见 5.6） |
| `shutdown` | 回复后退出主循环 |

```{figure} images/robot_snapshot_t2.jpg
:width: 90%

eval port 的 `snapshot` 截出的第三人称画面（任务 2 场景 `t2_L0_0000`，扫码枪放到条码前方之后）。扫码枪前方黄色箭头是实际手腕位姿标记。这类标记只在第三人称渲染期间显示，机器人自己的头部和手腕相机看不到。
```

#### 5.5 主循环

```
while running:
    for sock, _ in poller.poll(1000):          # 每次最多等 1000 ms
        req = unpack(sock.recv())
        table = R if sock is rs else E
        ...
```

两个 socket 由同一个线程轮询。模块 docstring 写明原因："Isaac Sim must be driven from its main thread"，Isaac Sim 只能从主线程驱动，所以不能为两个端口各开一个线程。每个请求的处理步骤：

1. 来自 eval port 但令牌不对：回复 `{"status": "error", "reason": "eval port: bad or missing token"}`。
2. 方法名不在对应方法表里：回复 `unknown method`。注意 robot port 只查表 `R`，所以 harness 即使知道 `get_gt` 这个名字，在 robot port 上调用也只会得到 `unknown method`。
3. 否则调用该方法；如果是 eval port 上的 `shutdown`，把 `running` 设为 False。
4. `except Exception` 捕获异常，把异常类型、信息和完整 traceback 回复给调用方，同时打印到服务器日志，服务器继续运行。注释原文是"report to the caller; the server keeps serving"。这是本组件里唯一一处宽泛捕获；错误不会被吞掉，因为客户端 `RpcClient.call` 看到 `traceback` 就会抛 `RuntimeError`。
5. 只有带 `status` 键的 dict 回复会加上 `_wall_s`（处理耗时，秒，保留 3 位小数）。注释说明 `get_gt`、阶段状态这类数据回复要保持原有结构，所以不加。

**退出**：循环结束后，用 `linger=0` 关闭两个 socket，`ctx.term()`，刷新标准输出和标准错误，然后 `os._exit(0)`。注释说明 `env.close()` 在 socket 还开着时可能卡住，所以发出回复后直接结束进程；这与 `G1BarcodeEnv.close()` 自己采用的立即退出方式（`SimulationApp.close(skip_cleanup=True)` 的语义）一致。这一做法的来历见演进记录 v2。

#### 5.6 `invisibility_check()`：标记不可见自检

需求要求诊断标记（真值、harness 目标等）不能出现在机器人相机画面里，否则 agent 可能从图像里"看到"真值。这个方法在一个固定的物理状态下（中间不走物理步）渲染若干组机器人相机画面并比较：

1. 在头部相机光轴前方 0.6 m 放一个红色探针，在手腕相机光轴前方 0.15 m 放一个紫色探针。
2. `cams()` 先连续渲染 `VIZ["invisibility_render_passes"]`（当前为 2）次，再强制更新头部和手腕相机，返回两张 RGB 图。
3. 先调用一次 `cams()` 预热。之后按"标记存在但隐藏 → 移除 → 存在 → 移除"的顺序各渲染一次（隐藏是正常运行时的状态）；再不做任何改动渲染一次，作为渲染器噪声基线；然后显示标记渲染一次，作为对照组 C；最后隐藏并移除标记再渲染一次，得到 B。注释说明这样安排是为了"consecutive renders at one state, toggling only the markers, so every compared pair has the same render history"，即每一对被比较的帧都有相同的渲染历史。
4. `diff(a, b)` 取逐像素各通道绝对差的最大值，输出 `max`、99.9 分位数 `p999`，以及差值超过 10、30、60、100 个灰度级的像素数。
5. 输出：相邻的"隐藏"与"移除"帧之间的三组差、基线差、"显示"与"移除"的差；另外比较截一次第三人称快照前后由 `get_obs` 交出的头部和手腕图像。

函数本身只返回统计量。判定写在测试 `tests/test_markers_invisible.py` 里：差值超过 `STRONG = 60` 个灰度级的像素算"有变化"；"隐藏 vs 移除"的三组都必须是 0 个，对照组"显示 vs 移除"必须大于 0 个，快照前后的差 `max` 必须为 0。测试注释给出 60 的依据：Isaac 的实时渲染器在相邻帧之间不是逐位相同的（时间抗锯齿），实测同一状态的噪声在少数像素上能到 55 个灰度级，而标记带来的变化超过 100 级。0.6 m、0.15 m 两个探针距离是代码常数，注释没有给出依据。

#### 5.7 模块级辅助函数

| 函数 | 作用 |
|---|---|
| `strip_obs(r)` | 去掉返回值里的 `obs` 字段（行走等动作的返回里带着完整观测，没必要经网络传回） |
| `wrist_obs(env, side)` | 扫码枪没开，或 `side` 不是扫码枪所在一侧时，返回 `{"status": "error", "reason": "no wrist camera on the … hand"}`；否则返回 `rgb, depth`（可为 `None`）`, K, T_root_wristcam, timestamp` |
| `set_arm_joints(env, side, q, duration)` | 步数 `n = max(1, round(duration / policy_dt))`；`env.arm.set_joint_targets(side, q, n)` 得到 `q_cmd`，然后 `env.stand_still(duration)`，最后读该侧手臂当前关节角 `q_now`。返回 `status, reason, q_cmd, q_now`，调用方可以据此算跟踪误差 |
| `robot_info(env)` | 关节名、`ARM_JOINTS`、头部和手腕相机的内参与分辨率、`wrist_side`、`policy_dt`、`velocity_limits`、每个手臂关节的软限位、手腕点的文字说明（"scanner tip (exit-window centre) of the scanner held by the left gripper"）。行走控制器有 `set_body_command` 时还返回 `body_command`：`height_default, height_range, rpy_limit_rad`；没有时为 `None` |
| `gt_standoff_fn(task)` | 返回函数"条码 → 扫码枪离条码的真值距离"，录像中的绿色目标点和 METRICS 面板用它，评测器也用同一规则。任务 2 固定用 `task2.yaml` 的 `offset_m`；其他任务用评测器的自适应规则 `gt_standoff(gt_symbol_size(bc), scanner_hfov_gt())` |

#### 5.8 `class Episode`：回合在评测侧的状态

`Episode` 保存一个回合内评测和录像需要的全部状态：agent 叠加层事件、阶段状态、robot port 的扫码日志和合成录像。场景重置时一起重置。

| 成员或方法 | 作用 |
|---|---|
| `reset()` | 正在录像时先 `record_stop()`；新建 `OverlayState`（`g1agent/viz/compositor.py`）；`stages = {}` 只保留每个阶段名**第一次**出现的记录（注释：评测器读 `stages["task1_end"]`）；`stage_log = []` 按顺序记录**每一次**阶段调用；`harness_t = {}` 记录每个 harness 标记 id 的原始描述和最后变更时的仿真时间 |
| `note_harness_markers(ms)` | harness 只要改了一个标记就会重发它的全部标记。描述与上次相同的标记保留原时间戳，有变化的记为当前 `t_sim`。docstring 说明这样做是为了不把"选定当前条码之前"设下的手腕目标误当成当前目标 |
| `harness_wrist_targets()` | 按"紫色（经手腕相机细化）在前、蓝色（来自头部相机）在后"的顺序，返回 harness 组中有时间记录的手腕目标 `[{"pos", "t_set"}]`；颜色含义依据 `viz.yaml colors` |
| `scan()` | 调用 `env.scan()`，把 `status, content, decoded, gt, T_world_tip, pelvis_z, height_cmd` 和 `t_sim` 交给 `ov.on_scan` 记录；正在录像时写 `detection_hold_s` 长的静止帧。返回完整结果，由 R 表负责删掉 `gt` |
| `annotate(event)` | 把 harness 事件交给 `ov.on_event`；需要静止帧时写入，类型 `"llm"` 用 `llm_hold_s`，其他用 `detection_hold_s` |
| `stage(name, info=None)` | 记录当时的实际状态，追加到 `stage_log`，用 `setdefault` 写入 `stages`，通知叠加层；正在录像且 `name == "face_start"` 时按 `info["face"]` 重新摆放录像相机 |
| `record_start(path, fps=None, task=None)` | 开始合成录像，步骤见下 |
| `_pose_cams(framing)` | 对每台录像相机用 `compositor.fit_fixed_camera` 按 `framing.hfov_deg`、`margin_frac` 和相机分辨率求位姿，调用 `env.pose_rec_camera`，最后 `env.settle_renders()`；返回 `{name: (K, T_world_cam)}` |
| `_pose_face_cameras(face_idx)` | 多面巡检时用 `compositor.face_framing` 重新摆相机：MAIN 放在当前面和下一个有标签的面之间的对角线上，SIDE 对准当前面；单面场景返回 `None`，不改动 |
| `record_stop()` | 关闭帧索引文件和视频写入器，返回帧数 |
| `on_step()` | 每个策略步计数加一，每 `every` 步写一帧 |
| `_write(hold, hold_s)` | 写一帧，或写 `round(hold_s * fps)` 张相同的静止帧 |

**`record_start` 的步骤**：

1. 前置检查，任一不满足就抛 `RuntimeError`，不悄悄降级：录像相机已开启、已经 reset 到某个场景、扫码枪相机已开启。
2. `fps` 默认取 `REC["fps"]`（当前 5）。`every = max(1, round(1 / (fps * policy_dt)))`，实际帧率 `fps_eff = 1 / (every * policy_dt)`。注释："1 video second = 1 simulated second"，视频时间等于仿真时间。
3. 缓存真值（回合内不变）。`task` 默认取场景自带的任务，可以被覆盖（对应 `run_task.py --task`）。真值目标点 `targets = C.gt_targets(gt, gt_standoff_fn(task))`。
4. 把机器人碰撞点转到根坐标系，用 `C.framing` 求出 MAIN、SIDE 两台固定录像相机的位姿。
5. 创建 `C.H264Writer(path, *REC["canvas"], fps_eff, ffmpeg)`（画布 1920×1080，ffmpeg 来自 `imageio_ffmpeg`），并打开同名的 `.frames.jsonl` 帧索引文件。`C.labelled_tour` 多于一面时标记 `multi_face`。
6. 立即写第一帧；返回 `path, fps, every_steps, task` 和每台相机的 `K`、`T_world_cam`。

**`_write` 每一帧的内容**：

- 当前目标：`ov.current_target(targets, harness_wrist_targets())`。
- 指标：`C.frame_metrics(gt, fs, targets, current, ov.scans, stages.get("task1_end"))`。
- 3D 标记（投影到 MAIN/SIDE 画面）：绿色真值观察位姿（多面场景另加当前面的真值观察位姿，按 `C.FACE_TOUR` 匹配）、绿色真值目标点（方向取 `-normal`）、harness 组的全部标记、黄色实际根位姿（标注 `root (pelvis)`，带 `body_z`）、黄色实际手腕位姿。
- 扫码文字：还没扫过显示 `scanned: - (no scan yet)`；成功显示 `scanned: yes - 内容 (duplicate|new)`，duplicate 表示之前已经成功扫过相同内容；失败显示 `scanned: no (status)`；后接 `k/n`。
- 四路图像 `main, side, ego(head), wrist`：直接从各相机的渲染缓冲区读取。模块 docstring 写明不额外渲染、不切换 USD 标记、不调用 `get_obs`，标记和检测框以 2D 叠加层画上去。
- 手腕画面只在最近一次扫码恰好发生在当前帧（`t_sim` 相同）时画出解码多边形。
- 头部画面文字 `boxes seen: …; localised target: …`；agent 面板显示阶段、LLM、工具、`"扫码 " + 扫码文字`和当前面。
- 写入视频后向 `.frames.jsonl` 追加一行 `{frame, n, hold, t_sim, t_unix, metrics}` 并立即 flush，进程中途被杀也不会丢失已写的索引。

### 6. 真值隔离：四个层面

| 层面 | 做法 |
|---|---|
| 接口 | `RobotInterface` 和 `SimRobot` 都没有真值方法；工具层 `Toolbox` 只拿到一个 `SimRobot` |
| 端口 | 真值只在 eval port 上，请求必须带令牌；robot port 只查方法表 `R`，`get_gt` 之类的名字在那里是 `unknown method` |
| 字段 | robot port 的 `scan` 返回前删掉 `gt`；带真值的扫码记录只能经 eval port 的 `get_scan_log` 读 |
| 渲染 | 可视化标记平时隐藏，只在第三人称渲染时短暂显示，之后再渲染一次；`invisibility_check` 和对应测试验证机器人相机画面不受标记影响 |

`get_robot_geometry` 返回的是机器人自身的碰撞形状，代码注释把它归为 CAD 知识：真机上同样可以从机器人模型得到，所以不算真值。

### 7. 配置

#### 7.1 `configs/system.yaml`：进程与地址

| 键 | 值 | 说明（多数来自行内注释） |
|---|---|---|
| `sim_server.conda_env` | `unitree_sim_env` | 环境名，或完整前缀路径；启动脚本用它激活环境 |
| `sim_server.sim_root` | `../g1_barcode_sim` | 仿真包目录，相对本仓库；可用 `$G1_SIM_ROOT` 覆盖 |
| `sim_server.robot_port` | `31555` | RobotInterface RPC；实例 i 使用 port + 10·i |
| `sim_server.eval_port` | `31556` | 评测与回合控制：真值、reset、真值颜色的标记、录像 |
| `sim_server.instance_port_stride` | `10` | 并行跑多个服务器（`SIM_INSTANCE=i`）做批量评测 |
| `sim_server.eval_token_env` | `G1_EVAL_TOKEN` | eval port 只接受带这个令牌的请求；令牌放在环境变量里，不写进文件 |
| `sim_server.host` | `127.0.0.1` | 只监听本机 |
| `sim_server.mode_policies` | `null` | 当前行走控制器是 `gear_wbc`（GR00T 自己在 Balance 和 Walk 之间切换），必须为 null。注释保留了切回旧 unitree 行走控制器的写法 |
| `sim_server.rpc_timeout_s` | `600` | 客户端接收超时；一次长距离行走或回合 reset 可能要几分钟墙钟时间 |
| `vlm_server.conda_env` / `port` | `vllm_env` / `31800` | VLM 服务（见 LLM 相关页面） |
| `vlm_server.gpu_memory_utilization` | `0.85` | 注释没有给出依据 |
| `vlm_server.max_model_len` | `32768` | 上下文长度，是显存方面的设置，不限制输出长度；注释没有给出具体数值的依据 |
| `harness_client.conda_env` | `g1_agent` | — |
| `harness_client.detector_device` | `cuda` | grounding-dino 用；CPU 也能跑，较慢 |
| `runs_dir` | `runs` | — |

#### 7.2 `configs/robot.yaml`：harness 侧的机器人参数

**坐标系与几何**

| 键 | 值 | 说明 |
|---|---|---|
| `urdf` | `assets/g1_dex1_scanner.urdf` | 从仿真的 USD 导出（`scripts/export_urdf.py`） |
| `root_frame` | `pelvis` | 根坐标系是骨盆 |
| `head_cam_frame` | `head_cam_optical` | — |
| `arm_side` | `left` | — |
| `wrist_point_frame` | `scanner_tip` | 需求原话"以后装上扫码枪，就改成枪口位置"，取扫码枪出光窗中心 |
| `wrist_cam_frame` | `wrist_cam_optical` | 扫码枪头部的相机：类 DS2208 成像器，640×480，水平视场角 32.8° |
| `wrist_forward_axis` | `[1, 0, 0]` | 扫码方向是 `scanner_tip` 的 x 轴 |

**`ik`（逆运动学）**

| 键 | 值 | 依据 |
|---|---|---|
| `dls_lambda` | `0.05` | 注释没有给出依据 |
| `step_gain` | `0.5` | 每次迭代修正误差的比例（离线计算，不含动力学）；数值没有给出依据 |
| `max_iters` | `300` | 注释没有给出依据 |
| `tol_pos` / `tol_angle_deg` | `0.002` m / `1.0` | 注释没有给出依据 |
| `aim_weight` | `0.20` m/rad | 与仿真器的瞄准 IK 一致：扫码距离 0.20 m 时把角度误差折算成位置误差 |
| `joint_limit_margin` | `0.05` rad | 与每个软限位保持的距离；数值没有给出依据 |
| `restarts` | `6` | 首次求解不收敛时额外做的随机重启次数；数值没有给出依据 |
| `plan_workers` | `16` | 站位规划的进程池大小（`tools/stance_eval.py`），0 为串行。2026-11-07 实测（3 个堆叠码、带 roll 约束、与串行得到的站位相同）：串行 10/30/104 s，8 进程 5/12/53 s，16 进程 5/10/47 s，32 进程 5/9/50 s。16 个已饱和，剩下的时间花在串行的间隙检查和鲁棒性检查上 |
| `plan_stall_window` / `plan_stall_eps_m` | `100` / `0.0005` m | 只用于站位规划，不用于 move_arm：100 次迭代内加权误差下降不到 0.5 mm 就停。2026-11-07 在 0.30 m 高的堆叠码上实测：705 个失败的求解都跑满 300 次迭代，占 IK 计算量的 95%；加这条规则后它们的中位停止迭代数是 118，代价是 205 个成功求解中有 1 个被误停（0.5%） |
| `seed` | `0` | — |

**`arm_motion`**：`duration_s: 2.0`（`set_arm_joints` 斜坡时间）、`settle_s: 1.0`（斜坡后额外站立时间）、`correction_iters: 2`（对指令目标做闭环修正的次数，补偿手臂 PD 控制的下垂）、`correction_tol: 0.01` m。这四个数值注释都没有给出依据。

**`limits`（安全约束）**：`max_move_m: 0.5`（单次探索移动）、`max_turn_deg: 15.0`、`max_approach_segment_m: 1.0`、`workspace_radius_m: 8.0`（以回合开始时的里程计位置为圆心，机器人不能离开这个半径）。前三项与需求文档一致：需求文档规定单次探索移动不超过 0.5 m、转角不超过 15°，分段接近每段最多走 1.0 m；需求文档只要求"不得离开工作区"，`workspace_radius_m` 的 8.0 m 没有写明来源。

**`body`（下蹲/起立与站位规划，见 `tools/body.py`）**

| 键 | 值 | 依据 |
|---|---|---|
| `calibration` | `assets/gear_body_calibration.json` | 在 Isaac 中实测（来源写在文件里）；规划时只用这里列出的下蹲高度档位 |
| `ramp_s` | `1.0` | 与标定时的斜坡时间相同，每个档位在这个斜坡下都是稳定的 |
| `settle_s` | `2.0` | 与标定时的保持时间相同，标定值就是在这个时刻取的 |
| `clearance_margin_m` | `0.03` | 规划姿态中机器人与码面、手臂与腿之间的最小间隙：运动测试中手臂瞄准误差最大 8 mm，加放置距离容差 0.02 m（`task2.yaml check.dist_tol_m`），向上取整 |
| `stance_robust_pos_m` / `stance_robust_yaw_deg` | `0.03` / `1.7` | 等于仿真包 `move_relative` 的容差（pos_tol 0.03 m、yaw_tol 0.03 rad = 1.7°） |

### 8. 启动脚本 `scripts/start_sim_server.sh`

用法（脚本注释）：`setsid nohup bash scripts/start_sim_server.sh > runs/sim_server.log 2>&1 &`，以脱离会话的方式在后台运行，SSH 断开也不会结束。

1. `cd` 到仓库根目录。仓库根目录下有 `local.env`（不纳入版本管理）时先 `source` 它。这个文件放与机器相关的设置：conda 环境所在目录 `G1_CONDA_PREFIX`、仿真包位置 `G1_SIM_ROOT`，以及可选的 shell 函数 `gpu_launch <label> <cmd...>`，用来把命令包进共享机器上的 GPU 队列。
2. 创建 `runs/`。`runs/.eval_token` 不存在时，在 `umask 077` 下用 `secrets.token_hex(16)`（32 个十六进制字符）生成一个，文件只有所有者可读写；然后 `export G1_EVAL_TOKEN`。评测侧（`run_task.py`、测试）从同一个文件读令牌。
3. 从 `system.yaml` 读 conda 环境名，加载 conda，`unset CC CXX CFLAGS CXXFLAGS CPPFLAGS LDFLAGS`（脚本没有注释原因），激活环境（有 `G1_CONDA_PREFIX` 时用"前缀/环境名"）。
4. `export OMNI_KIT_ACCEPT_EULA=YES`；`SIM_INSTANCE` 默认 0。
5. 选 GPU：设置了 `SIM_GPU` 时直接把 `CUDA_VISIBLE_DEVICES` 设为它，然后 `exec python -u -m g1agent.robot.sim_server`；否则，如果 `local.env` 定义了 `gpu_launch`，就经由它启动（例如经共享 GPU 队列租一张卡）；两者都没有时继承调用方的 `CUDA_VISIBLE_DEVICES` 直接启动。

### 9. M1/M2 自检 `scripts/check_m1_m2.py`

M1 检查各进程能互相通信，M2 检查客户端的正运动学（FK）与仿真器一致。运行前仿真服务器必须已经启动。参数只有 `--out`（默认 `runs/check_m1_m2.json`）；docstring 里提到的 `--vlm` 在代码里没有实现。

流程：

1. `ping` 并记录往返时间；用 `EvalClient` reset 到场景 `t2_L0_0000`；`get_robot_info`；记录头部和手腕观测的形状与 `K[0,0]`。
2. `fk_check(tag)`：读关节状态，用 `kin.q_from_state` 映射成运动学模型的 `q`；对 `head_cam_optical` 和 `wrist_cam_optical`，分别比较 FK 位姿与仿真器返回的 `T_root_headcam`、`T_root_wristcam`。位置误差取欧氏距离（mm），旋转误差取 `arccos((tr(R_fk^T R_sim) - 1) / 2)`（度）。两台相机都满足位置误差 < 5 mm 且旋转误差 < 1° 才算通过。
3. 初始姿态做一次 `fk_check("start")`。
4. 行走：`walk(0.3, 0.0, 0.0, 1.5)`，记录前后两次里程计之间的相对 x 位移 `walk_odom_dx`。
5. 手臂：用 IK 解胸前的扫码枪目标，位置 `[0.40, 0.18, 0.15]`（根坐标系，米）、方向 `[1, 0, 0]`；`set_arm_joints("left", q_arm, 2.0)` 后 `stand(1.0)`；用 FK 算 `scanner_tip` 的位置误差、x 轴方向误差，以及 `q_cmd` 与 `q_now` 的最大关节跟踪误差。
6. 手臂动过后再 `fk_check("arm_moved")`；然后 `scan()`，只记录 `status` 和 `content`。
7. 判定：
   - `M1_pass`：头部和手腕 RGB 都是 `[480, 640, 3]`，`walk_odom_dx > 0.05` m，且枪尖位置误差 `< 0.05` m。行内注释说明枪尖容差对应开环指令，存在 PD 下垂，正式的 `move_arm` 还会做闭环修正（`arm_motion.correction_iters`）。
   - `M2_pass`：两次 `fk_check` 都通过。
8. 结果写入 JSON 并打印（不打印 ping）。

脚本里的常数：场景 `t2_L0_0000`、行走 0.3 m/s × 1.5 s、IK 目标 `[0.40, 0.18, 0.15]` 和 `[1, 0, 0]`、斜坡 2.0 s 加站立 1.0 s、阈值 5 mm、1°、0.05 m（里程计位移）、0.05 m（枪尖误差）、图像形状 480×640×3。除枪尖误差有"开环、PD 下垂"的说明外，其他常数注释都没有给出依据；里程计位移阈值的修改过程见演进记录 v0。

## 演进记录

### v0 · 2026-11-05 · 初版：抽象接口、ZeroMQ 双端口服务器、自写序列化

**初版设计。**

- `g1agent/robot/interface.py` 定义 `RobotInterface`，列出 harness 能做的全部操作：`get_robot_info`、`get_head_obs` / `get_wrist_obs`（rgb、以米为单位的深度、K、相机相对 root 的位姿、时间戳）、`get_joint_state`、`get_odom`、`walk`、`move_root_relative`、`stand`、`set_arm_joints`、`set_gripper`、`scan`、`draw_markers`、`clear_markers`。仿真客户端和将来的真机实现提供同一组方法，接口里没有真值。
- 仿真服务器开两个端口：robot port 只提供 `RobotInterface`；eval port 负责评测和回合控制（真值、reset、真值颜色的标记、录像），只接受带 `G1_EVAL_TOKEN` 的请求。令牌放在环境变量里，服务器首次启动时生成 `runs/.eval_token`。这样从结构上就把真值通道和 agent 分开了，而不是靠"agent 不去调用"的约定。
- 序列化：开工前检查了仿真环境 `unitree_sim_env`，它有 pyzmq 27.0.0 和 msgpack 1.2.4，但 `import msgpack_numpy` 报 `ModuleNotFoundError`。我们没有往仿真环境里加装这个包，而是自己写了 `g1agent/common/serialization.py`（不到 30 行，docstring 写明"no extra dependency"，见当前实现 1.2）。客户端和服务器共用同一份 `pack` / `unpack`。
- `RpcClient` 超时后重建 REQ socket 再抛 `TimeoutError`（REQ socket 错过一次回复后就无法再用）；`rpc_timeout_s` 取 600 s，因为一次长距离行走或回合 reset 可能要几分钟墙钟时间。
- 行走控制器：最初 `system.yaml` 里只有一个 `policy_model: assets/model/policy.onnx`，注释是"policy.onnx keeps the base still while the arm moves"，即选它是因为动手臂时底盘不动。
- 录像：最初的 eval port 录像是在每个策略步钩子里每隔若干步截一张跟随机器人的第三人称画面，用 cv2 的 `mp4v` 编码写进 mp4（这一版的问题见 v5）。

**首次自检的结果。** 2026-11-05 第一次运行 `check_m1_m2.py`：

| 项 | 结果 |
|---|---|
| 头部、手腕 RGB 形状 | 都是 [480, 640, 3] |
| FK 与仿真器的相机位姿差（初始姿态） | 头部、手腕都是 0.0 mm、0.0° |
| `walk(0.3, 0, 0, 1.5)` 后的 `walk_odom_dx` | 0.163 m |

M2 一次通过；进度报告里记为"pinocchio 算出的头部和腕部相机位姿与仿真相比误差为 0.000 mm / 0.000°"，原因是 URDF 直接从仿真 USD 导出。

**遇到的问题：里程计位移阈值。** M1 判定最初要求 `walk_odom_dx > 0.2` m，实测 0.163 m 没达到。指令是 0.3 m/s 走 1.5 s，名义位移 0.45 m，实际少得多（推测是起步加速和行走控制器的跟踪误差，日志里没有专门分析）。这一项要验证的是"行走指令能经 RPC 传到仿真器并让机器人前进、里程计能读出来"，而不是行走精度，所以我们把阈值改成 `> 0.05` m，含义变成"确实向前走了"。行走精度另由任务 1 的移动测试评估。同一次修改给枪尖误差阈值 `< 0.05` m 加了注释"open loop (PD sag); move_arm adds closed-loop correction"。

### v1 · 2026-11-05 · 行走控制器改为按模式切换两个策略

**当时的版本**：只用 `policy.onnx` 一个策略，好处是动手臂时底盘不动。

**改动**：`system.yaml` 的 `policy_model` 换成 `mode_policies`：行走时用 `policy1.onnx`（注释："tracks base commands"，能跟踪底盘速度指令），站立和动手臂时用 `policy.onnx`（底盘保持不动）。`sim_server` 构造 `G1BarcodeEnv` 时把它作为 `{"robot": {"mode_policies": ...}}` 覆盖传入，由仿真包在行走和站立之间切换策略。

**依据（写在当时的配置注释里）**：实测移动误差 3–4 cm；行走与站立来回切换 5 次都稳定（倾斜不超过 2.7°）；瞄准加扫码过程中底盘漂移 1.9 cm。

### v2 · 2026-11-05 · 修复：shutdown 后服务器进程不退出

**现象**：通过 eval port 调用 `shutdown` 后重启仿真服务器，`pgrep` 发现旧进程（pid 2981167）还活着，和新进程（pid 3035749）同时存在，只能手动 kill。

**根因**：服务器回复 `shutdown` 后调用 `env.close()`，而此时两个 ZeroMQ socket 还开着，`env.close()` 会卡住，进程一直不退出。

**修改**（`sim_server.py` 主循环之后）：回复发出后，先以 `linger=0` 关闭 robot 和 eval 两个 socket，`ctx.term()`，刷新 stdout/stderr，然后 `os._exit(0)` 立即结束进程。这与 `G1BarcodeEnv.close()` 自己用的 `SimulationApp.close(skip_cleanup=True)` 语义一致：Isaac 进程本来就是直接退出的，不做清理。

**效果**：修改后再次重启，旧进程能自行退出（当时日志："Shutdown is clean now"）。

### v3 · 2026-11-05 · 多实例：端口步长与 `SIM_INSTANCE`

**当时的版本**：一台机器上只能跑一个仿真服务器，端口固定为 47555/47556。批量评测只能串行跑场景。

**改动**：

- `system.yaml` 加 `instance_port_stride: 10`；实例 i 的端口是 `robot_port + 10·i`、`eval_port + 10·i`。
- 服务器、`SimRobot`、`EvalClient` 都从环境变量 `SIM_INSTANCE` 读实例号（`SimRobot` / `EvalClient` 也可以直接传 `instance`）。
- 启动脚本加 `SIM_INSTANCE` 默认值，以及 `SIM_GPU`（直接指定一张共享的 GPU，与那张卡上已有的任务共存）。
- 就绪行改为 `SIM_SERVER_READY instance=… robot=… eval=…`。

**效果**：同一天晚些时候实例 0 和实例 2 同时运行，分别打印 `SIM_SERVER_READY instance=0 robot=47555 eval=47556` 和 `SIM_SERVER_READY instance=2 robot=47575 eval=47576`，两个服务器都能按各自的 eval 客户端正常 `shutdown`。

### v4 · 2026-11-05 · 标记不可见自检：从"任意像素不同"到"相邻渲染、强差异计数"

**初版**：诊断标记做成 `/World/Markers` 下的 USD 几何体，平时隐藏，只在第三人称渲染期间显示，之后再渲染一次（Isaac 的 debug-draw 叠加层只存在于视口中，而我们无界面运行，所以不能用它）。第一次跑完整测试集时，这项测试失败了一次（失败时的数值没有保存）。随后 `invisibility_check` 改写为：依次渲染"标记隐藏"A、"标记移除"B、再渲染 B 作基线、"标记显示"C，`diff` 统计**任意通道有差异**的像素数。

**问题 1：差值全是噪声。** 改写后第一次运行（场景 `t2_L0_0000`）：

| 比较 | 头部相机差异像素数 | 手腕相机差异像素数 |
|---|---|---|
| 隐藏 vs 移除 | 253317 | 273038 |
| 基线（同一状态再渲染一次） | 134212 | 135998 |
| 显示 vs 移除（对照组） | 147716 | 155759 |

"隐藏 vs 移除"比对照组"显示 vs 移除"还大，这个统计量区分不了"标记泄漏"和"渲染噪声"。根因：Isaac 的实时渲染器在相邻两帧之间不是逐位相同的（时间抗锯齿），哪怕场景完全不变，也有十几万个像素有微小差异。

**修改 1**：`diff` 改为逐像素取各通道绝对差的最大值，输出 `max`、99.9 分位数和差值超过 10/30/60/100 级的像素数。

**问题 2：仍然有强差异。** 重跑后头部相机"隐藏 vs 移除"的 `max` 为 97，超过 60 级的像素有 15 个；基线 `max` 为 55，超过 60 级的为 0 个。根因：被比较的两帧渲染历史不同。A、B、C 的渲染顺序中间夹着别的渲染，时间抗锯齿会把前几帧的内容混进当前帧，所以两帧之差不只反映标记本身。

**修改 2**：改成在一个物理状态下连续渲染、只切换标记：预热一次，然后按"存在（隐藏）→ 移除 → 存在 → 移除"各渲染一次，相邻两帧比较；再不做改动渲染一次作基线；最后显示标记作对照 C，移除后得到 B。

**效果**（场景 `t2_L0_0000`）：

| 比较 | 头部相机 | 手腕相机 |
|---|---|---|
| 隐藏 vs 移除（三组相邻比较） | 三组都是 0 | 三组都是 0 |
| 基线 | `max` 32，超过 30 级的 1 个，超过 60 级的 0 个 | `max` 2 |
| 显示 vs 移除（对照组） | `max` 119，超过 60 级的 38 个 | `max` 149，超过 60 级的 22425 个 |
| 第三人称快照前后 | `max` 0 | `max` 0 |

测试 `tests/test_markers_invisible.py` 据此把"有变化"定义为差值超过 60 级（同状态噪声实测最高 55 级，标记带来的变化超过 100 级）。进度报告记录这项测试共跑 12 次，11 次通过。唯一一次失败是第一次跑完整测试集时，发生在上面的改写和两处修改之前；按开发日志，修改后先单独运行 1 次，再连续 3 次完整测试和 8 次单独运行，全部通过，共 12 次（日志里合计 13 次运行、12 次通过，比报告的"12 次、11 次通过"多计到 1 次单独运行）。报告把它记为"第一次失败未能复现，原因待查"：它可能与改写前的自检方法有关，但当时没有保存失败时的数值，无法确认。

### v5 · 2026-11-06 · 行走控制器换成 GR00T，`mode_policies` 改为 null

**当时的版本**：v1 的双策略切换（行走 `policy1.onnx`，站立 `policy.onnx`）。

**改动**：仿真包新增 GR00T 的 `gear_wbc` 行走控制器并设为默认。GR00T 自己在 Balance 和 Walk 两种模式之间切换，而且 `gear_wbc` 模式下 `mode_policies` 必须为 null。因此把 `system.yaml` 的 `mode_policies` 改为 `null`，旧配置 `{walk: assets/model/policy1.onnx, stand: assets/model/policy.onnx}` 以注释形式留下，便于回滚（回滚还要在仿真包 `robot.yaml` 里设 `locomotion: unitree`）。`sim_server.py` 的覆盖逻辑不变。控制器对比的数据见仿真与控制器相关页面。

### v6 · 2026-11-06 · `Episode` 记录器与合成录像重做

**当时的版本**：录像就是一个策略步钩子，每隔 `rec_every` 步调用一次 `third_person_view()`，用 `cv2.VideoWriter(..., fourcc('mp4v'), fps)` 写入；fps 默认 2。

**问题**：需求反馈"每个 episode.mp4 都无法打开"，并且要求把多个视角拼成一个画面（左后方固定视角、箱子侧面视角、头部视角、手腕视角，加检测框、扫码结果和与目标点的三维误差）。排查结论：

- 编码：`mp4v` 是 MPEG-4 Part 2，浏览器、QuickTime 和常见预览都只解码 H.264。仿真环境里的 cv2 自带的 FFmpeg 只有 `mpeg4`、`libvpx`、`libvpx-vp9` 编码器，没有 libx264；而且代码没有检查 `isOpened()`，换成 `avc1` 会悄悄写不出任何帧。
- 截断：`record_stop` 没被调用（例如服务器被杀）时 mp4 没有 moov atom。当时 20 个已有文件里有 2 个因此损坏。
- 视角：相机按机器人当前位姿每帧重算，实际是一个跟随机器人的左前方相机，不是固定视角。
- 开销和副作用：每帧都调用 `env.get_ground_truth()`（重新读 yaml、重算观察位姿）和 `actual_state()` → `env.get_obs()`（完整拷贝头部和手腕图像、做深度后处理），并重新写 USD 标记。`get_obs` 会从深度噪声的随机数流里取数，所以录像本身会改变 agent 后续拿到的深度噪声（推测会让开不开录像的两次运行结果不同，日志里没有专门做对照）。

**修改**：

- 新增 `class Episode`，集中保存一个回合的评测侧状态：`OverlayState`（agent 事件、检测、扫码日志）、`stages`、`harness_t` 和录像状态；`reset` 时一起重置。
- eval port 新增 `annotate`、`stage`、`get_stage_states`、`get_scan_log`；`eval_client.py` 加对应方法。`run_task.py` 把 harness 日志里的阶段和事件转发过来。任务 3 的评测从 `get_scan_log` 读码（每个内容只计一次），按 `stages["task1_end"]` 时的 root 坐标系排序期望集合。
- 合成录像：两台固定录像相机 MAIN、SIDE 按场景布局用 `fit_fixed_camera`（固定朝向和视场角下的闭式视锥拟合）每回合摆一次；四路图像直接从渲染缓冲区读，不额外渲染、不切换 USD 标记、不调用 `get_obs`；标记和检测框改成 2D 叠加层；用 ffmpeg 的 H.264 写入器 `H264Writer`；每帧在 `.frames.jsonl` 里记一行并立即 flush。
- `actual_state` 改用仿真包新增的轻量访问器 `frame_state()`：数值和 `get_obs` 相同，但不拷贝图像、不做深度处理，不消耗深度噪声的随机数流。
- `note_harness_markers` / `harness_wrist_targets`：记录每个 harness 标记最后一次变化的仿真时间，`current_target` 只采用"当前条码选定之后"设下的手腕目标。
- `record_start` 加 `task` 参数（对应 `run_task.py --task`），任务 3 用自适应真值站距画绿色目标点；前置条件不满足时直接抛 `RuntimeError`。

**效果**：录像改为 1920×1080 的 H.264 合成视频，视频时间等于仿真时间（当前 `viz.yaml` 的 `fps` 为 5）；回合中途结束时帧索引仍保留到最后一帧。录像与 agent 观测之间不再共享随机数流。

### v7 · 2026-11-07 · 新增 `set_body_pose`、`get_robot_geometry`；里程计保留横滚和俯仰

**当时的版本**：接口只有行走、站立和手臂动作。GR00T 控制器的身体高度和躯干姿态指令在启动时就固定了，机器人不能下蹲或弯腰，够不到堆叠箱底层的条码。

**问题与修改**：

1. **身体指令。** 在仿真包里给 GR00T 控制器加了身体指令接口：高度范围 0.20–0.74 m（取自上游 GR00T 遥操作的限位）；躯干限位 roll 0.4、pitch 0.6、yaw 0.8 rad，这是我们在 MuJoCo 里跑同一策略后自己定的，Isaac 中确认了 pitch 0.6 可用；超出范围直接报错，reset 或瞬移时恢复站立。单独测试中下蹲到骨盆 0.25 m、弯腰 43°、站立举手使枪尖到 1.21 m 都稳定。随后在 `RobotInterface`、`SimRobot`、`sim_server` 三层加了 `set_body_pose(height, torso_rpy, duration, settle)`，`robot_info` 增加 `body_command`（`height_default, height_range, rpy_limit_rad`）。工具层另加了 `set_base_height` 工具；行走类工具在下蹲状态下会先让机器人站起来，手臂没有收回时直接报错（见工具层页面）。
2. **里程计丢了骨盆倾斜。** 下蹲时骨盆最多前倾 26°，而里程计只保留了 x、y 和航向，所有固定在骨盆或里程计上的目标都跟着偏；有一次蹲到骨盆 0.43 m 时枪尖比目标低约 9 cm。根因是里程计模型把 roll 和 pitch 置零了。修改为里程计保留 IMU 测得的横滚和俯仰，只有 x、y 和航向带漂移（仿真包内修改，经 `get_odom` 传给 harness）。
3. **手臂与腿的碰撞没有仿真。** 机器人模型关掉了自碰撞，下蹲时手臂可以直接穿过大腿和膝盖；真机上这就是碰撞。agent 自己的 URDF 运动学模型又没有碰撞形状，而仿真的间隙模型（`collision.py`）有每个连杆的凸包顶点和脚部球体。我们决定让 agent 在规划下蹲站位时自己检查手臂与腿的间隙，不依赖仿真器，于是新增 `get_robot_geometry`：返回每个连杆在连杆坐标系下的碰撞点（凸包顶点、球心和半径，含扫码枪角点）。注释写明这是 CAD 知识而不是真值，不违反真值隔离。离线导出的几何共 29,588 个点，用于离线检查站位规划。

### v8 · 2026-11-07 · 扫码日志记录骨盆高度和高度指令

**改动**：`Episode.scan` 的记录新增 `pelvis_z`（取 `T_world_root` 的 z）和 `height_cmd`；仿真包 `frame_state()` 新增 `height_cmd` 字段；合成录像的 METRICS 面板相应显示。用来在评测和视频里看出每次扫码时机器人蹲得多低。

**问题**：初版直接读 `self.env.walker.height_cmd`。测试 `tests/test_sim_server_episode.py` 用的假环境没有 walker，测试报错（1 failed, 12 passed）。

**根因**：`Episode` 直接访问了仿真环境的内部对象，而它本应只通过 `frame_state()` 这个公开的状态接口读数据。

**修改**：改为 `fs.get("height_cmd")`，从 `frame_state()` 读（没有身体指令的控制器上为 `None`）。

**效果**：`test_compositor.py` 和 `test_sim_server_episode.py` 共 13 项通过。

### v9 · 2026-11-07 · 完整阶段日志与按面重新摆放录像相机

**当时的版本**：`stages` 只按阶段名存一条记录（`self.stages[name] = ...`，同名阶段后一次覆盖前一次），评测器读 `stages["task1_end"]`。

**问题**：任务 3 加入多面巡检后，`face_start` 等阶段在一个回合里会出现多次，只留最后一次就丢失了前面各面的信息；单个固定录像相机也拍不全绕到侧面后的画面。

**修改**：`Episode` 新增 `stage_log`，按顺序记录每一次 `stage` 调用（`{name, info, t_sim, actual}`），eval port 新增 `get_stage_log`；`stages` 改为用 `setdefault` 保留每个阶段名的**第一次**记录（行为变化：之前是最后一次），评测器读的是第一次 `task1_end`。录像中途收到 `face_start` 时调用 `_pose_face_cameras`，用 `compositor.face_framing` 把 MAIN 放到当前面与下一个有标签面之间的对角线上、SIDE 对准当前面；单面场景保持 `record_start` 时的相机。多面场景的每帧标记另加当前面的真值观察位姿。

这一版的开发日志只记录了改动内容，没有可引用的前后对比数据。

### v10 · 2026-11-08 · 整理为独立仓库：去掉与机器相关的设置

**当时的版本**：`system.yaml` 里写着仿真包、conda 环境和模型缓存的本机绝对路径；启动脚本里直接写了共享 GPU 队列的调用命令和默认预计时长。

**问题**：把 agent 系统整理成单独的 git 仓库后，换一台机器或换一个人就无法直接运行，而且配置里带着本机信息。

**修改**：

- `system.yaml`：`sim_root` 改为相对本仓库的 `../g1_barcode_sim`，可用 `$G1_SIM_ROOT` 覆盖（`config._resolve_paths` 负责解析）；`conda_env` 只写环境名；模型缓存目录等移出配置。
- `start_sim_server.sh`：先 `source` 不纳入版本管理的 `local.env`，由它提供 `G1_CONDA_PREFIX`、`G1_SIM_ROOT` 和可选的 `gpu_launch` 函数；脚本本身不再写死任何 GPU 队列命令。`SIM_GPU` 仍可直接指定一张 GPU。
- `.gitignore` 排除 `local.env`、`runs/`、`.eval_token`、视频和数组文件。

这一步只改了路径和启动方式，没有改接口和服务器行为。

## 已知限制

- **单线程串行。** 两个端口由同一个线程处理，一个请求没处理完，其他请求（包括另一个端口上的）都要等。一次长距离行走可能占用几分钟墙钟时间，期间 eval port 也不响应。这是 Isaac Sim 只能从主线程驱动的直接结果。
- **令牌只防误用，不防有意绕过。** 令牌文件 `runs/.eval_token` 只有所有者可读，但 harness 和运行器在同一用户、同一进程里：运行器（`run_task.py`）本身就持有 `EvalClient`，把阶段和事件转发到 eval port。真值不会进入 agent，靠的是工具层只拿到 `SimRobot`、真值只在 eval port 上这两点代码结构，而不是进程级的权限隔离。
- **`stage` 的回复含真实状态。** eval port 的 `stage` 返回 `{"status": "ok", "actual": a}`，运行器拿到后不交给 harness；如果以后有人把这个返回值传回工具层，就会泄漏真值。
- **真机实现尚不存在。** `RobotInterface` 是按"真机同样能提供"设计的，但目前只有仿真实现；`draw_markers` 在真机上应为空操作，`get_robot_geometry` 在真机上需要从 URDF 或 CAD 模型另行得到。
- **标记不可见测试有一次未复现的失败。** 进度报告记为 12 次运行中 1 次失败，发生在自检方法改写之前，原因没有最终确认（见 v4）。这项测试需要一个正在运行的仿真服务器；连不上时要等 RPC 超时（测试里设为 300 s）后才跳过，日常跑单元测试时通常用 `-k "not marker"` 或 `--ignore` 把它排除，因此不覆盖它。
- **部分常数没有依据。** `RobotInterface` 的默认超时和斜坡时间、`invisibility_check` 的探针距离、`check_m1_m2.py` 中的大部分阈值、`robot.yaml` 中 `ik` 的多数参数、`arm_motion` 的具体数值、`limits.workspace_radius_m`、启动脚本中 `unset CC CXX ...` 的原因，注释里都没有写依据，已在上文逐项标出。
- **`check_m1_m2.py` 的 docstring 与代码不一致。** docstring 写了 `--vlm` 参数，代码没有实现。
- **里程计位移阈值很宽。** M1 只要求 1.5 s 内前进超过 0.05 m，只能说明通信和行走指令有效，不能反映行走精度。
