# 感知

感知模块把相机图像变成几何量，具体是箱子在哪里、箱子正面这个平面的位置和朝向、标签在面上的位置和条纹方向，以及条码符号的实际宽度。它只做计算，不移动机器人，也不保存状态。拍照、坐标系转换、写入世界状态、决定何时再看一次，都由工具层（`g1agent/tools/toolbox.py`）负责。

## 作用与在系统中的位置

感知代码在 `g1agent/perception/` 下，共四个文件（`__init__.py` 为空），参数集中在 `configs/perception.yaml`：

| 文件 | 作用 | 依赖 |
|---|---|---|
| `g1agent/perception/geometry.py` | 相机几何基础函数：反投影、像素射线、投影、视场判断、射线与平面求交、两条射线三角化、到平面的水平距离 | 只用 numpy |
| `g1agent/perception/box_detector.py` | `BoxDetector`：用 GroundingDINO 做开放词汇检测，后接 NMS 和"容器框"过滤。箱子和标签都由它检测 | transformers、torch |
| `g1agent/perception/plane_fit.py` | `fit_front_face`：用检测框内的深度拟合箱子的竖直前表面 | `geometry.backproject`、cv2 |
| `g1agent/perception/barcode_detector.py` | 条码/标签定位（cv2 检测器、GroundingDINO 标签检测加 cv2 精修），以及用 zbar 多角度解码取符号轮廓 | cv2、pyzbar、`BoxDetector` 实例 |
| `configs/perception.yaml` | 上述模块的全部参数 | — |
| `scripts/check_m3.py` | 里程碑 M3 验收脚本：比较头部相机估计的面法向、条码中心与真值 | 仿真器仓库、`Toolbox`、评测客户端 |

和其它组件的接口：

| 对方 | 方向 | 内容 |
|---|---|---|
| 工具层 `toolbox.py` | 调用感知 | `detect_boxes` → `BoxDetector.detect`；`_fit_one`、`refine_with_wrist` → `fit_front_face`；`detect_barcode_head`、`_labels_on_rectified_face`、`_detect_wrist` → `detect_label_dino` / `detect_barcodes`；`_wrist_symbol_width` → `decode_symbol_polygons`；视场检查、射线求交、三角化、站位距离 → `geometry` 中的函数 |
| 工具层 | 感知返回 | 平面拟合的结果在机器人根坐标系（root）中，标签检测的结果是像素坐标。工具层负责转到里程计坐标系（odom），再存入 `WorldState` |
| 编排器 `g1agent/orchestrators/rule.py` | 调用 | `geometry.pixel_ray` |
| 评测 `g1agent/eval/evaluator.py` | 调用 | `geometry.project`、`geometry.in_image`（边距 0.0）、`geometry.horizontal_distance_to_plane` |
| 配置 `g1agent/common/config` | 读取 | 各函数用 `config.load("perception")[...]` 读对应小节；也接受一个可选的 `cfg` 参数覆盖配置（测试和工具箱这样用） |
| `configs/system.yaml` | 读取 | `harness_client.detector_device`：检测器运行的设备 |

坐标约定（`geometry.py` 文件头）：相机坐标系用 ROS optical 约定，x 向右、y 向下、z 向前；`K` 为 3×3 内参矩阵；`T_a_b` 把 b 坐标系中的坐标变换到 a 坐标系。

系统里有两台相机，感知对它们的处理完全不同：

| 相机 | 焦距 fx | 在感知中的用途 |
|---|---|---|
| 头部相机（RGB + 深度） | 约 243 px | 检测箱子、拟合箱面、定位标签。分辨率低，8 cm 的 code128 在 0.8 m 外只占约 24 px |
| 腕部（扫描）相机（只有 RGB） | 1087 px | 近距离重新定位标签、解码并测量符号宽度。能直接分辨条纹 |

```{figure} images/perception_head_view.jpg
:width: 480px

头部相机看到的印花外观箱堆（场景 stack9p，回合开始时的画面）。标签只占十几到几十个像素，条纹分辨不出来，所以头部相机只负责"找到标签在哪"，读码交给腕部相机。
```

## 当前实现

### 1. 相机几何：`geometry.py`

所有函数都是无状态的纯 numpy 运算。

**`backproject(K, u, v, z)`**：像素 (u, v) 加上沿光轴的深度 z，算出相机系中的点（N×3）。公式为 `x = (u - cx)/fx · z`，`y = (v - cy)/fy · z`。调用方是 `plane_fit.fit_front_face`。

**`pixel_ray(K, u, v)`**：经过像素 (u, v) 的单位方向向量，即 `[(u-cx)/fx, (v-cy)/fy, 1]` 归一化。调用方有 `orchestrators/rule.py`，以及工具箱的 `_pixel_on_face`（像素射线与目标面求交）和 `stereo_check_wrist`（两次腕部观测的射线）。

**`project(K, T_cam_x, pts)`**：把 x 坐标系中的点投影到图像，返回 `(uv, depth)`。除法放在 `np.errstate(divide="ignore", invalid="ignore")` 里，深度为 0 或负时得到 inf/NaN 而不报错。点是否在相机前方交给 `in_image` 判断。

**`in_image(uv, z, w, h, margin_frac=0.0)`**：点满足 `z > 0`，且 u 在 `[margin·w, w - margin·w]`、v 在 `[margin·h, h - margin·h]` 内时为真。工具箱的 `compute_obs_pose`、`check_face_in_view`、`_low_view_update` 都用 `fov_margin_frac = 0.05` 作为边距（设计文档 AGENT §4.2 第 4 步：面的角点必须落在头部图像内，留 5% 边距）。评测端 `_task1` 用 0.0。

**`ray_plane(origin, direction, plane_point, plane_normal, min_cos=1e-3)`**：射线 `origin + t·direction`（要求 t > 0）与平面的交点。`|direction·normal| < min_cos`（近似平行）或 `t <= 0`（交点在射线后方）时返回 `None`。

**`triangulate(o1, d1, o2, d2)`**：两条射线最短连线段的中点和该线段长度 `gap`，用最近点闭式解。`|den| < 1e-12`（射线平行）时返回 `(None, inf)`。调用方是工具箱的 `stereo_check_wrist`：腕部相机平移约 4 cm 后拍第二张图，与第一张三角化（AGENT §4.3 第 7 步）。

**`horizontal_distance_to_plane(p, plane_point, plane_normal)`**：沿平面的水平法向测量点到平面的有符号距离（法向一侧为正）。先把法向的 z 分量置 0 再归一化，然后取 `(p - plane_point)[:2] · n[:2]`。法向水平分量的模长小于 `1e-9`（平面是水平的）时抛出 `ValueError("plane is horizontal")`，不做降级。调用方是工具箱的 `_fit_one`（机器人到目标面的距离）、`_plan_stance`（站位约束 `min_face_dist_m`）和评测 `_task1`。

### 2. 箱子检测：`box_detector.py`

**`BoxDetector.__init__(device=None)`**：在构造函数内部才导入 transformers，按 `box_detector.model`（`IDEA-Research/grounding-dino-tiny`）加载 processor 和模型，放到 `device` 上（没有传入时读 `system.yaml` 的 `harness_client.detector_device`）。工具箱接受外部注入的实例；`scripts/run_task.py` 和 `scripts/check_m3.py` 各自构造一个实例并注入。同一个实例也用来检测标签。

**`detect(rgb, prompt=None)`**（在 `torch.no_grad()` 下运行）分五步：

1. 文本提示默认是 `"a cardboard box."`。调用方可以换成别的提示，`detect_label_dino` 就传入标签提示。
2. 调用 `post_process_grounded_object_detection`，阈值 `box_threshold = 0.30`、`text_threshold = 0.25`（GroundingDINO 模型卡的默认值），`target_sizes` 设为原图尺寸，所以输出框直接是原图像素坐标。label 优先取 `text_labels` 字段，没有时取 `labels`，兼容不同版本的 transformers。
3. 按得分从高到低做贪心 NMS：一个框与所有已保留框的 IoU 都小于 `nms_iou = 0.5` 才保留。
4. 容器过滤：一个框如果"包含"了至少 `container_min_children = 2` 个其它框，就删掉它，因为它框住的是一整组箱子。"包含"的意思是子框面积至少 `container_overlap = 0.8` 落在这个框里。所有框都基于 NMS 之后的同一份列表计数。
5. 按框中心横坐标从左到右排序后返回 `{"bbox", "score", "label"}`。

辅助函数 `_iou`、`_frac_inside` 的分母都加了 `1e-9`，防止除零。

### 3. 箱子前表面拟合：`plane_fit.py`

`fit_front_face(depth, K, T_root_cam, bbox, cfg=None)` 用头部深度图里检测框内的像素，拟合箱子朝向相机的那个竖直面。输出在 root 坐标系中。设计思路（文件头引用 AGENT §4.2 第 3 步）是让约束本身排除错误的平面：RANSAC 只接受法向接近水平、并且朝向相机的假设，所以顶面、桌面、地面从构造上就不会被选中。

| 步骤 | 做法 | 参数 |
|---|---|---|
| 1. 采样 | bbox 取整并裁剪到图像内，按 `stride` 做网格采样，取出深度 | `stride = 2` |
| 2. 有效深度 | 保留有限值，且 `0.2 < z < 3.0` m 的像素；少于 `min_inliers` 个就返回失败，原因里写明有效像素数和深度范围 | `depth_valid = [0.2, 3.0]`（相机规格，与仿真器一致）；`min_inliers = 150` |
| 3. 反投影 | `backproject` 得到相机系点，用 `T_root_cam` 变换成 root 系点集 `P` | — |
| 4. 局部法向 | 把点放回采样网格 `G`（无效位置填 NaN）。网格任一维少于 2 就失败（"detection box too small for local normals"）。用 `np.gradient` 求两个方向的差分（内部用中心差分，边界用单侧差分），叉乘后归一化，得到每个点的局部法向 `Pn`。与无效深度相邻的点，法向为 NaN | — |
| 5. RANSAC | `np.random.default_rng(seed)`，迭代 `ransac_iters` 次。每次不放回地抽 3 个点叉乘得到法向（模长 `< 1e-9` 跳过）；若 `n·(cam - a) < 0` 则翻转，使法向朝向相机；`|n_z| > sin(max_normal_tilt_deg)` 跳过（不是近似竖直的面）；统计 `|(P - a)·n| < inlier_dist` 的点数，保留最多的假设。这一阶段不看局部法向 | `ransac_iters = 400`；`max_normal_tilt_deg = 20`（AGENT §4.2 第 3 步）；`inlier_dist = 0.01` m；`seed = 0` |
| 6. 合格检查 | 没有合格假设，或最优假设内点少于 `min_inliers`，返回失败 | `min_inliers = 150` |
| 7. 二次筛选 | 最终内点必须同时满足：到平面的距离小于 `inlier_dist`，且 `|Pn·n| >= cos(max_normal_tilt_deg)`（局部法向与平面法向一致，取绝对值，不区分符号）。法向为 NaN 的点不计入。筛完少于 `min_inliers` 返回失败 | 与平面使用同一个倾角上限 |
| 8. 精修 | 内点去中心后做 SVD，取最小奇异值对应的向量作为新法向，并保持与 RANSAC 法向同向 | — |
| 9. 面内坐标轴 | `up` 为世界 z 轴在平面内的投影；`right = up × n`（从相机一侧看指向右） | — |
| 10. 矩形 | 内点投到 `(right, up)` 二维坐标，`cv2.minAreaRect` 求最小外接矩形，`cv2.boxPoints` 取 4 个角点并还原到 3D | — |

成功时返回 `status="ok"`、`center`、`normal`、`right`、`up`、`corners`（4×3）、`size=[宽, 高]`、`n_inliers`、`n_valid`、`inlier_frac`、`rms`（内点到平面距离的均方根）。失败时返回 `status="fail"`、可读的 `reason` 和 `n_valid`。函数内没有 try/except，每条失败路径都通过返回值上报。

`min_inliers` 在代码里用了三次：有效像素数下限、RANSAC 内点数下限、二次筛选后的内点数下限。

### 4. 条码与标签定位：`barcode_detector.py`

这个文件只做定位和测宽，读码由仿真器的扫描接口 `robot.scan()` 完成。

#### 4.1 条纹方向 `_bar_direction(gray, quad)`

在四边形掩码内用 Sobel（`ksize=3`）求 `gx`、`gy`，构造结构张量 `J = [[Σgx², Σgxgy], [Σgxgy, Σgy²]]`，用 `np.linalg.eigh` 分解。最大特征值对应的特征向量是横穿条纹的梯度方向，函数返回它的垂直向量（沿条纹的方向）。同时返回一致性 `coherence = (λ1 - λ0)/(λ1 + λ0 + 1e-9)`，取值 0 到 1，越接近 1 表示方向越单一。

#### 4.2 cv2 检测器 `detect_barcodes(rgb, roi=None, cfg=None)`

1. ROI 默认为整张图，四边各外扩 `roi_pad_frac = 0.05`（相对 ROI 的宽或高），再裁剪到图像内。宽或高小于 4 px 时返回空列表。
2. 转灰度，按 `upscale` 倍用 `INTER_CUBIC` 放大。
3. 依次运行 `cv2.barcode.BarcodeDetector().detectMulti` 和 `cv2.QRCodeDetector().detectMulti`，kind 分别记为 `"barcode"`、`"qr"`。
4. 在放大图上算 `_bar_direction`，角点映射回原图（`q / s + [x0, y0]`）。
5. 输出 `kind`、`corners`（4×2）、`center`、`bar_dir`、`coherence`、`width_px`（两条相邻边中较长的那条），按四边形面积从大到小排序。

调用方：工具箱的 `_detect_wrist` 在整张腕部图上调用它，用 `upscale = wrist_upscale = 1`，因为腕部相机能直接分辨条纹；检测不到时改用 `detect_label_dino`。`detect_label_dino` 也在每个标签框里调用它做精修。

#### 4.3 GroundingDINO 标签检测 `detect_label_dino(rgb, roi, box_detector, cfg=None)`

这是头部相机定位标签的主路径，输出格式与 `detect_barcodes` 相同。

1. ROI 取整并裁剪到图像内（这里不外扩），宽或高小于 4 px 时返回空。
2. 把 RGB 裁剪图按 `upscale = 3` 倍用 `INTER_CUBIC` 放大（保留彩色）。放大的依据：头部相机看到的标签约 20–40 px 宽。
3. 临时替换阈值：保存 `box_detector.cfg["box_threshold"]`，改成 `label_threshold = 0.25`，用 `label_prompt = "a barcode label."` 调用 `box_detector.detect`，然后在 `finally` 中恢复原值。这里只用 try/finally 保证恢复，不捕获异常。`text_threshold`、NMS 和容器过滤沿用箱子检测的设置。
4. 面积过滤：面积超过裁剪图 `label_max_area_frac = 0.5` 的检测丢掉，因为那是箱面本身。
5. 包含过滤：一个检测框如果包含了另一个检测框（`_encloses(a, b)`：b 完全在 a 内且面积严格更小，相等的框互不包含），就丢掉外层框。理由是标签里不会再套着标签，外层框是承载标签的箱面；同一标签上的两个嵌套框，较松的那个也会因此被去掉。
6. 对保留下来的每个检测：映射回原图，得到轴对齐四边形；记录 `kind="label"`、`corners`、`center`、`score`、`width_px`（框宽）；在原分辨率灰度图上算 `bar_dir`、`coherence`；再以该框为 ROI 调用 `detect_barcodes`，cv2 检测得到结果时，用面积最大的那个结果覆盖 `kind`、`corners`、`center`、`bar_dir`、`coherence`，并标记 `refined_by_cv2=True`。`width_px` 和 `score` 不被覆盖。
7. 按 `score` 从高到低排序后返回。

调用方：

| 调用方 | ROI |
|---|---|
| `toolbox.detect_barcode_head`（任务 2 路径） | 目标面 4 个角点投影后的外接框 |
| `toolbox._labels_on_rectified_face`（任务 3 的 `plan_scans`） | 用单应变换拉成正视图的整张箱面图。采样密度取 `fx / 最近角点深度`（px/m），即源图上最细的那处采样 |
| `toolbox._detect_wrist` | 整张腕部图，作为 cv2 检测失败时的后备 |

#### 4.4 zbar 多角度解码 `decode_symbol_polygons(rgb, rotation_step_deg)`

返回 `{内容: [多边形, ...]}`，每个能解出该内容的旋转角度贡献一个多边形（原图像素坐标）。

1. 转灰度。对 `np.arange(0, 90, rotation_step_deg)` 中的每个角度，求绕图像中心的旋转矩阵，并扩大画布使旋转后不裁掉任何像素（新宽 `H·|sin| + W·|cos|`，新高同理），空白处填 255。
2. 在旋转图上运行 `pyzbar.decode`，用逆仿射把多边形顶点映射回原图。
3. 多边形有任一顶点满足 `x < 1`、`y < 1`、`x > W-2` 或 `y > H-2`（贴到图像边缘）时跳过，因为这是被图像边缘截断的符号，轮廓不完整。
4. 按解码内容分组累积。

需要多角度的原因：zbar 只能读与图像轴夹角约 ±17° 以内的一维码。`wrist_decode_rotation_step_deg = 15`，即旋转 0、15、…、75°。这个取值沿用仿真器扫描器（`g1_barcode_sim` 的 `configs/scanner.yaml` 中 `scan.rotation_step_deg`）当时的值 15。之后仿真器扫描器按"真实扫码枪读不了与画面倾斜的码"这一决定改成了 90（只解原图），腕部测宽这里仍是 15，`perception.yaml` 里"与仿真器扫描器相同"的注释没有随之更新。

调用方是工具箱的 `_wrist_symbol_width`：取中心离检测点最近的那个内容的所有轮廓，把轮廓像素射线投到面平面上，在面内用 `cv2.minAreaRect` 求最小外接矩形取长边，再对所有轮廓取中位数，得到符号宽度 `width_m`，同时返回轮廓数 `n_outlines`（用于"支持度"判断，见演进记录 v6）。

```{figure} images/perception_wrist_view.jpg
:width: 400px

腕部（扫描）相机近距离看到的一维码。标签相对图像轴转了二十多度，超出 zbar 约 ±17° 的可读范围，所以 `decode_symbol_polygons` 把图像旋转到多个角度分别解码。
```

### 5. M3 验收脚本：`scripts/check_m3.py`

M3 是需求文档里的感知里程碑：在 L0 任务 2 场景中，把机器人放在真值观测位姿上，比较头部相机估计的目标面法向、条码中心与真值。通过标准为法向误差 < 3°、条码中心误差 < 2 cm。

- 命令：`python scripts/check_m3.py [--suite t2_L0] [--device cpu]`。
- 只构造一次机器人接口、评测客户端和 `BoxDetector`，所有场景共用。每个场景先 reset，再新建一个 `Toolbox`，依次调用 `detect_boxes`、`select_target_box`、`fit_front_face`、`detect_barcode_head`、`estimate_barcode_pose` 这 5 个工具。这 5 个调用写在同一个列表表达式里，前一步失败后后面几步仍然执行。
- 5 步全部成功才计算指标：`normal_err_deg`（法向转到世界系后与真值的夹角）、`barcode_center_err_m`（世界状态里**第一个**条码的中心与真值目标条码中心的距离）、`face_center_err_m`（只记录，不参与判定）。任一步失败则 `pass=False`。
- 每个场景打印一行 JSON，汇总写到 `runs/check_m3_<suite>.json`。

真值只在这个验收脚本里通过评测客户端读取；感知和工具层的算法代码不使用真值，由 `tests/test_no_gt_leak.py` 检查。

### 6. 参数一览与依据

带测量或文档依据的参数：

| 参数 | 值 | 依据 |
|---|---|---|
| `box_detector.box_threshold` / `text_threshold` | 0.30 / 0.25 | GroundingDINO 模型卡默认值 |
| `plane_fit.depth_valid` | [0.2, 3.0] m | 头部深度相机规格，与仿真器一致 |
| `plane_fit.max_normal_tilt_deg` | 20 | AGENT §4.2 第 3 步 |
| `fov_margin_frac` | 0.05 | AGENT §4.2 第 4 步 |
| `barcode.upscale` | 3 | 头部相机看到的标签约 20–40 px 宽 |
| `barcode.label_threshold` | 0.25 | 14 张头部视图上的得分分布（见 v7） |
| `barcode.wrist_upscale` | 1 | 腕部相机 fx 1087 px，可直接分辨条纹 |
| `barcode.wrist_decode_rotation_step_deg` | 15 | zbar 的 ±17° 限制；沿用仿真器扫描器改为只解原图之前的取值 |

只说明了用途、没有给出数值依据的参数：`nms_iou`（0.5）、`container_min_children`（2）、`container_overlap`（0.8）、`stride`（2）、`ransac_iters`（400）、`inlier_dist`（0.01 m）、`min_inliers`（150）、`roi_pad_frac`（0.05）、`label_max_area_frac`（0.5）、`label_prompt`、`box_detector.prompt`。

写死在代码里、没有进入配置的常量：

| 位置 | 常量 | 说明 |
|---|---|---|
| `detect_barcodes` / `detect_label_dino` | ROI 宽或高 < 4 px 时返回空 | 无依据说明 |
| `_bar_direction` | Sobel `ksize=3`；分母加 `1e-9` | 无依据说明 |
| 两处放大 | `cv2.INTER_CUBIC` | 无说明 |
| `decode_symbol_polygons` | 旋转范围 [0, 90)；填充 255；贴边判据 `< 1`、`> W-2 / H-2` | 范围依据 zbar 的 ±17°；贴边判据只说明用途 |
| `ray_plane` | `min_cos = 1e-3` | 无依据说明 |
| `triangulate`、`horizontal_distance_to_plane`、`plane_fit`、`_iou` | `1e-12`、`1e-9` | 数值容差 |
| `check_m3.py` | 3.0°、0.02 m | 需求文档的 M3 标准 |

## 演进记录

下面按时间记录感知模块每一次改动。有几次改动的代码落在工具层（`toolbox.py`），但改的是"怎样调用感知、怎样使用感知结果"，问题也出在感知上，所以一并记在这里。

### v0 · 2026-11-05 · 初版：GroundingDINO 检箱子、深度拟合前表面、相机几何、cv2 检条码

**初版设计。**

- `box_detector.py`：GroundingDINO（`grounding-dino-tiny`，提示 `"a cardboard box."`，阈值 0.30 / 0.25 取模型卡默认值），NMS 的 IoU 为 0.5，外加容器过滤（包含 2 个以上其它框、每个子框 80% 以上面积落在其中的框判为"一整组箱子"，丢掉）。写完后先在任务 2 的预览图上试跑了检测（日志没有保留这次的得分）。
- `plane_fit.py`：在检测框内按步长 2 采样深度，有效范围 [0.2, 3.0] m，RANSAC 只接受近似竖直、朝向相机的平面，再用 SVD 精修，`minAreaRect` 求面矩形。初版只按"到平面的距离"判定内点，没有局部法向检查。
- `geometry.py`：反投影、像素射线、投影、射线与平面求交、两视图三角化，统一用 ROS optical 坐标系。
- `barcode_detector.py`：在放大后的 ROI 上运行 cv2 的 `barcode` 和 `QRCodeDetector`，在检出的四边形里用结构张量求主梯度方向，取其垂直方向作为条纹方向。

### v1 · 2026-11-05 · 头部相机条码定位改为"GroundingDINO 找标签 + cv2 精修"

**当时的版本**：v0 的 `detect_barcodes`，即直接在头部相机的箱面 ROI 上运行 cv2 条码/QR 检测器。

**遇到的问题**：在 t2_L0 和 t2_L1_barcode_type 的头部预览图上，把 ROI 放大 1/2/3/4/6 倍检测，默认参数下一次也没检出（配置注释记为 0/42）。再在 t2_L0 和 t2_L1_barcode_size 的 18 张预览图上扫参数（梯度阈值 8/16/32、两组 detector scales、放大 2/4 倍），最多只有 5 张检出（5/18）。

**根因**：头部相机 fx 约 243 px，8 cm 宽的 code128 在 0.8 m 外只占约 24 px，每个模块不到 0.2 px。条纹在像素层面就不存在，放大和调参都恢复不出来。

**怎么改的**：

- 新增 `detect_label_dino`：在箱面裁剪图上放大 3 倍（`upscale: 3`），用同一个 GroundingDINO 实例、换提示 `"a barcode label."` 定位"标签"这个整体（白底贴纸），不要求看清条纹；cv2 检测器在标签框里能检出时，再用它精修角点和条纹方向。
- 标签阈值 `label_threshold` 初定为 0.35，当时配置里的注释是"标签得分 0.50–0.65，整个箱面约 0.33"。这批数据只来自素色箱子（后来在 v7 被修正）。
- 头部相机从此只负责"找标签在哪"，解码交给腕部相机近距离扫描（`robot.scan()`）。

**改后效果**：从 t2_L0、t2_L1_barcode_size、t2_L1_barcode_type、t2_L1_background 中选的 13 张预览图上，GroundingDINO 都检出了标签（13/13）。M3 验收（`scripts/check_m3.py`，L0 场景）8/8 通过，箱面法向误差 ≤ 0.2°，头部相机估计的条码中心误差 ≤ 1.6 cm（标准为 < 3° 和 < 2 cm）。

### v2 · 2026-11-07 · 新增 zbar 多角度解码，用腕部相机实测符号宽度

**当时的版本**：任务 3 的扫描距离（stand-off）按标签宽度自适应计算，宽度只来自头部相机的 GroundingDINO 标签框。

**遇到的问题**：任务 3 的三种编排模式下，扫描枪尖端都有约 4 cm 的系统误差（3.6–3.9 cm）。查下来，头部相机估计的标签宽度比实际偏大约 20%（10 cm 的标签测成 12 cm），自适应距离因此远了约 4 cm。扫码本身没受影响，但宽度估计需要校准。

**根因**：头部相机分辨率太低。0.10 m 的标签在头部图像里只有约 35–37 px 宽，框边松几个像素就差 20%。这个误差在头部相机上无法消除。

**离线实验**：用仿真器自己的标签贴图合成了 120 张腕部相机图像（code128，有/无白底，3 种尺寸 × 4 个距离 × 5 个旋转角，宽 0.05–0.10 m，距离 0.20–0.40 m，角度 0–80°），比较几种测宽方法：

| 方法 | 检出 | 宽度误差 |
|---|---|---|
| 头部 GroundingDINO 框 | — | 约 +20% |
| 腕部 cv2.barcode 框 | 115/120 | 均值 +16.2%，中位数 +20.1%，最差 −58.4% |
| pyzbar 解码多边形 + SVD 主方向 | 97/120 | 均值 −3.0%，中位数 +0.5%，有离群 |
| pyzbar 多边形，用结构张量条纹方向 | — | 均值 −7.6% |
| pyzbar 多边形，去掉贴边多边形，多角度取中位数（采用） | 97/120 | 均值 −0.4%，中位数 +0.5%，97 个中 96 个在 ±3% 内 |

唯一的离群样本是 5 cm 的码在 0.30 m 处（偏小约 65%）。漏检的 23 个大多是 5 cm 小码离得太远。

**怎么改的**：

- `barcode_detector.py` 新增 `decode_symbol_polygons(rgb, rotation_step_deg)`：在 0、15、…、75° 多个角度旋转图像（扩大画布，不裁掉内容），用 zbar 解码，把多边形映射回原图；贴到图像边缘（1 px 以内）的多边形是被截断的残缺码，丢弃。`wrist_decode_rotation_step_deg: 15` 写入配置，与仿真器扫描器当时的取值相同。
- 工具层新增 `_wrist_symbol_width`：把轮廓投到面平面，用最小外接矩形的长边作为宽度，多个轮廓取中位数。
- 宽度的定义改为"印刷符号本身"（条和模块，不含白底），`fill_frac` 改为 0.68，使标称 8 cm 的码仍落在已验证的 0.20 m 处。头部估计只用来规划站位和预位姿（因为偏大，所以偏保守），到达预位姿后换成腕部实测值再算最终距离。

**改后效果**（规则模式，任务 3）：

| 场景 | 腕部实测宽度 | 目标距离 | 枪尖误差 | 用时 |
|---|---|---|---|---|
| t1_L0_0000（8 cm 码） | 8.0–8.3 cm | — | 0.7 / 1.0 cm | 453 s |
| t2_L1_barcode_size_0001（5 cm 码） | 5.1 cm | 0.12 m | 0.4 / 0.5 cm | 356 s |
| t2_L1_barcode_size_0009（12 cm 码） | 12.1–12.2 cm | 0.30 m | 0.5 / 0.8 cm | 414 s |

三个场景都是 2/2 个码、顺序正确，枪尖误差从约 4 cm 降到 0.4–1.0 cm。代价是 t1_L0_0000 的仿真时间从 72 s 增加到 109 s（预位姿从 0.34 m 变成 0.40 m，多了一轮修正）。有一次 5 cm 码在 0.31 m 的预位姿上没解出，系统保留了头部估计，更近处测到 5.1 cm 后距离正确。

之后在工具层又加了两道保护，都和"感知结果不可信时怎么办"有关：腕部重检测和双目三角化给出的新位置偏离原估计超过一个标签宽度时不采用（曾经有误检把码位置从 1.10 m 改到 1.70 m）；头部估计的宽度只能把扫描距离拉远、不能拉近到 `offset_m = 0.20` 以下（头部框有时偏小，8 cm 的码测成 3.5 或 4.9 cm，距离被压到 0.12 m 后读不出码）。

### v3 · 2026-11-07 · 前表面拟合加入局部法向一致性检查

**当时的版本**：`fit_front_face` 的内点只看"到平面的距离 < `inlier_dist`（1 cm）"。

**遇到的问题**：在落地箱堆的观察位视图上，拟合出的面宽 1.3 m（y 从 −0.715 到 0.601 m），实际只有 0.40 m。

**根因**：箱堆底部有一条地面像素，恰好落在箱面平面的 1 cm 范围内（168 个内点在 |y| > 0.22 m 处，z 约为 0.002 m）。这些点离平面很近，按距离判断就是内点，但它们属于地面，法向朝上。

**第一次修改（无效）**：用采样网格中"右边一格"和"下边一格"两个邻点叉乘求每个点的法向，要求与平面法向一致。宽度没有变化。检查发现，地面条带上有 50.6% 的点因为邻点缺失（相邻处深度无效）得到 NaN 法向，而当时 NaN 法向的点没有被排除。

**第二次修改（采用）**：

- 改用 `np.gradient` 求网格差分（内部中心差分，边界单侧差分），叉乘得到局部法向；网格任一维少于 2 时返回失败（"detection box too small for local normals"）。
- 最终内点必须同时满足"离平面近"和"局部法向与平面法向一致"（`|Pn·n| >= cos(max_normal_tilt_deg)`，沿用同一个 20° 上限）；局部法向为 NaN 的点不计入。

这是从"其它表面的点虽然离平面近，但法向不同"这个原理出发的通用修复，没有对地面做特殊处理。

**改后效果**：同一组离线视角上面宽恢复为 0.40 m、高 1.14 m：

| 视角 | 修改前 y 范围 (m) | 修改后 y 范围 (m) |
|---|---|---|
| obs0.6 站立 | −0.715 … 0.601 | −0.196 … 0.199 |
| d0.9 站立 | −0.473 … 0.526 | −0.201 … 0.196 |
| d1.2 站立 | −0.394 … 0.448 | −0.195 … 0.195 |
| obs0.6 下蹲 0.45 | −0.538 … 0.306 | −0.196 … 0.198 |

### v4 · 2026-11-07 · 先把箱面矫正成正视图，再检测标签

**当时的版本**：`plan_scans` 把面角点投影到头部原图上，取外接框作为 ROI，放大 3 倍后调用 `detect_label_dino`。

**遇到的问题**：在落地的 3 箱堆上，从观察位只检测到 3 个标签中的 1 个。

**根因**：头部相机向下约 47°，箱面下部的标签离相机远、又被透视压扁，在原图中只剩约 20×5 px 的细条。整块 ROI 按同一倍数放大，压扁的形状不变，GroundingDINO 认不出来。

**怎么改的**（工具层 `toolbox._labels_on_rectified_face`）：用拟合出的平面和相机几何计算单应变换，把整个箱面矫正成正视图像。采样密度取 `fx / 最近角点深度`，也就是原图上最细的那处采样，所以矫正不会丢掉原图的细节。在矫正图上整张运行同一个 `detect_label_dino`，检测中心映射回平面得到 3D 位置。

**改后效果**：离线复核 obs0.6 视角 3/3（z = 1.099 / 0.568 / 0.189 m，位置误差约 1 cm，自上而下排序正确），d0.9 视角 3/3，d1.2 视角 2/3。

### v5 · 2026-11-07 · 规划阶段增加"蹲下再看一次"（单档）

**当时的版本**：`plan_scans` 只在站立观察位看一次。

**遇到的问题**：在 3×3 的箱堆 stack9（5 个码）第一轮中，规则模式在观察位只检测到 4/5 个码，漏了左下角的码（约 0.30 m 高）；完整 LLM 编排只检测到 3/5，左下和右下两个底层码都漏了。规则模式在扫完中间码、仍处于下蹲状态时重新检测，才补上左下码，但因为它属于最左列，被插到计划最前面，扫码顺序就乱了。

**根因**：头部相机从约 1.2 m 高往下斜看，墙越宽，底部两角离相机越远、看得越斜，分辨率最低，GroundingDINO 检不出来。

**怎么改的**（工具层 `_low_view_update` / `_merge_into_plan`）：站着规划完成后，工具自己蹲下，再拍一次头部图像，检测新标签并合并进计划，然后站起来。蹲多低不是手选的：用实测的下蹲标定数据和头部相机几何，预测每一档的头部相机位姿，取"底部角点处每米像素数"最大的那一档。

**改后效果**（stack9 第二轮）：规则模式站着看到 4 个码，蹲下补上 1 个；完整 LLM 编排站着看到 3 个码，蹲下补上 2 个。两种模式开扫前计划里都有完整的 5 个码，顺序为左上 → 左下 → 中 → 右上 → 右下。这个"只取一档"的选法后来在 v8 被修正。

### v6 · 2026-11-07 · 腕部宽度测量加支持度保护

**当时的版本**：v2 中 `refine_with_wrist` 每次测到腕部宽度就覆盖原值。

**遇到的问题**：stack9 第一轮中，规则模式扫左下码失败（重试 1 次后仍读不出）。日志显示，一次宽度为 0 的错误腕部测量覆盖了之前正确的码宽，扫描距离因此算错。代码注释记录的情形是：一个退化轮廓测出 0.0 m，覆盖了此前两次各有 3 个轮廓支持的 8.5 cm 测量值，距离掉到 0.12 m，读不出码。

**根因**：`decode_symbol_polygons` 偶尔只返回一个退化轮廓，宽度测出 0。原逻辑不区分"一个轮廓的结果"和"多个旋转角一致的结果"。

**怎么改的**：`world_state.BarcodeEstimate` 新增 `width_support` 字段，记录腕部宽度是多少个解码轮廓的中位数。新测量的宽度必须大于 0；如果已有值也来自腕部，新测量的支持数还必须不少于它，才替换原估计（已有值来自头部时，任何正宽度的腕部测量都会替换）。

**改后效果**：修复后 72 个测试通过。stack9 第二轮中，规则模式的左下码第一次扫描就读到（无重试）。

### v7 · 2026-11-07 · 标签阈值从 0.35 降到 0.25，并新增"包含过滤"

**当时的版本**：`label_threshold = 0.35`（v1 按素色箱子的得分定），`detect_label_dino` 只有面积过滤。

**遇到的问题**：印花快递箱外观的箱堆 stack9p（5 个码）上，观察阶段只规划出 2 个码（左上和左下），中间、右上、右下三个都漏了。离线采集 7 个头部视角复现：站立观察位 2 个，下蹲 0.45 档 3 个，d0.9 站立 2 个。同一场景中箱子检测和平面拟合都正常（1 个框覆盖整个箱堆，拟合面 1.181×1.075 m，内点 5004，法向正确），问题出在标签检测。

**根因**：把箱面矫正成正视图后，5 个标签都清晰可见，但 GroundingDINO 打分偏低。用 0.10 的低阈值对照真值统计，这个视角上真标签的得分是 0.42 / 0.38 / 0.31 / 0.28 / 0.27，非标签（宽 0.40 m 的整个箱面）是 0.19 / 0.10。GroundingDINO 是开放词汇检测器，背景越杂，标签得分越低。旧阈值 0.35 只按素色箱数据定，在印花箱上 5 个丢了 3 个。

两种箱堆共 14 个采集视角的统计：

| 类别 | 得分范围 |
|---|---|
| 素色箱上的真标签 | 0.41–0.72 |
| 印花箱上的真标签 | 0.25–0.51（`detect_label_dino` 的代码注释写的是 0.25–0.42，两处来源不一致，这里照录） |
| 非标签（整个箱面） | ≤ 0.20 |

**怎么改的**：

- `configs/perception.yaml`：`label_threshold` 改为 0.25，注释写明 14 个视角的得分分布。这个值是按这批数据手定的，与非标签之间只有 0.05 的余量，已作为风险报告。
- 阈值降低后，整个箱面有可能被当成标签。`barcode_detector.py` 新增 `_encloses(a, b)`，`detect_label_dino` 在面积过滤之后再丢掉"包含了其它检测框"的框。这条规则不依赖分数，依据是结构：标签里不会再有标签，套着别的框的框是承载标签的箱面。
- 新增单元测试 `test_label_dino_drops_enclosing_box`。

**改后效果**：用真实的 `plan_scans` 在 7 个视角上离线复核，stack9p 观察位从 2/5 提升到 5/5，没有误检；素色箱堆结果不变（3/3）。全量测试 96 个通过、1 个跳过。

### v8 · 2026-11-07 · 纸箱贴纸假标签：检测端不做针对性过滤；下蹲补看改为合并所有合格档位

**当时的版本**：v7 的标签检测（提示 `"a barcode label."`，阈值 0.25，包含过滤）；v5 的单档下蹲补看（取底部分辨率最高的一档，当时是 0.25 m 的深蹲）。

**遇到的问题 1：假标签。** 在纸箱外观的两面场景 stack4c_L0_0001 上，正面第一次规划了 5 个码，最后扩到 7 个，而真值只有 3 个。例如 z = 0.71 m 处的一个计划目标那里根本没有真码，腕部相机近看结果是"整个腕部视野里没有解出任何符号"。纸箱模型的贴图自带小条码贴纸、"PARCEL POST"贴纸和大张面单，上面印着解不出的假条码和二维码图案。检测器在纸箱面上一共检出 35 个框，其中 29 个是这类贴纸。

**排查**：在 stack4c_L0_0001 四个面的观察位上各采集站立和下蹲 0.45 档两组头部图像（共 8 个视角），用真实的 `plan_scans` 流水线离线跑，检测中心离真值标签 4 cm 以内算真码。对比 4 种提示词（印花、素色两列用的是 v7 中采集的视角）：

| 提示词 | 纸箱 真/假 | 印花 真/假 | 素色 真/假 |
|---|---|---|---|
| `"a barcode label."`（当前） | 6 / 29 | 26 / 0 | 14 / 0 |
| `"a barcode."` | 5 / 18 | 22 / 0 | 14 / 0 |
| `"a barcode sticker."` | 6 / 27 | 24 / 0 | 14 / 0 |
| `"a white label with a barcode."` | 5 / 13 | 14 / 0 | 12 / 0 |

换提示词不能消除纸箱上的假检，还会在其它外观上损失真码。逐个比较检测得分、条纹方向一致性、kind、cv2 是否精修成功、宽度：纸箱上真码的得分只有 0.31–0.44，所有特征上真码和贴纸都重叠，没有哪一个特征能把它们分开。

**根因**：贴纸本身就是"印着条码图案的标签"，从头部相机的分辨率看，它与真标签在外观上没有区别。能否解码只有走近用腕部相机才能判断。

**决定**：不在检测端做针对这种外观的过滤（那会是对单一样本的特例），而是在扫码结果端判定：走近后腕部相机确认解不出任何符号时，把该目标标为"无可读码"（`unreadable`），与"无法到达或放置"的 `failed` 区分。这部分实现在工具层的 `scan()`，详见工具层页面。

**遇到的问题 2：下层真码漏检。** 这个场景的第一轮结果是 4/6（漏了正面底层和背面中层），用时 4055 s。除了贴纸，另一个原因是 v5 的单档下蹲补看。逐档实测（11 个下蹲档位 × 有码的 2 个面，两面共 4 个下层真码）：在纸箱纹理上，分辨率最高的 0.25 m 深蹲一个下层真码都找不到；0.40–0.55 m 各档能找到全部 4 个；0.30、0.60、0.65 m 档找到 3 个；0.35 m 档只找到 2 个（每面 1 个）。召回率随高度并不单调。

**根因**：v5 假设"底部像素越多，检测越好"，但开放词汇检测器的召回还取决于视角、纹理和取景范围，像素密度只是其中一个因素。

**怎么改的**（工具层 `_low_view_update`）：遍历所有已标定的下蹲档位，凡是能看全该面底部、并且底部分辨率高于站立时的档位都去看一次，检测结果取并集。档位只来自已有的标定数据，没有新增阈值。另外 `next_barcode` 每扫完一个码，都会自动从当前更近的位置再看一次并更新计划，不再依赖 LLM 自己调用 `plan_scans`。

**改后效果**：离线验证两个面 6/6 全部检出。

## 已知限制

- **标签阈值余量小**：`label_threshold = 0.25` 是按 14 个视角手定的，与非标签最高分 0.20 只差 0.05。换一种箱子外观或光照，需要重新测得分分布。
- **头部相机无法区分真码和印着条码图案的贴纸**：纸箱贴图上的贴纸会进入扫描计划，只能靠腕部相机走近核验后标为 `unreadable`，每个假目标都要花一次走位和抬臂的时间。
- **共面的多个箱子会被检成一个**：三个箱子前面共面时，GroundingDINO 只给出一个框（得分 0.838），平面也拟合成一整块。共面时这不影响找码和排序；箱子之间如果有间隙或前后错位，合并成一个面会出错（推测，尚未在场景中测试）。
- **头部相机测宽偏差大**：头部标签框宽度约偏大 20%，在矫正面上只框到一部分时又会偏小。所以头部估计只用于规划，最终距离依赖腕部测量；腕部也解不出码时，只能沿用保守的头部估计。
- **腕部测宽对远处的小码不可靠**：离线实验中 120 张图漏检 23 张，大多是 5 cm 小码离得太远；唯一的离群是 5 cm 码在 0.30 m 处偏小约 65%。
- **一批参数没有数值依据**：`nms_iou`、`container_*`、`stride`、`ransac_iters`、`inlier_dist`、`min_inliers`、`roi_pad_frac`、`label_max_area_frac` 以及两条提示词，配置里只写了用途。`perception.yaml` 文件头说"tuned"的值是在 L0 预览上设定的，但当前文件没有逐项标注哪些调过。
- **`ray_plane` 的平行判据依赖单位向量**：比较的是点积 `d·n`，只有两个向量都已归一化时它才等于夹角余弦。
- **`check_m3.py` 的指标细节**：条码误差用的是世界状态中第一个条码，场景里有多个标签时可能比错对象；汇总时缺失值以 NaN 参与 Python 的 `max`，结果取决于元素顺序；5 个工具调用写在同一个表达式里，前一步失败后后面几步仍会执行。
- **代码注释有几处过时或与实测记录不一致**：`plane_fit.py` 局部法向那段注释仍写着"neighbours one stride right / down"（v3 第一次修改的做法），实际代码已经改用 `np.gradient`；`perception.yaml` 中 `wrist_decode_rotation_step_deg` 的注释仍说与仿真器扫描器取值相同，但仿真器扫描器已改为 90；`toolbox._low_view_update` 的注释写"0.40–0.60 档找到全部 4 个、0.35 档只找到 1 个"，实测记录是 0.40–0.55 档 4 个、0.60 档 3 个、0.35 档 2 个（见 v8）。
