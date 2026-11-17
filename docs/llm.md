# LLM 服务与客户端

本页介绍 agent 系统里和大模型有关的部分。它分成两层：

- **服务端**：用 vLLM 在本机一张 GPU 上部署多模态模型 Qwen3-VL-8B-Instruct，对外提供 OpenAI 兼容的 HTTP 接口。具体如下：
  - 模型为 Hugging Face 上的 `Qwen/Qwen3-VL-8B-Instruct`，约 80 亿参数，是 Instruct 版，不是 Thinking 版；
  - 以 bfloat16 精度加载，没有量化，vLLM 日志记录权重占 16.8 GiB 显存；
  - 上下文长度设为 32768 token（`configs/system.yaml` 的 `max_model_len`）；
- **客户端**：`g1agent/llm/client.py` 里的 `LLMClient`，负责把对话、图像和工具定义发给服务端，处理截断和超时，并把模型返回的工具调用解析成 Python 对象。

本页涉及的文件（路径相对于仓库根目录 `g1_agent_system/`）：

| 文件 | 作用 |
|---|---|
| `g1agent/llm/client.py` | `LLMClient` 和异常 `LLMTruncated` |
| `configs/llm.yaml` | 端点、模型名、采样、超时、重抽配置 |
| `configs/system.yaml` 的 `vlm_server` 段 | 服务端口、显存比例、上下文长度、conda 环境名 |
| `scripts/start_vllm.sh` | 启动 vLLM 服务 |
| `scripts/smoke_vlm.py` | 冒烟测试：一张图加一个工具，检查能否拿到合法的工具调用 |
| `tests/test_llm_client.py` | 用本地 mock 服务器测试客户端 |

## 作用与在系统中的位置

LLM 在系统里只做一件事：根据对话历史、相机图像和可用工具列表，决定下一步调用哪个工具、参数是什么。工具本身的执行、参数校验、预算和记账都由 Harness 负责（见 Harness 页）。LLM 层不知道机器人和仿真器的存在，只看到消息和工具 schema。

```
编排器 agent_assist / agent_full
   │  messages（文本 + image_part 编码的图像）
   │  tools = registry.openai_tools(...)
   ▼
LLMClient.chat()  ── HTTP（OpenAI chat.completions + tools）──►  vLLM 服务（start_vllm.sh）
   │                                                              Qwen3-VL-8B-Instruct
   │  返回 {content, tool_calls[解析后的参数], attempts, t_start, t_end, ...}
   ▼
编排器把 tool_calls 逐个交给 Harness.call()；log_llm() 把这次交互写入 llm.jsonl
```

和其它组件的接口：

| 对方 | 方向 | 内容 |
|---|---|---|
| `g1agent/harness/registry.py` | 提供给 LLM | `openai_tools(names)` 生成 OpenAI function-calling 格式的工具定义；`agent_full` 按任务白名单 `TASK_TOOLS[task]` 取工具，`agent_assist` 只取 `explore` 加它自己的决策工具 |
| `g1agent/orchestrators/agent_full.py` | 调用 | `llm.chat(clean(msgs), tools=tools, tool_choice="auto")` |
| `g1agent/orchestrators/agent_assist.py` | 调用 | `llm.chat(clean(msgs), tools=DECISION_TOOLS, tool_choice="required")` |
| `g1agent/orchestrators/common.py` | 调用 | `image_message()` 用 `llm.image_part(rgb)` 编码图像，同时把原图存进 episode 目录；`log_llm()` 把回复文本、`reasoning_content`、工具调用、`finish_reason`、`usage`、`t_start`/`t_end` 和每次尝试的记录写进 `llm.jsonl` |
| `g1agent/harness/logger.py` | 消费 | `EpisodeLogger.llm()` 把回复文本（模型写的一句理由）、`reasoning_content`、工具调用和耗时 `t_end - t_start` 作为事件推给视频叠加层 |
| `scripts/run_task.py` | 创建与汇总 | 每个 episode 用同一个 `LLMClient`；结束时把 `llm.stats` 写进 `result.json` 的 `llm` 字段，然后把计数清零。编排器抛出 `LLMTruncated` 时，把它的 `attempts` 写进 `orchestrator_error.txt` 的 `llm_attempts` |

换模型只需改 `configs/llm.yaml` 的 `base_url`、`model`、`api_key_env` 三项：客户端只用了 openai SDK 的 `chat.completions` 和 `tools`，没有用 vLLM 特有的接口（重抽时的 `top_k` 通过 `extra_body` 传入，见下文）。

## 当前实现

### `configs/llm.yaml`：客户端配置

| 键 | 当前值 | 含义与依据 |
|---|---|---|
| `provider` | `openai_compatible` | 说明性字段 |
| `base_url` | `http://127.0.0.1:31800/v1` | 本机 vLLM。换 GPT 时改成 `https://api.openai.com/v1` |
| `model` | `Qwen/Qwen3-VL-8B-Instruct` | 换 GPT 时改成例如 `gpt-4.1` |
| `api_key_env` | `LLM_API_KEY` | 从这个环境变量读 key，key 不写进任何文件；vLLM 接受任意值 |
| `temperature` | `0.0` | 正常请求用贪心解码。配置里没有写依据 |
| `max_tokens` | `null` | 项目的设计原则：不设输出长度上限，生成到 EOS 为止，上下文长度由服务端限制。客户端构造时会检查（见下） |
| `timeout_s` | `900` | 单次请求的超时。依据：生成到上下文上限 32768 token，按 vLLM 日志实测的单请求解码速度约 65 tok/s，约需 500 s；900 s 留有余量（余量大小为手定）。见演进记录 v5 |
| `image_max_side` | `640` | 图像长边超过 640 像素时缩小。配置里没有写依据；仿真头部相机本身就是 640×480，所以对它不起作用 |
| `max_retries_on_length` | `2` | 截断或超时后最多重抽 2 次，即每次 `chat()` 最多 3 次尝试。没有写为什么是 2 |
| `redraw_sampling` | `{temperature: 0.7, top_p: 0.8, top_k: 20}` | 重抽时的采样参数，照抄 Qwen3-VL-8B-Instruct 自带的 `generation_config.json`。温度为 0 时重抽会逐 token 重复同一条失控回复。见演进记录 v5 |

### `LLMClient(cfg=None)`：构造

1. 没传 `cfg` 时用 `config.load("llm")` 读 `configs/llm.yaml`。测试里会直接传一个 dict。
2. **强制不设输出上限**：`cfg["max_tokens"]` 不是 `None` 就抛 `ValueError`。这样即使有人改了配置，也会在启动时就失败，而不是悄悄截断生成。
3. 创建 `openai.OpenAI(base_url, api_key=环境变量值 或 "EMPTY", timeout=timeout_s, max_retries=0)`。
   - `max_retries=0` 关闭 SDK 自带的自动重发。SDK 超时后会把同一个请求原样再发一遍，在温度 0 下得到的还是同一条失控生成；重试统一由 `chat()` 负责，而且换用采样参数。
4. 初始化统计 `stats = {calls, attempts, length_truncations, timeouts, prompt_tokens, completion_tokens, seconds}`。`calls` 是 `chat()` 被调用的次数，`attempts` 是实际发出的请求次数（含重抽），二者之差就是重抽次数。截断率 = `length_truncations / attempts`。

### `image_part(rgb)`：图像编码

输入 RGB uint8 数组，输出一个 OpenAI `image_url` 内容块：

1. 计算 `s = image_max_side / max(h, w)`；`s < 1` 时用 `cv2.INTER_AREA` 缩小，从不放大；
2. RGB 转 BGR 后用 OpenCV 编码成 JPEG，质量 90（无注释依据）；编码失败抛 `RuntimeError`；
3. 返回 `{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}`。

编排器通过 `common.image_message()` 使用它：同一张图一份 base64 发给模型，一份原图存到 episode 的 `images/` 目录，消息里用私有键 `_path` 记住文件路径，发送前由 `common.clean()` 去掉这个键，写日志时用路径代替图像内容。

### `chat(messages, tools=None, tool_choice="auto")`：一次决策

一次 `chat()` 最多做 `1 + max_retries_on_length` 次尝试（当前为 3 次）。每次尝试：

1. **组装参数**：`model`、`messages`、`temperature=cfg.temperature`（第一次尝试为 0.0）。
   - 第 2 次起是**重抽**：复制 `redraw_sampling`，取出 `temperature` 和 `top_p` 作为正式参数，剩下的键（目前只有 `top_k`，它不在 OpenAI 标准参数里）放进 `extra_body`，由 vLLM 识别。
   - 有工具时加上 `tools` 和 `tool_choice`。
   - 请求里从不出现 `max_tokens`。
2. **发请求** `client.chat.completions.create(**kw)`。只捕获 `openai.APITimeoutError`：记一条 `finish_reason: "timeout"` 的尝试记录，`stats.timeouts += 1`，进入下一次尝试。超时被当成截断处理，因为在 900 s 的超时下，只有失控生成（不停重复）才会超时。其他异常（连接失败、HTTP 错误）不捕获，直接抛给调用方。
3. **正常返回**：累加 `attempts`、`seconds` 和 token 用量，把本次尝试的起止时间、`finish_reason`、`usage` 记进 `attempts` 列表。
4. **截断**（`finish_reason == "length"`，即生成碰到了服务端的上下文上限）：`stats.length_truncations += 1`，截断内容存进这次尝试的 `truncated_content` 以便事后查看，**不解析**，进入下一次尝试。
5. **解析工具调用**：对每个 `tool_call` 做 `json.loads(arguments)`，参数为空串时得到 `{}`。只捕获 `json.JSONDecodeError`：此时 `arguments=None`，`parse_error` 写入 `"arguments are not valid JSON: ..."`。解析失败不重抽，交给编排器决定怎么处理（`agent_full` 把 `parse_error` 当作一次失败的工具结果回给模型）。
6. **构造 `assistant_message`**：`{"role": "assistant", "content", "tool_calls"}`，其中 `tool_calls` 保留原始 JSON 字符串，可以直接追加回对话历史。
7. **返回**：

   ```
   {"content", "reasoning_content",            # 服务端不返回思考内容时为 None
    "tool_calls": [{"id", "name", "arguments", "arguments_raw", "parse_error"}],
    "finish_reason", "usage", "assistant_message",
    "t_start",                                  # 整次 chat() 开始的 unix 时间
    "t_end",                                    # 最后一次尝试结束的 unix 时间
    "attempts": [{"t_start", "t_end", "finish_reason", "usage", ["truncated_content"]}]}
   ```

所有尝试都截断或超时后，抛 `LLMTruncated`（`RuntimeError` 的子类），`attempts` 属性带着全部尝试记录。`run_task.py` 把这个异常记为 episode 失败，并把 `attempts` 写进 `orchestrator_error.txt`。

`reasoning_content` 用 `getattr(message, "reasoning_content", None)` 取。当前用的 Instruct 模型不单独输出思考内容，所以它总是 `None`；为了在日志和视频里看到决策理由，提示词要求模型在每次调用工具前用一句话写出理由，这句话就是 `content`。

### `scripts/start_vllm.sh`：服务端

用法：`setsid nohup bash scripts/start_vllm.sh > runs/vllm.log 2>&1 &`，脱离终端在后台运行。

1. 切到仓库根目录。仓库根目录下如果有不进版本库的 `local.env`，先 `source` 它。它可以设置下面用到的环境变量，也可以定义一个 shell 函数 `gpu_launch <label> <cmd...>`，用来把启动命令包进本机的 GPU 排队工具。
2. 用内嵌 Python 读 `configs/system.yaml` 的 `vlm_server` 段（`port`、`gpu_memory_utilization`、`max_model_len`、`conda_env`）和 `configs/llm.yaml` 的 `model`。当前 `system.yaml` 的值是端口 31800、显存比例 0.85、`max_model_len` 32768、环境 `vllm_env`。
3. 激活 conda 环境（`G1_CONDA_PREFIX` 可指定环境所在目录）；清掉 `CC CXX CFLAGS CXXFLAGS CPPFLAGS LDFLAGS`；默认 `HF_HUB_OFFLINE=1`，只从本地缓存读模型。`HF_HOME`、`TMPDIR`、`VLLM_CACHE_ROOT` 由环境变量给出。
4. 公共参数：
   - `serve $MODEL --host 127.0.0.1 --port $PORT --max-model-len $MAXLEN`；
   - `--enable-auto-tool-choice --tool-call-parser hermes`：hermes 是 Qwen2.5/Qwen3 聊天模板的工具调用格式（`<tool_call>{json}</tool_call>`），由冒烟测试验证；
   - `--limit-mm-per-prompt '{"image": 8, "video": 0}'`：项目不发视频，`video: 0` 还能跳过 vLLM 按最大尺寸视频做的显存 profiling（在繁忙的卡上这一步要 10 分钟以上）。单个 prompt 最多 8 张图这个值没有注释依据。
5. 三种启动方式，按顺序判断：
   - **共享卡模式**（设置了 `VLLM_GPU`）：用于和其他任务共用一张 GPU 的情况，要求不得影响卡上的其他任务。必须同时设置 `VLLM_KV_GIB`（KV cache 大小，GiB）和 `VLLM_GMU`（vLLM 可占的显存比例），缺一个就报错退出。追加参数 `--gpu-memory-utilization $VLLM_GMU --kv-cache-memory-bytes <VLLM_KV_GIB × 2^30> --enforce-eager`。固定 KV 大小是为了跳过依赖"其他进程显存不变"的 profiling；`--enforce-eager` 跳过 torch.compile 和 CUDA graph 捕获以加快启动；`--gpu-memory-utilization` 启动时仍会被检查（空闲显存必须够），所以它同时是本进程在这张卡上的显存上限。原因见演进记录 v2。
   - **排队模式**（`local.env` 定义了 `gpu_launch`）：`gpu_launch "vLLM <模型名>" vllm ... --gpu-memory-utilization $GMU`。
   - **直接启动**：`exec vllm ... --gpu-memory-utilization $GMU`。

### `scripts/smoke_vlm.py`：冒烟测试

```{figure} images/llm_smoke_head.jpg
:width: 480px

冒烟测试的默认输入：仿真套件 `t2_L0` 第 0 个场景的头部相机预览图（640×480）。
```

- 用法：`python scripts/smoke_vlm.py [image.png]`。不给图时使用仿真包里 `suites/t2_L0/previews/t2_L0_0000_head.png`。
- 发一次请求：系统提示 "You control a humanoid robot. Always answer by calling exactly one tool."；用户消息是一句"调用 explore 向左移动 0.3 m"的文字加上这张图；工具只有一个手写的 `explore`（它的 direction 枚举里没有 `forward`，和 `registry.py` 不同）。
- 通过条件：恰好一个工具调用，名字是 `explore`，且 `parse_error is None`。输出 `SMOKE_VLM PASS`（退出码 0）或 `SMOKE_VLM FAIL`（退出码 1）。
- 它验证三件事：服务端能收多模态输入；hermes 工具调用解析器能用；`LLMClient` 端到端能用。不检查参数取值。

### `tests/test_llm_client.py`：单元测试

测试在本地起一个 mock 的 OpenAI 兼容 HTTP 服务器，按预设脚本回复。服务器对每个请求断言其中没有 `max_tokens` 和 `max_completion_tokens`。客户端相关的 4 个测试：

| 测试 | 检查内容 |
|---|---|
| `test_tool_call_parsing` | 工具调用参数被解析成 dict |
| `test_truncated_reply_is_redrawn_not_parsed` | 第一次回复截断，第二次正常：返回第二次的工具调用，`length_truncations == 1`；连续截断时抛 `LLMTruncated` |
| `test_redraw_samples_and_timeout_is_redrawn` | 第一次回复故意睡 3 s 超过 2 s 的超时：记为 `timeout`，第二次请求的参数是 `temperature 0.7, top_p 0.8, top_k 20`，第一次是 `temperature 0.0` |
| `test_malformed_arguments` | 参数不是合法 JSON 时 `arguments is None`，`parse_error` 含 "not valid JSON" |

同一文件里还有一个 `test_agent_full_with_mock_llm`，用 mock LLM 驱动 `agent_full` 编排器走完一个任务 1 episode。

## 演进记录

### v0 · 2026-11-05 · 用 vLLM 部署 Qwen3-VL-8B-Instruct

**初版设计。** 决策模型选用 Qwen3-VL-8B-Instruct，因为任务需要模型直接看相机图像并调用工具；用 vLLM 0.11.2 部署成 OpenAI 兼容服务，这样客户端可以和调用 GPT 的代码完全一样。

`scripts/start_vllm.sh` 的第一版：

- 从 `configs/system.yaml` 的 `vlm_server` 段读端口、`gpu_memory_utilization`、`max_model_len`、`hf_home`，从 `configs/llm.yaml` 读模型名；
- 在单独的 vLLM conda 环境里以离线模式（`HF_HUB_OFFLINE=1`）启动；
- 打开 `--enable-auto-tool-choice --tool-call-parser hermes`；
- GPU 通过项目共享的 GPU 排队工具租一张卡，租约备注为 `HumanScan vLLM Qwen/Qwen3-VL-8B-Instruct`，预计时长默认 8h（无依据）。

**效果。** vLLM 日志显示模型加载占用 16.78 GiB 显存，耗时 19.8 s（`Model loading took 16.7803 GiB memory and 19.768 s`）。

### v1 · 2026-11-05 · 客户端初版：不设输出上限，截断就重抽

**初版设计。** `g1agent/llm/client.py` 只用 openai SDK 的 `chat.completions` 和 `tools`。按项目的设计原则：

- 请求里不发送 `max_tokens`，`llm.yaml` 里设了非空值会在构造时报错；
- 回复仍以 `finish_reason == "length"` 结束时不解析，记日志、计数后重新生成，次数由 `max_retries_on_length`（2）控制，截断率在 `stats` 里报告；
- 定义 `LLMTruncated` 异常；图像用 base64 编码放进消息（`image_part`）。

这一版的另外三个设置后来被证明有问题（见 v5）：`timeout_s` 为 120 s；SDK 的 `max_retries` 保持默认，超时后由 SDK 原样重发；重新生成也沿用温度 0。

同时写了 `scripts/smoke_vlm.py`。

**效果。** 冒烟测试（一张头部图像加 `explore` 工具定义）返回了合法的工具调用 `explore(direction=left, distance=0.3)`，`finish_reason` 为 `tool_calls`，`completion_tokens=28`，`parse_error` 为 null。这一版作为 `agent_assist` 和 `agent_full` 两种编排器的决策模型。

### v2 · 2026-11-06 · 共享 GPU 上启动失败和启动太慢：固定 KV 大小、eager、关闭视频输入

**当时的版本。** v0 的启动脚本通过排队工具租一张卡，按显存比例自动分配 KV。演示流程需要把 vLLM 放在一张和其他任务共用的 GPU 上（经决定这张卡不通过排队工具租用，但不能影响卡上的其他任务），于是先加了一个最简单的共享卡模式：设置 `VLLM_GPU` 时直接在这张卡上启动，用必填的 `VLLM_GMU`（0.32）作为 `--gpu-memory-utilization` 限制显存比例，KV cache 仍由 vLLM 自动分配。

**问题。** 演示流程卡在 vLLM 启动这一步，接连出现三个问题（`runs/vllm_m5.log`）：

1. vLLM 启动失败，报 `AssertionError`，`Engine core initialization failed`。
2. 第一次修复后又失败，报 `ValueError: Free memory on device (61.7/… GiB) on startup is less than desired GPU memory utilization (0.9, … GiB)`。
3. 启动本身很慢，开发记录中记为约 13 分钟。

**根因。**

1. vLLM 默认在一次 dummy 前向的前后各测一次空闲显存来决定 KV cache 大小，并断言这段时间里其他进程没有释放显存。共享卡保证不了这一点：卡上的另一个任务在 profiling 期间释放了 15 GiB（看起来是重启了）。
2. 第一次修复改用固定 KV 大小时，启动命令里漏掉了 `--gpu-memory-utilization`，vLLM 用了默认值 0.9。固定 KV 大小后启动时仍会检查这个比例，要求空闲显存不少于"比例 × 总显存"；共享卡上的空闲显存只有 61.7 GiB。
3. vLLM 启动时按最大尺寸的视频输入做显存 profiling，而本项目从不发视频；卡上另有任务时这一步很慢。

**改法。** 修改 `start_vllm.sh` 的共享卡模式（设置 `VLLM_GPU` 时进入）：

- 必须设置 `VLLM_KV_GIB`，作为 `--kv-cache-memory-bytes` 传入（当时设为 8 GiB），跳过 KV profiling；
- `VLLM_GMU` 仍必须设置，并重新作为 `--gpu-memory-utilization` 传入（当时设为 0.32），既用来通过启动检查，也作为本进程在这张卡上的显存上限；
- 加 `--enforce-eager`，跳过 torch.compile 和 CUDA graph 捕获；
- 所有模式的 `--limit-mm-per-prompt` 都改为 `{"image": 8, "video": 0}`。

**效果。**

| 指标 | 改前 | 改后 |
|---|---|---|
| 启动结果 | `AssertionError` / `ValueError` | 正常启动 |
| 启动耗时 | 约 13 分钟 | 45 s |
| Encoder cache 预算 | 153600 token | 16384 token |
| KV cache 容量 | – | 58,240 token |
| vLLM 显存占用 | – | 约 33.8 GB |

和卡上另一个任务（33.4 GB，未受影响）以及仿真实例共存时，卡上还剩约 8 GB 空闲。`--enforce-eager` 关掉了 CUDA graph，推测会让单 token 解码变慢一些，但没有做过对比测量。

### v3 · 2026-11-06 · 记录模型"想了什么、什么时候想的"

**当时的版本。** `chat()` 只返回回复文本、工具调用、`finish_reason` 和 `usage`，被截断的尝试只计数。

**问题。** 我们要为每个 episode 生成时间线和带叠加层的视频，需要知道 agent 在什么时刻做了哪个决策、理由是什么、花了多久。原来的返回值里没有时间，也没有每次尝试的明细；全部尝试都截断而导致 episode 失败时，日志里看不到这几次尝试分别是什么情况。

**改法。**

- `chat()` 额外返回 `reasoning_content`（`getattr(message, "reasoning_content", None)`）、整次调用的 `t_start` / `t_end`（unix 时间），以及 `attempts`：每次尝试的起止时间、`finish_reason`、`usage`；被截断的尝试把内容存在 `truncated_content` 里，只记录，不解析。
- `LLMTruncated` 带上 `attempts`；全部尝试都截断时，`orchestrator_error.txt` 里写入 `llm_attempts`。
- `g1agent/orchestrators/common.py` 的 `log_llm()` 把这些字段一并写进 `llm.jsonl`。
- Qwen3-VL-8B-Instruct 不单独输出思考内容（`reasoning_content` 总是 `None`），所以提示词改为要求模型在每次调用工具前用一句话写出理由，这句话作为 "thought" 进入 `llm.jsonl` 和时间线。需要完整推理过程时可以换成 Qwen3-VL-8B-Thinking（未试过）。

**效果。** `EpisodeLogger.llm()` 能把每次决策的理由、工具调用和耗时推给视频叠加层。

### v4 · 2026-11-07 · 四面箱堆上下文溢出（服务端上限 32768 token）

**当时的版本。** 服务端 `max_model_len` 为 32768；`agent_full` 编排器只在任务 1 阶段结束（第一次 `plan_scans` 成功）时压缩一次对话历史。

**问题。** 第 8 轮四面箱堆（stack4f，共 12 个条码）实验中，正面 3/3、右侧面 3/3 都扫到，走向背面时 vLLM 报上下文溢出（33288 > 32768 token），episode 被中止，停在 6/12。`llm.jsonl` 的 `prompt_tokens`：第一阶段从 3888 涨到 5363，在第一次 `plan_scans` 处压缩回 4314，之后每扫一个码大约增长 4K token。

**根因。** 对话历史在第一次压缩之后随条码数线性增长，码越多一定会溢出。这不是客户端或服务端的问题；客户端按设计不截断输入，溢出就作为错误抛出。

**改法。** 在编排器里修（详见编排器页）：每次 `next_barcode` 成功都压缩一次历史，只保留系统提示、任务说明、第一阶段摘要和这次 `next_barcode` 的完整结果（其中有所有条码的状态）。LLM 层和服务端配置都没有改，`max_model_len` 仍是 32768。

**效果。** 每个码开始时提示长度回落到约 4.8K token，一个码扫完最多约 7.7K token，上下文长度不再随条码数和侧面数增长。

### v5 · 2026-11-07 · 超时 120 s → 900 s，关闭 SDK 自动重发，重抽改用模型自带采样参数

**当时的版本。** v1 的设置：`timeout_s: 120`；`OpenAI(...)` 的 `max_retries` 保持 SDK 默认，超时后 SDK 自动原样重发；截断后的重新生成仍用 `temperature: 0.0`。

**问题。** 两面箱堆实验 af4c_2side_v4 启动后很快就结束了。`result.json` 显示 wall 为 660.2 s，`orchestrator_error` 是 `APITimeoutError: Request timed out`；此时只完成了 13 次工具调用，刚开始处理第一个条码 bc0。

**根因。**

1. 模型进入失控生成，不停重复内容。vLLM 日志显示单请求解码速度约 65 tok/s，生成到 32768 token 的上下文上限大约要 500 s，120 s 的超时在生成自然结束（碰到上限、返回 `length`）之前就切断了请求。
2. 超时后 OpenAI SDK 自动原样重发，SDK 层面一共发了 3 次；温度 0 是贪心解码，重发得到的是同一条失控生成，3 次都超时，异常抛出，整轮中止。
3. 即使客户端自己的截断重抽逻辑生效，温度 0 下重抽也只会逐 token 重复同一条回复。

**改法。**

- `configs/llm.yaml`：`timeout_s` 从 120 改为 900。依据是上面约 500 s 的估算；开发记录里给的留余量理由是可能有两路请求并发，余量大小为手定。
- `g1agent/llm/client.py`：构造 `OpenAI(..., max_retries=0)`，关闭 SDK 自动重发；`chat()` 捕获 `APITimeoutError`，记一条 `finish_reason: "timeout"` 的尝试，计入 `stats["timeouts"]`，然后和截断一样进入重抽。
- 新增 `redraw_sampling: {temperature: 0.7, top_p: 0.8, top_k: 20}`，取自 Qwen3-VL-8B-Instruct 的 `generation_config.json`；第 2 次及以后的尝试用这组参数，`top_k` 走 `extra_body`。第一次尝试仍是温度 0。
- `max_retries_on_length` 仍为 2，仍然不设任何输出长度上限。
- `tests/test_llm_client.py` 的 mock 服务器增加"先睡再回复"的能力，模拟模型还在生成时客户端超时；同时记录每个请求，用来检查重抽时的采样参数。测试 5 个全部通过。

**效果。** af4c_2side_v4 的记录归档，v5 在 19:41 重新启动。之后的两面箱堆运行 af4c_2side_v7（一个 episode，6 个条码全部扫到）中，LLM 统计为：

| 指标 | 值 |
|---|---|
| `chat()` 调用次数 | 459 |
| 实际请求次数（含重抽） | 460 |
| 长度截断 | 1 次，即 1/460（0.2%），已自动重抽 |
| 超时 | 0 |
| prompt token 合计 | 4,992,683 |
| completion token 合计 | 62,152 |
| LLM 请求总耗时 | 1007.9 s（该 episode wall 6778.2 s） |

### v6 · 2026-11-08 · 抽成独立仓库时把本机相关设置移出启动脚本

**当时的版本。** v2 之后的 `start_vllm.sh` 里写死了本机的数据盘目录（conda 环境、临时目录、vLLM 缓存目录都在它下面），`HF_HOME` 从 `system.yaml` 读，默认模式直接调用本机的 GPU 排队工具。

**问题。** 把 agent 系统整理成独立仓库后，这些路径和工具在别的机器上不存在，脚本无法直接使用。

**改法。**

- 脚本开头 `source` 仓库根目录下不进版本库的 `local.env`（如果有）；本机路径和环境变量都放在那里。
- conda 环境名改从 `system.yaml` 的 `vlm_server.conda_env` 读，环境所在目录由 `G1_CONDA_PREFIX` 给出；`HF_HOME`、`TMPDIR`、`VLLM_CACHE_ROOT` 改由环境变量给出；`HF_HUB_OFFLINE` 默认仍为 1，但可以被覆盖。
- GPU 排队工具改为可选的钩子：`local.env` 定义了 `gpu_launch` 函数就用它包住启动命令，否则直接 `exec vllm`。共享卡模式的逻辑不变。

**效果。** 启动参数和 v2 相同，只是本机相关的部分移到了仓库外。

## 已知限制

- **温度 0 的失控生成仍然会发生**，只是能自动恢复。af4c_2side_v7 中 460 次请求有 1 次截断；每次失控在被识别前要等生成到上下文上限（约 500 s）或者 900 s 超时，这段时间里 episode 停着不动。
- **重抽只有 2 次。** 3 次尝试都截断或超时，episode 就以 `LLMTruncated` 失败。`max_retries_on_length: 2` 没有数据依据。
- **工具参数解析失败不重抽。** 参数不是合法 JSON 时只把 `parse_error` 交给编排器，`agent_full` 把错误回给模型，要多花一轮对话。
- **没有单独的思考输出。** Instruct 模型的 `reasoning_content` 总是 `None`，日志里的决策理由只是模型按提示词写的一句话，不一定反映真实的决策过程。
- **上下文上限 32768 token 是硬约束。** 客户端不截断输入，超出就报错；能否不超出完全依赖编排器的历史压缩（v4）。
- **共享卡模式的 KV 大小和显存比例是手工给的**（v2 时为 8 GiB 和 0.32），按当时卡上的空闲显存选的，换卡或者卡上任务变了需要重新选。`--enforce-eager` 对解码速度的影响没有测过。
- **若干数值没有依据**：`temperature: 0.0`、`image_max_side: 640`、JPEG 质量 90、单 prompt 最多 8 张图、默认显存比例 0.85。
- **冒烟测试不检查参数取值**，它手写的 `explore` 工具定义也和 `registry.py` 里的不完全一样（缺 `forward`），只能证明链路通，不能证明模型在真实工具集上的表现。
- **其他错误不重试。** 连接失败、HTTP 5xx 等异常直接抛给 `run_task.py`，该 episode 记为失败。
