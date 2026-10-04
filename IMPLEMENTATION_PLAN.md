# π₀.₅ × RoboCasa365：实施计划

## 固定输入与边界

- 上游源码只读使用：`sources/openpi` @ `215abfb217dbac7d5f1273282331b9b1866c0479`；`sources/robocasa` @ `456174f62b89b8fca99eaaf33949c29fec9cfc2a`；官方基线 `sources/robocasa-openpi` @ `ca4c6d710db75e276bc7c866a57bd7e4aee5b6e8`。对照源码 `sources/robocasa-v1.0` 来自 v1.0 标签 @ `8f3c96ec8d1bfcd8126cad2bca887da98d30e997`。`sources/` 已忽略；不在上游目录写项目适配代码。
- 所有新增文件留在 `/share/zhuweiwen-local/projects/pi05-robocasa/`：适配代码放 `src/`，测试放 `tests/`，两套独立 Python 环境与下载缓存放 `cache/`，数据放 `datasets/`，权重放 `checkpoints/`，运行记录放 `logs/` 或 `outputs/`。这些运行产物均应保持忽略状态。
- 已在忽略的 `cache/` 下安装 uv 0.12.19 和 CPython 3.11.16，并建立两个空虚拟环境；未初始化 OpenPI 子模块、安装项目依赖、下载场景资产或权重、启动仿真或训练。

## 技术路线

1. **环境隔离**：项目内 CPython 3.11.16 已备好。原版 OpenPI 锁文件的 NumPy 1.26.4/MuJoCo 2.3.7/LeRobot 0.1.0 与 RoboCasa 的 NumPy 2.2.5/MuJoCo 3.3.1/LeRobot 0.3.3 冲突；官方 fork 的 `pyproject.toml` 已改为 LeRobot 0.3.3 且未限制 NumPy 上界，但推理端仍需单独核实 JAX/CUDA 依赖。保持推理端与仿真端两个进程，通过官方 WebSocket 客户端连接；服务端只能绑定回环地址和经检查的空闲端口，不能直接使用上游 `serve_policy.py` 的 `0.0.0.0` 绑定。
2. **官方 π₀.₅ 基线**：以 `sources/robocasa-openpi` 的 `pi05_pretrain_human300` 配置、`RobocasaInputs/RobocasaOutputs`、`examples/robocasa/main.py` 和 `get_eval_stats.py` 为起点；无需从头写三相机/状态/12 维动作映射和成功率统计。官方提交登记的 RoboCasa 版本为 1.0.0，checkpoint 为 `robocasa/robocasa365_checkpoints/pi05_pretrain_human300/multitask_learning/75000`，尚未下载。先针对 `pretrain` split 复核多任务基线；`target` split 另作迁移评测，不混入原排行榜数值。
3. **本项目需要的小范围兼容代码**：上游 `main.py` 的 `Args` 声明 `task_set`，实际访问不存在的 `task_soup`；需在项目本地入口处理多个 task set。上游用 `get_task_horizon()*1.5`；1.0.1 注册表的 317 个任务已全部把 1.0.0 horizon 乘 1.5，故 1.0.1 入口不能再乘。为保持仿真端精简，本地入口可去掉上游未使用的 `FileUtils/ObsUtils/EnvUtils` 导入（前两者在模块加载时引入 PyTorch）；还需限制日志路径、检查动作形状/有限值、固定种子并记录版本。不要修改上游文件。
4. **验证顺序**：静态接口与依赖核查 → 仿真端导入和版本检查 → 无权重单场景冒烟检查 → 经批准后取得官方权重与资产并短时验证推理 → 单任务少量诊断回合 → 相同 split、horizon 与每任务 50 回合的正式评测 → 基线成立后再讨论小样本适配。原版 `main.py` 在 1.0.1 下不能直接执行；少量诊断回合不能当正式基准。

## 资源预算与停止条件

- 项目所在分区在本次检查时可用约 4.1 TB。新增官方 OpenPI fork 源码目录约 2.3 MB；RoboCasa v1.0 对照源码约 44 MB。RoboCasa 厨房资产文档估计约 **10 GB**，下载前需确认目标和实际清单。π₀.₅ 权重及数据集大小尚未核实；单项超过 3 GB 先报告用途、大小和预计时间并等待决定。依赖环境/缓存暂按数 GB 至数十 GB 预留，安装前以下载清单核实，不把估算当实际占用。
- OpenPI README 估计推理需 >8 GB 显存、LoRA 需 >22.5 GB，24 GB RTX 3090 对 LoRA 余量很小。默认最多使用一张空闲 GPU，仿真渲染还可能占显存；每次 GPU 任务前检查占用。沙箱外只读查询确认 4× RTX 3090、驱动 535.179，当时均显示 0 MiB 已用。OpenPI 锁文件在 x86_64 上带 CUDA 12.6 用户态包；NVIDIA 的 CUDA 12.x 次版本兼容表列出驱动下限 525，但功能有限制。GPU 框架能否实际运行仍需短时探针验证。
- 遇到 GPU 不可见、驱动/轮子不兼容、依赖解析失败、robosuite 接口不匹配、动作坐标/尺度不能验证、资源下载超过约定阈值、无空闲 GPU 或任务预计运行过长时停在相应关口并报告。不要占用现有端口；确需推理服务时先检查监听状态，仅绑定回环地址并选空闲端口，绝不使用 33393。任何训练开始前另行明确预算和时长。

## 环境核查进展与待选路线

- `cache/tools/` 中 uv 官方归档为 19.8 MB，SHA-256 已核对；`cache/python/` 中 CPython 3.11.16 约 93 MB；`cache/venvs/openpi` 与 `cache/venvs/robocasa` 目前均为空环境。项目分区可用约 4.1 TB。
- 完整 RoboCasa 依赖加 PyPI `robosuite==1.5.2` 解析**失败**：该 wheel 强制 `mink==0.0.5`，后者要求 `numpy<2`，与 RoboCasa 的 `numpy==2.2.5` 不可共存。RoboCasa 文档要求 robosuite master；核对的 master 提交 `5ce6643f3092639d08f7b0f90ed1c6a84f50552c` 将 mink 改为可选 extra，基础依赖允许 NumPy 2。该提交尚未在本地下载，运行兼容性未验证。
- 排除上述必装 mink 后，完整 RoboCasa 声明依赖的 `uv --dry-run` 可解析为 132 个包，其中 LeRobot/Tianshou 连带 PyTorch 2.7.1 与 CUDA 12.6 包；这会使仿真环境体积达数 GB。现阶段不安装完整或精简依赖；官方评测入口所需的更小清单见下节，安装前仍须确认目标路径、总下载量和磁盘余量。

## 官方评测入口的精简依赖核查（未安装）

- 对 `examples/robocasa/main.py`、`openpi_client` 和 RoboCasa 1.0.1 做静态顶层 import 递归，覆盖 444 个本地模块（含可选导入分支）：原样入口需要 NumPy、Gymnasium、MuJoCo/robosuite、OpenCV、Pillow、ImageIO、HDF5、lxml、SciPy、Tyro、msgpack、websockets、PyTorch 等；没有 LeRobot、Tianshou 或 JAX 顶层导入。`mimicgen` 位于 `try/except ImportError` 中。上游未使用的 robomimic 工具导入会带入 PyTorch；仿真端不需要 CUDA 运算，若保留原样入口应选 PyTorch CPU wheel。
- 精简清单的直接要求：`numpy==2.2.5 numba==0.61.2 scipy==1.15.3 mujoco==3.3.1 qpsolvers[quadprog]>=4.3.1 Pillow opencv-python pynput termcolor pytest tqdm pyyaml imageio h5py lxml gymnasium==0.29.1 tyro msgpack websockets`，另需固定 robosuite 源码提交 `5ce6643f3092639d08f7b0f90ed1c6a84f50552c` 和从当前 `sources/robocasa`、`sources/robocasa-openpi/packages/openpi-client/src` 引入代码。此清单不包括训练和数据转换包。
- `uv pip install --dry-run` 在空的 Python 3.11 仿真环境上解析上述清单为 **39 包**；加 `torch==2.7.1+cpu` 与 PyTorch CPU 索引后为 **46 包**，无 CUDA 包。两次 dry-run 均未纳入 robosuite 源码包本身，只包含其固定提交声明的直接基础依赖；未验证实际导入与运行。按当前指令，以上均不安装。

## 里程碑二：项目内 RoboCasa 1.0.1 评测入口

- 官方基线在 `sources/robocasa-openpi@ca4c6d710db75e276bc7c866a57bd7e4aee5b6e8` 已有 π₀.₅ 配置、三相机与状态变换、WebSocket 客户端、12 维动作转换和 RoboCasa 评测入口；本项目贡献是 `src/robocasa_eval/` 中的 1.0.1 兼容评测入口与后续复现实验，**不是从头编写策略适配层**。项目入口沿用该提交 `examples/robocasa/main.py` 的 Apache-2.0 许可并注明来源。
- `--task-set` 对应 `TASK_SET_REGISTRY`，支持多个集合并按首次出现顺序去重。RoboCasa 1.0.1 注册表的 317 个任务 horizon 已相对 1.0.0 扩为 1.5 倍；入口直接使用 `get_task_horizon` 的整数值，运行时拒绝非 1.0.1。官方排行榜 checkpoint `robocasa/robocasa365_checkpoints/pi05_pretrain_human300/multitask_learning/75000` 的原始结果来自 RoboCasa 1.0.0；1.0.1 的结果独立报告，不直接与原榜单数值混算。
- 保留官方三相机键、状态拼接顺序、每 `replan_steps` 步重新规划、`convert_action`、`info["success"]`、20 fps 回放视频、`stats.json` 的 `num_episodes`/`success_rate` 及 `evals_1.5/split/task/run` 结构。`stats.json` 另记录两套协议版本与 checkpoint。默认客户端连接 `127.0.0.1:8000`；正式运行前检查服务器端口与 GPU。评测输出默认写入项目 `outputs/`。
- 纯函数测试在 `tests/test_protocol.py`，不加载 RoboCasa、MuJoCo、PyTorch 或 GPU。当前仅做语法、单元和静态导入检查；没有安装依赖或启动评测。真实运行前仍需验证 robosuite 与 1.0.1 导入、资产路径、图像方向、动作尺度及单场景冒烟。

### 里程碑二时的精简环境草案（已由下节实际结果取代）

以下为安装前的历史方案；实际安装清单及 OpenCV 修复见里程碑三。

- 目标为现有项目私有 `cache/venvs/robocasa`（Python 3.11.16）；uv 缓存为 `cache/uv`。直接安装清单拟为 `numpy==2.2.5 numba==0.61.2 scipy==1.15.3 mujoco==3.3.1 "qpsolvers[quadprog]>=4.3.1" Pillow opencv-python pynput termcolor pytest tqdm pyyaml imageio imageio-ffmpeg h5py lxml gymnasium==0.29.1 msgpack websockets`，外加固定 robosuite 源码提交 `5ce6643f3092639d08f7b0f90ed1c6a84f50552c`；设置 `PYTHONPATH` 指向 `src`、`sources/robocasa`、`sources/robocasa-openpi/packages/openpi-client/src`。本机当前 `command -v ffmpeg` 无结果，故视频输出需要 `imageio-ffmpeg`。项目入口已使用标准库 argparse，故前节原样上游脚本的 `tyro` 不再是直接要求；不安装 torch、LeRobot、Tianshou、JAX 或 CUDA 包。
- 安装前需先获取该固定 robosuite 源码，并重新 dry-run 包含其源码声明的完整解析，核对实际 wheel 大小和磁盘余量。当前 **预计新增下载约 0.3–0.8 GB**（Python 和 uv 已存在，排除资产、checkpoint、数据集）；这是预算区间，不是实测。运行前再确认 RoboCasa/robosuite 对未列出的可选依赖是否有实际导入需求。资源达到既定阈值则停下讨论。

## 里程碑三：精简仿真环境与导入验证

- Python 固定为项目内 3.11.16。`sources/robosuite` 固定到 `5ce6643f3092639d08f7b0f90ed1c6a84f50552c`（1.5.2），`sources/robocasa` 为 `456174f62b89b8fca99eaaf33949c29fec9cfc2a`（1.0.1），官方 OpenPI fork 为 `ca4c6d710db75e276bc7c866a57bd7e4aee5b6e8`。直接依赖在 `requirements-sim.txt`；Python 3.11 的已解析版本在无机器绝对路径的 `requirements-sim.lock`；本项目 editable 打包配置在 `pyproject.toml`。四个源码包均以 editable、`--no-deps` 安装；不修改上游源码。
- 首次完整 dry-run（含 robosuite 本地源码）解析 38 包，37 个第三方下载归档的保守大小合计约 237 MB；robosuite 固定提交源码归档为 260,653,994 字节。首次导入遇到容器缺少 `libGL.so.1`，沿 `robosuite → cv2` 触发；项目环境中的带 GUI `opencv-python` 已替换为 `opencv-python-headless==5.0.0.93`。补入 `dm-tree`、`tree`、`pygame`、`hidapi` 和其轻量传递依赖，第二次 dry-run 的增量为 10 包、约 78 MB 下载归档。总下载预算约 0.58 GB，低于 0.8 GB；没有安装 torch、jax、lerobot、tianshou、nvidia-* 或 CUDA 包。
- 最终环境包含 50 个发行包，虚拟环境占用约 695 MB；NumPy 2.2.5、MuJoCo 3.3.1、robosuite 1.5.2、RoboCasa 1.0.1、Gymnasium 0.29.1、openpi-client 0.1.0 和本项目包 0.1.0 均成功导入。7 个纯函数测试和评测入口 `--help` 通过。仅完成导入验证，没有实例化环境、渲染、下载场景资产或访问 GPU。
- `uv pip check` 检查 50 包后仍报告 **4 项元数据缺失**：robosuite/RoboCasa 声明 `opencv-python`，实际使用可导入的 headless 版本；RoboCasa 声明 `tianshou==0.4.10` 和 `lerobot==0.3.3`，本精简仿真环境按约定不安装。不能声称完整上游元数据一致性通过，也不能为了消除警告装入 ML/CUDA 依赖。依“所有验证通过后才提交”条件，暂不创建本地提交。
- 下一阶段厨房资产的上游入口见 `sources/robocasa/robocasa/scripts/download_kitchen_assets.py` 与 `sources/robocasa/robocasa/models/assets/box_links/box_links_assets.json`：六类资源来自 UT Austin Box，官方 README 预计总计约 10 GB。建议先下载到本项目 `datasets/robocasa-kitchen-assets/`，逐项以 HEAD 核实 `Content-Length`、磁盘空间和单项 3 GB 门槛，再校验归档 SHA-256（本地记录）、ZIP CRC 与解压路径。上游脚本把目标硬编码在 `sources/robocasa/robocasa/models/assets/`，与上游只读边界冲突；在下载和实例化环境前，需要先确定项目自有的资产路径适配方案，不能直接运行该脚本。

## 里程碑四准备：环境验证与项目资产根

- 可复现验证入口为 `python -m robocasa_eval.verify_environment`：仅导入 NumPy、MuJoCo、cv2、robosuite、RoboCasa、Gymnasium、openpi-client、本项目包，逐个核对版本；从发行包元数据检查禁止包及所有活跃依赖，仅接受 `(robosuite, opencv-python)`、`(robocasa, opencv-python)`、`(robocasa, tianshou==0.4.10)`、`(robocasa, lerobot==0.3.3)` 四项**原文完全一致**的缺项。第五项、四项中任何一项变化或消失、安装 GUI OpenCV/ML/CUDA 包都会失败。这四项可保留，是因为当前仿真入口和 WebSocket 客户端导入已通过，headless `cv2` 提供所需 Python API；LeRobot/Tianshou 未处于该入口的导入闭包。真实场景与渲染仍未验证，不能从导入成功推出评测可运行。
- 源码调用链：`robocasa/models/__init__.py` 在导入时以包内 `models/assets` 设置 `assets_root`；场景 YAML、fixture XML/纹理通过 `robocasa.models.assets_root` 传入 `xml_path_completion`，`kitchen_objects.py`、`kitchen_object_utils.py` 又在导入时缓存对象目录并生成对象目录列表。`download_kitchen_assets.py` 则把六类包的解压目录硬编码到 `robocasa.__path__[0]/models/assets`。`macros.py` 的 `DATASET_BASE_PATH` 仅被 `dataset_registry_utils.py` / `download_datasets.py` 用于示教数据；未找到厨房资产的官方环境变量、macro 或统一参数。`macros_private.py` 不存在。
- 项目方案：固定根目录 `datasets/robocasa-kitchen-assets/`。在真正取资产时，先把上游已有约 3.4 MB 的基础 `models/assets` 内容复制到该目录，再将选定的官方归档按其预期子目录安全解压到同一根。`src/robocasa_eval/assets.py` 在 RoboCasa 首次导入**之前**安装仅针对 `robocasa.models` 的加载钩子，将 `assets_root` 指向项目根；`main.py` 在任何仿真包导入前调用它，让对象目录缓存使用项目路径。钩子要求基础文件存在且目标不经软链接；不会修改上游工作树或虚拟环境。此时尚未创建该资产根、复制基础文件或实例化环境；重建虚拟环境后只需 editable 安装本项目即可复现代码路径。
- 六类上游资源与预期目录：`tex → textures`、`tex_generative → generative_textures`、`fixtures_lw → fixtures`、`objs_objaverse → objects/objaverse`、`objs_aigen → objects/aigen_objs`、`objs_lw → objects/lightwheel`。具体公开 Box URL 由上游 `box_links_assets.json` 与项目 `scripts/asset_inventory.py` 只读解析。发现 Python `urllib` 默认可能把重定向的 HEAD 改成 GET 后，清单脚本已禁用自动重定向；修复后重新发送的六个纯 HEAD 直链请求均返回 404，`Content-Length` 当时全为**未知**，没有正文下载。这不足以断定资源失效。首次查询没有保留完整重定向记录，不能追溯证明其未转换方法；本轮按新授权进行了下述受限 Range GET。上游 downloader 提供 `--type` 分项选项，但 ZIP 内部路径仍需下载前/下载后单独核对。
- 最小单任务 smoke test 的修正候选为 `tex + tex_generative + fixtures_lw + objs_lw`，另加上游包内小型基础文件。源码依据：`Kitchen.__init__` 接受 `obj_registries=("lightwheel",)` 与 `generative_textures="100p"`；`create_env` 将额外参数传到 `robosuite.make`；`OpenDrawer` 从 `tool/utensil` 组取对象，而 `kitchen_objects.py` 的 Lightwheel 注册表含这两类。项目资产钩子同时覆盖 `robocasa.models.assets_root` 与 `robocasa.utils.texture_swap.TEXTURES_DIR`，后者原本将生成纹理写死为上游包路径。此四类是**源代码层面的候选**，并非已经通过实例化的充分条件：目前评测入口未暴露 `obj_registries`、`generative_textures`，若做该 smoke test 需先添加专用参数或单独的短时检查入口，且需固定任务、layout/style 并核对实际引用。默认纹理模式不启用 `tex_generative`，则前三类中的生成纹理可省，候选为 `tex + fixtures_lw + objs_lw`。完整 π₀.₅ benchmark 使用默认跨任务对象库，若不逐任务静态证明可裁剪，应准备六类资产，不能用单任务 smoke 子集替代。
- `scripts/asset_inventory.py --range-probe` 对上游六个 Box 直链逐项发送 `Range: bytes=0-0` 的 HTTPS GET，HTTPS 重定向保留 Range，每个最终响应最多调用一次 `read(1)` 并立即关闭；连接/读取均有 15 秒超时，不创建文件。最终 Box CDN 路径可能含临时访问令牌，清单输出只保留主机与遮盖后的路径。此前六个 HEAD 404 不能判定失效；此次六个受限请求均返回 `206 application/zip`、`Content-Range: bytes 0-0/总字节`，每项应用实际读取 1 字节。归档字节数：`tex` 543,180,421；`tex_generative` 1,184,675,536；`fixtures_lw` 529,745,465；`objs_objaverse` 2,163,884,721；`objs_aigen` 5,773,223,039；`objs_lw` 791,939,656。四类 smoke 候选合计 3,049,541,078 字节；默认纹理的三类候选合计 1,864,865,542 字节；六类合计 10,986,648,838 字节，均仅是压缩归档大小，解压后占用仍未知。`objs_aigen` 单项超过 3 GB，下载前须报告用途、来源、目标路径、磁盘余量和运行时间并取得决定。下载后校验本地 SHA-256、ZIP CRC、归档路径、预期目录及上游工作树状态；本阶段不下载、不解压、不实例化仿真。
