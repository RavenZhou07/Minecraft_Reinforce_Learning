# Minecraft 资源搜索基线（MineRL 0.4.4）

简体中文 | [English](README.md)

这是一个刻意保持小型、显式且易调试的 Minecraft 研究项目，用于实现资源搜索、
候选记忆、失败恢复和重新规划。项目基于旧版 MineRL/Malmo，优先采用 wrapper、
独立模块和可检查状态机，而不是大型端到端强化学习框架。

当前 curriculum 要求 agent 在多棵外观相同的树中找到“初始最近树”。策略会完成
整圈扫描、建立 object-centric candidate map、选择和接近目标，并在视觉进展
停滞时执行局部重捕获、恢复或重规划。

> 本仓库没有 DQN、PPO、LLM agent，也不宣称具备通用 Minecraft 能力。只有在
> “故意首次选错后仍能稳定恢复”的 gate 通过后，才进入自然生成的
> `MineRLTreechop-v0` 测试。

## 当前状态

已验证环境：Windows x64、Python 3.8.20、MineRL 0.4.4、Gym 0.19.0、
OpenJDK 8、Minecraft 1.11.2/Malmo。

| 实验 | 结果 | 平均步数 |
|---|---:|---:|
| 单树、±180°、距离 3–10 | Oracle 20/20；Visual 20/20 | 38.5 / 45.5 |
| 一个目标 + 两棵相同干扰树，旧视觉基线 | 16/20 | 107.5 |
| 显式 candidate search，seeds 11000–11019 | 20/20 | 75.2 |
| 显式 candidate search，未见 seeds 12000–12029 | 30/30 | 72.5 |
| 强制首次选错，seeds 13000–13002 | 最佳完整运行 2/3 | — |

普通 20/30-seed 实验没有 300 步失败，但 50 局首次都正确选择了最近候选，因此
不能证明失败重规划真正有效。严格诊断中的 seed 13000 仍然失败：agent 走到
干扰树后，只保存 bearing 的记忆无法在平移后更新远处目标方向。

当前开发中的可选 `f3_telemetry` profile 只暴露 agent 自身坐标、yaw/pitch 和
群系元数据，再用 POV 估计的距离得到 candidate 粗略世界坐标。它不暴露 log
grid、目标坐标、最近树标签或 oracle distance，结果必须与 `pov_only` 分开报告。
在完整 Minecraft evaluation 落盘前，不应把它视为已验证结果。

完整结果、置信区间、gate 和失败分析见
[`docs/find_tree_curriculum.md`](docs/find_tree_curriculum.md)。

## 设计

```text
POV（+ 可选 F3 自身 telemetry）
                │
                ▼
       resource detector/adapter
                │
                ▼
SCAN → candidate map → SELECT → ALIGN → APPROACH
                                      │
                                      ▼
                         progress / stall monitor
                                      │
                                      ▼
                    LOCAL_REACQUIRE → RECOVER → REPLAN
```

部署评分保持简单，并逐项写入日志：

```text
confidence
+ 2 * log(apparent_size + 1e-3)
- 0.15 * abs(turn) / 180
- approach_attempts
- 0.002 * age
```

树木检测使用可解释的 RGB 连通区域，并按方向和视觉尺度合并检测。在 telemetry
模式中，candidate 还保存估计世界坐标和不确定度，因此 agent 平移后可以重新
计算旧候选方向。

## 项目结构

```text
mc_rl/
  actions.py             离散动作映射
  candidates.py          candidate 表示、合并、评分、cooldown
  envs.py                环境构造和隔离运行时配置
  find_tree_env.py       自定义 curriculum 与评估 oracle
  navigation.py          几何计算和 oracle 验证控制器
  progress.py            视觉 stalled / lost 诊断
  resource_adapters.py   通用资源接口和 Tree adapter
  search_policy.py       扫描、接近、恢复、重规划状态机
  telemetry.py           可选 F3 状态与视觉到世界坐标几何
  vision.py              小型视觉基线
  wrappers.py            Gym wrappers 和 one-log 终止
scripts/                  安装、采集、训练、评估和 demo
tests/                    单元测试及可选 Minecraft 集成测试
docs/                     curriculum 和失败分析
```

## 安装

不要升级 MineRL、Gym 或旧版依赖，也不要安装到全局 Python/Java 环境。

```powershell
conda env create --prefix ..\.conda-env -f environment.yml
conda activate ..\.conda-env
python scripts/install_minerl.py
python -m pip install -r requirements.txt
python scripts/check_install.py
python -m pip check
```

顺序不能颠倒。`install_minerl.py` 只修复 MineRL 0.4.4 历史 source distribution
失效的构建仓库引用。pip 固定在 24.1 以下，因为新版会拒绝 Gym 0.19.0 的历史
metadata。启动前确认 Python 为 3.8、Java 为 1.8。

## 测试

快速测试不启动 Minecraft：

```powershell
..\.conda-env\python.exe -m pytest -m "not integration" -q
```

真实集成测试需显式开启，并顺序运行：

```powershell
$env:RUN_MINERL_INTEGRATION = "1"
..\.conda-env\python.exe -m pytest -m integration -q
```

Minecraft 冷启动可能需要 4–6 分钟。控制台暂时没有输出不代表卡死，不要并行
启动多个实例。

## Candidate search smoke test

POV-only：

```powershell
..\.conda-env\python.exe -m scripts.evaluate_candidate_search `
  --episodes 3 --seed 10000 --max-steps 300 `
  --yaw-noise 180 --distance-min 3 --distance-max 10 `
  --distractor-trees 2 --modes candidate --sensor-profile pov_only `
  --output logs/find_tree/my_candidate_smoke.csv
```

使用自身 telemetry 的强制选错诊断：

```powershell
..\.conda-env\python.exe -m scripts.evaluate_candidate_search `
  --episodes 3 --seed 13000 --max-steps 300 `
  --yaw-noise 180 --distance-min 3 --distance-max 10 `
  --distractor-trees 2 --modes candidate --force-initial-rank 1 `
  --sensor-profile f3_telemetry `
  --output logs/find_tree/my_f3_recovery_smoke.csv
```

评估脚本默认拒绝覆盖已有输出。比较策略时不要替换失败 seed 或降低成功标准。

## Sensor 与 oracle 边界

- `pov_only`：POV、已执行 camera delta 和内部记忆；
- `f3_telemetry`：增加 agent 自身 `x/y/z`、yaw、pitch 和群系，类似玩家可见的 F3。

两者都禁止读取生成的 log grid、目标坐标、oracle distance 或正确候选标签。
Oracle 只能用于评估，两类 profile 的结果必须分别汇报。

## Git 中保留的产物

Git 只保留源码、文档、当前最佳的小型 checkpoint 和少量稳定 JSON 摘要。原始
数据集、POV 帧、完整 trace、Minecraft 日志、watcher 文件、临时世界和本地
Conda 环境默认忽略。

当前最佳 checkpoint：

```text
checkpoints/find_tree_visual_distance3_10_stack4.npz
```

## 兼容性说明

- MineRL 0.4.4 必须使用 JDK 8；全局 JDK 21 可能因缺少 JAXB 而失败。
- Windows 关闭时可能打印 `Failed to delete the temporary minecraft directory`
  或 `process already exited`。应保留 warning 并确认没有残留 Java 进程。
- 项目使用旧 Gym API：`reset() -> observation`，
  `step() -> observation, reward, done, info`。

## 许可证

目前尚未选择开源许可证。公开发布前应明确添加 LICENSE；在此之前默认版权限制
仍然适用。

