# MCRL Runtime Integrity, Collapse Causality, Observability & Coverage Audit

**Experiment:** `exp12_runtime_observability_audit`  
**Repository HEAD at predeclaration:** `6549d11673cda4a3393d220552bd5461089f53f2`  
**Decision:** `previous-action feedback supported`  
**Promotion:** none  
**New policy training:** none

## 1. Executive conclusion

**当前 500-step single-action fixed point 的主要证据支持 H2：explicit previous-action embedding 造成强烈的 closed-loop positive feedback。**

Runtime/input correctness gate 全部通过；previous-token 在固定 observation 和固定 hidden 下可使 `46.4%` 的 sweep 结果改变 argmax，teacher-forced replay 的 balanced accuracy 为 `32.6%`，free-running 仅 `8.0%`，previous-token 引起的 policy variation 约为连续 observation variation 的 `59.7×`。这满足预声明的 Case B 判定规则。

GRU hidden 对 probability distribution 有明显影响，但 hidden intervention 只在 `9.2%` 的状态改变 argmax，未达到 hidden-attractor 主因门槛。现有 frozen CNN embedding 在关键 `valid_attack_geometry` validation probe 上反而优于 four-frame stack 和旧 hand-written features，因此 learned representation failure、legal observation aliasing 和 coverage/OOD 均不受支持为本轮主因。

本轮唯一下一步是：

> 使用完全相同的数据、CNN、GRU、loss、optimizer 和 training seed 29，只移除 explicit previous-action embedding，训练一个 controlled GRU actor。

不得同时加入 frame stack、新数据、DAgger2、模型扩容或 PPO。

## 2. Scope and integrity boundaries

由于 registry 中 `exp10` 和 `exp11` 已被上一轮占用，本轮使用下一个可用 ID `exp12_runtime_observability_audit`。预声明配置在观察本轮结果前写入：

- `configs/learning/runtime_observability_audit_exp12.json`
- config SHA-256: `372890e3770a51191197abd53cfba733d37fb0cb3b50c0492b075c8e08141814`
- `artifacts/exp12/audit_label_schema.json`

本轮只读取显式列出的：

- `teacher_dev`
- `bc_train`
- `bc_validation`
- existing DAgger1
- newly generated `student_dev` diagnostic traces

`student_holdout` 和 `final_test` 未运行、未加载、未扫描或统计。教师、v9.12、success definition、legal observation boundary、14-action space 和环境规则均未改变。

所有 autonomous traces 满足：

```text
teacher actions executed = 0
privileged actor inputs = 0
```

Traces 被标记为 `diagnostic_only_not_policy_training_data`，没有加入 BC/DAgger 数据集。

## 3. Runtime integrity gate

Stage A 使用 recurrent seed 29 checkpoint 和 environment seed 18500，保留完整 500-step trace。

| Check | Result |
|---|---:|
| selected action = executed action | pass |
| action IDs/names match the shared 14-action mapping | pass |
| previous token is START at reset, then exactly `action[t-1]` | pass |
| one hidden advance per environment decision | pass |
| hidden reset / fresh legal adapter | pass |
| RGB HWC uint8 → CHW float/255 parity | pass |
| legal vector preprocessing parity | pass |
| model eval mode; dropout/batch norm mismatch | pass; neither layer type exists |
| stale observation | not observed |
| live vs standalone replay | exact pass |

Live/standalone replay 对 CNN embedding、scalar embedding、hidden、logits 和 probabilities 的最大绝对误差均为 `0.0`，argmax 每步一致。Train batched sequence 与逐步 recurrent inference 的 logits 最大误差为 `2.86e-6`，低于独立的 `1e-5` 浮点容差。

Seed29/18500 中：

- 499/499 后续步 RGB 改变；共有 82 个不同 RGB hash；
- 499/499 后续步 legal vector 改变；
- CNN embedding 和 hidden 持续变化；
- actor 仍在全部 500 步选择 `fine_turn_left`；
- mean softmax policy entropy 为 `1.762` nats。

因此固定点不是 stale frame、stale legal vector、action mapping、hidden update/reset、checkpoint eval mode 或 preprocessing drift 造成的。

## 4. Cross-checkpoint matched behavior

三个 frozen checkpoints 分别在相同的 `student_dev` seeds 18500–18503 上运行，共 12 个 autonomous episodes。

| Training seed | Dominant actions | Success | Timeout | Episodes with transitions | Valid-attack episodes | Mean policy entropy | Mean CNN Δ | Mean hidden Δ |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 11 | `turn_right` ×4 | 0/4 | 4/4 | 0/4 | 0/4 | 1.112 | 1.180 | 0.469 |
| 29 | `fine_turn_left` ×1, `turn_right` ×3 | 0/4 | 4/4 | 0/4 | 0/4 | 1.167 | 1.705 | 0.737 |
| 47 | `attack` ×1, `turn_right` ×3 | 0/4 | 4/4 | 0/4 | 1/4 | 0.998 | 0.742 | 0.333 |

Aggregate：

- inventory success: `0/12`；
- timeout: `12/12`；
- pure 500-step single-action episodes: `12/12`；
- total action transitions: `0`；
- block break / pickup: `0/12`；
- 每个 trace 均通过独立 standalone replay parity。

Seed47/18500 的 `attack` fixed point 在部分 timestep 满足 audit valid-attack geometry，但仍没有 break 或 pickup。固定点机制跨 initialization 稳定存在，但吸引到的具体动作受 checkpoint 和初始环境状态影响，并非三个模型总是复制同一个动作。

## 5. Collapse causality

### 5.1 Previous-action token sweep

在 268 个 stratified teacher/student states 上固定 RGB、legal vector 和 GRU hidden，只遍历 14 action tokens + START：

| Metric | Result |
|---|---:|
| argmax changed by previous token | 46.37% |
| argmax equals swept previous token | 24.83% |
| mean pairwise JS across tokens | 0.03511 |
| mean logit variance due to token | 0.10375 |
| token JS / consecutive-observation JS | 59.70× |

这证明 explicit previous token 本身是强因果输入，而不只是与 rollout state 相关。

### 5.2 Hidden intervention

对相同 observation/token 比较 normal、zero、earlier-same-episode 和 foreign diagnostic hidden：

| Metric | Result |
|---|---:|
| mean hidden-intervention JS | 0.04265 |
| hidden JS / observation JS | 72.53× |
| hidden intervention argmax-change rate | 9.20% |

Hidden 会重分配 probability mass，但大部分 intervention 没有改变最终 action。它是可能的放大/记忆通道，但没有满足预声明的 hidden-attractor 主因条件 `argmax-change >= 25%`。

### 5.3 Teacher-forced vs free-running replay

固定使用 recorded `bc_validation` observations：

| Replay mode | Mean balanced accuracy |
|---|---:|
| teacher-forced previous actions | 32.61% |
| free-running predicted previous actions | 7.97% |
| difference | 24.64 percentage points |

四条 validation trajectory 的 teacher-forced action accuracy 分别约为 `78.2% / 59.8% / 66.6% / 78.2%`；free-running 降至 `16.4% / 6.0% / 10.2% / 19.8%`，并快速进入长重复动作区间。

这构成 Case B 的第二条直接证据：即使 observation sequence 不变，只把 previous action 从 teacher history 改成 model history，性能也会快速坍缩。

## 6. Observability and representation probes

所有 probe 按 episode/seed split：fit=`bc_train`，validation=`bc_validation`；所有 normalization/PCA 只在 `bc_train` fit。模型为固定配置的 balanced logistic、ridge 和 kNN diagnostic probes，不用于 rollout 或 promotion。

Validation fixed-logistic balanced accuracy：

| Feature | Tree visible | Contact range | Valid attack geometry |
|---|---:|---:|---:|
| F1 legal vector | 0.403 | 0.389 | 0.417 |
| F2 current RGB | 0.424 | 0.596 | 0.561 |
| F3 four-frame stack | 0.495 | 0.616 | 0.581 |
| F4 prior hand-written | 0.556 | 0.581 | 0.571 |
| F5 frozen CNN embedding | 0.548 | **0.699** | **0.761** |
| F6 teacher-forced GRU hidden | **0.628** | 0.637 | 0.746 |

关键结论：

- `valid_attack_geometry` 在 F5 CNN embedding 中可诊断分离，BA `0.761`，episode-bootstrap interval 约 `[0.693, 0.843]`；
- F5 在该关键 label 上明显高于 F3 four-frame `0.581` 和 F4 hand-written `0.571`；
- 因此预声明的 Case C representation-failure 条件不成立；
- broad visual visibility 仍然偏弱，但当前 `tree_visible` 实际定义是“crosshair raycast hits log”，不是完整画面内任意树可见性，解释时必须保守。

`roughly_centered` 没有可靠的独立 privileged field。当前数据没有 broad tree mask、angular error 或 target screen coordinate，因此被标为 `unsupported_by_current_audit_data`，没有拿 teacher phase 替代。

Raycast distance ridge regression 的 validation 泛化较弱；最佳 F3 four-frame stack `R²≈0.155`。这表明精确连续距离估计仍困难，但不足以解释离散 action fixed point，因为关键 valid-attack label 已经可分，且 token intervention 的因果效应远大于 observation-to-observation policy change。

## 7. Coverage, nearest-neighbor, and OOD

### 7.1 Critical transition coverage

`bc_train` 包含 8 个 seeds，其中 7 条成功 trajectory。现有 labels 支持：

| Transition | Occurrences | Distinct trajectories | Distinct seeds |
|---|---:|---:|---:|
| tree-not-visible → tree-visible | 64 | 7 | 7 |
| visible → approaching | 120 | 8 | 8 |
| approaching → contact range | 46 | 8 | 8 |
| contact range → valid attack geometry | 34 | **7** | **7** |

因此 successful `bc_train` 的七个独立 seeds 都覆盖了 contact → valid-attack，而不是只有一两个 seed。该事实不支持“关键 transition 几乎没有训练覆盖”作为当前主因。

现有数据没有独立的 roughly-centered label，也没有 post-transition frame/vector 来可靠定位 visual block-break 与 pickup 的两个不同时间点。因此：

- visible → roughly-centered：unsupported；
- valid geometry → block break：unsupported；
- block break → pickup：unsupported。

`audit_reward` 没有被猜测成 block-break label。

### 7.2 Existing DAgger1 reuse audit

已有 DAgger1 数据：

- 1,000 ordered samples；
- 2 episodes；
- observation sequence、previous executed action、episode boundary、legal RGB/vector、oracle label 和 audit labels 完整；
- previous executed action 因果对齐通过；
- 可在下一轮作为 recurrent data option，但本轮未用于训练。

其 transition breadth 仍有限，例如 tree visibility event 只来自一个独立 seed；这属于后续数据设计信息，不是本轮 fixed point 的首要解释。

### 7.3 Student OOD

以 `bc_train` leave-one-episode-out nearest-neighbor distance 建立 95th percentile：

| Feature | student_dev over train 95th | student_dev over train 99th |
|---|---:|---:|
| F1 legal vector | 0.0% | 0.0% |
| F2 current RGB | 7.65% | 2.80% |
| F3 four-frame stack | 5.60% | 0.0% |
| F4 hand-written | **17.73%** | 4.90% |
| F5 CNN embedding | 7.72% | 0.0% |
| F6 GRU hidden | 0.0% | 0.0% |

所有 feature spaces 都远低于预声明的 `student OOD95 >= 50%` 主因门槛。Fixed-point orbit 大部分位于训练参考分布的邻近区域，而不是持续进入极端 OOD state。

F4 hand-written space 中，student query 的 nearest train neighbor 有约 `19.4%` 出现 valid-geometry label 不一致，说明局部 collision 确实存在；但 F3/F4 在 student states 上仍能达到约 `0.828/0.793` valid-geometry balanced accuracy，且 F5/F6 validation 也具可分性。因此这些 collision 不足以支持 legal observation aliasing 为主要原因。

## 8. Decision framework result

| Hypothesis | Decision | Reason |
|---|---|---|
| H1 runtime/input integrity | not supported | live/replay exact parity，inputs/hidden dynamic，mapping/reset/preprocessing pass |
| H2 previous-action feedback | **supported, primary** | token argmax change 46.4%，TF–free BA gap 24.6pp，token/obs JS 59.7× |
| H3 hidden attractor | secondary sensitivity | hidden JS large but argmax change only 9.2% |
| H4 learned CNN representation failure | not supported as primary | F5 valid-geometry BA 0.761，优于 F3/F4 |
| H5 observation aliasing | not supported as primary | key label separable；collision exists but not dominant |
| H5 coverage/OOD | not supported as primary | contact→valid covered by 7 seeds；max student OOD95 17.73% |

## 9. Exactly one next intervention

下一轮只训练一个 controlled actor：

```text
current legal RGB
+ legal vector 16
+ GRU hidden
→ action

explicit previous-action embedding removed
```

必须固定：

- 相同 `bc_train` / `bc_validation` 文件和 SHA-256；
- 相同 current `64×64×3` RGB；
- 相同 16-value legal vector；
- 相同 CNN；
- 相同 one-layer GRU hidden size 128；
- 相同 optimizer、loss、class weights、epochs、patience；
- 相同 initialization/training seed 29；
- 相同 teacher、task、success rule 和 autonomous seeds。

本次 intervention 不得同时：

- 恢复 four-frame input；
- 添加 center crop；
- 增加数据或 DAgger2；
- 扩大 CNN/GRU；
- 修改 observation boundary；
- 运行 PPO 或 reward shaping。

这样下一轮可以直接回答：移除 explicit autoregressive action channel 是否足以阻断 fixed point，而不会被表示、数据或规模变化混淆。

## 10. Tests and reproducibility

- Fast tests: `294 passed, 4 deselected`；
- 新增并通过：live/standalone parity、selected/executed mapping、hidden single-step、trace serialization、episode split leakage、train-only scaler/PCA；
- 12/12 runtime traces standalone replay exact；
- MineRL 启动时 JitPack SNAPSHOT metadata 连续超时两次；最终只对 disposable temporary launch script 启用 Gradle `--offline`，复用已经存在的本地 cache。安装目录、actor、teacher、环境规则和实验配置未修改；
- Append-only registry 保留 predeclared record；
- no checkpoint promotion；
- no policy training；
- no protected split access。

主要产物：

- `artifacts/exp12/runtime_integrity_gate.json`
- `artifacts/exp12/checkpoint_rollout_summary.csv`
- `artifacts/exp12/previous_action_sweep.csv`
- `artifacts/exp12/hidden_intervention.csv`
- `artifacts/exp12/forced_vs_free_replay.csv`
- `artifacts/exp12/probe_support.csv`
- `artifacts/exp12/probe_metrics.csv`
- `artifacts/exp12/transition_coverage.csv`
- `artifacts/exp12/nearest_neighbor_audit.csv`
- `artifacts/exp12/ood_summary.json`
- `artifacts/exp12/critical_collision_examples/`
- `artifacts/exp12/decision_summary.json`
