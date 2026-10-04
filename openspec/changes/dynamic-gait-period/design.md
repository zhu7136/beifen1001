# Design

## Context

当前 `gait_phase` observation 和相关 reward（`feet_gait`、`cross_arm_swing_stance`）使用固定的 period=0.72 秒。在高速（2.4-2.8 m/s）训练时，固定步频导致策略需要在较大步幅下维持稳定性，加剧了扭矩饱和问题。

See proposal.md for full motivation.

## Goals / Non-Goals

**Goals:**
- 实现速度相关的动态步态周期，从 0.72s（2.0 m/s）平滑过渡到 0.60s（2.8 m/s）
- 维护相位累积器，确保 command 重采样时相位连续
- 环境变量可配置周期参数
- observation 和 reward 使用相同的相位计算逻辑

**Non-Goals:**
- 不修改 curriculum 速度门限逻辑
- 不改 reward 权重或网络结构
- 不改变低速（<2.0 m/s）的固定周期行为
- 不变推理/eval 的代码路径（仅训练时使用动态周期）

## Decisions

### Decision 1: Phase accumulator 模式

**Approach:** 使用 `phase = (phase + step_dt / period) % 1.0` 累积相位，而非 `(episode_length * step_dt) % period`。

**Rationale:**
- 如果 period 在 episode 中间变化，`(t % period)` 会导致相位跳跃
- Accumulator 模式每次步进增加 `step_dt/period`，自然适应周期变化
- Episode reset 时相位归零，保持一致性

**Alternatives considered:**
1. **使用 episode_length 直接计算**: 简单但 period 变化时会跳跃 - 被否决
2. **在每个 step 重新计算相位**: 计算开销大且复杂 - 被否决

### Decision 2: 环境变量配置

**Approach:** 在 `high_speed_env_cfg.py` 的 `__post_init__()` 中读取 4 个环境变量，同步到 observation 和 reward 的参数中。

**Rationale:**
- 与现有的 `LIMX_HS_*` 环境变量模式一致
- 无需修改配置文件格式
- 训练脚本可以灵活调整参数

**Alternatives considered:**
1. **YAML 配置文件**: 需要额外的配置加载逻辑 - 被否决
2. **命令行参数**: 需要修改训练脚本的 arg 解析 - 被否决

### Decision 3: 线性插值公式

**Approach:** `period = period_start + alpha * (period_end - period_start)`，其中 `alpha = clamp((speed - speed_start) / (speed_end - speed_start), 0, 1)`。

**Rationale:**
- 简单且可解释
- 在范围外自动钳位，避免极端值
- 与现有的 curriculum 速度插值模式一致

**Alternatives considered:**
1. **分段线性或多项式拟合**: 更灵活但需要调参 - 被否决
2. **可学习的周期函数**: 增加复杂性且不稳定 - 被否决

### Decision 4: 四处处同步参数

**Approach:** observation (`gait_phase`)、reward (`feet_gait`、`cross_arm_swing_stance`)、以及任何使用 phase 的地方都从环境变量读取相同的参数。

**Rationale:**
- 避免相位错位导致策略学习到错误的时序关系
- 单一数据源（环境变量）确保一致性

**Alternatives considered:**
1. **集中配置对象**: 需要全局配置管理器 - 被否决
2. **每个模块独立配置**: 容易出错 - 被否决

## Risks / Trade-offs

### Risk 1: Phase 不一致导致策略性能下降

**Impact:** 如果 observation 的 phase 和 reward 的 phase 不同步，策略会收到矛盾的信号。

**Mitigation:**
- 使用共享的辅助函数计算周期
- 单元测试验证 observation 和 reward 的 phase 相同
- 训练初期监控 reward 和 observation 的相关性

### Risk 2: 旧 checkpoint 与新周期不兼容

**Impact:** 从旧 checkpoint 恢复训练时，策略可能不适应新的周期计算方式。

**Mitigation:**
- 使用 `--init_checkpoint` 而不是 `--resume`，作为新训练处理
- 在前 100-200 次迭代使用较小的学习率暖身
- 记录新实验的 run name（如 `hs-dynamic-gait-072-to-060-001`）以便区分

### Risk 3: 环境变量未设置时的默认行为

**Impact:** 如果操作员忘记设置环境变量，系统必须回退到合理默认值。

**Mitigation:**
- 默认值 `period_start=0.72`、`period_end=0.60` 匹配当前固定周期行为
- 在训练日志中打印实际使用的周期参数
- 文档明确说明默认值

### Trade-off: 简单性 vs. 灵活性

**Trade-off:** 当前设计仅支持线性插值，不支持更复杂的周期 - 速度关系（如多项式、分段函数）。

**Rationale:** 线性插值已足够覆盖 2.0-2.8 m/s 范围，且更容易调试和解释。未来如需要可扩展。

## Migration Plan

### 步骤 1: 修改 observation 和 reward 函数
- 在 `observations.py` 的 `gait_phase()` 中添加 phase accumulator 逻辑
- 在 `rewards.py` 的 `feet_gait()` 和 `cross_arm_swing_stance()` 中使用相同的 accumulator

### 步骤 2: 修改 high_speed_env_cfg.py
- 在 `__post_init__()` 中读取 4 个环境变量
- 将参数传递给 observation 和 reward 配置

### 步骤 3: 测试
- 运行单元测试验证 observation 和 reward 的 phase 一致
- 运行短训练（100-200 次迭代）验证无崩溃

### 步骤 4: 启动完整训练
- 使用 `--init_checkpoint` 从 `model_900.pt` 或最新 checkpoint 启动
- 设置 4 个环境变量
- 监控 wandb 日志确认周期随速度变化

## Open Questions

无。所有设计决策已确定。
