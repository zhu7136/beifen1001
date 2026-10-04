# Proposal

## Why

当前高速训练中，步态周期（gait period）固定为 0.72 秒，在 2.4-2.8 m/s 速度区间无法适配更高的步频需求。实测在 2.4 m/s 时策略已出现明显的扭矩饱和（45-49%），而固定步态周期导致：
1. 低速和高速使用相同的步频，无法在高速时提高步频以降低步幅
2. 策略观察的 gait phase 与奖励计算的相位同步，但周期固定
3.  curriculum 在 2.4 m/s 卡点，部分原因是步态周期与速度不匹配

通过引入动态步态周期，让周期随指令速度从 0.72s（2.0 m/s）平滑过渡到 0.60s（2.8 m/s），可以在 2.4 m/s 时步频提高约 9%，帮助策略更好地分配步态相位，缓解扭矩饱和问题。

## What Changes

- **新增动态步态周期计算**：根据当前指令速度动态调整 gait period，从 0.72s（2.0 m/s）线性过渡到 0.60s（2.8 m/s）
- **引入 phase accumulator**：维护每个环境的累积相位，避免 command 重采样时相位跳跃
- **环境变量配置**：支持通过 `LIMX_HS_GAIT_PERIOD_START`、`LIMX_HS_GAIT_PERIOD_END`、`LIMX_HS_GAIT_SPEED_START`、`LIMX_HS_GAIT_SPEED_END` 配置动态周期的参数
- **同步修改三处**：`gait_phase` observation、`feet_gait` reward、`cross_arm_swing_stance` reward 使用统一的动态周期
- **BREAKING**：动态周期下，训练和推理必须使用相同的周期计算逻辑，否则相位会错位

## Capabilities

### New Capabilities
- `dynamic-gait-period`: 根据指令速度动态调整步态周期，maintain 相位连续性，支持环境变量配置

### Modified Capabilities
- `velocity-curriculum`: 步态周期不再是固定值，而是速度的函数，影响 curriculum 的速度 - 步态匹配行为

## Impact

- **文件修改**:
  - `source/limx_rl_lab/limx_rl_lab/tasks/locomotion/mdp/observations.py` - gait_phase 观察量支持动态周期
  - `source/limx_rl_lab/limx_rl_lab/tasks/locomotion/mdp/rewards.py` - feet_gait 和 cross_arm_swing_stance 奖励使用动态周期
  - `source/limx_rl_lab/limx_rl_lab/tasks/locomotion/robots/limx/high_speed_env_cfg.py` - 添加环境变量配置
- **训练兼容性**: 旧的 checkpoint 可以继续使用，但 period 相关超参数会变化
- **推理一致性**: 推理/eval 时必须使用与训练相同的周期参数
- **向后兼容**: 默认参数（0.72s 固定周期）保持不变，通过环境变量启用动态周期
