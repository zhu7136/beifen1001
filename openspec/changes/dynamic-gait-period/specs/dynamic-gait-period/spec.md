# Spec Delta

## Purpose

根据指令速度动态调整步态周期，实现从低速到高速的平滑步频过渡，同时保持相位连续性，避免 command 重采样时相位跳跃。

## ADDED Requirements

### Requirement: 动态步态周期计算
系统 SHALL 根据当前指令速度动态计算步态周期，从 `period_start`（对应 `speed_start`）线性过渡到 `period_end`（对应 `speed_end`）。

#### Scenario: 2.0 m/s 时使用 0.72s 周期
- **WHEN** 指令速度为 2.0 m/s
- **THEN** 步态周期为 0.72 秒

#### Scenario: 2.8 m/s 时使用 0.60s 周期
- **WHEN** 指令速度为 2.8 m/s
- **THEN** 步态周期为 0.60 秒

#### Scenario: 2.4 m/s 时线性插值
- **WHEN** 指令速度为 2.4 m/s
- **THEN** 步态周期为 0.66 秒（(2.4-2.0)/(2.8-2.0) = 0.5，0.72 + 0.5*(0.60-0.72) = 0.66）

#### Scenario: 速度低于 2.0 m/s 时使用起始周期
- **WHEN** 指令速度 < 2.0 m/s
- **THEN** 步态周期保持 0.72 秒（不继续降低）

#### Scenario: 速度高于 2.8 m/s 时使用结束周期
- **WHEN** 指令速度 > 2.8 m/s
- **THEN** 步态周期保持 0.60 秒（不继续提高）

### Requirement: 相位累积器维护
系统 SHALL 为每个环境维护相位累积器（phase accumulator），使用 `phase = (phase + step_dt / period) % 1.0` 更新，避免 command 重采样时相位跳跃。

#### Scenario: Episode reset 时相位清零
- **WHEN** 环境 episode reset
- **THEN** 对应环境的相位累积器归零

#### Scenario: 周期变化时相位连续
- **WHEN** 步态周期从一个值变为另一个值
- **THEN** 相位累积器继续累加，不发生跳跃

#### Scenario: Command 重采样时相位稳定
- **WHEN** command 在 resampling interval 重新采样
- **THEN** 相位累积器不受影响，保持连续

### Requirement: 环境变量配置
系统 SHALL 支持通过环境变量配置动态步态周期的参数：`LIMX_HS_GAIT_PERIOD_START`、`LIMX_HS_GAIT_PERIOD_END`、`LIMX_HS_GAIT_SPEED_START`、`LIMX_HS_GAIT_SPEED_END`。

#### Scenario: 默认周期参数
- **WHEN** 环境变量未设置
- **THEN** 使用默认值：period_start=0.72, period_end=0.60, speed_start=2.0, speed_end=2.8

#### Scenario: 自定义周期范围
- **WHEN** 设置 `LIMX_HS_GAIT_PERIOD_START=0.70` 和 `LIMX_HS_GAIT_PERIOD_END=0.55`
- **THEN** 周期范围从 0.70s 过渡到 0.55s

#### Scenario: 自定义速度范围
- **WHEN** 设置 `LIMX_HS_GAIT_SPEED_START=1.8` 和 `LIMX_HS_GAIT_SPEED_END=3.0`
- **THEN** 速度范围从 1.8 m/s 到 3.0 m/s 之间进行线性插值

### Requirement: 观察量、奖励和约束同步使用动态周期
gait phase observation、feet_gait reward、cross_arm_swing_stance reward 三处 SHALL 使用相同的动态周期计算逻辑，保持相位一致性。

#### Scenario: 观察与奖励相位一致
- **WHEN** gait phase observation 计算相位
- **THEN** feet_gait 和 cross_arm_swing_stance reward 使用相同的相位值

#### Scenario: 四处处配置参数一致
- **WHEN** 动态周期参数被配置
- **THEN** observations.gait_phase、rewards.feet_gait、rewards.cross_arm_swing_stance、rewards.gait（如果存在）使用相同的 period、period_start、period_end、speed_start、speed_end 参数

## Modified Requirements

无 - `velocity-curriculum` spec 不需要修改，因为动态周期是内部实现细节，curriculum 的速度门限行为保持不变。
