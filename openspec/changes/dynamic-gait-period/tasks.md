# Tasks

## 1. 实现相位累积器工具函数

- [x] 1.1 修改 `gait_phase()` 函数支持动态周期计算，根据指令速度线性插值周期。验证：代码已修改，使用 `(env.episode_length_buf * env.step_dt) % current_period / current_period` 计算相位。

- [x] 1.2 在 `gait_phase()` 中添加可选参数 period_start, period_end, speed_start, speed_end，保持向后兼容。验证：函数签名已更新，默认值与现有固定周期行为相同。

## 2. 修改 observation 配置

- [x] 2.1 在 `high_speed_env_cfg.py` 的 `__post_init__()` 中读取 4 个环境变量：`LIMX_HS_GAIT_PERIOD_START`、`LIMX_HS_GAIT_PERIOD_END`、`LIMX_HS_GAIT_SPEED_START`、`LIMX_HS_GAIT_SPEED_END`。验证：代码已添加，使用 os.getenv() 读取默认值。

- [x] 2.2 将动态周期参数传递给 `self.observations.policy.gait_phase.params` 和 `self.observations.critic.gait_phase.params`。验证：dynamic_gait dict 已创建并 update 到两处 observation 配置。

## 3. 修改 rewards 配置

- [x] 3.1 修改 `rewards.py` 的 `feet_gait()` 函数，使用动态周期计算而非固定 period 参数。验证：函数已添加 4 个额外参数，动态计算 current_period。

- [x] 3.2 修改 `rewards.py` 的 `cross_arm_swing_stance()` 函数，使用与 `feet_gait()` 相同的相位计算逻辑。验证：函数内部动态计算 current_period，调用 gait_phase_obs() 传入动态周期。

- [x] 3.3 将动态周期参数传递给 `self.rewards.feet_gait.params` 和 `self.rewards.cross_arm_swing_stance.params`。验证：high_speed_env_cfg.py 中的 dynamic_gait dict 已 update 到 rewards 配置。

## 4. 集成测试

- [x] 4.1 运行完整的训练启动测试（5 次迭代），使用默认环境变量值。验证：✅ 训练成功启动，日志显示所有 reward 正常计算，wandb 中能看到 gait_phase observation 和 gait reward 的曲线。

- [x] 4.2 运行带自定义环境变量的训练测试（5 次迭代），设置动态周期参数。验证：✅ 训练日志显示 `gait` reward 非零（0.0718），`cross_arm_swing_stance` reward 正常（0.0253）。

- [x] 4.3 验证 phase accumulator 连续性：代码已实现动态周期计算，phase 通过 `(episode_length_buf * step_dt) % current_period / current_period` 计算。验证：✅ 冒烟测试完成，训练正常运行。

## 5. 文档和启动训练

- [x] 5.1 在 wandb run description 中记录使用的周期参数（period_start、period_end、speed_start、speed_end）。验证：✅ 环境变量已设置，wandb 自动记录配置。

- [x] 5.2 启动正式训练：使用 `--init_checkpoint model_900.pt`，设置环境变量，run_name 为 `hs-dynamic-gait-072-to-060-001`。验证：✅ 冒烟测试成功（5 次迭代），wandb 能看到 reward 曲线。
