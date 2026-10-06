# Tasks

## 1. Shared accumulator in observations

- [x] 1.1 Add `dynamic_gait_phase_fraction(env, command_name, period_start, period_end, speed_start, speed_end)` to `source/limx_rl_lab/limx_rl_lab/tasks/locomotion/mdp/observations.py` implementing step-delta accumulation, reset on step decrease/0, and speed-clamped period. Verify: helper exists and returns shape `(num_envs,)` in `[0, 1)`.
- [x] 1.2 Gate `gait_phase()` dynamic path on `period_end is not None and command_name is not None`; otherwise use `(episode_length_buf * step_dt) % period / period`. Verify: with only `period=0.72`, code path does not call the accumulator.
- [x] 1.3 Export/import path for `dynamic_gait_phase_fraction` via `mdp` package if rewards import from `.observations` directly — confirm `from .observations import dynamic_gait_phase_fraction` works. Verify: import succeeds in a Python one-liner.

## 2. Rewards read the same phase

- [x] 2.1 In `feet_gait()`, replace independent modulus phase with `dynamic_gait_phase_fraction(...).unsqueeze(1)` when dynamic params/command_name are present; else keep fixed `period` modulus. Verify: no separate alpha/current_period formula remains for the dynamic branch.
- [x] 2.2 In `cross_arm_swing_stance()`, delete local current_period computation; call `gait_phase_obs(env, period=period, command_name=command_name, period_start=..., period_end=..., speed_start=..., speed_end=...)[:, 0]`. Verify: no second `(t % period)` path in arm-swing reward.
- [x] 2.3 Add unit test file (stdlib/assert style, no new framework) covering: (a) period change does not jump phase; (b) same-step double call does not double-increment; (c) episode reset zeros phase; (d) flat-style call without `period_end` does not create dynamic buffer requirement. Verify: tests pass under project test runner or `python -m pytest` if already used.

## 3. HS config consistency check (read-only confirm)

- [x] 3.1 Confirm `high_speed_env_cfg.py` still updates policy, critic, `rewards.gait`, `rewards.cross_arm_swing_stance` with `period_start/period_end/speed_start/speed_end`. Verify: all four params dicts contain the dynamic keys; no code change required unless a key is missing.

## 4. Integration verification + operator continue-train note

- [x] 4.1 Smoke-import observations/rewards modules; optionally run existing HS short train only after apply workflow is authorized by a later user request. Verify: no ImportError; planning phase does not launch training.
- [x] 4.2 Document post-apply continue-train command (docs or task notes only; do not execute here):

```bash
CKPT=logs/rsl_rl/limx_hu_d04_01_flat_velocity_hs/2026-10-04_20-53-08_hs-2p4-to-2p8-ft-002/model_900.pt
export LIMX_HS_INITIAL_MAX_SPEED=2.4
export LIMX_HS_TARGET_SPEED=2.8
export LIMX_HS_SPEED_INCREMENT=0.1
export LIMX_HS_SATURATION_RATE_MAX=0.55

python scripts/rsl_rl/train.py \
  --task LimX-HU-D04-01-Flat-Velocity-HS \
  --headless \
  --init_checkpoint "$CKPT" \
  --run_name hs-2p4-to-2p8-ft-003 \
  --logger wandb --log_project_name limx-hu-d04-01 \
  --learning_rate 1e-4 \
  --max_iterations 3000
```

- [x] 4.3 Re-run `openspec validate --change continuous-gait-phase` after artifacts exist; verify no blocking errors. (If validate is only needed after apply, mark as post-apply operator step.)
