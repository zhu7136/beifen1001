# Proposal

## Why

Previous HS pipeline (resume from `model_2999.pt` / `model_4999.pt`, rolling-frontier curriculum, target 2.78) produced training that is not trusted. Operators want a clean restart: from-scratch Flat velocity training with command ceiling fixed at **2.8 m/s**, no warm-start, no HS curriculum task.

## What Changes

- **Flat task command surface:** `LimX-HU-D04-01-Flat-Velocity` command ranges/limits become forward `(0.0, 2.8)` m/s, lateral `(-0.05, 0.05)`, yaw `(-0.25, 0.25)`. `ranges == limit_ranges` so reward-gated `lin_vel_cmd_levels` has nothing left to expand.
- **From-scratch training path:** launch Flat task without `--init_checkpoint` / `--resume`, using `BasePPORunnerCfg` (LR `1e-3`, action noise `1.0`, `max_iterations` 8000). Do **not** use `LimX-HU-D04-01-Flat-Velocity-HS` or HS fine-tune PPO defaults.
- **Authority metric:** capability claims at 2.8 m/s use fixed-speed eval at 2.8 (completion ≥ 0.97, orientation term rate < 0.02, speed RMSE < 0.25 m/s), not mixed train reward.
- **Specs:** update `velocity-curriculum` and `velocity-evaluation` so the 2.8 from-scratch path is the documented contract; resume/HS staged curriculum is no longer the path for this goal.
- **BREAKING (for this goal):** resume-based HS curriculum and 2.78 ceiling are out of scope for the 2.8 from-scratch run. HS task registration may remain in code but is not part of this training path.

## Capabilities

### New Capabilities

_(none)_

### Modified Capabilities

- `velocity-curriculum`: from-scratch Flat command contract at 2.8 m/s; remove resume/gated-increment requirements as the path for this target; from-scratch PPO defaults; fixed command sampling (no near/mid/low HS mix).
- `velocity-evaluation`: fixed-speed matrix and pass gates center on 2.8 m/s; high-speed claims use 2.8 eval results.

## Impact

- Code: `source/limx_rl_lab/limx_rl_lab/tasks/locomotion/robots/limx/velocity_env_cfg.py` (`CommandsCfg.ranges` / `limit_ranges`); optional `RobotPlayEnvCfg` consistency; docs `docs/high_speed_velocity_training.md` (path section only).
- Train entry: `scripts/rsl_rl/train.py --task LimX-HU-D04-01-Flat-Velocity` from scratch (no checkpoint/resume flags).
- Eval: `scripts/rsl_rl/eval_velocity.py --task LimX-HU-D04-01-Flat-Velocity --checkpoint <run>/model_*.pt --speeds ...,2.8`.
- Logs: new run under `logs/rsl_rl/limx_hu_d04_01_flat_velocity/<timestamp>_flat-2p8-scratch-*/`.
- Unchanged: robot assets, reward terms/weights, network dims, Rough task, HS env code (not used by this path).
- OpenSpec: in-progress change `rolling-frontier-speed-curriculum` remains for the abandoned resume pipeline; this change does not implement that pipeline.
