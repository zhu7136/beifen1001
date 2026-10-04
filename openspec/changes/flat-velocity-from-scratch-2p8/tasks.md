# Tasks

## 1. Flat command contract

- [x] 1.1 Edit `CommandsCfg` in `source/limx_rl_lab/limx_rl_lab/tasks/locomotion/robots/limx/velocity_env_cfg.py`: `ranges.lin_vel_x=(0.0, 2.8)`, `limit_ranges.lin_vel_x=(0.0, 2.8)`, `lin_vel_y=(-0.05, 0.05)`, `ang_vel_z=(-0.25, 0.25)` on both ranges and limit_ranges; verify the file loads and the values match the delta spec
- [x] 1.2 Confirm `RobotPlayEnvCfg` still uses `ranges = limit_ranges` and that train/play command envelope is 2.8 after the edit; verify by reading the post-init override and the new `CommandsCfg` values
- [x] 1.3 Confirm Flat task still binds `BasePPORunnerCfg` in `tasks/locomotion/robots/limx/__init__.py`; verify registration kwargs unchanged for `LimX-HU-D04-01-Flat-Velocity`

## 2. From-scratch train launch path

- [x] 2.1 Document operator command in change notes / docs section: `python scripts/rsl_rl/train.py --task LimX-HU-D04-01-Flat-Velocity --headless --run_name flat-2p8-scratch-001 --max_iterations 8000` with no `--init_checkpoint`/`--resume`; verify the command string matches `train.py` CLI and does not reference the HS task
- [x] 2.2 Launch or dry-check that a new run would log under `logs/rsl_rl/limx_hu_d04_01_flat_velocity/<timestamp>_flat-2p8-scratch-*`; verify log root is experiment_name `limx_hu_d04_01_flat_velocity`, not `..._hs`

## 3. Fixed-speed eval at 2.8

- [x] 3.1 Document eval command: `scripts/rsl_rl/eval_velocity.py --task LimX-HU-D04-01-Flat-Velocity --checkpoint <run>/model_XXXX.pt --speeds 1.0,1.5,2.0,2.4,2.8 --num_seeds 5 --headless --output logs/eval/<run>_2p8.json`; verify CLI accepts `--speeds` including `2.8` and task Flat
- [x] 3.2 Confirm eval gates for 2.8 use `SpeedGates.for_target` / report fields for completion, orientation rate, and RMSE; verify JSON output contains per-seed records at 2.8 with pass/fail thresholds completion ≥ 0.97, orientation < 0.02, RMSE < 0.25
- [ ] 3.3 After training produces checkpoints, run eval at 2.8 on the latest model and save report; verify `logs/eval/*_2p8.json` exists and summarizes pass/fail at 2.8

## 4. Validation

- [x] 4.1 Run `openspec validate --change flat-velocity-from-scratch-2p8`; verify no blocking errors
- [x] 4.2 Re-read delta specs vs `velocity_env_cfg.py` / eval gates; verify no requirement depends on resume checkpoint, HS task, or `LIMX_HS_*` env vars for this path
