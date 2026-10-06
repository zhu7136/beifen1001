# Change notes — flat-velocity-from-scratch-2p8

## Operator commands (run yourself; training not started by agent)

### Train (from scratch, Flat, cmd ceiling 2.8)

```bash
export WANDB_API_KEY=your_key
tmux new -s flat28

python scripts/rsl_rl/train.py \
  --task LimX-HU-D04-01-Flat-Velocity \
  --headless \
  --run_name flat-2p8-scratch-001 \
  --logger wandb \
  --log_project_name limx-hu-d04-01 \
  --max_iterations 8000
```

- No `--init_checkpoint`, no `--resume`
- Not `LimX-HU-D04-01-Flat-Velocity-HS`
- Log path: `logs/rsl_rl/limx_hu_d04_01_flat_velocity/<timestamp>_flat-2p8-scratch-001/`
- Agent defaults: `BasePPORunnerCfg` LR `1e-3`, noise `1.0`, `save_interval=100`

### Eval after checkpoints exist

```bash
python scripts/rsl_rl/eval_velocity.py \
  --task LimX-HU-D04-01-Flat-Velocity \
  --checkpoint logs/rsl_rl/limx_hu_d04_01_flat_velocity/<run>/model_7999.pt \
  --speeds 1.0,1.5,2.0,2.4,2.8 \
  --num_seeds 5 \
  --headless \
  --output logs/eval/<run>_2p8.json
```

Gates at 2.8: completion ≥ 0.97, orientation < 0.02, RMSE < 0.25 m/s.

## Code landed

- `velocity_env_cfg.py` `CommandsCfg`: `lin_vel_x=(0.0, 2.8)` on ranges and limit_ranges; `lin_vel_y=(-0.05, 0.05)`; `ang_vel_z=(-0.25, 0.25)`.
- `RobotPlayEnvCfg` still `ranges = limit_ranges` → same envelope.
- Flat gym registration still uses `BasePPORunnerCfg`.
- Docs: `docs/high_speed_velocity_training.md` §0/§1 for this path.

## Not done until you train

- Task 3.3: run eval at 2.8 on produced checkpoints (blocked on training output).
