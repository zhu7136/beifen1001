# High-speed velocity training plan (LimX HU-D04 flat)

Authoritative acceptance metric for high-speed capability is **fixed-speed evaluation**, not mixed `Train/mean_reward`.

## 1. Fixed-speed evaluation

```bash
python scripts/rsl_rl/eval_velocity.py \
  --task LimX-HU-D04-01-Flat-Velocity \
  --checkpoint logs/rsl_rl/limx_hu_d04_01_flat_velocity/<run>/model_4999.pt \
  --speeds 1.0,1.4,1.8,2.2,2.5,2.78 \
  --num_seeds 5 \
  --headless \
  --output logs/eval/<run>_model_4999.json
```

Per speed × seed, the report records:

- forward-speed mean and RMSE
- full-episode completion rate
- `bad_orientation` / base contact / base height termination rates
- foot slide
- joint velocity/torque saturation and action clipping
- mean joint power and cost-of-transport

Default gates (used for curriculum start):

| Gate | Threshold |
|---|---|
| Completion rate | > 95% |
| Orientation term rate | < 2% |
| Speed RMSE | < 0.25 m/s |

`initial_max_speed_mps` in the JSON is the highest speed that passes these gates.

## 2. Resume-based speed curriculum (not from scratch)

Register task: `LimX-HU-D04-01-Flat-Velocity-HS`

```bash
# set start speed from eval
export LIMX_HS_INITIAL_MAX_SPEED=1.4   # example; use your eval recommendation
export LIMX_HS_TARGET_SPEED=2.78
export LIMX_HS_SPEED_INCREMENT=0.2

python scripts/rsl_rl/train.py \
  --task LimX-HU-D04-01-Flat-Velocity-HS \
  --headless \
  --init_checkpoint logs/rsl_rl/limx_hu_d04_01_flat_velocity/<run>/model_4999.pt \
  --run_name hs-ft-001 \
  --logger wandb --log_project_name limx-hu-d04-01 \
  --max_iterations 3000 \
  --learning_rate 1e-4
```

Agent defaults for `HighSpeedPPORunnerCfg`:

- LR `1e-4`
- action noise `0.42` (override via rsl-rl CLI if needed)
- `save_interval=100`
- `max_iterations=3000` (2000–4000 band)

Curriculum behavior:

- Promote `lin_vel_x` max by `0.2 m/s` only after gates hold for ~150 PPO iterations (`stable_min_steps=150*24`)
- Ceiling `target_speed_mps` (default 2.78)
- Freeze when gates fail (completion / orientation / RMSE / saturation)
- Command mix ~40% near cap, ~40% mid, ~20% low/stand
- Straight-line phase: `vy≈0`, `|wz|≤0.25`
- **Does not** expand `lin_vel_y` with x (unlike baseline `lin_vel_cmd_levels`)
- First round keeps baseline reward weights (track_lin_vel_xy=1.5, etc.)

Curriculum logs (W&B / TensorBoard):

```
Curriculum/gated_speed_levels/current_max_speed_mps
Curriculum/gated_speed_levels/completion_rate
Curriculum/gated_speed_levels/speed_rmse_mps
Curriculum/gated_speed_levels/frozen
Curriculum/gated_speed_levels/freeze_reason
```

## 3. After fine-tune

Re-run fixed-speed eval matrix on the HS checkpoint. Compare 2.78 m/s completion and RMSE to gates.

If curriculum freezes below target, diagnose **without** default reward reweighting first:

1. tracking error / RMSE at the frozen max
2. torque / joint velocity / action-rate saturation
3. feet clearance vs slide
4. orientation / contact penalties

## 4. Rollback

Stop HS run; keep using flat checkpoints under `logs/rsl_rl/limx_hu_d04_01_flat_velocity/`.
