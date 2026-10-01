# Round-2 HS curriculum — checkpoint & operator notes

## Resume checkpoint order (required)

1. **Preferred:** `logs/rsl_rl/limx_hu_d04_01_flat_velocity_hs/2026-10-01_20-46-27_hs-ft-001/model_2999.pt`
2. **Fallback:** W&B `model_2900.pt` from run `clykvseg` / `2026-10-01_20-46-27_hs-ft-001`
3. **Do not use:** `model_2400.pt` as round-2 start (wastes ~500 iters already adapted at 2.0 m/s)

## Round-2 config

| Field | Value |
|---|---|
| initial_max_speed_mps | 2.0 |
| target_speed_mps | **2.4** |
| speed_increment_mps | 0.1 |
| max_iterations | 3000 |
| save_interval | 100 |
| near/mid/low mix | 0.50 / 0.35 / 0.15 |
| lin_vel_y | [-0.05, 0.05] |
| ang_vel_z | [-0.25, 0.25] |

**Staged order:** 2.0 → 2.1 → 2.2 → 2.3 → 2.4  
**Next round (2.78 m/s):** only after fixed-speed eval at 2.4 passes strict gates (completion ≥ 97%, RMSE ≤ 0.25).

## Rolling frontier gates (curriculum promote)

- Frontier samples: `vx_cmd >= current_max_speed_mps`
- Rolling window: **200** iterations
- Consecutive pass required: **200**
- Dwell at current speed: **≥ 300** iterations
- Gates: completion ≥ **0.97**, orientation ≤ **0.02**, RMSE ≤ **0.28**, saturation ≤ **0.10**
- On fail: **hold** current speed — no auto-degrade, no promote

## Isolation vs round-1 (`clykvseg`)

Changed only: increment 0.2→0.1, instant→rolling frontier gates, near-fraction 40%→50%.  
Rewards / PPO / network unchanged (LR 1e-4, noise 0.42).

## Launch

```bash
CKPT=logs/rsl_rl/limx_hu_d04_01_flat_velocity_hs/2026-10-01_20-46-27_hs-ft-001/model_2999.pt
export LIMX_HS_INITIAL_MAX_SPEED=2.0
export LIMX_HS_TARGET_SPEED=2.4
export LIMX_HS_SPEED_INCREMENT=0.1
export LIMX_HS_NEAR_FRAC=0.50
export LIMX_HS_MID_FRAC=0.35
export LIMX_HS_LOW_FRAC=0.15

python scripts/rsl_rl/train.py \
  --task LimX-HU-D04-01-Flat-Velocity-HS \
  --headless \
  --init_checkpoint "$CKPT" \
  --run_name hs-round2-001 \
  --logger wandb --log_project_name limx-hu-d04-01 \
  --max_iterations 3000
```
