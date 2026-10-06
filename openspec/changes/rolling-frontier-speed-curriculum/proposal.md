# Proposal

## Why

Round-1 HS fine-tune (`hs-ft-001`, `model_2999.pt`) reached command max 2.0 m/s then froze. Instant episode-boundary gates let low-speed mixed commands dilute RMSE, so speeds such as 1.6–1.8 m/s could promote despite frontier tracking failing. Next round must start from the late checkpoint (2.0-adapted) and fix gate math before pushing to 2.4 m/s.

## What Changes

- **Resume checkpoint policy:** prefer local `model_2999.pt` if present; else W&B `model_2900.pt`. Do **not** restart from `model_2400.pt` (later policy already adapted ~500 iters at 2.0 m/s).
- **Round-2 curriculum config (gate-fix round only):**
  - `initial_max_speed_mps=2.0`
  - `target_speed_mps=2.4` (not 2.78 this round)
  - `speed_increment_mps=0.1`
  - `max_iterations=3000`, `save_interval=100`
  - command mix: near **50%**, mid **35%**, low/stand **15%**
  - `lin_vel_y=[-0.05,0.05]`, `ang_vel_z=[-0.25,0.25]`
  - staged levels 2.0 → 2.1 → … → 2.4; only after 2.4 fixed-speed eval passes open the next round toward 2.78
- **Replace instant gates with rolling frontier gates:**
  - rolling window over last **200 iterations**
  - minimum **300 iterations** at each speed before promotion eligibility
  - stats computed only on **frontier** samples: `frontier_mask = vx_cmd >= current_max_speed_mps`
  - promote only if for 200 consecutive iterations: frontier completion ≥ 97%, frontier bad-orientation ≤ 2%, frontier RMSE ≤ 0.28 m/s, actuator saturation ≤ 10%
  - on failure: **hold** current speed; no auto-degrade; no promotion
- **Post-2.4 fixed-speed eval is stricter:** RMSE ≤ 0.25 m/s and completion ≥ 97%.
- **Round-2 changes only these levers** (so results compare cleanly to `clykvseg`): increment 0.2→0.1, instant→rolling gates, near-fraction 40%→50%. Rewards and PPO stay unchanged.
- **Per-iteration frontier metrics** (not only episode-boundary): curriculum speed/frontier RMSE/completion/orientation/saturation, consecutive pass iterations, gate reason code, plus commanded/measured vx, speed-bucketed RMSE, joint-vel/torque saturation, action clipping.

## Capabilities

### New Capabilities

### Modified Capabilities
- `velocity-curriculum`: rolling frontier gate calculation, staged increment/sampling defaults for round 2, hold-on-fail (no auto-degrade), per-iteration frontier curriculum metrics.
- `velocity-evaluation`: stricter high-speed acceptance gates used after each staged speed (especially ≥2.4 m/s fixed-speed eval).

## Impact

- Code: `limx_rl_lab/utils/velocity_curriculum.py`, `tasks/locomotion/mdp/curriculums.py`, `mdp/commands/velocity_command.py`, `tasks/locomotion/robots/limx/high_speed_env_cfg.py`, agent/env cfg overrides, train logging path.
- Ops: `scripts/rsl_rl/train.py` resume from `model_2999.pt`; optional pre-eval at fixed 2.0 m/s; W&B project `limx-hu-d04-01`.
- Logs: `logs/rsl_rl/limx_hu_d04_01_flat_velocity_hs/2026-10-01_20-46-27_hs-ft-001/` (`model_2999.pt` exists locally).
- No robot assets, no BeyondMimic, no reward reweight, no PPO redesign in this round.
