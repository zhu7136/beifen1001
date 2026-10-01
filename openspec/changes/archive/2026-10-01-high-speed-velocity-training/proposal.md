# Proposal

## Why

Flat-Velocity training currently measures only aggregate mixed-command error under curriculum ranges (initially tiny, up to limit ~0.8 m/s). That does **not** prove the policy can hold a fixed high speed such as 2.78 m/s. Without a fixed-speed evaluation gate and a gated speed curriculum from the final checkpoint, speed claims are unverifiable and retraining risks forgetting low-speed control.

## What Changes

- Add a **fixed-speed evaluation protocol** that measures, per target speed and seed: forward-speed mean/RMSE, full-episode completion rate, termination rates (`bad_orientation`, base contact, base height), foot slide, joint velocity/torque saturation and action clipping, and power / cost-of-transport.
- Add a **resume-based speed curriculum** that continues from a final checkpoint (e.g. `model_4999.pt`) instead of retraining from scratch.
  - Start from the highest speed the policy already passes in fixed-speed eval.
  - Increase command max by `0.2 m/s` only after ~100–200 iterations with gates met.
  - Freeze curriculum when gates fail; do not force speed.
- Change **command sampling** for high-speed runs:
  - ~40% near current max, ~40% mid-range, ~20% low/stand.
  - Restrict lateral velocity near 0 and yaw to roughly ±0.2–0.3 rad/s during straight-line high-speed phase.
- **Prefer curriculum expansion first**; keep network structure and most reward weights unchanged in the first fine-tune round.
- First fine-tune hyperparameters (conservative): LR ~1e-4, restore actor+critic, action noise ~0.40–0.45, save every 100 iterations, 2000–4000 iterations subject to gates.
- Success criterion is **fixed 2.78 m/s eval success rate and RMSE**, not mixed `Train/mean_reward`.

## Capabilities

### New Capabilities
- `velocity-evaluation`: Fixed-speed velocity policy evaluation — how target speeds, seeds, metrics, and pass/fail gates are defined and reported.
- `velocity-curriculum`: Speed-curriculum training behavior — resume checkpoint, gated speed increments, command sampling mix, straight-line high-speed constraints, and freeze rules when gates fail.

### Modified Capabilities

## Impact

- Code under `source/limx_rl_lab/limx_rl_lab/tasks/locomotion/` (curriculum, command sampling, env cfg overrides).
- Eval tooling under `scripts/` (fixed-speed play/eval script + metric logging).
- Training entry `scripts/rsl_rl/train.py` / agent cfg for resume, LR, noise, save interval overrides.
- W&B project `limx-hu-d04-01` metrics schema (per-speed eval panels vs mixed train reward).
- Does **not** change robot assets, IsaacLab base, or BeyondMimic tasks in the first round.
