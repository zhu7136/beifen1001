# Proposal

## Why

`dynamic-gait-period` was supposed to keep gait phase continuous under speed-dependent periods, but the live code still uses `(episode_length_buf * step_dt) % current_period / current_period`. When period changes with command speed, that modulus jumps. Policy, critic, `feet_gait`, and `cross_arm_swing_stance` each recompute phase independently (and `cross_arm_swing_stance` only passes `period=current_period` into `gait_phase`), so the four consumers can disagree on the same simulation step.

## What Changes

- Add shared per-env accumulator `dynamic_gait_phase_fraction()` in `mdp/observations.py`.
- `gait_phase()` uses the accumulator only on high-speed paths (`period_end is not None` and `command_name is not None`); flat/rough keep `(t % period) / period`.
- `feet_gait()` reads the same accumulator instead of recomputing modulus phase.
- `cross_arm_swing_stance()` stops recomputing `current_period` and calls `gait_phase()` with full dynamic params.
- Accumulator advances once per real env step (`episode_length_buf` delta); same-step repeats from policy/critic/rewards do not double-increment; episode reset zeros phase.
- Operator follow-up after apply (not this planning step): warm-start HS continue-train from `logs/rsl_rl/limx_hu_d04_01_flat_velocity_hs/2026-10-04_20-53-08_hs-2p4-to-2p8-ft-002/model_900.pt` via `--init_checkpoint`.

## Capabilities

### New Capabilities
- `dynamic-gait-phase`: Continuous, shared gait-phase accumulator for dynamic high-speed gait; policy/critic/rewards must observe the same phase on a step; flat/rough fixed-period behavior unchanged.

### Modified Capabilities
- (none)

## Impact

- **Files**: `mdp/observations.py`, `mdp/rewards.py`; HS config already injects `period_start`/`period_end`/`speed_start`/`speed_end` on policy, critic, `gait`, `cross_arm_swing_stance`.
- **Checkpoint**: same network/obs dims; warm-start compatible; new run needed for clean phase semantics.
- **Incompatibility**: training and play/eval must use the same dynamic-phase code path, or standing/stamp phase will disagree.
- **Not in scope**: curriculum speed gates, reward weights, network, `LIMX_HS_*` curriculum env-var schema.
