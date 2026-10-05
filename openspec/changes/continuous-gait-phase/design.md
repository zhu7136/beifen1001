# Design

## Context

See proposal.md - Why.

Observed live code (`source/limx_rl_lab/limx_rl_lab/tasks/locomotion/`):

- `observations.gait_phase()` always interpolates `current_period` from `base_velocity` speed with hardcoded defaults `period_start=0.72`, `period_end=0.60`, then uses `(episode_length_buf * step_dt) % current_period / current_period`. The `period` argument is ignored when phase is computed. Flat/rough tasks that pass only `period=0.72` still take the dynamic modulus path.
- `rewards.feet_gait()` and `cross_arm_swing_stance()` each recompute the same alpha/period formula; `cross_arm_swing_stance` then calls `gait_phase_obs(env, period=current_period)` without `command_name`/`period_*`/`speed_*`, so observation and arm-swing can disagree.
- `high_speed_env_cfg.py` already injects `period`, `period_start`, `period_end`, `speed_start`, `speed_end` into policy, critic, `rewards.gait`, and `rewards.cross_arm_swing_stance`.
- Prior OpenSpec `dynamic-gait-period` design required accumulator semantics; tasks.md still records modulus implementation. This change is the fix, not a rewrite of curriculum gates.

Operator continue-train target after apply (planning note only):  
`logs/rsl_rl/limx_hu_d04_01_flat_velocity_hs/2026-10-04_20-53-08_hs-2p4-to-2p8-ft-002/model_900.pt` via `--init_checkpoint`.

## Goals / Non-Goals

**Goals:**
- One per-env phase buffer advanced only when `episode_length_buf` actually increases
- Dynamic period interpolation shared by observation + gait rewards on HS tasks
- Flat/rough fixed-period path preserved when dynamic params are absent
- Same-step idempotence for policy/critic/reward re-entry

**Non-Goals:**
- Curriculum speed promotion / saturation / RMSE gates
- Reward weight or network changes
- New gym task ids or W&B run management
- Launching training in this planning phase

## Decisions

### Decision 1: Accumulator on env attribute, advanced by step delta

Store `env._dynamic_gait_phase` and `env._dynamic_gait_last_step`. Each call computes `delta_steps = clamp(steps - last_step, min=0)`, updates phase by `delta_steps * step_dt / current_period`, `remainder_(1.0)`, zeros on `steps < last_step` or `steps == 0`, then `last_step.copy_(steps)`.

**Rationale:** Correct under repeated same-step manager evaluation; continuous when period changes; resets with episode.

**Alternatives:** Recompute `(t % period)/period` each call — jumps when period changes; rejected. Torch.compile-free stateless formula cannot know when period last changed without step bookkeeping.

### Decision 2: Dynamic path gated by `period_end is not None` and `command_name is not None`

`gait_phase()` uses accumulator only when both are set; else fixed `(t % period) / period`.

**Rationale:** High-speed config already sets `period_end`; flat/rough configs do not. Avoids silently applying HS defaults to flat/rough.

**Alternatives:** Always dynamic — breaks flat/rough contract; rejected. Separate functions for HS vs flat — larger API churn.

### Decision 3: Rewards import shared helpers

`rewards.py` imports `dynamic_gait_phase_fraction` and `gait_phase as gait_phase_obs`. `feet_gait` reads accumulator when dynamic params present, else fixed modulus. `cross_arm_swing_stance` does not recompute `current_period`; it calls `gait_phase_obs` with full params so standing mask and phase source match policy.

**Rationale:** Single source of truth for phase; arm-swing currently passes only `period`, which is the desync bug.

### Decision 4: HS config unchanged

Keep `high_speed_env_cfg` env-var injection of gait params. No new `LIMX_HS_*` names required for continuity.

## Risks / Trade-offs

- [Attribute pollution on env] → Private `_dynamic_gait_*` attrs; shape mismatch recreates buffer. Acceptable vs global state.
- [First call zeros phase until first real step advance] → First observation after creation may be standing sin=0,cos=1; subsequent steps accumulate. Same as reset semantics.
- [Period=0 or degenerate span] → `max(speed_end - speed_start, 1e-6)` in alpha; period itself taken from configured start/end (defaults positive).
- [Play/eval without dynamic params] → Fixed path; if play env inherits HS dynamic params, same code path applies automatically.

## Migration Plan

1. Implement shared helpers + rewire `gait_phase` / `feet_gait` / `cross_arm_swing_stance`.
2. Unit-test accumulator continuity and same-step idempotence.
3. Confirm flat/rough still use fixed period when `period_end` unset.
4. After apply, operator warm-starts HS continue-train from `model_900.pt` with documented `--init_checkpoint` command (not run in propose).
5. Rollback: revert the three mdp files; curriculum env vars remain valid.

## Open Questions

None blocking. Operator may choose a new `run_name` suffix for the post-apply fine-tune; recorded in tasks as a documented command only.
