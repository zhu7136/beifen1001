# Proposal

## Why

Training resamples velocity commands every ~10 s and `current_period` is speed-dependent, so a phase defined as `(episode_time % current_period) / current_period` jumps when period changes. Fixed-speed eval holds one speed, so training can look improved while transfer to fixed eval fails. Live `mdp/observations.py` still keeps a modulus fallback when `period_end`/`command_name` are unset, and `gait_phase` / `feet_gait` / `cross_arm_swing_stance` do not always share one phase path; ordinary configs that only pass `period` must not accidentally become dynamic.

## What Changes

- Add shared `gait_phase_fraction()` in `mdp/observations.py`: per-env phase accumulator advanced by `step_dt / current_period` once per sim step; reset on episode restart. **BREAKING** (behavior): consumers stop using `(t % period) / period` for phase.
- Defaults: `period_start`/`period_end` are `None`; if unset both fall back to `period` (fixed period). Dynamic period only when operator/HS cfg explicitly sets different start/end (e.g. 0.72→0.60).
- All three consumers call the same shared function — `observations.gait_phase`, `rewards.feet_gait`, `rewards.cross_arm_swing_stance` — and never recompute modulus phase independently. Arm-swing uses `sin(2π·global_phase)` from that shared phase.
- Dedup key: same-step re-entry (policy/critic/rewards) must not double-increment; episode reset zeros phase.
- **Non-goals**: Frontier curriculum fractions, reward weights, network, `LIMX_HS_*` curriculum gates, fixed-speed eval protocol.

## Capabilities

### New Capabilities

- `gait-phase-continuity`: durable gait-phase semantics — continuous accumulator under changing command/period, shared across observation and rewards, fixed-period default for plain configs.

### Modified Capabilities

- (none — main `openspec/specs` has only `velocity-curriculum` / `velocity-evaluation`; gait-phase capability is introduced by this change.)

## Impact

- Code: `source/limx_rl_lab/limx_rl_lab/tasks/locomotion/mdp/observations.py`, `.../mdp/rewards.py`; tests `tests/test_continuous_gait_phase.py` (and related).
- HS cfg already injects explicit `period_start`/`period_end`/`speed_*` — dynamic path remains intentional for HS; plain Velocity only `period=0.72` stays fixed.
- Checkpoint obs/reward structure unchanged (same phase tensor shape); training and play must share this code path.
- Related prior changes (`continuous-gait-phase`, `dynamic-gait-period`) remain historical; this change supersedes phase continuity behavior if applied after them.
