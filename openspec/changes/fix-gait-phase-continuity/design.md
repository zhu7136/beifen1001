# Design

## Context

See proposal.md. Current `observations.py` has `dynamic_gait_phase_fraction()` (accumulator on dynamic path only) and `gait_phase()` still uses `(episode_length_buf * step_dt) % period / period` when `period_end` or `command_name` is None. `feet_gait` has the same split. HS cfg always injects `period_end` default 0.60, so HS is dynamic; plain Velocity typically only sets `period`. Training resamples commands every ~10 s → period changes → modulus phase jumps. Specs define required behavior.

## Goals / Non-Goals

**Goals:**

- One shared `gait_phase_fraction()` used by obs + both rewards.
- Continuous accumulator under changing period/speed.
- `period_start`/`period_end=None` → fixed `period` (no accidental dynamic).
- Same-step single increment; reset zeros phase.

**Non-Goals:**

- Frontier curriculum mix / gates / reward redesign.
- Changing HS env-var schema beyond what is needed for explicit dynamic period.
- Rewriting historical training logs.

## Decisions

1. **API: `gait_phase_fraction(env, period, command_name=..., period_start=None, period_end=None, speed_start=2.0, speed_end=2.8)`**
   - `period_start = period if period_start is None else period_start`
   - `period_end = period if period_end is None else period_end`
   - Dynamic interpolation always available when start≠end; if start==end, current_period is constant but still uses accumulator (continuous).
   - Alternative: keep separate modulus fallback — rejected: causes train/eval divergence and jump on period change.

2. **Accumulator update once per sim step**
   - Prefer user-specified `common_step_counter` when present: advance only when it changes; else fall back to `episode_length_buf` delta for stubs/tests.
   - `phase = (phase + step_dt / current_period) % 1.0`
   - Reset when episode restarts (`episode_length_buf <= 1` or counter reset).

3. **Rewire three consumers**
   - `gait_phase`: always call `gait_phase_fraction` (no modulus branch).
   - `feet_gait`: same helper + existing offset packing.
   - `cross_arm_swing_stance`: same helper; `phase_signal = sin(2π * global_phase)`.
   - Keep standing/command-threshold masking in `gait_phase` observation output only (rewards already mask by command norm).

4. **HS cfg**
   - Keep explicit `period_start`/`period_end` injection; no default change required for HS.
   - Plain Velocity configs that only set `period` get fixed behavior via None defaults.

5. **Tests**
   - Unit (no Isaac): period change continuity; same-step re-entry; episode reset; None→fixed period; three-consumer shared phase via stubs.

## Risks / Trade-offs

- [Consumers still diverge if someone reintroduces modulus] → Spec forbids independent modulus; tests pin shared path.
- [common_step_counter missing on stub env] → Fallback to episode_length_buf delta.
- [Training mid-run phase semantics change] → New run after apply; warm-start weights OK; phase reward signal changes slightly under dynamic period.
- [HS period_end default 0.60 remains in env_cfg] → Intentional for HS; not a bug for plain Velocity.

## Migration Plan

1. Implement `gait_phase_fraction` + rewire three call sites.
2. Update unit tests; run package tests.
3. New HS train/eval runs use shared continuous phase; do not mix old checkpoint phase semantics with new code without a new run.
4. Rollback: revert commit; training path otherwise unchanged.

## Open Questions

None blocking.
