# Tasks

## 1. Shared phase helper

- [x] 1.1 Implement `gait_phase_fraction(env, period, command_name="base_velocity", period_start=None, period_end=None, speed_start=2.0, speed_end=2.8)` in `mdp/observations.py`: None→fall back to `period`; speed interpolate `current_period`; per-env accumulator `(phase + step_dt/current_period) % 1.0`; once per sim step (`common_step_counter` if available, else `episode_length_buf` delta); reset on episode restart; verify unit test period-change continuity and same-step re-entry

## 2. Rewire consumers

- [x] 2.1 `gait_phase()`: always call `gait_phase_fraction`; remove `(episode_length_buf * step_dt) % period / period`; keep sin/cos packing + standing mask; verify source has no modulus phase fallback
- [x] 2.2 `feet_gait()`: call `gait_phase_fraction` for all configs; keep offset/stance logic; verify no separate modulus branch remains
- [x] 2.3 `cross_arm_swing_stance()`: call `gait_phase_fraction` with full period/command params; `phase_signal = torch.sin(2.0 * torch.pi * global_phase)`; verify no tensor-as-period misuse

## 3. Defaults + tests

- [x] 3.1 Unit tests in `tests/test_continuous_gait_phase.py` (or new): (a) period change does not jump; (b) same-step multi-call single increment; (c) episode reset zeros phase; (d) `period_start/period_end=None` keeps fixed period; (e) three consumers share same phase on stub env; verify pytest passes without Isaac
- [x] 3.2 Confirm plain Velocity path with only `period=0.72` does not enable dynamic interpolation; HS cfg still injects explicit start/end; verify code review + optional cfg import check
- [x] 3.3 Run full `source/limx_rl_lab/tests/` suite; verify no failures related to gait phase
- [x] 3.4 `openspec validate --changes fix-gait-phase-continuity`; verify change validates and artifacts complete
