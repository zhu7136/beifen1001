# Spec Delta

## Purpose

Define continuous, shared gait-phase behavior for locomotion observation and rewards so phase does not jump when command speed or gait period changes, and so plain fixed-period configs are not accidentally treated as dynamic.

## ADDED Requirements

### Requirement: Continuous gait phase accumulator
The system SHALL maintain a per-env gait phase in `[0, 1)` that advances by `step_dt / current_period` on each real simulation step, not by re-evaluating `(episode_time % current_period) / current_period`. When `current_period` changes (speed-dependent period), phase SHALL continue from the last accumulator value rather than jump.

#### Scenario: Period change does not jump phase
- **WHEN** commanded speed (or configured period) changes between steps while dynamic period is enabled
- **THEN** phase continues as previous phase plus `step_dt / current_period` (modulo 1), not as a modulus of absolute time

#### Scenario: Episode reset zeros phase
- **WHEN** an env episode ends and a new episode starts
- **THEN** that env’s gait phase is reset to 0

### Requirement: Shared phase for observation and rewards
`gait_phase` observation, `feet_gait` reward, and `cross_arm_swing_stance` reward SHALL obtain gait phase from one shared helper and SHALL NOT each recompute a separate modulus phase on the same step.

#### Scenario: One helper, three consumers
- **WHEN** policy observation and both gait-related rewards run on the same simulation step
- **THEN** all three observe the same `global_phase` from the shared accumulator

#### Scenario: Arm swing uses shared phase
- **WHEN** `cross_arm_swing_stance` computes its phase signal
- **THEN** it uses `sin(2π · global_phase)` from the shared helper, not a separately passed tensor as a period

### Requirement: Fixed-period default for plain configs
The shared phase helper SHALL accept optional `period_start` / `period_end`. When either is `None`, it SHALL fall back to `period` (fixed). Dynamic period SHALL apply only when explicit start/end values differ from plain `period` (e.g. HS 0.72→0.60).

#### Scenario: Ordinary velocity config stays fixed
- **WHEN** a non-HS env config passes only `period=0.72` (no explicit dynamic start/end)
- **THEN** phase uses fixed period 0.72 and is not treated as speed-dependent dynamic period

#### Scenario: HS config enables dynamic period
- **WHEN** HS env cfg explicitly sets `period_start=0.72` and `period_end=0.60` with speed interpolation
- **THEN** `current_period` interpolates with command speed and phase remains continuous across period changes

### Requirement: Same-step single accumulation
Observation and reward calls on the same simulation step SHALL increment the phase accumulator at most once.

#### Scenario: Multiple consumers same step
- **WHEN** `gait_phase`, `feet_gait`, and `cross_arm_swing_stance` all call the shared helper in one step
- **THEN** the accumulator advances once for that step
