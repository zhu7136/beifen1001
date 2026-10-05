# Spec Delta

## Purpose

Keep gait phase continuous and shared across policy, critic, and gait-related rewards when high-speed gait period depends on command speed, while preserving fixed-period behavior for flat/rough tasks.

## ADDED Requirements

### Requirement: Shared continuous gait phase accumulator
High-speed gait phase SHALL advance from a single per-environment accumulator that integrates `step_dt / current_period` on real env-step advances, not from per-call `(elapsed_time % period) / period`. The current period SHALL be linear-interpolated from command speed between configured start/end speed and start/end period, clamped outside the range.

#### Scenario: Period change does not jump phase
- **WHEN** command speed changes and the interpolated gait period changes within the same episode
- **THEN** the accumulator continues from the previous phase value with `delta_steps * step_dt / current_period`, and phase does not wrap due to a modulus of total elapsed time

#### Scenario: Episode reset zeros phase
- **WHEN** `episode_length_buf` decreases or an environment is at step 0
- **THEN** that environment's accumulator is set to 0.0

#### Scenario: Low speed clamps to start period
- **WHEN** command speed is below `speed_start`
- **THEN** current period equals `period_start`

#### Scenario: High speed clamps to end period
- **WHEN** command speed is above `speed_end`
- **THEN** current period equals `period_end`

### Requirement: Same-step consumers do not double-count phase
Policy observation, critic observation, and gait rewards SHALL observe the accumulator after at most one advance for a given simulation step. Repeated calls in the same step that see unchanged `episode_length_buf` SHALL not add more phase.

#### Scenario: Multiple consumers in one step
- **WHEN** policy, critic, feet-gait reward, and arm-swing reward are evaluated after the env has advanced one step
- **THEN** all four use the same phase value for that step; the accumulator advanced by at most `step_dt / current_period`

### Requirement: High-speed tasks share one phase source
When gait-period config includes a dynamic end period and a command name (high-speed velocity tasks), policy phase observation, critic phase observation, `feet_gait`, and `cross_arm_swing_stance` SHALL read the same accumulator rather than recomputing independent phase signals.

#### Scenario: Standing stills phase encoding
- **WHEN** gait phase is computed for high-speed tasks
- **THEN** near-zero commands still produce standing phase encoding (`sin=0`, `cos=1`) after the shared phase value is formed

#### Scenario: Arm-swing reward uses shared phase
- **WHEN** `cross_arm_swing_stance` is evaluated on a high-speed task
- **THEN** it obtains phase from the shared gait-phase path and does not recompute a separate period-dependent phase from total episode time

### Requirement: Flat and rough tasks keep fixed-period gait phase
When dynamic end period is not configured, gait phase observation and feet-gait reward SHALL use fixed `period` via `(elapsed_time % period) / period`. They SHALL NOT force high-speed dynamic interpolation defaults.

#### Scenario: Flat velocity observation
- **WHEN** `LimX-HU-D04-01-Flat-Velocity` (or rough) gait-phase observation runs with only fixed `period=0.72`
- **THEN** phase follows fixed period; no dynamic accumulator path is required

#### Scenario: Flat feet-gait reward
- **WHEN** flat/rough `feet_gait` runs without dynamic period params
- **THEN** stance phase uses fixed `period`, not high-speed interpolation defaults
