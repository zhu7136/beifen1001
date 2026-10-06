# Spec Delta

## Purpose

Provide a durable fixed-speed command mode on the high-speed velocity command term so evaluation (or operator freeze) pins every env to one forward speed and cannot be re-mixed by curriculum, resets, or timed resampling.

## ADDED Requirements

### Requirement: Fixed-speed mode configuration
The velocity command term SHALL expose a nullable fixed forward-speed setting (`fixed_speed_mps`). When set, it is the single commanded forward speed in m/s for all environments; when unset, training mixed sampling remains in effect.

#### Scenario: Unset keeps mixed training sampling
- **WHEN** the command term is configured without `fixed_speed_mps`
- **THEN** curriculum mixed near/mid/low sampling and rolling-frontier updates continue as today

#### Scenario: Set enables fixed mode
- **WHEN** an operator or evaluator sets `fixed_speed_mps` to e.g. `2.4`
- **THEN** the command term treats fixed mode as active for resampling and curriculum writes

### Requirement: Resample pins command components
In fixed-speed mode, command resampling SHALL set every selected env to `vx = fixed_speed_mps`, `vy = 0`, `wz = 0`, and SHALL NOT sample mixed velocity commands or standing/heading envs for those envs.

#### Scenario: Pin on resample
- **WHEN** fixed mode is active at 2.4 m/s and commands are resampled for any env subset
- **THEN** those envs observe exactly `(2.4, 0, 0)` with no standing or heading flags

#### Scenario: Episode reset cannot re-mix
- **WHEN** an env episode ends and resample runs during evaluation with fixed mode on
- **THEN** the env returns to the pinned speed, not a mixed sample

### Requirement: Curriculum writes disabled in fixed mode
While fixed-speed mode is active, the command term SHALL NOT update rolling-frontier curriculum state or rewrite `ranges` from curriculum logic; evaluation must not be contaminated by training curriculum.

#### Scenario: Frontier metrics frozen
- **WHEN** fixed mode is active and `compute(dt)` runs after env stepping
- **THEN** curriculum range promotion/hold logic does not change the pinned command or ranges used for commands

#### Scenario: No silent mixed eval
- **WHEN** evaluation runs only through fixed mode without also setting mixed-sample fractions
- **THEN** no code path samples mid/low/stand commands for the evaluated envs
