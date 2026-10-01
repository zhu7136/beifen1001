# velocity-evaluation Specification

## Purpose

Define a fixed-speed evaluation protocol for LimX velocity policies so operators can verify whether a checkpoint holds a target forward speed (e.g. 2.78 m/s) with stable completion, acceptable terminations, and bounded motion quality — not only mixed-command aggregate training metrics.

## Requirements

### Requirement: Fixed-speed evaluation matrix
The system SHALL evaluate a velocity policy at configurable target forward speeds with multiple random seeds per speed, independent of curriculum command sampling used during training.

#### Scenario: Evaluate checkpoint at multiple speeds
- **WHEN** an operator runs fixed-speed evaluation on a checkpoint for speeds `{1.0, 1.4, 1.8, 2.2, 2.5, 2.78}` m/s
- **THEN** the system produces one evaluation record per speed and seed, and does not mix those commands with curriculum training commands

#### Scenario: Multiple seeds per speed
- **WHEN** fixed-speed evaluation is configured with 5 seeds at a given target speed
- **THEN** the system runs 5 independent episodes (or episode sets) and reports seed-level and aggregate statistics

### Requirement: Per-run velocity metrics
For each fixed-speed run, the system SHALL record measured forward speed mean, forward speed RMSE against the target, and full-episode completion rate.

#### Scenario: Report speed accuracy
- **WHEN** a run targets 2.78 m/s and the measured mean forward speed is lower with high variance
- **THEN** the report includes mean forward speed and RMSE for that run so failure to hold the target is visible

#### Scenario: Report completion rate
- **WHEN** many episodes terminate early under fixed-speed commands
- **THEN** the evaluation report shows the proportion of episodes that ran to timeout without early termination

### Requirement: Termination and motion-quality metrics
Fixed-speed evaluation SHALL record termination rates for bad orientation, base contact, and base height violations, plus foot-slide metrics and joint/action saturation indicators.

#### Scenario: Termination breakdown
- **WHEN** evaluation completes for a target speed
- **THEN** the report includes rates for `bad_orientation`, base contact, and base height terminations

#### Scenario: Actuator health indicators
- **WHEN** evaluation completes
- **THEN** the report includes foot slide, joint velocity and torque levels, and action clipping ratio

### Requirement: Energy / effort metrics
Fixed-speed evaluation SHALL report a power or cost-of-transport style metric so high-speed policies can be compared on effort, not only tracking error.

#### Scenario: Compare effort at high speed
- **WHEN** two checkpoints are compared at 2.78 m/s
- **THEN** the evaluation report includes power or cost-of-transport metrics for each

### Requirement: High-speed pass gates
Evaluation output SHALL support explicit pass/fail gates for a target speed, including completion rate, orientation termination rate, and speed RMSE thresholds used by the speed curriculum.

#### Scenario: Gate evaluation for curriculum start
- **WHEN** evaluation is used to choose `initial_max_speed_mps` for curriculum training
- **THEN** each candidate speed has recorded pass/fail against the configured gates (completion > 95%, orientation terminations < 2%, speed RMSE < 0.25 m/s by default)

### Requirement: Evaluation does not prove mixed-training success
Reporting SHALL treat fixed-speed metrics as the authority for high-speed capability claims; mixed-command training aggregates SHALL NOT be used alone as evidence that a policy holds a high target speed.

#### Scenario: Claiming 2.78 m/s capability
- **WHEN** an operator asks whether a policy can stably run at 2.78 m/s
- **THEN** the system answers using fixed-speed evaluation results at 2.78 m/s, not only mixed curriculum train metrics
