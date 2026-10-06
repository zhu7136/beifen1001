# Spec Delta

## MODIFIED Requirements

### Requirement: Fixed-speed evaluation matrix
The system SHALL evaluate a velocity policy at configurable target forward speeds with multiple random seeds per speed, independent of curriculum command sampling used during training, including 2.8 m/s for the from-scratch Flat path.

#### Scenario: Evaluate checkpoint at multiple speeds
- **WHEN** an operator runs fixed-speed evaluation on a checkpoint for speeds including `{1.0, 1.5, 2.0, 2.4, 2.8}` m/s
- **THEN** the system produces one evaluation record per speed and seed, and does not mix those commands with curriculum training commands

#### Scenario: Multiple seeds per speed
- **WHEN** fixed-speed evaluation is configured with 5 seeds at a given target speed such as 2.8 m/s
- **THEN** the system runs 5 independent episodes (or episode sets) and reports seed-level and aggregate statistics

### Requirement: Per-run velocity metrics
For each fixed-speed run, the system SHALL record measured forward speed mean, forward speed RMSE against the target, and full-episode completion rate.

#### Scenario: Report speed accuracy
- **WHEN** a run targets a high speed such as 2.8 m/s and the measured mean forward speed is lower with high variance
- **THEN** the report includes mean forward speed and RMSE for that run so failure to hold the target is visible

#### Scenario: Report completion rate
- **WHEN** many episodes terminate early under fixed-speed commands
- **THEN** the evaluation report shows the proportion of episodes that ran to timeout without early termination

### Requirement: High-speed pass gates
Evaluation output SHALL support explicit pass/fail gates for a target speed, including completion rate, orientation termination rate, and speed RMSE thresholds. For target speeds ≥ 2.4 m/s, including 2.8 m/s, gates SHALL be completion rate ≥ 0.97, orientation termination rate < 0.02, and speed RMSE < 0.25 m/s.

#### Scenario: Gate evaluation for curriculum start
- **WHEN** evaluation is used to choose an initial training command ceiling or to judge high-speed readiness
- **THEN** each candidate speed has recorded pass/fail against the configured gates (completion > 95%, orientation terminations < 2%, speed RMSE < 0.25 m/s by default; stricter completion ≥ 0.97 when target ≥ 2.4 m/s)

#### Scenario: Gate evaluation at 2.8 m/s
- **WHEN** evaluation runs for target 2.8 m/s on the from-scratch Flat path
- **THEN** each seed/aggregate record is marked pass/fail against completion ≥ 0.97, orientation terminations < 0.02, and speed RMSE < 0.25 m/s

### Requirement: Evaluation does not prove mixed-training success
Reporting SHALL treat fixed-speed metrics as the authority for high-speed capability claims; mixed-command training aggregates SHALL NOT be used alone as evidence that a policy holds a high target speed.

#### Scenario: Claiming 2.78 m/s capability
- **WHEN** an operator asks whether a policy can stably run at 2.78 m/s
- **THEN** the system answers using fixed-speed evaluation results at 2.78 m/s, not only mixed curriculum train metrics

#### Scenario: Claiming 2.8 m/s capability
- **WHEN** an operator asks whether a policy can stably run at 2.8 m/s on the from-scratch Flat path
- **THEN** the system answers using fixed-speed evaluation results at 2.8 m/s, not only mixed train reward metrics
