# Spec Delta

## ADDED Requirements

### Requirement: True fixed-speed enforcement
Evaluation SHALL enable fixed-speed command mode on the task's velocity command term, set degenerate ranges to the target speed, resample all envs once, and after every environment step verify that commanded forward velocity matches the target within `1e-5` m/s. On mismatch the evaluation SHALL raise an error rather than write a JSON report that looks valid.

#### Scenario: Enable fixed mode before stepping
- **WHEN** evaluation targets 2.4 m/s
- **THEN** the command term is in fixed-speed mode with `vx` pinned to 2.4, `vy=0`, `wz=0` before the first step

#### Scenario: Command drift fails the run
- **WHEN** any step's commanded `vx` is not within `1e-5` of the target (e.g. min 2.0 / max 2.5 after curriculum resample)
- **THEN** evaluation raises a runtime error naming target, min, max, and mean of commanded `vx`

### Requirement: Report commanded speed stats
Eval JSON SHALL record `command_vx_min_mps`, `command_vx_max_mps`, and `command_vx_mean_mps` per speed summary from all step/env commanded `vx` samples. A qualified 2.4 m/s evaluation SHALL have all three approximately equal to 2.4.

#### Scenario: Clean fixed-speed summary
- **WHEN** fixed mode holds 2.4 m/s for all seeds/steps
- **THEN** command vx min, max, and mean are all ≈ 2.4

#### Scenario: Contaminated summary visible
- **WHEN** commanded speeds are mixed despite a 2.4 m/s target
- **THEN** min/max/mean expose the spread instead of only reporting measured velocity error

### Requirement: Highest passing speed only
Eval reports SHALL expose `highest_passing_speed_mps`: the maximum target speed whose summary passes all configured gates, or `null` when no tested speed passes. Reports SHALL NOT invent a non-null acceptance speed from partial completion when gates fail.

#### Scenario: All gates fail
- **WHEN** every tested speed fails completion, orientation, and/or RMSE gates
- **THEN** `highest_passing_speed_mps` is `null` and the report does not claim e.g. 2.4 m/s as accepted

#### Scenario: One speed passes
- **WHEN** 2.2 m/s passes all gates but 2.4 m/s does not
- **THEN** `highest_passing_speed_mps` is 2.2

## MODIFIED Requirements

### Requirement: Fixed-speed evaluation matrix
The system SHALL evaluate a velocity policy at configurable target forward speeds with multiple random seeds per speed. Commands during evaluation SHALL be fixed-speed (not curriculum mixed sampling), independent of training-time command sampling.

#### Scenario: Evaluate checkpoint at multiple speeds
- **WHEN** an operator runs fixed-speed evaluation on a checkpoint for speeds `{1.0, 1.4, 1.8, 2.2, 2.5, 2.78}` m/s
- **THEN** the system produces one evaluation record per speed and seed, and does not mix those commands with curriculum training commands

#### Scenario: Multiple seeds per speed
- **WHEN** fixed-speed evaluation is configured with 5 seeds at a given target speed
- **THEN** the system runs 5 independent episodes (or episode sets) and reports seed-level and aggregate statistics under the same pinned command for each speed

#### Scenario: Target speed is commanded, not merely requested
- **WHEN** evaluation targets 2.4 m/s and the policy tracks poorly
- **THEN** reported metrics still reflect evaluation at commanded 2.4 m/s; low completion or high RMSE is a policy failure, not evidence that a different mixed speed was used

### Requirement: Per-run velocity metrics
For each fixed-speed run, the system SHALL record measured forward speed mean, forward speed RMSE against the target, and full-episode completion rate. Summary outputs SHALL also include commanded forward speed min/max/mean.

#### Scenario: Report speed accuracy
- **WHEN** a run targets 2.78 m/s and the measured mean forward speed is lower with high variance
- **THEN** the report includes mean forward speed and RMSE for that run so failure to hold the target is visible

#### Scenario: Report completion rate
- **WHEN** many episodes terminate early under fixed-speed commands
- **THEN** the evaluation report shows the proportion of episodes that ran to timeout without early termination

#### Scenario: Distinguish tracking error from command contamination
- **WHEN** RMSE is high but commanded min/max/mean equal the target
- **THEN** the report shows a tracking failure at fixed speed, not mixed-command sampling

### Requirement: Termination and motion-quality metrics
Fixed-speed evaluation SHALL record termination rates for bad orientation, base contact, and base height violations, plus foot-slide and joint/action saturation indicators. Foot-slide SHALL be the mean horizontal foot speed while feet are in ground contact (m/s), measured only on the repo-defined feet (`FOOT_BODY_NAMES`: `left_ankle_roll_link`, `right_ankle_roll_link`), not on arbitrary first bodies and not without contact gating.

#### Scenario: Termination breakdown
- **WHEN** evaluation completes for a target speed
- **THEN** the report includes rates for `bad_orientation`, base contact, and base height terminations

#### Scenario: Actuator health indicators
- **WHEN** evaluation completes
- **THEN** the report includes foot slide, joint velocity and torque levels, and action clipping ratio

#### Scenario: Contact-gated foot slide on real feet
- **WHEN** evaluation samples robot linear velocity and contact sensor contact time for `FOOT_BODY_NAMES`
- **THEN** each step contributes only feet with `current_contact_time > 0`, using `||body_lin_vel_w[:, foot_ids, :2]||`, and the reported `foot_slide` is `slide_sum / max(slide_samples, 1)` in m/s

#### Scenario: Foot slide not inflated by base speed
- **WHEN** the robot runs at 2.4 m/s forward with modest foot slip
- **THEN** reported foot-slide is on the order of actual contact-period foot slip, not ≈ base forward speed

### Requirement: Energy / effort metrics
Fixed-speed evaluation SHALL report whole-robot mean joint mechanical power and cost-of-transport. Power SHALL be the mean over envs of the sum over joints of `|τ·ω|`, published as `mean_total_joint_power_w`. Robot mass SHALL be the total articulation body mass (per-env sum of `get_masses()`, then mean over envs when masses are per-env), not the mean single-body mass. CoT SHALL be `total_power_w / (robot_mass_kg * 9.81 * max(abs(mean_vx_mps), 1e-3))`. Historical JSON CoT must not be compared directly with post-fix CoT.

#### Scenario: Compare effort at high speed
- **WHEN** two checkpoints are compared at 2.78 m/s under the fixed protocol
- **THEN** the evaluation report includes `mean_total_joint_power_w` and cost-of-transport metrics for each

#### Scenario: CoT uses total mass and total power
- **WHEN** robot body mass and joint τ/ω are readable at eval time
- **THEN** CoT uses summed robot mass and whole-robot mechanical power, not mean body mass and per-joint mean power

#### Scenario: CoT scale break is acknowledged
- **WHEN** operators compare a new eval JSON CoT to an older report
- **THEN** they treat the values as non-comparable after the power/mass fix

### Requirement: High-speed pass gates
Evaluation output SHALL support explicit pass/fail gates for a target speed using completion rate, orientation termination rate, and speed RMSE. Gate comparisons SHALL be inclusive of configured thresholds: pass when completion ≥ min, orientation rate ≤ max, and speed RMSE ≤ max.

#### Scenario: Gate evaluation for curriculum start
- **WHEN** evaluation is used to choose the curriculum start speed
- **THEN** each candidate speed has recorded pass/fail against the configured gates (default completion ≥ 95% / ≥ 97% at ≥2.4 m/s staged, orientation ≤ 2%, speed RMSE ≤ 0.25 m/s; inclusive)

#### Scenario: Exact threshold counts as pass
- **WHEN** completion equals the minimum threshold, or RMSE equals the maximum
- **THEN** that gate passes (boundary values are not rejected by strict inequality)
