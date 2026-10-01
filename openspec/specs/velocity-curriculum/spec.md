# velocity-curriculum Specification

## Purpose

Define resume-based speed-curriculum training behavior for LimX flat velocity policies: continue from a final checkpoint, raise command max speed only when fixed-speed gates hold, sample commands to preserve low-speed control, and freeze curriculum when quality degrades.

## Requirements

### Requirement: Resume from final checkpoint
The system SHALL support resuming training from a final checkpoint (for example `model_4999.pt`) instead of always retraining from scratch when pursuing higher target speeds.

#### Scenario: Continue after flat training
- **WHEN** an operator starts high-speed curriculum training with resume checkpoint `model_4999.pt`
- **THEN** actor and critic weights are restored from that checkpoint and training continues without discarding prior learning

### Requirement: Gated speed increments
Command maximum forward speed SHALL start at the highest speed that passed fixed-speed evaluation (`initial_max_speed_mps`) and SHALL increase by `speed_increment_mps` (default 0.2 m/s) only after stability gates are met for ~100–200 consecutive iterations.

#### Scenario: Promote curriculum after stable high-speed tracking
- **WHEN** at current max speed the policy meets completion > 95%, orientation termination rate < 2%, speed RMSE < 0.25 m/s, and no persistent joint velocity/torque saturation for 100–200 iterations
- **THEN** command max speed increases by 0.2 m/s toward `target_speed_mps` (e.g. 2.78 m/s)

#### Scenario: Freeze curriculum when gates fail
- **WHEN** any stability gate fails after a speed increase
- **THEN** the curriculum freezes at the current max speed and does not raise the limit further until gates recover

### Requirement: Target speed ceiling
Curriculum training SHALL treat `target_speed_mps` (default 2.78 m/s) as the ceiling for straight-line high-speed promotion and SHALL NOT require mixed-command train reward alone to justify promotion.

#### Scenario: Stop at target speed
- **WHEN** current command max reaches 2.78 m/s
- **THEN** the curriculum does not promote beyond `target_speed_mps`; quality is judged by fixed-speed eval at that speed

### Requirement: Command sampling mix
High-speed curriculum training SHALL sample commands so that approximately 40% are near the current speed cap, 40% are mid-range, and 20% are low speed or standing, rather than concentrating all mass at the cap.

#### Scenario: Preserve low-speed control while learning high speed
- **WHEN** curriculum max is e.g. 1.6 m/s
- **THEN** a training batch still includes mid-range and low/stand commands at the configured mix, not only 1.6 m/s

### Requirement: Straight-line high-speed command constraints
During the straight-line high-speed phase, lateral velocity commands SHALL be held near zero and yaw commands SHALL be constrained to approximately ±0.2–0.3 rad/s until high-speed linear stability is achieved; complex turning/rough terrain is deferred.

#### Scenario: Early high-speed phase is straight-line
- **WHEN** curriculum is learning high forward speed before stable linear tracking
- **THEN** sampled commands keep `|vy|` near 0 and `|wz|` within about 0.2–0.3 rad/s

#### Scenario: Add turning after linear stability
- **WHEN** fixed-speed eval shows stable linear high-speed tracking
- **THEN** the operator may later relax yaw/lateral limits and consider terrain or turning curricula as a follow-up change

### Requirement: Curriculum before reward redesign
The first fine-tune round SHALL prefer curriculum expansion while keeping network structure and most reward weights unchanged; reward redesign is secondary when speed remains gated by curriculum.

#### Scenario: First round does not reweight rewards
- **WHEN** resuming from final checkpoint for high-speed curriculum
- **THEN** the first round keeps existing reward terms and weights unless curriculum cannot progress for reasons outside speed limits

#### Scenario: Diagnose residual saturation later
- **WHEN** curriculum freezes even with lower incremental speed increases
- **THEN** operators inspect tracking saturation and penalty terms as a later step, not as the default first action

### Requirement: Fine-tune hyperparameter defaults
First-round high-speed fine-tuning SHALL default to learning rate ~1e-4, restored actor+critic, action noise ~0.40–0.45, checkpoint save every 100 iterations, and 2000–4000 iterations subject to curriculum gates.

#### Scenario: Conservative fine-tune configuration
- **WHEN** starting curriculum from final checkpoint
- **THEN** training uses the conservative fine-tune defaults above unless overridden

#### Scenario: Early stop on curriculum freeze quality
- **WHEN** curriculum has frozen for a long period at a speed below target with no gate recovery
- **THEN** training can be stopped early and evaluated with fixed-speed protocol at the frozen max speed
