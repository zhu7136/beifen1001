# Spec Delta

## ADDED Requirements

### Requirement: From-scratch flat velocity training
The system SHALL support training `LimX-HU-D04-01-Flat-Velocity` from random policy weights for a fixed forward-speed target without warm-starting from a prior checkpoint.

#### Scenario: Launch without checkpoint
- **WHEN** an operator starts Flat velocity training for 2.8 m/s without `--init_checkpoint` and without `--resume`
- **THEN** actor and critic weights initialize fresh and training runs under the Flat task agent defaults (`BasePPORunnerCfg`: LR 1e-3, action noise 1.0, max_iterations configurable, default 8000)

#### Scenario: Do not use HS task for this path
- **WHEN** the training goal is the from-scratch 2.8 m/s Flat path
- **THEN** the launch uses `LimX-HU-D04-01-Flat-Velocity` and SHALL NOT require `LimX-HU-D04-01-Flat-Velocity-HS` or HS fine-tune agent defaults

### Requirement: Fixed command ceiling at 2.8 m/s
For the from-scratch 2.8 m/s Flat path, command forward velocity SHALL be sampled over a fixed range with maximum 2.8 m/s from the start of training, not via resume-based staged promotion.

#### Scenario: Command ranges at training start
- **WHEN** Flat velocity training starts for the 2.8 m/s goal
- **THEN** `commands.base_velocity.ranges.lin_vel_x` is `(0.0, 2.8)` and `limit_ranges.lin_vel_x` is also `(0.0, 2.8)`

#### Scenario: Reward-gated expansion is a no-op
- **WHEN** `ranges` equals `limit_ranges` on the Flat task
- **THEN** curriculum term `lin_vel_cmd_levels` does not expand command max speed beyond 2.8 m/s

### Requirement: Resume-based HS curriculum is not required for 2.8
The from-scratch 2.8 m/s Flat path SHALL NOT require resume from a final checkpoint, HS task registration, rolling-frontier promotion, or `target_speed_mps=2.78` staged curriculum.

#### Scenario: Abandoned pipeline not used
- **WHEN** operators follow the recommended 2.8 from-scratch path
- **THEN** training does not pass `--init_checkpoint` / `--resume`, does not export `LIMX_HS_*` env vars for this run, and success is judged by fixed-speed eval at 2.8 m/s

## MODIFIED Requirements

### Requirement: Target speed ceiling
For from-scratch Flat 2.8 m/s training, the system SHALL treat **2.8 m/s** as the command ceiling from training start; mixed-command train reward alone SHALL NOT be used as evidence that a policy holds 2.8 m/s.

#### Scenario: Stop at target speed
- **WHEN** command max is fixed at 2.8 m/s for the from-scratch Flat path
- **THEN** training does not require curriculum promotion beyond 2.8 m/s; quality is judged by fixed-speed eval at that speed

#### Scenario: Capability claims use 2.8 eval
- **WHEN** an operator asks whether a policy holds 2.8 m/s
- **THEN** the claim is based on fixed-speed evaluation at 2.8 m/s, not mixed train reward aggregates

### Requirement: Command sampling mix
For the from-scratch Flat 2.8 path, commands SHALL be sampled across the configured forward range with lateral velocity near zero and yaw constrained to approximately ±0.25 rad/s, rather than concentrating all mass at the cap or using a separate HS near/mid/low curriculum mix.

#### Scenario: Preserve low-speed control while learning high speed
- **WHEN** a training batch samples commands under fixed `lin_vel_x=(0.0, 2.8)`
- **THEN** sampled commands include low, mid, and near-cap forward speeds from the same uniform command term, not only 2.8 m/s

#### Scenario: Lateral and yaw bounds
- **WHEN** Flat training samples velocity commands for the 2.8 m/s goal
- **THEN** sampled `lin_vel_y` is within `(-0.05, 0.05)` and sampled `ang_vel_z` is within `(-0.25, 0.25)`

### Requirement: Straight-line high-speed command constraints
During from-scratch 2.8 m/s Flat training, lateral velocity commands SHALL be held near zero and yaw commands SHALL be constrained to approximately ±0.25 rad/s; complex turning/rough terrain is deferred.

#### Scenario: Early high-speed phase is straight-line
- **WHEN** the from-scratch Flat path trains high forward speed on flat terrain
- **THEN** sampled commands keep `|vy|` near 0 and `|wz|` within about 0.25 rad/s

#### Scenario: Add turning after linear stability
- **WHEN** fixed-speed eval shows stable linear tracking at 2.8 m/s
- **THEN** the operator may later relax yaw/lateral limits and consider terrain or turning curricula as a follow-up change

### Requirement: Fine-tune hyperparameter defaults
From-scratch Flat 2.8 m/s training SHALL default to learning rate 1e-3, action noise 1.0, checkpoint save every 100 iterations, and a training budget of at least 5000 iterations (recommended 8000), without mandatory HS fine-tune defaults (LR 1e-4, noise 0.42) or curriculum gates before evaluating.

#### Scenario: Conservative fine-tune configuration
- **WHEN** an operator launches the recommended from-scratch Flat 2.8 path
- **THEN** training uses `BasePPORunnerCfg` defaults unless CLI overrides are explicit, and does not require HS fine-tune LR 1e-4 / noise 0.42

#### Scenario: Early stop on curriculum freeze quality
- **WHEN** intermediate checkpoints are saved every 100 iterations and later eval fails at 2.8 m/s
- **THEN** operators may stop early and evaluate the best fixed-speed checkpoint at 2.8 m/s without waiting for a curriculum freeze that this path does not run

## REMOVED Requirements

### Requirement: Resume from final checkpoint
**Reason**: From-scratch 2.8 m/s Flat path abandons resume/warm-start training; prior checkpoint restore is not required or recommended for this goal.
**Migration**: Launch `LimX-HU-D04-01-Flat-Velocity` without `--init_checkpoint` / `--resume`; judge quality only with fixed-speed eval at 2.8 m/s.

### Requirement: Gated speed increments
**Reason**: Command max is fixed at 2.8 m/s from training start (`ranges == limit_ranges`); reward-gated staged promotion is not part of this path.
**Migration**: Use fixed command ranges in Flat `CommandsCfg`; do not rely on HS rolling-frontier or `lin_vel_cmd_levels` expansion toward 2.8.

### Requirement: Curriculum before reward redesign
**Reason**: This change does not run a resume-based speed curriculum, so curriculum-first fine-tune guidance does not apply.
**Migration**: For from-scratch 2.8, keep existing reward weights unless a later change explicitly redesigns rewards after fixed-speed eval fails for non-speed reasons.
