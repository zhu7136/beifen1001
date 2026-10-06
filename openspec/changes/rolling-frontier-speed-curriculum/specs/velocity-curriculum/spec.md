# Spec Delta

## MODIFIED Requirements

### Requirement: Gated speed increments
Command maximum forward speed SHALL start at `initial_max_speed_mps` and SHALL increase by `speed_increment_mps` (round-2 default 0.1 m/s) only after rolling frontier gates hold for 200 consecutive iterations, after at least 300 iterations at the current speed. Promotion SHALL use frontier-only statistics (`vx_cmd >= current_max_speed_mps`), not mixed-batch means.

#### Scenario: Promote curriculum after stable high-speed tracking
- **WHEN** at current max speed the rolling frontier gates meet completion ≥ 97%, orientation termination rate ≤ 2%, frontier speed RMSE ≤ 0.28 m/s, saturation ≤ 10%, dwell ≥ 300 iterations, for 200 consecutive iterations
- **THEN** command max speed increases by `speed_increment_mps` toward `target_speed_mps` (round-2: 0.1 m/s steps to 2.4 m/s)

#### Scenario: Freeze curriculum when gates fail
- **WHEN** any rolling frontier gate fails after a speed increase
- **THEN** the curriculum holds the current max speed, does not auto-degrade, and does not promote further until gates recover

#### Scenario: Promote after rolling frontier gates hold
- **WHEN** rolling frontier statistics over the last 200 iterations satisfy all promotion gates and dwell time is met
- **THEN** the curriculum promotes by exactly `speed_increment_mps`

#### Scenario: Instant mixed-batch gates must not promote
- **WHEN** mixed-batch RMSE looks acceptable but rolling frontier RMSE remains above 0.28 m/s
- **THEN** the curriculum SHALL NOT promote the speed cap

#### Scenario: Minimum dwell before promotion
- **WHEN** frontier gates pass but the policy has been at the current speed for fewer than 300 iterations
- **THEN** the curriculum SHALL NOT promote yet

### Requirement: Target speed ceiling
Curriculum training SHALL treat `target_speed_mps` as the ceiling for the current training round. Round 2 SHALL use `target_speed_mps=2.4`; after fixed-speed eval passes at 2.4 m/s, a later round may raise the target toward 2.78 m/s. Mixed-command train reward alone SHALL NOT justify promotion.

#### Scenario: Stop at target speed
- **WHEN** current command max reaches the round target (2.4 m/s in round 2)
- **THEN** the curriculum does not promote beyond that ceiling; quality is judged by fixed-speed eval at that speed

#### Scenario: Round-2 stops at 2.4 m/s
- **WHEN** the curriculum reaches 2.4 m/s in round 2
- **THEN** it does not promote beyond 2.4 in this round

#### Scenario: Open next round only after 2.4 eval passes
- **WHEN** fixed-speed eval at 2.4 m/s meets stricter gates (RMSE ≤ 0.25 m/s, completion ≥ 97%)
- **THEN** the operator may start the next curriculum round toward 2.78 m/s

### Requirement: Command sampling mix
High-speed curriculum training SHALL sample commands with a configurable mix. Round-2 defaults SHALL be approximately 50% near the current speed cap, 35% mid-range, and 15% low speed or standing.

#### Scenario: Preserve low-speed control while learning high speed
- **WHEN** curriculum max is e.g. 2.0 m/s
- **THEN** a training batch still includes mid-range and low/stand commands at the configured mix, not only 2.0 m/s

#### Scenario: Round-2 sampling mix
- **WHEN** round-2 HS curriculum training samples commands at current max speed 2.0 m/s
- **THEN** the mix is approximately 50% near cap, 35% mid-range, 15% low/stand; `lin_vel_y` stays in [-0.05, 0.05] and `ang_vel_z` in [-0.25, 0.25]

### Requirement: Straight-line high-speed command constraints
During the straight-line high-speed phase, lateral velocity commands SHALL be held near zero and yaw commands SHALL be constrained to approximately ±0.25 rad/s (round-2 band [-0.25, 0.25]) until high-speed linear stability is achieved; complex turning/rough terrain is deferred.

#### Scenario: Early high-speed phase is straight-line
- **WHEN** round-2 curriculum is learning 2.0–2.4 m/s forward speed
- **THEN** sampled commands keep `|vy|` near 0 and `|wz|` within about 0.25 rad/s

#### Scenario: Add turning after linear stability
- **WHEN** fixed-speed eval shows stable linear high-speed tracking
- **THEN** the operator may later relax yaw/lateral limits and consider terrain or turning curricula as a follow-up change

### Requirement: Curriculum before reward redesign
Round-2 fine-tune SHALL change only curriculum mechanics (increment, rolling gates, sampling mix). Network structure, PPO hyperparameters, and reward weights SHALL remain unchanged so results compare directly to the prior round. Reward redesign remains secondary when speed remains gated by curriculum.

#### Scenario: First round does not reweight rewards
- **WHEN** resuming from final checkpoint for high-speed curriculum
- **THEN** the first round keeps existing reward terms and weights unless curriculum cannot progress for reasons outside speed limits

#### Scenario: Diagnose residual saturation later
- **WHEN** curriculum freezes even with lower incremental speed increases
- **THEN** operators inspect tracking saturation and penalty terms as a later step, not as the default first action

#### Scenario: Round-2 isolates gate-fix effects
- **WHEN** operators compare round-2 to the prior HS run (e.g. `clykvseg`)
- **THEN** the only intentional differences are increment 0.2→0.1, instant→rolling frontier gates, and near-fraction 40%→50%

### Requirement: Fine-tune hyperparameter defaults
First-round high-speed fine-tuning SHALL default to learning rate ~1e-4, restored actor+critic, action noise ~0.40–0.45, checkpoint save every 100 iterations, and max_iterations ~3000 subject to curriculum gates. Round-2 resume SHALL prefer `model_2999.pt` when present, else `model_2900.pt`; `model_2400.pt` SHALL NOT be used as the round-2 start.

#### Scenario: Conservative fine-tune configuration
- **WHEN** starting curriculum from final checkpoint
- **THEN** training uses the conservative fine-tune defaults above unless overridden

#### Scenario: Early stop on curriculum freeze quality
- **WHEN** curriculum has frozen for a long period at a speed below target with no gate recovery
- **THEN** training can be stopped early and evaluated with fixed-speed protocol at the frozen max speed

#### Scenario: Prefer late checkpoint
- **WHEN** `model_2999.pt` exists on the training machine
- **THEN** round-2 resumes from `model_2999.pt`

#### Scenario: Fallback checkpoint
- **WHEN** `model_2999.pt` is missing but `model_2900.pt` is available on W&B
- **THEN** round-2 resumes from `model_2900.pt` instead of `model_2400.pt`

## ADDED Requirements

### Requirement: Rolling frontier gate statistics
Curriculum gate statistics SHALL be computed on a rolling window over the most recent 200 iterations, using only frontier samples where `vx_cmd >= current_max_speed_mps`. Overall mixed-batch RMSE SHALL NOT be used as the promotion gate.

#### Scenario: Frontier-only rolling RMSE
- **WHEN** the training batch mixes low, mid, and near-cap commands
- **THEN** promotion RMSE is the rolling frontier RMSE over 200 iterations, not the all-command mean

### Requirement: Per-iteration frontier curriculum metrics
The training logger SHALL record the following every iteration (not only at episode boundaries): `Curriculum/current_max_speed_mps`, `Curriculum/frontier_speed_rmse_mps`, `Curriculum/frontier_completion_rate`, `Curriculum/frontier_orientation_rate`, `Curriculum/frontier_saturation_rate`, `Curriculum/consecutive_pass_iterations`, `Curriculum/gate_reason_code`. It SHALL also record commanded vx, measured vx, speed-bucketed RMSE, joint-velocity saturation rate, torque saturation rate, and action clipping ratio.

#### Scenario: Per-iteration W&B keys
- **WHEN** HS curriculum training runs with W&B logging
- **THEN** the listed `Curriculum/*` frontier keys and actuator/speed-bucket metrics appear every iteration

#### Scenario: Speed-bucketed RMSE
- **WHEN** aggregate RMSE is dominated by low-speed samples
- **THEN** the logger also provides RMSE broken down by commanded speed buckets so frontier capability remains visible
