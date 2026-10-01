# Tasks

## 1. Resume checkpoint and pre-eval

- [x] 1.1 Prefer local `logs/rsl_rl/limx_hu_d04_01_flat_velocity_hs/2026-10-01_20-46-27_hs-ft-001/model_2999.pt`; document fallback to W&B `model_2900.pt`; forbid `model_2400.pt` for round-2 start; verify chosen path exists before train
- [x] 1.2 Run fixed-speed eval at 2.0 m/s on the chosen checkpoint (short matrix OK); verify report records mean_vx, RMSE, completion, terminations, and initial_max recommendation consistency with 2.0 m/s start

## 2. Rolling frontier gate implementation

- [x] 2.1 Implement frontier mask `vx_cmd >= current_max_speed_mps` and rolling stats over last 200 iterations in pure helpers + curriculum term; verify unit tests for promote/hold on synthetic rolling series
- [x] 2.2 Implement promotion only after ≥300 iterations at current speed and 200 consecutive passing rolling iterations; verify dwell blocks early promote
- [x] 2.3 Gate thresholds for curriculum: completion ≥ 0.97, orientation ≤ 0.02, RMSE ≤ 0.28, saturation ≤ 0.10; on any fail hold current max, no auto-degrade, no promote; verify freeze/hold in unit tests
- [x] 2.4 Stop using mixed-batch RMSE as promotion gate; verify logs show frontier RMSE not diluted all-command mean

## 3. Round-2 curriculum config

- [x] 3.1 Configure HS env/agent for round-2: `initial_max_speed_mps=2.0`, `target_speed_mps=2.4`, `speed_increment_mps=0.1`, `max_iterations=3000`, `save_interval=100`; verify env/agent yaml dump
- [x] 3.2 Configure command mix near=0.50, mid=0.35, low=0.15; `lin_vel_y=[-0.05,0.05]`, `ang_vel_z=[-0.25,0.25]`; verify sampled command histogram and ranges
- [x] 3.3 Keep rewards/PPO/network unchanged vs round-1 (LR 1e-4, noise 0.42); verify cfg snapshot matches baseline reward weights
- [x] 3.4 Document staged order 2.0→2.1→…→2.4 and that 2.78 is a later round only after 2.4 strict eval passes

## 4. Per-iteration metrics logging

- [x] 4.1 Log every iteration: `Curriculum/current_max_speed_mps`, `frontier_speed_rmse_mps`, `frontier_completion_rate`, `frontier_orientation_rate`, `frontier_saturation_rate`, `consecutive_pass_iterations`, `gate_reason_code`; verify keys appear each iteration in train log/W&B
- [x] 4.2 Log commanded_vx_mps, measured_vx_mps, speed-bucketed RMSE, joint-vel saturation rate, torque saturation rate, action clipping ratio each iteration; verify keys present and numeric
- [x] 4.3 Ensure logging is numeric-only for rsl_rl (encode reasons as codes); verify no `too many dimensions 'str'` crash

## 5. Strict fixed-speed acceptance at 2.4

- [x] 5.1 Implement/use stricter eval gates for staged high-speed acceptance: completion ≥ 97%, RMSE ≤ 0.25 m/s; verify summary pass/fail uses these thresholds when target ≥ 2.4
- [ ] 5.2 After round-2 freeze or reaching 2.4, re-run fixed-speed eval matrix on the HS checkpoint; verify 2.4 (and lower) completion/RMSE recorded against strict gates
- [x] 5.3 If 2.4 strict eval fails, write diagnostics without default reward reweight; verify diagnostics file lists frontier gate failures vs fixed-speed gaps

## 6. Launch and integration

- [x] 6.1 Launch round-2 HS train resume from `model_2999.pt` (or fallback `model_2900.pt`) with round-2 overrides in tmux/W&B; verify warm-start load, checkpoint save interval 100, and curriculum metrics streaming
- [x] 6.2 Smoke: rolling gate unit tests + short HS launch produce artifacts (cfg dump, metrics keys, at least model_0.pt) without crash
- [x] 6.3 Validate OpenSpec change `rolling-frontier-speed-curriculum`; verify no blocking validation errors
