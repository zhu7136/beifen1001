# Tasks

## 1. Fixed-speed evaluation

- [x] 1.1 Add fixed-speed eval entrypoint under `scripts/` (new script or `play.py` mode) supporting target speeds `{1.0,1.4,1.8,2.2,2.5,2.78}`, `--num_seeds` (default 5), and checkpoint path; verify CLI `--help` lists speeds/seeds/checkpoint
- [x] 1.2 Implement per-run metrics: forward-speed mean, RMSE vs target, full-episode completion rate; verify a dry-run/fixture produces a JSON record with those fields for one speed
- [x] 1.3 Add termination-rate metrics (`bad_orientation`, base contact, base height) and foot-slide; verify rates appear in the eval JSON
- [x] 1.4 Add joint velocity/torque saturation flags, action clipping ratio, and power/COT metric; verify fields present and non-NaN on a real checkpoint smoke run
- [x] 1.5 Aggregate seed-level records into a per-speed summary (mean±std, pass/fail vs gates: completion>95%, orientation term rate<2%, RMSE<0.25 m/s); verify summary marks gate pass/fail per speed
- [x] 1.6 Emit optional W&B metrics under `eval/{speed}/...` when `--logger wandb`; verify a short run creates the eval panels or logs the metric keys
- [x] 1.7 Document that mixed `Train/mean_reward` is not an acceptance metric in the eval script docstring/README snippet; verify doc states fixed-speed eval is authoritative for high-speed claims
- [x] 1.8 Run fixed-speed eval on an existing final checkpoint (e.g. flat `model_4999.pt` if present) and record `initial_max_speed_mps`; verify eval report file lists all configured speeds with metrics

## 2. High-speed curriculum training surface

- [x] 2.1 Add high-speed env cfg override (subclass or Hydra/CLI) raising `limit_ranges.lin_vel_x` to cover ≥2.78 m/s (default max 3.0) and conservative y/wz during straight-line phase; verify cfg dumps show new limits
- [x] 2.2 Implement gated speed curriculum term: promote `ranges.lin_vel_x[1]` by `0.2 m/s` only after ~100–200 iters with gates met; clamp to `target_speed_mps` (2.78); verify promotion and clamp in unit/synthetic test
- [x] 2.3 Implement curriculum freeze on gate failure (completion, orientation term rate, speed RMSE, saturation); verify current max speed stops increasing when a gate fails
- [x] 2.4 Replace/ignore dual-sided `lin_vel_cmd_levels` expansion for high-speed runs so y is not expanded with x; verify high-speed curriculum does not widen `lin_vel_y` on promotion
- [x] 2.5 Implement command sampling mix (~40% near cap, ~40% mid, ~20% low/stand) and straight-line constraints (`vy≈0`, `|wz|≤0.25` rad/s during high-speed phase); verify sampled command histogram and constraints
- [x] 2.6 Wire curriculum promotion metrics (current max speed, gate history, freeze reason) into W&B/train logs; verify logs show level index and promotion events
- [x] 2.7 Register or document task/override surface (e.g. high-speed task id or Hydra overrides) without breaking `LimX-HU-D04-01-Flat-Velocity`; verify baseline flat task still launches

## 3. Resume fine-tune configuration

- [x] 3.1 Document/start fine-tune defaults: resume/init checkpoint, `target_speed_mps=2.78`, `initial_max_speed_mps` from eval, `speed_increment_mps=0.2`, LR `1e-4`, action noise `0.40–0.45`, `save_interval=100`, iters `2000–4000`; verify run config dump contains these values
- [x] 3.2 Confirm actor+critic restore path works for high-speed fine-tune (existing `--init_checkpoint` / `--resume`); verify training log shows warm-start or resume load and first iteration runs
- [x] 3.3 Keep first-round reward weights/network structure unchanged by default; verify env cfg snapshot matches baseline reward weights unless operator overrides
- [x] 3.4 Start a curriculum fine-tune from evaluated checkpoint; verify checkpoints are saved every 100 iters under a distinct run name
- [x] 3.5 After freeze or target hit, re-run fixed-speed eval matrix; verify 2.78 m/s completion rate and RMSE are recorded and compared to gates
- [x] 3.6 If curriculum freezes below target, capture diagnostic notes (tracking error, penalties, saturation) without default reward reweighting; verify diagnostics file exists and lists suspected blockers
- [x] 3.7 Archive final checkpoint path and eval summary under the change/logs for the recommended training plan; verify a short operator-facing summary (speeds tested, best passing speed, RMSE, completion) is written

## 4. Integration verification

- [x] 4.1 Smoke: fixed-speed eval + high-speed curriculum launch end-to-end on a short iteration budget; verify no crash and artifacts (metrics JSON, cfg dump, at least one model checkpoint)
- [x] 4.2 Verify baseline flat training/play paths still work (launch or dry-run); verify `LimX-HU-D04-01-Flat-Velocity` remains available
- [x] 4.3 Validate OpenSpec change (`openspec validate --change high-speed-velocity-training`); verify no blocking errors before implementation sign-off
