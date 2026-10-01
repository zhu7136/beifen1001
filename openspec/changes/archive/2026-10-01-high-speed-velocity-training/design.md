# Design

## Context

Flat-Velocity uses `UniformLevelVelocityCommandCfg` with:
- initial ranges ≈ ±0.1 m/s (x/y) and ±0.1 rad/s (yaw)
- `limit_ranges` for x ∈ [-0.4, 0.8], y ∈ [-0.2, 0.2], wz ∈ [-0.5, 0.5]
- curriculum `lin_vel_cmd_levels` expands **both** x and y by ±0.1 when mean `track_lin_vel_xy` reward exceeds 80% of its weight

Training logs report mixed-command aggregates only. They cannot show whether a checkpoint holds a fixed 2.78 m/s command. `limit_ranges.lin_vel_x[1] = 0.8` is also far below the 2.78 m/s target.

`train.py` already supports `--init_checkpoint` / `--resume`, `--max_iterations`, and rsl-rl overrides (`--learning_rate`, etc.). `play.py` loads `play_env_cfg` (full `limit_ranges`) but does not implement a multi-speed fixed-command metric suite.

See `proposal.md` for motivation; specs define the behavior contract.

## Goals / Non-Goals

**Goals:**
- Verifiable fixed-speed evaluation (speeds, seeds, metrics, gates)
- Resume-based curriculum that raises command max only when gates hold
- Command mix + straight-line constraints during high-speed learning
- First-round fine-tune keeps rewards/network mostly unchanged

**Non-Goals:**
- BeyondMimic / rough-terrain high-speed work in round 1
- Rewriting PPO, robot assets, or IsaacLab core
- Proving 2.78 m/s via mixed `Train/mean_reward` alone
- Hardware sim2real bring-up in this change

## Decisions

### D1. Fixed-speed eval as a dedicated play/eval entrypoint
**Decision:** Extend eval tooling under `scripts/` (new `eval_velocity.py` or `play.py --fixed_speed` mode) that runs the checkpoint on a fixed command grid `{1.0,1.4,1.8,2.2,2.5,2.78}` × N seeds and dumps metrics JSON + optional W&B panels.

**Why:** Training metrics remain mixed-curriculum aggregates; a separate gate avoids polluting train logs and makes pass/fail reusable by curriculum logic.

**Alternative:** Log fixed-speed probes inside every train iteration — rejected: too costly and blurs the success criterion.

### D2. Raise high-speed `limit_ranges` via env cfg override, not by rewriting base cfg
**Decision:** Introduce a high-speed env cfg subclass (or CLI/Hydra override) that sets `limit_ranges.lin_vel_x` up to ≥2.78 m/s (e.g. max 3.0) while keeping other axes conservative during straight-line phase.

**Why:** Existing flat cfg cannot even sample 2.78 m/s. Subclass/override is the project’s existing pattern (`RobotPlayEnvCfg`).

### D3. Replace dual-sided curriculum with gated max-speed curriculum
**Decision:** Add a new curriculum term (e.g. `gated_speed_levels`) that:
- reads gates from a metrics buffer (completion, orientation term rate, speed RMSE, saturation flags)
- promotes `ranges.lin_vel_x[1]` by `speed_increment_mps` after G stable episodes (~100–200 iters)
- clamps to `target_speed_mps`
- freezes on gate failure

**Why:** Current `lin_vel_cmd_levels` expands y as well and uses training reward fraction — not fixed-speed RMSE/completion — so it is the wrong signal for 2.78 m/s claims.

**Alternative:** Keep reward-based curriculum and only raise limits — rejected: same unverifiable promotion criterion.

### D4. Command sampling mix inside the command manager
**Decision:** Override command sampling for high-speed runs to allocate ~40% near cap, ~40% mid-range, ~20% low/stand; hold `vy≈0` and `|wz|≤0.25` during straight-line phase.

**Why:** Uniform sampling wastes the batch at the cap early on and forgets low-speed control. Straight-line constraint matches the staged plan.

**Alternative:** Post-process commands in an event — worse: resample timing and curriculum interplay get messy.

### D5. Fine-tune hyperparameters as CLI defaults for this change
**Decision:** Document and implement defaults: resume checkpoint required for this flow; `learning_rate=1e-4`; `init_noise_std=0.40–0.45`; `save_interval=100`; max iters 2000–4000; restore actor+critic via existing warm-start/resume paths.

**Note:** Base agent cfg has `init_noise_std=1.0`; actual mid-training noise should be read from the run’s `agent.yaml` / metrics. CLI override remains the source of truth for fine-tune noise.

### D6. Metrics schema for W&B
**Decision:** Emit eval metrics as structured keys under a fixed prefix (e.g. `eval/{speed}/mean_vx`, `eval/{speed}/rmse_vx`, `eval/{speed}/episode_completion`, termination rates, foot_slide, joint/torque stats, action_clip_ratio, power/COT). Curriculum promotion logs current max speed and gate history.

**Why:** Operators need per-speed panels; mixed reward remains secondary.

### D7. Curriculum-first, rewards last
**Decision:** Round 1 freezes reward weights unless curriculum cannot progress after gates and incremental increases are exhausted. Secondary diagnosis list (tracking saturation, torque/energy penalties, feet clearance, slide/orientation) is operational guidance, not a default code change.

**Why:** Existing reward stack already yields high completion and low orientation failure on flat; first lever is speed limit + sampling.

## Risks / Trade-offs

- **Risk:** Current `limit_ranges` max is 0.8 m/s — curriculum start may be 1.0–1.4 m/s after eval, far from 2.78 → many levels needed.  
  **Mitigation:** Cap total iterations; freeze early; report eval per level.
- **Risk:** High command max + aggressive noise may cause falls / torque spikes.  
  **Mitigation:** Saturation and termination gates; freeze on failure; start noise 0.40 not 1.0.
- **Risk:** Gate metrics need reliable episode bookkeeping in `ManagerBasedRLEnv`.  
  **Mitigation:** Prefer termination manager + measured root velocity over custom reward sums where possible; unit-test the metrics buffer.
- **Risk:** Eval throughput on single 4090.  
  **Mitigation:** Batch seeds in fewer envs; optional offline metric CSV; don’t block training on full matrix every iteration.
- **Risk:** Confusing train reward with capability claims.  
  **Mitigation:** Docs and dashboards label fixed-speed eval as the acceptance metric.
- **Trade-off:** Straight-line first delays turning/rough high-speed — intentional staged curriculum.

## Migration Plan

1. Add eval CLI + metrics JSON (no training behavior change).
2. Add high-speed env override + gated curriculum + sampling mix behind flags/task id or Hydra overrides.
3. Run fixed-speed eval on `model_4999.pt` to set `initial_max_speed_mps`.
4. Start resume curriculum run; log promotion history to W&B.
5. Re-eval matrix after each promoted level and at the end; compare RMSE/completion at 2.78 m/s.
6. Rollback: stop fine-tune; re-export policy from last good checkpoint; previous flat run remains available under `logs/rsl_rl/limx_hu_d04_01_flat_velocity/`.

## Open Questions

- Exact power/COT formula (average mechanical power vs contact-based COT) — can be chosen in eval implementation as long as it is consistent across checkpoints.
- Whether curriculum gates live in a new task id (e.g. `LimX-HU-D04-01-Flat-Velocity-HS`) or as Hydra overrides on the existing task — prefer separate task/override surface to avoid breaking the flat baseline run.
- Final `limit_ranges.lin_vel_x` upper bound (2.78 vs 3.0) — default target remains 2.78 m/s; sampling may allow a small margin above target for RMSE headroom.
