# Design

## Context

Flat task `LimX-HU-D04-01-Flat-Velocity` currently samples commands from `ranges.lin_vel_x=(-0.1, 0.1)` with `limit_ranges.lin_vel_x=(-0.4, 0.8)`. Curriculum `lin_vel_cmd_levels` expands max speed only when tracking reward exceeds 80% of the reward weight, clamped to `limit_ranges`. HS task `LimX-HU-D04-01-Flat-Velocity-HS` already supports high ceilings via env vars and rolling-frontier gates, but that path is the abandoned resume-based pipeline. See proposal.md Why.

## Goals / Non-Goals

**Goals:**

- From-scratch Flat training whose command contract is fixed max 2.8 m/s.
- Single authoritative fixed-speed eval protocol at 2.8 m/s.
- Launch/eval operator commands documented and reproducible.

**Non-Goals:**

- Resume / warm-start from any prior checkpoint.
- HS task curriculum, rolling-frontier promotion, or `LIMX_HS_*` env vars.
- Reward reweighting, PPO redesign, Rough terrain, sim2real export changes.
- Deleting or rewriting the HS task code in this change (not used by this path).

## Decisions

1. **Reuse Flat task + `BasePPORunnerCfg`, not a new task id.**  
   Minimal surface change; agent defaults already match from-scratch PPO.  
   Alternative: new gym id `...-Flat-Velocity-2p8` — rejected for now (extra registration without behavior gain).

2. **Set `ranges` and `limit_ranges` both to the 2.8 contract.**  
   Makes reward-gated expansion a no-op; avoids mid-run speed creep and HS mix logic.  
   Alternative: start `ranges=(0, 0.5)`, `limit=(0, 2.8)` — optional fallback if direct 2.8 is unstable; not the default path.

3. **Keep lateral/yaw tight for straight-line focus.**  
   `lin_vel_y=(-0.05, 0.05)`, `ang_vel_z=(-0.25, 0.25)` match HS straight-line intent without HS curriculum code.  
   `RobotPlayEnvCfg` currently assigns `ranges = limit_ranges`; after the edit both are already the same, so play/eval training-config consistency holds.

4. **Acceptance = fixed-speed eval at 2.8, not train reward.**  
   Gates already in `velocity_eval.SpeedGates.for_target` for ≥2.4: completion 0.97, orientation 0.02, RMSE 0.25. Spec delta centers the matrix on 2.8.

5. **Do not auto-archive or implement `rolling-frontier-speed-curriculum`.**  
   That change remains the abandoned resume pipeline; this change is a parallel path. Sync of main specs happens when this change is archived.

## Risks / Trade-offs

- [From-scratch may fail to hold 2.8] → Evaluate early checkpoints; if completion/RMSE fail badly, fall back to slower expansion (`ranges.lin_vel_x=(0.0, 0.5)` + same limits) without reusing HS task.
- [Training long before first usable checkpoint] → `save_interval=100`; eval intermediate `model_*.pt` before 8000 iterations finish.
- [Spec conflict with in-progress HS change] → Explicitly document non-use of resume/HS for this goal; archive path updates main specs separately.
- [Play task vs train command ranges] → After setting both range fields equal, train/play use the same command envelope.

## Migration Plan

1. Edit Flat `CommandsCfg` to the 2.8 contract (see tasks).
2. Launch new from-scratch run; do not point W&B/log names at old HS runs.
3. Run fixed-speed eval at 2.8 on a late checkpoint.
4. On archive of this change, main `velocity-curriculum` / `velocity-evaluation` specs absorb these deltas.

## Open Questions

None that block planning. Optional: exact `max_iterations` default (8000 vs 5000) is operator CLI; specs require ≥5000.
