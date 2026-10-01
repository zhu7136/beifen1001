# Design

## Context

Round-1 HS run `logs/rsl_rl/limx_hu_d04_01_flat_velocity_hs/2026-10-01_20-46-27_hs-ft-001/` completed 3000 iterations with warm-start from flat `model_4999.pt`. Curriculum reached `current_max_speed_mps=2.0` and then `frozen=1`. Local checkpoints exist through `model_2999.pt`.

Current `gated_speed_levels` evaluates gates only when `common_step_counter % max_episode_length == 0` using mixed-command RMSE over all envs. That dilutes frontier error with low/mid commands and can promote on accidental episode snapshots.

See `proposal.md` for motivation. Main specs: `openspec/specs/velocity-curriculum/spec.md`, `openspec/specs/velocity-evaluation/spec.md`.

## Goals / Non-Goals

**Goals:**
- Rolling frontier gates that cannot promote on mixed-batch luck
- Round-2 staged 2.0→2.4 with increment 0.1 and mix 50/35/15
- Per-iteration W&B curriculum/frontier metrics
- Hold current speed on gate failure (no auto-degrade)
- Cleaner A/B vs `clykvseg` (only gate/mix/increment differ)

**Non-Goals:**
- Reward reweight, PPO redesign, network changes
- Jumping straight to 2.78 in this round
- Auto-degrade on failure
- BeyondMimic / rough terrain

## Decisions

### D1. Resume checkpoint order
**Decision:** `model_2999.pt` if present → else W&B `model_2900.pt`. Never `model_2400.pt` for round 2.

**Why:** Later checkpoints already adapted ~500 iters at 2.0 m/s; restart from 2.400 wastes that.

### D2. Frontier mask
**Decision:** `frontier_mask = vx_cmd >= current_max_speed_mps`.

**Why:** User specified frontier-only stats so overall RMSE is not diluted. With 50% near-cap sampling this yields enough samples at the cap.

**Alternative:** `>= 0.9 * current_max` — more samples, slightly looser frontier; not used unless sampling at exact cap is too sparse in practice.

### D3. Rolling window + dwell
**Decision:** Maintain deque/ring of per-iteration frontier stats over last 200 iterations. Promotion requires all gates pass on that rolling window for 200 consecutive iterations **and** ≥300 iterations since last promotion/speed change.

**Why:** Instant episode-boundary snapshots caused false promotions at 1.6–1.8 m/s.

### D4. Frontier gates (round 2)
**Decision:** Rolling frontier completion ≥ 0.97, orientation ≤ 0.02, RMSE ≤ 0.28, saturation ≤ 0.10 for **200 consecutive** iterations. Failure → freeze/hold current max; no auto-degrade.

**Why:** Slightly looser RMSE (0.28) during curriculum than fixed-speed acceptance (0.25) gives margin to learn the level; final acceptance stays strict.

### D5. Per-iteration logging
**Decision:** Curriculum term (or a lightweight metrics hook) updates/logging path every env step/iteration, not only at episode boundary. Keys listed in spec; all numeric (strings encoded as `gate_reason_code` bitmask).

**Why:** Operators need continuous frontier visibility; episode-boundary only is too coarse and hides mid-episode failures.

### D6. Round-2 isolation
**Decision:** Change only increment (0.1), gates (rolling frontier), mix (0.50/0.35/0.15), target (2.4), start speed (2.0). Keep LR 1e-4, noise 0.42, rewards, network.

**Why:** Direct comparison to `clykvseg` isolates gate-fix effectiveness.

## Risks / Trade-offs

- **Risk:** Frontier mask too strict → sparse samples → noisy rolling stats.  
  **Mitigation:** 50% near-cap mix; if still sparse, fall back to `>=0.9*max` (record decision in logs).
- **Risk:** Saturation proxy always high on early steps → permanent freeze.  
  **Mitigation:** saturation gate 10% on rolling window; use soft effort/vel limits; log vel vs torque saturation separately.
- **Risk:** 200-iter rolling window + 300 dwell delays promotion vs prior run.  
  **Mitigation:** intentional; 3000 iters still allow multiple 0.1 levels if gates pass.
- **Trade-off:** No auto-degrade means a bad promote freezes the round — operator must intervene or re-eval; preferred over silent collapse.

## Migration Plan

1. Implement rolling frontier stats + per-iteration metrics (code).
2. Set round-2 env/agent overrides: start 2.0, target 2.4, increment 0.1, mix 50/35/15.
3. Optional pre-eval: fixed 2.0 m/s on `model_2999.pt` to confirm start.
4. Launch HS train resume from `model_2999.pt` (or `model_2900.pt` fallback).
5. After freeze or reaching 2.4, run fixed-speed eval with strict gates (RMSE≤0.25, completion≥0.97).
6. If pass → next round target 2.78; if fail → diagnose without reward reweight first.
7. Rollback: keep `model_2999.pt` / prior run; stop round-2 if gate-fix causes instability.

## Open Questions

- Exact saturation definition in rolling stats: joint-vel only vs max(vel, torque). Plan: log both; gate uses max(vel, torque) rate ≤10% unless it always saturates, then gate vel-only and note it.
- Whether curriculum metrics hook lives in `gated_speed_levels` return path every step or a separate lightweight function called from env step. Prefer curriculum term called every iteration if manager allows; else a small hook on command term metrics.
