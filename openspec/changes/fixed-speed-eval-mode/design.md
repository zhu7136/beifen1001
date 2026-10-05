# Design

## Context

`HighSpeedUniformVelocityCommand` always runs `sample_mixed_velocity_commands` in `_resample_command()` and always runs `_update_frontier_metrics()` in `compute()`, which rewrites `cfg.ranges.lin_vel_x` from curriculum state. `scripts/rsl_rl/eval_velocity.py` only sets degenerate ranges and calls `_resample_command` once — resets and later steps can re-mix commands. `velocity_eval.recommend_initial_max_speed` falls back to the lowest “completed” speed when gates fail, and JSON still shows `initial_max_speed_mps`. Foot-slide in eval averages `robot.data.body_lin_vel_w` over all bodies (≈ base speed). Gate checks in eval use strict `>`/`<` while curriculum gates already use `>=`/`<=`. See proposal.md for motivation; specs define the required behavior.

## Goals / Non-Goals

**Goals:**

- Command-term fixed mode that pins `(vx, vy, wz)` and blocks curriculum writes.
- Eval protocol that enables that mode, hard-fails on command drift, and records commanded vx stats.
- Inclusive pass gates and `highest_passing_speed_mps` (null when none pass).
- Correct foot-slide / power / mass reporting for acceptance JSON.

**Non-Goals:**

- Changing training mixed-sampling fractions or curriculum promotion thresholds.
- Reward redesign, new RL algorithm, or multi-agent eval.
- Rewriting historical `logs/eval/*.json` files.
- Broader robot asset API changes beyond reading foot body ids for slide metric.

## Decisions

1. **Config flag `fixed_speed_mps: float | None` on `HighSpeedUniformVelocityCommandCfg`**
   - Rationale: single explicit mode switch; `None` preserves training behavior.
   - Alternative: only set `near_frac=1.0` in eval — rejected: curriculum, resets, and timed resamples can still rewrite commands.
   - Alternative: separate command class — rejected: larger surface; existing cfg already shared by training/eval envs.

2. **Fixed handling at the top of `_resample_command` and early-return in `compute`**
   - Rationale: every resample path (reset, slice, full env) must pin; `compute` must not let frontier logic mutate ranges.
   - Alternative: mutate ranges only — insufficient; curriculum still updates `current_max_speed_mps` and y/z ranges.
   - Note: `getattr(self.cfg, "fixed_speed_mps", None)` for robustness on partial cfg mocks.

3. **Eval sets both `fixed_speed_mps` and degenerate ranges, then one full-env resample**
   - Rationale: ranges alone are not enough; fixed flag is the authority. Degenerate ranges keep any residual sampler consistent.
   - `rel_standing_envs = 0.0` plus fixed path clearing `is_standing_env` / `is_heading_env` when applicable.

4. **Hard fail on per-step `torch.allclose(vx_cmd, target, atol=1e-5, rtol=0)`**
   - Rationale: silent mixed JSON is worse than no report. Error message includes target/min/max/mean.
   - Alternative: warn and continue — rejected; acceptance must not look valid when contaminated.

5. **JSON: keep measured metrics; add `command_vx_min_mps/max/mean_mps`; rename report field to `highest_passing_speed_mps`**
   - Rationale: distinguish “policy failed to track target” from “evaluator never commanded target”. Breaking rename is intentional (see proposal).
   - Training cfg field `HighSpeedCurriculumParams.initial_max_speed_mps` stays; operator copies from `highest_passing_speed_mps`.

6. **Inclusive gates in `summarize_speed` / `recommend_initial_max_speed`**
   - Align with curriculum `evaluate_speed_gates` (`completion >= min`, `orient <= max`, `rmse <= max`, and eval already records saturation separately).
   - `recommend_initial_max_speed`: only `max(target)` over `gates_pass`; else `None`. Drop completion>0.5 fallback.

7. **Foot-slide / power / mass / CoT (precise)**
   - **Foot-slide** (eval loop): once at start, `foot_body_ids, _ = robot.find_bodies(FOOT_BODY_NAMES)` where `FOOT_BODY_NAMES = ["left_ankle_roll_link", "right_ankle_roll_link"]`; `contact_sensor = env.unwrapped.scene.sensors["contact_forces"]`; `foot_sensor_ids, _ = contact_sensor.find_bodies(FOOT_BODY_NAMES)`. Per step: `foot_vel_xy = torch.linalg.vector_norm(robot.data.body_lin_vel_w[:, foot_body_ids, :2], dim=-1)`; `contact = contact_sensor.data.current_contact_time[:, foot_sensor_ids] > 0.0`; `slide_sum += float((foot_vel_xy * contact).sum().item())`; `slide_samples += int(contact.sum().item())`. Final: `foot_slide = slide_sum / max(slide_samples, 1)` → pass as `foot_slide=foot_slide` into `compute_run_metrics`. Unit: contact-period horizontal foot speed, m/s.
   - Alternative rejected: all-body `body_lin_vel_w[:, :2]` (current) — not feet, not contact-gated, inflated by base (~1.94 at 2.4 m/s).
   - **Mass**: `masses = robot.root_physx_view.get_masses()`; `mass = float(masses.sum().item())` if `masses.ndim == 1` else `float(masses.sum(dim=-1).mean().item())` (per-env body-mass sum, then mean over envs). Do **not** use `.mean()` of raw bodies.
   - **Power**: `mechanical_power = (tau * omega).abs().sum(dim=-1).mean()`; `power_hist.append(float(mechanical_power.item()))`. Whole-robot power averaged over envs.
   - **Field rename**: `RunMetrics.mean_joint_power_w` → `mean_total_joint_power_w` (dataclass, `to_dict`, `summarize_speed`, `build_eval_report`, `_wandb_metrics`, unit tests). BREAKING for JSON consumers.
   - **CoT**: `cot = total_power_w / (robot_mass_kg * 9.81 * max(abs(mean_vx_mps), 1e-3))`. Document that post-fix CoT ≠ historical JSON CoT; do not compare across the break.
   - Record mass used for CoT in report extras when readable.

8. **Unit tests without Isaac**
   - Pure-Python tests for inclusive gates, null recommendation, report key rename, and (where possible) a lightweight stub/resample pin test for fixed mode if command term can be constructed without sim; otherwise test cfg field defaults + eval helper logic only. Full sim fixed-mode E2E is manual/operator-level.

## Risks / Trade-offs

- [Command still drifts after fix] → Per-step assert catches residual bugs; fail closed.
- [Callers break on `initial_max_speed_mps` removal] → Keep note in report + tasks update operator scripts/tests; training config field name unchanged.
- [Foot-slide body name mismatch across robots] → Use `robot.find_bodies(FOOT_BODY_NAMES)` + contact sensor ids; if either resolves empty, record null foot_slide rather than all-body mean.
- [CoT historical discontinuity] → Renamed power field + total mass/power change scale; operators must not compare old vs new CoT.
- [Inclusive boundary flips a prior “fail” to pass] → Intended alignment with curriculum; unit tests pin boundary cases.
- [Eval reuses one env across speeds] → Fixed flag + resample must be set per speed before each seed loop; residual state risk → set flag every `_eval_one_speed` entry (not only first call).

## Migration Plan

1. Land command-term fixed mode + unit tests (no behavior change when `fixed_speed_mps` is None).
2. Land eval enablement, hard fail, JSON fields, gate/report semantics; update unit tests.
3. Land contact-gated foot-slide, total mass, total power, `mean_total_joint_power_w` rename, CoT formula; note CoT break vs historical JSON.
4. Update any in-repo scripts/docs that read `initial_max_speed_mps` from eval JSON to `highest_passing_speed_mps`, and any readers of `mean_joint_power_w`.
5. Re-run fixed-speed eval on an existing checkpoint; verify command vx stats ≈ target, null recommendation when gates fail, foot_slide ≪ base speed when slip is modest, and CoT uses new definition only.
6. Rollback: revert commit(s); training path unchanged if flag defaults to None.

## Re-eval checklist (operator, after apply)

Requires Isaac Lab + a checkpoint; not run in unit CI.

```bash
python scripts/rsl_rl/eval_velocity.py \
  --task LimX-HU-D04-01-Flat-Velocity \
  --checkpoint <ckpt.pt> \
  --speeds 2.4 --num_seeds 3 --num_envs 3 \
  --headless --output logs/eval/fixed_speed_2p4.json
```

Verify JSON:

- `summary[].command_vx_min_mps` / `command_vx_max_mps` / `command_vx_mean_mps` all ≈ 2.4
- `highest_passing_speed_mps` is `null` when gates fail (no `initial_max_speed_mps`)
- `foot_slide_mean` ≪ base speed under modest slip (contact-gated feet)
- `mean_total_joint_power_w` present; CoT uses total mass + total power
- CoT values not compared to historical `logs/eval/*.json` CoT

## Open Questions

None blocking: implementation can proceed from specs + this design.
