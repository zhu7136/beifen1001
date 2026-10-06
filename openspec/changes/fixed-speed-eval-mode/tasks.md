# Tasks

## 1. Command-term fixed-speed mode

- [x] 1.1 Add `fixed_speed_mps: float | None = None` to `HighSpeedUniformVelocityCommandCfg` in `velocity_command.py`; verify config import still succeeds and default remains None (mixed training path unchanged)
- [x] 1.2 In `_resample_command`, when `fixed_speed_mps is not None`, pin `vel_command_b[:,0]=fixed`, `[:,1]=0`, `[:,2]=0`, clear `is_standing_env` (and heading flags when `heading_command`), skip `sample_mixed_velocity_commands`; verify unit/stub test or code review shows fixed branch returns before mixed sampling
- [x] 1.3 In `compute()`, early-return before `_update_frontier_metrics()` when fixed mode active so curriculum cannot rewrite ranges; verify frontier log keys / range mutation do not occur in fixed mode
- [x] 1.4 Add/extend unit test for fixed-mode defaults and pin behavior (cfg field + resample helper/stub without Isaac); verify `pytest source/limx_rl_lab/tests/test_high_speed_velocity.py -k 'fixed or cfg or command'` (or nearest existing test module) passes

## 2. Eval hard fixed-speed protocol

- [x] 2.1 In `scripts/rsl_rl/eval_velocity.py` `_eval_one_speed`, set `cmd_term.cfg.fixed_speed_mps = float(target_speed_mps)`, degenerate ranges to target/(0,0), `rel_standing_envs=0.0`, resample all env ids; verify each speed/seed entry re-applies fixed mode before stepping
- [x] 2.2 After every `env.step()`, assert `torch.allclose(cmd_term.command[:,0], full_like(..., target), atol=1e-5, rtol=0)`; on failure raise `RuntimeError` with target/min/max/mean; verify intentional range-mixing or curriculum write trips the error (code path / dry review)
- [x] 2.3 Record `command_vx_min_mps`, `command_vx_max_mps`, `command_vx_mean_mps` from `vx_cmd_cat` into eval report (via `RunMetrics.extras` or `build_eval_report`); verify JSON keys present after helper unit test of report assembly

## 3. Gate semantics + acceptance field

- [x] 3.1 Change `summarize_speed` gates to inclusive: `completion >= min`, `orient <= max`, `rmse <= max`; verify boundary unit tests (e.g. completion=0.95, rmse=0.25) pass as `gates_pass` when other gates pass
- [x] 3.2 Change `recommend_initial_max_speed` to return `None` unless some summary has `gates_pass`; drop completion-only fallback; verify all-fail input returns None
- [x] 3.3 Rename eval report field to `highest_passing_speed_mps` (value from 3.2); stop emitting `initial_max_speed_mps` from `build_eval_report`; update `eval_velocity.py` print and any in-repo readers/tests that assert `initial_max_speed_mps`; verify failing all-speed run prints `highest_passing_speed_mps=None`
- [x] 3.4 Update `source/limx_rl_lab/tests/test_high_speed_velocity.py` (and any other unit tests) for inclusive gates, null recommendation, and report key rename; verify full package unit tests for `velocity_eval` / high-speed helpers pass without Isaac

## 4. Motion quality / effort metrics

- [x] 4.1 Fix foot-slide in `eval_velocity.py`: resolve `FOOT_BODY_NAMES=["left_ankle_roll_link","right_ankle_roll_link"]` once via `robot.find_bodies` and `scene.sensors["contact_forces"].find_bodies`; per step add `||body_lin_vel_w[:, foot_body_ids, :2]|| * (current_contact_time[:, foot_sensor_ids] > 0)` to `slide_sum` and contact count to `slide_samples`; set `foot_slide = slide_sum / max(slide_samples, 1)` and pass `foot_slide=foot_slide` into `compute_run_metrics`; verify synthetic case with base vel ≫ foot contact slip reports contact-gated value, not all-body mean
- [x] 4.2 Fix mass/power/CoT in `eval_velocity.py` + `velocity_eval.py`: mass = `get_masses()` 1D-sum else `sum(dim=-1).mean()`; power = `(tau*omega).abs().sum(dim=-1).mean()`; rename `mean_joint_power_w` → `mean_total_joint_power_w` through `RunMetrics`/`summarize_speed`/`build_eval_report`/`_wandb_metrics`; CoT = `total_power_w / (robot_mass_kg * 9.81 * max(abs(mean_vx_mps), 1e-3))`; verify unit test for sum-over-joints power, total mass, CoT formula, and JSON key `mean_total_joint_power_w`; note new CoT not comparable to old reports
- [x] 4.3 Re-run or document re-eval checklist: one checkpoint at 2.4 m/s fixed mode; verify JSON `command_vx_*` ≈ 2.4, `highest_passing_speed_mps` null when gates fail, foot_slide ≪ base speed under modest slip, `mean_total_joint_power_w` present, CoT uses new definition only

## 5. Integration verification

- [x] 5.1 Run full relevant unit suite (`source/limx_rl_lab/tests/` velocity/eval tests); verify no failures
- [x] 5.2 Confirm training env config still has `fixed_speed_mps` unset by default; verify curriculum mixed sampling path tests still pass
- [x] 5.3 Confirm no unit tests still assert old `mean_joint_power_w` key or `get_masses().mean()` mass; verify full suite green
- [x] 5.4 `openspec validate --changes fixed-speed-eval-mode` (or project equivalent); verify change validates and artifacts complete
