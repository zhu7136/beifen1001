# Proposal

## Why

Fixed-speed velocity evaluation is currently unreliable: `eval_velocity.py` only narrows `cfg.ranges`, but `HighSpeedUniformVelocityCommand` still runs mixed sampling + rolling-frontier curriculum, so episode resets and timed resamples can rewrite commands away from the target. Existing 2.4 m/s reports show this contamination (`mean_vx≈1.93` while `target=2.4`) and still emit `"initial_max_speed_mps": 2.4` when all gates fail — easy to misread as acceptance. Related metrics are also wrong: foot-slide uses `robot.data.body_lin_vel_w[:, :2]`, which is not the repo-defined feet (`FOOT_BODY_NAMES`) and is not gated on ground contact, so it tracks base speed (~1.94 at 2.4 m/s) instead of contact-period foot slip; robot mass is `get_masses().mean()` (average body mass, not total); power is per-joint mean `|(τ·ω)|`, not whole-robot mechanical power; eval gate comparisons are strict (`>`/`<`) unlike curriculum; nothing fails loudly on command drift.

## What Changes

- Add `fixed_speed_mps: float | None` to `HighSpeedUniformVelocityCommandCfg`. When set, `_resample_command()` pins every env to `(vx, vy, wz) = (fixed_speed, 0, 0)` and skips mixed sampling; `compute()` skips frontier/curriculum updates so ranges/state cannot re-write commands.
- Evaluation enables fixed mode (`cmd_term.cfg.fixed_speed_mps = target_speed_mps`), forces degenerate ranges, resamples once, then **raises** if any step's `command[:, 0]` is not the target within `atol=1e-5`.
- Eval JSON records `command_vx_min_mps` / `command_vx_max_mps` / `command_vx_mean_mps` (qualified 2.4 m/s runs must show all ≈ 2.4).
- Gate semantics become inclusive: `completion >= min`, `orientation <= max`, `rmse <= max` (align with curriculum gates in `velocity_curriculum.py`).
- `recommend_initial_max_speed` returns `None` when no speed passes; report field renamed to `highest_passing_speed_mps`. **BREAKING**: reports no longer carry a non-null `initial_max_speed_mps` on total failure.
- Fix foot-slide: resolve `FOOT_BODY_NAMES = ["left_ankle_roll_link", "right_ankle_roll_link"]` via `robot.find_bodies` + contact sensor ids; per step accumulate `sum(||body_lin_vel_w[:, foot_ids, :2]|| * (current_contact_time > 0))` and contact-sample counts; final `foot_slide = slide_sum / max(slide_samples, 1)` (m/s, contact-period horizontal slip).
- Fix mass: `masses = robot.root_physx_view.get_masses()`; if `masses.ndim == 1` use `masses.sum()`, else `masses.sum(dim=-1).mean()` (per-env body-mass sum, then mean over envs).
- Fix power + CoT: `mechanical_power = (tau * omega).abs().sum(dim=-1).mean()`; rename report field `mean_joint_power_w` → `mean_total_joint_power_w` (**BREAKING**); CoT = `total_power_w / (robot_mass_kg * 9.81 * max(abs(mean_vx_mps), 1e-3))`. Post-fix CoT is not comparable to historical JSON CoT.
- Unit tests updated for fixed-mode config defaults, inclusive gates, `None` recommendation, report field rename, and CoT/power/mass unit expectations.

## Capabilities

### New Capabilities

- `velocity-command-fixed-mode`: durable command-term behavior for evaluation (and optional operator freeze) via `fixed_speed_mps`: pin vx, zero vy/wz, bypass mixed sampling and rolling-frontier curriculum writes when fixed.

### Modified Capabilities

- `velocity-evaluation`: require true fixed-speed enforcement with hard fail on drift; command vx min/max/mean in JSON; inclusive gate comparisons; `highest_passing_speed_mps` with null-when-none; corrected foot-slide/power/mass metrics.
- `velocity-curriculum`: no training-sampling changes; document that `fixed_speed_mps` is the eval/freeze path that must not be conflated with mixed training sampling. Curriculum still starts from the highest speed that **passed** fixed-speed eval.

## Impact

- Code: `source/limx_rl_lab/limx_rl_lab/tasks/locomotion/mdp/commands/velocity_command.py`, `scripts/rsl_rl/eval_velocity.py`, `source/limx_rl_lab/limx_rl_lab/utils/velocity_eval.py`, `source/limx_rl_lab/tests/test_high_speed_velocity.py` (+ related unit tests).
- Consumers: any operator script or dashboard reading `initial_max_speed_mps` from eval JSON must switch to `highest_passing_speed_mps`; training `HighSpeedCurriculumParams.initial_max_speed_mps` config field is unchanged.
- Existing eval artifacts under `logs/eval/*.json` remain historical; new runs use the fixed protocol.
