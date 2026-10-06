from __future__ import annotations

from collections import deque
from dataclasses import MISSING, replace

import torch

from isaaclab.envs.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.utils import configclass

from limx_rl_lab.utils.velocity_curriculum import (
    CurriculumState,
    apply_rolling_frontier_step,
    compute_frontier_stats,
    sample_mixed_velocity_commands,
)


@configclass
class UniformLevelVelocityCommandCfg(UniformVelocityCommandCfg):
    limit_ranges: UniformVelocityCommandCfg.Ranges = MISSING


class HighSpeedUniformVelocityCommand(UniformVelocityCommand):
    """Velocity command with mix sampling + rolling frontier stats every step."""

    cfg: "HighSpeedUniformVelocityCommandCfg"

    def _resample_command(self, env_ids):
        if isinstance(env_ids, slice):
            env_ids = list(range(self.num_envs))
        n = len(env_ids)
        if n == 0:
            return

        if hasattr(self, "_hs_command_age_s"):
            age = self._hs_command_age_s
            # Eval steps run under inference_mode; out-of-band resample must not
            # in-place-update an inference tensor.
            if age.is_inference():
                age = age.detach().clone()
                self._hs_command_age_s = age
            age[env_ids] = 0.0

        fixed_speed = getattr(self.cfg, "fixed_speed_mps", None)
        if fixed_speed is not None:
            self.vel_command_b[env_ids, 0] = float(fixed_speed)
            self.vel_command_b[env_ids, 1] = 0.0
            self.vel_command_b[env_ids, 2] = 0.0
            self.is_standing_env[env_ids] = False
            if self.cfg.heading_command:
                self.is_heading_env[env_ids] = False
            if not hasattr(self, "_hs_target_vx"):
                self._hs_target_vx = torch.zeros(self.num_envs, device=self.device)
            self._hs_target_vx[env_ids] = float(fixed_speed)
            return

        ranges = self.cfg.ranges
        cmd = sample_mixed_velocity_commands(
            num_envs=n,
            lin_vel_x_max=float(ranges.lin_vel_x[1]),
            lin_vel_x_min=float(ranges.lin_vel_x[0]),
            lin_vel_y_range=(float(ranges.lin_vel_y[0]), float(ranges.lin_vel_y[1])),
            ang_vel_z_range=(float(ranges.ang_vel_z[0]), float(ranges.ang_vel_z[1])),
            near_frac=self.cfg.near_frac,
            mid_frac=self.cfg.mid_frac,
            low_frac=self.cfg.low_frac,
            low_stand_max=self.cfg.low_stand_max,
            device=self.device,
            generator=None,
        )
        r = torch.empty(n, device=self.device)
        self.vel_command_b[env_ids, 0] = cmd[:, 0]
        if not hasattr(self, "_hs_target_vx"):
            self._hs_target_vx = torch.zeros(
                self.num_envs, device=self.device
            )
        self._hs_target_vx[env_ids] = cmd[:, 0]
        self.vel_command_b[env_ids, 1] = cmd[:, 1]
        self.vel_command_b[env_ids, 2] = cmd[:, 2]
        if self.cfg.heading_command:
            self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
            self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

    def compute(self, dt: float):
        """Update commands + rolling frontier curriculum every step."""
        if not hasattr(self, "_hs_command_age_s"):
            self._hs_command_age_s = torch.zeros(
                self.num_envs, device=self.device
            )
        self._hs_command_age_s.add_(dt)
        super().compute(dt)
        if getattr(self.cfg, "fixed_speed_mps", None) is not None:
            return
        self._update_frontier_metrics()

    def _update_command(self):
        fixed_speed = getattr(self.cfg, "fixed_speed_mps", None)
        if fixed_speed is not None:
            # Fixed-speed eval: keep pinned command; parent only handles standing/heading.
            super()._update_command()
            return

        # Restore the sampled target before inherited standing/heading handling.
        if hasattr(self, "_hs_target_vx"):
            self.vel_command_b[:, 0] = self._hs_target_vx

        super()._update_command()

        # Effective final target: includes standing environments.
        self._hs_effective_target_vx = self.vel_command_b[:, 0].clone()

        accel = float(self.cfg.vx_accel_limit_mps2)
        if accel <= 0.0:
            return

        if not hasattr(self, "_hs_applied_vx"):
            self._hs_applied_vx = torch.zeros(
                self.num_envs, device=self.device
            )

        lengths = self._env.episode_length_buf

        # Only reset ramp state for environments that start a new episode.
        # Ordinary command resampling continues from the previous command.
        reset = lengths <= 1
        if hasattr(self, "_hs_ramp_last_episode_length"):
            reset = reset | (
                lengths < self._hs_ramp_last_episode_length
            )
        self._hs_applied_vx[reset] = 0.0

        max_delta = accel * float(self._env.step_dt)
        delta = (
            self._hs_effective_target_vx - self._hs_applied_vx
        ).clamp(min=-max_delta, max=max_delta)

        self._hs_applied_vx.add_(delta)
        self.vel_command_b[:, 0] = self._hs_applied_vx
        self._hs_ramp_last_episode_length = lengths.clone()

    def _update_frontier_metrics(self):
        env = self._env
        if env is None:
            return
        state = get_or_init_hs_state(self)
        robot = env.scene[self.cfg.asset_name]
        hs = getattr(self.cfg, "high_speed", None)

        tm = env.termination_manager
        term_idx = {n: i for i, n in enumerate(tm.active_terms)}
        last_dones = getattr(tm, "_last_episode_dones", None)
        if last_dones is not None and last_dones.numel() > 0:
            if "time_out" in term_idx:
                completion_rate = float(last_dones[:, term_idx["time_out"]].float().mean().item())
            else:
                completion_rate = 0.0
            if "bad_orientation" in term_idx:
                orient_rate = float(last_dones[:, term_idx["bad_orientation"]].float().mean().item())
            else:
                orient_rate = 0.0
            done_rows = last_dones.any(dim=1)
            if done_rows.numel() > 0 and completion_rate <= 0.0:
                non_to = ~last_dones[:, term_idx.get("time_out", 0)]
                early = float((done_rows & non_to).float().mean().item())
                completion_rate = 1.0 - early
        else:
            completion_rate = float((~tm.terminated).float().mean().item())
            orient_rate = (
                float(tm.get_term("bad_orientation").float().mean().item())
                if "bad_orientation" in term_idx
                else 0.0
            )

        jv = robot.data.joint_vel
        lim = robot.data.soft_joint_vel_limits
        sat_vel = float((jv.abs() > lim).any(dim=1).float().mean().item())
        at = robot.data.applied_torque
        if hasattr(robot.data, "soft_joint_effort_limits") and robot.data.soft_joint_effort_limits is not None:
            sat_tq = float((at.abs() > robot.data.soft_joint_effort_limits).any(dim=1).float().mean().item())
        else:
            sat_tq = 0.0
        # action clip proxy: actions beyond unit box
        try:
            act = env.action_manager.action
            clip_ratio = float((act.abs() > 1.0).float().mean().item())
        except Exception:  # noqa: BLE001
            clip_ratio = 0.0

        # Actual command seen by policy, rewards, and gait phase.
        vx_applied = self.command[:, 0]

        # Keep curriculum and existing diagnostics against the final target.
        vx_cmd = getattr(
            self, "_hs_effective_target_vx", vx_applied
        )
        vx_meas = robot.data.root_lin_vel_b[:, 0]
        frontier_min_frac = float(getattr(hs, "frontier_min_frac", 1.0)) if hs else 1.0
        iter_stats = compute_frontier_stats(
            vx_cmd=vx_cmd,
            vx_meas=vx_meas,
            completion_rate=completion_rate,
            orientation_rate=orient_rate,
            joint_vel_sat_rate=sat_vel,
            joint_torque_sat_rate=sat_tq,
            action_clip_ratio=clip_ratio,
            current_max_speed_mps=float(state.current_max_speed_mps),
            min_frac=frontier_min_frac,
        )
        # Capture the sampling frontier before a possible promotion.
        diag_frontier = vx_cmd >= float(state.current_max_speed_mps) * float(frontier_min_frac)

        # Preserve full-process statistics separately.
        if not hasattr(self, "_hs_full_history"):
            self._hs_full_history = deque(maxlen=state.rolling_window)
        self._hs_full_history.append(iter_stats)

        warmup_s = float(self.cfg.gate_warmup_s)
        gate_n = iter_stats.n_frontier
        gate_rmse = iter_stats.rmse_mps
        gate_sat = iter_stats.saturation_rate
        steady_vel_logs = {}

        if warmup_s <= 0.0:
            apply_rolling_frontier_step(state, iter_stats)
        else:
            episode_age_s = env.episode_length_buf * float(env.step_dt)
            steady = (
                diag_frontier
                & (episode_age_s >= warmup_s)
                & (self._hs_command_age_s >= warmup_s)
            )
            gate_n = int(steady.sum().item())

            if gate_n > 0:
                error = vx_cmd[steady] - vx_meas[steady]
                gate_rmse = float(error.square().mean().sqrt().item())

                # Same exceedance definition, but genuinely steady-frontier-only.
                tolerance_rel = float(
                    self.cfg.gate_vel_tolerance_rel
                )
                if tolerance_rel < 0.0:
                    raise ValueError(
                        "gate_vel_tolerance_rel must be non-negative"
                    )

                steady_ratio = (
                    jv[steady].abs() / lim[steady].clamp_min(1e-6)
                )

                strict_vel_sat = float(
                    (steady_ratio > 1.0)
                    .any(dim=1).float().mean().item()
                )
                vel_sat = float(
                    (steady_ratio > 1.0 + tolerance_rel)
                    .any(dim=1).float().mean().item()
                )

                steady_vel_logs[
                    "Diagnostics/gate/strict_vel_sat_rate"
                ] = strict_vel_sat
                steady_vel_logs[
                    "Diagnostics/gate/tolerant_vel_sat_rate"
                ] = vel_sat
                steady_vel_logs[
                    "Diagnostics/gate/vel_tolerance_rel"
                ] = tolerance_rel

                effort_lim = getattr(
                    robot.data, "soft_joint_effort_limits", None
                )
                if effort_lim is not None:
                    torque_sat = float(
                        (at[steady].abs() > effort_lim[steady])
                        .any(dim=1).float().mean().item()
                    )
                else:
                    torque_sat = 0.0

                gate_sat = max(vel_sat, torque_sat)

                gate_stats = replace(
                    iter_stats,
                    rmse_mps=gate_rmse,
                    saturation_rate=gate_sat,
                    joint_vel_sat_rate=vel_sat,
                    joint_torque_sat_rate=torque_sat,
                    commanded_vx_mps=float(vx_cmd[steady].mean().item()),
                    measured_vx_mps=float(vx_meas[steady].mean().item()),
                    mixed_rmse_mps=gate_rmse,
                    n_frontier=gate_n,
                )
                previous_max = float(state.current_max_speed_mps)
                apply_rolling_frontier_step(state, gate_stats)

                if float(state.current_max_speed_mps) != previous_max:
                    self._hs_full_history.clear()

                # Dimensionless speed ratio: abs(joint velocity) / limit.
                ratio = jv[steady].abs() / lim[steady].clamp_min(1e-6)

                thresholds = (1.00, 1.01, 1.05, 1.10)

                # Fraction of steady environments where ANY joint exceeds
                # each threshold. Comparable to the existing gate metric.
                any_rates = torch.stack([
                    (ratio > threshold).any(dim=1).float().mean()
                    for threshold in thresholds
                ]).detach().cpu().tolist()

                for threshold, value in zip(thresholds, any_rates):
                    suffix = f"{threshold:.2f}".replace(".", "p")
                    steady_vel_logs[
                        f"Diagnostics/steady_vel/any_gt_{suffix}"
                    ] = value

                focus_joints = (
                    "left_hip_pitch_joint",
                    "right_hip_pitch_joint",
                    "left_knee_joint",
                    "right_knee_joint",
                )

                for joint_name in focus_joints:
                    joint_id = robot.joint_names.index(joint_name)
                    joint_ratio = ratio[:, joint_id]

                    values = torch.stack([
                        *[
                            (joint_ratio > threshold).float().mean()
                            for threshold in thresholds
                        ],
                        torch.quantile(joint_ratio, 0.95),
                        torch.quantile(joint_ratio, 0.99),
                        joint_ratio.max(),
                    ]).detach().cpu().tolist()

                    prefix = f"Diagnostics/steady_vel/{joint_name}"

                    for index, threshold in enumerate(thresholds):
                        suffix = f"{threshold:.2f}".replace(".", "p")
                        steady_vel_logs[
                            f"{prefix}/gt_{suffix}"
                        ] = values[index]

                    steady_vel_logs[f"{prefix}/ratio_p95"] = values[4]
                    steady_vel_logs[f"{prefix}/ratio_p99"] = values[5]
                    steady_vel_logs[f"{prefix}/ratio_max"] = values[6]
            else:
                # Missing samples must not count as a passing evaluation.
                state.history.clear()
                state.consecutive_pass_iterations = 0
                state.last_gate_reason = "window_incomplete"
                state.frozen = True

        # apply held/promoted max to command ranges (no auto-degrade)
        new_max = min(state.current_max_speed_mps, state.target_speed_mps)
        self.cfg.ranges.lin_vel_x = (self.cfg.ranges.lin_vel_x[0], float(new_max))
        if hs is not None:
            self.cfg.ranges.lin_vel_y = (-0.05, 0.05)
            self.cfg.ranges.ang_vel_z = (-0.25, 0.25)

        # per-iteration numeric logs for rsl_rl / W&B
        # Existing frontier metrics remain full-process metrics.
        hist = self._hs_full_history
        if len(hist) > 0:
            frontier_rmse = sum(x.rmse_mps for x in hist) / len(hist)
            frontier_comp = sum(x.completion_rate for x in hist) / len(hist)
            frontier_ori = sum(x.orientation_rate for x in hist) / len(hist)
            frontier_sat = sum(x.saturation_rate for x in hist) / len(hist)
        else:
            frontier_rmse = iter_stats.rmse_mps
            frontier_comp = iter_stats.completion_rate
            frontier_ori = iter_stats.orientation_rate
            frontier_sat = iter_stats.saturation_rate

        log = env.extras.setdefault("log", {})
        log.update(steady_vel_logs)
        log["Diagnostics/gate/warmup_s"] = warmup_s
        log["Diagnostics/gate/steady_n"] = float(gate_n)

        # Keep sums for sample-weighted analysis.
        if gate_n > 0:
            log["Diagnostics/gate/rmse_instant_mps"] = gate_rmse
            log["Diagnostics/gate/saturation_rate"] = gate_sat

        if state.history:
            log["Diagnostics/gate/rolling_rmse_mps"] = (
                sum(x.rmse_mps for x in state.history)
                / len(state.history)
            )
            log["Diagnostics/gate/rolling_saturation_rate"] = (
                sum(x.saturation_rate for x in state.history)
                / len(state.history)
            )

        log["Diagnostics/ramp/accel_limit_mps2"] = float(
            self.cfg.vx_accel_limit_mps2
        )
        log["Diagnostics/ramp/active_rate"] = float(
            ((vx_cmd - vx_applied).abs() > 1e-4)
            .float().mean().item()
        )
        log["Diagnostics/ramp/applied_command_mean_mps"] = float(
            vx_applied.mean().item()
        )
        log["Diagnostics/ramp/applied_command_rmse_mps"] = float(
            (vx_applied - vx_meas).square().mean().sqrt().item()
        )
        with torch.no_grad():
            episode_age_s = env.episode_length_buf * float(env.step_dt)
            command_age_s = self._hs_command_age_s

            # Mutually exclusive groups inside the frontier.
            groups = {
                "reset_first2s": diag_frontier & (episode_age_s < 2.0),
                "resample_first2s": (
                    diag_frontier
                    & (episode_age_s >= 2.0)
                    & (command_age_s < 2.0)
                ),
                "steady_after2s": (
                    diag_frontier
                    & (episode_age_s >= 2.0)
                    & (command_age_s >= 2.0)
                ),
                "reset_0p0_to_0p5s": (
                    diag_frontier & (episode_age_s < 0.5)
                ),
                "reset_0p5_to_1p0s": (
                    diag_frontier
                    & (episode_age_s >= 0.5)
                    & (episode_age_s < 1.0)
                ),
                "reset_1p0_to_2p0s": (
                    diag_frontier
                    & (episode_age_s >= 1.0)
                    & (episode_age_s < 2.0)
                ),
            }

            error = vx_cmd - vx_meas

            for name, mask in groups.items():
                count = int(mask.sum().item())
                prefix = f"Diagnostics/phase/{name}"
                log[f"{prefix}/count"] = float(count)

                # Log sums as well as counts for weighted comparisons.
                # Empty groups contribute zero, not a fake zero RMSE.
                values = torch.stack([
                    error[mask].sum(),
                    error[mask].square().sum(),
                ]).detach().cpu().tolist()

                log[f"{prefix}/error_sum"] = values[0]
                log[f"{prefix}/squared_error_sum"] = values[1]

        # Diagnostic only: does not change rewards or promotion gates.
        with torch.no_grad():
            n_diag = int(diag_frontier.sum().item())
            log["Diagnostics/frontier_n"] = float(n_diag)

            if n_diag > 0:
                # Positive error = moving slower than commanded.
                error = vx_cmd[diag_frontier] - vx_meas[diag_frontier]
                bias = error.mean()
                error_std = error.std(unbiased=False)
                rmse = error.square().mean().sqrt()

                speed_stats = torch.stack([
                    bias,
                    error_std,
                    rmse,
                    (error > 0.5).float().mean(),
                ]).detach().cpu().tolist()

                log["Diagnostics/frontier_bias_mps"] = speed_stats[0]
                log["Diagnostics/frontier_error_std_mps"] = speed_stats[1]
                log["Diagnostics/frontier_rmse_instant_mps"] = speed_stats[2]
                log["Diagnostics/frontier_under_by_0p5_rate"] = speed_stats[3]

                # Keep original saturation metrics unchanged.
                # These new metrics really are frontier-only.
                vel_ratio = (
                    jv.abs() / lim.clamp_min(1e-6)
                )[diag_frontier]

                sat_by_joint = (vel_ratio > 1.0).float().mean(dim=0)
                near_by_joint = (vel_ratio > 0.9).float().mean(dim=0)

                joint_stats = torch.stack([
                    sat_by_joint, near_by_joint
                ]).detach().cpu().tolist()

                log["Diagnostics/frontier_any_joint_sat_rate"] = float(
                    (vel_ratio > 1.0).any(dim=1).float().mean().item()
                )

                for joint_id, joint_name in enumerate(robot.joint_names):
                    log[
                        f"Diagnostics/joint_sat/{joint_name}"
                    ] = joint_stats[0][joint_id]
                    log[
                        f"Diagnostics/joint_near_limit/{joint_name}"
                    ] = joint_stats[1][joint_id]
                    log[
                        f"Diagnostics/joint_vel_limit_rad_s/{joint_name}"
                    ] = float(lim[..., joint_id].mean().item())

        log["Curriculum/current_max_speed_mps"] = float(state.current_max_speed_mps)
        log["Curriculum/target_speed_mps"] = float(state.target_speed_mps)
        log["Curriculum/frontier_speed_rmse_mps"] = float(frontier_rmse)
        log["Curriculum/frontier_completion_rate"] = float(frontier_comp)
        log["Curriculum/frontier_orientation_rate"] = float(frontier_ori)
        log["Curriculum/frontier_saturation_rate"] = float(frontier_sat)
        log["Curriculum/consecutive_pass_iterations"] = float(state.consecutive_pass_iterations)
        log["Curriculum/gate_reason_code"] = float(_gate_reason_code(state.last_gate_reason))
        log["Curriculum/iterations_at_current_speed"] = float(state.iterations_at_current_speed)
        log["Curriculum/frozen"] = float(state.frozen)
        log["Metrics/commanded_vx_mps"] = float(iter_stats.commanded_vx_mps)
        log["Metrics/measured_vx_mps"] = float(iter_stats.measured_vx_mps)
        log["Metrics/joint_vel_saturation_rate"] = float(iter_stats.joint_vel_sat_rate)
        log["Metrics/joint_torque_saturation_rate"] = float(iter_stats.joint_torque_sat_rate)
        log["Metrics/action_clip_ratio"] = float(iter_stats.action_clip_ratio)
        log["Metrics/mixed_rmse_mps"] = float(iter_stats.mixed_rmse_mps)
        log["Metrics/frontier_n"] = float(iter_stats.n_frontier)
        # coarse speed buckets: 0.0,0.1,... as mean rmse for that bucket this step
        if state.speed_buckets:
            for bucket, vals in state.speed_buckets.items():
                if vals:
                    log[f"Metrics/speed_bucket_rmse/{bucket:.1f}"] = float(sum(vals) / len(vals))


def _gate_reason_code(reason: str) -> int:
    code = 0
    if not reason:
        return 0
    mapping = {
        "completion": 1,
        "orientation": 2,
        "rmse": 4,
        "saturation": 8,
        "window_incomplete": 16,
    }
    for part in str(reason).split(","):
        code |= mapping.get(part.strip(), 16)
    return code


@configclass
class HighSpeedCurriculumParams:
    """Static parameters for gated high-speed curriculum (round-2)."""

    target_speed_mps: float = 2.4
    initial_max_speed_mps: float = 2.0
    speed_increment_mps: float = 0.1
    rolling_window: int = 200
    consecutive_pass_required: int = 200
    dwell_min_iterations: int = 300
    completion_rate_min: float = 0.97
    orientation_term_rate_max: float = 0.02
    speed_rmse_max_mps: float = 0.28
    saturation_rate_max: float = 0.10
    frontier_min_frac: float = 1.0
    command_name: str = "base_velocity"
    # legacy fields kept for compatibility
    stable_min_steps: int = 300
    fail_streak_freeze: int = 2


@configclass
class HighSpeedUniformVelocityCommandCfg(UniformLevelVelocityCommandCfg):
    class_type: type = HighSpeedUniformVelocityCommand

    # 0 disables smoothing; positive values limit forward command acceleration.
    vx_accel_limit_mps2: float = 0.0

    # 0 keeps the original evaluation.
    gate_warmup_s: float = 0.0

    # Relative tolerance for velocity exceedance in the curriculum gate.
    # 0.0 retains strict comparison.
    gate_vel_tolerance_rel: float = 0.0

    # None: mixed sampling during training;
    # float: evaluation (or operator freeze) pins all envs to this speed.
    fixed_speed_mps: float | None = None

    high_speed: HighSpeedCurriculumParams = HighSpeedCurriculumParams()
    near_frac: float = 0.5
    mid_frac: float = 0.35
    low_frac: float = 0.15
    low_stand_max: float = 0.2
    straight_line: bool = True


def get_or_init_hs_state(command_term) -> CurriculumState:
    env = command_term._env
    if not hasattr(env, "_hs_curriculum_state"):
        hs = getattr(command_term.cfg, "high_speed", None) or HighSpeedCurriculumParams()
        ranges = command_term.cfg.ranges
        env._hs_curriculum_state = CurriculumState(
            current_max_speed_mps=float(ranges.lin_vel_x[1]),
            target_speed_mps=float(hs.target_speed_mps),
            speed_increment_mps=float(hs.speed_increment_mps),
            rolling_window=int(getattr(hs, "rolling_window", 200)),
            consecutive_pass_required=int(getattr(hs, "consecutive_pass_required", 200)),
            dwell_min_iterations=int(getattr(hs, "dwell_min_iterations", 300)),
            completion_rate_min=float(getattr(hs, "completion_rate_min", 0.97)),
            orientation_term_rate_max=float(getattr(hs, "orientation_term_rate_max", 0.02)),
            speed_rmse_max_mps=float(getattr(hs, "speed_rmse_max_mps", 0.28)),
            saturation_rate_max=float(getattr(hs, "saturation_rate_max", 0.10)),
            stable_min_steps=int(getattr(hs, "stable_min_steps", 300)),
            fail_streak_freeze=int(getattr(hs, "fail_streak_freeze", 2)),
        )
    return env._hs_curriculum_state
