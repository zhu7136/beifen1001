from __future__ import annotations

from dataclasses import MISSING

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

        fixed_speed = getattr(self.cfg, "fixed_speed_mps", None)
        if fixed_speed is not None:
            self.vel_command_b[env_ids, 0] = float(fixed_speed)
            self.vel_command_b[env_ids, 1] = 0.0
            self.vel_command_b[env_ids, 2] = 0.0
            self.is_standing_env[env_ids] = False
            if self.cfg.heading_command:
                self.is_heading_env[env_ids] = False
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
        self.vel_command_b[env_ids, 1] = cmd[:, 1]
        self.vel_command_b[env_ids, 2] = cmd[:, 2]
        if self.cfg.heading_command:
            self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
            self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

    def compute(self, dt: float):
        """Update commands + rolling frontier curriculum every step."""
        super().compute(dt)
        if getattr(self.cfg, "fixed_speed_mps", None) is not None:
            return
        self._update_frontier_metrics()

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

        vx_cmd = self.command[:, 0]
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
        apply_rolling_frontier_step(state, iter_stats)

        # apply held/promoted max to command ranges (no auto-degrade)
        new_max = min(state.current_max_speed_mps, state.target_speed_mps)
        self.cfg.ranges.lin_vel_x = (self.cfg.ranges.lin_vel_x[0], float(new_max))
        if hs is not None:
            self.cfg.ranges.lin_vel_y = (-0.05, 0.05)
            self.cfg.ranges.ang_vel_z = (-0.25, 0.25)

        # per-iteration numeric logs for rsl_rl / W&B
        hist = state.history
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
