"""High-speed flat velocity env: rolling frontier curriculum + command mix, rewards unchanged."""

from __future__ import annotations

import os

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass

from limx_rl_lab.tasks.locomotion import mdp
from limx_rl_lab.tasks.locomotion.robots.limx.velocity_env_cfg import RobotEnvCfg


@configclass
class HighSpeedCommandsCfg:
    """Round-2 HS velocity commands: mix 50/35/15, rolling frontier gates."""

    base_velocity = mdp.HighSpeedUniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.08,
        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=False,
        ranges=mdp.HighSpeedUniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 2.0),
            lin_vel_y=(-0.05, 0.05),
            ang_vel_z=(-0.25, 0.25),
        ),
        limit_ranges=mdp.HighSpeedUniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 3.0),
            lin_vel_y=(-0.05, 0.05),
            ang_vel_z=(-0.25, 0.25),
        ),
        high_speed=mdp.HighSpeedCurriculumParams(
            target_speed_mps=2.4,
            initial_max_speed_mps=2.0,
            speed_increment_mps=0.1,
            rolling_window=200,
            consecutive_pass_required=200,
            dwell_min_iterations=300,
            completion_rate_min=0.97,
            orientation_term_rate_max=0.02,
            speed_rmse_max_mps=0.28,
            saturation_rate_max=0.10,
        ),
        near_frac=0.50,
        mid_frac=0.35,
        low_frac=0.15,
        low_stand_max=0.2,
        straight_line=True,
    )


@configclass
class HighSpeedCurriculumCfg:
    """Rolling frontier curriculum term (state surface; promote in command update)."""

    gated_speed_levels = CurrTerm(mdp.gated_speed_levels, params={"command_name": "base_velocity"})


@configclass
class RobotHighSpeedEnvCfg(RobotEnvCfg):
    """Round-2: start 2.0, target 2.4, increment 0.1; rewards/network unchanged."""

    commands: HighSpeedCommandsCfg = HighSpeedCommandsCfg()
    curriculum: HighSpeedCurriculumCfg = HighSpeedCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        hs = self.commands.base_velocity.high_speed
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, float(hs.initial_max_speed_mps))
        self.commands.base_velocity.ranges.lin_vel_y = (-0.05, 0.05)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.25, 0.25)
        self.commands.base_velocity.limit_ranges.lin_vel_x = (-0.5, 3.0)
        self.commands.base_velocity.limit_ranges.lin_vel_y = (-0.05, 0.05)
        self.commands.base_velocity.limit_ranges.ang_vel_z = (-0.25, 0.25)

        if "LIMX_HS_TARGET_SPEED" in os.environ:
            hs.target_speed_mps = float(os.environ["LIMX_HS_TARGET_SPEED"])
        if "LIMX_HS_INITIAL_MAX_SPEED" in os.environ:
            init = float(os.environ["LIMX_HS_INITIAL_MAX_SPEED"])
            hs.initial_max_speed_mps = init
            self.commands.base_velocity.ranges.lin_vel_x = (0.0, init)
        if "LIMX_HS_SPEED_INCREMENT" in os.environ:
            hs.speed_increment_mps = float(os.environ["LIMX_HS_SPEED_INCREMENT"])
        if "LIMX_HS_NEAR_FRAC" in os.environ:
            self.commands.base_velocity.near_frac = float(os.environ["LIMX_HS_NEAR_FRAC"])
        if "LIMX_HS_MID_FRAC" in os.environ:
            self.commands.base_velocity.mid_frac = float(os.environ["LIMX_HS_MID_FRAC"])
        if "LIMX_HS_LOW_FRAC" in os.environ:
            self.commands.base_velocity.low_frac = float(os.environ["LIMX_HS_LOW_FRAC"])


@configclass
class RobotHighSpeedPlayEnvCfg(RobotHighSpeedEnvCfg):
    """Play/eval: fixed command at initial max speed."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        max_speed = float(self.commands.base_velocity.high_speed.initial_max_speed_mps)
        self.commands.base_velocity.ranges.lin_vel_x = (max_speed, max_speed)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.rel_standing_envs = 0.0
