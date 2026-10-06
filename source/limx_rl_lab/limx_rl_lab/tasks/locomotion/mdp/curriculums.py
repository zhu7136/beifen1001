from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_levels_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")

    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up

    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())


def lin_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.lin_vel_x = torch.clamp(
                torch.tensor(ranges.lin_vel_x, device=env.device) + delta_command,
                limit_ranges.lin_vel_x[0],
                limit_ranges.lin_vel_x[1],
            ).tolist()
            ranges.lin_vel_y = torch.clamp(
                torch.tensor(ranges.lin_vel_y, device=env.device) + delta_command,
                limit_ranges.lin_vel_y[0],
                limit_ranges.lin_vel_y[1],
            ).tolist()

    return torch.tensor(ranges.lin_vel_x[1], device=env.device)


def ang_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_ang_vel_z",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.ang_vel_z = torch.clamp(
                torch.tensor(ranges.ang_vel_z, device=env.device) + delta_command,
                limit_ranges.ang_vel_z[0],
                limit_ranges.ang_vel_z[1],
            ).tolist()

    return torch.tensor(ranges.ang_vel_z[1], device=env.device)


def gated_speed_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str = "base_velocity",
) -> dict:
    """Rolling-frontier high-speed curriculum (numeric state for logging).

    Promotion/hold is applied every step in HighSpeedUniformVelocityCommand.
    This term only surfaces numeric curriculum state on curriculum compute (reset).
    """
    from limx_rl_lab.tasks.locomotion.mdp.commands.velocity_command import get_or_init_hs_state

    command_term = env.command_manager.get_term(command_name)
    state = get_or_init_hs_state(command_term)

    return {
        "current_max_speed_mps": float(state.current_max_speed_mps),
        "target_speed_mps": float(state.target_speed_mps),
        "level": float(state.level),
        "frozen": float(state.frozen),
        "frontier_speed_rmse_mps": float(state.last_rmse_mps),
        "frontier_completion_rate": float(state.last_completion_rate),
        "frontier_orientation_rate": float(state.last_orientation_term_rate),
        "frontier_saturation_rate": float(state.last_saturation_rate),
        "consecutive_pass_iterations": float(state.consecutive_pass_iterations),
        "iterations_at_current_speed": float(state.iterations_at_current_speed),
        "gate_reason_code": float(_gate_reason_code(state.last_gate_reason)),
        "mixed_rmse_mps": float(state.last_mixed_rmse_mps),
        "commanded_vx_mps": float(state.last_commanded_vx_mps),
        "measured_vx_mps": float(state.last_measured_vx_mps),
        "joint_vel_saturation_rate": float(state.last_joint_vel_sat_rate),
        "joint_torque_saturation_rate": float(state.last_joint_torque_sat_rate),
        "action_clip_ratio": float(state.last_action_clip_ratio),
    }


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
