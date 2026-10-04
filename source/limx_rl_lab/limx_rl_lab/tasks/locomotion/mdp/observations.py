from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def gait_phase(
    env: ManagerBasedRLEnv,
    period: float,
    command_name: str | None = None,
    command_threshold: float = 0.1,
    period_start: float = 0.72,
    period_end: float = 0.60,
    speed_start: float = 2.0,
    speed_end: float = 2.8,
) -> torch.Tensor:
    """Compute gait phase observation with optional dynamic period based on command speed."""
    if not hasattr(env, "episode_length_buf"):
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    # Compute current gait period based on command speed
    command = env.command_manager.get_command("base_velocity")
    command_speed = command[:, 0].abs()
    
    alpha = torch.clamp(
        (command_speed - speed_start) / max(speed_end - speed_start, 1.0e-6),
        0.0,
        1.0,
    )
    current_period = period_start + alpha * (period_end - period_start)
    
    # Use episode_length_buf * step_dt as phase accumulator
    global_phase = (env.episode_length_buf * env.step_dt) % current_period / current_period

    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)

    if command_name is not None:
        command_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        standing_mask = command_norm < command_threshold
        phase[standing_mask, 0] = 0.0
        phase[standing_mask, 1] = 1.0

    return phase
