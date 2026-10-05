from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def gait_phase_fraction(
    env: ManagerBasedRLEnv,
    period: float,
    command_name: str = "base_velocity",
    period_start: float | None = None,
    period_end: float | None = None,
    speed_start: float = 2.0,
    speed_end: float = 2.8,
) -> torch.Tensor:
    """Continuous per-env gait phase in [0, 1).

    ``period_start`` / ``period_end`` default to ``period`` (fixed). Dynamic
    period only when explicit values differ. Phase advances by
    ``step_dt / current_period`` once per simulation step; period changes do
    not jump phase.
    """
    period_start = period if period_start is None else period_start
    period_end = period if period_end is None else period_end

    command = env.command_manager.get_command(command_name)
    speed = command[:, 0].abs()

    alpha = torch.clamp(
        (speed - speed_start) / max(speed_end - speed_start, 1.0e-6),
        0.0,
        1.0,
    )
    current_period = period_start + alpha * (period_end - period_start)

    if not hasattr(env, "episode_length_buf"):
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    if (
        not hasattr(env, "_gait_phase_accumulator")
        or env._gait_phase_accumulator.shape != env.episode_length_buf.shape
        or env._gait_phase_accumulator.device != env.episode_length_buf.device
    ):
        env._gait_phase_accumulator = torch.zeros(
            env.num_envs, device=env.device, dtype=torch.float
        )
        env._gait_phase_last_step = -1
        env._gait_phase_last_steps = env.episode_length_buf.clone()

    step_id = -1
    use_common = hasattr(env, "common_step_counter")
    if use_common:
        try:
            step_id = int(env.common_step_counter)
        except Exception:  # noqa: BLE001
            step_id = -1

    if not use_common or step_id < 0:
        # stub / missing common_step_counter: episode_length_buf delta
        steps = env.episode_length_buf
        last = env._gait_phase_last_steps
        reset_mask = steps < last
        delta = torch.clamp(steps - last, min=0).float()
        env._gait_phase_accumulator = (env._gait_phase_accumulator + delta * env.step_dt / current_period) % 1.0
        env._gait_phase_accumulator[reset_mask | (steps == 0)] = 0.0
        env._gait_phase_last_steps = steps.clone()
        env._gait_phase_last_step = step_id
        return env._gait_phase_accumulator

    if env._gait_phase_last_step != step_id:
        reset_mask = env.episode_length_buf <= 1
        env._gait_phase_accumulator = (
            env._gait_phase_accumulator + env.step_dt / current_period
        ) % 1.0
        env._gait_phase_accumulator[reset_mask] = 0.0
        env._gait_phase_last_step = step_id
        env._gait_phase_last_steps = env.episode_length_buf.clone()

    return env._gait_phase_accumulator


def dynamic_gait_phase_fraction(
    env: ManagerBasedRLEnv,
    command_name: str,
    period_start: float,
    period_end: float,
    speed_start: float,
    speed_end: float,
) -> torch.Tensor:
    """Backward-compatible wrapper around :func:`gait_phase_fraction`."""
    return gait_phase_fraction(
        env,
        period=float(period_start),
        command_name=command_name,
        period_start=float(period_start),
        period_end=float(period_end),
        speed_start=float(speed_start),
        speed_end=float(speed_end),
    )


def gait_phase(
    env: ManagerBasedRLEnv,
    period: float,
    command_name: str | None = None,
    command_threshold: float = 0.1,
    period_start: float | None = None,
    period_end: float | None = None,
    speed_start: float = 2.0,
    speed_end: float = 2.8,
) -> torch.Tensor:
    """Gait phase observation via shared continuous accumulator."""
    if not hasattr(env, "episode_length_buf"):
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    global_phase = gait_phase_fraction(
        env,
        period=period,
        command_name=command_name or "base_velocity",
        period_start=period_start,
        period_end=period_end,
        speed_start=speed_start,
        speed_end=speed_end,
    )

    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)

    if command_name is not None:
        command_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        standing_mask = command_norm < command_threshold
        phase[standing_mask, 0] = 0.0
        phase[standing_mask, 1] = 1.0

    return phase
