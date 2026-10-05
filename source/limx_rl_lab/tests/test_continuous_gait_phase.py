"""Unit tests for shared continuous gait phase (no Isaac required)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

OBS_PATH = Path(__file__).resolve().parents[1] / "limx_rl_lab/tasks/locomotion/mdp/observations.py"


def _load_observations():
    spec = importlib.util.spec_from_file_location("limx_gait_observations", OBS_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


class _CommandManager:
    def __init__(self, cmd: torch.Tensor):
        self._cmd = cmd

    def get_command(self, name: str) -> torch.Tensor:
        return self._cmd


class _Env:
    def __init__(self, num_envs: int = 4, cmd_speed: float = 2.4, step_dt: float = 0.02):
        self.num_envs = num_envs
        self.device = torch.device("cpu")
        self.step_dt = step_dt
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.command_manager = _CommandManager(
            torch.tensor([[cmd_speed, 0.0, 0.0]], dtype=torch.float, device=self.device).repeat(num_envs, 1)
        )


def test_period_change_does_not_jump_phase():
    obs = _load_observations()
    env = _Env(cmd_speed=2.0)
    env.episode_length_buf += 1
    p1 = obs.gait_phase_fraction(env, period=0.72, period_start=0.72, period_end=0.60).clone()
    p1b = obs.gait_phase_fraction(env, period=0.72, period_start=0.72, period_end=0.60)
    assert torch.allclose(p1, p1b)
    env.command_manager._cmd[:, 0] = 2.8
    env.episode_length_buf += 1
    p2 = obs.gait_phase_fraction(env, period=0.72, period_start=0.72, period_end=0.60)
    expected = (p1 + (env.step_dt / 0.60)) % 1.0
    assert torch.allclose(p2, expected, atol=1e-6)
    assert torch.all((p2 >= 0) & (p2 < 1))


def test_same_step_double_call_no_double_increment():
    obs = _load_observations()
    env = _Env(cmd_speed=2.4)
    env.episode_length_buf += 1
    a = obs.gait_phase_fraction(env, period=0.72, period_start=0.72, period_end=0.60)
    b = obs.gait_phase_fraction(env, period=0.72, period_start=0.72, period_end=0.60)
    c = obs.gait_phase_fraction(env, period=0.72, period_start=0.72, period_end=0.60)
    assert torch.allclose(a, b)
    assert torch.allclose(b, c)


def test_episode_reset_zeros_phase():
    obs = _load_observations()
    env = _Env(cmd_speed=2.4)
    env.episode_length_buf += 5
    p = obs.gait_phase_fraction(env, period=0.72)
    assert torch.all(p >= 0)
    env.episode_length_buf[:] = 0
    p2 = obs.gait_phase_fraction(env, period=0.72)
    assert torch.allclose(p2, torch.zeros_like(p2))


def test_none_period_defaults_keep_fixed_period():
    obs = _load_observations()
    env = _Env(cmd_speed=2.4)
    env.episode_length_buf += 3
    # period_start/end None → current_period = period (0.72), no dynamic interp
    p1 = obs.gait_phase_fraction(env, period=0.72)
    env.episode_length_buf += 1
    p2 = obs.gait_phase_fraction(env, period=0.72)
    expected = (p1 + env.step_dt / 0.72) % 1.0
    assert torch.allclose(p2, expected, atol=1e-6)

    # gait_phase observation uses same helper (not modulus of absolute time)
    env2 = _Env(cmd_speed=2.4)
    env2.episode_length_buf += 3
    phase = obs.gait_phase(env2, period=0.72, command_name="base_velocity")
    assert hasattr(env2, "_gait_phase_accumulator")
    frac = env2._gait_phase_accumulator
    assert torch.allclose(phase[:, 0], torch.sin(frac * torch.pi * 2.0), atol=1e-6)


def test_three_consumers_share_phase():
    """gait_phase, and helper used by feet_gait / cross_arm — same accumulator value."""
    obs = _load_observations()
    env = _Env(cmd_speed=2.4)
    env.episode_length_buf += 1
    phase_obs = obs.gait_phase(
        env,
        period=0.72,
        command_name="base_velocity",
        period_start=0.72,
        period_end=0.60,
    )
    # second/third consumers re-enter same step
    phase_again = obs.gait_phase(
        env,
        period=0.72,
        command_name="base_velocity",
        period_start=0.72,
        period_end=0.60,
    )
    assert torch.allclose(phase_obs, phase_again)
    frac = obs.gait_phase_fraction(env, period=0.72, period_start=0.72, period_end=0.60)
    assert torch.allclose(phase_obs[:, 0], torch.sin(frac * torch.pi * 2.0), atol=1e-6)

    # rewards call sites import gait_phase_fraction (source check)
    rewards_src = (OBS_PATH.parent / "rewards.py").read_text()
    assert "gait_phase_fraction" in rewards_src
    assert "% period / period" not in rewards_src
    assert "phase_signal = torch.sin(2.0 * torch.pi * global_phase)" in rewards_src


def test_speed_clamp_periods():
    obs = _load_observations()
    env = _Env(cmd_speed=0.5)
    env.episode_length_buf += 1
    p_init = obs.gait_phase_fraction(env, period=0.72, period_start=0.72, period_end=0.60)
    assert torch.allclose(p_init, torch.zeros_like(p_init))
    env.episode_length_buf += 1
    p_low = obs.gait_phase_fraction(env, period=0.72, period_start=0.72, period_end=0.60)
    assert torch.allclose(p_low, torch.full_like(p_low, env.step_dt / 0.72), atol=1e-6)
    env.command_manager._cmd[:, 0] = 3.5
    env.episode_length_buf += 1
    p_high = obs.gait_phase_fraction(env, period=0.72, period_start=0.72, period_end=0.60)
    expected = (env.step_dt / 0.72 + env.step_dt / 0.60) % 1.0
    assert torch.allclose(p_high, torch.full_like(p_high, expected), atol=1e-6)


def test_common_step_counter_single_increment():
    obs = _load_observations()
    env = _Env(cmd_speed=2.4)
    env.common_step_counter = 0
    env.episode_length_buf += 2  # past reset window (buf<=1 zeros)
    a = obs.gait_phase_fraction(env, period=0.72)
    env.common_step_counter = 1
    b = obs.gait_phase_fraction(env, period=0.72)
    env.common_step_counter = 1  # same step re-entry
    c = obs.gait_phase_fraction(env, period=0.72)
    assert torch.allclose(b, c)
    expected = (a + env.step_dt / 0.72) % 1.0
    assert torch.allclose(b, expected, atol=1e-6)
    assert torch.all(a > 0)


if __name__ == "__main__":
    test_period_change_does_not_jump_phase()
    test_same_step_double_call_no_double_increment()
    test_episode_reset_zeros_phase()
    test_none_period_defaults_keep_fixed_period()
    test_three_consumers_share_phase()
    test_speed_clamp_periods()
    test_common_step_counter_single_increment()
    print("OK: continuous gait phase tests passed")
