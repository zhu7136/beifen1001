"""Unit tests for rolling frontier curriculum + staged eval gates (no Isaac required)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PKG_ROOT))
ROOT = REPO_ROOT


def test_frontier_mask_and_stats():
    import torch

    from limx_rl_lab.utils.velocity_curriculum import compute_frontier_stats, frontier_mask

    vx_cmd = torch.tensor([0.2, 0.5, 2.0, 2.0, 1.0, 2.0])
    vx_meas = torch.tensor([0.2, 0.4, 1.5, 2.0, 1.0, 1.8])
    mask = frontier_mask(vx_cmd, current_max_speed_mps=2.0)
    assert mask.tolist() == [False, False, True, True, False, True]

    stats = compute_frontier_stats(
        vx_cmd=vx_cmd,
        vx_meas=vx_meas,
        completion_rate=0.99,
        orientation_rate=0.01,
        joint_vel_sat_rate=0.02,
        joint_torque_sat_rate=0.01,
        action_clip_ratio=0.05,
        current_max_speed_mps=2.0,
    )
    assert stats.n_frontier == 3
    # frontier rmse only over [2,2,2] vs [1.5,2,1.8]
    expected = math.sqrt(((2.0 - 1.5) ** 2 + 0 + (2.0 - 1.8) ** 2) / 3)
    assert abs(stats.rmse_mps - expected) < 1e-5
    # mixed dilutes rmse
    assert stats.mixed_rmse_mps < stats.rmse_mps or stats.n_frontier < len(vx_cmd)
    assert stats.saturation_rate == 0.02


def test_rolling_promote_and_hold():
    from limx_rl_lab.utils.velocity_curriculum import (
        CurriculumState,
        FrontierIterStats,
        apply_rolling_frontier_step,
        evaluate_rolling_frontier_gates,
    )

    good = FrontierIterStats(
        completion_rate=0.99,
        orientation_rate=0.01,
        rmse_mps=0.20,
        saturation_rate=0.05,
        joint_vel_sat_rate=0.04,
        joint_torque_sat_rate=0.03,
        action_clip_ratio=0.02,
        commanded_vx_mps=2.0,
        measured_vx_mps=2.0,
        mixed_rmse_mps=0.15,
        n_frontier=100,
    )
    bad = FrontierIterStats(
        completion_rate=0.80,
        orientation_rate=0.10,
        rmse_mps=0.50,
        saturation_rate=0.40,
        n_frontier=50,
    )

    state = CurriculumState(
        current_max_speed_mps=2.0,
        target_speed_mps=2.4,
        speed_increment_mps=0.1,
        rolling_window=200,
        consecutive_pass_required=200,
        dwell_min_iterations=300,
    )

    # dwell blocks early promote even if gates pass
    for _ in range(200):
        apply_rolling_frontier_step(state, good)
    assert state.consecutive_pass_iterations == 200
    assert state.current_max_speed_mps == 2.0  # dwell < 300

    # continue good until dwell met
    for _ in range(100):
        apply_rolling_frontier_step(state, good)
    assert state.current_max_speed_mps == 2.1
    assert state.promoted_count == 1
    assert state.level == 1
    assert state.iterations_at_current_speed == 0
    assert state.frozen is False

    # fill window again then fail -> hold, no degrade
    for _ in range(200):
        apply_rolling_frontier_step(state, good)
    for _ in range(200):
        apply_rolling_frontier_step(state, bad)
    assert state.current_max_speed_mps == 2.1  # no auto-degrade
    assert state.frozen is True
    assert state.consecutive_pass_iterations == 0
    assert "completion" in state.last_gate_reason or "rmse" in state.last_gate_reason

    # empty history cannot promote
    state2 = CurriculumState(current_max_speed_mps=2.0, target_speed_mps=2.4, rolling_window=200)
    gate = evaluate_rolling_frontier_gates(state2)
    assert not gate.gates_pass
    assert gate.reason == "window_incomplete"

    # clamp at target
    state3 = CurriculumState(
        current_max_speed_mps=2.4,
        target_speed_mps=2.4,
        rolling_window=5,
        consecutive_pass_required=5,
        dwell_min_iterations=0,
    )
    for _ in range(10):
        apply_rolling_frontier_step(state3, good)
    assert state3.current_max_speed_mps == 2.4


def test_sample_mix_round2():
    import torch

    from limx_rl_lab.utils.velocity_curriculum import sample_mixed_velocity_commands

    gen = torch.Generator().manual_seed(0)
    cmd = sample_mixed_velocity_commands(
        num_envs=1000,
        lin_vel_x_max=2.0,
        device="cpu",
        lin_vel_x_min=-0.5,
        near_frac=0.50,
        mid_frac=0.35,
        low_frac=0.15,
        generator=gen,
    )
    assert cmd.shape == (1000, 3)
    vx = cmd[:, 0]
    # frontier coverage: some samples at exactly max
    assert float((vx >= 2.0 - 1e-6).float().mean()) > 0.05
    assert float(vy_abs := cmd[:, 1].abs().max()) <= 0.05 + 1e-5
    assert float(cmd[:, 2].abs().max()) <= 0.25 + 1e-5


def test_staged_eval_gates():
    from limx_rl_lab.utils.velocity_eval import SpeedGates, build_eval_report, compute_run_metrics
    from limx_rl_lab.utils.velocity_curriculum import gates_for_staged_eval

    g20 = SpeedGates.for_target(2.0)
    g24 = SpeedGates.for_target(2.4)
    assert g20.completion_rate_min == 0.95
    assert g24.completion_rate_min == 0.97
    assert g24.speed_rmse_max_mps == 0.25
    assert gates_for_staged_eval(2.4)["completion_rate_min"] == 0.97

    n = 100
    target = 2.4
    rec = compute_run_metrics(
        target_speed_mps=target,
        seed=0,
        vx_cmd=[target] * n,
        vx_meas=[target - 0.1] * n,
        timed_out=[1.0] * 4,
        terminated=[0.0] * 4,
        bad_orientation=[0.0] * 4,
        base_contact=[0.0] * 4,
        base_height=[0.0] * 4,
        foot_slide=0.1,
        joint_vel_sat=0.01,
        joint_torque_sat=0.01,
        action_clip_ratio=0.01,
        joint_power=40.0,
        robot_mass_kg=30.0,
    )
    # rmse=0.1 <= 0.25, completion=1.0 >= 0.97
    report = build_eval_report("model_2999.pt", [2.4], [rec])
    assert report["summary"][0]["gates"]["completion_rate_min"] == 0.97
    assert report["summary"][0]["gates_pass"] is True


def test_checkpoint_and_docs():
    ckpt = ROOT / "logs/rsl_rl/limx_hu_d04_01_flat_velocity_hs/2026-10-01_20-46-27_hs-ft-001/model_2999.pt"
    assert ckpt.exists(), f"missing preferred checkpoint: {ckpt}"
    doc = ROOT / "docs/hs_round2_rolling_frontier.md"
    assert doc.exists()
    text = doc.read_text()
    assert "model_2999.pt" in text
    assert "model_2400.pt" in text
    assert "2.4" in text
    assert "0.50" in text


if __name__ == "__main__":
    test_frontier_mask_and_stats()
    test_rolling_promote_and_hold()
    test_sample_mix_round2()
    test_staged_eval_gates()
    test_checkpoint_and_docs()
    print("ALL TESTS PASSED")
