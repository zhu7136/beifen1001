"""Unit tests for high-speed velocity curriculum + fixed-speed eval helpers (no Isaac required)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

# allow running from repo root without install
PKG_ROOT = Path(__file__).resolve().parents[1]  # source/limx_rl_lab
REPO_ROOT = Path(__file__).resolve().parents[3]  # limx_rl_lab-main
sys.path.insert(0, str(PKG_ROOT))
ROOT = REPO_ROOT


def test_evaluate_speed_gates_promote_and_freeze():
    from limx_rl_lab.utils.velocity_curriculum import apply_speed_promotion, evaluate_speed_gates

    good = evaluate_speed_gates(
        completion_rate=0.98,
        orientation_term_rate=0.01,
        speed_rmse_mps=0.15,
        saturation_rate=0.05,
    )
    assert good.gates_pass
    assert good.reason == ""

    bad = evaluate_speed_gates(
        completion_rate=0.80,
        orientation_term_rate=0.10,
        speed_rmse_mps=0.40,
        saturation_rate=0.30,
    )
    assert not bad.gates_pass
    assert "completion" in bad.reason
    assert "orientation" in bad.reason
    assert "rmse" in bad.reason
    assert "saturation" in bad.reason

    from limx_rl_lab.utils.velocity_curriculum import CurriculumState

    state = CurriculumState(
        current_max_speed_mps=1.4,
        target_speed_mps=2.78,
        speed_increment_mps=0.2,
        dwell_min_iterations=0,
        consecutive_pass_required=1,
        rolling_window=1,
    )
    # first passing gate promotes (dwell already met)
    apply_speed_promotion(state, good, current_step=0)
    assert state.current_max_speed_mps == 1.6
    assert state.promoted_count == 1
    assert state.level == 1
    # further good gates continue promoting at increment
    for step in (50, 100, 150):
        apply_speed_promotion(state, good, current_step=step)
    assert state.promoted_count == 4
    assert abs(state.current_max_speed_mps - 2.2) < 1e-6

    # freeze on repeated failure (hold, no degrade)
    fail = evaluate_speed_gates(
        completion_rate=0.5,
        orientation_term_rate=0.2,
        speed_rmse_mps=0.5,
        saturation_rate=0.4,
    )
    apply_speed_promotion(state, fail, current_step=300)
    apply_speed_promotion(state, fail, current_step=350)
    assert state.frozen
    assert state.freeze_reason
    frozen_max = state.current_max_speed_mps
    apply_speed_promotion(state, fail, current_step=400)
    assert state.current_max_speed_mps == frozen_max

    # clamp at target
    state2 = CurriculumState(current_max_speed_mps=2.78, target_speed_mps=2.78)
    apply_speed_promotion(state2, good, current_step=0)
    assert state2.current_max_speed_mps == 2.78


def test_command_mix_histogram_and_straight_line():
    import torch

    from limx_rl_lab.utils.velocity_curriculum import sample_mixed_velocity_commands

    gen = torch.Generator().manual_seed(0)
    cmd = sample_mixed_velocity_commands(
        num_envs=1000,
        lin_vel_x_max=2.0,
        device="cpu",
        lin_vel_x_min=-0.5,
        lin_vel_y_range=(-0.05, 0.05),
        ang_vel_z_range=(-0.25, 0.25),
        near_frac=0.4,
        mid_frac=0.4,
        low_frac=0.2,
        low_stand_max=0.2,
        generator=gen,
    )
    assert cmd.shape == (1000, 3)
    vx = cmd[:, 0]
    vy = cmd[:, 1]
    wz = cmd[:, 2]
    near = float(((vx >= 0.9 * 2.0 - 1e-6) & (vx <= 2.0 + 1e-6)).float().mean())
    mid = float(((vx >= 0.3 * 2.0 - 1e-6) & (vx < 0.8 * 2.0)).float().mean())
    low = float((vx <= 0.2 + 1e-6).float().mean())
    assert near > 0.25
    assert mid > 0.25
    assert low > 0.1
    assert float(vy.abs().max()) <= 0.05 + 1e-5
    assert float(wz.abs().max()) <= 0.25 + 1e-5


def test_eval_metrics_and_gates():
    from limx_rl_lab.utils.velocity_eval import (
        SpeedGates,
        build_eval_report,
        compute_run_metrics,
        recommend_initial_max_speed,
        summarize_speed,
    )

    n = 200
    target = 2.78
    vx_cmd = [target] * n
    vx_meas = [target - 0.1 + 0.05 * math.sin(i / 10) for i in range(n)]
    rec = compute_run_metrics(
        target_speed_mps=target,
        seed=0,
        vx_cmd=vx_cmd,
        vx_meas=vx_meas,
        timed_out=[1.0, 1.0, 1.0, 1.0],
        terminated=[0.0, 0.0, 0.0, 0.0],
        bad_orientation=[0.0, 0.0, 0.0, 0.0],
        base_contact=[0.0] * 4,
        base_height=[0.0] * 4,
        foot_slide=0.05,
        joint_vel_sat=0.01,
        joint_torque_sat=0.01,
        action_clip_ratio=0.02,
        joint_power=50.0,
        robot_mass_kg=30.0,
    )
    assert rec.mean_vx_mps > 2.0
    assert rec.rmse_vx_mps < 0.25
    assert rec.completion_rate == 1.0
    assert rec.mean_total_joint_power_w == 50.0
    assert rec.extras["robot_mass_kg"] == 30.0
    # CoT = total_power / (mass * 9.81 * max(|mean_vx|, 1e-3))
    expected_cot = 50.0 / (30.0 * 9.81 * max(abs(rec.mean_vx_mps), 1e-3))
    assert math.isclose(rec.cost_of_transport, expected_cot, rel_tol=1e-9)
    assert rec.command_vx_min_mps == target
    assert rec.command_vx_max_mps == target

    rec2 = compute_run_metrics(
        target_speed_mps=target,
        seed=1,
        vx_cmd=[1.0] * n,
        vx_meas=[0.4] * n,
        timed_out=[0.0, 0.0, 0.0],
        terminated=[1.0, 1.0, 1.0],
        bad_orientation=[1.0, 1.0, 1.0],
        base_contact=[1.0, 1.0, 0.0],
        base_height=[0.0, 0.0, 0.0],
        foot_slide=0.4,
        joint_vel_sat=0.3,
        joint_torque_sat=0.3,
        action_clip_ratio=0.4,
        joint_power=80.0,
        robot_mass_kg=30.0,
    )

    s_pass = summarize_speed([rec], SpeedGates())
    assert s_pass["gates_pass"] is True
    assert "mean_vx_mps" in s_pass
    assert "rmse_vx_mps" in s_pass
    assert "completion_rate" in s_pass
    assert "bad_orientation_rate" in s_pass
    assert "foot_slide_mean" in s_pass
    assert "joint_vel_saturation_rate" in s_pass
    assert "action_clip_ratio" in s_pass
    assert "cost_of_transport" in s_pass
    assert "mean_total_joint_power_w" in s_pass
    assert "mean_joint_power_w" not in s_pass
    assert s_pass["command_vx_min_mps"] == target
    assert s_pass["command_vx_max_mps"] == target

    s_fail = summarize_speed([rec2], SpeedGates())
    assert s_fail["gates_pass"] is False
    assert s_fail["gates"]["orientation_pass"] is False

    # Inclusive boundary: exact threshold values count as pass
    boundary = compute_run_metrics(
        target_speed_mps=2.0,
        seed=0,
        vx_cmd=[2.0] * 10,
        vx_meas=[1.85] * 10,
        timed_out=[1.0] * 4,
        terminated=[0.0] * 4,
        bad_orientation=[0.0] * 4,
        joint_power=10.0,
        robot_mass_kg=30.0,
        # force boundary by passing custom gates below
    )
    s_bound = summarize_speed(
        [boundary],
        SpeedGates(completion_rate_min=1.0, orientation_term_rate_max=0.0, speed_rmse_max_mps=0.15),
    )
    assert s_bound["completion_rate"] == 1.0
    assert s_bound["gates"]["completion_pass"] is True
    assert s_bound["gates"]["orientation_pass"] is True
    assert s_bound["gates"]["rmse_pass"] is True
    assert s_bound["gates_pass"] is True

    report = build_eval_report(
        checkpoint="model_4999.pt",
        speeds=[1.0, 2.78],
        seed_records=[rec],  # only the passing 2.78 run
        gates=SpeedGates(),
    )
    assert report["highest_passing_speed_mps"] == 2.78
    assert "initial_max_speed_mps" not in report
    assert "eval/2.78/rmse_vx" in report["wandb_metrics"]
    assert "eval/2.78/mean_total_joint_power_w" in report["wandb_metrics"]
    assert "eval/2.78/command_vx_mean_mps" in report["wandb_metrics"]
    assert "authoritative" in report["note"].lower() or "Fixed-speed" in report["note"]
    assert "not comparable" in report["note"].lower()

    # failing only → null highest passing
    report_fail = build_eval_report(
        checkpoint="model_4999.pt",
        speeds=[2.78],
        seed_records=[rec2],
        gates=SpeedGates(),
    )
    assert report_fail["summary"][0]["gates_pass"] is False
    assert report_fail["highest_passing_speed_mps"] is None
    assert "initial_max_speed_mps" not in report_fail

    assert recommend_initial_max_speed([s_fail, s_pass]) == 2.78
    assert recommend_initial_max_speed([s_fail]) is None


def test_fixed_speed_command_mode_source():
    """Command term exposes fixed_speed_mps default None and fixed resample path."""
    src = (PKG_ROOT / "limx_rl_lab/tasks/locomotion/mdp/commands/velocity_command.py").read_text()
    assert "fixed_speed_mps: float | None = None" in src
    assert "if fixed_speed is not None:" in src
    assert "if getattr(self.cfg, \"fixed_speed_mps\", None) is not None:" in src

    try:
        from limx_rl_lab.tasks.locomotion.mdp.commands.velocity_command import (
            HighSpeedUniformVelocityCommand,
            HighSpeedUniformVelocityCommandCfg,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[skip] isaaclab command import unavailable: {exc}")
        return

    cfg = HighSpeedUniformVelocityCommandCfg()
    assert getattr(cfg, "fixed_speed_mps", None) is None

    import torch

    class _Stub:
        pass

    stub = _Stub()
    stub.device = "cpu"
    stub.num_envs = 4
    stub.vel_command_b = torch.zeros(4, 3)
    stub.is_standing_env = torch.ones(4, dtype=torch.bool)
    stub.is_heading_env = torch.ones(4, dtype=torch.bool)
    stub.heading_target = torch.zeros(4)

    class _Ranges:
        heading = (0.0, 0.0)
        lin_vel_x = (0.0, 2.0)
        lin_vel_y = (-0.05, 0.05)
        ang_vel_z = (-0.25, 0.25)

    class _Cfg:
        fixed_speed_mps = 2.4
        heading_command = False
        rel_standing_envs = 0.5
        rel_heading_envs = 0.0
        ranges = _Ranges()
        near_frac = 0.5
        mid_frac = 0.35
        low_frac = 0.15
        low_stand_max = 0.2

    stub.cfg = _Cfg()
    HighSpeedUniformVelocityCommand._resample_command(stub, list(range(4)))
    assert torch.allclose(stub.vel_command_b[:, 0], torch.full((4,), 2.4))
    assert torch.allclose(stub.vel_command_b[:, 1], torch.zeros(4))
    assert torch.allclose(stub.vel_command_b[:, 2], torch.zeros(4))
    assert not bool(stub.is_standing_env.any())


def test_foot_slide_contact_gated_and_total_mass():
    """Contact-gated foot slide + total mass helpers (synthetic, no Isaac)."""
    import torch

    # foot-slide accumulation logic mirrors eval_velocity.py
    foot_vel_xy = torch.tensor([[0.1, 0.0], [2.4, 0.0], [0.05, 0.0], [0.0, 2.4]])
    contact = torch.tensor([[True, False], [False, False], [True, True], [False, True]])
    slide_sum = float((foot_vel_xy * contact).sum().item())
    slide_samples = int(contact.sum().item())
    foot_slide = slide_sum / max(slide_samples, 1)
    # contact samples only: 0.1 + 0.05 + 2.4 = 2.55 / 4
    assert math.isclose(foot_slide, 2.55 / 4.0, rel_tol=1e-5, abs_tol=1e-6)
    # non-contact feet are ignored even if sliding
    assert math.isclose(slide_sum, 2.55, rel_tol=1e-5, abs_tol=1e-6)
    assert slide_samples == 4

    class _View:
        def __init__(self, masses):
            self._m = masses

        def get_masses(self):
            return self._m

    class _Robot:
        def __init__(self, masses):
            self.root_physx_view = _View(masses)

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "eval_velocity_mod", ROOT / "scripts" / "rsl_rl" / "eval_velocity.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    _total_robot_mass = mod._total_robot_mass

    # 1D: sum of body masses
    r1 = _Robot(torch.tensor([10.0, 15.0, 5.0]))
    assert _total_robot_mass(r1) == 30.0
    # 2D: per-env body sum, then mean over envs
    r2 = _Robot(torch.tensor([[10.0, 15.0, 5.0], [8.0, 8.0, 8.0]]))
    assert _total_robot_mass(r2) == 27.0  # (30 + 24) / 2
    # mean-of-bodies would be 12.0 — must not be used
    assert _total_robot_mass(r2) != 12.0


def test_highspeed_cfg_and_registration():
    """Config import + gym registration without Isaac sim runtime."""
    # high_speed_env_cfg imports isaaclab — skip if unavailable
    try:
        import limx_rl_lab.tasks.locomotion.robots.limx.high_speed_env_cfg as hs_cfg  # noqa: F401
        import limx_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg as agent_cfg  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        print(f"[skip] isaaclab not importable: {exc}")
        return

    cfg = hs_cfg.RobotHighSpeedEnvCfg()
    assert cfg.commands.base_velocity.limit_ranges.lin_vel_x[1] >= 2.78
    assert cfg.commands.base_velocity.high_speed.target_speed_mps == 2.78
    assert cfg.commands.base_velocity.ranges.lin_vel_y[1] <= 0.05
    assert hasattr(cfg.curriculum, "gated_speed_levels")
    # training path stays mixed (fixed mode off)
    assert getattr(cfg.commands.base_velocity, "fixed_speed_mps", None) is None
    # baseline rewards preserved (not redesigned)
    assert cfg.rewards.track_lin_vel_xy.weight == 1.5
    assert agent_cfg.HighSpeedPPORunnerCfg.learning_rate == 1e-4
    assert agent_cfg.HighSpeedPPORunnerCfg.save_interval == 100
    assert 0.40 <= agent_cfg.HighSpeedPPORunnerCfg.policy.init_noise_std <= 0.45


def test_eval_cli_help():
    """eval_velocity.py --help lists speeds/seeds/checkpoint without launching Isaac."""
    import subprocess

    script = ROOT / "scripts" / "rsl_rl" / "eval_velocity.py"
    # prefer isaaclab python.sh so PATH matches training env
    candidates = [
        ["/workspace/isaaclab/_isaac_sim/python.sh", str(script), "--help"],
        [sys.executable, str(script), "--help"],
    ]
    out = ""
    for cmd in candidates:
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            out = (proc.stdout or "") + (proc.stderr or "")
            if "--checkpoint" in out:
                break
        except Exception as exc:  # noqa: BLE001
            out += f"\n{exc}"
    assert "--checkpoint" in out, out
    assert "--num_seeds" in out, out
    assert "--speeds" in out, out
    assert "authoritative" in out.lower() or "Fixed-speed" in out
    assert "fixed_speed_mps" in out or "pinned" in out


if __name__ == "__main__":
    test_evaluate_speed_gates_promote_and_freeze()
    test_command_mix_histogram_and_straight_line()
    test_eval_metrics_and_gates()
    test_fixed_speed_command_mode_source()
    test_foot_slide_contact_gated_and_total_mass()
    test_highspeed_cfg_and_registration()
    test_eval_cli_help()
    print("ALL TESTS PASSED")
