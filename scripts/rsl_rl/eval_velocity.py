#!/usr/bin/env python3
"""Fixed-speed velocity policy evaluation for LimX flat velocity tasks.

Authoritative high-speed acceptance metric. Mixed Train/mean_reward aggregates
are NOT sufficient evidence that a policy holds a high target speed (e.g. 2.78 m/s).

Example:
  python scripts/rsl_rl/eval_velocity.py \\
    --task LimX-HU-D04-01-Flat-Velocity \\
    --checkpoint logs/rsl_rl/limx_hu_d04_01_flat_velocity/<run>/model_4999.pt \\
    --headless --num_envs 5 --output eval_hs.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

FOOT_BODY_NAMES = ["left_ankle_roll_link", "right_ankle_roll_link"]


def _total_robot_mass(robot) -> float | None:
    """Total articulation mass: per-env sum of body masses, then mean over envs."""
    try:
        masses = robot.root_physx_view.get_masses()
    except Exception:  # noqa: BLE001
        return None
    if masses.ndim == 1:
        return float(masses.sum().item())
    return float(masses.sum(dim=-1).mean().item())


def _resolve_foot_ids(robot, contact_sensor):
    foot_body_ids = None
    foot_sensor_ids = None
    if robot is not None and hasattr(robot, "find_bodies"):
        ids, _ = robot.find_bodies(FOOT_BODY_NAMES)
        if ids is not None and len(ids) > 0:
            foot_body_ids = ids
    if contact_sensor is not None and hasattr(contact_sensor, "find_bodies"):
        ids, _ = contact_sensor.find_bodies(FOOT_BODY_NAMES)
        if ids is not None and len(ids) > 0:
            foot_sensor_ids = ids
    return foot_body_ids, foot_sensor_ids


def _build_parser() -> argparse.ArgumentParser:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(
        description="Fixed-speed velocity evaluation (authoritative for high-speed capability)."
    )
    parser.add_argument("--task", type=str, default="LimX-HU-D04-01-Flat-Velocity")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model_*.pt checkpoint to evaluate.",
    )
    parser.add_argument(
        "--speeds",
        type=str,
        default="1.0,1.4,1.8,2.2,2.5,2.78",
        help="Comma-separated target forward speeds in m/s.",
    )
    parser.add_argument("--num_seeds", type=int, default=5, help="Seeds per speed (default 5).")
    parser.add_argument("--num_envs", type=int, default=None, help="Envs to create (default: num_seeds).")
    parser.add_argument("--episode_length_s", type=float, default=20.0)
    parser.add_argument("--output", type=str, default="eval_velocity_report.json")
    parser.add_argument("--logger", type=str, default=None, choices=["wandb", "tensorboard", "none"])
    parser.add_argument("--log_project_name", type=str, default="limx-hu-d04-01")
    parser.add_argument("--seed", type=int, default=0)
    # --device / --headless come from AppLauncher.add_app_launcher_args
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _parse_args(argv=None):
    """Parse args without importing Isaac when only --help is requested."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--help", action="store_true")
    pre.add_argument("--checkpoint", type=str, default=None)
    pre.add_argument("--num_seeds", type=int, default=None)
    pre.add_argument("--speeds", type=str, default=None)
    pre.add_argument("--headless", action="store_true", default=False)
    pre_args, _ = pre.parse_known_args(argv)
    if pre_args.help:
        print("usage: eval_velocity.py [-h] --task TASK --checkpoint PATH")
        print("  --speeds SPEEDS   Comma-separated m/s list (default 1.0,1.4,1.8,2.2,2.5,2.78)")
        print("  --num_seeds N     Seeds per speed (default 5)")
        print("  --num_envs N      Envs to create (default: num_seeds)")
        print("  --episode_length_s SEC")
        print("  --output PATH     JSON report path")
        print("  --logger {wandb,tensorboard,none}")
        print("  --log_project_name NAME")
        print("  --seed N  --device DEV  --headless")
        print("")
        print("Fixed-speed evaluation is authoritative for high-speed capability claims.")
        print("Do not use mixed Train/mean_reward alone as acceptance evidence.")
        print("Commanded vx is pinned via command-term fixed_speed_mps; drift raises.")
        sys.exit(0)
    parser = _build_parser()
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    speeds = [float(s) for s in args.speeds.split(",") if s.strip()]

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import torch

    import isaaclab_tasks  # noqa: F401
    from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
    from isaaclab.utils.assets import retrieve_file_path
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    from rsl_rl.runners import OnPolicyRunner

    import limx_rl_lab.tasks  # noqa: F401
    from limx_rl_lab.utils.parser_cfg import parse_env_cfg
    from limx_rl_lab.utils.velocity_eval import SpeedGates, build_eval_report

    env_cfg = parse_env_cfg(
        args.task,
        device=getattr(args, "device", "cuda:0"),
        num_envs=args.num_envs if args.num_envs else max(args.num_seeds, 1),
        entry_point_key="play_env_cfg_entry_point",
    )
    env_cfg.episode_length_s = args.episode_length_s
    env_cfg.seed = args.seed
    env_cfg.sim.device = getattr(args, "device", "cuda:0")

    # Deterministic evaluation: disable training-time disturbances.
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.events.push_robot = None

    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    resume_path = retrieve_file_path(args.checkpoint)

    env = gym.make(args.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=getattr(agent_cfg, "clip_actions", None))

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    cmd_term = env.unwrapped.command_manager.get_term("base_velocity")
    robot = env.unwrapped.scene["robot"]
    device = env.unwrapped.device
    scene = getattr(env.unwrapped, "scene", None)
    contact_sensor = None
    if scene is not None:
        sensors = getattr(scene, "sensors", None) or {}
        contact_sensor = sensors.get("contact_forces")

    mass = _total_robot_mass(robot)
    foot_body_ids, foot_sensor_ids = _resolve_foot_ids(robot, contact_sensor)

    seed_records = []
    for speed in speeds:
        for seed in range(args.num_seeds):
            record = _eval_one_speed(
                env=env,
                policy=policy,
                cmd_term=cmd_term,
                robot=robot,
                device=device,
                target_speed_mps=speed,
                seed=seed,
                global_seed=args.seed,
                mass=mass,
                contact_sensor=contact_sensor,
                foot_body_ids=foot_body_ids,
                foot_sensor_ids=foot_sensor_ids,
            )
            seed_records.append(record)
            print(
                f"[eval] speed={speed:.2f} seed={seed} "
                f"mean_vx={record.mean_vx_mps:.3f} rmse={record.rmse_vx_mps:.3f} "
                f"completion={record.completion_rate:.3f} "
                f"cmd_vx=[{record.command_vx_min_mps},{record.command_vx_max_mps}]"
            )

    report = build_eval_report(
        checkpoint=resume_path,
        speeds=speeds,
        seed_records=seed_records,
        gates=SpeedGates(),
    )
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"[eval] wrote {out_path}")
    print(f"[eval] highest_passing_speed_mps={report.get('highest_passing_speed_mps')}")
    print("[eval] NOTE: fixed-speed metrics are authoritative; do not use mixed Train/mean_reward alone.")
    print("[eval] NOTE: post-fix CoT is not comparable to historical JSON CoT.")

    if args.logger == "wandb":
        try:
            import wandb

            wandb.login()
            run = wandb.init(
                project=args.log_project_name,
                name=f"eval_{Path(resume_path).parent.name}_{Path(resume_path).stem}",
                config={"checkpoint": resume_path, "speeds": speeds, "num_seeds": args.num_seeds},
            )
            wandb.log(report["wandb_metrics"])
            run.finish()
        except Exception as exc:  # noqa: BLE001
            print(f"[eval] wandb logging failed: {exc}")

    env.close()
    simulation_app.close()


def _eval_one_speed(
    env,
    policy,
    cmd_term,
    robot,
    device,
    target_speed_mps,
    seed,
    global_seed,
    mass,
    contact_sensor=None,
    foot_body_ids=None,
    foot_sensor_ids=None,
):
    """Run one independent fixed-speed evaluation."""
    import torch

    from limx_rl_lab.utils.velocity_eval import compute_run_metrics

    run_seed = global_seed + seed
    torch.manual_seed(run_seed)

    # Seed the Isaac environment when supported.
    if hasattr(env.unwrapped, "seed"):
        env.unwrapped.seed(run_seed)

    # Fixed command mode: pin via cfg + degenerate ranges; re-apply every call.
    cmd_term.cfg.fixed_speed_mps = float(target_speed_mps)
    cmd_term.cfg.ranges.lin_vel_x = (
        float(target_speed_mps),
        float(target_speed_mps),
    )
    cmd_term.cfg.ranges.lin_vel_y = (0.0, 0.0)
    cmd_term.cfg.ranges.ang_vel_z = (0.0, 0.0)
    cmd_term.cfg.rel_standing_envs = 0.0

    # Resample command with the new fixed speed.
    with torch.inference_mode():
        cmd_term._resample_command(list(range(env.unwrapped.num_envs)))

    # Get fresh observations after command change.
    obs = env.get_observations()
    if isinstance(obs, tuple):
        obs = obs[0]

    episode_len = int(env.unwrapped.max_episode_length)
    num_envs = env.unwrapped.num_envs

    vx_cmd_hist = []
    vx_meas_hist = []
    power_hist = []
    sat_vel_hits = 0
    sat_tq_hits = 0
    step_count = 0
    clip_hits = 0
    clip_total = 0
    slide_sum = 0.0
    slide_samples = 0

    timed_out = torch.zeros(num_envs, dtype=torch.bool, device=device)
    terminated = torch.zeros(num_envs, dtype=torch.bool, device=device)
    bad_ori = torch.zeros(num_envs, dtype=torch.bool, device=device)
    base_contact = torch.zeros(num_envs, dtype=torch.bool, device=device)
    base_height = torch.zeros(num_envs, dtype=torch.bool, device=device)
    episodes_done = torch.zeros(num_envs, dtype=torch.long, device=device)

    max_steps = episode_len + 5
    for _ in range(max_steps):
        with torch.inference_mode():
            actions = policy(obs)
            clip_total += actions.numel()
            clip_hits += int((actions.abs() > 1.0).sum().item())
            obs, _, _, _extras = env.step(actions)
        if isinstance(obs, tuple):
            obs = obs[0]

        vx_cmd = cmd_term.command[:, 0]
        expected = torch.full_like(vx_cmd, float(target_speed_mps))
        if not torch.allclose(vx_cmd, expected, atol=1e-5, rtol=0.0):
            raise RuntimeError(
                "Fixed-speed evaluation was contaminated by command resampling: "
                f"target={target_speed_mps:.4f}, "
                f"min={vx_cmd.min().item():.4f}, "
                f"max={vx_cmd.max().item():.4f}, "
                f"mean={vx_cmd.mean().item():.4f}"
            )

        vx_meas = robot.data.root_lin_vel_b[:, 0]
        vx_cmd_hist.append(vx_cmd.detach().cpu().clone())
        vx_meas_hist.append(vx_meas.detach().cpu().clone())

        if foot_body_ids is not None and foot_sensor_ids is not None and contact_sensor is not None:
            foot_vel_xy = torch.linalg.vector_norm(
                robot.data.body_lin_vel_w[:, foot_body_ids, :2],
                dim=-1,
            )
            contact = contact_sensor.data.current_contact_time[:, foot_sensor_ids] > 0.0
            slide_sum += float((foot_vel_xy * contact).sum().item())
            slide_samples += int(contact.sum().item())

        tau = robot.data.applied_torque
        omega = robot.data.joint_vel
        mechanical_power = (tau * omega).abs().sum(dim=-1).mean()
        power_hist.append(float(mechanical_power.item()))

        jv = robot.data.joint_vel
        lim = robot.data.soft_joint_vel_limits
        sat_vel_hits += int((jv.abs() > lim).any(dim=1).sum().item())
        at = robot.data.applied_torque
        if hasattr(robot.data, "soft_joint_effort_limits") and robot.data.soft_joint_effort_limits is not None:
            sat_tq_hits += int((at.abs() > robot.data.soft_joint_effort_limits).any(dim=1).sum().item())
        else:
            ct = robot.data.computed_torque
            sat_tq_hits += int((~torch.isclose(ct, at)).any(dim=1).sum().item())
        step_count += 1

        tm = env.unwrapped.termination_manager
        if tm is not None:
            term_idx = {n: i for i, n in enumerate(tm.active_terms)}
            last = getattr(tm, "_last_episode_dones", None)
            if last is not None:
                finished = tm.dones.nonzero(as_tuple=False).flatten()
                if finished.numel() > 0:
                    if "time_out" in term_idx:
                        timed_out[finished] = last[finished, term_idx["time_out"]]
                    if "bad_orientation" in term_idx:
                        bad_ori[finished] = last[finished, term_idx["bad_orientation"]]
                    if "base_contact" in term_idx:
                        base_contact[finished] = last[finished, term_idx["base_contact"]]
                    if "base_height" in term_idx:
                        base_height[finished] = last[finished, term_idx["base_height"]]
                    terminated[finished] = ~timed_out[finished]
                    episodes_done[finished] += 1

        if episodes_done.min() >= 1:
            break

    vx_cmd_cat = torch.cat(vx_cmd_hist, dim=0).numpy().tolist() if vx_cmd_hist else []
    vx_meas_cat = torch.cat(vx_meas_hist, dim=0).numpy().tolist() if vx_meas_hist else []

    if episodes_done.max() == 0:
        timed_out = torch.ones_like(timed_out)
        terminated = torch.zeros_like(terminated)

    foot_slide = slide_sum / max(slide_samples, 1)
    n_env = num_envs
    return compute_run_metrics(
        target_speed_mps=target_speed_mps,
        seed=seed,
        vx_cmd=vx_cmd_cat,
        vx_meas=vx_meas_cat,
        timed_out=timed_out.cpu(),
        terminated=terminated.cpu(),
        bad_orientation=bad_ori.cpu(),
        base_contact=base_contact.cpu(),
        base_height=base_height.cpu(),
        foot_slide=foot_slide,
        joint_vel_sat=sat_vel_hits / max(n_env * max(step_count, 1), 1),
        joint_torque_sat=sat_tq_hits / max(n_env * max(step_count, 1), 1),
        action_clip_ratio=(clip_hits / clip_total) if clip_total else 0.0,
        joint_power=(sum(power_hist) / len(power_hist)) if power_hist else 0.0,
        robot_mass_kg=mass,
        g=9.81,
    )


if __name__ == "__main__":
    main()
