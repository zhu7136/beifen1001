#!/usr/bin/env python3
"""Fixed-speed velocity policy evaluation for LimX flat velocity tasks (v3).

Authoritative high-speed acceptance metric. Mixed Train/mean_reward aggregates
are NOT sufficient evidence that a policy holds a high target speed (e.g. 2.78 m/s).

v3 changes vs eval_velocity.py:
  - explicit env.reset() per seed after pinning fixed_speed_mps
  - first-episode-only health / velocity stats (valid mask)
  - phase stats: full / startup (<2s) / steady (>=2s)
  - joint-vel exceedance bins gt_1pct / gt_5pct
  - W&B phase metrics via pooled SSE (not mean of per-seed RMSE)

Example:
  python scripts/rsl_rl/eval_velocity_v3.py \\
    --task LimX-HU-D04-01-Flat-Velocity-HS \\
    --checkpoint logs/rsl_rl/.../model_2500.pt \\
    --headless --num_envs 5 --output eval_hs_v3.json
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
        description="Fixed-speed velocity evaluation v3 (authoritative for high-speed capability)."
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
    parser.add_argument("--output", type=str, default="eval_velocity_report_v3.json")
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
        print("usage: eval_velocity_v3.py [-h] --task TASK --checkpoint PATH")
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
        print("v3: explicit reset per seed; first-episode valid-mask stats; phase metrics.")
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

    # Phase metrics: pool SSE / sample counts across seeds, then sqrt.
    report.setdefault("wandb_metrics", {})
    for speed in speeds:
        records = [
            r for r in seed_records
            if r.target_speed_mps == speed
        ]
        for phase in ("full", "startup", "steady"):
            stats = [
                r.extras["phase_stats"][phase]
                for r in records
            ]
            n = sum(s["count"] for s in stats)
            if not n:
                continue

            prefix = f"eval/{speed:g}/{phase}"
            report["wandb_metrics"][f"{prefix}/rmse_vx"] = (
                sum(s["squared_error_sum"] for s in stats) / n
            ) ** 0.5
            report["wandb_metrics"][f"{prefix}/mean_vx"] = (
                sum(s["vx_sum"] for s in stats) / n
            )
            report["wandb_metrics"][f"{prefix}/sample_count"] = n
            report["wandb_metrics"][f"{prefix}/bias_mps"] = (
                sum(s["error_sum"] for s in stats) / n
            )
            g1 = sum(s.get("gt_1pct_hits", 0) for s in stats)
            g5 = sum(s.get("gt_5pct_hits", 0) for s in stats)
            report["wandb_metrics"][f"{prefix}/gt_1pct_rate"] = g1 / n
            report["wandb_metrics"][f"{prefix}/gt_5pct_rate"] = g5 / n

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"[eval] wrote {out_path}")
    print(f"[eval] highest_passing_speed_mps={report.get('highest_passing_speed_mps')}")
    print("[eval] NOTE: fixed-speed metrics are authoritative; do not use mixed Train/mean_reward alone.")
    print("[eval] NOTE: v3 phase metrics are new; old eval runs do not backfill them.")

    if args.logger == "wandb":
        try:
            import wandb

            wandb.login()
            run = wandb.init(
                project=args.log_project_name,
                name=f"eval_v3_{Path(resume_path).parent.name}_{Path(resume_path).stem}",
                config={"checkpoint": resume_path, "speeds": speeds, "num_seeds": args.num_seeds},
            )
            wandb.log(report["wandb_metrics"])
            wandb.save(
                str(out_path.resolve()),
                base_path=str(out_path.resolve().parent),
                policy="now",
            )
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
    """Run one independent fixed-speed evaluation (first episode only)."""
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

    # 固定指令配置已设置；每个 seed 从一个新回合开始。
    with torch.inference_mode():
        env.reset()
        cmd_term._resample_command(
            list(range(env.unwrapped.num_envs))
        )
        obs = env.get_observations()

    if isinstance(obs, tuple):
        obs = obs[0]

    episode_len = int(env.unwrapped.max_episode_length)
    num_envs = env.unwrapped.num_envs

    vx_cmd_hist = []
    vx_meas_hist = []
    power_sum = 0.0
    sat_vel_hits = 0
    sat_tq_hits = 0
    gt_1pct_hits = 0
    gt_5pct_hits = 0
    clip_hits = 0
    clip_total = 0
    slide_sum = 0.0
    slide_samples = 0
    sample_count = 0
    phase_stats = {
        p: {"count": 0, "error_sum": 0.0,
            "squared_error_sum": 0.0, "vx_sum": 0.0,
            "gt_1pct_hits": 0, "gt_5pct_hits": 0}
        for p in ("full", "startup", "steady")
    }
    dt = float(env.unwrapped.step_dt)

    timed_out = torch.zeros(num_envs, dtype=torch.bool, device=device)
    terminated = torch.zeros(num_envs, dtype=torch.bool, device=device)
    bad_ori = torch.zeros(num_envs, dtype=torch.bool, device=device)
    base_contact = torch.zeros(num_envs, dtype=torch.bool, device=device)
    base_height = torch.zeros(num_envs, dtype=torch.bool, device=device)
    finished_once = torch.zeros(
        num_envs, dtype=torch.bool, device=device
    )

    max_steps = episode_len + 5
    for step_idx in range(max_steps):
        alive_before = ~finished_once

        with torch.inference_mode():
            actions = policy(obs)
            live_actions = actions[alive_before]
            clip_total += live_actions.numel()
            clip_hits += int(
                (live_actions.abs() > 1.0).sum().item()
            )
            obs, _, dones, _extras = env.step(actions)

        if isinstance(obs, tuple):
            obs = obs[0]

        dones = dones.to(device=device).bool().reshape(-1)
        first_done = alive_before & dones

        tm = env.unwrapped.termination_manager
        if tm is None:
            raise RuntimeError("无法读取回合终止信息，停止评估")

        # 同时出现超时和失败时，失败优先，不能算完成。
        terminated[first_done] = tm.terminated[first_done]
        timed_out[first_done] = (
            tm.time_outs[first_done]
            & ~tm.terminated[first_done]
        )

        # 用当前步的公开接口，不依赖 _last_episode_dones。
        for name, output in (
            ("bad_orientation", bad_ori),
            ("base_contact", base_contact),
            ("base_height", base_height),
        ):
            if name in tm.active_terms:
                output[first_done] = tm.get_term(name)[first_done]

        finished_once |= first_done

        # env.step() 会自动重置结束的环境。
        # 所以结束步返回的机器人状态不能混入原回合统计。
        valid = alive_before & ~dones
        if not valid.any():
            if finished_once.all():
                break
            continue

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
        if valid.any():
            command = vx_cmd[valid]
            measured = vx_meas[valid]
            error = command - measured

            vx_cmd_hist.append(command.detach().cpu().clone())
            vx_meas_hist.append(measured.detach().cpu().clone())

            n = int(valid.sum().item())
            sample_count += n
            phase = (
                "startup"
                if (step_idx + 1) * dt < 2.0
                else "steady"
            )

            for label in ("full", phase):
                st = phase_stats[label]
                st["count"] += n
                st["error_sum"] += float(error.sum().item())
                st["squared_error_sum"] += float(
                    error.square().sum().item()
                )
                st["vx_sum"] += float(measured.sum().item())

            jv = robot.data.joint_vel
            lim = robot.data.soft_joint_vel_limits
            strict_hits = (jv.abs() > lim).any(dim=1)
            gt1_hits = (jv.abs() > lim * 1.01).any(dim=1)
            gt5_hits = (jv.abs() > lim * 1.05).any(dim=1)
            phase_stats["full"]["gt_1pct_hits"] += int(gt1_hits[valid].sum().item())
            phase_stats["full"]["gt_5pct_hits"] += int(gt5_hits[valid].sum().item())
            phase_stats[phase]["gt_1pct_hits"] += int(gt1_hits[valid].sum().item())
            phase_stats[phase]["gt_5pct_hits"] += int(gt5_hits[valid].sum().item())
            sat_vel_hits += int(strict_hits[valid].sum().item())
            gt_1pct_hits += int(gt1_hits[valid].sum().item())
            gt_5pct_hits += int(gt5_hits[valid].sum().item())

        if foot_body_ids is not None and foot_sensor_ids is not None and contact_sensor is not None:
            foot_vel_xy = torch.linalg.vector_norm(
                robot.data.body_lin_vel_w[:, foot_body_ids, :2],
                dim=-1,
            )
            contact = contact_sensor.data.current_contact_time[:, foot_sensor_ids] > 0.0
            contact_valid = contact & valid[:, None]
            slide_sum += float((foot_vel_xy * contact_valid).sum().item())
            slide_samples += int(contact_valid.sum().item())

        tau = robot.data.applied_torque
        omega = robot.data.joint_vel
        mech_power_env = (tau * omega).abs().sum(dim=-1)
        power_sum += float(mech_power_env[valid].sum().item())

        jv = robot.data.joint_vel
        lim = robot.data.soft_joint_vel_limits
        at = robot.data.applied_torque
        if hasattr(robot.data, "soft_joint_effort_limits") and robot.data.soft_joint_effort_limits is not None:
            torque_hits = (at.abs() > robot.data.soft_joint_effort_limits).any(dim=1)
        else:
            ct = robot.data.computed_torque
            torque_hits = (~torch.isclose(ct, at)).any(dim=1)
        sat_tq_hits += int(torque_hits[valid].sum().item())

        if finished_once.all():
            break

    if not finished_once.all():
        pending = (~finished_once).nonzero(
            as_tuple=False
        ).flatten().tolist()
        raise RuntimeError(
            f"以下环境未记录到第一次回合结束：{pending}"
        )

    vx_cmd_cat = torch.cat(vx_cmd_hist, dim=0).numpy().tolist() if vx_cmd_hist else []
    vx_meas_cat = torch.cat(vx_meas_hist, dim=0).numpy().tolist() if vx_meas_hist else []

    foot_slide = slide_sum / max(slide_samples, 1)
    n_samples = max(sample_count, 1)

    record = compute_run_metrics(
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
        joint_vel_sat=sat_vel_hits / n_samples,
        joint_torque_sat=sat_tq_hits / n_samples,
        action_clip_ratio=(clip_hits / clip_total) if clip_total else 0.0,
        joint_power=(power_sum / n_samples) if sample_count else 0.0,
        robot_mass_kg=mass,
        g=9.81,
    )
    record.extras["phase_stats"] = phase_stats
    record.extras["warmup_s"] = 2.0
    record.extras["gt_1pct_rate"] = gt_1pct_hits / n_samples
    record.extras["gt_5pct_rate"] = gt_5pct_hits / n_samples

    for phase, st in phase_stats.items():
        n = st["count"]
        record.extras[phase] = {
            "count": n,
            "rmse_vx_mps": (
                (st["squared_error_sum"] / n) ** 0.5
                if n else None
            ),
            "mean_vx_mps": st["vx_sum"] / n if n else None,
            "bias_mps": st["error_sum"] / n if n else None,
            "gt_1pct_rate": (
                st["gt_1pct_hits"] / n if n else None
            ),
            "gt_5pct_rate": (
                st["gt_5pct_hits"] / n if n else None
            ),
        }

    return record


if __name__ == "__main__":
    main()
