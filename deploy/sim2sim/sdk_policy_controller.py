import argparse
import copy
import ctypes
import csv
import glob
import os
import subprocess
import tempfile
import time
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation as R


def ensure_limxsdk_importable():
    try:
        import limxsdk.datatypes as datatypes  # noqa: F401
        import limxsdk.robot.Rate as Rate  # noqa: F401
        import limxsdk.robot.Robot as Robot  # noqa: F401
        import limxsdk.robot.RobotType as RobotType  # noqa: F401
        return
    except ImportError:
        pass

    wheel_candidates = sorted(
        glob.glob("/home/edy/limx_ws/humanoid-mujoco-sim/limxsdk-lowlevel/python3/*/limxsdk-*-py3-none-any.whl")
    )
    if not wheel_candidates:
        raise ImportError(
            "limxsdk is required for sdk_policy_controller.py. "
            "Could not find a local wheel under ~/limx_ws/humanoid-mujoco-sim/limxsdk-lowlevel/python3/."
        )

    wheel_path = Path(wheel_candidates[0])
    extract_dir = Path(tempfile.mkdtemp(prefix="limxsdk_runtime_"))
    robot_lib_dir = extract_dir / "limxsdk" / "robot"

    try:
        subprocess.run(
            ["unzip", "-q", str(wheel_path), "-d", str(extract_dir)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as exc:
        raise ImportError(
            "Failed to extract the local limxsdk wheel with `unzip`. "
            "Please make sure `unzip` is available, or install limxsdk manually."
        ) from exc

    os.environ["LD_LIBRARY_PATH"] = f"{robot_lib_dir}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    sys.path.insert(0, str(extract_dir))
    try:
        ctypes.CDLL(str(robot_lib_dir / "libpython3.8.so.1.0"))
    except OSError as exc:
        raise ImportError(
            "Failed to preload limxsdk's bundled libpython3.8.so.1.0. "
            "Please install limxsdk with pip from ~/limx_ws, or check that `unzip` extracted the wheel correctly."
        ) from exc

    try:
        import limxsdk.datatypes as datatypes  # noqa: F401
        import limxsdk.robot.Rate as Rate  # noqa: F401
        import limxsdk.robot.Robot as Robot  # noqa: F401
        import limxsdk.robot.RobotType as RobotType  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "limxsdk could not be imported even after extracting the local wheel. "
            f"Tried wheel: {wheel_path}"
        ) from exc


ensure_limxsdk_importable()

import limxsdk.datatypes as datatypes
import limxsdk.robot.Rate as Rate
import limxsdk.robot.Robot as Robot
import limxsdk.robot.RobotType as RobotType


def resolve_path(base_dir: Path, path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    # Prefer CWD-relative paths (repo-root CLI usage), fall back to script_dir-relative.
    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (base_dir / path).resolve()


def resolve_policy_path(config_path: Path, policy_root: str | None, path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path

    project_root = config_path.parents[3]

    # Repo-root-relative or CWD-relative --policy paths take precedence.
    for candidate in ((Path.cwd() / path).resolve(), (project_root / path).resolve()):
        if candidate.exists():
            return candidate

    if policy_root:
        policy_root_path = Path(policy_root)
        if not policy_root_path.is_absolute():
            policy_root_path = (project_root / policy_root_path).resolve()
        else:
            policy_root_path = policy_root_path.resolve()
        return (policy_root_path / path).resolve()

    return resolve_path(config_path.parent, path_str)


def as_float_array(value, length: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size == 1:
        return np.full(length, float(array.item()), dtype=np.float32)
    if array.size != length:
        raise ValueError(f"{name} must have length {length}, got {array.size}.")
    return array


def find_training_deploy_path(policy_path: Path) -> Path | None:
    candidates = [policy_path.parent / "deploy.yaml"]
    if len(policy_path.parents) >= 2:
        candidates.append(policy_path.parents[1] / "params" / "deploy.yaml")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_training_deploy_overrides(policy_path: Path, num_actions: int, update_rate: int) -> dict:
    deploy_path = find_training_deploy_path(policy_path)
    if deploy_path is None:
        return {}

    deploy_cfg = yaml.safe_load(deploy_path.read_text())
    overrides = {}

    action_cfg = deploy_cfg.get("actions", {}).get("JointPositionAction")
    if action_cfg is not None:
        if "offset" in action_cfg:
            # Action offsets are exported in the policy/action order, so they are safe to
            # reuse directly for observation centering and target-position reconstruction.
            overrides["default_angles"] = as_float_array(action_cfg["offset"], num_actions, "action offset")
        elif "default_joint_pos" in deploy_cfg:
            overrides["default_angles"] = as_float_array(
                deploy_cfg["default_joint_pos"], num_actions, "default_joint_pos"
            )

        if "scale" in action_cfg:
            overrides["action_scale"] = as_float_array(action_cfg["scale"], num_actions, "action scale")

    joint_ids_map = deploy_cfg.get("joint_ids_map")
    if joint_ids_map is not None:
        joint_ids_map = np.asarray(joint_ids_map, dtype=np.int64).reshape(-1)
        if joint_ids_map.size != num_actions:
            raise ValueError(f"joint_ids_map must have length {num_actions}, got {joint_ids_map.size}.")

        if "stiffness" in deploy_cfg:
            stiffness_sdk = as_float_array(deploy_cfg["stiffness"], num_actions, "stiffness")
            overrides["kps"] = stiffness_sdk[joint_ids_map]
        if "damping" in deploy_cfg:
            damping_sdk = as_float_array(deploy_cfg["damping"], num_actions, "damping")
            overrides["kds"] = damping_sdk[joint_ids_map]

    step_dt = deploy_cfg.get("step_dt")
    if step_dt is not None:
        overrides["control_decimation"] = max(1, int(round(float(step_dt) * update_rate)))

    gait_phase_cfg = deploy_cfg.get("observations", {}).get("gait_phase")
    if gait_phase_cfg is not None:
        p = gait_phase_cfg.get("params", {})
        base_period = float(p.get("period", 0.72))

        start = p.get("period_start")
        end = p.get("period_end")

        overrides["gait_period"] = base_period
        overrides["gait_period_start"] = (
            base_period if start is None else float(start)
        )
        overrides["gait_period_end"] = (
            base_period if end is None else float(end)
        )
        overrides["gait_speed_start"] = float(p.get("speed_start", 2.0))
        overrides["gait_speed_end"] = float(p.get("speed_end", 2.8))

        if "command_threshold" in p:
            overrides["command_threshold"] = float(p["command_threshold"])

    overrides["deploy_path"] = deploy_path
    return overrides


def get_projected_gravity(quat_wxyz):
    quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float64)
    rotation = R.from_quat(quat_xyzw)
    gravity_world = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    return rotation.inv().apply(gravity_world).astype(np.float32)


def get_gait_phase(sim_time, period, cmd=None, command_threshold=0.1):
    if cmd is not None and np.linalg.norm(cmd) < command_threshold:
        return np.array([0.0, 1.0], dtype=np.float32)
    period = max(period, 1e-6)
    phase = 2.0 * np.pi * (sim_time % period) / period
    return np.array([np.sin(phase), np.cos(phase)], dtype=np.float32)


def get_command(cmd, sim_time, warmup_time):
    if warmup_time <= 0.0:
        return cmd
    scale = min(1.0, sim_time / warmup_time)
    return cmd * scale


class RobotIO:
    def __init__(self):
        self.imu_data = None
        self.robot_state = None

    def imu_callback(self, imu):
        self.imu_data = copy.deepcopy(imu)

    def robot_state_callback(self, robot_state):
        self.robot_state = copy.deepcopy(robot_state)


class LimxSDKPolicyController:
    def __init__(self, args):
        self.args = args
        self.script_dir = Path(__file__).resolve().parent
        self.config_path = resolve_path(self.script_dir, args.config)
        self.config = yaml.safe_load(self.config_path.read_text())

        policy_path = resolve_policy_path(
            self.config_path, self.config.get("policy_root"), args.policy or self.config["policy_path"]
        )
        if not policy_path.exists():
            raise FileNotFoundError(f"Policy file not found: {policy_path}")

        try:
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

        self.policy = torch.jit.load(str(policy_path), map_location="cpu")
        self.policy.eval()

        self.num_actions = int(self.config["num_actions"])
        self.num_obs = int(self.config["num_obs"])
        self.use_training_deploy_overrides = bool(self.config.get("use_training_deploy_overrides", False))
        deploy_overrides = {}
        if self.use_training_deploy_overrides:
            deploy_overrides = load_training_deploy_overrides(
                policy_path=policy_path, num_actions=self.num_actions, update_rate=int(self.config["update_rate"])
            )
            if "deploy_path" in deploy_overrides:
                print(f"[sdk-sim2sim] loaded training parameters from {deploy_overrides['deploy_path']}")
        else:
            print("[sdk-sim2sim] using repository sim2sim config; training deploy.yaml overrides disabled.")

        self.policy_joints = self.config["policy_joints"]
        self.policy_joint_index = {name: idx for idx, name in enumerate(self.policy_joints)}
        self.default_angles = as_float_array(
            deploy_overrides.get("default_angles", self.config["default_angles"]), self.num_actions, "default_angles"
        )
        self.kps = as_float_array(deploy_overrides.get("kps", self.config["kps"]), self.num_actions, "kps")
        self.kds = as_float_array(deploy_overrides.get("kds", self.config["kds"]), self.num_actions, "kds")
        self.action_scale = as_float_array(
            deploy_overrides.get("action_scale", self.config["action_scale"]), self.num_actions, "action_scale"
        )
        self.user_torque_limit = as_float_array(
            self.config["user_torque_limit"], self.num_actions, "user_torque_limit"
        )
        self.ang_vel_scale = float(self.config["ang_vel_scale"])
        self.dof_pos_scale = float(self.config["dof_pos_scale"])
        self.dof_vel_scale = float(self.config["dof_vel_scale"])
        self.cmd_scale = np.asarray(self.config["cmd_scale"], dtype=np.float32)
        self.control_decimation = int(deploy_overrides.get("control_decimation", self.config["control_decimation"]))
        self.update_rate = int(self.config["update_rate"])
        self.gait_period = float(deploy_overrides.get("gait_period", self.config["gait_period"]))
        print(f"[CHECK] deployed gait_period={self.gait_period:.6f} s")

        self.gait_period_start = float(
            deploy_overrides.get(
                "gait_period_start",
                self.config.get("gait_period_start", self.gait_period),
            )
        )
        self.gait_period_end = float(
            deploy_overrides.get(
                "gait_period_end",
                self.config.get("gait_period_end", self.gait_period),
            )
        )
        self.gait_speed_start = float(
            deploy_overrides.get(
                "gait_speed_start", self.config.get("gait_speed_start", 2.0)
            )
        )
        self.gait_speed_end = float(
            deploy_overrides.get(
                "gait_speed_end", self.config.get("gait_speed_end", 2.8)
            )
        )

        if min(self.gait_period_start, self.gait_period_end) <= 0.0:
            raise ValueError("Gait periods must be positive.")
        if self.gait_speed_end <= self.gait_speed_start:
            raise ValueError("gait_speed_end must exceed gait_speed_start.")

        self._gait_phase_fraction = 0.0
        self._gait_last_time = None

        print(
            "[CHECK] dynamic gait: "
            f"period={self.gait_period_start:.3f}"
            f"->{self.gait_period_end:.3f} s, "
            f"speed={self.gait_speed_start:.3f}"
            f"->{self.gait_speed_end:.3f} m/s"
        )

        self.command_threshold = float(deploy_overrides.get("command_threshold", self.config.get("command_threshold", 0.1)))
        self.command_warmup_s = float(self.config.get("command_warmup_s", 0.0))
        self.parallel_solve_required = bool(self.config.get("parallel_solve_required", True))
        self.clip_observations = float(self.config.get("clip_observations", 100.0))
        self.clip_actions = float(self.config.get("clip_actions", 100.0))

        self.cmd = np.array(self.config["cmd_init"], dtype=np.float32)
        if args.lin_vel_x is not None:
            self.cmd[0] = args.lin_vel_x
        if args.lin_vel_y is not None:
            self.cmd[1] = args.lin_vel_y
        if args.ang_vel_z is not None:
            self.cmd[2] = args.ang_vel_z

        self.io = RobotIO()
        self.robot = Robot(RobotType.Humanoid)
        if not self.robot.init(self.config.get("robot_ip", "127.0.0.1")):
            raise RuntimeError("Failed to initialize limxsdk Robot. Check robot_ip and simulator.")

        self.robot.subscribeImuData(self.io.imu_callback)
        self.robot.subscribeRobotState(self.io.robot_state_callback)

        self.last_action = np.zeros(self.num_actions, dtype=np.float32)
        self.loop_count = 0
        self.robot_order_index = None

    def wait_for_first_state(self):
        print("[sdk-sim2sim] waiting for robot state and imu data...")
        while self.io.robot_state is None or self.io.imu_data is None:
            time.sleep(0.01)

        motor_names = list(self.io.robot_state.motor_names)
        missing = [name for name in self.policy_joints if name not in motor_names]
        if missing:
            raise RuntimeError(
                "The simulator did not provide PR-space joint names. "
                "Make sure you are running ~/limx_ws/humanoid-mujoco-sim/simulator.py "
                "with kinematic_projection enabled. Missing joints: "
                f"{missing}"
            )

        self.robot_order_index = [motor_names.index(name) for name in self.policy_joints]
        print("[sdk-sim2sim] connected to simulator in PR space.")

    def compute_gait_phase(self, current_cmd, sim_time):
        speed = abs(float(current_cmd[0]))
        alpha = float(np.clip(
            (speed - self.gait_speed_start)
            / (self.gait_speed_end - self.gait_speed_start),
            0.0,
            1.0,
        ))
        period = (
            self.gait_period_start
            + alpha * (self.gait_period_end - self.gait_period_start)
        )

        now = float(sim_time)

        if self._gait_last_time is None or now < self._gait_last_time:
            # Initial call or controller clock reset.
            self._gait_phase_fraction = 0.0
        else:
            elapsed = now - self._gait_last_time
            self._gait_phase_fraction = (
                self._gait_phase_fraction + elapsed / period
            ) % 1.0

        self._gait_last_time = now

        # Match training: hide phase while standing,
        # but keep the internal accumulator advancing.
        if np.linalg.norm(current_cmd) < self.command_threshold:
            return np.array([0.0, 1.0], dtype=np.float32)

        angle = 2.0 * np.pi * self._gait_phase_fraction
        return np.array(
            [np.sin(angle), np.cos(angle)], dtype=np.float32
        )

    def build_observation(self, robot_state, imu_data, sim_time):
        joint_pos = np.asarray(robot_state.q, dtype=np.float32)[self.robot_order_index]
        joint_vel = np.asarray(robot_state.dq, dtype=np.float32)[self.robot_order_index]
        current_cmd = get_command(self.cmd, sim_time, self.command_warmup_s)

        obs = np.zeros(self.num_obs, dtype=np.float32)
        obs[:3] = np.asarray(imu_data.gyro, dtype=np.float32) * self.ang_vel_scale
        obs[3:6] = get_projected_gravity(imu_data.quat)
        obs[6:9] = current_cmd * self.cmd_scale
        obs[9 : 9 + self.num_actions] = (joint_pos - self.default_angles) * self.dof_pos_scale
        obs[9 + self.num_actions : 9 + 2 * self.num_actions] = joint_vel * self.dof_vel_scale
        obs[9 + 2 * self.num_actions : 9 + 3 * self.num_actions] = self.last_action
        obs[9 + 3 * self.num_actions : 9 + 3 * self.num_actions + 2] = (
            self.compute_gait_phase(current_cmd, sim_time)
        )
        return np.clip(obs, -self.clip_observations, self.clip_observations)

    def compute_action(self, obs):
        # 测量真实的策略推理间隔，而不是名义上的 20 ms
        now = time.perf_counter()
        previous = getattr(self, "_diag_previous_policy_time", None)
        self._diag_policy_dt = (
            float("nan") if previous is None else now - previous
        )
        self._diag_previous_policy_time = now

        with torch.inference_mode():
            obs_tensor = torch.from_numpy(obs).unsqueeze(0)
            action = self.policy(obs_tensor)
            if isinstance(action, (tuple, list)):
                action = action[0]
            action = action.detach().cpu().numpy().squeeze().astype(np.float32)
        return np.clip(action, -self.clip_actions, self.clip_actions)

    def publish_command(self, robot_state):
        motor_names = list(robot_state.motor_names)
        motor_count = len(motor_names)
        joint_pos = np.asarray(robot_state.q, dtype=np.float32)[self.robot_order_index]
        joint_vel = np.asarray(robot_state.dq, dtype=np.float32)[self.robot_order_index]
        safe_scale = np.maximum(self.action_scale, 1e-6)
        safe_kp = np.maximum(self.kps, 1e-6)
        soft_torque_limit = 0.95

        action_min = joint_pos - self.default_angles + (
            self.kds * joint_vel - self.user_torque_limit * soft_torque_limit
        ) / safe_kp
        action_max = joint_pos - self.default_angles + (
            self.kds * joint_vel + self.user_torque_limit * soft_torque_limit
        ) / safe_kp
        limited_action = np.clip(self.last_action, action_min / safe_scale, action_max / safe_scale)
        desired_q_policy = limited_action * self.action_scale + self.default_angles

        # 只记录数据，不修改 action、目标或力矩。
        if not hasattr(self, "_diag_csv"):
            self._diag_file = open(
                "controller_diag.csv", "w", newline="", buffering=1
            )
            self._diag_csv = csv.writer(self._diag_file)
            self._diag_csv.writerow([
                "wall_time_s", "cmd_vx", "policy_dt_s",
                "joint", "q", "dq", "action",
                "q_raw", "q_limited", "target_clip_rad",
                "tau_requested", "tau_after_target_limit", "tau_limit",
            ])
            self._diag_next_time = 0.0

        now = time.perf_counter()
        if now >= self._diag_next_time:
            self._diag_next_time = now + 0.02  # 约 50 Hz

            raw_q = self.default_angles + self.last_action * self.action_scale
            tau_raw = self.kps * (raw_q - joint_pos) - self.kds * joint_vel
            tau_limited = (
                self.kps * (desired_q_policy - joint_pos)
                - self.kds * joint_vel
            )

            watched = [
                "left_ankle_pitch_joint", "right_ankle_pitch_joint",
                "left_ankle_roll_joint", "right_ankle_roll_joint",
                "left_hip_pitch_joint", "right_hip_pitch_joint",
            ]
            wall_time = time.time()
            for name in watched:
                i = self.policy_joint_index[name]
                self._diag_csv.writerow([
                    wall_time, float(self.cmd[0]),
                    getattr(self, "_diag_policy_dt", float("nan")),
                    name,
                    float(joint_pos[i]), float(joint_vel[i]),
                    float(self.last_action[i]),
                    float(raw_q[i]), float(desired_q_policy[i]),
                    float(abs(raw_q[i] - desired_q_policy[i])),
                    float(tau_raw[i]), float(tau_limited[i]),
                    float(0.95 * self.user_torque_limit[i]),
                ])

        # Diagnostic only: does not change commands or limits.
        if not hasattr(self, "_limit_diag_count"):
            self._limit_diag_count = 0
            self._limit_diag_hits = np.zeros(
                self.num_actions, dtype=np.float64
            )
            self._limit_diag_peak = np.zeros(
                self.num_actions, dtype=np.float64
            )
            self._limit_diag_last_print = time.perf_counter()

        changed = np.abs(self.last_action - limited_action) > 1e-5
        target_change_rad = np.abs(
            (self.last_action - limited_action) * self.action_scale
        )

        self._limit_diag_count += 1
        self._limit_diag_hits += changed
        self._limit_diag_peak = np.maximum(
            self._limit_diag_peak, target_change_rad
        )

        now = time.perf_counter()
        if now - self._limit_diag_last_print >= 1.0:
            rates = self._limit_diag_hits / self._limit_diag_count
            top = np.argsort(rates)[::-1][:4]

            details = ", ".join(
                f"{self.policy_joints[i]}:"
                f"limited={rates[i]:.1%},"
                f"max_dq={self._limit_diag_peak[i]:.3f}rad"
                for i in top
                if rates[i] > 0.0
            )

            print(
                "[LIMIT] "
                f"cmd_vx={self.cmd[0]:.2f} "
                + (details or "no target limiting")
            )

            self._limit_diag_count = 0
            self._limit_diag_hits.fill(0.0)
            self._limit_diag_peak.fill(0.0)
            self._limit_diag_last_print = now

        cmd_msg = datatypes.RobotCmd()
        cmd_msg.stamp = time.time_ns()
        cmd_msg.mode = [0 for _ in range(motor_count)]
        cmd_msg.q = [float(x) for x in robot_state.q]
        cmd_msg.dq = [0.0 for _ in range(motor_count)]
        cmd_msg.tau = [0.0 for _ in range(motor_count)]
        cmd_msg.Kp = [0.0 for _ in range(motor_count)]
        cmd_msg.Kd = [0.0 for _ in range(motor_count)]
        cmd_msg.motor_names = motor_names
        if hasattr(cmd_msg, "parallel_solve_required"):
            cmd_msg.parallel_solve_required = [self.parallel_solve_required for _ in range(motor_count)]

        for policy_idx, joint_name in enumerate(self.policy_joints):
            robot_idx = self.robot_order_index[policy_idx]
            cmd_msg.q[robot_idx] = float(desired_q_policy[policy_idx])
            cmd_msg.Kp[robot_idx] = float(self.kps[policy_idx])
            cmd_msg.Kd[robot_idx] = float(self.kds[policy_idx])

        self.robot.publishRobotCmd(cmd_msg)

    def run(self):
        self.wait_for_first_state()
        rate = Rate(self.update_rate)
        t0 = time.perf_counter()

        while True:
            robot_state = copy.deepcopy(self.io.robot_state)
            imu_data = copy.deepcopy(self.io.imu_data)
            sim_time = self.loop_count / float(self.update_rate)

            if self.loop_count % self.control_decimation == 0:
                obs = self.build_observation(robot_state, imu_data, sim_time)
                self.last_action = self.compute_action(obs)

            self.publish_command(robot_state)

            self.loop_count += 1
            if self.loop_count % self.update_rate == 0:
                fps = self.loop_count / max(1e-6, (time.perf_counter() - t0))
                print(
                    f"[sdk-sim2sim] fps={fps:.1f} "
                    f"cmd=({self.cmd[0]:+.2f}, {self.cmd[1]:+.2f}, {self.cmd[2]:+.2f})",
                    end="\r",
                )
            rate.sleep()


def parse_args():
    parser = argparse.ArgumentParser(description="LimX sdk-based sim2sim controller for the official MuJoCo simulator.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/limx_flat_velocity_sdk.yaml",
        help="Path to sdk sim2sim yaml config.",
    )
    parser.add_argument("--policy", type=str, default=None, help="Optional override path to exported TorchScript policy.")
    parser.add_argument("--lin-vel-x", type=float, default=0.5, help="Override forward velocity command.")
    parser.add_argument("--lin-vel-y", type=float, default=None, help="Override lateral velocity command.")
    parser.add_argument("--ang-vel-z", type=float, default=None, help="Override yaw velocity command.")
    return parser.parse_args()


if __name__ == "__main__":
    controller = LimxSDKPolicyController(parse_args())
    controller.run()
