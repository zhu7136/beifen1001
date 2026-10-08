import sys
import numpy as np
from gamepad_policy_controller import GamepadPolicyController, parse_args

PROFILE = sys.argv.pop(1) if len(sys.argv) > 1 else ""
if PROFILE not in ("baseline", "mid", "slow", "fast"):
    raise SystemExit(
        "Usage: python fixed_command_test.py baseline|mid|slow|fast --policy PATH"
    )


def trajectory(t, profile):
    target = {
        "baseline": 2.25,
        "mid": 2.30,
        "slow": 2.35,
        "fast": 2.35,
    }[profile]
    ramp = 1.0 if profile == "fast" else 0.2

    # 仿真时间：零指令 8 秒，10 秒升到 2.0，保持 5 秒
    stages = [(8.0, 0.0, 0.0), (10.0, 0.0, 2.0), (5.0, 2.0, 2.0)]
    if profile != "fast":
        stages += [(0.65, 2.0, 2.13), (5.0, 2.13, 2.13)]
        start = 2.13
    else:
        start = 2.0

    stages += [
        ((target - start) / ramp, start, target),
        (30.0, target, target),
        (target / 0.7, target, 0.0),
        (5.0, 0.0, 0.0),
    ]
    for duration, a, b in stages:
        if t < duration:
            return a + (b - a) * max(t, 0.0) / duration, False
        t -= duration
    return 0.0, True


class FixedTest(GamepadPolicyController):
    def __init__(self, args):
        super().__init__(args)
        # 忽略手柄输入；速度完全由测试轨迹决定
        self.teleop.read_command = lambda: np.zeros(3, dtype=np.float32)
        self.teleop.consume_mode_request = lambda: None
        self.test_prev = None
        self.test_origin = None
        self.bad_since = None
        self.abort = None
        self.report_at = 0.0

    def _run_walk_step(self, robot_state, imu_data, sim_time):
        if self.test_prev is not None and sim_time < self.test_prev:
            self.test_origin = sim_time
            self.bad_since = self.abort = None
            self.report_at = 0.0
            self._test_finished = False
            print(f"[TEST] RESET: starting {PROFILE} profile", flush=True)
        self.test_prev = sim_time

        # 必须连接后再 Reset，才开始自动升速
        if self.test_origin is None:
            self.cmd.fill(0.0)
            return super()._run_walk_step(robot_state, imu_data, sim_time)

        t = sim_time - self.test_origin
        vx, done = trajectory(t, PROFILE)

        q = np.asarray(imu_data.quat, dtype=float)
        norm = np.linalg.norm(q)
        if not np.isfinite(norm) or norm < 1e-6:
            raise RuntimeError("Invalid IMU quaternion; inspect simulator")
        w, x, y, z = q / norm
        roll = np.degrees(
            np.arctan2(2 * (w*x + y*z), 1 - 2 * (x*x + y*y))
        )
        pitch = np.degrees(
            np.arcsin(np.clip(2 * (w*y - z*x), -1, 1))
        )

        # 这些是本次测试的中止阈值，不是已验证的安全边界
        bad = abs(roll) > 10 or abs(pitch) > 12
        if t >= 5 and bad:
            if self.bad_since is None:
                self.bad_since = sim_time
        else:
            self.bad_since = None

        trigger = (
            self.bad_since is not None
            and sim_time - self.bad_since >= 0.2
        )
        if t >= 5 and (abs(roll) > 20 or abs(pitch) > 20):
            trigger = True

        if trigger and self.abort is None:
            self.abort = (sim_time, vx)
            print(
                f"[TEST] ABORT t={t:.3f}: "
                f"roll={roll:.2f}, pitch={pitch:.2f}",
                flush=True,
            )

        if self.abort is not None:
            elapsed = sim_time - self.abort[0]
            vx = max(0.0, self.abort[1] - 0.7 * elapsed)
            done = elapsed >= self.abort[1] / 0.7 + 5.0

        self.cmd[:] = [vx, 0.0, 0.0]
        self.target_cmd[:] = self.cmd

        if t >= self.report_at:
            print(
                f"[TEST] t={t:.2f} vx={vx:.3f} "
                f"roll={roll:.2f} pitch={pitch:.2f}",
                flush=True,
            )
            self.report_at = t + 1.0

        super()._run_walk_step(robot_state, imu_data, sim_time)
        if done and not getattr(self, "_test_finished", False):
            self._test_finished = True
            print(
                "[TEST] DONE: "
                + ("aborted" if self.abort else "completed")
                + "; continuing zero-command control",
                flush=True,
            )


if __name__ == "__main__":
    controller = FixedTest(parse_args())
    print(
        "[TEST] Simulation only. Reset MuJoCo AFTER controller connects.",
        flush=True,
    )
    try:
        controller.run()
    except KeyboardInterrupt:
        pass
    finally:
        controller.close()
