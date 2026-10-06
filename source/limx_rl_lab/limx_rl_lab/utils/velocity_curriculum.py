"""Pure helpers for high-speed velocity curriculum and command sampling mix."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class SpeedGateResult:
    gates_pass: bool
    completion_pass: bool
    orientation_pass: bool
    rmse_pass: bool
    saturation_pass: bool
    completion_rate: float
    orientation_term_rate: float
    speed_rmse_mps: float
    saturation_rate: float
    reason: str = ""


def evaluate_speed_gates(
    completion_rate: float,
    orientation_term_rate: float,
    speed_rmse_mps: float,
    saturation_rate: float = 0.0,
    completion_rate_min: float = 0.95,
    orientation_term_rate_max: float = 0.02,
    speed_rmse_max_mps: float = 0.25,
    saturation_rate_max: float = 0.20,
) -> SpeedGateResult:
    # inclusive thresholds for round-2 frontier gates
    completion_pass = completion_rate >= completion_rate_min
    orientation_pass = orientation_term_rate <= orientation_term_rate_max
    rmse_pass = speed_rmse_mps <= speed_rmse_max_mps
    saturation_pass = saturation_rate <= saturation_rate_max
    gates_pass = completion_pass and orientation_pass and rmse_pass and saturation_pass
    reasons = []
    if not completion_pass:
        reasons.append("completion")
    if not orientation_pass:
        reasons.append("orientation")
    if not rmse_pass:
        reasons.append("rmse")
    if not saturation_pass:
        reasons.append("saturation")
    return SpeedGateResult(
        gates_pass=gates_pass,
        completion_pass=completion_pass,
        orientation_pass=orientation_pass,
        rmse_pass=rmse_pass,
        saturation_pass=saturation_pass,
        completion_rate=float(completion_rate),
        orientation_term_rate=float(orientation_term_rate),
        speed_rmse_mps=float(speed_rmse_mps),
        saturation_rate=float(saturation_rate),
        reason=",".join(reasons),
    )


@dataclass
class FrontierIterStats:
    """One-iteration frontier sample statistics."""

    completion_rate: float
    orientation_rate: float
    rmse_mps: float
    saturation_rate: float
    joint_vel_sat_rate: float = 0.0
    joint_torque_sat_rate: float = 0.0
    action_clip_ratio: float = 0.0
    commanded_vx_mps: float = 0.0
    measured_vx_mps: float = 0.0
    mixed_rmse_mps: float = 0.0
    n_frontier: int = 0


@dataclass
class CurriculumState:
    current_max_speed_mps: float
    target_speed_mps: float = 2.4
    speed_increment_mps: float = 0.1
    # rolling window over PPO/env iterations
    rolling_window: int = 200
    consecutive_pass_required: int = 200
    dwell_min_iterations: int = 300
    # round-2 frontier gates
    completion_rate_min: float = 0.97
    orientation_term_rate_max: float = 0.02
    speed_rmse_max_mps: float = 0.28
    saturation_rate_max: float = 0.10
    history: deque = field(default_factory=lambda: deque(maxlen=200))
    consecutive_pass_iterations: int = 0
    iterations_at_current_speed: int = 0
    frozen: bool = False
    freeze_reason: str = ""
    level: int = 0
    promoted_count: int = 0
    last_gate_reason: str = ""
    last_completion_rate: float = 0.0
    last_orientation_term_rate: float = 0.0
    last_rmse_mps: float = 0.0
    last_saturation_rate: float = 0.0
    last_joint_vel_sat_rate: float = 0.0
    last_joint_torque_sat_rate: float = 0.0
    last_action_clip_ratio: float = 0.0
    last_commanded_vx_mps: float = 0.0
    last_measured_vx_mps: float = 0.0
    last_mixed_rmse_mps: float = 0.0
    last_n_frontier: int = 0
    # deprecated alias kept for older tests/docs
    stable_min_steps: int = 300
    fail_streak_freeze: int = 2
    log: list[str] | None = None
    # speed-bucketed RMSE: bucket -> running list of rmse
    speed_buckets: dict[float, list[float]] = field(default_factory=dict)

    def __post_init__(self):
        if self.history.maxlen != self.rolling_window:
            self.history = deque(self.history, maxlen=self.rolling_window)


def frontier_mask(vx_cmd, current_max_speed_mps: float, min_frac: float = 1.0):
    """Boolean mask for frontier samples: vx_cmd >= current_max * min_frac."""
    import torch

    if not torch.is_tensor(vx_cmd):
        vx_cmd = torch.as_tensor(vx_cmd, dtype=torch.float32)
    thr = float(current_max_speed_mps) * float(min_frac)
    return vx_cmd >= thr


def compute_frontier_stats(
    vx_cmd,
    vx_meas,
    completion_rate: float,
    orientation_rate: float,
    joint_vel_sat_rate: float = 0.0,
    joint_torque_sat_rate: float = 0.0,
    action_clip_ratio: float = 0.0,
    current_max_speed_mps: float | None = None,
    min_frac: float = 1.0,
) -> FrontierIterStats:
    """Compute frontier-only and mixed metrics for one iteration."""
    import torch

    if not torch.is_tensor(vx_cmd):
        vx_cmd = torch.as_tensor(vx_cmd, dtype=torch.float32).reshape(-1)
    if not torch.is_tensor(vx_meas):
        vx_meas = torch.as_tensor(vx_meas, dtype=torch.float32).reshape(-1)
    n = min(vx_cmd.numel(), vx_meas.numel())
    if n == 0:
        return FrontierIterStats(
            completion_rate=completion_rate,
            orientation_rate=orientation_rate,
            rmse_mps=0.0,
            saturation_rate=max(joint_vel_sat_rate, joint_torque_sat_rate),
            joint_vel_sat_rate=joint_vel_sat_rate,
            joint_torque_sat_rate=joint_torque_sat_rate,
            action_clip_ratio=action_clip_ratio,
            n_frontier=0,
        )
    vx_cmd = vx_cmd[:n]
    vx_meas = vx_meas[:n]
    mixed_rmse = float(torch.sqrt(torch.mean((vx_cmd - vx_meas) ** 2)).item())

    thr = 0.0 if current_max_speed_mps is None else float(current_max_speed_mps) * float(min_frac)
    mask = vx_cmd >= thr
    n_frontier = int(mask.sum().item())
    if n_frontier > 0:
        rmse = float(torch.sqrt(torch.mean((vx_cmd[mask] - vx_meas[mask]) ** 2)).item())
        cmd_f = float(vx_cmd[mask].mean().item())
        meas_f = float(vx_meas[mask].mean().item())
    else:
        # no frontier samples this step: fall back to mixed (logged separately)
        rmse = mixed_rmse
        cmd_f = float(vx_cmd.mean().item())
        meas_f = float(vx_meas.mean().item())

    sat = max(joint_vel_sat_rate, joint_torque_sat_rate)
    return FrontierIterStats(
        completion_rate=float(completion_rate),
        orientation_rate=float(orientation_rate),
        rmse_mps=rmse,
        saturation_rate=float(sat),
        joint_vel_sat_rate=float(joint_vel_sat_rate),
        joint_torque_sat_rate=float(joint_torque_sat_rate),
        action_clip_ratio=float(action_clip_ratio),
        commanded_vx_mps=cmd_f,
        measured_vx_mps=meas_f,
        mixed_rmse_mps=mixed_rmse,
        n_frontier=n_frontier,
    )


def rolling_mean(history: deque, key: str) -> float:
    if not history:
        return 0.0
    vals = [getattr(item, key) for item in history]
    return float(sum(vals) / len(vals))


def evaluate_rolling_frontier_gates(state: CurriculumState) -> SpeedGateResult:
    """Evaluate promotion gates on rolling means of available frontier history."""
    hist = state.history
    if len(hist) < state.rolling_window:
        return SpeedGateResult(
            gates_pass=False,
            completion_pass=False,
            orientation_pass=False,
            rmse_pass=False,
            saturation_pass=False,
            completion_rate=0.0,
            orientation_term_rate=0.0,
            speed_rmse_mps=0.0,
            saturation_rate=0.0,
            reason="window_incomplete",
        )
    completion = rolling_mean(hist, "completion_rate")
    orient = rolling_mean(hist, "orientation_rate")
    rmse = rolling_mean(hist, "rmse_mps")
    sat = rolling_mean(hist, "saturation_rate")
    return evaluate_speed_gates(
        completion_rate=completion,
        orientation_term_rate=orient,
        speed_rmse_mps=rmse,
        saturation_rate=sat,
        completion_rate_min=state.completion_rate_min,
        orientation_term_rate_max=state.orientation_term_rate_max,
        speed_rmse_max_mps=state.speed_rmse_max_mps,
        saturation_rate_max=state.saturation_rate_max,
    )


def apply_rolling_frontier_step(
    state: CurriculumState,
    iter_stats: FrontierIterStats,
) -> CurriculumState:
    """Append one iteration of frontier stats, update consecutive pass, maybe promote.

    - Gate stats = rolling means over last `rolling_window` frontier iterations.
    - Promote requires `consecutive_pass_required` consecutive passing rolling evaluations
      AND `dwell_min_iterations` at current speed.
    - On gate failure: hold current max; set frozen; do NOT auto-degrade.
    """
    if state.log is None:
        state.log = []

    state.history.append(iter_stats)
    state.last_gate_reason = ""
    state.last_completion_rate = iter_stats.completion_rate
    state.last_orientation_term_rate = iter_stats.orientation_rate
    state.last_rmse_mps = iter_stats.rmse_mps
    state.last_saturation_rate = iter_stats.saturation_rate
    state.last_joint_vel_sat_rate = iter_stats.joint_vel_sat_rate
    state.last_joint_torque_sat_rate = iter_stats.joint_torque_sat_rate
    state.last_action_clip_ratio = iter_stats.action_clip_ratio
    state.last_commanded_vx_mps = iter_stats.commanded_vx_mps
    state.last_measured_vx_mps = iter_stats.measured_vx_mps
    state.last_mixed_rmse_mps = iter_stats.mixed_rmse_mps
    state.last_n_frontier = iter_stats.n_frontier

    if iter_stats.n_frontier > 0 or iter_stats.commanded_vx_mps > 0:
        bucket = round(iter_stats.commanded_vx_mps * 10.0) / 10.0
        state.speed_buckets.setdefault(bucket, []).append(iter_stats.rmse_mps)

    state.iterations_at_current_speed += 1

    if state.current_max_speed_mps >= state.target_speed_mps - 1e-6:
        state.consecutive_pass_iterations = 0
        return state

    gate = evaluate_rolling_frontier_gates(state)
    state.last_gate_reason = gate.reason

    if gate.gates_pass:
        state.consecutive_pass_iterations += 1
        dwell_ok = state.iterations_at_current_speed >= state.dwell_min_iterations
        if state.consecutive_pass_iterations >= state.consecutive_pass_required and dwell_ok:
            old = state.current_max_speed_mps
            state.current_max_speed_mps = min(
                state.target_speed_mps,
                round(old + state.speed_increment_mps, 6),
            )
            state.level += 1
            state.promoted_count += 1
            state.iterations_at_current_speed = 0
            state.consecutive_pass_iterations = 0
            state.history.clear()
            state.frozen = False
            state.freeze_reason = ""
            state.log.append(
                f"PROMOTE {old:.3f}->{state.current_max_speed_mps:.3f} "
                f"completion={gate.completion_rate:.3f} rmse={gate.speed_rmse_mps:.3f} "
                f"sat={gate.saturation_rate:.3f}"
            )
        return state

    # gate failed: hold current speed, no auto-degrade
    state.consecutive_pass_iterations = 0
    state.frozen = True
    state.freeze_reason = gate.reason or "rolling_frontier"
    state.log.append(
        f"HOLD max={state.current_max_speed_mps:.3f} reason={state.freeze_reason} "
        f"completion={gate.completion_rate:.3f} orient={gate.orientation_term_rate:.3f} "
        f"rmse={gate.speed_rmse_mps:.3f} sat={gate.saturation_rate:.3f}"
    )
    return state


# Back-compat wrapper used by older unit tests / docs
def apply_speed_promotion(
    state: CurriculumState,
    gate: SpeedGateResult,
    current_step: int,
) -> CurriculumState:
    """Legacy helper: evaluate an instantaneous gate against rolling state.

    For rolling curriculum prefer apply_rolling_frontier_step.
    """
    if state.log is None:
        state.log = []
    state.last_gate_reason = gate.reason
    state.last_completion_rate = gate.completion_rate
    state.last_orientation_term_rate = gate.orientation_term_rate
    state.last_rmse_mps = gate.speed_rmse_mps
    state.last_saturation_rate = gate.saturation_rate

    if state.current_max_speed_mps >= state.target_speed_mps - 1e-6:
        return state

    if gate.gates_pass:
        state.consecutive_pass_iterations += 1
        dwell_ok = state.iterations_at_current_speed >= state.dwell_min_iterations
        if state.consecutive_pass_iterations >= state.consecutive_pass_required and dwell_ok:
            old = state.current_max_speed_mps
            state.current_max_speed_mps = min(
                state.target_speed_mps,
                round(old + state.speed_increment_mps, 6),
            )
            state.level += 1
            state.promoted_count += 1
            state.iterations_at_current_speed = 0
            state.consecutive_pass_iterations = 0
            state.frozen = False
            state.log.append(f"PROMOTE {old:.3f}->{state.current_max_speed_mps:.3f} step={current_step}")
        return state

    state.consecutive_pass_iterations = 0
    state.frozen = True
    state.freeze_reason = gate.reason or "unknown"
    state.log.append(f"HOLD max={state.current_max_speed_mps:.3f} reason={state.freeze_reason} step={current_step}")
    return state


def sample_mixed_velocity_commands(
    num_envs: int,
    lin_vel_x_max: float,
    device,
    lin_vel_x_min: float = -0.5,
    lin_vel_y_range: tuple[float, float] = (-0.05, 0.05),
    ang_vel_z_range: tuple[float, float] = (-0.25, 0.25),
    near_frac: float = 0.5,
    mid_frac: float = 0.35,
    low_frac: float = 0.15,
    low_stand_max: float = 0.2,
    generator=None,
):
    """Sample mixed speed commands with configurable mix fractions.

    Straight-line high-speed phase: y near 0 and yaw within ang_vel_z_range.
    Returns Tensor (num_envs, 3) = [vx, vy, wz].
    """
    import torch

    if generator is None:
        generator = torch.Generator(device="cpu")
    near_n = int(round(num_envs * near_frac))
    mid_n = int(round(num_envs * mid_frac))
    low_n = num_envs - near_n - mid_n
    if low_n < 0:
        low_n = 0
        mid_n = num_envs - near_n

    def _uniform(lo, hi, n):
        if n <= 0:
            return torch.empty(0, device=device)
        u = torch.rand(n, generator=generator)
        return (lo + (hi - lo) * u).to(device=device, dtype=torch.float32)

    near_hi = max(lin_vel_x_max, 0.0)
    # near cap includes the cap itself so frontier_mask (>= max) is populated
    near_lo = max(lin_vel_x_min, 0.9 * near_hi) if near_hi > 0 else lin_vel_x_min
    vx_near = _uniform(near_lo, near_hi, near_n)
    if near_n > 0 and near_hi > 0:
        # force a subset exactly at cap for frontier coverage
        cap_n = max(1, int(round(0.5 * near_n)))
        vx_near[:cap_n] = near_hi

    mid_hi = max(0.8 * near_hi, near_lo)
    mid_lo = max(lin_vel_x_min, 0.3 * near_hi)
    if mid_hi < mid_lo:
        mid_hi = mid_lo
    vx_mid = _uniform(mid_lo, mid_hi, mid_n)
    vx_low = _uniform(0.0, max(low_stand_max, 0.05), low_n)

    vx = torch.cat([vx_near, vx_mid, vx_low], dim=0)
    if vx.numel() < num_envs:
        pad = torch.full((num_envs - vx.numel(),), float(lin_vel_x_min), device=device)
        vx = torch.cat([vx, pad], dim=0)
    vx = vx[:num_envs]

    y_lo, y_hi = lin_vel_y_range
    w_lo, w_hi = ang_vel_z_range
    vy = _uniform(y_lo, y_hi, num_envs)
    wz = _uniform(w_lo, w_hi, num_envs)
    return torch.stack([vx, vy, wz], dim=-1)


def gates_for_staged_eval(target_speed_mps: float) -> dict:
    """Stricter fixed-speed gates for staged high-speed acceptance (>=2.4 m/s)."""
    if target_speed_mps >= 2.4 - 1e-6:
        return {
            "completion_rate_min": 0.97,
            "orientation_term_rate_max": 0.02,
            "speed_rmse_max_mps": 0.25,
        }
    return {
        "completion_rate_min": 0.95,
        "orientation_term_rate_max": 0.02,
        "speed_rmse_max_mps": 0.25,
    }
