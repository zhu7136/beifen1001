"""Pure helpers for fixed-speed velocity evaluation metrics and pass gates."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


DEFAULT_SPEEDS_MPS = (1.0, 1.4, 1.8, 2.2, 2.5, 2.78)
DEFAULT_GATES = {
    "completion_rate_min": 0.95,
    "orientation_term_rate_max": 0.02,
    "speed_rmse_max_mps": 0.25,
}


@dataclass
class SpeedGates:
    completion_rate_min: float = 0.95
    orientation_term_rate_max: float = 0.02
    speed_rmse_max_mps: float = 0.25

    @classmethod
    def from_dict(cls, data: dict | None) -> "SpeedGates":
        if not data:
            return cls()
        return cls(
            completion_rate_min=float(data.get("completion_rate_min", 0.95)),
            orientation_term_rate_max=float(data.get("orientation_term_rate_max", 0.02)),
            speed_rmse_max_mps=float(data.get("speed_rmse_max_mps", 0.25)),
        )

    @classmethod
    def for_target(cls, target_speed_mps: float) -> "SpeedGates":
        """Staged high-speed acceptance: stricter gates when target >= 2.4 m/s."""
        if target_speed_mps >= 2.4 - 1e-6:
            return cls(completion_rate_min=0.97, orientation_term_rate_max=0.02, speed_rmse_max_mps=0.25)
        return cls()


@dataclass
class RunMetrics:
    """Per-seed fixed-speed evaluation record."""

    target_speed_mps: float
    seed: int
    num_episodes: int
    mean_vx_mps: float
    rmse_vx_mps: float
    completion_rate: float
    bad_orientation_rate: float
    base_contact_rate: float
    base_height_rate: float
    foot_slide_mean: float
    joint_vel_saturation_rate: float
    joint_torque_saturation_rate: float
    action_clip_ratio: float
    mean_joint_power_w: float
    cost_of_transport: float
    extras: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        data = asdict(self)
        return data


def compute_run_metrics(
    target_speed_mps: float,
    seed: int,
    vx_cmd,
    vx_meas,
    timed_out,
    terminated,
    bad_orientation=None,
    base_contact=None,
    base_height=None,
    foot_slide=None,
    joint_vel_sat=None,
    joint_torque_sat=None,
    action_clip_ratio=0.0,
    joint_power=None,
    robot_mass_kg=None,
    g: float = 9.81,
) -> RunMetrics:
    """Compute fixed-speed run metrics from per-step / per-episode tensors.

    Args:
        vx_cmd: commanded forward velocity samples (1D).
        vx_meas: measured forward velocity samples (1D).
        timed_out: per-episode bool, True if episode ran to timeout.
        terminated: per-episode bool, True if any non-timeout termination fired.
        bad_orientation / base_contact / base_height: optional per-episode rates (0..1)
            or bool arrays; if None, rates are 0.0.
        foot_slide: optional scalar mean foot-slide magnitude.
        joint_vel_sat / joint_torque_sat: optional scalars in [0, 1].
        action_clip_ratio: optional scalar in [0, 1].
        joint_power: optional scalar mean mechanical power (W).
        robot_mass_kg: optional robot mass for cost-of-transport.
    """
    vx_cmd = _to_float_list(vx_cmd)
    vx_meas = _to_float_list(vx_meas)
    n = max(len(vx_cmd), len(vx_meas))
    if n == 0:
        mean_vx = 0.0
        rmse = 0.0
    else:
        # pad shorter series by repeating last value for stable stats
        if len(vx_cmd) < n:
            vx_cmd = vx_cmd + [vx_cmd[-1]] * (n - len(vx_cmd))
        if len(vx_meas) < n:
            vx_meas = vx_meas + [vx_meas[-1]] * (n - len(vx_meas))
        mean_vx = sum(vx_meas) / n
        rmse = (sum((c - m) ** 2 for c, m in zip(vx_cmd, vx_meas)) / n) ** 0.5

    timed_out_list = _to_float_list(timed_out)
    terminated_list = _to_float_list(terminated)
    num_episodes = max(len(timed_out_list), len(terminated_list), 1)
    if timed_out_list:
        completion_rate = sum(timed_out_list) / len(timed_out_list)
    else:
        completion_rate = 0.0
    if terminated_list:
        early_term_rate = sum(1.0 for t in terminated_list if t > 0.5) / len(terminated_list)
        # completion counts full episodes that were not early-terminated
        completion_rate = 1.0 - early_term_rate if not timed_out_list else completion_rate
    else:
        early_term_rate = 0.0

    mean_power = float(joint_power) if joint_power is not None else 0.0
    mean_speed_safe = max(abs(mean_vx), 1e-3)
    if robot_mass_kg:
        cot = mean_power / (float(robot_mass_kg) * g * mean_speed_safe)
    else:
        cot = 0.0

    return RunMetrics(
        target_speed_mps=float(target_speed_mps),
        seed=int(seed),
        num_episodes=int(num_episodes),
        mean_vx_mps=float(mean_vx),
        rmse_vx_mps=float(rmse),
        completion_rate=float(completion_rate),
        bad_orientation_rate=_rate(bad_orientation),
        base_contact_rate=_rate(base_contact),
        base_height_rate=_rate(base_height),
        foot_slide_mean=float(foot_slide) if foot_slide is not None else 0.0,
        joint_vel_saturation_rate=float(joint_vel_sat) if joint_vel_sat is not None else 0.0,
        joint_torque_saturation_rate=float(joint_torque_sat) if joint_torque_sat is not None else 0.0,
        action_clip_ratio=float(action_clip_ratio) if action_clip_ratio is not None else 0.0,
        mean_joint_power_w=mean_power,
        cost_of_transport=float(cot),
        extras={"early_term_rate": float(early_term_rate)},
    )


def summarize_speed(records: list[RunMetrics], gates: SpeedGates | None = None) -> dict:
    """Aggregate seed-level records into a per-speed summary with gate pass/fail."""
    if not records:
        return {
            "num_seeds": 0,
            "gates_pass": False,
            "initial_max_speed_candidate_mps": None,
        }
    gates = gates or SpeedGates()
    n = len(records)
    mean_vx = sum(r.mean_vx_mps for r in records) / n
    rmse = sum(r.rmse_vx_mps for r in records) / n
    completion = sum(r.completion_rate for r in records) / n
    orient = sum(r.bad_orientation_rate for r in records) / n
    contact = sum(r.base_contact_rate for r in records) / n
    height = sum(r.base_height_rate for r in records) / n
    foot_slide = sum(r.foot_slide_mean for r in records) / n
    jv_sat = sum(r.joint_vel_saturation_rate for r in records) / n
    tq_sat = sum(r.joint_torque_saturation_rate for r in records) / n
    clip = sum(r.action_clip_ratio for r in records) / n
    power = sum(r.mean_joint_power_w for r in records) / n
    cot = sum(r.cost_of_transport for r in records) / n

    gate_pass = (
        completion > gates.completion_rate_min
        and orient < gates.orientation_term_rate_max
        and rmse < gates.speed_rmse_max_mps
    )
    target = records[0].target_speed_mps
    return {
        "target_speed_mps": target,
        "num_seeds": n,
        "mean_vx_mps": mean_vx,
        "rmse_vx_mps": rmse,
        "completion_rate": completion,
        "bad_orientation_rate": orient,
        "base_contact_rate": contact,
        "base_height_rate": height,
        "foot_slide_mean": foot_slide,
        "joint_vel_saturation_rate": jv_sat,
        "joint_torque_saturation_rate": tq_sat,
        "action_clip_ratio": clip,
        "mean_joint_power_w": power,
        "cost_of_transport": cot,
        "gates": {
            "completion_rate_min": gates.completion_rate_min,
            "orientation_term_rate_max": gates.orientation_term_rate_max,
            "speed_rmse_max_mps": gates.speed_rmse_max_mps,
            "completion_pass": completion > gates.completion_rate_min,
            "orientation_pass": orient < gates.orientation_term_rate_max,
            "rmse_pass": rmse < gates.speed_rmse_max_mps,
        },
        "gates_pass": bool(gate_pass),
        "seeds": [r.to_dict() for r in records],
    }


def recommend_initial_max_speed(summaries: list[dict]) -> float | None:
    """Highest speed with gates_pass, else lowest tested speed that completed reasonably."""
    passing = [s for s in summaries if s.get("gates_pass")]
    if passing:
        return max(s["target_speed_mps"] for s in passing)
    completed = [s for s in summaries if s.get("completion_rate", 0.0) > 0.5]
    if completed:
        return min(s["target_speed_mps"] for s in completed)
    return None


def build_eval_report(
    checkpoint: str,
    speeds: list[float] | tuple[float, ...],
    seed_records: list[RunMetrics],
    gates: SpeedGates | None = None,
) -> dict:
    gates = gates or SpeedGates()
    by_speed: dict[float, list[RunMetrics]] = {}
    for rec in seed_records:
        by_speed.setdefault(rec.target_speed_mps, []).append(rec)
    summaries = []
    for s in sorted(by_speed):
        # per-speed gates: stricter staged acceptance when target >= 2.4
        spd_gates = SpeedGates.for_target(s) if gates is None or gates.completion_rate_min <= 0.95 else gates
        if gates is not None and gates.completion_rate_min > 0.95:
            # operator forced stricter global gates
            spd_gates = gates
        else:
            spd_gates = SpeedGates.for_target(s)
        summaries.append(summarize_speed(by_speed[s], spd_gates))
    initial = recommend_initial_max_speed(summaries)
    return {
        "checkpoint": checkpoint,
        "speeds_mps": list(speeds),
        "gates": asdict(gates),
        "summary": summaries,
        "initial_max_speed_mps": initial,
        "note": (
            "Fixed-speed eval is authoritative for high-speed capability claims. "
            "Do not use mixed Train/mean_reward alone as acceptance evidence."
        ),
        "wandb_metrics": _wandb_metrics(summaries),
    }


def _wandb_metrics(summaries: list[dict]) -> dict:
    """Map summaries to eval/{speed}/... keys for W&B."""
    out = {}
    for s in summaries:
        speed = s["target_speed_mps"]
        prefix = f"eval/{speed:g}"
        out[f"{prefix}/mean_vx"] = s["mean_vx_mps"]
        out[f"{prefix}/rmse_vx"] = s["rmse_vx_mps"]
        out[f"{prefix}/episode_completion"] = s["completion_rate"]
        out[f"{prefix}/bad_orientation_rate"] = s["bad_orientation_rate"]
        out[f"{prefix}/base_contact_rate"] = s["base_contact_rate"]
        out[f"{prefix}/base_height_rate"] = s["base_height_rate"]
        out[f"{prefix}/foot_slide"] = s["foot_slide_mean"]
        out[f"{prefix}/joint_vel_saturation_rate"] = s["joint_vel_saturation_rate"]
        out[f"{prefix}/joint_torque_saturation_rate"] = s["joint_torque_saturation_rate"]
        out[f"{prefix}/action_clip_ratio"] = s["action_clip_ratio"]
        out[f"{prefix}/mean_joint_power_w"] = s["mean_joint_power_w"]
        out[f"{prefix}/cost_of_transport"] = s["cost_of_transport"]
        out[f"{prefix}/gates_pass"] = float(bool(s["gates_pass"]))
    return out


def _rate(value) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    items = _to_float_list(value)
    if not items:
        return 0.0
    return sum(1.0 for v in items if v > 0.5) / len(items)


def _to_float_list(value) -> list[float]:
    if value is None:
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if hasattr(value, "detach"):
        value = value.detach().cpu().reshape(-1).tolist()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    return [float(value)]
