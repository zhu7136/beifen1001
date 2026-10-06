# Design

## Context

The current high-speed environment configuration in `high_speed_env_cfg.py` already supports environment variables for several curriculum parameters (target speed, initial max speed, speed increment, command mix fractions). However, the curriculum progression thresholds (`speed_rmse_max_mps` and `saturation_rate_max`) are hardcoded, preventing operators from adjusting these gates without code changes.

Recent training runs at 2.4-2.5 m/s show torque saturation rates of 45-49%, which exceed the default threshold of 0.10, causing the curriculum to freeze indefinitely. This blocks progression to higher speeds even when the policy demonstrates adequate tracking performance.

See proposal.md for full motivation.

## Goals / Non-Goals

**Goals:**
- Enable runtime configuration of curriculum progression thresholds via environment variables
- Maintain backward compatibility with existing training configurations
- Support the specific use case of progressing from 2.4 m/s to 2.8 m/s with adjusted saturation tolerance
- Keep implementation minimal and focused on the existing `__post_init__()` pattern

**Non-Goals:**
- No changes to default threshold values (backward compatible)
- No changes to deployment-time safety standards (training gate adjustment only)
- No reward function modifications
- No network architecture changes

## Decisions

### Decision 1: Add two environment variables in `__post_init__()`

**Approach:** Follow the existing pattern for environment variable handling in `RobotHighSpeedEnvCfg.__post_init__()`:
- `LIMX_HS_SPEED_RMSE_MAX` → overrides `hs.speed_rmse_max_mps`
- `LIMX_HS_SATURATION_RATE_MAX` → overrides `hs.saturation_rate_max`

**Rationale:**
- Consistent with existing env vars (`LIMX_HS_INITIAL_MAX_SPEED`, `LIMX_HS_TARGET_SPEED`, etc.)
- Minimal code change: two additional `if` blocks
- No breaking changes: defaults preserved when env vars not set
- Training scripts can set these at the shell level before invoking Python

**Alternatives considered:**
1. **Configuration file**: Would require file I/O, parsing, and schema validation. Overkill for two parameters.
2. **Command-line arguments**: Would require modifying argument parsing in training scripts. Environment variables are already the established pattern.
3. **YAML config**: Adds dependency on external config file management. Environment variables are more operationally simple for cluster training.

### Decision 2: Parse as float without validation

**Approach:** Convert environment variable values directly to float using `float()`, matching the existing pattern for other env vars in the same file.

**Rationale:**
- Consistent with existing env var handling (`LIMX_HS_INITIAL_MAX_SPEED`, etc.)
- Invalid values will raise ValueError naturally, providing immediate feedback
- No need for complex validation at the configuration layer

**Alternatives considered:**
- Add range validation (e.g., ensure saturation_rate_max is between 0 and 1). However, this adds complexity and the operator can misconfigure via code anyway.

### Decision 3: Training-phase threshold, not deployment standard

**Approach:** Document that `saturation_rate_max` configured via environment variable is a curriculum progression gate for training, not a deployment safety limit.

**Rationale:**
- Training can tolerate higher saturation rates temporarily while learning
- Deployment safety standards are separate and may be more conservative
- Clear documentation prevents confusion about acceptable saturation levels

## Risks / Trade-offs

### Risk 1: Operators may set saturation threshold too high

**Impact:** Policy could learn to rely on saturation, degrading real-world performance or safety.

**Mitigation:** 
- Document that training-phase thresholds (e.g., 0.55) are temporary and should be tightened for deployment evaluation
- The default 0.10 remains appropriate for final evaluation

### Risk 2: Environment variable name conflicts

**Impact:** Future env vars might conflict with existing naming convention.

**Mitigation:** 
- Use `LIMX_HS_` prefix consistently (already established)
- Consider documenting reserved names as the set grows

### Trade-off: Hardcoded defaults vs. YAML configuration

**Trade-off:** Environment variables are operationally simple but lack schema validation and version control.

**Rationale:** Environment variables win for this use case because:
- Already established pattern in the codebase
- Cluster training workflows already use env vars extensively
- Two parameters don't justify config file complexity
