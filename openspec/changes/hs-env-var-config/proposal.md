# Proposal

## Why

The current high-speed environment configuration has hardcoded curriculum progression thresholds (`speed_rmse_max_mps=0.28` and `saturation_rate_max=0.10`) that prevent curriculum advancement at 2.4-2.5 m/s speeds. The default saturation rate threshold (0.10) is incompatible with observed torque saturation rates of 45-49% atthese speeds, causing the curriculum to stall indefinitely. Adding environment variable overrides enables operators to adjust these thresholds for training without modifying code, allowing progression to higher speeds while maintaining separate safety standards for deployment.

## What Changes

- Add environment variable support for `LIMX_HS_SPEED_RMSE_MAX` to configure speed tracking RMSE threshold for curriculum progression
- Add environment variable support for `LIMX_HS_SATURATION_RATE_MAX` to configure maximum allowable torque saturation rate
- Both variables override defaults in `HighSpeedCurriculumParams` during `RobotHighSpeedEnvCfg.__post_init__()`
- Training scripts can now set these values via environment variables without code changes
- Default values remain unchanged for backward compatibility

## Capabilities

### Modified Capabilities
- `velocity-curriculum`: Add requirements for environment variable configuration of curriculum thresholds (`speed_rmse_max_mps` and `saturation_rate_max`) to enable runtime configuration without code changes

## Impact

- **File modified**: `source/limx_rl_lab/limx_rl_lab/tasks/locomotion/robots/limx/high_speed_env_cfg.py`
- **Training scripts**: Can now set `LIMX_HS_SPEED_RMSE_MAX` and `LIMX_HS_SATURATION_RATE_MAX` environment variables
- **Backward compatible**: Default behavior unchanged when environment variables are not set
- **Training workflow**: Enables 2.4 m/s to 2.8 m/s progression with configurable thresholds
- **No breaking changes**: Existing checkpoints and training configurations continue to work
