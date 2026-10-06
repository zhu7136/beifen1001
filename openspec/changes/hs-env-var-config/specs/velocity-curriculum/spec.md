# Spec Delta

## ADDED Requirements

### Requirement: Environment variable configuration for curriculum thresholds
The system SHALL allow operators to override curriculum progression thresholds via environment variables without modifying code, enabling runtime configuration for different training phases while maintaining backward-compatible defaults.

#### Scenario: Override speed RMSE threshold via environment variable
- **WHEN** the `LIMX_HS_SPEED_RMSE_MAX` environment variable is set before environment initialization
- **THEN** the `speed_rmse_max_mps` value in `HighSpeedCurriculumParams` is set to the float value from the environment variable

#### Scenario: Override saturation rate threshold via environment variable
- **WHEN** the `LIMX_HS_SATURATION_RATE_MAX` environment variable is set before environment initialization
- **THEN** the `saturation_rate_max` value in `HighSpeedCurriculumParams` is set to the float value from the environment variable

#### Scenario: Use default thresholds when environment variables are not set
- **WHEN** neither `LIMX_HS_SPEED_RMSE_MAX` nor `LIMX_HS_SATURATION_RATE_MAX` environment variables are set
- **THEN** curriculum progression thresholds retain their hardcoded default values (0.28 m/s RMSE, 0.10 saturation rate)

#### Scenario: Enable progression through current saturation plateau
- **WHEN** training at 2.4-2.5 m/s with observed torque saturation rates of 45-49% and `LIMX_HS_SATURATION_RATE_MAX` is set to 0.55
- **THEN** curriculum progression is not blocked by the saturation rate gate, allowing advancement beyond 2.4 m/s when other gates are met

#### Scenario: Independent configuration of training vs deployment thresholds
- **WHEN** an operator sets `LIMX_HS_SATURATION_RATE_MAX=0.55` for training phase
- **THEN** this configuration acts as a curriculum progression gate during training, not as a deployment safety standard
