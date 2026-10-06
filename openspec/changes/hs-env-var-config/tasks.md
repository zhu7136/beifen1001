# Tasks

## 1. Add environment variable handling in high_speed_env_cfg.py

- [x] 1.1 Add `LIMX_HS_SPEED_RMSE_MAX` environment variable check in `RobotHighSpeedEnvCfg.__post_init__()` after existing env var handling, converting to float and assigning to `hs.speed_rmse_max_mps`. Verify by reading the file and confirming the new if-block exists at the correct location (after `LIMX_HS_LOW_FRAC` check).

- [x] 1.2 Add `LIMX_HS_SATURATION_RATE_MAX` environment variable check in `RobotHighSpeedEnvCfg.__post_init__()` after the RMSE check, converting to float and assigning to `hs.saturation_rate_max`. Verify by reading the file and confirming both new if-blocks are present.

## 2. Verify environment variable override behavior

- [ ] 2.1 Test that setting `LIMX_HS_SPEED_RMSE_MAX=0.35` overrides the default 0.28 value by running a Python one-liner that imports the config, sets the env var, instantiates the env cfg, and prints `hs.speed_rmse_max_mps`. Verify output shows 0.35.

- [ ] 2.2 Test that setting `LIMX_HS_SATURATION_RATE_MAX=0.55` overrides the default 0.10 value by running a Python one-liner that imports the config, sets the env var, instantiates the env cfg, and prints `hs.saturation_rate_max`. Verify output shows 0.55.

- [ ] 2.3 Test backward compatibility: without either environment variable set, verify that default values (0.28 RMSE, 0.10 saturation) are retained by running a Python one-liner and checking both values.

## 3. Validate training launch with environment variables

- [ ] 3.1 Run a short training test (10-20 iterations) with the full environment variable set from the proposal to verify no import errors or configuration failures. Verify training starts and logs show correct initial max speed from curriculum params.

- [ ] 3.2 Verify in Weights & Biases or terminal output that the curriculum parameters reflect the environment variable overrides (speed_rmse_max and saturation_rate_max visible in config logging).
