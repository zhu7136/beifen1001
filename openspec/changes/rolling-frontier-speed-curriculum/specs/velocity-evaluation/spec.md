# Spec Delta

## MODIFIED Requirements

### Requirement: High-speed pass gates
Evaluation output SHALL support explicit pass/fail gates for a target speed, including completion rate, orientation termination rate, and speed RMSE thresholds used by the speed curriculum. Default gates remain completion > 95%, orientation terminations < 2%, and speed RMSE < 0.25 m/s for general evaluation. For staged high-speed acceptance (especially after reaching 2.4 m/s in round 2), fixed-speed eval SHALL apply stricter gates: completion rate ≥ 97% and speed RMSE ≤ 0.25 m/s.

#### Scenario: Gate evaluation for curriculum start
- **WHEN** evaluation is used to choose `initial_max_speed_mps` for curriculum training
- **THEN** each candidate speed has recorded pass/fail against the configured gates (default completion > 95%, orientation terminations < 2%, speed RMSE < 0.25 m/s unless a stricter staged profile is selected)

#### Scenario: Stricter gates after 2.4 m/s
- **WHEN** fixed-speed evaluation is run on a checkpoint whose curriculum target is 2.4 m/s (or claims ≥ 2.4 m/s capability)
- **THEN** pass/fail uses completion rate ≥ 97% and speed RMSE ≤ 0.25 m/s at the target speed

#### Scenario: Fail holds round, does not auto-degrade
- **WHEN** fixed-speed eval at 2.4 m/s fails any stricter gate
- **THEN** the report marks fail at that speed; the training curriculum holds the current speed and does not auto-degrade or promote
