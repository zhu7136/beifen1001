# Spec Delta

## ADDED Requirements

### Requirement: Fixed-speed mode is not mixed training sampling
High-speed curriculum training SHALL continue to use mixed near/mid/low command sampling and rolling-frontier promotion. Fixed-speed command mode SHALL be reserved for evaluation (or explicit operator freeze) and SHALL NOT be the training sampling path.

#### Scenario: Training keeps mixed batches
- **WHEN** curriculum training runs without evaluator-set fixed mode
- **THEN** command samples remain mixed near the current cap, mid-range, and low/stand according to curriculum fractions

#### Scenario: Eval freeze does not redefine training mix
- **WHEN** an evaluation run sets fixed-speed mode on a live command term
- **THEN** that pin is for the eval session; it does not rewrite training curriculum sampling fractions or gate logic for subsequent training sessions

### Requirement: Curriculum start speed from passing eval only
Training curriculum `initial_max_speed_mps` SHALL be set by the operator from the highest speed that **passed** fixed-speed evaluation (`highest_passing_speed_mps`). If no speed passed, training SHALL NOT assume a high start speed from partial completion metrics.

#### Scenario: Start after a real pass
- **WHEN** fixed-speed eval reports `highest_passing_speed_mps = 2.2` and 2.4 fails gates
- **THEN** curriculum training starts from 2.2 (or lower if operator chooses), not from the failed 2.4 target

#### Scenario: No pass means no invented start
- **WHEN** fixed-speed eval reports `highest_passing_speed_mps = null`
- **THEN** operator does not start high-speed curriculum at a failed target based on eval JSON alone
