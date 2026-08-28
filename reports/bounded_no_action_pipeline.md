# Bounded No-Action Pipeline

## Executive conclusion

`previous_action_removal_partially_changes_dynamics`

Disabling the explicit previous-action channel eliminated the historical **pure 500-step single-action** outcome (4/4 → 0/4), but did not establish stable closed-loop control. All four episodes still spent at least 88.4% of their steps in a dominant period-1 action run; median transitions were only 1.5 and median dominant-action fraction was 97.8%. No episode produced valid attack, block break, pickup, or inventory acquisition.

The original automatic label `period_1_collapse_replaced_by_low_period_cycle` was corrected after a decision-code audit: every episode's period-2-to-4 cycle fraction was exactly 0.0. The observed failure is near-period-1 dominance with a few brief deviations, not a period-2-to-4 oscillator. No training or rollout was repeated for this correction.

Exactly one next action: **freeze this bounded disabled-zero branch. Do not train seeds 11/47, run another actor micro-ablation, collect new data, or start DAgger/PPO in this branch.**

## Stages and budget

- Executed: PRECHECK → MULTI_CAPACITY → FORMAL_TRAIN_SEED29 → RECORDED_REPLAY → AUTONOMOUS_SEED29.
- Skipped by the seed-29 replication gate: CONDITIONAL_REPLICATION and aggregate 12-episode evaluation.
- Capacity training runs: 1; 206 epochs.
- Formal policy training runs: 1; seed 29; 60 epochs.
- Total autonomous episodes: 4.
- No retry, horizon extension, hyperparameter search, or threshold change occurred.

## Correctness and controlled-variable audit

- Full regression suite: **307 passed, 2 skipped**.
- Train SHA-256: `4a0f66617689f8c510cd7fbac7c8803f85a8165daecf47de0c3035e41f0282d3`.
- Validation SHA-256: `51921c5cd6c9523a11294896d6991d3bd75a05fdc397fb0ee5a3929eecdef368`.
- GRU input width 144 and hidden size 128 were preserved.
- The 16-wide disabled slot remained exactly zero in training, validation, replay, and rollout.
- No serialized or trainable action embedding exists in the disabled actor.
- START + 14 token mutation, complete-sequence mutation, masked loss, and shared gradients were exactly invariant.
- Episode-local hidden reset, causal alignment, padding mask, checkpoint reload, and live/standalone replay parity passed.
- Shared tensors were exactly paired within each new run. Historical exp09 pre-training weights were not saved, so exact historical initialization reconstruction cannot be independently proven.
- Teacher actions executed: 0. Privileged actor inputs: 0.
- `student_holdout` and `final_test` were not accessed. No new data was collected. Promotion: none.

The first PRECHECK attempt encountered a Windows permission error in the system pytest temporary directory after 295 tests had passed. The runner was repaired to use the repository-local `.pytest_tmp`; the preserved retry then passed the full suite before training began. This consumed no training budget.

## Multi-trajectory capacity gate

The frozen seeds 18201, 18204, and 18207 contained 332 timesteps and 10 observed actions. The first checkpoint satisfying every threshold occurred at epoch 206, and training stopped immediately:

- Accuracy: **96.99%** (threshold 95%).
- Balanced accuracy: **90.53%** (threshold 90%).
- Cross entropy: 0.1206.
- Zero-slot maximum absolute value: 0.0.
- Reload exact: true.
- Trained-checkpoint token and sequence invariance: true.

## Formal seed-29 training and offline evaluation

The one bounded run reached its minimum horizon of 60 epochs and stopped there because the selected validation-CE minimum was at epoch 29 and post-minimum patience had expired. The selected checkpoint was not changed after inspection.

| Metric | Historical embedded seed29 | Disabled-zero seed29 |
|---|---:|---:|
| Validation accuracy | 70.97% | 54.68% |
| Validation balanced accuracy | 32.35% | 9.37% |
| Validation cross entropy | 1.1423 | 1.8085 |
| Prediction entropy | 1.2747 nats | 1.8865 nats |
| Best epoch | 52 | 29 |

The selected model predicted only forward, turn-left, turn-right, and attack on validation; attack recall was 0 despite 37 examples. Offline degradation did not block the predeclared closed-loop evaluation.

## Recorded-observation replay

Teacher previous actions, model-predicted previous actions, all START, all noop, and random valid actions produced exactly identical hidden states, logits, probabilities, and argmax outputs. Maximum absolute difference was 0 and replay BA was 9.37% for every history mode.

The predictions were already strongly repetitive on recorded observations: seed 18302 had one transition and a 96.1% dominant period-1 fraction; seed 18303 had eight transitions and a 64.9% dominant period-1 fraction. Period-2-to-4 fractions were 0.

## Autonomous results

| Env seed | Approach | Contact | Valid attack | Break | Pickup | Inventory | Transitions | Dominant fraction | Longest run | Period 2–4 fraction |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 18500 | yes | yes | no | no | no | no | 2 | 98.4% | 492 | 0.0% |
| 18501 | no | no | no | no | no | no | 4 | 98.0% | 442 | 0.0% |
| 18502 | yes | yes | no | no | no | no | 1 | 97.6% | 488 | 0.0% |
| 18503 | yes | no | no | no | no | no | 1 | 97.6% | 488 | 0.0% |

Aggregate closed-loop metrics:

- Pure 500-step single-action episodes: **0/4** (historical embedded seed29: 4/4).
- Median action transitions: **1.5** (required: at least 10).
- Median dominant-action fraction: **97.8%** (required: below 95%).
- Episodes below 80% in one dominant period-1-to-4 cycle: **0/4** (required: at least 3/4).
- Episodes at or above 80% in a period-2-to-4 cycle: **0/4**.
- Median longest identical-action streak: 488/500.
- Valid attack / block break / pickup / inventory: **0/4 / 0/4 / 0/4 / 0/4**.

## Causal decision

Removing previous-action information is a causal intervention that weakens the exact pure fixed point, but it is **not sufficient** to produce the predeclared collapse break. The dynamics changed from 500 identical actions to a nearly all-forward period-1 regime with one to four brief deviations per episode. Because the full replication gate failed and no deeper progression trigger occurred, seeds 11 and 47 were correctly not trained.

Final branch decision: `previous_action_removal_partially_changes_dynamics`.
