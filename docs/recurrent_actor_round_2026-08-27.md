# Recurrent Actor Validation Round — 2026-08-27

## Conclusion

**Result: not supported.** On the unchanged BC dataset, a trainable CNN/GRU did
not meaningfully change the autonomous closed-loop bottleneck. The recurrent
policy reached some privileged-audit approach/contact states, but every one of
the four matched `student_dev` episodes was a 500-step single-action fixed
point with zero action transitions, zero valid attacks, and zero inventory
success. The recurrent checkpoint is retained as an experiment artifact but is
not promoted.

The next recommendation is **Case C: stop architecture scaling and investigate
observability/state aliasing**. Do not start PPO or another major DAgger round
from this checkpoint.

## Scope and repository audit

No Treechop teacher code, recovery state, heuristic, success definition, seed
split, or legal observation semantic was changed. v9.12 remains skipped.
`student_holdout` and `final_test` were not loaded or run.

The pre-implementation audit answered the sequence questions as follows:

1. Spatial input is a legal `64 x 64 x 3` uint8 RGB POV frame. The prior linear
   actor formed a four-frame causal stack and extracted hand-written image
   features; the recurrent actor encodes each current legal frame and carries
   learned history in its GRU.
2. Vector input is the unchanged 16-value legal player state: origin-relative
   and step-delta pose, yaw/pitch, legal biome telemetry, inventory log delta,
   and episode-step fraction.
3. The trajectory stores the previously executed action. The linear actor used
   an eight-action one-hot deque. The recurrent actor uses a learned 16-wide
   embedding over 14 actions plus a dedicated START token (id 14).
4. Both NPZ datasets retain complete timestep ordering.
5. Boundaries are explicit through `episode`, `episode_seed`, `episode_step`,
   and `episode_success`; all retained episodes start at step 0 and have
   contiguous steps.
6. The old trainer converted trajectories to individual stacked samples. Its
   NumPy fit was full-batch, but no sequence boundary reached the model. The new
   loader shuffles complete episodes only, pads within an episode batch, and
   masks all padded loss positions.
7. Rollout keeps one GRU hidden tensor inside the student agent and advances it
   once per student action decision.
8. `reset_episode()` zeros hidden state, restores START, and creates a fresh
   `LegalObservationAdapter`; the evaluator calls it after every environment
   reset.
9. Every non-first dataset timestep satisfies
   `previous_action[t] == action[t-1]` (`3346/3346` checked across train and
   validation). At step 0 the loader replaces the stored noop placeholder with
   START. No target action enters the input used to predict itself.

## Implementation

The single frozen architecture contains 164,526 trainable parameters:

- Spatial encoder: three ReLU convolutions
  (`3→8, k5/s4`; `8→16, k3/s2`; `16→32, k3/s2`) and a 96-wide projection.
- Scalar encoder: `16→32→32` ReLU MLP.
- Previous action: learned 16-wide embedding with a dedicated START token.
- Memory: one-layer GRU with hidden size 128.
- Head: one linear layer producing 14 categorical logits.
- Loss: class-weighted cross entropy, using the same inverse-frequency power
  `0.5` as the prior linear controlled baseline.

Training semantics are strictly:

```text
current legal RGB + current legal vector + previously executed action
    + episode-local hidden from timesteps < t
        -> teacher action at t
```

Complete episodes are batched (`batch_size=2` formally), each batch row starts
from zero hidden state, and padding never contributes to loss. No phase head,
privileged label, target coordinate, raycast, teacher state, or oracle array is
read by the actor loader.

Key implementation files:

- `mc_rl/recurrent_treechop_bc.py`
- `scripts/train_recurrent_treechop_bc.py`
- `scripts/evaluate_natural_treechop_student.py`
- `tests/test_recurrent_treechop_bc.py`
- `requirements-learning.txt`

## Correctness gates

| Gate | Data | Result | Decision |
|---|---:|---:|---|
| Single-trajectory overfit (`exp08a`) | seed 18204, 56 steps | 100.00% accuracy, 100.00% balanced accuracy, CE 0.0181 | pass |
| Multi-trajectory overfit (`exp08b`) | seeds 18201/18204/18207, 332 steps, 10 observed actions | 100.00% accuracy, 100.00% balanced accuracy, CE 0.0130 | pass |
| Checkpoint reload | both sanity checkpoints | bit-identical model tensors | pass |
| Legal/privileged audit | explicit eight-field loader allowlist | 0 privileged actor inputs, 0 privileged supervision heads | pass |
| Fast regression | non-integration suite | `288 passed, 4 deselected` | pass |

New automated coverage checks causal action alignment, START semantics,
non-crossing episode boundaries, padding loss masking, independent GRU batch
rows, environment hidden reset, checkpoint reload, privileged-array absence,
and fixed-point metric calculation.

## Controlled recurrent BC training

Dataset and architecture were frozen before the three runs:

- Train: existing successful `bc_train` episodes only, seeds
  18200/18201/18203/18204/18205/18206/18207, 1,324 samples, SHA-256
  `4a0f66617689f8c510cd7fbac7c8803f85a8165daecf47de0c3035e41f0282d3`.
- Validation: existing successful `bc_validation` episodes only, seeds
  18302/18303, 534 samples, SHA-256
  `51921c5cd6c9523a11294896d6991d3bd75a05fdc397fb0ee5a3929eecdef368`.
- AdamW, learning rate `3e-4`, weight decay `1e-4`, up to 60 epochs,
  patience 15, gradient clip 1.0, class-weight power 0.5.
- Initialization/training RNG seeds: 11, 29, and 47. No hyperparameter sweep.

| Train seed | Validation accuracy | Balanced accuracy | Cross entropy | Prediction entropy (nats) | Best epoch |
|---:|---:|---:|---:|---:|---:|
| 11 | 72.85% | 34.86% | 1.0852 | 1.2797 | 41 |
| 29 | 70.97% | 32.35% | 1.1423 | 1.2747 | 52 |
| 47 | 70.60% | 32.51% | 1.1520 | 1.3054 | 48 |
| Mean ± population SD | 71.47% ± 0.98 | 33.24% ± 1.15 | 1.1265 ± 0.0295 | 1.2866 ± 0.0134 | — |

The autonomous checkpoint was predeclared as seed 29 (the middle
initialization seed), independent of offline model ranking.

Representative seed-29 validation recall shows the remaining imbalance:

| Action | Support | Recall |
|---|---:|---:|
| noop | 8 | 25.0% |
| forward | 277 | 92.1% |
| forward_jump | 27 | 18.5% |
| turn_left / turn_right | 22 / 74 | 27.3% / 83.8% |
| look_up / look_down | 7 / 5 | 14.3% / 0.0% |
| attack / forward_attack | 37 / 8 | 91.9% / 62.5% |
| backward | 6 | 0.0% |
| fine_turn_left / fine_turn_right | 25 / 18 | 32.0% / 5.6% |
| fine_look_up / fine_look_down | 11 / 9 | 0.0% / 0.0% |

## Autonomous recurrent smoke

Matched `student_dev` seeds 18500–18503 were run for 500 steps each. Teacher
actions executed = 0; privileged actor inputs = 0. Success remained the true
inventory natural-log delta rule.

| Seed | Outcome / layer | Meaningful interaction | Approach | Contact | Valid attack | Break | Pickup | Timeout | Dominant action | Dominant fraction | Longest streak | Transitions |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 18500 | contact_without_valid_attack | — | step 0 | step 0 | — | — | — | yes | fine_turn_left | 1.000 | 500 | 0 |
| 18501 | contact_without_valid_attack | — | step 17 | step 17 | — | — | — | yes | turn_right | 1.000 | 500 | 0 |
| 18502 | search_timeout | — | — | — | — | — | — | yes | turn_right | 1.000 | 500 | 0 |
| 18503 | approach_timeout | — | step 0 | — | — | — | — | yes | turn_right | 1.000 | 500 | 0 |

Aggregate recurrent result:

- Inventory success: `0/4`, Wilson 95% interval `[0.0000, 0.4899]`.
- Timeout: `4/4`.
- Progression: meaningful interaction `0/4`, approach `3/4`, contact `2/4`,
  valid attack `0/4`, inferred break `0/4`, pickup `0/4`.
- Fixed point: median dominant fraction `1.0`, median longest identical-action
  streak `500`, median transitions `0`, median action entropy `0` nats.

## Controlled comparison

| Policy | Training samples | Offline accuracy | Offline balanced accuracy | Success | Meaningful | Approach | Contact | Valid attack | Break | Pickup | Median dominant fraction | Pure single-action episodes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Linear BC | 1,324 | 72.28% | 54.07% | 0/4 | 0/4 | 2/4 | 2/4 | 0/4 | 0/4 | 0/4 | 1.0 | 3/4 |
| DAgger1 Linear BC | 2,324 | 72.85% | 57.51% | 0/4 | 0/4 | 2/4 | 1/4 | 0/4 | 0/4 | 0/4 | 1.0 | 3/4 |
| Recurrent BC (seed 29) | 1,324 | 70.97% | 32.35% | 0/4 | 0/4 | 3/4 | 2/4 | 0/4 | 0/4 | 0/4 | 1.0 | 4/4 |

Historical linear CSVs did not retain the ordered per-step action sequence, so
their exact longest-streak and transition counts cannot be reconstructed.
Their action histograms do show median entropy 0 nats and the pure-single-action
counts above. The recurrent evaluator now records the ordered diagnostics.

The extra recurrent approach/contact observations are not evidence of learned
temporal control: each occurred while executing one unchanged turn action for
all 500 steps. There was no action transition or valid attack in any recurrent
trajectory. Offline balanced accuracy also regressed substantially and all
three initialization seeds showed the same weak generalization pattern.

## Regressions and decisions

Regressions:

- Representative recurrent validation balanced accuracy fell from 54.07%
  (Linear BC) to 32.35%; it also trails DAgger1's 57.51%.
- Recurrent autonomous behavior had four pure fixed points versus three for
  each prior linear policy.
- No policy reached a meaningful interaction, valid attack, break, or pickup.

Decisions:

- Keep the recurrent implementation, strict sequence loader, tests, checkpoints,
  and closed-loop diagnostics as reusable infrastructure.
- Do not promote any recurrent checkpoint.
- Do not change the teacher or implement v9.12.
- Do not collect recurrent DAgger2 data from these shallow fixed-point states.
- Do not start PPO; the valid-attack/block-break evidence gate was not met.
- Do not increase GRU size or run an architecture sweep.

## Next recommendation: Case C

The next round should test whether the legal observation can distinguish the
critical Treechop states before spending more data or compute:

1. Use privileged labels only as offline audit strata for `tree_not_visible`,
   `tree_visible`, `roughly_centered`, `approaching`, `contact_range`, and
   `valid_attack_geometry`; never feed them to the actor.
2. Measure legal-observation aliasing across those strata with small diagnostic
   probes and nearest-neighbor/collision analysis, especially around
   contact-versus-valid-attack geometry.
3. Inspect whether the existing 64x64 POV and legal motion features provide a
   stable observable signal for camera/interaction state, and whether the seven
   successful trajectories cover transitions between those states.
4. Only after that evidence should the project choose between representation
   changes and targeted data coverage. Continue to defer DAgger2 and PPO.

## Reproducibility

- Baseline: `f09face`.
- Recurrent implementation/tests: `de74cd9`.
- Sanity gates: `00cae17`.
- Controlled training: `e8526d1`.
- Autonomous evaluation: `8f3938c`.
- Registry: `experiments/registry.jsonl` (append-only).
- Predeclared controlled config:
  `configs/learning/recurrent_bc_controlled_exp09.json`.
- Recurrent checkpoints are retained locally alongside both preserved linear
  checkpoints; none is marked as a milestone.
