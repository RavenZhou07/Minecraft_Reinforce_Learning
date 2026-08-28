# Controlled Previous-Action Channel Ablation (exp13)

## 1. Executive conclusion

**correctness/sanity gate failed**

The `disabled_zero` intervention was implemented and passed the model-level
correctness tests, but it did not pass the predeclared single-trajectory
capacity gate.  On frozen seed 18204 (56 steps), the checkpoint selected by the
unchanged minimum-validation-cross-entropy rule reached 94.64% action accuracy
and 81.05% balanced accuracy, below the required 95%/95% thresholds.

Stop condition 10 therefore fired.  The multi-trajectory gate, formal seed-29
training, recorded-observation replay, and autonomous `student_dev` evaluation
were not run.  This experiment does **not** answer whether removing the
previous-action channel is sufficient to break the closed-loop fixed point.

## 2. Controlled-variable audit

| Control | Result |
|---|---|
| Train dataset | Exact SHA-256 match, 1,324 samples, seeds 18200/18201/18203/18204/18205/18206/18207 |
| Validation dataset | Exact SHA-256 match, 534 samples, seeds 18302/18303 |
| RGB and legal vector | Unchanged: current 64x64 RGB and current legal vector width 16 |
| CNN / scalar encoder / GRU / action head | Architectures and all shared tensor shapes unchanged |
| GRU input width / hidden size | 144 / 128, unchanged |
| Previous-action intervention | Internal constant-zero slot, width 16; dataset token ignored |
| Training seed 29 | Predeclared, but formal run not started because of the gate failure |
| Formal optimizer/loss | Frozen to exp09, but not executed |
| New data / DAgger1 training | None |
| Privileged actor input / teacher execution | 0 / 0 |
| Autonomous seeds | Frozen to 18500–18503, but not accessed |
| `student_holdout` / `final_test` | Not accessed |
| Promotion | None |

The trajectory loader still reads the stored previous action only for causal
alignment validation.  In `disabled_zero` mode, the model neither calls a token
lookup nor registers/saves an action-embedding parameter.

## 3. Initialization audit

A freshly constructed original-style embedded actor was created first under
seed 29.  All 18 shared tensors in the spatial encoder, scalar encoder, GRU,
and action head were copied into the disabled-zero actor and compared exactly.

| Audit | Result |
|---|---:|
| Embedded initial shared-state SHA-256 | `e945a28b65e00e2bcc516d59d9bb23ad6f3f8cb4502a0378efb33bc6253751ae` |
| Disabled-zero initial shared-state SHA-256 | `e945a28b65e00e2bcc516d59d9bb23ad6f3f8cb4502a0378efb33bc6253751ae` |
| Shared tensors exact equal | Yes |
| Disabled checkpoint has action embedding | No |
| GRU input width | 144 |

This proves exact within-new-run pairing.  It does not prove exact historical
pairing to exp09 seed29 because the historical actor's untrained initial state
was not saved.  No claim of exact historical initialization reconstruction is
made.

## 4. Correctness tests and sanity

The following implementation gates passed:

- START plus all 14 action-token mutations produced identical combined
  embeddings, next hidden states, logits, probabilities, and argmax actions.
- Complete-sequence previous-action mutation produced identical training
  logits, masked loss, and all shared-parameter gradients.
- The disabled slot had exact maximum absolute value 0.0.
- No trainable or serialized action embedding exists in the disabled model.
- Hidden reset, episode-local batching, padding mask, causal alignment,
  independent batch rows, privileged-array absence, checkpoint reload, and
  live/standalone infrastructure regressions remained covered.
- Full fast suite: 304 passed, 2 skipped.

### Single-trajectory gate

The gate reused seed 18204, all 56 steps, and the exp08a optimizer profile:
AdamW, learning rate 1e-3, no weight decay, one complete episode per batch,
class-weight power 0, gradient clip 1.0, maximum 250 epochs, and patience 80.

| Metric | Required | Selected checkpoint | Result |
|---|---:|---:|---|
| Accuracy | >=95% | 94.64% | Fail |
| Balanced accuracy | >=95% | 81.05% | Fail |
| Cross entropy | Recorded | 0.15919 | — |
| Selected epoch | Minimum CE | 248 | — |
| Zero-slot max abs | 0 | 0 | Pass |
| Reload parity | Exact | Exact | Pass |

The highest ordinary accuracy observed was 96.43% at epochs 230 and 240, but
balanced accuracy was only 82.72% at epoch 240.  Those epochs therefore do not
pass the gate and cannot be cherry-picked.  At the selected checkpoint, the
single `forward_jump` example had recall 0 and was predicted as `forward`; one
`attack`/`forward` confusion in each direction also remained.

The loss was still improving near the 250-epoch boundary.  The evidence is
consistent with insufficient rare-action/sequence fitting under this
sanity-only budget, but it does not distinguish optimization budget from a
deeper no-action sequence-capacity limitation.  The passed mutation and zero
channel tests provide no evidence of an implementation leak or accidental
nonzero slot.

### Multi-trajectory gate

Not run.  The single-trajectory failure is a hard prerequisite failure.

## 5. Offline results

No formal seed-29 policy was trained, so there are no new validation accuracy,
balanced accuracy, CE, entropy, per-action recall, confusion matrix, or formal
training curves to compare with exp09.

| Policy | Previous-action information | Train seed | Offline BA | Pure fixed points | Median transitions | Median dominant fraction | Low-period cycle | Valid attack | Break | Pickup | Inventory |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| Historical recurrent seed29 | enabled | 29 | 32.35% | 4/4 | 0 | 1.0 | period-1 | 0/4 | 0/4 | 0/4 | 0/4 |
| Controlled no-action actor | disabled | 29 | not trained | not evaluated | not evaluated | not evaluated | not evaluated | not evaluated | not evaluated | not evaluated | not evaluated |

The sanity checkpoint is not a formal policy and is not substituted into this
comparison.

## 6. Recorded-observation replay

Not run because no formal checkpoint exists.  The pretraining unit and sequence
tests already established exact token-history invariance of the implementation,
but they are not reported as formal validation replay results.

## 7. Autonomous results

Not run.  No `student_dev` environment was launched and no autonomous trace was
created.  Consequently, the primary endpoint and period-1-to-4 trajectory
metrics remain unobserved.  Teacher actions executed and privileged actor
inputs remain zero; protected splits were not accessed.

## 8. Causal decision

The question “is disabling previous-action information sufficient to block the
current autonomous fixed-point mechanism?” remains unanswered.  Model-level
correctness passed, but the intervention failed its required sequence-capacity
prerequisite.  Running the formal training or rollout anyway would violate the
predeclared stop rule and confound an autonomous result with a known sanity
failure.

## 9. Exactly one recommendation

Run one predeclared **sanity-only capacity audit** on the same seed-18204
trajectory with the same disabled-zero actor, initialization seed, optimizer,
learning rate, loss, zero channel, checkpoint-selection rule, and unchanged
95%/95% thresholds, changing only the diagnostic maximum epoch budget from 250
to 1,000 (with patience 250).  Do not run multi-trajectory sanity or formal
seed-29 training unless that single gate passes.

This is the only recommended next stage.  It does not add data, restore the
previous-action channel, alter observations, run DAgger/PPO, access protected
splits, or promote a checkpoint.
