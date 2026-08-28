# No-Action Single-Trajectory Capacity Audit (exp14)

## Executive conclusion

**Single-trajectory capacity gate passed with the extended diagnostic budget.**

The disabled-zero actor reached 100% accuracy and 100% balanced accuracy on
the frozen 56-step seed-18204 trajectory.  The unchanged minimum-cross-entropy
selection rule chose epoch 997 with CE 0.02586.  Checkpoint reload was exact,
the disabled action slot remained exactly zero, and the trained checkpoint was
exactly invariant to all 15 START/action token values.

This resolves the exp13 sanity blocker.  It does not establish multi-trajectory
capacity, validation generalization, autonomous fixed-point removal, or task
success.

## Controlled audit

| Variable | exp13 | exp14 |
|---|---:|---:|
| Dataset / selected trajectory | seed 18204, 56 steps | identical |
| Train/validation hashes | frozen | exact match |
| Actor / zero slot / GRU width | disabled-zero / 16 / 144 | identical |
| Initialization seed | 7 | 7 |
| Optimizer | AdamW | AdamW |
| Learning rate / weight decay | 1e-3 / 0 | identical |
| Batch / class-weight power | 1 episode / 0 | identical |
| Gradient clip | 1.0 | 1.0 |
| Checkpoint selection | minimum CE | identical |
| Maximum epochs | 250 | 1,000 |
| Patience | 80 | 250 |

All first 250 training-log rows match exp13 exactly across every recorded
column.  This confirms that exp14 reproduced the original run and continued it
under the predeclared larger diagnostic budget rather than obtaining a new
trajectory from an unrelated change.

The train and validation dataset hashes remained:

- train: `4a0f66617689f8c510cd7fbac7c8803f85a8165daecf47de0c3035e41f0282d3`
- validation control: `51921c5cd6c9523a11294896d6991d3bd75a05fdc397fb0ee5a3929eecdef368`

No new data, DAgger data, teacher input, privileged input, or protected split
was used.

## Results

| Metric | exp13 selected | exp14 selected | Threshold |
|---|---:|---:|---:|
| Accuracy | 94.64% | 100.00% | >=95% |
| Balanced accuracy | 81.05% | 100.00% | >=95% |
| Cross entropy | 0.15919 | 0.02586 | recorded |
| Selected epoch | 248 | 997 | minimum CE |
| Mean prediction entropy | 0.33838 | 0.07484 | recorded |
| Zero-slot max abs | 0 | 0 | exactly 0 |
| Checkpoint reload | exact | exact | exact |

The first checkpoint meeting both thresholds appeared at epoch 288, only 38
epochs after the exp13 horizon.  It was also the first 100%/100% checkpoint.
Across the extended run, 701 epochs met both acceptance thresholds and 104
epochs were perfect on both metrics.  The final selected checkpoint correctly
recalled every supported action, including the single `forward_jump` sample
that exp13 missed.

## Interpretation

The exp13 failure was caused by an insufficient sanity optimization horizon in
this fixed run.  It was not evidence that the disabled-zero architecture is
fundamentally unable to represent the one-trajectory sequence.  The conclusion
is deliberately narrow: memorizing one trajectory does not show that the actor
can model several independent episode histories or behave usefully in a new
closed loop.

The high selected epoch also means this diagnostic checkpoint is an overfit
capacity artifact.  It is not a candidate formal policy and must not be
promoted or evaluated as though it were the controlled seed-29 model.

## Correctness and boundaries

- Targeted recurrent/zero-channel/periodic-cycle/runtime regression tests:
  19 passed.
- Shared initialization tensors: exact equal.
- Trained checkpoint mutation sweep: action, probabilities, hidden, logits,
  and combined embedding exactly equal across START plus 14 action IDs.
- Serialized action embedding: absent.
- Teacher actions executed: 0.
- Privileged actor inputs: 0.
- Multi-trajectory sanity, formal training, recorded replay, and autonomous
  rollout: not run.
- `student_holdout` and `final_test`: not accessed.
- Promotion: none.

## Exactly one recommendation

Run the predeclared disabled-zero **multi-trajectory sanity gate** on frozen
seeds 18201, 18204, and 18207 using the unchanged exp08b profile and its 95%
accuracy / 90% balanced-accuracy thresholds.  Do not start formal seed-29
training in the same experiment.
