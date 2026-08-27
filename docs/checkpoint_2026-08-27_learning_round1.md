# Learning checkpoint — 2026-08-27

## Changes

- Froze the previously uncommitted v9.11/BC state at Git `f09face`.
- Added an immutable seed manifest and append-only experiment registry.
- Added explicit inventory observations and made Natural Treechop learning
  environments require inventory log delta for success.
- Added a strict legal observation adapter and separate privileged audit arrays.
- Added complete-trajectory collection, 14-action end-to-end BC, autonomous
  evaluation, failure timing/taxonomy and DAgger corrective collection.
- Skipped v9.12; no new teacher recovery state or heuristic was added.

## Hypotheses and experiments

1. **Full-trajectory smoke.** Two fixed `teacher_dev` seeds should validate
   action alignment, inventory success and leakage separation.
2. **First training data.** Eight fixed `bc_train` and four disjoint
   `bc_validation` seeds should provide a small but phase-complete baseline.
3. **BC architecture comparison.** Predicted privileged-phase auxiliary
   supervision should improve a legal-observation action head.
4. **Autonomous BC.** Offline accuracy should transfer enough to reveal the
   first closed-loop failure distribution without teacher fallback.
5. **Bounded DAgger iteration 1.** Corrective labels on student-induced states
   should break observed action fixed points without hurting offline validation.

## Results

| Experiment | Result |
|---|---|
| Fast baseline tests | 269 passed, 4 deselected |
| Fast tests after learning pipeline | 280 passed, 4 deselected |
| Full-trajectory smoke, seeds 18000–18001 | 2/2 inventory success; 258 samples |
| BC train, seeds 18200–18207 | 7/8 success; 1,824 total / 1,324 successful samples |
| BC validation, seeds 18300–18303 | 2/4 success; 1,534 total / 534 successful samples |
| BC without phase head | validation accuracy 72.28%; balanced 54.07% |
| BC with predicted phase head | accuracy 72.28%; balanced 53.77% (regression) |
| Autonomous BC, seeds 18500–18503 | 0/4; 100% timeout; Wilson 95% 0–48.99% |
| DAgger collection, seeds 18400–18401 | 1,000 corrections; student/oracle agreement 16.6% |
| DAgger1 aggregated BC | validation accuracy 72.85%; balanced 57.51% |
| Autonomous DAgger1, seeds 18500–18503 | 0/4; 100% timeout; Wilson 95% 0–48.99% |

The original autonomous failure taxonomy was two search timeouts and two
contact-without-valid-attack failures. DAgger1 produced two search, one
approach and one contact failure. Neither policy produced a first meaningful
interaction, block break or pickup. In both evaluations teacher actions
executed and privileged actor inputs were zero.

## Regressions

- The predicted-phase auxiliary head slightly reduced action balanced accuracy
  and was rejected.
- DAgger1 improved offline metrics but did not improve task success or timeout
  rate. It shifted one failure layer without reaching valid attack.
- Linear autoregression collapsed into episode-long noop, turn or attack fixed
  points. This is a model/closed-loop state problem, not evidence for another
  Treechop teacher patch.

## Decisions

- Keep the legal/privileged adapters, seed manifest, registry, datasets,
  inventory success rule, evaluation harness and both positive/negative logs.
- Keep the no-phase and DAgger1 checkpoints as reproducible diagnostics; promote
  neither as a milestone policy.
- Reject the phase-head variant for rollout.
- Stop DAgger on the current linear architecture after one iteration.
- Do not use `student_holdout` or `final_test`; both remain untouched.
- Do not implement v9.12 in this round.

## Current bottleneck

The actor lacks a learned recurrent representation and enters self-reinforcing
action-history fixed points under covariate shift. Dataset size is small, but
the immediate blocker is not teacher coverage: fixed training/validation data
contain all fourteen actions and all five coarse phases, while autonomous
trajectories still collapse.

## Next actions

1. Introduce a trainable CNN/GRU actor over the same legal schema and repeat the
   smoke/load/autonomous checks before expanding data substantially.
2. Expand fixed `bc_train`/`bc_validation` only after the recurrent smoke proves
   useful; do not touch holdout or final.
3. Allow at most one additional major DAgger iteration with the recurrent actor.
4. Then start BC-logit residual PPO with a legal actor, privileged asymmetric
   critic and audited dense rewards; keep inventory acquisition as final reward
   and evaluation success.
