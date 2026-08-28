# Repository-Wide Audit Policy

This file is the persistent, repository-wide audit policy for every Codex task and prompt executed anywhere in this repository.

- Treat the policy below as mandatory for all project work, subject only to higher-priority system, developer, security, and safety requirements.
- Do not modify, delete, rename, weaken, bypass, or replace this file or any part of the policy unless the user explicitly requests that specific change.
- A request to implement project work does not by itself authorize changing this policy.

# Global Override — Anti-Stall and Automated Progression Protocol

This section overrides any conflicting instruction in the current or previous
round that requires a manual pause after every successful diagnostic stage.

The project must not continue creating an open-ended sequence of small
experiments whose primary purpose is to approach an offline metric threshold.

Treechop is one curriculum step in the larger Minecraft completion project.
It must not indefinitely block broader data collection, curriculum expansion,
or formal training.

---

# 1. Global objective

The purpose of correctness and capacity gates is only to prevent invalid formal
experiments.

They are not independent optimization targets.

A gate may answer:

```text
is the implementation correct?
can the architecture represent the frozen sanity data?
```

A gate must not become:

```text
keep changing training details until a desired score is reached
```

Once correctness and minimum capacity are established, the pipeline must
advance automatically to formal training and closed-loop evaluation.

---

# 2. Hard anti-stall budget

For the disabled-zero branch, the remaining pre-scale budget is fixed to:

```text
capacity runs:
    one primary run
    plus at most one automatic horizon extension

formal architecture runs:
    one seed-29 run

replication runs:
    seeds 11 and 47, only if seed 29 passes the closed-loop collapse gate

new architecture interventions before scale-or-freeze:
    zero

teacher patches:
    zero

new observability audits:
    zero

new previous-action causality audits:
    zero
```

No third capacity attempt is allowed.

No alternative:

- initialization seed;
- learning rate;
- optimizer;
- class weighting;
- model width;
- loss;
- observation representation;

may be tried to force the capacity metric above threshold.

At the end of this bounded pipeline, the project must either:

```text
enter scaled data collection/training
```

or:

```text
freeze the disabled-zero Treechop learning branch
and move to the next macro Minecraft milestone
```

It must not create another narrow diagnostic branch by default.

---

# 3. One-command automated pipeline

Implement a resumable orchestrator, suggested name:

```text
scripts/run_treechop_no_action_pipeline.py
```

Suggested invocation:

```bash
python scripts/run_treechop_no_action_pipeline.py \
  --config configs/learning/treechop_no_action_pipeline_exp15.json \
  --resume
```

The orchestrator must:

1. load the frozen pipeline config;
2. inspect the append-only registry;
3. verify all hashes and protected-split boundaries;
4. skip stages already completed successfully;
5. resume incomplete stages from existing artifacts when safe;
6. execute the next eligible stage;
7. evaluate predeclared transition conditions;
8. automatically launch the next permitted stage;
9. stop only at a hard stop condition or final pipeline decision;
10. generate a consolidated report.

Do not require a new human prompt between predeclared stages.

Each scientific stage may retain its own append-only experiment ID and report,
but all stages should be launched by the same orchestrator.

---

# 4. Pipeline state machine

Use an explicit state machine equivalent to:

```text
PRECHECK
    |
    v
MULTI_CAPACITY_PRIMARY
    |
    +-- pass --------------------------> FORMAL_TRAIN
    |
    +-- fail, horizon-limited --------> MULTI_CAPACITY_EXTENSION
    |                                      |
    |                                      +-- pass --> FORMAL_TRAIN
    |                                      |
    |                                      +-- fail --> FREEZE_BRANCH
    |
    +-- fail, plateau -----------------> FREEZE_BRANCH

FORMAL_TRAIN
    |
    v
RECORDED_REPLAY
    |
    v
AUTONOMOUS_STUDENT_DEV
    |
    +-- strong collapse break --------> REPLICATION
    |
    +-- partial/no collapse break ----> FREEZE_BRANCH

REPLICATION
    |
    +-- replicated -------------------> SCALE_DATA
    |
    +-- not replicated ---------------> FREEZE_BRANCH

SCALE_DATA
    |
    v
SCALED_BC_TRAINING
    |
    v
NEXT PREDECLARED RL GATE
```

No other automatic branch is permitted.

---

# 5. Capacity stage

Use the frozen multi-trajectory subset:

```text
18201
18204
18207
```

and the verified exp08b profile.

The existing thresholds remain:

```text
accuracy >= 95%
balanced accuracy >= 90%
```

These thresholds are binary sanity gates, not objectives to optimize
indefinitely.

## 5.1 Primary horizon

Run through the exact historical exp08b horizon.

At that horizon:

### Pass

Proceed immediately to formal seed-29 training.

Do not stop for a separate report review.

### Fail with clear plateau

Freeze the disabled-zero branch immediately.

Do not extend.

### Fail while clearly horizon-limited

Continue automatically into one and only one extended phase.

“Horizon-limited” must be decided using predeclared rules based on:

- final-window CE slope;
- final-window balanced-accuracy trend;
- absence of divergence;
- correctness gates remaining valid.

Do not decide it informally after seeing the result.

## 5.2 Absolute extension cap

The extension cap must be fixed before the primary run begins.

Use:

```text
absolute maximum epochs =
min(4 × historical exp08b maximum epochs, 1000)
```

and a corresponding predeclared patience no greater than 250.

At the absolute cap:

```text
pass → formal training
fail → freeze branch
```

No third run or further horizon increase is allowed.

The primary and extended phases may be implemented as one continuous training
process with milestone checkpoints. They do not need separate manual
experiments.

---

# 6. Formal training must not be blocked by offline optimization

Once the capacity gate passes, automatically run formal disabled-zero seed-29
training on the frozen complete BC train/validation datasets.

Do not add a new offline-performance gate before autonomous evaluation.

The formal actor must be evaluated in closed loop even if:

- validation balanced accuracy is lower than expected;
- some rare-action recall remains poor;
- validation CE is worse than the historical embedded actor.

As long as:

- correctness tests pass;
- checkpoint reload passes;
- no privileged input exists;
- training is numerically valid;

the predeclared autonomous evaluation must run.

Offline metrics are diagnostics, not permission to continue the pipeline.

---

# 7. Efficient formal training horizon

Avoid opening a separate experiment later merely because the no-action actor
optimizes more slowly.

Use one continuous formal training run with two recorded checkpoints:

```text
matched-budget checkpoint:
    best checkpoint within the original exp09 60-epoch budget

adequately-optimized checkpoint:
    best checkpoint within one larger predeclared absolute budget
```

Suggested absolute budget:

```text
maximum epochs: 240
early stopping begins no earlier than epoch 60
post-60 patience: 45
```

Keep unchanged:

- data;
- optimizer;
- learning rate;
- weight decay;
- loss;
- class weights;
- architecture;
- seed 29.

This produces both:

1. a compute-matched offline comparison;
2. one adequately optimized candidate;

without training two separate models or requesting another prompt.

The autonomous primary evaluation should use the adequately optimized
predeclared checkpoint.

Do not extend beyond this formal absolute budget.

---

# 8. Autonomous decision

Run deterministic autonomous evaluation on:

```text
18500
18501
18502
18503
```

regardless of offline ranking.

Use the previously declared period-1-to-4 cycle diagnostics.

## 8.1 Strong collapse break

The seed-29 actor passes the collapse gate only if all conditions hold:

```text
pure 500-step single-action fixed points <= 1/4
median action transitions >= 10
median dominant action fraction < 0.95
at least 3/4 episodes spend <80% of steps in one period-1-to-4 cycle
```

Passing this gate means only:

```text
the dominant closed-loop collapse mechanism was broken
```

It does not mean Treechop is solved.

On pass, automatically launch replication seeds 11 and 47.

## 8.2 Partial or failed result

If the seed-29 actor does not satisfy the full collapse gate:

- do not open a new hidden-state audit;
- do not try a new model width;
- do not restore previous-action input;
- do not alter the observation;
- do not collect DAgger2;
- do not start PPO.

Freeze the disabled-zero learning branch.

Retain:

- actor implementation;
- datasets;
- diagnostics;
- teacher;
- evaluator;
- registry.

Use the existing Treechop teacher as a temporary curriculum/data-generation
module and proceed to the next macro Minecraft milestone rather than allowing
this Treechop learner to block the entire project.

---

# 9. Replication gate

If seed 29 passes, train exactly two replications:

```text
training seeds:
11
47
```

Use the same architecture, data, optimizer, horizon and checkpoint rule.

No additional seeds are allowed before scale.

Evaluate each on the same four `student_dev` seeds.

Replication passes if at least two of the three total checkpoints:

```text
seed 29
seed 11
seed 47
```

satisfy the same closed-loop collapse gate.

If replication fails:

```text
freeze branch
```

Do not diagnose initialization sensitivity further.

If replication passes:

```text
enter scale phase automatically
```

---

# 10. Mandatory scale trigger

A replicated collapse-breaking result must immediately trigger a fixed
data-scaling phase.

Do not insert another representation or observability diagnostic before
scaling.

Initial scale target:

```text
successful bc_train trajectories: 64
successful bc_validation trajectories: 16
```

Use new fixed, disjoint seeds allocated through the immutable seed manifest.

Collection attempt caps:

```text
bc_train attempts: 96
bc_validation attempts: 32
```

If the success quota is not reached within the attempt cap, use all collected
data and report the achieved count. Do not keep collecting indefinitely.

Train exactly three frozen no-action seeds on the scaled dataset and run the
same dev evaluation.

This is the point at which the project begins substantive data-scale training.

---

# 11. Post-scale cap

After scaled BC, allow at most:

```text
one major recurrent DAgger iteration
```

and only if scaled BC already reaches student-induced meaningful states such
as valid attack/contact recovery.

PPO remains gated on evidence that the actor can reach at least:

```text
valid attack
or block break
```

in closed-loop dev evaluation.

No repeated DAgger rounds are allowed on policies that remain in shallow
periodic collapse.

---

# 12. Automation artifacts

Maintain:

```text
artifacts/pipeline/
  pipeline_state.json
  stage_history.jsonl
  current_decision.json
  failure.json
```

Each stage record must contain:

```text
stage name
status
experiment ID
git commit
config hash
dataset/checkpoint hashes
start/end timestamps
exit reason
next state
artifact paths
```

The runner must be idempotent:

- completed stages are not rerun;
- partial stages are resumed only when provenance matches;
- mismatched artifacts cause a safe stop;
- append-only registry entries are never overwritten.

---

# 13. Human-intervention boundaries

The automated runner should stop for human review only if the next action would:

- modify legal observation semantics;
- modify teacher/task/success definitions;
- access `student_holdout` or `final_test`;
- exceed predeclared data/compute caps;
- begin a new PPO design not already predeclared;
- violate provenance or correctness.

It must not stop merely because:

- an offline score is lower than hoped;
- a rare action has low recall;
- a stage completed and a report was generated;
- another predeclared stage is ready.

---

# 14. Final pipeline outcomes

The pipeline must terminate in exactly one of these states:

```text
SCALE_STARTED
```

Meaning:

- collapse break replicated;
- fixed data-scale collection/training has begun.

```text
TREECHOP_LEARNER_FROZEN
```

Meaning:

- the bounded no-action branch did not produce a replicated closed-loop
  improvement;
- no further Treechop micro-optimization is authorized;
- the project proceeds using the teacher/scaffold while broader Minecraft
  curriculum work continues.

```text
CORRECTNESS_BLOCKED
```

Meaning:

- provenance, legality, or implementation correctness failed;
- performance optimization is not attempted.

Do not terminate in an ambiguous state such as:

```text
more diagnostics recommended
```

unless a protected semantic boundary requires human approval.

---

# 15. Core principle

The project must distinguish between:

```text
minimum validation needed to avoid an invalid experiment
```

and:

```text
repeated optimization performed only to improve a diagnostic score
```

Use diagnostics to open or close a branch.

Do not let diagnostics become the branch itself.

Within the remaining bounded budget, automatically reach one of two practical
outcomes:

```text
replicate and scale
```

or:

```text
freeze and move on
```
