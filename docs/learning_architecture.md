# Minecraft learning architecture

## Objective and current milestone

The long-term objective is a learned Minecraft autonomous-completion agent.
Natural Treechop v1 is an early curriculum milestone: starting in a natural
world without a supplied tree target, the actor must obtain at least one
naturally generated log in inventory. Breaking a block without pickup is not
success.

The active pipeline is:

```text
Minecraft runtime
  -> legal observation adapter
  -> learned 14-action actor
  -> inventory-confirmed task outcome

privileged bootstrap teacher/oracle
  -> complete demonstrations
  -> auxiliary labels
  -> DAgger corrective labels
  -> evaluation-only failure taxonomy
```

The teacher may run beside an autonomous actor for audit, but its action is
never executed and never used as a fallback. Evaluation summaries record both
`teacher_actions_executed = 0` and `privileged_actor_inputs = 0`.

## Student observation boundary

`mc_rl.learning_observation.LegalObservationAdapter` is the executable schema.
It directly allowlists and copies:

- four causal RGB POV frames;
- player-visible F3 state: XYZ, yaw/pitch and biome metadata;
- observable inventory log count;
- episode-relative position/motion and progress derived from that legal state;
- an eight-action causal history maintained by the actor.

Absolute XYZ is converted to episode-origin-relative displacement before model
input. The model does not receive a pointer to the raw environment mapping.

Train-only privileged state is stored in separate `audit_*` arrays:

- raycast block identity, distance, in-range contact and exact hit point;
- teacher search/contact phase and action source;
- teacher target/candidate/recovery internals;
- world/block truth, reachability and failure classification when available.

These fields may generate teacher actions, auxiliary labels, DAgger labels,
dense rewards or evaluation taxonomy. They are forbidden actor inputs. No
currently used F3 field is disputed; any future F3 extension not already listed
above must be marked `REQUIRES_PROJECT_DECISION` before use.

## Deterministic and learned boundary

Deterministic infrastructure owns environment construction, input conversion,
action clipping, reset/seed handling, inventory-confirmed success, legal F3
parsing, logging, checkpoint loading and safety. The autonomous student owns
all semantic choices across search, target acquisition, navigation, camera,
attack, recovery and pickup through the same fourteen primitive actions.

The v9.11 state machine remains a bootstrap oracle. It is not part of the
autonomous controller. v9.12 was intentionally skipped because repairing one
late diagnostic reach-loss path would not address the learning bottleneck.

## Data and experiment integrity

`configs/seeds/natural_treechop_v1.json` fixes disjoint teacher, BC, DAgger,
student-dev, student-holdout and final-test splits. The final test is rejected
by default in code and requires explicit owner approval. Historical hard seeds
are a diagnostic suite, not a promotion criterion.

Every dataset and checkpoint is content-hashed. `experiments/registry.jsonl`
is append-only and binds hypothesis, Git state, config, seed split, artifacts,
metrics, runtime and conclusion. Failed experiments are retained.

## First learned baseline and next architecture

The first linear sequence baseline used legal POV/F3/inventory features and
causal action history. Offline validation reached 72.28% action accuracy but
autonomous rollout was 0/4. One DAgger iteration raised offline balanced
accuracy from 54.07% to 57.51% yet remained 0/4; episodes collapsed into long
noop/turn/attack fixed points.

The next actor should therefore be a trainable visual encoder plus recurrent
state (CNN/GRU or equivalent), initialized by full-trajectory BC. At most one
additional major DAgger iteration should test whether the recurrent actor
breaks the fixed points. If closed-loop performance remains correction-bound,
transition to a discrete residual actor-critic: BC logits plus a learned PPO
residual with a KL/conservative constraint. The actor keeps the legal boundary;
an asymmetric critic may consume train-only privileged state and dense shaping
for discovery, distance progress, contact, valid attack, break and pickup.
Final evaluation remains inventory success only.
