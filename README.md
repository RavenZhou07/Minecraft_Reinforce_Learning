# Minecraft Resource Search Baselines (MineRL 0.4.4)

[简体中文](README.zh-CN.md) | English

An inspectable Minecraft learning project whose long-term goal is autonomous
survival and completion. Natural Treechop is the first learning milestone, not
the final product. The repository now contains a strict legal-observation
adapter, complete teacher-trajectory collection, end-to-end behaviour cloning,
autonomous learned-policy rollout, and limited DAgger data aggregation on the
legacy MineRL/Malmo stack.

The current curriculum asks an agent to locate the initially nearest tree among
visually identical distractors. The policy scans the full scene, builds an
object-centric candidate map, selects and approaches a target, monitors visual
progress, and attempts recovery or replanning when progress stalls.

> The current learned baseline is deliberately weak and is not a claim of
> general Minecraft competence. The first honest autonomous result is 0/4;
> every failure and timeout is retained. Teacher state machines are bootstrap
> infrastructure and do not execute actions for the autonomous student.

See [`docs/learning_architecture.md`](docs/learning_architecture.md) for the
observation boundary and [`docs/checkpoint_2026-08-27_learning_round1.md`](docs/checkpoint_2026-08-27_learning_round1.md)
for the first end-to-end learning checkpoint.

## Current status

Verified stack: Windows x64, Python 3.8.20, MineRL 0.4.4, Gym 0.19.0,
OpenJDK 8, and Minecraft 1.11.2/Malmo.

| Experiment | Result | Mean steps |
|---|---:|---:|
| Single tree, ±180° yaw, distance 3–10 | Oracle 20/20; visual 20/20 | 38.5 / 45.5 |
| Target + two identical distractors, old visual baseline | 16/20 | 107.5 |
| Explicit candidate search, seeds 11000–11019 | 20/20 | 75.2 |
| Explicit candidate search, unseen seeds 12000–12029 | 30/30 | 72.5 |
| Forced wrong initial selection, seeds 13000–13002 | best complete run 2/3 | — |

The ordinary 20/30-seed runs had no 300-step failures, but all 50 episodes
selected the correct candidate initially. The stricter diagnostic still fails
on seed 13000: after walking to a distractor, a bearing-only memory cannot
update the now-distant target direction after translation.

The current work-in-progress adds an optional `f3_telemetry` profile. It exposes
only the agent's own position, yaw/pitch, and biome metadata, then estimates
candidate world coordinates from POV-derived range. It does not expose the log
grid, target position, nearest-tree label, or oracle distance. Its results must
be reported separately from `pov_only` and are not verified until a complete
Minecraft evaluation is recorded.

See [`docs/find_tree_curriculum.md`](docs/find_tree_curriculum.md) for detailed
results, confidence intervals, task gates, and failure analysis.

## Design

```text
POV (+ optional F3 self telemetry)
                │
                ▼
       resource detector/adapter
                │
                ▼
SCAN → candidate map → SELECT → ALIGN → APPROACH
                                      │
                                      ▼
                         progress / stall monitor
                                      │
                                      ▼
                    LOCAL_REACQUIRE → RECOVER → REPLAN
```

The deployment score is logged term by term:

```text
confidence
+ 2 * log(apparent_size + 1e-3)
- 0.15 * abs(turn) / 180
- approach_attempts
- 0.002 * age
```

Tree detection uses explainable RGB connected components. Detections are merged
by bearing and visual scale. In telemetry mode, candidates also carry an
estimated world position and uncertainty, allowing their bearing to be
recomputed after the agent moves.

## Repository layout

```text
mc_rl/
  actions.py             discrete action mapping
  candidates.py          representation, merging, scoring, cooldown
  envs.py                environment factory and isolated runtime config
  find_tree_env.py       custom curriculum and evaluation-only oracle
  navigation.py          geometry and oracle validation controller
  progress.py            visual stall/loss diagnostics
  resource_adapters.py   generic adapter and tree implementation
  search_policy.py       scan/approach/recovery/replan state machine
  telemetry.py           optional F3 state and visual-to-world geometry
  vision.py              compact visual baseline
  wrappers.py            Gym wrappers and one-log termination
scripts/                  install, collect, train, evaluate, and demos
tests/                    unit and opt-in Minecraft integration tests
docs/                     curriculum and failure analysis
```

## Installation

Do not upgrade MineRL, Gym, or the legacy dependencies. Do not install them into
the global Python or Java environment.

```powershell
conda env create --prefix ..\.conda-env -f environment.yml
conda activate ..\.conda-env
python scripts/install_minerl.py
python -m pip install -r requirements.txt
python scripts/check_install.py
python -m pip check
```

The order matters. `install_minerl.py` applies a narrow build-repository repair
required by the historical MineRL 0.4.4 source distribution. Pip is pinned
below 24.1 because newer versions reject Gym 0.19.0's historical metadata.
Confirm Python 3.8 and Java 1.8 before launching Minecraft.

## Tests

Fast tests do not launch Minecraft:

```powershell
..\.conda-env\python.exe -m pytest -m "not integration" -q
```

Minecraft integration tests are opt-in and sequential:

```powershell
$env:RUN_MINERL_INTEGRATION = "1"
..\.conda-env\python.exe -m pytest -m integration -q
```

Minecraft cold start can take 4–6 minutes. A temporarily silent console is not
evidence of a hang. Do not launch multiple instances in parallel.

## Candidate-search smoke tests

POV-only:

```powershell
..\.conda-env\python.exe -m scripts.evaluate_candidate_search `
  --episodes 3 --seed 10000 --max-steps 300 `
  --yaw-noise 180 --distance-min 3 --distance-max 10 `
  --distractor-trees 2 --modes candidate --sensor-profile pov_only `
  --output logs/find_tree/my_candidate_smoke.csv
```

Forced wrong selection with self telemetry:

```powershell
..\.conda-env\python.exe -m scripts.evaluate_candidate_search `
  --episodes 3 --seed 13000 --max-steps 300 `
  --yaw-noise 180 --distance-min 3 --distance-max 10 `
  --distractor-trees 2 --modes candidate --force-initial-rank 1 `
  --sensor-profile f3_telemetry `
  --output logs/find_tree/my_f3_recovery_smoke.csv
```

Evaluation scripts refuse to overwrite existing outputs by default. Do not
replace failed seeds or relax success criteria when comparing policies.

## Sensor/oracle boundary

- `pov_only`: POV, commanded camera deltas, and internal memory.
- `f3_telemetry`: POV plus the agent's own `x/y/z`, yaw, pitch, and biome,
  analogous to player-visible F3 information.

Both profiles prohibit policy access to the generated log grid, target
coordinates, oracle distance, or correctness labels. Oracle information is
evaluation-only. Results from the two profiles must remain separate.

## Tracked artifacts

Git is configured to keep source, documentation, the compact best checkpoint,
and a few stable JSON summaries. Raw datasets, POV frames, full traces,
Minecraft logs, watcher files, temporary worlds, and the local Conda environment
are ignored.

Current best visual checkpoint:

```text
checkpoints/find_tree_visual_distance3_10_stack4.npz
```

## Compatibility notes

- MineRL 0.4.4 requires JDK 8. Global JDK 21 can fail because JAXB is absent.
- Windows shutdown may print `Failed to delete the temporary minecraft
  directory` or `process already exited`. Preserve the warning and verify that
  no Java process remains.
- The original Gym API is used: `reset() -> observation` and
  `step() -> observation, reward, done, info`.

## License

No open-source license has been selected. Add an explicit license before public
release; until then, normal copyright restrictions apply.
