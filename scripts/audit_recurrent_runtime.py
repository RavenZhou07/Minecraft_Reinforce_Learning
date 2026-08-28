"""Instrument recurrent Treechop rollouts without changing actor behaviour."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import psutil

from mc_rl.experiments import file_sha256
from mc_rl.learning_observation import LegalObservationAdapter
from mc_rl.recurrent_treechop_bc import START_ACTION_TOKEN, RecurrentTreechopPolicy
from mc_rl.runtime_observability import (
    RuntimeTraceRecorder,
    atomic_csv,
    atomic_json,
    atomic_save_trace,
    load_trace,
    summarize_trace,
    validate_trace_integrity,
)
from mc_rl.telemetry_treechop_env import make_telemetry_treechop_env


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/learning/runtime_observability_audit_exp12.json",
    )
    parser.add_argument("--stage", choices=("runtime_gate", "matched"), required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--offline-gradle",
        action="store_true",
        help="Use already cached MineRL Gradle artifacts in the temporary instance.",
    )
    return parser.parse_args()


def enable_temporary_gradle_offline_mode() -> None:
    """Add --offline only to each disposable MineRL launch script copy.

    MineRL copies its Minecraft project into a fresh system-temp directory for
    every instance.  Patching that copy avoids changing the installed runtime
    while preventing an expired SNAPSHOT metadata check from blocking an
    otherwise fully cached local launch.
    """

    import minerl.env.malmo as malmo

    original = malmo.MinecraftInstance._launch_minecraft
    if getattr(original, "_exp12_offline_wrapper", False):
        return

    def launch_offline(instance, port, headless, minecraft_dir, replaceable=True):
        script = Path(minecraft_dir) / "launchClient.bat"
        contents = script.read_text(encoding="utf-8")
        expected = "call gradlew runClient --no-daemon"
        replacement = "call gradlew runClient --offline --no-daemon"
        if expected not in contents and replacement not in contents:
            raise RuntimeError("unexpected MineRL launchClient.bat format")
        if expected in contents:
            script.write_text(contents.replace(expected, replacement), encoding="utf-8")
        return original(instance, port, headless, minecraft_dir, replaceable=replaceable)

    launch_offline._exp12_offline_wrapper = True
    malmo.MinecraftInstance._launch_minecraft = launch_offline


def load_config(path: Path) -> Dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    forbidden = set(config["forbidden_splits"])
    requested = {"student_dev"}
    if forbidden & requested:
        raise PermissionError("diagnostic split intersects a forbidden split")
    for checkpoint in config["checkpoints"]:
        actual = file_sha256(Path(checkpoint["path"]))
        if actual != checkpoint["sha256"]:
            raise ValueError("checkpoint hash mismatch: {}".format(checkpoint["path"]))
    return config


def requested_pairs(config: Dict[str, Any], stage: str):
    if stage == "runtime_gate":
        gate = config["runtime_gate"]
        return [(int(gate["training_seed"]), int(gate["environment_seed"]))]
    return [
        (int(checkpoint_seed), int(environment_seed))
        for checkpoint_seed in config["matched_rollouts"]["training_seeds"]
        for environment_seed in config["matched_rollouts"]["student_dev_environment_seeds"]
    ]


def main():
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    artifact_root = Path("artifacts/exp12")
    trace_root = artifact_root / "runtime_traces"
    checkpoint_by_seed = {
        int(item["training_seed"]): item for item in config["checkpoints"]
    }
    pairs = requested_pairs(config, args.stage)
    max_steps = int(config["matched_rollouts"]["max_steps"])
    rows: List[Dict[str, Any]] = []

    if args.offline_gradle:
        enable_temporary_gradle_offline_mode()

    missing_pairs = []
    for checkpoint_seed, environment_seed in pairs:
        trace_path = trace_root / "seed{}_env{}.npz".format(
            checkpoint_seed, environment_seed
        )
        if trace_path.exists() and not args.overwrite:
            trace = load_trace(trace_path)
            integrity = validate_trace_integrity(trace)
            if not integrity["passed"]:
                raise ValueError("existing trace failed integrity: {}".format(trace_path))
            rows.append(dict(summarize_trace(trace), trace_path=str(trace_path)))
        else:
            missing_pairs.append((checkpoint_seed, environment_seed, trace_path))

    env = None
    if missing_pairs:
        env = make_telemetry_treechop_env(
            seed=missing_pairs[0][1], max_episode_steps=max_steps, include_raycast=True
        )
    try:
        for checkpoint_seed, environment_seed, trace_path in missing_pairs:
            checkpoint = checkpoint_by_seed[checkpoint_seed]
            policy = RecurrentTreechopPolicy.load(checkpoint["path"])
            if policy.model.training:
                raise RuntimeError("checkpoint model was not placed in eval mode")
            env.seed(environment_seed)
            observation = env.reset()
            adapter = LegalObservationAdapter(max_steps)
            hidden = None
            previous_action_token = START_ACTION_TOKEN
            recorder = RuntimeTraceRecorder(
                checkpoint=checkpoint["path"],
                checkpoint_seed=checkpoint_seed,
                environment_seed=environment_seed,
                max_steps=max_steps,
            )
            done = False
            info: Dict[str, Any] = {}
            step = 0
            while not done:
                legal = adapter.reset(observation) if step == 0 else adapter.adapt(observation, step)
                action, probabilities, hidden, diagnostics = policy.predict_step_with_diagnostics(
                    legal.pov, legal.vector, previous_action_token, hidden
                )
                next_observation, _, done, info = env.step(action)
                recorder.append(
                    step=step,
                    observation=observation,
                    pov=legal.pov,
                    legal_vector=legal.vector,
                    previous_action_token=previous_action_token,
                    probabilities=probabilities,
                    diagnostics=diagnostics,
                    next_hidden=hidden,
                    selected_action=action,
                    executed_action=action,
                )
                previous_action_token = int(action)
                observation = next_observation
                step += 1
            recorder.metadata["model_eval_mode"] = True
            recorder.metadata["actor_decisions"] = step
            recorder.metadata["environment_steps"] = step
            recorder.metadata["hidden_advances"] = step
            recorder.metadata["config_path"] = str(config_path)
            recorder.metadata["config_sha256"] = file_sha256(config_path)
            recorder.finalize(done, info)
            atomic_save_trace(trace_path, recorder.arrays())
            trace = load_trace(trace_path)
            integrity = validate_trace_integrity(trace)
            if not integrity["passed"]:
                raise RuntimeError("new trace failed integrity: {}".format(integrity))
            row = dict(summarize_trace(trace), trace_path=str(trace_path))
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
    finally:
        if env is not None:
            try:
                env.close()
            except psutil.NoSuchProcess as error:
                print("WARNING: Minecraft already exited during close: {}".format(error))

    all_trace_rows = []
    for trace_path in sorted(trace_root.glob("seed*_env*.npz")):
        trace = load_trace(trace_path)
        all_trace_rows.append(dict(summarize_trace(trace), trace_path=str(trace_path)))
    atomic_csv(artifact_root / "checkpoint_rollout_summary.csv", all_trace_rows)
    atomic_json(
        artifact_root / "{}_run.json".format(args.stage),
        {
            "experiment_id": config["experiment_id"],
            "stage": args.stage,
            "pairs_requested": [[a, b] for a, b in pairs],
            "pairs_present": len(rows),
            "teacher_actions_executed": int(sum(row["teacher_actions_executed"] for row in rows)),
            "privileged_actor_inputs": int(sum(row["privileged_actor_inputs"] for row in rows)),
            "config_sha256": file_sha256(config_path),
            "infrastructure": {
                "temporary_gradle_offline": bool(args.offline_gradle),
                "reason": "reuse cached MineRL dependencies after remote SNAPSHOT metadata timeouts"
                if args.offline_gradle
                else None,
            },
        },
    )


if __name__ == "__main__":
    main()
