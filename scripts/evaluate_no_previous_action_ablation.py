"""Autonomous student_dev evaluation for the controlled exp13 intervention."""

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import psutil

from mc_rl.experiments import file_sha256
from mc_rl.learning_observation import LegalObservationAdapter
from mc_rl.recurrent_treechop_bc import (
    PREVIOUS_ACTION_DISABLED_ZERO,
    START_ACTION_TOKEN,
    RecurrentTreechopPolicy,
)
from mc_rl.runtime_observability import (
    RuntimeTraceRecorder,
    atomic_csv,
    atomic_json,
    atomic_save_trace,
    load_trace,
    standalone_replay,
    summarize_trace,
    validate_trace_integrity,
)
from mc_rl.telemetry_treechop_env import make_telemetry_treechop_env
from scripts.audit_recurrent_runtime import enable_temporary_gradle_offline_mode


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-root", default="artifacts/exp13")
    parser.add_argument("--offline-gradle", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def failure_taxonomy(row: Dict[str, Any]) -> str:
    if row["inventory_success"]:
        return "success"
    if not row["approach"]:
        return "search_timeout"
    if not row["contact"]:
        return "approach_timeout"
    if not row["valid_attack"]:
        return "contact_without_valid_attack"
    if not row["block_break"]:
        return "attack_without_observed_break"
    return "break_without_inventory_pickup"


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["previous_action_mode"] != PREVIOUS_ACTION_DISABLED_ZERO:
        raise ValueError("exp13 evaluator requires disabled_zero intervention")
    if set(config["protected_splits"]) != {"student_holdout", "final_test"}:
        raise PermissionError("protected split declaration changed")
    evaluation = config["autonomous_evaluation"]
    if evaluation["split"] != "student_dev":
        raise PermissionError("only the predeclared student_dev split is permitted")
    seeds = [int(seed) for seed in evaluation["seeds"]]
    if seeds != [18500, 18501, 18502, 18503]:
        raise PermissionError("autonomous seeds differ from predeclaration")
    max_steps = int(evaluation["max_episode_steps"])
    if max_steps != 500 or evaluation["action_selection"] != "deterministic_argmax":
        raise ValueError("autonomous rollout protocol differs from predeclaration")
    policy = RecurrentTreechopPolicy.load(args.checkpoint)
    if policy.architecture.previous_action_mode != PREVIOUS_ACTION_DISABLED_ZERO:
        raise ValueError("checkpoint is not a disabled-zero actor")
    if policy.model.training:
        raise RuntimeError("checkpoint was not loaded in eval mode")
    if any(
        name.startswith("previous_action_embedding.")
        for name in policy.model.state_dict()
    ):
        raise RuntimeError("disabled checkpoint unexpectedly contains action embedding")

    if args.offline_gradle:
        enable_temporary_gradle_offline_mode()
    output_root = Path(args.output_root)
    trace_root = output_root / "autonomous_step_traces"
    trace_root.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    parity_rows: List[Dict[str, Any]] = []
    env = None
    started = time.perf_counter()
    try:
        env = make_telemetry_treechop_env(
            seed=seeds[0], max_episode_steps=max_steps, include_raycast=True
        )
        for seed in seeds:
            trace_path = trace_root / "seed{}.npz".format(seed)
            if trace_path.exists() and not args.overwrite:
                trace = load_trace(trace_path)
            else:
                env.seed(seed)
                observation = env.reset()
                adapter = LegalObservationAdapter(max_steps)
                hidden = None
                previous_action_token = START_ACTION_TOKEN
                recorder = RuntimeTraceRecorder(
                    checkpoint=args.checkpoint,
                    checkpoint_seed=29,
                    environment_seed=seed,
                    max_steps=max_steps,
                    previous_action_mode=PREVIOUS_ACTION_DISABLED_ZERO,
                )
                done = False
                info: Dict[str, Any] = {}
                step = 0
                while not done:
                    legal = (
                        adapter.reset(observation)
                        if step == 0
                        else adapter.adapt(observation, step)
                    )
                    action, probabilities, hidden, diagnostics = (
                        policy.predict_step_with_diagnostics(
                            legal.pov,
                            legal.vector,
                            previous_action_token,
                            hidden,
                        )
                    )
                    if float(np.max(np.abs(diagnostics["action_embedding"]))) != 0.0:
                        raise RuntimeError("disabled action slot became nonzero in rollout")
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
                recorder.metadata.update(
                    {
                        "model_eval_mode": True,
                        "actor_decisions": step,
                        "environment_steps": step,
                        "hidden_advances": step,
                        "config_path": str(config_path),
                        "config_sha256": file_sha256(config_path),
                    }
                )
                recorder.finalize(done, info)
                atomic_save_trace(trace_path, recorder.arrays())
                trace = load_trace(trace_path)
            integrity = validate_trace_integrity(trace)
            if not integrity["passed"]:
                raise RuntimeError("autonomous trace integrity failed: {}".format(integrity))
            parity = standalone_replay(args.checkpoint, trace)
            if not parity["passed"]:
                raise RuntimeError("standalone replay parity failed: {}".format(parity))
            row = dict(summarize_trace(trace), trace_path=str(trace_path))
            row["failure_taxonomy"] = failure_taxonomy(row)
            rows.append(row)
            parity_rows.append({"environment_seed": seed, **parity})
            atomic_csv(output_root / "autonomous_episode_summary.csv", rows)
            print(json.dumps(row, sort_keys=True), flush=True)
    finally:
        if env is not None:
            try:
                env.close()
            except psutil.NoSuchProcess as error:
                print("WARNING: Minecraft already exited during close: {}".format(error))

    periodic_fields = (
        "environment_seed",
        "max_period_1_streak",
        "max_period_2_cycle_streak",
        "max_period_3_cycle_streak",
        "max_period_4_cycle_streak",
        "dominant_period_1_to_4",
        "fraction_of_episode_in_dominant_period_1_to_4_cycle",
        "dominant_period_2_to_4_cycle_fraction",
        "dominant_action_bigram",
        "dominant_bigram_fraction",
        "time_to_first_action_transition",
        "pure_single_action_fixed_point",
    )
    atomic_csv(
        output_root / "periodic_cycle_summary.csv",
        [{key: row[key] for key in periodic_fields} for row in rows],
    )
    atomic_json(
        output_root / "autonomous_replay_parity.json",
        {"traces": parity_rows, "passed": all(row["passed"] for row in parity_rows)},
    )
    pure_fixed = sum(bool(row["pure_single_action_fixed_point"]) for row in rows)
    summary = {
        "experiment": config["experiment"],
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": file_sha256(Path(args.checkpoint)),
        "seeds": seeds,
        "episodes": len(rows),
        "max_episode_steps": max_steps,
        "deterministic_action_selection": "argmax",
        "pure_500_step_single_action_fixed_point_episode_count": pure_fixed,
        "median_action_transitions": float(np.median([row["action_transitions"] for row in rows])),
        "median_longest_identical_action_streak": float(
            np.median([row["longest_identical_action_streak"] for row in rows])
        ),
        "median_dominant_action_fraction": float(
            np.median([row["dominant_fraction"] for row in rows])
        ),
        "median_dominant_period_1_to_4_cycle_fraction": float(
            np.median(
                [row["fraction_of_episode_in_dominant_period_1_to_4_cycle"] for row in rows]
            )
        ),
        "episodes_below_0_80_dominant_period_1_to_4_cycle": sum(
            row["fraction_of_episode_in_dominant_period_1_to_4_cycle"] < 0.80
            for row in rows
        ),
        "episodes_with_action_transitions": sum(row["action_transitions"] > 0 for row in rows),
        "progression_counts": {
            "meaningful_interaction": sum(row["meaningful_interaction"] for row in rows),
            "approach": sum(row["approach"] for row in rows),
            "contact": sum(row["contact"] for row in rows),
            "valid_attack": sum(row["valid_attack"] for row in rows),
            "block_break": sum(row["block_break"] for row in rows),
            "pickup": sum(row["pickup"] for row in rows),
            "inventory_acquisition": sum(row["inventory_success"] for row in rows),
        },
        "failure_taxonomy": dict(Counter(row["failure_taxonomy"] for row in rows)),
        "teacher_actions_executed": sum(row["teacher_actions_executed"] for row in rows),
        "privileged_actor_inputs": sum(row["privileged_actor_inputs"] for row in rows),
        "max_abs_disabled_action_channel": max(
            row["disabled_action_channel_max_abs"] for row in rows
        ),
        "success_definition": "inventory natural log delta >= 1",
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "protected_splits_accessed": False,
    }
    if summary["teacher_actions_executed"] != 0 or summary["privileged_actor_inputs"] != 0:
        raise RuntimeError("autonomous policy boundary was violated")
    if summary["max_abs_disabled_action_channel"] != 0.0:
        raise RuntimeError("disabled action channel was nonzero")
    atomic_json(output_root / "autonomous_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
