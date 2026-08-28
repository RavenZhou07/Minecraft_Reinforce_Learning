"""Evaluate disabled-zero recurrent checkpoints on frozen observation sequences."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from mc_rl.actions import ACTION_NAMES
from mc_rl.experiments import file_sha256
from mc_rl.natural_treechop_bc import balanced_accuracy
from mc_rl.recurrent_treechop_bc import (
    ACTION_COUNT,
    PREVIOUS_ACTION_DISABLED_ZERO,
    START_ACTION_TOKEN,
    RecurrentTreechopPolicy,
    load_episode_sequences,
)
from mc_rl.runtime_observability import periodic_cycle_diagnostics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--training-summary")
    parser.add_argument("--output-root", default="artifacts/exp13")
    return parser.parse_args()


def atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def atomic_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def replay_episode(policy, episode, mode: str):
    rng = np.random.RandomState(1300 + episode.seed)
    hidden = None
    predicted_previous = START_ACTION_TOKEN
    outputs = {name: [] for name in ("hidden", "logits", "probabilities", "argmax", "combined")}
    zero_max = 0.0
    for step in range(episode.length):
        if mode == "teacher_previous_action":
            token = int(episode.previous_action_token[step])
        elif mode == "model_predicted_previous_action":
            token = int(predicted_previous)
        elif mode == "all_start":
            token = START_ACTION_TOKEN
        elif mode == "all_noop":
            token = 0
        elif mode == "random_valid_action":
            token = int(rng.randint(0, ACTION_COUNT))
        else:
            raise ValueError("unknown history mode: {}".format(mode))
        action, probabilities, hidden, diagnostics = policy.predict_step_with_diagnostics(
            episode.pov[step], episode.legal_vector[step], token, hidden
        )
        predicted_previous = action
        outputs["hidden"].append(hidden.detach().cpu().numpy().reshape(-1))
        outputs["logits"].append(diagnostics["logits"])
        outputs["probabilities"].append(probabilities)
        outputs["argmax"].append(action)
        outputs["combined"].append(diagnostics["combined_embedding"])
        zero_max = max(
            zero_max,
            float(np.max(np.abs(np.asarray(diagnostics["action_embedding"])))),
        )
    return {key: np.asarray(value) for key, value in outputs.items()}, zero_max


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["previous_action_mode"] != PREVIOUS_ACTION_DISABLED_ZERO:
        raise ValueError("history invariance evaluator requires disabled_zero")
    if set(config["protected_splits"]) != {"student_holdout", "final_test"}:
        raise PermissionError("protected split declaration changed")
    validation = config["datasets"]["validation"]
    validation_path = Path(validation["path"])
    if file_sha256(validation_path) != validation["sha256"]:
        raise RuntimeError("validation dataset hash mismatch")
    policy = RecurrentTreechopPolicy.load(args.checkpoint)
    if policy.architecture.previous_action_mode != PREVIOUS_ACTION_DISABLED_ZERO:
        raise ValueError("checkpoint is not a disabled-zero actor")
    episodes = load_episode_sequences(validation_path, include_failure_teacher=False)
    modes = (
        "teacher_previous_action",
        "model_predicted_previous_action",
        "all_start",
        "all_noop",
        "random_valid_action",
    )
    rows: List[Dict[str, Any]] = []
    aggregate_labels: Dict[str, List[np.ndarray]] = {mode: [] for mode in modes}
    aggregate_predictions: Dict[str, List[np.ndarray]] = {mode: [] for mode in modes}
    maximum_errors = {
        mode: {key: 0.0 for key in ("hidden", "logits", "probabilities", "combined")}
        for mode in modes
    }
    exact = {mode: True for mode in modes}
    max_zero_channel = 0.0
    for episode in episodes:
        replayed = {}
        for mode in modes:
            replayed[mode], zero_max = replay_episode(policy, episode, mode)
            max_zero_channel = max(max_zero_channel, zero_max)
        reference = replayed[modes[0]]
        for mode in modes:
            current = replayed[mode]
            for key in ("hidden", "logits", "probabilities", "combined"):
                difference = np.abs(
                    current[key].astype(np.float64) - reference[key].astype(np.float64)
                )
                error = float(difference.max()) if difference.size else 0.0
                maximum_errors[mode][key] = max(maximum_errors[mode][key], error)
                exact[mode] = exact[mode] and np.array_equal(current[key], reference[key])
            exact[mode] = exact[mode] and np.array_equal(current["argmax"], reference["argmax"])
            predictions = current["argmax"].astype(np.int64)
            labels = episode.action.astype(np.int64)
            aggregate_labels[mode].append(labels)
            aggregate_predictions[mode].append(predictions)
            repetition = periodic_cycle_diagnostics(predictions)
            rows.append(
                {
                    "episode_seed": episode.seed,
                    "steps": episode.length,
                    "action_history_mode": mode,
                    "accuracy": float(np.mean(predictions == labels)),
                    "balanced_accuracy": balanced_accuracy(
                        predictions, labels, np.arange(ACTION_COUNT)
                    ),
                    "action_transitions": int(np.sum(predictions[1:] != predictions[:-1])),
                    "dominant_action_fraction": max(
                        np.bincount(predictions, minlength=ACTION_COUNT)
                    ) / len(predictions),
                    **repetition,
                }
            )

    aggregate = {}
    for mode in modes:
        labels = np.concatenate(aggregate_labels[mode])
        predictions = np.concatenate(aggregate_predictions[mode])
        aggregate[mode] = {
            "samples": int(len(labels)),
            "accuracy": float(np.mean(predictions == labels)),
            "balanced_accuracy": balanced_accuracy(
                predictions, labels, np.arange(ACTION_COUNT)
            ),
            "exactly_equal_to_teacher_history": bool(exact[mode]),
            "maximum_absolute_errors": maximum_errors[mode],
        }
    tolerance = 1e-7
    passed = bool(
        max_zero_channel == 0.0
        and all(
            np.array_equal(
                np.concatenate(aggregate_predictions[mode]),
                np.concatenate(aggregate_predictions[modes[0]]),
            )
            and all(error <= tolerance for error in maximum_errors[mode].values())
            for mode in modes
        )
    )
    output_root = Path(args.output_root)
    atomic_csv(output_root / "recorded_observation_replay.csv", rows)
    invariance = {
        "experiment": config["experiment"],
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": file_sha256(Path(args.checkpoint)),
        "validation_dataset_sha256": validation["sha256"],
        "history_modes": list(modes),
        "strict_tolerance": tolerance,
        "max_abs_disabled_action_channel": max_zero_channel,
        "aggregate": aggregate,
        "passed": passed,
    }
    atomic_json(output_root / "zero_channel_invariance.json", invariance)

    if args.training_summary:
        training = json.loads(Path(args.training_summary).read_text(encoding="utf-8"))
        atomic_json(
            output_root / "training_metrics.json",
            {
                "checkpoint": training["checkpoint"],
                "checkpoint_sha256": training["checkpoint_sha256"],
                "training_epochs": training["training_epochs"],
                "best_epoch": training["best_epoch"],
                "train_metrics": training["train_metrics"],
                "initialization_pairing": training["initialization_pairing"],
                "zero_channel_assertion_passed": training["zero_channel_assertion_passed"],
            },
        )
        atomic_json(output_root / "validation_metrics.json", training["validation_metrics"])
        per_action_rows = []
        for action_name in ACTION_NAMES:
            values = training["validation_metrics"]["per_action"][action_name]
            per_action_rows.append(
                {
                    "action_id": values["id"],
                    "action_name": action_name,
                    "support": values["support"],
                    "recall": values["recall"],
                    "precision": values["precision"],
                }
            )
        atomic_csv(output_root / "per_action_recall.csv", per_action_rows)
    print(json.dumps(invariance, indent=2), flush=True)
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
