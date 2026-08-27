"""Train the complete Natural Treechop student from legal observations."""

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from mc_rl.experiments import append_experiment, file_sha256
from mc_rl.natural_treechop_bc import (
    ACTION_CLASSES,
    PHASE_CLASSES,
    NaturalTreechopBCPolicy,
    balanced_accuracy,
    build_causal_action_history,
    phase_ids,
)
from mc_rl.vision import build_frame_stacks


STUDENT_DATASET_FIELDS = (
    "pov",
    "legal_vector",
    "action",
    "previous_action",
    "episode",
    "episode_seed",
    "episode_step",
    "episode_success",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dataset", required=True)
    parser.add_argument("--validation-dataset", required=True)
    parser.add_argument("--dagger-dataset", action="append", default=[])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--training-log", required=True)
    parser.add_argument("--config-output", required=True)
    parser.add_argument("--feature-size", type=int, default=6)
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--action-history", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--class-weight-power", type=float, default=0.5)
    parser.add_argument("--no-phase-head", dest="use_phase_head", action="store_false", default=True)
    parser.add_argument("--include-failure-teacher", action="store_true")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--hypothesis", required=True)
    return parser.parse_args()


def atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def atomic_history(path: Path, histories: Dict[str, List[Dict[str, float]]]) -> None:
    rows = []
    for head, history in histories.items():
        rows.extend(dict(row, head=head) for row in history)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fields = [
        "head", "epoch", "train_loss", "validation_loss", "train_accuracy",
        "validation_accuracy", "validation_balanced_accuracy",
    ]
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def load_dataset(path: Path, include_failure_teacher: bool) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        missing = [field for field in STUDENT_DATASET_FIELDS if field not in data.files]
        if missing or "audit_coarse_phase" not in data.files:
            raise KeyError("dataset {} is missing {}".format(path, missing or ["audit_coarse_phase"]))
        result = {field: np.asarray(data[field]) for field in STUDENT_DATASET_FIELDS}
        result["phase"] = np.asarray(data["audit_coarse_phase"])
        result["source"] = (
            np.asarray(data["source"])
            if "source" in data.files
            else np.asarray(["teacher"] * len(result["action"]))
        )
        result["seed_split"] = str(data["seed_split"]) if "seed_split" in data.files else ""
        result["student_input_manifest"] = (
            tuple(str(value) for value in data["student_input_manifest"].tolist())
            if "student_input_manifest" in data.files
            else ()
        )
    if not include_failure_teacher:
        keep = (result["episode_success"].astype(bool)) | (result["source"] == "dagger")
        for key, value in list(result.items()):
            if isinstance(value, np.ndarray) and value.ndim > 0 and len(value) == len(keep):
                result[key] = value[keep]
    return result


def prepared(dataset: Dict[str, np.ndarray], frame_stack: int, action_history: int):
    return {
        "pov": build_frame_stacks(dataset["pov"], dataset["episode"], frame_stack),
        "legal_vector": dataset["legal_vector"].astype(np.float32),
        "history": build_causal_action_history(
            dataset["previous_action"], dataset["episode"], action_history
        ),
        "action": dataset["action"].astype(np.int64),
        "phase": dataset["phase"].astype(str),
        "episode_seed": dataset["episode_seed"].astype(np.int64),
        "source": dataset["source"].astype(str),
    }


def concatenate(datasets: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    return {
        key: np.concatenate([dataset[key] for dataset in datasets], axis=0)
        for key in datasets[0]
    }


def prediction_metrics(policy, split):
    probabilities, phase_probabilities = policy.probabilities(
        split["pov"], split["legal_vector"], split["history"]
    )
    predictions = ACTION_CLASSES[probabilities.argmax(axis=1)]
    labels = split["action"]
    metrics = {
        "accuracy": float((predictions == labels).mean()),
        "balanced_accuracy": balanced_accuracy(predictions, labels, ACTION_CLASSES),
        "action_counts": {
            str(key): int(value) for key, value in sorted(Counter(labels.tolist()).items())
        },
        "predicted_action_counts": {
            str(key): int(value) for key, value in sorted(Counter(predictions.tolist()).items())
        },
    }
    if phase_probabilities is not None:
        phase_labels = phase_ids(split["phase"])
        phase_predictions = PHASE_CLASSES[phase_probabilities.argmax(axis=1)]
        metrics["phase_accuracy"] = float((phase_predictions == phase_labels).mean())
        metrics["phase_balanced_accuracy"] = balanced_accuracy(
            phase_predictions, phase_labels, PHASE_CLASSES
        )
    return metrics, predictions


def main():
    args = parse_args()
    train_path = Path(args.train_dataset)
    validation_path = Path(args.validation_dataset)
    base_train = prepared(
        load_dataset(train_path, args.include_failure_teacher),
        args.frame_stack,
        args.action_history,
    )
    train_parts = [base_train]
    for dagger_path in args.dagger_dataset:
        train_parts.append(
            prepared(
                load_dataset(Path(dagger_path), include_failure_teacher=True),
                args.frame_stack,
                args.action_history,
            )
        )
    train = concatenate(train_parts)
    validation = prepared(
        load_dataset(validation_path, args.include_failure_teacher),
        args.frame_stack,
        args.action_history,
    )
    overlap = sorted(
        set(train["episode_seed"].tolist()) & set(validation["episode_seed"].tolist())
    )
    if overlap:
        raise ValueError("train/validation seed overlap: {}".format(overlap))

    policy = NaturalTreechopBCPolicy(
        feature_size=args.feature_size,
        frame_stack=args.frame_stack,
        action_history=args.action_history,
        use_phase_head=args.use_phase_head,
    )
    started = time.perf_counter()
    histories = policy.fit(
        train["pov"],
        train["legal_vector"],
        train["history"],
        train["action"],
        train["phase"],
        validation["pov"],
        validation["legal_vector"],
        validation["history"],
        validation["action"],
        validation["phase"],
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
        patience=args.patience,
        class_weight_power=args.class_weight_power,
    )
    runtime = round(time.perf_counter() - started, 3)
    atomic_history(Path(args.training_log), histories)
    train_metrics, _ = prediction_metrics(policy, train)
    validation_metrics, validation_predictions = prediction_metrics(policy, validation)
    dataset_hashes = {
        "train": file_sha256(train_path),
        "validation": file_sha256(validation_path),
    }
    for index, dagger_path in enumerate(args.dagger_dataset):
        dataset_hashes["dagger_{}".format(index + 1)] = file_sha256(Path(dagger_path))
    seed_manifest = "configs/seeds/natural_treechop_v1.json"
    policy.save(args.checkpoint, dataset_hashes, seed_manifest)
    reloaded = NaturalTreechopBCPolicy.load(args.checkpoint)
    reload_probabilities, _ = reloaded.probabilities(
        validation["pov"], validation["legal_vector"], validation["history"]
    )
    reload_predictions = ACTION_CLASSES[reload_probabilities.argmax(axis=1)]
    reload_consistent = bool(np.array_equal(validation_predictions, reload_predictions))

    summary = {
        "model_version": policy.model_version,
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": file_sha256(Path(args.checkpoint)),
        "dataset_hashes": dataset_hashes,
        "seed_manifest": seed_manifest,
        "student_input_manifest": list(policy.student_input_manifest),
        "train_only_privileged_supervision": ["audit_coarse_phase"],
        "forbidden_student_inputs": [
            "raycast", "tree_xyz", "target_vector", "log_grid",
            "teacher_phase_label", "teacher_candidate_list", "reachability",
        ],
        "config": {
            "feature_size": args.feature_size,
            "frame_stack": args.frame_stack,
            "action_history": args.action_history,
            "use_phase_head": args.use_phase_head,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "l2": args.l2,
            "patience": args.patience,
            "class_weight_power": args.class_weight_power,
            "include_failure_teacher": args.include_failure_teacher,
        },
        "data": {
            "train_samples": int(len(train["action"])),
            "validation_samples": int(len(validation["action"])),
            "train_seeds": sorted(set(train["episode_seed"].tolist())),
            "validation_seeds": sorted(set(validation["episode_seed"].tolist())),
            "source_counts": dict(Counter(train["source"].tolist())),
        },
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "selected_epochs": {
            "phase": policy.phase_best_epoch if policy.use_phase_head else None,
            "action": policy.action_best_epoch,
        },
        "checkpoint_reload_consistent": reload_consistent,
        "privileged_actor_inputs": 0,
        "training_seconds": runtime,
    }
    atomic_json(Path(args.summary), summary)
    config = {
        "profile": args.experiment_id,
        "model": summary["config"],
        "student_input_manifest": summary["student_input_manifest"],
        "train_only_privileged_supervision": summary["train_only_privileged_supervision"],
        "forbidden_student_inputs": summary["forbidden_student_inputs"],
        "seed_manifest": seed_manifest,
        "datasets": dataset_hashes,
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": summary["checkpoint_sha256"],
        "training_summary": args.summary,
    }
    atomic_json(Path(args.config_output), config)
    append_experiment(
        {
            "experiment_id": args.experiment_id,
            "hypothesis": args.hypothesis,
            "config": config,
            "seed_manifest": seed_manifest,
            "dataset_hashes": dataset_hashes,
            "checkpoint": {
                "path": args.checkpoint,
                "sha256": summary["checkpoint_sha256"],
            },
            "metrics": {
                "train": train_metrics,
                "validation": validation_metrics,
                "checkpoint_reload_consistent": reload_consistent,
            },
            "runtime_seconds": runtime,
            "conclusion": "Offline end-to-end BC training completed; autonomous rollout is required to judge utility.",
            "status": "kept_for_rollout",
        }
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
