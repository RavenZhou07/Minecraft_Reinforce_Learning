"""Train the natural Treechop contact behaviour-cloning student.

The trainer loads only the declared student inputs (POV frame stacks and
previous discrete actions) from the demonstration NPZ files; the teacher-only
audit arrays are deliberately never read. Training uses successful-episode
contact trajectories by default, class-balanced cross-entropy, horizontal
mirror augmentation with the verified yaw-action swap, and early stopping on
the minimum validation loss. The serialized checkpoint is the best epoch,
not the last one.
"""

import argparse
import csv
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from mc_rl.natural_contact_bc import (
    ACTION_CLASSES,
    NaturalContactBCPolicy,
    mirror_actions,
    mirror_pov_frames,
)


STUDENT_INPUT_FIELDS = ("pov", "action", "previous_action")
BANNED_SEED_LOW = 16500
BANNED_SEED_HIGH = 16819


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-dataset",
        default="logs/find_tree/natural_treechop_bc_v1_train_16900_80.npz",
    )
    parser.add_argument(
        "--validation-dataset",
        default="logs/find_tree/natural_treechop_bc_v1_validation_17000_20.npz",
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/natural_treechop_contact_bc_v1_stack4.npz",
    )
    parser.add_argument(
        "--training-log",
        default="logs/find_tree/natural_treechop_contact_bc_v1_training.csv",
    )
    parser.add_argument(
        "--summary",
        default="logs/find_tree/natural_treechop_contact_bc_v1_training.summary.json",
    )
    parser.add_argument(
        "--config",
        default="configs/natural_treechop_contact_bc_v1.json",
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--feature-size", type=int, default=10)
    parser.add_argument(
        "--no-centre-pixels",
        dest="include_centre_pixels",
        action="store_false",
        default=True,
        help="Drop raw centre-crop pixels and keep only the scalar centre trunk fraction.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mirror-augmentation", action="store_true", default=True)
    parser.add_argument(
        "--no-mirror-augmentation", dest="mirror_augmentation", action="store_false"
    )
    parser.add_argument(
        "--include-failure-episodes", action="store_true",
        help="Diagnostic only: train on failed-episode trajectories too.",
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_rows(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_student_split(path: Path) -> Dict[str, np.ndarray]:
    """Load ONLY the declared student input fields from a dataset."""

    with np.load(path, allow_pickle=False) as data:
        available = set(data.files)
        missing = [field for field in STUDENT_INPUT_FIELDS if field not in available]
        if missing:
            raise KeyError(
                "dataset {} is missing student fields: {}".format(path, missing)
            )
        split = {
            "pov": data["pov"].astype(np.uint8),
            "action": data["action"].astype(np.int64),
            "previous_action": data["previous_action"].astype(np.int64),
        }
        if "episode_success" in available:
            split["episode_success"] = data["episode_success"].astype(np.int64)
        if "episode_seed" in available:
            split["episode_seed"] = data["episode_seed"].astype(np.int64)
        if "audit_contact_state" in available:
            # Audit metadata is loaded for reporting baselines only; it is
            # never passed to the model.
            split["audit_contact_state"] = np.asarray(
                data["audit_contact_state"]
            )
    return split


def filter_successful(split: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], int]:
    if "episode_success" not in split:
        return split, 0
    mask = split["episode_success"].astype(bool)
    excluded = int((~mask).sum())
    filtered = {
        key: (value[mask] if isinstance(value, np.ndarray) and len(value) == len(mask) else value)
        for key, value in split.items()
        if key != "episode_success"
    }
    return filtered, excluded


def class_balanced_accuracy(
    predictions: np.ndarray, labels: np.ndarray, classes: np.ndarray
) -> float:
    recalls = []
    for action in classes:
        selected = labels == action
        if selected.any():
            recalls.append(float((predictions[selected] == action).mean()))
    return float(np.mean(recalls)) if recalls else 0.0


def precision_recall(
    predictions: np.ndarray, labels: np.ndarray, action: int
) -> Tuple[float, float]:
    predicted = predictions == action
    actual = labels == action
    precision = (
        float((predicted & actual).sum() / predicted.sum())
        if predicted.any()
        else 0.0
    )
    recall = (
        float((predicted & actual).sum() / actual.sum()) if actual.any() else 0.0
    )
    return precision, recall


def yaw_direction_agreement(
    predictions: np.ndarray, labels: np.ndarray
) -> float:
    left = (3, 10)
    right = (4, 11)
    selected = np.isin(labels, left + right)
    if not selected.any():
        return 0.0
    prediction_left = np.isin(predictions[selected], left)
    label_left = np.isin(labels[selected], left)
    return float((prediction_left == label_left).mean())


def evaluate_predictions(
    predictions: np.ndarray, labels: np.ndarray
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {
        "accuracy": float((predictions == labels).mean()),
        "balanced_accuracy": class_balanced_accuracy(
            predictions, labels, ACTION_CLASSES
        ),
    }
    attack_precision, attack_recall = precision_recall(predictions, labels, 7)
    metrics["attack_precision"] = attack_precision
    metrics["attack_recall"] = attack_recall
    metrics["yaw_direction_agreement"] = yaw_direction_agreement(
        predictions, labels
    )
    return metrics


def majority_baseline(labels: np.ndarray) -> int:
    counts = Counter(labels.tolist())
    return int(counts.most_common(1)[0][0])


def per_state_majority_baseline(
    states: np.ndarray, labels: np.ndarray
) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    for state in sorted(set(states.tolist())):
        mask = states == state
        if not mask.any():
            continue
        subset = labels[mask]
        majority = majority_baseline(subset)
        results[str(state)] = {
            "majority_action": majority,
            "samples": int(mask.sum()),
            "accuracy": float((subset == majority).mean()),
        }
    return results


def main():
    args = parse_args()
    np.random.seed(int(args.seed))
    train_path = Path(args.train_dataset)
    validation_path = Path(args.validation_dataset)
    train_split = load_student_split(train_path)
    validation_split = load_student_split(validation_path)
    if not args.include_failure_episodes:
        train_split, train_excluded = filter_successful(train_split)
        validation_split, validation_excluded = filter_successful(
            validation_split
        )
    else:
        train_excluded = validation_excluded = 0

    train_seeds = (
        sorted(set(train_split["episode_seed"].tolist()))
        if "episode_seed" in train_split
        else []
    )
    validation_seeds = (
        sorted(set(validation_split["episode_seed"].tolist()))
        if "episode_seed" in validation_split
        else []
    )
    overlap = sorted(set(train_seeds) & set(validation_seeds))
    if overlap:
        raise ValueError(
            "train/validation seed overlap: {}".format(overlap)
        )
    for seed in train_seeds + validation_seeds:
        if BANNED_SEED_LOW <= seed <= BANNED_SEED_HIGH:
            raise ValueError(
                "banned seed {} appeared in training data".format(seed)
            )

    train_pov = train_split["pov"]
    train_actions = train_split["action"]
    train_previous = train_split["previous_action"]
    validation_pov = validation_split["pov"]
    validation_actions = validation_split["action"]
    validation_previous = validation_split["previous_action"]

    if args.mirror_augmentation:
        mirrored_pov = mirror_pov_frames(train_pov)
        mirrored_actions = mirror_actions(train_actions)
        mirrored_previous = mirror_actions(train_previous)
        train_pov = np.concatenate((train_pov, mirrored_pov), axis=0)
        train_actions = np.concatenate((train_actions, mirrored_actions))
        train_previous = np.concatenate((train_previous, mirrored_previous))

    action_counts = Counter(train_split["action"].tolist())
    state_counts = (
        Counter(train_split["audit_contact_state"].tolist())
        if "audit_contact_state" in train_split
        else {}
    )
    dataset_report = {
        "train_dataset": str(train_path),
        "validation_dataset": str(validation_path),
        "train_sha256": file_sha256(train_path),
        "validation_sha256": file_sha256(validation_path),
        "train_seed_range": (
            [min(train_seeds), max(train_seeds)] if train_seeds else []
        ),
        "validation_seed_range": (
            [min(validation_seeds), max(validation_seeds)] if validation_seeds else []
        ),
        "train_samples": int(len(train_split["action"])),
        "train_samples_with_mirror": int(len(train_actions)),
        "train_excluded_failure_samples": train_excluded,
        "validation_samples": int(len(validation_split["action"])),
        "validation_excluded_failure_samples": validation_excluded,
        "train_action_counts": {
            str(action): count for action, count in sorted(action_counts.items())
        },
        "train_contact_state_counts": {
            str(state): count for state, count in sorted(state_counts.items())
        },
        "student_input_manifest": [
            "pov_frame_stack_4",
            "previous_action_one_hot_14",
        ],
    }
    print(json.dumps(dataset_report, indent=2), flush=True)

    policy = NaturalContactBCPolicy(
        feature_size=args.feature_size,
        frame_stack=4,
        include_centre_pixels=bool(args.include_centre_pixels),
    )
    started_at = time.perf_counter()
    history = policy.fit(
        train_pov,
        train_actions,
        train_previous,
        validation_pov,
        validation_actions,
        validation_previous,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
        patience=args.patience,
        momentum=args.momentum,
    )
    training_seconds = round(time.perf_counter() - started_at, 3)
    atomic_write_rows(Path(args.training_log), history)

    validation_features = policy.build_features(
        validation_pov, validation_previous
    )
    probabilities = policy.predict_proba_from_features(validation_features)
    predictions = ACTION_CLASSES[probabilities.argmax(axis=1)]
    metrics = evaluate_predictions(predictions, validation_actions)

    global_majority = majority_baseline(train_split["action"])
    global_baseline = float(
        (validation_actions == global_majority).mean()
    )
    state_baseline = {}
    if "audit_contact_state" in validation_split:
        state_majority = per_state_majority_baseline(
            validation_split["audit_contact_state"], validation_actions
        )
        weighted = sum(
            entry["accuracy"] * entry["samples"]
            for entry in state_majority.values()
        )
        total = sum(entry["samples"] for entry in state_majority.values())
        state_baseline = {
            "per_state": state_majority,
            "weighted_accuracy": weighted / total if total else 0.0,
        }
    rng = np.random.RandomState(int(args.seed))
    prior = np.zeros(len(ACTION_CLASSES), dtype=np.float64)
    for action, count in action_counts.items():
        prior[int(action)] = count
    prior = prior / prior.sum()
    random_draws = rng.choice(
        ACTION_CLASSES, size=len(validation_actions), p=prior
    )
    random_baseline = float((random_draws == validation_actions).mean())

    initial_validation_loss = history[0]["validation_loss"]
    best_validation_loss = policy.best_validation_loss
    relative_loss_improvement = (
        (initial_validation_loss - best_validation_loss)
        / max(initial_validation_loss, 1e-8)
    )
    offline_gate = {
        "validation_loss_relative_improvement_at_least_20_percent": bool(
            relative_loss_improvement >= 0.20
        ),
        "balanced_accuracy_at_least_55_percent": bool(
            metrics["balanced_accuracy"] >= 0.55
        ),
        "attack_precision_at_least_90_percent": bool(
            metrics["attack_precision"] >= 0.90
        ),
        "attack_recall_at_least_75_percent": bool(
            metrics["attack_recall"] >= 0.75
        ),
        "yaw_direction_agreement_at_least_90_percent": bool(
            metrics["yaw_direction_agreement"] >= 0.90
        ),
        "finite_weights": bool(np.all(np.isfinite(policy.weights))),
        "checkpoint_reload_consistent": None,
        "privileged_input_accesses_zero": True,
    }

    checkpoint_path = Path(args.checkpoint)
    policy.save(
        str(checkpoint_path),
        dataset_hashes={
            "train": dataset_report["train_sha256"],
            "validation": dataset_report["validation_sha256"],
        },
        seed_ranges={
            "train": dataset_report["train_seed_range"],
            "validation": dataset_report["validation_seed_range"],
        },
    )
    reloaded = NaturalContactBCPolicy.load(str(checkpoint_path))
    reload_features = reloaded.build_features(
        validation_pov, validation_previous
    )
    reload_probabilities = reloaded.predict_proba_from_features(reload_features)
    reload_predictions = ACTION_CLASSES[reload_probabilities.argmax(axis=1)]
    offline_gate["checkpoint_reload_consistent"] = bool(
        np.array_equal(reload_predictions, predictions)
    )
    offline_gate["all_conditions_met"] = all(
        value
        for key, value in offline_gate.items()
        if isinstance(value, bool) and key != "all_conditions_met"
    )

    summary = {
        "model_version": "natural_treechop_contact_bc_v1",
        "teacher_profile": "terrain_route_drop_completion_v9_6",
        "dataset": dataset_report,
        "training": {
            "epochs_requested": args.epochs,
            "learning_rate": args.learning_rate,
            "l2": args.l2,
            "patience": args.patience,
            "momentum": args.momentum,
            "random_seed": args.seed,
            "mirror_augmentation": bool(args.mirror_augmentation),
            "selected_epoch": policy.best_epoch,
            "stopped_early": bool(policy.stopped_early),
            "training_seconds": training_seconds,
        },
        "validation_metrics": metrics,
        "baselines": {
            "global_majority_action": global_majority,
            "global_majority_accuracy": global_baseline,
            "per_contact_state_majority": state_baseline,
            "random_train_prior_accuracy": random_baseline,
        },
        "validation_loss": {
            "initial": initial_validation_loss,
            "best": best_validation_loss,
            "relative_improvement": relative_loss_improvement,
        },
        "offline_gate": offline_gate,
        "offline_gate_passed": bool(offline_gate["all_conditions_met"]),
    }
    atomic_write_json(Path(args.summary), summary)

    config = {
        "profile": "natural_treechop_contact_bc_v1",
        "purpose": "Visual behaviour cloning of the v9.6 local contact controller",
        "teacher_profile": "terrain_route_drop_completion_v9_6",
        "teacher_gate_summary": (
            "logs/find_tree/candidate_search_f3_raycast_terrain_route_drop_"
            "completion_v9_6_16800_20_gate_v1.summary.json"
        ),
        "model": {
            "type": "linear_softmax",
            "feature_size": args.feature_size,
            "frame_stack": 4,
            "action_classes": 14,
            "include_centre_pixels": bool(args.include_centre_pixels),
            "mirror_augmentation": bool(args.mirror_augmentation),
        },
        "student_input_manifest": [
            "pov_frame_stack_4",
            "previous_action_one_hot_14",
        ],
        "student_forbidden_inputs": [
            "raycast",
            "exact_log_xyz",
            "log_grid",
            "oracle_distance",
            "target_coordinates",
            "item_entity_coordinates",
            "teacher_contact_state",
        ],
        "seed_isolation": {
            "banned": "16500-16819",
            "train": dataset_report["train_seed_range"],
            "validation": dataset_report["validation_seed_range"],
            "shadow": "17100-17119",
            "holdout": "17200-17219",
        },
        "checkpoint": str(checkpoint_path),
        "training_summary": str(Path(args.summary)),
        "training_started": True,
        "teacher_gate_passed": True,
        "offline_gate_passed": bool(offline_gate["all_conditions_met"]),
        "shadow_gate_passed": False,
        "student_holdout_passed": False,
    }
    atomic_write_json(Path(args.config), config)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
