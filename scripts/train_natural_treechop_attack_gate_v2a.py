"""Train the binary POV-only attack permission gate for BC v2a."""

import argparse
import csv
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from mc_rl.natural_attack_gate_bc import (
    ATTACK,
    GATE_CONTACT_STATES,
    HOLD,
    MODEL_VERSION,
    NaturalAttackGatePolicy,
    attack_gate_labels,
    attack_gate_sample_mask,
)
from mc_rl.natural_contact_bc import mirror_actions, mirror_pov_frames


REQUIRED_FIELDS = (
    "pov",
    "action",
    "previous_action",
    "episode_success",
    "episode_seed",
    "audit_contact_state",
)
BANNED_SEED_RANGE = (16500, 16819)


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
        default="checkpoints/natural_treechop_attack_gate_bc_v2a_experimental.npz",
    )
    parser.add_argument(
        "--training-log",
        default="logs/find_tree/natural_treechop_attack_gate_bc_v2a_training.csv",
    )
    parser.add_argument(
        "--summary",
        default="logs/find_tree/natural_treechop_attack_gate_bc_v2a_training.summary.json",
    )
    parser.add_argument(
        "--config", default="configs/natural_treechop_attack_gate_bc_v2a.json"
    )
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=300)
    parser.add_argument("--feature-size", type=int, default=10)
    parser.add_argument(
        "--hard-negative-repeat",
        type=int,
        default=1,
        help=(
            "Repeat training HOLD samples with audit raycast log+in_range. "
            "Audit flags select weights only and are never model inputs."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-mirror-augmentation", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def atomic_write_rows(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def load_attack_gate_split(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        available = set(data.files)
        missing = [field for field in REQUIRED_FIELDS if field not in available]
        if missing:
            raise KeyError("dataset {} is missing {}".format(path, missing))
        result = {
            "pov": data["pov"].astype(np.uint8),
            "environment_action": data["action"].astype(np.int64),
            "previous_action": data["previous_action"].astype(np.int64),
            "episode_success": data["episode_success"].astype(np.int64),
            "episode_seed": data["episode_seed"].astype(np.int64),
            "contact_state": np.asarray(data["audit_contact_state"]).astype(str),
        }
        if "episode_step" in available:
            result["episode_step"] = data["episode_step"].astype(np.int64)
        for field in ("audit_raycast_is_log", "audit_raycast_in_range"):
            if field in available:
                result[field] = np.asarray(data[field]).astype(np.float32)
    return result


def select_attack_gate_samples(
    split: Dict[str, np.ndarray]
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    mask = attack_gate_sample_mask(
        split["contact_state"], split["episode_success"]
    )
    selected = {
        key: value[mask]
        for key, value in split.items()
        if isinstance(value, np.ndarray) and len(value) == len(mask)
    }
    selected["label"] = attack_gate_labels(selected["environment_action"])
    labels = selected["label"]
    states = Counter(selected["contact_state"].tolist())
    report = {
        "total_contact_samples": int(len(mask)),
        "selected_successful_gate_samples": int(mask.sum()),
        "excluded_samples": int((~mask).sum()),
        "hold_samples": int((labels == HOLD).sum()),
        "attack_samples": int((labels == ATTACK).sum()),
        "contact_state_counts": dict(sorted(states.items())),
    }
    if "audit_raycast_is_log" in selected:
        positives = labels == ATTACK
        report["attack_label_raycast_audit"] = {
            "positive_samples": int(positives.sum()),
            "raycast_log_fraction": (
                float(selected["audit_raycast_is_log"][positives].mean())
                if positives.any()
                else 0.0
            ),
            "raycast_in_range_fraction": (
                float(selected["audit_raycast_in_range"][positives].mean())
                if positives.any()
                else 0.0
            ),
        }
    return selected, report


def hard_negative_training_indices(
    split: Dict[str, np.ndarray], repeat_factor: int
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Deterministically repeat near-attack HOLD examples for training only."""

    if repeat_factor < 1:
        raise ValueError("hard-negative repeat factor must be at least one")
    required = ("label", "audit_raycast_is_log", "audit_raycast_in_range")
    missing = [field for field in required if field not in split]
    if missing:
        raise KeyError("hard-negative audit fields missing: {}".format(missing))
    hard_mask = (
        (split["label"] == HOLD)
        & split["audit_raycast_is_log"].astype(bool)
        & split["audit_raycast_in_range"].astype(bool)
    )
    base = np.arange(len(split["label"]), dtype=np.int64)
    hard = np.flatnonzero(hard_mask).astype(np.int64)
    if repeat_factor > 1 and len(hard):
        indices = np.concatenate(
            (base, np.tile(hard, repeat_factor - 1)), axis=0
        )
    else:
        indices = base
    return indices, {
        "repeat_factor": int(repeat_factor),
        "original_samples": int(len(base)),
        "hard_negative_samples": int(len(hard)),
        "added_hard_negative_samples": int(len(indices) - len(base)),
        "training_samples_before_mirror": int(len(indices)),
    }


def binary_metrics(
    probabilities: np.ndarray, labels: np.ndarray, threshold: float
) -> Dict[str, float]:
    predictions = (np.asarray(probabilities) >= float(threshold)).astype(np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    positive = labels == ATTACK
    negative = labels == HOLD
    predicted_positive = predictions == ATTACK
    true_positive = int((predicted_positive & positive).sum())
    false_positive = int((predicted_positive & negative).sum())
    true_negative = int(((~predicted_positive) & negative).sum())
    false_negative = int(((~predicted_positive) & positive).sum())
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    specificity = true_negative / max(1, true_negative + false_positive)
    return {
        "threshold": float(threshold),
        "accuracy": float((predictions == labels).mean()),
        "balanced_accuracy": 0.5 * (recall + specificity),
        "attack_precision": precision,
        "attack_recall": recall,
        "hold_specificity": specificity,
        "false_positive_rate": 1.0 - specificity,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
    }


def select_conservative_threshold(
    probabilities: np.ndarray,
    labels: np.ndarray,
    minimum_precision: float = 0.97,
    minimum_recall: float = 0.75,
) -> Tuple[float, Dict[str, float], bool]:
    """Choose the lowest threshold meeting the predeclared safety targets."""

    candidates = np.linspace(0.50, 0.995, 100)
    rows = [binary_metrics(probabilities, labels, value) for value in candidates]
    eligible = [
        row
        for row in rows
        if row["attack_precision"] >= minimum_precision
        and row["attack_recall"] >= minimum_recall
    ]
    if eligible:
        chosen = eligible[0]
        return float(chosen["threshold"]), chosen, True
    # Diagnostic fallback only; the formal gate remains false.
    chosen = max(
        rows,
        key=lambda row: (
            row["attack_precision"],
            row["attack_recall"],
            row["balanced_accuracy"],
        ),
    )
    return float(chosen["threshold"]), chosen, False


def _assert_seed_isolation(train, validation) -> None:
    train_seeds = set(train["episode_seed"].tolist())
    validation_seeds = set(validation["episode_seed"].tolist())
    overlap = sorted(train_seeds & validation_seeds)
    if overlap:
        raise ValueError("train/validation seed overlap: {}".format(overlap))
    low, high = BANNED_SEED_RANGE
    burned = sorted(
        seed for seed in train_seeds | validation_seeds if low <= seed <= high
    )
    if burned:
        raise ValueError("banned teacher development/gate seeds: {}".format(burned))


def main():
    args = parse_args()
    if args.hard_negative_repeat < 1:
        raise ValueError("hard-negative-repeat must be at least one")
    outputs = [
        Path(args.checkpoint),
        Path(args.training_log),
        Path(args.summary),
        Path(args.config),
    ]
    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "refusing to overwrite attack-gate outputs: {}".format(
                ", ".join(str(path) for path in existing)
            )
        )

    np.random.seed(args.seed)
    train_path = Path(args.train_dataset)
    validation_path = Path(args.validation_dataset)
    train_raw = load_attack_gate_split(train_path)
    validation_raw = load_attack_gate_split(validation_path)
    _assert_seed_isolation(train_raw, validation_raw)
    train, train_report = select_attack_gate_samples(train_raw)
    validation, validation_report = select_attack_gate_samples(validation_raw)

    coverage_gate = {
        "train_samples_at_least_2000": len(train["label"]) >= 2000,
        "validation_samples_at_least_300": len(validation["label"]) >= 300,
        "train_attack_samples_at_least_500": int((train["label"] == ATTACK).sum()) >= 500,
        "validation_attack_samples_at_least_100": int((validation["label"] == ATTACK).sum()) >= 100,
        "validation_hold_samples_at_least_100": int((validation["label"] == HOLD).sum()) >= 100,
        "validation_success_seeds_at_least_10": len(set(validation["episode_seed"].tolist())) >= 10,
    }
    coverage_gate["all_conditions_met"] = all(coverage_gate.values())

    training_indices, hard_negative_report = hard_negative_training_indices(
        train, args.hard_negative_repeat
    )
    train_pov = train["pov"][training_indices]
    train_labels = train["label"][training_indices]
    train_previous = train["previous_action"][training_indices]
    if not args.no_mirror_augmentation:
        train_pov = np.concatenate((train_pov, mirror_pov_frames(train_pov)))
        train_labels = np.concatenate((train_labels, train_labels.copy()))
        train_previous = np.concatenate(
            (train_previous, mirror_actions(train_previous))
        )

    policy = NaturalAttackGatePolicy(
        feature_size=args.feature_size,
        frame_stack=4,
        include_centre_pixels=False,
    )
    started = time.perf_counter()
    history = policy.fit(
        train_pov,
        train_labels,
        train_previous,
        validation["pov"],
        validation["label"],
        validation["previous_action"],
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
        patience=args.patience,
    )
    training_seconds = round(time.perf_counter() - started, 3)
    validation_features = policy.build_features(
        validation["pov"], validation["previous_action"]
    )
    probabilities = policy.predict_proba_from_features(validation_features)
    attack_index = int(np.flatnonzero(policy.classes == ATTACK)[0])
    attack_probabilities = probabilities[:, attack_index]
    threshold, metrics, threshold_gate = select_conservative_threshold(
        attack_probabilities, validation["label"]
    )
    policy.decision_threshold = threshold

    initial_loss = history[0]["validation_loss"]
    relative_loss_improvement = (
        (initial_loss - policy.best_validation_loss) / max(initial_loss, 1e-8)
    )
    dataset_hashes = {
        "train": file_sha256(train_path),
        "validation_development": file_sha256(validation_path),
    }
    seed_ranges = {
        "train": sorted(set(train["episode_seed"].tolist())),
        "validation_development": sorted(
            set(validation["episode_seed"].tolist())
        ),
    }
    policy.save(args.checkpoint, dataset_hashes, seed_ranges)
    reloaded = NaturalAttackGatePolicy.load(args.checkpoint)
    reload_features = reloaded.build_features(
        validation["pov"], validation["previous_action"]
    )
    reload_probabilities = reloaded.predict_proba_from_features(reload_features)
    reload_attack = reload_probabilities[:, int(np.flatnonzero(reloaded.classes == ATTACK)[0])]
    reload_consistent = bool(
        reloaded.decision_threshold == policy.decision_threshold
        and np.array_equal(
            reload_attack >= reloaded.decision_threshold,
            attack_probabilities >= policy.decision_threshold,
        )
    )

    offline_gate = {
        "coverage_gate_passed": bool(coverage_gate["all_conditions_met"]),
        "threshold_meets_precision_and_recall_targets": threshold_gate,
        "validation_loss_improvement_at_least_20_percent": relative_loss_improvement >= 0.20,
        "balanced_accuracy_at_least_80_percent": metrics["balanced_accuracy"] >= 0.80,
        "attack_precision_at_least_97_percent": metrics["attack_precision"] >= 0.97,
        "attack_recall_at_least_75_percent": metrics["attack_recall"] >= 0.75,
        "false_positive_rate_at_most_2_percent": metrics["false_positive_rate"] <= 0.02,
        "finite_weights": bool(np.all(np.isfinite(policy.weights))),
        "checkpoint_reload_consistent": reload_consistent,
        "privileged_model_inputs_zero": True,
    }
    offline_gate["all_conditions_met"] = all(offline_gate.values())

    summary = {
        "model_version": MODEL_VERSION,
        "status": (
            "offline_gate_passed"
            if offline_gate["all_conditions_met"]
            else "experimental_offline_gate_failed"
        ),
        "teacher_profile": "terrain_route_drop_completion_v9_6",
        "runtime_semantics": {
            "gate_states": sorted(GATE_CONTACT_STATES),
            "hold": "prevent attack state mutation; after three consecutive rejected attack opportunities recenter visually",
            "attack": "permit the frozen contact controller to start or continue attack",
            "non_attack_teacher_actions": "unchanged",
            "recovery_scan_and_drop_states": "never gated",
        },
        "student_input_manifest": [
            "pov_frame_stack_4",
            "previous_action_one_hot_14",
        ],
        "dataset": {
            "train": str(train_path),
            "validation_development": str(validation_path),
            "hashes": dataset_hashes,
            "train_selection": train_report,
            "validation_selection": validation_report,
            "validation_note": "17000-17019 is a reused development set; 17100-17119 remains the first untouched shadow range",
        },
        "coverage_gate": coverage_gate,
        "training": {
            "epochs_requested": args.epochs,
            "selected_epoch": policy.best_epoch,
            "stopped_early": bool(policy.stopped_early),
            "learning_rate": args.learning_rate,
            "l2": args.l2,
            "patience": args.patience,
            "mirror_augmentation": not args.no_mirror_augmentation,
            "hard_negative_weighting": hard_negative_report,
            "training_seconds": training_seconds,
        },
        "decision_threshold": threshold,
        "hard_negative_repeat": args.hard_negative_repeat,
        "validation_metrics": metrics,
        "validation_loss": {
            "initial": initial_loss,
            "best": policy.best_validation_loss,
            "relative_improvement": relative_loss_improvement,
        },
        "offline_gate": offline_gate,
    }
    atomic_write_rows(Path(args.training_log), history)
    atomic_write_json(Path(args.summary), summary)
    config = {
        "profile": MODEL_VERSION,
        "status": summary["status"],
        "checkpoint": args.checkpoint,
        "training_summary": args.summary,
        "runtime_semantics": summary["runtime_semantics"],
        "student_input_manifest": summary["student_input_manifest"],
        "student_forbidden_inputs": [
            "raycast",
            "telemetry",
            "exact_log_xyz",
            "log_grid",
            "target_coordinates",
            "teacher_contact_state_as_model_feature",
        ],
        "decision_threshold": threshold,
        "gate_status": {
            "teacher_gate_passed": True,
            "offline_gate_passed": bool(offline_gate["all_conditions_met"]),
            "shadow_gate_passed": False,
            "autonomous_smoke_passed": False,
            "student_holdout_passed": False,
            "next_stage_rl_authorized": False,
        },
    }
    atomic_write_json(Path(args.config), config)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
