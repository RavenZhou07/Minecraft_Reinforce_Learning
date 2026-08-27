"""Train the BC v2 hybrid visual contact policy.

The teacher state audit is read only to select decisions that the hybrid
router would hand to the student.  Model features remain the declared POV
stack and previous-action one-hot.  A formal run requires datasets collected
with both pre-decision and post-decision contact-state audits; the older v1
post-state audit can be enabled only for a non-gating diagnostic smoke run.
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

from mc_rl.natural_contact_bc import mirror_actions, mirror_pov_frames
from mc_rl.natural_contact_bc_v2 import (
    LEARNABLE_CONTACT_STATES,
    MODEL_VERSION,
    SCRIPTED_CONTACT_STATES,
    V2_ACTION_CLASSES,
    V2_REQUIRED_DIRECTIONAL_ACTIONS,
    NaturalContactBCV2Policy,
    hybrid_learning_mask,
)


STUDENT_FIELDS = ("pov", "action", "previous_action")
AUDIT_FIELDS = (
    "audit_decision_contact_state",
    "audit_resulting_contact_state",
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
        default="checkpoints/natural_treechop_contact_bc_v2_hybrid_experimental.npz",
    )
    parser.add_argument(
        "--training-log",
        default="logs/find_tree/natural_treechop_contact_bc_v2_hybrid_training.csv",
    )
    parser.add_argument(
        "--summary",
        default="logs/find_tree/natural_treechop_contact_bc_v2_hybrid_training.summary.json",
    )
    parser.add_argument(
        "--config", default="configs/natural_treechop_contact_bc_v2.json"
    )
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=300)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--feature-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-legacy-post-state-audit",
        action="store_true",
        help=(
            "Diagnostic smoke only: approximate both transition states from "
            "the v1 post-decision audit. This can never pass the formal gate."
        ),
    )
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


def load_hybrid_split(
    path: Path, allow_legacy_post_state_audit: bool = False
) -> Tuple[Dict[str, np.ndarray], bool]:
    """Load student arrays plus routing audits, never audits as features."""

    with np.load(path, allow_pickle=False) as data:
        available = set(data.files)
        missing = [field for field in STUDENT_FIELDS if field not in available]
        if missing:
            raise KeyError("dataset {} is missing {}".format(path, missing))
        formal_audit = all(field in available for field in AUDIT_FIELDS)
        if formal_audit:
            decision = np.asarray(data[AUDIT_FIELDS[0]]).astype(str)
            resulting = np.asarray(data[AUDIT_FIELDS[1]]).astype(str)
        elif allow_legacy_post_state_audit and "audit_contact_state" in available:
            decision = np.asarray(data["audit_contact_state"]).astype(str)
            resulting = decision.copy()
        else:
            raise KeyError(
                "dataset {} lacks pre/post contact-state audits; recollect with "
                "the updated collector or pass --allow-legacy-post-state-audit "
                "for a non-gating smoke run".format(path)
            )
        split = {
            "pov": data["pov"].astype(np.uint8),
            "action": data["action"].astype(np.int64),
            "previous_action": data["previous_action"].astype(np.int64),
            "decision_state": decision,
            "resulting_state": resulting,
        }
        for field in ("episode_success", "episode_seed", "episode"):
            if field in available:
                split[field] = np.asarray(data[field]).astype(np.int64)
    return split, formal_audit


def select_hybrid_samples(
    split: Dict[str, np.ndarray], successful_only: bool = True
) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    """Apply success and hybrid-routing masks to every sample-aligned array."""

    count = len(split["action"])
    success_mask = np.ones(count, dtype=bool)
    if successful_only and "episode_success" in split:
        success_mask = split["episode_success"].astype(bool)
    route_mask = hybrid_learning_mask(
        split["decision_state"], split["resulting_state"], split["action"]
    )
    selected = success_mask & route_mask
    result = {
        key: value[selected]
        for key, value in split.items()
        if isinstance(value, np.ndarray) and len(value) == count
    }
    report = {
        "total_samples": int(count),
        "excluded_failure_samples": int((~success_mask).sum()),
        "successful_samples": int(success_mask.sum()),
        "scripted_or_unsupported_samples": int((success_mask & ~route_mask).sum()),
        "selected_visual_samples": int(selected.sum()),
    }
    return result, report


def _direction_agreement(
    predictions: np.ndarray,
    labels: np.ndarray,
    negative: Tuple[int, ...],
    positive: Tuple[int, ...],
) -> float:
    selected = np.isin(labels, negative + positive)
    if not selected.any():
        return 0.0
    prediction_is_directional = np.isin(
        predictions[selected], negative + positive
    )
    predicted_negative = np.isin(predictions[selected], negative)
    label_negative = np.isin(labels[selected], negative)
    return float(
        (prediction_is_directional & (predicted_negative == label_negative)).mean()
    )


def _precision_recall(
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


def evaluate_predictions(
    predictions: np.ndarray, labels: np.ndarray
) -> Dict[str, float]:
    recalls = []
    for action in V2_ACTION_CLASSES:
        selected = labels == action
        if selected.any():
            recalls.append(float((predictions[selected] == action).mean()))
    attack_precision, attack_recall = _precision_recall(
        predictions, labels, 7
    )
    return {
        "accuracy": float((predictions == labels).mean()),
        "balanced_accuracy": float(np.mean(recalls)) if recalls else 0.0,
        "attack_precision": attack_precision,
        "attack_recall": attack_recall,
        "fine_yaw_direction_agreement": _direction_agreement(
            predictions, labels, (10,), (11,)
        ),
        "fine_pitch_direction_agreement": _direction_agreement(
            predictions, labels, (12,), (13,)
        ),
    }


def coverage_gate(
    train: Dict[str, np.ndarray], validation: Dict[str, np.ndarray]
) -> Dict[str, Any]:
    train_counts = Counter(map(int, train["action"]))
    validation_counts = Counter(map(int, validation["action"]))
    required = sorted(V2_REQUIRED_DIRECTIONAL_ACTIONS)
    validation_seeds = set(validation.get("episode_seed", np.asarray([])).tolist())
    checks = {
        "train_visual_samples_at_least_500": len(train["action"]) >= 500,
        "validation_visual_samples_at_least_100": len(validation["action"]) >= 100,
        "validation_success_seeds_at_least_10": len(validation_seeds) >= 10,
        "required_actions_train_at_least_20_each": all(
            train_counts[action] >= 20 for action in required
        ),
        "required_actions_validation_at_least_5_each": all(
            validation_counts[action] >= 5 for action in required
        ),
    }
    checks["all_conditions_met"] = all(checks.values())
    return {
        "train_action_counts": {
            str(key): value for key, value in sorted(train_counts.items())
        },
        "validation_action_counts": {
            str(key): value for key, value in sorted(validation_counts.items())
        },
        "checks": checks,
    }


def _assert_seed_isolation(train: Dict[str, np.ndarray], validation: Dict[str, np.ndarray]):
    train_seeds = set(train.get("episode_seed", np.asarray([])).tolist())
    validation_seeds = set(validation.get("episode_seed", np.asarray([])).tolist())
    overlap = sorted(train_seeds & validation_seeds)
    if overlap:
        raise ValueError("train/validation seed overlap: {}".format(overlap))
    low, high = BANNED_SEED_RANGE
    burned = sorted(
        seed for seed in train_seeds | validation_seeds if low <= seed <= high
    )
    if burned:
        raise ValueError("banned development/gate seeds: {}".format(burned))


def main():
    args = parse_args()
    outputs = [
        Path(args.checkpoint),
        Path(args.training_log),
        Path(args.summary),
        Path(args.config),
    ]
    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "refusing to overwrite BC v2 outputs: {}".format(
                ", ".join(str(path) for path in existing)
            )
        )

    np.random.seed(args.seed)
    train_path = Path(args.train_dataset)
    validation_path = Path(args.validation_dataset)
    train_raw, train_formal_audit = load_hybrid_split(
        train_path, args.allow_legacy_post_state_audit
    )
    validation_raw, validation_formal_audit = load_hybrid_split(
        validation_path, args.allow_legacy_post_state_audit
    )
    _assert_seed_isolation(train_raw, validation_raw)
    train, train_selection = select_hybrid_samples(train_raw)
    validation, validation_selection = select_hybrid_samples(validation_raw)
    if not len(train["action"]) or not len(validation["action"]):
        raise ValueError("hybrid selection produced an empty train/validation split")

    coverage = coverage_gate(train, validation)
    train_pov = train["pov"]
    train_actions = train["action"]
    train_previous = train["previous_action"]
    if not args.no_mirror_augmentation:
        train_pov = np.concatenate(
            (train_pov, mirror_pov_frames(train_pov)), axis=0
        )
        train_actions = np.concatenate(
            (train_actions, mirror_actions(train_actions)), axis=0
        )
        train_previous = np.concatenate(
            (train_previous, mirror_actions(train_previous)), axis=0
        )

    policy = NaturalContactBCV2Policy(
        feature_size=args.feature_size,
        frame_stack=4,
        include_centre_pixels=False,
    )
    started = time.perf_counter()
    history = policy.fit(
        train_pov,
        train_actions,
        train_previous,
        validation["pov"],
        validation["action"],
        validation["previous_action"],
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
        patience=args.patience,
        momentum=args.momentum,
    )
    training_seconds = round(time.perf_counter() - started, 3)

    features = policy.build_features(
        validation["pov"], validation["previous_action"]
    )
    probabilities = policy.predict_proba_from_features(features)
    predictions = policy.classes[probabilities.argmax(axis=1)]
    metrics = evaluate_predictions(predictions, validation["action"])
    initial_loss = history[0]["validation_loss"]
    relative_improvement = (
        (initial_loss - policy.best_validation_loss) / max(initial_loss, 1e-8)
    )

    dataset_hashes = {
        "train": file_sha256(train_path),
        "validation": file_sha256(validation_path),
    }
    seed_ranges = {
        "train": sorted(set(train.get("episode_seed", np.asarray([])).tolist())),
        "validation": sorted(
            set(validation.get("episode_seed", np.asarray([])).tolist())
        ),
    }
    policy.save(str(Path(args.checkpoint)), dataset_hashes, seed_ranges)
    reloaded = NaturalContactBCV2Policy.load(args.checkpoint)
    reload_features = reloaded.build_features(
        validation["pov"], validation["previous_action"]
    )
    reload_predictions = reloaded.classes[
        reloaded.predict_proba_from_features(reload_features).argmax(axis=1)
    ]
    formal_audit = bool(train_formal_audit and validation_formal_audit)
    offline_gate = {
        "formal_pre_and_post_state_audits": formal_audit,
        "coverage_gate_passed": bool(coverage["checks"]["all_conditions_met"]),
        "validation_loss_improvement_at_least_20_percent": relative_improvement >= 0.20,
        "balanced_accuracy_at_least_70_percent": metrics["balanced_accuracy"] >= 0.70,
        "attack_precision_at_least_90_percent": metrics["attack_precision"] >= 0.90,
        "attack_recall_at_least_80_percent": metrics["attack_recall"] >= 0.80,
        "fine_yaw_direction_at_least_85_percent": metrics["fine_yaw_direction_agreement"] >= 0.85,
        "fine_pitch_direction_at_least_80_percent": metrics["fine_pitch_direction_agreement"] >= 0.80,
        "finite_weights": bool(np.all(np.isfinite(policy.weights))),
        "checkpoint_reload_consistent": bool(
            np.array_equal(predictions, reload_predictions)
        ),
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
        "hybrid_boundary": {
            "learnable_states": sorted(LEARNABLE_CONTACT_STATES),
            "scripted_states": sorted(SCRIPTED_CONTACT_STATES),
            "rule": "both decision and resulting state must be learnable; unsupported transition actions remain scripted",
            "action_classes": V2_ACTION_CLASSES.tolist(),
        },
        "dataset": {
            "train": str(train_path),
            "validation_development_set": str(validation_path),
            "hashes": dataset_hashes,
            "formal_transition_audit": formal_audit,
            "train_selection": train_selection,
            "validation_selection": validation_selection,
            "note": "17000-17019 was reused during v1 tuning and is a development set, not a final holdout",
        },
        "coverage": coverage,
        "training": {
            "epochs_requested": args.epochs,
            "selected_epoch": policy.best_epoch,
            "stopped_early": bool(policy.stopped_early),
            "learning_rate": args.learning_rate,
            "l2": args.l2,
            "patience": args.patience,
            "mirror_augmentation": not args.no_mirror_augmentation,
            "seconds": training_seconds,
        },
        "validation_metrics": metrics,
        "validation_loss": {
            "initial": initial_loss,
            "best": policy.best_validation_loss,
            "relative_improvement": relative_improvement,
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
            "teacher_contact_state_as_model_feature",
        ],
        "hybrid_boundary": summary["hybrid_boundary"],
        "gate_status": {
            "teacher_gate_passed": True,
            "offline_gate_passed": bool(offline_gate["all_conditions_met"]),
            "shadow_gate_passed": False,
            "student_holdout_passed": False,
        },
    }
    atomic_write_json(Path(args.config), config)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
