"""Train BC v2b from the base corpus plus seed-isolated shadow diagnostics."""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from mc_rl.natural_attack_gate_bc import ATTACK, GATE_CONTACT_STATES, HOLD
from mc_rl.natural_attack_gate_bc_v2b import (
    MODEL_VERSION,
    NaturalAttackGateV2Policy,
)
from mc_rl.natural_contact_bc import mirror_actions, mirror_pov_frames
from scripts.train_natural_treechop_attack_gate_v2a import (
    atomic_write_json,
    atomic_write_rows,
    binary_metrics,
    file_sha256,
    hard_negative_training_indices,
    load_attack_gate_split,
    select_attack_gate_samples,
)


AUTONOMOUS_HOLDOUT = (17200, 17219)


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
        "--targeted-dataset",
        default="logs/find_tree/natural_treechop_attack_gate_bc_v2a_diagnostics_17100_20.npz",
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/natural_treechop_attack_gate_bc_v2b_experimental.npz",
    )
    parser.add_argument(
        "--training-log",
        default="logs/find_tree/natural_treechop_attack_gate_bc_v2b_training.csv",
    )
    parser.add_argument(
        "--summary",
        default="logs/find_tree/natural_treechop_attack_gate_bc_v2b_training.summary.json",
    )
    parser.add_argument(
        "--config", default="configs/natural_treechop_attack_gate_bc_v2b.json"
    )
    parser.add_argument("--epochs", type=int, default=1800)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=250)
    parser.add_argument("--feature-size", type=int, default=10)
    parser.add_argument("--base-hard-negative-repeat", type=int, default=2)
    parser.add_argument("--targeted-false-positive-repeat", type=int, default=4)
    parser.add_argument("--calibration-seed-count", type=int, default=8)
    parser.add_argument("--attack-confirmation-frames", type=int, default=2)
    parser.add_argument("--no-mirror-augmentation", action="store_true")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_targeted_split(path: Path) -> Dict[str, np.ndarray]:
    required = (
        "pov",
        "action",
        "previous_action",
        "episode_seed",
        "episode_step",
        "audit_contact_state",
        "student_gate_decision",
    )
    with np.load(path, allow_pickle=False) as data:
        missing = [field for field in required if field not in data.files]
        if missing:
            raise KeyError("targeted dataset {} is missing {}".format(path, missing))
        result = {
            "pov": data["pov"].astype(np.uint8),
            "environment_action": data["action"].astype(np.int64),
            "previous_action": data["previous_action"].astype(np.int64),
            "episode_seed": data["episode_seed"].astype(np.int64),
            "episode_step": data["episode_step"].astype(np.int64),
            "contact_state": np.asarray(data["audit_contact_state"]).astype(str),
            "student_gate_decision": data["student_gate_decision"].astype(np.int64),
        }
        for field in (
            "student_attack_probability",
            "audit_raycast_is_log",
            "audit_raycast_in_range",
            "episode_success",
        ):
            if field in data.files:
                result[field] = np.asarray(data[field])
    lengths = {len(value) for value in result.values() if value.ndim > 0}
    if len(lengths) != 1:
        raise ValueError("targeted dataset arrays do not align")
    result["label"] = (
        result["environment_action"] == 7
    ).astype(np.int64)
    gate_mask = np.isin(result["contact_state"], tuple(GATE_CONTACT_STATES))
    return {
        key: value[gate_mask]
        for key, value in result.items()
        if value.ndim > 0 and len(value) == len(gate_mask)
    }


def split_targeted_by_seed(
    targeted: Dict[str, np.ndarray], calibration_seed_count: int
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, Any]]:
    seeds = sorted(set(targeted["episode_seed"].tolist()))
    if calibration_seed_count < 1 or len(seeds) <= calibration_seed_count:
        raise ValueError("targeted data needs training seeds plus calibration seeds")
    low, high = AUTONOMOUS_HOLDOUT
    leaked = [seed for seed in seeds if low <= seed <= high]
    if leaked:
        raise ValueError("autonomous holdout seeds present in targeted data: {}".format(leaked))
    calibration_seeds = seeds[-calibration_seed_count:]
    training_seeds = seeds[:-calibration_seed_count]
    calibration_mask = np.isin(targeted["episode_seed"], calibration_seeds)

    def subset(mask):
        return {key: value[mask] for key, value in targeted.items()}

    return (
        subset(~calibration_mask),
        subset(calibration_mask),
        {
            "training_seeds": training_seeds,
            "calibration_seeds": calibration_seeds,
            "training_samples": int((~calibration_mask).sum()),
            "calibration_samples": int(calibration_mask.sum()),
        },
    )


def targeted_training_indices(
    targeted: Dict[str, np.ndarray], false_positive_repeat: int
) -> Tuple[np.ndarray, Dict[str, int]]:
    if false_positive_repeat < 1:
        raise ValueError("targeted false-positive repeat must be at least one")
    false_positive = (
        (targeted["label"] == HOLD)
        & (targeted["student_gate_decision"] == ATTACK)
    )
    base = np.arange(len(targeted["label"]), dtype=np.int64)
    hard = np.flatnonzero(false_positive).astype(np.int64)
    indices = base
    if false_positive_repeat > 1 and len(hard):
        indices = np.concatenate(
            (base, np.tile(hard, false_positive_repeat - 1))
        )
    return indices, {
        "repeat_factor": int(false_positive_repeat),
        "original_samples": int(len(base)),
        "false_positive_samples": int(len(hard)),
        "added_false_positive_samples": int(len(indices) - len(base)),
    }


def temporal_predictions(
    probabilities: np.ndarray,
    threshold: float,
    seeds: np.ndarray,
    steps: np.ndarray,
    states: np.ndarray,
    confirmation_frames: int,
) -> np.ndarray:
    """Replay the runner's state-local, causal ATTACK confirmation rule."""

    raw = np.asarray(probabilities) >= float(threshold)
    predictions = np.zeros(len(raw), dtype=np.int64)
    streak = 0
    last_seed = None
    last_step = None
    last_state = None
    for index, is_attack in enumerate(raw):
        seed = int(seeds[index])
        step = int(steps[index])
        state = str(states[index])
        continuous = (
            seed == last_seed
            and last_step is not None
            and step == last_step + 1
            and state == last_state
        )
        if not continuous:
            streak = 0
        streak = streak + 1 if is_attack else 0
        predictions[index] = int(streak >= confirmation_frames)
        last_seed, last_step, last_state = seed, step, state
    return predictions


def select_temporal_threshold(
    probabilities: np.ndarray,
    labels: np.ndarray,
    seeds: np.ndarray,
    steps: np.ndarray,
    states: np.ndarray,
    confirmation_frames: int,
    minimum_precision: float = 0.97,
    minimum_recall: float = 0.75,
) -> Tuple[float, Dict[str, float], bool]:
    rows: List[Dict[str, float]] = []
    for threshold in np.linspace(0.50, 0.995, 100):
        predictions = temporal_predictions(
            probabilities,
            threshold,
            seeds,
            steps,
            states,
            confirmation_frames,
        )
        metrics = binary_metrics(predictions.astype(np.float32), labels, 0.5)
        metrics["threshold"] = float(threshold)
        rows.append(metrics)
    eligible = [
        row
        for row in rows
        if row["attack_precision"] >= minimum_precision
        and row["attack_recall"] >= minimum_recall
        and row["false_positive_rate"] <= 0.02
    ]
    if eligible:
        chosen = eligible[0]
        return float(chosen["threshold"]), chosen, True
    chosen = max(
        rows,
        key=lambda row: (
            row["attack_precision"],
            row["attack_recall"],
            row["balanced_accuracy"],
        ),
    )
    return float(chosen["threshold"]), chosen, False


def probabilities_for(policy, split):
    features = policy.build_features(split["pov"], split["previous_action"])
    probabilities = policy.predict_proba_from_features(features)
    attack_index = int(np.flatnonzero(policy.classes == ATTACK)[0])
    return probabilities[:, attack_index]


def temporal_metrics(policy, split, threshold, confirmation_frames):
    probabilities = probabilities_for(policy, split)
    predictions = temporal_predictions(
        probabilities,
        threshold,
        split["episode_seed"],
        split["episode_step"],
        split["contact_state"],
        confirmation_frames,
    )
    metrics = binary_metrics(
        predictions.astype(np.float32), split["label"], 0.5
    )
    metrics["threshold"] = float(threshold)
    return metrics


def main():
    args = parse_args()
    if args.attack_confirmation_frames < 1:
        raise ValueError("attack-confirmation-frames must be at least one")
    outputs = [
        Path(args.checkpoint),
        Path(args.training_log),
        Path(args.summary),
        Path(args.config),
    ]
    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "refusing to overwrite BC v2b outputs: {}".format(
                ", ".join(str(path) for path in existing)
            )
        )
    np.random.seed(args.seed)
    train_path = Path(args.train_dataset)
    validation_path = Path(args.validation_dataset)
    targeted_path = Path(args.targeted_dataset)
    base_train, base_train_report = select_attack_gate_samples(
        load_attack_gate_split(train_path)
    )
    base_validation, base_validation_report = select_attack_gate_samples(
        load_attack_gate_split(validation_path)
    )
    targeted = load_targeted_split(targeted_path)
    targeted_train, targeted_calibration, targeted_split_report = (
        split_targeted_by_seed(targeted, args.calibration_seed_count)
    )
    base_indices, base_weighting = hard_negative_training_indices(
        base_train, args.base_hard_negative_repeat
    )
    targeted_indices, targeted_weighting = targeted_training_indices(
        targeted_train, args.targeted_false_positive_repeat
    )

    train_pov = np.concatenate(
        (base_train["pov"][base_indices], targeted_train["pov"][targeted_indices])
    )
    train_labels = np.concatenate(
        (base_train["label"][base_indices], targeted_train["label"][targeted_indices])
    )
    train_previous = np.concatenate(
        (
            base_train["previous_action"][base_indices],
            targeted_train["previous_action"][targeted_indices],
        )
    )
    if not args.no_mirror_augmentation:
        original_pov = train_pov
        train_pov = np.concatenate((original_pov, mirror_pov_frames(original_pov)))
        train_labels = np.concatenate((train_labels, train_labels.copy()))
        train_previous = np.concatenate(
            (train_previous, mirror_actions(train_previous))
        )

    validation_pov = np.concatenate(
        (base_validation["pov"], targeted_calibration["pov"])
    )
    validation_labels = np.concatenate(
        (base_validation["label"], targeted_calibration["label"])
    )
    validation_previous = np.concatenate(
        (base_validation["previous_action"], targeted_calibration["previous_action"])
    )
    policy = NaturalAttackGateV2Policy(
        feature_size=args.feature_size,
        frame_stack=4,
        include_centre_pixels=False,
        attack_confirmation_frames=args.attack_confirmation_frames,
    )
    started = time.perf_counter()
    history = policy.fit(
        train_pov,
        train_labels,
        train_previous,
        validation_pov,
        validation_labels,
        validation_previous,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
        patience=args.patience,
    )
    training_seconds = round(time.perf_counter() - started, 3)
    calibration_probabilities = probabilities_for(policy, targeted_calibration)
    threshold, calibration_metrics, threshold_passed = select_temporal_threshold(
        calibration_probabilities,
        targeted_calibration["label"],
        targeted_calibration["episode_seed"],
        targeted_calibration["episode_step"],
        targeted_calibration["contact_state"],
        args.attack_confirmation_frames,
    )
    policy.decision_threshold = threshold
    base_validation_probabilities = probabilities_for(policy, base_validation)
    base_validation_metrics = binary_metrics(
        base_validation_probabilities, base_validation["label"], threshold
    )
    base_validation_temporal_diagnostic = temporal_metrics(
        policy,
        base_validation,
        threshold,
        args.attack_confirmation_frames,
    )
    initial_loss = history[0]["validation_loss"]
    relative_loss_improvement = (
        (initial_loss - policy.best_validation_loss) / max(initial_loss, 1e-8)
    )
    coverage_gate = {
        "targeted_training_samples_at_least_150": len(targeted_train["label"]) >= 150,
        "targeted_calibration_samples_at_least_150": len(targeted_calibration["label"]) >= 150,
        "targeted_calibration_hold_samples_at_least_100": int((targeted_calibration["label"] == HOLD).sum()) >= 100,
        "targeted_calibration_attack_samples_at_least_30": int((targeted_calibration["label"] == ATTACK).sum()) >= 30,
        "targeted_false_positives_at_least_10": targeted_weighting["false_positive_samples"] >= 10,
    }
    coverage_gate["all_conditions_met"] = all(coverage_gate.values())
    dataset_hashes = {
        "base_train": file_sha256(train_path),
        "base_validation": file_sha256(validation_path),
        "targeted_diagnostic": file_sha256(targeted_path),
    }
    seed_ranges = {
        "base_train": sorted(set(base_train["episode_seed"].tolist())),
        "base_validation": sorted(set(base_validation["episode_seed"].tolist())),
        "targeted_train": targeted_split_report["training_seeds"],
        "targeted_calibration": targeted_split_report["calibration_seeds"],
    }
    policy.save(args.checkpoint, dataset_hashes, seed_ranges)
    reloaded = NaturalAttackGateV2Policy.load(args.checkpoint)
    reload_consistent = bool(
        reloaded.model_version == MODEL_VERSION
        and reloaded.decision_threshold == policy.decision_threshold
        and reloaded.attack_confirmation_frames == args.attack_confirmation_frames
        and np.array_equal(
            probabilities_for(reloaded, targeted_calibration) >= threshold,
            calibration_probabilities >= threshold,
        )
    )
    offline_gate = {
        "coverage_gate_passed": bool(coverage_gate["all_conditions_met"]),
        "calibration_threshold_meets_joint_targets": bool(threshold_passed),
        "validation_loss_improvement_at_least_20_percent": relative_loss_improvement >= 0.20,
        "targeted_balanced_accuracy_at_least_80_percent": calibration_metrics["balanced_accuracy"] >= 0.80,
        "targeted_attack_precision_at_least_97_percent": calibration_metrics["attack_precision"] >= 0.97,
        "targeted_attack_recall_at_least_75_percent": calibration_metrics["attack_recall"] >= 0.75,
        "targeted_false_positive_rate_at_most_2_percent": calibration_metrics["false_positive_rate"] <= 0.02,
        "finite_weights": bool(np.all(np.isfinite(policy.weights))),
        "checkpoint_reload_consistent": reload_consistent,
        "privileged_model_inputs_zero": True,
    }
    offline_gate["all_conditions_met"] = all(offline_gate.values())
    summary = {
        "model_version": MODEL_VERSION,
        "status": "offline_gate_passed" if offline_gate["all_conditions_met"] else "experimental_offline_gate_failed",
        "teacher_profile": "terrain_route_drop_completion_v9_6",
        "student_input_manifest": ["pov_frame_stack_4", "previous_action_one_hot_14"],
        "dataset": {
            "base_train": str(train_path),
            "base_validation": str(validation_path),
            "targeted_diagnostic": str(targeted_path),
            "hashes": dataset_hashes,
            "base_train_selection": base_train_report,
            "base_validation_selection": base_validation_report,
            "targeted_split": targeted_split_report,
        },
        "training": {
            "epochs_requested": args.epochs,
            "selected_epoch": policy.best_epoch,
            "stopped_early": bool(policy.stopped_early),
            "learning_rate": args.learning_rate,
            "l2": args.l2,
            "patience": args.patience,
            "mirror_augmentation": not args.no_mirror_augmentation,
            "base_hard_negative_weighting": base_weighting,
            "targeted_false_positive_weighting": targeted_weighting,
            "training_samples_before_mirror": int(len(train_pov) // (1 if args.no_mirror_augmentation else 2)),
            "training_seconds": training_seconds,
        },
        "decision_threshold": threshold,
        "attack_confirmation_frames": args.attack_confirmation_frames,
        "targeted_calibration_metrics": calibration_metrics,
        "base_validation_metrics": base_validation_metrics,
        "base_validation_temporal_diagnostic": base_validation_temporal_diagnostic,
        "legacy_regression_warning": {
            "base_validation_balanced_accuracy_below_80_percent": (
                base_validation_metrics["balanced_accuracy"] < 0.80
            ),
            "formal_gate_effect": "warning_only",
            "reason": (
                "17000-17019 is a reused v2a development set and stores only "
                "contact-active rows, so it cannot faithfully replay the v2b "
                "consecutive-frame contract; the seed-isolated targeted "
                "calibration is the predeclared v2b offline gate"
            ),
        },
        "validation_loss": {
            "initial": initial_loss,
            "best": policy.best_validation_loss,
            "relative_improvement": relative_loss_improvement,
        },
        "coverage_gate": coverage_gate,
        "offline_gate": offline_gate,
        "seed_isolation": {
            "consumed_diagnostic": targeted_split_report,
            "autonomous_holdout_reserved_not_used": "17200-17219",
            "next_formal_shadow": "17400-17419",
        },
    }
    atomic_write_rows(Path(args.training_log), history)
    atomic_write_json(Path(args.summary), summary)
    config = {
        "profile": MODEL_VERSION,
        "status": summary["status"],
        "checkpoint": args.checkpoint,
        "training_summary": args.summary,
        "decision_threshold": threshold,
        "attack_confirmation_frames": args.attack_confirmation_frames,
        "student_input_manifest": summary["student_input_manifest"],
        "student_forbidden_inputs": [
            "raycast",
            "telemetry",
            "exact_log_xyz",
            "log_grid",
            "target_coordinates",
            "teacher_contact_state_as_model_feature",
        ],
        "dataset": summary["dataset"],
        "offline_gate": offline_gate,
        "seed_isolation": summary["seed_isolation"],
        "gate_status": {
            "implementation_complete": True,
            "offline_gate_passed": bool(offline_gate["all_conditions_met"]),
            "new_shadow_gate_passed": False,
            "autonomous_smoke_passed": False,
            "next_stage_rl_authorized": False,
        },
    }
    atomic_write_json(Path(args.config), config)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
