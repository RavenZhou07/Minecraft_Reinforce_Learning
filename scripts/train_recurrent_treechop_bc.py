"""Train the controlled Natural Treechop CNN/GRU behaviour-cloning baseline."""

import argparse
import csv
import json
import math
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from mc_rl.actions import ACTION_NAMES
from mc_rl.experiments import append_experiment, file_sha256
from mc_rl.natural_treechop_bc import balanced_accuracy
from mc_rl.recurrent_treechop_bc import (
    ACTION_COUNT,
    PREVIOUS_ACTION_DISABLED_ZERO,
    PREVIOUS_ACTION_EMBEDDED,
    EpisodeSequence,
    RecurrentArchitecture,
    RecurrentTreechopPolicy,
    class_weights_for_episodes,
    episode_batches,
    load_episode_sequences,
    masked_cross_entropy,
    paired_disabled_zero_policy,
    student_input_manifest_for_architecture,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dataset", required=True)
    parser.add_argument("--validation-dataset", required=True)
    parser.add_argument("--expected-train-sha256")
    parser.add_argument("--expected-validation-sha256")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--training-log", required=True)
    parser.add_argument("--config-output", required=True)
    parser.add_argument("--seed-manifest", default="configs/seeds/natural_treechop_v1.json")
    parser.add_argument("--train-seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--class-weight-power", type=float, default=0.5)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--overfit-seeds", type=int, nargs="*", default=[])
    parser.add_argument("--acceptance-accuracy", type=float)
    parser.add_argument("--acceptance-balanced-accuracy", type=float)
    parser.add_argument(
        "--previous-action-mode",
        choices=(PREVIOUS_ACTION_EMBEDDED, PREVIOUS_ACTION_DISABLED_ZERO),
        default=PREVIOUS_ACTION_EMBEDDED,
    )
    parser.add_argument("--initialization-audit")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--hypothesis", required=True)
    return parser.parse_args()


def atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def atomic_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def set_training_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def split_metrics(
    policy: RecurrentTreechopPolicy,
    episodes: Sequence[EpisodeSequence],
    batch_size: int,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    labels: List[np.ndarray] = []
    predictions: List[np.ndarray] = []
    probabilities: List[np.ndarray] = []
    loss_total = 0.0
    sample_total = 0
    max_abs_action_slot = 0.0
    policy.model.eval()
    with torch.no_grad():
        for batch in episode_batches(
            episodes,
            batch_size=batch_size,
            shuffle=False,
            rng=np.random.RandomState(0),
        ):
            batch = batch.to(policy.device)
            logits, _, diagnostics = policy.model.forward_with_diagnostics(
                batch.pov,
                batch.legal_vector,
                batch.previous_action_token,
                hidden=None,
            )
            max_abs_action_slot = max(
                max_abs_action_slot,
                float(diagnostics["action_embedding"].abs().max().item()),
            )
            selected_logits = logits[batch.mask]
            selected_actions = batch.action[batch.mask]
            batch_loss = torch.nn.functional.cross_entropy(
                selected_logits, selected_actions, reduction="sum"
            )
            probs = torch.softmax(selected_logits, dim=-1)
            labels.append(selected_actions.cpu().numpy())
            predictions.append(torch.argmax(probs, dim=-1).cpu().numpy())
            probabilities.append(probs.cpu().numpy())
            loss_total += float(batch_loss.item())
            sample_total += int(len(selected_actions))
    label_array = np.concatenate(labels)
    prediction_array = np.concatenate(predictions)
    probability_array = np.concatenate(probabilities)
    confusion = np.zeros((ACTION_COUNT, ACTION_COUNT), dtype=np.int64)
    for label, prediction in zip(label_array, prediction_array):
        confusion[int(label), int(prediction)] += 1
    per_action: Dict[str, Any] = {}
    for action_id, action_name in enumerate(ACTION_NAMES):
        support = int(confusion[action_id].sum())
        predicted = int(confusion[:, action_id].sum())
        true_positive = int(confusion[action_id, action_id])
        per_action[action_name] = {
            "id": action_id,
            "support": support,
            "recall": true_positive / support if support else None,
            "precision": true_positive / predicted if predicted else None,
        }
    entropy = -np.sum(
        probability_array * np.log(np.clip(probability_array, 1e-12, 1.0)), axis=1
    )
    metrics = {
        "samples": sample_total,
        "episodes": len(episodes),
        "episode_seeds": [episode.seed for episode in episodes],
        "cross_entropy": loss_total / sample_total,
        "accuracy": float((prediction_array == label_array).mean()),
        "balanced_accuracy": balanced_accuracy(
            prediction_array, label_array, np.arange(ACTION_COUNT)
        ),
        "mean_prediction_entropy_nats": float(entropy.mean()),
        "action_counts": {
            str(key): int(value)
            for key, value in sorted(Counter(label_array.tolist()).items())
        },
        "predicted_action_counts": {
            str(key): int(value)
            for key, value in sorted(Counter(prediction_array.tolist()).items())
        },
        "per_action": per_action,
        "confusion_matrix_rows_label_columns_prediction": confusion.tolist(),
        "max_abs_action_slot": max_abs_action_slot,
    }
    return metrics, prediction_array, probability_array


def train(
    policy: RecurrentTreechopPolicy,
    train_episodes: Sequence[EpisodeSequence],
    validation_episodes: Sequence[EpisodeSequence],
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    patience: int,
    class_weight_power: float,
    gradient_clip: float,
    train_seed: int,
) -> List[Dict[str, Any]]:
    optimizer = torch.optim.AdamW(
        policy.model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    class_weights = class_weights_for_episodes(
        train_episodes, class_weight_power
    ).to(policy.device)
    rng = np.random.RandomState(train_seed)
    history: List[Dict[str, Any]] = []
    best_loss = math.inf
    best_state: Optional[Dict[str, torch.Tensor]] = None
    stale_epochs = 0
    for epoch in range(1, epochs + 1):
        policy.model.train()
        total_loss = 0.0
        total_samples = 0
        gradient_norms: List[float] = []
        max_abs_action_slot = 0.0
        for batch in episode_batches(
            train_episodes,
            batch_size=batch_size,
            shuffle=True,
            rng=rng,
        ):
            batch = batch.to(policy.device)
            optimizer.zero_grad(set_to_none=True)
            logits, _, diagnostics = policy.model.forward_with_diagnostics(
                batch.pov,
                batch.legal_vector,
                batch.previous_action_token,
                hidden=None,
            )
            slot_max = float(diagnostics["action_embedding"].abs().max().item())
            max_abs_action_slot = max(max_abs_action_slot, slot_max)
            if (
                policy.architecture.previous_action_mode
                == PREVIOUS_ACTION_DISABLED_ZERO
                and slot_max != 0.0
            ):
                raise RuntimeError("disabled previous-action slot became nonzero")
            loss = masked_cross_entropy(
                logits, batch.action, batch.mask, class_weights=class_weights
            )
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                policy.model.parameters(), gradient_clip
            )
            gradient_norms.append(float(gradient_norm.item()))
            optimizer.step()
            valid = int(batch.mask.sum().item())
            total_loss += float(loss.item()) * valid
            total_samples += valid
        validation, _, _ = split_metrics(policy, validation_episodes, batch_size)
        train_metrics, _, _ = split_metrics(policy, train_episodes, batch_size)
        row = {
            "epoch": epoch,
            "weighted_train_loss": total_loss / total_samples,
            "train_cross_entropy": train_metrics["cross_entropy"],
            "train_accuracy": train_metrics["accuracy"],
            "train_balanced_accuracy": train_metrics["balanced_accuracy"],
            "validation_cross_entropy": validation["cross_entropy"],
            "validation_accuracy": validation["accuracy"],
            "validation_balanced_accuracy": validation["balanced_accuracy"],
            "mean_gradient_norm_before_clip": float(np.mean(gradient_norms)),
            "max_gradient_norm_before_clip": float(np.max(gradient_norms)),
            "max_abs_action_slot": max_abs_action_slot,
        }
        history.append(row)
        if validation["cross_entropy"] < best_loss - 1e-7:
            best_loss = validation["cross_entropy"]
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in policy.model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        print(
            "epoch={:03d} train_acc={:.4f} val_acc={:.4f} val_bal={:.4f} val_ce={:.4f}".format(
                epoch,
                train_metrics["accuracy"],
                validation["accuracy"],
                validation["balanced_accuracy"],
                validation["cross_entropy"],
            ),
            flush=True,
        )
        if stale_epochs >= patience:
            break
    if best_state is None:
        raise RuntimeError("training produced no checkpoint state")
    policy.model.load_state_dict(best_state)
    return history


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.patience <= 0:
        raise ValueError("epochs, batch-size and patience must be positive")
    set_training_seed(args.train_seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    train_path = Path(args.train_dataset)
    validation_path = Path(args.validation_dataset)
    dataset_hashes = {
        "train": file_sha256(train_path),
        "validation": file_sha256(validation_path),
    }
    if (
        args.expected_train_sha256
        and dataset_hashes["train"] != args.expected_train_sha256.lower()
    ):
        raise RuntimeError("train dataset SHA-256 mismatch; training stopped")
    if (
        args.expected_validation_sha256
        and dataset_hashes["validation"] != args.expected_validation_sha256.lower()
    ):
        raise RuntimeError("validation dataset SHA-256 mismatch; training stopped")
    overfit_seeds = tuple(int(seed) for seed in args.overfit_seeds)
    train_episodes = load_episode_sequences(
        train_path,
        include_failure_teacher=False,
        selected_seeds=overfit_seeds or None,
    )
    validation_episodes = (
        train_episodes
        if overfit_seeds
        else load_episode_sequences(validation_path, include_failure_teacher=False)
    )
    train_seeds = {episode.seed for episode in train_episodes}
    validation_seeds = {episode.seed for episode in validation_episodes}
    if not overfit_seeds and train_seeds & validation_seeds:
        raise ValueError("train/validation seed overlap")

    architecture = RecurrentArchitecture(previous_action_mode=args.previous_action_mode)
    initialization_audit: Optional[Dict[str, Any]] = None
    if args.previous_action_mode == PREVIOUS_ACTION_DISABLED_ZERO:
        if not args.initialization_audit:
            raise ValueError("disabled_zero requires --initialization-audit")
        policy, initialization_audit = paired_disabled_zero_policy(
            seed=args.train_seed, device="cpu"
        )
        atomic_json(Path(args.initialization_audit), initialization_audit)
    else:
        policy = RecurrentTreechopPolicy(architecture=architecture, device="cpu")
    started = time.perf_counter()
    history = train(
        policy=policy,
        train_episodes=train_episodes,
        validation_episodes=validation_episodes,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        patience=args.patience,
        class_weight_power=args.class_weight_power,
        gradient_clip=args.gradient_clip,
        train_seed=args.train_seed,
    )
    train_metrics, _, _ = split_metrics(policy, train_episodes, args.batch_size)
    validation_metrics, _, _ = split_metrics(
        policy, validation_episodes, args.batch_size
    )
    training_metadata = {
        "train_seed": args.train_seed,
        "best_epoch": min(
            history, key=lambda row: row["validation_cross_entropy"]
        )["epoch"],
        "overfit_seeds": list(overfit_seeds),
        "previous_action_mode": args.previous_action_mode,
        "initialization_audit": initialization_audit,
    }
    policy.save(
        args.checkpoint,
        dataset_hashes=dataset_hashes,
        seed_manifest=args.seed_manifest,
        training_metadata=training_metadata,
    )
    reloaded = RecurrentTreechopPolicy.load(args.checkpoint)
    reload_consistent = all(
        torch.equal(policy.model.state_dict()[key].cpu(), value.cpu())
        for key, value in reloaded.model.state_dict().items()
    )
    elapsed = time.perf_counter() - started
    acceptance_checks = []
    if args.acceptance_accuracy is not None:
        acceptance_checks.append(train_metrics["accuracy"] >= args.acceptance_accuracy)
    if args.acceptance_balanced_accuracy is not None:
        acceptance_checks.append(
            train_metrics["balanced_accuracy"] >= args.acceptance_balanced_accuracy
        )
    acceptance_passed = None if not acceptance_checks else all(acceptance_checks)
    config = {
        "profile": args.experiment_id,
        "architecture": {
            **architecture.__dict__,
            "spatial_encoder": "conv(3,8,k5,s4)-conv(8,16,k3,s2)-conv(16,32,k3,s2)",
            "scalar_encoder": "mlp(16,32,32)",
            "gru_layers": 1,
        },
        "optimizer": {
            "name": "AdamW",
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "batch_size_episodes": args.batch_size,
            "patience": args.patience,
            "class_weight_power": args.class_weight_power,
            "gradient_clip": args.gradient_clip,
            "train_seed": args.train_seed,
        },
        "sequence_semantics": (
            "current legal observation + legal-observation-only episode-local hidden -> action_t"
            if args.previous_action_mode == PREVIOUS_ACTION_DISABLED_ZERO
            else "obs_t + previous executed action + episode-local hidden -> action_t"
        ),
        "previous_action_mode": args.previous_action_mode,
        "disabled_action_slot": (
            {"width": architecture.action_embedding, "source": "constant_zero", "trainable": False}
            if args.previous_action_mode == PREVIOUS_ACTION_DISABLED_ZERO
            else None
        ),
        "episode_start_previous_action": (
            "diagnostic/alignment only; ignored by actor"
            if args.previous_action_mode == PREVIOUS_ACTION_DISABLED_ZERO
            else "dedicated START token (id 14)"
        ),
        "hidden_reset": "zeros at each complete-episode batch row and environment reset",
        "padding_loss_masked": True,
        "student_input_manifest": list(student_input_manifest_for_architecture(architecture)),
        "train_only_privileged_supervision": [],
        "privileged_actor_inputs": 0,
        "datasets": dataset_hashes,
        "overfit_seeds": list(overfit_seeds),
        "seed_manifest": args.seed_manifest,
    }
    summary = {
        "experiment_id": args.experiment_id,
        "hypothesis": args.hypothesis,
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": file_sha256(Path(args.checkpoint)),
        "checkpoint_reload_consistent": reload_consistent,
        "dataset_hashes": dataset_hashes,
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "training_epochs": len(history),
        "best_epoch": training_metadata["best_epoch"],
        "acceptance_accuracy": args.acceptance_accuracy,
        "acceptance_balanced_accuracy": args.acceptance_balanced_accuracy,
        "acceptance_passed": acceptance_passed,
        "previous_action_mode": args.previous_action_mode,
        "initialization_pairing": initialization_audit,
        "zero_channel_assertion_passed": (
            train_metrics["max_abs_action_slot"] == 0.0
            and validation_metrics["max_abs_action_slot"] == 0.0
            if args.previous_action_mode == PREVIOUS_ACTION_DISABLED_ZERO
            else None
        ),
        "privileged_actor_inputs": 0,
        "train_only_privileged_supervision": [],
        "elapsed_seconds": round(elapsed, 3),
    }
    atomic_csv(Path(args.training_log), history)
    atomic_json(Path(args.config_output), config)
    atomic_json(Path(args.summary), summary)
    append_experiment(
        {
            "experiment_id": args.experiment_id,
            "hypothesis": args.hypothesis,
            "config": config,
            "seed_manifest": args.seed_manifest,
            "seed_split": "bc_train_overfit_sanity" if overfit_seeds else "bc_train/bc_validation",
            "dataset": dataset_hashes,
            "checkpoint": {
                "path": args.checkpoint,
                "sha256": summary["checkpoint_sha256"],
            },
            "metrics": {
                "train": train_metrics,
                "validation": validation_metrics,
                "acceptance_passed": acceptance_passed,
                "checkpoint_reload_consistent": reload_consistent,
                "privileged_actor_inputs": 0,
            },
            "runtime_seconds": summary["elapsed_seconds"],
            "conclusion": (
                "Recurrent sequence correctness overfit gate passed."
                if acceptance_passed is True
                else "Controlled recurrent BC training completed."
                if acceptance_passed is None
                else "Recurrent sequence correctness overfit gate failed."
            ),
            "status": "kept" if acceptance_passed is not False else "rejected",
        }
    )
    print(json.dumps(summary, indent=2), flush=True)
    if acceptance_passed is False:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
