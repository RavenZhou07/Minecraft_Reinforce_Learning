"""Bounded trainers for the controlled disabled-zero Treechop branch."""

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mc_rl.experiments import file_sha256
from mc_rl.recurrent_treechop_bc import (
    ACTION_COUNT,
    EpisodeSequence,
    RecurrentTreechopPolicy,
    class_weights_for_episodes,
    episode_batches,
    load_episode_sequences,
    masked_cross_entropy,
    paired_disabled_zero_policy,
)
from scripts.train_recurrent_treechop_bc import atomic_csv, atomic_json, set_training_seed, split_metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--training-seed", type=int)
    parser.add_argument("--output-root")
    return parser.parse_args()


def exact_state(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def verify_hash(path: str, expected: str, label: str) -> str:
    actual = file_sha256(Path(path))
    if actual != expected.lower():
        raise RuntimeError("{} dataset SHA-256 mismatch; training stopped".format(label))
    return actual


def train_epoch(
    policy: RecurrentTreechopPolicy,
    episodes: Sequence[EpisodeSequence],
    optimizer: torch.optim.Optimizer,
    weights: torch.Tensor,
    batch_size: int,
    gradient_clip: float,
    rng: np.random.RandomState,
) -> Dict[str, float]:
    policy.model.train()
    weighted_loss = 0.0
    samples = 0
    gradient_norms: List[float] = []
    zero_max = 0.0
    for batch in episode_batches(episodes, batch_size, True, rng):
        batch = batch.to(policy.device)
        optimizer.zero_grad(set_to_none=True)
        logits, _, diagnostics = policy.model.forward_with_diagnostics(
            batch.pov, batch.legal_vector, batch.previous_action_token, hidden=None
        )
        slot_max = float(diagnostics["action_embedding"].abs().max().item())
        zero_max = max(zero_max, slot_max)
        if slot_max != 0.0:
            raise RuntimeError("disabled previous-action slot became nonzero")
        loss = masked_cross_entropy(logits, batch.action, batch.mask, weights)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(policy.model.parameters(), gradient_clip)
        optimizer.step()
        valid = int(batch.mask.sum().item())
        weighted_loss += float(loss.item()) * valid
        samples += valid
        gradient_norms.append(float(gradient_norm.item()))
    return {
        "weighted_train_loss": weighted_loss / samples,
        "mean_gradient_norm_before_clip": float(np.mean(gradient_norms)),
        "max_gradient_norm_before_clip": float(np.max(gradient_norms)),
        "max_abs_action_slot": zero_max,
    }


def save_and_reload(
    policy: RecurrentTreechopPolicy,
    checkpoint: Path,
    dataset_hashes: Dict[str, str],
    metadata: Dict[str, Any],
) -> bool:
    policy.save(
        str(checkpoint),
        dataset_hashes=dataset_hashes,
        seed_manifest="configs/seeds/natural_treechop_v1.json",
        training_metadata=metadata,
    )
    reloaded = RecurrentTreechopPolicy.load(str(checkpoint))
    exact = all(
        torch.equal(value.cpu(), reloaded.model.state_dict()[name].cpu())
        for name, value in policy.model.state_dict().items()
    )
    if not exact:
        raise RuntimeError("checkpoint reload was not bit-exact")
    return exact


def run_capacity(config: Dict[str, Any], config_path: Path, output_root: Path) -> Dict[str, Any]:
    specification = config["dataset"]
    train_hash = verify_hash(specification["path"], specification["sha256"], "capacity")
    verify_hash(
        config["validation_hash_control"]["path"],
        config["validation_hash_control"]["sha256"],
        "validation control",
    )
    selected = [int(value) for value in specification["selected_seeds"]]
    episodes = load_episode_sequences(
        Path(specification["path"]), include_failure_teacher=False, selected_seeds=selected
    )
    if [episode.seed for episode in episodes] != selected:
        # The file may store complete episodes in a different order.  Preserve file order,
        # but require the exact declared set and no duplicates.
        if sorted(episode.seed for episode in episodes) != sorted(selected):
            raise RuntimeError("capacity subset seeds do not match predeclaration")
    timesteps = sum(episode.length for episode in episodes)
    observed_actions = len(set(np.concatenate([episode.action for episode in episodes]).tolist()))
    if timesteps != int(specification["expected_timesteps"]):
        raise RuntimeError("capacity subset timestep count mismatch")
    if observed_actions != int(specification["expected_observed_actions"]):
        raise RuntimeError("capacity subset observed-action count mismatch")
    provenance = {
        "dataset_path": specification["path"],
        "dataset_sha256": train_hash,
        "selected_seeds_declared": selected,
        "selected_seeds_file_order": [episode.seed for episode in episodes],
        "episode_ids": [episode.episode_id for episode in episodes],
        "episode_lengths": [episode.length for episode in episodes],
        "timesteps": timesteps,
        "observed_action_ids": sorted(set(np.concatenate([episode.action for episode in episodes]).tolist())),
        "observed_action_count": observed_actions,
        "all_episode_steps_start_at_zero_and_are_contiguous": True,
        "explicit_boundaries": True,
        "episode_local_hidden": True,
        "padding_excluded_from_loss": True,
        "passed": True,
    }
    atomic_json(output_root / "subset_provenance.json", provenance)

    optimizer_config = config["optimizer"]
    seed = int(optimizer_config["training_seed"])
    set_training_seed(seed)
    policy, pairing = paired_disabled_zero_policy(seed)
    atomic_json(output_root / "initialization_pairing_audit.json", pairing)
    optimizer = torch.optim.AdamW(
        policy.model.parameters(),
        lr=float(optimizer_config["learning_rate"]),
        weight_decay=float(optimizer_config["weight_decay"]),
    )
    weights = class_weights_for_episodes(episodes, float(optimizer_config["class_weight_power"])).to(policy.device)
    rng = np.random.RandomState(seed)
    bounded = config["bounded_training"]
    history: List[Dict[str, Any]] = []
    first_passing_epoch = None
    started = time.perf_counter()
    for epoch in range(1, int(bounded["maximum_epochs"]) + 1):
        optimization = train_epoch(
            policy, episodes, optimizer, weights,
            int(optimizer_config["batch_size_episodes"]),
            float(optimizer_config["gradient_clip"]), rng,
        )
        metrics, _, _ = split_metrics(policy, episodes, int(optimizer_config["batch_size_episodes"]))
        row = {
            "epoch": epoch,
            **optimization,
            "train_cross_entropy": metrics["cross_entropy"],
            "train_accuracy": metrics["accuracy"],
            "train_balanced_accuracy": metrics["balanced_accuracy"],
        }
        history.append(row)
        if epoch == 1 or epoch % 25 == 0:
            print(json.dumps(row, sort_keys=True), flush=True)
        if (
            metrics["accuracy"] >= float(bounded["minimum_accuracy"])
            and metrics["balanced_accuracy"] >= float(bounded["minimum_balanced_accuracy"])
            and metrics["max_abs_action_slot"] == float(bounded["required_zero_slot_max_abs"])
        ):
            first_passing_epoch = epoch
            break
    atomic_csv(output_root / "capacity_training_log.csv", history)
    passed = first_passing_epoch is not None
    checkpoint = output_root / "first_passing_checkpoint.pt"
    reload_exact = False
    metrics = split_metrics(policy, episodes, int(optimizer_config["batch_size_episodes"]))[0]
    if passed:
        reload_exact = save_and_reload(
            policy, checkpoint, {"train": train_hash},
            {
                "experiment_id": config["experiment_id"],
                "selection_rule": "first_epoch_meeting_all_capacity_thresholds",
                "first_passing_epoch": first_passing_epoch,
                "training_seed": seed,
            },
        )
        metrics = split_metrics(RecurrentTreechopPolicy.load(str(checkpoint)), episodes, int(optimizer_config["batch_size_episodes"]))[0]
    summary = {
        "experiment_id": config["experiment_id"],
        "config": str(config_path),
        "config_sha256": file_sha256(config_path),
        "training_epochs": len(history),
        "first_passing_epoch": first_passing_epoch,
        "checkpoint": str(checkpoint) if passed else None,
        "checkpoint_sha256": file_sha256(checkpoint) if passed else None,
        "checkpoint_reload_exact": reload_exact,
        "train_metrics": metrics,
        "initialization_pairing": pairing,
        "acceptance_passed": passed and reload_exact,
        "zero_channel_assertion_passed": metrics["max_abs_action_slot"] == 0.0,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "protected_splits_accessed": False,
    }
    atomic_json(output_root / "capacity_training_summary.json", summary)
    if not summary["acceptance_passed"]:
        raise SystemExit(2)
    return summary


def formal_dataset_config(config: Dict[str, Any]) -> Dict[str, Any]:
    datasets = config["datasets"]
    if "train" in datasets:
        return datasets
    return {
        "train": {"path": datasets["train_path"], "sha256": datasets["train_sha256"]},
        "validation": {"path": datasets["validation_path"], "sha256": datasets["validation_sha256"]},
    }


def run_formal(
    config: Dict[str, Any], config_path: Path, output_root: Path, training_seed: int
) -> Dict[str, Any]:
    datasets = formal_dataset_config(config)
    hashes = {
        "train": verify_hash(datasets["train"]["path"], datasets["train"]["sha256"], "train"),
        "validation": verify_hash(datasets["validation"]["path"], datasets["validation"]["sha256"], "validation"),
    }
    train_episodes = load_episode_sequences(Path(datasets["train"]["path"]), include_failure_teacher=False)
    validation_episodes = load_episode_sequences(Path(datasets["validation"]["path"]), include_failure_teacher=False)
    if {episode.seed for episode in train_episodes} & {episode.seed for episode in validation_episodes}:
        raise RuntimeError("formal train/validation seed overlap")
    audit = {
        "train": {"path": datasets["train"]["path"], "sha256": hashes["train"], "samples": sum(e.length for e in train_episodes), "seeds": [e.seed for e in train_episodes]},
        "validation": {"path": datasets["validation"]["path"], "sha256": hashes["validation"], "samples": sum(e.length for e in validation_episodes), "seeds": [e.seed for e in validation_episodes]},
        "hashes_match": True,
        "seed_disjoint": True,
        "protected_splits_accessed": False,
    }
    atomic_json(output_root / "dataset_hash_audit.json", audit)
    training = config["training"]
    set_training_seed(training_seed)
    policy, pairing = paired_disabled_zero_policy(training_seed)
    atomic_json(output_root / "initialization_pairing_audit.json", pairing)
    optimizer = torch.optim.AdamW(
        policy.model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"])
    )
    weights = class_weights_for_episodes(train_episodes, float(training["class_weight_power"])).to(policy.device)
    rng = np.random.RandomState(training_seed)
    minimum = int(training["minimum_recorded_horizon"])
    maximum = int(training["maximum_epochs"])
    patience = int(training["validation_ce_patience_after_minimum_horizon"])
    best_loss = math.inf
    best_epoch = 0
    best_state = None
    stale = 0
    history: List[Dict[str, Any]] = []
    epoch60_snapshot = None
    started = time.perf_counter()
    for epoch in range(1, maximum + 1):
        optimization = train_epoch(
            policy, train_episodes, optimizer, weights,
            int(training["batch_size_episodes"]), float(training["gradient_clip"]), rng,
        )
        train_metrics = split_metrics(policy, train_episodes, int(training["batch_size_episodes"]))[0]
        validation_metrics = split_metrics(policy, validation_episodes, int(training["batch_size_episodes"]))[0]
        row = {
            "epoch": epoch, **optimization,
            "train_cross_entropy": train_metrics["cross_entropy"],
            "train_accuracy": train_metrics["accuracy"],
            "train_balanced_accuracy": train_metrics["balanced_accuracy"],
            "validation_cross_entropy": validation_metrics["cross_entropy"],
            "validation_accuracy": validation_metrics["accuracy"],
            "validation_balanced_accuracy": validation_metrics["balanced_accuracy"],
        }
        history.append(row)
        if validation_metrics["cross_entropy"] < best_loss - 1e-7:
            best_loss = validation_metrics["cross_entropy"]
            best_epoch = epoch
            best_state = exact_state(policy.model)
            stale = 0
        else:
            stale += 1
        if epoch == minimum:
            epoch60_snapshot = {
                "epoch": epoch,
                "current_train_metrics": train_metrics,
                "current_validation_metrics": validation_metrics,
                "best_epoch_within_matched_budget": best_epoch,
                "best_validation_cross_entropy_within_matched_budget": best_loss,
                "diagnostic_only": True,
            }
            atomic_json(output_root / "epoch60_snapshot_metrics.json", epoch60_snapshot)
        if epoch == 1 or epoch % 10 == 0:
            print(json.dumps(row, sort_keys=True), flush=True)
        if epoch >= minimum and stale >= patience:
            break
    if best_state is None or epoch60_snapshot is None:
        raise RuntimeError("formal training did not produce required checkpoints")
    policy.model.load_state_dict(best_state)
    checkpoint = output_root / "no_action_formal_seed{}.pt".format(training_seed)
    reload_exact = save_and_reload(
        policy, checkpoint, hashes,
        {
            "experiment_id": config["experiment_id"],
            "training_seed": training_seed,
            "best_epoch": best_epoch,
            "selection_rule": "minimum_validation_cross_entropy_within_single_run",
            "minimum_recorded_horizon": minimum,
            "maximum_epochs": maximum,
            "post_minimum_patience": patience,
        },
    )
    reloaded = RecurrentTreechopPolicy.load(str(checkpoint))
    train_metrics = split_metrics(reloaded, train_episodes, int(training["batch_size_episodes"]))[0]
    validation_metrics = split_metrics(reloaded, validation_episodes, int(training["batch_size_episodes"]))[0]
    atomic_csv(output_root / "training_log.csv", history)
    atomic_json(output_root / "selected_validation_metrics.json", validation_metrics)
    summary = {
        "experiment_id": config["experiment_id"],
        "config": str(config_path),
        "config_sha256": file_sha256(config_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "checkpoint_reload_exact": reload_exact,
        "training_seed": training_seed,
        "training_epochs": len(history),
        "best_epoch": best_epoch,
        "stopping_reason": "post_60_patience" if len(history) < maximum else "maximum_epochs",
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "initialization_pairing": pairing,
        "zero_channel_assertion_passed": train_metrics["max_abs_action_slot"] == validation_metrics["max_abs_action_slot"] == 0.0,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "protected_splits_accessed": False,
    }
    atomic_json(output_root / "training_summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("previous_action_mode") != "disabled_zero":
        raise ValueError("bounded trainer requires disabled_zero")
    if set(config.get("protected_splits", [])) != {"student_holdout", "final_test"}:
        raise PermissionError("protected split declaration changed")
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    output_root = Path(args.output_root or config["outputs"]["root"])
    output_root.mkdir(parents=True, exist_ok=True)
    if config["stage"] == "MULTI_CAPACITY":
        result = run_capacity(config, config_path, output_root)
    else:
        seed = args.training_seed
        if seed is None:
            seed = int(config["training"]["training_seed"])
        result = run_formal(config, config_path, output_root, int(seed))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
