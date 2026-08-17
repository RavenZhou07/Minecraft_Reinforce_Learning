"""Train the small POV-only policy on privileged teacher trajectories."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from mc_rl.vision import (
    LinearVisualPolicy,
    build_frame_stacks,
    navigation_class_labels,
    trend_summary,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="logs/find_tree/oracle_dataset.npz")
    parser.add_argument("--model", default="checkpoints/find_tree_visual_linear.npz")
    parser.add_argument("--history", default="logs/find_tree/training_history.csv")
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--feature-size", type=int, default=6)
    parser.add_argument("--frame-stack", type=int, default=1)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--validation-fraction", type=float, default=0.25)
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0 < args.validation_fraction < 1:
        raise ValueError("validation-fraction must be between zero and one")

    data = np.load(args.dataset)
    episode_ids = data["episode"]
    unique_episodes = np.unique(episode_ids)
    validation_count = max(1, int(round(len(unique_episodes) * args.validation_fraction)))
    validation_episodes = unique_episodes[-validation_count:]
    validation_mask = np.isin(episode_ids, validation_episodes)
    training_mask = ~validation_mask
    if not training_mask.any() or not validation_mask.any():
        raise ValueError("dataset does not contain enough episodes for a split")

    # The flat task has no obstacles. Teacher jump actions are intentionally
    # reduced to forward so the visual policy has three interpretable classes.
    actions = navigation_class_labels(data["action"])
    stacked_pov = build_frame_stacks(data["pov"], episode_ids, args.frame_stack)
    training_pov = stacked_pov[training_mask]
    training_actions = actions[training_mask]
    # Horizontal reflection is an exact symmetry of this flat curriculum.
    # It doubles rare turn examples and swaps left/right labels without using
    # any validation frames or privileged state as model input.
    flipped_actions = training_actions.copy()
    flipped_actions[training_actions == 3] = 4
    flipped_actions[training_actions == 4] = 3
    if training_pov.ndim == 5:
        flipped_pov = training_pov[:, :, :, ::-1, :]
    else:
        flipped_pov = training_pov[:, :, ::-1, :]
    training_pov = np.concatenate((training_pov, flipped_pov))
    training_actions = np.concatenate((training_actions, flipped_actions))

    policy = LinearVisualPolicy(
        feature_size=args.feature_size, frame_stack=args.frame_stack
    )
    history = policy.fit(
        training_pov,
        training_actions,
        stacked_pov[validation_mask],
        actions[validation_mask],
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        patience=args.patience,
    )
    calibration = policy.calibrate_forward_bias(
        stacked_pov[validation_mask], actions[validation_mask]
    )
    policy.save(args.model)

    history_path = Path(args.history)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    summary = trend_summary(history)
    summary.update(
        {
            "training_transitions": int(training_mask.sum()),
            "augmented_training_samples": int(len(training_actions)),
            "validation_transitions": int(validation_mask.sum()),
            "training_episodes": int(len(unique_episodes) - validation_count),
            "validation_episodes": int(validation_count),
            "model": args.model,
            "frame_stack": args.frame_stack,
            "selected_checkpoint_epoch": int(policy.best_epoch),
            "stopped_early": bool(policy.stopped_early),
        }
    )
    summary.update(calibration)
    summary_path = history_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
