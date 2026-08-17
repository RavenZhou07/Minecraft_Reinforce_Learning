"""Run a reproducible uniform-random baseline for the one-log milestone."""

import argparse
import csv
import json
import math
import time
from collections import Counter
from pathlib import Path
from statistics import mean, median

import numpy as np
from PIL import Image

from mc_rl.actions import ACTION_NAMES
from mc_rl.envs import make_env


FIELDNAMES = (
    "episode",
    "seed",
    "steps",
    "cumulative_reward",
    "success",
    "termination",
    "reset_seconds",
    "rollout_seconds",
    "total_seconds",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", default="logs/treechop_random_baseline_seed42.csv"
    )
    parser.add_argument(
        "--frame-dir", default="logs/treechop_random_baseline_frames"
    )
    return parser.parse_args()


def wilson_interval(successes, trials, z=1.96):
    """Return a 95% Wilson interval for a Bernoulli success rate."""

    if trials <= 0:
        return 0.0, 0.0
    probability = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (probability + z * z / (2.0 * trials)) / denominator
    margin = (
        z
        * math.sqrt(
            probability * (1.0 - probability) / trials
            + z * z / (4.0 * trials * trials)
        )
        / denominator
    )
    return max(0.0, centre - margin), min(1.0, centre + margin)


def summarize(rows, action_counts):
    successes = sum(row["success"] for row in rows)
    rewards = [row["cumulative_reward"] for row in rows]
    steps = [row["steps"] for row in rows]
    resets = [row["reset_seconds"] for row in rows]
    rollouts = [row["rollout_seconds"] for row in rows]
    lower, upper = wilson_interval(successes, len(rows))
    return {
        "episodes": len(rows),
        "successes": successes,
        "success_rate": successes / len(rows) if rows else 0.0,
        "success_rate_wilson_95": [lower, upper],
        "mean_reward": mean(rewards) if rewards else 0.0,
        "median_reward": median(rewards) if rewards else 0.0,
        "mean_steps": mean(steps) if steps else 0.0,
        "median_steps": median(steps) if steps else 0.0,
        "mean_reset_seconds": mean(resets) if resets else 0.0,
        "median_reset_seconds": median(resets) if resets else 0.0,
        "mean_rollout_seconds": mean(rollouts) if rollouts else 0.0,
        "total_actions": sum(action_counts.values()),
        "action_counts": {
            ACTION_NAMES[action_id]: action_counts.get(action_id, 0)
            for action_id in range(len(ACTION_NAMES))
        },
    }


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    if args.episodes <= 0 or args.max_steps <= 0:
        raise ValueError("episodes and max-steps must be positive")

    output_path = Path(args.output)
    summary_path = output_path.with_suffix(".summary.json")
    frame_dir = Path(args.frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(
        "MineRLTreechop-v0",
        discrete_actions=True,
        max_episode_steps=args.max_steps,
        one_log_treechop=True,
    )
    # A baseline should expose the first crash instead of silently restarting
    # Minecraft and contaminating timing measurements.
    env.unwrapped._is_fault_tolerant = False

    rows = []
    action_counts = Counter()
    experiment_started = time.perf_counter()
    try:
        for episode_index in range(args.episodes):
            episode = episode_index + 1
            episode_seed = args.seed + episode_index
            env.seed(episode_seed)
            action_rng = np.random.RandomState(episode_seed)

            episode_started = time.perf_counter()
            reset_started = time.perf_counter()
            observation = env.reset()
            reset_seconds = time.perf_counter() - reset_started
            Image.fromarray(observation["pov"]).save(
                frame_dir / "episode_{:03d}_initial.png".format(episode)
            )

            rollout_started = time.perf_counter()
            cumulative_reward = 0.0
            done = False
            info = {}
            steps = 0
            while not done:
                action = int(action_rng.randint(env.action_space.n))
                action_counts[action] += 1
                observation, reward, done, info = env.step(action)
                steps += 1
                cumulative_reward += float(reward)
            rollout_seconds = time.perf_counter() - rollout_started

            success = bool(info.get("success", False))
            if success:
                termination = "one_log"
            elif info.get("TimeLimit.truncated", False):
                termination = "step_limit"
            else:
                termination = "minerl_done"
            Image.fromarray(observation["pov"]).save(
                frame_dir / "episode_{:03d}_terminal.png".format(episode)
            )

            row = {
                "episode": episode,
                "seed": episode_seed,
                "steps": steps,
                "cumulative_reward": cumulative_reward,
                "success": success,
                "termination": termination,
                "reset_seconds": round(reset_seconds, 3),
                "rollout_seconds": round(rollout_seconds, 3),
                "total_seconds": round(time.perf_counter() - episode_started, 3),
            }
            rows.append(row)
            # Rewrite after every episode so completed data survives a later
            # Minecraft failure or an interrupted long run.
            write_rows(output_path, rows)
            print(
                "episode={}/{} seed={} steps={} reward={:.3f} success={} "
                "reset={:.2f}s rollout={:.2f}s termination={}".format(
                    episode,
                    args.episodes,
                    episode_seed,
                    steps,
                    cumulative_reward,
                    success,
                    reset_seconds,
                    rollout_seconds,
                    termination,
                ),
                flush=True,
            )
    finally:
        env.close()

    summary = summarize(rows, action_counts)
    summary["base_seed"] = args.seed
    summary["max_episode_steps"] = args.max_steps
    summary["experiment_wall_seconds"] = round(
        time.perf_counter() - experiment_started, 3
    )
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("episode CSV:", output_path)
    print("summary JSON:", summary_path)


if __name__ == "__main__":
    main()
