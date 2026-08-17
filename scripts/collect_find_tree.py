"""Collect privileged-teacher trajectories from the custom FindTree task."""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from mc_rl.find_tree_env import close_find_tree_env, make_find_tree_env
from mc_rl.navigation import OracleNavigator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--max-steps", type=int, default=160)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--yaw-noise", type=float, default=30.0)
    parser.add_argument("--distance-min", type=int, default=5)
    parser.add_argument("--distance-max", type=int, default=8)
    parser.add_argument("--distractor-trees", type=int, default=0)
    parser.add_argument(
        "--student-observable-teacher",
        action="store_true",
        help="Always scan right while the target is outside the visual FOV.",
    )
    parser.add_argument("--output", default="logs/find_tree/oracle_dataset.npz")
    return parser.parse_args()


def save_dataset(path, pov, oracle, actions, episodes, seeds, steps):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        pov=np.asarray(pov, dtype=np.uint8),
        oracle=np.asarray(oracle, dtype=np.float32),
        action=np.asarray(actions, dtype=np.int64),
        episode=np.asarray(episodes, dtype=np.int64),
        episode_seed=np.asarray(seeds, dtype=np.int64),
        episode_step=np.asarray(steps, dtype=np.int64),
    )


def main():
    args = parse_args()
    if args.episodes <= 1 or args.max_steps <= 0:
        raise ValueError("episodes must exceed one and max-steps must be positive")

    output_path = Path(args.output)
    episodes_path = output_path.with_suffix(".episodes.csv")
    summary_path = output_path.with_suffix(".summary.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pov_frames = []
    oracle_states = []
    actions = []
    episode_ids = []
    episode_seeds = []
    episode_steps = []
    episode_rows = []
    controller = OracleNavigator(
        search_clockwise_outside_fov=args.student_observable_teacher
    )
    env = make_find_tree_env(
        args.seed,
        args.max_steps,
        args.yaw_noise,
        args.distance_min,
        args.distance_max,
        args.distractor_trees,
    )
    started_at = time.perf_counter()
    try:
        for episode_index in range(args.episodes):
            episode_seed = args.seed + episode_index
            env.seed(episode_seed)
            reset_started = time.perf_counter()
            observation = env.reset()
            reset_seconds = time.perf_counter() - reset_started
            if observation["oracle"][0] <= 0:
                raise RuntimeError(
                    "seed {} produced no log in the privileged grid".format(
                        episode_seed
                    )
                )

            controller.reset()
            done = False
            cumulative_reward = 0.0
            step = 0
            info = {}
            while not done:
                action = controller.act(observation["oracle"])
                pov_frames.append(observation["pov"].copy())
                oracle_states.append(observation["oracle"].copy())
                actions.append(action)
                episode_ids.append(episode_index)
                episode_seeds.append(episode_seed)
                episode_steps.append(step)

                observation, reward, done, info = env.step(action)
                cumulative_reward += float(reward)
                step += 1

            row = {
                "episode": episode_index + 1,
                "seed": episode_seed,
                "steps": step,
                "cumulative_reward": round(cumulative_reward, 6),
                "success": bool(info.get("success", False)),
                "final_distance": info.get("target_distance"),
                "reset_seconds": round(reset_seconds, 3),
            }
            episode_rows.append(row)
            save_dataset(
                output_path,
                pov_frames,
                oracle_states,
                actions,
                episode_ids,
                episode_seeds,
                episode_steps,
            )
            print(
                "episode={}/{} seed={} steps={} reward={:.3f} success={} "
                "distance={:.3f} reset={:.2f}s".format(
                    episode_index + 1,
                    args.episodes,
                    episode_seed,
                    step,
                    cumulative_reward,
                    row["success"],
                    float(row["final_distance"]),
                    reset_seconds,
                ),
                flush=True,
            )
    finally:
        close_find_tree_env(env)

    with episodes_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=episode_rows[0].keys())
        writer.writeheader()
        writer.writerows(episode_rows)

    summary = {
        "episodes": args.episodes,
        "successful_episodes": sum(row["success"] for row in episode_rows),
        "success_rate": sum(row["success"] for row in episode_rows)
        / args.episodes,
        "transitions": len(actions),
        "base_seed": args.seed,
        "max_steps": args.max_steps,
        "yaw_noise_degrees": args.yaw_noise,
        "target_distance_range": [args.distance_min, args.distance_max],
        "distractor_tree_count": args.distractor_trees,
        "student_observable_teacher": args.student_observable_teacher,
        "elapsed_seconds": round(time.perf_counter() - started_at, 3),
        "dataset": str(output_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
