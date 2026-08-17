"""Compare random, privileged-oracle, and POV-only FindTree policies."""

import argparse
import csv
import json
import time
from collections import Counter, defaultdict, deque
from pathlib import Path

import cv2
import numpy as np

from mc_rl.find_tree_env import close_find_tree_env, make_find_tree_env
from mc_rl.navigation import OracleNavigator
from mc_rl.vision import LinearVisualPolicy, clockwise_search_action


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=160)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--yaw-noise", type=float, default=30.0)
    parser.add_argument("--distance-min", type=int, default=5)
    parser.add_argument("--distance-max", type=int, default=8)
    parser.add_argument("--distractor-trees", type=int, default=0)
    parser.add_argument("--model", default="checkpoints/find_tree_visual_linear.npz")
    parser.add_argument("--output", default="logs/find_tree/evaluation.csv")
    parser.add_argument(
        "--trace-dir",
        default=None,
        help="Optional directory for per-step CSV traces and diagnostic POV frames.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Save one diagnostic POV frame every N steps when --trace-dir is set.",
    )
    parser.add_argument(
        "--clockwise-search",
        action="store_true",
        help="Map either visual turn prediction to a consistent clockwise scan.",
    )
    parser.add_argument(
        "--modes", nargs="+", choices=("random", "oracle", "visual"),
        default=("random", "oracle", "visual")
    )
    return parser.parse_args()


def write_rows(path, rows):
    """Persist completed episodes so a later Minecraft failure loses no data."""

    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def save_rgb(path, frame):
    """Save MineRL's RGB POV in a format image viewers display correctly."""

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


def main():
    args = parse_args()
    if args.episodes <= 0 or args.max_steps <= 0 or args.save_every <= 0:
        raise ValueError("episodes, max-steps and save-every must be positive")

    visual_policy = (
        LinearVisualPolicy.load(args.model) if "visual" in args.modes else None
    )
    rng = np.random.RandomState(args.seed)
    controller = OracleNavigator()
    rows = []
    output = Path(args.output)
    trace_root = Path(args.trace_dir) if args.trace_dir else None
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
        for mode in args.modes:
            for episode_index in range(args.episodes):
                episode_seed = args.seed + episode_index
                env.seed(episode_seed)
                observation = env.reset()
                visual_history = None
                if visual_policy is not None:
                    visual_history = deque(
                        [observation["pov"].copy()] * visual_policy.frame_stack,
                        maxlen=visual_policy.frame_stack,
                    )
                controller.reset()
                initial_distance = float(observation["oracle"][1])
                done = False
                step = 0
                cumulative_reward = 0.0
                info = {}
                action_counts = Counter()
                trace_rows = []
                while not done:
                    probabilities = None
                    if mode == "random":
                        action = int(rng.randint(env.action_space.n))
                    elif mode == "oracle":
                        action = controller.act(observation["oracle"])
                    else:
                        if visual_policy.frame_stack == 1:
                            model_input = observation["pov"]
                        else:
                            model_input = np.asarray(visual_history)[None, ...]
                        probabilities = visual_policy.predict_proba(model_input)[0]
                        calibrated = probabilities.copy()
                        forward_index = int(
                            np.where(visual_policy.classes == 1)[0][0]
                        )
                        calibrated[forward_index] += visual_policy.forward_bias
                        action = int(visual_policy.classes[calibrated.argmax()])
                        if args.clockwise_search:
                            action = clockwise_search_action(action)

                    if trace_root is not None and step % args.save_every == 0:
                        save_rgb(
                            trace_root / mode / "seed_{}".format(episode_seed)
                            / "step_{:04d}.png".format(step),
                            observation["pov"],
                        )
                    observation, reward, done, info = env.step(action)
                    if visual_history is not None:
                        visual_history.append(observation["pov"].copy())
                    cumulative_reward += float(reward)
                    action_counts[action] += 1
                    if trace_root is not None:
                        trace_rows.append(
                            {
                                "step": step,
                                "action": action,
                                "prob_forward": "" if probabilities is None else float(probabilities[0]),
                                "prob_left": "" if probabilities is None else float(probabilities[1]),
                                "prob_right": "" if probabilities is None else float(probabilities[2]),
                                "reward": float(reward),
                                "cumulative_reward": cumulative_reward,
                                "target_distance": info.get("target_distance"),
                                "done": done,
                            }
                        )
                    step += 1

                if trace_root is not None:
                    episode_trace_dir = (
                        trace_root / mode / "seed_{}".format(episode_seed)
                    )
                    save_rgb(episode_trace_dir / "terminal.png", observation["pov"])
                    write_rows(episode_trace_dir / "steps.csv", trace_rows)

                rows.append(
                    {
                        "mode": mode,
                        "episode": episode_index + 1,
                        "seed": episode_seed,
                        "steps": step,
                        "success": bool(info.get("success", False)),
                        "initial_distance": initial_distance,
                        "final_distance": info.get("target_distance"),
                        "cumulative_reward": round(cumulative_reward, 6),
                        "action_counts": json.dumps(
                            dict(sorted(action_counts.items())), sort_keys=True
                        ),
                    }
                )
                write_rows(output, rows)
                print(
                    "mode={} episode={}/{} seed={} steps={} success={} "
                    "distance={:.3f}->{:.3f} reward={:.3f}".format(
                        mode,
                        episode_index + 1,
                        args.episodes,
                        episode_seed,
                        step,
                        rows[-1]["success"],
                        initial_distance,
                        float(rows[-1]["final_distance"]),
                        cumulative_reward,
                    ),
                    flush=True,
                )
    finally:
        close_find_tree_env(env)

    write_rows(output, rows)

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["mode"]].append(row)
    summary = {
        "episodes_per_mode": args.episodes,
        "base_seed": args.seed,
        "yaw_noise_degrees": args.yaw_noise,
        "target_distance_range": [args.distance_min, args.distance_max],
        "distractor_tree_count": args.distractor_trees,
        "clockwise_search": args.clockwise_search,
    }
    for mode, mode_rows in grouped.items():
        summary[mode] = {
            "success_rate": sum(row["success"] for row in mode_rows)
            / len(mode_rows),
            "mean_steps": float(np.mean([row["steps"] for row in mode_rows])),
            "mean_final_distance": float(
                np.mean([float(row["final_distance"]) for row in mode_rows])
            ),
            "mean_reward": float(
                np.mean([row["cumulative_reward"] for row in mode_rows])
            ),
        }
    summary["elapsed_seconds"] = round(time.perf_counter() - started_at, 3)
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
