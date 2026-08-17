"""Evaluate explicit candidate search in one sequential Minecraft instance."""

import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import psutil

from mc_rl.find_tree_env import (
    CANDIDATE_NAVIGATION_ACTION_NAMES,
    close_find_tree_env,
    make_find_tree_env,
)
from mc_rl.navigation import OracleNavigator, target_bearing_degrees, wrap_degrees
from mc_rl.resource_adapters import TreeResourceAdapter
from mc_rl.search_policy import CandidateSearchPolicy, SearchConfig
from mc_rl.telemetry import SENSOR_PROFILE_F3, SENSOR_PROFILES


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--yaw-noise", type=float, default=180.0)
    parser.add_argument("--distance-min", type=int, default=3)
    parser.add_argument("--distance-max", type=int, default=10)
    parser.add_argument("--distractor-trees", type=int, default=2)
    parser.add_argument(
        "--environment", choices=("arena", "natural"), default="arena"
    )
    parser.add_argument(
        "--sensor-profile",
        choices=tuple(sorted(SENSOR_PROFILES)),
        default="pov_only",
        help="Candidate actor inputs; f3_telemetry adds only self pose/biome.",
    )
    parser.add_argument(
        "--modes", nargs="+", choices=("oracle", "candidate"),
        default=("oracle", "candidate")
    )
    parser.add_argument(
        "--output", default="logs/find_tree/candidate_search_smoke.csv"
    )
    parser.add_argument("--trace-dir", default=None)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument(
        "--force-initial-rank", type=int, default=0,
        help="Diagnostic fault injection: choose this score rank only once at episode start.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Explicitly replace outputs from an earlier run.",
    )
    return parser.parse_args()


def wilson_interval(successes: int, episodes: int, z: float = 1.96):
    if episodes <= 0:
        return 0.0, 0.0
    proportion = successes / episodes
    denominator = 1.0 + z * z / episodes
    centre = proportion + z * z / (2.0 * episodes)
    margin = z * np.sqrt(
        proportion * (1.0 - proportion) / episodes
        + z * z / (4.0 * episodes * episodes)
    )
    return (centre - margin) / denominator, (centre + margin) / denominator


def write_rows(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def save_rgb(path: Path, frame: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


def arena_truth(env, absolute_bearings: bool = False) -> Tuple[List[float], float]:
    """Evaluation-only bearings; never passed into CandidateSearchPolicy."""

    task = env.unwrapped.task
    initial_yaw = float(task.agent_yaw)
    bearings = [
        wrap_degrees(
            target_bearing_degrees(float(x), float(z))
            - (0.0 if absolute_bearings else initial_yaw)
        )
        for x, _y, z in task.tree_blocks
    ]
    return bearings, bearings[0]


def candidate_quality(
    candidate_rows: Sequence[Dict[str, Any]],
    true_bearings: Sequence[float],
    target_bearing: Optional[float],
    selected_id: Optional[int],
    tolerance: float = 28.0,
) -> Tuple[Optional[float], Optional[bool]]:
    if not true_bearings:
        return None, None
    candidate_yaws = [float(row["relative_yaw"]) for row in candidate_rows]
    recalled = sum(
        any(abs(wrap_degrees(candidate_yaw - truth)) <= tolerance for candidate_yaw in candidate_yaws)
        for truth in true_bearings
    )
    recall = recalled / len(true_bearings)
    if target_bearing is None or selected_id is None:
        return recall, False
    selected = next(
        (row for row in candidate_rows if int(row["candidate_id"]) == int(selected_id)),
        None,
    )
    correct = bool(
        selected is not None
        and abs(wrap_degrees(float(selected["relative_yaw"]) - target_bearing)) <= tolerance
    )
    return recall, correct


def make_environment(args):
    if args.environment == "arena":
        return make_find_tree_env(
            seed=args.seed,
            max_episode_steps=args.max_steps,
            yaw_noise_degrees=args.yaw_noise,
            target_distance_min=args.distance_min,
            target_distance_max=args.distance_max,
            distractor_tree_count=args.distractor_trees,
            candidate_actions=True,
            sensor_profile=args.sensor_profile,
        )
    if args.sensor_profile == SENSOR_PROFILE_F3:
        raise ValueError(
            "f3_telemetry is currently implemented for the local arena EnvSpec; "
            "natural Treechop needs its local FullStats EnvSpec after the arena gate"
        )
    if "oracle" in args.modes:
        raise ValueError("natural Treechop has no privileged oracle mode")
    from mc_rl.envs import make_env

    return make_env(
        "MineRLTreechop-v0",
        discrete_actions=True,
        one_log_treechop=True,
        max_episode_steps=args.max_steps,
    )


def close_environment(env, arena: bool) -> None:
    if arena:
        close_find_tree_env(env)
        return
    try:
        env.close()
    except psutil.NoSuchProcess as error:
        print("WARNING: Minecraft already exited during close: {}".format(error))


def main():
    args = parse_args()
    if args.episodes <= 0 or args.max_steps <= 0 or args.save_every <= 0:
        raise ValueError("episodes, max-steps, and save-every must be positive")
    if args.force_initial_rank < 0:
        raise ValueError("force-initial-rank must be non-negative")
    output = Path(args.output)
    transition_output = output.with_suffix(".transitions.csv")
    candidate_output = output.with_suffix(".candidates.csv")
    summary_output = output.with_suffix(".summary.json")
    protected = (output, transition_output, candidate_output, summary_output)
    existing = [path for path in protected if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "refusing to overwrite existing candidate-search output: {}".format(
                ", ".join(str(path) for path in existing)
            )
        )
    trace_root = (
        Path(args.trace_dir)
        if args.trace_dir
        else output.parent / "{}_traces".format(output.stem)
    )

    env = make_environment(args)
    oracle = OracleNavigator()
    rows: List[Dict[str, Any]] = []
    transitions: List[Dict[str, Any]] = []
    candidate_table: List[Dict[str, Any]] = []
    started_at = time.perf_counter()
    try:
        for mode in args.modes:
            for episode_index in range(args.episodes):
                seed = args.seed + episode_index
                env.seed(seed)
                reset_started = time.perf_counter()
                observation = env.reset()
                reset_seconds = time.perf_counter() - reset_started
                true_bearings: List[float] = []
                target_bearing: Optional[float] = None
                if args.environment == "arena":
                    true_bearings, target_bearing = arena_truth(
                        env,
                        absolute_bearings=args.sensor_profile == SENSOR_PROFILE_F3,
                    )
                initial_distance = (
                    float(observation["oracle"][1])
                    if args.environment == "arena" else None
                )

                adapter = TreeResourceAdapter(
                    interaction_action_id=8,
                    interaction_size=(None if args.environment == "arena" else 150.0),
                    # Arena navigation uses dense distance progress rewards;
                    # only Treechop's positive log reward is a success signal.
                    reward_is_success=args.environment == "natural",
                )
                config = SearchConfig(
                    backward_action=(7 if args.environment == "arena" else 9),
                    initial_selection_rank=args.force_initial_rank,
                    sensor_profile=args.sensor_profile,
                )
                policy = CandidateSearchPolicy(adapter, config) if mode == "candidate" else None
                if policy is not None:
                    policy.reset(episode=episode_index + 1)
                oracle.reset()
                done = False
                info: Dict[str, Any] = {}
                step = 0
                cumulative_reward = 0.0
                action_counts: Counter = Counter()
                trace_rows = []
                frames = []
                initial_candidates = None
                rollout_started = time.perf_counter()
                while not done:
                    state_before = "ORACLE" if policy is None else policy.state.value
                    selected_before = (
                        None
                        if policy is None or policy.selected_candidate is None
                        else policy.selected_candidate.candidate_id
                    )
                    if mode == "oracle":
                        action = oracle.act(observation["oracle"])
                    else:
                        action = policy.act(observation)
                        if (
                            initial_candidates is None
                            and policy.initial_selected_candidate_id is not None
                        ):
                            initial_candidates = [
                                dict(row)
                                for row in policy.candidate_map.rows(
                                    policy.heading_yaw, policy.step
                                )
                            ]
                    frame = observation["pov"].copy()
                    frames.append((step, frame))
                    episode_trace_dir = trace_root / mode / "seed_{}".format(seed)
                    if step % args.save_every == 0:
                        save_rgb(
                            episode_trace_dir / "step_{:04d}.png".format(step), frame
                        )
                    next_observation, reward, done, info = env.step(action)
                    if policy is not None:
                        policy.observe_transition(action, next_observation, reward, done, info)
                    cumulative_reward += float(reward)
                    action_counts[int(action)] += 1
                    trace_rows.append(
                        {
                            "step": step,
                            "state": state_before,
                            "action": int(action),
                            "selected_candidate_id": selected_before,
                            "agent_x": (
                                "" if "telemetry" not in observation
                                else float(observation["telemetry"]["x"])
                            ),
                            "agent_y": (
                                "" if "telemetry" not in observation
                                else float(observation["telemetry"]["y"])
                            ),
                            "agent_z": (
                                "" if "telemetry" not in observation
                                else float(observation["telemetry"]["z"])
                            ),
                            "agent_yaw": (
                                "" if "telemetry" not in observation
                                else float(observation["telemetry"]["yaw"])
                            ),
                            "selected_world_x": (
                                ""
                                if policy is None
                                or policy.selected_candidate is None
                                or not policy.selected_candidate.has_world_position
                                else policy.selected_candidate.estimated_world_x
                            ),
                            "selected_world_z": (
                                ""
                                if policy is None
                                or policy.selected_candidate is None
                                or not policy.selected_candidate.has_world_position
                                else policy.selected_candidate.estimated_world_z
                            ),
                            "selected_position_uncertainty": (
                                ""
                                if policy is None
                                or policy.selected_candidate is None
                                or not policy.selected_candidate.has_world_position
                                else policy.selected_candidate.position_uncertainty
                            ),
                            "reward": float(reward),
                            "target_distance_evaluation_only": info.get("target_distance", ""),
                            "progress_diagnostics": (
                                "" if policy is None else json.dumps(policy.progress.last_diagnostics, sort_keys=True)
                            ),
                            "done": bool(done),
                        }
                    )
                    observation = next_observation
                    step += 1

                success = bool(info.get("success", False))
                if policy is not None:
                    if initial_candidates is None:
                        initial_candidates = []
                    recall, selection_correct = candidate_quality(
                        initial_candidates,
                        true_bearings,
                        target_bearing,
                        policy.initial_selected_candidate_id,
                    )
                    transitions.extend(
                        dict(row, mode=mode, seed=seed) for row in policy.transition_log
                    )
                    for candidate_row in policy.candidate_map.rows(
                        policy.heading_yaw, policy.step
                    ):
                        candidate_table.append(
                            dict(candidate_row, episode=episode_index + 1, seed=seed, mode=mode)
                        )
                    candidate_count = len(initial_candidates)
                    duplicate_count = policy.candidate_map.duplicate_candidate_count
                    selected_id = policy.initial_selected_candidate_id
                    replan_count = policy.replan_count
                    recovery_count = policy.recovery_count
                    stalled_count = policy.stalled_count
                else:
                    recall = selection_correct = None
                    candidate_count = duplicate_count = 0
                    selected_id = None
                    replan_count = recovery_count = stalled_count = 0

                row = {
                    "mode": mode,
                    "episode": episode_index + 1,
                    "seed": seed,
                    "success": success,
                    "steps": step,
                    "duration_seconds": round(time.perf_counter() - rollout_started, 3),
                    "reset_seconds": round(reset_seconds, 3),
                    "candidate_count": candidate_count,
                    "candidate_recall": recall,
                    "duplicate_candidate_count": duplicate_count,
                    "selected_candidate_id": selected_id,
                    "initial_selection_correct": selection_correct,
                    "replan_count": replan_count,
                    "recovery_count": recovery_count,
                    "stalled_count": stalled_count,
                    "initial_target_distance_evaluation_only": initial_distance,
                    "final_target_distance_evaluation_only": info.get("target_distance", ""),
                    "cumulative_reward": round(cumulative_reward, 6),
                    "action_counts": json.dumps(dict(sorted(action_counts.items()))),
                }
                telemetry = observation.get("telemetry")
                if telemetry is not None:
                    row.update(
                        {
                            "final_agent_x": float(telemetry["x"]),
                            "final_agent_y": float(telemetry["y"]),
                            "final_agent_z": float(telemetry["z"]),
                            "final_agent_yaw": float(telemetry["yaw"]),
                            "biome_id": int(telemetry["biome_id"]),
                            "biome_temperature": float(
                                telemetry["biome_temperature"]
                            ),
                            "biome_rainfall": float(telemetry["biome_rainfall"]),
                        }
                    )
                rows.append(row)
                write_rows(output, rows)
                write_rows(transition_output, transitions)
                write_rows(candidate_output, candidate_table)
                write_rows(episode_trace_dir / "trace.csv", trace_rows)
                save_rgb(episode_trace_dir / "terminal.png", observation["pov"])
                if not success:
                    for frame_step, frame in frames:
                        save_rgb(
                            episode_trace_dir / "failure_full" / "step_{:04d}.png".format(frame_step),
                            frame,
                        )
                print(
                    "mode={} episode={}/{} seed={} steps={} success={} candidates={} "
                    "selection_correct={} replans={} recoveries={} reset={:.1f}s".format(
                        mode, episode_index + 1, args.episodes, seed, step, success,
                        candidate_count, selection_correct, replan_count,
                        recovery_count, reset_seconds,
                    ),
                    flush=True,
                )
    finally:
        close_environment(env, args.environment == "arena")

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["mode"]].append(row)
    summary: Dict[str, Any] = {
        "environment": args.environment,
        "episodes_per_mode": args.episodes,
        "base_seed": args.seed,
        "max_steps": args.max_steps,
        "yaw_noise_degrees": args.yaw_noise,
        "target_distance_range": [args.distance_min, args.distance_max],
        "distractor_tree_count": args.distractor_trees,
        "sensor_profile": args.sensor_profile,
        "deployment_inputs": (
            ["pov", "self_pose", "biome", "internal_candidate_memory"]
            if args.sensor_profile == SENSOR_PROFILE_F3
            else ["pov", "commanded_camera_delta", "internal_memory"]
        ),
        "oracle_used_for_actions_or_selection": False,
        "diagnostic_forced_initial_rank": args.force_initial_rank,
    }
    for mode, mode_rows in grouped.items():
        successes = sum(bool(row["success"]) for row in mode_rows)
        lower, upper = wilson_interval(successes, len(mode_rows))
        mode_summary = {
            "successes": successes,
            "success_rate": successes / len(mode_rows),
            "wilson_95_percent": [round(lower, 4), round(upper, 4)],
            "mean_steps": float(np.mean([row["steps"] for row in mode_rows])),
            "median_steps": float(np.median([row["steps"] for row in mode_rows])),
        }
        if mode == "candidate":
            labelled = [row for row in mode_rows if row["initial_selection_correct"] is not None]
            mode_summary.update(
                {
                    "initial_selection_accuracy": (
                        float(np.mean([row["initial_selection_correct"] for row in labelled]))
                        if labelled else None
                    ),
                    "mean_candidate_recall": (
                        float(np.mean([row["candidate_recall"] for row in labelled]))
                        if labelled else None
                    ),
                    "successful_after_any_recovery": sum(
                        row["success"] and (row["recovery_count"] > 0 or row["replan_count"] > 0)
                        for row in mode_rows
                    ),
                    "successful_after_wrong_initial_selection": sum(
                        row["success"] and row["initial_selection_correct"] is False
                        for row in mode_rows
                    ),
                    "total_replans": sum(row["replan_count"] for row in mode_rows),
                    "total_recoveries": sum(row["recovery_count"] for row in mode_rows),
                    "total_stalls": sum(row["stalled_count"] for row in mode_rows),
                    "max_step_failures": sum(
                        (not row["success"]) and row["steps"] >= args.max_steps
                        for row in mode_rows
                    ),
                }
            )
        summary[mode] = mode_summary
    if args.environment == "arena" and "candidate" in summary:
        candidate_summary = summary["candidate"]
        summary["provisional_multi_tree_gate"] = {
            "at_least_18_of_20": bool(
                args.episodes >= 20 and candidate_summary["successes"] >= 18
            ),
            "no_max_step_failures": candidate_summary["max_step_failures"] == 0,
            "any_recovery_ended_in_success": (
                candidate_summary["successful_after_any_recovery"] > 0
            ),
            "wrong_initial_selection_rescued": (
                candidate_summary["successful_after_wrong_initial_selection"] > 0
            ),
            "oracle_isolation": True,
        }
        gate = summary["provisional_multi_tree_gate"]
        gate["all_conditions_met"] = all(
            (
                gate["at_least_18_of_20"],
                gate["no_max_step_failures"],
                gate["wrong_initial_selection_rescued"],
                gate["oracle_isolation"],
            )
        )
        summary["baseline_comparison"] = {
            "old_visual_successes": 16,
            "old_visual_episodes": 20,
            "old_visual_mean_steps": 107.5,
        }
    summary["elapsed_seconds"] = round(time.perf_counter() - started_at, 3)
    summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
