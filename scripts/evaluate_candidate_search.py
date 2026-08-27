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
from mc_rl.telemetry import (
    SENSOR_PROFILE_F3,
    SENSOR_PROFILE_RAYCAST,
    SENSOR_PROFILES,
    sensor_uses_telemetry,
)
from mc_rl.trunk_contact import (
    CONTACT_PROFILES,
    CONTACT_PROFILE_CLEAR_OCCLUSION,
    CONTACT_PROFILE_COORDINATE_AIM,
    CONTACT_PROFILE_COORDINATE_RECOVERY,
    CONTACT_PROFILE_COORDINATE_RECOVERY_V9_1,
    CONTACT_PROFILE_COORDINATE_CONTACT_GUARD_V9_2,
    CONTACT_PROFILE_CANDIDATE_HANDOFF_GUARD_V9_3,
    CONTACT_PROFILE_CONTACT_DROP_COMPLETION_V9_4,
    CONTACT_PROFILE_CONTACT_OWNERSHIP_SPATIAL_GUARD_V9_5,
    CONTACT_PROFILE_TERRAIN_ROUTE_DROP_COMPLETION_V9_6,
    CONTACT_PROFILE_TRACE_GUIDED_DROP_RECOVERY_V9_7,
    CONTACT_PROFILE_RAYCAST_OWNED_HANDOFF_V9_8,
    CONTACT_PROFILE_EARLY_ROUTE_RECOVERY_V9_9,
    CONTACT_PROFILE_COORDINATE_TARGET_PREEMPTION_V9_10,
    CONTACT_PROFILE_EMERGENCY_TARGET_PREEMPTION_V9_11,
)


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
        "--disable-trunk-contact", action="store_true",
        help="Turn off the v5 vision-guided trunk contact controller (A/B comparison).",
    )
    parser.add_argument(
        "--contact-profile",
        choices=tuple(sorted(CONTACT_PROFILES)),
        default=CONTACT_PROFILE_CLEAR_OCCLUSION,
        help="Versioned natural contact profile; v6_1_baseline remains frozen.",
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
        if args.sensor_profile == SENSOR_PROFILE_RAYCAST:
            raise ValueError("f3_raycast diagnostic is natural-only")
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
    if "oracle" in args.modes:
        raise ValueError("natural Treechop has no privileged oracle mode")
    if sensor_uses_telemetry(args.sensor_profile):
        from mc_rl.telemetry_treechop_env import make_telemetry_treechop_env

        return make_telemetry_treechop_env(
            seed=args.seed,
            max_episode_steps=args.max_steps,
            include_raycast=(
                args.sensor_profile == SENSOR_PROFILE_RAYCAST
            ),
        )
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
    trunk_target_output = output.with_suffix(".trunk_targets.csv")
    trunk_target_event_output = output.with_suffix(".trunk_target_events.csv")
    drop_waypoint_output = output.with_suffix(".drop_waypoints.csv")
    route_recovery_output = output.with_suffix(".route_recoveries.csv")
    summary_output = output.with_suffix(".summary.json")
    protected = (
        output,
        transition_output,
        candidate_output,
        trunk_target_output,
        trunk_target_event_output,
        drop_waypoint_output,
        route_recovery_output,
        summary_output,
    )
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
    trunk_target_table: List[Dict[str, Any]] = []
    trunk_target_event_table: List[Dict[str, Any]] = []
    drop_waypoint_table: List[Dict[str, Any]] = []
    route_recovery_table: List[Dict[str, Any]] = []
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
                        absolute_bearings=sensor_uses_telemetry(
                            args.sensor_profile
                        ),
                    )
                initial_distance = (
                    float(observation["oracle"][1])
                    if args.environment == "arena" else None
                )

                adapter = TreeResourceAdapter(
                    interaction_action_id=8,
                    interaction_size=(None if args.environment == "arena" else 45.0),
                    interaction_uses_geometry=args.environment == "natural",
                    interaction_min_apparent_size=(
                        0.0 if args.environment == "arena" else 12.0
                    ),
                    range_size_cap=(
                        None if args.environment == "arena" else 120.0
                    ),
                    # Arena navigation uses dense distance progress rewards;
                    # only Treechop's positive log reward is a success signal.
                    reward_is_success=args.environment == "natural",
                )
                config = SearchConfig(
                    backward_action=(7 if args.environment == "arena" else 9),
                    initial_selection_rank=args.force_initial_rank,
                    sensor_profile=args.sensor_profile,
                    # Dense natural forests can alternate between adjacent
                    # trunk components. The deadband must exceed one 10-degree
                    # camera command to prevent left/right limit cycles.
                    align_threshold_degrees=(
                        6.0 if args.environment == "arena" else 12.0
                    ),
                    # v5: natural runs hand the terminal approach to the
                    # trunk contact controller. Arena keeps the v4 direct
                    # interaction so earlier curriculum results stay
                    # reproducible on reruns.
                    enable_trunk_contact=(
                        args.environment == "natural"
                        and not args.disable_trunk_contact
                    ),
                    contact_profile=args.contact_profile,
                    episode_max_steps=args.max_steps,
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
                    contact_diag = (
                        {} if policy is None else policy.contact_diagnostics()
                    )
                    handoff_diag = (
                        {} if policy is None else policy.handoff_diagnostics()
                    )
                    drop_diag = contact_diag.get("drop_recovery", {})
                    post_recovery_diag = contact_diag.get(
                        "coordinate_post_recovery", {}
                    )
                    exact_rescan_diag = contact_diag.get(
                        "exact_log_rescan", {}
                    )
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
                            "agent_pitch": (
                                "" if "telemetry" not in observation
                                else float(observation["telemetry"]["pitch"])
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
                            "route_target_x": handoff_diag.get(
                                "route_target_x", ""
                            ),
                            "route_target_z": handoff_diag.get(
                                "route_target_z", ""
                            ),
                            "route_target_uncertainty": handoff_diag.get(
                                "route_target_uncertainty", ""
                            ),
                            "handoff_contact_region_limit": handoff_diag.get(
                                "contact_region_limit", ""
                            ),
                            "handoff_guard_checks": handoff_diag.get(
                                "checks", 0
                            ),
                            "handoff_guard_rejections": handoff_diag.get(
                                "rejections", 0
                            ),
                            "handoff_spatial_rejections": handoff_diag.get(
                                "spatial_rejections", 0
                            ),
                            "contact_owner_lock_steps": handoff_diag.get(
                                "contact_owner_lock_steps", 0
                            ),
                            "contact_owner_mismatches": handoff_diag.get(
                                "contact_owner_mismatches", 0
                            ),
                            "terrain_route_recovery_attempts": handoff_diag.get(
                                "terrain_route_recovery_attempts", 0
                            ),
                            "terrain_route_recovery_steps": handoff_diag.get(
                                "terrain_route_recovery_steps", 0
                            ),
                            "terrain_route_recovery_successes": handoff_diag.get(
                                "terrain_route_recovery_successes", 0
                            ),
                            "terrain_route_recovery_failures": handoff_diag.get(
                                "terrain_route_recovery_failures", 0
                            ),
                            "route_blocked_region_count": len(
                                handoff_diag.get("route_blocked_regions", [])
                            ),
                            "reward": float(reward),
                            "target_distance_evaluation_only": info.get("target_distance", ""),
                            "contact_state": (
                                "" if policy is None
                                else (
                                    contact_diag.get("state", "")
                                    if contact_diag.get("active", False)
                                    else ""
                                )
                            ),
                            "contact_engaged": contact_diag.get(
                                "engaged", False
                            ),
                            "contact_active": contact_diag.get(
                                "active", False
                            ),
                            "contact_attempt_id": contact_diag.get(
                                "attempt_id", ""
                            ),
                            "contact_candidate_id": contact_diag.get(
                                "candidate_id", ""
                            ),
                            "contact_attempt_step": contact_diag.get(
                                "attempt_step", ""
                            ),
                            "contact_crosshair_trunk_fraction": (
                                contact_diag.get("crosshair_trunk_fraction", "")
                            ),
                            "contact_trunk_area_px": (
                                contact_diag.get("trunk_area_px", "")
                            ),
                            "contact_trunk_material": contact_diag.get(
                                "trunk_material", ""
                            ),
                            "contact_leaf_occlusion_fraction": (
                                contact_diag.get(
                                    "leaf_occlusion_fraction", ""
                                )
                            ),
                            "contact_last_x": contact_diag.get(
                                "last_contact_x", ""
                            ),
                            "contact_last_y": contact_diag.get(
                                "last_contact_y", ""
                            ),
                            "contact_last_z": contact_diag.get(
                                "last_contact_z", ""
                            ),
                            "drop_block_target_x": contact_diag.get(
                                "drop_target_x", ""
                            ),
                            "drop_block_target_y": contact_diag.get(
                                "drop_target_y", ""
                            ),
                            "drop_block_target_z": contact_diag.get(
                                "drop_target_z", ""
                            ),
                            "coordinate_target_id": contact_diag.get(
                                "coordinate_target_id", ""
                            ),
                            "coordinate_target_x": contact_diag.get(
                                "coordinate_target_x", ""
                            ),
                            "coordinate_target_y": contact_diag.get(
                                "coordinate_target_y", ""
                            ),
                            "coordinate_target_z": contact_diag.get(
                                "coordinate_target_z", ""
                            ),
                            "coordinate_target_count": contact_diag.get(
                                "coordinate_target_count", 0
                            ),
                            "coordinate_target_yaw": contact_diag.get(
                                "coordinate_target_yaw", ""
                            ),
                            "coordinate_target_pitch": contact_diag.get(
                                "coordinate_target_pitch", ""
                            ),
                            "coordinate_yaw_error": contact_diag.get(
                                "coordinate_yaw_error", ""
                            ),
                            "coordinate_pitch_error": contact_diag.get(
                                "coordinate_pitch_error", ""
                            ),
                            "coordinate_horizontal_distance": contact_diag.get(
                                "coordinate_horizontal_distance", ""
                            ),
                            "coordinate_distance": contact_diag.get(
                                "coordinate_distance", ""
                            ),
                            "coordinate_target_score": contact_diag.get(
                                "coordinate_target_score", ""
                            ),
                            "coordinate_target_score_terms": json.dumps(
                                contact_diag.get("coordinate_target_score_terms", {}),
                                sort_keys=True,
                            ),
                            "coordinate_progress": json.dumps(
                                contact_diag.get("coordinate_progress", {}),
                                sort_keys=True,
                            ),
                            "post_recovery_translation_samples": (
                                post_recovery_diag.get("translation_samples", "")
                            ),
                            "post_recovery_initial_distance": (
                                post_recovery_diag.get("initial_distance", "")
                            ),
                            "post_recovery_minimum_distance": (
                                post_recovery_diag.get("minimum_distance", "")
                            ),
                            "exact_log_rescan_remaining_actions": (
                                exact_rescan_diag.get("remaining_actions", "")
                            ),
                            "exact_log_rescan_reason": exact_rescan_diag.get(
                                "reason", ""
                            ),
                            "exact_log_failed_spatial_regions": len(
                                exact_rescan_diag.get("failed_regions", [])
                            ),
                            "drop_waypoint_index": drop_diag.get(
                                "waypoint_index", ""
                            ),
                            "drop_target_x": drop_diag.get("target_x", ""),
                            "drop_target_z": drop_diag.get("target_z", ""),
                            "raycast_has_block": (
                                "" if "raycast" not in observation
                                else float(observation["raycast"]["has_block"])
                            ),
                            "raycast_is_log": (
                                "" if "raycast" not in observation
                                else float(observation["raycast"]["is_log"])
                            ),
                            "raycast_is_leaves": (
                                "" if "raycast" not in observation
                                else float(observation["raycast"]["is_leaves"])
                            ),
                            "raycast_in_range": (
                                "" if "raycast" not in observation
                                else float(observation["raycast"]["in_range"])
                            ),
                            "raycast_distance": (
                                "" if "raycast" not in observation
                                else float(observation["raycast"]["distance"])
                            ),
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
                    obstacle_recovery_count = policy.obstacle_recovery_count
                else:
                    recall = selection_correct = None
                    candidate_count = duplicate_count = 0
                    selected_id = None
                    replan_count = recovery_count = stalled_count = 0
                    obstacle_recovery_count = 0
                    contact_counters = {}
                    contact_result = ""
                    contact_rows = []
                if policy is not None:
                    final_contact = policy.contact_diagnostics()
                    contact_counters = final_contact.get("counters", {})
                    contact_result = final_contact.get("result") or ""
                    contact_rows = final_contact.get(
                        "transition_records", []
                    )
                    for target_row in final_contact.get(
                        "coordinate_target_rows", []
                    ):
                        trunk_target_table.append(
                            dict(
                                target_row,
                                episode=episode_index + 1,
                                seed=seed,
                                mode=mode,
                            )
                        )
                    for event_row in final_contact.get(
                        "coordinate_selection_records", []
                    ):
                        trunk_target_event_table.append(
                            dict(
                                event_row,
                                episode=episode_index + 1,
                                seed=seed,
                                mode=mode,
                            )
                        )
                    for waypoint_row in final_contact.get(
                        "drop_waypoint_records", []
                    ):
                        drop_waypoint_table.append(
                            dict(
                                waypoint_row,
                                episode=episode_index + 1,
                                seed=seed,
                                mode=mode,
                            )
                        )
                    for recovery_row in policy.terrain_route_recovery_records:
                        route_recovery_table.append(
                            dict(
                                recovery_row,
                                episode=episode_index + 1,
                                seed=seed,
                                mode=mode,
                            )
                        )
                    contact_transition_records = final_contact.get(
                        "transition_records", []
                    )
                    remaining_at_disappearance = [
                        args.max_steps - int(record["global_step"])
                        for record in contact_transition_records
                        if record["new_state"] == "BLOCK_DISAPPEARED"
                    ]
                    remaining_at_drop_start = [
                        args.max_steps - int(record["global_step"])
                        for record in contact_transition_records
                        if record["new_state"] == "DROP_RECOVERY"
                    ]
                    episode_drop_waypoints = [
                        row
                        for row in drop_waypoint_table
                        if row["seed"] == seed and row["mode"] == mode
                    ]
                    consecutive_blocked = 0
                    max_consecutive_blocked = 0
                    for waypoint_row in episode_drop_waypoints:
                        if waypoint_row.get("end_reason") in (
                            "reached",
                            "reward",
                        ):
                            consecutive_blocked = 0
                        else:
                            consecutive_blocked += 1
                            max_consecutive_blocked = max(
                                max_consecutive_blocked, consecutive_blocked
                            )
                else:
                    remaining_at_disappearance = []
                    remaining_at_drop_start = []
                    max_consecutive_blocked = 0

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
                    "split_candidate_count": (
                        policy.candidate_map.split_candidate_count
                        if policy is not None else 0
                    ),
                    "selected_candidate_id": selected_id,
                    "initial_selection_correct": selection_correct,
                    "replan_count": replan_count,
                    "recovery_count": recovery_count,
                    "stalled_count": stalled_count,
                    "obstacle_recovery_count": obstacle_recovery_count,
                    "handoff_guard_checks": (
                        policy.handoff_guard_checks if policy is not None else 0
                    ),
                    "handoff_guard_rejections": (
                        policy.handoff_guard_rejections
                        if policy is not None else 0
                    ),
                    "handoff_visual_confirmations": (
                        policy.handoff_visual_confirmations
                        if policy is not None else 0
                    ),
                    "handoff_raycast_confirmations": (
                        policy.handoff_raycast_confirmations
                        if policy is not None else 0
                    ),
                    "handoff_spatial_rejections": (
                        policy.handoff_spatial_rejections
                        if policy is not None else 0
                    ),
                    "contact_owner_lock_steps": (
                        policy.contact_owner_lock_steps
                        if policy is not None else 0
                    ),
                    "contact_owner_mismatches": (
                        policy.contact_owner_mismatches
                        if policy is not None else 0
                    ),
                    "terrain_route_recovery_attempts": (
                        policy.terrain_route_recovery_attempts
                        if policy is not None else 0
                    ),
                    "terrain_route_recovery_steps": (
                        policy.terrain_route_recovery_steps
                        if policy is not None else 0
                    ),
                    "terrain_route_recovery_successes": (
                        policy.terrain_route_recovery_successes
                        if policy is not None else 0
                    ),
                    "terrain_route_recovery_failures": (
                        policy.terrain_route_recovery_failures
                        if policy is not None else 0
                    ),
                    "repeated_physical_region_route_rejections": (
                        policy.repeated_physical_region_route_rejections
                        if policy is not None else 0
                    ),
                    "coordinate_climb_bursts": contact_counters.get(
                        "coordinate_climb_bursts", 0
                    ),
                    "coordinate_climb_successes": contact_counters.get(
                        "coordinate_climb_successes", 0
                    ),
                    "coordinate_climb_failures": contact_counters.get(
                        "coordinate_climb_failures", 0
                    ),
                    "coordinate_climb_steps": contact_counters.get(
                        "coordinate_climb_steps", 0
                    ),
                    "rescan_success_loop_resets": contact_counters.get(
                        "rescan_success_loop_resets", 0
                    ),
                    "drop_elevated_pickup_attempts": contact_counters.get(
                        "drop_elevated_pickup_attempts", 0
                    ),
                    "drop_elevated_jump_steps": contact_counters.get(
                        "drop_elevated_jump_steps", 0
                    ),
                    "min_remaining_steps_at_block_disappearance": (
                        min(remaining_at_disappearance)
                        if remaining_at_disappearance else ""
                    ),
                    "min_remaining_steps_at_drop_recovery_start": (
                        min(remaining_at_drop_start)
                        if remaining_at_drop_start else ""
                    ),
                    "max_consecutive_blocked_drop_waypoints": (
                        max_consecutive_blocked
                    ),
                    "handoff_relocalization_scans": (
                        policy.handoff_relocalization_scans
                        if policy is not None else 0
                    ),
                    "handoff_relocalization_skipped_late": (
                        policy.handoff_relocalization_skipped_late
                        if policy is not None else 0
                    ),
                    "suppressed_contact_position_updates": (
                        policy.suppressed_contact_position_updates
                        if policy is not None else 0
                    ),
                    "contact_attempts": contact_counters.get("attempts", 0),
                    "contact_total_steps": contact_counters.get("steps", 0),
                    "contact_attack_steps": contact_counters.get("attack_steps", 0),
                    "contact_attack_rounds": contact_counters.get("attack_rounds", 0),
                    "contact_backoffs": contact_counters.get("backoffs", 0),
                    "contact_orbits": contact_counters.get("orbits", 0),
                    "trunk_reacquires": contact_counters.get("trunk_reacquires", 0),
                    "occlusion_clears": contact_counters.get(
                        "occlusion_clears", 0
                    ),
                    "occlusion_clear_steps": contact_counters.get(
                        "occlusion_clear_steps", 0
                    ),
                    "raycast_log_actions": contact_counters.get(
                        "raycast_log_actions", 0
                    ),
                    "raycast_leaf_actions": contact_counters.get(
                        "raycast_leaf_actions", 0
                    ),
                    "raycast_in_range_attack_steps": contact_counters.get(
                        "raycast_in_range_attack_steps", 0
                    ),
                    "block_disappearance_count": contact_counters.get(
                        "block_disappearances", 0
                    ),
                    "drop_recovery_attempts": contact_counters.get(
                        "drop_recovery_attempts", 0
                    ),
                    "drop_recovery_steps": contact_counters.get(
                        "drop_recovery_steps", 0
                    ),
                    "drop_waypoints_reached": contact_counters.get(
                        "drop_waypoints_reached", 0
                    ),
                    "drop_blocked_waypoints": contact_counters.get(
                        "drop_blocked_waypoints", 0
                    ),
                    "drop_block_center_normalizations": contact_counters.get(
                        "drop_block_center_normalizations", 0
                    ),
                    "pickup_after_disappearance": contact_counters.get(
                        "pickup_after_disappearance", 0
                    ),
                    "same_trunk_reacquire_count": contact_counters.get(
                        "same_trunk_reacquires", 0
                    ),
                    "coordinate_target_count": final_contact.get(
                        "coordinate_target_count", 0
                    ) if policy is not None else 0,
                    "coordinate_target_selections": contact_counters.get(
                        "coordinate_target_selections", 0
                    ),
                    "coordinate_aim_steps": contact_counters.get(
                        "coordinate_aim_steps", 0
                    ),
                    "coordinate_aim_fallbacks": contact_counters.get(
                        "coordinate_aim_fallbacks", 0
                    ),
                    "coordinate_leaf_clears": contact_counters.get(
                        "coordinate_leaf_clears", 0
                    ),
                    "coordinate_attacks": contact_counters.get(
                        "coordinate_attacks", 0
                    ),
                    "coordinate_progress_stalls": contact_counters.get(
                        "coordinate_progress_stalls", 0
                    ),
                    "coordinate_recoveries": contact_counters.get(
                        "coordinate_recoveries", 0
                    ),
                    "coordinate_post_recovery_verifications": contact_counters.get(
                        "coordinate_post_recovery_verifications", 0
                    ),
                    "coordinate_post_recovery_progress": contact_counters.get(
                        "coordinate_post_recovery_progress", 0
                    ),
                    "coordinate_post_recovery_no_progress": contact_counters.get(
                        "coordinate_post_recovery_no_progress", 0
                    ),
                    "coordinate_target_switches": contact_counters.get(
                        "coordinate_target_switches", 0
                    ),
                    "coordinate_all_targets_cooldown": contact_counters.get(
                        "coordinate_all_targets_cooldown", 0
                    ),
                    "coordinate_no_eligible_targets": contact_counters.get(
                        "coordinate_no_eligible_targets", 0
                    ),
                    "exact_log_rescan_attempts": contact_counters.get(
                        "exact_log_rescan_attempts", 0
                    ),
                    "exact_log_rescan_steps": contact_counters.get(
                        "exact_log_rescan_steps", 0
                    ),
                    "exact_log_rescan_successes": contact_counters.get(
                        "exact_log_rescan_successes", 0
                    ),
                    "exact_log_rescan_failures": contact_counters.get(
                        "exact_log_rescan_failures", 0
                    ),
                    "spatial_exact_log_rescan_rejections": contact_counters.get(
                        "spatial_exact_log_rescan_rejections", 0
                    ),
                    "failed_exact_log_scan_regions": len(
                        final_contact.get("exact_log_rescan", {}).get(
                            "failed_regions", []
                        )
                    ),
                    "prevented_unconfirmed_attacks": contact_counters.get(
                        "prevented_unconfirmed_attacks", 0
                    ),
                    "center_adjust_loop_cycles": contact_counters.get(
                        "center_adjust_loop_cycles", 0
                    ),
                    "center_find_loop_cycles": contact_counters.get(
                        "center_find_loop_cycles", 0
                    ),
                    "attack_out_of_range_loops": contact_counters.get(
                        "attack_out_of_range_loops", 0
                    ),
                    "contact_result": contact_result,
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
                write_rows(trunk_target_output, trunk_target_table)
                write_rows(trunk_target_event_output, trunk_target_event_table)
                write_rows(drop_waypoint_output, drop_waypoint_table)
                write_rows(route_recovery_output, route_recovery_table)
                write_rows(episode_trace_dir / "trace.csv", trace_rows)
                write_rows(
                    episode_trace_dir / "contact_transitions.csv", contact_rows
                )
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
        "contact_profile": args.contact_profile,
        "deployment_inputs": (
            [
                "pov",
                "self_pose",
                "biome",
                "privileged_crosshair_raycast",
                "privileged_raycast_log_xyz_memory",
                "internal_candidate_memory",
            ]
            if args.sensor_profile == SENSOR_PROFILE_RAYCAST
            else (
                ["pov", "self_pose", "biome", "internal_candidate_memory"]
                if args.sensor_profile == SENSOR_PROFILE_F3
                else ["pov", "commanded_camera_delta", "internal_memory"]
            )
        ),
        "privileged_deployment_profile": bool(
            args.sensor_profile == SENSOR_PROFILE_RAYCAST
        ),
        "privileged_raycast_used_for_local_target_selection": bool(
            args.sensor_profile == SENSOR_PROFILE_RAYCAST
            and args.contact_profile
            in (
                CONTACT_PROFILE_COORDINATE_AIM,
                CONTACT_PROFILE_COORDINATE_RECOVERY,
                CONTACT_PROFILE_COORDINATE_RECOVERY_V9_1,
                CONTACT_PROFILE_COORDINATE_CONTACT_GUARD_V9_2,
                CONTACT_PROFILE_CANDIDATE_HANDOFF_GUARD_V9_3,
                CONTACT_PROFILE_CONTACT_DROP_COMPLETION_V9_4,
                CONTACT_PROFILE_CONTACT_OWNERSHIP_SPATIAL_GUARD_V9_5,
                CONTACT_PROFILE_TERRAIN_ROUTE_DROP_COMPLETION_V9_6,
                CONTACT_PROFILE_TRACE_GUIDED_DROP_RECOVERY_V9_7,
                CONTACT_PROFILE_RAYCAST_OWNED_HANDOFF_V9_8,
                CONTACT_PROFILE_EARLY_ROUTE_RECOVERY_V9_9,
                CONTACT_PROFILE_COORDINATE_TARGET_PREEMPTION_V9_10,
                CONTACT_PROFILE_EMERGENCY_TARGET_PREEMPTION_V9_11,
            )
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
                    "total_obstacle_recoveries": sum(
                        row.get("obstacle_recovery_count", 0)
                        for row in mode_rows
                    ),
                    "total_handoff_guard_checks": sum(
                        row.get("handoff_guard_checks", 0)
                        for row in mode_rows
                    ),
                    "total_handoff_guard_rejections": sum(
                        row.get("handoff_guard_rejections", 0)
                        for row in mode_rows
                    ),
                    "total_handoff_visual_confirmations": sum(
                        row.get("handoff_visual_confirmations", 0)
                        for row in mode_rows
                    ),
                    "total_handoff_raycast_confirmations": sum(
                        row.get("handoff_raycast_confirmations", 0)
                        for row in mode_rows
                    ),
                    "total_handoff_spatial_rejections": sum(
                        row.get("handoff_spatial_rejections", 0)
                        for row in mode_rows
                    ),
                    "total_contact_owner_lock_steps": sum(
                        row.get("contact_owner_lock_steps", 0)
                        for row in mode_rows
                    ),
                    "total_contact_owner_mismatches": sum(
                        row.get("contact_owner_mismatches", 0)
                        for row in mode_rows
                    ),
                    "total_handoff_relocalization_scans": sum(
                        row.get("handoff_relocalization_scans", 0)
                        for row in mode_rows
                    ),
                    "total_handoff_relocalization_skipped_late": sum(
                        row.get("handoff_relocalization_skipped_late", 0)
                        for row in mode_rows
                    ),
                    "total_suppressed_contact_position_updates": sum(
                        row.get("suppressed_contact_position_updates", 0)
                        for row in mode_rows
                    ),
                    "total_contact_attempts": sum(
                        row.get("contact_attempts", 0) for row in mode_rows
                    ),
                    "total_contact_steps": sum(
                        row.get("contact_total_steps", 0) for row in mode_rows
                    ),
                    "max_step_failures": sum(
                        (not row["success"]) and row["steps"] >= args.max_steps
                        for row in mode_rows
                    ),
                    "total_split_candidates": sum(
                        row.get("split_candidate_count", 0) for row in mode_rows
                    ),
                    "total_contact_attack_steps": sum(
                        row.get("contact_attack_steps", 0) for row in mode_rows
                    ),
                    "total_contact_orbits": sum(
                        row.get("contact_orbits", 0) for row in mode_rows
                    ),
                    "total_contact_backoffs": sum(
                        row.get("contact_backoffs", 0) for row in mode_rows
                    ),
                    "total_trunk_reacquires": sum(
                        row.get("trunk_reacquires", 0) for row in mode_rows
                    ),
                    "total_occlusion_clears": sum(
                        row.get("occlusion_clears", 0) for row in mode_rows
                    ),
                    "total_occlusion_clear_steps": sum(
                        row.get("occlusion_clear_steps", 0)
                        for row in mode_rows
                    ),
                    "total_raycast_log_actions": sum(
                        row.get("raycast_log_actions", 0)
                        for row in mode_rows
                    ),
                    "total_raycast_leaf_actions": sum(
                        row.get("raycast_leaf_actions", 0)
                        for row in mode_rows
                    ),
                    "episodes_with_in_range_raycast_contact": sum(
                        row.get("raycast_in_range_attack_steps", 0) > 0
                        for row in mode_rows
                    ),
                    "successful_after_in_range_raycast_contact": sum(
                        row.get("raycast_in_range_attack_steps", 0) > 0
                        and row.get("success", False)
                        for row in mode_rows
                    ),
                    "total_block_disappearances": sum(
                        row.get("block_disappearance_count", 0)
                        for row in mode_rows
                    ),
                    "total_drop_recovery_attempts": sum(
                        row.get("drop_recovery_attempts", 0)
                        for row in mode_rows
                    ),
                    "total_drop_recovery_steps": sum(
                        row.get("drop_recovery_steps", 0)
                        for row in mode_rows
                    ),
                    "total_drop_waypoints_reached": sum(
                        row.get("drop_waypoints_reached", 0)
                        for row in mode_rows
                    ),
                    "total_drop_blocked_waypoints": sum(
                        row.get("drop_blocked_waypoints", 0)
                        for row in mode_rows
                    ),
                    "total_drop_block_center_normalizations": sum(
                        row.get("drop_block_center_normalizations", 0)
                        for row in mode_rows
                    ),
                    "total_pickups_after_disappearance": sum(
                        row.get("pickup_after_disappearance", 0)
                        for row in mode_rows
                    ),
                    "total_same_trunk_reacquires": sum(
                        row.get("same_trunk_reacquire_count", 0)
                        for row in mode_rows
                    ),
                    "episodes_with_coordinate_targets": sum(
                        row.get("coordinate_target_count", 0) > 0
                        for row in mode_rows
                    ),
                    "total_coordinate_targets": sum(
                        row.get("coordinate_target_count", 0)
                        for row in mode_rows
                    ),
                    "total_coordinate_target_selections": sum(
                        row.get("coordinate_target_selections", 0)
                        for row in mode_rows
                    ),
                    "total_coordinate_aim_steps": sum(
                        row.get("coordinate_aim_steps", 0)
                        for row in mode_rows
                    ),
                    "total_coordinate_aim_fallbacks": sum(
                        row.get("coordinate_aim_fallbacks", 0)
                        for row in mode_rows
                    ),
                    "total_coordinate_leaf_clears": sum(
                        row.get("coordinate_leaf_clears", 0)
                        for row in mode_rows
                    ),
                    "total_coordinate_attacks": sum(
                        row.get("coordinate_attacks", 0)
                        for row in mode_rows
                    ),
                    "successful_after_coordinate_recovery": sum(
                        row.get("success", False)
                        and row.get("coordinate_recoveries", 0) > 0
                        for row in mode_rows
                    ),
                    "successful_after_coordinate_target_switch": sum(
                        row.get("success", False)
                        and row.get("coordinate_target_switches", 0) > 0
                        for row in mode_rows
                    ),
                    "total_coordinate_progress_stalls": sum(
                        row.get("coordinate_progress_stalls", 0)
                        for row in mode_rows
                    ),
                    "total_coordinate_recoveries": sum(
                        row.get("coordinate_recoveries", 0)
                        for row in mode_rows
                    ),
                    "total_coordinate_post_recovery_verifications": sum(
                        row.get("coordinate_post_recovery_verifications", 0)
                        for row in mode_rows
                    ),
                    "total_coordinate_post_recovery_progress": sum(
                        row.get("coordinate_post_recovery_progress", 0)
                        for row in mode_rows
                    ),
                    "total_coordinate_post_recovery_no_progress": sum(
                        row.get("coordinate_post_recovery_no_progress", 0)
                        for row in mode_rows
                    ),
                    "total_coordinate_target_switches": sum(
                        row.get("coordinate_target_switches", 0)
                        for row in mode_rows
                    ),
                    "total_coordinate_all_targets_cooldown": sum(
                        row.get("coordinate_all_targets_cooldown", 0)
                        for row in mode_rows
                    ),
                    "total_coordinate_no_eligible_targets": sum(
                        row.get("coordinate_no_eligible_targets", 0)
                        for row in mode_rows
                    ),
                    "total_exact_log_rescan_attempts": sum(
                        row.get("exact_log_rescan_attempts", 0)
                        for row in mode_rows
                    ),
                    "total_exact_log_rescan_steps": sum(
                        row.get("exact_log_rescan_steps", 0)
                        for row in mode_rows
                    ),
                    "total_exact_log_rescan_successes": sum(
                        row.get("exact_log_rescan_successes", 0)
                        for row in mode_rows
                    ),
                    "total_exact_log_rescan_failures": sum(
                        row.get("exact_log_rescan_failures", 0)
                        for row in mode_rows
                    ),
                    "total_spatial_exact_log_rescan_rejections": sum(
                        row.get("spatial_exact_log_rescan_rejections", 0)
                        for row in mode_rows
                    ),
                    "total_failed_exact_log_scan_regions": sum(
                        row.get("failed_exact_log_scan_regions", 0)
                        for row in mode_rows
                    ),
                    "total_prevented_unconfirmed_attacks": sum(
                        row.get("prevented_unconfirmed_attacks", 0)
                        for row in mode_rows
                    ),
                    "total_center_adjust_loop_cycles": sum(
                        row.get("center_adjust_loop_cycles", 0)
                        for row in mode_rows
                    ),
                    "total_center_find_loop_cycles": sum(
                        row.get("center_find_loop_cycles", 0)
                        for row in mode_rows
                    ),
                    "total_attack_out_of_range_loops": sum(
                        row.get("attack_out_of_range_loops", 0)
                        for row in mode_rows
                    ),
                    "total_terrain_route_recovery_attempts": sum(
                        row.get("terrain_route_recovery_attempts", 0)
                        for row in mode_rows
                    ),
                    "total_terrain_route_recovery_steps": sum(
                        row.get("terrain_route_recovery_steps", 0)
                        for row in mode_rows
                    ),
                    "total_terrain_route_recovery_successes": sum(
                        row.get("terrain_route_recovery_successes", 0)
                        for row in mode_rows
                    ),
                    "total_terrain_route_recovery_failures": sum(
                        row.get("terrain_route_recovery_failures", 0)
                        for row in mode_rows
                    ),
                    "total_repeated_physical_region_route_rejections": sum(
                        row.get(
                            "repeated_physical_region_route_rejections", 0
                        )
                        for row in mode_rows
                    ),
                    "total_coordinate_climb_bursts": sum(
                        row.get("coordinate_climb_bursts", 0)
                        for row in mode_rows
                    ),
                    "total_coordinate_climb_successes": sum(
                        row.get("coordinate_climb_successes", 0)
                        for row in mode_rows
                    ),
                    "total_coordinate_climb_failures": sum(
                        row.get("coordinate_climb_failures", 0)
                        for row in mode_rows
                    ),
                    "total_coordinate_climb_steps": sum(
                        row.get("coordinate_climb_steps", 0)
                        for row in mode_rows
                    ),
                    "total_rescan_success_loop_resets": sum(
                        row.get("rescan_success_loop_resets", 0)
                        for row in mode_rows
                    ),
                    "total_drop_elevated_pickup_attempts": sum(
                        row.get("drop_elevated_pickup_attempts", 0)
                        for row in mode_rows
                    ),
                    "total_drop_elevated_jump_steps": sum(
                        row.get("drop_elevated_jump_steps", 0)
                        for row in mode_rows
                    ),
                    "remaining_steps_at_block_disappearance": [
                        row.get("min_remaining_steps_at_block_disappearance", "")
                        for row in mode_rows
                        if row.get("min_remaining_steps_at_block_disappearance", "")
                        != ""
                    ],
                    "remaining_steps_at_drop_recovery_start": [
                        row.get("min_remaining_steps_at_drop_recovery_start", "")
                        for row in mode_rows
                        if row.get("min_remaining_steps_at_drop_recovery_start", "")
                        != ""
                    ],
                    "max_consecutive_blocked_drop_waypoints": max(
                        (
                            row.get(
                                "max_consecutive_blocked_drop_waypoints", 0
                            )
                            for row in mode_rows
                        ),
                        default=0,
                    ),
                    "drop_waypoint_end_reason_counts": dict(
                        Counter(
                            str(row.get("end_reason", ""))
                            for row in drop_waypoint_table
                            if row.get("mode") == mode
                        )
                    ),
                    "coarse_route_failure_count": sum(
                        (not row["success"])
                        and row.get("contact_attempts", 0) == 0
                        and row.get("raycast_in_range_attack_steps", 0) == 0
                        for row in mode_rows
                    ),
                    "local_contact_failure_count": sum(
                        (not row["success"])
                        and row.get("contact_attempts", 0) > 0
                        and row.get("raycast_in_range_attack_steps", 0) == 0
                        for row in mode_rows
                    ),
                    "post_disappearance_pickup_failure_count": sum(
                        (not row["success"])
                        and row.get("block_disappearance_count", 0) > 0
                        and row.get("pickup_after_disappearance", 0) == 0
                        for row in mode_rows
                    ),
                    "trunk_contact_enabled": bool(
                        args.environment == "natural"
                        and not args.disable_trunk_contact
                    ),
                }
            )
            in_range_episodes = mode_summary[
                "episodes_with_in_range_raycast_contact"
            ]
            disappearances = mode_summary["total_block_disappearances"]
            mode_summary["success_given_in_range_raycast_contact"] = (
                None
                if not in_range_episodes
                else mode_summary[
                    "successful_after_in_range_raycast_contact"
                ]
                / in_range_episodes
            )
            mode_summary["pickup_given_block_disappearance"] = (
                None
                if not disappearances
                else mode_summary["total_pickups_after_disappearance"]
                / disappearances
            )
        summary[mode] = mode_summary
    if (
        args.environment == "natural"
        and args.contact_profile == CONTACT_PROFILE_COORDINATE_RECOVERY_V9_1
        and "candidate" in summary
    ):
        candidate_summary = summary["candidate"]
        summary["natural_v9_1_training_gate"] = {
            "at_least_9_of_10": bool(
                args.episodes >= 10 and candidate_summary["successes"] >= 9
            ),
            "no_max_step_failures": bool(
                candidate_summary["max_step_failures"] == 0
            ),
            "successful_after_coordinate_recovery": bool(
                candidate_summary["successful_after_coordinate_recovery"] > 0
            ),
            "oracle_or_log_grid_used_for_actions_or_scoring": False,
        }
        summary["natural_v9_1_training_gate"]["all_conditions_met"] = all(
            (
                summary["natural_v9_1_training_gate"]["at_least_9_of_10"],
                summary["natural_v9_1_training_gate"]["no_max_step_failures"],
                summary["natural_v9_1_training_gate"][
                    "successful_after_coordinate_recovery"
                ],
                not summary["natural_v9_1_training_gate"][
                    "oracle_or_log_grid_used_for_actions_or_scoring"
                ],
            )
        )
    if (
        args.environment == "natural"
        and args.contact_profile == CONTACT_PROFILE_COORDINATE_CONTACT_GUARD_V9_2
        and "candidate" in summary
    ):
        candidate_summary = summary["candidate"]
        summary["natural_v9_2_training_gate"] = {
            "at_least_9_of_10": bool(
                args.episodes >= 10 and candidate_summary["successes"] >= 9
            ),
            "no_max_step_failures": bool(
                candidate_summary["max_step_failures"] == 0
            ),
            "successful_after_coordinate_recovery": bool(
                candidate_summary["successful_after_coordinate_recovery"] > 0
            ),
            "oracle_or_log_grid_used_for_actions_or_scoring": False,
        }
        gate = summary["natural_v9_2_training_gate"]
        gate["all_conditions_met"] = all(
            (
                gate["at_least_9_of_10"],
                gate["no_max_step_failures"],
                gate["successful_after_coordinate_recovery"],
                not gate["oracle_or_log_grid_used_for_actions_or_scoring"],
            )
        )
    if (
        args.environment == "natural"
        and args.contact_profile == CONTACT_PROFILE_CANDIDATE_HANDOFF_GUARD_V9_3
        and "candidate" in summary
    ):
        candidate_summary = summary["candidate"]
        summary["natural_v9_3_training_gate"] = {
            "at_least_9_of_10": bool(
                args.episodes >= 10 and candidate_summary["successes"] >= 9
            ),
            "no_max_step_failures": bool(
                candidate_summary["max_step_failures"] == 0
            ),
            "successful_after_coordinate_recovery": bool(
                candidate_summary["successful_after_coordinate_recovery"] > 0
            ),
            "oracle_or_log_grid_used_for_actions_or_scoring": False,
        }
        gate = summary["natural_v9_3_training_gate"]
        gate["all_conditions_met"] = all(
            (
                gate["at_least_9_of_10"],
                gate["no_max_step_failures"],
                gate["successful_after_coordinate_recovery"],
                not gate["oracle_or_log_grid_used_for_actions_or_scoring"],
            )
        )
    if (
        args.environment == "natural"
        and args.contact_profile == CONTACT_PROFILE_CONTACT_DROP_COMPLETION_V9_4
        and "candidate" in summary
    ):
        candidate_summary = summary["candidate"]
        summary["natural_v9_4_training_gate"] = {
            "at_least_9_of_10": bool(
                args.episodes >= 10 and candidate_summary["successes"] >= 9
            ),
            "no_max_step_failures": bool(
                candidate_summary["max_step_failures"] == 0
            ),
            "successful_after_coordinate_recovery": bool(
                candidate_summary["successful_after_coordinate_recovery"] > 0
            ),
            "oracle_or_log_grid_used_for_actions_or_scoring": False,
        }
        gate = summary["natural_v9_4_training_gate"]
        gate["all_conditions_met"] = all(
            (
                gate["at_least_9_of_10"],
                gate["no_max_step_failures"],
                gate["successful_after_coordinate_recovery"],
                not gate["oracle_or_log_grid_used_for_actions_or_scoring"],
            )
        )
    if (
        args.environment == "natural"
        and args.contact_profile
        == CONTACT_PROFILE_CONTACT_OWNERSHIP_SPATIAL_GUARD_V9_5
        and "candidate" in summary
    ):
        candidate_summary = summary["candidate"]
        summary["natural_v9_5_training_gate"] = {
            "at_least_18_of_20": bool(
                args.episodes >= 20 and candidate_summary["successes"] >= 18
            ),
            "at_most_2_max_step_failures": bool(
                candidate_summary["max_step_failures"] <= 2
            ),
            "successful_after_coordinate_recovery": bool(
                candidate_summary["successful_after_coordinate_recovery"] > 0
            ),
            "oracle_or_log_grid_used_for_actions_or_scoring": False,
        }
        gate = summary["natural_v9_5_training_gate"]
        gate["all_conditions_met"] = all(
            (
                gate["at_least_18_of_20"],
                gate["at_most_2_max_step_failures"],
                gate["successful_after_coordinate_recovery"],
                not gate["oracle_or_log_grid_used_for_actions_or_scoring"],
            )
        )
    if (
        args.environment == "natural"
        and args.contact_profile
        in (
            CONTACT_PROFILE_TERRAIN_ROUTE_DROP_COMPLETION_V9_6,
            CONTACT_PROFILE_TRACE_GUIDED_DROP_RECOVERY_V9_7,
            CONTACT_PROFILE_RAYCAST_OWNED_HANDOFF_V9_8,
            CONTACT_PROFILE_EARLY_ROUTE_RECOVERY_V9_9,
            CONTACT_PROFILE_COORDINATE_TARGET_PREEMPTION_V9_10,
            CONTACT_PROFILE_EMERGENCY_TARGET_PREEMPTION_V9_11,
        )
        and "candidate" in summary
    ):
        candidate_summary = summary["candidate"]
        gate_name = (
            "natural_v9_11_training_gate"
            if args.contact_profile
            == CONTACT_PROFILE_EMERGENCY_TARGET_PREEMPTION_V9_11
            else
            "natural_v9_10_training_gate"
            if args.contact_profile
            == CONTACT_PROFILE_COORDINATE_TARGET_PREEMPTION_V9_10
            else
            "natural_v9_9_training_gate"
            if args.contact_profile
            == CONTACT_PROFILE_EARLY_ROUTE_RECOVERY_V9_9
            else
            "natural_v9_8_training_gate"
            if args.contact_profile
            == CONTACT_PROFILE_RAYCAST_OWNED_HANDOFF_V9_8
            else "natural_v9_7_training_gate"
            if args.contact_profile
            == CONTACT_PROFILE_TRACE_GUIDED_DROP_RECOVERY_V9_7
            else "natural_v9_6_training_gate"
        )
        summary[gate_name] = {
            "at_least_18_of_20": bool(
                args.episodes >= 20 and candidate_summary["successes"] >= 18
            ),
            "at_most_2_max_step_failures": bool(
                candidate_summary["max_step_failures"] <= 2
            ),
            "successful_after_coordinate_recovery": bool(
                candidate_summary["successful_after_coordinate_recovery"] > 0
            ),
            "contact_owner_mismatches_zero": bool(
                candidate_summary["total_contact_owner_mismatches"] == 0
            ),
            "oracle_or_log_grid_used_for_actions_or_scoring": False,
        }
        gate = summary[gate_name]
        gate["all_conditions_met"] = all(
            (
                gate["at_least_18_of_20"],
                gate["at_most_2_max_step_failures"],
                gate["successful_after_coordinate_recovery"],
                gate["contact_owner_mismatches_zero"],
                not gate["oracle_or_log_grid_used_for_actions_or_scoring"],
            )
        )
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
