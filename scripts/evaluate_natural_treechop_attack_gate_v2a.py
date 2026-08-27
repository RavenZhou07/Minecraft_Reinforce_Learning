"""Minecraft shadow/autonomous evaluation for BC v2a/v2b attack gates."""

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import psutil
import cv2

from mc_rl.natural_attack_gate_bc import NaturalAttackGatePolicy, attack_gate_labels
from mc_rl.natural_attack_gate_runner import NaturalAttackGateRunner
from mc_rl.natural_contact_bc import StudentContactAgent
from mc_rl.resource_adapters import TreeResourceAdapter
from mc_rl.search_policy import CandidateSearchPolicy, SearchConfig
from mc_rl.telemetry_treechop_env import make_telemetry_treechop_env
from mc_rl.trunk_contact import CONTACT_PROFILE_TERRAIN_ROUTE_DROP_COMPLETION_V9_6
from scripts.train_natural_treechop_attack_gate_v2a import binary_metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=17100)
    parser.add_argument(
        "--seeds",
        default=None,
        help="Optional comma-separated exact seed list; overrides episodes/seed.",
    )
    parser.add_argument("--mode", choices=("shadow", "autonomous"), default="shadow")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/natural_treechop_attack_gate_bc_v2a.npz",
    )
    parser.add_argument(
        "--output",
        default="logs/find_tree/natural_treechop_attack_gate_bc_v2a_shadow_17100_20.csv",
    )
    parser.add_argument(
        "--contact-profile",
        default=CONTACT_PROFILE_TERRAIN_ROUTE_DROP_COMPLETION_V9_6,
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--attack-confirmation-frames",
        type=int,
        default=None,
        help="Override the checkpoint's causal ATTACK confirmation length.",
    )
    parser.add_argument("--diagnostics-output", default=None)
    parser.add_argument("--diagnostics-dataset", default=None)
    parser.add_argument("--contact-trace-output", default=None)
    parser.add_argument("--diagnostic-image-dir", default=None)
    parser.add_argument("--diagnostic-image-prefix", default="gate_diag")
    parser.add_argument(
        "--diagnostic-margin",
        type=float,
        default=0.05,
        help="Also retain predictions this close to the decision threshold.",
    )
    return parser.parse_args()


def parse_seed_list(value: str) -> List[int]:
    seeds = [int(item.strip()) for item in str(value).split(",") if item.strip()]
    if not seeds:
        raise ValueError("seeds must contain at least one integer")
    if len(seeds) != len(set(seeds)):
        raise ValueError("seeds must not contain duplicates")
    return seeds


def atomic_write_rows(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def atomic_write_npz(path: Path, arrays: Dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def atomic_write_rgb_png(path: Path, frame: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", np.asarray(frame)[..., ::-1])
    if not ok:
        raise RuntimeError("failed to encode diagnostic frame: {}".format(path))
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(encoded.tobytes())
    temporary.replace(path)


def classify_gate_outcome(teacher_label: int, gate_decision: int) -> str:
    if teacher_label == 1 and gate_decision == 1:
        return "true_positive"
    if teacher_label == 0 and gate_decision == 1:
        return "false_positive"
    if teacher_label == 0 and gate_decision == 0:
        return "true_negative"
    return "false_negative"


def diagnostic_sample_selected(
    outcome: str, probability: float, threshold: float, margin: float
) -> bool:
    return outcome in ("false_positive", "false_negative") or abs(
        float(probability) - float(threshold)
    ) <= float(margin)


def diagnostic_arrays(samples: Dict[str, List[Any]], metadata: Dict[str, Any]):
    arrays = {
        key: (
            np.asarray(value, dtype=np.uint8)
            if key == "pov"
            else np.asarray(value)
        )
        for key, value in samples.items()
    }
    for key, value in metadata.items():
        arrays[key] = np.asarray(value)
    return arrays


def compact_contact_trace(
    diagnostics: Dict[str, Any], counters: Dict[str, Any]
) -> Dict[str, Any]:
    transitions = diagnostics.get("transition_records", [])
    drop = diagnostics.get("drop_recovery", {}) or {}
    return {
        "contact_state": diagnostics.get("state", ""),
        "contact_result": diagnostics.get("result"),
        "contact_engaged": diagnostics.get("engaged", False),
        "contact_active": diagnostics.get("active", False),
        "contact_attempt_id": diagnostics.get("attempt_id", 0),
        "candidate_id": diagnostics.get("candidate_id"),
        "attempt_step": diagnostics.get("attempt_step", 0),
        "crosshair_trunk_fraction": diagnostics.get("crosshair_trunk_fraction"),
        "trunk_area_px": diagnostics.get("trunk_area_px"),
        "raycast_is_log": diagnostics.get("raycast_is_log"),
        "raycast_in_range": diagnostics.get("raycast_in_range"),
        "raycast_distance": diagnostics.get("raycast_distance"),
        "transition_count": len(transitions),
        "last_transition_reason": (
            transitions[-1].get("reason", "") if transitions else ""
        ),
        "attack_steps": counters.get("attack_steps", 0),
        "raycast_in_range_attack_steps": counters.get(
            "raycast_in_range_attack_steps", 0
        ),
        "prevented_unconfirmed_attacks": counters.get(
            "prevented_unconfirmed_attacks", 0
        ),
        "block_disappearances": counters.get("block_disappearances", 0),
        "drop_recovery_attempts": counters.get("drop_recovery_attempts", 0),
        "drop_recovery_steps": counters.get("drop_recovery_steps", 0),
        "pickup_after_disappearance": counters.get(
            "pickup_after_disappearance", 0
        ),
        "coordinate_recoveries": counters.get("coordinate_recoveries", 0),
        "coordinate_target_preemptions": counters.get(
            "coordinate_target_preemptions", 0
        ),
        "coordinate_emergency_preemptions": counters.get(
            "coordinate_emergency_preemptions", 0
        ),
        "exact_log_rescan_attempts": counters.get(
            "exact_log_rescan_attempts", 0
        ),
        "drop_phase": drop.get("phase", ""),
        "drop_steps": drop.get("steps", 0),
        "drop_distance": drop.get("distance"),
        "drop_progress": drop.get("progress"),
    }


def compact_search_trace(diagnostics: Dict[str, Any]) -> Dict[str, Any]:
    transitions = diagnostics.get("transitions", [])
    handoff = diagnostics.get("handoff", {}) or {}
    return {
        "search_state": diagnostics.get("state", ""),
        "search_remaining_steps": diagnostics.get("remaining_steps"),
        "search_selected_candidate_id": diagnostics.get(
            "selected_candidate_id"
        ),
        "search_candidate_count": diagnostics.get("candidate_count", 0),
        "search_route_distance": diagnostics.get("route_distance"),
        "search_route_yaw_error": diagnostics.get("route_yaw_error"),
        "search_contact_region_limit": diagnostics.get(
            "contact_region_limit"
        ),
        "search_scan_cycles": diagnostics.get("scan_cycles", 0),
        "exact_log_early_scan_exits": diagnostics.get(
            "exact_log_early_scan_exits", 0
        ),
        "dynamic_exact_route_updates": diagnostics.get(
            "dynamic_exact_route_updates", 0
        ),
        "search_replan_count": diagnostics.get("replan_count", 0),
        "search_recovery_count": diagnostics.get("recovery_count", 0),
        "search_stalled_count": diagnostics.get("stalled_count", 0),
        "search_obstacle_recoveries": diagnostics.get(
            "obstacle_recovery_count", 0
        ),
        "search_transition_count": len(transitions),
        "search_last_transition_reason": (
            transitions[-1].get("reason", "") if transitions else ""
        ),
        "handoff_checks": handoff.get("checks", 0),
        "handoff_rejections": handoff.get("rejections", 0),
        "handoff_raycast_memory_confirmations": handoff.get(
            "raycast_memory_confirmations", 0
        ),
        "raycast_memory_route_selections": handoff.get(
            "raycast_memory_route_selections", 0
        ),
        "handoff_relocalization_scans": handoff.get(
            "relocalization_scans", 0
        ),
        "handoff_relocalization_skipped_late": handoff.get(
            "relocalization_skipped_late", 0
        ),
        "terrain_route_recovery_attempts": handoff.get(
            "terrain_route_recovery_attempts", 0
        ),
        "terrain_route_recovery_steps": handoff.get(
            "terrain_route_recovery_steps", 0
        ),
    }


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


def main():
    args = parse_args()
    if args.episodes <= 0 or args.max_steps <= 0:
        raise ValueError("episodes and max-steps must be positive")
    if args.attack_confirmation_frames is not None and args.attack_confirmation_frames < 1:
        raise ValueError("attack-confirmation-frames must be at least one")
    if args.diagnostic_margin < 0.0:
        raise ValueError("diagnostic-margin must be non-negative")
    output = Path(args.output)
    summary_output = output.with_suffix(".summary.json")
    diagnostics_output = (
        Path(args.diagnostics_output) if args.diagnostics_output else None
    )
    diagnostics_dataset = (
        Path(args.diagnostics_dataset) if args.diagnostics_dataset else None
    )
    contact_trace_output = (
        Path(args.contact_trace_output) if args.contact_trace_output else None
    )
    contact_episode_output = (
        contact_trace_output.with_suffix(".episodes.json")
        if contact_trace_output is not None
        else None
    )
    diagnostic_image_dir = (
        Path(args.diagnostic_image_dir) if args.diagnostic_image_dir else None
    )
    protected = [output, summary_output]
    protected.extend(
        path
        for path in (
            diagnostics_output,
            diagnostics_dataset,
            contact_trace_output,
            contact_episode_output,
        )
        if path is not None
    )
    existing = [path for path in protected if path.exists()]
    if diagnostic_image_dir is not None and diagnostic_image_dir.exists():
        existing.extend(
            diagnostic_image_dir.glob(
                "{}_seed_*.png".format(args.diagnostic_image_prefix)
            )
        )
    if existing and not args.overwrite:
        raise FileExistsError(
            "refusing to overwrite attack-gate evaluation: {}".format(
                ", ".join(str(path) for path in existing)
            )
        )

    model = NaturalAttackGatePolicy.load(args.checkpoint)
    supported_versions = {
        "natural_treechop_attack_gate_bc_v2a",
        "natural_treechop_attack_gate_bc_v2b",
    }
    if model.model_version not in supported_versions:
        raise ValueError("checkpoint is not a supported BC attack gate")
    confirmation_frames = int(
        args.attack_confirmation_frames
        if args.attack_confirmation_frames is not None
        else getattr(model, "attack_confirmation_frames", 1)
    )
    seeds = (
        parse_seed_list(args.seeds)
        if args.seeds
        else [args.seed + index for index in range(args.episodes)]
    )
    episode_count = len(seeds)
    rows: List[Dict[str, Any]] = []
    all_teacher_labels: List[int] = []
    all_gate_decisions: List[int] = []
    all_confirmed_gate_decisions: List[int] = []
    diagnostic_rows: List[Dict[str, Any]] = []
    contact_trace_rows: List[Dict[str, Any]] = []
    contact_episode_diagnostics: List[Dict[str, Any]] = []
    samples: Dict[str, List[Any]] = {
        "pov": [],
        "action": [],
        "previous_action": [],
        "episode": [],
        "episode_seed": [],
        "episode_step": [],
        "episode_success": [],
        "audit_contact_state": [],
        "audit_raycast_is_log": [],
        "audit_raycast_in_range": [],
        "student_attack_probability": [],
        "student_gate_decision": [],
        "student_confirmed_gate_decision": [],
        "diagnostic_outcome": [],
    }
    total_privileged_accesses = 0
    started = time.perf_counter()

    env = make_telemetry_treechop_env(
        seed=args.seed,
        max_episode_steps=args.max_steps,
        include_raycast=True,
    )
    try:
        for episode_index, seed in enumerate(seeds):
            env.seed(seed)
            observation = env.reset()
            adapter = TreeResourceAdapter(
                interaction_action_id=8,
                interaction_size=45.0,
                interaction_uses_geometry=True,
                interaction_min_apparent_size=12.0,
                range_size_cap=120.0,
                reward_is_success=True,
            )
            policy = CandidateSearchPolicy(
                adapter,
                SearchConfig(
                    backward_action=9,
                    sensor_profile="f3_raycast",
                    align_threshold_degrees=12.0,
                    enable_trunk_contact=True,
                    contact_profile=args.contact_profile,
                    episode_max_steps=args.max_steps,
                ),
            )
            policy.reset(episode=episode_index + 1)
            runner = NaturalAttackGateRunner(
                policy,
                StudentContactAgent(model),
                args.mode,
                model.frame_stack,
                attack_confirmation_frames=confirmation_frames,
            )
            done = False
            info: Dict[str, Any] = {}
            step = 0
            episode_teacher_labels: List[int] = []
            episode_gate_decisions: List[int] = []
            episode_confirmed_gate_decisions: List[int] = []
            episode_sample_start = len(samples["episode"])
            while not done:
                previous_action = int(runner.previous_action)
                executed, record = runner.act(observation)
                if record["gate_decision"] is not None:
                    teacher_label = int(record["teacher_action"] == 7)
                    gate_decision = int(record["gate_decision"])
                    confirmed_decision = int(record["confirmed_gate_decision"])
                    probability = float(record["gate_probability"])
                    outcome = classify_gate_outcome(teacher_label, gate_decision)
                    episode_teacher_labels.append(teacher_label)
                    episode_gate_decisions.append(gate_decision)
                    episode_confirmed_gate_decisions.append(confirmed_decision)
                    raycast = observation.get("raycast", {})
                    samples["pov"].append(
                        np.asarray(runner.student.history.current_stack(), dtype=np.uint8)
                    )
                    samples["action"].append(int(record["teacher_action"]))
                    samples["previous_action"].append(previous_action)
                    samples["episode"].append(episode_index + 1)
                    samples["episode_seed"].append(seed)
                    samples["episode_step"].append(step)
                    samples["episode_success"].append(0)
                    samples["audit_contact_state"].append(record["contact_state_before"])
                    samples["audit_raycast_is_log"].append(
                        float(raycast.get("is_log", 0.0))
                    )
                    samples["audit_raycast_in_range"].append(
                        float(raycast.get("in_range", 0.0))
                    )
                    samples["student_attack_probability"].append(probability)
                    samples["student_gate_decision"].append(gate_decision)
                    samples["student_confirmed_gate_decision"].append(
                        confirmed_decision
                    )
                    samples["diagnostic_outcome"].append(outcome)
                    if diagnostic_sample_selected(
                        outcome,
                        probability,
                        model.decision_threshold,
                        args.diagnostic_margin,
                    ):
                        image_path = ""
                        if diagnostic_image_dir is not None:
                            image_path = str(
                                diagnostic_image_dir
                                / (
                                    "{}_seed_{:05d}_step_{:03d}_{}_p{:04d}.png".format(
                                        args.diagnostic_image_prefix,
                                        seed,
                                        step,
                                        outcome,
                                        int(round(probability * 1000)),
                                    )
                                )
                            )
                            atomic_write_rgb_png(
                                Path(image_path), np.asarray(observation["pov"])
                            )
                        diagnostic_rows.append(
                            {
                                "episode": episode_index + 1,
                                "seed": seed,
                                "step": step,
                                "outcome": outcome,
                                "teacher_label": teacher_label,
                                "gate_decision": gate_decision,
                                "confirmed_gate_decision": confirmed_decision,
                                "attack_probability": probability,
                                "decision_threshold": model.decision_threshold,
                                "contact_state": record["contact_state_before"],
                                "previous_action": previous_action,
                                "raycast_is_log": float(raycast.get("is_log", 0.0)),
                                "raycast_in_range": float(raycast.get("in_range", 0.0)),
                                "image_path": image_path,
                            }
                        )
                next_observation, reward, done, info = env.step(executed)
                policy.observe_transition(
                    executed, next_observation, reward, done, info
                )
                if contact_trace_output is not None:
                    step_diagnostics = policy.contact_diagnostics()
                    step_counters = step_diagnostics.get("counters", {})
                    search_diagnostics = policy.search_diagnostics()
                    contact_trace_rows.append(
                        {
                            "episode": episode_index + 1,
                            "seed": seed,
                            "step": step,
                            "action": int(executed),
                            "reward": float(reward),
                            "done": bool(done),
                            "success": bool(info.get("success", False)),
                            "time_limit_truncated": bool(
                                info.get("TimeLimit.truncated", False)
                            ),
                            "state_before": record["contact_state_before"],
                            "state_after_action": record["contact_state_after"],
                            "action_source": record["action_source"],
                            **compact_contact_trace(
                                step_diagnostics, step_counters
                            ),
                            **compact_search_trace(search_diagnostics),
                        }
                    )
                runner.observe_transition(executed)
                observation = next_observation
                step += 1

            success = bool(info.get("success", False))
            all_teacher_labels.extend(episode_teacher_labels)
            all_gate_decisions.extend(episode_gate_decisions)
            all_confirmed_gate_decisions.extend(
                episode_confirmed_gate_decisions
            )
            for index in range(episode_sample_start, len(samples["episode"])):
                samples["episode_success"][index] = int(success)
            total_privileged_accesses += runner.privileged_student_input_accesses
            diagnostics = policy.contact_diagnostics()
            counters = diagnostics.get("counters", {})
            if contact_trace_output is not None:
                contact_episode_diagnostics.append(
                    {
                        "episode": episode_index + 1,
                        "seed": seed,
                        "success": success,
                        "steps": step,
                        "time_limit_truncated": bool(
                            info.get("TimeLimit.truncated", False)
                        ),
                        "terminal_info": {
                            key: value
                            for key, value in info.items()
                            if isinstance(value, (str, int, float, bool))
                        },
                        "diagnostics": diagnostics,
                        "search_diagnostics": policy.search_diagnostics(),
                    }
                )
            row = {
                "mode": args.mode,
                "episode": episode_index + 1,
                "seed": seed,
                "success": success,
                "steps": step,
                "contact_steps": runner.contact_steps,
                "gate_predictions": runner.gate_predictions,
                "gate_attack_predictions": runner.gate_attack_predictions,
                "gate_hold_predictions": runner.gate_hold_predictions,
                "gate_confirmed_attack_predictions": (
                    runner.gate_confirmed_attack_predictions
                ),
                "gate_permissions_applied": runner.gate_permissions_applied,
                "external_attack_gate_checks": counters.get("external_attack_gate_checks", 0),
                "external_attack_gate_allows": counters.get("external_attack_gate_allows", 0),
                "external_attack_gate_rejections": counters.get("external_attack_gate_rejections", 0),
                "external_attack_gate_recenters": counters.get("external_attack_gate_recenters", 0),
                "coordinate_recoveries": counters.get("coordinate_recoveries", 0),
                "block_disappearances": counters.get("block_disappearances", 0),
                "pickups_after_disappearance": counters.get("pickup_after_disappearance", 0),
                "contact_owner_mismatches": policy.contact_owner_mismatches,
                "privileged_student_input_accesses": runner.privileged_student_input_accesses,
                "failure_layer": (
                    "success"
                    if success
                    else (
                        "upstream_no_gate"
                        if runner.gate_predictions == 0
                        else "contact_or_completion_after_gate"
                    )
                ),
            }
            rows.append(row)
            atomic_write_rows(output, rows)
            if diagnostics_output is not None:
                atomic_write_rows(diagnostics_output, diagnostic_rows)
            if contact_trace_output is not None:
                atomic_write_rows(contact_trace_output, contact_trace_rows)
                atomic_write_json(
                    contact_episode_output,
                    {"episodes": contact_episode_diagnostics},
                )
            if diagnostics_dataset is not None:
                atomic_write_npz(
                    diagnostics_dataset,
                    diagnostic_arrays(
                        samples,
                        {
                            "teacher_profile": args.contact_profile,
                            "frame_stack": model.frame_stack,
                            "source_checkpoint": args.checkpoint,
                            "source_model_version": model.model_version,
                            "decision_threshold": model.decision_threshold,
                            "attack_confirmation_frames": confirmation_frames,
                        },
                    ),
                )
            print(
                "mode={} episode={}/{} seed={} success={} steps={} gate={} rejects={}".format(
                    args.mode,
                    episode_index + 1,
                    episode_count,
                    seed,
                    success,
                    step,
                    runner.gate_predictions,
                    row["external_attack_gate_rejections"],
                ),
                flush=True,
            )
    finally:
        try:
            env.close()
        except psutil.NoSuchProcess as error:
            print("WARNING: Minecraft already exited during close: {}".format(error))

    successes = sum(bool(row["success"]) for row in rows)
    max_step_failures = sum(
        not row["success"] and row["steps"] >= args.max_steps for row in rows
    )
    labels = np.asarray(all_teacher_labels, dtype=np.int64)
    decisions = np.asarray(all_gate_decisions, dtype=np.int64)
    confirmed_decisions = np.asarray(
        all_confirmed_gate_decisions, dtype=np.int64
    )
    agreement = (
        binary_metrics(decisions.astype(np.float32), labels, 0.5)
        if len(labels)
        else {}
    )
    confirmed_agreement = (
        binary_metrics(confirmed_decisions.astype(np.float32), labels, 0.5)
        if len(labels)
        else {}
    )
    lower, upper = wilson_interval(successes, episode_count)
    summary: Dict[str, Any] = {
        "mode": args.mode,
        "checkpoint": args.checkpoint,
        "decision_threshold": model.decision_threshold,
        "attack_confirmation_frames": confirmation_frames,
        "teacher_profile": args.contact_profile,
        "episodes": episode_count,
        "seed_range": [seeds[0], seeds[-1]],
        "successes": successes,
        "success_rate": successes / episode_count,
        "wilson_95_percent": [round(lower, 4), round(upper, 4)],
        "mean_steps": float(np.mean([row["steps"] for row in rows])),
        "median_steps": float(np.median([row["steps"] for row in rows])),
        "max_step_failures": max_step_failures,
        "gate_samples": len(labels),
        "gate_agreement": agreement,
        "confirmed_gate_agreement": confirmed_agreement,
        "gate_permissions_applied": sum(row["gate_permissions_applied"] for row in rows),
        "external_attack_gate_checks": sum(row["external_attack_gate_checks"] for row in rows),
        "external_attack_gate_allows": sum(row["external_attack_gate_allows"] for row in rows),
        "external_attack_gate_rejections": sum(row["external_attack_gate_rejections"] for row in rows),
        "external_attack_gate_recenters": sum(row["external_attack_gate_recenters"] for row in rows),
        "contact_owner_mismatches": sum(row["contact_owner_mismatches"] for row in rows),
        "privileged_student_input_accesses": total_privileged_accesses,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "diagnostics": {
            "selected_rows": len(diagnostic_rows),
            "rows_output": str(diagnostics_output) if diagnostics_output else None,
            "dataset_output": str(diagnostics_dataset) if diagnostics_dataset else None,
            "image_directory": str(diagnostic_image_dir) if diagnostic_image_dir else None,
            "selection_margin": args.diagnostic_margin,
            "contact_trace_output": (
                str(contact_trace_output) if contact_trace_output else None
            ),
            "contact_episode_output": (
                str(contact_episode_output) if contact_episode_output else None
            ),
        },
    }
    if args.mode == "shadow":
        agreement_for_gate = confirmed_agreement
        gate = {
            "at_least_15_teacher_successes": successes >= 15,
            "gate_samples_at_least_300": len(labels) >= 300,
            "balanced_accuracy_at_least_80_percent": agreement_for_gate.get("balanced_accuracy", 0.0) >= 0.80,
            "attack_precision_at_least_97_percent": agreement_for_gate.get("attack_precision", 0.0) >= 0.97,
            "attack_recall_at_least_75_percent": agreement_for_gate.get("attack_recall", 0.0) >= 0.75,
            "false_positive_rate_at_most_2_percent": agreement_for_gate.get("false_positive_rate", 1.0) <= 0.02,
            "contact_owner_mismatches_zero": summary["contact_owner_mismatches"] == 0,
            "privileged_student_input_accesses_zero": total_privileged_accesses == 0,
        }
        gate["all_conditions_met"] = all(gate.values())
        summary["shadow_gate"] = gate
        summary["shadow_gate_passed"] = gate["all_conditions_met"]
    else:
        gate = {
            "success_rate_at_least_60_percent": successes / episode_count >= 0.60,
            "at_most_40_percent_max_step_failures": max_step_failures / episode_count <= 0.40,
            "at_least_one_gate_check": summary["external_attack_gate_checks"] > 0,
            "gate_permissions_were_applied": summary["gate_permissions_applied"] > 0,
            "contact_owner_mismatches_zero": summary["contact_owner_mismatches"] == 0,
            "privileged_student_input_accesses_zero": total_privileged_accesses == 0,
        }
        gate["all_conditions_met"] = all(gate.values())
        summary["autonomous_smoke_gate"] = gate
        summary["autonomous_smoke_passed"] = gate["all_conditions_met"]
    atomic_write_json(summary_output, summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
