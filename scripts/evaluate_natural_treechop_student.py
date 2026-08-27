"""Autonomous end-to-end rollout and limited DAgger data collection."""

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import psutil

from mc_rl.experiments import append_experiment, file_sha256, seeds_for_split
from mc_rl.learning_observation import STUDENT_VECTOR_NAMES
from mc_rl.natural_treechop_bc import (
    NaturalTreechopBCPolicy,
    NaturalTreechopStudentAgent,
    coarse_teacher_phase,
)
from mc_rl.natural_treechop_runtime import make_bootstrap_teacher
from mc_rl.telemetry_treechop_env import make_telemetry_treechop_env
from mc_rl.trunk_contact import CONTACT_PROFILE_EMERGENCY_TARGET_PREEMPTION_V9_11


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("dagger_rollout", "student_dev", "student_holdout"), required=True)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--manifest", default="configs/seeds/natural_treechop_v1.json")
    parser.add_argument("--mode", choices=("autonomous", "dagger"), default="autonomous")
    parser.add_argument("--output", required=True)
    parser.add_argument("--dagger-output")
    parser.add_argument("--contact-profile", default=CONTACT_PROFILE_EMERGENCY_TARGET_PREEMPTION_V9_11)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--hypothesis", required=True)
    return parser.parse_args()


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> Tuple[float, float]:
    if trials <= 0:
        return 0.0, 0.0
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    centre = proportion + z * z / (2.0 * trials)
    margin = z * np.sqrt(
        proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials * trials)
    )
    return (centre - margin) / denominator, (centre + margin) / denominator


def atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def atomic_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def classify_failure(
    success: bool,
    saw_log: bool,
    reached_log: bool,
    meaningful_interaction: Optional[int],
    inferred_break: Optional[int],
) -> str:
    if success:
        return "success"
    if not saw_log:
        return "search_timeout"
    if not reached_log:
        return "approach_timeout"
    if meaningful_interaction is None:
        return "contact_without_valid_attack"
    if inferred_break is None:
        return "attack_without_observed_break"
    return "break_without_inventory_pickup"


def duration(start: Optional[int], end: Optional[int], fallback: int) -> Optional[int]:
    if start is None:
        return None
    return max(0, int((fallback if end is None else end) - start))


def atomic_dagger(path: Path, samples: Dict[str, List[Any]], metadata: Dict[str, Any]):
    arrays = {
        "pov": np.asarray(samples["pov"], dtype=np.uint8),
        "legal_vector": np.asarray(samples["legal_vector"], dtype=np.float32),
        "action": np.asarray(samples["action"], dtype=np.int32),
        "previous_action": np.asarray(samples["previous_action"], dtype=np.int32),
        "episode": np.asarray(samples["episode"], dtype=np.int32),
        "episode_seed": np.asarray(samples["episode_seed"], dtype=np.int32),
        "episode_step": np.asarray(samples["episode_step"], dtype=np.int32),
        "episode_success": np.asarray(samples["episode_success"], dtype=np.int8),
        "source": np.asarray(["dagger"] * len(samples["action"])),
        "audit_coarse_phase": np.asarray(samples["audit_coarse_phase"]),
        "audit_search_state": np.asarray(samples["audit_search_state"]),
        "audit_contact_state": np.asarray(samples["audit_contact_state"]),
        "audit_student_action": np.asarray(samples["audit_student_action"], dtype=np.int32),
        "audit_raycast_is_log": np.asarray(samples["audit_raycast_is_log"], dtype=np.int8),
        "audit_raycast_in_range": np.asarray(samples["audit_raycast_in_range"], dtype=np.int8),
    }
    arrays.update({key: np.asarray(value) for key, value in metadata.items()})
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def main():
    args = parse_args()
    if args.mode == "dagger" and not args.dagger_output:
        raise ValueError("dagger mode requires --dagger-output")
    seeds = seeds_for_split(args.split, args.manifest, limit=args.episodes)
    output = Path(args.output)
    summary_output = output.with_suffix(".summary.json")
    protected = [output, summary_output]
    if args.dagger_output:
        protected.append(Path(args.dagger_output))
    existing = [path for path in protected if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError("refusing to overwrite: {}".format(existing))

    policy = NaturalTreechopBCPolicy.load(args.checkpoint)
    rows: List[Dict[str, Any]] = []
    dagger = {key: [] for key in (
        "pov", "legal_vector", "action", "previous_action", "episode",
        "episode_seed", "episode_step", "episode_success", "audit_coarse_phase",
        "audit_search_state", "audit_contact_state", "audit_student_action",
        "audit_raycast_is_log", "audit_raycast_in_range",
    )}
    started = time.perf_counter()
    env = make_telemetry_treechop_env(
        seed=seeds[0], max_episode_steps=args.max_steps, include_raycast=True
    )
    try:
        for episode_index, seed in enumerate(seeds, start=1):
            env.seed(seed)
            observation = env.reset()
            student = NaturalTreechopStudentAgent(policy, args.max_steps)
            teacher = make_bootstrap_teacher(args.max_steps, args.contact_profile)
            teacher.reset(episode=episode_index)
            done = False
            info: Dict[str, Any] = {}
            step = 0
            previous_action = 0
            action_counts: Counter = Counter()
            phase_counts: Counter = Counter()
            first_tree_visible = None
            first_in_range = None
            first_interaction = None
            inferred_break = None
            sustained_valid_attacks = 0
            episode_dagger_indices: List[int] = []
            episode_started = time.perf_counter()
            while not done:
                raycast = observation["raycast"]
                is_log = bool(raycast["is_log"])
                in_range = bool(raycast["in_range"])
                if is_log and first_tree_visible is None:
                    first_tree_visible = step
                if is_log and in_range and first_in_range is None:
                    first_in_range = step
                search_before = str(getattr(teacher.state, "value", teacher.state))
                contact_before = str(teacher.contact_state or "")
                corrective_action = int(teacher.act(observation))
                phase = coarse_teacher_phase(
                    search_before, contact_before, str(teacher.last_action_source)
                )
                student_action, legal_vector = student.act(observation, step)
                if first_interaction is None and student_action in (7, 8) and is_log and in_range:
                    first_interaction = step
                valid_attack = student_action in (7, 8) and is_log and in_range
                if valid_attack:
                    sustained_valid_attacks += 1
                if (
                    inferred_break is None
                    and sustained_valid_attacks >= 5
                    and not is_log
                    and first_interaction is not None
                ):
                    inferred_break = step
                if args.mode == "dagger":
                    index = len(dagger["action"])
                    episode_dagger_indices.append(index)
                    legal_pov = student.frames[-1]
                    dagger["pov"].append(np.asarray(legal_pov, dtype=np.uint8))
                    dagger["legal_vector"].append(legal_vector)
                    dagger["action"].append(corrective_action)
                    dagger["previous_action"].append(previous_action)
                    dagger["episode"].append(episode_index)
                    dagger["episode_seed"].append(seed)
                    dagger["episode_step"].append(step)
                    dagger["episode_success"].append(0)
                    dagger["audit_coarse_phase"].append(phase)
                    dagger["audit_search_state"].append(search_before)
                    dagger["audit_contact_state"].append(contact_before)
                    dagger["audit_student_action"].append(student_action)
                    dagger["audit_raycast_is_log"].append(is_log)
                    dagger["audit_raycast_in_range"].append(in_range)
                next_observation, reward, done, info = env.step(student_action)
                teacher.observe_transition(
                    student_action, next_observation, reward, done, info
                )
                student.observe_transition(student_action)
                action_counts[student_action] += 1
                phase_counts[phase] += 1
                previous_action = student_action
                observation = next_observation
                step += 1
            success = bool(info.get("success", False))
            for index in episode_dagger_indices:
                dagger["episode_success"][index] = int(success)
            failure = classify_failure(
                success,
                first_tree_visible is not None,
                first_in_range is not None,
                first_interaction,
                inferred_break,
            )
            row = {
                "mode": args.mode,
                "episode": episode_index,
                "seed": seed,
                "success": success,
                "steps": step,
                "timeout": bool(not success and step >= args.max_steps),
                "failure_taxonomy": failure,
                "inventory_log_delta": info.get("inventory_log_delta"),
                "success_source": info.get("success_source", ""),
                "first_tree_visible_step": first_tree_visible,
                "first_in_range_step": first_in_range,
                "first_meaningful_interaction_step": first_interaction,
                "inferred_break_step": inferred_break,
                "search_time": first_tree_visible if first_tree_visible is not None else step,
                "approach_time": duration(first_tree_visible, first_in_range, step),
                "contact_time": duration(first_in_range, first_interaction, step),
                "chop_time": duration(first_interaction, inferred_break, step),
                "pickup_time": duration(inferred_break, step if success else None, step),
                "action_counts": json.dumps(dict(sorted(action_counts.items()))),
                "oracle_phase_counts": json.dumps(dict(sorted(phase_counts.items()))),
                "runtime_seconds": round(time.perf_counter() - episode_started, 3),
            }
            rows.append(row)
            atomic_csv(output, rows)
            print(
                "{} episode={}/{} seed={} success={} steps={} failure={}".format(
                    args.mode, episode_index, len(seeds), seed, success, step, failure
                ),
                flush=True,
            )
    finally:
        try:
            env.close()
        except psutil.NoSuchProcess as error:
            print("WARNING: Minecraft already exited during close: {}".format(error))

    successes = sum(bool(row["success"]) for row in rows)
    successful_steps = [row["steps"] for row in rows if row["success"]]
    lower, upper = wilson_interval(successes, len(rows))
    summary = {
        "mode": args.mode,
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": file_sha256(Path(args.checkpoint)),
        "seed_manifest": args.manifest,
        "seed_split": args.split,
        "seeds": seeds,
        "episodes": len(rows),
        "successes": successes,
        "success_rate": successes / len(rows),
        "wilson_95_percent": [round(lower, 4), round(upper, 4)],
        "median_completion_steps": (
            float(np.median(successful_steps)) if successful_steps else None
        ),
        "p90_completion_steps": (
            float(np.percentile(successful_steps, 90)) if successful_steps else None
        ),
        "timeout_rate": float(np.mean([row["timeout"] for row in rows])),
        "failure_taxonomy": dict(Counter(row["failure_taxonomy"] for row in rows)),
        "phase_time_medians": {
            key: float(np.median([row[key] for row in rows if row[key] is not None]))
            if any(row[key] is not None for row in rows)
            else None
            for key in ("search_time", "approach_time", "contact_time", "chop_time", "pickup_time")
        },
        "median_first_meaningful_interaction_step": (
            float(np.median([
                row["first_meaningful_interaction_step"]
                for row in rows
                if row["first_meaningful_interaction_step"] is not None
            ]))
            if any(row["first_meaningful_interaction_step"] is not None for row in rows)
            else None
        ),
        "privileged_actor_inputs": 0,
        "teacher_actions_executed": 0,
        "success_definition": "inventory natural log delta >= 1",
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    if args.mode == "dagger":
        dagger_path = Path(args.dagger_output)
        metadata = {
            "dataset_version": "natural_treechop_dagger_v1",
            "student_vector_names": np.asarray(STUDENT_VECTOR_NAMES),
            "student_input_manifest": np.asarray(policy.student_input_manifest),
            "seed_manifest": str(args.manifest),
            "seed_split": str(args.split),
            "source_checkpoint": str(args.checkpoint),
            "teacher_profile": str(args.contact_profile),
        }
        atomic_dagger(dagger_path, dagger, metadata)
        summary["dagger_dataset"] = {
            "path": str(dagger_path),
            "sha256": file_sha256(dagger_path),
            "samples": len(dagger["action"]),
            "student_teacher_action_agreement": float(
                np.mean(
                    np.asarray(dagger["action"], dtype=np.int64)
                    == np.asarray(dagger["audit_student_action"], dtype=np.int64)
                )
            ),
        }
    atomic_json(summary_output, summary)
    append_experiment(
        {
            "experiment_id": args.experiment_id,
            "hypothesis": args.hypothesis,
            "config": {
                "script": "scripts/evaluate_natural_treechop_student.py",
                "mode": args.mode,
                "max_steps": args.max_steps,
                "teacher_profile_for_audit_only": args.contact_profile,
            },
            "seed_manifest": args.manifest,
            "seed_split": args.split,
            "checkpoint": {
                "path": args.checkpoint,
                "sha256": summary["checkpoint_sha256"],
            },
            "dataset": summary.get("dagger_dataset"),
            "metrics": summary,
            "runtime_seconds": summary["elapsed_seconds"],
            "conclusion": (
                "Student-induced states and corrective labels collected."
                if args.mode == "dagger"
                else "Autonomous rollout completed with no teacher action execution."
            ),
            "status": "kept",
        }
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
