"""Shadow or closed-loop evaluation for the BC v2 hybrid controller."""

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import psutil

from mc_rl.natural_bc_v2_runner import HybridNaturalContactRunner
from mc_rl.natural_contact_bc import StudentContactAgent
from mc_rl.natural_contact_bc_v2 import NaturalContactBCV2Policy
from mc_rl.resource_adapters import TreeResourceAdapter
from mc_rl.search_policy import CandidateSearchPolicy, SearchConfig
from mc_rl.telemetry_treechop_env import make_telemetry_treechop_env
from mc_rl.trunk_contact import CONTACT_PROFILE_TERRAIN_ROUTE_DROP_COMPLETION_V9_6
from scripts.train_natural_treechop_bc_v2 import evaluate_predictions


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=17100)
    parser.add_argument("--mode", choices=("shadow", "autonomous"), default="shadow")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/natural_treechop_contact_bc_v2_hybrid.npz",
    )
    parser.add_argument(
        "--output",
        default="logs/find_tree/natural_treechop_bc_v2_hybrid_shadow_17100_20.csv",
    )
    parser.add_argument(
        "--contact-profile",
        default=CONTACT_PROFILE_TERRAIN_ROUTE_DROP_COMPLETION_V9_6,
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


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
    output = Path(args.output)
    summary_output = output.with_suffix(".summary.json")
    existing = [path for path in (output, summary_output) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "refusing to overwrite BC v2 evaluation outputs: {}".format(
                ", ".join(str(path) for path in existing)
            )
        )

    model = NaturalContactBCV2Policy.load(args.checkpoint)
    if model.model_version != "natural_treechop_contact_bc_v2_hybrid":
        raise ValueError("checkpoint is not a BC v2 hybrid model")
    seeds = [args.seed + index for index in range(args.episodes)]
    rows: List[Dict[str, Any]] = []
    learned_teacher: List[int] = []
    learned_student: List[int] = []
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
            obs = env.reset()
            adapter = TreeResourceAdapter(
                interaction_action_id=8,
                interaction_size=45.0,
                interaction_uses_geometry=True,
                interaction_min_apparent_size=12.0,
                range_size_cap=120.0,
                reward_is_success=True,
            )
            search = CandidateSearchPolicy(
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
            search.reset(episode=episode_index + 1)
            student = StudentContactAgent(model)
            runner = HybridNaturalContactRunner(
                search, student, args.mode, model.frame_stack
            )
            done = False
            info: Dict[str, Any] = {}
            step = 0
            episode_learned_teacher: List[int] = []
            episode_learned_student: List[int] = []
            owner_counts: Counter = Counter()
            while not done:
                executed, record = runner.act(obs)
                owner_counts[record["control_owner"]] += 1
                if record["learned_boundary"] and record["student_action"] is not None:
                    episode_learned_teacher.append(record["teacher_action"])
                    episode_learned_student.append(record["student_action"])
                next_obs, reward, done, info = env.step(executed)
                search.observe_transition(executed, next_obs, reward, done, info)
                runner.observe_transition(executed)
                obs = next_obs
                step += 1

            success = bool(info.get("success", False))
            learned_teacher.extend(episode_learned_teacher)
            learned_student.extend(episode_learned_student)
            total_privileged_accesses += runner.privileged_student_input_accesses
            diagnostics = search.contact_diagnostics()
            counters = diagnostics.get("counters", {})
            row = {
                "mode": args.mode,
                "episode": episode_index + 1,
                "seed": seed,
                "success": success,
                "steps": step,
                "contact_steps": runner.contact_steps,
                "visual_student_predictions": runner.visual_student_predictions,
                "visual_student_actions_executed": runner.visual_student_steps,
                "scripted_contact_steps": runner.scripted_contact_steps,
                "privileged_student_input_accesses": runner.privileged_student_input_accesses,
                "contact_owner_mismatches": search.contact_owner_mismatches,
                "coordinate_recoveries": counters.get("coordinate_recoveries", 0),
                "block_disappearances": counters.get("block_disappearances", 0),
                "pickups_after_disappearance": counters.get("pickup_after_disappearance", 0),
                "control_owner_counts": json.dumps(dict(sorted(owner_counts.items()))),
            }
            rows.append(row)
            atomic_write_rows(output, rows)
            print(
                "mode={} episode={}/{} seed={} success={} steps={} visual={} scripted={}".format(
                    args.mode,
                    episode_index + 1,
                    args.episodes,
                    seed,
                    success,
                    step,
                    runner.visual_student_predictions,
                    runner.scripted_contact_steps,
                ),
                flush=True,
            )
    finally:
        try:
            env.close()
        except psutil.NoSuchProcess as error:
            print("WARNING: Minecraft already exited during close: {}".format(error))

    teacher_array = np.asarray(learned_teacher, dtype=np.int64)
    student_array = np.asarray(learned_student, dtype=np.int64)
    agreement = (
        evaluate_predictions(student_array, teacher_array)
        if len(teacher_array)
        else {}
    )
    successes = sum(bool(row["success"]) for row in rows)
    max_step_failures = sum(
        not row["success"] and row["steps"] >= args.max_steps for row in rows
    )
    lower, upper = wilson_interval(successes, args.episodes)
    summary: Dict[str, Any] = {
        "mode": args.mode,
        "checkpoint": args.checkpoint,
        "teacher_profile": args.contact_profile,
        "episodes": args.episodes,
        "seed_range": [seeds[0], seeds[-1]],
        "successes": successes,
        "success_rate": successes / args.episodes,
        "wilson_95_percent": [round(lower, 4), round(upper, 4)],
        "mean_steps": float(np.mean([row["steps"] for row in rows])),
        "median_steps": float(np.median([row["steps"] for row in rows])),
        "max_step_failures": max_step_failures,
        "learned_boundary_samples": len(teacher_array),
        "learned_boundary_agreement": agreement,
        "visual_student_actions_executed": sum(
            row["visual_student_actions_executed"] for row in rows
        ),
        "scripted_contact_steps": sum(row["scripted_contact_steps"] for row in rows),
        "contact_owner_mismatches": sum(row["contact_owner_mismatches"] for row in rows),
        "privileged_student_input_accesses": total_privileged_accesses,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    if args.mode == "shadow":
        gate = {
            "at_least_15_teacher_successes": successes >= 15,
            "learned_boundary_samples_at_least_100": len(teacher_array) >= 100,
            "balanced_accuracy_at_least_70_percent": agreement.get("balanced_accuracy", 0.0) >= 0.70,
            "attack_precision_at_least_90_percent": agreement.get("attack_precision", 0.0) >= 0.90,
            "attack_recall_at_least_80_percent": agreement.get("attack_recall", 0.0) >= 0.80,
            "fine_yaw_direction_at_least_85_percent": agreement.get("fine_yaw_direction_agreement", 0.0) >= 0.85,
            "fine_pitch_direction_at_least_80_percent": agreement.get("fine_pitch_direction_agreement", 0.0) >= 0.80,
            "contact_owner_mismatches_zero": summary["contact_owner_mismatches"] == 0,
            "privileged_student_input_accesses_zero": total_privileged_accesses == 0,
        }
        gate["all_conditions_met"] = all(gate.values())
        summary["shadow_gate"] = gate
        summary["shadow_gate_passed"] = gate["all_conditions_met"]
    else:
        disappearances = sum(row["block_disappearances"] for row in rows)
        pickups = sum(row["pickups_after_disappearance"] for row in rows)
        gate = {
            "at_least_16_of_20_successes": args.episodes >= 20 and successes >= 16,
            "at_most_4_max_step_failures": max_step_failures <= 4,
            "at_least_one_visual_student_action": summary["visual_student_actions_executed"] > 0,
            "contact_owner_mismatches_zero": summary["contact_owner_mismatches"] == 0,
            "privileged_student_input_accesses_zero": total_privileged_accesses == 0,
            "pickup_given_disappearance_at_least_80_percent": (
                disappearances == 0 or pickups / disappearances >= 0.80
            ),
        }
        gate["all_conditions_met"] = all(gate.values())
        summary["autonomous_gate"] = gate
        summary["autonomous_gate_passed"] = gate["all_conditions_met"]
    atomic_write_json(summary_output, summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
