"""Shadow and closed-loop evaluation of the natural contact BC student.

Shadow mode executes the v9.6 teacher action while recording the student
prediction; the prediction never influences the environment. Autonomous mode
executes the student action whenever the contact owner is active, with no
teacher fallback inside the contact phase; the teacher only continues to run
its own state machine to decide handoff, replan, and episode termination.

Both modes use the ``f3_raycast`` environment so the frozen v9.6 upstream
behaves exactly as it did during its 18/20 gate. The student's only input
channel is a guarded POV-only view; raycast telemetry never reaches the
student action path.
"""

import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import psutil

from mc_rl.natural_bc_runner import NaturalContactRunner
from mc_rl.natural_contact_bc import (
    ACTION_CLASSES,
    NaturalContactBCPolicy,
    StudentContactAgent,
    mirror_actions,
)
from mc_rl.resource_adapters import TreeResourceAdapter
from mc_rl.search_policy import CandidateSearchPolicy, SearchConfig
from mc_rl.telemetry_treechop_env import make_telemetry_treechop_env
from mc_rl.trunk_contact import CONTACT_PROFILE_TERRAIN_ROUTE_DROP_COMPLETION_V9_6

LEFT_ACTIONS = (3, 10)
RIGHT_ACTIONS = (4, 11)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=17100)
    parser.add_argument(
        "--mode", choices=("shadow", "autonomous"), default="shadow"
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/natural_treechop_contact_bc_v1_stack4.npz",
    )
    parser.add_argument(
        "--output",
        default="logs/find_tree/natural_treechop_bc_v1_shadow_17100_20.csv",
    )
    parser.add_argument(
        "--contact-profile",
        default=CONTACT_PROFILE_TERRAIN_ROUTE_DROP_COMPLETION_V9_6,
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


def atomic_write_rows(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def agreement_metrics(
    teacher_actions: List[int], student_actions: List[int]
) -> Dict[str, Any]:
    if not teacher_actions:
        return {}
    teacher = np.asarray(teacher_actions)
    student = np.asarray(student_actions)
    agreement = student == teacher
    selected = np.isin(teacher, LEFT_ACTIONS + RIGHT_ACTIONS)
    direction = (
        float(
            (
                np.isin(student[selected], LEFT_ACTIONS)
                == np.isin(teacher[selected], LEFT_ACTIONS)
            ).mean()
        )
        if selected.any()
        else 0.0
    )
    per_class = {}
    for action in ACTION_CLASSES:
        mask = teacher == action
        if mask.any():
            per_class[str(int(action))] = {
                "samples": int(mask.sum()),
                "agreement": float(agreement[mask].mean()),
            }
    attack_predicted = student == 7
    attack_actual = teacher == 7
    return {
        "samples": int(len(teacher)),
        "balanced_agreement": float(
            np.mean(
                [
                    agreement[teacher == action].mean()
                    for action in np.unique(teacher)
                ]
            )
        ),
        "overall_agreement": float(agreement.mean()),
        "yaw_direction_agreement": direction,
        "attack_precision": (
            float((attack_predicted & attack_actual).sum() / attack_predicted.sum())
            if attack_predicted.any()
            else 0.0
        ),
        "attack_recall": (
            float((attack_predicted & attack_actual).sum() / attack_actual.sum())
            if attack_actual.any()
            else 0.0
        ),
        "per_teacher_action": per_class,
    }


def main():
    args = parse_args()
    if args.episodes <= 0 or args.max_steps <= 0:
        raise ValueError("episodes and max-steps must be positive")
    output = Path(args.output)
    summary_output = output.with_suffix(".summary.json")
    protected = (output, summary_output)
    existing = [path for path in protected if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "refusing to overwrite existing BC evaluation output: {}".format(
                ", ".join(str(path) for path in existing)
            )
        )

    policy_model = NaturalContactBCPolicy.load(args.checkpoint)
    seeds = [args.seed + index for index in range(args.episodes)]
    rows: List[Dict[str, Any]] = []
    started_at = time.perf_counter()
    all_teacher_contact: List[int] = []
    all_student_contact: List[int] = []
    state_action_agreement: Dict[str, List[int]] = defaultdict(list)
    state_action_total: Dict[str, int] = defaultdict(int)
    total_privileged_accesses = 0

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
            config = SearchConfig(
                backward_action=9,
                sensor_profile="f3_raycast",
                align_threshold_degrees=12.0,
                enable_trunk_contact=True,
                contact_profile=args.contact_profile,
                episode_max_steps=args.max_steps,
            )
            policy = CandidateSearchPolicy(adapter, config)
            policy.reset(episode=episode_index + 1)
            student_agent = StudentContactAgent(policy_model)
            runner = NaturalContactRunner(
                policy, student_agent, args.mode, policy_model.frame_stack
            )
            done = False
            info: Dict[str, Any] = {}
            step = 0
            episode_teacher_contact: List[int] = []
            episode_student_contact: List[int] = []
            executed_action_counts: Counter = Counter()
            contact_state_counts: Counter = Counter()
            contact_agreement_steps = 0
            while not done:
                executed, record = runner.act(observation)
                contact_state = policy.contact_state or ""
                if record["contact_active"]:
                    episode_teacher_contact.append(record["teacher_action"])
                    episode_student_contact.append(record["student_action"])
                    executed_action_counts[int(executed)] += 1
                    contact_state_counts[contact_state] += 1
                    state_action_total[contact_state] += 1
                    if record["student_action"] == record["teacher_action"]:
                        contact_agreement_steps += 1
                        state_action_agreement[contact_state].append(1)
                    else:
                        state_action_agreement[contact_state].append(0)
                next_observation, reward, done, info = env.step(executed)
                policy.observe_transition(
                    executed, next_observation, reward, done, info
                )
                runner.observe_transition(executed)
                observation = next_observation
                step += 1
            total_privileged_accesses += (
                runner.privileged_student_input_accesses
            )
            success = bool(info.get("success", False))
            all_teacher_contact.extend(episode_teacher_contact)
            all_student_contact.extend(episode_student_contact)
            final_contact = policy.contact_diagnostics()
            counters = final_contact.get("counters", {})
            row = {
                "mode": args.mode,
                "episode": episode_index + 1,
                "seed": seed,
                "success": success,
                "steps": step,
                "contact_steps": runner.contact_steps,
                "contact_attempts": runner.attempt_id,
                "contact_owner_mismatches": policy.contact_owner_mismatches,
                "privileged_student_input_accesses": (
                    runner.privileged_student_input_accesses
                ),
                "student_actions_executed": runner.student_actions_executed,
                "teacher_contact_actions": (
                    runner.teacher_actions_in_contact if args.mode == "shadow" else 0
                ),
                "contact_agreement_steps": contact_agreement_steps,
                "contact_agreement_fraction": (
                    contact_agreement_steps / runner.contact_steps
                    if runner.contact_steps
                    else 0.0
                ),
                "block_disappearances": counters.get(
                    "block_disappearances", 0
                ),
                "pickups_after_disappearance": counters.get(
                    "pickup_after_disappearance", 0
                ),
                "coordinate_recoveries": counters.get(
                    "coordinate_recoveries", 0
                ),
                "contact_result": final_contact.get("result") or "",
                "action_counts": json.dumps(
                    dict(sorted(executed_action_counts.items()))
                ),
                "contact_state_counts": json.dumps(
                    dict(sorted(contact_state_counts.items()))
                ),
            }
            rows.append(row)
            atomic_write_rows(output, rows)
            print(
                "mode={} episode={}/{} seed={} success={} steps={} "
                "contact={} agreement={:.2f}".format(
                    args.mode,
                    episode_index + 1,
                    args.episodes,
                    seed,
                    success,
                    step,
                    runner.contact_steps,
                    row["contact_agreement_fraction"],
                ),
                flush=True,
            )
    finally:
        try:
            env.close()
        except psutil.NoSuchProcess as error:
            print(
                "WARNING: Minecraft already exited during close: {}".format(error)
            )

    successes = sum(row["success"] for row in rows)
    max_step_failures = sum(
        (not row["success"]) and row["steps"] >= args.max_steps for row in rows
    )
    agreement = agreement_metrics(
        all_teacher_contact, all_student_contact
    )
    disappearances = sum(
        row["block_disappearances"] for row in rows
    )
    pickups = sum(row["pickups_after_disappearance"] for row in rows)
    assisted = 0  # No teacher fallback exists; any manual intervention would
    # be recorded here explicitly.
    lower, upper = wilson_interval(successes, args.episodes)
    summary: Dict[str, Any] = {
        "mode": args.mode,
        "checkpoint": args.checkpoint,
        "teacher_profile": args.contact_profile,
        "episodes": args.episodes,
        "seed_range": [seeds[0], seeds[-1]],
        "max_steps": args.max_steps,
        "successes": successes,
        "success_rate": successes / args.episodes,
        "wilson_95_percent": [round(lower, 4), round(upper, 4)],
        "mean_steps": float(np.mean([row["steps"] for row in rows])),
        "median_steps": float(np.median([row["steps"] for row in rows])),
        "max_step_failures": max_step_failures,
        "assisted_episodes": assisted,
        "contact_owner_mismatches": sum(
            row["contact_owner_mismatches"] for row in rows
        ),
        "privileged_student_input_accesses": total_privileged_accesses,
        "episodes_with_contact": sum(row["contact_steps"] > 0 for row in rows),
        "success_given_contact": (
            successes / sum(row["contact_steps"] > 0 for row in rows)
            if any(row["contact_steps"] > 0 for row in rows)
            else 0.0
        ),
        "block_disappearances": disappearances,
        "pickups_after_disappearance": pickups,
        "pickup_given_disappearance": (
            pickups / disappearances if disappearances else None
        ),
        "teacher_action_reference_gate": (
            "v9.6 teacher gate 18/20 (historical reference only)"
        ),
        "agreement": agreement,
        "per_contact_state_agreement": {
            state: float(np.mean(values))
            for state, values in sorted(state_action_agreement.items())
        },
        "per_contact_state_samples": dict(
            sorted(state_action_total.items())
        ),
        "elapsed_seconds": round(time.perf_counter() - started_at, 3),
    }
    if args.mode == "shadow":
        summary["shadow_gate"] = {
            "at_least_15_teacher_successes": bool(successes >= 15),
            "balanced_agreement_at_least_60_percent": bool(
                agreement.get("balanced_agreement", 0.0) >= 0.60
            ),
            "attack_precision_at_least_90_percent": bool(
                agreement.get("attack_precision", 0.0) >= 0.90
            ),
            "attack_recall_at_least_75_percent": bool(
                agreement.get("attack_recall", 0.0) >= 0.75
            ),
            "yaw_direction_agreement_at_least_90_percent": bool(
                agreement.get("yaw_direction_agreement", 0.0) >= 0.90
            ),
            "privileged_student_input_accesses_zero": bool(
                total_privileged_accesses == 0
            ),
        }
        gate = summary["shadow_gate"]
        gate["all_conditions_met"] = all(
            value for key, value in gate.items() if isinstance(value, bool)
        )
        summary["shadow_gate_passed"] = bool(gate["all_conditions_met"])
    else:
        summary["autonomous_gate"] = {
            "at_least_16_of_20": bool(
                args.episodes >= 20 and successes >= 16
            ),
            "at_most_4_max_step_failures": bool(max_step_failures <= 4),
            "assisted_episodes_zero": bool(assisted == 0),
            "contact_owner_mismatches_zero": bool(
                summary["contact_owner_mismatches"] == 0
            ),
            "privileged_student_input_accesses_zero": bool(
                total_privileged_accesses == 0
            ),
            "at_least_one_success_through_recovery": bool(
                sum(
                    row["success"]
                    and (
                        row["coordinate_recoveries"] > 0
                        or row["contact_attempts"] > 1
                    )
                    for row in rows
                )
                >= 1
            ),
            "pickup_given_disappearance_at_least_80_percent": bool(
                summary["pickup_given_disappearance"] is None
                or summary["pickup_given_disappearance"] >= 0.80
            ),
        }
        gate = summary["autonomous_gate"]
        gate["all_conditions_met"] = all(
            value for key, value in gate.items() if isinstance(value, bool)
        )
        summary["autonomous_gate_passed"] = bool(gate["all_conditions_met"])
    atomic_write_json(summary_output, summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
