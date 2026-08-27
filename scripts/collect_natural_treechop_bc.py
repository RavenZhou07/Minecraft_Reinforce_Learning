"""Collect v9.6 teacher contact-owner demonstrations for behaviour cloning.

The collector runs the frozen ``terrain_route_drop_completion_v9_6`` teacher
with the ``f3_raycast`` diagnostic sensor and records only the steps during
which the contact controller owns the action. Every contact sample stores
its causal four-frame POV stack and the previously executed discrete action;
teacher-only audit metadata (contact state, raycast flags, recovery
activity, transition reasons) is stored in separate arrays that the trainer
must not load as model input.

Outputs are written atomically (temporary file then replace) after every
episode, and existing outputs are never overwritten without an explicit
flag. Minecraft runs as a single sequential instance.
"""

import argparse
import csv
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import psutil

from mc_rl.natural_bc_runner import NaturalContactRunner
from mc_rl.resource_adapters import TreeResourceAdapter
from mc_rl.search_policy import CandidateSearchPolicy, SearchConfig
from mc_rl.telemetry_treechop_env import make_telemetry_treechop_env
from mc_rl.trunk_contact import CONTACT_PROFILE_TERRAIN_ROUTE_DROP_COMPLETION_V9_6

FRAME_STACK = 4
BANNED_SEED_LOW = 16500
BANNED_SEED_HIGH = 16819


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=80)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=16900)
    parser.add_argument(
        "--output",
        default="logs/find_tree/natural_treechop_bc_v1_train_16900_80.npz",
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


def dataset_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, extrasaction="ignore"
        )
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


def contact_audit_columns(policy: CandidateSearchPolicy) -> Dict[str, Any]:
    diagnostics = policy.contact_diagnostics()
    counters = diagnostics.get("counters", {})
    transitions = diagnostics.get("transition_records", [])
    latest_reason = transitions[-1]["reason"] if transitions else ""
    state = policy.contact_state or ""
    coordinate_recovery_active = state in (
        "COORDINATE_RECOVER",
        "POST_RECOVERY_VERIFY",
    )
    drop_recovery_active = state in ("BLOCK_DISAPPEARED", "DROP_RECOVERY")
    return {
        "contact_state": state,
        "coordinate_recovery_active": coordinate_recovery_active,
        "drop_recovery_active": drop_recovery_active,
        "transition_reason": latest_reason,
        "attempt_id": diagnostics.get("attempt_id", ""),
        "counters_snapshot": counters,
    }


def main():
    args = parse_args()
    if args.episodes <= 0 or args.max_steps <= 0:
        raise ValueError("episodes and max-steps must be positive")
    seeds = [args.seed + index for index in range(args.episodes)]
    for seed in seeds:
        if BANNED_SEED_LOW <= seed <= BANNED_SEED_HIGH:
            raise ValueError(
                "seed {} is inside the banned development/gate range "
                "{}-{}".format(seed, BANNED_SEED_LOW, BANNED_SEED_HIGH)
            )
    output = Path(args.output)
    episodes_output = output.with_suffix(".episodes.csv")
    summary_output = output.with_suffix(".summary.json")
    protected = (output, episodes_output, summary_output)
    existing = [path for path in protected if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "refusing to overwrite existing BC collection output: {}".format(
                ", ".join(str(path) for path in existing)
            )
        )

    samples: Dict[str, List[Any]] = {
        "pov": [],
        "action": [],
        "previous_action": [],
        "episode": [],
        "episode_seed": [],
        "episode_step": [],
        "contact_step": [],
        "attempt_id": [],
        "episode_success": [],
        # Teacher-only audit metadata; the trainer must not load these as
        # model inputs.
        "audit_contact_state": [],
        "audit_decision_contact_state": [],
        "audit_resulting_contact_state": [],
        "audit_raycast_is_log": [],
        "audit_raycast_in_range": [],
        "audit_coordinate_recovery_active": [],
        "audit_drop_recovery_active": [],
        "audit_transition_reason": [],
    }
    episode_rows: List[Dict[str, Any]] = []
    successes = 0
    started_at = time.perf_counter()

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
            runner = NaturalContactRunner(policy, None, "teacher", FRAME_STACK)
            done = False
            info: Dict[str, Any] = {}
            step = 0
            episode_samples = 0
            episode_action_counts: Counter = Counter()
            contact_attempts = 0
            contact_state_counts: Counter = Counter()
            while not done:
                executed, record = runner.act(observation)
                raycast = observation.get("raycast", {})
                audit = contact_audit_columns(policy)
                if record["contact_active"]:
                    if record["attempt_id"] > contact_attempts:
                        contact_attempts = record["attempt_id"]
                    stack = runner.frame_history.current_stack()
                    samples["pov"].append(np.asarray(stack, dtype=np.uint8))
                    samples["action"].append(int(executed))
                    samples["previous_action"].append(
                        int(runner.previous_action)
                    )
                    samples["episode"].append(episode_index + 1)
                    samples["episode_seed"].append(seed)
                    samples["episode_step"].append(step)
                    samples["contact_step"].append(
                        runner.contact_steps
                    )
                    samples["attempt_id"].append(record["attempt_id"])
                    samples["episode_success"].append(0)
                    samples["audit_contact_state"].append(
                        audit["contact_state"]
                    )
                    samples["audit_decision_contact_state"].append(
                        record["contact_state_before"]
                    )
                    samples["audit_resulting_contact_state"].append(
                        record["contact_state_after"]
                    )
                    samples["audit_raycast_is_log"].append(
                        float(raycast.get("is_log", 0.0))
                    )
                    samples["audit_raycast_in_range"].append(
                        float(raycast.get("in_range", 0.0))
                    )
                    samples["audit_coordinate_recovery_active"].append(
                        bool(audit["coordinate_recovery_active"])
                    )
                    samples["audit_drop_recovery_active"].append(
                        bool(audit["drop_recovery_active"])
                    )
                    samples["audit_transition_reason"].append(
                        audit["transition_reason"]
                    )
                    episode_samples += 1
                    episode_action_counts[int(executed)] += 1
                    contact_state_counts[audit["contact_state"]] += 1
                next_observation, reward, done, info = env.step(executed)
                policy.observe_transition(
                    executed, next_observation, reward, done, info
                )
                runner.observe_transition(executed)
                observation = next_observation
                step += 1

            success = bool(info.get("success", False))
            if success:
                successes += 1
            final_contact = policy.contact_diagnostics()
            final_counters = final_contact.get("counters", {})
            # Backfill the terminal success flag for this episode's samples.
            for index in range(len(samples["episode"]) - episode_samples, len(samples["episode"])):
                if samples["episode"][index] == episode_index + 1:
                    samples["episode_success"][index] = int(success)
            episode_rows.append(
                {
                    "episode": episode_index + 1,
                    "seed": seed,
                    "success": success,
                    "steps": step,
                    "contact_attempts": contact_attempts,
                    "contact_steps": runner.contact_steps,
                    "contact_samples": episode_samples,
                    "terrain_route_recoveries": (
                        policy.terrain_route_recovery_attempts
                    ),
                    "terrain_route_recovery_successes": (
                        policy.terrain_route_recovery_successes
                    ),
                    "coordinate_recoveries": final_counters.get(
                        "coordinate_recoveries", 0
                    ),
                    "exact_log_rescan_successes": final_counters.get(
                        "exact_log_rescan_successes", 0
                    ),
                    "block_disappearances": final_counters.get(
                        "block_disappearances", 0
                    ),
                    "drop_recovery_attempts": final_counters.get(
                        "drop_recovery_attempts", 0
                    ),
                    "pickups_after_disappearance": final_counters.get(
                        "pickup_after_disappearance", 0
                    ),
                    "contact_owner_mismatches": policy.contact_owner_mismatches,
                    "action_counts": json.dumps(
                        dict(sorted(episode_action_counts.items()))
                    ),
                    "contact_state_counts": json.dumps(
                        dict(sorted(contact_state_counts.items()))
                    ),
                }
            )
            arrays = {
                key: (np.asarray(value) if key != "pov" else np.asarray(value, dtype=np.uint8))
                for key, value in samples.items()
            }
            arrays["teacher_profile"] = np.array(args.contact_profile)
            arrays["frame_stack"] = np.array(FRAME_STACK)
            atomic_write_npz(output, arrays)
            atomic_write_rows(episodes_output, episode_rows)
            print(
                "episode={}/{} seed={} success={} steps={} contact_samples={} "
                "attempts={}".format(
                    episode_index + 1,
                    args.episodes,
                    seed,
                    success,
                    step,
                    episode_samples,
                    contact_attempts,
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

    total_samples = len(samples["action"])
    successful_samples = int(
        np.sum(np.asarray(samples["episode_success"], dtype=np.int64))
    )
    action_counts = Counter(int(a) for a in samples["action"])
    successful_action_counts = Counter(
        int(action)
        for action, success in zip(
            samples["action"], samples["episode_success"]
        )
        if success
    )
    state_counts = Counter(samples["audit_contact_state"])
    coordinate_recovery_samples = int(
        np.sum(np.asarray(samples["audit_coordinate_recovery_active"]))
    )
    drop_recovery_samples = int(
        np.sum(np.asarray(samples["audit_drop_recovery_active"])
    ))
    successful_episodes_rows = [row for row in episode_rows if row["success"]]
    summary = {
        "purpose": "v9.6 teacher contact-owner demonstrations for natural Treechop behaviour cloning",
        "teacher_profile": args.contact_profile,
        "teacher_sensor_profile": "f3_raycast",
        "mode": "teacher",
        "episodes": args.episodes,
        "seed_range": [seeds[0], seeds[-1]],
        "max_steps": args.max_steps,
        "successes": successes,
        "failures": args.episodes - successes,
        "total_contact_samples": total_samples,
        "successful_episode_samples": successful_samples,
        "excluded_failure_samples": total_samples - successful_samples,
        "exclusion_reason": "first-version training uses only contact trajectories from successful episodes",
        "contact_attempts_total": sum(
            row["contact_attempts"] for row in episode_rows
        ),
        "action_counts": {
            str(action): count for action, count in sorted(action_counts.items())
        },
        "action_fractions": {
            str(action): (count / total_samples if total_samples else 0.0)
            for action, count in sorted(action_counts.items())
        },
        "successful_episode_action_counts": {
            str(action): count
            for action, count in sorted(successful_action_counts.items())
        },
        "contact_state_counts": {
            str(state): count for state, count in sorted(state_counts.items())
        },
        "coordinate_recovery_samples": coordinate_recovery_samples,
        "drop_recovery_samples": drop_recovery_samples,
        "successful_episodes_with_coordinate_recovery": sum(
            row["coordinate_recoveries"] > 0
            for row in successful_episodes_rows
        ),
        "successful_episodes_with_exact_rescan": sum(
            row["exact_log_rescan_successes"] > 0
            for row in successful_episodes_rows
        ),
        "successful_episodes_with_terrain_recovery": sum(
            row["terrain_route_recoveries"] > 0
            for row in successful_episodes_rows
        ),
        "successful_episodes_with_drop_recovery": sum(
            row["drop_recovery_attempts"] > 0
            for row in successful_episodes_rows
        ),
        "successful_episodes_with_block_disappearance": sum(
            row["block_disappearances"] > 0
            for row in successful_episodes_rows
        ),
        "contact_owner_mismatches": sum(
            row["contact_owner_mismatches"] for row in episode_rows
        ),
        "student_input_manifest": [
            "pov_frame_stack_4",
            "previous_action_one_hot_14",
        ],
        "audit_only_fields": [
            "audit_contact_state",
            "audit_decision_contact_state",
            "audit_resulting_contact_state",
            "audit_raycast_is_log",
            "audit_raycast_in_range",
            "audit_coordinate_recovery_active",
            "audit_drop_recovery_active",
            "audit_transition_reason",
        ],
        "frame_stack": FRAME_STACK,
        "elapsed_seconds": round(time.perf_counter() - started_at, 3),
    }
    atomic_write_json(summary_output, summary)
    summary["dataset_sha256"] = dataset_sha256(output)
    atomic_write_json(summary_output, summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
