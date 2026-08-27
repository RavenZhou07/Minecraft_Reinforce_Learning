"""Collect complete bootstrap-teacher trajectories for end-to-end BC."""

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import psutil

from mc_rl.experiments import append_experiment, file_sha256, seeds_for_split
from mc_rl.learning_observation import (
    STUDENT_OBSERVATION_SCHEMA_VERSION,
    STUDENT_VECTOR_NAMES,
    LegalObservationAdapter,
    student_input_manifest,
)
from mc_rl.natural_treechop_bc import coarse_teacher_phase
from mc_rl.natural_treechop_runtime import make_bootstrap_teacher
from mc_rl.telemetry_treechop_env import make_telemetry_treechop_env
from mc_rl.trunk_contact import CONTACT_PROFILE_EMERGENCY_TARGET_PREEMPTION_V9_11


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="bc_train")
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--manifest", default="configs/seeds/natural_treechop_v1.json")
    parser.add_argument(
        "--output",
        default="logs/learning/natural_treechop_e2e_bc_train_smoke.npz",
    )
    parser.add_argument(
        "--contact-profile",
        default=CONTACT_PROFILE_EMERGENCY_TARGET_PREEMPTION_V9_11,
    )
    parser.add_argument("--flush-every", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--experiment-id", default="e2e_dataset_collection")
    parser.add_argument("--hypothesis", default="The bootstrap teacher yields complete, leakage-safe trajectories for an end-to-end student.")
    return parser.parse_args()


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


def dataset_arrays(samples: Dict[str, List[Any]], metadata: Dict[str, Any]):
    arrays = {
        "pov": np.asarray(samples["pov"], dtype=np.uint8),
        "legal_vector": np.asarray(samples["legal_vector"], dtype=np.float32),
        "action": np.asarray(samples["action"], dtype=np.int32),
        "previous_action": np.asarray(samples["previous_action"], dtype=np.int32),
        "episode": np.asarray(samples["episode"], dtype=np.int32),
        "episode_seed": np.asarray(samples["episode_seed"], dtype=np.int32),
        "episode_step": np.asarray(samples["episode_step"], dtype=np.int32),
        "episode_success": np.asarray(samples["episode_success"], dtype=np.int8),
        "source": np.asarray(samples["source"]),
        "audit_search_state": np.asarray(samples["audit_search_state"]),
        "audit_contact_state": np.asarray(samples["audit_contact_state"]),
        "audit_action_source": np.asarray(samples["audit_action_source"]),
        "audit_coarse_phase": np.asarray(samples["audit_coarse_phase"]),
        "audit_raycast_is_log": np.asarray(samples["audit_raycast_is_log"], dtype=np.int8),
        "audit_raycast_in_range": np.asarray(samples["audit_raycast_in_range"], dtype=np.int8),
        "audit_raycast_distance": np.asarray(samples["audit_raycast_distance"], dtype=np.float32),
        "audit_reward": np.asarray(samples["audit_reward"], dtype=np.float32),
    }
    arrays.update({key: np.asarray(value) for key, value in metadata.items()})
    return arrays


def atomic_npz(path: Path, arrays: Dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def main():
    args = parse_args()
    if args.episodes <= 0 or args.max_steps <= 0 or args.flush_every <= 0:
        raise ValueError("episodes, max-steps and flush-every must be positive")
    seeds = seeds_for_split(args.split, args.manifest, limit=args.episodes)
    output = Path(args.output)
    episodes_output = output.with_suffix(".episodes.csv")
    summary_output = output.with_suffix(".summary.json")
    existing = [path for path in (output, episodes_output, summary_output) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError("refusing to overwrite: {}".format(existing))

    samples = {key: [] for key in (
        "pov", "legal_vector", "action", "previous_action", "episode",
        "episode_seed", "episode_step", "episode_success", "source",
        "audit_search_state", "audit_contact_state", "audit_action_source",
        "audit_coarse_phase", "audit_raycast_is_log", "audit_raycast_in_range",
        "audit_raycast_distance", "audit_reward",
    )}
    rows: List[Dict[str, Any]] = []
    metadata = {
        "dataset_version": "natural_treechop_full_trajectory_v1",
        "observation_schema": STUDENT_OBSERVATION_SCHEMA_VERSION,
        "student_vector_names": np.asarray(STUDENT_VECTOR_NAMES),
        "student_input_manifest": np.asarray(student_input_manifest(4, 8)),
        "seed_manifest": str(args.manifest),
        "seed_split": str(args.split),
        "teacher_profile": str(args.contact_profile),
        "success_definition": "inventory natural log delta >= 1",
    }
    started = time.perf_counter()
    env = make_telemetry_treechop_env(
        seed=seeds[0], max_episode_steps=args.max_steps, include_raycast=True
    )
    try:
        for episode_index, seed in enumerate(seeds, start=1):
            env.seed(seed)
            observation = env.reset()
            legal_adapter = LegalObservationAdapter(args.max_steps)
            legal_adapter.reset(observation)
            teacher = make_bootstrap_teacher(args.max_steps, args.contact_profile)
            teacher.reset(episode=episode_index)
            previous_action = 0
            done = False
            info: Dict[str, Any] = {}
            step = 0
            episode_indices: List[int] = []
            action_counts: Counter = Counter()
            phase_counts: Counter = Counter()
            reward_total = 0.0
            first_interaction = None
            episode_started = time.perf_counter()
            while not done:
                legal = (
                    legal_adapter.reset(observation)
                    if step == 0
                    else legal_adapter.adapt(observation, step)
                )
                search_before = str(getattr(teacher.state, "value", teacher.state))
                contact_before = str(teacher.contact_state or "")
                action = int(teacher.act(observation))
                action_source = str(teacher.last_action_source)
                phase = coarse_teacher_phase(search_before, contact_before, action_source)
                raycast = observation["raycast"]
                sample_index = len(samples["action"])
                episode_indices.append(sample_index)
                samples["pov"].append(legal.pov)
                samples["legal_vector"].append(legal.vector)
                samples["action"].append(action)
                samples["previous_action"].append(previous_action)
                samples["episode"].append(episode_index)
                samples["episode_seed"].append(seed)
                samples["episode_step"].append(step)
                samples["episode_success"].append(0)
                samples["source"].append("teacher")
                samples["audit_search_state"].append(search_before)
                samples["audit_contact_state"].append(contact_before)
                samples["audit_action_source"].append(action_source)
                samples["audit_coarse_phase"].append(phase)
                samples["audit_raycast_is_log"].append(bool(raycast["is_log"]))
                samples["audit_raycast_in_range"].append(bool(raycast["in_range"]))
                samples["audit_raycast_distance"].append(float(raycast["distance"]))
                next_observation, reward, done, info = env.step(action)
                teacher.observe_transition(action, next_observation, reward, done, info)
                samples["audit_reward"].append(float(reward))
                action_counts[action] += 1
                phase_counts[phase] += 1
                if first_interaction is None and action in (7, 8):
                    first_interaction = step
                reward_total += float(reward)
                previous_action = action
                observation = next_observation
                step += 1
            success = bool(info.get("success", False))
            for index in episode_indices:
                samples["episode_success"][index] = int(success)
            row = {
                "episode": episode_index,
                "seed": seed,
                "success": success,
                "steps": step,
                "timeout": bool(not success and step >= args.max_steps),
                "inventory_log_delta": info.get("inventory_log_delta"),
                "success_source": info.get("success_source", ""),
                "reward_total": reward_total,
                "first_meaningful_interaction_step": first_interaction,
                "action_counts": json.dumps(dict(sorted(action_counts.items()))),
                "phase_counts": json.dumps(dict(sorted(phase_counts.items()))),
                "runtime_seconds": round(time.perf_counter() - episode_started, 3),
            }
            rows.append(row)
            atomic_csv(episodes_output, rows)
            if episode_index % args.flush_every == 0 or episode_index == len(seeds):
                atomic_npz(output, dataset_arrays(samples, metadata))
            print(
                "collection episode={}/{} seed={} success={} steps={}".format(
                    episode_index, len(seeds), seed, success, step
                ),
                flush=True,
            )
    finally:
        try:
            env.close()
        except psutil.NoSuchProcess as error:
            print("WARNING: Minecraft already exited during close: {}".format(error))

    successes = sum(bool(row["success"]) for row in rows)
    summary = {
        "dataset_version": metadata["dataset_version"],
        "output": str(output),
        "dataset_sha256": file_sha256(output),
        "seed_manifest": args.manifest,
        "seed_split": args.split,
        "seeds": seeds,
        "teacher_profile": args.contact_profile,
        "episodes": len(rows),
        "successes": successes,
        "success_rate": successes / len(rows),
        "samples": len(samples["action"]),
        "successful_episode_samples": sum(samples["episode_success"]),
        "student_input_manifest": list(metadata["student_input_manifest"]),
        "train_only_privileged_arrays": [
            key for key in dataset_arrays(samples, metadata) if key.startswith("audit_")
        ],
        "success_definition": metadata["success_definition"],
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    atomic_json(summary_output, summary)
    append_experiment(
        {
            "experiment_id": args.experiment_id,
            "hypothesis": args.hypothesis,
            "config": {
                "script": "scripts/collect_natural_treechop_trajectories.py",
                "teacher_profile": args.contact_profile,
                "max_steps": args.max_steps,
            },
            "seed_manifest": args.manifest,
            "seed_split": args.split,
            "dataset": {"path": str(output), "sha256": summary["dataset_sha256"]},
            "metrics": {
                "episodes": len(rows),
                "successes": successes,
                "samples": summary["samples"],
            },
            "runtime_seconds": summary["elapsed_seconds"],
            "conclusion": "Full trajectories collected; dataset is ready for smoke training.",
            "status": "kept",
        }
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
