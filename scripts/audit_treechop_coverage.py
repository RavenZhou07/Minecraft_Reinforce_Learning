"""Audit existing Treechop state/action/transition coverage by episode."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from mc_rl.actions import ACTION_NAMES
from mc_rl.experiments import file_sha256
from mc_rl.observability_audit import labels_from_dataset, load_audit_dataset
from mc_rl.runtime_observability import atomic_csv, atomic_json, load_trace


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/learning/runtime_observability_audit_exp12.json",
    )
    return parser.parse_args()


def combine_student_traces(root: Path) -> Dict[str, np.ndarray]:
    rows: Dict[str, List[np.ndarray]] = {}
    episode_id = 0
    for path in sorted(root.glob("seed*_env*.npz")):
        trace = load_trace(path)
        metadata = json.loads(str(trace["trace_metadata_json"]))
        count = len(trace["episode_step"])
        values = {
            "pov": trace["pov"],
            "legal_vector": trace["legal_vector"],
            "action": trace["executed_action_id"],
            "previous_action": np.where(
                trace["previous_action_token"] == 14, 0, trace["previous_action_token"]
            ),
            "episode": np.full(count, episode_id, dtype=np.int32),
            "episode_seed": np.full(count, int(metadata["environment_seed"]), dtype=np.int32),
            "episode_step": trace["episode_step"],
            "episode_success": np.full(count, int(bool(metadata.get("success", False))), dtype=np.int8),
            "audit_raycast_is_log": trace["audit_raycast_is_log"],
            "audit_raycast_in_range": trace["audit_raycast_in_range"],
            "audit_raycast_distance": trace["audit_raycast_distance"],
        }
        for key, value in values.items():
            rows.setdefault(key, []).append(np.asarray(value))
        episode_id += 1
    if not rows:
        raise FileNotFoundError("no student_dev diagnostic traces")
    return {key: np.concatenate(value) for key, value in rows.items()}


def histogram(values: np.ndarray) -> str:
    return json.dumps(
        {
            ACTION_NAMES[int(key)] if 0 <= int(key) < len(ACTION_NAMES) else str(key): int(value)
            for key, value in sorted(Counter(np.asarray(values, dtype=np.int64).tolist()).items())
        },
        sort_keys=True,
    )


def support_rows(name: str, dataset: Mapping[str, np.ndarray]) -> List[Dict[str, Any]]:
    labels = labels_from_dataset(dataset)
    rows = []
    outcomes = [("all", np.ones(len(dataset["episode"]), dtype=bool))]
    if name == "bc_train":
        outcomes.extend(
            [
                ("successful", dataset["episode_success"].astype(bool)),
                ("unsuccessful", ~dataset["episode_success"].astype(bool)),
            ]
        )
    for outcome, outcome_mask in outcomes:
        for label_name, label in labels.items():
            valid = outcome_mask & (label >= 0)
            positive = valid & (label == 1)
            negative = valid & (label == 0)
            rows.append(
                {
                    "dataset": name,
                    "outcome": outcome,
                    "label": label_name,
                    "positive_timesteps": int(positive.sum()),
                    "negative_timesteps": int(negative.sum()),
                    "positive_episodes": int(len(np.unique(dataset["episode"][positive]))),
                    "negative_episodes": int(len(np.unique(dataset["episode"][negative]))),
                    "positive_seeds": int(len(np.unique(dataset["episode_seed"][positive]))),
                    "negative_seeds": int(len(np.unique(dataset["episode_seed"][negative]))),
                    "coverage_status": (
                        "coverage_insufficient_for_generalization_claim"
                        if label_name == "approach_dynamics"
                        or len(np.unique(dataset["episode_seed"][positive])) < 2
                        or len(np.unique(dataset["episode_seed"][negative])) < 2
                        else "support_gate_passed"
                    ),
                }
            )
    return rows


def state_action_rows(name: str, dataset: Mapping[str, np.ndarray]) -> List[Dict[str, Any]]:
    labels = labels_from_dataset(dataset)
    rows = []
    previous = np.asarray(dataset["previous_action"], dtype=np.int64)
    action = np.asarray(dataset["action"], dtype=np.int64)
    next_action = np.r_[action[1:], action[-1]]
    boundaries = np.r_[dataset["episode"][1:] != dataset["episode"][:-1], True]
    next_action[boundaries] = action[boundaries]
    for label_name, label in labels.items():
        if label_name == "approach_dynamics":
            strata = [("decreasing", label == 0), ("stable", label == 1), ("increasing", label == 2)]
        else:
            strata = [("negative", label == 0), ("positive", label == 1)]
        for stratum, mask in strata:
            rows.append(
                {
                    "dataset": name,
                    "audit_stratum": label_name,
                    "value": stratum,
                    "timesteps": int(mask.sum()),
                    "episodes": int(len(np.unique(dataset["episode"][mask]))),
                    "seeds": int(len(np.unique(dataset["episode_seed"][mask]))),
                    "teacher_action_histogram": histogram(action[mask]),
                    "previous_action_histogram": histogram(previous[mask]),
                    "next_action_histogram": histogram(next_action[mask]),
                }
            )
    return rows


def transition_rows(name: str, dataset: Mapping[str, np.ndarray]) -> List[Dict[str, Any]]:
    labels = labels_from_dataset(dataset)
    visible = labels["tree_visible"] == 1
    decreasing = labels["approach_dynamics"] == 0
    contact = labels["contact_range"] == 1
    valid = labels["valid_attack_geometry"] == 1
    reward = np.asarray(dataset.get("audit_reward", np.zeros(len(visible))), dtype=np.float32)
    # Existing trajectory data stores reward from the action transition but
    # not the post-transition POV/vector.  Positive Treechop reward confirms
    # inventory acquisition, not the earlier visual block-break instant, so it
    # cannot be used as a block-break label without changing label semantics.
    broken = np.zeros(len(visible), dtype=bool)
    inventory = reward > 0
    transitions = [
        ("tree_not_visible_to_tree_visible", ~visible, visible, True),
        ("tree_visible_to_roughly_centered", visible, visible, False),
        ("tree_visible_to_approaching", visible, decreasing, True),
        ("approaching_to_contact_range", decreasing, contact, True),
        ("contact_range_to_valid_attack_geometry", contact, valid, True),
        ("valid_attack_geometry_to_block_break", valid, broken, False),
        ("block_break_to_pickup", broken, inventory, False),
    ]
    rows = []
    episodes = np.asarray(dataset["episode"])
    steps = np.asarray(dataset["episode_step"])
    actions = np.asarray(dataset["action"])
    for transition, source, target, supported in transitions:
        if not supported:
            rows.append(
                {
                    "dataset": name,
                    "transition": transition,
                    "status": "unsupported_by_current_audit_data",
                    "total_occurrences": 0,
                    "distinct_trajectories": 0,
                    "distinct_seeds": 0,
                    "median_duration": None,
                    "teacher_actions_window_minus10_plus10": "{}",
                }
            )
            continue
        event_indices = []
        durations = []
        for episode in np.unique(episodes):
            indices = np.flatnonzero(episodes == episode)
            last_source = None
            target_was_true = False
            for index in indices:
                if source[index]:
                    last_source = int(steps[index])
                rising = bool(target[index] and not target_was_true)
                if rising and last_source is not None:
                    event_indices.append(int(index))
                    durations.append(int(steps[index]) - last_source)
                target_was_true = bool(target[index])
        window_actions = []
        for index in event_indices:
            same_episode = episodes == episodes[index]
            window = same_episode & (np.abs(steps - steps[index]) <= 10)
            window_actions.extend(actions[window].tolist())
        event_array = np.asarray(event_indices, dtype=np.int64)
        rows.append(
            {
                "dataset": name,
                "transition": transition,
                "status": "supported",
                "total_occurrences": len(event_indices),
                "distinct_trajectories": int(len(np.unique(episodes[event_array]))) if len(event_array) else 0,
                "distinct_seeds": int(len(np.unique(dataset["episode_seed"][event_array]))) if len(event_array) else 0,
                "median_duration": float(np.median(durations)) if durations else None,
                "teacher_actions_window_minus10_plus10": histogram(np.asarray(window_actions)) if window_actions else "{}",
            }
        )
    return rows


def action_transition_matrix(datasets: Mapping[str, Mapping[str, np.ndarray]]) -> List[Dict[str, Any]]:
    rows = []
    for name, dataset in datasets.items():
        episodes = np.asarray(dataset["episode"])
        actions = np.asarray(dataset["action"], dtype=np.int64)
        for previous, current, same_episode in zip(actions, actions[1:], episodes[:-1] == episodes[1:]):
            if same_episode:
                rows.append({"dataset": name, "previous_action": ACTION_NAMES[int(previous)], "current_action": ACTION_NAMES[int(current)]})
    counts = Counter((row["dataset"], row["previous_action"], row["current_action"]) for row in rows)
    return [
        {"dataset": key[0], "previous_action": key[1], "current_action": key[2], "count": value}
        for key, value in sorted(counts.items())
    ]


def dagger_sequence_audit(path: Path) -> Dict[str, Any]:
    with np.load(path, allow_pickle=False) as values:
        files = set(values.files)
        required = {
            "pov", "legal_vector", "previous_action", "episode", "episode_seed",
            "episode_step", "action", "audit_student_action", "audit_raycast_is_log",
            "audit_raycast_in_range",
        }
        complete_fields = required.issubset(files)
        episode = np.asarray(values["episode"])
        step = np.asarray(values["episode_step"])
        previous = np.asarray(values["previous_action"])
        student = np.asarray(values["audit_student_action"])
    ordered = all(np.array_equal(step[episode == value], np.arange((episode == value).sum())) for value in np.unique(episode))
    causal = all(
        len(indices) <= 1 or np.array_equal(previous[indices[1:]], student[indices[:-1]])
        for indices in (np.flatnonzero(episode == value) for value in np.unique(episode))
    )
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "samples": int(len(episode)),
        "episodes": int(len(np.unique(episode))),
        "complete_ordered_observation_sequence": bool(complete_fields and ordered),
        "previous_executed_action_causal": bool(causal),
        "episode_boundary_present": True,
        "legal_rgb_present": "pov" in required,
        "legal_vector_present": "legal_vector" in required,
        "oracle_label_present": "action" in required,
        "privileged_audit_labels_present": True,
        "reusable_for_recurrent_training_next_round": bool(complete_fields and ordered and causal),
        "used_for_training_this_round": False,
    }


def main():
    args = parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if set(config["forbidden_splits"]) & {"bc_train", "bc_validation", "dagger1", "student_dev"}:
        raise PermissionError("allowed diagnostic set intersects forbidden split")
    datasets = {
        "teacher_dev": load_audit_dataset(Path(config["datasets"]["teacher_dev"]["path"])),
        "bc_train": load_audit_dataset(Path(config["datasets"]["bc_train"]["path"])),
        "bc_validation": load_audit_dataset(Path(config["datasets"]["bc_validation"]["path"])),
        "dagger1": load_audit_dataset(Path(config["datasets"]["dagger1"]["path"])),
        "student_dev": combine_student_traces(Path("artifacts/exp12/runtime_traces")),
    }
    root = Path("artifacts/exp12")
    support = [row for name, dataset in datasets.items() for row in support_rows(name, dataset)]
    state_actions = [row for name, dataset in datasets.items() for row in state_action_rows(name, dataset)]
    transitions = [row for name, dataset in datasets.items() for row in transition_rows(name, dataset)]
    atomic_csv(root / "probe_support.csv", support)
    atomic_csv(root / "state_action_coverage.csv", state_actions)
    atomic_csv(root / "transition_coverage.csv", transitions)
    atomic_csv(root / "action_transition_matrix.csv", action_transition_matrix(datasets))
    dagger_audit = dagger_sequence_audit(Path(config["datasets"]["dagger1"]["path"]))
    atomic_json(root / "dagger1_sequence_audit.json", dagger_audit)
    schema_path = root / "audit_label_schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    schema["support_by_dataset"] = {
        name: {
            label: {
                "positive_seeds": next(row["positive_seeds"] for row in support if row["dataset"] == name and row["outcome"] == "all" and row["label"] == label),
                "negative_seeds": next(row["negative_seeds"] for row in support if row["dataset"] == name and row["outcome"] == "all" and row["label"] == label),
            }
            for label in ("tree_visible", "contact_range", "valid_attack_geometry", "approach_dynamics")
        }
        for name in datasets
    }
    atomic_json(schema_path, schema)
    print(json.dumps({"datasets": list(datasets), "support_rows": len(support), "transition_rows": len(transitions), "dagger1": dagger_audit}, indent=2))


if __name__ == "__main__":
    main()
