"""Frozen-policy collapse causality, representation probes, and OOD audit."""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import cv2
import numpy as np
import torch

from mc_rl.actions import ACTION_NAMES
from mc_rl.natural_treechop_bc import ACTION_CLASSES, balanced_accuracy
from mc_rl.observability_audit import (
    FixedLogisticProbe,
    FixedRidgeProbe,
    TrainOnlyPCA,
    TrainOnlyStandardizer,
    assert_disjoint_episode_splits,
    binary_metrics,
    feature_spaces,
    knn_probabilities,
    labels_from_dataset,
    load_audit_dataset,
    regression_metrics,
)
from mc_rl.recurrent_treechop_bc import START_ACTION_TOKEN, RecurrentTreechopPolicy
from mc_rl.runtime_observability import (
    atomic_csv,
    atomic_json,
    js_divergence,
    load_trace,
)
from scripts.audit_treechop_coverage import combine_student_traces


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/learning/runtime_observability_audit_exp12.json",
    )
    parser.add_argument("--stage", choices=("collapse", "probes", "all"), default="all")
    return parser.parse_args()


def entropy(probabilities: np.ndarray) -> float:
    values = np.asarray(probabilities, dtype=np.float64)
    return -float(np.sum(values * np.log(np.maximum(values, 1e-12))))


def policy_step(policy, pov, vector, token, hidden):
    action, probabilities, next_hidden, diagnostics = policy.predict_step_with_diagnostics(
        pov, vector, int(token), hidden
    )
    return action, probabilities, next_hidden, diagnostics["logits"]


def teacher_hidden_before(dataset: Mapping[str, np.ndarray], policy: RecurrentTreechopPolicy) -> np.ndarray:
    rows = np.zeros((len(dataset["episode"]), policy.architecture.hidden_size), dtype=np.float32)
    hidden = None
    previous_episode = None
    for index, episode in enumerate(dataset["episode"]):
        if previous_episode != int(episode):
            hidden = None
            token = START_ACTION_TOKEN
        else:
            rows[index] = hidden.detach().cpu().numpy().reshape(-1)
            token = int(dataset["previous_action"][index])
        _, _, hidden, _ = policy_step(
            policy, dataset["pov"][index], dataset["legal_vector"][index], token, hidden
        )
        previous_episode = int(episode)
    return rows


def tensor_hidden(values: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(values, dtype=np.float32)).reshape(1, 1, -1)


def select_even(indices: np.ndarray, count: int) -> np.ndarray:
    values = np.asarray(indices, dtype=np.int64)
    if len(values) <= count:
        return values
    return values[np.linspace(0, len(values) - 1, count, dtype=np.int64)]


def sampled_states(
    validation: Mapping[str, np.ndarray],
    policy: RecurrentTreechopPolicy,
    trace_root: Path,
) -> List[Dict[str, Any]]:
    states: List[Dict[str, Any]] = []
    hidden_before = teacher_hidden_before(validation, policy)
    phases = np.asarray(validation.get("audit_coarse_phase", np.asarray(["unknown"] * len(validation["episode"]))))
    for phase in np.unique(phases):
        for index in select_even(np.flatnonzero(phases == phase), 8):
            episode_indices = np.flatnonzero(validation["episode"] == validation["episode"][index])
            local = int(np.flatnonzero(episode_indices == index)[0])
            earlier_index = episode_indices[max(0, local - 20)]
            states.append(
                {
                    "source": "bc_validation",
                    "stratum": "phase:{}".format(phase),
                    "state_id": "validation:{}:{}".format(int(validation["episode_seed"][index]), int(validation["episode_step"][index])),
                    "pov": validation["pov"][index],
                    "vector": validation["legal_vector"][index],
                    "token": START_ACTION_TOKEN if local == 0 else int(validation["previous_action"][index]),
                    "hidden": hidden_before[index],
                    "earlier_hidden": hidden_before[earlier_index],
                }
            )
    for path in sorted(trace_root.glob("seed*_env*.npz")):
        trace = load_trace(path)
        metadata = json.loads(str(trace["trace_metadata_json"]))
        count = len(trace["episode_step"])
        hidden_after = np.asarray(trace["hidden"], dtype=np.float32)
        before = np.vstack([np.zeros((1, hidden_after.shape[1]), dtype=np.float32), hidden_after[:-1]])
        bands = {
            "early": np.arange(0, min(167, count)),
            "mid": np.arange(min(167, count), min(334, count)),
            "late": np.arange(min(334, count), count),
            "raycast_log": np.flatnonzero(trace["audit_raycast_is_log"]),
            "contact": np.flatnonzero(trace["audit_raycast_in_range"]),
        }
        for band, indices in bands.items():
            for index in select_even(indices, 4):
                states.append(
                    {
                        "source": "student_dev",
                        "stratum": band,
                        "state_id": "student:{}:{}:{}".format(metadata["checkpoint_seed"], metadata["environment_seed"], int(index)),
                        "pov": trace["pov"][index],
                        "vector": trace["legal_vector"][index],
                        "token": int(trace["previous_action_token"][index]),
                        "hidden": before[index],
                        "earlier_hidden": before[max(0, int(index) - 20)],
                    }
                )
    return states


def collapse_interventions(config: Dict[str, Any], validation: Mapping[str, np.ndarray]) -> Dict[str, Any]:
    root = Path("artifacts/exp12")
    checkpoint = next(item["path"] for item in config["checkpoints"] if int(item["training_seed"]) == 29)
    policy = RecurrentTreechopPolicy.load(checkpoint)
    states = sampled_states(validation, policy, root / "runtime_traces")
    sweep_rows = []
    per_state_token_js = []
    per_state_logit_variance = []
    for state in states:
        normal_hidden = tensor_hidden(state["hidden"])
        distributions = []
        for token in range(15):
            action, probabilities, _, logits = policy_step(
                policy, state["pov"], state["vector"], token, normal_hidden
            )
            ordered = np.argsort(probabilities)[::-1]
            distributions.append(probabilities)
            sweep_rows.append(
                {
                    "source": state["source"],
                    "stratum": state["stratum"],
                    "state_id": state["state_id"],
                    "normal_previous_token": state["token"],
                    "swept_previous_token": token,
                    "swept_previous_name": "START" if token == 14 else ACTION_NAMES[token],
                    "argmax_action": action,
                    "argmax_name": ACTION_NAMES[action],
                    "argmax_equals_previous_token": bool(token < 14 and action == token),
                    "argmax_changed_from_normal_token": None,
                    "entropy": entropy(probabilities),
                    "top1_probability": float(probabilities[ordered[0]]),
                    "top1_margin": float(probabilities[ordered[0]] - probabilities[ordered[1]]),
                    "logits": json.dumps(np.asarray(logits, dtype=float).round(7).tolist()),
                    "probabilities": json.dumps(np.asarray(probabilities, dtype=float).round(7).tolist()),
                }
            )
        normal_argmax = int(np.argmax(distributions[int(state["token"])]))
        state_rows = sweep_rows[-15:]
        for row in state_rows:
            row["argmax_changed_from_normal_token"] = bool(row["argmax_action"] != normal_argmax)
        pairwise = [
            js_divergence(distributions[a], distributions[b])
            for a in range(15)
            for b in range(a + 1, 15)
        ]
        per_state_token_js.append(float(np.mean(pairwise)))
        per_state_logit_variance.append(
            float(np.var(np.asarray([json.loads(row["logits"]) for row in state_rows]), axis=0).mean())
        )
    atomic_csv(root / "previous_action_sweep.csv", sweep_rows)

    hidden_rows = []
    hidden_js = []
    hidden_changes = []
    for index, state in enumerate(states):
        foreign = states[(index + max(1, len(states) // 2)) % len(states)]["hidden"]
        interventions = {
            "normal": state["hidden"],
            "zero": np.zeros_like(state["hidden"]),
            "earlier_same_episode": state["earlier_hidden"],
            "foreign_diagnostic_state": foreign,
        }
        outputs = {}
        for intervention, hidden in interventions.items():
            action, probabilities, _, logits = policy_step(
                policy, state["pov"], state["vector"], state["token"], tensor_hidden(hidden)
            )
            outputs[intervention] = (action, probabilities)
            hidden_rows.append(
                {
                    "source": state["source"],
                    "stratum": state["stratum"],
                    "state_id": state["state_id"],
                    "intervention": intervention,
                    "argmax_action": action,
                    "argmax_name": ACTION_NAMES[action],
                    "policy_entropy": entropy(probabilities),
                    "top1_probability": float(np.max(probabilities)),
                    "logits": json.dumps(np.asarray(logits, dtype=float).round(7).tolist()),
                    "probabilities": json.dumps(np.asarray(probabilities, dtype=float).round(7).tolist()),
                }
            )
        normal_action, normal_probs = outputs["normal"]
        for intervention in ("zero", "earlier_same_episode", "foreign_diagnostic_state"):
            action, probabilities = outputs[intervention]
            divergence = js_divergence(normal_probs, probabilities)
            hidden_js.append(divergence)
            hidden_changes.append(int(action != normal_action))
            hidden_rows[-4 + list(interventions).index(intervention)]["js_from_normal"] = divergence
            hidden_rows[-4 + list(interventions).index(intervention)]["argmax_changed_from_normal"] = bool(action != normal_action)
    atomic_csv(root / "hidden_intervention.csv", hidden_rows)

    replay_rows = []
    observation_js = []
    teacher_forced_ba = []
    free_ba = []
    for episode in np.unique(validation["episode"]):
        indices = np.flatnonzero(validation["episode"] == episode)
        labels = validation["action"][indices].astype(np.int64)
        predictions_by_mode = {}
        probabilities_by_mode = {}
        for mode in ("teacher_forced", "free_running", "always_start", "hidden_zero_each_step"):
            hidden = None
            previous_prediction = START_ACTION_TOKEN
            predictions = []
            probabilities_rows = []
            for local, index in enumerate(indices):
                if mode == "teacher_forced":
                    token = START_ACTION_TOKEN if local == 0 else int(labels[local - 1])
                elif mode == "always_start":
                    token = START_ACTION_TOKEN
                else:
                    token = previous_prediction
                input_hidden = None if mode == "hidden_zero_each_step" else hidden
                action, probabilities, next_hidden, _ = policy_step(
                    policy, validation["pov"][index], validation["legal_vector"][index], token, input_hidden
                )
                if mode != "hidden_zero_each_step":
                    hidden = next_hidden
                previous_prediction = action
                predictions.append(action)
                probabilities_rows.append(probabilities)
            predictions_by_mode[mode] = np.asarray(predictions, dtype=np.int64)
            probabilities_by_mode[mode] = np.asarray(probabilities_rows)
            prediction = predictions_by_mode[mode]
            transitions = int((prediction[1:] != prediction[:-1]).sum())
            longest = 1
            current = 1
            for left, right in zip(prediction, prediction[1:]):
                current = current + 1 if left == right else 1
                longest = max(longest, current)
            first_divergence = next((i for i, (a, b) in enumerate(zip(prediction, labels)) if a != b), None)
            terminal_fixed_point_start = len(prediction) - 1
            while (
                terminal_fixed_point_start > 0
                and prediction[terminal_fixed_point_start - 1] == prediction[-1]
            ):
                terminal_fixed_point_start -= 1
            replay_rows.append(
                {
                    "episode": int(episode),
                    "episode_seed": int(validation["episode_seed"][indices[0]]),
                    "mode": mode,
                    "action_accuracy": float((prediction == labels).mean()),
                    "balanced_accuracy": balanced_accuracy(prediction, labels, ACTION_CLASSES),
                    "first_prediction_divergence": first_divergence,
                    "steps_from_first_divergence_to_end": None if first_divergence is None else int(len(indices) - first_divergence),
                    "steps_from_first_divergence_to_fixed_point": (
                        None
                        if first_divergence is None
                        else max(0, int(terminal_fixed_point_start - first_divergence))
                    ),
                    "longest_repeated_predicted_action": longest,
                    "action_transition_count": transitions,
                    "argmax_copy_rate": float(np.mean(prediction[1:] == prediction[:-1])) if len(prediction) > 1 else 0.0,
                }
            )
        teacher_forced_ba.append(replay_rows[-4]["balanced_accuracy"])
        free_ba.append(replay_rows[-3]["balanced_accuracy"])
        probabilities = probabilities_by_mode["free_running"]
        observation_js.extend(js_divergence(a, b) for a, b in zip(probabilities, probabilities[1:]))
    atomic_csv(root / "forced_vs_free_replay.csv", replay_rows)

    token_change_rate = float(np.mean([row["argmax_changed_from_normal_token"] for row in sweep_rows]))
    token_copy_rate = float(np.mean([row["argmax_equals_previous_token"] for row in sweep_rows]))
    summary = {
        "sampled_states": len(states),
        "argmax_changed_by_previous_token_rate": token_change_rate,
        "argmax_equals_previous_token_rate": token_copy_rate,
        "mean_pairwise_js_across_previous_tokens": float(np.mean(per_state_token_js)),
        "mean_logit_variance_due_to_previous_token": float(np.mean(per_state_logit_variance)),
        "hidden_intervention_argmax_change_rate": float(np.mean(hidden_changes)),
        "mean_hidden_intervention_js": float(np.mean(hidden_js)),
        "teacher_forced_balanced_accuracy": float(np.mean(teacher_forced_ba)),
        "free_running_balanced_accuracy": float(np.mean(free_ba)),
        "teacher_forced_minus_free_balanced_accuracy": float(np.mean(teacher_forced_ba) - np.mean(free_ba)),
        "mean_consecutive_observation_policy_js": float(np.mean(observation_js)),
        "token_to_observation_js_ratio": float(np.mean(per_state_token_js) / max(np.mean(observation_js), 1e-12)),
        "hidden_to_observation_js_ratio": float(np.mean(hidden_js) / max(np.mean(observation_js), 1e-12)),
    }
    atomic_json(root / "collapse_causality_summary.json", summary)
    return summary


def bootstrap_balanced_interval(
    labels: np.ndarray, probabilities: np.ndarray, episodes: np.ndarray, repetitions: int = 200
) -> Tuple[float, float]:
    unique = np.unique(episodes)
    if len(unique) < 2:
        return float("nan"), float("nan")
    rng = np.random.RandomState(12)
    values = []
    for _ in range(repetitions):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([np.flatnonzero(episodes == episode) for episode in sampled])
        values.append(binary_metrics(labels[indices], probabilities[indices])["balanced_accuracy"])
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def prepare_feature_transforms(
    train_features: Mapping[str, np.ndarray],
    query_features: Mapping[str, Mapping[str, np.ndarray]],
) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, np.ndarray]]]:
    transformed_train = {}
    transformed_query = {name: {} for name in query_features}
    for feature_name, values in train_features.items():
        initial_scaler = TrainOnlyStandardizer().fit(values, "bc_train")
        train_initial = initial_scaler.transform(values)
        if feature_name in {"F2_current_rgb", "F3_four_frame_stack"}:
            pca = TrainOnlyPCA(32).fit(train_initial, "bc_train")
            train_reduced = pca.transform(train_initial)
            query_reduced = {
                name: pca.transform(initial_scaler.transform(features[feature_name]))
                for name, features in query_features.items()
            }
        else:
            train_reduced = train_initial
            query_reduced = {
                name: initial_scaler.transform(features[feature_name])
                for name, features in query_features.items()
            }
        final_scaler = TrainOnlyStandardizer().fit(train_reduced, "bc_train")
        transformed_train[feature_name] = final_scaler.transform(train_reduced)
        for name in query_features:
            transformed_query[name][feature_name] = final_scaler.transform(query_reduced[name])
    return transformed_train, transformed_query


def support_pass(dataset: Mapping[str, np.ndarray], label: np.ndarray, validation_label: np.ndarray) -> bool:
    train_positive = label == 1
    train_negative = label == 0
    return bool(
        len(np.unique(dataset["episode"][train_positive])) >= 2
        and len(np.unique(dataset["episode"][train_negative])) >= 2
        and len(np.unique(dataset["episode_seed"][train_positive])) >= 2
        and (validation_label == 0).any()
        and (validation_label == 1).any()
    )


def cross_episode_nn_distances(features: np.ndarray, episodes: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    result = np.zeros(len(values), dtype=np.float32)
    reference_norm = (values * values).sum(axis=1)
    for start in range(0, len(values), 256):
        block = values[start : start + 256]
        distances = block @ values.T
        distances *= -2.0
        distances += (block * block).sum(axis=1, keepdims=True)
        distances += reference_norm[None, :]
        same = episodes[start : start + len(block), None] == episodes[None, :]
        distances[same] = np.inf
        result[start : start + len(block)] = np.sqrt(np.maximum(distances.min(axis=1), 0.0) / values.shape[1])
    return result


def query_neighbors(reference: np.ndarray, query: np.ndarray, k: int = 10):
    ref = np.asarray(reference, dtype=np.float32)
    values = np.asarray(query, dtype=np.float32)
    indices_rows = []
    distance_rows = []
    ref_norm = (ref * ref).sum(axis=1)
    for start in range(0, len(values), 256):
        block = values[start : start + 256]
        distances = (block * block).sum(axis=1, keepdims=True) + ref_norm[None, :] - 2.0 * block @ ref.T
        indices = np.argpartition(distances, k - 1, axis=1)[:, :k]
        ordered = np.take_along_axis(indices, np.argsort(np.take_along_axis(distances, indices, axis=1), axis=1), axis=1)
        selected = np.take_along_axis(distances, ordered, axis=1)
        indices_rows.append(ordered)
        distance_rows.append(np.sqrt(np.maximum(selected, 0.0) / ref.shape[1]))
    return np.vstack(indices_rows), np.vstack(distance_rows)


def categorical_entropy(values: Sequence[int]) -> float:
    counts = np.asarray(list(Counter(int(v) for v in values).values()), dtype=np.float64)
    probabilities = counts / counts.sum()
    return entropy(probabilities)


def observability_and_ood(config: Dict[str, Any]) -> Dict[str, Any]:
    root = Path("artifacts/exp12")
    checkpoint = next(item["path"] for item in config["checkpoints"] if int(item["training_seed"]) == 29)
    datasets = {
        "bc_train": load_audit_dataset(Path(config["datasets"]["bc_train"]["path"])),
        "bc_validation": load_audit_dataset(Path(config["datasets"]["bc_validation"]["path"])),
        "dagger1": load_audit_dataset(Path(config["datasets"]["dagger1"]["path"])),
        "student_dev": combine_student_traces(root / "runtime_traces"),
    }
    assert_disjoint_episode_splits(datasets["bc_train"]["episode_seed"], datasets["bc_validation"]["episode_seed"])
    raw_features = {name: feature_spaces(dataset, checkpoint) for name, dataset in datasets.items()}
    train_features, query_features = prepare_feature_transforms(
        raw_features["bc_train"], {name: value for name, value in raw_features.items() if name != "bc_train"}
    )
    labels = {name: labels_from_dataset(dataset) for name, dataset in datasets.items()}
    metric_rows = []
    key_labels = ("tree_visible", "contact_range", "valid_attack_geometry")
    for label_name in key_labels:
        train_label = labels["bc_train"][label_name]
        validation_label = labels["bc_validation"][label_name]
        train_valid = train_label >= 0
        validation_valid = validation_label >= 0
        sufficient = support_pass(
            datasets["bc_train"], train_label, validation_label[validation_valid]
        )
        for feature_name, feature_train in train_features.items():
            if not sufficient:
                metric_rows.append(
                    {
                        "label": label_name,
                        "feature_space": feature_name,
                        "model": "not_fit",
                        "evaluation_split": "bc_validation",
                        "status": "coverage_insufficient_for_generalization_claim",
                    }
                )
                continue
            probe = FixedLogisticProbe().fit(feature_train[train_valid], train_label[train_valid])
            for model_name, probability_fn in (
                ("fixed_logistic", lambda query: probe.probabilities(query)),
                ("knn_k5", lambda query: knn_probabilities(feature_train[train_valid], train_label[train_valid], query, 5)),
            ):
                for split_name in ("bc_train", "bc_validation", "dagger1", "student_dev"):
                    split_label = labels[split_name][label_name]
                    valid = split_label >= 0
                    if not valid.any() or len(np.unique(split_label[valid])) < 2:
                        metric_rows.append(
                            {
                                "label": label_name,
                                "feature_space": feature_name,
                                "model": model_name,
                                "evaluation_split": split_name,
                                "status": "coverage_insufficient_for_generalization_claim",
                            }
                        )
                        continue
                    split_features = feature_train if split_name == "bc_train" else query_features[split_name][feature_name]
                    probabilities = probability_fn(split_features[valid])
                    metrics = binary_metrics(split_label[valid], probabilities)
                    low, high = bootstrap_balanced_interval(
                        split_label[valid], probabilities, datasets[split_name]["episode"][valid]
                    )
                    metric_rows.append(
                        {
                            "label": label_name,
                            "feature_space": feature_name,
                            "model": model_name,
                            "evaluation_split": split_name,
                            "status": "diagnostic_only",
                            "majority_baseline": 0.5,
                            "balanced_accuracy": metrics["balanced_accuracy"],
                            "balanced_accuracy_ci_low": low,
                            "balanced_accuracy_ci_high": high,
                            "macro_f1": metrics["macro_f1"],
                            "auroc": metrics["auroc"],
                            "pr_auc": metrics["pr_auc"],
                            "confusion_matrix": json.dumps(metrics["confusion_matrix"]),
                        }
                    )

    train_distance = np.asarray(
        datasets["bc_train"].get("audit_raycast_distance", np.full(len(datasets["bc_train"]["episode"]), np.nan)),
        dtype=np.float32,
    )
    train_distance_valid = np.isfinite(train_distance)
    for feature_name, feature_train in train_features.items():
        ridge = FixedRidgeProbe(0.01).fit(
            feature_train[train_distance_valid], train_distance[train_distance_valid]
        )
        for split_name in ("bc_train", "bc_validation", "dagger1", "student_dev"):
            distance = np.asarray(
                datasets[split_name].get("audit_raycast_distance", np.full(len(datasets[split_name]["episode"]), np.nan)),
                dtype=np.float32,
            )
            valid = np.isfinite(distance)
            if not valid.any():
                metric_rows.append(
                    {
                        "label": "raycast_distance_regression",
                        "feature_space": feature_name,
                        "model": "fixed_ridge",
                        "evaluation_split": split_name,
                        "status": "unsupported_by_current_audit_data",
                    }
                )
                continue
            split_features = feature_train if split_name == "bc_train" else query_features[split_name][feature_name]
            metrics = regression_metrics(distance[valid], ridge.predict(split_features[valid]))
            metric_rows.append(
                {
                    "label": "raycast_distance_regression",
                    "feature_space": feature_name,
                    "model": "fixed_ridge",
                    "evaluation_split": split_name,
                    "status": "diagnostic_only",
                    **metrics,
                }
            )
    atomic_csv(root / "probe_metrics.csv", metric_rows)

    ood_summary: Dict[str, Any] = {}
    neighbor_rows = []
    collision_candidates = []
    train_valid_geometry = labels["bc_train"]["valid_attack_geometry"]
    train_actions = datasets["bc_train"]["action"]
    for feature_name, reference in train_features.items():
        loo = cross_episode_nn_distances(reference, datasets["bc_train"]["episode"])
        thresholds = {str(value): float(np.percentile(loo, value)) for value in (50, 90, 95, 99)}
        ood_summary[feature_name] = {"train_leave_one_episode_out_percentiles": thresholds, "queries": {}}
        for split_name in ("bc_validation", "dagger1", "student_dev"):
            query = query_features[split_name][feature_name]
            indices, distances = query_neighbors(reference, query, 10)
            nearest = distances[:, 0]
            ood_summary[feature_name]["queries"][split_name] = {
                "states": int(len(query)),
                "over_train_95th_fraction": float((nearest > thresholds["95"]).mean()),
                "over_train_99th_fraction": float((nearest > thresholds["99"]).mean()),
                "median_nearest_distance": float(np.median(nearest)),
            }
            query_geometry = labels[split_name]["valid_attack_geometry"]
            for query_index in range(len(query)):
                neighbors = indices[query_index]
                valid_neighbor_labels = train_valid_geometry[neighbors]
                neighbor_rows.append(
                    {
                        "feature_space": feature_name,
                        "query_split": split_name,
                        "query_index": query_index,
                        "query_seed": int(datasets[split_name]["episode_seed"][query_index]),
                        "query_step": int(datasets[split_name]["episode_step"][query_index]),
                        "nearest_distance": float(distances[query_index, 0]),
                        "over_train_95th": bool(distances[query_index, 0] > thresholds["95"]),
                        "over_train_99th": bool(distances[query_index, 0] > thresholds["99"]),
                        "query_valid_attack_geometry": int(query_geometry[query_index]),
                        "neighbor_valid_attack_geometry": int(valid_neighbor_labels[0]),
                        "neighbor_teacher_action": ACTION_NAMES[int(train_actions[neighbors[0]])],
                        "local_privileged_label_entropy_k10": categorical_entropy(valid_neighbor_labels[valid_neighbor_labels >= 0]) if (valid_neighbor_labels >= 0).any() else None,
                        "local_teacher_action_entropy_k10": categorical_entropy(train_actions[neighbors]),
                        "neighbor_indices_k10": json.dumps(neighbors.tolist()),
                    }
                )
                if (
                    split_name == "student_dev"
                    and feature_name == "F4_handwritten"
                    and query_geometry[query_index] >= 0
                    and valid_neighbor_labels[0] >= 0
                    and query_geometry[query_index] != valid_neighbor_labels[0]
                    and labels[split_name]["contact_range"][query_index] == 1
                ):
                    collision_candidates.append(
                        (float(distances[query_index, 0]), query_index, int(neighbors[0]))
                    )
    atomic_csv(root / "nearest_neighbor_audit.csv", neighbor_rows)
    atomic_json(root / "ood_summary.json", ood_summary)

    example_root = root / "critical_collision_examples"
    example_root.mkdir(parents=True, exist_ok=True)
    example_rows = []
    for rank, (distance, query_index, neighbor_index) in enumerate(sorted(collision_candidates)[:8], start=1):
        query_rgb = cv2.cvtColor(datasets["student_dev"]["pov"][query_index], cv2.COLOR_RGB2BGR)
        neighbor_rgb = cv2.cvtColor(datasets["bc_train"]["pov"][neighbor_index], cv2.COLOR_RGB2BGR)
        montage = np.concatenate((query_rgb, neighbor_rgb), axis=1)
        cv2.putText(montage, "student query", (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
        cv2.putText(montage, "train neighbor", (66, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
        path = example_root / "collision_{:02d}.png".format(rank)
        cv2.imwrite(str(path), montage)
        example_rows.append(
            {
                "rank": rank,
                "path": str(path),
                "distance": distance,
                "query_seed": int(datasets["student_dev"]["episode_seed"][query_index]),
                "query_step": int(datasets["student_dev"]["episode_step"][query_index]),
                "query_valid_attack_geometry": int(labels["student_dev"]["valid_attack_geometry"][query_index]),
                "neighbor_seed": int(datasets["bc_train"]["episode_seed"][neighbor_index]),
                "neighbor_step": int(datasets["bc_train"]["episode_step"][neighbor_index]),
                "neighbor_valid_attack_geometry": int(train_valid_geometry[neighbor_index]),
                "neighbor_teacher_action": ACTION_NAMES[int(train_actions[neighbor_index])],
                "legal_vector_l2_difference": float(np.linalg.norm(datasets["student_dev"]["legal_vector"][query_index] - datasets["bc_train"]["legal_vector"][neighbor_index])),
            }
        )
    atomic_csv(example_root / "index.csv", example_rows)
    summary = {
        "probe_metric_rows": len(metric_rows),
        "nearest_neighbor_rows": len(neighbor_rows),
        "critical_collision_examples": len(example_rows),
    }
    atomic_json(root / "observability_probe_summary.json", summary)
    return summary


def main():
    args = parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    allowed = {"teacher_dev", "bc_train", "bc_validation", "dagger1", "student_dev"}
    if allowed & set(config["forbidden_splits"]):
        raise PermissionError("allowed diagnostics intersect forbidden splits")
    validation = load_audit_dataset(Path(config["datasets"]["bc_validation"]["path"]))
    result = {}
    if args.stage in {"collapse", "all"}:
        result["collapse"] = collapse_interventions(config, validation)
    if args.stage in {"probes", "all"}:
        result["probes"] = observability_and_ood(config)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
