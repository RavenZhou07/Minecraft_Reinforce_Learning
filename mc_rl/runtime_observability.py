"""Read-only diagnostics for the recurrent Treechop actor.

The actor-input arrays and privileged audit arrays are intentionally emitted
under disjoint manifests.  Nothing in this module changes an action, breaks a
loop, invokes a teacher, or adds a trace to a training dataset.
"""

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from mc_rl.actions import ACTION_NAMES
from mc_rl.learning_observation import STUDENT_VECTOR_NAMES
from mc_rl.recurrent_treechop_bc import (
    ACTION_COUNT,
    PREVIOUS_ACTION_DISABLED_ZERO,
    PREVIOUS_ACTION_EMBEDDED,
    START_ACTION_TOKEN,
    RecurrentTreechopPolicy,
)
from mc_rl.wrappers import inventory_log_count


TRACE_VERSION = "recurrent_runtime_observability_v1"
DIAGNOSTIC_ONLY = "diagnostic_only_not_policy_training_data"
ACTOR_INPUT_FIELDS = ("pov", "legal_vector", "previous_action_token")
PRIVILEGED_AUDIT_FIELDS = (
    "audit_raycast_is_log",
    "audit_raycast_in_range",
    "audit_raycast_distance",
)


def _array_hash(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _entropy(probabilities: np.ndarray) -> float:
    values = np.asarray(probabilities, dtype=np.float64)
    return -float(np.sum(values * np.log(np.maximum(values, 1e-12))))


def js_divergence(left: np.ndarray, right: np.ndarray) -> float:
    p = np.asarray(left, dtype=np.float64)
    q = np.asarray(right, dtype=np.float64)
    middle = 0.5 * (p + q)
    return 0.5 * float(np.sum(p * np.log(np.maximum(p, 1e-12) / np.maximum(middle, 1e-12)))) + 0.5 * float(
        np.sum(q * np.log(np.maximum(q, 1e-12) / np.maximum(middle, 1e-12)))
    )


def raw_legal_values(observation: Mapping[str, Any]) -> np.ndarray:
    telemetry = observation["telemetry"]
    return np.asarray(
        [
            telemetry["x"],
            telemetry["y"],
            telemetry["z"],
            telemetry["yaw"],
            telemetry["pitch"],
            telemetry["biome_id"],
            telemetry["biome_temperature"],
            telemetry["biome_rainfall"],
            inventory_log_count(observation) or 0,
        ],
        dtype=np.float32,
    )


RAW_LEGAL_NAMES = (
    "x",
    "y",
    "z",
    "yaw",
    "pitch",
    "biome_id",
    "biome_temperature",
    "biome_rainfall",
    "inventory_log_count",
)


class RuntimeTraceRecorder:
    """Accumulate one autonomous episode without feeding audit values back."""

    def __init__(
        self,
        checkpoint: str,
        checkpoint_seed: int,
        environment_seed: int,
        max_steps: int,
        previous_action_mode: str = PREVIOUS_ACTION_EMBEDDED,
    ) -> None:
        actor_input_fields = (
            ["pov", "legal_vector", "episode_local_gru_history"]
            if previous_action_mode == PREVIOUS_ACTION_DISABLED_ZERO
            else list(ACTOR_INPUT_FIELDS)
        )
        self.metadata = {
            "trace_version": TRACE_VERSION,
            "usage": DIAGNOSTIC_ONLY,
            "checkpoint": str(checkpoint),
            "checkpoint_seed": int(checkpoint_seed),
            "environment_seed": int(environment_seed),
            "max_steps": int(max_steps),
            "actor_input_fields": actor_input_fields,
            "previous_action_mode": str(previous_action_mode),
            "previous_action_trace_usage": (
                "diagnostic_alignment_only"
                if previous_action_mode == PREVIOUS_ACTION_DISABLED_ZERO
                else "actor_input"
            ),
            "privileged_audit_fields": list(PRIVILEGED_AUDIT_FIELDS),
            "raw_legal_names": list(RAW_LEGAL_NAMES),
            "legal_vector_names": list(STUDENT_VECTOR_NAMES),
            "teacher_actions_executed": 0,
            "privileged_actor_inputs": 0,
        }
        self.rows: Dict[str, List[Any]] = {
            key: []
            for key in (
                "episode_step",
                "pov",
                "raw_rgb_shape",
                "raw_rgb_hash",
                "raw_rgb_mean",
                "raw_rgb_std",
                "raw_rgb_delta_l1_from_previous",
                "raw_rgb_changed",
                "legal_vector_raw",
                "legal_vector",
                "legal_vector_delta",
                "yaw",
                "pitch",
                "origin_relative_position",
                "step_delta_position",
                "inventory_log_delta",
                "previous_action_token",
                "previous_executed_action",
                "disabled_action_channel_max_abs",
                "gru_hidden_norm",
                "gru_hidden_delta_l2",
                "cnn_embedding",
                "cnn_embedding_norm",
                "cnn_embedding_delta_l2",
                "scalar_embedding",
                "scalar_embedding_norm",
                "action_logits",
                "action_probabilities",
                "policy_entropy",
                "top1_action",
                "top1_probability",
                "top2_action",
                "top1_minus_top2_margin",
                "policy_js_from_previous",
                "selected_action_id",
                "selected_action_name",
                "executed_action_id",
                "executed_action_name",
                "current_identical_action_streak",
                "dominant_action_so_far",
                "dominant_fraction_so_far",
                "action_transition_count",
                "hidden",
                "audit_raycast_is_log",
                "audit_raycast_in_range",
                "audit_raycast_distance",
                "teacher_action_executed",
                "privileged_actor_input",
            )
        }
        self._origin: Optional[np.ndarray] = None
        self._previous_raw: Optional[np.ndarray] = None
        self._initial_inventory = 0
        self._previous_pov: Optional[np.ndarray] = None
        self._previous_vector: Optional[np.ndarray] = None
        self._previous_cnn: Optional[np.ndarray] = None
        self._previous_hidden: Optional[np.ndarray] = None
        self._previous_probabilities: Optional[np.ndarray] = None
        self._executed: List[int] = []

    def append(
        self,
        step: int,
        observation: Mapping[str, Any],
        pov: np.ndarray,
        legal_vector: np.ndarray,
        previous_action_token: int,
        probabilities: np.ndarray,
        diagnostics: Mapping[str, np.ndarray],
        next_hidden: torch.Tensor,
        selected_action: int,
        executed_action: int,
    ) -> None:
        frame = np.asarray(pov, dtype=np.uint8)
        vector = np.asarray(legal_vector, dtype=np.float32)
        raw = raw_legal_values(observation)
        if self._origin is None:
            self._origin = raw[:3].copy()
            self._initial_inventory = int(raw[-1])
        previous_raw = raw if self._previous_raw is None else self._previous_raw
        previous_frame = frame if self._previous_pov is None else self._previous_pov
        previous_vector = vector if self._previous_vector is None else self._previous_vector
        cnn = np.asarray(diagnostics["cnn_embedding"], dtype=np.float32)
        previous_cnn = cnn if self._previous_cnn is None else self._previous_cnn
        hidden = next_hidden.detach().cpu().numpy().reshape(-1).astype(np.float32)
        previous_hidden = np.zeros_like(hidden) if self._previous_hidden is None else self._previous_hidden
        probs = np.asarray(probabilities, dtype=np.float32)
        previous_probs = probs if self._previous_probabilities is None else self._previous_probabilities
        ordered = np.argsort(probs)[::-1]

        executed = int(executed_action)
        self._executed.append(executed)
        counts = Counter(self._executed)
        dominant_id, dominant_count = min(counts.items(), key=lambda item: (-item[1], item[0]))
        streak = 1
        for value in reversed(self._executed[:-1]):
            if value != executed:
                break
            streak += 1
        transitions = sum(int(a != b) for a, b in zip(self._executed, self._executed[1:]))
        raycast = observation.get("raycast", {})

        values = {
            "episode_step": int(step),
            "pov": frame.copy(),
            "raw_rgb_shape": np.asarray(frame.shape, dtype=np.int16),
            "raw_rgb_hash": _array_hash(frame),
            "raw_rgb_mean": float(frame.mean()),
            "raw_rgb_std": float(frame.std()),
            "raw_rgb_delta_l1_from_previous": float(np.abs(frame.astype(np.int16) - previous_frame.astype(np.int16)).mean()),
            "raw_rgb_changed": bool(not np.array_equal(frame, previous_frame)) if step else False,
            "legal_vector_raw": raw.copy(),
            "legal_vector": vector.copy(),
            "legal_vector_delta": (vector - previous_vector).copy(),
            "yaw": float(raw[3]),
            "pitch": float(raw[4]),
            "origin_relative_position": (raw[:3] - self._origin).copy(),
            "step_delta_position": (raw[:3] - previous_raw[:3]).copy(),
            "inventory_log_delta": int(raw[-1]) - self._initial_inventory,
            "previous_action_token": int(previous_action_token),
            "previous_executed_action": int(self._executed[-2]) if len(self._executed) > 1 else START_ACTION_TOKEN,
            "disabled_action_channel_max_abs": float(
                np.max(np.abs(np.asarray(diagnostics["action_embedding"], dtype=np.float32)))
            ),
            "gru_hidden_norm": float(np.linalg.norm(hidden)),
            "gru_hidden_delta_l2": float(np.linalg.norm(hidden - previous_hidden)),
            "cnn_embedding": cnn.copy(),
            "cnn_embedding_norm": float(np.linalg.norm(cnn)),
            "cnn_embedding_delta_l2": float(np.linalg.norm(cnn - previous_cnn)),
            "scalar_embedding": np.asarray(diagnostics["scalar_embedding"], dtype=np.float32).copy(),
            "scalar_embedding_norm": float(np.linalg.norm(diagnostics["scalar_embedding"])),
            "action_logits": np.asarray(diagnostics["logits"], dtype=np.float32).copy(),
            "action_probabilities": probs.copy(),
            "policy_entropy": _entropy(probs),
            "top1_action": int(ordered[0]),
            "top1_probability": float(probs[ordered[0]]),
            "top2_action": int(ordered[1]),
            "top1_minus_top2_margin": float(probs[ordered[0]] - probs[ordered[1]]),
            "policy_js_from_previous": js_divergence(previous_probs, probs),
            "selected_action_id": int(selected_action),
            "selected_action_name": ACTION_NAMES[int(selected_action)],
            "executed_action_id": executed,
            "executed_action_name": ACTION_NAMES[executed],
            "current_identical_action_streak": streak,
            "dominant_action_so_far": ACTION_NAMES[dominant_id],
            "dominant_fraction_so_far": dominant_count / len(self._executed),
            "action_transition_count": transitions,
            "hidden": hidden.copy(),
            "audit_raycast_is_log": bool(raycast.get("is_log", False)),
            "audit_raycast_in_range": bool(raycast.get("in_range", False)),
            "audit_raycast_distance": float(raycast.get("distance", np.nan)),
            "teacher_action_executed": 0,
            "privileged_actor_input": 0,
        }
        for key, value in values.items():
            self.rows[key].append(value)
        self._previous_raw = raw.copy()
        self._previous_pov = frame.copy()
        self._previous_vector = vector.copy()
        self._previous_cnn = cnn.copy()
        self._previous_hidden = hidden.copy()
        self._previous_probabilities = probs.copy()

    def arrays(self) -> Dict[str, np.ndarray]:
        arrays = {key: np.asarray(value) for key, value in self.rows.items()}
        arrays["trace_metadata_json"] = np.asarray(json.dumps(self.metadata, sort_keys=True))
        arrays["raw_rgb_dtype"] = np.asarray("uint8")
        arrays["usage"] = np.asarray(DIAGNOSTIC_ONLY)
        return arrays

    def finalize(self, done: bool, info: Mapping[str, Any]) -> None:
        """Attach terminal task facts after the last selected action executes."""

        self.metadata["environment_done"] = bool(done)
        self.metadata["success"] = bool(info.get("success", False))
        self.metadata["final_inventory_log_delta"] = int(
            info.get("inventory_log_delta", self.rows["inventory_log_delta"][-1] if self.rows["inventory_log_delta"] else 0)
            or 0
        )


def atomic_save_trace(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(output)


def load_trace(path: Path) -> Dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as values:
        return {name: np.asarray(values[name]) for name in values.files}


def validate_trace_integrity(trace: Mapping[str, np.ndarray]) -> Dict[str, bool]:
    selected = np.asarray(trace["selected_action_id"], dtype=np.int64)
    executed = np.asarray(trace["executed_action_id"], dtype=np.int64)
    previous = np.asarray(trace["previous_action_token"], dtype=np.int64)
    result = {
        "selected_executed_match": bool(np.array_equal(selected, executed)),
        "action_ids_valid": bool(((executed >= 0) & (executed < ACTION_COUNT)).all()),
        "start_token_at_reset": bool(len(previous) and previous[0] == START_ACTION_TOKEN),
        "previous_action_causal": bool(len(previous) <= 1 or np.array_equal(previous[1:], executed[:-1])),
        "teacher_actions_zero": bool(np.asarray(trace["teacher_action_executed"]).sum() == 0),
        "privileged_actor_inputs_zero": bool(np.asarray(trace["privileged_actor_input"]).sum() == 0),
        "rgb_shape_valid": bool(np.asarray(trace["pov"]).shape[1:] == (64, 64, 3)),
        "rgb_dtype_valid": bool(np.asarray(trace["pov"]).dtype == np.uint8),
        "vector_shape_valid": bool(np.asarray(trace["legal_vector"]).shape[1:] == (len(STUDENT_VECTOR_NAMES),)),
    }
    metadata = json.loads(str(trace["trace_metadata_json"]))
    if metadata.get("previous_action_mode") == PREVIOUS_ACTION_DISABLED_ZERO:
        result["disabled_action_channel_zero"] = bool(
            np.asarray(trace["disabled_action_channel_max_abs"]).max() == 0.0
        )
        result["previous_action_is_diagnostic_only"] = bool(
            "previous_action_token" not in metadata.get("actor_input_fields", [])
        )
    result["passed"] = all(result.values())
    return result


def standalone_replay(
    checkpoint: str,
    trace: Mapping[str, np.ndarray],
) -> Dict[str, Any]:
    policy = RecurrentTreechopPolicy.load(checkpoint)
    hidden = None
    replay = {key: [] for key in ("cnn_embedding", "scalar_embedding", "hidden", "action_logits", "action_probabilities", "selected_action_id")}
    for pov, vector, token in zip(trace["pov"], trace["legal_vector"], trace["previous_action_token"]):
        action, probabilities, hidden, diagnostics = policy.predict_step_with_diagnostics(
            pov, vector, int(token), hidden
        )
        replay["cnn_embedding"].append(diagnostics["cnn_embedding"])
        replay["scalar_embedding"].append(diagnostics["scalar_embedding"])
        replay["hidden"].append(hidden.detach().cpu().numpy().reshape(-1))
        replay["action_logits"].append(diagnostics["logits"])
        replay["action_probabilities"].append(probabilities)
        replay["selected_action_id"].append(action)
    replay_arrays = {key: np.asarray(value) for key, value in replay.items()}
    comparisons = {}
    for key in ("cnn_embedding", "scalar_embedding", "hidden", "action_logits", "action_probabilities"):
        difference = np.abs(replay_arrays[key].astype(np.float64) - np.asarray(trace[key], dtype=np.float64))
        comparisons[key + "_max_abs_error"] = float(difference.max()) if difference.size else 0.0
    comparisons["argmax_match"] = bool(np.array_equal(replay_arrays["selected_action_id"], trace["selected_action_id"]))
    comparisons["passed"] = bool(
        comparisons["argmax_match"]
        and all(comparisons[key] <= 1e-6 for key in comparisons if key.endswith("_max_abs_error"))
    )
    return comparisons


def summarize_trace(trace: Mapping[str, np.ndarray]) -> Dict[str, Any]:
    actions = np.asarray(trace["executed_action_id"], dtype=np.int64)
    metadata = json.loads(str(trace["trace_metadata_json"]))
    counts = Counter(actions.tolist())
    dominant, dominant_count = min(counts.items(), key=lambda item: (-item[1], item[0]))
    inventory_success = bool(metadata.get("success", False))
    periodic = periodic_cycle_diagnostics(actions)
    is_log = trace["audit_raycast_is_log"].astype(bool)
    in_range = trace["audit_raycast_in_range"].astype(bool)
    valid_attack_mask = np.isin(actions, [7, 8]) & is_log & in_range
    sustained_valid_attacks = 0
    inferred_break_step: Optional[int] = None
    for step, (action, log_visible, contact) in enumerate(
        zip(actions.tolist(), is_log.tolist(), in_range.tolist())
    ):
        if action in (7, 8) and log_visible and contact:
            sustained_valid_attacks += 1
        if (
            inferred_break_step is None
            and sustained_valid_attacks >= 5
            and not log_visible
        ):
            inferred_break_step = step

    def first_step(mask: np.ndarray) -> Optional[int]:
        indices = np.flatnonzero(mask)
        return int(indices[0]) if len(indices) else None

    inventory_first = first_step(
        np.asarray(trace["inventory_log_delta"], dtype=np.int64) >= 1
    )
    if inventory_first is None and inventory_success:
        inventory_first = int(len(actions))
    return {
        "checkpoint_seed": int(metadata["checkpoint_seed"]),
        "environment_seed": int(metadata["environment_seed"]),
        "steps": int(len(actions)),
        "inventory_success": inventory_success,
        "timeout": bool(not inventory_success and len(actions) >= int(metadata["max_steps"])),
        "meaningful_interaction": bool(valid_attack_mask.any()),
        "first_meaningful_interaction_step": first_step(valid_attack_mask),
        "approach": bool(is_log.any()),
        "first_approach_step": first_step(is_log),
        "contact": bool((is_log & in_range).any()),
        "first_contact_step": first_step(is_log & in_range),
        "valid_attack": bool(valid_attack_mask.any()),
        "first_valid_attack_step": first_step(valid_attack_mask),
        "block_break": inferred_break_step is not None,
        "first_block_break_step": inferred_break_step,
        "pickup": inventory_success,
        "first_pickup_step": inventory_first,
        "first_inventory_acquisition_step": inventory_first,
        "dominant_action": ACTION_NAMES[dominant],
        "dominant_fraction": dominant_count / max(len(actions), 1),
        "longest_identical_action_streak": int(np.asarray(trace["current_identical_action_streak"]).max()),
        "action_transitions": int(np.asarray(trace["action_transition_count"])[-1]),
        "unique_actions": int(len(counts)),
        "executed_action_entropy": _entropy(np.asarray(list(counts.values()), dtype=np.float64) / len(actions)),
        "mean_policy_entropy": float(np.asarray(trace["policy_entropy"]).mean()),
        "median_top1_probability": float(np.median(trace["top1_probability"])),
        "median_top1_top2_margin": float(np.median(trace["top1_minus_top2_margin"])),
        "mean_policy_js_between_steps": float(np.asarray(trace["policy_js_from_previous"])[1:].mean()) if len(actions) > 1 else 0.0,
        "mean_cnn_embedding_delta": float(np.asarray(trace["cnn_embedding_delta_l2"])[1:].mean()) if len(actions) > 1 else 0.0,
        "mean_hidden_delta": float(np.asarray(trace["gru_hidden_delta_l2"])[1:].mean()) if len(actions) > 1 else 0.0,
        "teacher_actions_executed": int(np.asarray(trace["teacher_action_executed"]).sum()),
        "privileged_actor_inputs": int(np.asarray(trace["privileged_actor_input"]).sum()),
        "disabled_action_channel_max_abs": float(
            np.asarray(trace["disabled_action_channel_max_abs"]).max()
        ),
        **periodic,
    }


def _is_primitive_pattern(pattern: np.ndarray) -> bool:
    length = len(pattern)
    if length <= 1:
        return True
    return not any(
        length % smaller == 0
        and all(pattern[index] == pattern[index % smaller] for index in range(length))
        for smaller in range(1, length)
    )


def max_periodic_cycle_streak(actions: Sequence[int], period: int) -> int:
    """Longest exact primitive p-cycle span, requiring at least two repeats."""

    values = np.asarray(actions, dtype=np.int64)
    period = int(period)
    if period < 1 or period > 4:
        raise ValueError("period must be within 1..4")
    if len(values) < 2 * period:
        return 0
    best = 0
    for start in range(0, len(values) - 2 * period + 1):
        pattern = values[start : start + period]
        if not _is_primitive_pattern(pattern):
            continue
        end = start + period
        while end < len(values) and values[end] == pattern[(end - start) % period]:
            end += 1
        length = end - start
        if length >= 2 * period:
            best = max(best, length)
    return int(best)


def periodic_cycle_diagnostics(actions: Sequence[int]) -> Dict[str, Any]:
    """Summarize period-1..4 collapse without treating transitions as success."""

    values = np.asarray(actions, dtype=np.int64)
    size = int(len(values))
    streaks = {period: max_periodic_cycle_streak(values, period) for period in range(1, 5)}
    if size:
        dominant_period = min(streaks, key=lambda period: (-streaks[period], period))
        dominant_fraction = streaks[dominant_period] / size
    else:
        dominant_period = None
        dominant_fraction = 0.0
    period_2_to_4_streak = max(streaks[period] for period in (2, 3, 4)) if size else 0
    bigrams = Counter(zip(values[:-1].tolist(), values[1:].tolist()))
    if bigrams:
        dominant_bigram, bigram_count = min(
            bigrams.items(), key=lambda item: (-item[1], item[0])
        )
        bigram_name = "{} -> {}".format(
            ACTION_NAMES[dominant_bigram[0]], ACTION_NAMES[dominant_bigram[1]]
        )
        bigram_fraction = bigram_count / (size - 1)
    else:
        bigram_name = None
        bigram_fraction = 0.0
    transitions = np.flatnonzero(values[1:] != values[:-1]) + 1 if size > 1 else np.asarray([], dtype=np.int64)
    return {
        "max_period_1_streak": streaks[1],
        "max_period_2_cycle_streak": streaks[2],
        "max_period_3_cycle_streak": streaks[3],
        "max_period_4_cycle_streak": streaks[4],
        "dominant_period_1_to_4": dominant_period,
        "fraction_of_episode_in_dominant_period_1_to_4_cycle": dominant_fraction,
        "dominant_period_2_to_4_cycle_fraction": period_2_to_4_streak / size if size else 0.0,
        "dominant_action_bigram": bigram_name,
        "dominant_bigram_fraction": bigram_fraction,
        "time_to_first_action_transition": int(transitions[0]) if len(transitions) else None,
        "pure_single_action_fixed_point": bool(size > 0 and len(set(values.tolist())) == 1),
    }


def atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output)


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(output)
