"""End-to-end behaviour cloning for Natural Treechop.

Unlike the older contact/gate models, this policy owns every environment
action during autonomous rollout.  Privileged teacher phase is an auxiliary
training label only: a legal-observation phase head predicts it at inference,
and the action head may consume only those predicted probabilities.
"""

from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from mc_rl.actions import ACTION_NAMES
from mc_rl.learning_observation import (
    STUDENT_OBSERVATION_SCHEMA_VERSION,
    LegalObservationAdapter,
    student_input_manifest,
)
from mc_rl.natural_contact_bc import extract_contact_features


MODEL_VERSION = "natural_treechop_end_to_end_bc_v1"
ACTION_CLASSES = np.arange(len(ACTION_NAMES), dtype=np.int64)
PHASE_NAMES = ("search", "approach", "contact", "recovery", "pickup")
PHASE_CLASSES = np.arange(len(PHASE_NAMES), dtype=np.int64)


def coarse_teacher_phase(
    search_state: str, contact_state: str, action_source: str
) -> str:
    """Compress privileged teacher state into an auxiliary supervision label."""

    contact = str(contact_state or "")
    search = str(search_state or "")
    if contact in {
        "BLOCK_DISAPPEARED",
        "DROP_RECOVERY",
        "COLLECT_DROP",
    }:
        return "pickup"
    if contact in {
        "COORDINATE_RECOVER",
        "POST_RECOVERY_VERIFY",
        "COORDINATE_REPLAN",
        "EXACT_LOG_RESCAN",
        "REACQUIRE_SAME_TRUNK",
        "BACKOFF",
        "ORBIT_REACQUIRE",
        "REPLAN",
    }:
        return "recovery"
    if action_source == "contact":
        return "contact"
    if search in {"APPROACH", "LOCAL_REACQUIRE"}:
        return "approach"
    if search in {"RECOVER", "REPLAN"}:
        return "recovery"
    return "search"


def phase_ids(values: Sequence[str]) -> np.ndarray:
    lookup = {name: index for index, name in enumerate(PHASE_NAMES)}
    try:
        return np.asarray([lookup[str(value)] for value in values], dtype=np.int64)
    except KeyError as error:
        raise ValueError("unknown coarse teacher phase: {}".format(error.args[0]))


def build_causal_action_history(
    previous_actions: Sequence[int],
    episode_ids: Sequence[int],
    history_length: int,
) -> np.ndarray:
    """Build padded action histories without crossing episode boundaries."""

    previous = np.asarray(previous_actions, dtype=np.int64)
    episodes = np.asarray(episode_ids, dtype=np.int64)
    if history_length <= 0 or len(previous) != len(episodes):
        raise ValueError("aligned actions/episodes and positive history required")
    if len(previous) and not np.isin(previous, ACTION_CLASSES).all():
        raise ValueError("action history contains an invalid action id")
    rows = np.zeros((len(previous), int(history_length)), dtype=np.int64)
    episode_start = 0
    for index in range(len(previous)):
        if index == 0 or episodes[index] != episodes[index - 1]:
            episode_start = index
        available = previous[episode_start : index + 1]
        take = available[-history_length:]
        rows[index, -len(take) :] = take
    return rows


def action_history_one_hot(histories: np.ndarray) -> np.ndarray:
    values = np.asarray(histories, dtype=np.int64)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or (len(values) and not np.isin(values, ACTION_CLASSES).all()):
        raise ValueError("action histories must be (N,H) valid action ids")
    encoded = np.eye(len(ACTION_CLASSES), dtype=np.float32)[values]
    return encoded.reshape(len(values), -1)


def balanced_accuracy(
    predictions: np.ndarray, labels: np.ndarray, classes: np.ndarray
) -> float:
    recalls = []
    for value in classes:
        mask = labels == value
        if mask.any():
            recalls.append(float((predictions[mask] == value).mean()))
    return float(np.mean(recalls)) if recalls else 0.0


class NaturalTreechopBCPolicy:
    """Two-head linear sequence policy over strictly legal observations."""

    def __init__(
        self,
        feature_size: int = 6,
        frame_stack: int = 4,
        action_history: int = 8,
        use_phase_head: bool = True,
    ):
        if feature_size <= 0 or frame_stack <= 0 or action_history <= 0:
            raise ValueError("feature, frame-stack, and action-history sizes must be positive")
        self.feature_size = int(feature_size)
        self.frame_stack = int(frame_stack)
        self.action_history = int(action_history)
        self.use_phase_head = bool(use_phase_head)
        self.model_version = MODEL_VERSION
        self.student_input_manifest = student_input_manifest(
            self.frame_stack, self.action_history
        )
        self.base_mean: Optional[np.ndarray] = None
        self.base_std: Optional[np.ndarray] = None
        self.phase_weights: Optional[np.ndarray] = None
        self.action_weights: Optional[np.ndarray] = None
        self.phase_best_epoch = 0
        self.action_best_epoch = 0

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - logits.max(axis=1, keepdims=True)
        probabilities = np.exp(shifted)
        return probabilities / np.maximum(probabilities.sum(axis=1, keepdims=True), 1e-8)

    def _base_features(
        self,
        pov_stacks: np.ndarray,
        legal_vectors: np.ndarray,
        action_histories: np.ndarray,
        fit_normalization: bool = False,
    ) -> np.ndarray:
        visual = extract_contact_features(
            pov_stacks,
            size=self.feature_size,
            include_centre_pixels=False,
        )
        vectors = np.asarray(legal_vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors[None, :]
        histories = action_history_one_hot(action_histories)
        if not (len(visual) == len(vectors) == len(histories)):
            raise ValueError("POV, legal vector, and action history batches must align")
        features = np.concatenate((visual, vectors, histories), axis=1).astype(np.float32)
        if fit_normalization:
            self.base_mean = features.mean(axis=0)
            self.base_std = features.std(axis=0)
            self.base_std[self.base_std < 1e-5] = 1.0
        if self.base_mean is None or self.base_std is None:
            raise RuntimeError("normalization has not been fitted")
        normalized = (features - self.base_mean) / self.base_std
        return np.concatenate(
            (normalized, np.ones((len(normalized), 1), dtype=np.float32)), axis=1
        )

    @staticmethod
    def _sample_weights(
        labels: np.ndarray, classes: np.ndarray, power: float
    ) -> np.ndarray:
        indices = np.searchsorted(classes, labels)
        counts = np.bincount(indices, minlength=len(classes)).astype(np.float32)
        present = counts > 0
        weights = np.zeros_like(counts)
        mean_count = float(counts[present].mean()) if present.any() else 1.0
        weights[present] = (mean_count / counts[present]) ** float(power)
        result = weights[indices]
        return result / max(float(result.mean()), 1e-8)

    def _fit_head(
        self,
        train_features: np.ndarray,
        train_labels: np.ndarray,
        validation_features: np.ndarray,
        validation_labels: np.ndarray,
        classes: np.ndarray,
        epochs: int,
        learning_rate: float,
        l2: float,
        patience: int,
        class_weight_power: float,
    ) -> Tuple[np.ndarray, List[Dict[str, float]], int]:
        if len(train_labels) == 0 or len(validation_labels) == 0:
            raise ValueError("non-empty train and validation splits required")
        if not np.isin(train_labels, classes).all() or not np.isin(validation_labels, classes).all():
            raise ValueError("labels are outside the declared head classes")
        train_indices = np.searchsorted(classes, train_labels)
        validation_indices = np.searchsorted(classes, validation_labels)
        weights = np.zeros((train_features.shape[1], len(classes)), dtype=np.float32)
        targets = np.eye(len(classes), dtype=np.float32)[train_indices]
        sample_weights = self._sample_weights(
            train_labels, classes, class_weight_power
        )
        sample_weight_sum = float(sample_weights.sum())
        history: List[Dict[str, float]] = []
        best_weights = weights.copy()
        best_loss = float("inf")
        best_epoch = 0
        stale = 0
        for epoch in range(int(epochs) + 1):
            if epoch > 0:
                probabilities = self._softmax(train_features @ weights)
                error = (probabilities - targets) * sample_weights[:, None]
                gradient = train_features.T @ error / sample_weight_sum
                gradient[:-1] += float(l2) * weights[:-1]
                update = float(learning_rate) * gradient
                if not np.all(np.isfinite(update)):
                    raise FloatingPointError("non-finite BC gradient")
                weights -= update
            train_probabilities = self._softmax(train_features @ weights)
            validation_probabilities = self._softmax(validation_features @ weights)
            train_chosen = train_probabilities[np.arange(len(train_indices)), train_indices]
            validation_chosen = validation_probabilities[
                np.arange(len(validation_indices)), validation_indices
            ]
            train_loss = float(
                (-np.log(np.maximum(train_chosen, 1e-8)) * sample_weights).sum()
                / sample_weight_sum
            )
            validation_loss = float(-np.log(np.maximum(validation_chosen, 1e-8)).mean())
            train_predictions = classes[train_probabilities.argmax(axis=1)]
            validation_predictions = classes[validation_probabilities.argmax(axis=1)]
            history.append(
                {
                    "epoch": int(epoch),
                    "train_loss": train_loss,
                    "validation_loss": validation_loss,
                    "train_accuracy": float((train_predictions == train_labels).mean()),
                    "validation_accuracy": float(
                        (validation_predictions == validation_labels).mean()
                    ),
                    "validation_balanced_accuracy": balanced_accuracy(
                        validation_predictions, validation_labels, classes
                    ),
                }
            )
            if validation_loss < best_loss - 1e-5:
                best_loss = validation_loss
                best_weights = weights.copy()
                best_epoch = epoch
                stale = 0
            else:
                stale += 1
            if patience > 0 and stale >= patience:
                break
        return best_weights, history, int(best_epoch)

    def fit(
        self,
        train_pov: np.ndarray,
        train_vectors: np.ndarray,
        train_histories: np.ndarray,
        train_actions: Sequence[int],
        train_phases: Sequence[str],
        validation_pov: np.ndarray,
        validation_vectors: np.ndarray,
        validation_histories: np.ndarray,
        validation_actions: Sequence[int],
        validation_phases: Sequence[str],
        epochs: int = 250,
        learning_rate: float = 0.03,
        l2: float = 1e-3,
        patience: int = 35,
        class_weight_power: float = 0.5,
    ) -> Dict[str, List[Dict[str, float]]]:
        train_actions_array = np.asarray(train_actions, dtype=np.int64)
        validation_actions_array = np.asarray(validation_actions, dtype=np.int64)
        train_phase_ids = phase_ids(train_phases)
        validation_phase_ids = phase_ids(validation_phases)
        train_base = self._base_features(
            train_pov, train_vectors, train_histories, fit_normalization=True
        )
        validation_base = self._base_features(
            validation_pov, validation_vectors, validation_histories
        )
        phase_history: List[Dict[str, float]] = []
        if self.use_phase_head:
            self.phase_weights, phase_history, self.phase_best_epoch = self._fit_head(
                train_base,
                train_phase_ids,
                validation_base,
                validation_phase_ids,
                PHASE_CLASSES,
                epochs,
                learning_rate,
                l2,
                patience,
                class_weight_power,
            )
            train_phase_probabilities = self._softmax(train_base @ self.phase_weights)
            validation_phase_probabilities = self._softmax(
                validation_base @ self.phase_weights
            )
            train_action_features = np.concatenate(
                (train_base, train_phase_probabilities), axis=1
            )
            validation_action_features = np.concatenate(
                (validation_base, validation_phase_probabilities), axis=1
            )
        else:
            self.phase_weights = None
            train_action_features = train_base
            validation_action_features = validation_base
        self.action_weights, action_history, self.action_best_epoch = self._fit_head(
            train_action_features,
            train_actions_array,
            validation_action_features,
            validation_actions_array,
            ACTION_CLASSES,
            epochs,
            learning_rate,
            l2,
            patience,
            class_weight_power,
        )
        return {"phase": phase_history, "action": action_history}

    def probabilities(
        self,
        pov_stacks: np.ndarray,
        legal_vectors: np.ndarray,
        action_histories: np.ndarray,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if self.action_weights is None:
            raise RuntimeError("policy has not been trained or loaded")
        base = self._base_features(pov_stacks, legal_vectors, action_histories)
        phase_probabilities = None
        action_features = base
        if self.use_phase_head:
            if self.phase_weights is None:
                raise RuntimeError("phase-enabled checkpoint has no phase head")
            phase_probabilities = self._softmax(base @ self.phase_weights)
            action_features = np.concatenate((base, phase_probabilities), axis=1)
        return self._softmax(action_features @ self.action_weights), phase_probabilities

    def predict(
        self,
        pov_stack: np.ndarray,
        legal_vector: np.ndarray,
        action_history: np.ndarray,
    ) -> int:
        stack = np.asarray(pov_stack)
        if stack.ndim == 4:
            stack = stack[None, ...]
        probabilities, _ = self.probabilities(
            stack,
            np.asarray(legal_vector)[None, ...]
            if np.asarray(legal_vector).ndim == 1
            else legal_vector,
            np.asarray(action_history)[None, ...]
            if np.asarray(action_history).ndim == 1
            else action_history,
        )
        return int(ACTION_CLASSES[int(probabilities[0].argmax())])

    def save(
        self,
        path: str,
        dataset_hashes: Optional[Dict[str, str]] = None,
        seed_manifest: str = "",
    ) -> None:
        if self.action_weights is None or self.base_mean is None or self.base_std is None:
            raise RuntimeError("cannot save an unfitted policy")
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                model_version=np.array(self.model_version),
                observation_schema=np.array(STUDENT_OBSERVATION_SCHEMA_VERSION),
                feature_size=np.array(self.feature_size),
                frame_stack=np.array(self.frame_stack),
                action_history=np.array(self.action_history),
                use_phase_head=np.array(self.use_phase_head),
                student_input_manifest=np.asarray(self.student_input_manifest),
                base_mean=self.base_mean,
                base_std=self.base_std,
                phase_weights=(
                    self.phase_weights
                    if self.phase_weights is not None
                    else np.zeros((0, 0), dtype=np.float32)
                ),
                action_weights=self.action_weights,
                phase_best_epoch=np.array(self.phase_best_epoch),
                action_best_epoch=np.array(self.action_best_epoch),
                dataset_hashes=np.asarray(
                    ["{}={}".format(key, value) for key, value in (dataset_hashes or {}).items()]
                ),
                seed_manifest=np.array(str(seed_manifest)),
            )
        temporary.replace(output)

    @classmethod
    def load(cls, path: str) -> "NaturalTreechopBCPolicy":
        with np.load(path, allow_pickle=False) as data:
            policy = cls(
                feature_size=int(data["feature_size"]),
                frame_stack=int(data["frame_stack"]),
                action_history=int(data["action_history"]),
                use_phase_head=bool(data["use_phase_head"]),
            )
            policy.model_version = str(data["model_version"])
            policy.base_mean = data["base_mean"].astype(np.float32)
            policy.base_std = data["base_std"].astype(np.float32)
            phase_weights = data["phase_weights"].astype(np.float32)
            policy.phase_weights = phase_weights if phase_weights.size else None
            policy.action_weights = data["action_weights"].astype(np.float32)
            policy.phase_best_epoch = int(data["phase_best_epoch"])
            policy.action_best_epoch = int(data["action_best_epoch"])
            policy.dataset_hashes = dict(
                item.split("=", 1) for item in data["dataset_hashes"].tolist()
            )
            policy.seed_manifest = str(data["seed_manifest"])
        return policy


class NaturalTreechopStudentAgent:
    """Episode-local autonomous actor; no teacher object is accepted."""

    def __init__(self, policy: NaturalTreechopBCPolicy, max_episode_steps: int):
        if policy.action_weights is None:
            raise RuntimeError("student requires a trained policy")
        self.policy = policy
        self.max_episode_steps = int(max_episode_steps)
        self.reset_episode()

    def reset_episode(self) -> None:
        self.observation_adapter = LegalObservationAdapter(self.max_episode_steps)
        self.frames: deque = deque(maxlen=self.policy.frame_stack)
        self.actions: deque = deque(
            [0] * self.policy.action_history,
            maxlen=self.policy.action_history,
        )
        self.started = False

    def act(self, raw_observation: Dict, episode_step: int) -> Tuple[int, np.ndarray]:
        legal = (
            self.observation_adapter.reset(raw_observation)
            if not self.started
            else self.observation_adapter.adapt(raw_observation, episode_step)
        )
        self.started = True
        self.frames.append(legal.pov)
        while len(self.frames) < self.policy.frame_stack:
            self.frames.appendleft(self.frames[0])
        stack = np.stack(tuple(self.frames), axis=0)
        history = np.asarray(tuple(self.actions), dtype=np.int64)
        action = self.policy.predict(stack, legal.vector, history)
        return int(action), legal.vector.copy()

    def observe_transition(self, executed_action: int) -> None:
        self.actions.append(int(executed_action))
