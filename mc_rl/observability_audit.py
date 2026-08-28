"""Episode-safe feature and probe helpers for diagnostic-only audits."""

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

from mc_rl.natural_contact_bc import extract_contact_features
from mc_rl.recurrent_treechop_bc import RecurrentTreechopPolicy
from mc_rl.vision import build_frame_stacks


ALLOWED_AUDIT_DATASET_FIELDS = (
    "pov",
    "legal_vector",
    "action",
    "previous_action",
    "episode",
    "episode_seed",
    "episode_step",
    "episode_success",
    "audit_raycast_is_log",
    "audit_raycast_in_range",
    "audit_raycast_distance",
    "audit_reward",
    "audit_coarse_phase",
    "audit_student_action",
)


def assert_disjoint_episode_splits(
    train_seeds: Sequence[int], validation_seeds: Sequence[int]
) -> None:
    overlap = sorted(set(int(v) for v in train_seeds) & set(int(v) for v in validation_seeds))
    if overlap:
        raise ValueError("episode/seed split leakage: {}".format(overlap))


class TrainOnlyStandardizer:
    def __init__(self) -> None:
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.fit_split: Optional[str] = None

    def fit(self, values: np.ndarray, split: str) -> "TrainOnlyStandardizer":
        if split != "bc_train":
            raise ValueError("scaler may only be fit on bc_train")
        array = np.asarray(values, dtype=np.float32)
        self.mean = array.mean(axis=0)
        self.std = array.std(axis=0)
        self.std[self.std < 1e-6] = 1.0
        self.fit_split = split
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("scaler has not been fitted")
        return (np.asarray(values, dtype=np.float32) - self.mean) / self.std


class TrainOnlyPCA:
    def __init__(self, components: int) -> None:
        self.requested_components = int(components)
        self.mean: Optional[np.ndarray] = None
        self.components: Optional[np.ndarray] = None
        self.fit_split: Optional[str] = None

    def fit(self, values: np.ndarray, split: str) -> "TrainOnlyPCA":
        if split != "bc_train":
            raise ValueError("PCA may only be fit on bc_train")
        array = np.asarray(values, dtype=np.float32)
        self.mean = array.mean(axis=0)
        centred = array - self.mean
        rank = max(1, min(self.requested_components, centred.shape[0] - 1, centred.shape[1]))
        torch.manual_seed(0)
        with torch.no_grad():
            _, _, vectors = torch.pca_lowrank(
                torch.from_numpy(centred), q=rank, center=False, niter=4
            )
        self.components = vectors.cpu().numpy().astype(np.float32)
        self.fit_split = split
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.mean is None or self.components is None:
            raise RuntimeError("PCA has not been fitted")
        return (np.asarray(values, dtype=np.float32) - self.mean) @ self.components


def load_audit_dataset(path: Path) -> Dict[str, np.ndarray]:
    """Load only explicitly named legal/audit arrays from an allowed path."""

    with np.load(Path(path), allow_pickle=False) as values:
        result = {
            name: np.asarray(values[name])
            for name in ALLOWED_AUDIT_DATASET_FIELDS
            if name in values.files
        }
    required = ("pov", "legal_vector", "action", "previous_action", "episode", "episode_seed", "episode_step", "episode_success")
    missing = [name for name in required if name not in result]
    if missing:
        raise KeyError("dataset is missing required diagnostic fields: {}".format(missing))
    return result


def labels_from_dataset(dataset: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    count = len(dataset["episode_step"])
    is_log = np.asarray(dataset.get("audit_raycast_is_log", np.full(count, -1)), dtype=np.int8)
    in_range = np.asarray(dataset.get("audit_raycast_in_range", np.full(count, -1)), dtype=np.int8)
    distance = np.asarray(dataset.get("audit_raycast_distance", np.full(count, np.nan)), dtype=np.float32)
    result = {
        "tree_visible": np.where(is_log >= 0, is_log, -1).astype(np.int8),
        "contact_range": np.where(np.isfinite(distance), distance <= 4.5, -1).astype(np.int8),
        "valid_attack_geometry": np.where(
            (is_log >= 0) & (in_range >= 0), (is_log.astype(bool) & in_range.astype(bool)).astype(np.int8), -1
        ).astype(np.int8),
    }
    trend = np.full(count, -1, dtype=np.int8)
    episodes = np.asarray(dataset["episode"])
    for index in range(3, count):
        if episodes[index] != episodes[index - 3]:
            continue
        window = distance[index - 3 : index + 1]
        if not np.isfinite(window).all():
            continue
        delta = float(window[-1] - window[0])
        trend[index] = 0 if delta <= -0.1 else (1 if abs(delta) < 0.1 else 2)
    result["approach_dynamics"] = trend
    return result


def causal_stacks(dataset: Mapping[str, np.ndarray], stack: int = 4) -> np.ndarray:
    return build_frame_stacks(dataset["pov"], dataset["episode"], stack)


def downsample_flat(frames: np.ndarray, size: int = 16) -> np.ndarray:
    values = np.asarray(frames)
    if values.ndim == 4:
        values = values[:, None]
    rows = []
    for stack in values:
        rows.append(
            np.concatenate(
                [cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA).reshape(-1) for frame in stack]
            )
        )
    return np.asarray(rows, dtype=np.float32) / 255.0


def frozen_recurrent_features(
    dataset: Mapping[str, np.ndarray], checkpoint: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Return current-frame CNN embeddings and teacher-forced GRU hidden."""

    policy = RecurrentTreechopPolicy.load(checkpoint)
    count = len(dataset["episode_step"])
    cnn = np.zeros((count, policy.architecture.spatial_embedding), dtype=np.float32)
    hidden_rows = np.zeros((count, policy.architecture.hidden_size), dtype=np.float32)
    episodes = np.asarray(dataset["episode"])
    actions = np.asarray(dataset["action"], dtype=np.int64)
    executed_actions = np.asarray(
        dataset.get("audit_student_action", actions), dtype=np.int64
    )
    stored_previous = np.asarray(dataset["previous_action"], dtype=np.int64)
    for episode in np.unique(episodes):
        indices = np.flatnonzero(episodes == episode)
        tokens = stored_previous[indices].copy()
        tokens[0] = 14
        if len(indices) > 1 and not np.array_equal(
            tokens[1:], executed_actions[indices[:-1]]
        ):
            raise ValueError("teacher-forced previous action is not causal")
        with torch.no_grad():
            _, hidden, diagnostics = policy.model.forward_with_diagnostics(
                torch.from_numpy(np.asarray(dataset["pov"])[indices])[None],
                torch.from_numpy(np.asarray(dataset["legal_vector"], dtype=np.float32)[indices])[None],
                torch.from_numpy(tokens.astype(np.int64))[None],
            )
        cnn[indices] = diagnostics["cnn_embedding"][0].cpu().numpy()
        hidden_rows[indices] = diagnostics["recurrent_output"][0].cpu().numpy()
    return cnn, hidden_rows


def feature_spaces(
    dataset: Mapping[str, np.ndarray], checkpoint: str
) -> Dict[str, np.ndarray]:
    stacks = causal_stacks(dataset, 4)
    cnn, hidden = frozen_recurrent_features(dataset, checkpoint)
    return {
        "F1_legal_vector": np.asarray(dataset["legal_vector"], dtype=np.float32),
        "F2_current_rgb": downsample_flat(dataset["pov"], 16),
        "F3_four_frame_stack": downsample_flat(stacks, 16),
        "F4_handwritten": extract_contact_features(stacks, size=6, include_centre_pixels=False),
        "F5_cnn_embedding": cnn,
        "F6_gru_hidden": hidden,
    }


def binary_metrics(labels: np.ndarray, probabilities: np.ndarray) -> Dict[str, Any]:
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(probabilities, dtype=np.float64)
    predictions = (p >= 0.5).astype(np.int64)
    tn = int(((y == 0) & (predictions == 0)).sum())
    fp = int(((y == 0) & (predictions == 1)).sum())
    fn = int(((y == 1) & (predictions == 0)).sum())
    tp = int(((y == 1) & (predictions == 1)).sum())
    tpr = tp / max(tp + fn, 1)
    tnr = tn / max(tn + fp, 1)
    f1_pos = 2 * tp / max(2 * tp + fp + fn, 1)
    f1_neg = 2 * tn / max(2 * tn + fp + fn, 1)
    order = np.argsort(-p, kind="mergesort")
    sorted_y = y[order]
    positives = max(int((y == 1).sum()), 1)
    negatives = max(int((y == 0).sum()), 1)
    tps = np.cumsum(sorted_y == 1) / positives
    fps = np.cumsum(sorted_y == 0) / negatives
    auroc = float(np.trapz(np.r_[0.0, tps], np.r_[0.0, fps]))
    precision = np.cumsum(sorted_y == 1) / np.arange(1, len(y) + 1)
    recall = np.cumsum(sorted_y == 1) / positives
    pr_auc = float(np.sum((recall - np.r_[0.0, recall[:-1]]) * precision))
    return {
        "balanced_accuracy": 0.5 * (tpr + tnr),
        "macro_f1": 0.5 * (f1_pos + f1_neg),
        "auroc": auroc,
        "pr_auc": pr_auc,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


class FixedLogisticProbe:
    def __init__(self, epochs: int = 400, learning_rate: float = 0.05, l2: float = 1e-3):
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.l2 = float(l2)
        self.weights: Optional[np.ndarray] = None

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "FixedLogisticProbe":
        x = np.c_[np.asarray(features, dtype=np.float32), np.ones(len(features), dtype=np.float32)]
        y = np.asarray(labels, dtype=np.float32)
        positives = max(float((y == 1).sum()), 1.0)
        negatives = max(float((y == 0).sum()), 1.0)
        sample_weights = np.where(y == 1, len(y) / (2 * positives), len(y) / (2 * negatives)).astype(np.float32)
        weights = np.zeros(x.shape[1], dtype=np.float32)
        for _ in range(self.epochs):
            logits = np.clip(x @ weights, -30.0, 30.0)
            probabilities = 1.0 / (1.0 + np.exp(-logits))
            gradient = x.T @ ((probabilities - y) * sample_weights) / sample_weights.sum()
            gradient[:-1] += self.l2 * weights[:-1]
            weights -= self.learning_rate * gradient
        self.weights = weights
        return self

    def probabilities(self, features: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("probe has not been fitted")
        x = np.c_[np.asarray(features, dtype=np.float32), np.ones(len(features), dtype=np.float32)]
        logits = np.clip(x @ self.weights, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-logits))


class FixedRidgeProbe:
    def __init__(self, l2: float = 0.01):
        self.l2 = float(l2)
        self.weights: Optional[np.ndarray] = None

    def fit(self, features: np.ndarray, targets: np.ndarray) -> "FixedRidgeProbe":
        x = np.c_[np.asarray(features, dtype=np.float64), np.ones(len(features))]
        y = np.asarray(targets, dtype=np.float64)
        penalty = np.eye(x.shape[1], dtype=np.float64) * self.l2
        penalty[-1, -1] = 0.0
        self.weights = np.linalg.solve(x.T @ x + penalty, x.T @ y)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("ridge probe has not been fitted")
        x = np.c_[np.asarray(features, dtype=np.float64), np.ones(len(features))]
        return x @ self.weights


def regression_metrics(targets: np.ndarray, predictions: np.ndarray) -> Dict[str, float]:
    y = np.asarray(targets, dtype=np.float64)
    prediction = np.asarray(predictions, dtype=np.float64)
    residual = prediction - y
    denominator = float(np.sum((y - y.mean()) ** 2))
    return {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "r2": float(1.0 - np.sum(residual ** 2) / denominator) if denominator > 0 else 0.0,
    }


def knn_probabilities(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    query_features: np.ndarray,
    k: int = 5,
) -> np.ndarray:
    reference = np.asarray(train_features, dtype=np.float32)
    query = np.asarray(query_features, dtype=np.float32)
    labels = np.asarray(train_labels, dtype=np.float32)
    result = np.zeros(len(query), dtype=np.float32)
    for start in range(0, len(query), 256):
        block = query[start : start + 256]
        distances = (
            (block * block).sum(axis=1, keepdims=True)
            + (reference * reference).sum(axis=1)[None, :]
            - 2.0 * block @ reference.T
        )
        indices = np.argpartition(distances, min(k, len(reference)) - 1, axis=1)[:, :k]
        result[start : start + len(block)] = labels[indices].mean(axis=1)
    return result
