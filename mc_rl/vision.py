"""Small NumPy visual policy used for the first FindTree learning check.

This deliberately avoids adding PyTorch before the custom environment and
labels are proven useful. It is a linear softmax policy over spatial RGB and
gradient features, not the final CNN/recurrent navigation architecture.
"""

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


VISUAL_ACTION_CLASSES = np.array([1, 3, 4], dtype=np.int64)  # forward, left, right


def clockwise_search_action(predicted_action: int) -> int:
    """Keep visual search direction consistent until forward is predicted.

    A POV-only policy cannot infer which way a hidden target lies. Mapping both
    turn predictions to the teacher's fixed clockwise action prevents an empty
    background from causing a left/right limit cycle. Forward remains entirely
    controlled by the visual model.
    """

    action = int(predicted_action)
    if action not in VISUAL_ACTION_CLASSES:
        raise ValueError("unsupported visual-policy action: {}".format(action))
    return 1 if action == 1 else 4


def extract_visual_features(pov_batch: np.ndarray, size: int = 10) -> np.ndarray:
    """Convert RGB POV frames into compact spatial colour/edge features."""

    frames = np.asarray(pov_batch)
    if frames.ndim == 3:
        frames = frames[None, ...]
    if frames.ndim == 5:
        if frames.shape[-1] != 3:
            raise ValueError("stacked POV must end with an RGB dimension")
        return np.concatenate(
            [extract_visual_features(frames[:, index], size) for index in range(frames.shape[1])],
            axis=1,
        )
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(
            "POV input must have shape (N,T,H,W,3), (N,H,W,3), or (H,W,3)"
        )

    rows = []
    for frame in frames:
        rgb = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA).astype(
            np.float32
        ) / 255.0
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        rows.append(np.concatenate((rgb.ravel(), grad_x.ravel(), grad_y.ravel())))
    return np.asarray(rows, dtype=np.float32)


def navigation_class_labels(actions: Sequence[int]) -> np.ndarray:
    """Map forward+jump teacher actions to forward for the flat curriculum."""

    labels = np.asarray(actions, dtype=np.int64).copy()
    labels[labels == 2] = 1
    if not np.isin(labels, VISUAL_ACTION_CLASSES).all():
        invalid = sorted(set(labels.tolist()) - set(VISUAL_ACTION_CLASSES.tolist()))
        raise ValueError("unsupported visual-policy labels: {}".format(invalid))
    return labels


def build_frame_stacks(
    frames: np.ndarray, episode_ids: Sequence[int], frame_stack: int
) -> np.ndarray:
    """Build causal frame stacks without crossing episode boundaries."""

    frames = np.asarray(frames)
    episode_ids = np.asarray(episode_ids)
    if frame_stack <= 0 or len(frames) != len(episode_ids):
        raise ValueError("frame_stack must be positive and episode IDs must align")
    if frame_stack == 1:
        return frames
    stacked = []
    episode_start = 0
    for index in range(len(frames)):
        if index == 0 or episode_ids[index] != episode_ids[index - 1]:
            episode_start = index
        indices = [max(episode_start, index - offset) for offset in reversed(range(frame_stack))]
        stacked.append(frames[indices])
    return np.asarray(stacked, dtype=frames.dtype)


class LinearVisualPolicy:
    """Deterministic linear softmax policy with explicit normalization."""

    def __init__(self, feature_size: int = 10, frame_stack: int = 1):
        self.feature_size = int(feature_size)
        self.frame_stack = int(frame_stack)
        if self.frame_stack <= 0:
            raise ValueError("frame_stack must be positive")
        self.classes = VISUAL_ACTION_CLASSES.copy()
        self.mean = None
        self.std = None
        self.weights = None
        self.forward_bias = 0.0

    def _normalized_features(self, pov_batch: np.ndarray, fit: bool = False):
        features = extract_visual_features(pov_batch, self.feature_size)
        if fit:
            self.mean = features.mean(axis=0)
            self.std = features.std(axis=0)
            self.std[self.std < 1e-5] = 1.0
        if self.mean is None or self.std is None:
            raise RuntimeError("policy normalization has not been fitted")
        normalized = (features - self.mean) / self.std
        return np.concatenate(
            (normalized, np.ones((len(normalized), 1), dtype=np.float32)), axis=1
        )

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - logits.max(axis=1, keepdims=True)
        probabilities = np.exp(shifted)
        return probabilities / probabilities.sum(axis=1, keepdims=True)

    def fit(
        self,
        train_pov: np.ndarray,
        train_actions: Sequence[int],
        validation_pov: np.ndarray,
        validation_actions: Sequence[int],
        epochs: int = 200,
        learning_rate: float = 0.08,
        l2: float = 1e-4,
        patience: Optional[int] = 30,
    ) -> List[Dict[str, float]]:
        """Train with class-balanced loss and restore the best held-out model."""

        if epochs <= 0 or learning_rate <= 0 or l2 < 0:
            raise ValueError("epochs/learning_rate must be positive and l2 non-negative")
        train_labels = navigation_class_labels(train_actions)
        validation_labels = navigation_class_labels(validation_actions)
        x_train = self._normalized_features(train_pov, fit=True)
        x_validation = self._normalized_features(validation_pov)
        class_indices = np.searchsorted(self.classes, train_labels)
        validation_indices = np.searchsorted(self.classes, validation_labels)

        class_counts = np.bincount(class_indices, minlength=len(self.classes)).astype(
            np.float32
        )
        class_weights = np.zeros_like(class_counts)
        present = class_counts > 0
        class_weights[present] = len(train_labels) / (
            present.sum() * class_counts[present]
        )
        sample_weights = class_weights[class_indices]
        weight_sum = float(sample_weights.sum())

        self.weights = np.zeros(
            (x_train.shape[1], len(self.classes)), dtype=np.float32
        )
        targets = np.eye(len(self.classes), dtype=np.float32)[class_indices]
        history = []
        best_validation_loss = float("inf")
        best_weights = self.weights.copy()
        best_epoch = 0
        epochs_without_improvement = 0
        for epoch in range(epochs + 1):
            if epoch > 0:
                probabilities = self._softmax(x_train @ self.weights)
                error = (probabilities - targets) * sample_weights[:, None]
                gradient = x_train.T @ error / weight_sum
                gradient[:-1] += l2 * self.weights[:-1]
                self.weights -= learning_rate * gradient

            train_metrics = self.metrics_from_features(x_train, class_indices)
            validation_metrics = self.metrics_from_features(
                x_validation, validation_indices
            )
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_metrics[0],
                    "train_accuracy": train_metrics[1],
                    "validation_loss": validation_metrics[0],
                    "validation_accuracy": validation_metrics[1],
                }
            )
            if validation_metrics[0] < best_validation_loss - 1e-5:
                best_validation_loss = validation_metrics[0]
                best_weights = self.weights.copy()
                best_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if (
                patience is not None
                and patience > 0
                and epochs_without_improvement >= patience
            ):
                break

        # The serialized policy is the minimum validation-loss checkpoint, not
        # the last optimization step. This is essential for the tiny first
        # curriculum dataset, where continuing to fit quickly overfits.
        self.weights = best_weights
        self.best_epoch = best_epoch
        self.best_validation_loss = best_validation_loss
        self.stopped_early = len(history) < epochs + 1
        return history

    def metrics_from_features(
        self, features: np.ndarray, class_indices: np.ndarray
    ) -> Tuple[float, float]:
        probabilities = self._softmax(features @ self.weights)
        chosen = probabilities[np.arange(len(class_indices)), class_indices]
        loss = float(-np.log(np.maximum(chosen, 1e-8)).mean())
        accuracy = float((probabilities.argmax(axis=1) == class_indices).mean())
        return loss, accuracy

    def predict_proba(self, pov_batch: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("policy has not been trained or loaded")
        features = self._normalized_features(pov_batch)
        return self._softmax(features @ self.weights)

    def predict(self, pov_batch: np.ndarray) -> np.ndarray:
        probabilities = self.predict_proba(pov_batch)
        calibrated = probabilities.copy()
        forward_index = int(np.where(self.classes == 1)[0][0])
        calibrated[:, forward_index] += float(self.forward_bias)
        return self.classes[calibrated.argmax(axis=1)]

    def calibrate_forward_bias(
        self,
        validation_pov: np.ndarray,
        validation_actions: Sequence[int],
        candidates: Optional[Sequence[float]] = None,
    ) -> Dict[str, float]:
        """Choose a small forward bias by held-out balanced accuracy.

        Class-balanced training makes turn logits deliberately strong. This
        calibration restores deployment priors without using oracle data at
        inference time.
        """

        labels = navigation_class_labels(validation_actions)
        probabilities = self.predict_proba(validation_pov)
        if candidates is None:
            candidates = np.linspace(0.0, 0.8, 17)
        best = None
        forward_index = int(np.where(self.classes == 1)[0][0])
        for bias in candidates:
            calibrated = probabilities.copy()
            calibrated[:, forward_index] += float(bias)
            predictions = self.classes[calibrated.argmax(axis=1)]
            recalls = [
                float(np.mean(predictions[labels == action] == action))
                for action in self.classes
                if np.any(labels == action)
            ]
            balanced_accuracy = float(np.mean(recalls))
            overall_accuracy = float(np.mean(predictions == labels))
            candidate = (balanced_accuracy, overall_accuracy, -float(bias))
            if best is None or candidate > best[0]:
                best = (candidate, float(bias), recalls)
        self.forward_bias = best[1]
        return {
            "forward_bias": self.forward_bias,
            "calibrated_balanced_accuracy": best[0][0],
            "calibrated_accuracy": best[0][1],
        }

    def save(self, path: str) -> None:
        if self.weights is None:
            raise RuntimeError("cannot save an untrained policy")
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            feature_size=np.array(self.feature_size),
            frame_stack=np.array(self.frame_stack),
            classes=self.classes,
            mean=self.mean,
            std=self.std,
            weights=self.weights,
            forward_bias=np.array(self.forward_bias, dtype=np.float32),
        )

    @classmethod
    def load(cls, path: str):
        data = np.load(path)
        frame_stack = int(data["frame_stack"]) if "frame_stack" in data.files else 1
        policy = cls(feature_size=int(data["feature_size"]), frame_stack=frame_stack)
        policy.classes = data["classes"].astype(np.int64)
        policy.mean = data["mean"].astype(np.float32)
        policy.std = data["std"].astype(np.float32)
        policy.weights = data["weights"].astype(np.float32)
        if "forward_bias" in data.files:
            policy.forward_bias = float(data["forward_bias"])
        return policy


def trend_summary(history: Sequence[Dict[str, float]], tail: int = 20) -> Dict[str, float]:
    """Summarize whether held-out loss improves and settles rather than diverges."""

    if len(history) < 2:
        raise ValueError("history needs at least two points")
    losses = np.array([row["validation_loss"] for row in history], dtype=float)
    accuracies = np.array(
        [row["validation_accuracy"] for row in history], dtype=float
    )
    tail_losses = losses[-min(tail, len(losses)) :]
    relative_improvement = float((losses[0] - losses[-1]) / max(losses[0], 1e-8))
    best_index = int(losses.argmin())
    best_relative_improvement = float(
        (losses[0] - losses[best_index]) / max(losses[0], 1e-8)
    )
    return {
        "initial_validation_loss": float(losses[0]),
        "final_validation_loss": float(losses[-1]),
        "relative_validation_loss_improvement": relative_improvement,
        "best_validation_loss": float(losses[best_index]),
        "best_validation_loss_epoch": best_index,
        "best_relative_validation_loss_improvement": best_relative_improvement,
        "best_validation_accuracy": float(accuracies.max()),
        "final_validation_accuracy": float(accuracies[-1]),
        "tail_validation_loss_std": float(tail_losses.std()),
        "stable_improving_trend": bool(
            best_relative_improvement >= 0.10 and best_index > 0
        ),
    }
