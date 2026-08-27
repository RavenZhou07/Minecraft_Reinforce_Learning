"""Visual behaviour-cloning policy for the natural Treechop contact phase.

The model reproduces only the local contact controller of the v9.6 scripted
teacher: it is consulted while the contact owner holds action ownership and
is deliberately blind to everything else. The declared student input is a
4-frame causal POV stack plus the previous discrete action; raycast, exact
log coordinates, the log grid, oracle distance, and teacher contact state
are never accepted. The first version is a linear softmax over NumPy/OpenCV
spatial colour, edge, centre-patch, and inter-frame motion features, reusing
the optimisation style of ``mc_rl.vision`` without the flat-arena three-class
label compression.
"""

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


MODEL_VERSION = "natural_treechop_contact_bc_v1"
ACTION_CLASSES = np.arange(14, dtype=np.int64)
FRAME_STACK = 4
STUDENT_INPUT_MANIFEST = (
    "pov_frame_stack_4",
    "previous_action_one_hot_14",
)
# Seeds consumed by v9.5/v9.6 development and gates; never valid for BC
# training, validation, or a fresh holdout claim.
BANNED_SEED_RANGE = (16500, 16819)


def assert_seed_isolation(
    train_seeds: Sequence[int],
    validation_seeds: Sequence[int],
) -> None:
    """Fail loudly on seed overlap or any banned development/gate seed."""

    train = [int(seed) for seed in train_seeds]
    validation = [int(seed) for seed in validation_seeds]
    overlap = sorted(set(train) & set(validation))
    if overlap:
        raise ValueError(
            "train/validation seed overlap: {}".format(overlap)
        )
    low, high = BANNED_SEED_RANGE
    for seed in train + validation:
        if low <= seed <= high:
            raise ValueError(
                "seed {} is inside the banned development/gate range "
                "{}-{}".format(seed, low, high)
            )

# Horizontal mirror must swap yaw actions and leave everything else fixed.
MIRROR_ACTION_MAP = {
    0: 0,   # noop
    1: 1,   # forward
    2: 2,   # forward_jump
    3: 4,   # turn_left <-> turn_right
    4: 3,
    5: 5,   # look_up
    6: 6,   # look_down
    7: 7,   # attack
    8: 8,   # forward_attack
    9: 9,   # backward
    10: 11,  # fine_turn_left <-> fine_turn_right
    11: 10,
    12: 12, # fine_look_up
    13: 13, # fine_look_down
}

_CENTRE_CROP = 16
_TRUNK_BGR_LO = np.array([40, 90, 90], dtype=np.float32)
_TRUNK_BGR_HI = np.array([90, 150, 160], dtype=np.float32)


class StudentObservation(dict):
    """POV-only guarded view; any privileged key access fails loudly."""

    def __missing__(self, key):
        raise KeyError(
            "student input must not access observation key {!r}".format(key)
        )


def student_observation(pov: np.ndarray) -> StudentObservation:
    return StudentObservation(pov=np.asarray(pov))


def mirror_actions(actions: Sequence[int]) -> np.ndarray:
    """Apply the verified horizontal-mirror action mapping."""

    mapped = np.array([MIRROR_ACTION_MAP[int(a)] for a in actions], dtype=np.int64)
    return mapped


def mirror_pov_frames(frames: np.ndarray) -> np.ndarray:
    """Horizontally flip (N, H, W, 3) or (H, W, 3) RGB frames."""

    return np.asarray(frames)[..., ::-1, :]


def extract_contact_features(
    pov_batch: np.ndarray, size: int = 10, include_centre_pixels: bool = True
) -> np.ndarray:
    """Spatial colour, edge, centre-patch, and motion features per frame.

    Input shape is (N, T, H, W, 3) for a T-frame causal stack; output is
    (N, D). Centre-patch features capture the crosshair contact cue that
    dominates the contact controller, and inter-frame differences give the
    classifier motion evidence without any privileged state. Setting
    ``include_centre_pixels`` to False keeps only the scalar centre trunk
    fraction, which removes the most episode-memorisable raw-pixel block.
    """

    frames = np.asarray(pov_batch)
    if frames.ndim == 4:
        frames = frames[None, ...]
    if frames.ndim != 5 or frames.shape[-1] != 3:
        raise ValueError(
            "contact features expect (N, T, H, W, 3) POV stacks, got shape {}".format(
                frames.shape
            )
        )
    count, stack = frames.shape[0], frames.shape[1]
    per_frame_rows: List[np.ndarray] = []
    for index in range(stack):
        per_frame_rows.append(
            _single_frame_features(
                frames[:, index], size, include_centre_pixels
            )
        )
    features = np.concatenate(per_frame_rows, axis=1)
    # Motion features: channel-mean absolute differences between adjacent
    # causal frames, plus their centre-patch restricted versions.
    motion_rows = []
    for index in range(1, stack):
        resized_prev = _resize_batch(frames[:, index - 1], size)
        resized_now = _resize_batch(frames[:, index], size)
        diff = np.abs(resized_now - resized_prev).mean(axis=(1, 2))
        centre_prev = _centre_crop_batch(frames[:, index - 1])
        centre_now = _centre_crop_batch(frames[:, index])
        centre_diff = np.abs(centre_now - centre_prev).mean(axis=(1, 2))
        motion_rows.append(np.concatenate((diff, centre_diff), axis=1))
    if motion_rows:
        features = np.concatenate(
            [features] + [np.asarray(row, dtype=np.float32) for row in motion_rows],
            axis=1,
        )
    return np.asarray(features, dtype=np.float32)


def _resize_batch(batch: np.ndarray, size: int) -> np.ndarray:
    rows = []
    for frame in batch:
        rows.append(
            cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
        )
    return np.asarray(rows, dtype=np.float32) / 255.0


def _centre_crop_batch(batch: np.ndarray) -> np.ndarray:
    height, width = batch.shape[1], batch.shape[2]
    top = max(0, (height - _CENTRE_CROP) // 2)
    left = max(0, (width - _CENTRE_CROP) // 2)
    return batch[
        :, top : top + _CENTRE_CROP, left : left + _CENTRE_CROP
    ].astype(np.float32)


def _trunk_mask_statistics(frame_rgb: np.ndarray) -> np.ndarray:
    """POV-derived geometric statistics of trunk-coloured pixels.

    The contact controller's decisions reduce to where the trunk sits in the
    frame (yaw/pitch corrections), how large it appears (attack reach), and
    how much of the crosshair patch it fills (attack confirmation). These
    five numbers encode exactly that from pixels alone: total mask fraction,
    centroid offsets from the image centre, and left/right plus top/bottom
    mask asymmetries.
    """

    frame = np.asarray(frame_rgb, dtype=np.uint8)
    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError("trunk mask expects an RGB frame")
    height, width = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    hue = hsv[..., 0]
    saturation = hsv[..., 1]
    value = hsv[..., 2]
    mask = (
        (hue >= 5)
        & (hue <= 25)
        & (saturation >= 60)
        & (value >= 40)
        & (value <= 230)
    )
    fraction = float(mask.mean())
    if int(mask.sum()) < 4:
        return np.zeros(5, dtype=np.float32)
    ys, xs = np.nonzero(mask)
    centre_x = float(xs.mean()) / width - 0.5
    centre_y = float(ys.mean()) / height - 0.5
    left = float(mask[:, : width // 2].mean())
    right = float(mask[:, width // 2 :].mean())
    top = float(mask[: height // 2, :].mean())
    bottom = float(mask[height // 2 :, :].mean())
    return np.array(
        [fraction, centre_x, centre_y, left - right, top - bottom],
        dtype=np.float32,
    )


def _single_frame_features(
    batch: np.ndarray, size: int, include_centre_pixels: bool = True
) -> np.ndarray:
    resized = _resize_batch(batch, size)
    rows = []
    for rgb in resized:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        rows.append(np.concatenate((rgb.ravel(), grad_x.ravel(), grad_y.ravel())))
    spatial = np.asarray(rows, dtype=np.float32)

    centre = _centre_crop_batch(batch)
    centre_rows = []
    for patch in centre:
        bgr = patch[..., ::-1]
        in_range = (
            (bgr >= _TRUNK_BGR_LO) & (bgr <= _TRUNK_BGR_HI)
        ).all(axis=-1)
        trunk_fraction = float(in_range.mean())
        if include_centre_pixels:
            centre_rows.append(
                np.concatenate(
                    (
                        patch.reshape(-1),
                        np.array([trunk_fraction], dtype=np.float32),
                    )
                )
            )
        else:
            centre_rows.append(
                np.array([trunk_fraction], dtype=np.float32)
            )
    centre_features = np.asarray(centre_rows, dtype=np.float32)
    mask_features = np.asarray(
        [_trunk_mask_statistics(frame) for frame in batch], dtype=np.float32
    )
    return np.concatenate(
        (spatial, centre_features, mask_features), axis=1
    )


def previous_action_one_hot(
    previous_actions: Sequence[int], classes: np.ndarray = ACTION_CLASSES
) -> np.ndarray:
    """Encode the previously executed discrete action as one-hot."""

    actions = np.asarray(previous_actions, dtype=np.int64)
    if actions.ndim == 0:
        actions = actions[None]
    index = np.searchsorted(classes, actions)
    if np.any(index >= len(classes)) or np.any(classes[np.minimum(index, len(classes) - 1)] != actions):
        raise ValueError("previous actions must be valid discrete ids")
    one_hot = np.zeros((len(actions), len(classes)), dtype=np.float32)
    one_hot[np.arange(len(actions)), index] = 1.0
    return one_hot


class NaturalContactBCPolicy:
    """Linear softmax policy over POV-stack and previous-action features."""

    def __init__(
        self,
        feature_size: int = 10,
        frame_stack: int = FRAME_STACK,
        include_centre_pixels: bool = True,
    ):
        self.feature_size = int(feature_size)
        self.frame_stack = int(frame_stack)
        self.include_centre_pixels = bool(include_centre_pixels)
        if self.frame_stack <= 0:
            raise ValueError("frame_stack must be positive")
        self.classes = ACTION_CLASSES.copy()
        self.mean = None
        self.std = None
        self.weights = None
        self.student_input_manifest = tuple(STUDENT_INPUT_MANIFEST)
        self.model_version = MODEL_VERSION

    # ------------------------------------------------------------------
    # Feature plumbing
    # ------------------------------------------------------------------

    def build_features(
        self,
        pov_stacks: np.ndarray,
        previous_actions: Sequence[int],
        fit_normalization: bool = False,
    ) -> np.ndarray:
        visual = extract_contact_features(
            pov_stacks, self.feature_size, self.include_centre_pixels
        )
        # The output label set may be narrowed by a hybrid student, but the
        # previous action is always one of the environment's fourteen
        # discrete actions.  Keeping these two vocabularies separate lets a
        # reduced-action policy observe scripted actions without either
        # treating them as prediction targets or changing the v1 feature
        # layout.
        one_hot = previous_action_one_hot(previous_actions, ACTION_CLASSES)
        features = np.concatenate((visual, one_hot), axis=1).astype(np.float32)
        if fit_normalization:
            self.mean = features.mean(axis=0)
            self.std = features.std(axis=0)
            self.std[self.std < 1e-5] = 1.0
        if self.mean is None or self.std is None:
            raise RuntimeError("normalization has not been fitted")
        normalized = (features - self.mean) / self.std
        return np.concatenate(
            (normalized, np.ones((len(normalized), 1), dtype=np.float32)),
            axis=1,
        )

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - logits.max(axis=1, keepdims=True)
        probabilities = np.exp(shifted)
        return probabilities / probabilities.sum(axis=1, keepdims=True)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        train_pov: np.ndarray,
        train_actions: Sequence[int],
        train_previous_actions: Sequence[int],
        validation_pov: np.ndarray,
        validation_actions: Sequence[int],
        validation_previous_actions: Sequence[int],
        epochs: int = 300,
        learning_rate: float = 0.05,
        l2: float = 1e-4,
        patience: Optional[int] = 40,
        momentum: float = 0.0,
    ) -> List[Dict[str, float]]:
        """Class-balanced training with early stopping on validation loss."""

        if epochs <= 0 or learning_rate <= 0 or l2 < 0:
            raise ValueError("epochs/lr positive and l2 non-negative required")
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must lie in [0, 1)")
        train_labels = self._validate_labels(train_actions)
        validation_labels = self._validate_labels(validation_actions)
        x_train = self.build_features(
            train_pov, train_previous_actions, fit_normalization=True
        )
        x_validation = self.build_features(
            validation_pov, validation_previous_actions
        )
        train_indices = np.searchsorted(self.classes, train_labels)
        validation_indices = np.searchsorted(self.classes, validation_labels)
        class_counts = np.bincount(
            train_indices, minlength=len(self.classes)
        ).astype(np.float32)
        class_weights = np.zeros_like(class_counts)
        present = class_counts > 0
        class_weights[present] = len(train_labels) / (
            present.sum() * class_counts[present]
        )
        sample_weights = class_weights[train_indices]
        weight_sum = float(sample_weights.sum())
        self.weights = np.zeros(
            (x_train.shape[1], len(self.classes)), dtype=np.float32
        )
        targets = np.eye(len(self.classes), dtype=np.float32)[train_indices]
        history: List[Dict[str, float]] = []
        best_validation_loss = float("inf")
        best_weights = self.weights.copy()
        best_epoch = 0
        epochs_without_improvement = 0
        velocity = np.zeros_like(self.weights)
        for epoch in range(epochs + 1):
            if epoch > 0:
                probabilities = self._softmax(x_train @ self.weights)
                error = (probabilities - targets) * sample_weights[:, None]
                gradient = x_train.T @ error / weight_sum
                gradient[:-1] += l2 * self.weights[:-1]
                velocity = momentum * velocity + gradient
                update = learning_rate * velocity
                if not np.all(np.isfinite(update)):
                    raise FloatingPointError(
                        "non-finite gradient at epoch {}".format(epoch)
                    )
                self.weights -= update
            if not np.all(np.isfinite(self.weights)):
                raise FloatingPointError("non-finite weights at epoch {}".format(epoch))
            train_loss, train_accuracy = self._metrics(
                x_train, train_indices, sample_weights
            )
            validation_loss, validation_accuracy = self._metrics(
                x_validation, validation_indices
            )
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "validation_loss": validation_loss,
                    "validation_accuracy": validation_accuracy,
                }
            )
            if validation_loss < best_validation_loss - 1e-5:
                best_validation_loss = validation_loss
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
        self.weights = best_weights
        self.best_epoch = best_epoch
        self.best_validation_loss = best_validation_loss
        self.stopped_early = len(history) < epochs + 1
        return history

    def _validate_labels(self, actions: Sequence[int]) -> np.ndarray:
        labels = np.asarray(actions, dtype=np.int64)
        if len(labels) and not np.isin(labels, self.classes).all():
            invalid = sorted(set(labels.tolist()) - set(self.classes.tolist()))
            raise ValueError("unsupported contact action labels: {}".format(invalid))
        return labels

    def _metrics(
        self,
        features: np.ndarray,
        indices: np.ndarray,
        sample_weights: Optional[np.ndarray] = None,
    ) -> Tuple[float, float]:
        probabilities = self._softmax(features @ self.weights)
        chosen = probabilities[np.arange(len(indices)), indices]
        loss = float(-np.log(np.maximum(chosen, 1e-8)).mean())
        if sample_weights is not None and float(sample_weights.sum()) > 0:
            weighted = -np.log(np.maximum(chosen, 1e-8)) * sample_weights
            loss = float(weighted.sum() / sample_weights.sum())
        accuracy = float((probabilities.argmax(axis=1) == indices).mean())
        return loss, accuracy

    # ------------------------------------------------------------------
    # Deployment
    # ------------------------------------------------------------------

    def predict(self, pov_stack: np.ndarray, previous_action: int) -> int:
        """One discrete action from a causal POV stack and previous action."""

        if self.weights is None:
            raise RuntimeError("policy has not been trained or loaded")
        stack = np.asarray(pov_stack)
        if stack.ndim == 3:
            stack = stack[None, ...]
        features = self.build_features(stack, [int(previous_action)])
        probabilities = self._softmax(features @ self.weights)
        return int(self.classes[int(probabilities.argmax(axis=1)[0])])

    def predict_proba_from_features(self, features: np.ndarray) -> np.ndarray:
        return self._softmax(features @ self.weights)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(
        self,
        path: str,
        dataset_hashes: Optional[Dict[str, str]] = None,
        seed_ranges: Optional[Dict[str, str]] = None,
    ) -> None:
        if self.weights is None:
            raise RuntimeError("cannot save an untrained policy")
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                model_version=np.array(self.model_version),
                feature_size=np.array(self.feature_size),
                frame_stack=np.array(self.frame_stack),
                include_centre_pixels=np.array(self.include_centre_pixels),
                classes=self.classes,
                mean=self.mean,
                std=self.std,
                weights=self.weights,
                student_input_manifest=np.array(self.student_input_manifest),
                dataset_hashes=np.array(
                    [
                        "{}={}".format(key, value)
                        for key, value in (dataset_hashes or {}).items()
                    ]
                ),
                seed_ranges=np.array(
                    [
                        "{}={}".format(key, value)
                        for key, value in (seed_ranges or {}).items()
                    ]
                ),
                selected_epoch=np.array(self.best_epoch),
                decision_threshold=np.array(
                    float(getattr(self, "decision_threshold", 0.5))
                ),
                attack_confirmation_frames=np.array(
                    int(getattr(self, "attack_confirmation_frames", 1))
                ),
            )
        temporary.replace(output)

    @classmethod
    def load(cls, path: str):
        data = np.load(path, allow_pickle=False)
        policy = cls(
            feature_size=int(data["feature_size"]),
            frame_stack=int(data["frame_stack"]),
            include_centre_pixels=(
                bool(data["include_centre_pixels"])
                if "include_centre_pixels" in data.files
                else True
            ),
        )
        policy.classes = data["classes"].astype(np.int64)
        policy.mean = data["mean"].astype(np.float32)
        policy.std = data["std"].astype(np.float32)
        policy.weights = data["weights"].astype(np.float32)
        policy.model_version = str(data["model_version"])
        policy.student_input_manifest = tuple(
            str(item) for item in data["student_input_manifest"].tolist()
        )
        policy.best_epoch = int(data["selected_epoch"])
        policy.decision_threshold = (
            float(data["decision_threshold"])
            if "decision_threshold" in data.files
            else 0.5
        )
        policy.attack_confirmation_frames = (
            int(data["attack_confirmation_frames"])
            if "attack_confirmation_frames" in data.files
            else 1
        )
        extras: Dict[str, Dict[str, str]] = {}
        for key in ("dataset_hashes", "seed_ranges"):
            if key in data.files:
                extras[key] = dict(
                    item.split("=", 1) for item in data[key].tolist()
                )
        policy.dataset_hashes = extras.get("dataset_hashes", {})
        policy.seed_ranges = extras.get("seed_ranges", {})
        return policy


class StudentContactAgent:
    """Deployment wrapper that consumes only a POV-only guarded observation.

    The agent owns its causal frame history and previous-action state. Its
    single input channel is a :class:`StudentObservation` mapping that only
    contains ``pov``; reading any other key raises, which makes privileged
    access fail immediately and auditably instead of silently succeeding.
    """

    def __init__(self, policy: "NaturalContactBCPolicy"):
        if policy.weights is None:
            raise RuntimeError("student policy has not been trained or loaded")
        self.policy = policy
        self.reset_episode()

    def reset_episode(self) -> None:
        self.history = ContactFrameHistory(self.policy.frame_stack)
        self.previous_action = 0

    def observe_pov(self, observation: StudentObservation) -> None:
        """Add one guarded POV frame without requesting a prediction."""

        pov = observation["pov"]
        self.history.push(pov)

    def predict_current(self) -> int:
        """Predict from the already-observed current frame history."""

        stack = self.history.current_stack()
        return int(
            self.policy.predict(stack, self.previous_action)
        )

    def act(self, observation: StudentObservation) -> int:
        self.observe_pov(observation)
        return self.predict_current()

    def observe_transition(self, action: int) -> None:
        self.previous_action = int(action)


class ContactFrameHistory:
    """Episode-local causal frame buffer for contact attempts.

    The buffer keeps the POV frames of the running episode. A contact
    sample's stack is the sliding causal window ending at the current frame,
    which means a fresh attempt starts with up to ``stack_size - 1`` frames
    of causal pre-roll from before the attempt; once the attempt has run
    ``stack_size`` steps the window lies entirely inside the attempt. Stacks
    never cross an episode boundary because ``reset_episode`` clears the
    buffer.
    """

    def __init__(self, stack_size: int = FRAME_STACK):
        if stack_size <= 0:
            raise ValueError("stack size must be positive")
        self.stack_size = int(stack_size)
        self._episode_frames: List[np.ndarray] = []

    def reset_episode(self) -> None:
        self._episode_frames = []

    def push(self, frame: np.ndarray) -> None:
        self._episode_frames.append(np.asarray(frame).copy())

    def current_stack(self) -> np.ndarray:
        """Return the causal stack ending at the most recent pushed frame."""

        if not self._episode_frames:
            raise RuntimeError("no frames pushed")
        end = len(self._episode_frames)
        start = max(0, end - self.stack_size)
        indices = list(range(start, end))
        while len(indices) < self.stack_size:
            indices.insert(0, indices[0])
        return np.stack(
            [self._episode_frames[index] for index in indices], axis=0
        )
