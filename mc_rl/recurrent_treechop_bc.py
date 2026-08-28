"""Minimal legal-observation CNN/GRU actor for Natural Treechop.

The recurrent policy consumes only the allowlisted student observation emitted
by :mod:`mc_rl.learning_observation`, plus the previously executed action.  The
dataset helpers deliberately name every array they read so privileged audit
arrays cannot become actor inputs by accident.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from mc_rl.actions import ACTION_NAMES
from mc_rl.learning_observation import (
    STUDENT_OBSERVATION_SCHEMA_VERSION,
    STUDENT_VECTOR_NAMES,
    LegalObservationAdapter,
)


RECURRENT_MODEL_VERSION = "natural_treechop_recurrent_bc_v1"
ACTION_COUNT = len(ACTION_NAMES)
START_ACTION_TOKEN = ACTION_COUNT
ACTION_TOKEN_COUNT = ACTION_COUNT + 1
RECURRENT_STUDENT_INPUT_MANIFEST = (
    "pov_rgb_64x64_current",
    "legal_player_state_vector_{}".format(len(STUDENT_VECTOR_NAMES)),
    "previous_executed_action_embedding_14_plus_start",
    "episode_local_gru_history",
)
RECURRENT_DATASET_FIELDS = (
    "pov",
    "legal_vector",
    "action",
    "previous_action",
    "episode",
    "episode_seed",
    "episode_step",
    "episode_success",
)


@dataclass(frozen=True)
class RecurrentArchitecture:
    image_height: int = 64
    image_width: int = 64
    vector_size: int = len(STUDENT_VECTOR_NAMES)
    spatial_embedding: int = 96
    scalar_embedding: int = 32
    action_embedding: int = 16
    hidden_size: int = 128
    action_count: int = ACTION_COUNT


@dataclass(frozen=True)
class EpisodeSequence:
    episode_id: int
    seed: int
    pov: np.ndarray
    legal_vector: np.ndarray
    previous_action_token: np.ndarray
    action: np.ndarray

    @property
    def length(self) -> int:
        return int(len(self.action))


@dataclass(frozen=True)
class SequenceBatch:
    pov: Tensor
    legal_vector: Tensor
    previous_action_token: Tensor
    action: Tensor
    mask: Tensor
    seeds: Tuple[int, ...]

    def to(self, device: torch.device) -> "SequenceBatch":
        return SequenceBatch(
            pov=self.pov.to(device),
            legal_vector=self.legal_vector.to(device),
            previous_action_token=self.previous_action_token.to(device),
            action=self.action.to(device),
            mask=self.mask.to(device),
            seeds=self.seeds,
        )


def _array(mapping: Mapping[str, Any], name: str) -> np.ndarray:
    """Read one explicit dataset field without enumerating sibling arrays."""

    if name not in mapping:
        raise KeyError("dataset is missing required field: {}".format(name))
    return np.asarray(mapping[name])


def episode_sequences_from_arrays(
    arrays: Mapping[str, Any],
    include_failure_teacher: bool = False,
    selected_seeds: Optional[Iterable[int]] = None,
) -> List[EpisodeSequence]:
    """Recover ordered, causally aligned episodes from a trajectory dataset."""

    values = {name: _array(arrays, name) for name in RECURRENT_DATASET_FIELDS}
    sample_count = len(values["action"])
    if any(len(value) != sample_count for value in values.values()):
        raise ValueError("trajectory arrays do not have equal sample counts")
    if values["pov"].shape[1:] != (64, 64, 3):
        raise ValueError("expected POV shape (N,64,64,3)")
    if values["legal_vector"].shape[1:] != (len(STUDENT_VECTOR_NAMES),):
        raise ValueError("legal vector width does not match the declared schema")

    allowed_seeds = None if selected_seeds is None else set(int(v) for v in selected_seeds)
    result: List[EpisodeSequence] = []
    episode_ids = values["episode"].astype(np.int64)
    ordered_ids: List[int] = []
    for value in episode_ids.tolist():
        if not ordered_ids or ordered_ids[-1] != int(value):
            if int(value) in ordered_ids:
                raise ValueError("episode samples are not contiguous")
            ordered_ids.append(int(value))

    for episode_id in ordered_ids:
        indices = np.flatnonzero(episode_ids == episode_id)
        seed_values = np.unique(values["episode_seed"][indices].astype(np.int64))
        success_values = np.unique(values["episode_success"][indices].astype(np.int64))
        if len(seed_values) != 1 or len(success_values) != 1:
            raise ValueError("episode metadata changes within an episode")
        seed = int(seed_values[0])
        if allowed_seeds is not None and seed not in allowed_seeds:
            continue
        if not include_failure_teacher and not bool(success_values[0]):
            continue

        steps = values["episode_step"][indices].astype(np.int64)
        if not np.array_equal(steps, np.arange(len(indices), dtype=np.int64)):
            raise ValueError("episode {} has non-contiguous timesteps".format(episode_id))
        actions = values["action"][indices].astype(np.int64)
        stored_previous = values["previous_action"][indices].astype(np.int64)
        if len(actions) and (
            (actions < 0).any()
            or (actions >= ACTION_COUNT).any()
            or (stored_previous < 0).any()
            or (stored_previous >= ACTION_COUNT).any()
        ):
            raise ValueError("episode contains an invalid action id")
        if len(actions) > 1 and not np.array_equal(stored_previous[1:], actions[:-1]):
            raise ValueError("action/previous_action alignment failed")

        previous_tokens = np.empty_like(actions)
        if len(actions):
            previous_tokens[0] = START_ACTION_TOKEN
            previous_tokens[1:] = actions[:-1]
        result.append(
            EpisodeSequence(
                episode_id=episode_id,
                seed=seed,
                pov=values["pov"][indices].astype(np.uint8, copy=True),
                legal_vector=values["legal_vector"][indices].astype(np.float32, copy=True),
                previous_action_token=previous_tokens,
                action=actions.copy(),
            )
        )
    if not result:
        raise ValueError("no eligible episode sequences")
    return result


def load_episode_sequences(
    path: Path,
    include_failure_teacher: bool = False,
    selected_seeds: Optional[Iterable[int]] = None,
) -> List[EpisodeSequence]:
    with np.load(Path(path), allow_pickle=False) as arrays:
        return episode_sequences_from_arrays(
            arrays,
            include_failure_teacher=include_failure_teacher,
            selected_seeds=selected_seeds,
        )


def collate_episode_sequences(episodes: Sequence[EpisodeSequence]) -> SequenceBatch:
    """Pad independent complete episodes; padded timesteps are mask=False."""

    if not episodes:
        raise ValueError("cannot collate an empty episode batch")
    batch_size = len(episodes)
    max_length = max(episode.length for episode in episodes)
    pov = torch.zeros((batch_size, max_length, 64, 64, 3), dtype=torch.uint8)
    vector = torch.zeros(
        (batch_size, max_length, len(STUDENT_VECTOR_NAMES)), dtype=torch.float32
    )
    previous = torch.full(
        (batch_size, max_length), START_ACTION_TOKEN, dtype=torch.long
    )
    actions = torch.zeros((batch_size, max_length), dtype=torch.long)
    mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
    for row, episode in enumerate(episodes):
        length = episode.length
        pov[row, :length] = torch.from_numpy(episode.pov)
        vector[row, :length] = torch.from_numpy(episode.legal_vector)
        previous[row, :length] = torch.from_numpy(episode.previous_action_token)
        actions[row, :length] = torch.from_numpy(episode.action)
        mask[row, :length] = True
    return SequenceBatch(
        pov=pov,
        legal_vector=vector,
        previous_action_token=previous,
        action=actions,
        mask=mask,
        seeds=tuple(episode.seed for episode in episodes),
    )


def episode_batches(
    episodes: Sequence[EpisodeSequence],
    batch_size: int,
    shuffle: bool,
    rng: np.random.RandomState,
) -> Iterable[SequenceBatch]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    indices = np.arange(len(episodes), dtype=np.int64)
    if shuffle:
        rng.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        yield collate_episode_sequences([episodes[index] for index in indices[start:start + batch_size]])


class RecurrentTreechopActor(nn.Module):
    """Small CNN + scalar MLP + previous-action embedding + one-layer GRU."""

    def __init__(self, architecture: RecurrentArchitecture = RecurrentArchitecture()):
        super().__init__()
        self.architecture = architecture
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=5, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, architecture.spatial_embedding),
            nn.ReLU(),
        )
        self.scalar_encoder = nn.Sequential(
            nn.Linear(architecture.vector_size, 32),
            nn.ReLU(),
            nn.Linear(32, architecture.scalar_embedding),
            nn.ReLU(),
        )
        self.previous_action_embedding = nn.Embedding(
            ACTION_TOKEN_COUNT, architecture.action_embedding
        )
        recurrent_input = (
            architecture.spatial_embedding
            + architecture.scalar_embedding
            + architecture.action_embedding
        )
        self.gru = nn.GRU(
            recurrent_input,
            architecture.hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.action_head = nn.Linear(architecture.hidden_size, architecture.action_count)

    def forward(
        self,
        pov: Tensor,
        legal_vector: Tensor,
        previous_action_token: Tensor,
        hidden: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        if pov.ndim != 5 or pov.shape[-3:] != (
            self.architecture.image_height,
            self.architecture.image_width,
            3,
        ):
            raise ValueError("pov must have shape (B,T,64,64,3)")
        batch_size, timesteps = pov.shape[:2]
        flat_pov = pov.reshape(batch_size * timesteps, 64, 64, 3)
        flat_pov = flat_pov.permute(0, 3, 1, 2).float().div(255.0)
        spatial = self.spatial_encoder(flat_pov).reshape(batch_size, timesteps, -1)
        scalars = self.scalar_encoder(
            legal_vector.reshape(batch_size * timesteps, -1)
        ).reshape(batch_size, timesteps, -1)
        action_embedding = self.previous_action_embedding(previous_action_token)
        encoded = torch.cat((spatial, scalars, action_embedding), dim=-1)
        recurrent, next_hidden = self.gru(encoded, hidden)
        return self.action_head(recurrent), next_hidden


def masked_cross_entropy(
    logits: Tensor,
    actions: Tensor,
    mask: Tensor,
    class_weights: Optional[Tensor] = None,
) -> Tensor:
    if logits.shape[:2] != actions.shape or actions.shape != mask.shape:
        raise ValueError("logits, actions and mask shapes are not aligned")
    selected_logits = logits[mask]
    selected_actions = actions[mask]
    if not len(selected_actions):
        raise ValueError("batch contains no valid timesteps")
    return F.cross_entropy(selected_logits, selected_actions, weight=class_weights)


def class_weights_for_episodes(
    episodes: Sequence[EpisodeSequence], power: float
) -> Tensor:
    if power < 0:
        raise ValueError("class weight power must be non-negative")
    counts = np.bincount(
        np.concatenate([episode.action for episode in episodes]), minlength=ACTION_COUNT
    ).astype(np.float64)
    weights = np.zeros(ACTION_COUNT, dtype=np.float32)
    present = counts > 0
    weights[present] = np.power(counts[present], -float(power)).astype(np.float32)
    if present.any():
        weights[present] /= float(weights[present].mean())
    return torch.from_numpy(weights)


class RecurrentTreechopPolicy:
    """Checkpointable recurrent policy with a one-step rollout interface."""

    def __init__(
        self,
        architecture: RecurrentArchitecture = RecurrentArchitecture(),
        device: str = "cpu",
    ):
        self.architecture = architecture
        self.device = torch.device(device)
        self.model = RecurrentTreechopActor(architecture).to(self.device)
        self.dataset_hashes: Dict[str, str] = {}
        self.seed_manifest = ""
        self.student_input_manifest = RECURRENT_STUDENT_INPUT_MANIFEST

    def predict_step(
        self,
        pov: np.ndarray,
        legal_vector: np.ndarray,
        previous_action_token: int,
        hidden: Optional[Tensor],
    ) -> Tuple[int, np.ndarray, Tensor]:
        self.model.eval()
        with torch.no_grad():
            logits, next_hidden = self.model(
                torch.from_numpy(np.asarray(pov, dtype=np.uint8))[None, None].to(self.device),
                torch.from_numpy(np.asarray(legal_vector, dtype=np.float32))[None, None].to(self.device),
                torch.tensor([[int(previous_action_token)]], dtype=torch.long, device=self.device),
                None if hidden is None else hidden.to(self.device),
            )
            probabilities = torch.softmax(logits[0, 0], dim=-1)
            action = int(torch.argmax(probabilities).item())
        return action, probabilities.cpu().numpy(), next_hidden.detach()

    def save(
        self,
        path: str,
        dataset_hashes: Mapping[str, str],
        seed_manifest: str,
        training_metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_version": RECURRENT_MODEL_VERSION,
            "observation_schema": STUDENT_OBSERVATION_SCHEMA_VERSION,
            "architecture": asdict(self.architecture),
            "student_input_manifest": list(self.student_input_manifest),
            "state_dict": self.model.state_dict(),
            "dataset_hashes": dict(dataset_hashes),
            "seed_manifest": str(seed_manifest),
            "training_metadata": dict(training_metadata or {}),
        }
        temporary = output.with_suffix(output.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(output)
        self.dataset_hashes = dict(dataset_hashes)
        self.seed_manifest = str(seed_manifest)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "RecurrentTreechopPolicy":
        payload = torch.load(path, map_location=device, weights_only=False)
        if payload.get("model_version") != RECURRENT_MODEL_VERSION:
            raise ValueError("unsupported recurrent checkpoint version")
        if payload.get("observation_schema") != STUDENT_OBSERVATION_SCHEMA_VERSION:
            raise ValueError("checkpoint observation schema mismatch")
        manifest = tuple(payload.get("student_input_manifest", ()))
        if manifest != RECURRENT_STUDENT_INPUT_MANIFEST:
            raise ValueError("checkpoint student input manifest mismatch")
        architecture = RecurrentArchitecture(**payload["architecture"])
        policy = cls(architecture=architecture, device=device)
        policy.model.load_state_dict(payload["state_dict"])
        policy.model.eval()
        policy.dataset_hashes = dict(payload.get("dataset_hashes", {}))
        policy.seed_manifest = str(payload.get("seed_manifest", ""))
        return policy


class RecurrentTreechopStudentAgent:
    """Episode-local recurrent rollout actor with explicit hidden reset."""

    def __init__(self, policy: RecurrentTreechopPolicy, max_episode_steps: int):
        self.policy = policy
        self.max_episode_steps = int(max_episode_steps)
        self.reset_episode()

    def reset_episode(self) -> None:
        self.observation_adapter = LegalObservationAdapter(self.max_episode_steps)
        self.hidden: Optional[Tensor] = None
        self.previous_action_token = START_ACTION_TOKEN
        self.started = False
        self.last_pov: Optional[np.ndarray] = None

    def act(self, raw_observation: Dict[str, Any], episode_step: int) -> Tuple[int, np.ndarray]:
        legal = (
            self.observation_adapter.reset(raw_observation)
            if not self.started
            else self.observation_adapter.adapt(raw_observation, episode_step)
        )
        self.started = True
        self.last_pov = legal.pov.copy()
        action, _, self.hidden = self.policy.predict_step(
            legal.pov,
            legal.vector,
            self.previous_action_token,
            self.hidden,
        )
        return int(action), legal.vector.copy()

    def observe_transition(self, executed_action: int) -> None:
        action = int(executed_action)
        if not 0 <= action < ACTION_COUNT:
            raise ValueError("executed action is outside the 14-action space")
        self.previous_action_token = action
