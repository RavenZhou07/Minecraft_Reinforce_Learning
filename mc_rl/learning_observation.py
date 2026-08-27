"""Strict legal-observation adapter for learned Natural Treechop policies.

The environment may expose privileged siblings so a teacher or evaluator can
label the same transition.  This adapter reads an explicit allowlist and emits
an object that contains no reference to the raw mapping, making accidental
raycast/world-state access impossible downstream.
"""

from dataclasses import dataclass
from math import cos, radians, sin
from typing import Any, Dict, Optional, Tuple

import numpy as np

from mc_rl.navigation import wrap_degrees
from mc_rl.wrappers import inventory_log_count


STUDENT_OBSERVATION_SCHEMA_VERSION = "natural_treechop_student_obs_v1"
LEGAL_RAW_KEYS = ("pov", "telemetry", "inventory")
TRAIN_ONLY_PRIVILEGED_KEYS = (
    "raycast",
    "nearest_tree_xyz",
    "log_grid",
    "target_coordinates",
    "teacher_phase",
    "teacher_candidate_list",
    "reachability",
)
LEGAL_TELEMETRY_FIELDS = (
    "x",
    "y",
    "z",
    "yaw",
    "pitch",
    "biome_id",
    "biome_temperature",
    "biome_rainfall",
)
STUDENT_VECTOR_NAMES = (
    "origin_relative_x_div16",
    "origin_relative_y_div8",
    "origin_relative_z_div16",
    "step_delta_x_div2",
    "step_delta_y_div2",
    "step_delta_z_div2",
    "yaw_sin",
    "yaw_cos",
    "pitch_div90",
    "step_yaw_delta_div45",
    "step_pitch_delta_div45",
    "biome_id_div255",
    "biome_temperature",
    "biome_rainfall",
    "inventory_log_delta_div4",
    "episode_step_fraction",
)


@dataclass(frozen=True)
class LegalStudentObservation:
    """A copied POV plus fixed-width legal player-state features."""

    pov: np.ndarray
    vector: np.ndarray
    inventory_log_count: int
    episode_step: int


def _telemetry_tuple(observation: Dict[str, Any]) -> Tuple[float, ...]:
    telemetry = observation["telemetry"]
    return tuple(float(telemetry[name]) for name in LEGAL_TELEMETRY_FIELDS)


class LegalObservationAdapter:
    """Convert raw Minecraft observations into the declared student schema."""

    def __init__(self, max_episode_steps: int):
        if max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive")
        self.max_episode_steps = int(max_episode_steps)
        self._origin: Optional[Tuple[float, ...]] = None
        self._previous: Optional[Tuple[float, ...]] = None
        self._initial_log_count = 0

    def reset(self, observation: Dict[str, Any]) -> LegalStudentObservation:
        pose = _telemetry_tuple(observation)
        self._origin = pose
        self._previous = pose
        count = inventory_log_count(observation)
        self._initial_log_count = int(count or 0)
        return self._adapt(observation, episode_step=0, pose=pose)

    def adapt(
        self, observation: Dict[str, Any], episode_step: int
    ) -> LegalStudentObservation:
        if self._origin is None or self._previous is None:
            if int(episode_step) != 0:
                raise RuntimeError("adapter must be reset at episode start")
            return self.reset(observation)
        pose = _telemetry_tuple(observation)
        result = self._adapt(observation, int(episode_step), pose)
        self._previous = pose
        return result

    def _adapt(
        self,
        observation: Dict[str, Any],
        episode_step: int,
        pose: Tuple[float, ...],
    ) -> LegalStudentObservation:
        if self._origin is None or self._previous is None:
            raise RuntimeError("adapter has no episode origin")
        x, y, z, yaw, pitch, biome_id, temperature, rainfall = pose
        ox, oy, oz = self._origin[:3]
        px, py, pz, pyaw, ppitch = self._previous[:5]
        count = inventory_log_count(observation)
        log_count = int(count or 0)
        vector = np.asarray(
            [
                (x - ox) / 16.0,
                (y - oy) / 8.0,
                (z - oz) / 16.0,
                (x - px) / 2.0,
                (y - py) / 2.0,
                (z - pz) / 2.0,
                sin(radians(yaw)),
                cos(radians(yaw)),
                pitch / 90.0,
                wrap_degrees(yaw - pyaw) / 45.0,
                (pitch - ppitch) / 45.0,
                biome_id / 255.0,
                temperature,
                rainfall,
                (log_count - self._initial_log_count) / 4.0,
                min(max(float(episode_step) / self.max_episode_steps, 0.0), 1.0),
            ],
            dtype=np.float32,
        )
        if vector.shape != (len(STUDENT_VECTOR_NAMES),):
            raise AssertionError("student vector manifest is out of sync")
        return LegalStudentObservation(
            pov=np.asarray(observation["pov"], dtype=np.uint8).copy(),
            vector=vector,
            inventory_log_count=log_count,
            episode_step=int(episode_step),
        )


def student_input_manifest(frame_stack: int, action_history: int) -> Tuple[str, ...]:
    return (
        "pov_frame_stack_{}".format(int(frame_stack)),
        "legal_player_state_vector_{}".format(len(STUDENT_VECTOR_NAMES)),
        "causal_action_history_{}_one_hot_14".format(int(action_history)),
    )
