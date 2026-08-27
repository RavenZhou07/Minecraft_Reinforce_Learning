"""Binary visual attack-permission model for natural Treechop BC v2a."""

from typing import Sequence

import numpy as np

from mc_rl.natural_contact_bc import NaturalContactBCPolicy


MODEL_VERSION = "natural_treechop_attack_gate_bc_v2a"
HOLD = 0
ATTACK = 1
GATE_CLASSES = np.asarray((HOLD, ATTACK), dtype=np.int64)

# The gate is relevant while the teacher is approaching an exact log or is
# already in local visual alignment/attack. Recovery and scripted scan phases
# never receive a gate decision.
GATE_CONTACT_STATES = frozenset(
    ("COORDINATE_AIM", "CENTER_TRUNK", "ADJUST_PITCH", "ATTACK_TRUNK")
)


def attack_gate_labels(actions: Sequence[int]) -> np.ndarray:
    """Map environment action 7 to ATTACK and every other action to HOLD."""

    actions_array = np.asarray(actions, dtype=np.int64)
    return (actions_array == 7).astype(np.int64)


def attack_gate_sample_mask(
    contact_states: Sequence, episode_success: Sequence[int]
) -> np.ndarray:
    """Successful, visually relevant contact samples used by BC v2a."""

    states = np.asarray(contact_states).astype(str)
    success = np.asarray(episode_success, dtype=np.int64).astype(bool)
    if len(states) != len(success):
        raise ValueError("contact state and success arrays must align")
    return success & np.isin(states, tuple(GATE_CONTACT_STATES))


class NaturalAttackGatePolicy(NaturalContactBCPolicy):
    """Binary HOLD/ATTACK classifier over the existing POV-only features."""

    def __init__(
        self,
        feature_size: int = 10,
        frame_stack: int = 4,
        include_centre_pixels: bool = False,
        decision_threshold: float = 0.5,
    ):
        super().__init__(
            feature_size=feature_size,
            frame_stack=frame_stack,
            include_centre_pixels=include_centre_pixels,
        )
        self.classes = GATE_CLASSES.copy()
        self.model_version = MODEL_VERSION
        self.decision_threshold = float(decision_threshold)
        if not 0.0 < self.decision_threshold < 1.0:
            raise ValueError("decision threshold must lie strictly inside (0, 1)")

    def attack_probability(
        self, pov_stack: np.ndarray, previous_action: int
    ) -> float:
        if self.weights is None:
            raise RuntimeError("policy has not been trained or loaded")
        stack = np.asarray(pov_stack)
        if stack.ndim == 3:
            stack = stack[None, ...]
        features = self.build_features(stack, [int(previous_action)])
        probabilities = self.predict_proba_from_features(features)
        attack_index = int(np.flatnonzero(self.classes == ATTACK)[0])
        return float(probabilities[0, attack_index])

    def predict(self, pov_stack: np.ndarray, previous_action: int) -> int:
        probability = self.attack_probability(pov_stack, previous_action)
        return ATTACK if probability >= self.decision_threshold else HOLD
