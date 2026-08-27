"""Hybrid visual behaviour cloning for natural Treechop contact control.

BC v2 deliberately does not imitate contact actions whose identity depends on
exact coordinates, a scripted scan queue, remembered target bearings, or drop
waypoints.  Those phases stay with the frozen v9.6 controller.  The student is
eligible only while a decision both starts and ends inside the pixel-grounded
centering, pitch, and attack states.

Teacher contact state is used by the hybrid router and by offline dataset
selection.  It is never added to the model feature vector: the learned policy
still consumes only a causal POV stack and the previous environment action.
"""

from typing import Sequence

import numpy as np

from mc_rl.natural_contact_bc import NaturalContactBCPolicy


MODEL_VERSION = "natural_treechop_contact_bc_v2_hybrid"

# These states base their local correction/continuation decision on the trunk
# visible at the crosshair.  All other contact states remain scripted.
LEARNABLE_CONTACT_STATES = frozenset(
    ("CENTER_TRUNK", "ADJUST_PITCH", "ATTACK_TRUNK")
)

SCRIPTED_CONTACT_STATES = frozenset(
    (
        "APPROACH_REGION",
        "COORDINATE_AIM",
        "COORDINATE_RECOVER",
        "POST_RECOVERY_VERIFY",
        "COORDINATE_REPLAN",
        "EXACT_LOG_RESCAN",
        "FIND_TRUNK",
        "CLEAR_OCCLUSION",
        "BLOCK_DISAPPEARED",
        "DROP_RECOVERY",
        "REACQUIRE_SAME_TRUNK",
        "COLLECT_DROP",
        "VERIFY_PROGRESS",
        "BACKOFF",
        "ORBIT_REACQUIRE",
        "SUCCESS",
        "REPLAN",
    )
)

# Possible actions from the three learned visual states.  Coarse yaw/pitch,
# jump, backoff, and forward-attack are owned by scripted phases.
V2_ACTION_CLASSES = np.asarray((0, 1, 7, 10, 11, 12, 13), dtype=np.int64)
V2_REQUIRED_DIRECTIONAL_ACTIONS = frozenset((7, 10, 11, 12, 13))


def normalize_contact_state(state) -> str:
    """Return a stable string for Enum, string, None, and audit values."""

    if state is None:
        return ""
    value = getattr(state, "value", state)
    return str(value)


def visual_student_eligible(
    decision_state,
    resulting_state,
    teacher_action: int,
) -> bool:
    """Whether one contact decision belongs to the learned v2 boundary.

    Requiring both sides of the transition to be visual prevents a transition
    such as ATTACK_TRUNK -> DROP_RECOVERY from leaking a coordinate/drop action
    into student control.  Unsupported actions conservatively stay scripted.
    """

    before = normalize_contact_state(decision_state)
    after = normalize_contact_state(resulting_state)
    return bool(
        before in LEARNABLE_CONTACT_STATES
        and after in LEARNABLE_CONTACT_STATES
        and int(teacher_action) in set(V2_ACTION_CLASSES.tolist())
    )


def hybrid_learning_mask(
    decision_states: Sequence,
    resulting_states: Sequence,
    actions: Sequence[int],
) -> np.ndarray:
    """Vector mask selecting exactly the samples the v2 student may own."""

    before = np.asarray(decision_states)
    after = np.asarray(resulting_states)
    labels = np.asarray(actions, dtype=np.int64)
    if not (len(before) == len(after) == len(labels)):
        raise ValueError("state/action audit arrays must have equal length")
    state_mask = np.isin(before.astype(str), tuple(LEARNABLE_CONTACT_STATES))
    state_mask &= np.isin(after.astype(str), tuple(LEARNABLE_CONTACT_STATES))
    return state_mask & np.isin(labels, V2_ACTION_CLASSES)


class NaturalContactBCV2Policy(NaturalContactBCPolicy):
    """Reduced-action visual policy used only behind the hybrid router."""

    def __init__(
        self,
        feature_size: int = 10,
        frame_stack: int = 4,
        include_centre_pixels: bool = False,
    ):
        super().__init__(
            feature_size=feature_size,
            frame_stack=frame_stack,
            include_centre_pixels=include_centre_pixels,
        )
        self.classes = V2_ACTION_CLASSES.copy()
        self.model_version = MODEL_VERSION
