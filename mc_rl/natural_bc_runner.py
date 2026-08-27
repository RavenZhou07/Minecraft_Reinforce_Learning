"""Shared rollout runner for natural Treechop contact behaviour cloning.

The runner wraps a frozen :class:`CandidateSearchPolicy` (v9.6 teacher) and
an optional visual student. It owns exactly one decision: when the teacher's
contact controller holds action ownership, the executed action may come from
the student instead. The upstream scan, candidate map, world route, terrain
recovery, handoff judgement, replan, relocalization, and contact-owner
invariants all remain the teacher's, untouched.

Three execution modes are supported and must never be conflated:

- ``teacher``: the teacher action is executed (data collection);
- ``shadow``: the teacher action is executed while the student prediction is
  only recorded (agreement evaluation, no environment influence);
- ``autonomous``: the student action is executed whenever the contact owner
  is active, and no teacher fallback exists inside the contact phase.

The student's only input channel is a guarded POV-only observation view; any
access to telemetry, raycast, oracle, or grid keys raises immediately and is
counted as a privileged student input access.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np

from mc_rl.natural_contact_bc import (
    ContactFrameHistory,
    StudentContactAgent,
    student_observation,
)


RUNNER_MODES = ("teacher", "shadow", "autonomous")


class NaturalContactRunner:
    """Drive one episode with a strict teacher/student action boundary."""

    def __init__(
        self,
        policy,
        student: Optional[StudentContactAgent] = None,
        mode: str = "teacher",
        frame_stack: int = 4,
    ):
        if mode not in RUNNER_MODES:
            raise ValueError("unknown runner mode: {}".format(mode))
        if mode != "teacher" and student is None:
            raise ValueError("{} mode requires a student agent".format(mode))
        if mode == "teacher" and student is not None:
            raise ValueError("teacher mode must not receive a student agent")
        self.policy = policy
        self.student = student
        self.mode = mode
        self.privileged_student_input_accesses = 0
        self.frame_stack = int(frame_stack)
        self.reset_episode()

    def reset_episode(self) -> None:
        self.frame_history = ContactFrameHistory(self.frame_stack)
        self.previous_action = 0
        self.in_contact = False
        self.attempt_id = 0
        self.contact_steps = 0
        self.teacher_actions_in_contact = 0
        self.student_actions_executed = 0
        self.student_predictions: list = []
        self.teacher_contact_actions: list = []

    # ------------------------------------------------------------------

    def _student_prediction(self, observation: Dict[str, Any]) -> int:
        guarded = student_observation(observation["pov"])
        try:
            return int(self.student.act(guarded))
        except KeyError:
            # The guarded view raised: a hard isolation violation that must
            # surface in the audit counters before failing loudly.
            self.privileged_student_input_accesses += 1
            raise

    def act(self, observation: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        """Return the executed action plus a per-step audit record."""

        contact_state_before = getattr(self.policy, "contact_state", "") or ""
        teacher_action = self.policy.act(observation)
        contact_state_after = getattr(self.policy, "contact_state", "") or ""
        source = self.policy.last_action_source
        self.frame_history.push(np.asarray(observation["pov"]))
        if source == "contact":
            if not self.in_contact:
                self.in_contact = True
                self.attempt_id += 1
            self.contact_steps += 1
            self.teacher_contact_actions.append(int(teacher_action))
        else:
            self.in_contact = False
        student_action: Optional[int] = None
        if source == "contact" and self.student is not None:
            student_action = self._student_prediction(observation)
            self.student_predictions.append(int(student_action))
            if self.mode == "autonomous":
                executed = int(student_action)
                self.student_actions_executed += 1
            else:
                executed = int(teacher_action)
                self.teacher_actions_in_contact += 1
        else:
            executed = int(teacher_action)
        record = {
            "teacher_action": int(teacher_action),
            "executed_action": int(executed),
            "student_action": (
                None if student_action is None else int(student_action)
            ),
            "action_source": source,
            "attempt_id": self.attempt_id if source == "contact" else 0,
            "contact_active": source == "contact",
            "contact_state_before": contact_state_before,
            "contact_state_after": contact_state_after,
            "privileged_student_input_accesses": (
                self.privileged_student_input_accesses
            ),
        }
        return executed, record

    def observe_transition(self, action: int) -> None:
        """Advance local causal state after the environment consumed the action."""

        self.previous_action = int(action)
        if self.student is not None:
            self.student.observe_transition(int(action))
