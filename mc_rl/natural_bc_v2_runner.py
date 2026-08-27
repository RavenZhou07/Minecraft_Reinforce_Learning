"""Hybrid rollout runner for the BC v2 visual contact student."""

from typing import Any, Dict, Optional, Tuple

import numpy as np

from mc_rl.natural_bc_runner import RUNNER_MODES, NaturalContactRunner
from mc_rl.natural_contact_bc import StudentContactAgent, student_observation
from mc_rl.natural_contact_bc_v2 import visual_student_eligible


class HybridNaturalContactRunner(NaturalContactRunner):
    """Give the student only pixel-grounded contact decisions.

    Script-owned contact phases always execute the v9.6 action.  In an
    eligible visual phase autonomous mode always executes the student action;
    there is no confidence or disagreement fallback inside that boundary.
    """

    def __init__(
        self,
        policy,
        student: Optional[StudentContactAgent] = None,
        mode: str = "teacher",
        frame_stack: int = 4,
    ):
        super().__init__(policy, student, mode, frame_stack)

    def reset_episode(self) -> None:
        super().reset_episode()
        self.visual_student_steps = 0
        self.scripted_contact_steps = 0
        self.visual_student_predictions = 0

    def _observe_without_prediction(self, observation: Dict[str, Any]) -> None:
        if self.student is None:
            return
        guarded = student_observation(observation["pov"])
        try:
            self.student.observe_pov(guarded)
        except KeyError:
            self.privileged_student_input_accesses += 1
            raise

    def act(self, observation: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        decision_state = getattr(self.policy, "contact_state", "") or ""
        teacher_action = int(self.policy.act(observation))
        resulting_state = getattr(self.policy, "contact_state", "") or ""
        source = self.policy.last_action_source
        self.frame_history.push(np.asarray(observation["pov"]))

        if source == "contact":
            if not self.in_contact:
                self.in_contact = True
                self.attempt_id += 1
            self.contact_steps += 1
            self.teacher_contact_actions.append(teacher_action)
        else:
            self.in_contact = False

        learned_boundary = bool(
            source == "contact"
            and visual_student_eligible(
                decision_state, resulting_state, teacher_action
            )
        )
        student_action = None
        if learned_boundary and self.student is not None:
            student_action = self._student_prediction(observation)
            self.student_predictions.append(int(student_action))
            self.visual_student_predictions += 1
            if self.mode == "autonomous":
                executed = int(student_action)
                self.student_actions_executed += 1
                self.visual_student_steps += 1
                control_owner = "visual_student"
            else:
                executed = teacher_action
                self.teacher_actions_in_contact += 1
                control_owner = "visual_student_shadow"
        else:
            executed = teacher_action
            if source == "contact":
                self.scripted_contact_steps += 1
                if self.student is not None:
                    self._observe_without_prediction(observation)
                if self.mode != "teacher":
                    self.teacher_actions_in_contact += 1
                control_owner = "scripted_teacher"
            else:
                if self.student is not None:
                    self._observe_without_prediction(observation)
                control_owner = "upstream_teacher"

        record = {
            "teacher_action": teacher_action,
            "executed_action": int(executed),
            "student_action": (
                None if student_action is None else int(student_action)
            ),
            "action_source": source,
            "attempt_id": self.attempt_id if source == "contact" else 0,
            "contact_active": source == "contact",
            "contact_state_before": str(
                getattr(decision_state, "value", decision_state)
            ),
            "contact_state_after": str(
                getattr(resulting_state, "value", resulting_state)
            ),
            "learned_boundary": learned_boundary,
            "control_owner": control_owner,
            "privileged_student_input_accesses": (
                self.privileged_student_input_accesses
            ),
        }
        return int(executed), record
