"""State-safe teacher/visual-gate runner for BC v2a/v2b."""

from typing import Any, Dict, Optional, Tuple

import numpy as np

from mc_rl.natural_bc_runner import NaturalContactRunner
from mc_rl.natural_contact_bc import StudentContactAgent, student_observation
from mc_rl.natural_attack_gate_bc import ATTACK, GATE_CONTACT_STATES


class NaturalAttackGateRunner(NaturalContactRunner):
    """Inject a visual permission before the teacher mutates attack state.

    In shadow mode the student prediction is observational and the frozen
    teacher is untouched. In autonomous mode HOLD prevents attack entry or
    continuation inside the contact controller itself, so rejected attacks do
    not increment burst counters or incorrectly trigger drop recovery.
    """

    def reset_episode(self) -> None:
        super().reset_episode()
        self.gate_predictions = 0
        self.gate_permissions_applied = 0
        self.gate_attack_predictions = 0
        self.gate_hold_predictions = 0
        self.gate_confirmed_attack_predictions = 0
        self._gate_attack_streak = 0
        self._gate_streak_state = ""

    def __init__(
        self,
        policy,
        student=None,
        mode: str = "teacher",
        frame_stack: int = 4,
        attack_confirmation_frames: Optional[int] = None,
    ):
        if attack_confirmation_frames is None and student is not None:
            attack_confirmation_frames = int(
                getattr(student.policy, "attack_confirmation_frames", 1)
            )
        self.attack_confirmation_frames = int(attack_confirmation_frames or 1)
        if self.attack_confirmation_frames < 1:
            raise ValueError("attack confirmation frames must be at least one")
        super().__init__(policy, student, mode, frame_stack)

    def _observe_without_prediction(self, observation: Dict[str, Any]) -> None:
        if self.student is None:
            return
        guarded = student_observation(observation["pov"])
        try:
            self.student.observe_pov(guarded)
        except KeyError:
            self.privileged_student_input_accesses += 1
            raise

    def _attack_probability(self, gate_decision: int) -> float:
        """Return the student's POV-only probability after it observed this frame."""

        policy = getattr(self.student, "policy", None)
        probability = getattr(policy, "attack_probability", None)
        history = getattr(self.student, "history", None)
        if probability is None or history is None:
            return float(gate_decision == ATTACK)
        return float(
            probability(
                history.current_stack(),
                int(getattr(self.student, "previous_action", 0)),
            )
        )

    def _confirmed_decision(self, decision_state: str, gate_decision: int) -> int:
        """Require a causal run of ATTACK predictions; HOLD revokes immediately."""

        if decision_state != self._gate_streak_state:
            self._gate_attack_streak = 0
            self._gate_streak_state = decision_state
        if gate_decision == ATTACK:
            self._gate_attack_streak += 1
        else:
            self._gate_attack_streak = 0
        return (
            ATTACK
            if self._gate_attack_streak >= self.attack_confirmation_frames
            else 0
        )

    def act(self, observation: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        decision_state = getattr(self.policy, "contact_state", "") or ""
        gate_eligible = decision_state in GATE_CONTACT_STATES
        gate_decision: Optional[int] = None
        confirmed_gate_decision: Optional[int] = None
        gate_probability: Optional[float] = None
        if self.student is not None:
            if gate_eligible:
                gate_decision = self._student_prediction(observation)
                gate_probability = self._attack_probability(gate_decision)
                confirmed_gate_decision = self._confirmed_decision(
                    decision_state, gate_decision
                )
                self.gate_predictions += 1
                if gate_decision == ATTACK:
                    self.gate_attack_predictions += 1
                else:
                    self.gate_hold_predictions += 1
                if confirmed_gate_decision == ATTACK:
                    self.gate_confirmed_attack_predictions += 1
            else:
                self._gate_attack_streak = 0
                self._gate_streak_state = ""
                self._observe_without_prediction(observation)

        permission_applied = bool(
            self.mode == "autonomous"
            and gate_eligible
            and confirmed_gate_decision is not None
        )
        setter = getattr(self.policy, "set_external_attack_permission", None)
        if setter is not None:
            setter(
                confirmed_gate_decision == ATTACK if permission_applied else None
            )
        try:
            teacher_action = int(self.policy.act(observation))
        finally:
            if setter is not None:
                setter(None)
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
        if permission_applied:
            self.gate_permissions_applied += 1

        record = {
            "teacher_action": teacher_action,
            "executed_action": teacher_action,
            "student_action": gate_decision,
            "gate_decision": gate_decision,
            "confirmed_gate_decision": confirmed_gate_decision,
            "gate_probability": gate_probability,
            "gate_eligible": gate_eligible,
            "gate_permission_applied": permission_applied,
            "attack_confirmation_frames": self.attack_confirmation_frames,
            "action_source": source,
            "attempt_id": self.attempt_id if source == "contact" else 0,
            "contact_active": source == "contact",
            "contact_state_before": decision_state,
            "contact_state_after": resulting_state,
            "privileged_student_input_accesses": (
                self.privileged_student_input_accesses
            ),
        }
        return teacher_action, record
