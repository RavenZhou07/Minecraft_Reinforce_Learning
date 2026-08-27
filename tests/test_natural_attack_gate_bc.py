from typing import Any, Dict

import numpy as np

from mc_rl.natural_attack_gate_bc import (
    ATTACK,
    HOLD,
    NaturalAttackGatePolicy,
    attack_gate_labels,
    attack_gate_sample_mask,
)
from mc_rl.natural_attack_gate_runner import NaturalAttackGateRunner
from mc_rl.natural_contact_bc import StudentContactAgent, StudentObservation
from mc_rl.resource_adapters import TreeResourceAdapter, TrunkView
from mc_rl.telemetry import AgentTelemetry, RaycastHit
from mc_rl.trunk_contact import (
    TrunkContactConfig,
    TrunkContactController,
    TrunkContactState,
)
from scripts.train_natural_treechop_attack_gate_v2a import (
    binary_metrics,
    hard_negative_training_indices,
    select_conservative_threshold,
)


RNG = np.random.RandomState(43)


def frame():
    return RNG.randint(0, 255, size=(16, 16, 3), dtype=np.uint8)


class FixedTrunkAdapter(TreeResourceAdapter):
    def __init__(self):
        super().__init__(interaction_size=None, reward_is_success=True)
        self.view = TrunkView(
            present=True,
            center_x=0.5,
            center_y=0.5,
            bottom_y=0.8,
            width_px=10.0,
            height_px=30.0,
            area_px=220.0,
            crosshair_trunk_fraction=0.7,
            horizontal_yaw=0.0,
            vertical_offset_deg=0.0,
            clipped_vertical=False,
            material="oak",
        )

    def trunk_view(self, pov):
        return self.view

    def trunk_views(self, pov):
        return [self.view]


def attack_controller():
    controller = TrunkContactController(
        FixedTrunkAdapter(),
        TrunkContactConfig(
            require_raycast_attack_confirmation=True,
            attack_action=7,
            noop_action=0,
            external_attack_gate_recenter_rejections=3,
        ),
    )
    controller.engaged = True
    controller.state = TrunkContactState.ATTACK_TRUNK
    return controller


def in_range_log():
    return RaycastHit(True, True, False, True, 2.0, 0.0, 65.0, 1.0)


def missing_raycast_target():
    return RaycastHit(False, False, False, False, 0.0, 0.0, 0.0, 0.0)


def telemetry():
    return AgentTelemetry(x=0.0, y=64.0, z=0.0, yaw=0.0, pitch=0.0)


def tiny_gate_policy() -> NaturalAttackGatePolicy:
    policy = NaturalAttackGatePolicy(feature_size=4, frame_stack=2)
    pov = np.repeat(frame()[None, None, ...], 8, axis=0)
    pov = np.repeat(pov, 2, axis=1)
    features = policy.build_features(pov, [0] * len(pov), fit_normalization=True)
    policy.weights = np.zeros((features.shape[1], 2), dtype=np.float32)
    policy.best_epoch = 0
    return policy


class FixedGateAgent(StudentContactAgent):
    def __init__(self, policy, decision):
        super().__init__(policy)
        self.decision = int(decision)
        self.calls = 0

    def act(self, observation: StudentObservation) -> int:
        self.observe_pov(observation)
        self.calls += 1
        return self.decision


class PermissionAwareTeacher:
    def __init__(self, state="ATTACK_TRUNK"):
        self.contact_state = state
        self.last_action_source = "contact"
        self.permission = None
        self.permissions = []

    def set_external_attack_permission(self, permission):
        self.permission = permission
        self.permissions.append(permission)

    def act(self, observation: Dict[str, Any]):
        return 7 if self.permission is not False else 0


def test_attack_gate_labels_and_state_filter_are_explicit():
    assert attack_gate_labels([0, 1, 7, 8, 13]).tolist() == [0, 0, 1, 0, 0]
    mask = attack_gate_sample_mask(
        ["COORDINATE_AIM", "ATTACK_TRUNK", "DROP_RECOVERY", "CENTER_TRUNK"],
        [1, 0, 1, 1],
    )
    assert mask.tolist() == [True, False, False, True]


def test_default_none_permission_preserves_frozen_attack_action():
    controller = attack_controller()
    action = controller.act(frame(), telemetry(), raycast=in_range_log())
    assert action == 7
    assert controller._burst_steps == 1
    assert controller.counters.external_attack_gate_checks == 0


def test_hold_permission_does_not_advance_attack_burst():
    controller = attack_controller()
    controller.set_external_attack_permission(False)
    action = controller.act(frame(), telemetry(), raycast=in_range_log())
    assert action == 0
    assert controller._burst_steps == 0
    assert controller.counters.attack_steps == 0
    assert controller.counters.external_attack_gate_rejections == 1


def test_gate_is_not_consulted_without_a_teacher_attack_opportunity():
    controller = attack_controller()
    controller.set_external_attack_permission(False)
    action = controller.act(
        frame(), telemetry(), raycast=missing_raycast_target()
    )
    assert action == 0
    assert controller.state == TrunkContactState.CENTER_TRUNK
    assert controller.counters.prevented_unconfirmed_attacks == 1
    assert controller.counters.external_attack_gate_checks == 0
    assert controller.counters.external_attack_gate_rejections == 0


def test_three_hold_rejections_recenter_without_drop_recovery():
    controller = attack_controller()
    for _ in range(3):
        controller.set_external_attack_permission(False)
        assert controller.act(frame(), telemetry(), raycast=in_range_log()) == 0
    assert controller.state == TrunkContactState.CENTER_TRUNK
    assert controller._burst_steps == 0
    assert controller.counters.external_attack_gate_recenters == 1
    assert controller.counters.drop_recovery_attempts == 0


def test_allow_permission_continues_attack_and_resets_rejection_streak():
    controller = attack_controller()
    controller.set_external_attack_permission(False)
    controller.act(frame(), telemetry(), raycast=in_range_log())
    controller.set_external_attack_permission(True)
    assert controller.act(frame(), telemetry(), raycast=in_range_log()) == 7
    assert controller._burst_steps == 1
    assert controller._external_attack_rejection_streak == 0
    assert controller.counters.external_attack_gate_allows == 1


def test_autonomous_runner_injects_hold_before_teacher_action_and_clears_it():
    teacher = PermissionAwareTeacher()
    agent = FixedGateAgent(tiny_gate_policy(), HOLD)
    runner = NaturalAttackGateRunner(teacher, agent, "autonomous", 2)
    action, record = runner.act({"pov": frame()})
    assert action == 0
    assert record["gate_permission_applied"]
    assert teacher.permissions == [False, None]
    assert agent.calls == 1


def test_shadow_runner_never_changes_teacher_permission():
    teacher = PermissionAwareTeacher()
    agent = FixedGateAgent(tiny_gate_policy(), HOLD)
    runner = NaturalAttackGateRunner(teacher, agent, "shadow", 2)
    action, record = runner.act({"pov": frame()})
    assert action == 7
    assert not record["gate_permission_applied"]
    assert teacher.permissions == [None, None]
    assert record["gate_probability"] == 0.5
    assert record["confirmed_gate_decision"] == HOLD


def test_two_frame_confirmation_blocks_first_attack_then_allows_second():
    teacher = PermissionAwareTeacher()
    agent = FixedGateAgent(tiny_gate_policy(), ATTACK)
    runner = NaturalAttackGateRunner(
        teacher,
        agent,
        "autonomous",
        2,
        attack_confirmation_frames=2,
    )
    first_action, first = runner.act({"pov": frame()})
    second_action, second = runner.act({"pov": frame()})
    assert first_action == 0
    assert first["confirmed_gate_decision"] == HOLD
    assert second_action == 7
    assert second["confirmed_gate_decision"] == ATTACK
    agent.decision = HOLD
    third_action, third = runner.act({"pov": frame()})
    assert third_action == 0
    assert third["confirmed_gate_decision"] == HOLD


def test_non_gate_state_stays_scripted_and_skips_prediction():
    teacher = PermissionAwareTeacher("DROP_RECOVERY")
    agent = FixedGateAgent(tiny_gate_policy(), HOLD)
    runner = NaturalAttackGateRunner(teacher, agent, "autonomous", 2)
    action, record = runner.act({"pov": frame()})
    assert action == 7
    assert not record["gate_eligible"]
    assert agent.calls == 0


def test_attack_gate_checkpoint_preserves_threshold_and_previous_action_vocab(tmp_path):
    policy = tiny_gate_policy()
    policy.decision_threshold = 0.83
    policy.attack_confirmation_frames = 2
    path = tmp_path / "gate.npz"
    policy.save(str(path))
    loaded = NaturalAttackGatePolicy.load(str(path))
    assert loaded.decision_threshold == 0.83
    assert loaded.attack_confirmation_frames == 2
    pov = np.repeat(frame()[None, ...], 2, axis=0)
    # Previous action 11 is an environment action, not a binary gate label.
    assert loaded.predict(pov, 11) in (HOLD, ATTACK)


def test_binary_metrics_and_threshold_selection_prioritize_precision():
    probabilities = np.asarray([0.99, 0.90, 0.80, 0.70, 0.10, 0.05])
    labels = np.asarray([1, 1, 1, 0, 0, 0])
    metrics = binary_metrics(probabilities, labels, 0.8)
    assert metrics["attack_precision"] == 1.0
    assert metrics["attack_recall"] == 1.0
    threshold, selected, passed = select_conservative_threshold(
        probabilities, labels, minimum_precision=1.0, minimum_recall=1.0
    )
    assert passed
    assert threshold >= 0.70
    assert selected["attack_precision"] == 1.0


def test_hard_negative_repeat_uses_audit_only_for_training_indices():
    split = {
        "label": np.asarray([0, 0, 1, 0]),
        "audit_raycast_is_log": np.asarray([1, 1, 1, 0]),
        "audit_raycast_in_range": np.asarray([1, 0, 1, 1]),
    }
    indices, report = hard_negative_training_indices(split, repeat_factor=4)
    assert indices.tolist() == [0, 1, 2, 3, 0, 0, 0]
    assert report == {
        "repeat_factor": 4,
        "original_samples": 4,
        "hard_negative_samples": 1,
        "added_hard_negative_samples": 3,
        "training_samples_before_mirror": 7,
    }
