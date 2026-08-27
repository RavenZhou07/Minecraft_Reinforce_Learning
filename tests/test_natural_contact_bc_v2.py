from typing import Any, Dict

import numpy as np
import pytest

from mc_rl.natural_bc_v2_runner import HybridNaturalContactRunner
from mc_rl.natural_contact_bc import StudentContactAgent, StudentObservation
from mc_rl.natural_contact_bc_v2 import (
    LEARNABLE_CONTACT_STATES,
    MODEL_VERSION,
    V2_ACTION_CLASSES,
    NaturalContactBCV2Policy,
    hybrid_learning_mask,
    visual_student_eligible,
)
from scripts.train_natural_treechop_bc_v2 import (
    evaluate_predictions,
    load_hybrid_split,
    select_hybrid_samples,
)


RNG = np.random.RandomState(29)


def frame():
    return RNG.randint(0, 255, size=(8, 8, 3), dtype=np.uint8)


def observation(pov=None) -> Dict[str, Any]:
    return {"pov": frame() if pov is None else pov}


def tiny_v2_policy() -> NaturalContactBCV2Policy:
    policy = NaturalContactBCV2Policy(feature_size=4, frame_stack=2)
    pov = np.repeat(frame()[None, None, ...], 8, axis=0)
    pov = np.repeat(pov, 2, axis=1)
    features = policy.build_features(pov, [0] * len(pov), fit_normalization=True)
    policy.weights = np.zeros(
        (features.shape[1], len(V2_ACTION_CLASSES)), dtype=np.float32
    )
    policy.best_epoch = 0
    return policy


class FixedV2Agent(StudentContactAgent):
    def __init__(self, policy, action):
        super().__init__(policy)
        self.action = int(action)
        self.calls = 0

    def act(self, guarded: StudentObservation) -> int:
        self.observe_pov(guarded)
        self.calls += 1
        return self.action


class StatefulTeacher:
    def __init__(self, initial_state, decisions):
        self.contact_state = initial_state
        self.decisions = list(decisions)
        self.index = 0
        self.last_action_source = "contact"

    def act(self, obs):
        action, resulting_state, source = self.decisions[self.index]
        self.index += 1
        self.contact_state = resulting_state
        self.last_action_source = source
        return int(action)


def test_v2_boundary_has_only_pixel_grounded_states():
    assert LEARNABLE_CONTACT_STATES == {
        "CENTER_TRUNK",
        "ADJUST_PITCH",
        "ATTACK_TRUNK",
    }
    assert visual_student_eligible("CENTER_TRUNK", "CENTER_TRUNK", 10)
    assert visual_student_eligible("ADJUST_PITCH", "ATTACK_TRUNK", 7)
    assert not visual_student_eligible("COORDINATE_AIM", "ATTACK_TRUNK", 7)
    assert not visual_student_eligible("ATTACK_TRUNK", "DROP_RECOVERY", 1)
    assert not visual_student_eligible("EXACT_LOG_RESCAN", "EXACT_LOG_RESCAN", 10)


def test_hybrid_mask_requires_both_transition_states_and_supported_action():
    mask = hybrid_learning_mask(
        ["CENTER_TRUNK", "ATTACK_TRUNK", "COORDINATE_AIM", "ADJUST_PITCH"],
        ["CENTER_TRUNK", "DROP_RECOVERY", "ATTACK_TRUNK", "ADJUST_PITCH"],
        [10, 1, 7, 6],
    )
    assert mask.tolist() == [True, False, False, False]


def test_autonomous_student_executes_only_inside_visual_boundary():
    teacher = StatefulTeacher(
        "EXACT_LOG_RESCAN",
        [
            (10, "EXACT_LOG_RESCAN", "contact"),
            (7, "ATTACK_TRUNK", "contact"),
            (7, "ATTACK_TRUNK", "contact"),
            (1, "DROP_RECOVERY", "contact"),
        ],
    )
    agent = FixedV2Agent(tiny_v2_policy(), action=0)
    runner = HybridNaturalContactRunner(teacher, agent, "autonomous", 2)
    first, first_record = runner.act(observation())
    second, second_record = runner.act(observation())
    third, third_record = runner.act(observation())
    fourth, fourth_record = runner.act(observation())
    assert [first, second, third, fourth] == [10, 7, 0, 1]
    assert first_record["control_owner"] == "scripted_teacher"
    # EXACT_LOG_RESCAN -> ATTACK_TRUNK is a transition, so the first attack
    # also remains scripted. Only ATTACK_TRUNK -> ATTACK_TRUNK is learned.
    assert second_record["control_owner"] == "scripted_teacher"
    assert third_record["control_owner"] == "visual_student"
    assert fourth_record["control_owner"] == "scripted_teacher"
    assert agent.calls == 1
    assert runner.visual_student_steps == 1
    assert runner.scripted_contact_steps == 3
    # Script-owned frames were still added to the student's causal history.
    assert len(agent.history._episode_frames) == 4


def test_shadow_predicts_on_visual_step_but_executes_teacher():
    teacher = StatefulTeacher(
        "CENTER_TRUNK", [(10, "CENTER_TRUNK", "contact")]
    )
    agent = FixedV2Agent(tiny_v2_policy(), action=11)
    runner = HybridNaturalContactRunner(teacher, agent, "shadow", 2)
    executed, record = runner.act(observation())
    assert executed == 10
    assert record["student_action"] == 11
    assert record["control_owner"] == "visual_student_shadow"
    assert runner.student_actions_executed == 0


def test_scripted_states_do_not_request_a_student_prediction():
    teacher = StatefulTeacher(
        "REACQUIRE_SAME_TRUNK",
        [(11, "REACQUIRE_SAME_TRUNK", "contact")],
    )
    agent = FixedV2Agent(tiny_v2_policy(), action=10)
    runner = HybridNaturalContactRunner(teacher, agent, "autonomous", 2)
    executed, record = runner.act(observation())
    assert executed == 11
    assert record["student_action"] is None
    assert agent.calls == 0


def test_reduced_output_policy_still_accepts_any_previous_environment_action():
    policy = tiny_v2_policy()
    pov = np.repeat(frame()[None, ...], 2, axis=0)
    # Coarse right (4) is scripted and is not a v2 output class, but it remains
    # a valid previous action input to the visual student.
    action = policy.predict(pov, previous_action=4)
    assert action in V2_ACTION_CLASSES
    with pytest.raises(ValueError):
        policy._validate_labels([4])


def test_direction_metrics_do_not_credit_non_direction_predictions():
    labels = np.asarray([10, 11, 12, 13])
    predictions = np.asarray([7, 7, 7, 7])
    metrics = evaluate_predictions(predictions, labels)
    assert metrics["fine_yaw_direction_agreement"] == 0.0
    assert metrics["fine_pitch_direction_agreement"] == 0.0


def test_v2_checkpoint_roundtrip_preserves_reduced_classes(tmp_path):
    policy = tiny_v2_policy()
    path = tmp_path / "v2.npz"
    policy.save(str(path))
    loaded = NaturalContactBCV2Policy.load(str(path))
    assert loaded.model_version == MODEL_VERSION
    assert loaded.classes.tolist() == V2_ACTION_CLASSES.tolist()
    pov = np.repeat(frame()[None, ...], 2, axis=0)
    assert loaded.predict(pov, 9) in V2_ACTION_CLASSES


def test_v2_loader_rejects_legacy_dataset_without_explicit_diagnostic_flag(tmp_path):
    path = tmp_path / "legacy.npz"
    np.savez_compressed(
        str(path),
        pov=np.repeat(frame()[None, None, ...], 2, axis=0),
        action=np.asarray([7, 10]),
        previous_action=np.asarray([0, 7]),
        audit_contact_state=np.asarray(["ATTACK_TRUNK", "CENTER_TRUNK"]),
    )
    with pytest.raises(KeyError, match="lacks pre/post"):
        load_hybrid_split(path)
    split, formal = load_hybrid_split(path, allow_legacy_post_state_audit=True)
    assert not formal
    selected, report = select_hybrid_samples(split)
    assert selected["action"].tolist() == [7, 10]
    assert report["selected_visual_samples"] == 2


def test_sample_selection_excludes_failures_and_script_owned_states():
    split = {
        "pov": np.repeat(frame()[None, None, ...], 4, axis=0),
        "action": np.asarray([7, 10, 11, 7]),
        "previous_action": np.asarray([0, 7, 10, 11]),
        "decision_state": np.asarray(
            ["ATTACK_TRUNK", "CENTER_TRUNK", "EXACT_LOG_RESCAN", "ATTACK_TRUNK"]
        ),
        "resulting_state": np.asarray(
            ["ATTACK_TRUNK", "CENTER_TRUNK", "EXACT_LOG_RESCAN", "ATTACK_TRUNK"]
        ),
        "episode_success": np.asarray([1, 1, 1, 0]),
    }
    selected, report = select_hybrid_samples(split)
    assert selected["action"].tolist() == [7, 10]
    assert report == {
        "total_samples": 4,
        "excluded_failure_samples": 1,
        "successful_samples": 3,
        "scripted_or_unsupported_samples": 1,
        "selected_visual_samples": 2,
    }
